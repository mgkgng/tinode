"""Regression tests for the tinode nodes.

Every test here corresponds to something that actually broke: the paste-back
size mismatch, the crop round trip drifting, an index parser silently eating a
malformed token, a stale selection surviving an input change, the Python/JS
colour functions falling out of sync.

Run either way (pytest is optional):
	python tests/test_nodes.py            # standalone runner
	pytest tests/                         # if pytest is installed

Needs torch, so run it with ComfyUI's interpreter, e.g.
	ComfyUI/venv/bin/python tests/test_nodes.py
"""

from __future__ import annotations

import json
import math
import os
import sys
import tempfile
import time
from fractions import Fraction

import torch

# Import the pack as `tinode` regardless of where the tests are run from.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import tinode  # noqa: E402,F401  (registers every node)
from tinode.base import first  # noqa: E402
from tinode.nodes.image.batch_drop import parse_keep  # noqa: E402
from tinode.nodes.image.batch_pick import parse_pick  # noqa: E402
from tinode.nodes.image.bbox_crop import MaskBboxCrop  # noqa: E402
from tinode.nodes.image.crop_bbox_manual import BboxCropManual  # noqa: E402
from tinode.nodes.image.insert_video import InsertVideo  # noqa: E402
from tinode.nodes.image.trim_video import TrimVideo  # noqa: E402
from tinode.nodes.image.cut_video import CutVideo, cut_bounds  # noqa: E402
from tinode.nodes.image.paste_back import MaskCropPasteBack  # noqa: E402
from tinode.nodes.image.pick_segments import (  # noqa: E402
	PickSegments, color_for_id, prune_asset_cache,
)
from tinode.nodes.image.add_segments import AddSegments, _MANUAL_ID_BASE  # noqa: E402
from tinode.nodes.image.mask_to_segment import MaskToSegment, mask_to_segments  # noqa: E402
from tinode.nodes.image.delete_segments import DeleteSegments, parse_deleted_items  # noqa: E402
from tinode.nodes.data.json_to_item_list import (  # noqa: E402
	JsonToItemList, parse_items,
)
from tinode.nodes.data.json_path import (  # noqa: E402
	INDEX, KEY, JsonPath, parse_path, render_path,
)
from tinode.nodes.image.video_concat import (  # noqa: E402
	Combined, ConcatenatedVideo, VideoConcatenate, audio_sample_count, combine,
	fit_audio, fit_channels, flatten, is_video, retime_indices,
)
from tinode.nodes.image.load_videos import resolve_dir, resolve_video_files  # noqa: E402
from tinode.nodes.image.crop_apply import CropByInfo  # noqa: E402
from tinode.nodes.image.video_source_path import video_source_path  # noqa: E402
from tinode.nodes.image import _mask_store as _mstore  # noqa: E402
from tinode.nodes.image import _preview_store as _pstore  # noqa: E402
from tinode.nodes.image._video_io import colour_flags, encode as _vio_encode, ffmpeg_exe  # noqa: E402
from tinode.nodes.image.video_preview import VideoPreview  # noqa: E402
from tinode.nodes.image.image_preview import ImagePreview  # noqa: E402
from tinode.nodes.sampling.seed_range_noise import Noise_SeedRange, seeds_for  # noqa: E402
from tinode.nodes.sampling.sigma_segment import SigmaSegment, slice_sigmas  # noqa: E402
from tinode.nodes.sampling.candidate_select import CandidateSelect  # noqa: E402
from tinode.nodes.sampling.noise_rotate import NoiseRotate, rotate_residual  # noqa: E402
from tinode.nodes.sampling.step_stamp import (  # noqa: E402
	StampStep, ResumeStep, stamped_step,
)
from tinode.schema import validate_crop_xform, validate_segments  # noqa: E402


def _unwrap(r):
	"""Nodes return either a tuple or a {'ui','result'} envelope."""
	return r["result"] if isinstance(r, dict) else r


def _segment(sid, x0, y0, x1, y1):
	return {"id": sid, "bbox": [x0, y0, x1, y1], "conf": 0.9,
			"mask": torch.ones(y1 - y0, x1 - x0, dtype=torch.uint8)}


def _segments(n_frames=2, h=40, w=40):
	return {"num_frames": n_frames, "height": h, "width": w,
			"frames": [[_segment(3, 0, 0, 10, 10), _segment(7, 20, 20, 30, 30)]]
					+ [[] for _ in range(n_frames - 1)],
			"ids": [3, 7]}


# --------------------------------------------------------------- registry
def test_registry_ids_are_namespaced_and_unique():
	ids = list(tinode.NODE_CLASS_MAPPINGS)
	assert ids, "no nodes registered"
	assert all(i.startswith("TI_") for i in ids), [i for i in ids if not i.startswith("TI_")]
	assert len(ids) == len(set(ids))
	# every node exposes the ComfyUI contract
	for nid, cls in tinode.NODE_CLASS_MAPPINGS.items():
		assert hasattr(cls, "INPUT_TYPES") and hasattr(cls, "RETURN_TYPES"), nid
		assert hasattr(cls, cls.FUNCTION), f"{nid} has no {cls.FUNCTION}()"


# ------------------------------------------------------------- video concat
class _Comp:
	"""Stand-in for VideoComponents (comfy_api is not importable in tests)."""

	def __init__(self, images, frame_rate, audio=None):
		self.images = images
		self.frame_rate = frame_rate
		self.audio = audio


class _FakeVideo:
	"""Stand-in for a native VIDEO: duck-typed, like is_video() expects."""

	def __init__(self, n, fps=24, h=8, w=6, audio_sr=None, value=None, channels=1):
		base = torch.arange(n, dtype=torch.float32) if value is None else torch.full((n,), float(value))
		self.images = base.view(n, 1, 1, 1).expand(n, h, w, 3).contiguous()
		self.frame_rate = Fraction(fps)
		self.audio = None
		if audio_sr:
			samples = audio_sample_count(n, self.frame_rate, audio_sr)
			# a ramp, so a mis-ordered or mis-padded join is visible
			wave = torch.linspace(0.1, 0.9, samples).repeat(channels, 1)
			self.audio = {"waveform": wave.unsqueeze(0), "sample_rate": audio_sr}

	def get_components(self):
		return _Comp(self.images, self.frame_rate, self.audio)

	def save_to(self, *a, **k):
		raise AssertionError("save_to should not be called in these tests")

	def get_frame_rate(self):
		return self.frame_rate

	def get_dimensions(self):
		return self.images.shape[2], self.images.shape[1]

	def get_bit_depth(self):
		return 8


def test_video_concat_passthrough_without_accumulator():
	b = _FakeVideo(4)
	# unconnected video_a
	assert _unwrap(VideoConcatenate().execute(b))[0] is b
	# ...and the Inspire loop's seed: ForeachListBegin hands initial_input
	# through as the first intermediate_output, and it is NOT a video.
	for seed in (None, "step one", 0, ["a"], {"a": 1}):
		assert _unwrap(VideoConcatenate().execute(b, video_a=seed))[0] is b
	# video_b must be real, though — that one is a wiring mistake
	try:
		VideoConcatenate().execute("not a video")
	except ValueError:
		pass
	else:
		raise AssertionError("accepted a non-VIDEO video_b")
	assert is_video(b) and not is_video("x") and not is_video(None)


def test_video_concat_is_a_hard_cut_in_order():
	a, b = _FakeVideo(3, value=1), _FakeVideo(2, value=2)
	out = combine([a.get_components(), b.get_components()])
	assert out.images.shape[0] == 5                       # nothing lost, nothing added
	assert out.frame_rate == Fraction(24)
	# chronological, and every output frame is an untouched input frame:
	# no blended/duplicated transition frame at the boundary.
	assert [float(f[0, 0, 0]) for f in out.images] == [1, 1, 1, 2, 2]


def test_video_concat_retimes_to_the_first_clips_rate():
	# 24 fps accumulator, 12 fps addition: the 12 fps clip must keep its
	# duration (2s), so it doubles to 48 frames by repeating, never blending.
	assert retime_indices(10, Fraction(24), Fraction(24)) is None
	idx = retime_indices(24, Fraction(12), Fraction(24))
	assert len(idx) == 48
	assert idx == sorted(idx) and set(idx) == set(range(24))   # only repeats
	# and dropping frames the other way
	assert len(retime_indices(24, Fraction(24), Fraction(12))) == 12

	a, b = _FakeVideo(24, fps=24), _FakeVideo(24, fps=12)
	out = combine([a.get_components(), b.get_components()])
	assert out.images.shape[0] == 24 + 48
	assert out.frame_rate == Fraction(24)
	# duration preserved: 1s + 2s
	assert abs(out.images.shape[0] / float(out.frame_rate) - 3.0) < 1e-9
	# every frame of the retimed part is still one of its source frames
	tail = {float(f[0, 0, 0]) for f in out.images[24:]}
	assert tail == set(range(24))


def test_video_concat_keeps_audio_in_sync():
	sr = 100
	a = _FakeVideo(10, fps=10, audio_sr=sr)      # 1.0 s
	b = _FakeVideo(5, fps=10, audio_sr=sr)       # 0.5 s
	out = combine([a.get_components(), b.get_components()])
	assert out.images.shape[0] == 15
	# audio covers exactly the video duration — no drift
	assert out.audio["waveform"].shape[-1] == audio_sample_count(15, Fraction(10), sr) == 150
	assert out.audio["sample_rate"] == sr


def test_video_concat_fills_silence_for_a_clip_without_audio():
	sr = 100
	loud = _FakeVideo(10, fps=10, audio_sr=sr)
	mute = _FakeVideo(10, fps=10)
	out = combine([loud.get_components(), mute.get_components()])
	w = out.audio["waveform"][0]
	assert w.shape[-1] == 200                          # video timing unchanged
	assert w[..., 100:].abs().max() == 0               # second half is silence
	assert w[..., :100].abs().max() > 0

	# and the other way round: silence goes FIRST, audio stays with its clip
	out = combine([mute.get_components(), loud.get_components()])
	w = out.audio["waveform"][0]
	assert w.shape[-1] == 200
	assert w[..., :100].abs().max() == 0
	assert w[..., 100:].abs().max() > 0


def test_video_concat_audio_is_fitted_to_its_own_clip():
	# audio that runs short is padded and audio that overruns is trimmed, so a
	# bad clip cannot desync everything after it
	assert fit_audio(torch.ones(1, 30), 50, 1).shape == (1, 50)
	assert float(fit_audio(torch.ones(1, 30), 50, 1)[0, 40]) == 0.0
	assert fit_audio(torch.ones(1, 80), 50, 1).shape == (1, 50)
	# mono upmixes to stereo by duplication rather than losing a channel
	up = fit_audio(torch.ones(1, 10), 10, 2)
	assert up.shape == (2, 10) and float(up[1, 0]) == 1.0
	assert fit_audio(torch.ones(2, 10), 10, 1).shape == (1, 10)
	# a mismatched sample rate is resampled, keeping the clip's duration
	sr_a, sr_b = 200, 100
	a = _FakeVideo(10, fps=10, audio_sr=sr_a)
	b = _FakeVideo(10, fps=10, audio_sr=sr_b)
	out = combine([a.get_components(), b.get_components()])
	assert out.audio["sample_rate"] == sr_a
	assert out.audio["waveform"].shape[-1] == audio_sample_count(20, Fraction(10), sr_a) == 400


def test_video_concat_rejects_a_resolution_mismatch():
	# silently rescaling would be worse than refusing: the whole clip would
	# inherit one step's wrong geometry
	a, b = _FakeVideo(2, h=8, w=6), _FakeVideo(2, h=8, w=7)
	try:
		combine([a.get_components(), b.get_components()])
	except ValueError as exc:
		assert "7x8" in str(exc) and "6x8" in str(exc), exc
	else:
		raise AssertionError("accepted mismatched resolutions")


def test_video_concat_normalizes_channels_and_dtype():
	rgba = _FakeVideo(2)
	rgba.images = torch.ones(2, 8, 6, 4, dtype=torch.float64)
	out = combine([_FakeVideo(2).get_components(), rgba.get_components()])
	assert out.images.shape == (4, 8, 6, 3) and out.images.dtype == torch.float32
	assert fit_channels(torch.ones(1, 2, 2, 1)).shape[-1] == 3


def test_video_concat_accumulates_flat_across_a_loop():
	# The loop shape: video_a = previous result, video_b = this step's clip.
	# Parts must stay a flat list — nesting would recurse once per step.
	acc = None
	for step in range(5):
		clip = _FakeVideo(2, value=step)
		acc = _unwrap(VideoConcatenate().execute(clip, video_a=acc))[0]
	assert isinstance(acc, ConcatenatedVideo)
	assert len(acc.parts) == 5
	assert all(not isinstance(p, ConcatenatedVideo) for p in acc.parts)
	assert flatten(acc) == acc.parts

	# ...and it is lazy: nothing was decoded or copied while looping
	assert acc._components is None
	c = acc._combined()
	assert isinstance(c, Combined) and c.images.shape[0] == 10
	assert [float(f[0, 0, 0]) for f in c.images] == [0, 0, 1, 1, 2, 2, 3, 3, 4, 4]
	assert acc._components is c                        # materialized once, cached
	# cheap metadata needs no decode
	assert acc.get_dimensions() == (6, 8) and acc.get_frame_rate() == Fraction(24)


# --------------------------------------------------------- json → item list
def test_json_to_item_list_splits_objects():
	out = _unwrap(JsonToItemList().execute('[{"a": 1}, {"a": 2}, {"a": 3}]'))
	item_list, items, count = out
	assert count == 3
	assert items == ['{"a":1}', '{"a":2}', '{"a":3}']
	# ITEM_LIST is what Inspire's ForeachListBegin consumes: a plain list it
	# indexes and slices. Must be equal in content but a distinct object.
	assert item_list == items and item_list is not items
	# and every item is parseable back into the original object
	assert [json.loads(i) for i in items] == [{"a": 1}, {"a": 2}, {"a": 3}]


def test_json_to_item_list_string_and_scalar_elements():
	# strings pass through unquoted (a prompt list is the whole point)...
	_, items, _ = _unwrap(JsonToItemList().execute('["a cat", "un chien"]'))
	assert items == ["a cat", "un chien"]
	# ...everything else is compact JSON, non-ASCII kept verbatim
	_, items, _ = _unwrap(JsonToItemList().execute('[1, true, null, [2], "é"]'))
	assert items == ["1", "true", "null", "[2]", "é"]


def test_json_to_item_list_accepts_object_and_jsonl():
	assert parse_items('{"a": 1}') == [{"a": 1}]              # lone object
	assert parse_items('{"a": 1}\n\n{"a": 2}\n') == [{"a": 1}, {"a": 2}]  # jsonl
	# valid JSON is never reinterpreted as JSONL, even spread over lines
	assert parse_items('[\n{"a": 1},\n{"a": 2}\n]') == [{"a": 1}, {"a": 2}]


def test_json_to_item_list_rejects_junk_and_empty():
	# An empty list would make ForeachList raise IndexError and would make the
	# ComfyUI-list output silently skip the branch — both must fail loudly here.
	for bad in ("", "   ", "[]", "not json", "[{,}]", '"a string"', "5"):
		try:
			JsonToItemList().execute(bad)
		except ValueError:
			continue
		raise AssertionError(f"accepted junk input: {bad!r}")


def test_json_to_item_list_output_contract():
	# item_list is a single value, items is a ComfyUI list — the flags must say so.
	assert JsonToItemList.RETURN_TYPES == ("ITEM_LIST", "STRING", "INT")
	assert JsonToItemList.OUTPUT_IS_LIST == (False, True, False)


# ------------------------------------------------------------------ json path
_DOC = json.dumps({
	"steps": [{"prompt": "a cat", "cfg": 7.5}, {"prompt": "un chien", "cfg": 3}],
	"nested": {"grid": [[1, 2], [3, 4]]},
	"meta": {"key.with.dots": "reached", "flag": True, "none": None},
})


def test_json_path_parses_keys_and_indices():
	assert parse_path("") == []
	assert parse_path("a.b.c") == [(KEY, "a"), (KEY, "b"), (KEY, "c")]
	assert parse_path("steps[0]") == [(KEY, "steps"), (INDEX, 0)]
	assert parse_path("a[0][2]") == [(KEY, "a"), (INDEX, 0), (INDEX, 2)]
	assert parse_path("[0].name") == [(INDEX, 0), (KEY, "name")]
	assert parse_path("-1") == [(INDEX, -1)]            # bare negative index
	assert parse_path('["a.b"]') == [(KEY, "a.b")]      # quoted key keeps its dots
	assert parse_path(" steps [ 0 ] ") == [(KEY, "steps"), (INDEX, 0)]
	# round trip back to text, used in the error messages
	for p in ("a.b.c", "steps[0]", "a[0][2]", "[0].name"):
		assert render_path(parse_path(p)) == p


def test_json_path_rejects_malformed_paths():
	for bad in ("a..b", ".a", "a.", "a[", "a[]", "a[x]", "[0]b", "a[1.5]", "."):
		try:
			parse_path(bad)
		except ValueError:
			continue
		raise AssertionError(f"accepted malformed path: {bad!r}")


def test_json_path_extracts_values():
	def v(path, **kw):
		return _unwrap(JsonPath().execute(_DOC, path, **kw))

	# the case Simple JSON Parser is built for
	assert v("steps[0].prompt") == ("a cat", -1, "string")
	# ...and the cases it cannot express
	assert v("nested.grid[1][0]") == ("3", -1, "number")       # chained indices
	assert v("steps[-1].prompt") == ("un chien", -1, "string")  # negative index
	assert v('meta["key.with.dots"]') == ("reached", -1, "string")
	# a leading index, against an array document
	assert _unwrap(JsonPath().execute('[{"a": 1}, 2]', "[0].a")) == ("1", -1, "number")
	assert _unwrap(JsonPath().execute('[{"a": 1}, 2]', "-1")) == ("2", -1, "number")

	# counts, and JSON out for containers
	assert v("steps")[1] == 2 and v("steps")[2] == "array"
	assert v("meta")[1] == 3 and v("meta")[2] == "object"
	assert v("")[2] == "object"                                # empty = whole doc
	# JSON literals, not Python repr — str(None) would be "None"
	assert v("meta.flag")[0] == "true"
	assert v("meta.none") == ("null", -1, "null")
	# non-ASCII survives, and the value re-parses into the original object
	assert json.loads(v("steps[0]")[0]) == {"prompt": "a cat", "cfg": 7.5}


def test_json_path_strict_and_default():
	# strict (default): a miss is a loud error, naming what was available
	for miss in ("steps[9]", "nope", "steps[0].nope", "meta.flag.deeper", "steps.0x"):
		try:
			JsonPath().execute(_DOC, miss)
		except ValueError:
			continue
		raise AssertionError(f"missing path did not raise: {miss!r}")
	try:
		JsonPath().execute(_DOC, "nope")
	except ValueError as exc:
		assert "available" in str(exc) and "steps" in str(exc), exc

	# strict off: fall back instead of raising
	assert _unwrap(JsonPath().execute(_DOC, "steps[9]", strict=False,
									default="fallback")) == ("fallback", -1, "missing")
	# but a MALFORMED path still raises even with strict off — a typo must never
	# quietly return the default.
	try:
		JsonPath().execute(_DOC, "a..b", strict=False, default="x")
	except ValueError:
		pass
	else:
		raise AssertionError("malformed path was swallowed by strict=False")

	# bad document / empty text always raise
	for bad_doc in ("", "   ", "not json"):
		try:
			JsonPath().execute(bad_doc, "a", strict=False)
		except ValueError:
			continue
		raise AssertionError(f"accepted bad document: {bad_doc!r}")


def test_json_path_indexing_a_string_is_a_miss_not_a_character():
	# "a cat"[0] == "a" would hide a wrong path; it must be reported instead.
	assert _unwrap(JsonPath().execute(_DOC, "steps[0].prompt[0]", strict=False,
									default="-")) == ("-", -1, "missing")


def test_data_nodes_are_cacheable():
	# The whole point of these two vs Simple JSON Parser: no IS_CHANGED, so
	# ComfyUI's input-signature cache key works and the node (plus everything
	# downstream of it) is not re-executed on every queue.
	for cls in (JsonPath, JsonToItemList):
		assert not hasattr(cls, "IS_CHANGED"), f"{cls.__name__} defeats the cache"
		assert not hasattr(cls, "fingerprint_inputs"), cls.__name__
		assert not getattr(cls, "NOT_IDEMPOTENT", False), cls.__name__
		# same inputs -> same outputs, which is what makes caching correct
		a = _unwrap(cls().execute(_DOC, "steps") if cls is JsonPath
					else cls().execute('[{"a": 1}]'))
		b = _unwrap(cls().execute(_DOC, "steps") if cls is JsonPath
					else cls().execute('[{"a": 1}]'))
		assert a == b


def test_json_path_feeds_json_to_item_list():
	# The intended chain: pull the array out, then split it into items.
	value, count, kind = _unwrap(JsonPath().execute(_DOC, "steps"))
	assert kind == "array" and count == 2
	item_list, items, n = _unwrap(JsonToItemList().execute(value))
	assert n == 2
	assert [json.loads(i)["prompt"] for i in items] == ["a cat", "un chien"]
	assert item_list == items


# ------------------------------------------------------------ index parsers
def test_index_parsers_reject_malformed_input():
	# A malformed token must no-op (None), never silently drop the wrong frames.
	for bad in ("1.5", "-2", "a", "1,,2", "1, x", "+3"):
		assert parse_keep(bad, 5, True) is None, bad
		assert parse_pick(bad, 5, True) is None, bad


def test_index_parsers_semantics():
	assert parse_pick("2,4", 5, True) == [1, 3]          # 1-based
	assert parse_pick("2,4", 5, False) == [2, 4]         # 0-based
	assert parse_pick("3,1", 5, True) == [2, 0]          # order preserved (reorder)
	assert parse_pick("1,1", 5, True) == [0, 0]          # duplicates allowed
	assert parse_pick("99", 5, True) is None             # all out of range -> no-op
	assert parse_keep("1", 3, True) == [1, 2]            # drop keeps the rest
	# Deliberate asymmetry on an empty spec: picking nothing is a no-op, while
	# dropping nothing keeps every frame.
	assert parse_pick("", 5, True) is None
	assert parse_keep("", 3, True) == [0, 1, 2]


def test_first_unwraps_input_is_list():
	assert first([8]) == 8
	assert first(8) == 8
	assert first([], "fallback") == "fallback"


# ------------------------------------------------------- crop <-> paste back
def test_crop_paste_roundtrip_is_bit_exact():
	"""The whole point of the crop/paste pair: no pixel may drift."""
	torch.manual_seed(0)
	img = torch.rand(1, 200, 300, 3)
	mask = torch.zeros(1, 200, 300)
	mask[0, 40:160, 60:240] = 1.0

	crops, _, info = MaskBboxCrop().execute(
		[img], [mask], context_padding=[0], divisible_by=[16],
		shared_bbox=[True], smoothing=[1], threshold=[0.5])
	validate_crop_xform(info)
	it = info["items"][0]
	# no rescale -> the paste must copy pixels through untouched
	assert (it["oy"], it["ox"]) == (0, 0) and (it["nh"], it["nw"]) == (it["h"], it["w"])

	(out,) = MaskCropPasteBack().execute([img], [crops], [info], masks=None, feather=0)
	y0, x0, h, w = it["y0"], it["x0"], it["h"], it["w"]
	assert torch.equal(out[0, y0:y0 + h, x0:x0 + w, :], img[0, y0:y0 + h, x0:x0 + w, :])
	# and nothing outside the box was touched either
	untouched = torch.ones(1, 200, 300, dtype=torch.bool)
	untouched[0, y0:y0 + h, x0:x0 + w] = False
	sel = untouched.unsqueeze(-1).expand_as(img)
	assert torch.equal(out[sel], img[sel])


def test_paste_feather_does_not_fade_at_image_edge():
	"""A crop ending at the canvas edge has no outside seam to feather.

	Zero-padding avg_pool used to make the final rows/columns translucent, which
	leaked the original frame through the processed crop as visible tail lines.
	"""
	src = torch.zeros(1, 20, 30, 3)
	crops = torch.ones(1, 15, 20, 3)
	item = {
		"y0": 5, "x0": 5, "h": 15, "w": 20,
		"oy": 0, "ox": 0, "nh": 15, "nw": 20,
	}
	info = {"H": 20, "W": 30, "C": 3, "items": [item]}

	(out,) = MaskCropPasteBack().execute(
		[src], [crops], [info], masks=None, feather=[4])

	# Bottom touches the image edge and must remain fully pasted. The crop's
	# internal left/right edges should still be feathered.
	assert torch.equal(out[0, -1, 9:21], torch.ones(12, 3))
	assert torch.all(out[0, 10, 5] < out[0, 10, 10])


def test_crop_respects_divisible_by():
	img = torch.rand(1, 200, 300, 3)
	mask = torch.zeros(1, 200, 300)
	mask[0, 30:97, 40:131] = 1.0
	_, _, info = MaskBboxCrop().execute(
		[img], [mask], context_padding=[0], divisible_by=[16],
		shared_bbox=[True], smoothing=[1], threshold=[0.5])
	it = info["items"][0]
	assert it["h"] % 16 == 0 and it["w"] % 16 == 0, it


def test_manual_crop_is_never_empty():
	"""A box saved against a taller frame must not slice a zero-sized crop.

	divisible_by used to round UP past the space left, so a y beyond the frame
	produced an [N,0,W,C] tensor that broke everything downstream silently.
	"""
	node = BboxCropManual()
	# y is past the bottom of a 1080-tall frame
	crop, info = _unwrap(node.execute(torch.rand(3, 1080, 1920, 3),
									  x=437, y=1309, width=1176, height=673, divisible_by=32))
	assert crop.shape[1] > 0 and crop.shape[2] > 0, crop.shape
	assert info["items"][0]["h"] == crop.shape[1]
	# a normal box still honours divisible_by
	crop, _ = _unwrap(node.execute(torch.rand(2, 1080, 1920, 3),
								   x=100, y=100, width=640, height=384, divisible_by=32))
	assert crop.shape[1] % 32 == 0 and crop.shape[2] % 32 == 0
	# a box smaller than one multiple keeps what it has rather than nothing
	crop, _ = _unwrap(node.execute(torch.rand(1, 64, 64, 3),
								   x=0, y=0, width=10, height=10, divisible_by=32))
	assert crop.shape[1:3] == (10, 10), crop.shape


def test_pad_image_border_and_mask():
	from tinode.nodes.image.pad_image import PadImage, _hex_to_rgb
	# 'bad' is valid hex; only genuinely non-hex input falls back to black
	assert _hex_to_rgb("bad") == (0xbb / 255, 0xaa / 255, 0xdd / 255)
	assert _hex_to_rgb("#fff") == (1, 1, 1)
	for junk in ("red", "xyz", "#12g", "", "#12345"):
		assert _hex_to_rgb(junk) == (0.0, 0.0, 0.0), junk

	img = torch.rand(2, 10, 8, 3)
	out, mask, t, b, l, r = PadImage().execute(
		img, top=3, bottom=1, left=2, right=4, color="#ff0000")
	assert tuple(out.shape) == (2, 14, 14, 3) and (t, b, l, r) == (3, 1, 2, 4)
	assert torch.equal(out[:, 3:13, 2:10, :], img)               # original untouched
	assert torch.allclose(out[0, 0, 0, :], torch.tensor([1.0, 0.0, 0.0]))
	assert mask[0, 0, 0] == 1.0 and mask[0, 3, 2] == 0.0         # border=1, original=0
	assert int(mask[0].sum()) == 14 * 14 - 10 * 8
	# invert flips which region is marked
	_, minv, *_ = PadImage().execute(img, top=3, left=2, invert_mask=True)
	assert minv[0, 0, 0] == 0.0 and minv[0, 3, 2] == 1.0
	# zero pad is a passthrough with an empty mask
	out0, mask0, *_ = PadImage().execute(img)
	assert torch.equal(out0, img) and int(mask0.sum()) == 0


def test_paste_back_rejects_a_wrong_sized_source():
	"""The classic mis-wire: the crop node's OUTPUT fed back in as the source."""
	info = {"H": 512, "W": 512, "C": 3, "items": [
		{"y0": 0, "x0": 0, "h": 512, "w": 512, "oy": 0, "ox": 0, "nh": 512, "nw": 512}]}
	try:
		MaskCropPasteBack().execute([torch.zeros(1, 328, 512, 3)],
									[torch.zeros(1, 512, 512, 3)], [info])
	except RuntimeError as exc:
		assert "328" in str(exc) and "512" in str(exc)
	else:
		raise AssertionError("expected a readable RuntimeError, not a silent paste")


# ------------------------------------------------------------------ schema
def test_schema_validators_reject_junk():
	for bad in ({}, {"nope": 1}, []):
		try:
			validate_segments(bad)
		except RuntimeError:
			pass
		else:
			raise AssertionError(f"validate_segments accepted {bad!r}")
	try:
		validate_crop_xform({"H": 1, "W": 1, "items": [{"y0": 0}]})
	except RuntimeError as exc:
		assert "missing" in str(exc)
	else:
		raise AssertionError("validate_crop_xform accepted an incomplete item")


# --------------------------------------------------------- segment pipeline
def test_half_rate_stride_stays_on_the_global_grid():
	# The core invariant: every chunk keeps SOURCE-global indices % n == 0, so
	# fills from chunks starting on odd frames still land on the frames the
	# composite decodes.
	assert _mstore.stride_indices(0, 10, 2) == [0, 2, 4, 6, 8]
	assert _mstore.stride_indices(5, 10, 2) == [1, 3, 5, 7, 9]     # globals 6,8,10,12,14
	assert _mstore.stride_indices(5, 10, 1) == list(range(10))     # nth=1 = everything
	# consecutive chunks tile the kept timeline with no gap and no overlap
	a, b = _mstore.stride_indices(0, 7, 2), _mstore.stride_indices(7, 8, 2)
	kept = [0 + i for i in a] + [7 + i for i in b]
	assert kept == [0, 2, 4, 6, 8, 10, 12, 14]
	# the kept-timeline start used by Load Clip Fills: ceil(start / n)
	for s, n_, want in ((0, 2, 0), (5, 2, 3), (7, 2, 4), (8, 2, 4)):
		assert (s + (-s % n_)) // n_ == want, (s, n_)


def test_half_rate_loaders_and_frame_stride():
	import numpy as np
	from PIL import Image
	from tinode.nodes.image.frame_stride import FrameStride

	with tempfile.TemporaryDirectory() as d:
		# a 10-frame mask sequence where frame i is filled with i/10
		for i in range(10):
			Image.fromarray(np.full((4, 6), i * 10, dtype=np.uint8), mode="L").save(
				os.path.join(d, _mstore.MASK_PATTERN % i))
		idx = _mstore.stride_indices(5, 10, 2)
		m = _mstore.load_mask_sequence(d, _mstore.MASK_PATTERN, 10, indices=idx)
		assert m.shape[0] == 5
		assert abs(m[0].max().item() - 10 / 255) < 1e-6     # frame 1, not frame 0

	# Frame Stride: kept frames are bit-exact, fps divides, mask in lockstep
	img = torch.rand(10, 4, 6, 3)
	mask = torch.rand(10, 4, 6)
	out, mo, n, fps = FrameStride().execute(img, every_nth=2, mask=mask, fps=50.0)
	assert n == 5 and fps == 25.0
	assert torch.equal(out, img[0::2]) and torch.equal(mo, mask[0::2])
	out, _m, n, fps = FrameStride().execute(img, every_nth=1, fps=50.0)
	assert n == 10 and fps == 50.0 and torch.equal(out, img)


def test_edit_segments_paints_and_sweeps():
	from tinode.nodes.image.edit_segments import (
		EditSegments, parse_strokes, rasterize_stroke, painted_segments,
	)
	# a dab becomes a disc, a drag becomes a capsule; both stay inside the frame
	dab = rasterize_stroke({"r": 3, "pts": [[20, 20]]}, 50, 50)
	assert dab["bbox"] == [16, 16, 25, 25]
	assert 20 <= int(dab["mask"].sum()) <= 40            # ~pi*r^2 = 28
	drag = rasterize_stroke({"r": 3, "pts": [[10, 20], [30, 20]]}, 50, 50)
	assert int(drag["mask"].sum()) > int(dab["mask"].sum())
	assert rasterize_stroke({"r": 3, "pts": []}, 50, 50) is None
	# clamped at the edge rather than running negative
	edge = rasterize_stroke({"r": 8, "pts": [[1, 1]]}, 50, 50)
	assert edge["bbox"][0] == 0 and edge["bbox"][1] == 0

	# strokes are per-frame and signature-guarded
	raw = json.dumps({"sig": "a", "strokes": [{"id": 2000001, "frame": 1, "r": 2,
											   "pts": [[5, 5]]}]})
	assert len(parse_strokes(raw, "a")) == 1
	assert parse_strokes(raw, "b") == []
	assert painted_segments(parse_strokes(raw, "a"), 0, 50, 50) == []      # other frame
	assert len(painted_segments(parse_strokes(raw, "a"), 1, 50, 50)) == 1

	# end to end: erase one input segment, paint a new one
	segs = {"num_frames": 1, "height": 40, "width": 40, "ids": [7],
			"frames": [[{"id": 7, "conf": 1.0, "bbox": [0, 0, 10, 10],
						 "mask": torch.ones(10, 10, dtype=torch.uint8)}]]}
	img = torch.zeros(1, 40, 40, 3)
	mask, _im, filt, _bb = _unwrap(EditSegments().execute(
		img, segs,
		painted=json.dumps({"strokes": [{"id": 2000001, "frame": 0, "r": 4,
										 "pts": [[30, 30]]}]})))
	assert int(mask[0, 0:10, 0:10].sum()) == 100          # the input segment survives
	assert int(mask[0, 26:35, 26:35].sum()) > 0           # the painted blob is there
	assert 2000001 in filt["ids"] and 7 in filt["ids"]
	# now delete the input segment too — only the painted one remains
	mask2, _i2, filt2, _b2 = _unwrap(EditSegments().execute(
		img, segs,
		deleted_items=json.dumps({"items": [{"frame": 0, "index": 0}]}),
		painted=json.dumps({"strokes": [{"id": 2000001, "frame": 0, "r": 4,
										 "pts": [[30, 30]]}]})))
	assert int(mask2[0, 0:10, 0:10].sum()) == 0 and filt2["ids"] == [2000001]


def test_edit_segments_erases_pixels_from_existing_masks():
	from tinode.nodes.image.edit_segments import (
		EditSegments, apply_strokes, erase_segment, rasterize_stroke, tighten,
	)
	# a 10x10 block at (10,10); rub a disc out of its bottom-right corner
	seg = {"id": 7, "conf": 1.0, "bbox": [10, 10, 20, 20],
		   "mask": torch.ones(10, 10, dtype=torch.uint8)}
	shape = rasterize_stroke({"r": 3, "pts": [[19, 19]]}, 40, 40)
	cut = erase_segment(dict(seg), shape)
	assert int(cut["mask"].sum()) < 100                     # pixels are gone
	assert cut["bbox"] == [10, 10, 20, 20]                  # nothing emptied a whole edge
	assert cut["mask"][9, 9] == 0 and cut["mask"][0, 0] == 1

	# a stroke that covers the segment entirely drops it
	assert erase_segment(dict(seg), rasterize_stroke(
		{"r": 12, "pts": [[15, 15]]}, 40, 40)) is None
	# ...and one that misses leaves it untouched
	assert erase_segment(seg, rasterize_stroke({"r": 2, "pts": [[35, 35]]}, 40, 40)) is seg

	# tighten pulls the bbox in once a whole band is erased
	band = {"id": 7, "conf": 1.0, "bbox": [10, 10, 20, 20],
			"mask": torch.ones(10, 10, dtype=torch.uint8)}
	band["mask"][0:4, :] = 0
	assert tighten(band)["bbox"] == [10, 14, 20, 20]

	# strokes replay IN ORDER: an erase then a draw paints back over the hole
	erase_st = {"id": 2000001, "mode": "erase", "frame": 0, "r": 12, "pts": [[15, 15]]}
	draw_st = {"id": 2000002, "mode": "draw", "frame": 0, "r": 3, "pts": [[15, 15]]}
	assert apply_strokes([dict(seg)], [erase_st], 0, 40, 40) == []
	after = apply_strokes([dict(seg)], [erase_st, draw_st], 0, 40, 40)
	assert [s["id"] for s in after] == [2000002]

	# end to end through the node: SAM3's own segment loses the erased pixels
	segs = {"num_frames": 1, "height": 40, "width": 40, "ids": [7],
			"frames": [[{"id": 7, "conf": 1.0, "bbox": [10, 10, 20, 20],
						 "mask": torch.ones(10, 10, dtype=torch.uint8)}]]}
	img = torch.zeros(1, 40, 40, 3)
	mask, _im, filt, _bb = _unwrap(EditSegments().execute(
		img, segs,
		painted=json.dumps({"strokes": [
			{"id": 2000001, "mode": "erase", "frame": 0, "r": 3, "pts": [[19, 19]]}]})))
	assert 0 < int(mask[0].sum()) < 100 and filt["ids"] == [7]
	assert float(mask[0, 19, 19]) == 0.0 and float(mask[0, 10, 10]) == 1.0
	# an erase drawn on another frame leaves this one alone
	mask_other, _i, _f, _b = _unwrap(EditSegments().execute(
		img, segs,
		painted=json.dumps({"strokes": [
			{"id": 2000001, "mode": "erase", "frame": 3, "r": 3, "pts": [[19, 19]]}]})))
	assert int(mask_other[0].sum()) == 100


def test_mask_frames_splits_a_batch_into_one_job_per_frame():
	from tinode.nodes.mask_motion.mask_frames import (
		MaskFrames, frame_indices, seed_for,
	)
	assert frame_indices(10) == list(range(10))
	assert frame_indices(10, stride=3) == [0, 3, 6, 9]
	assert frame_indices(10, start=4, limit=2) == [4, 5]
	assert frame_indices(10, start=4, stride=2) == [4, 6, 8]
	assert frame_indices(0) == []
	# a start past the end clamps to the last frame rather than selecting nothing
	assert frame_indices(3, start=99) == [2]

	# seeds are spread, not adjacent — twelve near-identical images otherwise
	seeds = [seed_for(7, i) for i in range(8)]
	assert len(set(seeds)) == 8
	assert min(abs(a - b) for a, b in zip(seeds, seeds[1:])) > 10 ** 8
	assert seed_for(7, 3) == seed_for(7, 3)          # and reproducible
	assert seed_for(8, 3) != seed_for(7, 3)
	assert all(0 <= s <= 0xFFFFFFFFFFFFFFFF for s in seeds)

	mask = torch.zeros(6, 8, 12)
	for i in range(6):
		mask[i, i, i] = 1.0                          # a marker per frame
	(masks, idx, sds, count, preview,
	 crops, xs, ys, ws, hs) = MaskFrames().execute(mask, seed=1, stride=2)
	assert idx == [0, 2, 4] and count == 3 and len(masks) == 3 and len(sds) == 3
	assert all(tuple(m.shape) == (1, 8, 12) for m in masks)
	# each returned frame really is ITS frame, in order
	for k, i in enumerate(idx):
		assert float(masks[k][0, i, i]) == 1.0
	assert tuple(preview.shape) == (3, 8, 12, 3)
	# each frame's box wraps ITS marker, and the crop is that box of the mask
	for k, i in enumerate(idx):
		assert xs[k] <= i < xs[k] + ws[k] and ys[k] <= i < ys[k] + hs[k]
		assert tuple(crops[k].shape) == (1, hs[k], ws[k])
		assert float(crops[k].sum()) == 1.0        # the one marker pixel, nothing else
	# a bare [H,W] mask is accepted as one frame
	one, _i, _s, c1, _p, *_g = MaskFrames().execute(torch.zeros(4, 5))
	assert c1 == 1 and tuple(one[0].shape) == (1, 4, 5)
	# a start past the end gives the last frame rather than failing the run
	assert MaskFrames().execute(mask, start=99)[3] == 1


def test_mask_bbox_boxes_the_mask_so_the_object_can_fill_it():
	from tinode.nodes.mask_motion.mask_frames import mask_bbox
	m = torch.zeros(100, 200)
	m[40:60, 90:110] = 1.0
	assert mask_bbox(m, square=False) == (90, 40, 20, 20)
	assert mask_bbox(m, pad=8) == (82, 32, 36, 36)
	# a wide mask is squared around its centre, so a generated square is not squashed
	w = torch.zeros(100, 200); w[45:55, 50:150] = 1.0
	x, y, bw, bh = mask_bbox(w)
	assert bw == bh == 100 and x == 50 and y == 0
	# against an edge the box SLIDES inside instead of being clipped out of square
	e = torch.zeros(100, 200); e[40:60, 0:20] = 1.0
	assert mask_bbox(e, pad=10) == (0, 30, 40, 40)
	# a box bigger than the frame clamps rather than going negative
	big = torch.ones(20, 30)
	x, y, bw, bh = mask_bbox(big, pad=50)
	assert (x, y) == (0, 0) and bw <= 30 and bh <= 20
	# an empty frame gives the whole frame — a zero-sized box divides by zero later
	assert mask_bbox(torch.zeros(100, 200)) == (0, 0, 200, 100)


def test_batch_join_puts_an_expanded_list_back_together():
	from tinode.nodes.image.batch_join import BatchJoin
	frames = [torch.full((1, 4, 6, 3), i / 10) for i in range(5)]
	masks = [torch.full((1, 4, 6), i / 10) for i in range(5)]
	out, m, n = BatchJoin().execute(frames, masks)
	assert tuple(out.shape) == (5, 4, 6, 3) and tuple(m.shape) == (5, 4, 6) and n == 5
	# order is preserved
	for i in range(5):
		assert abs(float(out[i].mean()) - i / 10) < 1e-6
	# no masks wired = a zero mask of the right shape, not a crash
	out2, m2, _n2 = BatchJoin().execute(frames)
	assert tuple(m2.shape) == (5, 4, 6) and float(m2.sum()) == 0.0
	# a bare [H,W,C] item is accepted as one frame
	assert BatchJoin().execute([torch.zeros(4, 6, 3)])[2] == 1
	# a frame of a different size is named, not swallowed by a torch error
	try:
		BatchJoin().execute([torch.zeros(1, 4, 6, 3), torch.zeros(1, 8, 8, 3)])
	except RuntimeError as exc:
		assert "item 1 is (8, 8, 3)" in str(exc) and "same size" in str(exc)
	else:
		raise AssertionError("expected a RuntimeError on a size mismatch")
	try:
		BatchJoin().execute([])
	except RuntimeError as exc:
		assert "nothing wired" in str(exc)
	else:
		raise AssertionError("expected a RuntimeError on an empty list")


def test_prompt_template_drops_empty_slots_with_their_punctuation():
	from tinode.nodes.data.prompt_template import PromptTemplate, fill_template
	t = "a {2} {1}, {3}, {4}"
	assert fill_template(t, ["Ball bearing", "polished chrome", "golden hour",
							 "still life", "", ""]) == \
		"a polished chrome Ball bearing, golden hour, still life"
	# a missing middle slot takes its comma with it — no ", ," left behind
	assert fill_template(t, ["Ball bearing", "", "golden hour", "", "", ""]) == \
		"a Ball bearing, golden hour"
	# ...including at the very front and the very end
	assert fill_template("{1}, {2}, {3}", ["", "b", ""]) == "b"
	assert fill_template("{1} and {2}", ["", ""]) == "and"
	# an unknown slot stays visible rather than vanishing silently
	assert "{9}" in fill_template("x {9}", ["a"])
	# literal braces survive
	assert fill_template("{{1}} is {1}", ["one"]) == "{1} is one"
	# no template at all = join what was wired, in order
	assert fill_template("", ["a", "", "b"]) == "a, b"
	# multi-line templates keep their lines
	assert fill_template("{1}\n{2}", ["one", "two"]) == "one\ntwo"

	out, = PromptTemplate().execute(
		template="a {1}, {2}", text_1="Pearl", text_2="", prefix="close-up of",
		suffix="8k")
	assert out == "close-up of, a Pearl, 8k"
	# nothing wired at all still returns a string rather than failing
	assert PromptTemplate().execute(template="{1}")[0] == ""


def test_track_motion_circle_makes_a_context_ring_and_a_radius_ramp():
	from tinode.nodes.mask_motion.circle_track import (
		TrackMotionCircle, ramp_radii, render_circles,
	)
	# radius ramp: first frame is `start`, last is `end`, 0 means hold
	assert ramp_radii(5, 10, 20) == [10.0, 12.5, 15.0, 17.5, 20.0]
	assert ramp_radii(4, 10, 0) == [10.0] * 4
	assert ramp_radii(1, 10, 20) == [10.0]
	assert ramp_radii(0, 10, 20) == []

	# a per-frame radius really is per frame
	m = render_circles([(0.5, 0.5), (0.5, 0.5)], 128, 128, [8, 24], 0)
	assert float(m[1].sum()) > float(m[0].sum()) * 5

	track = json.dumps({"pts": [[0.3, 0.5, 0], [0.7, 0.5, 500]]})
	mask, _img, n, context, outside = TrackMotionCircle().execute(
		width=256, height=256, max_frames=5, radius=20, track=track,
		context_width=10)
	assert n == 5
	# the ring hugs the circle: it stays out of the circle's interior, meeting it
	# only where both edges are already fading into each other
	assert not bool(((mask > 0.9) & (context > 0.1)).any())
	assert float(context.sum()) > 0
	# and it sits immediately outside — area of an annulus 20..30
	ratio = float(context[0].sum()) / float(mask[0].sum())
	assert abs(ratio - ((30 ** 2 - 20 ** 2) / 20 ** 2)) < 0.15, ratio
	# `outside` is the complement of the circle
	assert float((mask + outside - 1.0).abs().max()) < 1e-6
	# no ring asked for = no ring given
	_m2, _i2, _n2, none_ring, _o2 = TrackMotionCircle().execute(
		width=64, height=64, max_frames=2, radius=8, track=track)
	assert float(none_ring.sum()) == 0.0
	# the circle really grows when radius_end is set
	grow, _i3, _n3, _c3, _o3 = TrackMotionCircle().execute(
		width=256, height=256, max_frames=5, radius=10, radius_end=40, track=track)
	assert float(grow[4].sum()) > float(grow[0].sum()) * 8
	# an empty track still returns all five outputs at the right shapes
	e_mask, e_img, e_n, e_ctx, e_out = TrackMotionCircle().execute(
		width=32, height=16, max_frames=3)
	assert e_n == 3 and tuple(e_mask.shape) == (3, 16, 32)
	assert tuple(e_img.shape) == (3, 16, 32, 3) and tuple(e_ctx.shape) == (3, 16, 32)
	assert float(e_ctx.sum()) == 0.0 and float(e_out.min()) == 1.0


def test_random_item_parses_a_messy_list():
	from tinode.nodes.data.random_item import format_item, parse_list
	items = parse_list("""
# Round things
> a note that is not an item

1. **Ball** — A portable spherical object.
12) Globe - A spherical representation of a world
- Pearl: a naturally formed smooth sphere
  *Marble*
209. **Yo-yo** — Two round disks joined around an axle.
200. **O-ring** — A flexible toroidal seal.

""")
	assert [i["name"] for i in items] == \
		["Ball", "Globe", "Pearl", "Marble", "Yo-yo", "O-ring"]
	# index is the position among KEPT items, not the number printed on the line
	assert [i["index"] for i in items] == [1, 2, 3, 4, 5, 6]
	assert items[0]["description"] == "A portable spherical object."
	assert items[1]["description"] == "A spherical representation of a world"
	assert items[2]["description"] == "a naturally formed smooth sphere"
	assert items[3]["description"] == ""            # a bare name is still an item
	# a hyphen with no spaces is part of the NAME, not a separator
	assert items[4]["name"] == "Yo-yo" and items[4]["description"].startswith("Two")
	assert items[5]["name"] == "O-ring"
	# headings, blockquotes and blank lines never become entries
	assert parse_list("# only a heading\n> and a note\n\n") == []
	assert parse_list("") == []

	it = items[0]
	assert format_item(it, "name") == "Ball"
	assert format_item(it, "description") == "A portable spherical object."
	assert format_item(it, "name — description") == "Ball — A portable spherical object."
	assert format_item(it, "raw line") == "1. **Ball** — A portable spherical object."
	# a nameless-description request on an item that has none falls back to the name
	assert format_item(items[3], "description") == "Marble"
	assert format_item(items[3], "name — description") == "Marble"


def test_random_item_draws_reproducibly():
	from tinode.nodes.data.random_item import RandomItem, choose, parse_list
	items = parse_list("\n".join(f"{i}. Item{i}" for i in range(1, 21)))
	assert len(items) == 20
	# the same seed is the same draw; a different seed is a different one
	assert choose(items, 7, 5) == choose(items, 7, 5)
	assert choose(items, 7, 5) != choose(items, 8, 5)
	# two DIFFERENT lists of the same length must not draw the same index off
	# one shared seed — five of these nodes normally run off a single seed
	other = parse_list("\n".join(f"{i}. Thing{i}" for i in range(1, 21)))
	assert len(other) == len(items)
	same_index = sum(1 for s in range(60)
					 if choose(items, s, 1)[0]["index"] == choose(other, s, 1)[0]["index"])
	assert same_index < 12, f"{same_index}/60 draws correlated across lists"
	# unique never repeats within a draw
	names = [i["name"] for i in choose(items, 3, 20, unique=True)]
	assert sorted(names) == sorted(i["name"] for i in items)
	# asking for more than the list holds gives the whole list, shuffled
	assert len(choose(items, 3, 99, unique=True)) == 20
	# without unique, repeats are allowed (and with this seed, happen)
	drawn = [i["name"] for i in choose(items, 1, 40, unique=False)]
	assert len(drawn) == 40 and len(set(drawn)) < 40
	assert choose([], 1, 3) == []

	# through the node, from the inline text source
	body, name, desc, index, size = RandomItem().execute(
		seed=42, source="text", text="1. **Ball** — round\n2. **Cube** — not round",
		count=1, format="name — description")
	assert size == 2 and index in (1, 2) and name in ("Ball", "Cube")
	assert body == f"{name} — {desc}"
	# count > 1 joins with newlines and the scalar outputs describe the first
	multi, first_name, _d, _i, _s = RandomItem().execute(
		seed=5, source="text", text="a\nb\nc\nd", count=3)
	assert len(multi.splitlines()) == 3
	assert multi.splitlines()[0] == first_name
	# every failure has to say what it actually looked at
	def fails(msg_part, **kw):
		try:
			RandomItem().execute(**kw)
		except RuntimeError as exc:
			assert msg_part in str(exc), f"{msg_part!r} not in {exc}"
			return str(exc)
		raise AssertionError(f"expected a RuntimeError mentioning {msg_part!r}")

	# a list of nothing but headings names the source and counts the lines
	msg = fails("no items found", seed=0, source="text",
				text="# a heading\n> a note\n\n")
	assert "inline text box" in msg and "3 line(s), 2 of them heading" in msg
	# the source switch on the wrong setting is called out by name, both ways
	fails("source is `text` but the text box is empty", seed=0, source="text",
		  text="", file_path="/some/list.txt")
	fails("There IS a list in the text box", seed=0, source="file",
		  file_path="", text="Ball")
	fails("file_path is empty", seed=0, source="file", file_path="")
	# a missing file names the path; a folder says so and lists what is inside
	fails("no such file", seed=0, source="file", file_path="/no/such/list.txt")
	folder = fails("That is a FOLDER", seed=0, source="file",
				   file_path=os.path.dirname(os.path.abspath(__file__)))
	assert "test_nodes.py" in folder


def test_track_motion_circle_replays_the_stroke_at_the_speed_it_was_drawn():
	from tinode.nodes.mask_motion.circle_track import parse_track, sample_positions
	# a stroke that crawls across the first half of the path and races the second:
	# 0 -> 0.5 takes 900ms, 0.5 -> 1.0 takes 100ms
	pts = [(0.0, 0.5, 0.0), (0.5, 0.5, 900.0), (1.0, 0.5, 1000.0)]
	rec = sample_positions(pts, 11, "recorded")
	assert len(rec) == 11
	assert rec[0] == (0.0, 0.5) and abs(rec[-1][0] - 1.0) < 1e-6
	# half the FRAMES are spent in the slow half, because half the TIME was
	assert abs(rec[5][0] - 0.5 * (500 / 900)) < 1e-6, rec[5]
	# 90% of the time was spent on the first half, so 10 of the 11 frames are
	# there and the fast half is crossed in the very last one
	assert abs(rec[9][0] - 0.5) < 1e-6, rec[9]
	assert sum(1 for x, _y in rec if x <= 0.5) == 10
	assert abs(rec[10][0] - 1.0) < 1e-6

	# even timing ignores the pace and walks equal distance per frame
	ev = sample_positions(pts, 11, "even")
	for f, (x, _y) in enumerate(ev):
		assert abs(x - f / 10) < 1e-6, (f, x)

	# a stroke drawn faster than the clock can see falls back to even, rather
	# than stacking every frame onto the last point
	flat = [(0.0, 0.0, 7.0), (1.0, 0.0, 7.0)]
	assert abs(sample_positions(flat, 3, "recorded")[1][0] - 0.5) < 1e-6
	# a press that never moved holds still for every frame
	assert sample_positions([(0.2, 0.3, 0.0), (0.2, 0.3, 500.0)], 4, "even") == \
		[(0.2, 0.3)] * 4
	# degenerate inputs give nothing rather than raising
	assert sample_positions([], 5, "recorded") == []
	assert sample_positions(pts, 0, "recorded") == []
	assert sample_positions(pts, 1, "recorded") == [(0.0, 0.5)]

	# the widget decoder survives junk, and defaults a missing timestamp
	assert parse_track("") == [] and parse_track("not json") == []
	assert parse_track(json.dumps({"pts": [[0.1, 0.2]]})) == [(0.1, 0.2, 0.0)]
	assert parse_track(json.dumps({"pts": [[0, 0, 0], "x", [1, 1, 5]]})) == \
		[(0.0, 0.0, 0.0), (1.0, 1.0, 5.0)]


def test_track_motion_circle_renders_a_circle_that_moves():
	from tinode.nodes.mask_motion.circle_track import TrackMotionCircle, render_circles
	m = render_circles([(0.5, 0.5)], 64, 64, 10, 0)
	assert tuple(m.shape) == (1, 64, 64)
	assert float(m[0, 32, 32]) == 1.0                       # centre is solid
	assert float(m[0, 0, 0]) == 0.0                         # corner is empty
	area = float(m.sum())
	assert abs(area - math.pi * 100) / (math.pi * 100) < 0.05, area   # ~pi r^2
	# radius is in PIXELS, so a non-square canvas still gives a round circle
	m2 = render_circles([(0.5, 0.5)], 128, 64, 10, 0)
	rows = (m2[0] > 0.5).sum(dim=1).max().item()
	cols = (m2[0] > 0.5).sum(dim=0).max().item()
	assert rows == cols, (rows, cols)
	# a circle wholly off-canvas leaves an empty frame instead of wrapping
	assert float(render_circles([(-2.0, -2.0)], 32, 32, 4, 0).sum()) == 0.0
	# feather softens the rim without hollowing the middle
	soft = render_circles([(0.5, 0.5)], 64, 64, 10, 6)
	assert float(soft[0, 32, 32]) == 1.0 and float(soft.sum()) < area

	# end to end: the circle is in a different place on the first and last frame
	track = json.dumps({"pts": [[0.1, 0.5, 0], [0.9, 0.5, 1000]]})
	mask, image, n, _ctx, _out = TrackMotionCircle().execute(
		width=256, height=128, max_frames=9, radius=12, track=track)
	assert n == 9 and tuple(mask.shape) == (9, 128, 256)
	assert tuple(image.shape) == (9, 128, 256, 3)
	def cx(fr):
		xs = (fr > 0.5).nonzero()
		return float(xs[:, 1].float().mean())
	assert abs(cx(mask[0]) - 0.1 * 256) < 1.5
	assert abs(cx(mask[8]) - 0.9 * 256) < 1.5
	assert cx(mask[4]) > cx(mask[0]) and cx(mask[8]) > cx(mask[4])
	# nothing drawn yet = empty frames of the right shape, not a crash
	blank, _img, bn, _c, _o = TrackMotionCircle().execute(
		width=64, height=32, max_frames=5)
	assert bn == 5 and tuple(blank.shape) == (5, 32, 64) and float(blank.sum()) == 0.0


def _write_png(path, arr):
	import numpy as np
	from PIL import Image
	os.makedirs(os.path.dirname(path), exist_ok=True)
	mode = "I;16" if arr.dtype == np.uint16 else ("RGBA" if arr.shape[-1] == 4 else "RGB")
	Image.fromarray(arr, mode=None if arr.ndim == 3 else mode).save(path)


def test_load_clip_frames_reads_a_png_sequence_in_filename_order():
	import numpy as np
	from tinode.nodes.image.clip_frames import (
		LoadClipFrames, find_frame_dir, list_frames, load_image_files,
	)
	with tempfile.TemporaryDirectory() as tmp:
		seq = os.path.join(tmp, "sh0090")
		# the AMIR shape: numbers run 0,2,4,... so index N is simply the Nth file
		for n, i in enumerate(range(0, 10, 2)):
			a = np.full((4, 6, 3), n * 20, dtype=np.uint8)
			_write_png(os.path.join(seq, f"sh0090.{i:04d}.png"), a)

		# the clip's own folder wins; a folder that holds the frames directly works too
		assert find_frame_dir(tmp, "sh0090") == seq
		assert find_frame_dir(seq, "nope") == seq
		assert find_frame_dir(tmp, "missing") == tmp
		assert find_frame_dir("", "x") is None

		files = list_frames(seq)
		assert [os.path.basename(f) for f in files] == [
			"sh0090.0000.png", "sh0090.0002.png", "sh0090.0004.png",
			"sh0090.0006.png", "sh0090.0008.png"]
		imgs = load_image_files(files)
		assert tuple(imgs.shape) == (5, 4, 6, 3) and imgs.dtype == torch.float32
		assert abs(float(imgs[2].mean()) - 40 / 255.0) < 1e-4     # index 2 = the 3rd file

		clip = {"stem": "sh0090", "crops": [], "fps": 25.0, "every_nth": 1}
		out, fps, n, folder = LoadClipFrames().execute(clip, frames_dir=tmp)
		assert n == 5 and fps == 25.0 and folder == seq
		assert torch.equal(out, imgs)

		# every_nth strides while loading — the skipped frames are never read
		out2, fps2, n2, _ = LoadClipFrames().execute(clip, frames_dir=tmp, every_nth=2)
		assert n2 == 3 and fps2 == 12.5
		assert torch.equal(out2, imgs[::2])
		# a wired source_stride overrides the widget, so the two can't disagree
		out3, fps3, n3, _ = LoadClipFrames().execute(
			clip, frames_dir=tmp, every_nth=1, every_nth_in=2)
		assert n3 == 3 and fps3 == 12.5 and torch.equal(out3, out2)
		# an explicit fps beats the manifest's
		assert LoadClipFrames().execute(clip, frames_dir=tmp, fps=50.0)[1] == 50.0

		# an empty folder names the folder it looked in rather than failing blankly
		try:
			LoadClipFrames().execute({"stem": "nothing", "crops": []}, frames_dir=tmp)
		except RuntimeError as exc:
			assert "no *.png" in str(exc)
		else:
			raise AssertionError("expected a RuntimeError for an empty sequence")


def test_load_clip_frames_keeps_16_bit_and_rejects_a_size_change():
	import numpy as np
	from tinode.nodes.image.clip_frames import list_frames, load_image_files
	with tempfile.TemporaryDirectory() as tmp:
		import cv2
		# 16-bit stays 16-bit: 32768/65535, not 128/255
		cv2.imwrite(os.path.join(tmp, "a0.png"),
					np.full((4, 4, 3), 32768, dtype=np.uint16))
		v = float(load_image_files([os.path.join(tmp, "a0.png")]).mean())
		assert abs(v - 32768 / 65535.0) < 1e-4, v
		# a frame of another size is caught, not silently stacked
		cv2.imwrite(os.path.join(tmp, "a1.png"), np.zeros((8, 8, 3), dtype=np.uint16))
		try:
			load_image_files(list_frames(tmp))
		except RuntimeError as exc:
			assert "must be the same size" in str(exc)
		else:
			raise AssertionError("expected a RuntimeError on a size change")


def test_load_clip_fills_can_skip_a_missing_source_video():
	from tinode.nodes.image import _mask_store as store
	from tinode.nodes.image.removal_pass import LoadClipFills
	with tempfile.TemporaryDirectory() as tmp:
		idir = os.path.join(tmp, "sh0090", "c00_k00")
		os.makedirs(store.mask_dir(idir), exist_ok=True)
		os.makedirs(store.filled_dir(idir), exist_ok=True)
		import numpy as np
		from PIL import Image
		for i in range(3):
			Image.fromarray(np.zeros((4, 6), np.uint8), "L").save(
				os.path.join(store.mask_dir(idir), store.MASK_PATTERN % i))
			Image.fromarray(np.zeros((4, 6, 3), np.uint8)).save(
				os.path.join(store.filled_dir(idir), store.FILLED_PATTERN % i))
		crop = {"stem": "sh0090", "crop_index": 0, "chunk_index": 0,
				"frame_start": 0, "frame_end": 3, "frame_count": 3,
				"has_filled": True, "filled_frames": 3, "source_every_nth": 1,
				"crop_info": {"H": 4, "W": 6, "C": 3,
							  "items": [{"y0": 0, "x0": 0, "h": 4, "w": 6,
										 "oy": 0, "ox": 0, "nh": 4, "nw": 6}]},
				"item_dir": idir, "mask_dir": store.mask_dir(idir)}
		clip = {"stem": "sh0090", "source_path": "", "fps": 25.0,
				"crops": [crop], "every_nth": 1}

		# the default still refuses, and says how to proceed
		try:
			LoadClipFills().execute(clip)
		except RuntimeError as exc:
			assert "Load Clip Frames" in str(exc)
		else:
			raise AssertionError("a missing source must fail by default")

		# ...and with require_source off, everything but `video` comes through
		video, crops, infos, masks, stem, count, starts, stride = \
			LoadClipFills().execute(clip, require_source=False)
		assert video is None and stem == "sh0090" and count == 1
		assert tuple(crops[0].shape) == (3, 4, 6, 3)
		assert tuple(masks[0].shape) == (3, 4, 6)
		assert starts == [0] and stride == 1 and infos[0] == crop["crop_info"]


def test_save_masks_drops_a_fill_belonging_to_a_re_authored_crop():
	from tinode.nodes.image.save_masks import fill_keys_to_carry
	prev = {"crop_width": 944, "crop_height": 1328, "frame_start": 12,
			"frame_end": 221, "frame_count": 209,
			"has_filled": True, "filled_frames": 112, "has_filled_pass1": True}
	same = dict(prev)                                   # re-saved, nothing moved
	carry, stale = fill_keys_to_carry(prev, same)
	assert sorted(carry) == ["filled_frames", "has_filled", "has_filled_pass1"]
	assert stale == []
	# a different BOX means the fill is of other pixels
	moved = dict(prev, crop_width=672, crop_height=1280)
	carry, stale = fill_keys_to_carry(prev, moved)
	assert carry == [] and sorted(stale) == ["filled_frames", "has_filled",
											 "has_filled_pass1"]
	# a different RANGE means the fill is of other frames
	shifted = dict(prev, frame_start=0, frame_end=209)
	assert fill_keys_to_carry(prev, shifted)[0] == []
	# a longer/shorter chunk at the same start, too
	assert fill_keys_to_carry(prev, dict(prev, frame_count=92))[0] == []
	# nothing rendered yet: nothing to carry and nothing to warn about
	assert fill_keys_to_carry({"crop_width": 10}, {"crop_width": 99}) == ([], [])


def test_pick_segments_grows_and_shrinks_selected_ids():
	from tinode.nodes.image.pick_segments import grow_segment
	seg = {"id": 7, "conf": 1.0, "bbox": [10, 10, 20, 20],
		   "mask": torch.ones(10, 10, dtype=torch.uint8)}
	# grow: the bbox expands by the margin and the mask fills it
	g = grow_segment(seg, 3, 100, 100)
	assert g["bbox"] == [7, 7, 23, 23] and tuple(g["mask"].shape) == (16, 16)
	assert int(g["mask"].sum()) == 16 * 16
	# shrink: the box stays, the mask erodes inward
	s = grow_segment(seg, -2, 100, 100)
	assert s["bbox"] == [10, 10, 20, 20]
	assert int(s["mask"].sum()) == 6 * 6
	# clamped at the frame edge, never negative coordinates
	edge = {"id": 1, "conf": 1.0, "bbox": [0, 0, 5, 5],
			"mask": torch.ones(5, 5, dtype=torch.uint8)}
	e = grow_segment(edge, 4, 8, 8)
	assert e["bbox"] == [0, 0, 8, 8]
	assert grow_segment(seg, 0, 100, 100) is seg          # 0 is a no-op

	# the widget only applies to the matching input signature
	assert PickSegments._grow_map('{"sig": "a", "grow": {"7": 3}}', "a") == {7: 3}
	assert PickSegments._grow_map('{"sig": "a", "grow": {"7": 3}}', "b") == {}
	assert PickSegments._grow_map("junk", "a") == {}

	# end to end: growing a selected id widens its mask in the output
	segs = {"num_frames": 1, "height": 40, "width": 40, "ids": [7],
			"frames": [[{"id": 7, "conf": 1.0, "bbox": [10, 10, 20, 20],
						 "mask": torch.ones(10, 10, dtype=torch.uint8)}]]}
	img = torch.zeros(1, 40, 40, 3)
	plain = _unwrap(PickSegments().execute(img, segs))[0]
	grown = _unwrap(PickSegments().execute(img, segs, grow_ids='{"grow": {"7": 3}}'))[0]
	assert int(grown.sum()) > int(plain.sum())
	assert int(plain.sum()) == 100 and int(grown.sum()) == 256


def test_pick_segments_filters_by_id():
	segs, img = _segments(), torch.zeros(2, 40, 40, 3)
	mask, _, out, _bbox = _unwrap(PickSegments().execute(img, segs, excluded_ids="[3]"))
	assert out["ids"] == [7]
	assert all(s["id"] != 3 for fr in out["frames"] for s in fr)
	assert int(mask[0].sum()) == 100                  # only id 7's 10x10 remains


def test_pick_segments_drops_a_stale_selection():
	"""Track ids are reused across clips, so a selection must not outlive its input."""
	# (import here to keep the signature helpers private to this test)
	from tinode.nodes.image.pick_segments import _img_signature, _seg_signature
	segs = _segments()
	img_a = torch.zeros(2, 40, 40, 3)
	sig_a = f"{_seg_signature(segs)}_{_img_signature(img_a)}"
	stored = json.dumps({"sig": sig_a, "ids": [3]})

	_, _, same, _b = _unwrap(PickSegments().execute(img_a, segs, excluded_ids=stored))
	assert same["ids"] == [7], "selection should apply to its own input"

	_, _, fresh, _b = _unwrap(PickSegments().execute(torch.rand(2, 40, 40, 3), segs,
												excluded_ids=stored))
	assert fresh["ids"] == [3, 7], "stale selection must be dropped on new input"

	_, _, legacy, _b = _unwrap(PickSegments().execute(img_a, segs, excluded_ids="[3]"))
	assert legacy["ids"] == [7], "bare-list selections stay supported"


def test_add_segments_boxes_persist_across_input_change():
	# Drawn boxes are reusable across takes: a signature from a DIFFERENT input
	# must NOT drop them (only the editor's clear button empties the list).
	raw = json.dumps({"sig": "some_other_input_signature",
					  "items": [{"id": 1000000, "frame": 0, "bbox": [1, 2, 5, 6]}]})
	kept = AddSegments._manual_items(raw, "the_current_signature")
	assert len(kept) == 1 and kept[0]["frame"] == 0
	# the legacy bare-list form still works, and junk is still ignored
	assert len(AddSegments._manual_items(json.dumps([{"id": 1, "frame": 0, "bbox": [0, 0, 2, 2]}]), "x")) == 1
	assert AddSegments._manual_items("not json", "x") == []


def test_add_segments_merges_and_matches_pick_outputs():
	assert AddSegments.RETURN_TYPES == PickSegments.RETURN_TYPES
	assert AddSegments.RETURN_NAMES == PickSegments.RETURN_NAMES

	segs, img = _segments(), torch.zeros(2, 40, 40, 3)
	manual = json.dumps([{"id": _MANUAL_ID_BASE, "frame": 1, "bbox": [0, 0, 5, 5]}])
	mask, _, out, _bbox = _unwrap(AddSegments().execute(img, segs, manual_segments=manual))
	assert out["ids"] == [3, 7, _MANUAL_ID_BASE]
	assert [s["id"] for s in out["frames"][1]] == [_MANUAL_ID_BASE]
	assert int(mask[1].sum()) == 25                   # the drawn 5x5 box
	# the incoming segments must not be mutated
	assert [s["id"] for s in segs["frames"][0]] == [3, 7]


def test_bbox_mask_is_filled_rectangles_and_is_appended():
	from tinode.nodes.image.segments_to_masks import bbox_mask
	segs = {
		"num_frames": 2, "height": 20, "width": 30, "ids": [1],
		"frames": [
			[{"id": 1, "bbox": [5, 4, 11, 9], "conf": 0.9,
			  "mask": torch.zeros(5, 6, dtype=torch.uint8)}],   # mask EMPTY on purpose
			[],
		],
	}
	bm = bbox_mask(segs)
	assert tuple(bm.shape) == (2, 20, 30)
	# the whole box is filled even though the segment's own mask is empty — the
	# point of this output is the rectangle, not the silhouette
	assert bm[0, 4:9, 5:11].min() == 1.0
	assert bm[0].sum() == 5 * 6
	assert bm[1].max() == 0.0                       # a frame with no segments stays black

	# appended last on every editor, so saved graphs keep their link slots
	for cls in (PickSegments, AddSegments, DeleteSegments):
		assert cls.RETURN_NAMES[:3] == ("mask", "image", "segments")
		assert cls.RETURN_NAMES[3] == "bbox_mask"
		assert cls.RETURN_TYPES[3] == "MASK"


def test_segments_to_masks_splits_per_id():
	from tinode.nodes.image.segments_to_masks import SegmentsToMasks
	segs = {"num_frames": 3, "height": 40, "width": 50, "ids": [5, 7], "frames": [
		[_segment(5, 0, 0, 10, 10)], [_segment(7, 20, 20, 35, 30)],
		[_segment(5, 5, 5, 15, 15)]]}
	masks, ids, count = SegmentsToMasks().execute(segs, object_ids="-1")
	assert count == 2 and ids == "5,7" and len(masks) == 2      # a LIST, one per id
	assert int(masks[0][0].sum()) == 100 and int(masks[0][1].sum()) == 0  # id5 absent frame1
	assert int(masks[1][1].sum()) == 150                        # id7 only on frame1
	# a single id yields a single-item list — "one at a time"
	only, ids1, c1 = SegmentsToMasks().execute(segs, object_ids="7")
	assert c1 == 1 and len(only) == 1 and ids1 == "7"
	# the list order follows the requested id order
	m2, ids2, _ = SegmentsToMasks().execute(segs, object_ids="7,5")
	assert ids2 == "7,5" and int(m2[0][1].sum()) == 150


def test_segment_mask_select_single_and_clamps():
	from tinode.nodes.image.segment_mask_select import SegmentMaskSelect
	segs = {"num_frames": 2, "height": 64, "width": 64, "ids": [5, 7, 9], "frames": [
		[_segment(5, 0, 0, 20, 20), _segment(7, 30, 30, 50, 50)],
		[_segment(9, 10, 40, 25, 60)]]}
	m0, id0, count = SegmentMaskSelect().execute(segs, index=0)
	assert torch.is_tensor(m0) and m0.shape == (2, 64, 64)      # a single MASK, not a list
	assert (id0, count) == (5, 3) and int(m0[0, 0:20, 0:20].sum()) == 400
	# out-of-range index clamps to the last object
	_, idlast, _ = SegmentMaskSelect().execute(segs, index=99)
	assert idlast == 9
	# empty set -> id -1, count 0, valid empty mask
	e, eid, ec = SegmentMaskSelect().execute(
		{"num_frames": 2, "height": 8, "width": 8, "ids": [], "frames": [[], []]})
	assert (eid, ec) == (-1, 0) and e.shape == (2, 8, 8)


def test_add_segments_ignores_degenerate_boxes():
	segs, img = _segments(), torch.zeros(2, 40, 40, 3)
	bad = json.dumps([{"id": 1, "frame": 99, "bbox": [0, 0, 5, 5]},     # frame OOR
					{"id": 2, "frame": 0, "bbox": [5, 5, 5, 9]}])     # zero width
	out = _unwrap(AddSegments().execute(img, segs, manual_segments=bad))[2]
	assert 1 not in out["ids"] and 2 not in out["ids"]


def test_mask_to_segment_is_compatible_and_tightly_cropped():
	mask = torch.zeros(3, 10, 12)
	mask[0, 2:7, 4:9] = 0.75
	mask[2, 1:3, 8:12] = 1.0
	segments = mask_to_segments(mask, threshold=0.5, segment_id=42)

	validate_segments(segments)
	assert segments["num_frames"] == 3
	assert (segments["height"], segments["width"]) == (10, 12)
	assert segments["ids"] == [42]
	assert segments["frames"][1] == []                 # alignment preserved
	first = segments["frames"][0][0]
	assert first["id"] == 42 and first["bbox"] == [4, 2, 9, 7]
	assert first["mask"].dtype == torch.uint8
	assert first["mask"].shape == (5, 5) and bool(first["mask"].all())

	# Its output flows directly into both existing segment editors.
	image = torch.zeros(3, 10, 12, 3)
	validate_segments(_unwrap(PickSegments().execute(image, segments))[2])
	validate_segments(_unwrap(AddSegments().execute(image, segments))[2])


def test_mask_to_segment_threshold_empty_and_bad_inputs():
	mask = torch.tensor([[0.5, 0.51]])
	segments = MaskToSegment().execute(mask, threshold=0.5)[0]
	assert segments["frames"][0][0]["bbox"] == [1, 0, 2, 1]

	empty = mask_to_segments(torch.zeros(2, 4, 5))
	assert empty["ids"] == [] and empty["frames"] == [[], []]
	for bad in (torch.zeros(1, 1, 2, 3), torch.empty(0, 2, 3)):
		try:
			mask_to_segments(bad)
		except ValueError:
			pass
		else:
			raise AssertionError(f"invalid mask shape {tuple(bad.shape)} was accepted")


def test_delete_segments_removes_only_the_clicked_frame_instance():
	segs, image = _segments(), torch.zeros(2, 40, 40, 3)
	from tinode.nodes.image.pick_segments import _img_signature, _seg_signature
	sig = f"{_seg_signature(segs)}_{_img_signature(image)}"
	raw = json.dumps({"sig": sig, "items": [{"frame": 0, "index": 1}]})

	mask, _, out, _bbox = _unwrap(DeleteSegments().execute(image, segs, deleted_items=raw))
	assert [s["id"] for s in out["frames"][0]] == [3]
	assert out["frames"][1] == []
	assert out["ids"] == [3]
	assert int(mask[0].sum()) == 100
	# Input objects are not mutated.
	assert [s["id"] for s in segs["frames"][0]] == [3, 7]


def test_delete_segments_drops_stale_and_malformed_selections():
	assert parse_deleted_items("bad json", "sig") == set()
	assert parse_deleted_items(
		json.dumps({"sig": "old", "items": [{"frame": 0, "index": 0}]}), "new"
	) == set()
	assert parse_deleted_items(
		json.dumps({"sig": "ok", "items": [
			{"frame": 2, "index": 4}, {"frame": -1, "index": 0}, {"bad": 1},
		]}), "ok"
	) == {(2, 4)}


# ---------------------------------------------------------------- utilities
def test_prune_asset_cache_keeps_newest_and_current():
	root = tempfile.mkdtemp()
	for i in range(7):
		d = os.path.join(root, f"sig{i}")
		os.makedirs(d)
		os.utime(d, (time.time() - 1000 + i,) * 2)
	removed = prune_asset_cache(root, keep="sig6", max_dirs=4)
	assert removed == 3
	assert sorted(os.listdir(root)) == ["sig3", "sig4", "sig5", "sig6"]
	# the current input survives even when it is the oldest
	root2 = tempfile.mkdtemp()
	for i in range(5):
		d = os.path.join(root2, f"s{i}")
		os.makedirs(d)
		os.utime(d, (time.time() - 1000 + i,) * 2)
	prune_asset_cache(root2, keep="s0", max_dirs=2)
	assert "s0" in os.listdir(root2)
	assert prune_asset_cache("/nonexistent/xyz", keep="a") == 0


def test_frame_pad_side_prepend_append_both():
	from tinode.nodes.image.frame_pad import FramePad
	base = torch.zeros(5, 8, 6, 3)
	for i in range(5):
		base[i] = i / 10.0

	# prepend only
	out, _m, info, total = FramePad().execute(
		base, side="prepend", auto_length=False, min_padding=2, head_fill="first_frame")
	assert info["head"] == 2 and info["side"] == "prepend" and total == 7
	assert torch.equal(out[2:], base) and torch.equal(out[0], base[0])

	# append only: nothing on the head, tail freezes the last frame
	out, _m, info, total = FramePad().execute(
		base, side="append", auto_length=False, min_padding=3, tail_fill="last_frame")
	assert info["head"] == 0 and total == 8 and torch.equal(out[:5], base) and torch.equal(out[-1], base[-1])

	# both: the pad is split across the two ends
	out, _m, info, total = FramePad().execute(
		base, side="both", auto_length=False, min_padding=4)
	assert info["head"] == 2 and total == 9 and torch.equal(out[2:7], base)

	# solid-colour head fill (the "empty video" you control)
	out, _m, _i, _t = FramePad().execute(
		base, side="prepend", auto_length=False, min_padding=2, head_fill="color",
		pad_color="#ff0000")
	assert torch.allclose(out[0], torch.tensor([1.0, 0.0, 0.0]).expand(8, 6, 3))

	# splice a differently-sized clip at the head — conformed, base untouched
	out, _m, info, _t = FramePad().execute(
		base, auto_length=False, min_padding=0, prepend_video=torch.rand(3, 16, 12, 4))
	assert out.shape[1:] == (8, 6, 3) and info["head"] == 3 and torch.equal(out[3:], base)


def test_frame_pad_rewind_pingpong():
	from tinode.nodes.image.frame_pad import FramePad, FrameUnpad, _rewind_indices
	assert _rewind_indices(5, 2, head=False) == [3, 2]     # after the end, play back
	assert _rewind_indices(5, 2, head=True) == [2, 1]       # leads into the start
	base = torch.zeros(6, 4, 5, 3)
	for i in range(6):
		base[i] = i / 10.0
	mask = torch.ones(6, 4, 5)
	out, m, info, total = FramePad().execute(
		base, mask=mask, side="append", auto_length=False, min_padding=3, tail_fill="rewind")
	assert info["head"] == 0 and total == 9
	assert torch.equal(out[:6], base)                       # real content untouched
	# the pad plays the shot backwards from the end (seam frame not duplicated)
	assert torch.equal(out[6], base[4]) and torch.equal(out[7], base[3]) and torch.equal(out[8], base[2])
	assert m[6:].min() == 1.0                               # mask carried, not zeroed
	back, n = FrameUnpad().execute(out, info)               # pad_info reverses it
	assert torch.equal(back, base) and n == 6


def test_frame_pad_every_side_round_trips():
	from tinode.nodes.image.frame_pad import FramePad, FrameUnpad
	img = torch.rand(30, 8, 10, 3)
	for side in ("prepend", "append", "both"):
		padded, _m, pc, total = FramePad().execute(img, side=side)   # modulo 8
		assert total % 8 == 0, side
		back, n = FrameUnpad().execute(padded, pc, expected_frames=30)
		assert torch.equal(back, img) and n == 30, side


def test_frame_pad_modulo_remainder_targets():
	from tinode.nodes.image.frame_pad import frame_pad_count
	# MiniMax H3: length % 17 == 5
	for L in range(1, 200):
		p = frame_pad_count(L, 8, 17, 5)
		assert p >= 8 and (L + p) % 17 == 5
	# VOID: multiple of 8 (remainder 0)
	for L in range(1, 200):
		assert (L + frame_pad_count(L, 8, 8, 0)) % 8 == 0


def test_insert_video_replace_and_insert():
	base = torch.zeros(10, 8, 6, 3)
	for i in range(10):
		base[i] = i / 10.0
	clip = torch.zeros(3, 8, 6, 3)
	for i in range(3):
		clip[i] = 0.5 + i / 100.0
	tag = lambda v: [round(float(f[0, 0, 0]), 3) for f in v]  # noqa: E731

	# replace: the clip's own length decides the end of the replaced span
	out, s, e = InsertVideo().execute(base, clip, start_frame=4, mode="replace")
	assert (s, e) == (4, 7) and out.shape[0] == 10
	assert tag(out) == [0.0, 0.1, 0.2, 0.3, 0.5, 0.51, 0.52, 0.7, 0.8, 0.9]

	# insert: nothing is lost, the clip pushes the rest later
	out, s, e = InsertVideo().execute(base, clip, start_frame=4, mode="insert")
	assert out.shape[0] == 13
	assert tag(out) == [0.0, 0.1, 0.2, 0.3, 0.5, 0.51, 0.52, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

	# inserting past the end extends rather than dropping frames
	out, s, _ = InsertVideo().execute(base, clip, start_frame=999, mode="replace")
	assert out.shape[0] == 13 and s == 10
	# a mismatched clip is conformed to the base
	out, _, _ = InsertVideo().execute(base, torch.rand(2, 16, 12, 4), start_frame=0)
	assert out.shape == (10, 8, 6, 3)


def test_trim_video_cuts_both_ends():
	base = torch.zeros(10, 8, 6, 3)
	for i in range(10):
		base[i] = i / 10.0
	out, n = TrimVideo().execute(base, trim_start=2, trim_end=3)
	assert n == 5 and torch.equal(out, base[2:7])       # exact original pixels
	assert torch.equal(TrimVideo().execute(base, 0, 0)[0], base)
	# never emit an empty batch
	_, n = TrimVideo().execute(base, trim_start=50, trim_end=50)
	assert n == 1


def test_cut_video_uses_python_indices_and_exact_counts():
	base = torch.arange(10, dtype=torch.float32).reshape(10, 1, 1, 1)
	(out,) = CutVideo().execute(base, start_index=2, frame_count=3)
	assert out[:, 0, 0, 0].tolist() == [2, 3, 4]

	(out,) = CutVideo().execute(base, start_index=-8, frame_count=3)
	assert out[:, 0, 0, 0].tolist() == [2, 3, 4]

	(out,) = CutVideo().execute(base, start_index=-1, frame_count=1)
	assert out[:, 0, 0, 0].tolist() == [9]


def test_cut_video_rejects_invalid_or_inexact_ranges():
	for args in ((10, 0, 0), (10, 10, 1), (10, -11, 1), (10, 8, 3)):
		try:
			cut_bounds(*args)
		except (ValueError, IndexError):
			pass
		else:
			raise AssertionError(f"cut_bounds{args} should have failed")

	try:
		CutVideo().execute(torch.empty(0, 4, 4, 3), 0, 1)
	except ValueError:
		pass
	else:
		raise AssertionError("an empty input batch should have failed")


def test_insert_then_trim_restores_the_base():
	base = torch.zeros(10, 8, 6, 3)
	for i in range(10):
		base[i] = i / 10.0
	clip = torch.rand(3, 8, 6, 3)
	out, s, e = InsertVideo().execute(base, clip, start_frame=0, mode="insert")
	back, _ = TrimVideo().execute(out, trim_start=e - s, trim_end=0)
	assert torch.equal(back, base)


def test_web_js_calls_are_all_defined():
	"""Catch a helper that is called but no longer defined.

	`node --check` only validates syntax, so deleting a function while leaving
	its call sites parses fine and then throws ReferenceError at runtime — which
	is exactly how the Add Segments preview broke once. Compare called names
	against what each module defines or imports.
	"""
	import re

	def strip_noise(src):
		"""Drop comments then string/template literals, so prose and CSS in
		them ('the boxes (above)', `rgb(...)`) aren't mistaken for calls."""
		src = re.sub(r"/\*.*?\*/", " ", src, flags=re.S)
		src = re.sub(r"//[^\n]*", " ", src)
		src = re.sub(r"`(?:\\.|[^`\\])*`", '""', src, flags=re.S)
		src = re.sub(r"'(?:\\.|[^'\\])*'", '""', src)
		src = re.sub(r'"(?:\\.|[^"\\])*"', '""', src)
		return src

	web = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "web")
	keywords = {
		"if", "for", "while", "switch", "catch", "return", "function", "typeof",
		"await", "new", "delete", "void", "in", "of", "do", "else",
	}
	globals_ = {
		"Math", "JSON", "Object", "Array", "Number", "String", "Boolean", "Set",
		"Map", "Image", "Promise", "parseInt", "parseFloat", "isNaN", "console",
		"document", "window", "requestAnimationFrame", "cancelAnimationFrame",
		"setTimeout", "clearTimeout", "setInterval", "clearInterval",
		"ResizeObserver", "performance",
		"encodeURIComponent", "decodeURIComponent", "Infinity",
		"alert", "confirm", "prompt", "fetch", "FormData", "URL", "Date", "Error",
	}
	problems = []
	for name in sorted(os.listdir(web)):
		if not name.endswith(".js"):
			continue
		src = strip_noise(open(os.path.join(web, name)).read())
		defined = set(re.findall(r"(?:export\s+)?function\s+(\w+)", src))
		defined |= set(re.findall(r"(?:const|let|var)\s+(\w+)\s*=", src))
		# object/class method shorthand, e.g. `async beforeRegisterNodeDef(a, b) {`
		defined |= set(re.findall(r"(?:async\s+)?(\w+)\s*\([^()]*\)\s*\{", src))
		# Parameter names: a callback passed in and invoked (`mkBtn(txt, fn)` ->
		# `fn()`) is defined at its call site, not a missing global.
		params_found = re.findall(r"(?:function\s*\w*|\w+)\s*\(([^()]*)\)\s*\{", src)
		# arrow functions, including `const f = (a, b) => {` where the token
		# before "(" is "=" rather than a name
		params_found += re.findall(r"\(([^()]*)\)\s*=>", src)
		for params in params_found:
			for tok in params.split(","):
				tok = tok.strip().lstrip(".").split("=")[0].strip()
				if re.fullmatch(r"\w+", tok or ""):
					defined.add(tok)
		for imp in re.findall(r"import\s*\{([^}]*)\}\s*from", src):
			defined |= {x.strip().split(" as ")[-1] for x in imp.split(",") if x.strip()}
		defined |= set(re.findall(r"import\s+(\w+)\s+from", src))
		# bare calls: `name(` not preceded by a dot (so not a method call)
		for call in set(re.findall(r"(?<![.\w])([a-z_]\w*)\s*\(", src)):
			if call in defined or call in globals_ or call in keywords:
				continue
			problems.append(f"{name}: calls {call}() which is not defined or imported")
	assert not problems, "undefined helper(s):\n  " + "\n  ".join(problems)


def test_web_js_imports_the_comfy_singletons_it_uses():
	"""A module using `app.` / `api.` must import it.

	The call-site check above only looks at bare calls, so `api.apiURL(...)`
	slipped through when a refactor dropped the `api` import — the module still
	parsed and only threw at runtime, leaving Bbox Crop stuck on its placeholder.
	Full scope analysis is overkill; these two ComfyUI singletons are the ones
	that actually get dropped.
	"""
	import re

	web = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "web")
	problems = []
	for root, _dirs, files in os.walk(web):
		for fname in sorted(f for f in files if f.endswith(".js")):
			path = os.path.join(root, fname)
			src = open(path).read()
			body = re.sub(r"^import[\s\S]*?;\s*$", "", src, flags=re.M)  # drop import lines
			for singleton in ("app", "api"):
				uses = re.search(rf"(?<![.\w]){singleton}\.", body)
				imported = re.search(rf"import\s*\{{[^}}]*\b{singleton}\b[^}}]*\}}\s*from", src)
				if uses and not imported:
					rel = os.path.relpath(path, web)
					problems.append(f"{rel}: uses `{singleton}.` but never imports it")
	assert not problems, "missing import(s):\n  " + "\n  ".join(problems)


def test_color_for_id_is_stable_and_matches_js():
	"""color_for_id is duplicated in web/lib/editor.js and MUST agree.

	These expected values are the shared golden-ratio HSV formula; if you change
	one implementation this test fails until you change the other.
	"""
	for sid in (0, 1, 3, 7, 42, 1000000):
		r, g, b = color_for_id(sid)
		assert all(0.0 <= c <= 1.0 for c in (r, g, b)), (sid, (r, g, b))
		assert max(r, g, b) == 1.0, f"id {sid} should hit full value"
	assert color_for_id(5) == color_for_id(5)                 # deterministic
	assert color_for_id(5) != color_for_id(6)                 # distinguishable
	# byte values the JS side must reproduce exactly
	as_bytes = lambda sid: tuple(round(c * 255) for c in color_for_id(sid))  # noqa: E731
	assert as_bytes(0) == (255, 89, 89), as_bytes(0)
	assert as_bytes(1) == (89, 138, 255), as_bytes(1)
	assert as_bytes(3) == (255, 89, 234), as_bytes(3)
	assert as_bytes(7) == (96, 255, 89), as_bytes(7)


def test_color_for_id_python_and_js_agree():
	"""Actually run the JS colorForId and compare, rather than trusting a comment.

	Skipped when node isn't installed. This is the only thing standing between
	the two implementations and a silent divergence, since a segment's colour is
	produced by Python for the image output and by JS for the editor.
	"""
	import re
	import shutil
	import subprocess

	if not shutil.which("node"):
		print("    (skipped: node not installed)", end="")
		return

	root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
	src = open(os.path.join(root, "web", "lib", "editor.js")).read()
	m = re.search(r"export (function colorForId\(id\) \{(?:.|\n)*?\n\})", src)
	assert m, "could not find colorForId in web/lib/editor.js"

	ids = [0, 1, 3, 7, 42, 1000000]
	script = m.group(1) + (
		f"\nconsole.log(JSON.stringify({ids}.map(colorForId)));\n"
	)
	out = subprocess.run(["node", "-e", script], capture_output=True, text=True, check=True)
	js = [tuple(v) for v in json.loads(out.stdout)]
	py = [tuple(round(c * 255) for c in color_for_id(i)) for i in ids]
	assert js == py, f"colour drift!\n  js={js}\n  py={py}"


def test_load_videos_scans_folder_sorted():
	with tempfile.TemporaryDirectory() as d:
		for name in ("b.mp4", "a.mov", "c.mkv", "notes.txt", "still.png"):
			open(os.path.join(d, name), "w").close()
		paths = resolve_video_files("", "", base_dir=d)
		assert [os.path.basename(p) for p in paths] == ["a.mov", "b.mp4", "c.mkv"]
		rev = resolve_video_files("", "", reverse=True, base_dir=d)
		assert [os.path.basename(p) for p in rev] == ["c.mkv", "b.mp4", "a.mov"]


def test_load_videos_explicit_filenames_keep_order():
	with tempfile.TemporaryDirectory() as d:
		for name in ("a.mp4", "b.mp4", "c.mp4"):
			open(os.path.join(d, name), "w").close()
		paths = resolve_video_files("", "c.mp4\na.mp4", base_dir=d)
		assert [os.path.basename(p) for p in paths] == ["c.mp4", "a.mp4"]


def test_load_videos_forgives_redundant_input_prefix():
	with tempfile.TemporaryDirectory() as base:
		# `base` stands in for ComfyUI's input/ dir; the real folder is base/dicaire.
		sub = os.path.join(base, "dicaire")
		os.makedirs(sub)
		# Every natural spelling the user might type must land on base/dicaire.
		for spec in ("dicaire", "input/dicaire", "/input/dicaire", "/dicaire"):
			assert resolve_dir(spec, base_dir=base) == sub, spec


def test_load_videos_pattern_filters_folder():
	with tempfile.TemporaryDirectory() as d:
		for name in ("PROJECT_AMIR_01.mp4", "PROJECT_AMIR_02.mov",
					 "OTHER_03.mp4", "notes.txt"):
			open(os.path.join(d, name), "w").close()
		# extension glob
		mp4 = resolve_video_files("", pattern="*.mp4", base_dir=d)
		assert [os.path.basename(p) for p in mp4] == ["OTHER_03.mp4", "PROJECT_AMIR_01.mp4"]
		# stem glob, case-insensitive, still videos-only (no notes.txt)
		amir = resolve_video_files("", pattern="project_amir_*", base_dir=d)
		assert [os.path.basename(p) for p in amir] == [
			"PROJECT_AMIR_01.mp4", "PROJECT_AMIR_02.mov"]


def test_load_videos_glob_in_filenames_and_dedup():
	with tempfile.TemporaryDirectory() as d:
		for name in ("a1.mp4", "a2.mp4", "b1.mp4"):
			open(os.path.join(d, name), "w").close()
		# a glob line plus an exact line; a1 appears in both but is not duplicated
		paths = resolve_video_files("", filenames="a*.mp4\na1.mp4", base_dir=d)
		assert [os.path.basename(p) for p in paths] == ["a1.mp4", "a2.mp4"]


def test_load_videos_pattern_no_match_errors():
	with tempfile.TemporaryDirectory() as d:
		open(os.path.join(d, "a.mp4"), "w").close()
		try:
			resolve_video_files("", pattern="ZZZ_*", base_dir=d)
		except RuntimeError as exc:
			assert "ZZZ_" in str(exc)
		else:
			raise AssertionError("expected a RuntimeError for a pattern with no match")


def test_load_videos_missing_named_file_errors():
	with tempfile.TemporaryDirectory() as d:
		open(os.path.join(d, "a.mp4"), "w").close()
		try:
			resolve_video_files("", "a.mp4\nghost.mp4", base_dir=d)
		except RuntimeError as exc:
			# names the missing file AND lists what is actually there
			assert "ghost.mp4" in str(exc) and "a.mp4" in str(exc)
		else:
			raise AssertionError("expected a RuntimeError for the missing file")


def test_load_videos_matches_name_without_extension():
	with tempfile.TemporaryDirectory() as d:
		open(os.path.join(d, "DICAIRE T 05 pour test IA.mp4"), "w").close()
		paths = resolve_video_files("", "DICAIRE T 05 pour test IA", base_dir=d)
		assert [os.path.basename(p) for p in paths] == ["DICAIRE T 05 pour test IA.mp4"]


def test_load_videos_lone_slash_is_the_input_dir_not_root():
	with tempfile.TemporaryDirectory() as d:
		open(os.path.join(d, "a.mp4"), "w").close()
		# "/" must resolve to base_dir (the input folder), never the FS root.
		assert resolve_dir("/", base_dir=d) == d
		paths = resolve_video_files("/", "", base_dir=d)
		assert [os.path.basename(p) for p in paths] == ["a.mp4"]


def test_load_videos_absolute_dir_is_used_as_is():
	with tempfile.TemporaryDirectory() as d:
		# An absolute `directory` must ignore base_dir entirely.
		assert resolve_dir(d, base_dir="/nonexistent/input") == d


def test_load_videos_empty_folder_errors():
	with tempfile.TemporaryDirectory() as d:
		try:
			resolve_video_files("", "", base_dir=d)
		except RuntimeError as exc:
			assert "no video files" in str(exc)
		else:
			raise AssertionError("expected a RuntimeError for an empty folder")


def test_split_video_chunks_by_string_and_roundtrip():
	from tinode.nodes.image.split_video import (
		ChunkItem, JoinChunks, SplitVideoChunks, chunk_bounds, parse_cuts)
	# the user's example: "18, 124" -> 0..17, 18..123, 124..end
	assert parse_cuts("18, 124", 200) == [18, 124]
	assert chunk_bounds(200, [18, 124]) == [(0, 18), (18, 124), (124, 200)]
	# sloppy input still parses; out-of-range and dupes dropped
	assert parse_cuts("18  18\n124, junk, 0, 999", 200) == [18, 124]
	assert parse_cuts("-1", 200) == [199]              # negative indexes from the end

	img = torch.rand(200, 8, 10, 3)
	items, count = SplitVideoChunks().execute(img, cuts="18, 124")
	assert count == 3
	# every chunk is a bit-exact slice, and joining them restores the clip
	acc = None
	for i, it in enumerate(items):
		imgs, _m, start, end, idx, total = ChunkItem().execute(it)
		assert (idx, total) == (i, 3)
		assert torch.equal(imgs, img[start:end])
		acc, n = JoinChunks().execute(imgs, accumulator=acc)
	assert torch.equal(acc, img) and n == 200          # lossless round trip


def test_split_video_max_length_splits_evenly_for_a_model_ceiling():
	from tinode.nodes.image.split_video import chunk_bounds
	# VOID: 197 cap, Frame Pad prepends >=8, so 189 is the usable ceiling.
	spans = chunk_bounds(540, [], max_length=189)
	assert len(spans) == 3
	assert spans[0][0] == 0 and spans[-1][1] == 540
	lengths = [b - a for a, b in spans]
	assert all(l <= 189 for l in lengths)
	assert max(lengths) - min(lengths) <= 1            # even, not a stub tail
	assert sum(lengths) == 540                          # covers everything, no gaps


def test_join_chunks_tolerates_the_loop_seed():
	from tinode.nodes.image.split_video import JoinChunks
	a = torch.rand(3, 4, 5, 3)
	# first iteration: the accumulator carries Inspire's seed, not an IMAGE
	out, n = JoinChunks().execute(a, accumulator=7)
	assert torch.equal(out, a) and n == 3


def test_load_masks_filters_and_reports():
	from tinode.nodes.image.load_masks import build_report, match_stem
	# stem filter: exact names and globs, empty selects everything
	assert match_stem("sh0010", "") is True
	assert match_stem("sh0010", "sh001*") is True
	assert match_stem("sh0010", "SH001*") is True          # case-insensitive
	assert match_stem("sh0020", "sh0010, sh0030") is False
	assert match_stem("sh0030", "sh0010, sh0030") is True

	found = [
		{"stem": "sh0010", "chunk_index": 0, "crop_index": 0, "frame_start": 0,
		 "frame_end": 85, "crop_width": 560, "crop_height": 944, "has_crop": True,
		 "has_filled": True},
		{"stem": "sh0010", "chunk_index": 1, "crop_index": 0, "frame_start": 85,
		 "frame_end": 172, "crop_width": 608, "crop_height": 832, "has_crop": True,
		 "has_filled": False},
	]
	rep = build_report(found, [found[1]], "/store")
	assert "2 crop(s) in 1 clip(s); 1 selected" in rep
	assert "filled" in rep and "pending" in rep
	assert "chunk 1 crop 0  frames 85-172" in rep


def test_chunked_crops_get_distinct_folders():
	# THE bug: with chunks, crop_index alone is not unique, so chunk 1's crop 0
	# overwrote chunk 0's crop 0 and only the last chunk survived.
	with tempfile.TemporaryDirectory() as root:
		a = _mstore.item_dir(root, "sh0030", 0, 0)
		b = _mstore.item_dir(root, "sh0030", 0, 1)
		c = _mstore.item_dir(root, "sh0030", 1, 0)
		assert a != b != c and a != c, "each chunk/crop pair needs its own folder"
		# an unchunked clip keeps the flat layout
		assert _mstore.item_dir(root, "sh0030", 0) .endswith("00")


def test_paste_back_frame_offset_lands_on_the_chunk_frames():
	# A crop taken from a CHUNK covers part of the clip; without the offset the
	# result was pasted onto the START of the video.
	img = torch.rand(20, 40, 50, 3)
	chunk = img[8:14]
	crop, info = _unwrap(BboxCropManual().execute(chunk, x=5, y=5, width=16, height=16))
	gen = torch.zeros_like(crop)
	mask = torch.ones(6, 16, 16)
	(out,) = MaskCropPasteBack().execute(img, gen, info, masks=mask, feather=0,
										 frame_offset=8)
	assert torch.equal(out[:8], img[:8]), "frames before the chunk must be untouched"
	assert torch.equal(out[14:], img[14:]), "frames after the chunk must be untouched"
	assert torch.equal(out[8:14, 5:21, 5:21, :], gen)     # the chunk's frames got it
	# and it refuses to run off the end rather than silently truncating
	try:
		MaskCropPasteBack().execute(img, gen, info, masks=mask, frame_offset=18)
	except RuntimeError as exc:
		assert "runs past" in str(exc)
	else:
		raise AssertionError("expected an error when the offset overruns the clip")


def test_item_params_stores_per_key_and_guards_stale_entries():
	from tinode.nodes.image.item_params import ItemParams

	def run(**kw):
		r = ItemParams().execute(**kw)
		return r["result"] if isinstance(r, dict) else r

	# an edit stamped with its own key is saved under that key
	j, table, _ = run(key="sh0030|0", entry='{"prompt": "jacket"}', entry_key="sh0030|0")
	assert json.loads(j)["prompt"] == "jacket"

	# THE important case: after stepping, the box still holds the PREVIOUS item's
	# text. Writing that onto the new key would silently ruin the batch.
	j2, table2, _ = run(key="sh0030|1", entry='{"prompt": "jacket"}',
						entry_key="sh0030|0", table=table)
	assert json.loads(j2) == {}, "a stale entry must not be saved onto the next key"
	assert "sh0030|1" not in json.loads(table2)

	# defaults fill in under a key's own settings, never over them
	j3, _, _ = run(key="sh0030|0", defaults='{"threshold": 0.4, "prompt": "x"}',
				   table=table)
	got = json.loads(j3)
	assert got["prompt"] == "jacket" and got["threshold"] == 0.4

	# stepping back returns the saved settings
	j4, _, _ = run(key="sh0030|0", table=table)
	assert json.loads(j4)["prompt"] == "jacket"


def test_param_get_types_and_fallback():
	from tinode.nodes.image.item_params import ParamGet
	blob = '{"prompt": "green jacket", "threshold": 0.45, "n": 3, "flag": true, "boxes": [{"x": 1}]}'
	assert ParamGet().execute(blob, "prompt")[0] == "green jacket"
	assert ParamGet().execute(blob, "threshold")[1] == 0.45      # FLOAT drives a threshold
	assert ParamGet().execute(blob, "n")[2] == 3                 # INT drives a count
	assert ParamGet().execute(blob, "flag")[3] is True
	assert ParamGet().execute(blob, "boxes")[0] == '[{"x": 1}]'  # JSON back out as text
	# a missing name falls back instead of erroring, so a half-filled table runs
	text, f, _i, _b, found = ParamGet().execute(blob, "missing", fallback="0.3")
	assert (text, f, found) == ("0.3", 0.3, False)
	assert ParamGet().execute("not json", "prompt", fallback="fb")[0] == "fb"


def test_item_cursor_steps_clamps_and_wraps():
	from tinode.nodes.image.item_cursor import ItemCursor, _describe
	items = [{"stem": "sh0030", "crop_index": i} for i in range(3)]

	def run(idx, wrap=False):
		r = ItemCursor().execute(items, index=idx, wrap=wrap)
		return r["result"] if isinstance(r, dict) else r

	item, i, n, last = run(1)
	assert item is items[1] and (i, n, last) == (1, 3, False)
	# past the end clamps to the last item (and reports it), or wraps when asked
	assert run(99)[1:] == (2, 3, True)
	assert run(3, wrap=True)[1] == 0
	assert run(-1)[1] == 0                       # never below the first item
	# the label is what the node shows, so it must name the actual work
	assert _describe(items[2]) == "sh0030 · crop 2"
	assert "chunk 1" in _describe({"chunk_index": 1, "start": 0, "end": 9})


def test_item_cursor_rejects_an_empty_list():
	from tinode.nodes.image.item_cursor import ItemCursor
	try:
		ItemCursor().execute([], index=0)
	except RuntimeError as exc:
		assert "empty" in str(exc)
	else:
		raise AssertionError("expected an error for an empty list")


def test_load_video_outputs_stay_backward_compatible():
	# A saved workflow stores links by slot INDEX, so the original three outputs
	# must keep their positions; stem/path/video are appended after them.
	from tinode.nodes.image.load_video import LoadVideo
	assert LoadVideo.RETURN_NAMES[:3] == ("images", "frame_count", "fps")
	assert LoadVideo.RETURN_NAMES[3:] == ("stem", "path", "video")
	assert LoadVideo.RETURN_TYPES[3:] == ("STRING", "STRING", "VIDEO")
	# the handles the per-clip store keys on now match Load Videos' item
	from tinode.nodes.image.load_masks import LoadMask
	assert "stem" in LoadMask.RETURN_NAMES


def test_match_source_format_maps_to_save_video_options():
	from tinode.nodes.image.match_format import (
		colorspace_for, format_for, pix_fmt_for, range_for)
	# The AMIR footage: h264 / yuv420p / tv / bt709 -> must round-trip to itself.
	assert format_for("h264") == "h264 (mp4)"
	assert format_for("vp9") == "vp9 (webm)"
	assert format_for("hevc") == "h264 (mp4)"          # unsupported -> our h264
	assert pix_fmt_for("yuv420p") == "yuv420p"
	assert pix_fmt_for("yuv444p10le") == "yuv444p"     # keep 4:4:4 when the source is
	assert range_for("tv") == "tv" and range_for("limited") == "tv"
	assert range_for("pc") == "pc" and range_for("full") == "pc"
	assert range_for("") == "tv"                        # untagged -> broadcast default
	assert colorspace_for("bt709") == "bt709"
	assert colorspace_for("smpte170m") == "smpte170m"
	assert colorspace_for("") == "bt709"
	# every mapped value must be a real Save Video option
	from tinode.nodes.image.save_video import _FORMATS
	assert format_for("h264") in _FORMATS


def test_validation_gate_passes_through_headless_and_resolves():
	from tinode.nodes.image.validation_gate import ValidationGate, ANY, _resolve
	# Headless (no server in the test env): must pass the value straight through,
	# never block a batch/cron run.
	(out,) = ValidationGate().execute("payload")
	assert out == "payload"
	assert _resolve("reject") == "reject" and _resolve("Rejected") == "reject"
	assert _resolve("approve") == "approve" and _resolve("anything else") == "approve"
	# the wildcard type compares equal to any concrete type string
	assert ANY == "MASK" and ANY == "IMAGE" and not (ANY != "TI_CROP_XFORM")


def test_bbox_multi_emits_an_item_list_for_a_foreach_loop():
	from tinode.nodes.image.crop_bbox_manual import BboxCropMulti, CropItem, resolve_box
	img = torch.rand(4, 100, 120, 3)
	boxes = json.dumps([{"x": 10, "y": 20, "w": 40, "h": 30},
						{"x": 60, "y": 5, "w": 32, "h": 48}])
	items, count = BboxCropMulti().execute(img, boxes=boxes)
	# An Inspire ITEM_LIST: a plain list the loop slices, one entry per box.
	assert isinstance(items, list) and count == 2
	# Crop Item unpacks one iteration's item back into the three signals.
	crop0, info0, idx0 = CropItem().execute(items[0])
	crop1, info1, idx1 = CropItem().execute(items[1])
	assert (idx0, idx1) == (0, 1)
	assert tuple(crop0.shape) == (4, 30, 40, 3)
	assert tuple(crop1.shape) == (4, 48, 32, 3)
	x0, y0, x1, y1 = resolve_box({"x": 10, "y": 20, "w": 40, "h": 30}, 120, 100, 1)
	assert torch.equal(crop0, img[:, y0:y1, x0:x1, :])
	(back,) = CropByInfo().execute(img, info1)
	assert torch.equal(back, crop1)


def test_composite_crops_use_mask_off_writes_the_whole_rectangle():
	from tinode.nodes.image.crop_bbox_manual import BboxCropMulti, CropItem
	from tinode.nodes.image.removal_pass import CompositeCrops
	img = torch.rand(2, 90, 120, 3)
	items, _ = BboxCropMulti().execute(img, boxes=json.dumps(
		[{"x": 10, "y": 10, "w": 40, "h": 30}]))
	crop, info, _ = CropItem().execute(items[0])
	gen = torch.zeros_like(crop)                     # the "removed" version
	mask = torch.zeros(crop.shape[0], crop.shape[1], crop.shape[2])
	mask[:, 5:15, 5:15] = 1.0                        # only a small region masked

	# use_mask on: only the masked part changes
	(on,) = CompositeCrops().execute(img, [crop * 0], [info], masks=[mask],
									 feather=0, use_mask=True)
	assert torch.equal(on[:, 10:15, 10:50, :], img[:, 10:15, 10:50, :]), \
		"inside the crop but outside the mask must stay original"

	# use_mask off: the whole crop rectangle is replaced
	(off,) = CompositeCrops().execute(img, [gen], [info], masks=[mask],
									  feather=0, use_mask=False)
	assert torch.equal(off[:, 10:40, 10:50, :], gen), "the whole rectangle is written"
	assert torch.equal(off[:, :10, :, :], img[:, :10, :, :]), "outside it is untouched"


def test_composite_crops_pastes_multiple_crops_back():
	from tinode.nodes.image.crop_bbox_manual import BboxCropMulti
	from tinode.nodes.image.removal_pass import CompositeCrops
	img = torch.rand(2, 90, 120, 3)
	boxes = json.dumps([{"x": 10, "y": 10, "w": 40, "h": 30},
						{"x": 70, "y": 40, "w": 32, "h": 40}])
	items, _count = BboxCropMulti().execute(img, boxes=boxes)
	from tinode.nodes.image.crop_bbox_manual import CropItem
	unpacked = [CropItem().execute(it) for it in items]
	crops = [u[0] for u in unpacked]
	infos = [u[1] for u in unpacked]
	# full masks per crop -> compositing the untouched crops back is the identity
	masks = [torch.ones(c.shape[0], c.shape[1], c.shape[2]) for c in crops]
	(out,) = CompositeCrops().execute(img, crops, infos, masks=masks, feather=0)
	assert torch.equal(out, img), "identity composite of every crop must restore the frame"


def test_bbox_multi_empty_is_whole_frame():
	from tinode.nodes.image.crop_bbox_manual import BboxCropMulti, CropItem
	img = torch.rand(2, 40, 50, 3)
	items, count = BboxCropMulti().execute(img, boxes="[]")
	assert count == 1
	crop, _info, idx = CropItem().execute(items[0])
	assert idx == 0 and torch.equal(crop, img)       # whole frame


def test_divide_rectangle_tiles_and_reassembles():
	from tinode.nodes.div.rectangle import DivideRectangle, split_spans
	# even split, and the remainder is spread over the FIRST cells
	assert split_spans(10, 2) == [(0, 5), (5, 10)]
	assert split_spans(10, 3) == [(0, 4), (4, 7), (7, 10)]
	assert split_spans(10, 3, overlap=2) == [(0, 6), (2, 9), (5, 10)]   # clamped at edges

	img = torch.rand(3, 60, 80, 3)
	tiles, grid, infos, count = DivideRectangle().execute(img, rows=2, cols=2)
	assert count == 4 and len(tiles) == 4 and len(infos) == 4
	assert all(tuple(t.shape) == (3, 30, 40, 3) for t in tiles)
	assert tuple(grid.shape) == (12, 30, 40, 3)            # 4 tiles x 3 frames
	# tiles are bit-exact slices, in reading order
	assert torch.equal(tiles[0], img[:, 0:30, 0:40, :])
	assert torch.equal(tiles[3], img[:, 30:60, 40:80, :])
	# crop_infos put every tile back exactly -> the original frame
	rebuilt = torch.zeros_like(img)
	for t, info in zip(tiles, infos):
		it = info["items"][0]
		rebuilt[:, it["y0"]:it["y0"] + it["h"], it["x0"]:it["x0"] + it["w"], :] = t
	assert torch.equal(rebuilt, img), "tiles must reassemble to the original"


def test_paste_back_reassembles_a_tiling_from_per_tile_crop_infos():
	"""Divide · Rectangle -> process each tile -> Paste Back = the frame again."""
	from tinode.nodes.div.rectangle import DivideRectangle
	from tinode.nodes.image.paste_back import MaskCropPasteBack

	img = torch.rand(1, 60, 80, 3)
	tiles, _grid, infos, count = DivideRectangle().execute(img, rows=2, cols=2)
	assert count == 4
	# stand in for the img2img pass: each tile comes back a different flat colour
	edited = [torch.full_like(t, 0.1 * (i + 1)) for i, t in enumerate(tiles)]

	# INPUT_IS_LIST, so every input arrives wrapped exactly as the executor sends it
	out, = MaskCropPasteBack().execute([img], edited, infos, feather=[0])
	assert tuple(out.shape) == (1, 60, 80, 3)
	# every tile landed in ITS OWN quarter, not stacked in the first one
	for t, info in zip(edited, infos):
		it = info["items"][0]
		region = out[:, it["y0"]:it["y0"] + it["h"], it["x0"]:it["x0"] + it["w"], :]
		assert torch.equal(region, t)
	assert not torch.equal(out, img)

	# a single transform with per-FRAME items still means what it always meant
	single = [{"H": 60, "W": 80, "C": 3,
			   "items": [{"y0": 0, "x0": 0, "h": 30, "w": 40,
						  "oy": 0, "ox": 0, "nh": 30, "nw": 40}]}]
	one, = MaskCropPasteBack().execute([img], [edited[0]], single, feather=[0])
	assert torch.equal(one[:, 0:30, 0:40, :], edited[0])
	assert torch.equal(one[:, 30:60, 40:80, :], img[:, 30:60, 40:80, :])


def test_paste_back_scales_a_tile_generated_at_another_resolution():
	"""Tiles are usually upscaled to the model's working size before generation."""
	from tinode.nodes.div.rectangle import DivideRectangle
	from tinode.nodes.image.paste_back import MaskCropPasteBack

	img = torch.rand(1, 60, 80, 3)
	tiles, _g, infos, _c = DivideRectangle().execute(img, rows=2, cols=2)
	assert tuple(tiles[0].shape) == (1, 30, 40, 3)
	# each tile comes back 4x bigger, as a flat colour so resampling is exact
	big = [torch.full((1, 120, 160, 3), 0.1 * (i + 1)) for i in range(4)]

	out, = MaskCropPasteBack().execute([img], big, infos, feather=[0])
	assert tuple(out.shape) == (1, 60, 80, 3)
	for t, info in zip(big, infos):
		it = info["items"][0]
		region = out[:, it["y0"]:it["y0"] + it["h"], it["x0"]:it["x0"] + it["w"], :]
		assert torch.allclose(region, t[:, :1, :1, :].expand_as(region), atol=1e-5)

	# a same-size crop is still pasted bit-exactly, with no resampling
	same, = MaskCropPasteBack().execute([img], list(tiles), infos, feather=[0])
	assert torch.equal(same, img)


def test_divide_rectangle_uneven_and_overlap():
	from tinode.nodes.div.rectangle import DivideRectangle
	img = torch.rand(1, 10, 10, 3)
	tiles, _g, infos, count = DivideRectangle().execute(img, rows=3, cols=1)
	assert count == 3 and [t.shape[1] for t in tiles] == [4, 3, 3]   # remainder first
	# overlapping tiles are bigger but still bit-exact slices of the source
	tiles, _g, infos, _c = DivideRectangle().execute(img, rows=2, cols=1, overlap=2)
	it = infos[1]["items"][0]
	assert torch.equal(tiles[1], img[:, it["y0"]:it["y0"] + it["h"], :, :])
	# more parts than pixels never yields an empty tile
	tiles, _g, _i, count = DivideRectangle().execute(torch.rand(1, 2, 2, 3), rows=8, cols=8)
	assert count == 4 and all(t.shape[1] > 0 and t.shape[2] > 0 for t in tiles)


def test_parse_frame_ranges_js():
	# The propagate modal's range parser, exercised in node (same file the
	# browser loads), since a wrong range silently stamps the wrong frames.
	import re
	import shutil
	import subprocess

	if not shutil.which("node"):
		print("    (skipped: node not installed)", end="")
		return
	root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
	src = open(os.path.join(root, "web", "add_segments.js")).read()
	m = re.search(r"export (function parseFrameRanges\([\s\S]*?\n\})", src)
	assert m, "could not find parseFrameRanges in web/add_segments.js"
	script = m.group(1) + """
const out = [
  parseFrameRanges("8-16, 32-48", 100),
  parseFrameRanges("5", 100),
  parseFrameRanges("3-5", 100, 4),        // the source frame is excluded
  parseFrameRanges("16-8", 100),          // reversed reads the same
  parseFrameRanges("0-999", 5),           // clamped to the clip
  parseFrameRanges("junk, -2, 7", 100),   // junk ignored
  parseFrameRanges("", 100),
];
console.log(JSON.stringify(out));
"""
	got = json.loads(subprocess.run(["node", "-e", script], capture_output=True,
									text=True, check=True).stdout)
	assert got[0] == list(range(8, 17)) + list(range(32, 49))
	assert got[1] == [5]
	assert got[2] == [3, 5]
	assert got[3] == list(range(8, 17))
	assert got[4] == [0, 1, 2, 3, 4]
	assert got[5] == [7]         # "junk" and a bare "-2" are dropped, not guessed at
	assert got[6] == []


def test_show_text_lines_numbers_and_flattens():
	from tinode.nodes.data.show_text import ShowTextLines
	n = ShowTextLines()
	# INPUT_IS_LIST wraps every socket; a list socket becomes one line per element
	r = n.execute(text_1=["hello"], text_2=[["a", "b"]], text_3=[None])
	assert r["result"][0] == "hello\na\nb"
	assert r["ui"]["text"][0] == "1. hello\n2. a\n3. b"
	# nothing wired says so rather than showing an empty box
	assert "nothing wired" in ShowTextLines().execute()["ui"]["text"][0]


def test_crop_divisible_trims_both_sides():
	from tinode.nodes.div.divisible import CropDivisible, trim_amounts
	# the remainder is split across BOTH sides, odd extra to the end
	assert trim_amounts(1080, 8) == (0, 0)               # already divisible
	assert trim_amounts(1000, 8) == (0, 0)
	assert trim_amounts(1006, 8) == (3, 3)               # 6 off -> 3 + 3
	assert trim_amounts(1005, 8) == (2, 3)               # 5 off -> centred, extra last
	assert trim_amounts(1005, 8, "center_bias_start") == (3, 2)
	assert trim_amounts(1005, 8, "start") == (0, 5)      # keep the top/left edge
	assert trim_amounts(1005, 8, "end") == (5, 0)

	img = torch.rand(2, 1005, 1006, 3)
	mask = torch.rand(2, 1005, 1006)
	out, m, info, w, h = CropDivisible().execute(
		img, divisible_width=8, divisible_height=8, mask=mask)
	assert (w, h) == (1000, 1000) and w % 8 == 0 and h % 8 == 0
	assert tuple(out.shape) == (2, 1000, 1000, 3) and tuple(m.shape) == (2, 1000, 1000)
	# bit-exact slice, taken from both sides (3 off left, 2 off top)
	assert torch.equal(out, img[:, 2:1002, 3:1003, :])
	assert torch.equal(m, mask[:, 2:1002, 3:1003])
	# width and height divide independently
	_o, _m, _i, w2, h2 = CropDivisible().execute(img, divisible_width=100, divisible_height=7)
	assert w2 % 100 == 0 and h2 % 7 == 0
	# crop_info puts it back in the original frame
	it = info["items"][0]
	assert (it["y0"], it["x0"], it["h"], it["w"]) == (2, 3, 1000, 1000)
	assert info["H"] == 1005 and info["W"] == 1006
	# a divisor bigger than the frame is a readable error, not an empty tensor
	try:
		CropDivisible().execute(torch.rand(1, 10, 10, 3), divisible_width=64)
	except RuntimeError as exc:
		assert "cannot be made divisible" in str(exc)
	else:
		raise AssertionError("expected an error when the whole frame would be trimmed")


def test_divide_rectangle_by_size():
	from tinode.nodes.div.rectangle import DivideRectangle, parts_for_size, size_spans
	# even mode: the count whose EVEN division lands closest to the target
	assert parts_for_size(1000, 300) == 3            # 3x333, not 3x300 + a sliver
	assert size_spans(1000, 300) == [(0, 334), (334, 667), (667, 1000)]
	# exact mode: tiles at exactly the target, remainder last
	assert parts_for_size(1000, 300, exact=True) == 4
	assert size_spans(1000, 300, exact=True) == [(0, 300), (300, 600), (600, 900), (900, 1000)]

	img = torch.rand(1, 1080, 1920, 3)
	tiles, _g, infos, count = DivideRectangle().execute(
		img, by="size", tile_width=512, tile_height=512)
	assert count == 4 * 2                            # 1920/512 -> 4 cols, 1080/512 -> 2 rows
	assert all(abs(t.shape[2] - 480) <= 1 for t in tiles)     # uniform, ~512
	# exact_size gives literal 512s plus the remainder
	tiles, _g, _i, _c = DivideRectangle().execute(
		img, by="size", tile_width=512, tile_height=512, exact_size=True)
	assert tiles[0].shape[2] == 512 and tiles[0].shape[1] == 512
	# and by=size still reassembles exactly
	tiles, _g, infos, _c = DivideRectangle().execute(
		torch.rand(2, 100, 100, 3), by="size", tile_width=30, tile_height=30)
	src = torch.rand(2, 100, 100, 3)
	tiles, _g, infos, _c = DivideRectangle().execute(src, by="size", tile_width=30, tile_height=30)
	rebuilt = torch.zeros_like(src)
	for t, info in zip(tiles, infos):
		it = info["items"][0]
		rebuilt[:, it["y0"]:it["y0"] + it["h"], it["x0"]:it["x0"] + it["w"], :] = t
	assert torch.equal(rebuilt, src)


def test_crop_by_info_reproduces_the_manual_crop():
	img = torch.rand(4, 120, 160, 3)
	crop, info = BboxCropManual().execute(img, x=20, y=10, width=64, height=48)
	(back,) = CropByInfo().execute(img, info)
	assert torch.equal(back, crop), "Crop By Info must reproduce the crop exactly"


def test_paste_back_full_mask_is_bit_exact_identity():
	# Crop then paste the SAME pixels back through a full mask: the frame must
	# be bit-for-bit the original — the "no quality loss" guarantee.
	img = torch.rand(3, 100, 128, 3)
	crop, info = BboxCropManual().execute(img, x=16, y=8, width=48, height=40)
	full_mask = torch.ones(3, 40, 48)
	(out,) = MaskCropPasteBack().execute(img, crop, info, masks=full_mask, feather=0)
	assert torch.equal(out, img), "identity paste-back must not alter any pixel"


def test_paste_back_leaves_outside_mask_untouched():
	# A generated crop replaces only the masked region; everything else stays
	# bit-exact original, so there is never a crop-rectangle seam.
	img = torch.rand(1, 80, 80, 3)
	crop, info = BboxCropManual().execute(img, x=10, y=10, width=40, height=40)
	gen = torch.zeros_like(crop)                      # "removed" fill
	mask = torch.zeros(1, 40, 40)
	mask[:, 8:32, 8:32] = 1.0                         # only an inner square
	(out,) = MaskCropPasteBack().execute(img, gen, info, masks=mask, feather=0)
	# outside the crop entirely: identical
	assert torch.equal(out[:, :10, :, :], img[:, :10, :, :])
	# inside the crop but outside the mask: still identical
	assert torch.equal(out[0, 10:18, 10:50, :], img[0, 10:18, 10:50, :])
	# inside the mask: replaced by the fill
	assert torch.equal(out[0, 18:42, 18:42, :], gen[0, 8:32, 8:32, :])


def test_paste_back_gaussian_feather_runs_and_blends():
	img = torch.rand(1, 60, 60, 3)
	crop, info = BboxCropManual().execute(img, x=10, y=10, width=32, height=32)
	gen = torch.zeros_like(crop)
	mask = torch.zeros(1, 32, 32)
	mask[:, 8:24, 8:24] = 1.0
	(g,) = MaskCropPasteBack().execute(img, gen, info, masks=mask, feather=4,
									   feather_mode="gaussian")
	(b,) = MaskCropPasteBack().execute(img, gen, info, masks=mask, feather=4,
									   feather_mode="box")
	assert g.shape == img.shape and b.shape == img.shape
	assert torch.isfinite(g).all()
	# feathering must actually soften — the hard-edged (feather 0) result differs
	(hard,) = MaskCropPasteBack().execute(img, gen, info, masks=mask, feather=0)
	assert not torch.equal(g, hard)


def test_crop_by_info_rejects_rescaled_xform():
	info = {"H": 100, "W": 100, "C": 3, "items": [
		{"y0": 0, "x0": 0, "h": 20, "w": 20, "oy": 4, "ox": 4, "nh": 12, "nw": 12}]}
	try:
		CropByInfo().execute(torch.rand(1, 100, 100, 3), info)
	except RuntimeError as exc:
		assert "rescaled" in str(exc)
	else:
		raise AssertionError("expected Crop By Info to reject a rescaled crop_info")


def test_mask_store_per_crop_roundtrip():
	import numpy as np
	from PIL import Image

	with tempfile.TemporaryDirectory() as root:
		stem = "CLIP A"
		# two crops of the same clip
		for ci in (0, 1):
			idir = _mstore.item_dir(root, stem, ci)
			mdir = _mstore.mask_dir(idir)
			os.makedirs(mdir)
			for i in range(3):
				a = np.zeros((8, 12), dtype=np.uint8)
				a[i:i + 2, :] = 255
				Image.fromarray(a, mode="L").save(os.path.join(mdir, _mstore.MASK_PATTERN % i))
			_mstore.write_manifest(idir, {
				"tinode_mask_manifest": 2, "stem": stem, "crop_index": ci,
				"source_path": "", "frame_count": 3,
				"mask_subfolder": "mask", "mask_pattern": _mstore.MASK_PATTERN,
				"crop_info": {"H": 20, "W": 30, "C": 3, "items": [None]},
			})
		items = _mstore.scan_items(root)
		assert len(items) == 2 and [it["crop_index"] for it in items] == [0, 1]
		m = _mstore.load_mask_sequence(items[0]["mask_dir"], _mstore.MASK_PATTERN, 3)
		assert tuple(m.shape) == (3, 8, 12)
		assert m.max() == 1.0 and m.min() == 0.0     # exact 8-bit round trip


def test_frame_unpad_trims_a_vae_that_returns_extra_frames():
	from tinode.nodes.image.frame_pad import FramePad, FrameUnpad
	# Real case: 85 source frames, pad 8 -> 93 in, but the VAE decoded
	# ratio x latents = 96 out, leaving 88 after unpad — 3 frames that would
	# spill into the NEXT chunk on paste-back.
	src = torch.rand(77, 4, 6, 3)
	padded, _m, pc, total = FramePad().execute(src)
	assert total % 8 == 0 and _void_output_len(total) == total
	out, n = FrameUnpad().execute(torch.rand(total, 4, 6, 3), pc)
	assert n == 77, "a multiple of 8 comes back unchanged, so the unpad is exact"
	# expected_frames stays as a belt-and-braces guard for a model that still
	# hands back extra frames
	out, n = FrameUnpad().execute(torch.rand(99, 4, 6, 3), pc, expected_frames=77)
	assert n == 77 and out.shape[0] == 77
	# a model returning too FEW frames is an error, not something to pad over
	try:
		FrameUnpad().execute(torch.rand(60, 4, 6, 3), pc, expected_frames=85)
	except RuntimeError as exc:
		assert "FEWER" in str(exc)
	else:
		raise AssertionError("expected an error when frames are missing")


def _void_output_len(length):
	"""What comfy_extras/nodes_void.py actually returns for a given length."""
	T, P = 4, 2
	latent_t = ((length - 1) // T) + 1
	if latent_t % P:                                  # rounded down to even
		latent_t = max(P, (latent_t // P) * P)
	return latent_t * T                               # decoder emits latent_t * 4


def test_paste_back_accepts_a_zero_offset_chunk():
	# Chunk 0's frame_start is a perfectly valid 0. Testing the offset for
	# truthiness made it look like "no offset given" and rejected the paste.
	img = torch.rand(172, 60, 80, 3)
	crop, info = _unwrap(BboxCropManual().execute(img[:85], x=5, y=5, width=32, height=32))
	gen = torch.zeros_like(crop)
	(out,) = MaskCropPasteBack().execute(img, gen, info, masks=torch.ones(85, 32, 32),
										 feather=0, frame_offset=0)
	assert out.shape[0] == 172
	assert torch.equal(out[85:], img[85:]), "frames past the chunk stay untouched"
	assert torch.equal(out[:85, 5:37, 5:37, :], gen)
	# more crops than frames is still a real error
	try:
		MaskCropPasteBack().execute(img[:10], gen, info, masks=torch.ones(85, 32, 32))
	except RuntimeError as exc:
		assert "more crops than frames" in str(exc)
	else:
		raise AssertionError("expected an error when crops outnumber source frames")


def test_format_outputs_can_drive_combo_widgets():
	# Save Video's format / pix_fmt / color_range / colorspace are COMBO widget
	# inputs, and the frontend refuses a STRING -> COMBO link. These outputs are
	# declared wildcard so the connection is accepted.
	from tinode.nodes.image.match_format import MatchSourceFormat
	from tinode.nodes.image.video_settings import LoadVideoSettings
	for cls, combo_slots in ((MatchSourceFormat, (0, 2, 3, 4)),
							 (LoadVideoSettings, (0, 2, 3, 4))):
		for i in combo_slots:
			assert cls.RETURN_TYPES[i] == "COMBO", (cls.__name__, i)
			assert cls.RETURN_TYPES[i] == "IMAGE", (cls.__name__, i)   # wildcard: equals anything


def test_match_crop_color_removes_a_cast_measured_outside_the_mask():
	from tinode.nodes.image.match_color import MatchCropColor
	ref = torch.rand(4, 30, 40, 3) * 0.6 + 0.2
	mask = torch.zeros(4, 30, 40)
	mask[:, 10:20, 10:20] = 1.0                      # the inpainted region
	drift = torch.tensor([-0.02, 0.0, -0.015])       # what the model got wrong
	tgt = (ref + drift).clamp(0, 1)
	tgt[:, 10:20, 10:20, :] = 0.0                    # the fill itself differs wildly

	un = mask <= 0.5
	before = (tgt - ref)[un].mean(0)
	assert before.abs().max() > 0.01                 # there IS a cast to remove

	out, report = MatchCropColor().execute(ref, tgt, mask=mask, method="mean")
	after = (out - ref)[un].mean(0)
	assert after.abs().max() < 1e-5, "the cast must be gone outside the mask"
	assert "offset(RGB)" in report

	# the masked region must NOT be used as reference — it is meant to differ, and
	# folding it in would drag the correction toward the fill
	out2, _ = MatchCropColor().execute(ref, tgt, mask=None, method="mean")
	assert (out2 - ref)[un].mean(0).abs().max() > (out - ref)[un].mean(0).abs().max()

	# strength 0 and method none are true pass-throughs, for A/B
	same, rep = MatchCropColor().execute(ref, tgt, mask=mask, method="none")
	assert torch.equal(same, tgt) and rep == "off"


def test_match_crop_color_per_frame_tracks_drift_that_changes_over_time():
	from tinode.nodes.image.match_color import MatchCropColor
	n, h, w = 6, 32, 40
	ref = torch.rand(n, h, w, 3) * 0.4 + 0.3
	# drift that GROWS through the clip — measured on VOID it swung 6-7 levels
	# from first frame to last, which is why one clip-wide correction cannot work
	tgt = ref.clone()
	for i in range(n):
		tgt[i] = (ref[i] + (i - n / 2) * 0.01).clamp(0, 1)
	mask = torch.zeros(n, h, w)
	mask[:, 12:18, 16:22] = 1.0
	tgt[:, 12:18, 16:22, :] = 0.0
	un = mask <= 0.5

	def worst(x):
		return max(((x[i] - ref[i])[un[i]].mean(0)).abs().max().item() for i in range(n))

	before = worst(tgt)
	whole, _ = MatchCropColor().execute(ref, tgt, mask=mask, method="low_freq",
										temporal="whole_clip", sigma=5.0)
	each, _ = MatchCropColor().execute(ref, tgt, mask=mask, method="low_freq",
									   temporal="per_frame", sigma=5.0)
	# one correction for the clip is right on average and wrong at the ends
	assert worst(whole) > before * 0.4, "whole_clip cannot follow a changing drift"
	assert worst(each) < before * 0.05, "per_frame must track it"


def test_match_crop_color_low_freq_removes_a_gradient_a_mean_cannot():
	from tinode.nodes.image.match_color import MatchCropColor
	h, w = 48, 64
	ref = torch.rand(2, h, w, 3) * 0.4 + 0.3
	# a drift that VARIES across the crop — this is what a single offset misses,
	# and what was still visible after the mean correction on the real data
	ramp = torch.linspace(-0.05, 0.05, w).view(1, 1, w, 1)
	tgt = (ref + ramp).clamp(0, 1)
	mask = torch.zeros(2, h, w)
	mask[:, 20:28, 28:36] = 1.0
	tgt[:, 20:28, 28:36, :] = 0.0                    # the fill itself
	un = mask <= 0.5

	def gradient_span(x):
		"""Left-vs-right mean error: how much of the ramp survives."""
		e = (x - ref)
		left = e[:, :, : w // 4][un[:, :, : w // 4]].mean()
		right = e[:, :, -w // 4 :][un[:, :, -w // 4 :]].mean()
		return (right - left).abs().item()

	before = gradient_span(tgt)
	mean_only, _ = MatchCropColor().execute(ref, tgt, mask=mask, method="mean")
	low, _ = MatchCropColor().execute(ref, tgt, mask=mask, method="low_freq", sigma=6.0)
	assert gradient_span(mean_only) > before * 0.8, "a single offset cannot flatten a ramp"
	assert gradient_span(low) < before * 0.2, "low_freq must flatten it"


def test_compare_videos_layouts_and_conforming():
	from tinode.nodes.image.compare_video import CompareVideos
	a = torch.rand(6, 40, 50, 3)
	b = a.clone()
	b[:, 10:20, 10:20, :] = 0.0                      # a "removed" patch

	wide, _ = CompareVideos().execute(a, b, layout="side_by_side", labels=False)
	assert wide.shape == (6, 40, 100, 3)
	assert torch.equal(wide[:, :, :50, :], a) and torch.equal(wide[:, :, 50:, :], b)

	tall, _ = CompareVideos().execute(a, b, layout="stacked", labels=False)
	assert tall.shape == (6, 80, 50, 3)

	# difference must be BLACK where nothing changed — that is the whole point,
	# it is what proves an untouched area really was untouched
	diff, _ = CompareVideos().execute(a, b, layout="difference", difference_gain=8.0)
	assert diff.shape == a.shape
	assert diff[:, 25:, 25:, :].max() == 0.0
	assert diff[:, 10:20, 10:20, :].max() > 0.0

	# a mismatched pair is conformed and REPORTED, not silently accepted
	out, notes = CompareVideos().execute(a, torch.rand(4, 20, 25, 3), layout="side_by_side",
										 labels=False)
	assert out.shape == (6, 40, 100, 3)
	assert "resized" in notes and "frames" in notes


def test_frame_pad_matches_voids_real_length_rule():
	from tinode.nodes.image.frame_pad import frame_pad_count
	# The observed failures, reproduced from VOID's own formula.
	assert _void_output_len(93) == 96                 # +3, what we measured
	assert _void_output_len(97) == 96                 # -1, what we measured
	# Output equals input only for multiples of 8 (temporal_compression 4 x
	# patch_size_t 2), which is what the default pad now targets.
	assert [L for L in range(1, 100) if _void_output_len(L) == L] == list(range(8, 100, 8))
	for L in range(1, 200):
		p = frame_pad_count(L, 8, 8)                  # plus_one defaults False
		total = L + p
		assert 8 <= p <= 15, (L, p)
		assert total % 8 == 0, (L, p)
		assert _void_output_len(total) == total, (L, total)   # round trip is exact
	# plus_one stays for a model that really wants ratio*n + 1
	for L in range(1, 200):
		assert (L + frame_pad_count(L, 8, 4, True) - 1) % 4 == 0


def test_frame_pad_context_continues_from_the_previous_chunk():
	from tinode.nodes.image.frame_pad import FramePad, FrameUnpad
	chunk = torch.rand(30, 8, 10, 3)
	mask = torch.ones(30, 8, 10)                     # object present all through
	prev = torch.rand(40, 8, 10, 3)                  # previous chunk's finished frames

	# without context: the head is frame 0 held, and its mask is held too
	pa, ma, ia, _ = FramePad().execute(chunk, mask=mask)
	h = ia["head"]
	assert torch.equal(pa[h - 1], chunk[0]) and ma[:h].min() == 1.0

	# with context: the head is the previous chunk's TAIL, in order, and its mask
	# is zeroed — those frames are already clean, so nothing is re-removed there
	pb, mb, ib, _ = FramePad().execute(chunk, mask=mask, context_images=prev)
	assert ib["head"] == h
	assert torch.equal(pb[:h], prev[-h:]), "head must be the previous tail, in order"
	assert mb[:h].max() == 0.0, "context frames must not be masked for removal"
	assert torch.equal(pb[h:], chunk)                # the clip itself is untouched
	# and the round trip still lands exactly (pad_info carries the side + counts)
	back, n = FrameUnpad().execute(pb, ib)
	assert torch.equal(back, chunk) and n == 30

	# ORIGINAL preceding frames still contain the object, so their own mask must
	# be carried — zeroing it would tell the model to PRESERVE what we remove
	prev_mask = torch.ones(40, 8, 10)
	_p, mc, ic, _ = FramePad().execute(chunk, mask=mask, context_images=prev,
									   context_mask=prev_mask)
	hc = ic["head"]
	assert torch.equal(mc[:hc], prev_mask[-hc:]), "context mask must be used as given"
	assert mc[:hc].min() == 1.0
	# without a context mask the head is zeroed (the finished-frames case)
	_i, mz, iz, _ = FramePad().execute(chunk, mask=mask, context_images=prev)
	assert mz[:iz["head"]].max() == 0.0

	# a context shorter than the pad is held out, never sliced short
	short, _m, isr, _ = FramePad().execute(chunk, mask=mask, context_images=prev[:3])
	hs = isr["head"]
	assert short.shape[0] == 30 + hs and torch.equal(short[hs - 1], prev[2])

	# a mismatched crop size is a wiring error, not something to silently resize
	try:
		FramePad().execute(chunk, mask=mask, context_images=torch.rand(4, 16, 20, 3))
	except RuntimeError as exc:
		assert "SAME crop" in str(exc)
	else:
		raise AssertionError("expected an error for a differently sized context")


def test_frame_pad_rejects_a_mask_that_does_not_line_up():
	from tinode.nodes.image.frame_pad import FramePad
	img = torch.rand(20, 8, 10, 3)
	# the mask usually arrives by a different route than the frames (segment
	# editors), so a length disagreement is possible — and padding both would
	# hand the model a mask offset in time from the video.
	for bad in (torch.zeros(19, 8, 10), torch.zeros(21, 8, 10)):
		try:
			FramePad().execute(img, mask=bad)
		except RuntimeError as exc:
			assert "line up" in str(exc) or "frame(s)" in str(exc)
		else:
			raise AssertionError("expected an error for a mask of the wrong length")
	try:
		FramePad().execute(img, mask=torch.zeros(20, 16, 20))
	except RuntimeError as exc:
		assert "16x20" in str(exc)
	else:
		raise AssertionError("expected an error for a mask of the wrong size")


def test_frame_pad_round_trips_at_a_valid_length():
	from tinode.nodes.image.frame_pad import FramePad, FrameUnpad
	img = torch.rand(30, 8, 10, 3)
	mask = torch.rand(30, 8, 10)
	padded, pmask, info, total = FramePad().execute(img, mask=mask)
	h = info["head"]
	assert total == 30 + h and total % 8 == 0
	assert padded.shape[0] == total and pmask.shape[0] == total
	assert torch.equal(padded[0], img[0]) and torch.equal(padded[h], img[0])
	assert _void_output_len(total) == total          # VOID keeps this length
	back, n = FrameUnpad().execute(padded, info)
	assert torch.equal(back, img) and n == 30
	# node round trip: pad then unpad restores the exact clip
	img = torch.rand(30, 8, 10, 3)
	mask = torch.rand(30, 8, 10)
	padded, pmask, info, total = FramePad().execute(img, mask=mask)
	h = info["head"]
	assert total == 30 + h and total % 8 == 0
	assert padded.shape[0] == total and pmask.shape[0] == total
	# prepended frames are copies of frame 0
	assert torch.equal(padded[0], img[0]) and torch.equal(padded[h], img[0])
	back, n = FrameUnpad().execute(padded, info)
	assert torch.equal(back, img) and n == img.shape[0], \
		"pad then unpad must restore the original clip"


def test_crop_sequence_roundtrip_8bit_and_16bit():
	# The lossless RGB export/import both directions, at each bit depth.
	img = torch.rand(3, 12, 16, 3)
	for bits, tol in ((8, 1.0 / 255), (16, 1.0 / 65535)):
		with tempfile.TemporaryDirectory() as d:
			n = _mstore.save_rgb_sequence(img, d, _mstore.CROP_PATTERN, bits)
			assert n == 3
			back = _mstore.load_rgb_sequence(d, _mstore.CROP_PATTERN, 3)
			assert tuple(back.shape) == (3, 12, 16, 3)
			# quantisation to `bits` is the only difference; within one step.
			assert (back - img).abs().max().item() <= tol + 1e-6, f"{bits}-bit drift"


def test_video_source_path_extracts_stem():
	class FakeVideo:
		def get_stream_source(self):
			return "/x/y/dicaire/DICAIRE T 01 pour test IA.mp4"

	assert video_source_path(FakeVideo()) == "/x/y/dicaire/DICAIRE T 01 pour test IA.mp4"
	assert video_source_path(object()) is None       # not file-backed


# --------------------------------------------------------- video preview (RAM)
def _gradient(n=4, h=32, w=48):
	"""A smooth ramp, continuous in every direction.

	Noise would measure DCT loss and a 255->0 cliff would measure chroma
	subsampling — neither of which is what these tests are about.
	"""
	import numpy as np

	yy, xx = np.mgrid[0:h, 0:w]
	f = np.stack([xx * 255.0 / (w - 1), yy * 255.0 / (h - 1),
				  (xx + yy) * 255.0 / (w + h - 2)], -1)
	# Frames differ by a gain, not by a roll: rolling wraps 255 next to 0 and the
	# resulting cliff measures chroma subsampling instead of what is under test.
	return np.stack([(f * (0.55 + 0.1 * i)).round().astype(np.uint8) for i in range(n)])


def _decode_rgb(data, ext, w, h, depth=8):
	"""Decode encoded bytes back to an array, to check what a player would see."""
	import subprocess

	import numpy as np

	with tempfile.TemporaryDirectory() as d:
		p = os.path.join(d, "c." + ext)
		with open(p, "wb") as fh:
			fh.write(data)
		out = subprocess.run(
			[ffmpeg_exe(), "-v", "error", "-i", p, "-f", "rawvideo",
			 "-pix_fmt", "rgb48le" if depth == 16 else "rgb24", "pipe:1"],
			capture_output=True)
		assert out.returncode == 0, out.stderr.decode()[-400:]
		return np.frombuffer(out.stdout, "<u2" if depth == 16 else np.uint8).reshape(-1, h, w, 3)


def test_colour_flags_convert_as_well_as_tag():
	"""The bug: -colorspace / -color_range only WRITE TAGS.

	ffmpeg's implicit rgb->yuv conversion is BT.601 limited whatever you tag, so
	Save Video's own defaults (bt709 + tv) converted as 601 and labelled 709.
	Players undid a matrix that was never applied: measured up to 32/255 off,
	mean 5.9, on the node that exists to keep a grade intact. The filter has to be
	emitted alongside the tags — and only when something was actually asked for.
	"""
	filt, tags = colour_flags("tv", "bt709")
	assert filt == ["-vf", "scale=in_range=full:out_color_matrix=bt709:out_range=limited"]
	assert tags == ["-colorspace", "bt709", "-color_primaries", "bt709",
					"-color_trc", "bt709", "-color_range", "tv"]

	assert colour_flags("pc", "bt709")[0][1].endswith("out_range=full")
	# Nothing requested -> nothing imposed; ffmpeg's own default is left alone.
	assert colour_flags("unspecified", "unspecified") == ([], [])
	# One without the other still converts for the one that was asked for.
	assert colour_flags("unspecified", "bt709")[0] == ["-vf", "scale=in_range=full:out_color_matrix=bt709"]
	assert colour_flags("tv", "unspecified")[0] == ["-vf", "scale=in_range=full:out_range=limited"]


def test_encode_round_trips_within_its_own_colour_tags():
	"""Save Video's output must decode back to what went in. Regression for the above."""
	if ffmpeg_exe() is None:
		print("    (skipped: no ffmpeg)")
		return
	import numpy as np

	src = _gradient()
	t = torch.from_numpy(src.astype("float32") / 255.0)
	for cr, cs, pf, tol in (("tv", "bt709", "yuv444p", 2), ("pc", "bt709", "yuv444p", 1),
							("tv", "bt709", "yuv420p", 10)):
		with tempfile.TemporaryDirectory() as d:
			out = os.path.join(d, "c.mp4")
			_vio_encode(t, out, fps=24.0, codec="libx264", crf=0, pix_fmt=pf,
						color_range=cr, colorspace=cs)
			with open(out, "rb") as fh:
				back = _decode_rgb(fh.read(), "mp4", src.shape[2], src.shape[1])
		err = int(np.abs(back.astype(int) - src.astype(int)).max())
		# 4:4:4 leaves only matrix rounding; 4:2:0 also subsamples chroma, which is
		# content-dependent — the tolerances are loose enough to be about the bug
		# (which measured 32-35) and not about the test image.
		assert err <= tol, f"{cs}/{cr}/{pf} came back {err} levels out (tolerance {tol})"


def test_video_preview_precision_auto_keeps_what_the_source_carries():
	"""auto must not quantise a >8-bit clip, and must not double the RAM of an 8-bit one."""
	import numpy as np

	eight = torch.from_numpy(_gradient().astype("float32") / 255.0)
	raw, n, h, w, depth = _pstore.pack_frames(eight, "auto")
	assert depth == 8 and len(raw) == n * h * w * 3
	assert np.frombuffer(raw, np.uint8).reshape(n, h, w, 3).tobytes() == _gradient().tobytes()

	deep = eight + 1.0 / 1000.0            # off the 1/255 grid: real extra precision
	raw16, _, _, _, d16 = _pstore.pack_frames(deep, "auto")
	assert d16 == 16 and len(raw16) == n * h * w * 3 * 2
	# ...and the override is obeyed in both directions.
	assert _pstore.pack_frames(deep, "8-bit")[4] == 8
	assert _pstore.pack_frames(eight, "16-bit")[4] == 16


def test_video_preview_lossless_downloads_are_bit_exact():
	"""CREATE VIDEO's lossless options must match the frames the graph produced.

	The preview you watch is a lossy proxy; the download re-encodes from the
	untouched master. ffv1 and the png zip claim bit-exactness, so prove it.
	"""
	if ffmpeg_exe() is None:
		print("    (skipped: no ffmpeg)")
		return
	import io
	import zipfile

	import numpy as np
	from PIL import Image

	src = _gradient()
	raw, n, h, w, depth = _pstore.pack_frames(
		torch.from_numpy(src.astype("float32") / 255.0), "auto")
	sess = {"raw": raw, "n": n, "h": h, "w": w, "depth": depth, "fps": 24.0}

	data, ext, _ = _pstore.render(sess, "ffv1")
	assert np.array_equal(_decode_rgb(data, ext, w, h), src), "ffv1 is not bit-exact"

	data, ext, mime = _pstore.render(sess, "png")
	assert (ext, mime) == ("zip", "application/zip")
	with zipfile.ZipFile(io.BytesIO(data)) as zf:
		names = sorted(zf.namelist())
		frames = np.stack([np.asarray(Image.open(io.BytesIO(zf.read(nm))).convert("RGB"))
						   for nm in names])
	assert names == [f"{i:05d}.png" for i in range(n)]
	assert np.array_equal(frames, src), "png sequence is not bit-exact"


def test_video_preview_holds_the_clip_only_in_ram():
	"""The whole point: previewing must leave nothing behind.

	The mp4/mov encoders need a seekable output, so they use a private tempfile
	dir — which must be gone by the time render() returns, whatever happened.
	"""
	if ffmpeg_exe() is None:
		print("    (skipped: no ffmpeg)")
		return
	scratch = lambda: {p for p in os.listdir(tempfile.gettempdir()) if p.startswith("tinode_vp_")}
	before = scratch()

	clip = torch.from_numpy(_gradient().astype("float32") / 255.0)
	res = VideoPreview().execute(clip, 24.0, "360p", "auto", 30, unique_id="ram-test")
	ui = res["ui"]["ti_vpreview"][0]
	try:
		assert res["result"][0] is clip, "the IMAGE passthrough must not copy or convert"
		assert ui["frames"] == 4 and ui["depth"] == 8
		for slug in _pstore.FORMATS:
			assert len(_pstore.render(_pstore.get(ui["id"]), slug)[0]) > 0
		assert scratch() == before, "a scratch dir survived render()"
	finally:
		_pstore.drop(ui["id"])
	assert _pstore.get(ui["id"]) is None


def test_video_preview_store_bounds_itself():
	"""A RAM cache turns into a leak by holding every run, or by evicting itself."""
	raw = b"\0" * 6144
	a = _pstore.put(raw, 2, 32, 32, 8, 24.0, owner="n1")
	b = _pstore.put(raw, 2, 32, 32, 8, 24.0, owner="n1")
	try:
		# Re-queuing one node swaps its clip instead of stacking a second copy.
		assert _pstore.get(a) is None and _pstore.get(b) is not None
		# ...but a different node keeps its own.
		c = _pstore.put(raw, 2, 32, 32, 8, 24.0, owner="n2")
		assert _pstore.get(b) is not None and _pstore.get(c) is not None
		assert _pstore.drop(c) and not _pstore.drop(c)

		# Expiry is swept lazily, on the next put.
		d = _pstore.put(raw, 2, 32, 32, 8, 24.0, owner="n3", hold=1.0)
		assert _pstore.get(d) is not None, "a clip must survive the sweep its own arrival triggers"
		_pstore._SESSIONS[d]["touched"] -= 10
		_pstore.put(raw, 2, 32, 32, 8, 24.0, owner="n4")
		assert _pstore.get(d) is None

		# Against a tight ceiling the newcomer evicts the others, never itself —
		# otherwise put() hands back an id that is already dead.
		cap = _pstore.MAX_BYTES
		try:
			_pstore.MAX_BYTES = len(raw) + 16
			tight = _pstore.put(raw, 2, 32, 32, 8, 24.0, owner="n5")
			assert _pstore.get(tight) is not None
			_pstore.MAX_BYTES = len(raw) // 2
			try:
				_pstore.put(raw, 2, 32, 32, 8, 24.0, owner="n6")
				raise AssertionError("a clip over the ceiling must raise, not vanish")
			except RuntimeError as exc:
				assert "ceiling" in str(exc)
		finally:
			_pstore.MAX_BYTES = cap
	finally:
		for sid in list(_pstore._SESSIONS):
			_pstore.drop(sid)


def _tone(seconds, rate=48000, channels=2):
	"""A stereo test track, as ComfyUI hands one over."""
	import numpy as np

	t = np.arange(int(seconds * rate), dtype=np.float32) / rate
	w = np.stack([np.sin(2 * np.pi * 440 * t) * 0.5,
				  np.sin(2 * np.pi * 660 * t) * 0.5][:channels])
	return {"waveform": torch.from_numpy(w).unsqueeze(0), "sample_rate": rate}


def _stream_counts(data, ext, w, h):
	"""Decode both streams and COUNT them.

	Container duration fields are not comparable across muxers — webm often has
	no per-stream duration and matroska rounds up to the last block — and the
	question is exactly how many frames and samples came out.
	"""
	import json
	import subprocess

	with tempfile.TemporaryDirectory() as d:
		p = os.path.join(d, "c." + ext)
		with open(p, "wb") as fh:
			fh.write(data)
		meta = json.loads(subprocess.run(
			[ffmpeg_exe().replace("ffmpeg", "ffprobe"), "-v", "error", "-show_entries",
			 "stream=codec_type,codec_name,sample_rate,channels", "-of", "json", p],
			capture_output=True, text=True).stdout)
		got = {x["codec_type"]: x for x in meta["streams"]}
		vid = subprocess.run([ffmpeg_exe(), "-v", "error", "-i", p, "-map", "0:v:0",
							  "-f", "rawvideo", "-pix_fmt", "rgb24", "pipe:1"],
							 capture_output=True).stdout
		got["n_frames"] = len(vid) // (w * h * 3)
		got["n_samples"] = 0
		if "audio" in got:
			ch = int(got["audio"]["channels"])
			pcm = subprocess.run([ffmpeg_exe(), "-v", "error", "-i", p, "-map", "0:a:0",
								  "-f", "f32le", "-ac", str(ch), "-ar",
								  str(got["audio"]["sample_rate"]), "pipe:1"],
								 capture_output=True).stdout
			got["n_samples"] = len(pcm) // (4 * ch)
		return got


def test_pack_audio_is_exact_and_interleaved():
	"""f32le means interleaved. Getting it planar swaps the channels silently."""
	import numpy as np

	track = _tone(0.25)
	a = _pstore.pack_audio(track)
	assert (a["rate"], a["channels"], a["samples"]) == (48000, 2, 12000)
	assert len(a["raw"]) == 12000 * 2 * 4
	back = np.frombuffer(a["raw"], np.float32).reshape(-1, 2)
	assert np.array_equal(back.T, track["waveform"][0].numpy())

	# Absent, malformed and empty tracks are all "no audio", never a crash.
	assert _pstore.pack_audio(None) is None
	assert _pstore.pack_audio({}) is None
	assert _pstore.pack_audio({"waveform": torch.zeros(1, 2, 0), "sample_rate": 48000}) is None
	assert _pstore.pack_audio({"waveform": torch.zeros(1, 2, 8), "sample_rate": 0}) is None
	assert _pstore.pack_audio(_tone(0.1, channels=1))["channels"] == 1


def test_video_preview_fits_the_track_to_the_picture():
	"""Audio must come out exactly as long as the video, from either direction.

	The first attempt used `-shortest`, which cuts at the last video packet's
	timestamp — the START of the final frame — so the track landed up to one
	frame-duration short (19 ms measured) and by a different amount per
	container. The fit is done in samples now.
	"""
	if ffmpeg_exe() is None:
		print("    (skipped: no ffmpeg)")
		return
	n, fps, rate = 12, 24.0, 48000
	src = _gradient(n=n, h=32, w=48)
	clip = torch.from_numpy(src.astype("float32") / 255.0)
	want = round(n / fps * rate)
	assert _pstore.fit_samples(n, fps, {"rate": rate}) == want

	for seconds in (n / fps, n / fps * 0.4, n / fps * 2.5):
		res = VideoPreview().execute(clip, fps, _tone(seconds), "360p", "auto", 30,
									 unique_id="fit-test")
		sess = _pstore.get(res["ui"]["ti_vpreview"][0]["id"])
		try:
			for slug in ("h264", "ffv1", "prores", "vp9"):
				data, ext, _ = _pstore.render(sess, slug)
				got = _stream_counts(data, ext, 48, 32)
				drift = got["n_samples"] - want
				assert got["n_frames"] == n, f"{slug}: {got['n_frames']} frames, wanted {n}"
				# A lossy encoder may round the tail up to its own frame size; it
				# must never come out short.
				assert 0 <= drift <= 2048, f"{slug}: {drift:+d} samples off at {seconds:.2f}s in"
		finally:
			_pstore.drop(sess["id"])


def test_video_preview_lossless_formats_carry_lossless_audio():
	""""Bit-exact" has to mean the whole file, not only the picture."""
	if ffmpeg_exe() is None:
		print("    (skipped: no ffmpeg)")
		return
	import io
	import subprocess
	import zipfile

	import numpy as np

	n, fps = 12, 24.0
	src = _gradient(n=n, h=32, w=48)
	raw, _, h, w, depth = _pstore.pack_frames(torch.from_numpy(src.astype("float32") / 255.0), "auto")
	track = _pstore.pack_audio(_tone(n / fps))
	sess = {"raw": raw, "n": n, "h": h, "w": w, "depth": depth, "fps": fps, "audio": track}
	want = np.frombuffer(track["raw"], np.float32)

	data, ext, _ = _pstore.render(sess, "ffv1")
	got = _stream_counts(data, ext, w, h)
	assert got["audio"]["codec_name"] == "pcm_f32le"      # matroska carries IEEE float
	with tempfile.TemporaryDirectory() as d:
		p = os.path.join(d, "c.mkv")
		with open(p, "wb") as fh:
			fh.write(data)
		pcm = subprocess.run([ffmpeg_exe(), "-v", "error", "-i", p, "-map", "0:a:0",
							  "-f", "f32le", "-ac", "2", "-ar", "48000", "pipe:1"],
							 capture_output=True).stdout
		vid = subprocess.run([ffmpeg_exe(), "-v", "error", "-i", p, "-map", "0:v:0",
							  "-f", "rawvideo", "-pix_fmt", "rgb24", "pipe:1"],
							 capture_output=True).stdout
	assert np.array_equal(np.frombuffer(pcm, np.float32), want), "ffv1 audio is not bit-exact"
	assert np.array_equal(np.frombuffer(vid, np.uint8).reshape(-1, h, w, 3), src), \
		"adding audio disturbed the video"

	# A still sequence has no container for a track, so it rides along as a WAV.
	z, ext, _ = _pstore.render(sess, "png")
	with zipfile.ZipFile(io.BytesIO(z)) as zf:
		assert "audio.wav" in zf.namelist() and len(zf.namelist()) == n + 1
		wav = zf.read("audio.wav")
	assert wav[:4] == b"RIFF" and wav[8:12] == b"WAVE"
	assert np.array_equal(np.frombuffer(wav[wav.index(b"data") + 8:], np.float32), want)


def test_video_preview_without_audio_writes_no_audio_stream():
	"""The optional input has to stay optional — no silent track, no crash."""
	if ffmpeg_exe() is None:
		print("    (skipped: no ffmpeg)")
		return
	clip = torch.from_numpy(_gradient(n=6, h=32, w=48).astype("float32") / 255.0)
	res = VideoPreview().execute(clip, 24.0, None, "360p", "auto", 30, unique_id="silent")
	ui = res["ui"]["ti_vpreview"][0]
	try:
		assert ui["audio"] is None
		assert res["result"][1] is None, "the AUDIO passthrough must forward None untouched"
		for slug in ("h264", "ffv1", "vp9", "prores"):
			data, ext, _ = _pstore.render(_pstore.get(ui["id"]), slug)
			assert "audio" not in _stream_counts(data, ext, 48, 32), f"{slug} grew an audio stream"
	finally:
		_pstore.drop(ui["id"])


def test_image_preview_stills_are_exact_at_the_masters_depth():
	"""png/tiff must round-trip a still exactly, at 8 AND at 16 bits.

	Checked through ffmpeg, not PIL: PIL has no 48-bit RGB mode and hands back
	uint8 for a 16-bit PNG without complaining, which would make this pass while
	measuring nothing.
	"""
	if ffmpeg_exe() is None:
		print("    (skipped: no ffmpeg)")
		return
	import io
	import subprocess

	import numpy as np
	from PIL import Image

	src = _gradient(n=4, h=32, w=48)
	clip = torch.from_numpy(src.astype("float32") / 255.0)

	def read(data, ext, depth):
		with tempfile.TemporaryDirectory() as d:
			p = os.path.join(d, "x." + ext)
			with open(p, "wb") as fh:
				fh.write(data)
			out = subprocess.run([ffmpeg_exe(), "-v", "error", "-i", p, "-f", "rawvideo",
								  "-pix_fmt", "rgb48le" if depth == 16 else "rgb24", "pipe:1"],
								 capture_output=True).stdout
		return np.frombuffer(out, "<u2" if depth == 16 else np.uint8).reshape(32, 48, 3)

	raw, n, h, w, depth = _pstore.pack_frames(clip, "auto")
	sess = {"raw": raw, "n": n, "h": h, "w": w, "depth": depth, "fps": 1.0}
	assert depth == 8
	for fmt in ("png", "tiff"):
		for i in range(n):
			data, ext, mime = _pstore.render_still(sess, i, fmt)
			assert (ext, mime) == _pstore.STILL_FORMATS[fmt][:2]
			assert np.array_equal(read(data, ext, 8), src[i]), f"{fmt} frame {i} is not exact"

	deep = clip + 1.0 / 1000.0
	raw16, n16, h16, w16, d16 = _pstore.pack_frames(deep, "auto")
	s16 = {"raw": raw16, "n": n16, "h": h16, "w": w16, "depth": d16, "fps": 1.0}
	assert d16 == 16
	want = np.frombuffer(raw16, "<u2").reshape(n16, h16, w16, 3)[0]
	for fmt in ("png", "tiff"):
		data, ext, _ = _pstore.render_still(s16, 0, fmt)
		assert np.array_equal(read(data, ext, 16), want), f"16-bit {fmt} is not exact"
	png16 = _pstore.render_still(s16, 0, "png")[0]
	assert png16[24] == 16, "a 16-bit master must produce a 16-bit PNG"
	assert np.asarray(Image.open(io.BytesIO(png16))).dtype == np.uint8   # the PIL trap, pinned

	# ...and an 8-bit master must NOT be inflated to 16.
	assert _pstore.render_still(sess, 0, "png")[0][24] == 8

	# An index off either end clamps instead of slicing past the master.
	assert _pstore.frame_bytes(sess, 99)[1] == n - 1
	assert _pstore.frame_bytes(sess, -5)[1] == 0


def test_image_preview_holds_a_batch_and_zips_it():
	"""The node's own contract, plus the all-frames download."""
	if ffmpeg_exe() is None:
		print("    (skipped: no ffmpeg)")
		return
	import io
	import zipfile

	import numpy as np
	from PIL import Image

	src = _gradient(n=4, h=32, w=48)
	clip = torch.from_numpy(src.astype("float32") / 255.0)
	res = ImagePreview().execute(clip, "auto", 30, unique_id="img-test")
	ui = res["ui"]["ti_ipreview"][0]
	try:
		assert res["result"][0] is clip, "the IMAGE passthrough must not copy or convert"
		assert (ui["frames"], ui["width"], ui["height"], ui["depth"]) == (4, 48, 32, 8)
		sess = _pstore.get(ui["id"])
		assert sess is not None and sess["proxy"] == b"", "a still batch needs no video proxy"

		for fmt, exact in (("png", True), ("tiff", True), ("jpg", False)):
			z, ext, mime = _pstore.render_still_zip(sess, fmt, 95)
			assert (ext, mime) == ("zip", "application/zip")
			with zipfile.ZipFile(io.BytesIO(z)) as zf:
				names = sorted(zf.namelist())
				e = _pstore.STILL_FORMATS[fmt][0]
				assert names == [f"{i:05d}.{e}" for i in range(4)]
				if exact:
					got = np.stack([np.asarray(Image.open(io.BytesIO(zf.read(nm))))[..., :3]
									for nm in names])
					assert np.array_equal(got, src), f"{fmt} zip is not bit-exact"
	finally:
		_pstore.drop(ui["id"])
	assert _pstore.get(ui["id"]) is None


# ------------------------------------------------------- tools/strip_png_metadata
def _load_stripper():
	"""Import tools/strip_png_metadata.py, which is a script rather than a node."""
	import importlib.util

	root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
	path = os.path.join(root, "tools", "strip_metadata.py")
	spec = importlib.util.spec_from_file_location("ti_strip_meta", path)
	mod = importlib.util.module_from_spec(spec)
	spec.loader.exec_module(mod)
	return mod


def _png_with_workflow(path, meta=True, size=(24, 18)):
	import numpy as np
	from PIL import Image
	from PIL.PngImagePlugin import PngInfo

	rng = np.random.default_rng(len(path))
	arr = rng.integers(0, 256, (size[1], size[0], 3), dtype=np.uint8)
	info = PngInfo()
	if meta:
		info.add_text("prompt", '{"3":{"class_type":"KSampler"}}' * 20)
		info.add_text("workflow", '{"nodes":[{"id":1}]}' * 200)
	Image.fromarray(arr).save(path, pnginfo=info if meta else None)
	return path


def test_strip_png_metadata_removes_the_workflow_without_touching_pixels():
	"""The whole point: a shared render must not carry the graph that made it.

	And the pixels must survive exactly — this is chunk surgery, not a re-encode,
	so the IDAT bytes have to come out byte-for-byte identical. Re-saving through
	PIL would also drop the text and would silently recompress, which is why the
	tool does not do that.
	"""
	import numpy as np
	from PIL import Image

	mod = _load_stripper()
	with tempfile.TemporaryDirectory() as d:
		p = _png_with_workflow(os.path.join(d, "shot.png"))
		before = np.asarray(Image.open(p)).copy()
		idat_before = [raw for k, raw in mod.read_chunks(p) if k == b"IDAT"]
		assert set(dict(Image.open(p).text)) == {"prompt", "workflow"}

		changed, removed = mod.strip_file(p)
		assert changed and removed > 4000, f"removed only {removed} bytes"
		assert dict(Image.open(p).text) == {}
		assert np.array_equal(np.asarray(Image.open(p)), before), "pixels changed"
		assert [raw for k, raw in mod.read_chunks(p) if k == b"IDAT"] == idat_before, \
			"IDAT was rewritten — the image got recompressed"

		# Running again is a no-op, and an already-clean file is never rewritten.
		st = os.stat(p)
		assert mod.strip_file(p) == (False, 0)
		assert os.stat(p).st_mtime == st.st_mtime


def test_strip_png_metadata_walks_a_directory_and_leaves_everything_else_alone():
	"""A directory argument means every PNG under it — and nothing that isn't one."""
	mod = _load_stripper()
	with tempfile.TemporaryDirectory() as d:
		os.makedirs(os.path.join(d, "a", "b"))
		pngs = [_png_with_workflow(os.path.join(d, "top.png")),
				_png_with_workflow(os.path.join(d, "a", "mid.PNG")),   # case-insensitive
				_png_with_workflow(os.path.join(d, "a", "b", "deep.png"))]
		with open(os.path.join(d, "notes.txt"), "w") as fh:
			fh.write("not an image")
		with open(os.path.join(d, "a", "photo.jpg"), "wb") as fh:
			fh.write(b"\xff\xd8\xff\xe0 not really a jpeg either")

		found = mod.collect([d])
		assert sorted(map(os.path.basename, found)) == ["deep.png", "mid.PNG", "top.png"]

		# --flat stops at the top level.
		assert [os.path.basename(x) for x in mod.collect([d], recurse=False)] == ["top.png"]

		# A path given twice is still only visited once.
		assert len(mod.collect([pngs[0], d, pngs[0]])) == 3

		for p in pngs:
			assert mod.strip_file(p)[0]
		assert all(k not in (b"tEXt", b"zTXt", b"iTXt")
				   for p in pngs for k, _ in mod.read_chunks(p))


def test_strip_metadata_removes_a_videos_workflow_without_re_encoding():
	"""ComfyUI hides the graph in a container tag too — and it is bigger there.

	Measured on a real session: 27 KB of workflow in a PNG, 174 KB in the mp4
	from the same graph. The remux must use `-c copy`, so the encoded packets
	come out byte-identical — checked here by hashing the copied streams, which
	is what would change the instant someone "fixed" this into a re-encode.
	"""
	mod = _load_stripper()
	exe = mod.ffmpeg_exe()
	if exe is None:
		print("    (skipped: no ffmpeg)")
		return
	import subprocess

	def stream_md5(path):
		out = subprocess.run([exe, "-v", "error", "-i", path, "-map", "0", "-c", "copy",
							  "-f", "md5", "-"], capture_output=True, text=True)
		return out.stdout.strip()

	with tempfile.TemporaryDirectory() as d:
		clip = os.path.join(d, "clip.mp4")
		graph = '{"nodes":[{"id":1,"type":"KSampler"}]}' * 200
		# use_metadata_tags is what lets the mp4 muxer write a non-standard tag at
		# all — without it ffmpeg silently drops `workflow`, and this fixture
		# would test nothing. It is also how ComfyUI gets the graph in there.
		assert subprocess.run(
			[exe, "-v", "error", "-y", "-f", "lavfi", "-i", "testsrc=size=64x48:rate=12:duration=1",
			 "-c:v", "libx264", "-pix_fmt", "yuv420p", "-movflags", "use_metadata_tags",
			 "-metadata", f"workflow={graph}", "-metadata", "prompt={\"3\":{}}", clip],
			capture_output=True).returncode == 0

		tags = mod.video_tags(clip)
		assert tags is not None and "workflow" in tags and "prompt" in tags
		before, size_before = stream_md5(clip), os.path.getsize(clip)

		changed, removed = mod.strip_video(clip, check=True)
		assert changed and removed > 5000
		assert mod.video_tags(clip)["workflow"] == graph, "--check must not write"

		changed, removed = mod.strip_file(clip)          # dispatches on the extension
		assert changed
		after_tags = mod.video_tags(clip)
		assert "workflow" not in after_tags and "prompt" not in after_tags, after_tags
		# Container brands and ffmpeg's own encoder line are all that may remain.
		assert set(after_tags) <= mod.BENIGN_TAGS, after_tags
		assert stream_md5(clip) == before, "the streams were re-encoded, not copied"
		assert os.path.getsize(clip) < size_before

		# A clip with nothing but benign tags is left alone.
		assert mod.strip_file(clip) == (False, 0)

	# ...and a directory sweep picks up both kinds.
	with tempfile.TemporaryDirectory() as d:
		_png_with_workflow(os.path.join(d, "a.png"))
		open(os.path.join(d, "b.mp4"), "wb").close()
		open(os.path.join(d, "c.txt"), "w").close()
		assert sorted(os.path.basename(x) for x in mod.collect([d])) == ["a.png", "b.mp4"]


def test_strip_png_metadata_refuses_what_it_cannot_read():
	"""A non-PNG and a truncated PNG must be reported, never half-written."""
	mod = _load_stripper()
	with tempfile.TemporaryDirectory() as d:
		jpg = os.path.join(d, "real.jpg")
		with open(jpg, "wb") as fh:
			fh.write(b"\xff\xd8\xff\xe0" + b"x" * 64)
		body = open(jpg, "rb").read()
		try:
			mod.strip_file(jpg)
			raise AssertionError("a .jpg must be refused")
		except mod.NotPng as exc:
			assert "not a PNG" in str(exc)
		assert open(jpg, "rb").read() == body, "the refused file was modified"

		src = _png_with_workflow(os.path.join(d, "src.png"))
		raw = open(src, "rb").read()
		cut = os.path.join(d, "cut.png")
		with open(cut, "wb") as fh:
			fh.write(raw[:len(raw) // 2])
		try:
			mod.strip_file(cut)
			raise AssertionError("a truncated PNG must be refused")
		except mod.NotPng as exc:
			assert "truncated" in str(exc)
		assert not any(f.endswith(".stripping") for f in os.listdir(d)), "temp file left behind"


# ------------------------------------------------------------- seed range noise
def _core_prepare_noise():
	"""comfy.sample.prepare_noise — the behaviour Seed Range Noise must match.

	Compared against core rather than a copy of its formula on purpose: the
	point is that our items stay interchangeable with core's, so if core ever
	changes how it draws noise, these tests should fail and tell us.

	ComfyUI's root is not on the test path (only custom_nodes/ is, so `import
	tinode` works), so it is added here and taken straight back out. Leaving it
	makes `folder_paths` importable, and nodes that probe for it — the video
	preview asset path — then take their ComfyUI-present branch for every test
	that follows, which quietly changes what the rest of the suite exercises.
	"""
	root = os.path.dirname(os.path.dirname(os.path.dirname(
		os.path.dirname(os.path.abspath(__file__)))))
	added = root not in sys.path
	if added:
		sys.path.insert(0, root)
	try:
		import comfy.sample  # noqa: PLC0415 — deliberately local, see above
		return comfy.sample.prepare_noise
	finally:
		if added:
			sys.path.remove(root)
		sys.modules.pop("folder_paths", None)


def test_seed_range_noise_item_matches_a_solo_run():
	"""The reason the node exists: item i's NOISE must equal a solo run at seed+i.

	Noise only — the rendered image still shifts slightly if you later re-run a
	candidate at a different batch size, because the UNet picks different GPU
	kernels per batch size. See the node docstring. If this drifts, a
	candidate's seed stops being a portable identity and we are back to
	carrying (seed, batch_index) pairs around.
	"""
	prepare_noise = _core_prepare_noise()
	seed = 740016770880953
	batch = Noise_SeedRange(seed).generate_noise({"samples": torch.zeros(5, 4, 64, 64)})
	assert batch.shape == (5, 4, 64, 64)
	for i in range(5):
		solo = prepare_noise(torch.zeros(1, 4, 64, 64), seed + i)
		assert torch.equal(batch[i:i + 1], solo), f"item {i} != solo run at seed+{i}"


def test_seed_range_noise_is_not_core_batching():
	"""Guards the premise: core slices one stream, so its item 1 is NOT seed+1.

	Should core ever change to per-item reseeding, this test fails and the node
	becomes redundant — that is worth being told about.
	"""
	prepare_noise = _core_prepare_noise()
	seed = 740016770880953
	core_batch = prepare_noise(torch.zeros(5, 4, 64, 64), seed)
	solo = prepare_noise(torch.zeros(1, 4, 64, 64), seed + 1)
	assert not torch.equal(core_batch[1:2], solo)


def test_seed_range_noise_honours_batch_index():
	# After Latent From Batch picks item 3, it must still be drawn from seed+3 —
	# otherwise "select a candidate, then continue it" silently changes the image.
	prepare_noise = _core_prepare_noise()
	seed = 740016770880953
	picked = Noise_SeedRange(seed).generate_noise(
		{"samples": torch.zeros(1, 4, 64, 64), "batch_index": [3]}
	)
	assert torch.equal(picked, prepare_noise(torch.zeros(1, 4, 64, 64), seed + 3))


def test_seed_range_noise_seed_mapping():
	assert seeds_for({"samples": torch.zeros(3, 4, 8, 8)}, 100) == [100, 101, 102]
	# A picked latent carries the positions it came from, not 0..n.
	assert seeds_for({"samples": torch.zeros(2, 4, 8, 8), "batch_index": [3, 7]}, 100) == [103, 107]


def test_seed_range_noise_refuses_nested_latents():
	class _Nested(torch.Tensor):
		is_nested = True

	samples = torch.zeros(1, 4, 8, 8).as_subclass(_Nested)
	try:
		Noise_SeedRange(0).generate_noise({"samples": samples})
		raise AssertionError("a nested latent must be refused, not silently mishandled")
	except RuntimeError as exc:
		assert "nested" in str(exc)


# ---------------------------------------------------------------- sigma segment
def test_sigma_segment_keeps_the_shared_boundary():
	"""Adjacent stages must overlap on one sigma, or the hand-off is not seamless.

	0->4 and 4->9 both contain sigmas[4]: the second stage has to be told the
	noise level it resumes at, not merely the sigmas that remain.
	"""
	sig = list(range(21))                      # stand-in for a 20-step schedule
	a = slice_sigmas(sig, 0, 4)
	b = slice_sigmas(sig, 4, 9)
	assert a == [0, 1, 2, 3, 4]
	assert b == [4, 5, 6, 7, 8, 9]
	assert a[-1] == b[0]
	# a segment runs (end - start) steps, holding one more sigma than that
	assert len(a) - 1 == 4 and len(b) - 1 == 5


def test_sigma_segment_indices_stay_absolute():
	# The whole point: asking for 9 means step 9 of the ORIGINAL schedule, with no
	# regard for how the earlier stages were cut. Chained SplitSigmas would need 5.
	sig = list(range(21))
	assert slice_sigmas(sig, 9, 14)[0] == 9


def test_sigma_segment_refuses_bad_ranges():
	sig = list(range(21))                      # 20 steps, indices 0..20
	for start, end in [(4, 4), (9, 4), (0, 21), (-1, 5)]:
		try:
			slice_sigmas(sig, start, end)
			raise AssertionError(f"({start},{end}) must be refused, not clamped")
		except ValueError:
			pass
	# The two invalid orderings are different mistakes and must not share a
	# message — "one sigma performs no sampling" does not explain a swapped pair.
	try:
		slice_sigmas(sig, 9, 4)
	except ValueError as exc:
		assert "before" in str(exc) and "swapped" in str(exc)
	try:
		slice_sigmas(sig, 4, 4)
	except ValueError as exc:
		assert "one sigma" in str(exc)
	# the exact end of the schedule is still valid
	assert len(slice_sigmas(sig, 14, 20)) == 7


def test_sigma_segment_node_reports_start_sigma():
	seg, start_sigma = SigmaSegment().execute([14.61, 10.74, 8.08, 6.20, 4.85, 3.86], 2, 4)
	assert seg == [8.08, 6.20, 4.85]
	assert abs(start_sigma - 8.08) < 1e-6



# ------------------------------------------------------------ candidate select
def test_candidate_select_reports_the_candidates_own_seed():
	"""The seed must match Seed Range Noise's mapping, or the identity is a lie."""
	lat = {"samples": torch.arange(5 * 4 * 8 * 8, dtype=torch.float32).reshape(5, 4, 8, 8)}
	out, _dn, img, seed, idx = CandidateSelect().execute(lat, index=2, origin_seed=100)["result"]
	assert seed == 102 and idx == 2
	assert torch.equal(out["samples"][0], lat["samples"][2])


def test_candidate_select_stamps_batch_index():
	# Without this, regenerating noise from the picked latent silently lands on a
	# different candidate's seed.
	lat = {"samples": torch.zeros(4, 4, 8, 8)}
	out = CandidateSelect().execute(lat, index=3)["result"][0]
	assert out["batch_index"] == [3]
	# and an already-sliced latent keeps its ORIGINAL position, not the new one
	nested = CandidateSelect().execute(
		{"samples": torch.zeros(2, 4, 8, 8), "batch_index": [7, 9]}, index=1)["result"][0]
	assert nested["batch_index"] == [9]



def test_candidate_select_slices_the_denoised_pair_too():
	"""Branching again needs BOTH of a sampler's outputs for the SAME candidate.

	Selecting them on two separate nodes would let the two indices drift apart,
	and the mismatch would be invisible — you would rotate one candidate's noise
	around another's prediction.
	"""
	lat = {"samples": torch.arange(3 * 4 * 8 * 8, dtype=torch.float32).reshape(3, 4, 8, 8)}
	den = {"samples": torch.ones(3, 4, 8, 8) * torch.arange(3).view(3, 1, 1, 1)}
	out, dn, _, _, _ = CandidateSelect().execute(lat, index=2, denoised=den)["result"]
	assert torch.equal(out["samples"][0], lat["samples"][2])
	assert float(dn["samples"].mean()) == 2.0

	# a mismatched pair cannot have come from one run, and must not be guessed at
	try:
		CandidateSelect().execute(lat, index=0, denoised={"samples": torch.zeros(5, 4, 8, 8)})
		raise AssertionError("a mismatched denoised batch must be refused")
	except RuntimeError as exc:
		assert "SAME sampler" in str(exc)


def test_candidate_select_clamps_and_reports_where_it_landed():
	# A cursor should stop at the end, not raise — but it must say where it is.
	lat = {"samples": torch.zeros(3, 4, 8, 8)}
	res = CandidateSelect().execute(lat, index=99, origin_seed=50)
	out, _dn, _im, seed, idx = res["result"]
	assert idx == 2 and seed == 52
	ui = res["ui"]["ti_candidate"][0]
	assert (ui["index"], ui["count"], ui["seed"]) == (2, 3, 52)


def test_candidate_select_survives_a_missing_preview_grid():
	# The thumbnails need ComfyUI's folder_paths, which the suite runs without.
	# Selection is the node's job; the grid is decoration and must not break it.
	lat = {"samples": torch.zeros(2, 4, 8, 8)}
	imgs = torch.zeros(2, 8, 8, 3)
	res = CandidateSelect().execute(lat, index=1, images=imgs, origin_seed=10)
	assert res["result"][3] == 11
	assert res["ui"]["ti_candidate"][0]["thumbs"] == []


def test_candidate_select_picks_the_matching_image():
	lat = {"samples": torch.zeros(3, 4, 8, 8)}
	imgs = torch.stack([torch.full((8, 8, 3), float(i)) for i in range(3)])
	img = CandidateSelect().execute(lat, index=1, images=imgs)["result"][2]
	assert img.shape[0] == 1 and float(img.mean()) == 1.0
	# images are optional: without them the slot is simply empty
	assert CandidateSelect().execute(lat, index=1)["result"][2] is None


def test_candidate_select_refuses_an_empty_batch():
	try:
		CandidateSelect().execute({"samples": torch.zeros(0, 4, 8, 8)})
		raise AssertionError("an empty batch must be refused")
	except RuntimeError as exc:
		assert "empty" in str(exc)



# ------------------------------------------------------------------- step stamp
def test_stamp_step_round_trips():
	"""The whole contract: what Stamp writes, Resume reads back."""
	lat = {"samples": torch.zeros(1, 4, 8, 8)}
	stamped = StampStep().execute(lat, 11)[0]
	assert ResumeStep().execute(stamped)[0] == 11


def test_stamp_survives_the_nodes_it_has_to_travel_through():
	"""The stamp is only useful if every hop preserves it.

	SamplerCustomAdvanced, Candidate Select and Noise Rotate all rebuild the
	latent with `latent.copy()`, so extra keys ride along. Pin that for the two
	nodes in this pack — if either ever stops copying, the stamp goes silently
	missing and stages resume at the wrong sigma.
	"""
	g = torch.Generator().manual_seed(3)
	batch = {"samples": torch.randn(4, 4, 8, 8, generator=g), "ti_step": 9}
	denoised = {"samples": torch.randn(4, 4, 8, 8, generator=g), "ti_step": 9}

	picked, picked_dn = CandidateSelect().execute(batch, index=2, denoised=denoised)["result"][:2]
	assert stamped_step(picked) == 9, "Candidate Select dropped the stamp"
	assert stamped_step(picked_dn) == 9, "Candidate Select dropped it on `denoised`"

	kids = NoiseRotate().execute(picked, picked_dn, 25.0, 3, 500)[0]
	assert stamped_step(kids) == 9, "Noise Rotate dropped the stamp"


def test_resume_step_falls_back_for_an_unstamped_latent():
	"""An Empty Latent has never been sampled, so step 0 is the truth."""
	assert ResumeStep().execute({"samples": torch.zeros(1, 4, 8, 8)})[0] == 0
	assert ResumeStep().execute({"samples": torch.zeros(1, 4, 8, 8)}, fallback=7)[0] == 7


def test_stamp_step_refuses_to_go_backwards():
	"""Checkpoints out of order is the mistake this catches.

	Sampling only moves forward, so a stage ending before the latent already is
	means the controls are misordered — worth an error while the numbers are
	still on screen, rather than a plausible image from the wrong sigma.
	"""
	at14 = {"samples": torch.zeros(1, 4, 8, 8), "ti_step": 14}
	try:
		StampStep().execute(at14, 9)
		assert False, "stamping backwards should raise"
	except ValueError as exc:
		assert "already at step 14" in str(exc)
	# equal is fine: re-stamping the same position is a no-op, not a mistake
	assert ResumeStep().execute(StampStep().execute(at14, 14)[0])[0] == 14


def test_stamp_step_does_not_mutate_its_input():
	"""The incoming latent may still be feeding other nodes."""
	lat = {"samples": torch.zeros(1, 4, 8, 8)}
	StampStep().execute(lat, 4)
	assert "ti_step" not in lat


def test_stamp_step_keeps_the_other_latent_keys():
	"""batch_index in particular — losing it would move a candidate's seed."""
	lat = {"samples": torch.zeros(1, 4, 8, 8), "batch_index": [3],
		   "noise_mask": torch.ones(1, 1, 8, 8)}
	out = StampStep().execute(lat, 4)[0]
	assert out["batch_index"] == [3]
	assert "noise_mask" in out


# ----------------------------------------------------------------- noise rotate
def _parent(sigma=4.8557, seed=0):
	"""A stand-in checkpoint: x = x0 + sigma*eps, the shape a stopped sampler emits."""
	g = torch.Generator().manual_seed(seed)
	x0 = torch.randn(1, 4, 16, 16, generator=g) * 0.5      # a hedged prediction
	eps = torch.randn(1, 4, 16, 16, generator=g)
	return {"samples": x0 + sigma * eps}, {"samples": x0}


def test_noise_rotate_preserves_the_noise_level():
	"""The invariant the whole method rests on: theta turns, it does not inflate.

	If ||x - x0|| grows with theta, the continuation runs a schedule expecting
	less noise than it gets and under-denoises — variation strength and quality
	loss confounded in one dial.
	"""
	lat, den = _parent()
	before = (lat["samples"] - den["samples"]).std().item()
	for theta in (0, 10, 30, 45, 90):
		out = NoiseRotate().execute(lat, den, theta, 3, 500)[0]["samples"]
		for i in range(out.shape[0]):
			after = (out[i:i+1] - den["samples"]).std().item()
			assert abs(after / before - 1) < 0.06, f"theta={theta} moved the magnitude"


def test_noise_rotate_theta_zero_is_the_exact_parent():
	lat, den = _parent()
	out = NoiseRotate().execute(lat, den, 0.0, 2, 7)[0]["samples"]
	for i in range(out.shape[0]):
		assert torch.allclose(out[i:i+1], lat["samples"], atol=1e-6)


def test_noise_rotate_divergence_grows_with_theta():
	lat, den = _parent()
	d = []
	for theta in (0, 10, 30, 60, 90):
		v = NoiseRotate().execute(lat, den, theta, 1, 11)[0]["samples"]
		d.append((v - lat["samples"]).abs().mean().item())
	assert d == sorted(d), f"divergence must be monotonic in theta, got {d}"
	assert d[0] == 0.0


def test_noise_rotate_theta_is_the_cosine_similarity():
	"""The widget number is the angle actually turned — that is the whole claim.

	Holds because `new` is drawn independently of `eps`, so the two are
	near-orthogonal and cos_sim(eps, eps_var) collapses to cos(theta).
	"""
	lat, den = _parent()
	eps = (lat["samples"] - den["samples"]).flatten()
	for theta in (10, 30, 45, 90, 135):
		v = NoiseRotate().execute(lat, den, float(theta), 1, 4)[0]["samples"]
		ev = (v - den["samples"]).flatten()
		cos = float(ev @ eps / (ev.norm() * eps.norm()))
		assert abs(cos - math.cos(math.radians(theta))) < 0.05, \
			f"theta={theta} turned {math.degrees(math.acos(max(-1, min(1, cos)))):.1f} deg"


def test_noise_rotate_past_ninety_collapses_the_siblings():
	"""Beyond 90 the descendants march away from the parent but back together.

	Sibling angle is acos(cos^2 theta), so it peaks at 90 and closes again — the
	reason 90 is the working maximum even though the widget now allows the whole
	circle.
	"""
	lat, den = _parent()

	def spread(theta):
		out = NoiseRotate().execute(lat, den, float(theta), 2, 21)[0]["samples"]
		a = (out[0:1] - den["samples"]).flatten()
		b = (out[1:2] - den["samples"]).flatten()
		return float(a @ b / (a.norm() * b.norm()))

	# cos of the sibling angle: 1 = identical, 0 = maximally spread.
	assert spread(90) < 0.15, "siblings should be near-orthogonal at 90"
	assert spread(135) > spread(90), "past 90 siblings close back up"
	assert spread(179) > spread(135)


def test_noise_rotate_at_one_eighty_ignores_the_variation_seed():
	"""sin(180) = 0, so `new` drops out: every descendant is the same negative.

	Worth pinning — a user who winds theta past 180 expecting more variety gets
	N copies of one image, and that surprise should be a documented property
	rather than a bug report.
	"""
	lat, den = _parent()
	a = NoiseRotate().execute(lat, den, 180.0, 3, 1)[0]["samples"]
	b = NoiseRotate().execute(lat, den, 180.0, 3, 999999)[0]["samples"]
	assert torch.allclose(a, b, atol=1e-5), "the variation seed still mattered at 180"
	for i in range(1, a.shape[0]):
		assert torch.allclose(a[0:1], a[i:i+1], atol=1e-5), "descendants differ at 180"
	# and that one image is the exact negative of the residual
	expected = den["samples"] - (lat["samples"] - den["samples"])
	assert torch.allclose(a[0:1], expected, atol=1e-5)


def test_noise_rotate_negative_theta_mirrors_positive():
	"""-theta has the same strength as +theta but is a different descendant."""
	lat, den = _parent()
	eps = (lat["samples"] - den["samples"]).flatten()
	pos = NoiseRotate().execute(lat, den, 30.0, 1, 55)[0]["samples"]
	neg = NoiseRotate().execute(lat, den, -30.0, 1, 55)[0]["samples"]

	def cos_to_parent(v):
		ev = (v - den["samples"]).flatten()
		return float(ev @ eps / (ev.norm() * eps.norm()))

	assert abs(cos_to_parent(pos) - cos_to_parent(neg)) < 0.02, "equal strength"
	assert not torch.allclose(pos, neg, atol=1e-3), "but a different descendant"


def test_noise_rotate_descendants_are_individually_seeded():
	# Variant i comes from variation_seed + i, mirroring Seed Range Noise, so a
	# descendant you liked can be reproduced on its own.
	lat, den = _parent()
	batch = NoiseRotate().execute(lat, den, 30.0, 3, 900)[0]["samples"]
	for i in range(3):
		solo = NoiseRotate().execute(lat, den, 30.0, 1, 900 + i)[0]["samples"]
		assert torch.allclose(batch[i:i+1], solo, atol=1e-6)
	assert not torch.allclose(batch[0:1], batch[1:2])


def test_noise_rotate_drops_the_parents_batch_index():
	# Descendants are a new lineage; keeping the parent's slot would send anything
	# that regenerates noise to the wrong seed.
	lat, den = _parent()
	lat["batch_index"] = [3]
	assert "batch_index" not in NoiseRotate().execute(lat, den, 20.0, 2, 0)[0]


def test_noise_rotate_refuses_impossible_inputs():
	lat, den = _parent()
	# a finished latent has no residual left to turn
	try:
		NoiseRotate().execute({"samples": den["samples"].clone()}, den, 30.0, 2, 0)
		raise AssertionError("a zero residual must be refused")
	except RuntimeError as exc:
		assert "unresolved noise" in str(exc)
	# mismatched shapes mean the two inputs came from different samplers
	try:
		NoiseRotate().execute(lat, {"samples": torch.zeros(1, 4, 8, 8)}, 30.0, 2, 0)
		raise AssertionError("a shape mismatch must be refused")
	except RuntimeError as exc:
		assert "SAME sampler" in str(exc)
	# and a batch has no single parent to branch from
	try:
		NoiseRotate().execute({"samples": torch.zeros(3, 4, 16, 16)},
							  {"samples": torch.ones(3, 4, 16, 16)}, 30.0, 2, 0)
		raise AssertionError("a batched parent must be refused")
	except RuntimeError as exc:
		assert "Candidate Select" in str(exc)



# ------------------------------------------------------------------ runner
def _main():
	tests = [(n, f) for n, f in sorted(globals().items())
			if n.startswith("test_") and callable(f)]
	failed = []
	for name, fn in tests:
		try:
			fn()
			print(f"  PASS  {name}")
		except Exception as exc:  # noqa: BLE001 — this is the reporter
			failed.append((name, exc))
			print(f"  FAIL  {name}: {exc}")
	print(f"\n{len(tests) - len(failed)}/{len(tests)} passed")
	return 1 if failed else 0


if __name__ == "__main__":
	sys.exit(_main())
