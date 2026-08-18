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
from tinode.nodes.image.extend_video import ExtendVideo  # noqa: E402
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
def test_pick_segments_filters_by_id():
	segs, img = _segments(), torch.zeros(2, 40, 40, 3)
	mask, _, out = _unwrap(PickSegments().execute(img, segs, excluded_ids="[3]"))
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

	_, _, same = _unwrap(PickSegments().execute(img_a, segs, excluded_ids=stored))
	assert same["ids"] == [7], "selection should apply to its own input"

	_, _, fresh = _unwrap(PickSegments().execute(torch.rand(2, 40, 40, 3), segs,
												excluded_ids=stored))
	assert fresh["ids"] == [3, 7], "stale selection must be dropped on new input"

	_, _, legacy = _unwrap(PickSegments().execute(img_a, segs, excluded_ids="[3]"))
	assert legacy["ids"] == [7], "bare-list selections stay supported"


def test_add_segments_merges_and_matches_pick_outputs():
	assert AddSegments.RETURN_TYPES == PickSegments.RETURN_TYPES
	assert AddSegments.RETURN_NAMES == PickSegments.RETURN_NAMES

	segs, img = _segments(), torch.zeros(2, 40, 40, 3)
	manual = json.dumps([{"id": _MANUAL_ID_BASE, "frame": 1, "bbox": [0, 0, 5, 5]}])
	mask, _, out = _unwrap(AddSegments().execute(img, segs, manual_segments=manual))
	assert out["ids"] == [3, 7, _MANUAL_ID_BASE]
	assert [s["id"] for s in out["frames"][1]] == [_MANUAL_ID_BASE]
	assert int(mask[1].sum()) == 25                   # the drawn 5x5 box
	# the incoming segments must not be mutated
	assert [s["id"] for s in segs["frames"][0]] == [3, 7]


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

	mask, _, out = _unwrap(DeleteSegments().execute(image, segs, deleted_items=raw))
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


def test_extend_video_holds_and_splices():
	base = torch.zeros(5, 8, 6, 3)
	for i in range(5):
		base[i] = i / 10.0

	out, pre, app = ExtendVideo().execute(base, prepend_mode="first_frame", prepend_frames=2,
										append_mode="last_frame", append_frames=3)
	assert (pre, app) == (2, 3) and out.shape[0] == 10
	assert torch.equal(out[2:7], base)                       # base sits between
	assert torch.equal(out[0], base[0]) and torch.equal(out[-1], base[-1])

	# a spliced clip of another size/channel count is conformed, base untouched
	out, pre, app = ExtendVideo().execute(base, prepend_mode="video",
										prepend_video=torch.rand(3, 16, 12, 4))
	assert out.shape == (8, 8, 6, 3) and pre == 3
	assert torch.equal(out[3:], base)

	# video mode with nothing wired is a no-op, not a crash
	out, pre, _ = ExtendVideo().execute(base, prepend_mode="video")
	assert out.shape[0] == 5 and pre == 0


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
		"document", "window", "requestAnimationFrame", "setTimeout", "ResizeObserver",
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
