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
from tinode.nodes.image.paste_back import MaskCropPasteBack  # noqa: E402
from tinode.nodes.image.pick_segments import (  # noqa: E402
	PickSegments, color_for_id, prune_asset_cache,
)
from tinode.nodes.image.add_segments import AddSegments, _MANUAL_ID_BASE  # noqa: E402
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


def test_add_segments_ignores_degenerate_boxes():
	segs, img = _segments(), torch.zeros(2, 40, 40, 3)
	bad = json.dumps([{"id": 1, "frame": 99, "bbox": [0, 0, 5, 5]},     # frame OOR
					{"id": 2, "frame": 0, "bbox": [5, 5, 5, 9]}])     # zero width
	out = _unwrap(AddSegments().execute(img, segs, manual_segments=bad))[2]
	assert 1 not in out["ids"] and 2 not in out["ids"]


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
