"""Load Masks — read back the per-crop store for the removal pass (phase 2a).

Load Masks scans the store and emits an Inspire ITEM_LIST with ONE item PER CROP
(flattened across every clip), so a Foreach loop removes them one crop at a time.
Inside the loop, Load Mask turns the current item into what the removal graph
needs, and Load Cropped Frames gives the lossless crop pixels.

  Load Mask   item -> video (original, re-decoded) + mask (crop space) +
                      crop_info + stem + positive/negative prompt + crop_index
  Load Cropped Frames  item -> the exported crop RGB (8/16-bit)

Source pairing: each item carries the source path Save Crop & Mask recorded; if
that file has moved, `source_dir` is searched for `<stem>.<ext>`.
"""

from __future__ import annotations

import os

import fnmatch

import torch

from ...base import TiNode, first
from ...registry import register
from ._video_io import VIDEO_EXTS
from . import _mask_store as store

try:
	from comfy_api.latest import InputImpl  # noqa: PLC0415

	VideoFromFile = InputImpl.VideoFromFile
except Exception:  # noqa: BLE001
	VideoFromFile = None


def resolve_source(recorded, stem, source_dir):
	"""Absolute path to a clip's source video, or None if it can't be found."""
	if recorded and os.path.isfile(recorded):
		return recorded
	if source_dir and os.path.isdir(source_dir):
		for ext in VIDEO_EXTS:
			cand = os.path.join(source_dir, stem + ext)
			if os.path.isfile(cand):
				return cand
	return None


def _abs_source_dir(source_dir):
	src = str(first(source_dir, "")).strip()
	if src and not os.path.isabs(src):
		try:
			import folder_paths  # noqa: PLC0415

			return os.path.join(folder_paths.get_input_directory(), src)
		except Exception:  # noqa: BLE001
			return src
	return src


def match_stem(stem, spec):
	"""True when `stem` passes the filter — exact names or globs, empty = all."""
	pats = [t for t in str(spec or "").replace(",", " ").split() if t]
	if not pats:
		return True
	low = str(stem).lower()
	return any(fnmatch.fnmatch(low, p.lower()) for p in pats)


def build_report(found, selected, root):
	"""A readable summary of the store and what this run selected."""
	by_clip = {}
	for it in found:
		by_clip.setdefault(it.get("stem", "?"), []).append(it)
	sel = {id(it) for it in selected}
	lines = [f"{root}", f"{len(found)} crop(s) in {len(by_clip)} clip(s); "
			 f"{len(selected)} selected"]
	for stem in sorted(by_clip):
		crops = sorted(by_clip[stem], key=lambda c: (int(c.get("chunk_index", -1)),
													 int(c.get("crop_index", 0))))
		done = sum(1 for c in crops if c.get("has_filled"))
		took = sum(1 for c in crops if id(c) in sel)
		lines.append(f"  {stem}: {len(crops)} crop(s), {done} filled, {took} selected")
		for c in crops:
			mark = "*" if id(c) in sel else " "
			ch = c.get("chunk_index", -1)
			lines.append(
				f"   {mark} chunk {ch if ch >= 0 else '-'} crop {c.get('crop_index')}"
				f"  frames {c.get('frame_start')}-{c.get('frame_end')}"
				f"  {c.get('crop_width')}x{c.get('crop_height')}"
				f"  {'filled' if c.get('has_filled') else 'pending'}"
				f"{'' if c.get('has_crop') else '  (no crop export)'}")
	return "\n".join(lines)


@register
class LoadMasks(TiNode):
	DISPLAY_NAME = "Load Masks (ti)"
	CATEGORY = "tinode/video"

	@classmethod
	def INPUT_TYPES(cls):
		return {
			"required": {
				"subdir": ("STRING", {"default": store.DEFAULT_SUBDIR, "tooltip":
					"Folder under ComfyUI's output/ that Save Crop & Mask wrote to."}),
			},
			"optional": {
				"source_dir": ("STRING", {"default": "", "tooltip":
					"Where the source videos live now, if they've moved. Relative "
					"to input/ or absolute. Empty = trust each manifest's path."}),
				"stems": ("STRING", {"default": "", "tooltip":
					"Only these clips — comma/space separated, wildcards allowed "
					"(`sh001*`, `sh0030, sh0040`). Empty = every clip."}),
				"only_unfilled": ("BOOLEAN", {"default": False, "tooltip":
					"Skip crops that already have a filled/ result, so a resumed "
					"batch continues where it stopped instead of redoing work."}),
				# APPENDED last: widget values are positional in saved graphs.
				"every_nth": ("INT", {"default": 1, "min": 1, "max": 10, "step": 1,
					"tooltip": "Work at reduced rate: 2 = every 2nd frame (25fps "
							   "from 50fps masks) on ONE global grid across chunks. "
							   "Stamped into every item, so Load Mask / Load Cropped "
							   "Frames stride identically from this single switch. "
							   "The store's masks stay full-rate."}),
			},
		}

	# `report` is appended so existing graphs keep their link slots.
	RETURN_TYPES = ("ITEM_LIST", "INT", "STRING")
	RETURN_NAMES = ("item_list", "count", "report")
	OUTPUT_TOOLTIPS = (
		"One item per saved CROP — wire to Item Cursor or ▶Foreach List.",
		"How many crops matched.",
		"What the store holds and what was selected — wire to a Preview Any to "
		"see the whole batch at a glance.",
	)
	FUNCTION = "execute"

	@classmethod
	def IS_CHANGED(cls, subdir=store.DEFAULT_SUBDIR, source_dir="", **_filters):
		try:
			root = store.output_root(subdir)
			sig = []
			for it in store.scan_items(root):
				sig.append(f"{it['item_dir']}:{os.path.getmtime(store.manifest_path(it['item_dir']))}")
			return "|".join(sig)
		except Exception as exc:  # noqa: BLE001
			return repr(exc)

	def execute(self, subdir=store.DEFAULT_SUBDIR, source_dir="", stems="",
				only_unfilled=False, every_nth=1):
		root = store.output_root(subdir)
		src = _abs_source_dir(source_dir)
		found = store.scan_items(root)
		if not found:
			raise RuntimeError(f"Load Masks: no saved crops under {root}")
		for it in found:
			it["source_path"] = resolve_source(
				it.get("source_path", ""), it.get("stem", ""), src) or ""

		nth = max(1, int(first(every_nth, 1)))
		for it in found:
			it["every_nth"] = nth              # ONE switch drives every loader
		items = [it for it in found if match_stem(it.get("stem", ""), stems)]
		if bool(first(only_unfilled, False)):
			items = [it for it in items if not it.get("has_filled")]
		if not items:
			raise RuntimeError(
				f"Load Masks: {len(found)} crop(s) in {root}, but none matched "
				f"(stems={str(first(stems,'')).strip()!r}, "
				f"only_unfilled={bool(first(only_unfilled, False))}).")

		report = build_report(found, items, root)
		print("[tinode] Load Masks:\n" + report)
		# A missing source is only fatal when that item is actually loaded, so
		# warn here instead of raising — the rest of the batch can still run.
		missing = [it["stem"] for it in items if not it.get("source_path")]
		if missing:
			print(f"[tinode] Load Masks: source video not found for {sorted(set(missing))} "
				  "— set source_dir.")
		return (list(items), len(items), report)


@register
class LoadMask(TiNode):
	DISPLAY_NAME = "Load Mask (ti)"
	CATEGORY = "tinode/video"

	@classmethod
	def INPUT_TYPES(cls):
		return {"required": {"item": ("TI_MASK_ITEM", {"tooltip":
			"One crop item from Load Masks, via ForeachListBegin's `item` output."})}}

	# frame_start / frame_end / chunk_index are APPENDED so saved graphs keep
	# their link slots.
	RETURN_TYPES = ("VIDEO", "MASK", "TI_CROP_XFORM", "STRING", "STRING", "STRING",
					"INT", "INT", "INT", "INT", "INT")
	RETURN_NAMES = ("video", "mask", "crop_info", "stem",
					"positive_prompt", "negative_prompt", "crop_index", "frame_count",
					"frame_start", "frame_end", "chunk_index")
	OUTPUT_TOOLTIPS = (
		"The original source clip (lazy — re-decoded, not a re-encode).",
		"The saved mask, in crop space ([frames, ch, cw]).",
		"Maps the crop back to the full frame.",
		"The clip's key.",
		"This crop's saved VOID positive prompt.",
		"This crop's saved negative prompt.",
		"Which crop of the clip.",
		"Number of frames.",
		"First frame of this chunk in the original clip — wire to Paste Back's "
		"frame_offset so the result lands where it came from.",
		"End frame (exclusive) in the original.",
		"Which chunk of the clip (-1 if it was not chunked).",
	)
	FUNCTION = "execute"

	def execute(self, item):
		item = first(item)
		if not isinstance(item, dict):
			raise RuntimeError("Load Mask: `item` must come from Load Masks.")
		stem = item.get("stem", "")
		source_path = item.get("source_path", "")
		if not source_path or not os.path.isfile(source_path):
			raise RuntimeError(
				f"Load Mask: source video for {stem!r} not found ({source_path!r}). "
				"Set Load Masks' source_dir to where the footage lives now.")
		if VideoFromFile is None:
			raise RuntimeError(
				"Load Mask: ComfyUI's native VIDEO API (comfy_api.latest) is unavailable.")

		n = int(item.get("frame_count", 0))
		fstart = int(item.get("frame_start", 0))
		nth = max(1, int(item.get("every_nth", 1)))
		# A pre-strided store (source_every_nth) is never strided again.
		nth = max(1, nth // max(1, int(item.get("source_every_nth", 1))))
		idx = store.stride_indices(fstart, n, nth) if nth > 1 else None
		mask = store.load_mask_sequence(item["mask_dir"], item.get("mask_pattern", store.MASK_PATTERN),
									 n, indices=idx)
		if nth > 1:
			print(f"[tinode] Load Mask: half-rate x{nth} — {mask.shape[0]} of {n} frame(s).")
		return (
			VideoFromFile(source_path), mask, item["crop_info"], stem,
			str(item.get("positive_prompt", "")), str(item.get("negative_prompt", "")),
			int(item.get("crop_index", 0)), int(mask.shape[0]),
			fstart, int(item.get("frame_end", fstart + (n or mask.shape[0]))),
			int(item.get("chunk_index", -1)),
		)


@register
class LoadCroppedFrames(TiNode):
	DISPLAY_NAME = "Load Cropped Frames (ti)"
	CATEGORY = "tinode/video"

	@classmethod
	def INPUT_TYPES(cls):
		return {"required": {"item": ("TI_MASK_ITEM", {"tooltip":
			"A crop item from Load Masks. Reads the lossless crop/ sequence Save "
			"Crop & Mask exported (crop_image must have been connected there)."})}}

	RETURN_TYPES = ("IMAGE", "INT")
	RETURN_NAMES = ("crops", "frame_count")
	OUTPUT_TOOLTIPS = (
		"The cropped frames, exactly as exported (8- or 16-bit, lossless).",
		"Number of frames.",
	)
	FUNCTION = "execute"

	def execute(self, item):
		item = first(item)
		if not isinstance(item, dict):
			raise RuntimeError("Load Cropped Frames: `item` must come from Load Masks.")
		if not item.get("has_crop"):
			raise RuntimeError(
				f"Load Cropped Frames: {item.get('stem','this clip')!r} crop "
				f"{item.get('crop_index','?')} has no exported crops. Connect "
				"crop_image on Save Crop & Mask, or rebuild with Crop By Info.")
		n = int(item.get("frame_count", 0))
		nth = max(1, int(item.get("every_nth", 1)))
		nth = max(1, nth // max(1, int(item.get("source_every_nth", 1))))
		idx = store.stride_indices(int(item.get("frame_start", 0)), n, nth) if nth > 1 else None
		crops = store.load_rgb_sequence(
			item["crop_dir"], item.get("crop_pattern", store.CROP_PATTERN), n, indices=idx)
		return (crops, int(crops.shape[0]))


@register
class LoadChunkContext(TiNode):
	DISPLAY_NAME = "Load Chunk Context (ti)"
	CATEGORY = "tinode/video"

	@classmethod
	def INPUT_TYPES(cls):
		return {
			"required": {
				"item": ("TI_MASK_ITEM", {"tooltip":
					"The crop being processed now, from Load Masks."}),
			},
			"optional": {
				"variant": ("STRING", {"default": "", "tooltip":
					"Which saved result to continue from — must match the variant "
					"Save Filled writes."}),
				"frames": ("INT", {"default": 16, "min": 1, "max": 512, "step": 1,
					"tooltip": "How many trailing frames to hand back. Frame Pad uses "
							   "only as many as it pads, so any value >= the pad works."}),
			},
		}

	RETURN_TYPES = ("IMAGE", "BOOLEAN", "INT")
	RETURN_NAMES = ("context", "found", "chunk_index")
	OUTPUT_TOOLTIPS = (
		"The previous chunk's last frames — wire to Frame Pad's context_images. "
		"For the FIRST chunk there is no previous one, so this is empty and Frame "
		"Pad falls back to repeating frame 0 on its own.",
		"Whether a previous chunk was actually found and loaded.",
		"The chunk index this context came from (-1 if none).",
	)
	FUNCTION = "execute"

	@classmethod
	def IS_CHANGED(cls, **kwargs):
		return float("nan")      # the previous chunk is written between runs

	def execute(self, item, variant="", frames=16):
		it = first(item)
		if not isinstance(it, dict):
			raise RuntimeError("Load Chunk Context: `item` must come from Load Masks.")
		variant = str(first(variant, "") or "").strip()
		chunk = int(it.get("chunk_index", -1))
		# "No context" must be None, not a black frame: Frame Pad uses ANY tensor
		# wired into context_images, so a placeholder image would pad the first
		# chunk with black. None makes it fall back to the frame-0 freeze.
		none = (None, False, -1)
		if chunk <= 0:
			return none              # first chunk (or unchunked): nothing precedes it

		root = os.path.dirname(os.path.dirname(it["item_dir"]))
		prev = store.item_dir(root, it.get("stem", ""), int(it.get("crop_index", 0)), chunk - 1)
		if not os.path.isfile(store.manifest_path(prev)):
			print(f"[tinode] Load Chunk Context: chunk {chunk - 1} not saved yet.")
			return none
		man = store.read_manifest(prev)
		if not man.get(store.filled_flag(variant)):
			print(f"[tinode] Load Chunk Context: chunk {chunk - 1} has no "
				  f"{'filled' if not variant else 'filled_' + variant}/ yet — "
				  "process the chunks in order.")
			return none

		# The fill's own recorded length, NOT the crop's: half-rate fills hold
		# every 2nd frame, and assuming frame_count would read missing files.
		n = int(man.get(store.filled_frames_key(variant), man.get("frame_count", 0)))
		want = min(int(first(frames, 16)), n)
		seq = store.load_rgb_sequence(
			store.filled_dir(prev, variant), man.get("filled_pattern", store.FILLED_PATTERN), n)
		print(f"[tinode] Load Chunk Context: continuing from chunk {chunk - 1}, "
			  f"last {want} of {n} frame(s).")
		return (seq[-want:].contiguous(), True, chunk - 1)
