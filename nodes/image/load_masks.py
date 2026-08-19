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
			},
		}

	RETURN_TYPES = ("ITEM_LIST", "INT")
	RETURN_NAMES = ("item_list", "count")
	OUTPUT_TOOLTIPS = (
		"One item per saved CROP — wire to ▶Foreach List, then Load Mask inside.",
		"How many crops were found (across all clips).",
	)
	FUNCTION = "execute"

	@classmethod
	def IS_CHANGED(cls, subdir=store.DEFAULT_SUBDIR, source_dir=""):
		try:
			root = store.output_root(subdir)
			sig = []
			for it in store.scan_items(root):
				sig.append(f"{it['item_dir']}:{os.path.getmtime(store.manifest_path(it['item_dir']))}")
			return "|".join(sig)
		except Exception as exc:  # noqa: BLE001
			return repr(exc)

	def execute(self, subdir=store.DEFAULT_SUBDIR, source_dir=""):
		root = store.output_root(subdir)
		src = _abs_source_dir(source_dir)
		items = store.scan_items(root)
		if not items:
			raise RuntimeError(f"Load Masks: no saved crops under {root}")
		for it in items:
			it["source_path"] = resolve_source(it.get("source_path", ""), it.get("stem", ""), src) or ""
		return (list(items), len(items))


@register
class LoadMask(TiNode):
	DISPLAY_NAME = "Load Mask (ti)"
	CATEGORY = "tinode/video"

	@classmethod
	def INPUT_TYPES(cls):
		return {"required": {"item": ("TI_MASK_ITEM", {"tooltip":
			"One crop item from Load Masks, via ForeachListBegin's `item` output."})}}

	RETURN_TYPES = ("VIDEO", "MASK", "TI_CROP_XFORM", "STRING", "STRING", "STRING", "INT", "INT")
	RETURN_NAMES = ("video", "mask", "crop_info", "stem",
					"positive_prompt", "negative_prompt", "crop_index", "frame_count")
	OUTPUT_TOOLTIPS = (
		"The original source clip (lazy — re-decoded, not a re-encode).",
		"The saved mask, in crop space ([frames, ch, cw]).",
		"Maps the crop back to the full frame.",
		"The clip's key.",
		"This crop's saved VOID positive prompt.",
		"This crop's saved negative prompt.",
		"Which crop of the clip.",
		"Number of frames.",
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
		mask = store.load_mask_sequence(item["mask_dir"], item.get("mask_pattern", store.MASK_PATTERN), n)
		return (
			VideoFromFile(source_path), mask, item["crop_info"], stem,
			str(item.get("positive_prompt", "")), str(item.get("negative_prompt", "")),
			int(item.get("crop_index", 0)), n or mask.shape[0],
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
		crops = store.load_rgb_sequence(
			item["crop_dir"], item.get("crop_pattern", store.CROP_PATTERN), n)
		return (crops, n)
