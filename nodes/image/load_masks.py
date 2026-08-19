"""Load Masks — read back the phase-1 mask store for the removal pass.

Phase 2 of the two-workflow object-removal pipeline. Load Masks scans the store
Save Crop & Mask wrote and emits an Inspire ITEM_LIST — one item per clip — so a
Foreach loop processes them one at a time. Inside the loop, Load Mask turns the
current item into the pieces the removal graph needs:

  video      the ORIGINAL source clip (lazy, re-decoded — never a re-encode)
  mask       the saved mask, in CROP space ([frames, ch, cw])
  crop_info  maps the crop back into the original frame (for Crop By Info and
             Mask Crop Paste Back)
  stem       the clip's key, e.g. to name the saved output

Source pairing: each item carries the source path Save Crop & Mask recorded. If that
file has moved, `source_dir` is searched for `<stem>.<ext>` as a fallback, so a
store stays usable after the footage is relocated.
"""

from __future__ import annotations

import os

import torch

from ...base import TiNode, first
from ...registry import register
from ._video_io import VIDEO_EXTS
from . import _mask_store as store

try:
	from comfy_api.latest import InputImpl  # noqa: PLC0415

	VideoFromFile = InputImpl.VideoFromFile
except Exception:  # noqa: BLE001 — no ComfyUI (tests)
	VideoFromFile = None


def _resolve_source(recorded, stem, source_dir):
	"""Absolute path to a clip's source video, or None if it can't be found."""
	if recorded and os.path.isfile(recorded):
		return recorded
	if source_dir and os.path.isdir(source_dir):
		for ext in VIDEO_EXTS:
			cand = os.path.join(source_dir, stem + ext)
			if os.path.isfile(cand):
				return cand
	return None


def scan_store(root, source_dir=""):
	"""List of per-clip items (augmented manifests) found under `root`, by stem.

	Pure/testable: pass a real directory tree; needs no comfy_api. Each item is
	the manifest plus absolute `clip_dir` / `mask_dir` / resolved `source_path`.
	"""
	if not os.path.isdir(root):
		raise RuntimeError(f"Load Masks: store not found: {root}")
	items = []
	for name in sorted(os.listdir(root)):
		cdir = os.path.join(root, name)
		if not os.path.isfile(store.manifest_path(cdir)):
			continue
		man = store.read_manifest(cdir)
		stem = man.get("stem", name)
		item = dict(man)
		item["clip_dir"] = cdir
		item["mask_dir"] = store.mask_dir(cdir)
		item["source_path"] = _resolve_source(man.get("source_path", ""), stem, source_dir) or ""
		items.append(item)
	if not items:
		raise RuntimeError(f"Load Masks: no saved clips (manifest.json) under {root}")
	return items


def load_mask_frames(mask_dir, pattern, frame_count):
	"""Load a clip's saved mask PNGs into a [frames, ch, cw] float tensor (0..1)."""
	import numpy as np  # noqa: PLC0415
	from PIL import Image  # noqa: PLC0415

	frames = []
	for i in range(int(frame_count)):
		p = os.path.join(mask_dir, pattern % i)
		if not os.path.isfile(p):
			raise RuntimeError(f"Load Mask: missing mask frame {p}")
		frames.append(np.asarray(Image.open(p).convert("L"), dtype=np.uint8))
	return torch.from_numpy(np.stack(frames)).float() / 255.0


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
					"Where to find the source videos if they've moved since phase "
					"1. Relative to input/ or an absolute path. Empty = trust the "
					"path recorded in each manifest."}),
			},
		}

	RETURN_TYPES = ("ITEM_LIST", "INT")
	RETURN_NAMES = ("item_list", "count")
	OUTPUT_TOOLTIPS = (
		"One item per saved clip — wire to Inspire's ▶Foreach List, then Load "
		"Mask inside the loop.",
		"How many clips were found.",
	)
	FUNCTION = "execute"

	@classmethod
	def IS_CHANGED(cls, subdir=store.DEFAULT_SUBDIR, source_dir=""):
		try:
			root = store.output_root(subdir)
			sig = []
			for name in sorted(os.listdir(root)):
				mp = store.manifest_path(os.path.join(root, name))
				if os.path.isfile(mp):
					sig.append(f"{name}:{os.path.getmtime(mp)}")
			return "|".join(sig)
		except Exception as exc:  # noqa: BLE001
			return repr(exc)

	def execute(self, subdir=store.DEFAULT_SUBDIR, source_dir=""):
		root = store.output_root(subdir)
		src = str(first(source_dir, "")).strip()
		if src and not os.path.isabs(src):
			try:
				import folder_paths  # noqa: PLC0415

				src = os.path.join(folder_paths.get_input_directory(), src)
			except Exception:  # noqa: BLE001
				pass
		items = scan_store(root, src)
		return (list(items), len(items))


@register
class LoadMask(TiNode):
	DISPLAY_NAME = "Load Mask (ti)"
	CATEGORY = "tinode/video"

	@classmethod
	def INPUT_TYPES(cls):
		return {"required": {"item": ("TI_MASK_ITEM", {"tooltip":
			"One item from Load Masks, via ForeachListBegin's `item` output."})}}

	RETURN_TYPES = ("VIDEO", "MASK", "TI_CROP_XFORM", "STRING", "INT")
	RETURN_NAMES = ("video", "mask", "crop_info", "stem", "frame_count")
	OUTPUT_TOOLTIPS = (
		"The original source clip (lazy — re-decoded, not a re-encode).",
		"The saved mask, in crop space ([frames, ch, cw]).",
		"Maps the crop back to the full frame — for Crop By Info and Paste Back.",
		"The clip's key.",
		"Number of frames.",
	)
	FUNCTION = "execute"

	def execute(self, item):
		item = first(item)
		if not isinstance(item, dict):
			raise RuntimeError(
				"Load Mask: `item` must come from Load Masks via ForeachListBegin.")
		stem = item.get("stem", "")
		source_path = item.get("source_path", "")
		if not source_path or not os.path.isfile(source_path):
			raise RuntimeError(
				f"Load Mask: source video for {stem!r} not found "
				f"({source_path!r}). Set Load Masks' source_dir to where the "
				"footage lives now.")
		if VideoFromFile is None:
			raise RuntimeError(
				"Load Mask: ComfyUI's native VIDEO API (comfy_api.latest) is "
				"unavailable.")

		mask = load_mask_frames(
			item["mask_dir"], item.get("mask_pattern", store.MASK_PATTERN),
			item.get("frame_count", 0))
		crop_info = item["crop_info"]
		video = VideoFromFile(source_path)
		return (video, mask, crop_info, stem, int(item.get("frame_count", mask.shape[0])))


@register
class LoadCroppedFrames(TiNode):
	DISPLAY_NAME = "Load Cropped Frames (ti)"
	CATEGORY = "tinode/video"

	@classmethod
	def INPUT_TYPES(cls):
		return {"required": {"item": ("TI_MASK_ITEM", {"tooltip":
			"An item from Load Masks. Reads the lossless crop/ sequence Save Crop "
			"& Mask exported (crop_image must have been connected there)."})}}

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
			raise RuntimeError(
				"Load Cropped Frames: `item` must come from Load Masks.")
		if not item.get("has_crop"):
			raise RuntimeError(
				f"Load Cropped Frames: {item.get('stem','this clip')!r} has no "
				"exported crops. Connect crop_image on Save Crop & Mask in phase 1, or "
				"rebuild the crop with Crop By Info from the video + crop_info.")
		cdir = os.path.join(item["clip_dir"], item.get("crop_subfolder", store.CROP_SUBFOLDER))
		n = int(item.get("frame_count", 0))
		crops = store.load_rgb_sequence(cdir, item.get("crop_pattern", store.CROP_PATTERN), n)
		return (crops, n)
