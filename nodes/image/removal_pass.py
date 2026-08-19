"""Phase 2a/2b nodes for multi-crop object removal.

Phase 2a (flat remove): for each crop, VOID fills it; Save Filled writes the
result back into that crop's folder.

Phase 2b (composite): Load Clips groups the store by clip; per clip, Load Clip
Fills hands back the source video plus a LIST of every crop's filled frames /
crop_info / mask; Composite Crops pastes them all onto the original and you save
once. Each crop is pasted independently through its own mask, so different-sized
crops compose cleanly and everything outside the masks stays the untouched
original.
"""

from __future__ import annotations

from ...base import TiNode, first
from ...registry import register
from . import _mask_store as store
from .load_masks import resolve_source, _abs_source_dir
from .paste_back import MaskCropPasteBack

try:
	from comfy_api.latest import InputImpl  # noqa: PLC0415

	VideoFromFile = InputImpl.VideoFromFile
except Exception:  # noqa: BLE001
	VideoFromFile = None


@register
class SaveFilled(TiNode):
	DISPLAY_NAME = "Save Filled (ti)"
	CATEGORY = "tinode/video"
	OUTPUT_NODE = True

	@classmethod
	def INPUT_TYPES(cls):
		return {
			"required": {
				"item": ("TI_MASK_ITEM", {"tooltip": "The crop item being removed."}),
				"images": ("IMAGE", {"tooltip": "VOID's filled crop (original length)."}),
			},
			"optional": {
				"bit_depth": (["16", "8"], {"default": "16"}),
			},
		}

	RETURN_TYPES = ("STRING",)
	RETURN_NAMES = ("manifest_path",)
	FUNCTION = "execute"

	def execute(self, item, images, bit_depth="16"):
		item = first(item)
		imgs = first(images)
		if not isinstance(item, dict):
			raise RuntimeError("Save Filled: `item` must come from Load Masks.")
		idir = item["item_dir"]
		fdir = store.filled_dir(idir)
		n = store.save_rgb_sequence(imgs if imgs.dim() == 4 else imgs.unsqueeze(0),
									fdir, store.FILLED_PATTERN, int(first(bit_depth, "16")))
		store.prune_stale(fdir, store.FILLED_PATTERN, n)
		try:
			man = store.read_manifest(idir)
			man["has_filled"] = True
			store.write_manifest(idir, man)
		except Exception:  # noqa: BLE001
			pass
		print(f"[tinode] Save Filled: {item.get('stem')!r} crop "
			  f"{item.get('crop_index')} — {n} frame(s) -> {fdir}")
		return {"ui": {"ti_filled": [{"stem": item.get("stem"), "frames": n}]},
				"result": (store.manifest_path(idir),)}


@register
class LoadClips(TiNode):
	DISPLAY_NAME = "Load Clips (ti)"
	CATEGORY = "tinode/video"

	@classmethod
	def INPUT_TYPES(cls):
		return {
			"required": {
				"subdir": ("STRING", {"default": store.DEFAULT_SUBDIR}),
			},
			"optional": {
				"source_dir": ("STRING", {"default": "", "tooltip":
					"Where the source videos live now, if moved (input/-relative or absolute)."}),
			},
		}

	RETURN_TYPES = ("ITEM_LIST", "INT")
	RETURN_NAMES = ("item_list", "count")
	OUTPUT_TOOLTIPS = (
		"One item per CLIP (grouping all its crops) — for the composite pass.",
		"How many clips.",
	)
	FUNCTION = "execute"

	@classmethod
	def IS_CHANGED(cls, subdir=store.DEFAULT_SUBDIR, source_dir=""):
		try:
			import os  # noqa: PLC0415

			root = store.output_root(subdir)
			return "|".join(f"{it['item_dir']}:{os.path.getmtime(store.manifest_path(it['item_dir']))}"
							for it in store.scan_items(root))
		except Exception as exc:  # noqa: BLE001
			return repr(exc)

	def execute(self, subdir=store.DEFAULT_SUBDIR, source_dir=""):
		root = store.output_root(subdir)
		src = _abs_source_dir(source_dir)
		grouped = {}
		for it in store.scan_items(root):
			grouped.setdefault(it.get("stem", ""), []).append(it)
		clips = []
		for stem, crops in grouped.items():
			crops.sort(key=lambda c: int(c.get("crop_index", 0)))
			recorded = crops[0].get("source_path", "")
			clips.append({
				"stem": stem,
				"source_path": resolve_source(recorded, stem, src) or "",
				"fps": crops[0].get("fps", 0.0),
				"crops": crops,
			})
		if not clips:
			raise RuntimeError(f"Load Clips: no saved clips under {root}")
		return (clips, len(clips))


@register
class LoadClipFills(TiNode):
	DISPLAY_NAME = "Load Clip Fills (ti)"
	CATEGORY = "tinode/video"

	@classmethod
	def INPUT_TYPES(cls):
		return {"required": {"item": ("TI_CLIP_ITEM", {"tooltip":
			"A clip item from Load Clips, via ForeachListBegin's `item`."})}}

	RETURN_TYPES = ("VIDEO", "IMAGE", "TI_CROP_XFORM", "MASK", "STRING", "INT")
	RETURN_NAMES = ("video", "filled_crops", "crop_infos", "masks", "stem", "crop_count")
	# filled_crops / crop_infos / masks are LISTs (one per crop); the rest scalar.
	OUTPUT_IS_LIST = (False, True, True, True, False, False)
	OUTPUT_TOOLTIPS = (
		"The original source clip.",
		"Each crop's VOID-filled frames (a list).",
		"Each crop's crop_info (a list).",
		"Each crop's mask (a list).",
		"The clip's key.",
		"How many crops this clip has.",
	)
	FUNCTION = "execute"

	def execute(self, item):
		clip = first(item)
		if not isinstance(clip, dict) or "crops" not in clip:
			raise RuntimeError("Load Clip Fills: `item` must come from Load Clips.")
		source_path = clip.get("source_path", "")
		import os  # noqa: PLC0415

		if not source_path or not os.path.isfile(source_path):
			raise RuntimeError(
				f"Load Clip Fills: source video for {clip.get('stem')!r} not found "
				f"({source_path!r}). Set Load Clips' source_dir.")
		if VideoFromFile is None:
			raise RuntimeError("Load Clip Fills: native VIDEO API unavailable.")

		crops, infos, masks = [], [], []
		for c in clip["crops"]:
			if not c.get("has_filled"):
				raise RuntimeError(
					f"Load Clip Fills: {clip.get('stem')!r} crop "
					f"{c.get('crop_index')} has no filled/ — run phase 2a (Save Filled) first.")
			n = int(c.get("frame_count", 0))
			crops.append(store.load_rgb_sequence(
				store.filled_dir(c["item_dir"]), c.get("filled_pattern", store.FILLED_PATTERN), n))
			infos.append(c["crop_info"])
			masks.append(store.load_mask_sequence(
				c["mask_dir"], c.get("mask_pattern", store.MASK_PATTERN), n))
		return (VideoFromFile(source_path), crops, infos, masks, clip.get("stem", ""), len(crops))


@register
class CompositeCrops(TiNode):
	DISPLAY_NAME = "Composite Crops (ti)"
	CATEGORY = "tinode/image"
	INPUT_IS_LIST = True

	@classmethod
	def INPUT_TYPES(cls):
		return {
			"required": {
				"image": ("IMAGE", {"tooltip": "The ORIGINAL full frames."}),
				"crops": ("IMAGE", {"tooltip": "Each crop's filled frames (a list)."}),
				"crop_infos": ("TI_CROP_XFORM", {"tooltip": "Each crop's crop_info (a list)."}),
			},
			"optional": {
				"masks": ("MASK", {"tooltip": "Each crop's mask (a list)."}),
				"feather": ("INT", {"default": 3, "min": 0, "max": 256, "step": 1}),
				"feather_mode": (["gaussian", "box"], {"default": "gaussian"}),
			},
		}

	RETURN_TYPES = ("IMAGE",)
	RETURN_NAMES = ("image",)
	FUNCTION = "execute"

	def execute(self, image, crops, crop_infos, masks=None, feather=3, feather_mode="gaussian"):
		import torch  # noqa: PLC0415

		# image arrives as a 1-element list (the original frames).
		ilist = image if isinstance(image, list) else [image]
		out = torch.cat([s if s.dim() == 4 else s.unsqueeze(0) for s in ilist], dim=0)

		clist = crops if isinstance(crops, list) else [crops]
		infos = crop_infos if isinstance(crop_infos, list) else [crop_infos]
		mlist = masks if isinstance(masks, list) else ([masks] if masks is not None else [])
		feather = int(first(feather, 3))
		feather_mode = first(feather_mode, "gaussian")

		pb = MaskCropPasteBack()
		for i, (crop, info) in enumerate(zip(clist, infos)):
			m = mlist[i] if i < len(mlist) else None
			(out,) = pb.execute(out, crop, info, masks=m, feather=feather, feather_mode=feather_mode)
		return (out,)
