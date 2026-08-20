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
from .load_masks import build_report, match_stem, resolve_source, _abs_source_dir
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
				"bit_depth": (["16", "8"], {"default": "16", "tooltip":
					"16 keeps the model's output above 8-bit until AFTER the "
					"feathered paste, so the blend ramp cannot band. Crops are "
					"small, so this costs little; use 8 to halve the disk."}),
				"variant": ("STRING", {"default": "", "tooltip":
					"Name this result so alternatives sit side by side — e.g. "
					"`pass1`. Empty writes the default filled/ folder, which is "
					"what Load Clip Fills composites unless told otherwise."}),
			},
		}

	RETURN_TYPES = ("STRING",)
	RETURN_NAMES = ("manifest_path",)
	FUNCTION = "execute"

	def execute(self, item, images, bit_depth="16", variant=""):
		item = first(item)
		imgs = first(images)
		if not isinstance(item, dict):
			raise RuntimeError("Save Filled: `item` must come from Load Masks.")
		variant = str(first(variant, "") or "").strip()
		idir = item["item_dir"]
		fdir = store.filled_dir(idir, variant)
		n = store.save_rgb_sequence(imgs if imgs.dim() == 4 else imgs.unsqueeze(0),
									fdir, store.FILLED_PATTERN, int(first(bit_depth, "16")))
		store.prune_stale(fdir, store.FILLED_PATTERN, n)
		try:
			man = store.read_manifest(idir)
			man[store.filled_flag(variant)] = True
			store.write_manifest(idir, man)
		except Exception:  # noqa: BLE001
			pass
		label = f" [{variant}]" if variant else ""
		print(f"[tinode] Save Filled{label}: {item.get('stem')!r} chunk "
			  f"{item.get('chunk_index')} crop {item.get('crop_index')} — "
			  f"{n} frame(s) -> {fdir}")
		return {"ui": {"ti_filled": [{"stem": item.get("stem"), "frames": n,
									  "variant": variant}]},
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
				"stems": ("STRING", {"default": "", "tooltip":
					"Only these clips — comma/space separated, wildcards allowed. "
					"Empty = every clip."}),
				"require_filled": ("BOOLEAN", {"default": True, "tooltip":
					"On: only clips whose crops are ALL removed — the safe default, "
					"since compositing mid-removal bakes half a result into the "
					"master. Off: composite with whatever IS done and ignore the "
					"rest, for looking at progress before the clip is finished."}),
				"variant": ("STRING", {"default": "", "tooltip":
					"Which saved result counts as done — must match the variant "
					"Load Clip Fills will read."}),
			},
		}

	# `report` is appended so existing graphs keep their link slots.
	RETURN_TYPES = ("ITEM_LIST", "INT", "STRING")
	RETURN_NAMES = ("item_list", "count", "report")
	OUTPUT_TOOLTIPS = (
		"One item per CLIP (grouping all its crops) — for the composite pass.",
		"How many clips matched.",
		"What the store holds and which clips are ready.",
	)
	FUNCTION = "execute"

	@classmethod
	def IS_CHANGED(cls, subdir=store.DEFAULT_SUBDIR, source_dir="", **_filters):
		try:
			import os  # noqa: PLC0415

			root = store.output_root(subdir)
			return "|".join(f"{it['item_dir']}:{os.path.getmtime(store.manifest_path(it['item_dir']))}"
							for it in store.scan_items(root))
		except Exception as exc:  # noqa: BLE001
			return repr(exc)

	def execute(self, subdir=store.DEFAULT_SUBDIR, source_dir="", stems="",
				require_filled=True, variant=""):
		root = store.output_root(subdir)
		src = _abs_source_dir(source_dir)
		found = store.scan_items(root)
		grouped = {}
		for it in found:
			if not match_stem(it.get("stem", ""), stems):
				continue
			grouped.setdefault(it.get("stem", ""), []).append(it)
		flag = store.filled_flag(str(first(variant, "") or "").strip())
		clips = []
		skipped = []
		partial = []
		for stem, crops in sorted(grouped.items()):
			# Order by chunk then crop: the composite pastes them in this order,
			# and a clip's chunks must not be interleaved.
			crops.sort(key=lambda c: (int(c.get("chunk_index", -1)),
									  int(c.get("crop_index", 0))))
			done = [c for c in crops if c.get(flag)]
			if bool(first(require_filled, True)):
				if len(done) != len(crops):
					skipped.append(stem)
					continue
			else:
				# Partial: composite what exists and leave the rest of the frame
				# untouched, rather than refusing or pasting a missing result.
				if not done:
					skipped.append(stem)
					continue
				if len(done) != len(crops):
					partial.append(f"{stem} ({len(done)}/{len(crops)})")
				crops = done
			recorded = crops[0].get("source_path", "")
			clips.append({
				"stem": stem,
				"source_path": resolve_source(recorded, stem, src) or "",
				"fps": crops[0].get("fps", 0.0),
				"crops": crops,
			})
		report = build_report(found, [c for cl in clips for c in cl["crops"]], root)
		if partial:
			report += ("\n  PARTIAL — compositing only the finished crops of: "
					   + ", ".join(partial))
		if skipped:
			report += ("\n  not ready (still being removed): " + ", ".join(skipped))
		print("[tinode] Load Clips:\n" + report)
		if not clips:
			raise RuntimeError(
				f"Load Clips: no clip is ready to composite under {root}"
				+ (f" — {', '.join(skipped)} have no finished crops; run phase 2a"
				   " (or turn require_filled off to composite a partial clip)."
				   if skipped else "."))
		return (clips, len(clips), report)


@register
class LoadClipFills(TiNode):
	DISPLAY_NAME = "Load Clip Fills (ti)"
	CATEGORY = "tinode/video"

	@classmethod
	def INPUT_TYPES(cls):
		return {
			"required": {"item": ("TI_CLIP_ITEM", {"tooltip":
				"A clip item from Load Clips, via Item Cursor or ForeachListBegin."})},
			"optional": {"variant": ("STRING", {"default": "", "tooltip":
				"Which saved result to composite — empty = the default filled/, "
				"or e.g. `pass1` to build the master from that render instead."})},
		}

	RETURN_TYPES = ("VIDEO", "IMAGE", "TI_CROP_XFORM", "MASK", "STRING", "INT", "INT")
	RETURN_NAMES = ("video", "filled_crops", "crop_infos", "masks", "stem",
					"crop_count", "frame_starts")
	# filled_crops / crop_infos / masks / frame_starts are LISTs (one per crop).
	OUTPUT_IS_LIST = (False, True, True, True, False, False, True)
	OUTPUT_TOOLTIPS = (
		"The original source clip.",
		"Each crop's VOID-filled frames (a list).",
		"Each crop's crop_info (a list).",
		"Each crop's mask (a list).",
		"The clip's key.",
		"How many crops this clip has.",
		"Each crop's first frame in the clip — Composite Crops uses these so a "
		"chunk's result lands on the frames it came from.",
	)
	FUNCTION = "execute"

	def execute(self, item, variant=""):
		clip = first(item)
		variant = str(first(variant, "") or "").strip()
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

		crops, infos, masks, starts = [], [], [], []
		for c in clip["crops"]:
			if not c.get(store.filled_flag(variant)):
				raise RuntimeError(
					f"Load Clip Fills: {clip.get('stem')!r} chunk "
					f"{c.get('chunk_index')} crop {c.get('crop_index')} has no "
					f"{store.FILLED_SUBFOLDER + ('_' + variant if variant else '')}/ "
					"— run phase 2a (Save Filled) for that variant first.")
			n = int(c.get("frame_count", 0))
			crops.append(store.load_rgb_sequence(
				store.filled_dir(c["item_dir"], variant),
				c.get("filled_pattern", store.FILLED_PATTERN), n))
			infos.append(c["crop_info"])
			masks.append(store.load_mask_sequence(
				c["mask_dir"], c.get("mask_pattern", store.MASK_PATTERN), n))
			starts.append(int(c.get("frame_start", 0)))
		return (VideoFromFile(source_path), crops, infos, masks,
				clip.get("stem", ""), len(crops), starts)


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
				"frame_starts": ("INT", {"tooltip":
					"Each crop's first frame in the clip (a list, from Load Clip "
					"Fills). Without these a chunk's crops all paste onto the "
					"start of the video."}),
				"feather": ("INT", {"default": 3, "min": 0, "max": 256, "step": 1,
					"tooltip": "Soften the edge that is pasted. With use_mask on "
							   "that is the mask outline; with it off it is the "
							   "crop RECTANGLE, which is what hides the box."}),
				"feather_mode": (["gaussian", "box"], {"default": "gaussian"}),
				"use_mask": ("BOOLEAN", {"default": True, "tooltip":
					"On: write only the masked region, so every other pixel stays "
					"the bit-exact original and there is no rectangle to see.\n"
					"Off: write the WHOLE crop rectangle — takes the model's version "
					"of the surroundings too, which can look more coherent, but any "
					"drift in those pixels shows up as a visible box. Feather then "
					"softens the rectangle edge."}),
			},
		}

	RETURN_TYPES = ("IMAGE",)
	RETURN_NAMES = ("image",)
	FUNCTION = "execute"

	def execute(self, image, crops, crop_infos, masks=None, feather=3,
				feather_mode="gaussian", frame_starts=None, use_mask=True):
		import torch  # noqa: PLC0415

		# image arrives as a 1-element list (the original frames).
		ilist = image if isinstance(image, list) else [image]
		out = torch.cat([s if s.dim() == 4 else s.unsqueeze(0) for s in ilist], dim=0)

		clist = crops if isinstance(crops, list) else [crops]
		infos = crop_infos if isinstance(crop_infos, list) else [crop_infos]
		mlist = masks if isinstance(masks, list) else ([masks] if masks is not None else [])
		slist = frame_starts if isinstance(frame_starts, list) else (
			[frame_starts] if frame_starts is not None else [])
		feather = int(first(feather, 3))
		feather_mode = first(feather_mode, "gaussian")
		use_mask = bool(first(use_mask, True))
		if not use_mask:
			# No mask -> Paste Back writes the whole placed region, i.e. the crop
			# rectangle, feathered at its border.
			mlist = []
			print("[tinode] Composite Crops: pasting the WHOLE crop rectangle "
				  f"(mask ignored), feather {feather}.")

		pb = MaskCropPasteBack()
		for i, (crop, info) in enumerate(zip(clist, infos)):
			m = mlist[i] if i < len(mlist) else None
			off = int(slist[i]) if i < len(slist) and slist[i] is not None else 0
			(out,) = pb.execute(out, crop, info, masks=m, feather=feather,
								feather_mode=feather_mode, frame_offset=off)
		return (out,)
