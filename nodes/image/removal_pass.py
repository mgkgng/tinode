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
				"preview": ("BOOLEAN", {"default": True, "tooltip":
					"Play the saved result back in the node. The preview is a "
					"throwaway mp4 in temp/ — the SAVED frames stay the lossless "
					"PNG sequence, so this never touches what gets composited."}),
				"preview_fps": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 240.0, "step": 0.01,
					"tooltip": "0 = take the clip's own fps from the manifest."}),
			},
		}

	RETURN_TYPES = ("STRING",)
	RETURN_NAMES = ("manifest_path",)
	FUNCTION = "execute"

	def execute(self, item, images, bit_depth="16", variant="", preview=True,
				preview_fps=0.0):
		item = first(item)
		imgs = first(images)
		if not isinstance(item, dict):
			raise RuntimeError("Save Filled: `item` must come from Load Masks.")
		variant = str(first(variant, "") or "").strip()
		idir = item["item_dir"]
		fdir = store.filled_dir(idir, variant)
		imgs = imgs if imgs.dim() == 4 else imgs.unsqueeze(0)
		bits = int(first(bit_depth, "16"))
		try:
			n = store.save_rgb_sequence(imgs, fdir, store.FILLED_PATTERN, bits)
		except RuntimeError as exc:
			# The render upstream took real GPU minutes — never throw it away
			# over a missing encoder. Degrade to 8-bit and say so loudly.
			if bits != 16 or "OpenCV" not in str(exc):
				raise
			print("[tinode] Save Filled: !!! cv2 missing — SAVING AT 8-BIT instead "
				  "of 16 so the render isn't lost. Install opencv-python(-headless) "
				  "in this ComfyUI's env for 16-bit fills. !!!")
			bits = 8
			n = store.save_rgb_sequence(imgs, fdir, store.FILLED_PATTERN, bits)
		store.prune_stale(fdir, store.FILLED_PATTERN, n)
		try:
			man = store.read_manifest(idir)
			man[store.filled_flag(variant)] = True
			# The fill's own length: half-rate renders hold every 2nd frame, so
			# readers must not assume the crop's frame_count.
			man[store.filled_frames_key(variant)] = int(n)
			store.write_manifest(idir, man)
		except Exception:  # noqa: BLE001
			pass
		label = f" [{variant}]" if variant else ""
		print(f"[tinode] Save Filled{label}: {item.get('stem')!r} chunk "
			  f"{item.get('chunk_index')} crop {item.get('crop_index')} — "
			  f"{n} frame(s) -> {fdir}")

		ui = {"ti_filled": [{"stem": item.get("stem"), "frames": n, "variant": variant}]}
		if bool(first(preview, True)):
			nth = max(1, int(item.get("every_nth", 1)))
			fps = (float(first(preview_fps, 0.0))
				   or (float(item.get("fps", 0.0)) / nth)
				   or 24.0)
			info = self._preview(imgs, item, variant, fps)
			if info:
				ui["ti_video"] = [info]
		return {"ui": ui, "result": (store.manifest_path(idir),)}

	@staticmethod
	def _preview(imgs, item, variant, fps):
		"""Encode a throwaway mp4 into temp/ so the node can play the result.

		Best-effort: the frames are already saved losslessly by the time this
		runs, so a missing ffmpeg (or any encode hiccup) must not fail the node —
		you just don't get the playback.
		"""
		try:
			import os  # noqa: PLC0415

			import folder_paths  # noqa: PLC0415

			from ._video_io import encode  # noqa: PLC0415

			sub = "ti_filled"
			out_dir = os.path.join(folder_paths.get_temp_directory(), sub)
			os.makedirs(out_dir, exist_ok=True)
			stem = str(item.get("stem", "clip")).replace(os.sep, "_")
			name = (f"{stem}_c{int(item.get('chunk_index', 0)):02d}"
					f"_k{int(item.get('crop_index', 0)):02d}"
					f"{('_' + variant) if variant else ''}.mp4")
			path = os.path.join(out_dir, name)
			# yuv420p and an even size: a preview that a browser refuses to play
			# is worse than none, and odd dimensions break h264.
			even = imgs[:, : imgs.shape[1] // 2 * 2, : imgs.shape[2] // 2 * 2, :]
			encode(even, path, fps=fps, codec="libx264", crf=20, pix_fmt="yuv420p")
			return {"filename": name, "subfolder": sub, "type": "temp",
					"format": "video/mp4", "frames": int(imgs.shape[0]), "fps": fps}
		except Exception as exc:  # noqa: BLE001 — preview is never worth failing a save
			print(f"[tinode] Save Filled: preview unavailable: {exc!r}")
			return None


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
				# APPENDED last: widget values are positional in saved graphs.
				"every_nth": ("INT", {"default": 1, "min": 1, "max": 10, "step": 1,
					"tooltip": "Composite at reduced rate: 2 = every 2nd source "
							   "frame (25fps from 50fps). Must match the rate the "
							   "fills were rendered at in phase 2a — Load Clip "
							   "Fills checks and says so if not. Pair with a Frame "
							   "Stride on the decoded source."}),
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
				require_filled=True, variant="", every_nth=1):
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
				# ONE switch: Load Clip Fills strides masks / validates fills /
				# maps frame starts from this.
				"every_nth": max(1, int(first(every_nth, 1))),
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
			"optional": {
				"variant": ("STRING", {"default": "", "tooltip":
					"Which saved result to composite — empty = the default filled/, "
					"or e.g. `pass1` to build the master from that render instead."}),
				# APPENDED: widget values are positional in saved graphs.
				"require_source": ("BOOLEAN", {"default": True, "tooltip":
					"Off: don't fail when the source VIDEO is missing. For a clip "
					"whose master is an image sequence — load those with Load Clip "
					"Frames instead; the `video` output is then empty."}),
			},
		}

	# source_stride is APPENDED so saved graphs keep their link slots.
	RETURN_TYPES = ("VIDEO", "IMAGE", "TI_CROP_XFORM", "MASK", "STRING", "INT", "INT", "INT")
	RETURN_NAMES = ("video", "filled_crops", "crop_infos", "masks", "stem",
					"crop_count", "frame_starts", "source_stride")
	# filled_crops / crop_infos / masks / frame_starts are LISTs (one per crop).
	OUTPUT_IS_LIST = (False, True, True, True, False, False, True, False)
	OUTPUT_TOOLTIPS = (
		"The original source clip.",
		"Each crop's VOID-filled frames (a list).",
		"Each crop's crop_info (a list).",
		"Each crop's mask (a list).",
		"The clip's key.",
		"How many crops this clip has.",
		"Each crop's first frame in the clip — Composite Crops uses these so a "
		"chunk's result lands on the frames it came from.",
		"The TOTAL source->fill frame stride — wire into Frame Stride's "
		"every_nth_in so the decoded source is reduced by exactly the same "
		"amount, whatever every_nth and the store's own rate are.",
	)
	FUNCTION = "execute"

	def execute(self, item, variant="", require_source=True):
		clip = first(item)
		variant = str(first(variant, "") or "").strip()
		if not isinstance(clip, dict) or "crops" not in clip:
			raise RuntimeError("Load Clip Fills: `item` must come from Load Clips.")
		nth = max(1, int(clip.get("every_nth", 1)))
		source_path = clip.get("source_path", "")
		import os  # noqa: PLC0415

		have_source = bool(source_path) and os.path.isfile(source_path)
		if not have_source and bool(first(require_source, True)):
			raise RuntimeError(
				f"Load Clip Fills: source video for {clip.get('stem')!r} not found "
				f"({source_path!r}). Set Load Clips' source_dir — or, if this clip's "
				"master is an image sequence, turn require_source off and read the "
				"frames with Load Clip Frames.")
		if have_source and VideoFromFile is None:
			raise RuntimeError("Load Clip Fills: native VIDEO API unavailable.")

		crops, infos, masks, starts, strides = [], [], [], [], []
		for c in clip["crops"]:
			if not c.get(store.filled_flag(variant)):
				raise RuntimeError(
					f"Load Clip Fills: {clip.get('stem')!r} chunk "
					f"{c.get('chunk_index')} crop {c.get('crop_index')} has no "
					f"{store.FILLED_SUBFOLDER + ('_' + variant if variant else '')}/ "
					"— run phase 2a (Save Filled) for that variant first.")
			n = int(c.get("frame_count", 0))
			s0 = int(c.get("frame_start", 0))
			# A store already converted to reduced rate (source_every_nth) must
			# not be strided AGAIN — that is the double-speed bug. The requested
			# rate divides by what the store already carries.
			base = max(1, int(c.get("source_every_nth", 1)))
			eff = max(1, nth // base)
			if base > 1 and eff != nth:
				print(f"[tinode] Load Clip Fills: {c.get('stem')!r} store is already "
					  f"1/{base} rate — effective stride {eff}, not {nth}.")
			idx = store.stride_indices(s0, n, eff)
			# Match the fill's recorded length to this rate. A FULL-rate fill can
			# always serve a reduced rate (stride it); a reduced-rate fill can
			# only serve its own rate — its frames don't exist in between.
			k = int(c.get(store.filled_frames_key(variant), n))
			if k == n:
				fill_idx = idx if eff > 1 else None      # stride a full-rate fill
			elif k == len(idx):
				fill_idx = None                          # rendered at this rate
			else:
				raise RuntimeError(
					f"Load Clip Fills: {clip.get('stem')!r} chunk "
					f"{c.get('chunk_index')} crop {c.get('crop_index')}: the fill "
					f"has {k} frame(s) but every_nth={nth} expects {len(idx)} "
					f"(or {n} full-rate). Re-run phase 2a, or set Load Clips' "
					"every_nth to the rate the fills were rendered at.")
			crops.append(store.load_rgb_sequence(
				store.filled_dir(c["item_dir"], variant),
				c.get("filled_pattern", store.FILLED_PATTERN),
				k if fill_idx is None else n, indices=fill_idx))
			infos.append(c["crop_info"])
			masks.append(store.load_mask_sequence(
				c["mask_dir"], c.get("mask_pattern", store.MASK_PATTERN), n,
				indices=idx if eff > 1 else None))
			# Position in the KEPT timeline: global kept index g maps to g//eff,
			# and this chunk's first kept global is s0 + (-s0 % eff).
			starts.append((s0 + (-s0 % eff)) // eff)
			strides.append(base * eff)
		if len(set(strides)) > 1:
			raise RuntimeError(
				f"Load Clip Fills: {clip.get('stem')!r} mixes rates across its "
				f"crops ({sorted(set(strides))}) — convert them to one rate first.")
		video = VideoFromFile(source_path) if have_source else None
		if not have_source:
			print(f"[tinode] Load Clip Fills: {clip.get('stem')!r} has no source "
				  "video — `video` is empty; load the frames with Load Clip Frames.")
		return (video, crops, infos, masks,
				clip.get("stem", ""), len(crops), starts, strides[0] if strides else 1)


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
