"""Add Segments — manually draw extra segments onto a TI_SAM3_SEGMENTS stream.

The companion to Pick Segments: where that one lets you drop detections SAM3
found, this one lets you add regions it missed. Same in-node editor feel — step
through frames with the arrows, see the existing segments as dim reference boxes,
and drag a new bounding box to add a segment on the current frame (right-click a
box you drew to remove it). Manual boxes are per-frame (you draw them on the
frame you're looking at), each a filled-rectangle mask.

Takes the same inputs as Pick Segments — image + segments — and emits the same
three outputs, so the two are drop-in interchangeable and chain in either order:
  mask      union of every outgoing segment, [frames,H,W]
  image     the source frames with those segments colorized
  segments  the incoming segments plus your manual boxes

Manual boxes get ids from a high base (>= 1,000,000) so they never collide with
SAM3 track ids. They are stamped with a signature of the image+segments they were
drawn against; when that input changes the boxes are stale and are dropped
automatically (both here and in the editor) rather than being re-applied to a
different clip.
"""

from __future__ import annotations

import json
import os

import torch

from ...base import TiNode
from ...registry import register
from ...schema import validate_segments
from .pick_segments import (
	_PREVIEW_MAX_SIDE, _img_signature, _seg_signature, color_for_id,
	prune_asset_cache,
)

# Manual ids start here so they never clash with SAM3 track ids.
_MANUAL_ID_BASE = 1_000_000


@register
class AddSegments(TiNode):
	DISPLAY_NAME = "Add Segments (ti)"
	CATEGORY = "tinode/image"
	OUTPUT_NODE = True

	@classmethod
	def INPUT_TYPES(cls):
		return {
			"required": {
				"image": ("IMAGE",),
				"segments": ("TI_SAM3_SEGMENTS",),
			},
			"optional": {
				# {"sig": <input signature>, "items": [{"id","frame","bbox"}...]}
				# bboxes in SOURCE pixels. Hidden and driven by the editor; saved
				# with the workflow. Items whose sig no longer matches the current
				# input are dropped (see _manual_items).
				"manual_segments": ("STRING", {"default": "{}"}),
				"current_frame": ("INT", {"default": 0, "min": 0, "max": 999999}),
				"overlay_alpha": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.05,
					"tooltip": "Opacity of the colorized segments in the image output."}),
			},
		}

	RETURN_TYPES = ("MASK", "IMAGE", "TI_SAM3_SEGMENTS")
	RETURN_NAMES = ("mask", "image", "segments")
	FUNCTION = "execute"

	@staticmethod
	def _manual_items(raw, sig):
		"""Decode the manual-box widget, dropping boxes drawn against other input.

		Accepts the current {"sig","items"} form and the older bare-list form (which
		carries no signature, so it is trusted as-is).
		"""
		try:
			data = json.loads(raw) if raw else {}
		except (ValueError, TypeError):
			return []
		if isinstance(data, list):
			return data
		if not isinstance(data, dict):
			return []
		stored = data.get("sig")
		if stored is not None and stored != sig:
			return []          # input changed -> previous boxes are stale
		items = data.get("items")
		return items if isinstance(items, list) else []

	def execute(self, image, segments, manual_segments="{}", current_frame=0,
				overlay_alpha=0.5):
		validate_segments(segments)
		imgs = image if image.dim() == 4 else image.unsqueeze(0)  # [N,H,W,3]
		H = int(segments.get("height", imgs.shape[1]))
		W = int(segments.get("width", imgs.shape[2]))
		in_frames = segments.get("frames", [])
		N = int(segments.get("num_frames", len(in_frames)))

		# Identity of this input — manual boxes are only valid against it.
		sig = f"{_seg_signature(segments)}_{_img_signature(imgs)}"
		manual = self._manual_items(manual_segments, sig)

		if imgs.shape[0] != N and imgs.shape[0] == 1:
			imgs = imgs.expand(N, -1, -1, -1)

		# Bucket the manual boxes by frame, each a filled-rectangle segment.
		manual_by_frame = {}
		for m in manual:
			try:
				f = int(m["frame"])
				x0, y0, x1, y1 = (int(round(v)) for v in m["bbox"])
			except (KeyError, TypeError, ValueError):
				continue
			x0, x1 = max(0, min(x0, W)), max(0, min(x1, W))
			y0, y1 = max(0, min(y0, H)), max(0, min(y1, H))
			if x1 - x0 < 1 or y1 - y0 < 1 or not (0 <= f < N):
				continue
			manual_by_frame.setdefault(f, []).append({
				"id": int(m.get("id", _MANUAL_ID_BASE)),
				"bbox": [x0, y0, x1, y1],
				"conf": 1.0,
				"mask": torch.ones(y1 - y0, x1 - x0, dtype=torch.uint8),
			})

		out_frames = []
		for f in range(N):
			base = list(in_frames[f]) if f < len(in_frames) else []
			out_frames.append(base + manual_by_frame.get(f, []))

		# Compose mask + colorized image over every outgoing segment, matching
		# what Pick Segments emits so the two are interchangeable.
		mask_out = torch.zeros((N, H, W), dtype=torch.float32)
		image_out = torch.zeros((N, H, W, 3), dtype=torch.float32)
		for f in range(N):
			base = imgs[f] if f < imgs.shape[0] else torch.zeros(H, W, 3)
			if base.shape[0] != H or base.shape[1] != W:
				base = torch.zeros(H, W, 3)
			canvas = base.clone()
			for s in out_frames[f]:
				x0, y0, x1, y1 = s["bbox"]
				m = s["mask"].to(torch.float32)
				dst = mask_out[f, y0:y1, x0:x1]
				mask_out[f, y0:y1, x0:x1] = torch.maximum(dst, m)
				r, g, b = color_for_id(s["id"])
				color = torch.tensor([r, g, b], dtype=torch.float32)
				a = (m * overlay_alpha).unsqueeze(-1)
				region = canvas[y0:y1, x0:x1, :]
				canvas[y0:y1, x0:x1, :] = region * (1 - a) + color * a
			image_out[f] = canvas

		manual_ids = {s["id"] for segs in manual_by_frame.values() for s in segs}
		out_segments = {
			"num_frames": N, "height": H, "width": W,
			"frames": out_frames,
			"ids": sorted(set(segments.get("ids", [])) | manual_ids),
		}
		result = (mask_out, image_out, out_segments)

		manifest = self._build_assets(imgs, segments, sig)
		if manifest is None:
			return result
		return {"ui": {"ti_add": [manifest]}, "result": result}

	# ------------------------------------------------------------------ assets
	def _build_assets(self, imgs, segments, sig):
		"""Save per-frame source previews and return the editor manifest.

		Only the source frames + existing boxes are needed here (no label maps —
		Add Segments never picks existing segments by pixel). Returns None on any
		failure so the outputs still flow.
		"""
		try:
			import numpy as np  # noqa: PLC0415
			from PIL import Image  # noqa: PLC0415
			import folder_paths  # noqa: PLC0415

			H = int(segments["height"])
			W = int(segments["width"])
			frames = segments.get("frames", [])
			N = int(segments.get("num_frames", len(frames)))

			scale = min(1.0, _PREVIEW_MAX_SIDE / max(H, W))
			pw, ph = max(1, round(W * scale)), max(1, round(H * scale))

			# Same folder scheme as Pick Segments so the source frames are shared.
			root = os.path.join(folder_paths.get_temp_directory(), "ti_pick", sig)
			os.makedirs(root, exist_ok=True)
			subfolder = os.path.join("ti_pick", sig)
			prune_asset_cache(os.path.dirname(root), sig)

			manifest_frames = []
			for f in range(N):
				frame = frames[f] if f < len(frames) else []
				src_name = f"src_{f:05d}.png"
				src_path = os.path.join(root, src_name)
				if not os.path.exists(src_path):
					base = imgs[f] if f < imgs.shape[0] else torch.zeros(H, W, 3)
					arr = (base.clamp(0, 1).cpu().numpy() * 255).astype(np.uint8)
					Image.fromarray(arr[..., :3]).resize((pw, ph), Image.BILINEAR).save(
						src_path, compress_level=1)

				manifest_frames.append({
					"src": {"filename": src_name, "subfolder": subfolder, "type": "temp"},
					"segs": [{
						"id": s["id"],
						"bbox": [round(s["bbox"][0] * scale), round(s["bbox"][1] * scale),
								 round(s["bbox"][2] * scale), round(s["bbox"][3] * scale)],
					} for s in frame],
				})

			return {
				"sig": sig, "num_frames": N, "pw": pw, "ph": ph,
				"full_w": W, "full_h": H, "frames": manifest_frames,
			}
		except Exception as exc:  # noqa: BLE001 — editor is best-effort
			print(f"[tinode] Add Segments editor assets unavailable: {exc!r}")
			return None
