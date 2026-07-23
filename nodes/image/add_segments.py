"""Add Segments — manually draw extra segments onto a TI_SAM3_SEGMENTS stream.

The companion to Pick Segments: where that one lets you drop detections SAM3
found, this one lets you add regions it missed. Same in-node editor feel — step
through frames with the arrows, see the existing segments as dim reference boxes,
and drag a new bounding box to add a segment on the current frame (right-click a
box you drew to remove it). Manual boxes are per-frame (you draw them on the
frame you're looking at), each a filled-rectangle mask.

Takes the same inputs as Pick Segments — image + segments — and outputs the
combined TI_SAM3_SEGMENTS: every incoming segment, plus your manual boxes on the
frames you drew them. Chain it before or after Pick Segments.

Manual boxes get ids from a high base (>= 1,000,000) so they never collide with
SAM3 track ids. Editor assets (the frame previews) are best-effort / lazily
imported, so the combined segments still flow on a headless host.
"""

from __future__ import annotations

import json
import os

import torch

from ...base import TiNode
from ...registry import register
from .pick_segments import _PREVIEW_MAX_SIDE, _img_signature, _seg_signature, color_for_id

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
				# JSON list of manual boxes [{"id","frame","bbox":[x0,y0,x1,y1]}...]
				# in SOURCE pixels. Hidden and driven by the editor; saved with the
				# workflow.
				"manual_segments": ("STRING", {"default": "[]"}),
				"current_frame": ("INT", {"default": 0, "min": 0, "max": 999999}),
			},
		}

	RETURN_TYPES = ("TI_SAM3_SEGMENTS",)
	RETURN_NAMES = ("segments",)
	FUNCTION = "execute"

	def execute(self, image, segments, manual_segments="[]", current_frame=0):
		imgs = image if image.dim() == 4 else image.unsqueeze(0)  # [N,H,W,3]
		H = int(segments.get("height", imgs.shape[1]))
		W = int(segments.get("width", imgs.shape[2]))
		in_frames = segments.get("frames", [])
		N = int(segments.get("num_frames", len(in_frames)))

		try:
			manual = json.loads(manual_segments) if manual_segments else []
		except (ValueError, TypeError):
			manual = []

		# Bucket the manual boxes by frame, converting each to a segment dict with
		# a filled-rectangle mask (clamped to the frame, min 1px).
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
			seg = {
				"id": int(m.get("id", _MANUAL_ID_BASE)),
				"bbox": [x0, y0, x1, y1],
				"conf": 1.0,
				"mask": torch.ones(y1 - y0, x1 - x0, dtype=torch.uint8),
			}
			manual_by_frame.setdefault(f, []).append(seg)

		out_frames = []
		for f in range(N):
			base = list(in_frames[f]) if f < len(in_frames) else []
			out_frames.append(base + manual_by_frame.get(f, []))

		manual_ids = {s["id"] for segs in manual_by_frame.values() for s in segs}
		out_ids = sorted(set(segments.get("ids", [])) | manual_ids)
		out_segments = {
			"num_frames": N, "height": H, "width": W,
			"frames": out_frames, "ids": out_ids,
		}
		result = (out_segments,)

		manifest = self._build_assets(imgs, segments)
		if manifest is None:
			return result
		return {"ui": {"ti_add": [manifest]}, "result": result}

	# ------------------------------------------------------------------ assets
	def _build_assets(self, imgs, segments):
		"""Save per-frame source previews and return the editor manifest.

		Only the source frames + existing boxes are needed here (no label maps —
		Add Segments never picks existing segments by pixel). Returns None on any
		failure so the combined segments still flow.
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
			sig = f"{_seg_signature(segments)}_{_img_signature(imgs)}"
			root = os.path.join(folder_paths.get_temp_directory(), "ti_pick", sig)
			os.makedirs(root, exist_ok=True)
			subfolder = os.path.join("ti_pick", sig)

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

				segs_meta = [{
					"id": s["id"],
					"bbox": [round(s["bbox"][0] * scale), round(s["bbox"][1] * scale),
							 round(s["bbox"][2] * scale), round(s["bbox"][3] * scale)],
				} for s in frame]
				manifest_frames.append({
					"src": {"filename": src_name, "subfolder": subfolder, "type": "temp"},
					"segs": segs_meta,
				})

			return {
				"num_frames": N, "pw": pw, "ph": ph, "full_w": W, "full_h": H,
				"frames": manifest_frames,
			}
		except Exception as exc:  # noqa: BLE001 — editor is best-effort
			print(f"[tinode] Add Segments editor assets unavailable: {exc!r}")
			return None
