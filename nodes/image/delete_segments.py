"""Delete Segments — interactively remove segment instances frame by frame."""

from __future__ import annotations

import json

import torch

from ...base import TiNode
from ...registry import register
from ...schema import validate_segments
from .pick_segments import (
	PickSegments, _img_signature, _seg_signature, color_for_id,
)


def parse_deleted_items(raw, sig):
	"""Return valid ``(frame, segment-index)`` deletion keys for this input."""
	try:
		data = json.loads(raw) if raw else {}
	except (TypeError, ValueError):
		return set()
	if isinstance(data, list):                         # tolerate a bare legacy list
		items = data
	elif isinstance(data, dict):
		if data.get("sig") is not None and data.get("sig") != sig:
			return set()
		items = data.get("items", [])
	else:
		return set()
	if not isinstance(items, list):
		return set()

	out = set()
	for item in items:
		try:
			if isinstance(item, dict):
				frame, index = int(item["frame"]), int(item["index"])
			else:
				frame, index = int(item[0]), int(item[1])
		except (KeyError, IndexError, TypeError, ValueError):
			continue
		if frame >= 0 and index >= 0:
			out.add((frame, index))
	return out


@register
class DeleteSegments(TiNode):
	DISPLAY_NAME = "Delete Segments (ti)"
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
				"deleted_items": ("STRING", {"default": "{}"}),
				"current_frame": ("INT", {"default": 0, "min": 0, "max": 999999}),
				"overlay_alpha": ("FLOAT", {
					"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.05,
					"tooltip": "Opacity of remaining segments in the image output.",
				}),
			},
		}

	RETURN_TYPES = ("MASK", "IMAGE", "TI_SAM3_SEGMENTS")
	RETURN_NAMES = ("mask", "image", "segments")
	FUNCTION = "execute"

	def execute(self, image, segments, deleted_items="{}", current_frame=0,
				overlay_alpha=0.5):
		validate_segments(segments)
		if not isinstance(image, torch.Tensor) or image.dim() not in (3, 4):
			raise ValueError(
				"Delete Segments: image must have shape [H,W,C] or [N,H,W,C]."
			)
		imgs = image if image.dim() == 4 else image.unsqueeze(0)
		H, W = int(segments["height"]), int(segments["width"])
		frames = segments.get("frames", [])
		N = int(segments["num_frames"])
		if N < 1 or len(frames) != N:
			raise ValueError(
				f"Delete Segments: segments declares {N} frame(s), but contains "
				f"{len(frames)} per-frame lists."
			)
		if imgs.shape[0] == 1 and N > 1:
			imgs = imgs.expand(N, -1, -1, -1)
		elif imgs.shape[0] != N:
			raise ValueError(
				f"Delete Segments: image has {imgs.shape[0]} frame(s), but segments "
				f"has {N}. Connect the matching source image/video."
			)
		if tuple(imgs.shape[1:3]) != (H, W):
			raise ValueError(
				f"Delete Segments: image resolution {tuple(imgs.shape[1:3])} does "
				f"not match segment resolution {(H, W)}."
			)

		sig = f"{_seg_signature(segments)}_{_img_signature(imgs)}"
		deleted = parse_deleted_items(deleted_items, sig)
		kept_frames = [
			[s for i, s in enumerate(frame) if (f, i) not in deleted]
			for f, frame in enumerate(frames)
		]

		mask_out = torch.zeros((N, H, W), dtype=torch.float32)
		image_out = torch.zeros((N, H, W, 3), dtype=torch.float32)
		for f, frame in enumerate(kept_frames):
			canvas = imgs[f, ..., :3].detach().to(device="cpu", dtype=torch.float32).clone()
			for s in frame:
				x0, y0, x1, y1 = s["bbox"]
				m = s["mask"].detach().to(device="cpu", dtype=torch.float32)
				dst = mask_out[f, y0:y1, x0:x1]
				mask_out[f, y0:y1, x0:x1] = torch.maximum(dst, m)
				color = torch.tensor(color_for_id(int(s["id"])), dtype=torch.float32)
				a = (m * float(overlay_alpha)).unsqueeze(-1)
				region = canvas[y0:y1, x0:x1, :]
				canvas[y0:y1, x0:x1, :] = region * (1 - a) + color * a
			image_out[f] = canvas

		remaining_ids = sorted({int(s["id"]) for frame in kept_frames for s in frame})
		filtered = {
			"num_frames": N, "height": H, "width": W,
			"frames": kept_frames, "ids": remaining_ids,
		}
		result = (mask_out, image_out, filtered)
		manifest = PickSegments._build_assets(self, imgs, segments, sig)
		if manifest is None:
			return result
		return {"ui": {"ti_delete": [manifest]}, "result": result}
