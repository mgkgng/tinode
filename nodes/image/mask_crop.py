"""Mask Crop · Center Fill — extract masked regions onto square black canvases.

Given a source IMAGE and one or more MASKs, crop each mask's bounding box,
zero everything outside the mask (black background), scale the crop to fill
the canvas minus padding (aspect preserved), and center it on a size x size
frame. A batch of N masks produces an IMAGE batch of N frames.

Built for crowd -> per-face pipelines: feed the masks of detected faces and
get uniform, centered, black-bg face crops ready for restore/embed.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from ...base import TiNode
from ...registry import register


@register
class MaskCropCenterFill(TiNode):
	DISPLAY_NAME = "Mask Crop · Center Fill (ti)"
	CATEGORY = "tinode/image"

	@classmethod
	def INPUT_TYPES(cls):
		return {
			"required": {
				"image": ("IMAGE",),
				"mask": ("MASK",),
				"size": ("INT", {"default": 640, "min": 64, "max": 4096, "step": 8}),
				"padding": ("INT", {"default": 32, "min": 0, "max": 2048, "step": 1}),
			},
			"optional": {
				"apply_mask": ("BOOLEAN", {"default": True}),
				"threshold": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),
			},
		}

	RETURN_TYPES = ("IMAGE", "MASK", "TI_CROP_XFORM")
	RETURN_NAMES = ("images", "masks", "crop_info")
	FUNCTION = "execute"

	# Consume the whole mask set in one call. A LIST source (SEGSToMaskList)
	# would otherwise run this once per face -> a separate 1-item crop_info per
	# face instead of one crop_info with N items. With this, collapse the mask
	# list to one [N,H,W] batch and emit a single aligned set.
	INPUT_IS_LIST = True

	@staticmethod
	def _first(v, default=None):
		if isinstance(v, list):
			return v[0] if v else default
		return v

	def execute(self, image, mask, size, padding, apply_mask=True, threshold=0.5):
		img = self._first(image)             # source image
		if img.dim() == 4:
			img = img[0]                     # [H,W,C]
		mlist = mask if isinstance(mask, list) else [mask]
		mask = torch.cat(
			[m if m.dim() == 3 else m.unsqueeze(0) for m in mlist], dim=0
		)
		size = int(self._first(size, 512))
		padding = int(self._first(padding, 32))
		apply_mask = bool(self._first(apply_mask, True))
		threshold = float(self._first(threshold, 0.5))

		H, W, C = img.shape
		inner = max(1, size - 2 * padding)
		out = []
		out_masks = []
		# Per-face transform so the crop can be pasted back later. None where
		# the mask was empty (no face), keeping index alignment with the batch.
		items = []

		for m in mask:
			# match mask resolution to image if they differ
			if m.shape[0] != H or m.shape[1] != W:
				m = F.interpolate(m[None, None], size=(H, W),
								  mode="bilinear", align_corners=False)[0, 0]

			ys, xs = torch.where(m > threshold)
			if ys.numel() == 0:
				out.append(torch.zeros(size, size, C))
				out_masks.append(torch.zeros(size, size))
				items.append(None)
				continue

			y0, y1 = int(ys.min()), int(ys.max()) + 1
			x0, x1 = int(xs.min()), int(xs.max()) + 1
			crop = img[y0:y1, x0:x1, :].clone()             # [h,w,C]
			mc = m[y0:y1, x0:x1].clamp(0, 1)                # [h,w]
			if apply_mask:
				crop = crop * mc.unsqueeze(-1)              # outside subject -> black

			h, w = crop.shape[0], crop.shape[1]
			scale = inner / max(h, w)                       # fit longest side, keep aspect
			nh, nw = max(1, round(h * scale)), max(1, round(w * scale))

			crop = crop.permute(2, 0, 1)[None]              # [1,C,h,w]
			crop = F.interpolate(crop, size=(nh, nw), mode="bilinear", align_corners=False)
			crop = crop[0].permute(1, 2, 0)                 # [nh,nw,C]

			# warp the mask through the SAME crop->resize->center transform
			mr = F.interpolate(mc[None, None], size=(nh, nw),
							   mode="bilinear", align_corners=False)[0, 0]  # [nh,nw]

			canvas = torch.zeros(size, size, C)
			mcanvas = torch.zeros(size, size)
			oy, ox = (size - nh) // 2, (size - nw) // 2
			canvas[oy:oy + nh, ox:ox + nw, :] = crop
			mcanvas[oy:oy + nh, ox:ox + nw] = mr
			out.append(canvas)
			out_masks.append(mcanvas)
			items.append({
				"y0": y0, "x0": x0, "h": h, "w": w,
				"oy": oy, "ox": ox, "nh": nh, "nw": nw,
			})

		crop_info = {"H": H, "W": W, "C": C, "size": size, "items": items}
		return (torch.stack(out, 0), torch.stack(out_masks, 0), crop_info)
