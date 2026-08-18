"""Mask Crop Paste Back — composite processed crops back into the source.

Inverts Mask Crop · Center Fill. Take the (inpainted) size x size crops plus
the `crop_info` that node emitted, undo the center+scale per face, resize
each back to its original bbox, and composite onto a copy of the source
image. Blends through the per-face mask region so only the subject is
written, edges feathered.

Crops must stay index-aligned with crop_info (same order/count as the Mask
Crop output) — don't reorder between crop and paste. Faces whose mask was
empty (None transform) are skipped.

A single source image takes every crop composited onto it (N faces -> one
photo). A source batch of the same length as the crops is paired frame to
frame instead (N video frames -> N crops), which is what Mask Bbox Crop emits.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from ...base import TiNode
from ...registry import register


def _first(v, default=None):
	if isinstance(v, list):
		return v[0] if v else default
	return v


def _gaussian_kernel1d(radius: int, device, dtype):
	"""Normalised 1-D Gaussian of length 2*radius+1 (sigma = radius/2)."""
	sigma = max(1e-6, radius / 2.0)
	xs = torch.arange(-radius, radius + 1, device=device, dtype=dtype)
	k = torch.exp(-(xs * xs) / (2 * sigma * sigma))
	return k / k.sum()


def _feather_alpha(full, radius: int, mode: str):
	"""Blur a [1,1,H,W] alpha plane by `radius`, replicate-padded at the frame edge.

	Box (avg_pool) is cheap; gaussian gives a smoother, more professional falloff
	at high resolution. Replicate padding stops a crop that touches the IMAGE
	edge from fading against imaginary zero-alpha pixels beyond the canvas.
	"""
	if mode == "gaussian":
		k1 = _gaussian_kernel1d(radius, full.device, full.dtype)
		full = F.pad(full, (radius, radius, radius, radius), mode="replicate")
		full = F.conv2d(full, k1.view(1, 1, 1, -1))       # horizontal
		full = F.conv2d(full, k1.view(1, 1, -1, 1))       # vertical
		return full
	k = radius * 2 + 1
	full = F.pad(full, (radius, radius, radius, radius), mode="replicate")
	return F.avg_pool2d(full, kernel_size=k, stride=1)


@register
class MaskCropPasteBack(TiNode):
	DISPLAY_NAME = "Mask Crop Paste Back (ti)"
	CATEGORY = "tinode/image"
	INPUT_IS_LIST = True

	@classmethod
	def INPUT_TYPES(cls):
		return {
			"required": {
				"image": ("IMAGE",),
				"crops": ("IMAGE",),
				"crop_info": ("TI_CROP_XFORM",),
			},
			"optional": {
				"masks": ("MASK",),
				"feather": ("INT", {"default": 0, "min": 0, "max": 256, "step": 1,
					"tooltip": "Soften the mask edge by this many pixels so the "
							   "removed region blends into the original."}),
				"feather_mode": (["gaussian", "box"], {"default": "gaussian",
					"tooltip": "Gaussian = smoother falloff (best for 4K); box = "
							   "the older, harder average blur."}),
			},
		}

	RETURN_TYPES = ("IMAGE",)
	RETURN_NAMES = ("image",)
	FUNCTION = "execute"

	def execute(self, image, crops, crop_info, masks=None, feather=0, feather_mode="gaussian"):
		ilist = image if isinstance(image, list) else [image]
		src = torch.cat(
			[s if s.dim() == 4 else s.unsqueeze(0) for s in ilist], dim=0
		)                                    # [S,H,W,C]
		info = _first(crop_info)
		feather = int(_first(feather, 0))
		feather_mode = _first(feather_mode, "gaussian")

		# crops can arrive as a list (per-frame) or a single batch tensor.
		clist = crops if isinstance(crops, list) else [crops]
		crop_batch = torch.cat(
			[c if c.dim() == 4 else c.unsqueeze(0) for c in clist], dim=0
		)

		mask_batch = None
		if masks is not None:
			mlist = masks if isinstance(masks, list) else [masks]
			if any(m is not None for m in mlist):
				mask_batch = torch.cat(
					[m if m.dim() == 3 else m.unsqueeze(0) for m in mlist], dim=0
				)

		out = src.clone()
		items = info["items"]
		n = min(len(items), crop_batch.shape[0])

		S = out.shape[0]
		if S != 1 and S != n:
			raise RuntimeError(
				f"Cannot paste {n} crop(s) into a {S}-frame source: pass a single "
				f"source image, or one source frame per crop."
			)

		for i in range(n):
			it = items[i]
			if it is None:
				continue
			f = 0 if S == 1 else i          # every crop onto one photo, or frame-paired
			y0, x0, h, w = it["y0"], it["x0"], it["h"], it["w"]
			oy, ox, nh, nw = it["oy"], it["ox"], it["nh"], it["nw"]

			# pull the placed region out of the size x size canvas
			region = crop_batch[i, oy:oy + nh, ox:ox + nw, :]      # [nh,nw,C]
			rescaled = (nh != h or nw != w)
			if rescaled:
				region = region.permute(2, 0, 1)[None]
				region = F.interpolate(region, size=(h, w), mode="bilinear", align_corners=False)
				region = region[0].permute(1, 2, 0)                # [h,w,C]
			# else: no resample — the fill is bit-exact the generated pixels.

			# alpha: mask region warped back, else solid box
			if mask_batch is not None and i < mask_batch.shape[0]:
				mr = mask_batch[i, oy:oy + nh, ox:ox + nw]         # [nh,nw]
				if rescaled:
					mr = F.interpolate(mr[None, None], size=(h, w),
									   mode="bilinear", align_corners=False)[0, 0]
			else:
				mr = torch.ones(h, w, dtype=region.dtype, device=region.device)

			if feather > 0:
				# Blur in full-frame coordinates so an internal crop boundary's
				# real zero-alpha seam still feathers, while a crop that touches
				# the IMAGE edge is replicate-padded and does not fade to nothing.
				full = torch.zeros(
					(1, 1, out.shape[1], out.shape[2]),
					dtype=mr.dtype, device=mr.device,
				)
				full[0, 0, y0:y0 + h, x0:x0 + w] = mr
				full = _feather_alpha(full, feather, feather_mode)
				mr = full[0, 0, y0:y0 + h, x0:x0 + w]

			a = mr.clamp(0, 1).unsqueeze(-1)                       # [h,w,1]
			dst = out[f, y0:y0 + h, x0:x0 + w, :]
			out[f, y0:y0 + h, x0:x0 + w, :] = dst * (1 - a) + region * a

		return (out,)
