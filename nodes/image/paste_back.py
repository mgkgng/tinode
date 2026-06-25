"""Mask Crop Paste Back — composite processed crops back into the source.

Inverts Mask Crop · Center Fill. Take the (inpainted) size x size crops plus
the `crop_info` that node emitted, undo the center+scale per face, resize
each back to its original bbox, and composite onto a copy of the source
image. Blends through the per-face mask region so only the subject is
written, edges feathered.

Crops must stay index-aligned with crop_info (same order/count as the Mask
Crop output) — don't reorder between crop and paste. Faces whose mask was
empty (None transform) are skipped.
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
				"feather": ("INT", {"default": 0, "min": 0, "max": 256, "step": 1}),
			},
		}

	RETURN_TYPES = ("IMAGE",)
	RETURN_NAMES = ("image",)
	FUNCTION = "execute"

	def execute(self, image, crops, crop_info, masks=None, feather=0):
		src = _first(image)                  # [1,H,W,C] or [H,W,C]
		if src.dim() == 4:
			src = src[0]
		info = _first(crop_info)
		feather = int(_first(feather, 0))

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

		for i in range(n):
			it = items[i]
			if it is None:
				continue
			y0, x0, h, w = it["y0"], it["x0"], it["h"], it["w"]
			oy, ox, nh, nw = it["oy"], it["ox"], it["nh"], it["nw"]

			# pull the placed region out of the size x size canvas
			region = crop_batch[i, oy:oy + nh, ox:ox + nw, :]      # [nh,nw,C]
			region = region.permute(2, 0, 1)[None]
			region = F.interpolate(region, size=(h, w), mode="bilinear", align_corners=False)
			region = region[0].permute(1, 2, 0)                    # [h,w,C]

			# alpha: mask region warped back, else solid box
			if mask_batch is not None and i < mask_batch.shape[0]:
				mr = mask_batch[i, oy:oy + nh, ox:ox + nw]         # [nh,nw]
				mr = F.interpolate(mr[None, None], size=(h, w),
								   mode="bilinear", align_corners=False)[0, 0]
			else:
				mr = torch.ones(h, w, dtype=region.dtype)

			if feather > 0:
				k = feather * 2 + 1
				mr = F.avg_pool2d(mr[None, None], kernel_size=k, stride=1,
								  padding=feather)[0, 0]

			a = mr.clamp(0, 1).unsqueeze(-1)                       # [h,w,1]
			dst = out[y0:y0 + h, x0:x0 + w, :]
			out[y0:y0 + h, x0:x0 + w, :] = dst * (1 - a) + region * a

		return (out[None],)
