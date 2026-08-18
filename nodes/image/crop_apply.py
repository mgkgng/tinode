"""Crop By Info — re-cut the crop a saved TI_CROP_XFORM describes.

The forward of Mask Crop Paste Back, for phase 2 of the removal pipeline: given
the original frames and the crop_info that was saved in phase 1, produce the
exact cropped frames the mask belongs to — no need to re-enter box coordinates,
and guaranteed identical to what phase 1 cropped.

Lossless by construction: it's a plain tensor slice at native scale (no resize),
so the crop is bit-for-bit the original pixels. It handles the non-rescaled
crops that Bbox Crop · Manual and Mask Bbox Crop emit (oy/ox = 0, nh/nw = h/w);
a rescaled/center-fill crop_info is rejected rather than silently resampled,
because reproducing that would mean interpolating — a quality loss.

Pairing mirrors Paste Back: one crop per item, frame-paired with the source
batch (or every item cut from a single source image).
"""

from __future__ import annotations

import torch

from ...base import TiNode, first
from ...registry import register
from ...schema import validate_crop_xform


def _is_rescaled(it):
	return (it["oy"] != 0 or it["ox"] != 0
			or it["nh"] != it["h"] or it["nw"] != it["w"])


@register
class CropByInfo(TiNode):
	DISPLAY_NAME = "Crop By Info (ti)"
	CATEGORY = "tinode/image"
	INPUT_IS_LIST = True

	@classmethod
	def INPUT_TYPES(cls):
		return {
			"required": {
				"image": ("IMAGE", {"tooltip":
					"The ORIGINAL frames the crop_info was built from."}),
				"crop_info": ("TI_CROP_XFORM",),
			},
		}

	RETURN_TYPES = ("IMAGE",)
	RETURN_NAMES = ("crops",)
	FUNCTION = "execute"

	def execute(self, image, crop_info):
		ilist = image if isinstance(image, list) else [image]
		src = torch.cat([s if s.dim() == 4 else s.unsqueeze(0) for s in ilist], dim=0)
		info = first(crop_info)
		validate_crop_xform(info)

		S, H, W, C = src.shape
		iH, iW = int(info.get("H", H)), int(info.get("W", W))
		if (iH, iW) != (H, W):
			raise RuntimeError(
				f"Crop By Info: image is {W}x{H} but crop_info was built for "
				f"{iW}x{iH}. Feed the ORIGINAL frames the crop came from.")

		items = info["items"]
		n = len(items)
		if S != 1 and S != n and n != 1:
			raise RuntimeError(
				f"Crop By Info: {n} crop item(s) but {S} source frame(s) — pass a "
				"single source, one source per item, or a single-item crop_info.")

		count = S if (n == 1 and S > 1) else n
		crops = []
		for i in range(count):
			it = items[0] if n == 1 else items[i]
			if it is None:
				raise RuntimeError(
					f"Crop By Info: crop_info item {i} is empty (no box) — this "
					"clip's crop can't be reproduced. Use a static crop (every "
					"frame the same box) for the removal pass.")
			if _is_rescaled(it):
				raise RuntimeError(
					"Crop By Info: this crop_info is rescaled (center-fill). Crop "
					"By Info only reproduces non-rescaled crops (Bbox Crop · "
					"Manual, Mask Bbox Crop) to stay lossless.")
			f = 0 if S == 1 else i
			y0, x0, h, w = it["y0"], it["x0"], it["h"], it["w"]
			crops.append(src[f, y0:y0 + h, x0:x0 + w, :])

		# Every crop is (h,w); a static box makes them uniform so they stack.
		shapes = {tuple(c.shape) for c in crops}
		if len(shapes) != 1:
			raise RuntimeError(
				f"Crop By Info: crops came out different sizes {shapes}. The "
				"removal pass needs one static crop size per clip.")
		return (torch.stack(crops, 0).contiguous(),)
