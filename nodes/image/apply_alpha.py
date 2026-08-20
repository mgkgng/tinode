"""Apply Mask as Alpha — attach a mask to an image as its 4th (alpha) channel.

For models that take a reference clip with the region-to-change carried in the
alpha channel (e.g. MiniMax H3 reference-to-video used for removal): the RGB
stays the crop, the alpha is the mask. Output is [N,H,W,4].

`invert` flips the mask first, for the opposite convention (alpha marks what to
KEEP rather than what to change).
"""

from __future__ import annotations

import torch

from ...base import TiNode, first
from ...registry import register


@register
class ApplyMaskAlpha(TiNode):
	DISPLAY_NAME = "Apply Mask as Alpha (ti)"
	CATEGORY = "tinode/image"

	@classmethod
	def INPUT_TYPES(cls):
		return {
			"required": {
				"image": ("IMAGE", {"tooltip": "RGB frames (the crop)."}),
				"mask": ("MASK", {"tooltip": "Becomes the alpha channel."}),
			},
			"optional": {
				"invert": ("BOOLEAN", {"default": False, "tooltip":
					"Flip the mask before using it as alpha."}),
			},
		}

	RETURN_TYPES = ("IMAGE",)
	RETURN_NAMES = ("rgba",)
	OUTPUT_TOOLTIPS = ("The image with the mask as its alpha channel ([N,H,W,4]).",)
	FUNCTION = "execute"

	def execute(self, image, mask, invert=False):
		img = first(image)
		m = first(mask)
		if img.dim() != 4:
			img = img.unsqueeze(0) if img.dim() == 3 else img
		rgb = img[..., :3]
		N, H, W, _ = rgb.shape

		if m.dim() == 2:
			m = m.unsqueeze(0)
		if bool(first(invert, False)):
			m = 1.0 - m
		# Match the mask to the frames: broadcast a single mask, and resize a
		# mismatched resolution so the alpha lines up with the pixels.
		if m.shape[0] == 1 and N > 1:
			m = m.expand(N, -1, -1)
		if m.shape[1:] != (H, W):
			m = torch.nn.functional.interpolate(
				m.unsqueeze(1), size=(H, W), mode="bilinear", align_corners=False)[:, 0]
		if m.shape[0] != N:
			raise RuntimeError(
				f"Apply Mask as Alpha: {m.shape[0]} mask frame(s) but {N} image "
				"frame(s) — counts must match (or one mask).")

		alpha = m.clamp(0, 1).unsqueeze(-1)
		return (torch.cat([rgb, alpha], dim=-1).contiguous(),)
