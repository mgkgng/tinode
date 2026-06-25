"""Mask Translate — shift a mask on X/Y, or sweep it into a growth sequence.

Single shift: move the mask by (offset_x, offset_y) pixels, zero-padding the
vacated area (no wrap by default). offset_y negative = up.

Growth sequence: set frames > 1 to emit a batch where frame i is shifted by
offset * i/(frames-1) — 0 at the first frame, full offset at the last. Drive
the inpaint region upward over time to draw a plant growing / flower rising.
"""

from __future__ import annotations

import torch

from ...base import TiNode
from ...registry import register


def shift2d(m: torch.Tensor, dy: int, dx: int, wrap: bool) -> torch.Tensor:
	"""Shift [...,H,W] by (dy,dx). dy>0 down, dx>0 right. Zero-pad unless wrap."""
	if wrap:
		return torch.roll(m, shifts=(dy, dx), dims=(-2, -1))
	H, W = m.shape[-2], m.shape[-1]
	out = torch.zeros_like(m)
	y0d, y1d = max(0, dy), min(H, H + dy)
	y0s, y1s = max(0, -dy), min(H, H - dy)
	x0d, x1d = max(0, dx), min(W, W + dx)
	x0s, x1s = max(0, -dx), min(W, W - dx)
	if y1d > y0d and x1d > x0d:
		out[..., y0d:y1d, x0d:x1d] = m[..., y0s:y1s, x0s:x1s]
	return out


@register
class MaskTranslate(TiNode):
	DISPLAY_NAME = "Mask Translate (ti)"
	CATEGORY = "tinode/image"

	@classmethod
	def INPUT_TYPES(cls):
		return {
			"required": {
				"mask": ("MASK",),
				"offset_x": ("INT", {"default": 0, "min": -8192, "max": 8192, "step": 1}),
				"offset_y": ("INT", {"default": 0, "min": -8192, "max": 8192, "step": 1}),
			},
			"optional": {
				"frames": ("INT", {"default": 1, "min": 1, "max": 1000, "step": 1}),
				"wrap": ("BOOLEAN", {"default": False}),
			},
		}

	RETURN_TYPES = ("MASK",)
	RETURN_NAMES = ("masks",)
	FUNCTION = "execute"

	def execute(self, mask, offset_x, offset_y, frames=1, wrap=False):
		if mask.dim() == 2:
			mask = mask.unsqueeze(0)

		if frames <= 1:
			return (shift2d(mask, offset_y, offset_x, wrap),)

		# growth sweep: shift the base mask from 0 -> full offset across
		# `frames`, emitting one shifted copy per frame.
		base = mask[0:1]                         # [1,H,W]
		out = []
		for i in range(frames):
			t = i / (frames - 1)
			out.append(shift2d(base, round(offset_y * t), round(offset_x * t), wrap))
		return (torch.cat(out, dim=0),)
