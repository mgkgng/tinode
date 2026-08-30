"""Crop · Divisible — trim a frame until its size divides evenly.

The fix for "1080 doesn't divide by 512": crop the smallest amount that makes
the width divisible by one value and the height by another (they are set
independently), so a later Divide · Rectangle — or a model that demands /8, /16,
/64 — gets a size that comes out exact.

The trim is taken from BOTH sides, not one: a 6px remainder is 3 off the left
and 3 off the right, so the framing stays centred instead of drifting toward one
corner. An odd remainder cannot split evenly, so the extra row/column comes off
the bottom/right (`anchor` flips that, or pins the crop to a corner outright).

Lossless: a plain tensor slice at native scale, so the kept pixels are bit-exact.
The emitted crop_info places the crop back in the original frame, so Mask Crop
Paste Back / Composite Crops restore the full-size frame — the trimmed border
comes back from the original, untouched.
"""

from __future__ import annotations

import torch

from ...base import TiNode, first
from ...registry import register

_ANCHOR = ["center", "center_bias_start", "start", "end"]


def trim_amounts(total, divisor, anchor="center"):
	"""(before, after) to cut off `total` so the rest divides by `divisor`.

	Centred by default — half the remainder from each side — because trimming a
	whole remainder off one edge shifts the framing. `center` gives an odd extra
	to the END (bottom/right), `center_bias_start` to the START.
	"""
	total = int(total)
	divisor = int(divisor)
	if divisor <= 1 or total <= 0:
		return 0, 0
	rem = total % divisor
	if rem == 0:
		return 0, 0
	if rem >= total:
		raise RuntimeError(
			f"Crop · Divisible: {total}px cannot be made divisible by {divisor} "
			"— the whole frame would be trimmed. Use a smaller divisor.")
	if anchor == "start":
		return 0, rem                     # keep the top/left edge
	if anchor == "end":
		return rem, 0                     # keep the bottom/right edge
	half = rem // 2
	if anchor == "center_bias_start":
		return rem - half, half           # odd extra off the start
	return half, rem - half               # centred; odd extra off the end


@register
class CropDivisible(TiNode):
	DISPLAY_NAME = "Crop · Divisible (ti)"
	CATEGORY = "tinode/div"

	@classmethod
	def INPUT_TYPES(cls):
		return {
			"required": {
				"image": ("IMAGE", {"tooltip":
					"The frame (or video batch) to trim. Every frame is cut the same."}),
				"divisible_width": ("INT", {"default": 8, "min": 1, "max": 4096, "step": 1,
					"tooltip": "Crop the WIDTH down until it divides by this. 1 = leave it."}),
				"divisible_height": ("INT", {"default": 8, "min": 1, "max": 4096, "step": 1,
					"tooltip": "Crop the HEIGHT down until it divides by this. 1 = leave it."}),
			},
			"optional": {
				"mask": ("MASK", {"tooltip":
					"Trimmed in lockstep so it still lines up with the frames."}),
				"anchor": (_ANCHOR, {"default": "center", "tooltip":
					"Where the trim comes from. center = half off each side (an odd "
					"extra off bottom/right), center_bias_start = odd extra off "
					"top/left, start = keep the top/left edge, end = keep bottom/right."}),
			},
		}

	RETURN_TYPES = ("IMAGE", "MASK", "TI_CROP_XFORM", "INT", "INT")
	RETURN_NAMES = ("image", "mask", "crop_info", "width", "height")
	OUTPUT_TOOLTIPS = (
		"The trimmed frames — both dimensions now divide evenly.",
		"The mask, trimmed to match (zeros when none was wired).",
		"Where the crop sits in the original — feed Paste Back to restore full size.",
		"The new width.",
		"The new height.",
	)
	FUNCTION = "execute"

	def execute(self, image, divisible_width=8, divisible_height=8, mask=None,
				anchor="center"):
		imgs = first(image)
		if imgs.dim() != 4:
			imgs = imgs.unsqueeze(0) if imgs.dim() == 3 else imgs
		N, H, W, C = imgs.shape
		anchor = first(anchor, "center")

		left, right = trim_amounts(W, int(first(divisible_width, 8)), anchor)
		top, bottom = trim_amounts(H, int(first(divisible_height, 8)), anchor)
		x0, x1 = left, W - right
		y0, y1 = top, H - bottom

		out = imgs[:, y0:y1, x0:x1, :].contiguous()
		nh, nw = y1 - y0, x1 - x0

		m = first(mask)
		if isinstance(m, torch.Tensor):
			if m.dim() == 2:
				m = m.unsqueeze(0)
			if m.shape[1] != H or m.shape[2] != W:
				raise RuntimeError(
					f"Crop · Divisible: the mask is {m.shape[1]}x{m.shape[2]} but the "
					f"image is {H}x{W} — they must match to be trimmed together.")
			mask_out = m[:, y0:y1, x0:x1].contiguous()
		else:
			mask_out = torch.zeros((out.shape[0], nh, nw), dtype=torch.float32)

		# No rescale, so the crop fills its canvas: oy/ox = 0, nh/nw = h/w.
		item = {"y0": y0, "x0": x0, "h": nh, "w": nw,
				"oy": 0, "ox": 0, "nh": nh, "nw": nw}
		crop_info = {"H": H, "W": W, "C": C, "items": [item] * N}
		if (left or right or top or bottom):
			print(f"[tinode] Crop · Divisible: {W}x{H} -> {nw}x{nh} "
				  f"(left {left}, right {right}, top {top}, bottom {bottom})")
		return (out, mask_out, crop_info, int(nw), int(nh))
