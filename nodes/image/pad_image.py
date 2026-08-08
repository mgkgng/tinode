"""Pad Image — enlarge a frame by adding a border of a solid colour.

Grow the canvas by `top` / `bottom` / `left` / `right` pixels (any of them 0),
filling the new border with `color` (a hex string, #000 by default). Works on a
single image or a whole video batch — the original pixels are never resampled,
they are just placed inside the larger canvas.

Also emits a MASK marking the border. By default the border (the added region)
is white (1) and the original area is black (0) — ready to drive an outpaint /
inpaint of exactly the new space. Flip `invert_mask` to mark the ORIGINAL area
instead.

The padded amount is returned per side too, so a later crop can peel the border
back off (e.g. feed them into Trim Video, or a manual crop).
"""

from __future__ import annotations

import re

import torch

from ...base import TiNode
from ...registry import register

_HEX3 = re.compile(r"^[0-9a-fA-F]{3}$")
_HEX6 = re.compile(r"^[0-9a-fA-F]{6}$")


def _hex_to_rgb(text: str):
	"""'#rrggbb' / 'rrggbb' / '#rgb' -> (r, g, b) floats 0..1. Black on junk.

	Note a 3-letter word like 'bad' is still all hex digits, so length alone is
	not enough — validate the characters explicitly.
	"""
	s = str(text).strip().lstrip("#")
	if _HEX3.match(s):
		s = "".join(c * 2 for c in s)        # #abc -> #aabbcc
	if not _HEX6.match(s):
		print(f"[tinode] Pad Image: could not parse color {text!r}, using black.")
		return (0.0, 0.0, 0.0)
	return tuple(int(s[i:i + 2], 16) / 255.0 for i in (0, 2, 4))


@register
class PadImage(TiNode):
	DISPLAY_NAME = "Pad Image · Add Border (ti)"
	CATEGORY = "tinode/image"

	@classmethod
	def INPUT_TYPES(cls):
		side = {"default": 0, "min": 0, "max": 8192, "step": 1}
		return {
			"required": {
				"image": ("IMAGE",),
				"top": ("INT", dict(side)),
				"bottom": ("INT", dict(side)),
				"left": ("INT", dict(side)),
				"right": ("INT", dict(side)),
			},
			"optional": {
				"color": ("STRING", {"default": "#000000",
					"tooltip": "Border fill colour, hex (e.g. #000, #ffffff, 1a2b3c)."}),
				"invert_mask": ("BOOLEAN", {"default": False,
					"tooltip": "Off: border=1, original=0 (mask the added region). "
							   "On: original=1, border=0."}),
			},
		}

	RETURN_TYPES = ("IMAGE", "MASK", "INT", "INT", "INT", "INT")
	RETURN_NAMES = ("image", "mask", "top", "bottom", "left", "right")
	FUNCTION = "execute"

	def execute(self, image, top=0, bottom=0, left=0, right=0,
				color="#000000", invert_mask=False):
		imgs = image if image.dim() == 4 else image.unsqueeze(0)  # [N,H,W,C]
		N, H, W, C = imgs.shape
		t, b, l, r = (max(0, int(v)) for v in (top, bottom, left, right))

		new_h, new_w = H + t + b, W + l + r
		rgb = _hex_to_rgb(color)

		# Colour canvas, then drop the untouched original into place.
		fill = torch.tensor(rgb[:C] if C <= 3 else rgb + (1.0,) * (C - 3),
							dtype=imgs.dtype, device=imgs.device)
		out = fill.view(1, 1, 1, C).expand(N, new_h, new_w, C).clone()
		out[:, t:t + H, l:l + W, :] = imgs

		# Mask the border (added region) by default; the original block is the hole.
		mask = torch.ones((N, new_h, new_w), dtype=torch.float32, device=imgs.device)
		mask[:, t:t + H, l:l + W] = 0.0
		if invert_mask:
			mask = 1.0 - mask

		return (out, mask, t, b, l, r)
