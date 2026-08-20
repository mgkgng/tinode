"""Mask Temporal Stabilize — stop a jittering mask from flickering the result.

A per-frame detector wobbles: measured on a real SAM3 mask the area swung from
1,895 to 25,750 pixels across one 85-frame chunk, and 30% of the crop was masked
in SOME frames but not others. Every pixel in that 30% therefore alternates
between the model's fill and the untouched original — which differ by ~15 levels
— so it flickers, in the air and on the floor wherever the boundary sweeps. The
fill itself is innocent: it is temporally SMOOTHER than the source.

The cure is to make the mask consistent over time, so the same pixels are filled
on every frame:

  majority   masked where most of the window agrees. Best on real data, and the
             default: on the measured mask it cut switching from 0.80 to 0.33
             per pixel at window 9 (0.23 at 15) while making the mask SMALLER
             (3.4% -> 2.9%), because it drops lone spurious frames as well as
             filling lone dropouts.
  median     the same idea per pixel; measured identical to majority here.
  union      masked anywhere in the window. The obvious choice, and the weaker
             one: it only fixes dropouts, never spurious detections, so it cut
             switching to just 0.61 while GROWING the mask 3.4% -> 5.0%. Reach
             for it only when the detector misses more than it hallucinates.

Because majority can also trim genuine mask, follow it with Grow Mask if you
need the margin back — the two together give a steady mask at the size you want.

`window` is the number of frames considered, centred. Bigger is steadier and
looser; it should comfortably exceed the length of a dropout.

Purely temporal: nothing here dilates or erodes in space, so an edge that is
already stable keeps its exact shape.
"""

from __future__ import annotations

import torch

from ...base import TiNode, first
from ...registry import register

_MODES = ["majority", "median", "union"]


def stabilize(mask, window=9, mode="majority", threshold=0.5):
	"""Temporally consistent version of a [N,H,W] mask."""
	m = (mask > threshold).float()
	n = m.shape[0]
	w = max(1, int(window))
	if w <= 1 or n <= 1:
		return m
	half = w // 2
	out = torch.empty_like(m)
	for i in range(n):
		lo, hi = max(0, i - half), min(n, i + half + 1)
		seg = m[lo:hi]
		if mode == "union":
			out[i] = seg.amax(0)
		elif mode == "median":
			out[i] = (seg.median(0).values > 0.5).float()
		else:                                   # majority
			out[i] = (seg.mean(0) > 0.5).float()
	return out


@register
class MaskTemporalStabilize(TiNode):
	DISPLAY_NAME = "Mask Temporal Stabilize (ti)"
	CATEGORY = "tinode/image"

	@classmethod
	def INPUT_TYPES(cls):
		return {
			"required": {
				"mask": ("MASK", {"tooltip": "The per-frame mask to steady."}),
				"window": ("INT", {"default": 9, "min": 1, "max": 999, "step": 2,
					"tooltip": "Frames considered, centred. Should comfortably "
							   "exceed the longest detection dropout."}),
			},
			"optional": {
				"mode": (_MODES, {"default": "majority", "tooltip":
					"majority (default) is the most effective measured: it drops "
					"lone spurious frames as well as filling lone dropouts. union "
					"only fills dropouts and grows the mask. median matches "
					"majority in practice."}),
				"threshold": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),
			},
		}

	RETURN_TYPES = ("MASK", "STRING")
	RETURN_NAMES = ("mask", "report")
	OUTPUT_TOOLTIPS = (
		"The steadied mask.",
		"Mask switches per pixel, before and after. This is the flicker figure: "
		"each switch is one frame where a pixel flips between the fill and the "
		"untouched original.",
	)
	FUNCTION = "execute"

	def execute(self, mask, window=9, mode="majority", threshold=0.5):
		m = first(mask)
		if not isinstance(m, torch.Tensor):
			raise RuntimeError("Mask Temporal Stabilize: `mask` must be a MASK.")
		if m.dim() == 2:
			m = m.unsqueeze(0)
		th = float(first(threshold, 0.5))
		out = stabilize(m, int(first(window, 9)), str(first(mode, "majority")), th)

		def toggles(x):
			"""Mean number of times a pixel switches masked/unmasked.

			NOT "masked in some frames but not all" — an object crossing the frame
			legitimately masks a pixel once and unmasks it once, and counting that
			as instability makes a moving subject look like flicker. What the eye
			sees as flicker is repeated switching, so that is what is counted.
			"""
			b = (x > 0.5).float()
			return float((b[1:] - b[:-1]).abs().sum(0).mean())

		before, after = toggles(m), toggles(out)
		rep = (f"{mode}/window {int(first(window, 9))}: mask switches per pixel "
			   f"{before:.2f} -> {after:.2f}  "
			   f"(mask covers {float((m > th).float().mean()) * 100:.1f}% -> "
			   f"{float(out.mean()) * 100:.1f}%)")
		print(f"[tinode] Mask Temporal Stabilize: {rep}")
		return (out, rep)
