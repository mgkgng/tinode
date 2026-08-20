"""Compare Videos — put before and after side by side, to judge a removal.

Two clips in, one clip out: the original crop next to what the model returned,
so a preview shows both at once and you can see whether the object is really
gone and whether anything else moved. Judging a fill from the result alone is
guesswork — you need the thing it replaced in the same glance.

Layouts:
  side_by_side  A | B, the honest default: nothing is hidden behind anything.
  stacked       A over B, for a wide crop where side-by-side gets tiny.
  split         one frame, left half A and right half B, with a divider. Best
                for spotting a seam or a shift, since matching pixels line up
                across the join — but it hides half of each, so use it after
                side-by-side, not instead of it.
  difference    |A - B| amplified: black where nothing changed, bright where it
                did. This is the one that proves an untouched area really was
                untouched.

Frame counts and sizes rarely match exactly (a model can return a different
length), so the shorter clip is held on its last frame and B is resized to A's
resolution — with a note when that happens, since a size difference usually
means the wrong pair got wired.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from ...base import TiNode, first
from ...registry import register

_LAYOUTS = ["side_by_side", "stacked", "split", "difference"]


def _conform(a, b):
	"""Make B match A in frame count and resolution; report what was changed."""
	notes = []
	if b.shape[1:3] != a.shape[1:3]:
		notes.append(f"resized B {tuple(b.shape[1:3])}->{tuple(a.shape[1:3])}")
		b = F.interpolate(b.permute(0, 3, 1, 2), size=(a.shape[1], a.shape[2]),
						  mode="bilinear", align_corners=False).permute(0, 2, 3, 1)
	if b.shape[-1] != a.shape[-1]:
		c = min(a.shape[-1], b.shape[-1])
		a, b = a[..., :c], b[..., :c]
	na, nb = a.shape[0], b.shape[0]
	if na != nb:
		notes.append(f"{na} vs {nb} frames — holding the shorter one's last frame")
		n = max(na, nb)
		if na < n:
			a = torch.cat([a, a[-1:].repeat(n - na, 1, 1, 1)], dim=0)
		if nb < n:
			b = torch.cat([b, b[-1:].repeat(n - nb, 1, 1, 1)], dim=0)
	return a, b, notes


def _label(img, text, right=False):
	"""Burn a small caption in, so a saved comparison is still readable later."""
	try:
		import numpy as np  # noqa: PLC0415
		from PIL import Image, ImageDraw  # noqa: PLC0415
	except Exception:  # noqa: BLE001 — captions are cosmetic
		return img
	h, w = img.shape[1], img.shape[2]
	pad = max(14, h // 28)
	strip = Image.new("RGB", (w, pad + 6), (0, 0, 0))
	d = ImageDraw.Draw(strip)
	d.text((8 if not right else max(8, w - 8 - 7 * len(text)), 3), text, fill=(255, 255, 255))
	band = torch.from_numpy(np.asarray(strip, dtype="float32") / 255.0)
	band = band.unsqueeze(0).repeat(img.shape[0], 1, 1, 1).to(img.dtype)
	if band.shape[-1] != img.shape[-1]:
		band = band[..., :1].repeat(1, 1, 1, img.shape[-1])
	out = img.clone()
	out[:, :band.shape[1], :, :] = out[:, :band.shape[1], :, :] * 0.25 + band * 0.75
	return out


@register
class CompareVideos(TiNode):
	DISPLAY_NAME = "Compare Videos · Before/After (ti)"
	CATEGORY = "tinode/video"

	@classmethod
	def INPUT_TYPES(cls):
		return {
			"required": {
				"before": ("IMAGE", {"tooltip": "The original crop, as it went in."}),
				"after": ("IMAGE", {"tooltip": "What the model returned."}),
			},
			"optional": {
				"layout": (_LAYOUTS, {"default": "side_by_side"}),
				"labels": ("BOOLEAN", {"default": True, "tooltip":
					"Burn BEFORE / AFTER captions in, so a saved comparison is "
					"still readable out of context."}),
				"difference_gain": ("FLOAT", {"default": 8.0, "min": 1.0, "max": 64.0, "step": 0.5,
					"tooltip": "Amplification for the difference layout. A clean "
							   "removal is black everywhere it did not touch, and "
							   "the gain is what makes a small change visible."}),
				"split_position": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01,
					"tooltip": "Where the split layout cuts, 0..1 across the width."}),
			},
		}

	RETURN_TYPES = ("IMAGE", "STRING")
	RETURN_NAMES = ("comparison", "notes")
	OUTPUT_TOOLTIPS = (
		"The comparison clip — send it to a Preview Image or Save Video.",
		"Anything that had to be conformed (frame count, size), which usually "
		"means the wrong pair is wired.",
	)
	FUNCTION = "execute"

	def execute(self, before, after, layout="side_by_side", labels=True,
				difference_gain=8.0, split_position=0.5):
		a = first(before)
		b = first(after)
		for name, t in (("before", a), ("after", b)):
			if not isinstance(t, torch.Tensor):
				raise RuntimeError(f"Compare Videos: `{name}` must be an IMAGE.")
		if a.dim() == 3:
			a = a.unsqueeze(0)
		if b.dim() == 3:
			b = b.unsqueeze(0)

		a, b, notes = _conform(a, b)
		layout = str(first(layout, "side_by_side"))
		if bool(first(labels, True)) and layout in ("side_by_side", "stacked"):
			a, b = _label(a, "BEFORE"), _label(b, "AFTER")

		if layout == "stacked":
			out = torch.cat([a, b], dim=1)
		elif layout == "split":
			x = int(round(float(first(split_position, 0.5)) * a.shape[2]))
			x = max(0, min(x, a.shape[2]))
			out = torch.cat([a[:, :, :x, :], b[:, :, x:, :]], dim=2)
			if 0 < x < out.shape[2]:                     # a divider, so the join is visible
				out[:, :, max(0, x - 1):x + 1, :] = 1.0
		elif layout == "difference":
			g = float(first(difference_gain, 8.0))
			out = ((a - b).abs() * g).clamp(0, 1)
		else:
			out = torch.cat([a, b], dim=2)

		note = "; ".join(notes) if notes else "ok"
		if notes:
			print(f"[tinode] Compare Videos: {note}")
		return (out.contiguous(), note)
