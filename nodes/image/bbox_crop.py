"""Mask Bbox Crop — crop the image down to the mask's bounding box.

The inpaint workflow's crop half: find the tightest box containing the mask,
grow it by `context_padding` so the sampler sees surrounding pixels, round the
box up to a multiple of `divisible_by` (VAE/UNet want /8), and cut the image
and mask down to it. Sampling a 512x512 box instead of a 4k frame is the whole
point. Pixels are taken at native scale — no resampling — so the round trip
through Mask Crop Paste Back is lossless outside the mask.

Unlike Mask Crop · Center Fill this does NOT black out the background or fit
each mask to its own square canvas; you get the real neighbourhood of the
subject, which is what an inpaint model needs for context.

Batch rules, given N images and M masks:
  M == N   pair them up, one crop per frame.
  N == 1   every mask is unioned into a single box (whole crowd in one crop).
  M == 1   the one mask is reused for every frame.
An IMAGE batch must be uniformly sized, so per-frame boxes are all grown to
the largest box in the batch and re-centred on their own mask. Raw per-frame
mask detection jitters, so the box position is smoothed across a temporal
`smoothing` window (a centred moving average) — dropped-detection frames are
filled from their neighbours instead of teleporting to the frame centre, and
the smoothed centre is always pulled back far enough to keep the subject inside
the crop (the `context_padding` slack is the smoothing headroom). This stops
the crop from swimming frame-to-frame. Set shared_bbox to instead union every
mask into ONE static box used by all frames (smoothing is then moot); pick this
when the subject barely moves and you want a perfectly still crop.

The emitted crop_info is a plain TI_CROP_XFORM, so it feeds the existing Mask
Crop Paste Back and Crop Info Drop/Pick Indices nodes unchanged.
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


def _round_up(value: int, multiple: int, limit: int) -> int:
	"""Smallest multiple of `multiple` >= value, never exceeding `limit`.

	Rounding up can overshoot the image, so fall back to rounding down. When
	the image itself is smaller than one `multiple` no multiple fits at all
	and the raw limit is returned — a 5px-tall image cannot be made /8.
	"""
	if multiple <= 1:
		return min(value, limit)
	up = ((value + multiple - 1) // multiple) * multiple
	if up <= limit:
		return up
	return ((limit // multiple) * multiple) or limit


def _center_start(center: float, span: int, limit: int) -> int:
	"""Window start of length `span` centred on `center`, kept inside [0,limit]."""
	start = int(round(center - span / 2))
	return max(0, min(start, limit - span))


def _cover_start(center: float, span: int, lo: int, hi: int, limit: int) -> int:
	"""Like `_center_start`, but guaranteed to also cover the subject span [lo,hi].

	Centre on `center` for stability, then pull the window back just enough that
	the tight subject box stays fully inside it (`span` is always >= the subject
	span, so both constraints are satisfiable), then clamp to the frame. This is
	what lets us smooth the centre without ever cropping the subject off: the box
	drifts freely within the `context_padding` slack and only snaps when the
	subject would otherwise leave the crop.
	"""
	start = int(round(center - span / 2))
	start = min(start, lo)          # keep the subject's top/left edge visible
	start = max(start, hi - span)   # keep the subject's bottom/right edge visible
	return max(0, min(start, limit - span))


def _fill_smooth(vals: list, window: int) -> list:
	"""Forward/backward-fill None centres, then centred moving-average.

	`vals` holds one centre coordinate per frame, None where the frame had no
	mask. Filling holds the nearest known position so a dropped detection reuses
	a neighbour's box instead of teleporting to the frame centre. The moving
	average removes the frame-to-frame tracking jitter that makes the crop swim.
	Caller guarantees at least one non-None entry.
	"""
	n = len(vals)
	filled = list(vals)
	last = None
	for i in range(n):                       # forward fill
		if filled[i] is None:
			filled[i] = last
		else:
			last = filled[i]
	last = None
	for i in range(n - 1, -1, -1):           # backward fill the leading gap
		if filled[i] is None:
			filled[i] = last
		else:
			last = filled[i]
	if window <= 1 or n <= 1:
		return filled
	half = window // 2
	smoothed = []
	for i in range(n):
		lo = max(0, i - half)
		hi = min(n, i + half + 1)
		seg = filled[lo:hi]
		smoothed.append(sum(seg) / len(seg))
	return smoothed


def _bbox(m: torch.Tensor, threshold: float):
	"""Tight (y0,y1,x0,x1) around mask pixels above threshold; None if empty."""
	ys, xs = torch.where(m > threshold)
	if ys.numel() == 0:
		return None
	return int(ys.min()), int(ys.max()) + 1, int(xs.min()), int(xs.max()) + 1


@register
class MaskBboxCrop(TiNode):
	DISPLAY_NAME = "Mask Bbox Crop (ti)"
	CATEGORY = "tinode/image"

	# Consume the whole batch in one call: a mask LIST upstream would otherwise
	# invoke this once per mask and emit a separate 1-item crop_info each time.
	INPUT_IS_LIST = True

	@classmethod
	def INPUT_TYPES(cls):
		return {
			"required": {
				"image": ("IMAGE",),
				"mask": ("MASK",),
				"context_padding": ("INT", {"default": 32, "min": 0, "max": 4096, "step": 1}),
				"divisible_by": ("INT", {"default": 8, "min": 1, "max": 256, "step": 1}),
			},
			"optional": {
				"shared_bbox": ("BOOLEAN", {"default": False}),
				"smoothing": ("INT", {"default": 5, "min": 1, "max": 99, "step": 2,
					"tooltip": "Temporal window (frames) to average the crop-box "
							   "position across a video, killing per-frame jitter. "
							   "1 = off. Ignored when shared_bbox is on."}),
				"threshold": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),
			},
		}

	RETURN_TYPES = ("IMAGE", "MASK", "TI_CROP_XFORM")
	RETURN_NAMES = ("images", "masks", "crop_info")
	FUNCTION = "execute"

	def execute(self, image, mask, context_padding=32, divisible_by=8,
				shared_bbox=False, smoothing=5, threshold=0.5):
		ilist = image if isinstance(image, list) else [image]
		imgs = torch.cat([i if i.dim() == 4 else i.unsqueeze(0) for i in ilist], dim=0)
		mlist = mask if isinstance(mask, list) else [mask]
		masks = torch.cat([m if m.dim() == 3 else m.unsqueeze(0) for m in mlist], dim=0)

		pad = int(_first(context_padding, 32))
		div = int(_first(divisible_by, 8))
		shared = bool(_first(shared_bbox, False))
		smoothing = int(_first(smoothing, 5))
		threshold = float(_first(threshold, 0.5))

		N, H, W, C = imgs.shape

		if masks.shape[1] != H or masks.shape[2] != W:
			masks = F.interpolate(masks[:, None], size=(H, W),
								  mode="bilinear", align_corners=False)[:, 0]

		M = masks.shape[0]
		if M != N:
			if N == 1:
				masks = masks.amax(dim=0, keepdim=True)   # union the crowd
			elif M == 1:
				masks = masks.expand(N, H, W)
			else:
				raise RuntimeError(
					f"Cannot pair {N} images with {M} masks: counts must match, "
					f"or one side must be a single image/mask."
				)

		tight = [_bbox(masks[i], threshold) for i in range(N)]
		if all(b is None for b in tight):
			# Nothing masked anywhere — no box to compute. Pass the batch through
			# with all-None items so a later Paste Back is a no-op.
			return (imgs, masks, {"H": H, "W": W, "C": C, "items": [None] * N})

		# Grow each box by the context margin, clamped to the frame.
		grown = [
			None if b is None else
			(max(0, b[0] - pad), min(H, b[1] + pad), max(0, b[2] - pad), min(W, b[3] + pad))
			for b in tight
		]

		if shared:
			present = [g for g in grown if g is not None]
			union = (min(g[0] for g in present), max(g[1] for g in present),
					 min(g[2] for g in present), max(g[3] for g in present))
			# One box for every frame, including frames whose mask was empty:
			# their crop is still valid pixels, and their zero mask means a
			# mask-driven Paste Back writes nothing.
			grown = [union] * N

		present = [g for g in grown if g is not None]
		ch = _round_up(max(g[1] - g[0] for g in present), div, H)
		cw = _round_up(max(g[3] - g[2] for g in present), div, W)

		# Window size (ch,cw) is fixed for the whole batch; only the box POSITION
		# varies per frame. Smoothing that position across frames is what stops the
		# crop from swimming. Shared boxes are already static, so skip it there.
		if shared:
			cy = (union[0] + union[1]) / 2
			cx = (union[2] + union[3]) / 2
			starts = [(_center_start(cy, ch, H), _center_start(cx, cw, W))] * N
		else:
			cys = _fill_smooth(
				[None if t is None else (t[0] + t[1]) / 2 for t in tight], smoothing)
			cxs = _fill_smooth(
				[None if t is None else (t[2] + t[3]) / 2 for t in tight], smoothing)
			starts = []
			for i in range(N):
				t = tight[i]
				if t is None:
					# No subject this frame: sit at the smoothed neighbour position
					# (filled above) rather than teleporting to the frame centre.
					starts.append((_center_start(cys[i], ch, H),
								   _center_start(cxs[i], cw, W)))
				else:
					# Smoothed centre, but never so far that the subject leaves the box.
					starts.append((_cover_start(cys[i], ch, t[0], t[1], H),
								   _cover_start(cxs[i], cw, t[2], t[3], W)))

		crops, out_masks, items = [], [], []
		for i in range(N):
			y0, x0 = starts[i]
			y1, x1 = y0 + ch, x0 + cw
			crops.append(imgs[i, y0:y1, x0:x1, :])
			out_masks.append(masks[i, y0:y1, x0:x1])
			# Under per-frame boxes an empty-mask frame keeps a None item so Paste
			# Back skips it; a shared box is valid for every frame, empty or not.
			items.append(None if (not shared and tight[i] is None) else {
				"y0": y0, "x0": x0, "h": ch, "w": cw,
				"oy": 0, "ox": 0, "nh": ch, "nw": cw,
			})

		crop_info = {"H": H, "W": W, "C": C, "items": items}
		return (torch.stack(crops, 0), torch.stack(out_masks, 0), crop_info)
