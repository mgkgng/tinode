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
the largest box in the batch and re-centred on their own mask. Set shared_bbox
to instead union every mask into ONE box used by all frames — the crop stops
tracking each frame and stays put, which is what you want for video (a box
that follows the mask makes the inpainted region swim).

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


def _window(center: int, span: int, limit: int) -> tuple[int, int]:
	"""A `span`-long window centred on `center`, shifted to sit inside [0,limit]."""
	start = max(0, min(center - span // 2, limit - span))
	return start, start + span


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
				"threshold": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),
			},
		}

	RETURN_TYPES = ("IMAGE", "MASK", "TI_CROP_XFORM")
	RETURN_NAMES = ("images", "masks", "crop_info")
	FUNCTION = "execute"

	def execute(self, image, mask, context_padding=32, divisible_by=8,
				shared_bbox=False, threshold=0.5):
		ilist = image if isinstance(image, list) else [image]
		imgs = torch.cat([i if i.dim() == 4 else i.unsqueeze(0) for i in ilist], dim=0)
		mlist = mask if isinstance(mask, list) else [mask]
		masks = torch.cat([m if m.dim() == 3 else m.unsqueeze(0) for m in mlist], dim=0)

		pad = int(_first(context_padding, 32))
		div = int(_first(divisible_by, 8))
		shared = bool(_first(shared_bbox, False))
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

		boxes = [_bbox(masks[i], threshold) for i in range(N)]
		if all(b is None for b in boxes):
			# Nothing masked anywhere — no box to compute. Pass the batch through
			# with all-None items so a later Paste Back is a no-op.
			return (imgs, masks, {"H": H, "W": W, "C": C, "items": [None] * N})

		# Grow each box by the context margin, clamped to the frame.
		grown = [
			None if b is None else
			(max(0, b[0] - pad), min(H, b[1] + pad), max(0, b[2] - pad), min(W, b[3] + pad))
			for b in boxes
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

		crops, out_masks, items = [], [], []
		for i in range(N):
			g = grown[i]
			if g is None:
				# Empty mask under per-frame boxes: emit a centred crop purely to
				# keep the batch stackable, and a None item so Paste Back skips it.
				cy, cx = H // 2, W // 2
			else:
				cy, cx = (g[0] + g[1]) // 2, (g[2] + g[3]) // 2

			y0, y1 = _window(cy, ch, H)
			x0, x1 = _window(cx, cw, W)
			crops.append(imgs[i, y0:y1, x0:x1, :])
			out_masks.append(masks[i, y0:y1, x0:x1])
			items.append(None if g is None else {
				"y0": y0, "x0": x0, "h": ch, "w": cw,
				"oy": 0, "ox": 0, "nh": ch, "nw": cw,
			})

		crop_info = {"H": H, "W": W, "C": C, "items": items}
		return (torch.stack(crops, 0), torch.stack(out_masks, 0), crop_info)
