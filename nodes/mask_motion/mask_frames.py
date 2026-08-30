"""Mask Frames · Iterate — turn a mask animation into one job per frame.

The tracker gives you [frames, H, W]: one moving circle. To generate a DIFFERENT
image inside each position you need each frame to be its own run, with its own
prompt and its own noise. That is what this does — it hands back the frames as a
LIST, so everything downstream is expanded by ComfyUI once per frame, plus the
two numbers each of those runs needs to differ from its neighbours:

  index   0, 1, 2 … — position in the batch, for labelling or lookups
  seed    a well-separated seed per frame, all rolled by ONE seed widget

The seeds are spread by golden-ratio hashing rather than `base + i`, because
adjacent integers land adjacent in most samplers' noise and you would get twelve
near-identical objects instead of twelve different ones.

`stride` and `limit` exist because generating on every frame of an 81-frame
tracker is a long wait for a first look. Take every 8th, see whether the idea
works, then open it up — the mask itself is untouched either way.
"""

from __future__ import annotations

import torch

from ...base import TiNode, first
from ...registry import register

# 2^32 / golden ratio: consecutive multiples land far apart, which is the whole
# point — one seed widget, twelve unrelated frames.
_GOLDEN = 2654435761
_SEED_MAX = 0xFFFFFFFFFFFFFFFF


def frame_indices(total, start=0, stride=1, limit=0):
	"""Which frames of a batch this run covers."""
	total = max(0, int(total))
	start = min(max(0, int(start)), max(0, total - 1))
	stride = max(1, int(stride))
	idx = list(range(start, total, stride))
	limit = int(limit)
	return idx[:limit] if limit > 0 else idx


def seed_for(base, index):
	"""A seed for frame `index`, far from its neighbours' but reproducible."""
	return (int(base) + (int(index) + 1) * _GOLDEN) & _SEED_MAX


def mask_bbox(frame, pad=0, square=True):
	"""(x, y, w, h) around everything set in one [H,W] mask frame.

	This is what lets the object be generated AT THE CIRCLE'S OWN SCALE. Ask a
	model to fill a 128px hole in a 1024px frame and it paints a small thing in a
	big picture; generate a square of the object and drop it in here, and the
	circle is full of it.

	A box that runs off the edge is SHIFTED back inside before it is clipped, so
	a circle near the border keeps its square shape (and the object keeps its
	proportions) instead of being squashed. An empty frame gives the whole
	frame — a degenerate box would divide by zero downstream.
	"""
	H, W = int(frame.shape[-2]), int(frame.shape[-1])
	hit = (frame > 1.0 / 255.0).nonzero()
	if hit.numel() == 0:
		return 0, 0, W, H
	y0, y1 = int(hit[:, 0].min()), int(hit[:, 0].max()) + 1
	x0, x1 = int(hit[:, 1].min()), int(hit[:, 1].max()) + 1
	pad = max(0, int(pad))
	x0, y0, x1, y1 = x0 - pad, y0 - pad, x1 + pad, y1 + pad
	if square:
		side = max(x1 - x0, y1 - y0)
		cx, cy = (x0 + x1) / 2.0, (y0 + y1) / 2.0
		x0 = int(round(cx - side / 2.0)); x1 = x0 + side
		y0 = int(round(cy - side / 2.0)); y1 = y0 + side
	# slide back inside the frame before clipping, so the box stays square
	if x1 > W:
		x0, x1 = x0 - (x1 - W), W
	if y1 > H:
		y0, y1 = y0 - (y1 - H), H
	if x0 < 0:
		x1, x0 = min(W, x1 - x0), 0
	if y0 < 0:
		y1, y0 = min(H, y1 - y0), 0
	x1, y1 = min(W, x1), min(H, y1)
	return x0, y0, max(1, x1 - x0), max(1, y1 - y0)


@register
class MaskFrames(TiNode):
	DISPLAY_NAME = "Mask Frames · Iterate (ti)"
	CATEGORY = "tinode/mask_motion"

	@classmethod
	def INPUT_TYPES(cls):
		return {
			"required": {
				"mask": ("MASK", {"tooltip":
					"A mask batch — e.g. Track Motion Editor's `mask` or `context`."}),
				"seed": ("INT", {"default": 0, "min": 0, "max": _SEED_MAX,
					"control_after_generate": True, "tooltip":
					"Rolls the WHOLE sequence. Each frame gets its own seed "
					"derived from this one, so a single randomize gives every "
					"frame a different image and a fixed value reproduces the lot."}),
			},
			"optional": {
				"stride": ("INT", {"default": 1, "min": 1, "max": 999, "step": 1,
					"tooltip": "Take every Nth frame. 8 turns an 81-frame tracker "
							   "into 11 generations for a first look."}),
				"limit": ("INT", {"default": 0, "min": 0, "max": 9999, "step": 1,
					"tooltip": "Stop after this many. 0 = no limit."}),
				"start": ("INT", {"default": 0, "min": 0, "max": 9999, "step": 1,
					"tooltip": "First frame to take."}),
				# APPENDED: widget values are positional in saved graphs.
				"pad": ("INT", {"default": 0, "min": 0, "max": 4096, "step": 1,
					"tooltip": "Grow the box around the mask by this many pixels, "
							   "so the generated object has a little air around it."}),
				"square": ("BOOLEAN", {"default": True, "tooltip":
					"Keep the box square — models generate a centred object best "
					"in a square, and a circle wants one anyway."}),
			},
		}

	# x / y / width / height / mask_crop are APPENDED so existing links keep
	# their slots.
	RETURN_TYPES = ("MASK", "INT", "INT", "INT", "IMAGE",
					"MASK", "INT", "INT", "INT", "INT")
	RETURN_NAMES = ("mask", "index", "seed", "count", "preview",
					"mask_crop", "x", "y", "width", "height")
	# Everything per-frame is a LIST, so it all expands together and stays
	# index-aligned; count and preview describe the whole run.
	OUTPUT_IS_LIST = (True, True, True, False, False,
					  True, True, True, True, True)
	OUTPUT_TOOLTIPS = (
		"One single-frame mask per run, full frame size.",
		"That frame's position in the batch.",
		"That frame's seed — wire to the sampler AND to Random Item.",
		"How many frames this run covers.",
		"The selected frames as one image batch, for a quick look.",
		"The mask cropped to its own box — the alpha for compositing the "
		"generated square, and the same size as it.",
		"Left edge of that box in the full frame — Image Composite Masked's x.",
		"Top edge of that box — Image Composite Masked's y.",
		"Box width. Scale the generated image to this before compositing.",
		"Box height.",
	)
	FUNCTION = "execute"

	def execute(self, mask, seed=0, stride=1, limit=0, start=0, pad=0, square=True):
		m = first(mask)
		if not isinstance(m, torch.Tensor):
			raise RuntimeError("Mask Frames: `mask` must be a MASK.")
		if m.dim() == 2:
			m = m.unsqueeze(0)
		if m.dim() != 3:
			raise RuntimeError(
				f"Mask Frames: expected [frames,H,W], got {tuple(m.shape)}.")

		idx = frame_indices(m.shape[0], first(start, 0), first(stride, 1),
							first(limit, 0))
		if not idx:
			raise RuntimeError(
				f"Mask Frames: no frames selected — the mask has {m.shape[0]} "
				f"frame(s) and start={int(first(start, 0))} is past the end.")
		base = int(first(seed, 0))
		pad = int(first(pad, 0))
		sq = bool(first(square, True))
		masks = [m[i:i + 1].contiguous() for i in idx]
		seeds = [seed_for(base, i) for i in idx]

		xs, ys, ws, hs, crops = [], [], [], [], []
		for fm in masks:
			x, y, w, h = mask_bbox(fm[0], pad, sq)
			xs.append(x); ys.append(y); ws.append(w); hs.append(h)
			crops.append(fm[:, y:y + h, x:x + w].contiguous())

		preview = torch.cat(masks, dim=0).unsqueeze(-1).expand(-1, -1, -1, 3)
		box = f"{ws[0]}x{hs[0]}" if len(set(zip(ws, hs))) == 1 else \
			f"{min(ws)}x{min(hs)}..{max(ws)}x{max(hs)}"
		print(f"[tinode] Mask Frames: {len(idx)} of {m.shape[0]} frame(s) "
			  f"-> {idx[:6]}{' …' if len(idx) > 6 else ''}  box {box}")
		return (masks, idx, seeds, len(idx), preview.contiguous(),
				crops, xs, ys, ws, hs)
