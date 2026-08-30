"""Frame Stride — keep every Nth frame of a batch (the half-rate switch).

The source-side half of reduced-rate work: the store loaders stride the masks
and crops via `every_nth` on Load Masks / Load Clips, and this node strides the
DECODED source frames the same way (Get Video Components has no such option).
Keep `offset` 0 so everything sits on the same global grid (frames 0, n, 2n, …).

Lossless — kept frames are the original frames, untouched; the others are simply
not passed on. `fps` in → `fps_out` = fps / every_nth, so the final save runs at
the true reduced rate (50fps in, every 2nd frame → 25fps out).
"""

from __future__ import annotations

import torch

from ...base import TiNode, first
from ...registry import register


@register
class FrameStride(TiNode):
	DISPLAY_NAME = "Frame Stride (ti)"
	CATEGORY = "tinode/video"

	@classmethod
	def INPUT_TYPES(cls):
		return {
			"required": {
				"images": ("IMAGE", {"tooltip": "The frames to stride."}),
				"every_nth": ("INT", {"default": 2, "min": 1, "max": 10, "step": 1,
					"tooltip": "Keep frames 0, n, 2n, … — 2 halves 50fps to 25fps. "
							   "Must match the loaders' every_nth."}),
			},
			"optional": {
				"mask": ("MASK", {"tooltip": "Strided in lockstep, if wired."}),
				"offset": ("INT", {"default": 0, "min": 0, "max": 9, "step": 1,
					"tooltip": "First kept frame. Leave 0 — the store loaders keep "
							   "the global 0-grid, and a different offset would put "
							   "the source between their frames."}),
				"fps": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 240.0, "step": 0.01,
					"forceInput": True, "tooltip":
					"Source fps in — fps_out = fps / every_nth for the final save."}),
				# APPENDED last: when wired (from Load Clip Fills' source_stride)
				# this OVERRIDES the widget — one knob instead of two to keep in sync.
				"every_nth_in": ("INT", {"default": 0, "min": 0, "max": 100, "step": 1,
					"forceInput": True, "tooltip":
					"Stride from upstream (Load Clip Fills.source_stride). When "
					"connected and > 0, the every_nth widget is ignored."}),
			},
		}

	RETURN_TYPES = ("IMAGE", "MASK", "INT", "FLOAT")
	RETURN_NAMES = ("images", "mask", "frame_count", "fps_out")
	OUTPUT_TOOLTIPS = (
		"Every Nth frame, untouched.",
		"The mask strided the same way (zeros when none was wired).",
		"How many frames remain.",
		"fps / every_nth — wire to the final save's frame_rate.",
	)
	FUNCTION = "execute"

	def execute(self, images, every_nth=2, mask=None, offset=0, fps=0.0,
				every_nth_in=0):
		imgs = first(images)
		if imgs.dim() != 4:
			imgs = imgs.unsqueeze(0) if imgs.dim() == 3 else imgs
		override = int(first(every_nth_in, 0) or 0)
		n = override if override > 0 else max(1, int(first(every_nth, 2)))
		off = min(max(0, int(first(offset, 0))), max(0, imgs.shape[0] - 1))
		out = imgs[off::n].contiguous() if n > 1 or off else imgs

		m = first(mask)
		if isinstance(m, torch.Tensor):
			if m.dim() == 2:
				m = m.unsqueeze(0)
			mask_out = m[off::n].contiguous() if n > 1 or off else m
		else:
			mask_out = torch.zeros((out.shape[0], out.shape[1], out.shape[2]),
								   dtype=torch.float32)

		f = float(first(fps, 0.0))
		return (out, mask_out, int(out.shape[0]), (f / n) if f > 0 else 0.0)
