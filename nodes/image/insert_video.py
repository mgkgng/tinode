"""Insert Video — drop one clip into another at a given frame.

Two modes, both starting at `start_frame`:

  replace   overwrite the frames the inserted clip covers. The end of the
            replaced span is the clip's own length, so you never compute it by
            hand — insert 40 frames at frame 100 and frames 100..139 are gone.
  insert    splice the clip in without removing anything, pushing the rest of
            the base video later. The result grows by the clip's length.

The inserted clip is conformed to the base automatically (resized to its
resolution, reconciled to its channel count), so mismatched sources join
instead of erroring. The base clip's own pixels are never resampled.

Inserting past the end simply extends the video — the tail is empty, so
nothing is silently dropped. Returns the start/end frame the clip ended up
occupying, which is what you feed a later Trim/Batch Pick to get it back.
"""

from __future__ import annotations

import torch

from ...base import TiNode
from ...registry import register
from .extend_video import _as_batch, _match

_MODES = ["replace", "insert"]


@register
class InsertVideo(TiNode):
	DISPLAY_NAME = "Insert Video (ti)"
	CATEGORY = "tinode/image"

	@classmethod
	def INPUT_TYPES(cls):
		return {
			"required": {
				"image": ("IMAGE",),
				"insert": ("IMAGE",),
				"start_frame": ("INT", {"default": 0, "min": 0, "max": 1000000, "step": 1,
					"tooltip": "Frame of the base video where the clip goes. "
							   "0-based. Equal to the length appends at the end."}),
				"mode": (_MODES, {"default": "replace",
					"tooltip": "replace: overwrite the frames the clip covers "
							   "(span length = the clip's own length). "
							   "insert: splice it in, growing the video."}),
			},
		}

	RETURN_TYPES = ("IMAGE", "INT", "INT")
	RETURN_NAMES = ("images", "start", "end")
	FUNCTION = "execute"

	def execute(self, image, insert, start_frame=0, mode="replace"):
		base = _as_batch(image)                       # [N,H,W,C]
		N, H, W, C = base.shape
		clip = _match(_as_batch(insert), H, W, C, base)
		M = clip.shape[0]

		start = max(0, min(int(start_frame), N))      # == N is a valid append point
		if M == 0:
			return (base, start, start)

		head = base[:start]
		if mode == "insert":
			tail = base[start:]
		else:                                          # replace
			# The clip's length decides the end of the replaced span. Past the
			# end of the base there is simply no tail left.
			tail = base[min(start + M, N):]

		parts = [p for p in (head, clip, tail) if p.shape[0] > 0]
		out = torch.cat(parts, dim=0) if len(parts) > 1 else parts[0]
		return (out, start, start + M)
