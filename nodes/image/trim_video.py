"""Trim Video — cut a number of frames off the head and/or tail of a clip.

The video equivalent of a string trim (ltrim/rtrim): `trim_start` frames come
off the front, `trim_end` frames off the back, and what is left in the middle
is the output. Nothing is resampled — the kept frames are the original pixels.

Pairs with Extend Video and Insert Video, both of which report how many frames
they added and where, so you can put a clip back exactly as it was.

At least one frame always survives: an empty IMAGE batch breaks everything
downstream, so a trim that would consume the whole clip is clamped and says so
rather than emitting nothing.
"""

from __future__ import annotations

from ...base import TiNode
from ...registry import register
from .extend_video import _as_batch


@register
class TrimVideo(TiNode):
	DISPLAY_NAME = "Trim Video · Cut Frames (ti)"
	CATEGORY = "tinode/image"

	@classmethod
	def INPUT_TYPES(cls):
		return {
			"required": {
				"image": ("IMAGE",),
				"trim_start": ("INT", {"default": 0, "min": 0, "max": 1000000, "step": 1,
					"tooltip": "Frames to cut from the BEGINNING."}),
				"trim_end": ("INT", {"default": 0, "min": 0, "max": 1000000, "step": 1,
					"tooltip": "Frames to cut from the END."}),
			},
		}

	RETURN_TYPES = ("IMAGE", "INT")
	RETURN_NAMES = ("images", "frame_count")
	FUNCTION = "execute"

	def execute(self, image, trim_start=0, trim_end=0):
		base = _as_batch(image)
		N = base.shape[0]
		s = max(0, int(trim_start))
		e = max(0, int(trim_end))

		if s + e >= N:
			# Would leave nothing. Keep a single frame instead of emitting an
			# empty batch, and be loud about it rather than silently truncating.
			s = min(s, N - 1)
			e = min(e, N - 1 - s)
			print(f"[tinode] Trim Video: trim_start+trim_end >= {N} frames — "
				  f"clamped to keep {N - s - e} frame(s).")

		out = base[s:N - e] if e > 0 else base[s:]
		return (out, out.shape[0])
