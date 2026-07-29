"""Cut Video — extract an exact, contiguous range of frames.

``start_index`` follows Python indexing: zero is the first frame, ``-1`` is
the last frame, ``-8`` is the eighth frame from the end. ``frame_count`` is
the exact number of frames to return.

Unlike a Python slice, this node does not silently shorten an invalid request.
It raises a useful error when the start is outside the clip or when the
requested range runs past its end. This keeps an accidentally short video
from flowing unnoticed into the rest of a workflow.
"""

from __future__ import annotations

from ...base import TiNode
from ...registry import register


def cut_bounds(length: int, start_index: int, frame_count: int) -> tuple[int, int]:
	"""Validate a cut request and return its resolved ``(start, end)`` bounds."""
	if length <= 0:
		raise ValueError("Cut Video: the input image batch contains no frames.")

	start = int(start_index)
	count = int(frame_count)

	if count <= 0:
		raise ValueError(
			f"Cut Video: frame_count must be at least 1; received {count}."
		)
	if start < -length or start >= length:
		raise IndexError(
			f"Cut Video: start_index {start} is outside a {length}-frame video. "
			f"Use an index from {-length} to {length - 1}."
		)

	resolved = start + length if start < 0 else start
	end = resolved + count
	if end > length:
		available = length - resolved
		raise ValueError(
			f"Cut Video: {count} frame(s) requested from index {start}, but only "
			f"{available} frame(s) are available through the end of this "
			f"{length}-frame video."
		)
	return resolved, end


@register
class CutVideo(TiNode):
	DISPLAY_NAME = "Cut Video · Start + Frame Count (ti)"
	CATEGORY = "tinode/image"

	@classmethod
	def INPUT_TYPES(cls):
		return {
			"required": {
				"images": ("IMAGE",),
				"start_index": ("INT", {
					"default": 0,
					"min": -1000000,
					"max": 1000000,
					"step": 1,
					"tooltip": "First frame to keep. Negative values count from "
							   "the end, like Python (-1 is the last frame).",
				}),
				"frame_count": ("INT", {
					"default": 1,
					"min": 1,
					"max": 1000000,
					"step": 1,
					"tooltip": "Exact number of consecutive frames to return.",
				}),
			},
		}

	RETURN_TYPES = ("IMAGE",)
	RETURN_NAMES = ("images",)
	FUNCTION = "execute"

	def execute(self, images, start_index=0, frame_count=1):
		if not hasattr(images, "dim"):
			raise TypeError("Cut Video: images must be an IMAGE tensor.")
		if images.dim() not in (3, 4):
			raise ValueError(
				"Cut Video: expected an IMAGE shaped [H,W,C] or [N,H,W,C], "
				f"but received shape {tuple(images.shape)}."
			)

		batch = images.unsqueeze(0) if images.dim() == 3 else images
		start, end = cut_bounds(batch.shape[0], start_index, frame_count)
		return (batch[start:end],)
