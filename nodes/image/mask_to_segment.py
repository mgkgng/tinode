"""Convert a MASK batch into editable TI_SAM3_SEGMENTS data."""

from __future__ import annotations

import math

import torch

from ...base import TiNode
from ...registry import register


def mask_to_segments(mask, threshold=0.5, segment_id=1):
	"""Treat the whole mask as one tracked object across all input frames."""
	if not isinstance(mask, torch.Tensor):
		raise TypeError(
			f"Mask to Segment: mask must be a torch tensor, got "
			f"{type(mask).__name__}."
		)
	if mask.dim() == 2:
		masks = mask.unsqueeze(0)
	elif mask.dim() == 3:
		masks = mask
	else:
		raise ValueError(
			"Mask to Segment: expected MASK shape [H,W] or [N,H,W], "
			f"but received {tuple(mask.shape)}."
		)

	N, H, W = (int(v) for v in masks.shape)
	if N < 1 or H < 1 or W < 1:
		raise ValueError(
			"Mask to Segment: every dimension must be non-zero; "
			f"received shape {tuple(masks.shape)}."
		)

	try:
		level = float(threshold)
	except (TypeError, ValueError) as exc:
		raise ValueError(
			f"Mask to Segment: threshold must be a number, got {threshold!r}."
		) from exc
	if not math.isfinite(level) or not 0.0 <= level <= 1.0:
		raise ValueError(
			f"Mask to Segment: threshold must be between 0 and 1; received {level}."
		)

	try:
		sid = int(segment_id)
	except (TypeError, ValueError) as exc:
		raise ValueError(
			f"Mask to Segment: segment_id must be an integer, got {segment_id!r}."
		) from exc
	if sid < 0:
		raise ValueError(
			f"Mask to Segment: segment_id must be non-negative; received {sid}."
		)
	if masks.is_floating_point() and not bool(torch.isfinite(masks).all()):
		raise ValueError("Mask to Segment: the mask contains NaN or infinite values.")

	# Pick/Add Segments create CPU canvases. Keeping compact payloads on CPU
	# avoids device mismatches and stores only each frame's occupied rectangle.
	binary = (masks > level).detach().to(device="cpu", dtype=torch.uint8)
	frames = []
	has_segment = False
	for frame in binary:
		points = torch.nonzero(frame, as_tuple=False)
		if points.numel() == 0:
			frames.append([])
			continue

		y0 = int(points[:, 0].min())
		y1 = int(points[:, 0].max()) + 1
		x0 = int(points[:, 1].min())
		x1 = int(points[:, 1].max()) + 1
		frames.append([{
			"id": sid,
			"bbox": [x0, y0, x1, y1],
			"conf": 1.0,
			"mask": frame[y0:y1, x0:x1].clone(),
		}])
		has_segment = True

	return {
		"num_frames": N,
		"height": H,
		"width": W,
		"frames": frames,
		"ids": [sid] if has_segment else [],
	}


@register
class MaskToSegment(TiNode):
	DISPLAY_NAME = "Mask to Segment (ti)"
	CATEGORY = "tinode/image"

	@classmethod
	def INPUT_TYPES(cls):
		return {
			"required": {
				"mask": ("MASK",),
				"threshold": ("FLOAT", {
					"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01,
					"tooltip": "Pixels strictly above this value belong to the segment.",
				}),
			},
			"optional": {
				"segment_id": ("INT", {
					"default": 1, "min": 0, "max": 999999, "step": 1,
					"tooltip": "Stable object id used on every non-empty frame.",
				}),
			},
		}

	RETURN_TYPES = ("TI_SAM3_SEGMENTS",)
	RETURN_NAMES = ("segments",)
	FUNCTION = "execute"

	def execute(self, mask, threshold=0.5, segment_id=1):
		return (mask_to_segments(mask, threshold, segment_id),)
