"""Segment Mask · Select — one object's mask, picked by index.

For stepping through objects one at a time into a mask-driven node like Mask
Bbox Crop (ti): unlike Segments to Masks (which emits a LIST that a list-consuming
node would fuse back together), this outputs a SINGLE [frames,H,W] MASK for the
`index`-th object, so it wires straight into a mask input.

Bump `index` (0 .. count-1) to move to the next object and re-run; `count` tells
you how many there are, and `id` is the track id you landed on. `object_ids`
optionally restricts the set you're stepping through.
"""

from __future__ import annotations

import torch

from ...base import TiNode
from ...registry import register
from ...schema import validate_segments
from .segments_to_masks import _parse_ids, build_id_mask


@register
class SegmentMaskSelect(TiNode):
	DISPLAY_NAME = "Segment Mask · Select (ti)"
	CATEGORY = "tinode/image"

	@classmethod
	def INPUT_TYPES(cls):
		return {
			"required": {
				"segments": ("TI_SAM3_SEGMENTS",),
				"index": ("INT", {"default": 0, "min": 0, "max": 99999, "step": 1,
					"tooltip": "Which object (0-based) among the available ids. "
							   "Clamped to the last one."}),
			},
			"optional": {
				"object_ids": ("STRING", {"default": "-1",
					"tooltip": "Restrict the set you're stepping through "
							   "(comma-separated). -1 or empty = all."}),
			},
		}

	RETURN_TYPES = ("MASK", "INT", "INT")
	RETURN_NAMES = ("mask", "id", "count")
	FUNCTION = "execute"

	def execute(self, segments, index=0, object_ids="-1"):
		validate_segments(segments)
		H = int(segments["height"])
		W = int(segments["width"])
		N = int(segments.get("num_frames", len(segments.get("frames", []))))

		ids = _parse_ids(object_ids, list(segments.get("ids", [])))
		count = len(ids)
		if count == 0:
			return (torch.zeros((N, H, W), dtype=torch.float32), -1, 0)

		i = max(0, min(int(index), count - 1))      # clamp into range
		seg_id = ids[i]
		return (build_id_mask(segments, seg_id), int(seg_id), count)
