"""Crop Info Drop / Pick Indices — curate the TI_CROP_XFORM transform.

So you can Pick/Drop faces and keep images, masks AND crop_info in lockstep:
apply the same index string to all three. These filter the per-face `items`
list inside crop_info with the same parse logic as the image/mask selects
(strict validation, 1-based default, out-of-range skipped, malformed/empty
-> no-op). Pick preserves the given order so the transform matches a
reordered crop batch.
"""

from __future__ import annotations

from ...base import TiNode
from ...registry import register
from .batch_drop import parse_keep
from .batch_pick import parse_pick


def _rebuild(info, keep):
	items = info["items"]
	new_items = [items[i] for i in keep]
	out = dict(info)
	out["items"] = new_items
	return out


@register
class CropInfoDropIndices(TiNode):
	DISPLAY_NAME = "Crop Info Drop Indices (ti)"
	CATEGORY = "tinode/image"

	@classmethod
	def INPUT_TYPES(cls):
		return {
			"required": {
				"crop_info": ("TI_CROP_XFORM",),
				"indices": ("STRING", {"default": "", "multiline": False}),
			},
			"optional": {
				"one_based": ("BOOLEAN", {"default": True}),
			},
		}

	RETURN_TYPES = ("TI_CROP_XFORM",)
	RETURN_NAMES = ("crop_info",)
	FUNCTION = "execute"

	def execute(self, crop_info, indices, one_based=True):
		keep = parse_keep(indices, len(crop_info["items"]), one_based)
		if keep is None:
			return (crop_info,)
		return (_rebuild(crop_info, keep),)


@register
class CropInfoPickIndices(TiNode):
	DISPLAY_NAME = "Crop Info Pick Indices (ti)"
	CATEGORY = "tinode/image"

	@classmethod
	def INPUT_TYPES(cls):
		return {
			"required": {
				"crop_info": ("TI_CROP_XFORM",),
				"indices": ("STRING", {"default": "", "multiline": False}),
			},
			"optional": {
				"one_based": ("BOOLEAN", {"default": True}),
			},
		}

	RETURN_TYPES = ("TI_CROP_XFORM",)
	RETURN_NAMES = ("crop_info",)
	FUNCTION = "execute"

	def execute(self, crop_info, indices, one_based=True):
		keep = parse_pick(indices, len(crop_info["items"]), one_based)
		if keep is None:
			return (crop_info,)
		return (_rebuild(crop_info, keep),)
