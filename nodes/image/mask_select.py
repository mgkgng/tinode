"""Mask Drop / Pick Indices — the batch-select nodes, for MASK batches.

ComfyUI types are strict: the IMAGE Batch Drop/Pick nodes can't take a
MASK. These mirror them exactly for MASK input, reusing the same parse
logic so behaviour stays identical (strict validation, 1-based default,
out-of-range skipped, malformed/empty -> no-op).
"""

from __future__ import annotations

import torch

from ...base import TiNode
from ...registry import register
from .batch_drop import parse_keep
from .batch_pick import parse_pick


def _collapse(masks):
	"""List or tensor of masks -> one [N,H,W] batch."""
	items = masks if isinstance(masks, list) else [masks]
	norm = [m if m.dim() == 3 else m.unsqueeze(0) for m in items]
	return torch.cat(norm, dim=0)


def _first(v, default=None):
	if isinstance(v, list):
		return v[0] if v else default
	return v


@register
class MaskDropIndices(TiNode):
	DISPLAY_NAME = "Mask Drop Indices (ti)"
	CATEGORY = "tinode/image"
	INPUT_IS_LIST = True

	@classmethod
	def INPUT_TYPES(cls):
		return {
			"required": {
				"masks": ("MASK",),
				"indices": ("STRING", {"default": "", "multiline": False}),
			},
			"optional": {
				"one_based": ("BOOLEAN", {"default": True}),
			},
		}

	RETURN_TYPES = ("MASK",)
	RETURN_NAMES = ("masks",)
	FUNCTION = "execute"

	def execute(self, masks, indices, one_based=True):
		batch = _collapse(masks)
		keep = parse_keep(_first(indices, ""), batch.shape[0], _first(one_based, True))
		if keep is None:
			return (batch,)
		return (batch[torch.tensor(keep, dtype=torch.long)],)


@register
class MaskPickIndices(TiNode):
	DISPLAY_NAME = "Mask Pick Indices (ti)"
	CATEGORY = "tinode/image"
	INPUT_IS_LIST = True

	@classmethod
	def INPUT_TYPES(cls):
		return {
			"required": {
				"masks": ("MASK",),
				"indices": ("STRING", {"default": "", "multiline": False}),
			},
			"optional": {
				"one_based": ("BOOLEAN", {"default": True}),
			},
		}

	RETURN_TYPES = ("MASK",)
	RETURN_NAMES = ("masks",)
	FUNCTION = "execute"

	def execute(self, masks, indices, one_based=True):
		batch = _collapse(masks)
		keep = parse_pick(_first(indices, ""), batch.shape[0], _first(one_based, True))
		if keep is None:
			return (batch,)
		return (batch[torch.tensor(keep, dtype=torch.long)],)
