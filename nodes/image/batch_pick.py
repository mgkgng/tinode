"""Batch Pick Indices — keep only chosen frames from an IMAGE batch.

The inverse of Batch Drop Indices. Given a comma-separated list, e.g.
"2, 3", output a batch containing ONLY those frames, in the order listed
(so it doubles as a reorder/select). Validation is strict: any malformed
token (a dot, a sign, letters, an empty slot) makes the node a no-op and
returns the batch unchanged.

1-based by default ("2" = second frame); toggle one_based off for 0-based.
Out-of-range indices are skipped; duplicates are allowed (a frame can
repeat). If nothing valid remains the node no-ops rather than emit an
empty batch.
"""

from __future__ import annotations

import re

import torch

from ...base import TiNode
from ...registry import register

_INT_TOKEN = re.compile(r"^[0-9]+$")


def parse_pick(spec: str, n: int, one_based: bool):
	"""Return the list of indices to KEEP (in given order), or None to no-op.

	None => caller returns the batch unchanged (invalid spec, or nothing
	selectable). Out-of-range indices are skipped; order and duplicates are
	preserved from the spec.
	"""
	s = spec.strip()
	if s == "":
		return None                              # pick nothing -> no-op

	tokens = [t.strip() for t in s.split(",")]
	if not all(_INT_TOKEN.match(t) for t in tokens):
		return None                              # malformed -> no-op

	keep = []
	for t in tokens:
		i = int(t)
		if one_based:
			i -= 1
		if 0 <= i < n:
			keep.append(i)

	if not keep:
		return None                              # all out of range -> no-op
	return keep


@register
class BatchPickIndices(TiNode):
	DISPLAY_NAME = "Batch Pick Indices (ti)"
	CATEGORY = "tinode/image"

	# Consume the whole batch in one call (mirrors Batch Drop) so a LIST
	# source doesn't make ComfyUI run this per-frame.
	INPUT_IS_LIST = True

	@classmethod
	def INPUT_TYPES(cls):
		return {
			"required": {
				"images": ("IMAGE",),
				"indices": ("STRING", {"default": "", "multiline": False}),
			},
			"optional": {
				"one_based": ("BOOLEAN", {"default": True}),
			},
		}

	RETURN_TYPES = ("IMAGE",)
	RETURN_NAMES = ("images",)
	FUNCTION = "execute"

	@staticmethod
	def _first(v, default=None):
		if isinstance(v, list):
			return v[0] if v else default
		return v

	def execute(self, images, indices, one_based=True):
		imgs = images if isinstance(images, list) else [images]
		batch = torch.cat(
			[im if im.dim() == 4 else im.unsqueeze(0) for im in imgs], dim=0
		)
		idx = self._first(indices, "")
		ob = self._first(one_based, True)

		keep = parse_pick(idx, batch.shape[0], ob)
		if keep is None:
			return (batch,)                      # invalid or empty -> unchanged
		return (batch[torch.tensor(keep, dtype=torch.long)],)
