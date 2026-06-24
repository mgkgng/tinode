"""Batch Drop Indices — remove chosen frames from an IMAGE batch.

Indices are given as a comma-separated string, e.g. "2, 3". Validation is
strict: every token must be a plain non-negative integer. If ANY token is
malformed (a dot, a sign, letters, an empty slot from a trailing comma) the
node does nothing and returns the batch unchanged.

By default indices are 1-based to match human counting ("2" = second
frame). Flip one_based off for 0-based. Out-of-range indices are ignored.
Dropping every frame would yield an empty batch, so that case is a no-op too.
"""

from __future__ import annotations

import re

import torch

from ...base import TiNode
from ...registry import register

_INT_TOKEN = re.compile(r"^[0-9]+$")


def parse_keep(spec: str, n: int, one_based: bool):
	"""Return the list of indices to KEEP, or None if the spec is invalid.

	None => caller should no-op (return the batch unchanged). An empty/space
	spec is valid and drops nothing. Out-of-range drop indices are ignored.
	"""
	s = spec.strip()
	if s == "":
		return list(range(n))                    # nothing to drop

	tokens = [t.strip() for t in s.split(",")]
	if not all(_INT_TOKEN.match(t) for t in tokens):
		return None                              # malformed -> signal no-op

	drop = set()
	for t in tokens:
		i = int(t)
		if one_based:
			i -= 1
		if 0 <= i < n:
			drop.add(i)

	keep = [i for i in range(n) if i not in drop]
	if not keep:
		return None                              # dropping all -> no-op
	return keep


@register
class BatchDropIndices(TiNode):
	DISPLAY_NAME = "Batch Drop Indices (ti)"
	CATEGORY = "tinode/image"

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

	def execute(self, images, indices, one_based=True):
		keep = parse_keep(indices, images.shape[0], one_based)
		if keep is None:
			return (images,)                     # invalid or empty result -> unchanged
		return (images[torch.tensor(keep, dtype=torch.long)],)
