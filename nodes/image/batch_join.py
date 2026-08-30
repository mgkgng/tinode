"""Batch Join — put a list back together into one batch.

The other half of Mask Frames · Iterate. Once a list has been expanded, every
node downstream runs per item and KEEPS producing a list — so a video saver at
the end would write one file per frame instead of one film. INPUT_IS_LIST here
gathers the whole run into a single call and concatenates it, which also makes
the saver wait for every frame to finish.

Frames must agree on size, because a batch cannot hold two shapes. Rather than
let torch raise something opaque about dimension 1, this says which item broke
ranks and what it measured.
"""

from __future__ import annotations

import torch

from ...base import TiNode
from ...registry import register


def _stack(items, what, dims):
	"""Concatenate a list of tensors on dim 0, or explain why it cannot."""
	tensors = []
	for i, t in enumerate(items):
		if t is None:
			continue
		if not isinstance(t, torch.Tensor):
			raise RuntimeError(f"Batch Join: {what} item {i} is not a tensor.")
		if t.dim() == dims - 1:
			t = t.unsqueeze(0)
		if t.dim() != dims:
			raise RuntimeError(
				f"Batch Join: {what} item {i} has shape {tuple(t.shape)}; "
				f"expected {dims} dimensions.")
		if tensors and t.shape[1:] != tensors[0].shape[1:]:
			raise RuntimeError(
				f"Batch Join: {what} item {i} is {tuple(t.shape[1:])} but the "
				f"first is {tuple(tensors[0].shape[1:])}. Every frame of a batch "
				"must be the same size — scale them before joining.")
		tensors.append(t)
	if not tensors:
		return None
	return torch.cat(tensors, dim=0)


@register
class BatchJoin(TiNode):
	DISPLAY_NAME = "Batch Join (ti)"
	CATEGORY = "tinode/util"
	INPUT_IS_LIST = True

	@classmethod
	def INPUT_TYPES(cls):
		return {
			"required": {
				"images": ("IMAGE", {"tooltip":
					"A per-item IMAGE list — the output of a run that was "
					"expanded over a list. Joined into one batch, in order."}),
			},
			"optional": {
				"masks": ("MASK", {"tooltip":
					"Optional matching MASK list, joined the same way."}),
			},
		}

	RETURN_TYPES = ("IMAGE", "MASK", "INT")
	RETURN_NAMES = ("images", "masks", "count")
	OUTPUT_TOOLTIPS = (
		"One IMAGE batch — wire to the video saver.",
		"One MASK batch (empty when no masks were wired).",
		"How many frames went in.",
	)
	FUNCTION = "execute"

	def execute(self, images, masks=None):
		ilist = images if isinstance(images, list) else [images]
		out = _stack(ilist, "images", 4)
		if out is None:
			raise RuntimeError("Batch Join: nothing wired to `images`.")
		mlist = masks if isinstance(masks, list) else ([] if masks is None else [masks])
		mout = _stack(mlist, "masks", 3)
		if mout is None:
			mout = torch.zeros((out.shape[0], out.shape[1], out.shape[2]),
							   dtype=torch.float32)
		print(f"[tinode] Batch Join: {out.shape[0]} frame(s) of "
			  f"{out.shape[2]}x{out.shape[1]}")
		return (out, mout, int(out.shape[0]))
