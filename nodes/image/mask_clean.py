"""Mask Clean Islands — delete speckle and fill pinholes in a segmentation mask.

SAM predicts low-resolution logits, upsamples them, then thresholds. Pixels
near the decision boundary flip on their own, so masks come out freckled with
stray foreground dots and pinholes inside the subject. This removes both,
using connected components — the only cleanup that leaves the boundary of the
real mask exactly where it was. Morphological open/close would clear the same
noise, but it erodes and dilates the outline, and `Mask Crop Paste Back`
blends through this mask: a boundary shifted by a pixel is a visible seam.

Each frame is handled independently. Two passes:
  islands  blobs whose confident core is smaller than `min_island_area` are
           zeroed across their whole extent.
  holes    background components that touch NO image border and are no larger
           than `max_hole_area` are set to 1.

Two levels matter, because a SAM mask is not binary:
  support  `mask > soft_floor` — every pixel with any mask value at all,
           including the grey falloff around a blob. Blobs are found here.
  core     `mask > threshold` — the confident interior. Blob area is measured
           here, and holes are judged against it.

Splitting them is what makes the cleanup actually erase a speck. Labelling on
the core alone leaves the speck's grey halo behind as a ghost, and a speck
that never rises above `threshold` is not a component at all, so it survives
untouched — dim, but still painted into every alpha blend downstream. Labelling
on the support instead means a dropped blob is erased halo and all, and a blob
whose core area is zero is dropped by definition.

Measuring on the core is what keeps the subject safe: its halo is part of its
support component, so it is never counted as separate speckle, and is never
written to.

The main mask is never multiplied, grown, eroded or blurred. Only pixels of a
deleted island or a filled hole are written; every other pixel keeps its exact
original float value, so antialiased edges survive.

Connectivity is deliberately not exposed. Digital topology admits one correct
pairing here — 8-connected foreground against 4-connected background. Using 8
for both (as SAM's own remove_small_regions does) lets a hole ringed by a thin
diagonal boundary leak to the exterior through the diagonal gaps, so it never
gets filled. Using 4 for both shatters diagonal foreground chains into specks
that `min_island_area` then deletes. Neither is a knob worth offering.

Caveat: if a speck's halo touches the subject's halo they are one support
component, so the speck rides along on the subject's core area and survives.
Raise `soft_floor` to break such bridges.
"""

from __future__ import annotations

import torch

from ...base import TiNode
from ...registry import register

# Foreground components: 8-connected. Background (holes): 4-connected.
_FG_CONNECTIVITY = 8
_BG_CONNECTIVITY = 4


def _cv2():
	try:
		import cv2
	except ImportError as exc:  # pragma: no cover — environment problem, not logic
		raise RuntimeError(
			"Mask Clean Islands needs opencv (cv2). Install opencv-python."
		) from exc
	return cv2


def _clean_frame(core, support, min_island_area: int, max_hole_area: int, keep_largest: bool):
	"""Return (removed, filled) boolean maps for one [H,W] frame."""
	import numpy as np

	cv2 = _cv2()
	H, W = core.shape
	removed = np.zeros((H, W), dtype=bool)

	if min_island_area > 0 and support.any():
		count, labels = cv2.connectedComponents(
			support.astype(np.uint8), connectivity=_FG_CONNECTIVITY
		)
		# Blobs live on the support, but only their confident core counts toward
		# area — so a blob that never crosses `threshold` has area 0 and dies.
		core_areas = np.bincount(labels[core], minlength=count)
		core_areas[0] = 0                            # label 0 is the background
		drop = core_areas[1:] < min_island_area
		if keep_largest and drop.size and drop.all() and core_areas[1:].max() > 0:
			# Everything is below the threshold — spare the biggest rather than
			# returning an empty mask. A frame of pure grey has no core to spare.
			drop[int(core_areas[1:].argmax())] = False
		if drop.any():
			lut = np.zeros(count, dtype=bool)        # label -> was it dropped
			lut[1:] = drop
			removed = lut[labels]                    # erases the halo too

	foreground = core & ~removed
	filled = np.zeros((H, W), dtype=bool)

	if max_hole_area > 0 and not foreground.all():
		count, labels, stats, _ = cv2.connectedComponentsWithStats(
			(~foreground).astype(np.uint8), connectivity=_BG_CONNECTIVITY
		)
		lefts = stats[1:, cv2.CC_STAT_LEFT]
		tops = stats[1:, cv2.CC_STAT_TOP]
		widths = stats[1:, cv2.CC_STAT_WIDTH]
		heights = stats[1:, cv2.CC_STAT_HEIGHT]
		areas = stats[1:, cv2.CC_STAT_AREA]
		# A hole is enclosed: it reaches no edge of the frame. Size alone is not
		# enough — on a close-up the exterior background can be small too, and
		# filling it would flood the whole frame.
		touches_border = (lefts == 0) | (tops == 0) | (lefts + widths == W) | (tops + heights == H)
		fill = ~touches_border & (areas <= max_hole_area)
		if fill.any():
			lut = np.zeros(count, dtype=bool)
			lut[1:] = fill
			filled = lut[labels]

	return removed, filled


@register
class MaskCleanIslands(TiNode):
	DISPLAY_NAME = "Mask Clean Islands (ti)"
	CATEGORY = "tinode/image"

	@classmethod
	def INPUT_TYPES(cls):
		return {
			"required": {
				"mask": ("MASK",),
				"min_island_area": ("INT", {"default": 64, "min": 0, "max": 16777216, "step": 1}),
				"max_hole_area": ("INT", {"default": 64, "min": 0, "max": 16777216, "step": 1}),
			},
			"optional": {
				"threshold": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),
				"soft_floor": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01}),
				"keep_largest": ("BOOLEAN", {"default": False}),
			},
		}

	RETURN_TYPES = ("MASK",)
	RETURN_NAMES = ("mask",)
	FUNCTION = "execute"

	def execute(self, mask, min_island_area=64, max_hole_area=64,
				threshold=0.5, soft_floor=0.0, keep_largest=False):
		min_island_area = int(min_island_area)
		max_hole_area = int(max_hole_area)
		if min_island_area <= 0 and max_hole_area <= 0:
			return (mask,)

		squeeze = mask.dim() == 2
		batch = mask.unsqueeze(0) if squeeze else mask

		out = batch.clone()
		core = (batch > float(threshold)).cpu().numpy()
		# `| core` keeps support a superset even if soft_floor is set above threshold.
		support = (batch > float(soft_floor)).cpu().numpy() | core

		for i in range(batch.shape[0]):
			removed, filled = _clean_frame(
				core[i], support[i], min_island_area, max_hole_area, bool(keep_largest)
			)
			if removed.any():
				out[i][torch.from_numpy(removed).to(out.device)] = 0.0
			if filled.any():
				out[i][torch.from_numpy(filled).to(out.device)] = 1.0

		return (out[0] if squeeze else out,)
