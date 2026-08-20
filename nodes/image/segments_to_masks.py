"""Segments to Masks — split a TI_SAM3_SEGMENTS stream into per-object masks.

EasySAM3 Segment's own `mask` output is MERGED (every selected object unioned
into one [frames,H,W] mask), so you can't get the objects back out of it. This
node instead reads the unmerged `segments` output and rebuilds a full-frame mask
batch PER object id — so you get them one at a time instead of all fused.

The masks output is a LIST (OUTPUT_IS_LIST): downstream nodes iterate it one mask
at a time, and its Nth item lines up with the Nth id in the `ids` string. To pull
a single object, set `object_ids` to that id (e.g. "5") and the list has just it.

Each per-id mask is [frames, H, W]: for every frame the object appears in, its
bbox-cropped mask is pasted into a zeroed full frame; frames where it is absent
stay black.
"""

from __future__ import annotations

import torch

from ...base import TiNode
from ...registry import register
from ...schema import validate_segments


def build_id_mask(segments, seg_id):
	"""Full-frame [frames,H,W] mask for one object id across the whole clip.

	The object's bbox-cropped mask is pasted into a zeroed frame wherever it
	appears; frames where it is absent stay black.
	"""
	H = int(segments["height"])
	W = int(segments["width"])
	frames = segments.get("frames", [])
	N = int(segments.get("num_frames", len(frames)))
	m = torch.zeros((N, H, W), dtype=torch.float32)
	for f in range(N):
		for s in (frames[f] if f < len(frames) else []):
			if s["id"] == seg_id:
				x0, y0, x1, y1 = s["bbox"]
				m[f, y0:y1, x0:x1] = torch.maximum(
					m[f, y0:y1, x0:x1], s["mask"].to(torch.float32))
	return m


def bbox_mask(segments, height=None, width=None):
	"""Filled RECTANGLES over every segment's bbox, as a [frames,H,W] mask.

	The union of the boxes rather than the objects' outlines. An inpainting model
	is usually happier being told to repaint a rectangle than a ragged silhouette:
	the silhouette's edge is exactly where a slightly-tight mask leaves a halo of
	the thing you removed, whereas a box gives the model clean margin on every
	side. The cost is that everything else inside the box is regenerated too.
	"""
	H = int(height if height is not None else segments["height"])
	W = int(width if width is not None else segments["width"])
	frames = segments.get("frames", [])
	N = int(segments.get("num_frames", len(frames)))
	m = torch.zeros((N, H, W), dtype=torch.float32)
	for f in range(N):
		for s in (frames[f] if f < len(frames) else []):
			x0, y0, x1, y1 = s["bbox"]
			m[f, max(0, y0):min(H, y1), max(0, x0):min(W, x1)] = 1.0
	return m


def _parse_ids(spec, available):
	"""'' / '-1' -> all ids (sorted); '5,7' -> those that exist, in that order."""
	s = str(spec).strip()
	if s in ("", "-1"):
		return list(available)
	out = []
	for tok in s.split(","):
		tok = tok.strip()
		if not tok:
			continue
		try:
			i = int(tok)
		except ValueError:
			continue
		if i in available and i not in out:
			out.append(i)
	return out


@register
class SegmentsToMasks(TiNode):
	DISPLAY_NAME = "Segments to Masks (ti)"
	CATEGORY = "tinode/image"

	@classmethod
	def INPUT_TYPES(cls):
		return {
			"required": {
				"segments": ("TI_SAM3_SEGMENTS",),
			},
			"optional": {
				"object_ids": ("STRING", {"default": "-1",
					"tooltip": "Which objects to emit, comma-separated (e.g. 5,7). "
							   "-1 or empty = every object. One id = one mask."}),
			},
		}

	# masks is a LIST — one [frames,H,W] MASK per id, iterated one by one.
	RETURN_TYPES = ("MASK", "STRING", "INT")
	RETURN_NAMES = ("masks", "ids", "count")
	OUTPUT_IS_LIST = (True, False, False)
	FUNCTION = "execute"

	def execute(self, segments, object_ids="-1"):
		validate_segments(segments)
		H = int(segments["height"])
		W = int(segments["width"])
		N = int(segments.get("num_frames", len(segments.get("frames", []))))
		available = list(segments.get("ids", []))

		wanted = _parse_ids(object_ids, available)
		if not wanted:
			# Nothing to emit — a single empty frame keeps the list non-empty so
			# downstream doesn't choke on a zero-length list.
			return ([torch.zeros((N, H, W), dtype=torch.float32)], "", 0)

		masks = [build_id_mask(segments, i) for i in wanted]
		ids_str = ",".join(str(i) for i in wanted)
		return (masks, ids_str, len(masks))
