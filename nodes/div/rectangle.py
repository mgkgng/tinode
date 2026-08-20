"""Divide · Rectangle — cut an image into a rows x cols grid of tiles.

The first of the tinode/div family: hand it a frame (or a whole video batch) and
it returns the tiles, reading left-to-right then top-to-bottom.

Divide two ways, whichever the job is phrased in:
  by = count  a rows x cols grid ("split it in 3x4")
  by = size   tiles of about tile_width x tile_height ("give me 512px tiles").
              By default the frame is spread EVENLY into the tile count closest
              to that size, so tiles stay uniform and no sliver is left over;
              exact_size keeps them at exactly the target and lets the last one
              be the remainder, for a model that demands a fixed tile size.

Two ways to take the tiles, because both are useful:
  tiles       a LIST — one [N,th,tw,C] IMAGE per cell, so downstream nodes run
              once per tile (and a tile keeps the batch, so a video stays a video)
  tile_grid   the same cells stacked into ONE batch, for a quick preview or a
              node that wants them all at once

`overlap` grows each tile outward by that many pixels (clamped at the frame
edge), which is what you want when the tiles are processed independently and
then blended back — a seam needs shared pixels to blend across.

Lossless: every tile is a plain tensor slice at native scale, so the pixels are
bit-exact. `crop_infos` maps each tile back to its place in the source frame, so
Mask Crop Paste Back / Composite Crops can reassemble them exactly. When the
size doesn't divide evenly the remainder is spread over the first cells rather
than left as a thin strip at the end.
"""

from __future__ import annotations

import torch

from ...base import TiNode, first
from ...registry import register


def split_spans(total, parts, overlap=0):
	"""[(start, end), ...] covering 0..total in `parts`, remainder spread first.

	`overlap` then grows each span outward, clamped to the frame — so tiles share
	a margin for blending instead of butting up against a hard seam.
	"""
	total = int(total)
	parts = max(1, int(parts))
	parts = min(parts, total) or 1          # never emit an empty tile
	base, extra = divmod(total, parts)
	spans, s = [], 0
	for i in range(parts):
		e = s + base + (1 if i < extra else 0)
		spans.append((s, e))
		s = e
	if overlap:
		o = int(overlap)
		spans = [(max(0, a - o), min(total, b + o)) for a, b in spans]
	return spans


def parts_for_size(total, size, exact=False):
	"""How many tiles of about `size` px fit across `total`.

	Default (exact False) picks the count whose EVEN division lands closest to
	`size`, so the tiles stay uniform and nothing is left over — 1000px asked in
	300s gives 3 tiles of ~333 rather than 3x300 plus a 100px sliver.

	exact True keeps tiles at exactly `size` and lets the last one be the
	remainder instead — use it when the model demands a fixed tile size.
	"""
	total = int(total)
	size = max(1, int(size))
	if total <= 0:
		return 1
	if exact:
		return max(1, -(-total // size))          # ceil: last tile is the remainder
	return max(1, round(total / size))            # closest even division


def size_spans(total, size, overlap=0, exact=False):
	"""[(start, end), ...] covering 0..total in tiles of about `size` px."""
	total = int(total)
	if exact:
		size = max(1, int(size))
		spans = [(s, min(total, s + size)) for s in range(0, total, size)] or [(0, total)]
		if overlap:
			o = int(overlap)
			spans = [(max(0, a - o), min(total, b + o)) for a, b in spans]
		return spans
	return split_spans(total, parts_for_size(total, size), overlap)


@register
class DivideRectangle(TiNode):
	DISPLAY_NAME = "Divide · Rectangle (ti)"
	CATEGORY = "tinode/div"

	@classmethod
	def INPUT_TYPES(cls):
		return {
			"required": {
				"image": ("IMAGE", {"tooltip":
					"The frame (or video batch) to divide. Every frame is cut the same."}),
				"rows": ("INT", {"default": 2, "min": 1, "max": 256, "step": 1,
					"tooltip": "Horizontal divisions (by = count)."}),
				"cols": ("INT", {"default": 2, "min": 1, "max": 256, "step": 1,
					"tooltip": "Vertical divisions (by = count)."}),
			},
			"optional": {
				"overlap": ("INT", {"default": 0, "min": 0, "max": 4096, "step": 1,
					"tooltip": "Grow each tile outward by this many pixels (clamped at "
							   "the frame edge), so independently-processed tiles have "
							   "a margin to blend across."}),
				# APPENDED after `overlap` on purpose: widget values are positional,
				# so inserting one mid-list shifts every value after it in a saved
				# workflow (overlap's 0 landing in cols, etc).
				"by": (["count", "size"], {"default": "count", "tooltip":
					"count = a rows x cols grid; size = tiles of about "
					"tile_width x tile_height pixels."}),
				"tile_width": ("INT", {"default": 512, "min": 8, "max": 8192, "step": 8,
					"tooltip": "Target tile width in pixels (by = size)."}),
				"tile_height": ("INT", {"default": 512, "min": 8, "max": 8192, "step": 8,
					"tooltip": "Target tile height in pixels (by = size)."}),
				"exact_size": ("BOOLEAN", {"default": False, "tooltip":
					"by = size: off spreads the frame evenly into tiles CLOSEST to the "
					"target (uniform, no leftover sliver); on keeps tiles at EXACTLY "
					"the target and lets the last one be the remainder."}),
			},
		}

	RETURN_TYPES = ("IMAGE", "IMAGE", "TI_CROP_XFORM", "INT")
	RETURN_NAMES = ("tiles", "tile_grid", "crop_infos", "count")
	# tiles / crop_infos are LISTs — one per cell; tile_grid and count are single.
	OUTPUT_IS_LIST = (True, False, True, False)
	OUTPUT_TOOLTIPS = (
		"One IMAGE per tile (a list) — downstream runs once per tile.",
		"Every tile stacked into one batch, for a quick look.",
		"Each tile's place in the source frame — feed Paste Back / Composite Crops.",
		"How many tiles (rows x cols).",
	)
	FUNCTION = "execute"

	def execute(self, image, rows=2, cols=2, overlap=0, by="count",
				tile_width=512, tile_height=512, exact_size=False, **_stale):
		imgs = first(image)
		if imgs.dim() != 4:
			imgs = imgs.unsqueeze(0) if imgs.dim() == 3 else imgs
		N, H, W, C = imgs.shape
		# A saved graph from the brief version that had `by` in the middle can hand
		# these over shifted; clamp rather than fail the run.
		rows = max(1, int(first(rows, 2)))
		cols = max(1, int(first(cols, 2)))
		overlap = max(0, int(first(overlap, 0)))

		if first(by, "count") == "size":
			exact = bool(first(exact_size, False))
			row_spans = size_spans(H, int(first(tile_height, 512)), overlap, exact)
			col_spans = size_spans(W, int(first(tile_width, 512)), overlap, exact)
		else:
			row_spans = split_spans(H, rows, overlap)
			col_spans = split_spans(W, cols, overlap)

		tiles, infos = [], []
		for y0, y1 in row_spans:                     # left-to-right, top-to-bottom
			for x0, x1 in col_spans:
				tiles.append(imgs[:, y0:y1, x0:x1, :].contiguous())
				item = {"y0": y0, "x0": x0, "h": y1 - y0, "w": x1 - x0,
						"oy": 0, "ox": 0, "nh": y1 - y0, "nw": x1 - x0}
				infos.append({"H": H, "W": W, "C": C, "items": [item] * N})

		# Tiles are only uniform when the divisions came out even (and no clamped
		# overlap), so stacking is best-effort — the LIST output always works.
		shapes = {tuple(t.shape) for t in tiles}
		grid = torch.cat(tiles, dim=0) if len(shapes) == 1 else tiles[0]
		if len(shapes) != 1:
			print(f"[tinode] Divide · Rectangle: tiles differ in size {shapes} "
				  "(uneven division or clamped overlap) — tile_grid shows the first "
				  "one; use the `tiles` list output.")
		return (tiles, grid, infos, len(tiles))
