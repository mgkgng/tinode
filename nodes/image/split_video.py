"""Split Video · Chunks — cut a clip into pieces and loop over them.

Type the frames where a new chunk starts — `18, 124` gives three chunks:
0..17, 18..123, 124..end — and each comes out as one item of an Inspire
ITEM_LIST, so a ▶Foreach List processes them one at a time. Chunk Item unpacks
the current one inside the loop; Join Chunks accumulates the processed pieces
back into a single clip.

Lossless by construction: chunking is a tensor slice and joining is a
concatenation, so every output frame is an input frame, untouched — nothing is
resampled, re-encoded or blended. Split then join with no processing in between
returns the original clip bit-for-bit.

`max_length` additionally caps chunk length, which is what makes a clip fit a
model's frame ceiling (VOID takes 197, and Frame Pad prepends >= 8, so 189 is
the useful cap). It splits evenly rather than leaving a long chunk plus a short
remainder — an 8-frame tail would inpaint far worse than two balanced halves.
"""

from __future__ import annotations

import torch

from ...base import TiNode, first
from ...registry import register
from .crop_bbox_manual import _save_frame_assets
from .pick_segments import _img_signature


def parse_cuts(spec, length):
	"""'18, 124' -> [18, 124]: the frames where a new chunk begins.

	Tolerant on purpose — separators can be commas, spaces or newlines. Points
	outside 1..length-1 are dropped (0 and length would make empty chunks), and
	duplicates are collapsed, so a sloppy string still yields a sane split
	instead of an error mid-batch. Negative values index from the end, matching
	the rest of the pack (-1 = the last frame).
	"""
	out = []
	for tok in str(spec or "").replace(",", " ").replace("\n", " ").split():
		try:
			i = int(tok)
		except ValueError:
			continue
		if i < 0:
			i += length
		if 0 < i < length and i not in out:
			out.append(i)
	return sorted(out)


def chunk_bounds(length, cuts, max_length=0):
	"""[(start, end), ...] half-open spans covering 0..length.

	`cuts` are the user's split points; `max_length` then subdivides any span
	still longer than it, into equal parts (ceil division) so no piece is a
	stub.
	"""
	length = int(length)
	if length <= 0:
		return []
	edges = [0] + list(cuts) + [length]
	spans = []
	for a, b in zip(edges, edges[1:]):
		if b <= a:
			continue
		n = b - a
		if max_length and n > max_length:
			parts = -(-n // int(max_length))          # ceil: fewest parts that fit
			base, extra = divmod(n, parts)
			s = a
			for k in range(parts):
				e = s + base + (1 if k < extra else 0)
				spans.append((s, e))
				s = e
		else:
			spans.append((a, b))
	return spans


@register
class SplitVideoChunks(TiNode):
	DISPLAY_NAME = "Split Video · Chunks (ti)"
	CATEGORY = "tinode/video"
	# Run every queue so the editor always has fresh frames to scrub, even when
	# the chunks are not consumed yet.
	OUTPUT_NODE = True

	@classmethod
	def INPUT_TYPES(cls):
		return {
			"required": {
				"images": ("IMAGE", {"tooltip": "The clip to cut."}),
				"cuts": ("STRING", {"default": "", "tooltip":
					"Frames where a new chunk STARTS, e.g. `18, 124` -> 0..17, "
					"18..123, 124..end. Empty = one chunk (or split by max_length)."}),
			},
			"optional": {
				"max_length": ("INT", {"default": 0, "min": 0, "max": 100000, "step": 1,
					"tooltip": "Also cap chunk length (0 = off). 189 fits VOID's 197 "
							   "once Frame Pad prepends 8. Long spans split evenly."}),
				"mask": ("MASK", {"tooltip":
					"Optional: cut in lockstep so each chunk keeps its own mask."}),
			},
		}

	RETURN_TYPES = ("ITEM_LIST", "INT")
	RETURN_NAMES = ("item_list", "count")
	OUTPUT_TOOLTIPS = (
		"One item per chunk — wire to ▶Foreach List, then Chunk Item inside.",
		"How many chunks.",
	)
	FUNCTION = "execute"

	def execute(self, images, cuts="", max_length=0, mask=None):
		imgs = first(images)
		if imgs.dim() != 4:
			imgs = imgs.unsqueeze(0)
		N = int(imgs.shape[0])
		m = first(mask)
		if isinstance(m, torch.Tensor) and m.dim() == 2:
			m = m.unsqueeze(0)

		spans = chunk_bounds(N, parse_cuts(cuts, N), int(first(max_length, 0)))
		if not spans:
			raise RuntimeError("Split Video · Chunks: nothing to split (0 frames).")

		items = []
		for i, (a, b) in enumerate(spans):
			item = {
				"images": imgs[a:b].contiguous(),
				"start": a, "end": b, "length": b - a,
				"chunk_index": i, "chunk_count": len(spans),
			}
			if isinstance(m, torch.Tensor):
				item["mask"] = m[a:b].contiguous() if m.shape[0] == N else m
			items.append(item)
		print(f"[tinode] Split Video · Chunks: {N} frames -> "
			  f"{len(spans)} chunk(s) {[(a, b) for a, b in spans]}")
		result = (list(items), len(items))

		# The editor needs the frames to scrub: you pick cut points by LOOKING at
		# the video, so shipping only numbers would make this node guesswork.
		manifest = _save_frame_assets(imgs, _img_signature(imgs), folder="ti_cuts")
		if manifest is None:
			return result
		manifest["spans"] = [[a, b] for a, b in spans]
		return {"ui": {"ti_cuts": [manifest]}, "result": result}


@register
class ChunkItem(TiNode):
	DISPLAY_NAME = "Chunk Item (ti)"
	CATEGORY = "tinode/video"

	@classmethod
	def INPUT_TYPES(cls):
		return {"required": {"item": ("TI_CHUNK_ITEM", {"tooltip":
			"One chunk from Split Video · Chunks, via ForeachListBegin's `item`."})}}

	RETURN_TYPES = ("IMAGE", "MASK", "INT", "INT", "INT", "INT")
	RETURN_NAMES = ("images", "mask", "start", "end", "chunk_index", "chunk_count")
	FUNCTION = "execute"

	def execute(self, item):
		it = first(item)
		if not isinstance(it, dict) or "images" not in it:
			raise RuntimeError(
				"Chunk Item: `item` must come from Split Video · Chunks via "
				"ForeachListBegin.")
		imgs = it["images"]
		m = it.get("mask")
		if not isinstance(m, torch.Tensor):
			m = torch.zeros((imgs.shape[0], imgs.shape[1], imgs.shape[2]), dtype=torch.float32)
		return (imgs, m, int(it["start"]), int(it["end"]),
				int(it["chunk_index"]), int(it["chunk_count"]))


@register
class JoinChunks(TiNode):
	DISPLAY_NAME = "Join Chunks (ti)"
	CATEGORY = "tinode/video"

	@classmethod
	def INPUT_TYPES(cls):
		return {
			"required": {
				"chunk": ("IMAGE", {"tooltip":
					"This iteration's processed chunk — appended after the accumulator."}),
			},
			"optional": {
				"accumulator": ("IMAGE", {"tooltip":
					"Wire ForeachListBegin.intermediate_output here. Anything that "
					"isn't an IMAGE (the loop's seed) means 'nothing yet', so the "
					"first chunk passes straight through."}),
			},
		}

	RETURN_TYPES = ("IMAGE", "INT")
	RETURN_NAMES = ("images", "frame_count")
	OUTPUT_TOOLTIPS = (
		"The chunks so far, in order — feed back to ForeachListEnd."
		"intermediate_output; after the last iteration it is the whole clip.",
		"Frames accumulated so far.",
	)
	FUNCTION = "execute"

	def execute(self, chunk, accumulator=None):
		new = first(chunk)
		if not isinstance(new, torch.Tensor):
			raise RuntimeError("Join Chunks: `chunk` must be an IMAGE.")
		if new.dim() != 4:
			new = new.unsqueeze(0)
		acc = first(accumulator)
		# Not an IMAGE means no accumulator yet: unconnected, or the loop's seed
		# value on the first pass. Same tolerance as Video Concatenate.
		if not isinstance(acc, torch.Tensor) or acc.dim() != 4:
			return (new, int(new.shape[0]))
		if acc.shape[1:] != new.shape[1:]:
			raise RuntimeError(
				f"Join Chunks: chunk is {tuple(new.shape[1:])} but the accumulator "
				f"is {tuple(acc.shape[1:])} — chunks must share resolution/channels.")
		out = torch.cat([acc, new.to(acc.dtype)], dim=0)
		return (out, int(out.shape[0]))
