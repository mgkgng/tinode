"""Frame Pad / Frame Unpad — force a clip's length to 4n+1 by head-padding.

Some video models (Netflix VOID, and the video VAE under it) only accept a clip
whose length is `4n + 1` — from
    video_length = (video_length - 1) // ratio * ratio + 1
with ratio = temporal_compression_ratio (4). Feed any other length and frames
get silently dropped.

On top of that, prepending a few copies of the first frame measurably reduces
error at the head of the result. So: prepend **at least `min_prepend`** copies
of frame 0, choosing the exact count that makes the *total* length land on 4n+1
(with min_prepend 8, ratio 4 that's one of {8, 9, 10, 11}), run the model, then
drop exactly that many frames off the front to recover the original clip.

  Frame Pad     images (L)            -> images (L+p), pad_count = p, total = L+p
                mask   (L)            -> mask   (L+p)
  [ your model on the padded clip ]
  Frame Unpad   images (L+p), p       -> images (L)

Pad the mask in lockstep so it stays aligned with the padded video; keep the
ORIGINAL (unpadded) mask for the final paste-back. Padding just repeats existing
frames — no interpolation, no quality loss.

The max length (e.g. VOID's 197) is handled upstream (chunking); this only
enforces the 4n+1 shape. Frame Pad warns if the padded total exceeds `max_frames`.
"""

from __future__ import annotations

import torch

from ...base import TiNode, first
from ...registry import register


def frame_pad_count(length, min_prepend=8, ratio=4):
	"""Frames to prepend so (length + p) is a multiple of ratio plus 1, p >= min_prepend.

	p is the unique value in [min_prepend, min_prepend + ratio - 1] with
	(length + p - 1) divisible by ratio. With ratio=4, min_prepend=8 that's one
	of {8, 9, 10, 11}.
	"""
	length = int(length)
	ratio = max(1, int(ratio))
	min_prepend = int(min_prepend)
	if length <= 0:
		return min_prepend
	return min_prepend + ((1 - length - min_prepend) % ratio)


@register
class FramePad(TiNode):
	DISPLAY_NAME = "Frame Pad (ti)"
	CATEGORY = "tinode/video"

	@classmethod
	def INPUT_TYPES(cls):
		return {
			"required": {
				"images": ("IMAGE", {"tooltip":
					"The clip to process. Frame 0 is duplicated at the front."}),
			},
			"optional": {
				"mask": ("MASK", {"tooltip":
					"Padded in lockstep so it stays aligned with the video. Keep "
					"the ORIGINAL mask for the final paste-back."}),
				"min_prepend": ("INT", {"default": 8, "min": 0, "max": 64, "step": 1,
					"tooltip": "Minimum duplicated head frames (VOID: 8 reduces error)."}),
				"temporal_ratio": ("INT", {"default": 4, "min": 1, "max": 16, "step": 1,
					"tooltip": "The length must be a multiple of this, plus 1 (4n+1). "
							   "= the VAE temporal_compression_ratio."}),
				"max_frames": ("INT", {"default": 197, "min": 1, "max": 100000, "step": 1,
					"tooltip": "Warn if the padded total exceeds this (e.g. VOID's "
							   "197). Chunk upstream to stay under it."}),
			},
		}

	RETURN_TYPES = ("IMAGE", "MASK", "INT", "INT")
	RETURN_NAMES = ("images", "mask", "pad_count", "total_frames")
	OUTPUT_TOOLTIPS = (
		"Head-padded clip, length 4n+1.",
		"Head-padded mask, aligned with the clip.",
		"How many frames were prepended — feed to Frame Unpad.",
		"Padded length (= original + pad_count).",
	)
	FUNCTION = "execute"

	def execute(self, images, mask=None, min_prepend=8, temporal_ratio=4, max_frames=197):
		imgs = first(images)
		if imgs.dim() != 4:
			imgs = imgs.unsqueeze(0) if imgs.dim() == 3 else imgs
		L, H, W, C = imgs.shape
		p = frame_pad_count(L, int(first(min_prepend, 8)), int(first(temporal_ratio, 4)))
		total = L + p
		cap = int(first(max_frames, 197))
		if total > cap:
			print(f"[tinode] Frame Pad: padded length {total} exceeds max_frames "
				  f"{cap} — chunk the clip upstream.")

		head = imgs[:1].repeat(p, 1, 1, 1) if p > 0 else imgs[:0]
		padded = torch.cat([head, imgs], dim=0) if p > 0 else imgs

		m = first(mask)
		if isinstance(m, torch.Tensor):
			if m.dim() == 2:
				m = m.unsqueeze(0)
			mhead = m[:1].repeat(p, 1, 1) if p > 0 else m[:0]
			padded_mask = torch.cat([mhead, m], dim=0) if p > 0 else m
		else:
			padded_mask = torch.zeros((total, H, W), dtype=torch.float32)

		return (padded, padded_mask, int(p), int(total))


@register
class FrameUnpad(TiNode):
	DISPLAY_NAME = "Frame Unpad (ti)"
	CATEGORY = "tinode/video"

	@classmethod
	def INPUT_TYPES(cls):
		return {
			"required": {
				"images": ("IMAGE", {"tooltip": "The model's result on the padded clip."}),
				"pad_count": ("INT", {"default": 0, "min": 0, "max": 100000, "step": 1,
					"tooltip": "Head frames to drop — from Frame Pad's pad_count."}),
			},
		}

	RETURN_TYPES = ("IMAGE",)
	RETURN_NAMES = ("images",)
	FUNCTION = "execute"

	def execute(self, images, pad_count=0):
		imgs = first(images)
		if imgs.dim() != 4:
			imgs = imgs.unsqueeze(0) if imgs.dim() == 3 else imgs
		p = int(first(pad_count, 0))
		N = imgs.shape[0]
		if p >= N:
			raise RuntimeError(
				f"Frame Unpad: pad_count {p} >= {N} frames — nothing would be left. "
				"Feed the same pad_count Frame Pad produced.")
		return (imgs[p:].contiguous() if p > 0 else imgs,)
