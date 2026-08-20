"""Frame Pad / Frame Unpad — pad a clip to a length the model returns unchanged.

The point of padding is that the model hands back EXACTLY the length it was
given: pad, run, drop the pad, and the clip is its original length again.

For VOID the rule comes straight from comfy_extras/nodes_void.py:

    TEMPORAL_COMPRESSION = 4 ; PATCH_SIZE_T = 2
    latent_t = ((length - 1) // 4) + 1      # rounded DOWN to an even value

CogVideoX-Fun-V1.5 uses patch_size_t=2, so latent_t must be even or the
transformer circular-pads a phantom latent frame. The decoder then emits
`latent_t * 4` pixel frames — NOT (latent_t-1)*4+1. So the output equals the
input only when `length == latent_t * 4` with latent_t even, i.e. when the
length is a **multiple of 8** (4 x 2).

That explains the observed failures exactly: 93 in gives latent_t 24 and 96 out
(+3); 97 rounds down to 93, also 96 out (-1). Neither 4n+1 nor "multiple of 32"
is the rule — 96 satisfied both by coincidence. `plus_one` remains for a model
whose length genuinely is ratio*n + 1.

  Frame Pad     images (L)            -> images (L+p), pad_count = p, total = L+p
                mask   (L)            -> mask   (L+p)
  [ your model on the padded clip ]
  Frame Unpad   images (L+p), p       -> images (L)

Pad the mask in lockstep so it stays aligned with the padded video; keep the
ORIGINAL (unpadded) mask for the final paste-back. Padding just repeats existing
frames — no interpolation, no quality loss. Frame Unpad's `expected_frames`
remains as a guard for a model that still disagrees.
"""

from __future__ import annotations

import torch

from ...base import TiNode, first
from ...registry import register


def frame_pad_count(length, min_prepend=8, ratio=8, plus_one=False):
	"""Frames to prepend so the padded total is a length the model returns unchanged.

	The point of padding is that the model hands back EXACTLY what it was given:
	pad to a length it accepts, run it, drop the pad, and the clip is its original
	length again. If the target is wrong the output length differs and every
	frame after it is misplaced.

	Two targets, because models differ:
	  plus_one False (default) -- total is a MULTIPLE of `ratio`. This is what
	      VOID does: it was measured returning 96 frames for BOTH 93 and 97 in,
	      i.e. it rounds to the nearest multiple (96 = 32*3), it does not honour
	      a ratio*n+1 length.
	  plus_one True -- total is ratio*n + 1, the shape a VAE's
	      (L-1)//ratio*ratio+1 formula describes.

	`p` is the smallest pad >= min_prepend that hits the target, so the head
	padding that reduces error is preserved.
	"""
	length = int(length)
	ratio = max(1, int(ratio))
	min_prepend = int(min_prepend)
	if length <= 0:
		return min_prepend
	if plus_one:
		return min_prepend + ((1 - length - min_prepend) % ratio)
	return min_prepend + ((-length - min_prepend) % ratio)


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
				"context_images": ("IMAGE", {"tooltip":
					"Frames to pad the HEAD with instead of repeating frame 0 — the "
					"TAIL of whatever comes immediately before this clip. The model "
					"then starts from real motion instead of a freeze, which is what "
					"keeps a chunk boundary from jumping.\n"
					"Give it the previous chunk's FINISHED result and leave "
					"context_mask empty; give it the ORIGINAL preceding frames and "
					"wire their mask into context_mask too."}),
				"min_prepend": ("INT", {"default": 8, "min": 0, "max": 64, "step": 1,
					"tooltip": "Minimum duplicated head frames (VOID: 8 reduces error)."}),
				"temporal_ratio": ("INT", {"default": 8, "min": 1, "max": 256, "step": 1,
					"tooltip": "Pad the total to a multiple of this. 8 for VOID = "
							   "temporal_compression(4) x patch_size_t(2): only a "
							   "multiple of 8 comes back the same length."}),
				"max_frames": ("INT", {"default": 197, "min": 1, "max": 100000, "step": 1,
					"tooltip": "Warn when images+pad exceeds what the model takes in "
							   "one pass (VOID's reference max is 197). A warning "
							   "only — the real limit here is VRAM."}),
				# APPENDED last on purpose: widget values are positional, so
				# inserting a widget mid-list silently shifts every value after it
				# in already-saved workflows.
				"plus_one": ("BOOLEAN", {"default": False, "tooltip":
					"Target ratio*n+1 instead of a plain multiple, for a model whose "
					"length really is (L-1)//ratio*ratio+1. Off for VOID."}),
				# APPENDED last: input order is what link slots index by, so a new
				# socket goes at the end or every saved link after it shifts.
				"context_mask": ("MASK", {"tooltip":
					"The mask belonging to context_images. Needed when the context is "
					"the ORIGINAL preceding frames, because the object is still in "
					"them — leaving the mask empty there would tell the model to KEEP "
					"it, and it can carry back into the clip. Leave empty when the "
					"context is already-finished frames."}),
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

	def execute(self, images, mask=None, min_prepend=8, temporal_ratio=8,
				max_frames=197, plus_one=False, context_images=None,
				context_mask=None):
		imgs = first(images)
		if imgs.dim() != 4:
			imgs = imgs.unsqueeze(0) if imgs.dim() == 3 else imgs
		L, H, W, C = imgs.shape
		p = frame_pad_count(L, int(first(min_prepend, 8)), int(first(temporal_ratio, 8)),
							bool(first(plus_one, False)))
		total = L + p
		cap = int(first(max_frames, 197))
		if total > cap:
			print(f"[tinode] Frame Pad: padded length {total} exceeds max_frames "
				  f"{cap} — chunk the clip upstream.")

		ctx = first(context_images)
		from_context = isinstance(ctx, torch.Tensor) and p > 0
		if from_context:
			if ctx.dim() != 4:
				ctx = ctx.unsqueeze(0) if ctx.dim() == 3 else ctx
			if ctx.shape[1:3] != imgs.shape[1:3]:
				raise RuntimeError(
					f"Frame Pad: context is {tuple(ctx.shape[1:3])} but the clip is "
					f"{tuple(imgs.shape[1:3])} — the previous chunk must be the SAME "
					"crop as this one.")
			tail = ctx[-p:]
			if tail.shape[0] < p:               # shorter than the pad: hold its first frame
				fill = tail[:1].repeat(p - tail.shape[0], 1, 1, 1)
				tail = torch.cat([fill, tail], dim=0)
			head = tail.to(imgs.dtype)
		else:
			head = imgs[:1].repeat(p, 1, 1, 1) if p > 0 else imgs[:0]
		padded = torch.cat([head, imgs], dim=0) if p > 0 else imgs

		m = first(mask)
		if isinstance(m, torch.Tensor):
			if m.dim() == 2:
				m = m.unsqueeze(0)
			# The mask and the video usually arrive by DIFFERENT routes (the mask
			# via a segment editor, the frames straight from the store), so their
			# lengths can silently disagree. Padding both then hands the model a
			# mask that is offset in time from the video it belongs to — the
			# removal drifts frame by frame. Say so instead.
			if m.shape[0] != L:
				raise RuntimeError(
					f"Frame Pad: the mask has {m.shape[0]} frame(s) but the video "
					f"has {L}. They must line up, or the mask is offset in time "
					"from the frames it describes. Check that the mask path "
					"(segment editors) and the image path came from the same clip.")
			if m.shape[1] != H or m.shape[2] != W:
				raise RuntimeError(
					f"Frame Pad: the mask is {m.shape[1]}x{m.shape[2]} but the video "
					f"is {H}x{W}.")
			if from_context:
				cm = first(context_mask)
				if isinstance(cm, torch.Tensor):
					# The context still contains the object, so carry its own mask:
					# zeroing it would ask the model to PRESERVE what we are removing.
					if cm.dim() == 2:
						cm = cm.unsqueeze(0)
					if cm.shape[1:3] != m.shape[1:3]:
						raise RuntimeError(
							f"Frame Pad: context_mask is {tuple(cm.shape[1:3])} but the "
							f"mask is {tuple(m.shape[1:3])} — same crop required.")
					tailm = cm[-p:]
					if tailm.shape[0] < p:
						tailm = torch.cat([tailm[:1].repeat(p - tailm.shape[0], 1, 1), tailm], 0)
					mhead = tailm.to(m.dtype)
				else:
					# No context mask: the context is already-finished frames, so
					# mask them OFF and the model just continues from them.
					mhead = torch.zeros((p, m.shape[1], m.shape[2]), dtype=m.dtype)
			else:
				mhead = m[:1].repeat(p, 1, 1) if p > 0 else m[:0]
			padded_mask = torch.cat([mhead, m], dim=0) if p > 0 else m
		else:
			padded_mask = torch.zeros((total, H, W), dtype=torch.float32)
		if from_context:
			how = ("with its own mask" if isinstance(first(context_mask), torch.Tensor)
				   else "mask zeroed there")
			print(f"[tinode] Frame Pad: head padded from {p} context frame(s) ({how}).")

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
			"optional": {
				"expected_frames": ("INT", {"default": 0, "min": 0, "max": 9999999, "step": 1,
					"tooltip": "Trim the result to exactly this many frames (0 = off). "
							   "A video VAE can return MORE frames than it was given "
							   "— it decodes ratio x latents — and those extras would "
							   "spill past the chunk into the next one. Wire the "
							   "clip's original frame_count here."}),
			},
		}

	RETURN_TYPES = ("IMAGE", "INT")
	RETURN_NAMES = ("images", "frame_count")
	FUNCTION = "execute"

	def execute(self, images, pad_count=0, expected_frames=0):
		imgs = first(images)
		if imgs.dim() != 4:
			imgs = imgs.unsqueeze(0) if imgs.dim() == 3 else imgs
		p = int(first(pad_count, 0))
		N = imgs.shape[0]
		if p >= N:
			raise RuntimeError(
				f"Frame Unpad: pad_count {p} >= {N} frames — nothing would be left. "
				"Feed the same pad_count Frame Pad produced.")
		out = imgs[p:].contiguous() if p > 0 else imgs

		want = int(first(expected_frames, 0) or 0)
		if want > 0 and out.shape[0] != want:
			if out.shape[0] < want:
				raise RuntimeError(
					f"Frame Unpad: got {out.shape[0]} frame(s) after unpadding but "
					f"expected {want} — the model returned FEWER frames than the "
					"clip had, so something upstream dropped frames.")
			# More frames than we started with: a video VAE decodes ratio x latents,
			# so it can hand back a few extra at the tail. Keeping them would push
			# this chunk past its own frame range and overwrite the next chunk.
			print(f"[tinode] Frame Unpad: model returned {out.shape[0]} frames, "
				  f"trimming to the original {want}.")
			out = out[:want].contiguous()
		return (out, int(out.shape[0]))
