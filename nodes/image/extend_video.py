"""Extend Video — add frames to the start and/or end of a video.

A video in ComfyUI is just an [N,H,W,C] IMAGE batch, so extending one is a
concatenation. Each end is independent and can be fed from either source:

  none          leave that end alone
  first_frame   hold (freeze) the base clip's FIRST frame for N frames
  last_frame    hold the base clip's LAST frame for N frames
  video         splice in the clip wired to prepend_video / append_video

Holding a frame is the usual way to pad a clip out to a length a video model
demands, or to give it a still lead-in/lead-out. Splicing lets you bolt a real
clip on either side.

Spliced clips are matched to the base automatically — resized to the base's
resolution and reconciled to its channel count — so mismatched inputs join
instead of erroring. The base clip's own pixels are never resampled.

Also returns how many frames went on each end, so a later Batch Drop/Pick can
trim exactly what was added back off.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from ...base import TiNode
from ...registry import register

_MODES = ["none", "first_frame", "last_frame", "video"]


def _as_batch(img):
	"""[H,W,C] or [N,H,W,C] -> [N,H,W,C]."""
	return img if img.dim() == 4 else img.unsqueeze(0)


def _match(other, H, W, C, ref):
	"""Conform `other` [M,h,w,c] to the base's resolution, channels, dtype/device."""
	other = other.to(device=ref.device, dtype=ref.dtype)
	if other.shape[1] != H or other.shape[2] != W:
		x = other.permute(0, 3, 1, 2)
		x = F.interpolate(x, size=(H, W), mode="bilinear", align_corners=False)
		other = x.permute(0, 2, 3, 1).contiguous()
	if other.shape[3] != C:
		if other.shape[3] > C:
			other = other[..., :C]                       # drop alpha
		else:                                            # pad opaque alpha
			pad = torch.ones(
				other.shape[0], H, W, C - other.shape[3],
				dtype=other.dtype, device=other.device,
			)
			other = torch.cat([other, pad], dim=3)
	return other


def _hold(frame, count):
	"""Repeat a single [H,W,C] frame `count` times -> [count,H,W,C]."""
	return frame.unsqueeze(0).repeat(count, 1, 1, 1)


@register
class ExtendVideo(TiNode):
	DISPLAY_NAME = "Extend Video · Prepend/Append (ti)"
	CATEGORY = "tinode/image"

	@classmethod
	def INPUT_TYPES(cls):
		return {
			"required": {
				"image": ("IMAGE",),
				"prepend_mode": (_MODES, {"default": "none",
					"tooltip": "What to add BEFORE the clip. first_frame/last_frame "
							   "hold that frame for prepend_frames frames; video "
							   "splices in the prepend_video input."}),
				"prepend_frames": ("INT", {"default": 0, "min": 0, "max": 10000, "step": 1,
					"tooltip": "How many held frames to add. Only used by "
							   "first_frame / last_frame."}),
				"append_mode": (_MODES, {"default": "none",
					"tooltip": "What to add AFTER the clip."}),
				"append_frames": ("INT", {"default": 0, "min": 0, "max": 10000, "step": 1,
					"tooltip": "How many held frames to add. Only used by "
							   "first_frame / last_frame."}),
			},
			"optional": {
				"prepend_video": ("IMAGE",),
				"append_video": ("IMAGE",),
			},
		}

	RETURN_TYPES = ("IMAGE", "INT", "INT")
	RETURN_NAMES = ("images", "prepended", "appended")
	FUNCTION = "execute"

	def _side(self, mode, count, clip, base, H, W, C, label):
		"""Build the frames for one end, or None when that end adds nothing."""
		if mode == "none":
			return None
		if mode == "video":
			if clip is None:
				print(f"[tinode] Extend Video: {label}_mode is 'video' but no "
					  f"{label}_video is connected — skipping.")
				return None
			return _match(_as_batch(clip), H, W, C, base)
		if count <= 0:
			return None
		frame = base[0] if mode == "first_frame" else base[-1]
		return _hold(frame, count)

	def execute(self, image, prepend_mode="none", prepend_frames=0,
				append_mode="none", append_frames=0,
				prepend_video=None, append_video=None):
		base = _as_batch(image)                       # [N,H,W,C]
		N, H, W, C = base.shape

		head = self._side(prepend_mode, int(prepend_frames), prepend_video,
						  base, H, W, C, "prepend")
		tail = self._side(append_mode, int(append_frames), append_video,
						  base, H, W, C, "append")

		parts = [p for p in (head, base, tail) if p is not None]
		out = torch.cat(parts, dim=0) if len(parts) > 1 else base
		return (out, 0 if head is None else head.shape[0],
				0 if tail is None else tail.shape[0])
