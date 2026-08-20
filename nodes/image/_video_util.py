"""Small IMAGE-batch helpers shared by the video nodes (insert / trim / …).

Previously these lived in extend_video.py, which merged into frame_pad.py; the
helpers moved here so the other nodes keep a stable import.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


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
			other = other[..., :C]
		else:
			pad = torch.ones(other.shape[0], H, W, C - other.shape[3],
							 dtype=other.dtype, device=other.device)
			other = torch.cat([other, pad], dim=3)
	return other
