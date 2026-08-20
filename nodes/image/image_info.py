"""Image Info — print an IMAGE batch's width, height and frame count when it runs.

A quick probe for wiring up a graph: drop it on any IMAGE and each run prints
`WxH, N frames` to the console (and shows it on the node), plus emits the three
numbers as outputs so they can drive other nodes.
"""

from __future__ import annotations

import torch

from ...base import TiNode, first
from ...registry import register


@register
class ImageInfo(TiNode):
	DISPLAY_NAME = "Image Info (ti)"
	CATEGORY = "tinode/util"
	OUTPUT_NODE = True

	@classmethod
	def INPUT_TYPES(cls):
		return {"required": {"images": ("IMAGE", {"tooltip":
			"The batch to measure. Frame count = the batch dimension."})}}

	RETURN_TYPES = ("INT", "INT", "INT")
	RETURN_NAMES = ("width", "height", "frame_count")
	OUTPUT_TOOLTIPS = ("Width in pixels.", "Height in pixels.", "Number of frames (batch size).")
	FUNCTION = "execute"

	def execute(self, images):
		imgs = first(images)
		if not isinstance(imgs, torch.Tensor):
			raise RuntimeError("Image Info: expected an IMAGE tensor.")
		if imgs.dim() == 3:
			imgs = imgs.unsqueeze(0)
		if imgs.dim() != 4:
			raise RuntimeError(f"Image Info: expected [N,H,W,C], got shape {tuple(imgs.shape)}.")
		n, h, w, c = imgs.shape
		n, h, w, c = int(n), int(h), int(w), int(c)
		text = f"{w}x{h}, {n} frame(s), {c} channel(s)"
		print(f"[tinode] Image Info: {text}")
		return {"ui": {"text": [text]}, "result": (w, h, n)}
