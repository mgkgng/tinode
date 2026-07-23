"""Bbox Crop · Manual — cut a user-drawn rectangle out of an IMAGE.

The box is driven from an interactive editor in the node body (web/bbox_crop.js):
run the graph once to load a preview of the incoming frame, then drag the four
corner handles to place the crop. The handles write straight into the x / y /
width / height widgets (and those numbers update live as you drag), so the box
is also editable by hand and saved with the workflow.

One box is applied to every frame in the batch, which makes the node
format-preserving by construction: a single image -> a single image, an N-frame
video -> an N-frame video. ComfyUI stores both as one [N,H,W,C] IMAGE tensor, so
"keep the input format" is just letting the batch dimension pass through.

Coordinates are pixels in the source frame. Conveniences:
  * width or height of 0 means "extend to the far edge" (x..W, y..H).
  * the box is clamped to the frame, so out-of-range values crop the overlap
    instead of erroring.
  * divisible_by rounds the final crop size down to a multiple (VAE/UNet want
    /8); set 1 to disable.

The emitted crop_info is a plain TI_CROP_XFORM (oy/ox = 0, nh/nw = h/w since
there is no rescale), so processed crops feed straight back through Mask Crop
Paste Back to composite into the original frame.

The input-frame preview is saved to ComfyUI's temp dir and handed to the browser
via the standard {"ui": {"images": [...]}} channel. That work is best-effort and
lazily imported, so the crop still runs on a headless host with no PIL / no
folder_paths — you just don't get the visual editor there.
"""

from __future__ import annotations

import os
import random

import torch

from ...base import TiNode
from ...registry import register

# Longest side of the preview handed to the browser. The editor maps handles
# using the TRUE dimensions (sent as src_dims), so downscaling the preview only
# affects display sharpness, never the crop coordinates.
_PREVIEW_MAX_SIDE = 768


def _round_down(value: int, multiple: int) -> int:
	"""Largest multiple of `multiple` <= value, but never below one multiple."""
	if multiple <= 1:
		return value
	return max(multiple, (value // multiple) * multiple)


def _save_preview(frame: torch.Tensor):
	"""Save the first frame to the temp dir for the JS editor.

	Returns ({"filename","subfolder","type"}, W, H) or None on any failure —
	the caller treats a None as "no editor available" and still returns the crop.
	"""
	try:
		import numpy as np  # noqa: PLC0415 — lazy so headless hosts still load
		from PIL import Image  # noqa: PLC0415
		import folder_paths  # noqa: PLC0415 — only exists inside ComfyUI

		H, W = int(frame.shape[0]), int(frame.shape[1])
		arr = frame.detach().clamp(0, 1).cpu().numpy()
		if arr.ndim == 2:                      # [H,W] -> [H,W,1]
			arr = arr[..., None]
		if arr.shape[2] == 1:                  # grayscale -> RGB
			arr = np.repeat(arr, 3, axis=2)
		arr = (arr[..., :3] * 255.0).astype(np.uint8)

		img = Image.fromarray(arr)
		scale = min(1.0, _PREVIEW_MAX_SIDE / max(H, W))
		if scale < 1.0:
			img = img.resize((max(1, round(W * scale)), max(1, round(H * scale))),
							 Image.BILINEAR)

		out_dir = folder_paths.get_temp_directory()
		os.makedirs(out_dir, exist_ok=True)
		suffix = "".join(random.choice("abcdefghijklmnopqrstuvwxyz") for _ in range(8))
		fname = f"ti_bbox_{suffix}.png"
		img.save(os.path.join(out_dir, fname), compress_level=1)
		return {"filename": fname, "subfolder": "", "type": "temp"}, W, H
	except Exception as exc:  # noqa: BLE001 — preview is best-effort
		print(f"[tinode] Bbox Crop preview unavailable: {exc!r}")
		return None


@register
class BboxCropManual(TiNode):
	DISPLAY_NAME = "Bbox Crop · Manual (ti)"
	CATEGORY = "tinode/image"

	# Run every queue even when the crop output is unconnected, so the editor
	# always has a fresh preview of the input to draw on.
	OUTPUT_NODE = True

	@classmethod
	def INPUT_TYPES(cls):
		return {
			"required": {
				"image": ("IMAGE",),
				"x": ("INT", {"default": 0, "min": 0, "max": 8192, "step": 1,
					"tooltip": "Left edge of the crop, in source pixels. "
							   "Set by the editor's handles; also editable here."}),
				"y": ("INT", {"default": 0, "min": 0, "max": 8192, "step": 1,
					"tooltip": "Top edge of the crop, in source pixels."}),
				"width": ("INT", {"default": 0, "min": 0, "max": 8192, "step": 1,
					"tooltip": "Crop width in pixels. 0 = extend to the right edge."}),
				"height": ("INT", {"default": 0, "min": 0, "max": 8192, "step": 1,
					"tooltip": "Crop height in pixels. 0 = extend to the bottom edge."}),
			},
			"optional": {
				"divisible_by": ("INT", {"default": 1, "min": 0, "max": 256, "step": 1,
					"tooltip": "Round the crop size down to a multiple of this "
							   "(8 for most latent models). 0 or 1 = off."}),
			},
		}

	RETURN_TYPES = ("IMAGE", "TI_CROP_XFORM")
	RETURN_NAMES = ("images", "crop_info")
	FUNCTION = "execute"

	def execute(self, image, x, y, width, height, divisible_by=1):
		imgs = image if image.dim() == 4 else image.unsqueeze(0)  # [N,H,W,C]
		N, H, W, C = imgs.shape

		# Resolve the box in pixels. width/height of 0 -> run to the far edge.
		x0 = max(0, min(int(x), W))
		y0 = max(0, min(int(y), H))
		x1 = W if int(width) <= 0 else int(x0 + int(width))
		y1 = H if int(height) <= 0 else int(y0 + int(height))

		# Clamp to the frame and guarantee at least a 1px box.
		x1 = max(x0 + 1, min(x1, W))
		y1 = max(y0 + 1, min(y1, H))

		cw = _round_down(x1 - x0, int(divisible_by))
		ch = _round_down(y1 - y0, int(divisible_by))
		x1, y1 = x0 + cw, y0 + ch

		crop = imgs[:, y0:y1, x0:x1, :].contiguous()  # [N,ch,cw,C] — every frame, same box

		# No rescale, so the placed region fills the crop: oy/ox = 0, nh/nw = h/w.
		# One item per frame keeps index-alignment with the IMAGE batch for Paste Back.
		item = {"y0": y0, "x0": x0, "h": ch, "w": cw, "oy": 0, "ox": 0, "nh": ch, "nw": cw}
		crop_info = {"H": H, "W": W, "C": C, "items": [item] * N}
		result = (crop, crop_info)

		preview = _save_preview(imgs[0])
		if preview is None:
			return result
		info, src_w, src_h = preview
		# Custom ui key (NOT "images"): "images" would make ComfyUI attach its own
		# built-in preview widget on top of our editor and blow up the node height.
		# onExecuted still fires for any ui dict, so the editor gets ti_preview.
		return {"ui": {"ti_preview": [info], "src_dims": [src_w, src_h]}, "result": result}
