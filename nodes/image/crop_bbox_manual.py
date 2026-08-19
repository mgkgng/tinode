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

import json
import os
import random

import torch

from ...base import TiNode, first as _first
from ...registry import register

# Longest side of the preview handed to the browser. The editor maps handles
# using the TRUE dimensions (sent as src_dims), so downscaling the preview only
# affects display sharpness, never the crop coordinates.
_PREVIEW_MAX_SIDE = 768


def _round_down(value: int, multiple: int) -> int:
	"""Largest multiple of `multiple` <= value.

	Never rounds UP past `value`: the caller has already clamped the box to the
	frame, so returning a bigger multiple would push the crop off the edge and
	slice an empty tensor. When the span is smaller than one multiple there is no
	multiple to give, so keep the span as-is rather than emitting nothing.
	"""
	if multiple <= 1:
		return value
	down = (value // multiple) * multiple
	return down if down > 0 else value


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
		# The origin is clamped to the LAST pixel, not past it: a box saved
		# against a taller frame (or simply dragged off the edge) would otherwise
		# start at y == H and slice an empty crop.
		x0 = max(0, min(int(x), max(0, W - 1)))
		y0 = max(0, min(int(y), max(0, H - 1)))
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


def resolve_box(box, W, H, divisible_by):
	"""One {x,y,w,h} dict -> a clamped, divisible-rounded (x0,y0,x1,y1) in the frame.

	Same rules as the single Bbox Crop: origin clamped to the last pixel, w/h of 0
	means "to the far edge", the size is rounded DOWN to a multiple of
	divisible_by, and at least a 1px box is guaranteed.
	"""
	x = int(box.get("x", 0)); y = int(box.get("y", 0))
	w = int(box.get("w", 0)); h = int(box.get("h", 0))
	x0 = max(0, min(x, max(0, W - 1)))
	y0 = max(0, min(y, max(0, H - 1)))
	x1 = W if w <= 0 else x0 + w
	y1 = H if h <= 0 else y0 + h
	x1 = max(x0 + 1, min(x1, W))
	y1 = max(y0 + 1, min(y1, H))
	cw = _round_down(x1 - x0, int(divisible_by))
	ch = _round_down(y1 - y0, int(divisible_by))
	return x0, y0, x0 + cw, y0 + ch


def parse_boxes(raw):
	"""Decode the editor's serialized box list; [] on anything malformed."""
	try:
		data = json.loads(raw) if raw else []
	except (ValueError, TypeError):
		return []
	if not isinstance(data, list):
		return []
	out = []
	for b in data:
		if isinstance(b, dict) and all(k in b for k in ("x", "y", "w", "h")):
			out.append(b)
	return out


@register
class BboxCropMulti(TiNode):
	DISPLAY_NAME = "Bbox Crop · Multi (ti)"
	CATEGORY = "tinode/image"
	OUTPUT_NODE = True

	@classmethod
	def INPUT_TYPES(cls):
		return {
			"required": {
				"image": ("IMAGE",),
			},
			"optional": {
				# JSON list of {x,y,w,h}, driven by the multi-box editor and saved
				# with the workflow. Empty = one crop of the whole frame.
				"boxes": ("STRING", {"default": "[]", "multiline": False}),
				"divisible_by": ("INT", {"default": 1, "min": 0, "max": 256, "step": 1,
					"tooltip": "Round each crop size down to a multiple (8 for most "
							   "latent models). 0 or 1 = off."}),
			},
		}

	# Two ways out; pick one per graph.
	#
	# crops / crop_info / crop_index are a ComfyUI LIST — one entry per box — so
	# whatever is wired to them runs once per crop automatically (list
	# expansion). No loop node needed, but there is no iteration to hook into:
	# nothing accumulates across crops and a loop's controls don't apply.
	#
	# item_list is a single Inspire ITEM_LIST value — one item per box — for
	# ▶Foreach List. That gives a real sequential loop over the crops, nestable
	# inside the outer per-clip Foreach, with an intermediate_output to
	# accumulate through and a Validation Gate that stops on each crop in turn.
	# Unpack the loop's `item` with Bbox Crop Item.
	RETURN_TYPES = ("IMAGE", "TI_CROP_XFORM", "INT", "ITEM_LIST", "INT")
	RETURN_NAMES = ("crops", "crop_info", "crop_index", "item_list", "count")
	# Only the first three expand; item_list and count are plain single values.
	OUTPUT_IS_LIST = (True, True, True, False, False)
	OUTPUT_TOOLTIPS = (
		"One IMAGE per box (ComfyUI list — downstream runs once per crop).",
		"One crop_info per box, index-aligned with `crops`.",
		"0-based index of each crop, for keying its saved artifacts.",
		"One item per box for ▶Foreach List — unpack it with Bbox Crop Item.",
		"How many boxes were drawn.",
	)
	FUNCTION = "execute"

	def execute(self, image, boxes="[]", divisible_by=1):
		imgs = image if image.dim() == 4 else image.unsqueeze(0)
		N, H, W, C = imgs.shape

		box_list = parse_boxes(boxes)
		if not box_list:
			box_list = [{"x": 0, "y": 0, "w": 0, "h": 0}]   # whole frame = one crop

		crops, infos, indices, items = [], [], [], []
		for i, b in enumerate(box_list):
			x0, y0, x1, y1 = resolve_box(b, W, H, divisible_by)
			crop = imgs[:, y0:y1, x0:x1, :].contiguous()      # [N,ch,cw,C]
			item = {"y0": y0, "x0": x0, "h": y1 - y0, "w": x1 - x0,
					"oy": 0, "ox": 0, "nh": y1 - y0, "nw": x1 - x0}
			info = {"H": H, "W": W, "C": C, "items": [item] * N}
			crops.append(crop)
			infos.append(info)
			indices.append(i)
			# The same three values, packed one-per-box for the Foreach path. The
			# crop tensor is the SAME object as in `crops` — carrying it costs
			# nothing beyond the list that was built anyway.
			items.append({"crop_index": i, "crop_count": len(box_list),
						  "image": crop, "crop_info": info,
						  "box": {"x": x0, "y": y0, "w": x1 - x0, "h": y1 - y0}})

		# A distinct list object: ForeachListBegin slices its ITEM_LIST down as it
		# iterates, so it must not be an alias of a list we also return elsewhere.
		result = (crops, infos, indices, list(items), len(items))
		preview = _save_preview(imgs[0])
		if preview is None:
			return result
		info, src_w, src_h = preview
		return {"ui": {"ti_preview": [info], "src_dims": [src_w, src_h],
					   "box_count": [len(crops)]}, "result": result}


@register
class BboxCropItem(TiNode):
	"""Unpack one ▶Foreach List `item` from Bbox Crop · Multi's item_list.

	The loop hands back whatever was in the list; this turns that opaque item
	into the same three values the list-expansion path gives you — the crop
	pixels, its crop_info, and its index — so the per-crop graph inside the loop
	is wired exactly like the one outside it.
	"""

	DISPLAY_NAME = "Bbox Crop Item (ti)"
	CATEGORY = "tinode/image"

	@classmethod
	def INPUT_TYPES(cls):
		return {"required": {"item": ("TI_BBOX_ITEM", {"tooltip":
			"One crop from Bbox Crop · Multi's item_list, via ForeachListBegin's "
			"`item` output."})}}

	RETURN_TYPES = ("IMAGE", "TI_CROP_XFORM", "INT", "INT")
	RETURN_NAMES = ("crop", "crop_info", "crop_index", "crop_count")
	OUTPUT_TOOLTIPS = (
		"This crop's frames, bit-exact from the source.",
		"Maps this crop back to the full frame.",
		"0-based index of this crop, for keying its saved artifacts.",
		"How many crops the loop will run in total.",
	)
	FUNCTION = "execute"

	def execute(self, item):
		item = _first(item)
		if not isinstance(item, dict) or "crop_info" not in item:
			raise RuntimeError(
				"Bbox Crop Item: `item` must be one item of Bbox Crop · Multi's "
				"item_list, taken from ForeachListBegin's `item` output.")
		return (item["image"], item["crop_info"],
				int(item.get("crop_index", 0)), int(item.get("crop_count", 1)))
