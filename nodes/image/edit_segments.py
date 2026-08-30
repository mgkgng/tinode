"""Edit Segments — brushes that reshape a segment stream by hand.

The freehand sibling of Delete / Add Segments. One node, three brushes:

  sweep  drag over segments and they go WHOLE, per FRAME instance — sweeping a
         segment out on frame 12 leaves that object alone everywhere else
  draw   drag to paint a NEW segment in the shape you actually want, instead
         of a rectangle you have to settle for
  erase  drag to rub mask PIXELS away — from SAM3's segments just as much as
         from your own strokes. This is the one that lets you fix a mask that
         is nearly right instead of throwing the whole detection away.

Every gesture is one step on an undo stack (↶ / ↷, Ctrl-Z / Ctrl-Shift-Z), and
the brush size is adjustable, so this is meant to be used like a paint tool.

Strokes are stored as polylines ({mode, frame, radius, points}), not as images:
a few hundred bytes in the workflow instead of a PNG per frame, and they stay
editable. The backend stamps discs along each polyline to rasterize them, then
replays the strokes IN ORDER on top of the surviving input segments — so a draw
after an erase paints back over the hole, exactly as it looked in the editor.
Draw strokes are appended AFTER the input segments, so the (frame, index) pairs
the sweep brush records never shift under them.

An erase that empties a segment drops it; one that only bites a piece off has
its bbox re-tightened, so the boxes output never claims area that is gone.

(Node id stays TI_EraseSegments for backward compatibility; the display name is
Edit Segments.)
"""

from __future__ import annotations

import json
import math

import torch

from ...registry import register
from ...schema import validate_segments
from .delete_segments import DeleteSegments, parse_deleted_items
from .pick_segments import PickSegments, _img_signature, _seg_signature, color_for_id
from .segments_to_masks import bbox_mask

# Painted ids start above SAM3's and Add Segments' (1,000,000) so the three
# never collide in a chained graph.
_PAINT_ID_BASE = 2_000_000


def parse_strokes(raw, sig):
	"""Decode the painted-stroke widget, ignoring strokes from other input."""
	try:
		data = json.loads(raw) if raw else {}
	except (TypeError, ValueError):
		return []
	if not isinstance(data, dict):
		return []
	if data.get("sig") is not None and data.get("sig") != sig:
		return []
	strokes = data.get("strokes")
	return strokes if isinstance(strokes, list) else []


def rasterize_stroke(stroke, H, W):
	"""One polyline -> {bbox, mask}, by stamping discs along it.

	Stamping into a local window per point keeps this O(points x r^2) instead of
	testing every pixel of the stroke's bounding box against every point, which
	matters for a long drag at a large radius.
	"""
	pts = [p for p in stroke.get("pts", []) if isinstance(p, (list, tuple)) and len(p) == 2]
	if not pts:
		return None
	r = max(1, int(stroke.get("r", 8)))
	xs = [float(p[0]) for p in pts]
	ys = [float(p[1]) for p in pts]
	x0 = max(0, int(min(xs)) - r - 1)
	y0 = max(0, int(min(ys)) - r - 1)
	x1 = min(W, int(max(xs)) + r + 2)
	y1 = min(H, int(max(ys)) + r + 2)
	if x1 <= x0 or y1 <= y0:
		return None

	d = torch.arange(-r, r + 1, dtype=torch.float32)
	disc = ((d.view(-1, 1) ** 2 + d.view(1, -1) ** 2) <= r * r)      # [2r+1, 2r+1]
	m = torch.zeros((y1 - y0, x1 - x0), dtype=torch.bool)

	# Densify the polyline so a fast drag paints a line, not dots.
	dense = []
	for i, (px, py) in enumerate(zip(xs, ys)):
		dense.append((px, py))
		if i + 1 < len(xs):
			nx, ny = xs[i + 1], ys[i + 1]
			dist = math.hypot(nx - px, ny - py)
			steps = int(dist / max(1.0, r * 0.5))
			for s in range(1, steps):
				dense.append((px + (nx - px) * s / steps, py + (ny - py) * s / steps))

	for px, py in dense:
		cx, cy = int(round(px)) - x0, int(round(py)) - y0
		ax0, ay0 = cx - r, cy - r
		sx0, sy0 = max(0, ax0), max(0, ay0)
		sx1, sy1 = min(m.shape[1], ax0 + 2 * r + 1), min(m.shape[0], ay0 + 2 * r + 1)
		if sx1 <= sx0 or sy1 <= sy0:
			continue
		m[sy0:sy1, sx0:sx1] |= disc[sy0 - ay0:sy1 - ay0, sx0 - ax0:sx1 - ax0]

	if not bool(m.any()):
		return None
	return {"bbox": [x0, y0, x1, y1], "mask": m.to(torch.uint8)}


def stroke_mode(stroke):
	"""'erase' or 'draw'. Strokes saved before the eraser existed are draws."""
	return "erase" if str(stroke.get("mode", "draw")) == "erase" else "draw"


def painted_segments(strokes, frame_idx, H, W):
	"""Every DRAW stroke made on `frame_idx`, as segment dicts."""
	out = []
	for i, st in enumerate(strokes):
		try:
			if int(st.get("frame", -1)) != frame_idx:
				continue
		except (TypeError, ValueError):
			continue
		if stroke_mode(st) != "draw":
			continue
		shape = rasterize_stroke(st, H, W)
		if shape is None:
			continue
		out.append({
			"id": int(st.get("id", _PAINT_ID_BASE + i)),
			"bbox": shape["bbox"],
			"conf": 1.0,
			"mask": shape["mask"],
		})
	return out


def tighten(seg):
	"""Shrink a segment's bbox to its mask, or None once the mask is empty.

	Erasing a corner off an object must not leave the box covering the part you
	just removed: the boxes output (and every crop derived from it) is built from
	bbox alone, so a stale box would keep asking a model to repaint empty space.
	"""
	m = seg["mask"]
	b = m.to(torch.bool)
	rows = torch.any(b, dim=1).nonzero().flatten()
	if rows.numel() == 0:
		return None
	cols = torch.any(b, dim=0).nonzero().flatten()
	ry0, ry1 = int(rows[0]), int(rows[-1]) + 1
	cx0, cx1 = int(cols[0]), int(cols[-1]) + 1
	if (ry0, cx0, ry1, cx1) == (0, 0, m.shape[0], m.shape[1]):
		return seg
	x0, y0, _x1, _y1 = seg["bbox"]
	return {**seg, "mask": m[ry0:ry1, cx0:cx1].contiguous(),
			"bbox": [x0 + cx0, y0 + ry0, x0 + cx1, y0 + ry1]}


def erase_segment(seg, shape):
	"""`seg` minus the rasterized stroke `shape`, or None if nothing is left.

	Both carry a bbox in FRAME coordinates and a mask cropped to it, so the cut
	happens in their overlap window — no full-frame buffer is ever allocated.
	"""
	x0, y0, x1, y1 = seg["bbox"]
	ex0, ey0, ex1, ey1 = shape["bbox"]
	ix0, iy0 = max(x0, ex0), max(y0, ey0)
	ix1, iy1 = min(x1, ex1), min(y1, ey1)
	if ix1 <= ix0 or iy1 <= iy0:
		return seg
	cut = shape["mask"][iy0 - ey0:iy1 - ey0, ix0 - ex0:ix1 - ex0].to(torch.bool)
	if not bool(cut.any()):
		return seg
	m = seg["mask"].clone()
	win = m[iy0 - y0:iy1 - y0, ix0 - x0:ix1 - x0]
	win[cut] = 0
	return tighten({**seg, "mask": m})


def apply_strokes(segs, strokes, frame_idx, H, W):
	"""Replay this frame's strokes, in the order they were made, over `segs`.

	Order is what makes the brushes feel like a paint tool: an erase only eats
	what already exists, and a draw made afterwards paints back over the hole.
	"""
	out = list(segs)
	for i, st in enumerate(strokes):
		try:
			if int(st.get("frame", -1)) != frame_idx:
				continue
		except (TypeError, ValueError):
			continue
		shape = rasterize_stroke(st, H, W)
		if shape is None:
			continue
		if stroke_mode(st) == "erase":
			out = [s for s in (erase_segment(s, shape) for s in out) if s is not None]
		else:
			out.append({
				"id": int(st.get("id", _PAINT_ID_BASE + i)),
				"bbox": shape["bbox"],
				"conf": 1.0,
				"mask": shape["mask"],
			})
	return out


@register
class EditSegments(DeleteSegments):
	NODE_ID = "EraseSegments"          # pinned: saved graphs reference TI_EraseSegments
	DISPLAY_NAME = "Edit Segments (ti)"
	CATEGORY = "tinode/image"
	OUTPUT_NODE = True

	@classmethod
	def INPUT_TYPES(cls):
		return {
			"required": {
				"image": ("IMAGE",),
				"segments": ("TI_SAM3_SEGMENTS",),
			},
			"optional": {
				"deleted_items": ("STRING", {"default": "{}"}),
				"current_frame": ("INT", {"default": 0, "min": 0, "max": 999999}),
				"overlay_alpha": ("FLOAT", {
					"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.05,
					"tooltip": "Opacity of the segments drawn on the image output.",
				}),
				# {"sig":..., "strokes":[{"id","mode","frame","r","pts":[[x,y],...]}]}
				# mode: "draw" (paint a new segment) or "erase" (rub pixels away
				# from whatever is already there). Absent = "draw", so strokes
				# saved before the eraser existed still mean what they meant.
				"painted": ("STRING", {"default": "{}"}),
				"brush_size": ("INT", {"default": 16, "min": 1, "max": 512, "step": 1,
					"tooltip": "Brush radius in source pixels — the editor's slider "
							   "writes here, so it is saved with the workflow."}),
			},
		}

	def execute(self, image, segments, deleted_items="{}", current_frame=0,
				overlay_alpha=0.5, painted="{}", brush_size=16):
		validate_segments(segments)
		if not isinstance(image, torch.Tensor) or image.dim() not in (3, 4):
			raise ValueError("Edit Segments: image must be [H,W,C] or [N,H,W,C].")
		imgs = image if image.dim() == 4 else image.unsqueeze(0)
		H, W = int(segments["height"]), int(segments["width"])
		frames = segments.get("frames", [])
		N = int(segments["num_frames"])
		if imgs.shape[0] == 1 and N > 1:
			imgs = imgs.expand(N, -1, -1, -1)
		elif imgs.shape[0] != N:
			raise ValueError(
				f"Edit Segments: image has {imgs.shape[0]} frame(s) but segments "
				f"has {N}. Connect the matching source image/video.")
		if tuple(imgs.shape[1:3]) != (H, W):
			raise ValueError(
				f"Edit Segments: image is {tuple(imgs.shape[1:3])} but the segments "
				f"are {(H, W)}.")

		# The signature covers the INPUT only, so painting never invalidates the
		# erase record (or vice versa) — they are independent edits of one clip.
		sig = f"{_seg_signature(segments)}_{_img_signature(imgs)}"
		deleted = parse_deleted_items(deleted_items, sig)
		strokes = parse_strokes(painted, sig)

		kept_frames = []
		for f in range(N):
			frame = frames[f] if f < len(frames) else []
			kept = [s for i, s in enumerate(frame) if (f, i) not in deleted]
			kept_frames.append(apply_strokes(kept, strokes, f, H, W))

		mask_out = torch.zeros((N, H, W), dtype=torch.float32)
		image_out = torch.zeros((N, H, W, 3), dtype=torch.float32)
		for f, frame in enumerate(kept_frames):
			canvas = imgs[f, ..., :3].detach().to(device="cpu", dtype=torch.float32).clone()
			for s in frame:
				x0, y0, x1, y1 = s["bbox"]
				m = s["mask"].detach().to(device="cpu", dtype=torch.float32)
				dst = mask_out[f, y0:y1, x0:x1]
				mask_out[f, y0:y1, x0:x1] = torch.maximum(dst, m)
				color = torch.tensor(color_for_id(int(s["id"])), dtype=torch.float32)
				a = (m * float(overlay_alpha)).unsqueeze(-1)
				region = canvas[y0:y1, x0:x1, :]
				canvas[y0:y1, x0:x1, :] = region * (1 - a) + color * a
			image_out[f] = canvas

		filtered = {
			"num_frames": N, "height": H, "width": W, "frames": kept_frames,
			"ids": sorted({int(s["id"]) for fr in kept_frames for s in fr}),
		}
		result = (mask_out, image_out, filtered, bbox_mask(filtered, H, W))
		manifest = PickSegments._build_assets(self, imgs, segments, sig)
		if manifest is None:
			return result
		return {"ui": {"ti_erase": [manifest]}, "result": result}
