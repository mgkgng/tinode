"""Track Motion Editor · Circle — draw a path, get a circle that follows it.

Brush a stroke across the editor and the node hands back an animated MASK: one
frame per `max_frames`, each holding a filled circle at the position the stroke
was at that moment.

The stroke is recorded WITH ITS TIMING. Every point carries the millisecond it
was drawn at, so pausing mid-drag makes the circle linger and a fast flick makes
it dart — the animation is the gesture, played back. `timing = even` throws the
timing away and walks the path at constant speed instead, for when you wanted a
steady sweep and your hand disagreed.

Points are stored NORMALISED (0..1 of the canvas), so changing width/height
rescales the path instead of stranding it off-frame. The radius is in output
PIXELS, so the circle stays a circle whatever the aspect ratio.

Nothing here needs the source footage: the editor is self-contained, the widgets
are the whole input, and the node is pure. Wire the mask wherever a moving
region is wanted — an inpaint, a light wipe, a reveal.
"""

from __future__ import annotations

import json
import math

import torch

from ...base import TiNode, first
from ...registry import register

# name -> width/height. "custom" leaves the two widgets alone.
ASPECTS = ["custom", "1:1", "16:9", "9:16", "4:3", "3:4", "3:2", "2:3", "21:9", "2:1"]
TIMINGS = ["recorded", "even"]


def parse_track(raw):
	"""Decode the editor's hidden `track` widget into [(x, y, t_ms), ...].

	Anything malformed decodes to an empty path rather than raising: a bad widget
	value should give you a blank mask to fix, not a red node in the middle of a
	graph that was otherwise fine.
	"""
	try:
		data = json.loads(raw) if raw else {}
	except (TypeError, ValueError):
		return []
	if not isinstance(data, dict):
		return []
	pts = data.get("pts")
	if not isinstance(pts, list):
		return []
	out = []
	for p in pts:
		if not isinstance(p, (list, tuple)) or len(p) < 2:
			continue
		try:
			x, y = float(p[0]), float(p[1])
			t = float(p[2]) if len(p) > 2 else float(len(out))
		except (TypeError, ValueError):
			continue
		if math.isfinite(x) and math.isfinite(y) and math.isfinite(t):
			out.append((x, y, t))
	return out


def _lerp_at(pts, key, value):
	"""Interpolate (x, y) where the monotonic `key` series reaches `value`."""
	lo, hi = 0, len(pts) - 1
	while lo < hi:                       # first index whose key >= value
		mid = (lo + hi) // 2
		if key[mid] < value:
			lo = mid + 1
		else:
			hi = mid
	if lo == 0:
		return pts[0][0], pts[0][1]
	a, b = pts[lo - 1], pts[lo]
	span = key[lo] - key[lo - 1]
	f = 0.0 if span <= 0 else (value - key[lo - 1]) / span
	return a[0] + (b[0] - a[0]) * f, a[1] + (b[1] - a[1]) * f


def sample_positions(pts, count, timing="recorded"):
	"""`count` normalised (x, y) positions along the stroke.

	`recorded` walks the stroke in equal slices of TIME, so the drag's own pace
	survives. `even` walks it in equal slices of DISTANCE, so the circle keeps a
	constant speed. A stroke drawn faster than the sampler can see — every point
	stamped the same millisecond — falls back to even rather than piling every
	frame onto the last point.
	"""
	count = int(count)
	if count <= 0 or not pts:
		return []
	if len(pts) == 1 or count == 1:
		return [(pts[0][0], pts[0][1])] * count

	if timing == "recorded":
		key = [p[2] for p in pts]
		for i in range(1, len(key)):      # a stalled clock must not go backwards
			if key[i] < key[i - 1]:
				key[i] = key[i - 1]
		if key[-1] - key[0] <= 0:
			timing = "even"

	if timing != "recorded":
		key = [0.0]
		for i in range(1, len(pts)):
			key.append(key[-1] + math.hypot(pts[i][0] - pts[i - 1][0],
											pts[i][1] - pts[i - 1][1]))
		if key[-1] <= 0:                  # a stroke that never moved
			return [(pts[0][0], pts[0][1])] * count

	span = key[-1] - key[0]
	return [_lerp_at(pts, key, key[0] + span * (f / (count - 1)))
			for f in range(count)]


def ramp_radii(count, start, end):
	"""One radius per frame, `start` -> `end` linearly. end <= 0 means "hold"."""
	count = max(0, int(count))
	s = max(0.5, float(start))
	e = float(end)
	if e <= 0 or count <= 1:
		return [s] * count
	e = max(0.5, e)
	return [s + (e - s) * (f / (count - 1)) for f in range(count)]


def render_circles(positions, width, height, radius, feather=0):
	"""[N, H, W] mask: a filled circle at each normalised position.

	`radius` is one value for the whole run, or one per frame (from ramp_radii)
	when the circle is meant to grow or shrink as it travels.

	Only the circle's own bounding window is touched per frame, so cost follows
	the radius rather than the canvas — a 60px dot on an 8K canvas is as cheap as
	on a thumbnail. The edge always carries at least a 1px ramp, because a
	stair-stepped mask shows up as a stair-stepped edge in whatever it drives.
	"""
	W, H = max(1, int(width)), max(1, int(height))
	N = len(positions)
	out = torch.zeros((max(1, N), H, W), dtype=torch.float32)
	if N == 0:
		return out
	radii = radius if isinstance(radius, (list, tuple)) else [radius] * N
	soft = max(1.0, float(feather))

	for i, (nx, ny) in enumerate(positions):
		r = max(0.5, float(radii[i if i < len(radii) else -1]))
		reach = int(math.ceil(r + soft)) + 1
		cx, cy = float(nx) * W, float(ny) * H
		x0, x1 = max(0, int(cx) - reach), min(W, int(cx) + reach + 1)
		y0, y1 = max(0, int(cy) - reach), min(H, int(cy) + reach + 1)
		if x1 <= x0 or y1 <= y0:
			continue                       # the circle is entirely off-canvas
		xs = torch.arange(x0, x1, dtype=torch.float32) + 0.5 - cx
		ys = torch.arange(y0, y1, dtype=torch.float32) + 0.5 - cy
		dist = torch.sqrt(ys.view(-1, 1) ** 2 + xs.view(1, -1) ** 2)
		out[i, y0:y1, x0:x1] = ((r + 0.5 - dist) / soft).clamp(0.0, 1.0)
	return out


@register
class TrackMotionCircle(TiNode):
	DISPLAY_NAME = "Track Motion Editor · Circle (ti)"
	CATEGORY = "tinode/mask_motion"

	@classmethod
	def INPUT_TYPES(cls):
		return {
			"required": {
				"width": ("INT", {"default": 1024, "min": 16, "max": 8192, "step": 8,
					"tooltip": "Output mask width. The editor canvas follows it."}),
				"height": ("INT", {"default": 576, "min": 16, "max": 8192, "step": 8,
					"tooltip": "Output mask height. Driven by `aspect` unless that "
							   "is `custom`."}),
				"max_frames": ("INT", {"default": 81, "min": 1, "max": 4096, "step": 1,
					"tooltip": "How many frames the stroke is spread over. The whole "
							   "stroke always fits: more frames = finer motion, not "
							   "a longer path."}),
				"radius": ("INT", {"default": 64, "min": 1, "max": 4096, "step": 1,
					"tooltip": "Circle radius in OUTPUT pixels — a circle stays "
							   "round whatever the aspect ratio."}),
			},
			"optional": {
				"aspect": (ASPECTS, {"default": "16:9", "tooltip":
					"Sets height from width so the editor matches your footage. "
					"`custom` leaves both alone."}),
				"timing": (TIMINGS, {"default": "recorded", "tooltip":
					"recorded: play the stroke back at the speed you drew it — "
					"pause mid-drag and the circle waits there.\n"
					"even: ignore the timing and travel the path at a constant "
					"speed."}),
				"feather": ("INT", {"default": 0, "min": 0, "max": 512, "step": 1,
					"tooltip": "Soft edge, in pixels. 0 still antialiases by 1px."}),
				# Written by the editor: {"pts": [[x, y, t_ms], ...]} normalised.
				"track": ("STRING", {"default": "{}", "tooltip":
					"The recorded path. The editor writes this — you draw instead."}),
				# APPENDED below `track`: widget values are positional in saved
				# graphs, so anything new has to go on the end.
				"radius_end": ("INT", {"default": 0, "min": 0, "max": 4096, "step": 1,
					"tooltip": "Radius at the LAST frame — the circle grows or "
							   "shrinks as it travels. 0 = hold `radius` throughout."}),
				"context_width": ("INT", {"default": 0, "min": 0, "max": 4096, "step": 1,
					"tooltip": "Width of the `context` ring around the circle, in "
							   "pixels. That output is the band where the "
							   "surroundings get generated so they meet the object "
							   "instead of stopping at its edge. 0 = no ring."}),
			},
		}

	# context / outside are APPENDED so existing links keep their slots.
	RETURN_TYPES = ("MASK", "IMAGE", "INT", "MASK", "MASK")
	RETURN_NAMES = ("mask", "image", "frame_count", "context", "outside")
	OUTPUT_TOOLTIPS = (
		"[frames, H, W] — the circle following the path. Generate the OBJECT here.",
		"The same thing as visible frames, to preview or use as a control image.",
		"How many frames were rendered.",
		"The ring just outside the circle (needs context_width > 0). Generate the "
		"SURROUNDINGS here so they meet the object instead of stopping at its edge.",
		"Everything except the circle — the whole rest of the frame.",
	)
	FUNCTION = "execute"

	def execute(self, width=1024, height=576, max_frames=81, radius=64,
				aspect="16:9", timing="recorded", feather=0, track="{}",
				radius_end=0, context_width=0):
		W = max(16, int(first(width, 1024)))
		H = max(16, int(first(height, 576)))
		n = max(1, int(first(max_frames, 81)))
		r = max(1, int(first(radius, 64)))
		r_end = int(first(radius_end, 0))
		ring = max(0, int(first(context_width, 0)))
		soft = int(first(feather, 0))
		mode = str(first(timing, "recorded"))
		pts = parse_track(first(track, "{}"))
		if not pts:
			print("[tinode] Track Motion · Circle: nothing drawn yet — returning "
				  f"{n} empty frame(s). Drag across the editor to record a path.")
			empty = torch.zeros((n, H, W), dtype=torch.float32)
			return (empty, torch.zeros((n, H, W, 3), dtype=torch.float32), n,
					empty.clone(), torch.ones((n, H, W), dtype=torch.float32))

		positions = sample_positions(pts, n, mode)
		radii = ramp_radii(n, r, r_end)
		mask = render_circles(positions, W, H, radii, soft)
		# The ring is the bigger disc minus the smaller one, so it hugs the
		# object's edge exactly however the radius ramps.
		if ring > 0:
			outer = render_circles(positions, W, H, [x + ring for x in radii], soft)
			context = (outer - mask).clamp(0.0, 1.0)
		else:
			context = torch.zeros_like(mask)
		dur = (pts[-1][2] - pts[0][2]) / 1000.0
		grew = f" -> {int(radii[-1])}" if r_end > 0 and int(radii[-1]) != r else ""
		print(f"[tinode] Track Motion · Circle: {len(pts)} point(s) over {dur:.2f}s "
			  f"-> {n} frame(s) of {W}x{H}, r={r}{grew}"
			  f"{f', context ring {ring}px' if ring else ''} ({mode} timing).")
		return (mask, mask.unsqueeze(-1).expand(-1, -1, -1, 3).contiguous(), n,
				context, (1.0 - mask))
