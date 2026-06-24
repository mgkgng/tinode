"""Face Landmark Morph — feature-aware morph across a sorted face batch.

Unlike frame interpolation (which blends pixels blind to anatomy), this
warps by facial landmarks: eyes map to eyes, mouth to mouth. For each
consecutive pair it detects a 468-point mesh (mediapipe) on both faces,
walks an intermediate shape, warps both faces onto it via per-triangle
affine (Delaunay), and cross-dissolves. The result slides features into
place — the classic face morph.

Faces that fail detection fall back to a plain cross-dissolve for any pair
that touches them, so the sequence never breaks.
"""

from __future__ import annotations

import numpy as np
import torch

from ...base import TiNode
from ...registry import register

# mediapipe FaceMesh is heavy to construct; build once, reuse.
_MESH = None


def _get_mesh():
	global _MESH
	if _MESH is None:
		import mediapipe as mp
		_MESH = mp.solutions.face_mesh.FaceMesh(
			static_image_mode=True, max_num_faces=1,
			refine_landmarks=False, min_detection_confidence=0.3,
		)
	return _MESH


def _border_points(w: int, h: int) -> np.ndarray:
	"""8 fixed frame-edge anchors so the background warps smoothly too."""
	return np.array([
		[0, 0], [w / 2, 0], [w - 1, 0], [w - 1, h / 2],
		[w - 1, h - 1], [w / 2, h - 1], [0, h - 1], [0, h / 2],
	], dtype=np.float32)


def _landmarks(frame_u8: np.ndarray):
	"""Return (N,2) pixel landmarks for an RGB uint8 frame, or None."""
	mesh = _get_mesh()
	res = mesh.process(frame_u8)
	if not res.multi_face_landmarks:
		return None
	h, w = frame_u8.shape[:2]
	lm = res.multi_face_landmarks[0].landmark
	return np.array([[p.x * w, p.y * h] for p in lm], dtype=np.float32)


def delaunay_indices(points: np.ndarray, w: int, h: int):
	"""Triangle list as point-index triples, via Delaunay over `points`."""
	import cv2
	rect = (0, 0, w + 1, h + 1)
	subdiv = cv2.Subdiv2D(rect)
	for p in points:
		subdiv.insert((float(np.clip(p[0], 0, w)), float(np.clip(p[1], 0, h))))
	tris = []
	for t in subdiv.getTriangleList():
		pts = [(t[0], t[1]), (t[2], t[3]), (t[4], t[5])]
		idx = []
		for px, py in pts:
			d = np.hypot(points[:, 0] - px, points[:, 1] - py)
			idx.append(int(d.argmin()))
		if len({*idx}) == 3:
			tris.append(idx)
	return tris


def _warp_triangle(src, dst, t_src, t_dst):
	"""Affine-warp the triangle t_src in src into t_dst in dst (in place)."""
	import cv2
	r1 = cv2.boundingRect(np.float32([t_src]))
	r2 = cv2.boundingRect(np.float32([t_dst]))
	x1, y1, w1, h1 = r1
	x2, y2, w2, h2 = r2
	if w1 == 0 or h1 == 0 or w2 == 0 or h2 == 0:
		return
	t1 = [(p[0] - x1, p[1] - y1) for p in t_src]
	t2 = [(p[0] - x2, p[1] - y2) for p in t_dst]
	src_crop = src[y1:y1 + h1, x1:x1 + w1]
	M = cv2.getAffineTransform(np.float32(t1), np.float32(t2))
	warped = cv2.warpAffine(
		src_crop, M, (w2, h2), flags=cv2.INTER_LINEAR,
		borderMode=cv2.BORDER_REFLECT_101,
	)
	mask = np.zeros((h2, w2, 1), dtype=np.float32)
	cv2.fillConvexPoly(mask, np.int32(t2), (1.0,), cv2.LINE_AA, 0)
	region = dst[y2:y2 + h2, x2:x2 + w2]
	dst[y2:y2 + h2, x2:x2 + w2] = region * (1 - mask) + warped * mask


def _warp_to_shape(img, src_pts, dst_pts, tris):
	import cv2
	out = np.zeros_like(img)
	for a, b, c in tris:
		_warp_triangle(img, out, [src_pts[a], src_pts[b], src_pts[c]],
					   [dst_pts[a], dst_pts[b], dst_pts[c]])
	return out


def morph_pair(img_a, img_b, la, lb, steps, tris):
	"""Yield `steps` morph frames from img_a->img_b (t in [0,1))."""
	frames = []
	for s in range(steps):
		t = s / steps
		lt = (1 - t) * la + t * lb
		wa = _warp_to_shape(img_a, la, lt, tris)
		wb = _warp_to_shape(img_b, lb, lt, tris)
		frames.append((1 - t) * wa + t * wb)
	return frames


@register
class FaceLandmarkMorph(TiNode):
	DISPLAY_NAME = "Face Landmark Morph (ti)"
	CATEGORY = "tinode/face"
	INPUT_IS_LIST = True

	@classmethod
	def INPUT_TYPES(cls):
		return {
			"required": {
				"images": ("IMAGE",),
				"steps": ("INT", {"default": 12, "min": 1, "max": 120, "step": 1}),
			},
			"optional": {
				"loop": ("BOOLEAN", {"default": False}),
			},
		}

	RETURN_TYPES = ("IMAGE",)
	RETURN_NAMES = ("frames",)
	FUNCTION = "execute"

	@staticmethod
	def _first(v, default=None):
		if isinstance(v, list):
			return v[0] if v else default
		return v

	def execute(self, images, steps, loop=False):
		imgs = images if isinstance(images, list) else [images]
		batch = torch.cat(
			[im if im.dim() == 4 else im.unsqueeze(0) for im in imgs], dim=0
		)
		steps = int(self._first(steps, 12))
		loop = bool(self._first(loop, False))

		n = batch.shape[0]
		if n < 2:
			return (batch,)

		np_imgs = [(batch[i].clamp(0, 1).cpu().numpy()).astype(np.float32) for i in range(n)]
		h, w = np_imgs[0].shape[:2]
		border = _border_points(w, h)

		# Landmarks per frame (+ frame anchors). None where detection failed.
		lms = []
		for im in np_imgs:
			u8 = (im * 255).astype(np.uint8)
			pts = _landmarks(u8)
			lms.append(None if pts is None else np.vstack([pts, border]))

		seq = list(range(n)) + ([0] if loop else [])
		out = []
		for k in range(len(seq) - 1):
			i, j = seq[k], seq[k + 1]
			a, b = np_imgs[i], np_imgs[j]
			la, lb = lms[i], lms[j]
			if la is None or lb is None:
				# fallback: plain cross-dissolve, no warp
				for s in range(steps):
					t = s / steps
					out.append((1 - t) * a + t * b)
				continue
			tris = delaunay_indices((la + lb) / 2.0, w, h)
			out.extend(morph_pair(a, b, la, lb, steps, tris))

		if not loop:
			out.append(np_imgs[-1])              # final endpoint, no dup at joins

		stacked = np.clip(np.stack(out, 0), 0, 1).astype(np.float32)
		return (torch.from_numpy(stacked),)
