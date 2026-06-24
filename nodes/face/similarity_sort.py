"""Face Similarity Sort — order a face batch into an identity gradient.

Start from a chosen index, embed every face with ArcFace (InsightFace),
then greedily walk: from the current face pick the most similar unused face
as the next one, repeat. The result is a batch reordered so each face looks
like its neighbour — a smooth drift through identities, ready to morph.

Faces with no detection are pushed to the end (treated as least similar).
Embeddings are L2-normalised, so cosine similarity is a plain dot product.
"""

from __future__ import annotations

import numpy as np
import torch

from ...base import TiNode
from ...registry import register

# Cache FaceAnalysis apps by (model, det_size) so we load ONNX once per config,
# not once per node run. InsightFace is imported lazily inside so the pack
# still loads even when insightface isn't installed yet.
_APPS: dict = {}


def _get_app(name: str, det_size: int):
	key = (name, det_size)
	if key not in _APPS:
		from insightface.app import FaceAnalysis
		app = FaceAnalysis(name=name, allowed_modules=["detection", "recognition"])
		app.prepare(ctx_id=0, det_size=(det_size, det_size))
		_APPS[key] = app
	return _APPS[key]


def greedy_chain(M: torch.Tensor, valid: torch.Tensor, start: int) -> list[int]:
	"""Greedy nearest-neighbour ordering over normalised embeddings.

	M: [N,D] L2-normalised rows (zero rows allowed where valid is False).
	valid: [N] bool — False = no face detected for that index.
	Returns a permutation of range(N) beginning at `start`. When either end
	of a comparison has no face, similarity is forced low so faceless frames
	sink to the end.
	"""
	n = M.shape[0]
	order = [start]
	used = {start}
	for _ in range(n - 1):
		last = order[-1]
		best, best_sim = -1, -2.0
		for j in range(n):
			if j in used:
				continue
			if bool(valid[last]) and bool(valid[j]):
				sim = float(torch.dot(M[last], M[j]))   # cosine (rows normed)
			else:
				sim = -1.0                              # faceless -> least similar
			if sim > best_sim:
				best_sim, best = sim, j
		order.append(best)
		used.add(best)
	return order


@register
class FaceSimilaritySort(TiNode):
	DISPLAY_NAME = "Face Similarity Sort (ti)"
	CATEGORY = "tinode/face"

	@classmethod
	def INPUT_TYPES(cls):
		return {
			"required": {
				"images": ("IMAGE",),
				"start_index": ("INT", {"default": 0, "min": 0, "max": 99999}),
			},
			"optional": {
				"model": (["buffalo_l", "antelopev2"], {"default": "buffalo_l"}),
				"det_size": ("INT", {"default": 640, "min": 128, "max": 2048, "step": 32}),
			},
		}

	RETURN_TYPES = ("IMAGE", "STRING")
	RETURN_NAMES = ("sorted_images", "order")
	FUNCTION = "execute"

	def _embed(self, app, frame: torch.Tensor):
		# frame [H,W,C] float 0-1 RGB -> InsightFace wants uint8 BGR
		rgb = (frame.clamp(0, 1).cpu().numpy() * 255).astype(np.uint8)
		bgr = np.ascontiguousarray(rgb[:, :, ::-1])
		faces = app.get(bgr)
		if not faces:
			return None
		f = max(faces, key=lambda x: (x.bbox[2] - x.bbox[0]) * (x.bbox[3] - x.bbox[1]))
		return torch.from_numpy(f.normed_embedding).float()

	# Consume the WHOLE batch in one call. A LIST source (SEGSToImageList,
	# MaskCrop feeding through a list node, ...) would otherwise make ComfyUI
	# run this once per frame -> n=1 -> nothing to sort. Collapse to one batch.
	INPUT_IS_LIST = True

	@staticmethod
	def _first(v, default=None):
		if isinstance(v, list):
			return v[0] if v else default
		return v

	def execute(self, images, start_index, model="buffalo_l", det_size=640):
		imgs = images if isinstance(images, list) else [images]
		images = torch.cat(
			[im if im.dim() == 4 else im.unsqueeze(0) for im in imgs], dim=0
		)
		start_index = int(self._first(start_index, 0))
		model = self._first(model, "buffalo_l")
		det_size = int(self._first(det_size, 640))

		n = images.shape[0]
		start_index = max(0, min(start_index, n - 1))
		if n <= 1:
			return (images, "0" if n else "")

		app = _get_app(model, det_size)
		embs = [self._embed(app, images[i]) for i in range(n)]
		dim = next((e.shape[0] for e in embs if e is not None), 512)

		M = torch.zeros(n, dim)
		valid = torch.zeros(n, dtype=torch.bool)
		for i, e in enumerate(embs):
			if e is not None:
				M[i] = e
				valid[i] = True

		order = greedy_chain(M, valid, start_index)
		sorted_images = images[torch.tensor(order, dtype=torch.long)]
		return (sorted_images, ",".join(map(str, order)))
