"""Pick Segments — interactively include/exclude SAM3 segments across a video.

Takes the TI_SAM3_SEGMENTS output of EasySAM3 Segment (every detection per
frame, unmerged) plus the original IMAGE, and drives an in-node editor
(web/pick_segments.js): step through frames, see every segment as a colorized
box, click an object to toggle it on/off. The choice is per-id and global, so
excluding an object drops it on every frame it appears in.

SAM3 runs once upstream (EasySAM3 is cached); toggling here only re-runs this
cheap node. Outputs:
  mask      union of the included segments, [frames,H,W]
  image     the source frames with only the included segments colorized
  segments  the same TI_SAM3_SEGMENTS filtered to the included ids (chainable)

The editor needs pixels in the browser, so on execution we save, per frame, a
downscaled source image and a "label map" (each pixel encodes which segment
owns it) to the temp dir, and hand the browser a manifest of boxes. That work
is best-effort / lazily imported: with no PIL / folder_paths the node still
produces all three outputs, you just don't get the visual editor.
"""

from __future__ import annotations

import hashlib
import json
import os

import torch

from ...base import TiNode
from ...registry import register
from ...schema import validate_segments

_PREVIEW_MAX_SIDE = 768
# How many distinct inputs keep their editor assets on disk before the oldest
# are pruned. Each is a few hundred PNGs.
_MAX_CACHED_INPUTS = 4


def color_for_id(seg_id: int):
	"""Deterministic bright RGB (0..1 floats) for a segment id.

	Golden-ratio hue spacing so adjacent ids look distinct. MUST match the JS
	color function so a segment is the same color in the editor and the output.
	"""
	h = (seg_id * 0.61803398875) % 1.0
	s, v = 0.65, 1.0
	i = int(h * 6)
	f = h * 6 - i
	p, q, t = v * (1 - s), v * (1 - s * f), v * (1 - s * (1 - f))
	r, g, b = [
		(v, t, p), (q, v, p), (p, v, t), (p, q, v), (t, p, v), (v, p, q),
	][i % 6]
	return r, g, b


def prune_asset_cache(root: str, keep: str, max_dirs: int = _MAX_CACHED_INPUTS) -> int:
	"""Keep the `max_dirs` most recent editor-asset folders under `root`.

	Each distinct input gets its own folder holding one PNG per frame (two, for
	Pick Segments). A 133-frame clip is a few hundred files, and every re-crop or
	changed image mints a fresh folder — so without pruning this grows without
	bound for the life of the temp dir. Oldest-first by mtime; the folder for the
	current input is never removed. Returns how many were deleted.
	"""
	import shutil  # noqa: PLC0415

	try:
		entries = [
			(os.path.getmtime(p), p)
			for name in os.listdir(root)
			for p in (os.path.join(root, name),)
			if os.path.isdir(p) and name != keep
		]
	except OSError:
		return 0
	# -1 for the current input's folder, which is kept regardless.
	excess = len(entries) - max(0, max_dirs - 1)
	if excess <= 0:
		return 0
	removed = 0
	for _, path in sorted(entries)[:excess]:
		try:
			shutil.rmtree(path)
			removed += 1
		except OSError:
			pass
	return removed


def _seg_signature(seg_data) -> str:
	"""Stable hash of the segment geometry (NOT the selection).

	Preview assets depend only on the detections, so keying on this lets repeat
	executions (e.g. after a toggle) skip regenerating identical files.
	"""
	parts = [str(seg_data.get("height")), str(seg_data.get("width"))]
	for frame in seg_data.get("frames", []):
		for s in frame:
			parts.append(f'{s["id"]}:{s["bbox"]}')
	return hashlib.md5("|".join(parts).encode()).hexdigest()[:16]


def _img_signature(imgs) -> str:
	"""Cheap content hash of the source frames.

	The preview cache MUST invalidate when the *image* input changes even if the
	segments don't (e.g. rewiring image from the full frame to a crop): the seg
	signature alone would collide and re-serve the previous image's frames.
	Strided sampling keeps this fast on large batches.
	"""
	t = imgs.detach()
	flat = t.reshape(-1)
	step = max(1, flat.numel() // 8192)
	h = hashlib.md5()
	h.update(repr(tuple(t.shape)).encode())
	h.update(flat[::step].contiguous().cpu().numpy().tobytes())
	return h.hexdigest()[:16]


@register
class PickSegments(TiNode):
	DISPLAY_NAME = "Pick Segments (ti)"
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
				# JSON list of excluded ids. Empty = everything on. Hidden and
				# driven by the editor; serialized with the workflow.
				"excluded_ids": ("STRING", {"default": "[]"}),
				# Last frame the editor showed, so it reopens where you left off.
				"current_frame": ("INT", {"default": 0, "min": 0, "max": 999999}),
				"overlay_alpha": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.05,
					"tooltip": "Opacity of the colorized segments in the image output."}),
			},
		}

	RETURN_TYPES = ("MASK", "IMAGE", "TI_SAM3_SEGMENTS")
	RETURN_NAMES = ("mask", "image", "segments")
	FUNCTION = "execute"

	@staticmethod
	def _excluded_ids(raw, sig):
		"""Decode the exclusion widget, ignoring a selection made against other input.

		Accepts the current {"sig","ids"} form and the older bare-list form (which
		carries no signature, so it is trusted as-is).
		"""
		try:
			data = json.loads(raw) if raw else []
		except (ValueError, TypeError):
			return set()
		if isinstance(data, list):
			return set(data)
		if not isinstance(data, dict):
			return set()
		if data.get("sig") is not None and data.get("sig") != sig:
			return set()          # input changed -> previous selection is stale
		ids = data.get("ids")
		return set(ids) if isinstance(ids, list) else set()

	def execute(self, image, segments, excluded_ids="[]", current_frame=0, overlay_alpha=0.5):
		validate_segments(segments)
		imgs = image if image.dim() == 4 else image.unsqueeze(0)  # [N,H,W,3]
		H = int(segments.get("height", imgs.shape[1]))
		W = int(segments.get("width", imgs.shape[2]))
		frames = segments.get("frames", [])
		N = int(segments.get("num_frames", len(frames)))

		# Identity of this input. Exclusions are SAM3 track ids, which are reused
		# across clips — so a selection made against different input is stale and
		# must not silently drop objects in the new one.
		sig = f"{_seg_signature(segments)}_{_img_signature(imgs)}"
		excluded = self._excluded_ids(excluded_ids, sig)

		def included(frame):
			return [s for s in frame if s["id"] not in excluded]

		# Match the image batch length to the segment frame count where possible.
		if imgs.shape[0] != N and imgs.shape[0] == 1:
			imgs = imgs.expand(N, -1, -1, -1)

		mask_out = torch.zeros((N, H, W), dtype=torch.float32)
		image_out = torch.zeros((N, H, W, 3), dtype=torch.float32)

		for f in range(N):
			frame = frames[f] if f < len(frames) else []
			base = imgs[f] if f < imgs.shape[0] else torch.zeros(H, W, 3)
			if base.shape[0] != H or base.shape[1] != W:
				base = torch.zeros(H, W, 3)
			canvas = base.clone()
			for s in included(frame):
				x0, y0, x1, y1 = s["bbox"]
				m = s["mask"].to(torch.float32)              # [bh,bw] 0/1
				# union into the mask output
				dst = mask_out[f, y0:y1, x0:x1]
				mask_out[f, y0:y1, x0:x1] = torch.maximum(dst, m)
				# colorized overlay into the image output
				r, g, b = color_for_id(s["id"])
				color = torch.tensor([r, g, b], dtype=torch.float32)
				a = (m * overlay_alpha).unsqueeze(-1)        # [bh,bw,1]
				region = canvas[y0:y1, x0:x1, :]
				canvas[y0:y1, x0:x1, :] = region * (1 - a) + color * a
			image_out[f] = canvas

		filtered = {
			"num_frames": N, "height": H, "width": W,
			"frames": [included(frames[f] if f < len(frames) else []) for f in range(N)],
			"ids": [i for i in segments.get("ids", []) if i not in excluded],
		}
		result = (mask_out, image_out, filtered)

		manifest = self._build_assets(imgs, segments, sig)
		if manifest is None:
			return result
		return {"ui": {"ti_pick": [manifest]}, "result": result}

	# ------------------------------------------------------------------ assets
	def _build_assets(self, imgs, segments, sig):
		"""Save per-frame source + label images and return the editor manifest.

		Returns None (no editor) on any failure so the outputs still flow.
		"""
		try:
			import numpy as np  # noqa: PLC0415
			from PIL import Image  # noqa: PLC0415
			import folder_paths  # noqa: PLC0415

			H = int(segments["height"])
			W = int(segments["width"])
			frames = segments.get("frames", [])
			N = int(segments.get("num_frames", len(frames)))

			scale = min(1.0, _PREVIEW_MAX_SIDE / max(H, W))
			pw, ph = max(1, round(W * scale)), max(1, round(H * scale))

			# Key on BOTH the segments and the source image so a changed image
			# input (e.g. full frame -> crop) writes to a fresh folder instead
			# of re-serving the previous run's stale frames.
			root = os.path.join(folder_paths.get_temp_directory(), "ti_pick", sig)
			os.makedirs(root, exist_ok=True)
			subfolder = os.path.join("ti_pick", sig)
			prune_asset_cache(os.path.dirname(root), sig)

			manifest_frames = []
			for f in range(N):
				frame = frames[f] if f < len(frames) else []
				src_name = f"src_{f:05d}.png"
				lbl_name = f"lbl_{f:05d}.png"
				src_path = os.path.join(root, src_name)
				lbl_path = os.path.join(root, lbl_name)

				if not os.path.exists(src_path):
					base = imgs[f] if f < imgs.shape[0] else torch.zeros(H, W, 3)
					arr = (base.clamp(0, 1).cpu().numpy() * 255).astype(np.uint8)
					Image.fromarray(arr[..., :3]).resize((pw, ph), Image.BILINEAR).save(
						src_path, compress_level=1)

				if not os.path.exists(lbl_path):
					# Each pixel = (segment index within frame)+1, RGB-encoded as
					# value = R + G*256 (0 = background). Topmost = last painted.
					label = np.zeros((H, W), dtype=np.int32)
					for i, s in enumerate(frame):
						x0, y0, x1, y1 = s["bbox"]
						m = s["mask"].cpu().numpy().astype(bool)
						region = label[y0:y1, x0:x1]
						region[m] = i + 1
					rgb = np.zeros((H, W, 3), dtype=np.uint8)
					rgb[..., 0] = (label & 0xFF).astype(np.uint8)
					rgb[..., 1] = ((label >> 8) & 0xFF).astype(np.uint8)
					Image.fromarray(rgb).resize((pw, ph), Image.NEAREST).save(
						lbl_path, compress_level=1)

				segs_meta = [{
					"id": s["id"],
					"conf": round(float(s["conf"]), 3),
					"bbox": [round(s["bbox"][0] * scale), round(s["bbox"][1] * scale),
							 round(s["bbox"][2] * scale), round(s["bbox"][3] * scale)],
				} for s in frame]
				manifest_frames.append({
					"src": {"filename": src_name, "subfolder": subfolder, "type": "temp"},
					"label": {"filename": lbl_name, "subfolder": subfolder, "type": "temp"},
					"segs": segs_meta,
				})

			return {
				"sig": sig, "num_frames": N, "pw": pw, "ph": ph, "full_w": W, "full_h": H,
				"ids": segments.get("ids", []), "frames": manifest_frames,
			}
		except Exception as exc:  # noqa: BLE001 — editor is best-effort
			print(f"[tinode] Pick Segments editor assets unavailable: {exc!r}")
			return None
