"""Export matched RGB frames and dynamic masks for reconstruction."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from ...base import TiNode
from ...registry import register
from .common import read_json, safe_name, write_json


def _as_image_batch(images) -> torch.Tensor:
	items = images if isinstance(images, list) else [images]
	return torch.cat([item if item.dim() == 4 else item.unsqueeze(0) for item in items], dim=0)


def _as_mask_batch(masks) -> torch.Tensor:
	items = masks if isinstance(masks, list) else [masks]
	return torch.cat([item if item.dim() == 3 else item.unsqueeze(0) for item in items], dim=0)


def _save_rgb(path: Path, frame: torch.Tensor) -> None:
	array = (frame[..., :3].detach().cpu().clamp(0, 1).numpy() * 255.0).round().astype(np.uint8)
	Image.fromarray(array, mode="RGB").save(path)


def _save_mask(path: Path, mask: torch.Tensor) -> None:
	array = (mask.detach().cpu().clamp(0, 1).numpy() * 255.0).round().astype(np.uint8)
	Image.fromarray(array, mode="L").save(path)


def _load_mask_batch(directory: Path, filenames: list[str]) -> torch.Tensor:
	masks = []
	for filename in filenames:
		path = directory / filename
		if not path.is_file():
			raise RuntimeError(f"Existing reconstruction mask is missing: {path}")
		with Image.open(path) as image:
			array = np.asarray(image.convert("L"), dtype=np.float32) / 255.0
		masks.append(torch.from_numpy(array))
	return torch.stack(masks)


@register
class ExportReconstructionDataset(TiNode):
	"""Persist IMAGE/MASK batches with conventions required by SfM and 3DGS."""

	DISPLAY_NAME = "Export Reconstruction Dataset (ti)"
	CATEGORY = "tinode/reconstruction"
	INPUT_IS_LIST = True
	OUTPUT_NODE = True

	@classmethod
	def INPUT_TYPES(cls):
		return {
			"required": {
				"images": ("IMAGE",),
				"masks": ("MASK",),
				"dataset_name": ("STRING", {"default": "reconstruction", "multiline": False}),
				"mask_mode": (["white_is_dynamic", "white_is_keep"],),
				"mask_threshold": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),
				"dilate_pixels": ("INT", {"default": 8, "min": 0, "max": 128, "step": 1}),
			},
			"optional": {
				"reuse_existing": ("BOOLEAN", {"default": True}),
			},
		}

	RETURN_TYPES = ("TI_RECON_DATASET", "STRING", "MASK")
	RETURN_NAMES = ("dataset", "dataset_path", "keep_masks")

	@staticmethod
	def _first(value, default=None):
		if isinstance(value, list):
			return value[0] if value else default
		return value

	def execute(
		self,
		images,
		masks,
		dataset_name,
		mask_mode="white_is_dynamic",
		mask_threshold=0.5,
		dilate_pixels=8,
		reuse_existing=True,
	):
		import folder_paths

		image_batch = _as_image_batch(images)
		mask_batch = _as_mask_batch(masks)
		if image_batch.shape[0] != mask_batch.shape[0]:
			raise RuntimeError(
				f"Image/mask batch mismatch: {image_batch.shape[0]} images, "
				f"{mask_batch.shape[0]} masks. Export both from the same frame batch."
			)
		if tuple(image_batch.shape[1:3]) != tuple(mask_batch.shape[1:3]):
			raise RuntimeError(
				f"Image/mask resolution mismatch: {tuple(image_batch.shape[1:3])} vs "
				f"{tuple(mask_batch.shape[1:3])}."
			)

		name = safe_name(str(self._first(dataset_name, "reconstruction")), "reconstruction")
		mode = str(self._first(mask_mode, "white_is_dynamic"))
		threshold = float(self._first(mask_threshold, 0.5))
		dilate = int(self._first(dilate_pixels, 9))
		reuse = bool(self._first(reuse_existing, True))
		root = Path(folder_paths.get_output_directory()).resolve() / "tinode" / "reconstruction" / name
		manifest_path = root / "manifest.json"
		if root.exists() and any(root.iterdir()):
			if not reuse or not manifest_path.is_file():
				raise RuntimeError(
					f"Dataset directory already exists and is not reusable: {root}\n"
					"Enable reuse_existing or choose another dataset_name."
				)
			manifest = read_json(manifest_path)
			expected = {
				"frame_count": int(image_batch.shape[0]),
				"width": int(image_batch.shape[2]),
				"height": int(image_batch.shape[1]),
				"mask_mode_input": mode,
				"mask_threshold": threshold,
				"dilate_pixels": dilate,
			}
			mismatches = [
				f"{key}: stored={manifest.get(key)!r}, requested={value!r}"
				for key, value in expected.items()
				if manifest.get(key) != value
			]
			if mismatches:
				raise RuntimeError(
					f"Existing dataset settings do not match: {root}\n"
					+ "\n".join(mismatches)
					+ "\nChoose another dataset_name to create a new dataset."
				)
			frames = manifest.get("frames", [])
			keep_dir = root / "masks_keep"
			keep = _load_mask_batch(keep_dir, frames)
			dataset = {
				"root": str(root),
				"images": str(root / "images"),
				"masks_dynamic": str(root / "masks_dynamic"),
				"masks_keep": str(keep_dir),
				"masks_colmap": str(root / "masks_colmap"),
				"manifest": str(manifest_path),
			}
			return (dataset, str(root), keep)

		images_dir = root / "images"
		dynamic_dir = root / "masks_dynamic"
		keep_dir = root / "masks_keep"
		colmap_dir = root / "masks_colmap"
		for directory in (images_dir, dynamic_dir, keep_dir, colmap_dir):
			directory.mkdir(parents=True, exist_ok=True)

		binary = (mask_batch > threshold).float()
		dynamic = binary if mode == "white_is_dynamic" else 1.0 - binary
		if dilate > 0:
			kernel = 2 * dilate + 1
			dynamic = F.max_pool2d(
				dynamic.unsqueeze(1), kernel_size=kernel, stride=1, padding=kernel // 2
			).squeeze(1)
		keep = 1.0 - dynamic

		frames = []
		for index, (image, dynamic_mask, keep_mask) in enumerate(zip(image_batch, dynamic, keep)):
			filename = f"frame_{index:06d}.png"
			_save_rgb(images_dir / filename, image)
			_save_mask(dynamic_dir / filename, dynamic_mask)
			_save_mask(keep_dir / filename, keep_mask)
			# COLMAP's canonical per-image mask name appends `.png` to the
			# complete source filename, hence frame_000000.png.png.
			_save_mask(colmap_dir / f"{filename}.png", keep_mask)
			frames.append(filename)

		manifest = {
			"version": 1,
			"frame_count": len(frames),
			"width": int(image_batch.shape[2]),
			"height": int(image_batch.shape[1]),
			"mask_mode_input": mode,
			"mask_threshold": threshold,
			"dilate_pixels": dilate,
			"frames": frames,
		}
		write_json(manifest_path, manifest)
		dataset = {
			"root": str(root),
			"images": str(images_dir),
			"masks_dynamic": str(dynamic_dir),
			"masks_keep": str(keep_dir),
			"masks_colmap": str(colmap_dir),
			"manifest": str(manifest_path),
		}
		return (dataset, str(root), keep)
