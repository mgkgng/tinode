"""Convert an existing COLMAP model to a masked Nerfstudio dataset."""

from __future__ import annotations

import shutil
from pathlib import Path

from PIL import Image

from ...base import TiNode
from ...registry import register
from .common import command_prefix, read_json, require_directory, run_logged, write_json


def _attach_masks(output: Path, source_masks: Path) -> int:
	transforms_path = output / "transforms.json"
	if not transforms_path.is_file():
		raise RuntimeError(f"Nerfstudio did not create {transforms_path}")
	data = read_json(transforms_path)
	frames = data.get("frames", [])
	if not frames:
		raise RuntimeError("Nerfstudio transforms.json contains no frames.")
	target_names = sorted(Path(frame["file_path"]).name for frame in frames)
	source_files = sorted(path for path in source_masks.iterdir() if path.suffix.lower() == ".png")
	if len(source_files) != len(target_names):
		raise RuntimeError(
			f"Nerfstudio frame/mask count mismatch: {len(target_names)} frames, "
			f"{len(source_files)} masks."
		)
	sources_by_name = {path.name: path for path in source_files}
	if all(name in sources_by_name for name in target_names):
		source_for_target = {name: sources_by_name[name] for name in target_names}
	else:
		# ns-process-data normally renames copied images. Both lists originate
		# from the same ordered frame batch, so preserve their sorted pairing.
		source_for_target = dict(zip(target_names, source_files))

	mask_root = output / "masks"
	mask_root.mkdir(parents=True, exist_ok=True)
	for frame in frames:
		image_path = output / frame["file_path"]
		if not image_path.is_file():
			raise RuntimeError(f"Nerfstudio image is missing: {image_path}")
		basename = image_path.name
		source = source_for_target[basename]
		with Image.open(image_path) as image, Image.open(source) as mask:
			converted = mask.convert("L")
			if converted.size != image.size:
				converted = converted.resize(image.size, Image.Resampling.NEAREST)
			converted.save(mask_root / basename)
		frame["mask_path"] = (Path("masks") / basename).as_posix()

	backup = output / "transforms.before-masks.json"
	if not backup.exists():
		shutil.copy2(transforms_path, backup)
	write_json(transforms_path, data)

	# Nerfstudio may choose a downscaled image directory. Mirror every emitted
	# image scale with nearest-neighbour binary masks.
	for image_dir in sorted(output.glob("images_*")):
		if not image_dir.is_dir():
			continue
		suffix = image_dir.name.removeprefix("images")
		scaled_masks = output / f"masks{suffix}"
		scaled_masks.mkdir(parents=True, exist_ok=True)
		for image_path in image_dir.iterdir():
			if not image_path.is_file():
				continue
			base_mask = mask_root / image_path.name
			if not base_mask.is_file():
				continue
			with Image.open(image_path) as image, Image.open(base_mask) as mask:
				mask.convert("L").resize(image.size, Image.Resampling.NEAREST).save(
					scaled_masks / image_path.name
				)
	return len(frames)


@register
class PrepareNerfstudioDataset(TiNode):
	DISPLAY_NAME = "Prepare Masked Nerfstudio Dataset (ti)"
	CATEGORY = "tinode/reconstruction"
	OUTPUT_NODE = True

	@classmethod
	def INPUT_TYPES(cls):
		return {
			"required": {
				"colmap_model": ("TI_COLMAP_MODEL",),
				"process_command": ("STRING", {"default": "ns-process-data", "multiline": False}),
				"num_downscales": ("INT", {"default": 2, "min": 0, "max": 4}),
				"reuse_existing": ("BOOLEAN", {"default": True}),
			},
		}

	RETURN_TYPES = ("TI_NERFSTUDIO_DATASET", "STRING", "STRING")
	RETURN_NAMES = ("nerfstudio_dataset", "dataset_path", "log_path")

	def execute(self, colmap_model, process_command="ns-process-data", num_downscales=2, reuse_existing=True):
		root = require_directory(colmap_model["root"], "Dataset")
		images = require_directory(colmap_model["images"], "RGB images")
		masks = require_directory(colmap_model["masks_keep"], "Training masks")
		model_path = require_directory(colmap_model["model_path"], "COLMAP model")
		output = root / "nerfstudio"
		transforms = output / "transforms.json"
		log_path = output / "process.log"

		if transforms.is_file() and reuse_existing:
			result = {**colmap_model, "nerfstudio_root": str(output), "transforms": str(transforms)}
			return (result, str(output), str(log_path))
		if output.exists() and any(output.iterdir()):
			raise RuntimeError(
				f"An incomplete Nerfstudio dataset already exists at {output}. "
				"Move it aside or choose a new dataset name before retrying."
			)
		output.mkdir(parents=True, exist_ok=True)
		command = command_prefix(process_command) + [
			"images",
			"--data", str(images),
			"--output-dir", str(output),
			"--skip-colmap",
			"--colmap-model-path", str(model_path),
			"--num-downscales", str(int(num_downscales)),
		]
		run_logged(command, log_path)
		_attach_masks(output, masks)
		result = {**colmap_model, "nerfstudio_root": str(output), "transforms": str(transforms)}
		return (result, str(output), str(log_path))
