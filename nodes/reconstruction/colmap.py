"""Run masked COLMAP structure-from-motion as a ComfyUI node."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

from ...base import TiNode
from ...registry import register
from .common import command_prefix, require_directory, run_logged


def _find_sparse_model(sparse_root: Path) -> Path:
	models = sorted(
		path for path in sparse_root.iterdir()
		if path.is_dir() and (path / "cameras.bin").is_file() and (path / "images.bin").is_file()
	)
	if not models:
		raise RuntimeError(f"COLMAP produced no usable sparse model in {sparse_root}")
	return models[0]


_COLMAP_UNSET_ENV = ("QT_PLUGIN_PATH", "QT_QPA_PLATFORM_PLUGIN_PATH")
_COLMAP_ENV = {"QT_QPA_PLATFORM": "offscreen"}


def _run_colmap(command: list[str], log_path: Path) -> str:
	"""Run COLMAP without inheriting ComfyUI/OpenCV's Qt plugin paths."""
	return run_logged(
		command,
		log_path,
		env=_COLMAP_ENV,
		unset_env=_COLMAP_UNSET_ENV,
	)


def _archive_incomplete(output: Path) -> Path:
	stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
	archive = output.with_name(f"{output.name}.failed-{stamp}")
	counter = 1
	while archive.exists():
		archive = output.with_name(f"{output.name}.failed-{stamp}-{counter}")
		counter += 1
	output.rename(archive)
	return archive


@register
class RunColmapSfM(TiNode):
	DISPLAY_NAME = "Run Masked COLMAP SfM (ti)"
	CATEGORY = "tinode/reconstruction"
	OUTPUT_NODE = True

	@classmethod
	def INPUT_TYPES(cls):
		return {
			"required": {
				"dataset": ("TI_RECON_DATASET",),
				"colmap_command": ("STRING", {"default": "colmap", "multiline": False}),
				"camera_model": (["OPENCV", "SIMPLE_RADIAL", "PINHOLE"],),
				"matcher": (["sequential", "exhaustive"],),
				"single_camera": ("BOOLEAN", {"default": True}),
				"reuse_existing": ("BOOLEAN", {"default": True}),
			},
			"optional": {
				"compute_device": (["CPU", "GPU"], {"default": "CPU"}),
				"restart_incomplete": ("BOOLEAN", {"default": True}),
			},
		}

	RETURN_TYPES = ("TI_COLMAP_MODEL", "STRING", "STRING")
	RETURN_NAMES = ("colmap_model", "model_path", "log_path")

	def execute(
		self,
		dataset,
		colmap_command="colmap",
		camera_model="OPENCV",
		matcher="sequential",
		compute_device="CPU",
		single_camera=True,
		reuse_existing=True,
		restart_incomplete=True,
	):
		import comfy.utils

		root = require_directory(dataset["root"], "Dataset")
		images = require_directory(dataset["images"], "RGB images")
		masks = require_directory(dataset["masks_colmap"], "COLMAP masks")
		out = root / "colmap"
		sparse = out / "sparse"
		log_path = out / "colmap.log"

		if sparse.is_dir() and reuse_existing:
			try:
				model_path = _find_sparse_model(sparse)
				result = {**dataset, "colmap_root": str(out), "model_path": str(model_path)}
				return (result, str(model_path), str(out / "mapping.log"))
			except RuntimeError:
				pass

		database = out / "database.db"
		if database.exists():
			if restart_incomplete:
				archive = _archive_incomplete(out)
				print(f"[tinode/reconstruction] Archived incomplete COLMAP run to {archive}", flush=True)
			else:
				raise RuntimeError(
					f"An incomplete COLMAP run already exists at {out}. "
					"Enable restart_incomplete, move it aside, or choose a new dataset name."
				)
		out.mkdir(parents=True, exist_ok=True)
		sparse.mkdir(parents=True, exist_ok=True)
		prefix = command_prefix(colmap_command)
		progress = comfy.utils.ProgressBar(3)

		feature_command = prefix + [
			"feature_extractor",
			"--database_path", str(database),
			"--image_path", str(images),
			"--ImageReader.mask_path", str(masks),
			"--ImageReader.camera_model", camera_model,
			"--ImageReader.single_camera", "1" if single_camera else "0",
			"--SiftExtraction.use_gpu", "1" if compute_device == "GPU" else "0",
		]
		_run_colmap(feature_command, log_path)
		progress.update(1)

		matcher_name = "sequential_matcher" if matcher == "sequential" else "exhaustive_matcher"
		_run_colmap(
			prefix + [
				matcher_name,
				"--database_path", str(database),
				"--SiftMatching.use_gpu", "1" if compute_device == "GPU" else "0",
			],
			out / "matching.log",
		)
		progress.update(1)

		_run_colmap(
			prefix + [
				"mapper",
				"--database_path", str(database),
				"--image_path", str(images),
				"--output_path", str(sparse),
			],
			out / "mapping.log",
		)
		progress.update(1)
		model_path = _find_sparse_model(sparse)
		result = {**dataset, "colmap_root": str(out), "model_path": str(model_path)}
		return (result, str(model_path), str(out / "mapping.log"))
