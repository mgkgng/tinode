"""Launch Splatfacto and export its Gaussian representation."""

from __future__ import annotations

from pathlib import Path

from ...base import TiNode
from ...registry import register
from .common import command_prefix, require_directory, require_file, run_logged, safe_name


def _latest_config(output: Path) -> Path:
	configs = sorted(output.rglob("config.yml"), key=lambda path: path.stat().st_mtime, reverse=True)
	if not configs:
		raise RuntimeError(f"Splatfacto completed but no config.yml was found in {output}")
	return configs[0]


@register
class TrainSplatfacto(TiNode):
	DISPLAY_NAME = "Train Splatfacto (ti)"
	CATEGORY = "tinode/reconstruction"
	OUTPUT_NODE = True

	@classmethod
	def INPUT_TYPES(cls):
		return {
			"required": {
				"dataset": ("TI_NERFSTUDIO_DATASET",),
				"train_command": ("STRING", {"default": "ns-train", "multiline": False}),
				"method": (["splatfacto", "splatfacto-big"],),
				"experiment_name": ("STRING", {"default": "gaussian-splat", "multiline": False}),
				"max_iterations": ("INT", {"default": 30000, "min": 100, "max": 200000, "step": 100}),
				"unload_comfy_models": ("BOOLEAN", {"default": True}),
				"reuse_existing": ("BOOLEAN", {"default": True}),
			},
		}

	RETURN_TYPES = ("TI_SPLAT_RUN", "STRING", "STRING")
	RETURN_NAMES = ("splat_run", "config_path", "log_path")

	def execute(
		self,
		dataset,
		train_command="ns-train",
		method="splatfacto",
		experiment_name="gaussian-splat",
		max_iterations=30000,
		unload_comfy_models=True,
		reuse_existing=True,
	):
		root = require_directory(dataset["root"], "Dataset")
		nerfstudio_root = require_directory(dataset["nerfstudio_root"], "Nerfstudio dataset")
		experiment = safe_name(experiment_name, "gaussian-splat")
		output = root / "splatfacto" / experiment
		log_path = output / "train.log"

		if output.is_dir() and reuse_existing:
			try:
				config = _latest_config(output)
				result = {**dataset, "splat_output": str(output), "config": str(config)}
				return (result, str(config), str(log_path))
			except RuntimeError:
				pass
		if output.exists() and any(output.iterdir()):
			raise RuntimeError(
				f"An incomplete Splatfacto run already exists at {output}. "
				"Move it aside or change experiment_name before retrying."
			)
		output.mkdir(parents=True, exist_ok=True)

		if unload_comfy_models:
			import comfy.model_management
			comfy.model_management.unload_all_models()
			comfy.model_management.soft_empty_cache()

		command = command_prefix(train_command) + [
			method,
			"--data", str(nerfstudio_root),
			"--output-dir", str(output),
			"--experiment-name", experiment,
			"--max-num-iterations", str(int(max_iterations)),
			"--vis", "tensorboard",
		]
		run_logged(command, log_path)
		config = _latest_config(output)
		result = {**dataset, "splat_output": str(output), "config": str(config)}
		return (result, str(config), str(log_path))


@register
class ExportGaussianSplat(TiNode):
	DISPLAY_NAME = "Export Gaussian Splat PLY (ti)"
	CATEGORY = "tinode/reconstruction"
	OUTPUT_NODE = True

	@classmethod
	def INPUT_TYPES(cls):
		return {
			"required": {
				"splat_run": ("TI_SPLAT_RUN",),
				"export_command": ("STRING", {"default": "ns-export", "multiline": False}),
				"output_name": ("STRING", {"default": "export", "multiline": False}),
				"reuse_existing": ("BOOLEAN", {"default": True}),
			},
		}

	RETURN_TYPES = ("STRING", "STRING")
	RETURN_NAMES = ("ply_path", "log_path")

	def execute(self, splat_run, export_command="ns-export", output_name="export", reuse_existing=True):
		root = require_directory(splat_run["splat_output"], "Splatfacto output")
		config = require_file(splat_run["config"], "Splatfacto config")
		output = root / safe_name(output_name, "export")
		log_path = output / "export.log"
		ply_files = sorted(output.rglob("*.ply")) if output.is_dir() else []
		if ply_files and reuse_existing:
			return (str(ply_files[0]), str(log_path))
		if output.exists() and any(output.iterdir()):
			raise RuntimeError(
				f"Export directory already exists without a PLY: {output}. "
				"Move it aside or change output_name before retrying."
			)
		output.mkdir(parents=True, exist_ok=True)
		command = command_prefix(export_command) + [
			"gaussian-splat",
			"--load-config", str(config),
			"--output-dir", str(output),
		]
		run_logged(command, log_path)
		ply_files = sorted(output.rglob("*.ply"))
		if not ply_files:
			raise RuntimeError(f"ns-export completed but no .ply was found in {output}")
		return (str(ply_files[0]), str(log_path))
