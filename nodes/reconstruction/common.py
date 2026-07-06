"""Shared helpers for external reconstruction tools.

COLMAP and Nerfstudio intentionally remain separate installations. These
helpers launch their CLIs without a shell, stream logs into ComfyUI's console,
and keep every artifact inside a dataset directory created by tinode.
"""

from __future__ import annotations

import json
import os
import re
import shlex
import shutil
import subprocess
from collections import deque
from pathlib import Path


_SAFE_NAME = re.compile(r"[^A-Za-z0-9._-]+")


def safe_name(value: str, fallback: str) -> str:
	name = _SAFE_NAME.sub("-", value.strip()).strip(".-")
	return name or fallback


def command_prefix(spec: str) -> list[str]:
	"""Parse a command prefix without invoking a shell.

	Supports both direct binaries (`/env/bin/ns-train`) and safe wrappers such
	as `conda run -n reconstruction ns-train`.
	"""
	parts = shlex.split(spec)
	if not parts:
		raise RuntimeError("External command cannot be empty.")
	executable = parts[0]
	if "/" in executable:
		if not Path(executable).expanduser().is_file():
			raise RuntimeError(f"Executable not found: {executable}")
		parts[0] = str(Path(executable).expanduser())
	elif shutil.which(executable) is None:
		raise RuntimeError(f"Executable not found on PATH: {executable}")
	return parts


def run_logged(
	command: list[str],
	log_path: Path,
	cwd: Path | None = None,
	*,
	env: dict[str, str] | None = None,
	unset_env: tuple[str, ...] = (),
) -> str:
	"""Run a command, stream merged output, and raise with a useful tail."""
	log_path.parent.mkdir(parents=True, exist_ok=True)
	print(f"[tinode/reconstruction] $ {shlex.join(command)}", flush=True)
	process_env = os.environ.copy()
	for name in unset_env:
		process_env.pop(name, None)
	if env:
		process_env.update(env)
	tail: deque[str] = deque(maxlen=40)
	with log_path.open("w", encoding="utf-8") as log:
		process = subprocess.Popen(
			command,
			cwd=str(cwd) if cwd else None,
			env=process_env,
			stdout=subprocess.PIPE,
			stderr=subprocess.STDOUT,
			text=True,
			bufsize=1,
		)
		assert process.stdout is not None
		for line in process.stdout:
			log.write(line)
			log.flush()
			tail.append(line.rstrip())
			print(f"[tinode/reconstruction] {line}", end="", flush=True)
		return_code = process.wait()

	if return_code != 0:
		detail = "\n".join(tail)
		raise RuntimeError(
			f"External command failed with exit code {return_code}.\n"
			f"Log: {log_path}\n\n{detail}"
		)
	return "\n".join(tail)


def require_directory(value: str | Path, label: str) -> Path:
	path = Path(value).expanduser().resolve()
	if not path.is_dir():
		raise RuntimeError(f"{label} directory not found: {path}")
	return path


def require_file(value: str | Path, label: str) -> Path:
	path = Path(value).expanduser().resolve()
	if not path.is_file():
		raise RuntimeError(f"{label} file not found: {path}")
	return path


def read_json(path: Path) -> dict:
	with path.open("r", encoding="utf-8") as handle:
		return json.load(handle)


def write_json(path: Path, value: dict) -> None:
	with path.open("w", encoding="utf-8") as handle:
		json.dump(value, handle, indent=2)
		handle.write("\n")
