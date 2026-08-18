"""Load Videos — many clips from a folder as an Inspire ITEM_LIST of native VIDEOs.

One node that gathers several video files and hands them to a loop, so a
sub-workflow runs once per clip. Built for Inspire's ▶Foreach List: wire
`item_list` into ForeachListBegin and each iteration gets one video.

Why this stays light on RAM even for a folder of multi-GB clips: every item is
a **lazy** native VIDEO (comfy_api's VideoFromFile) — just a file path until
something asks for pixels. The whole list weighs almost nothing; only the clip
the current iteration touches is decoded, and it is released before the next.
That is the streaming answer to "load >5GB of video and work on them one by
one" — you never hold more than one clip's frames at a time. Contrast with
loading every clip to an IMAGE batch up front, which would need the sum of all
of them in memory at once.

Files come from `directory`: a path relative to ComfyUI's input/ folder, or an
absolute path (so you can point straight at a source folder elsewhere without
copying gigabytes into input/). With `filenames` empty every video file in the
folder is taken, sorted by name; give it a newline-separated list to pick an
exact set in an exact order.

Outputs both loop idioms, like JSON To Item List:
  item_list — one ITEM_LIST value for ▶Foreach List (sequential, accumulates)
  videos    — a ComfyUI list, so every downstream node runs once per clip
  count     — how many were found
"""

from __future__ import annotations

import os

from ...base import TiNode
from ...registry import register
from ._video_io import VIDEO_EXTS

# ComfyUI's native lazy file-backed VIDEO. Imported defensively so the module's
# pure helpers stay importable in the test runner (no comfy_api / av there).
try:
	from comfy_api.latest import InputImpl  # noqa: PLC0415

	VideoFromFile = InputImpl.VideoFromFile
except Exception:  # noqa: BLE001 — no ComfyUI available (tests, linting)
	VideoFromFile = None


def _input_dir():
	try:
		import folder_paths  # noqa: PLC0415

		return folder_paths.get_input_directory()
	except Exception:  # noqa: BLE001
		return os.getcwd()


def resolve_dir(directory, base_dir=None):
	"""Absolute folder to scan. Absolute `directory` is used as-is; a relative
	one (or empty) is taken under ComfyUI's input/ directory."""
	directory = str(directory or "").strip()
	if directory and os.path.isabs(directory):
		return directory
	base = base_dir if base_dir is not None else _input_dir()
	return os.path.join(base, directory) if directory else base


def resolve_video_files(directory, filenames="", reverse=False, base_dir=None):
	"""Ordered list of absolute video paths to load.

	`filenames` (one per line) picks an exact set in that order; empty scans the
	folder for every video-extension file, sorted by name. Pure and testable —
	pass `base_dir` to avoid needing folder_paths.
	"""
	root = resolve_dir(directory, base_dir)
	if not os.path.isdir(root):
		raise RuntimeError(f"Load Videos: folder not found: {root}")

	names = [ln.strip() for ln in str(filenames or "").splitlines()]
	names = [n for n in names if n]
	if names:
		paths, missing = [], []
		for n in names:
			p = n if os.path.isabs(n) else os.path.join(root, n)
			if os.path.isfile(p):
				paths.append(p)
			else:
				missing.append(n)
		if missing:
			raise RuntimeError(
				"Load Videos: these files were not found in "
				f"{root}: " + ", ".join(missing))
	else:
		paths = sorted(
			os.path.join(root, f) for f in os.listdir(root)
			if os.path.isfile(os.path.join(root, f))
			and f.lower().endswith(VIDEO_EXTS))
		if not paths:
			raise RuntimeError(f"Load Videos: no video files in {root}")

	if reverse:
		paths = list(reversed(paths))
	return paths


@register
class LoadVideos(TiNode):
	DISPLAY_NAME = "Load Videos (ti)"
	CATEGORY = "tinode/video"

	@classmethod
	def INPUT_TYPES(cls):
		return {
			"required": {
				"directory": ("STRING", {"default": "", "tooltip":
					"Folder to load from. Relative = under ComfyUI's input/ "
					"folder; absolute = that exact path (point straight at a "
					"source folder without copying it into input/). Empty = "
					"input/ itself."}),
			},
			"optional": {
				"filenames": ("STRING", {"default": "", "multiline": True,
					"tooltip": "Optional: exact files to load, one per line, in "
					"order. Empty = every video in the folder, sorted by name."}),
				"reverse": ("BOOLEAN", {"default": False,
					"tooltip": "Reverse the order the clips are loaded in."}),
			},
		}

	RETURN_TYPES = ("ITEM_LIST", "VIDEO", "INT")
	RETURN_NAMES = ("item_list", "videos", "count")
	# Slot 1 only: `videos` is a ComfyUI list (downstream runs once per clip),
	# while item_list is one ITEM_LIST value and count a plain int.
	OUTPUT_IS_LIST = (False, True, False)
	OUTPUT_TOOLTIPS = (
		"Connect to Inspire's ▶Foreach List `item_list` to iterate the clips "
		"one at a time (only that clip is decoded — low peak RAM).",
		"The same clips as a ComfyUI list — every downstream node runs once per "
		"clip.",
		"How many videos were found.",
	)

	DESCRIPTION = (
		"Load several video files from a folder as native VIDEOs, for looping "
		"over them one at a time.\n"
		"item_list feeds Inspire's ▶Foreach List (sequential — pair with Video "
		"Concatenate to accumulate); videos is a ComfyUI list that makes "
		"downstream nodes run once per clip.\n"
		"Each clip is lazy (file-backed), so a folder of multi-GB videos costs "
		"almost nothing until an iteration actually decodes one."
	)

	FUNCTION = "execute"

	@classmethod
	def IS_CHANGED(cls, directory="", filenames="", reverse=False):
		# Re-run when the set of files, their order, or their contents change.
		try:
			paths = resolve_video_files(directory, filenames, reverse)
		except Exception as exc:  # noqa: BLE001 — surface as a changed sig, node errors on run
			return repr(exc)
		return "|".join(f"{p}:{os.path.getmtime(p)}" for p in paths)

	def execute(self, directory="", filenames="", reverse=False):
		paths = resolve_video_files(directory, filenames, reverse)
		if VideoFromFile is None:
			raise RuntimeError(
				"Load Videos: ComfyUI's native VIDEO API (comfy_api.latest) is "
				"unavailable, so the clips cannot be loaded.")

		videos = [VideoFromFile(p) for p in paths]
		# Separate list identity for the ITEM_LIST: a loop slices it, so it must
		# not share the object handed to the per-item ComfyUI-list output.
		return (list(videos), videos, len(videos))
