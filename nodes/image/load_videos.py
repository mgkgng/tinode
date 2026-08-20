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
copying gigabytes into input/). Narrow what gets loaded three ways, in order of
precedence: `filenames` (one per line — exact names or wildcards), else
`pattern` (a single wildcard like `*.mp4` or `PROJECT_AMIR_*` over the folder),
else every video file sorted by name.

Outputs both loop idioms, like JSON To Item List:
  item_list — one ITEM_LIST value for ▶Foreach List (sequential, accumulates)
  videos    — a ComfyUI list, so every downstream node runs once per clip
  count     — how many were found
"""

from __future__ import annotations

import fnmatch
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
	"""Absolute folder to scan, resolved forgivingly.

	The base is ComfyUI's input/ directory. `directory` may be written any of the
	natural ways and the first that actually exists wins:

	  * empty, or a lone "/"                → input/ itself
	  * a relative path (`dicaire`)         → under input/
	  * a redundant `input/` or `/input/`   → the input/ is stripped (input/ is
	    prefix (`/input/dicaire`)              already the base), so it still lands
	                                           on input/dicaire
	  * a path under the ComfyUI root       → tried relative to the root too
	  * a real absolute path (`/home/me/…`) → used as-is

	A bare "/" is the input root, not the filesystem root: nobody loads videos
	from "/", and it is the natural thing to type for "the default folder". When
	nothing exists, the input-relative reading is returned so the error names the
	folder the user most likely meant.
	"""
	directory = str(directory or "").strip()
	base = base_dir if base_dir is not None else _input_dir()
	if not directory or directory.strip("/\\") == "":
		return base

	stripped = directory.replace("\\", "/").strip("/")     # no leading/trailing slash
	root = os.path.dirname(base.rstrip("/\\"))              # ComfyUI root (parent of input/)

	candidates = []
	if os.path.isabs(directory):
		candidates.append(directory)                       # honour a real absolute path first
	candidates.append(os.path.join(base, stripped))        # the documented case: under input/
	if stripped.lower().startswith("input/"):              # redundant input/ prefix -> drop it
		candidates.append(os.path.join(base, stripped[len("input/"):]))
	if root:
		candidates.append(os.path.join(root, stripped))    # e.g. literal "input/dicaire" from root

	for c in candidates:
		if os.path.isdir(c):
			return c
	# Nothing exists: prefer the input-relative reading in the error message,
	# stripping a redundant input/ prefix so it names the folder they meant.
	guess = stripped[len("input/"):] if stripped.lower().startswith("input/") else stripped
	return os.path.join(base, guess)


def _list_videos(root):
	"""Every video-extension file directly in `root`, sorted by name."""
	return sorted(f for f in os.listdir(root)
				  if os.path.isfile(os.path.join(root, f))
				  and f.lower().endswith(VIDEO_EXTS))


def _match_name(root, name):
	"""Resolve one requested filename to an existing path, or None.

	Tries the name as given, then — if it carries no video extension — the name
	with each known video extension, so "clip" finds "clip.mp4". This is what
	lets you list bare names without remembering each container.
	"""
	p = name if os.path.isabs(name) else os.path.join(root, name)
	if os.path.isfile(p):
		return p
	if not name.lower().endswith(VIDEO_EXTS):
		for ext in VIDEO_EXTS:
			if os.path.isfile(p + ext):
				return p + ext
	return None


_GLOB_CHARS = set("*?[")


def _is_glob(s):
	return any(c in s for c in _GLOB_CHARS)


def _match_glob(pattern, listing):
	"""Video files matching a shell glob, case-insensitively, in listing order.

	`listing` is already video-extension-filtered, so a bare stem like
	`PROJECT_AMIR_*` still only ever selects videos (never a stray sidecar
	`.txt`), while `*.mp4` narrows to that extension as you'd expect.
	"""
	pat = pattern.strip().lower()
	return [f for f in listing if fnmatch.fnmatch(f.lower(), pat)]


def resolve_video_files(directory, filenames="", pattern="", reverse=False, base_dir=None):
	"""Ordered list of absolute video paths to load.

	Selection precedence:
	  * `filenames` non-empty — one entry per line, each an exact name OR a glob
	    (`PROJECT_AMIR_*.mp4`, `clip` → `clip.mp4`); globs expand in folder order,
	    exact names keep their line order, all deduped. A line that matches
	    nothing raises.
	  * else `pattern` non-empty — a single glob over the folder (`*.mp4`,
	    `PROJECT_AMIR_*`).
	  * else — every video-extension file in the folder, sorted by name.

	Pure and testable — pass `base_dir` to avoid needing folder_paths.
	"""
	root = resolve_dir(directory, base_dir)
	if not os.path.isdir(root):
		raise RuntimeError(f"Load Videos: folder not found: {root}")

	def _folder_hint():
		have = _list_videos(root)
		if not have:
			return " — the folder has no video files."
		return (" — videos in the folder: " + ", ".join(have[:12])
				+ (" …" if len(have) > 12 else ""))

	names = [ln.strip() for ln in str(filenames or "").splitlines()]
	names = [n for n in names if n]

	if names:
		listing = _list_videos(root)
		paths, missing = [], []
		for n in names:
			if _is_glob(n):
				hits = _match_glob(n, listing)
				if not hits:
					missing.append(n)
				for f in hits:
					p = os.path.join(root, f)
					if p not in paths:
						paths.append(p)
			else:
				hit = _match_name(root, n)
				if hit is None:
					missing.append(n)
				elif hit not in paths:
					paths.append(hit)
		if missing:
			raise RuntimeError(
				f"Load Videos: no match / not found in {root}: "
				+ ", ".join(missing) + _folder_hint())
	elif str(pattern or "").strip():
		files = _match_glob(pattern, _list_videos(root))
		if not files:
			raise RuntimeError(
				f"Load Videos: nothing matches {pattern.strip()!r} in {root}"
				+ _folder_hint())
		paths = [os.path.join(root, f) for f in files]
	else:
		paths = [os.path.join(root, f) for f in _list_videos(root)]
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
					"Folder to load from. Empty (or just \"/\") = ComfyUI's "
					"input/ folder. A relative path is under input/ (dicaire → "
					"input/dicaire); a leading input/ is fine too (input/dicaire, "
					"/input/dicaire both work). A full absolute path (e.g. "
					"/home/me/clips) is used as-is, so you can point straight at a "
					"source folder without copying it into input/."}),
			},
			"optional": {
				"pattern": ("STRING", {"default": "", "tooltip":
					"Optional wildcard filter over the folder, e.g. *.mp4 or "
					"PROJECT_AMIR_* (case-insensitive). Empty = every video. "
					"Ignored when filenames is filled in."}),
				"filenames": ("STRING", {"default": "", "multiline": True,
					"tooltip": "Optional: files to load, one per line, in order. "
					"Each line is an exact name (\"clip\" finds \"clip.mp4\") or a "
					"wildcard (PROJECT_AMIR_*.mp4). Empty = use pattern / whole "
					"folder."}),
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
	def IS_CHANGED(cls, directory="", pattern="", filenames="", reverse=False):
		# Re-run when the set of files, their order, or their contents change.
		try:
			paths = resolve_video_files(directory, filenames, pattern, reverse)
		except Exception as exc:  # noqa: BLE001 — surface as a changed sig, node errors on run
			return repr(exc)
		return "|".join(f"{p}:{os.path.getmtime(p)}" for p in paths)

	def execute(self, directory="", pattern="", filenames="", reverse=False):
		paths = resolve_video_files(directory, filenames, pattern, reverse)
		if VideoFromFile is None:
			raise RuntimeError(
				"Load Videos: ComfyUI's native VIDEO API (comfy_api.latest) is "
				"unavailable, so the clips cannot be loaded.")

		videos = [VideoFromFile(p) for p in paths]
		# Separate list identity for the ITEM_LIST: a loop slices it, so it must
		# not share the object handed to the per-item ComfyUI-list output.
		return (list(videos), videos, len(videos))
