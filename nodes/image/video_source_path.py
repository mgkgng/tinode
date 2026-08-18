"""Video Source Path — the file a native VIDEO was loaded from.

In a Foreach loop over Load Videos, the loop item is a VIDEO. To save an
artifact keyed to that clip (a mask, a manifest) you need its filename — this
pulls the source path out of the VIDEO and splits it into the pieces you key on:

  stem      DICAIRE T 01 pour test IA   (basename without extension — the key)
  filename  DICAIRE T 01 pour test IA.mp4
  path      /…/input/dicaire/DICAIRE T 01 pour test IA.mp4

Works on clips from Load Videos / Load Video (file-backed native VIDEOs). A
VIDEO built inside the graph (Create Video, a concatenation) has no file, so it
raises rather than inventing a name.
"""

from __future__ import annotations

import os

from ...base import TiNode
from ...registry import register


def video_source_path(video):
	"""Absolute file path a VIDEO was loaded from, or None if not file-backed."""
	if video is None:
		return None
	getter = getattr(video, "get_stream_source", None)
	if callable(getter):
		try:
			src = getter()
			if isinstance(src, str) and src:
				return src
		except Exception:  # noqa: BLE001 — fall through to attribute probing
			pass
	# VideoFromFile stores the path privately; other impls vary. Probe the common
	# names rather than depend on one private attribute.
	for attr in ("_VideoFromFile__file", "_file", "file", "path", "filename"):
		v = getattr(video, attr, None)
		if isinstance(v, str) and v:
			return v
	return None


@register
class VideoSourcePath(TiNode):
	DISPLAY_NAME = "Video Source Path (ti)"
	CATEGORY = "tinode/video"

	@classmethod
	def INPUT_TYPES(cls):
		return {"required": {"video": ("VIDEO", {"tooltip":
			"A file-backed VIDEO (from Load Videos / Load Video). Its source "
			"file's name is what keys the saved mask/manifest."})}}

	RETURN_TYPES = ("STRING", "STRING", "STRING")
	RETURN_NAMES = ("stem", "filename", "path")
	OUTPUT_TOOLTIPS = (
		"Basename without extension — the key to save/load a clip's mask by.",
		"Basename with extension.",
		"Full absolute path to the source file.",
	)
	FUNCTION = "execute"

	def execute(self, video):
		path = video_source_path(video)
		if not path:
			raise RuntimeError(
				"Video Source Path: this VIDEO isn't file-backed (no source path). "
				"It works on clips from Load Videos / Load Video; a VIDEO built in "
				"the graph (e.g. Create Video, Video Concatenate) has no file to name."
			)
		filename = os.path.basename(path)
		stem = os.path.splitext(filename)[0]
		return (stem, filename, path)
