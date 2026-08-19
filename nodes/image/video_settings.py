"""Save / Load Video Settings — carry Save Video · Combine's encode format.

Set the output format once (in phase 1, next to the source) and reuse it in the
final save, so every clip is encoded identically to match the delivery spec.
Save Video Settings writes format / crf / pix_fmt / color_range / colorspace to a
small JSON under output/; Load Video Settings reads it back as the exact typed
values to wire into Save Video · Combine.
"""

from __future__ import annotations

import json
import os

from ...base import TiNode, first
from ...registry import register

_SUBDIR = "ti_video_settings"
_FORMATS = ["h264 (mp4)", "vp9 (webm)", "png (lossless frames)"]
_PIXFMT = ["yuv420p", "yuv444p"]
_RANGE = ["tv", "pc", "unspecified"]
_SPACE = ["bt709", "bt470bg", "smpte170m", "unspecified"]

_DEFAULTS = {
	"format": _FORMATS[0], "crf": 17, "pix_fmt": "yuv420p",
	"color_range": "tv", "colorspace": "bt709",
}


def _path(name):
	import folder_paths  # noqa: PLC0415

	root = os.path.join(folder_paths.get_output_directory(), _SUBDIR)
	os.makedirs(root, exist_ok=True)
	return os.path.join(root, f"{os.path.basename(str(name).strip()) or 'default'}.json")


@register
class SaveVideoSettings(TiNode):
	DISPLAY_NAME = "Save Video Settings (ti)"
	CATEGORY = "tinode/video"
	OUTPUT_NODE = True

	@classmethod
	def INPUT_TYPES(cls):
		return {
			"required": {
				"name": ("STRING", {"default": "default"}),
				"format": (_FORMATS, {"default": _FORMATS[0]}),
				"crf": ("INT", {"default": 17, "min": 0, "max": 51, "step": 1}),
				"pix_fmt": (_PIXFMT, {"default": "yuv420p"}),
				"color_range": (_RANGE, {"default": "tv"}),
				"colorspace": (_SPACE, {"default": "bt709"}),
			},
		}

	RETURN_TYPES = ("STRING",)
	RETURN_NAMES = ("path",)
	FUNCTION = "execute"

	def execute(self, name, format, crf, pix_fmt, color_range, colorspace):
		data = {"format": format, "crf": int(crf), "pix_fmt": pix_fmt,
				"color_range": color_range, "colorspace": colorspace}
		p = _path(name)
		with open(p, "w", encoding="utf-8") as fh:
			json.dump(data, fh, ensure_ascii=False, indent=2)
		print(f"[tinode] Save Video Settings: {data} -> {p}")
		return {"ui": {"ti_settings": [data]}, "result": (p,)}


@register
class LoadVideoSettings(TiNode):
	DISPLAY_NAME = "Load Video Settings (ti)"
	CATEGORY = "tinode/video"

	@classmethod
	def INPUT_TYPES(cls):
		return {"required": {"name": ("STRING", {"default": "default"})}}

	RETURN_TYPES = ("STRING", "INT", "STRING", "STRING", "STRING")
	RETURN_NAMES = ("format", "crf", "pix_fmt", "color_range", "colorspace")
	OUTPUT_TOOLTIPS = (
		"Wire these into Save Video · Combine's matching inputs.",
		"", "", "", "",
	)
	FUNCTION = "execute"

	@classmethod
	def IS_CHANGED(cls, name="default"):
		try:
			return str(os.path.getmtime(_path(name)))
		except Exception:  # noqa: BLE001
			return "missing"

	def execute(self, name):
		p = _path(name)
		d = dict(_DEFAULTS)
		if os.path.isfile(p):
			try:
				with open(p, "r", encoding="utf-8") as fh:
					d.update(json.load(fh))
			except Exception as exc:  # noqa: BLE001
				print(f"[tinode] Load Video Settings: bad file {p}: {exc!r} — using defaults")
		else:
			print(f"[tinode] Load Video Settings: {p} not found — using defaults")
		return (d["format"], int(d["crf"]), d["pix_fmt"], d["color_range"], d["colorspace"])
