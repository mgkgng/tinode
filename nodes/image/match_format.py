"""Match Source Format — read a source clip's format, emit the encode settings.

So the delivered file matches the footage it came from without you re-typing
(or mis-typing) codec / pixel format / colour tags per shot. Point it at a
source video and wire its outputs straight into Save Video · Combine.

Quality: h264 is a lossy codec, so "same format as the source" and "no quality
loss" only meet at **crf 0**, x264's mathematically lossless mode — that is the
default here. The one unavoidable step is the YUV<->RGB round trip: ComfyUI
works in RGB, so a 4:2:0 source is chroma-upsampled on decode and re-subsampled
on encode. That touches chroma only, and matching the source's own pix_fmt keeps
it to a single generation. Choose `visually lossless (crf 17)` for a much
smaller file if a one-generation perceptual loss is acceptable, or the lossless
PNG frames mode when the result is an intermediate you will grade further.

The probe reads the container, not the pixels, so it costs nothing.
"""

from __future__ import annotations

import os

from ...base import TiNode, first
from ...registry import register
from .validation_gate import ANY
from ._video_io import probe
from .video_source_path import video_source_path

_QUALITY = ["lossless (crf 0)", "visually lossless (crf 17)", "high (crf 20)", "custom"]


def format_for(codec):
	"""Our Save Video format option that matches the source codec."""
	c = (codec or "").lower()
	if "vp9" in c or "vp09" in c:
		return "vp9 (webm)"
	return "h264 (mp4)"                      # h264/h265/anything else -> h264


def pix_fmt_for(pix_fmt):
	"""Keep the source's chroma subsampling: only 4:4:4 sources stay 4:4:4."""
	return "yuv444p" if "444" in (pix_fmt or "") else "yuv420p"


def range_for(color_range):
	r = (color_range or "").lower()
	if r in ("pc", "full", "jpeg"):
		return "pc"
	if r in ("tv", "limited", "mpeg"):
		return "tv"
	return "tv"                              # broadcast default when untagged


def colorspace_for(space):
	s = (space or "").lower()
	for known in ("bt709", "bt470bg", "smpte170m"):
		if known in s:
			return known
	return "bt709"                           # HD default when untagged


@register
class MatchSourceFormat(TiNode):
	DISPLAY_NAME = "Match Source Format (ti)"
	CATEGORY = "tinode/video"

	@classmethod
	def INPUT_TYPES(cls):
		return {
			"required": {
				"quality": (_QUALITY, {"default": _QUALITY[0], "tooltip":
					"crf 0 = mathematically lossless (big files). 17 is visually "
					"lossless but a real lossy generation."}),
			},
			"optional": {
				"video": ("VIDEO", {"tooltip":
					"A file-backed source clip — its format is copied."}),
				"source_path": ("STRING", {"default": "", "tooltip":
					"Or the path directly (e.g. Load Mask's item source). Used when "
					"`video` isn't connected."}),
				"custom_crf": ("INT", {"default": 0, "min": 0, "max": 51, "step": 1,
					"tooltip": "Used only when quality is `custom`."}),
				"force_png": ("BOOLEAN", {"default": False, "tooltip":
					"Ignore the source codec and emit lossless PNG frames instead — "
					"for an intermediate you will grade or re-encode later."}),
			},
		}

	# format / pix_fmt / color_range / colorspace feed COMBO widget inputs on Save
	# Video, and the frontend refuses STRING -> COMBO. They are declared wildcard
	# so the link is accepted; the values are still exactly the combo's options.
	RETURN_TYPES = (ANY, "INT", ANY, ANY, ANY, "FLOAT", "INT", "INT", "STRING")
	RETURN_NAMES = ("format", "crf", "pix_fmt", "color_range", "colorspace",
					"fps", "width", "height", "report")
	OUTPUT_TOOLTIPS = (
		"Wire into Save Video · Combine's `format`.",
		"…its `crf`.", "…its `pix_fmt`.", "…its `color_range`.", "…its `colorspace`.",
		"…its `frame_rate` (the source's own fps).",
		"Source width.", "Source height.",
		"Human-readable summary of what was detected.",
	)
	FUNCTION = "execute"

	def execute(self, quality=_QUALITY[0], video=None, source_path="",
				custom_crf=0, force_png=False):
		path = str(first(source_path, "")).strip()
		vid = first(video)
		if vid is not None and not path:
			path = video_source_path(vid) or ""
		if not path or not os.path.isfile(path):
			raise RuntimeError(
				"Match Source Format: no source file to read. Connect a file-backed "
				f"`video`, or give `source_path` (got {path!r}).")

		info = probe(path)
		fmt = ("png (lossless frames)" if bool(first(force_png, False))
			   else format_for(_codec_of(path, info)))
		pix = pix_fmt_for(info.get("pix_fmt"))
		rng = range_for(info.get("color_range"))
		spc = colorspace_for(info.get("color_space") or info.get("color_primaries"))
		fps = float(info.get("fps") or 0.0)
		w, h = int(info.get("width") or 0), int(info.get("height") or 0)

		q = str(first(quality, _QUALITY[0]))
		crf = {"lossless (crf 0)": 0, "visually lossless (crf 17)": 17, "high (crf 20)": 20}.get(
			q, int(first(custom_crf, 0)))

		report = (f"{os.path.basename(path)}: {w}x{h} @ {fps:g}fps, "
				  f"{info.get('pix_fmt') or '?'} {rng}/{spc} -> {fmt}, crf {crf}, {pix}")
		print(f"[tinode] Match Source Format: {report}")
		return (fmt, crf, pix, rng, spc, fps, w, h, report)


def _codec_of(path, info):
	"""Codec name for the format choice; probe() doesn't return it, so ask ffprobe."""
	import subprocess  # noqa: PLC0415

	from ._video_io import ffprobe_exe  # noqa: PLC0415

	fp = ffprobe_exe()
	if not fp:
		return "h264"
	try:
		return subprocess.run(
			[fp, "-v", "error", "-select_streams", "v:0", "-show_entries",
			 "stream=codec_name", "-of", "csv=p=0", path],
			capture_output=True, text=True, check=True).stdout.strip() or "h264"
	except Exception:  # noqa: BLE001
		return "h264"
