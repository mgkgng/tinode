"""Save Video — encode an IMAGE batch to a video file, like VHS Video Combine.

Writes to ComfyUI's output (or temp) directory through ffmpeg and previews the
result in the node. The parameters that matter for keeping a grade intact are
exposed directly:

  pix_fmt       yuv420p (compatible) or yuv444p (no chroma subsampling).
  crf           quality; lower = better, 17 is near-visually-lossless for x264.
  color_range   tv (limited, 16-235) / pc (full). Match your SOURCE so the file
                is interpreted the way it was graded.
  colorspace    bt709 (HD default) — tags primaries/matrix/transfer together.

For a truly lossless intermediate, choose the "png (lossless frames)" format: it
writes a PNG per frame (no colour conversion, no compression) that you can mux
yourself later.

Not a 1:1 VHS clone: no audio muxing, no GIF/WebP palette handling, no batch
manager. It covers the encode path with the colour controls that fix the shift.
"""

from __future__ import annotations

import os

import torch

from ...base import TiNode
from ...registry import register
from ._video_io import encode

_FORMATS = ["h264 (mp4)", "vp9 (webm)", "png (lossless frames)"]


@register
class SaveVideo(TiNode):
	DISPLAY_NAME = "Save Video · Combine (ti)"
	CATEGORY = "tinode/video"
	OUTPUT_NODE = True

	@classmethod
	def INPUT_TYPES(cls):
		return {
			"required": {
				"images": ("IMAGE",),
				"frame_rate": ("FLOAT", {"default": 24.0, "min": 0.1, "max": 240.0, "step": 0.01}),
				"filename_prefix": ("STRING", {"default": "tinode/video"}),
				"format": (_FORMATS, {"default": "h264 (mp4)"}),
			},
			"optional": {
				"crf": ("INT", {"default": 17, "min": 0, "max": 51, "step": 1,
					"tooltip": "Quality: lower is better. 0 = lossless (x264). Ignored for png."}),
				"pix_fmt": (["yuv420p", "yuv444p"], {"default": "yuv420p"}),
				"color_range": (["tv", "pc", "unspecified"], {"default": "tv",
					"tooltip": "Match your SOURCE. tv=limited(16-235), pc=full(0-255)."}),
				"colorspace": (["bt709", "bt470bg", "smpte170m", "unspecified"], {"default": "bt709"}),
				"save_output": ("BOOLEAN", {"default": True,
					"tooltip": "On: save to output/. Off: temp/ (preview only)."}),
			},
		}

	RETURN_TYPES = ("STRING",)
	RETURN_NAMES = ("filepath",)
	FUNCTION = "execute"

	def execute(self, images, frame_rate=24.0, filename_prefix="tinode/video",
				format="h264 (mp4)", crf=17, pix_fmt="yuv420p", color_range="tv",
				colorspace="bt709", save_output=True):
		import folder_paths  # noqa: PLC0415

		imgs = images if images.dim() == 4 else images.unsqueeze(0)
		N, H, W, C = imgs.shape

		out_dir = (folder_paths.get_output_directory() if save_output
				   else folder_paths.get_temp_directory())
		full_dir, base, counter, subfolder, _ = folder_paths.get_save_image_path(
			filename_prefix, out_dir, W, H)
		ftype = "output" if save_output else "temp"

		if format.startswith("png"):
			filename, ui = self._save_png_sequence(imgs, full_dir, base, counter, subfolder, ftype)
			return {"ui": {"ti_video": [ui]}, "result": (os.path.join(full_dir, filename),)}

		codec = "libx264" if format.startswith("h264") else "libvpx-vp9"
		ext = "mp4" if codec == "libx264" else "webm"
		filename = f"{base}_{counter:05}.{ext}"
		path = os.path.join(full_dir, filename)
		encode(imgs, path, fps=float(frame_rate), codec=codec, crf=int(crf),
			   pix_fmt=pix_fmt, color_range=color_range, colorspace=colorspace)

		ui = {"filename": filename, "subfolder": subfolder, "type": ftype,
			  "format": "video/mp4" if ext == "mp4" else "video/webm",
			  "frame_rate": float(frame_rate)}
		return {"ui": {"ti_video": [ui]}, "result": (path,)}

	def _save_png_sequence(self, imgs, full_dir, base, counter, subfolder, ftype):
		"""Lossless per-frame PNGs — no colour conversion, no compression loss."""
		import numpy as np  # noqa: PLC0415
		from PIL import Image  # noqa: PLC0415

		folder = os.path.join(full_dir, f"{base}_{counter:05}")
		os.makedirs(folder, exist_ok=True)
		rel = os.path.join(subfolder, f"{base}_{counter:05}")
		for i in range(imgs.shape[0]):
			arr = (imgs[i, ..., :3].clamp(0, 1).cpu().numpy() * 255.0 + 0.5).astype(np.uint8)
			Image.fromarray(arr).save(os.path.join(folder, f"{i:05}.png"), compress_level=4)
		# preview the first frame
		return folder, {"filename": "00000.png", "subfolder": rel, "type": ftype,
						"format": "image/png"}
