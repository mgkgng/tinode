"""Load Video — decode a video file to an IMAGE batch, like VHS Load Video.

Reads a file from ComfyUI's input directory through ffmpeg and returns the
frames as an IMAGE tensor, plus the frame count and fps. Parameters mirror the
common VHS ones: frame_load_cap, skip_first_frames, select_every_nth,
force_rate, and an optional resize.

Colour: the default faithful decode lets ffmpeg convert using the file's own
tagged colour metadata. If a clip is MIS-tagged (its look changes vs a player),
`force_full_range` re-reads it treating the source as full-range, which undoes
the most common "washed out on load" case. See README > Video colour.

Not a 1:1 VHS clone: no audio output, no in-browser upload (drop files in the
input folder), no batch manager. It covers the load path itself.
"""

from __future__ import annotations

import os

import torch

from ...base import TiNode
from ...registry import register
from ._video_io import VIDEO_EXTS, decode, probe


def _input_videos():
	try:
		import folder_paths  # noqa: PLC0415
		d = folder_paths.get_input_directory()
		return sorted(f for f in os.listdir(d)
					  if os.path.isfile(os.path.join(d, f)) and f.lower().endswith(VIDEO_EXTS))
	except Exception:
		return []


@register
class LoadVideo(TiNode):
	DISPLAY_NAME = "Load Video (ti)"
	CATEGORY = "tinode/video"

	@classmethod
	def INPUT_TYPES(cls):
		files = _input_videos() or [""]
		return {
			"required": {
				"video": (files, {"tooltip": "A file in ComfyUI's input/ folder."}),
			},
			"optional": {
				"force_rate": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 240.0, "step": 0.01,
					"tooltip": "Resample to this fps before selecting frames. 0 = keep source."}),
				"frame_load_cap": ("INT", {"default": 0, "min": 0, "max": 100000, "step": 1,
					"tooltip": "Max frames to load. 0 = all."}),
				"skip_first_frames": ("INT", {"default": 0, "min": 0, "max": 100000, "step": 1}),
				"select_every_nth": ("INT", {"default": 1, "min": 1, "max": 1000, "step": 1}),
				"custom_width": ("INT", {"default": 0, "min": 0, "max": 8192, "step": 8,
					"tooltip": "Resize width. 0 with custom_height=0 keeps source size."}),
				"custom_height": ("INT", {"default": 0, "min": 0, "max": 8192, "step": 8}),
				"tonemap_hdr": (["auto", "on", "off"], {"default": "auto",
					"tooltip": "Tone-map HDR (BT.2020/PQ or HLG) sources to SDR. "
							   "auto = only when the source is HDR. Fixes the flat, "
							   "washed-out look of HDR footage decoded as SDR."}),
				"force_full_range": ("BOOLEAN", {"default": False,
					"tooltip": "Decode as full-range (for a clip mis-tagged as "
							   "limited). Not for HDR — use tonemap_hdr for that."}),
			},
		}

	RETURN_TYPES = ("IMAGE", "INT", "FLOAT")
	RETURN_NAMES = ("images", "frame_count", "fps")
	FUNCTION = "execute"

	@classmethod
	def IS_CHANGED(cls, video, **kwargs):
		# Re-run when the chosen file's contents change.
		try:
			import folder_paths  # noqa: PLC0415
			p = os.path.join(folder_paths.get_input_directory(), video)
			return f"{video}:{os.path.getmtime(p)}"
		except Exception:
			return video

	def execute(self, video, force_rate=0.0, frame_load_cap=0, skip_first_frames=0,
				select_every_nth=1, custom_width=0, custom_height=0,
				tonemap_hdr="auto", force_full_range=False):
		import folder_paths  # noqa: PLC0415
		if not video:
			raise RuntimeError("No video selected — put a file in ComfyUI's input/ folder.")
		path = os.path.join(folder_paths.get_input_directory(), video)
		if not os.path.isfile(path):
			raise RuntimeError(f"Video not found: {path}")

		# custom size needs BOTH dims; fall back to source if only one is given.
		info0 = probe(path)
		w = int(custom_width) or (int(custom_height) and info0["width"])
		h = int(custom_height) or (int(custom_width) and info0["height"])

		imgs, info = decode(
			path, force_rate=float(force_rate), skip_first=int(skip_first_frames),
			every_nth=int(select_every_nth), cap=int(frame_load_cap),
			width=int(w or 0), height=int(h or 0), full_range=bool(force_full_range),
			tonemap=str(tonemap_hdr),
		)
		fps = float(force_rate) if force_rate and force_rate > 0 else (info["fps"] or 0.0)
		return (imgs, imgs.shape[0], fps)
