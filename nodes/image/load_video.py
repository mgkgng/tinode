"""Load Video — decode a video file to an IMAGE batch, like VHS Load Video.

Reads a file from ComfyUI's input directory through ffmpeg and returns the
frames as an IMAGE tensor, plus the frame count and fps. Parameters mirror the
common VHS ones: frame_load_cap, skip_first_frames, select_every_nth,
force_rate, and an optional resize.

Colour: the default faithful decode lets ffmpeg convert using the file's own
tagged colour metadata. If a clip is MIS-tagged (its look changes vs a player),
`force_full_range` re-reads it treating the source as full-range, which undoes
the most common "washed out on load" case. See README > Video colour.

Besides the frames it also emits `stem` / `path` / `video`, the same handles
Load Videos gives, so a single clip can drive the per-clip store (Save Crop &
Mask keys on the stem, phase 2 reopens the recorded path) without swapping in
the multi-file loader.

Not a 1:1 VHS clone: no audio output, no batch manager. It covers the load path
itself.
"""

from __future__ import annotations

import os

import torch

from ...base import TiNode
from ...registry import register
from ._video_io import VIDEO_EXTS, decode, probe

# ComfyUI's native lazy file-backed VIDEO, so this node can hand on the source
# clip itself the way Load Videos does. Imported defensively (tests/headless).
try:
	from comfy_api.latest import InputImpl  # noqa: PLC0415

	VideoFromFile = InputImpl.VideoFromFile
except Exception:  # noqa: BLE001
	VideoFromFile = None


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
				# video_upload adds ComfyUI's built-in "choose file to upload"
				# button next to the combo (same as the core image loader).
				"video": (files, {"video_upload": True,
					"tooltip": "Pick a file from input/, or upload one with the button."}),
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

	# stem / path / video are APPENDED, never inserted: a saved workflow stores
	# links by slot INDEX, so adding outputs at the end keeps old graphs wired.
	RETURN_TYPES = ("IMAGE", "INT", "FLOAT", "STRING", "STRING", "VIDEO")
	RETURN_NAMES = ("images", "frame_count", "fps", "stem", "path", "video")
	OUTPUT_TOOLTIPS = (
		"The decoded frames.",
		"How many frames were loaded.",
		"Frames per second.",
		"Filename without extension — the key Save Crop & Mask stores a clip by.",
		"Absolute path to the source file, recorded so a later pass can reopen it.",
		"The source clip as a native VIDEO — the WHOLE untouched file, ignoring "
		"the frame-selection and resize options above.",
	)
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

		stem = os.path.splitext(os.path.basename(path))[0]
		# The VIDEO is the file as-is. The IMAGE output may have been capped,
		# strided or resized, so the two only match at default settings — say so
		# rather than silently handing on a clip that disagrees with the frames.
		native = VideoFromFile(path) if VideoFromFile is not None else None
		if native is not None and (frame_load_cap or skip_first_frames
								   or int(select_every_nth) > 1 or force_rate):
			print("[tinode] Load Video: frame selection is active, so `video` (the "
				  "whole source file) does not match `images`.")
		return (imgs, imgs.shape[0], fps, stem, path, native)
