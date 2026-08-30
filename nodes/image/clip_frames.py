"""Load Clip Frames — the source side of phase 2b when the master is a PNG
sequence rather than a video file.

Some shots never became a video: the master is a numbered image sequence
(AMIR/PNGs/sh0090/sh0090.0000.png, .0002.png, …). Load Clip Fills hands back a
VIDEO for the composite to paste onto, and there is no file to hand back — so
this node takes the clip item and reads the frames straight off disk instead.

It replaces `Load Clip Fills.video -> Get Video Components -> Frame Stride` with
one node: wire `source_stride` from Load Clip Fills into `every_nth_in` and the
frames come back already reduced to the rate the fills were rendered at, with
`fps_out` divided to match. Turn Load Clip Fills' `require_source` off so it
stops asking for a video that does not exist.

Frames are read in FILENAME order, not by parsing the numbers out of them: the
AMIR sequences are the halved 25fps material, so their numbers run 0, 2, 4, …
and index N of the sequence is simply the Nth file. That is the same stream the
store's frame_start/frame_count count in.

Lossless: 16-bit PNGs stay 16-bit through the float conversion (/65535), 8-bit
divide by 255, and nothing is resampled.
"""

from __future__ import annotations

import glob
import os

from ...base import TiNode, first
from ...registry import register
from .load_masks import _abs_source_dir

# 4K RGB float32 is ~100 MB per frame; a long sequence is the single biggest
# allocation in the pipeline, so say so rather than let the machine find out.
_BYTES_PER_PIXEL = 12


def find_frame_dir(root, stem):
	"""`root/stem` when it holds the frames, else `root` itself, else None."""
	if not root:
		return None
	cand = os.path.join(root, str(stem))
	if os.path.isdir(cand):
		return cand
	return root if os.path.isdir(root) else None


def list_frames(dirpath, pattern="*.png"):
	"""Every matching file in the directory, in filename order."""
	if not dirpath or not os.path.isdir(dirpath):
		return []
	return sorted(glob.glob(os.path.join(dirpath, pattern)))


def load_image_files(paths):
	"""[N,H,W,3] float tensor (0..1) from a list of image files, bit depth kept."""
	import numpy as np  # noqa: PLC0415
	import torch  # noqa: PLC0415

	frames = []
	for p in paths:
		try:
			import cv2  # noqa: PLC0415

			raw = cv2.imread(p, cv2.IMREAD_UNCHANGED)
			if raw is None:
				raise RuntimeError(f"Load Clip Frames: could not read {p}")
			if raw.ndim == 2:
				raw = cv2.cvtColor(raw, cv2.COLOR_GRAY2RGB)
			elif raw.shape[2] == 4:
				raw = cv2.cvtColor(raw, cv2.COLOR_BGRA2RGB)
			else:
				raw = cv2.cvtColor(raw, cv2.COLOR_BGR2RGB)
			a = raw.astype(np.float32) / (65535.0 if raw.dtype == np.uint16 else 255.0)
		except ImportError:
			from PIL import Image  # noqa: PLC0415

			a = np.asarray(Image.open(p).convert("RGB"), dtype=np.float32) / 255.0
		if frames and a.shape != frames[0].shape:
			raise RuntimeError(
				f"Load Clip Frames: {os.path.basename(p)} is {a.shape[1]}x{a.shape[0]} "
				f"but the sequence started {frames[0].shape[1]}x{frames[0].shape[0]}. "
				"Every frame of a clip must be the same size.")
		frames.append(a)
	return torch.from_numpy(np.stack(frames))


@register
class LoadClipFrames(TiNode):
	DISPLAY_NAME = "Load Clip Frames (ti)"
	CATEGORY = "tinode/video"

	@classmethod
	def INPUT_TYPES(cls):
		return {
			"required": {
				"item": ("TI_CLIP_ITEM", {"tooltip":
					"The clip item from Load Clips — same one Load Clip Fills gets."}),
				"frames_dir": ("STRING", {"default": "AMIR/PNGs", "tooltip":
					"Where the image sequences live. Relative paths resolve under "
					"ComfyUI's input/. The clip's own folder is <frames_dir>/<stem> "
					"if that exists, otherwise frames_dir itself."}),
			},
			"optional": {
				"pattern": ("STRING", {"default": "*.png", "tooltip":
					"Which files to take, in filename order."}),
				"every_nth": ("INT", {"default": 1, "min": 1, "max": 10, "step": 1,
					"tooltip": "Keep frames 0, n, 2n, … Leave 1 and drive it from "
							   "every_nth_in instead."}),
				"every_nth_in": ("INT", {"default": 0, "min": 0, "max": 100, "step": 1,
					"forceInput": True, "tooltip":
					"Stride from Load Clip Fills' source_stride. Connected and > 0, "
					"it overrides the widget — one knob, and the frames can never "
					"come back at a different rate than the fills."}),
				"fps": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 240.0, "step": 0.01,
					"tooltip": "0 = the rate recorded in the clip's manifests."}),
			},
		}

	RETURN_TYPES = ("IMAGE", "FLOAT", "INT", "STRING")
	RETURN_NAMES = ("images", "fps_out", "frame_count", "frames_dir")
	OUTPUT_TOOLTIPS = (
		"The source frames, untouched — wire to Composite Crops' image.",
		"fps / every_nth — wire to the final save's frame_rate.",
		"How many frames were loaded.",
		"The folder they came from, for the record.",
	)
	FUNCTION = "execute"

	@classmethod
	def IS_CHANGED(cls, item=None, frames_dir="", pattern="*.png", **_kw):
		root = find_frame_dir(_abs_source_dir(frames_dir), "")
		return f"{root}:{pattern}:{len(list_frames(root, pattern))}"

	def execute(self, item, frames_dir="AMIR/PNGs", pattern="*.png", every_nth=1,
				every_nth_in=0, fps=0.0):
		clip = first(item)
		if not isinstance(clip, dict) or "crops" not in clip:
			raise RuntimeError("Load Clip Frames: `item` must come from Load Clips.")
		stem = clip.get("stem", "")
		root = _abs_source_dir(first(frames_dir, ""))
		pattern = str(first(pattern, "*.png") or "*.png")
		folder = find_frame_dir(root, stem)
		files = list_frames(folder, pattern)
		if not files:
			raise RuntimeError(
				f"Load Clip Frames: no {pattern} under {folder or root!r} for "
				f"{stem!r}. Point frames_dir at the folder holding the sequences "
				f"(the clip's own is <frames_dir>/{stem}).")

		override = int(first(every_nth_in, 0) or 0)
		n = override if override > 0 else max(1, int(first(every_nth, 1)))
		kept = files[::n]

		rate = float(first(fps, 0.0)) or float(clip.get("fps", 0.0) or 0.0)
		mb = len(kept) * _BYTES_PER_PIXEL / 1e6
		print(f"[tinode] Load Clip Frames: {stem!r} {len(files)} file(s) in {folder} "
			  f"-> keeping {len(kept)} (every {n}) @ {rate / n if rate else 0:.2f} fps"
			  + (f" — about {mb * 8.3:.1f} GB at 4K" if len(kept) > 50 else ""))
		images = load_image_files(kept)
		return (images, (rate / n) if rate else 0.0, int(images.shape[0]), folder)
