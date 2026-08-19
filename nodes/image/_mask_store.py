"""Shared on-disk layout for the mask handoff between the two removal passes.

Save Crop & Mask (phase 1) writes it, Load Masks (phase 2) reads it. Keeping the
layout in one place means the two nodes can never drift out of agreement.

    <output>/<subdir>/<stem>/
        manifest.json          # crop_info + frame count + source path + fps
        mask/00000.png ...      # lossless 8-bit, crop-space, one per frame

The mask is stored in CROP space (small) alongside the crop_info that maps it
back to the original frame; the source video itself is never copied or
re-encoded, so nothing here is lossy.
"""

from __future__ import annotations

import json
import os

MANIFEST_NAME = "manifest.json"
MASK_SUBFOLDER = "mask"
CROP_SUBFOLDER = "crop"
MASK_PATTERN = "%05d.png"
CROP_PATTERN = "%05d.png"
MANIFEST_VERSION = 1
DEFAULT_SUBDIR = "ti_masks"


def output_root(subdir=DEFAULT_SUBDIR):
	"""The mask store under ComfyUI's output directory."""
	import folder_paths  # noqa: PLC0415

	return os.path.join(folder_paths.get_output_directory(), subdir or DEFAULT_SUBDIR)


def clip_dir(root, stem):
	# basename guards against a stem that contains path separators (traversal).
	return os.path.join(root, os.path.basename(str(stem).strip()))


def manifest_path(cdir):
	return os.path.join(cdir, MANIFEST_NAME)


def mask_dir(cdir):
	return os.path.join(cdir, MASK_SUBFOLDER)


def crop_dir(cdir):
	return os.path.join(cdir, CROP_SUBFOLDER)


def save_rgb_sequence(imgs, out_dir, pattern, bit_depth):
	"""Write an [N,H,W,3] float IMAGE (0..1) as a lossless PNG sequence.

	bit_depth 16 (default) is exact for up to 16-bit sources — an 8-bit PNG would
	silently truncate a 10-bit graded master. 16-bit RGB PNG needs cv2; 8-bit
	goes through PIL. Round-to-nearest, no dithering. Returns the frame count.
	"""
	import numpy as np  # noqa: PLC0415
	import os as _os  # noqa: PLC0415

	_os.makedirs(out_dir, exist_ok=True)
	x = imgs[..., :3].detach().clamp(0, 1).cpu().numpy()
	n = int(x.shape[0])
	if int(bit_depth) == 16:
		try:
			import cv2  # noqa: PLC0415
		except Exception as exc:  # noqa: BLE001
			raise RuntimeError(
				"Save Crop & Mask: 16-bit crop export needs OpenCV (cv2), which isn't "
				"available. Set crop_bit_depth to 8, or install opencv-python."
			) from exc
		arr = (x * 65535.0 + 0.5).astype(np.uint16)
		for i in range(n):
			cv2.imwrite(_os.path.join(out_dir, pattern % i),
						cv2.cvtColor(arr[i], cv2.COLOR_RGB2BGR))
	else:
		from PIL import Image  # noqa: PLC0415

		arr = (x * 255.0 + 0.5).astype(np.uint8)
		for i in range(n):
			Image.fromarray(arr[i], mode="RGB").save(
				_os.path.join(out_dir, pattern % i), compress_level=6)
	return n


def load_rgb_sequence(in_dir, pattern, frame_count):
	"""Load a PNG sequence back to an [N,H,W,3] float tensor (0..1), 8- or 16-bit."""
	import numpy as np  # noqa: PLC0415
	import os as _os  # noqa: PLC0415

	import torch  # noqa: PLC0415

	frames = []
	for i in range(int(frame_count)):
		p = _os.path.join(in_dir, pattern % i)
		if not _os.path.isfile(p):
			raise RuntimeError(f"Load Cropped Frames: missing frame {p}")
		try:
			import cv2  # noqa: PLC0415

			bgr = cv2.imread(p, cv2.IMREAD_UNCHANGED)
			if bgr is None:
				raise RuntimeError(f"could not read {p}")
			rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
			a = rgb.astype(np.float32) / (65535.0 if rgb.dtype == np.uint16 else 255.0)
		except ImportError:
			from PIL import Image  # noqa: PLC0415

			a = np.asarray(Image.open(p).convert("RGB"), dtype=np.float32) / 255.0
		frames.append(a)
	return torch.from_numpy(np.stack(frames))


def write_manifest(cdir, manifest):
	with open(manifest_path(cdir), "w", encoding="utf-8") as fh:
		json.dump(manifest, fh, ensure_ascii=False, indent=2)


def read_manifest(cdir):
	with open(manifest_path(cdir), "r", encoding="utf-8") as fh:
		return json.load(fh)
