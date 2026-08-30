"""Shared on-disk layout for the crop/mask handoff across the removal passes.

One folder PER CROP (a clip has several), so multi-crop removal is just more
folders. Phase 1 (Save Crop & Mask) writes mask/ + crop/ + manifest; phase 2a
(Save Filled) adds filled/; phase 2b reads them back to composite. Keeping the
layout here means every node agrees on it.

    <output>/<subdir>/<stem>/<crop_index:02d>/
        manifest.json          # crop_info, frame count, source, fps, prompt
        mask/00000.png ...      # lossless 8-bit, crop space, the region to remove
        crop/00000.png ...      # lossless RGB crop (8/16-bit), what VOID edits
        filled/00000.png ...    # VOID's result, written in phase 2a

The source video is never copied or re-encoded — only crops/masks are stored.
"""

from __future__ import annotations

import json
import os

MANIFEST_NAME = "manifest.json"
MASK_SUBFOLDER = "mask"
CROP_SUBFOLDER = "crop"
FILLED_SUBFOLDER = "filled"
MASK_PATTERN = "%05d.png"
CROP_PATTERN = "%05d.png"
FILLED_PATTERN = "%05d.png"
MANIFEST_VERSION = 2
DEFAULT_SUBDIR = "ti_masks"


def output_root(subdir=DEFAULT_SUBDIR):
	"""The mask store under ComfyUI's output directory."""
	import folder_paths  # noqa: PLC0415

	return os.path.join(folder_paths.get_output_directory(), subdir or DEFAULT_SUBDIR)


def _safe(part):
	# guard a stem/index that might contain path separators (traversal)
	return os.path.basename(str(part).strip())


def clip_dir(root, stem):
	"""The folder holding all of one clip's crops."""
	return os.path.join(root, _safe(stem))


def item_dir(root, stem, crop_index, chunk_index=None):
	"""The folder for one crop of one chunk of one clip.

	A clip is split into chunks and each chunk into crops, so the crop index
	alone is NOT unique: chunk 0 crop 0 and chunk 1 crop 0 would land in the same
	folder and the second would overwrite the first. The chunk is therefore part
	of the folder name whenever there is one (`c00_k01`), and omitted only when a
	clip was never chunked, which keeps older single-level stores readable.
	"""
	stem = _safe(stem)
	if chunk_index is None:
		return os.path.join(root, stem, f"{int(crop_index):02d}")
	return os.path.join(root, stem, f"c{int(chunk_index):02d}_k{int(crop_index):02d}")


def manifest_path(idir):
	return os.path.join(idir, MANIFEST_NAME)


def mask_dir(idir):
	return os.path.join(idir, MASK_SUBFOLDER)


def crop_dir(idir):
	return os.path.join(idir, CROP_SUBFOLDER)


def filled_dir(idir, variant=""):
	"""Where a removal result goes. A variant keeps alternatives side by side —
	VOID emits a pass-1 and a pass-2 image and you want to compare them, so they
	cannot share one folder."""
	v = str(variant or "").strip()
	return os.path.join(idir, FILLED_SUBFOLDER if not v else f"{FILLED_SUBFOLDER}_{_safe(v)}")


def filled_flag(variant=""):
	"""Manifest key recording that this variant has been rendered."""
	v = str(variant or "").strip()
	return "has_filled" if not v else f"has_filled_{_safe(v)}"


def filled_frames_key(variant=""):
	"""Manifest key recording how many frames that variant's fill actually has.

	The fill can be shorter than the crop (half-rate mode renders every 2nd
	frame), so its length must be recorded rather than assumed from frame_count.
	"""
	v = str(variant or "").strip()
	return "filled_frames" if not v else f"filled_frames_{_safe(v)}"


def stride_indices(frame_start, frame_count, every_nth):
	"""Local frame indices to keep so the kept frames sit on the GLOBAL grid.

	Half-rate work must keep source-global indices divisible by every_nth —
	per-chunk striding from each chunk's own 0 would drift off the grid whenever
	a chunk starts on an odd frame, and the fills would land between the frames
	the composite decodes. Global index of local i is frame_start + i.
	"""
	s = int(frame_start)
	n = max(1, int(every_nth))
	return [i for i in range(int(frame_count)) if (s + i) % n == 0]


def write_manifest(idir, manifest):
	os.makedirs(idir, exist_ok=True)
	with open(manifest_path(idir), "w", encoding="utf-8") as fh:
		json.dump(manifest, fh, ensure_ascii=False, indent=2)


def read_manifest(idir):
	with open(manifest_path(idir), "r", encoding="utf-8") as fh:
		return json.load(fh)


def prune_stale(folder, pattern, keep):
	"""Delete frames >= `keep` left by a previous, longer save of this crop."""
	i = keep
	while True:
		stale = os.path.join(folder, pattern % i)
		if not os.path.exists(stale):
			break
		os.remove(stale)
		i += 1


def save_rgb_sequence(imgs, out_dir, pattern, bit_depth):
	"""Write an [N,H,W,3] float IMAGE (0..1) as a lossless PNG sequence.

	bit_depth 16 (default) is exact for up to 16-bit sources — an 8-bit PNG would
	silently truncate a 10-bit graded master. 16-bit RGB PNG needs cv2; 8-bit
	goes through PIL. Round-to-nearest, no dithering. Returns the frame count.
	"""
	import numpy as np  # noqa: PLC0415

	os.makedirs(out_dir, exist_ok=True)
	x = imgs[..., :3].detach().clamp(0, 1).cpu().numpy()
	n = int(x.shape[0])
	if int(bit_depth) == 16:
		try:
			import cv2  # noqa: PLC0415
		except Exception as exc:  # noqa: BLE001
			raise RuntimeError(
				"16-bit export needs OpenCV (cv2). Set bit depth to 8, or install "
				"opencv-python."
			) from exc
		arr = (x * 65535.0 + 0.5).astype(np.uint16)
		for i in range(n):
			cv2.imwrite(os.path.join(out_dir, pattern % i),
						cv2.cvtColor(arr[i], cv2.COLOR_RGB2BGR))
	else:
		from PIL import Image  # noqa: PLC0415

		arr = (x * 255.0 + 0.5).astype(np.uint8)
		for i in range(n):
			Image.fromarray(arr[i], mode="RGB").save(
				os.path.join(out_dir, pattern % i), compress_level=6)
	return n


def load_rgb_sequence(in_dir, pattern, frame_count, indices=None):
	"""Load a PNG sequence back to an [N,H,W,3] float tensor (0..1), 8- or 16-bit.

	`indices` loads only those frame numbers, in that order — the half-rate path
	— instead of 0..frame_count-1.
	"""
	import numpy as np  # noqa: PLC0415
	import torch  # noqa: PLC0415

	frames = []
	for i in (indices if indices is not None else range(int(frame_count))):
		p = os.path.join(in_dir, pattern % i)
		if not os.path.isfile(p):
			raise RuntimeError(f"missing frame {p}")
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


def load_mask_sequence(in_dir, pattern, frame_count, indices=None):
	"""Load a mask PNG sequence into a [frames, H, W] float tensor (0..1).

	`indices` loads only those frame numbers, in that order (half-rate path).
	"""
	import numpy as np  # noqa: PLC0415
	from PIL import Image  # noqa: PLC0415
	import torch  # noqa: PLC0415

	frames = []
	for i in (indices if indices is not None else range(int(frame_count))):
		p = os.path.join(in_dir, pattern % i)
		if not os.path.isfile(p):
			raise RuntimeError(f"missing mask frame {p}")
		frames.append(np.asarray(Image.open(p).convert("L"), dtype=np.uint8))
	return torch.from_numpy(np.stack(frames)).float() / 255.0


def scan_items(root):
	"""Every saved crop under root, as augmented-manifest dicts, sorted.

	One entry per <stem>/<NN>/ that has a manifest. Each carries absolute
	`item_dir` / `mask_dir` / `crop_dir` / `filled_dir` plus its manifest fields.
	Pure/testable — needs no comfy_api.
	"""
	if not os.path.isdir(root):
		raise RuntimeError(f"store not found: {root}")
	items = []
	for stem in sorted(os.listdir(root)):
		sdir = os.path.join(root, stem)
		if not os.path.isdir(sdir):
			continue
		for nn in sorted(os.listdir(sdir)):
			idir = os.path.join(sdir, nn)
			if not os.path.isfile(manifest_path(idir)):
				continue
			man = read_manifest(idir)
			it = dict(man)
			it["item_dir"] = idir
			it["mask_dir"] = mask_dir(idir)
			it["crop_dir"] = crop_dir(idir)
			it["filled_dir"] = filled_dir(idir)
			items.append(it)
	return items
