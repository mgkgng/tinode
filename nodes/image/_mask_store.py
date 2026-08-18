"""Shared on-disk layout for the mask handoff between the two removal passes.

Save Masks (phase 1) writes it, Load Masks (phase 2) reads it. Keeping the
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
MASK_PATTERN = "%05d.png"
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


def write_manifest(cdir, manifest):
	with open(manifest_path(cdir), "w", encoding="utf-8") as fh:
		json.dump(manifest, fh, ensure_ascii=False, indent=2)


def read_manifest(cdir):
	with open(manifest_path(cdir), "r", encoding="utf-8") as fh:
		return json.load(fh)
