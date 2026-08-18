"""Save Masks — persist a clip's mask + crop transform for a later removal pass.

Phase 1 of the two-workflow object-removal pipeline. After you've cropped a
video and masked the thing to remove, this writes, keyed by the source clip's
stem: the mask (lossless) and the crop_info that maps the crop back to the
original frame. Phase 2 (Load Masks) reads them to run removal and paste it in.

Nothing here is lossy:
  * the mask is saved as a lossless PNG sequence in CROP space — 8-bit is exact
    for the binary masks SAM3 produces (and 256 levels is plenty for a soft
    edge, which anyway gets re-feathered at paste time);
  * the SOURCE video is never touched or re-encoded — only the mask and a small
    JSON manifest are written, so there is no generation loss. Phase 2
    re-decodes the untouched original and reproduces the crop from crop_info.

Drop it at the end of a Foreach loop over Load Videos: wire the mask, the
crop_info, and the clip's stem (from Video Source Path), then send its output to
ForeachListEnd so the loop advances to the next clip.
"""

from __future__ import annotations

import os

import torch

from ...base import TiNode, first
from ...registry import register
from ...schema import validate_crop_xform
from . import _mask_store as store


@register
class SaveMasks(TiNode):
	DISPLAY_NAME = "Save Masks (ti)"
	CATEGORY = "tinode/video"
	OUTPUT_NODE = True

	@classmethod
	def INPUT_TYPES(cls):
		return {
			"required": {
				"mask": ("MASK", {"tooltip":
					"The mask to remove, in CROP space ([frames, ch, cw])."}),
				"crop_info": ("TI_CROP_XFORM", {"tooltip":
					"From the crop node — maps the crop back to the full frame."}),
				"stem": ("STRING", {"default": "", "tooltip":
					"Key for this clip (from Video Source Path). One folder per stem."}),
			},
			"optional": {
				"source_path": ("STRING", {"default": "", "tooltip":
					"Absolute path to the source video, stored so phase 2 can find "
					"it. From Video Source Path's `path` output."}),
				"fps": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 240.0, "step": 0.01,
					"tooltip": "Recorded in the manifest (informational)."}),
				"subdir": ("STRING", {"default": store.DEFAULT_SUBDIR, "tooltip":
					"Folder under ComfyUI's output/ to write the mask store into."}),
				"overwrite": ("BOOLEAN", {"default": True, "tooltip":
					"Off = skip a clip whose mask is already saved (resume a batch)."}),
			},
		}

	RETURN_TYPES = ("STRING",)
	RETURN_NAMES = ("manifest_path",)
	OUTPUT_TOOLTIPS = (
		"Path to the written manifest.json — wire to ForeachListEnd's "
		"intermediate_output to advance the loop.",
	)
	FUNCTION = "execute"

	def execute(self, mask, crop_info, stem, source_path="", fps=0.0,
				subdir=store.DEFAULT_SUBDIR, overwrite=True):
		import numpy as np  # noqa: PLC0415
		from PIL import Image  # noqa: PLC0415

		crop_info = first(crop_info)
		validate_crop_xform(crop_info)
		stem = os.path.basename(str(first(stem, "")).strip())
		if not stem:
			raise RuntimeError(
				"Save Masks: empty stem. Wire Video Source Path's `stem` output "
				"here so each clip's mask gets its own folder.")

		m = first(mask)
		if not isinstance(m, torch.Tensor):
			raise RuntimeError("Save Masks: `mask` must be a MASK tensor.")
		if m.dim() == 2:
			m = m.unsqueeze(0)                          # [ch,cw] -> [1,ch,cw]
		if m.dim() != 3:
			raise RuntimeError(
				f"Save Masks: expected a [frames,H,W] mask, got shape {tuple(m.shape)}.")
		N, ch, cw = m.shape

		root = store.output_root(subdir)
		cdir = store.clip_dir(root, stem)
		mdir = store.mask_dir(cdir)

		if not overwrite and os.path.isfile(store.manifest_path(cdir)):
			try:
				prev = store.read_manifest(cdir)
				if int(prev.get("frame_count", -1)) == N:
					return {"ui": {"ti_saved_mask": [{"stem": stem, "skipped": True}]},
							"result": (store.manifest_path(cdir),)}
			except Exception:  # noqa: BLE001 — a broken manifest just gets rewritten
				pass

		os.makedirs(mdir, exist_ok=True)
		# Lossless: exact 8-bit for a binary mask; round-to-nearest, no dithering.
		arr = (m.detach().clamp(0, 1).cpu().numpy() * 255.0 + 0.5).astype(np.uint8)
		for i in range(N):
			Image.fromarray(arr[i], mode="L").save(
				os.path.join(mdir, store.MASK_PATTERN % i), compress_level=6)
		# Remove any stale frames from a previous, longer save of the same clip.
		i = N
		while True:
			stale = os.path.join(mdir, store.MASK_PATTERN % i)
			if not os.path.exists(stale):
				break
			os.remove(stale)
			i += 1

		manifest = {
			"tinode_mask_manifest": store.MANIFEST_VERSION,
			"stem": stem,
			"source_path": str(first(source_path, "")),
			"frame_count": int(N),
			"crop_height": int(ch),
			"crop_width": int(cw),
			"fps": float(first(fps, 0.0)),
			"crop_info": crop_info,
			"mask_subfolder": store.MASK_SUBFOLDER,
			"mask_pattern": store.MASK_PATTERN,
		}
		store.write_manifest(cdir, manifest)
		print(f"[tinode] Save Masks: wrote {N} mask frame(s) for {stem!r} -> {cdir}")
		return {"ui": {"ti_saved_mask": [{"stem": stem, "frames": N}]},
				"result": (store.manifest_path(cdir),)}
