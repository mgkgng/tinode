"""Save Crop & Mask — persist one crop's mask + RGB + transform + prompt.

Phase 1 of the multi-crop object-removal pipeline. A clip has several crops
(from Bbox Crop · Multi); this saves ONE of them, keyed by the clip's stem and
the crop_index, so the whole thing runs per-crop under list expansion and each
crop lands in its own folder <stem>/<crop_index>/.

Writes, losslessly:
  * the mask (crop space) as an 8-bit PNG sequence — exact for a binary mask;
  * optionally the cropped RGB frames (8/16-bit) so removal — even an external
    tool — works on the exact pixels;
  * the crop_info that maps the crop back to the full frame;
  * this crop's VOID prompt (positive/negative), so phase 2 reuses it.

The SOURCE video is never touched or re-encoded — phase 2 re-decodes the
untouched original and reproduces the crop from crop_info. (Node id stays
TI_SaveMasks for backward compatibility; the display name is Save Crop & Mask.)
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
	NODE_ID = "SaveMasks"
	DISPLAY_NAME = "Save Crop & Mask (ti)"
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
					"Clip key (from Video Source Path)."}),
				"crop_index": ("INT", {"default": 0, "min": 0, "max": 9999, "step": 1,
					"tooltip": "Which crop of this clip (from Bbox Crop · Multi)."}),
			},
			"optional": {
				"crop_image": ("IMAGE", {"tooltip":
					"Optional: the CROPPED frames to also export losslessly (crop/), "
					"so removal works on the exact pixels. Wire the crop output."}),
				"positive_prompt": ("STRING", {"default": "", "multiline": True,
					"tooltip": "VOID prompt for THIS crop — stored and reused in phase 2."}),
				"negative_prompt": ("STRING", {"default": "", "multiline": True,
					"tooltip": "Negative prompt for this crop."}),
				"source_path": ("STRING", {"default": "", "tooltip":
					"Absolute path to the source video (from Video Source Path)."}),
				"fps": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 240.0, "step": 0.01}),
				"subdir": ("STRING", {"default": store.DEFAULT_SUBDIR}),
				"crop_bit_depth": (["16", "8"], {"default": "16", "tooltip":
					"Crop export depth. 16 = exact for a 10-bit master (needs cv2)."}),
				"overwrite": ("BOOLEAN", {"default": True, "tooltip":
					"Off = skip a crop already saved (resume a batch)."}),
			},
		}

	RETURN_TYPES = ("STRING",)
	RETURN_NAMES = ("manifest_path",)
	OUTPUT_TOOLTIPS = (
		"Path to the written manifest.json — wire to ForeachListEnd to advance.",
	)
	FUNCTION = "execute"

	def execute(self, mask, crop_info, stem, crop_index=0, crop_image=None,
				positive_prompt="", negative_prompt="", source_path="", fps=0.0,
				subdir=store.DEFAULT_SUBDIR, crop_bit_depth="16", overwrite=True):
		import numpy as np  # noqa: PLC0415
		from PIL import Image  # noqa: PLC0415

		crop_info = first(crop_info)
		validate_crop_xform(crop_info)
		stem = os.path.basename(str(first(stem, "")).strip())
		crop_index = int(first(crop_index, 0))
		if not stem:
			raise RuntimeError(
				"Save Crop & Mask: empty stem. Wire Video Source Path's `stem` here.")

		m = first(mask)
		if not isinstance(m, torch.Tensor):
			raise RuntimeError("Save Crop & Mask: `mask` must be a MASK tensor.")
		if m.dim() == 2:
			m = m.unsqueeze(0)
		if m.dim() != 3:
			raise RuntimeError(
				f"Save Crop & Mask: expected a [frames,H,W] mask, got {tuple(m.shape)}.")
		N, ch, cw = m.shape

		root = store.output_root(subdir)
		idir = store.item_dir(root, stem, crop_index)
		mdir = store.mask_dir(idir)

		if not overwrite and os.path.isfile(store.manifest_path(idir)):
			try:
				if int(store.read_manifest(idir).get("frame_count", -1)) == N:
					return {"ui": {"ti_saved_mask": [{"stem": stem, "crop": crop_index,
													  "skipped": True}]},
							"result": (store.manifest_path(idir),)}
			except Exception:  # noqa: BLE001
				pass

		os.makedirs(mdir, exist_ok=True)
		arr = (m.detach().clamp(0, 1).cpu().numpy() * 255.0 + 0.5).astype(np.uint8)
		for i in range(N):
			Image.fromarray(arr[i], mode="L").save(
				os.path.join(mdir, store.MASK_PATTERN % i), compress_level=6)
		store.prune_stale(mdir, store.MASK_PATTERN, N)

		crop = first(crop_image)
		crop_saved = False
		crop_bits = int(first(crop_bit_depth, "16"))
		if isinstance(crop, torch.Tensor):
			c = crop if crop.dim() == 4 else crop.unsqueeze(0)
			cframes = store.crop_dir(idir)
			store.save_rgb_sequence(c, cframes, store.CROP_PATTERN, crop_bits)
			store.prune_stale(cframes, store.CROP_PATTERN, int(c.shape[0]))
			crop_saved = True

		manifest = {
			"tinode_mask_manifest": store.MANIFEST_VERSION,
			"stem": stem,
			"crop_index": crop_index,
			"source_path": str(first(source_path, "")),
			"frame_count": int(N),
			"crop_height": int(ch),
			"crop_width": int(cw),
			"fps": float(first(fps, 0.0)),
			"crop_info": crop_info,
			"positive_prompt": str(first(positive_prompt, "")),
			"negative_prompt": str(first(negative_prompt, "")),
			"mask_subfolder": store.MASK_SUBFOLDER,
			"mask_pattern": store.MASK_PATTERN,
			"has_crop": crop_saved,
			"crop_subfolder": store.CROP_SUBFOLDER if crop_saved else "",
			"crop_pattern": store.CROP_PATTERN,
			"crop_bit_depth": crop_bits if crop_saved else 0,
			"has_filled": False,
			"filled_subfolder": store.FILLED_SUBFOLDER,
			"filled_pattern": store.FILLED_PATTERN,
		}
		store.write_manifest(idir, manifest)
		extra = f" + {crop_bits}-bit crops" if crop_saved else ""
		print(f"[tinode] Save Crop & Mask: {stem!r} crop {crop_index} — "
			  f"{N} mask frame(s){extra} -> {idir}")
		return {"ui": {"ti_saved_mask": [{"stem": stem, "crop": crop_index,
										  "frames": N, "crop_saved": crop_saved}]},
				"result": (store.manifest_path(idir),)}
