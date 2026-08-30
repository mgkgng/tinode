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

# What has to be unchanged for an existing render to still describe this crop.
FILL_IDENTITY = ("crop_width", "crop_height", "frame_start", "frame_end",
				 "frame_count")


def fill_keys_to_carry(prev, manifest):
	"""(carry, stale) — which fill flags survive a re-save of this crop.

	Re-saving the SAME crop must keep its renders: the filled/ folders are still
	on disk, and dropping the flags makes Load Clips call a finished clip "not
	ready". But re-AUTHORING it — a different box, or a different stretch of the
	clip — leaves a fill of the old pixels behind, and a flag saying it is
	current turns phase 2b into a size mismatch, or a silent paste of another
	take. So the flags travel only while the crop's identity is unchanged.
	"""
	stale = [k for k, v in prev.items()
			 if k.startswith(("has_filled", "filled_frames")) and v]
	if all(prev.get(k) == manifest.get(k) for k in FILL_IDENTITY):
		return stale, []
	return [], stale


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
					"tooltip": "Which crop of this chunk (from Crop Item)."}),
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
				"chunk_index": ("INT", {"default": -1, "min": -1, "max": 9999, "step": 1,
					"tooltip": "Which chunk of the clip (from Chunk Item). -1 = the "
							   "clip was not chunked. Without it, every chunk's "
							   "crop 0 would overwrite the same folder."}),
				"frame_start": ("INT", {"default": 0, "min": 0, "max": 9999999, "step": 1,
					"tooltip": "First frame of this chunk in the ORIGINAL clip (from "
							   "Chunk Item). Phase 2 pastes the result back here."}),
				"frame_end": ("INT", {"default": 0, "min": 0, "max": 9999999, "step": 1,
					"tooltip": "End frame (exclusive) of this chunk in the original."}),
				# APPENDED last: widget values are positional in saved graphs.
				"source_every_nth": ("INT", {"default": 1, "min": 1, "max": 10, "step": 1,
					"tooltip": "Set to the select_every_nth used at Load Video when "
							   "authoring from a REDUCED-rate clip (2 = the store "
							   "holds every 2nd source frame). The loaders then "
							   "never stride this crop again — the double-speed "
							   "guard."}),
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
				subdir=store.DEFAULT_SUBDIR, crop_bit_depth="16", overwrite=True,
				chunk_index=-1, frame_start=0, frame_end=0, source_every_nth=1):
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

		chunk = int(first(chunk_index, -1))
		chunk = None if chunk < 0 else chunk
		fstart = int(first(frame_start, 0))
		fend = int(first(frame_end, 0)) or (fstart + N)

		root = store.output_root(subdir)
		idir = store.item_dir(root, stem, crop_index, chunk)
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
			"chunk_index": chunk if chunk is not None else -1,
			"frame_start": fstart,
			"frame_end": fend,
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
			"source_every_nth": max(1, int(first(source_every_nth, 1))),
		}
		# Re-saving a crop's mask must NOT forget its renders: the filled/
		# folders are still on disk, and losing the flags makes Load Clips call
		# a finished clip "not ready". Carry every fill-related key forward —
		# but ONLY while the crop is still the same crop. Re-authoring it with a
		# different box or a different stretch of the clip leaves a fill of the
		# OLD pixels behind, and a flag that says it is current turns phase 2b
		# into a size mismatch (or worse, a silent paste of another take).
		try:
			prev = store.read_manifest(idir)
			carry, stale = fill_keys_to_carry(prev, manifest)
			for key in carry:
				manifest[key] = prev[key]
			if stale:
				print(f"[tinode] Save Crop & Mask: {stem!r} crop {crop_index} was "
					  f"re-authored ({prev.get('crop_width')}x{prev.get('crop_height')} "
					  f"frames {prev.get('frame_start')}-{prev.get('frame_end')} -> "
					  f"{cw}x{ch} frames {fstart}-{fend}) — its existing fill is of the "
					  f"OLD crop and is now marked NOT rendered. Delete "
					  f"{store.filled_dir(idir)} and run phase 2a again.")
		except Exception:  # noqa: BLE001 — first save: nothing to carry
			pass
		store.write_manifest(idir, manifest)
		extra = f" + {crop_bits}-bit crops" if crop_saved else ""
		where = f"chunk {chunk} crop {crop_index}" if chunk is not None else f"crop {crop_index}"
		print(f"[tinode] Save Crop & Mask: {stem!r} {where} frames {fstart}-{fend} — "
			  f"{N} mask frame(s){extra} -> {idir}")
		return {"ui": {"ti_saved_mask": [{"stem": stem, "crop": crop_index,
										  "chunk": chunk, "start": fstart,
										  "frames": N, "crop_saved": crop_saved}]},
				"result": (store.manifest_path(idir),)}
