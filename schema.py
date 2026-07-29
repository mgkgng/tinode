"""The custom data types tinode passes between its own nodes.

ComfyUI treats any unknown type string as an opaque token: it only checks that
the string on an output matches the string on an input, never the shape of what
actually flows. So these contracts live nowhere in the engine — they exist only
in the producers and consumers agreeing. This module writes them down, and the
validators turn "wrong shape" into a readable error instead of an IndexError or
a tensor-size crash deep inside a blend.

---------------------------------------------------------------- TI_CROP_XFORM
How to undo a crop. Produced by Mask Bbox Crop / Mask Crop · Center Fill /
Bbox Crop · Manual, consumed by Mask Crop Paste Back and the Crop Info
Drop/Pick nodes.

    {
      "H": int, "W": int, "C": int,   # the ORIGINAL frame the crop came from
      "size": int,                    # optional, square-canvas crops only
      "items": [ item | None, ... ],  # one per crop, index-aligned with them
    }

    item = {
      "y0","x0": top-left of the crop in the ORIGINAL frame
      "h","w":   size of the region it occupies there
      "oy","ox": where the pixels sit inside the emitted crop canvas
      "nh","nw": their size on that canvas
    }

`item is None` marks a frame with nothing to paste (empty mask); consumers skip
it and keep index alignment. When the crop is not rescaled, oy/ox are 0 and
nh/nw equal h/w — Paste Back then writes the pixels through unchanged, which is
what makes the crop -> paste round trip bit-exact.

H/W are load-bearing: Paste Back composites into a frame of exactly that size.
Handing it a differently-sized image (the classic mistake is wiring the crop
node's OUTPUT back in instead of its input) means the paste runs off the canvas.

------------------------------------------------------------ TI_SAM3_SEGMENTS
Every detection SAM3 made, per frame, before any merge. Produced by the patched
EasySAM3 Segment node or Mask to Segment, consumed and re-emitted by Pick
Segments and Add Segments (so they chain in either order).

    {
      "num_frames": int, "height": int, "width": int,
      "frames": [ [segment, ...], ... ],   # one list per frame, len == num_frames
      "ids": [int, ...],                   # sorted unique ids across the clip
    }

    segment = {
      "id":   int    stable track id (SAM3), or >= 1_000_000 for a manual box
      "bbox": [x0,y0,x1,y1] in FULL-resolution source pixels
      "conf": float
      "mask": uint8 tensor [y1-y0, x1-x0], cropped to its own bbox
    }

The mask is stored bbox-cropped, not full-frame: a segment is mostly empty
space, and a 90-segment clip would not otherwise fit in memory. Rebuild a
full-frame mask by pasting it into a zeros([height,width]) canvas at y0,x0.

Ids are the unit of selection: Pick Segments filters by id across every frame,
so an id means the same object for the whole clip.
"""

from __future__ import annotations

_ITEM_KEYS = ("y0", "x0", "h", "w", "oy", "ox", "nh", "nw")


def validate_crop_xform(info, *, where="crop_info"):
	"""Raise RuntimeError unless `info` is a well-formed TI_CROP_XFORM."""
	if not isinstance(info, dict):
		raise RuntimeError(f"{where} must be a TI_CROP_XFORM dict, got {type(info).__name__}.")
	for key in ("H", "W", "items"):
		if key not in info:
			raise RuntimeError(f"{where} is missing {key!r}.")
	if not isinstance(info["items"], list):
		raise RuntimeError(f"{where}['items'] must be a list.")
	for i, item in enumerate(info["items"]):
		if item is None:
			continue                      # legitimately empty frame
		missing = [k for k in _ITEM_KEYS if k not in item]
		if missing:
			raise RuntimeError(f"{where}['items'][{i}] is missing {missing}.")
	return info


def validate_segments(seg, *, where="segments"):
	"""Raise RuntimeError unless `seg` is a well-formed TI_SAM3_SEGMENTS."""
	if not isinstance(seg, dict):
		raise RuntimeError(
			f"{where} must be a TI_SAM3_SEGMENTS dict, got {type(seg).__name__}. "
			f"Connect the 'segments' output of EasySAM3 Segment."
		)
	for key in ("num_frames", "height", "width", "frames"):
		if key not in seg:
			raise RuntimeError(f"{where} is missing {key!r}.")
	if not isinstance(seg["frames"], list):
		raise RuntimeError(f"{where}['frames'] must be a list of per-frame lists.")
	return seg


def crop_xform_matches(info, height, width):
	"""True when a source frame of (height,width) is the one `info` was built for."""
	return int(info.get("H", -1)) == int(height) and int(info.get("W", -1)) == int(width)
