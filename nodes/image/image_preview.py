"""Image Preview — look at an image, download it, leave nothing behind.

The same bargain as Video Preview, for stills. ComfyUI's own Preview Image writes
a PNG into temp/ and Save Image writes one into output/; both leave a file and a
queue-history entry for every run, which is a lot of clutter for a look. This
holds the batch in RAM, serves each frame from there, and only writes something
when you press DOWNLOAD — and then it writes it to your machine, not the server's.

Batches are handled as batches: step through them in the node, and download
either the frame you are looking at or all of them as a zip.

png and tiff go out at the master's own bit depth with no colour conversion, so
they are bit-exact against what the graph produced; jpg and webp are there for
when you just want to paste something into a message. `precision: auto` keeps 16
bits only when the source actually carries more than 8 (see
_preview_store.pack_frames), so an 8-bit graph pays nothing for the guarantee.

The IMAGE passthrough means this can sit mid-chain as a monitor rather than only
at the end of one.
"""

from __future__ import annotations

from ...base import TiNode
from ...registry import register
from . import _preview_store as store


@register
class ImagePreview(TiNode):
	DISPLAY_NAME = "Image Preview · Ephemeral (ti)"
	CATEGORY = "tinode/image"
	OUTPUT_NODE = True

	@classmethod
	def INPUT_TYPES(cls):
		return {
			"required": {
				"images": ("IMAGE",),
			},
			"optional": {
				"precision": (["auto", "8-bit", "16-bit"], {"default": "auto",
					"tooltip": "How the held batch is stored. auto keeps 16-bit only when the "
							   "source actually carries more than 8 bits. 16-bit doubles the RAM."}),
				"hold_minutes": ("INT", {"default": 30, "min": 1, "max": 720, "step": 1,
					"tooltip": "How long the batch stays in RAM before it is released."}),
			},
			"hidden": {"unique_id": "UNIQUE_ID"},
		}

	RETURN_TYPES = ("IMAGE",)
	RETURN_NAMES = ("images",)
	FUNCTION = "execute"

	def execute(self, images, precision="auto", hold_minutes=30, unique_id=None):
		raw, n, h, w, depth = store.pack_frames(images, precision)

		# No proxy: a still is cheap enough to encode per view, and pre-encoding
		# a whole batch would cost more than the looking is worth.
		sid = store.put(raw, n, h, w, depth, 1.0, owner=str(unique_id or ""),
						hold=int(hold_minutes) * 60, proxy=b"", proxy_mime="image/png")

		st = store.stats()
		print(f"[tinode] Image Preview: holding {n} image(s) {w}x{h} {depth}-bit "
			  f"({len(raw) / 1e6:.0f} MB) in RAM — nothing written to disk. "
			  f"store {st['bytes'] / 1e6:.0f}/{st['cap'] / 1e6:.0f} MB")

		ui = {
			"id": sid, "frames": n, "width": w, "height": h, "depth": depth,
			"bytes": len(raw), "hold": int(hold_minutes) * 60,
			"store_bytes": st["bytes"], "store_cap": st["cap"],
		}
		return {"ui": {"ti_ipreview": [ui]}, "result": (images,)}
