"""Video Preview — watch a clip play in the node, leaving nothing behind.

Save Video · Combine writes a file: it lands in output/, it shows up in the
queue history, and after an afternoon of iterating you have two hundred of them
to sort through. Most of the time you only wanted to *look* at the result.

This node plays the clip and stores nothing. The frames are held in RAM for the
session, a small proxy streams to a <video> in the node body, and when you
actually want the file there is a CREATE VIDEO button that encodes it on the
spot and downloads it to your machine. The server keeps no copy: output/ and
temp/ are untouched, and a ComfyUI restart erases every trace.

The clip is held at the precision it arrived with (see _preview_store.pack_frames),
so a download is not limited by what the preview looked like. The proxy you watch
is a lossy h264 — CREATE VIDEO always re-encodes from the untouched master, and
its ffv1 and png-sequence options are bit-exact against the frames the graph
produced.

Audio is optional and rides along everywhere: you hear it in the node, and every
download muxes it. The lossless formats carry it as float PCM so "bit-exact"
covers the whole file and not just the picture. It is fitted to the picture's
length — padded with silence if it ran short, cut if it ran long — because the
video timing is the thing that must not move.

The IMAGE/AUDIO passthrough means this can sit mid-chain as a monitor rather
than only at the end of one.
"""

from __future__ import annotations

from ...base import TiNode
from ...registry import register
from . import _preview_store as store

# Long side of the proxy. Not the resolution of anything you download.
_SIZES = {"360p": 640, "540p": 960, "720p": 1280, "1080p": 1920, "full": 0}


@register
class VideoPreview(TiNode):
	DISPLAY_NAME = "Video Preview · Ephemeral (ti)"
	CATEGORY = "tinode/video"
	OUTPUT_NODE = True

	@classmethod
	def INPUT_TYPES(cls):
		return {
			"required": {
				"images": ("IMAGE",),
				"frame_rate": ("FLOAT", {"default": 24.0, "min": 0.1, "max": 240.0, "step": 0.01}),
			},
			"optional": {
				"audio": ("AUDIO", {
					"tooltip": "Optional. Plays with the preview and is muxed into every "
							   "download, fitted to the picture's length."}),
				"preview_size": (list(_SIZES), {"default": "720p",
					"tooltip": "Resolution of the proxy you watch. Downloads are always full size."}),
				"precision": (["auto", "8-bit", "16-bit"], {"default": "auto",
					"tooltip": "How the held clip is stored. auto keeps 16-bit only when the "
							   "source actually carries more than 8 bits. 16-bit doubles the RAM."}),
				"hold_minutes": ("INT", {"default": 30, "min": 1, "max": 720, "step": 1,
					"tooltip": "How long the clip stays in RAM before it is released."}),
			},
			"hidden": {"unique_id": "UNIQUE_ID"},
		}

	RETURN_TYPES = ("IMAGE", "AUDIO")
	RETURN_NAMES = ("images", "audio")
	FUNCTION = "execute"

	def execute(self, images, frame_rate=24.0, audio=None, preview_size="720p",
				precision="auto", hold_minutes=30, unique_id=None):
		raw, n, h, w, depth = store.pack_frames(images, precision)
		track = store.pack_audio(audio)

		proxy, pw, ph = store.make_proxy(
			raw, n, h, w, depth, float(frame_rate),
			long_side=_SIZES.get(preview_size, 1280), audio=track)

		sid = store.put(raw, n, h, w, depth, float(frame_rate),
						owner=str(unique_id or ""), hold=int(hold_minutes) * 60,
						proxy=proxy, proxy_mime="video/mp4", audio=track)

		st = store.stats()
		# Say the track's length against the picture's: a mismatch here is the
		# usual cause of a preview that drifts, and it is invisible otherwise.
		secs = n / float(frame_rate)
		note = (f", audio {track['seconds']:.2f}s {track['rate']}Hz x{track['channels']}"
				f" vs {secs:.2f}s of picture" if track else ", no audio")
		print(f"[tinode] Video Preview: holding {n} frame(s) {w}x{h} {depth}-bit "
			  f"({len(raw) / 1e6:.0f} MB){note} in RAM — nothing written to disk. "
			  f"store {st['bytes'] / 1e6:.0f}/{st['cap'] / 1e6:.0f} MB")

		ui = {
			"id": sid, "frames": n, "width": w, "height": h,
			"proxy_width": pw, "proxy_height": ph,
			"fps": float(frame_rate), "depth": depth,
			"audio": ({"rate": track["rate"], "channels": track["channels"],
					   "seconds": track["seconds"]} if track else None),
			"bytes": len(raw) + len(proxy) + (len(track["raw"]) if track else 0),
			"hold": int(hold_minutes) * 60,
			"store_bytes": st["bytes"], "store_cap": st["cap"],
		}
		return {"ui": {"ti_vpreview": [ui]}, "result": (images, audio)}
