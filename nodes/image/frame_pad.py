"""Frame Pad / Frame Unpad — extend a clip's ends and pad it to a model length.

A merge of the old Extend Video (prepend/append with freeze / splice / colour)
and Frame Pad (auto-pad to the length a video model returns unchanged). One node
now does both: add real lead-in/-out frames AND grow the head to a valid length,
then Frame Unpad drops the pad so the clip is its original length again.

Why a length target at all: a video model hands back EXACTLY the length it was
given only when that length fits its temporal grid. The rule is `length % modulo
== remainder`:

  * VOID   -> multiple of 8   (temporal_compression 4 x patch_size_t 2)  => modulo 8, remainder 0
  * MiniMax H3 -> length % 17 == 5                                        => modulo 17, remainder 5
  * a plain ratio*n+1 VAE                                                 => modulo=ratio, remainder 1

Head fill (what the pad frames are):
  auto         context tail if context_images is wired, else the first frame
  first_frame  freeze the clip's first frame
  last_frame   freeze the clip's last frame
  color        a solid colour (pad_color) — the "empty video" you control

Padding just repeats/holds frames (or a flat colour) — no interpolation, no
quality loss. The mask is padded in lockstep so it stays aligned with the video;
keep the ORIGINAL mask for the final paste-back.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from ...base import TiNode, first
from ...registry import register

_HEAD_FILL = ["auto", "first_frame", "last_frame", "color", "rewind"]
_TAIL_FILL = ["last_frame", "first_frame", "color", "rewind"]


def _rewind_indices(n, count, head):
	"""Ping-pong (boomerang) frame indices for `count` pad frames.

	Position stays continuous at the seam and the seam frame is NOT duplicated,
	so the clip keeps *playing* (in reverse) at the boundary instead of freezing —
	the best filler for a model that regenerates every frame. head=True places the
	frames BEFORE frame 0 (display order leads into it); head=False after n-1.
	"""
	if count <= 0:
		return []
	if n <= 1:
		return [0] * count                      # a 1-frame clip can't bounce; hold it
	down = list(range(n - 2, -1, -1))           # n-2 .. 0
	up = list(range(1, n))                      # 1 .. n-1
	unit = down + up                            # period 2(n-1), seamless, no dup seam
	idx = [unit[i % len(unit)] for i in range(count)]
	if head:                                    # mirror to the start, lead into 0
		idx = [(n - 1 - j) for j in idx][::-1]
	return idx


def frame_pad_count(length, min_pad=8, modulo=8, remainder=0):
	"""Frames to add so (length + p) satisfies `% modulo == remainder`, p >= min_pad.

	Generalises the old ratio rule: remainder 0 = a plain multiple of `modulo`
	(VOID), remainder 1 = modulo*n + 1, remainder 5 with modulo 17 = MiniMax H3.
	`p` is the smallest pad at or above min_pad that hits the target, so the
	padding that reduces model error is preserved.
	"""
	length = int(length)
	modulo = max(1, int(modulo))
	min_pad = int(min_pad)
	remainder = int(remainder) % modulo
	if length <= 0:
		return min_pad
	base = length + min_pad
	return min_pad + ((remainder - base) % modulo)


def _hex_rgb(s):
	"""'#rrggbb' / '#rgb' -> (r,g,b) floats 0..1; black on anything invalid."""
	s = str(s or "").strip().lstrip("#")
	if len(s) == 3:
		s = "".join(c * 2 for c in s)
	if len(s) != 6:
		return (0.0, 0.0, 0.0)
	try:
		return tuple(int(s[i:i + 2], 16) / 255.0 for i in (0, 2, 4))
	except ValueError:
		return (0.0, 0.0, 0.0)


def _as_batch(t):
	return t if t.dim() == 4 else t.unsqueeze(0)


def _conform(other, base):
	"""Resize + channel/dtype/device match `other` to `base` [.,H,W,C]."""
	H, W, C = base.shape[1], base.shape[2], base.shape[3]
	other = _as_batch(other).to(device=base.device, dtype=base.dtype)
	if other.shape[1] != H or other.shape[2] != W:
		x = other.permute(0, 3, 1, 2)
		x = F.interpolate(x, size=(H, W), mode="bilinear", align_corners=False)
		other = x.permute(0, 2, 3, 1).contiguous()
	if other.shape[3] > C:
		other = other[..., :C]
	elif other.shape[3] < C:
		pad = torch.ones(other.shape[0], H, W, C - other.shape[3],
						 dtype=other.dtype, device=other.device)
		other = torch.cat([other, pad], dim=3)
	return other


def _fill_frames(count, mode, base, color_rgb):
	"""`count` head/tail frames of a fill mode -> [count,H,W,C]."""
	H, W, C = base.shape[1], base.shape[2], base.shape[3]
	if count <= 0:
		return base[:0]
	if mode == "color":
		f = torch.zeros(1, H, W, C, dtype=base.dtype, device=base.device)
		for i, v in enumerate(color_rgb[:min(3, C)]):
			f[0, :, :, i] = v
		return f.repeat(count, 1, 1, 1)
	frame = base[-1] if mode == "last_frame" else base[0]
	return frame.unsqueeze(0).repeat(count, 1, 1, 1)


def _fill_mask(count, mode, m):
	"""Mask matched to a fill: a colour fill masks nothing; a freeze keeps its
	own frame's mask (the object is still there)."""
	if count <= 0:
		return m[:0]
	if mode == "color":
		return torch.zeros((count, m.shape[1], m.shape[2]), dtype=m.dtype)
	frame = m[-1:] if mode == "last_frame" else m[:1]
	return frame.repeat(count, 1, 1)


@register
class FramePad(TiNode):
	DISPLAY_NAME = "Frame Pad (ti)"
	CATEGORY = "tinode/video"

	@classmethod
	def INPUT_TYPES(cls):
		return {
			"required": {
				"images": ("IMAGE", {"tooltip": "The clip to pad / extend."}),
			},
			"optional": {
				"mask": ("MASK", {"tooltip":
					"Padded in lockstep. Keep the ORIGINAL mask for paste-back."}),
				"side": (["prepend", "append", "both"], {"default": "prepend",
					"tooltip": "Which end to pad: prepend (head), append (tail), or "
							   "both (split). For append/both, set Frame Unpad's "
							   "expected_frames to the original length so the tail pad "
							   "is trimmed."}),
				"auto_length": ("BOOLEAN", {"default": True, "tooltip":
					"Grow the padding so the TOTAL length hits the model's grid "
					"(% modulo == remainder). Off = add exactly min_padding frames."}),
				"modulo": ("INT", {"default": 8, "min": 1, "max": 4096, "step": 1,
					"tooltip": "Length must be a multiple of this (plus remainder). "
							   "VOID 8; MiniMax H3 17; a ratio*n+1 VAE = the ratio."}),
				"remainder": ("INT", {"default": 0, "min": 0, "max": 4096, "step": 1,
					"tooltip": "Target remainder: 0 = plain multiple (VOID); 1 = "
							   "modulo*n+1; 5 with modulo 17 = MiniMax H3 (%17==5)."}),
				"min_padding": ("INT", {"default": 8, "min": 0, "max": 4096, "step": 1,
					"tooltip": "Minimum frames to add (auto) / exact frames to add "
							   "(auto off). 8 reduces model error at the head."}),
				"head_fill": (_HEAD_FILL, {"default": "auto", "tooltip":
					"Head pad source (prepend/both): auto (context if wired, else first "
					"frame), first_frame, last_frame, color (pad_color), or rewind "
					"(play the clip backwards — continuous motion, best for a "
					"generative model)."}),
				"tail_fill": (_TAIL_FILL, {"default": "last_frame", "tooltip":
					"Tail pad source (append/both): last_frame, first_frame, color, or "
					"rewind (ping-pong: the shot keeps playing in reverse instead of "
					"freezing — no deceleration for the model to smear back)."}),
				"pad_color": ("STRING", {"default": "#000000", "tooltip":
					"Colour for a 'color' fill — the empty video you control (#hex)."}),
				"max_frames": ("INT", {"default": 197, "min": 0, "max": 100000, "step": 1,
					"tooltip": "Warn if the total exceeds this (0 = off)."}),
				"context_images": ("IMAGE", {"tooltip":
					"Head-pad from these frames (the previous chunk's TAIL) instead of "
					"a freeze, for continuity across a cut. Needs head_fill = auto."}),
				"context_mask": ("MASK", {"tooltip":
					"The mask for context_images — needed when the context is the "
					"ORIGINAL preceding frames (the object is still in them)."}),
				"prepend_video": ("IMAGE", {"tooltip":
					"Splice a real clip in at the HEAD (conformed to this clip)."}),
				"append_video": ("IMAGE", {"tooltip":
					"Splice a real clip in at the TAIL (conformed to this clip)."}),
			},
		}

	RETURN_TYPES = ("IMAGE", "MASK", "TI_FRAME_PAD", "INT")
	RETURN_NAMES = ("images", "mask", "pad_info", "total_frames")
	OUTPUT_TOOLTIPS = (
		"Padded clip.",
		"The mask, padded to match.",
		"Everything Frame Unpad needs to reverse this exactly — the side, the head "
		"count and the original length. Wire it straight into Frame Unpad.",
		"Total length after padding.",
	)
	FUNCTION = "execute"

	def execute(self, images, mask=None, side="prepend", auto_length=True, modulo=8,
				remainder=0, min_padding=8, head_fill="auto", tail_fill="last_frame",
				pad_color="#000000", max_frames=197,
				context_images=None, context_mask=None,
				prepend_video=None, append_video=None, **_stale):
		imgs = first(images)
		if imgs.dim() != 4:
			imgs = imgs.unsqueeze(0) if imgs.dim() == 3 else imgs
		L, H, W, C = imgs.shape
		side = first(side, "prepend")
		modulo = int(first(modulo, 8))
		remainder = int(first(remainder, 0))
		min_padding = int(first(min_padding, 8))
		head_fill = first(head_fill, "auto")
		tail_fill = first(tail_fill, "last_frame")
		color = _hex_rgb(first(pad_color, "#000000"))
		auto = bool(first(auto_length, True))

		# Splices sit next to the base and count as content for the length target.
		pre_vid = first(prepend_video)
		splice_pre = _conform(pre_vid, imgs) if isinstance(pre_vid, torch.Tensor) else imgs[:0]
		app_vid = first(append_video)
		splice_app = _conform(app_vid, imgs) if isinstance(app_vid, torch.Tensor) else imgs[:0]

		content = L + splice_pre.shape[0] + splice_app.shape[0]
		p = frame_pad_count(content, min_padding, modulo, remainder) if auto else min_padding

		# Distribute the pad across the chosen end(s).
		if side == "append":
			hp, tp = 0, p
		elif side == "both":
			tp = p // 2
			hp = p - tp
		else:                                    # prepend (default)
			hp, tp = p, 0

		# head pad pixels. head_ridx / tail_ridx hold the rewind frame order so the
		# mask can be reordered the same way.
		head_ridx = tail_ridx = None
		ctx = first(context_images)
		from_context = head_fill == "auto" and isinstance(ctx, torch.Tensor) and hp > 0
		if from_context:
			ctx = _as_batch(ctx)
			# The context must be the SAME crop — resizing it would misalign the
			# carried motion. Splice inputs conform; context does not.
			if ctx.shape[1:3] != imgs.shape[1:3]:
				raise RuntimeError(
					f"Frame Pad: context_images is {tuple(ctx.shape[1:3])} but the clip "
					f"is {tuple(imgs.shape[1:3])} — the previous chunk must be the SAME crop.")
			ctx = ctx.to(imgs.dtype)
			t = ctx[-hp:]
			if t.shape[0] < hp:
				t = torch.cat([t[:1].repeat(hp - t.shape[0], 1, 1, 1), t], 0)
			head_pad = t
		elif head_fill == "rewind":
			head_ridx = _rewind_indices(L, hp, head=True)
			head_pad = imgs[torch.tensor(head_ridx, dtype=torch.long)] if head_ridx else imgs[:0]
		else:
			head_pad = _fill_frames(hp, "first_frame" if head_fill == "auto" else head_fill,
									imgs, color)

		if tail_fill == "rewind":
			tail_ridx = _rewind_indices(L, tp, head=False)
			tail_pad = imgs[torch.tensor(tail_ridx, dtype=torch.long)] if tail_ridx else imgs[:0]
		else:
			tail_pad = _fill_frames(tp, tail_fill, imgs, color)

		out = torch.cat([head_pad, splice_pre, imgs, splice_app, tail_pad], 0)
		head_added = head_pad.shape[0] + splice_pre.shape[0]
		total = out.shape[0]
		cap = int(first(max_frames, 0))
		if cap and total > cap:
			print(f"[tinode] Frame Pad: total {total} exceeds max_frames {cap} — chunk upstream.")

		# --- mask, padded to match every added frame --------------------------
		m = first(mask)
		if isinstance(m, torch.Tensor):
			if m.dim() == 2:
				m = m.unsqueeze(0)
			if m.shape[0] != L:
				raise RuntimeError(
					f"Frame Pad: the mask has {m.shape[0]} frame(s) but the video has "
					f"{L}. They must line up — check the mask and image came from the "
					"same clip.")
			if m.shape[1] != H or m.shape[2] != W:
				raise RuntimeError(
					f"Frame Pad: the mask is {m.shape[1]}x{m.shape[2]} but the video is {H}x{W}.")
			# Rewind pads are REAL frames (object still present), so carry the mask
			# reordered the same way — not zeros — so removal stays consistent.
			if head_ridx is not None:
				mhead = m[torch.tensor(head_ridx, dtype=torch.long)] if head_ridx else m[:0]
			else:
				mhead = self._head_mask(hp, head_fill, from_context, first(context_mask), m)
			if tail_ridx is not None:
				mtail = m[torch.tensor(tail_ridx, dtype=torch.long)] if tail_ridx else m[:0]
			else:
				mtail = _fill_mask(tp, tail_fill, m)
			zeros_pre = torch.zeros((splice_pre.shape[0], H, W), dtype=m.dtype)
			zeros_app = torch.zeros((splice_app.shape[0], H, W), dtype=m.dtype)
			padded_mask = torch.cat([mhead, zeros_pre, m, zeros_app, mtail], 0)
		else:
			padded_mask = torch.zeros((total, H, W), dtype=torch.float32)

		pad_info = {"side": side, "head": int(head_added), "original": int(L),
					"total": int(total)}
		return (out, padded_mask, pad_info, int(total))

	@staticmethod
	def _head_mask(p, head_fill, from_context, context_mask, m):
		"""Mask for the head pad — matched to what the head pixels are."""
		H, W = m.shape[1], m.shape[2]
		if p <= 0:
			return m[:0]
		if from_context:
			cm = context_mask
			if isinstance(cm, torch.Tensor):
				# context still holds the object -> carry its mask so it's removed too
				if cm.dim() == 2:
					cm = cm.unsqueeze(0)
				t = cm[-p:]
				if t.shape[0] < p:
					t = torch.cat([t[:1].repeat(p - t.shape[0], 1, 1), t], 0)
				return t.to(m.dtype)
			return torch.zeros((p, H, W), dtype=m.dtype)   # finished frames: nothing to remove
		return _fill_mask(p, "first_frame" if head_fill == "auto" else head_fill, m)


@register
class FrameUnpad(TiNode):
	DISPLAY_NAME = "Frame Unpad (ti)"
	CATEGORY = "tinode/video"

	@classmethod
	def INPUT_TYPES(cls):
		return {
			"required": {
				"images": ("IMAGE", {"tooltip": "The model's result on the padded clip."}),
			},
			"optional": {
				"pad_info": ("TI_FRAME_PAD", {"tooltip":
					"From Frame Pad — reverses exactly what it added (prepend / append "
					"/ both), whatever the side. The clean way to wire this."}),
				"pad_count": ("INT", {"default": 0, "min": 0, "max": 100000, "step": 1,
					"tooltip": "Manual HEAD-drop fallback when pad_info isn't wired."}),
				"expected_frames": ("INT", {"default": 0, "min": 0, "max": 9999999, "step": 1,
					"tooltip": "Recovered length (0 = take it from pad_info). A video "
							   "VAE can return MORE frames than it was given; this clips "
							   "the spill. With only pad_count, set it to the original "
							   "frame_count."}),
			},
		}

	RETURN_TYPES = ("IMAGE", "INT")
	RETURN_NAMES = ("images", "frame_count")
	FUNCTION = "execute"

	def execute(self, images, pad_info=None, pad_count=0, expected_frames=0):
		imgs = first(images)
		if imgs.dim() != 4:
			imgs = imgs.unsqueeze(0) if imgs.dim() == 3 else imgs
		N = imgs.shape[0]

		info = first(pad_info)
		want_override = int(first(expected_frames, 0) or 0)
		if isinstance(info, dict):
			head = int(info.get("head", 0))
			want = want_override or int(info.get("original", 0))
		else:
			head = int(first(pad_count, 0))
			want = want_override

		# Drop the head pad; the original content sits right after it for EVERY
		# side (prepend/append/both), so trimming to `want` from there also drops
		# any tail pad — one rule reverses all three ends.
		if head >= N:
			raise RuntimeError(
				f"Frame Unpad: head drop {head} >= {N} frames — nothing would be left. "
				"Feed the pad_info (or pad_count) that Frame Pad produced.")
		out = imgs[head:].contiguous() if head > 0 else imgs

		if want > 0 and out.shape[0] != want:
			if out.shape[0] < want:
				raise RuntimeError(
					f"Frame Unpad: got {out.shape[0]} frame(s) after dropping the head "
					f"but expected {want} — the model returned FEWER frames than the clip had.")
			print(f"[tinode] Frame Unpad: {out.shape[0]} frames after the head drop, "
				  f"trimming to the original {want}.")
			out = out[:want].contiguous()
		return (out, int(out.shape[0]))
