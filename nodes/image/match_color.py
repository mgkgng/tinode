"""Match Crop Colour — undo a model's colour drift before pasting it back.

A generative fill re-renders the WHOLE crop, and its output drifts: measured on
VOID, 88.8% of the pixels it was asked to LEAVE ALONE came back different, with
a mean shift of about -1.4 / -0.1 / -1.3 per channel. Pasting that fill in puts
a tinted patch into an otherwise untouched frame, which reads as a visible tile
however good the inpainting itself is.

The fix is that the drift is measurable. Outside the mask we know exactly what
the pixels should be — they are the original crop — so comparing the model's
version of those pixels against the real ones gives the correction, and applying
it to the whole crop brings the masked region back in line too.

  reference  the original crop (ground truth)
  target     the model's output for that crop
  mask       what was inpainted — EXCLUDED from the statistics, since those
             pixels are supposed to differ

method:
  mean       shift each channel so the unmasked means agree. Safest: it cannot
             change contrast, only level.
  mean_std   also scale so the spreads agree, for a fill that is flat or harsh
             as well as tinted. Stronger, and it can amplify noise.
  none       pass through, to A/B the correction.

`band` restricts the statistics to a ring around the mask rather than the whole
crop, which matches the fill to what it actually touches — better when lighting
varies across the crop, worse when the ring is too thin to be representative.

`temporal` decides whether each frame gets its own correction or the clip shares
one. The drift is NOT constant over a shot: measured on VOID it swung by 6-7
levels from the first frame to the last, and its spatial spread doubled wherever
the content changed hard — so one clip-wide correction is right on average and
increasingly wrong at the ends, which shows up as a tile that grows as the clip
plays and jumps when something crosses the camera. per_frame therefore tracks
it and is the default; `smoothed` averages the field over `smooth_frames` if a
per-frame estimate ever shimmers; whole_clip is kept for a genuinely static
shot.
"""

from __future__ import annotations

import torch

from ...base import TiNode, first
from ...registry import register

_METHOD = ["low_freq", "mean", "mean_std", "none"]
_TEMPORAL = ["per_frame", "smoothed", "whole_clip"]


def _blur(x, sigma):
	"""Separable Gaussian blur of an [N,C,H,W] tensor.

	sigma is floored: a zero or negative sigma makes the kernel divide by zero and
	the whole field comes back NaN, which is far worse than no correction.
	"""
	import torch.nn.functional as F  # noqa: PLC0415

	sigma = max(0.5, float(sigma))
	r = max(1, int(round(sigma * 3)))
	xs = torch.arange(-r, r + 1, dtype=x.dtype, device=x.device)
	k = torch.exp(-(xs * xs) / (2 * sigma * sigma))
	k = (k / k.sum()).view(1, 1, 1, -1).repeat(x.shape[1], 1, 1, 1)
	x = F.conv2d(F.pad(x, (r, r, 0, 0), mode="replicate"), k, groups=x.shape[1])
	k = k.view(x.shape[1], 1, -1, 1)
	return F.conv2d(F.pad(x, (0, 0, r, r), mode="replicate"), k, groups=x.shape[1])


def low_freq_field(ref, tgt, keep, sigma):
	"""Smooth per-pixel correction, measured only where `keep` is true.

	Normalized convolution: blur (difference x keep) and divide by blur(keep), so
	masked pixels contribute nothing and the field is EXTRAPOLATED across the
	hole from its surroundings rather than being pulled toward the fill.
	"""
	d = (ref - tgt).permute(0, 3, 1, 2)                   # [N,C,H,W]
	w = keep.unsqueeze(1).to(d.dtype)                     # [N,1,H,W]
	num = _blur(d * w, sigma)
	den = _blur(w.repeat(1, d.shape[1], 1, 1), sigma).clamp_min(1e-6)
	return (num / den).permute(0, 2, 3, 1)                # [N,H,W,C]


def _temporal_smooth(field, window):
	"""Centred moving average of a per-frame field, so it tracks the drift
	without inheriting frame-to-frame noise."""
	n = field.shape[0]
	if window <= 1 or n <= 1:
		return field
	half = max(1, int(window) // 2)
	out = torch.empty_like(field)
	for i in range(n):
		lo, hi = max(0, i - half), min(n, i + half + 1)
		out[i] = field[lo:hi].mean(0)
	return out


def _ring(mask, band):
	"""A `band`-pixel ring just outside the mask, as a bool tensor."""
	import torch.nn.functional as F  # noqa: PLC0415

	m = (mask > 0.5).float().unsqueeze(1)                 # [N,1,H,W]
	k = int(band) * 2 + 1
	grown = F.max_pool2d(m, kernel_size=k, stride=1, padding=int(band))
	return ((grown - m) > 0.5).squeeze(1)                 # grown minus the mask itself


def correction(ref, tgt, sample, method):
	"""Per-channel (gain, offset) mapping tgt onto ref over `sample` pixels."""
	c = ref.shape[-1]
	gain = torch.ones(c, dtype=torch.float32)
	off = torch.zeros(c, dtype=torch.float32)
	if not sample.any():
		return gain, off                                   # nothing to learn from
	r = ref[sample]                                        # [P,C]
	t = tgt[sample]
	rm, tm = r.mean(0), t.mean(0)
	if method == "mean_std":
		rs, ts = r.std(0), t.std(0)
		gain = torch.where(ts > 1e-6, rs / ts, torch.ones_like(ts))
		off = rm - gain * tm
	else:
		off = rm - tm
	return gain, off


@register
class MatchCropColor(TiNode):
	DISPLAY_NAME = "Match Crop Colour (ti)"
	CATEGORY = "tinode/image"

	@classmethod
	def INPUT_TYPES(cls):
		return {
			"required": {
				"reference": ("IMAGE", {"tooltip":
					"The ORIGINAL crop — the pixels the model was supposed to keep."}),
				"target": ("IMAGE", {"tooltip": "The model's output for that crop."}),
			},
			"optional": {
				"mask": ("MASK", {"tooltip":
					"The inpainted region. Excluded from the statistics: those "
					"pixels are meant to differ, so measuring them would fold the "
					"removal itself into the correction."}),
				"method": (_METHOD, {"default": "mean"}),
				"temporal": (_TEMPORAL, {"default": "per_frame", "tooltip":
					"per_frame follows the drift, which changes through a shot "
					"(measured: 6-7 levels start to end). smoothed averages the "
					"field over smooth_frames to kill any shimmer. whole_clip is "
					"one correction for the whole clip — only right for a static "
					"shot, and the cause of a tile that worsens as the clip plays."}),
				"sigma": ("FLOAT", {"default": 20.0, "min": 1.0, "max": 400.0, "step": 1.0,
					"tooltip": "Smoothness of the low_freq field, in pixels. Big "
							   "enough to carry shading but not the picture. Measured "
							   "on VOID: 40 leaves a 9-level residual, 20 leaves 4.8, "
							   "10 leaves 1.8 — lower it if a tile is still visible, "
							   "raise it if the correction starts eating detail."}),
				"band": ("INT", {"default": 0, "min": 0, "max": 512, "step": 1,
					"tooltip": "Measure only this many pixels around the mask "
							   "instead of the whole crop. 0 = whole crop."}),
				"strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.05,
					"tooltip": "Blend toward the correction. 1 = full, 0 = off."}),
				# APPENDED last on purpose: widget values are positional, so
				# inserting one mid-list silently shifts every value after it in
				# already-saved workflows — which turned sigma into 0 and made the
				# blur produce NaN.
				"smooth_frames": ("INT", {"default": 9, "min": 1, "max": 999, "step": 2,
					"tooltip": "Temporal window for `smoothed`, in frames."}),
			},
		}

	RETURN_TYPES = ("IMAGE", "STRING")
	RETURN_NAMES = ("images", "report")
	OUTPUT_TOOLTIPS = (
		"The target with the drift removed.",
		"The measured shift, so you can see how far off the model was.",
	)
	FUNCTION = "execute"

	def execute(self, reference, target, mask=None, method="low_freq",
				temporal="per_frame", band=0, strength=1.0, sigma=20.0,
				smooth_frames=9):
		ref = first(reference)
		tgt = first(target)
		for nm, t in (("reference", ref), ("target", tgt)):
			if not isinstance(t, torch.Tensor):
				raise RuntimeError(f"Match Crop Colour: `{nm}` must be an IMAGE.")
		if ref.dim() == 3:
			ref = ref.unsqueeze(0)
		if tgt.dim() == 3:
			tgt = tgt.unsqueeze(0)
		method = str(first(method, "mean"))
		temporal = str(first(temporal, "per_frame"))
		band = int(first(band, 0))
		strength = float(first(strength, 1.0))
		if method == "none" or strength <= 0:
			return (tgt, "off")

		n = min(ref.shape[0], tgt.shape[0])
		ref, tgt = ref[:n], tgt[:n]
		if ref.shape[1:3] != tgt.shape[1:3]:
			raise RuntimeError(
				f"Match Crop Colour: reference is {tuple(ref.shape[1:3])} but target "
				f"is {tuple(tgt.shape[1:3])} — they must be the same crop.")

		m = first(mask)
		if isinstance(m, torch.Tensor):
			if m.dim() == 2:
				m = m.unsqueeze(0)
			m = m[:n] if m.shape[0] >= n else m[:1].repeat(n, 1, 1)
			keep = _ring(m, band) if band > 0 else (m <= 0.5)
		else:
			keep = torch.ones(tgt.shape[:3], dtype=torch.bool)

		if method == "low_freq":
			field = low_freq_field(ref, tgt, keep, float(first(sigma, 20.0)))
			if temporal == "whole_clip":
				field = field.mean(0, keepdim=True).expand_as(field)
			elif temporal == "smoothed":
				field = _temporal_smooth(field, int(first(smooth_frames, 9)))
			if not torch.isfinite(field).all():
				raise RuntimeError(
					"Match Crop Colour: the correction field came out non-finite "
					f"(sigma={float(first(sigma, 20.0))!r}). Check that sigma is a "
					"sensible pixel radius.")
			out = (tgt + field * strength).clamp(0, 1)
			off = field.reshape(-1, field.shape[-1]).mean(0)
			rep = (f"low_freq/{temporal} sigma={float(first(sigma,20.0)):g}"
				   " mean offset(RGB)=" + ",".join(f"{v * 255:+.2f}" for v in off.tolist()))
			print(f"[tinode] Match Crop Colour: {rep}")
			return (out.contiguous(), rep)

		out = tgt.clone()
		if temporal == "per_frame":
			shifts = []
			for i in range(n):
				g, o = correction(ref[i], tgt[i], keep[i], method)
				g, o = 1 + (g - 1) * strength, o * strength
				out[i] = (tgt[i] * g + o).clamp(0, 1)
				shifts.append(o)
			off = torch.stack(shifts).mean(0)
			gain = torch.tensor([float("nan")])
		else:
			gain, off = correction(ref, tgt, keep, method)
			gain, off = 1 + (gain - 1) * strength, off * strength
			out = (tgt * gain + off).clamp(0, 1)

		rep = (f"{method}/{temporal}"
			   + (f" band={band}" if band else "")
			   + " offset(RGB)=" + ",".join(f"{v * 255:+.2f}" for v in off.tolist())
			   + (" gain=" + ",".join(f"{v:.4f}" for v in gain.tolist())
				  if method == "mean_std" and temporal == "whole_clip" else ""))
		print(f"[tinode] Match Crop Colour: {rep}")
		return (out.contiguous(), rep)
