"""Noise Rotate — controlled descendants of a partial latent.

You stopped a generation part-way, looked at it, and want to see several ways it
could still go. A partial latent splits into what the model has already decided
and what is still undecided:

    x  =  x0_pred  +  residual

`x0_pred` is the model's current guess at the finished image — the part you
picked. The residual is the noise it has not resolved yet, and that is where the
remaining freedom lives. So: keep x0_pred, and turn the residual toward a
different answer.

    eps     = x - x0_pred
    new     = randn(variation_seed) * eps.std()
    eps_var = cos(theta) * eps + sin(theta) * new
    variant = x0_pred + eps_var

Rotating rather than adding is the whole point. `x + d*eps'` inflates the noise
level (by sqrt(1+d^2)), so the continuation would run a schedule that expects
less noise than it gets and quietly under-denoise — variation strength and
quality loss confounded in one dial. Because cos^2 + sin^2 = 1, this leaves
||x - x0_pred|| exactly unchanged: theta moves the direction, never the amount.
Scaling `new` to the residual's OWN magnitude rather than assuming a unit
variance is what keeps that true at any checkpoint, and means the node needs
neither sigma nor a latent-space conversion.

    theta = 0    exact continuation (the control)
    10-45        the useful working range
    90           fresh direction at the same noise level, x0_pred still kept

Measured on SD1.5: branch LATE. At sigma 4.86 (96% noise) x0_pred is barely
committed, so rotating re-rolls the composition rather than varying it. Deeper in
(sigma ~1.8 or below) the layout holds while the interpretation changes, which is
what "show me alternatives" actually means.

Variant i uses `variation_seed + i`, mirroring Seed Range Noise, so any single
descendant stays reproducible on its own.
"""

from __future__ import annotations

import math

import torch

from ...base import TiNode, first
from ...registry import register


def rotate_residual(x, x0, theta_deg: float, generator):
	"""One descendant: x0 plus the residual of `x` turned `theta_deg` degrees.

	Magnitude-preserving by construction, so the continuation still receives the
	noise level its schedule expects.
	"""
	eps = x - x0
	new = torch.randn(eps.shape, generator=generator, dtype=eps.dtype) * eps.std()
	t = math.radians(theta_deg)
	return x0 + (math.cos(t) * eps + math.sin(t) * new)


@register
class NoiseRotate(TiNode):
	DISPLAY_NAME = "Noise Rotate (ti)"
	CATEGORY = "tinode/sampling"

	@classmethod
	def INPUT_TYPES(cls):
		return {
			"required": {
				"latent": ("LATENT", {"tooltip":
					"`output` from the sampler that stopped at the checkpoint — "
					"the state still carrying its noise."}),
				"denoised": ("LATENT", {"tooltip":
					"`denoised_output` from that same sampler: what the model "
					"currently believes the image is becoming. This is preserved."}),
				"theta": ("FLOAT", {"default": 25.0, "min": 0.0, "max": 90.0,
					"step": 0.5, "tooltip":
					"How far to turn the unresolved noise. 0 = exact continuation, "
					"10-45 useful, 90 = a fresh direction at the same noise level."}),
				"count": ("INT", {"default": 4, "min": 1, "max": 64, "tooltip":
					"How many descendants. They come out as a batch — pick one "
					"with Candidate Select."}),
				"variation_seed": ("INT", {"default": 0, "min": 0,
					"max": 0xffffffffffffffff - 0xffff, "control_after_generate": True,
					"tooltip": "Descendant i is drawn from variation_seed + i."}),
			},
		}

	RETURN_TYPES = ("LATENT",)
	RETURN_NAMES = ("latents",)
	OUTPUT_TOOLTIPS = (
		"The descendants, as a batch, at the same noise level as the input.",
	)
	FUNCTION = "execute"

	def execute(self, latent, denoised, theta, count, variation_seed):
		x, x0 = latent["samples"], denoised["samples"]
		if x.shape != x0.shape:
			raise RuntimeError(
				f"Noise Rotate: `latent` is {tuple(x.shape)} but `denoised` is "
				f"{tuple(x0.shape)}. Both must come from the SAME sampler — wire "
				f"its `output` and `denoised_output`."
			)
		if x.shape[0] != 1:
			raise RuntimeError(
				f"Noise Rotate: expected one latent, got a batch of {x.shape[0]}. "
				f"Choose a candidate first (Candidate Select)."
			)
		if float(x.sub(x0).std()) == 0.0:
			# Nothing undecided left: at sigma 0 there is no residual to turn, and
			# every "variant" would be the same finished image.
			raise RuntimeError(
				"Noise Rotate: `latent` and `denoised` are identical — this "
				"checkpoint has no unresolved noise left to vary. Branch from an "
				"earlier point in the schedule."
			)

		theta = float(first(theta, 0.0))
		base = int(first(variation_seed, 0))
		variants = [
			rotate_residual(x, x0, theta,
							torch.Generator().manual_seed(base + i))
			for i in range(int(first(count, 1)))
		]

		out = latent.copy()
		out["samples"] = torch.cat(variants, dim=0)
		# The descendants are their own lineage now; the parent's batch position
		# would mislead anything that regenerates noise from them.
		out.pop("batch_index", None)
		print(f"[tinode] Noise Rotate: {len(variants)} descendant(s) "
			  f"at theta={theta:g}deg from seed {base}")
		return (out,)
