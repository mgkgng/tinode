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

theta is an angle, and it is exactly the cosine similarity between the old
residual and the new one: cos_sim(eps, eps_var) = cos(theta). That identity holds
because `new` is drawn independently of `eps`, and two independent gaussians in
D dimensions are near-orthogonal (cos ~ 1/sqrt(D), about 0.6% at 4x113x64). So
the number on the widget is the angle actually turned, to within a fraction of a
degree — and it gets more exact at higher resolution, not less.

    theta = 0    exact continuation (the control)
    10-45        the useful working range
    90           fresh direction at the same noise level, x0_pred still kept

theta steers TWO things at once, and they are not the same number. The angle
from the parent is theta. The angle between two SIBLINGS is wider — at the
default spread of 90:

    angle(variant_i, variant_j) = acos(cos^2 theta)

    theta:           0    10     30     45     60     90    120    150    180
    from parent:   0.0  10.1   30.2   45.3   60.2   90.0  119.8  149.8  180.0
    between sibs:  0.0  14.3   41.8   60.4   75.9   90.4   76.0   41.8    0.0

Siblings are always further from each other than from the parent, and their
spread PEAKS at 90 and then closes again. That is the real reason 90 is the
working maximum: past it the descendants keep marching away from the parent
while collapsing back together, and at 180 the term sin(theta)*new vanishes
entirely — `variation_seed` stops mattering, every descendant is the same
image, and that image is x0_pred minus the residual, the exact negative of the
continuation. At 360 you are back at the original. Values beyond 90 are allowed
because they are interesting to look at, not because they produce more variety.

`spread` separates those two distances. Without it, how different the children
are from EACH OTHER is locked to how far they are from the parent, and there is
no way to ask for "move somewhere new, but keep this family tight". With it,
every child still sits exactly theta from the parent while spread decides how
widely they fan out around a shared family direction:

    spread ->       0      10      20      30      45      60      75      90
    theta  25      0.0     5.9    11.7    17.2    24.4    30.0    33.6    34.8
    theta  45      0.0    10.0    19.7    29.0    41.4    51.3    57.8    60.0
    theta  60      0.0    12.2    24.2    35.7    51.3    64.1    72.5    75.5
                                    (sibling angle in degrees)

spread = 90 is exactly what this node did before it existed, which is why it is
the default. spread = 0 collapses the family to one image repeated `count`
times. Above 90 is a mirror (100 behaves as 80), so the range stops there —
unlike theta, where past 90 is genuinely different.

Note the ceiling is set by theta: spread can only distribute the room theta has
opened. At theta 25 the children can never be more than 34.8 degrees apart.

Negative angles are a free extra axis: cos is even and sin is odd, so -theta has
the identical strength as +theta but is a different descendant.

Measured on SD1.5: branch LATE. At sigma 4.86 (96% noise) x0_pred is barely
committed, so rotating re-rolls the composition rather than varying it. Deeper in
(sigma ~1.8 or below) the layout holds while the interpretation changes, which is
what "show me alternatives" actually means.

`keep_parent` answers a different question from either dial: "this state is
already good — continue it, but show me alternatives too". Every descendant sits
at the SAME theta from the parent, so a family cannot contain both the faithful
continuation and variations of it; two children each theta from the parent can
differ by at most 2*theta, which is why theta=0 makes them all identical no
matter what spread says. That is a fact about distances, not about this code.
`keep_parent` sidesteps it by letting ONE child sit at a different distance —
zero — instead of trying to make zero-distance children differ. Descendant 1 is
then the parent's own continuation, bit-exact, sharing a batch with its
alternatives so they can be judged against each other rather than from memory.

It REPLACES the first descendant instead of prepending one, so descendants
2..count keep the seeds and indices they had with the toggle off.

Variant i uses `variation_seed + i`, and the shared family direction is derived
from the same integer — so one number still addresses the whole set. Note the
pair (variation_seed, index) is what identifies a descendant, not a seed on its
own: because the family direction is shared, a solo run at `variation_seed + i`
builds a DIFFERENT family. Child i is stable under changing `count`, which is
what makes the index a durable address. Reproducing a descendant needs the
parent latent and theta regardless, so a standalone seed never sufficed here.
"""

from __future__ import annotations

import math

import torch

from ...base import TiNode, first
from ...registry import register


# An odd 64-bit constant (splitmix64's) used to derive the family seed from the
# same integer the children come from. One number still addresses the whole
# family, and the derived value cannot land on a child's own seed.
_FAMILY_MIX = 0x9E3779B97F4A7C15
_U64 = 0xFFFFFFFFFFFFFFFF


def _family_seed(base: int) -> int:
	"""A seed for the shared direction, derived from — but never equal to — a child's."""
	fam = ((base + 1) * _FAMILY_MIX) & _U64
	if base <= fam <= base + 4096:
		# Cannot happen in practice, but a seed collision is exactly the bug this
		# node exists to avoid, so refuse to rely on luck.
		fam ^= 0xA5A5A5A5A5A5A5A5
	return fam


def _project_out(x, a):
	"""`x` with its component along `a` removed: one step of Gram-Schmidt.

	Two independent gaussians in D dimensions are ALMOST perpendicular already
	(cos ~ 1/sqrt(D), 0.7% at 4x113x64), which is why a single rotation was
	honest to about 0.2 degrees without this. `spread` composes two rotations
	and is used at small angles, where that same 0.2 degrees is a tenth of the
	value on the dial — so here it is made exact instead of nearly-exact.
	"""
	xf, af = x.flatten().double(), a.flatten().double()
	return x - float(xf @ af / (af @ af)) * a


def rotate_family(x, x0, theta_deg: float, spread_deg: float, count: int, base_seed: int,
				  keep_parent: bool = False):
	"""`count` descendants, each exactly `theta_deg` from the parent.

	    v      one shared family direction, perpendicular to the residual
	    q_i    each child's own direction, perpendicular to both
	    w_i =  cos(spread)*v + sin(spread)*q_i
	    d_i =  cos(theta)*eps + sin(theta)*w_i

	Because every w_i is perpendicular to eps and unit-scaled to it, ||d_i|| ==
	||eps|| exactly and d_i . eps == cos(theta) for every child — so `spread`
	moves the children around each other WITHOUT moving them nearer or further
	from the parent. That separation is the whole point: theta says how far the
	family travels, spread says how far apart its members are.

	`keep_parent` REPLACES child 0 with the parent itself, rather than inserting
	it. Inserting would shift every other child one place along, and an index is
	an address here — Candidate Select reports `seed + index`, and a descendant
	is identified by (variation_seed, index). Replacing leaves children 1..n-1
	byte-identical to what the same seed produces with the flag off, so turning
	the toggle on does not silently rename anything you already chose.
	"""
	eps = x - x0
	scale = eps.std()
	t, p = math.radians(theta_deg), math.radians(spread_deg)

	def draw(seed):
		return torch.randn(eps.shape, dtype=eps.dtype,
						   generator=torch.Generator().manual_seed(int(seed) & _U64))

	v = _project_out(draw(_family_seed(base_seed)), eps)
	v = v / v.std() * scale

	out = []
	for i in range(count):
		if keep_parent and i == 0:
			# x, not x0 + eps: the same tensor the sampler handed us, so this is
			# the parent bit-for-bit and not a reconstruction of it.
			out.append(x.clone())
			continue
		q = _project_out(_project_out(draw(base_seed + i), eps), v)
		q = q / q.std() * scale
		w = math.cos(p) * v + math.sin(p) * q
		out.append(x0 + (math.cos(t) * eps + math.sin(t) * w))
	return out


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
				# The full circle is allowed, not just the useful quadrant: 90 is
				# where sibling spread peaks, but seeing what lies past it is
				# worth more than a guard rail. See the module docstring.
				"theta": ("FLOAT", {"default": 25.0, "min": -360.0, "max": 360.0,
					"step": 0.5, "tooltip":
					"Angle to turn the unresolved noise — literally the cosine "
					"similarity to the original residual, cos_sim = cos(theta). "
					"0 = exact continuation, 10-45 useful, 90 = uncorrelated "
					"(and the widest spread between descendants). Past 90 the "
					"descendants move further from the parent but back TOWARD "
					"each other; at 180 the variation seed stops mattering and "
					"they collapse to one image, the negative of the "
					"continuation. -theta mirrors +theta at equal strength."}),
				"count": ("INT", {"default": 4, "min": 1, "max": 64, "tooltip":
					"How many descendants. They come out as a batch — pick one "
					"with Candidate Select."}),
				"variation_seed": ("INT", {"default": 0, "min": 0,
					"max": 0xffffffffffffffff - 0xffff, "control_after_generate": True,
					"tooltip": "Descendant i is drawn from variation_seed + i. The "
							   "shared family direction is derived from the same "
							   "number, so one integer still addresses the whole set."}),
				# Declared LAST on purpose: widgets_values is positional, so adding
				# a widget in the middle would silently re-map every saved graph.
				"spread": ("FLOAT", {"default": 90.0, "min": 0.0, "max": 90.0,
					"step": 0.5, "tooltip":
					"How far apart the descendants are FROM EACH OTHER, at the "
					"same distance from the parent. 90 = independent (the "
					"default, and what this node always did); 0 = every "
					"descendant identical, so `count` costs you N renders of one "
					"image. The sibling angle it produces is capped by theta: at "
					"theta 25 they can never be more than 34.8 degrees apart."}),
				# Also last, and for the same positional reason as `spread`.
				"keep_parent": ("BOOLEAN", {"default": False,
					"label_on": "Yes", "label_off": "No", "tooltip":
					"Make the FIRST descendant the parent's own continuation, "
					"unchanged, and vary the rest. Use it when the current state "
					"is already good: you keep it in the same batch as its "
					"alternatives, so Candidate Select compares them side by "
					"side instead of you re-running the branch to get it back. "
					"It replaces descendant 1 rather than adding one, so the "
					"others keep the seeds and indices they already had."}),
			},
		}

	RETURN_TYPES = ("LATENT",)
	RETURN_NAMES = ("latents",)
	OUTPUT_TOOLTIPS = (
		"The descendants, as a batch, at the same noise level as the input.",
	)
	FUNCTION = "execute"

	def execute(self, latent, denoised, theta, count, variation_seed, spread=90.0,
				keep_parent=False):
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
		spread = float(first(spread, 90.0))
		base = int(first(variation_seed, 0))
		n = int(first(count, 1))
		keep = bool(first(keep_parent, False))
		if n > 1 and spread < 1.0:
			print(f"[tinode] Noise Rotate: spread={spread:g} — {n} descendants will be "
				  f"near-identical; you are paying {n} renders for one image.")
		if keep and n == 1:
			# Not an error — the graph still runs — but the node has been asked
			# for one descendant and told to make it the parent, so it is doing
			# nothing at all and the user almost certainly meant count > 1.
			print("[tinode] Noise Rotate: keep_parent with count=1 — the only "
				  "descendant IS the parent, so nothing varies. Raise count.")
		variants = rotate_family(x, x0, theta, spread, n, base, keep)

		out = latent.copy()
		out["samples"] = torch.cat(variants, dim=0)
		# The descendants are their own lineage now; the parent's batch position
		# would mislead anything that regenerates noise from them.
		out.pop("batch_index", None)
		kept = " (descendant 1 is the parent)" if keep else ""
		print(f"[tinode] Noise Rotate: {len(variants)} descendant(s) at "
			  f"theta={theta:g}deg spread={spread:g}deg from seed {base}{kept}")
		return (out,)
