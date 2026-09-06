"""Sigma Segment — one absolute slice of an immutable sigma schedule.

Why this exists: a staged workflow runs 0->4, then 4->9, then 9->20, and each
stage needs its own piece of the schedule. Core's SplitSigmas does that by
cutting a schedule in two, so chaining it means each stage cuts the LEFTOVER of
the previous one — and the step numbers stop being the ones the user typed.
Splitting a 20-step schedule at 4 and then wanting absolute step 9 requires
asking for relative index 5. That off-by-N is invisible until an image comes out
wrong.

Instead: build the full schedule ONCE and let every stage name absolute steps
against it. Stage boundaries stay the numbers on screen, and nothing has to know
what the stages before it did.

    sigmas[start_step : end_step + 1]

The +1 keeps the boundary sigma in the segment. Adjacent stages therefore SHARE
their boundary value (0->4 ends on the same sigma 4->9 starts on), which is what
makes the hand-off seamless: the second stage is told the noise level it is
resuming at, not just the remaining itinerary. A segment holds
`end_step - start_step` steps.

Invalid ranges raise instead of clamping. A silently clamped range yields a
plausible image from the wrong part of the trajectory, which is far more
expensive to notice than an error.
"""

from __future__ import annotations

from ...base import TiNode
from ...registry import register


def slice_sigmas(sigmas, start_step: int, end_step: int):
	"""`sigmas[start:end+1]`, or raise ValueError explaining the range.

	`sigmas` holds one more entry than the schedule has steps (the terminal
	sigma), so the last addressable step index is `len(sigmas) - 1`.
	"""
	last = len(sigmas) - 1
	if start_step < 0:
		raise ValueError(f"start_step must be >= 0, got {start_step}.")
	if end_step < start_step:
		raise ValueError(
			f"end_step ({end_step}) is before start_step ({start_step}); a segment "
			f"runs forwards through the schedule. Did the two get swapped?"
		)
	if end_step == start_step:
		# A single sigma is zero denoising steps: the sampler would hand the
		# latent straight back and the stage would look like it did nothing.
		raise ValueError(
			f"start_step and end_step are both {start_step}: a segment of one sigma "
			f"performs no sampling. Give the stage at least one step."
		)
	if end_step > last:
		raise ValueError(
			f"end_step ({end_step}) is past the end of this schedule: it has "
			f"{last} steps ({len(sigmas)} sigmas), so the highest valid step is {last}. "
			f"Raise the scheduler's step count, or lower end_step."
		)
	return sigmas[start_step:end_step + 1]


@register
class SigmaSegment(TiNode):
	DISPLAY_NAME = "Sigma Segment (ti)"
	CATEGORY = "tinode/sampling"

	@classmethod
	def INPUT_TYPES(cls):
		return {
			"required": {
				"sigmas": ("SIGMAS",),
				"start_step": ("INT", {"default": 0, "min": 0, "max": 10000}),
				"end_step": ("INT", {"default": 4, "min": 0, "max": 10000}),
			},
		}

	# start_sigma is informational: it is the noise level this stage resumes at,
	# and seeing it on the canvas is what makes an abstract step number concrete.
	RETURN_TYPES = ("SIGMAS", "FLOAT")
	RETURN_NAMES = ("sigmas", "start_sigma")
	FUNCTION = "execute"

	def execute(self, sigmas, start_step, end_step):
		seg = slice_sigmas(sigmas, start_step, end_step)
		return (seg, float(seg[0]))
