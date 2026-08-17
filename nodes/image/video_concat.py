"""Video Concatenate — append one native VIDEO after another with a hard cut.

Built for accumulating a VIDEO across an Inspire ▶Foreach List: wire `video_a`
to ForeachListBegin.intermediate_output and `video_b` to the clip this
iteration generated, then send the result to ForeachListEnd.intermediate_output.
After the last iteration ForeachListEnd.result is the whole thing, ready for one
SaveVideo.

Three things make that loop cheap and correct.

**Appending is lazy.** The node returns a ConcatenatedVideo holding an ordered
list of parts; nothing is decoded, copied or re-encoded until something asks for
pixels. Concatenating tensors on every iteration instead would re-copy the whole
accumulated clip once per step — O(N^2) memcpy over a recipe, with a 2x memory
spike each time. Here the single copy happens once, when SaveVideo materializes
the result.

**A hard cut, in every stream.** No interpolation, crossfade, or duplicated
transition frame: every output frame is some input frame, untouched. A part
whose frame rate differs from the accumulator's is retimed by nearest-neighbour
index mapping — frames repeat or drop, never blend — which preserves its
duration. Each part's audio is fitted to that part's own video duration before
joining (padded with silence, or trimmed), so a part with no audio, or with
audio that ran short, cannot shift everything after it out of sync.

**The loop's seed is tolerated.** ForeachListBegin feeds `initial_input`
through as the first `intermediate_output`, and it is not a VIDEO — whatever
you seeded the loop with (and if you seed nothing, Inspire quietly uses
item_list[0], i.e. your first recipe step). Anything that is not a VIDEO is
treated as "no accumulator yet" and `video_b` passes through untouched. See the
README for why you should still seed the loop explicitly.

The result is a native VideoInput: get_components() returns one VideoComponents
and save_to() hands the encode to core's VideoFromComponents, so SaveVideo — or
any other VIDEO consumer — sees an ordinary video. Nothing touches disk.
"""

from __future__ import annotations

import logging
from fractions import Fraction
from typing import NamedTuple

import torch

from ...base import TiNode
from ...registry import register

# ComfyUI's native VIDEO API. Imported defensively so the module (and its pure
# helpers) stay importable in the test runner, which does not put the ComfyUI
# root on sys.path.
try:
	from comfy_api.latest import Input, InputImpl, Types  # noqa: PLC0415

	VideoInput = Input.Video
	VideoFromComponents = InputImpl.VideoFromComponents
	VideoComponents = Types.VideoComponents
except Exception:  # noqa: BLE001 — no ComfyUI available (tests, linting)
	VideoInput = object
	VideoFromComponents = None
	VideoComponents = None


def is_video(value) -> bool:
	"""Is this a VIDEO object, as opposed to a loop's seed sentinel?

	Duck-typed rather than isinstance: other packs hand around VIDEO objects
	that implement the interface without subclassing VideoInput.
	"""
	return (
		value is not None
		and callable(getattr(value, "get_components", None))
		and callable(getattr(value, "save_to", None))
	)


def retime_indices(n_in: int, fps_in, fps_out) -> list[int] | None:
	"""Source frame index per output frame when moving n_in frames to fps_out.

	None when the rates already match, so the caller can skip the work
	entirely. Nearest-neighbour by design: an output frame is always exactly
	one source frame, so the cut stays hard and no frame is ever blended. The
	output length is chosen to preserve the part's duration, to within one
	output frame.
	"""
	if fps_in == fps_out:
		return None
	if n_in <= 0:
		return []
	n_out = max(1, int(round(n_in * float(fps_out) / float(fps_in))))
	# Sample at the centre of each output frame's interval, so the mapping is
	# symmetric and never favours the head of the clip.
	return [min(n_in - 1, int((i + 0.5) * n_in / n_out)) for i in range(n_out)]


def audio_sample_count(n_frames: int, frame_rate, sample_rate: int) -> int:
	"""Samples that cover exactly n_frames of video — the sync contract."""
	return int(round(n_frames * float(sample_rate) / float(frame_rate)))


def fit_channels(images: torch.Tensor) -> torch.Tensor:
	"""[N,H,W,C] -> [N,H,W,3]; the encoder is RGB.

	Alpha is dropped (core's VideoFromComponents ignores it when saving) and
	grey is expanded, so a stray RGBA or 1-channel part still joins cleanly
	instead of failing the concatenation.
	"""
	c = images.shape[-1]
	if c == 3:
		return images
	if c > 3:
		return images[..., :3]
	return images[..., :1].repeat(1, 1, 1, 3)


def fit_audio(waveform: torch.Tensor, samples: int, channels: int) -> torch.Tensor:
	"""One part's audio as [channels, samples] exactly.

	Short audio is padded with silence and long audio is trimmed, because video
	timing is what must not move: the alternative — letting a part's audio
	length decide the join point — desynchronises every part after it.
	"""
	w = waveform[0] if waveform.dim() == 3 else waveform
	if w.dim() == 1:
		w = w.unsqueeze(0)

	have = w.shape[0]
	if have == 0:
		return torch.zeros(channels, samples, dtype=torch.float32)
	if have > channels:
		w = w[:channels]
	elif have < channels:
		# Upmix by duplication (mono -> stereo), never by dropping a channel.
		w = w.repeat((channels + have - 1) // have, 1)[:channels]

	if w.shape[-1] > samples:
		return w[..., :samples].contiguous()
	if w.shape[-1] < samples:
		pad = w.new_zeros(w.shape[0], samples - w.shape[-1])
		return torch.cat([w, pad], dim=-1)
	return w.contiguous()


def resample_audio(waveform: torch.Tensor, from_rate: int, to_rate: int) -> torch.Tensor:
	"""Change an audio sample rate, so parts recorded differently can join."""
	if from_rate == to_rate:
		return waveform
	try:
		import torchaudio  # noqa: PLC0415

		return torchaudio.functional.resample(waveform, from_rate, to_rate)
	except Exception:  # noqa: BLE001 — torchaudio is optional in some installs
		n_out = max(1, int(round(waveform.shape[-1] * to_rate / from_rate)))
		return torch.nn.functional.interpolate(
			waveform.unsqueeze(0).float(), size=n_out, mode="linear", align_corners=False
		)[0]


class Combined(NamedTuple):
	"""The concatenation result, before it is wrapped as a VIDEO."""

	images: torch.Tensor
	audio: dict | None
	frame_rate: Fraction


def combine(components: list) -> Combined:
	"""Join a list of VideoComponents-likes (.images/.audio/.frame_rate).

	The first part sets the contract for the rest: its frame rate, its
	resolution, and (if it has audio) its sample rate. Kept free of any
	comfy_api import so it is testable on its own.
	"""
	parts = [c for c in components if c.images is not None and c.images.shape[0] > 0]
	if not parts:
		raise ValueError("Video Concatenate: no frames to concatenate.")

	frame_rate = parts[0].frame_rate
	if not frame_rate or float(frame_rate) <= 0:
		raise ValueError(f"Video Concatenate: first clip has an invalid frame rate ({frame_rate!r}).")
	height, width = parts[0].images.shape[1], parts[0].images.shape[2]

	# Audio contract: the first part that has audio sets the sample rate, and
	# the widest part sets the channel count (upmixing mono is lossless,
	# downmixing stereo is not). Silent parts are filled to match.
	with_audio = [c for c in parts if c.audio]
	sample_rate = int(with_audio[0].audio["sample_rate"]) if with_audio else None
	channels = 1
	for c in with_audio:
		w = c.audio["waveform"]
		channels = max(channels, (w[0] if w.dim() == 3 else w).shape[0])

	images: list[torch.Tensor] = []
	audio: list[torch.Tensor] = []

	for i, c in enumerate(parts):
		frames = c.images
		if frames.shape[1] != height or frames.shape[2] != width:
			raise ValueError(
				f"Video Concatenate: clip {i + 1} is "
				f"{frames.shape[2]}x{frames.shape[1]}, but the first clip is "
				f"{width}x{height}. Concatenation cannot rescale — put a resize "
				"or upscale node on the odd one out so every clip matches."
			)
		frames = fit_channels(frames)
		# CPU float32: a growing accumulator has no business sitting in VRAM,
		# and mixed dtypes cannot be concatenated.
		if frames.device.type != "cpu":
			frames = frames.cpu()
		if frames.dtype != torch.float32:
			frames = frames.float()

		idx = retime_indices(frames.shape[0], c.frame_rate, frame_rate)
		if idx is not None:
			logging.info(
				"[tinode] Video Concatenate: retiming clip %d from %s to %s fps "
				"(%d -> %d frames, hard cut, no blending).",
				i + 1, c.frame_rate, frame_rate, frames.shape[0], len(idx),
			)
			frames = frames[torch.tensor(idx, dtype=torch.long)]
		images.append(frames)

		if sample_rate is not None:
			want = audio_sample_count(frames.shape[0], frame_rate, sample_rate)
			if c.audio:
				w = c.audio["waveform"]
				w = w[0] if w.dim() == 3 else w
				w = resample_audio(w.float().cpu(), int(c.audio["sample_rate"]), sample_rate)
				audio.append(fit_audio(w, want, channels))
			else:
				# Silence, not a shorter clip: video timing must not move.
				audio.append(torch.zeros(channels, want, dtype=torch.float32))

	combined_images = images[0] if len(images) == 1 else torch.cat(images, dim=0)
	combined_audio = None
	if sample_rate is not None and audio:
		combined_audio = {
			"waveform": torch.cat(audio, dim=-1).unsqueeze(0),
			"sample_rate": sample_rate,
		}

	return Combined(images=combined_images, audio=combined_audio, frame_rate=frame_rate)


class ConcatenatedVideo(VideoInput):
	"""A native VIDEO that is the hard-cut join of an ordered list of VIDEOs.

	Lazy on purpose: `parts` is just pointers until something asks for pixels,
	which is what keeps an accumulate-in-a-loop pattern from being quadratic.
	The materialized components are cached, so get_dimensions() followed by
	save_to() (exactly what SaveVideo does) decodes once.
	"""

	def __init__(self, parts: list):
		self.parts = list(parts)
		self._components = None

	def _combined(self):
		if self._components is None:
			self._components = combine([p.get_components() for p in self.parts])
		return self._components

	# --- VideoInput contract ------------------------------------------------
	def get_components(self):
		if VideoComponents is None:
			raise RuntimeError(
				"Video Concatenate: ComfyUI's native VIDEO API "
				"(comfy_api.latest) is unavailable, so the joined clip cannot "
				"be handed on as a VIDEO."
			)
		c = self._combined()
		return VideoComponents(images=c.images, audio=c.audio, frame_rate=c.frame_rate)

	def save_to(self, path, format=None, codec=None, metadata=None, bit_depth=None, crf=None):
		"""Delegate the encode to core, so the output file is byte-for-byte
		whatever CreateVideo -> SaveVideo would have produced."""
		kwargs = {"metadata": metadata, "bit_depth": bit_depth, "crf": crf}
		if format is not None:
			kwargs["format"] = format
		if codec is not None:
			kwargs["codec"] = codec
		return self._as_video().save_to(path, **kwargs)

	def as_trimmed(self, start_time=None, duration=None, strict_duration=False):
		return self._as_video().as_trimmed(start_time, duration, strict_duration)

	def _as_video(self):
		return VideoFromComponents(self.get_components(), bit_depth=self.get_bit_depth())

	# --- cheap overrides: answerable without decoding anything -------------
	def get_frame_rate(self) -> Fraction:
		return Fraction(self.parts[0].get_frame_rate())

	def get_dimensions(self) -> tuple[int, int]:
		# Every part is required to share the first one's resolution.
		return self.parts[0].get_dimensions()

	def get_bit_depth(self) -> int:
		return max((p.get_bit_depth() for p in self.parts), default=8)

	def __repr__(self):
		return f"ConcatenatedVideo({len(self.parts)} parts)"


def flatten(video) -> list:
	"""One video as a list of parts, unwrapping an earlier concatenation.

	Without this, iteration N of a loop would nest N ConcatenatedVideos inside
	each other, and materializing would recurse N deep.
	"""
	return list(video.parts) if isinstance(video, ConcatenatedVideo) else [video]


@register
class VideoConcatenate(TiNode):
	DISPLAY_NAME = "Video Concatenate (ti)"
	CATEGORY = "tinode/video"

	@classmethod
	def INPUT_TYPES(cls):
		return {
			"required": {
				"video_b": ("VIDEO", {
					"tooltip": "The clip appended after video_a — this "
							"iteration's newly generated video.",
				}),
			},
			"optional": {
				"video_a": ("VIDEO", {
					"tooltip": "The clip that plays first — the accumulator. "
							"Wire ForeachListBegin.intermediate_output here. "
							"Left unconnected (or handed the loop's seed value) "
							"video_b passes straight through.",
				}),
			},
		}

	RETURN_TYPES = ("VIDEO",)
	RETURN_NAMES = ("video",)
	OUTPUT_TOOLTIPS = (
		"video_a followed by video_b, as a native VIDEO. Feed it back into "
		"ForeachListEnd.intermediate_output, and SaveVideo at the end.",
	)

	DESCRIPTION = (
		"Append video_b after video_a as one native VIDEO — hard cut, no "
		"crossfade, no re-encode until save.\n"
		"For accumulating a clip per iteration across Inspire's ▶Foreach List: "
		"video_a = ForeachListBegin.intermediate_output, result -> "
		"ForeachListEnd.intermediate_output, then one SaveVideo.\n"
		"video_b is retimed to video_a's frame rate if they differ; audio is "
		"kept in sync per clip, with silence filled in where a clip has none."
	)

	FUNCTION = "execute"

	def execute(self, video_b, video_a=None):
		if not is_video(video_b):
			raise ValueError(
				"Video Concatenate: video_b is not a VIDEO "
				f"(got {type(video_b).__name__}). It must be the clip generated "
				"this iteration, e.g. a Create Video output."
			)

		# Not a VIDEO means there is no accumulator yet: either video_a is
		# unconnected, or this is the loop's first pass and it carries whatever
		# seeded ForeachListBegin.initial_input.
		if not is_video(video_a):
			if video_a is not None:
				logging.info(
					"[tinode] Video Concatenate: video_a is %s, not a VIDEO — "
					"treating it as the loop seed and passing video_b through.",
					type(video_a).__name__,
				)
			return (video_b,)

		return (ConcatenatedVideo(flatten(video_a) + flatten(video_b)),)
