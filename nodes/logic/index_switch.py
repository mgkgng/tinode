"""Index Switch — pick the Nth connected input, with truly dynamic inputs.

The node starts with one input. The JS (web/index_switch.js) calls addInput
when the last slot gets connected and removes the trailing empties when you
disconnect — so the slot count grows/shrinks with what you actually wire.
No pre-declared ghost slots.

For that to work the backend must accept inputs it never declared: that's
what VALIDATE_INPUTS(**kwargs)->True does (the same escape hatch rgthree /
custom-scripts use). execute() catches every wired slot via **kwargs.

`index` selects among the connected inputs in slot order, and is CLAMPED to
range — out-of-range never errors (negative -> first, too big -> last).
"""

from __future__ import annotations

from ...base import TiNode
from ...registry import register


def _gather(kwargs: dict, prefix: str) -> list:
	"""Connected inputs (non-None) in numeric slot order: prefix1, prefix2…"""
	pairs = []
	for k, v in kwargs.items():
		if v is None or not k.startswith(prefix):
			continue
		try:
			n = int(k[len(prefix):])
		except ValueError:
			continue
		pairs.append((n, v))
	pairs.sort(key=lambda p: p[0])
	return [v for _, v in pairs]


def _clamp_pick(items: list, index: int):
	if not items:
		return None
	return items[max(0, min(int(index), len(items) - 1))]


@register
class ImageIndexSwitch(TiNode):
	DISPLAY_NAME = "Image Index Switch (ti)"
	CATEGORY = "tinode/logic"

	@classmethod
	def INPUT_TYPES(cls):
		# Only the first slot is declared; the JS adds image2, image3, … live.
		return {
			"required": {"index": ("INT", {"default": 0, "min": 0, "max": 9999})},
			"optional": {"image1": ("IMAGE",)},
		}

	@classmethod
	def VALIDATE_INPUTS(cls, **kwargs):
		# Accept the dynamically-added slots the backend never declared.
		return True

	RETURN_TYPES = ("IMAGE",)
	RETURN_NAMES = ("image",)
	FUNCTION = "execute"

	def execute(self, index=0, **kwargs):
		out = _clamp_pick(_gather(kwargs, "image"), index)
		if out is None:
			raise ValueError("Image Index Switch: no images connected")
		return (out,)


@register
class StringIndexSwitch(TiNode):
	DISPLAY_NAME = "String Index Switch (ti)"
	CATEGORY = "tinode/logic"

	@classmethod
	def INPUT_TYPES(cls):
		return {
			"required": {"index": ("INT", {"default": 0, "min": 0, "max": 9999})},
			"optional": {"string1": ("STRING", {"forceInput": True})},
		}

	@classmethod
	def VALIDATE_INPUTS(cls, **kwargs):
		return True

	RETURN_TYPES = ("STRING",)
	RETURN_NAMES = ("string",)
	FUNCTION = "execute"

	def execute(self, index=0, **kwargs):
		out = _clamp_pick(_gather(kwargs, "string"), index)
		return (out if out is not None else "",)
