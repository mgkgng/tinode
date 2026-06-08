"""Trivial working node — proves the scaffold loads end to end.

Drop the pack into custom_nodes/, restart ComfyUI, right-click ->
Add Node -> tinode/example -> "String Passthrough". If it shows up and
runs, auto-discovery + registration work. Delete this file once you have
real nodes.
"""

from ...base import TiNode
from ...registry import register


@register
class StringPassthrough(TiNode):
	DISPLAY_NAME = "String Passthrough (ti)"
	CATEGORY = "tinode/example"

	@classmethod
	def INPUT_TYPES(cls):
		return {
			"required": {
				"text": ("STRING", {"multiline": True, "default": ""}),
			}
		}

	RETURN_TYPES = ("STRING",)
	RETURN_NAMES = ("text",)
	FUNCTION = "execute"

	def execute(self, text):
		return (text,)
