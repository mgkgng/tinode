from ...base import TiNode
from ...registry import register


@register
class ClipTextEncodeOverride(TiNode):
	NODE_ID = "ClipTextEncodeOverride"
	DISPLAY_NAME = "CLIP Text Encode (Override) (ti)"
	CATEGORY = "tinode/conditioning"

	@classmethod
	def INPUT_TYPES(cls):
		return {
			"required": {
				"clip": ("CLIP",),
				"text": ("STRING", {"multiline": True, "dynamicPrompts": True, "default": ""}),
			},
			"optional": {
				"text_override": ("STRING", {"forceInput": True}),
			},
		}

	RETURN_TYPES = ("CONDITIONING",)
	RETURN_NAMES = ("conditioning",)
	FUNCTION = "execute"

	@classmethod
	def IS_CHANGED(cls, clip, text, text_override=None):
		return text_override if text_override else text

	def execute(self, clip, text, text_override=None):
		if clip is None:
			raise RuntimeError("clip input is None — connect a CLIP model.")
		prompt = text_override if text_override else text
		tokens = clip.tokenize(prompt)
		return (clip.encode_from_tokens_scheduled(tokens),)
