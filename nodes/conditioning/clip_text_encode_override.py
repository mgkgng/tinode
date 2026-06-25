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
				"text": ("STRING", {"forceInput": True}),
			},
		}

	RETURN_TYPES = ("CONDITIONING",)
	RETURN_NAMES = ("conditioning",)
	FUNCTION = "execute"

	@classmethod
	def IS_CHANGED(cls, clip, text):
		return text

	def execute(self, clip, text):
		if clip is None:
			raise RuntimeError("clip input is None — connect a CLIP model.")
		tokens = clip.tokenize(text)
		return (clip.encode_from_tokens_scheduled(tokens),)
