"""Prompt Template — assemble a prompt from slots, and survive the empty ones.

Six STRING inputs, one template, `{1}` .. `{6}` where they go:

    template : a {2} {1}, {3}, {4}
    1 = Ball bearing   2 = polished chrome   3 = golden hour   4 = still life
    ->  a polished chrome Ball bearing, golden hour, still life

The reason this exists rather than a string concat: slots go EMPTY all the time —
a list you did not wire, a random draw you turned off — and naive joining leaves
`a  Ball bearing, , golden hour,` behind. Here an empty slot takes its
surrounding punctuation with it, so a half-filled template still reads as a
sentence. Whatever survives is squeezed: repeated separators collapse, spaces
before commas go, and a stray separator at either end is trimmed.

`{{` and `}}` are literal braces, and a placeholder with no matching input is
left alone rather than silently deleted — a typo should be visible.
"""

from __future__ import annotations

import re

from ...base import TiNode, first
from ...registry import register

SLOTS = 6
# A slot plus the punctuation glued to it, so removing one removes its comma.
# Horizontal space only: a newline in the template is a deliberate line break
# (a second prompt line, a negative below a positive) and must survive.
_SLOT_RUN = re.compile(r"[ \t]*(?<!\{)\{(\d)\}(?!\})[ \t]*([,;:·—–\-/|]+[ \t]*)?")
_DOUBLE_SEP = re.compile(r"\s*([,;:·—–/|])\s*(?=[,;:·—–/|])")
_SPACE_BEFORE = re.compile(r"\s+([,;:.!?])")
_MULTISPACE = re.compile(r"[ \t]{2,}")
_EDGE_SEP = re.compile(r"^\s*[,;:·—–/|]+\s*|\s*[,;:·—–/|]+\s*$")


def fill_template(template, values):
	"""Substitute {1}..{N} from `values`, dropping empty slots cleanly."""
	text = str(template or "")
	if not text:
		# No template: just join whatever was wired, in order.
		return ", ".join(v for v in (str(x or "").strip() for x in values) if v)

	def sub(m):
		i = int(m.group(1))
		sep = m.group(2) or ""
		if i < 1 or i > len(values):
			return m.group(0)                 # unknown slot: leave it visible
		val = str(values[i - 1] or "").strip()
		if not val:
			return " " if sep else ""          # the slot AND its separator go
		return f" {val}{' ' if not sep else ''}{sep}"

	text = _SLOT_RUN.sub(sub, text)
	text = text.replace("{{", "\0L").replace("}}", "\0R")
	text = _DOUBLE_SEP.sub("", text)
	text = _SPACE_BEFORE.sub(r"\1", text)
	text = _MULTISPACE.sub(" ", text)
	text = "\n".join(_EDGE_SEP.sub("", line).strip() for line in text.split("\n"))
	return text.replace("\0L", "{").replace("\0R", "}").strip()


@register
class PromptTemplate(TiNode):
	DISPLAY_NAME = "Prompt Template (ti)"
	CATEGORY = "tinode/data"

	@classmethod
	def INPUT_TYPES(cls):
		return {
			"required": {
				"template": ("STRING", {
					"default": "a {1}, {2}, {3}, {4}", "multiline": True,
					"tooltip": "Where each wired string goes. {1}..{6}. An empty "
							   "slot takes its comma with it. {{ }} for literal "
							   "braces. Leave the template blank to just join "
							   "everything with commas."}),
			},
			"optional": {
				**{f"text_{i}": ("STRING", {"default": "", "forceInput": True,
					"tooltip": f"Fills {{{i}}}."}) for i in range(1, SLOTS + 1)},
				"prefix": ("STRING", {"default": "", "multiline": True,
					"tooltip": "Put in front of the result, if not empty."}),
				"suffix": ("STRING", {"default": "", "multiline": True,
					"tooltip": "Added after the result, if not empty."}),
			},
		}

	RETURN_TYPES = ("STRING",)
	RETURN_NAMES = ("prompt",)
	OUTPUT_TOOLTIPS = ("The assembled prompt.",)
	FUNCTION = "execute"

	def execute(self, template="", prefix="", suffix="", **slots):
		values = [str(first(slots.get(f"text_{i}"), "") or "")
				  for i in range(1, SLOTS + 1)]
		body = fill_template(first(template, ""), values)
		parts = [str(first(prefix, "") or "").strip(), body,
				 str(first(suffix, "") or "").strip()]
		out = ", ".join(p for p in parts if p)
		print(f"[tinode] Prompt Template: {out}")
		return (out,)
