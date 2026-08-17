"""JSON To Item List — split a JSON array into one item per element.

`[{"a": 1}, {"a": 2}]` in, two items out. Feed `item_list` straight into
Inspire's ▶Foreach List to run a sub-workflow once per object.

Why an ITEM_LIST and not just a ComfyUI list: Inspire's Worklist To Item List
takes the OPPOSITE route — it declares `INPUT_IS_LIST` and collapses a batch
that ComfyUI already ran per-item into a single ITEM_LIST value. That needs an
upstream node that emits a list in the first place. Here the whole list is one
string, so there is nothing to collapse: we build the same plain-Python-list
payload ITEM_LIST carries and hand it over directly. ForeachListBegin only ever
indexes and slices it (`item_list[0]`, `item_list[1:]`) and wraps the remainder
itself, so no Inspire import — and no dependency — is needed.

Both idioms come out anyway:
  item_list — one ITEM_LIST value, for ▶Foreach List (sequential, accumulates)
  items     — a ComfyUI list, so every downstream node runs once per item

Items are emitted as **strings**, because that is what a ComfyUI socket can
actually carry: objects and arrays are re-serialized as compact JSON (matching
what the simple-json / JSON-utility packs expect), while a JSON string element
passes through unquoted so `["cat", "dog"]` yields `cat` and `dog`, not
`"cat"`.
"""

from __future__ import annotations

import json

from ...base import TiNode
from ...registry import register


def parse_items(text: str) -> list:
	"""Parse `text` into the list of raw (still-Python) elements.

	Accepts a JSON array, a single JSON object (one item), or JSON Lines — one
	object per line, which is how logs and LLM output usually arrive. JSONL is
	only tried if the strict parse fails, so it can never reinterpret valid
	JSON. Raises ValueError with a readable message on anything else.
	"""
	s = text.strip()
	if not s:
		raise ValueError("JSON To Item List: input is empty.")

	try:
		data = json.loads(s)
	except json.JSONDecodeError as exc:
		items = _parse_jsonl(s)
		if items is None:
			raise ValueError(f"JSON To Item List: invalid JSON — {exc}") from exc
		return items

	if isinstance(data, list):
		return data
	if isinstance(data, dict):
		return [data]                            # a lone object = one item
	raise ValueError(
		"JSON To Item List: expected a JSON array (or one object), got "
		f"{type(data).__name__}."
	)


def _parse_jsonl(s: str):
	"""Every non-blank line parsed as its own JSON value, or None if any fails."""
	lines = [ln.strip() for ln in s.splitlines()]
	lines = [ln for ln in lines if ln]
	if len(lines) < 2:
		return None                              # not a multi-line document
	items = []
	for ln in lines:
		try:
			items.append(json.loads(ln))
		except json.JSONDecodeError:
			return None
	return items


def to_socket_str(value, indent: int | None = None) -> str:
	"""One JSON value as the string a ComfyUI socket can carry.

	Strings pass through verbatim; everything else is JSON — compact by
	default, indented when `indent` is given. Non-ASCII is kept as-is rather
	than \\uXXXX-escaped, so a prompt survives the trip. Shared with JSON Path
	so both nodes serialize a value identically.
	"""
	if isinstance(value, str):
		return value
	if indent is not None:
		return json.dumps(value, ensure_ascii=False, indent=indent)
	return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


@register
class JsonToItemList(TiNode):
	DISPLAY_NAME = "JSON To Item List (ti)"
	CATEGORY = "tinode/data"

	@classmethod
	def INPUT_TYPES(cls):
		return {
			"required": {
				"json_text": ("STRING", {
					"default": '[{"a": 1}, {"a": 2}]',
					"multiline": True,
					"tooltip": "A JSON array — one item per element. A single "
							"object counts as one item, and JSON Lines (one "
							"object per line) is accepted too.",
				}),
			},
			"optional": {
				"pretty": ("BOOLEAN", {
					"default": False,
					"tooltip": "Indent each object instead of emitting it "
							"compact. Cosmetic — for reading in a preview.",
				}),
			},
		}

	RETURN_TYPES = ("ITEM_LIST", "STRING", "INT")
	RETURN_NAMES = ("item_list", "items", "count")
	# Slot 1 only: `items` is a ComfyUI list (downstream runs per item), while
	# item_list is a single ITEM_LIST value and count a plain int.
	OUTPUT_IS_LIST = (False, True, False)
	OUTPUT_TOOLTIPS = (
		"Connect to Inspire's ▶Foreach List `item_list` to iterate the objects.",
		"The same items as a ComfyUI list — every downstream node runs once per item.",
		"How many items were found.",
	)

	DESCRIPTION = (
		"Split a JSON array into one item per element.\n"
		'[{"a": 1}, {"a": 2}] gives 2 items.\n'
		"item_list feeds Inspire's ▶Foreach List; items is a ComfyUI list that "
		"makes downstream nodes run once per element.\n"
		"Objects/arrays come out as JSON text, plain strings unquoted."
	)

	FUNCTION = "execute"

	def execute(self, json_text, pretty=False):
		raw = parse_items(json_text)
		if not raw:
			# ForeachListBegin would die on an empty list with an IndexError,
			# and an empty ComfyUI list silently skips the whole branch.
			raise ValueError("JSON To Item List: the JSON array is empty.")

		items = [to_socket_str(v, indent=2 if pretty else None) for v in raw]

		# Separate list objects: ITEM_LIST is consumed by a loop that slices it,
		# so it must never share identity with the ComfyUI list output.
		return (list(items), items, len(items))
