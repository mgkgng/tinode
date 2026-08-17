"""JSON Path — pull one value out of a JSON document by key and/or index.

The tinode answer to Simple JSON Parser, with a real path parser and normal
caching. `steps[0].prompt` works, and so does everything that node can't reach:

	steps[0]        key then index
	a.b.c           nested keys
	0  /  -1        a bare index — negative counts from the end
	[0].name        a leading index (the document itself is an array)
	a[0][2]         chained indices
	["my.key"]      a quoted key, for keys containing a dot or a bracket
	<empty>         the whole document

**It caches.** Deliberately no IS_CHANGED: the output is a pure function of
(json_text, path), so ComfyUI's input-signature key is already exactly right.
Simple JSON Parser returns `float("NaN")` from IS_CHANGED, which lands in the
cache key (comfy_execution/caching.py) and mints a fresh un-equal NaN every
run — so it re-executes every queue, and because a node's key folds in all of
its ancestors' signatures, so does every node downstream of it. This node
re-runs only when its text or path actually changes.

A malformed path (`a..b`, `a[`) always raises — that is a bug in the workflow,
not data. A path that simply isn't there respects `strict`: raise, or fall back
to `default`. That split is the whole point: a typo'd path never silently
returns your default.
"""

from __future__ import annotations

import json
import re

from ...base import TiNode
from ...registry import register
from .json_to_item_list import to_socket_str

_INT_SEGMENT = re.compile(r"^-?[0-9]+$")

# Accessor kinds produced by parse_path.
KEY = "key"
INDEX = "index"


class JsonPathMiss(Exception):
	"""The path is well-formed but the document has nothing there.

	Separate from ValueError so `strict=False` can fall back on a missing
	value while still reporting a malformed path.
	"""


def parse_path(path: str) -> list[tuple[str, object]]:
	"""Compile a path string into a list of (KEY, name) / (INDEX, int) steps.

	Empty path -> empty list (the document itself). Raises ValueError with the
	offending position on a malformed path.
	"""
	s = path.strip()
	if not s:
		return []

	acc: list[tuple[str, object]] = []
	i, n = 0, len(s)
	# True once a segment has been read: a bare name may not directly follow
	# another segment ("[0]name"), it needs a '.' first. Brackets may.
	need_sep = False

	while i < n:
		c = s[i]

		if c == ".":
			if not need_sep:
				raise ValueError(
					f"JSON Path: empty segment at position {i} in {path!r} "
					"(a stray or doubled '.')."
				)
			i += 1
			need_sep = False
			if i >= n:
				raise ValueError(f"JSON Path: {path!r} ends with '.'.")
			continue

		if c == "[":
			j = s.find("]", i)
			if j < 0:
				raise ValueError(f"JSON Path: unclosed '[' at position {i} in {path!r}.")
			acc.append(_parse_bracket(s[i + 1:j].strip(), i, path))
			i = j + 1
			need_sep = True
			continue

		if need_sep:
			raise ValueError(
				f"JSON Path: expected '.' or '[' at position {i} in {path!r}."
			)

		j = i
		while j < n and s[j] not in ".[":
			j += 1
		name = s[i:j].strip()
		if not name:
			raise ValueError(f"JSON Path: empty segment at position {i} in {path!r}.")
		acc.append((INDEX, int(name)) if _INT_SEGMENT.match(name) else (KEY, name))
		i = j
		need_sep = True

	return acc


def _parse_bracket(inner: str, pos: int, path: str) -> tuple[str, object]:
	"""One [...] accessor: an index, or a quoted key."""
	if len(inner) >= 2 and inner[0] == inner[-1] and inner[0] in "\"'":
		return (KEY, inner[1:-1])              # ["a.b"] — dots stay in the key
	if _INT_SEGMENT.match(inner):
		return (INDEX, int(inner))
	raise ValueError(
		f"JSON Path: [{inner}] at position {pos} in {path!r} is neither an "
		'integer index nor a quoted key (["name"]).'
	)


def render_path(acc: list[tuple[str, object]]) -> str:
	"""Accessors back into path text, for pointing at where a walk failed."""
	out = ""
	for kind, val in acc:
		if kind == INDEX:
			out += f"[{val}]"
		elif out:
			out += f".{val}"
		else:
			out = str(val)
	return out


def type_name(value) -> str:
	"""JSON's name for the value's type — handy for debugging a path."""
	if value is None:
		return "null"
	if isinstance(value, bool):                # before int: bool is an int
		return "boolean"
	if isinstance(value, (int, float)):
		return "number"
	if isinstance(value, str):
		return "string"
	if isinstance(value, list):
		return "array"
	if isinstance(value, dict):
		return "object"
	return type(value).__name__


def walk(data, acc: list[tuple[str, object]]):
	"""Follow the accessors into `data`. Raises JsonPathMiss if absent."""
	cur = data
	for step, (kind, val) in enumerate(acc):
		where = render_path(acc[:step]) or "the document"

		if kind == INDEX:
			# Strings are not indexed on purpose: "abc"[0] silently yielding
			# "a" hides a wrong path far more often than it helps.
			if not isinstance(cur, list):
				raise JsonPathMiss(
					f"[{val}] needs an array at {where}, found {type_name(cur)}."
				)
			if not -len(cur) <= val < len(cur):
				raise JsonPathMiss(
					f"index {val} is out of range at {where} "
					f"(length {len(cur)})."
				)
			cur = cur[val]
		else:
			if not isinstance(cur, dict):
				raise JsonPathMiss(
					f"key {val!r} needs an object at {where}, found {type_name(cur)}."
				)
			if val not in cur:
				keys = list(cur)
				shown = ", ".join(repr(k) for k in keys[:8])
				more = f" (+{len(keys) - 8} more)" if len(keys) > 8 else ""
				raise JsonPathMiss(
					f"key {val!r} not found at {where}; available: "
					f"{shown or '(none)'}{more}."
				)
			cur = cur[val]

	return cur


@register
class JsonPath(TiNode):
	DISPLAY_NAME = "JSON Path (ti)"
	CATEGORY = "tinode/data"

	@classmethod
	def INPUT_TYPES(cls):
		return {
			"required": {
				"json_text": ("STRING", {
					"default": "",
					"multiline": True,
					"forceInput": False,
					"tooltip": "The JSON document to read from.",
				}),
				"path": ("STRING", {
					"default": "",
					"multiline": False,
					"tooltip": "Keys and indices: steps[0].prompt · a.b.c · -1 "
							'· [0].name · a[0][2] · ["key.with.dots"]. '
							"Empty = the whole document.",
				}),
			},
			"optional": {
				"strict": ("BOOLEAN", {
					"default": True,
					"tooltip": "On: a missing key or out-of-range index is an "
							"error. Off: return `default` instead. A malformed "
							"path always errors either way.",
				}),
				"default": ("STRING", {
					"default": "",
					"multiline": False,
					"tooltip": "Returned when strict is off and the path is "
							"not present.",
				}),
				"pretty": ("BOOLEAN", {
					"default": False,
					"tooltip": "Indent object/array output. Cosmetic — JSON "
							"parses the same either way.",
				}),
			},
		}

	RETURN_TYPES = ("STRING", "INT", "STRING")
	RETURN_NAMES = ("value", "count", "type")
	OUTPUT_TOOLTIPS = (
		"The value at the path. Strings come out unquoted; objects and arrays "
		"as JSON text — feed it straight into JSON To Item List.",
		"Length if the value is an array or object, otherwise -1.",
		"object / array / string / number / boolean / null (or 'missing').",
	)

	DESCRIPTION = (
		"Read one value out of a JSON document by path — keys and indices.\n"
		'steps[0].prompt · a.b.c · -1 · [0].name · a[0][2] · ["key.with.dots"]\n'
		"Caches normally: re-runs only when json_text or path changes."
	)

	FUNCTION = "execute"

	# No IS_CHANGED on purpose — see the module docstring. Output is a pure
	# function of the inputs, so ComfyUI's own cache key is already correct.

	def execute(self, json_text, path, strict=True, default="", pretty=False):
		text = (json_text or "").strip()
		if not text:
			raise ValueError("JSON Path: json_text is empty.")

		try:
			data = json.loads(text)
		except json.JSONDecodeError as exc:
			raise ValueError(f"JSON Path: invalid JSON — {exc}") from exc

		acc = parse_path(path)                 # malformed path always raises

		try:
			value = walk(data, acc)
		except JsonPathMiss as miss:
			if strict:
				raise ValueError(f"JSON Path: {miss}") from None
			return (default, -1, "missing")

		count = len(value) if isinstance(value, (list, dict)) else -1
		return (to_socket_str(value, indent=2 if pretty else None), count,
				type_name(value))
