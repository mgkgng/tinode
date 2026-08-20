"""Item Params / Param Get — per-item settings that survive the batch run.

The problem these solve: a node's widgets are GLOBAL. Step to crop 1 with Item
Cursor, find that it needs the SAM3 prompt "green jacket" at threshold 0.5, step
to crop 2 which needs "logo" at 0.35 — the widget only holds one value, so the
unattended Foreach run applies whichever was typed last to everything.

Item Params is a keyed table: it stores one JSON object per item key and hands
back the entry for the CURRENT key. Tune an item, its settings are written under
its key; step away and back, they return. The Foreach run reads the same table,
so every item is processed with the settings you found for it.

    Item Cursor ─item→ (…)                  Item Params ─json→ Param Get ─value→ any widget input
    Video Source Path ─stem→ key ───────────┘

Param Get pulls one named value out with a type that can drive a widget input
(ComfyUI lets you convert any widget to an input): STRING for a prompt, FLOAT
for a threshold, INT for a count, BOOLEAN for a flag. Missing names fall back
rather than erroring, so a half-filled table still runs.

Anything with a widget can be driven this way — EasySAM3 Segment's prompt /
threshold / max_segments, Pick Segments' overlay_alpha, Add Segments'
manual_segments, Bbox Crop · Multi's boxes, Split Video · Chunks' cuts.
"""

from __future__ import annotations

import json as _json

from ...base import TiNode, first
from ...registry import register


def _loads(text, what):
	"""Parse a JSON object, tolerating empty text; raise readably on junk."""
	s = str(text or "").strip()
	if not s:
		return {}
	try:
		v = _json.loads(s)
	except ValueError as exc:
		raise RuntimeError(f"Item Params: {what} is not valid JSON — {exc}") from exc
	if not isinstance(v, dict):
		raise RuntimeError(f"Item Params: {what} must be a JSON object, got {type(v).__name__}.")
	return v


@register
class ItemParams(TiNode):
	DISPLAY_NAME = "Item Params (ti)"
	CATEGORY = "tinode/util"
	OUTPUT_NODE = True

	@classmethod
	def INPUT_TYPES(cls):
		return {
			"required": {
				"key": ("STRING", {"default": "", "tooltip":
					"Identifies the current item — e.g. the clip stem, or "
					"`stem|crop_index` when settings are per crop."}),
			},
			"optional": {
				"entry": ("STRING", {"default": "", "multiline": True, "tooltip":
					"Settings for the CURRENT key, as JSON — e.g.\n"
					'{"prompt": "green jacket", "threshold": 0.5}\n'
					"Edited here and saved under the key when you queue."}),
				"defaults": ("STRING", {"default": "{}", "multiline": True, "tooltip":
					"Fallback settings for keys with no entry of their own."}),
				"table": ("STRING", {"default": "{}", "multiline": True, "tooltip":
					"The saved table: {key: {...}}. Written for you; kept visible "
					"so a whole batch's settings can be reviewed or pasted."}),
				"entry_key": ("STRING", {"default": "", "tooltip":
					"Which key `entry` belongs to — set by the node. Guards against "
					"saving one item's settings onto the next."}),
				"sub": ("STRING", {"default": "", "tooltip":
					"Optional second part of the key, joined as `key|sub` — wire a "
					"crop_index here to make settings per CROP rather than per clip."}),
			},
		}

	RETURN_TYPES = ("STRING", "STRING", "STRING")
	RETURN_NAMES = ("json", "table", "key")
	OUTPUT_TOOLTIPS = (
		"This key's settings (defaults merged under its own entry) — feed Param Get.",
		"The whole updated table.",
		"The key, passed through.",
	)
	FUNCTION = "execute"

	@classmethod
	def IS_CHANGED(cls, **kwargs):
		return float("nan")      # always re-read: the table is edited between runs

	def execute(self, key, entry="", defaults="{}", table="{}", entry_key="", sub=""):
		key = str(first(key, "")).strip()
		part = str(first(sub, "") or "").strip()
		if part:
			key = f"{key}|{part}"
		if not key:
			raise RuntimeError(
				"Item Params: empty key. Wire the clip stem (or `stem|crop`) so "
				"settings are stored per item.")
		tbl = _loads(first(table, "{}"), "table")
		dflt = _loads(first(defaults, "{}"), "defaults")
		txt = str(first(entry, "") or "").strip()
		belongs = str(first(entry_key, "") or "").strip()

		# Only save `entry` when it was edited FOR this key. After stepping to a
		# new item the box still shows the previous one's text, and blindly
		# writing that would overwrite the new item's settings with its
		# predecessor's — the one bug that would quietly ruin a whole batch.
		if txt and belongs == key:
			tbl[key] = _loads(txt, f"entry for {key!r}")

		merged = dict(dflt)
		merged.update(tbl.get(key) or {})
		out_entry = _json.dumps(tbl.get(key, {}), ensure_ascii=False, indent=2)
		out_table = _json.dumps(tbl, ensure_ascii=False, indent=2)
		print(f"[tinode] Item Params: {key} -> {merged}")
		return {
			"ui": {"ti_params": [{"key": key, "entry": out_entry,
								  "known": sorted(tbl.keys())}]},
			"result": (_json.dumps(merged, ensure_ascii=False), out_table, key),
		}


@register
class ParamGet(TiNode):
	DISPLAY_NAME = "Param Get (ti)"
	CATEGORY = "tinode/util"

	@classmethod
	def INPUT_TYPES(cls):
		return {
			"required": {
				"json": ("STRING", {"default": "{}", "tooltip":
					"From Item Params — this item's settings."}),
				"name": ("STRING", {"default": "prompt", "tooltip":
					"Which setting to read, e.g. prompt / threshold / cuts."}),
			},
			"optional": {
				"fallback": ("STRING", {"default": "", "tooltip":
					"Used when the name is absent, so a half-filled table still runs."}),
			},
		}

	RETURN_TYPES = ("STRING", "FLOAT", "INT", "BOOLEAN", "BOOLEAN")
	RETURN_NAMES = ("string", "float", "int", "boolean", "found")
	OUTPUT_TOOLTIPS = (
		"The value as text — for a prompt, a box list, a cuts string.",
		"As a float — for a threshold or an alpha.",
		"As an int — for a count or a frame number.",
		"As a boolean — for a flag.",
		"Whether the name was actually present (vs the fallback).",
	)
	FUNCTION = "execute"

	# `json` is the socket name ComfyUI passes, so the stdlib module is aliased at
	# import; keep the parameter name to match the socket.
	def execute(self, json, name, fallback=""):
		raw = str(first(json, "{}") or "{}")
		key = str(first(name, "")).strip()
		fb = str(first(fallback, "") or "")
		try:
			data = _json.loads(raw) if raw.strip() else {}
		except ValueError:
			data = {}
		found = isinstance(data, dict) and key in data
		val = data.get(key) if found else fb

		if isinstance(val, bool):
			text = "true" if val else "false"
		elif isinstance(val, (int, float)):
			text = repr(val)
		elif isinstance(val, (dict, list)):
			text = _json.dumps(val, ensure_ascii=False)
		else:
			text = "" if val is None else str(val)

		try:
			f = float(val) if isinstance(val, (int, float, bool)) else float(text)
		except (TypeError, ValueError):
			f = 0.0
		i = int(f)
		if isinstance(val, bool):
			b = val
		else:
			b = text.strip().lower() in ("true", "1", "yes", "on")
		return (text, f, i, b, bool(found))
