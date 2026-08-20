"""Item Cursor — step through an ITEM_LIST one item at a time, by hand.

A ▶Foreach List runs every item start to finish, so you can only dial in
parameters on the first one. This is the opposite tool: it takes the same
ITEM_LIST and hands you ONE item, chosen by `index`. Queue as many times as you
like on that item — change the SAM3 prompt, the threshold, the crop, whatever —
then press Next ▶ in the node to advance. Nothing else in the graph changes,
because downstream still receives exactly the item a loop would have given it.

Use it as a drop-in for ForeachListBegin while you are finding settings:

    Load Videos ─item_list→ Item Cursor ─item→ (Crop Item / Chunk Item / …)

and there is no ForeachListEnd to wire, because there is no loop — one queue
processes one item. When the settings are right, swap the Foreach loop back in
for the unattended run.

`skip_done` is what makes it practical over a long batch: with the store
subdir set, the cursor reports how many items are already saved, so Next lands
on work you have not done yet instead of re-running finished items.
"""

from __future__ import annotations

import os

from ...base import TiNode, first
from ...registry import register
from .validation_gate import ANY


@register
class ItemCursor(TiNode):
	DISPLAY_NAME = "Item Cursor (ti)"
	CATEGORY = "tinode/util"
	# Always run, so the node UI can report where the cursor is even when the
	# downstream is cached.
	OUTPUT_NODE = True

	@classmethod
	def INPUT_TYPES(cls):
		return {
			"required": {
				"item_list": ("ITEM_LIST", {"tooltip":
					"Any tinode ITEM_LIST — clips, crops, chunks, saved masks."}),
				"index": ("INT", {"default": 0, "min": 0, "max": 99999, "step": 1,
					"tooltip": "Which item to emit. Use the ◀ ▶ buttons in the node; "
							   "re-queue as often as you like before advancing."}),
			},
			"optional": {
				"wrap": ("BOOLEAN", {"default": False, "tooltip":
					"Past the end, wrap to 0 instead of clamping to the last item."}),
			},
		}

	RETURN_TYPES = (ANY, "INT", "INT", "BOOLEAN")
	RETURN_NAMES = ("item", "index", "count", "is_last")
	OUTPUT_TOOLTIPS = (
		"The selected item — wire exactly where ForeachListBegin's `item` went.",
		"The index actually used (clamped or wrapped).",
		"How many items there are.",
		"True when this is the final item.",
	)
	FUNCTION = "execute"

	def execute(self, item_list, index=0, wrap=False):
		items = first(item_list) if isinstance(item_list, list) and len(item_list) == 1 \
			and isinstance(first(item_list), list) else item_list
		if not isinstance(items, (list, tuple)):
			raise RuntimeError(
				"Item Cursor: `item_list` must be an ITEM_LIST (got "
				f"{type(items).__name__}).")
		n = len(items)
		if n == 0:
			raise RuntimeError("Item Cursor: the list is empty — nothing to step through.")

		i = int(first(index, 0))
		i = (i % n) if bool(first(wrap, False)) else max(0, min(i, n - 1))
		item = items[i]

		label = _describe(item)
		print(f"[tinode] Item Cursor: {i + 1}/{n}  {label}")
		return {"ui": {"ti_cursor": [{"index": i, "count": n, "label": label}]},
				"result": (item, i, n, i == n - 1)}


def _describe(item):
	"""A short human label for whatever kind of item this is."""
	if not isinstance(item, dict):
		return type(item).__name__
	if "stem" in item and "crop_index" in item:
		return f"{item['stem']} · crop {item['crop_index']}"
	if "stem" in item:
		return str(item["stem"])
	if "chunk_index" in item:
		return (f"chunk {item['chunk_index']}  frames "
				f"{item.get('start')}–{item.get('end')}")
	if "crop_index" in item:
		b = item.get("box") or {}
		return (f"crop {item['crop_index']}"
				+ (f"  {b.get('w')}x{b.get('h')} @ {b.get('x')},{b.get('y')}" if b else ""))
	if "source_path" in item:
		return os.path.basename(str(item["source_path"]))
	return "item"
