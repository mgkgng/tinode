"""Collect List — collapse a list back to a single value.

When a node emits a LIST (e.g. Bbox Crop · Multi -> one crop per box) and the
downstream runs per item under ComfyUI list expansion, a loop-control node like
Inspire's ForeachListEnd must NOT be list-expanded too — feeding it the per-item
list would run the loop end once per item and break iteration. Collect List sits
in between: INPUT_IS_LIST gathers every item into one call and returns a single
`count`, which also makes the loop end depend on all items finishing.

This is the escape hatch for the list-expansion path. When you want the per-item
work to be a real loop instead — sequential, accumulating, pausable — take the
producer's `ITEM_LIST` output into a nested ▶Foreach List (Bbox Crop · Multi ->
item_list -> Bbox Crop Item); the inner Foreach List◀ already collapses to one
value and no Collect List is needed.
"""

from __future__ import annotations

from ...base import TiNode
from ...registry import register
from .validation_gate import ANY


@register
class CollectList(TiNode):
	DISPLAY_NAME = "Collect List (ti)"
	CATEGORY = "tinode/util"
	INPUT_IS_LIST = True

	@classmethod
	def INPUT_TYPES(cls):
		return {"required": {"value": (ANY, {"tooltip":
			"A list output (e.g. per-crop Save results). Collapsed to a count so a "
			"Foreach end doesn't expand per item."})}}

	RETURN_TYPES = ("INT", ANY)
	RETURN_NAMES = ("count", "last")
	FUNCTION = "execute"

	def execute(self, value):
		v = value if isinstance(value, list) else [value]
		return (len(v), v[-1] if v else None)
