"""Foreach List Begin / End — iterate an ITEM_LIST without the Inspire Pack.

A self-contained port of Inspire Pack's ForeachListBegin/End (GPL-3.0,
https://github.com/ltdrdata/ComfyUI-Inspire-Pack), which is itself based on
BadCafeCode's execution-inversion reference loop
(https://github.com/BadCafeCode/execution-inversion-demo-comfyui). Ported into
tinode so a machine with only this pack installed can still run the removal
workflows' loops.

How the loop works: End receives Begin's link (rawLink), and while the remaining
list is non-empty it CLONES every node between Begin and End into an expanded
subgraph whose Begin gets the remainder — so the body re-executes once per item
with `intermediate_output` threading state through (the accumulator pattern).
Same socket names and type strings as Inspire's nodes, so swapping a node's
class between the two packs ports a saved graph.
"""

from __future__ import annotations

import logging

from ...base import TiNode
from ...registry import register

try:
	from comfy_execution.graph_utils import GraphBuilder, is_link
except Exception:  # noqa: BLE001 — not inside ComfyUI (tests); nodes then refuse to run
	GraphBuilder = None
	is_link = None


class ListWrapper:
	"""A list that carries `aux` (total count, reporter id) through the loop."""

	def __init__(self, data, aux=None):
		if isinstance(data, ListWrapper):
			self._data = data._data
			self.aux = data.aux if aux is None else aux
		else:
			self._data = list(data)
			self.aux = aux

	def __getitem__(self, index):
		if isinstance(index, slice):
			return ListWrapper(self._data[index], self.aux)
		return self._data[index]

	def __setitem__(self, index, value):
		self._data[index] = value

	def __len__(self):
		return len(self._data)

	def __repr__(self):
		return f"ListWrapper({self._data!r}, aux={self.aux})"


@register
class ForeachListBegin(TiNode):
	DISPLAY_NAME = "Foreach List Begin (ti)"
	CATEGORY = "tinode/util"

	@classmethod
	def INPUT_TYPES(cls):
		return {
			"required": {
				"item_list": ("ITEM_LIST", {"tooltip":
					"The items to iterate, one per loop pass."}),
			},
			"optional": {
				"initial_input": ("*", {"tooltip":
					"Seed for intermediate_output on the FIRST pass. CONNECT "
					"SOMETHING (any value): left empty, item_list[0] is consumed "
					"as the seed and the first item is never iterated."}),
			},
		}

	RETURN_TYPES = ("FOREACH_LIST_CONTROL", "ITEM_LIST", "*", "*")
	RETURN_NAMES = ("flow_control", "remained_list", "item", "intermediate_output")
	OUTPUT_TOOLTIPS = (
		"Wire straight to Foreach List End.",
		"Wire straight to Foreach List End.",
		"The current item.",
		"The accumulator from the previous pass (initial_input on the first).",
	)
	FUNCTION = "doit"

	def doit(self, item_list, initial_input=None):
		if initial_input is None:
			initial_input = item_list[0]
			item_list = item_list[1:]

		if len(item_list) > 0:
			next_list = ListWrapper(item_list[1:])
			next_item = item_list[0]
		else:
			next_list = ListWrapper([])
			next_item = None

		if next_list.aux is None:
			next_list.aux = (len(item_list), None)

		return ("stub", next_list, next_item, initial_input)


@register
class ForeachListEnd(TiNode):
	DISPLAY_NAME = "Foreach List End (ti)"
	CATEGORY = "tinode/util"

	@classmethod
	def INPUT_TYPES(cls):
		return {
			"required": {
				"flow_control": ("FOREACH_LIST_CONTROL", {"rawLink": True, "tooltip":
					"Directly from Foreach List Begin."}),
				"remained_list": ("ITEM_LIST", {"tooltip":
					"Directly from Foreach List Begin."}),
				"intermediate_output": ("*", {"tooltip":
					"This pass's result — it becomes the next pass's "
					"intermediate_output, and the final one is `result`."}),
			},
			"hidden": {"dynprompt": "DYNPROMPT", "unique_id": "UNIQUE_ID"},
		}

	RETURN_TYPES = ("*",)
	RETURN_NAMES = ("result",)
	FUNCTION = "doit"

	def _explore_dependencies(self, node_id, dynprompt, upstream):
		node_info = dynprompt.get_node(node_id)
		if "inputs" not in node_info:
			return
		for v in node_info["inputs"].values():
			if is_link(v):
				parent_id = v[0]
				if parent_id not in upstream:
					upstream[parent_id] = []
					self._explore_dependencies(parent_id, dynprompt, upstream)
				upstream[parent_id].append(node_id)

	def _collect_contained(self, node_id, upstream, contained):
		if node_id not in upstream:
			return
		for child_id in upstream[node_id]:
			if child_id not in contained:
				contained[child_id] = True
				self._collect_contained(child_id, upstream, contained)

	def doit(self, flow_control, remained_list, intermediate_output, dynprompt, unique_id):
		if GraphBuilder is None:
			raise RuntimeError(
				"Foreach List End: ComfyUI's graph expansion API "
				"(comfy_execution.graph_utils) is unavailable.")

		if hasattr(remained_list, "aux"):
			total = remained_list.aux[0]
			done = total - len(remained_list)
			print(f"[tinode] Foreach List: {done}/{total} steps")
		else:
			logging.warning(
				"[tinode] Foreach List End: `remained_list` did not come from "
				"Foreach List Begin.")

		if len(remained_list) == 0:
			return (intermediate_output,)

		# Clone every node between Begin and End into an expanded subgraph whose
		# Begin gets the remainder — one more pass of the loop.
		upstream = {}
		self._explore_dependencies(unique_id, dynprompt, upstream)

		contained = {}
		open_node = flow_control[0]
		self._collect_contained(open_node, upstream, contained)
		contained[unique_id] = True
		contained[open_node] = True

		graph = GraphBuilder()
		for node_id in contained:
			original_node = dynprompt.get_node(node_id)
			node = graph.node(original_node["class_type"],
							  "Recurse" if node_id == unique_id else node_id)
			node.set_override_display_id(node_id)

		for node_id in contained:
			original_node = dynprompt.get_node(node_id)
			node = graph.lookup_node("Recurse" if node_id == unique_id else node_id)
			for k, v in original_node["inputs"].items():
				if is_link(v) and v[0] in contained:
					parent = graph.lookup_node(v[0])
					node.set_input(k, parent.out(v[1]))
				else:
					node.set_input(k, v)

		new_open = graph.lookup_node(open_node)
		new_open.set_input("item_list", remained_list)
		new_open.set_input("initial_input", intermediate_output)

		my_clone = graph.lookup_node("Recurse")
		return {
			"result": (my_clone.out(0),),
			"expand": graph.finalize(),
		}
