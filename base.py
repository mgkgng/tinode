"""Base class for tinode nodes.

ComfyUI does not require a base class — it duck-types on INPUT_TYPES,
RETURN_TYPES and FUNCTION. This base just supplies sane defaults so each
node file stays small, and gives one place to add shared helpers later
(logging, error wrapping, common type constants, etc.).
"""

from __future__ import annotations


class TiNode:
	# --- registry metadata (read by @register) -------------------------
	# Override per node. NODE_ID defaults to the class name when None.
	NODE_ID: str | None = None
	DISPLAY_NAME: str | None = None

	# --- ComfyUI contract defaults -------------------------------------
	# Right-click menu path. Convention: "tinode/<group>".
	CATEGORY = "tinode"
	# Name of the instance method ComfyUI calls to run the node.
	FUNCTION = "execute"

	# Sensible empty defaults so an unfinished node still loads as a no-op
	# instead of crashing the pack at import time. Real nodes override both.
	RETURN_TYPES: tuple = ()

	@classmethod
	def INPUT_TYPES(cls):  # noqa: N802 — ComfyUI requires this exact name
		return {"required": {}}
