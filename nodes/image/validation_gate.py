"""Validation Gate — pause the workflow until the user approves, then continue.

Drop it anywhere in a loop and each iteration STOPS at it: the frontend pops a
panel with a preview (e.g. the mask overlay) and Approve / Reject. Approve lets
the value through and the graph continues; Reject stops the run. This is what
lets you curate every clip / every crop by hand inside an otherwise automatic
Foreach loop — the loop waits for you at each item instead of running blind.

Mechanism: execute() blocks the execution thread on a threading.Event and sends
the frontend a message with a token; a small POST route sets the event when you
click. It polls for ComfyUI's interrupt so Cancel still works, and (optionally)
times out. Headless (no server) it just passes the value through, so batch/cron
runs don't hang.
"""

from __future__ import annotations

import os
import threading
import time

from ...base import TiNode
from ...registry import register

# Server + interrupt hooks. Guarded so the module still imports in the test
# runner (no ComfyUI server there) — execute() then just passes through.
try:
	from server import PromptServer  # noqa: PLC0415
except Exception:  # noqa: BLE001
	PromptServer = None
try:
	import comfy.model_management as _mm  # noqa: PLC0415
except Exception:  # noqa: BLE001
	_mm = None


class AnyType(str):
	"""A type string that compares equal to everything, so the socket is wildcard."""

	def __ne__(self, other):  # noqa: D105
		return False

	def __eq__(self, other):  # noqa: D105
		return True

	def __hash__(self):  # noqa: D105
		return hash("*")


ANY = AnyType("*")

_PENDING: dict = {}          # token -> {"event": Event, "decision": str|None}
_LOCK = threading.Lock()


def _resolve(decision):
	"""Map a raw decision string to 'approve' / 'reject' (default approve)."""
	return "reject" if str(decision).lower().startswith("rej") else "approve"


def _register_route():
	if PromptServer is None or getattr(_register_route, "_done", False):
		return
	try:
		routes = PromptServer.instance.routes
	except Exception:  # noqa: BLE001 — server not ready yet
		return

	@routes.post("/tinode/gate")
	async def _gate(request):  # noqa: ANN001
		from aiohttp import web  # noqa: PLC0415

		data = await request.json()
		token = data.get("token")
		with _LOCK:
			slot = _PENDING.get(token)
			if slot is not None:
				slot["decision"] = _resolve(data.get("decision", "approve"))
				slot["event"].set()
		return web.json_response({"ok": slot is not None})

	_register_route._done = True


_register_route()


def _save_preview(image):
	"""Save the first frame of an IMAGE to temp for the review panel, or None."""
	try:
		import numpy as np  # noqa: PLC0415
		from PIL import Image  # noqa: PLC0415
		import folder_paths  # noqa: PLC0415

		import random  # noqa: PLC0415

		frame = image[0] if image.dim() == 4 else image
		arr = (frame[..., :3].clamp(0, 1).cpu().numpy() * 255.0 + 0.5).astype(np.uint8)
		name = "ti_gate_" + "".join(random.choice("abcdefghijklmnop") for _ in range(8)) + ".png"
		out = folder_paths.get_temp_directory()
		os.makedirs(out, exist_ok=True)
		Image.fromarray(arr).save(os.path.join(out, name), compress_level=1)
		return {"filename": name, "subfolder": "", "type": "temp"}
	except Exception as exc:  # noqa: BLE001
		print(f"[tinode] Validation Gate preview unavailable: {exc!r}")
		return None


@register
class ValidationGate(TiNode):
	DISPLAY_NAME = "Validation Gate (ti)"
	CATEGORY = "tinode/util"

	@classmethod
	def INPUT_TYPES(cls):
		return {
			"required": {
				"value": (ANY, {"tooltip":
					"Passed through once you approve. Wire the thing whose "
					"downstream you want gated (e.g. the mask before Save)."}),
			},
			"optional": {
				"preview": ("IMAGE", {"tooltip":
					"Shown in the approval panel — e.g. the mask overlay to review."}),
				"title": ("STRING", {"default": "Approve to continue?", "multiline": False}),
				"timeout": ("INT", {"default": 0, "min": 0, "max": 86400, "step": 1,
					"tooltip": "Seconds to wait before failing. 0 = wait forever."}),
			},
			"hidden": {"unique_id": "UNIQUE_ID"},
		}

	RETURN_TYPES = (ANY,)
	RETURN_NAMES = ("value",)
	FUNCTION = "execute"

	@classmethod
	def IS_CHANGED(cls, **kwargs):
		# Never cache an approval — the gate must re-run (and re-ask) every queue.
		return float("nan")

	def execute(self, value, preview=None, title="Approve to continue?", timeout=0, unique_id=None):
		# Headless / no server: don't hang a batch run — pass straight through.
		if PromptServer is None:
			return (value,)
		_register_route()   # ensure the route exists (server may not have been ready at import)

		token = f"{unique_id}:{time.time_ns()}"
		img_info = _save_preview(preview) if preview is not None else None
		ev = threading.Event()
		with _LOCK:
			_PENDING[token] = {"event": ev, "decision": None}

		PromptServer.instance.send_sync("tinode.gate", {
			"node_id": str(unique_id), "token": token,
			"title": str(title), "image": img_info,
		})

		start = time.time()
		try:
			while not ev.wait(timeout=0.2):
				if _mm is not None and _mm.processing_interrupted():
					raise _mm.InterruptProcessingException()
				if timeout and (time.time() - start) > timeout:
					raise RuntimeError(
						f"Validation Gate: timed out after {timeout}s waiting for approval.")
			with _LOCK:
				decision = _PENDING.get(token, {}).get("decision")
		finally:
			with _LOCK:
				_PENDING.pop(token, None)
			try:
				PromptServer.instance.send_sync("tinode.gate_done", {"token": token})
			except Exception:  # noqa: BLE001
				pass

		if decision == "reject":
			raise RuntimeError("Validation Gate: rejected — run stopped by the user.")
		return (value,)
