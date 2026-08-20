// Shared helpers for the tinode in-node canvas editors
// (bbox_crop.js, pick_segments.js, add_segments.js).
//
// Only the pure, stateless pieces live here — colour, widget access, asset URLs,
// letterbox maths and pointer mapping. Each editor keeps its own setup and
// interaction code, because those genuinely differ.
//
// Extracting these is not just tidiness: the screen->canvas pointer rescale
// below was fixed once and then existed as three drifting copies. One home
// means one fix.

import { app } from "../../../scripts/app.js";
import { api } from "../../../scripts/api.js";

/** Deterministic bright colour for a segment id, as [r,g,b] 0-255.
 *  MUST stay in sync with color_for_id() in nodes/image/pick_segments.py —
 *  tests/test_color_parity.py asserts the two agree. */
export function colorForId(id) {
	const h = ((id * 0.61803398875) % 1.0 + 1.0) % 1.0;
	const s = 0.65, v = 1.0;
	const i = Math.floor(h * 6);
	const f = h * 6 - i;
	const p = v * (1 - s), q = v * (1 - s * f), t = v * (1 - s * (1 - f));
	const [r, g, b] = [[v, t, p], [q, v, p], [p, v, t], [p, q, v], [t, p, v], [v, p, q]][i % 6];
	return [Math.round(r * 255), Math.round(g * 255), Math.round(b * 255)];
}

export function clamp(v, lo, hi) { return Math.max(lo, Math.min(v, hi)); }

/** Parse one aspect ratio token ("16:9", "1.777", "4:3") to W/H, or null. */
export function parseRatio(s) {
	if (s == null) return null;
	s = String(s).trim();
	if (!s) return null;
	let r;
	if (s.includes(":")) {
		const [a, b] = s.split(":").map(Number);
		if (!a || !b) return null;
		r = a / b;
	} else {
		r = Number(s);
	}
	return (isFinite(r) && r > 0) ? r : null;
}

/** Parse a comma-separated list of ratios ("16:9, 1:1, 4:3") to an array of
 *  W/H numbers (invalid tokens dropped). Empty list = free. */
export function parseRatios(s) {
	if (s == null) return [];
	return String(s).split(",").map(parseRatio).filter((r) => r != null);
}

/** The ratio for box index `i` from a ratios list: the i-th, or the last one
 *  when there are fewer ratios than boxes (so a single ratio applies to all). */
export function ratioForIndex(ratios, i) {
	if (!ratios || !ratios.length) return null;
	return ratios[Math.min(i, ratios.length - 1)];
}

/** Reshape a just-resized box to width/height == ratio, keeping the grip's
 *  anchored edge(s) fixed. `grip` is one of nw/ne/sw/se (corner) or n/s/e/w
 *  (edge); "move" and a null ratio pass through unchanged. Frame clamping is
 *  left to the caller. */
export function constrainToRatio(box, grip, ratio) {
	if (!ratio || !grip || grip === "move") return box;
	const { x, y, w, h } = box;
	// corner: size to cover the drag, anchored at the opposite corner
	const nw = Math.max(w, h * ratio), nh = nw / ratio;
	switch (grip) {
		case "se": return { x, y, w: nw, h: nh };
		case "nw": return { x: x + w - nw, y: y + h - nh, w: nw, h: nh };
		case "ne": return { x, y: y + h - nh, w: nw, h: nh };
		case "sw": return { x: x + w - nw, y, w: nw, h: nh };
		case "e": case "w": {         // width fixed by the drag; height from it, centred
			const nh2 = w / ratio;
			return { x, y: y + h / 2 - nh2 / 2, w, h: nh2 };
		}
		case "n": case "s": {         // height fixed by the drag; width from it, centred
			const nw2 = h * ratio;
			return { x: x + w / 2 - nw2 / 2, y, w: nw2, h };
		}
		default: return box;
	}
}

export function getWidget(node, name) {
	return node.widgets?.find((w) => w.name === name);
}

/** Set a widget value and fire its callback (so it serializes + repaints). */
export function setWidget(node, name, value) {
	const w = getWidget(node, name);
	if (!w || w.value === value) return;
	w.value = value;
	w.callback?.(value, app.canvas, node);
}

/** Hide a widget the editor drives itself. Still serialized with the workflow. */
export function hideWidget(node, name) {
	const w = getWidget(node, name);
	if (w) { w.type = "hidden"; w.computeSize = () => [0, -4]; }
}

/** URL for a {filename, subfolder, type} asset produced by a node.
 *  The cache-buster is keyed on subfolder+filename: the subfolder encodes the
 *  input signature, so it changes whenever the input does and the browser
 *  refetches instead of serving a stale frame. */
export function urlFor(info) {
	return api.apiURL(
		`/view?filename=${encodeURIComponent(info.filename)}` +
		`&type=${info.type || "temp"}` +
		`&subfolder=${encodeURIComponent(info.subfolder || "")}` +
		`&rand=${encodeURIComponent((info.subfolder || "") + "/" + info.filename)}`,
	);
}

/** Letterbox a natW x natH frame into a canvas -> {ox, oy, scale}. */
export function viewRect(canvas, natW, natH) {
	const scale = Math.min(canvas.width / natW, canvas.height / natH);
	return { ox: (canvas.width - natW * scale) / 2, oy: (canvas.height - natH * scale) / 2, scale };
}

/** Pointer event -> canvas backing-store coordinates.
 *  getBoundingClientRect() is in SCREEN px (scaled by the LiteGraph zoom) while
 *  the canvas draws in backing-store px. Without this rescale every hit-test
 *  misses at any zoom other than 100%. */
export function pointerPos(canvas, e) {
	const r = canvas.getBoundingClientRect();
	return [
		(e.clientX - r.left) * (canvas.width / (r.width || 1)),
		(e.clientY - r.top) * (canvas.height / (r.height || 1)),
	];
}

/** Release a stale LiteGraph pointer capture, else our pointermove never fires
 *  and the drag "sticks". Call at the start of a drag. */
export function releaseGraphPointer(e) {
	const lg = document.querySelector("canvas.litegraph, canvas.lgraphcanvas");
	if (lg && typeof e.pointerId === "number") {
		try { if (lg.hasPointerCapture(e.pointerId)) lg.releasePointerCapture(e.pointerId); } catch {}
	}
}

/** Force a DOM-widget element to span the node's inner width.
 *
 *  ComfyUI sizes a DOM widget from its widget layout, and DOMWidgetImpl's
 *  computeLayoutSize() hardcodes `minWidth: 0` — it only exposes getMinHeight /
 *  getMaxHeight, with no width hook whatsoever. So the editors grew in HEIGHT
 *  but the element stayed narrow no matter how wide the node was.
 *
 *  The positioner writes a plain inline `width: Npx`, so an !important inline
 *  width wins. Returns the width applied. */
export function fillNodeWidth(node, el) {
	const w = Math.max(64, Math.round((node.size?.[0] ?? 320) - 30));
	el.style.setProperty("width", `${w}px`, "important");
	return w;
}

/** Match the canvas backing store to its laid-out size. Returns false when the
 *  element has no layout yet (nothing worth drawing). */
export function syncCanvasSize(canvas) {
	const w = Math.max(1, Math.floor(canvas.clientWidth));
	const h = Math.max(1, Math.floor(canvas.clientHeight));
	if (canvas.width !== w || canvas.height !== h) { canvas.width = w; canvas.height = h; }
	return canvas.clientWidth > 0 && canvas.clientHeight > 0;
}
