// Interactive bbox editor for the "Bbox Crop · Manual (ti)" node.
//
// Mounts a canvas into the node body. After the graph runs once, the Python
// node ships a preview of the incoming frame (+ its true pixel dimensions) via
// the standard {"ui":{"images":[...], "src_dims":[W,H]}} channel; onExecuted
// loads it and draws it here. You drag the four corner handles (or an edge, or
// the whole box) and the x / y / width / height widgets update live — those
// widget values are what the backend actually crops with, and they're saved
// with the workflow.
//
// The editor only touches its own <canvas>; a failure here can't affect the
// rest of the graph. Coordinates the user manipulates are always TRUE source
// pixels — the displayed preview may be downscaled, but src_dims keeps the
// handles mapped to full-resolution coordinates.

import { app } from "../../scripts/app.js";
import {
	clamp, constrainToRatio, fillNodeWidth, getWidget, parseRatios, pointerPos,
	ratioForIndex, releaseGraphPointer, urlFor,
} from "./lib/editor.js";

const NODE_TYPE = "TI_BboxCropManual";
const HANDLE_HIT = 12;   // px radius (canvas space) to grab a corner/edge
const HANDLE_VIS = 5;    // half-size of a drawn handle square
const MIN_BOX = 1;       // smallest box in source pixels
const MIN_NODE_W = 320;  // node minimum width  (leaves room for the canvas)
const MIN_NODE_H = 440;
const MAX_CANVAS_H = 2000;  // node minimum height (fixed widgets + canvas)

// Read the box from the four widgets, resolving the "0 = full extent" default
// the backend uses so an unset box shows as the whole frame instead of empty.
function readBox(node) {
	const iw = node._ti.imgW, ih = node._ti.imgH;
	let x = Math.round(getWidget(node, "x")?.value ?? 0);
	let y = Math.round(getWidget(node, "y")?.value ?? 0);
	let w = Math.round(getWidget(node, "width")?.value ?? 0);
	let h = Math.round(getWidget(node, "height")?.value ?? 0);
	x = clamp(x, 0, Math.max(0, iw - MIN_BOX));
	y = clamp(y, 0, Math.max(0, ih - MIN_BOX));
	if (w <= 0) w = iw - x;            // 0 -> to the right edge
	if (h <= 0) h = ih - y;            // 0 -> to the bottom edge
	w = clamp(w, MIN_BOX, iw - x);
	h = clamp(h, MIN_BOX, ih - y);
	return { x, y, w, h };
}

// Write the box back into the widgets, firing their callbacks so the numeric
// fields repaint and the value is serialized with the workflow.
function writeBox(node, box) {
	const iw = node._ti.imgW, ih = node._ti.imgH;
	const x = clamp(Math.round(box.x), 0, Math.max(0, iw - MIN_BOX));
	const y = clamp(Math.round(box.y), 0, Math.max(0, ih - MIN_BOX));
	const w = clamp(Math.round(box.w), MIN_BOX, iw - x);
	const h = clamp(Math.round(box.h), MIN_BOX, ih - y);
	setWidget(node, "x", x);
	setWidget(node, "y", y);
	setWidget(node, "width", w);
	setWidget(node, "height", h);
}

function setWidget(node, name, value) {
	const w = getWidget(node, name);
	if (!w || w.value === value) return;
	w.value = value;
	w.callback?.(value, app.canvas, node);
}

// Fit the source frame into the canvas as a letterboxed rectangle, returning
// the placement + the px<->source scale so hit-testing and drawing agree.
function viewRect(node) {
	const cv = node._ti.canvas;
	const cw = cv.width, ch = cv.height;
	const iw = node._ti.imgW, ih = node._ti.imgH;
	const scale = Math.min(cw / iw, ch / ih);
	const dw = iw * scale, dh = ih * scale;
	return { ox: (cw - dw) / 2, oy: (ch - dh) / 2, scale };
}

function toCanvas(node, sx, sy) {
	const v = node._ti.viewRect;
	return [v.ox + sx * v.scale, v.oy + sy * v.scale];
}
function toSource(node, cx, cy) {
	const v = node._ti.viewRect;
	return [(cx - v.ox) / v.scale, (cy - v.oy) / v.scale];
}

// Which grip is under the pointer: a corner, an edge, the box interior, or none.
function hitTest(node, cx, cy) {
	const b = readBox(node);
	const [x0, y0] = toCanvas(node, b.x, b.y);
	const [x1, y1] = toCanvas(node, b.x + b.w, b.y + b.h);
	const corners = {
		nw: [x0, y0], ne: [x1, y0], sw: [x0, y1], se: [x1, y1],
	};
	for (const [name, [px, py]] of Object.entries(corners)) {
		if (Math.hypot(cx - px, cy - py) <= HANDLE_HIT) return name;
	}
	// Reach HANDLE_HIT OUTSIDE the edge, but only a third of the box INWARD, so a
	// small box always keeps a central zone for moving instead of resizing.
	const inMx = Math.min(HANDLE_HIT, (x1 - x0) / 3);
	const inMy = Math.min(HANDLE_HIT, (y1 - y0) / 3);
	const nearL = cx >= x0 - HANDLE_HIT && cx <= x0 + inMx;
	const nearR = cx <= x1 + HANDLE_HIT && cx >= x1 - inMx;
	const nearT = cy >= y0 - HANDLE_HIT && cy <= y0 + inMy;
	const nearB = cy <= y1 + HANDLE_HIT && cy >= y1 - inMy;
	const inYs = cy >= y0 - HANDLE_HIT && cy <= y1 + HANDLE_HIT;
	const inXs = cx >= x0 - HANDLE_HIT && cx <= x1 + HANDLE_HIT;
	if (nearL && inYs) return "w";
	if (nearR && inYs) return "e";
	if (nearT && inXs) return "n";
	if (nearB && inXs) return "s";
	if (cx > x0 && cx < x1 && cy > y0 && cy < y1) return "move";
	return null;
}

const CURSOR = {
	nw: "nwse-resize", se: "nwse-resize", ne: "nesw-resize", sw: "nesw-resize",
	n: "ns-resize", s: "ns-resize", e: "ew-resize", w: "ew-resize", move: "move",
};

// Apply a drag of `grip` to the box, keeping the opposite edge/corner anchored.
function applyDrag(node, grip, sx, sy) {
	const b = readBox(node);
	const iw = node._ti.imgW, ih = node._ti.imgH;
	let left = b.x, top = b.y, right = b.x + b.w, bottom = b.y + b.h;
	sx = clamp(sx, 0, iw);
	sy = clamp(sy, 0, ih);

	if (grip === "move") {
		const d = node._ti.drag;
		let nx = clamp(d.ox + (sx - d.sx), 0, iw - b.w);
		let ny = clamp(d.oy + (sy - d.sy), 0, ih - b.h);
		writeBox(node, { x: nx, y: ny, w: b.w, h: b.h });
		return;
	}
	if (grip.includes("w")) left = Math.min(sx, right - MIN_BOX);
	if (grip.includes("e")) right = Math.max(sx, left + MIN_BOX);
	if (grip.includes("n")) top = Math.min(sy, bottom - MIN_BOX);
	if (grip.includes("s")) bottom = Math.max(sy, top + MIN_BOX);
	let box = { x: left, y: top, w: right - left, h: bottom - top };
	// One box, so the first ratio in the list applies.
	box = constrainToRatio(box, grip, ratioForIndex(parseRatios(getWidget(node, "aspect_ratio")?.value), 0));
	writeBox(node, box);
}

function draw(node) {
	const ti = node._ti;
	const cv = ti.canvas;
	const ctx = cv.getContext("2d");
	fillNodeWidth(node, ti.container || ti.wrap);

	// Keep the backing store matched to the on-screen size for crisp lines.
	const w = Math.max(1, Math.floor(cv.clientWidth));
	const h = Math.max(1, Math.floor(cv.clientHeight));
	if (cv.width !== w || cv.height !== h) { cv.width = w; cv.height = h; }

	// A zero-sized canvas draws nothing and looks identical to "the node is
	// broken". Say so once, so it is obvious this is layout and not the preview.
	if ((cv.clientWidth < 2 || cv.clientHeight < 2) && !ti._warnedSize) {
		ti._warnedSize = true;
		console.warn("[tinode] Bbox Crop: canvas has no size "
			+ `(${cv.clientWidth}x${cv.clientHeight}) — the DOM widget got no space.`);
	}

	ctx.clearRect(0, 0, cv.width, cv.height);
	ctx.fillStyle = "#181818";
	ctx.fillRect(0, 0, cv.width, cv.height);

	if (!ti.img) {
		ctx.fillStyle = "#888";
		ctx.font = "12px sans-serif";
		ctx.textAlign = "center";
		ctx.fillText("Run once to load the frame, then drag the box.", cv.width / 2, cv.height / 2);
		return;
	}

	ti.viewRect = viewRect(node);
	const v = ti.viewRect;
	ctx.drawImage(ti.img, v.ox, v.oy, ti.imgW * v.scale, ti.imgH * v.scale);

	const b = readBox(node);
	const [bx, by] = toCanvas(node, b.x, b.y);
	const bw = b.w * v.scale, bh = b.h * v.scale;

	// Dim everything outside the crop so the kept region reads clearly.
	ctx.fillStyle = "rgba(0,0,0,0.45)";
	ctx.beginPath();
	ctx.rect(v.ox, v.oy, ti.imgW * v.scale, ti.imgH * v.scale);
	ctx.rect(bx, by, bw, bh);
	ctx.fill("evenodd");

	ctx.strokeStyle = "#4aa3ff";
	ctx.lineWidth = 1.5;
	ctx.strokeRect(bx, by, bw, bh);

	// Corner handles.
	ctx.fillStyle = "#4aa3ff";
	for (const [hx, hy] of [[bx, by], [bx + bw, by], [bx, by + bh], [bx + bw, by + bh]]) {
		ctx.fillRect(hx - HANDLE_VIS, hy - HANDLE_VIS, HANDLE_VIS * 2, HANDLE_VIS * 2);
	}

	// Live readout.
	const label = `x ${b.x}   y ${b.y}   w ${b.w}   h ${b.h}`;
	ctx.font = "11px monospace";
	ctx.textAlign = "left";
	const tw = ctx.measureText(label).width;
	ctx.fillStyle = "rgba(0,0,0,0.6)";
	ctx.fillRect(4, 4, tw + 10, 18);
	ctx.fillStyle = "#e6e6e6";
	ctx.fillText(label, 9, 17);
}

function loadPreview(node, imageInfo, srcDims) {
	const ti = node._ti;
	if (srcDims && srcDims.length === 2) {
		ti.imgW = Math.max(1, Math.round(srcDims[0]));
		ti.imgH = Math.max(1, Math.round(srcDims[1]));
	}
	const url = urlFor(imageInfo);
	const img = new Image();
	img.onload = () => {
		ti.img = img;
		if (!srcDims) { ti.imgW = img.naturalWidth; ti.imgH = img.naturalHeight; }
		node.setDirtyCanvas?.(true, true);
		draw(node);
	};
	img.onerror = () => console.error("[tinode] Bbox Crop: preview failed to load", url);
	img.src = url;
}

// The DOM widget declares its OWN height: ComfyUI's DOMWidgetImpl reads
// options.getMinHeight in computeLayoutSize(), and that is what makes the node's
// size account for the canvas. Overriding node.computeSize instead either
// collapses the node on every move (default height ignores the canvas) or
// starves the widget of space — both of which we shipped by mistake.
function canvasHeightFor(node) {
	const t = node._ti;
	if (!t) return 260;
	const w = Math.max(node.size?.[0] || MIN_NODE_W, MIN_NODE_W) - 20;
	return Math.round(clamp(w * ((t.imgH || 1) / (t.imgW || 1)), 260, MAX_CANVAS_H));
}

function setupEditor(node) {
	if (node._ti) return;
	// Same layout as the two segment editors, which render reliably: a flex
	// column with the canvas as a flex child. The previous absolute/inset:0
	// canvas collapsed to zero height whenever the container's own height did
	// not resolve. min-height is a floor so the canvas is never invisible.
	const container = document.createElement("div");
	container.style.cssText = "position:relative;width:100%;height:100%;min-height:260px;"
		+ "display:flex;flex-direction:column;box-sizing:border-box;";
	const canvas = document.createElement("canvas");
	canvas.style.cssText = "flex:1;min-height:200px;width:100%;border-radius:4px;touch-action:none;display:block;";
	container.appendChild(canvas);

	node._ti = {
		container, canvas, img: null,
		imgW: 512, imgH: 512, viewRect: { ox: 0, oy: 0, scale: 1 },
		drag: null,
	};

	// Let the DOM widget fill the leftover node body; control size via the node
	// itself (below), NOT a fixed widget.computeSize that overflows the node.
	node.addDOMWidget("bbox_editor", "ti_bbox_editor", container, {
		serialize: false, hideOnZoom: false,
		getMinHeight: () => canvasHeightFor(node),
	});

	// Enforce a roomy minimum so there is space for the canvas, and grow the
	// node to it on creation.

	// Redraw whenever the container is resized (node resize, collapse, zoom).
	const ro = new ResizeObserver(() => draw(node));
	ro.observe(container);

	const pos = (e) => pointerPos(canvas, e);

	const onMove = (e) => {
		if (!node._ti.img) return;
		const [cx, cy] = pos(e);
		if (node._ti.drag) {
			const [sx, sy] = toSource(node, cx, cy);
			applyDrag(node, node._ti.drag.grip, sx, sy);
			draw(node);
			e.preventDefault();
			e.stopPropagation();
			return;
		}
		const grip = hitTest(node, cx, cy);
		canvas.style.cursor = grip ? CURSOR[grip] : "default";
	};

	const onDown = (e) => {
		if (!node._ti.img || e.button !== 0) return;
		const [cx, cy] = pos(e);
		const grip = hitTest(node, cx, cy);
		if (!grip) return;            // let LiteGraph handle clicks outside the box
		releaseGraphPointer(e);
		const b = readBox(node);
		const [sx, sy] = toSource(node, cx, cy);
		node._ti.drag = { grip, sx, sy, ox: b.x, oy: b.y };
		canvas.setPointerCapture?.(e.pointerId);
		e.preventDefault();
		e.stopPropagation();          // don't let LiteGraph start dragging the node
	};

	const onUp = (e) => {
		if (!node._ti.drag) return;
		node._ti.drag = null;
		canvas.releasePointerCapture?.(e.pointerId);
		draw(node);
	};

	canvas.addEventListener("pointerdown", onDown);
	canvas.addEventListener("pointermove", onMove);
	canvas.addEventListener("pointerup", onUp);
	canvas.addEventListener("pointercancel", onUp);
	// Swallow these so a drag inside the editor never pans/zooms the graph.
	for (const ev of ["wheel", "contextmenu"]) {
		canvas.addEventListener(ev, (e) => e.stopPropagation());
	}

	// Repaint when the number widgets are edited by hand.
	for (const name of ["x", "y", "width", "height"]) {
		const w = getWidget(node, name);
		if (!w) continue;
		const prev = w.callback;
		w.callback = function (...args) {
			const r = prev?.apply(this, args);
			// During a drag the handler repaints once per move; only redraw here
			// for manual edits of the number field.
			if (!node._ti.drag) draw(node);
			return r;
		};
	}

	// First paint (placeholder until a preview arrives).
	requestAnimationFrame(() => draw(node));
}

app.registerExtension({
	name: "tinode.bboxCrop",
	async beforeRegisterNodeDef(nodeType, nodeData) {
		if (nodeData.name !== NODE_TYPE) return;

		const onNodeCreated = nodeType.prototype.onNodeCreated;
		nodeType.prototype.onNodeCreated = function () {
			const r = onNodeCreated?.apply(this, arguments);
			setupEditor(this);
			this.setSize([
				Math.max(this.size[0], MIN_NODE_W),
				Math.max(this.size[1], MIN_NODE_H),
			]);
			return r;
		};

		const onExecuted = nodeType.prototype.onExecuted;
		nodeType.prototype.onExecuted = function (message) {
			onExecuted?.apply(this, arguments);
			const imgs = message?.ti_preview;
			if (imgs && imgs.length) {
				loadPreview(this, imgs[0], message.src_dims);
			} else {
				console.warn("[tinode] Bbox Crop: executed but no ti_preview in message", message);
			}
		};

		// Redraw after the node reloads from a saved workflow.
		const onConfigure = nodeType.prototype.onConfigure;
		nodeType.prototype.onConfigure = function () {
			const r = onConfigure?.apply(this, arguments);
			if (this._ti) requestAnimationFrame(() => draw(this));
			return r;
		};
	},
});
