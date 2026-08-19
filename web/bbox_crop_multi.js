// Multi-box editor for "Bbox Crop · Multi (ti)".
//
// Like the single Bbox Crop editor, but you draw MANY boxes: drag on empty
// space to create one, grab a corner/edge/interior to reshape or move it,
// right-click a box to delete it. Each box becomes its own crop (its own SAM3
// pass, its own removal), so this is where you mark every region to fix in a
// frame. The box list is serialized into a hidden `boxes` widget as
// [{x,y,w,h}, ...] in TRUE source pixels and saved with the workflow.

import { app } from "../../scripts/app.js";
import {
	clamp, colorForId, fillNodeWidth, getWidget, hideWidget, pointerPos,
	releaseGraphPointer, urlFor,
} from "./lib/editor.js";

const NODE_TYPE = "TI_BboxCropMulti";
const HANDLE_HIT = 12;
const HANDLE_VIS = 5;
const MIN_BOX = 4;          // smallest box worth keeping, source px
const MIN_NODE_W = 320;
const MIN_NODE_H = 460;
const MAX_CANVAS_H = 2000;

function readBoxes(node) {
	const w = getWidget(node, "boxes");
	try {
		const d = JSON.parse(w?.value || "[]");
		return Array.isArray(d) ? d.filter(b => b && "x" in b && "y" in b && "w" in b && "h" in b) : [];
	} catch { return []; }
}

function writeBoxes(node) {
	const w = getWidget(node, "boxes");
	if (!w) return;
	const clean = node._ti.boxes.map(b => ({
		x: Math.round(b.x), y: Math.round(b.y),
		w: Math.max(MIN_BOX, Math.round(b.w)), h: Math.max(MIN_BOX, Math.round(b.h)),
	}));
	w.value = JSON.stringify(clean);
	w.callback?.(w.value, app.canvas, node);
}

function viewRect(node) {
	const cv = node._ti.canvas;
	const iw = node._ti.imgW, ih = node._ti.imgH;
	const scale = Math.min(cv.width / iw, cv.height / ih);
	return { ox: (cv.width - iw * scale) / 2, oy: (cv.height - ih * scale) / 2, scale };
}
function toCanvas(node, sx, sy) { const v = node._ti.viewRect; return [v.ox + sx * v.scale, v.oy + sy * v.scale]; }
function toSource(node, cx, cy) { const v = node._ti.viewRect; return [(cx - v.ox) / v.scale, (cy - v.oy) / v.scale]; }

// Topmost box (+ grip) under the cursor, or null. Later boxes draw on top, so
// hit-test them first.
function hitTest(node, cx, cy) {
	const boxes = node._ti.boxes;
	for (let i = boxes.length - 1; i >= 0; i--) {
		const b = boxes[i];
		const [x0, y0] = toCanvas(node, b.x, b.y);
		const [x1, y1] = toCanvas(node, b.x + b.w, b.y + b.h);
		const corners = { nw: [x0, y0], ne: [x1, y0], sw: [x0, y1], se: [x1, y1] };
		for (const [name, [px, py]] of Object.entries(corners)) {
			if (Math.hypot(cx - px, cy - py) <= HANDLE_HIT) return { i, grip: name };
		}
		const nearL = Math.abs(cx - x0) <= HANDLE_HIT, nearR = Math.abs(cx - x1) <= HANDLE_HIT;
		const nearT = Math.abs(cy - y0) <= HANDLE_HIT, nearB = Math.abs(cy - y1) <= HANDLE_HIT;
		const inYs = cy >= y0 - HANDLE_HIT && cy <= y1 + HANDLE_HIT;
		const inXs = cx >= x0 - HANDLE_HIT && cx <= x1 + HANDLE_HIT;
		if (nearL && inYs) return { i, grip: "w" };
		if (nearR && inYs) return { i, grip: "e" };
		if (nearT && inXs) return { i, grip: "n" };
		if (nearB && inXs) return { i, grip: "s" };
		if (cx > x0 && cx < x1 && cy > y0 && cy < y1) return { i, grip: "move" };
	}
	return null;
}

const CURSOR = {
	nw: "nwse-resize", se: "nwse-resize", ne: "nesw-resize", sw: "nesw-resize",
	n: "ns-resize", s: "ns-resize", e: "ew-resize", w: "ew-resize", move: "move",
};

function applyDrag(node, sx, sy) {
	const d = node._ti.drag;
	const b = node._ti.boxes[d.i];
	if (!b) return;
	const iw = node._ti.imgW, ih = node._ti.imgH;
	sx = clamp(sx, 0, iw); sy = clamp(sy, 0, ih);
	if (d.grip === "move") {
		b.x = clamp(d.ox + (sx - d.sx), 0, iw - b.w);
		b.y = clamp(d.oy + (sy - d.sy), 0, ih - b.h);
		return;
	}
	let left = b.x, top = b.y, right = b.x + b.w, bottom = b.y + b.h;
	if (d.grip.includes("w")) left = Math.min(sx, right - MIN_BOX);
	if (d.grip.includes("e")) right = Math.max(sx, left + MIN_BOX);
	if (d.grip.includes("n")) top = Math.min(sy, bottom - MIN_BOX);
	if (d.grip.includes("s")) bottom = Math.max(sy, top + MIN_BOX);
	b.x = left; b.y = top; b.w = right - left; b.h = bottom - top;
}

function draw(node) {
	const ti = node._ti;
	const cv = ti.canvas;
	const ctx = cv.getContext("2d");
	fillNodeWidth(node, ti.container);
	const w = Math.max(1, Math.floor(cv.clientWidth));
	const h = Math.max(1, Math.floor(cv.clientHeight));
	if (cv.width !== w || cv.height !== h) { cv.width = w; cv.height = h; }

	ctx.clearRect(0, 0, cv.width, cv.height);
	ctx.fillStyle = "#181818";
	ctx.fillRect(0, 0, cv.width, cv.height);

	if (!ti.img) {
		ctx.fillStyle = "#888";
		ctx.font = "12px sans-serif";
		ctx.textAlign = "center";
		ctx.fillText("Run once to load the frame, then drag to add crops.", cv.width / 2, cv.height / 2);
		return;
	}

	ti.viewRect = viewRect(node);
	const v = ti.viewRect;
	ctx.drawImage(ti.img, v.ox, v.oy, ti.imgW * v.scale, ti.imgH * v.scale);

	const drawn = ti.creating ? ti.boxes.concat([ti.creating]) : ti.boxes;
	drawn.forEach((b, i) => {
		const [bx, by] = toCanvas(node, b.x, b.y);
		const bw = b.w * v.scale, bh = b.h * v.scale;
		const [r, g, bl] = colorForId(i + 1);
		const col = `rgb(${r},${g},${bl})`;
		ctx.strokeStyle = col;
		ctx.lineWidth = 1.5;
		ctx.strokeRect(bx, by, bw, bh);
		ctx.fillStyle = col;
		for (const [hx, hy] of [[bx, by], [bx + bw, by], [bx, by + bh], [bx + bw, by + bh]]) {
			ctx.fillRect(hx - HANDLE_VIS, hy - HANDLE_VIS, HANDLE_VIS * 2, HANDLE_VIS * 2);
		}
		// index badge
		ctx.fillStyle = "rgba(0,0,0,0.6)";
		ctx.fillRect(bx, by - 15, 20, 14);
		ctx.fillStyle = col;
		ctx.font = "11px monospace";
		ctx.textAlign = "left";
		ctx.fillText(String(i), bx + 5, by - 4);
	});

	// Readout for the selected (or newest) box.
	const b = ti.creating || ti.boxes[ti.sel] || ti.boxes[ti.boxes.length - 1];
	const label = b
		? `#${ti.boxes.length} crop(s)   sel x ${Math.round(b.x)} y ${Math.round(b.y)} w ${Math.round(b.w)} h ${Math.round(b.h)}`
		: "drag to add a crop · right-click a box to delete";
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
	const img = new Image();
	img.onload = () => {
		ti.img = img;
		if (!srcDims) { ti.imgW = img.naturalWidth; ti.imgH = img.naturalHeight; }
		node.setDirtyCanvas?.(true, true);
		draw(node);
	};
	img.onerror = () => console.error("[tinode] Bbox Crop Multi: preview failed to load");
	img.src = urlFor(imageInfo);
}

function canvasHeightFor(node) {
	const t = node._ti;
	if (!t) return 260;
	const w = Math.max(node.size?.[0] || MIN_NODE_W, MIN_NODE_W) - 20;
	return Math.round(clamp(w * ((t.imgH || 1) / (t.imgW || 1)), 260, MAX_CANVAS_H));
}

function setupEditor(node) {
	if (node._ti) return;
	const container = document.createElement("div");
	container.style.cssText = "position:relative;width:100%;height:100%;min-height:260px;"
		+ "display:flex;flex-direction:column;box-sizing:border-box;";
	const canvas = document.createElement("canvas");
	canvas.style.cssText = "flex:1;min-height:200px;width:100%;border-radius:4px;touch-action:none;display:block;";
	container.appendChild(canvas);

	node._ti = {
		container, canvas, img: null, imgW: 512, imgH: 512,
		viewRect: { ox: 0, oy: 0, scale: 1 },
		boxes: readBoxes(node), sel: null, drag: null, creating: null,
	};

	hideWidget(node, "boxes");

	node.addDOMWidget("bbox_multi_editor", "ti_bbox_multi_editor", container, {
		serialize: false, hideOnZoom: false, getMinHeight: () => canvasHeightFor(node),
	});

	const ro = new ResizeObserver(() => draw(node));
	ro.observe(container);
	const pos = (e) => pointerPos(canvas, e);

	const onMove = (e) => {
		if (!node._ti.img) return;
		const [cx, cy] = pos(e);
		const [sx, sy] = toSource(node, cx, cy);
		if (node._ti.creating) {
			const c = node._ti.creating;
			c.w = Math.abs(sx - c._sx); c.h = Math.abs(sy - c._sy);
			c.x = Math.min(sx, c._sx); c.y = Math.min(sy, c._sy);
			draw(node); e.preventDefault(); e.stopPropagation(); return;
		}
		if (node._ti.drag) {
			applyDrag(node, sx, sy);
			draw(node); e.preventDefault(); e.stopPropagation(); return;
		}
		const hit = hitTest(node, cx, cy);
		canvas.style.cursor = hit ? CURSOR[hit.grip] : "crosshair";
	};

	const onDown = (e) => {
		if (!node._ti.img || e.button !== 0) return;
		const [cx, cy] = pos(e);
		releaseGraphPointer(e);
		const hit = hitTest(node, cx, cy);
		if (hit) {
			const b = node._ti.boxes[hit.i];
			node._ti.sel = hit.i;
			node._ti.drag = { i: hit.i, grip: hit.grip, sx: toSource(node, cx, cy)[0],
				sy: toSource(node, cx, cy)[1], ox: b.x, oy: b.y };
		} else {
			const [sx, sy] = toSource(node, cx, cy);
			node._ti.creating = { x: sx, y: sy, w: 0, h: 0, _sx: sx, _sy: sy };
		}
		canvas.setPointerCapture?.(e.pointerId);
		e.preventDefault(); e.stopPropagation();
	};

	const onUp = (e) => {
		const ti = node._ti;
		if (ti.creating) {
			const c = ti.creating;
			if (c.w >= MIN_BOX && c.h >= MIN_BOX) {
				ti.boxes.push({ x: c.x, y: c.y, w: c.w, h: c.h });
				ti.sel = ti.boxes.length - 1;
				writeBoxes(node);
			}
			ti.creating = null;
			draw(node);
		} else if (ti.drag) {
			ti.drag = null;
			writeBoxes(node);
			draw(node);
		}
		canvas.releasePointerCapture?.(e.pointerId);
	};

	const onContext = (e) => {
		e.preventDefault(); e.stopPropagation();
		if (!node._ti.img) return;
		const [cx, cy] = pos(e);
		const hit = hitTest(node, cx, cy);
		if (hit) {
			node._ti.boxes.splice(hit.i, 1);
			node._ti.sel = null;
			writeBoxes(node);
			draw(node);
		}
	};

	canvas.addEventListener("pointerdown", onDown);
	canvas.addEventListener("pointermove", onMove);
	canvas.addEventListener("pointerup", onUp);
	canvas.addEventListener("pointercancel", onUp);
	canvas.addEventListener("contextmenu", onContext);
	canvas.addEventListener("wheel", (e) => e.stopPropagation());

	requestAnimationFrame(() => draw(node));
}

app.registerExtension({
	name: "tinode.bboxCropMulti",
	async beforeRegisterNodeDef(nodeType, nodeData) {
		if (nodeData.name !== NODE_TYPE) return;

		const onNodeCreated = nodeType.prototype.onNodeCreated;
		nodeType.prototype.onNodeCreated = function () {
			const r = onNodeCreated?.apply(this, arguments);
			setupEditor(this);
			this.setSize([Math.max(this.size[0], MIN_NODE_W), Math.max(this.size[1], MIN_NODE_H)]);
			return r;
		};

		const onExecuted = nodeType.prototype.onExecuted;
		nodeType.prototype.onExecuted = function (message) {
			onExecuted?.apply(this, arguments);
			const imgs = message?.ti_preview;
			if (imgs && imgs.length) loadPreview(this, imgs[0], message.src_dims);
		};

		const onConfigure = nodeType.prototype.onConfigure;
		nodeType.prototype.onConfigure = function () {
			const r = onConfigure?.apply(this, arguments);
			if (this._ti) {
				this._ti.boxes = readBoxes(this);
				requestAnimationFrame(() => draw(this));
			}
			return r;
		};
	},
});
