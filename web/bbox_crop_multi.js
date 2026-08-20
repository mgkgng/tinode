// Multi-box editor for "Bbox Crop · Multi (ti)".
//
// Like the single Bbox Crop editor, but you draw MANY boxes: drag on empty
// space to create one, grab a corner/edge/interior to reshape or move it,
// right-click a box to delete it. Each box becomes its own crop (its own SAM3
// pass, its own removal), so this is where you mark every region to fix.
//
// It scrubs the WHOLE clip — slider, ◀ ▶ buttons, arrow keys — because drawing
// a crop off frame 0 alone is a trap: the subject moves, so a box that fits at
// the start can miss it by the end. Check the beginning AND the end before you
// commit a box. The box list is serialized into a hidden `boxes` widget as
// [{x,y,w,h}, ...] in TRUE source pixels and saved with the workflow.

import { app } from "../../scripts/app.js";
import {
	clamp, colorForId, constrainToRatio, fillNodeWidth, getWidget, hideWidget,
	parseRatios, pointerPos, ratioForIndex, releaseGraphPointer, urlFor,
} from "./lib/editor.js";

// Per-box ratios: box i uses the i-th ratio in the comma-separated list (or the
// last, so one ratio applies to every box).
const ratiosOf = (node) => parseRatios(getWidget(node, "aspect_ratio")?.value);

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
		// Reach HANDLE_HIT OUTSIDE the edge, but only a third of the box INWARD, so
		// a small box always keeps a central zone for moving instead of resizing.
		const inMx = Math.min(HANDLE_HIT, (x1 - x0) / 3);
		const inMy = Math.min(HANDLE_HIT, (y1 - y0) / 3);
		const nearL = cx >= x0 - HANDLE_HIT && cx <= x0 + inMx;
		const nearR = cx <= x1 + HANDLE_HIT && cx >= x1 - inMx;
		const nearT = cy >= y0 - HANDLE_HIT && cy <= y0 + inMy;
		const nearB = cy <= y1 + HANDLE_HIT && cy >= y1 - inMy;
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
	const box = constrainToRatio(
		{ x: left, y: top, w: right - left, h: bottom - top }, d.grip,
		ratioForIndex(ratiosOf(node), d.i));
	b.x = box.x; b.y = box.y; b.w = box.w; b.h = box.h;
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
		ctx.fillText("Run once to load the clip, then scrub and drag to add crops.", cv.width / 2, cv.height / 2);
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
	const fr = ti.frames.length ? `f ${ti.frame}/${ti.frames.length - 1}   ` : "";
	const label = b
		? `${fr}#${ti.boxes.length} crop(s)   sel x ${Math.round(b.x)} y ${Math.round(b.y)} w ${Math.round(b.w)} h ${Math.round(b.h)}`
		: `${fr}drag to add a crop · right-click a box to delete`;
	ctx.font = "11px monospace";
	ctx.textAlign = "left";
	const tw = ctx.measureText(label).width;
	ctx.fillStyle = "rgba(0,0,0,0.6)";
	ctx.fillRect(4, 4, tw + 10, 18);
	ctx.fillStyle = "#e6e6e6";
	ctx.fillText(label, 9, 17);
}

// --- frames ---------------------------------------------------------------
function showFrame(node, i) {
	const ti = node._ti;
	if (!ti.frames.length) return;
	ti.frame = clamp(i, 0, ti.frames.length - 1);
	if (ti.slider) ti.slider.value = String(ti.frame);
	if (ti.label) ti.label.textContent = `${ti.frame}/${ti.frames.length - 1}`;
	// Remember the frame in node.properties: it is pure UI state, so it must not
	// be a backend input (a saved graph carrying a widget the node no longer has
	// fails validation before execute ever runs).
	node.properties = node.properties || {};
	node.properties.ti_frame = ti.frame;

	const cached = ti.cache.get(ti.frame);
	if (cached) { ti.img = cached; draw(node); return; }
	const img = new Image();
	const want = ti.frame;
	img.onload = () => {
		ti.cache.set(want, img);
		if (ti.frame === want) { ti.img = img; draw(node); }
	};
	img.onerror = () => console.error("[tinode] Bbox Crop Multi: frame failed to load", want);
	img.src = urlFor(ti.frames[want]);
}

function step(node, d) { showFrame(node, node._ti.frame + d); }

// The whole clip arrives as a manifest of per-frame images, so the editor can
// scrub start to end rather than judging a box off frame 0.
function applyManifest(node, m) {
	const ti = node._ti;
	if (!m || !Array.isArray(m.frames) || !m.frames.length) return;
	ti.imgW = Math.max(1, Math.round(m.full_w || ti.imgW));
	ti.imgH = Math.max(1, Math.round(m.full_h || ti.imgH));
	if (m.sig !== ti.sig) { ti.cache.clear(); ti.sig = m.sig; }
	ti.frames = m.frames;
	if (ti.slider) ti.slider.max = String(ti.frames.length - 1);
	const saved = Math.round(node.properties?.ti_frame ?? 0);
	showFrame(node, clamp(saved, 0, ti.frames.length - 1));
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

	// Frame transport: ◀ ▶ + slider, so the whole clip is reachable.
	const bar = document.createElement("div");
	bar.style.cssText = "display:flex;align-items:center;gap:6px;padding:4px 2px 0;"
		+ "font:11px monospace;color:#ccc;flex:0 0 auto;";
	const mkBtn = (txt, d) => {
		const b = document.createElement("button");
		b.textContent = txt;
		b.style.cssText = "background:#333;color:#eee;border:1px solid #555;border-radius:3px;"
			+ "cursor:pointer;padding:1px 7px;font:12px monospace;";
		b.onclick = (ev) => { ev.stopPropagation(); step(node, d); };
		return b;
	};
	const slider = document.createElement("input");
	slider.type = "range"; slider.min = "0"; slider.max = "0"; slider.value = "0";
	slider.style.cssText = "flex:1;min-width:40px;accent-color:#4aa3ff;";
	slider.oninput = (ev) => { ev.stopPropagation(); showFrame(node, parseInt(slider.value, 10)); };
	const label = document.createElement("span");
	label.textContent = "0/0";
	label.style.cssText = "min-width:56px;text-align:right;";
	bar.appendChild(mkBtn("◀", -1));
	bar.appendChild(slider);
	bar.appendChild(mkBtn("▶", 1));
	bar.appendChild(label);
	container.appendChild(bar);

	node._ti = {
		container, canvas, img: null, imgW: 512, imgH: 512,
		viewRect: { ox: 0, oy: 0, scale: 1 },
		boxes: readBoxes(node), sel: null, drag: null, creating: null,
		frames: [], frame: 0, cache: new Map(), bar: null, slider: null, label: null,
	};

	node._ti.bar = bar; node._ti.slider = slider; node._ti.label = label;

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
			const r = ratioForIndex(ratiosOf(node), node._ti.boxes.length);
			if (r) {                       // grow from the start corner, ratio-locked
				const nw = Math.max(c.w, c.h * r);
				c.w = nw; c.h = nw / r;
				c.x = (sx < c._sx) ? c._sx - c.w : c._sx;
				c.y = (sy < c._sy) ? c._sy - c.h : c._sy;
			}
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

	// Arrow keys step frames while the pointer is over the editor.
	canvas.tabIndex = 0;
	canvas.addEventListener("keydown", (e) => {
		if (e.key === "ArrowLeft") { step(node, -1); e.preventDefault(); e.stopPropagation(); }
		else if (e.key === "ArrowRight") { step(node, 1); e.preventDefault(); e.stopPropagation(); }
	});
	canvas.addEventListener("pointerenter", () => canvas.focus({ preventScroll: true }));

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
			const m = message?.ti_bboxm;
			if (m && m.length) applyManifest(this, m[0]);
			else console.warn("[tinode] Bbox Crop Multi: no ti_bboxm in message", message);
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
