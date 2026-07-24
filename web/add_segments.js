// Interactive manual-segment editor for the "Add Segments (ti)" node.
//
// Same feel as Pick Segments — scrub frames with the slider / arrows — but here
// you ADD regions SAM3 missed: drag a box on the current frame to create a
// segment, right-click a box you drew to remove it. Existing segments are drawn
// dim for reference (not editable here; that's Pick Segments' job).
//
// Manual boxes are per-frame and stored in SOURCE pixels in the hidden
// manual_segments widget (serialized with the workflow). The Python node turns
// each into a filled-rectangle segment and merges it into the outgoing stream.

import { app } from "../../scripts/app.js";
import {
	colorForId, clamp, getWidget, urlFor, pointerPos, releaseGraphPointer,
} from "./lib/editor.js";

const NODE_TYPE = "TI_AddSegments";
const MIN_NODE_W = 360;
const MIN_NODE_H = 320;
const MANUAL_ID_BASE = 1000000;
const MIN_DRAW = 5;                 // ignore accidental micro-drags (canvas px)
const MANUAL_RGB = [255, 220, 0];   // manual boxes: bright yellow
function readManual(node) { return readManualObj(node).items; }

function writeManual(node, arr) {
	const w = getWidget(node, "manual_segments");
	if (!w) return;
	const sig = node._tas?.manifest?.sig ?? readManualObj(node).sig ?? null;
	w.value = JSON.stringify({ sig, items: arr });
	w.callback?.(w.value, app.canvas, node);
}
function nextManualId(arr) {
	return arr.length ? Math.max(...arr.map((m) => m.id)) + 1 : MANUAL_ID_BASE;
}
function setFrameWidget(node, f) {
	const w = getWidget(node, "current_frame");
	if (w && w.value !== f) { w.value = f; w.callback?.(f, app.canvas, node); }
}

function viewRect(tas) {
	const cw = tas.canvas.width, ch = tas.canvas.height;
	const scale = Math.min(cw / tas.natW, ch / tas.natH);
	return { ox: (cw - tas.natW * scale) / 2, oy: (ch - tas.natH * scale) / 2, scale };
}

// SOURCE px <-> canvas px (natW/natH are the preview size pw/ph).
function srcToCanvas(tas, sx, sy) {
	const v = tas.view;
	return [v.ox + (sx * tas.natW / tas.fullW) * v.scale,
			v.oy + (sy * tas.natH / tas.fullH) * v.scale];
}
function canvasToSrc(tas, cx, cy) {
	const v = tas.view;
	return [((cx - v.ox) / v.scale) * tas.fullW / tas.natW,
			((cy - v.oy) / v.scale) * tas.fullH / tas.natH];
}

function loadFrame(node, f) {
	const tas = node._tas;
	if (!tas || !tas.manifest) return;
	f = clamp(f, 0, tas.manifest.num_frames - 1);
	tas.frameIdx = f;
	setFrameWidget(node, f);
	if (tas.slider) tas.slider.value = String(f);
	updateCounter(node);

	let fr = tas.frames[f];
	if (!fr) {
		const meta = tas.manifest.frames[f];
		fr = tas.frames[f] = { meta, src: null };
		const img = new Image();
		img.onload = () => { fr.src = img; if (tas.frameIdx === f) draw(node); };
		img.onerror = () => console.error("[tinode] Add Segments: src load failed", f);
		img.src = urlFor(meta.src);
	}
	draw(node);
}

function draw(node) {
	const tas = node._tas;
	if (!tas) return;
	const cv = tas.canvas, ctx = cv.getContext("2d");
	const w = Math.max(1, Math.floor(cv.clientWidth));
	const h = Math.max(1, Math.floor(cv.clientHeight));
	if (cv.width !== w || cv.height !== h) { cv.width = w; cv.height = h; }

	ctx.clearRect(0, 0, cv.width, cv.height);
	ctx.fillStyle = "#141414";
	ctx.fillRect(0, 0, cv.width, cv.height);

	if (!tas.manifest) {
		ctx.fillStyle = "#888"; ctx.font = "12px sans-serif"; ctx.textAlign = "center";
		ctx.fillText("Run once to load frames, then drag to add a box.",
			cv.width / 2, cv.height / 2);
		return;
	}

	tas.view = viewRect(tas);
	const v = tas.view;
	const fr = tas.frames[tas.frameIdx];
	if (fr && fr.src) ctx.drawImage(fr.src, v.ox, v.oy, tas.natW * v.scale, tas.natH * v.scale);
	if (!fr || !fr.meta) return;

	// Existing segments: dim reference boxes (preview-px bbox).
	ctx.lineWidth = 1;
	ctx.setLineDash([3, 3]);
	for (const s of fr.meta.segs) {
		const [x0, y0, x1, y1] = s.bbox;
		const [r, g, b] = colorForId(s.id);
		ctx.strokeStyle = `rgba(${r},${g},${b},0.35)`;
		ctx.strokeRect(v.ox + x0 * v.scale, v.oy + y0 * v.scale,
			(x1 - x0) * v.scale, (y1 - y0) * v.scale);
	}
	ctx.setLineDash([]);

	// Manual boxes for this frame: bright, solid, with a remove hint.
	const [mr, mg, mb] = MANUAL_RGB;
	ctx.lineWidth = 2;
	ctx.font = "10px monospace";
	ctx.textAlign = "left";
	for (const m of readManual(node)) {
		if (m.frame !== tas.frameIdx) continue;
		const [cx0, cy0] = srcToCanvas(tas, m.bbox[0], m.bbox[1]);
		const [cx1, cy1] = srcToCanvas(tas, m.bbox[2], m.bbox[3]);
		ctx.strokeStyle = `rgb(${mr},${mg},${mb})`;
		ctx.strokeRect(cx0, cy0, cx1 - cx0, cy1 - cy0);
		ctx.fillStyle = `rgb(${mr},${mg},${mb})`;
		ctx.fillRect(cx0, cy0 - 12, 34, 12);
		ctx.fillStyle = "#000";
		ctx.fillText(`+${m.id - MANUAL_ID_BASE}`, cx0 + 2, cy0 - 2);
	}

	// In-progress rubber band.
	if (tas.creating) {
		const c = tas.creating;
		ctx.setLineDash([5, 3]);
		ctx.strokeStyle = `rgba(${mr},${mg},${mb},0.9)`;
		ctx.strokeRect(Math.min(c.x0, c.x1), Math.min(c.y0, c.y1),
			Math.abs(c.x1 - c.x0), Math.abs(c.y1 - c.y0));
		ctx.setLineDash([]);
	}
}

function updateCounter(node) {
	const tas = node._tas;
	if (!tas.manifest) return;
	const added = readManual(node).length;
	const onFrame = readManual(node).filter((m) => m.frame === tas.frameIdx).length;
	tas.counter.textContent =
		`frame ${tas.frameIdx + 1}/${tas.manifest.num_frames}   ·   +${onFrame} here / ${added} total`;
}

// Smallest manual box (this frame) under a canvas point — for right-click remove.
function manualAt(node, cx, cy) {
	const tas = node._tas;
	let best = null, bestArea = Infinity;
	for (const m of readManual(node)) {
		if (m.frame !== tas.frameIdx) continue;
		const [x0, y0] = srcToCanvas(tas, m.bbox[0], m.bbox[1]);
		const [x1, y1] = srcToCanvas(tas, m.bbox[2], m.bbox[3]);
		if (cx >= x0 && cx <= x1 && cy >= y0 && cy <= y1) {
			const area = (x1 - x0) * (y1 - y0);
			if (area < bestArea) { best = m; bestArea = area; }
		}
	}
	return best;
}

function setup(node) {
	if (node._tas) return;
	const wrap = document.createElement("div");
	wrap.style.cssText = "position:relative;width:100%;height:100%;display:flex;flex-direction:column;box-sizing:border-box;";

	const bar = document.createElement("div");
	bar.style.cssText = "display:flex;align-items:center;gap:6px;padding:4px;flex:0 0 auto;";
	const mkBtn = (txt) => {
		const b = document.createElement("button");
		b.textContent = txt;
		b.style.cssText = "background:#2a2f3a;color:#cfe3ff;border:1px solid #3a4150;border-radius:4px;font:600 12px sans-serif;padding:2px 8px;cursor:pointer;";
		return b;
	};
	const prev = mkBtn("◀"), next = mkBtn("▶");
	const slider = document.createElement("input");
	slider.type = "range"; slider.min = "0"; slider.max = "0"; slider.value = "0";
	slider.style.cssText = "flex:1;min-width:0;";
	const counter = document.createElement("span");
	counter.style.cssText = "font:11px monospace;color:#cfd3da;white-space:nowrap;";
	bar.append(prev, slider, next, counter);

	const canvas = document.createElement("canvas");
	canvas.style.cssText = "flex:1;min-height:0;width:100%;border-radius:4px;touch-action:none;display:block;cursor:crosshair;";
	wrap.append(bar, canvas);

	node._tas = {
		wrap, bar, canvas, slider, counter, prev, next,
		manifest: null, frames: [], frameIdx: 0,
		natW: 512, natH: 512, fullW: 512, fullH: 512,
		view: { ox: 0, oy: 0, scale: 1 }, creating: null,
	};

	node.addDOMWidget("add_seg_editor", "ti_add_editor", wrap, { serialize: false, hideOnZoom: false });
	for (const name of ["manual_segments", "current_frame"]) {
		const w = getWidget(node, name);
		if (w) { w.type = "hidden"; w.computeSize = () => [0, -4]; }
	}

	const prevCompute = node.computeSize;
	node.computeSize = function () {
		const base = prevCompute ? prevCompute.apply(this, arguments) : [MIN_NODE_W, MIN_NODE_H];
		return [Math.max(base[0], MIN_NODE_W), Math.max(base[1], MIN_NODE_H)];
	};
	new ResizeObserver(() => draw(node)).observe(wrap);

	// Navigation.
	prev.onclick = (e) => { e.stopPropagation(); loadFrame(node, node._tas.frameIdx - 1); };
	next.onclick = (e) => { e.stopPropagation(); loadFrame(node, node._tas.frameIdx + 1); };
	slider.addEventListener("input", (e) => { e.stopPropagation(); loadFrame(node, parseInt(slider.value, 10)); });
	for (const el of [prev, next, slider]) el.addEventListener("pointerdown", (e) => e.stopPropagation());

	wrap.tabIndex = 0;
	wrap.addEventListener("pointerenter", () => wrap.focus({ preventScroll: true }));
	wrap.addEventListener("keydown", (e) => {
		if (e.key === "ArrowLeft") { loadFrame(node, node._tas.frameIdx - 1); e.stopPropagation(); e.preventDefault(); }
		else if (e.key === "ArrowRight") { loadFrame(node, node._tas.frameIdx + 1); e.stopPropagation(); e.preventDefault(); }
	});

	const pos = (e) => pointerPos(canvas, e);

	// Left-drag = create a box.
	canvas.addEventListener("pointerdown", (e) => {
		if (!node._tas.manifest || e.button !== 0) return;
		releaseGraphPointer(e);
		const [cx, cy] = pos(e);
		node._tas.creating = { x0: cx, y0: cy, x1: cx, y1: cy };
		canvas.setPointerCapture?.(e.pointerId);
		e.stopPropagation(); e.preventDefault();
	});
	canvas.addEventListener("pointermove", (e) => {
		if (!node._tas.creating) return;
		const [cx, cy] = pos(e);
		node._tas.creating.x1 = cx; node._tas.creating.y1 = cy;
		draw(node);
		e.stopPropagation();
	});
	canvas.addEventListener("pointerup", (e) => {
		const c = node._tas.creating;
		if (!c) return;
		node._tas.creating = null;
		canvas.releasePointerCapture?.(e.pointerId);
		if (Math.abs(c.x1 - c.x0) >= MIN_DRAW && Math.abs(c.y1 - c.y0) >= MIN_DRAW) {
			const [sx0, sy0] = canvasToSrc(node._tas, Math.min(c.x0, c.x1), Math.min(c.y0, c.y1));
			const [sx1, sy1] = canvasToSrc(node._tas, Math.max(c.x0, c.x1), Math.max(c.y0, c.y1));
			const arr = readManual(node);
			arr.push({
				id: nextManualId(arr), frame: node._tas.frameIdx,
				bbox: [Math.round(sx0), Math.round(sy0), Math.round(sx1), Math.round(sy1)],
			});
			writeManual(node, arr);
			updateCounter(node);
		}
		draw(node);
	});

	// Right-click = remove the box under the cursor.
	canvas.addEventListener("contextmenu", (e) => {
		e.preventDefault(); e.stopPropagation();
		if (!node._tas.manifest) return;
		const [cx, cy] = pos(e);
		const hit = manualAt(node, cx, cy);
		if (hit) {
			writeManual(node, readManual(node).filter((m) => m.id !== hit.id));
			updateCounter(node);
			draw(node);
		}
	});

	requestAnimationFrame(() => draw(node));
}

function fitNodeToAspect(node) {
	const tas = node._tas;
	if (!tas.manifest || !tas.natW || !tas.natH) return;
	const barH = tas.bar.offsetHeight || 32;
	const width = Math.max(node.size[0], MIN_NODE_W);
	const canvasH = clamp((width - 20) * (tas.natH / tas.natW), 200, 760);
	node.setSize([width, Math.round(barH + canvasH + 20)]);
	node.setDirtyCanvas?.(true, true);
	requestAnimationFrame(() => draw(node));
}

function applyManifest(node, manifest) {
	const tas = node._tas;
	// A different input (image and/or segments) invalidates everything drawn
	// against the old one — reset instead of stamping stale boxes onto a new clip.
	const prev = readManualObj(node);
	const isNewInput = prev.sig !== manifest.sig;

	tas.manifest = manifest;
	tas.frames = [];
	tas.natW = manifest.pw || tas.natW;
	tas.natH = manifest.ph || tas.natH;
	tas.fullW = manifest.full_w || tas.natW;
	tas.fullH = manifest.full_h || tas.natH;
	tas.slider.max = String(Math.max(0, manifest.num_frames - 1));

	if (isNewInput) {
		writeManual(node, []);                       // clears, stamped with the new sig
		setFrameWidget(node, 0);
		if (prev.items.length) {
			console.info(`[tinode] Add Segments: new input — cleared ${prev.items.length} drawn box(es).`);
		}
	}

	fitNodeToAspect(node);
	const start = isNewInput
		? 0
		: clamp(getWidget(node, "current_frame")?.value ?? 0, 0, manifest.num_frames - 1);
	loadFrame(node, start);
}

app.registerExtension({
	name: "tinode.addSegments",
	async beforeRegisterNodeDef(nodeType, nodeData) {
		if (nodeData.name !== NODE_TYPE) return;

		const onNodeCreated = nodeType.prototype.onNodeCreated;
		nodeType.prototype.onNodeCreated = function () {
			const r = onNodeCreated?.apply(this, arguments);
			setup(this);
			this.setSize([Math.max(this.size[0], MIN_NODE_W), Math.max(this.size[1], MIN_NODE_H)]);
			return r;
		};

		const onExecuted = nodeType.prototype.onExecuted;
		nodeType.prototype.onExecuted = function (message) {
			onExecuted?.apply(this, arguments);
			const m = message?.ti_add;
			if (m && m.length) applyManifest(this, m[0]);
			else console.warn("[tinode] Add Segments: no ti_add in message", message);
		};

		const onConfigure = nodeType.prototype.onConfigure;
		nodeType.prototype.onConfigure = function () {
			const r = onConfigure?.apply(this, arguments);
			if (this._tas) requestAnimationFrame(() => draw(this));
			return r;
		};
	},
});
