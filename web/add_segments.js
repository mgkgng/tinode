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
	colorForId, clamp, fillNodeWidth, getWidget, urlFor, pointerPos, releaseGraphPointer,
} from "./lib/editor.js";

const NODE_TYPE = "TI_AddSegments";
const MIN_NODE_W = 360;
// Width the editor grows to on load — a segment picker is unusable small.
// Only ever grows: a node you widened yourself keeps its size.
const PREFERRED_W = 720;
const MAX_CANVAS_H = 2000;
const MIN_NODE_H = 320;
const MANUAL_ID_BASE = 1000000;
const MIN_DRAW = 5;                 // ignore accidental micro-drags (canvas px)
const MANUAL_RGB = [255, 220, 0];   // manual boxes: bright yellow

// Boxes are stored as {sig, items} — sig identifies the image+segments they were
// drawn against, so a new input can drop them instead of re-applying them.
function readManualObj(node) {
	try {
		const v = JSON.parse(getWidget(node, "manual_segments")?.value || "{}");
		if (Array.isArray(v)) return { sig: null, items: v };   // legacy bare list
		if (!v || typeof v !== "object") return { sig: null, items: [] };
		return { sig: v.sig ?? null, items: Array.isArray(v.items) ? v.items : [] };
	} catch { return { sig: null, items: [] }; }
}

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
		fr = tas.frames[f] = { meta, src: null, mask: null };
		if (meta.mask) {
			const mimg = new Image();
			mimg.onload = () => { fr.mask = mimg; if (tas.frameIdx === f) draw(node); };
			mimg.src = urlFor(meta.mask);
		}
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
	fillNodeWidth(node, tas.container || tas.wrap);
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

	// The CURRENT MASK, filled. A bounding rectangle tells you nothing about the
	// actual mask shape, so the real thing is drawn — tinted, at a controllable
	// opacity, with a bright edge so the boundary is unmistakable.
	if (fr.mask && tas.maskAlpha > 0) {
		ctx.save();
		ctx.globalAlpha = tas.maskAlpha;
		ctx.drawImage(fr.mask, v.ox, v.oy, tas.natW * v.scale, tas.natH * v.scale);
		// Draw it again to harden the edge: overlapping copies build up alpha
		// fastest where the mask is solid, so the boundary pops.
		ctx.globalAlpha = Math.min(1, tas.maskAlpha * 0.6);
		ctx.drawImage(fr.mask, v.ox, v.oy, tas.natW * v.scale, tas.natH * v.scale);
		ctx.restore();
	}

	// Existing segments: reference boxes, solid enough to actually see.
	ctx.lineWidth = 2;
	ctx.setLineDash([6, 4]);
	for (const s of fr.meta.segs) {
		const [x0, y0, x1, y1] = s.bbox;
		const [r, g, b] = colorForId(s.id);
		ctx.strokeStyle = `rgba(${r},${g},${b},0.95)`;
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
		// A propagated copy IS an ordinary box — same look, same handles, same
		// interactions — so nothing here distinguishes it from a hand-drawn one.
		ctx.strokeStyle = `rgb(${mr},${mg},${mb})`;
		ctx.strokeRect(cx0, cy0, cx1 - cx0, cy1 - cy0);
		ctx.fillStyle = `rgb(${mr},${mg},${mb})`;
		for (const [hx, hy] of [[cx0, cy0], [cx1, cy0], [cx0, cy1], [cx1, cy1]]) {
			ctx.fillRect(hx - HANDLE_VIS, hy - HANDLE_VIS, HANDLE_VIS * 2, HANDLE_VIS * 2);
		}
		const tag = `+${m.id - MANUAL_ID_BASE}`;
		ctx.fillRect(cx0, cy0 - 12, ctx.measureText(tag).width + 6, 12);
		ctx.fillStyle = "#000";
		ctx.fillText(tag, cx0 + 3, cy0 - 2);
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

// "8-16, 32-48" -> [8,9,...,16,32,...,48]. Tolerant on separators (commas,
// spaces, newlines), clamped to the clip, deduped and sorted. `exclude` drops
// the frame the box already lives on, so propagating never duplicates itself.
export function parseFrameRanges(spec, numFrames, exclude = -1) {
	const out = new Set();
	for (const tok of String(spec || "").replace(/[\n,]+/g, " ").split(/\s+/)) {
		if (!tok) continue;
		const m = tok.match(/^(\d+)\s*-\s*(\d+)$/);
		if (m) {
			let a = parseInt(m[1], 10), b = parseInt(m[2], 10);
			if (a > b) [a, b] = [b, a];                 // "16-8" reads the same
			for (let f = a; f <= b; f++) out.add(f);
		} else if (/^\d+$/.test(tok)) {
			out.add(parseInt(tok, 10));
		}
	}
	return [...out]
		.filter((f) => f >= 0 && f < numFrames && f !== exclude)
		.sort((a, b) => a - b);
}

const HANDLE_HIT = 10;   // canvas px to grab a corner/edge
const HANDLE_VIS = 4;    // half-size of a drawn handle
const MIN_BOX_SRC = 4;   // smallest box worth keeping, source px

const CURSOR = {
	nw: "nwse-resize", se: "nwse-resize", ne: "nesw-resize", sw: "nesw-resize",
	n: "ns-resize", s: "ns-resize", e: "ew-resize", w: "ew-resize", move: "move",
};

// Which box (and which grip of it) is under the cursor — the editing hit test.
// Topmost first, since later boxes draw over earlier ones. The inward reach of
// an edge is capped at a third of the box so a SMALL box always keeps a central
// zone for moving instead of being all resize-band.
function gripAt(node, cx, cy) {
	const tas = node._tas;
	const here = readManual(node).filter((m) => m.frame === tas.frameIdx);
	for (let i = here.length - 1; i >= 0; i--) {
		const m = here[i];
		const [x0, y0] = srcToCanvas(tas, m.bbox[0], m.bbox[1]);
		const [x1, y1] = srcToCanvas(tas, m.bbox[2], m.bbox[3]);
		const corners = { nw: [x0, y0], ne: [x1, y0], sw: [x0, y1], se: [x1, y1] };
		for (const [name, [px, py]] of Object.entries(corners)) {
			if (Math.hypot(cx - px, cy - py) <= HANDLE_HIT) return { id: m.id, grip: name };
		}
		const inMx = Math.min(HANDLE_HIT, (x1 - x0) / 3);
		const inMy = Math.min(HANDLE_HIT, (y1 - y0) / 3);
		const inYs = cy >= y0 - HANDLE_HIT && cy <= y1 + HANDLE_HIT;
		const inXs = cx >= x0 - HANDLE_HIT && cx <= x1 + HANDLE_HIT;
		if (cx >= x0 - HANDLE_HIT && cx <= x0 + inMx && inYs) return { id: m.id, grip: "w" };
		if (cx <= x1 + HANDLE_HIT && cx >= x1 - inMx && inYs) return { id: m.id, grip: "e" };
		if (cy >= y0 - HANDLE_HIT && cy <= y0 + inMy && inXs) return { id: m.id, grip: "n" };
		if (cy <= y1 + HANDLE_HIT && cy >= y1 - inMy && inXs) return { id: m.id, grip: "s" };
		if (cx > x0 && cx < x1 && cy > y0 && cy < y1) return { id: m.id, grip: "move" };
	}
	return null;
}

// Apply a move/resize drag, in SOURCE pixels so nothing drifts through repeated
// canvas<->source rounding, clamped to the frame.
function applyBoxDrag(node, sx, sy) {
	const tas = node._tas;
	const d = tas.drag;
	const arr = readManual(node);
	const m = arr.find((b) => b.id === d.id);
	if (!m) return;
	const W = tas.fullW, H = tas.fullH;
	let [x0, y0, x1, y1] = d.bbox;
	const dx = sx - d.sx, dy = sy - d.sy;

	if (d.grip === "move") {
		const w = x1 - x0, h = y1 - y0;
		const nx = clamp(x0 + dx, 0, W - w), ny = clamp(y0 + dy, 0, H - h);
		m.bbox = [Math.round(nx), Math.round(ny), Math.round(nx + w), Math.round(ny + h)];
	} else {
		if (d.grip.includes("w")) x0 = Math.min(clamp(x0 + dx, 0, W), x1 - MIN_BOX_SRC);
		if (d.grip.includes("e")) x1 = Math.max(clamp(x1 + dx, 0, W), x0 + MIN_BOX_SRC);
		if (d.grip.includes("n")) y0 = Math.min(clamp(y0 + dy, 0, H), y1 - MIN_BOX_SRC);
		if (d.grip.includes("s")) y1 = Math.max(clamp(y1 + dy, 0, H), y0 + MIN_BOX_SRC);
		m.bbox = [Math.round(x0), Math.round(y0), Math.round(x1), Math.round(y1)];
	}
	writeManual(node, arr);
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

// The DOM widget declares its OWN height: ComfyUI's DOMWidgetImpl reads
// options.getMinHeight in computeLayoutSize(), and that is what makes the node's
// size account for the canvas. Overriding node.computeSize instead either
// collapses the node on every move (default height ignores the canvas) or
// starves the widget of space — both of which we shipped by mistake.
function canvasHeightFor(node) {
	const t = node._tas;
	if (!t) return 260;
	const w = Math.max(node.size?.[0] || MIN_NODE_W, MIN_NODE_W) - 20;
	return Math.round(clamp(w * ((t.natH || 1) / (t.natW || 1)), 260, MAX_CANVAS_H));
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
	// Mask visibility: strong by default, and adjustable — you need it bold to
	// judge the mask, but faint to check the pixels underneath it.
	const mlabel = document.createElement("span");
	mlabel.textContent = "mask";
	mlabel.style.cssText = "font:11px monospace;color:#ff4fa3;white-space:nowrap;";
	const mask = document.createElement("input");
	mask.type = "range"; mask.min = "0"; mask.max = "100"; mask.value = "70";
	mask.title = "Mask overlay opacity (press M to toggle)";
	mask.style.cssText = "width:70px;flex:0 0 auto;accent-color:#ff0080;";
	const clearBtn = mkBtn("✕ clear");
	clearBtn.title = "Delete every drawn box, on ALL frames";
	clearBtn.style.color = "#ff9a9a";
	clearBtn.onclick = () => {
		const n = readManual(node).length;
		if (!n) return;
		if (!window.confirm(`Add Segments: delete all ${n} drawn box(es) on every frame?`)) return;
		writeManual(node, []);
		draw(node);
	};
	bar.append(prev, slider, next, counter, mlabel, mask, clearBtn);

	const canvas = document.createElement("canvas");
	canvas.style.cssText = "flex:1;min-height:0;width:100%;border-radius:4px;touch-action:none;display:block;cursor:crosshair;";
	wrap.append(bar, canvas);

	node._tas = {
		wrap, bar, canvas, slider, counter, prev, next, mask,
		maskAlpha: (node.properties?.ti_mask_alpha ?? 70) / 100,
		manifest: null, frames: [], frameIdx: 0,
		natW: 512, natH: 512, fullW: 512, fullH: 512,
		view: { ox: 0, oy: 0, scale: 1 }, creating: null, drag: null, propTarget: null,
	};

	node.addDOMWidget("add_seg_editor", "ti_add_editor", wrap, {
		serialize: false, hideOnZoom: false,
		getMinHeight: () => canvasHeightFor(node),
	});
	for (const name of ["manual_segments", "current_frame"]) {
		const w = getWidget(node, name);
		if (w) { w.type = "hidden"; w.computeSize = () => [0, -4]; }
	}

	// --- propagate modal ---------------------------------------------------
	// Lives inside the node (wrap is position:relative) rather than as a browser
	// dialog, so it appears where you are looking and cannot be lost behind the
	// graph. Ctrl/Cmd-click or double-click a drawn box to open it.
	const modal = document.createElement("div");
	modal.style.cssText = "position:absolute;left:50%;top:50%;transform:translate(-50%,-50%);"
		+ "z-index:10;display:none;flex-direction:column;gap:6px;background:#1e2229;"
		+ "border:1px solid #4aa3ff;border-radius:6px;padding:10px;min-width:260px;"
		+ "box-shadow:0 8px 28px rgba(0,0,0,.65);font:12px sans-serif;color:#e6e6e6;";
	const mTitle = document.createElement("div");
	mTitle.style.cssText = "font-weight:600;";
	const mInput = document.createElement("input");
	mInput.type = "text";
	mInput.placeholder = "8-16, 32-48";
	mInput.style.cssText = "background:#12151a;color:#e6e6e6;border:1px solid #3a4150;"
		+ "border-radius:4px;padding:4px 6px;font:12px monospace;width:100%;box-sizing:border-box;";
	const mHint = document.createElement("div");
	mHint.style.cssText = "font:11px monospace;color:#9fb3c8;min-height:14px;";
	const mRow = document.createElement("div");
	mRow.style.cssText = "display:flex;gap:6px;justify-content:flex-end;";
	const mCancel = mkBtn("Cancel");
	const mOk = mkBtn("Propagate");
	mOk.style.background = "#1f5c34"; mOk.style.color = "#d8ffe6"; mOk.style.borderColor = "#2f8a4e";
	mRow.append(mCancel, mOk);
	modal.append(mTitle, mInput, mHint, mRow);
	wrap.appendChild(modal);

	const closeProp = () => {
		modal.style.display = "none";
		node._tas.propTarget = null;
	};
	const hintProp = () => {
		const tas = node._tas;
		if (!tas.propTarget) return;
		const frames = parseFrameRanges(mInput.value, tas.manifest?.num_frames ?? 0,
									   tas.propTarget.frame);
		mHint.textContent = frames.length
			? `→ ${frames.length} frame(s): ${frames.slice(0, 12).join(", ")}${frames.length > 12 ? " …" : ""}`
			: "→ nothing yet (e.g. 8-16, 32-48)";
		mOk.disabled = !frames.length;
		mOk.style.opacity = frames.length ? "1" : "0.5";
	};
	const openProp = (box) => {
		const tas = node._tas;
		tas.propTarget = box;
		mTitle.textContent = `Propagate box #${box.id} (frame ${box.frame})`;
		mInput.value = "";
		modal.style.display = "flex";
		hintProp();
		setTimeout(() => mInput.focus({ preventScroll: true }), 0);
	};
	const doProp = () => {
		const tas = node._tas;
		const box = tas.propTarget;
		if (!box) return;
		const frames = parseFrameRanges(mInput.value, tas.manifest?.num_frames ?? 0, box.frame);
		if (!frames.length) return;
		const arr = readManual(node);
		let id = nextManualId(arr);
		for (const f of frames) {
			// Skip a frame that already carries this exact stamp, so propagating
			// twice over the same range does not pile up duplicates.
			if (arr.some((m) => m.frame === f && m.bbox.every((v, k) => v === box.bbox[k]))) continue;
			arr.push({ id: id++, frame: f, bbox: [...box.bbox], src: box.id });
		}
		writeManual(node, arr);
		updateCounter(node);
		closeProp();
		draw(node);
	};
	mOk.onclick = (e) => { e.stopPropagation(); doProp(); };
	mCancel.onclick = (e) => { e.stopPropagation(); closeProp(); };
	mInput.addEventListener("input", hintProp);
	// Keep typing (and Enter/Esc) inside the modal — the graph steals keys otherwise.
	modal.addEventListener("pointerdown", (e) => e.stopPropagation());
	modal.addEventListener("keydown", (e) => {
		e.stopPropagation();
		if (e.key === "Enter") { e.preventDefault(); doProp(); }
		else if (e.key === "Escape") { e.preventDefault(); closeProp(); }
	});

	new ResizeObserver(() => draw(node)).observe(wrap);

	// Navigation.
	prev.onclick = (e) => { e.stopPropagation(); loadFrame(node, node._tas.frameIdx - 1); };
	next.onclick = (e) => { e.stopPropagation(); loadFrame(node, node._tas.frameIdx + 1); };
	slider.addEventListener("input", (e) => { e.stopPropagation(); loadFrame(node, parseInt(slider.value, 10)); });
	mask.value = String(Math.round(node._tas.maskAlpha * 100));
	mask.addEventListener("input", (e) => {
		e.stopPropagation();
		const v = parseInt(mask.value, 10);
		node._tas.maskAlpha = v / 100;
		node.properties = node.properties || {};
		node.properties.ti_mask_alpha = v;      // UI state, never a backend input
		draw(node);
	});
	for (const el of [prev, next, slider]) el.addEventListener("pointerdown", (e) => e.stopPropagation());

	wrap.tabIndex = 0;
	wrap.addEventListener("pointerenter", () => wrap.focus({ preventScroll: true }));
	wrap.addEventListener("keydown", (e) => {
		if (e.key === "m" || e.key === "M") {
			const t = node._tas;
			t.maskAlpha = t.maskAlpha > 0 ? 0 : ((node.properties?.ti_mask_alpha ?? 70) / 100 || 0.7);
			t.mask.value = String(Math.round(t.maskAlpha * 100));
			draw(node); e.stopPropagation(); e.preventDefault(); return;
		}
		if (e.key === "ArrowLeft") { loadFrame(node, node._tas.frameIdx - 1); e.stopPropagation(); e.preventDefault(); }
		else if (e.key === "ArrowRight") { loadFrame(node, node._tas.frameIdx + 1); e.stopPropagation(); e.preventDefault(); }
	});

	const pos = (e) => pointerPos(canvas, e);

	// Left-drag = create a box.
	canvas.addEventListener("pointerdown", (e) => {
		if (!node._tas.manifest || e.button !== 0) return;
		releaseGraphPointer(e);
		const [cx, cy] = pos(e);
		// Ctrl/Cmd-click a drawn box = propagate it, not start a new box.
		if (e.ctrlKey || e.metaKey) {
			const hit = manualAt(node, cx, cy);
			if (hit) openProp(hit);
			e.stopPropagation(); e.preventDefault();
			return;
		}
		// Grab an existing box (move or resize) before starting a new one, so the
		// boxes stay editable instead of write-once.
		const grip = gripAt(node, cx, cy);
		if (grip) {
			const m = readManual(node).find((b) => b.id === grip.id);
			const [gsx, gsy] = canvasToSrc(node._tas, cx, cy);
			node._tas.drag = { id: grip.id, grip: grip.grip, sx: gsx, sy: gsy, bbox: [...m.bbox] };
			canvas.setPointerCapture?.(e.pointerId);
			e.stopPropagation(); e.preventDefault();
			return;
		}
		node._tas.creating = { x0: cx, y0: cy, x1: cx, y1: cy };
		canvas.setPointerCapture?.(e.pointerId);
		e.stopPropagation(); e.preventDefault();
	});

	// Double-click does the same — a modifier-free path, since the graph claims
	// some modifier+click combinations before they reach this canvas.
	canvas.addEventListener("dblclick", (e) => {
		if (!node._tas.manifest) return;
		e.stopPropagation(); e.preventDefault();
		const [cx, cy] = pos(e);
		const hit = manualAt(node, cx, cy);
		if (hit) openProp(hit);
	});
	canvas.addEventListener("pointermove", (e) => {
		const tas = node._tas;
		const [cx, cy] = pos(e);
		if (tas.drag) {
			const [sx, sy] = canvasToSrc(tas, cx, cy);
			applyBoxDrag(node, sx, sy);
			draw(node);
			e.stopPropagation();
			return;
		}
		if (tas.creating) {
			tas.creating.x1 = cx; tas.creating.y1 = cy;
			draw(node);
			e.stopPropagation();
			return;
		}
		if (!tas.manifest) return;
		// Hover feedback: show what this spot would grab.
		const grip = gripAt(node, cx, cy);
		canvas.style.cursor = grip ? CURSOR[grip.grip] : "crosshair";
	});
	canvas.addEventListener("pointerup", (e) => {
		if (node._tas.drag) {                     // finish a move / resize
			node._tas.drag = null;
			canvas.releasePointerCapture?.(e.pointerId);
			updateCounter(node);
			draw(node);
			return;
		}
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
		// macOS turns ctrl-click into a context menu; that gesture means
		// "propagate" here, so it must not delete the box instead.
		if (e.ctrlKey || e.metaKey) return;
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
	const width = Math.max(node.size[0], PREFERRED_W);
	const canvasH = clamp((width - 20) * (tas.natH / tas.natW), 260, MAX_CANVAS_H);
	node.setSize([width, Math.round(barH + canvasH + 20)]);
	node.setDirtyCanvas?.(true, true);
	requestAnimationFrame(() => draw(node));
}

function applyManifest(node, manifest) {
	const tas = node._tas;
	// Boxes PERSIST across input changes (they're cleared only by the ✕ clear
	// button) — a new clip just restamps them with its signature. Boxes whose
	// frame is beyond the new clip are simply ignored by the backend.
	const prev = readManualObj(node);
	const isNewInput = prev.sig !== manifest.sig;

	tas.manifest = manifest;
	tas.frames = [];
	tas.natW = manifest.pw || tas.natW;
	tas.natH = manifest.ph || tas.natH;
	tas.fullW = manifest.full_w || tas.natW;
	tas.fullH = manifest.full_h || tas.natH;
	tas.slider.max = String(Math.max(0, manifest.num_frames - 1));

	if (isNewInput && prev.items.length) {
		writeManual(node, prev.items);               // keep, restamped with the new sig
		console.info(`[tinode] Add Segments: new input — kept ${prev.items.length} drawn box(es).`);
	}

	fitNodeToAspect(node);
	const start = clamp(getWidget(node, "current_frame")?.value ?? 0, 0, manifest.num_frames - 1);
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


		// Keep the node's aspect locked to the frame while the user resizes, so
		// the image always fills the body instead of sitting in dead space.
		const onResize = nodeType.prototype.onResize;
		nodeType.prototype.onResize = function (size) {
			onResize?.apply(this, arguments);
			const tas = this._tas;
			if (!tas || !tas.manifest || !tas.natW || !tas.natH) return;
			if (!Array.isArray(size)) return;
			const barH = tas.bar.offsetHeight || 32;
			const canvasH = clamp((size[0] - 20) * (tas.natH / tas.natW), 260, MAX_CANVAS_H);
			size[1] = Math.round(barH + canvasH + 20);
			requestAnimationFrame(() => draw(this));
		};
		const onConfigure = nodeType.prototype.onConfigure;
		nodeType.prototype.onConfigure = function () {
			const r = onConfigure?.apply(this, arguments);
			if (this._tas) requestAnimationFrame(() => draw(this));
			return r;
		};
	},
});
