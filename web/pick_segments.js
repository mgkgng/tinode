// Interactive segment picker for the "Pick Segments (ti)" node.
//
// After the graph runs, the Python node ships a manifest (per-frame source +
// "label map" images and segment boxes) via ui.ti_pick. Here you scrub frames,
// see every segment as a colorized box, hover to light up its real mask shape,
// and click the object itself to include/exclude it. The choice is per-id and
// global (persisted in the hidden excluded_ids widget), so it holds on every
// frame. Re-queue to apply it to the mask/image/segments outputs.
//
// Picking is pixel-accurate: the label map encodes which segment owns each
// pixel (value = R + G*256, 0 = background), so a click resolves to the exact
// object under the cursor even where 90 boxes overlap. Background clicks fall
// back to the smallest box containing the point.

import { app } from "../../scripts/app.js";
import {
	colorForId, clamp, fillNodeWidth, getWidget, urlFor, pointerPos, releaseGraphPointer,
} from "./lib/editor.js";

const NODE_TYPES = new Set(["TI_PickSegments", "TI_DeleteSegments", "TI_EraseSegments"]);
// Edit Segments (node id still TI_EraseSegments) removes per-frame instances
// like Delete Segments, but opens in brush mode.
const DELETE_TYPES = new Set(["TI_DeleteSegments", "TI_EraseSegments"]);
const MIN_NODE_W = 360;
// Width the editor grows to on load — a segment picker is unusable small.
// Only ever grows: a node you widened yourself keeps its size.
const PREFERRED_W = 720;
const MAX_CANVAS_H = 2000;
const MIN_NODE_H = 320;

// Stored as {sig, ids} — sig identifies the image+segments the selection was
// made against, so a new input drops it instead of silently re-excluding ids
// that mean a different object in the new clip.
function readExcludedObj(node) {
	try {
		const v = JSON.parse(getWidget(node, "excluded_ids")?.value || "[]");
		if (Array.isArray(v)) return { sig: null, ids: v };      // legacy bare list
		if (!v || typeof v !== "object") return { sig: null, ids: [] };
		return { sig: v.sig ?? null, ids: Array.isArray(v.ids) ? v.ids : [] };
	} catch { return { sig: null, ids: [] }; }
}

function readExcluded(node) { return new Set(readExcludedObj(node).ids); }

function writeExcluded(node, set) {
	const w = getWidget(node, "excluded_ids");
	if (!w) return;
	const sig = node._tps?.manifest?.sig ?? readExcludedObj(node).sig ?? null;
	w.value = JSON.stringify({ sig, ids: [...set] });
	w.callback?.(w.value, app.canvas, node);
}

function readDeletedObj(node) {
	try {
		const v = JSON.parse(getWidget(node, "deleted_items")?.value || "{}");
		if (!v || typeof v !== "object") return { sig: null, items: [] };
		return { sig: v.sig ?? null, items: Array.isArray(v.items) ? v.items : [] };
	} catch { return { sig: null, items: [] }; }
}

// Per-object grow, stored as {sig, grow:{id: px}} — the backend dilates (or
// erodes, when negative) each of these segments by that many pixels.
function readGrowObj(node) {
	try {
		const v = JSON.parse(getWidget(node, "grow_ids")?.value || "{}");
		if (!v || typeof v !== "object") return { sig: null, grow: {} };
		return { sig: v.sig ?? null, grow: (v.grow && typeof v.grow === "object") ? v.grow : {} };
	} catch { return { sig: null, grow: {} }; }
}

function writeGrow(node, grow) {
	const w = getWidget(node, "grow_ids");
	if (!w) return;
	for (const k of Object.keys(grow)) if (!grow[k]) delete grow[k];   // 0 = nothing to store
	const sig = node._tps?.manifest?.sig ?? readGrowObj(node).sig ?? null;
	w.value = JSON.stringify({ sig, grow });
	w.callback?.(w.value, app.canvas, node);
}

// Painted strokes, stored as {sig, strokes:[{id, frame, r, pts:[[x,y]...]}]} in
// SOURCE pixels — the backend stamps discs along each polyline.
function readPaintObj(node) {
	try {
		const v = JSON.parse(getWidget(node, "painted")?.value || "{}");
		if (!v || typeof v !== "object") return { sig: null, strokes: [] };
		return { sig: v.sig ?? null, strokes: Array.isArray(v.strokes) ? v.strokes : [] };
	} catch { return { sig: null, strokes: [] }; }
}

function writePaint(node, strokes) {
	const w = getWidget(node, "painted");
	if (!w) return;
	const sig = node._tps?.manifest?.sig ?? readPaintObj(node).sig ?? null;
	w.value = JSON.stringify({ sig, strokes });
	w.callback?.(w.value, app.canvas, node);
}

// Canvas point -> SOURCE pixels (the manifest's preview is downscaled, so the
// stroke must be stored in full-resolution coordinates like every other bbox).
function canvasToSource(tps, cx, cy) {
	const v = tps.view;
	const sx = (cx - v.ox) / v.scale;
	const sy = (cy - v.oy) / v.scale;
	const kx = (tps.manifest?.full_w || tps.natW) / tps.natW;
	const ky = (tps.manifest?.full_h || tps.natH) / tps.natH;
	return [Math.round(sx * kx), Math.round(sy * ky)];
}

function sourceToCanvas(tps, sx, sy) {
	const v = tps.view;
	const kx = (tps.manifest?.full_w || tps.natW) / tps.natW;
	const ky = (tps.manifest?.full_h || tps.natH) / tps.natH;
	return [v.ox + (sx / kx) * v.scale, v.oy + (sy / ky) * v.scale];
}

function nextPaintId(node) {
	const base = 2000000;
	const ids = readPaintObj(node).strokes.map((st) => st.id || 0);
	return Math.max(base - 1, ...ids) + 1;
}

function readSelection(node) {
	if (!node._tps?.deleteMode) return readExcluded(node);
	return new Set(readDeletedObj(node).items.map((v) => `${v.frame}:${v.index}`));
}

function writeSelection(node, set) {
	if (!node._tps?.deleteMode) {
		writeExcluded(node, set);
		return;
	}
	const w = getWidget(node, "deleted_items");
	if (!w) return;
	const sig = node._tps?.manifest?.sig ?? readDeletedObj(node).sig ?? null;
	const items = [...set].map((key) => {
		const [frame, index] = key.split(":").map(Number);
		return { frame, index };
	});
	w.value = JSON.stringify({ sig, items });
	w.callback?.(w.value, app.canvas, node);
}

function setFrameWidget(node, f) {
	const w = getWidget(node, "current_frame");
	if (w && w.value !== f) { w.value = f; w.callback?.(f, app.canvas, node); }
}

// Fit the frame (natW x natH) into the canvas, letterboxed.
function viewRect(tps) {
	const cw = tps.canvas.width, ch = tps.canvas.height;
	const nw = tps.natW, nh = tps.natH;
	const scale = Math.min(cw / nw, ch / nh);
	return { ox: (cw - nw * scale) / 2, oy: (ch - nh * scale) / 2, scale };
}

// Label id (segment array index +1) at a canvas point, or 0 for background.
function labelAt(tps, cx, cy) {
	const fr = tps.frames[tps.frameIdx];
	if (!fr || !fr.labelData) return 0;
	const v = tps.view;
	const ix = Math.floor((cx - v.ox) / v.scale);
	const iy = Math.floor((cy - v.oy) / v.scale);
	if (ix < 0 || iy < 0 || ix >= tps.natW || iy >= tps.natH) return 0;
	const d = fr.labelData.data;
	const p = (iy * tps.natW + ix) * 4;
	return d[p] + d[p + 1] * 256;
}

// Offscreen tint of every pixel owned by segment index `segIdx` (0-based).
function buildHighlight(tps, segIdx) {
	const fr = tps.frames[tps.frameIdx];
	if (!fr || !fr.labelData) return null;
	if (fr._hlIdx === segIdx && fr._hl) return fr._hl;
	const seg = fr.meta.segs[segIdx];
	if (!seg) return null;
	const off = document.createElement("canvas");
	off.width = tps.natW; off.height = tps.natH;
	const octx = off.getContext("2d");
	const out = octx.createImageData(tps.natW, tps.natH);
	const src = fr.labelData.data, dst = out.data;
	const want = segIdx + 1;
	const [r, g, b] = colorForId(seg.id);
	for (let p = 0; p < tps.natW * tps.natH; p++) {
		if (src[p * 4] + src[p * 4 + 1] * 256 === want) {
			dst[p * 4] = r; dst[p * 4 + 1] = g; dst[p * 4 + 2] = b; dst[p * 4 + 3] = 150;
		}
	}
	octx.putImageData(out, 0, 0);
	fr._hl = off; fr._hlIdx = segIdx;
	return off;
}

// Offscreen tint of EVERY segment's real mask on this frame, coloured per id.
// Editing a shape by hand against a rectangle is guesswork — the box says
// nothing about where the object's edge actually is — so this paints the label
// map itself. Cached per (frame, selection): the palette's alpha is what turns a
// swept-off segment transparent, so the cache only rebuilds when that changes.
function buildMaskOverlay(node) {
	const tps = node._tps;
	const fr = tps.frames[tps.frameIdx];
	if (!fr || !fr.labelData || !fr.meta) return null;
	const excluded = readSelection(node);
	const segs = fr.meta.segs;
	const keyOf = (i) => (tps.deleteMode ? `${tps.frameIdx}:${i}` : segs[i].id);
	const offList = segs.map((_s, i) => (excluded.has(keyOf(i)) ? "1" : "0")).join("");
	const key = `${segs.length}|${offList}`;
	if (fr._allKey === key && fr._all) return fr._all;

	// Palette indexed by label value (segment index + 1); 0 stays background.
	const pal = new Uint8Array((segs.length + 1) * 4);
	for (let i = 0; i < segs.length; i++) {
		const [r, g, b] = colorForId(segs[i].id);
		const p = (i + 1) * 4;
		pal[p] = r; pal[p + 1] = g; pal[p + 2] = b;
		pal[p + 3] = excluded.has(keyOf(i)) ? 0 : 1;      // 1 = on, 0 = swept off
	}
	const W = tps.natW, H = tps.natH;
	const off = document.createElement("canvas");
	off.width = W; off.height = H;
	const octx = off.getContext("2d");
	const out = octx.createImageData(W, H);
	const src = fr.labelData.data, dst = out.data;
	// Two alphas: a light wash over the body so the footage still shows through,
	// and a near-opaque line on the boundary. The edge is the thing you aim at
	// when trimming a mask, and a flat wash hides exactly that.
	const FILL_A = 90, EDGE_A = 235;
	for (let y = 0; y < H; y++) {
		for (let x = 0; x < W; x++) {
			const p = y * W + x;
			const v = src[p * 4] + src[p * 4 + 1] * 256;
			if (v === 0 || v > segs.length) continue;
			const q = v * 4;
			if (!pal[q + 3]) continue;
			// Boundary = any 4-neighbour that is not this segment (frame edge counts).
			const edge =
				x === 0 || y === 0 || x === W - 1 || y === H - 1 ||
				src[(p - 1) * 4] + src[(p - 1) * 4 + 1] * 256 !== v ||
				src[(p + 1) * 4] + src[(p + 1) * 4 + 1] * 256 !== v ||
				src[(p - W) * 4] + src[(p - W) * 4 + 1] * 256 !== v ||
				src[(p + W) * 4] + src[(p + W) * 4 + 1] * 256 !== v;
			dst[p * 4] = pal[q]; dst[p * 4 + 1] = pal[q + 1];
			dst[p * 4 + 2] = pal[q + 2]; dst[p * 4 + 3] = edge ? EDGE_A : FILL_A;
		}
	}
	octx.putImageData(out, 0, 0);
	fr._all = off; fr._allKey = key;
	return off;
}

function loadFrame(node, f) {
	const tps = node._tps;
	if (!tps || !tps.manifest) return;
	f = clamp(f, 0, tps.manifest.num_frames - 1);
	tps.frameIdx = f;
	setFrameWidget(node, f);
	if (tps.slider) tps.slider.value = String(f);
	if (tps.counter) updateCounter(node);

	let fr = tps.frames[f];
	if (!fr) {
		const meta = tps.manifest.frames[f];
		fr = tps.frames[f] = { meta, src: null, labelData: null };
		const src = new Image();
		src.onload = () => { fr.src = src; if (tps.frameIdx === f) draw(node); };
		src.onerror = () => console.error("[tinode] Pick Segments: src load failed", f);
		src.src = urlFor(meta.src);

		const lbl = new Image();
		lbl.onload = () => {
			const off = document.createElement("canvas");
			off.width = lbl.naturalWidth; off.height = lbl.naturalHeight;
			const octx = off.getContext("2d", { willReadFrequently: true });
			octx.drawImage(lbl, 0, 0);
			fr.labelData = octx.getImageData(0, 0, off.width, off.height);
			tps.natW = off.width; tps.natH = off.height;
			if (tps.frameIdx === f) draw(node);
		};
		lbl.src = urlFor(meta.label);
	}
	draw(node);
}

function draw(node) {
	const tps = node._tps;
	if (!tps) return;
	const cv = tps.canvas, ctx = cv.getContext("2d");
	fillNodeWidth(node, tps.container || tps.wrap);
	const w = Math.max(1, Math.floor(cv.clientWidth));
	const h = Math.max(1, Math.floor(cv.clientHeight));
	if (cv.width !== w || cv.height !== h) { cv.width = w; cv.height = h; }

	ctx.clearRect(0, 0, cv.width, cv.height);
	ctx.fillStyle = "#141414";
	ctx.fillRect(0, 0, cv.width, cv.height);

	if (!tps.manifest) {
		ctx.fillStyle = "#888"; ctx.font = "12px sans-serif"; ctx.textAlign = "center";
		ctx.fillText(tps.deleteMode
			? "Run once, then click a segment to delete it (shift-drag to sweep)."
			: "Run once to load the segments, then click to toggle (shift-drag to sweep).",
			cv.width / 2, cv.height / 2);
		return;
	}

	tps.view = viewRect(tps);
	const v = tps.view;
	const fr = tps.frames[tps.frameIdx];
	if (fr && fr.src) ctx.drawImage(fr.src, v.ox, v.oy, tps.natW * v.scale, tps.natH * v.scale);

	if (!fr || !fr.meta) return;
	const excluded = readSelection(node);

	// Every segment's real shape, under the hover highlight and the boxes.
	if (tps.showMasks) {
		const all = buildMaskOverlay(node);
		if (all) ctx.drawImage(all, v.ox, v.oy, tps.natW * v.scale, tps.natH * v.scale);
	}

	// Hovered segment: paint its real mask shape.
	if (tps.hoverIdx != null && tps.hoverIdx >= 0) {
		const hl = buildHighlight(tps, tps.hoverIdx);
		if (hl) ctx.drawImage(hl, v.ox, v.oy, tps.natW * v.scale, tps.natH * v.scale);
	}

	// Boxes: included solid in their color, excluded dashed/dim.
	const growMap = readGrowObj(node).grow;
	ctx.lineWidth = 2;
	ctx.font = "10px monospace";
	ctx.textAlign = "left";
	for (let i = 0; i < fr.meta.segs.length; i++) {
		const s = fr.meta.segs[i];
		const [x0, y0, x1, y1] = s.bbox;
		const bx = v.ox + x0 * v.scale, by = v.oy + y0 * v.scale;
		const bw = (x1 - x0) * v.scale, bh = (y1 - y0) * v.scale;
		const [r, g, b] = colorForId(s.id);
		const selectionKey = tps.deleteMode ? `${tps.frameIdx}:${i}` : s.id;
		const off = excluded.has(selectionKey);
		ctx.setLineDash(off ? [4, 3] : []);
		ctx.strokeStyle = off ? `rgba(${r},${g},${b},0.35)` : `rgb(${r},${g},${b})`;
		ctx.strokeRect(bx, by, bw, bh);
		// Selected (ctrl-clicked) segments get a bright outline + their grow
		// amount, so you can see what + / − is about to act on.
		if (tps.selected?.has(s.id)) {
			ctx.save();
			ctx.setLineDash([]);
			ctx.strokeStyle = "#ffd479";
			ctx.lineWidth = 3;
			ctx.strokeRect(bx - 2, by - 2, bw + 4, bh + 4);
			const amt = growMap[s.id] || 0;
			if (amt) {
				const t = `${amt > 0 ? "+" : ""}${amt}px`;
				ctx.font = "10px monospace";
				const tw = ctx.measureText(t).width;
				ctx.fillStyle = "#ffd479";
				ctx.fillRect(bx + bw - tw - 6, by + bh, tw + 6, 12);
				ctx.fillStyle = "#000";
				ctx.fillText(t, bx + bw - tw - 3, by + bh + 10);
			}
			ctx.restore();
		}
		if (i === tps.hoverIdx || !off) {
			const label = `#${s.id}`;
			ctx.fillStyle = off ? "rgba(0,0,0,0.5)" : `rgb(${r},${g},${b})`;
			const tw = ctx.measureText(label).width;
			ctx.fillRect(bx, by - 12, tw + 6, 12);
			ctx.fillStyle = off ? "#aaa" : "#000";
			ctx.fillText(label, bx + 3, by - 2);
		}
	}
	ctx.setLineDash([]);

	// Brush strokes on this frame + the one in progress, as real brush shapes.
	// Draw strokes wear their segment colour; erase strokes are drawn as a hole —
	// a dark core with a pale halo, so they read as "taken away" over any image
	// and are never mistaken for something painted in.
	const strokes = readPaintObj(node).strokes.filter((st) => st.frame === tps.frameIdx);
	const live = tps.stroke ? [tps.stroke] : [];
	const kx = (tps.manifest?.full_w || tps.natW) / tps.natW;
	for (const st of strokes.concat(live)) {
		const cut = st.mode === "erase";
		ctx.lineCap = "round";
		ctx.lineJoin = "round";
		// The stored radius is in source px; scale it to the canvas.
		const lw = Math.max(1, (st.r * 2 / kx) * v.scale);
		const path = () => {
			ctx.beginPath();
			st.pts.forEach(([sx, sy], i) => {
				const [px, py] = sourceToCanvas(tps, sx, sy);
				if (i === 0) ctx.moveTo(px, py); else ctx.lineTo(px, py);
			});
			if (st.pts.length === 1) {          // a dab, not a drag
				const [px, py] = sourceToCanvas(tps, st.pts[0][0], st.pts[0][1]);
				ctx.lineTo(px + 0.01, py);
			}
			ctx.stroke();
		};
		if (cut) {
			ctx.strokeStyle = "rgba(255,255,255,0.45)";
			ctx.lineWidth = lw + 2;
			path();
			ctx.strokeStyle = "rgba(10,10,12,0.85)";
			ctx.lineWidth = lw;
			path();
		} else {
			const [r, g, b] = colorForId(st.id || 2000000);
			ctx.strokeStyle = `rgba(${r},${g},${b},0.85)`;
			ctx.lineWidth = lw;
			path();
		}
	}

	// Brush-size cursor: you cannot judge a brush you cannot see. Sweep has no
	// radius (it takes whole segments), so it gets a fixed small ring.
	if (tps.cursor && (tps.drawMode || tps.cutMode || tps.eraseMode)) {
		const sized = tps.drawMode || tps.cutMode;
		const rr = sized ? Math.max(2, (tps.brushR / kx) * v.scale) : 6;
		ctx.beginPath();
		ctx.arc(tps.cursor[0], tps.cursor[1], rr, 0, Math.PI * 2);
		ctx.strokeStyle = tps.drawMode ? "#4aa3ff" : (tps.cutMode ? "#e0a33a" : "#ff6b81");
		ctx.lineWidth = 1.5;
		ctx.stroke();
	}
}

function updateCounter(node) {
	const tps = node._tps;
	const excluded = readSelection(node);
	if (tps.deleteMode) {
		const total = tps.manifest.frames.reduce((n, f) => n + f.segs.length, 0);
		const here = tps.manifest.frames[tps.frameIdx]?.segs.length || 0;
		const deletedHere = [...excluded].filter(
			(key) => key.startsWith(`${tps.frameIdx}:`)
		).length;
		tps.counter.textContent =
			`frame ${tps.frameIdx + 1}/${tps.manifest.num_frames}   ·   ` +
			`${deletedHere}/${here} deleted here · ${excluded.size}/${total} total`;
	} else {
		const total = tps.manifest.ids.length;
		const on = tps.manifest.ids.filter((i) => !excluded.has(i)).length;
		tps.counter.textContent =
			`frame ${tps.frameIdx + 1}/${tps.manifest.num_frames}   ·   ${on}/${total} on`;
	}
	// Every path that changes an edit lands here, so this is where the clear
	// button learns whether there is anything left to clear.
	tps.refreshClear?.();
}

// The DOM widget declares its OWN height: ComfyUI's DOMWidgetImpl reads
// options.getMinHeight in computeLayoutSize(), and that is what makes the node's
// size account for the canvas. Overriding node.computeSize instead either
// collapses the node on every move (default height ignores the canvas) or
// starves the widget of space — both of which we shipped by mistake.
function canvasHeightFor(node) {
	const t = node._tps;
	if (!t) return 260;
	const w = Math.max(node.size?.[0] || MIN_NODE_W, MIN_NODE_W) - 20;
	return Math.round(clamp(w * ((t.natH || 1) / (t.natW || 1)), 260, MAX_CANVAS_H));
}

// --- undo / redo -----------------------------------------------------------
// A step is a SNAPSHOT of the editor-driven widgets, not a diff: they are small
// JSON strings, and restoring one is exact by construction. One entry per
// gesture (a whole brush stroke, a click, a grow press), never per pointermove.
function snapshot(node) {
	return {
		sel: getWidget(node, node._tps?.deleteMode ? "deleted_items" : "excluded_ids")?.value ?? "",
		grow: getWidget(node, "grow_ids")?.value ?? "",
		paint: getWidget(node, "painted")?.value ?? "",
	};
}

function restore(node, snap) {
	const selName = node._tps?.deleteMode ? "deleted_items" : "excluded_ids";
	for (const [name, value] of [[selName, snap.sel], ["grow_ids", snap.grow],
								 ["painted", snap.paint]]) {
		const w = getWidget(node, name);
		if (w && value !== "" && w.value !== value) {
			w.value = value;
			w.callback?.(value, app.canvas, node);
		}
	}
}

function pushHistory(node) {
	const tps = node._tps;
	if (!tps) return;
	tps.history.push(snapshot(node));
	if (tps.history.length > 100) tps.history.shift();   // bound the stack
	tps.future.length = 0;                                // a new act forks the timeline
	tps.refreshHistory?.();
}

function undo(node) {
	const tps = node._tps;
	if (!tps || !tps.history.length) return;
	tps.future.push(snapshot(node));
	restore(node, tps.history.pop());
	updateCounter(node);
	tps.refreshHistory?.();
	draw(node);
}

function redo(node) {
	const tps = node._tps;
	if (!tps || !tps.future.length) return;
	tps.history.push(snapshot(node));
	restore(node, tps.future.pop());
	updateCounter(node);
	tps.refreshHistory?.();
	draw(node);
}

// How much hand editing this node is currently carrying, per kind. Drives the
// clear button's label and lets it stay inert when there is nothing to lose.
function editCounts(node) {
	const sel = readSelection(node).size;
	const strokes = readPaintObj(node).strokes;
	const grow = Object.values(readGrowObj(node).grow).filter(Boolean).length;
	const drawn = strokes.filter((s) => s.mode !== "erase").length;
	const erased = strokes.length - drawn;
	return { sel, drawn, erased, grow, total: sel + strokes.length + grow };
}

function setup(node, deleteMode = false, brushFirst = false) {
	if (node._tps) return;
	const wrap = document.createElement("div");
	wrap.style.cssText = "position:relative;width:100%;height:100%;display:flex;flex-direction:column;box-sizing:border-box;";

	// Two rows: the frame scrubber gets a line to itself so its handle can travel
	// the full width of the node, and the tools get another. Sharing one line
	// squeezed the slider to a stub between the buttons, and half the clip was
	// unreachable by dragging.
	const bar = document.createElement("div");
	bar.style.cssText = "display:flex;flex-direction:column;gap:3px;padding:4px;flex:0 0 auto;";
	const navRow = document.createElement("div");
	navRow.style.cssText = "display:flex;align-items:center;gap:6px;width:100%;box-sizing:border-box;";
	const toolRow = document.createElement("div");
	toolRow.style.cssText = "display:flex;align-items:center;gap:6px;width:100%;flex-wrap:wrap;box-sizing:border-box;";
	bar.append(navRow, toolRow);
	const mkBtn = (txt) => {
		const b = document.createElement("button");
		b.textContent = txt;
		b.style.cssText = "background:#2a2f3a;color:#cfe3ff;border:1px solid #3a4150;border-radius:4px;font:600 12px sans-serif;padding:2px 8px;cursor:pointer;";
		return b;
	};
	const prev = mkBtn("◀");
	const next = mkBtn("▶");
	const slider = document.createElement("input");
	slider.type = "range"; slider.min = "0"; slider.max = "0"; slider.value = "0";
	slider.style.cssText = "flex:1;min-width:0;";
	const counter = document.createElement("span");
	counter.style.cssText = "font:11px monospace;color:#cfd3da;white-space:nowrap;";
	// Sweep MODE, not just the shift modifier: ComfyUI's canvas claims shift-drag
	// for its own multi-select, so the modifier alone never reaches us reliably.
	// Toggle this on and a plain drag takes whole segments off.
	const eraser = mkBtn("⌫ sweep");
	eraser.title = "Sweep: drag across segments to switch them off whole (Shift also works)";
	// Grow/shrink the ctrl-clicked selection. Only Pick Segments carries the
	// grow widget; Delete Segments shares this editor but not that output.
	const growBtn = mkBtn("+");
	const shrinkBtn = mkBtn("−");
	growBtn.title = "Grow the selected segments' masks by 2px (ctrl-click to select)";
	shrinkBtn.title = "Shrink the selected segments' masks by 2px";
	const selInfo = document.createElement("span");
	selInfo.style.cssText = "font:11px monospace;color:#ffd479;white-space:nowrap;";
	const undoBtn = mkBtn("↶");
	const redoBtn = mkBtn("↷");
	undoBtn.title = "Undo (Ctrl-Z)";
	redoBtn.title = "Redo (Ctrl-Shift-Z)";
	// Draw + erase brushes and their shared size, for the node that can paint.
	const drawBtn = mkBtn("✏ draw");
	drawBtn.title = "Draw: paint a new segment freehand";
	// The pixel eraser. Unlike sweep, this bites into whatever mask is under it —
	// SAM3's own segments included — so a nearly-right detection can be trimmed
	// instead of thrown away.
	const cutBtn = mkBtn("◍ erase");
	cutBtn.title = "Erase: rub mask pixels away from any segment, drawn or detected";
	const brush = document.createElement("input");
	brush.type = "range"; brush.min = "1"; brush.max = "128"; brush.value = "16";
	brush.title = "Brush size";
	brush.style.cssText = "width:80px;flex:0 0 auto;accent-color:#4aa3ff;";
	// Mask shapes, not just boxes — you cannot erase accurately against a
	// rectangle that says nothing about where the object's edge actually is.
	const maskBtn = mkBtn("◨ masks");
	maskBtn.title = "Show every segment's real mask shape, not only its box";
	// Start over: drop every hand edit on this clip in one step.
	const clearBtn = mkBtn("✕ clear");
	navRow.append(prev, slider, next, counter);
	toolRow.append(eraser, drawBtn, cutBtn, brush, maskBtn,
				   shrinkBtn, growBtn, selInfo, undoBtn, redoBtn, clearBtn);

	const canvas = document.createElement("canvas");
	canvas.style.cssText = "flex:1;min-height:0;width:100%;border-radius:4px;touch-action:none;display:block;";
	wrap.append(bar, canvas);

	node._tps = {
		wrap, bar, navRow, toolRow, canvas, slider, counter, prev, next, eraser,
		manifest: null, frames: [], frameIdx: 0,
		natW: 512, natH: 512, view: { ox: 0, oy: 0, scale: 1 }, hoverIdx: null,
		deleteMode, erasing: false, eraseMode: brushFirst, lastErase: null,
		selected: new Set(),   // ctrl-clicked ids, the target of + / −
		history: [], future: [],
		drawMode: false, cutMode: false, stroke: null, brushR: 16, cursor: null,
		// ON by default, everywhere: the mask is what the node actually outputs
		// and the box is only its bounding rectangle, so the shape has to be
		// visible standing still — not just for whatever the cursor happens to
		// be over. The ◨ button turns it off when a box is what you want to see.
		showMasks: true,
	};

	// --- brushes -----------------------------------------------------------
	const hasPaint = !!getWidget(node, "painted");
	if (!hasPaint) {
		drawBtn.style.display = "none";
		cutBtn.style.display = "none";
		brush.style.display = "none";
	}
	const brushW = getWidget(node, "brush_size");
	if (brushW) brush.value = String(brushW.value ?? 16);
	node._tps.brushR = parseInt(brush.value, 10) || 16;

	// One accent per tool, so the active brush is obvious at a glance:
	// sweep = red, draw = blue, erase = amber, masks = green.
	const ACCENT = new Map([
		[eraser, ["#7a2230", "#c04a5e"]],
		[drawBtn, ["#1f4d7a", "#4aa3ff"]],
		[cutBtn, ["#7a5a1f", "#e0a33a"]],
		[maskBtn, ["#1f5a3a", "#3fbf82"]],
	]);
	const refreshBrushes = () => {
		const tps = node._tps;
		for (const [b, on] of [[eraser, tps.eraseMode], [drawBtn, tps.drawMode],
							   [cutBtn, tps.cutMode], [maskBtn, tps.showMasks]]) {
			const [bg, border] = ACCENT.get(b);
			b.style.background = on ? bg : "#2a2f3a";
			b.style.color = on ? "#fff" : "#cfe3ff";
			b.style.borderColor = on ? border : "#3a4150";
		}
		canvas.style.cursor = tps.eraseMode ? "cell"
			: ((tps.drawMode || tps.cutMode) ? "none" : "default");
	};
	// The three brushes are mutually exclusive — a drag has to mean one thing.
	const setMode = (which) => {
		const tps = node._tps;
		tps.drawMode = which === "draw" ? !tps.drawMode : false;
		tps.cutMode = which === "cut" ? !tps.cutMode : false;
		tps.eraseMode = which === "sweep" ? !tps.eraseMode : false;
		refreshBrushes();
		draw(node);
	};
	drawBtn.onclick = (e) => { e.stopPropagation(); setMode("draw"); };
	cutBtn.onclick = (e) => { e.stopPropagation(); setMode("cut"); };
	maskBtn.onclick = (e) => {
		e.stopPropagation();
		node._tps.showMasks = !node._tps.showMasks;
		refreshBrushes();
		draw(node);
	};
	for (const b of [drawBtn, cutBtn, maskBtn]) {
		b.addEventListener("pointerdown", (e) => e.stopPropagation());
	}
	brush.addEventListener("input", (e) => {
		e.stopPropagation();
		const r = parseInt(brush.value, 10) || 16;
		node._tps.brushR = r;
		if (brushW && brushW.value !== r) { brushW.value = r; brushW.callback?.(r, app.canvas, node); }
		draw(node);
	});
	brush.addEventListener("pointerdown", (e) => e.stopPropagation());

	node._tps.refreshHistory = () => {
		const tps = node._tps;
		for (const [b, n] of [[undoBtn, tps.history.length], [redoBtn, tps.future.length]]) {
			b.disabled = !n;
			b.style.opacity = n ? "1" : "0.45";
		}
	};
	undoBtn.onclick = (e) => { e.stopPropagation(); undo(node); };
	redoBtn.onclick = (e) => { e.stopPropagation(); redo(node); };
	for (const b of [undoBtn, redoBtn]) b.addEventListener("pointerdown", (e) => e.stopPropagation());
	node._tps.refreshHistory();

	const GROW_STEP = 2;
	const hasGrow = !!getWidget(node, "grow_ids");
	const refreshSel = () => {
		const tps = node._tps;
		const n = tps.selected.size;
		const g = readGrowObj(node).grow;
		const amounts = [...tps.selected].map((id) => g[id] || 0);
		const same = amounts.length && amounts.every((a) => a === amounts[0]);
		selInfo.textContent = n
			? `${n} selected${same ? ` · grow ${amounts[0] > 0 ? "+" : ""}${amounts[0]}px` : ""}`
			: "";
		for (const b of [growBtn, shrinkBtn]) {
			b.disabled = !n || !hasGrow;
			b.style.opacity = (!n || !hasGrow) ? "0.45" : "1";
		}
	};
	const bumpGrow = (delta) => {
		const tps = node._tps;
		if (!hasGrow || !tps.selected.size) return;
		pushHistory(node);
		const { grow } = readGrowObj(node);
		for (const id of tps.selected) {
			grow[id] = clamp((grow[id] || 0) + delta, -64, 64);
		}
		writeGrow(node, grow);
		refreshSel();
		node._tps.refreshClear?.();
		draw(node);
	};
	growBtn.onclick = (e) => { e.stopPropagation(); bumpGrow(GROW_STEP); };
	shrinkBtn.onclick = (e) => { e.stopPropagation(); bumpGrow(-GROW_STEP); };
	for (const b of [growBtn, shrinkBtn]) b.addEventListener("pointerdown", (e) => e.stopPropagation());
	refreshSel();

	eraser.onclick = (e) => { e.stopPropagation(); setMode("sweep"); };
	eraser.addEventListener("pointerdown", (e) => e.stopPropagation());
	refreshBrushes();

	// --- clear ---------------------------------------------------------------
	// Wipes every editor-driven widget at once: swept segments, drawn strokes,
	// erase strokes and per-object grow. Two clicks, because it throws away work
	// that took real time — the first arms it and says how much is at stake, and
	// it disarms itself after a few seconds. It is ONE undo step, so ↶ brings
	// everything back while the editor is still open.
	let armed = null;
	const disarm = () => {
		if (armed) { clearTimeout(armed); armed = null; }
		clearBtn.textContent = "✕ clear";
		clearBtn.style.background = "#2a2f3a";
		clearBtn.style.color = "#cfe3ff";
		clearBtn.style.borderColor = "#3a4150";
	};
	const refreshClear = () => {
		if (armed) return;                       // don't overwrite the armed label
		const c = editCounts(node);
		clearBtn.disabled = !c.total;
		clearBtn.style.opacity = c.total ? "1" : "0.45";
		const parts = [];
		if (c.sel) parts.push(`${c.sel} swept`);
		if (c.drawn) parts.push(`${c.drawn} drawn`);
		if (c.erased) parts.push(`${c.erased} erased`);
		if (c.grow) parts.push(`${c.grow} grown`);
		clearBtn.title = c.total
			? `Clear every edit on this clip (${parts.join(", ")}). Click twice; ↶ undoes it.`
			: "Nothing to clear — no edits on this clip yet.";
	};
	node._tps.refreshClear = refreshClear;
	clearBtn.onclick = (e) => {
		e.stopPropagation();
		const c = editCounts(node);
		if (!c.total) { disarm(); refreshClear(); return; }
		if (!armed) {
			clearBtn.textContent = `✕ clear ${c.total}?`;
			clearBtn.style.background = "#7a2230";
			clearBtn.style.color = "#fff";
			clearBtn.style.borderColor = "#c04a5e";
			armed = setTimeout(() => { armed = null; disarm(); refreshClear(); }, 3000);
			return;
		}
		disarm();
		pushHistory(node);                       // one step: ↶ restores the lot
		writeSelection(node, new Set());
		if (getWidget(node, "painted")) writePaint(node, []);
		if (hasGrow) writeGrow(node, {});
		node._tps.selected.clear();
		refreshSel();
		refreshClear();
		updateCounter(node);
		draw(node);
	};
	clearBtn.addEventListener("pointerdown", (e) => e.stopPropagation());
	refreshClear();

	node.addDOMWidget("segment_picker", "ti_pick_editor", wrap, {
		serialize: false, hideOnZoom: false,
		getMinHeight: () => canvasHeightFor(node),
	});

	// Hide the editor-driven widgets (still serialized with the workflow).
	for (const name of ["excluded_ids", "deleted_items", "current_frame"]) {
		const w = getWidget(node, name);
		if (w) { w.type = "hidden"; w.computeSize = () => [0, -4]; }
	}


	new ResizeObserver(() => draw(node)).observe(wrap);

	// Navigation.
	prev.onclick = (e) => { e.stopPropagation(); loadFrame(node, node._tps.frameIdx - 1); };
	next.onclick = (e) => { e.stopPropagation(); loadFrame(node, node._tps.frameIdx + 1); };
	slider.addEventListener("input", (e) => { e.stopPropagation(); loadFrame(node, parseInt(slider.value, 10)); });
	for (const el of [prev, next, slider]) {
		el.addEventListener("pointerdown", (e) => e.stopPropagation());
	}

	// Keyboard stepping while the editor is focused.
	wrap.tabIndex = 0;
	wrap.addEventListener("pointerenter", () => wrap.focus({ preventScroll: true }));
	wrap.addEventListener("keydown", (e) => {
		if ((e.ctrlKey || e.metaKey) && (e.key === "z" || e.key === "Z")) {
			e.stopPropagation(); e.preventDefault();
			if (e.shiftKey) redo(node); else undo(node);
			return;
		}
		if ((e.ctrlKey || e.metaKey) && (e.key === "y" || e.key === "Y")) {
			e.stopPropagation(); e.preventDefault();
			redo(node);
			return;
		}
		if (e.key === "ArrowLeft") { loadFrame(node, node._tps.frameIdx - 1); e.stopPropagation(); e.preventDefault(); }
		else if (e.key === "ArrowRight") { loadFrame(node, node._tps.frameIdx + 1); e.stopPropagation(); e.preventDefault(); }
	});

	const pos = (e) => pointerPos(canvas, e);

	// Smallest box containing a point — the background-click fallback.
	const boxAt = (node2, cx, cy) => {
		const tps = node2._tps;
		const fr = tps.frames[tps.frameIdx];
		if (!fr || !fr.meta) return -1;
		const v = tps.view;
		let best = -1, bestArea = Infinity;
		for (let i = 0; i < fr.meta.segs.length; i++) {
			const [x0, y0, x1, y1] = fr.meta.segs[i].bbox;
			const bx = v.ox + x0 * v.scale, by = v.oy + y0 * v.scale;
			const bw = (x1 - x0) * v.scale, bh = (y1 - y0) * v.scale;
			if (cx >= bx && cx <= bx + bw && cy >= by && cy <= by + bh && bw * bh < bestArea) {
				best = i; bestArea = bw * bh;
			}
		}
		return best;
	};

	// The "#54" tag sits ABOVE its box, i.e. outside it, so neither the label map
	// nor boxAt can catch it — yet it is the thing you aim at when the boxes are
	// a thicket. Hit-test the tag rect too, mirroring exactly how draw() lays it
	// out (bx, by-12, measureText+6 wide, 12 tall). Topmost first, since later
	// segments are drawn over earlier ones.
	const tagIdxAt = (node2, cx, cy) => {
		const tps = node2._tps;
		const fr = tps.frames[tps.frameIdx];
		if (!fr || !fr.meta) return -1;
		const v = tps.view;
		const ctx = tps.canvas.getContext("2d");
		ctx.font = "10px monospace";
		for (let i = fr.meta.segs.length - 1; i >= 0; i--) {
			const s = fr.meta.segs[i];
			const bx = v.ox + s.bbox[0] * v.scale, by = v.oy + s.bbox[1] * v.scale;
			const tw = ctx.measureText(`#${s.id}`).width + 6;
			if (cx >= bx && cx <= bx + tw && cy >= by - 12 && cy <= by) return i;
		}
		return -1;
	};

	const segIdxAt = (node2, cx, cy) => {
		const tag = tagIdxAt(node2, cx, cy);
		if (tag >= 0) return tag;              // the tag wins: it is drawn on top
		const v = labelAt(node2._tps, cx, cy);
		return v > 0 ? v - 1 : boxAt(node2, cx, cy);
	};

	// Shift-drag = eraser: every segment the cursor touches is EXCLUDED (never
	// re-included), so one sweep can clear a crowd of overlapping detections
	// without 30 individual clicks. `key` matches the click path, so it honours
	// delete-mode (per-frame) vs normal (per-id) exactly the same way.
	// EVERY segment whose bbox (or tag) contains the point — not just the
	// smallest. Clicking picks one thing on purpose; the eraser is a bulk tool,
	// so touching anywhere inside a box takes that box out, overlaps included.
	const hitIndices = (cx, cy) => {
		const tps = node._tps;
		const fr = tps.frames[tps.frameIdx];
		if (!fr || !fr.meta) return [];
		const v = tps.view;
		const out = [];
		for (let i = 0; i < fr.meta.segs.length; i++) {
			const [x0, y0, x1, y1] = fr.meta.segs[i].bbox;
			const bx = v.ox + x0 * v.scale, by = v.oy + y0 * v.scale;
			const bw = (x1 - x0) * v.scale, bh = (y1 - y0) * v.scale;
			if (cx >= bx && cx <= bx + bw && cy >= by && cy <= by + bh) out.push(i);
		}
		const tag = tagIdxAt(node, cx, cy);
		if (tag >= 0 && !out.includes(tag)) out.push(tag);
		return out;
	};

	// A pointermove only reports where the cursor LANDED, so a quick sweep jumps
	// over whole boxes between samples. Walk the line from the previous point in
	// small steps and erase along it, so nothing is skipped at speed.
	const eraseStroke = (cx, cy) => {
		const tps = node._tps;
		const fr = tps.frames[tps.frameIdx];
		if (!fr || !fr.meta) return false;
		let changed = false;
		const pts = [];
		const last = tps.lastErase;
		if (last) {
			const dx = cx - last[0], dy = cy - last[1];
			const steps = Math.max(1, Math.ceil(Math.hypot(dx, dy) / 4));
			for (let s = 1; s <= steps; s++) {
				pts.push([last[0] + (dx * s) / steps, last[1] + (dy * s) / steps]);
			}
		} else {
			pts.push([cx, cy]);
		}
		tps.lastErase = [cx, cy];

		// A painted stroke is erasable too — the brush should not care whether
		// what it passes over came from SAM3 or from your own hand.
		if (getWidget(node, "painted")) {
			const { strokes } = readPaintObj(node);
			const kx = (tps.manifest?.full_w || tps.natW) / tps.natW;
			const keep = strokes.filter((st) => {
				if (st.frame !== tps.frameIdx) return true;
				return !st.pts.some(([sx, sy]) => {
					const [px, py] = sourceToCanvas(tps, sx, sy);
					const rr = Math.max(3, (st.r / kx) * tps.view.scale);
					return pts.some(([qx, qy]) => Math.hypot(qx - px, qy - py) <= rr);
				});
			});
			if (keep.length !== strokes.length) { writePaint(node, keep); changed = true; }
		}

		const excluded = readSelection(node);
		for (const [px, py] of pts) {
			for (const i of hitIndices(px, py)) {
				const key = tps.deleteMode ? `${tps.frameIdx}:${i}` : fr.meta.segs[i].id;
				if (!excluded.has(key)) { excluded.add(key); changed = true; }
			}
		}
		if (changed) { writeSelection(node, excluded); updateCounter(node); }
		return changed;
	};

	canvas.addEventListener("pointermove", (e) => {
		if (!node._tps.manifest) return;
		const [cx, cy] = pos(e);
		node._tps.cursor = [cx, cy];               // for the brush-size ring
		if (node._tps.stroke) {                    // painting
			const [sx, sy] = canvasToSource(node._tps, cx, cy);
			const pts = node._tps.stroke.pts;
			const last = pts[pts.length - 1];
			if (!last || Math.abs(last[0] - sx) > 1 || Math.abs(last[1] - sy) > 1) {
				pts.push([sx, sy]);
			}
			draw(node);
			e.stopPropagation(); e.preventDefault();
			return;
		}
		if (node._tps.erasing) {
			if (eraseStroke(cx, cy)) draw(node);
			e.stopPropagation();
			e.preventDefault();
			return;
		}
		if (node._tps.drawMode || node._tps.cutMode) {
			canvas.style.cursor = "none";            // the ring IS the cursor
			draw(node);
			return;
		}
		const idx = segIdxAt(node, cx, cy);
		if (idx !== node._tps.hoverIdx) {
			node._tps.hoverIdx = idx;
			canvas.style.cursor = node._tps.eraseMode
				? "cell" : (idx >= 0 ? "pointer" : "default");
			draw(node);
		}
	});
	canvas.addEventListener("pointerleave", () => {
		node._tps.cursor = null;
		if (node._tps.hoverIdx != null) { node._tps.hoverIdx = null; }
		draw(node);
	});
	canvas.addEventListener("pointerdown", (e) => {
		if (!node._tps.manifest || e.button !== 0) return;
		releaseGraphPointer(e);
		const [cx, cy] = pos(e);
		// Draw / erase brush: start a new stroke. Both are polylines; only `mode`
		// tells the backend whether to add a segment or bite into the ones there.
		if ((node._tps.drawMode || node._tps.cutMode) && !(e.ctrlKey || e.metaKey)) {
			pushHistory(node);
			const [sx, sy] = canvasToSource(node._tps, cx, cy);
			node._tps.stroke = { id: nextPaintId(node),
								 mode: node._tps.cutMode ? "erase" : "draw",
								 frame: node._tps.frameIdx,
								 r: node._tps.brushR, pts: [[sx, sy]] };
			canvas.setPointerCapture?.(e.pointerId);
			draw(node);
			e.stopPropagation(); e.preventDefault();
			return;
		}
		// Ctrl/Cmd-click = SELECT (for + / −), a separate thing from on/off.
		if (e.ctrlKey || e.metaKey) {
			const idx = segIdxAt(node, cx, cy);
			if (idx >= 0) {
				const fr = node._tps.frames[node._tps.frameIdx];
				const id = fr.meta.segs[idx].id;
				if (node._tps.selected.has(id)) node._tps.selected.delete(id);
				else node._tps.selected.add(id);
				refreshSel();
				draw(node);
			}
			e.stopPropagation(); e.preventDefault();
			return;
		}
		if (e.shiftKey || node._tps.eraseMode) {   // start an eraser stroke
			pushHistory(node);                 // one undo step per STROKE
			node._tps.erasing = true;
			node._tps.lastErase = null;        // a fresh stroke starts at this point
			node._tps.hoverIdx = null;
			canvas.style.cursor = "cell";
			canvas.setPointerCapture?.(e.pointerId);
			eraseStroke(cx, cy);               // take whatever is under the press too
			draw(node);
			e.stopPropagation();
			e.preventDefault();
			return;
		}
		const idx = segIdxAt(node, cx, cy);
		if (idx < 0) return;
		const fr = node._tps.frames[node._tps.frameIdx];
		const key = node._tps.deleteMode
			? `${node._tps.frameIdx}:${idx}`
			: fr.meta.segs[idx].id;
		pushHistory(node);
		const excluded = readSelection(node);
		if (excluded.has(key)) excluded.delete(key); else excluded.add(key);
		writeSelection(node, excluded);
		updateCounter(node);
		draw(node);
		e.stopPropagation();
		e.preventDefault();
	});
	const endStroke = (e) => {
		const tps = node._tps;
		if (!tps.stroke) return;
		const st = tps.stroke;
		tps.stroke = null;
		canvas.releasePointerCapture?.(e.pointerId);
		const { strokes } = readPaintObj(node);
		strokes.push(st);
		writePaint(node, strokes);
		updateCounter(node);
		draw(node);
	};
	canvas.addEventListener("pointerup", endStroke);
	canvas.addEventListener("pointercancel", endStroke);

	const endErase = (e) => {
		if (!node._tps.erasing) return;
		node._tps.erasing = false;
		node._tps.lastErase = null;
		// Stay in sweep mode between strokes — only the modifier is momentary.
		canvas.style.cursor = node._tps.eraseMode ? "cell"
			: ((node._tps.drawMode || node._tps.cutMode) ? "none" : "default");
		canvas.releasePointerCapture?.(e.pointerId);
		draw(node);
	};
	canvas.addEventListener("pointerup", endErase);
	canvas.addEventListener("pointercancel", endErase);
	canvas.addEventListener("contextmenu", (e) => e.stopPropagation());

	requestAnimationFrame(() => draw(node));
}

// Size the node so the canvas matches the frame's aspect ratio — the image
// then fills the width instead of sitting in a small letterboxed strip.
function fitNodeToAspect(node) {
	const tps = node._tps;
	if (!tps.manifest || !tps.natW || !tps.natH) return;
	const barH = tps.bar.offsetHeight || 32;
	const width = Math.max(node.size[0], PREFERRED_W);
	const canvasW = width - 20;                       // wrap/border slack
	const canvasH = clamp(canvasW * (tps.natH / tps.natW), 260, MAX_CANVAS_H);
	node.setSize([width, Math.round(barH + canvasH + 20)]);
	node.setDirtyCanvas?.(true, true);
	requestAnimationFrame(() => draw(node));
}

function applyManifest(node, manifest) {
	const tps = node._tps;
	// A different input invalidates the selection: track ids are reused across
	// clips, so keeping it would drop unrelated objects in the new one.
	const prev = tps.deleteMode ? readDeletedObj(node) : readExcludedObj(node);
	const isNewInput = prev.sig !== manifest.sig;

	tps.manifest = manifest;
	tps.frames = [];
	if (isNewInput) {
		writeSelection(node, new Set());         // clears, stamped with the new sig
		setFrameWidget(node, 0);
		const oldCount = tps.deleteMode ? prev.items.length : prev.ids.length;
		if (oldCount) {
			const label = tps.deleteMode ? "Delete Segments" : "Pick Segments";
			console.info(`[tinode] ${label}: new input — cleared ${oldCount} selection(s).`);
		}
	}
	// Authoritative coordinate space for boxes/label — set NOW so the first
	// paint doesn't use the 512² default before the label image loads.
	tps.natW = manifest.pw || tps.natW;
	tps.natH = manifest.ph || tps.natH;
	tps.slider.max = String(Math.max(0, manifest.num_frames - 1));
	fitNodeToAspect(node);
	const start = isNewInput
		? 0
		: clamp(getWidget(node, "current_frame")?.value ?? 0, 0, manifest.num_frames - 1);
	loadFrame(node, start);
}

app.registerExtension({
	name: "tinode.pickSegments",
	async beforeRegisterNodeDef(nodeType, nodeData) {
		if (!NODE_TYPES.has(nodeData.name)) return;
		const deleteMode = DELETE_TYPES.has(nodeData.name);
		const brushFirst = nodeData.name === "TI_EraseSegments";

		const onNodeCreated = nodeType.prototype.onNodeCreated;
		nodeType.prototype.onNodeCreated = function () {
			const r = onNodeCreated?.apply(this, arguments);
			setup(this, deleteMode, brushFirst);
			this.setSize([Math.max(this.size[0], MIN_NODE_W), Math.max(this.size[1], MIN_NODE_H)]);
			return r;
		};

		const onExecuted = nodeType.prototype.onExecuted;
		nodeType.prototype.onExecuted = function (message) {
			onExecuted?.apply(this, arguments);
			const m = deleteMode ? (message?.ti_delete || message?.ti_erase) : message?.ti_pick;
			if (m && m.length) applyManifest(this, m[0]);
			else console.warn(
				`[tinode] ${deleteMode ? "Delete" : "Pick"} Segments: no editor manifest`,
				message,
			);
		};


		// Keep the node's aspect locked to the frame while the user resizes, so
		// the image always fills the body instead of sitting in dead space.
		const onResize = nodeType.prototype.onResize;
		nodeType.prototype.onResize = function (size) {
			onResize?.apply(this, arguments);
			const tps = this._tps;
			if (!tps || !tps.manifest || !tps.natW || !tps.natH) return;
			if (!Array.isArray(size)) return;
			const barH = tps.bar.offsetHeight || 32;
			const canvasH = clamp((size[0] - 20) * (tps.natH / tps.natW), 260, MAX_CANVAS_H);
			size[1] = Math.round(barH + canvasH + 20);
			requestAnimationFrame(() => draw(this));
		};
		const onConfigure = nodeType.prototype.onConfigure;
		nodeType.prototype.onConfigure = function () {
			const r = onConfigure?.apply(this, arguments);
			if (this._tps) requestAnimationFrame(() => draw(this));
			return r;
		};
	},
});
