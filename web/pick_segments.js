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
import { api } from "../../scripts/api.js";

const NODE_TYPE = "TI_PickSegments";
const MIN_NODE_W = 360;
const MIN_NODE_H = 520;

// Must match color_for_id() in pick_segments.py so a segment is the same color
// in the editor and in the rendered image output.
function colorForId(id) {
	const h = ((id * 0.61803398875) % 1.0 + 1.0) % 1.0;
	const s = 0.65, v = 1.0;
	const i = Math.floor(h * 6);
	const f = h * 6 - i;
	const p = v * (1 - s), q = v * (1 - s * f), t = v * (1 - s * (1 - f));
	const table = [[v, t, p], [q, v, p], [p, v, t], [p, q, v], [t, p, v], [v, p, q]];
	const [r, g, b] = table[i % 6];
	return [Math.round(r * 255), Math.round(g * 255), Math.round(b * 255)];
}

function getWidget(node, name) {
	return node.widgets?.find((w) => w.name === name);
}

function readExcluded(node) {
	try {
		return new Set(JSON.parse(getWidget(node, "excluded_ids")?.value || "[]"));
	} catch {
		return new Set();
	}
}

function writeExcluded(node, set) {
	const w = getWidget(node, "excluded_ids");
	if (!w) return;
	w.value = JSON.stringify([...set]);
	w.callback?.(w.value, app.canvas, node);
}

function setFrameWidget(node, f) {
	const w = getWidget(node, "current_frame");
	if (w && w.value !== f) { w.value = f; w.callback?.(f, app.canvas, node); }
}

function urlFor(info) {
	return api.apiURL(
		`/view?filename=${encodeURIComponent(info.filename)}` +
		`&type=${info.type || "temp"}` +
		`&subfolder=${encodeURIComponent(info.subfolder || "")}` +
		`&rand=${encodeURIComponent(info.filename)}`,
	);
}

function clamp(v, lo, hi) { return Math.max(lo, Math.min(v, hi)); }

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
	const w = Math.max(1, Math.floor(cv.clientWidth));
	const h = Math.max(1, Math.floor(cv.clientHeight));
	if (cv.width !== w || cv.height !== h) { cv.width = w; cv.height = h; }

	ctx.clearRect(0, 0, cv.width, cv.height);
	ctx.fillStyle = "#141414";
	ctx.fillRect(0, 0, cv.width, cv.height);

	if (!tps.manifest) {
		ctx.fillStyle = "#888"; ctx.font = "12px sans-serif"; ctx.textAlign = "center";
		ctx.fillText("Run once to load the segments, then click objects to toggle.",
			cv.width / 2, cv.height / 2);
		return;
	}

	tps.view = viewRect(tps);
	const v = tps.view;
	const fr = tps.frames[tps.frameIdx];
	if (fr && fr.src) ctx.drawImage(fr.src, v.ox, v.oy, tps.natW * v.scale, tps.natH * v.scale);

	if (!fr || !fr.meta) return;
	const excluded = readExcluded(node);

	// Hovered segment: paint its real mask shape.
	if (tps.hoverIdx != null && tps.hoverIdx >= 0) {
		const hl = buildHighlight(tps, tps.hoverIdx);
		if (hl) ctx.drawImage(hl, v.ox, v.oy, tps.natW * v.scale, tps.natH * v.scale);
	}

	// Boxes: included solid in their color, excluded dashed/dim.
	ctx.lineWidth = 2;
	ctx.font = "10px monospace";
	ctx.textAlign = "left";
	for (let i = 0; i < fr.meta.segs.length; i++) {
		const s = fr.meta.segs[i];
		const [x0, y0, x1, y1] = s.bbox;
		const bx = v.ox + x0 * v.scale, by = v.oy + y0 * v.scale;
		const bw = (x1 - x0) * v.scale, bh = (y1 - y0) * v.scale;
		const [r, g, b] = colorForId(s.id);
		const off = excluded.has(s.id);
		ctx.setLineDash(off ? [4, 3] : []);
		ctx.strokeStyle = off ? `rgba(${r},${g},${b},0.35)` : `rgb(${r},${g},${b})`;
		ctx.strokeRect(bx, by, bw, bh);
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
}

function updateCounter(node) {
	const tps = node._tps;
	const excluded = readExcluded(node);
	const total = tps.manifest.ids.length;
	const on = tps.manifest.ids.filter((i) => !excluded.has(i)).length;
	tps.counter.textContent =
		`frame ${tps.frameIdx + 1}/${tps.manifest.num_frames}   ·   ${on}/${total} on`;
}

function setup(node) {
	if (node._tps) return;
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
	const prev = mkBtn("◀");
	const next = mkBtn("▶");
	const slider = document.createElement("input");
	slider.type = "range"; slider.min = "0"; slider.max = "0"; slider.value = "0";
	slider.style.cssText = "flex:1;min-width:0;";
	const counter = document.createElement("span");
	counter.style.cssText = "font:11px monospace;color:#cfd3da;white-space:nowrap;";
	bar.append(prev, slider, next, counter);

	const canvas = document.createElement("canvas");
	canvas.style.cssText = "flex:1;min-height:0;width:100%;border-radius:4px;touch-action:none;display:block;";
	wrap.append(bar, canvas);

	node._tps = {
		wrap, bar, canvas, slider, counter, prev, next,
		manifest: null, frames: [], frameIdx: 0,
		natW: 512, natH: 512, view: { ox: 0, oy: 0, scale: 1 }, hoverIdx: null,
	};

	node.addDOMWidget("segment_picker", "ti_pick_editor", wrap, { serialize: false, hideOnZoom: false });

	// Hide the editor-driven widgets (still serialized with the workflow).
	for (const name of ["excluded_ids", "current_frame"]) {
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
		if (e.key === "ArrowLeft") { loadFrame(node, node._tps.frameIdx - 1); e.stopPropagation(); e.preventDefault(); }
		else if (e.key === "ArrowRight") { loadFrame(node, node._tps.frameIdx + 1); e.stopPropagation(); e.preventDefault(); }
	});

	// ── pointer on the canvas ──
	const pos = (e) => {
		const r = canvas.getBoundingClientRect();
		return [
			(e.clientX - r.left) * (canvas.width / (r.width || 1)),
			(e.clientY - r.top) * (canvas.height / (r.height || 1)),
		];
	};

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

	const segIdxAt = (node2, cx, cy) => {
		const v = labelAt(node2._tps, cx, cy);
		return v > 0 ? v - 1 : boxAt(node2, cx, cy);
	};

	canvas.addEventListener("pointermove", (e) => {
		if (!node._tps.manifest) return;
		const [cx, cy] = pos(e);
		const idx = segIdxAt(node, cx, cy);
		if (idx !== node._tps.hoverIdx) {
			node._tps.hoverIdx = idx;
			canvas.style.cursor = idx >= 0 ? "pointer" : "default";
			draw(node);
		}
	});
	canvas.addEventListener("pointerleave", () => {
		if (node._tps.hoverIdx != null) { node._tps.hoverIdx = null; draw(node); }
	});
	canvas.addEventListener("pointerdown", (e) => {
		if (!node._tps.manifest || e.button !== 0) return;
		const [cx, cy] = pos(e);
		const idx = segIdxAt(node, cx, cy);
		if (idx < 0) return;
		const fr = node._tps.frames[node._tps.frameIdx];
		const id = fr.meta.segs[idx].id;
		const excluded = readExcluded(node);
		if (excluded.has(id)) excluded.delete(id); else excluded.add(id);
		writeExcluded(node, excluded);
		updateCounter(node);
		draw(node);
		e.stopPropagation();
		e.preventDefault();
	});
	canvas.addEventListener("contextmenu", (e) => e.stopPropagation());

	requestAnimationFrame(() => draw(node));
}

function applyManifest(node, manifest) {
	const tps = node._tps;
	tps.manifest = manifest;
	tps.frames = [];
	tps.slider.max = String(Math.max(0, manifest.num_frames - 1));
	const start = clamp(getWidget(node, "current_frame")?.value ?? 0, 0, manifest.num_frames - 1);
	loadFrame(node, start);
}

app.registerExtension({
	name: "tinode.pickSegments",
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
			const m = message?.ti_pick;
			if (m && m.length) applyManifest(this, m[0]);
			else console.warn("[tinode] Pick Segments: no ti_pick in message", message);
		};

		const onConfigure = nodeType.prototype.onConfigure;
		nodeType.prototype.onConfigure = function () {
			const r = onConfigure?.apply(this, arguments);
			if (this._tps) requestAnimationFrame(() => draw(this));
			return r;
		};
	},
});
