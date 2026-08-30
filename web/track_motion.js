// Interactive path recorder for "Track Motion Editor · Circle (ti)".
//
// Press and drag across the canvas: the stroke is recorded WITH ITS TIMING —
// every sample carries the millisecond it was made — so the circle plays back at
// the speed you drew. Pause mid-drag and it waits there; flick and it darts.
// Release and it plays the result back once, immediately.
//
// Everything lives client-side. The path is written to the hidden `track` widget
// as normalised 0..1 points, so it serializes with the workflow and rescales
// with width/height instead of stranding itself off-canvas. The Python node
// re-samples it with the same algorithm mirrored below, so what you preview is
// what renders.

import { app } from "../../scripts/app.js";
import { clamp, fillNodeWidth, getWidget, pointerPos, releaseGraphPointer } from "./lib/editor.js";

const NODE_TYPE = "TI_TrackMotionCircle";
const MIN_NODE_W = 420;
// This is a drawing surface, not a thumbnail — a cramped one makes for a
// cramped gesture. Open big; a node you resized yourself keeps its size.
const PREFERRED_W = 900;
const MIN_CANVAS_H = 240;
const MAX_CANVAS_H = 1400;
const TRAIL = "#4aa3ff";
const CIRCLE = "#ffd479";

const ASPECTS = {
	"1:1": 1, "16:9": 16 / 9, "9:16": 9 / 16, "4:3": 4 / 3, "3:4": 3 / 4,
	"3:2": 3 / 2, "2:3": 2 / 3, "21:9": 21 / 9, "2:1": 2,
};

function readTrack(node) {
	try {
		const v = JSON.parse(getWidget(node, "track")?.value || "{}");
		return Array.isArray(v?.pts) ? v.pts : [];
	} catch { return []; }
}

function writeTrack(node, pts) {
	const w = getWidget(node, "track");
	if (!w) return;
	w.value = JSON.stringify({ pts });
	w.callback?.(w.value, app.canvas, node);
}

function widgetNum(node, name, dflt) {
	const v = getWidget(node, name)?.value;
	const n = typeof v === "number" ? v : parseFloat(v);
	return Number.isFinite(n) ? n : dflt;
}

// Mirrors sample_positions() in circle_track.py — the preview must agree with
// the render, so the two resampling rules are written the same way.
function samplePositions(pts, count, timing) {
	if (count <= 0 || !pts.length) return [];
	if (pts.length === 1 || count === 1) {
		return Array.from({ length: count }, () => [pts[0][0], pts[0][1]]);
	}
	let key;
	let mode = timing;
	if (mode === "recorded") {
		key = pts.map((p) => p[2] ?? 0);
		for (let i = 1; i < key.length; i++) if (key[i] < key[i - 1]) key[i] = key[i - 1];
		if (key[key.length - 1] - key[0] <= 0) mode = "even";
	}
	if (mode !== "recorded") {
		key = [0];
		for (let i = 1; i < pts.length; i++) {
			key.push(key[i - 1] + Math.hypot(pts[i][0] - pts[i - 1][0], pts[i][1] - pts[i - 1][1]));
		}
		if (key[key.length - 1] <= 0) {
			return Array.from({ length: count }, () => [pts[0][0], pts[0][1]]);
		}
	}
	const span = key[key.length - 1] - key[0];
	const at = (value) => {
		let lo = 0, hi = key.length - 1;
		while (lo < hi) {
			const mid = (lo + hi) >> 1;
			if (key[mid] < value) lo = mid + 1; else hi = mid;
		}
		if (lo === 0) return [pts[0][0], pts[0][1]];
		const a = pts[lo - 1], b = pts[lo];
		const s = key[lo] - key[lo - 1];
		const f = s <= 0 ? 0 : (value - key[lo - 1]) / s;
		return [a[0] + (b[0] - a[0]) * f, a[1] + (b[1] - a[1]) * f];
	};
	return Array.from({ length: count }, (_v, f) => at(key[0] + span * (f / (count - 1))));
}

// The canvas shows the OUTPUT frame, letterboxed — so what you draw is framed
// exactly the way the mask will be.
function viewRect(tm, node) {
	const cw = tm.canvas.width, ch = tm.canvas.height;
	const w = Math.max(1, widgetNum(node, "width", 1024));
	const h = Math.max(1, widgetNum(node, "height", 576));
	const scale = Math.min(cw / w, ch / h);
	return { ox: (cw - w * scale) / 2, oy: (ch - h * scale) / 2, scale, w, h };
}

function toNorm(tm, node, cx, cy) {
	const v = viewRect(tm, node);
	return [(cx - v.ox) / (v.w * v.scale), (cy - v.oy) / (v.h * v.scale)];
}

function toCanvas(tm, node, nx, ny) {
	const v = viewRect(tm, node);
	return [v.ox + nx * v.w * v.scale, v.oy + ny * v.h * v.scale];
}

function draw(node) {
	const tm = node._tm;
	if (!tm) return;
	const cv = tm.canvas, ctx = cv.getContext("2d");
	fillNodeWidth(node, tm.wrap);
	const w = Math.max(1, Math.floor(cv.clientWidth));
	const h = Math.max(1, Math.floor(cv.clientHeight));
	if (cv.width !== w || cv.height !== h) { cv.width = w; cv.height = h; }

	ctx.clearRect(0, 0, cv.width, cv.height);
	ctx.fillStyle = "#141414";
	ctx.fillRect(0, 0, cv.width, cv.height);

	const v = viewRect(tm, node);
	// The frame itself, so you can see where the edges are while drawing.
	ctx.fillStyle = "#000";
	ctx.fillRect(v.ox, v.oy, v.w * v.scale, v.h * v.scale);
	ctx.strokeStyle = "#3a4150";
	ctx.lineWidth = 1;
	ctx.strokeRect(v.ox + 0.5, v.oy + 0.5, v.w * v.scale - 1, v.h * v.scale - 1);

	const pts = tm.recording ? tm.live : readTrack(node);
	if (!pts.length) {
		ctx.fillStyle = "#7d8590";
		ctx.font = "13px sans-serif";
		ctx.textAlign = "center";
		ctx.fillText("Press and drag to record a path — the speed you draw is the speed it moves.",
					 cv.width / 2, cv.height / 2);
		ctx.textAlign = "left";
		return;
	}

	// The trail.
	ctx.strokeStyle = TRAIL;
	ctx.lineWidth = 2;
	ctx.lineCap = "round";
	ctx.lineJoin = "round";
	ctx.beginPath();
	pts.forEach(([nx, ny], i) => {
		const [px, py] = toCanvas(tm, node, nx, ny);
		if (i === 0) ctx.moveTo(px, py); else ctx.lineTo(px, py);
	});
	ctx.stroke();

	// Time ticks: a dot every 250ms, so you can SEE where the stroke dawdled —
	// bunched dots are slow, spread dots are fast.
	if (!tm.recording && pts.length > 1 && pts[pts.length - 1][2] > 0) {
		ctx.fillStyle = "rgba(74,163,255,0.85)";
		let next = 250;
		for (const [nx, ny, t] of pts) {
			if (t < next) continue;
			const [px, py] = toCanvas(tm, node, nx, ny);
			ctx.beginPath();
			ctx.arc(px, py, 2.5, 0, Math.PI * 2);
			ctx.fill();
			while (next <= t) next += 250;
		}
	}

	// The circle, at the frame the slider is on, at its true radius — including
	// the radius ramp, so a growing circle is visibly growing as you scrub.
	const total = Math.max(1, Math.round(widgetNum(node, "max_frames", 81)));
	const timing = getWidget(node, "timing")?.value ?? "recorded";
	const pos = samplePositions(pts, total, timing);
	const f = clamp(tm.frame, 0, total - 1);
	const here = pos[f] || pos[pos.length - 1];
	if (here) {
		const [px, py] = toCanvas(tm, node, here[0], here[1]);
		const r0 = Math.max(1, widgetNum(node, "radius", 64));
		const r1 = widgetNum(node, "radius_end", 0);
		const r = (r1 > 0 && total > 1) ? r0 + (r1 - r0) * (f / (total - 1)) : r0;
		const rr = Math.max(1, r * v.scale);
		const ring = Math.max(0, widgetNum(node, "context_width", 0)) * v.scale;
		const feather = Math.max(0, widgetNum(node, "feather", 0) * v.scale);

		// The context ring first, underneath — it is the band the surroundings
		// get generated into, so it has to read as "around", not "on top".
		if (ring > 0.5) {
			ctx.beginPath();
			ctx.arc(px, py, rr + ring, 0, Math.PI * 2);
			ctx.arc(px, py, rr, 0, Math.PI * 2, true);       // punch the hole
			ctx.fillStyle = "rgba(63,191,130,0.28)";
			ctx.fill("evenodd");
			ctx.strokeStyle = "rgba(63,191,130,0.9)";
			ctx.lineWidth = 1;
			ctx.beginPath();
			ctx.arc(px, py, rr + ring, 0, Math.PI * 2);
			ctx.stroke();
		}

		const g = ctx.createRadialGradient(px, py, Math.max(0, rr - feather), px, py, rr);
		g.addColorStop(0, "rgba(255,212,121,0.55)");
		g.addColorStop(1, "rgba(255,212,121,0)");
		ctx.fillStyle = feather > 0.5 ? g : "rgba(255,212,121,0.45)";
		ctx.beginPath();
		ctx.arc(px, py, rr, 0, Math.PI * 2);
		ctx.fill();
		ctx.strokeStyle = CIRCLE;
		ctx.lineWidth = 1.5;
		ctx.stroke();

		// Where it starts and where it ends, faintly, so a ramp is legible
		// without scrubbing the whole clip.
		if (r1 > 0 && Math.abs(r1 - r0) > 1) {
			ctx.setLineDash([3, 3]);
			ctx.strokeStyle = "rgba(255,212,121,0.35)";
			ctx.lineWidth = 1;
			for (const rad of [r0, r1]) {
				ctx.beginPath();
				ctx.arc(px, py, Math.max(1, rad * v.scale), 0, Math.PI * 2);
				ctx.stroke();
			}
			ctx.setLineDash([]);
		}
	}
}

function updateInfo(node) {
	const tm = node._tm;
	if (!tm) return;
	const pts = readTrack(node);
	const total = Math.max(1, Math.round(widgetNum(node, "max_frames", 81)));
	tm.slider.max = String(total - 1);
	tm.frame = clamp(tm.frame, 0, total - 1);
	tm.slider.value = String(tm.frame);
	const dur = pts.length > 1 ? (pts[pts.length - 1][2] - pts[0][2]) / 1000 : 0;
	tm.counter.textContent = pts.length
		? `frame ${tm.frame + 1}/${total}  ·  ${pts.length} pts · ${dur.toFixed(2)}s drawn`
		: `frame ${tm.frame + 1}/${total}  ·  no path yet`;
	tm.clearBtn.disabled = !pts.length;
	tm.clearBtn.style.opacity = pts.length ? "1" : "0.45";
	tm.playBtn.disabled = !pts.length;
	tm.playBtn.style.opacity = pts.length ? "1" : "0.45";
}

// Play at the REAL recorded duration, so the preview runs at the pace the
// stroke will actually produce rather than a fixed frame rate.
function play(node) {
	const tm = node._tm;
	if (!tm || tm.playing) { stop(node); return; }
	const pts = readTrack(node);
	if (!pts.length) return;
	const total = Math.max(1, Math.round(widgetNum(node, "max_frames", 81)));
	const dur = Math.max(0.2, (pts[pts.length - 1][2] - pts[0][2]) / 1000);
	const started = performance.now();
	tm.playing = true;
	tm.playBtn.textContent = "■ stop";
	const step = () => {
		if (!tm.playing) return;
		const f = Math.floor(((performance.now() - started) / 1000 / dur) * total);
		if (f >= total) { stop(node); tm.frame = total - 1; updateInfo(node); draw(node); return; }
		tm.frame = f;
		updateInfo(node);
		draw(node);
		tm.raf = requestAnimationFrame(step);
	};
	tm.raf = requestAnimationFrame(step);
}

function stop(node) {
	const tm = node._tm;
	if (!tm) return;
	tm.playing = false;
	if (tm.raf) cancelAnimationFrame(tm.raf);
	tm.raf = null;
	tm.playBtn.textContent = "▶ play";
}

// Aspect drives HEIGHT from WIDTH: one of the two has to give, and width is the
// one people type. `custom` leaves both alone.
function applyAspect(node) {
	const a = getWidget(node, "aspect")?.value;
	const ratio = ASPECTS[a];
	if (!ratio) return false;
	const wW = getWidget(node, "width"), wH = getWidget(node, "height");
	if (!wW || !wH) return false;
	const h = Math.max(16, Math.round(wW.value / ratio / 8) * 8);
	if (wH.value !== h) { wH.value = h; wH.callback?.(h, app.canvas, node); return true; }
	return false;
}

// The node's body is sized so the canvas matches the output aspect exactly —
// otherwise you draw in a letterboxed strip and lose half the surface.
function fitNode(node, grow = false) {
	const tm = node._tm;
	if (!tm) return;
	const w = Math.max(1, widgetNum(node, "width", 1024));
	const h = Math.max(1, widgetNum(node, "height", 576));
	const width = grow ? Math.max(node.size[0], PREFERRED_W) : Math.max(node.size[0], MIN_NODE_W);
	const canvasH = clamp((width - 20) * (h / w), MIN_CANVAS_H, MAX_CANVAS_H);
	const barH = tm.bar.offsetHeight || 60;
	node.setSize([width, Math.round(barH + canvasH + 20)]);
	node.setDirtyCanvas?.(true, true);
	requestAnimationFrame(() => draw(node));
}

function setup(node) {
	if (node._tm) return;
	const wrap = document.createElement("div");
	wrap.style.cssText = "position:relative;width:100%;height:100%;display:flex;flex-direction:column;box-sizing:border-box;";

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
	const playBtn = mkBtn("▶ play");
	playBtn.title = "Play the path back at the speed it was drawn";
	const slider = document.createElement("input");
	slider.type = "range"; slider.min = "0"; slider.max = "0"; slider.value = "0";
	slider.style.cssText = "flex:1;min-width:0;";
	const counter = document.createElement("span");
	counter.style.cssText = "font:11px monospace;color:#cfd3da;white-space:nowrap;";
	const clearBtn = mkBtn("✕ clear");
	clearBtn.title = "Drop the recorded path and start over";
	const fitBtn = mkBtn("⤢ fit");
	fitBtn.title = "Resize the node so the canvas matches width x height exactly";
	const hint = document.createElement("span");
	hint.style.cssText = "font:11px sans-serif;color:#7d8590;white-space:nowrap;";
	hint.textContent = "drag on the canvas to record";
	navRow.append(playBtn, slider, counter);
	toolRow.append(clearBtn, fitBtn, hint);

	const canvas = document.createElement("canvas");
	canvas.style.cssText = "flex:1;min-height:0;width:100%;border-radius:4px;touch-action:none;display:block;cursor:crosshair;";
	wrap.append(bar, canvas);

	node._tm = {
		wrap, bar, canvas, slider, counter, clearBtn, playBtn,
		frame: 0, recording: false, live: [], t0: 0, playing: false, raf: null,
	};

	node.addDOMWidget("track_editor", "ti_track_editor", wrap, {
		serialize: false, hideOnZoom: false,
		getMinHeight: () => {
			const w = Math.max(node.size?.[0] || MIN_NODE_W, MIN_NODE_W) - 20;
			const ww = Math.max(1, widgetNum(node, "width", 1024));
			const hh = Math.max(1, widgetNum(node, "height", 576));
			return Math.round(clamp(w * (hh / ww), MIN_CANVAS_H, MAX_CANVAS_H));
		},
	});

	const tw = getWidget(node, "track");
	if (tw) { tw.type = "hidden"; tw.computeSize = () => [0, -4]; }

	// Width/height/aspect reshape the canvas the moment they change, so the
	// surface you draw on is always the frame you are drawing for.
	for (const name of ["width", "height", "max_frames", "radius", "feather",
						"radius_end", "context_width"]) {
		const w = getWidget(node, name);
		if (!w) continue;
		const prev = w.callback;
		w.callback = function (...args) {
			const r = prev?.apply(this, args);
			if (name === "width") applyAspect(node);
			if (name === "width" || name === "height") fitNode(node);
			updateInfo(node);
			draw(node);
			return r;
		};
	}
	const aw = getWidget(node, "aspect");
	if (aw) {
		const prev = aw.callback;
		aw.callback = function (...args) {
			const r = prev?.apply(this, args);
			if (applyAspect(node)) fitNode(node);
			draw(node);
			return r;
		};
	}
	const tim = getWidget(node, "timing");
	if (tim) {
		const prev = tim.callback;
		tim.callback = function (...args) {
			const r = prev?.apply(this, args);
			draw(node);
			return r;
		};
	}

	new ResizeObserver(() => draw(node)).observe(wrap);

	playBtn.onclick = (e) => { e.stopPropagation(); play(node); };
	clearBtn.onclick = (e) => {
		e.stopPropagation();
		stop(node);
		writeTrack(node, []);
		node._tm.frame = 0;
		updateInfo(node);
		draw(node);
	};
	fitBtn.onclick = (e) => { e.stopPropagation(); fitNode(node, true); };
	slider.addEventListener("input", (e) => {
		e.stopPropagation();
		stop(node);
		node._tm.frame = parseInt(slider.value, 10) || 0;
		updateInfo(node);
		draw(node);
	});
	for (const el of [playBtn, clearBtn, fitBtn, slider]) {
		el.addEventListener("pointerdown", (e) => e.stopPropagation());
	}

	// --- recording ----------------------------------------------------------
	// The clock starts on pointerdown and every sample is stamped against it.
	// Samples are kept even when the cursor has not moved: standing still IS the
	// gesture — that is how you make the circle hold on a spot.
	const pos = (e) => pointerPos(canvas, e);
	canvas.addEventListener("pointerdown", (e) => {
		if (e.button !== 0) return;
		releaseGraphPointer(e);
		stop(node);
		const tm = node._tm;
		tm.recording = true;
		tm.t0 = performance.now();
		const [cx, cy] = pos(e);
		tm.live = [[...toNorm(tm, node, cx, cy), 0]];
		// A held-still press still records: sample on a timer as well as on move.
		tm.tick = setInterval(() => {
			if (!tm.recording || !tm.last) return;
			tm.live.push([tm.last[0], tm.last[1], performance.now() - tm.t0]);
			draw(node);
		}, 33);
		tm.last = [tm.live[0][0], tm.live[0][1]];
		canvas.setPointerCapture?.(e.pointerId);
		draw(node);
		e.stopPropagation(); e.preventDefault();
	});
	canvas.addEventListener("pointermove", (e) => {
		const tm = node._tm;
		if (!tm.recording) return;
		const [cx, cy] = pos(e);
		const [nx, ny] = toNorm(tm, node, cx, cy);
		tm.last = [nx, ny];
		tm.live.push([nx, ny, performance.now() - tm.t0]);
		draw(node);
		e.stopPropagation(); e.preventDefault();
	});
	const finish = (e) => {
		const tm = node._tm;
		if (!tm.recording) return;
		tm.recording = false;
		if (tm.tick) { clearInterval(tm.tick); tm.tick = null; }
		canvas.releasePointerCapture?.(e.pointerId);
		// Thin the samples: a 60Hz drag over several seconds is hundreds of
		// points that all say the same thing. Keep anything that moved or that
		// carries timing information, drop the rest.
		const kept = tm.live.filter((p, i, a) =>
			i === 0 || i === a.length - 1 ||
			Math.hypot(p[0] - a[i - 1][0], p[1] - a[i - 1][1]) > 0.002 ||
			p[2] - a[i - 1][2] > 40);
		writeTrack(node, kept);
		tm.live = [];
		tm.frame = 0;
		updateInfo(node);
		draw(node);
		play(node);                    // show what was just recorded, straight away
	};
	canvas.addEventListener("pointerup", finish);
	canvas.addEventListener("pointercancel", finish);
	canvas.addEventListener("contextmenu", (e) => e.stopPropagation());

	updateInfo(node);
	requestAnimationFrame(() => { applyAspect(node); fitNode(node, true); });
}

app.registerExtension({
	name: "tinode.trackMotion",
	async beforeRegisterNodeDef(nodeType, nodeData) {
		if (nodeData.name !== NODE_TYPE) return;

		const onNodeCreated = nodeType.prototype.onNodeCreated;
		nodeType.prototype.onNodeCreated = function () {
			const r = onNodeCreated?.apply(this, arguments);
			setup(this);
			return r;
		};

		// Keep the canvas locked to the output aspect while the node is resized.
		const onResize = nodeType.prototype.onResize;
		nodeType.prototype.onResize = function (size) {
			onResize?.apply(this, arguments);
			const tm = this._tm;
			if (!tm || !Array.isArray(size)) return;
			const w = Math.max(1, widgetNum(this, "width", 1024));
			const h = Math.max(1, widgetNum(this, "height", 576));
			const barH = tm.bar.offsetHeight || 60;
			const canvasH = clamp((size[0] - 20) * (h / w), MIN_CANVAS_H, MAX_CANVAS_H);
			size[1] = Math.round(barH + canvasH + 20);
			requestAnimationFrame(() => draw(this));
		};

		const onConfigure = nodeType.prototype.onConfigure;
		nodeType.prototype.onConfigure = function () {
			const r = onConfigure?.apply(this, arguments);
			if (this._tm) requestAnimationFrame(() => { updateInfo(this); draw(this); });
			return r;
		};

		const onRemoved = nodeType.prototype.onRemoved;
		nodeType.prototype.onRemoved = function () {
			stop(this);
			if (this._tm?.tick) clearInterval(this._tm.tick);
			return onRemoved?.apply(this, arguments);
		};
	},
});
