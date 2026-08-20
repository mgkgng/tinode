// Frame editor for "Split Video · Chunks (ti)".
//
// You pick cut points by LOOKING at the video, so this node shows it: scrub the
// whole clip (slider, ◀ ▶, arrow keys), and press "cut here" to split at the
// frame you are on — the `cuts` widget is written for you, so you never have to
// read frame numbers off another player and type them in.
//
// A timeline strip under the image draws the resulting chunks in alternating
// colours with the cut positions marked, so the split is legible at a glance
// and you can see which chunk the current frame falls in.

import { app } from "../../scripts/app.js";
import { clamp, fillNodeWidth, getWidget, urlFor } from "./lib/editor.js";

const NODE_TYPE = "TI_SplitVideoChunks";
const MIN_NODE_W = 320;
const MIN_NODE_H = 420;
const MAX_CANVAS_H = 2000;
const STRIP_H = 26;             // timeline height, canvas px
const CHUNK_COLS = ["#3f789e", "#9e6b3f", "#3f9e6b", "#7a3f9e", "#9e3f5c"];

function readCuts(node) {
	const raw = String(getWidget(node, "cuts")?.value ?? "");
	const n = node._tsv?.total || 0;
	const out = [];
	for (const tok of raw.replace(/,/g, " ").split(/\s+/)) {
		if (!tok) continue;
		let i = parseInt(tok, 10);
		if (Number.isNaN(i)) continue;
		if (i < 0) i += n;
		if (i > 0 && (!n || i < n) && !out.includes(i)) out.push(i);
	}
	return out.sort((a, b) => a - b);
}

function writeCuts(node, cuts) {
	const w = getWidget(node, "cuts");
	if (!w) return;
	w.value = cuts.join(", ");
	w.callback?.(w.value, app.canvas, node);
	draw(node);
}

// Chunk spans implied by the current cuts — mirrors the backend's chunk_bounds
// (without max_length, which the backend applies on top).
function spans(node) {
	const t = node._tsv;
	const n = t.total;
	if (!n) return [];
	const edges = [0, ...readCuts(node), n];
	const out = [];
	for (let i = 0; i < edges.length - 1; i++) {
		if (edges[i + 1] > edges[i]) out.push([edges[i], edges[i + 1]]);
	}
	return out;
}

function chunkAt(node, frame) {
	const s = spans(node);
	for (let i = 0; i < s.length; i++) if (frame >= s[i][0] && frame < s[i][1]) return i;
	return -1;
}

function draw(node) {
	const t = node._tsv;
	if (!t) return;
	const cv = t.canvas, ctx = cv.getContext("2d");
	fillNodeWidth(node, t.container);
	const w = Math.max(1, Math.floor(cv.clientWidth));
	const h = Math.max(1, Math.floor(cv.clientHeight));
	if (cv.width !== w || cv.height !== h) { cv.width = w; cv.height = h; }

	ctx.clearRect(0, 0, cv.width, cv.height);
	ctx.fillStyle = "#181818";
	ctx.fillRect(0, 0, cv.width, cv.height);

	const imgH = Math.max(1, cv.height - STRIP_H - 6);
	if (!t.img) {
		ctx.fillStyle = "#888";
		ctx.font = "12px sans-serif";
		ctx.textAlign = "center";
		ctx.fillText("Run once to load the clip, then scrub and cut.", cv.width / 2, imgH / 2);
		return;
	}

	// Letterbox the frame above the strip.
	const scale = Math.min(cv.width / t.natW, imgH / t.natH);
	const dw = t.natW * scale, dh = t.natH * scale;
	const ox = (cv.width - dw) / 2, oy = (imgH - dh) / 2;
	ctx.drawImage(t.img, ox, oy, dw, dh);

	// Timeline: one band per chunk, cut positions marked, playhead on top.
	const sy = cv.height - STRIP_H;
	const sp = spans(node);
	const n = Math.max(1, t.total);
	ctx.fillStyle = "#111";
	ctx.fillRect(0, sy, cv.width, STRIP_H);
	sp.forEach(([a, b], i) => {
		const x0 = (a / n) * cv.width, x1 = (b / n) * cv.width;
		ctx.fillStyle = CHUNK_COLS[i % CHUNK_COLS.length];
		ctx.fillRect(x0, sy + 4, Math.max(1, x1 - x0 - 1), STRIP_H - 8);
		const label = `${i}: ${b - a}f`;
		ctx.font = "10px monospace";
		if (ctx.measureText(label).width < x1 - x0 - 6) {
			ctx.fillStyle = "#fff";
			ctx.textAlign = "left";
			ctx.fillText(label, x0 + 4, sy + STRIP_H / 2 + 4);
		}
	});
	// Cut markers.
	ctx.strokeStyle = "#fff";
	ctx.lineWidth = 1;
	for (const c of readCuts(node)) {
		const x = (c / n) * cv.width;
		ctx.beginPath(); ctx.moveTo(x, sy); ctx.lineTo(x, sy + STRIP_H); ctx.stroke();
	}
	// Playhead.
	const px = (t.frame / n) * cv.width;
	ctx.strokeStyle = "#ff3b3b";
	ctx.lineWidth = 2;
	ctx.beginPath(); ctx.moveTo(px, sy - 3); ctx.lineTo(px, sy + STRIP_H); ctx.stroke();

	// Readout.
	const ci = chunkAt(node, t.frame);
	const label = `frame ${t.frame}/${Math.max(0, t.total - 1)}   chunk ${ci < 0 ? "-" : ci}`
		+ `   ${sp.length} chunk(s)`;
	ctx.font = "11px monospace";
	ctx.textAlign = "left";
	const tw = ctx.measureText(label).width;
	ctx.fillStyle = "rgba(0,0,0,0.6)";
	ctx.fillRect(4, 4, tw + 10, 18);
	ctx.fillStyle = "#e6e6e6";
	ctx.fillText(label, 9, 17);
}

function showFrame(node, i) {
	const t = node._tsv;
	if (!t.frames.length) return;
	t.frame = clamp(i, 0, t.frames.length - 1);
	if (t.slider) t.slider.value = String(t.frame);
	node.properties = node.properties || {};
	node.properties.ti_frame = t.frame;      // UI state, never a backend input
	const cached = t.cache.get(t.frame);
	if (cached) { t.img = cached; draw(node); return; }
	const img = new Image();
	const want = t.frame;
	img.onload = () => { t.cache.set(want, img); if (t.frame === want) { t.img = img; draw(node); } };
	img.onerror = () => console.error("[tinode] Split Video: frame failed", want);
	img.src = urlFor(t.frames[want]);
}

function step(node, d) { showFrame(node, node._tsv.frame + d); }

// Toggle a cut at the current frame: press once to split here, again to undo.
function toggleCut(node) {
	const t = node._tsv;
	const f = t.frame;
	if (f <= 0) return;                       // 0 would make an empty first chunk
	const cuts = readCuts(node);
	const at = cuts.indexOf(f);
	if (at >= 0) cuts.splice(at, 1); else cuts.push(f);
	writeCuts(node, cuts.sort((a, b) => a - b));
}

function applyManifest(node, m) {
	const t = node._tsv;
	if (!m || !Array.isArray(m.frames) || !m.frames.length) return;
	if (m.sig !== t.sig) { t.cache.clear(); t.sig = m.sig; }
	t.natW = Math.max(1, m.pw || m.full_w || t.natW);
	t.natH = Math.max(1, m.ph || m.full_h || t.natH);
	t.frames = m.frames;
	t.total = m.num_frames || m.frames.length;
	if (t.slider) t.slider.max = String(t.frames.length - 1);
	showFrame(node, clamp(Math.round(node.properties?.ti_frame ?? 0), 0, t.frames.length - 1));
}

function canvasHeightFor(node) {
	const t = node._tsv;
	if (!t) return 300;
	const w = Math.max(node.size?.[0] || MIN_NODE_W, MIN_NODE_W) - 20;
	return Math.round(clamp(w * ((t.natH || 1) / (t.natW || 1)) + STRIP_H + 6, 300, MAX_CANVAS_H));
}

function setupEditor(node) {
	if (node._tsv) return;
	const container = document.createElement("div");
	container.style.cssText = "position:relative;width:100%;height:100%;min-height:300px;"
		+ "display:flex;flex-direction:column;box-sizing:border-box;";
	const canvas = document.createElement("canvas");
	canvas.style.cssText = "flex:1;min-height:220px;width:100%;border-radius:4px;display:block;";
	container.appendChild(canvas);

	const bar = document.createElement("div");
	bar.style.cssText = "display:flex;align-items:center;gap:6px;padding:4px 2px 0;"
		+ "font:11px monospace;color:#ccc;flex:0 0 auto;";
	const mkBtn = (txt, fn, wide) => {
		const b = document.createElement("button");
		b.textContent = txt;
		b.style.cssText = "background:#2a2f3a;color:#cfe3ff;border:1px solid #3a4150;"
			+ `border-radius:4px;cursor:pointer;padding:1px 7px;font:${wide ? "600 11px sans-serif" : "12px monospace"};`;
		b.onclick = (e) => { e.stopPropagation(); fn(); };
		return b;
	};
	const slider = document.createElement("input");
	slider.type = "range"; slider.min = "0"; slider.max = "0"; slider.value = "0";
	slider.style.cssText = "flex:1;min-width:40px;accent-color:#4aa3ff;";
	slider.oninput = (e) => { e.stopPropagation(); showFrame(node, parseInt(slider.value, 10)); };
	bar.append(
		mkBtn("◀", () => step(node, -1)),
		slider,
		mkBtn("▶", () => step(node, 1)),
		mkBtn("✂ cut here", () => toggleCut(node), true),
		mkBtn("clear", () => writeCuts(node, []), true),
	);
	container.appendChild(bar);

	node._tsv = {
		container, canvas, slider, img: null, frames: [], cache: new Map(),
		frame: 0, total: 0, natW: 512, natH: 512, sig: null,
	};

	node.addDOMWidget("split_editor", "ti_split_editor", container, {
		serialize: false, hideOnZoom: false, getMinHeight: () => canvasHeightFor(node),
	});
	new ResizeObserver(() => draw(node)).observe(container);

	canvas.tabIndex = 0;
	canvas.addEventListener("pointerenter", () => canvas.focus({ preventScroll: true }));
	canvas.addEventListener("keydown", (e) => {
		if (e.key === "ArrowLeft") { step(node, -1); e.preventDefault(); e.stopPropagation(); }
		else if (e.key === "ArrowRight") { step(node, 1); e.preventDefault(); e.stopPropagation(); }
		else if (e.key === "c" || e.key === "C") { toggleCut(node); e.preventDefault(); e.stopPropagation(); }
	});
	// Click the timeline strip to jump there.
	canvas.addEventListener("pointerdown", (e) => {
		const r = canvas.getBoundingClientRect();
		const y = (e.clientY - r.top) * (canvas.height / r.height);
		if (y < canvas.height - STRIP_H - 4) return;
		const x = (e.clientX - r.left) * (canvas.width / r.width);
		showFrame(node, Math.round((x / canvas.width) * Math.max(1, node._tsv.total)));
		e.stopPropagation();
	});
	canvas.addEventListener("wheel", (e) => e.stopPropagation());

	// Repaint when `cuts` is typed by hand.
	const w = getWidget(node, "cuts");
	if (w) {
		const prev = w.callback;
		w.callback = function (...a) { const r = prev?.apply(this, a); draw(node); return r; };
	}
	requestAnimationFrame(() => draw(node));
}

app.registerExtension({
	name: "tinode.splitVideo",
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
			const m = message?.ti_cuts;
			if (m && m.length) applyManifest(this, m[0]);
		};

		const onConfigure = nodeType.prototype.onConfigure;
		nodeType.prototype.onConfigure = function () {
			const r = onConfigure?.apply(this, arguments);
			if (this._tsv) requestAnimationFrame(() => draw(this));
			return r;
		};
	},
});
