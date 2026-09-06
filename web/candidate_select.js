// Click-to-pick grid for "Candidate Select (ti)".
//
// Choosing a candidate is a visual judgement, so the node shows the candidates
// rather than a number: a thumbnail grid, click one, it gets a bright border and
// the `index` widget follows. Selecting is still an ordinary graph edit, so you
// re-queue when you are ready — the click does not run anything by itself.
//
// The grid is whatever the last run produced. When the cursor sits on a
// candidate the run did not cover (index typed by hand, or a re-queue pending),
// the header says so instead of implying the picture below is current.

import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";
import { getWidget } from "./lib/editor.js";

const NODE_TYPE = "TI_CandidateSelect";

// Tiles are a fixed small size rather than a fraction of the node. With `1fr`
// columns a wide node grew the thumbnails until the grid outgrew the widget's
// reserved height and spilled past the node border. A candidate only has to be
// recognisable, so cap it and let the row wrap instead.
const TILE = 56;
const GAP = 4;
const HEAD_H = 24;
// Litegraph gives the DOM widget a content-sized box, so `width:100%` inside it
// resolves against nothing and fixed-width grid tracks push the widget WIDER
// than the node instead of wrapping. Size it from node.size[0] instead, which is
// the only authoritative width, and never measure our own (possibly overflowing)
// element to decide the layout.
const PAD = 26;

// How many tiles fit across the node as it is RIGHT NOW. Fixing the column
// count instead (repeat(5, ...)) forced everything onto one row and ran off the
// edge as soon as the batch outgrew the node's width.
//   n columns occupy n*TILE + (n-1)*GAP, so n <= (width + GAP) / (TILE + GAP)
function innerWidth(node) {
	return Math.max(TILE, Math.round(node.size?.[0] ?? 300) - PAD);
}

function colsFor(node, n) {
	//   n columns occupy n*TILE + (n-1)*GAP,  so  n <= (width + GAP) / (TILE + GAP)
	const fit = Math.floor((innerWidth(node) + GAP) / (TILE + GAP));
	return Math.max(1, Math.min(n, fit));
}

function setIndex(node, v) {
	const w = getWidget(node, "index");
	if (!w) return;
	w.value = Math.max(0, Math.round(v));
	w.callback?.(w.value, app.canvas, node);
	paint(node);
	node.setDirtyCanvas?.(true, true);
}

function paint(node) {
	const t = node._tcs;
	if (!t) return;
	const idx = Math.round(getWidget(node, "index")?.value ?? 0);
	const { count, origin, thumbs } = t.state;

	t.head.textContent = !count
		? "queue once to see the candidates"
		: `${Math.min(idx, count - 1) + 1} / ${count}`
			+ (origin !== null ? `   ·   seed ${origin + Math.min(idx, count - 1)}` : "")
			+ (idx >= count ? "   ·   out of range" : "");

	// Pin the box to the node's width so nothing inside can widen it.
	const inner = innerWidth(node) + "px";
	t.wrap.style.width = inner;
	t.wrap.style.maxWidth = inner;
	t.grid.style.width = inner;
	t.grid.style.maxWidth = inner;

	t.grid.replaceChildren();
	t.rows = 0;
	if (!thumbs.length) return;
	const cols = colsFor(node, thumbs.length);
	t.rows = Math.ceil(thumbs.length / cols);
	// Fixed-width columns: the row fills up, then wraps to the next line.
	t.grid.style.gridTemplateColumns = `repeat(${cols}, ${TILE}px)`;
	thumbs.forEach((th, i) => {
		const cell = document.createElement("div");
		const on = i === idx;
		cell.style.cssText = "position:relative;border-radius:4px;overflow:hidden;cursor:pointer;"
			+ `min-width:0;height:${TILE}px;box-sizing:border-box;`
			+ `border:2px solid ${on ? "#7fd1ff" : "#2a2f3a"};`
			+ (on ? "box-shadow:0 0 0 2px rgba(127,209,255,.35);" : "opacity:.72;");
		const img = document.createElement("img");
		img.src = api.apiURL(`/view?filename=${encodeURIComponent(th.filename)}`
			+ `&type=${th.type}&subfolder=${encodeURIComponent(th.subfolder || "")}`
			+ `&t=${t.state.stamp}`);
		img.style.cssText = "width:100%;height:100%;display:block;object-fit:cover;";
		const tag = document.createElement("div");
		tag.textContent = i + 1;
		tag.style.cssText = "position:absolute;left:3px;top:2px;font:600 10px sans-serif;"
			+ "color:#fff;text-shadow:0 1px 3px #000;";
		cell.append(img, tag);
		cell.onmouseenter = () => { if (i !== idx) cell.style.opacity = "1"; };
		cell.onmouseleave = () => { if (i !== idx) cell.style.opacity = ".72"; };
		cell.onclick = (e) => { e.stopPropagation(); setIndex(node, i); };
		t.grid.append(cell);
	});

	// A different candidate count needs a different number of rows, so let
	// litegraph re-measure (it consults getMinHeight) and grow the node if the
	// grid no longer fits. Only ever grows: shrinking would fight a manual resize.
	if (t.rows !== t.lastRows) {
		t.lastRows = t.rows;
		const want = node.computeSize?.();
		if (want && node.size[1] < want[1]) node.setSize([node.size[0], want[1]]);
		node.setDirtyCanvas?.(true, true);
	}
}

function setup(node) {
	if (node._tcs) return;
	const wrap = document.createElement("div");
	// overflow:hidden is the backstop — whatever the grid does, it cannot paint
	// outside the box litegraph reserved for this widget.
	wrap.style.cssText = "display:flex;flex-direction:column;gap:4px;padding:2px;"
		+ "box-sizing:border-box;width:100%;overflow:hidden;";

	const head = document.createElement("div");
	head.style.cssText = "font:11px monospace;color:#cfe3ff;background:#1b1f27;"
		+ `border:1px solid #333a46;border-radius:4px;padding:3px 6px;text-align:center;`
		+ `height:${HEAD_H}px;box-sizing:border-box;flex:0 0 auto;`
		+ "white-space:nowrap;overflow:hidden;text-overflow:ellipsis;";

	const grid = document.createElement("div");
	// justify-content:start keeps a short row left-aligned instead of stretching.
	grid.style.cssText = "display:grid;gap:4px;justify-content:start;"
		+ "align-content:start;overflow-y:auto;overflow-x:hidden;";

	wrap.append(head, grid);
	node._tcs = { wrap, head, grid, rows: 0, lastRows: -1,
				  state: { count: 0, origin: null, thumbs: [], stamp: 0 } };
	node.addDOMWidget("candidate_ui", "ti_candidate_ui", wrap, {
		serialize: false, hideOnZoom: false,
		// Ask for exactly the height the current grid needs, so the node neither
		// clips the last row nor reserves empty space for candidates it lacks.
		getMinHeight: () => HEAD_H + 6 + Math.max(1, node._tcs?.rows || 1) * (TILE + 4),
	});

	const w = getWidget(node, "index");
	if (w) {
		const prevCb = w.callback;
		w.callback = function (...a) { const r = prevCb?.apply(this, a); paint(node); return r; };
	}
	requestAnimationFrame(() => paint(node));
}

app.registerExtension({
	name: "tinode.candidateSelect",
	async beforeRegisterNodeDef(nodeType, nodeData) {
		if (nodeData.name !== NODE_TYPE) return;

		const onNodeCreated = nodeType.prototype.onNodeCreated;
		nodeType.prototype.onNodeCreated = function () {
			const r = onNodeCreated?.apply(this, arguments);
			setup(this);
			// Wide enough for a full row of tiles, tall enough for one row of them.
			this.setSize([Math.max(this.size[0], 300), Math.max(this.size[1], 180)]);
			return r;
		};

		const onExecuted = nodeType.prototype.onExecuted;
		nodeType.prototype.onExecuted = function (message) {
			onExecuted?.apply(this, arguments);
			const m = message?.ti_candidate;
			if (m && m.length && this._tcs) {
				this._tcs.state = {
					count: m[0].count || 0,
					origin: m[0].origin_seed ?? null,
					thumbs: m[0].thumbs || [],
					// Filenames are reused across runs; bust the browser cache.
					stamp: Date.now(),
				};
				paint(this);
			}
		};

		// Dragging the node's edge changes how many tiles fit, so rebuild then —
		// watching our own element instead would just observe the width we set.
		const onResize = nodeType.prototype.onResize;
		nodeType.prototype.onResize = function (...a) {
			const r = onResize?.apply(this, a);
			if (this._tcs) paint(this);
			return r;
		};

		const onConfigure = nodeType.prototype.onConfigure;
		nodeType.prototype.onConfigure = function () {
			const r = onConfigure?.apply(this, arguments);
			if (this._tcs) requestAnimationFrame(() => paint(this));
			return r;
		};
	},
});
