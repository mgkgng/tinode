// Step controls for "Item Cursor (ti)".
//
// Adds ◀ Prev / Next ▶ / Reset under the node and shows where the cursor is
// ("3 / 19 · sh0030 · crop 1"). The buttons just move the `index` widget, so
// advancing is an ordinary graph edit — you re-queue when you are ready, which
// is the whole point: retry the current item as often as you like, then step.

import { app } from "../../scripts/app.js";
import { getWidget } from "./lib/editor.js";

const NODE_TYPE = "TI_ItemCursor";

function setIndex(node, v) {
	const w = getWidget(node, "index");
	if (!w) return;
	const next = Math.max(0, Math.round(v));
	if (w.value === next) return;
	w.value = next;
	w.callback?.(next, app.canvas, node);
	paint(node);
	node.setDirtyCanvas?.(true, true);
}

function paint(node) {
	const t = node._tic;
	if (!t) return;
	const idx = Math.round(getWidget(node, "index")?.value ?? 0);
	const { count, label } = t.state;
	t.readout.textContent = count
		? `${Math.min(idx, count - 1) + 1} / ${count}${label ? "  ·  " + label : ""}`
		: `index ${idx}  ·  queue once to see the list`;
	// Make "there is nothing after this" obvious before you press Next.
	t.next.style.opacity = count && idx >= count - 1 ? "0.45" : "1";
	t.prev.style.opacity = idx <= 0 ? "0.45" : "1";
}

function setup(node) {
	if (node._tic) return;
	const wrap = document.createElement("div");
	wrap.style.cssText = "display:flex;flex-direction:column;gap:5px;padding:2px;"
		+ "box-sizing:border-box;width:100%;";

	const readout = document.createElement("div");
	readout.style.cssText = "font:12px monospace;color:#cfe3ff;background:#1b1f27;"
		+ "border:1px solid #333a46;border-radius:4px;padding:5px 7px;text-align:center;"
		+ "white-space:nowrap;overflow:hidden;text-overflow:ellipsis;";

	const row = document.createElement("div");
	row.style.cssText = "display:flex;gap:5px;";
	const mk = (text, title, onClick, grow) => {
		const b = document.createElement("button");
		b.textContent = text;
		b.title = title;
		b.style.cssText = `flex:${grow || 1};padding:5px 0;border:1px solid #3a4150;`
			+ "border-radius:4px;background:#2a2f3a;color:#cfe3ff;cursor:pointer;"
			+ "font:600 12px sans-serif;";
		b.onclick = (e) => { e.stopPropagation(); onClick(); };
		return b;
	};
	const prev = mk("◀ Prev", "Back one item", () => setIndex(node, cur(node) - 1));
	const next = mk("Next ▶", "Forward one item", () => setIndex(node, cur(node) + 1));
	const reset = mk("⟲", "Back to the first item", () => setIndex(node, 0), 0.4);
	row.append(reset, prev, next);
	wrap.append(readout, row);

	node._tic = { wrap, readout, prev, next, state: { count: 0, label: "" } };
	node.addDOMWidget("cursor_ui", "ti_cursor_ui", wrap, {
		serialize: false, hideOnZoom: false, getMinHeight: () => 74,
	});

	// Repaint when the index is typed by hand.
	const w = getWidget(node, "index");
	if (w) {
		const prevCb = w.callback;
		w.callback = function (...a) { const r = prevCb?.apply(this, a); paint(node); return r; };
	}
	requestAnimationFrame(() => paint(node));
}

function cur(node) { return Math.round(getWidget(node, "index")?.value ?? 0); }

app.registerExtension({
	name: "tinode.itemCursor",
	async beforeRegisterNodeDef(nodeType, nodeData) {
		if (nodeData.name !== NODE_TYPE) return;

		const onNodeCreated = nodeType.prototype.onNodeCreated;
		nodeType.prototype.onNodeCreated = function () {
			const r = onNodeCreated?.apply(this, arguments);
			setup(this);
			this.setSize([Math.max(this.size[0], 260), Math.max(this.size[1], 150)]);
			return r;
		};

		const onExecuted = nodeType.prototype.onExecuted;
		nodeType.prototype.onExecuted = function (message) {
			onExecuted?.apply(this, arguments);
			const m = message?.ti_cursor;
			if (m && m.length && this._tic) {
				this._tic.state = { count: m[0].count || 0, label: m[0].label || "" };
				paint(this);
			}
		};

		const onConfigure = nodeType.prototype.onConfigure;
		nodeType.prototype.onConfigure = function () {
			const r = onConfigure?.apply(this, arguments);
			if (this._tic) requestAnimationFrame(() => paint(this));
			return r;
		};
	},
});
