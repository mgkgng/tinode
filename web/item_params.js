// Node UI for "Item Params (ti)".
//
// Keeps the `entry` box showing the CURRENT key's settings. When you queue, the
// backend replies with the key it actually used and that key's stored entry;
// this writes both back into the widgets, so stepping to another item swaps the
// box to that item's settings instead of leaving the previous one's text behind
// (which the backend would otherwise refuse to save, by design).

import { app } from "../../scripts/app.js";
import { getWidget } from "./lib/editor.js";

const NODE_TYPE = "TI_ItemParams";

function setW(node, name, value) {
	const w = getWidget(node, name);
	if (!w || w.value === value) return;
	w.value = value;
	w.callback?.(value, app.canvas, node);
}

function paint(node, detail) {
	const t = node._tip;
	if (!t) return;
	const known = detail?.known || t.known || [];
	t.known = known;
	const key = detail?.key ?? getWidget(node, "key")?.value ?? "";
	t.readout.textContent = key
		? `${key}${known.includes(key) ? "  ·  saved" : "  ·  new"}   (${known.length} stored)`
		: "no key — wire the clip stem";
	t.readout.style.color = known.includes(key) ? "#8fe3b0" : "#ffd479";
}

function setup(node) {
	if (node._tip) return;
	const wrap = document.createElement("div");
	wrap.style.cssText = "display:flex;flex-direction:column;gap:4px;padding:2px;width:100%;box-sizing:border-box;";
	const readout = document.createElement("div");
	readout.style.cssText = "font:11px monospace;background:#1b1f27;border:1px solid #333a46;"
		+ "border-radius:4px;padding:4px 6px;text-align:center;white-space:nowrap;"
		+ "overflow:hidden;text-overflow:ellipsis;";
	wrap.appendChild(readout);
	node._tip = { wrap, readout, known: [] };
	node.addDOMWidget("params_ui", "ti_params_ui", wrap, {
		serialize: false, hideOnZoom: false, getMinHeight: () => 26,
	});
	// entry_key is bookkeeping the node maintains; hide it so it isn't edited.
	const ek = getWidget(node, "entry_key");
	if (ek) { ek.type = "hidden"; ek.computeSize = () => [0, -4]; }
	requestAnimationFrame(() => paint(node));
}

app.registerExtension({
	name: "tinode.itemParams",
	async beforeRegisterNodeDef(nodeType, nodeData) {
		if (nodeData.name !== NODE_TYPE) return;

		const onNodeCreated = nodeType.prototype.onNodeCreated;
		nodeType.prototype.onNodeCreated = function () {
			const r = onNodeCreated?.apply(this, arguments);
			setup(this);
			return r;
		};

		const onExecuted = nodeType.prototype.onExecuted;
		nodeType.prototype.onExecuted = function (message) {
			onExecuted?.apply(this, arguments);
			const m = message?.ti_params;
			if (!m || !m.length) return;
			const d = m[0];
			// Show this key's stored settings, and stamp which key they belong to
			// so the backend will accept the next edit.
			setW(this, "entry", d.entry ?? "{}");
			setW(this, "entry_key", d.key ?? "");
			paint(this, d);
		};

		const onConfigure = nodeType.prototype.onConfigure;
		nodeType.prototype.onConfigure = function () {
			const r = onConfigure?.apply(this, arguments);
			if (this._tip) requestAnimationFrame(() => paint(this));
			return r;
		};
	},
});
