// Truly dynamic inputs for the Index Switch nodes.
//
// The node ships with one slot (image1 / string1). When you connect the last
// slot, we addInput a fresh one; when you disconnect and leave trailing empty
// slots, we collapse them back to a single empty. So the slot count tracks
// what you actually wire — no pre-declared ghosts.
//
// Backend accepts the added slots via VALIDATE_INPUTS(**kwargs)->True.

import { app } from "../../scripts/app.js";

const NODES = {
	TI_ImageIndexSwitch: { prefix: "image", type: "IMAGE" },
	TI_StringIndexSwitch: { prefix: "string", type: "STRING" },
};

function dynList(node, prefix) {
	return (node.inputs || []).filter((i) => i && i.name && i.name.startsWith(prefix));
}

function update(node) {
	const { prefix, type } = node.__ti;

	// 1) collapse trailing empties down to a single one
	let guard = 0;
	while (guard++ < 500) {
		const dyn = dynList(node, prefix);
		if (dyn.length < 2) break;
		const last = dyn[dyn.length - 1];
		const prev = dyn[dyn.length - 2];
		if (last.link == null && prev.link == null) {
			node.removeInput(node.inputs.indexOf(last));
			continue;
		}
		break;
	}

	// 2) ensure exactly one empty trailing slot
	const dyn = dynList(node, prefix);
	const last = dyn[dyn.length - 1];
	if (!last || last.link != null) {
		let maxN = 0;
		for (const i of dyn) {
			const n = parseInt(i.name.slice(prefix.length), 10);
			if (n > maxN) maxN = n;
		}
		node.addInput(prefix + (maxN + 1), type);
	}

	node.setDirtyCanvas?.(true, true);
}

function safeUpdate(node, cfg) {
	node.__ti = cfg;
	try { update(node); } catch (e) { console.error("[tinode] indexSwitch", e); }
}

app.registerExtension({
	name: "tinode.indexSwitch",
	async setup() {
		console.log("[tinode] indexSwitch extension loaded");
	},
	beforeRegisterNodeDef(nodeType, nodeData) {
		const cfg = NODES[nodeData.name];
		if (!cfg) return;

		const onCreated = nodeType.prototype.onNodeCreated;
		nodeType.prototype.onNodeCreated = function () {
			onCreated?.apply(this, arguments);
			const self = this;
			requestAnimationFrame(() => safeUpdate(self, cfg));
		};

		const onConn = nodeType.prototype.onConnectionsChange;
		nodeType.prototype.onConnectionsChange = function () {
			onConn?.apply(this, arguments);
			safeUpdate(this, cfg);
		};
	},
});
