// Show "Image Info (ti)"'s reading in the node body.
//
// The Python node returns {"ui": {"text": [ "WxH, N frame(s), C channel(s)" ]}};
// onExecuted renders that into a small read-only panel mounted on the node, so
// the width / height / frame count are visible on the graph without opening the
// console or wiring the outputs.

import { app } from "../../scripts/app.js";

function panel(node) {
	if (node._ti_info) return node._ti_info;
	const el = document.createElement("div");
	el.style.cssText = [
		"width:100%", "box-sizing:border-box", "padding:6px 8px",
		"font:13px/1.4 ui-monospace,monospace", "color:#e6e6e6",
		"background:#181818", "border-radius:4px", "text-align:center",
		"white-space:pre-wrap", "word-break:break-word",
	].join(";");
	el.textContent = "run to read";
	node.addDOMWidget("ti_info", "ti_info", el, {
		serialize: false, hideOnZoom: false, getMinHeight: () => 30,
	});
	node._ti_info = el;
	return el;
}

app.registerExtension({
	name: "tinode.imageInfo",
	async beforeRegisterNodeDef(nodeType, nodeData) {
		if (nodeData.name !== "TI_ImageInfo") return;

		const onNodeCreated = nodeType.prototype.onNodeCreated;
		nodeType.prototype.onNodeCreated = function () {
			const r = onNodeCreated?.apply(this, arguments);
			panel(this);
			return r;
		};

		const onExecuted = nodeType.prototype.onExecuted;
		nodeType.prototype.onExecuted = function (message) {
			onExecuted?.apply(this, arguments);
			const text = message?.text;
			if (text && text.length) {
				panel(this).textContent = Array.isArray(text) ? text.join("\n") : String(text);
				this.setDirtyCanvas?.(true, true);
			}
		};
	},
});
