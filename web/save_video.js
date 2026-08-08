// Preview for "Save Video · Combine (ti)".
//
// ComfyUI renders image previews natively but not video, so on execution the
// node hands us {filename, subfolder, type, format} via ui.ti_video and we mount
// a looping <video> (or an <img> for the png-sequence format) in the node body.

import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

const NODE_TYPE = "TI_SaveVideo";

function urlFor(info) {
	return api.apiURL(
		`/view?filename=${encodeURIComponent(info.filename)}` +
		`&type=${info.type || "output"}` +
		`&subfolder=${encodeURIComponent(info.subfolder || "")}` +
		`&rand=${Math.random().toString(36).slice(2)}`,
	);
}

function setup(node) {
	if (node._tv) return;
	const wrap = document.createElement("div");
	wrap.style.cssText = "width:100%;height:100%;min-height:180px;display:flex;"
		+ "align-items:center;justify-content:center;background:#141414;border-radius:4px;overflow:hidden;";
	const hint = document.createElement("span");
	hint.style.cssText = "color:#888;font:12px sans-serif;";
	hint.textContent = "Queue to render — the video previews here.";
	wrap.appendChild(hint);
	node._tv = { wrap, hint, media: null };
	node.addDOMWidget("video_preview", "ti_video_preview", wrap, { serialize: false, hideOnZoom: false });
	if (node.size[1] < 300) node.setSize([Math.max(node.size[0], 320), 360]);
}

function show(node, info) {
	const tv = node._tv;
	if (!tv) return;
	if (tv.media) { tv.media.remove(); tv.media = null; }
	tv.hint.style.display = "none";

	const isVideo = (info.format || "").startsWith("video");
	const el = document.createElement(isVideo ? "video" : "img");
	el.style.cssText = "max-width:100%;max-height:100%;display:block;";
	if (isVideo) {
		el.autoplay = true; el.loop = true; el.muted = true; el.controls = true;
		el.playsInline = true;
	}
	el.src = urlFor(info);
	tv.hint.style.display = "none";
	tv.wrap.appendChild(el);
	tv.media = el;
}

app.registerExtension({
	name: "tinode.saveVideo",
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
			const v = message?.ti_video;
			if (v && v.length) show(this, v[0]);
		};
	},
});
