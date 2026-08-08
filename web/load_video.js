// UI for "Load Video (ti)": a file-browse/upload button and a video preview.
//
// The `video` widget is a combo of files already in input/. This adds a button
// that opens the OS file picker, uploads the chosen file to input/ (via the
// same /upload/image endpoint the core image loader uses), selects it, and
// shows it in a <video> you can scrub — updated whenever the selection changes.

import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

const NODE_TYPE = "TI_LoadVideo";

function videoWidget(node) {
	return node.widgets?.find((w) => w.name === "video");
}

function previewUrl(name) {
	return api.apiURL(
		`/view?filename=${encodeURIComponent(name)}&type=input&rand=${Math.random().toString(36).slice(2)}`,
	);
}

function showPreview(node, name) {
	const lv = node._lv;
	if (!lv) return;
	if (!name) { lv.video.removeAttribute("src"); lv.video.style.display = "none"; lv.hint.style.display = ""; return; }
	lv.hint.style.display = "none";
	lv.video.style.display = "";
	lv.video.src = previewUrl(name);
}

function setup(node) {
	if (node._lv) return;

	// Video preview.
	const wrap = document.createElement("div");
	wrap.style.cssText = "width:100%;height:100%;min-height:160px;display:flex;align-items:center;"
		+ "justify-content:center;background:#141414;border-radius:4px;overflow:hidden;";
	const hint = document.createElement("span");
	hint.style.cssText = "color:#888;font:12px sans-serif;text-align:center;padding:8px;";
	hint.textContent = "Pick or upload a video — it previews here.";
	const video = document.createElement("video");
	video.controls = true; video.loop = true; video.muted = true; video.playsInline = true;
	video.style.cssText = "max-width:100%;max-height:100%;display:none;";
	wrap.append(hint, video);

	node._lv = { wrap, hint, video };
	node.addDOMWidget("video_preview", "ti_video_preview", wrap, { serialize: false, hideOnZoom: false });

	// The upload button itself comes from the combo's video_upload flag (declared
	// in Python) — ComfyUI's built-in, wired to /upload/image. We just keep the
	// preview in sync with whatever the combo ends up selected on.
	const w = videoWidget(node);
	if (w) {
		const prev = w.callback;
		w.callback = function (v, ...rest) {
			const r = prev?.call(this, v, ...rest);
			showPreview(node, v);
			return r;
		};
	}

	if (node.size[1] < 320) node.setSize([Math.max(node.size[0], 340), 380]);
	requestAnimationFrame(() => showPreview(node, w?.value));
}

app.registerExtension({
	name: "tinode.loadVideo",
	async beforeRegisterNodeDef(nodeType, nodeData) {
		if (nodeData.name !== NODE_TYPE) return;

		const onNodeCreated = nodeType.prototype.onNodeCreated;
		nodeType.prototype.onNodeCreated = function () {
			const r = onNodeCreated?.apply(this, arguments);
			setup(this);
			return r;
		};

		// Restore the preview when a saved workflow loads.
		const onConfigure = nodeType.prototype.onConfigure;
		nodeType.prototype.onConfigure = function () {
			const r = onConfigure?.apply(this, arguments);
			if (this._lv) requestAnimationFrame(() => showPreview(this, videoWidget(this)?.value));
			return r;
		};
	},
});
