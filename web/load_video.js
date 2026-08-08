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

async function uploadFile(node, file) {
	try {
		const body = new FormData();
		body.append("image", file, file.name);           // endpoint field is "image"
		const resp = await api.fetchApi("/upload/image", { method: "POST", body });
		if (resp.status !== 200) { alert(`Upload failed (${resp.status})`); return; }
		const data = await resp.json();
		const name = data.subfolder ? `${data.subfolder}/${data.name}` : data.name;

		const w = videoWidget(node);
		if (w) {
			const vals = w.options.values || (w.options.values = []);
			if (!vals.includes(name)) vals.push(name);
			w.value = name;
			w.callback?.(name);
		}
		showPreview(node, name);
		node.setDirtyCanvas?.(true, true);
	} catch (e) {
		console.error("[tinode] Load Video upload failed", e);
		alert("Upload failed — see console.");
	}
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

	// Hidden native file picker.
	const picker = document.createElement("input");
	picker.type = "file";
	picker.accept = "video/*";
	picker.style.display = "none";
	picker.addEventListener("change", () => {
		if (picker.files?.[0]) uploadFile(node, picker.files[0]);
		picker.value = "";
	});
	document.body.appendChild(picker);

	node._lv = { wrap, hint, video, picker };
	node.addDOMWidget("video_preview", "ti_video_preview", wrap, { serialize: false, hideOnZoom: false });

	// Browse/upload button.
	node.addWidget("button", "📁 choose / upload video", null, () => picker.click());

	// Keep the preview in sync with the combo selection.
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
