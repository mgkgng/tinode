// Frontend for "Video Preview · Ephemeral (ti)".
//
// The node holds the clip in the server's RAM and hands us a session id. We mount
// a <video> pointed at a proxy streamed from that RAM — nothing was written to
// output/ or temp/, so there is no /view URL to use and no file to clean up.
//
// CREATE VIDEO posts the encode settings back, gets the finished file as the
// response body, and hands it to the browser as a download. The server keeps no
// copy: it is encoded, sent, and forgotten.

import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";
import {
	CTRL, PRIMARY, el, errorText, mb, number, prop, releaseSession,
	releaseUnlessShared, safeName, saveBlob, select,
} from "./lib/preview_ui.js";

const NODE_TYPE = "TI_VideoPreview";

// slug -> [label, extension, lossy?]. Lossy formats get the quality + colour row.
const FORMATS = {
	h264:   ["mp4 · h264", "mp4", true],
	vp9:    ["webm · vp9", "webm", true],
	ffv1:   ["mkv · ffv1 — lossless", "mkv", false],
	prores: ["mov · prores 4444", "mov", false],
	png:    ["zip · png frames — lossless", "zip", false],
};

function build(node) {
	if (node._tvp) return node._tvp;

	const root = el("div", "width:100%;height:100%;display:flex;flex-direction:column;gap:4px;"
		+ "box-sizing:border-box;");

	const stage = el("div", "flex:1 1 auto;min-height:120px;display:flex;align-items:center;"
		+ "justify-content:center;background:#111;border-radius:4px;overflow:hidden;");
	const hint = el("span", "color:#777;font:12px sans-serif;text-align:center;padding:0 10px;",
		"Queue to render — the clip plays here and is never saved.");
	stage.appendChild(hint);

	const video = document.createElement("video");
	video.style.cssText = "max-width:100%;max-height:100%;display:none;";
	video.loop = true;
	video.controls = true;
	video.autoplay = true;
	video.playsInline = true;
	// Muted so autoplay is allowed at all. Once the user has unmuted once, the
	// page has had a gesture and later runs can come back with sound — so the
	// choice is remembered rather than reimposed on every queue.
	video.muted = true;
	stage.appendChild(video);

	const row1 = el("div", "display:flex;gap:4px;align-items:center;");
	const row2 = el("div", "display:flex;gap:4px;align-items:center;");
	const status = el("div", "font:10px/1.3 ui-monospace,monospace;color:#8a8a8a;"
		+ "white-space:nowrap;overflow:hidden;text-overflow:ellipsis;", "nothing held");

	const fmt = select(Object.entries(FORMATS).map(([k, v]) => [k, v[0]]),
		prop(node, "ti_dl_format", "h264"), (v) => { node.properties.ti_dl_format = v; sync(); });

	const create = el("button", PRIMARY, "CREATE VIDEO");
	const release = el("button", CTRL + "flex:0 0 auto;cursor:pointer;padding:3px 7px;", "✕");
	release.title = "Release this clip from RAM now";

	const crfLabel = el("span", "font:10px sans-serif;color:#888;flex:0 0 auto;", "crf");
	const crf = number(prop(node, "ti_dl_crf", 17), 0, 51,
		"Quality — lower is better. 0 is lossless for h264.",
		(v) => { node.properties.ti_dl_crf = v; });

	const pix = select([["yuv420p", "4:2:0"], ["yuv444p", "4:4:4"]],
		prop(node, "ti_dl_pix_fmt", "yuv420p"), (v) => { node.properties.ti_dl_pix_fmt = v; });
	pix.title = "Chroma subsampling. 4:4:4 keeps full colour resolution.";

	const range = select([["tv", "tv range"], ["pc", "pc range"], ["unspecified", "untagged"]],
		prop(node, "ti_dl_range", "tv"), (v) => { node.properties.ti_dl_range = v; });
	range.title = "Match your source. tv = limited (16-235), pc = full (0-255).";

	row1.append(fmt, create, release);
	row2.append(crfLabel, crf, pix, range);
	root.append(stage, row1, row2, status);

	const tvp = { root, stage, hint, video, fmt, crf, pix, range, create, release, status,
				  row2, session: null, info: null, busy: false };
	node._tvp = tvp;

	function sync() {
		// The lossless formats have no quality or colour knobs to offer.
		tvp.row2.style.display = FORMATS[prop(node, "ti_dl_format", "h264")]?.[2] ? "flex" : "none";
	}
	tvp.sync = sync;
	sync();

	video.addEventListener("volumechange", () => {
		node.properties.ti_unmuted = !video.muted;
		node.properties.ti_volume = video.volume;
	});
	create.addEventListener("click", () => download(node));
	release.addEventListener("click", () => releaseNow(node));

	node.addDOMWidget("ti_vpreview", "ti_vpreview", root, {
		serialize: false, hideOnZoom: false, getMinHeight: () => 200,
	});
	if (node.size[1] < 320) node.setSize([Math.max(node.size[0], 340), 380]);
	return tvp;
}

function setStatus(node, text, colour) {
	const tvp = node._tvp;
	if (!tvp) return;
	tvp.status.textContent = text;
	tvp.status.style.color = colour || "#8a8a8a";
}

function describe(info) {
	const a = info.audio;
	const track = a
		? ` · audio ${(+a.seconds).toFixed(2)}s ${Math.round(a.rate / 100) / 10} kHz `
			+ (a.channels === 1 ? "mono" : a.channels === 2 ? "stereo" : `${a.channels} ch`)
		: " · no audio";
	return `${info.frames} frame${info.frames === 1 ? "" : "s"} · ${info.width}×${info.height}`
		+ ` · ${(+info.fps).toFixed(2).replace(/\.?0+$/, "")} fps · ${info.depth}-bit${track}`
		+ ` · ${mb(info.bytes)} held in RAM, no file written`;
}

function show(node, info) {
	const tvp = build(node);
	tvp.session = info.id;
	tvp.info = info;
	// Remembered so a browser reload can pick the clip back up — the server still
	// holds it, and losing the panel to an F5 would be a needless re-run. Dead
	// after a ComfyUI restart, which is what the stat check on load is for.
	node.properties.ti_session = info.id;
	node.properties.ti_session_info = info;
	tvp.hint.style.display = "none";
	tvp.video.style.display = "block";
	if (info.audio && node.properties.ti_unmuted) {
		tvp.video.muted = false;
		tvp.video.volume = node.properties.ti_volume ?? 1;
	}
	// Cache-bust: a new run reuses the element but never the bytes.
	tvp.video.src = api.apiURL(`/tinode/vpreview/proxy?id=${encodeURIComponent(info.id)}&v=${Date.now()}`);
	tvp.video.play?.().catch(() => {});
	setStatus(node, describe(info));
	node.setDirtyCanvas?.(true, true);
}

function expired(node, msg) {
	const tvp = node._tvp;
	if (!tvp) return;
	tvp.session = null;
	tvp.info = null;
	delete node.properties.ti_session;
	delete node.properties.ti_session_info;
	tvp.video.removeAttribute("src");
	tvp.video.load?.();
	tvp.video.style.display = "none";
	tvp.hint.style.display = "block";
	tvp.hint.textContent = msg || "Preview released — queue the graph again.";
	setStatus(node, "nothing held");
	node.setDirtyCanvas?.(true, true);
}

async function releaseNow(node) {
	const id = node._tvp?.session;
	if (!id) return;
	expired(node, "Released. Queue the graph again to preview.");
	await releaseSession(id);
}

async function download(node) {
	const tvp = node._tvp;
	if (!tvp || tvp.busy) return;
	if (!tvp.session) {
		setStatus(node, "nothing to render — queue the graph first", "#d08770");
		return;
	}

	const fmt = prop(node, "ti_dl_format", "h264");
	const label = tvp.create.textContent;
	tvp.busy = true;
	tvp.create.disabled = true;
	tvp.create.textContent = "RENDERING…";
	setStatus(node, `encoding ${FORMATS[fmt][0]} …`, "#9ecbff");

	try {
		const res = await api.fetchApi("/tinode/vpreview/render", {
			method: "POST", headers: { "Content-Type": "application/json" },
			body: JSON.stringify({
				id: tvp.session,
				format: fmt,
				fps: tvp.info?.fps,
				crf: Number(tvp.crf.value) || 17,
				pix_fmt: tvp.pix.value,
				color_range: tvp.range.value,
				colorspace: "bt709",
				name: safeName(node, `_${tvp.info?.frames || 0}f`),
			}),
		});

		if (!res.ok) {
			if (res.status === 404) expired(node);
			setStatus(node, await errorText(res, `render failed (${res.status})`), "#d08770");
			return;
		}

		const blob = await res.blob();
		const name = `${safeName(node, `_${tvp.info.frames}f`)}.${FORMATS[fmt][1]}`;
		saveBlob(blob, name);
		setStatus(node, `downloaded ${name} — ${mb(blob.size)} · ${describe(tvp.info)}`, "#a3d9a5");
	} catch (err) {
		setStatus(node, `render failed: ${err?.message || err}`, "#d08770");
	} finally {
		tvp.busy = false;
		tvp.create.disabled = false;
		tvp.create.textContent = label;
	}
}

// On load, ask the server whether the remembered clip is still held. It usually
// is after a browser reload and never is after a ComfyUI restart, and the panel
// has to be honest about which — a dead <video> src just shows a broken player.
async function recheck(node) {
	const tvp = node._tvp;
	const id = tvp?.session || node.properties?.ti_session;
	if (!id) return;
	try {
		const res = await api.fetchApi(`/tinode/vpreview/stat?id=${encodeURIComponent(id)}`);
		const stat = await res.json();
		if (!stat.ok) {
			expired(node);
			return;
		}
		if (!tvp.session) show(node, { ...(node.properties.ti_session_info || {}), ...stat });
	} catch (e) { /* server not reachable — leave the panel as it is */ }
}

app.registerExtension({
	name: "tinode.videoPreview",
	async beforeRegisterNodeDef(nodeType, nodeData) {
		if (nodeData.name !== NODE_TYPE) return;

		const onNodeCreated = nodeType.prototype.onNodeCreated;
		nodeType.prototype.onNodeCreated = function () {
			const r = onNodeCreated?.apply(this, arguments);
			build(this);
			return r;
		};

		const onExecuted = nodeType.prototype.onExecuted;
		nodeType.prototype.onExecuted = function (message) {
			onExecuted?.apply(this, arguments);
			const v = message?.ti_vpreview;
			if (v && v.length) show(this, v[0]);
		};

		const onConfigure = nodeType.prototype.onConfigure;
		nodeType.prototype.onConfigure = function () {
			const r = onConfigure?.apply(this, arguments);
			build(this).sync();
			recheck(this);
			return r;
		};

		const onRemoved = nodeType.prototype.onRemoved;
		nodeType.prototype.onRemoved = function () {
			releaseUnlessShared(this, this._tvp?.session);
			return onRemoved?.apply(this, arguments);
		};
	},
});
