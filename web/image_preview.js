// Frontend for "Image Preview · Ephemeral (ti)".
//
// The batch lives in the server's RAM. Each frame is encoded on demand and
// streamed from there into an <img> — no PNG in temp/, no queue-history entry,
// nothing to tidy up afterwards. DOWNLOAD asks for the frame (or the whole
// batch as a zip) at full size and full bit depth, and hands the response body
// straight to the browser's downloader.

import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";
import {
	CTRL, PRIMARY, el, errorText, mb, number, prop, releaseSession,
	releaseUnlessShared, safeName, saveBlob, select,
} from "./lib/preview_ui.js";

const NODE_TYPE = "TI_ImagePreview";

// slug -> [label, extension, has a quality knob?]
const FORMATS = {
	png:  ["png — lossless", "png", false],
	tiff: ["tiff — lossless", "tiff", false],
	jpg:  ["jpg", "jpg", true],
	webp: ["webp", "webp", true],
};

// What the on-screen view is fetched at. Big enough to judge, small enough to
// feel instant on a 4K batch; a download never goes through this.
const VIEW_MAX = 1600;

function build(node) {
	if (node._tip) return node._tip;

	const root = el("div", "width:100%;height:100%;display:flex;flex-direction:column;gap:4px;"
		+ "box-sizing:border-box;");

	const stage = el("div", "flex:1 1 auto;min-height:120px;display:flex;align-items:center;"
		+ "justify-content:center;background:#111;border-radius:4px;overflow:hidden;");
	const hint = el("span", "color:#777;font:12px sans-serif;text-align:center;padding:0 10px;",
		"Queue to render — the image appears here and is never saved.");
	stage.appendChild(hint);

	const img = document.createElement("img");
	img.style.cssText = "max-width:100%;max-height:100%;display:none;object-fit:contain;";
	stage.appendChild(img);

	// Batch navigation. Hidden for a single image, where it would be noise.
	const nav = el("div", "display:flex;gap:4px;align-items:center;justify-content:center;");
	const prev = el("button", CTRL + "cursor:pointer;padding:2px 8px;", "‹");
	const next = el("button", CTRL + "cursor:pointer;padding:2px 8px;", "›");
	const counter = el("span", "font:10px ui-monospace,monospace;color:#9a9a9a;min-width:56px;"
		+ "text-align:center;", "1 / 1");
	const scrub = el("input", "flex:1 1 auto;min-width:0;accent-color:#4b82bd;");
	scrub.type = "range";
	scrub.min = "0";
	scrub.value = "0";
	nav.append(prev, scrub, counter, next);

	const row = el("div", "display:flex;gap:4px;align-items:center;");
	const status = el("div", "font:10px/1.3 ui-monospace,monospace;color:#8a8a8a;"
		+ "white-space:nowrap;overflow:hidden;text-overflow:ellipsis;", "nothing held");

	const fmt = select(Object.entries(FORMATS).map(([k, v]) => [k, v[0]]),
		prop(node, "ti_img_format", "png"), (v) => { node.properties.ti_img_format = v; sync(); });

	const quality = number(prop(node, "ti_img_quality", 95), 1, 100,
		"Quality, 1-100. 100 is lossless for webp.",
		(v) => { node.properties.ti_img_quality = v; });

	const scope = select([["one", "this frame"], ["all", "all frames · zip"]],
		prop(node, "ti_img_scope", "one"), (v) => { node.properties.ti_img_scope = v; });
	scope.title = "Download the frame on screen, or the whole batch as a zip.";

	const save = el("button", PRIMARY, "DOWNLOAD");
	const release = el("button", CTRL + "flex:0 0 auto;cursor:pointer;padding:3px 7px;", "✕");
	release.title = "Release these images from RAM now";

	row.append(fmt, quality, scope, save, release);
	root.append(stage, nav, row, status);

	const tip = { root, stage, hint, img, nav, prev, next, counter, scrub, fmt, quality,
				  scope, save, release, status, session: null, info: null, index: 0,
				  busy: false };
	node._tip = tip;

	function sync() {
		tip.quality.style.display = FORMATS[prop(node, "ti_img_format", "png")]?.[2] ? "" : "none";
		const many = (tip.info?.frames || 1) > 1;
		tip.nav.style.display = many ? "flex" : "none";
		tip.scope.style.display = many ? "" : "none";
	}
	tip.sync = sync;
	sync();

	prev.addEventListener("click", () => step(node, -1));
	next.addEventListener("click", () => step(node, +1));
	scrub.addEventListener("input", () => showFrame(node, Number(scrub.value)));
	save.addEventListener("click", () => download(node));
	release.addEventListener("click", () => releaseNow(node));

	node.addDOMWidget("ti_ipreview", "ti_ipreview", root, {
		serialize: false, hideOnZoom: false, getMinHeight: () => 200,
	});
	if (node.size[1] < 320) node.setSize([Math.max(node.size[0], 360), 380]);
	return tip;
}

function setStatus(node, text, colour) {
	const tip = node._tip;
	if (!tip) return;
	tip.status.textContent = text;
	tip.status.style.color = colour || "#8a8a8a";
}

function describe(info) {
	return `${info.frames} image${info.frames === 1 ? "" : "s"} · ${info.width}×${info.height}`
		+ ` · ${info.depth}-bit · ${mb(info.bytes)} held in RAM, no file written`;
}

function showFrame(node, i) {
	const tip = node._tip;
	if (!tip?.session || !tip.info) return;
	const n = tip.info.frames || 1;
	tip.index = Math.max(0, Math.min(Math.round(i), n - 1));
	tip.scrub.value = String(tip.index);
	tip.counter.textContent = `${tip.index + 1} / ${n}`;
	tip.img.src = api.apiURL(
		`/tinode/vpreview/frame?id=${encodeURIComponent(tip.session)}`
		+ `&i=${tip.index}&fmt=jpg&q=92&max=${VIEW_MAX}&v=${tip.info.id}`);
}

function step(node, by) {
	const tip = node._tip;
	if (!tip?.info) return;
	// Wrap: stepping past the last frame of a batch should land on the first.
	const n = tip.info.frames || 1;
	showFrame(node, (tip.index + by + n) % n);
}

function show(node, info) {
	const tip = build(node);
	tip.session = info.id;
	tip.info = info;
	node.properties.ti_session = info.id;
	node.properties.ti_session_info = info;
	tip.hint.style.display = "none";
	tip.img.style.display = "block";
	tip.scrub.max = String(Math.max(0, (info.frames || 1) - 1));
	tip.sync();
	showFrame(node, Math.min(tip.index, (info.frames || 1) - 1));
	setStatus(node, describe(info));
	node.setDirtyCanvas?.(true, true);
}

function expired(node, msg) {
	const tip = node._tip;
	if (!tip) return;
	tip.session = null;
	tip.info = null;
	delete node.properties.ti_session;
	delete node.properties.ti_session_info;
	tip.img.removeAttribute("src");
	tip.img.style.display = "none";
	tip.hint.style.display = "block";
	tip.hint.textContent = msg || "Preview released — queue the graph again.";
	tip.sync();
	setStatus(node, "nothing held");
	node.setDirtyCanvas?.(true, true);
}

async function releaseNow(node) {
	const id = node._tip?.session;
	if (!id) return;
	expired(node, "Released. Queue the graph again to preview.");
	await releaseSession(id);
}

async function download(node) {
	const tip = node._tip;
	if (!tip || tip.busy) return;
	if (!tip.session) {
		setStatus(node, "nothing to save — queue the graph first", "#d08770");
		return;
	}

	const fmt = prop(node, "ti_img_format", "png");
	const all = (tip.info.frames || 1) > 1 && prop(node, "ti_img_scope", "one") === "all";
	const stem = safeName(node, all ? `_${tip.info.frames}f` : `_${String(tip.index).padStart(5, "0")}`);
	const label = tip.save.textContent;
	tip.busy = true;
	tip.save.disabled = true;
	tip.save.textContent = "SAVING…";
	setStatus(node, all ? `encoding ${tip.info.frames} × ${FORMATS[fmt][0]} …`
					   : `encoding ${FORMATS[fmt][0]} …`, "#9ecbff");

	try {
		const res = await api.fetchApi("/tinode/vpreview/still", {
			method: "POST", headers: { "Content-Type": "application/json" },
			body: JSON.stringify({
				id: tip.session, format: fmt, all,
				index: tip.index, quality: Number(tip.quality.value) || 95, name: stem,
			}),
		});

		if (!res.ok) {
			if (res.status === 404) expired(node);
			setStatus(node, await errorText(res, `save failed (${res.status})`), "#d08770");
			return;
		}

		const blob = await res.blob();
		const name = `${stem}.${all ? "zip" : FORMATS[fmt][1]}`;
		saveBlob(blob, name);
		setStatus(node, `downloaded ${name} — ${mb(blob.size)} · ${describe(tip.info)}`, "#a3d9a5");
	} catch (err) {
		setStatus(node, `save failed: ${err?.message || err}`, "#d08770");
	} finally {
		tip.busy = false;
		tip.save.disabled = false;
		tip.save.textContent = label;
	}
}

async function recheck(node) {
	const tip = node._tip;
	const id = tip?.session || node.properties?.ti_session;
	if (!id) return;
	try {
		const res = await api.fetchApi(`/tinode/vpreview/stat?id=${encodeURIComponent(id)}`);
		const stat = await res.json();
		if (!stat.ok) {
			expired(node);
			return;
		}
		if (!tip.session) show(node, { ...(node.properties.ti_session_info || {}), ...stat });
	} catch (e) { /* server not reachable — leave the panel as it is */ }
}

app.registerExtension({
	name: "tinode.imagePreview",
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
			const v = message?.ti_ipreview;
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
			releaseUnlessShared(this, this._tip?.session);
			return onRemoved?.apply(this, arguments);
		};
	},
});
