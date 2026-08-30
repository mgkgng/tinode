// Shared chrome for the two ephemeral preview nodes (video_preview.js and
// image_preview.js).
//
// Both mount the same shape of panel — a dark stage, a row of small controls
// under it, a one-line status — and both talk to the same in-RAM store on the
// server. Only the stage differs: a <video> fed a proxy, or an <img> fed one
// frame at a time. Everything around it lived twice and drifted once, which is
// what this file is for.

import { app } from "../../../scripts/app.js";
import { api } from "../../../scripts/api.js";

export const CTRL = "background:#2a2a2a;color:#ddd;border:1px solid #3d3d3d;border-radius:3px;"
	+ "font:11px/1.4 sans-serif;padding:2px 4px;outline:none;";

export const PRIMARY = CTRL + "flex:0 0 auto;cursor:pointer;font-weight:600;"
	+ "background:#3b6ea5;border-color:#4b82bd;color:#fff;padding:3px 9px;";

export const mb = (b) => (b >= 1e9 ? `${(b / 1e9).toFixed(2)} GB`
	: b >= 1e6 ? `${Math.round(b / 1e6)} MB` : `${Math.max(1, Math.round(b / 1e3))} KB`);

export function el(tag, css, text) {
	const e = document.createElement(tag);
	e.style.cssText = css;
	if (text !== undefined) e.textContent = text;
	return e;
}

export function select(options, value, onChange) {
	const s = el("select", CTRL + "flex:1 1 auto;min-width:0;cursor:pointer;");
	for (const [val, label] of options) {
		const o = document.createElement("option");
		o.value = val;
		o.textContent = label;
		s.appendChild(o);
	}
	s.value = value;
	s.addEventListener("change", () => onChange(s.value));
	return s;
}

export function number(value, min, max, title, onChange) {
	const n = el("input", CTRL + "flex:0 0 46px;width:46px;");
	n.type = "number";
	n.min = String(min);
	n.max = String(max);
	n.value = String(value);
	n.title = title;
	n.addEventListener("change", () => onChange(Number(n.value) || min));
	return n;
}

// Download-time settings live in node.properties rather than as Python widgets:
// they only affect the download, and making them widgets would invalidate the
// node's cache and re-run the graph every time one changed.
export function prop(node, key, fallback) {
	node.properties ??= {};
	if (node.properties[key] === undefined) node.properties[key] = fallback;
	return node.properties[key];
}

export function safeName(node, suffix) {
	const t = (node.title || "preview").replace(/[^\w.-]+/g, "_").replace(/^_+|_+$/g, "");
	return `${t || "preview"}${suffix || ""}`;
}

// Hand a response body to the browser's downloader. The only copy that ever
// reaches a filesystem is the one the user chooses to save.
export function saveBlob(blob, filename) {
	const url = URL.createObjectURL(blob);
	const a = document.createElement("a");
	a.href = url;
	a.download = filename;
	document.body.appendChild(a);
	a.click();
	a.remove();
	setTimeout(() => URL.revokeObjectURL(url), 30000);
}

export async function errorText(res, fallback) {
	try {
		return (await res.json()).error || fallback;
	} catch (e) {
		return fallback;
	}
}

export function releaseSession(id) {
	if (!id) return Promise.resolve();
	return api.fetchApi("/tinode/vpreview/drop", {
		method: "POST", headers: { "Content-Type": "application/json" },
		body: JSON.stringify({ id }),
	}).catch(() => {});
}

// Deleting a node frees its clip immediately rather than waiting out the hold —
// unless a copy of the node is still on the graph pointing at the same one.
export function releaseUnlessShared(node, id) {
	if (!id) return;
	const shared = (app.graph?._nodes || []).some(
		(n) => n !== node && (n._tvp?.session === id || n._tip?.session === id));
	if (!shared) releaseSession(id);
}
