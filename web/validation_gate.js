// Review panel for "Validation Gate (ti)".
//
// While the node blocks the running workflow, the backend sends a "tinode.gate"
// message; we pop a floating panel with the preview and Approve / Reject. The
// click POSTs the decision back to /tinode/gate, which unblocks the node and the
// graph continues (or stops, on Reject). "tinode.gate_done" tears the panel down
// if it's still up (e.g. the run was cancelled).

import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";
import { urlFor } from "./lib/editor.js";

let panel = null;
let currentToken = null;

function teardown() {
	if (panel) { panel.remove(); panel = null; }
	currentToken = null;
	window.removeEventListener("keydown", onKey);
}

function decide(decision) {
	const token = currentToken;
	teardown();
	if (!token) return;
	api.fetchApi("/tinode/gate", {
		method: "POST",
		headers: { "Content-Type": "application/json" },
		body: JSON.stringify({ token, decision }),
	}).catch((e) => console.error("[tinode] gate POST failed", e));
}

function onKey(e) {
	if (!panel) return;
	if (e.key === "Enter") { e.preventDefault(); decide("approve"); }
	else if (e.key === "Escape") { e.preventDefault(); decide("reject"); }
}

function showPanel(detail) {
	teardown();
	currentToken = detail.token;

	panel = document.createElement("div");
	panel.style.cssText = [
		"position:fixed", "z-index:10000", "top:50%", "left:50%",
		"transform:translate(-50%,-50%)", "background:#1e1e1e",
		"border:1px solid #4aa3ff", "border-radius:8px", "padding:14px",
		"box-shadow:0 8px 40px rgba(0,0,0,0.6)", "max-width:min(90vw,900px)",
		"max-height:90vh", "display:flex", "flex-direction:column", "gap:10px",
		"font-family:sans-serif", "color:#e6e6e6",
	].join(";");

	const title = document.createElement("div");
	title.textContent = detail.title || "Approve to continue?";
	title.style.cssText = "font-size:14px;font-weight:600;text-align:center;";
	panel.appendChild(title);

	if (detail.image) {
		const img = document.createElement("img");
		img.src = urlFor(detail.image);
		img.style.cssText = "max-width:100%;max-height:70vh;border-radius:4px;object-fit:contain;";
		panel.appendChild(img);
	}

	const row = document.createElement("div");
	row.style.cssText = "display:flex;gap:10px;justify-content:center;";
	const mk = (label, decision, bg) => {
		const b = document.createElement("button");
		b.textContent = label;
		b.style.cssText = `flex:1;padding:8px 16px;border:none;border-radius:5px;`
			+ `background:${bg};color:#fff;font-size:13px;font-weight:600;cursor:pointer;`;
		b.onclick = () => decide(decision);
		return b;
	};
	row.appendChild(mk("Reject (Esc)", "reject", "#a33"));
	row.appendChild(mk("Approve (Enter)", "approve", "#2a7"));
	panel.appendChild(row);

	document.body.appendChild(panel);
	window.addEventListener("keydown", onKey);
}

app.registerExtension({
	name: "tinode.validationGate",
	async setup() {
		api.addEventListener("tinode.gate", (e) => showPanel(e.detail));
		api.addEventListener("tinode.gate_done", (e) => {
			if (currentToken && e.detail?.token === currentToken) teardown();
		});
	},
});
