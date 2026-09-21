// Agency OS bridge: edit a workflow here, save it back to Agency OS.
//
// Dormant unless ComfyUI is opened inside Agency OS's editor page, which loads
// it in an iframe as  /?agency_origin=<Agency OS origin>.  Then:
//
//   ComfyUI  → parent  {type: "agency:ready"}
//   parent   → ComfyUI {type: "agency:load", workflow, api}   (layout, or API json)
//   ComfyUI  → parent  {type: "agency:save", workflow, api}   ("Save to Agency OS")
//
// Messages are only accepted from, and only sent to, that one origin.

import { app } from "../../scripts/app.js";

const AGENCY = new URLSearchParams(window.location.search).get("agency_origin");

function send(message) {
	window.parent.postMessage(message, AGENCY);
}

let saveToAgency = null;

// Inside Agency OS, Ctrl/Cmd+S means "Save to Agency OS". ComfyUI's own save
// writes only to this editor's private user folder, which Agency OS never
// reads — so it would look saved and not be.
function takeOverCtrlS() {
	window.addEventListener(
		"keydown",
		(event) => {
			if ((event.ctrlKey || event.metaKey) && !event.shiftKey && event.key.toLowerCase() === "s") {
				event.preventDefault();
				event.stopImmediatePropagation();
				saveToAgency?.();
			}
		},
		true,
	);
}

function saveButton() {
	const button = document.createElement("button");
	button.textContent = "Save to Agency OS";
	button.title = "Send this workflow back to the Agency OS Lab (Ctrl+S)";
	button.style.cssText = [
		// Bottom centre: the corners hold ComfyUI's own canvas controls.
		"position:fixed", "left:50%", "bottom:16px", "transform:translateX(-50%)", "z-index:10000",
		"padding:8px 14px", "border-radius:8px", "border:1px solid #6366f1",
		"background:#4f46e5", "color:white", "font:600 13px system-ui,sans-serif",
		"cursor:pointer", "box-shadow:0 4px 16px rgba(0,0,0,.4)",
	].join(";");
	const save = async () => {
		if (button.disabled) return;
		button.disabled = true;
		button.textContent = "Saving…";
		try {
			const { workflow, output } = await app.graphToPrompt();
			send({ type: "agency:save", workflow, api: output });
		} catch (error) {
			send({ type: "agency:error", message: String(error) });
			button.disabled = false;
			button.textContent = "Save to Agency OS";
		}
	};
	saveToAgency = save;
	button.addEventListener("click", save);
	window.addEventListener("message", (event) => {
		if (event.origin !== AGENCY) return;
		if (event.data?.type === "agency:saved" || event.data?.type === "agency:save-failed") {
			button.disabled = false;
			button.textContent = "Save to Agency OS";
		}
	});
	document.body.appendChild(button);
}

app.registerExtension({
	name: "tinode.AgencyBridge",
	async setup() {
		if (!AGENCY || window.parent === window) return;

		window.addEventListener("message", async (event) => {
			if (event.origin !== AGENCY || event.data?.type !== "agency:load") return;
			const { workflow, api, name } = event.data;
			try {
				if (workflow && workflow.nodes) {
					// The saved editor layout: positions, groups, subgraphs intact.
					await app.loadGraphData(workflow, true, true, name || null);
				} else if (api) {
					// Only the API export exists yet: ComfyUI lays it out itself.
					app.loadApiJson(api, name || "workflow.json");
				}
				send({ type: "agency:loaded" });
			} catch (error) {
				send({ type: "agency:error", message: `Could not load the workflow: ${error}` });
			}
		});

		saveButton();
		takeOverCtrlS();
		send({ type: "agency:ready" });
	},
});
