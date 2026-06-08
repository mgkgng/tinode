// tinode frontend extensions.
//
// WEB_DIRECTORY in __init__.py points here. ComfyUI loads every .js in this
// folder in the browser. Use this for custom widgets, node colors, previews,
// or right-click menu items. The stub below is a no-op that proves the web
// pipeline is wired — extend or delete.
//
// Docs: https://docs.comfy.org/custom-nodes/js/javascript_overview

import { app } from "../../scripts/app.js";

app.registerExtension({
	name: "tinode.core",
	async setup() {
		console.log("[tinode] web extensions loaded");
	},
});
