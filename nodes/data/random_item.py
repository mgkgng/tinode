"""Random Item · List — pick one line out of a list, at random.

Point it at a file (or paste a list in) and it hands back one entry per run. The
seed is a normal ComfyUI seed widget, so `control_after_generate: randomize`
rolls a new one every queue and a fixed seed reproduces a pick exactly — the
same contract as a sampler's.

The parser is deliberately forgiving, because a list you actually maintain is
never clean. All of these are one item:

    1. **Ball** — A portable spherical object.
    12) Globe - A spherical representation of a world
    - Pearl: a naturally formed smooth sphere
    Marble

Leading numbers, bullets and `**bold**` are stripped; the first em dash, en
dash, spaced hyphen or colon splits NAME from DESCRIPTION. Markdown headings
(`#`), blockquotes (`>`) and blank lines are skipped, so a list file can carry
its own title and notes without them turning into entries. A name containing a
hyphen with no spaces around it — Yo-yo, O-ring, Pom-pom — stays whole, which is
why the split requires the spaces.
"""

from __future__ import annotations

import os
import random
import re

from ...base import TiNode, first
from ...registry import register

_NUMBER = re.compile(r"^\s*\d+\s*[.)\]]\s*")
_BULLET = re.compile(r"^\s*[-*+•]\s+")
_BOLD = re.compile(r"\*\*(.+?)\*\*")
_ITALIC = re.compile(r"(?<!\*)\*(?!\*)(.+?)(?<!\*)\*(?!\*)")
# NAME — DESC.  Spaces around the separator are required so Yo-yo survives.
_SPLIT = re.compile(r"\s+[—–]\s+|\s+-{1,2}\s+|\s*:\s+")

FORMATS = ["name", "description", "name — description", "raw line"]


def parse_list(text):
	"""Every item in `text`, as {index, name, description, raw}.

	`index` is 1-based over the items KEPT, not the number printed on the line:
	a list with a gap in its numbering still hands back a usable position, and
	the original text is preserved in `raw` if the printed number mattered.
	"""
	items = []
	for line in str(text or "").splitlines():
		raw = line.strip()
		if not raw or raw[0] in "#>":
			continue
		body = _BULLET.sub("", _NUMBER.sub("", raw))
		body = _BOLD.sub(r"\1", body)
		body = _ITALIC.sub(r"\1", body).strip()
		if not body:
			continue
		parts = _SPLIT.split(body, maxsplit=1)
		name = parts[0].strip().rstrip(".").strip()
		desc = parts[1].strip() if len(parts) > 1 else ""
		if not name:
			continue
		items.append({"index": len(items) + 1, "name": name,
					  "description": desc, "raw": raw})
	return items


def format_item(item, style):
	"""One item rendered the way the `format` widget asks for."""
	if style == "description":
		return item["description"] or item["name"]
	if style == "raw line":
		return item["raw"]
	if style == "name — description" and item["description"]:
		return f"{item['name']} — {item['description']}"
	return item["name"]


def stream_seed(items, seed):
	"""A seed unique to (this list, this number).

	Several of these nodes normally run off ONE seed — object, material,
	lighting, environment, style — and seeding them all with the same integer
	makes any two lists of equal length draw the same index every time. Mixing
	the list's own identity in decorrelates them while staying exactly
	reproducible: the same seed on the same list is always the same pick.
	"""
	return f"{int(seed)}|{len(items)}|{items[0]['name']}|{items[-1]['name']}"


def choose(items, seed, count=1, unique=True):
	"""`count` items picked with `seed`. Unique draws without replacement.

	A unique draw of more than the list holds gives the whole list shuffled
	rather than raising — asking for 300 of 250 things plainly means "all of
	them, in some order".
	"""
	if not items:
		return []
	rng = random.Random(stream_seed(items, seed))
	n = max(1, int(count))
	if not unique:
		return [items[rng.randrange(len(items))] for _ in range(n)]
	if n >= len(items):
		out = list(items)
		rng.shuffle(out)
		return out
	return rng.sample(items, n)


def resolve_path(file_path):
	"""Absolute path to a list file, or None.

	Tried in order: as given, ~-expanded, then relative to ComfyUI's input/ —
	so a folder symlinked into input/ can be addressed the short way
	(`013-MASK_TRACK/objects.md`) exactly like any other input.
	"""
	path = str(file_path or "").strip()
	if not path:
		return None
	cand = os.path.expanduser(path)
	if os.path.isfile(cand):
		return cand
	if not os.path.isabs(cand):
		try:
			import folder_paths  # noqa: PLC0415

			under = os.path.join(folder_paths.get_input_directory(), cand)
			if os.path.isfile(under):
				return under
		except Exception:  # noqa: BLE001 — outside ComfyUI, absolute paths only
			pass
	return None


def read_source(source, file_path, text):
	"""(list text, where it came from) — from a file or the inline widget.

	The second value exists so a failure downstream can name what was actually
	read. "the list is empty" on its own sends you hunting; "read 0 bytes from
	the inline text box, while file_path is set" tells you the source switch is
	on the wrong setting, which is the mistake this node invites.
	"""
	inline = str(text or "")
	path_given = str(file_path or "").strip()

	if str(source) == "text":
		where = f"the inline text box ({len(inline)} bytes)"
		if not inline.strip() and path_given:
			raise RuntimeError(
				f"Random Item: source is `text` but the text box is empty — "
				f"while file_path is set to {path_given!r}. Switch source to "
				"`file`, or paste the list into the box.")
		return inline, where

	if not path_given:
		raise RuntimeError(
			"Random Item: source is `file` but file_path is empty. Point it at a "
			"list file, or switch source to `text` and paste the list in."
			+ (" (There IS a list in the text box.)" if inline.strip() else ""))
	path = resolve_path(path_given)
	if path is None:
		import glob  # noqa: PLC0415

		hint = ""
		if os.path.isdir(os.path.expanduser(path_given)):
			found = sorted(os.path.basename(p) for p in
						   glob.glob(os.path.join(os.path.expanduser(path_given), "*.*")))
			hint = (f" That is a FOLDER — name a file inside it"
					f"{': ' + ', '.join(found[:8]) if found else ''}.")
		raise RuntimeError(
			f"Random Item: no such file — {path_given!r}.{hint} Give an absolute "
			"path, or one relative to ComfyUI's input/ folder.")
	with open(path, encoding="utf-8") as fh:
		body = fh.read()
	return body, f"{path} ({len(body)} bytes)"


@register
class RandomItem(TiNode):
	DISPLAY_NAME = "Random Item · List (ti)"
	CATEGORY = "tinode/data"

	@classmethod
	def INPUT_TYPES(cls):
		return {
			"required": {
				"seed": ("INT", {"default": 0, "min": 0, "max": 0xFFFFFFFFFFFFFFFF,
					"control_after_generate": True, "tooltip":
					"Set it to randomize and every queue draws again; fix it and "
					"the same pick comes back."}),
				"source": (["file", "text"], {"default": "file", "tooltip":
					"Read the list from a file on disk, or from the `text` box."}),
				"file_path": ("STRING", {"default": "", "tooltip":
					"Path to the list. ~ is expanded."}),
			},
			"optional": {
				"text": ("STRING", {"default": "", "multiline": True, "tooltip":
					"The list itself, when source is `text`. One item per line."}),
				"count": ("INT", {"default": 1, "min": 1, "max": 999, "step": 1,
					"tooltip": "How many to draw. More than 1 joins them with "
							   "newlines in the `text` output; `name` and the "
							   "other single outputs describe the FIRST."}),
				"unique": ("BOOLEAN", {"default": True, "tooltip":
					"On: no repeats within one draw. Off: each pick is "
					"independent, so the same item can come up twice."}),
				"format": (FORMATS, {"default": "name", "tooltip":
					"What the `text` output carries."}),
			},
		}

	RETURN_TYPES = ("STRING", "STRING", "STRING", "INT", "INT")
	RETURN_NAMES = ("text", "name", "description", "index", "list_size")
	OUTPUT_TOOLTIPS = (
		"The draw, formatted — one line per item when count > 1.",
		"The first pick's name on its own.",
		"The first pick's description on its own.",
		"The first pick's 1-based position in the list.",
		"How many items the list holds.",
	)
	FUNCTION = "execute"

	@classmethod
	def IS_CHANGED(cls, seed=0, source="file", file_path="", **_kw):
		# The seed already forces a re-run; the file's mtime catches a list
		# edited between queues while the seed stayed put.
		path = resolve_path(file_path)
		stamp = os.path.getmtime(path) if path else 0
		return f"{seed}:{source}:{path}:{stamp}"

	def execute(self, seed=0, source="file", file_path="", text="", count=1,
				unique=True, format="name"):
		body, where = read_source(first(source, "file"), first(file_path, ""),
								  first(text, ""))
		items = parse_list(body)
		if not items:
			lines = body.splitlines()
			skipped = sum(1 for ln in lines if ln.strip() and ln.strip()[0] in "#>")
			raise RuntimeError(
				f"Random Item: no items found in {where}. "
				f"{len(lines)} line(s), {skipped} of them heading/quote, the rest "
				"blank. A list needs one item per line, e.g. "
				"`1. **Ball** - A portable spherical object.` or just `Ball`.")
		style = str(first(format, "name"))
		picks = choose(items, int(first(seed, 0)), int(first(count, 1)),
					   bool(first(unique, True)))
		body = "\n".join(format_item(p, style) for p in picks)
		head = picks[0]
		print(f"[tinode] Random Item: {len(picks)} of {len(items)} "
			  f"(seed {int(first(seed, 0))}) -> {head['name']!r}")
		return (body, head["name"], head["description"], head["index"], len(items))
