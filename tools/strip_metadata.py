#!/usr/bin/env python3
"""Strip ComfyUI's embedded workflow out of the files it saves.

ComfyUI writes the whole graph into everything it exports, and the frontend reads
it back when you drag the file onto the canvas — which is why sharing a render
also hands over the workflow that made it. Two places it hides:

  PNG    tEXt chunks named `prompt` and `workflow` (nodes.py, SaveImage).
  video  container-level metadata tags of the same names (nodes_video.py,
         SaveVideo). Routinely far bigger — 174 KB of workflow in one measured
         mp4, against 27 KB in a PNG from the same session.

Neither is re-encoded. PNG is chunk surgery: every chunk but the text/EXIF ones
is copied byte for byte in its original order, so IDAT is untouched and the
pixels are bit-identical. Video is an ffmpeg remux with `-c copy`, so the encoded
bitstream is copied packet for packet — the picture and sound that come out are
the ones that went in. Modification time and permissions are preserved, so a
folder does not resort itself.

    strip_metadata.py RENDER.png                 one file
    strip_metadata.py clip.mp4                   or one video
    strip_metadata.py ~/Downloads                every .png/.mp4/... under it
    strip_metadata.py a.png shots/ b.mp4         any mix of both
    strip_metadata.py ~/Downloads --check        report only, change nothing
    strip_metadata.py shots/ --flat              top level only, do not recurse

Writes through a temporary file in the same directory and renames over the
original, so an interrupted run cannot leave a half-written file behind.

Video needs ffmpeg on PATH; PNG needs nothing but the standard library.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import struct
import subprocess
import sys

MAGIC = b"\x89PNG\r\n\x1a\n"

# The ancillary chunks worth removing. tEXt/zTXt/iTXt are where ComfyUI (and
# most other tools) put text; eXIf is camera/EXIF metadata. All four are
# position-independent, so dropping them cannot invalidate the stream.
STRIP = {b"tEXt", b"zTXt", b"iTXt", b"eXIf"}


# Containers ComfyUI (and VHS) write. Everything else is left alone.
VIDEO_EXTS = {".mp4", ".mov", ".mkv", ".webm", ".m4v", ".avi"}
IMAGE_EXTS = {".png"}

# Tags that describe the container rather than the run. Everything else in a
# video's global metadata is treated as something to remove.
BENIGN_TAGS = {"major_brand", "minor_version", "compatible_brands", "encoder"}


class Unreadable(Exception):
	"""Raised for anything this cannot safely rewrite."""


# Kept as an alias: the PNG-only version of this script raised NotPng.
NotPng = Unreadable


def ffmpeg_exe():
	"""ffmpeg from PATH, or from the usual places when PATH has been stripped."""
	found = shutil.which("ffmpeg")
	if found:
		return found
	for d in ("/usr/bin", "/usr/local/bin", "/opt/homebrew/bin", "/bin"):
		cand = os.path.join(d, "ffmpeg")
		if os.access(cand, os.X_OK):
			return cand
	return None


def read_chunks(path):
	"""Yield (kind, whole_chunk_bytes) for every chunk, in file order.

	The chunk is handed back with its length header and CRC intact so a caller
	can write it out again without recomputing anything.
	"""
	with open(path, "rb") as fh:
		if fh.read(8) != MAGIC:
			raise Unreadable("not a PNG")
		while True:
			head = fh.read(8)
			if len(head) < 8:
				raise Unreadable("truncated PNG (no IEND)")
			length, kind = struct.unpack(">I4s", head)
			body = fh.read(length)
			crc = fh.read(4)
			if len(body) != length or len(crc) != 4:
				raise Unreadable("truncated PNG (chunk runs past the end)")
			yield kind, head + body + crc
			if kind == b"IEND":
				return


def inspect(path):
	"""Return (kept_chunks, [(kind, key, size), ...]) without writing anything."""
	kept, found = [], []
	for kind, raw in read_chunks(path):
		if kind in STRIP:
			# The key is the NUL-terminated first field of a text chunk; eXIf
			# has none, which is why this can come back empty.
			key = raw[8:].split(b"\x00", 1)[0].decode("latin1", "replace")[:40]
			found.append((kind.decode("latin1"), key, len(raw)))
		else:
			kept.append(raw)
	return kept, found


def strip_png(path, check=False):
	"""Remove the metadata chunks from one PNG. Returns (changed, bytes_removed).

	`changed` is False for a file that was already clean — which is then left
	completely alone, not rewritten with identical content.
	"""
	kept, found = inspect(path)
	removed = sum(n for _, _, n in found)
	if not found or check:
		return bool(found), removed

	tmp = path + ".stripping"
	try:
		with open(tmp, "wb") as fh:
			fh.write(MAGIC)
			for raw in kept:
				fh.write(raw)
		_adopt(path, tmp)
	except BaseException:
		_discard(tmp)
		raise
	return True, removed


def video_tags(path):
	"""The container's global metadata tags, or None when ffprobe is unavailable."""
	exe = ffmpeg_exe()
	probe = os.path.join(os.path.dirname(exe), "ffprobe") if exe else None
	if not probe or not os.access(probe, os.X_OK):
		probe = shutil.which("ffprobe")
	if not probe:
		return None
	out = subprocess.run([probe, "-v", "error", "-show_entries", "format_tags",
						  "-of", "json", path], capture_output=True, text=True)
	if out.returncode != 0:
		raise Unreadable("ffprobe could not read it")
	try:
		return json.loads(out.stdout).get("format", {}).get("tags", {}) or {}
	except json.JSONDecodeError:
		return {}


def strip_video(path, check=False):
	"""Drop a container's global metadata by remuxing. Returns (changed, bytes_removed).

	`-c copy` means the encoded streams are copied packet for packet: no decode,
	no re-encode, no quality change, and it runs at disk speed rather than encode
	speed. `-map 0` keeps every stream (video, audio, subtitles) instead of
	ffmpeg's default of one each.
	"""
	exe = ffmpeg_exe()
	if exe is None:
		raise Unreadable("ffmpeg not found (needed for video)")

	tags = video_tags(path)
	if tags is not None:
		interesting = {k: v for k, v in tags.items() if k.lower() not in BENIGN_TAGS}
		if not interesting:
			return False, 0
		removed = sum(len(str(k)) + len(str(v)) for k, v in interesting.items())
	else:
		# No ffprobe: we cannot tell whether there is anything to remove, so do
		# the work and report the size difference rather than claim it was clean.
		removed = 0
	if check:
		return True, removed

	before = os.path.getsize(path)
	tmp = path + ".stripping" + os.path.splitext(path)[1]
	try:
		cmd = [exe, "-v", "error", "-y", "-i", path, "-map", "0", "-c", "copy",
			   "-map_metadata", "-1", "-map_chapters", "-1"]
		if os.path.splitext(path)[1].lower() in (".mp4", ".mov", ".m4v"):
			cmd += ["-movflags", "+faststart"]
		proc = subprocess.run(cmd + [tmp], capture_output=True)
		if proc.returncode != 0:
			raise Unreadable("ffmpeg remux failed: "
							 + proc.stderr.decode("utf-8", "replace").strip()[-200:])
		_adopt(path, tmp)
	except BaseException:
		_discard(tmp)
		raise
	return True, max(0, before - os.path.getsize(path)) or removed


def _adopt(path, tmp):
	"""Give the replacement the original's mode and mtime, then swap it in."""
	st = os.stat(path)
	os.chmod(tmp, st.st_mode & 0o7777)
	os.utime(tmp, (st.st_atime, st.st_mtime))
	os.replace(tmp, path)                  # atomic: no half-written file on a crash


def _discard(tmp):
	if os.path.exists(tmp):
		try:
			os.remove(tmp)
		except OSError:
			pass


def strip_file(path, check=False):
	"""Clean one file, picking the right method from its extension."""
	ext = os.path.splitext(path)[1].lower()
	if ext in VIDEO_EXTS:
		return strip_video(path, check)
	return strip_png(path, check)


def collect(targets, recurse=True):
	"""Expand files and directories into a de-duplicated list of PNG paths.

	A file named on the command line is taken at its word whatever its extension
	— strip_file will say so if it cannot handle it. A file found by walking a
	directory has to look like one of the types we handle, because a folder is
	full of things that are not.
	"""
	wanted = IMAGE_EXTS | VIDEO_EXTS
	out, seen = [], set()

	def add(p):
		real = os.path.realpath(p)
		if real not in seen:
			seen.add(real)
			out.append(p)

	for t in targets:
		if os.path.isdir(t):
			if recurse:
				# followlinks=False (the default) so a symlink loop cannot hang us.
				for root, dirs, files in os.walk(t):
					dirs.sort()
					for f in sorted(files):
						if os.path.splitext(f)[1].lower() in wanted:
							add(os.path.join(root, f))
			else:
				for f in sorted(os.listdir(t)):
					p = os.path.join(t, f)
					if os.path.isfile(p) and os.path.splitext(f)[1].lower() in wanted:
						add(p)
		elif os.path.exists(t):
			add(t)
		else:
			print(f"  MISSING {t}", file=sys.stderr)
	return out


def display(path):
	"""The shorter of the relative and absolute path.

	Plain relpath turns a file in /tmp into ../../../../tmp/... when you run this
	from a project directory, which is longer AND harder to read than the
	absolute path it replaced.
	"""
	try:
		rel = os.path.relpath(path)
	except ValueError:                     # different drive on Windows
		return path
	return rel if len(rel) <= len(path) else path


def human(n):
	return f"{n / 1e6:.1f} MB" if n >= 1e6 else f"{n / 1e3:.0f} KB" if n >= 1e3 else f"{n} B"


def main(argv=None):
	ap = argparse.ArgumentParser(
		prog="strip_metadata.py",
		description="Remove ComfyUI workflow / prompt metadata from PNG and video files.",
		epilog="Nothing is re-encoded: PNG chunks and video packets are copied verbatim.")
	ap.add_argument("targets", nargs="+", metavar="PATH",
					help="PNG/MP4/MOV/MKV/WEBM files and/or directories to clean")
	ap.add_argument("-n", "--check", action="store_true",
					help="report what would be removed, change nothing")
	ap.add_argument("--flat", action="store_true",
					help="do not descend into subdirectories")
	ap.add_argument("-q", "--quiet", action="store_true",
					help="only print files that carried metadata, plus the summary")
	args = ap.parse_args(argv)

	paths = collect(args.targets, recurse=not args.flat)
	if not paths:
		print("No PNG or video files found.")
		return 1

	hits = total = failed = 0
	for path in paths:
		name = display(path)
		try:
			changed, removed = strip_file(path, args.check)
		except Unreadable as exc:
			print(f"  SKIP    {name}: {exc}", file=sys.stderr)
			failed += 1
			continue
		except OSError as exc:
			print(f"  ERROR   {name}: {exc}", file=sys.stderr)
			failed += 1
			continue

		if changed:
			hits += 1
			total += removed
			verb = "carries" if args.check else "cleaned"
			print(f"  {verb:7s} {name}  ({human(removed)})")
		elif not args.quiet:
			print(f"  clean   {name}")

	verb = "carry" if args.check else "had"
	print(f"\n{hits} of {len(paths)} file(s) {verb} embedded metadata"
		  f"{f' — {human(total)}' if total else ''}"
		  f"{', nothing was written' if args.check and hits else ''}.")
	if failed:
		print(f"{failed} file(s) could not be read.", file=sys.stderr)
	return 1 if failed else 0


if __name__ == "__main__":
	sys.exit(main())
