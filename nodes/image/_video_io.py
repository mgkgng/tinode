"""ffmpeg helpers shared by Load Video and Save Video.

We shell out to ffmpeg (system binary, or imageio-ffmpeg's bundled one) and pipe
raw rgb24 in and out. That is what gives us explicit control over the colour
range / matrix on encode — the thing that shifts a graded clip's look when a
video round-trips through a decode/encode that mislabels it.

Not a node module (no @register) — just imported by the two video nodes.
"""

from __future__ import annotations

import json
import shutil
import subprocess

VIDEO_EXTS = (".mp4", ".mov", ".mkv", ".webm", ".avi", ".m4v", ".gif", ".mpg", ".mpeg", ".wmv")


def ffmpeg_exe():
	exe = shutil.which("ffmpeg")
	if exe:
		return exe
	try:
		import imageio_ffmpeg  # noqa: PLC0415
		return imageio_ffmpeg.get_ffmpeg_exe()
	except Exception:
		return None


def ffprobe_exe():
	return shutil.which("ffprobe")


def probe(path):
	"""Return {width,height,fps,nb_frames,pix_fmt,color_range,color_space,duration}.

	Zeros/empties on anything ffprobe can't tell us; callers fall back to the
	decoded frame size.
	"""
	info = {"width": 0, "height": 0, "fps": 0.0, "nb_frames": 0, "pix_fmt": "",
			"color_range": "", "color_space": "", "color_transfer": "",
			"color_primaries": "", "duration": 0.0}
	fp = ffprobe_exe()
	if not fp:
		return info
	try:
		out = subprocess.run(
			[fp, "-v", "error", "-select_streams", "v:0", "-show_entries",
			 "stream=width,height,r_frame_rate,nb_frames,pix_fmt,color_range,"
			 "color_space,color_transfer,color_primaries,duration",
			 "-of", "json", path],
			capture_output=True, text=True, check=True).stdout
		st = (json.loads(out).get("streams") or [{}])[0]
		info["width"] = int(st.get("width") or 0)
		info["height"] = int(st.get("height") or 0)
		rate = str(st.get("r_frame_rate") or "0/1")
		num, _, den = rate.partition("/")
		info["fps"] = (float(num) / float(den)) if den and float(den) else 0.0
		nb = str(st.get("nb_frames") or "")
		info["nb_frames"] = int(nb) if nb.isdigit() else 0
		info["pix_fmt"] = st.get("pix_fmt") or ""
		info["color_range"] = st.get("color_range") or ""
		info["color_space"] = st.get("color_space") or ""
		info["color_transfer"] = st.get("color_transfer") or ""
		info["color_primaries"] = st.get("color_primaries") or ""
		info["duration"] = float(st.get("duration") or 0.0)
	except Exception as exc:  # noqa: BLE001
		print(f"[tinode] ffprobe failed on {path!r}: {exc!r}")
	return info


HDR_TRANSFERS = ("smpte2084", "arib-std-b67")   # PQ (HDR10) and HLG

_FILTERS_CACHE = None


def _has_filter(name):
	"""True if this ffmpeg build has the named filter (cached)."""
	global _FILTERS_CACHE
	if _FILTERS_CACHE is None:
		exe = ffmpeg_exe()
		try:
			out = subprocess.run([exe, "-hide_banner", "-filters"],
								 capture_output=True, text=True).stdout
			_FILTERS_CACHE = {ln.split()[1] for ln in out.splitlines()
							  if len(ln.split()) > 1 and ln.startswith(" ")}
		except Exception:
			_FILTERS_CACHE = set()
	return name in _FILTERS_CACHE


def _is_hdr(info):
	return info.get("color_transfer") in HDR_TRANSFERS


# Preferred HDR->SDR: libplacebo with the BT.2446a method — the modern, best
# looking tone-mapper, and it does gamut + curve + range in one GPU pass.
_LIBPLACEBO = ("libplacebo=colorspace=bt709:color_primaries=bt709:"
			   "color_trc=bt709:range=tv:tonemapping=bt.2446a")

# Fallback when libplacebo is absent or has no GPU: a CPU zscale + tonemap chain.
# Reads PQ/wide-gamut correctly instead of as sRGB (the flat, washed-out look).
_TONEMAP_CHAIN = [
	"zscale=t=linear:npl=100", "format=gbrpf32le", "zscale=p=bt709",
	"tonemap=hable:desat=0", "zscale=t=bt709:m=bt709:r=tv",
]


def decode(path, *, force_rate=0.0, skip_first=0, every_nth=1, cap=0, width=0, height=0,
		   full_range=False, tonemap="auto"):
	"""Decode a video to a float IMAGE tensor [N,H,W,3] in 0..1, plus its probe info.

	force_rate>0 resamples to that fps first; then skip_first / every_nth select
	frames; cap>0 limits the count; width&height (both) resize. Matches the order
	VHS applies these in.

	tonemap: "auto" tone-maps HDR (PQ/HLG) sources to SDR and leaves SDR alone;
	"on" forces it; "off" never does. HDR read as SDR is the classic flat,
	desaturated result — auto fixes it without touching normal clips.

	full_range: force the YUV->RGB conversion to treat the source as full range.
	The default (False) trusts the file's own tag, which is faithful for correctly
	tagged clips; flip it when a clip loads washed-out from a wrong/limited tag.
	"""
	import numpy as np  # noqa: PLC0415
	import torch  # noqa: PLC0415

	exe = ffmpeg_exe()
	if exe is None:
		raise RuntimeError("ffmpeg not found (install ffmpeg or imageio-ffmpeg).")

	info = probe(path)
	do_tonemap = tonemap == "on" or (tonemap == "auto" and _is_hdr(info))

	pre = []                                       # fps + frame selection
	if force_rate and force_rate > 0:
		pre.append(f"fps={force_rate}")
	conds = []
	if skip_first and skip_first > 0:
		conds.append(f"gte(n\\,{int(skip_first)})")
	if every_nth and every_nth > 1:
		base = f"n-{int(skip_first)}" if skip_first > 0 else "n"
		conds.append(f"not(mod({base}\\,{int(every_nth)}))")
	if conds:
		pre.append("select=" + "*".join(conds))

	post = []                                      # resize, after any tonemap
	# Force full range only matters for a mis-tagged SDR clip, never for tonemapped HDR.
	if full_range and not do_tonemap:
		post.append("scale=in_range=full:out_range=full")
	if width and height:
		post.append(f"scale={int(width)}:{int(height)}")

	def build(tm):
		# tm: "placebo" | "zscale" | None
		fl = list(pre)
		if tm == "placebo":
			fl.append(_LIBPLACEBO)
		elif tm == "zscale":
			fl += _TONEMAP_CHAIN
		fl += post
		cmd = [exe, "-v", "error", "-i", path]
		if fl:
			cmd += ["-vf", ",".join(fl)]
		cmd += ["-vsync", "0"]                     # keep exactly the frames select() passed
		if cap and cap > 0:
			cmd += ["-frames:v", str(int(cap))]
		cmd += ["-f", "rawvideo", "-pix_fmt", "rgb24", "pipe:1"]
		return cmd

	if do_tonemap and _has_filter("libplacebo"):
		attempts = ["placebo", "zscale"]           # libplacebo is best; zscale is the fallback
	elif do_tonemap:
		attempts = ["zscale"]
	else:
		attempts = [None]

	proc = None
	for i, tm in enumerate(attempts):
		proc = subprocess.run(build(tm), capture_output=True)
		if proc.returncode == 0:
			break
		if i < len(attempts) - 1:
			print(f"[tinode] Load Video: {tm} tonemap failed, trying {attempts[i + 1]}…")
	if proc.returncode != 0:
		raise RuntimeError("ffmpeg decode failed:\n" + proc.stderr.decode("utf-8", "replace")[-800:])

	W = int(width) if (width and height) else info["width"]
	H = int(height) if (width and height) else info["height"]
	raw = proc.stdout
	if not W or not H:
		raise RuntimeError("could not determine video dimensions (ffprobe unavailable?).")
	frame_bytes = W * H * 3
	n = len(raw) // frame_bytes
	if n == 0:
		raise RuntimeError("decoded 0 frames — check the file, skip/every-nth, or cap.")
	arr = np.frombuffer(raw[:n * frame_bytes], dtype=np.uint8).reshape(n, H, W, 3)
	return torch.from_numpy(arr.copy()).float() / 255.0, info


def encode(frames, out_path, *, fps=8.0, codec="libx264", crf=17, pix_fmt="yuv420p",
		   color_range="tv", colorspace="bt709"):
	"""Encode an IMAGE tensor [N,H,W,C] to a video file via ffmpeg.

	color_range / colorspace are tagged AND applied so the file's look matches how
	players interpret it — this is the knob that keeps grading intact. Pass
	"" / "unspecified" to leave a flag off.
	"""
	import numpy as np  # noqa: PLC0415

	exe = ffmpeg_exe()
	if exe is None:
		raise RuntimeError("ffmpeg not found (install ffmpeg or imageio-ffmpeg).")

	N, H, W, C = frames.shape
	data = (frames[..., :3].clamp(0, 1).cpu().numpy() * 255.0 + 0.5).astype(np.uint8).tobytes()

	cmd = [exe, "-v", "error", "-y",
		   "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}", "-r", str(fps), "-i", "pipe:0"]
	if codec == "libx264":
		cmd += ["-c:v", "libx264", "-preset", "medium", "-crf", str(int(crf)), "-pix_fmt", pix_fmt]
	elif codec == "libvpx-vp9":
		cmd += ["-c:v", "libvpx-vp9", "-crf", str(int(crf)), "-b:v", "0", "-pix_fmt", pix_fmt]
	else:
		raise RuntimeError(f"unknown codec {codec!r}")

	if colorspace and colorspace != "unspecified":
		cmd += ["-colorspace", colorspace, "-color_primaries", colorspace, "-color_trc", colorspace]
	if color_range and color_range != "unspecified":
		cmd += ["-color_range", color_range]

	cmd += ["-movflags", "+faststart", out_path]
	proc = subprocess.run(cmd, input=data, capture_output=True)
	if proc.returncode != 0:
		raise RuntimeError("ffmpeg encode failed:\n" + proc.stderr.decode("utf-8", "replace")[-800:])
	return out_path
