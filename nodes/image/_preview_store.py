"""In-memory clip store and HTTP routes behind Video Preview.

Nothing here touches the disk in any place ComfyUI looks. A run parks its frames
in a module-global dict keyed by a random id, the browser streams a small proxy
out of RAM to play it, and CREATE VIDEO encodes a fresh file — also in RAM —
straight into the download. Restart ComfyUI and every trace is gone: output/ and
temp/ stay exactly as clean as they were, and nothing lands in the queue history.

Bounded on purpose. Raw frames are big (a 1080p frame is 6 MB at 8-bit), so the
store enforces a total byte ceiling and a per-session TTL, evicting the
least-recently-touched clip when either is passed. Without that, a long session
would quietly eat the box's RAM instead of its disk, which is not an improvement.
The default ceiling is 8 GB across every held clip; TINODE_PREVIEW_RAM_MB
overrides it, and is read once at import so a change needs a restart.

The master is kept at the precision the source actually carries (see
`pack_frames`), so a download can be bit-exact — nothing is quantised on the way
in and re-expanded on the way out.

The two mp4/mov exports go through a private temp dir from tempfile, deleted
before the response is sent: those muxers seek back over their own output, and a
fragmented stream handed to an editor is a worse bug than a file that lives for
half a second outside ComfyUI's tree. mkv, webm and the PNG zip never leave RAM.

Not a node module (no @register) — imported by video_preview.py.
"""

from __future__ import annotations

import io
import os
import re
import shutil
import subprocess
import tempfile
import threading
import time
import uuid
import zipfile

from ._video_io import colour_flags, ffmpeg_exe

# How long an untouched clip survives, and the ceiling across all of them.
DEFAULT_HOLD_S = 30 * 60
MAX_BYTES = int(float(os.environ.get("TINODE_PREVIEW_RAM_MB", "8192")) * 1024 * 1024)

_SESSIONS: dict[str, dict] = {}
_LOCK = threading.RLock()


# --------------------------------------------------------------------------
# frame packing
# --------------------------------------------------------------------------

def probe_depth(imgs, sample=8):
	"""8 or 16 — the bit depth this clip actually needs to survive a round trip.

	An IMAGE arrives as float32 0..1 whatever produced it. Storing that as uint8
	is free and exact for the usual 8-bit clip, but silently truncates a 10- or
	16-bit graded master. So: check whether the values already sit on the 1/255
	grid. If they do, 8 bits loses nothing; if they don't, keep 16.

	Sampled rather than exhaustive — a clip whose frames are all 8-bit except one
	does not exist, and a full pass over a long 4K batch costs seconds.
	"""
	import torch  # noqa: PLC0415

	n = int(imgs.shape[0])
	idx = sorted({int(round(i * (n - 1) / max(1, sample - 1))) for i in range(min(sample, n))})
	for i in idx:
		x = imgs[i, ..., :3].detach().float().clamp(0, 1)
		if ((x * 255.0).round() / 255.0 - x).abs().max().item() > 1.0 / 65535.0:
			return 16
	return 8


def pack_frames(imgs, precision="auto"):
	"""Flatten an IMAGE tensor to contiguous raw RGB bytes plus its geometry.

	Returns (raw_bytes, n, h, w, depth). Bytes rather than an ndarray because the
	bytes are what ffmpeg's stdin and zipfile both want — keeping the array too
	would double the footprint for no gain, and .tobytes() later would copy the
	whole master at the worst moment.
	"""
	import numpy as np  # noqa: PLC0415

	x = imgs if imgs.dim() == 4 else imgs.unsqueeze(0)
	x = x[..., :3].detach().float().clamp(0, 1).cpu()
	n, h, w = int(x.shape[0]), int(x.shape[1]), int(x.shape[2])

	depth = probe_depth(x) if precision == "auto" else (16 if str(precision).startswith("16") else 8)
	if depth == 16:
		arr = (x.numpy() * 65535.0 + 0.5).astype("<u2")
	else:
		arr = (x.numpy() * 255.0 + 0.5).astype(np.uint8)
	return np.ascontiguousarray(arr).tobytes(), n, h, w, depth


def pack_audio(audio):
	"""Flatten an AUDIO input to raw interleaved float32 PCM, or None.

	ComfyUI's AUDIO is {"waveform": [B, C, N] float, "sample_rate": int}. The
	samples are kept exactly as they arrive — float32, no resample, no dither, no
	integer conversion — so a lossless container can carry what the graph made.

	Only the first item of the batch is taken: a batch of waveforms is several
	takes, and there is one picture here to lay them against.
	"""
	if not isinstance(audio, dict):
		return None
	import numpy as np  # noqa: PLC0415

	w = audio.get("waveform")
	rate = int(audio.get("sample_rate") or 0)
	if w is None or rate <= 0:
		return None
	w = w.detach().cpu().float()
	if w.dim() == 3:
		w = w[0]
	elif w.dim() == 1:
		w = w.unsqueeze(0)
	if w.dim() != 2 or w.shape[0] == 0 or w.shape[1] == 0:
		return None            # an empty track is the same as no track
	ch, n = int(w.shape[0]), int(w.shape[1])
	# [C, N] -> [N, C] so the flat buffer is interleaved, which is what f32le means.
	raw = np.ascontiguousarray(w.transpose(0, 1).numpy().astype(np.float32)).tobytes()
	return {"raw": raw, "rate": rate, "channels": ch, "samples": n,
			"seconds": n / float(rate)}


def _src_pix_fmt(depth):
	return "rgb48le" if depth == 16 else "rgb24"


# --------------------------------------------------------------------------
# store
# --------------------------------------------------------------------------

def _total_locked():
	return sum(s["bytes"] for s in _SESSIONS.values())


def _sweep_locked(protect=None):
	"""Drop expired clips, then the least-recently-touched until under the cap.

	`protect` is never evicted. A clip must survive the sweep that its own
	arrival triggers: without that, putting a large clip into a nearly-full
	store can evict the clip being put — it is the newest, but it is also the
	one making the store overflow — and the caller gets back an id that is
	already dead.
	"""
	now = time.time()
	for sid in [k for k, s in _SESSIONS.items()
				if k != protect and now - s["touched"] > s["hold"]]:
		_SESSIONS.pop(sid, None)
	while _total_locked() > MAX_BYTES:
		alive = [kv for kv in _SESSIONS.items() if kv[0] != protect]
		if not alive:
			break
		oldest = min(alive, key=lambda kv: kv[1]["touched"])[0]
		print(f"[tinode] Video Preview: RAM ceiling reached, releasing {oldest}")
		_SESSIONS.pop(oldest, None)


def put(raw, n, h, w, depth, fps, *, owner="", hold=DEFAULT_HOLD_S, proxy=b"",
		proxy_mime="video/mp4", audio=None):
	"""Park a clip and return its session id, replacing this node's previous one.

	Keyed by owner (the node's unique id) so re-queuing a graph swaps the clip
	instead of stacking copies of it — the common way an in-RAM cache turns into
	a leak is holding every run, not holding one.
	"""
	size = len(raw) + len(proxy) + (len(audio["raw"]) if audio else 0)
	if size > MAX_BYTES:
		# One clip on its own exceeds the ceiling. Refuse here, with the numbers,
		# rather than let the frontend puzzle over a session that was never alive.
		raise RuntimeError(
			f"clip needs {size / 1e6:.0f} MB but the preview ceiling is "
			f"{MAX_BYTES / 1e6:.0f} MB — raise TINODE_PREVIEW_RAM_MB, drop the "
			f"precision to 8-bit, or preview fewer frames."
		)
	sid = uuid.uuid4().hex[:16]
	now = time.time()
	with _LOCK:
		for old in [k for k, s in _SESSIONS.items() if owner and s["owner"] == owner]:
			_SESSIONS.pop(old, None)
		_SESSIONS[sid] = {
			"id": sid, "owner": str(owner), "raw": raw, "n": n, "h": h, "w": w,
			"depth": depth, "fps": float(fps), "proxy": proxy, "proxy_mime": proxy_mime,
			"audio": audio, "bytes": size, "created": now, "touched": now,
			"hold": float(hold),
		}
		_sweep_locked(protect=sid)
	return sid


def get(sid):
	with _LOCK:
		s = _SESSIONS.get(sid)
		if s is not None:
			s["touched"] = time.time()
		return s


def drop(sid):
	with _LOCK:
		return _SESSIONS.pop(sid, None) is not None


def stats():
	with _LOCK:
		return {"sessions": len(_SESSIONS), "bytes": _total_locked(), "cap": MAX_BYTES}


# --------------------------------------------------------------------------
# encoding — all of it fed from RAM, none of it written where ComfyUI looks
# --------------------------------------------------------------------------

def _run(cmd, data, aux=None):
	"""Run ffmpeg with raw video on stdin and, optionally, raw audio on a second pipe.

	Two piped inputs need two file descriptors. ffmpeg's pipe: protocol accepts
	any fd number, so the audio goes down an anonymous pipe whose read end is
	handed to the child and named in the command as pipe:<fd> — the `pipe:AUDIO`
	token the callers write is substituted here, once the fd exists.

	A scratch file for the audio would have been simpler and would have broken the
	promise that mkv, webm and the png zip never touch a filesystem. The writer
	has to be a thread: ffmpeg interleaves its reads, and filling one pipe while
	nothing drains the other is a deadlock.
	"""
	if aux is None:
		proc = subprocess.run(cmd, input=data, capture_output=True)
	else:
		rfd, wfd = os.pipe()
		os.set_inheritable(rfd, True)
		cmd = [c.replace("pipe:AUDIO", f"pipe:{rfd}") for c in cmd]

		def feed():
			try:
				with os.fdopen(wfd, "wb") as fh:
					fh.write(aux)
			except (BrokenPipeError, OSError):
				pass          # -shortest cut the track early; that is the design
		writer = threading.Thread(target=feed, daemon=True)
		writer.start()
		try:
			proc = subprocess.run(cmd, input=data, capture_output=True, pass_fds=(rfd,))
		finally:
			# Must close before the join: while the parent still holds the read
			# end, a blocked writer can never see EPIPE.
			os.close(rfd)
			writer.join(timeout=10)
	if proc.returncode != 0:
		raise RuntimeError("ffmpeg failed:\n" + proc.stderr.decode("utf-8", "replace")[-900:])
	return proc.stdout


def _input_args(exe, w, h, depth, fps, audio=None):
	args = [exe, "-v", "error", "-y", "-f", "rawvideo",
			"-pix_fmt", _src_pix_fmt(depth), "-s", f"{w}x{h}", "-r", f"{fps:.6f}", "-i", "pipe:0"]
	if audio:
		args += ["-f", "f32le", "-ar", str(audio["rate"]),
				 "-ac", str(audio["channels"]), "-i", "pipe:AUDIO"]
	return args


# What each container carries the track as. The lossless video formats get
# lossless audio to match, so "bit-exact" means the whole file and not just the
# picture; the delivery formats get the best codec their container allows.
_AUDIO_CODEC = {
	"h264":   ["-c:a", "aac", "-b:a", "320k"],
	"vp9":    ["-c:a", "libopus", "-b:a", "192k"],
	"ffv1":   ["-c:a", "pcm_f32le"],     # matroska carries IEEE float PCM: exact
	"prores": ["-c:a", "pcm_s24le"],     # what an NLE expects next to ProRes
	"proxy":  ["-c:a", "aac", "-b:a", "128k"],
}


def fit_samples(n_frames, fps, audio):
	"""How many samples the track must have to be exactly as long as the picture."""
	return int(round(int(n_frames) / float(fps) * int(audio["rate"])))


def _mux_args(audio, kind, samples):
	"""Map and fit the track, or nothing at all when there is no audio.

	The fit is done in SAMPLES, not with -shortest. -shortest cuts at the last
	video packet's timestamp, which is the START of the final frame — so it left
	the track up to one frame-duration short (measured: 19 ms), and by a different
	amount per container. apad to an exact length and atrim at the same sample
	handles both directions with no rounding: audio that ran short is padded with
	silence, audio that ran long is cut, and the result is the picture's length to
	the sample. Lossy encoders may still round up to their own frame size, which
	is padding at the tail, not drift at the head.
	"""
	if not audio:
		return []
	fit = f"apad=whole_len={samples},atrim=end_sample={samples}"
	return ["-map", "0:v:0", "-map", "1:a:0", "-af", fit] + _AUDIO_CODEC[kind]


def _encode_piped(raw, w, h, depth, fps, args, fmt, audio=None):
	"""Encode straight to stdout — the containers that never seek backwards."""
	exe = ffmpeg_exe()
	if exe is None:
		raise RuntimeError("ffmpeg not found (install ffmpeg or imageio-ffmpeg).")
	cmd = _input_args(exe, w, h, depth, fps, audio) + args + ["-f", fmt, "pipe:1"]
	return _run(cmd, raw, audio["raw"] if audio else None)


def _encode_tempfile(raw, w, h, depth, fps, args, ext, audio=None):
	"""Encode via a private scratch dir, read it back, delete it.

	For mp4/mov only: those muxers rewrite their header at the end, so a piped
	stream has to be fragmented, and fragmented files behave badly in editors.
	The dir comes from tempfile, not from ComfyUI's temp/, and is gone before
	this returns whether or not the encode worked.
	"""
	exe = ffmpeg_exe()
	if exe is None:
		raise RuntimeError("ffmpeg not found (install ffmpeg or imageio-ffmpeg).")
	scratch = tempfile.mkdtemp(prefix="tinode_vp_")
	try:
		path = os.path.join(scratch, f"clip.{ext}")
		cmd = _input_args(exe, w, h, depth, fps, audio) + args + ["-movflags", "+faststart", path]
		_run(cmd, raw, audio["raw"] if audio else None)
		with open(path, "rb") as fh:
			return fh.read()
	finally:
		shutil.rmtree(scratch, ignore_errors=True)


def wav_bytes(audio):
	"""The track as a float32 WAV, built in RAM. Bit-exact, and what goes in the zip."""
	exe = ffmpeg_exe()
	if exe is None:
		raise RuntimeError("ffmpeg not found (install ffmpeg or imageio-ffmpeg).")
	return _run([exe, "-v", "error", "-y", "-f", "f32le", "-ar", str(audio["rate"]),
				 "-ac", str(audio["channels"]), "-i", "pipe:0",
				 "-c:a", "pcm_f32le", "-f", "wav", "pipe:1"], audio["raw"])


def _png_zip(raw, n, h, w, depth, audio=None, compression=4):
	"""A zip of lossless PNGs, built in RAM.

	Written by ffmpeg rather than PIL/cv2 because 16-bit RGB PNG needs OpenCV,
	which is exactly the dependency this box does not have — and ffmpeg is
	already required for everything else here. A still sequence has no container
	to carry a track, so the audio rides along as a float32 WAV beside the frames.
	"""
	exe = ffmpeg_exe()
	if exe is None:
		raise RuntimeError("ffmpeg not found (install ffmpeg or imageio-ffmpeg).")
	scratch = tempfile.mkdtemp(prefix="tinode_vp_")
	try:
		pat = os.path.join(scratch, "%05d.png")
		_run(_input_args(exe, w, h, depth, 24.0) +
			 ["-c:v", "png", "-pix_fmt", "rgb48be" if depth == 16 else "rgb24",
			  "-compression_level", str(int(compression)), "-start_number", "0", pat], raw)
		buf = io.BytesIO()
		with zipfile.ZipFile(buf, "w", zipfile.ZIP_STORED) as z:
			for name in sorted(os.listdir(scratch)):
				z.write(os.path.join(scratch, name), name)
			if audio:
				z.writestr("audio.wav", wav_bytes(audio))
		return buf.getvalue()
	finally:
		shutil.rmtree(scratch, ignore_errors=True)


def make_proxy(raw, n, h, w, depth, fps, long_side=1280, crf=23, audio=None):
	"""A small h264 the browser can play and scrub — the thing you watch.

	Deliberately lossy and deliberately not what CREATE VIDEO produces: this is
	the monitor feed, re-encoded down so a 4K batch starts playing immediately.
	Every download re-encodes from the untouched master instead. The track is here
	so you can hear the cut in sync, not so you can judge the mix.
	"""
	tw, th = w, h
	if long_side and max(w, h) > long_side:
		k = long_side / float(max(w, h))
		tw, th = max(2, int(round(w * k)) // 2 * 2), max(2, int(round(h * k)) // 2 * 2)
	args = ["-c:v", "libx264", "-preset", "veryfast", "-crf", str(int(crf)),
			"-pix_fmt", "yuv420p", "-g", "24"]
	if (tw, th) != (w, h):
		args = ["-vf", f"scale={tw}:{th}:flags=bicubic"] + args
	args += _mux_args(audio, "proxy", fit_samples(n, fps, audio) if audio else 0)
	return _encode_tempfile(raw, w, h, depth, fps, args, "mp4", audio), tw, th


# slug -> (extension, mime, label). The slugs are what the frontend posts.
FORMATS = {
	"h264":   ("mp4",  "video/mp4",        "mp4 · h264"),
	"vp9":    ("webm", "video/webm",       "webm · vp9"),
	"ffv1":   ("mkv",  "video/x-matroska", "mkv · ffv1 (lossless)"),
	"prores": ("mov",  "video/quicktime",  "mov · prores 4444"),
	"png":    ("zip",  "application/zip",  "zip · png frames (lossless)"),
}


def render(sess, fmt="h264", *, fps=None, crf=17, pix_fmt="yuv420p",
		   color_range="tv", colorspace="bt709"):
	"""Encode a held clip to finished bytes. Returns (data, extension, mime).

	Always from the master, never from the proxy — the thing you download has
	never been through the preview encode.

	ffv1 and png carry the pixels through untouched: RGB in, RGB out, no matrix
	and no chroma subsampling, so they are bit-exact against what the graph
	produced — and they take no colour flags precisely because there is no
	conversion to get wrong. Their audio is float PCM, exact for the same reason.
	prores 4444 goes through 10-bit 4:4:4 (visually lossless on real footage, but
	it is still a lossy DCT codec — pure noise is its worst case). h264 and vp9 are
	the lossy delivery options and get both the conversion and the matching tags,
	so a graded clip is interpreted the way it was graded.
	"""
	raw, n, h, w, depth = sess["raw"], sess["n"], sess["h"], sess["w"], sess["depth"]
	audio = sess.get("audio")
	rate = float(fps if fps else sess["fps"])
	fit = fit_samples(n, rate, audio) if audio else 0
	ext, mime, _ = FORMATS.get(fmt, FORMATS["h264"])

	if fmt == "png":
		return _png_zip(raw, n, h, w, depth, audio), ext, mime

	if fmt == "ffv1":
		args = ["-c:v", "ffv1", "-level", "3", "-coder", "1", "-context", "1", "-g", "1",
				"-pix_fmt", "gbrp16le" if depth == 16 else "gbrp"] + _mux_args(audio, "ffv1", fit)
		return _encode_piped(raw, w, h, depth, rate, args, "matroska", audio), ext, mime

	# The filter has to precede the codec and the tags follow it — see colour_flags.
	# Skipping it is what makes a "lossless" encode come back 32 levels out.
	filt, tags = colour_flags(color_range, colorspace)
	chroma = pix_fmt if pix_fmt in ("yuv420p", "yuv444p") else "yuv420p"

	if fmt == "vp9":
		args = filt + ["-c:v", "libvpx-vp9", "-crf", str(int(crf)), "-b:v", "0",
					   "-pix_fmt", chroma] + tags + _mux_args(audio, "vp9", fit)
		return _encode_piped(raw, w, h, depth, rate, args, "webm", audio), ext, mime

	if fmt == "prores":
		# Always limited range, whatever was asked for. ProRes has no dependable
		# full-range signalling and every decoder reads it as limited, so writing
		# full-range data and tagging it pc comes back 20 levels out — measured.
		pfilt, ptags = colour_flags("tv", colorspace)
		args = pfilt + ["-c:v", "prores_ks", "-profile:v", "4444",
						"-pix_fmt", "yuv444p10le", "-vendor", "apl0"] + ptags \
			+ _mux_args(audio, "prores", fit)
		return _encode_tempfile(raw, w, h, depth, rate, args, ext, audio), ext, mime

	args = filt + ["-c:v", "libx264", "-preset", "medium", "-crf", str(int(crf)),
				   "-pix_fmt", chroma] + tags + _mux_args(audio, "h264", fit)
	return _encode_tempfile(raw, w, h, depth, rate, args, ext, audio), ext, mime


# --------------------------------------------------------------------------
# stills — the same held clip, one frame at a time
# --------------------------------------------------------------------------

# slug -> (extension, mime, lossless?). png and tiff keep the master's bit depth.
STILL_FORMATS = {
	"png":  ("png",  "image/png",  True),
	"tiff": ("tiff", "image/tiff", True),
	"jpg":  ("jpg",  "image/jpeg", False),
	"webp": ("webp", "image/webp", False),
}


def frame_bytes(sess, index):
	"""The raw RGB of one held frame, as a view — no copy of the master."""
	stride = sess["h"] * sess["w"] * 3 * (2 if sess["depth"] == 16 else 1)
	i = max(0, min(int(index), sess["n"] - 1))
	return sess["raw"][i * stride:(i + 1) * stride], i


def render_still(sess, index=0, fmt="png", quality=95, max_side=0):
	"""Encode one held frame. Returns (data, extension, mime).

	png and tiff go out at the master's own depth with no colour conversion, so
	they are bit-exact; jpg and webp are the share-it-now options. max_side>0
	downscales, which is only ever used for the on-screen view — a download always
	passes 0.
	"""
	exe = ffmpeg_exe()
	if exe is None:
		raise RuntimeError("ffmpeg not found (install ffmpeg or imageio-ffmpeg).")
	ext, mime, lossless = STILL_FORMATS.get(fmt, STILL_FORMATS["png"])
	one, i = frame_bytes(sess, index)
	h, w, depth = sess["h"], sess["w"], sess["depth"]

	pre = []
	if max_side and max(w, h) > max_side:
		k = max_side / float(max(w, h))
		pre = ["-vf", f"scale={max(1, int(round(w * k)))}:{max(1, int(round(h * k)))}:flags=bicubic"]

	if fmt == "tiff":
		# compression_algo=lzw keeps it lossless; raw would double the size.
		codec = ["-c:v", "tiff", "-compression_algo", "lzw",
				 "-pix_fmt", "rgb48le" if depth == 16 else "rgb24"]
	elif fmt == "jpg":
		# ffmpeg's -q:v runs 2 (best) to 31 (worst); the UI speaks 1-100.
		q = max(2, min(31, int(round(31 - (max(1, min(100, int(quality))) - 1) * 29 / 99))))
		codec = ["-c:v", "mjpeg", "-q:v", str(q), "-pix_fmt", "yuvj444p"]
	elif fmt == "webp":
		codec = (["-c:v", "libwebp", "-lossless", "1"] if int(quality) >= 100
				 else ["-c:v", "libwebp", "-lossless", "0", "-quality", str(int(quality))])
	else:
		codec = ["-c:v", "png", "-compression_level", "6",
				 "-pix_fmt", "rgb48be" if depth == 16 else "rgb24"]

	cmd = ([exe, "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", _src_pix_fmt(depth),
			"-s", f"{w}x{h}", "-i", "pipe:0"] + pre + codec
		   + ["-frames:v", "1", "-f", "image2pipe", "pipe:1"])
	return _run(cmd, one), ext, mime


def render_still_zip(sess, fmt="png", quality=95):
	"""Every held frame in one zip, in the chosen still format.

	png delegates to the batched writer — one ffmpeg call for the whole sequence
	instead of one per frame, which is the difference between a moment and a
	minute on a long batch. The other formats have no batched path, so they pay
	per frame.
	"""
	ext, _, _ = STILL_FORMATS.get(fmt, STILL_FORMATS["png"])
	if fmt == "png":
		return _png_zip(sess["raw"], sess["n"], sess["h"], sess["w"], sess["depth"]), \
			"zip", "application/zip"
	buf = io.BytesIO()
	with zipfile.ZipFile(buf, "w", zipfile.ZIP_STORED) as z:
		for i in range(sess["n"]):
			z.writestr(f"{i:05d}.{ext}", render_still(sess, i, fmt, quality)[0])
	return buf.getvalue(), "zip", "application/zip"


# --------------------------------------------------------------------------
# routes
# --------------------------------------------------------------------------

try:
	from server import PromptServer  # noqa: PLC0415
except Exception:  # noqa: BLE001 — importable in the test runner, no server there
	PromptServer = None


def _ranged(request, data, mime):
	"""Serve bytes with Range support so <video> can scrub instead of restart."""
	from aiohttp import web  # noqa: PLC0415

	total = len(data)
	base = {"Accept-Ranges": "bytes", "Cache-Control": "no-store"}
	m = re.match(r"bytes=(\d*)-(\d*)\s*$", request.headers.get("Range", ""))
	if not m or not total:
		return web.Response(body=data, content_type=mime, headers=base)
	lo, hi = m.group(1), m.group(2)
	if lo:
		start = int(lo)
		end = int(hi) if hi else total - 1
	else:                                   # suffix form: bytes=-N
		start, end = max(0, total - int(hi or 0)), total - 1
	end = min(end, total - 1)
	start = max(0, min(start, end))
	return web.Response(
		status=206, body=data[start:end + 1], content_type=mime,
		headers={**base, "Content-Range": f"bytes {start}-{end}/{total}"})


def _register_routes():
	if PromptServer is None or getattr(_register_routes, "_done", False):
		return
	try:
		routes = PromptServer.instance.routes
	except Exception:  # noqa: BLE001 — server not up yet
		return

	@routes.get("/tinode/vpreview/proxy")
	async def _proxy(request):  # noqa: ANN001
		from aiohttp import web  # noqa: PLC0415

		sess = get(request.query.get("id", ""))
		if sess is None:
			return web.json_response({"error": "expired"}, status=404)
		return _ranged(request, sess["proxy"], sess["proxy_mime"])

	@routes.get("/tinode/vpreview/frame")
	async def _frame(request):  # noqa: ANN001
		"""One held frame as an image. What Image Preview shows, and its download.

		`max` downscales for the on-screen view; a download omits it and gets the
		frame at full size and full depth.
		"""
		import asyncio  # noqa: PLC0415

		from aiohttp import web  # noqa: PLC0415

		q = request.query
		sess = get(q.get("id", ""))
		if sess is None:
			return web.json_response({"error": "expired"}, status=404)
		fmt = q.get("fmt", "jpg")
		if fmt not in STILL_FORMATS:
			return web.json_response({"error": f"unknown format {fmt!r}"}, status=400)
		try:
			data, ext, mime = await asyncio.get_running_loop().run_in_executor(
				None, lambda: render_still(
					sess, int(q.get("i", 0) or 0), fmt,
					int(q.get("q", 90) or 90), int(q.get("max", 0) or 0)))
		except Exception as exc:  # noqa: BLE001 — the message is the useful part
			return web.json_response({"error": str(exc)}, status=500)

		headers = {"Cache-Control": "no-store"}
		if q.get("download"):
			stem = re.sub(r"[^\w.-]+", "_", str(q.get("name") or ""))[:120].strip("._-")
			headers["Content-Disposition"] = f'attachment; filename="{stem or "ti_frame"}.{ext}"'
		return web.Response(body=data, content_type=mime, headers=headers)

	@routes.get("/tinode/vpreview/stat")
	async def _stat(request):  # noqa: ANN001
		from aiohttp import web  # noqa: PLC0415

		sess = get(request.query.get("id", ""))
		if sess is None:
			return web.json_response({"ok": False, **stats()})
		aud = sess.get("audio")
		return web.json_response({
			"ok": True, "id": sess["id"], "frames": sess["n"], "width": sess["w"],
			"height": sess["h"], "fps": sess["fps"], "depth": sess["depth"],
			"bytes": sess["bytes"], "age": time.time() - sess["created"],
			"hold": sess["hold"],
			"audio": ({"rate": aud["rate"], "channels": aud["channels"],
					   "seconds": aud["seconds"]} if aud else None),
			**stats(),
		})

	@routes.post("/tinode/vpreview/still")
	async def _still(request):  # noqa: ANN001
		"""DOWNLOAD on Image Preview: one frame, or the whole batch as a zip."""
		import asyncio  # noqa: PLC0415

		from aiohttp import web  # noqa: PLC0415

		data = await request.json()
		sess = get(data.get("id", ""))
		if sess is None:
			return web.json_response({"error": "This preview has expired — queue the graph again."},
									 status=404)
		fmt = str(data.get("format", "png"))
		if fmt not in STILL_FORMATS:
			return web.json_response({"error": f"unknown format {fmt!r}"}, status=400)
		everything = bool(data.get("all"))
		try:
			out, ext, mime = await asyncio.get_running_loop().run_in_executor(
				None, lambda: (render_still_zip(sess, fmt, int(data.get("quality", 95)))
							   if everything else
							   render_still(sess, int(data.get("index", 0)), fmt,
											int(data.get("quality", 95)))))
		except Exception as exc:  # noqa: BLE001 — the message is the useful part
			return web.json_response({"error": str(exc)}, status=500)

		stem = re.sub(r"[^\w.-]+", "_", str(data.get("name") or ""))[:120].strip("._-")
		return web.Response(body=out, content_type=mime, headers={
			"Content-Disposition": f'attachment; filename="{stem or "ti_frame"}.{ext}"',
			"Cache-Control": "no-store",
		})

	@routes.post("/tinode/vpreview/drop")
	async def _drop(request):  # noqa: ANN001
		from aiohttp import web  # noqa: PLC0415

		data = await request.json()
		return web.json_response({"ok": drop(data.get("id", "")), **stats()})

	@routes.post("/tinode/vpreview/render")
	async def _render(request):  # noqa: ANN001
		"""CREATE VIDEO: encode on demand and hand the bytes to the download.

		The result is never stored — it exists as a response body and then it is
		gone, which is the difference between this and Save Video.
		"""
		import asyncio  # noqa: PLC0415

		from aiohttp import web  # noqa: PLC0415

		data = await request.json()
		sess = get(data.get("id", ""))
		if sess is None:
			return web.json_response({"error": "This preview has expired — queue the graph again."},
									 status=404)
		fmt = str(data.get("format", "h264"))
		if fmt not in FORMATS:
			return web.json_response({"error": f"unknown format {fmt!r}"}, status=400)
		try:
			# Off the event loop: a long clip takes real seconds to encode and the
			# UI must stay responsive while it does.
			out, ext, mime = await asyncio.get_running_loop().run_in_executor(
				None, lambda: render(
					sess, fmt,
					fps=float(data.get("fps") or 0) or None,
					crf=int(data.get("crf", 17)),
					pix_fmt=str(data.get("pix_fmt", "yuv420p")),
					color_range=str(data.get("color_range", "tv")),
					colorspace=str(data.get("colorspace", "bt709")),
				))
		except Exception as exc:  # noqa: BLE001 — the message is the useful part
			return web.json_response({"error": str(exc)}, status=500)

		# The name comes from the browser, and it goes into a response header.
		# Anything but word characters is dropped rather than escaped.
		stem = re.sub(r"[^\w.-]+", "_", str(data.get("name") or ""))[:120].strip("._-")
		name = f"{stem or 'ti_preview'}.{ext}"
		return web.Response(body=out, content_type=mime, headers={
			"Content-Disposition": f'attachment; filename="{name}"',
			"Cache-Control": "no-store",
		})

	_register_routes._done = True


_register_routes()
