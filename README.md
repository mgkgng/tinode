# tinode

Custom ComfyUI nodes for video retouching work — interactive cropping, mask
curation, and picking your way through a SAM3 segmentation with 90 overlapping
objects in it.

Built around one idea: **a video is just an `[N,H,W,C]` IMAGE batch**, so a node
that handles a batch correctly handles a clip correctly. Crops are taken at
native scale (no resampling), so a crop → process → paste round trip leaves
every untouched pixel bit-identical.

---

## Installation

Clone into `ComfyUI/custom_nodes/` and restart:

```bash
git clone https://github.com/mgkgng/tinode ComfyUI/custom_nodes/tinode
```

The default install has **no dependencies** — the image, video and segment nodes
run on the torch / numpy / Pillow that ComfyUI already provides.

Two optional pieces:

| What | When you need it |
|---|---|
| `opencv-python` | **Mask Clean Islands** only. Left unpinned deliberately: naming a variant (`-headless`, `-contrib`) can clobber one another pack installed. The node says so if it's missing. |
| `pip install -r requirements-face.txt` | The two face nodes only (insightface / onnxruntime / mediapipe — hundreds of MB). Both import lazily, so everything else works without them. |

After changing any `web/*.js`, **hard-refresh the browser** (Ctrl/Cmd-Shift-R).

---

## The interactive nodes

Three nodes mount a canvas in the node body. They all follow the same two-pass
rhythm, which is a consequence of how ComfyUI executes:

> **Queue once** so the node produces the frames → **edit on the canvas** →
> **queue again** to apply your edit.

The frontend can't see an upstream image until the graph runs. The heavy node
upstream (SAM3, a crop) is cached, so editing and re-queuing does **not**
re-run it.

### Bbox Crop · Manual
Drag four corner handles (or an edge, or the whole box) over a preview of the
incoming frame; `x / y / width / height` update live and are saved with the
workflow. One box applies to every frame, so image → image and video → video.
Emits `TI_CROP_XFORM` for the round trip back.

### Pick Segments
Feed it the `segments` output of a patched EasySAM3 Segment plus the original
image. Scrub frames (slider, ◀ ▶, arrow keys); every detection is a colorized
box; hovering lights up its real mask shape; **clicking the object toggles it**.
Picking is pixel-accurate — a per-frame label map resolves the click to the
exact object under the cursor even where dozens of boxes overlap.

Selection is **per-id and global**: excluding an object drops it on every frame
it appears in. Outputs the union `mask`, the `image` with the kept segments
drawn, and the filtered `segments`.

### Add Segments
The complement: **drag a box to add** a region SAM3 missed on the current frame,
**right-click a box you drew to remove it**. Boxes are per-frame. Same three
outputs as Pick Segments, so the two are interchangeable and chain in either
order.

### Delete Segments
Scrub to a frame and **click a segment to delete that exact instance**. Click it
again to restore it. Unlike Pick Segments, deletion is per-frame rather than
per-id, so removing one bad detection does not remove the tracked object from
the rest of the video.

All segment editors **reset when the input changes** — a selection, deletion,
or drawn box is stamped with a signature of the image + segments it was made
against, so it is never silently re-applied to a different clip.

---

## Node reference

### `tinode/image` — crop & paste
| Node | Does |
|---|---|
| **Mask Bbox Crop** | Crop to a mask's bounding box + padding, rounded to `divisible_by`. Per-frame boxes are temporally smoothed so the crop stops swimming; `shared_bbox` gives one static box instead. |
| **Mask Crop · Center Fill** | Crop each mask onto its own square black canvas, scaled to fill — for crowd → per-face pipelines. |
| **Bbox Crop · Manual** | Interactive crop (above). |
| **Crop By Info** | Re-cut the exact crop a saved `crop_info` describes (the forward of Paste Back) — a native-scale slice, bit-exact, no coords to re-enter. Rejects rescaled crop_info rather than resample. |
| **Mask Crop Paste Back** | Composite processed crops back using `crop_info`. Blends through an optional mask, `gaussian`/`box` feathered; **bit-exact outside the mask**, and no resample when the crop isn't rescaled. |
| **Image Preview · Ephemeral** | Look at an image (or step through a batch) without saving it — held in **RAM only**, no PNG in `temp/`, no queue-history entry. A **DOWNLOAD** button writes png / tiff / jpg / webp to your machine, one frame or the whole batch as a zip. png and tiff are bit-exact at the master's own bit depth. Passes `images` through. |

> **The one rule for Paste Back:** its `image` must be the **original frame the
> crop node consumed**, never the crop node's output. `crop_info` coordinates
> live in that original frame. It now raises a readable error if they mismatch.

### `tinode/image` — masks & batches
| Node | Does |
|---|---|
| **Mask Clean Islands** | Delete speckle, fill pinholes via connected components. Never erodes/dilates, so the real boundary and its antialiasing survive exactly. |
| **Pad Image · Add Border** | Enlarge a frame by adding a solid-colour border on any side (hex colour, `#000` default). Outputs a MASK of the added region (invertible) so you can outpaint exactly the new space, plus the padded amount per side. |
| **Mask Translate** | Shift a mask. |
| **Batch Drop / Pick Indices** | Keep or remove frames by index list; Pick preserves order, so it doubles as a reorder. |
| **Mask Drop / Pick Indices** | Same, for MASK batches. |
| **Crop Info Drop / Pick Indices** | Same, for `TI_CROP_XFORM` — so images, masks and crop_info stay in lockstep. |

Index lists are 1-based by default. Validation is strict: any malformed token
makes the node a **no-op** rather than silently selecting the wrong frames.

### `tinode/image` — video & segments
| Node | Does |
|---|---|
| **Extend Video · Prepend/Append** | Add frames at either end: hold the first/last frame, or splice in another clip (auto-conformed to the base resolution/channels). Returns how many frames it added at each end so you can trim them later. |
| **Insert Video** | Drop a clip into another at a frame. `replace` overwrites the frames it covers — the span's end is the clip's own length, so you never compute it by hand; `insert` splices it in and grows the video. Returns the start/end it occupies. |
| **Trim Video · Cut Frames** | Cut N frames off the head and/or tail (the video ltrim/rtrim). Never emits an empty batch. |
| **Cut Video · Start + Frame Count** | Extract an exact contiguous span. The start supports Python-style negative indices (`-1` is the last frame); invalid or overlong ranges report a clear error. |
| **Mask to Segment** | Convert a MASK batch into one tracked, editable `TI_SAM3_SEGMENTS` object for Pick Segments or Add Segments. Empty frames and video alignment are preserved. |
| **Pick Segments** / **Add Segments** / **Delete Segments** | Interactive segment curation: toggle whole tracked objects, draw new per-frame boxes, or remove individual segment instances from specific frames. All three also emit **`bbox_mask`** — filled rectangles over each segment's bbox instead of its outline, which gives an inpainting model clean margin on every side (at the cost of regenerating everything else inside the box). |
| **Segments to Masks** | Split the (unmerged) `segments` stream into a per-object mask batch — one `[frames,H,W]` MASK per id, as a LIST you can take one at a time. Set `object_ids` to a single id to get just that object. |
| **Segment Mask · Select** | One object's mask as a **single** MASK, picked by `index` (0..count-1) — for stepping through objects into a mask input like Mask Bbox Crop. Reports the `id` and total `count`. |

### `tinode/video`
| Node | Does |
|---|---|
| **Video Concatenate** | Append one native `VIDEO` after another — hard cut, audio kept in sync. Built to accumulate a clip per iteration across a Foreach loop. |
| **Load Video** | Decode a file from `input/` to an IMAGE batch via ffmpeg (frame cap / skip / every-nth / force-rate / resize). Faithful colour by default; `force_full_range` fixes a mis-tagged clip. Outputs images, frame count, fps, plus **`stem` / `path` / `video`** — the same handles Load Videos gives, so one clip can drive the per-clip store. `video` is the whole untouched file, so it only matches `images` at default frame-selection settings (the node says so when it doesn't). |
| **Load Videos** | Gather many clips from a folder as an Inspire `ITEM_LIST` of **lazy** native `VIDEO`s, to loop over one at a time. `directory` is relative to `input/` or an absolute path; `pattern` filters by wildcard (`*.mp4`, `PROJECT_AMIR_*`); `filenames` (one per line, exact or wildcard) picks an exact set. Outputs `item_list`, a per-clip `videos` list, and `count`. |
| **Save Video · Combine** | Encode an IMAGE batch to mp4 / webm / lossless PNG frames via ffmpeg, with the colour controls (`color_range`, `colorspace`, `pix_fmt`, `crf`) that keep a grade intact. Previews in the node. |
| **Video Preview · Ephemeral** | Watch a clip without saving it. Holds the frames in **RAM only** — nothing in `output/`, nothing in `temp/`, nothing in the queue history — streams a proxy into a `<video>` in the node body, and a **CREATE VIDEO** button encodes on demand and downloads to your machine. Optional `audio` plays with it and is muxed into every download, fitted to the picture's length. Passes `images` / `audio` straight through, so it can sit mid-chain as a monitor. |
| **Video Source Path** | The file a native `VIDEO` was loaded from → `stem` / `filename` / `path`. Keys a clip's saved artifacts inside a Foreach loop. |
| **Save Crop & Mask** | Phase 1 of batch removal: write a clip's mask (lossless PNG, crop space) + its `crop_info` + a manifest, keyed by `stem`, under `output/<subdir>/`. Optionally also exports the **cropped RGB frames** as a lossless PNG sequence (`crop_image`, 16- or 8-bit) so the removal step — even an external tool — works on the exact pixels. The source video is never re-encoded. |
| **Load Masks** | Phase 2: scan the mask store → an Inspire `ITEM_LIST`, one item per clip. `source_dir` re-locates moved footage. |
| **Load Mask** | Inside the phase-2 loop: one item → `video` (original, re-decoded) + `mask` (crop space) + `crop_info` + `stem`. |
| **Load Cropped Frames** | Inside the phase-2 loop: one item → the lossless `crops` Save Crop & Mask exported (8- or 16-bit), for feeding removal without re-decoding. |
| **Split Video · Chunks** | Cut a clip into pieces by a `cuts` string (`18, 124` → 0..17, 18..123, 124..end) and emit an Inspire `ITEM_LIST` — one chunk per loop iteration. `max_length` also caps chunk length (189 fits VOID's 197 once Frame Pad prepends 8), splitting long spans evenly rather than leaving a stub tail. Lossless: a tensor slice. |
| **Chunk Item** | Inside the loop: one item → `images` + `mask` + `start`/`end` + `chunk_index`/`chunk_count`. |
| **Join Chunks** | Accumulate processed chunks back into one clip across the loop (wire the accumulator to `ForeachListBegin.intermediate_output`). Split → join with no processing is bit-exact. |
| **Frame Pad** | Prepend duplicate head frames so the clip length is `4n+1` (a video-VAE constraint, e.g. Netflix VOID) while adding ≥`min_prepend` (8 reduces head error). Pads the mask in lockstep; outputs `pad_count`. |
| **Frame Unpad** | Drop the `pad_count` head frames Frame Pad added, recovering the original-length result. |

#### Video Concatenate

The only node here that speaks ComfyUI's **native `VIDEO`** type (`comfy_api`'s
`VideoInput`) rather than an IMAGE batch — in, out, and all the way to
`SaveVideo`. Load/Save Video above are the ffmpeg IMAGE-batch pair; this is a
different world, don't mix them up.

Its reason to exist is accumulating one clip per iteration of Inspire's
**▶Foreach List**:

```
ForeachListBegin.intermediate_output ──► video_a ┐
                                                 ├─ Video Concatenate ──► ForeachListEnd.intermediate_output
        this iteration's generated VIDEO ──► video_b ┘

ForeachListEnd.result ──► SaveVideo          (one file, one encode, at the end)
```

**Seed `ForeachListBegin.initial_input` or you will silently lose step 1.**
This is not about this node — it is how Inspire's loop starts:

```python
if initial_input is None:
    initial_input = item_list[0]   # your first item becomes the seed...
    item_list = item_list[1:]      # ...and is never iterated
```

So leave it unconnected and the first item is eaten as the accumulator's initial
value instead of being processed. Connect **any** value — an `Int` primitive set
to 0 is fine — and all N items iterate. Video Concatenate treats a non-`VIDEO`
`video_a` as "no accumulator yet" and passes `video_b` straight through, so the
seed never has to be a real video, and there is no empty-video node to fake.
Verified against Inspire's actual `ForeachListBegin`, 5 recipe steps:

```
initial_input SEEDED  : steps generated [1, 2, 3, 4, 5]   30 frames, 1.250s
initial_input EMPTY   : steps generated [2, 3, 4, 5]      24 frames, 1.000s   <-- step 1 gone
```

**Appending is lazy.** The output holds an ordered list of its parts; nothing is
decoded, copied, or re-encoded until something asks for pixels. Concatenating
tensors on every iteration instead would re-copy the whole accumulated clip once
per step — O(N²) memcpy, with a 2x memory spike each time. Here the single copy
happens once, when `SaveVideo` materializes the result. Parts stay a **flat**
list, so iteration N does not nest N videos deep.

**A hard cut, in every stream.** Every output frame is one input frame,
untouched — no interpolation, crossfade, or duplicated transition frame. Decode
a 5-clip join back and you get exactly 5 runs of identical frames, no
in-between. Specifically:

- **Frame rate** — a clip whose rate differs from `video_a`'s is retimed by
  nearest-neighbour index mapping: frames repeat or drop, never blend. Its
  duration survives (24fps + 2s of 12fps → 72 frames, 3.000s).
- **Audio** — each part's audio is fitted to *that part's own* video duration
  before joining, padded with silence or trimmed. A part with no audio gets
  silence rather than a shorter slot, so it cannot shift everything after it out
  of sync. Sample rates are resampled to the first audio-bearing part's rate;
  mono upmixes to stereo by duplication rather than losing a channel.
- **Resolution** — a mismatch **raises**. Silently rescaling would give the
  whole recipe one step's wrong geometry; the error tells you which clip differs
  and by how much.

Frames are materialized as CPU float32 — a growing accumulator has no business
sitting in VRAM. Nothing is written to disk: `save_to()` delegates the encode to
core's `VideoFromComponents`, so the file is exactly what `Create Video` →
`Save Video` would have produced.

These exist so the pack can load and save video without a separate video-nodes
install. They are **not** 1:1 VHS clones — no audio, no in-browser upload (drop
files in `input/`), no batch manager — they cover the decode/encode path itself.

#### Load Videos — a folder of clips, one iteration each

`Load Video` (singular) decodes **one** file to an IMAGE batch. `Load Videos`
(plural) is for the other case: you have many clips — possibly gigabytes total —
and want a loop to process them one at a time.

It scans `directory` and outputs an Inspire **`ITEM_LIST`** of native `VIDEO`s.
Wire `item_list` into **▶Foreach List** and each iteration's `item` is one clip:

```
Load Videos.item_list ──► ForeachListBegin.item_list
                          ForeachListBegin.item ──► (Get Video Components ──► your per-clip graph)
                                                     ...accumulate with Video Concatenate...
                          ForeachListEnd.result ──► Save Video   (once, at the end)
```

**Why this handles ">5GB of video" without exhausting RAM.** Every item is a
*lazy* `VideoFromFile` — a path, not pixels. The whole list costs almost
nothing; only the clip the current iteration decodes is ever in memory, and it
is released before the next. Loading every clip to an IMAGE batch up front would
instead need the **sum** of all of them at once. The file's size on disk is
never the wall — decoded frames are (~25 MB per 1080p frame, ~100 MB per 4K
frame), and the loop keeps that to one clip at a time.

- `directory` — relative to `input/`, or an **absolute** path so you can point
  straight at a source folder elsewhere without copying gigabytes into `input/`.
- `pattern` — optional **wildcard** filter over the folder: `*.mp4`,
  `PROJECT_AMIR_*` (case-insensitive, videos only). Empty = every video.
- `filenames` — optional, one per line, to load an exact set in an exact order.
  Each line is an exact name (`clip` finds `clip.mp4`) **or** a wildcard
  (`PROJECT_AMIR_*.mp4`); globs expand in folder order, exact names keep their
  line order, duplicates are dropped. Takes precedence over `pattern`. Empty (and
  no pattern) loads every video, sorted by name (`reverse` flips the result).
- `videos` (a ComfyUI list) is the other idiom: wire it anywhere and every
  downstream node runs once per clip, no Foreach node needed.

The folder is empty-checked and missing named files **raise** — a silently empty
list makes ▶Foreach List throw and makes a per-item branch skip without a word.

#### Two-workflow object removal (mask now, remove later)

For removing an object across many clips, split the work in two passes with a
disk handoff — so masking (light, reviewable) and removal (heavy, unattended)
don't have to run together, and a crash mid-batch never re-does finished work.

**Phase 1 — author masks** (loop over `Load Videos`):
```
item(VIDEO) ─┬─► Get Video Components ─► Bbox Crop · Manual ─► SAM3 ─► … ─► mask
             │                                    └─────────► crop_info ─┐
             └─► Video Source Path ─► stem ──────────────────────────────┤
                                                     mask + crop_info + stem ─► Save Crop & Mask ─► ForeachListEnd
```
`Save Crop & Mask` writes, per clip under `output/ti_masks/<stem>/`: the mask as a
**lossless** PNG sequence in crop space, the `crop_info`, and a manifest. The
source is never re-encoded.

**Phase 2 — remove + paste back** (loop over `Load Masks`):
```
item ─► Load Mask ─┬─ video ─► Get Video Components ─► frames ─┬─► Crop By Info ─► [ your removal model + mask ] ─► filled crop ─┐
                   ├─ crop_info ──────────────────────────────┼──────────────────────────────────────────────────────────────┤
                   └─ mask ───────────────────────────────────┘                                     frames + filled + crop_info + mask ─► Mask Crop Paste Back ─► Save Video
```
`Crop By Info` reproduces phase 1's exact crop (bit-exact); `Mask Crop Paste
Back` composites the filled crop back through the mask — everything outside the
mask stays the untouched original, so there is no crop-rectangle seam.

A ready-to-run **phase-2 template** ships at
[`workflows/phase2_object_removal.json`](workflows/phase2_object_removal.json):
it loads the store, loops, and pastes back to a lossless master, wired as an
identity no-op today — drop your removal model into the marked gap (crops + mask
→ filled crops).

**No quality loss anywhere in the chain:** masks are lossless PNG, crops and
paste-back are native-scale tensor ops (no resample unless you rescale), and the
source is only ever decoded, never re-encoded. The single lossy step in the
whole system is the *final* `Save Video` encode — set it to `png` frames or
`crf 0` + `yuv444p` for a lossless master.

#### Paste-back quality — is it the best for 4K?

For **video** object removal, a **feathered alpha composite over the untouched
original** (what Paste Back does) is the right default, and better than the
fancier options for this job:

- **Feathered alpha (default).** Only the masked hole is written; every other
  pixel is bit-exact original, so no global colour shift and nothing to seam.
  It is deterministic per frame, so it does **not** flicker. Use `gaussian`
  feather at 4K for a smoother edge than `box`.
- **Laplacian (multi-band) blending.** Great for compositing two *different*
  images across a long seam (panorama stitching). For removal, if your inpainter
  fills plausibly it buys little, and applied per-frame it can smear
  high-frequency detail near the edge. Worth it only as a *narrow-band* edge
  refiner when a residual tone step remains — say the word and it's a bolt-on.
- **Poisson (seamless cloning).** **Not recommended for video.** Solving each
  frame independently in the gradient domain drifts frame-to-frame → **flicker**,
  can bleed colour across strong edges, and is expensive at 4K.

Bottom line: get the *fill* right (a temporally-aware video inpainter) and a
feathered alpha composite is professional-grade. The blend is not where 4K
quality is won or lost — the inpainter and staying lossless are.

#### Video Preview — look at it without keeping it

Save Video writes a file: it lands in `output/`, shows up in the queue history,
and after an afternoon of iterating there are two hundred to sort through. Most
of those runs you only wanted to *look* at.

Video Preview plays the clip and stores nothing.

- The frames are held in a **process-global RAM store**, keyed per node: re-queue
  and the node's previous clip is replaced, not stacked. Restart ComfyUI and every
  trace is gone.
- What you watch is a small **h264 proxy** (default 720p long side), streamed from
  RAM with HTTP Range support so the scrub bar works.
- **CREATE VIDEO** re-encodes from the untouched master and hands the bytes
  straight to the browser download. The server never keeps a copy.
- The **✕** button releases the clip immediately; deleting the node does too;
  otherwise it expires after `hold_minutes` (default 30).

Download formats, and what each one costs you:

| Format | Fidelity |
|---|---|
| `mkv · ffv1` | **Bit-exact.** RGB in, RGB out — no matrix, no subsampling. 8- or 16-bit. |
| `zip · png frames` | **Bit-exact.** One lossless PNG per frame, 8- or 16-bit, numbered from `00000`. |
| `mov · prores 4444` | 10-bit 4:4:4, visually lossless on real footage, but still a lossy DCT codec. |
| `mp4 · h264`, `webm · vp9` | Lossy delivery. `crf`, chroma and range are exposed. |

**Audio** is optional and rides along everywhere: you hear it in the node, and
every download muxes it. The lossless formats carry it losslessly too — float PCM
in mkv, a float WAV beside the frames in the png zip — so "bit-exact" covers the
whole file. mp4 gets AAC 320k, webm Opus 192k, mov 24-bit PCM.

The track is fitted to the picture in **samples**: padded with silence if it ran
short, cut if it ran long. The first attempt used ffmpeg's `-shortest`, which cuts
at the last video packet's timestamp — the *start* of the final frame — and left
the track up to one frame-duration short (19 ms measured), by a different amount
per container. Video timing is the thing that must not move, so the fit is exact
now and `test_video_preview_fits_the_track_to_the_picture` pins it.

The player starts muted, because browsers refuse to autoplay with sound. Unmute
once and the choice is remembered — by then the page has had a user gesture, so
later runs come back with sound.

`precision` decides what the held master keeps. **auto** checks whether the
values already sit on the 1/255 grid: a clip straight off Load Video does, and is
held at 8 bits; anything that has been through a blend, a resize or a colour match
carries genuine sub-8-bit detail and is held at **16 bits, which doubles the RAM**.
Force it either way if you would rather have the memory back. The ceiling across
all held clips is 8 GB — change it with `TINODE_PREVIEW_RAM_MB` (in megabytes),
which is read once at import, so it needs a ComfyUI restart.

The one caveat on "never touches the disk": the mp4 and mov muxers rewrite their
header at the end, so those two encode via a private `tempfile` directory that is
deleted before the response is sent. A fragmented stream handed to an editor is a
worse problem than a file that exists for half a second outside ComfyUI's tree.
mkv, webm and the PNG zip never leave RAM — including with audio, which goes down
a second anonymous pipe (`pipe:<fd>`) rather than a scratch file, since two piped
inputs need two file descriptors.

#### Image Preview — the same bargain, for stills

ComfyUI's Preview Image writes a PNG into `temp/` and Save Image writes one into
`output/`; both leave a file and a queue-history entry for every run. Image
Preview holds the batch in RAM, encodes each frame on demand for the view, and
only writes something when you press **DOWNLOAD** — to your machine, not the
server's. Batches get ‹ › arrows and a scrubber, and the download is either the
frame on screen or all of them as a zip.

| Format | Fidelity |
|---|---|
| `png` | **Bit-exact**, at the master's own depth — 16-bit in, 16-bit PNG out. |
| `tiff` | **Bit-exact**, LZW, 8- or 16-bit. For a compositing handoff. |
| `jpg`, `webp` | Lossy, with a quality knob. webp at 100 is lossless. |

The on-screen view is a downscaled jpg (long side 1600) so a 4K batch steps
instantly; a download always asks for the frame at full size and full depth.

One trap worth knowing: **PIL cannot read a 16-bit RGB PNG.** It has no 48-bit
RGB mode and hands back `uint8` without complaining, so a "check" of a 16-bit
export through Pillow will silently measure nothing. Use ffmpeg or OpenCV. The
regression test pins this, because it first made a bit-exactness test pass for
the wrong reason.

#### Video colour

A graded clip can look flat or wrong after a decode/encode round trip. The pixels
in between are untouched (every crop node here is bit-exact); the shift is the
YUV↔RGB conversion. Two rules:

- **HDR sources** (BT.2020 + PQ/`smpte2084` or HLG — common with 4K footage) look
  flat and washed-out when decoded as SDR, because the PQ curve and wide gamut get
  read as sRGB. Load Video's `tonemap_hdr` (default **auto**) tone-maps them to
  BT.709 SDR and leaves normal clips alone. It uses ffmpeg's **libplacebo**
  (BT.2446a) when available — the best-looking option — and falls back to a CPU
  zscale+tonemap chain otherwise. `ffprobe` your file — if `color_transfer` is
  `smpte2084`/`arib-std-b67`, this is your fix, and no source re-encode is needed.
  (8-bit note: tone-mapping into ComfyUI's 8-bit pipeline can band slightly in
  smooth gradients — correct look, not full HDR depth.)
- **On load** for SDR, the decode trusts the file's colour tags — faithful, and it
  matches ffmpeg's own `rgb24` conversion exactly. If an SDR clip still loads
  washed-out it is mis-tagged: turn on `force_full_range`.
- **On save**, set `color_range` (tv = limited/16–235, pc = full/0–255) and
  `colorspace` to **match your source** (`ffprobe` it), so a player interprets the
  file the way it was graded. For a lossless intermediate, save **PNG frames**
  (no colour conversion, no compression) and mux to video yourself.
- **The tags are not the conversion.** ffmpeg's `-colorspace` / `-color_range`
  only *label* the file; its implicit RGB→YUV conversion is BT.601 limited no
  matter what you label it. Encoding at the usual bt709 + tv therefore converted
  as 601 and labelled 709, and any player that trusted the label undid a matrix
  that was never applied — measured at up to **32/255 out, mean 5.9**, on a
  gradient. `_video_io.colour_flags()` now emits the matching `scale` filter
  alongside the tags, which brings the same round trip to 2/255 at 4:4:4 (matrix
  rounding) and ~5/255 at 4:2:0 (chroma subsampling, irreducible). This affected
  **Save Video · Combine at its defaults**; `test_colour_flags_convert_as_well_as_tag`
  and `test_encode_round_trips_within_its_own_colour_tags` pin it.

### `tinode/data`
| Node | Does |
|---|---|
| **JSON Path** | Read one value out of a JSON document by path — keys *and* indices: `steps[0].prompt`. |
| **JSON To Item List** | Split a JSON array into one item per element — `[{"a":1},{"a":2}]` gives 2 items. |

They chain: **JSON Path** `steps` → **JSON To Item List** → **▶Foreach List**.

#### JSON Path

Every accessor form, including the ones Simple JSON Parser cannot express:

| Path | Reaches |
|---|---|
| `steps[0].prompt` | key, then index, then key |
| `a.b.c` | nested keys |
| `-1` | a bare index — negative counts from the end |
| `[0].name` | a leading index (the document itself is an array) |
| `a[0][2]` | chained indices |
| `["key.with.dots"]` | a quoted key, for keys containing a dot or bracket |
| *(empty)* | the whole document |

Outputs `value` (strings unquoted, objects/arrays as JSON text), `count` (length
for an array or object, else `-1`) and `type` (`object`/`array`/`string`/
`number`/`boolean`/`null`).

Two rules that keep a wrong path from becoming a silent wrong result:

- A **malformed** path (`a..b`, `a[`) always raises — that's a workflow bug, not
  data — while a path that simply **isn't there** respects `strict`: raise, or
  fall back to `default`. So a typo can never quietly hand you your default.
- Indexing a string is a miss, not a character. `prompt[0]` yielding `"a"` from
  `"a cat"` hides a wrong path far more often than it helps.

Errors name what was actually available: `key 'nope' not found at the document;
available: 'steps', 'nested', 'meta'`.

#### Caching

**Both nodes cache**, deliberately. Neither defines `IS_CHANGED`, because both
are pure functions of their inputs — ComfyUI's own input-signature key is
already exactly right, so they re-run only when the text or path really changes.

This is the one behavioural difference from **Simple JSON Parser**, which
returns `float("NaN")` from `IS_CHANGED`. That value is folded straight into the
cache key (`comfy_execution/caching.py`), and a fresh, never-equal `NaN` is
minted on every queue — so that node re-executes every single run. Worse, a
node's key also folds in **all of its ancestors'** signatures, so everything
downstream of it re-executes too. Verified against the real
`HierarchicalCache`/`IsChangedCache`: identical inputs → hit on every one of
these nodes and their children; change only `path` → just JSON Path and its
children miss; change the document → all miss.

If you *want* a branch to re-run every queue, that's what Inspire's reroute /
seed-style nodes are for — don't reach for these.

Two outputs, two idioms:

- **`item_list`** is an `ITEM_LIST`, the type Inspire's **▶Foreach List** consumes
  — a sequential loop that threads an accumulator through each item.
- **`items`** is a plain ComfyUI list, so every node downstream simply runs once
  per item, no loop nodes involved.

Inspire's own **Worklist To Item List** goes the other way: it sets
`INPUT_IS_LIST` to *collapse* a batch ComfyUI already ran per-item back into one
`ITEM_LIST`, which needs an upstream node that emits a list. Here the array is a
single string, so there is nothing to collapse — this node builds that same
payload directly. Nothing is imported from Inspire, so the pack stays optional;
you only need it if you want the loop nodes.

Items come out as **strings**, since that is what a socket can carry: objects
and arrays are re-serialized as compact JSON, while string elements pass through
unquoted (`["cat","dog"]` → `cat`, `dog`). A lone object counts as one item, and
JSON Lines is accepted as a fallback. Invalid JSON and an **empty array both
raise** — an empty list makes ▶Foreach List throw an `IndexError`, and makes the
ComfyUI-list output skip the whole branch in silence.

### `tinode/util`
| Node | Does |
|---|---|
| **Item Cursor** | Step through an `ITEM_LIST` **by hand**, one item per queue. A ▶Foreach List runs everything start to finish, so you can only tune settings on item 0; this emits the item at `index` instead, with ◀ Prev / Next ▶ / ⟲ buttons and a readout ("3 / 19 · sh0030 · crop 1"). Re-queue as often as you like on the current item, then step. Drop-in for `ForeachListBegin` — same `item`, and no loop end to wire. |
| **Item Params** | Per-item settings that survive the batch. A node's widgets are GLOBAL, so tuning crop 1 then crop 2 leaves only the last value — this stores one JSON object **per key** and hands back the current key's entry. Tune an item, step away, step back: its settings return, and the Foreach run replays them. Guards against saving one item's edits onto the next. |
| **Param Get** | Pull one named value out of an Item Params entry, typed to drive a widget input: `string` (prompt, boxes, cuts), `float` (threshold, alpha), `int` (counts), `boolean` (flags), plus `found`. Missing names fall back rather than erroring. |
| **Validation Gate** | Pause the running workflow until you approve. A floating panel shows a `preview` (e.g. the mask overlay) with Approve (Enter) / Reject (Esc); Approve passes the value through and the graph continues, Reject stops the run. Blocks the execution thread on an event released by a POST route, polling ComfyUI's interrupt so Cancel still works; headless (no server) it passes straight through so batch/cron runs never hang. This is what lets you curate every clip/crop by hand *inside* a Foreach loop — the loop waits for you at each item. |

### `tinode/face`
| Node | Does |
|---|---|
| **Face Similarity Sort** | Reorder a face batch into an identity gradient (ArcFace + greedy nearest-neighbour). |
| **Face Landmark Morph** | Feature-aware warping from a 468-point face mesh. |

### `tinode/conditioning`
| Node | Does |
|---|---|
| **CLIP Text Encode (Override)** | Text-as-input CLIP encode. |

---

## Custom types

Two data types travel between these nodes. ComfyUI only matches type *strings*
between slots — it never checks the shape — so both contracts are written down
in [`schema.py`](schema.py) with validators. Full field docs live there.

- **`TI_CROP_XFORM`** — how to undo a crop: the original `H/W`, and one item per
  crop (`y0,x0,h,w` in the original frame; `oy,ox,nh,nw` on the crop canvas).
  `None` marks a frame with nothing to paste, preserving index alignment.
- **`TI_SAM3_SEGMENTS`** — every SAM3 detection per frame, unmerged: `id`,
  `bbox`, `conf`, and a **bbox-cropped** mask (full-frame masks for 90 objects
  would not fit in memory). Ids are stable track ids, which is what makes an
  id-based selection mean the same object across the whole clip.

`TI_SAM3_SEGMENTS` requires a small patch to EasySAM3 Segment that adds a third
output exposing the detections before it merges them — see
[mgkgng/ComfyUI-EasySAM3](https://github.com/mgkgng/ComfyUI-EasySAM3).

---

## Development

Nodes are auto-discovered: drop a file under `nodes/<group>/`, subclass
`TiNode`, decorate with `@register`. `registry.py` walks the package, namespaces
every id with `TI_` (node ids are **global** across all installed packs), raises
on duplicates instead of silently shadowing, and isolates import errors so one
broken node can't take down the pack.

Shared frontend helpers live in [`web/lib/editor.js`](web/lib/editor.js) —
notably the screen → canvas pointer rescale, without which every hit-test misses
at any zoom other than 100%.

Tests need torch, so run them with ComfyUI's interpreter:

```bash
ComfyUI/venv/bin/python tests/test_nodes.py   # standalone, no pytest needed
pytest tests/                                 # also works
```

They cover the things that regress silently: crop/paste bit-exactness, the
index parsers, the type contracts, stale-selection resets, cache pruning, and a
check that the Python and JS colour functions still agree (it actually executes
the JS).

## License

MIT.
