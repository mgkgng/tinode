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
| **Mask Crop Paste Back** | Composite processed crops back using `crop_info`. Blends through an optional mask, feathered. |

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
| **Pick Segments** / **Add Segments** / **Delete Segments** | Interactive segment curation: toggle whole tracked objects, draw new per-frame boxes, or remove individual segment instances from specific frames. |

### `tinode/video`
| Node | Does |
|---|---|
| **Load Video** | Decode a file from `input/` to an IMAGE batch via ffmpeg (frame cap / skip / every-nth / force-rate / resize). Faithful colour by default; `force_full_range` fixes a mis-tagged clip. Outputs images, frame count, fps. |
| **Save Video · Combine** | Encode an IMAGE batch to mp4 / webm / lossless PNG frames via ffmpeg, with the colour controls (`color_range`, `colorspace`, `pix_fmt`, `crf`) that keep a grade intact. Previews in the node. |

These exist so the pack can load and save video without a separate video-nodes
install. They are **not** 1:1 VHS clones — no audio, no in-browser upload (drop
files in `input/`), no batch manager — they cover the decode/encode path itself.

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
