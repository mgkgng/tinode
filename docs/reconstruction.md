# Reconstruction nodes

Tinode exposes a modular ComfyUI bridge from tracked video masks to a Gaussian
Splat. COLMAP and Nerfstudio remain external installations; the nodes invoke
their command-line programs as subprocesses so compiled CUDA/Torch dependencies
do not contaminate ComfyUI's Python environment.

## Nodes

```text
IMAGE + MASK
  -> Export Reconstruction Dataset
  -> Run Masked COLMAP SfM
  -> Prepare Masked Nerfstudio Dataset
  -> Train Splatfacto
  -> Export Gaussian Splat PLY
```

### Export Reconstruction Dataset

Accepts matched IMAGE and MASK batches. `white_is_dynamic` means white pixels
identify moving/transient content to exclude. The node validates batch shape,
thresholds and dilates the mask by the requested pixel radius, and writes:

- lossless RGB frames;
- white-dynamic masks for inspection;
- black-ignore masks for Nerfstudio;
- COLMAP masks named `frame_000000.png.png`.

Artifacts live under:

```text
ComfyUI/output/tinode/reconstruction/<dataset_name>/
```

Directories are never silently overwritten. Change `dataset_name` to start a
new run.

### Run Masked COLMAP SfM

Runs feature extraction, matching, and mapping. `sequential` is appropriate for
ordered video frames; `exhaustive` is useful for smaller unordered image sets.
The camera is shared by default because frames normally come from one clip.

`colmap_command` can be a direct executable or a safely parsed prefix:

```text
colmap
/opt/colmap/bin/colmap
conda run -n reconstruction colmap
```

### Prepare Masked Nerfstudio Dataset

Calls `ns-process-data images` with `--skip-colmap`, preserving the masked
camera solution instead of solving SfM again. It attaches `mask_path` to every
frame and creates matching binary masks for Nerfstudio's downscaled image
directories.

### Train Splatfacto

Calls `ns-train` in the external reconstruction environment. ComfyUI models can
be unloaded first to release VRAM. TensorBoard logging is used instead of an
interactive viewer so the ComfyUI queue can finish cleanly.

Example external command:

```text
conda run -n reconstruction ns-train
```

Long training occupies the ComfyUI queue. Logs stream to the terminal and are
also stored inside the dataset. Failed directories are retained for diagnosis
and are never automatically deleted.

Nerfstudio applies these masks to the photometric loss. This strongly reduces
transient reconstruction, especially because masked COLMAP also avoids seeding
points on moving objects, but it is not a full 3D scene-completion algorithm.
Regions unseen in every source frame still require reconstruction-aware
inpainting, and stray Gaussians may require cleanup.

### Export Gaussian Splat PLY

Calls `ns-export gaussian-splat` with the trained `config.yml` and returns the
resulting `.ply` path.

## External environment

Install COLMAP and Nerfstudio separately. A typical isolated environment is:

```bash
conda create -n reconstruction python=3.10 -y
conda activate reconstruction
python -m pip install --upgrade pip
python -m pip install nerfstudio
```

Then enter commands such as `conda run -n reconstruction ns-process-data` in
the corresponding node fields, or enter absolute executable paths.

## Current boundary

The first implementation is synchronous and intentionally transparent. It does
not provide background-job polling, cancellation, a native splat viewer, or
reconstruction-aware 3D inpainting. Those can be added after the basic pipeline
is validated against real footage.
