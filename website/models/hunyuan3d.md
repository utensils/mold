# Hunyuan3D

Image-to-3D. Give it one photograph and it returns a triangle mesh.

Unlike every other family in mold, Hunyuan3D has **no text encoder at all**.
The source image is the only conditioning, so a prompt is recorded as
provenance and never read, and a request without an image is refused rather
than answered from an empty prompt.

**No prompt is needed anywhere.** The model's generation profile says so, and
every surface reads it, so `mold run hunyuan3d-mini-turbo --image chair.png` is
a complete request from the CLI, the API, the TUI, Discord and the apps alike.

**Available in the web SPA, the desktop app, and the iPhone app**, not only
the CLI. Picking this model in Create reshapes the form from its generation
profile: the raster controls disappear and a **Mesh** control group (Octree,
Iso threshold, Target faces) takes their place. See
[3D Meshes → From the apps](/guide/mesh#from-the-apps) for the walkthrough.

- **Developer**:
  [Tencent Hunyuan](https://github.com/Tencent-Hunyuan/Hunyuan3D-2)
- **License**: separate Tencent Hunyuan 3D 2.0 and 2.1 terms — see
  [Licence](#licence) below, because it has real restrictions
- **HuggingFace**: [tencent/Hunyuan3D-2](https://huggingface.co/tencent/Hunyuan3D-2),
  [tencent/Hunyuan3D-2mini](https://huggingface.co/tencent/Hunyuan3D-2mini)

## Variants

| Model                       | Steps | Size    | VRAM                      | Notes                                            |
| --------------------------- | ----- | ------- | ------------------------- | ------------------------------------------------ |
| `hunyuan3d-mini-turbo:fp16` | 5     | 3.6 GiB | ~5 GB                     | 0.6B, step-distilled. **The default.**           |
| `hunyuan3d-turbo:fp16`      | 5     | 4.6 GiB | ~6 GB                     | 1.1B, step-distilled                             |
| `hunyuan3d:fp16`            | 30    | 4.6 GiB | ~6 GB                     | 1.1B, undistilled                                |
| `hunyuan3d-2.1:fp16`        | 30    | 6.9 GiB | qualification in progress | 3.3B MoE shape transformer; separate 2.1 licence |

Each is ONE self-contained file carrying the shape transformer, the shape VAE
and an image encoder (DINOv2-large for 2.1, giant for 2.0), which is why a "0.6B" model is still 3.6 GiB
— the vision tower is 1.1B parameters on its own.

No quantized (GGUF or FP8) variants exist for this family upstream.

## Measured on Apple Silicon

`hunyuan3d-mini-turbo:fp16` on an M4 Max (48 GB), fp16, the default decode
chunk, one seed, a background-removed armchair. ComfyUI is the same checkpoint
and seed on the same machine through PyTorch MPS.

| Octree | Wall  | Peak RSS | Mesh                     | ComfyUI wall |
| ------ | ----- | -------- | ------------------------ | ------------ |
| 192    | 77 s  | 7.2 GB   | 145k vertices, 317k tris | 44 s         |
| 256    | 144 s | 7.2 GB   | 264k vertices, 590k tris | 79 s         |
| 320    | 256 s | 7.2 GB   | 417k vertices, 930k tris | 136 s        |

The geometry matches ComfyUI's to a normalised Chamfer distance of 0.011 on
every rung, with bounding-box extents within 5 % and triangle counts within
19 %; two mold seeds differ from each other by 0.0025 on the same scale. mold
is slower than PyTorch here: the volume decode (over 90 % of the wall time) runs
through candle's chunked math attention on Metal rather than a fused kernel,
and the tile size has already been swept — 512-row tiles are the fastest. CUDA
has not been measured yet.

## Hunyuan3D 2.1 shape

```bash
mold pull hunyuan3d-2.1 --accept-license tencent-hunyuan3d-2.1
mold run hunyuan3d-2.1 --image chair.png -o chair.glb
```

The 2.1 checkpoint uses a different transformer with sparse experts and 4,096
shape latents. Accepting the 2.0 terms does not accept the 2.1 terms. The default
remains mini-turbo. CUDA component and full-render validation is recorded in the
campaign qualification ledger; broader parity qualification is still running.

## Getting good results

Prompt quality is irrelevant here. Source image quality is everything.

- **One object, centred, filling most of the frame.** The model reconstructs
  what it can see; a subject occupying a tenth of the frame reconstructs at a
  tenth of the detail.
- **A plain or removed background.** There is no segmentation stage, so a busy
  background is read as geometry. An image with an alpha channel is letterboxed
  on its cutout, which is the best input you can give it.
- **A three-quarter view.** A straight-on photograph gives the model no depth
  cue for the sides.

## Usage

```bash
mold pull hunyuan3d-mini-turbo --accept-license tencent-hunyuan3d-2.0
mold run hunyuan3d-mini-turbo --image chair.png -o chair.glb
```

Higher detail, undistilled tier:

```bash
mold run hunyuan3d --image chair.png --octree 320 -o chair.glb
```

Pipe it straight into a viewer:

```bash
mold run hunyuan3d-mini-turbo --image chair.png --output - | some-gltf-viewer
```

## Controls

| Flag               | Default | What it does                                               |
| ------------------ | ------- | ---------------------------------------------------------- |
| `--octree`         | 256     | Query-grid resolution. The detail knob; **cost is cubic**. |
| `--mesh-threshold` | 0.6     | Iso-level. Lower recovers thin features and adds noise.    |
| `--target-faces`   | none    | Decimate to approximately this triangle count.             |

The threshold is a level on the same `[0, 1]` occupancy scale ComfyUI's
`VoxelToMesh` node thresholds, so a value that works in ComfyUI works here
unchanged. (Internally the VAE's raw logits are mapped through
`(x + 1) / 2`, clamped, before the surface is extracted — 0.6 sits at raw
logit 0.2.)

`--octree` accepts 128, 192, 256, 320 or 384. It is an allowlist rather than a
range because the shape VAE evaluates its occupancy field on `(n + 1)³` points
— 256 is about 17 million — so an arbitrary number between two rungs buys
nothing and can cost you an out-of-memory failure minutes into a render.

## Output

The stored artifact is always **binary glTF** (`.glb`): one self-contained file
with geometry, normals and materials embedded, so a mesh is one library print
exactly like an image or a clip. It gets a rendered poster tile in the gallery,
and it lists, downloads, trashes, restores and reuses its settings like
anything else.

A 3-D model has exactly one deliverable container, so a request naming a raster
format is **pinned** to `glb` rather than refused. `-o` is the exception: a
filename ending in `.png` or `.mp4` names a file this render will not write, so
it is refused before any weight is read.

### Export as OBJ, STL or PLY

Everything except GLB is an **export** — a transcode of a mesh that already
exists, never a stored format — because each of them loses something the glTF
carries.

```bash
mold library export chair.glb --format stl
mold library export chair.glb --format obj -o ~/chair.obj
```

| Format | Carries                                             | Reach for it when                      |
| ------ | --------------------------------------------------- | -------------------------------------- |
| `glb`  | Geometry, normals, UVs, materials, embedded texture | Anything. This is the stored file.     |
| `obj`  | Positions, normals, UVs. No materials.              | Blender, MeshLab, most DCC importers.  |
| `stl`  | Triangles and one normal each. No UVs, no colour.   | 3-D printing and CAD.                  |
| `ply`  | Positions and per-vertex normals, vertices shared.  | Point-and-mesh tooling, research code. |

The gallery file is never renamed or replaced. The same conversions are on
`POST /api/gallery/export/:filename` and the `export_mesh` MCP tool, and a host
advertises what it can convert on `/api/capabilities.mesh.export_formats`.
`--size-mm`, `--up-axis` and `--origin` make an STL or PLY print-ready for a
slicer or DCC tool by default; see
[Print-ready exports](/guide/mesh#print-ready-exports).

## Licence

Tencent's community licence for these weights has two clauses worth reading
before you build on them:

> THIS LICENSE AGREEMENT DOES NOT APPLY IN THE EUROPEAN UNION, UNITED KINGDOM
> AND SOUTH KOREA

and a separate written licence from Tencent is required once your products
exceed **1 million monthly active users**.

Outputs are unencumbered — _"Tencent claims no rights in Outputs You
generate"_ — so the meshes themselves are yours.

Because of this, mold refuses to download the weights until you have recorded
an acceptance:

```bash
mold licenses                      # read the terms
mold licenses accept tencent-hunyuan3d-2.0   # agree without downloading
mold pull hunyuan3d --accept-license tencent-hunyuan3d-2.0
```

This is the same gate the InsightFace face models use. It exists so a
server-side auto-pull can never quietly acquire restricted weights on your
behalf.

## Not yet supported

Texture and PBR material generation, multi-view input and
text-to-3D are tracked in
[#1496](https://github.com/utensils/mold/issues/1496). Today's output is
geometry only.

## Accepting the licence from the apps

The terms are not CLI-only. Selecting a Hunyuan3D model on web, desktop or
mobile raises the shared licence dialog before anything downloads, as does
installing it from the Models page. Acceptance is recorded per Mold data root,
so it is stored on the host that will fetch the weights — on desktop, pick that
machine in **Settings → Model licenses**, and on mobile it is whichever host you
have selected.

The 2.1 shape model and texturing weights (`hunyuan3d-paint`) share a _separate_ Tencent 2.1
agreement and must be accepted on their own. They install today, but the PBR
paint engine is not implemented yet, so they satisfy the gate without rendering.
