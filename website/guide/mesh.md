# 3D Meshes

mold generates 3-D meshes from a single photograph using
[Hunyuan3D](/models/hunyuan3d). The result is a durable library print like any
other — it lands in the gallery, gets a poster tile, and lists, downloads,
trashes, restores and reuses its settings exactly like an image or a clip.

## Quick start

```bash
mold licenses                                                   # read the terms
mold pull hunyuan3d-mini-turbo --accept-license tencent-hunyuan3d-2.0
mold run hunyuan3d-mini-turbo --image chair.png -o chair.glb
```

`mold pull` refuses until the licence is accepted. That is deliberate: the
Tencent community licence does not apply in the EU, the UK or South Korea, so
mold will not acquire the weights on your behalf without an explicit
acceptance. See [Licence](/models/hunyuan3d#licence).

## What makes a good input

There is **no text encoder in this family**. The image is the entire
conditioning, and the prompt — if you pass one — is recorded as provenance and
never read. **You do not need to write one**: every surface reads the prompt
requirement off the model's own generation profile, so an empty prompt is
admitted here and refused everywhere it still means something.

```bash
mold run hunyuan3d-mini-turbo --image chair.png -o chair.glb   # no prompt
```

The usual prompt-engineering advice does not apply, and the image advice
matters more than usual:

- One object, centred, filling most of the frame.
- A plain or removed background. An image with an alpha channel is the best
  input; mold letterboxes on the cutout.
- A three-quarter view, not a straight-on one.

A request without a source image is refused rather than answered from nothing.

Prompt expansion follows the same rule. `mold expand`, `mold remix`, the
Expand and Remix controls on every surface, and the MCP `expand_prompt` /
`remix_prompt` tools do not call a language model for this family: the one
answer is the guide's advice above on preparing the image, and `--expand` on
a mesh run is skipped rather than rewriting the recorded prompt.

The same picture also meshes differently here than in ComfyUI. mold prepares
the image the way Tencent's `ImageProcessorV2` does — crop to the alpha
bounding box, then letterbox on a white square, so nothing is cut away —
while ComfyUI's `clip_preprocess` drops the alpha channel and centre-crops the
shorter side to a square. An off-centre or wide subject loses its edges there
and keeps them here.

## Controls

```bash
mold run hunyuan3d --image chair.png \
  --octree 320 \
  --mesh-threshold 0.55 \
  --target-faces 50000 \
  -o chair.glb
```

| Flag               | Default | What it does                                               |
| ------------------ | ------- | ---------------------------------------------------------- |
| `--octree`         | 256     | Query-grid resolution. The detail knob; **cost is cubic**. |
| `--mesh-threshold` | 0.6     | Iso-level. Lower recovers thin features and adds noise.    |
| `--target-faces`   | none    | Decimate to approximately this triangle count.             |

`--mesh-threshold` is a level on the same `[0, 1]` occupancy scale ComfyUI's
`VoxelToMesh` node uses, so a value tuned there carries over unchanged.

`--octree` accepts 128, 192, 256, 320 or 384 — an allowlist, not a range,
because the model evaluates its occupancy field on `(n + 1)³` points and an
arbitrary value between two rungs buys nothing while risking an
out-of-memory failure part-way through a render.

Flags that describe a raster or a timeline are **refused**, not ignored:
`--frames`, `--fps`, a mask, a ControlNet and an explicit canvas all name
something a mesh does not have, and silently dropping them would make "Reuse
settings" replay numbers that never applied.

## Output format

The stored artifact is always binary glTF (`.glb`) — one self-contained file
with geometry, normals and materials embedded. That is what makes a mesh a
single library row, with no special-case handling anywhere downstream.

You do not have to ask for it. A 3-D model has exactly one deliverable
container, so a request naming `png` is **pinned** to `glb` rather than
refused — an older client that always sends a raster format still gets its
mesh. `-o` is the one place that is an error instead: a filename ending in
`.png`, `.mp4` or `.wav` names a file this render will not write, and mold says
so before a weight is read rather than after a two-minute render.

## Export as OBJ, STL or PLY

Everything except GLB is an **export**: a transcode of geometry that already
exists, never a generation target, because each container loses something the
stored glTF carries.

```bash
mold library export chair.glb --format stl               # writes chair.stl
mold library export chair.glb --format obj -o ~/chair.obj
mold library export chair.glb --format ply --output -    # to stdout
```

| Format | Carries                                             | Reach for it when                      |
| ------ | --------------------------------------------------- | -------------------------------------- |
| `glb`  | Geometry, normals, UVs, materials, embedded texture | Anything. This is the stored file.     |
| `obj`  | Positions, normals, UVs. No materials.              | Blender, MeshLab, most DCC importers.  |
| `stl`  | Triangles and one normal each. No UVs, no colour.   | 3-D printing and CAD.                  |
| `ply`  | Positions and per-vertex normals, vertices shared.  | Point-and-mesh tooling, research code. |

The gallery file is never renamed or replaced — an export writes a copy where
you asked for it. The same conversions are available on every surface: the API
(`POST /api/gallery/export/:filename`) and the `export_mesh` MCP tool. A host
advertises what it can convert on `/api/capabilities.mesh.export_formats`.

USDZ is tracked separately; it is the format Apple's AR Quick Look wants and it
carries textures, so it belongs with the texturing work rather than here.

## Piping

`mold run` is pipe-friendly here as everywhere:

```bash
mold run hunyuan3d-mini-turbo --image chair.png --output - > chair.glb
cat chair.png | mold run hunyuan3d-mini-turbo --image - -o chair.glb
```

## In the TUI

Pick a Hunyuan3D model in `mold tui`'s Create form and the form reshapes
itself from the model's generation profile rather than from its name:

- **Source image** is the only conditioning row. Strength, Mask and the
  Negative prompt disappear because the profile advertises no strength
  (`supports_strength` is false), a hidden mask, and no negative prompt.
- **Advanced ▸ 3-D mesh** appears with three rows — **Octree** (`◀▶` walks
  the advertised allowlist), **Iso threshold** (0.05 per press inside the
  advertised range) and **Target faces** (10 000 per press; stepping below
  the minimum turns decimation off). Each row reads `default` until touched,
  showing the profile's own default, and an untouched row sends nothing so
  the recipe's defaults apply.
- **Format** is pinned to `glb`; `◀▶` cannot walk it onto a raster container
  the server would only pin straight back.
- **Generate** submits with an empty prompt, because the profile advertises
  `prompt.mode: ignored`; the same gate still refuses an empty prompt on a
  text model.

A finished mesh saves `mold-<model>-<timestamp>.glb` beside your other
prints, caches its poster where the Library looks for thumbnails, shows the
poster in the Preview panel, and captions it with
`49,152 tris · 24,576 verts · 1.00×0.80×0.60`.

In the **Library**, a `.glb` tile shows its poster (fetched from the owning
machine's thumbnail route; never the geometry through a raster decoder), and
`x` opens an export picker offering OBJ, STL and PLY — the list the owning
machine advertises on `capabilities.mesh.export_formats`, or every container
for a print that lives only on this machine. The converted copy is written
beside your other saves as `<print>.<ext>` and its path is shown when it
lands; the gallery file is untouched.

## In Discord

`/generate` with a Hunyuan3D `model` and a `source_image` attachment renders
a mesh. The `prompt` option is optional whenever a source image is attached
(Discord cannot make an option optional per model, and a source image is
exactly what image-to-video and image-to-3D have in common), the
`video_format` option is ignored because the family has one deliverable
container, and the reply embeds the rendered poster with the `.glb` attached
beside it as a download. The summary reads **Mesh Generated** with the
triangle and vertex counts, the bounds, the format and the seed. A mesh
larger than Discord's upload limit posts the poster alone with a note saying
to fetch the `.glb` from the gallery.

## In the gallery

A mesh cannot be decoded by a raster thumbnailer, so mold renders a **poster
PNG** at save time and stores it in the shared thumbnail cache. Grids and the
TUI cell show that poster; only the lightbox loads the geometry itself. If a
poster is missing, surfaces fall back to a placeholder rather than trying to
draw glTF bytes as a picture.

## What is not supported yet

Texture and PBR material generation, the Hunyuan3D 2.1 shape model, multi-view
input, and text-to-3D are tracked in
[#1496](https://github.com/utensils/mold/issues/1496). Today's output is
geometry only — a clean white mesh.
