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
conditioning, and nothing typed in the prompt field reaches the model. **You
do not need to write one** on the CLI, the API, the TUI, Discord or MCP: each
reads the prompt requirement off the model's own generation profile, so an
empty prompt is admitted here and refused everywhere it still means something.
The web, desktop and mobile apps still apply the legacy prompt rule until the
GUI release wires the profile in.

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
while ComfyUI's `clip_preprocess` drops the alpha channel and, with CLIP
Vision Encode's default `crop: center`, centre-crops the shorter side to a
square (`crop: none` squashes to a square instead, distorting rather than
cropping). An off-centre or wide subject loses its edges there and keeps them
here.

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
you asked for it. The same conversions are available from the API
(`POST /api/gallery/export/:filename`), the TUI's export picker, and the
`export_mesh` MCP tool. A host
advertises what it can convert on `/api/capabilities.mesh.export_formats`.

USDZ is tracked separately; it is the format Apple's AR Quick Look wants and it
carries textures, so it belongs with the texturing work rather than here.

## Share a turntable

Nothing outside a 3-D tool opens a `.glb`, and the gallery poster shows one
view. A **turntable** is that poster set spinning: the same camera, lighting
and slate background, swept a full turn around the mesh and written as an
animated GIF, APNG or WebP you can drop into a chat, a README or a browser.
The first frame is the poster itself.

```bash
mold library export chair.glb --format gif                       # chair.gif: 36 frames, 10 fps, 512 px, loops
mold library export chair.glb --format gif --playback bounce --repeat once
mold library export chair.glb --format webp --frames 72 --fps 24 --max-dimension 768
mold library export chair.glb --format apng -o chair-turntable.png
```

| Flag              | Values            | Default   | Meaning                                                                                    |
| ----------------- | ----------------- | --------- | ------------------------------------------------------------------------------------------ |
| `--playback`      | `loop`, `bounce`  | `loop`    | GIF only. `loop` is one seamless full turn; `bounce` sweeps half a turn and plays it back. |
| `--repeat`        | `forever`, `once` | `forever` | GIF only. `once` plays through and rests on the final frame.                               |
| `--max-dimension` | 240 to 2160       | 512       | Frame edge in pixels; frames are square like the poster.                                   |
| `--frames`        | 8 to 180          | 36        | Views rendered around the mesh. 36 is a 10° step; 72 is smoother and twice the size.       |
| `--fps`           | 1 to 30           | 10        | Playback rate. 36 frames at 10 fps is a 3.6 s turn.                                        |

The two sweeps are shaped for how the encoders play them back. A **loop**
renders one full turn whose last frame stops one step short of the first, so
the wrap from last to first is a step like any other rather than the poster
held twice. A **bounce** renders half a turn, first frame to last inclusive;
the GIF encoder appends the interior frames in reverse, so the animation
swings out to the far side and back, and the reversal reads as deliberate
instead of a full turn snapping into reverse the moment it comes round.
Bounce and `--repeat once` are GIF contracts — APNG and WebP always loop —
exactly as they are for a video export. A turntable is a **render**, not the
mesh: it carries no geometry, and the flags are refused on a geometry format
rather than ignored.

The same options are on `POST /api/gallery/export/:filename` (`playback`,
`repeat`, `max_dimension`, `frames`, `fps`, the video export's own field
names) and the `export_mesh` MCP tool. A host lists `gif`, `apng` and — on a
build with the `webp` feature — `webp` in `capabilities.mesh.export_formats`
beside the geometry containers, so a client learns what it can ask for
without trying. Rendering is pure CPU on the serving host; 36 frames at 512 px
take well under a second, and the frame buffer is capped at the same 256 MiB
the video export allows, so 180 frames at the largest size is a `422` naming
the two flags that bring it under.

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
`x` opens an export picker offering OBJ, STL, PLY and the turntable formats
(GIF, APNG, and WebP on a build that encodes it) — the list the owning
machine advertises on `capabilities.mesh.export_formats`, or the same set
from the in-process writer for a print that lives only on this machine,
rendered through the same code the server uses. The picker has no turntable
knobs: it renders at the defaults (one full turn, 36 frames, 512 px, 10 fps,
looping) and its hint says so, pointing at `mold library export` for bounce,
once, or other sizes. The converted copy is written beside your other saves
as `<print>.<ext>` (an APNG as `.png`) and its path is shown when it lands;
the gallery file is untouched.

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
