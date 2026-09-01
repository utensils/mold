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
never read. So the usual prompt-engineering advice does not apply, and the
image advice matters more than usual:

- One object, centred, filling most of the frame.
- A plain or removed background. An image with an alpha channel is the best
  input; mold letterboxes on the cutout.
- A three-quarter view, not a straight-on one.

A request without a source image is refused rather than answered from nothing.

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

OBJ is available as a gallery **export**, never as a generation target: an
`.obj` on its own carries neither materials nor textures, so mold does not
publish one as though it were complete.

## Piping

`mold run` is pipe-friendly here as everywhere:

```bash
mold run hunyuan3d-mini-turbo --image chair.png --output - > chair.glb
cat chair.png | mold run hunyuan3d-mini-turbo --image - -o chair.glb
```

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
