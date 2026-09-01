# Hunyuan3D

Image-to-3D. Give it one photograph and it returns a triangle mesh.

Unlike every other family in mold, Hunyuan3D has **no text encoder at all**.
The source image is the only conditioning, so a prompt is recorded as
provenance and never read, and a request without an image is refused rather
than answered from an empty prompt.

- **Developer**:
  [Tencent Hunyuan](https://github.com/Tencent-Hunyuan/Hunyuan3D-2)
- **License**: Tencent Hunyuan 3D 2.0 Community License — see
  [Licence](#licence) below, because it has real restrictions
- **HuggingFace**: [tencent/Hunyuan3D-2](https://huggingface.co/tencent/Hunyuan3D-2),
  [tencent/Hunyuan3D-2mini](https://huggingface.co/tencent/Hunyuan3D-2mini)

## Variants

| Model                       | Steps | Size    | VRAM  | Notes                                   |
| --------------------------- | ----- | ------- | ----- | --------------------------------------- |
| `hunyuan3d-mini-turbo:fp16` | 5     | 3.6 GiB | ~5 GB | 0.6B, step-distilled. **The default.**  |
| `hunyuan3d-turbo:fp16`      | 5     | 4.6 GiB | ~6 GB | 1.1B, step-distilled                    |
| `hunyuan3d:fp16`            | 30    | 4.6 GiB | ~6 GB | 1.1B, undistilled — best shape fidelity |

Each is ONE self-contained file carrying the shape transformer, the shape VAE
and a DINOv2-giant image encoder, which is why a "0.6B" model is still 3.6 GiB
— the vision tower is 1.1B parameters on its own.

No quantized (GGUF or FP8) variants exist for this family upstream.

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

OBJ is available as a gallery **export**, never as a stored format — an `.obj`
alone carries neither materials nor textures, so mold does not publish one as
if it were complete.

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

Texture and PBR material generation, the 2.1 shape model, multi-view input and
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

Texturing weights (`hunyuan3d-paint`) are covered by a _separate_ Tencent 2.1
agreement and must be accepted on their own. They install today, but the PBR
paint engine is not implemented yet, so they satisfy the gate without rendering.
