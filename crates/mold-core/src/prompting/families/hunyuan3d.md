# Hunyuan3D prompting

Manifest family: `hunyuan3d`.

## Prompt style

Write no prompt. There is no text encoder anywhere in this family. The source
image or named multiview set is the entire conditioning; nothing typed in the prompt field reaches
the model, and a request without images is refused rather than answered
from nothing. The {{word_limit}}-word budget is therefore unused: spend the
effort on the image instead. `mold expand` and `mold remix` say so and answer
with the image advice below instead of calling a language model.

## Syntax

Nothing in the text field reaches the model, so there is no weighting, no
negative prompt, no quoted text, and no reference addressing to write. The one
image is passed as the source, not named in prose.

## Generation context

Three properties of the image move the result, and none of them are prose.

- One object, centred, filling most of the frame. The model reconstructs what
  it can see; a subject occupying a tenth of the frame reconstructs at a tenth
  of the detail.
- A plain or removed background. There is no segmentation stage, so a busy
  background is read as geometry. An image with an alpha channel is the best
  input, because mold letterboxes on the cutout.
- A three-quarter view. A straight-on photograph gives no depth cue for the
  sides.

## Examples

Input: 3-D model of my dining chair, photo attached

Output: No prompt. Supply chair.png cropped so the chair fills the frame,
background removed to alpha, shot from a three-quarter angle.

Input: turn this asset concept into a mesh

Output: No prompt. Supply concept.png with the single object centred on a plain
ground and every other prop cropped away.

## Pitfalls

- Frames, fps, masks, ControlNet, and an explicit canvas are refused for this
  family rather than ignored.
- Output is always binary glTF, so `-o` must name a `.glb` file or `-` for
  stdout; a raster, video, or audio extension is refused before any weight is
  read. An explicit `png` in a request is coerced to `glb`, not refused.
- OBJ, OBJ+PBR ZIP, STL, and PLY exist only as gallery exports of the stored glTF, never as
  generation targets, because each loses something the glTF carries.
- The same picture gives a different mesh here than in ComfyUI. mold prepares
  the image the way Tencent's `ImageProcessorV2` does: crop to the alpha
  bounding box, then letterbox on a white square, so nothing is cut away.
  ComfyUI's `clip_preprocess` drops the alpha channel and, with CLIP Vision
  Encode's default `crop: center`, centre-crops the shorter side to a square,
  so an off-centre or wide subject loses its edges there and keeps them here
  (`crop: none` squashes to a square instead, distorting rather than
  cropping); a threshold tuned on one is a fair start on the other, a crop
  is not.
- The web, desktop, and mobile apps' export menu offers the same OBJ, OBJ+PBR ZIP, STL,
  and PLY geometry exports and the GIF, APNG, and WebP turntables as
  `mold library export` and the `export_mesh` MCP tool.
- Text-to-3D is not supported. Select a 2mv tier for named front, left, back,
  and right inputs; every other shape tier takes one source image.
- Detail is bought with the octree resolution and its cost is cubic.

## CLI

```bash
# The default tier: 0.6B, step-distilled, ~5 GB VRAM
mold run hunyuan3d-mini-turbo --image chair.png -o chair.glb

# 2.1 shape uses the same image-only prompt contract and separate 2.1 terms
mold run hunyuan3d-2.1 --image chair.png -o chair.glb

# Undistilled 1.1B, 30 guided steps, higher detail
mold run hunyuan3d --image chair.png --octree 320 -o chair.glb

# Recover thin features by lowering the surface threshold
mold run hunyuan3d-turbo --image lamp.png --mesh-threshold 0.4 -o lamp.glb

# Named views keep semantic slots; any non-empty subset is accepted
mold run hunyuan3d-2mv-turbo --front front.png --left left.png --back back.png -o object.glb

# Decimate to a face budget; matting and delight prepare the source image
mold run hunyuan3d-2.1 --image chair.png --target-faces 40000 --matting auto -o chair.glb
mold run hunyuan3d-2.1 --image chair.png --matting on --delight -o chair.glb

# Paint PBR textures as well as geometry (the host must advertise them)
mold run hunyuan3d-2.1 --image chair.png --texture -o chair.glb
mold run hunyuan3d-2.1 --image chair.png --texture --texture-resolution 2048 -o chair.glb

# Export a saved mesh from the gallery as STL, OBJ+PBR ZIP, or PLY
mold library export chair.glb --format stl -o chair.stl

# Preserve the painted material maps in a portable OBJ bundle
mold library export chair.glb --format zip -o chair.zip

# Share a turntable: the poster spun a full turn as an animated GIF (or apng, webp)
mold library export chair.glb --format gif
mold library export chair.glb --format gif --playback bounce --repeat once --frames 24

# The durable form: every stage retained, resumable, followed to settlement
mold mesh-workflow create --mesh chair.glb --image chair-albedo.png --follow
```

`--texture` asks for PBR maps beside the geometry and needs the paint bundle;
without it the request is refused rather than answered with a bare white
mesh. `--texture-resolution` sets the atlas edge (1024, 2048 or 4096) and only
means anything with `--texture`. `--matting` decides background removal before
shape conditioning — `auto` preserves useful alpha and removes opaque
backgrounds, `on` recomputes every cutout, `off` keeps the pixels — and
`--delight` runs the fixed lighting and highlight removal stage after matting
and before shape or paint, only where the profile advertises it.

`--octree`, `--mesh-threshold`, and `--target-faces` are the three geometry
controls, and the model's generation profile is the authority on their
values: its `capabilities.mesh` block advertises the octree allowlist and
default, the threshold range and default, and the face bounds, so read them
from the profile (`/api/models`) rather than from this page. `--octree` is
the detail knob and its cost is cubic. `--mesh-threshold` moves the extracted
surface: lower recovers thin features and adds noise; it is the same `[0, 1]`
occupancy scale ComfyUI's `VoxelToMesh` thresholds, so a value tuned there
carries over. `--target-faces` decimates after extraction. A geometry-only
export keeps the raw surface when it is absent; a TEXTURED render decimates
to the profile's `capabilities.mesh.target_faces_texture_default` instead,
mirroring Tencent's own paint pipeline, because UV unwrapping is superlinear
in triangle count. `mold library export`, the `export_mesh` MCP tool, and the
gallery export menu all transcode the same stored `.glb`.

`mold run` is one render. `mold mesh-workflow` is the durable form of the same
work: each stage is admitted as its own generation and keeps its own retained
artifact, the job survives a server restart, and a resume picks up at the
first unfinished stage. It takes the same geometry, matting and delight
controls, infers its mode from what it is given (`--prompt` to render a
picture and reconstruct it, `--mesh` with `--image` to paint a supplied mesh,
`--mesh` alone to rebuild one), and is remote by construction — the job lives
in one machine's data root.

## Sources

- https://github.com/Tencent-Hunyuan/Hunyuan3D-2
  (`hy3dgen/shapegen/preprocessors.py`, `ImageProcessorV2.recenter`: the
  alpha-bounding-box crop and white letterbox mold mirrors)
- https://github.com/comfyanonymous/ComfyUI (`comfy/clip_model.py`
  `clip_preprocess`: the centre crop; `comfy_extras/nodes_hunyuan3d.py`
  `VoxelToMesh`: the threshold scale)
- Best practice: the centred-subject, cutout-background, three-quarter-view
  image advice is community practice, not a published upstream rule.

- https://github.com/Tencent-Hunyuan/Hunyuan3D-2.1/tree/82920d643c0dc2f7bfd7255f45f62d386edfe60c/hy3dshape
  — 2.1 shape retains image conditioning without a text encoder.
