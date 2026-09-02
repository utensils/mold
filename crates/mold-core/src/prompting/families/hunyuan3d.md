# Hunyuan3D prompting

Manifest family: `hunyuan3d`.

## Prompt style

Write no prompt. There is no text encoder anywhere in this family. The source
image is the entire conditioning, nothing typed in the prompt field reaches
the model, and a request without an image is refused rather than answered
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
- OBJ, STL, and PLY exist only as gallery exports of the stored glTF, never as
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
- Texturing, the 2.1 shape model, multi-view input, and text-to-3D are not
  supported, so today's result is geometry only.
- Detail is bought with the octree resolution and its cost is cubic.

## CLI

```bash
# The default tier: 0.6B, step-distilled, ~5 GB VRAM
mold run hunyuan3d-mini-turbo --image chair.png -o chair.glb

# Undistilled 1.1B, 30 guided steps, higher detail
mold run hunyuan3d --image chair.png --octree 320 -o chair.glb

# Recover thin features by lowering the surface threshold
mold run hunyuan3d-turbo --image lamp.png --mesh-threshold 0.4 -o lamp.glb

# Export a saved mesh from the gallery as STL, OBJ, or PLY
mold library export chair.glb --format stl -o chair.stl

# Share a turntable: the poster spun a full turn as an animated GIF (or apng, webp)
mold library export chair.glb --format gif
mold library export chair.glb --format gif --playback bounce --repeat once --frames 24
```

`--octree`, `--mesh-threshold`, and `--target-faces` are the three mesh
controls, and the model's generation profile is the authority on their
values: its `capabilities.mesh` block advertises the octree allowlist and
default, the threshold range and default, and the face bounds, so read them
from the profile (`/api/models`) rather than from this page. `--octree` is
the detail knob and its cost is cubic. `--mesh-threshold` moves the extracted
surface: lower recovers thin features and adds noise; it is the same `[0, 1]`
occupancy scale ComfyUI's `VoxelToMesh` thresholds, so a value tuned there
carries over. `--target-faces` decimates after extraction and is absent
until asked for. `mold library export`, the `export_mesh` MCP tool, and the
gallery export menu all transcode the same stored `.glb`.

## Sources

- https://github.com/Tencent-Hunyuan/Hunyuan3D-2
  (`hy3dgen/shapegen/preprocessors.py`, `ImageProcessorV2.recenter`: the
  alpha-bounding-box crop and white letterbox mold mirrors)
- https://github.com/comfyanonymous/ComfyUI (`comfy/clip_model.py`
  `clip_preprocess`: the centre crop; `comfy_extras/nodes_hunyuan3d.py`
  `VoxelToMesh`: the threshold scale)
- Best practice: the centred-subject, cutout-background, three-quarter-view
  image advice is community practice, not a published upstream rule.
