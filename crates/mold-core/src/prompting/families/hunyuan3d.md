# Hunyuan3D prompting

Manifest family: `hunyuan3d`.

## Prompt style

Write no prompt. There is no text encoder anywhere in this family. The source
image is the entire conditioning, a prompt is recorded as provenance and never
read, and a request without an image is refused rather than answered from
nothing. The {{word_limit}}-word budget is therefore unused: spend the effort
on the image instead.

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
- Output is always binary glTF; OBJ exists only as a gallery export.
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
```

`--octree` (128 | 192 | 256 | 320 | 384, default 256) is the detail knob and
its cost is cubic. `--mesh-threshold` (default 0.6) moves the extracted
surface: lower recovers thin features and adds noise.

## Sources

- https://github.com/Tencent-Hunyuan/Hunyuan3D-2
- Best practice: the centred-subject, cutout-background, three-quarter-view
  image advice is community practice, not a published upstream rule.
