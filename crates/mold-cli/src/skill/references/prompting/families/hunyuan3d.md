# Hunyuan3D prompting

Manifest family: `hunyuan3d`.

**There is no text encoder anywhere in this family.** The source image is the
entire conditioning; a prompt is recorded as provenance and never read, and a
request without an image is refused rather than answered from nothing. Do not
author a prompt for a 3-D request — spend the effort on the image.

What actually moves the result:

- One object, centred, filling most of the frame. The model reconstructs what
  it can see; a subject occupying a tenth of the frame reconstructs at a tenth
  of the detail.
- A plain or removed background. There is no segmentation stage, so a busy
  background is read as geometry. An image with an alpha channel is the best
  input — mold letterboxes on the cutout.
- A three-quarter view. A straight-on photograph gives no depth cue for the
  sides.

```bash
# The default tier: 0.6B, step-distilled, ~5 GB VRAM.
mold run hunyuan3d-mini-turbo --image chair.png -o chair.glb

# Undistilled 1.1B, 30 guided steps, higher detail.
mold run hunyuan3d --image chair.png --octree 320 -o chair.glb
```

`--octree` (128 | 192 | 256 | 320 | 384, default 256) is the detail knob and
its cost is CUBIC. `--mesh-threshold` (default 0.6) moves the extracted
surface: lower recovers thin features and adds noise. Output is always binary
glTF; OBJ exists only as a gallery export. `--frames`, `--fps`, masks,
ControlNet and an explicit canvas are REFUSED for this family rather than
ignored.

Texturing, the 2.1 shape model, multi-view input and text-to-3D are not
supported yet; today's output is geometry only.
