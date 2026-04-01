# Image-to-Image

Transform existing images with a text prompt. Source images auto-resize to fit
the target model's native resolution (preserving aspect ratio).

## Basic img2img

```bash
# Stylize a photo
mold run "oil painting style" --image photo.png

# Control how much changes (0.0 = no change, 1.0 = full denoise)
mold run "watercolor" --image photo.png --strength 0.5

# Pipe an image through
cat photo.png | mold run "sketch style" --image - | viu -

# Override auto-resize with explicit dimensions
mold run "pencil sketch" --image photo.png --width 768 --height 512
```

### Strength Guide

| Strength | Effect                              |
| -------- | ----------------------------------- |
| `0.0`    | No change                           |
| `0.3`    | Subtle adjustments                  |
| `0.5`    | Balanced — noticeable but faithful  |
| `0.75`   | Strong transformation (default)     |
| `1.0`    | Full txt2img (ignores source image) |

## Inpainting

Selectively edit parts of an image using a mask. White pixels are repainted,
black pixels are preserved.

```bash
mold run "a golden retriever" --image park.png --mask mask.png
```

::: tip Mask format The mask must be the same dimensions as the source image.
Use any image editor to paint white on areas you want regenerated. :::

## ControlNet (SD1.5)

Guide generation with a control image (edge map, depth map, pose):

```bash
# First download a ControlNet model
mold pull controlnet-canny-sd15

# Use with a control image
mold run sd15:fp16 "a futuristic city" \
  --control edges.png \
  --control-model controlnet-canny-sd15

# Adjust conditioning scale (0.0–2.0, default 1.0)
mold run sd15:fp16 "interior design" \
  --control depth.png \
  --control-model controlnet-depth-sd15 \
  --control-scale 0.8
```

Available ControlNet models:

| Model                           | Input      |
| ------------------------------- | ---------- |
| `controlnet-canny-sd15:fp16`    | Edge maps  |
| `controlnet-depth-sd15:fp16`    | Depth maps |
| `controlnet-openpose-sd15:fp16` | Pose data  |
