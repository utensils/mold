# LTX-2 / LTX-2.3

LTX-2 is Lightricks' joint audio-video family. In mold it is exposed as a
separate `ltx2` family from the older `ltx-video` checkpoints, with defaults
aimed at synchronized MP4 output and the upstream two-stage / distilled
pipelines.

::: warning Current implementation
LTX-2 currently runs through the upstream Lightricks Python pipelines via a
small bridge script. Local runs therefore require `python3`, `uv`, `ffmpeg`,
and a checked-out upstream tree at `tmp/LTX-2-upstream`.
:::

## Supported Models

| Model | Path | Notes |
| ----- | ---- | ----- |
| `ltx-2-19b-dev:fp8` | Two-stage | Highest-quality published 19B FP8 checkpoint |
| `ltx-2-19b-distilled:fp8` | Distilled | Fastest 19B path, recommended default |
| `ltx-2.3-22b-dev:fp8` | Two-stage | Highest-quality 22B FP8 checkpoint |
| `ltx-2.3-22b-distilled:fp8` | Distilled | Fastest 22B path |

## What Works

- Text-to-audio+video with synchronized MP4 output
- First-frame image-to-video via `--image`
- Audio-to-video via `--audio-file`
- Keyframe interpolation via repeatable `--keyframe`
- Retake / partial regeneration via `--video` + `--retake`
- IC-LoRA and stacked LoRAs via repeatable `--lora`
- Camera-control preset names for the published LTX-2 19B camera LoRAs

## Current Constraints

- Default output is `mp4` for this family. `gif`, `apng`, and `webp` are also
  supported, but they are treated as silent exports.
- `x2` spatial upscaling is wired. `x1.5` is not.
- Temporal upscaling is not wired yet.
- Camera-control preset aliases are currently published for LTX-2 19B only. For
  LTX-2.3, pass an explicit `.safetensors` path.
- The Gemma text encoder source is gated on Hugging Face, so you must have
  access approved before `mold pull` will complete.
- On 24 GB Ada GPUs such as the RTX 4090, mold runs the bridge with layer
  streaming and the upstream `fp8-cast` path rather than Hopper-only
  `fp8-scaled-mm`.

## Examples

```bash
# Fast default: text to synchronized MP4
mold run ltx-2-19b-distilled:fp8 \
  "cinematic close-up of rain on a neon taxi window" \
  --frames 97 \
  --format mp4

# Audio-to-video
mold run ltx-2-19b-distilled:fp8 \
  "paper cutout forest reacting to a violin solo" \
  --audio-file ./solo.wav \
  --format mp4

# Keyframe interpolation
mold run ltx-2-19b-distilled:fp8 \
  "a drone shot over volcanic cliffs" \
  --pipeline keyframe \
  --frames 97 \
  --keyframe 0:./start.png \
  --keyframe 96:./end.png

# Camera-control preset
mold run ltx-2-19b-distilled:fp8 \
  "a lantern-lit cave entrance" \
  --camera-control dolly-in \
  --format mp4
```

## Notes

- `--audio` and `--no-audio` control whether the returned MP4 keeps the audio
  track. If you explicitly choose `gif`, `apng`, or `webp`, mold exports a
  silent animation.
- `--lora` is repeatable for this family. The single legacy `lora` request
  field is still populated for backward compatibility, but the LTX-2 bridge uses
  the stacked `loras` list.
- `ffmpeg` is used for muxing, silent-export stripping, thumbnails, and GIF
  previews.
