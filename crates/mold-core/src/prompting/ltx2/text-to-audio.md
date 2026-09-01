# LTX-2 text-to-audio

## Prompt style

Describe the audible result rather than a camera scene: sound sources, their
sequence, the space, distance, dynamics, and texture. For speech, give the
exact short line and describe voice and delivery. Keep dialogue, ambience,
effects, and music in separate clauses so their roles do not conflict.

## Generation context

Duration comes from `--frames` and `--fps`, so 121 frames at 24 fps is about
five seconds. Write only as much sound as fits that span. Text-to-audio
rejects every conditioning input and renders no picture, so describe nothing
visual.

## Examples

Input: rain on a roof

Output: Heavy rain drums on a tin roof at a steady rate as water spills from a
gutter to the left. Distant thunder rolls twice, seconds apart, with no music.

## CLI

Inspect `mold --help` and the selected host's capabilities for the current
text-to-audio invocation. Verify the returned artifact is audio before
reporting success.

```bash
# Audio only, 16-bit stereo WAV; duration is frames divided by fps
mold run ltx-2.3-22b-dev:fp8 "heavy rain on a tin roof, distant thunder" --pipeline t2a --frames 121 --fps 24 --output rain.wav
```

## Sources

- https://github.com/Lightricks/LTX-2 (audio VAE and vocoder, `packages/ltx-pipelines`)
