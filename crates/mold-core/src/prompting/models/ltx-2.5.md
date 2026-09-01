# LTX-2.5 22B

## Generation context

Both base names resolve to the `int8-conv` pack by default. The prompt is
encoded by a packed Gemma 4 text encoder this family requires. Audio renders
by default on MP4 output; `--audio` states it explicitly and `--no-audio`
suppresses it. `--predict-duration` lets a qualified duration head choose a
one to twenty second clip in place of `--frames`, so describe an action whose
length is not fixed.

## Pitfalls

The distilled recipe fixes guidance at 1.0. Do not override it.

## CLI

```bash
# Distilled default pack, duration head picks the clip length
mold run ltx-2.5-22b-distilled "a complete product reveal" --predict-duration --fps 24 --audio --format mp4

# Base checkpoint at an explicit length
mold run ltx-2.5-22b-dev "a slow flyover of a salt flat at dawn" --frames 121 --fps 24 --audio
```

## Sources

- https://github.com/Lightricks/LTX-2 (LTX-2.5 checkpoints and duration head)
- https://huggingface.co/Lightricks/LTX-2.5
