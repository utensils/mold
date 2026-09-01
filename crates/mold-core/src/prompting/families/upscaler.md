# Upscaler prompting

Manifest family: `upscaler`.

## Prompt style

Write no prompt. Real-ESRGAN upscalers have no text encoder, so the
{{word_limit}}-word budget is unused and the model choice does all the work.

## Syntax

No weighting, no negative prompt, no quoted text, and no reference addressing.

## Generation context

The scale factor is fixed by the chosen model, 2x or 4x, and the canvas follows
the source image. Pick the general model for photographs and mixed artwork, or
an anime model for line art. Lower the tile size when VRAM is tight.

## Examples

Input: make this photo bigger

Output: No prompt. Run the general 4x model and compare against the original.

Input: upscale this anime cel

Output: No prompt. Run an anime 4x model so the line art stays clean.

## Pitfalls

- Judge fine texture, halos, and facial detail against the original at 100
  percent zoom.
- Upscaling cannot recover detail the source never held.

## CLI

```bash
# Default general model, auto-downloads on the host that runs the job
mold upscale photo.png

# Anime line art
mold upscale photo.png -m real-esrgan-x4plus-anime:fp16

# Custom output path with an inline preview
mold upscale photo.png -o photo_4x.png --preview

# Smaller tile size for limited VRAM
mold upscale large_photo.png --tile-size 256
```

## Sources

- https://github.com/xinntao/Real-ESRGAN
