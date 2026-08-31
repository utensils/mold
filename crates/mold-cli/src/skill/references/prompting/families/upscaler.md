# Upscaler guidance

Manifest family: `upscaler`.

Real-ESRGAN upscalers take no text prompt. Choose the general model for photos
and mixed artwork, or an anime model for line art. Judge fine texture, halos,
and facial detail against the original at 100% zoom.

```bash
mold upscale input.png --model real-esrgan-x4plus:fp16 --output output-4x.png
```
