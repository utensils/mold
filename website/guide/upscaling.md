# Upscaling

mold supports image upscaling using Real-ESRGAN super-resolution models. Upscale generated images or existing photos to 2x or 4x their original resolution with AI-enhanced detail.

## Quick Start

```bash
# Upscale an image with the default model (Real-ESRGAN x4+)
mold upscale photo.png

# Specify a model
mold upscale photo.png -m real-esrgan-x4v3:fp32

# Upscale and save to a specific path
mold upscale photo.png -o photo_hires.png

# Pipe from generation to upscale
mold run "a cat" | mold upscale -
```

## Available Models

| Model | Architecture | Scale | Size | Speed | Quality |
|-------|-------------|-------|------|-------|---------|
| `real-esrgan-x4plus:fp16` | RRDBNet (23 blocks) | 4x | 32 MB | Medium | Best |
| `real-esrgan-x4plus:fp32` | RRDBNet (23 blocks) | 4x | 64 MB | Medium | Best |
| `real-esrgan-x2plus:fp16` | RRDBNet (23 blocks) | 2x | 32 MB | Medium | Best |
| `real-esrgan-x4plus-anime:fp16` | RRDBNet (6 blocks) | 4x | 8.5 MB | Fast | Great (anime) |
| `real-esrgan-x4v3:fp32` | SRVGGNetCompact | 4x | 4.7 MB | Fastest | Good |
| `real-esrgan-anime-v3:fp32` | SRVGGNetCompact | 4x | 2.4 MB | Fastest | Good (anime) |

### Choosing a Model

- **Best quality**: `real-esrgan-x4plus:fp16` -- full RRDBNet with 23 residual blocks
- **Fastest**: `real-esrgan-x4v3:fp32` -- compact architecture, great for batch processing
- **Anime/illustration**: `real-esrgan-x4plus-anime:fp16` or `real-esrgan-anime-v3:fp32`
- **2x upscale**: `real-esrgan-x2plus:fp16` -- when 4x is too much

## CLI Reference

```
mold upscale <IMAGE> [OPTIONS]

Arguments:
  <IMAGE>  Input image file path (or - for stdin)

Options:
  -m, --model <MODEL>      Upscaler model [default: real-esrgan-x4plus:fp16]
  -o, --output <PATH>      Output file path [default: <input>_upscaled.<ext>]
      --format <FORMAT>     Output format: png or jpeg [default: png]
      --tile-size <N>       Tile size for tiled inference (0 to disable) [default: 512]
      --host <URL>          Server URL override
      --local               Skip server, run inference locally
```

## Tiled Inference

Large images are automatically split into overlapping tiles for memory-efficient processing. The default tile size is 512 pixels with 32 pixels of overlap. Tiles are blended using linear gradient weights to eliminate visible seams.

```bash
# Custom tile size (smaller = less VRAM, slower)
mold upscale large_photo.png --tile-size 256

# Disable tiling (process entire image at once -- needs more VRAM)
mold upscale small_image.png --tile-size 0
```

### Memory Requirements

Upscaler models are lightweight compared to diffusion models:

- RRDBNet (x4plus): ~32-64 MB model + ~200 MB activations per 512x512 tile
- SRVGGNetCompact: ~2-5 MB model + ~50 MB activations per 512x512 tile

With the default 512px tiling, any GPU with 1 GB+ VRAM can upscale images of any size.

## Post-Generation Upscaling

Upscale images immediately after generation using the `--upscale` flag on `mold run`:

```bash
mold run "a cat" --upscale real-esrgan-x4plus:fp16
```

This generates the image at the model's native resolution (e.g. 1024x1024) and then upscales it (to 4096x4096 at 4x).

## Piping

mold upscale is fully pipe-compatible:

```bash
# Generate and upscale in a pipeline
mold run "a sunset" | mold upscale - | viu -

# Read from stdin, write to file
cat photo.png | mold upscale - -o upscaled.png

# Chain with other tools
mold upscale photo.png | convert - -resize 50% final.png
```

## Server API

When a mold server is running, the upscale command uses the server for inference:

```bash
# Server handles the upscaling
MOLD_HOST=http://gpu-server:7680 mold upscale photo.png

# Direct API call
curl -X POST http://localhost:7680/api/upscale \
  -H "Content-Type: application/json" \
  -d '{"model": "real-esrgan-x4plus:fp16", "image": "<base64>"}'
```

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `MOLD_UPSCALE_MODEL` | `real-esrgan-x4plus:fp16` | Default upscaler model |
| `MOLD_UPSCALE_TILE_SIZE` | `512` | Default tile size |
