# Upscaler Models

mold supports Real-ESRGAN super-resolution models for image upscaling. These models enhance image resolution by 2x or 4x using neural networks trained on image restoration tasks.

## Model List

### RRDBNet Architecture (High Quality)

The Residual-in-Residual Dense Block Network uses deep convolutional layers with dense connections for maximum quality.

| Model | Scale | Params | Size (FP16) | Description |
|-------|-------|--------|-------------|-------------|
| `real-esrgan-x4plus:fp16` | 4x | 16.7M | 32 MB | General-purpose, best quality |
| `real-esrgan-x4plus:fp32` | 4x | 16.7M | 64 MB | Same model, FP32 precision |
| `real-esrgan-x2plus:fp16` | 2x | 16.7M | 32 MB | 2x upscale, best quality |
| `real-esrgan-x4plus-anime:fp16` | 4x | 4.5M | 8.5 MB | Anime/illustration optimized (6 blocks) |

### SRVGGNetCompact Architecture (Fast)

A lightweight linear chain architecture optimized for speed. Uses significantly less compute than RRDBNet while maintaining good quality.

| Model | Scale | Params | Size | Description |
|-------|-------|--------|------|-------------|
| `real-esrgan-anime-v3:fp32` | 4x | 0.6M | 2.4 MB | Fast anime/video upscaler |

## Architecture Details

### RRDBNet

```
Input (3, H, W)
  ↓ Conv2d(3 → 64)
  ↓ [RRDB block × 23] — each: 3 × ResidualDenseBlock (5 convs with dense connections)
  ↓ Conv2d(64 → 64)
  ↓ Upsample 2x (nearest) + Conv2d
  ↓ Upsample 2x (nearest) + Conv2d   ← only for 4x models
  ↓ Conv2d(64 → 64) + LeakyReLU
  ↓ Conv2d(64 → 3)
Output (3, H×scale, W×scale)
```

### SRVGGNetCompact

```
Input (3, H, W)
  ↓ Conv2d(3 → 64)
  ↓ [PReLU + Conv2d(64 → 64)] × N
  ↓ PReLU
  ↓ Conv2d(64 → 3×scale²)
  ↓ PixelShuffle(scale)
Output (3, H×scale, W×scale)
```

## Downloading

```bash
# Pull the default high-quality upscaler
mold pull real-esrgan-x4plus:fp16

# Pull the fast compact upscaler
mold pull real-esrgan-anime-v3:fp32

# List all available models including upscalers
mold list
```

## HuggingFace Sources

All upscaler models are sourced from trusted HuggingFace repositories:

- **RRDBNet models**: [Comfy-Org/Real-ESRGAN_repackaged](https://huggingface.co/Comfy-Org/Real-ESRGAN_repackaged) (safetensors)
- **x2plus model**: [hlky/RealESRGAN_x2plus](https://huggingface.co/hlky/RealESRGAN_x2plus) (safetensors)

## Comparison

| Use Case | Recommended Model | Why |
|----------|------------------|-----|
| Photo upscaling | `real-esrgan-x4plus:fp16` | Best detail preservation |
| Anime/manga | `real-esrgan-x4plus-anime:fp16` | Trained on anime data |
| Batch processing | `real-esrgan-anime-v3:fp32` | 5-10x faster |
| Video frames | `real-esrgan-anime-v3:fp32` | Smallest, fastest |
| Subtle enhancement | `real-esrgan-x2plus:fp16` | 2x is less aggressive |
