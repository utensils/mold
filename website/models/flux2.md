# Flux.2 Klein

A lightweight 4B parameter FLUX variant. Fast 4-step generation with low VRAM
requirements.

## Variants

| Model              | Steps | Size   | Notes             |
| ------------------ | ----- | ------ | ----------------- |
| `flux2-klein:q8`   | 4     | 4.3 GB | Good quality      |
| `flux2-klein:q4`   | 4     | 2.6 GB | Smallest FLUX     |
| `flux2-klein:bf16` | 4     | 7.8 GB | Full precision 4B |

## Defaults

- **Resolution**: 1024x1024
- **Guidance**: 0.0
- **Steps**: 4

## Architecture

Flux.2 Klein uses a Qwen3 text encoder (BF16 or GGUF, layers 9/18/27), a shared
modulation transformer (BF16 or GGUF), and a BN-VAE decoder.
