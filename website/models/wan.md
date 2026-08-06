# Wan Video

Text-to-video generation from [Alibaba's Wan team](https://github.com/Wan-Video),
based on a flow-matching DiT with a UMT5-XXL text encoder and a causal 3D
video VAE that streams decoding one latent frame at a time. mold implements
the family natively in Rust.

- **Developer**: [Wan-AI](https://huggingface.co/Wan-AI)
- **License**: Apache 2.0
- **Reference**: [Wan-Video/Wan2.1](https://github.com/Wan-Video/Wan2.1) ·
  [Wan-Video/Wan2.2](https://github.com/Wan-Video/Wan2.2)

> **Note**: Video output defaults to MP4. Also supports GIF, WebP, and APNG
> via `--format`. Frame count must be 4n+1 (77, 81, 121, ...) due to the
> VAE's 4x temporal compression. Wan generation currently targets CUDA;
> CPU runs are correctness-only.

## Variants

| Model                 | Steps | Approx total pull | Notes                                      |
| --------------------- | ----- | ----------------- | ------------------------------------------ |
| `wan21-t2v-1.3b:bf16` | 30    | ~14.5 GB          | 480p text-to-video; smallest, fastest pull |
| `wan22-ti2v-5b:fp16`  | 20    | ~22.8 GB          | 720p24 text-to-video (image-to-video soon) |

Totals include the shared UMT5-XXL encoder (~11.4 GB), tokenizer, and the
variant's VAE. The encoder is shared across every Wan model under
`shared/wan/`, so a second Wan pull only fetches the checkpoint and VAE.

## Usage

```bash
# 480p, 81 frames @ 16 fps (defaults)
mold run wan21-t2v-1.3b "a red fox trotting through fresh snow, golden hour"

# 720p24, 121 frames — Wan 2.2 5B
mold run wan22-ti2v-5b "aerial view of waves breaking on a black sand beach" \
  --width 1280 --height 704 --frames 121 --fps 24
```

Wan checkpoints were tuned against a specific negative prompt; mold applies
it automatically when `--negative` is not given.

## Defaults and limits

| Property   | `wan21-t2v-1.3b`  | `wan22-ti2v-5b`     |
| ---------- | ----------------- | ------------------- |
| Resolution | 832x480 / 480x832 | 1280x704 / 704x1280 |
| Frames     | 81 @ 16 fps       | 121 @ 24 fps        |
| Guidance   | 6.0               | 5.0                 |
| Sampler    | FlowUniPC (bh2)   | FlowUniPC (bh2)     |

The sampler schedule matches the one lightx2v's Lightning distills were
trained against (diffusers' flow-UniPC grid), so future 4-step fast-tier
checkpoints reproduce their published timesteps exactly.

## Roadmap

Image-to-video conditioning, the Wan 2.2 A14B two-expert models with GGUF
quantization and the 4-step Lightning fast tier, and catalog discovery are
tracked in the
[Wan Video milestone](https://github.com/utensils/mold/milestone/4).
