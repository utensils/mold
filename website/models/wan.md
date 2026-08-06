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

| Model                 | Steps | Approx total pull | Notes                                        |
| --------------------- | ----- | ----------------- | -------------------------------------------- |
| `wan21-t2v-1.3b:bf16` | 30    | ~14.5 GB          | 480p text-to-video; smallest, fastest pull   |
| `wan22-ti2v-5b:fp16`  | 20    | ~22.8 GB          | 720p24 text- and image-to-video              |
| `wan22-t2v-a14b:q5`   | 4     | ~36 GB            | 480p16 text-to-video, 4-step Lightning tier  |
| `wan22-t2v-a14b:q8`   | 20    | ~42 GB            | Same weights at Q8_0, no distill             |
| `wan22-i2v-a14b:q5`   | 4     | ~36 GB            | 480p16 image-to-video, 4-step Lightning tier |
| `wan22-i2v-a14b:q8`   | 20    | ~42 GB            | Same weights at Q8_0, no distill             |

Totals include the shared UMT5-XXL encoder (~11.4 GB), tokenizer, and the
variant's VAE. The encoder is shared across every Wan model under
`shared/wan/`, so a second Wan pull only fetches the checkpoint and VAE.

### A14B is two models

Wan 2.2 A14B is a mixture of experts along the _noise_ axis: two complete 14B
transformers, one trained for the early, structural part of the schedule and
one for the late, detail part. mold loads the high-noise expert first, switches
once when the schedule crosses the boundary (timestep 875 for T2V, 900 for
I2V), and drops each expert before loading its partner — so **VRAM is the
larger of the two experts, not their sum** (~10.8 GB at `:q5`, ~15.4 GB at
`:q8`). Disk is the sum, which is why the pull totals are large.

The `:q5` tier additionally pulls lightx2v's 4-step distill — a separate
adapter for each expert — and defaults to guidance 1.0. That is not a weak
setting: at guidance ≤ 1 mold skips the unconditional pass entirely, so each
step is one forward instead of two. Four steps at one forward each is where
the tier's speed comes from.

## Usage

```bash
# 480p, 81 frames @ 16 fps (defaults)
mold run wan21-t2v-1.3b "a red fox trotting through fresh snow, golden hour"

# 720p24, 121 frames — Wan 2.2 5B
mold run wan22-ti2v-5b "aerial view of waves breaking on a black sand beach" \
  --width 1280 --height 704 --frames 121 --fps 24

# Wan 2.2 A14B, 4-step Lightning tier
mold run wan22-t2v-a14b:q5 "a paper boat drifting down a rain gutter"

# A14B image-to-video from a still
mold run wan22-i2v-a14b:q5 "the balloon lifts off" --image balloon.png
```

Wan checkpoints were tuned against a specific negative prompt; mold applies
it automatically when `--negative` is not given.

## Defaults and limits

| Property   | `wan21-t2v-1.3b`  | `wan22-ti2v-5b`     | `wan22-*-a14b:q5` | `wan22-*-a14b:q8` |
| ---------- | ----------------- | ------------------- | ----------------- | ----------------- |
| Resolution | 832x480 / 480x832 | 1280x704 / 704x1280 | 832x480           | 832x480           |
| Frames     | 81 @ 16 fps       | 121 @ 24 fps        | 81 @ 16 fps       | 81 @ 16 fps       |
| Steps      | 30                | 20                  | 4                 | 20                |
| Guidance   | 6.0               | 5.0                 | 1.0 (no CFG pass) | 3.5               |
| Flow shift | 8.0               | 8.0                 | 5.0               | 5.0               |
| Sampler    | FlowUniPC (bh2)   | FlowUniPC (bh2)     | FlowUniPC (bh2)   | FlowUniPC (bh2)   |

The sampler schedule matches the one lightx2v's Lightning distills were
trained against (diffusers' flow-UniPC grid), so the 4-step tier reproduces
its published timesteps exactly. Override the flow shift with
`MOLD_WAN_SHIFT`.

## Quantized checkpoints and adapters

A14B ships as GGUF. Quantized weights stay quantized in memory and dequantize
inside the matmul, which is what keeps a 14B expert at ~10.8 GB rather than
~28 GB. A LoRA cannot be merged into a weight in that state without
requantizing it, so on GGUF mold applies adapters as a parallel low-rank
branch instead — the same arithmetic, applied at full precision, with no load
cost. On bf16 and fp8-scaled safetensors the adapter is merged as the weights
are read.

`*_fp8_e4m3fn_scaled` safetensors also load: the weights stay 1 byte per
parameter and dequantize per call against their per-module scale. The `e5m2`
variants some repositories publish beside them are refused by name — mold
reads the e4m3 flavour only.

## Discovery

Wan models are installable from the catalog as well as by manifest name. Open
**Models → Discover** in Mold Studio and search for `wan`, or install a specific
entry directly:

```bash
mold pull hf:Wan-AI/Wan2.2-T2V-A14B
mold pull cv:<version-id>
```

Every Wan checkpoint in the wild ships the transformer alone, so an install
also pulls the shared UMT5-XXL encoder and the matching VAE. Those are the same
files the manifest models use, under `shared/wan/`, so a second Wan install
reuses them.

Two things the catalog deliberately does not offer. Wan 2.1 **image-to-video**
entries are filtered out: they condition through a CLIP-vision cross-attention
branch mold's transformer does not implement, so the download would install and
then fail to generate. Wan 2.5 and 2.7 are later architectures with no mold
engine and are filtered out for the same reason. Wan 2.1 text-to-video, both
A14B experts, and TI2V-5B are all installable.

## Roadmap

Remaining Wan work is tracked in the
[Wan Video milestone](https://github.com/utensils/mold/milestone/4).
