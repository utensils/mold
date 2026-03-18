# Outstanding Issues — New Model Support

Tracking document for SD3.5, Flux.2 Klein, and Qwen-Image-2512 engine issues on branch `feature/sd3-flux2-qwen-image`.

---

## SD3.5 (Quantized GGUF) — Black Images

**Status:** FIXED (commit 9394e8b)

**Symptom:** `mold run sd3.5-large:q4 "prompt"` completes all 28 denoising steps but produces an all-black 1024x1024 image (18KB PNG). The denoised latent contains `inf`/`NaN`.

**Root Cause Chain:**

1. **Candle CUDA QMatMul produces hidden NaN.** The `quantized_nn::Linear` (which wraps `QMatMul`) produces NaN in some output elements when processing large tensors on CUDA. The NaN values are hidden because `Tensor::min_all()`/`max_all()` skip NaN, so diagnostics show finite min/max while NaN lurks in the tensor.

2. **NaN propagates through attention.** The QKV projection linear produces a tensor with hidden NaN. This flows into the attention weight computation (`Q @ K^T`), producing NaN in the attention weights. The softmax of NaN produces NaN. The attention output becomes NaN/inf.

3. **inf cascades through all blocks.** Once block 0 produces inf, all subsequent blocks and denoising steps produce inf. The VAE decodes inf latents to all-zero pixels (black).

**Evidence:**
- `[sd3-chunk] chunk 0 weights=[-8.75,10.85] scores=[inf,-inf]` — weights look finite but contain hidden NaN
- `[DIRECT] w_cpu: [-8.76,10.85] sm_cpu: [NaN,NaN]` — CPU F32 softmax of the weights produces NaN, confirming hidden NaN in the weight tensor
- One run with F16 attention + manual softmax produced finite block 0 output `x=[-79.3,18.7]`, proving the architecture is correct when NaN is avoided

**Attempted Fixes:**
- Chunked attention (256-token chunks) — reduces VRAM but doesn't fix NaN from QMatMul
- F16/BF16 attention — halves memory but F16 matmul also produces NaN
- Manual F32 softmax — doesn't help because the NaN is in the matmul input
- CPU softmax — confirms NaN comes from the matmul, not softmax

**Proposed Fix:**
- **Option A (workaround):** Add `nan_to_num(0.0)` after every `quantized_nn::Linear` forward call in the quantized MMDiT. This clamps NaN to zero, preventing propagation. May slightly affect image quality.
- **Option B (proper fix):** Dequantize GGUF weights to BF16 safetensors at load time, then use candle's regular `nn::Linear` (which doesn't produce NaN). This matches ComfyUI's approach of dequantizing GGUF to F16 for inference.
- **Option C (upstream):** File a candle issue for the CUDA QMatMul NaN bug. The bug appears to be non-deterministic and related to tensor size/alignment.

**Files Involved:**
- `crates/mold-inference/src/sd3/quantized_mmdit.rs` — quantized MMDiT transformer
- `crates/mold-inference/src/sd3/pipeline.rs` — SD3 engine pipeline
- `crates/mold-inference/src/sd3/sampling.rs` — Euler flow-matching sampler

**How to Reproduce:**
```bash
mold run sd3.5-large:q4 "a turtle on the beach near sunset"
# Output: all-black 1024x1024 PNG

# With debug diagnostics:
MOLD_SD3_DEBUG=1 mold run sd3.5-large:q4 "a turtle"
# Shows: block 0 output: x=[inf,-inf]
```

**Reference Implementations:**
- ComfyUI-GGUF (city96): dequantizes to F16, runs attention in F16
- stable-diffusion.cpp: uses ggml's native F16 attention kernels
- Candle SD3 example: uses BF16 throughout (non-quantized)

---

## SD3.5 — `--steps` and `--width`/`--height` CLI flags ignored

**Status:** FIXED (commit 9394e8b)

**Symptom:** The `--steps`, `--width`, and `--height` CLI arguments don't override the model's default values from the manifest/config. The model always uses its defaults (28 steps, 1024x1024 for sd3.5-large).

**Expected:** CLI args should override manifest defaults.

---

## Qwen-Image-2512 — VAE Decode Produces Wrong Output

**Status:** Architecture mismatch identified by codex

**Symptom:** Qwen-Image generates degenerate images:
- Default temporal slice: near-white frames
- Forcing temporal slice 0: flat brown frame

**Root Cause:** The `AutoencoderKLQwenImage` VAE uses a Wan video model architecture with 3D causal convolutions (`QwenImageCausalConv3d`). Our current VAE implementation uses a 2D temporal-slice approximation with candle's standard `AutoEncoderKL`, which doesn't match the actual architecture.

**What's Needed (from research):**

Candle has **NO native Conv3d**. The only 3D-like code is `conv3d_temporal_2.rs` (hardcoded temporal_patch_size=2).

**Recommended approach: Simulate 3D CausalConv via temporal slicing**

For each 3D conv weight `[out_c, in_c, T_k, H_k, W_k]`, decompose into T_k separate Conv2d layers (one per temporal kernel slice). For single-frame (T=1) with k=3 and causal padding:
- Input gets padded to `[B, C, 3, H', W']` (2 zero-pad frames + 1 real frame)
- Slice 0: `Conv2d(weight[:,:,0,:,:])` on zeros → contributes 0 (no bias in these convs)
- Slice 1: `Conv2d(weight[:,:,1,:,:])` on zeros → contributes 0
- Slice 2: `Conv2d(weight[:,:,2,:,:])` on real frame → full contribution
- **Optimization**: For T=1, only the last temporal slice actually contributes

**Decoder architecture** (from safetensors weight inspection):
```
conv_in: CausalConv3d(16→384, k=3)
mid_block: 2× ResBlock(384) + Attention(384, spatial-only)
up_blocks: [384→384, 384→192, 192→96, 96→96] with temporal_upsample on blocks 0,1
norm_out: RMSNorm(96)
conv_out: CausalConv3d(96→3, k=3)
```

**Per-channel latent denormalization**: `latents = latents * std + mean` (16-channel arrays from VAE config)

**Reference implementations:**
- Diffusers `AutoencoderKLQwenImage` / `AutoencoderKLWan`
- ComfyUI `comfy/ldm/wan/vae.py`

**Files Involved:**
- `crates/mold-inference/src/qwen_image/vae.rs` — needs rewrite
- `crates/mold-inference/src/qwen_image/pipeline.rs` — decode path
- `crates/mold-inference/src/encoders/qwen2_text.rs` — codex improved

---

## Flux.2 Klein — Critical Architecture Discrepancies Found

**Status:** Research complete, multiple fixes needed before end-to-end testing

**Transformer config verified correct:** `in_channels=128`, `hidden_size=3072`, `num_heads=24`, `depth=5`, `depth_single=20`, `axes_dim=[32,32,32,32]`, `theta=2000`, `mlp_ratio=3.0`.

**Critical Discrepancies (from diffusers `Flux2Transformer2DModel` + `pipeline_flux2_klein.py`):**

| Issue | Priority | Our Code | Correct |
|-------|----------|----------|---------|
| Text encoding | CRITICAL | Repeats final Qwen3 layer 3x | Stack hidden states from layers 9, 18, 27 |
| VAE latent normalization | CRITICAL | scale_factor/shift_factor | BatchNorm2d with running_mean/running_var from weights |
| Chat template | HIGH | Raw prompt tokenization | `apply_chat_template(messages, enable_thinking=False)` |
| MLP activation | HIGH | GELU (FLUX.1 style) | SwiGLU in single-stream blocks |
| Linear bias | HIGH | bias=True throughout | bias=False on most layers |
| Scheduler | HIGH | Simple linear (no shift) | Dynamic exponential time-shifting (shift=3.0) |
| CFG guidance | MEDIUM | No CFG (guidance=0.0) | guidance_scale=4.0 with dual forward pass |
| Single-stream fused layers | MEDIUM | Separate linear1/linear2 | Fused QKV+MLP projections |

**VAE BatchNorm details:** The Flux.2 VAE uses `BatchNorm2d(128, affine=False)` on patchified latents (128 = 2×2×32 channels). Encoding normalizes with `(latents - running_mean) / sqrt(running_var + eps)`, decoding denormalizes with `latents * sqrt(running_var + eps) + running_mean`. The running stats must be loaded from the VAE weights.

**Patchify/Unpatchify:** The pipeline patchifies latents `(B,32,H,W) → (B,128,H/2,W/2)` by grouping 2×2 spatial patches into channels, then packs to sequence `(B,H/2*W/2,128)`.

**GGUF availability:** `unsloth/FLUX.2-klein-4B-GGUF` has 14 variants (Q2_K through BF16).

**Files Involved:**
- `crates/mold-inference/src/flux2/pipeline.rs` — text encoding, VAE decode, scheduler
- `crates/mold-inference/src/flux2/transformer.rs` — SwiGLU, bias, fused layers
- `crates/mold-inference/src/flux2/sampling.rs` — scheduler shift
- `crates/mold-inference/src/flux2/vae.rs` — BatchNorm latent normalization

---

## Per-Step Progress Bars

**Status:** Fixed

All 7 model families now emit `DenoiseStep` progress events, enabling live `it/s` progress bars in the CLI during denoising. Previously only SD1.5, SDXL, Z-Image, and Qwen-Image had per-step progress; FLUX.1, Flux.2, and SD3 used opaque sampling functions.

---

## General Notes

- All code compiles cleanly (`cargo check`, `cargo clippy`, `cargo fmt`)
- 184 tests pass across all workspace crates
- SD3.5 GGUF models from city96 are verified (SHA matches)
- FLUX.1 inference works correctly (verified with `flux-schnell:q4`)
- SD1.5, SDXL, Z-Image engines work correctly (pre-existing)
