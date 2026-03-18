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

**What's Needed:**
- Implement proper `QwenImageCausalConv3d` layers (3D causal convolutions)
- Handle the temporal dimension correctly for single-frame (image) generation
- Apply per-channel latent normalization using the model's `latents_mean`/`latents_std` arrays
- The `post_quant_conv` is a causal 3D convolution reducing from `z_dim*2` to `z_dim`

**Codex's Progress:**
- Fixed text encoder: prompt formatting, penultimate hidden states, text masking, token ordering, RoPE bounds
- Removed CUDA crashes and NaN collapse in denoiser
- VAE remains the blocker

**Reference:**
- Diffusers `AutoencoderKLQwenImage`: uses `QwenImageCausalConv3d`, tiled encode/decode support
- HuggingFace docs: https://huggingface.co/docs/diffusers/main/api/models/autoencoderkl_qwenimage

**Files Involved:**
- `crates/mold-inference/src/qwen_image/vae.rs` — needs rewrite
- `crates/mold-inference/src/qwen_image/pipeline.rs` — decode path
- `crates/mold-inference/src/encoders/qwen2_text.rs` — codex improved

---

## Flux.2 Klein — Not Yet Tested End-to-End

**Status:** Engine scaffolded, not tested with real weights

**Notes:**
- Transformer architecture implemented (5 double + 20 single stream blocks)
- Qwen3 encoder integration done (reuses Z-Image encoder)
- VAE (`AutoencoderKLFlux2`) with `latent_channels=32` implemented
- Sampling utilities adapted for 128-channel latents
- Manifest entry for `flux2-klein:bf16` added
- Model weights need to be downloaded and inference tested

**Potential Issues:**
- The 4D RoPE `[32,32,32,32]` implementation may need verification
- The Qwen3 encoder output stacking (3x for `joint_attention_dim=7680`) is untested
- The Flux2 VAE (`latent_channels=32`) differs from FLUX.1 (`latent_channels=16`)

**Files Involved:**
- `crates/mold-inference/src/flux2/pipeline.rs`
- `crates/mold-inference/src/flux2/transformer.rs`
- `crates/mold-inference/src/flux2/sampling.rs`
- `crates/mold-inference/src/flux2/vae.rs`

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
