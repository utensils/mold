# Outstanding Issues — New Model Support

Tracking document for SD3.5, Flux.2 Klein, and Qwen-Image-2512 engine issues on branch `feature/sd3-flux2-qwen-image`.

---

## SD3.5 (Quantized GGUF) — Black Images

**Status:** FIXED (commit 9394e8b)

Resolved via NaN-safe quantized inference (`linear_nan_safe` wrapper). See git history for details.

---

## Flux.2 Klein — End-to-End Pipeline Working

**Status:** Pipeline runs end-to-end, output quality needs tuning

**What works:**
- Transformer loads from HuggingFace diffusers format (`Flux2Transformer2DModel`)
- Shared modulation across all blocks (not per-block like FLUX.1)
- Separate Q/K/V attention in double-stream blocks
- VAE loads from diffusers format with BatchNorm2d latent denormalization
- Qwen3 text encoding with multi-layer extraction (layers 9, 18, 27)
- Chat template with `enable_thinking=False` (`<think>\n\n</think>` block)
- All linear layers bias=False
- Linear timestep schedule (no shift for distilled model)
- SwiGLU activation in both double and single-stream blocks
- 4-step distilled inference completes in ~6 seconds on RTX 4090

**Remaining quality issue:** Output images show correct compositional structure (colors, layout match prompt) but are noisy. With 20 steps, more structure emerges. Possible causes:
- Text conditioning quality (Qwen3 layer extraction or chat template details)
- Subtle attention or RoPE implementation differences from reference
- Need to compare intermediate tensor values with diffusers reference

**Files:**
- `crates/mold-inference/src/flux2/transformer.rs` — rewritten for diffusers weight format
- `crates/mold-inference/src/flux2/vae.rs` — rewritten for diffusers weight format
- `crates/mold-inference/src/flux2/pipeline.rs` — linear schedule, chat template
- `crates/mold-inference/src/encoders/qwen3.rs` — Flux.2 chat template format
- `crates/mold-core/src/manifest.rs` — sharded text encoder, steps=4 default

---

## Qwen-Image-2512 — VAE Verified Correct, Denoising Quality Low

**Status:** VAE architecture verified correct via Python reference comparison

**VAE:** Independently verified against diffusers reference decoder (both 3D full and 2D temporal-slice extraction produce identical output). The apparent "checkerboard" artifacts were Moire display aliasing from 1024x1024→display scaling, not actual pixel-level issues. Pixel analysis confirmed even/odd means are identical.

**Denoising quality:** The Q4 quantized transformer produces noisy output. This may be inherent to aggressive Q4 quantization of the 20B parameter model, or there may be issues in the quantized transformer implementation.

**Files:**
- `crates/mold-inference/src/qwen_image/vae.rs` — correct 2D temporal-slice approximation

---

## Per-Step Progress Bars

**Status:** Fixed

All 7 model families emit `DenoiseStep` progress events.

---

## General Notes

- All code compiles cleanly (`cargo check`, `cargo clippy`, `cargo fmt`)
- 192 tests pass across all workspace crates
- SD3.5 GGUF inference works correctly (verified with Q4 turtle image)
- FLUX.1 inference works correctly (verified with flux-schnell:q4)
- SD1.5, SDXL, Z-Image engines work correctly (pre-existing)
- Flux.2 Klein BF16 pipeline runs end-to-end (6s on RTX 4090)
- Qwen-Image Q4 pipeline runs end-to-end (135s on RTX 4090)
