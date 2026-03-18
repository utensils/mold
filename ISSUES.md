# Outstanding Issues — New Model Support

Tracking document for SD3.5, Flux.2 Klein, and Qwen-Image-2512 engine issues on branch `feature/sd3-flux2-qwen-image`.

---

## SD3.5 (Quantized GGUF) — Black Images

**Status:** FIXED (commit 9394e8b)

Resolved via NaN-safe quantized inference (`linear_nan_safe` wrapper). See git history for details.

---

## Flux.2 Klein — Working

**Status:** FIXED — generates coherent images in 4 steps / ~5 seconds

**Critical fix:** Scale/shift swap in `LastLayer::forward` (commit 5c47d67). The diffusers `AdaLayerNormContinuous` uses `scale, shift = chunk(emb, 2)` (scale first), but our code assigned them as `shift, scale` (shift first). This corrupted every denoising step's output.

**Architecture:**
- Transformer loads from HuggingFace diffusers format with shared modulation
- Separate Q/K/V attention in double-stream blocks
- VAE with diffusers naming + BatchNorm2d latent denormalization
- Qwen3 text encoding (layers 9, 18, 27) with `enable_thinking=False` chat template
- All linear layers bias=False, SwiGLU activation
- Linear timestep schedule (distilled model, no time shifting)

---

## Qwen-Image-2512 — Q4 Quality Limited by Quantization

**Status:** Pipeline correct, Q4 image quality limited

**Verified correct:**
- VAE architecture: independently verified against diffusers reference (3D full and 2D temporal-slice produce identical results)
- No hidden NaN in quantized transformer (unlike SD3.5)
- Modulation ordering matches Wan convention (shift, scale, gate)
- Output layer ordering correct (shift first for Wan)

**Quality limitation:** Q4 quantization of the 20B parameter model produces blurry/noisy output. This is expected for such aggressive quantization. Higher quality variants (Q8, Q6) are available via `mold pull`.

---

## Per-Step Progress Bars

**Status:** Fixed

All 7 model families emit `DenoiseStep` progress events.

---

## General Notes

- All code compiles cleanly (`cargo check`, `cargo clippy`, `cargo fmt`)
- 203 tests pass across all workspace crates
- SD3.5 GGUF inference produces excellent images (verified with Q4 turtle on beach)
- Flux.2 Klein BF16 generates coherent images in ~5s on RTX 4090 (4 steps)
- FLUX.1 inference works correctly (verified with flux-schnell:q4)
- SD1.5, SDXL, Z-Image engines work correctly (pre-existing)
- Qwen-Image Q4 pipeline runs end-to-end (VAE verified correct)
