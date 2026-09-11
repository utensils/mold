- **FLUX.1 and FLUX.2 are fast by default on CUDA.** Both families now render
  through FlashAttention-2 wherever the kernel is compiled in (every shipped
  sm89 `h3-cuda` artifact is such a build) and their VAE convolutions take
  cuDNN wherever that feature is compiled in. Every other still family — SD1.5,
  SDXL, SD3, Qwen-Image, Z-Image, LTX-Video, Hunyuan3D, MiniMax-H3 — keeps the
  byte-stable math/im2col defaults it has always had, unchanged in every build.
  `MOLD_ATTN=math` and `MOLD_CONV=im2col` remain the opt-outs and remain the
  cross-build determinism contract going forward.
- **A FLUX print archived before this release will not re-render byte-for-byte
  after it, under any setting.** Flash attention, cuDNN convolutions and the
  folded softmax scale all change reduction order, and restoring the old bytes
  would mean keeping the code path this release deletes. Renders made from here
  on are reproducible among themselves — same seed, same settings, same
  backend, same bytes — and the execution plan records which convolution
  backend and which attention kernel actually ran, so two renders that differ
  are never silently filed as the same execution. Dense BF16 FLUX.1 is the one
  exception: it attends through upstream Candle, which has no policy hook, so
  it stays on math.
