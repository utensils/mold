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
- **FLUX.1 GGUF renders in BF16 on CUDA, through one transformer.** Every GGUF
  load — with or without a LoRA — now goes through mold's own transformer;
  the candle fork's quantized model, which the commonest no-LoRA render used
  to take, carried no attention-backend switch and F32 norm weights, so that
  render could reach neither FlashAttention nor half-precision activations
  however the binary was built. The two were verified bit-identical before the
  old path was deleted. Activations follow the working dtype instead of being
  pinned to F32, which halves the bandwidth every matmul moves; the weights
  stay quantized in VRAM exactly as before. `MOLD_WAN_FORCE_DMMV=1` still
  forces F32, because the fallback it selects reads activations as f32.
- **Flux.2 GGUF renders in BF16 and no longer scrubs NaN after every linear.**
  The GGUF transformer wrapped all eighteen of its linear sites in a full-tensor
  NaN compare, a zeros allocation and a `where_cond` — copied from SD3 without a
  Flux.2 NaN ever having been observed, and measured at about half a second per
  step. It is gone, and masking a non-finite value was the wrong shape anyway: a
  transformer emitting NaN has a bug, and zeroing the element turns a loud
  failure into a quietly wrong picture. `MOLD_FLUX_DEBUG_NONFINITE=1` replaces
  it with one check per denoise STEP that names the step and fails — off by
  default, and available for FLUX.1 too. Activations now follow the working
  dtype (BF16 on CUDA) instead of being cast to F32 at the transformer
  boundary; position ids stay F32, because the rotary embedding is built from
  them. `MOLD_FLUX2_QMATMUL=0` restores the per-forward dequantization arm if a
  render comes out wrong.
- **A `--lora` on a prompt-only render is no longer a silent no-op over the
  server.** Durable admission seals every request authority it later scrubs off
  the copy it hands the GPU worker, but the predicate deciding whether to seal
  anything at all counted only conditioning media — so a text-to-image render
  with an adapter sealed nothing, lost the adapter on its way to the worker,
  and produced pixels byte-identical to the same prompt with no LoRA while its
  print recorded none. Every LoRA-capable family was affected, FLUX.1 and
  FLUX.2 included; a LoRA beside an image, mask or video source always worked,
  and `--local` was never affected.
