- **Speed up Qwen Image 2.1 on Mac.** Use fused Metal attention, fewer
  normalization/rotary copies, compact cached-step modulation, and a BF16
  denoiser while keeping the encoder and VAE in F32. The precision change can
  alter fine details for a fixed seed. `MOLD_QWEN_IMAGE21_DTYPE=f32` restores
  full denoiser precision; combine it with `MOLD_ATTN=math` for the original
  Metal computation path. Other families and CPU/CUDA precision are unchanged.
