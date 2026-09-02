- **Wan 2.1 T2V 1.3B Turbo (`wan21-t2v-1.3b:turbo`).** FastVideo's DMD distill
  of the same 1.3B transformer renders 480p text-to-video in three denoise
  rungs (timesteps 1000 / 757 / 522) with no classifier-free pass — 20x fewer
  transformer forwards per clip than the 30-step `:bf16` tier, on the same
  UMT5 encoder and Wan 2.1 VAE. The ladder is the checkpoint's published
  schedule, so steps, guidance, sample solver, and flow shift are fixed: a
  request that sets one is refused by name rather than silently ignored.
  `:bf16` is unchanged, and the bare `wan21-t2v-1.3b` still resolves to it.
