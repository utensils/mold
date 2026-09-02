- **Wan 2.2 TI2V 5B DMD (`wan22-ti2v-5b:dmd`).** FastVideo's DMD distill of
  the same 2.2 TI2V-5B transformer renders 720p24 video in three denoise
  rungs (timesteps 1000 / 757 / 522) with no classifier-free pass — `(20 x
  2) / 3` ≈ 13.3x fewer transformer forwards per clip than the 20-step
  `:fp16` tier, on the same UMT5 encoder and 2.2 VAE. The ladder walks its
  own shift-5 flow-match table, the shift its distillation actually trained
  against, not the shift-8 table FastVideo's own inference code hardcodes
  for every DMD tier it ships. Steps, guidance, sample solver, and flow
  shift are fixed: a request that sets one is refused by name rather than
  silently ignored. Image-to-video is refused on this
  tier, where the other three 5B tiers accept it: upstream ships no image
  branch for this checkpoint, and measured against `:turbo` from the same
  stills and seeds, the distilled student abandons the pinned first frame
  within about four frames instead of continuing from it. Use `:turbo`,
  `:fp16`, or `:q8` for image-to-video. `:fp16`,
  `:q8`, and `:turbo` are unchanged, and the bare `wan22-ti2v-5b` still
  resolves to `:fp16` ([#TBD](https://github.com/utensils/mold/pull/TBD)).
