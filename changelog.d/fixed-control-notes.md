- **Fixed generation controls now explain themselves with the server's own
  words.** Create's Detail and Prompt strength fields on web, desktop, and
  iPhone render an additive per-control note from the generation profile, so a
  MiniMax H3 Turbo tier finally says why Steps reads `9` ("Fixed by the 8-step
  Turbo tier: 9 terminal-inclusive sampler grid points (8 denoise intervals).")
  and why Guidance is locked ("MiniMax H3 does not use classifier-free
  guidance; guidance is fixed at 0."). Every surface previously hard-coded a
  distilled-FLUX sentence that was simply false for H3 — it claimed CFG was
  fixed at 1.0 and offered a Dev checkpoint that does not exist. Distilled
  FLUX/LTX recipes keep their existing wording, now generated from the value
  the recipe actually pinned, and a control an older server fixed without a
  note renders nothing rather than invented copy.
- **H3 size pills no longer read "Reviewed".** The private-runtime bridge
  overrode every H3 resolution preset's tier, putting a mark on every pill that
  named a qualification rather than anything about the size; the presets now
  keep the shared `recommended` tier every other model uses, and the `Default`
  mark still points at the model's own default canvas.
