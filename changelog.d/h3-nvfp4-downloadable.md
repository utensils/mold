- **MiniMax H3 pruned NVFP4 transformer is downloadable.** `minimax-h3-fl2va:comfy-pruned-nvfp4`
  and `minimax-h3-ref2va:comfy-pruned-nvfp4` pull, verify, and appear in
  Models → Discover/Installed like any other model — a 12.529 GB transformer
  download on top of an already-installed compact variant, since every other
  component is shared. Mold has no engine arm for this weight layout yet
  (tracked in [#1318](https://github.com/utensils/mold/issues/1318)), so
  generation is refused up front with a truthful "no runtime for this weight
  layout" message rather than a licensing error, and `GET /api/models` reports
  the new additive `runtime_available: false` on the row. The pinned
  `official-bf16` qualification references are refused the same honest way
  now, instead of the misleading compliance-gated message they returned before
  ([#1319](https://github.com/utensils/mold/issues/1319)).
