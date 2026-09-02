- **Three rank-21 MiniMax H3 Turbo tiers.**
  `minimax-h3-fl2va:comfy-pruned-int8-turbo-4step-768p-r21`,
  `minimax-h3-fl2va:comfy-pruned-int8-turbo-8step-r21`, and
  `minimax-h3-ref2va:comfy-pruned-int8-turbo-4step-r21` pull the same compact
  stacks with drbaph's SVD-resized adapters (about 300 MB each instead of
  1.96 GB, about 1.63 GB less to download and the same again off resident
  VRAM); they are lossy low-rank approximations of the reviewed adapters,
  carrying pinned-identity evidence only — the measured A/B against each
  full-rank source tier has not run yet
  ([#814](https://github.com/utensils/mold/issues/814)).
