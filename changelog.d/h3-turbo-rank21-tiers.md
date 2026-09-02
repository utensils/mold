- **Three rank-21 MiniMax H3 Turbo tiers.**
  `minimax-h3-fl2va:comfy-pruned-int8-turbo-4step-768p-r21`,
  `minimax-h3-fl2va:comfy-pruned-int8-turbo-8step-r21`, and
  `minimax-h3-ref2va:comfy-pruned-int8-turbo-4step-r21` pull the same compact
  stacks with drbaph's SVD-resized adapters (about 300 MB each instead of
  1.96 GB, about 1.63-1.66 GB less to download and a measured 1.60-1.70 GB
  less resident VRAM); they are lossy low-rank approximations of the
  reviewed adapters, and the measured A/B against each full-rank source tier
  is recorded in `docs/qualification/minimax-h3.md` (visual parity on all
  six pairs, 1528-1622 MiB (1.60-1.70 GB) less VRAM measured, PSNR 21-29 dB
  on the FL2VA pairs and 16-17 dB on the two panning Ref2VA pairs)
  ([#814](https://github.com/utensils/mold/issues/814)).
