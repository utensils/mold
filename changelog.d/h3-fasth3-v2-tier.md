- **MiniMax H3 FastH3 V2 8-step tier.** `mold pull minimax-h3-fl2va:fasth3-v2-int8`
  adds FastVideo's FastH3 V2: a full-weight distilled FL2VA transformer
  (22.128 GB, pinned from `FastVideo/FastVideo-FastH3-Comfy`) that runs on the
  existing compact stack, so a machine with any compact FL2VA tag installed
  downloads only the transformer. It renders at FastVideo's own operating
  point with no configuration: 8 steps of `res_multistep` (9 terminal-inclusive
  grid points, a fixed Steps control that explains itself), sigma shift 10 for
  video and 3 for audio, guidance 0. Text-only, first-frame, last-frame, and
  first+last-frame requests all work, on CUDA and Apple Metal
  ([#1812](https://github.com/utensils/mold/issues/1812)).
