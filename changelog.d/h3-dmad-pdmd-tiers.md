- **MiniMax H3 DMAD and PDMD distill tiers.**
  `minimax-h3-fl2va:comfy-pruned-int8-dmad-4step` (4 steps),
  `minimax-h3-fl2va:comfy-pruned-int8-pdmd-2step` (2 steps), and
  `minimax-h3-fl2va:comfy-pruned-int8-pdmd-4step` (4 steps) pull the FL2VA
  compact stack plus Kijai's rank-reduced build of each distillation LoRA
  (367-559 MB, from `Kijai/MiniMax-H3-experimental`). Each runs its published
  recipe: DMAD uses ComfyUI's `lcm` re-noise sampler at video/audio shift
  12/2, PDMD 2-step uses Euler at 12/6, and PDMD 4-step uses Euler at 12/3.
  The adapters are lossy low-rank approximations of the upstream full-rank
  releases. `MOLD_H3_TURBO_TIER` (capture-scope UAT override) accepts
  `dmad-4step`, `pdmd-2step`, and `pdmd-4step`. The fixed-steps note now reads
  "Fixed by the N-step distilled tier"
  ([#1812](https://github.com/utensils/mold/issues/1812)).
