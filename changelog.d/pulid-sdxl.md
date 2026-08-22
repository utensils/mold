- **PuLID face-identity conditioning for SDXL.** `--id-image` now works on
  `sdxl-base:fp16`, `juggernaut-xl:fp16`, `realvis-xl:fp16`, and
  `dreamshaper-xl:fp16` — the checkpoints upstream's own PuLID v1.1 release
  qualifies — alongside the existing FLUX support. `mold pull pulid-sdxl` adds
  only the 984 MB v1.1 adapter on a machine that already has `pulid-flux`,
  because the face extractor is shared. Clients need no change: every surface
  already gates on the server's advertised
  `/api/models[].supports_identity`. `true_cfg` / `cfg_start_step` stay FLUX-only
  — SDXL's ordinary `guidance` already is the classifier-free scale
  ([#1228](https://github.com/utensils/mold/issues/1228)).
