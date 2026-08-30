- **Documentation sync.** A verified audit against the code fixed 338 stale or wrong
  statements across the website, README, CLI skill, desktop docs, and internal
  architecture notes: broken example commands and model ids (`mold run --model`,
  `sdxl:fp16`, `z-image:bf16`, `--negative`, `MOLD_DEFAULT_MODEL=ltx-2`,
  `mold runpod generate`), PuLID qualification stated once (every FLUX and every
  SDXL checkpoint except `sdxl-turbo:fp16`), HTTP API route tables regenerated from
  the router, LTX-2 `max_frames` (481 at 24 fps), model download sizes and
  per-checkpoint defaults, TUI keymaps, deployment notes, and superseded design
  documents marked as such. `CLAUDE.md` is split into a lean root plus path-scoped
  `.claude/rules/`, with a format-on-edit hook
  ([#1470](https://github.com/utensils/mold/pull/1470)).
