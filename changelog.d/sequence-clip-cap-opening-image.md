- **Sequence clips are capped at each model's own clip size, and the opening
  image moved out of Advanced.** `GET /api/capabilities/chain-limits` now
  advertises `frames_per_clip_cap` as the per-model clip size one generation
  renders (97 for LTX-2; for Wan the checkpoint's own manifest default over a
  53-frame A14B / 121-frame floor) instead of the family's 20 s duration budget, so the Sequence
  composer on web, desktop, and iPhone no longer offers a single 481-frame
  LTX-2 clip that the one-shot Duration slider would have split into five.
  The Studio pickers lock to the same per-model size even against an older
  server, and the explicit CLI `--clip-frames` escape hatch keeps its full
  single-request budget. The **Opening sequence image** (with its source
  strength and fit controls) now sits in the primary Create form on all three
  surfaces — where one-shot source media already lives — and the primary
  ↺ Reset clears it while the Advanced reset leaves it alone.
