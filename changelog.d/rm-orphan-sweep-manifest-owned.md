- **`mold rm` no longer deletes shared files that installed models still own.**
  Ownership is now read from the model's own manifest instead of its
  `[models]` config projection, which had no slot for an audio VAE,
  task/processor/scheduler/architecture configs, or a second file of a role
  (LTX-2.3 ships two spatial upscalers) — so the post-removal sweep read those
  files as orphans and deleted them out from under installed MiniMax H3 and
  LTX-2.3 checkpoints. A partially installed model now also keeps its shared
  dependencies so `mold pull` can repair it, `.sha256-verified` markers are
  kept with the files they attest, and every path the sweep deletes is printed
  instead of only being counted.
