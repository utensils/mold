- **`mold rm` no longer deletes shared files that installed models still own.**
  Ownership is now read from the model's own manifest instead of its
  `[models]` config projection, which had no slot for an audio VAE,
  task/processor/scheduler/architecture configs, or a second file of a role
  (LTX-2.3 ships two spatial upscalers) — so the post-removal sweep read those
  files as orphans and deleted them out from under installed MiniMax H3 and
  LTX-2.3 checkpoints. A partially installed model now also keeps its shared
  dependencies so `mold pull` can repair it, `.sha256-verified` markers are
  kept with the files they attest. Whether a model is complete enough to own
  shared components is asked of its manifest for the same reason. The sweep now
  prints every path it unlinks — the shared file plus an hf-cache-backed
  orphan's blob and snapshot links, which it also stopped leaving dangling —
  instead of only counting them. A removed file's `.sha256-verified` marker is
  now deleted with it and the emptied model directory is cleaned up, so no
  stale attestation is left behind.
