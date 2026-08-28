- **Gallery import publication is one record per attempt.** The server's atomic
  gallery-import transaction now stages and publishes exactly one file; the
  multi-child batch staging path (per-child leases, auxiliary thumbnail/preview
  receipts, and the `child_record_updated` / `child_auxiliary_staged` /
  `child_auxiliaries_cleared` / `child_unstaged` journal events) is retired.
  Startup recovery fails closed on a crash-orphaned attempt that an earlier
  multi-child release left behind — an attempt directory holding more than one
  child, or a journal carrying one of those retired events — naming the
  directory to move out of `.mold-batch-transactions` after inspection. The
  committed archive is unaffected and still reads multi-child manifests.
