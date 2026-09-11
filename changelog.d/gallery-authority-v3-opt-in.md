- **Gallery archive-authority storage version 3 is now opt-in and reversible.**
  The faster append-only delta log is written only when you ask for it
  (`mold config set gallery.authority_log true`, or
  `MOLD_GALLERY_AUTHORITY_LOG=1`); a default build leaves an existing
  version-2 store exactly as it found it. Reading a version-3 store never
  needs the switch. This matters for a `$MOLD_HOME` shared between binaries: a
  mold older than 0.29 reads version 2 only and refuses to publish against a
  version-3 store, so previously one newer process starting was enough to lock
  the others out of the home.
- **`mold system gallery-authority status` and `… downgrade`.** `status`
  reports a store's on-disk version, generation, and delta-log size without
  touching it. `downgrade` folds a version-3 store back to version 2 so an
  older mold can publish against the home again — run it with the newer build
  while no server is writing to that output directory. It is idempotent,
  verifies the result by reading it back, parks the retired version-3
  directory rather than deleting it, and refuses if a mutation is still
  pending or the log tail is torn.
- **Opting in writes a new store beside the old one, never over it.** The
  version-3 store gets its own directory and the version-2 one is left intact,
  so a rollback needs no restored backup. Note that while a version-2 writer
  and a version-3 writer share one home they keep separate indexes that drift
  apart — enable the switch only where every binary using the home is new
  enough.
