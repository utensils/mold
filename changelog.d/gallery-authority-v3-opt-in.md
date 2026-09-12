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
- **`downgrade` now refuses while a server is publishing, instead of rewriting
  the store under it.** Every mold process that can publish to a gallery holds
  a writer lease on it for as long as it runs, and the downgrade refuses on
  contention, naming the process and its pid and reporting that nothing was
  changed; `status` shows `live writer` so you can see it first. The lease is
  shared — several servers still share one home — and the operating system
  releases it if a process is killed. Previously the command took only the
  bookkeeping lock, which a server holds for the length of one publication, so
  it waited for the gap between two prints and succeeded against a live
  server; the server's next print then wrote version-3 bytes into the
  version-2 directory and an older binary refused to start on that home.
- **A commit can no longer land version-3 bytes under the version-2 name.**
  Independently of the lease: a server whose version-3 store disappears
  underneath it now re-establishes one beside the frozen version-2 store
  instead of writing into it, and refuses the publication outright if it
  cannot.
- **Opting in writes a new store beside the old one, never over it.** The
  version-3 store gets its own directory and the version-2 one is left intact,
  so a rollback needs no restored backup. Note that while a version-2 writer
  and a version-3 writer share one home they keep separate indexes that drift
  apart — enable the switch only where every binary using the home is new
  enough.
