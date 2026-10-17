- **Mac Library Sync costs what changed, not the whole library.** Sync All
  and Keep This Mac in Sync now read every machine at once through its gallery
  replica. Keep This Mac in Sync plans only the prints that changed since its
  last run (plus anything still owed a retry, and a full check once a day); a
  Sync All press verifies every print against one checkpoint read per
  machine. Either way a run where nothing changed asks no machine anything per
  print. Copy receipts come from the import's own answer, synchronized to disk
  before retained inputs move, instead of a per-print listing read; final
  verification asks the bounded `POST /api/gallery/sync-checkpoint` for just
  the touched prints (older machines fall back to the full checkpoint); and
  copy records moved out of the preferences suite into their own file, written
  off the main thread, with a one-time migration and pruning of copies gone
  from both sides. An unreadable records file is set aside and pauses Sync All
  instead of recreating copies you removed. Collection visibility is synced in
  one pass and progress updates at most ten times a second.
