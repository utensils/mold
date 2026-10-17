- **Mac Library Sync costs what changed, not the whole library.** Sync All
  and Keep This Mac in Sync now read every machine at once through its gallery
  replica and plan only the prints that changed since the last run (plus
  anything still owed a retry), so a run where nothing changed asks no machine
  anything per print. Copy receipts come from the import's own answer instead
  of a per-print listing read, final verification asks the bounded
  `POST /api/gallery/sync-checkpoint` for just the touched prints (older
  machines fall back to the full checkpoint), and copy records moved out of
  the preferences suite into their own file, written off the main thread, with
  a one-time migration and pruning of copies gone from both sides. Collection
  visibility is synced in one pass and progress updates at most ten times a
  second. A retained-input change on a source with no gallery change is picked
  up by the first Sync after Mold Studio relaunches.
