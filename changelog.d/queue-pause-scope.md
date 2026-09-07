- **A job you pause says it was paused, not that the queue restarted.** A row
  someone paused and a queue parked by a server restart both arrive as
  `paused`, and every client captioned both "Paused after restart" — so
  pausing one waiting job read as though the whole queue had stopped.
  `GET /api/queue` now reports `explicitly_paused` on a paused row, and the
  Queue view and sidebar rail say "Paused" for a job you paused. Dispatch was
  already correctly scoped, and the three pause controls — whole queue, one
  machine, one job — are now pinned by tests on both sides.
