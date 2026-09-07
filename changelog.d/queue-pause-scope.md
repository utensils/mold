- **A job you pause says it was paused, not that the queue restarted.** A row
  someone paused and a queue parked by a server restart both arrive as
  `paused`, and every client — the apps, the queue detail panel, and
  `mold queue list` — captioned both "Paused after restart", so pausing one
  waiting job read as though the whole queue had stopped. `GET /api/queue` now
  reports `explicitly_paused` on a paused row, and every surface says "Paused"
  for a job you paused. The detail panel also stopped reporting a paused row's
  place in a line it is not standing in. Dispatch was already correctly
  scoped, and the three pause controls — whole queue, one machine, one job —
  are now pinned by tests on both sides.
