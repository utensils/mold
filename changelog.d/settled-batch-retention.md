- **Settled batch summaries are retained for `queue.held_retention_days`, not forever.**
  A batch whose every print has completed, failed, or been cancelled is only a
  receipt for a client reconnecting after a dropped stream, and nothing ever
  purged one, so the `generation_batches` tables grew without bound. The hourly
  queue sweeper now purges fully settled batches once their newest child
  settlement is older than the existing `queue.held_retention_days` (`0` keeps
  them forever), a purged batch answers `404 GENERATION_BATCH_NOT_FOUND` — which
  clients already read as missing without reopening finished work — and
  `POST /api/generation-batches/sweep` runs that pass on demand.
