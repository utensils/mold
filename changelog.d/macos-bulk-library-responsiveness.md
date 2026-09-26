### Fixed

- Native macOS Studio now shows status for Library bulk operations, sends trash/restore/permanent deletion in bounded batches, lets you stop after the current batch, and reconciles partial or uncertain results instead of restoring stale rows. Bulk trash shares one durable archive commit per server chunk; gallery reads can advance between chunks or individual restore/purge items, and disconnects cannot leave blocking filesystem work outside its publication lock.
