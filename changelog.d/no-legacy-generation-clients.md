### Changed

- Web, desktop, and iPhone now admit every print — Batch 1, Batch N, and each prepared variation — through the durable `POST /api/generation-batches` queue. The attached `/api/generate/stream` submission path, the mixed-version capability probes (`queue.heterogeneous_batch`, `queue.durable_batch_outcomes`, `queue.admission_protocol_version`), and the 404/405 placement-preview fallbacks are gone from the browser surfaces. A machine that cannot carry a request is now refused inline by name with nothing queued, instead of being silently re-routed to a second submission path.
