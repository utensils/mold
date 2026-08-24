- **Overlapping MiniMax H3 prints now wait in the queue for host memory.**
  Placement preview reports memory held by an active GPU render as a
  non-authoritative transient condition, allowing compatible clients to use
  their existing direct-queue fallback. Admission remains pending while that
  owner runs, then retries from a fresh sample after it settles. Idle cached
  models are still reclaimed immediately before the request waits.
