### Added

- Let `mold gpu list` inspect local runtime-visible compute devices when the
  loopback server is stopped, and let `mold gpu enable|disable` persist the
  selected stable device preference for the next server start. GPU stable IDs
  and completion-shell names now participate in dynamic shell completion.
- Make `mold ps`, `mold info`, and the idempotent `mold unload` useful with no
  local server running while keeping configured remote-host failures explicit.
