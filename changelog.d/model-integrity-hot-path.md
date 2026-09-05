### Fixed

- Avoid full model integrity scans when queueing, preparing, switching, or reloading complete installed models, including legacy installations and server restarts. Verify new downloads and derived outputs once; retain explicit `mold info MODEL --verify` checks.
