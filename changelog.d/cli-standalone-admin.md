- **CLI administration now works without a running local server.** `mold gpu
  list` inspects runtime-visible devices, `gpu enable|disable` persists stable
  startup preferences, and `ps`, `info`, and idempotent `unload` provide useful
  standalone behavior without masking configured remote-host failures. GPU
  stable IDs and completion-shell names also participate in dynamic shell
  completion.
