- **ZFS hosts stop refusing renders for memory ZFS will give back.** Host-RAM
  admission now counts evictable ZFS ARC — OpenZFS's own reclaimable figure,
  `min(Σ mru/mfu evictable, size − c_min)`, zero while ZFS is self-evicting or
  its shrinker is rate-limited — beside `MemAvailable` on every path: the
  scheduler ledger, MiniMax H3 admission, the between-eviction re-sample, and
  `mold run --local`. Refusals, `/api/status` and `GET /api/queue` host
  telemetry (`host_memory.reclaimable_zfs_arc_bytes`), `/api/resources`, the
  Machines pages, and the TUI name the credit whenever it is above zero;
  `MOLD_HOST_RAM_ZFS_ARC=0` disables it
  ([#1439](https://github.com/utensils/mold/issues/1439)).
