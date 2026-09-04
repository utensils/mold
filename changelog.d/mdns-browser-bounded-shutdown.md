- **A server with unusable multicast now stops when asked.** The mDNS peer
  browser's shutdown joined its worker thread with no bound, and that thread
  woke only when the mDNS daemon closed its channel. On a host where multicast
  never works the daemon may never close it, so the server hung instead of
  finishing shutdown — the desktop app reported that the embedded engine did
  not stop and that gallery authority remained with the server. Shutdown is
  now bounded by mold rather than by the daemon.
