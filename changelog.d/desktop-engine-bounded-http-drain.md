- **The embedded desktop engine now stops even when a client stops reading.**
  Graceful shutdown waits for every in-flight HTTP request, and a client the
  app does not control — a paused video that stops draining its socket, a
  request whose body never arrives — could hold the engine open past the app's
  stop budget, leaving "gallery authority remains with the server". The
  engine now gives in-flight requests a short grace after a stop is requested
  and then finishes stopping. The same mechanism was skipping the desktop
  nightly: its boot test left a megabyte-sized `/api/models` response unread,
  so whether shutdown completed was up to the kernel's socket buffers
  ([#1582](https://github.com/utensils/mold/issues/1582)).
