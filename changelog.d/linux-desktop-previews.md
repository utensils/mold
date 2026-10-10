- **Desktop previews load again on Linux.** In dev builds every reconnect leaked
  an `/api/events` stream, because the events store keyed its bookkeeping on
  Pinia's per-call `this` proxy and never aborted the stream it replaced; six of
  them filled WebKitGTK's per-host connection pool, so every viewer request to
  that machine (media tickets, clips) queued forever behind "Loading preview".
  The store now keys on the raw store, and `desktop-dev` exports the GStreamer
  plugin path the packaged app gets from `wrapGAppsHook3` — without it WebKit
  aborted the web process the moment a clip loaded.
- **Clips with sound stream in every player.** The AAC mux wrote all video
  samples and then all audio, so a front-to-back demuxer (WebKitGTK on Linux)
  had to buffer the whole picture track before its first audio sample and never
  started playback. Samples are now written in presentation order, interleaving
  the tracks in one-second chunks. Clips rendered before this change keep the
  old layout.
