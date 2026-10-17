- **Gallery change feed for faster library refresh.** New
  `GET /api/gallery/changes?since=<cursor>` returns only the prints that
  changed since a client's last read (trash included, with removals and a
  `reset` signal), advertised as `capabilities.gallery.changes_feed`, so
  library clients can stay current without re-downloading the whole gallery.
  `POST /api/gallery/sync-checkpoint` answers the mirror checkpoint for just
  the named prints, and an unchanged `GET /api/gallery` now answers `304`
  without rebuilding the listing, which makes every existing client's idle
  poll cheap.
