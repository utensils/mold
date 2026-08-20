- **TUI Library respects the gallery trash.** The Library lists only live
  prints (trashed ones stay out until restored), shows a print's editable
  title above the prompt in the Details panel and full detail view, and `d`
  moves the selected print to the trash instead of deleting it — the local
  DB-backed `.trash/` move with a tombstone, or the owning server's trash —
  with the hint and confirm copy honestly falling back to permanent-delete
  wording whenever any owning machine cannot trash (DB-less local scan,
  older server, capabilities not yet read). Settings gains a Library ▸
  Trash (days) row editing the shared `gallery.trash_retention_days`
  retention window (0 = keep forever).
