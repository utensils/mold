- **Library organization API: titles, favorites, tags, and collections.**
  `PATCH /api/gallery/image/:filename` edits one print's title / favorite /
  tags, `POST /api/gallery/organize` applies the same edit to many prints in
  one transaction, `/api/gallery/collections` (+ `/:id`, `/:id/items`) manages
  manual collections, and `/api/gallery/tags` (+ `/:name`) lists, renames
  (merging), and deletes tags. `GET /api/gallery` rows now carry additive
  `title`, `tags`, `favorite`, and `collections`; new SSE events
  `gallery_updated` and `gallery_collections_changed` announce edits;
  `GET /api/capabilities` advertises `gallery.organize`.
- **Gallery trash with retention.** `DELETE /api/gallery/image/:filename` now
  moves a print to `<output_dir>/.trash/` (tombstone + row flag, thumbnail
  kept, `gallery_trashed` event) instead of deleting it; `?permanent=true`
  hard-deletes. `GET /api/gallery?view=trash` lists trashed prints with
  `trashed_at` / `purge_at`; `POST /api/gallery/trash`,
  `POST /api/gallery/trash/restore` (`gallery_restored`),
  `DELETE /api/gallery/trash` (empty), and `POST /api/gallery/trash/sweep`
  round out the surface, and the media/thumbnail/preview routes still serve
  trashed prints. An hourly sweeper
  (plus one pass at startup) purges prints older than the new
  `gallery.trash_retention_days` setting (default 30, `0` keeps forever,
  `MOLD_GALLERY_TRASH_RETENTION_DAYS` overrides); capabilities advertise
  `gallery.trash { enabled, retention_days }`. With the metadata DB disabled,
  delete stays permanent. `mold clean --older-than` no longer reaches into
  `.trash/`.
- **Titled prints get titled filenames.** A `GenerateRequest.title` is
  validated at admission, embedded in the saved `mold:parameters`, seeded into
  the gallery row, and folded into the output filename as a lossy slug
  (`mold-{model}-{ts}[-{idx}]~{slug}.{ext}`); untitled prints keep their
  byte-identical legacy names.
