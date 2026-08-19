- **`mold trash` and `mold run --title`.** `mold trash list|restore|empty|sweep`
  inspects and acts on the serving host's gallery trash over HTTP (`list`
  shows filename, title, when trashed, the purge countdown — `in 27d`, `kept`,
  or `due` — and size; `--json` for the raw rows; `empty` confirms unless
  `--yes`). `mold run --title "…"` names a print at creation: the title is
  validated at parse time, embedded in the output metadata, seeded into the
  gallery row, and folded into the default filename as
  `mold-{model}-{timestamp}~{slug}.{ext}`. `MoldClient` gains typed methods for
  every library-organization and trash endpoint (`list_gallery_view`,
  `trash_gallery_image`, `delete_gallery_image_forever`, `restore_trashed`,
  `empty_trash`, `sweep_trash`, `patch_gallery_image`, `organize_gallery`,
  collections and tags CRUD).
