- **iPhone Library organization: collections, tags, favorites, titles, and
  trash.** The Library gains a 44pt Prints | Collections | Trash scope row
  (shown when a connected host advertises `capabilities.gallery.organize` /
  `.trash`), a filtering chip row (♥ Favorites, tags, machines), collection
  cards with drill-in, rename, and a two-step delete, and a Trash scope with
  per-tile purge countdowns, Restore, two-step Delete forever, and a two-step
  Empty trash. Select mode adds Add to collection, Tag, ♥, and a Trash action
  that replaces hard delete on trash-capable machines; the full-screen viewer
  adds an Info sheet editing the title, favorite, tags, and collections, plus
  restore/delete for trashed prints. Every edit fans out to each copy's exact
  Keychain-authenticated machine and reports failures inline. Create gains a
  Title field carried on every mobile request and restored by Use as prompt,
  and host detail gains a Library card editing that machine's
  `gallery.trash_retention_days` (the first mobile `/api/config` client) with
  Prints-in-trash and Empty trash.
