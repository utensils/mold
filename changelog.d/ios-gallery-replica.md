- **iPhone and iPad Library follows each machine's gallery change feed.** Mold
  Studio Companion now keeps one replica per machine, as the Mac app does:
  where a machine advertises `capabilities.gallery.changes_feed` it reads only
  the prints that changed since its saved cursor, and older machines keep the
  conditional listings, diffed into the same changes. Gallery events no longer
  re-read the whole machine; a burst of them is one coalesced catch-up, a
  return to the foreground pulls once, and a refresh that finds nothing new no
  longer redraws the grid. New and edited prints are applied without
  re-merging the whole library, collections are re-read only when something
  says they changed, and the offline Library moves into the replicas' saved
  state on first launch after the update.
