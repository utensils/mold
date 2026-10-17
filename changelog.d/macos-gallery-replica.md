- **Mac Library follows each machine's gallery change feed.** Mold Studio keeps
  one replica per machine: where a machine advertises
  `capabilities.gallery.changes_feed` it reads only the prints that changed
  since its saved cursor (resumed across relaunches), live gallery events and
  reconnects become one coalesced catch-up instead of a rebuild per event, and
  a refresh that finds nothing new no longer redraws or re-sorts the grid.
  Older machines keep the conditional full listing, diffed into the same
  changes. Tile lookups, small edits and new prints no longer re-merge the
  whole library, grid geometry is cached per derived section revision, and
  reloading collections no longer forces every machine to re-send its index.
