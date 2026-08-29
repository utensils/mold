- **Desktop Library stays smooth past 1 000 prints.** The gallery store now
  indexes each print's organization and cross-host copies once per data change
  instead of re-deriving them for every tile on every render (one filter change
  used to run ~10 full passes over the library, and each visible tile scanned
  every bucket), the grid renders one flat print-keyed tile layer so a
  thumbnail-size drag or window resize moves tiles instead of remounting their
  media, a 304 poll no longer invalidates the merged grid, gallery rows are
  raw immutable snapshots rather than deep-reactive proxies, the thumbnail
  scheduler dispatches in O(hosts) instead of re-sorting the whole queue, and
  History ▸ Runs is capped like Sequences. Operation-count regression guards
  (`gallery.perf.test.ts`, `LibraryView.perf.test.ts`, the scheduler drain
  test) fail CI if any of these hot paths regress.
