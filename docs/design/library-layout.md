# Continuous Library layout

The four maintained Library surfaces use ordered justified rows, preserving the
listing order and actual media ratios. This is a row layout, not masonry: masonry
columns leave the reading order and horizontal edges uneven. A global dynamic
programming partition could minimize height variance, but would repack earlier
rows when pagination appends and costs more than needed for a thumbnail browser.
We use an O(n) greedy partition, choosing the closer row height on either side of
the target when closing a row. A completed row may grow to 1.5 times the target;
otherwise the crossing item joins it. No third-party runtime dependency is needed.

Research: [Flickr justified layout](https://flickr.github.io/justified-layout/)
separates target height, spacing, tolerance and widow behavior;
[Flickr's photolist architecture](https://code.flickr.net/2015/03/24/much-photos/)
separates geometry from viewport rendering. SwiftUI's
[LazyVStack](https://developer.apple.com/documentation/swiftui/lazyvstack)
provides row-level laziness for native apps. The algorithms here are small,
application-owned implementations with matching Swift and TypeScript tests.

- Final incomplete rows remain left aligned at target height, or smaller if
  necessary to fit. They never stretch a lone portrait across the viewport.
- Positive dimensions are authoritative, including panoramas and very tall
  images. We do not clamp their ratio or crop them. Unavailable, zero or invalid
  dimensions fall back to square until metadata changes. Thumbnail decode cannot
  alter row geometry. Audio and mesh entries follow the same metadata policy.
- Two-point seams and square corners keep rows continuous. Selection and focus
  rings draw inside tiles; hover does not lift a tile out of the row.
- Web uses binary search over actual row offsets and two overscan rows. Desktop
  retains TanStack row virtualization and its flat print-keyed thumbnail layer.
  Both keep decoded nodes when a print moves across row boundaries. Native apps
  use lazy rows and a geometry cache independent of image/selection state.
  Native row cells use print identities across deletion/reordering, and canceled
  thumbnail loads cannot replace the current image.
- Resize and zoom retain a visible print anchor. Browser grids also retain the
  offset into its row. Native pinch retains its initial top print. Viewer return
  retains the exact covered viewport, including a partially visible row.
- macOS Up/Down selects the closest horizontal center in the adjacent row;
  Left/Right and Shift/Command keep the existing ordered selection contract.
  Keyboard reveal requires the row to be almost completely visible; the lower
  visibility threshold used for reflow anchors cannot suppress that scroll.
- iOS captures the visible print before size bindings change. Its Favorites
  rotor uses [explicit per-print targets](https://developer.apple.com/documentation/swiftui/view/accessibilityrotorentry(id:in:))
  and reveals the containing lazy row before VoiceOver moves to a print.
- iOS removes only the visible host-name badge. Details, host scope, other badges,
  preview environment injection and unseen-media semantics remain intact.

Geometry: `studio/lib/justifiedLayout.ts` and
`apps/shared/Packages/MoldClient/Sources/MoldClient/JustifiedLayout.swift`.
Tests cover row width, ratios, ordering, final rows, invalid dimensions,
panoramas, 30,000 items and native keyboard geometry. Extremely thin portraits
close a capped row before seams consume its image area; this rare row may leave
spare width rather than collapsing its tiles to zero height.
