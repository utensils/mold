# Lane C · Library — ledger

Branch `lane-c-library`, off `bc96d415`. Findings from `03-library.md`.

| id | status | commit | test |
| --- | --- | --- | --- |
| H1 path traversal | fixed | `b3cdcbdd` | `SafeFilenameSuite` (17 refusals + containment), `GalleryListingSuite`, `PrintMaterializerTests` |
| M1 live events dropped while the outbox is pending | fixed | `1e90154f` | `GalleryEchoTests`, `LibraryStoreLiveTests` |
| M2 the grid ignores Sort By | fixed | `89407056` | `theGridDrawsTheOrderTheQueryAskedFor` (all four sorts), `LibraryGroupingTests` |
| M11 the library is re-derived per body pass | fixed | `89407056` | `LibraryShowingCacheTests` (counting budget) |
| M3 arrows read the last mouse-down | fixed | `dbb47b16` | `LibraryGridKeysTests` |
| M4 a bare ⌫ trashes | fixed | `dbb47b16` | `LibraryGridKeysTests.aBareBackspaceDoesNothing` |
| M5 Space steals from text fields | fixed | `fa3a5767` | `LibrarySelectionTests` |
| M7 `removeAllActions()` wipes the window's stack | fixed | `231bcacf` | `LibraryUndoTests.forgettingOurEntriesLeavesEverybodyElsesAlone` |
| L5 a failed edit leaves its undo entry | fixed | `231bcacf` | `LibraryUndoTests.aRefusedEditLeavesNoUndoEntryBehind` |
| M8 the budget deletes the file it just wrote | fixed | `006641a9` | `PrintMaterializerTests` (oversize, in-use, still-evicted) |
| L4 LRU touch vs access dates | fixed | `006641a9` | `touchingAFileIsWhatDecidesHowRecentlyItWasUsed` |
| M6 a GLB never finishes loading | **symptom only** | `37cbc8a5` | none (view-level; see below) |
| M9 unbounded response bodies | fixed at the call sites | `3bf1ea36` | `ResponseCeilingSuite` |
| L2 the thumbnail cache is uncapped and never purged | fixed | `047ebe54` | `ThumbnailCacheTests` |
| L3 a new collection always on `hosts.first` | fixed | `7122c3f3` | none (one-line picker; `isUp` is Lane D's tested surface) |
| L6 `TitleField` can commit the previous draft | fixed | `7122c3f3` | none (SwiftUI ordering; `.id` removes the question) |
| M10 Reuse + retained source media | **deferred — Wave 3 F2** | — | — |
| L1 export menu from `capabilities.mesh.export_formats` | **deferred — Wave 3 F1** | — | — |

Test gaps named at the end of the report: the outbox gate (M1), the grid's sort
(M2), the modifier source (M3), the bare-⌫ binding (M4) and `PrintMaterializer`'s
path construction (H1) all have tests now; `LibraryShowingTests`' self-derived
assertion is replaced by `sections.flatMap(items) == visible` per sort;
`ThumbnailCache` has a suite with an injected session; `CacheBudget`'s use by
the materializer is covered.

## Deferred, and why

- **M6 (interactive mesh viewer)** is Wave 3 F1. What landed is the symptom fix
  the task asked for: a GLB draws its poster at full opacity with a line saying
  what it is, and stops fetching bytes `NSImage` cannot read. The seam the
  `MTKView` replaces is named in `LibraryViewer+Mesh.swift`.
- **M10 (Reuse + retained source media)** is Wave 3 F2 — it widens
  `OutputMetadata`, which is Lane B's.
- **L1 (export menu from the advertised list)** is Wave 3 F1: it deletes
  `GalleryMutations`' client constants, a Lane A file, and wants the turntable
  sheet that ships with the viewer.

## Cross-lane edits (smallest possible, listed for sequencing)

- `Packages/MoldClient/Sources/MoldClient/GalleryPrint.swift` — the sanctioned
  one: `init(from:)` in an EXTENSION at the foot of the file (an initializer in
  the body would suppress the memberwise init every fixture uses), validating
  `filename`. `OutputMetadata` untouched.
- `Packages/MoldClient/Sources/MoldClient/HTTPBackend.swift` — one line in
  `galleryListing`: `decode(GalleryListing.self)` instead of
  `decode([GalleryPrint].self)`, so one refused row does not blank a library.
- `Packages/MoldClient/Sources/MoldClient/{LibrarySection,LibraryShowing}.swift`
  — M2 lives there (`byDay` re-sorted; `LibraryShowing` chose the grouping).
  `LibrarySort.groupsByDay` is an extension in `LibrarySection.swift` rather
  than an edit to `LibraryToken.swift`.
- `Sources/Mold/Shell/LibraryCommands.swift` + new `Shell/LibrarySelection.swift`
  — M5 needs `isEditingText` on `LibrarySelection`; the values moved to their
  own file to keep both under 150 lines.
- `Sources/Mold/Shell/CollectionRow.swift` — L3, three lines.
- `Sources/Mold/Support/MoldUndo.swift` — M7, one line (`withTarget:`).
- `Sources/Mold/Support/MoldAppDelegate.swift`, `Sources/Mold/MoldApp.swift`,
  `Sources/Mold/Shell/GeneralSettings.swift` — L2 needs the thumbnail cache
  purged by the same two doors as the media cache (a property, a line at each
  door, and the Settings scene's environment).
- `apps/macos/README.md` — the caches paragraph, the Library row (Sort By, the
  mesh poster, Space yielding) and "Not built yet".

## Judgement calls a reviewer should look at

1. **M2's second half is mine, not the report's.** The report's suggested
   assertion, `sections.flatMap(items) == visible` for every sort, CANNOT hold
   while days are grouped: Largest First and Name interleave days, so grouping
   necessarily re-orders. An order days cannot describe is therefore drawn in
   one piece with no heading (`LibrarySection.day` is now optional), which is
   what the Finder does under "sort by size". Newest/Oldest are unchanged.
2. **M9 is at the call sites, not in the transport.** The real fix is
   `download(for:)` on the media routes, which is Lane A's
   `HTTPBackend+Transport.swift`. `ResponseCeiling` states the limits and the
   Library's two unbounded asks go through it; the type's own doc says where it
   belongs.
3. **L5 can only be a targeted `removeAllActions(withTarget:)`.** `UndoManager`
   offers no per-registration removal, so a refused edit drops this store's
   whole stack rather than one entry. Registering after the round trip is not
   an option — the synchronous registration is load-bearing for redo.
4. **H1 found one more site than the report did**: `media_version` is
   server-supplied too and is the only thing this app makes a DIRECTORY name
   out of. It is FOLDED rather than refused (a print with an odd version is
   still a print).
5. Five files crossed the 150-line advisory as a result of this work and were
   split (`LibraryViewer+Mesh`, `LibraryGrid+Selection`, `LibraryStore+Rows`,
   `LibraryPane+Wiring` absorbing the pane's tail, `Shell/LibrarySelection`).
   `LibraryStore`'s TYPE total rose by ~11 lines (`revision`, `echo`, one
   comment); the new behaviour itself is in new types.
