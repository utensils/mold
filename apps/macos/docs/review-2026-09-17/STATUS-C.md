# Lane C · Library — ledger

Branch `lane-c-library`, off `bc96d415`. Findings from `03-library.md`.

| id | status | commit | test |
| --- | --- | --- | --- |
| H1 path traversal | fixed | `b3cdcbdd`, `ab997d18` | `SafeFilenameSuite` (17 refusals + containment), `GalleryListingSuite`, `PrintMaterializerTests` |
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

Housekeeping: `f9d71ed3` splits the five files this work pushed past the
150-line advisory and updates `apps/macos/README.md`; `ab997d18` is an H1
follow-up (a folded cache key has to leave room for the machine's UUID beside
it, or a long `media_version` made a component nothing could create and the
print materialized as nothing).

Verified on this branch: `make lint` clean (the three pre-existing large types
only), package `swift test` 443 tests green, app bundle `xcodebuild test` 408
tests in 62 suites green.

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
4. **M1's echo window closes when the chain does, not a moment later.** An
   echo that arrives AFTER its entry settles is no longer recognised as ours,
   so a `gallery_updated` carrying no row would cost one extra listing. mold
   records an organization mutation before it emits, so the row is normally
   present and the frame is applied in place; the alternative -- a grace period
   after each edit -- is timing state in exchange for a GET this app already
   makes on every ⌘R.
5. **H1 found one more site than the report did**: `media_version` is
   server-supplied too and is the only thing this app makes a DIRECTORY name
   out of. It is FOLDED rather than refused (a print with an odd version is
   still a print).
6. Five files crossed the 150-line advisory as a result of this work and were
   split (`LibraryViewer+Mesh`, `LibraryGrid+Selection`, `LibraryStore+Rows`,
   `LibraryPane+Wiring` absorbing the pane's tail, `Shell/LibrarySelection`).
   `LibraryStore`'s TYPE total rose by ~11 lines (`revision`, `echo`, one
   comment); the new behaviour itself is in new types.

## Round two — the adversarial review (`REVIEW-C.md`, 15 findings)

| id | status | commit | test |
| --- | --- | --- | --- |
| C-8 the budget measured one file per folder | fixed | `46162dad` | `afolderHoldingSeveralPrintsIsMeasuredWhole`, `anEmptyFolderIsAccountedForAndSweptAway` |
| C-6 an edit could be stranded by the drain's trailing awaits | fixed | `add8fa79` | `anEditMadeDuringTheTrailingRelistStillReachesTheMachine` |
| the test that could not fail | fixed | `add8fa79` | `aMachineWhoseFrameWasSkipped…` now holds the mutation open and asserts the ORDER |
| C-14 M5 left two of four fields unfixed | fixed | `4d827066` | `TextEditingFocusTests` |
| resync relists were unbounded (Lane D seam) | fixed | `885c9799` | `RelistGateTests` (K markers → 2 reads) |
| C-4 dead `json` ceiling, overstated guarantee | fixed | `ad6d3201` | `collectingStopsAtTheCeilingRatherThanAfterIt`, `anUnboundedAnswerIsRefusedRatherThanDecoded` |
| C-1 the shipped thumbnail session was untested | fixed | `ad6d3201` | `ThumbnailCacheTests` now builds it through `init(stubbing:)` |
| C-2 containment did not resist a symlink | fixed | `ef67a573` | `aSymbolicLinkIsNotAFreshDestination`, `aSymlinkedDirectoryResolves…` |
| C-5 a legal `a%2Fb.png` was refused | fixed | `ef67a573` | `anOrdinaryPrintNameIsKept` |
| C-3 multi-save destroyed an existing file | fixed | `c7d42583` | `SaveNamesSuite` (7 cases) |
| C-11 a refusal wiped every undo entry | fixed | `e3ab9d91` | `aRefusedEditTakesBackItsOwnInverseAndNoOthers` |
| C-9 Quick Look pinned folders forever | fixed | `9c76fa48` | `quickLookLettingGoUnpinsWhatItWasShowing` |
| C-9b a spared entry stopped eviction | fixed | `9c76fa48` | `whatIsSparedComesOffTheBudgetRatherThanOutOfTheReckoning` |
| C-10 `enforceBudget`'s doc contradicted it | fixed | `9c76fa48` | — (prose) |
| C-7 a relist discarded what landed during it | fixed | `7c72a50b` | `aPrintLandingDuringARelistSurvivesIt`, `aRelistStillDropsWhatTheMachineNoLongerLists` |
| C-12 caveat, the calendar day | fixed | `72615319` | `aNewDayReDerivesTheCut` |
| C-13 the unreachable `?? ""` | fixed | `72615319` | — (dead branch removed) |
| C-15 two menus that disagreed | fixed | `112c8974` | `LibraryMenuPlanSuite` (12 cases), `theMenuBarOffersThePlanAndNothingOfItsOwn` |
| `try?` hiding real failures | fixed | `ef67a573`, `c7d42583` | the write and both save paths report now |
| `LibraryStore` grew | partly given back | `56a458e9` | — |

### Decisions a second reviewer should look at

1. **C-12 (a)**: the design stands as the coordinator confirmed -- no day
   headings under a non-chronological sort, the way Photos does it. Only the
   caveat (the calendar was not in the cache key) was a defect.
2. **C-4**: the thumbnail route is now bounded AS IT READS, because this lane
   owns that session. The media route is still bounded on RETENTION only and
   the type says so in as many words: the allocation bound is
   `HTTPBackend+Transport.swift`'s to give, and `ResponseCeiling`'s doc names
   it. Nothing dead is left.
3. **C-11**: a per-edit `UndoToken` is the target, so `removeAllActions(withTarget:)`
   removes exactly one entry. `forget()` survives for the case that really does
   invalidate the stack (a tag deleted everywhere) and walks this store's own
   tokens.
4. **C-15**: `Share` is deliberately outside the plan -- it is a `ShareLink`,
   a system control rather than an action this app performs -- and the menu bar
   keeps its three chords, which a contextual menu has no business carrying.
   Both surfaces draw everything else from `LibraryMenuPlan`.
5. **The size rule**: `LibraryStore`'s type total is 894 against 834 at the
   branch point. The review's own fixes put behaviour there (the outer drain
   round for C-6, the relist seam for C-7, the echo and the gate). What could
   leave has left -- `LibraryRevision`, `RelistMerge`, `TagRewrite`, and
   earlier `GalleryEcho`, `RelistGate`, `LibraryShowingCache`, `SaveNames`,
   `TextEditingFocus`, `LibraryMenuPlan` -- and the residual is declared here
   rather than hidden.

### Cross-lane edits added in this round

- `Tests/MoldTests/FakeBackend.swift` -- a per-route `delays` knob and the two
  routes that await it. Additive: a route with no entry is instant, so no
  existing test changes behaviour. It is what makes "something happened WHILE
  this call was in flight" testable at all.
- `Sources/Mold/Shell/{LibraryCommands,LibrarySelection,CollectionRow}.swift`
  -- C-15 and C-14.
- `Sources/Mold/MoldApp.swift` -- one `.task` starts `TextEditingFocus`.
- `Sources/Mold/Support/{MoldUndo,MoldAppDelegate}.swift` -- C-11 and a stale
  comment.
