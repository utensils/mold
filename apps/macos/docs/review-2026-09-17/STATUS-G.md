# Lane G — cleanup after Wave 1

Consolidation only. No new feature, no key-binding change, and no behaviour a
person can see — except where a menu was silently broken (three empty
contextual menus) or a failure was silently swallowed, both of which are
listed as such below.

## TASK 1 — one menu model

| id | status | commit | test |
| --- | --- | --- | --- |
| 1a One model | fixed | `60b7c52` | `MoldClientTests/RowActionSuite` (6 tests) |
| 1b Generate + Library migrated | fixed | `bcf1298`, `8bfeaf7` | `LibraryMenuPlanSuite`, `GenerateMenusTests` |
| 1c QueueHoldRow + the rest | fixed | `0cccdce` | `QueueHoldTests`, `MenuSurfaceTests` |
| 1d Invariants pinned once | fixed | `60b7c52` | `RowActionSuite` |
| 1e One wording | fixed | `f70a674` | `PrintEditTests`, `LibraryUndoTests` |

**The model.** `RowAction<Kind>` (MoldClient, pure): `kind` (nil on a submenu
and on a separator), `title`, `isDestructive`, `isDisabled`, `children`, plus
`.separator` and `mapKind`. `RowAction.rendered(_:)` is the ONE place the
house rules live — drop a submenu with nothing enabled in it; a list with no
separator of its own gets the house one before its destructive tail; a list
that declares its own grouping keeps its order; never open, close or double on
a divider. `RowAction.offersMenu` decides whether a menu exists at all.
`RowActionMenu` (app, SwiftUI) draws it, with an optional per-kind
`shortcut` because only the menu bar carries chords, and
`View.rowActionMenu(_:perform:extra:)` / `TableRowContent.rowActionMenu` are
the only doors. It lives in MoldClient so a menu is a `swift test`.

**Deleted**: `GenerateAction`'s `GenerateMenuItems`; `LibraryMenuItem` and its
`LibraryMenuItems` renderer (and the plan's own `collapsingDividers`);
`ModelActions.Item` the struct (with `role`, `startsGroup` and a `systemImage`
nothing read); `DeviceControl.MenuItem`; `QueueHoldRow.menuTitles` and its
hand-written `@ViewBuilder`; `QueueSelection.offeredTitles`' hand-maintained
copy of `QueueCommands.body`; `DiscoverRow.Item`'s own `id`/`Identifiable`.
`GenerateMenus`, `LibraryMenuPlan` and `ModelActions.menu` survive as what the
brief allows: per-surface functions returning `[RowAction]`.

**Every `.contextMenu` in `Sources/` is now the one modifier's** —
`MenuSurfaceTests.everyMenuIsAttachedThroughTheOneModifier` scans for it, the
way `NativeUATTests` scans for the UAT hooks. The one thing outside the model
is `ShareLink` (a system control, not an action this app performs); it rides
the door's `extra` slot.

**Three menus were silently wrong and are fixed by construction** — a device
card this app cannot change, a peer already on the list, and a finished
download each attached a `.contextMenu` whose body was empty.

**Two deliberate consequences, said plainly:**

- `DiscoverRow`'s Open Page was a `Link`; it is a `Button` handing the URL to
  `NSWorkspace`. Same browser, same page — a `Link` is the one row the shared
  renderer cannot draw.
- The Queue menu can no longer begin with a divider. Today, a selected job
  offering nothing at all plus an available Empty Queue… drew a leading
  separator; `rendered` trims it.

**Tests changed, and why.** `LibraryMenuPlanSuite` asserted on `id` strings
that belonged to the deleted type; it now asserts the full list of TITLES in
order, which pins the wording as well. `GenerateMenusTests`/`CapsuleTests`
`.all` became `map(\.kind)`. `ModelActionsTests` asserted `startsGroup`; it now
asserts the rendered list puts the separator before Delete — the same promise,
about what is drawn. `DeviceControlTests`, `DiscoverTests`,
`LibrarySelectionTests`, `QueueHoldTests` follow the renamed shapes. Nothing
was weakened; `QueueHoldTests` gained
`givingUpOnAHoldIsLastAndBehindADivider`.

**Wording (1e).** `Favorite` → `Favourite` in `LibraryViewer` (label and both
tooltips), `InspectorActions`, and `PrintChange.actionName` — the Edit menu
offered "Undo Favorite" beside a menu item called "Add to Favourites".
`ResultBar` hard-coded "Save a Copy" while its OWN contextual menu said "Save
a Copy…"; it reads `GenerateAction` now. File ▸ Save a Copy… takes the count
its right-click twin already took. Swept and found clean: Colour, Licence,
Catalogue, Organise, Cancelled, Behaviour, Centre, Grey.

## TASK 2 — back under budget

| type | before Wave 1 | after Wave 1 | now |
| --- | --- | --- | --- |
| `HTTPBackend` | 1130 | 1364 | **1116** |
| `LibraryStore` | 834 | 894 | **751** |

Pure moves into types the two now COMPOSE, never a renamed file.
`HTTPBackend`: `SSEStream`, `APIError`, `RefusalBody`, `RouteEscaping`,
`RouteRequest`, `HTTPRefusal`, `TransportFailure`, `TransportLog` — each its
own small type in its own file, the idiom `PartialBody`, `RedirectGuard` and
`RouteTemplate` already set. `LibraryStore`: `LibraryMutations` (+`Wire`) owns
the outbox and the one drain task per machine — the store is the merged
timeline and what is showing in it; a queue of edits with a retry policy is a
different thing that needs the store to report and repair, so it takes one as
a parameter. `QueueHoldRow`'s menu moved beside it for the file lint.

## TASK 3 — the sweep

Findings from a full read of the five landed lanes. Everything below is in
`704a658` unless noted.

| # | finding | status |
| --- | --- | --- |
| 1 | `dismissResult()` — no caller | deleted |
| 2 | `SubmissionFence.isPending` — no caller | deleted |
| 3 | `ResultHandoff.isHolding` — test-only | kept, documented as the test seam |
| 4 | `LicenseStore.isAccepted` — test-only | kept, documented as the test seam |
| 5 | five unread `Chrome` tokens | deleted |
| 6 | six unread MoldClient accessors | 4 deleted; `canPauseQueue` and `canUpscaleClips` KEPT for Wave 3 |
| 7 | three hand-rolled copies of `PictureImport` | fixed, `9868e96` |
| 8 | three private byte formatters | one `FileBytes` |
| 9 | three "days until purge" computations | one `TrashCountdown` |
| 10 | two pasteboard writes bypassing `Clipboard` | fixed |
| 11 | `QueueSelection.offeredTitles` mirrors `QueueCommands.body` | fixed — one declared list, see below |
| 12–15 | stale comments naming types that never existed | fixed |
| 16 | `SizeMenu.swift` holds only `SeedControl` | renamed |
| 17 | export writes nothing, says nothing | reports |
| 18 | Generate's Save/Copy fail silently | reports |
| 20 | import drops unreadable files silently | reports |
| 21 | a failed host encode wipes the machine list | writes nothing instead of nil |
| 22–24 | wording | see 1e |
| 25 | three "Show in Library" literals | LEFT — a machine row and a finished result are different domains, and a shared constant would tie them together for a coincidence |

**#7 is the one real bug in the sweep.** `ControlPictureWell` offers `.heic` in
its open panel — the format every iPhone photograph arrives in — and never
transcoded it, so the bytes uploaded whole and came back a 422. All three
wells now read through `PictureImport` (or `MediaImport`, for the wells that
hold opaque video/audio bytes), off the main actor, newest pick wins, with the
failure said beside the control. Failing test first:
`PictureImportTests.noWellReadsAFileItself`.

**#9 changes one number.** The tile badge and VoiceOver counted elapsed
86,400-second chunks while the sentence under Recently Deleted counted
calendar days. All three now count calendar days — the sentence's answer, and
the one that is a promise about a date. The badge's tooltip also stops saying
"Purged in 0 days" under a badge reading "today".

**#11**: `QueueRowActions.Kind.title` is the one spelling of all six words, and
`QueueSelection.offered` is the one list the Queue menu draws and
`offeredTitles` reads. `⌘⌫` on Cancel Job survives through the renderer's new
per-kind `shortcut` hook.

## One more, found by the gates

`FakeBackend.releaseStatus()` resumed whoever happened to be parked at that
instant, so a `status()` that had recorded its call but not yet parked its
continuation waited for a wake-up that had already happened. It is a latch
now. Measured once during this lane: one suite, 22 minutes, no output, the
whole app bundle wedged behind it (`0e83c32`).

## Gates

Run LAST, on the settled tree, after every helper had finished:

- `make lint` — layers ok, colour ok, a11y ok, no `large:` file lines.
  `HTTPBackend 1116` (was 1130 before Wave 1, 1364 after) and
  `LibraryStore 751` (was 834, then 894). `GenerateController` dropped off
  the list entirely.
- `cd Packages/MoldClient && swift test` — 628 tests, 26 suites, green.
- app bundle `xcodebuild … test` under the shared lock — 539 tests, 80
  suites, green.

## Cross-lane edits

None. Nothing outside this lane's owned paths was touched — no
`Engine/**`, `rust/**`, `Makefile`, `project.yml`, `scripts/**`, `Info.plist`,
`flake.nix`, `.github/**`, `MoldApp.swift` or `MoldAppDelegate.swift`.

## For the integrator

- `Generate/SizeMenu.swift` is renamed to `SeedControl.swift`; a lane that
  touched it will conflict on a rename rather than on a line.
- `RowAction` moved from `Sources/Mold/Shell/RowAction.swift` into MoldClient.
  Any lane adding a surface with `.rowActionMenu` needs no change; a lane that
  added a hand-written `.contextMenu` will fail `MenuSurfaceTests`.
