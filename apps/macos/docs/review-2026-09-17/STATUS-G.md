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
86,400-second chunks while the sentence under Recently Deleted asked
`dateComponents` between two instants — which is the same elapsed count with
daylight saving folded in, NOT calendar days, as the second review round
caught. `TrashCountdown` now counts midnights in the viewer's own calendar, so
a purge at 01:00 tomorrow read at 23:00 tonight is 1 day on all three surfaces
instead of "today" on all three. The badge's tooltip also stops saying "Purged
in 0 days" under a badge reading "today".

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

## Adversarial review, second round

| # | finding | status | commit | test |
| --- | --- | --- | --- | --- |
| 1 | a multi-file import ABANDONED the batch on the first unreadable file | fixed — `continue`, reported once after the batch, and the read moved off-main | `dc0bee5` | `LibraryImportTests` (3) |
| 2 | every separator shared one `ForEach` identity | fixed — a drawn row is keyed by POSITION; `RowAction` is no longer `Identifiable` | `c832c96` | `RowActionSuite.aDrawnMenuRepeatsItselfAndIsKeyedByPosition` |
| 3 | the Installed-models menu lost its `host` gate | **not a bug** — the gate is in `ModelsPane.menuItems(for:)` (`+Actions.swift:37`), which is `guard let host else { return [] }`; the review compared against the VIEW's copy of the same gate. It cannot be unit-tested without a view hierarchy (`@Environment` stores), so it is stated here rather than pinned | — | — |
| 4 | `KeyframeTable` still swallowed its import failure | fixed — the same caption its two siblings draw | `2afc861` | — (a `@State` caption) |
| 5 | both source scans could pass vacuously | fixed — a `count >` floor on each, the menu scan widened to `Packages/*/Sources`, the well scan from `Generate/` to all of `Sources/Mold` | `38e8805` | the scans themselves |
| 6 | `TrashCountdown` did not count calendar days | fixed — it counts MIDNIGHTS now, in the viewer's calendar, which is what every claim said | `f345e23` | `TrashCountdownSuite` (5) |
| 7 | the ledger said "cross-lane edits: none" | corrected below | this file | — |
| 8 | Discover's Open Page lost the `.link` trait | LEFT — see below | — | — |
| 9 | an orphaned doc comment | fixed | `1562c65` | — |
| 10 | the fake's latch was sticky | fixed — arming the hold clears it | `1562c65` | — |
| 11 | two `rendered` edge cases | the policy asymmetry is now stated in the doc; the submenu-of-submenus case has no surface and is left | `1562c65` | — |

**#1 is the one that mattered.** The sweep's fix reported the failure and then
`return`ed from inside the loop, so ten files with a bad one second in imported
ONE and never attempted eight. A read failure is about THAT FILE and now
`continue`s; a refused UPLOAD is about the MACHINE and still stops. Both are
pinned. The test then found a second half nobody had looked for: every
successful import calls `hosts.succeeded(on:)`, which clears that machine's
failures, so a report made mid-loop is wiped by the next file that works. The
batch reports ONCE when it is done, naming the file or saying how many.

**#2**: `RowAction.id` was `kind ?? title` and every separator is
`RowAction(title: "")`, so the Queue menu (up to four dividers) and the adapter
row (two) handed SwiftUI the same identity repeatedly. The conformance is gone
rather than patched, because nothing in the VALUE can tell two dividers apart;
`RowActionMenu` keys its `ForEach` on the offset. The test that should have
caught it was written over a list rendering zero separators.

**#8, left**: `Link` → `Button` loses the `.link` accessibility trait and the
`\.openURL` environment. Restoring it would put the one control the shared
renderer cannot draw back inside a menu, which is the thing this lane removed.
Worth a follow-up that models a link IN the shared type (a `RowAction` whose
kind is a URL the renderer draws as a `Link`) rather than an exception here.
The review's related note — neither spelling validates the scheme, so a
server-supplied `file://` would launch — is real and is NOT a regression of
this lane; it wants its own commit.

## Cross-lane edits

One, and it is not in the brief's list: `Sources/Mold/Machines/PairingSection.swift`
(the paired-client row's `.contextMenu { Button("Revoke…") }` became
`.rowActionMenu`). The engine lane has since landed `MoldEngine.isPairable` in
that neighbourhood — if its guard is in the client row this is a textual
conflict on adjacent lines; if it is in the pairing-code section it merges. The
integrator resolves it; this lane did not rebase.

Nothing else outside this lane's owned paths was touched — no `Engine/**`,
`rust/**`, `Makefile`, `project.yml`, `scripts/**`, `Info.plist`, `flake.nix`,
`.github/**`, `MoldApp.swift` or `MoldAppDelegate.swift`.

## For the integrator

- `Generate/SizeMenu.swift` is renamed to `SeedControl.swift`; a lane that
  touched it will conflict on a rename rather than on a line.
- `RowAction` moved from `Sources/Mold/Shell/RowAction.swift` into MoldClient.
  Any lane adding a surface with `.rowActionMenu` needs no change; a lane that
  added a hand-written `.contextMenu` will fail `MenuSurfaceTests`.
