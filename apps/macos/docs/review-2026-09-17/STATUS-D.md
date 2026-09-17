# Lane D — Queue / Models / Machines / events

Branch `lane-d-queue`, worktree `.claude/worktrees/agent-aee5eb5b6bf49344c`.
Source: `04-queue-models-machines.md`, minus M6 (Lane A fixes its root cause in
`QueueListing.merged`), M7 (`/api/activity` — a Wave 3 feature lane) and L3
(ATS / self-signed TLS — Lane F).

| id | status | commit | test |
| --- | --- | --- | --- |
| H1 pairing code always "Expired" | fixed | `245c2bed` | `PairingTests.theCountdownReadsTheMachinesOwnUnixSeconds` (+ the two existing cases re-based on seconds) |
| H2 a reconnect never reconciles, nothing polls | fixed | `02f8a8bf` | `HeartbeatTests` (6 cases) |
| M2 `catalog_ready` phantom download row | fixed | `371d606c` | `DownloadEventEffectTests` (6), `DownloadStoreTests.aCatalogReadyFrameLeavesNoRowBehind`, `.aProgressFrameForAnUnknownJobCreatesNothing`, `.anEnqueuedFrameStillIntroducesAJobThisAppNeverStarted` |
| M3 row Pause/Resume ungated | fixed | `e91b996e` | `QueueRowActionsTests.pauseAndResumeAreOfferedOnlyWhereTheMachineAdvertisesThem`, `.aRunningRowIsNotOfferedPause` |
| M4 `cooperative_cancellation` never read | fixed | `e91b996e` | `QueueRowActionsTests.aRunningRowIsCancellableOnlyWhereTheMachineCanStopWorkSafely`, `.aWaitingHeldOrPausedRowIsCancellableEverywhere`, `QueuePaneTests.aGroupCancelSkipsARunningChildTheMachineCannotStop` |
| M5 two hydrations double-notify | fixed | `946febda` | `NotificationTests.twoOverlappingHydrationsNotifyAboutOneFailureOnce` |
| M6 held/paused rows re-shuffle | not mine | — | Lane A, `QueueListing.merged` (01#9) |
| M7 `/api/activity` | deferred — Wave 3 | — | feature lane F3, not a fix |
| M8 first notification dropped | fixed | `3020cb65` | `NotificationTests.theFirstNotificationWaitsForTheAnswerInsteadOfBeingDropped`, `.authorizationIsAskedForOnceAndTheRestJustArrive` |
| L1 refusal quotes "about 48 MB" | fixed | `fc94ae34` | `MoldClientTests.aBodyTooLargeSaysWhatTheLimitActuallyIs` |
| L2 `awaitSettlement` spins when cancelled | fixed | `5eb07872` | `QueueHoldTests.aWaitForADownloadThatNeverSettlesIsBoundedAndCancellable`, `.aCancelledWaitEndsRatherThanSpinning` |
| L3 blanket `NSAllowsArbitraryLoads` | not mine | — | Lane F |
| L4 Dock badge on a view modifier | fixed | `e3f4fd7f` | `LandedPrintsTests.theBadgeFollowsTheCountWithNoViewInvolved`, `.anEmptyCountPaintsNoBadgeAtAll` |
| context menus — Queue | added | `e91b996e` | `QueueRowActionsTests.theContextualMenuMatchesTheQueueMenusWordsAndOrder`, `.aBatchsContextualMenuNamesEveryJobItReaches` |
| context menus — Models | added | `ee0c0f7e` | `ModelActionsTests.deleteIsLastAndBehindADividerWhereverItIsDrawn`, `DiscoverTests.aDiscoverRowsMenuCarriesTheSameActionsItsCellsDo` |
| context menus — Machines | added | `361742c6` | `DeviceControlTests.aCardsMenuOffersExactlyTheControlItsRowDraws` |
| flaky app tests (coordinator) | fixed | `654102a8` | see below |

## Contextual-menu audit

Every row/tile/list in the three panes, and what it offers now. The rule
throughout: ONE declaration read by the row's inline controls, its contextual
menu and the menu bar; destructive items last, behind a divider, role
`.destructive`; absent rather than disabled.

| surface | before | now | declared once in |
| --- | --- | --- | --- |
| Queue row | Try Again, Move Up/Down, Cancel Job — wrong order vs the Queue menu, no Pause/Resume, no divider | Pause Job, Resume Job, Try Again, Move Up, Move Down, divider, **Cancel Job** | `QueueRowActions.offered` (rendered by `.rowActionMenu`) |
| Queue batch row | Move Up/Down only | Pause Every Job, Resume Every Job, Move Up, Move Down, divider, **Cancel Every Job** | `QueueRowActions.groupOffered` |
| Queue held row | already correct (`d9421ae0`) | unchanged | `QueueHoldRow.menuTitles` — the ONE parallel copy left, because `Move to ▾` is a submenu `RowAction` cannot express (D6) |
| Queue batch CHILD row | none of its own | the child's own `QueueRowActions` | same |
| Models ▸ Installed row | the Model menu's list, Delete… with no divider | same list, **divider before Delete…** in both menus | `ModelActions.Item.startsGroup` |
| Models ▸ Discover row | **none** | Details…, Install, or Open Page | `DiscoverRow.menuItems(for:)` |
| Models ▸ download row | **none** | **Cancel Download** on a live row; none on a finished one | inline in `DownloadsPopover` |
| Machines ▸ machine row | Show in Library, Check Now | Check Now, Set as Default, divider, Show in Library | `SidebarMachineActions`, which `MachineCommands` reads too |
| Machines ▸ GPU row | **none** | Use / Stop Using <card>, or Enable at next restart | `DeviceControl.menuItem(named:)` |
| Machines ▸ paired phone | **none** | **Revoke…** | inline; the row's link gains the same ellipsis |
| Machines ▸ nearby machine | **none** | Add / Add… | shared `add(_:)`, one `PeerAction.resolve` |

The AX trap holds: `AXShowMenu` on a SwiftUI `List` row answers -25206, so none
of this can be photographed from the harness. Every menu is therefore pinned by
the pure list the VIEW renders (`offered`, `menuItems`, `menuItem(named:)`) --
never a second copy of it beside the view. See D6 below for why that
distinction had to be made after the first pass.

## Behaviour changes worth naming

- **Pause is no longer offered on a RUNNING row** (`e91b996e`).
  `set_one_queue_job_paused` refuses one by name — "queue job {id} is already
  running; only waiting jobs can be paused or resumed" (`routes.rs:7706-7710`)
  — so the button was a guaranteed 409, and web already reads the same two
  states (`useQueueInspection.ts:341-348`). Not in the review's text; found
  while wiring M3's gate.
- **A Discover row's double-click was eating its own selection**
  (`ee0c0f7e`). `.onTapGesture(count: 2)` on a `Table` cell is the same trap
  M8 hit on a `List` row; `.simultaneousGesture(TapGesture(count: 2))` now
  leaves the single click alone.
- **A paired phone's inline Revoke gains an ellipsis** (`361742c6`) — it
  opens a dialog, which is the rule "Delete…", "Components…" and "Empty
  Queue…" already follow, and it keeps the link and the menu item identical.
- **The Queue menu's Try Again stays narrower than the row's**
  (`e91b996e`, deliberate): the menu resolves the row's BATCH CHILD, which is
  the only place `error_code` and the host's own `retryable` live
  (`routes.rs:2951-2956`); a missing-model hold's resolvable action is
  Pull-then-Retry, which needs the download store a menu item cannot carry.
  Documented at the call site.

## The flaky app tests (`654102a8`)

- `QueueStoreLiveTests.aJobFrameReReadsThatMachineOnceForABurst` and its
  sibling `aBulkCommitFrameReconcilesTheMachineOnce` settled on
  `callCount("queue") == 1`. The fake records a call BEFORE the store has
  applied its answer, so that count is satisfied by a read whose result is not
  in `byHost` yet — the assertion then read `[]`. Both now settle on the rows
  themselves. The read count was briefly widened to `1...2` here; D3 below
  reverts that — it was justified by a resync that cannot occur in those
  tests, and the real second read came from a coalesce delay shorter than the
  burst.
- `ModelActionsTests.deleteAsksFirstAndTheFakeRecordsNothingUntilPerform`
  settled on `callCount("deleteModel")`. The fake records and appends in one
  turn there, so the count was not the race — the wait was simply too short.
  It now settles on `fake.deletedModels` and asserts the count afterwards.
- `settle(until:)`'s budget went from ~0.5 s to ~2 s — and see D3: what needs
  it is a real 500 ms coalesce, not a bad settle. It returns the instant the
  condition holds, so a green run costs exactly what it did before. No sleeps
  were added anywhere; every timing constant this lane introduced is a
  constructor parameter (`HostHeartbeat(interval:)`,
  `awaitSettlement(within:polling:)`).
- `RunQueueTests.aSecondGenerateWhileOneRunsIsAdmittedAndQueued` (Lane B's
  file) and `QueueTransferTests.a413SaysTooLargeAndLeavesTheSourceHeld` had
  the same shape and came out of the full run red too; both are fixed the
  same way. The 413 case also pinned the "about 48 MB" sentence L1 replaced.
- Neither exposed a real race in a store.

## Files split past the 150-line advisory (`654102a8`)

This pass pushed seven files over it. New siblings, no behaviour change:
`QueueStore+Actions.swift`, `QueueStore+GroupAction.swift`,
`QueuePane+Rows.swift`, `QueueBatchRow+Move.swift`,
`MoldNotifications+Delivery.swift`, `DiscoverTable+Cells.swift`,
`DownloadsPopover+Rows.swift`. A few members lost `private`, each with a
comment saying why (`private` does not cross a file boundary).

## Cross-lane edits

Smallest possible, listed so the integrator can sequence them:

- `Packages/MoldClient/Sources/MoldClient/TransferPlan.swift` (Lane A) — one
  sentence now interpolates `RequestBodyLimit.sentence` (L1).
- `Packages/MoldClient/Tests/MoldClientTests/PairingWireTests.swift` and
  `Fixtures/pairing-session-keyed.json` (Lane A) — the hand-built keyed
  fixture carried a millisecond-shaped `expires_at`; re-based on seconds with
  the assertions (H1).
- `Sources/Mold/Shell/ModelCommands.swift` (unowned) — one `if
  item.startsGroup { Divider() }`, so the Model menu and the row's menu stay
  identical.
- `Sources/Mold/MoldApp.swift`, `Sources/Mold/Support/MoldAppDelegate.swift` —
  the composition root and the delegate, for `HostHeartbeat` and `DockBadge`
  (both reached by the brief's "notification / Dock-badge files" grep).
- `Tests/MoldTests/FakeBackend.swift` — `batchStatusesYields` (M5) and
  `settle(until:)`'s budget; `Tests/MoldTests/FakeFixtures.swift` — a
  `cooperativeCancellation` axis on the queue capabilities overload;
  `Tests/MoldTests/FakeNotificationCenter.swift` — it now models a dropped
  request and a deferred answer (M8).
- `Tests/MoldTests/RunQueueTests.swift` (Lane B) — one settle, on the state
  the assertion reads rather than a call count; no production change.
- `Sources/Mold/Shell/RowAction.swift` — BYTE-IDENTICAL to Lane E's copy on
  `feat/macos-native-app`, added here so this lane could adopt it rather than
  invent a third menu abstraction. Two identical adds collapse; do not merge
  them by hand. Lane E's `RowActionTests.swift` was NOT copied (it references
  Lane E's own types).
- `Sources/Mold/Shell/MachineCommands.swift` (unowned) — its two button
  titles now read `SidebarMachineActions`' constants (D7).

## Adversarial review (`REVIEW-D.md`), second pass

| id | status | commit | test |
| --- | --- | --- | --- |
| D1 the `(64 MB)` sentence vs Lane A's landed 413 test | integrator | — | the literal to change is `TransferAdmitFailureTests.swift:77-78` → `"request (\(RequestBodyLimit.sentence)). "`; that file is not in this worktree |
| D2 resync storm — `refresh(on:)` unthrottled | fixed | `15f2a090`, `43ba9a68` | `QueueStoreLiveTests.aStormOfRefreshesReadsTheMachineTwiceNotOncePerCaller`, `.oneRefreshIsOneRead`, `.aBurstOfResyncMarkersRepairsTheMachineWithoutAReadEach` |
| D3 the `1...2` widening | fixed | `15f2a090` | both burst tests back to `== 1` |
| D4 capabilities never re-fetched for an up host | fixed | `6249c218` | `HeartbeatTests.anUpMachineThatNeverSaidWhatItCanDoIsAskedAgain`, `.aMachineThatKeepsRefusingItsCapabilitiesIsAskedLessOften` |
| D5 `tick()` sequential | fixed | `6249c218` | `HeartbeatTests.oneSlowMachineDoesNotHoldUpTheRest` |
| D6 menu tests pinned a parallel copy | fixed | `43ba9a68` | `QueueRowActionsTests.theContextualMenuIsTheListTheViewDraws`, `.theRowsMenuAndTheQueueMenuOfferTheSameWords`, `.aBatchsContextualMenuNamesEveryJobItReaches` |
| D7 `MachineRow` duplicated `MachineCommands` | fixed | `43ba9a68` | `DeviceControlTests.aMachineRowOffersTheMachineMenusOwnItemsFirst` |
| D8 `wakeObserver` write-only, never removed | fixed | `6249c218` | `HeartbeatTests.aCancelledTickAsksNothing` (the tick half); the removal is a `deinit`, pinned by construction |
| D9 `awaitSettlement`'s doc named absent parameters | fixed | `6249c218` | doc only |

**D2's shape.** `SingleFlight` is its own small type rather than more of
`QueueStore`, which the fix pushed to 615 lines across its files and back
under the 600 budget. One run at a time per machine, at most one PROMISED
behind it: twelve callers are two reads, one caller is one. `poll` and
`hydrateNow` check `Task.isCancelled` too, so a superseded read really stops.

**D3's real cause.** The `1...2` was justified by a resync that cannot occur
in those tests -- `FakeBackend.events()` is unbounded and never goes through
`yieldOrResync`, and neither test emits an `authority` frame. The 64-frame
test's second read came from a 5 ms `coalesceDelay` shorter than the burst;
it is 500 ms now, and both assert `== 1`. The resync path has its own tests,
where a marker is actually delivered. `settle(until:)` keeps its 2 s budget,
and its doc now names what needs it (that 500 ms wait) rather than implying it
is a cure for a bad settle.

**D6's shape, and Lane E's `RowAction`.** It FITS, so this lane adopted it
rather than inventing a third abstraction: `Sources/Mold/Shell/RowAction.swift`
here is BYTE-IDENTICAL to the copy on `feat/macos-native-app`, so the two
adds collapse at integration. `RowActionTests.swift` was NOT copied -- it
references Lane E's own types. Nothing is missing from the type; the one thing
it cannot express is a SUBMENU, which is why `QueueHoldRow` keeps its hand-
written menu and its `menuTitles` (its `Move to ▾` is a `Menu`). That is the
one parallel copy left in this lane, and it is named rather than hidden.
`ModelActions.Item` predates `RowAction` and is the shape it was modelled on
(`RowAction.swift`'s own doc says so); its view renders from that list, so it
is not a parallel copy -- collapsing the two types is a follow-up, not a fix.

**A settled row now opens NO menu.** `.rowActionMenu` attaches none when the
list is empty, so a right-click on a finished row no longer opens an empty
one. That is Lane E's rule arriving with the type.

## Verification

`make lint` clean (no file over 150 lines, no new type over 600).
`swift test` in `Packages/MoldClient` — 420 tests, green.
Full `xcodebuild … test` on the app bundle — 420 tests in 58 suites, green
(re-run after the review pass).

New MoldClient files (allowed): `DownloadEventEffect.swift`,
`Capabilities+Queue.swift`, `RequestBodyLimit.swift`, and their tests.

## Nothing judged wrong

Every finding in this lane's list reproduced against the code as written. Two
were larger than the report said: M2's phantom row is created by `dequeued`
too, not only `catalog_ready`; and M3's ungated Pause was also offered in a
state the server refuses outright.
