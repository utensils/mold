# Lane F3 — Library upscale, `/api/activity`, whole-queue pause

Worktree branch `worktree-agent-a6134a2a18bde6b88`, cut from `a5172d65`.

| id | status | commit | test |
| --- | --- | --- | --- |
| F3 upscale wire + `defaultUpscaler` policy | fixed | `6c387e28` | `UpscaleWireTests` (10) |
| F3 clip-job following (start / poll / recover / pause / resume / cancel) | fixed | `c7899440` | `UpscaleStoreTests` (5), `UpscaleStorePollTests` (6) |
| F3 **Make Bigger…** in the Library menu + tile menu | fixed | `c79bdc77` | `LibraryMenuPlanUpscaleTests` (6) |
| F3 `/api/activity` wire + reconcile + `ActivityStore` + Also Running | fixed | `1f6d8681` | `ActivityReconcileTests` (16), `ActivityStoreTests` (6), `AlsoRunningTests` (9) |
| F3 whole-queue pause / resume | fixed | `b82697f8` | `QueueGateTests` (11) |

## Fixtures

Captured read-only from **hal9000**, mold 0.29.0 (`b015496e`), 2026-09-17, keyless:

- `video-upscale-jobs-hal9000.json` — `GET /api/video-upscale-jobs`, three durable
  rows (one completed, two failed).
- `activity-hal9000.json` — `GET /api/activity` on an IDLE machine.

The machine had nothing running, and this lane does not write to a live host, so
every activity row with work in it is built from the wire shape in
`crates/mold-core/src/types.rs:5030-5087` and the kinds `routes_activity.rs:180-396`
emits. Said so in `ActivityReconcileTests`' own header.

## Decisions worth the integrator's eye

- **A clip upscale is not a queue row and not in `/api/activity`.**
  `video_upscale.rs` drives its own engine cache instead of submitting scheduler
  work, so polling the job is the only way anybody learns where it got to. That is
  why `UpscaleStore` lives beside the queue and why its rows are a second source
  under Also Running.
- **What Also Running excludes** is written as "the pane is already drawing this id"
  rather than as a kind list, so it stays true for the ephemeral chain (which
  reports as an ordinary generation) and for a kind added after this build; plus
  `download`, which has its own surface.
- **Cancel on a reported row is deliberately not offered.** Of the kinds drawn, the
  scheduler-owned ones report `can_cancel: false` themselves
  (`routes_activity.rs:340`); the one that reports `true` is a durable `sequence`,
  whose cancel is the `/api/chain-jobs` family this app does not speak and which the
  plan puts out of scope. A button that quietly does nothing would be worse.
  Pinned by `aReportedRowOffersNoCancelItCannotPerform`.
- **No keyboard chord for the gate.** Desktop binds Space; this app cannot, because
  the Library owns a bare Space for Quick Look and the README warns about binding a
  chord twice. Nothing else is both free and conventional, so the item carries none
  (`theGateCarriesNoKeyboardChord` reads the source and pins it).
- **"Make Bigger…" acts immediately** rather than opening a sheet. The wave's file
  ownership puts a Library dialog out of reach; if the ellipsis should be earned, the
  follow-up is a small confirm in `LibraryPane` naming the upscaler and showing
  `video_upscale.disclosure`. The wording is the plan's.
- **No upscaler installed** is reported through `HostStore.report`, the app's one
  door for "this machine could not", with a sentence that names Models. A BUTTON to
  Models would need the same dialog as above.
- **An upscale poll that fails stops**, mirroring `LibraryView.vue:549-554`; the
  repair is `recover()`, which the Library runs on open and which finds the job
  again (pinned in `aMachineThatGoesAwayMidPollStopsBeingAskedAndSaysSo`).

## Cross-lane edits

New protocol requirements land in NEW `MoldBackend+…` files and new
`FakeBackend+….swift` extensions, as asked. What is left touches files other lanes
own — all additive, all one or two lines:

| file | edit |
| --- | --- |
| `Packages/.../MoldBackend.swift` | composition list gains `MoldUpscaleBackend, MoldActivityBackend, MoldQueueGateBackend` |
| `Packages/.../MoldHost.swift` | `ServerStatus.queuePaused: Bool?` (`/api/status.queue_paused`) |
| `Packages/.../LibraryMenuPlan.swift` | `LibraryAction.upscale`; `canUpscale` stored property + defaulted init parameter |
| `Packages/.../LibraryMenuPlan+Items.swift` | the `Make Bigger…` item, after Use These Settings |
| `Sources/Mold/Library/LibraryActions.swift` | `var upscales: UpscaleStore?` (optional, so no other construction site changes) |
| `Sources/Mold/Library/LibraryActions+Menu.swift` | `case .upscale: upscale(targets)` |
| `Sources/Mold/Library/LibraryMenu.swift` | `canUpscale: actions.canUpscale(targets),` in the plan |
| `Sources/Mold/Library/LibraryPane+Menu.swift` | `canUpscale: actions.canUpscale(entries),` |
| `Sources/Mold/Library/LibraryPane.swift` | `@Environment(UpscaleStore.self)`, `upscales:` in `actions`, `.task { await upscales.recover() }` |
| `Sources/Mold/Shell/LibrarySelection.swift` | `canUpscale` field, into the plan, into `==` |
| `Sources/Mold/Shell/QueueCommands.swift` | `QueueSelection.gate` (defaulted empty), `Item.pauseQueue`, its menu rows and its `perform` arm |
| `Sources/Mold/MoldApp.swift` | two `@State` stores, two `State(initialValue:)` lines, `activity.start()` beside `heartbeat.start()`, two `.environment(…)` |
| `Tests/MoldTests/FakeBackend.swift` | ONE line: `nonisolated(unsafe) var extras = FakeExtras()` |
| `Tests/MoldTests/LibrarySelectionTests.swift` | `canUpscale: false` in its one constructor |

`MoldApp.swift` is now 159 lines, over the 150-line file ADVISORY (it was at exactly
150). `make lint` stays green — that rule prints, it does not fail — and splitting
the composition root belongs to whoever owns that file, not to this lane.
`QueueStore` was pushed over the 600-line type advisory by the gate and was pulled
back under it by moving the gate into its own `QueueGateControl`.

## Nothing deferred, nothing judged wrong

Every item in the plan's F3 section landed. No finding in the review reports was
judged incorrect, because F3 is a parity section rather than a findings list.

## Gates run

- `make lint` — green (advisories: `MoldApp.swift` 159 lines; `HTTPBackend` and
  `LibraryStore` type sizes, both pre-existing).
- `cd Packages/MoldClient && swift test` — 669 tests, 31 suites, passed.
- App bundle `xcodebuild … test` under the shared lock — 595 tests, 89 suites, passed.

## UAT owed (a real machine, not a fake)

1. **Upscale a still.** Library → a picture on a machine advertising
   `video_upscale.gallery_image` → **Make Bigger…**. A bigger PNG appears in that
   machine's gallery within a refresh, and no job row appears anywhere.
2. **Upscale a clip.** Same on an `.mp4`. A row appears under **Also Running** in
   the Queue pane counting frames (`Upscaling frame 12 of 124`), Pause holds it,
   Resume picks it up, Cancel stops it, and the finished clip lands in the gallery.
3. **Recovery.** Start a clip upscale, quit Mold, reopen it, open the Library: the
   row is back under Also Running still counting.
4. **No upscaler.** On a machine with none installed, **Make Bigger…** reports
   "… there is no upscaler on it yet. Install one from Models." and sends nothing.
5. **Absent, not disabled.** On a machine that does not advertise `video_upscale`,
   there is no **Make Bigger…** item at all — not a greyed one.
6. **Pause the queue.** Queue ▸ Pause Queue. The pane says "‹machine› is not
   starting anything new — its queue is paused." Queue a render: it waits. Resume:
   it starts. Pause from the web UI instead and watch the pane follow.
7. **Also Running from the machine itself.** Start a render whose model needs
   preparing, or a prompt rewrite, and confirm the row appears with the machine's
   own phase sentence and does NOT duplicate a queue row.
