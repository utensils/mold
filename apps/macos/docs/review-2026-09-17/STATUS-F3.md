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

## Adversarial review round (`REVIEW-F3.md`) — all ten addressed

| id | status | commit |
| --- | --- | --- |
| #1 HIGH a clip upscale IS in `/api/activity`, drawing two rows | fixed | `5a3c6a57` |
| #2 HIGH "no upscaler" banner on a machine that has one | fixed | `884239c2` |
| #3 MED-HIGH a failed Pause stops following a live job | fixed | `6ec287a0` |
| #6 MED `transition` did not bump the epoch | fixed | `6ec287a0` |
| #5 MED no host-side duplicate check; recovery only from the Library | fixed | `44dc3254` |
| #4 MED a still upscale had no feedback; `isWorking` never called | fixed | `cb0dbb19` |
| #10 LOW completion re-read the whole fleet | fixed | `cb0dbb19` |
| dead code (5 symbols) | deleted or wired | `44dc3254`, `cb0dbb19`, `afde5ff0` |
| #8 LOW the paused sentence was a selectable row | fixed | `080f42aa` |
| #9 LOW a cached gate outlived a dropped frame | fixed | `080f42aa` |
| #7 MED cross-lane conflicts with the landed mesh lane | for the integrator, below |

**#1 was the lane's central premise and it was wrong.** `upscale_frame` routes
every frame through `schedule_standalone_upscale` on any host with a v2 scheduler
or a GPU worker (`video_upscale.rs:1271-1281`) — every real one — and that mints a
fresh uuid per frame. The comments in `UpscaleStore.swift`, `VideoUpscale.swift`,
`AlsoRunning.swift` and the bullet that used to stand here all said otherwise and
are corrected. What is true: a clip upscale is not a QUEUE row, and what
`/api/activity` reports about it is one frame with no idea which print it belongs
to — which is why the durable job is still followed, and why the machine's own
upscale row gives way where this app is following one.

**#7 is not this lane's to resolve.** The mesh lane landed first at `e806343d`;
six shared files conflict, every one by insertion. The review's §7 lists the merged
form of each line, including the memberwise-init parameter order for
`LibraryMenuPlan` (`… exportFormats:, meshExports:, canReuse:, canUpscale:,
trashCount:`) and the one-line merge for `LibrarySelectionTests`. Menu order after
the merge reads correctly and needs no negotiation.

## Decisions worth the integrator's eye

- **The machine's own upscale row gives way to the job row, but only where there
  IS one.** A still upscale and a job started from another client are the same
  `standalone_upscale` work, and there the machine's row is the only feedback —
  so the suppression is per machine and per live job, never per kind.
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
- **"Make Bigger…" acts immediately** rather than opening a sheet, and the review
  confirmed desktop surfaces none of `disclosure` / `preserves_primary_audio` /
  `supports_hdr` / `supports_vfr` either, so that is not a regression against it.
  What it now has instead: an Also Running row from the moment it starts, the
  machine's own sentence on a failure, and a completion line naming the file. The
  menu item is ABSENT while this app is already making that print bigger.
  Desktop's model PICKER is closed too: with more than one upscaler installed
  on that machine, **Make Bigger** is a submenu naming each, the default first
  and marked as such, and the chosen one is what `start` sends. One installed or
  none read is the plain **Make Bigger…**, which sends no model name and lets the
  machine choose. The list is CACHE-ONLY — a right-click puts nothing on the wire
  — and `recover()` warms that cache once per machine, since nothing else on the
  Library path reads `/api/models`.
- **There is no local "is an upscaler installed" check.** Nothing reads
  `/api/models` on the Library path, so that cache is empty there; the ported
  policy's last fallback is the manifest name and the HOST is the authority.
- **An upscale poll that fails stops**, mirroring `LibraryView.vue:549-554`; the
  repair is `recover()`, which BOTH panes run on open and which finds the job again
  (pinned in `aMachineThatGoesAwayMidPollStopsBeingAskedAndSaysSo`). A failed
  TRANSITION is different and does restart it: the transition failed, the job did
  not.
- **A `queue_resumed` missed inside a stream gap** is handled by clearing the
  cached gate on a resync, which hands the question back to
  `/api/status.queue_paused`. A gap with no resync marker would still hold a stale
  cached value until the next frame; that is the residual, and it is bounded by the
  event stream's own reconciliation.

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
| `Sources/Mold/Library/LibraryStore.swift` | `apply(_:for:)` loses `private` so `LibraryStore+OneHost.swift` can re-read ONE machine |
| `Sources/Mold/Shell/LibrarySelection.swift` | `canUpscale` and `upscalers` fields, into the plan, into `==` |
| `Sources/Mold/Shell/QueueCommands.swift` | `QueueSelection.gate` (defaulted empty), `Item.pauseQueue`, its menu rows and its `perform` arm |
| `Sources/Mold/MoldApp.swift` | two `@State` stores, two `State(initialValue:)` lines, `activity.start()` beside `heartbeat.start()`, two `.environment(…)` |
| `Tests/MoldTests/FakeBackend.swift` | ONE line: `nonisolated(unsafe) var extras = FakeExtras()` |
| `Tests/MoldTests/LibrarySelectionTests.swift` | `canUpscale: false` in its one constructor |

`MoldApp.swift` is now 159 lines, over the 150-line file ADVISORY (it was at exactly
150). `make lint` stays green — that rule prints, it does not fail — and splitting
the composition root belongs to whoever owns that file, not to this lane.
`QueueStore` was pushed over the 600-line type advisory by the gate and was pulled
back under it by moving the gate into its own `QueueGateControl`; `UpscaleStore`
was split again at `+Reading` for the same reason.

## Nothing deferred, nothing judged wrong

Every item in the plan's F3 section landed, and every one of the ten adversarial
findings was verified against the source and acted on. None was judged incorrect;
#1 in particular was checked against `video_upscale.rs` and the reviewer is right.

## Gates run

- `make lint` — green (advisories: `MoldApp.swift` 159 lines; `HTTPBackend` and
  `LibraryStore` type sizes, both pre-existing).
- `cd Packages/MoldClient && swift test` — 672 tests, 31 suites, passed.
- App bundle `xcodebuild … test` under the shared lock — 612 tests, 89 suites, passed.

## UAT owed (a real machine, not a fake)

1. **Upscale a still.** Library → a picture on a machine advertising
   `video_upscale.gallery_image` → **Make Bigger…**. A row says *Making a bigger
   copy* straight away, the menu item is gone while it runs, and it settles to
   *Complete — <filename>* with the bigger picture in that machine's gallery.
   **Do this from a cold launch, straight to the Library**, which is the sequence
   that used to answer "there is no upscaler on it yet".
2. **Upscale a clip.** Same on an `.mp4`. Exactly ONE row appears under **Also
   Running** in the Queue pane counting frames (`Upscaling frame 12 of 124`) --
   not two, and its identity does not churn. Pause holds it, Resume picks it up,
   Cancel stops it, and the finished clip lands in the gallery.
3. **Recovery, and no duplicate.** Start a clip upscale, quit Mold, reopen it and
   go STRAIGHT TO THE QUEUE pane: the row is back, still counting. Then press
   **Make Bigger…** on the same clip from the Library: it must adopt that job, not
   start a second one (watch `/api/video-upscale-jobs` on the host).
3b. **Choosing the upscaler.** On a machine with two installed, **Make Bigger**
   is a submenu listing both with the default marked; picking the other one sends
   it (check the job's `model`). With one, it is the plain item.
4. **No upscaler.** On a machine with none installed, the request goes out and
   the MACHINE refuses; the banner carries its sentence, which names the model.
5. **Absent, not disabled.** On a machine that does not advertise `video_upscale`,
   there is no **Make Bigger…** item at all — not a greyed one.
6. **Pause the queue.** Queue ▸ Pause Queue. The pane says "‹machine› is not
   starting anything new — its queue is paused." Queue a render: it waits. Resume:
   it starts. Pause from the web UI instead and watch the pane follow.
7. **Also Running from the machine itself.** Start a render whose model needs
   preparing, or a prompt rewrite, and confirm the row appears with the machine's
   own phase sentence and does NOT duplicate a queue row.
