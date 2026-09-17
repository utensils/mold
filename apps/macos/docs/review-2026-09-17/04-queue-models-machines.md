# Review 04 — Queue, Models, Machines (`apps/macos`)

Read-only review of `Sources/Mold/{Queue,Models,Machines}`, their `Packages/MoldClient`
counterparts, and `Tests/MoldTests`, against `desktop/`, `studio/`, `web/` and
`crates/mold-server`. Nothing was built or run.

Confidence is stated per finding. Everything marked HIGH confidence was verified by
reading both sides.

---

## HIGH

### H1 — The pairing code is always shown as "Expired", and every reopen burns a new session

- **kind**: bug (wire unit mismatch) · **confidence**: HIGH (both sides read; tests confirm the wrong unit is baked in)
- `apps/macos/Sources/Mold/Machines/PairingSheet.swift:48-51`
- Server: `crates/mold-server/src/auth.rs:216` (`expires_at = unix_timestamp().saturating_add(PAIRING_TOKEN_TTL_SECS)`), `auth.rs:633-638` (`unix_timestamp()` → `.as_secs()`), wire field `crates/mold-server/src/routes.rs:9502`
- Counterpart: `studio/components/MobilePairingCard.vue:24,36-37` — `session.expires_at - Math.floor(Date.now() / 1000)`, i.e. **seconds**.

`PairingSession.expiresAt` is unix **seconds**, but `Countdown.resolve` divides it by
1000 before comparing against `now.timeIntervalSince1970`:

```swift
let remaining = TimeInterval(expiresAt) / 1_000 - now.timeIntervalSince1970
```

For a real value (~1.79e9) that is `1.79e6 - 1.79e9`, hugely negative, so `Countdown`
is **always `.expired`**. Two consequences, both user-visible on the first real use of
Machines ▸ Pair a Phone…: the QR is drawn at `opacity(0.3)` under the words "Expired"
with a "New Code" button beside a code that is in fact valid
(`PairingSheet.swift:112-129`); and `needsFreshCode` (`:34-39`) returns `true` for every
reopen, so the documented "reopening inside the two-minute window shows the code a
phone may already be scanning" behaviour is inverted — reopening kills the live token
(`auth.rs:251-257`, tokens are single-use and `MAX_PAIRING_SESSIONS` evicts oldest).

This is also a **test-gap**: `Tests/MoldTests/PairingTests+Sheet.swift:33-34,44,46` and
`PairingTests.swift:144` all use millisecond fixtures (`1_700_000_030_000`,
`4_102_444_800_000`), so code and test agree with each other and both disagree with the
server. The `/1000` is almost certainly copied from `PairedClient.lastUsedAtMs`, which
genuinely *is* milliseconds (`PairingSection.swift:84` is correct).

**Fix**: drop the `/ 1_000` in `Countdown.resolve` and re-base the fixtures on seconds
(`1_700_000_030`), with one test asserting a value the server would actually emit.

---

### H2 — A dropped-and-reconnected event stream never reconciles, and nothing polls

- **kind**: bug / parity-gap · **confidence**: HIGH
- `apps/macos/Sources/Mold/Support/HostStore+Events.swift:62-102`, `HostStore+Reachability.swift:23-32`
- Counterpart: `desktop/src/stores/hosts.ts:49` (`POLL_INTERVAL_MS = 10_000`, loop `:1573-1586`), `web/src/composables/useHostRouting.ts:241` (8 s), `web/src/components/machines/hostClient.ts:406` (5 s)
- Server: `crates/mold-server/src/routes.rs:11751-11755` — a gap is repaired "from `GET /api/queue`, `GET /api/devices`, and `GET /api/gallery`"

`watch(_:)` reconnects with backoff after the stream errors, but on reconnect the
opening `authority` frame is swallowed by `deliver` when the instance id is unchanged:

```swift
guard let known, known != instanceID else { return }   // :97
listeners.forEach { $0(host, .resyncRequired) }
```

So the *only* thing that fires a resync is a **changed** instance id. `instance_id` is
persisted per data-dir-and-port (`crates/mold-server/src/instance.rs:20-28`) and
therefore **survives a server restart**, which is exactly the scenario the task names.
Combined with:

- `refreshAll()` having only four callers, all manual — `RootView.swift:34` (once at
  launch), `Sidebar.swift:65` (pull to refresh), `MachinesSettings.swift:82` ("Check
  All"), plus host edits. There is **no timer anywhere in `Sources/`** (grep for
  `Timer`/`scheduledTimer`/`NSWorkspace.didWakeNotification` finds only `PairingSheet`'s
  `TimelineView`);
- `QueueStore.wantsPoll(_:)` (`QueueStore.swift:86-88`) documenting a poll fallback for a
  host with no event route — and having **no production caller at all** (only
  `Tests/MoldTests/QueueStoreLiveTests.swift:111`);

the concrete failures are: (a) Mac sleeps with three jobs queued, wakes after they all
finished — the stream reconnects, no frame is missed *after* reconnect, so the Queue
pane shows three rows that no longer exist until the user presses ⌘R; (b) `mold serve`
restarts, its restart sweep parks every row as `paused` — same stale view, same
instance id, no resync; (c) a machine that was `.down` at launch is never re-probed, so
`wantsEvents` never returns true for it and it stays dead for the whole session; (d) a
machine on an older build with no `events.available` is read exactly once, ever.

**Fix**: track "has this watcher connected before" and deliver `.resyncRequired` on every
reconnect (not only on an identity change); and add a low-frequency reachability tick
(desktop's 10 s, or at minimum a `NSWorkspace.didWakeNotification` + `didBecomeActive`
hook) so `refreshAll` runs without the user asking. `wantsPoll` should then be wired up
or deleted.

---

## MED

### M2 — `catalog_ready` creates a permanent phantom download row

- **kind**: bug · **confidence**: HIGH
- `apps/macos/Sources/Mold/Models/DownloadStore+Stream.swift:30-56`, `Packages/MoldClient/Sources/MoldClient/Downloads.swift:48-50`
- Server: `crates/mold-core/src/types.rs:13118-13125` (`CatalogReady { id, ok }` where **`id` is the catalog id**, e.g. `hf:owner/repo`), emitted from `crates/mold-server/src/downloads.rs:521-523` *after* the group's last terminal arm
- Counterpart: `desktop/src/lib/downloads.ts:127-128` — `case "catalog_ready": return state;` (explicitly ignored)

`apply` recognises only `snapshot` and `isTerminal ∈ {job_done, job_failed,
job_cancelled}`. A `catalog_ready` frame has a non-nil `id`, is not terminal, so it
falls into the progress branch and inserts

```swift
Progress(model: event.model ?? "")   // model: "", fraction nil, bytes nil
```

keyed by the **catalog id**, into `active[host]` — after the real job row was already
removed by `job_done`. Nothing ever removes it: there is no further frame for that id.
Result after installing any `cv:`/`hf:` model: the toolbar Downloads button shows work
in flight forever, `DownloadsPopover` (`DownloadsPopover.swift:41-60`) draws a nameless
row stuck on "Starting…" with a Cancel button that would `DELETE
/api/downloads/hf%3Aowner%2Frepo` → 404 `unknown download id`, and
`DownloadStore.reconcile()` (`DownloadStore.swift:104-113`) keeps that host's SSE
connection open for the rest of the launch because `active` is never empty.

**Fix**: ignore `catalog_ready` (and `dequeued`, which desktop treats as "remove from
queued", not "create") explicitly, and make the progress branch only *update* a row it
already knows, as `desktop/src/lib/downloads.ts:96-97` does.

---

### M3 — Pause/Resume on a queue row is not capability-gated, while the same action in the Queue menu is

- **kind**: bug (internal inconsistency) / parity-gap · **confidence**: HIGH
- `apps/macos/Sources/Mold/Queue/QueueRow.swift:60-67` (ungated) vs `QueuePane+Commands.swift:28-31` (gated on `canPauseOneJob`)
- Also `QueueBatchRow.swift:56-63` — the batch-wide Pause/Resume buttons are ungated too
- Server flag: `crates/mold-core/src/types.rs:11464-11490` (`QueueCapabilities.can_pause_job`), set `crates/mold-server/src/routes.rs:8122-8132`
- Counterpart: `desktop/src/composables/useQueueCommands.ts:349-363` — returns `[]`, i.e. the item is **absent, not disabled**, with a comment saying so; `web/src/composables/useQueueInspection.ts:200-204` and re-checked at action time `:341-348`

`Capabilities.canPauseOneJob` exists and is read in exactly one place. The row's own
glyph buttons — which is where a user actually presses Pause — ignore it. On a host
that predates per-job pause the button is offered and the request fails. Every other
queue control in this app follows the "absent, not disabled" rule correctly
(`canReorderQueue` at `QueuePane.swift:65`, `canCancelAllQueued` at
`QueuePane+Toolbar.swift:61`), so this is a miss rather than a policy.

**Fix**: thread `canPauseJob` into `QueueRow`/`QueueBatchRow` the way `isReorderable`
already is, and add a test alongside the existing `QueuePaneTests` gating cases.

### M4 — `cooperative_cancellation` is decoded and never read

- **kind**: parity-gap · **confidence**: HIGH that the flag is unused and that web gates on it; MEDIUM on the user-visible consequence (I did not trace what a non-cooperative runtime does with `DELETE /api/queue/:id` on a running row)
- `apps/macos/Packages/MoldClient/Sources/MoldClient/Capabilities.swift:52` declares `cooperativeCancellation`; grep over `Sources/`, `Tests/` and `Packages/` finds **no reader**
- `QueueRow.swift:68-71` offers ✕ on every `state.isLive` row, `.running` included
- Counterpart: `web/src/composables/useQueueInspection.ts:153-155` (`canCancelRunning`) re-checked at `:299-305`; `desktop/src/stores/jobs.ts:257`. Web also reads a per-row `row.can_cancel` (`useQueueInspection.ts:180-184`) that `QueueEntry` does not model.

**Fix**: gate the running-row ✕ (and the group cancel) on `cooperativeCancellation`, or
delete the field and write down why this app does not need it.

### M5 — Two concurrent `hydrate(on:)` calls fire the same failure notification twice

- **kind**: bug (race) · **confidence**: HIGH on the race; MEDIUM on how often it lands
- `apps/macos/Sources/Mold/Queue/QueueStore+Batches.swift:18-46` and `reportOutcomes` `:53-65`

`hydrate` snapshots `children[host]` into a local `merged` *before* its first `await`,
and writes it back only at the end:

```swift
var merged = (children[host] ?? [:]).filter { ids.contains($0.key) }
for chunk in … { let listing = try await client.batchStatuses(…)   // suspension
                 let before = merged[status.id] ?? []
                 … reportOutcomes(before:after:on:) }
children[host] = merged
```

There is no in-flight guard, and `refresh(on:)` has several concurrent callers: the SSE
coalescer (`QueueStore+Live.swift:70-74`), `QueuePane.load()` → `queue.refresh()` for
every host, `MachinesPane`'s `.task(id:)` (`MachinesPane.swift:52`), and `refresh()`
(`MachinesPane.swift:120`). Two overlapping hydrates both compute `before` from the same
snapshot, both see `held → failed`, and both call `onOutcome` — so
`MoldNotifications.noteFailed` (`MoldNotifications.swift:123-127`, deliberately *not*
coalesced) posts two "Failed on plato" banners for one job. Pressing ⌘R while a
`job_state_committed` frame is in flight is enough.

**Fix**: serialize `hydrate` per host (one task slot, same shape as `coalescers`), or
re-read `children[host]` as `before` immediately before comparing rather than from the
pre-await snapshot.

### M6 — Held and paused rows re-shuffle on every refresh

- **kind**: bug (unstable ordering) · **confidence**: HIGH
- `apps/macos/Packages/MoldClient/Sources/MoldClient/QueueEntry.swift:76-81`
- Server: `crates/mold-server/src/job_registry.rs:57-64` — `assign_positions` gives a held row **the position of the next schedulable row**, so ties are normal, by design

```swift
var byID: [String: QueueEntry] = [:]
…
return byID.values.sorted { ($0.position ?? .max) < ($1.position ?? .max) }
```

Two things compound: the intermediate dictionary destroys the server's own array order
(which *is* dispatch order), and Swift's `sorted(by:)` is not stable. Three held rows
all carrying position 4 therefore come back in a different order on each poll, and
because `List`/`ForEach` identify by `QueueEntry.id`, SwiftUI animates them swapping
places. `QueueGroup.build` (`QueueGroup.swift:31-40`) keeps a group where its first row
sat, so whole batches jitter too when their children are held.

**Fix**: preserve the server's order — iterate `entries` then append any
`liveOnlyEntries` not already present — or at minimum break ties on `id`.

### M7 — Scheduler-owned work that has no queue row is invisible

- **kind**: parity-gap · **confidence**: MEDIUM (verified the route exists and that no Swift file calls it; did not enumerate every work kind that lacks a queue row)
- No reference to `/api/activity` anywhere under `apps/macos` (grep)
- Server: `crates/mold-server/src/routes_activity.rs:173-181`, `crates/mold-core/src/types.rs:5031-5066` — `ActiveWorkItem.phase` includes `preparing`, `loading`, `downloading`, none of which exist in `JobLifecycle`; `kind` includes `sequence` and `download`
- Counterpart: desktop's "Now Developing" reconciles from it; doc comment says it is "used to reconcile Now Developing after a client restart or reconnect"

The Queue pane is built purely from `GET /api/queue` + `POST
/api/generation-batches/status`. Work the scheduler owns that never gets a
`generation_queue` row — a video upscale, a chain stage, a preparation phase — shows
nothing. The README's "no chain jobs" omission covers part of this, but `video_upscale`
is a capability this app *does* advertise reading (`Capabilities+Reading.swift:303`), so
a clip upscale running on a machine is work the Queue pane cannot see.

### M8 — The first notification of a session is silently dropped

- **kind**: bug · **confidence**: MEDIUM-HIGH (behaviour of `UNUserNotificationCenter` is well established; not empirically confirmed here)
- `apps/macos/Sources/Mold/Support/MoldNotifications.swift:129-143`

```swift
private func post(…) {
    requestAuthorizationIfNeeded()          // async, fire-and-forget
    …
    center.add(request, withCompletionHandler: nil)   // runs immediately
}
```

Authorization is requested on first need — correct, and well-reasoned in the doc
comment — but `add` is called in the same turn, before the user has answered the
prompt. The very first finished-render notification is therefore posted while
authorization is still `.notDetermined` and is dropped. The user sees the permission
alert and no notification, which reads as the toggle not working.

**Fix**: make `requestAuthorizationIfNeeded` `async` (or hold a continuation) and await
it before the first `add`; subsequent posts can skip the wait.

---

## LOW

### L1 — The transfer refusal quotes a size limit that does not exist

- **kind**: quality (wrong user-facing fact) · **confidence**: HIGH
- `apps/macos/Packages/MoldClient/Sources/MoldClient/TransferPlan.swift:103-105` — "about 48 MB", pinned by `Tests/MoldClientTests/TransferPlanTests.swift:104`
- Server: `crates/mold-server/src/lib.rs:178` `MAX_REQUEST_BODY_BYTES = 64 * 1024 * 1024`, applied at `lib.rs:1347`

`HTTPBackend+Transfer.swift:19-22` correctly documents 64 MiB; the sentence the user
actually reads says 48 MB. "about 48 MB" matches no constant in the repo (the nearby
real ones are 64 MiB body, 32 MiB journal `queue_journal.rs:52-53`, 32 MiB inline H3
reference `minimax_h3.rs:786`).

### L2 — `awaitSettlement` becomes a tight main-actor spin if its task is cancelled

- **kind**: bug · **confidence**: HIGH on the mechanism, LOW on reachability
- `apps/macos/Sources/Mold/Models/DownloadStore+Install.swift:74-79`

```swift
while isBusy(model, on: host) { try? await Task.sleep(for: .milliseconds(100)) }
```

Once cancelled, `Task.sleep` throws immediately and `try?` swallows it, so the loop
spins at full speed on `@MainActor` until the download settles. The only caller
(`QueuePane+Transfer.swift:16-21`) launches an unstructured `Task`, so cancellation is
unlikely today — but there is also no timeout, so a host whose download stream is alive
but whose job never reaches a terminal frame parks this forever.

**Fix**: `guard !Task.isCancelled else { return false }` inside the loop, plus a bounded
wait.

### L3 — Blanket `NSAllowsArbitraryLoads`, and no self-signed-TLS path

- **kind**: security / gap · **confidence**: HIGH on the configuration; the trade-off is a judgement call
- `apps/macos/Sources/Mold/Resources/Info.plist:31-35`
- `Packages/MoldClient/Sources/MoldClient/HTTPBackend+Transport.swift:140-142` — `X-Api-Key` header; `HTTPBackend.swift:11` uses `URLSession.shared` with no `URLSessionDelegate`

The comment justifies the exception well (LAN, Tailscale, loopback, no trustable certs)
and I agree a narrower key like `NSAllowsLocalNetworking` would not cover
Tailscale's 100.64/10 range. Two things still follow from the blanket form: the
operator API key travels in cleartext to any `http://` host on any network with no
warning anywhere in the UI; and because arbitrary loads relaxes cleartext but **not**
certificate validation, an `https://` host with a self-signed or private-CA certificate
is unreachable and there is no delegate to opt into one. Worth at least a sentence in
the host editor when a key is set on a plain-`http` non-loopback address.

### L4 — The Dock badge is driven from a view modifier

- **kind**: quality · **confidence**: MEDIUM (I did not confirm this app's last-window-closed behaviour)
- `apps/macos/Sources/Mold/MoldApp.swift:88-90`

`NSApp.dockTile.badgeLabel` is set from `.onChange(of:)` on `RootView`. `LandedPrints`
keeps counting while the app is in the background whatever the window is doing, but if
the main window is ever closed while the process lives, the modifier is gone and the
badge stops tracking the count it is supposed to display. The store is deliberately
AppKit-free, which is good; the observer belongs on the app delegate rather than on a
view.

---

## Verified and correct (things I looked hard at and found right)

- **Reorder index derivation.** `QueueOrder` (`QueueOrder.swift:1-66`) is the best code
  in this area: it names all three competing indices, removes the moving rows from the
  candidate set before computing the target (mirroring the server's remove-then-reinsert
  at `crates/mold-db/src/generation_queue.rs:1815-1863`), clamps rather than trusting the
  caller, and issues **ascending** calls for a batch so children land contiguous. This is
  the trap `desktop/src/composables/useQueueCommands.ts:299-303` also documents, and the
  Swift version handles it more thoroughly than either JS client.
- **Transfer cannot lose a job.** `TransferStore`/`TransferPlan` run complete *last*,
  treat a failed `complete` as success-with-caveat rather than a retry
  (`TransferStore+Steps.swift:93-97`, `TransferPlan.swift:123-129`), and recover an
  ambiguous admit with a second lookup instead of a second admit
  (`:112-121`). `QueueTransferID.derive` is byte-compatible with
  `studio/api/queueTransfer.ts:27-48`, so a re-run finds the prior attempt. The
  destination-instance fence is sent as the header the server actually checks
  (`HTTPBackend+Transfer.swift:74`, `crates/mold-server/src/routes.rs:2982-2993`), and
  keeping the exported bytes opaque is the right call given `GenerateRequest`'s
  hand-written encoder. The "job deleted on A but never created on B" failure the brief
  worried about is structurally impossible here.
- **Held-row semantics.** `QueueHold.resolve` reads `error_code`/`retryable` only from
  the batch child and only while held, matching
  `crates/mold-server/src/routes.rs:2952-2957` exactly; `BatchChild.supersedes` orders on
  `revision` with a timestamp fallback only at 0/absent, which is precisely the retry
  back-transition the server documents at
  `crates/mold-db/src/generation_batches.rs:923-932`. Every held row getting the same ✕
  as every other row is the right UX call.
- **Capability absence rules.** `Capabilities+Reading.swift` is a genuinely better
  expression of the "absence means something different per field" contract than the
  scattered `=== true` checks in `desktop/` — and `DeviceControl.resolve`
  (`DeviceControl.swift:24-47`) gets the two-flag rule right, including the
  `startup_excluded` pre-check the shared `studio/lib/deviceLifecycle.ts:40-52` also has.
- **Credential handling.** Catalog tokens are written to the machine and never held
  locally (`CatalogStore+Credentials.swift`), which is stricter than desktop's
  `secrets.json` (`desktop/src-tauri/src/secrets.rs:1-29`); host API keys go to the
  Keychain and never to the plist (`Keychain.swift`, `HostPersistence.swift:6-8`).
  Telemetry is capped at one live stream and stopped on disappear
  (`MachineStore+Telemetry.swift:14-35`, `MachinesPane.swift:57`) — no shipped client
  does that.
