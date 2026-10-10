---
paths:
  - "apps/macos/**"
---

# Native Mac interaction contracts

Library and Generate media viewers fit both viewport axes by default and reserve
space for their toolbar/transport controls. Actual Size means one media pixel per
display backing pixel, with scrolling when the picture exceeds the viewport.
Fit/Actual Size changes preserve the video player and playback position. AVKit
hover transport must not dim the picture: use a control-free player with separate
accessible playback controls. Alpha checkerboards follow the rendered picture in
both image sizing modes. Test portrait/landscape fit, Retina sizing, scrolling,
resize and the player control policy.

Generate submission feedback is separate from the canvas following an earlier
job. Start feedback synchronously on the press, acknowledge only actual backend
acceptance, surface preparation/refusal failures, and fence late replies by the
latest press ID. Persistent acceptance text describes the acceptance event; it
must not claim a job is still running after that job completes.

Prompt editor height remembers the preferred size but clamps the rendered height
to available space. Keep the Generate action row outside scrolling controls.
Inspector headings have one full-width disclosure action with accessory buttons
as siblings; controls must fit both narrow and wide inspector widths. Test
rendered layout limits, including full UInt64 seed values.

Job Details derives actions and status from the current queue entry, preserving
host/instance/job identity fences for async responses. Hide stale running progress
when a job leaves that state. Explain control actions in plain English with
`.help`; do not claim a setting's write store is its effective value's origin.
See `apps/macos/docs/tooltip-audit.md` for the reviewed inventory and verification.

Native Mac draft persistence saves scalar settings and local authoring inputs as
one consistent document: make the private input snapshot durable before publishing
the descriptor that references it, and preserve the previous document on failure.
Preserve active and parked inputs, original source pixels and boundary-frame
meaning across relaunch; reuse unchanged input snapshots across scalar edits.
Never persist upload leases or scoped media permissions as reusable authority.
Unresolved retained references require fresh verification of their original
instance, archive, output and reference order; local snapshots cannot bypass it
or overwrite explicitly replaced attachments. Surface save failures. Missing or
corrupt snapshots block Generate and remain preserved until the user explicitly
chooses **Use current inputs** after reattaching any needed files. This snapshot
behavior is Mac-only; preserve the existing scalar-only iOS persistence path.

Library New badges compare filenames with the previous session visit, using the
whole active pool and a stable per-view snapshot. First visit establishes a
baseline; selected viewer media immediately calls `Visit.markViewed`, preserving
unopened badges within the visit and the previous next-visit clearing behavior.
`LibraryUnreadLedger` persists device-local icon counts across launches. Count the
merged visible active inventory, excluding hidden collections and Trash; retain
offline read state. Opening Library uses its existing whole-pool seen behavior
to clear the icon count. iOS serializes badge writes and flushes background
refresh; macOS updates the Dock through one preference-aware controller. The
existing Dock badge toggle suppresses presentation and immediately clears it;
re-enabling restores the current ledger count. Never also follow LandedPrints
for the Dock, since app activation would overwrite the persisted count.
Badge rows must fit machine
labels beside playback without overlap, including narrow tiles.

**Visible reuse media.** Restore supported legacy retained roles into ordinary authoring wells before allowing submission, including all endpoint/keyframe and reference-image inputs. Preserve list order, exact frame indices, manual canvas, continuation overlap and explicit reference strength. Retire each materialized or superseded legacy role, including the mask paired with a replaced source; removal must never revive an archived fallback. Fence every asynchronous operation by reuse identity, immutable origin route/instance, per-role monotonic attachment revisions and component lifetime. Scalar edits remain live. Keep restoration failures blocking until deliberate recovery/discard; descriptor-only typed references retain their exact-set authority.

Library machine filters project merged tiles to only the selected hosts before
presenting or acting on them. Never expand those rows back to fleet copies.
Trash remains accessible when empty, ignores hidden collections, and resets
prior browsing filters on entry while preserving the selected hosts. Put Back,
Delete Immediately and Empty Trash act only on the displayed host scope; Empty
Trash uses the trash-only endpoint and confirms its destination machines.

Sidebar counts use the current machine scope, including collection drag filing.
Collection inventory absence requires a successful read; failed/offline reads say
unavailable. Shared CollectionVisibilityLedger retains hide/show intent through
partial/offline writes, fences superseded edits and routes, and repairs mixed
hidden replicas; protect logical hidden media before pruning copies for display.
A fresh read must confirm a visibility write before retiring its pending intent.

Library Sync is opt-in for the current app session: immediate run, five-minute
non-overlapping repeats, visible next-run/completion status and Stop. Successful
runs never require a completion sheet. Repeat issue acknowledgment is bound to
origin route/instance, output version/recipe and exact error, never authentication
or connectivity failures; preserve retries and inspectable details. Sync must not
recreate destination trash or a recorded copy intentionally removed from the
local listing. Explicit Save remains deliberate recovery. Copy hidden collection
attributes for existing and empty replicas as well as newly filed outputs.

The Sync report acknowledgment checkbox binds directly to eligible issue keys and the persisted session ledger; reopening, Reset, and new keys must update its visible state without a separate sheet-local toggle state.

Sync Details dismisses both its request and native presentation on Done/Escape;
observe deferred sheet requests in the presenter body. Keep issue acknowledgment
controls out of clean reports. Quit-drain panels use constrained content margins
and native button padding, sized to fit their wrapped message. Verify actual
presented sheets and rendered quit-panel geometry, not only detached views.

Queue Cancel remains visibly available for every currently actionable row, independently of batch metadata, retryability and transfer destinations. Failure Details preserves optional `error_detail` separately from the plain-English row explanation, falling back to the older machine’s reason. Copy includes machine and job identity. Cancel rechecks current state; Held actions always use the held-only endpoint and must never widen their intent after a state change. Current servers enforce this guard atomically; older servers may ignore the query, so do not promise that safeguard on older hosts.

Queue action offers use QueueStore's current-host/current-row authority across
rows, child rows, Job Details and focused menus. Offline, removed, cancelling,
terminal and in-flight rows offer no mutations. Held Retry requires a durable
batch identity and known server instance, independently of full metadata; missing
models use Download and Retry. Preserve exact host/job reservations through the
post-mutation refresh, and refuse stale state, retry identity or replaced host
routes. Group dispatch reserves each eligible child separately and preserves
Held-only intent. Reorder requires the advertised capability and current queued
membership. Read-only Job Details and failure diagnostics remain accessible.

Displayed ephemeral generation chains use the chain endpoints, their own
`can_cancel` authority and paused Resume, independently of singleton cooperative
cancellation. Re-read status and activity before dispatch and refresh refusals
before releasing the exact host/kind/job reservation. Durable authored sequences
remain unsupported here. Existing framewise upscale controls require a connected
host, known applicable state and current rendered job identity; reserve the exact
print/job key until the authoritative transition response reconciles its state.

Continuous Library rows preserve actual metadata aspect ratios with 2-point seams
and square corners. Missing/invalid dimensions use a square placeholder; decode
completion never changes geometry. Incomplete final rows stay left aligned at
target height. Valid extreme ratios remain uncropped. Keep ordered row geometry
shared (studio/lib/justifiedLayout.ts and MoldClient.JustifiedLayout), lazy/windowed
rendering, stable item identities, and a visible item anchor through resize/zoom.
Native macOS vertical arrows follow adjacent row centers, never a guessed column
count. Viewer return restores the covered viewport; native previews explicitly
inject thumbnail loaders. Native iOS omits the visual host-name thumbnail badge;
Info and host sorting/filtering retain their authority. Unseen semantics are unchanged.
