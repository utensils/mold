---
paths:
  - "apps/macos/**"
---

# Native Mac interaction contracts

Every streamed `PlayableMedia` player awaits `prepare(_:)` before publication;
it awaits AVFoundation native preroll without starting playback, with a 30-second
timeout and cancellation/item fences. Do not require a fixed buffered duration:
a paused custom-loaded audio/video item may validly buffer only half a second.
Apply `configure(_:)` before a resumed seek, then prepare at that position.
AVFoundation's predictive stall waiting does not support media bytes
supplied by a resource-loader delegate; the shared policy disables it and keeps
the default for local files. Deliver bounded chunks to the delegate on the main actor.

Native Xcode builds keep Swift package checkouts in their own
`build/DerivedData/SourcePackages` and disable the shared repository cache.
Do not clear developer-global SwiftPM caches to repair a build. Keep that local
artifact path aligned with Sparkle tool discovery in `make-appcast.sh`.

Native pairing defaults to Secure and offers explicit Regular direct HTTP(S) access.
The proxy always remains encrypted. Regular has its own persisted opt-in and listener;
stopping proxy access must not disable Regular, and disabling Regular must not remove
host identity or grants. Produce a Regular QR only after the listener reports its actual
bound direct origin. Mode changes require a new pairing or a distinct Regular credential;
never prefill or probe with the former Secure key. Preserve one physical machine record.

Active Library sync progress uses one status banner; there is no countdown.
Expose one Stop Sync control during transfers, including their finishing phase.

Boundary-frame fit controls remain available for last-only inputs. Fit active
endpoints once on the captured submission snapshot before admission, preserving
original authoring bytes, roles, batch IDs, seeds and synchronous queue reservation.
Ordinary sources keep their existing canvas-space mask fitting pipeline; boundary
frames never gain repaint masks. Test crop/pad pixels through the submitted batch.

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

Library New badges and unread counts share persisted client-local viewing state.
Individual viewer display marks that logical print read immediately. A Library
visit captures its displayed eligible projections and marks that captured set
read when navigation leaves Library. Keep New labels during the visit; exclude
Trash and unvisited hidden/filtered media, and preserve arrivals after departure. Preserve current
read status on upgrade and establish an initial baseline for newly connected
hosts. Later arrivals remain unread until viewed or acknowledged by a Library visit. Count merged
visible active inventory, excluding hidden collections and Trash; retain offline
history. iOS navigation and native icon badges follow that same ledger, with the
existing permission/preference controls. Do not let activation-only arrival
trackers overwrite the persistent count.
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

Library Sync runs once per press. The toolbar offers Sync All to every connected
machine and selection/viewer Copy to; Settings Library and the toolbar expose one
Keep in Sync toggle per captured destination route/installation. Legacy v1 true
migrates only to historical This Mac. One global run serializes transfers, coalesces
one sync follow-up per destination and preserves manual selections. Automatic runs
never raise connectivity alerts; Stop drops follow-ups and quit cancels/drains.
Verified copy components span destination buckets and collision-renamed copies;
filename/seed grouping is privacy evidence, never transfer or echo authority.
Destination Trash and intentionally removed verified copies prevent automatic
recreation; explicit Copy restores deliberately. Source removal never deletes backups.

Organization tracks versioned observed source, actually applied contributions and
local removal suppression per verified source. Remove only sync-owned unclaimed
memberships; preserve destination-only organization and differing scalar edits.
Ordinary local removals suppress unchanged re-additions until source remove/re-add or
explicit Copy, while hidden membership always reasserts. Legacy observation never
retroactively owns existing values or authorizes removals. Persist immutable mutation
intent with durable UUIDs before writes; verify resulting organization and include
cached organization-only copies in final checkpoint verification. Journal a newly imported output’s initial-organization obligation until verification; a bare interrupted receipt must not become a destination override. Explicit Copy discovers verified component hosts and current readable root organization; forwarded privacy changes must schedule root reconciliation even when its own cursor did not change. Union every verified raw lineage edge before canonicalization; never overwrite a shared endpoint in a neighbor dictionary. Keep provisional/unfinished receipt links separate as conservative hide-only privacy evidence, scoped to captured routes and installations. They must not grant output skip tokens, ordinary merge authority or organization ownership; preserve them through listed-facts upgrades until organization verification. Source-access caches and previews observe a separate privacy identity revision so receipt-only links revoke existing visible originals without waiting for row changes. Organization-only journal writes must not invalidate that identity clock. Partial/ambiguous
writes remain retryable. Keep scalar authority attached to the verified origin across
forwarding, never traversal order. A surviving forwarded copy must still back up to a new destination when its root is offline or trashed, using the verified origin organization snapshot; privacy-only handling applies when its destination copy already exists or the readable root is separately considered. Keep source-access row and privacy revisions as independent equatable fields, never an arithmetic sum. Hidden privacy checks merge live and Trash evidence
before host projection on grids, counts, unread and source/picker access.

Successful runs need no completion sheet. Repeat issue acknowledgment binds to origin
route/instance, output recipe/version and exact error, never authentication/connectivity
failures. Preserve retries and inspectable details. Sync hidden attributes for existing
and empty collections as well as newly filed outputs.

Sidebar destination tags wrap badges so native List selection receives SidebarRow,
not the ForEach destination ID. Verify rendered top-level navigation from Library.

Legacy missing-input notices participate in per-source/version acknowledgment,
including successful copies. The checkbox reads "Never show me this again for these sources."
Acknowledgment removes repeated source notices from the banner, preserving full Details.

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
before releasing the exact host/kind/job reservation. Durable authored videos use the shared NativeSequence session for Add clip, parent progress, admission recovery and original-owner Library Continue video. Existing framewise upscale controls require a connected
host, known applicable state and current rendered job identity; reserve the exact
print/job key until the authoritative transition response reconciles its state.

Queue rows reserve fixed preview and progress slots across state changes. Preview caching must fence host, instance and job, bound decoded memory and concurrent reads, and coalesce in-flight requests; loading images must not move the scroll anchor. Sync resume receipts precede retained-input and organization work; only stable final source/destination evidence may seal skip tokens.

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

The compact Generate prompt stays bounded; Edit prompt opens a spacious native sheet live-bound to the same authoritative draft. Done and ordinary dismissal retain edits. Keep native selection, clipboard, IME and undo; editor arrows never recall history, and generation shortcuts cannot dispatch behind the sheet. Clear/Undo clear preserve original prompt provenance and the existing rewrite undo marker until a later edit or replacement. Prompt-only history replacement clears unrelated rewrite provenance without changing model, inputs or settings; failed/offline history reads remain visible and retryable.

Show date separators is a persisted Mac-local Library Settings preference, default on and included in preferences reset. Off projects one continuous section without changing visible sort/filter order, selection or unread identity. Include it in section cache invalidation and retain a visible print anchor through regrouping.

Library viewer removal preserves the previous filtered order and copy identities: keep any surviving copy of the current print, otherwise advance to the next survivor, then the previous survivor at the end. Close only when the list is empty. Web updates the print URL when advancing; explicit scope/filter changes retain their existing navigation behavior. Native Mac successful trash moves do not leave a completion notice; preserve progress and uncertain/interrupted outcomes.

**Source Media.** The sidebar destination and Details cards use the shared
`SourceMediaCatalog` and `SourceMediaAttachment` contracts. The dedicated native Source Media section includes all live collections, including hidden ones, and applies its selected collection/machine filters before enumerating physical owners independently;
never use a different copy's member ID. A Source Media row creates no pin.
Authenticated thumbnails are separate from original attachment bytes. Fence
imports by current model/recipe/media revisions, origin route/key/instance and
view lifetime; preserve authored scalar edits while a download completes.

**Stable library status and clip duration.** Library connection/failure/cache status lives in an always-present toolbar slot, never an inserted grid row. Both native apps read model `supports_duration_prediction` and `runtime_ready` for automatic clip length: omit frames only for the active supported automatic preference, persist/restore that preference independently of the explicit frame count, and preserve it through unsupported models.

Native model recovery prefers the queue row’s `model_download` hint, requiring a nonblank exact match to the row model. A present malformed/null/mismatched hint blocks legacy fallback without dropping the row. Only absent hints permit typed batch-code fallback. Preserve actual missing-component prose and diagnostics; use the existing exact-host, license-aware download-and-retry coordinator.

Source browser filters project visible usages before selecting preview or reuse authority.
Opening a source shows its original with existing native zoom/playback components,
metadata and every host/print/member use; never collapse several roles in one print.
Load only the selected original into a bounded private temporary file, cancel on
dismissal, revoke on authority changes, and release playback/files on close.

Source access must use revision-keyed indexes, never repeated whole-library scans
per usage row. Cover same-count metadata, visibility, route/key/instance changes.
Open gallery originals require current exact member inventory, including refusal
revocation; owner-specific generations must not reload unrelated originals.
Unknown originals are inspected on demand from bounded authenticated files, never
inferred from filename. Detected kinds are ephemeral and bound to exact load keys.
Sync comparison/hash/import metadata work stays off-main with chunk cancellation.
Draft navigation saves asynchronously; app-lifetime termination flushes the latest
snapshot even when Generate is no longer mounted.

**Shared native read policies.** Use `QueueTransferSource` for move retries: only authoritative 404 plus an existing verified destination means the original is absent. A present job that started rendering is ineligible, not removed; transport/auth/server errors remain errors. Source Media filtering and original/inventory publication fencing belong in MoldClient's `SourceMediaDisplay` and `SourceMediaLoading`, while AppKit playback and presentation remain local. Revoked/cancelled responses and stale errors must not publish. New/unread history uses `LibraryUnreadLedger` and active video layout uses `VideoViewport`.

**Local-sync stages.** Keep source discovery, destination preparation, planning, transfer, organization, final checkpoint verification and reporting as explicit typed stages. Preserve backend call order, durable output receipts before retained-input transfer, route/instance fencing, intentionally removed copies and per-file/per-collection failure exclusions. A stage extraction must retain the LocalSave and SyncSession regression suites, including final-checkpoint ordering.

**Sync costs O(changes).** Discovery pulls every source replica concurrently
(`.all` for Sync All, so tombstones know both partitions) and reads one
`ReplicaMirrorView` per machine. An explicit Sync All press
(`syncNow`, `syncAllLocally(fullPlan: true)`) plans EVERY print, skipping only on
matching source+destination checkpoint evidence (one GET per machine, cached
digests: still nothing per print when unchanged). Keep in Sync's automatic runs
(`fullPlan: false`) consider only names in
`GalleryReplica.changedNames(since:)` from each source's persisted mirror
cursor (`LibrarySyncRecords.SourceCursor`, keyed by host, route-checked with the
origin context), plus the bucket's retry set, pending organization, and names
This Mac's own log changed (mapped back through the records). A nil answer
(new epoch after relaunch, trimmed log, route/instance change) is a full plan,
exactly the old behaviour, as is a bucket whose `lastFullPlan` is over 24 h
old. A retained-input-only change with no row change is found by the next
press or full plan, never later than a day. Machines a run could not read keep
their source cursor, and This Mac's changes to their copies go to retry (or the
destination cursor stays put when those names are unknown). Every considered key not sealed this run goes into
retry; cursors advance only on an unstopped run, and the destination cursor
advances past this run's own imports while carrying any name in that stretch it
did not seal. Receipts come from the import answer (`LocalSyncRecord(receipt:)`,
`provisional`, never tokens) and are journaled with `synchronize()` before any
retained input moves -- a receipt that cannot be written fails the item. No
per-print `galleryPrint`, and lane refill never waits on bookkeeping. A
provisional record never matches by facts: it links only through the
occupied-name byte comparison, unless this same run upgraded it from the
refreshed listing. Checkpoints are the bounded `gallerySyncCheckpoint(filenames:)`
for considered/touched names (the whole GET only for full plans; 404/405 falls
back inside MoldClient). Records live in `LibrarySyncRecords` (Application
Support `LibrarySync/records.json` + `journal.jsonl`, buckets per destination
instance, one-time migration of the legacy defaults keys, tombstone pruning only
when source and copy are gone from both partitions). A document that cannot be
read (corrupt or newer version) is moved to `records.json.unreadable-<ms>`, never
overwritten, and blocks Sync All with a reported issue until that file is
removed. Each `LibraryStore` defaults to its own memory-only store; tests that
"relaunch" pass two stores over one temporary directory. Links refresh once per run, never per receipt. Recipe
digests are cached per host/filename/media version/raw-recipe SHA-256 (rows
without raw bytes are not cached). Collection visibility is one
pass under the cancellable `VisibilityWriterLock` with one inventory assignment; progress
publishes at most ~10 Hz. A non-echo remote replica delta (`absorb`, outside a
run) calls `noteRemoteChange`.

**Encrypted remote access.** Remote host additions claim a version 2 pairing link and persist public pins/grants separately from SecretStore credentials. The shared MoldClient transport and media loader are mandatory for all remote requests. Only numeric loopback reaches the embedded API directly. Remote Access opt-in opens the secure LAN WebSocket listener and a Lambda connector using the server’s existing owner-only `secure/host.identity`; discovery is public metadata, not owner enrollment. Publish the validated WSS endpoint into ConnectionAddresses before producing a QR. Stop closes cloud and LAN channels. Never expose the operator bearer, retain old cloud owner tokens, restore plaintext remote fallback or replay lost mutations. Native builds automatically build the GPU-free Rust XCFramework, including remote-only builds. Tests must use an isolated app bundle ID while an installed host is running.

Source reuse stays discoverable in inspector cards and original viewers through the same declared button/context action list. A native chooser offers only connected, installed, runtime-ready compatible recipes; explicitly adopt the selected recipe once, without reconciling through the default. Revalidate source access before changing the draft, and dismiss original details on successful staging.

**Input inspection.** Populated PictureWell and MediaWell offer a separate View
control; PictureWell adds View Original to its declared menu. View never imports,
fits, reorders, acknowledges unread prints or changes the draft. Authoring sources
prefer sourceImageOriginal. Typed retained references resolve the exact ordered
member after fresh inventory verification. Queue inspection uses the explicit-index
input-original route and says submitted input: sealed bytes can already be fitted.
InputMediaViewer owns private files/players only while selected; cancellation,
replacement and route/instance changes revoke inspection. Native image decoding
preserves pixel geometry within the 32-Mi-pixel decoded safety limit.

Known Secure credential fingerprints are retained across native mode changes and aliases. Regular requests refuse matching host keys, X-API-Key and Bearer headers before transport; the Rust C ABI hashes the denial history, which never stores credential bytes or grants access.

Whole-batch cancellation rechecks identity but follows current child lifecycle changes while earlier cancellations await. Queued-to-running/held transitions must not skip an otherwise cancellable sibling. Individual held-only cancellation and Pause/Resume retain their narrower state guards.

Library machine and media toolbar menus keep their current query selection visible,
even under the toolbar’s icon-only label style. Preserve native selection checkmarks,
multi-token summaries, bounded long host names and full accessible/help labels.

Completed retained-input materialization records local ownership only when the
authoritative inventory has no hidden roles or unresolved descriptor references.
A durable independent snapshot then restores without reviving the archive locator;
missing, stale and descriptor-bearing snapshots stay fail-closed. Queue preview
slots show the shared neutral placeholder, preserving row dimensions through loading.

Older invalidated source-only snapshots may become independent only after fresh
origin route/instance and archive/output verification plus a singleton source-image
inventory. Preserve local bytes; never mint authority, replace attachments, or
waive hidden inputs and stale/cancelled responses during this migration.

**Library replicas.** `LibraryStore` reads each machine through one MoldClient
`GalleryReplica`, kept across route changes and replaced only when the API key,
connection mode or server instance identity changes. Every pull carries the exact
record it read through (`Pull.origin`); `absorb` refuses a delta whose origin is
no longer the current record and calls `redeliver()` so the next pull is a reset.
A legacy frame spliced into `perHost` must call `spliced` (`GalleryReplica.forget`)
so the next listing re-emits those names; a `.restored` frame for a missing row
inserts it. Listing answers carry per-partition generations (an older answer
landing last is dropped) and feed pages commit once at the end (no rollback).
Replica files live in Caches, are never rewritten after `discard()`, are pruned
for removed machines, store only the raw recipe, debounce 5 s and flush on quit. The
feed is used only when `capabilities.gallery.changes_feed.protocol_version == 1`;
404/405/`503 GALLERY_FEED_NOT_READY` falls back to the conditional listings for
that pull. The replica owns the listing ETags (`invalidateListings`, never a
store-side ETag map) and delivers deltas in revision order through its sink
BEFORE `pull()` returns; `absorb` only edits `perHost`/`trashPerHost`, replays
the outbox, and notes ids, and whoever started the pull calls
`settleTimelines()`. A nil delta must not bump `rows` or re-sort. Single flight is
per lane (feed, live listing, trash listing): a tickle mid-pull queues exactly
one follow-up; cancelling the caller that started a pull changes nothing. On a
feed machine gallery frames and `resyncRequired` only `tickle` the replica (echo
suppression, bulk targets and import suppression still apply first); legacy
machines keep in-place frame application, `RelistGate` and `RelistMerge`.
`relist` is a FULL pull (a reset even when unchanged) so a refused optimistic
edit is undone. `LibraryTimeline` patches in place only when a full merge could
not differ (unchanged merge identity/order key, no shared filename/identity/link,
no key tie); otherwise it rebuilds. The legacy trash LISTING, not `trashedAt`, decides a row's partition (`ReplicaDelta.inTrash`). A direct `perHost` write outside those paths
marks the timeline for a full rebuild. Tests construct `LibraryStore` without a
`replicaStore`; only `AppStores` persists to Application Support.

Library grid/viewer Delete (both SwiftUI U+0008 and physical AppKit U+007F) and forward Delete trash; in Trash they confirm deletion using
only the trash endpoint. Bare F is scoped to Library interaction focus. Editable fields keep
all typing keys. Shift-Command-Delete confirms machine-scoped Empty Trash. Print menus never
contain collection Rename/Hide/Delete; the collection row and menu bar own those actions.
Operation status uses the content-column inset and operation/issue identity for dismissal.
Dismissing never stops work or acknowledges issues; toolbar status restores it and retains Stop.
Source file actions use SourceMediaLoading.original with exact authority/revision fences and
release private files on dismissal/revocation. Copy commands include the sanitized owning-host
selector, full member ID and shell-quoted filenames. Interpolation endpoint replacement keeps
interior keyframes and migrates an existing first-frame source into frame 0 when adding a
last endpoint; explicit refused boundary capabilities cannot use the legacy fallback.

Library menu supplements and Commands capture destination snapshots and callbacks before
native hosting; they cannot require window-only environment stores.


**Fresh typed-video preparation.** Keep ordered authoring inputs in `AuthoringReference`: canonical descriptors and immutable fresh original snapshots are distinct. Pending facts cannot encode through `GenerateRequest` or placement. Use only negotiated keyed upload V2 + nested normalization V1 for fresh typed `reference_video`; scalar source/continuation paths, retained archives and old leases stay on their existing contracts. Freeze the candidate POST bytes and domain-separated digest, validate original/canonical receipts and exact final descriptors, explicitly finalize, then price and admit the unchanged canonical request on the captured normalizer. Auto filters existing host order by health/model/capability; a hard machine selection never falls back. Full route/key/mode/secure identity/instance and cancellation checks fence every await; release every unused session. Persist originals, never ephemeral handles or candidate authority.

**Playback endpoints.** `VideoPlaybackEndpoint` resolves the last displayed sample in asset presentation time, including edit lists and B-frame ordering; item duration minus guessed frame intervals is not an endpoint. Both native transports share `VideoPlaybackSession` for ended/replay/repeat/full-end seek behavior and item/seek-generation fencing. Preserve legitimate audio tails and native controls. Cancel asynchronous endpoint/seeks on replacement or dismissal, and prevent late resumes after pause/inactivity. Keep local and encrypted fixture coverage for CFR/VFR, offset timelines, one-frame video and audio overhang.

**Observational stalls.** Native Queue rows/detail render the optional shared
`StallDiagnostic`; executing preview polling includes cancelling work and clears
on authoritative null. Hide stale ETA/fractions while stalled, label decode
counts as completed current-phase work, and preserve pending cancellation after
202 until actual host state settles. Queue row builders receive captured status
text rather than introducing an environment dependency in detached content.
The curated Performance threshold mirrors 0–86400 seconds and never changes
lease ownership. Keep copyable diagnostic facts and operator recovery guidance.

**Native performance diagnostics.** Nightly versions enable the bounded
`PerformanceDiagnostics` service; stable builds opt in with launch environment
`MOLD_PERFORMANCE_DIAGNOSTICS=1` (`0` disables any build). Instrument whole
operations/cache misses using the fixed `PerformanceOperation` vocabulary,
never every row/body/cache hit or private strings. Keep heartbeat work on its
utility queue, one outstanding main acknowledgment, lifecycle/sleep suppression
and 30-second warning limits. Use `apps/macos/scripts/capture-performance.sh`
against the existing PID for private local evidence; never commit raw stack
samples. See `apps/macos/docs/performance-diagnostics.md`.

Source Media is a separate Library section, not a Prints/Source toggle or interleaved collection content. Reuse Library filter labels, machine selection, sort, tokenized search and thumbnail sizing. Cache owner projections independently of scan progress and zoom. Keep empty source details invisible and one-shot; negative inventory answers expire on relevant owner/host invalidation. Keep same-authority previews visible while refreshing. Library subtitle/sidebar count caches must key on row and receipt privacy revisions, never sync progress; ordinary print hiding is unchanged. The receipt privacy revision is content-addressed: `SyncPrivacyEdge` (MoldClient) is a record's whole fingerprint, with a missing destination host normalized to the local host, so reconcile bookkeeping never moves it. `privacyLinks` spans every record (a conservative superset of the verified `links` snapshot) and caches on that revision, never `recordsRevision`. Counts read `countEpoch` (`SyncBatchedValue`): at most one new value per second while a sync runs, the first at once, exact on stop; the grid and source access stay exact. So during sync a sidebar/subtitle count may include a print newly hidden by filing or receipt for up to one second; toggling a hidden collection is immediate (`hiddenCollectionIDs` stays in the key). Privacy links follow records live, so a copy replaced mid-run stops protecting at once rather than at `finishRun`. A subtitle with no filter beyond machines shows the grid count with no second count, and a filtered baseline is never below the rows shown.

Source Media uses shared `JustifiedLayout` geometry, oriented source ratios (never generated-output dimensions), stable source identities and a single bounded synchronous warm-pixel cache. Batch ratio discoveries and retain a visible anchor through reflow. Do not poll inventories/thumbnails or turn warm cache hits into loading states. Invalidate old pixel work before publishing the next content epoch; keep same-authority display rows read-only during revalidation, while revoked authority and server refusal still remove access. Display-only sync updates never invalidate source content; identical-content row events remain conservative because retained-source repairs can arrive independently of output media versions.

Unread ledger derived copy indexes/counts are transient and retain the persisted
wire shape. Safe timeline patches may observe only changed logical groups when
there is no hidden privacy graph; a merge/visibility/host change uses a full
projection. During Library Sync, bound full unread projection to one leading
one-second update, with immediate view/visit, sync completion and quit flushes.
Never mark hidden, unvisited or later-arriving prints read through this batching.

**Multi-clip video.** In Short clip, choose **Add another clip** to enter ordered prompts and lengths, then **Generate video**. The app validates the draft automatically before submitting it. Move or remove rows without opening a separate editor. Library **Continue video** makes the next separate linked MP4 from the latest completed authored take on its original machine. Ordinary one-prompt generation and Reuse settings remain available. Shared wire/draft authority lives in MoldClient; SwiftUI session/editor composition lives in `apps/shared/NativeSequence`. Retained reference bytes stay in memory. Recovered drafts with missing references require explicit restoration or removal before publishing. Follow `chain-sequences.md` for owner, latest-take and operation recovery fences.

**Pre-store startup.** The first window contains only startup state. A single
owned worker resolves the existing Mold home/private operator key and completes
the one-shot FFI preamble before AppStores, HostStore, app-owned networking or
Sparkle activation. Never retry credential IO from engine Start after stores
exist, substitute another home/key, or create a timeout task group that waits on
an uncancellable filesystem syscall. Cancellation revokes late bootstrap/store
publication; preparation failures require relaunch. Startup remains responsive
and Quit remains available while storage is blocked. Access recovery uses
NSOpenPanel for the exact canonical existing home, never a replacement home;
retain granted access for the process and fence late panel answers. Opening or
cancelling this panel must not start another bootstrap worker.

Extend restores use shared token-owned cancellation/authority state. A changed captured owner or cancelled restore permits fresh inspection; old completion cannot overwrite its replacement or a newer explicit Library action. Queue listing errors stay page-owned, and the shared content predicate retains page Retry/loading rows in an otherwise-empty Queue.

Duration progress followers bind their slot to the full captured owner. A replacement owner supersedes a pending stale reader without waiting for its socket; token-owned cleanup cannot remove the successor. Cache-only status is explicitly last known whenever the row’s current-owner predicate refuses controls.

Completed-parent dismissal is durable, device-local and scoped to host UUID, installation and operation; captured recovery/page/activity reads must never recreate a removed row. Recheck the live row before accepting a completed-only gesture. Confirmed server deletion remains distinct, with immediate busy feedback; only the exact owner-fenced VIDEO_JOB_NOT_FOUND response may settle an already-removed job. Gallery navigation verifies the exact physical output, keeps unknown/offline separate from missing/Trash, and observes authenticated Gallery invalidations. Pending output reads coalesce only for the same full owner and cannot overwrite a removal or successor.

Video parents and authored sequences are ordinary Queue rows inside their machine's section (`NativeVideoQueue` items drawn by `Queue/VideoQueueRow.swift`): the fixed thumbnail slot with the neutral placeholder, a state glyph from the shared `VideoJobRow` symbol, the model presentation headline, the shared caption and detail (`docs/contracts/video-job-rows.json`: "Waiting", "Rendering clip 2 of 4", "Joining clips", never "1 clips"), icon-only borderless Stop / Resume / Open in Library / Dismiss buttons with `.help`, `.rowActionMenu` carrying every offered action (Start again…, Delete job… last), and a row click opening the details sheet. There is no Check queue button: `NativeVideoQueue.recheck` (the owner re-check plus saved-output verification) runs on the pane's Refresh, its load and a 15-second poll. The subtitle (`QueueSummary.sentence`), the sidebar Queue badge and Machines work figures add live video parents (`NativeVideoQueue.liveSymbols`); their clips never arrive as `/api/queue` rows, and their activity ids are excluded from Also Running, so nothing counts twice.

Completed and stopped video parents and authored sequences leave the Queue on their own, like every other row (`ObservedVideoJob.belongsInQueue`, `NativeSequenceSession.queueJobs`); they also stop holding an otherwise-empty Queue open. Failed and interrupted parents stay as recovery work. The video itself is in Library.

Video Queue rows use an explicit independent button style inside List; a Delete gesture must not also run Dismiss, the owner re-check or Library navigation. Anchor its confirmation to the exact selected row and owner. Cover real app taps, including Cancel followed by one-press dismissal and relaunch.

**Length and seed (both native apps).** `apps/shared/NativeSequence/NativeLengthControl.swift` is the one Length control. On a Duration machine it is seconds in 0.1 s steps from 1.0 s up to one generation, read exactly ("14.0 s"; a restored off-grid value is shown and sent unchanged), because the server trims to the exact length; its window and overlap come from the video-job PLAN (`VideoLengthGrid.duration`, which is the recommended clip window for every family but H3), with chain limits only until a plan exists. An ordinary render's slider snaps to the model's frame grid (`VideoLengthGrid.resolve`: chain limits, else the recipe; never hard-coded). with **Automatic from prompt** hiding it (Automatic is an ordinary render; Extend always stays a video parent) and **Longer video** switching to the scenes editor, where each scene has a prompt and its own 0.1 s slider from 0.1 s up to the always-one-clip scene length `(single_clip_max_frames - overlap_frames) / fps` floored to 0.1 s, plus Add scene and "Total 38.6 s · 3 scenes". Off, a length within one generation still submits through the Duration flow as a one-clip `/api/video-jobs` parent. A saved length the sliders cannot show becomes equal scenes of the same total (`VideoDurationDraft.fit`), never a silent change. Plan warnings (repeat / shorter than the shortest generation) are shown. `NativeSeedControl` is the one seed control: "Seed", segmented `Random | Fixed`, a string-backed full-range UInt64 field (empty or partial text keeps Fixed and the last valid seed) and a dice New seed button; words and logic are MoldClient's `SeedChoice`. The Duration editor and the Length bounds read `VideoPlanController.shownPlan`, the last answer kept while the next length is calculated; submission checks (`openingConflict`, `usesSourceEnding`) read the strict `plan`. The clip line and plan total appear only when the plan joins more than one clip (or when extending), so dragging within one generation never resizes the panel.
