# Client reliability implementation plan

Scope: one PR for sync correctness/efficiency/settings, persistent per-client new-media semantics and iOS navigation badge, macOS queue scrolling, and cross-client pre-render queue transfers. The separately delegated justified grid is excluded. No merge or deployment is authorized.

## 1. Sync

Replace misleading whole-inventory Saving progress with structured discovery/check/copy/organization phases and actual counters. Preserve retained-input repair independently of output versions. Introduce safe change evidence and durable checkpoints as needed to skip unchanged per-item work without losing late retained inputs or intentional local removals. Fence cache trust by source/destination identity. Persist a configurable nonoverlapping sync interval in Mac Settings, default five minutes; changes take effect on pending/next repeat and Stop cancels pending repeats and stops remaining work at the existing safe transfer boundaries. Only add interval UI where recurring sync exists. Audit analogous manual/offline progress across other clients.

Tests: small delta against large existing inventory, unchanged repeat request counts, changed retained inputs with unchanged output, cancellation/restart boundaries, removal, replaced hosts, failures, interval persistence/change/validation/nonoverlap.

## 2. Client-local viewing history

One persistent viewed authority per client drives New labels and applicable counts. Opening Library, filtering, refreshing and preloading neighbors do not mark media read; actual displayed viewer media does. Preserve merged-copy deduplication and host-safe identity, hidden/Trash suppression and offline history. Add iOS Images navigation badge. Apply equivalent semantics to macOS, Tauri and web. Migration preserves existing read status (user accepted recommended UX); future arrivals remain new until individually viewed. Bootstrap new installations/hosts conservatively to avoid an unsolicited historical badge flood, documenting the baseline exception. Existing known-unread media stays unread. Prefer successful display over failed loading as the read boundary where feasible.

Tests: restart, separate clients, one viewed group only, arrival while Library open, filters, same names/different outputs, copied media, hidden/trash/restore, migration, navigation and icon count agreement.

## 3. macOS queue scroll performance

Reproduce with safe fixtures before selecting a fix. Measure/inspect row geometry changes, thumbnail work and publication churn during scrolling and refresh. Stabilize geometry and cache/fence preview work where supported by evidence. Preserve row controls, accessibility, batch expansion and multi-host ordering.

Tests/UAT: long mixed queue, missing and delayed thumbnails, bottom/middle/top, repeated equivalent refresh, batch hydration, stable content extent and responsive scrolling. Do not mutate real queues.

## 4. Pre-render transfer

Extend current Held-only protocol through an atomic server reservation that excludes dispatch and preserves recoverable original state/media. Explicitly handle queued/paused/Held and pre-render claims. The server Running transition at worker dispatch is the existing UI Rendering boundary; already-dispatched jobs are excluded. Renderer ownership must be fenced, never inferred from a stale displayed label. Reuse export, identity checks, destination idempotency and admission; source completion only after proven durable destination acceptance. Ambiguous acceptance must reconcile, never blindly resume source. Seal the reservation before admission; a durable destination abort tombstone must exclude late admission before a confirmed refusal releases the source. Add additive capability/action authority so old servers remain Held-only. Expose visible Move to on native Mac/iOS, Tauri/web and maintained mobile queue variants when another distinct eligible host exists; explain incompatibility and preserve original on refusal.

Tests: transfer versus claim/render/cancel/resume, restart and response loss, failed admission, duplicate routes, missing capabilities, input retention/order, per-row busy state and UI state matrix.

## Delivery

Read owning rules; failing regression tests before changes. Agents have exclusive file ownership and may coordinate contract changes. Parent reviews every diff and independently validates targeted checks/UAT plus combined required CI routes. Update changelog and affected docs/rules. Conventional commits on feat/client-reliability, push and open one PR, leave unmerged. Record unavailable validation honestly. No real generation/queue mutations for testing.

## Implementation evidence and review

- Sync: conditional listings and per-print source/destination evidence eliminate repeated no-op offers while preserving retained-input repair. Per-copy receipts survive failed media/organization phases. The batched endpoint still walks O(N) metadata and manifests; it is not a constant-time change cursor.
- New media: persisted client-local ledgers drive labels and counts; successful selected media display marks only that logical print viewed. Neighbor preload, failed decode, and Library entry do not. Migration preserves existing read status.
- Mac queue: the original native List regression reproduced a 33-point row shrink and 330-point bottom scroll jump when missing previews resolved. Fixed caption/image slots and bounded cached decoding pass the same regression.
- Transfers: review required durable sealing and destination abort tombstones to exclude duplicate rendering across concurrent clients, restarts, and lost responses. Modern sources require a matching reservation even for Held exports; older clients receive an upgrade refusal.
- Parent review also corrected stale viewer callbacks, duplicate polling, local storage pressure/failure behavior, reservation recovery wording, native fixture authority, preference reset inventory, and shared button styling.
- Results displayed before their gallery listing persist a pending viewed identity; background media loading does not advance viewing history. Native repeated-inventory reconciliation uses a set intersection instead of a quadratic scan (20,000-print fixture covered).
- Transfer reservations bind the destination's opaque durable queue identity, while admission still fences its current runtime identity. This permits reconciliation after a restart and refuses a replaced queue owner.
- Independent final sync review added a regression for a source/destination route change while checkpoint requests are in flight; even a no-op skip must revalidate the captured host records.

Final combined checks passed 1,031 Mac tests, 1,270 shared Swift tests, 2,411 studio tests, 2,199 web tests, 7,440 desktop/mobile tests, 1,941 Rust core tests, and 356 database tests. Native lint, frontend architecture/dead-code/format checks, web/desktop/mobile production builds, and website verification/build passed. Rust core's shared-process socket fixture failures disappear with serial execution. The server suite passed 2,475 tests but two unchanged VAE companion-resolution fixtures selected installed model-cache files during the full run; both pass individually with isolated configuration. Queue protocol, restart, race, and OpenAPI tests pass.

Browser fixture UAT confirms visible queued/paused Move to controls, no Rendering move, and a distinct destination chooser. Viewing preserved one unviewed badge after viewing its neighbor, then changed the remaining badge from 1 to 0 on display and kept 0 after reload. iOS simulator UAT caught a real missed player-readiness event (decoded video with a lingering spinner); current-view KVO now drives the successful-display boundary, with a positive readiness assertion. Final simulator and PR-head results are recorded in the PR. No production queues or media were mutated.
