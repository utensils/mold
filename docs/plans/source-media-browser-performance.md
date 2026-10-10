# Source Media performance and deduplication plan

Scope: native iOS/macOS plus web and Tauri desktop/mobile, including shared source preview loading in detail panes. Browser presents identical verified source bytes once, preserving every owning print, member role and exact reuse route. No storage deduplication, pin deletion, retention or GC changes.

## Evidence and targets

Current measured Vue fixture: adding one owner refetches 100 inventories; reorder refetches 100;100 owners × 30 members mounts 3,000 cards. Both native browsers reset catalogs/IDs and repeat full-library projection. Client previews refetch on remount. Server inventories decrypt an entire archive; each thumbnail resolves/decrypts again and materializes sibling plaintext. Server fixture:11 references → 4 distinct contents across 3 owners.

Measure before/after using identical fixtures: request/decrypt/render counts, first usable cards, warm reopen/search, mounted card count, native scroll responsiveness and memory. Report real runtime measurements separately from deterministic counters; no invented speedup percentages.

## Server and wire (server agent)

- Add optional content_sha256 for the exact independently reusable content bytes. Raw members use authenticated manifest SHA; serialized keyframes require image-byte SHA derived during the already-required first authenticated scan. Never trust archived keyframe metadata alone. Missing/oversized/unverifiable digest remains unique by owner/member.
- Bounded per-store verified-manifest/digest LRU and per-key singleflight. Each hit still opens/validates exact pin using no-follow and robust file identity/version; fresh publication authority remains mandatory. Compare file version before/after cold decode; bypass cache if a robust fingerprint is unavailable. No decode under global cache lock.
- Selected-member reads avoid retaining/materializing sibling plaintext while preserving whole-stream authentication.
- Bounded in-memory derived-thumbnail LRU/singleflight, consulted only after fresh auth, publication and pin validation. Cache keys include validated content identity and transform/version; no-store remains; decode semaphore only held for misses. No persistent plaintext cache.
- Cold unseen archives still require one full scan with existing format; return small initial pages promptly rather than promising zero cold cost.

## Shared browser behavior (native and Vue agents)

- Stable incremental inventory reconciliation: retain unchanged owner results on reorder/append; immediately remove changed/hidden/deleted/unauthorized owners; only fetch new/revised owners with bounded concurrency and cancellation. Start a small batch for first content, then bounded subsequent batches.
- Deduplicate by verified content_sha256 alone across visible owners/hosts. Unknown identity uses exact owner/member fallback, never name, size, preview similarity or unverified metadata. Stable aggregate IDs must not depend on a random reload UUID or current first owner.
- One card per unique content, summarize all roles and owning prints, expose owner context accessibly. Reuse captures an exact currently authorized owner/member and existing draft revision; all existing capability/attachment routing fences remain. A hidden owner may not supply metadata, preview or reuse just because another owner has identical bytes.
- Memory-only preview cache/coalescing with a fixed byte budget and bounded inflight requests. Bind authority to route, credential generation, server instance and owner revision. No persistent private bytes; immediate privacy invalidation independent of TTL; cancel requests no longer needed. Shared across browser/details where identity is sufficient; fail safely when identity is absent.
- Bound mounted member cards with windowing or incremental member pages, not merely 100-owner pages. Keep keyboard/touch access and load-more behavior usable, and avoid layout jumps.
- Native: cache owner/group/search presentation, avoid deep Set<LibraryEntry> hashing and repeated body projection; batch catalog publication; move preview decode/downsample off main actor as needed. Preview coordinator bounded cost/inflight and stable row identity.

## Ownership and review

Changes use branch `fix/source-media-browser-performance`. Server owns Rust/core contracts; native owns apps/** Swift models, UI and tests; Vue owns studio/** and desktop/web integration. Root coordinates docs, builds, UAT and publication. No agent edits another area without agreement.

Peer review this plan before implementation: native reviews server cache/digest/lifecycle; Vue reviews native/UI dedup and resource strategy; server reviews client authority/cache/reuse invariants. Resolve blockers first. Add failing tests for confirmed regressions, then targeted behavior tests. Preserve all earlier source privacy/corruption/auth distinctions.

## Verification and delivery

- Counter fixtures: append one owner fetches one inventory; reorder fetches none; warm reopen/search0duplicate thumbnail/decrypt work; unique counts correct; mounted cards bounded even with 3,000 members.
- Auth/lifecycle: stale async completion, credential/route/instance changes, hidden/deleted/replaced owner, pin tamper, unknown digest, distinct content with same name, keyframe image digest, duplicate roles/owners, cache eviction/retry.
- Actual iOS simulator and macOS UI UAT; browser/Tauri mobile/desktop responsive UAT with repeat browsing and source reuse through existing non-generating attachment flow. Only one simulator at a time; no generation jobs.
- Update owning docs and changelog. Independent code review; relevant exact-head CI; merge under existing authorization after UAT and green checks. Server deployment only for required server changes, coordinated to avoid interrupting active work. Clean agent build artifacts after completion, retaining evidence.

The LTX decoder investigation and qualification are tracked separately in PR #1849.

## Peer-review resolutions (required implementation constraints)

All three peers reviewed before implementation. Their corrections supersede earlier wording:

- Group solely by valid verified content_sha256, not media kind (identical bytes may have different role-derived kind labels). Choose presentation kind from available known metadata; never lose any usage.
- Client preview keys remain exact owner/member + route/credential/instance/revision. Content dedup reduces card/request duplication without sharing authority across owners. Capture a visible eligible owner at click; do not silently fail over after download starts. Aggregate cards expose an accessible usage list/chooser including multiple roles in one print; keep large usage lists bounded/lazy.
- Server adds a shared two-operation cold-decrypt concurrency bound across inventory and thumbnail misses, with bounded extracted-member bytes/count. A separate two-operation thumbnail pipeline limit spans selected-byte allocation through rendering and plaintext disposal. Singleflight and LRU alone are insufficient. Validate whole-stream authentication and recheck current publication/pin before publishing/cache serving.
- Client inventories have bounded freshness (60-second target), explicit refresh and retained-media/gallery invalidation. Transfer may add pins without changing filename/output identity. Warm-zero-request claims apply within freshness, and failed/unavailable answers remain retryable. Display-only changes update titles/labels without invalidating unchanged content; hidden/replaced owners invalidate immediately.
- Preview coalescing uses subscriber reference counts: one cancellation cannot kill another consumer, final subscriber cancels, authority revocation clears all affected subscriptions/visible state.
- Vue uses 48-card replacement pages with Previous sources and Next sources, separate from Check more library items discovery. Search scans all loaded aggregates; counts explicitly describe loaded/checked scope. Native lazy presentation also needs bounded preview caches and stable projection; large owner lists are not eagerly expanded.
- Default preview budgets: native 32 MiB decoded cost, Vue 16 MiB blobs, at most 4 client requests in flight. Server cache budgets and entry limits must be explicit and tested; do not hold cache locks during I/O or rendering.

Implementation can proceed on the reviewed plan. Root will independently review final code and require native/browser UAT plus before/after evidence before merging.
