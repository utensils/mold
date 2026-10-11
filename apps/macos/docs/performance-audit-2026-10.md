# Native macOS performance audit — October 2026

## Baseline and scope

Installed and running Mold Studio 0.34.1-nightly.1505 matches the published
nightly and commit 238370af005c23715cbdde7034f955ca10e701f1 (PR #1854).
The user reported a near hang opening a retained reference in a 3,669-print
library. A three-second sample of the actual process on October 10 showed
SourceMediaDetails.current → usages → LibraryStore.allowsSource →
sourceMediaOwners repeatedly projecting the complete library on the main thread.
CPU was 100.9% of one core; sample physical footprint 433.7 MiB, peak 810.6 MiB.
The app was terminated at the user's request after preserving the trace.

The same reference is classified `unknown`; the viewer rejects it without
reading its original. This is separate from the repeated projection hang.
No private source images or host credentials belong in this report or fixtures.

## Implementation plan (independently reviewed)

1. Replace per-member full-library authority scans with a reusable indexed
   projection. Cache only across unchanged rows, trash, collection visibility,
   host routes/credentials and instance identities. Resolve visible usages and
   selection once per details render. Keep immediate privacy revocation and
   separate gallery versus explicit print-details access.
2. Inspect unknown originals only on deliberate open, through an authenticated,
   size-bounded temporary file. Identify image/video/audio from actual content
   using platform decoders off the main actor, not filenames or an assumption
   that references are images. Preserve unsupported/corrupt failure states.
   Investigate archived metadata provenance; prevent new missing kinds if the
   evidence identifies a producer bug. Do not bulk-download originals to classify.
3. Avoid restarting the selected original for unrelated gallery events. Scope
   viewer invalidation to its owner/member and current authority; cancel and
   clean up on dismissal, source replacement, visibility or host changes.
4. Audit source catalog refresh/cache behavior. Keep unchanged authorized display
   snapshots during refresh where safe, revoke on explicit changes, and bound
   background work. No stale cache may authorize reuse or bypass new refusals.
5. Audit the rest of the native app: sidebar counts, library projection and image
   decoding, sync/persistence, queue previews, model lists, composer and playback.
   Record evidence and distinguish measured bottlenecks from code suspicions.
   Fix confirmed high-impact costs without changing user-visible capabilities.
6. Tests precede fixes. Cover invalidation, same-count edits, hidden mirrors,
   host removal/rekey, trash, cancellation, unknown/corrupt media, request counts,
   and reuse semantics. Use 3,669- and 10,000-print synthetic libraries with many
   usages and collections; tiny fixtures alone cannot qualify this change.
7. Release-mode rendered UAT: open/close/select original, filter/scroll/resize,
   navigate all top-level destinations, image/video/audio playback, unavailable
   hosts, live updates, and repeated open/close. Record interaction latency,
   main-thread samples, CPU, peak/settled memory and temporary-file cleanup.
   No production generation or model download is required.
8. Update affected docs and changelog, independent final code review, one PR,
   exact-head CI. Avoid extra TestFlight releases. Remove task build outputs,
   stop test services and close every launched app/simulator, preserving evidence.

## Acceptance criteria

- Opening details performs no per-usage full-library projection. Repeated UI
  reads at unchanged authority do not rebuild the index.
- The initial details shell and dismissal stay responsive at realistic scale;
  network latency may delay an original, never lock navigation or cancellation.
- Unrelated owner updates do not restart current download/playback.
- Repeated open/close settles memory and leaves no original temporary files.
- Every optimization retains hidden-collection, exact-owner and host authority.
- Audit findings and runtime coverage limits are explicit in the final report.

## Independent plan review

The peer reviewer approved starting TDD with these required clarifications:

- Use monotonic data revisions, not full-array hashing on each index lookup.
  Retain before/after-await route and instance checks independently of monitors.
- Track per-owner content generations: a selected owner event reloads its
  original even when metadata stays unchanged; unrelated events do not.
- Bound unknown-content inspection by bytes, decoded dimensions, cancellation
  and an AVAsset inspection timeout. Use content-aware temporary-file naming.
- Separate cached display from mutation authority. An explicit auth/404/revocation
  must remove stale access. Do not relax cache validity merely for faster paint.
- Local targets: details shell/Done feedback under 100 ms p95; no sampled local
  main-thread computation stall over 250 ms; 20 open/close cycles without
  monotonic memory or temporary-file growth. Report network readiness separately.
- Broad audit does not justify speculative rewrites; profile suspected costs and
  record remaining limits honestly.

## Findings and disposition

| Area                       | Evidence                                                                                                                          | Disposition                                                                                                                    |
| -------------------------- | --------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------ |
| Source details             | Actual nightly sample repeatedly projects all owners from each usage row                                                          | Fixed: revision-keyed authority index and one selection projection per render                                                  |
| Unknown retained reference | Viewer refused metadata kind before reading authenticated bytes                                                                   | Fixed: on-demand bounded ImageIO/AVFoundation inspection; detected reuse type stays scoped to the inspected member             |
| Minute refresh             | Expiry removed every cached row before fetching replacements                                                                      | Fixed: preserve unchanged-authority display during requests; refusals revoke captured members                                  |
| Original reload            | Global gallery revision restarted unrelated open originals                                                                        | Fixed: per-owner content generations with exact member and authority checks                                                    |
| Sync                       | Whole-file comparison/hash and import metadata scan ran on main actor                                                             | Fixed: detached workers, chunk cancellation, unchanged route checks and receipts                                               |
| Navigation saving          | Generate disappearance synchronously saved inputs and waited on writer                                                            | Fixed: immediate background write; app-lifetime termination observer still flushes latest pending draft                        |
| Startup draft restore      | Synchronous restore reads/hashes/decodes potentially large input snapshot before reuse restoration                                | Remaining: needs an explicit app-wide startup readiness gate to avoid overwriting edits/attachments; not changed speculatively |
| Sidebar collection counts  | Each collection independently scans the merged library                                                                            | Scaling candidate: benchmark many collections before adopting an indexed count projection                                      |
| Library thumbnail pipeline | Per-request byte bound and cache cost bound, but distinct concurrent requests are unbounded; purge does not cancel in-flight work | Remaining lifecycle/resource audit finding; requires dedicated scrolling and cache-purge regression coverage                   |
| Library/result viewer      | Full image bytes and NSImage construction on main; no explicit decoded-pixel ceiling in older viewer                              | Remaining large-image candidate; NSImage can decode lazily, so initializer cost alone is not proof                             |
| Composer previews          | Detached decode exists, but no global pixel budget/caller cancellation                                                            | Remaining large-reference candidate                                                                                            |
| Queue                      | Preview cache bounds reads to four, entries to 128, images to 192 px; details recreate NSImage while rendering                    | No measured queue hang; render-time image creation remains a profiling candidate                                               |
| Models                     | Discover debounce, cancellation and pagination already exist; installed grouping sorts in render                                  | No evidence for a network rewrite; large installed inventories remain a profiling candidate                                    |
| Sync scheduling            | Three transfer lanes, reduced to one for large files/low disk; progress coalesced                                                 | Existing bounds preserved                                                                                                      |

## Validation record

- Shared regression run: 73 tests across source media and mirror file suites pass.
- Draft navigation/termination: four production-source hostless tests pass,
  including real termination notification after the Generate pane is absent.
- Native macOS and iOS architecture/accessibility source lints pass.
- Actual native app UAT used an optimized Debug build (`-O`, test hooks enabled),
  one read-only loopback host, 3,669 prints and 11,007 retained usages collapsed
  to three distinct assets. No production generation, transfer or model download.
- Extensionless unknown image opens as 1024×1024 pixels; all 3,669 uses remain
  linked. Twenty close/open cycles displayed the original successfully.
- Original video played to 0:02/0:02; audio advanced to 1/1 seconds. Escape/Done
  return to the browser; Generate and Models navigation remains functional.
- Details-open idle sample: 0–0.1% CPU, main thread waiting in its run loop.
  The original nightly was using 100.9% with repeated source authority scans.
- No private original temporary directories remained after image dismissal.
- Memory needs cautious interpretation: RSS at 5/10/15/20-cycle checkpoints
  was approximately 745/772/759/839 MiB; it was not strictly monotonic, but this
  does not establish a stable leak-free plateau. The full fixture plus native
  accessibility inspection differs from the user's original library.
  `leaks` could not inspect writable memory of the signed process completely;
  its 175 KiB report is not a clean leak certificate. Do not claim otherwise.
- Native automation includes accessibility settling and is not a valid
  sub-100 ms click-latency measurement. CPU samples and operation-count tests
  prove removal of the reported scan loop, not every frame-time target above.
- Remaining limits: no physical iOS-device runtime qualification, no all-screen
  Instruments trace, and no claim that every performance candidate is resolved.

Evidence is retained locally in `/tmp/mold-performance-uat`, the
`/tmp/mold-performance-*.log` files and the original
`/tmp/mold-native-performance-baseline.sample.txt`.
