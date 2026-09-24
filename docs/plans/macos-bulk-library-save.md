# Native macOS bulk Library save repair

## Findings

1. `LibraryGrid.cell` calls `targets(for:)`, which filters all entries for every selected tile while rebuilding each menu. Select All therefore induces quadratic work. Precompute selection once while keeping keyboard and range semantics.
2. Save is strictly serial and downloads every image to `Data`, then builds another framed `Data` body. The destination also publishes one gallery event per import, causing repeated row rebuilds and background render notifications. Bounded concurrency requires a memory limit or streaming body.
3. The native import descriptor currently takes JSON metadata from the listing. The server compares that descriptor to metadata embedded in the exact file and rejects conflicts. The screenshot shows those conflicts, then an unbounded filename-by-filename error dump in a modal alert.
4. A partial save of 500 out of 1,522 is visible. The loop has no fixed cap, so the count is partial success; diagnose failures and make retry skip already saved items without duplicating them.
5. Mirrored prints currently lose collection membership because remote collection IDs are host-local. Map remote collections to local collections by server slug/name and file the imported local print under local IDs.
6. No screenshot-denial code was found in the native app. Verify whether the restriction comes from macOS Screen Recording/managed-device policy or a Mold window setting before changing app behavior.

## Implementation

- Keep one canonical import path; resolve immutable embedded metadata from the downloaded file before constructing the import descriptor, and add fixtures for the reported conflict.
- Add bounded concurrent transfer with cancellation and a compact progress presentation. Avoid copying each whole file into a second body, repeated full gallery re-list and grid rebuild during the batch. Skip already-local identical items on retry.
- Mark import events distinctly from generation completion so badges and notifications count renders only. Preserve normal gallery updates.
- Transfer each print's source-host collection membership through a local host mapping by slug. Create missing local collections once, file the returned import filename only on the local host, and make retries idempotent without removing existing memberships.
- Replace the error alert with a readable summary and details affordance; never show thousands of raw errors in one alert.
- Profile selection on a large fixture; reduce per-cell focus/menu recomputation, and verify Command-A plus range selection remains responsive.
- Check screenshot behavior on a signed app and report any external macOS permission boundary accurately.

## Verification and delivery

- Add focused Swift client/app and Rust server tests for metadata, collection mapping, notifications, partial retries, and selection behavior.
- Run native macOS tests and relevant Rust contracts; perform rendered native UAT with a large fixture if available.
- Refresh Understand Anything graphs, run Claude review on the final diff, address findings, open one PR, wait for exact-head CI, merge and sync the branch.

## Review record

- A subagent reviewed the plan before implementation and confirmed the native macOS path, the metadata conflict, the absence of a fixed 500-item cap, and the host-local collection ID boundary.
- Claude reviewed the implementation before the PR. The follow-up addressed cancellation, transport error mapping, collection lookup failure isolation, and the desktop notification ledger. The screenshot denial did not reproduce in an isolated app capture or a system capture on this Mac.
