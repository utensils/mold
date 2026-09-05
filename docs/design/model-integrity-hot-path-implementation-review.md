# Model integrity hot-path implementation review

Independent review of baseline `6e5ac9e84d83889e6b391b8b00e1f90d5613d1c1`
through the candidate working tree on `fix/model-integrity-hot-path`.
Reviewed the implementation plan and its earlier peer review. This is a source
review; the implementing agent owns ongoing test runs and host benchmarks.

Final source-review disposition: all actionable findings below are addressed in
`046fdd47` plus the reviewed follow-up working tree. No remaining source-level
blocking findings were identified. The findings and intermediate dispositions
are retained as review history. Full server, H3/identity and clippy validation
was still running when this final source review completed.

## Findings

### P1: new acquisitions become reusable before verification finishes

`crates/mold-core/src/download.rs:2338-2339` still places the HF file at its
canonical model path before checking its checksum. The new
`is_already_placed`/`resolve_installed_file` path accepts that full-length file
without participating in the writer's HF flight. A simultaneous pull of a model
sharing that encoder can therefore accept it while the first pull is still
hashing, including bytes that the first pull subsequently rejects and removes.
The per-model `.pulling` marker cannot fence another model's shared artifact.
The new dependency registry check covers only acquisitions owned by that
registry, not manifest pulls or other processes.

Recipe acquisition has the same publication seam: it writes to the final path,
then hashes it, while `fetch_recipe_inner` now accepts a same-length file without
a completion marker. A second fetch can skip it and remove the shared `.pulling`
marker before the first finishes. A process exit after placement and before
verification also leaves an indistinguishable apparent legacy installation.

Verify newly acquired bytes in private staging before atomic publication, or add
durable per-artifact acquisition fencing consumed by every installed resolver.
Keep the authorized legacy-installation policy. Add a deterministic test that
pauses acquisition before verification completes and attempts reuse from a
second consumer, plus interrupted-acquisition/restart coverage. A checksum
mismatch must never publish a usable new installation.

### P2: the key end-to-end zero-hash and identity stability tests are absent

The added child-process test in `download.rs:4386` exercises the core identity
resolver only. The server changes update existing expectations, but do not assert
actual model-body hash counters for preparation, repeated queueing, A-to-B-to-A
switches, or durable replay. The receipt stability test calls the digest helper;
it does not explicitly establish and remove a durable receipt across server
fact-cache eviction and dispatch. Those are the regressions singled out in the
accepted plan and its peer review.

Add counter-based tests through the real server preparation boundary, exercising
receipt availability changes, fact-cache eviction and frozen replacement checks.
Include a replay/reprepare test proving process-local equivalence identities do
not strand durable jobs after restart. Keep the existing focused core, ONNX,
PuLID and H3 tests, but do not treat their narrower coverage as end-to-end proof.

### P3: source contracts and pull progress still describe removed verification

`encoders/pickle_convert.rs:39-47` says every reused artifact is hashed against a
compiled pin. `identity/onnx_graph.rs` still describes a runtime load as hashing
its retained buffer, and `identity/mod.rs` says no unverified extractor variant
exists. The manifest pre-scan in `download.rs` still reports `Verifying file`
while performing metadata-only resolution. These statements now contradict both
the implementation and the newly documented user policy.

Update these owning contracts and progress strings to distinguish installed
loading from explicit verification and new-output acquisition. Preserve accurate
documentation of retained descriptors, private buffers, sizes and format checks.

## Additional audit observations

- Ordinary server artifact equivalence uses the non-hashing installed identity
  API; the core registry freezes SHA versus local representation and remains
  bounded without evicting active representations.
- Both H3 initial preparation and frozen reopen call the installed resolver.
  Bulk model identities are distinct from expected pins; small authorization and
  support evidence still has separate verification.
- ONNX runtime callers and derived PuLID loaders have explicit installed paths;
  new derived output digests are checked before atomic publication.
- Windows core identity includes file ID and ChangeTime. Windows execution-plan
  metadata already includes those fields.
- The exotic non-Unix/non-Windows `fingerprint_path` branch still hashes artifact
  bodies. It is outside the presently supported server platforms, but should be
  removed or explicitly excluded from the universal no-runtime-hashing claim.
- Existing same-size corruption being accepted is intentional and authorized;
  this review does not recommend restoring runtime checksums.

## Focused re-review of staging fixes

The candidate now stages manifest, recipe and pinned dependency acquisitions
before publishing. This closes the original direct visibility of unverified
new bytes. Server tests now exercise isolated-child preparation counters and
receipt creation/removal across fact-cache eviction; the inaccurate progress
and loader documentation identified above have also been corrected.

Three follow-up findings remain in the reviewed staging candidate:

1. **P1: publication can attest the other concurrent writer's file.**
   `AcquisitionStage::publish` renames its staged file, then opens the destination
   and associates its own digest with that descriptor's identity. Another
   publisher can replace the path between those operations. Writer A then stores
   digest A against file B's identity, creating a false observed-SHA receipt.
   Recipe writers and sync-versus-manifest paths do not share a publication
   lock. Retain the staged descriptor, bind the receipt to its post-rename
   identity, compare destination identity before publishing metadata, and add a
   forced-interleaving test. An optional receipt failure should safely fall back
   to a local identity, never bind a digest to another file.

2. **P1: pinned dependency repair can return the existing truncated file.**
   `download_single_file_sync_with_adapter` now returns any nonempty existing
   clean file via `installed_file_is_complete(clean_path, None)`. The server
   correctly rejects a truncated cached dependency using `expected_bytes`, then
   enters this downloader, which returns the same truncated file before checking
   its pin. Thread the expected completeness contract into this boundary and
   test an Admission repair through the real path. The skip must be consistent
   with the caller's earlier rejection.

3. **P2: the new private HF cache is disconnected from installed cache reuse.**
   New pulls write `.hf-acquisition-cache`, while companion resolution in
   `Config::discovered_manifest_paths` still reads only `.hf-cache` to reuse a
   file installed under another canonical layout. The documented Gemma encoder
   shared between an LTX-2 install and the `ltx2-te` catalog companion therefore
   loses that reuse. `required_download_bytes_in` also inspects the old cache,
   so space preflight disagrees with actual acquisition. The new
   `target_subdir=None` output under `shared/acquired` is absent from cache lookup
   APIs as well. Publish a verified installed index/cache or resolve known clean
   paths; keep acquisition-private bytes hidden. Cover cross-layout companion
   reuse and the no-subdirectory API round trip.

## Re-review of core commit `220dec41`

The retained staged descriptor fixes the false receipt identity issue. Removing
the unconditional sync reuse shortcut prevents the previously reported truncated
dependency repair. Verified publication into the discoverable HF cache restores
new-install cross-layout discovery, and the added cache publication regression
exercises that path. The implementing agent reports 1,723 passing core tests;
this reviewer has inspected the fixes rather than independently rerunning them.

Two follow-ups remain:

- **P1: ordinary direct sync callers now rehash installed models.**
  `ltx2/lora.rs::resolve_camera_control_preset_path` calls
  `download_single_file_sync` on every request without a separate cache check.
  The sync implementation now always stages a fresh hardlink and calls
  `pinned_file_digest`, so this runtime path hashes the installed camera LoRA
  again (the staged path and link-induced metadata identity change each time).
  Preserve a metadata-only reuse path for ordinary sync calls before network
  access, while making repair use its expected completeness contract. Add a
  direct repeat-call regression proving no network or checksum work.
- **P2: old-cache-only reuse and space accounting still disagree.**
  Preflight accepts a complete old `.hf-cache` blob as zero required bytes, but
  actual acquisition only downloads from `.hf-acquisition-cache`. An older
  installation with no canonical file can therefore download the existing blob
  again despite passing a zero-space preflight. Reuse/migrate the complete
  published cache entry before network acquisition, or count the actual missing
  private-cache bytes. Cover an old-cache-only fixture, distinct from the new
  verified-cache publication test.

## Re-review of commit `046fdd47`

`SingleFileAcquisition::ReuseInstalled` now handles direct sync callers before
HF API creation, and `Repair` retains explicit acquisition behavior after the
server rejects incomplete bytes. Manifest resolution now migrates published
legacy-cache files without hashing. These address the two follow-ups above;
direct repeated-sync and old-cache migration regression tests were added. The
implementing agent reports the complete core suite passing with 1,725 tests.

One legacy-cache ingress remains inconsistent:

- **P2: server dependency admission treats ordinary HF snapshot links as a
  missing installation.** `variant_dependencies::ensure_downloaded` passes the
  raw HF cache result to `installed_file_is_complete`. Standard older HF caches
  use snapshot symlinks to their blob files, so the no-follow regular-file opener
  rejects this path. Admission then invokes `Repair` and hashes or redownloads
  already-complete local weights. The new manifest and direct-sync paths resolve
  these known cache links before checking completeness, but this server ingress
  does not. Apply the same known-cache normalization here, preserving ordinary
  clean-path constraints, and exercise both read-only preview and Admission with
  an actual snapshot-symlink fixture.

## Final focused re-review

The server now resolves cache-origin HF snapshot links to regular blobs before
its metadata completeness check. Canonical clean paths retain their no-follow
policy. The added Unix regression uses an actual snapshot-to-blob symlink,
checks both ExistingOnly and Admission, and installs a downloader that panics if
either path attempts acquisition. This addresses the last legacy-cache ingress
finding. The wrong-size local test fixture, ONNX error context and H3 source
contract guard were also corrected without restoring runtime hashing.

All prior actionable source findings are addressed. The implementing agent
reports core 1,725 tests, PuLID derived 35 tests, and the isolated-child server
zero-hash preparation regression passing. Full server, final H3/identity and
clippy runs were still in progress; this review does not label unfinished checks
green or claim production deployment. The reported host benchmark is an isolated
artifact-resolution experiment, not an end-to-end GPU model-switch benchmark.
No implementation changes, commits, pushes, deployment or PR creation were
performed by this reviewer.
