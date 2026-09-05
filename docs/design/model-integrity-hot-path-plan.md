# Download-only model verification implementation plan

Status: peer-reviewed and ready for implementation; review findings addressed.
No runtime changes yet.

Branch: `fix/model-integrity-hot-path`. Deliver all changes in one PR; do not
open the PR until requested. Use conventional commits. Baseline:
`6e5ac9e84d83889e6b391b8b00e1f90d5613d1c1`.

## Outcome and scope

Installed models must queue, prepare, switch, reload, and resume without a
full-file model integrity hash, including first use after restart and legacy
installations without digest attestations. Downloaded/repaired bytes are checked
once at acquisition; explicit `mold info MODEL --verify` remains an intentional
full scan. Derived artifacts are verified when produced, not every load.

This is a deliberate change from runtime content authentication to trusting the
local installation. Existence does not prove completeness: retain regular-file,
size/completion, containment, format/header, and descriptor/replacement checks.
Do not pretend metadata proves that unchanged-size bytes have not been corrupted.
Same-size corruption is detected by explicit verification or possibly loading,
not guaranteed to be detected by queue admission. Document this tradeoff plainly.

Do not alter model numerical behavior, activation/license eligibility, private
qualification authorization, request-media integrity, gallery/archive integrity,
database checks, or updater signature verification. Necessary weight reads for
loading and bounded header/config parsing are outside the zero-hash guarantee.
Keep small authorization/support-document validation distinct from bulk model
weight integrity; classify every H3 call site before changing its purpose.

## Findings and exact implementation seams

Paths below are repository-relative; verify current definitions before editing.

| Seam | Current behavior | Change |
| --- | --- | --- |
| `crates/mold-core/src/download.rs`: `pinned_file_digest_from_open_file` | Memory cache, persistent attestation, then full scan on a miss | Keep as acquisition/explicit verification machinery; remove runtime reachability |
| Same: `verify_file_integrity`, `is_already_placed`, `find_existing_placed_file` | Existing manifest files enter verification | Accept complete installed files without hashing; distinguish missing/partial files from existing complete files |
| Same: `fetch_recipe_inner`, per-file HF acquisition | Recipe existing-file fast path already avoids hashing; fresh files are hashed | Preserve fast path, unify completion semantics and verify fresh/resumed content before success |
| `crates/mold-server/src/variant_dependencies.rs`: `verify_cached_dependency`, `ensure_downloaded` | Admission verifies existing pinned dependencies; preview may call them unproven | Both use metadata-only installed resolution; only a real download verifies |
| `crates/mold-server/src/execution_plan.rs`: `warm_execution_equivalence_cache`, `artifact_facts_path_with_policy_and_progress` | Hashes concrete artifact set for content identity on misses | Resolve saved digest or local installation identity and probe bounded headers only |
| Same: `EquivalenceContentIdentity`, `fingerprint_path`, pre-CUDA validation | SHA identity and unknown/pending domains; ordinary Unix/Windows file fingerprint already metadata-only | Add explicit local identity domain; keep frozen plan/replacement fencing |
| `crates/mold-inference/src/minimax_h3/private_qualification.rs`: `hash_exact_file` and callers | Shared verifier authenticates bulk artifacts during qualification/admission | Separate explicit qualification from ordinary public runtime preparation; runtime consumes installation facts without bulk hashing |
| `crates/mold-inference/src/identity/onnx_graph.rs`: `AuthenticatedBytes::read_once` | SHA hashes every loaded byte buffer | Runtime read/decode without hashing; install/explicit checks retain verification |
| `crates/mold-inference/src/encoders/pickle_convert.rs`: `open_authenticated` | Process-only memo; cold loads hash derived artifacts | Verify at conversion/publication, accept complete existing derived outputs without cold-load hashing |
| `crates/mold-core/src/config.rs`: `file_is_complete` | Metadata and completion sidecar; legacy declared-size fallback | Preserve compatibility and extract/share the applicable cheap completeness rules |
| `nix/module.nix` | Private persistent digest store already configured | Retain it; absence must no longer cause generation to hash or block |

The current progress label `Verifying model files` is published even for cache
hits. Changing just this label or making the cache more durable is insufficient.

## 1. Define acquisition and runtime contracts first

- Introduce a core, typed metadata-only installed-artifact resolver. Inputs:
  path, expected size when known, completion requirements, optional known pin,
  and existing caller-specific path constraints. Outputs distinguish absent,
  incomplete, invalid path/format, and usable installed artifact.
- Keep verification in separately named APIs for newly acquired bytes and
  explicit checking. Avoid a boolean `skip_verify` propagated through generation:
  runtime should have no fallback that can hash at all.
- Preserve `--skip-verify` behavior for explicit pulls; never manufacture a
  verified digest for a skipped check. Distinguish completion from verification
  in any new receipt schema, preserving old marker compatibility.
- For unknown-size local files, use existing local configuration/catalog rules
  and bounded container validation. Do not require a marker introduced by this
  change, or accept an in-progress `.pulling` file merely because it exists.
- On missing/truncated files use current repair/admission errors and acquisition
  ownership. No automatic whole-file scan, deletion, or repair solely because a
  digest record is missing, unreadable, stale, or mismatches a new manifest pin.
- A persisted observed digest and the manifest's expected digest are different
  facts. Never turn the latter into a claim about bytes not actually checked.

## 2. Preserve execution identity without hashing

- Add `LocalInstallation` alongside existing SHA/pending/unknown identity
  variants. Build it from an opaque server/process scope plus normalized path
  and current platform file identity (size, inode/file ID, timestamps). It is
  stable for repeated preparation and A→B→A within that process, but explicitly
  does not promise cross-host byte equivalence. Hashing these small metadata
  structures is permitted; hashing artifact bodies is not.
- Read valid existing durable digest receipts through a non-hashing API. Use an
  observed SHA identity only while the receipt matches the current file. If a
  receipt is unavailable, fall back to local identity immediately.
- Freeze the chosen identity representation for each unchanged artifact within
  the process, shared across preparation, batching and worker validation. Receipt
  appearance/disappearance must not flip an existing local identity into SHA or
  vice versa under an active lease. Maintain this identity registry separately
  from evictable header/fact caches so eviction cannot change the answer. A new
  file identity or process may choose again; explicit verification can update
  receipts without mutating already-frozen runtime identities. Test receipt
  creation/removal and fact-cache eviction between preparation and dispatch.
- Use a process-scoped fallback initially to avoid introducing persistent host
  identity storage just for this change. After restart, reprepare durable jobs
  and invalidate incompatible local timing/equivalence records. Verify this
  does not strand paused chains, restored jobs, or previously frozen plans.
- Inspect every consumer of `EquivalenceContentIdentity`: batch grouping,
  execution-environment fingerprints, timing/estimate buckets, cache lookup,
  serialized/provenance output, retry reconstruction, and worker validation.
  Do not reuse `Unknown` for a valid local file if that domain means refusal.
- Bump the affected equivalence fingerprint domain/version where necessary;
  ensure older durable records follow the existing reprepare/migration path.
- Worker dispatch still checks the frozen file identity. A replacement between
  preparation and dispatch invalidates/reprepares the plan through bounded
  existing retry handling; it never triggers a runtime SHA fallback.
- Keep global read-only preview behavior: no receipt writes, downloads, or body
  hashing. Cold installed dependencies must cease appearing as pending downloads
  merely because there is no attestation. Header probing remains off coordinator
  threads using the existing asynchronous preparation/cache boundaries.

## 3. Verify and record once when bytes are acquired

- Cover manifest pulls, catalog HF/Civitai recipes, dependency auto-downloads,
  resumed transfers, and derived conversion outputs. Inventory exact shared
  callers before editing so none is bypassed.
- Publish completion only after a fresh download's verification succeeds. Keep
  existing partial-file cleanup/cancellation ownership and mismatch rejection.
- Finalize placement and file metadata before writing installation receipts.
  Hardlinks/copies/legacy migrations must not force verification of already
  installed files. A copy may carry completion facts, but cannot invent a
  content proof; fall back to local identity when exact receipt identity differs.
- Retain existing on-disk markers and private attestation store. A failure to
  persist an optional digest record must not make usable installed weights
  unqueueable or cause a later hidden scan. Required completion publication
  failures must still be surfaced as installation failures where applicable.
- Streaming SHA is optional within this PR only if it is straightforward and
  resume semantics are proven. The required outcome is at most one verification
  pass per completed acquisition and zero runtime passes. Do not expand scope
  into replacing HF transfer infrastructure just to stream hashes.
- Explicit `info --verify` must bypass cached digest answers and read bytes;
  preserve its current reporting and non-destructive behavior.

## 4. Switch all runtime paths to installed facts

- Ordinary queued work: replace warm-cache hashing with metadata identity and
  bounded format probing. Retain path deduplication, concurrency limits and
  scheduler responsiveness. Revisit cancellation/flight code only as needed.
- Dependency preparation: accept complete cached files regardless of digest
  cache warmth; fresh downloads still verify before becoming available.
- Trace forced-local CLI, TUI local mode, desktop native/local execution,
  remote server, batch children, authored/ephemeral chains, expansion and upscale
  utility acquisition. Shared APIs should enforce the rule; add targeted tests
  at distinct doors, not duplicate implementations per surface.
- H3: trace public reviewed execution through its shared qualification helpers.
  Keep private explicit qualification capable of computing actual digest
  evidence; do not populate SHA fields with expected pins for legacy files.
  Runtime records must explicitly carry observed-digest versus local-identity
  state, with compatibility changes wherever a SHA-only record currently gates
  activation. Preserve reviewed ID gates, support-document authorization and
  size/header contracts. No new license/authorization prompts.
- ONNX and derived PuLID outputs: keep retained descriptors, read bounds, parser
  checks, expected lengths and replacement fences while removing load-time SHA.
  Rename types/docs that would otherwise claim cryptographic authentication of
  runtime bytes. Verify derived outputs at creation. Do not retain hundreds of
  MB indefinitely as a substitute for eliminating hashes.
- Migrate `identity/extraction.rs::pinned_asset_digests` and its pre-extraction
  cache keys explicitly. Today they assume actual loaded bytes necessarily equal
  compiled pins. Build keys from the current identities of adapter, vision,
  detector, recognizer and parser before consulting caches; preserve family,
  photograph digest, pipeline version and parent single-flight/pinning. Update
  `IdentityAssetDigests`/frozen embedding provenance compatibly so a local token
  never appears as a SHA-256 digest. Test replacement of each asset invalidates
  both conditional and unconditional caches, while siblings still share exactly
  one extraction and unrelated families cannot collide.
- H3 has a second runtime qualification on frozen reopen in
  `minimax_h3/private_server.rs`, as well as the preparation call. Both must use
  the same installed identity authority and compare it against the frozen fence;
  fixing the preparation pass alone leaves switching/reloading able to hash.
- PuLID's `ensure_derived` also converts on first use when only the source is
  installed. Move eager conversion into acquisition where supported, but retain
  legacy source-only compatibility: runtime staging of an existing source must
  copy/read through the retained descriptor without hashing that source. Verify
  the newly produced derived output once as artifact acquisition, then publish
  it atomically. That new-output verification is an explicit acquisition phase,
  not hidden installed-model validation; measure it separately. Do not let
  `stage_private_copy` silently retain its old source digest pass. Update the
  source/derived policy and error contracts rather than supplying expected pins
  as if an observed source digest had been computed.
- Search all production SHA/file-hash APIs again. Classify remaining calls as
  acquisition, explicit verification/qualification, request media, small metadata
  or documents, or unrelated integrity. No unclassified model-runtime call may
  remain. Include feature-gated H3 and PuLID code in this inventory.

## 5. Observability and user-facing behavior

- Replace routine `Verifying model files` with `Resolving installed model` or
  the relevant loading phase. Use verification wording/byte percentages only
  when a real acquisition or explicit verification is hashing bytes.
- Add structured counters/timing at actual full-body hash implementations:
  invocation count, bytes processed, duration, purpose. Count attempts and
  partial reads, not only successful completed hashes. Include ONNX/derived
  implementations until removed. Avoid public filesystem paths in API metrics.
- Keep request-media hashing separate in metrics so conditioned renders do not
  falsely fail the model-only guarantee. Header reads and necessary loader reads
  get separate accounting when benchmarking; zero disk I/O is not the target.
- Expose sufficient existing logging/metrics diagnostics to explain acquisition
  work and cache/receipt failures without per-file user notifications.

## 6. TDD and validation matrix

Write failing tests before implementation. Prefer exported contracts and
observable byte counts over string-only source assertions or elapsed-time tests.

| Scenario | Required evidence |
| --- | --- |
| Fresh download with a matching pin | One acquisition verification; usable only after completion |
| Fresh/resumed download mismatch or interruption | Reject mismatch; partial never becomes installed; retry/cancel ownership preserved |
| Fresh unpinned / skip-verify acquisition | Completion works; no false expected/observed digest claims |
| Existing complete manifest and catalog model | Zero model integrity bytes on queue/repair no-op |
| Legacy/manual install without receipt | First queue and first load do not hash; usable local identity |
| Private digest store missing, read-only, wrong owner, or shared home | Runtime remains metadata-only; no hidden scan or new permission refusal |
| Same model repeated; A→B→A; simultaneous jobs | Zero integrity bytes on all existing artifacts including shared encoders/LoRAs |
| Real child-process restart | No inherited memory memo; installed file still queues without hashing |
| Truncated file, active partial, dangling link, invalid header | Existing completeness/path/format contract still rejects; no SHA fallback |
| File replaced after admission | Frozen identity invalidates; bounded reprepare/error, no repeated hashing loop |
| Receipt appears/disappears without artifact mutation | Stable identity across batching, fact-cache eviction and active leases; no local/SHA transition loop |
| Two hosts/local copies without receipts | Never falsely claim shared SHA equivalence or cross-host cache identity |
| Batch/retry/chain replay after identity version change | Work resumes/reprepares correctly; grouping and lifecycle remain valid |
| ONNX/PuLID cold load and reload | Necessary model read retained, integrity hash bytes zero |
| Legacy PuLID source present, derived output absent | Existing source is not hashed; new output is verified once during explicit derivation/acquisition and reused thereafter |
| H3 public runtime vs explicit qualification | Runtime zero bulk model hashes; explicit evidence remains truthful |
| Explicit `info --verify` | Reads current bytes and detects same-size corruption, despite warm receipts |

Run owning core/server/inference tests under `nix develop`; use actual package
names (`mold-ai-core`, `mold-ai-server`, `mold-ai-inference`). Inspect
`scripts/ci-local.sh --list` for relevant suites; run rust/contracts and affected
web/docs/Nix gates. Compile feature-gated H3/PuLID in supported configurations;
workspace defaults alone cannot validate them. Keep full suite requirements
for the eventual single PR. Plan-only work needs Markdown/diff validation only.

## 7. Server benchmark and rollout

Before runtime implementation is considered complete, identify the affected
server and capture revision, service settings, receipt availability and existing
phase timings with read-only inspection. Do not infer live state from this
checkout. Do not deploy or restart production solely to gather a baseline.

Use an isolated candidate server or scheduled authorized production validation.
Select a large video model and another model sharing an encoder, plus a distinct
model; include H3/PuLID when supported. Hold prompt/seed/shape constant and capture:

- queue acknowledgment, preparation, model-load and time-to-first-step timings;
- model integrity invocation/byte counts, process disk reads, CPU and GPU idle
  intervals, host memory, and output success;
- same-model repeat, A→B→A, concurrent submission, service restart, and legacy
  receipt absence in an isolated copy (never delete live receipts for a test).

Acceptance: exactly zero bulk model integrity hashing during existing-model
runtime paths, including first use after restart. Report timing improvement from
measured runs; do not promise a percentage or claim eliminated model-load I/O.
Fresh acquisition and explicit verification remain visible and correct.

Rollback: revert the single code change/PR as appropriate; preserve backward
readable completion markers and existing digest records. New optional receipt
fields/domains must not corrupt older binaries' installation inventory. Ensure
older binaries reprepare or reject unsupported plan versions clearly.

## 8. Documentation and delivery checklist

- [ ] Contract tests fail for current runtime hashing.
- [ ] Core acquisition/runtime APIs separated.
- [ ] Local identity and replay/equivalence semantics implemented and tested.
- [ ] Queue/dependencies, public H3, ONNX and derived loaders migrated.
- [ ] Runtime hashing inventory has no unclassified model-body callers.
- [ ] Observability and truthful progress labels updated across shared clients.
- [ ] Docs updated: relevant model/storage and queue rules, H3/identity rules,
  README, website configuration/model guides, CLI skill renderer and owning app
  docs where behavior is described; no divergent generated skill copies.
- [ ] One `changelog.d/model-integrity-hot-path.md` fragment; no manual version
  bumps or edits to CHANGELOG's Unreleased section.
- [ ] Required checks and server benchmark evidence recorded.
- [ ] Independent final implementation review; address valid findings.
- [ ] All work stays on `fix/model-integrity-hot-path`; one eventual PR.
- [ ] Do not open PR, merge, deploy or restart production until requested.

Peer review of this plan is recorded separately in
`docs/design/model-integrity-hot-path-plan-review.md`; amend this plan for valid
findings before implementation.
