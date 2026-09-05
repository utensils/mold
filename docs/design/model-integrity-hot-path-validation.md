# Download-only integrity validation

Branch: `fix/model-integrity-hot-path`. The user authorized opening one PR and
will monitor CI. No deployment, production restart, or production model/receipt
modification was performed.

## Behavior verified

- Core installed resolution trusts complete regular files, including legacy
  files without receipts, without reading their bodies. Replacement changes the
  local identity; later receipts do not change an already chosen identity.
- Server preparation and dependency availability use installed identities and
  completeness. The execution-equivalence schema and hash domain are v4.
- Both H3 preparation and frozen reopen use installed artifact resolution;
  expected pins and observed digests are recorded separately.
- ONNX and PuLID installed loads preserve bounded private buffers, parsing and
  descriptor fences. Embedding caches use the identities of all actual assets.
- New manifest, recipe and dependency acquisitions verify private staged bytes
  before publication. The private HF acquisition cache is excluded from runtime
  discovery; verified files are also published into the existing HF cache layout
  for cross-layout companion reuse. Derived output pins are checked before rename.
- Explicit `mold info MODEL --verify` still performs a fresh full scan.

## Artifact-resolution benchmark

Measured 2026-09-05 on `hal9000`, x86_64 Linux. The running service was active,
with zero restarts and an empty queue. Its revision was
`10ebd23a1a28bd4e286c2d64a7d59d889564b02d`, version 0.27.0. The candidate
benchmark was compiled independently under `/tmp/mold-integrity-validation.xoeg9w`;
it did not replace or interact with the running inference service. The configured
receipt store remained `/var/lib/mold-artifact-attestations-v1`, mode 0700, owner
`mold:mold`. The benchmark ran as the development user and did not write receipts.

Command shape:

```sh
nix develop --offline -c cargo run --release -p mold-ai-core \
  --example installed_artifact_benchmark -- MODEL_A MODEL_B
```

| Measurement | Result |
| --- | ---: |
| FLUX.2 Klein Q4 artifact | 2,604,311,104 bytes |
| LTX-2 19B FP8 artifact | 27,078,716,018 bytes |
| Explicit full-hash baseline, both files | 19,466.566 ms |
| First installed-identity resolution, both files | 239.774 us |
| 200 alternating installed-identity resolutions | 1,810.288 us |
| Runtime integrity hash attempts / bytes | **0 / 0** |

These are artifact-resolution measurements, not end-to-end generation timings.
The resolver was cold in the process; filesystem caches had been warmed by the
explicit baseline scan. No inference, queue acknowledgment, GPU idle interval,
or time-to-first-step improvement is inferred from these numbers. The benchmark
is checked in so an authorized isolated candidate server/rollout can repeat it
on the target storage and collect full generation timings separately.

A local debug run on sparse 256 MiB and 128 MiB fixtures also recorded zero
runtime hash attempts and bytes across first resolution and 200 alternations.
It is a deterministic regression aid, not a representative speed comparison.

## Checks

- Core library suite: 1,725 passed. Linux download-focused suite: 107 passed.
- Server execution-planning module: 79 passed, including isolated child-process
  zero-hash preparation and receipt/cache stability.
- Both catalog arrival/companion tests passed when rerun individually.
- PuLID derived-artifact module: 35 passed, 4 heavyweight fixture tests ignored.
- Documentation build, formatting and reference checks passed. H3 release
  contract passed after updating its runtime-resolver marker. Linux-specific
  contracts were rerun on Linux: CUDA distribution/qualification, matrix
  aggregation/concurrency/family sizing/source inputs/retries, Wan regression,
  LTX-2.5 CUDA verification and local multi-GPU qualification passed.
- Server/PuLID feature compilation: passed before the final acquisition refinements.
- H3/Metal feature compilation: passed before the final acquisition refinements.
- Identity module with PuLID enabled: 139 passed, 6 heavyweight fixture tests ignored.
- Clippy for core/server/inference, all targets with PuLID enabled: passed with warnings denied.
- Server dependency module: 40 passed, including legacy HF snapshot reuse.
- H3/Metal installed-artifact regression: 1 passed on the final source, covering
  no observed digest, replacement fencing and truncation.
- Rust formatting and whitespace checks: passed.
- The broad local server run was not clean. It recorded 24 failures before
  interruption during an unrelated 10,000-entry gallery-authority test. One
  failure was the new HF fixture comparing a canonical path with macOS’s `/var`
  alias; its expectation was corrected. Two catalog failures passed in isolation.
  Other failures involved queue-media/lifecycle/runtime tests outside the changed
  subsystem; the chain-prefix test reproduces `queue-media master key is missing
  while stored media exists` even with a fresh test home. This report does not
  claim a passing full local server suite or infer that every failure is pre-existing.

## Remaining hash inventory

| Production entry point | Purpose |
| --- | --- |
| Core `verify_file_integrity` / `verify_pinned_file` / staged recipe digest | New acquisition only |
| Core `compute_sha256` / CLI `info --verify` | Explicit verification and benchmark baseline |
| ONNX `load_onnx_model` | Explicit probe/verification; runtime uses `load_installed_onnx_model` |
| PuLID explicit `open_authenticated` / source-copy verification | Explicit conversion/verification helpers; installed paths use separate policy |
| PuLID `write_atomically_checked` | Verify newly derived output before publication |
| H3 `hash_exact_file` | Explicit artifact qualification; both runtime doors use installed resolution |
| H3 task/support/authorization record hashes | Small evidence documents, not model-weight integrity |
| Request-media, gallery, database, updater and configuration hashes | Independent integrity/identity contracts; unchanged |

## Review and rollout

The independent implementation review found acquisition publication races,
receipt binding, repair/reuse distinctions and HF-cache compatibility issues.
Fixes and their re-review disposition are tracked in
`model-integrity-hot-path-implementation-review.md`.

Same-size corruption is intentionally not detected by routine queue admission.
Use explicit verification or repair when corruption is suspected. Necessary
weight-loading and bounded header reads remain. A production rollout should
record actual queue/preparation/load/first-step timings for repeat, A-B-A,
concurrent and restarted jobs; this branch does not deploy itself.

The automatic architecture-graph hook found hundreds of changes since its old
baseline and required a full `/understand --full` rebuild. Following its
FULL_UPDATE stop rule, this task did not rewrite that unrelated graph.

## Greptile follow-up validation

The marker/ref publication regressions both failed before the fix. Afterward,
the download module passed 109 tests and the identity module passed 139 tests
(with 6 heavyweight fixture tests ignored). ONNX regressions now assert explicit
absence/presence of an observed digest. The parity integration test uses the new
explicitly verifying constructor; its external-asset cases remain opt-in.

The parity integration suite passed 10 tests, with 4 external-asset cases ignored.

The Rust CI run at `92c6aff1` exposed two old execution-equivalence integration
fixtures that expected planning to hash newly written, unverified files. Those
fixtures now explicitly establish verified receipts before planning, retaining
the original equal-content/different-runtime-layout assertions. This changes
fixture setup rather than restoring a runtime integrity scan.

Clippy passed for core/inference with PuLID, all targets and warnings denied.
Receipt-specific integration fixtures run in isolated child processes with
private writable attestation stores on Unix, where durable receipts are
supported. A separate portable regression covers distinct local identities for
unverified equal-byte files.

The final execution-equivalence integration suite passed all 14 tests.
