# Independent review: download-only model verification plan

Reviewed: 2026-09-05. Source baseline:
`6e5ac9e84d83889e6b391b8b00e1f90d5613d1c1`.

Disposition: **ready for implementation after the revisions below**. The current
`model-integrity-hot-path-plan.md` incorporates them. No unresolved blocking
plan findings remain. This is a source-backed plan review, not implementation
approval or measured server-performance evidence.

The reviewer inspected the proposed plan and the core download verifier,
server artifact-fact/equivalence cache and dependency admission, H3 qualification
and opened support evidence, ONNX loading, PuLID conversion/extraction, and
explicit CLI verification. Review edits are confined to this document; no code,
branch, server, or PR changes were made by the reviewer.

## Findings and disposition

### P1: source-only legacy PuLID installs still hash during first use — addressed

`pickle_convert.rs:774-791` converts when the derived file is absent.
`convert_pickle` calls `stage_private_copy` at line 678, which hashes the
already-installed source while staging it. Removing `open_authenticated`'s
load-time hash alone would therefore leave a first-use source scan.

The revised plan explicitly moves eager derivation to acquisition where
supported, stages legacy installed sources without hashing, and verifies a newly
produced output once as acquisition. The validation matrix now includes a source
present/derived output absent case. This preserves the distinction between
trusting installed bytes and checking newly created artifacts.

### P2: receipt changes can change equivalence without file replacement — addressed

`execution_plan.rs:4378` onward keys artifact facts by path and metadata. A new
runtime policy that chooses SHA when a receipt exists and local identity
otherwise introduces mutable state outside that cache key. Explicit verification,
receipt-store availability, or fact-cache eviction could then change the
representation between preparation and dispatch while the file stays unchanged.

The revised plan freezes the chosen representation for an unchanged artifact
within the process in an identity registry independent of evictable fact caches.
It adds tests for receipt creation/removal and eviction under active leases.
This is sufficient at plan level; implementation review must confirm bounded
registry lifecycle and consistent use by all identity consumers.

### Identity embedding cache/provenance dependency — incorporated and verified

The author identified this during review, and independent source inspection
confirmed it. `identity/extraction.rs:296-325` derives pre-load cache identities
from manifest pins under the invariant that loaders refuse different bytes.
That invariant changes with download-only verification. Leaving these keys
unchanged could reuse stale embeddings or record an expected digest as observed.

Section 4 now explicitly migrates `pinned_asset_digests`,
`IdentityAssetDigests`, frozen embedding provenance, and conditional/unconditional
cache keys to actual installed identities. It requires per-asset replacement
tests and preservation of family separation and parent single-flight behavior.

### H3 preparation and frozen reopen — incorporated

The plan now names both runtime qualification routes in `private_server.rs`,
rather than relying on changing only initial preparation. Its policy separates
bulk artifact integrity from small task/support/authorization evidence, and
keeps explicit qualification capable of producing truthful observed digests.
Source inspection confirms that these distinctions matter: qualification records
currently store actual SHA values, while opened task/support evidence has
separate revalidation duties.

## Overall assessment and implementation gates

The user authorized trusting complete installed files without integrity scans.
Removing same-size corruption detection from ordinary runtime admission is
therefore an intentional policy change, not an unresolved review objection.
The plan correctly retains acquisition verification, explicit verification,
completeness and bounded format checks, replacement fencing, and truthful
identity/provenance domains.

Implementation should follow the documented failing-test-first sequence. The
critical evidence is measured model-body hash bytes, including a real child
process restart, legacy installs, feature-gated H3/PuLID routes, and frozen-job
replay. Elapsed time or progress-label changes alone cannot establish success.
Fresh derivation/acquisition must be reported separately from installed-model
resolution. Source audits must also classify remaining direct SHA calls so
instrumenting only the shared verifier cannot produce a false zero.

All implementation remains on `fix/model-integrity-hot-path` for one eventual
PR. This review does not authorize opening that PR or deploying the changes.
