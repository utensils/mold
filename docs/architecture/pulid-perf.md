# PuLID performance: GPU face extraction, embedding cache, and qualification

Issue [#1227](https://github.com/utensils/mold/issues/1227). This is the
research-and-design record for the perf half of face-identity conditioning:
whether extraction should move off `candle-onnx`/CPU, a cross-request
identity-embedding cache, EVA-CLIP residency under a GPU path, and the
benchmark protocol that qualifies whichever of these ships.

This began as a **decision record, written before code**, exactly as
[#1222](https://github.com/utensils/mold/issues/1222)'s Step 0 was. Phase 1 has
since shipped, so the sections below now carry what was measured beside what was
predicted — and in one case (§1's re-materialization hypothesis) the measurement
**falsified** the prediction. Those corrections are inline and marked
`MEASURED`, never quietly edited over the original reasoning: a decision record
that reads as if it had been right all along teaches nobody why it was wrong.
For what already shipped before this issue, see `docs/architecture/pulid-face-extraction.md`
(Step 0: the `candle-onnx` runtime decision, the measured budget, the stage
table), `docs/architecture/pulid.md` (encoder lifecycle, residency, the
extraction call site), and `docs/architecture/pulid-uat.md` (real-hardware
render numbers) for what already shipped. Code cited below:
`crates/mold-inference/src/identity/`, `crates/mold-inference/src/flux/identity.rs`,
`crates/mold-server/src/identity_extraction.rs`,
`crates/mold-server/src/identity_dependencies.rs`,
`crates/mold-core/src/identity.rs`, `crates/mold-core/src/pulid_assets.rs`,
`crates/mold-inference/src/bin/pulid_face_probe.rs`,
`crates/mold-scheduler/src/estimates.rs`, `crates/mold-inference/src/progress.rs`.

---

## 0. What changed since Step 0, and the gap Step 0 left open

Two facts qualify everything below.

**The candle-fork Resize PR Step 0 deferred has already landed upstream, but
mold does not consume it.** `docs/architecture/pulid-face-extraction.md`
records the fix as "a separate candle-fork PR, deliberately not folded into
this work" (`candle-onnx/src/eval.rs`, the `Resize` arm, lines 2291-2301). It
has since merged into `utensils/candle`'s `fix/mold-compat-0.11` branch:
commit `87f3adb3` ("fix(onnx): treat zero-element optional Resize inputs as
absent"), merged via PR #8 (`0655b4c1`), rewriting exactly the
`optional_input` presence check the mold doc predicted. **This does not change
anything below** — it fixes a correctness bug (an exporter's empty-tensor
idiom for an unset `Resize` input), not a performance one, and mold's own
`identity/onnx_graph.rs::normalize_empty_optional_resize_inputs` workaround
stays load-bearing regardless, because `Cargo.toml` does not patch
`candle-onnx` from the fork at all — it takes `candle-onnx = { version =
"0.11", optional = true }` straight from crates.io
(`crates/mold-inference/Cargo.toml:158`), unlike `candle-core`/`candle-nn`/
`candle-transformers`, which are `[patch.crates-io]`'d to the fork
(`Cargo.toml:59-60`). Noted here as bookkeeping only: a future housekeeping PR
could add `candle-onnx` to the patch table and drop the mold-side
normalization as redundant, but that is independent of this issue.

> **Superseded 2026-08-26 (#1399).** #1393 retired the `[patch.crates-io]`
> table for Candle entirely: every candle crate, `candle-onnx` included, is now
> a direct git dependency on one `utensils/candle` revision
> (`crates/mold-inference/Cargo.toml`). The prerequisite this section describes
> as unmet is therefore met, and whether
> `normalize_empty_optional_resize_inputs` is still load-bearing is now a
> question about that revision's `candle-onnx`, not about where the crate comes
> from. The rest of the section's reasoning is unchanged.

**The measured 2.0 s p95 budget covers less than half of what one identity
extraction actually does.** `pulid_face_probe bench` measures exactly
`ScrfdDetector::detect` + `ArcFaceRecognizer::embed_crop`
(`crates/mold-inference/src/bin/pulid_face_probe.rs`, the `run_bench` loop) —
the two `candle-onnx`-evaluated stages. The EVA02-CLIP vision tower and the
IDFormer are never invoked by that binary. But
`mold_inference::identity::extraction::extract_identity_embedding` — the ONE
production entry point (`crates/mold-server/src/identity_extraction.rs::
resolve_identity_embedding` calls it directly) — always runs SCRFD → ArcFace
→ EVA tower → IDFormer in sequence
(`crates/mold-inference/src/identity/extraction.rs:93-126`, `147-218`), and
the tower + IDFormer halves are unmeasured, ungated, ordinary
`candle-core`/`candle-nn` forward passes on `Device::Cpu`
(`extraction.rs:152`: `let device = Device::Cpu;`, hardcoded independently of
the ONNX constraint — see §1). EVA02-CLIP-L-14-336 is a 24-block, 1024-wide,
16-head ViT running 577 tokens through an MLP hidden width of 2730
(`docs/architecture/pulid.md`, "Vision tower"), in f32, single-threaded CPU —
a workload of the same order of magnitude as the `glintr100` ArcFace backbone
`pulid_face_probe` already measures at 222–1051 ms, plausibly more. **No
number for it exists anywhere in the repository.** Any perf-qualification
claim this issue produces is incomplete until this is measured — §4 makes
extending the bench harness to cover it the first concrete deliverable of the
implementation phase, ahead of deciding where to spend GPU/hand-port effort.

> **MEASURED (phase 1).** `pulid_face_probe bench --full` now measures it, and
> the suspicion was an understatement. On halcyon, one identity extraction is
> **2,840 ms** (p50) end to end, of which the EVA tower is **2,237 ms — 79%**
> and the face stack `pulid_face_probe` had been gating on is **360 ms — 13%**.
> The full table is in §4. Every number below that reasons about SCRFD and
> ArcFace is reasoning about an eighth of the problem, which is exactly what §0
> was written to find out and is what §5's phase-2 plan is aimed at.

---

## 1. Runtime decision: hand-ported candle vs. a cached ONNX evaluator

### The two options on the table

**A — hand-ported iResNet100 (`glintr100`) + SCRFD backbone in
`candle-core`/`candle-nn`**, loaded once via an ordinary `VarBuilder` and run
on whichever device (CPU, CUDA, Metal) the caller supplies — the same shape
every other mold engine (including this crate's own EVA tower) already takes.
Named, and scoped as mechanical, in Step 0's "What was NOT needed" section:
"Conv → BN → PReLU → Add, Flatten → FC → BN, no global pooling... would load
weights straight from the ONNX initializers, which `candle-onnx` can still
parse" (`pulid-face-extraction.md`).

**B — a device-aware cached evaluator inside the `utensils/candle` fork**:
change `candle-onnx::simple_eval` (or a new entry point) to materialize
initializers once and place them on a caller-supplied device, rather than
re-copying every initializer onto `Device::Cpu` on every call
(`candle-onnx/src/eval.rs`, `simple_eval_`'s `for t in graph.initializer.iter()
{ let tensor = get_tensor(t, ...); values.insert(...) }` loop, confirmed at
this exact fork revision — `get_tensor` hardcodes `Device::Cpu` at four call
sites, `eval.rs:192-231`, and the `Gemm` arm builds its `alpha`/`beta` scalars
on `Device::Cpu` too, `eval.rs:1796-1797`).

### What the numbers say

The Step-0 stage table (`pulid-face-extraction.md`) is the only measured
evidence, and it covers SCRFD + ArcFace only:

| | SCRFD share of p95 | ArcFace share of p95 | EVA + IDFormer share |
| --- | --- | --- | --- |
| halcyon | 180.8 / 415.7 = **43%** | 234.9 / 415.7 = **57%** | **unmeasured** |
| plato | 536.4 / 1574.5 = **34%** | 1050.7 / 1574.5 = **67%** | **unmeasured** |

ArcFace (`glintr100`, 260,665,334 bytes of initializers) dominates the
measured half on both hosts, and dominates harder on plato — consistent with
`simple_eval_`'s per-call re-materialization cost scaling with initializer
size, not with per-request compute (`eval.rs:191-232`, `:249-257`; the Step-0
doc: "the second call costs what the thousandth does and there is nothing to
amortize"). SCRFD's 16,923,827-byte graph pays the same tax at ~1/15th the
size and shows a correspondingly smaller, though still real, share.

> **MEASURED (phase 1) — this inference was wrong.** "Consistent with" was
> doing all the work: ArcFace also dominates because it is ~23 GFLOP against
> SCRFD's ~10, and that, not the memcpy, is why it costs more. Running both
> evaluators **alternately inside one loop** on byte-identical blobs
> (`pulid_face_probe bench --compare`, so both see the same contention) puts
> the resident port at **1.04x** the `candle-onnx` evaluator on the face stack —
> 371.0 ms to 357.9 ms mean, 376.5 ms to 364.9 ms p95. Re-materializing 278 MB
> is worth about 3-4%, not the majority of anything: halcyon is an M4 Max, whose
> memory bandwidth makes a 261 MB copy cheap next to 23 GFLOP of f32
> convolution at candle's ~110 GFLOP/s. The hypothesis was testable and is now
> tested; §4 records the consequence for the acceptance criterion.

### Why B does not answer the question B looks like it answers

A cached evaluator removes the re-materialization tax — real, and probably
the majority of ArcFace's cost — but it **cannot reach a GPU**, because
`get_tensor`'s `Device::Cpu` and the `Gemm` arm's `Device::Cpu` are
independent of caching; a cache still has to build tensors somewhere the
first time, and every other evaluator arm (`Conv`, `BatchNormalization`,
`PRelu`, ...) reads the cached CPU tensors and computes on `Device::Cpu`
throughout `simple_eval_`. Making the *whole* evaluator device-generic is a
materially larger fork change than the already-merged Resize fix — every
arm's implicit `Device::Cpu` would need auditing, not one presence check —
and it is a **hard external prerequisite**: nothing in mold can ship GPU
identity extraction until that lands, reviews, and mold repoints
`candle-onnx` at the fork in `[patch.crates-io]` (which it does not do today,
per §0). Option A has no such gate: `IdentityExtractor::load` already
asserts CPU today only because `candle-onnx` forces it
(`identity/mod.rs:106-114`), not because SCRFD/ArcFace math is CPU-bound —
the anchor decode, NMS, alignment, and warp code the ONNX graphs feed into
are already plain candle/host Rust with no such constraint
(`identity/scrfd.rs`, `identity/align.rs`, `identity/warp.rs`).

> **Superseded (phases 1 and 2).** There is no CPU assertion any more, and
> there never was one in the arithmetic: `IdentityExtractor::load(paths,
> device)` takes and honours a device, `from_paths` merely defaults to
> `Device::Cpu`. Phase 1 replaced the evaluator with resident candle ports and
> phase 2 supplied the leased device, so the "hard external prerequisite" this
> paragraph names never had to land.

### Cost of A under the drop-and-reload rule

SCRFD's weights are 17 MB; `glintr100`'s are 261 MB (f32, per the manifest
pins in `pulid-face-extraction.md`'s op-gate table). A `VarBuilder`-resident
copy of both plus forward-pass activations at their respective fixed input
sizes (640×640 detector, 112×112 recognizer) is on the order of 300–350 MB
device-resident, held only for the extraction and dropped immediately after
— the same build-encode-drop discipline the EVA tower already follows
(`pulid.md`: "The tower is ~609 MB and follows the crate's drop-and-reload
rule: build, encode, drop. Nothing caches it."). That is *smaller* than the
`IDENTITY_VRAM_OVERHEAD_BYTES` already budgeted for the adapter
(1,250,000,000 bytes, `pulid.md`'s memory table) and does not add host RAM —
if anything it reduces `EXTRACTION_HOST_PEAK_BYTES` (`extraction.rs:61-77`)
by moving the two largest host allocations (the re-materialized ONNX graphs)
onto a device.

### The tension a GPU path reopens, and why milestone 1 should not resolve it yet

Extraction runs today at **admission, before the scheduler leases any
device** — deliberately. `identity_extraction.rs`'s own module doc states the
reason plainly: it "runs at admission, before the scheduler has leased a
device, so the T5 and CLIP encoders it must not compete with do not exist
yet... its ~1.4 GB of host RAM is allocated and released before the job is
dispatched. That is a stronger guarantee than a scheduled slot: the two peaks
cannot coexist rather than being arranged not to." Running SCRFD/ArcFace on a
GPU means running them on *some* device, and there is no device to name until
a lease exists — which is exactly the ordering #1223 built this architecture
to avoid needing. Resolving that ordering is a real, separate design (§3
proposes it as a typed scheduler phase); it should not be smuggled in as a
side effect of "port the backbone to candle."

### Recommendation

> **MEASURED (phase 1).** Option A shipped and the port is parity-exact, but
> read the "buys most of the re-materialization win" clause below as the
> prediction it was: the win it buys is ~4%, because the tax it removes was
> ~4%. What survives unqualified is the SECOND half of the argument — "it is a
> strict prerequisite for a future GPU path either way, because candle-onnx
> cannot reach a GPU at all" — and that is now the load-bearing reason the port
> is worth keeping. It also removes a retained 261 MB `ModelProto` per
> extractor, which the evaluator had to keep alive to re-read on every call.

**Ship option A (hand-ported SCRFD + iResNet100 in candle), but keep it
CPU-resident at today's call site for milestone 1.** Concretely: replace only
the internals of `IdentityExtractor` (`identity/mod.rs`,
`identity/scrfd.rs`'s backbone forward, `identity/arcface.rs`'s backbone
forward) from `candle-onnx::simple_eval` calls to ordinary
`VarBuilder`-loaded candle-core/candle-nn forward passes — loading the ONNX
initializers once (candle-onnx's tensor-proto parsing can still be reused
*at load time only*, to pull weights out, exactly as `onnx_inventory.rs`
already introspects the graphs, or the weights can be converted once to
safetensors mirroring `encoders::pickle_convert`'s pattern) and calling
`.forward()` per request against a resident, mmap-backed set of tensors. Keep
`IdentityExtractor::load`'s `Device::Cpu` argument and assertion exactly as
they are (`identity/mod.rs:106-114`) — the call site, the lifetime, the
`ExtractionSlot`, and the pre-lease admission-time architecture all stay
untouched. This buys most of the re-materialization win with **zero**
scheduler, ledger, or lease-ordering changes, and it is a strict prerequisite
for a future GPU path either way (candle-onnx cannot reach a GPU at all,
hand-ported candle can once someone asks it to). Defer GPU dispatch (all four
stages on the render's leased device) to the phase §3 designs, gated on
actually needing it once §4's extended measurement shows where the real time
goes — including the currently-unmeasured EVA/IDFormer half, which may turn
out to dominate SCRFD/ArcFace entirely.

### Acceptance test

- **Parity**: the hand-ported CPU forward passes must match the existing
  `candle-onnx` CPU output within the milestone-1 tolerances already recorded
  in `crates/mold-inference/testdata/pulid/README.md` and
  `pulid-face-extraction.md`'s weight-gated table — landmark position ≤ 1.0
  px, bbox corner ≤ 2.0 px, detection score ≤ 0.02, ArcFace cosine ≥ 0.99 vs.
  the same InsightFace/upstream goldens (`MOLD_TEST_PULID_ASSETS`-gated,
  `capture_goldens.py`). No new tolerance is invented; the hand port is
  qualified against the same fixtures the shipped path is.
- **Speed**: p95 ≥ 25% faster than the recorded baseline (415.7 ms halcyon,
  1574.5 ms plato, SCRFD+ArcFace only — see §4 for how this is checked
  mechanically rather than by eyeballing a percentage).

> **MEASURED (phase 1): parity PASSES, speed FAILS, and the threshold was not
> moved.** Parity is exact — SCRFD is bit-identical to `simple_eval` on all
> nine head tensors across the whole fixture set, ArcFace agrees to 2.62e-6
> with cosine 1.0, and the unchanged InsightFace golden gate still reports
> worst landmark 0.232 px and worst cosine 0.999384. Speed does not:
> face-stack p95 is **370.9 ms against the 415.7 ms baseline, 10.8% faster**,
> where the criterion asked for 25% (ceiling 311.8 ms). That gap is not a
> missing optimization in the port, it is the falsified premise in §1's "what
> the numbers say" — the criterion was sized against a cost that turned out not
> to exist. `--regress-against halcyon` reports exactly this and exits non-zero,
> which is what it is for. The port is kept anyway, for the prerequisite
> argument above; the 25% target moves to §5, restated over the whole
> extraction, where there is a 79% cost centre to take it out of.

---

## 2. Cross-request identity-embedding cache

### Where it lives

> **SHIPPED, one layer lower than proposed.** The cache lives inside
> `mold_inference::identity::extraction::extract_identity_embeddings` (§5), not
> in the server. `resolve_identity_embedding` and `ExtractionSlot` are both
> gone — #1227 phase 2 retired the slot with the call site — and the server
> entry point is now
> `mold_server::identity_extraction::resolve_identity_for_lease`. Everything
> below about key composition, invalidation, and the memory bound shipped as
> written; references to `EXTRACTION_SLOT` describe the admission-side
> substrate phase 2 retired.

The proposal was
`crates/mold-server/src/identity_extraction.rs::resolve_identity_embedding`,
immediately before `ExtractionSlot::acquire()` — a cache hit skips
the slot, the host-memory gate, and the extraction entirely; nothing is
serialized or counted for a hit, because nothing is computed. This was the
single call site every admission path already funnelled through
(`request_resolves_identity`, `EXTRACTIONS`, the `#[cfg(test)] test_stub`
seam), so the cache needed no second integration point.

### Key composition

Enumerated exactly, because the task calls for the exact asset SHAs and
version constants involved:

| Component | Source | Available before extraction runs? |
| --- | --- | --- |
| `sha256(id_image bytes)` | `mold_core::identity::id_image_sha256` (`identity.rs:517-522`) | Yes — pure function of the request |
| **`IDENTITY_PIPELINE_VERSION: u32`** | Shipped in `mold_core::identity` (currently `1`) | Yes — a compiled constant |
| Adapter SHA | `mold_core::pulid_assets::pulid_manifest_for(family)`'s pin for `ModelComponent::IdentityAdapter` — the same read `extraction.rs::adapter_sha256()` (lines 133-141) already performs | Yes — a manifest pin, no file read |
| Vision (derived tower) SHA | `crate::encoders::pickle_convert::EVA_DERIVED_SHA256` — a compiled constant | Yes |
| Face-detector SHA | `onnx_graph::pinned_artifact(ModelComponent::FaceDetector)`'s pin | Yes — the manifest pin, not the post-load `det.sha256` (which is checked equal to the pin or the load fails, so they never disagree) |
| Face-recognizer SHA | `onnx_graph::pinned_artifact(ModelComponent::FaceRecognizer)`'s pin | Yes, same reasoning |
| Face-parser (derived BiSeNet) SHA | `crate::encoders::pickle_convert`'s compiled derived digest, added with #1225 | Yes |

This is deliberately the same shape `IdentityAssetDigests` already carries —
five digests since #1225 added the derived BiSeNet parser — but that struct is
populated **after** extraction, from what actually ran. The cache key needs the
same five digests **before** extraction runs, which is why the table above
resolves each one from its manifest/compiled-constant source rather than from an
`IdentityAssetDigests` a request hasn't produced yet. Compose the key exactly
as `fingerprint_of` already composes the *output* fingerprint
(`identity.rs:496-513`) — domain-separated, newline/NUL-joined SHA-256 — but
over the photograph, the version, and the five asset digests instead of the
extracted values. It shipped as `mold_core::identity::identity_cache_key`, fed
by `extraction::pinned_asset_digests(family)`:

```
sha256("mold.identity.cache.v1\0"
       || id_image_sha256 || "\0"
       || IDENTITY_PIPELINE_VERSION.to_le_bytes() || "\0"
       || adapter_sha || "\0" || vision_sha || "\0"
       || face_detector_sha || "\0" || face_recognizer_sha || "\0"
       || face_parser_sha)
```

**Never key on `id_image` bytes alone** (the task's own framing, and the
reason the table above exists): a bare-photo key would serve a stale
embedding across a repair-pull that swapped the adapter or a code change that
altered SCRFD/ArcFace/EVA/IDFormer semantics without re-running anything —
exactly the failure `IDENTITY_PIPELINE_VERSION` exists to catch (see
invalidation, below), and exactly why every asset SHA is a mandatory input
rather than an optimization to drop for a smaller key.

### Storage

An in-process LRU, small — 8 to 16 entries is enough (see memory bound,
below) — living as a new module-level static beside `EXTRACTION_SLOT` and
`EXTRACTIONS` in `identity_extraction.rs`, same file, same
process-global-admission-state ownership boundary those already establish.
Value = `FrozenIdentityEmbedding` (already `Clone`, `Eq`, `Hash`,
device-independent, 256 KiB — `identity.rs:405-482`) plus the `Option<String>`
multi-face warning. On a hit: return both immediately, touch neither
`EXTRACTION_SLOT` nor `EXTRACTIONS`. On a miss: proceed exactly as today (slot
→ counter → `extract_blocking`), then insert.

### Invalidation

1. **Different photo bytes** → different key automatically; not
   invalidation, just a fresh entry via content addressing.
2. **A repair-pull swaps an asset** (`mold pull pulid-flux
   --accept-license ...` after `mold rm pulid-flux`, or a re-pinned release) →
   the manifest pin or `DERIVED_SHA256` changes → the key changes
   automatically; stale entries simply become unreachable and age out under
   LRU pressure. No explicit purge needed.
3. **A pipeline code change** (this issue's own hand-port swap in §1, or
   #1225's future BiSeNet mask) → **must** bump `IDENTITY_PIPELINE_VERSION`,
   or a cache entry computed by the old code silently serves a result the new
   code would compute differently, under an unchanged photo+asset-SHA key.
   This is the one invalidation case that is not structural — it needs a test
   pinning "the key changes when the version changes" (so a reviewer can
   check the constant actually moved) and a note in whichever PR ships a
   semantic change to SCRFD/align/warp/arcface/eva_clip_preprocess/
   eva_clip_vision/pulid_encoder.
4. **Process restart** → the cache is in-process only; cold on every
   restart. Acceptable — see persistence, below.

### Hit/miss tests to write

In `identity_extraction.rs`, alongside the existing lifetime tests
(`a_conditioned_request_extracts_exactly_once`,
`concurrent_identity_preparations_serialize_on_the_extraction_slot`):

- Two requests, byte-identical `id_image` → `extraction_count()` advances by
  exactly 1 total; a new `cache_hit_count()` (mirroring `extraction_count()`'s
  existing test-only counter pattern) advances by exactly 1 on the second.
- Two requests, different `id_image` bytes → `extraction_count()` advances by
  2, zero cache hits.
- Same photo, second request has `id_weight: 0.0` → zero extractions **and**
  zero cache lookups — `request_needs_identity_assets`
  (`identity_dependencies.rs`) short-circuits before the cache is even
  consulted, exactly as it short-circuits `ExtractionSlot` today
  (`a_zero_weight_request_performs_no_extraction`).
- A slow/panicking stub installed via the existing `StubbedExtractor` seam,
  driven twice with the same photo: the stub must be entered exactly once —
  proves a hit never touches `ExtractionSlot::acquire()`.
- Bump a test-local `IDENTITY_PIPELINE_VERSION` stand-in and confirm the key
  changes for byte-identical photo + assets — pins the one non-structural
  invalidation case.

### Memory bound

Each entry is 256 KiB (`ID_EMBEDDING_VALUES = 32 × 2048` f32,
`identity.rs:369-370`) plus a short warning string and a ~64-byte key — call
it 260 KiB. An 8–16 entry cap is 2–4 MiB total, irrelevant next to
`EXTRACTION_HOST_PEAK_BYTES`'s 1.4 GB transient extraction peak. Size the cap
for *this session's* concurrent hot set — a batch's siblings, a sequence's
per-clip identity, a `Prepare N variations` review — not for a photo
library; a byte-budget cap is unneeded complexity given the fixed entry size,
so an entry-count cap alone is sufficient.

### Interaction with #1226's multi-photo averaging

Confirmed directly with the #1226 implementer while writing this doc, rather
than assumed: averaging happens **after** the IDFormer, per-photo, mirroring
`cubiq/PuLID_ComfyUI`'s `pulid.py:406,415-419` (append each photo's IDFormer
*output*, then `torch.mean(dim=0)` across the stacked outputs) —
`ToTheBeginning/PuLID`'s own `pipeline_flux.py:120-194` only ever handles one
photo, so ComfyUI is the reference for the N-photo case. **This means caching
the final per-photo `[1, 32, 2048]` tokens (256 KiB) is sufficient and
correct** — there is no need to cache the larger pre-IDFormer intermediates
(ArcFace raw + EVA hidden states + CLS projection, ~2.4 MB/photo) this doc
originally considered before confirming the averaging stage. Each photo in a
multi-photo request independently hits or misses this same per-photo cache;
#1226's averaging step composes over N independent lookups because the mean
is order-independent at the *value* level even though #1226's extended
`FrozenIdentityEmbedding` fingerprint carries an ordered `source_sha256s` list
and is therefore order-*sensitive* at the fingerprint level — that ordering
concern applies to the composed multi-photo value #1226 produces downstream
of this cache, not to this cache's own per-photo key, which stays a scalar
per photo and is unaffected by request-level photo ordering.

One more fact from #1226 worth designing for now rather than retrofitting
later: a true-CFG **uncond** embedding — `IDFormer(zeros_like(id_cond),
[zeros_like(h) for h in id_vit_hidden])` (`pipeline_flux.py:188-192`) — is a
pure function of the adapter checkpoint alone, never of any photo. Cache it
as a **separate, degenerate one-time memo** keyed only on the adapter SHA
(effectively a lazily-initialized `OnceCell`, not an LRU entry keyed on "no
photo") — it is computed once per process and reused by every subsequent
true-CFG request regardless of which photo(s) are supplied. Folding it into
the per-photo LRU would waste a slot on a value that never needs eviction and
would need an awkward sentinel key.

### Persistence

**Recommend no**, unless a future need is demonstrated. Three reasons:

1. `FrozenIdentityEmbedding`'s `Debug` impl exists specifically to redact its
   values because "they are a biometric derivative and must never reach a log
   line, an error body, or a probe payload" (`identity.rs:484-494`). Writing
   them to a persistent on-disk store — even one scoped to `$MOLD_HOME` —
   introduces a retention and deletion story (how long, who can read it, what
   `mold rm pulid-flux` or a gallery purge should do to it) that nothing in
   the codebase's current posture toward these bytes anticipates; every other
   identity artifact is either a manifest-pinned model file or a SHA, never a
   raw biometric-derived float array at rest.
2. The entries are cheap enough to recompute — especially once §1's hand
   port lands — that persistence buys comparatively little against that
   liability.
3. The in-process LRU already covers the actual hot path this issue is aimed
   at: repeat renders within one session (batch siblings, `Prepare N
   variations`, a sequence's per-clip identity, an interactive retry). A
   cross-restart cache would only help a rarer workflow (the same face across
   separate `mold run` invocations on different days), which is exactly where
   the privacy cost of a persistent biometric store is least justified
   relative to the benefit.

### Interaction with the durable queue/journal

This cache sits entirely upstream of the durable queue: it is consulted (or
populated) during `resolve_identity_embedding`, at admission, before a
`QueueTicket` or `generation_queue` row exists at all (CLAUDE.md's queue
durability section; `mold-server`'s `scheduler/mod.rs`/`gpu_worker.rs`). A
replayed job after a crash re-submits the ORIGINAL request — including its
full `id_image` bytes, already retained by the journal within its byte budget
— through the ordinary admission path, so replay simply re-runs
`resolve_identity_embedding`: a cache hit if the process never restarted (or
another still-live entry matches the same photo+assets), an ordinary cache
miss otherwise. The cache has no observable effect on the frozen embedding's
*value*, only on how fast it is produced, so it needs no special-casing in
the journal/replay path — but it deserves one regression test: call
`resolve_identity_embedding` twice for the same request, once simulating "no
restart" (cache warm) and once after clearing the process-global cache
(simulating "restarted before replay"), and assert the resulting
`FrozenIdentityEmbedding` is byte-identical either way.

---

## 3. EVA-CLIP residency and a proposed typed scheduler phase

### Current residency already matches the drop-and-reload rule

The EVA tower and the IDFormer already run through ordinary
`candle-core`/`candle-nn` (never `candle-onnx`), on `Device::Cpu`, inside
`compose_identity_tokens` (`extraction.rs:147-218`): loaded via
`VarBuilder::from_mmaped_safetensors`, built (`EvaClipVisionTower::new`),
forwarded once, and released when the block scope ends — before the IDFormer
is even built (`extraction.rs:163-166`'s comment: "The tower and the IDFormer
are built and dropped in sequence, never held together... admission is the
one place in the process where host RAM is not already committed to a
render."). `pulid.md` confirms this is the intended policy: "The tower is ~609
MB and follows the crate's drop-and-reload rule: build, encode, drop. Nothing
caches it." **There is no live residency gap today** — this already satisfies
CLAUDE.md's drop-and-reload discipline for identity assets exactly as it does
for T5/CLIP.

### What changes if GPU dispatch (§1's deferred follow-up) ships

If the EVA tower and IDFormer (and, per §1, the hand-ported SCRFD/ArcFace
backbones) move from `Device::Cpu` to the render's leased device, the same
build-encode-drop discipline must hold on that device: build, forward, drop
— fully released — **before** `flux::identity::EngineIdentityState`'s
ordinary adapter-residency logic begins for that dispatch
(`crates/mold-inference/src/flux/identity.rs:8-21`'s own residency doc). The
extraction's transient allocation must never coexist with the resident
cross-attention adapter or the transformer's own peak; it is a strictly
earlier, strictly disjoint phase of the same lease, not a third permanent
resident.

### The host-RAM ledger fold-in this implies

Today, extraction's host RAM is charged at `ExtractionSlot::acquire()`
(`identity_extraction.rs:106-134`) against a **fresh** `ram_snapshot()` read,
gated by a bare `tokio::sync::Semaphore::const_new(1)` — deliberately **not**
the scheduler's `HostMemoryLedger`, because (the module's own doc) "extraction
happens strictly BEFORE this job has a lease... Re-entering the planner from
admission to charge a transient CPU peak would invert that ownership." Once
extraction runs on a leased device, that ordering argument inverts: the job
now *has* a lease, so its host-RAM charge belongs in the same
`admission_host_demand_bytes` frozen-plan-recheck discipline every other
phase already uses (CLAUDE.md: "the worker rechecks the exact frozen
`admission_host_demand_bytes` against headroom from the granting ledger...
before any fresh load"). Concretely, this milestone would retire
`ExtractionSlot`'s bespoke semaphore-plus-snapshot gate in favor of the
ledger, rather than running the two in parallel — two independent admission-
time memory gates that can disagree is worse than one, and the ledger's
recheck-at-dispatch discipline already exists to answer exactly this
question for six other phases.

### The proposed phase

Add `ProgressPhase::IdentityExtract` to
`crates/mold-inference/src/progress.rs:74-89` (alongside `ModelLoad`,
`PromptEncode`, `Vae`, `VisualDecode`, `AudioDecode`, `Mux`, `Upscale`),
emitted as `ProgressEvent::PhaseDone { phase: ProgressPhase::IdentityExtract,
name, elapsed }` bracketing the (now post-lease) SCRFD+ArcFace+EVA+IDFormer
run.

Files that change, and why:

| File | Change |
| --- | --- |
| `crates/mold-inference/src/progress.rs` | New `IdentityExtract` arm on `ProgressPhase` (line ~74-89) |
| `crates/mold-scheduler/src/estimates.rs` | New `identity_extract_ms: Option<u64>` on `EstimatePhaseTimings` (beside `cold_load_ms`/`prompt_encode_ms`, ~line 90-101); new `ewma_identity_extract_ms: Option<f64>` on `EstimateBucket` (~line 128-145) |
| `crates/mold-server/src/gpu_worker.rs` | New `add_phase_sample(&mut timings.identity_extract_ms, elapsed)` arm in `record_phase_timing`'s match (mirrors the existing `PromptEncode`/`Vae`/etc. arms) — and a decision that only the ONE sibling that actually performed the (once-per-parent) extraction records the sample; every other sibling's `identity_extract_ms` stays `None`, exactly like `cold_load_ms` already being `None` on a warm reuse — the EWMA machinery is already `Option`-tolerant throughout |
| `crates/mold-db/src/scheduler_estimates.rs` | New `ewma_identity_extract_ms` column in the `scheduler_estimates` read/write/upsert SQL, mirroring every existing `ewma_*_ms` column |
| `crates/mold-db/src/migrations.rs` | A schema bump: `SCHEMA_VERSION` 20 → 21, adding the new column via `ALTER TABLE scheduler_estimates ADD COLUMN ewma_identity_extract_ms REAL` — this one is **not** a bare additive Rust `Option` an old row already reads as `NULL`; SQLite needs the column to exist before the new `INSERT`/`UPDATE` statement can reference it |
| `crates/mold-server/src/identity_dependencies.rs` / the scheduler dispatch path | The call site for extraction moves (or gains a second, post-lease trigger) from `variant_dependencies::prepare_inputs_for_devices` (pre-lease, current) to inside the leased job's own pipeline, first, before prompt encode |

### Recommendation

**Do not build this phase for milestone 1.** It only pays for itself once
extraction genuinely competes for time on a leased device — i.e., once GPU
dispatch ships. Wiring the enum arm, the `EstimatePhaseTimings`/`EstimateBucket`
fields, the schema bump, and the ledger fold-in now, while extraction stays
CPU/pre-lease (§1's milestone-1 recommendation), would land a migration and
five touched files that only ever record `None`. Land this phase in the same
change that moves extraction post-lease, not before it.

---

## 4. Qualification protocol

### Existing harness

`pulid_face_probe bench <dir> [--warmups N] [--runs N]`
(`crates/mold-inference/src/bin/pulid_face_probe.rs`), gated at
`LATENCY_BUDGET_MS = 2000.0` per image (SCRFD detect + ArcFace embed only,
per §0). `GATE_WARMUPS = 5` / `GATE_RUNS = 20` is the real protocol;
anything short of it prints an ADVISORY, not a GATE verdict
(`validate_bench_args`), and `--runs 0` is refused outright rather than
reporting a vacuous PASS.

```bash
cargo build --release -p mold-ai-inference --features dev-bins,pulid --bin pulid_face_probe
./target/release/pulid_face_probe bench /path/to/pulid-assets-dir
```

### The extension, as it shipped

`pulid_face_probe bench <dir>` gained three flags in phase 1:

| flag | what it adds | why |
| --- | --- | --- |
| `--full --adapter <safetensors> --eva <pt>` | four more rows: `eva-build`, `eva-forward`, `idformer-build`, `idformer-fwd`, plus a whole-extraction total | §0's finding. Build and forward are separate rows because they answer different questions: the forward is arithmetic, the build is what the drop-and-reload rule **re-pays on every request** and is therefore what §5 would buy back |
| `--compare` | re-measures the retained `candle-onnx` oracle, **alternating with the port inside one iteration**, on byte-identical blobs | Two sequential blocks measure whatever else a shared box was doing during each block. Alternating makes both evaluators see the same contention, which is what makes the 1.04x number above trustworthy on a machine that never goes quiet |
| `--regress-against halcyon\|plato` | checks the face-stack p95 against `BASELINE_*_P95_MS` at `REGRESSION_MARGIN` (0.75) | §1's "25% faster" as a mechanical check with a non-zero exit, not a percentage a reviewer recomputes |

It runs through `extraction::compose_identity_tokens_observed` — the SAME
implementation the server calls, with a per-stage observer;
`compose_identity_tokens` delegates to it with a no-op closure, so the benchmark
cannot drift from production by measuring a copy of it.

Two scoping rules, both load-bearing:

- **The whole-extraction total is reported, never gated.** #1222's
  `LATENCY_BUDGET_MS` was stated over SCRFD + ArcFace and nothing else, so
  applying it to a four-stage measurement would fail a gate this build was
  never subject to. The gate stays on the face stack; the total is printed as
  the first per-request figure for the whole extraction, and §5 states the
  budget that *should* cover it.
- **Every run prints the 1-minute load average at both ends.**
  `pulid-face-extraction.md`'s own cautionary example is a p95 that tripled
  under load average 83. A benchmark that does not report this invites the same
  mistake again.

```bash
cargo build --release -p mold-ai-inference --features dev-bins,pulid --bin pulid_face_probe
./target/release/pulid_face_probe bench /path/to/antelopev2 --compare --full \
  --adapter /path/to/pulid_flux_v0.9.1.safetensors \
  --eva /path/to/EVA02_CLIP_L_336_psz14_s6B.pt
```

### MEASURED: one identity extraction on halcyon

Apple M4 Max, 16 cores, 48 GiB, macOS aarch64, release build, the 5-warmup /
20-run gate protocol, **load average 9.76 at start and 12.22 at end** — the box
was running peer builds throughout and a quiet window never came, which is why
the paired `--compare` numbers rather than the absolutes carry the argument.
Sample spread was nonetheless tight (every stage's min-to-max is under 6%).

| stage | p50 (ms) | p95 (ms) | share of p50 total |
| --- | ---: | ---: | ---: |
| `scrfd` — letterbox, blob, detect, decode, NMS | 160.0 | 165.1 | 5.6% |
| `arcface` — align, blob, embed | 198.7 | 209.4 | 7.0% |
| **face stack** (what #1222 measured) | **359.7** | **370.9** | **12.7%** |
| `eva-build` — re-authenticate 609 MB, mmap, widen f16→f32, construct | 1267.9 | 1273.1 | 44.6% |
| `eva-forward` — 24 blocks, 577 tokens, f32 | 969.2 | 981.6 | 34.1% |
| `idformer-build` | 46.0 | 47.1 | 1.6% |
| `idformer-fwd` | 194.2 | 202.4 | 6.8% |
| **whole extraction** | **2840.4** | **2863.9** | **100%** |

Three things fall out of that table and none of them was predictable from the
Step-0 numbers:

1. **The EVA tower is 79% of an extraction** (2,237 ms of 2,840 ms). Every
   optimization argument in §1 was about the 13% the harness happened to be
   pointed at.
2. **`eva-build` is the single largest line item in the entire pipeline** —
   larger than the tower's own forward pass, and nearly four times the whole
   face stack. It is pure per-request setup: `derived_artifact_is_authentic`
   re-hashes the 609 MB derived safetensors on every call (deliberately —
   CLAUDE.md's "reuse is authenticated by the compiled-in `DERIVED_SHA256`,
   never by the sidecar"), and `VarBuilder::from_mmaped_safetensors(...,
   DType::F32, ...)` then widens those f16 weights into ~1.2 GB of f32. The
   drop-and-reload rule pays for both on every conditioned request. Splitting
   that line further — hash vs. mmap vs. construct — is the first thing §5
   should measure, because the two halves have completely different fixes.
3. **The IDFormer's build is nearly free** (46 ms) while its forward is 194 ms,
   the opposite shape, because its `VarBuilder` is lazily mmap-backed over the
   adapter file rather than a re-authenticated widening copy.

### MEASURED: the port against the evaluator it replaced

Same run, `--compare`, alternating inside one iteration:

| | `candle-onnx` mean | resident port mean | speedup |
| --- | ---: | ---: | ---: |
| SCRFD graph | 163.7 ms | 159.1 ms | 1.03x |
| ArcFace graph | 207.3 ms | 198.8 ms | 1.04x |
| both | 371.0 ms | 357.9 ms | 1.04x |
| both, p95 | 376.5 ms | 364.9 ms | 1.03x |

That is the honest value of the hand port as a speed change, and it is the
number §1's "what the numbers say" was wrong about. Its real value is
elsewhere: `candle-onnx` cannot place a tensor anywhere but `Device::Cpu`, so
without this port §5 is not implementable at all.

The regression pin committed for this is deliberately **relative** —
`the_resident_port_is_never_slower_than_the_evaluator_it_replaced` in
`tests/pulid_handport_parity.rs`, asset-gated and `#[ignore]`d, alternating the
two evaluators exactly as `--compare` does. A millisecond ceiling is a property
of the machine and its load; a ratio survives both. It is not a hypothetical
guard: the first version of `arcface_net` computed the fully-connected layer as
`X @ W^T`, materializing a transpose of the 51 MB weight on **every forward**,
which made the "faster" port about a tenth slower than `simple_eval` on the
recognizer. This ratio is what caught it, before it shipped.

### plato: not measured

Deliberately skipped in phase 1 and named rather than quietly omitted. A release
build of `mold-ai-inference` on plato is well past the time this issue had for
it, and the conclusion does not turn on it: the falsified claim is about
*whether re-materialization is the cost centre*, and the four-stage table shows
the answer is "no, and neither is the face stack" on any host where the EVA
tower runs the same 300 GFLOP. plato's own 1574.5 ms face-stack baseline stays
committed in `pulid_face_probe` for whoever runs `--regress-against plato`
there. A phase-2 measurement should take both hosts, because §5's win is a
device question and plato has four L40S.

### Warm-repeat protocol

Unchanged from Step 0: `simple_eval` (and, after §1's hand port, an ordinary
resident `VarBuilder` forward) has no cold/warm split to amortize once
weights are resident — "the second call costs what the thousandth does" — so
5 warmups / 20 runs is the floor, and a short run reports noise as signal (the
Step-0 doc's own cautionary example: 588.4 ms p95 under load average 83 with
the full protocol vs. 2533.7 ms from `--runs 2` on the same host).

### Named configurations

Identical to Step 0's, so before/after numbers are directly comparable
without re-baselining:

- **halcyon** — Apple M4 Max, 16 cores, 48 GiB, macOS (aarch64-darwin), this
  Mac.
- **plato** — 128-core x86_64 NixOS, 1.5 TiB RAM, 4× L40S, reached via `ssh
  plato` (Tailscale `100.105.134.43`).

### Regression test at the measured numbers

Shipped as described: `BASELINE_HALCYON_P95_MS = 415.7`,
`BASELINE_PLATO_P95_MS = 1574.5`, `REGRESSION_MARGIN = 0.75`, and
`--regress-against <halcyon|plato>` exiting non-zero when the face-stack p95
exceeds the ceiling. On halcyon it currently **fails at 370.9 ms against a
311.8 ms ceiling**, and that is left as-is rather than loosened: the flag is
reporting a real gap between what §1 predicted and what the port delivers, and
a threshold edited to match the outcome measures nothing. §5 is where the 25%
is now expected to come from.

The committed *pass/fail* pin is the relative one described above
(`the_resident_port_is_never_slower_than_the_evaluator_it_replaced`), because
it holds on any machine under any load. The absolute baselines stay as
documentation of where the numbers came from, checked into
`pulid_face_probe.rs` where the flag that reads them lives, and pinned by
`the_named_baselines_are_the_ones_the_doc_records`.

### Device-path parity

No new tolerances: the hand port from §1 is qualified against the exact
tolerances already recorded in `crates/mold-inference/testdata/pulid/README.md`
and `pulid-face-extraction.md`'s weight-gated table (landmark ≤ 1.0 px, bbox
≤ 2.0 px, score ≤ 0.02, ArcFace cosine ≥ 0.99) and `pulid.md`'s encoder-parity
numbers (IDFormer output to 1.5e-7 of scale, CLS projection to 1.3e-5
absolute, hidden states to ~1e-4 of peak magnitude) — run through the
existing `capture_goldens.py`/`capture_eva_goldens.py` fixtures,
`MOLD_TEST_PULID_ASSETS`-gated, `#[ignore]`d in the ordinary suite exactly as
they are today.


---

## 4b. MEASURED: phase 2, on halcyon

Same box, same 5-warmup / 20-run protocol, same release build, load average
reported at both ends of every run. Three configurations, taken back to back:
**CPU before** is this branch's parent commit (35b58acc, which already carries
#1292's BiSeNet mask — the phase-1 table in §4 predates it and therefore never
paid the `bisenet` row); **CPU after** is item 1 alone; **Metal** is items 1
and 2 together, which is what halcyon actually runs.

| stage | CPU before | CPU after | Metal |
| --- | ---: | ---: | ---: |
| `scrfd` | 158.0 | 158.9 | 63.9 |
| `arcface` | 193.8 | 198.5 | 34.4 |
| **face stack** | **351.7** | **365.0** | **97.9** |
| `bisenet` (per-crop parse + mask, #1292) | 147.4 | 149.7 | 51.1 |
| `eva-build` | 1355.6 | 186.2 | 81.6 |
| — `parser` (materialize + build BiSeNet) | 99.1 | 5.6 | 8.8 |
| — `eva-auth` (609 MB private read + SHA-256) | 1124.2 | 51.5 | 32.5 |
| — `eva-ctor` (`VarBuilder` at the working dtype) | 126.4 | 128.5 | 40.9 |
| `eva-forward` | 986.6 | 974.7 | 92.9 |
| `idformer-build` | 46.0 | 46.2 | 55.1 |
| `idformer-fwd` | 206.3 | 196.4 | 17.9 |
| **whole extraction** | **3073.6** | **1907.3** | **395.3** |

All figures are p95 milliseconds per image. Load average 7.6–9.6 throughout;
the box was running peer builds and never went quiet, which is why the three
columns were taken in one session rather than compared against §4's absolutes.

### The split §5 asked for, and what it said

§5 predicted `eva-build` was "roughly half `derived_artifact_is_authentic`
re-hashing 609 MB and half widening f16 → f32 into ~1.2 GB". **It is not.**
The re-proof is **1,124 ms of the 1,356 ms** and the widening is **126 ms**.
On an M4 Max a 609 MB → 1.2 GB dtype conversion is a memory-bandwidth
operation and costs about what you would expect from 500 GB/s; the SHA-256
is not, because the `sha2` crate takes its portable path here and runs at
roughly 550 MB/s.

That matters beyond bookkeeping: had the split not been measured first, the
obvious fix — build the tower at its stored f16 — would have bought 126 ms and
the 25% criterion would have been missed on the host by a wide margin. The
decision record's own instruction to measure before deciding is what produced
the right target.

So item 1 is a memo, not a dtype change:
`pickle_convert::open_authenticated` memoizes the SHA-256 pass on the file's
own `(dev, ino, size, mtime, ctime)` identity **plus the pin**, re-read from
the same retained descriptor after the private copy is taken, with a mismatch
falling through to an ordinary full read-and-hash. It is
`mold_core::download::pinned_file_digest`'s memo applied to the one artifact
that reads its bytes privately. `ctime` is the load-bearing field: `utimensat`
lets an owner set `mtime` to anything, but no userspace call holds `ctime`
still across a write. The private-read-then-build contract is untouched — the
copy is still what the `VarBuilder` reads, and the memo only ever removes a
pass over it. `eva-auth` 1,124 → 51 ms, and the BiSeNet parser's own
materialization falls out with it, 99 → 6 ms.

The dtype change shipped anyway, as part of item 2: `eva_working_dtype` keeps
the tower f32 on the host — where candle has no narrow kernels and where every
committed parity golden was captured — and f16 on a device, which is the dtype
the derived file already stores and *narrower* than upstream's own cast
(`PuLID/pulid/pipeline_flux.py:60` casts the tower to `weight_dtype`, bf16 in
`PuLID/app_flux.py:45`). The IDFormer half still computes in f32 whatever the
tower did.

### The device path

`eva-forward` **986.6 → 92.9 ms**, a 10.6x speedup on the stage that was 79%
of a phase-1 extraction, and the face stack 351.7 → 97.9 ms almost for free
once the device exists — which is exactly the shape §5 predicted and §1 got
wrong. `idformer-build` is the one stage that got *slower* on Metal (46 →
55 ms), because it stops being a lazy mmap and starts being a 605 MB host →
device copy; it is 14% of a 395 ms extraction and not worth a residency
argument.

A note on how these were measured: Metal enqueues, so the first device run of
this benchmark reported `eva-forward` at **3.2 ms** and hid 300 GFLOP inside
the IDFormer's `to_vec1`. `compose_identity_token_sets_observed` now
synchronizes at every stage boundary (`settle`), which costs nothing — the
pipeline was already serial — and is the difference between a measurement and
a story.

### Acceptance

| criterion | result |
| --- | --- |
| whole-extraction p95 ≤ **2,147.9 ms** on halcyon | **PASS** — 1,907.3 ms on the CPU (33.4% under), 395.3 ms on Metal (86.2% under) |
| device path within the recorded parity tolerance | **PASS** — worst 3.82e-5 of peak across the four committed portraits, against the 5e-5 the whole-stack golden already states. No new tolerance |
| measured device peak within 10% of the charged term | **PASS** — 643,825,664 measured on plato against a 700,000,000 charge (8.7% margin). See "plato: the memory measurement", below, including why the first version of the test reported zero |
| ordering: the tower is released before the adapter is resident | **PASS** — structural: extraction runs before the model load, and `EngineIdentityState` cannot begin until the engine exists |

CUDA is now measured too: whole extraction **573.2 ms** on an L40S, parity
worst 4.908e-5 against the 5e-5 budget, and a real render at cosine 0.6259.
Those numbers and their caveats are in the plato subsections below.

The threshold was not moved. It is 25% under the phase-1 figure of 2,863.9 ms,
which is a *harder* target than it looks now: the phase-1 run predates #1292's
BiSeNet mask, so the comparable before-figure on this branch's parent is
3,073.6 ms and the CPU result is 38% under that.

### The memo's residual, and why it is the one already accepted

`open_authenticated`'s memo is keyed on the file's METADATA — `(dev, ino, len,
mtime, ctime)` plus the pin — so a same-length in-place overwrite landing
inside one timestamp tick would be served without being hashed. Two things
bound that:

- **Coarse filesystems are fenced out entirely.** A mount reporting
  whole-second timestamps sets both nanosecond fields to zero, and
  `ArtifactIdentity::is_fine_grained` refuses to memoize such an identity in
  either direction — never consulted, never recorded. Every load there pays the
  full SHA-256, which is slower and correct.
- **What remains is exactly `mold_core::download::pinned_file_digest`'s
  residual**, which already guards every model weight mold loads: a same-length
  write landing in the same nanosecond on a filesystem that reports
  nanoseconds, where `ctime` is not settable from userspace at all. Holding the
  609 MB tower to a stricter standard than the 24 GB checkpoint beside it would
  need a reason nobody can state.

The alternative was considered: keep the verified private copy resident for the
process lifetime, which makes the bytes immutable by construction and removes
the re-read as well as the hash. It was rejected on cost — 609 MB of host RAM
held forever, against a re-read the memo already reduces to ~50 ms, on a phase
whose entire point was to stop paying for the tower when nothing is using it.

### The cache is single-flight, and that is not an optimization

§2 designed the cache as a lock-free get/put around the extraction, which was
correct while extraction ran once per parent at admission. Phase 2 moved it
into each lease, so the callers are now N sibling GPU worker threads that
arrive together — and a plain get/put lets all N miss a cold cache in the
window between the get and the put. That costs N times the work, which is
merely bad, and produces N embeddings that differ at the measured 3.82e-5
device tolerance, which is worse: four siblings of one print conditioned on
four slightly different faces, with four different frozen fingerprints.

So the miss path takes a per-key lock (`flights_for` / `release_flights`), and
a caller that waits **re-reads the cache and takes the winner's tokens** rather
than computing its own. Sibling embeddings are byte-identical by construction,
not equal within a tolerance. Locks are acquired in sorted key order because a
multi-photograph set takes several at once and two requests sharing a subset in
different orders would otherwise deadlock; the unconditional identity gets its
own flight key for the same reason its value reaches the fingerprint. A failed
flight stores nothing and releases the key — there is no negative caching,
because a torn file or a momentarily undetectable face is not a permanent
answer.

The counter moved with it. `identity_extraction_count` counts what was
COMPOSED, and `ResolvedIdentity::extracted` carries that back to the server, so
the once-per-parent contract is checkable again (four siblings, one extraction)
and `ProgressPhase::IdentityExtract` is emitted only for the sibling that did
the work — a cache hit reporting its ~2 ms would drag
`ewma_identity_extract_ms` to a figure no cold request could meet.

The children need no frozen key handed down from the parent: the key is a pure
function of the photograph bytes each child already carries plus the build's
own asset digests, so four siblings derive one key by content addressing. A
second copy travelling in the plan would be an authority that could disagree
with the bytes it claims to describe. What preparation could not give them, and
the flight does, is computing it once.

### An uncond-only miss loads only the IDFormer

The first true-CFG request after an ordinary one has its photograph cached and
only its unconditional identity missing. That value depends on no photograph
(`pipeline_flux.py:188-192`), so the face stack is not loaded at all — opening
SCRFD and ArcFace would place ~278 MB on the device to run neither. Measured on
Metal: **60.6 ms**, against ~340 ms if the graphs had been decoded.

The cache (item 3) is deliberately absent from every number above.
`bench --full` drives `compose_identity_tokens_observed`, which does not
consult it, so these are cold-extraction figures. What the cache is worth is a
separate measurement, taken through the production entry point on Metal:
**cold 2,184.8 ms → warm 1.8 ms**, with the second extraction opening no
detector, recognizer, parser, tower, or adapter at all.

### plato (CUDA): measured

Measured on **plato** (128-core x86_64 NixOS, 4x L40S) at `3163ed47`, after PR
#1295 opened — the phase-2 branch's own head, not a reconstruction. Same
5-warmup / 20-run protocol. Both device arms of the same build, back to back:

| stage | CUDA (L40S) | CPU (same box) |
| --- | ---: | ---: |
| `scrfd` | 18.9 | 498.6 |
| `arcface` | 6.7 | 848.7 |
| **face stack** | **25.4** | **1346.2** |
| `bisenet` | 25.3 | 412.6 |
| `eva-build` | 464.2 | 1248.9 |
| — `parser` | 44.6 | 24.3 |
| — `eva-auth` | 311.1 | 362.7 |
| — `eva-ctor` | 69.7 | 831.5 |
| `eva-forward` | 12.4 | 2266.9 |
| `idformer-build` | 45.6 | 183.4 |
| `idformer-fwd` | 5.8 | 536.4 |
| **whole extraction** | **573.2** | **6024.9** |

p95 milliseconds per image, **on a quiet box**. That caveat is load-bearing:
under plato's normal load average of 10-32 the same build measures the whole
extraction at **688-795 ms**. Two quiet-box runs a commit apart came in at
**573.2** and **565.3 ms**, so the floor is reproducible to about 1.5%; it is
still a floor and not the expectation, and any comparison drawn against it has
to carry the load average the way §4's own cautionary example demands.
`--regress-against-full plato` passes at 90.6% faster than the CPU baseline
below.

`BASELINE_PLATO_FULL_P95_MS` is the **CPU 6,024.9**, not this CUDA number, and
the first attempt got that wrong in an instructive way. A
`--regress-against-full` baseline is a *before* figure: halcyon's 2,863.9 is
phase 1's own pre-phase-2 measurement, so demanding 25% off it means something.
Putting phase 2's own CUDA result in the same slot produced a check that
demanded the run be 25% faster than itself — it failed by construction, and on
a loaded box it failed loudly. The CPU full stack is the pre-phase-2 analogue
for this host, and 25% under it (4,518.7 ms) is a bar both arms clear, which is
correct, because both got faster.

Three things in that table are worth reading twice. `eva-forward` is **12.4 ms
against the CPU's 2,266.9** — 183x, which is what a 300 GFLOP dense ViT is
supposed to do on a datacentre card and is the single result phase 2 was built
to get. `eva-auth` is **311.1 ms on CUDA against 362.7 on the CPU**, i.e. the
memo works identically because it is a host-side file read either way — the
absolute figure is higher than halcyon's 51 ms because plato's storage is
slower, not because the memo is weaker. And `eva-ctor` collapses from 831.5 to
69.7 ms for the reason `eva_working_dtype` exists: the CPU arm widens f16 to
f32 and the device arm does not.

`--regress-against plato` (the face-stack criterion, stated against #1222's
1,574.5 ms `candle-onnx` CPU baseline — also a before figure, also left alone)
**passes on CUDA at 98.4% faster** and
**fails on the CPU at 14.5%** — the same shape halcyon shows, and for the same
reason §1 records: the criterion was sized against a re-materialization cost
that turned out not to exist. The face-stack baseline is deliberately left as
the CPU measurement it always was.

### plato: parity, and how little margin CUDA leaves

Device-vs-host token parity **passes**, against the same 5e-5 the whole-stack
golden already states:

| portrait | relative error of peak |
| --- | ---: |
| rubio | 2.661e-5 |
| chari | 2.062e-5 |
| barron | 3.828e-5 |
| **jemison** | **4.908e-5** |

**Record this: CUDA leaves almost no margin.** 4.908e-5 against a 5e-5 budget
is 98% of it, where Metal's worst was 3.82e-5 (76%). The budget is not being
loosened — it is the tolerance the whole stack is already qualified at, and
inventing a wider one for the device path would retire the very check this is.
But a future change that moves the tower's arithmetic even slightly on CUDA
will fail here first, and the correct response is to investigate the change,
not the constant.

### plato: the memory measurement, and the test that reported zero

The first version of `the_measured_device_peak_is_within_ten_percent_of_the_charged_term`
reported **"measured device peak 0 bytes, charged 1100000000"** on an L40S —
and passed, because zero is under any ceiling. Both flaws are recorded here
rather than quietly fixed, because both are easy to reintroduce:

1. **It warmed up first.** candle's CUDA allocator does not return freed blocks
   to the driver, so the warm-up left the whole peak already reserved: the
   baseline was sampled with the memory outstanding and the measured run reused
   the same blocks. Run 1 dropped 643,825,664 bytes and never recovered them;
   runs 2 and 3 dipped by nothing.
2. **It measured the wrong function.** `compose_identity_tokens_observed` takes
   an already-computed ArcFace vector and an already-aligned crop, so SCRFD and
   ArcFace never ran and their weights were never placed — on a measurement
   whose charge covers the whole extraction.

Measured properly — through `extract_identity_embeddings`, from a fresh CUDA
context, cold, no warm-up, taking the allocator high-water — the whole-extraction
device peak is **637,534,208 bytes**, reproduced bit-for-bit at `d9cf0ebe`
(643,825,664 at `3163ed47`; 945,815,552 at the earlier head `99889dd5`, where the ~302 MB
difference is `glintr100` and its activations no longer coexisting with the
tower, after the face stack's lifetime was narrowed to the photographs that
actually need detecting). Per-stage deltas: parser
+33.6 MB, `eva-ctor` +570.4 MB (the f16 tower, as designed — the widening never
appears), `eva-forward` +6.3 MB, `idformer-build` +33.6 MB.

Three cold runs came in at **637,534,208 / 643,825,664 / 643,825,664** — a
6.3 MB spread — and that spread is why the CHECK, not the constant, had to
change. A naive `measured >= 0.9 x charged` floors at 630,000,000, only 7.5 MB
under the smallest observation, and raising the charge makes it worse rather
than better: 710,000,000 floors at 639,000,000 and fails on a run already
taken. The tension is structural — the measurement is one photograph and the
charge budgets for `ID_IMAGES_MAX` — so the over-charge half of the test now
nets out `EXTRACTION_DEVICE_MULTI_IMAGE_ALLOWANCE_BYTES` first, leaving a real
±10% band around a single-photograph run (floor 597,600,000, ~40 MB of margin).
The coverage half still uses the full charge, because under-charging is what
OOMs a card mid-extraction.

So the pre-measurement 1.1 GB derivation was **1.71x the truth**: it assumed
SCRFD and ArcFace stayed resident beside the tower and added ~120 MB of
activation headroom, when in fact the allocator hands each stage's freed blocks
to the next. `EXTRACTION_DEVICE_PEAK_BYTES` is **700,000,000**: it covers the
largest admissible set (643,825,664 + 36 MB of retained hidden states for the
`ID_IMAGES_MAX - 1` further photographs = 679,825,664) with ~20 MB left for
allocator block rounding, which differs by driver and card. Over-charging by
1.71x would park renders an L40S could run, which is exactly the mistake #1223
named when it removed #1220's placeholder.

### plato: the render itself

Not only the harness. `mold run --local flux-dev:q8` with a reference
photograph produced a print scoring **cosine 0.6259** against it — inside
PuLID's own reported 0.6-0.8 band and well above InsightFace's 0.28
same-person threshold. A server render on a private `mold serve` was
**byte-identical** to the forced-local one, which is the local/remote parity
`resolve_identity_for_lease` being one function is supposed to give, and the
new phase shows up in the client as `✓ Extracting face identity [10.4s]`.

---

## 5. Phase 2 — the plan the phase-1 numbers actually justify

> **SHIPPED.** Every item below is implemented; §4b records what it measured
> and where the plan was wrong. Read this section as the plan it was — the two
> predictions it got wrong (the `eva-build` split, and "the tower wants its
> stored dtype" as the fix for the expensive half) are corrected inline and in
> §4b rather than edited away, for the same reason §1's falsified premise was
> left standing. One item is **not** verified: the CUDA device-peak
> measurement, because plato was again out of budget. §4b says so explicitly.

Phase 1 kept extraction exactly where it was: CPU, at admission, before any
lease. §4 says that is now the expensive decision, not a free one. This section
is the concrete follow-up, written against the measured table rather than
against the issue title. It is **not** implemented on the phase-1 branch, for a
reason worth stating: everything below edits
`crates/mold-server/src/identity_extraction.rs`, which #1226 is rewriting for
multi-photo and true-CFG conditioning. Phase 2 rebases onto #1226; landing both
into that file at once buys nothing and costs a merge.

### What moves, and what deliberately does not

| stage | p50 today | phase 2 |
| --- | ---: | --- |
| `eva-build` | 1268 ms | **the first target, and it is not a device question.** Roughly half is `derived_artifact_is_authentic` re-hashing 609 MB and half is widening f16 → f32 into ~1.2 GB. Measure that split first (one more `ComposeStage`); the hash half wants a process-lifetime memo keyed on `(dev, ino, size, mtime)` **plus** a re-verify on any mismatch — never a bare "we checked once" flag, because the whole point of the compiled-in `DERIVED_SHA256` is that the file may change under us — and the widening half wants the tower built at its stored f16/bf16 dtype on a device that has one. <br><br> **MEASURED: the split is 1,124 / 126, not half and half.** The memo is the whole win (1,124 → 51 ms) and the dtype change is worth 126 ms. Both shipped, and the instruction to measure the split before choosing is what stopped the cheap half being mistaken for the expensive one. §4b has the table. The memo key also carries `ctime` and the pin, neither of which this line named. |
| `eva-forward` | 969 ms | **moves to the leased device.** ~300 GFLOP of dense f32 matmul is the canonical GPU workload; candle already runs this tower on any device, and phase 1 is what makes the whole stack device-capable. |
| `idformer-fwd` | 194 ms | moves with it — same `VarBuilder`, same device, no separate decision. |
| `scrfd` / `arcface` | 360 ms | move last, or not at all. They are 13% of the extraction and their kernels are small; taking them along is nearly free once the device exists, but they do not justify the lease-ordering change on their own. That was §1's mistake and phase 2 must not repeat it. |
| `idformer-build` | 46 ms | leave alone. |

### Where it runs, and the ordering that makes it legal

Extraction moves from `variant_dependencies::prepare_inputs_for_devices`
(pre-lease) to **inside the leased job, first, before prompt encode** — the
`ProgressPhase::IdentityExtract` arm §3 designs, with the
`EstimatePhaseTimings` / `EstimateBucket` fields and the schema bump §3
enumerates. Only the sibling that actually performs the once-per-parent
extraction records a sample; every other sibling's `identity_extract_ms` stays
`None`, exactly as `cold_load_ms` already does on a warm reuse.

`ExtractionSlot`'s bespoke semaphore-plus-`ram_snapshot()` gate is **retired**,
not run in parallel with the ledger. Its module doc's own justification —
"extraction happens strictly BEFORE this job has a lease" — stops being true the
moment this lands, and two admission-time memory gates that can disagree is
worse than one.

### The drop-before-adapter rule

Non-negotiable, and the reason this is a phase and not a residency change: the
tower and the IDFormer are **built, forwarded, and fully released** before
`flux::identity::EngineIdentityState` begins the adapter's ordinary residency
for that dispatch. The extraction is a strictly earlier, strictly disjoint
phase of the same lease — never a third permanent resident alongside the
~1.14 GB adapter and the transformer's own peak. Phase 1's measurement makes
this concrete: the tower is ~609 MB at rest and ~1.2 GB widened, which is not
memory a conditioned FLUX render has spare.

### CUDA versus Metal are different problems

- **CUDA**: a discrete VRAM budget. The tower's transient peak is charged
  against the frozen plan's grant and released before the adapter loads. This
  is the ordinary case and the ledger discipline already exists for it.
- **Metal**: unified memory, so "moving to the device" does not move bytes off
  the host, and mold's existing rule is that Metal reserves no host RAM
  separately — its host claim rides the unified device gate. Phase 2 must
  therefore charge the tower ONCE on Apple Silicon, through that gate, not
  once to the host ledger and once to the device. It is also the host where
  phase 1 measured everything, so a Metal `eva-forward` number is the direct
  before/after: 969 ms on the CPU of the same machine.

### Acceptance

Stated over the **whole extraction**, which is what §4 showed the previous
criterion should have been stated over all along. **Results in §4b**; one
clause below did not survive contact and is corrected there rather than here:
"byte-identical to the CPU path's" is not achievable across devices at all —
different kernels reassociate — so the criterion that shipped is the
already-recorded 5e-5 whole-stack tolerance, met at a measured 3.82e-5. The
consequence that clause was worried about is handled instead by the cache key
carrying no device (§2), so one fleet still has one fingerprint per face.

- **Speed**: whole-extraction p95 at least **25% under 2,863.9 ms**, i.e.
  **≤ 2,147.9 ms** on halcyon under `pulid_face_probe bench --full` at the
  5-warmup / 20-run protocol, with the load average reported. Add
  `BASELINE_HALCYON_FULL_P95_MS` beside the existing constants and a
  `--regress-against-full` that reads it, so the check stays mechanical. Take
  plato too — it has four L40S and phase 2 is a device change, so skipping it
  a second time would leave the interesting host unmeasured.
- **Parity**: unchanged tolerances, and one addition — the device path must
  produce a `FrozenIdentityEmbedding` byte-identical to the CPU path's for the
  same photo and assets, or the fingerprint stops identifying an identity and
  every cache and provenance claim built on it breaks.
- **Ordering**: a test proving the tower is released before the adapter is
  resident, not merely that both fit.

---

## Summary of what this issue does and does not decide

- **Decides**: hand-port SCRFD + iResNet100 into candle (§1), keep it
  CPU-resident at today's admission-time call site for milestone 1, add a
  small in-process per-photo LRU cache of final IDFormer output keyed on
  photo + pipeline-version + all four asset SHAs (§2), and extend
  `pulid_face_probe` to measure and gate the EVA/IDFormer half nobody has
  measured yet (§4).
- **Designs but defers**: GPU dispatch of the whole extraction stack on the
  render's leased device, the `ExtractionSlot`-to-`HostMemoryLedger` fold-in
  it implies, and the `ProgressPhase::IdentityExtract` scheduler phase +
  schema v21 bump that would give it learned runtime evidence (§3) — gated on
  §4's extended measurement actually showing GPU dispatch is worth the
  lease-ordering change, not assumed from the issue title alone. **All of this
  shipped in phase 2**, at schema v22 rather than v21 (the tree had moved),
  and §4b records the result.

### What phase 2 shipped

- **Item 1, the `eva-build` tax**: measured first (three new decomposing
  `ComposeStage`s), then memoized. `eva-auth` 1,124 → 51 ms. The prediction
  about which half was expensive was wrong and §4b says so.
- **Item 2, post-lease device dispatch**: the whole extraction moved from
  `variant_dependencies::prepare_inputs_for_devices` to inside the leased job,
  before the model load, as `ProgressPhase::IdentityExtract` with schema v22's
  `ewma_identity_extract_ms`. `ExtractionSlot` retired in favour of the frozen
  plan's own ledger discipline; `EXTRACTION_DEVICE_PEAK_BYTES` charged as its
  own named term, once through the unified gate on Metal.
- **Item 3, the per-photograph cache**: §2's design, placed inside
  `extract_identity_embeddings` rather than at the server call site, so a set
  with two cached references and one new one extracts exactly the new one. Cold
  2,184.8 ms → warm 1.8 ms.
- **Item 4, qualification**: `--device`, `--regress-against-full`, the
  decomposed rows, and the numbers in §4b. plato skipped again, named again.

### A known gap: `EXTRACTION_HOST_PEAK_BYTES` is now documentation

It was the figure `ExtractionSlot::acquire()` charged against a fresh
`ram_snapshot()`. With the slot retired (§4b, "the cache is single-flight")
nothing reads it as a gate any more — the host-placement path is charged the
ordinary way, through the identity artifacts' own `is_host_only` component
roles and their pinned sizes, and CUDA's device-resident path is charged
separately through `EXTRACTION_DEVICE_PEAK_BYTES`. The constant still
describes the `ExtractionPlacement::Host` arm correctly (a CPU placement
really does peak there) and `mold_inference` still derives it from the
artifacts' own sizes via `the_charged_peaks_match_their_measurements`, so it
is not stale — just no longer load-bearing for admission. A follow-up could
narrow it to the private-copy figure the device path actually reads, since
that is smaller than the full CPU-resident peak this constant states.

### What phase 1 actually shipped, and what it changed about the above

- **Shipped**: the §1 hand port (parity-exact) and the §4 harness, with the
  measurements in §4 and §5's plan derived from them.
- **Not shipped, deliberately**: §2's embedding cache. It lives in
  `mold-server/src/identity_extraction.rs`, which
  [#1226](https://github.com/utensils/mold/issues/1226) is rewriting for
  multi-photo and true-CFG conditioning; landing a cache into that file
  concurrently would conflict for no benefit, and §2's own design already
  depends on #1226's post-IDFormer averaging. It rebases on top.
- **Changed**: §3's gating condition is satisfied, from the other direction
  than expected. §4 was supposed to decide "is GPU dispatch worth the
  lease-ordering change"; the answer is yes, but the case rests on the EVA
  tower rather than on SCRFD/ArcFace, and half of the tower's cost is setup
  rather than arithmetic. §5 restates the phase accordingly.
