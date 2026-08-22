# PuLID performance: GPU face extraction, embedding cache, and qualification

Issue [#1227](https://github.com/utensils/mold/issues/1227). This is the
research-and-design record for the perf half of face-identity conditioning:
whether extraction should move off `candle-onnx`/CPU, a cross-request
identity-embedding cache, EVA-CLIP residency under a GPU path, and the
benchmark protocol that qualifies whichever of these ships.

This is a **decision record, written before code**, exactly as
[#1222](https://github.com/utensils/mold/issues/1222)'s Step 0 was. Nothing in
this document changes Rust — see `docs/architecture/pulid-face-extraction.md`
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

**Ship option A (hand-ported SCRFD + iResNet100 in candle), but keep it
CPU-resident at today's call site for milestone 1.** Concretely: replace only
the internals of `IdentityExtractor` (`identity/mod.rs`,
`identity/scrfd.rs`'s backbone forward, `identity/arcface.rs`'s backbone
forward) from `candle-onnx::simple_eval` calls to ordinary
`VarBuilder`-loaded candle-core/candle-nn forward passes — loading the ONNX
initializers once (candle-onnx's tensor-proto parsing can still be reused
*at load time only*, to pull weights out, exactly as `onnx_inventory.rs`
already introspects the graphs, or the weights can be converted once to
safetensors mirroring `encoders::eva_clip_convert`'s pattern) and calling
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

---

## 2. Cross-request identity-embedding cache

### Where it lives

`crates/mold-server/src/identity_extraction.rs::resolve_identity_embedding`,
immediately before `ExtractionSlot::acquire()` (line 192) — a cache hit skips
the slot, the host-memory gate, and the extraction entirely; nothing is
serialized or counted for a hit, because nothing is computed. This is the
single call site every admission path already funnels through
(`request_resolves_identity`, `EXTRACTIONS`, the `#[cfg(test)] test_stub`
seam), so the cache needs no second integration point.

### Key composition

Enumerated exactly, because the task calls for the exact asset SHAs and
version constants involved:

| Component | Source | Available before extraction runs? |
| --- | --- | --- |
| `sha256(id_image bytes)` | `mold_core::identity::id_image_sha256` (`identity.rs:517-522`) | Yes — pure function of the request |
| **A new `IDENTITY_PIPELINE_VERSION: u32`** | Does not exist today; add to `mold_core::identity` | Yes — a compiled constant |
| Adapter SHA | `mold_core::pulid_assets::pulid_manifest()`'s pin for `ModelComponent::IdentityAdapter` — the same read `extraction.rs::adapter_sha256()` (lines 133-141) already performs | Yes — a manifest pin, no file read |
| Vision (derived tower) SHA | `crate::encoders::eva_clip_convert::DERIVED_SHA256` — a compiled constant | Yes |
| Face-detector SHA | `onnx_graph::pinned_artifact(ModelComponent::FaceDetector)`'s pin | Yes — the manifest pin, not the post-load `det.sha256` (which is checked equal to the pin or the load fails, so they never disagree) |
| Face-recognizer SHA | `onnx_graph::pinned_artifact(ModelComponent::FaceRecognizer)`'s pin | Yes, same reasoning |

This is deliberately the same four-asset shape `IdentityAssetDigests` already
carries (`identity.rs:380-391`) — but that struct is populated **after**
extraction, from what actually ran. The cache key needs the same four
digests **before** extraction runs, which is why the table above resolves
each one from its manifest/compiled-constant source rather than from an
`IdentityAssetDigests` a request hasn't produced yet. Compose the key exactly
as `fingerprint_of` already composes the *output* fingerprint
(`identity.rs:496-513`) — domain-separated, newline/NUL-joined SHA-256 — but
over these six inputs instead of the extracted values:

```
sha256("mold.identity.cache.v1\0"
       || id_image_sha256 || "\0"
       || IDENTITY_PIPELINE_VERSION.to_le_bytes() || "\0"
       || adapter_sha || "\0" || vision_sha || "\0"
       || face_detector_sha || "\0" || face_recognizer_sha)
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

### Required extension (first implementation-phase deliverable, not this doc)

A second bench mode covering the currently-unmeasured half of §0's finding —
`pulid_face_probe bench --full` (or a separate `bench-full` subcommand) that
additionally warm-repeats `extraction::compose_identity_tokens` (EVA tower +
IDFormer forward, given a fixed aligned 512×512 crop and ArcFace vector so
the bench isolates encode cost from detection) per image, reusing the exact
5-warmup/20-run protocol and `validate_bench_args`'s existing refusal/advisory
logic verbatim. Report a **four-row** stage table (SCRFD / ArcFace / EVA /
IDFormer / per-image total) rather than today's two-row one. This must land
and produce real numbers before any claim of "performance qualification" is
made, because today's gate is silent on what may be the larger half of the
real cost.

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

Add pinned baseline constants beside `LATENCY_BUDGET_MS` in
`pulid_face_probe.rs` — `BASELINE_HALCYON_P95_MS: f64 = 415.7`,
`BASELINE_PLATO_P95_MS: f64 = 1574.5` (the SCRFD+ArcFace numbers this doc
cites from `pulid-face-extraction.md`; add the EVA/IDFormer baselines once
the extended bench above produces them) — and a `--regress-against
<halcyon|plato>` flag that fails (non-zero exit, matching the existing gate's
convention) when the measured p95 exceeds 0.75× the named baseline, i.e. the
literal "p95 ≥ 25% faster" acceptance criterion expressed as a mechanically
checkable ratio against numbers already committed to this repository, rather
than a comment a reviewer has to recompute by hand.

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
  lease-ordering change, not assumed from the issue title alone.
