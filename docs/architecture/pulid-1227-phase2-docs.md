# Staged doc deltas — #1227 phase 2

Everything in this file belongs in a document this branch deliberately did not
edit, because a docs agent is consolidating those in parallel. Nothing here is
new reasoning: the reasoning lives in `docs/architecture/pulid-perf.md` §4b and
§5, in `docs/architecture/pulid.md`'s lifecycle sections, and in the source
doc comments cited beside each item. This is the transcription.

Branch: `perf/pulid-extraction-phase2`. Issue:
[#1227](https://github.com/utensils/mold/issues/1227).

---

## 1. `CLAUDE.md` / `AGENTS.md`

### 1a. Amend the identity-photo invariant

The existing **Identity-photo invariant** bullet is a browser-side policy
statement and stays as it is. Add nothing to it; add the paragraph in 1b as a
sibling bullet instead.

### 1b. New bullet, in the "Non-obvious architectural patterns" list

Place it immediately after the existing **PuLID's identity encoders are a port,
and the EVA02-CLIP release is converted before it is ever loaded** bullet.

> - **PuLID face extraction runs inside the lease, on the render's own device,
>   and it is a typed scheduler phase.** #1223 ran it at admission on the host
>   for two reasons that have both expired: `candle-onnx` could not place a
>   tensor off the CPU (#1227 phase 1 replaced it with resident candle ports),
>   and a pre-lease phase cannot overlap the T5/CLIP encode peak by
>   construction. Phase 2 traded the second for its cost —
>   `docs/architecture/pulid-perf.md` §4 measured one extraction at 2,840 ms of
>   which the EVA02-CLIP tower alone was 79% — and moved the whole stack
>   (detector, recognizer, BiSeNet parser, tower, IDFormer) into the leased job
>   as `ProgressPhase::IdentityExtract`, emitted **before the model load**, not
>   merely before prompt encode. That ordering IS the drop-before-adapter rule:
>   every one of those five is built, forwarded, and fully released in turn, so
>   on a cold load none coexists with the transformer, let alone with the
>   ~1.14 GB adapter `EngineIdentityState` makes resident. A warm engine cache
>   does have the transformer resident, which is exactly why
>   `memory_preflight::IDENTITY_EXTRACTION_VRAM_OVERHEAD_BYTES`
>   (= `mold_core::identity::EXTRACTION_DEVICE_PEAK_BYTES`) is charged
>   ADDITIVELY beside `IDENTITY_VRAM_OVERHEAD_BYTES` rather than as a maximum.
>   Metal charges it ONCE through the unified device gate and nothing to the
>   host ledger; on CUDA the host side is only the private authenticated copy
>   the `VarBuilder` reads from, which the identity artifacts' own
>   `is_host_only` component roles already charge. `ExtractionSlot`'s bespoke
>   semaphore-plus-`ram_snapshot()` is RETIRED, not run beside the ledger — its
>   own justification was "extraction happens strictly BEFORE this job has a
>   lease", which expired with the call site, and two admission-time memory
>   gates that can disagree is worse than one
>   (`no_second_memory_gate_lives_in_this_module` is the structural pin).
>   Serialization comes free: a lease is already exclusive on its device.
>   "Exactly once per parent" is unchanged and still counted — the first
>   sibling to reach a device extracts, the rest are answered by the cache
>   without opening a model, and a re-prepared batch child still carries its
>   parent's frozen value. The forced-local CLI runs the identical
>   `mold_server::identity_extraction::resolve_identity_for_lease` at the
>   identical point, so a CPU-only local device is just the
>   `ExtractionPlacement::Host` arm of the same code. The tower's working dtype
>   is `identity::extraction::eva_working_dtype`: f32 on the host (candle has
>   no narrow CPU kernels, and every committed golden was captured there), f16
>   on a device — the dtype the derived file already stores, and narrower than
>   upstream's own cast (`PuLID/pulid/pipeline_flux.py:60`,
>   `PuLID/app_flux.py:45`). The IDFormer half computes in f32 whatever the
>   tower did. Device and host agree on the final tokens to a measured
>   **3.82e-5** of peak, inside the 5e-5 the whole-stack golden already states;
>   no new tolerance was invented. Measured on halcyon (M4 Max, 5-warmup /
>   20-run): whole extraction **3,073.6 → 1,907.3 ms** on the CPU (the memo
>   alone) and **395.3 ms** on Metal. CUDA is designed for and unmeasured; see
>   `pulid-perf.md` §4b.
>
> - **A verified derived artifact is proven once per process, never once per
>   request.** `encoders::pickle_convert::open_authenticated` memoizes its
>   SHA-256 pass on the file's own `(dev, ino, size, mtime, ctime)` identity
>   PLUS the pin, re-read from the SAME retained descriptor after the private
>   copy is taken; a mismatch falls through to an ordinary full read-and-hash.
>   This is `mold_core::download::pinned_file_digest`'s memo applied to the one
>   artifact that reads its bytes privately, and `ctime` is the load-bearing
>   field — `utimensat` lets an owner set `mtime` to anything, but no userspace
>   call holds `ctime` still across a write. The private-read-then-build
>   contract in `AuthenticatedArtifact` is UNCHANGED: the copy is still what the
>   `VarBuilder` reads, and the memo only ever removes a pass over it. Never
>   degrade it to a "we checked once" flag, and never key it on the pathname —
>   the whole point of the compiled-in `DERIVED_SHA256` is that a group-writable
>   model root may change the file underneath us. It is worth **1,073 ms of a
>   3,074 ms extraction** on halcyon, because the 609 MB re-hash was the single
>   largest line item in the entire PuLID pipeline and the drop-and-reload rule
>   re-paid it on every conditioned request.
>
> - **One identity photograph is extracted once per process, and the cache key
>   is the whole pipeline.** `mold_inference::identity::extraction` holds a
>   16-entry LRU (fixed 256 KiB each, ~4.2 MB) of final `[1, 32, 2048]` tokens
>   keyed by `mold_core::identity::identity_cache_key` — the photograph's
>   SHA-256, `IDENTITY_PIPELINE_VERSION`, and all five asset digests, composed
>   the way `fingerprint_of` composes the output fingerprint but over inputs
>   knowable BEFORE anything is opened (`pinned_asset_digests`). **Bump
>   `IDENTITY_PIPELINE_VERSION` in the same PR as any semantic change to SCRFD,
>   the alignment, the warp, ArcFace, the EVA preprocessing, the tower, the
>   mask, or the IDFormer** — it is the one invalidation case that is not
>   structural, because none of those files moves. The key carries no device,
>   deliberately: host and device agree to a measured tolerance rather than
>   bit-for-bit, so keying on the device would give one fleet several
>   fingerprints for one face. The unconditional true-CFG identity is a
>   SEPARATE degenerate memo on the adapter digest, because it is a pure
>   function of the checkpoint (`pipeline_flux.py:188-192`) and would otherwise
>   force a cached request to materialize the 605 MB `pulid_encoder.*` half for
>   a tensor that never varies; that is why its zeros are shaped from the
>   tower's declared geometry rather than a live output. In-process only and
>   never persisted: these are a biometric derivative, and `pulid-perf.md` §2's
>   argument against persistence is a retention story, not a performance one.
>   Cold 2,184.8 ms → warm 1.8 ms, opening no models at all.
>
> - **The identity cache is SINGLE-FLIGHT, because the callers are sibling
>   GPU threads.** A plain get/put was right while extraction ran once per
>   parent at admission; post-lease, a `batch_size = 4` parent's children are
>   dispatched together and all four miss a cold cache in the window between
>   the get and the put — N times the work, and N embeddings differing at the
>   measured 3.82e-5 device tolerance, i.e. four siblings of one print
>   conditioned on four slightly different faces with four different frozen
>   fingerprints. The miss path therefore takes a per-key lock and a waiter
>   **re-reads the cache and takes the winner's tokens** rather than computing
>   its own; sibling embeddings are byte-identical by construction. Locks are
>   taken in SORTED key order (a multi-photograph set takes several, and two
>   requests sharing a subset in different orders would deadlock), the
>   unconditional identity has its own flight key because its value reaches the
>   fingerprint, and a failed flight stores nothing and releases the key —
>   never negative-cache an extraction failure. The children need no frozen key
>   from the parent: the key is a pure function of the bytes each child already
>   carries, so content addressing already gives them one, and a second copy in
>   the plan would be an authority that could disagree with those bytes.
>   `identity_extraction_count` counts what was COMPOSED and
>   `ResolvedIdentity::extracted` carries that to the server, so the
>   once-per-parent counter is checkable again AND
>   `ProgressPhase::IdentityExtract` is emitted only by the sibling that did
>   the work — a cache hit reporting its ~2 ms would drag
>   `ewma_identity_extract_ms` to a figure no cold request could meet.
>   An uncond-only miss (first true-CFG request after an ordinary one) loads
>   the IDFormer alone and never the face stack: 60.6 ms against ~340 ms.

### 1c. Amend the metadata-DB paragraph

The **Metadata DB** section's schema sentence needs the bump. Current text ends
`... (schema v20; FK ...)` for the organization tables and states migrations are
forward-only. Add to the same paragraph:

> Schema **v22** adds `scheduler_estimates.ewma_identity_extract_ms`, the
> learned runtime for `ProgressPhase::IdentityExtract` (#1227 phase 2),
> appended after every existing column so no `SELECT` index moves.

### 1d. Amend the "Learned scheduling separates setup from execution" bullet

Its phase list currently reads "typed cold-load, warm-reload, prompt-encode,
denoise, legacy VAE, visual-decode, audio-decode, mux, and upscale phases".
Insert `identity-extract` between "warm-reload" and "prompt-encode", and append
to the schema clause: `schema v22 adds face-identity extraction`.

---

## 2. `crates/mold-cli/src/skill/SKILL.md`

Wherever the skill describes identity conditioning's cost or timing, replace any
claim that extraction runs on the CPU at admission. Suggested replacement prose:

> Face-identity extraction runs on the same GPU that renders the print, as the
> first thing that job does, and is reported as an **Extracting face identity**
> stage. One extraction is ~0.4 s on an M4 Max GPU and ~1.9 s on a CPU-only
> host; a photograph already used in this server run is reused from memory in
> under 2 ms. It needs ~1.1 GB of device memory while it runs, which the
> scheduler reserves as part of the job's plan and which is released before the
> checkpoint loads — separate from the ~1.25 GB the identity adapter holds for
> the whole denoise.

`pulid_face_probe` gained `--device cpu|metal|cuda` and
`--regress-against-full halcyon|plato` if the skill documents that binary.

---

## 3. `README.md`

If the README quotes identity-conditioning cost or says extraction is a CPU
step, use the one-line form:

> Identity extraction runs on the rendering GPU (~0.4 s on an M4 Max, ~1.9 s
> CPU-only) and is cached per photograph for the life of the server.

---

## 4. Already done on this branch — do NOT re-edit

These files are owned by this branch and are current:

- `docs/architecture/pulid-perf.md` — §4b is the phase-2 measurement record;
  §5 is marked SHIPPED with its two wrong predictions corrected inline.
- `crates/mold-inference/testdata/pulid/README.md` — the device-parity
  tolerance row (5e-5 reused, measured worst 3.82e-5).
- `website/guide/identity.md` — "What actually happens" now says GPU, with a
  "When it runs, and what it costs" subsection.
- `website/guide/configuration.md` — the placement/estimates paragraph names
  the identity-extract phase, schema v22, the Metal unified-memory
  reservation, and the in-memory cache.
- `changelog.d/pulid-perf-gpu.md`.

## 5. Known gaps a reviewer should see

- **CUDA is measured** (plato, 4x L40S, at `3163ed47`): whole extraction
  573.2 ms, parity worst 4.908e-5 against the 5e-5 budget, device peak
  643,825,664 bytes against a 700,000,000 charge, render cosine 0.6259.
  `pulid-perf.md` §4b's plato subsections carry the tables. Two things a
  reviewer should carry forward: **CUDA uses 98% of the parity budget**
  (Metal used 76%), so a future change to the tower's CUDA arithmetic fails
  there first and the constant must not be loosened in response; and the
  device-peak test must run cold on a fresh context through
  `extract_identity_embeddings`, because candle's CUDA allocator never returns
  freed blocks to the driver and a warmed-up measurement reports zero.
- **`EXTRACTION_HOST_PEAK_BYTES` is now documentation.** It was the figure
  `ExtractionSlot` charged; with the slot retired nothing reads it as a gate.
  It still describes the host path correctly (a CPU placement really does peak
  there) and `mold_inference` still derives it from the artifacts' own sizes,
  but a follow-up could narrow it to the private-copy figure for the device
  path.
