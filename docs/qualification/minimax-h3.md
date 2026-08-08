# MiniMax H3 qualification and authorization status

- Status: **unavailable; compliance-gated**
- Evidence snapshot: **2026-08-08 05:14 UTC, Mold main `50f28de3`**
- Authorization owner: [issue #831](https://github.com/utensils/mold/issues/831)
- Final qualification owner: [issue #827](https://github.com/utensils/mold/issues/827)

This is an engineering status record and the acceptance plan for a possible
future MiniMax H3 qualification. It is not a support announcement, a license
grant, or legal advice. Mold main contains fail-closed H3 contracts, primitives,
factory authorities, and dispatch seams, but does not currently advertise,
download, load, or execute H3.

The research used public implementation source and small textual repository
metadata only. The reviewed
[MiniMax H3 Community License](https://huggingface.co/MiniMaxAI/MiniMax-H3/blob/bfc8ed0353f5a9733be73e6b2c98ec0948195b86/LICENSE)
defines a default territory that excludes the United States, European Union,
United Kingdom, and Republic of Korea. The current development environment is
in an excluded territory and has no accepted written authorization record.
Accordingly, no binary H3 checkpoint or other model payload, production
safetensors header range, generated media, or real-checkpoint UAT was
downloaded, read, executed, or retained for the work recorded here. The planned
external-volume UAT home was not used for H3 artifacts.

See the separate [authorization decision record](../architecture/minimax-h3-authorization.md)
for the reviewed license identity, activation requirements, review ownership,
and revocation procedure.

Terms in this record are deliberately narrow:

- **Source/metadata evidence** means public source code and small textual
  repository listings, configuration, index, and license files. It excludes
  Git LFS objects, checkpoint shards, and production safetensors headers.
- **Synthetic evidence** means deterministic fixtures or tensors created by
  the tests. It does not mean a real model was loaded.
- **Exact head** identifies a Git snapshot only. It does not mean its checks
  passed or that the snapshot is merged, authorized, or qualified.
- **Available** and **qualified** are reserved for an authorized runtime and an
  exact release artifact that has passed the future acceptance matrix. No
  current H3 state meets that definition.

## Fail-closed product behavior

[`mold-core::model_policy`](../../crates/mold-core/src/model_policy.rs) is the
single authorization authority. Until issue #831 records approved written
authority, all of these statements must remain true:

- H3 is absent from ordinary catalog, install, and model-family discovery.
- Raw `hf:` identities, aliases, resolved `cv:` metadata, configured families,
  nested generation artifacts, and root-relative model paths are checked
  before network transfer, persistence, placement, or queue admission.
- Existing weight files do not imply authorization.
- `runtime_available` remains `false`; hidden manifests are identity and
  accounting contracts, not runnable model registrations.
- An environment variable, HTTP header, client switch, weight presence, or
  inferred location cannot open the gate.

Server mutation paths return HTTP **451 Unavailable For Legal Reasons** with
the stable machine-readable code:

```json
{
  "code": "MINIMAX_H3_AUTHORIZATION_REQUIRED",
  "error": "MiniMax H3 support is compliance-gated and is not activated in this build (...)"
}
```

The exact message also links issue #831. Client capabilities project the same
restriction; they do not implement a second policy. Authentication and input
validation may still run before the gate where required for security, but H3
bytes must not be staged and work must not be queued.

## Engineering implementation ledger

This ledger separates code presence from qualification and release support.
Every merged row remains subordinate to the authorization gate above, and none
of the pending rows changes the current product status.

| State                     | Scope                                                                                                                                                                                                                                                                                 | Exact evidence                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                               | Qualification effect                                                                                                   |
| ------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------- |
| Merged on main            | Compliance policy, synthetic conformance, dual samplers, synchronized A/V mux, core request contracts, AudioVAE, Qwen layer 50, visual VAE, ordered-reference contracts, DiT, Comfy schema validation, secure ingress, frozen admission/block streaming, and T2VA/FL2VA orchestration | [#843](https://github.com/utensils/mold/pull/843), [#844](https://github.com/utensils/mold/pull/844), [#845](https://github.com/utensils/mold/pull/845), [#846](https://github.com/utensils/mold/pull/846), [#848](https://github.com/utensils/mold/pull/848), [#849](https://github.com/utensils/mold/pull/849), [#850](https://github.com/utensils/mold/pull/850), [#851](https://github.com/utensils/mold/pull/851), [#852](https://github.com/utensils/mold/pull/852), [#853](https://github.com/utensils/mold/pull/853), [#854](https://github.com/utensils/mold/pull/854), [#855](https://github.com/utensils/mold/pull/855), [#856](https://github.com/utensils/mold/pull/856), and [#858](https://github.com/utensils/mold/pull/858) | Weight-free structure and deterministic synthetic evidence only                                                        |
| Merged on main            | Memory-efficient full attention, ordered Ref2VA orchestration, portable Comfy quantization, fail-closed backend adapter, per-layer comparator, frozen factory authority, FL2VA dispatch, scaled FP8/Qwen INT8 execution, and Ref2VA dispatch                                          | [#857](https://github.com/utensils/mold/pull/857), [#859](https://github.com/utensils/mold/pull/859), [#860](https://github.com/utensils/mold/pull/860), [#861](https://github.com/utensils/mold/pull/861), [#862](https://github.com/utensils/mold/pull/862), [#863](https://github.com/utensils/mold/pull/863), [#864](https://github.com/utensils/mold/pull/864), [#865](https://github.com/utensils/mold/pull/865), and [#866](https://github.com/utensils/mold/pull/866)                                                                                                                                                                                                                                                                | Both task dispatch seams exist, but frozen `runtime_available = false` authority rejects activation before a real load |
| Merged on main            | Web, desktop, iPhone, CLI, TUI, and Discord fail-closed authoring; gated inventory/readiness; durable reference media; canonical upload leases; exact recovery identity; closing-frame and media provenance                                                                           | [#867](https://github.com/utensils/mold/pull/867) and [#868](https://github.com/utensils/mold/pull/868), merged through main `50f28de3c88eb7058ca16061fa2753895aa3fc5e`                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                      | Authoring and provenance contracts only; every surface remains unavailable while runtime capability is false           |
| Open release-candidate PR | Isolated SM89 H3 FlashAttention feature, synthetic qualification probe, release contracts, distribution-fixture provenance hardening, and explicit exclusion from ordinary release binaries                                                                                           | [#871](https://github.com/utensils/mold/pull/871), exact head `a249455c8355a3708279795614a16dde3f819ab4`, source tree `b78bacfd0f7c509951c13b4ecd4eae3b51054fe3`. At the evidence timestamp it was open, ready for review, based on main `50f28de3`, and its newly queued GitHub checks had not completed. Plato synthetic prequalification is recorded below. The ordinary shipping-binary fixture was still building, so release-exclusion verification against that fixture remained pending.                                                                                                                                                                                                                                             | No runtime activation, published-binary, real-checkpoint, output-quality, or licensed UAT claim                        |

The factory and dispatch seams are deliberately useful while unavailable: they
let CI prove that model identity, frozen artifact authority, ordered
conditioning, admission, worker handoff, and terminal errors stay consistent.
They must not be described as an executable H3 backend until authorization,
real artifact identity, numerical parity, memory qualification, and release
artifact inspection all pass.

The ledger is a point-in-time audit, not a merge plan. Merged authoring code
does not qualify a model, and an open PR or synthetic GPU check does not become
release evidence. In particular, #871 remains an unmerged, opt-in development
candidate. Its exact ordinary shipping-binary exclusion fixture and GitHub
checks were still pending at the evidence timestamp.

## Open-source scope and closed system modules

The pinned
[official implementation README](https://github.com/MiniMax-AI/MiniMax-H3/blob/8d8824efaf94586c0cc9ac7ad8d0723d4d6420ea/README.md)
describes a three-part H3 system. The milestone intentionally has a narrower
boundary:

| Module                  | Milestone status                                                                                          | Mold claim                                                       |
| ----------------------- | --------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------- |
| H3-Base FL2VA           | Open implementation scope: text-to-audio-video plus optional first frame, last frame, or both             | Contract and weight-free primitives only; unavailable            |
| H3-Base Ref2VA          | Open implementation scope: ordered image, video, and audio references within the released limits          | Contract and weight-free primitives only; unavailable            |
| H3-Context-IR           | Closed: hosted multi-stage models/services are not in the open-source release                             | Not implemented; hosted prompt processing is not a parity oracle |
| H3-Regenerate-2K        | Closed: upstream says this module is not yet open-sourced                                                 | Not implemented; no 2K or hosted-workflow parity claim           |
| Native sparse attention | Closed: upstream says the initial release provides full attention and will publish sparse attention later | Not implemented; not an acceptance dependency for H3-Base        |

The open implementation scope is therefore the released 768p-class H3-Base
checkpoints only. A result from MiniMax's hosted full workflow cannot close a
local H3-Base parity row without component-level evidence, and Mold must never
describe Base output as Context-IR or Regenerate-2K parity.

## Revision-locked sources

The machine-readable authority is
[`tests/fixtures/minimax_h3/conformance-manifest.json`](../../tests/fixtures/minimax_h3/conformance-manifest.json).
The Rust family contract repeats the implementation and checkpoint identities
in [`mold-core::minimax_h3`](../../crates/mold-core/src/minimax_h3.rs). Updating
one source requires updating both authorities and re-reviewing the license
gate.

| Source                                                                                                                              | Pinned revision                            | Authority                                                         |
| ----------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------ | ----------------------------------------------------------------- |
| [MiniMax official implementation](https://github.com/MiniMax-AI/MiniMax-H3/tree/8d8824efaf94586c0cc9ac7ad8d0723d4d6420ea)           | `8d8824efaf94586c0cc9ac7ad8d0723d4d6420ea` | Architecture and semantic contract                                |
| [MiniMax official checkpoint repository](https://huggingface.co/MiniMaxAI/MiniMax-H3/tree/bfc8ed0353f5a9733be73e6b2c98ec0948195b86) | `bfc8ed0353f5a9733be73e6b2c98ec0948195b86` | Checkpoint configuration, component indexes, and reviewed license |
| [Diffusers](https://github.com/huggingface/diffusers/tree/9c6a68c32b3b2a64db91800b624d33cec6e25ab8)                                 | `9c6a68c32b3b2a64db91800b624d33cec6e25ab8` | Full-precision executable numerical oracle                        |
| [ComfyUI](https://github.com/Comfy-Org/ComfyUI/tree/a464ac33588ae182f81a090d910cfbf21e255b73)                                       | `a464ac33588ae182f81a090d910cfbf21e255b73` | Pruned/quantized deployment implementation                        |
| [Comfy H3 checkpoint repository](https://huggingface.co/Comfy-Org/MiniMax-H3/tree/eb8a16107c595128b3a578f82d2ce2f75920c355)         | `eb8a16107c595128b3a578f82d2ce2f75920c355` | Pruned/quantized checkpoint schema and file identities            |
| [SGLang](https://github.com/sgl-project/sglang/tree/0c3a76fa0a5bfab410b645f4143e7e8e3cc25c77)                                       | `0c3a76fa0a5bfab410b645f4143e7e8e3cc25c77` | Conditioner/distributed performance reference only                |
| [vLLM-Omni](https://github.com/vllm-project/vllm-omni/tree/3d7fc3b9ba3cac88d579d4dc35b78b0b641675fc)                                | `3d7fc3b9ba3cac88d579d4dc35b78b0b641675fc` | Loader, offload, and CUDA-kernel reference only                   |

Only Diffusers' official BF16/FP32 mixed execution is the current numerical
oracle. Performance references do not become correctness authorities merely
because they are faster.

The two rows named as checkpoint repositories identify revisions and textual
metadata authorities. They do not record a local checkout containing Git LFS
objects, a fetched checkpoint shard, or an opened production safetensors
header. Source-checkout verification against binary model content is not part
of the current evidence.

## Frozen H3-Base media contract

The source contract currently freezes:

- 24 fps output on the `17n+5` frame grid;
- the representative grid points 124, 243, and 362 frames;
- actual media durations of approximately 5.1667, 10.125, and 15.0833 seconds
  at those three points;
- Mold's explicit acceptance of aligned 362-frame nominal-15-second output;
- mandatory synchronized 32 kHz, two-channel stereo audio;
- MP4 output for the synchronized audio-video path; and
- one native output per generation request.

The 362-frame decision is intentionally different from the pinned Diffusers
path, which aligns 360 to 362 and then rejects the result for exceeding 15
seconds. The [conformance guide](./minimax-h3-conformance.md) records the exact
day-zero fixture and external evidence schema.

## Weight layouts and storage facts

The file names and object byte counts below were recorded from public repository
metadata in the hidden, revision-locked manifests in `mold-core::minimax_h3`.
They are planning facts, not evidence that any binary object or header was
downloaded, opened, or hashed locally.

| Layout                                                      | Weight-file footprint                                             | Qualification meaning                                                                     |
| ----------------------------------------------------------- | ----------------------------------------------------------------- | ----------------------------------------------------------------------------------------- |
| Official BF16, one task transformer plus shared Qwen/VAEs   | 144,016,405,316 bytes, reported as approximately **134.125 GiB**  | Full correctness layout for either FL2VA or Ref2VA                                        |
| Official BF16, both task transformers plus shared Qwen/VAEs | 210,296,909,532 bytes, reported as approximately **195.854 GiB**  | Both tasks share Qwen, video VAE, and AudioVAE; this is not concurrent two-task execution |
| Comfy pruned/quantized, one task                            | 42,470,585,471 weight bytes, approximately 42.471 GB / 39.554 GiB | Practical deployment candidate; never the full-precision oracle                           |

Small processor, scheduler, config, and index files are outside those
weight-file totals. A future UAT capacity plan must also reserve space for
download staging, caches, evidence, logs, generated media, and failure
recovery; the table is not a minimum-volume-size recommendation.

The official layout uses a 14-shard BF16 task transformer, the full BF16
Qwen3-VL-32B checkpoint with layer-50 output authority, an official FP32 video
VAE, and an FP32 AudioVAE. The Comfy deployment layout uses one pruned task
transformer, INT8 ConvRot-eligible block matrices, an NVFP4/AWQ layer-50 Qwen
conditioner, an FP16 video VAE, and an FP32 AudioVAE.

Comfy's curve-AdaLN representation replaces the official timestep MLP and
full-width AdaLN projections. Its QKV packing also differs from the official
layout. The current main-branch validator uses synthetic headers and tiny
zero-filled tensors; production H3 headers have not been read, and the
validator deliberately rejects runtime activation.

### Local UAT storage status

`/Volumes/ExternalStorage` was identified as the requested future `MOLD_HOME`
and model-download volume. At the evidence timestamp, `df -h`, `diskutil info`,
and `stat` returned nominal mount metadata for a mounted APFS volume with about
966 GiB available and SMART status reported as Verified. However, time-bounded
`ls` and `find` directory enumeration did not complete; an earlier bounded
create/fsync/read/checksum/delete probe and later filesystem commands also did
not complete. It is not a qualified storage target. Nominal metadata is not a
substitute for usable filesystem operations.

The volume is excluded from Mold UAT. This audit performed no write, repair, or
remount attempt. A separate owner must authorize recovery, after which a fresh
non-H3 storage probe and clean inventory must pass before the volume can even be
proposed for an authorized campaign.

No H3 model download, `MOLD_HOME`, fixture bundle, checkpoint shard or header,
generated output, or other H3 artifact was read from or placed on that volume.
The storage failure is independent of the license restriction: fixing the
volume would not authorize H3 artifact access or execution.

### Parity and approximate-path rules

- Official full-precision outputs and intermediates must be compared with the
  pinned Diffusers oracle before any exact-path claim.
- Comfy INT8/NVFP4 output must be judged with spatial, temporal, audio,
  synchronization, and ordering metrics against the full path. Structural
  schema acceptance is not quality parity.
- The official fresh seed-42 visual-condition posterior is authoritative.
  Comfy's mean-only shortcut cannot silently replace it in Mold's exact path.
- SageAttention, Cache-DiT, FP8 shortcuts, TF32, approximate attention,
  stochastic sampling, and other non-bit-stable accelerations are excluded
  from ground-truth capture.
- A non-parity acceleration may be evaluated only as a separately named tier
  after the exact path passes; it must never supply golden fixtures or close a
  full-precision acceptance row.

## Scheduler and device boundary

The frozen admission contract in
[`h3_admission.rs`](../../crates/mold-server/src/h3_admission.rs) requires
exactly one Scheduler V2 CUDA GPU for one H3 generation. Passing zero or more
than one device is a typed admission error. One backend owns Qwen, the visual
and audio conditions, every DiT block, both decoders, and mux staging on that
frozen route.

Scheduler V2 may eventually distribute independent generation jobs across
multiple GPUs. That is batch/job distribution, not tensor, sequence, or
pipeline parallelism. The milestone contains no cross-GPU model partitioning
claim, and combining VRAM across devices must never make one request appear
feasible.

The curated admission policy currently records a 128 GiB host-RAM
recommendation and enforces a safety floor of `max(8 GiB, 15% of physical
RAM)`. These are admission-policy constants, not measured H3 production peaks.
Exact artifact sizes, header facts, attention workspace, resident block count,
prefetch, dequantization workspace, and every phase allocation must be frozen
before a real run can be admitted.

## Current evidence status

All evidence in this section is weight-free. A passing synthetic test proves a
Rust contract or a tiny kernel route; it does not prove real-checkpoint
correctness, throughput, memory fit, output quality, or release support.

| Surface             | Current evidence                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                          | Current claim                                                                                                                                                                                                                                                                                                                                             |
| ------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| CPU                 | Deterministic frame/sampler/layout/noise fixtures; tiny processor, Qwen, VAE, DiT, ordered FL2VA/Ref2VA pipeline, mux, cancellation, memory-accounting, portable quantization, frozen-factory, and dispatch tests                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                         | Reference and contract testing only; H3 CPU runtime is unsupported                                                                                                                                                                                                                                                                                        |
| CUDA primitives     | Tiny synthetic CPU/CUDA parity and execution tests for the sampler ([#845](https://github.com/utensils/mold/pull/845)), AudioVAE ([#849](https://github.com/utensils/mold/pull/849)), Qwen layer 50 ([#850](https://github.com/utensils/mold/pull/850)), visual VAE ([#851](https://github.com/utensils/mold/pull/851)), DiT ([#853](https://github.com/utensils/mold/pull/853)), portable quantization ([#860](https://github.com/utensils/mold/pull/860), [#865](https://github.com/utensils/mold/pull/865)), and both fail-closed dispatch seams ([#864](https://github.com/utensils/mold/pull/864), [#866](https://github.com/utensils/mold/pull/866))                                                                                                                                                                                                                                                                                                                                                                                                                                | Intended production backend only; no real H3 execution or hardware qualification                                                                                                                                                                                                                                                                          |
| CUDA full attention | [PR #857](https://github.com/utensils/mold/pull/857) records the earlier synthetic dense-reference-to-FlashAttention-v2 work. Exact PR #871 head `a249455c` and source tree `b78bacfd` were then prequalified on Plato (4× L40S with 46,068 MiB each, SM89, driver 595.71.05, CUDA 12.8.93, Rust 1.95) without model data. The H3 release contract and all 36 adversarial CUDA distribution/parser cases passed; the isolated 53-kernel candidate built offline with warnings denied; 10 attention tests passed with 141 filtered. Ten network-isolated BF16 probes ran on GPU UUID `GPU-9ffc81c5-3944-6490-bfd9-f68366f98226`: 5,035–6,014 microseconds, p50 5,213, p95 5,978, maximum absolute delta `0.00048828125` against a `0.02` bound, with stable identities. The candidate binary SHA-256 was `da025539ecaed413afff79767806e47f684c09b3010a1162136339983697d96d`; tracing found no internet socket or H3/model-artifact path, and recorded `model_artifacts_accessed = false` and `runtime_activated = false`. Long-row workspace shapes 37,296 and 107,856 were planning-only. | Synthetic development evidence only. The opt-in candidate correctly carries the compiled-kernel claim and is not a shipping binary. The separate ordinary shipping-binary fixture and exclusion-verifier result were still pending, so this does not establish published-artifact exclusion, real-model correctness, quality, peak memory, or throughput. |
| Metal               | Shared primitive feature compilation and forced-local typecheck only; the H3 capability and admission contracts reject Metal                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                              | Unsupported; no H3 Metal execution or UAT claim                                                                                                                                                                                                                                                                                                           |
| Server/factory      | Hidden identities, HTTP 451 policy, request contracts, secure reference ingress, prepared shapes, frozen single-GPU admission, block-streaming ownership, immutable factory authority, fail-closed backend adapter, and FL2VA/Ref2VA worker dispatch                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                      | No catalog/download activation, admitted real engine, or public H3 runtime at this snapshot                                                                                                                                                                                                                                                               |
| Studio surfaces     | [#867](https://github.com/utensils/mold/pull/867) is merged in main `50f28de3` with web, desktop, and iPhone authoring, recovery, canonical upload, and provenance contracts. Its media-arithmetic fixtures use ordinary generated test media, not H3 output.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                             | Authoring and recovery contracts only; gated readiness cannot make the runtime available                                                                                                                                                                                                                                                                  |
| CLI/TUI/Discord     | [#868](https://github.com/utensils/mold/pull/868) is merged in main `50f28de3` with weight-free ordered-reference authoring, canonical reference leases, and media provenance.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                            | Authoring and provenance contracts only; no runtime activation                                                                                                                                                                                                                                                                                            |
| Real checkpoint     | None                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                      | Unqualified and unavailable                                                                                                                                                                                                                                                                                                                               |

The named Plato runs used only synthetic tensors and fixtures. They are not
real CUDA UAT, not a benchmark of H3, and not evidence about the excluded
external-volume `MOLD_HOME`. No H3 artifact payload or production safetensors
header-range byte was accessed for any row in this document.

## Authorized future UAT acceptance matrix

Do not begin any row below until issue #831 contains a reviewed approval whose
scope covers checkpoint access and execution, automated fixtures, retained
outputs, distribution, local/server/remote-client use, and the intended
territory. The external authorization record and fixture bundle must first
pass the commands in the [conformance guide](./minimax-h3-conformance.md).
Every row is currently **blocked and unattempted**; the table defines future
acceptance evidence and does not report licensed UAT.

| Gate                          | Required campaign                                                                                                                                                                    | Passing evidence                                                                                                                                                                                          |
| ----------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Authorization                 | Revalidate the pinned license/Q&A and obtain written authority for every intended product, territory, user, artifact, and output flow                                                | Approved external authorization record with reviewed source-document hash, scope, owner, expiry/revocation terms, and issue #831 approval                                                                 |
| Clean storage                 | Re-qualify the intended external volume, then use a fresh absolute `MOLD_HOME` and separate external fixture root; never reuse an ordinary Mold home                                 | Bounded create/fsync/read/checksum/delete probe, capacity report, mount identity, clean before/after inventory, and no H3 bytes in the checkout                                                           |
| Artifact identity             | Fetch only the approved task/layout and every pinned companion                                                                                                                       | Exact repository/revision/path/byte count/full SHA-256, component-index hashes, license/NOTICE capture, and no unexpected file                                                                            |
| Full-path numerical parity    | Run tokenizer/processor, Qwen layer 50, visual VAE, AudioVAE, token refiner, transformer block, packed layout, noise allocation, and dual sampler against pinned Diffusers BF16/FP32 | External fixture bundle passes schema/hash validation and every recorded tolerance; no approximate backend contributed a golden value                                                                     |
| T2VA                          | Generate 1344x768 at 124 and 362 frames, plus the 243-frame grid control                                                                                                             | Decoded 24 fps MP4, exact frame count, synchronized 32 kHz stereo audio, stable seed/provenance, phase telemetry, and full-reference quality metrics                                                      |
| FL2VA                         | First-only, last-only, and first+last at the same grid points, including mismatched source aspects                                                                                   | Exact endpoint signatures/order, official resize/crop and fresh seed-42 posterior evidence, preserved boundary behavior, decoded A/V validation, and quality metrics                                      |
| Ref2VA                        | Image-only; video with soundtrack; image+standalone audio; mixed ordered image/video/audio; swapped-order comparison; every count/duration/type failure                              | Exact packed order, modality tags, rotary clocks, soundtrack association, negative-case codes, decoded A/V validation, and order-sensitive quality comparison                                             |
| Comfy deployment path         | Compare the approved pruned INT8/NVFP4 layout with the full path on the same prompts, sources, shapes, seeds, and hardware                                                           | Spatial/temporal perceptual metrics, audio spectral/loudness/channel metrics, exact A/V timing, measured deltas, named accuracy tier, and no exact-parity label unless actually proven                    |
| Memory/performance            | Measure 960x544 and 1344x768 at 124 frames, then 362-frame feasibility, on both the declared high-memory tier and declared streamed consumer tier                                    | Exact GPU/driver/backend, CPU/RAM/storage, artifact identities, attention kernel, resident/prefetch plan, cold/warm setup, per-phase timing, peak VRAM/RAM, and output hashes                             |
| Cancellation                  | Cancel during Qwen load/encode, visual/audio reference encode, block load/prefetch, denoise, visual decode, audio decode, mux, and gallery persistence                               | Typed terminal event, bounded cancellation latency, released Scheduler V2/device/host/staging ownership, no partial gallery row, and documented cleanup                                                   |
| Fault recovery                | Exercise missing/partial/corrupt components, wrong task/layout, unavailable attention, host/VRAM pressure, instance change, network loss, and closed authorization                   | Early typed failure at the owning boundary, no silent fallback/reroute, no queued work after infeasible placement, no leaked path/key, and repeatable retry semantics                                     |
| Single-device authority       | Attempt one valid request on each qualified CUDA target and negative zero/two-device admissions; distribute independent sibling jobs separately                                      | One immutable device/instance/artifact/execution lease per request, typed multi-device rejection, and no aggregate-VRAM feasibility claim                                                                 |
| Surface and clean-install UAT | Exercise CLI, server/API, web, desktop, iPhone remote client, TUI, and Discord only where the approved scope permits                                                                 | Identical capability/policy semantics, truthful unsupported fields, exact-host recovery, decoded gallery media, and clean-install evidence using a future storage root that passed the clean-storage gate |
| Release artifact              | Inspect the exact candidate binaries/archives and repeat the approved smoke from a clean install                                                                                     | Required attention/quantization code present, license/NOTICE/attribution assets present, checksum/provenance bound, all exact-head CI green, and no unauthorized catalog/download exposure                |

The campaign must retain commands, logs, environment, exact Mold commit and
binary hash, device UUID, driver/toolkit, component hashes, decoded media facts,
metrics, and failure evidence outside the repository. Aspirational upstream
performance numbers are not baselines. Regression budgets may be adopted only
after a stable, independently reproduced Mold baseline exists.

## Updating or closing this record

Issue #831 remains open until affirmative authority and all resulting product
obligations are implemented and reviewed. Issue #827 remains open until the
authorized acceptance matrix has real evidence and the exact release artifact
passes. Landing more weight-free code or synthetic CUDA tests does not close
either gate.

Any change to the pinned license, Q&A, source revisions, component identities,
supported territory, execution layout, attention backend, quantization policy,
or public product surface requires updating this record and re-running the
applicable review before release.
