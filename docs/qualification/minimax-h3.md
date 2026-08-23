# MiniMax H3 qualification and authorization status

- Public acquisition status: **compact FL2VA and Ref2VA available by upstream-direct download**
- Runtime status: **compact FL2VA publicly available on the supported SM89 CUDA
  route; the Apple Silicon Metal route is admitted and shipped but not
  hardware-qualified**
- Private qualification status: **authorized under the external private-UAT record**
- Evidence snapshot: **2026-08-08, Mold main `12bbad65`**
- Governance decision: **complete in [issue #831](https://github.com/utensils/mold/issues/831)**
- Final qualification owner: [issue #827](https://github.com/utensils/mold/issues/827)

This is an engineering status record, not legal advice. Mold lists the two
compact Comfy manifests and may download their pinned artifacts directly from
the reviewed upstream repositories. Downloaded weights are not bundled in Mold
releases. Execution remains fail-closed unless the live SM89 CUDA or Apple
Silicon Metal route, exact artifact graph, task, fixed request envelope, and
conservative memory admission profile all validate. Public activation does not
require a private campaign record.

The research used public implementation source and small textual repository
metadata only. The reviewed
[MiniMax H3 Community License](https://huggingface.co/MiniMaxAI/MiniMax-H3/blob/bfc8ed0353f5a9733be73e6b2c98ec0948195b86/LICENSE)
defines a default territory that excludes the United States, European Union,
United Kingdom, and Republic of Korea. On 2026-08-08, the maintainer accepted a
direct attestation that MiniMax authorized H3 integration with Mold. The
[decision record](../architecture/minimax-h3-authorization.md) authorizes use in
every territory and Mold surface, including local, remote-client, shared-server,
hosted, output-distribution, and redistribution paths, while retaining exact
technical runtime admission. Private correspondence and owner-only
qualification evidence remain confidential.
The external campaign contains revision-pinned official and
practical Comfy artifacts whose size and full SHA-256 identities were verified;
an authenticated real-checkpoint block-0 qualification also ran on CUDA. Later
campaigns produced and independently reviewed the exact synchronized H.264/AAC
compact FL2VA output recorded below.

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

## Separate acquisition and execution behavior

[`mold-core::model_policy`](../../crates/mold-core/src/model_policy.rs) owns two
separate authorities. Acquisition permits the reviewed compact upstream files;
runtime activation remains qualification-gated. These statements must remain
true:

- The two compact manifests appear in catalog, install, and model-family
  discovery; official BF16 qualification manifests remain hidden.
- Acquisition accepts only the two registered compact manifest IDs. Raw
  repository IDs, arbitrary catalog recipes, configured H3 aliases, and
  caller-authored manifests remain blocked; the registry fixes every upstream
  repository, revision, filename, destination, byte count, and SHA-256 before
  transfer.
- Existing weight files do not imply technical runtime support.
- Ordinary compact rows do not imply runtime availability. Only the separately
  authenticated additive capability and exact generation-profile row may clear
  the execution restriction.
- An environment variable, HTTP header, client switch, weight presence, or
  inferred location cannot manufacture a technical capability.

The two reviewed compact IDs never return a compliance-authorization error.
Unsupported execution reports the actual task, backend, hardware, artifact,
memory, or request-profile limitation. Raw repositories, arbitrary aliases,
caller-authored manifests, and other unreviewed identities remain unavailable
before network transfer or queueing. Client capabilities consume that same core
policy rather than implementing a second decision.

## Engineering implementation ledger

This ledger separates code presence from qualification and release support.
Every merged row remains subordinate to the technical capability and evidence
gates above, and none changes the current product status by itself.

| State          | Scope                                                                                                                                                                                                                                                                                 | Exact evidence                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                               | Qualification effect                                                                                                   |
| -------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------- |
| Merged on main | Compliance policy, synthetic conformance, dual samplers, synchronized A/V mux, core request contracts, AudioVAE, Qwen layer 50, visual VAE, ordered-reference contracts, DiT, Comfy schema validation, secure ingress, frozen admission/block streaming, and T2VA/FL2VA orchestration | [#843](https://github.com/utensils/mold/pull/843), [#844](https://github.com/utensils/mold/pull/844), [#845](https://github.com/utensils/mold/pull/845), [#846](https://github.com/utensils/mold/pull/846), [#848](https://github.com/utensils/mold/pull/848), [#849](https://github.com/utensils/mold/pull/849), [#850](https://github.com/utensils/mold/pull/850), [#851](https://github.com/utensils/mold/pull/851), [#852](https://github.com/utensils/mold/pull/852), [#853](https://github.com/utensils/mold/pull/853), [#854](https://github.com/utensils/mold/pull/854), [#855](https://github.com/utensils/mold/pull/855), [#856](https://github.com/utensils/mold/pull/856), and [#858](https://github.com/utensils/mold/pull/858) | Weight-free structure and deterministic synthetic evidence only                                                        |
| Merged on main | Memory-efficient full attention, ordered Ref2VA orchestration, portable Comfy quantization, fail-closed backend adapter, per-layer comparator, frozen factory authority, FL2VA dispatch, scaled FP8/Qwen INT8 execution, and Ref2VA dispatch                                          | [#857](https://github.com/utensils/mold/pull/857), [#859](https://github.com/utensils/mold/pull/859), [#860](https://github.com/utensils/mold/pull/860), [#861](https://github.com/utensils/mold/pull/861), [#862](https://github.com/utensils/mold/pull/862), [#863](https://github.com/utensils/mold/pull/863), [#864](https://github.com/utensils/mold/pull/864), [#865](https://github.com/utensils/mold/pull/865), and [#866](https://github.com/utensils/mold/pull/866)                                                                                                                                                                                                                                                                | Both task dispatch seams exist, but frozen `runtime_available = false` authority rejects activation before a real load |
| Merged on main | Web, desktop, iPhone, CLI, TUI, and Discord fail-closed authoring; gated inventory/readiness; durable reference media; canonical upload leases; exact recovery identity; closing-frame and media provenance                                                                           | [#867](https://github.com/utensils/mold/pull/867) and [#868](https://github.com/utensils/mold/pull/868), merged through main `50f28de3c88eb7058ca16061fa2753895aa3fc5e`                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                      | Authoring and provenance contracts only; every surface remains unavailable while runtime capability is false           |
| Merged on main | Isolated SM89 H3 FlashAttention feature, hardened synthetic qualification probe, checked `usize`-to-`u32` launch boundaries, production-dispatch release guard, release contracts, distribution-fixture provenance hardening, and explicit exclusion from ordinary release binaries   | [#871](https://github.com/utensils/mold/pull/871), exact qualified implementation head `c062540270583bc5890a078a296df1d01be2169a`, source tree `b1452b5f0ad740591c9bd0609e8620fb9d1bd3a1`, squash-merged to main as `ff92708650861f800d0c916805b02bb9cc37fdd4`. All applicable exact-head GitHub checks passed. private UAT host synthetic qualification and the same-tree ordinary shipping-feature fixture are recorded below. That fixture passed release-candidate exclusion verification but is deliberately non-publishable.                                                                                                                                                                                                           | No runtime activation, public-release, real-checkpoint, output-quality, or licensed UAT claim                          |

The factory and dispatch seams are deliberately useful while unavailable: they
let CI prove that model identity, frozen artifact authority, ordered
conditioning, admission, worker handoff, and terminal errors stay consistent.
They must not be described as an executable H3 backend until authorization,
real artifact identity, numerical parity, memory qualification, and release
artifact inspection all pass.

The ledger is a point-in-time audit, not a merge plan. Merged code and synthetic
GPU checks do not qualify a model or become release evidence. Although #871 is
merged, its H3 attention feature remains an opt-in development candidate that
ordinary shipping features exclude. Its exact ordinary shipping-feature fixture
proves exclusion for that non-publishable build shape, not a public release. All
applicable GitHub checks passed on the exact qualified implementation head.

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
- the representative grid points 124, 243, and 345 frames;
- actual media durations of approximately 5.1667, 10.125, and 14.375 seconds
  at those three points;
- Mold's cap at the largest grid-aligned output under 15 seconds, 345 frames;
- mandatory synchronized 32 kHz, two-channel stereo audio;
- MP4 output for the synchronized audio-video path; and
- one native output per generation request.

The pinned Diffusers path aligns 360 to 362 and then rejects the result for
exceeding 15 seconds. Mold therefore caps the selectable grid at 345. The
[conformance guide](./minimax-h3-conformance.md) records the exact day-zero
fixture and external evidence schema.

## Weight layouts and storage facts

The file names and object byte counts below were recorded from public repository
metadata in the revision-locked manifests in `mold-core::minimax_h3`. Official
BF16 variants remain hidden; compact variants are visible for acquisition.
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

A third layout, added in #1319, swaps that pruned task transformer for a
third-party pruned NVFP4 one (`Abiray/Minimax-H3-nvfp4-INT4-INT8-Convrot`)
while keeping the same NVFP4/AWQ Qwen conditioner and VAEs; it is
downloadable-only (`runtime_available: false`) and carries none of the
artifact-identity, numerical-parity, or block-execution evidence this section
records for the INT8 layout above.

Executing that layout is a recorded no-go. Issue
[#1318](https://github.com/utensils/mold/issues/1318) was closed on 2026-08-23
because the probe in
[`minimax-h3-nvfp4-layout-probe.md`](minimax-h3-nvfp4-layout-probe.md) measured
the NVFP4 transformer at 6.39x-7.63x the INT8 ConvRot layout's activation-space
distance from the BF16 reference on every one of twelve probed linears. Reopen
it only if a published NVFP4 H3 transformer lands within 2x the INT8 layout's
activation error under that same probe; the weights stay downloadable and
`runtime_available: false` until then.

The sampler is also layout-specific. Official BF16 follows the pinned
Diffusers terminal-inclusive rectified-flow Euler path with 50 grid points (49
transformer evaluations). The released ComfyUI workflow pairs the compact
checkpoint stack with deterministic RES multistep sampling and 20 transformer
evaluations; Mold expresses that as 21 terminal-inclusive grid points. Its RES
history and coefficients operate on Comfy's discrete shift-12 video-sigma
table. Audio is carried in that coordinate for integration, while the mapped
shift-3 sigma remains authoritative for the audio network timestep and native
audio state. The
retained 960×544, 124-frame campaign used two grid points and therefore only
one transformer evaluation. It remains a memory/runtime smoke, not quality
evidence. A fresh compact quality campaign must use the Comfy sampler and 21
grid points before its output can support visual or audio acceptance.

Comfy's curve-AdaLN representation replaces the official timestep MLP and
full-width AdaLN projections. Its QKV packing also differs from the official
layout. Ordinary CI uses synthetic headers and tiny zero-filled tensors.
Separately, the development-only private path authenticated the released
headers and full transformer objects before isolated block-0 CUDA execution.
That narrow evidence does not activate the factory or qualify a full runtime;
`runtime_available` remains false.

### Local UAT storage status

The modes in this section record how the private campaign was operated; they
are not runtime requirements for model storage. Public and private H3 model
weights, configs, support files, model roots, and staging roots are accepted
regardless of Unix owner/group identity or group/other write bits. Runtime
trust comes from pinned content authentication, regular-file and symlink checks,
canonical containment, and descriptor identity fencing. Staged VAE construction
uses retained process descriptors rather than replaceable paths. Owner-only
modes remain required for authorization records and retained private evidence.

The earlier operational exclusion of `/Volumes/ExternalStorage` was superseded
on 2026-08-08 after the maintainer selected it for private H3 UAT and a fresh
qualification campaign passed directory enumeration, create/fsync/read/hash/
delete probes, capacity checks, and repeat artifact inventories. The isolated
campaign root is `/Volumes/ExternalStorage/mold/uat-h3`; its campaign,
`mold-home`, `mold-home/models`, and `compliance` directories are owner-only
mode `0700`, and authorization/model/evidence files are mode `0600`. The volume
retained about 760 GiB free at this snapshot.

The canonical private `MOLD_HOME` is
`/Volumes/ExternalStorage/mold/uat-h3/mold-home`. Fifty official payloads at
revision `bfc8ed0` total 144,028,152,581 bytes; 17 practical Comfy payloads at
revision `eb8a161` total 42,482,090,318 bytes. Repeated size and SHA-256
verification found no missing or partial payload. The external authorization
record and its content-addressed source evidence remain under the sibling
`compliance` directory. This storage result authorizes only the private scope
in the decision record and does not alter the public-product gate.

### Private artifact campaign

After the private authorization decision was accepted on 2026-08-08, the
campaign was established under the qualified external root above. The pinned
Comfy FL2VA transformer, NVFP4-AWQ Qwen3-VL conditioner, FP16 visual VAE, FP32
AudioVAE, and exact small official support files were downloaded directly into
its private `mold-home/models`. All complete objects matched the sizes and
SHA-256 identities in the pinned manifest. No artifact or header was copied
into the repository, and no real-checkpoint report is public evidence.

The development-only qualifier repeats full-content authentication and bounded
structural inspection before any later runtime qualification. It requires the
exact external authorization-record schema and license pins, hashes both the
record and its source document, accepts the configured
`<campaign>/mold-home/models` regardless of its Unix owner or write-mode bits,
and separately requires the sibling `<campaign>/compliance` evidence layout to
remain owner-only. It does not trust a host name, caller-asserted scope, or
hardcoded storage path.
Qualify the two tasks independently so a shared component cannot hide a
task-transformer mismatch:

```bash
umask 077
export MOLD_HOME=/Volumes/ExternalStorage/mold/uat-h3/mold-home
export CARGO_TARGET_DIR=/Volumes/ExternalStorage/mold/uat-h3/cargo-target-qualification-record
authorization_record=/Volumes/ExternalStorage/mold/uat-h3/compliance/minimax-h3-authorization.v1.json

for model in \
  minimax-h3-fl2va:comfy-pruned-int8 \
  minimax-h3-ref2va:comfy-pruned-int8
do
  task=${model#minimax-h3-}
  task=${task%%:*}
  nix develop --offline --no-write-lock-file -c \
    cargo run --locked --offline --release \
    -p mold-ai-inference \
    --features dev-bins,h3-private-uat \
    --bin h3_artifact_qualification -- \
    --models-root "$MOLD_HOME/models" \
    --model "$model" \
    --authorization-record "$authorization_record" \
    > "/Volumes/ExternalStorage/mold/uat-h3/logs/artifact-qualification-$task.json"
done
```

Each report contains only relative artifact paths and content identities, binds
the external record/source identities, says explicitly that no runtime or
generated media was constructed, and remains private with the model campaign.
The feature is not forwarded by `mold-ai`; every published binary is scanned
for its claim marker and rejected if the private reader is present. This
qualifies artifact identity only. It does not satisfy numerical parity, CUDA
generation UAT, public authorization, or release activation.

### Private runtime-record candidate

The reviewed runtime-record allowlist contains one independently reviewed
candidate for the exact compact FL2VA quality envelope described below:
`f624f71ce1eba7ebb75a13801da855a92f5eec0fccbcb9783f547479c7abfce5`.
The candidate binds source `66fc4d9ea8dc8df0c96718c0f82c0de3ff552bfa`,
runtime-code identity
`0f37dc42394a25a2b612a61f3b476f24959193fd7d84bd4a8fde68f95245fe57`,
and a successful L40S campaign whose H.264/AAC output hash is
`7e34b50a40eead47ae232982e7d048c2ac326fce9174b7f57d747c2464a522d6`.
The campaign authenticated all 42,482,090,318 artifact bytes and retained
all thirteen synchronized runtime observations plus 18,611 external telemetry
samples. This authority is FL2VA-only
and does not extend to Ref2VA, other envelopes, devices, artifacts, attention
implementations, runtime builds, public release, or broader license scope.

The development-only `h3_runtime_qualification_record` binary breaks the
evidence collection/review cycle without self-authorizing: it re-hashes the
complete 42.5 GB FL2VA artifact set, validates an owner-only capture manifest,
hashes every retained evidence file, and writes deterministic record bytes to
stdout. It never edits the source allowlist, constructs the runtime, or
activates a public capability. Published-binary verification rejects its
dedicated claim marker as well as the underlying private artifact reader.

An API-key-authenticated private server may expose that exact reviewed
partition through the additive `GET /api/capabilities` field
`minimax_h3`. Emission additionally requires the configured authorization
wrapper, authorization source document, and exact allowlisted runtime record to
pass the inference-owned no-follow, bounded-read, ownership, permission, hash,
schema, scope, source/runtime-identity, and live CUDA route checks. The response
contains only the FL2VA compact partition, CUDA/attention/quantization
presentation facts, and five manifest-derived component groups; it contains no
paths, hashes, credentials, prompts, media, or download recipe. The component
states are informational. Generation still opens and authenticates all artifact
bytes before admission. The family-wide model-access restriction remains for
legacy clients; H3-aware clients may override it only with this exact additive
FL2VA partition. Ordinary builds omit the field, while Ref2VA remains absent.

The ordinary model list carries two acquisition rows with their upstream source
and download accounting. The authenticated presentation boundary may replace
the FL2VA acquisition row with one exact executable row, but only when all five
referenced component groups are installed. Its generation profile fixes width
1344, height 768, 124 frames at 24
fps, 21 terminal-inclusive grid points, batch one, MP4 delivery, and a required
first-frame source. Web, desktop, and iPhone remove the family-wide denial only
when that exact model name and request envelope agree with the complete
additive component graph. A missing component, widened axis, absent first
frame, supplied last frame, unavailable MP4 encoder, or legacy/partial
capability keeps execution unavailable. Ref2VA may be downloaded but remains
unexecutable; the CUDA-only private qualification is not broadened.

The next capture manifest must use
[`mold.minimax-h3.private-runtime-bound-capture.v5`](./minimax-h3-private-runtime-capture.schema.json).
It binds the exact 40-character Mold source SHA, the stable runtime-code
identity, the measured server executable, artifact/authorization identities,
stable `cuda:<32 lowercase UUID hex>` route plus its process-local ordinal, compute capability, attention
runtime/kernel/qualification identities, and a sorted list of relative
evidence paths. Version 5 retains the exact campaign bootstrap record and its
runtime identity, the serving Linux PID/start-time/boot identity,
executing ELF device/inode/size/SHA-256, domain-separated launch argument and
sorted-environment hashes, and live CUDA driver plus compiled toolkit
versions. Raw arguments and environment values are not serialized. Version 5
invalidates the earlier one-forward smoke envelope and requires the exact
compact-quality route selected by the released workflow: 1344×768, 124 frames
at 24 fps, batch one, exactly 21 terminal-inclusive grid points (20 transformer
evaluations), one first-frame FL2VA endpoint, and explicit ceilings for Qwen
text/vision, condition visual, target video/audio, and total packed rows copied
from the fresh structured observation. Admission checks that envelope after
source preprocessing, the prepared attempt checks it again, and final dispatch
repeats the check before any model execution. The candidate
producer requires its own embedded source SHA and
runtime-code identity to equal the capture and proves that both exact values
occur in the retained ELF server executable before recording that executable's
independently measured SHA-256. Each of these bounds has an
independent `{observed_bytes,bound_bytes,evidence_artifact}` record, with a
nonzero proposed bound. Every observation is nonzero except the legitimately
zero VAE-construction transient described below, and no observation may exceed
its proposed bound:

- fixed runtime host and device bytes;
- Qwen activation and VAE-construction device workspaces;
- condition-VAE, attention, FFN, decoder-tile, and audio-decode device
  workspaces;
- encoded-video, thumbnail, mux-output, and AAC-staging host bounds.

For the first compact-quality record, the four media bounds are not derived from
one scene's compressed sizes. Production enforces a 256 MiB video-only MP4
limit before final container allocation, a 4 MiB bounded PNG writer, an 8 MiB
AAC staging limit that includes the caller-owned interleaved F32 samples, and a
512 MiB bounded final MP4 writer. The reviewed record must carry the exact
conservative admission charges: 1 GiB for simultaneous H.264 sample/container
staging, 8 MiB for the retained first RGB frame plus PNG output/scratch, 8 MiB
for waveform/AAC staging, and 512 MiB for final mux output. Smaller observed
capacities remain evidence, but they cannot reduce these enforced bounds.

The evidence root and every directory below that root must be mode `0700`; the
capture and evidence files must be mode `0600`, process-owned, regular,
non-symlink files outside the checkout. Ancestors above the private root are not
part of this rule because the root itself prevents traversal. The measured
server executable must be one of those files. Paths must be sorted, unique, and
canonical.

The server records process/executable authority from `/proc/self` and the live
CUDA driver API in the same terminal observation as the synchronized memory
measurements. Candidate production exact-crosses those fields with the capture
manifest and requires the observed executable size and SHA-256 to equal the
retained measured ELF. Independent review must still inspect the complete
campaign and its proposed bounds before allowlisting.

The private campaign server emits one structured
`mold.minimax-h3.private-uat-runtime-bound-observation.v5` record only after a
successful terminal attempt. The record includes the actual request geometry,
requested grid-point count, endpoint count/anchor, and prepared row counts.
It also records the exact bootstrap runtime-record file/identity, stable CUDA
UUID, process-local ordinal, compute capability, and private attention
runtime/kernel/qualification identities retained by the executing owner, plus
the process/executable/launch/driver authority described above.
Candidate production reads this designated record through the same no-follow,
size-bounded descriptor used to authenticate its bytes and requires exact
equality with every manifest envelope field and all thirteen claimed
observations. It samples Mold's CUDA allocation-pool high-water
marks synchronously at the VAE construction, Qwen encode, condition VAE,
visual decode, audio decode, and whole-attempt boundaries. Attention and FFN
workspaces are counted by the exact production operators from their actual
shapes and chunk policy. Encoded-video, thumbnail, interleaved-F32 AAC staging,
and final-mux capacities are recorded while their owning buffers are live. The
fixed host observation is the process's resident set at capture entry, and the
fixed device observation is the device-global used-byte baseline immediately
after the attempt constructs its CUDA context. The campaign host must therefore
be quiescent and retain independent per-process GPU attestation; unrelated
device allocations conservatively increase the proposed bound. CPU-offloaded
Qwen uses its process high-water growth over the exact Qwen encode boundary;
an accelerated Qwen uses the CUDA-pool phase growth instead. Every value except
the VAE-construction transient must be nonzero, so the campaign fixture must
exercise a real FL2VA visual condition rather than an unconditioned T2VA
request. A VAE load may legitimately report zero transient bytes when its pool
high-water growth is entirely retained weights that admission already counts
separately; its reviewed bound must still be nonzero. The observer is
diagnostic evidence only: it cannot update admission bounds or authorize its
own record, and shipping verification rejects its private marker.

Every CUDA phase boundary is synchronized before the observer resets or reads
the allocation-pool high-water mark. Construction and condition/decode
workspaces subtract allocations still live at phase exit because the admission
model accounts those retained weights and outputs separately; Qwen retains its
complete activation peak under the route-specific host/device policy required
by its lower-bound validator.

External SSE collection must use a non-buffering read such as Python
`HTTPResponse.read1`; a fixed-size blocking `read` can delay low-volume phase
events until the terminal base64 payload and destroy their timestamps. Retain
the response thumbnail and MP4 as separate hashed evidence files, and bind the
candidate to the SHA-256 of the executable that actually served the request.

The stable runtime-code identity hashes `Cargo.toml`, `Cargo.lock`, Cargo build
configuration, and every regular manifest, build script, Rust source, and
non-Rust compiled input in the complete local `mold-server` dependency closure:
core, catalog, database, scheduler, Candle adapter, inference, and server.
Traversal fails on symbolic links and non-regular entries. The only normalized
source region is the reviewed runtime-record allowlist value array, which
avoids an executable-hash fixed point: a later review-only allowlist rebuild
may have a different exact source SHA and executable SHA, but activation still
requires the same runtime-code identity. Any other local runtime or dependency
input change invalidates the campaign. `Cargo.lock` binds external registry and
Git dependency revisions. The identity also binds `rustc -vV`, host and target,
profile and optimization mode, the sorted Cargo feature and target-cfg axes,
encoded Rust flags, and relevant compiler, linker, and CUDA configuration. The
build script declares every captured environment key as a rebuild input. The
private server build separately rejects any feature set beyond the canonical
CUDA, private bridge/runtime, MP4, and NVML campaign edge, and that canonical
server feature set is itself part of the composite identity. That validation
lives in `crates/mold-server/build_support/` rather than being shared from
`mold-inference`, because `mold-ai-server` is published to crates.io and a
published `.crate` cannot carry a sibling crate's files; the release contract
fails if the two copies of the canonical feature set drift.

Configuration paths are not compiler identities. `CUDA_HOME` and the host
compiler variables record _where_ the toolchain is, so a toolkit upgraded in
place under the same prefix would otherwise leave the identity unchanged while
producing a different executable. The identity therefore also binds the
resolved absolute path and self-reported version of both `nvcc` and the host
compiler, and the build script watches those two binaries so replacing either
invalidates a cached identity. Resolution mirrors `cudaforge`'s own search
order — `NVCC`, then `PATH`, then `CUDA_HOME`, then `CUDA_PATH` — and takes the
host compiler from `NVCC_CCBIN` before `CC`, because naming a different
compiler than the one that built the kernels would be worse than naming none.
A CUDA campaign build whose `nvcc` cannot be resolved or run fails rather than
producing an identity that cannot tell two toolkits apart.

**The measured build must start from a clean target directory.** Watching the
compiler binaries reruns only Mold's own build scripts. The CUDA objects are
produced by dependency build scripts — `candle-kernels`, `candle-flash-attn` —
whose cached outputs Mold cannot invalidate, so a toolchain replaced under an
existing `target/` yields an executable that links objects from the previous
compiler while the identity reports the new one. Nothing in the build can
detect that, which makes it a campaign procedure requirement rather than a
code-enforced invariant: run `cargo clean` (or build into a fresh target
directory) before the measured build, and record that in the campaign notes
alongside the measured server ELF hash.

```bash
umask 077
evidence_root=/Volumes/ExternalStorage/mold/uat-h3/evidence/runtime-candidate
capture_manifest="$evidence_root/runtime-bound-capture.json"

nix develop --offline --no-write-lock-file -c \
  cargo run --locked --offline --release \
  -p mold-ai-inference \
  --features dev-bins,h3 \
  --bin h3_runtime_qualification_record -- \
  --models-root "$MOLD_HOME/models" \
  --authorization-record "$authorization_record" \
  --evidence-root "$evidence_root" \
  --capture-manifest "$capture_manifest" \
  > "$evidence_root/runtime-qualification.candidate.json"
```

The final stderr line names the exact candidate file SHA-256, record identity,
and retained evidence count. Treat that hash as unreviewed input. A later PR
must independently inspect the measurements, reproduce the bounds, and add the
exact candidate file hash to the allowlist; generating a candidate is not a
passing runtime qualification or permission to run the server path. Candidate
serialization and activation share one 128 KiB record limit, so the producer
cannot emit a record the runtime will refuse solely for size.

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
exactly one Scheduler V2 CUDA or Metal GPU for one H3 generation
(`crates/mold-server/src/h3_admission.rs:1291`, `:1736`). Passing zero or more
than one device, or a device on any other backend, is a typed admission error.
One backend owns Qwen, the visual and audio conditions, every DiT block, both
decoders, and mux staging on that frozen route.

Scheduler V2 may eventually distribute independent generation jobs across
multiple GPUs. That is batch/job distribution, not tensor, sequence, or
pipeline parallelism. The milestone contains no cross-GPU model partitioning
claim, and combining VRAM across devices must never make one request appear
feasible.

The curated admission policy currently records a 64 GiB host-RAM
recommendation and enforces a safety floor of `max(8 GiB, 15% of physical
RAM)`. The per-phase host ledger behind that tier peaks at about 22.75 GB in
the Qwen phases (see `H3_CURATED_HOST_RAM_RECOMMENDATION_BYTES`), which fits a
32 GiB host on paper with under 9% of the tier to spare; the recommendation
deliberately stays at 64 GiB because the ledger does not measure allocator
bookkeeping, transient conversion vectors, or backend workspaces. These are
admission-policy constants, not measured H3 production peaks.
Exact artifact sizes, header facts, attention workspace, resident block count,
prefetch, dequantization workspace, and every phase allocation must be frozen
before a real run can be admitted.

## Current evidence status

Most evidence in this section is weight-free. The explicit real-checkpoint row
is the sole exception: it records authenticated, isolated block-0 execution for
both released task transformers. A passing synthetic test proves only a Rust
contract or tiny kernel route, while that block result proves neither an
end-to-end run nor throughput, memory fit, output quality, or release support.

| Surface             | Current evidence                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                       | Current claim                                                                                                                                                                                                                                                                                                                                                          |
| ------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| CPU                 | Deterministic frame/sampler/layout/noise fixtures; tiny processor, Qwen, VAE, DiT, ordered FL2VA/Ref2VA pipeline, mux, cancellation, memory-accounting, portable quantization, frozen-factory, and dispatch tests                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                      | Reference and contract testing only; H3 CPU runtime is unsupported                                                                                                                                                                                                                                                                                                     |
| CUDA primitives     | Tiny synthetic CPU/CUDA parity and execution tests on private UAT host for the sampler ([#845](https://github.com/utensils/mold/pull/845)), AudioVAE ([#849](https://github.com/utensils/mold/pull/849)), Qwen layer 50 ([#850](https://github.com/utensils/mold/pull/850)), visual VAE ([#851](https://github.com/utensils/mold/pull/851)), DiT ([#853](https://github.com/utensils/mold/pull/853)), and portable quantization ([#860](https://github.com/utensils/mold/pull/860)). The scaled-FP8/Qwen-INT8 slice ([#865](https://github.com/utensils/mold/pull/865)) and fail-closed dispatch seams ([#864](https://github.com/utensils/mold/pull/864), [#866](https://github.com/utensils/mold/pull/866)) have synthetic CPU/reference tests plus CUDA contract/typecheck evidence, not physical CUDA execution.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                   | Selected tiny primitives only; CPU-only quantization/dispatch evidence and CUDA typechecks are identified separately. No end-to-end H3 generation or hardware qualification; see the separate real-checkpoint row.                                                                                                                                                     |
| CUDA full attention | [PR #857](https://github.com/utensils/mold/pull/857) records the earlier synthetic dense-reference-to-FlashAttention-v2 work. Exact PR #871 implementation head `c0625402` and source tree `b1452b5f` were qualified on private UAT host (4× L40S with 46,068 MiB each, SM89, driver 595.71.05, CUDA 12.8.93, Rust 1.95) without model data or runtime activation. The H3 release contract and all 36 adversarial CUDA distribution/parser cases passed; the production dispatch stayed directly guarded by the release-candidate configuration; every actual `usize` tensor stride and rounded launch row was checked before conversion to the kernel's `u32` contract; the isolated 53-kernel candidate built offline with warnings denied; and 11 attention tests passed with 141 filtered. Ten network-isolated BF16 probes ran on GPU UUID `GPU-9ffc81c5-3944-6490-bfd9-f68366f98226`. Their timing samples were 4,974, 5,006, 5,045, 5,047, 5,056, 5,123, 5,130, 5,167, 5,708, and 168,321 microseconds, with p50 5,056 and p95 5,708 under the probe's percentile convention. Maximum absolute parity delta was `0.00048828125` against a `0.02` bound; swapping Q/K produced `0.04736328`, exceeding the required `0.02` sensitivity floor. The candidate binary SHA-256 was `0d494cfc8a165ff1f00ec9b48c6ce370375bd4ebf7634b5894d042d7e9f453af`; tracing found no internet socket or H3/model-artifact path, and recorded `model_artifacts_accessed = false` and `runtime_activated = false`. Long-row workspace shapes 37,296 and 107,856 were planning-only. | Synthetic development evidence only. The opt-in candidate correctly carries the compiled-kernel claim and is not a shipping binary. The same-tree ordinary fixture proves release-candidate exclusion only for a non-publishable shipping-feature build; it does not establish a public release artifact, real-model correctness, quality, peak memory, or throughput. |
| Metal               | Shared primitive feature compilation, forced-local typecheck, and [#860](https://github.com/utensils/mold/pull/860)'s tiny deterministic CPU/Metal quantized-forward parity. [#865](https://github.com/utensils/mold/pull/865)'s newer scaled-FP8/Qwen-INT8 cases are CPU-only. [PR #1183](https://github.com/utensils/mold/pull/1183) (`29d6af20`) added the candle-layer Metal execution path — `H3AttentionBackend::MetalChunkedDenseMath` with its shape-derived chunk frozen into and hashed by the plan, the audio VAE's rank-4 head mean folded through `metal_reduce`, `select_h3_int8_linear_kind`'s portable dequant arm, fp8-scaled weights refused by name, and `H3CandleBackendDevice::Metal` / `H3ConditionerExecution::MetalResident` accepted through the frozen plan — and ran it on real Apple Silicon (bender, M4 Mac mini): `cargo check --features metal,h3` exit 0, 233 mold-candle tests including chunked-plan-vs-CPU equality, never-selects-native-INT8, and the fp8 refusal, plus 85 mold-inference `minimax_h3` tests. That is unit-suite execution on a 16 GB machine. Since [PR #1323](https://github.com/utensils/mold/pull/1323) (`0d191cd3`) the path is reachable in a shipped build: `h3_admission` admits `GpuBackend::Cuda | GpuBackend::Metal` (`crates/mold-server/src/h3_admission.rs:1291`, `:1736`) and places the conditioner through `AssignedMetalThenDrop` (`:1760`), the public runtime profile is `supported-compact-fl2va-cuda-sm89-or-metal` (`crates/mold-inference/src/minimax_h3/private_server.rs:4792`) with an absent compute capability meaning Metal, and the shipped macOS artifacts carry `h3` (`flake.nix:228`, `.github/workflows/release.yml:74`) and are compiled by macOS CI (`.github/workflows/ci.yml:693`). What has still never happened is a render: a Metal attempt is refused below a unified-memory floor of `max(device floor, host floor)` (`crates/mold-inference/src/minimax_h3/private_server.rs:672`), and the compact stack's 42.5 GB unified working set stalls a 48 GB M4 Max. | `CorrectnessOnly` (`crates/mold-core/src/minimax_h3.rs:474`), following the Wan #800 precedent: the candle execution layer is implemented, unit-tested on real Apple Silicon, admitted by the frozen contract, and present in the shipped macOS artifacts — but no H3 checkpoint has ever completed a render on Metal. Not hardware-qualified, no UAT — lifting it needs a 64 GB-class Apple Silicon machine ([#1164](https://github.com/utensils/mold/issues/1164), [#1296](https://github.com/utensils/mold/issues/1296)). |
| Server/factory      | Pinned identities, upstream-direct compact acquisition, runtime request contracts, secure reference ingress, prepared shapes, frozen single-GPU admission, block-streaming ownership, immutable factory authority, fail-closed backend adapter, and FL2VA/Ref2VA worker dispatch                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                       | Compact downloads are available; execution is limited to the exact reviewed FL2VA route on CUDA and on the admitted but correctness-only Metal backend, while Ref2VA and broader routes remain unavailable                                                                                                                                                                                                                       |
| Studio surfaces     | [#867](https://github.com/utensils/mold/pull/867) is merged in main `50f28de3` with web, desktop, and iPhone authoring, recovery, canonical upload, and provenance contracts. Its media-arithmetic fixtures use ordinary generated test media, not H3 output.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                          | Authoring and recovery contracts only; gated readiness cannot make the runtime available                                                                                                                                                                                                                                                                               |
| CLI/TUI/Discord     | [#868](https://github.com/utensils/mold/pull/868) is merged in main `50f28de3` with weight-free ordered-reference authoring, canonical reference leases, and media provenance.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                         | Authoring and provenance contracts only; no runtime activation                                                                                                                                                                                                                                                                                                         |
| Real checkpoint     | [PR #883](https://github.com/utensils/mold/pull/883) authenticated both released 20.97 GB INT8 task transformers from retained descriptors, validated the shared 932-tensor header and 200 quantization sidecars, executed isolated block 0 on CUDA, and returned to zero live blocks; both reports recorded `factory_activated: false`.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                               | Private block-0 execution only. Conditioner execution, all 50 denoise blocks, visual/audio decode, mux, synchronized output, quality, and end-to-end memory/performance remain unqualified; the product runtime remains unavailable.                                                                                                                                   |

The ordinary shipping-feature fixture used the same exact `b1452b5f` source
tree. Its release-profile build completed in 766 seconds, sealed all
10 generated `sm_89` PTX modules, verified the CUDA SM89 binary/archive
relationship, and passed the MiniMax H3 attention release-candidate exclusion
verifier. The fixture binary was 115,546,272 bytes with SHA-256
`1f48362462b3956959c90d58caa6bc53f18910dcc9480236a0f8be17c953c411`; its
40,335,347-byte archive had SHA-256
`741380db3dff4de70a414293c6f3225153b16f9173a171aeee3baf20ccd91ee4`.
The CUDA release-verification log had SHA-256
`4d85664707667f52431a2f757038a86d163a156aa72e601f8fcb568a59b8eb13`, and
the H3 exclusion log had SHA-256
`72e4923f8a21ad1a647c125d81774a99014110db1782c148fa4529086aad9b8c`.
Binary inspection found only the exact omitted/omitted provenance marker, with
neither compiled provenance nor the release-candidate claim marker.

That fixture is deliberately non-publishable. It demonstrates exclusion in an
ordinary shipping-feature build assembled from the exact source tree; it is
not a public Mold release artifact, H3 runtime activation, real-checkpoint
qualification, or licensed H3 UAT.

The earlier attention and primitive probes used synthetic tensors and fixtures.
Separately, the qualified external campaign stores and authenticates the
approved payloads, and authenticated real-checkpoint block-0 execution ran on
CUDA. That narrow block result is not end-to-end synchronized-A/V generation,
full runtime qualification, output-quality evidence, or a benchmark; no
generated media was retained.

## Private qualification and future public acceptance matrix

Private artifact-identity, numerical-parity, T2VA/FL2VA/Ref2VA, Comfy,
memory/performance, cancellation, fault-recovery, and single-device rows may be
attempted under the private evidence record. Their payloads and evidence remain
private. Product-surface, hosted/remote-client, distribution, and
release-artifact rows are authorized; they remain evidence-gated where the
acceptance matrix requires technical qualification. Private
authorization, clean storage, artifact identity, and isolated block execution
have partial evidence described above; numerical parity and end-to-end
generation rows remain unattempted. The table defines final required evidence
and does not itself report a completed release gate.

| Gate                          | Required campaign                                                                                                                                                                    | Passing evidence                                                                                                                                                                                          |
| ----------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Authorization                 | Revalidate the pinned license/Q&A and completed governance decision for every intended product, territory, user, artifact, and output flow                                           | Current decision record plus the README and H3 user guide carrying the pinned license link and required user-facing notice                                                                                |
| Clean storage                 | Use the qualified isolated external campaign with its access-controlled absolute `MOLD_HOME`; never reuse an ordinary Mold home                                                      | Capacity and mount report, private ownership/mode, clean before/after inventory, and no H3 bytes in the checkout                                                                                          |
| Artifact identity             | Fetch only the approved task/layout and every pinned companion                                                                                                                       | Exact repository/revision/path/byte count/full SHA-256, component-index hashes, license/NOTICE capture, and no unexpected file                                                                            |
| Full-path numerical parity    | Run tokenizer/processor, Qwen layer 50, visual VAE, AudioVAE, token refiner, transformer block, packed layout, noise allocation, and dual sampler against pinned Diffusers BF16/FP32 | External fixture bundle passes schema/hash validation and every recorded tolerance; no approximate backend contributed a golden value                                                                     |
| T2VA                          | Generate 1344x768 at 124 and 345 frames, plus the 243-frame grid control                                                                                                             | Decoded 24 fps MP4, exact frame count, synchronized 32 kHz stereo audio, stable seed/provenance, phase telemetry, and full-reference quality metrics                                                      |
| FL2VA                         | First-only, last-only, and first+last at the same grid points, including mismatched source aspects                                                                                   | Exact endpoint signatures/order, official resize/crop and fresh seed-42 posterior evidence, preserved boundary behavior, decoded A/V validation, and quality metrics                                      |
| Ref2VA                        | Image-only; video with soundtrack; image+standalone audio; mixed ordered image/video/audio; swapped-order comparison; every count/duration/type failure                              | Exact packed order, modality tags, rotary clocks, soundtrack association, negative-case codes, decoded A/V validation, and order-sensitive quality comparison                                             |
| Comfy deployment path         | Compare the approved pruned INT8/NVFP4 layout with the full path on the same prompts, sources, shapes, seeds, and hardware                                                           | Spatial/temporal perceptual metrics, audio spectral/loudness/channel metrics, exact A/V timing, measured deltas, named accuracy tier, and no exact-parity label unless actually proven                    |
| Memory/performance            | Measure 960x544 and 1344x768 at 124 frames, then 345-frame feasibility, on both the declared high-memory tier and declared streamed consumer tier                                    | Exact GPU/driver/backend, CPU/RAM/storage, artifact identities, attention kernel, resident/prefetch plan, cold/warm setup, per-phase timing, peak VRAM/RAM, and output hashes                             |
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

Issue #831 records the completed governance decision: the README and H3 user
guide satisfy the resulting documentation obligations, and no additional
H3-specific product or acceptable-use control is required. Issue #827 remains
open until the authorized acceptance matrix has real evidence and the exact
release artifact passes. Landing more weight-free code or synthetic CUDA tests
does not close that engineering gate.

Any change to the pinned license, Q&A, source revisions, component identities,
supported territory, execution layout, attention backend, quantization policy,
or public product surface requires updating this record and re-running the
applicable review before release.
