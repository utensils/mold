# H3 Metal next campaign: default resolution and FL2VA coverage

Execution status: **shipping qualification complete for the request-gated Apple
Metal portability tier**. On release-candidate inference binary
`c7aca386f6338f74b313d713220674b7d3f039a8a55e6036890f7f4fed979e4f`,
row A completed through forced-local and cold-server routes and produced
byte-identical MP4s. The fused last-dimension softmax removed the command-buffer
lifetime mismatch found by the earlier row B attempt. Corrected E/F invocations
retain the reviewed endpoint-contract refusals. B/C/D/G full renders are
deferred: their exact allocation-free budgets remain the admission evidence,
not a blanket promise that every shape fits or a Metal performance claim.
Completed bounded CUDA results remain the separate backend-preservation record.

The original plan was audited 2026-09-05 against merged `27ed658e` (PR #1604).
The current campaign binary uses Candle
`bf2cd29a791dbc053df826b4377d092c3809d17f`; completed bounded CUDA results
remain in [the CUDA follow-up](minimax-h3-cuda-post-1604.md). No kernel
setting or shared service change is required by this plan.

## Acceptance audit

| Item | Current evidence | Remaining evidence |
| --- | --- | --- |
| #1164 Metal execution and dispatch | Shipped Metal admission, portable quantization, streamed Qwen, owned local execution; bounded-query and fused-softmax lifetime tests pass CPU/Metal parity and allocation/retention checks; current forced-local and cold-server row A outputs are byte-identical | Larger rows remain optional capacity/performance characterization, not a support gate |
| #1542 compact memory fit | Exact A/B/C/D/G request budgets are exported; current row A measured 1,164,279,920 local and 1,174,984,416 server native peak bytes under the 8 GiB ceiling | Longer rows remain governed by their live host/device fit decision |
| Conditioned FL2VA | Turbo 4-step, 256×256, 107 frames, seed 42 completed through both public execution doors; corrected last-frame and two-endpoint requests were refused by the reviewed envelope | Larger admitted shapes may be characterized independently |
| CUDA preservation | Real attention/INT8 tests and a no-skip FlashAttention probe remain retained; the Metal-only attention path does not alter CUDA dispatch | A new paired full render is required before making a new cross-backend quality claim |
| Quality | The retained row has 107 coherent H.264 frames, 24 fps, 4.458 seconds, and matching 32 kHz stereo AAC; historical reduced-size CUDA comparison remains SSIM 0.988634 / PSNR 37.762865 dB / audio correlation 0.973562 | No CUDA-relative performance claim is made from the current row |
| Tier / issue completion | Metal is `Supported` and exact-request-memory-gated; #1164 and #1542 close with the shipping PR | Performance qualification remains a separate claim |

The issue threads carry the final current-binary measurements and the explicit
request-gated support boundary. The conditional GGUF proposal in #1542 is not
triggered: the existing streamed layout completes the supported route, and a
request whose derived phase cannot fit is refused before allocation. This
campaign does not touch Candle VAE #1040, Wan #1059/#1094, or LTX #1462.

The retained measurements, binary limitations, hashes and source references
remain in [the previous campaign](minimax-h3-metal-memory.md). Its 8 GiB
allocation ceiling and 7,757,168,640-byte highest sample describe the earlier
instrumented binary. The current fused-softmax binary's row A native peaks are
1,164,279,920 bytes local and 1,174,984,416 bytes through the cold server.

## Freeze the paired request matrix

All rows use batch one, 24 fps, seed 42, explicit width/height/frame count,
MP4 with audio, and no expansion or prompt rewriting. Retain the literal
prompt, PNG bytes and SHA-256, normalized conditioning identity, exact model
and adapter revisions, resolved guidance/shift/sigma ladder, and canonical
request JSON. Reuse the old gradient/source and prompt for row A only if the
retained bytes are available; otherwise name a new fixture, never claim a
reproduction. For other rows freeze one scenic image and one distinct last
frame once; Metal and CUDA consume identical bytes. Do not infer the canvas
from a source image: that intentionally selects a different aspect-derived
size in the CLI.

| Order / case | Model tag suffix (`minimax-h3-fl2va:`) | Canvas / frames | Conditioning / purpose |
| --- | --- | --- | --- |
| A / smoke-replay | `comfy-pruned-int8-turbo-4step-768p` | 256×256 / 107 | First frame; revalidate guard and exact-current local/server identity |
| B / intermediate | `comfy-pruned-int8-turbo-4step-768p` | 768×768 / 107 | First frame; phase scaling and decoder checkpoint before default size |
| C / default | `comfy-pruned-int8-turbo-4step-768p` | 1344×768 / 124 | First frame; actual default canvas and duration, Metal/CUDA pair |
| D / base-first | `comfy-pruned-int8` | 1344×768 / 124 | First frame; base schedule and quality independently of Turbo |
| E / base-last | `comfy-pruned-int8` | 768×768 / 107 | Last frame; only if current profile accepts this mode |
| F / base-both | `comfy-pruned-int8` | 768×768 / 107 | First and last frames; endpoint overlap and conditioning cost |
| G / long | `comfy-pruned-int8-turbo-4step-768p` | 768×768 / 345 | First frame; duration scaling, only after a separate safe budget decision |

Turbo 4-step is **five terminal-inclusive sigma points / four evaluations**;
base defaults to 21 points / 20 evaluations. Resolve defaults from the exact
model profile, recording the resolved values; never transplant another
Turbo tier's flow shift. Last-only or two-endpoint refusal from a current
profile is a retained contract result, not permission to bypass it. Broader
FL2VA does not imply Ref2VA or other Turbo adapters were qualified.

Run cold process cases first. A warm server repeat is separately labeled:
`QwenConditioningCached` is not a conditioner load/encode measurement.
After each case, inspect its measurements before considering the next.
Do not launch all rows as a batch. Default-resolution fit is not a claim
that the maximum-duration default canvas fits.

## Exact accounting to capture before and during every case

Before allocating model tensors, retain the actual prepared request's
`H3FactoryTargetBudgetInput`, frozen plan/factory identity and the resolved
Qwen placement. Export the **existing authority**, not a separately
reimplemented formula. `private_server.rs::private_h3_unified_target_peak_bytes`
computes `max(device_bytes + host_bytes)` across these budget prefixes:

```text
reference_decode, reference_preprocess, reference_visual_encode,
reference_audio_encode, vae_load, qwen_encode, qwen_transfer,
condition_encode, noise_allocation, transformer_load, denoise,
visual_decode, audio_decode, waveform_transfer, mux
```

For each prefix preserve the exact `_phase_device_bytes` and
`_phase_host_bytes`, checked sum, applicability and binding maximum. Keep
inapplicable reference phases explicit. These per-phase host fields remain
separate and can be nonzero on Metal: preserve every host and device value
before summing. Only the final Metal owner grant projects the maximum combined
phase sum into unified device bytes with a zero **additional** host claim
(`private_server.rs::owner_fence_budget_preserves_cuda_and_projects_metal_unified_memory`).
That owner projection must never overwrite the phase report's host columns or
be interpreted as zero host residency. Also retain all fields from
`public_runtime_bounds_for_shape`, adapter resident charge, packed video/audio/condition rows, Qwen pre-merge patch
rows and merged text pads. A 256-square **base** budget cannot price row C's
Turbo adapter or 124-frame sequence.

Record one machine-readable row at each phase entry/exit, each streamed
Qwen layer/DiT block boundary, each decoder chunk, and every allocation
high-water change. Required columns:

```text
case_id, request_sha256, executable_sha256, monotonic_ns, phase,
event, iteration, allocated_native_bytes, native_peak_bytes,
requested_allocation_bytes, allocator_ceiling_bytes,
process_resident_bytes, process_peak_resident_bytes,
host_available_bytes, swap_used_bytes, kernel_pressure,
budget_device_bytes, budget_host_bytes, phase_complete, error
```

Use the runtime's `H3PipelinePhase` names, including `QwenLoad`,
`QwenEncode`, `VaeLoad`, `VisualConditionEncode`, `NoiseAllocation`,
`TransformerLoad`, `Denoise`/`TransformerBlock`, `VisualDecode`,
`AudioDecode`, `VideoEncode`, `Mux`, `Staged`, `Complete`. Attribute attention
and FFN transients inside the denoise phase separately. Transfer-only budget
phases need explicit measurement boundaries even where there is no matching
pipeline enum. Preserve nested phase membership instead of attributing a
chunk twice. Synchronize phase boundaries before reporting completed-device
residency; label sampled RSS/high-water values as samples. Report absolute
native peaks and changes from the phase-entry baseline separately.

Do not add process RSS to native Metal allocation to claim physical unified
use: they may overlap. Do not subtract consecutive process-lifetime RSS
high-water marks and call the result a phase peak. Native allocation peak,
process RSS, system headroom, calculated admission and retained post-call
allocation answer different questions and must stay separate.

**Instrumentation status (2026-09-18):** the allocator/watchdog gap is closed
on `h3-macos-final`. The shipped `minimax_h3::campaign_capture` writes the
machine-readable rows this document requires (schema
`minimax-h3-metal-campaign-capture.schema.json`) plus a `phase_budget.json`
sidecar exporting `H3FactoryTargetBudgetInput::phase_budget_rows()` at
prepare time; `MOLD_H3_METAL_CAMPAIGN=1` opts the shipped Metal memory guard
into the 12 GiB / 256 MiB campaign invariants and requires
`MOLD_H3_METAL_CAMPAIGN_CEILING_MB`; `MOLD_H3_METAL_CAMPAIGN_BUDGET_ONLY=1`
makes the owned attempt refuse at device attach — after the sidecar exists,
before any model tensor is allocated — which is the allocation-free
pre-flight pass. The external supervisor is the `h3_metal_campaign_watch`
dev-bin (`dev-bins` feature): it injects the campaign environment, enforces
the host-level gates from outside the GPU process, kills the owned process
group on violation, and verifies no descendant survives. The observer's
process probes now have macOS arms (`proc_pidinfo`/`task_info`); the
Linux-shaped capture remains the CUDA report and is never quoted as Metal
evidence.

## Fail-closed launch and cleanup gates (after the hold is released)

1. Confirm explicit release of the user hold, exclusive host ownership via
   `/tmp/mold-metal-qualification.lock`, no competing GPU work, and the exact
   binary/source/instrumentation/checkpoint identities. Use an isolated
   `MOLD_HOME`, output directory and process group; do not change the existing
   server. Missing lock ownership or missing evidence directory refuses launch.
2. Prove the recovered allocator ceiling and external watchdog with test
   doubles before any model run: over-ceiling allocation, failed/stale memory
   sample, pressure change, swap growth, missing child, cancellation and
   cleanup paths. An environment variable alone is not proof a binary has
   a pre-allocation ceiling. No working verified ceiling means **no launch**.
3. Read the native automatic wired limit and effective capacity; never change
   them to make a case pass. Require normal pressure and at least the previous
   24 GiB available baseline. For larger cases additionally require the
   prepared unified peak to fit below current availability minus a 12 GiB
   host floor and below effective device headroom. Reserve that exact grant;
   do not infer it from checkpoint file sizes or enlarge it after refusal.
4. Set `MOLD_H3_METAL_CAMPAIGN=1` on the exact campaign binary to opt the
   shipped Metal memory guard into its narrower invariants: 12 GiB available
   floor and 256 MiB maximum attempt swap growth, at the same 250 ms sample
   cadence. This is a safety-policy switch, not an output-semantics switch; it
   must never be used to relax the shipped 8 GiB / 2 GiB default. The
   independent native-allocation ceiling is set with
   `MOLD_H3_METAL_CAMPAIGN_CEILING_MB` and is a required launch input.
   Before a case launches, its allocation-free budget pass runs the same
   owned attempt with `MOLD_H3_METAL_CAMPAIGN_BUDGET_ONLY=1` and a
   provisional ceiling: the attempt refuses at device attach with the
   prepared phase budget exported, which is the measured phase plan gate 4
   prices the real ceiling from. A budget-only refusal is not a case and
   never counts as render evidence.
5. Derive and record a separate native-allocation ceiling from current
   capacity/headroom and the measured phase plan, retaining the host floor.
   Reject a case whose safe ceiling cannot cover its planned device phase.
   Keep the previous 8 GiB ceiling for row A; do not reuse it blindly for C.
   The independent watchdog samples at most every 250 ms, aborts below 12 GiB
   available, on non-normal pressure or more than 256 MiB swap growth, and
   fails closed on a missing/invalid/stale sample. These campaign limits are
   tighter than the built-in 8 GiB/2 GiB cooperative guard.
6. Freeze a per-case wall-clock deadline before launch. A watchdog failure
   or missing allocation event stream cancels the isolated child; enforce
   a bounded shutdown deadline and terminate only that owned process group
   if cooperative cancellation does not settle. Record abnormal exits and
   incomplete commands as failures, never resumable successful evidence.
7. In every exit path, reap the child, confirm the owned process group has
   no descendants, retain stdout/stderr/request/telemetry and refusal facts,
   stop its sampler, and release only this attempt's reservation and lock.
   Verify pressure and swap after settlement. Preserve partial media as
   failed artifacts; never silently retry, shrink, change steps or clear a
   stale lock that may belong to a live process.

### Predeclared paired-render acceptance

These gates were frozen before inspecting any new Metal/CUDA pair from this
campaign. Every pair must decode to the requested dimensions, frame count,
24 fps clock, and 32 kHz stereo stream with finite samples and no clipping.
Against the CUDA render, mean RGB PSNR must be at least 24 dB, minimum
per-frame PSNR at least 18 dB, and mean 8x8-luma SSIM at least 0.85. Aligned
zero-lag audio correlation must be at least 0.95 with RMS difference at most
0.002. Human review must find the same shot and motion, intelligible matching
audio, and no new flicker, banding, ghosting, collapse, or dropout. A miss is
retained as a failure. These gates qualify cross-backend fidelity; they do not
turn the supported portability tier into a performance claim or bypass exact
request admission.

## CUDA and output comparison gates

Bounded checks at `d6096446` completed after the user released CUDA
validation; the linked follow-up retains their results. For a later runtime
revision, qualify that exact campaign revision on one idle SM89 GPU. Keep the native CUDA INT8 and FlashAttention routes. The bounded tests
are `minimax_h3::attention::tests` and `comfy_int8::tests` in `mold-ai-candle`
with `h3,cuda,h3-flash-attn-rc,flash-attn`; build the
`h3_attention_qualification` binary from `mold-ai-inference` with
`dev-bins,h3-cuda`. Require actual CUDA device creation, nonzero probe rows
and no skips. These are GPU tests, not hold-safe CPU checks. Their completed run does not
authorize the separate paired full-render campaign.

Pair rows A–F on the same source revision, checkpoint/adapter and request;
row G requires its own capacity decision on both hosts. Retain decoded frame
count, dimensions, fps, duration, audio sample rate/channels, decode errors,
finite/sample-clipping checks, whole-video and per-frame PSNR/SSIM, aligned
zero-lag audio correlation/RMS difference, sampled frames and a human
`Visual:` and `Audio:` assessment. Report alignment method and both original
stream lengths; no trimming to conceal duration drift. Exact local/server
byte equality is useful but does not replace cross-backend comparison.
Historical 256-square metrics are observations, not universal thresholds.
Document any comparison threshold before inspecting new results. A failed or
unreviewed result remains evidence and cannot be hidden by `Supported` status;
the request must be refused or the separate quality claim withheld.

## Final execution result

The released hold exposed two real Metal lifetime defects in row B's first
transformer block. First, the chunked dense path retained fused QKV and two full
F32 key layouts beside a 1 GiB score/probability pair. The source now releases
fused QKV after owned views exist, converts only the active query slice,
transposes K directly, synchronizes conversion staging before scores, and caps
each score matrix at 768 MiB.

The first guarded rerun then crossed the floor at 11.2 GiB despite entering the
block with 30.68 GB available, normal pressure and unchanged swap. A focused
Metal regression measured why: generic softmax submitted 50,462,720 bytes for
a 16,777,216-byte probability matrix, retaining two additional score-sized
intermediates that the workspace did not price. Dense attention now uses
Candle's fused last-dimension softmax, and the same regression submits exactly
16,777,216 bytes. The B-envelope workspace test remains below the existing
4.5 GiB denoise-workspace grant; CPU/Metal output parity passes; the exclusive
attention check retains 786,432 bytes after return. CUDA dispatch is unchanged.

The exact revised release binary, watchdog, source patch and fixture identities
are retained in the external campaign directory. Row A's local and cold-server
outputs both hash to
`dfd95b7db9117e639ae3570252241ef50eddb736034338cf4cae2ec8845b14af`.
They contain 107 H.264 frames at 256×256/24 fps and 32 kHz stereo AAC. Local /
server generation times were 875.2 / 885.3 seconds; native peaks were
1,164,279,920 / 1,174,984,416 bytes; minimum availability was
25,544,704,000 / 25,502,777,344 bytes; pressure stayed normal; swap growth was
0 / 1,572,864 bytes; both watchdogs reaped every descendant and released the
coordinated lock. A sampled frame was visually coherent.

Corrected row E refused `LastFrameToAudioVideo` and `endpoint_anchor last`
against the reviewed first-frame envelope. Corrected row F refused its two
endpoint conditioning at `qwen_vision_rows 4608` (cap 4032) and
`condition_visual_rows 1152` (cap 1008). B/C/D/G retain their allocation-free
budget sidecars and were not launched as long renders under the shipping
decision. Metal is **supported with exact request admission**; the campaign
does not claim default-shape fit on every Mac or CUDA-equivalent throughput.
