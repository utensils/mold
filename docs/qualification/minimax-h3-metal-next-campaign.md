# H3 Metal next campaign: default resolution and FL2VA coverage

Preparation status: **READY FOR UAT planning; execution ON HOLD**.
Audited 2026-09-05 against merged `27ed658e` (PR #1604), Candle
`744ae3b83cfac18db28107a353c449cc9b80d4ec`. This document records no new
hardware result and authorizes no launch. The user subsequently released bounded H3 CUDA validation; its completed
results are in [the CUDA follow-up](minimax-h3-cuda-post-1604.md). Metal GPU
device creation, tests, model loads, upstream inference, pressure tests and
UAT remain on hold. CPU review and preparation may continue. No kernel setting or service change is required by this plan.

## Acceptance audit

| Item | Current evidence | Remaining evidence |
| --- | --- | --- |
| #1164 Metal execution and dispatch | Shipped Metal admission, portable quantization, streamed Qwen; #1604 merged attention/INT8 lifetime fixes and owned local execution | Verify exact campaign binary identity; do not reopen implemented dispatch work |
| #1542 compact memory fit | Phase accounting exists; disk size is not residency; 256-square guarded render fits | Exact Turbo/default request phase budgets and measured phase peaks on 48 GiB hardware |
| Conditioned FL2VA | First-frame Turbo 4-step, 256×256, 107 frames, seed 42; local/server identical MP4 | Default 1344×768 **and 124 frames**; base last-frame/two-endpoint coverage where advertised |
| CUDA preservation | Real attention/INT8 tests, no-skip FlashAttention probe, matching reduced-size video/audio | Paired default/broader fixtures at the campaign revision |
| Quality | Reduced-size SSIM 0.988634, PSNR 37.762865 dB, audio correlation 0.973562 | Default-resolution visual review and audio listening, plus numerical comparison |
| Tier / issue completion | Metal remains `CorrectnessOnly`; both issues open | Separate memory, correctness and performance conclusions; no automatic promotion or closure |

Issue bodies still say #1604 is open and fixes are unmerged. Those historical
checkboxes are superseded by `27ed658e`; they are not new implementation
requirements. No issue was edited during this preparation. The conditional
GGUF proposal in #1542 remains conditional: investigate a new layout only if
measured phase residency cannot fit safely. This campaign does not touch
Candle VAE #1040, Wan #1059/#1094, or LTX #1462.

The retained measurements, binary limitations, hashes and source references
remain in [the previous campaign](minimax-h3-metal-memory.md). Its 8 GiB
allocation ceiling and 7,757,168,640-byte highest sample apply to the small
request only. Neither number is a default-resolution bound.

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
| B / intermediate | same | 768×768 / 107 | First frame; phase scaling and decoder checkpoint before default size |
| C / default | same | 1344×768 / 124 | First frame; actual default canvas and duration, Metal/CUDA pair |
| D / base-first | `comfy-pruned-int8` | 1344×768 / 124 | First frame; base schedule and quality independently of Turbo |
| E / base-last | same base | 768×768 / 107 | Last frame; only if current profile accepts this mode |
| F / base-both | same base | 768×768 / 107 | First and last frames; endpoint overlap and conditioning cost |
| G / long | original Turbo tag | 768×768 / 345 | First frame; duration scaling, only after a separate safe budget decision |

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
inapplicable reference phases explicit. Metal's zero separate host increment
means host cost is already included; it does not mean host residency is zero.
Also retain all fields from `public_runtime_bounds_for_shape`, adapter
resident charge, packed video/audio/condition rows, Qwen pre-merge patch
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

**Instrumentation gap:** the retained allocator/watchdog modifications were
unshipped, and are not reproducible from main alone. Recover and hash their
exact sources/binary or prepare a separately reviewed qualification build.
The existing `private_runtime_observer.rs` process attestation is Linux/CUDA
shaped and `process_peak_resident_bytes` reads `/proc/self/status`; do not
claim its observation schema is Metal evidence or fabricate CUDA fields.
Use a clearly separate qualification report for macOS. This document adds
no production accounting formula, new layout, or launcher.

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
4. Derive and record a separate native-allocation ceiling from current
   capacity/headroom and the measured phase plan, retaining the host floor.
   Reject a case whose safe ceiling cannot cover its planned device phase.
   Keep the previous 8 GiB ceiling for row A; do not reuse it blindly for C.
   The independent watchdog samples at most every 250 ms, aborts below 12 GiB
   available, on non-normal pressure or more than 256 MiB swap growth, and
   fails closed on a missing/invalid/stale sample. These campaign limits are
   tighter than the built-in 8 GiB/2 GiB cooperative guard.
5. Freeze a per-case wall-clock deadline before launch. A watchdog failure
   or missing allocation event stream cancels the isolated child; enforce
   a bounded shutdown deadline and terminate only that owned process group
   if cooperative cancellation does not settle. Record abnormal exits and
   incomplete commands as failures, never resumable successful evidence.
6. In every exit path, reap the child, confirm the owned process group has
   no descendants, retain stdout/stderr/request/telemetry and refusal facts,
   stop its sampler, and release only this attempt's reservation and lock.
   Verify pressure and swap after settlement. Preserve partial media as
   failed artifacts; never silently retry, shrink, change steps or clear a
   stale lock that may belong to a live process.

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
Document any acceptance threshold before inspecting new results; a failure
or unreviewed result cannot promote Metal beyond `CorrectnessOnly`.

## Hold-safe preparation result and handoff

This audit found no demonstrated additional production fix: the known
lifetime/dispatch fixes are merged. Only documentation changes are prepared.
The initial preparation initialized no GPU. Subsequently authorized bounded
CUDA validation completed as recorded in the linked follow-up; no model was
loaded, upstream inference run, service changed, pressure test started or
kernel setting modified.

The next operator needs: recovered verified instrumentation and watchdog;
exact per-case allocation-free budgets; frozen input fixtures; explicit hold
release and exclusive host availability; then sequential phase measurements,
paired CUDA media and visual/listening review. Until those exist, report
**READY FOR UAT preparation, not qualified and not cleared to launch**.
