# Wan Metal: correctness, admission and performance campaign

**Prepared on 2026-09-05; all GPU work and UAT are on user hold.** This is a
runbook, not qualification evidence. Do not execute its device tests, reference
models, server renders or pressure helper until the user releases that hold.
Issues [#1059](https://github.com/utensils/mold/issues/1059) and
[#1094](https://github.com/utensils/mold/issues/1094) remain open.

## Starting evidence and ownership

The earlier [guarded attempts](metal-server-pressure-1059.md) did not finish.
On clean `b880155ff72b84c4f1df50368cf98bf64c024358`, following cache purge, a cold
Wan 1.3B BF16/Q8 UMT5 server request completed: 62.038 s, 15.605 GiB minimum
sampled native available memory, zero positive swap growth and normal pressure.
All 17 MP4 frames decoded, but the scene was replaced by saturated horizontal
artifacts. A warm lossless APNG repeat took 44.977 s, decoded 17 frames and had
the same visual failure. Its 27.091 GiB minimum is not a cold-load measurement.
Transformer/VAE manifest hashes and the retained encoder download hash match.
Both renders used zero artificial pressure. Independent review verified all 54
raw evidence hashes. The final acceptance item is still unchecked.

Raw records, requests, outputs, contact sheets and the reviewed report are at
`/Volumes/ExternalStorage/mold-1059-qualification/resumed-20260905/` on the
qualification host. This is a local evidence location, not a portable download.
No root cause was isolated. Lossless corruption excludes MP4 packaging as the
sole cause; successful memory execution does not establish correct inference.

Preparation is based on main `27ed658e` after the H3 fix. Task #1040 owns the
shared Candle convolution/command-completion candidate. Wan must consume that
reviewed candidate through the single Candle revision contract, never duplicate
its kernel changes or change CUDA policy. In particular, Wan's
`WanCausalConv3d::forward` applies Candle `Conv2d` to each temporal slice. This
is a relevant dependency, not evidence that the shared candidate fixes Wan.
The candidate is locally committed as
`bedc287458e0d890dd6ed1c298c99e991e066fe1` in #1040's isolated Candle worktree;
it is not yet pushed or pinned into Mold. Recheck its owner and revision before
integration.

## Work allowed during the hold

Read source/history and upstream implementations, validate request documents,
and run specifically selected small CPU tests with Metal/CUDA features absent.
Do not start a server, create a GPU device, load real model/reference weights,
allocate pressure, or run a nominally CPU test that probes available devices.
The VAE's `wan21_decode_matches_the_upstream_golden` fixture uses tiny synthesized
CPU weights and retained upstream outputs; it does not qualify Metal or the
installed checkpoint. All four existing Wan 2.1/2.2 encode/decode golden tests
passed during this preparation with `--no-default-features`; each took at most
0.03 s, with no real checkpoint loaded. Preserve `CorrectnessOnly` for Wan on
Metal.

## Prerequisites after release of the hold

1. Re-fetch main and inspect intervening changes. Record Mold/Candle commits,
   lockfiles, dirty diff, build command/features, binary hash and model hashes.
   A candidate and control must differ only in the change being investigated.
2. Acquire the shared exclusive reservation atomically. Confirm no other task
   is running local inference before starting anything; never stop another
   task's processes. Use a fresh scratch home and loopback-only server.
3. Keep automatic wired limits and existing safety policy unchanged. Record
   temperature/thermal state, native free+inactive, swap, compression, Metal
   allocations and recommended/headroom policy. Filesystem cache purge is an
   authorized preparation step; record whether it was used for each run.
4. Use the reviewed independent native guard: stop below 12 GiB free+inactive,
   above 256 MiB positive swap growth, non-normal pressure, failed/stale native
   sampling or timeout. Continue native sampling during cleanup. These are stop
   triggers, not a proof that memory never crosses them between samples.
5. Capture authoritative preview, durable batch response/detail, queue/activity,
   resource/device snapshots, server progress events and all native samples.
   Save cold/warm status, execution fingerprint and timestamps. A missing event
   stream or unobserved reservation transition is an evidence gap, not a pass.

## Gate 1: executable correctness isolation

Use the previous small request unchanged before increasing its shape:

```json
{
  "model": "wan21-t2v-1.3b:bf16",
  "prompt": "Medium wide shot in soft morning light. A red fox walks slowly through fresh snow in a quiet pine forest. Its paws lift small puffs of powder. The camera remains steady, showing the fox in profile against dark green trees.",
  "width": 512, "height": 288, "frames": 17, "fps": 16,
  "steps": 30, "guidance": 6.0, "seed": 1059,
  "output_format": "apng", "expand": false
}
```

Set the scratch server's UMT5 variant explicitly to Q8; record the resolved
component paths/hashes. Query `POST /api/generate/placement-preview` with
`{"request": REQUEST, "copies": 1}` and require a planned candidate with no
missing/download dependencies. Submit once through `POST /api/generation-batches`
with a fresh `client_batch_id` and `requests: [REQUEST]`; poll that exact batch.

Before claiming a fix, compare a bounded Wan operation against an executable
reference with identical inputs/weights. Start with the tiny upstream VAE
fixture, then an identical real-checkpoint latent decode only if its measured
working set fits the guard. Record F32 reference and BF16 candidate error,
finite values and per-frame/channel/spatial statistics; define tolerances before
examining candidate results. Inspect command errors, not only output finiteness.
The official reference is `Wan-Video/Wan2.1`, `wan/modules/vae.py`; preparation
read revision `9737cba9c1c3c4d04b33fcad41c111989865d315` after a fresh pull.
Refresh and pin the actual reference at execution. For a sampler comparison,
use the documented diffusers/Lightning flow-UniPC reference, not upstream Wan's
different `fm_solvers_unipc.py` schedule. Python stays in scratch reference work.

If the VAE comparison passes while the render fails, compare retained text
embeddings and an identical-input first denoise step before a full pipeline.
Matching seeds alone does not establish identical random tensors across
frameworks. Capture tensors explicitly. Keep the same Q8 encoder semantics or
label an FP16 reference as a changed configuration; never attribute that delta
to Metal. Diagnostic tensor-capture instrumentation is not implemented yet.

Require a recognizable, coherent lossless scene and correct 17-frame geometry,
then repeat MP4 and decode every frame. Numerical parity and visual inspection
are complementary. If any stage fails, retain the smallest failing evidence,
stop escalation and keep both issues open.

## Gate 2: #1059 admission margin and chain reservations

Each proposed larger shape needs its own correct unpressured baseline and
observed whole-run peak before pressure is allowed. Try legal `4k+1` frame counts
incrementally; do not infer safe memory from the 17-frame warm repeat.

Bound the pressure target by the smaller of 6 GiB and the cold native available
memory minus the measured baseline's incremental peak minus 14 GiB. A negative
bound means no safe pressure experiment. Grow touched random memory in 64 MiB
increments, refuse growth below 14 GiB, and keep the independent guard active.
Record actual resident helper memory, not just its requested size.

Unload only scratch-owned models using `DELETE /api/models/unload`, verify idle
state and stable samples, then bracket repeated authoritative previews on the
same shape around a memory-only planned/refused transition. Do not use
`/api/generate/estimate` as unified demand: it reports a diagnostic raw peak.
Warm-cache reclaim credit also means raw API headroom alone is not the boundary.
Record the unified demand, policy headroom, reservations and fingerprints that
the scheduler actually compares; add diagnostics first if they are not exposed.

On the accepted side, hold pressure constant from before submission through
completion and output inspection. On the refused side, submit the exact durable
request, verify that it does not dispatch, and cancel/settle that exact job
before releasing pressure. A strategy, dependency or device change invalidates
the memory-only comparison. Never relax guards to force a boundary. If no safe
overlap exists, report that limit and leave margin acceptance open.

At an already qualified sustained pressure level, submit an authored two-stage
chain via `POST /api/chain-jobs`: use the same model/shape/fps/steps/guidance/seed,
`motion_tail_frames: 0`, two stages at the already qualified per-stage frame
count, and `transition: "cut"` on stage 2.
Each stage's exact checkpoint, conditioning and shape must have passed the
correctness and admission-boundary gates at that pressure. Otherwise the result
is only chain behavior under pressure, not chain-stage margin qualification.
Stage 1 uses the baseline prompt; stage 2 has the fox pause and turn its head.
Use an authored cut because this text-only tier cannot carry visual conditioning
across an automatic split. Capture `/api/chain-jobs/{id}/events`, detail and queue
work items throughout. Verify parent/stage identity, device and fingerprint,
lease ownership transfer, no double charge and final release on every stage.
Decode both clips and the stitched result (twice the per-stage frame count;
34 frames when each stage has 17), including the cut boundary.
Then release pressure, unload scratch models and complete another real render
to prove recovery. Missing transition evidence requires another guarded pass.

## Gate 3: #1094 performance and capability decision

Only run this after correctness is established. Qualify 1.3B BF16 at 832×480 and
`wan22-ti2v-5b:q8` T2V and `wan22-ti2v-5b:fp16` I2V at 1280×704, starting at
17 frames and increasing only after each baseline fits. Gate visual correctness
separately for each exact checkpoint/conditioning/shape before timing; the 1.3B
pass does not qualify 5B. Keep fixed prompts, encoder tier, seed, steps, guidance
and artifact format, with a pinned source-image hash for I2V. An omitted
checkpoint or workload remains explicitly open.

Measure one cold load and at least three comparable warm repetitions separately.
Capture UMT5 load/encode, transformer load, every denoise step, VAE load/decode,
artifact encoding and total wall time. Use progress events, including typed
`PhaseDone`/wire `StageDone`, rather than estimating phases from RSS. Record
step-cache policy/hits, math attention, any block parking, ambient/thermal state
and peak memory. Synchronize diagnostic GPU boundaries so async submission time
is not reported as kernel execution time; distinguish instrumented timing runs
from ordinary end-to-end runs. Report each observation and warm median/range.

Do not use the thermally caveated August timings as a performance baseline or
the corrupt September render's short runtime as a speedup. A performance knob
needs same-input before/after correctness and timing evidence. Only then consider
promotion from `CorrectnessOnly` to `Supported`, with the matrix, docs, renderer
and capability tests changed together. CUDA behavior stays unchanged.

## Completion record

Hash raw evidence and outputs; independently review conclusions against them.
Update issue checkboxes only for satisfied acceptance, with source/build IDs and
remaining gaps. Any fix PR waits for its promised validation and final review;
no draft PR or capability promotion under the hold. At the end, stop only owned
children, verify their exit and scratch-port closure, sample recovered memory,
and release the reservation before another local task starts inference.
