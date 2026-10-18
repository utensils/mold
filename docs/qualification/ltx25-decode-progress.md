# LTX-2.5 decode progress and stalled-worker recovery

`Stalled` is an observation that no completed-work boundary has arrived within
`queue.no_progress_timeout_seconds` (default 300 seconds). It is not a failed
attempt or proof of a driver fault. The key is a profile-scoped SQLite setting;
`MOLD_QUEUE_NO_PROGRESS_TIMEOUT_SECONDS` overrides it. Accepted values are
0–86400; zero disables diagnostics. When a job is prepared, an async read of the live server config refreshes
the host's threshold. Relays never reload configuration from disk; profile
switches hydrate only the DB-owned queue preferences. Changing this setting never interrupts work or frees a GPU.

Queue listing/detail rows and per-job progress snapshots include an additive
`stall` diagnostic with `stage`, `last_completed_work_unix_ms`,
`elapsed_stall_ms`, `cancel_requested` and `device_still_held`. The existing
progress SSE stream carries `type: "stall"` with the same object; null clears
it after completed work resumes. Status enums remain unchanged. Older stream
readers ignore unknown progress events. Queued work keeps its durable order;
`blocked_device_reason` explains a required device held by stalled work only
when the host has sufficient evidence. Healthy devices remain schedulable.

The diffusion video decoder reports actual completed blocks and QKV, MLP and
projection chunks. CUDA synchronization precedes those counters, so a long
kernel cannot manufacture progress. Attempt cancellation is checked at these
boundaries and before subsequent work. A long kernel itself remains
noninterruptible. Logs explicitly identify whole-clip diffusion decoding;
convolutional spatial/temporal tiling settings do not describe this decoder.

`DELETE /api/queue/{id}` returns 202 while the running owner retains its device:
`{"cancel_requested":true,"device_still_held":true,"stall":null}` (or a current
stall object). The durable child remains `cancelling` until actual worker
return. Queued terminal cancellation retains the existing 204 response. A
successful cancellation request must not be presented as terminal cancellation
while the device is still held. Client disconnect or a dropped observer does
not release the owner or its resources.

## Explicit operator recovery

1. Inspect the queue, current stage/last completed work, per-device utilization
   and memory, host CPU/thread activity and available kernel/Xid logs. Record
   the host instance and affected job IDs. Long work and a driver fault can
   present similarly; utilization alone cannot distinguish them.
2. Request cooperative cancellation for the intended IDs and wait for actual
   owner return. Preserve device ownership throughout that wait; submitting
   another job is not a recovery action.
3. If the owner remains stuck, inventory every other running/queued job before
   an operator chooses a service restart. Graceful shutdown can itself wait on
   the live worker. Follow the service manager's stop/timeout/escalation policy
   and verify the old process has actually exited before starting a replacement.
   Escalating termination affects all work in that process. A GPU reset is a
   separate operator action, never an automatic response to this diagnostic.
4. Normal startup recovery fences stale claims, parks interrupted attempts and
   retains queue order for explicit resume; durable cancellation intent remains
   authoritative. In-memory leases disappear only because the previous process
   and its CUDA objects are gone. Inspect held/interrupted attempts and resume
   only intended work. No fail-by-deadline or guaranteed hard-stall recovery is
   promised.

## Reference and arithmetic boundary

The reference is Lightricks/LTX-2 commit
`e98f6d927d83c92ef2be6aec76c8440b19306e39`:

- `packages/ltx-core/src/ltx_core/model/video_vae/transformer/rope_math.py:39–107`
  scopes per-axis table reuse to one decode.
- `rope_math.py:133–171` owns frequency/table construction and axis rotation.
- `transformer/det_attn_rope.py:299` and the attention module's
  `rope_compute_dtype` default select F32 intermediates upstream.

Mold retains its existing f64 angle construction and working-dtype arithmetic.
The new axis tables are bounded by O(T+H+W), reused across Q/K and blocks at a
stage's geometry, then dropped; no expanded full-volume tables persist.
Regression tests require bitwise equality to Mold's previous expanded-table
implementation for F32 and BF16. A CPU-only official-reference fixture pins
upstream outputs. The independently reproduced existing BF16 intermediate
rounding difference has maximum absolute error 0.5 (two BF16 ulps near 64) on
that fixture; F32 differs by at most 7.63e-6. This change does not silently
replace the established arithmetic with upstream F32 rotation.

A CPU-only microbenchmark rotates a 16×16×16 grid with 64 channels twenty
times per process (three fresh-process trials, no model weights or GPU
context). Expanded tables carry 1,048,576 bytes; reused axis tables carry
4,096 bytes. Median elapsed time was 382.8 ms versus 344.6 ms. Process peak RSS
overlapped (roughly 22–24 MiB), so this small fixture demonstrates bounded
table payload and removes repeated construction; it does not establish an
end-to-end decode speedup or a peak process-memory reduction on the workstation. The two
CPU test entries are `rope_reuse_microbenchmark_legacy` and
`rope_reuse_microbenchmark_reused`; run each test binary separately with
`--nocapture` under the platform's process-memory timer for paired receipts.
They assert no timing threshold and introduce no runtime environment setting.

## Initial hardware observation

On 2026-10-18, the existing 0.36.0 workstation service admitted one atomic
four-child batch on an idle four-L40S fleet. Each child used installed
`ltx-2.5-22b-distilled:bf16`, 576×1024, 481 frames at 24 fps, eight distilled steps,
guidance 1, audio enabled and explicit single-clip length. All four devices were
assigned before observation. Stage 2 denoise took approximately 69 seconds; the
diffusion decode then stopped reporting completed work while all GPUs remained
at 100%, using approximately 33–38 GiB, and each owner thread used approximately
one CPU core. Available kernel logs showed no Xid. The old log incorrectly
claimed spatial tiling, and its preview retained the prior transformer stage.

The campaign requested cooperative cancellation only for its four IDs at the
20-minute observation cap. At that point every durable child was `cancelling`
and every device remained owned; cancellation was not terminal. During that initial measurement checkpoint, no service restart, GPU reset or
device-policy mutation was performed. These observations
reproduce missing progress and cancellation boundaries; they do not establish a
driver, kernel or CPU cause, or demonstrate that the large shape completes
within a particular time. Subsequent measurements and final owner settlement
must be recorded separately from this initial observation.

The original workers still had not returned after roughly fifty minutes. Under
explicit deployment/testing authorization, cleanup stopped the unchanged
production service at 19:15:57 UTC. Its existing 105-second stop policy escalated
to SIGKILL (exit signal 9); at 19:17:48 UTC the old PID and every GPU context
were verified absent. The unchanged service was started at 19:17:56 UTC. This
was process termination, not a cooperative cancellation or proof of automatic
hard-stall recovery. Twenty unrelated durable jobs had arrived during the
campaign. Their identities and order were captured, preserved across startup
pause, and only those previously queued jobs were individually resumed after
verification; subsequent arrivals were left untouched. Production generation
then progressed on all four devices. Only the four original paused campaign
rows were cancelled to settle their retained intent. No GPU reset, device-policy
change, production-binary replacement or model mutation was performed.

## Bounded shipping-runtime qualification

A separate idle four-L40S window on 2026-10-18 completed in 309 seconds,
including shutdown and restoration of the unchanged production service. The
CUDA/H3 shipping build carried private snapshot commit
`84d32c8c294cb22e4cdd75c4005134049a81238b` (binary SHA-256
`74a18f84ff155a866576017cfbf9985a40e82adb99263294626759c23f5d4444`).
A file manifest compared its production Rust sources with branch commit
`bcb7d828469bf65fce9a2cc677abb007bae3295f`: all production files matched;
the sole differing Rust source was a test-only routes file. The private
snapshot identity is not the final published branch identity.

- LTX-2.5 distilled BF16 completed 481 frames at 24 fps, 128×128, eight steps,
  guidance 1, audio enabled and a single 481-frame clip. The downloaded export
  and published Library file had identical SHA-256. Full media decoding counted
  exactly 481 H.264 frames and an AAC stream; video duration was 20.041667 s.
- Installed `minimax-h3-ref2va:comfy-pruned-int8-turbo-4step` completed three
  256×256, 107-frame, 24-fps audio renders, each with five terminal-inclusive
  steps, guidance 0 and strength 1. Inputs were real HEVC MOV, VP9 WebM and VFR
  MP4 fixtures. Standard client candidate/upload/finalize negotiation produced
  canonical MP4 references consumed by the actual runtime. Canonical HEVC and
  VP9 inputs had 48 frames at 24 fps; VFR had 34 frames at 17 fps, each lasting
  2000 ms. The runtime prepared each into its own 24-Hz reference shape. Every
  export decoded all 107 H.264 frames with AAC, and matched its Library digest.
- With an intentionally short two-second observational threshold, all five jobs
  (four completions and one subsequent cancellation) emitted real stall objects
  and explicit null recovery after new completed work. This exercises reporting
  during legitimate quiet work; it does not reproduce a hard driver stall.
- The subsequent warm 128×128 LTX cancellation was accepted with HTTP 202 and
  `device_still_held:true` at decoder counter 30/30. The worker returned about
  three seconds later, and no export or Library row was published. This proves
  cancellation after the final decode boundary before publication, rather than
  cancellation between unfinished decoder units.

First, middle and final decoded frames from all four successful exports were
saved as twelve small PNGs and four contact sheets. RGB samples were finite,
nonempty, nonblack and nonconstant. Visual inspection shows substantial
artifacts at these deliberately small shapes; these are technical media and
transport checks, not aesthetic or semantic quality qualification. Neither this
window nor the CPU RoPE microbenchmark establishes a GPU speedup or completion
bound for the original 576×1024 shape.

Every private attempt used separate homes, credentials and output directories;
only its explicitly recorded IDs were cancelled. Fresh live-queue, durable-row
and GPU-owner checks preceded service interruption. All sixty-seven historical
held/paused external rows retained their exact identity, order, owner and pause
intent. The twenty jobs preserved during the earlier recovery were each traced
by queue-job identity to a published result, rather than inferred complete from
an empty queue. New unrelated arrivals always take precedence over further UAT.

A separately preflighted, single-job window qualified cancellation between
unfinished decoder work units: LTX-2.5 distilled BF16, 256×256, 481 frames at
24 fps, eight steps, guidance 1 and audio. At 20:36:13.317810 UTC the exact-job
observer saw `Diffusion video decode QKV chunks` at **10/16**. Its DELETE
returned HTTP 202 at 20:36:13.319156 with `cancel_requested:true` and
`device_still_held:true`. The cancelled client returned at 20:36:13.703538;
the private queue emptied, and no export or Library row was published. This
qualifies cooperative cancellation between unfinished decoder chunks, not
interruption of an in-flight kernel. The task server exited normally on SIGTERM,
all its GPU contexts disappeared, and unchanged production was restored within
269.574 seconds for the whole window. All sixty-seven historical external
rows remained unchanged. No additional H3 render or full LTX completion was
submitted in this window.
