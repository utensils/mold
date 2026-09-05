# H3 Metal memory audit — 2026-09-05

Status: **attention and INT8 regressions fixed; reduced-size Metal FL2VA render retained**.
Tracks [#1164](https://github.com/utensils/mold/issues/1164) and
[#1542](https://github.com/utensils/mold/issues/1542). Audited base:
`e1bf871c799fd6dc88628dc69fcbb98fbd43c692`; Candle:
`744ae3b83cfac18db28107a353c449cc9b80d4ec`.

## Local memory controls

On this 48 GiB Apple Silicon host, the current-main CLI reports
`iogpu.wired_limit_mb = 0` (automatic), no persistent override, and
40,200,896,512 bytes recommended/effective device capacity. It exposes
`mold system metal-memory status`, `set`, and `reset`. This is a **sysctl**,
not an ioctl or a per-process RSS limit. The budget also preserves host
headroom. No system limit was changed during this audit. The installed
0.26.0 CLI predates these commands; the current-main qualification binary
was used for the read-only check. All 11 administration-protocol tests passed (verified readback, limits, permission failure, persistence and rollback); these use test doubles and do not change the live kernel setting.

## Artifact size is not live residency

Header-only inspection of the installed compact FL2VA artifacts found:

| Component        |     File bytes |
| ---------------- | -------------: |
| INT8 ConvRot DiT | 20,970,379,616 |
| NVFP4-AWQ Qwen   | 15,687,142,551 |
| Video VAE        |  5,207,808,496 |
| Audio VAE        |    605,254,808 |

These are disk sizes, not a measured simultaneous working set. Metal's
`MetalStreamed` Qwen authority retains 1,052,855,836 host parameter bytes
and 1,191,583,200 device parameter bytes, plus separately charged staging
and activation workspaces. It completes and drops each language layer.
The DiT also streams main blocks. CUDA retains its separate accelerated
Qwen route. `private_h3_unified_target_peak_bytes` takes the largest phase's
host-plus-device sum, not the sum of all artifact sizes.

A new GGUF layout is therefore a conditional follow-up, not a demonstrated
prerequisite. Neither these static facts nor an accepted budget proves
that a complete Metal render fits.

## Reproduced attention lifetime regression

A model-free, isolated native test uses F32 tensors shaped
`[1, 2048, 4, 8]`, with 512 query rows per chunk. Each score chunk is
16 MiB. The original implementation retained **135,462,912 bytes** beyond
its synchronized input baseline after attention returned. Adding only a
completion fence still retained **68,419,584 bytes**: small chunk outputs
could reuse large pooled score buffers and keep them alive in the parts
vector.

The correction allocates one output before the score passes, copies each
chunk into its destination, drops temporary handles, and completes Metal
work before the next chunk. The same test then retained **1,572,864 bytes**
and its output matched the analytic constant-value result. The ordinary
attention suite passed **22 tests**, including real Metal versus CPU
numerical parity; the isolated allocation test passed separately.

Run the allocation test alone, because native allocation accounting is
process-wide:

```sh
nix develop -c cargo test -p mold-ai-candle --features h3,metal --lib \
  metal_attention_releases_chunk_temporaries_before_return -- \
  --ignored --nocapture --test-threads=1
```

The test's buffers are small and contain no model weights. These retained
allocation samples are not peak RSS measurements or whole-model UAT.
The CPU-only `h3` library check, Candle single-source contract, and H3 attention release contract passed. Clippy passed with only the existing unused re-export imports allowed (`-D warnings -A unused-imports`).

CUDA FlashAttention dispatch and arithmetic are untouched; this Mac cannot
execute a CUDA hardware regression.

## Reproduced INT8 row-chunk lifetime regression

The subsequent guarded server attempt completed multimodal conditioning
(56.4 seconds), loaded the VAEs, and encoded visual conditions. During the
first transformer evaluation, repeated 115,605,504-byte allocations reached
8,570,241,024 allocated bytes, and the next allocation was refused by the
8 GiB ceiling. The process exited cleanly; kernel pressure stayed normal
and swap did not grow. There was no completed video.

A model-free `ComfyInt8ConvRotLinear::forward_reference` regression with
1024×1024 activations, 4096 output features and 64-row chunks reproduced
the pooled-buffer retention: **345,374,720 bytes** after forward. Metal now
copies each row result into one destination and completes the row pass
before proceeding; retained allocation fell to **29,491,200 bytes**.
The CUDA native early return and CPU arithmetic are unchanged. Six ordinary
INT8 tests pass, including F32 CPU parity and BF16 single-pass Metal parity
with bias and a partial final chunk. CPU BF16 matmul is unavailable, so
the BF16 comparison is explicitly a chunk-assembly check, not a CPU oracle.

The ignored native allocation regression must run alone:

```sh
nix develop -c cargo test -p mold-ai-candle --features h3,metal --lib \
  metal_reference_releases_row_chunk_temporaries -- \
  --ignored --nocapture --test-threads=1
```

## Upstream comparison

- ComfyUI `250b2e9551a7bc7a8ebb5beb07e0fecd2983e04a`,
  `comfy/ldm/minimax/model.py:195`, dispatches full noncausal attention
  through its optimized attention implementation.
- PipeNetwork/minimax-h3-mlx
  `b2f7e4d2b7861cefe68b75e4b59ab81cc4e7c318`,
  `minimax_h3_mlx/dit.py:163`, uses MLX scaled dot-product attention.
  Its README's large-host T2VA measurements do not qualify Mold's compact
  INT8/NVFP4 conditioned FL2VA path on a 48 GiB Mac.
- [minimax-h3-mac](https://github.com/Bambushu/minimax-h3-mac/tree/46054af3df86a52ac92fdd1a6c648bb6fc2c48bb)
  reports a 48 GB M5 Pro ComfyUI workflow using a GGUF conditioner and
  sequential loading. That is useful comparison evidence, not Mold parity.

The fix preserves the existing full-attention mathematics and only changes
Metal resource lifetimes. No upstream Python or MLX runtime is shipped. The pinned H3 license was fetched again; its SHA-256 still matches `59b99642b95ea21630e311198ddbfffbfe05aadba0c2f5d884cbdf4efcc90f44`.

## Admission and guarded-load result

The allocation-free probe authenticated and opened the installed checkpoint
set for a first-frame-conditioned base compact request: 256×256, 107 frames,
21 schedule points, seed 42, synthetic RGB gradient, and a short prompt.
It returned **8,576,179,488 bytes (7.99 GiB)** for the unified peak and zero
separate host increment, because the host charge is already included. This
is a calculated budget for that exact request, not an observed runtime peak
or a budget for the default 1344×768 shape or Turbo adapter.

A separately instrumented, unshipped qualification binary was prepared with
an **8 GiB native Metal allocation ceiling** (checked before allocation),
plus an external watchdog requiring 24 GiB reclaimable memory before launch,
a 12 GiB available-memory floor, normal kernel pressure, and no more than
256 MiB swap growth. The intended run uses the installed Turbo 4-step 768p
tier at 256×256 and 107 frames with a synthetic first frame, in a separate
`MOLD_HOME`; its own admission must recalculate its budget.

The initial watchdog attempts **refused before launching the model process**: the baseline
had less than 24 GiB available. No full-model GPU allocation, output video,
or end-to-end H3 qualification resulted. The reservation was released.
The qualification-only Candle instrumentation and path patches are not
part of the shipped change; the repository's Candle pin is unchanged.

After memory was freed, the branch was rebased onto
`b880155ff72b84c4f1df50368cf98bf64c024358` and the guarded binary rebuilt
at `a0bb2e3edf57d32509614f513becf0be8267594b`. The forced-local attempt
passed the baseline guard, but failed before inference after 198.6 seconds:
`public runtime registry is incomplete: prepared_attempt=false;
budget_echo=false; typed_attention=false; runnable_contract=true;
family_registry=true`. The local engine constructor uses the generic
factory with admission-only authority; the server owner prepares the
one-shot runnable attempt separately. This is a dispatch integration
failure, not evidence that the checkpoint exceeds memory. Kernel pressure
remained normal and swap did not grow. No video was produced.

## Retained reduced-size Metal render

The corrected production code (equivalent to `4f74f8e7`; the instrumented
binary was built before that commit) completed the same server-route request
on an **Apple M4 Max with 48 GiB unified memory**:

- `minimax-h3-fl2va:comfy-pruned-int8-turbo-4step-768p`, first-frame conditioned,
  256×256, 107 frames, 24 fps, seed 42, guidance 0, five sigma points/four evaluations.
- 1,286.3 seconds wall time including admission; archived generation time
  1,080.1 seconds.
- H.264 video: 107 frames, 4.458333 seconds. AAC audio: stereo 32 kHz,
  4.458344 seconds. Both streams decode without errors.
- Highest reported native Metal allocation: **7,757,168,640 bytes**; the
  independent allocation ceiling remained **8 GiB**. The reported sample is
  not a process-RSS peak. Minimum observed host availability:
  **17,138,401,280 bytes (15.96 GiB)**. No swap growth, kernel pressure change,
  allocation refusal, or non-completed Metal command was recorded.
- The sampled frames preserve the synthetic sky/ground/sun composition and
  show the requested moving cloud. Faint colored artifacts remain at this
  reduced size, which is below the tier's recommended resolutions. This is
  an execution smoke test, not a default-resolution quality endorsement.
- Audio is non-silent (overall RMS about −49.2 dBFS), with no NaNs or
  infinities. Signal validation is not a listening evaluation.

The build initially left too little available memory for the unchanged
24 GiB starting guard. Releasing the filesystem cache with macOS `purge`
restored headroom; no application/service was stopped and no sysctl changed.
The successful attempt is retained under
`/Volumes/ExternalStorage/mold-h3-metal-qualification/evidence/server-first-frame-256-int8-cold/`.
The clip is `mold-guarded-int8-256.mp4` in that qualification root, SHA-256
`280284c009f3cdb9e5466ee81b69ee92e1e03ad93b5f98edec5a42c106cb3c4f`.
Its pipeline provenance is
`d1e10963a4ef3ea9430df9389d989341ad7e67cbaa4eaaae52a1a288117031e8`.

On HAL9000's RTX 4090, the uninstrumented `4f74f8e7` CUDA build passed
**20 H3 attention tests and 9 INT8 tests**. A non-skipping, model-free
257-row H3 FlashAttention probe created a CUDA device and measured maximum
absolute error **0.00390625** against the CPU dense reference (limit 0.02).
These checks exercise CUDA kernels.

The matching first-frame request also completed on HAL9000's existing CUDA
server (`10ebd23a`, SM89 RTX 4090), using the same installed checkpoint,
adapter, prompt, source image, seed, dimensions and schedule. The differences
between that server's H3 execution source and this branch are the Metal-only
fixes, timing reporting and installed-artifact verification changes; this
comparison is not a run of a newly deployed CUDA server. Its archived runtime
was **94.837 seconds**. Both outputs contain the same 107 video frames and
matching 32 kHz stereo durations. Whole-video SSIM is **0.988634** and PSNR
**37.762865 dB**. Decoded audio correlation is **0.973562**, with RMS difference
**0.00080596** over 285,334 interleaved samples. Sampled CUDA frames show the
same scene, cloud motion and faint colored artifacts. These are measured
comparisons for one reduced-size fixture, not bit-exactness or general quality
thresholds. The reference and its batch identity are retained in the sibling
`hal9000-cuda/` evidence directory.

## Forced-local owned-attempt repair

The CLI now prepares one exact H3 request through the server's shared owned
runtime boundary, preserving the device grant, identity validation,
cancellation and terminal media checks. Local batches, chains and Ref2VA
references are refused before artifact preparation until those paths have
per-request ownership. The process's existing Ctrl+C handler cancels the
active local H3 attempt instead of exiting underneath GPU work.

On HPE/plato, the repair (`bf25f9c6`, integrated here as `bd4bb0fa`) passed
CUDA CLI checking and Clippy with warnings denied, 53 focused server
ownership/identity/budget/publication tests, and 11 CLI cancellation and
request-provenance tests. Both CLI and server test harnesses compile.
The parent reviewed the owned-attempt extraction; an independent review of
the attention and INT8 lifetime fixes found no actionable issues, including
checking the pinned Candle `slice_set` semantics. This review does not
substitute for the retained Metal run or the separate CUDA hardware tests.
An additional 31 H3 quantized-operation tests passed on Metal.

The final guarded forced-local render at `bd4bb0fa` completed in
1,270.1 seconds wall time (1,076.0 seconds reported generation time).
Its MP4 SHA-256 is **identical to the retained server render above**, proving
byte-for-byte output agreement for this exact request through both owners.
Minimum host availability was 17,917,067,264 bytes (16.69 GiB); maximum
reported native allocation was 7,757,168,640 bytes under the same 8 GiB ceiling.
Pressure stayed normal, swap did not grow, and no Metal command failure
occurred. The separate qualification process and reservation were released.

## Remaining acceptance

- Complete per-phase runtime measurements for the intended Turbo/default shape; the small base admission budget above is established.
- Extend the completed reduced-size conditioner, DiT and VAE evidence to
  default-resolution requests under the same safety controls.
- Qualify default-resolution quality and memory separately from the completed
  reduced-size CUDA/Metal comparison.
- Keep both issues open and `CorrectnessOnly` until the relevant evidence
  exists; do not infer a 48 GiB fit from disk sizes or component tests.
