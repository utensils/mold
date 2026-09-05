# H3 Metal memory audit — 2026-09-05

Status: **component regression fixed; end-to-end Metal render unqualified**.
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

The watchdog **refused before launching the model process**: the baseline
had less than 24 GiB available. No full-model GPU allocation, output video,
or end-to-end H3 qualification resulted. The reservation was released.
The qualification-only Candle instrumentation and path patches are not
part of the shipped change; the repository's Candle pin is unchanged.

## Remaining acceptance

- Complete per-phase runtime measurements for the intended Turbo/default shape; the small base admission budget above is established.
- Qualify real conditioner, DiT and VAE phases under an exclusive hardware
  reservation, allocation ceiling, and external host-pressure watchdog.
- Retain a complete conditioned FL2VA video with audio, inspect its output,
  compare an equivalent CUDA/reference case, and record peak memory.
- Keep both issues open and `CorrectnessOnly` until the relevant evidence
  exists; do not infer a 48 GiB fit from disk sizes or component tests.
