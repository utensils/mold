# Qwen Image 2.1 Metal performance qualification

This experiment compares the original F32 math attention path with Metal SDPA,
contiguous Q/K normalization, interleaved rotary embedding, and compact cached
timestep modulation. A separate BF16 candidate changes transformer weights,
conditioning and latent storage together; scheduler arithmetic, rotary tables,
the text encoder and VAE remain F32. It is not a transformer-only precision
measurement.

The optimized target attention requires Metal, a supported head dimension and
no bias. Causal text-prefix attention and padded target batches retain math
attention. `MOLD_ATTN=math` (including its legacy `sdpa` alias) uses the shared
override parser and disables fused attention, projection/RoPE operations and
compact modulation. CUDA attention policy is unchanged. Precision has a
separate control: `MOLD_QWEN_IMAGE21_DTYPE=f32` keeps the Metal denoiser in F32;
`auto` (the default) and `bf16` select BF16. CPU and CUDA retain their existing
precision policy. To reproduce the original Metal computation, combine
`MOLD_QWEN_IMAGE21_DTYPE=f32 MOLD_ATTN=math`. Both variables participate in
frozen engine identity. Admission retains its conservative F32 memory budget;
these measurements do not qualify a smaller-memory placement.

## Reproduction

Install the official checkpoint with Mold and close other GPU workloads. Run
one process at a time, using a fresh output directory for each case:

```sh
QWEN_IMAGE21_MODEL_ROOT=/path/to/mold/models \
QWEN_IMAGE21_BENCH_OUTPUT=/tmp/qwen21-math \
QWEN_IMAGE21_BENCH_MODE=math \
cargo test --profile dev-fast -p mold-ai-inference --features metal --lib \
  official_metal_mode_benchmark -- --ignored --nocapture --test-threads=1
```

Modes are `math`, `fused`, `bf16`, `optimized` and `bf16-optimized`.
`optimized` includes SDPA, projection/RoPE changes and compact modulation;
`fused` changes target attention alone. BF16 modes keep encoding and decoding
in F32. The default is 1024x1024, guidance 1, seed 210001, and 40 steps.
`QWEN_IMAGE21_BENCH_PROMPT` and `QWEN_IMAGE21_BENCH_SEED` select another case.
`QWEN_IMAGE21_BENCH_LIMIT` can execute a prefix of the schedule, but such a run
does not produce a decoded image or qualify a complete render.

The harness synchronizes around each transformer forward and records first-step
and subsequent forward timings separately. Whole denoise timing also includes
scheduler updates and per-step finite checks/readback. It saves the first four
predictions, final latent, decoded PNG, and JSON receipt. Metal allocation
samples include allocator pools and are post-step observations, **not peak
working memory**. Components load
sequentially; these totals are not measurements of an already-loaded desktop
session or the eager placement path.

## Correctness boundaries

Synthetic tests exercise rectangular Q/K lengths, F32 and BF16 SDPA, masked
fallback, padded batches across multiple cached steps, and exact compact
modulation parity. Real-checkpoint image comparisons additionally test complete
40-step trajectories. Fused arithmetic can change the last bits; BF16 can
change composition and detail, so numerical differences alone do not establish
visual equivalence.

The F32 rotary tables and singleton timestep modulation row follow the
[upstream Qwen Image 2.1 transformer](https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/transformers/transformer_qwenimage21.py).
The checkpoint is pinned in the [prefix-cache qualification](qwen-image-2.1-prefix-kv.md).

## Local results

Apple M5 Max, 128 GiB unified memory, macOS 27.0 (26A428), release-derived
`dev-fast` build. Base source `bac5978129f60f22cc985185fb8b21d41e56e582`
plus this patch; Candle fork `bf2cd29a791dbc053df826b4377d092c3809d17f`.
Mold was closed and renders ran sequentially without concurrent compilation.

| Teapot, 1024², 40 steps |   Denoise | Total including loads and decode |
| ----------------------- | --------: | -------------------------------: |
| Original F32 math       | 453.061 s |                        475.876 s |
| Optimized F32           | 378.110 s |                        420.438 s |
| Optimized BF16          | 309.446 s |                        323.574 s |

The F32 candidate used 16.5% less denoise time (1.20x throughput) in these
individual runs. Load times varied substantially, and the baseline's step
times drifted from 7–9 seconds to 13–16 seconds. Post-step baseline allocation
stayed near 33.03 GiB. These observations do not identify the drift's cause or
establish a repeatable cross-device speedup.

All 40 predictions, updated latents, and decoded outputs were finite. Relative
RMS difference of the final latent was 0.000301. The decoded RGB images differ
by at most one 8-bit level, with mean absolute difference 0.006912 levels.
The original and optimized F32 teapots are visually consistent.

The BF16 teapot used 31.7% less denoise time than the original and 18.2% less
than optimized F32. Its final latent relative RMS difference from the original
was 0.05839; mean absolute RGB difference was 2.524 levels. It preserved the
composition and produced a coherent teapot with small texture changes. The
owner accepted both the original/optimized F32 comparison and the BF16 teapot.

| Café lettering, 1024², 40 steps |   Denoise | Total including loads and decode |
| ------------------------------- | --------: | -------------------------------: |
| Optimized F32                   | 387.373 s |                        402.724 s |
| Optimized BF16                  | 302.319 s |                        316.535 s |

The lettering case used seed 210002 and asked for a coffee shop with the sign
“MOON CAFE” and two cups. Both outputs preserve the exact requested lettering
and cup count. BF16 changes the right-hand chair's geometry and some textures;
mean absolute RGB difference is 3.057 levels against optimized F32. This is a
quality check, not pixel equivalence. BF16 used 22.0% less denoise time in this
second comparison. All intermediate checks and decoded outputs were finite.

## Decision and evidence

Enable the Metal optimizations and BF16 denoiser by default, with the documented
F32 and math fallbacks. Preserve F32 encoding, decoding, rotary tables, and
existing normalization/scheduler accumulation. Leave admission estimates
conservative and other backends' precision boundaries unchanged.

This is a two-prompt, guidance-one, 1024² qualification on one M5 Max. It does
not establish quality equivalence for all prompts, long contexts, resolutions,
CFG settings or other Macs. The standalone harness uses the actual model
components sequentially; it does not constitute desktop/server end-to-end UAT.

[Machine-readable results](qwen-image-2.1-metal-performance.json) retain prompts,
seeds, all forward timings, phase totals, allocation samples' maxima and output
differences. Full receipts, saved tensors and images are in the local artifact
directory recorded there. Initial debug-build probes and the intentionally
stopped candidate run are excluded from these performance claims.

Fable 5.1 at medium effort reviewed the dispatch, caches, precision boundaries
and runtime integration. Final checks passed:

- 25 Qwen Image 2.1 tests with default settings, and the same 25 with
  `MOLD_ATTN=math MOLD_QWEN_IMAGE21_DTYPE=f32`; the two real-weight harnesses
  remain ignored in ordinary test runs.
- Four runtime-environment tests, the server's complete engine-variable
  classification contract, and its precision-identity canonicalization test.
- Scoped inference/server Clippy with Metal, library and test targets,
  `-D warnings`; workspace Rust formatting and patch whitespace checks.
- Five complete real-checkpoint render cases, each with 40 finite denoise
  steps and a finite decoded output, summarized in the JSON receipt.

The desktop application was not replaced or restarted. These are worktree
changes plus standalone qualification artifacts.
