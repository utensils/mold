# Qwen Image 2.1 prefix KV cache

The request-local cache follows the official checkpoint's causal-condition
transformer: text uses timestep zero and attends only to earlier text. Each
block therefore retains its text K (after normalization and RoPE) and V from
the first denoising step. Later steps process only image queries, attending to
the retained text plus the current image keys/values. This is not approximate
denoise-step skipping or cross-request prompt reuse.

The prepared branch borrows its transformer and immutable conditioning. Positive
and negative CFG prompts have separate caches. Cache publication requires a
successful complete prefill; return, error and cancellation drop request state.
`force_contiguous()` copies exactly the retained slice: Candle's `copy()` clones
the entire backing storage, while `contiguous()` may return an existing view.

Retention is capped at 512 text tokens per branch. Longer prompts are not
truncated: they use the original full forward. Admission adds a conservative
1 GiB per sample outside the activation area estimate, sufficient for two
512-token branches across 32 blocks at width 4096 and F32. Actual retention for
a 29-token, guidance-one prompt is 29 MiB. CUDA BF16 needs half that storage;
CUDA execution is not qualified by the Metal measurements below.

## Reproduction

Install the pinned official artifacts using Mold, close other GPU workloads,
then run:

```sh
QWEN_IMAGE21_MODEL_ROOT=/path/to/mold/models \
  cargo test -p mold-ai-inference --lib --features metal \
  official_prefix_cache_metal_parity_and_timing -- --ignored --nocapture
```

`QWEN_IMAGE21_BENCH_REPEAT=12` repeats the receipt's prompt twelve times to
exercise a longer prefix. The probe loads the real Qwen3-VL encoder and all
32 transformer blocks, uses F32 Metal, 1024x1024, seed 210001, and the first four
steps of the 40-step schedule. It compares both paths on the same latent at
each step, advances using the uncached prediction, synchronizes before/after
each timing, alternates pair order and excludes prefill from steady timings.
It does not benchmark VAE decode or claim full-render acceleration.

The source checkpoint is
[Qwen/Qwen-Image-2.1 at b3179ad](https://huggingface.co/Qwen/Qwen-Image-2.1/tree/b3179ad355be050328e483a9dfdd9e60cd62adfa).
The original full-render receipt is `qwen-image-2.1-metal-uat.json`.

## Observed on 2026-09-20

Apple M5 Max (40 GPU cores, 128 GiB unified memory), official checkpoint,
F32 Metal. Both prompt lengths produced bit-identical predictions at all four
tested steps (maximum absolute and relative RMS error both zero).

| Prefix     | Uncached reuse steps | Cached reuse steps | Result                                                                                                                            |
| ---------- | -------------------: | -----------------: | --------------------------------------------------------------------------------------------------------------------------------- |
| 29 tokens  |            26.4185 s |          26.8411 s | No measured gain; this exploratory run overlapped a tiny Metal test and compilation, so it is not a clean performance comparison. |
| 271 tokens |            25.8385 s |          24.2836 s | 1.064x throughput, 6.0% less time, across three paired steps. No concurrent GPU test.                                             |

For the 271-token case, uncached/cached times were respectively
8.4033/7.9158, 8.6647/8.0866 and 8.7706/8.2812 seconds. The initial prefill
was 7.7704 seconds with extraction versus 8.1996 without; exclude that
cold/order-sensitive pair from the performance claim. This is a small local
sample, not a cross-device benchmark or a CUDA qualification.

Synthetic tests cover multiple blocks, heads, batch rows, masked padding,
changing latents/timesteps, separate positive/negative prompts, fresh-request
isolation, failed prefill, exact compact storage, and over-limit fallback.
CPU predictions are exact. The tiny one-head Metal fixture changes GEMM shape
and differs by at most 4.52e-5, within its 1e-4 absolute bound; the multi-head
Metal fixture is exact. The real-checkpoint probe additionally enforces a
1e-3 maximum absolute and 1e-4 relative RMS bound rather than relying only on
visual inspection.
