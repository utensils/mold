# Qwen Image 2.1 CUDA performance qualification

This ledger records the CUDA performance of `qwen-image-2.1:bf16` on the
path mold shipped in v0.32, and the evidence for the fast path that replaces
it. The raw receipts are in
[`qwen-image-2.1-cuda-performance.json`](qwen-image-2.1-cuda-performance.json).

Host: plato, NVIDIA L40S 46 GB, `CUDA_VISIBLE_DEVICES=2`. GPU 1 was retired
during the campaign after fatal PCIe errors, so every number below comes
from GPU 2. The checkpoint is HF snapshot `790c926`. Harness builds use
`dev-fast` with `cuda,cudnn,flash-attn` on Candle fork `bf2cd29`.

## The v0.32 path

At v0.32 every Qwen Image 2.1 CUDA render ran the same way:

- chunked math attention under the image policy;
- Wan-style composite RoPE on BF16 tables;
- modulation broadcast to every token;
- a hand-written F32 LayerNorm;
- an im2col VAE.

`Qwen21ExecPath::legacy()` is that arithmetic. Setting
`MOLD_ATTN=math MOLD_CONV=im2col` selects it on CUDA.

## Legacy byte-identity reference

The production v0.32.0 binary rendered the reference images (the
`mold.service` ExecStart, forced local, `--no-expand`):

- first on GPU 1;
- then again on GPU 2, which gave byte-identical PNGs.

The images, the CLI transcripts, `render.sh` and a README are in
`/storage/mold/fixtures/qwen_image21/legacy-v032/`.

| Case | PNG sha256 | Decoded 8-bit RGB sha256 |
|---|---|---|
| 1024² g1, seed 210001 | `f1fa6bc2…c8e5ae` | `a6d45454…525df6` |
| 1024² g1, seed 210002 | `0b0a1d9b…b6f859` | `a3a19249…1a7956` |
| 1344x768 g4 + negative, seed 7 | `518bd8e7…50e74b` | `9868de33…fad4c4` |

The prompts are the bakery example from the prompting guide and the fox
prompt with negative `blurry, lowres, watermark`. Seed 210001 was rendered
twice in separate processes, with identical bytes.

The harness's `legacy` mode reproduces the same decoded RGB hash for the
1024² and 1344x768 cases. That mode includes this branch's VAE changes: the
conv scope and banded convolutions, with every v0.32 canvas unbanded. So
those changes keep the v0.32 pixels. Compare decoded RGB rather than PNG
bytes, because new metadata chunks change the file but not the pixels:
`magick <png> -depth 8 rgb:- | sha256sum`.

## Baseline timings

### CLI (v0.32 binary, cold process, GPU 2)

| Case | Denoise | VAE decode | Total |
|---|---:|---:|---:|
| 1024² g1 (3 runs) | 38.0 / 38.2 / 38.3 s | 1.1 s | 39.2–39.7 s |
| 1344x768 g4 + negative | 74.4 s | 1.0 s | 75.6 s |

### Kernel harness, `legacy` mode (40 steps)

The harness synchronizes around every forward. Peak memory is sampled by a
`mem_get_info` poller at 250 µs, and "increment" means above the resident
transformer.

| Case | First step | Steady step (median) | Denoise | Denoise peak increment | VAE (im2col) | VAE peak increment |
|---|---:|---:|---:|---:|---:|---:|
| 1024² g1 | 1.090 s | 0.968 s | 38.87 s | 0.91 GB | 2.38 s | 12.31 GB |
| 1344x768 g4 + negative | 1.996 s | 1.902 s | 76.72 s | 0.94 GB | 2.37 s | 12.11 GB |
| 2048² g1 | 13.20 s | 13.13 s | 525.5 s | 3.69 GB | 5.79 s | 24.90 GB |
| 2752x1536 g1 | 13.46 s | 13.40 s | 536.9 s | 3.66 GB | 5.55 s | 25.07 GB |

- The resident BF16 transformer takes 14.84 GB.
- At 2K the math score matrix dominates: 13.1 s per step, 13.6 times the
  1024² step for 4 times the tokens. That is why flash attention has to come
  before 2K.
- Neither 2K case was admissible at v0.32, whose limit was 1.8 MP. They were
  rendered through the harness. Both images were checked visually: the
  lettering is correct and the composition is sound.

### M5 gates (L40S, BF16, warm denoise)

Harness denoise seconds, 40 steps, `fast` mode (`Qwen21ExecPath::cuda_fast()`:
FlashAttention through the joint-layout segment dispatch, the fused
q/k RMSNorm + transpose + F32 RoPE kernel, cached F32 RoPE tables, compact
modulation, fused adaLN, dtype-rounded timestep). Every gate passes.

| Case | v0.32 | Required | Fast path | Speedup |
|---|---:|---:|---:|---:|
| 1024² g1 | 38.9 s | ≤ 22 s (stretch ≤ 18.5 s) | **15.2 s** | 2.56x |
| 1344x768 g4 + negative | 76.7 s | ≤ 42 s | **31.0 s** | 2.47x |
| 2048² g1 | 525.5 s | ≤ 85 s | **82.6 s** | 6.4x |
| 2752x1536 g1 | 536.9 s | ≤ 90 s | **85.9 s** | 6.2x |

The CLI agrees end to end (the branch binary, forced local, `--no-expand`,
cold process, GPU 2): 1024² denoise 15.0 s / total 15.8 s; 1344x768 g4 +
negative 29.6 s / 30.4 s; 2752x1536 82.4 s / 97.6 s. At 2K the CLI total also
covers loading the weights, encoding, and parking the BF16 text encoder in
host RAM so the decode fits beside the transformer.

## Fast path

### Per-mode ablation (harness, steady step median)

| Case | `legacy` | `flash` | `fast` |
|---|---:|---:|---:|
| 1024² g1 | 0.968 s | 0.569 s | 0.373 s |
| 1344x768 g4 + negative (both branches) | 1.902 s | 1.137 s | 0.753 s |
| 2048² g1 | 13.13 s | 2.98 s | 2.07 s |
| 2752x1536 g1 | 13.40 s | 3.04 s | 2.15 s |

Flash alone is the 2K fix (4.4x per step); the fused elementwise work is
another 1.5x on top at every size. The denoise peak increment above the
resident transformer is 0.81 GB at 1024², 0.97 GB for the guided 1344x768
case and 3.1–3.2 GB at 2K (legacy: 0.91 / 0.94 / 3.7 GB).

### Determinism and byte identity

- `fast` is deterministic: two harness runs at 1024² (seed 210001) and at
  1344x768 g4 give the same decoded RGB (`c7ea2be9…bb13c2`,
  `f799c54c…d3b1a0`).
- `legacy` on the merged branch still reproduces the v0.32 RGB hash
  `a6d45454…525df6`.
- The branch binary under `MOLD_ATTN=math MOLD_CONV=im2col` reproduced all
  three v0.32 reference PNGs **byte for byte** (`f1fa6bc2…`, `0b0a1d9b…`,
  `518bd8e7…`), not only their pixels.
- The fast images were checked visually: the "MOLD & FLOUR" lettering is
  exact and the fox render keeps its composition and fur detail. Pixels move
  relative to v0.32, as FastStill accepts.

#### Re-verified after merging reference conditioning, transparency and LoRA

Branch `5a0a7675` (the M4+M6 merge), `mold run --local` on GPU 2, cold
process, `nvidia-smi` sampled every 100 ms:

| Render | Denoise | Peak (MiB, process) | Result |
|---|---:|---:|---|
| `MOLD_ATTN=math MOLD_CONV=im2col`, 1024² s210001 | 38.2 s | 41,975 | PNG `f1fa6bc2…` — identical to v0.32 |
| same, s210002 | 38.4 s | 41,975 | PNG `0b0a1d9b…` — identical |
| same, 1344x768 g4 + negative, s7 | 74.6 s | 41,751 | PNG `518bd8e7…` — identical |
| fast, 1024² s210001 | **15.2 s** | 34,683 | gate ≤ 22 s met |
| fast, 1 reference (fox), 1024² | 17.3 s | 35,323 | fox from the reference in front of a teal storefront |
| fast, 3 references, g4 + negative | 167.4 s | 35,803 | fox, bakery and bicycle composed; lettering exact |
| fast, `--transparent` teapot | 15.4 s | 34,523 | RGBA; background alpha 0, subject 255 |

Every image was viewed. The three-reference guided render recomputes its
16.8k-token prefix every step in both branches (the request-only cache rule
retains nothing past the legacy bound), which is why it costs 4.2 s/step.

#### Prefix KV cache follows the card on the fast path

Upstream always caches the prefix (`pipeline_qwenimage21.py:528`,
`:750-764`). The request-only 6 GiB rule made the guided three-reference
render above recompute its ~12k-token prefix in both branches every step. On
the CUDA fast path `auto` now retains every branch whenever the summed cache
fits what the card has left beside the weights, the denoise workspace and a
1 GiB margin; the legacy path, Metal and CPU keep the request-only rule.
Branch `d249fa19`, binary `mold-c-kv2`, `mold run --local` on GPU 2, cold
process, 1024² output, each reference 1536x1024:

| Render | Cache decision | Denoise | Peak (MiB) | Result |
|---|---|---:|---:|---|
| 3 references, g4 + negative | retained, 12.0 GiB (TE parked) | **44.2 s** (was 167.4 s) | 35,801 | fox, bakery and bicycle composed; lettering exact |
| 10 references, g1 | recomputed: 19.9 GiB > 16.3 GiB left | 343.7 s | 38,201 | fox before the storefront, lettering exact |
| 10 references, g4 + negative | recomputed: 39.8 GiB > 16.2 GiB left | 696.5 s | 38,169 | fox before the storefront, lettering exact |
| 10 references, g1, `MOLD_QWEN_IMAGE21_KV_CACHE=on` | forced | — | — | refused by the planner: ~61.3 GB needed, ~47.2 GB usable |
| `MOLD_ATTN=math MOLD_CONV=im2col`, 1024² s210001 | request-only (legacy) | 38.2 s | — | PNG `f1fa6bc2…` — identical to v0.32 |
| fast, 1024² s210001 | retained (text prefix) | 14.9 s | — | gate ≤ 22 s met |

Every image was viewed. The first attempt at this rule sampled the driver's
free memory and recomputed the three-reference render with ~20 GiB idle:
after the text encoder parks, candle's stream-ordered pool keeps the freed
pages reserved, and the driver reports them as used. The engine now
synchronizes and counts the pool's reserved-but-unused bytes
(`device::usable_allocatable_vram_bytes`).

### Batched CFG (A8): not adopted

`official_cuda_cfg_batch_probe` measures the upper bound of batching both
CFG branches: one batch-2 cached forward of two equal-length rows (no key
padding, no varlen packing) against two batch-1 forwards.

| Canvas | Two batch-1 | One batch-2 | Gain |
|---|---:|---:|---:|
| 1024² | 0.735 s | 0.780 s | −6.1% |
| 1344x768 | 0.741 s | 0.787 s | −6.1% |
| 2048² | 4.121 s | 4.257 s | −3.3% |

Each batch-1 forward already fills the L40S (M = 4,096 rows per GEMM), so a
batch only adds work. The design's gate was ≥ 5% at a token count; the best
case is negative everywhere, so the engine keeps sequential CFG and no
`MOLD_QWEN_IMAGE21_CFG_BATCH` switch exists.

### Native 2K presets (M8)

All seven upstream presets, 40 steps, g1, seed 210001, `fast` mode, cuDNN VAE.
"Peak inc." is the denoise peak above the resident transformer; "VAE peak" is
the decode's increment after the transformer drop.

| Tier | Canvas | Steady step | Denoise | Peak inc. | VAE decode | VAE peak |
|---|---|---:|---:|---:|---:|---:|
| bf16 | 2048x2048 | 2.071 s | 82.6 s | 3.12 GB | 3.92 s | 27.2 GB |
| bf16 | 2400x1792 | 2.176 s | 87.1 s | 3.19 GB | 3.92 s | 28.1 GB |
| bf16 | 1792x2400 | 2.181 s | 87.4 s | 3.19 GB | 3.87 s | 28.1 GB |
| bf16 | 2528x1696 | 2.173 s | 86.8 s | 3.19 GB | 3.86 s | 28.0 GB |
| bf16 | 1696x2528 | 2.177 s | 87.2 s | 3.19 GB | 3.87 s | 28.0 GB |
| bf16 | 2752x1536 | 2.146 s | 85.9 s | 3.15 GB | 3.83 s | 27.6 GB |
| bf16 | 1536x2752 | 2.122 s | 85.5 s | 3.15 GB | 3.83 s | 27.6 GB |
| int8-conv | 1024x1024 | 0.371 s | 15.3 s | 1.24 GB | 1.95 s | 6.5 GB |
| int8-conv | 2048x2048 | 1.996 s | 81.2 s | 5.30 GB | 4.27 s | 22.5 GB |
| int8-conv | 2752x1536 | 2.037 s | 82.4 s | 5.34 GB | 3.74 s | 22.7 GB |

- Every render was finite and every PNG was checked visually: the lettering
  is exact in all ten and each composition fits its aspect.
- The resident transformer is 14.84 GB (bf16) and 7.82 GB (int8-conv).
- The 2752x1536 preset is 16,512 tokens, past the scheduler's 8,192
  `max_seq`; the dynamic shift extrapolates unclamped as diffusers does.
- `device::qwen_image21_activation_bytes` (token-based, backend-aware) is
  fitted to the `flash`-mode peaks, which the `fast` path stays under.
- The int8-conv tier's W8A8 linears peak 0.44 GB (1024²) to 2.18 GB (2K)
  above bf16: they quantize every activation to I8 and accumulate in I32.
  `device::qwen_image21_linear_workspace_bytes` charges that (95 KB per joint
  token) wherever the checkpoint header is int8-conv — the engine's
  text-encoder residency decision and both server estimates — and a unit test
  holds the sum to each int8 measurement within 15%.

## VAE decode

The VAE now runs under `ConvScope::for_family("qwen-image21")`. With cuDNN
compiled, that is the FastStill cuDNN backend. The decode logs its cuDNN
dispatch delta, which is 39 dispatches per decode under cuDNN.

The decoder's 3x3 convolutions are `BandedConv2d`. When a canvas is larger
than v0.32's 1.8 MP limit, a CUDA im2col decode splits each column buffer
above 2 GiB into row bands with a one-row halo.

On CPU the banded result is bit-identical. On CUDA, cuBLAS picks its GEMM
algorithm from the problem size. With a smaller M per band:

- two bands at the 288→144 shape were bitwise identical;
- three bands were not, though they stayed within one BF16 ulp.

For that reason `BandScope` keeps every v0.32-renderable canvas unbanded.

The table gives decode seconds and peak increment in GB. Standalone runs use
random latents, with a cold and a warm run. In-render decodes run after the
transformer is dropped, with a 2-step schedule.

| Canvas | cuDNN warm | cuDNN in render | im2col warm | im2col in render |
|---|---:|---:|---:|---:|
| 1024² | 0.58 s / 6.44 | 1.94 s / 5.84 | 1.14 s / 13.52 | 2.41 s / 12.31 |
| 1344x768 | 0.57 s / 6.34 | 1.95 s / 5.74 | 1.09 s / 13.32 | 2.39 s / 12.15 |
| 2048² | 2.68 s / 22.98 | 3.98 s / 27.25 | 4.76 s / 21.47 (banded) | 5.39 s / 21.47 (banded) |
| 2752x1536 | 2.37 s / 22.62 | 3.91 s / 27.55 | 4.09 s / 22.62 (banded) | 5.34 s / 22.62 (banded) |

- The "in render" cuDNN timings include cuDNN's first-call algorithm search.
- At 2K the peak comes from activations, not the column buffer. The
  full-resolution 288- and 144-channel tensors are widened to F32 by each
  `RmsNorm2d`. A decode inside a render peaks higher than a standalone one
  because the stream-ordered pool is fragmented after the transformer drop.
- A 2752x1536 decode fits on the 46 GB card under both backends once the
  transformer has been dropped.
- **The eager engine does not fit at 2K.** In the eager engine the
  transformer (14.8 GB) and the BF16 text encoder (16.4 GB) stay resident
  through the decode. Adding a 27.6 GB decode gives about 59 GB. At 2K the
  text-encoder residency decision (`text_encoder_residency::decide`) has to
  park the encoder, or the transformer has to be released before the decode.
  This contradicted the design's "48 GB stays Resident at 2K" row, and the
  shipped decision follows the measurement: at 2K the encoder parks in host
  RAM for the denoise (the 2752x1536 CLI render above logs the park), and the
  transformer is released for the decode when the card still cannot hold it.

`device::qwen_image21_vae_decode_peak_bytes(width, height, conv_backend,
vae_dtype_bytes)` is fitted to the worst of both columns:

- base 0.3 GB plus 6,600 B per pixel;
- plus 6,700 B per pixel for unbanded im2col inside the v0.32 limit;
- scaled for F32 VAEs.

A test holds it to each measurement with at most 15% slack.

## adaLN precision (design step A6)

The fork's `candle_nn::ops::layer_norm` CUDA kernel (`candle-kernels`
`reduce.cu`, `layernorm`) makes one F32 pass, accumulating `sum(x)` and
`sum(x²)`, and computes `var = E[x²] − mean²`. That formula cancels
catastrophically when a row's |mean| is large compared with its standard
deviation.

`official_cuda_adaln_precision_study` ran the legacy forward at 1024² and
three timesteps (1.0, 0.5, 0.05). At every block's norm1 and norm2 input,
with text and target rows measured separately, it compared two paths against
an f64 two-pass reference:

- the fused `layer_norm(x, alpha = 1 + scale, beta = 0)`;
- the hand-written F32 LayerNorm followed by the BF16 `× (1 + scale)`.

| Result | Value |
|---|---:|
| Worst fused max relative error | 3.6e-3 (block 20, norm2, target, t = 0.05) |
| Worst hand-path max relative error | 6.5e-3 |
| Worst row \|mean\| / std | 0.11 |

The residual stream is close to zero-mean in every row, so the one-pass
variance does not cancel. The fused kernel is more accurate than the legacy
path, because it rounds once where the legacy path rounds twice. It passes
the 1e-2 gate, so no custom `adaln_bf16` kernel is needed.

## FlashAttention API for segment dispatch

- `candle_flash_attn::flash_attn_windowed(q, k, v, scale, None, Some(0))`,
  with `q_len < kv_len`, is bottom-right causal. Query row `i` sees keys
  `0..=i + kv_len − q_len`, following `kernels/mask.h`
  (`row + max_seqlen_k − max_seqlen_q`). Called with `q[s..e]` against
  `k[0..e]`, a text segment `[s, e)` therefore attends every key up to its own
  absolute position.
  - The test `attention::tests::flash_windowed_causal_is_bottom_right_and_varlen_matches_dense`
    compares a 64-query segment of a 101-key prefix against a masked math
    reference (max abs error < 2e-2 in BF16).
  - The same test shows that top-left alignment would be wrong.
- `flash_attn_varlen(q, k, v, cu_q, cu_k, max_q, max_k, scale, causal)` takes
  packed `[total, H, D]` tensors with U32 cumulative lengths. Each packed
  sample is bit-identical to its dense `flash_attn` call, so padded CFG
  batches can be packed.
- `flash_attn_varlen_windowed` also exists and gives causal varlen.
  `mm_prefix_ranges` is reachable only through the paged variant, as the
  design says.

## cuBLAS reduced precision: no effect

`official_cuda_reduced_precision_gemm_benchmark` switches the fork's BF16
switch, which changes `CUBLAS_COMPUTE_32F` to `CUBLAS_COMPUTE_32F_FAST_16BF`,
on the transformer's GEMM shapes:

| M x K x N | Default | Reduced | Output difference |
|---|---:|---:|---:|
| 4177 x 4096 x 4096 | 0.822 ms | 0.835 ms | 0 |
| 4177 x 4096 x 12288 | 3.074 ms | 3.166 ms | 0 |
| 4177 x 12288 x 4096 | 2.904 ms | 2.974 ms | 0 |

The outputs are identical and the reduced setting is not faster. It is also a
process-global switch, so mold does not use it.

## Reproduction

```sh
export CUDA_VISIBLE_DEVICES=2 QWEN_IMAGE21_MODEL_ROOT=/storage/mold/models \
       QWEN_IMAGE21_BENCH_OUTPUT=/tmp/q21
cargo test --profile dev-fast -p mold-ai-inference --features cuda,cudnn,flash-attn --lib \
  -- --ignored --exact --nocapture \
  qwen_image21::transformer::performance_tests::cuda::official_cuda_mode_benchmark
```

The environment contract is at the top of
`crates/mold-inference/src/qwen_image21/transformer/performance_tests/cuda.rs`:

- `QWEN_IMAGE21_BENCH_{MODE,TIER,WIDTH,HEIGHT,GUIDANCE,NEGATIVE,STEPS,LIMIT,PROMPT,SEED,CONV,REFERENCE,LABEL}`;
- plus `QWEN_IMAGE21_BENCH_SIZES` and `_CONVS` for
  `official_cuda_vae_decode_benchmark`.

Modes are `legacy`, `flash`, `ops` and `fast`. `official_cuda_cfg_batch_probe`
is the A8 measurement and `official_cuda_vae_decode_benchmark` the decode
calibration.

`scripts/bench-qwen21.sh --host http://127.0.0.1:7681 --gates` runs the
server end-to-end matrix and applies the M5 gates to the median denoise time.
`--dry-run` prints the plan.
