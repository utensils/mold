# Qwen-Image MMQ NaN defect

**Status:** open, root cause unknown. Tracked by [#1048](https://github.com/utensils/mold/issues/1048); the
workaround shipped in [#1045](https://github.com/utensils/mold/issues/1045).

**One-line summary:** candle's fast MMQ CUDA kernels return 100 % NaN for
Qwen-Image's quantized linear shapes at the very first forward, while the same
kernels run Wan's shapes correctly and the dequantize-per-forward arm renders
Qwen correctly on the same checkpoint and the same GPU.

This document exists so the kernel investigation starts from a reproduction and
a list of what has already been ruled out, rather than from the changelog.

---

## What ships today

`MOLD_QWEN_QMATMUL` is **off by default**. The quantized Qwen-Image CUDA linear
arm dequantizes each GGUF weight to BF16 per forward
(`crates/mold-inference/src/qwen_image/quantized_transformer.rs`, `QwenLinear::Dequant`),
which is correct and slow. Setting `MOLD_QWEN_QMATMUL=1` opts into `QMatMul`,
which is where candle's MMQ/MMVQ kernels are reached — and where this defect is.

FLUX, Wan, and SD3 use `QMatMul` on CUDA and render correctly. Z-Image used it
too until 2026-08-15, when its CUDA renders were found solid black from the
same defect (see "Z-Image is a second victim" below); it now defaults to a
per-forward dequant arm exactly like Qwen's, with `MOLD_ZIMAGE_QMATMUL=1` as
the opt-in.

---

## Reproduction

Requires a CUDA build and an installed `qwen-image-2512:q4`.

```bash
MOLD_QWEN_QMATMUL=1 mold run --local qwen-image-2512:q4 \
  "<any prompt>" --width 1024 --height 1024 --steps 1
```

**Observed:** the render aborts at step 0 with the boundary validator's error —

```
Qwen diagnostic boundary 'noise_pred[0]' contains non-finite values:
[qwen-debug] noise_pred[0]: min=NaN max=NaN mean=NaN NaN=<N>/<N> (100.0%) ...
```

`--steps 1` is enough: the defect is present in the first forward, so there is
no need to pay for a full schedule. `MOLD_QWEN_DEBUG=1` prints the full
per-boundary stats on the way there.

**Control:** the same command with `MOLD_QWEN_QMATMUL` unset (the dequant arm)
renders correctly.

Measured 2026-08-14 on an RTX 4090, `qwen-image-2512:q4`, 1024², CUDA 12.8.

### Contrast case — the same kernels are not simply broken

Wan runs quantized CUDA `QMatMul` through the identical
`candle-core/src/quantized/cuda.rs` dispatch and renders correctly. Whatever is
wrong is a function of the affected models' shapes, dtype mix, or activation
statistics — not of the kernels being universally wrong. Any hypothesis that
would also break Wan is, on that evidence alone, the wrong hypothesis.

### Z-Image is a second victim (2026-08-15)

Every quantized Z-Image render on CUDA (`z-image-turbo:q4`/`q6`/`q8`) produced
solid-black images. Instrumented on an RTX 4090 at 512², seed-independent:

- The failure is **timestep-value-dependent, not shape-dependent**: the first
  denoise steps are finite, and the forward whose timestep first exceeds
  t≈0.07 (threshold bracketed between 0.062 and 0.086 across 4/9/20-step
  schedules) goes non-finite. All shapes are identical across steps.
- The eruption site is the **unified-stream feed-forward** (`w2` over the
  SwiGLU product, `k = 10240`, 1035 rows): its output reaches `inf` from
  finite ±40 inputs — impossible for a correct matmul at these magnitudes, so
  the kernel is producing garbage, not merely rounding. Z-Image's activations
  are F32 here (its quantized runtime computes in F32), so this is not a BF16
  artifact.
- Forcing `FORCE_DMMV` (skipping both fast arms) renders the identical request
  correctly — the same control result as Qwen's.

This falsifies the earlier framing that only Qwen's shapes trigger the defect,
and adds a useful datum: Z-Image's per-block intermediates are legitimately
huge (sandwich-norm design; feed-forward outputs at ±1e5 that the post-norm
rescales), so activation dynamic range is now the strongest of the
distinguishing features listed in step 5 below. The shipped mitigation mirrors
Qwen's: the Z-Image quantized linears default to a per-forward dequant arm on
CUDA (`crates/mold-inference/src/zimage/quantized_transformer.rs`,
`select_linear_kind`), with `MOLD_ZIMAGE_QMATMUL=1` re-enabling the fast path
for this investigation. Metal keeps `QMatMul`.

---

## Where the code is

Fork branch: **`fix/mold-compat-0.11`** of <https://github.com/utensils/candle>
(what `[patch]` in the repo-root `Cargo.toml` pins).

| Thing | Path |
| --- | --- |
| Kernels | `candle-kernels/src/mmq_gguf/` |
| Activation quantize to Q8_1 | `candle-kernels/src/mmq_gguf/mmq_quantize.cu` |
| Per-dtype MMQ instances | `candle-kernels/src/mmq_gguf/mmq_instance_q*.cu` |
| MMQ host entry | `candle-core/src/quantized/fast_mmq.rs` (`try_fwd`) |
| MMVQ host entry | `candle-core/src/quantized/fast_mmvq.rs` (`try_fwd`) |
| Dispatch order | `candle-core/src/quantized/cuda.rs` (`QCudaStorage::fwd`) |

Dispatch is MMVQ first, then MMQ, then the legacy dequantize fallback. MMVQ
caps at `MMVQ_MAX_BATCH = 8` rows (`fast_mmvq.rs`), so a Qwen image-stream
linear — thousands of tokens — always lands on **MMQ**, while the tiny
projections (timestep embedding, modulation) can still take MMVQ. Both arms are
skipped entirely while `FORCE_DMMV` is set.

Gates a weight must pass before either arm is reached (`fast_mmq::supports` /
`qk_for`): the GGML dtype must be one of `Q4_0/Q4_1/Q5_0/Q5_1/Q8_0/Q2K/Q3K/Q4K/Q5K/Q6K`,
and the row width must be a multiple of that dtype's block size (32 for the
legacy quants, 256 for the k-quants). mold restates those gates on its own side
in `select_linear_kind`, because failing them lands in `dequantize_matmul`,
which reads the activation as `f32` and errors on BF16.

---

## Hypothesis tested and found insufficient: the zero-`amax` guard

`mmq_quantize.cu` computes the Q8_1 activation scale as:

```c
const float d_inv = 127.0f / amax;   // mmq_quantize.cu
```

`amax` is the warp-reduced max absolute value over the four floats a thread
loaded. When an entire quantization group is exactly zero, `amax == 0`,
`d_inv == +inf`, `roundf(0 * inf)` is `NaN`, and the stored scale `1.0f / d_inv`
is `0`. Upstream ggml guards this case; this kernel does not. A Qwen forward
plausibly produces all-zero groups — padded text positions and post-modulation
zeros are the obvious candidates.

**It was tested and it is not the fix.** Adding the guard alone does not make
the render finite. Treat the missing guard as a real latent bug worth fixing on
its own merits, and as an *incomplete* explanation of this defect — do not stop
the investigation there.

---

## Next steps

1. **Build a minimal candle-side reproducer.** Quantize a known tensor to Q4_K,
   matmul it against a BF16 right-hand side shaped like Qwen's image-stream
   linears (thousands of rows, `k` of 3072 or 12288), and compare `QMatMul`
   against `dequantize` + dense matmul. A failing test inside `candle-core` is
   worth more than any amount of end-to-end bisection, and it is the artefact
   an upstream report needs.
2. **Bisect the dtype mix.** A Qwen `q4` GGUF is not uniformly Q4_K — the
   published quantizations mix Q4_K with Q6_K for selected tensors. Enumerate
   the dtypes present in `qwen-image-2512:q4` and in the Wan GGUF that works,
   and check whether the difference is a dtype Wan never exercises. Forcing the
   whole checkpoint to a single dtype and re-running the reproduction isolates
   this in one step.
3. **Separate the MMQ and MMVQ arms.** Qwen reaches both — MMQ for the token
   streams, MMVQ for the small projections. Disable each in turn in
   `QCudaStorage::fwd` and re-run. If only one arm NaNs, the search space
   collapses to that arm's kernels.
4. **Bisect by shape, not by model.** Once (1) exists, sweep `k`, row count,
   and batch across the reproducer to find the boundary. `k = 12288` (the
   feed-forward width) and the batched-CFG row doubling are the first things to
   vary.
5. **Check the activation, not only the weight.** MMQ quantizes the
   *activation* to Q8_1 per forward; the weight is already quantized on disk and
   is shared with the dequant arm that works. That asymmetry points at
   `mmq_quantize.cu` and at whatever Qwen activations look like that Wan's do
   not — dynamic range, exact zeros, and post-RMSNorm scale are the
   distinguishing features to measure.

---

## When this is fixed

Flip the default in `select_linear_kind`, drop the `MOLD_QWEN_QMATMUL` opt-in
(or keep it as an escape hatch), and update the env-var tables in
`website/guide/configuration.md` and `crates/mold-cli/src/skill/SKILL.md` plus the
`CLAUDE.md` invariant on the quantized linear arm. Re-run the reproduction above
and the `bench-qwen.sh` matrix; the whole point of the flip is throughput, so a
number belongs in the changelog entry.
