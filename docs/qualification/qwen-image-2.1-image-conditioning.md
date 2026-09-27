# Qwen Image 2.1 image conditioning parity

Reference-image conditioning (ordered `edit_images`) against the upstream
diffusers pipeline at `e0abab83b` (`pipeline_qwenimage21.py`,
`transformer_qwenimage21.py`, `autoencoder_kl_qwenimage21.py`) on the pinned
checkpoint `Qwen/Qwen-Image-2.1@b3179ad` (weights byte-identical at
`790c926`). The captures and their SHA-256s are listed in
`crates/mold-inference/testdata/qwen_image21/README.md`; small ones are
committed there, large ones live under `/storage/mold/fixtures/qwen_image21/captures/`.

Every number below was measured on 2026-09-27 on plato (NVIDIA L40S 46 GB,
CUDA 12.9), `--profile dev-fast --features cuda,cudnn,flash-attn`, i.e. the
CUDA fast path (`Qwen21ExecPath::cuda_fast`). Relative errors are
`max|a-b| / max|b|` ("max") and `mean|a-b| / mean|b|` ("mean"). "Upstream
bf16" is upstream's own bf16 run scored against upstream's fp32 run.

## Reproduction

```sh
QWEN_IMAGE21_MODEL_ROOT=/storage/mold/models \
QWEN_IMAGE21_FIXTURES=/storage/mold/fixtures/qwen_image21/captures \
CUDA_VISIBLE_DEVICES=3 \
  cargo test --profile dev-fast -p mold-ai-inference --features cuda,cudnn,flash-attn --lib \
  qwen_image21::parity_tests qwen_image21::vae_encoder qwen_image21::conditioning \
  -- --ignored --test-threads=1 --nocapture
```

`QWEN_IMAGE21_MODEL_ROOT` must hold `shared/qwen-image21/{vae,text_encoder,processor}`
and `qwen-image-2.1-bf16/transformer`; `QWEN_IMAGE21_FIXTURES` is the capture
directory (the Viggle adapters are read from its sibling `../viggle`). Every
gated test is `#[ignore]` and PANICS naming the missing variable when run
without it, so a parity run can never pass unrun. The CPU-only P5 layout test
runs in the default `cargo test`.

## Results

| Gate       | What                                                                   | Measured                                                                                        | Gate                                                     |
| ---------- | ---------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------- | -------------------------------------------------------- |
| P1         | Processor `input_ids`, `image_grid_thw`, `pixel_values` (2 references) | exact, 0 differing pixel values                                                                 | exact                                                    |
| P2         | Vision tower, fp32: merger                                             | max 9.99e-5, mean 2.21e-5                                                                       | max 3e-4                                                 |
| P2         | DeepStack 0 / 1 / 2                                                    | max 3.11e-6 / 2.38e-5 / 3.57e-5                                                                 | max 1e-3                                                 |
| P3 fp32    | `prompt_embeds` p6_pos / p6_neg / p8_pos / t2i_p8                      | max 1.07e-3 / 1.07e-3 / 7.78e-4 / 8.88e-6 (mean 1.74e-4 / 1.76e-4 / 1.12e-4 / 1.86e-6)          | max 5e-3                                                 |
| P3 bf16    | mean vs fp32 truth, p6_pos / p6_neg / p8_pos / t2i_p8                  | mold 5.18e-2 / 5.28e-2 / 4.36e-2 / 2.86e-2; upstream bf16 2.99e-1 / 3.01e-1 / 2.85e-1 / 2.45e-2 | ≤ 6.6e-2 / 6.6e-2 / 5.5e-2 / 3.6e-2 and ≤ 1.25x upstream |
| P3 LM fp32 | per-layer mean, scatter, layers 0-3, 17, 34, pre-norm                  | 1.02e-5, 8.74e-6, 7.49e-6, 6.95e-6, 6.92e-6, 2.07e-5, 5.05e-5, 5.43e-5                          | 1e-4 ×5, 2.5e-4, 6e-4, 6e-4                              |
| P3 LM bf16 | same layers, BF16 LM + F32 tower vs fp32                               | 1.41e-3, 3.42e-3, 4.24e-3, 4.82e-3, 5.55e-3, 2.15e-2, 3.92e-2, 4.34e-2                          | 1.5x measured, and ≤ upstream bf16 per layer             |
| P3 LM bf16 | upstream bf16 vs fp32, same layers                                     | 1.13e-1, 9.57e-2, 8.20e-2, 7.60e-2, 7.55e-2, 1.65e-1, 2.68e-1, 2.80e-1                          | (reference)                                              |
| P4         | VAE encoder stages (64x96 RGBA crop, CPU F32), worst stage             | max 2.47e-5                                                                                     | max 1e-4                                                 |
| P4         | `encode_packed` opaque 1248x832 / rgba 928x1152                        | max 4.93e-6 / 3.24e-5                                                                           | max 1e-4                                                 |
| P5         | Joint layout (image ids, target mask, RoPE ids, segments), every case  | exact                                                                                           | exact                                                    |
| P6 fp32    | one 2-reference forward: full_a / full_b / extract_a / cached_b        | max 1.18e-5 / 1.55e-6 / 1.18e-5 / 2.50e-6                                                       | max 1e-4                                                 |
| P6 bf16    | mean vs fp32 truth                                                     | mold 1.67e-2 / 7.85e-3 / 1.67e-2 / 7.80e-3; upstream 2.28e-2 / 9.24e-3 / 2.28e-2 / 9.05e-3      | ≤ 1.5x upstream                                          |
| P7 fp32    | Viggle r128 bypass forward                                             | max 2.59e-5                                                                                     | max 1e-4                                                 |
| P7 bf16    | mean vs fp32 truth                                                     | mold 2.37e-2; upstream 2.85e-2                                                                  | ≤ 1.5x upstream                                          |
| P8 base4   | engine end to end, PSNR vs upstream fp32 render                        | mold 38.80 dB; upstream bf16 37.59 dB (+1.21)                                                   | ≥ upstream + 0.5 dB                                      |
| P8 turbo6  | engine end to end (Viggle r256, 6-step recipe)                         | mold 34.67 dB; upstream bf16 33.22 dB (+1.45)                                                   | ≥ upstream + 1.0 dB                                      |

P1/P2/P3/P4/P6/P7 fp32 gates are absolute. The bf16 rows compare both
implementations to the fp32 truth, because a bf16-to-bf16 comparison measures
rounding chaos: two correct bf16 pipelines disagree by about as much as either
disagrees with fp32.

P6's bf16 timesteps are `0.8984375` / `0.6015625` (upstream casts `t` to the
latent dtype before dividing by 1000); fp32 uses `0.9` / `0.6`.

## The language-model internals

`p3_p8_language_model_internals_match_the_fp32_capture` injects P2's captured
fp32 merger and DeepStack rows, so only the language model is measured, and
compares its hidden states at the scatter, after layers 0-3, 17 and 34, and the
pre-norm output. Measured errors are 7e-6 to 5.4e-5; the ceilings (10x, floored
at 1e-4) are three orders of magnitude below the percent-level error a wrong
RoPE section, epsilon or DeepStack layer produces.

The bf16 test gates the SHIPPED configuration (BF16 language model, F32 tower)
against the fp32 truth at 1.5x the measurement, and also requires every layer
to be no further from fp32 than upstream's own bf16 run. The bf16 capture is
not a usable target: upstream's bf16 tower moves the merger 14.1% (mean), and
mold's bf16 LM scored against that capture reads 7.6-31% whichever tower it
uses (`p3_p8_language_model_internals_study` prints both), so a several-percent
defect would be invisible against it.

Making the per-layer comparison assertable surfaced a diagnostic bug:
`Qwen3BF16Model::multimodal_hidden_states` recorded each state after the
DeepStack add, while transformers records the decoder layer's output before it
(`modeling_qwen3_vl.py:636-637`, `:844-852`). That read as 27-49% error at
exactly layers 0-2 (the three DeepStack layers) while layer 3 matched to 1e-5.
The production forward, which only returns the final state, was unaffected.

## Why the vision tower and VAE encoder run F32

Upstream runs both in the pipeline dtype. mold runs them in F32 on every device
(`reference::{vision_tower_dtype, vae_encoder_dtype}`), a deliberate divergence
chosen on this evidence. Reproduce it with:

```sh
QWEN_IMAGE21_MODEL_ROOT=/storage/mold/models \
QWEN_IMAGE21_FIXTURES=/storage/mold/fixtures/qwen_image21/captures \
CUDA_VISIBLE_DEVICES=3 \
  cargo test --profile dev-fast -p mold-ai-inference --features cuda,cudnn,flash-attn --lib \
  qwen_image21::parity_tests::conditioning_precision_study -- --ignored --nocapture
```

The study runs each component in BF16 and F32 against the fp32 captures, then
drives the P8 base and turbo trajectories from each variant's own conditioning
and condition latents (the engine's composition, component by component: BF16
language model and transformer, rounded timestep, retained prefix cache, BF16
VAE decode) and applies the P8 gate. It asserts that F32 passes both gates and
that BF16 FAILS the turbo gate.

| Component                                              | BF16 (upstream's dtype)     | F32 (shipped)               |
| ------------------------------------------------------ | --------------------------- | --------------------------- |
| Vision merger (P2), mean / max                         | 1.46e-1 / 3.88e-1           | 2.21e-5 / 9.99e-5           |
| DeepStack 0 / 1 / 2, mean                              | 3.00e-2 / 6.22e-2 / 8.13e-2 | 4.69e-6 / 9.02e-6 / 1.20e-5 |
| P8 prompt_embeds (P3 p8_pos, BF16 LM), mean            | 3.15e-1                     | 4.36e-2                     |
| VAE encoder packed latents opaque / rgba, mean         | 1.33e-2 / 1.58e-2           | 1.77e-6 / 1.62e-6           |
| P8 base4 (component study), margin over upstream bf16  | +4.71 dB (42.31)            | −3.93 dB (33.66)            |
| P8 turbo6 (component study), margin over upstream bf16 | **−6.39 dB (26.83)**        | +2.99 dB (36.20)            |

Turbo P8 delta, F32 − BF16: **+9.37 dB**. The study asserts the turbo rows
only. Its base4 rows are not a stable measurement: the switch to upstream's
float32 rotary angles (a sub-1e-5 change to every table value) moved the F32
base4 row from 39.39 to 33.66 dB and left the BF16 row above it (43.71 →
42.31), while the engine's own base4 render moved 0.11 dB (38.69 → 38.80).
Base4 is gated at engine level only. The 6-step distilled trajectory
amplifies conditioning error; the 4-step base render does not discriminate (its
BF16 variant happens to land closer to upstream fp32 on this one case).

The engine itself was also run with both consts flipped to BF16 (a local,
uncommitted edit), on the final arithmetic (upstream float32 rotary angles,
rounded timestep): P8 base4 38.73 dB (passes), P8 turbo6 **33.48 dB, +0.26 dB
over upstream bf16, failing the +1.0 dB gate by 0.74 dB** — against 34.67 dB
shipped. The component study's BF16 turbo number is lower than the engine's;
both fail.

### The P8 gate

`P8_BASE_MARGIN_DB = 0.5` and `P8_TURBO_MARGIN_DB = 1.0`: mold's BF16 engine
must BEAT upstream's bf16 render against the fp32 truth, not merely come
within a dB of it (the old gate, `ours >= theirs - 1.0`, which the BF16-tower
engine passes). The turbo margin sits between the shipped engine's +1.45 dB
and the BF16-tower engine's +0.26 dB, so flipping either dtype const back to
BF16 fails P8 turbo and the study, and the choice is pinned by a measurement
rather than by a `const fn`.

### Rotary angles move P8 toward upstream bf16, away from fp32

Before the rotary tables matched upstream's float32 `rope_params`
(`transformer_qwenimage21.py:673-675`) bit for bit, mold evaluated the angles
in f64. Measured on the same L40S, same run otherwise:

| Rotary angles      | base4 vs fp32 | base4 vs upstream bf16 | turbo6 vs fp32 | turbo6 vs upstream bf16 |
| ------------------ | ------------- | ---------------------- | -------------- | ----------------------- |
| f64 (v0.32)        | 38.69 dB      | 34.26 dB               | 36.23 dB       | 37.88 dB                |
| float32 (upstream) | 38.80 dB      | 34.75 dB               | 34.67 dB       | 40.53 dB                |

Matching upstream's arithmetic moves the turbo render 2.65 dB closer to
upstream's own bf16 render and 1.56 dB further from the fp32 truth: the f64
angles were accidentally more accurate than the reference being ported, and
the 6-step turbo amplifies either. The port follows upstream; the turbo margin
was re-derived on the float32 angles (it was +1.5 dB against the f64 +3.01).

## Why reference prompts never auto-select a quantized text encoder

Text-to-image auto mode falls back from the BF16 Qwen3-VL-8B shards to the
official Q8_0 GGUF when BF16 does not fit beside the transformer and VAE
(`variant_resolution::choose_qwen3_vl_variant`). For a reference-conditioned
request it does not: auto mode keeps BF16 on the card, or BF16 on the CPU
(Metal: on the unified pool), and the planner (`text_encoder_residency::plan`,
`sequential_peak_bytes`) reads the same rule. An explicit
`MOLD_QWEN3_VARIANT=q8|q4` still wins and logs a warning; an eager engine
that auto-loaded the GGUF before any request reloads the BF16 shards when the
first reference request arrives.

The hidden-state gate
(`encoders::qwen3_vl_gguf_parity::gguf_multimodal_conditioning_tracks_the_bf16_encoder`,
2 references, 2,097 tokens) localizes the Q8_0 loss to the visual rows: text
rows mean 0.99891 / worst 0.99285, visual rows mean 0.99412 / worst 0.40068
(146 under 0.99), against BF16-in-BF16's 0.99635 / 0.45722. The GGUF code path
on unquantized weights matches the F32 oracle exactly, so this is
quantization. Whether it matters was decided end to end:

```sh
MOLD_MODELS_DIR=/storage/mold/models \
QWEN_IMAGE21_MODEL_ROOT=/storage/mold/models \
QWEN_IMAGE21_FIXTURES=/storage/mold/fixtures/qwen_image21/captures \
QWEN_IMAGE21_TE_STUDY_DIR=/tmp/te_study CUDA_VISIBLE_DEVICES=3 \
  cargo test --profile dev-fast -p mold-ai-inference --features cuda,cudnn,flash-attn --lib \
  qwen_image21::parity_tests::reference_text_encoder_variant_study -- --ignored --nocapture
```

Same engine, BF16 transformer, upstream's injected P8 noise, 512², seed 1234;
only the language model differs. All twelve renders were reviewed and each
pair is visually equivalent.

| Recipe | Case             | Q8-TE vs BF16-TE | BF16-TE vs fp32 | Q8-TE vs fp32 | upstream bf16 vs fp32 |
| ------ | ---------------- | ---------------- | --------------- | ------------- | --------------------- |
| base4  | text-to-image    | 39.08 dB         | —               | —             | —                     |
| base4  | 1 reference (P8) | 40.99 dB         | 38.80 dB        | 42.20 dB      | 37.59 dB              |
| base4  | 3 references     | 37.11 dB         | —               | —             | —                     |
| turbo6 | text-to-image    | 33.47 dB         | —               | —             | —                     |
| turbo6 | 1 reference (P8) | 34.58 dB         | 34.67 dB        | **31.90 dB**  | 33.22 dB              |
| turbo6 | 3 references     | 31.98 dB         | —               | —             | —                     |

The Q8-vs-BF16 distance on reference prompts is the same order as on
text-to-image, where Q8_0 stays auto-selectable. The discriminating number is
the one the P8 gate reads: on the 6-step turbo trajectory, which amplifies
conditioning error (see the F32 vision tower above), the Q8_0 encoder lands
1.32 dB BELOW upstream's own bf16 pipeline and 2.32 dB short of the P8 turbo
gate the shipped path holds — a larger loss than the BF16 vision tower this
record rejects at 33.48 dB. The 4-step base render does not discriminate
(Q8 happens to land closer to fp32 there). The study asserts the turbo
verdict: BF16 passes the P8 turbo gate, Q8_0 does not. The hidden-state test
now gates the policy rather than row equality: the GGUF path is exact on
unquantized weights, Q8_0's multimodal text rows hold the text gate, and no
card size makes auto mode pick a GGUF tier for a reference request.

## Cost

F32 costs about 1 GiB more resident vision tower (it runs once per request and
is dropped before the transformer denoises) and 0.3 GB for the VAE encoder,
which runs once per reference. Admission charges both
(`device::qwen_image21_encode_phase_bytes`, `device.rs` notes on
`qwen_image21::reference::{vision_tower_dtype, vae_encoder_dtype}`).

## Moving numbers

P6 and P8 sit on the CUDA fast path's arithmetic (FlashAttention segments,
F32 RoPE tables, rounded timestep). A change to that arithmetic — the RoPE
angle precision or the per-request timestep rounding — moves them (the switch
to upstream's float32 angles moved turbo6 by 1.56 dB, above); rerun the
reproduction and update this record whenever it changes. Every number in this
record was measured on the final arithmetic.
