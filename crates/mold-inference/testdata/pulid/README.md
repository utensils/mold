# PuLID encoder parity goldens

Fixtures for `crates/mold-inference`'s EVA02-CLIP-L-14-336 vision tower
(`src/encoders/eva_clip_vision.rs`), its preprocessing
(`src/encoders/eva_clip_preprocess.rs`), and PuLID's IDFormer
(`src/flux/pulid_encoder.rs`). Issue
[#1229](https://github.com/utensils/mold/issues/1229).

> This directory holds two independent golden sets. The **face extraction**
> fixtures — detection, alignment, and the ArcFace embedding
> ([#1222](https://github.com/utensils/mold/issues/1222)) — live in `faces/`
> with their own `faces/README.md` and their own `capture_goldens.py`. This
> file covers only the encoder set below, whose capture script is
> `capture_eva_goldens.py`.

## Provenance

| | |
| --- | --- |
| Upstream | <https://github.com/ToTheBeginning/PuLID> |
| Commit | `1aa2fc7df4bf51080df39f355f9abdc1cbfefbaa` |
| Capture script | `capture_eva_goldens.py` (in this directory) |
| Captured | 2026-08-21, aarch64-darwin, CPU, float32 |
| torch | 2.x CPU wheel in a scratch venv |

`capture_eva_goldens.py` is committed **as documentation of provenance only**.
Nothing in mold's build, test, or runtime path executes it, and mold ships no
Python. Re-run it by hand to refresh these files:

```bash
python3 -m venv /tmp/pulid-venv
/tmp/pulid-venv/bin/pip install torch torchvision numpy pillow einops timm ftfy regex safetensors
git clone https://github.com/ToTheBeginning/PuLID /tmp/PuLID
/tmp/pulid-venv/bin/python capture_eva_goldens.py \
  --pulid-repo /tmp/PuLID \
  --eva   /path/to/EVA02_CLIP_L_336_psz14_s6B.pt \
  --adapter /path/to/pulid_flux_v0.9.1.safetensors \
  --out .
```

## Checkpoints

Both are pinned in `crates/mold-core/src/manifest.rs` (`pulid_manifests`) and
verified before use.

| File | SHA-256 | License |
| --- | --- | --- |
| `EVA02_CLIP_L_336_psz14_s6B.pt` (`QuanSun/EVA-CLIP`, 856 MB) | `84c3a17a228c567a155259b2245b0b59072bf7da510260a0a02ec54de6d50b05` | MIT |
| `pulid_flux_v0.9.1.safetensors` (`guozinan/PuLID`, 1.14 GB) | `92c41c3af322b02e58e1b32842e4601e08c8f16ec1fe80089dbe957df510f51d` | Apache-2.0 |

**Derived** vision-only safetensors, produced by
`encoders::eva_clip_convert::convert_eva_clip_vision` from the `.pt` above:

| File | SHA-256 |
| --- | --- |
| `eva02_clip_l_336_vision.safetensors` (514 tensors, f16, ~609 MB) | `2b0b0ab0baed6ee968c8a08a9dcba908fb602630303faa3515eeaf8e264f136b` |

That digest is compiled in as `eva_clip_convert::DERIVED_SHA256` and asserted
by `conversion_is_deterministic_on_the_pinned_source`, so a re-uploaded source,
a changed retention rule, or a `safetensors` layout change fails loudly rather
than silently producing different weights. It is also what authenticates a
derived file that is being reused — the sidecar beside it is provenance for a
human and is never trusted, because mold writes it and anything able to tamper
with the weights could forge it to match.

## Files

| File | What it is |
| --- | --- |
| `goldens.safetensors` | Every numeric fixture (290 KB) |
| `goldens.json` | Human-readable statistics and capture parameters |
| `input_pattern.png` | The 512x512 preprocessing input (force-added past the repo-wide `*.png` ignore) |
| `capture_eva_goldens.py` | Provenance, above |
| `true_cfg_goldens.safetensors` | The unconditional identity embedding (256 KB) |
| `true_cfg_goldens.json` | Its statistics and capture parameters |
| `capture_true_cfg_goldens.py` | Provenance for the two files above |

`faces/`, `fetch_faces.py`, `onnx-inventory.json` and `capture_goldens.py`
belong to the face-extraction goldens and are documented in
`faces/README.md`.

### The true-CFG golden

A separate file and a separate script
([#1226](https://github.com/utensils/mold/issues/1226)) so that regenerating
either set never requires re-running the other's 609 MB tower load. It holds one
tensor and its statistics: the **unconditional** identity embedding PuLID's true
classifier-free guidance conditions its negative branch on, which upstream
builds by running the IDFormer on all-zero conditioning
(`PuLID/pulid/pipeline_flux.py:188-192`).

It is not a zero tensor — the IDFormer's biases, LayerNorms, and learned latent
queries land all-zero inputs around ±13000 — and it depends on no photograph at
all, which is why one committed tensor is the complete answer. Captured on the
same pinned `pulid_flux_v0.9.1.safetensors` as the IDFormer golden above.

```bash
/tmp/pulid-venv/bin/python capture_true_cfg_goldens.py \
  --pulid-repo /tmp/PuLID \
  --adapter /path/to/pulid_flux_v0.9.1.safetensors \
  --out .
```

Checked by `flux::pulid_encoder::tests::the_unconditional_identity_matches_upstream`,
which is `#[ignore]` behind `MOLD_TEST_PULID_ASSETS` like the rest of the
weight-gated set. Measured error: **3.7e-7** of the tensor's own scale.

### Inputs are generated, not committed

Every fixture input except the PNG comes from one deterministic value stream —
`xorshift64*`, four lines, implemented identically in `capture_eva_goldens.py` and
in `crates/mold-inference/src/pulid_fixtures.rs`. A `[1, 577, 1024]` fixture
therefore costs zero bytes in the repository, and
`pulid_fixtures::tests::the_value_stream_is_pinned` guards the stream itself so
a drift there cannot be mistaken for a model bug.

Seeds (the hex spells the name):

| Seed | Drives |
| --- | --- |
| `PULIDTOW` | The tower's `[1, 3, 336, 336]` input |
| `PULIDPRB` + i | Probe indices for hidden state `i` |
| `PULIDIDF` | The IDFormer's `[1, 1280]` identity condition |
| `PULIDVIT` + i | The IDFormer's `i`-th `[1, 577, 1024]` vision hidden state |
| `PULIDIMG` | The input image's high-frequency term (`^ 1` for its probe indices) |

`input_pattern.png` is procedurally generated by the capture script — smooth
trigonometric fields plus deterministic per-pixel noise — so it carries no
third-party image provenance. It is 512x512 because that is PuLID's aligned
face size (`pipeline_flux.py:50`), making the resize under test the same
512 -> 336 downscale production performs.

### Golden arrays

| Name | Shape | Pins |
| --- | --- | --- |
| `preprocess.probe` | `[512]` | 512 scattered values of the preprocessed tensor |
| `preprocess.probe_indices` | `[512]` i64 | The indices above, recorded for audit |
| `preprocess.row_g_168` | `[336]` | A full green-channel row — catches a channel swap or HWC/CHW transpose |
| `rope.freqs_cos.rows` / `.sin.rows` | `[6, 64]` | Rows 0, 1, 23, 24, 300, 575 of the checkpoint's OWN `visual.rope.*` buffer |
| `tower.hidden_{0..4}.probe` | `[512]` | 512 scattered values of each tapped hidden state |
| `tower.hidden_{0..4}.stats` | `[5]` | `[mean, std, min, max, peak]` of the whole tensor |
| `tower.cls_projection` | `[768]` | The raw `visual.head` output |
| `tower.cls_projection_normalized` | `[768]` | After the pipeline's L2 normalization |
| `idformer.output` | `[32, 2048]` | The COMPLETE output tensor (256 KB) |
| `idformer.output.stats` | `[5]` | As above |

Large tensors are pinned by a 512-element probe **plus** whole-tensor
statistics rather than in full: the probe catches a value error, the statistics
catch a defect that misses every probe index. The IDFormer output is small
enough to commit whole, so it is.

## Tensor mapping

### `visual.*` (EVA02-CLIP-L-14-336) -> `EvaClipVisionTower`

The conversion strips the `visual.` prefix, so the names below are what the
`VarBuilder` sees.

| Checkpoint | Shape | Rust |
| --- | --- | --- |
| `cls_token` | `[1, 1, 1024]` | `EvaClipVisionTower::cls_token` |
| `pos_embed` | `[1, 577, 1024]` | `pos_embed` |
| `patch_embed.proj.{weight,bias}` | `[1024, 3, 14, 14]`, `[1024]` | `patch_embed` (`Conv2d`, stride 14) |
| `rope.freqs_{cos,sin}` | `[576, 64]` | Retained for the golden test; mold derives its own `VisionRotaryEmbedding` |
| `blocks.{i}.norm1.{weight,bias}` | `[1024]` | `Block::norm1` (eps 1e-6) |
| `blocks.{i}.attn.q_proj.weight` | `[1024, 1024]` | `Attention::q_proj` weight |
| `blocks.{i}.attn.q_bias` | `[1024]` | `Attention::q_proj` **bias** — out of band, `q_proj` itself is biasless |
| `blocks.{i}.attn.k_proj.weight` | `[1024, 1024]` | `Attention::k_proj` — **no k bias exists** |
| `blocks.{i}.attn.v_proj.weight` / `v_bias` | `[1024, 1024]` / `[1024]` | `Attention::v_proj` weight / bias |
| `blocks.{i}.attn.inner_attn_ln.{weight,bias}` | `[1024]` | `Attention::inner_attn_ln`, between attention and `proj` |
| `blocks.{i}.attn.proj.{weight,bias}` | `[1024, 1024]`, `[1024]` | `Attention::proj` |
| `blocks.{i}.attn.rope.freqs_*` | `[576, 64]` | **Dropped** — 48 byte-identical copies of `rope.freqs_*` |
| `blocks.{i}.norm2.{weight,bias}` | `[1024]` | `Block::norm2` |
| `blocks.{i}.mlp.w1.{weight,bias}` | `[2730, 1024]`, `[2730]` | `SwiGlu::w1` (the SiLU-gated branch) |
| `blocks.{i}.mlp.w2.{weight,bias}` | `[2730, 1024]`, `[2730]` | `SwiGlu::w2` (the value branch) |
| `blocks.{i}.mlp.ffn_ln.{weight,bias}` | `[2730]` | `SwiGlu::ffn_ln` |
| `blocks.{i}.mlp.w3.{weight,bias}` | `[1024, 2730]`, `[1024]` | `SwiGlu::w3` |
| `norm.{weight,bias}` | `[1024]` | `EvaClipVisionTower::norm` |
| `head.{weight,bias}` | `[768, 1024]`, `[768]` | `EvaClipVisionTower::head` |

`text.*` and `logit_scale` are not retained: mold never uses the CLIP text
tower. 562 `visual.*` tensors become 514.

Hidden states are the residual stream captured **entering** blocks 4, 8, 12,
16 and 20, not those blocks' outputs — `eva_vit_model.py:526` appends before
running the block.

### `pulid_encoder.*` -> `IdFormer`

172 tensors, all BF16 in the checkpoint. `Sequential` indices are positions, so
0/3/6 are linears and 1/4 are norms; 2 and 5 are activations with no weights.

| Checkpoint | Shape | Rust |
| --- | --- | --- |
| `latents` | `[1, 32, 1024]` | `IdFormer::latents` |
| `proj_out` | `[1024, 2048]` | `IdFormer::proj_out` — a bare parameter used as `latents @ proj_out`, **not** transposed |
| `id_embedding_mapping.0.{weight,bias}` | `[1024, 1280]`, `[1024]` | `MappingMlp::fc1` |
| `id_embedding_mapping.1.{weight,bias}` | `[1024]` | `MappingMlp::norm1` |
| `id_embedding_mapping.3.{weight,bias}` | `[1024, 1024]`, `[1024]` | `MappingMlp::fc2` |
| `id_embedding_mapping.4.{weight,bias}` | `[1024]` | `MappingMlp::norm2` |
| `id_embedding_mapping.6.{weight,bias}` | `[5120, 1024]`, `[5120]` | `MappingMlp::fc3` — 5 identity tokens x 1024 |
| `mapping_{0..4}.{0,1,3,4,6}.*` | as above, `6` is `[1024, 1024]` | `IdFormer::mappings[i]` |
| `layers.{0..9}.0.norm1.{weight,bias}` | `[1024]` | `PerceiverAttention::norm1` (context) |
| `layers.{0..9}.0.norm2.{weight,bias}` | `[1024]` | `PerceiverAttention::norm2` (latents) |
| `layers.{0..9}.0.to_q.weight` | `[1024, 1024]` | `to_q`, biasless |
| `layers.{0..9}.0.to_kv.weight` | `[2048, 1024]` | `to_kv`, biasless, chunked into k and v |
| `layers.{0..9}.0.to_out.weight` | `[1024, 1024]` | `to_out`, biasless |
| `layers.{0..9}.1.0.{weight,bias}` | `[1024]` | `FeedForward::norm` |
| `layers.{0..9}.1.1.weight` | `[4096, 1024]` | `FeedForward::up`, biasless |
| `layers.{0..9}.1.3.weight` | `[1024, 4096]` | `FeedForward::down`, biasless |

Layers `2i` and `2i+1` belong to vision scale `i` (`depth // 5 == 2`).

## Tolerances

Measured on aarch64-darwin, CPU, f32, `--test-threads=1`. Hidden-state and
IDFormer errors are the largest absolute deviation as a fraction of the golden
tensor's own peak magnitude; the CLS projection is quoted absolutely because it
is a unit vector.

| Fixture | Measured | Asserted | Weights needed |
| --- | --- | --- | --- |
| `preprocess.probe`, `preprocess.row_g_168` | < 1e-4 abs | 1e-4 abs | no |
| `rope.freqs_*.rows` | < 1e-3 abs | 1e-3 abs | no |
| `tower.hidden_0` | 1.8e-5 | 1e-3 | yes |
| `tower.hidden_1` | 1.2e-4 | 1e-3 | yes |
| `tower.hidden_2` | 1.1e-4 | 1e-3 | yes |
| `tower.hidden_3` | 1.9e-4 | 1e-3 | yes |
| `tower.hidden_4` | 1.3e-4 | 1e-3 | yes |
| `tower.*.stats` | — | 1e-3 of peak | yes |
| `tower.cls_projection_normalized` | 1.3e-5 abs | 1e-4 abs | yes |
| `idformer.output` | 1.5e-7 | 1e-4 | yes |
| `idformer.output.stats` | — | 1e-4 of peak | yes |

The RoPE table is compared at 1e-3 because the checkpoint stores those buffers
as **f16**, so the bound is f16 resolution near 1.0 rather than f32 parity.

The preprocessing bound is 1e-4 because mold's antialiased bicubic reproduces
torchvision's to about 1.5e-5 in f32 before the `/ std` division scales it up;
the two differ only in accumulation order.

The tower's hidden states sit around 1e-4 of scale while the IDFormer, sharing
this harness, lands at 1.5e-7. That difference is f32 accumulation, not a port
defect — see the long comment on
`eva_clip_vision::tests::tower_matches_upstream_hidden_states_and_projection`
for the argument (it grows with depth then plateaus; the strictly deeper CLS
projection is a hundred times tighter). An f64 cross-check would settle it
outright but candle's fused LayerNorm has no F64 kernel.

## Running the tests

Hermetic tests (no weights, no network) run in the ordinary suite:

```bash
cargo test -p mold-ai-inference --lib -- pulid eva_clip
```

Weight-gated tests need the two pinned checkpoints. Point
`MOLD_TEST_PULID_ASSETS` at a directory holding them (searched one level deep,
so `hf download --local-dir` layouts work). The conversion stages a transient
856 MB copy of the source pickle under `$XDG_RUNTIME_DIR` or `$TMPDIR`, so that
volume needs roughly 1 GB free; set `TMPDIR` if it does not.

```bash
MOLD_TEST_PULID_ASSETS=/path/to/pulid-assets \
  cargo test --release -p mold-ai-inference --lib -- \
    --ignored --nocapture --test-threads=1 pulid eva_clip
```

`--test-threads=1` matters: candle's CPU gemm splits work by the parallelism it
finds, so running these beside other heavy tests changes the accumulation order
enough to move the last digits.

# PuLID parity goldens

Fixtures captured from upstream `ToTheBeginning/PuLID` so mold's candle ports
can be falsified against the reference implementation rather than against
another reading of it.

The face-extraction goldens (#1222) live in `faces/`, with their own
`README.md` and `capture_goldens.py`; this file covers the adapter.

## `ca_goldens.safetensors` — `PerceiverAttentionCA` (#1221)

The cross-attention module the FLUX adapter is twenty copies of
(`pulid/encoders_transformer.py:29-72`). Compared by
`crates/mold-inference/tests/pulid_adapter_parity.rs`.

| | |
| --- | --- |
| Capture script | `capture_ca_goldens.py` (imports upstream's module; it does not restate it) |
| Weights | `pulid_flux_v0.9.1.safetensors`, sha256 `92c41c3af322b02e58e1b32842e4601e08c8f16ec1fe80089dbe957df510f51d`, from [`guozinan/PuLID`](https://huggingface.co/guozinan/PuLID) |
| Upstream commit | `ToTheBeginning/PuLID` `main` (shallow clone, 2026-08-21) |
| torch | 2.13.0, CPU, float32 |
| Geometry | `dim=3072`, `dim_head=128`, `heads=16`, `kv_dim=2048` |
| Inputs | image tokens `[1, 64, 3072]` (seed `0x50554C4944434149`), identity `[1, 32, 2048]` (seed `0x50554C4944434144`) |
| Modules sampled | 0, 5, 9, 10, 15, 19 — the ends of both index ranges (double 0–9, single 10–19) plus one interior module each |

Inputs are **not** committed. Both the script and the Rust test generate them
from the same `xorshift64*` stream, so a fixture of any size costs nothing in
the repository; the Rust test pins the stream so a drift in either copy is a
test failure rather than a silent comparison against different inputs.

Each module contributes two arrays:

- `ca{i}.probe` — 512 scattered elements of the `[1, 64, 3072]` output, at flat
  indices drawn from seed `0x50554C4944434150`.
- `ca{i}.stats` — mean, sample standard deviation, and max absolute value over
  the whole output, so a change that misses every probe index still shows up.

`ca_goldens.json` records the same provenance in machine-readable form.

### Measured agreement

mold's port against these goldens, CPU f32:

| module | max abs | max rel |
| --- | --- | --- |
| 0 | 9.91e-7 | 9.91e-7 |
| 5 | 2.27e-6 | 2.27e-6 |
| 9 | 6.97e-6 | 6.97e-6 |
| 10 | 4.05e-6 | 3.73e-6 |
| 15 | 1.68e-5 | 1.26e-5 |
| 19 | 2.29e-5 | 1.12e-5 |

The budgets in the test are `1e-4` absolute and `5e-5` relative — a little
above the worst measurement so an attention-path change is a visible regression
rather than a flake, and far below the values themselves (`absmax` reaches 39.4
on `pulid_ca.19`), so a wrong port still fails.

The error grows with depth because the later modules have larger weights, not
because anything accumulates: each module is evaluated independently on the
same input. The test computes its summary statistics in `f64` — a naive `f32`
sum over 196 608 elements loses more precision than the port does, which would
make the assertion a measurement of the harness.

### Regenerating

```sh
python crates/mold-inference/testdata/pulid/capture_ca_goldens.py \
  --pulid-weights /path/to/pulid_flux_v0.9.1.safetensors \
  --pulid-repo tmp/PuLID
```

Needs only `torch` and `safetensors`. `tmp/` is gitignored and is where the
upstream clone lives.
||||||| cc276d63
