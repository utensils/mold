# PuLID v1.1 (SDXL) parity goldens

Fixtures captured from upstream `ToTheBeginning/PuLID`'s **v1.1** pipeline
(`pulid/pipeline_v1_1.py`), which targets Stable Diffusion XL rather than
FLUX. This directory is independent of `crates/mold-inference/testdata/pulid/`
(the FLUX golden set) — nothing here overlaps it, and nothing there was
touched to produce this.

Captured **as documentation of provenance only**. Nothing in mold's build,
test, or runtime path executes these scripts, and mold ships no Python.

## Provenance

| | |
| --- | --- |
| Upstream | <https://github.com/ToTheBeginning/PuLID> |
| Commit | `1aa2fc7df4bf51080df39f355f9abdc1cbfefbaa` (2025-08-01) |
| Checkpoint | `pulid_v1.1.safetensors`, [`guozinan/PuLID`](https://huggingface.co/guozinan/PuLID) |
| Checkpoint size | 984,405,232 bytes |
| Checkpoint SHA-256 | `4cb8ceec1078e0165399b88332ab3c5971619111b8e1730e6bae64144aabae41` |
| Checkpoint license | Apache-2.0 |
| Captured | 2026-08-22, aarch64-darwin, CPU, float32 |
| torch | 2.13.0 (CPU wheel) |
| diffusers | 0.40.0 |

Re-run any capture by hand:

```bash
python3 -m venv /tmp/pulid-sdxl-venv
/tmp/pulid-sdxl-venv/bin/pip install torch --index-url https://download.pytorch.org/whl/cpu
/tmp/pulid-sdxl-venv/bin/pip install diffusers safetensors numpy einops accelerate transformers huggingface_hub
git clone https://github.com/ToTheBeginning/PuLID /tmp/PuLID  # or reuse tmp/PuLID in this repo

# 1. Layer ordering map (no checkpoint weights required for SD1.5's map;
#    SDXL's map is cross-checked against the checkpoint when --pulid-adapter is given)
/tmp/pulid-sdxl-venv/bin/python capture_attn_layer_map.py \
  --pulid-adapter /path/to/pulid_v1.1.safetensors \
  --out .

# 2. IDFormer goldens
/tmp/pulid-sdxl-venv/bin/python capture_idformer_goldens.py \
  --pulid-repo /tmp/PuLID \
  --adapter /path/to/pulid_v1.1.safetensors \
  --out . --f16-tolerance-check

# 3. IDAttnProcessor2_0 ID-branch goldens
/tmp/pulid-sdxl-venv/bin/python capture_attn_goldens.py \
  --pulid-repo /tmp/PuLID \
  --adapter /path/to/pulid_v1.1.safetensors \
  --attn-layer-map attn_layer_map.json \
  --out . --f16-tolerance-check
```

All three scripts need network access once, to fetch the small
`unet/config.json` files for `stabilityai/stable-diffusion-xl-base-1.0` and
`stable-diffusion-v1-5/stable-diffusion-v1-5` from the Hugging Face Hub (no
model weights are downloaded — the UNet is constructed on `torch.device
("meta")` from config alone).

## Two corrections to the original task brief

Both are settled by the checkpoint header and the upstream source, not by
assumption, and are called out here so a future reader doesn't "fix" the
scripts back to the brief's numbers.

1. **`id_cond` is `[*, *, 1280]`, not `[*, *, 1792]`.** PuLID v1.1's
   `id_cond = torch.cat([id_ante_embedding, id_cond_vit], dim=-1)`
   (`pipeline_v1_1.py:230`) concatenates the antelopev2 ArcFace embedding
   (512-d) with the EVA02-CLIP-L-14-336 projected CLS token — and that
   tower's `embed_dim` is **768**, per
   `PuLID/eva_clip/model_configs/EVA02-CLIP-L-14-336.json`, not 1280.
   512 + 768 = 1280. This is directly confirmed by the checkpoint's own
   `id_adapter.id_embedding_mapping.0.weight` shape `[1024, 1280]`
   (`nn.Linear` stores `[out_features, in_features]`) — see
   `attn_layer_map.json`'s sibling assertions and
   `idformer_goldens.json`'s `note` field. It is also exactly the FLUX
   adapter's `ID_COND_DIM` in
   `../pulid/capture_true_cfg_goldens.py`, because `IDFormer` is the
   identical class in both pipelines (same file:
   `pulid/encoders_transformer.py`; v1.1 just loads it from a checkpoint
   prefixed `id_adapter.*` instead of FLUX's `pulid_encoder.*`).

2. **The `id_embedding` an attn2 layer receives carries 32 tokens, not 37.**
   `IDFormer.forward` internally concatenates its 32 learned query latents
   with `num_id_token=5` per-image identity tokens
   (`encoders_transformer.py`'s `torch.cat((latents, x), dim=1)`) so that
   both can attend to the vision-tower context together, but slices back to
   `latents[:, :self.num_queries]` — 32 — before returning
   (`encoders_transformer.py`, final lines). That sliced, 32-token tensor
   is exactly what `pipeline_v1_1.py` hands to every UNet cross-attention
   layer as `id_embedding`. The internal 37-token concatenation never
   crosses the `IDFormer` boundary.

## Files

| File | What it is | Size |
| --- | --- | --- |
| `capture_attn_layer_map.py` | Provenance for both `attn_layer_map*.json` | 10 KB |
| `attn_layer_map.json` | SDXL UNet attn-processor traversal + checkpoint cross-check | 49 KB |
| `attn_layer_map_sd15.json` | SD1.5 UNet attn-processor traversal (no checkpoint; none exists) | 8.7 KB |
| `capture_idformer_goldens.py` | Provenance for `idformer_goldens.*` | 11 KB |
| `idformer_goldens.safetensors` | IDFormer outputs: single image, uncond, two images | 787 KB |
| `idformer_goldens.json` | Their statistics and capture parameters | 2.0 KB |
| `capture_attn_goldens.py` | Provenance for `attn_goldens.*` | 15 KB |
| `attn_goldens.safetensors` | `IDAttnProcessor2_0` ID-branch probes for 3 layers x 2 `id_scale`s | 20 KB |
| `attn_goldens.json` | Their statistics and capture parameters | 4.1 KB |

Total: 904 KB (repository budget was 3 MB).

## `attn_layer_map.json` / `attn_layer_map_sd15.json`

Built by constructing the real `diffusers.UNet2DConditionModel` module graph
on `torch.device("meta")` from each model's published `unet/config.json`
(no weights fetched) and enumerating `unet.attn_processors.items()` — the
exact traversal `PuLIDPipeline.hack_unet_attn_layers`
(`pipeline_v1_1.py:129-149`) walks to build
`id_adapter_attn_layers = nn.ModuleList(unet.attn_processors.values())`, whose
position **is** `processor_index` and **is** the index PuLID's own
`id_adapter_attn_layers.<i>.id_to_k`/`id_to_v` checkpoint keys use
(`load_pretrain`, `pipeline_v1_1.py:151-163`, splits the checkpoint by
leading module name — `id_adapter` vs `id_adapter_attn_layers`).

Each entry:

```json
{
  "processor_index": 1,
  "module_name": "down_blocks.1.attentions.0.transformer_blocks.0.attn2",
  "kind": "attn2",
  "hidden_size": 640,
  "cross_attention_dim": 2048,
  "heads": 10,
  "dim_head": 64,
  "attn2_ordinal": 0,
  "checkpoint": { "id_to_k.weight": [640, 2048], "id_to_v.weight": [640, 2048] }
}
```

`kind` is `"attn1"` (self-attention, no PuLID weights — `cross_attention_dim`
is `None`) exactly when the processor name ends `attn1.processor`, `"attn2"`
otherwise, matching `hack_unet_attn_layers`'s own test. `attn2_ordinal` is
present only on attn2 entries: a 0-based count over attn2 modules in
traversal order (0..69 for SDXL, 0..15 for SD1.5).

**SDXL**: 140 total processors, 70 attn1 + 70 attn2. Every attn2
`processor_index` has `id_adapter_attn_layers.<i>.id_to_k.weight` /
`.id_to_v.weight` in the checkpoint; every attn1 index has neither; every
`id_to_k` out-features equals that layer's own `hidden_size`; the checkpoint's
weighted indices are exactly the traversal's attn2 indices (no orphans). All
four assertions are enforced by `capture_attn_layer_map.py` itself — a
maintainer re-running it re-verifies the claim rather than trusting this file.
`cross_attention_dim` is a UNet-wide constant, 2048 (SDXL's concatenated
OpenCLIP-ViT-L + OpenCLIP-ViT-bigG pooled/hidden width).

**SD1.5**: 32 total processors, 16 attn1 + 16 attn2 — pinned here as the
traversal shape only, since **no PuLID-SD1.5 checkpoint exists**;
`checkpoint` is `null` throughout and no `attn2_ordinal`-vs-weights assertion
runs. `cross_attention_dim` is 768 (CLIP ViT-L/14 alone).

## `idformer_goldens.{json,safetensors}`

Upstream `pulid/encoders_transformer.py::IDFormer` — the SAME class the FLUX
adapter uses — constructed with every default argument (matching
`PuLIDPipeline.__init__`'s bare `IDFormer()`) and loaded strictly from
`pulid_v1.1.safetensors`'s `id_adapter.*` tensors (172 tensors; the FLUX
checkpoint uses `pulid_encoder.*` instead — different prefix, identical
architecture and shapes, confirmed by `id_adapter.proj_out`'s `[1024, 2048]`
and `id_adapter.latents`'s `[1, 32, 1024]` matching the FLUX golden's values
exactly).

Inputs are generated from the `xorshift64*` stream already shared by
`../pulid/capture_eva_goldens.py` and
`crates/mold-inference/src/pulid_fixtures.rs::DeterministicStream` — bit for
bit the same four-line algorithm, new seeds (`PULIDSXI`, `PULIDSXV`,
`PULIDSX2`) so this fixture set never collides with the FLUX one. Nothing but
the checkpoint and these seeds is committed.

Three cases, all `[1, 32, 2048]` output, per `get_id_embedding`
(`pipeline_v1_1.py:171-259`):

| Golden | id_cond shape | vit_hidden shape (x5 scales) | Source |
| --- | --- | --- | --- |
| `idformer.single.output` | `[1, 1, 1280]` | `[1, 577, 1024]` | ordinary single-image path |
| `idformer.uncond.output` | zeros of the same shape | zeros of the same shape | `pipeline_v1_1.py:243-247` — PuLID's true-CFG negative branch conditions on this |
| `idformer.two_image.output` | `[1, 2, 1280]` (stacked) | `[1, 1154, 1024]` (concatenated along dim 1) | `pipeline_v1_1.py:249-256` — the multi-reference-image path |

Each golden also has a `.stats` sibling: `[mean, std, min, max, peak]` in that
order (`STAT_SLOTS`), computed in float64 to avoid the harness itself losing
precision a port needs to be held to.

### Measured f16 tolerance

`--f16-tolerance-check` re-runs each case with every input cast through f16
and back to f32, reporting the max absolute difference against the f32
golden:

| Case | max abs diff | peak magnitude | relative |
| --- | --- | --- | --- |
| `idformer.single.output` | 7.647276e-04 | 11.682 | 6.546e-05 |
| `idformer.uncond.output` | 0.0 (zeros round-trip exactly) | 11.566 | 0.0 |
| `idformer.two_image.output` | 5.633235e-04 | 12.088 | 4.660e-05 |

A Rust port compared against the f32 golden should budget noticeably tighter
than these numbers (they measure input-precision sensitivity, not port
error) — something in the `1e-4`–`1e-3` relative range, consistent with the
FLUX `IDFormer` golden's own measured `1.5e-7` absolute-vs-peak port error
(`../pulid/README.md`) plus headroom for this checkpoint's own weights.

## `attn_goldens.{json,safetensors}`

Upstream `pulid/attention_processor.py::IDAttnProcessor2_0`'s identity branch
(`__call__`'s `if id_embedding is not None:` block, lines ~299-333),
replicated verbatim for the pinned globals this checkpoint runs under —
`NUM_ZERO = 0` (no zero-token padding) and `ORTHO = ORTHO_v2 = False` (plain
additive combination, no orthogonal-projection variant). The capture script
asserts those three globals still hold in the cloned upstream before running,
so a future upstream default change cannot silently invalidate the numbers.

The full `__call__` also derives `query` and the pre-id `hidden_states` from
real UNet activations through `attn.to_q`/`to_k`/`to_v`, which needs a whole
`diffusers.Attention` module this fixture has no other use for. Instead —
exactly as `../pulid/capture_ca_goldens.py` does for `PerceiverAttentionCA` —
`query` (`[2, 64, hidden_size]`, pre head-reshape — i.e. `attn.to_q`'s own
output shape) and `attended` (`[2, 64, hidden_size]`, standing in for the
text-branch's `scaled_dot_product_attention` output, reshaped, before
`attn.to_out`) are synthetic. `id_embedding` is `[2, 32, 2048]` (32 tokens —
see the correction above), and both `id_to_k`/`id_to_v` are the pinned
checkpoint's real weights for that `processor_index`, loaded through a real
`IDAttnProcessor2_0` instance (constructed with that layer's own
`hidden_size`) so nothing about their shape or layout is reimplemented.

Three layers, chosen from `attn_layer_map.json` to cover both hidden sizes
and all three UNet regions that carry PuLID weights:

| `processor_index` | `module_name` | `hidden_size` | `heads` | `dim_head` |
| --- | --- | --- | --- | --- |
| 1 | `down_blocks.1.attentions.0.transformer_blocks.0.attn2` | 640 | 10 | 64 |
| 121 | `mid_block.attentions.0.transformer_blocks.0.attn2` | 1280 | 20 | 64 |
| 49 | `up_blocks.0.attentions.0.transformer_blocks.0.attn2` | 1280 | 20 | 64 |

For each layer and `id_scale in {1.0, 0.7}`:

- `attn{idx}.combined_s{1p0,0p7}.probe` — 512 scattered elements (seed
  `PULIDSXP + idx`) of `attended + id_scale * id_hidden_states` — the value
  written back into `hidden_states` just before `attn.to_out` runs.
- `attn{idx}.combined_s{1p0,0p7}.stats` — `[mean, std, min, max, peak]` of the
  whole `[2, 64, hidden_size]` tensor.
- `attn{idx}.id_hidden_states.probe` / `.stats` — the scale-independent
  cross-attention output over `(query, id_key, id_value)` alone (recorded
  once per layer, since it does not depend on `id_scale`).

### Measured f16 tolerance

`--f16-tolerance-check` casts `query`/`attended`/`id_embedding` through f16
and back, then re-runs the `id_scale=1.0` combination:

| Layer | max abs diff | peak magnitude | relative |
| --- | --- | --- | --- |
| `attn1` (down_blocks, 640) | 3.900528e-04 | 1.918 | 2.033e-04 |
| `attn121` (mid_block, 1280) | 3.530979e-04 | 2.378 | 1.485e-04 |
| `attn49` (up_blocks, 1280) | 3.275275e-04 | 2.309 | 1.418e-04 |

As with the IDFormer table above, these bound input-precision sensitivity,
not port correctness — a candle port compared against these f32 goldens
should target a tighter budget, on the order of the FLUX `PerceiverAttentionCA`
golden's measured port error (`1e-7`–`3e-5`, `../pulid/README.md`), with
headroom scaled to this fixture's larger `absmax` values.
