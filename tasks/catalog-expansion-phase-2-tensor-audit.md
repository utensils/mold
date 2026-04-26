# Phase-2 task 2.2 — SD1.5 / SDXL tensor-prefix audit

> Findings from running
> `cargo run -p mold-ai-inference --features dev-bins --bin sd_singlefile_inspect`
> against three top-downloaded Civitai checkpoints. The inspector source
> lives at `crates/mold-inference/src/bin/sd_singlefile_inspect.rs`.

## Fixtures

| File | Civitai | Size | Base model |
|---|---|---|---|
| `dreamshaper-8.safetensors` | model 4384, version 128713 | 2.0 GB | SD 1.5 |
| `juggernaut-xl.safetensors` | model 133005, version 1759168 (Ragnarok) | 6.6 GB | SDXL 1.0 |
| `pony-v6.safetensors` | model 257749, version 290640 | 6.5 GB | Pony |

Stored at `~/Downloads/civitai-fixtures/` — **not** committed to the repo. Phase 2 reproduces them by re-downloading from Civitai's public `/api/download/models/<versionId>` endpoint.

## Cross-file probe table

```
probe                     dreamshaper-8  juggernaut-xl  pony-v6
unet                                686           1680      1680
vae                                 248            248       248
clip_l_sdxl                           0            197       197
clip_g_sdxl                           0            390       390
clip_l_sd15                         197              0         0
text_encoder_diffusers                0              0         0
text_encoder_2_diffusers              0              0         0
```

Probe definitions (from `sd_singlefile_inspect.rs::PROBES`):

| Label | Prefix |
|---|---|
| `unet` | `model.diffusion_model.*` |
| `vae` | `first_stage_model.*` |
| `clip_l_sdxl` | `conditioner.embedders.0.transformer.text_model.*` |
| `clip_g_sdxl` | `conditioner.embedders.1.model.*` |
| `clip_l_sd15` | `cond_stage_model.transformer.text_model.*` |
| `text_encoder_diffusers` | `text_encoder.*` (diffusers stragglers) |
| `text_encoder_2_diffusers` | `text_encoder_2.*` (diffusers stragglers) |

## Findings

1. **The UNet and VAE prefixes are universal across SD1.5 + SDXL Civitai single-files**: `model.diffusion_model.*` and `first_stage_model.*`. No A1111 / kohya / WebUI variant in the audited set deviates.
2. **CLIP location is family-determined, not checkpoint-determined.** SD15 has CLIP-L at `cond_stage_model.transformer.text_model.*` only. SDXL has CLIP-L at `conditioner.embedders.0.transformer.text_model.*` and CLIP-G at `conditioner.embedders.1.model.*`. The handoff's concern that "some SDXL checkpoints might mix in `cond_stage_model.*` for CLIP-L" did **not** materialize — neither Pony nor Juggernaut shows any SD-style prefix.
3. **VAE counts are identical (248) across SD1.5 and SDXL.** Both families use the same VAE architecture (different trained weights). The loader can share VAE-key remap logic between SD15 and SDXL paths.
4. **Pony is structurally indistinguishable from generic SDXL.** Same UNet count (1680), same VAE count (248), same CLIP-L (197), same CLIP-G (390). Pony-specific flavor lives in trained weights only — no architectural divergence — so it goes through the SDXL loader unchanged. Same expected for Illustrious / NoobAI / SDXL Lightning / SDXL Hyper (verify in task 2.10 UAT).
5. **Juggernaut carries an extra `denoiser.sigmas` tensor (1 × F32).** Looks like a custom sigma noise schedule baked in for some downstream tool. Inert to the canonical SDXL engine — loader should silently ignore unmapped tensors at the `denoiser.*` prefix rather than error.
6. **CLIP-G uses OpenCLIP layout, not HuggingFace CLIP layout.** Sample tensor: `conditioner.embedders.1.model.ln_final.bias`. The `.model.` prefix here is OpenCLIP's, not HF's. CLIP-G needs a OpenCLIP→HF rename pass on top of the prefix strip. CLIP-L is HF-format and only needs the prefix strip.
7. **Inner block / attention / FF / transformer-block naming is identical between A1111 and diffusers** below the path-translation boundary. Sample: `input_blocks.1.1.transformer_blocks.0.attn1.to_q.weight` vs diffusers `down_blocks.0.attentions.0.transformer_blocks.0.attn1.to_q.weight` — `attn1.to_q.weight` matches verbatim. Only the **outer** `input_blocks.X.Y` ↔ `down_blocks.A.{resnets,attentions}.B` portion needs translation. ResNet sub-block (`in_layers/out_layers/emb_layers`) and VAE attention (`mid.attn_1.{q,k,v,proj_out,norm}` ↔ `mid_block.attentions.0.{to_q,to_k,to_v,to_out.0,group_norm}`) need element-level rename rules.

## Implications for tasks 2.3 / 2.4 / 2.5

- **2.3 dispatcher** — branch only on `Family::{Sd15, Sdxl}` plus single-file flag. No need for per-checkpoint sniffing logic; the family alone is sufficient.
- **2.4 SD15 loader** — three rename passes:
  1. UNet: strip `model.diffusion_model.` then translate A1111 outer block path → diffusers (`input_blocks/middle_block/output_blocks` → `down_blocks/mid_block/up_blocks`, ResNet `in_layers/out_layers/emb_layers` → `norm1/conv1/...`, attention is wrapped in `Transformer2DModel` so `input_blocks.X.1.norm` → `down_blocks.A.attentions.B.norm`, etc.)
  2. VAE: strip `first_stage_model.` then translate `mid.attn_1.{q,k,v,proj_out,norm}` → `mid_block.attentions.0.{to_q,to_k,to_v,to_out.0,group_norm}` and similar `mid.block_1/block_2` → `mid_block.resnets.{0,1}`. Encoder/decoder up/down naming roughly aligns.
  3. CLIP-L: strip `cond_stage_model.transformer.` and pass straight through — keys land at `text_model.*` which is what candle expects.
- **2.5 SDXL loader** — same three passes plus:
  4. CLIP-G: strip `conditioner.embedders.1.model.` and apply OpenCLIP→HF rename pass. The OpenCLIP layout uses `transformer.resblocks.X.{ln_1, ln_2, attn.in_proj_weight, attn.in_proj_bias, attn.out_proj, mlp.c_fc, mlp.c_proj}` whereas HF CLIP uses `text_model.encoder.layers.X.{layer_norm1, layer_norm2, self_attn.{q_proj,k_proj,v_proj,out_proj}, mlp.{fc1,fc2}}`. The `attn.in_proj_weight` is a fused `[3*d, d]` slab that has to be split into `q_proj/k_proj/v_proj` weights — that's a tensor-data transformation, not just a rename.
- **Tolerate unmapped extras.** `denoiser.sigmas` (and any future stragglers) should warn, not error. The loader's contract: every diffusers key the engine asks for must resolve; any extra single-file keys that don't match a remap rule are silently dropped.

## Reference implementations

- diffusers' `convert_diffusers_to_original_stable_diffusion.py` — has the inverse direction (diffusers → A1111). Phase 2 ports its mapping table in reverse.
- diffusers' `convert_open_clip_checkpoint.py` — handles the OpenCLIP `attn.in_proj_weight` split for CLIP-G.
- The two together cover the full SDXL single-file → diffusers translation. SD1.5 only needs the first.

## Reproducing

```bash
mkdir -p ~/Downloads/civitai-fixtures && cd ~/Downloads/civitai-fixtures
curl -fsSL -o juggernaut-xl.safetensors  'https://civitai.com/api/download/models/1759168'
curl -fsSL -o pony-v6.safetensors        'https://civitai.com/api/download/models/290640'
curl -fsSL -o dreamshaper-8.safetensors  'https://civitai.com/api/download/models/128713'

cd /Users/jeffreydilley/github/mold
cargo build -p mold-ai-inference --features dev-bins --bin sd_singlefile_inspect
./target/debug/sd_singlefile_inspect \
    ~/Downloads/civitai-fixtures/dreamshaper-8.safetensors \
    ~/Downloads/civitai-fixtures/juggernaut-xl.safetensors \
    ~/Downloads/civitai-fixtures/pony-v6.safetensors
```

Re-running the inspector against any future Civitai checkpoint that surfaces a tensor-loader edge case (e.g. an Illustrious or NoobAI variant once they ship through the catalog) is the standard regression check. Add the fixture filename + observed probe row to the cross-file table above and decide whether the loader needs a new branch.
