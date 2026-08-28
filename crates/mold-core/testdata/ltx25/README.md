# LTX-2.5 golden headers

Header-only fixtures for the LTX-2.5 probes (`ltx25_probe.rs`) and the
transformer weight index (`ltx2_weight_index.rs`). None of them carries tensor
data; every reader they exercise stops at the header.

| File                                        | Bytes   | sha256                                                             |
| ------------------------------------------- | ------- | ------------------------------------------------------------------ |
| `distilled-int8-convrot.header.safetensors` | 160,298 | `71f9a0aef5344d5c92b31e0b2f9c54bee623c575560272dfcb7e79b33843060e` |
| `distilled-q4-k-m.header.gguf`              | 402,810 | `58db89941d9bcde9a4ca4b57bfbd8bf246afe11f1233d88d8a322e5d82463486` |
| `audio-vae-stub-194.safetensors`            | 194     | `b802d81d68ae6450441b83fe4be6eefcfd2239d25b1bfc734da786c8ecf14157` |

## `distilled-int8-convrot.header.safetensors`

Cut from the official `Lightricks/LTX-2.5` file
`diffusion_models/ltx-2.5-22b-distilled-transformer-comfy-int8-convrot.safetensors`
(21,504,034,224 bytes, sha256
`2edbdb4465cd6c3b532cd67a31ddb38a63e97dcad20be3729675e2a4e8caf92b`, the
`ltx-2.5-22b-distilled:int8-conv` manifest's transformer). Recipe:

1. Read the 8-byte little-endian header length and the JSON header.
2. Keep `__metadata__` verbatim (`config`, `model_version`, `license`,
   `gemma_source_checkpoint`).
3. Keep every tensor outside `model.diffusion_model.transformer_blocks.*`
   (509 tensors: `adaln_single`, `prompt_adaln_single`, `audio_*`, the four
   `av_ca_*` gates, `patchify_proj`, `proj_out`, the two
   `*_embeddings_connector` stacks, `keyframes_abs_pos_embedding`, the
   scale-shift tables) and blocks **0** and **47** complete (140 tensors each:
   `I8 .weight` + `F32 .weight_scale` + `U8 .comfy_quant` per linear, BF16
   norms/biases). Drop blocks 1..46.
4. Keep real `dtype` and `shape`; rewrite `data_offsets` contiguously in the
   original offset order; serialize minified (`separators=(',', ':')`).
5. Write `len(json)` as 8 LE bytes + the JSON. No data section.

Measured from it: block 0 at rest 388,065,632 B, widened to BF16
773,349,760 B (the `.weight_scale` / `.comfy_quant` sidecars are consumed by
the loader, not materialized); `adaln_single.linear.weight` is
`[36864, 4096]` while `prompt_adaln_single.linear.weight` is `[8192, 4096]`.

## `distilled-q4-k-m.header.gguf`

Bytes `0..402810` of the public, ungated
`Abiray/LTX-2.5-Distilled-GGUF/LTX-2.5-Distilled-Q4_K_M.gguf` at revision
`7b0c2025441f1bf12c18eac375ad21f5e3d3c9e0` — exactly the GGUF v3 header
(7 metadata entries, 4,349 tensor infos); the data section starts at byte
402,810 and is not included, so the bounded reader must accept a truncated
data section.

```sh
curl -sSL -r 0-402809 -o distilled-q4-k-m.header.gguf \
  "https://huggingface.co/Abiray/LTX-2.5-Distilled-GGUF/resolve/7b0c2025441f1bf12c18eac375ad21f5e3d3c9e0/LTX-2.5-Distilled-Q4_K_M.gguf"
```

Facts the tests pin: bare ComfyUI tensor names (no `model.diffusion_model.`
prefix), `model_version` is the plain string `"2.5.0"`, `config` and
`gemma_source_checkpoint` are JSON documents stored as strings,
`config.transformer.num_layers = 48`, no `caption_projection` tensors,
no `audio_ff_bias` key; dtypes F32 ×2,603 / Q5_K ×1,292 / Q4_K ×350 /
Q6_K ×102 / F16 ×2; block 0 at rest 249,582,336 B.

## `audio-vae-stub-194.safetensors`

A byte-faithful reconstruction of the 194-byte file found at
`shared/ltx2/vae/ltx-2.5-audio-vae-bf16.safetensors` on hal9000 on
2026-08-28 in place of the real 364,866,540-byte audio VAE, under that
download's still-valid `.sha256-verified` sidecar: two `F32 [1]` tensors
(`audio_vae.per_channel_statistics.mean-of-means`,
`vocoder.vocoder.conv_pre.weight`) sharing `data_offsets [0, 4]`, no
`__metadata__`, one 4-byte zero blob — the layout of a synthetic probe
fixture. `validate_ltx25_audio_components` refuses it with
"expected an LTX-2.5 model_version", and `Config::file_is_complete` must not
let the marker vouch for it.
