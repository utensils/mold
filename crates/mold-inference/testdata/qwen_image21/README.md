# Qwen-Image 2.1 parity goldens

Fixtures for the Qwen-Image 2.1 reference-image, transparency, and LoRA/turbo
work: the Qwen3-VL processor and vision tower, the trimmed conditioning, the
RGBA VAE encoder, the joint text/image layout and RoPE, the block-causal
transformer (full, prefix-cache extract, and cached passes), a Viggle LoRA
step, and short end-to-end denoises with injected noise. The Rust parity tests
(P1–P8 in the campaign's engine design) compare against these.

`capture.py` and `run_capture.sh` are committed **as documentation of
provenance only**. Nothing in mold's build, test, or runtime path executes
them, and mold ships no Python.

## Provenance

| | |
| --- | --- |
| Checkpoint | [`Qwen/Qwen-Image-2.1`](https://huggingface.co/Qwen/Qwen-Image-2.1/tree/b3179ad355be050328e483a9dfdd9e60cd62adfa) at `b3179ad355be050328e483a9dfdd9e60cd62adfa` |
| Oracle | diffusers [`e0abab83b5df05de9e7abd788643c1a7c1e42e28`](https://github.com/huggingface/diffusers/tree/e0abab83b5df05de9e7abd788643c1a7c1e42e28) (`pipelines/qwenimage21/pipeline_qwenimage21.py`, `models/transformers/transformer_qwenimage21.py`, `models/autoencoders/autoencoder_kl_qwenimage21.py`), installed editable from a source export of that commit |
| LoRA | [`Viggle/Qwen-Image-2.1-viggle-turbo`](https://huggingface.co/Viggle/Qwen-Image-2.1-viggle-turbo/tree/bb26a0f38e5fe6c124aaccc9187a87eed5d9ed13) at `bb26a0f38e5fe6c124aaccc9187a87eed5d9ed13` |
| Captured | 2026-09-26, plato, one NVIDIA L40S (46 GB), driver 595.71.05 |
| Python stack | Python 3.13, torch 2.11.0+cu128 (bundled cuDNN 9.19), torchvision 0.26.0+cu128, transformers 5.17.0, diffusers 0.41.0.dev0 (the commit above), peft 0.21.0, safetensors 0.8.0, numpy 2.5.3, **Pillow 12.3.0** |
| Precision | fp32 with TF32 disabled for matmul **and** cuDNN (`allow_tf32 = False` on both); `*_bf16` files are the same inputs through the bf16 checkpoint weights |
| Attention | torch SDPA: diffusers' default `QwenImage21AttnProcessor` (one call per prefix segment), transformers' `sdpa` for Qwen3-VL |

`manifest.json` records all of this mechanically (`environment`, `upstream`,
`inputs`), plus every file's SHA-256 and the capture's own checks
(`results`).

### Pinned revision

The generation profile cites `b3179ad`; the snapshot the production home
installed is `790c926` (HF `main`). The Hub's own file listing at both
revisions (`GET /api/models/Qwen/Qwen-Image-2.1/revision/<rev>?blobs=true`)
shows **every weight, config, processor, and scheduler file identical**: the
four text-encoder shards, both transformer shards, and the VAE carry the same
LFS SHA-256 at both revisions, and those are exactly the `sha256` values in
`manifest.rs::qwen_image21_manifests`. The only differences are `README.md`
(a Discord link and a WeChat QR line), `.gitattributes`, and a new
`assets/qr.png`. The pin therefore stays **`b3179ad`** everywhere; the
installed `790c926` snapshot is byte-identical for every file mold loads, so
the weights are symlinked from it rather than downloaded again. Note the repo
has no `tokenizer/` directory: the tokenizer lives in `processor/`.

### Scheduler constants

`scheduler/scheduler_config.json` at `b3179ad`: `FlowMatchEulerDiscreteScheduler`,
`use_dynamic_shifting: true`, `time_shift_type: "exponential"`,
`base_image_seq_len: 256`, `max_image_seq_len: 8192`, `base_shift: 0.5`,
`max_shift: 0.9`, `shift_terminal: 0.02`, `shift: 1.0`. These are mold's
`qwen_image/sampling.rs` constants exactly. The `4096 / 1.15` pair in
`pipeline_qwenimage21.py:61-71` are only `calculate_shift`'s argument
defaults and the `config.get(..., default)` fallbacks at `:724-730`; the
pipeline always passes the scheduler config's values, so they never apply to
this checkpoint. `mu` is linear and unclamped: at 2048² (16,384 target
tokens) it extrapolates to 1.3129 (`schedules.json`). Viggle's turbo scheduler
is the same config with `shift_terminal: null`.

## Re-running

```bash
# 1. venv (uv; plain pip works the same)
uv venv --python python3 tmp/qwen21-venv
VIRTUAL_ENV=tmp/qwen21-venv uv pip install --index-url https://download.pytorch.org/whl/cu128 torch torchvision
VIRTUAL_ENV=tmp/qwen21-venv uv pip install "transformers==5.17.0" "pillow==12.3.0" "peft==0.21.0" \
  "safetensors==0.8.0" accelerate numpy huggingface_hub
# diffusers at the pinned commit (a source export; no git needed)
mkdir -p tmp/diffusers-e0abab
curl -sL https://codeload.github.com/huggingface/diffusers/tar.gz/e0abab83b5df05de9e7abd788643c1a7c1e42e28 \
  | tar xz -C tmp/diffusers-e0abab --strip-components=1
VIRTUAL_ENV=tmp/qwen21-venv uv pip install -e tmp/diffusers-e0abab

# 2. model root: small files at b3179ad, weights symlinked from any snapshot
#    whose LFS digests match manifest.rs (b3179ad and 790c926 both do)
#    -> /storage/mold/fixtures/qwen_image21/model-b3179ad/{model_index.json,processor,scheduler,text_encoder,transformer,vae}
# 3. Viggle LoRAs at bb26a0f -> /storage/mold/fixtures/qwen_image21/viggle/

# 4. capture (one 46 GB GPU; each memory phase is its own process)
CUDA_VISIBLE_DEVICES=0 crates/mold-inference/testdata/qwen_image21/run_capture.sh
```

`run_capture.sh` runs the stages `images,viggle,pillow,cpu` → `encode` →
`transformer,lora` → `e2e` → `bf16` → `alpha` → `manifest`. The fp32 text
encoder (31.6 GiB resident) and fp32 transformer (~28 GiB) never share the
card: P8 feeds the pipeline the prompt embeddings the `encode` stage captured
from the very same `_get_qwen_prompt_embeds` call, in place of re-running the
text encoder, and the rest of `__call__` runs unmodified.

Two environment traps, both guarded:

- **cuDNN.** A Nix devshell's `LD_LIBRARY_PATH` carries its own cuDNN engine
  libraries; torch then loads its bundled `libcudnn.so.9` beside a *different
  release's* engine `.so`s, convolutions keep working, and only
  `torch.backends.cudnn.version()` notices. `run_capture.sh` drops every
  `LD_LIBRARY_PATH` entry that holds a `libcudnn*`, and `capture.py` refuses to
  run if any `libcudnn*` outside torch's own wheel is mapped.
- **Autograd.** `encode_prompt`, `vae.encode`, and the transformer forward are
  not `no_grad` upstream (only `__call__` is); `capture.py` disables grad
  globally, or the saved activations alone overflow the card at fp32.

One deliberate edit to an upstream object: the text encoder's `lm_head` is
replaced by an identity before it is moved to the GPU. The pipeline reads
hidden states only (`pipeline_qwenimage21.py:310`), and the fp32 head plus a
`[L, 151936]` logits tensor is what overflowed the single card. No hidden
state depends on it.

## Inputs

- `ref_opaque.png` — 1536×1024 RGB scene (house, sun, a sign reading
  `MOLD 2.1`), resized by the pipeline to **1248×832** (78×52 latent tokens,
  1014 image-pad tokens). Downscale path.
- `ref_rgba.png` — 640×800 RGBA sticker: soft 14 px alpha edge, opaque stem,
  α≈200 leaf, α=96 glass pane, and **magenta under α=0** so a port that does
  not premultiply shows it. Resized to **928×1152** (58×72 tokens, 1044 pad
  tokens). Upscale path.
- P6: both references `[opaque, rgba]`, prompt `Put the red apple from image 2
  on the white sign in image 1, keep everything else unchanged.`, negative
  `blurry, lowres, watermark`, 512×512 target (1024 tokens), timesteps
  `900/1000` then `600/1000`, target latents `torch.randn` on a CPU generator
  (seeds 21, 22).
- P8: `ref_opaque.png` only, prompt `Change the sky to a warm sunset with
  orange clouds, keep the house and the sign unchanged.`, 512×512 target,
  `output_resolution=1024`, `use_kv_cache=True`, `true_cfg_scale=1`, initial
  latents from `p8_noise.safetensors` (seed 1234, passed as `latents=`).
  `base4`: 4 steps, default sigmas, shipped scheduler. `turbo6`: Viggle r256,
  `sigmas=[1.0, 0.9375, 0.875, 0.75, 0.5, 0.25]`, `shift_terminal=None`.

Noise is torch's, not mold's ChaCha stream, so the engine's parity test must
inject `p6_inputs`/`p8_noise` latents rather than seeding.

## Files

Every `.safetensors` carries one `__metadata__` key, `capture`, holding a
sorted JSON document (prompts, shapes, seeds, notes): safetensors writes its
metadata map in hash order, so more than one key would make a re-capture's
bytes differ. Each tensor keeps the dtype it had upstream (the header says
which): fp32 captures are F32, bf16 captures keep BF16 where the pipeline
held BF16 (P3 embeddings, P6/P7 outputs) and are widened to F32 where the
capture copied them (P8 per-step latents, decoded pixels).

Small fixtures are committed here; large ones live in
`/storage/mold/fixtures/qwen_image21/captures/` (`manifest.json` `large_dir`)
and are pinned by SHA-256 below. Parity tests read that directory from
`QWEN_IMAGE21_FIXTURES`.

| File | What |
| --- | --- |
| `calculate_dimensions.json` | U1: `calculate_dimensions` rows, including exact `.5` ties both ways (15.5→16, 20.5→20, 31.5→32, 33.5→34, 40.5→40) |
| `templates.json` | U2: the exact strings the pipeline hands the processor for 0–3 references, their UTF-8 bytes and token ids, and what `apply_chat_template` would have produced instead |
| `hf_configs.json` | U14: `model_index`, scheduler, transformer, VAE, text-encoder, and processor configs at `b3179ad`, verbatim, each with its SHA-256 |
| `schedules.json` | U11: sigmas, timesteps, and `mu` for base (40/4 steps at 512²–2752×1536) and turbo (6 steps, `shift_terminal: null`), plus `transformer_timesteps_bf16` (`t.to(bfloat16) / 1000`, what the bf16 transformer receives) |
| `pillow_resize.safetensors` | U8: Pillow 12.3.0 LANCZOS resizes (up/down/odd) of small RGBA and RGB inputs, and their white composites, as `uint8 HxWxC` |
| `p1_processor_ids.safetensors` | P1: `input_ids`, `attention_mask`, `mm_token_type_ids`, `image_grid_thw` for the two references |
| `p3_*_image_pad_mask.safetensors` | P3: `image_pad_mask` after the system-prompt drop, per case |
| `p5_layout.safetensors` / `.json` | P5: joint image mask, `image_ids`, target mask, RoPE frame/height/width indices; segments, `prefix_len`, and block spans per case (`p6_pos`, `p6_neg`, `p8_pos`, `t2i_p8`) |
| `alpha_histograms.json` | Alpha statistics of six bf16 1024² t2i renders |
| `viggle_lora_layout.json` | Viggle v0.2.1 r128/r256 key layout (for the LoRA mapper) |
| large `p1_processor_pixels` | P1: `pixel_values` `[8232, 1536]` f32 and `image_grid_thw` |
| large `p2_vision_fp32` | P2: vision `last_hidden_state`, merger output, three deepstack outputs |
| large `p3_<case>_{fp32,bf16}` | P3: trimmed pre-norm hidden states (`prompt_embeds`) + `image_pad_mask` |
| large `p3_<case>_lm_internals_*` | P3 debug: MRoPE `position_ids`, `visual_pos_masks`, `inputs_embeds`, and LM hidden states 0–4, 18, 35, 36 (36 is pre-norm) |
| large `p4_vae_encode_fp32` | P4: VAE input, `mode()`, normalized, packed latents, and a decode round trip, for both references |
| large `p4_vae_encoder_internals_64x96_fp32` | P4 debug: every encoder stage (incl. each `avg_shortcut`) on a 64×96 RGBA crop |
| large `p5_rope_freqs` | P5: the complex RoPE tables (`view_as_real`) |
| large `p6_inputs` / `p6_outputs_{fp32,bf16}` / `p6_internals_*` | P6: inputs; `full_a`, `extract_a`, `cached_b`, `full_b`; block 0/31 activations, modulation, temb, layer 0/31 K/V |
| large `p7_lora_r128_{fp32,bf16}` | P7: `full_a` with Viggle r128 applied by PEFT (unmerged) and without |
| large `p8_noise`, `p8_{base4,turbo6}_{fp32,bf16}` | P8: per-step and final latents, decoded RGBA float, and the PNG |
| large `alpha_t2i_*.png` | the six renders behind `alpha_histograms.json` |

## Findings a port must reproduce

- **Templates are raw strings**, not `apply_chat_template`
  (`pipeline_qwenimage21.py:207-220, 252-259, 276-285`), verified by recording
  the processor's own call. The two differ: the chat template drops the
  `<imageN>` markers and the space before `<image2>`. System prompt
  `Comprehend and analyze the provided prompt.`, `drop_idx` 14,
  `<|image_pad|>` id 151655. Padding side is left.
- **`pixel_values` = `(x − 127.5) / 127.5` in f32** (torchvision backend, the
  default whenever torchvision is installed). Without torchvision, transformers
  picks its PIL backend, which computes `x/255·2−1` — up to 1 ULP different.
  The processor's `smart_resize` is a no-op on these sizes (checked exactly).
- **The VL copy is resized RGBA, then pasted over white with the alpha mask**
  (`:266-271`); resize is premultiplied inside Pillow (RGBA→RGBa→LANCZOS→RGBA,
  `reducing_gap=None`), so colour under α=0 becomes 0 and the VAE sees it as
  −1.
- **Pre-norm hidden states are large**: `prompt_embeds` reach ~860 at fp32; a
  port that reads the normalized state is off by the norm, not a tolerance.
- **bf16 quantizes the timestep.** `timestep = t.to(latents.dtype) / 1000`
  (`:770, 775`): in bf16, 900 → `0.8984375`, 600 → `0.6015625`. The bf16 P6
  fixtures carry those exact values.
- **The prefix cache is exact in fp32** (`extract == full` bitwise;
  `cached_b` vs `full_b` target relative error 7.7e-7) and ~4.8e-3 relative in
  bf16, matching the pipeline's own docstring (`:585-589`).
- **Opaque t2i renders are not all-255 alpha.** Every 40-step 1024² render
  without the recipe had alpha bytes below 255 (1.5–5.3% of pixels; minimum
  250–252, and the flat-illustration prompt reached 204 on ~100 edge pixels
  around the mountains). No render had α < 204. A rule of "emit RGB only when
  every alpha byte is 255" would therefore keep RGBA for essentially every
  opaque render. The 4-step P8 edits show the same (min 249–253). See
  `alpha_histograms.json`.
- **Transparency recipe** (model card and GitHub README):
  `This is an RGBA image with transparency. <description>. The image has alpha
  channel and the background is transparent.` Both recipe renders produced
  clean transparent backgrounds (41% and 40% of pixels at α=255, 50%/59% below
  128), with the translucent glass rendered partially transparent.

## Viggle LoRA layout

`viggle_lora_layout.json`. Both v0.2.1 files use diffusers naming with a
`transformer.` prefix, `lora_A.weight [r, in]` / `lora_B.weight [out, r]`, all
BF16, 454 tensors over 227 modules: `modulation.1`,
`time_text_embed.timestep_embedder.linear_{1,2}`, and per block 0–31
`attn.{to_q,to_k,to_v,to_out.0}` and `img_mlp.{proj,gate_layer,out}`. There
are **no `.alpha` tensors**: alpha lives only in `__metadata__`
`lora_adapter_metadata`, a JSON string whose keys are themselves prefixed
(`"transformer.lora_alpha": 256`, `"transformer.r": 256`, empty
`rank_pattern`/`alpha_pattern`), so alpha/rank = 1. Licence: Qwen Research
License (non-commercial), byte-identical to the base model's `LICENSE`.

## SHA-256

Generated from `manifest.json` (the authority; `large` = `/storage/mold/fixtures/qwen_image21/captures`).

| Location | File | Bytes | SHA-256 |
| --- | --- | ---: | --- |
| committed | `alpha_histograms.json` | 11210 | `674abcdadd5dc9f43c7cbe2184813143807bed59f381d3c030fa1ac9baed1ebf` |
| committed | `calculate_dimensions.json` | 3562 | `38b1eb895bda582fc37020a007afc7d331da8912a10d1a1a7cbb2bd1834ad72f` |
| committed | `hf_configs.json` | 6989 | `6271e6ef2836213f187b76f63b556ffa72de3f0ba80c557adc9cf3c9e57944e9` |
| committed | `p1_processor_ids.safetensors` | 51616 | `3f27422158de83d97a893c820f538473b2c0629086fabd3dd68786650a57de0f` |
| committed | `p3_p6_neg_image_pad_mask.safetensors` | 2205 | `81ce1b14862b9404abbe4a47af85acf8a1775bf0ad3b5c6534367ed4ab055d76` |
| committed | `p3_p6_pos_image_pad_mask.safetensors` | 2220 | `189e475fa3bee8e1a2bbc2343f2ac08275a9a0f28eb09b0cae82219a8474f2c4` |
| committed | `p3_p8_pos_image_pad_mask.safetensors` | 1167 | `1d61a18758b9442629902dd4b9b35284fb3503e1f6ac44b387d5e8bae53f1080` |
| committed | `p3_t2i_p8_image_pad_mask.safetensors` | 139 | `7dc896c843be0e873cb0b6d17c813e7368725ec989801019a336c3751bf936bd` |
| committed | `p5_layout.json` | 2722 | `fc1033a68554a67897b002ca5ec575f8d40b66e158ff4520ef800592029bc4eb` |
| committed | `p5_layout.safetensors` | 850117 | `6e9350848db898bce2c93e3deeba059ae9d3a27fe47f7018ea40e59ff06cda5b` |
| committed | `pillow_resize.safetensors` | 53215 | `1646e9eec58e189a54ee3c4c613c6e7a78754347f255e4abb6631f236deb5eae` |
| committed | `ref_opaque.png` | 20613 | `803d81c457de3fc530484d8c30bdbdf22543514780eec927cce4d6199a068dbd` |
| committed | `ref_rgba.png` | 42518 | `7489f9a36012b5c90d00dac49c318faa6b4ecf7e72ef119786c35b1467789b87` |
| committed | `run_capture.sh` | 1364 | `fb53590eaf06cdce7aa2da08e11209a416079efbd716e35321ca1afd0d4ec8ac` |
| committed | `schedules.json` | 20300 | `19b8e12bdf27225285f1be3a2ac795c392609c452be6fef9425916e51ba32146` |
| committed | `templates.json` | 11187 | `7386049ae273acd37239a42dbed7d2ccef2b05b45c78104d487d0d19328e35d1` |
| committed | `viggle_lora_layout.json` | 7943 | `c0c953630b1e6e502d5a549127d1b1de8eb914470c086d3372b26661a3cad030` |
| large | `alpha_t2i_bakery.png` | 2207460 | `2fc7da38f3ddc1a5ee793fddd640886f4bb8a58a2269660f48f0aecb4ce62676` |
| large | `alpha_t2i_dragon.png` | 1051102 | `6a042d26a1961501c25e35f52a5f15d024a9315484c24e60f771b3614069f82f` |
| large | `alpha_t2i_fox.png` | 1547781 | `255a4d448b7ff2679db2e7361c0351275800ab6eabfda801c1558127c88f3056` |
| large | `alpha_t2i_juice.png` | 845553 | `12709961b6ab0d9c5a7791298a5c5c820ac3d38b1ab9fc98e771a4e42adf559f` |
| large | `alpha_t2i_lake.png` | 716938 | `939962b06e15f9db86030b38d9a061f01646890c76777f8af5fa9186e73de50c` |
| large | `alpha_t2i_teapot.png` | 1316803 | `558e9bea81599a8626bf6dde19cef8d6fc7b0728c937da284e509944e19e5064` |
| large | `encode_record_bf16.json` | 1260 | `6ece3d475f0405eb637332ccbdb38e1804b55f3f1cf5e4097be41fbccde69554` |
| large | `encode_record_fp32.json` | 6926 | `3b878aadbba90bfeca4a8f7e9b33ed12cd4f1d67d65b626fa32f7989fc1cb093` |
| large | `p1_processor_pixels.safetensors` | 50577744 | `247b568e118fcff28b37959140d21581b02da71ebea1ea8a858f14b9c383d036` |
| large | `p2_vision_fp32.safetensors` | 172806736 | `a96f3d6af2bf00f6b5cc3c36c8d0051600e407650a1217b276f9aedffb7044cb` |
| large | `p3_p6_neg_bf16.safetensors` | 17082701 | `8e60689f58d9d8bcdaa6fa4ba2b27a352ca0caec45258c9ac30b94de76570711` |
| large | `p3_p6_neg_fp32.safetensors` | 34163021 | `300bdca86d47a9ce809ac70dbdcf194aca2bc4e2e2a573ed10de70ac0fa120e9` |
| large | `p3_p6_neg_lm_internals_bf16.safetensors` | 275173667 | `98ed776d771ac109c488d98422e46286f97194bfa185f783dbd163c4076ec391` |
| large | `p3_p6_neg_lm_internals_fp32.safetensors` | 275173667 | `86c81b7b4cebbfb18c5a736dd840c59a3cabe56915f06201008bd339834b2ee1` |
| large | `p3_p6_pos_bf16.safetensors` | 17205660 | `b77cfd7d2688baf0a63d0660e12c1c4a4e8d4d0e5ec3730bc9e741c4b4aa89c1` |
| large | `p3_p6_pos_fp32.safetensors` | 34408860 | `0f94f587f569734ab96c2a90b136a0a15832409ab7b8fdb8ddb5e5b7238c8b7a` |
| large | `p3_p6_pos_lm_internals_bf16.safetensors` | 277140122 | `8679b62a334fa5cd8ae0686c0f187e640e837804c0639f8485efa82b4448ae89` |
| large | `p3_p6_pos_lm_internals_fp32.safetensors` | 277140122 | `d3180a0aba420d257bfa7786314c74076930f49dc2aa7c4523f9e609b94cb25b` |
| large | `p3_p8_pos_bf16.safetensors` | 8578431 | `c7216751d6f5c2a3a2b9ba9b9226d59ac98eb8f301e1fa84b20562d7e68a5968` |
| large | `p3_p8_pos_fp32.safetensors` | 17155455 | `bbf020f061e09bc5454fec0d8833af2e922e840bde707757b7d0a0d9153453ee` |
| large | `p3_p8_pos_lm_internals_bf16.safetensors` | 139094973 | `eae4d9bfd748ba425370d219014bcf5b5fdefd74c37c503a5ad181884ac7bba4` |
| large | `p3_p8_pos_lm_internals_fp32.safetensors` | 139094973 | `e033b9d6b0c45531e47c422502a4090aeb89bda0cd99c8d5c6bc194d414230e1` |
| large | `p3_t2i_p8_bf16.safetensors` | 221563 | `fc95d0db5363f4d8b6d1eaec0ab5db34fbe622539adc66d9614c039f4ceb2d9c` |
| large | `p3_t2i_p8_fp32.safetensors` | 442747 | `398c1f5111ea09952dc80ed7cb2bc17e7c88f3b5b8f8136ffa1535304fda1bbc` |
| large | `p3_t2i_p8_lm_internals_fp32.safetensors` | 5374800 | `edf28d48e2c0d228bd9864058baf162de73513f01409e396bd80409c79e3898e` |
| large | `p4_opaque_roundtrip.png` | 282995 | `022d8808242326900907cf52cab0bc4ac27adfadcfe9d18041ceb56eecdf3acc` |
| large | `p4_rgba_roundtrip.png` | 567001 | `f933cd7e0b169c8795fe2f87eac43aff6ac7e114d880dd2fde2442311d656a5f` |
| large | `p4_vae_encode_fp32.safetensors` | 73759808 | `900b2a6f156dcb2d4e77ea8194953811ea4de4631fea9118306c057bd098e30b` |
| large | `p4_vae_encoder_internals_64x96_fp32.safetensors` | 4996784 | `720df3e49c8b504dd399cb7ad99776114c939c1a9de1616d001508b24469e8ac` |
| large | `p5_rope_freqs.safetensors` | 12669952 | `51c3f99d1c92e3d2ba9375c5fbfd2802e56127bcce538498475605c3c4e9bc09` |
| large | `p6_inputs.safetensors` | 37041204 | `4f2d68670727a47af360288768d5c287ce614e1db52a32bea2b42e7b7b2eedca` |
| large | `p6_internals_bf16.safetensors` | 499712944 | `78e833bf713f61749afe8898a9a238e6d1f3ce8a07144dbb828626f70e064493` |
| large | `p6_internals_fp32.safetensors` | 999424936 | `fc700371096ff5a9f1fb85617fc9c5c1d46d83b3f9daa4ee6cb6eaecb4904ade` |
| large | `p6_outputs_bf16.safetensors` | 3702156 | `d98b942171354e0cba08880f37bc7c2df2b878dd403cf061ff99aa96f349e36f` |
| large | `p6_outputs_fp32.safetensors` | 7403656 | `b660ae0230ef1db5c83012b6b77d2c00c5a47568e60b862008e166bde1abeae9` |
| large | `p7_lora_r128_bf16.safetensors` | 2380970 | `8cc0033dab9310a6256fceb6cf25a7aafaaab0d69aa671c1b306363b3356e0a5` |
| large | `p7_lora_r128_fp32.safetensors` | 4761260 | `128f25b63758675583441055370cb675f699e643904775071ed85b99630e6892` |
| large | `p8_base4_bf16.png` | 180803 | `c5a7b524ca8e23ea80a7620443a4ef4cfb463701d3582f947f7190bb9fb2c994` |
| large | `p8_base4_bf16.safetensors` | 5506072 | `10662d187fb1c3bfa4a5b373b9e452fc5c6a32672b74ec30cd98791320747b3f` |
| large | `p8_base4_fp32.png` | 174815 | `73bfd1e8bf26b5115cab9d98e5da3787d7cb0162dff2dd2143c49da839149fcf` |
| large | `p8_base4_fp32.safetensors` | 5506072 | `4339612de5949e3dd8c31ff5e313c6c597c9c8391ab881eeb1134a4d953305b6` |
| large | `p8_noise.safetensors` | 262368 | `bf2e6c832e2591761c636e077f20d3f2471a6367dc32c1a049861919167a8124` |
| large | `p8_turbo6_bf16.png` | 290705 | `f5eb7c94e825f05449fabc924e7ea83348789ddea83790ee9f82dfa8059810e4` |
| large | `p8_turbo6_bf16.safetensors` | 6030664 | `a8e2cda722d8561d1f3cdf05eda2b953a31846a22aba4bb1f768950282cfbc57` |
| large | `p8_turbo6_fp32.png` | 298467 | `a4d4d22b4feaa49978edafdbb3fce4792733aa9eea67aa4c13a071687066dc48` |
| large | `p8_turbo6_fp32.safetensors` | 6030664 | `6c8bd4f39a5806cbce9f5afd61fba65f796c5d75dc76ed16eba0f7bd6fb1767c` |
| large | `pillow_reference_resize.safetensors` | 14752176 | `0721e23b644fd735494fbd3ca1ec45bbab9c74f63158d9383ff8ce56751c6b70` |
| large | `ref_opaque_resized_1248x832.png` | 44652 | `61c06d64419e6cc557291db23eac7c5e14e34ac7571c3595c5167af5ffab1d82` |
| large | `ref_opaque_resized_1248x832_white.png` | 39625 | `b801a35b92a8f3cba1211376d54ff0ac7e86520681ae73db8bc1e0807168cc79` |
| large | `ref_rgba_resized_928x1152.png` | 148544 | `205e19808b9610dfa166570c78e44c7c1a1c3a078feb8a22a6c292fbf5f0de15` |
| large | `ref_rgba_resized_928x1152_white.png` | 116273 | `aa69ace6b057845a0ffedbe749d4592ce298d65838773dd1cdaccbcfe9b0abcb` |
| large | `results.json` | 7044 | `2975ec8aeb4f37552339be7ecc2857c627c7be5071d3ddd8c06a7bd41269279d` |
