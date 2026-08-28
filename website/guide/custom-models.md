# Custom Models & LoRA

mold works best when models come from the built-in manifest, but you can also
point it at manual weight paths or set per-model defaults in `config.toml`.

## Manual Model Entries

The `[models]` table lets you define or override model paths and defaults:

```toml
[models."flux-dev:q4"]
transformer = "/srv/mold/models/flux1-dev-Q4_1.gguf"
vae = "/srv/mold/models/ae.safetensors"
t5_encoder = "/srv/mold/models/t5xxl_fp16.safetensors"
clip_encoder = "/srv/mold/models/clip_l.safetensors"
t5_tokenizer = "/srv/mold/models/t5.tokenizer.json"
clip_tokenizer = "/srv/mold/models/clip.tokenizer.json"
default_steps = 25
default_guidance = 3.5
```

This is useful when:

- you keep weights outside the default Hugging Face cache layout
- you want a custom local model definition
- you want per-model defaults without repeating CLI flags

## Path Override Environment Variables

If you need a one-off override, environment variables take precedence over the
config file. See [Configuration](/guide/configuration#advanced) for the full
list, including:

- `MOLD_TRANSFORMER_PATH`
- `MOLD_VAE_PATH`
- `MOLD_SPATIAL_UPSCALER_PATH`
- `MOLD_T5_PATH`
- `MOLD_CLIP_PATH`
- `MOLD_TEXT_TOKENIZER_PATH`
- `MOLD_DECODER_PATH`

These are primarily debugging and advanced deployment tools, not the normal path
for day-to-day use.

## LoRA Defaults

For LoRA-capable models, you can keep a default adapter in config:

```toml
[models."flux-dev:q4"]
lora = "/path/to/style.safetensors"
lora_scale = 0.8
```

At runtime, CLI flags still win:

```bash
mold run flux-dev:q4 "portrait" --lora another-style.safetensors --lora-scale 1.1
```

## Multi-LoRA Stacking

Pass `--lora` more than once to stack adapters. The deltas merge additively
into the base weights -- `W' = W + Σ scale_i · B_i @ A_i` -- so the result is
the same as composing two style edits in series, but only one transformer
build is required:

```bash
mold run flux-dev "an epic portrait" \
  --lora cinematic.safetensors --lora-scale 0.8 \
  --lora moody-light.safetensors --lora-scale 0.4
```

The web UI's LoRA picker stacks up to **4** LoRAs per generation; each row
gets its own scale slider and remove button.

Repeating the same stack is cheap: FLUX and Qwen-Image fingerprint the adapters,
their order, and their scales, and reuse the merged transformer when the next
request asks for exactly that stack. Changing any of the three rebuilds it.

Wan 2.2 A14B adapters usually ship as a high-noise/low-noise pair; bind an
adapter to one expert with `--lora file.safetensors@high` or `@low`. When the
marker is omitted mold infers the binding from the adapter's filename and
discloses the inference in progress output; an adapter with no expert affinity
applies to both experts. The fp8-scaled A14B tiers refuse LoRA stacks -- use
the GGUF or bf16 tiers for adapters.

## Step-Distilled LoRAs

Some adapters change the sampling schedule rather than the style. Qwen-Image's
Lightning LoRAs run the model in 4 or 8 CFG-free steps on any checkpoint you
already have, instead of downloading a second pre-merged transformer:

```bash
mold run qwen-image-2512:q8 "a snowy mountain cabin at twilight" \
  --lora ~/loras/Qwen-Image-2512-Lightning-8steps-V1.0-bf16.safetensors \
  --steps 8 --guidance 1.0
```

Set `--steps` to the adapter's step count and `--guidance 1.0` -- mold turns on
classifier-free guidance above `1.0`, which costs a second forward pass per step
that a CFG-free distilled adapter was not trained for. See
[Qwen-Image](/models/qwen-image#lightning-loras-on-a-checkpoint-you-already-have)
for the exact adapter files, which checkpoint line each belongs to, and the
caveat that these recipes come from upstream rather than from a mold test run.

## Browsing & Installing LoRAs

The web, desktop, and iPhone Models surfaces search Hugging Face and Civitai
live. Open **Models** (the Discover segment), choose a family and LoRA kind
where available, then use **Install** on the web or **Pull** in a native app.
Once installed, it appears in compatible Create LoRA pickers on that host.
The web Installed shelf refreshes when the pull finishes. Model details mark
each runtime component ready or missing; when the server supplies a repair
target, **Repair** queues that exact component rather than reinstalling an
ambiguous repository. Unload a model before deleting its on-disk files.

The same flow is available over the server API:

```bash
# Search installable FLUX LoRAs
curl "http://localhost:7680/api/catalog/search?q=cinematic&family=flux&kind=lora"

# Install by catalog id; cv: and hf: ids are both supported
curl -X POST "http://localhost:7680/api/catalog/cv:1234567/download"

# List installed LoRAs for a family
curl "http://localhost:7680/api/catalog/installed?kind=lora&family=flux"
```

Model downloads may also queue companion files such as tokenizers, VAEs, text
encoders, or LTX-2 support assets. See [Model Discovery Catalog](/docs/catalog)
for the complete API surface.

## Civitai Trigger Words

LoRAs trained with explicit trigger phrases (Civitai `trainedWords`) surface
those phrases as click-to-insert chips next to the LoRA picker in the web UI.
Clicking a chip appends the phrase to the active prompt with sensible
comma-separation -- no more flipping back to the Civitai page to copy/paste.

## LoRA Rules

- Supported families: **FLUX, Flux.2, LTX-2, SD1.5, SD3, SDXL, Qwen-Image
  (+ Qwen-Image-Edit), Wan, Z-Image**. Wuerstchen and LTX-Video are not yet wired --
  attaching a LoRA there returns a 400 with the current supported-family list.
  (Source of truth: `mold-core::validation::require_lora_capable_family`.)
- `.safetensors` only
- scale must be between `0.0` and `2.0`
- the server resolves the path on the machine doing inference

That last point matters for remote setups. If you call a remote `mold serve`,
the LoRA file path must exist on the server host.

## Auto-Pulled Models vs Manual Paths

If you use the built-in manifest:

- `mold pull` downloads shared files once
- `mold list` and `mold info` know the model metadata
- checksums and `.pulling` markers work normally

If you bypass the manifest with custom paths:

- you own the file layout
- you own any upgrades or replacements
- docs and tooling will know less about that model automatically

## Good Defaults

For most users:

1. use manifest-backed models first
2. add per-model defaults in `config.toml`
3. use env-path overrides only for debugging or special deployments
