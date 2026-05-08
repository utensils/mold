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

For FLUX models, you can keep a default adapter in config:

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
into the base weights — `W' = W + Σ scale_i · B_i @ A_i` — so the result is
the same as composing two style edits in series, but only one transformer
build is required:

```bash
mold run flux-dev "an epic portrait" \
  --lora cinematic.safetensors --lora-scale 0.8 \
  --lora moody-light.safetensors --lora-scale 0.4
```

The web UI's LoRA picker stacks up to **4** LoRAs per generation; each row
gets its own scale slider and remove button.

## Browsing & Installing LoRAs

`mold catalog` indexes LoRAs from Civitai (`types=LORA`) and Hugging Face
(detected via tag / file-name / repo-id heuristics). To find FLUX LoRAs:

```bash
# CLI: list installable FLUX LoRAs
mold catalog refresh --family flux         # one-time scrape (or after a release)
mold catalog list --family flux --kind lora --limit 20

# Install a specific LoRA by catalog id
mold pull cv:1234567
```

Or in the web UI: open the **Catalog** tab, click the **LoRAs** chip, then
**Install** on the LoRA you want. Once installed, it appears in the
**Generate → Settings → LoRA** picker.

## Civitai Trigger Words

LoRAs trained with explicit trigger phrases (Civitai `trainedWords`) surface
those phrases as click-to-insert chips next to the LoRA picker in the web UI.
Clicking a chip appends the phrase to the active prompt with sensible
comma-separation — no more flipping back to the Civitai page to copy/paste.

## LoRA Rules

- FLUX only — SD1.5 / SDXL LoRA inference is not yet implemented. The
  server returns a 400 with "LoRA is currently supported only for FLUX
  models" if you attach a LoRA to any other family.
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
