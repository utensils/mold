# Configuration

mold keeps configuration in two places by design:

- **`config.toml`** — the hand-editable _bootstrap_ file in `~/.mold/` (or
  `$MOLD_HOME`). Owns paths, ports, credentials, and the model-path
  entries that `mold pull` writes.
- **`mold.db` (SQLite)** — owns user preferences: the `[expand]` section,
  global generation defaults (`default_width`, `default_height`,
  `default_steps`, `embed_metadata`, `t5_variant`, `qwen3_variant`,
  `default_negative_prompt`), the `last-model` sidecar, and per-model
  generation defaults (`default_steps`, `default_guidance`,
  `default_width`, `default_height`, `scheduler`, `negative_prompt`,
  `lora`, `lora_scale`). These fields moved to the DB in
  [#265](https://github.com/utensils/mold/issues/265) so GUI writes and
  hand-curated TOML no longer fight over the same file.

Environment variables still override both surfaces at read time.
Upgrading from an earlier release runs a one-shot import of the
preference slices of `config.toml` into the DB on first launch — your
existing values carry over.

## Managing Config from the CLI

`mold config` routes writes to the right surface based on the key
prefix:

```bash
mold config list                    # Show all settings (both surfaces)
mold config get server_port         # Get a value
mold config set server_port 8080    # Bootstrap key → writes config.toml
mold config set expand.enabled true # User preference → writes mold.db
mold config set default_width 1024  # Generation default → writes mold.db
mold config where expand.enabled    # Print which surface owns this key
mold config edit                    # Open config.toml in $EDITOR
```

`mold config set` prints the resolved surface in the output line (for
example `Set expand.enabled = true [db]`) so you can tell at a glance
where the write landed. `mold config where <key>` also reports any
active `MOLD_*` env var that overrides the stored value at runtime.

See the [CLI Reference](/guide/cli-reference#mold-config) for the full list of
keys and options.

## Config File

```toml{8-11,14-17}
default_model = "flux2-klein:q8"
models_dir = "~/.mold/models"
server_port = 7680
default_width = 1024
default_height = 1024

# Global default negative prompt (CFG models only)
# default_negative_prompt = "low quality, worst quality, blurry, watermark"

[models."flux-dev:bf16"]
default_steps = 25
default_guidance = 3.5
# lora = "/path/to/adapter.safetensors"
# lora_scale = 0.8

[models."sd15:fp16"]
default_steps = 25
default_guidance = 7.5
negative_prompt = "worst quality, low quality, bad anatomy"

[expand]
enabled = false
backend = "local"
model = "qwen3-expand:q8"
temperature = 0.7

# Per-family expansion tuning
# [expand.families.sd15]
# word_limit = 50
# style_notes = "Short keyword phrases for CLIP-L."

# [expand.families.flux]
# word_limit = 200
# style_notes = "Rich natural language descriptions."

[logging]
# level = "info"              # Log level (overridden by MOLD_LOG env var)
# file = false                # Enable file logging to ~/.mold/logs/
# dir = "~/.mold/logs"        # Custom log directory
# max_days = 7                # Days to retain rotated log files
```

## Environment Variables

Environment variables take precedence over config file values.

### Core

| Variable             | Default                 | Description                         |
| -------------------- | ----------------------- | ----------------------------------- |
| `MOLD_HOME`          | `~/.mold`               | Base directory for config and cache |
| `MOLD_DEFAULT_MODEL` | `flux2-klein`           | Default model name                  |
| `MOLD_HOST`          | `http://localhost:7680` | Remote server URL                   |
| `MOLD_MODELS_DIR`    | `$MOLD_HOME/models`     | Model storage directory             |
| `MOLD_PORT`          | `7680`                  | Server port                         |
| `MOLD_LOG`           | `warn` / `info`         | Log level                           |

### Generation

| Variable              | Default | Description                              |
| --------------------- | ------- | ---------------------------------------- |
| `MOLD_EAGER`          | —       | `1` to keep all components loaded        |
| `MOLD_OFFLOAD`        | —       | `1` to force CPU↔GPU block streaming     |
| `MOLD_EMBED_METADATA` | `1`     | `0` to disable PNG metadata              |
| `MOLD_PREVIEW`        | —       | `1` to display images inline in terminal |
| `MOLD_T5_VARIANT`     | `auto`  | T5 encoder: auto/fp16/q8/q6/q5/q4/q3     |
| `MOLD_QWEN3_VARIANT`  | `auto`  | Qwen3 encoder: auto/bf16/q8/q6/iq4/q3    |
| `MOLD_SCHEDULER`      | —       | SD1.5/SDXL: ddim/euler-ancestral/uni-pc  |

### Prompt Expansion

| Variable                    | Default           | Description                      |
| --------------------------- | ----------------- | -------------------------------- |
| `MOLD_EXPAND`               | —                 | `1` to enable expansion          |
| `MOLD_EXPAND_BACKEND`       | `local`           | `local` or OpenAI-compatible URL |
| `MOLD_EXPAND_MODEL`         | `qwen3-expand:q8` | LLM model for expansion          |
| `MOLD_EXPAND_TEMPERATURE`   | `0.7`             | Sampling temperature             |
| `MOLD_EXPAND_THINKING`      | —                 | `1` to enable thinking mode      |
| `MOLD_EXPAND_SYSTEM_PROMPT` | —                 | Custom system prompt template    |
| `MOLD_EXPAND_BATCH_PROMPT`  | —                 | Custom batch prompt template     |

### Server

| Variable                    | Default             | Description                                                                                                                                                       |
| --------------------------- | ------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `MOLD_GPUS`                 | all visible         | Which GPUs the server uses: comma-separated ordinals (`0,1`) or `all`. See [Multi-GPU](/guide/cli-reference#multi-gpu)                                            |
| `MOLD_QUEUE_SIZE`           | `200`               | Max queued generation jobs; overflow returns HTTP 503 with `Retry-After`                                                                                          |
| `MOLD_OUTPUT_DIR`           | `~/.mold/output`    | Image output directory (set empty to disable)                                                                                                                     |
| `MOLD_THUMBNAIL_WARMUP`     | —                   | `1` to prebuild gallery thumbnails at server startup (default: disabled)                                                                                          |
| `MOLD_WEB_DIR`              | —                   | Override the web gallery SPA bundle location. First resolved path among this, `$XDG_DATA_HOME/mold/web`, `~/.mold/web`, `<binary dir>/web`, and `./web/dist` wins |
| `MOLD_GALLERY_ALLOW_DELETE` | —                   | `1` to allow `DELETE /api/gallery/image/:filename` (default: off — returns 403). Pair with `MOLD_API_KEY` when public-facing                                      |
| `MOLD_DB_PATH`              | `MOLD_HOME/mold.db` | Override the SQLite gallery metadata DB location                                                                                                                  |
| `MOLD_DB_DISABLE`           | —                   | `1` to disable the SQLite metadata DB entirely — server and CLI fall back to filesystem walks                                                                     |
| `MOLD_CORS_ORIGIN`          | —                   | Restrict CORS to specific origin                                                                                                                                  |
| `MOLD_API_KEY`              | —                   | API key for authentication (single key, comma-separated, or `@/path/to/keys.txt`)                                                                                 |
| `MOLD_RATE_LIMIT`           | —                   | Per-IP rate limit for generation endpoints (e.g., `10/min`, `5/sec`, `100/hour`)                                                                                  |
| `MOLD_RATE_LIMIT_BURST`     | —                   | Burst allowance override (defaults to 2x rate, capped at 100)                                                                                                     |

### Upscaling

| Variable                 | Default | Description                                                    |
| ------------------------ | ------- | -------------------------------------------------------------- |
| `MOLD_UPSCALE_MODEL`     | —       | Default upscaler model for `mold upscale`                      |
| `MOLD_UPSCALE_TILE_SIZE` | —       | Tile size for memory-efficient upscaling (0 to disable tiling) |

### Auth

| Variable   | Default | Description                        |
| ---------- | ------- | ---------------------------------- |
| `HF_TOKEN` | —       | HuggingFace token for gated models |

### Gallery Metadata Database

mold persists generation metadata in a SQLite database at `MOLD_HOME/mold.db`
(override with `MOLD_DB_PATH`). Both surfaces — the CLI's local generation
path and the HTTP server — write a row per saved file: prompt, negative
prompt, model, seed, steps, guidance, dimensions, LoRA, scheduler, the
file's mtime/size, the generation duration, and a `source` column
(`server` / `cli` / `backfill`).

The DB powers `/api/gallery` so listings stay fast on large directories
(no per-request file walk) and surface metadata for formats that don't
embed it (mp4, gif, webp). PNG / JPEG outputs still get the existing
embedded `mold:parameters` chunk in addition to the row.

On server startup the DB runs an asynchronous reconciliation pass:

- new files in `MOLD_OUTPUT_DIR` get rows added (synthesizing metadata
  from the filename when no embedded chunk is present)
- rows whose backing files have been removed (manual `rm`, file manager,
  etc.) get pruned
- size/mtime changes trigger a row refresh

Set `MOLD_DB_DISABLE=1` to opt out — both surfaces fall back to the
filesystem walk + embedded-metadata behavior from before. The NixOS
module exposes the same toggle:

```nix
services.mold = {
  enable = true;
  metadataDb.enable = false;          # opt out
  # metadataDb.path = "/var/lib/mold/custom.db";   # override location
};
```

## Advanced

### Device and Path Overrides

| Variable                       | Default | Description                                                                        |
| ------------------------------ | ------- | ---------------------------------------------------------------------------------- |
| `MOLD_DEVICE`                  | —       | Force device placement, currently `cpu` for debugging                              |
| `MOLD_TRANSFORMER_PATH`        | —       | Override transformer weights path                                                  |
| `MOLD_VAE_PATH`                | —       | Override VAE weights path                                                          |
| `MOLD_SPATIAL_UPSCALER_PATH`   | —       | Override LTX spatial upscaler path                                                 |
| `MOLD_TEMPORAL_UPSCALER_PATH`  | —       | Override LTX temporal upscaler path                                                |
| `MOLD_DISTILLED_LORA_PATH`     | —       | Override the default LTX-2 distilled LoRA path                                     |
| `MOLD_T5_PATH`                 | —       | Override T5 encoder path                                                           |
| `MOLD_CLIP_PATH`               | —       | Override CLIP-L encoder path                                                       |
| `MOLD_CLIP2_PATH`              | —       | Override CLIP-G encoder path for SDXL                                              |
| `MOLD_T5_TOKENIZER_PATH`       | —       | Override T5 tokenizer path                                                         |
| `MOLD_CLIP_TOKENIZER_PATH`     | —       | Override CLIP-L tokenizer path                                                     |
| `MOLD_CLIP2_TOKENIZER_PATH`    | —       | Override CLIP-G tokenizer path for SDXL                                            |
| `MOLD_TEXT_TOKENIZER_PATH`     | —       | Override generic text tokenizer path for Qwen/Z-Image                              |
| `MOLD_DECODER_PATH`            | —       | Override Wuerstchen decoder weights path                                           |
| `MOLD_QWEN2_VARIANT`           | `auto`  | Qwen-family Qwen2.5-VL encoder: `auto`, `bf16`, `q8`, `q6`, `q5`, `q4`, `q3`, `q2` |
| `MOLD_QWEN2_TEXT_ENCODER_MODE` | `auto`  | Qwen-family placement mode: `auto`, `gpu`, `cpu-stage`, `cpu`                      |

These are mainly useful for custom local model layouts, manual debugging, or
testing alternative weight files without editing `config.toml`.

### Per-component device placement

Override which device (CPU or a specific GPU) runs each part of the diffusion
pipeline. All variables accept the same four forms: `auto` (preserve the
engine's VRAM-aware default), `cpu`, `gpu` (= `gpu:0`), or `gpu:N` for a
specific ordinal.

| Variable                   | Applies to                        | Notes                                                                                                                                               |
| -------------------------- | --------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------- |
| `MOLD_PLACE_TEXT_ENCODERS` | Every model family (Tier 1)       | Single knob that moves every text encoder slot as a group. Picking `cpu` frees the transformer's full VRAM budget without triggering block offload. |
| `MOLD_PLACE_TRANSFORMER`   | FLUX, Flux.2, Z-Image, Qwen-Image | Per-component override. Interacts with `MOLD_OFFLOAD` — blocks still stream from CPU but target the chosen ordinal.                                 |
| `MOLD_PLACE_VAE`           | FLUX, Flux.2, Z-Image, Qwen-Image | Decode stage; CPU is fine for preview, GPU is faster.                                                                                               |
| `MOLD_PLACE_T5`            | FLUX                              | Per-encoder override; unset falls through to `MOLD_PLACE_TEXT_ENCODERS`.                                                                            |
| `MOLD_PLACE_CLIP_L`        | FLUX                              | Per-encoder override.                                                                                                                               |
| `MOLD_PLACE_CLIP_G`        | SDXL and others that use CLIP-G   | Per-encoder override.                                                                                                                               |
| `MOLD_PLACE_QWEN`          | Flux.2, Z-Image, Qwen-Image       | Per-encoder override for the Qwen text encoder.                                                                                                     |

Precedence (highest wins): CLI flag (`--device-text-encoders`, `--device-vae`, …)
→ env var → `[models."name:tag".placement]` TOML block → engine auto.

The web UI's **Placement** panel, the `PUT /api/config/model/:name/placement`
route, and `mold run --device-*` flags all write/read the same shape, so any
surface can drive it.

Tier 2 per-component controls are intentionally gated: families other than
FLUX, Flux.2, Z-Image, and Qwen-Image only honor Tier 1 (`MOLD_PLACE_TEXT_ENCODERS`)
— their engines don't yet split encoder/transformer/VAE across devices. Setting
the advanced variables on a Tier 1-only family is a no-op (the web UI hides
the Advanced disclosure for those families so it isn't misleading).

For Qwen-Image and Qwen-Image-Edit:

- CUDA `auto` prefers BF16 when enough headroom remains after the transformer
  load, and falls back to quantized GGUF variants when a resident encoder or
  edit-conditioning path would otherwise be too heavy.
- Metal/MPS `auto` prefers the quantized Qwen2.5-VL GGUF encoder path to reduce
  memory pressure during prompt encoding.
- `qwen-image-edit` still loads the Qwen2.5-VL vision tower for image
  conditioning, but quantized `MOLD_QWEN2_VARIANT` values keep the language side
  smaller and stage the vision weights only when needed.

### Debug and Family-Specific Knobs

| Variable                                              | Default                    | Description                                                                                                  |
| ----------------------------------------------------- | -------------------------- | ------------------------------------------------------------------------------------------------------------ |
| `MOLD_SD3_DEBUG`                                      | —                          | Enable verbose SD3.5 pipeline logging                                                                        |
| `MOLD_QWEN_DEBUG`                                     | —                          | Enable verbose Qwen-Image pipeline logging                                                                   |
| `MOLD_ZIMAGE_DEBUG`                                   | —                          | Enable verbose Z-Image pipeline logging                                                                      |
| `MOLD_LTX_DEBUG`                                      | —                          | Enable verbose LTX Video / LTX-2 pipeline logging                                                            |
| `MOLD_LTX_DEBUG_FILE`                                 | `/tmp/mold-ltx2-debug.log` | Append LTX Video / LTX-2 debug output to a file                                                              |
| `MOLD_LTX_DEBUG_COMPARE_UNCOND`                       | —                          | Log conditional vs unconditional LTX-2 prompt-context comparisons                                            |
| `MOLD_LTX_DEBUG_ALT_PROMPT`                           | —                          | Use an alternate prompt string for LTX-2 prompt-sensitivity debugging                                        |
| `MOLD_LTX_DEBUG_DISABLE_AUDIO_BRANCH`                 | —                          | Debug-only LTX-2 switch to disable the audio branch during native runs                                       |
| `MOLD_LTX_DEBUG_DISABLE_CROSS_ATTENTION_ADALN`        | —                          | Debug-only LTX-2 switch to bypass cross-attention AdaLN modulation                                           |
| `MOLD_LTX2_DEBUG_DISABLE_TRANSFORMER_GATED_ATTENTION` | —                          | Debug-only LTX-2 switch to bypass transformer gated attention                                                |
| `MOLD_LTX2_DEBUG_FORCE_CPU_PROMPT_ENCODER`            | —                          | Force the native LTX-2 prompt encoder onto CPU while leaving the rest of the runtime on the selected backend |
| `MOLD_LTX2_DEBUG_TIMINGS`                             | —                          | Emit native LTX-2 pipeline, phase, and denoise timing summaries for optimization work                        |
| `MOLD_LTX2_DEBUG_STAGE_PREFIX`                        | —                          | Write decoded native LTX-2 stage artifacts using this filename prefix                                        |
| `MOLD_LTX2_DEBUG_BLOCKS`                              | —                          | Emit per-block native LTX-2 transformer debug logs                                                           |
| `MOLD_LTX2_DEBUG_BLOCK_DETAIL`                        | —                          | Restrict detailed native LTX-2 block logging to a specific transformer block index                           |
| `MOLD_LTX2_DEBUG_LOAD_BLOCKS`                         | —                          | Log native LTX-2 transformer block loading details                                                           |
| `MOLD_LTX2_FORCE_EAGER`                               | —                          | Force eager native LTX-2 transformer loading instead of layer streaming                                      |
| `MOLD_LTX2_FORCE_STREAMING`                           | —                          | Force native LTX-2 transformer layer streaming                                                               |
| `MOLD_LTX2_FP8_INPUT_SCALE_MODE`                      | `skip`                     | Debug override for native LTX-2 FP8 input-scale handling (`skip`, `emulate`, `divide`, `multiply`)           |
| `MOLD_LTX2_FP8_WEIGHT_SCALE_MODE`                     | `apply`                    | Debug override for native LTX-2 FP8 checkpoint weight-scale handling (`apply`, `skip`, `scaled-mm`)          |
| `MOLD_WUERSTCHEN_DEBUG`                               | —                          | Enable verbose Wuerstchen pipeline logging                                                                   |
| `MOLD_WUERSTCHEN_DECODER_GUIDANCE`                    | `0.0`                      | Override decoder-stage CFG guidance for Wuerstchen                                                           |

These are intended for troubleshooting and development rather than normal use.

### Build-Time Metadata

| Variable            | Default | Description                                                 |
| ------------------- | ------- | ----------------------------------------------------------- |
| `MOLD_FULL_VERSION` | —       | Internal build-time version string embedded into CLI output |

This variable is set during the build and is not normally configured by users at
runtime.
