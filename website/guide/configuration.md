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
mold config list                       # All settings tagged [db] / [file] / [env]
mold config list --json                # JSON form: { "value": …, "surface": … } per key
mold config get server_port            # Get a value
mold config set server_port 8080       # Bootstrap key → writes config.toml
mold config set expand.enabled true    # User preference → writes mold.db
mold config set default_width 1024     # Generation default → writes mold.db
mold config where expand.enabled       # Print which surface owns this key
mold config reset expand.enabled       # Drop the DB row; next read falls back to TOML/env/default
mold config reset --all --yes          # Drop every DB row under the active profile
mold config --profile portrait list    # Scope a command to an explicit profile (v6)
mold config edit                       # Open config.toml in $EDITOR
```

`mold config list` tags every row with its surface so you can see at a
glance which store owns each key — `[db]` for `mold.db`, `[file]` for
`config.toml`, `[env]` when a `MOLD_*` env var is currently overriding.
`mold config set` prints the same tag in its output (for example
`Set expand.enabled = true [db]`). `mold config where <key>` also
reports any env override that beats both stores at runtime.

`mold config reset <key>` drops the DB row so the next read falls back
to the TOML/env/compiled default — useful for "undo the wrong setting"
without hand-editing `mold.db`. TOML-only keys are rejected with a
pointer at `mold config set` since those live in the hand-edited file.
`mold config reset --all` purges every DB row under the active profile
(prompts for confirmation unless `--yes` is passed).

### Multi-profile (schema v6)

`settings` and `model_prefs` rows are keyed on `(profile, key)` /
`(profile, model)` — one DB can host multiple independent preference
sets (`default`, `dev`, `portrait`, …). Active profile resolves in
priority order:

1. `MOLD_PROFILE` env var,
2. the `profile.active` setting row under the `default` profile,
3. `"default"`.

Every `mold config` subcommand accepts `--profile <name>` to scope for
a single invocation without touching the env or the meta setting.

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

[lambda]
# api_key = "..."             # Prefer LAMBDA_API_KEY for shells
# endpoint = "https://cloud.lambda.ai/api/v1"
# image_repository = "ghcr.io/utensils/mold"
# ssh_key_name = "mold-laptop"
# ssh_private_key_path = "~/.ssh/id_ed25519"
# filesystem_prefix = "mold"
# filesystem_mount_path = "/data/mold"
# confirm_hourly_usd = 5.0
# local_port = 7680
```

## Environment Variables

Environment variables take precedence over config file values.

### Core

| Variable             | Default                            | Description                                                                                                     |
| -------------------- | ---------------------------------- | --------------------------------------------------------------------------------------------------------------- |
| `MOLD_HOME`          | `~/.mold`                          | Base directory for config and cache                                                                             |
| `MOLD_DEFAULT_MODEL` | `flux2-klein:q8`                   | Default model name                                                                                              |
| `MOLD_HOST`          | `http://localhost:7680`            | Remote server URL                                                                                               |
| `MOLD_MODELS_DIR`    | `$MOLD_HOME/models`                | Model storage directory                                                                                         |
| `MOLD_PORT`          | `7680`                             | Server port                                                                                                     |
| `MOLD_MDNS`          | `1` (on)                           | Set `0`/`false` to stop `mold serve` advertising itself on the LAN via mDNS (requires the `mdns` build feature) |
| `LAMBDA_API_KEY`     | unset                              | Overrides `lambda.api_key`                                                                                      |
| `MOLD_LOG`           | `info` (serve) / `warn` (cli, tui) | Log level                                                                                                       |

### Generation

| Variable                              | Default                                 | Description                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                  |
| ------------------------------------- | --------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `MOLD_EAGER`                          | —                                       | `1` to keep all components loaded                                                                                                                                                                                                                                                                                                                                                                                                                                                                            |
| `MOLD_OFFLOAD`                        | —                                       | `1` to force block offload for FLUX, Flux.2, Z-Image, Qwen-Image, LTX-2, and SD3 BF16/FP8 paths where implemented. FLUX / Flux.2 / Z-Image / Qwen-Image keep fitting blocks GPU-resident and stream only overflow blocks; LTX-2 and SD3 full-stream.                                                                                                                                                                                                                                                         |
| `MOLD_OFFLOAD_PREFETCH`               | `on`                                    | FLUX offload async H2D prefetch stream — set `off` to revert to synchronous                                                                                                                                                                                                                                                                                                                                                                                                                                  |
| `MOLD_PINNED_VRAM_MAX_GB`             | RAM × 0.5                               | Cap on pinned host memory used by the FLUX offload path                                                                                                                                                                                                                                                                                                                                                                                                                                                      |
| `MOLD_RESERVE_VRAM_MB`                | 400 (Linux) / 600 (Windows) / 0 (macOS) | OS / cuBLAS workspace reserve subtracted from `free_vram_bytes` before any budget decision. Set explicitly to override the platform default; `0` disables                                                                                                                                                                                                                                                                                                                                                    |
| `MOLD_KEEP_TE_RAM`                    | —                                       | `1` to park text encoders on CPU between requests. FP16/BF16 only (GGUF falls through to drop+reload). Disabled on Metal (unified memory).                                                                                                                                                                                                                                                                                                                                                                   |
| `MOLD_LORA_BYPASS`                    | `auto`                                  | FLUX LoRA application path: `auto` enables bypass-mode when LoRAs are present (covers offload AND the GGUF/quantized path via `quantized_transformer.rs`), `on` always bypasses, `off` reverts to legacy merge-into-base / `gguf_lora_var_builder`                                                                                                                                                                                                                                                           |
| `MOLD_VAE_TILED`                      | `auto`                                  | Tiled VAE decode for FLUX/FLUX2/SDXL/SD3: `auto` retries with tiling on OOM, `force` always tiles, `off` disables                                                                                                                                                                                                                                                                                                                                                                                            |
| `MOLD_STEP_PREVIEW`                   | `1`                                     | Live denoise previews over `/api/generate/stream` (`preview` SSE events; FLUX.1/Flux.2/Z-Image): a small latent-resolution PNG per step via linear latent→RGB projection, throttled to ~700 ms. `0` disables.                                                                                                                                                                                                                                                                                                |
| `MOLD_LONG_PROMPTS`                   | —                                       | `1` enables ComfyUI-style chunked CLIP encoding (75-token windows, BOS/EOS framing, pooled outputs averaged into the FLUX `vector_in` 768-dim conditioning). Default off — pre-Tier-2 hard truncation at 77 preserved.                                                                                                                                                                                                                                                                                       |
| `MOLD_ATTN`                           | `math`                                  | Attention backend: `math` (default) or `flash` (needs `--features cuda,flash-attn` AND `RUSTFLAGS='--cfg mold_flash_attn_real'`; falls back to math with a one-shot warning otherwise)                                                                                                                                                                                                                                                                                                                       |
| `MOLD_ATTN_CHUNK`                     | auto                                    | Override math-attention query chunk size. Positive integers below the sequence length enable chunking; `0` or `off` disables it. The CUDA default chunks long queries at `512` to reduce peak VRAM.                                                                                                                                                                                                                                                                                                          |
| `MOLD_EMBED_METADATA`                 | `1`                                     | `0` to disable PNG metadata                                                                                                                                                                                                                                                                                                                                                                                                                                                                                  |
| `MOLD_MEDIA_ROOTS`                    | —                                       | Platform path-list of allow roots for trusted server-local LTX-2 `audio_file_path` / `source_video_path` requests. Targets are canonicalized and must resolve to files under one configured root.                                                                                                                                                                                                                                                                                                            |
| `MOLD_PREVIEW`                        | —                                       | `1` to display images inline in terminal                                                                                                                                                                                                                                                                                                                                                                                                                                                                     |
| `MOLD_T5_VARIANT`                     | `auto`                                  | T5 encoder: auto/fp16/q8/q6/q5/q4/q3                                                                                                                                                                                                                                                                                                                                                                                                                                                                         |
| `MOLD_QWEN3_VARIANT`                  | `auto`                                  | Qwen3 encoder: auto/bf16/q8/q6/iq4/q3                                                                                                                                                                                                                                                                                                                                                                                                                                                                        |
| `MOLD_SCHEDULER`                      | —                                       | SD1.5/SDXL: ddim/euler-ancestral/uni-pc                                                                                                                                                                                                                                                                                                                                                                                                                                                                      |
| `MOLD_CFG_PLUS`                       | —                                       | `1` to enable CFG++ (manifold-projection guidance, Chung et al. 2024). Drops usable CFG to ~1.5–2.5 and removes guidance artifacts. Per-request `--cfg-plus` overrides. Supported on SD3, SDXL, and SD1.5 (DDIM only — Euler-A / UniPC fall back). Ignored by FLUX / Z-Image / Flux.2 (distilled).                                                                                                                                                                                                           |
| `MOLD_VAE_DTYPE`                      | `auto`                                  | Override VAE precision: `auto`, `bf16`, `fp16`, `fp32`. Use `fp32` to fix banding artifacts on FLUX/SD3 finetuned VAEs (~2× decode VRAM; tiled VAE absorbs OOM via existing fallback). Wired into FLUX, FLUX2, SD3, SDXL, SD1.5.                                                                                                                                                                                                                                                                             |
| `MOLD_NVFP4_BACKEND`                  | `auto`                                  | NVFP4 backend selection for Flux.2 and LTX-2: `auto` and `portable` use portable CPU BF16 streaming dequant; `native` is reserved for validated sm_120/Blackwell tensor-core execution and fails clearly on non-Blackwell hosts.                                                                                                                                                                                                                                                                             |
| `MOLD_LTX2_GEMMA_DEVICE`              | `auto`                                  | LTX-2 Gemma 3 12B prompt encoder placement: `auto` (active GPU → sibling GPU → CPU on free VRAM, threshold 24 GB), `cpu` (force system RAM, slower encode but no VRAM contention — required on a single 24 GB card running cv:2752735), `gpu` (pin to active GPU, surface OOM rather than auto-offload). The deprecated `MOLD_LTX2_DEBUG_FORCE_CPU_PROMPT_ENCODER=1` is a one-shot-warn alias. Server preflight uses the same resolver so the load admits/rejects in lockstep with what the runtime will do. |
| `MOLD_LTX2_GEMMA_VARIANT`             | `auto`                                  | LTX-2 Gemma 3 12B weight format: `auto` (BF16 if both formats present, GGUF if only GGUF), `q4` (force Q4 GGUF — `google/gemma-3-12b-it-qat-q4_0-gguf`, ~7 GB; fits on a 24 GB card alongside the streaming transformer so auto-placement keeps Gemma GPU-resident), `bf16` (force BF16 split — `google/gemma-3-12b-it-qat-q4_0-unquantized`, ~23 GB; historical default). For V1, place the Q4 GGUF in your gemma_root manually — manifest auto-fetch is deferred to a follow-up.                           |
| `MOLD_LTX2_VAE_FORCE_FULL_DECODE`     | —                                       | `1` to disable adaptive temporal chunked LTX-2 VAE decode and force one full decode pass. Useful for debugging/comparison; long or high-resolution clips may OOM.                                                                                                                                                                                                                                                                                                                                            |
| `MOLD_LTX2_VAE_FORCE_FRAMEWISE`       | —                                       | `1` to force temporal-chunk LTX-2 VAE decode even when a full decode would fit. Reduces peak VRAM at a small decode-time cost.                                                                                                                                                                                                                                                                                                                                                                               |
| `MOLD_LTX2_VAE_DECODE_CHUNK_FRAMES`   | `4` latent frames                       | Positive integer number of latent frames per LTX-2 VAE decode chunk when chunked decode is active.                                                                                                                                                                                                                                                                                                                                                                                                           |
| `MOLD_LTX2_VAE_DECODE_CONTEXT_FRAMES` | auto                                    | Positive integer latent-frame overlap/context around each LTX-2 decode chunk. Default derives from the decoder causal-conv receptive field.                                                                                                                                                                                                                                                                                                                                                                  |

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

| Variable                      | Default             | Description                                                                                                                                                                              |
| ----------------------------- | ------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `MOLD_GPUS`                   | all visible         | Which GPUs the server uses: comma-separated ordinals (`0,1`) or `all`. See [Multi-GPU](/guide/cli-reference#multi-gpu)                                                                   |
| `MOLD_QUEUE_SIZE`             | `200`               | Max queued generation jobs; overflow returns HTTP 503 with `Retry-After`                                                                                                                 |
| `MOLD_OUTPUT_DIR`             | `~/.mold/output`    | Image output directory (set empty to disable)                                                                                                                                            |
| `MOLD_THUMBNAIL_WARMUP`       | —                   | `1` to prebuild gallery thumbnails at server startup (default: disabled)                                                                                                                 |
| `MOLD_WEB_DIR`                | —                   | Override the web gallery SPA bundle location. First resolved path among this, `$XDG_DATA_HOME/mold/web`, `~/.mold/web`, `<binary dir>/web`, and `./web/dist` wins                        |
| `MOLD_DB_PATH`                | `MOLD_HOME/mold.db` | Override the SQLite gallery metadata DB location                                                                                                                                         |
| `MOLD_DB_DISABLE`             | —                   | `1` to disable the SQLite metadata DB entirely — server and CLI fall back to filesystem walks                                                                                            |
| `MOLD_CORS_ORIGIN`            | —                   | Restrict CORS to specific origin                                                                                                                                                         |
| `MOLD_API_KEY`                | —                   | API key for authentication (single key, comma-separated, or `@/path/to/keys.txt`)                                                                                                        |
| `MOLD_RATE_LIMIT`             | —                   | Per-IP rate limit for generation endpoints (e.g., `10/min`, `5/sec`, `100/hour`)                                                                                                         |
| `MOLD_RATE_LIMIT_BURST`       | —                   | Burst allowance override (defaults to 2x rate, capped at 100)                                                                                                                            |
| `MOLD_MAX_CACHED_MODELS`      | `3`                 | LRU model-cache capacity (range `1..=16`). At most one entry stays GPU-resident; the rest are parked in CPU RAM. Out-of-range values warn and fall back to default.                      |
| `MOLD_CACHE_IDLE_TTL_SECS`    | `1800` (30 min)     | Idle timeout for parked cache entries (range `60..=86400`). Untouched entries are evicted past this TTL.                                                                                 |
| `MOLD_QUEUE_LOOKAHEAD_BUFFER` | `8`                 | Server queue lookahead size (range `1..=64`). The dispatcher peeks this many jobs ahead to honour locality.                                                                              |
| `MOLD_QUEUE_MAX_DEFERRALS`    | `3`                 | Per-job starvation budget (range `0..=32`). A job can be deferred this many times before forced pickup.                                                                                  |
| `MOLD_MALLOC_TRIM`            | `1` (Linux/glibc)   | `0` disables the post-generation `malloc_trim(0)` call. Cheap (~ms) but Linux-only; reclaims arena pages after large GGUF+LoRA rebuilds.                                                 |
| `MOLD_FLUX_DELTA_CACHE`       | `1`                 | `0` disables the CPU-side FLUX LoRA delta cache (~25 GB host RAM on typical FLUX LoRAs). Disabling forces a sub-second `B@A·scale` recompute on each rebuild.                            |
| `MOLD_FLUX_KEEP_TRANSFORMER`  | `0`                 | `1` keeps the FLUX transformer GPU-resident across same-LoRA generations (saves a full GGUF+LoRA rebuild). Server force-drops it if VAE decode headroom is too tight at that resolution. |

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

The DB also stores the full generation metadata JSON for rows written by
current versions, so gallery clients can recreate outputs with advanced
options such as LoRA stacks, ControlNet settings, CFG++, output format,
and LTX-2 audio/video pipeline controls.

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
| `MOLD_PLACE_TRANSFORMER`   | FLUX, Flux.2, Z-Image, Qwen-Image | Per-component override. Interacts with `MOLD_OFFLOAD` — resident and streamed blocks target the chosen ordinal.                                     |
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

- CUDA `auto` prefers BF16 when enough text-encoder headroom remains, and
  falls back to quantized GGUF variants for local sequential, resident, and
  edit-conditioning paths when BF16 would be too heavy.
- Metal/MPS `auto` prefers the quantized Qwen2.5-VL GGUF encoder path to reduce
  memory pressure during prompt encoding.
- `qwen-image-edit` still loads the Qwen2.5-VL vision tower for image
  conditioning, but quantized `MOLD_QWEN2_VARIANT` values keep the language side
  smaller and stage the vision weights only when needed.

### Debug and Family-Specific Knobs

| Variable                                              | Default                    | Description                                                                                                           |
| ----------------------------------------------------- | -------------------------- | --------------------------------------------------------------------------------------------------------------------- |
| `MOLD_SD3_DEBUG`                                      | —                          | Enable verbose SD3.5 pipeline logging                                                                                 |
| `MOLD_QWEN_DEBUG`                                     | —                          | Enable verbose Qwen-Image pipeline logging                                                                            |
| `MOLD_ZIMAGE_DEBUG`                                   | —                          | Enable verbose Z-Image pipeline logging                                                                               |
| `MOLD_LTX_DEBUG`                                      | —                          | Enable verbose LTX Video / LTX-2 pipeline logging                                                                     |
| `MOLD_LTX_DEBUG_FILE`                                 | `/tmp/mold-ltx2-debug.log` | Append LTX Video / LTX-2 debug output to a file                                                                       |
| `MOLD_LTX_DEBUG_COMPARE_UNCOND`                       | —                          | Log conditional vs unconditional LTX-2 prompt-context comparisons                                                     |
| `MOLD_LTX_DEBUG_ALT_PROMPT`                           | —                          | Use an alternate prompt string for LTX-2 prompt-sensitivity debugging                                                 |
| `MOLD_LTX_DEBUG_DISABLE_AUDIO_BRANCH`                 | —                          | Debug-only LTX-2 switch to disable the audio branch during native runs                                                |
| `MOLD_LTX_DEBUG_DISABLE_CROSS_ATTENTION_ADALN`        | —                          | Debug-only LTX-2 switch to bypass cross-attention AdaLN modulation                                                    |
| `MOLD_LTX2_DEBUG_DISABLE_TRANSFORMER_GATED_ATTENTION` | —                          | Debug-only LTX-2 switch to bypass transformer gated attention                                                         |
| `MOLD_LTX2_DEBUG_FORCE_CPU_PROMPT_ENCODER`            | —                          | Deprecated alias for `MOLD_LTX2_GEMMA_DEVICE=cpu`. Emits a one-shot warn at runtime; remove in favor of the new knob. |
| `MOLD_LTX2_DEBUG_TIMINGS`                             | —                          | Emit native LTX-2 pipeline, phase, and denoise timing summaries for optimization work                                 |
| `MOLD_LTX2_DEBUG_STAGE_PREFIX`                        | —                          | Write decoded native LTX-2 stage artifacts using this filename prefix                                                 |
| `MOLD_LTX2_DEBUG_BLOCKS`                              | —                          | Emit per-block native LTX-2 transformer debug logs                                                                    |
| `MOLD_LTX2_DEBUG_BLOCK_DETAIL`                        | —                          | Restrict detailed native LTX-2 block logging to a specific transformer block index                                    |
| `MOLD_LTX2_DEBUG_LOAD_BLOCKS`                         | —                          | Log native LTX-2 transformer block loading details                                                                    |
| `MOLD_LTX2_FORCE_EAGER`                               | —                          | Force eager native LTX-2 transformer loading instead of layer streaming                                               |
| `MOLD_LTX2_FORCE_STREAMING`                           | —                          | Force native LTX-2 transformer layer streaming                                                                        |
| `MOLD_LTX2_FP8_INPUT_SCALE_MODE`                      | `skip`                     | Debug override for native LTX-2 FP8 input-scale handling (`skip`, `emulate`, `divide`, `multiply`)                    |
| `MOLD_LTX2_FP8_WEIGHT_SCALE_MODE`                     | `apply`                    | Debug override for native LTX-2 FP8 checkpoint weight-scale handling (`apply`, `skip`, `scaled-mm`)                   |
| `MOLD_WUERSTCHEN_DEBUG`                               | —                          | Enable verbose Wuerstchen pipeline logging                                                                            |
| `MOLD_WUERSTCHEN_DECODER_GUIDANCE`                    | `0.0`                      | Override decoder-stage CFG guidance for Wuerstchen                                                                    |

These are intended for troubleshooting and development rather than normal use.

### Build-Time Metadata

| Variable            | Default | Description                                                 |
| ------------------- | ------- | ----------------------------------------------------------- |
| `MOLD_FULL_VERSION` | —       | Internal build-time version string embedded into CLI output |

This variable is set during the build and is not normally configured by users at
runtime.
