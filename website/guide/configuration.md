# Configuration

mold looks for `config.toml` inside the base mold directory (`~/.mold/` by
default, or override with `MOLD_HOME`).

## Managing Config from the CLI

Use `mold config` to view and edit settings without manually editing TOML:

```bash
mold config list                    # Show all settings
mold config get server_port         # Get a value
mold config set server_port 8080    # Set a value
mold config edit                    # Open in $EDITOR
```

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

| Variable                | Default          | Description                                                                       |
| ----------------------- | ---------------- | --------------------------------------------------------------------------------- |
| `MOLD_OUTPUT_DIR`       | `~/.mold/output` | Image output directory (set empty to disable)                                     |
| `MOLD_CORS_ORIGIN`      | —                | Restrict CORS to specific origin                                                  |
| `MOLD_API_KEY`          | —                | API key for authentication (single key, comma-separated, or `@/path/to/keys.txt`) |
| `MOLD_RATE_LIMIT`       | —                | Per-IP rate limit for generation endpoints (e.g., `10/min`, `5/sec`, `100/hour`)  |
| `MOLD_RATE_LIMIT_BURST` | —                | Burst allowance override (defaults to 2x rate, capped at 100)                     |

### Auth

| Variable   | Default | Description                        |
| ---------- | ------- | ---------------------------------- |
| `HF_TOKEN` | —       | HuggingFace token for gated models |

## Advanced

### Device and Path Overrides

| Variable                       | Default | Description                                                                       |
| ------------------------------ | ------- | --------------------------------------------------------------------------------- |
| `MOLD_DEVICE`                  | —       | Force device placement, currently `cpu` for debugging                             |
| `MOLD_TRANSFORMER_PATH`        | —       | Override transformer weights path                                                 |
| `MOLD_VAE_PATH`                | —       | Override VAE weights path                                                         |
| `MOLD_SPATIAL_UPSCALER_PATH`   | —       | Override LTX spatial upscaler path                                                |
| `MOLD_T5_PATH`                 | —       | Override T5 encoder path                                                          |
| `MOLD_CLIP_PATH`               | —       | Override CLIP-L encoder path                                                      |
| `MOLD_CLIP2_PATH`              | —       | Override CLIP-G encoder path for SDXL                                             |
| `MOLD_T5_TOKENIZER_PATH`       | —       | Override T5 tokenizer path                                                        |
| `MOLD_CLIP_TOKENIZER_PATH`     | —       | Override CLIP-L tokenizer path                                                    |
| `MOLD_CLIP2_TOKENIZER_PATH`    | —       | Override CLIP-G tokenizer path for SDXL                                           |
| `MOLD_TEXT_TOKENIZER_PATH`     | —       | Override generic text tokenizer path for Qwen/Z-Image                             |
| `MOLD_DECODER_PATH`            | —       | Override Wuerstchen decoder weights path                                          |
| `MOLD_QWEN2_VARIANT`           | `auto`  | Qwen-Image Qwen2.5-VL encoder: `auto`, `bf16`, `q8`, `q6`, `q5`, `q4`, `q3`, `q2` |
| `MOLD_QWEN2_TEXT_ENCODER_MODE` | `auto`  | Qwen-Image placement mode: `auto`, `gpu`, `cpu-stage`, `cpu`                      |

These are mainly useful for custom local model layouts, manual debugging, or
testing alternative weight files without editing `config.toml`.

For Qwen-Image specifically:

- CUDA `auto` keeps the existing BF16-first behavior unless you explicitly set
  `MOLD_QWEN2_VARIANT`.
- Metal/MPS `auto` prefers the quantized Qwen2.5-VL GGUF encoder path to reduce
  memory pressure during prompt encoding.

### Debug and Family-Specific Knobs

| Variable                           | Default | Description                                        |
| ---------------------------------- | ------- | -------------------------------------------------- |
| `MOLD_SD3_DEBUG`                   | —       | Enable verbose SD3.5 pipeline logging              |
| `MOLD_QWEN_DEBUG`                  | —       | Enable verbose Qwen-Image pipeline logging         |
| `MOLD_LTX_DEBUG`                   | —       | Enable verbose LTX Video pipeline logging          |
| `MOLD_WUERSTCHEN_DEBUG`            | —       | Enable verbose Wuerstchen pipeline logging         |
| `MOLD_WUERSTCHEN_DECODER_GUIDANCE` | `0.0`   | Override decoder-stage CFG guidance for Wuerstchen |

These are intended for troubleshooting and development rather than normal use.

### Build-Time Metadata

| Variable            | Default | Description                                                 |
| ------------------- | ------- | ----------------------------------------------------------- |
| `MOLD_FULL_VERSION` | —       | Internal build-time version string embedded into CLI output |

This variable is set during the build and is not normally configured by users at
runtime.
