# Configuration

mold looks for `config.toml` inside the base mold directory (`~/.mold/` by
default, or `~/.config/mold/` via XDG).

## Config File

```toml
default_model = "flux-schnell:q8"
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
```

## Environment Variables

Environment variables take precedence over config file values.

### Core

| Variable             | Default                 | Description                         |
| -------------------- | ----------------------- | ----------------------------------- |
| `MOLD_HOME`          | `~/.mold`               | Base directory for config and cache |
| `MOLD_DEFAULT_MODEL` | `flux-schnell`          | Default model name                  |
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

| Variable           | Default | Description                          |
| ------------------ | ------- | ------------------------------------ |
| `MOLD_OUTPUT_DIR`  | —       | Save server-generated images to disk |
| `MOLD_CORS_ORIGIN` | —       | Restrict CORS to specific origin     |

### Auth

| Variable   | Default | Description                        |
| ---------- | ------- | ---------------------------------- |
| `HF_TOKEN` | —       | HuggingFace token for gated models |
