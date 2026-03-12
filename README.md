# mold

> Like ollama, but for diffusion models.

mold is a CLI/TUI tool for AI image generation using FLUX models, powered by [candle](https://github.com/huggingface/candle).

## Quick Start

```bash
# Generate an image
mold generate "a cat sitting on a cloud"

# Use a specific model
mold generate -m flux-dev "cyberpunk cityscape" --steps 25

# Start the inference server (on GPU host)
mold serve

# Generate remotely
MOLD_HOST=http://gpu-host:7680 mold generate "a sunset over mountains"

# Interactive TUI
mold run
```

## Commands

| Command | Description |
|---------|-------------|
| `mold generate <PROMPT>` | Generate images from text |
| `mold serve` | Start inference server |
| `mold pull <MODEL>` | Download model from HuggingFace |
| `mold list` | List available models |
| `mold ps` | Show server status |
| `mold run [MODEL]` | Interactive TUI session |

## Available Models

| Model | Steps | Description |
|-------|-------|-------------|
| `flux-schnell` | 4 | Fast generation (default) |
| `flux-dev` | 25 | Higher quality |

## Configuration

Config file: `~/.mold/config.toml`

```toml
default_model = "flux-schnell"
models_dir = "~/.mold/models"
server_port = 7680
```

Environment variables: `MOLD_HOST`, `MOLD_MODELS_DIR`, `MOLD_PORT`, `MOLD_LOG`

## Building

```bash
cargo build --release
```

## License

MIT
