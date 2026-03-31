# CLI Reference

## `mold run`

Generate images from text prompts.

```
mold run [MODEL] [PROMPT...] [OPTIONS]
```

The first positional argument is treated as MODEL if it matches a known model
name; otherwise it is the prompt. Prompt can also be piped via stdin.

### Options

| Flag                      | Description                                      |
| ------------------------- | ------------------------------------------------ |
| `-m, --model <MODEL>`     | Explicit model override                          |
| `-o, --output <PATH>`     | Output file (default: `./mold-{model}-{ts}.png`) |
| `--width <N>`             | Image width                                      |
| `--height <N>`            | Image height                                     |
| `--steps <N>`             | Inference steps                                  |
| `--seed <N>`              | Random seed                                      |
| `--batch <N>`             | Number of images (1–16)                          |
| `--guidance <N>`          | Guidance scale                                   |
| `--format <FMT>`          | `png` or `jpeg`                                  |
| `--local`                 | Skip server, run locally                         |
| `--eager`                 | Keep all components loaded (more VRAM)           |
| `--offload`               | CPU↔GPU block streaming (less VRAM)              |
| `--lora <PATH>`           | LoRA adapter safetensors                         |
| `--lora-scale <FLOAT>`    | LoRA strength (0.0–2.0)                          |
| `-i, --image <PATH>`      | Source image for img2img (`-` for stdin)         |
| `--strength <FLOAT>`      | Denoising strength (0.0–1.0)                     |
| `--mask <PATH>`           | Inpainting mask                                  |
| `--control <PATH>`        | ControlNet control image                         |
| `--control-model <NAME>`  | ControlNet model name                            |
| `--control-scale <FLOAT>` | ControlNet scale (0.0–2.0)                       |
| `-n, --negative-prompt`   | Negative prompt (CFG models)                     |
| `--no-negative`           | Suppress config default negative                 |
| `--no-metadata`           | Disable PNG metadata                             |
| `--preview`               | Display image inline in terminal                 |
| `--expand`                | Enable prompt expansion                          |
| `--no-expand`             | Disable prompt expansion                         |
| `--expand-backend <URL>`  | Expansion backend URL                            |
| `--expand-model <MODEL>`  | LLM model for expansion                          |
| `--t5-variant <TAG>`      | T5 encoder variant                               |
| `--qwen3-variant <TAG>`   | Qwen3 encoder variant                            |
| `--scheduler <SCHED>`     | Noise scheduler (ddim, euler-ancestral, uni-pc)  |
| `--host <URL>`            | Override MOLD_HOST                               |

## `mold expand`

Preview prompt expansion without generating.

```bash
mold expand <PROMPT> [OPTIONS]
```

| Flag                     | Description                |
| ------------------------ | -------------------------- |
| `-m, --model <MODEL>`    | Target model (for style)   |
| `--variations <N>`       | Number of variations       |
| `--json`                 | Output as JSON array       |
| `--backend <URL>`        | Expansion backend override |
| `--expand-model <MODEL>` | LLM model override         |

## `mold serve`

Start the HTTP inference server.

```bash
mold serve [--port N] [--bind ADDR] [--models-dir PATH] [--discord]
```

## `mold pull`

Download a model from HuggingFace.

```bash
mold pull <MODEL> [--skip-verify]
```

## `mold list`

List configured and available models with download status and disk usage.

## `mold info`

Show installation overview, or model details with optional SHA-256 verification.

```bash
mold info              # overview
mold info flux-dev:q4  # model details
mold info --verify     # verify all checksums
```

## `mold default`

Get or set the default model.

```bash
mold default              # show current
mold default flux-dev:q4  # set default
```

## `mold rm`

Remove downloaded models.

```bash
mold rm <MODELS...> [--force]
```

## `mold ps`

Show server status and loaded model.

## `mold unload`

Unload the current model from the server to free GPU memory.

## `mold version`

Show version, build date, and git SHA.

## `mold completions`

Generate shell completions.

```bash
mold completions bash | zsh | fish
```
