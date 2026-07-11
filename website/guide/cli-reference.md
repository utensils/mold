# CLI Reference

## `mold run`

Generate images or video from prompts.

```bash
mold run [MODEL] [PROMPT...] [OPTIONS]
```

The first positional argument is treated as the model only when it resolves to a
known model name. Otherwise it becomes part of the prompt. Prompt text can also
come from stdin.

### Options

| Flag                                                                                         | Description                                                                                        |
| -------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------- |
| `-o, --output <PATH>`                                                                        | Output path; `-` writes media bytes to stdout                                                      |
| `--format <FMT>`                                                                             | `png`, `jpeg`, `gif`, `apng`, `webp`, or `mp4`                                                     |
| `--width <N>`, `--height <N>`                                                                | Output dimensions                                                                                  |
| `--steps <N>`, `--guidance <N>`, `--seed <N>`, `--batch <N>`                                 | Core generation controls                                                                           |
| `--prompt <TEXT>`                                                                            | Repeat for multi-stage LTX-2 chain sugar                                                           |
| `--frames-per-clip <N>`                                                                      | Per-stage frame count for repeated `--prompt`                                                      |
| `--script <PATH>`                                                                            | Submit a `mold.chain.v1` TOML chain script                                                         |
| `--dry-run`                                                                                  | Parse/normalise repeated prompts or scripts without generating                                     |
| `--frames <N>`, `--fps <N>`                                                                  | Video frame count and output FPS                                                                   |
| `--clip-frames <N>`                                                                          | Per-clip cap for chained LTX-2 renders                                                             |
| `--motion-tail <N>`                                                                          | Overlap frames reused between chained clips                                                        |
| `--audio`, `--no-audio`                                                                      | Keep or strip synchronized LTX-2 MP4 audio                                                         |
| `--audio-file <PATH>`                                                                        | LTX-2 audio-to-video conditioning                                                                  |
| `--video <PATH>`                                                                             | LTX-2 source video for retake/video-conditioning                                                   |
| `--keyframe <FRAME:PATH>`                                                                    | Repeatable LTX-2 keyframe conditioning                                                             |
| `--pipeline <MODE>`                                                                          | `one-stage`, `two-stage`, `two-stage-hq`, `distilled`, `ic-lora`, `keyframe`, `a2vid`, or `retake` |
| `--retake <START:END>`                                                                       | LTX-2 retake range in seconds                                                                      |
| `--camera-control <NAME\|PATH>`                                                              | LTX-2 camera-control preset or `.safetensors` path                                                 |
| `--spatial-upscale <MODE>`                                                                   | LTX-2 spatial upscaling, such as `x1.5` or `x2`                                                    |
| `--temporal-upscale <MODE>`                                                                  | LTX-2 temporal upscaling, currently `x2`                                                           |
| `-i, --image <PATH>`                                                                         | Source image; repeat for `qwen-image-edit`; `-` is stdin for single-image families                 |
| `--strength <FLOAT>`, `--mask <PATH>`                                                        | img2img/inpainting controls                                                                        |
| `--control <PATH>`, `--control-model <NAME>`, `--control-scale <FLOAT>`                      | SD1.5 ControlNet controls                                                                          |
| `-n, --negative-prompt <TEXT>`, `--no-negative`                                              | CFG-family negative prompt controls                                                                |
| `--lora <PATH>`, `--lora-scale <FLOAT>`                                                      | LoRA adapter path and scale; `--lora` is repeatable                                                |
| `--upscale <MODEL>`                                                                          | Apply a Real-ESRGAN upscaler after generation                                                      |
| `--no-metadata`                                                                              | Disable embedded PNG metadata for this run                                                         |
| `--preview`                                                                                  | Display output inline in the terminal                                                              |
| `--expand`, `--no-expand`, `--expand-backend <URL>`, `--expand-model <MODEL>`                | Prompt expansion controls                                                                          |
| `--local`                                                                                    | Skip the server and run local inference                                                            |
| `--host <URL>`                                                                               | Override `MOLD_HOST`                                                                               |
| `--gpus <SPEC>`                                                                              | Local GPU ordinals (`0,1`) or `all`                                                                |
| `--eager`, `--offload`                                                                       | VRAM/performance placement modes                                                                   |
| `--t5-variant <TAG>`, `--qwen3-variant <TAG>`, `--qwen2-variant <TAG>`                       | Text encoder variant overrides                                                                     |
| `--qwen2-text-encoder-mode <MODE>`                                                           | `auto`, `gpu`, `cpu-stage`, or `cpu`                                                               |
| `--scheduler <SCHED>`                                                                        | `ddim`, `euler-ancestral`, or `uni-pc`                                                             |
| `--cfg-plus`                                                                                 | Enable CFG++ on supported SD-family paths                                                          |
| `--device-text-encoders <DEV>`                                                               | Place all text encoders on `auto`, `cpu`, `gpu`, or `gpu:N`                                        |
| `--device-transformer <DEV>`, `--device-vae <DEV>`                                           | Advanced family placement overrides                                                                |
| `--device-t5 <DEV>`, `--device-clip-l <DEV>`, `--device-clip-g <DEV>`, `--device-qwen <DEV>` | Per-encoder placement overrides                                                                    |

### Qwen Family Encoder Controls

- `--qwen2-variant auto|bf16|q8|q6|q5|q4|q3|q2`
- `--qwen2-text-encoder-mode auto|gpu|cpu-stage|cpu`

`qwen-image-edit-2511:*` treats repeated `--image` flags as ordered
`edit_images`; non-edit families accept at most one source image.

### LTX-2 Notes

LTX-2 defaults to MP4, supports synchronized audio, and runs real generation on
CUDA. CPU is correctness-only and Metal is unsupported for this family. Chaining
works through repeated `--prompt`, `--script`, or large `--frames` requests.

## `mold chain validate`

Validate and normalise a `mold.chain.v1` TOML script.

```bash
mold chain validate shot.toml
mold run --script shot.toml --dry-run
```

## `mold jobs`

Inspect and control durable chain jobs on a running `mold serve` instance. The
commands use `MOLD_HOST` and send `MOLD_API_KEY` when configured.

```bash
mold jobs list [--json]
mold jobs show <id> [--json]
mold jobs resume <id>
mold jobs retake <id> --stage <N> [--mode cascade|splice] [--seed-offset <U64>] [--prompt <TEXT>]
mold jobs cancel <id>
mold jobs delete <id> [--yes]
mold jobs gc
```

Durable chain jobs store checkpoints under `MOLD_HOME/jobs/<job_id>`.
`mold jobs gc` mirrors `POST /api/chain-jobs/gc`, pruning successful ephemeral
shim jobs and completed non-ephemeral job artifacts older than
`chain.jobs_artifact_ttl_days`.

## `mold expand`

Preview prompt expansion without generating.

```bash
mold expand <PROMPT> [OPTIONS]
```

| Flag                     | Description                    |
| ------------------------ | ------------------------------ |
| `-m, --model <MODEL>`    | Target model for style/context |
| `--variations <N>`       | Number of variations           |
| `--json`                 | Output as JSON array           |
| `--backend <URL>`        | Expansion backend override     |
| `--expand-model <MODEL>` | LLM model override             |

## `mold serve`

Start the HTTP inference server.

```bash
mold serve [--port N] [--bind ADDR] [--models-dir PATH] [--gpus SPEC] [--queue-size N] [--log-format json|text] [--log-file] [--discord] [--no-mdns]
```

| Flag                  | Description                                                                |
| --------------------- | -------------------------------------------------------------------------- |
| `--port <N>`          | Port, defaults to `7680` or `MOLD_PORT`                                    |
| `--bind <ADDR>`       | Bind address, defaults to `0.0.0.0`                                        |
| `--models-dir <PATH>` | Override the models directory                                              |
| `--gpus <SPEC>`       | GPU ordinals (`0,1`) or `all`; defaults to every visible GPU               |
| `--queue-size <N>`    | Max queued jobs; overflow returns HTTP 503 + `Retry-After`                 |
| `--log-format <FMT>`  | `json` or `text`                                                           |
| `--log-file`          | Enable rotated logs under `~/.mold/logs/`                                  |
| `--discord`           | Start the built-in Discord bot in the same process                         |
| `--no-mdns`           | Don't advertise this server on the LAN (`mdns` builds; also `MOLD_MDNS=0`) |

`GET /api/status` returns `gpus[]` with per-worker state and
`queue_depth`/`queue_capacity` for queue health.

## `mold server discover`

Browse the local network (mDNS/DNS-SD, `_mold._tcp`) for running `mold serve`
instances that advertise themselves. Available in builds compiled with the
`mdns` feature (included in release binaries and the Nix package).

```bash
mold server discover [--timeout-secs N] [--json] [--probe]
```

| Flag                 | Description                                                   |
| -------------------- | ------------------------------------------------------------- |
| `--timeout-secs <N>` | How long to browse before reporting (default `3`)             |
| `--json`             | Emit the raw list of discovered servers as JSON               |
| `--probe`            | Also time each server's `/health` and show a `LATENCY` column |

The table lists NAME, URL, VERSION, AUTH (whether an API key is required), and a
GPU summary, followed by a `export MOLD_HOST=…` hint for the first result.
Advertising is on by default when a server is built with the `mdns` feature;
disable it per-server with `mold serve --no-mdns` or `MOLD_MDNS=0`.

## `mold mcp`

Start a stdio Model Context Protocol server that proxies to `mold serve`.

```bash
mold mcp [--host URL]
```

MCP exposes generation, async generation, gallery lookup, installed LoRA
listing, model listing, and server status. It intentionally proxies the server
surface instead of embedding local inference.

## `mold pull`, `mold list`, `mold info`

```bash
mold pull flux-schnell:q8
mold list
mold info
mold info flux-dev:q4
mold info flux-dev:q4 --verify
```

`mold pull` downloads manifest models locally or through the reachable server.
`mold info <model> --verify` verifies checksums for that model.

## `mold config`

View and edit configuration settings.

```bash
mold config list [--json]
mold config get <KEY> [--raw]
mold config set <KEY> <VALUE>
mold config reset <KEY>
mold config reset --all
mold config where <KEY>
mold config path
mold config edit
```

| Section   | Keys                                                                                                                                                                                                                               |
| --------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| General   | `default_model`, `models_dir`, `output_dir`, `server_port`, `default_width`, `default_height`, `default_steps`, `embed_metadata`, `t5_variant`, `qwen3_variant`, `default_negative_prompt`                                         |
| Expand    | `expand.enabled`, `expand.backend`, `expand.model`, `expand.api_model`, `expand.temperature`, `expand.top_p`, `expand.max_tokens`, `expand.thinking`                                                                               |
| Logging   | `logging.level`, `logging.file`, `logging.dir`, `logging.max_days`                                                                                                                                                                 |
| RunPod    | `runpod.api_key`, `runpod.default_gpu`, `runpod.default_datacenter`, `runpod.default_network_volume_id`, `runpod.auto_teardown`, `runpod.auto_teardown_idle_mins`, `runpod.cost_alert_usd`, `runpod.endpoint`                      |
| Lambda    | `lambda.api_key`, `lambda.endpoint`, `lambda.image_repository`, `lambda.ssh_key_name`, `lambda.ssh_private_key_path`, `lambda.filesystem_prefix`, `lambda.filesystem_mount_path`, `lambda.confirm_hourly_usd`, `lambda.local_port` |
| Per-model | `models.<name>.<field>` where field is one of `default_steps`, `default_guidance`, `default_width`, `default_height`, `scheduler`, `negative_prompt`, `lora`, `lora_scale`                                                         |

`config.toml` owns bootstrap paths, ports, credentials, logging, and model path
overrides. The SQLite settings DB owns user preferences and per-model
generation defaults.

## `mold tui`

Launch the terminal UI.

```bash
mold tui [--host URL] [--local]
```

See [Terminal UI](/guide/tui) for views, keybindings, script mode, and
settings persistence.

## `mold discord`

Start the Discord bot, or run it in-process with `mold serve --discord`.

```bash
mold discord
```

The Discord bot exposes a smaller slash-command surface for generation,
expansion, model listing, and status. Advanced catalog, placement, and script
authoring flows remain in the web UI/API. See [Discord Bot](/api/discord).

## `mold upscale`

Upscale an existing image with Real-ESRGAN.

```bash
mold upscale photo.png
mold upscale photo.png -m real-esrgan-x4plus:fp16 -o photo_4x.png
mold upscale - < input.png > output.png
mold run "a cat" | mold upscale -
```

| Flag                  | Description                    |
| --------------------- | ------------------------------ |
| `-m, --model <NAME>`  | Upscaler model                 |
| `-o, --output <PATH>` | Output path                    |
| `--format <FMT>`      | `png` or `jpeg`                |
| `--tile-size <N>`     | Tile size; `0` disables tiling |
| `--host <URL>`        | Override `MOLD_HOST`           |
| `--local`             | Skip server and run locally    |
| `--preview`           | Display output inline          |

## `mold runpod`

Manage RunPod pods or generate on a fresh pod end-to-end.

```bash
mold config set runpod.api_key <key>
mold runpod doctor
mold runpod run "a cat on a skateboard"
mold runpod create --gpu 5090
mold runpod network-volume create --name models --size 100 --dc US-KS-2
mold runpod run "a cat" --network-volume <volume-id>
mold runpod connect <pod-id>
mold runpod delete <pod-id>
```

Common subcommands are `doctor`, `gpus`, `datacenters`, `list`, `get`,
`create`, `start`, `stop`, `delete`, `connect`, `logs`, `usage`, and `run`.
See [mold runpod CLI](/deployment/runpod-cli).

## `mold lambda`

Deploy and manage private mold servers on Lambda Cloud.

```bash
mold config set lambda.api_key <key>
mold lambda doctor
mold lambda availability
mold lambda deploy --instance-type gpu_1x_a10 --region us-west-1
mold lambda tunnel
mold lambda terminate
```

Common subcommands are `doctor`, `availability`, `deploy`, `status`, `logs`,
`tunnel`, `ssh`, `filesystems`, `terminate`, and `reset`. See
[mold lambda CLI](/deployment/lambda-cli).

## Other Commands

| Command                                           | Purpose                                                         |
| ------------------------------------------------- | --------------------------------------------------------------- |
| `mold default [MODEL]`                            | Get or set the default model                                    |
| `mold stats [--json]`                             | Show disk usage for models, output, logs, and shared components |
| `mold clean [--force] [--older-than DURATION]`    | Remove stale downloads, orphaned files, and old outputs         |
| `mold server start/status/stop`                   | Manage a background server daemon                               |
| `mold server discover`                            | Find mold servers advertised on the local network (mDNS)        |
| `mold rm <MODELS...> [--force]`                   | Remove downloaded models                                        |
| `mold ps`                                         | Show server status or local mold processes                      |
| `mold unload`                                     | Unload the current server model                                 |
| `mold update [--check] [--force] [--version TAG]` | Update a release binary                                         |
| `mold version`                                    | Show version, build date, and git SHA                           |

## `mold completions`

Generate shell completions.

```bash
mold completions zsh
mold completions bash
mold completions fish
mold completions elvish
mold completions powershell
```

Common setup:

```bash
source <(mold completions zsh)
source <(mold completions bash)
mold completions fish > ~/.config/fish/completions/mold.fish
```
