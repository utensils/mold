# CLI Reference

The CLI is Mold's native interface and the contract from which its richer
clients grow. Commands are designed for direct use, shell composition, scripts,
CI jobs, and agent tool calls, with pipe-friendly media I/O and machine-readable
forms where automation needs them.

## `mold run`

Generate images or video from prompts.

```bash
mold run [MODEL] [PROMPT...] [OPTIONS]
```

The first positional argument is treated as the model only when it resolves to a
known model name. Otherwise it becomes part of the prompt. Prompt text can also
come from stdin.

`PROMPT` is required, with one exception: an LTX-2 or LTX-Video run that already
carries visual conditioning (`--image`, `--keyframe`, `--video`, or `--extend`)
may be left unprompted, so `mold run ltx-2-19b-distilled:fp8 --image still.png
--frames 97` is a complete command. It buys no VRAM and usually renders
near-static motion; see
[the LTX-2 page](/models/ltx2#the-prompt-is-optional-for-image-to-video). Every
other run, including img2img on an image family, still errors with
`no prompt provided`. An empty prompt also skips prompt expansion for that run.

### Options

| Flag                                                                                         | Description                                                                                                                           |
| -------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------- |
| `-o, --output <PATH>`                                                                        | Output path; `-` writes media bytes to stdout                                                                                         |
| `--format <FMT>`                                                                             | `png`, `jpeg`/`jpg`, `gif`, `apng`, `webp`, `mp4`, or `wav` (LTX-2 `--pipeline t2a`)                                                  |
| `--width <N>`, `--height <N>`                                                                | Output dimensions                                                                                                                     |
| `--steps <N>`, `--guidance <N>`, `--seed <N>`, `--batch <N>`                                 | Core generation controls                                                                                                              |
| `--prompt <TEXT>`                                                                            | Repeat for multi-stage video chain sugar (LTX-2, LTX-Video, Wan)                                                                      |
| `--frames-per-clip <N>`                                                                      | Per-stage frame count for repeated `--prompt`                                                                                         |
| `--script <PATH>`                                                                            | Submit a `mold.chain.v1` TOML chain script                                                                                            |
| `--dry-run`                                                                                  | Parse/normalise repeated prompts or scripts without generating                                                                        |
| `--frames <N>`, `--fps <N>`                                                                  | Video frame count and output FPS                                                                                                      |
| `--duration <SECONDS>`                                                                       | MiniMax H3 duration from 4–15 seconds; resolves to its exact `17n+5` frame grid at 24 fps                                             |
| `--predict-duration`                                                                         | Let a qualified LTX-2.5 model choose a clip length from 1–20 seconds                                                                  |
| `--clip-frames <N>`                                                                          | Per-clip cap for chained video renders                                                                                                |
| `--motion-tail <N>`                                                                          | Overlap frames reused between chained clips                                                                                           |
| `--extend <PATH>`                                                                            | Continue an existing video clip (LTX-2 and image-conditioned Wan); mutually exclusive with `--video`/`--image`/`--keyframe`           |
| `--extend-overlap <N>`                                                                       | Source-tail frames reused as motion context for `--extend`; family grid (8k+1 LTX-2, exactly 1 for Wan)                               |
| `--audio`, `--no-audio`                                                                      | Keep or strip synchronized LTX-2 MP4 audio                                                                                            |
| `--video-only`                                                                               | Skip the LTX-2 audio branch entirely; output-changing, and conflicts with `--audio` / `--audio-file`                                  |
| `--audio-file <PATH>`                                                                        | LTX-2 audio-to-video conditioning                                                                                                     |
| `--video <PATH>`                                                                             | LTX-2 source video for retake/video-conditioning                                                                                      |
| `--ic-lora-control <ID>`                                                                     | Official compatible LTX-2 reference control; requires `--video` and selects `ic-lora` (or `lip-dub` for `lipdub`)                     |
| `--keyframe <FRAME:PATH>`                                                                    | Repeatable LTX-2 keyframe conditioning; current H3 uses the required `--first-frame` instead                                          |
| `--last-image <PATH>`                                                                        | Closing frame for a Wan first/last-frame render; pairs with `--image`                                                                 |
| `--first-frame <PATH>`                                                                       | MiniMax H3 FL2VA opening frame; required by the current compact runtime                                                               |
| `--last-frame <PATH>`                                                                        | MiniMax H3 closing endpoint flag; the current compact runtime refuses this not-yet-qualified route                                    |
| `--reference <KIND=PATH>`                                                                    | Repeatable ordered MiniMax H3 Ref2VA image/video/audio input; remote upload requires `MOLD_API_KEY`                                   |
| `--pipeline <MODE>`                                                                          | `one-stage`, `two-stage`, `two-stage-hq`, `distilled`, `ic-lora`, `keyframe`, `a2-vid`, `retake`, `lip-dub`, or `t2a`                 |
| `--retake <START:END>`                                                                       | LTX-2 retake range in seconds                                                                                                         |
| `--camera-control <NAME\|PATH>`                                                              | LTX-2 camera-control preset or `.safetensors` path                                                                                    |
| `--spatial-upscale <MODE>`                                                                   | LTX-2 spatial upscaling, such as `x1.5` or `x2`                                                                                       |
| `--temporal-upscale <MODE>`                                                                  | LTX-2 temporal upscaling, currently `x2`                                                                                              |
| `--stg-scale <SCALE>`, `--stg-blocks <BLOCKS>`                                               | LTX-2 spatiotemporal guidance strength and the perturbed transformer blocks                                                           |
| `--rescale-scale <SCALE>`, `--modality-scale <SCALE>`                                        | LTX-2 CFG-rescale factor and audio/video cross-modality guidance                                                                      |
| `--guidance-skip-step <N>`                                                                   | Apply LTX-2 guidance every `N + 1` steps instead of every step                                                                        |
| `--spatial-tile <off\|auto\|PX[:OVERLAP]>`                                                   | LTX-2 spatial tiling for stage 2 and VAE decode (env: `MOLD_LTX2_SPATIAL_TILE`)                                                       |
| `--hdr-exr-dir <DIR>`                                                                        | Also write the render as a scene-referred linear OpenEXR sequence in this directory; requires `--ic-lora-control hdr`                 |
| `--hdr-exr-full-float`                                                                       | Write EXR samples at 32-bit float instead of 16-bit half; requires `--hdr-exr-dir`                                                    |
| `--sample-solver <SOLVER>`                                                                   | Wan denoise solver: `unipc` (default), `euler`, or `dpm++`                                                                            |
| `--sample-shift <SHIFT>`                                                                     | Wan flow shift; overrides the per-tier default                                                                                        |
| `--distill-strength <SPEC>`                                                                  | Wan Lightning distill strength: `high=X,low=Y` or one number for both experts                                                         |
| `-i, --image <PATH>`                                                                         | Source image; repeat for `qwen-image-edit`; `-` is stdin for single-image families                                                    |
| `--strength <FLOAT>`, `--mask <PATH>`                                                        | img2img/inpainting controls                                                                                                           |
| `--control <PATH>`, `--control-model <NAME>`, `--control-scale <FLOAT>`                      | SD1.5 ControlNet controls                                                                                                             |
| `-n, --negative-prompt <TEXT>`, `--no-negative`                                              | CFG-family negative prompt controls                                                                                                   |
| `--lora <PATH>`, `--lora-scale <FLOAT>`                                                      | LoRA adapter path and scale; `--lora` is repeatable; suffix `@high`/`@low` binds an adapter to one Wan 2.2 A14B expert                |
| `--upscale <MODEL>`                                                                          | Apply a Real-ESRGAN upscaler after generation                                                                                         |
| `--no-metadata`                                                                              | Disable embedded PNG metadata for this run                                                                                            |
| `--title <TEXT>`                                                                             | Print title (≤ 120 chars): embedded in metadata, seeded into the gallery row, slugged into the default filename                       |
| `--tag <TAG>`                                                                                | File the print under a tag; repeatable, up to 20 tags of 1–64 chars, matched case-insensitively                                       |
| `--collection <NAME>`                                                                        | File the print into a collection, creating it if absent; collections merge across machines by name                                    |
| `--no-auto-tag`                                                                              | Do not add the title as a tag, whatever `generate.auto_tag_title` says                                                                |
| `--preview`                                                                                  | Display output inline in the terminal                                                                                                 |
| `--expand`, `--no-expand`, `--expand-backend <URL>`, `--expand-model <MODEL>`                | Prompt expansion controls                                                                                                             |
| `--local`                                                                                    | Skip the server and run local inference                                                                                               |
| `--host <URL>`                                                                               | Override `MOLD_HOST`                                                                                                                  |
| `--gpus <SPEC>`                                                                              | Local GPUs: `all`, `none`, ordinals, or stable `cuda:`/`metal:`/`GPU-`/`MIG-` IDs                                                     |
| `--eager`, `--offload`                                                                       | VRAM/performance placement modes                                                                                                      |
| `--t5-variant <TAG>`, `--qwen3-variant <TAG>`, `--qwen2-variant <TAG>`                       | Text encoder variant overrides                                                                                                        |
| `--qwen2-text-encoder-mode <MODE>`                                                           | `auto`, `gpu`, `cpu-stage`, or `cpu`                                                                                                  |
| `--scheduler <SCHED>`                                                                        | `ddim`, `euler-ancestral`, `uni-pc`, or `edm-dpm-pp-2m` (Playground v2.5 only); Wan uses `--sample-solver`                            |
| `--cfg-plus`                                                                                 | Enable CFG++ on supported SD-family paths                                                                                             |
| `--device-text-encoders <DEV>`                                                               | Place all text encoders on `auto`, `cpu`, `gpu:N`, or an exact `/api/devices` ID                                                      |
| `--device-transformer <DEV>`, `--device-vae <DEV>`                                           | Advanced family placement overrides; accepts the same device forms                                                                    |
| `--device-t5 <DEV>`, `--device-clip-l <DEV>`, `--device-clip-g <DEV>`, `--device-qwen <DEV>` | Per-encoder placement overrides                                                                                                       |
| `--id-image <PATH>`                                                                          | Face reference photograph (PuLID); repeat up to 4 times to average several references of one person — see [Identity](/guide/identity) |
| `--id-weight <FLOAT>`                                                                        | Identity strength, `0.0`–`3.0` (default `1.0`); exactly `0.0` renders the unconditioned print                                         |
| `--id-start-step <N>`                                                                        | First denoise step identity is applied from (default `0`)                                                                             |
| `--true-cfg <SCALE>`                                                                         | True classifier-free guidance scale, `1.0`–`10.0` (default `1.0` = off); FLUX only                                                    |
| `--cfg-start-step <N>`                                                                       | First denoise step the true-CFG negative branch runs at (default `1`); requires `--true-cfg`                                          |

For video, the `--output` extension outranks the family's container default:
`mold run <video-model> "…" -o clip.gif` writes a real GIF even where the family
would have picked MP4. An extension this binary cannot encode (`.mp4` without
the `mp4` feature, `.webp` without `webp`) is refused before any weight is read
rather than filled with another container's bytes, as is a raster or audio
extension on a video render, and an explicit `--format` that disagrees with the
filename is reported instead of silently overriding it. `--output -` claims no
extension, so stdout keeps whatever container the family resolved.

### Qwen Family Encoder Controls

- `--qwen2-variant auto|bf16|q8|q6|q5|q4|q3|q2`
- `--qwen2-text-encoder-mode auto|gpu|cpu-stage|cpu`

`qwen-image-edit-2511:*` treats repeated `--image` flags as ordered
`edit_images`; non-edit families accept at most one source image.

### LTX-2 Notes

LTX-2 defaults to MP4, supports synchronized audio, and runs real generation on
CUDA and Apple Metal (Metal is performance-qualified on the 19B/22B distilled
FP8 tiers, slower than a comparable CUDA card); CPU is correctness-only.
Chaining works through repeated `--prompt`, `--script`, or large `--frames`
requests.

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
shim jobs and explicitly discarding completed jobs' editable scene caches.
Automatic maintenance leaves durable scene caches intact.

## `mold queue`

Inspect and control the generation queue on a running `mold serve` instance.
The commands use `MOLD_HOST` and send `MOLD_API_KEY` when configured; there is
no local fallback, because a queue belongs to one serving host.

```bash
mold queue list [--held] [--json]        # job, state, model, batch, prompt, admitted
mold queue show <JOB-ID> [--json]        # one job in full, with its batch progress
mold queue cancel <JOB-ID>...            # DELETE /api/queue/{id}
mold queue cancel --all [--yes]          # DELETE /api/queue for queued rows only
mold queue cancel --batch <BATCH-ID>     # DELETE /api/generation-batches/{id}
mold queue retry <JOB-ID>... | --held    # POST /api/queue/{id}/retry
mold queue move <JOB-ID> --to <N>        # PATCH /api/queue/{id} {position}
mold queue pause | mold queue resume     # POST /api/queue/pause | /resume
mold queue sweep                         # POST /api/queue/held/sweep + /api/generation-batches/sweep
```

The `STATE` column is the same vocabulary the web, desktop, and iPhone
surfaces render: a running row counts its denoise steps, position 0 is
`Next up`, everyone behind is `#N in line`, and only an actionable scheduler
reason (`Model not installed`, `Waiting for GPU memory`, …) replaces the
position. Ordinary serialization on a busy host (no idle device, a warm
wait, a lower-priority opening) keeps the row counting rather than reading
as a fault.

`mold queue list` and `mold queue retry --held` walk every durable
continuation page, so a backlog longer than the host's queue window is not
silently truncated. Held rows are listed again beneath the table with the
server's own error sentence and whether a retry is allowed. `mold queue retry` composes the full
retry authority (serving instance, batch, client batch, job) from the row's
own `batch_id` / `client_batch_id`; a hold that needs operator repair is
refused by name rather than silently skipped. `--json` prints the raw server
documents.

## `mold library`

Browse and organize existing prints on the server selected by `MOLD_HOST`
(`MOLD_API_KEY` is sent when configured). Non-grid commands never fall back to
direct filesystem access.

```bash
mold library list [--query TEXT] [--tag TAG] [--tag TAG]... [--collection NAME-OR-SLUG] [--favorite] [--format FORMAT] [--limit N] [--offset N] [--json]
mold library show <FILENAME> [--json | --preview]
mold library grid [--host URL | --local]
mold library title <FILENAME> <TEXT>
mold library title <FILENAME> --clear
mold library favorite <FILENAME>...
mold library unfavorite <FILENAME>...
mold library trash <FILENAME>...

mold library tag list [--json]
mold library tag add <FILENAME>... --tag <TAG> [--tag <TAG>]...
mold library tag remove <FILENAME>... --tag <TAG> [--tag <TAG>]...
mold library tag rename <OLD> <NEW>
mold library tag delete <TAG> [--yes]

mold library collection list [--json]
mold library collection show <NAME-OR-SLUG> [--json]
mold library collection create <NAME> [--description TEXT]
mold library collection update <NAME-OR-SLUG> [--name TEXT] [--description TEXT | --clear-description] [--cover FILENAME | --clear-cover] [--hidden | --visible]
mold library collection delete <NAME-OR-SLUG> [--yes]
mold library collection add <NAME-OR-SLUG> <FILENAME>...
mold library collection remove <NAME-OR-SLUG> <FILENAME>...
```

Repeat `--tag` for every tag; one flag consumes exactly one value. Multiple
`--tag` filters use AND semantics. Listing filters first, orders by
newest timestamp and then filename, and only then applies `--offset` and
`--limit` (50 by default, 1,000 maximum). JSON contains the identical selected
page and never includes preview or terminal escape bytes.

Tag and favorite edits use the replay-safe bulk mutation route when the host
advertises it, with an automatic fallback to the older organization route.
Hosts without Library organization fail with an upgrade-or-metadata-database
diagnostic. `mold library trash` is allowed only when the host explicitly
advertises recoverable trash, so an older server cannot reinterpret it as a
permanent delete.

`mold library show --preview` reuses the same inline renderer as `mold run
--preview`; video entries prefer their animated preview and fall back to the
thumbnail. `mold library grid [--host URL | --local]` opens the existing TUI
directly on its protocol-aware Library grid and carries `MOLD_API_KEY` only for
that process. An unreachable host or rejected gallery credential is an error;
the strict grid never switches to local files.

## `mold trash`

Inspect, restore, or empty the gallery trash on a running `mold serve`
instance. Deleting a print from any surface moves it to the host's trash
(`<output_dir>/.trash/`) instead of removing it; the server purges trashed
prints after `gallery.trash_retention_days` (default 30, `0` keeps them
forever; see [Configuration](/guide/configuration#library-trash)). The
commands use `MOLD_HOST` and send `MOLD_API_KEY` when configured; there is no
local fallback, because the trash belongs to that host's gallery.

```bash
mold trash list [--json]          # filename, title, trashed, purges, size
mold trash restore <FILENAME>...  # back to the live gallery (409 if a live print took the name)
mold trash empty [--yes]          # purge everything; confirms unless --yes
mold trash sweep                  # run the retention sweep now
```

`mold trash list` shows each print's purge countdown as `in 27d`, `kept`
when retention is keep-forever, or `due` when the next sweep will remove it.
`--json` prints the raw `GET /api/gallery?view=trash` rows. `mold trash
empty` and `mold trash sweep` mirror `DELETE /api/gallery/trash` and
`POST /api/gallery/trash/sweep`.

## `mold expand`

Preview prompt expansion without generating.

```bash
mold expand <PROMPT> [OPTIONS]
```

| Flag                     | Description                    |
| ------------------------ | ------------------------------ |
| `-m, --model <MODEL>`    | Target model for style/context |
| `--task <TASK>`          | Conditioning task to preview   |
| `--variations <N>`       | Number of variations           |
| `--json`                 | Output as JSON array           |
| `--backend <URL>`        | Expansion backend override     |
| `--expand-model <MODEL>` | LLM model override             |

## `mold remix`

Preview subject-preserving alternatives without queueing generation. Three
variants are returned by default; use `--json` for the structured source and
dimension provenance.

```console
mold remix <SOURCE_PROMPT> [OPTIONS]
mold remix "a lighthouse" --dimensions camera,lighting --variations 4
mold remix "she turns" --model ltx-2-19b-distilled:fp8 --task image-to-video
```

Use `--source original|current|direct` and optional `--root-prompt` to describe
where the selected source came from. `--style` is locked across every variant.

## `mold serve`

Start the HTTP inference server.

```bash
mold serve [--port N] [--bind ADDR] [--models-dir PATH] [--gpus SPEC] [--queue-size N] [--log-format json|text] [--log-file] [--discord] [--no-mdns]
```

| Flag                  | Description                                                                                                                                                                                      |
| --------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `--port <N>`          | Port, defaults to `7680` or `MOLD_PORT`                                                                                                                                                          |
| `--bind <ADDR>`       | Bind address, defaults to `0.0.0.0`                                                                                                                                                              |
| `--models-dir <PATH>` | Override the models directory                                                                                                                                                                    |
| `--gpus <SPEC>`       | `all`, `none`, ordinals, or stable `cuda:`/`metal:`/`GPU-`/`MIG-` IDs; defaults to `all`                                                                                                         |
| `--queue-size <N>`    | Jobs hydrated into the runtime window (overflow returns HTTP 503 `QUEUE_FULL` + `Retry-After`); the durable SQLite backlog itself is uncapped — see [Configuration](/guide/configuration#server) |
| `--log-format <FMT>`  | `json` or `text`                                                                                                                                                                                 |
| `--log-file`          | Enable rotated logs under `~/.mold/logs/`                                                                                                                                                        |
| `--discord`           | Start the built-in Discord bot in the same process                                                                                                                                               |
| `--no-mdns`           | Disable LAN advertising and server-assisted peer browsing (`mdns` builds; also `MOLD_MDNS=0`)                                                                                                    |

`GET /api/status` returns `gpus[]` with per-worker state and
`queue_depth`/`queue_capacity` for queue health.

### Multi-GPU

`--gpus all` (the default) starts every runtime-visible device with a stable
identity. `none` starts no inference workers: the server remains available for
inventory, telemetry, downloads, and settings, while generation and admin
model-load requests return `503 GENERATION_UNAVAILABLE`.

Specific selectors are comma-separated. Numeric ordinals such as `0,1` are
process-local and kept for compatibility. Persistent configuration should use
IDs returned by `GET /api/devices`: `cuda:<32-hex-uuid>` for CUDA devices or
`metal:default` for Apple Metal. NVIDIA `GPU-...` and `MIG-...` UUID spellings
are also accepted. CUDA/MIG UUID prefixes may be abbreviated only when they
match exactly one runtime-visible device; ambiguous or missing selectors fail
startup rather than choosing another GPU.

Runtime controls target the serving host (`MOLD_HOST` and `MOLD_API_KEY`
apply):

```bash
mold gpu list [--json]
mold gpu disable <stable-id-or-ordinal>
mold gpu enable <stable-id-or-ordinal>
```

When the target is loopback and no server is running, `gpu list` discovers the
devices visible to the current Mold runtime directly. The JSON schema stays the
same; operational telemetry that only the server samples remains `null` rather
than being fabricated. `gpu enable` and `gpu disable` persist the stable
device's startup preference in the local metadata database and report that it
takes effect on the next `mold serve`. They never fall back to local hardware
when `MOLD_HOST` names another machine.

Disable removes the device from future scheduling immediately. Active work
finishes before Mold drops its device-backed caches on the owner thread and
joins it. Re-enable starts a fresh owner thread; it never resets and reuses a
CUDA primary context in-process. Desired enablement is machine-wide and
persists across restarts and temporary device absence. A startup-excluded
device still requires a restart with a broader `--gpus` selection.
Live changes require Scheduler V2. In legacy or observe mode, `gpu enable`
can recover a persistently-disabled, startup-selected device for the next
server restart; live disable remains unavailable.

## `mold server discover`

Browse the local network (mDNS/DNS-SD, `_mold._tcp`) for running `mold serve`
instances that advertise themselves. Available in builds compiled with the
`mdns` feature (included in release binaries and the Nix package).

```bash
mold server discover [--timeout-secs N] [--json] [--probe]
```

| Flag                 | Description                                                                    |
| -------------------- | ------------------------------------------------------------------------------ |
| `--timeout-secs <N>` | How long to browse before reporting (default `3`)                              |
| `--json`             | Emit the raw list of discovered servers as JSON                                |
| `--probe`            | Also probe each server's `/health` + `/api/status` and show a `LATENCY` column |

The table lists NAME, URL, VERSION, AUTH (whether an API key is required), and a
GPU summary, followed by a `export MOLD_HOST=…` hint for the first result.
Advertising and server-assisted browsing are on by default when a server is
built with the `mdns` feature; disable both per-server with
`mold serve --no-mdns` or `MOLD_MDNS=0`.

## `mold mcp`

Start a stdio Model Context Protocol server that proxies to `mold serve`.

```bash
mold mcp [--host URL]
```

MCP exposes nine tools: generation, async generation with job status and
retry, gallery listing and lookup, installed LoRA listing, model listing, and
server status. It intentionally proxies the server
surface instead of embedding local inference.

## `mold pull`, `mold list`, `mold info`

```bash
mold pull flux-schnell:q8
mold pull pulid-flux --accept-license insightface-antelopev2
mold list
mold info
mold info flux-dev:q4
mold info flux-dev:q4 --verify
```

`mold pull <MODEL> [--skip-verify] [--accept-license <ID>]` downloads manifest
models locally or through the reachable server. `--skip-verify` skips the
SHA-256 pass after the download; `--accept-license` records a third-party
model-license acceptance before pulling (see
[Configuration](/guide/configuration#third-party-model-licenses)).
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
| Scheduler | `scheduler.replan_debounce_ms`, `scheduler.replan_max_delay_ms`, `scheduler.warm_wait_max_ms`                                                                                                                                      |
| Gallery   | `gallery.trash_retention_days` (days a trashed print is kept before the sweeper purges it; `0` = forever, default 30, stored in `mold.db`)                                                                                         |
| Queue     | `queue.held_retention_days` (days a held queue row is kept before the sweeper purges it; `0` = forever, default 30, stored in `mold.db`)                                                                                           |
| Generate  | `generate.auto_tag_title` (add the print title as a tag by default; on unless set to `false`)                                                                                                                                      |
| Logging   | `logging.level`, `logging.file`, `logging.dir`, `logging.max_days`                                                                                                                                                                 |
| RunPod    | `runpod.api_key`, `runpod.default_gpu`, `runpod.default_datacenter`, `runpod.default_network_volume_id`, `runpod.auto_teardown`, `runpod.auto_teardown_idle_mins`, `runpod.cost_alert_usd`, `runpod.endpoint`                      |
| Lambda    | `lambda.api_key`, `lambda.endpoint`, `lambda.image_repository`, `lambda.ssh_key_name`, `lambda.ssh_private_key_path`, `lambda.filesystem_prefix`, `lambda.filesystem_mount_path`, `lambda.confirm_hourly_usd`, `lambda.local_port` |
| Per-model | `models.<name>.<field>` where field is one of `default_steps`, `default_guidance`, `default_width`, `default_height`, `scheduler`, `negative_prompt`, `lora`, `lora_scale`                                                         |

`config.toml` owns bootstrap paths, ports, credentials, logging, and model path
overrides. The SQLite settings DB owns user preferences and per-model
generation defaults. The TUI's own `tui.*` preferences are DB-backed too, but
are written by the TUI rather than listed in the static key registry.
`umt5_variant` is registered as a key name only: it has no read/write arm and no
DB slot ([#778](https://github.com/utensils/mold/issues/778)), and it is
stripped whenever mold rewrites `config.toml`, so set
the Wan UMT5 encoder variant with `MOLD_UMT5_VARIANT` instead.

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

The Discord bot exposes slash commands for generation, durable LTX-2 sequences,
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

Common subcommands are `doctor`, `gpus`, `datacenters`, `network-volume`,
`list`, `get`, `create`, `start`, `stop`, `delete`, `connect`, `logs` (RunPod
console handoff), `usage`, and `run`.
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

| Command                                                       | Purpose                                                                                           |
| ------------------------------------------------------------- | ------------------------------------------------------------------------------------------------- |
| `mold default [MODEL]`                                        | Get or set the default model                                                                      |
| `mold stats [--json]`                                         | Show disk usage for models, output, logs, and shared components                                   |
| `mold clean [--force] [--older-than DURATION]`                | Report stale downloads, orphaned files, and old outputs (dry run); `--force` deletes them         |
| `mold server start/status/stop`                               | Manage a background server daemon                                                                 |
| `mold server discover`                                        | Find mold servers advertised on the local network (mDNS)                                          |
| `mold rm <MODELS...> [--force]`                               | Remove downloaded models                                                                          |
| `mold ps`                                                     | Show server status or local mold processes                                                        |
| `mold unload`                                                 | Unload the current server model                                                                   |
| `mold update [--check] [--force] [--nightly] [--version TAG]` | Update a stable, nightly, or exact release binary                                                 |
| `mold licenses [--local]`                                     | Show third-party model licenses and whether the machine that would run the pull has accepted them |
| `mold skill <COMMAND>`                                        | Manage Mold's embedded Agent Skill                                                                |
| `mold version`                                                | Show version, build date, and git SHA                                                             |

## Running commands without `mold serve`

The CLI does not need a daemon for work whose authority already exists on disk,
in the local runtime, or in a named cloud API. Server-first commands fall back
only when doing so preserves the target the user asked about.

| Behavior without a local server    | Commands                                                                                                              | Result                                                                                                               |
| ---------------------------------- | --------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------- |
| Fully standalone, local files      | `list`, `info`, `default`, `config`, `stats`, `clean`, `rm`, `chain validate`                                         | Reads or changes `MOLD_HOME` directly                                                                                |
| Fully standalone, local runtime    | `gpu list`, `gpu enable`, `gpu disable`, `ps`, `unload`                                                               | Lists local devices, persists next-start device preferences, reports processes, or completes an already-empty unload |
| Server-first with local execution  | `run`, `pull`, `upscale`                                                                                              | Uses the server when reachable, otherwise executes or downloads locally                                              |
| Standalone prompt tooling          | `expand`, `remix`                                                                                                     | Uses the configured local expansion model or external API backend                                                    |
| Standalone lifecycle/discovery     | `serve`, `server start`, `server status`, `server stop`, `server discover`                                            | Starts or inspects processes, or browses mDNS directly                                                               |
| Standalone utility/network clients | `version`, `update`, `completions`, `skill`, `runpod`, `lambda`                                                       | Uses embedded data, GitHub, agent paths, or the explicitly named cloud API                                           |
| Requires a live Mold server        | `jobs`, `queue`, `library` (except `library grid --local`), `trash`, `mcp`, `discord`; `tui` unless `--local` is used | These operate on server-owned queue, gallery, tool, or UI state and do not substitute a different local authority    |

An unreachable non-loopback `MOLD_HOST` remains an error for host-administration
commands. In particular, `gpu`, `ps`, and `unload` do not answer with this
machine's state when the user selected a remote machine.

## `mold skill`

Install the embedded Mold Agent Skill for AI coding agents:

```bash
mold skill list
mold skill install codex
mold skill install --detected
mold skill install --all
mold skill install claude codex --project
mold skill install openclaw --dir ~/repo
mold skill uninstall codex
mold skill uninstall --project
mold skill show
```

Supported targets match nxv: `claude`, `codex`, `pi`, `openclaw`, `copilot`,
`cursor`, `gemini`, `amp`, `goose`, and generic `agents`. User-wide is the
default. `--project` uses the current directory, while `--dir` selects another
project root. Install requires explicit names, `--detected`, or `--all` and
atomically replaces only `mold/SKILL.md`; uninstall preserves sibling files.

## `mold completions`

Generate shell completions.

```bash
mold completions zsh
mold completions bash
mold completions fish
mold completions elvish
mold completions powershell
```

Dynamic completion includes command and flag names, known and installed model
IDs where appropriate, upscaler IDs, config keys, RunPod resources, completion
shell names, and locally visible stable GPU IDs for `gpu enable|disable`.

Common setup:

```bash
source <(mold completions zsh)
source <(mold completions bash)
mold completions fish > ~/.config/fish/completions/mold.fish
```
