# Configuration

mold keeps configuration in two places by design:

- **`config.toml`** — the hand-editable _bootstrap_ file in `~/.mold/` (or
  `$MOLD_HOME`). Owns paths, ports, credentials, and the model-path
  entries that `mold pull` writes.
- **`mold.db` (SQLite)** — owns user preferences: the `[expand]` section,
  scheduler timing preferences (`scheduler.*`),
  global generation defaults (`default_width`, `default_height`,
  `default_steps`, `embed_metadata`, `t5_variant`, `qwen3_variant`,
  `umt5_variant`, `default_negative_prompt`), the `last-model` sidecar, and per-model
  generation defaults (`default_steps`, `default_guidance`,
  `default_width`, `default_height`, `scheduler`, `negative_prompt`,
  `lora`, `lora_scale`). These fields moved to the DB in
  [#265](https://github.com/utensils/mold/issues/265) so GUI writes and
  hand-curated TOML no longer fight over the same file.

Environment variables still override both surfaces at read time.
Upgrading from an earlier release runs a one-shot import of the
preference slices of `config.toml` into the DB on first launch — your
existing values carry over.

The web app's **Settings → All settings** panel exposes this same effective
configuration surface. It provides typed editors for known keys, keeps unknown
future keys visible under Advanced, labels each value's DB/file/environment
provenance, prevents writes that an environment variable would override, and
supports per-key reset plus profile creation and switching.

`output_dir` is a startup-only trust root for a running server. The live
`PUT /api/config/output_dir` endpoint returns `409 RESTART_REQUIRED` without
changing memory or disk. Stop the server, run
`mold config set output_dir <path>`, then restart it.

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
mold config set scheduler.replan_debounce_ms 2000 # Scheduler timing → mold.db
mold config set gallery.trash_retention_days 7    # Library trash retention → mold.db (0 = keep forever)
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

Scheduler V2 reads its profile-scoped timings when the coordinator starts.
`scheduler.replan_debounce_ms` defaults to 2000,
`scheduler.replan_max_delay_ms` to 5000, and
`scheduler.warm_wait_max_ms` to 2000. Each accepts 0–30000 milliseconds;
maximum delay must be at least the debounce. The config API and Mold Studio
mark these rows restart-required. Restart the server after changing them.

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

| Variable                          | Default                            | Description                                                                                                                                                                                                                                                                                                                                                                          |
| --------------------------------- | ---------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `MOLD_HOME`                       | `~/.mold`                          | Base directory for config and cache                                                                                                                                                                                                                                                                                                                                                  |
| `MOLD_DEFAULT_MODEL`              | `flux2-klein:q8`                   | Default model name                                                                                                                                                                                                                                                                                                                                                                   |
| `MOLD_HOST`                       | `http://localhost:7680`            | Remote server URL                                                                                                                                                                                                                                                                                                                                                                    |
| `MOLD_MODELS_DIR`                 | `$MOLD_HOME/models`                | Model storage directory                                                                                                                                                                                                                                                                                                                                                              |
| `MOLD_PORT`                       | `7680`                             | Server port                                                                                                                                                                                                                                                                                                                                                                          |
| `MOLD_MDNS`                       | `1` (on)                           | Set `0`/`false` to disable `mold serve` LAN advertising and server-assisted DNS-SD browsing (requires the `mdns` build feature)                                                                                                                                                                                                                                                      |
| `MOLD_DISPATCH_MODE`              | `v2`                               | Restart-time GPU dispatch owner: V2 is authoritative by default. During the one-release rollback window, `legacy` restores the prior depth-two transport and `observe` retains legacy ownership while recording request/placement/host-memory-feasible V2 decisions read-only. Queue pause gates generation, utility, and admin GPU work in every mode. Invalid values fail startup. |
| `MOLD_DISTRIBUTION_IMAGE_VERSION` | `latest`                           | Release-build input: official stable builds embed their exact release, fetch that release's target digest manifest, and submit `repository@sha256:…`; source/Nix/rolling builds use mutable `latest*`. End users should not override it at runtime.                                                                                                                                  |
| `LAMBDA_API_KEY`                  | unset                              | Overrides `lambda.api_key`                                                                                                                                                                                                                                                                                                                                                           |
| `MOLD_LOG`                        | `info` (serve) / `warn` (cli, tui) | Log level                                                                                                                                                                                                                                                                                                                                                                            |

### Development and qualification

| Variable                 | Default | Description                                                                                                                                                                           |
| ------------------------ | ------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `MOLD_TEST_PULID_ASSETS` | unset   | Test-only path to the pinned PuLID and AntelopeV2 assets used by ignored parity tests. It may name the checkpoint file or a directory searched one level deep; production ignores it. |

### Generation

| Variable                              | Default                                 | Description                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                           |
| ------------------------------------- | --------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `MOLD_EAGER`                          | —                                       | `1` to keep all components loaded                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                     |
| `MOLD_OFFLOAD`                        | —                                       | `1` to force block offload for FLUX, Flux.2, Z-Image, Qwen-Image, LTX-2, Wan, and SD3 BF16/FP8 paths where implemented. FLUX / Flux.2 / Z-Image / Qwen-Image keep fitting blocks GPU-resident and stream only overflow blocks; LTX-2, Wan (GGUF only — it parks every block), and SD3 full-stream.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                    |
| `MOLD_OFFLOAD_PREFETCH`               | `on`                                    | FLUX offload async H2D prefetch stream — set `off` to revert to synchronous                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                           |
| `MOLD_PINNED_VRAM_MAX_GB`             | RAM × 0.5                               | Cap on pinned host memory used by the FLUX offload path                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                               |
| `MOLD_RESERVE_VRAM_MB`                | 400 (Linux) / 600 (Windows) / 0 (macOS) | OS / cuBLAS workspace reserve subtracted from `free_vram_bytes` before any budget decision. Set explicitly to override the platform default; `0` disables                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                             |
| `MOLD_KEEP_TE_RAM`                    | off                                     | `1` parks text encoders on CPU host RAM between requests instead of dropping them; any other value (and the unset default) keeps the drop-and-reload. It stays opt-in deliberately: a park is a multi-gigabyte **host** allocation, and mold has no way to see a container's cgroup limit, so nothing decides this for you. Since [#1044](https://github.com/utensils/mold/issues/1044) it also covers **Qwen-Image's quantized GGUF Qwen2 encoder** — its `QTensor` bytes move host↔device losslessly rather than being re-read from disk, worth a measured 35.1 s per cold prompt. Other families' GGUF encoders (FLUX/SD3's T5, Z-Image's Qwen3, Wan's UMT5) still drop and reload; only their FP16/BF16 encoders park. Disabled on Metal (unified memory). Includes Wan's UMT5-XXL, where the saving is largest: the parked copy keeps the checkpoint's F16, so a second consecutive render skips an 11.4 GB disk read for ~11.4 GB of host RAM.                                                                                                                                                                                                                                                                                                  |
| `MOLD_LORA_BYPASS`                    | `auto`                                  | FLUX LoRA application path: `auto` enables bypass-mode when LoRAs are present (covers offload AND the GGUF/quantized path via `quantized_transformer.rs`), `on` always bypasses, `off` reverts to legacy merge-into-base / `gguf_lora_var_builder`                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                    |
| `MOLD_VAE_TILED`                      | `auto`                                  | Tiled VAE decode for FLUX/FLUX2/SDXL/SD3: `auto` retries with tiling on OOM, `force` always tiles, `off` disables                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                     |
| `MOLD_STEP_PREVIEW`                   | `1`                                     | Live denoise previews over `/api/generate/stream` (`preview` SSE events; FLUX.1/Flux.2/Z-Image/Wan — Wan projects the clip's middle latent frame): a small latent-resolution PNG per step via linear latent→RGB projection, throttled to ~700 ms. `0` disables.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                       |
| `MOLD_LONG_PROMPTS`                   | —                                       | `1` enables ComfyUI-style chunked CLIP encoding (75-token windows, BOS/EOS framing, pooled outputs averaged into the FLUX `vector_in` 768-dim conditioning). Default off — pre-Tier-2 hard truncation at 77 preserved.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                |
| `MOLD_ATTN`                           | `math`                                  | Attention backend: `math` (hand-rolled SDP) or `flash` (candle-flash-attn v2). Flash needs a `--features cuda,flash-attn` build and a CUDA fp16/bf16 tensor, and is opt-in even in that build — the default is `math` everywhere, so the image a seed renders never depends on which artifact you installed. Reaches FLUX offload/GGUF-bypass, all Flux.2, Z-Image offload, the Wan DiT's self/cross attention, and all three Qwen-Image transformers (BF16, quantized, offloaded) on CUDA/CPU — Qwen keeps candle's fused `sdpa` on Metal; dense BF16 FLUX and Z-Image use upstream Candle's own attention and are unaffected. On Wan, flash is a speed lever rather than a memory one: measured on an RTX 4090 (`wan22-t2v-a14b:q5`, 53f/832x480, same binary, only the backend varying) it renders in 75.3 s against math's 158.4 s — 2.1x — while peak VRAM falls only from 22,250 MiB to 21,354 MiB. Ineligible tensors fall back to math silently; a build without the feature warns once.                                                                                                                                                                                                                                                      |
| `MOLD_ATTN_CHUNK`                     | auto                                    | Override math-attention query chunk size. Positive integers below the sequence length enable chunking; `0` or `off` disables it. The CUDA default chunks long queries at `512` to reduce peak VRAM.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                   |
| `MOLD_EMBED_METADATA`                 | `1`                                     | `0` to disable PNG metadata                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                           |
| `MOLD_MEDIA_ROOTS`                    | —                                       | Platform path-list of allow roots for trusted server-local LTX-2 `audio_file_path` / `source_video_path` requests. Targets are canonicalized and must resolve to files under one configured root.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                     |
| `MOLD_PREVIEW`                        | —                                       | `1` to display images inline in terminal                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                              |
| `MOLD_T5_VARIANT`                     | `auto`                                  | T5 encoder: auto/fp16/q8/q6/q5/q4/q3                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                  |
| `MOLD_UMT5_VARIANT`                   | `auto`                                  | Wan UMT5 encoder: auto/fp16/q8/q6/q5. Auto prefers FP16 on GPU when it fits, else the largest GGUF that does — a CPU-parked UMT5 widens to F32                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        |
| `MOLD_QWEN3_VARIANT`                  | `auto`                                  | Qwen3 encoder: auto/bf16/q8/q6/iq4/q3                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                 |
| `MOLD_SCHEDULER`                      | —                                       | SD1.5/SDXL: ddim/euler-ancestral/uni-pc                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                               |
| `MOLD_CFG_PLUS`                       | —                                       | `1` to enable CFG++ (manifold-projection guidance, Chung et al. 2024). Drops usable CFG to ~1.5–2.5 and removes guidance artifacts. Per-request `--cfg-plus` overrides. Supported on SD3, SDXL, and SD1.5 (DDIM only — Euler-A / UniPC fall back). Ignored by FLUX / Z-Image / Flux.2 (distilled).                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                    |
| `MOLD_VAE_DTYPE`                      | `auto`                                  | Override VAE precision: `auto`, `bf16`, `fp16`, `fp32`. Use `fp32` to fix banding artifacts on FLUX/SD3 finetuned VAEs (~2× decode VRAM; tiled VAE absorbs OOM via existing fallback). Wired into FLUX, FLUX2, SD3, SDXL, SD1.5.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                      |
| `MOLD_NVFP4_BACKEND`                  | `auto`                                  | NVFP4 backend selection for Flux.2 and LTX-2: `auto` and `portable` use portable CPU BF16 streaming dequant; `native` is reserved for validated sm_120/Blackwell tensor-core execution and fails clearly on non-Blackwell hosts.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                      |
| `MOLD_LTX2_GEMMA_DEVICE`              | `auto`                                  | LTX-2 Gemma 3 12B prompt encoder placement: `auto` uses the GPU leased to the stage when it has more than 6 GB free, otherwise CPU; it never allocates on an unleased sibling GPU. The encoder is built one decoder layer at a time and each layer is dropped before the next, so its real peak residency is ~3.3 GB rather than the ~23 GB of BF16 weights on disk — a 24 GB card keeps Gemma on the GPU, where encoding costs seconds instead of a minute or more. `cpu` forces system RAM (slower, but no VRAM contention); `gpu` pins the assigned GPU and surfaces OOM instead of auto-offloading. An auto-placement OOM retries only Gemma on CPU while the transformer and video VAE stay on CUDA. The deprecated `MOLD_LTX2_DEBUG_FORCE_CPU_PROMPT_ENCODER=1` is a one-shot-warn alias. Server preflight uses the same resolver as runtime.                                                                                                                                                                                                                                                                                                                                                                                                   |
| `MOLD_LTX2_GEMMA_VARIANT`             | `auto`                                  | LTX-2 Gemma 3 12B weight format: `auto` prefers BF16 but downgrades to a present Q4 GGUF when BF16 will not fit in available host memory (mirrors admission's max(15% of RAM, 8 GiB) headroom; disclosed at WARN, never silent; requires exactly one local `.gguf` — an ambiguous set refuses; hosts without a reclaimable-memory figure keep BF16), `q4` (force Q4 GGUF — `google/gemma-3-12b-it-qat-q4_0-gguf`, ~7 GB on disk), `bf16` (force BF16 split — `google/gemma-3-12b-it-qat-q4_0-unquantized`, ~23 GB; historical default). For V1, place the Q4 GGUF in your gemma_root manually — manifest auto-fetch is deferred to a follow-up.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                       |
| `MOLD_LTX2_KEEP_SESSION`              | on                                      | LTX-2 retains its runtime session across generations ([#1099](https://github.com/utensils/mold/issues/1099)): a repeat of the **same prompt** serves the session's cached encoding instead of reloading the ~24 GB Gemma 3 12B prompt encoder. The retained session is small — it holds the cached encoding and a device handle, not the encoder or transformer — so a changed prompt still performs a full load, and steady-state memory is essentially unchanged. Set `0`, `false`, or `off` to restore the previous behavior of rebuilding the session every generation. The session is released when the model is unloaded or evicted from the cache.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                             |
| `MOLD_LTX2_SPATIAL_TILE`              | `auto`                                  | LTX-2 spatial tiling for stage-2 refinement and VAE decode, also settable per-run as `--spatial-tile`. `auto` tiles only past the 2048-px axis span the checkpoints' RoPE was trained on — so nothing mold currently renders is affected — `off` never tiles, and `<px>` or `<px>:<overlap>` (multiples of 32, overlap defaults to 256) forces that tile size at any resolution. A tiled stage 2 refines video only and carries stage 1's audio through unrefined, matching upstream.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                 |
| `MOLD_LTX2_VAE_FORCE_FULL_DECODE`     | —                                       | `1` to disable adaptive temporal chunked LTX-2 VAE decode and force one full decode pass. Useful for debugging/comparison; long or high-resolution clips may OOM.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                     |
| `MOLD_LTX2_VAE_FORCE_FRAMEWISE`       | —                                       | `1` to force temporal-chunk LTX-2 VAE decode even when a full decode would fit. Reduces peak VRAM at a small decode-time cost.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        |
| `MOLD_LTX2_VAE_DECODE_CHUNK_FRAMES`   | `4` latent frames                       | Positive integer number of latent frames per LTX-2 VAE decode chunk when chunked decode is active.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                    |
| `MOLD_WAN_SHIFT`                      | per tier (`8.0`; A14B `5.0`)            | Wan flow shift (upstream `--sample_shift`), the family's primary quality/character knob. Process-wide fallback for `mold serve`; a request-level `sample_shift` (CLI `--sample-shift`, TUI Advanced ▸ Video ▸ Flow shift) always wins. Must be finite and positive.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                   |
| `MOLD_WAN_SOLVER`                     | `unipc`                                 | Wan sample solver fallback: `unipc` (FlowUniPC, the UAT'd default), `euler` (the solver the 4-step Lightning distills were tuned for), or `dpm++` (upstream's alternative, on its own sigma grid). A request-level `scheduler` / `--sample-solver` always wins.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                       |
| `MOLD_WAN_OFFLOAD_BLOCKS`             | auto                                    | Trailing Wan transformer blocks to park in host RAM ([#776](https://github.com/utensils/mold/issues/776)). Unset lets the engine decide: it parks only when the render's activation budget will not fit what is free once the weights land, so a clip that already fits is unaffected. `0` disables it, and an explicit count wins over `MOLD_OFFLOAD`. GGUF checkpoints only — the move is a raw-byte round trip the plain and fp8 sources have no equivalent of. Measured on an RTX 4090: `wan22-t2v-a14b:q5` at 81 frames / 832x480 renders in 316.3 s at a 17,322 MiB peak, where it previously OOM'd; `:q8` reaches 73 frames in 2,235.0 s at 16,650 MiB. Parked blocks are rebuilt on the device once per step, so this trades wall clock for VRAM.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                             |
| `MOLD_WAN_PREFETCH`                   | `1`                                     | `0` disables the background page-cache warm for the A14B partner expert ([#802](https://github.com/utensils/mold/issues/802)). The warm is host I/O only — the bytes are read and discarded — so it never allocates on the device and the max-of-pair VRAM invariant is unchanged. Measured on an RTX 4090 (`wan22-t2v-a14b:q5`, cold page cache): the mid-loop swap load falls from 8.7-22.2 s to a consistent 5.6 s.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                |
| `MOLD_WAN_STEP_CACHE`                 | `off`                                   | Wan first-block residual reuse ([#801](https://github.com/utensils/mold/issues/801)) for the non-distilled quality tiers: `off`, `auto` (relative-L1 threshold 0.10), or an explicit positive threshold. Block 0 runs every step; when its residual moves less than the threshold, blocks 1..N are skipped and their previous contribution is replayed. Refused with a message when a distill adapter is active or the schedule is under 12 steps — neither has redundant steps to skip. Changing the threshold changes the output: a cached run is a different sample of comparable quality, not the same frames faster. `off` is bit-identical to denoising every block. Measured on an RTX 4090 (`wan22-t2v-a14b:q8`, 33f/832x480, 20 steps): 605.6 s off, 327.4 s at 0.10 (1.85x).                                                                                                                                                                                                                                                                                                                                                                                                                                                                |
| `MOLD_WAN_STEP_PROFILE`               | off                                     | Diagnostic ([#775](https://github.com/utensils/mold/issues/775)): `1` prints one per-phase, device-synced timing line per Wan denoise step (self/cross attention, ffn, quantized matmuls, casts). The syncs inflate wall time — for auditing, never production.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                       |
| `MOLD_WAN_FORCE_DMMV`                 | off                                     | Diagnostic ([#775](https://github.com/utensils/mold/issues/775)): `1` forces the quantized matmuls onto the dequantize-per-forward fallback for A/B comparison against the MMQ fast path. Changes numerics, runtime, and transient memory; registered as an engine-shaping variable so scheduler fingerprints and learned estimates never mix forced runs with normal ones.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                           |
| `MOLD_LTX2_VAE_DECODE_CONTEXT_FRAMES` | auto                                    | Positive integer latent-frame overlap/context around each LTX-2 decode chunk. Default derives from the decoder causal-conv receptive field.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                           |
| `MOLD_QWEN_QMATMUL`                   | `0`                                     | Quantized (GGUF) Qwen-Image linears on CUDA ([#1045](https://github.com/utensils/mold/issues/1045)). **Experimental, off by default**: `1` (or `true`/`on`/`yes`) keeps the checkpoint quantized and feeds candle's MMQ/MMVQ kernels directly — measured on an RTX 4090 (2026-08-14, `qwen-image-2512:q4`, 1024², 20 steps) those kernels currently return 100% NaN on Qwen's shapes at the very first forward, which the boundary validator catches by aborting the render, so the default stays on the dequantize-to-BF16-per-forward arm until the kernel defect is root-caused ([#1048](https://github.com/utensils/mold/issues/1048); the reproduction recipe lives in `docs/architecture/qwen-mmq-nan.md`). Weights the kernels cannot take (an `IQ*` or float-stored tensor, or a row width the dtype's MMQ block size does not divide), CPU-staged weights, and a process already forced onto the fallback by `MOLD_WAN_FORCE_DMMV=1` keep the dequant arm regardless. Metal and CPU ignore it. Changes numerics slightly (int8 MMQ vs a BF16 GEMM over dequantized weights), so it is registered as an engine-shaping variable and never shares a scheduler fingerprint or learned estimate with the other setting.                          |
| `MOLD_ZIMAGE_QMATMUL`                 | `0`                                     | Quantized (GGUF) Z-Image linears on CUDA. **Experimental, off by default**: `1` (or `true`/`on`/`yes`) feeds candle's quantized MMQ/MMVQ CUDA kernels directly — measured on an RTX 4090 (2026-08-15, `z-image-turbo:q4`, 512², seed 42) those kernels return non-finite values for Z-Image's linears (the feed-forward output reaches `inf` from finite inputs at the first denoise step past t≈0.07) and every render comes out solid black, the same kernel defect family as `MOLD_QWEN_QMATMUL` ([#1048](https://github.com/utensils/mold/issues/1048)). The default dequantizes each weight to the activation dtype per forward, which renders correctly and is somewhat slower. Metal and CPU ignore it and keep `QMatMul`. Registered as an engine-shaping variable so scheduler fingerprints and learned estimates never mix the two arms.                                                                                                                                                                                                                                                                                                                                                                                                    |
| `MOLD_QWEN_FP8_CACHE`                 | off                                     | `1` keeps the widened BF16 copy of an FP8 Qwen-Image checkpoint's weights instead of re-casting them every forward ([#1045](https://github.com/utensils/mold/issues/1045)). Roughly doubles the transformer's resident VRAM — an FP8 build is normally chosen precisely because that headroom is missing, so this is opt-in and only worth it on a card with room to spare. It governs the in-memory transformer only: `--offload` streams its blocks through `candle_nn::Linear`, which has no FP8 arm at all, so those blocks are always widened once at load (roughly doubling their host-RAM footprint) whatever this is set to.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                  |
| `MOLD_H3_TURBO_ADAPTER`               | unset                                   | **Capture-scope UAT override, honored only by `h3-private-uat` builds.** Ordinary Turbo use selects a reviewed Turbo model tag instead (`minimax-h3-fl2va:comfy-pruned-int8-turbo-8step` or `…-turbo-4step-768p`), which resolves the manifest-pinned adapter with no environment configuration; a set pair in any other build is a hard error rather than a second, silently divergent selection authority. Filesystem path to a reviewed MiniMax H3 Turbo LoRA adapter file ([#1172](https://github.com/utensils/mold/issues/1172)). A Turbo tier is a LoRA overlaid on the _same_ compact INT8 ConvRot checkpoint, so no base artifact contract relaxes; what the tier adds is a distillation triple (sampler kind, terminal-inclusive step count, video shift) plus the adapter's own authenticated identity and resident cost. The file is authenticated against the pinned digest of the tier named by `MOLD_H3_TURBO_TIER`, which is why the two must be set together — a path with no tier cannot be authenticated, and a tier with no path has nothing to authenticate. Setting either alone is an error. Registered as an engine-shaping variable so scheduler fingerprints and learned estimates never mix a Turbo render with a base one. |
| `MOLD_H3_TURBO_TIER`                  | unset                                   | **Capture-scope UAT override, honored only by `h3-private-uat` builds** — see `MOLD_H3_TURBO_ADAPTER`. Which reviewed Turbo tier the file at `MOLD_H3_TURBO_ADAPTER` must be. Accepts the short alias `fl2v-8step` (9 grid points, 8 transformer evaluations, trained at 544p), `fl2v-4step-768p` (5 grid points, and the only tier that moves the video shift, to 6), or `ref2v-4step` (5 grid points), or the corresponding full stable id such as `minimax-h3.turbo-lora.fl2v-8step-v1.0.comfyui-bf16.v1`. Matching is case-insensitive. An unreviewed value is rejected and names the accepted set. Note mold counts terminal-inclusive grid points, so a "Turbo 8-step" checkpoint is 9 mold steps and exactly 8 forwards. Registered as an engine-shaping variable.                                                                                                                                                                                                                                                                                                                                                                                                                                                                             |

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

| Variable                            | Default             | Description                                                                                                                                                                                                                  |
| ----------------------------------- | ------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `MOLD_GPUS`                         | `all`               | `all`, `none` (maintenance), or comma-separated ordinals/stable `cuda:`/`metal:`/`GPU-`/`MIG-` IDs. Prefer IDs from `/api/devices` in persistent config. See [Multi-GPU](/guide/cli-reference#multi-gpu)                     |
| `MOLD_QUEUE_SIZE`                   | `200`               | Max queued generation jobs; overflow returns HTTP 503 with `Retry-After`                                                                                                                                                     |
| `MOLD_OUTPUT_DIR`                   | `~/.mold/output`    | Image output directory (set empty to disable)                                                                                                                                                                                |
| `MOLD_THUMBNAIL_WARMUP`             | —                   | `1` to prebuild gallery thumbnails at server startup (default: disabled)                                                                                                                                                     |
| `MOLD_WEB_DIR`                      | —                   | Override the web gallery SPA bundle location. First resolved path among this, `$XDG_DATA_HOME/mold/web`, `~/.mold/web`, `<binary dir>/web`, and `./web/dist` wins                                                            |
| `MOLD_DB_PATH`                      | `MOLD_HOME/mold.db` | Override the SQLite gallery metadata DB location                                                                                                                                                                             |
| `MOLD_DB_DISABLE`                   | —                   | `1` to disable the SQLite metadata DB entirely — server and CLI fall back to filesystem walks                                                                                                                                |
| `MOLD_GALLERY_TRASH_RETENTION_DAYS` | `30`                | Days a trashed print stays in `<output_dir>/.trash/` before the sweeper purges it; `0` keeps trashed prints forever (max 3650). Env override of the `gallery.trash_retention_days` key — see [Library trash](#library-trash) |
| `MOLD_CORS_ORIGIN`                  | —                   | Restrict CORS to specific origin                                                                                                                                                                                             |
| `MOLD_API_KEY`                      | —                   | API key for authentication (single key, comma-separated, or `@/path/to/keys.txt`)                                                                                                                                            |
| `MOLD_RATE_LIMIT`                   | —                   | Per-IP rate limit for generation endpoints (e.g., `10/min`, `5/sec`, `100/hour`)                                                                                                                                             |
| `MOLD_RATE_LIMIT_BURST`             | —                   | Burst allowance override (defaults to 2x rate, capped at 100)                                                                                                                                                                |
| `MOLD_MAX_CACHED_MODELS`            | `3`                 | LRU model-cache capacity (range `1..=16`). At most one entry stays GPU-resident; the rest are parked in CPU RAM. Out-of-range values warn and fall back to default.                                                          |
| `MOLD_CACHE_IDLE_TTL_SECS`          | `1800` (30 min)     | Idle timeout for parked cache entries (range `60..=86400`). Untouched entries are evicted past this TTL.                                                                                                                     |
| `MOLD_QUEUE_LOOKAHEAD_BUFFER`       | `8`                 | Server queue lookahead size (range `1..=64`). The dispatcher peeks this many jobs ahead to honour locality.                                                                                                                  |
| `MOLD_QUEUE_MAX_DEFERRALS`          | `3`                 | Per-job starvation budget (range `0..=32`). A job can be deferred this many times before forced pickup.                                                                                                                      |
| `MOLD_MALLOC_TRIM`                  | `1` (Linux/glibc)   | `0` disables the post-generation `malloc_trim(0)` call. Cheap (~ms) but Linux-only; reclaims arena pages after large GGUF+LoRA rebuilds.                                                                                     |
| `MOLD_FLUX_DELTA_CACHE`             | `1`                 | `0` disables the CPU-side FLUX LoRA delta cache (~25 GB host RAM on typical FLUX LoRAs). Disabling forces a sub-second `B@A·scale` recompute on each rebuild.                                                                |
| `MOLD_FLUX_KEEP_TRANSFORMER`        | `0`                 | `1` keeps the FLUX transformer GPU-resident across same-LoRA generations (saves a full GGUF+LoRA rebuild). Server force-drops it if VAE decode headroom is too tight at that resolution.                                     |

### Durable queue and shutdown

A queued generation survives a server restart: it is recorded in `mold.db`
before it is queued and replayed automatically at the next start, under its
original job id. `GET /api/capabilities` reports `queue.durable_queue` — false on a host with
server gallery output disabled, which cannot promise durability for anything —
and each row in `GET /api/queue` reports whether that particular job is durable —
a job with no gallery target, one carrying reference-upload media, or one
whose request exceeds the payload ceiling runs normally but is not replayed.

| Variable                           | Default             | Description                                                                                                                                                                                                                                                                                                                                                                   |
| ---------------------------------- | ------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `MOLD_QUEUE_JOURNAL_DISABLE`       | —                   | `1` turns the durable queue off entirely. Jobs still run; nothing survives a restart, and `queue.durable_queue` reports `false`.                                                                                                                                                                                                                                              |
| `MOLD_QUEUE_JOURNAL_MAX_BYTES`     | `33554432` (32 MiB) | Ceiling on one recorded request. A larger request (an inline video, say) runs normally and is reported `durable: false` rather than being half-persisted.                                                                                                                                                                                                                     |
| `MOLD_QUEUE_MAX_DISPATCH_ATTEMPTS` | `2`                 | How many times a worker may start a job before it is **held** instead of retried. Charged only when a worker actually claims the job, so a job that merely waits behind a long render through many restarts is never charged.                                                                                                                                                 |
| `MOLD_QUEUE_MAX_REPLAY_SEEN`       | `10`                | How many restarts may replay a job that never starts before it is **held**. Sized for a crash loop; ordinary deploys never approach it.                                                                                                                                                                                                                                       |
| `MOLD_QUEUE_ADOPT_OWNER`           | —                   | Adopt a specific orphaned queue by its owner id, printed in the startup warning. Only needed when several retained queues share one `MOLD_HOME` and none matches this server.                                                                                                                                                                                                 |
| `MOLD_SHUTDOWN_ABORT_SECS`         | `45`                | Hard deadline for the whole shutdown after `SIGTERM`. The running generation is aborted at its next checkpoint and requeued; queued work is retained and replayed. If shutdown overruns, `mold serve` ends the process rather than wait — a cold model load is not interruptible, and hanging past systemd's stop timeout is what used to get the server SIGKILLed mid-write. |

A **held** job is listed by `GET /api/queue` with `state: "held"` and a reason,
and is never started automatically — it is waiting for you to look at it.
Clear one with `DELETE /api/queue/{id}`.

### Sharing one MOLD_HOME between servers

Each server owns its queue through a record under
`$MOLD_HOME/queue-owners/`, so two servers on different ports never replay each
other's work. A server recognises its own queue after a restart, including
after a port change, in every case but one: if several retained queues are
present and none was last used by this server, it cannot tell which is its own.
It then starts with a fresh queue and warns at startup, naming each orphaned
owner with its last-known instance and its queued and held row counts. Adopt
one deliberately with:

```bash
MOLD_QUEUE_ADOPT_OWNER=<owner-id-from-the-warning> mold serve
```

There is one case the server deliberately gets wrong rather than leave work
stranded. If exactly **one** retained queue is present, nobody is running it,
and it was last used by a different server, the starting server adopts it and
says so loudly in the log. Nearly always that is correct — it is the same
server coming back on a changed port, which is precisely what the queue is
designed to survive, and what makes the "this job will finish on the host"
message clients show at shutdown true rather than a lie.

But a genuinely **new** second server, started while the first is stopped
against a `MOLD_HOME` that holds one retained queue, cannot be distinguished
from that and will adopt the queue too. It runs the other server's jobs. The
trade is deliberate: a silently stranded queue is worse than an announced
adoption, because nothing tells the user their job is never coming. If you are
adding a second server to an existing `MOLD_HOME`, start it while the first is
running, or drain the first server's queue before you do.

Under systemd, set the budget through the NixOS module rather than the
environment, so the unit's stop timeout stays derived from it:

```nix
services.mold.shutdown.abortSeconds = 45;  # TimeoutStopSec becomes 105s
```

Do not set `TimeoutStopSec=infinity`: with a durable queue the right response
to a wedged worker is to exit and replay, not to hang the deploy.

When the deadline expires the server exits with status 0 for an ordinary stop
that merely overran, and 1 for a shutdown triggered by a fatal CUDA error so
`Restart=on-failure` brings it back. The desktop app's built-in engine never
does this — it runs inside a process it does not own, so its budget only stops
it waiting.

### Upscaling

| Variable                 | Default | Description                                                    |
| ------------------------ | ------- | -------------------------------------------------------------- |
| `MOLD_UPSCALE_MODEL`     | —       | Default upscaler model for `mold upscale`                      |
| `MOLD_UPSCALE_TILE_SIZE` | —       | Tile size for memory-efficient upscaling (0 to disable tiling) |

### Auth

| Variable        | Default | Description                                                                             |
| --------------- | ------- | --------------------------------------------------------------------------------------- |
| `HF_TOKEN`      | —       | Default Hugging Face token for gated models; web Settings can override it until cleared |
| `CIVITAI_TOKEN` | —       | Default Civitai token for gated models; web Settings can override it until cleared      |

### Third-party model licenses

Some auxiliary weights carry terms that Mold's MIT license does not cover. The
InsightFace antelopev2 face models that PuLID identity conditioning needs are
licensed for non-commercial research only, so Mold refuses to download them —
from `mold pull`, the server's auto-pull, or any client-triggered download —
until you record acceptance once per `MOLD_HOME`:

```bash
mold pull pulid-flux --accept-license insightface-antelopev2
```

The command prints the restriction and both terms URLs before it writes the
record. Acceptances live in owner-only `$MOLD_HOME/license-acceptances.json`.

The terms are pinned to an **exact upstream commit**, not to a branch: Mold
stores the commit-addressed URL of the license text alongside its SHA-256, and
verified that pair when the pin landed. A commit URL serves the same bytes
forever, so the digest can never drift away from the document it describes —
which a `master` link would, quietly leaving you consented to text that had
since been rewritten. Each acceptance is bound to that `(url, sha256)` pair, so
a Mold release that re-pins a license to a newer upstream revision invalidates
your existing acceptance and asks again with the new text.

Recording an acceptance is entirely offline — Mold never fetches the license
text, so `--accept-license` works on an air-gapped host. A refused download
names the license and the exact command to run; there is no
environment-variable bypass. See `THIRD_PARTY_NOTICES.md` for the full notice,
including the pinned commit and digest.

#### Which machine records the acceptance

The acceptance has to live on the machine that does the downloading, and
`mold pull` follows the pull:

- **A server answers at `MOLD_HOST`** (including the local `mold serve`): the
  id is sent with the request and the **server** writes it into its own
  `$MOLD_HOME`. Nothing is recorded on the calling machine.
- **No server, or `--local`**: the pull runs here, so the acceptance is
  recorded here.

Either way the terms are printed before the request goes out — and they are the
terms of the machine that will record them. When a server will do the
recording, `mold pull` reads that server's `GET /api/licenses`, displays what
it returned, and sends back exactly that; the server refuses anything else.
This matters because the two sides can be on different Mold releases pinning
different revisions of the same license, and consent has to mean the document
that was actually shown.

Run `mold licenses` to see what needs accepting and which root was read:

```bash
mold licenses           # asks the server at MOLD_HOST when one answers
mold licenses --local   # this machine's own acceptances, without asking
```

`mold licenses` reads this machine only when nothing is listening at
`MOLD_HOST`, or when you pass `--local`. If a server is there but the request
fails — authentication, a 5xx, an unreadable body — the command reports that
against the host rather than quietly showing you a different machine's
acceptances.

Over HTTP, `GET /api/licenses` returns each license with `accepted` and
`required_by`, and `POST /api/downloads` / `POST /api/models/pull` accept an
additive `accept_licenses` array of `{ id, url, sha256 }` entries. A gated
download without one is refused with `403` and code `LICENSE_NOT_ACCEPTED`;
terms the server does not pin are refused with `409` and code
`LICENSE_TERMS_MISMATCH`. Both carry a structured `license` object — including
the server's own `url` and `sha256` — that a UI can build its own prompt from. Servers that support this advertise
`capabilities.licenses: true`; older ones do not, and can only be accepted by
running `mold pull --accept-license` in a shell on that host.

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

At each open, mold runs SQLite's `quick_check`. If SQLite reports a corrupt or
non-database file at startup—or an indexed gallery query discovers corruption
later—mold serializes recovery across local mold processes, copies the database and
any WAL/SHM sidecars to `mold.db.corrupt-<timestamp>*`, replaces the live schema
through SQLite's coordinated online-backup API, and rebuilds gallery rows from
the files in `MOLD_OUTPUT_DIR`. The quarantined files remain available
for manual inspection or salvage. Because the same database also contains user
preferences and prompt history, those values reset unless they are manually
recovered from the quarantined copy.

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

### Library trash

Deleting a print from the Library (desktop, web, iPhone, TUI, or
`DELETE /api/gallery/image/:filename`) moves it to the host's trash instead
of removing it: the file goes to `<output_dir>/.trash/<filename>` next to a
small `<filename>.trash.json` tombstone, the gallery row keeps its title,
tags, favorite flag, and collection membership, and the print can be
restored until the retention sweep purges it. The sweep runs at server
startup and hourly; `mold trash sweep` and `POST /api/gallery/trash/sweep`
run it on demand. Appending `?permanent=true` to the delete (or "Delete
forever" in a client) skips the trash.

| Key                            | Env var                             | Default | Description                                                                                     |
| ------------------------------ | ----------------------------------- | ------- | ----------------------------------------------------------------------------------------------- |
| `gallery.trash_retention_days` | `MOLD_GALLERY_TRASH_RETENTION_DAYS` | `30`    | Days a trashed print is kept before it is purged. `0` keeps trashed prints forever; max `3650`. |

```bash
mold config set gallery.trash_retention_days 7     # purge a week after trashing
mold config set gallery.trash_retention_days 0     # keep trashed prints until emptied by hand
mold config where gallery.trash_retention_days     # → db
```

The key lives in `mold.db` (profile-scoped, `gallery.` prefix) and is read
fresh on every sweep, so a change takes effect without a restart. It is
**per host**: each `mold serve` keeps its own trash and its own retention, and
the desktop, web, and iPhone apps edit a remote machine's value through that
host's `/api/config`. `GET /api/capabilities` advertises the effective value
as `gallery.trash.retention_days`; with `MOLD_DB_DISABLE=1` there is no trash
and `capabilities.gallery.trash` is absent, so delete is permanent as before.

### Auto-tagging titled prints

| Key                       | Env var | Default | Description                                                              |
| ------------------------- | ------- | ------- | ------------------------------------------------------------------------ |
| `generate.auto_tag_title` | —       | `true`  | Whether a titled CLI or TUI run also tags the print with its title slug. |

```bash
mold config set generate.auto_tag_title false   # stop tagging titled prints
mold config where generate.auto_tag_title       # → db
```

With it on, `mold run "a village" --title "Smurf village"` also files the
print under the tag `smurf-village` and discloses it on stderr
(`filing under tag "smurf-village"`). `mold run --no-auto-tag` overrides the
setting for one invocation.

This is deliberately a **client** setting with no env override: it shapes what
the CLI puts in a request, not how a server behaves. The server never
auto-tags — it cannot tell a title a person typed from one a script
generated, and a host quietly adding tags to every print that crossed it
would be surprising from every other machine on the fleet.

## Advanced

### Device and Path Overrides

| Variable                             | Default | Description                                                                        |
| ------------------------------------ | ------- | ---------------------------------------------------------------------------------- |
| `MOLD_DEVICE`                        | —       | Force device placement, currently `cpu` for debugging                              |
| `MOLD_TRANSFORMER_PATH`              | —       | Override transformer weights path                                                  |
| `MOLD_LOW_NOISE_TRANSFORMER_PATH`    | —       | Override the low-noise expert of a Wan 2.2 A14B pair                               |
| `MOLD_VAE_PATH`                      | —       | Override VAE weights path                                                          |
| `MOLD_SPATIAL_UPSCALER_PATH`         | —       | Override LTX spatial upscaler path                                                 |
| `MOLD_TEMPORAL_UPSCALER_PATH`        | —       | Override LTX temporal upscaler path                                                |
| `MOLD_DISTILLED_LORA_PATH`           | —       | Override the default LTX-2 distilled LoRA path                                     |
| `MOLD_LOW_NOISE_DISTILLED_LORA_PATH` | —       | Override the distill for a Wan A14B low-noise expert                               |
| `MOLD_T5_PATH`                       | —       | Override T5 encoder path                                                           |
| `MOLD_CLIP_PATH`                     | —       | Override CLIP-L encoder path                                                       |
| `MOLD_CLIP2_PATH`                    | —       | Override CLIP-G encoder path for SDXL                                              |
| `MOLD_T5_TOKENIZER_PATH`             | —       | Override T5 tokenizer path                                                         |
| `MOLD_CLIP_TOKENIZER_PATH`           | —       | Override CLIP-L tokenizer path                                                     |
| `MOLD_CLIP2_TOKENIZER_PATH`          | —       | Override CLIP-G tokenizer path for SDXL                                            |
| `MOLD_TEXT_TOKENIZER_PATH`           | —       | Override generic text tokenizer path for Qwen/Z-Image                              |
| `MOLD_DECODER_PATH`                  | —       | Override Wuerstchen decoder weights path                                           |
| `MOLD_QWEN2_VARIANT`                 | `auto`  | Qwen-family Qwen2.5-VL encoder: `auto`, `bf16`, `q8`, `q6`, `q5`, `q4`, `q3`, `q2` |
| `MOLD_QWEN2_TEXT_ENCODER_MODE`       | `auto`  | Qwen-family placement mode: `auto`, `gpu`, `cpu-stage`, `cpu`                      |

These are mainly useful for custom local model layouts, manual debugging, or
testing alternative weight files without editing `config.toml`.

### Per-component device placement

Override which device (CPU or a specific GPU) runs each part of the diffusion
pipeline. All variables accept `auto` (preserve the engine's VRAM-aware
default), `cpu`, `gpu` (= `gpu:0`), or `gpu:N` for a process-local ordinal.
They also accept the exact opaque ID reported by `GET /api/devices`, such as
`cuda:0123…` or `metal:default`, either directly or as
`device:cuda:0123…`. Durable IDs survive ordinal reordering; raw NVIDIA
`GPU-`/`MIG-` UUID selectors are reserved for `--gpus` / `MOLD_GPUS` startup
selection.

| Variable                   | Applies to                        | Notes                                                                                                                                               |
| -------------------------- | --------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------- |
| `MOLD_PLACE_TEXT_ENCODERS` | Every model family (Tier 1)       | Single knob that moves every text encoder slot as a group. Picking `cpu` frees the transformer's full VRAM budget without triggering block offload. |
| `MOLD_PLACE_TRANSFORMER`   | FLUX, Flux.2, Z-Image, Qwen-Image | Per-component override. Interacts with `MOLD_OFFLOAD` — resident and streamed blocks target the chosen ordinal.                                     |
| `MOLD_PLACE_VAE`           | FLUX, Flux.2, Z-Image, Qwen-Image | Decode stage; CPU is fine for preview, GPU is faster.                                                                                               |
| `MOLD_PLACE_T5`            | FLUX                              | Per-encoder override; unset falls through to `MOLD_PLACE_TEXT_ENCODERS`.                                                                            |
| `MOLD_PLACE_CLIP_L`        | FLUX                              | Per-encoder override.                                                                                                                               |
| `MOLD_PLACE_CLIP_G`        | SDXL and others that use CLIP-G   | Per-encoder override.                                                                                                                               |
| `MOLD_PLACE_QWEN`          | Flux.2, Z-Image, Qwen-Image       | Per-encoder override for the Qwen text encoder.                                                                                                     |

For local CLI generation, precedence (highest wins) is CLI flag
(`--device-text-encoders`, `--device-vae`, …) → environment variable →
`[models."name:tag".placement]` TOML block → engine auto. For server requests,
an explicit request placement is the complete placement decision; otherwise
environment values override the persisted per-model placement and unspecified
components remain `auto`. The server normalizes this once before admission, so
validation, scheduling, and inference consume the same placement.

Scheduler V2 resolves that normalized shape into a concrete admission plan per
eligible device before dispatch. Plans include exact artifact paths and identity
fingerprints, materialized placement, inferred precision/quantization metadata,
planned load/offload mode, sampled free-VRAM peak, and incremental host RAM.
Explicit CPU/device values never become scoring hints: an unavailable device or
components pinned across different GPUs blocks the request. Automatic CPU
placement is considered only under measured/static memory pressure and only for
a family/component path Mold implements. The GPU owner validates the selected
device and artifacts again before CUDA work; a changed artifact invalidates the
plan instead of being silently substituted.

Scheduler observations keep setup separate from execution. Typed cold-load,
warm-reload, prompt-encode, denoise, VAE, and upscale timings feed bounded
learned estimates; metadata schema v15 persists runtime independently so a
candidate receives exactly its cold or warm setup charge. Multi-host Create
uses `POST /api/generate/placement-preview` as a read-only final feasibility
check for ordinary generation. A planned response can name known encoder
dependencies in `pending_downloads` — and, for a request that conditions on a
face, the five PuLID identity assets under the `identity_adapter`,
`identity_vision_encoder`, `face_detector`, `face_recognizer`, and
`face_parser` kinds; those downloads and the low-confidence
estimate include only devices selected by that candidate plan. The preview does not
fetch them, uses a separate registry-identity fingerprint, and admission
recomputes the plan after the files land. Cold installed Civitai and Hugging
Face IDs resolve from contained local sidecars, with their synthesized runtime
configuration carried through scheduling and final GPU validation even if the
server refreshes its model list. An infeasible response can name absent
manifest files in `missing_components` so clients can explain the repair
instead of discarding the server's reason. Current chain and local
prompt-expansion/post-generation-upscale utility previews deliberately return
non-authoritative `unsupported`: those paths are not advertised as exact until
their real device/CPU fallbacks are represented.

Forced-local batches (`mold run --local --batch N`) use the same deterministic
assignment core across all GPUs selected by `--gpus`/`MOLD_GPUS`. There is no
two-GPU limit; a one-item run keeps the existing best-free-GPU selection.

The web UI's **Placement** panel, the desktop app's **Settings → Advanced**
placement editor, the `GET`/`PUT`/`DELETE /api/config/model/:name/placement`
routes (read a saved default, save one, clear one — `GET` returns `404` when
none is saved), and `mold run --device-*` flags all write/read the same shape,
so any surface can drive it.

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
