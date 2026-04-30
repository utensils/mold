# CLAUDE.md

Guidance for Claude Code working in this repo. Keep lean: only things not obvious from the code, `--help`, or `git log`.

## What mold is

Local AI image/video generation CLI built on [candle](https://github.com/huggingface/candle). Supports FLUX, SD1.5, SDXL, SD3.5, Z-Image, Flux.2 Klein, Qwen-Image, Wuerstchen v2, LTX-Video, and LTX-2. Runs locally on GPU or talks to a remote `mold serve` over HTTP. Single binary, everything feature-gated.

## Commands

```bash
# Nix (preferred)
nix build                   # Build mold (default CUDA/Metal)
nix fmt                     # treefmt (nixfmt + rustfmt)
nix flake check             # CI-equivalent gate

# Cargo — common loops
cargo check                                                                    # type check
cargo clippy --workspace --all-targets -- -D warnings                          # what CI runs
cargo fmt --all -- --check                                                     # what CI runs
cargo test --workspace                                                         # what CI runs
cargo check -p mold-ai --features preview,discord,expand,tui,webp,mp4         # what CI also runs
./scripts/coverage.sh [--html]                                                 # coverage

# Local dev run (MUST prefix with ensure-web-dist so the embedded SPA isn't a stub)
./scripts/ensure-web-dist.sh && cargo run --profile dev-fast -p mold-ai \
  --features metal,preview,expand -- run "a cat"
```

Inside `nix develop` the devshell exposes shortcuts (`build`, `build-release`, `serve`, `mold`, `clippy`, `run-tests`, `coverage`, `fmt`). Run `type <cmd>` to see the underlying invocation.

**CI gates** (`.github/workflows/ci.yml`): `rust` (fmt + check + clippy-deny-warnings + test + feature-combo check), `coverage` (cargo-llvm-cov → Codecov), `docs` (`bun run fmt:check && bun run verify && bun run build` in `website/`). All must pass.

## Crates

```
crates/
├── mold-core/        Shared types, HTTP client, config, manifest, validation, download
├── mold-db/          SQLite (rusqlite, bundled, WAL) — gallery, settings, model_prefs, prompt_history
├── mold-inference/   Candle engines per family (FLUX, SD1.5/XL/3, Z-Image, Flux.2, Qwen-Image, Wuerstchen, LTX-Video, LTX-2)
├── mold-server/      Axum HTTP server (consumed as lib by mold-cli)
├── mold-cli/         The `mold` binary (clap)
├── mold-discord/     Discord bot (poise + serenity), HTTP-only dep on mold-core
└── mold-tui/         Interactive TUI (ratatui)
```

**Directory ≠ package name.** Use these with `-p`:

| Dir | Package |
|---|---|
| `mold-cli/` | `mold-ai` (binary: `mold`) |
| `mold-core/` | `mold-ai-core` |
| `mold-db/` | `mold-ai-db` |
| `mold-inference/` | `mold-ai-inference` |
| `mold-server/` | `mold-ai-server` |
| `mold-discord/` | `mold-ai-discord` |
| `mold-tui/` | `mold-ai-tui` |

**MSRV**: 1.85.

**Feature flags** (`mold-cli`): `cuda`, `metal`, `preview`, `discord`, `expand`, `tui`, `metrics`, `webp`, `mp4`. GPU features forward through to `mold-inference`. H.264 decode is baseline for LTX-2 source ingest; `mp4` only gates AAC mux.

## Non-obvious architectural patterns

Most are in `mold-inference`. When touching engines, these are the rules that matter:

- **Lazy load** — engines load on first `generate()`, not startup. Most hold mmap'd safetensors.
- **Drop-and-reload text encoders** — T5/CLIP/Qwen3 are dropped from GPU after encoding so the transformer has VRAM to denoise, then reloaded next request.
- **Dynamic device placement** — text encoders go to GPU or CPU based on remaining VRAM after the transformer loads (thresholds: `device.rs`).
- **Quantized encoder auto-fallback** — when FP16/BF16 doesn't fit, the largest GGUF variant that fits is auto-selected. Override: `--t5-variant` / `--qwen3-variant` / `--qwen2-variant` or `MOLD_*_VARIANT`.
- **Block-level offloading** (FLUX) — `flux/offload.rs` streams transformer blocks CPU↔GPU one at a time: ~24 GB → 2–4 GB VRAM, 3–5× slower. Auto-enabled under pressure; force with `--offload` / `MOLD_OFFLOAD=1`.
- **LoRA backend is custom** — candle has no LoRA. BF16 path: `LoraBackend` (a `SimpleBackend`) intercepts `vb.get()` during model construction and applies `W' = W + scale·(B @ A)` inline. GGUF path: `gguf_lora_var_builder()` selectively dequantizes affected tensors, merges, re-quantizes. Both work with offloading. See `flux/lora.rs`.
- **LoRA caching** — `LoraDeltaCache` (pre-computed `B @ A · scale` on CPU, ~80–120 MB) survives transformer rebuilds. `LoraFingerprint` on `FluxEngine` skips redundant rebuilds when the same LoRA/scale reappears.
- **Shared tokenizer pool** — `shared_pool.rs`: `Arc<Tokenizer>` keyed by file path, shared across engines via `create_engine_with_pool()`. Saves ~100–150 ms on model swap for FLUX variants.
- **CPU-based noise** — `seeded_randn()` in `engine.rs` generates initial noise on CPU via `StdRng`/ChaCha20, then moves to GPU. This is load-bearing for cross-backend seed determinism (CUDA/Metal/CPU produce identical images).
- **Z-Image has a bespoke quantized transformer** — `zimage/quantized_transformer.rs` lives here (not candle); GGUF tensor naming differs from BF16 (`attention.qkv` vs split Q/K/V, etc.).
- **LTX-2 is CUDA-only for real generation.** CPU is correctness-only; Metal is unsupported. Native runtime lives in `ltx2/` with its own media pipeline (MP4 first, real AAC).

## Inference modes (`mold run`)

1. **Remote** (default) — HTTP to `$MOLD_HOST` (default `http://localhost:7680`).
2. **Local fallback** — server unreachable → local GPU (auto-pulls model if missing).
3. **Forced local** — `--local` skips the server attempt.

`mold run [MODEL] [PROMPT]` disambiguates the first positional at runtime: matches a known model name → model, otherwise → prompt.

**Pipe-friendly**: `echo "a cat" | mold run flux2-klein | viu -`. stdin for prompt, stdout for image bytes when not a TTY. `--output -` forces stdout; `--image -` reads source from stdin. `IsTerminal` detection + SIGPIPE reset to default + `status!` macro route text to stderr.

**Name resolution** (`manifest::resolve_model_name`): `model:tag` (e.g. `flux-dev:q4`); bare names try `:q8` → `:fp16` → `:bf16` → `:fp8`; legacy dash `flux-dev-q4` resolves to colon form.

## Multi-prompt chain authoring

- `mold run --script shot.toml` — canonical TOML, schema `mold.chain.v1`. Per-stage `prompt` / `frames` / `transition`.
- `mold chain validate shot.toml` or `mold run --script ... --dry-run` to inspect without submitting.
- Sugar: `mold run <model> --prompt "..." --prompt "..." --frames-per-clip 97` (uniform smooth only).
- Transitions: `smooth` (default, motion-tail morph), `cut` (fresh latent), `fade` (cut + RGB crossfade).
- Per-stage source images: `source_image_path` (relative to script file) or `source_image_b64`. Resolved by `mold_core::chain_toml::read_script_resolving_paths`.

## Config

Two stores, one logical `Config` view:

| Surface | Owns |
|---|---|
| `~/.config/mold/config.toml` (XDG) or `~/.mold/config.toml` (legacy) | Bootstrap: paths, ports, credentials, `[logging]`, `[runpod]`, per-model component paths |
| `mold.db` `settings` + `model_prefs` | User prefs: `expand.*`, `generate.*`, `tui.*`, per-model defaults |
| `MOLD_*` env vars | Runtime override (highest precedence) |

Every `main()` calls `mold_db::config_sync::install_config_post_load_hook()`, which runs a one-shot idempotent `config.toml → DB` migration on first boot (renames original to `config.toml.migrated`) and overlays DB onto every `Config::load_or_default()`. Consumers still read `cfg.expand.*` unchanged.

`mold config set <key> <val>` routes by key prefix (`expand.*` → DB, `models_dir` → TOML). `mold config where <key>` prints the surface. `mold config list --json` tags each row `[db]` / `[file]` / `[env]`. Multi-profile: `settings` and `model_prefs` are keyed on `(profile, key)`; active profile resolves `MOLD_PROFILE` → `settings.profile.active` → `"default"`.

## Metadata DB

`MOLD_HOME/mold.db` (override: `MOLD_DB_PATH`; disable: `MOLD_DB_DISABLE=1`). Current `SCHEMA_VERSION` lives in `crates/mold-db/src/migrations.rs`. Tables: `generations` (gallery rows), `settings` (KV, profile-scoped), `model_prefs` (per-resolved-model generation params, profile-scoped), `prompt_history`. Migrations are forward-only via `PRAGMA user_version`; add to `MIGRATIONS[]` and bump `SCHEMA_VERSION`.

Gallery writes happen in the server (`queue.rs` upserts after disk write; background `reconcile(output_dir)` on startup), CLI (`crates/mold-cli/src/metadata_db.rs` via `record_local_save`), and TUI (`gallery_scan.rs`). DB is additive — embedded PNG/JPEG `mold:parameters` still get written, and open/upsert failures log and keep working.

## Web UI

SPA in `web/` (Vue 3 + Vite 7 + Tailwind v4). Embedded into the `mold` binary at compile time via `rust-embed`. `crates/mold-server/build.rs` resolves the bundle from `$MOLD_WEB_DIST` → `web/dist` → a placeholder stub (detected at runtime and swapped for an inline page). The devshell `build`/`serve`/`generate` commands run `./scripts/ensure-web-dist.sh` first, so dev builds ship the real SPA. `MOLD_WEB_DIR` still overrides the embedded bundle at runtime for `bun run dev` hot-iteration.

**Gallery delete is always enabled.** `DELETE /api/gallery/image/:filename` is a destructive endpoint — pair with `MOLD_API_KEY` when the server is exposed beyond localhost. `GET /api/capabilities` still returns `{ gallery: { can_delete: true } }` so older clients keep a stable shape.

## Workflow

- **TDD.** Every bug fix and feature: failing test first, then code. Prefer unit tests on exported contracts (key→action maps, focus transitions, serialization round-trips, layout invariants) over E2E. Layout constants need a test that asserts the inner area fits the rendered row count — otherwise they drift.
- **Keep in sync when models / CLI flags / env vars / endpoints change:** `CHANGELOG.md` (Keep-a-Changelog format, under `[Unreleased]`), `.claude/skills/mold/SKILL.md` (used by other agents), `website/` (VitePress docs).
- **Don't break centered TUI gallery thumbnails.** `crates/mold-tui/src/ui/gallery.rs` uses a fixed-protocol thumbnail path. Do not revert the grid to plain `StatefulImage` for Kitty/Sixel/iTerm2 — it reintroduces top-left-padded thumbnails instead of centered aspect-correct ones. Keep the regression tests passing.

## Key design decisions

1. **Crate boundaries are clean** — `mold-cli` doesn't depend on candle; `mold-server` doesn't depend on clap; `mold-discord` only depends on `mold-core`.
2. **candle over tch/ort** — pure Rust, no libtorch. Uses a published fork (`candle-*-mold` on crates.io) for Metal quantized matmul precision + seed buffer size fixes.
3. **Single binary** — `mold` includes `serve` via `mold-server` library; GPU flags forward `mold-cli` → `mold-server` → `mold-inference`.
4. **`tokio::sync::Mutex` + `spawn_blocking`** — single-model-at-a-time fits GPU workloads. `AppState.model_cache` is an LRU (max 3) with `ModelResidency { Gpu, Parked, Unloaded }`; at most one engine is GPU-resident.
5. **Nix flake (flake-parts + crane)** — CUDA 12.8 on Linux (default `CUDA_COMPUTE_CAP=89` Ada; `mold-sm120` for Blackwell; `mkMold` for any), Metal on macOS. Devshell sets `CPATH`/`LIBRARY_PATH`/`LD_LIBRARY_PATH` for CUDA compilation.
6. **Shell completions** — static via `clap_complete` + dynamic via `CompleteEnv` with `ArgValueCandidates` for model names.
