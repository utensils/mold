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

**CI gates** (`.github/workflows/ci.yml`): `rust` (fmt + check + clippy-deny-warnings + test + feature-combo check), `coverage` (cargo-llvm-cov → Codecov), `docs` (`bun run fmt:check && bun run verify && bun run build` in `website/`), and a path-gated `release` check for release-plz synchronization scripts. All triggered gates must pass.

## Crates

```
crates/
├── mold-core/        Shared types, HTTP client, config, manifest, validation, download
├── mold-catalog/     Live HF + Civitai model-discovery proxy (5-min in-proc cache, no bulk-scrape DB). Depended on by mold-cli + mold-server only — mold-discord and mold-tui MUST NOT transitively depend on it.
├── mold-db/          SQLite (rusqlite, bundled, WAL) — gallery, settings, model_prefs, prompt_history
├── mold-inference/   Candle engines per family (FLUX, SD1.5/XL/3, Z-Image, Flux.2, Qwen-Image, Wuerstchen, LTX-Video, LTX-2)
├── mold-server/      Axum HTTP server (consumed as lib by mold-cli)
├── mold-cli/         The `mold` binary (clap)
├── mold-discord/     Discord bot (poise + serenity), HTTP-only dep on mold-core
└── mold-tui/         Interactive TUI (ratatui)
```

**Directory ≠ package name.** Use these with `-p`:

| Dir               | Package                    |
| ----------------- | -------------------------- |
| `mold-cli/`       | `mold-ai` (binary: `mold`) |
| `mold-core/`      | `mold-ai-core`             |
| `mold-catalog/`   | `mold-ai-catalog`          |
| `mold-db/`        | `mold-ai-db`               |
| `mold-inference/` | `mold-ai-inference`        |
| `mold-server/`    | `mold-ai-server`           |
| `mold-discord/`   | `mold-ai-discord`          |
| `mold-tui/`       | `mold-ai-tui`              |

**MSRV**: 1.85.

**iPhone app** (`apps/mobile/src-tauri`, frontend entry in `desktop/src/mobile/`): a standalone thin Tauri crate, excluded from the root workspace so the desktop GPU/server dependency tree is never cross-compiled for iOS. It is always remote-only (`HostView.kind = "remote"`, `primary = false`) and owns Generate, a merged remote Gallery, Catalog, Hosts/host detail, and a header-pushed Settings screen. Reuse `desktop/src/lib` API types, explicit-target HTTP/SSE helpers, capability/request form, source fitting, gallery media, catalog, and theme logic; never import a desktop store that assumes a local **This device** engine.

Host metadata and selection live in WebView storage, but per-host API keys must remain in the native iOS Keychain under `com.utensils.mold.remote-api-key`. Host entry accepts LAN/IP/DNS/HTTPS and Tailscale MagicDNS names; Tailscale support is the existing tailnet/DNS path, not an embedded SDK. Native Apple DNS-SD discovers `_mold._tcp` under Local Network permission. Authenticated image/video elements must use `POST /api/gallery/media-token` for a 15-minute, exact-path read ticket; never put the durable API key in a URL or buffer an entire video as the legacy fallback.

Generate shares the desktop capability/request contract, independently queues and cancels batch siblings, and keeps proportional resolution plus explicit Random/Fixed seed controls. Gallery must keep native video playback and horizontal print navigation with explicit **Use as prompt** / **Use as source** actions. Catalog Pull must immediately progress Connecting → Starting → Queued → Pulling N%, subscribe and reduce the opening download snapshot before POST, suppress duplicate POSTs, and retain the returned job-id association across catalog/source/display-name differences. Host detail owns telemetry, models-disk, queue, downloads, loaded models, and installed inventory.

Mobile settings persist at `mold.mobile.settings.v1`, default fresh installs to System + Safelight, preserve valid saved choices, and use shared `src/lib/theme.ts`; apply the theme before mount. The serialized `set_mobile_appearance` bridge must coalesce rapid changes and update UIKit traits/status-bar appearance to the final value. Preserve the iPhone interaction invariants: safe-area layout, 44pt controls, editable text at least 16px, `maximum-scale=1` / `user-scalable=no`, shell `touch-action: manipulation` and `overscroll-behavior: none`, with only the gallery viewer's scoped pan/swipe behavior reserved for navigation. iPad remains a responsive secondary target.

Devshell: `ios-dev` / `ios-run` / `ios-check` / `ios-build`; direct helper: `scripts/ios.sh` (`init`, `simulator` also exist). CI: `.github/workflows/ios.yml`; eligible successful `main` runs chain to `.github/workflows/testflight-ios.yml` (despite its name there is no cron). Release completion means App Store Connect `VALID` plus `Mold Internal` tester-membership verification; `testflight-ios-verify.yml` resumes an already uploaded exact bundle version. The Vite build must rename its separate source entry to `dist-mobile/index.html`, and `scripts/tests/ios-release-assets.sh` guards that entry and the opaque Apple icon catalog; regenerate icons with `scripts/generate-ios-icons.sh`. Release automation synchronizes the mobile crate/lockfile/Tauri config version; the archive build supplies its exact bundle build number. `apps/mobile/README.md` is the detailed maintainer reference.

**Desktop app** (`desktop/`): Tauri 2 macOS/Linux app (Vue 3 + TS, "Safelight" design language), package `mold-desktop` — its own cargo root, `exclude`d from the workspace. It reuses a running local `mold serve` on :7680 or embeds an authenticated, wildcard-bound, mDNS-advertised `mold-ai-server` (`metal` on macOS, `cuda` on Linux; both off by default so tests are CPU-only); that built-in engine is permanently the app's internal primary (**This device**) — remote servers are only ever list entries, never a primary the app switches to (`settings::migrate_remote_primary` re-homes old remote-primary installs on boot: mode→local, ex-primary becomes a connected host, key moves to its per-host slot, generation pinned to it). Local and remote use one HTTP+SSE wire contract. Platform chrome/bundling lives in `tauri.{macos,linux}.conf.json`; Linux provides `mold-desktop` (sm_89), `mold-desktop-sm120`, and an AppImage from `desktop-build`, while in-app updates remain macOS-only. It also owns native-only integrations: image clipboard/context menus, a unified multi-host gallery (every connected host + this device merged into one grid — filter chips, per-tile host badges, per-origin delete/save/reveal, host-keyed media blob cache; no separate IPC bucket — the local primary's `/api/gallery` is this device's gallery; the All view collapses cross-host copies of one print by filename, and by seed+byte-size identity for legacy copies — auto-saved remote outputs keep the origin's gallery filename and recorded metadata via the complete event's additive `filename`/`metadata` fields), persistent 80–130% webview scaling, file-backed app secrets (owner-only `secrets.json` under app data — including the persistent `desktop-local-api-key`; deliberately NOT the macOS Keychain, whose prompts users found obnoxious; don't reintroduce `keyring`), a remembered-hosts MRU with per-host API keys (`remote-api-key.<host-id>`; identity stays the URL slug, but entries dedupe by the server's instance UUID — `/api/status.instance_id`, mDNS `id` TXT — with display names from the server hostname, so one box reached by hostname/mDNS/IP collapses to one row), a single **Settings → Hosts** panel (This-device card, Add host, Connected, Remembered, On your network — no engine modes or "Use this host"), boot-time reconnect of every host in `connectedHostIds` (unreachable hosts stay listed as errored rows the status poll self-heals), a per-host detail view (click a sidebar host: live GPU/CPU/RAM, `models_disk` storage, queue, installed models), a single unified **Catalog** models view (one list merging the live catalog with installed models — installed rows deduped by name, host-tagged, sorted first; source chips All/HuggingFace/Civitai/**Installed** with All the default, the Installed chip rendering the full-featured installed rows; NSFW entries tagged while the include-NSFW checkbox remains; All/Images/Video media chips via `?type=`; downloads pinned top with source glyph + target host), multi-host queueing (jobs carry per-host API targets and the Generate view waits for the complete per-host `/api/models` union before showing the empty state, then routes via model-aware Auto/least-busy — hosts that already have the selected model win — a sticky pick, or "Most capable" (`generateTargetHost = "capable"` sentinel: backend CUDA > Metal > unknown, then VRAM, then queue, using the additive `gpu_info.backend` status field with GPU-name inference for older servers)), and RunPod pod/network-volume provisioning. Devshell: `desktop-dev` / `desktop-build` / `desktop-check` / `desktop-test` / `desktop-ui` / `desktop-bun-lock`; CI: `.github/workflows/desktop.yml`. Its version tracks the workspace version (synced automatically on the release PR by `scripts/release/sync-release-pr.sh`; `tauri.conf.json` has no version field on purpose). Gotcha: the tauri dev watcher does not rebuild on `crates/` changes — relaunch `desktop-dev` to pick up engine changes. In Vue stores, `reactive()`-wrap any object mutated from SSE/closure callbacks (raw-object mutations bypass proxy traps and freeze the UI).

**Desktop updater:** signed release builds support persisted **Stable** and **Nightly** channels. Startup and menu/Settings checks are check-only; an available update is announced in persistent app chrome and by a native notification while backgrounded, and installation requires the user's explicit **Update and restart** action. Stable reads the public `mold-desktop-stable.json` manifest from the latest tagged release; Nightly reads `mold-desktop-nightly.json` from the rolling `latest` prerelease, which `.github/workflows/desktop.yml` rebuilds only after the frontend and Rust gates pass for a desktop-relevant `main` commit. Before touching the installed app, Tauri verifies the Minisign signature and Mold fully extracts the archive to temporary storage, binds its bundle ID/version to the manifest, runs strict Apple code-signature and Gatekeeper checks, validates the current app identity and install location, and proves the app's parent directory is replaceable. Only then does Mold atomically exchange the staged and installed bundles with macOS `RENAME_SWAP` and restart. There is deliberately no post-launch watchdog, probation, or Mold-owned rollback; the update either passes preflight and installs or fails while the running version remains in place. Do not add a silent downgrade comparator: selecting Stable from a newer Nightly waits for a later stable version. Distribution CI requires `TAURI_SIGNING_PRIVATE_KEY` and `TAURI_SIGNING_PRIVATE_KEY_PASSWORD` in addition to the Apple signing/notarization secrets, verifies signatures against the checked-in public key, and retains ten Nightly desktop generations. The updater private key must have controlled offline backup and must never enter the repo or logs; losing it strands installed clients, and rotation must first be delivered by an update signed with the existing key.

**Desktop model-library invariant:** the Installed shelf and download progress are all-host views, host detail mirrors the selected host's pulls, and built-in manifest variants take precedence over ambiguous multi-checkpoint Hugging Face repositories so a Pull targets one runnable model rather than an aggregate repository. Every saved host is attempted immediately on launch in parallel with This Mac; legacy `connectedHostIds` may preserve ordering but must never suppress a remembered host.

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
- **Tier 1 perf knobs** (`MOLD_KEEP_TE_RAM`, `MOLD_LORA_BYPASS`, `MOLD_VAE_TILED`, `MOLD_ATTN`) are documented in `website/guide/configuration.md` and `.claude/skills/mold/SKILL.md`. Don't repeat their semantics here — they're opt-in / auto-on knobs that surface in those tables and in `[Unreleased]` of `CHANGELOG.md`.

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

| Surface                                                              | Owns                                                                                     |
| -------------------------------------------------------------------- | ---------------------------------------------------------------------------------------- |
| `~/.config/mold/config.toml` (XDG) or `~/.mold/config.toml` (legacy) | Bootstrap: paths, ports, credentials, `[logging]`, `[runpod]`, per-model component paths |
| `mold.db` `settings` + `model_prefs`                                 | User prefs: `expand.*`, `generate.*`, `tui.*`, per-model defaults                        |
| `MOLD_*` env vars                                                    | Runtime override (highest precedence)                                                    |

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
- **Keep user and agent docs in sync with every feature:** `CHANGELOG.md` (Keep-a-Changelog format, under `[Unreleased]`), `README.md`, this file (`AGENTS.md` is its symlink), canonical `.claude/skills/mold/SKILL.md`, the owning app README such as `apps/mobile/README.md`, relevant `desktop/docs/`, and the VitePress `website/` pages/navigation. Model, CLI flag, env-var, endpoint, and native UI changes are incomplete until every affected surface agrees; do not maintain a divergent second agent-skill copy.
- **Releases are automated by release-plz** (`release-plz.toml`, `.github/workflows/release-plz.yml`). Every push to `main` maintains a release PR: release-plz bumps the shared `[workspace.package]` version (conventional commits decide the bump; `feat:` ⇒ minor on 0.x), then a scripted bot commit (`scripts/release/sync-release-pr.sh`) promotes `CHANGELOG.md` `[Unreleased]`, syncs the desktop cargo root's version, and copies the promoted notes into the PR body — never bump versions or promote the changelog by hand. Merging the release PR makes the next `release-plz-release` run push the `vX.Y.Z` tag, which fans out to release.yml (binaries + signed/notarized desktop DMG via `desktop-distribution.yml` + GitHub release + crates.io + AUR) and FlakeHub. Both workflow jobs authenticate as the `release-plz-mold` GitHub App (`RELEASE_PLZ_APP_ID`/`RELEASE_PLZ_APP_PRIVATE_KEY`) so CI runs on the release PR and the tag triggers workflows; the follow-up commit must retain that bot identity or release-plz will replace the PR to preserve human-authored history. Merging the release PR ships the release — treat it as outward-facing. Gotcha: release-plz refuses to run if any tracked file is also gitignored (`git ls-files -ci --exclude-standard` must stay empty).
- **Don't break centered TUI gallery thumbnails.** `crates/mold-tui/src/ui/gallery.rs` uses a fixed-protocol thumbnail path. Do not revert the grid to plain `StatefulImage` for Kitty/Sixel/iTerm2 — it reintroduces top-left-padded thumbnails instead of centered aspect-correct ones. Keep the regression tests passing.
- **AUR PKGBUILDs auto-publish on `v*` tags.** `packaging/aur/{mold-ai-bin, mold-ai}/PKGBUILD` are the source of truth; CI rewrites their `pkgver` + `sha256sums` in the `publish-aur` job and pushes to the AUR git repos. The `-bin` package tracks the `cuda-sm89` release tarball; Blackwell (sm_120) users build the source PKGBUILD with `CUDA_COMPUTE_CAP=120`. `mold-ai-git` is hand-pushed when the build recipe changes. AUR packages declare `conflicts=('mold')` because they collide with the rui314 linker at `/usr/bin/mold`. The in-tree `mold update` correctly bows out on `/usr/bin/mold` and `/usr/sbin/mold` Linux installs.

## Key design decisions

1. **Crate boundaries are clean** — `mold-cli` doesn't depend on candle; `mold-server` doesn't depend on clap; `mold-discord` only depends on `mold-core`.
2. **candle over tch/ort** — pure Rust, no libtorch. Uses a published fork (`candle-*-mold` on crates.io) for Metal quantized matmul precision + seed buffer size fixes.
3. **Single binary** — `mold` includes `serve` via `mold-server` library; GPU flags forward `mold-cli` → `mold-server` → `mold-inference`.
4. **`tokio::sync::Mutex` + `spawn_blocking`** — single-model-at-a-time fits GPU workloads. `AppState.model_cache` is an LRU (max 3) with `ModelResidency { Gpu, Parked, Unloaded }`; at most one engine is GPU-resident.
5. **Nix flake (flake-parts + crane)** — CUDA 12.8 on Linux (default `CUDA_COMPUTE_CAP=89` Ada; `mold-sm120` for Blackwell; `mkMold` for any), Metal on macOS. Devshell sets `CPATH`/`LIBRARY_PATH`/`LD_LIBRARY_PATH` for CUDA compilation.
6. **Shell completions** — static via `clap_complete` + dynamic via `CompleteEnv` with `ArgValueCandidates` for model names.
