# CLAUDE.md

Guidance for Claude Code in this repository. `AGENTS.md` is a symlink to this file.

Only rules that apply to ANY change live here. Area-specific invariants live in path-scoped `.claude/rules/*.md` files, which load automatically when you read matching files — read the relevant one before editing an area it covers (e.g. `inference.md`, `server-queue.md`, `studio-web.md`, `desktop.md`, `mesh-3d.md`, `gallery-authority.md`, `reference-images.md`, `chain-sequences.md`, `prompting-corpus.md`, `release-ci.md`, `ios-native.md`). New area-specific rules go there, not here.

## What mold is

Local AI image/video/3-D generation CLI built on [candle](https://github.com/huggingface/candle): FLUX, SD1.5, SDXL, SD3.5, Z-Image, Flux.2 Klein/Dev, Qwen-Image, Qwen Image 2.1, Wuerstchen v2, LTX-Video, LTX-2, Wan 2.1/2.2, Hunyuan3D, MiniMax H3. Single binary, everything feature-gated. `mold run` talks to `$MOLD_HOST` (default `http://localhost:7680`) and falls back to the local GPU when the server is unreachable (auto-pulling the model); `--local` skips the server.

## Commands

```bash
# Nix (preferred)
nix build                   # Build mold (default CUDA/Metal)
nix fmt                     # treefmt (nixfmt + rustfmt), configured inline in flake.nix; no rustfmt.toml
nix flake check             # CI-equivalent gate

# Local CI runner (what to run before a PR; devshell alias: ci-local)
./scripts/ci-local.sh [rust|web|docs|contracts|gpu|nix] [-k] [--list]

# Cargo — common loops
cargo check
cargo clippy --workspace --all-targets -- -D warnings
cargo fmt --all -- --check
cargo nextest run --profile main --workspace   # full suite, what a push to main runs (cargo test --workspace is equivalent, slower)
cargo nextest run --profile pr $(python3 scripts/ci/affected-packages.py --base origin/main --head HEAD | sed -n 's/^packages=//p')   # what a PR runs (see .config/nextest.toml; add --features from the same script)
cargo test -p mold-ai-core --lib <filter>    # single test/module; use the PACKAGE name (table below), not the dir
cargo test -p mold-ai-server --features mdns --lib mdns   # feature-gated modules (mdns, pulid, h3) never compile under --workspace
cargo +1.93 check -p mold-ai --locked --features preview,discord,expand,metrics,webp,mp4,mdns,pulid   # MSRV gate (weekly msrv.yml, not on the merge path)
cargo check -p mold-ai --features cuda,preview,discord,expand,webp,mp4,metrics,mdns,pulid,mesh-texture,mesh-matting,mesh-delight   # the PR-route cuda-typecheck (needs nvcc); the h3 arm is covered by metal-check on macOS; a default-feature build type-checks none of the GPU cfg arms
./scripts/coverage.sh [--html]

# CI contracts
cargo run -p mold-ai-core --bin generate_prompting_guides -- --check
cargo run -p mold-ai-core --bin generate_generation_profiles -- --check
bash scripts/tests/ci-routing-contract.sh
bash scripts/tests/candle-single-identity.sh     # every candle crate on ONE fork rev

# Frontend (one Bun workspace at repo root; prettier scoped to studio/, desktop override in .prettierrc)
bun run check:frontend        # architecture check + tests + web/desktop builds
bun run check:architecture    # scripts/tests/frontend-architecture.sh
bun run check:dead-code       # knip
bun run fmt:check
bunx vitest run --config studio/vitest.config.ts studio/lib/<file>.test.ts   # single studio test
cd web && bunx vitest run src/<file>.test.ts                                 # single web test (desktop/ likewise)
cd desktop && bunx vitest run ../ui/<path>.test.ts                           # ui/ tests only run through desktop's vitest

# Local dev run (MUST prefix with ensure-web-dist so the embedded SPA isn't a stub)
./scripts/ensure-web-dist.sh && cargo run --profile dev-fast -p mold-ai \
  --features metal,preview,expand -- run "a cat"
```

Inside `nix develop` the devshell exposes ~60 shortcuts (`build`, `serve`, `mold`, `clippy`, `run-tests`, `coverage`, `fmt`, `ci-local`, `desktop-*`, `ios-*`, `android-*`, `frontend-bun-lock`, …); `type <cmd>` shows the underlying invocation.

**MSRV** 1.93. **Rust 2024** only in `apps/mobile/src-tauri` (excluded from treefmt).

**Nix flake (flake-parts + crane):** CUDA 12.9 on Linux (default sm_89 Ada; `mold-sm86` for RTX 3090/A40, `mold-sm100` for B200/B300 — server-only, simulated, not hardware-qualified; `mold-sm120` for RTX 50-series; `mkMold` for any), Metal on macOS. The devshell sets `CPATH`/`LIBRARY_PATH`/`LD_LIBRARY_PATH` for CUDA compilation **and execution** — a devshell binary gets no RUNPATH, so every library the release feature set links (cuDNN included) must also be on `LD_LIBRARY_PATH`; the `devshell-cuda-load-path` check enforces it (#1510).

## Crates

```
crates/
├── mold-core/        Shared types, HTTP client, config, manifest, validation, download, generation profiles
├── mold-catalog/     Live HF + Civitai model-discovery proxy (5-min in-proc cache); mold-discord MUST NOT depend on it
├── mold-db/          SQLite (rusqlite, bundled, WAL) — gallery, settings, model_prefs, prompt_history
├── mold-inference/   Candle engines per family
├── mold-candle/      Application-owned candle models + public-API extensions (backend changes go in the utensils/candle fork)
├── mold-scheduler/   Placement / admission planner
├── mold-server/      Axum HTTP server (consumed as lib by mold-cli)
├── mold-cli/         The `mold` binary (clap)
└── mold-discord/     Discord bot (poise + serenity), HTTP-only via mold-core (mold-db for the config hook)
ui/        @mold/ui      — visual tokens + low-level Vue primitives (lowest layer)
studio/    @mold/studio  — HTTP contracts, Pinia state, shared domain logic; must never import Tauri or a shell
web/       SPA embedded in the binary       desktop/  Tauri 2 app (own cargo root, excluded from workspace)
apps/mobile/  iPhone/Android thin Tauri crate (own cargo root, remote-only)
apps/macos/   Mold Studio, the native SwiftUI Mac app (XcodeGen; embeds the engine)
apps/ios/     Mold Studio Companion, the native SwiftUI iPhone/iPad app (remote-only)
apps/shared/  MoldClient + MoldStyle Swift packages both native apps build on
```

**Directory ≠ package name.** Use these with `-p`:

| Dir               | Package                    |
| ----------------- | -------------------------- |
| `mold-cli/`       | `mold-ai` (binary: `mold`) |
| `mold-core/`      | `mold-ai-core`             |
| `mold-catalog/`   | `mold-ai-catalog`          |
| `mold-db/`        | `mold-ai-db`               |
| `mold-inference/` | `mold-ai-inference`        |
| `mold-candle/`    | `mold-ai-candle`           |
| `mold-scheduler/` | `mold-ai-scheduler`        |
| `mold-server/`    | `mold-ai-server`           |
| `mold-discord/`   | `mold-ai-discord`          |

## Workflow

- **TDD.** Every bug fix and feature: failing test first, then code. Prefer unit tests on exported contracts (key→action maps, focus transitions, serialization round-trips, layout invariants) over E2E. Layout constants need a test that asserts the inner area fits the rendered row count — otherwise they drift.
- **Port from upstream reference implementations, never from memory.** For any model family, sampler, scheduler, VAE, or pipeline, consult the authoritative upstream first — the official model repo (e.g. Wan-Video/Wan2.2, Lightricks/LTX-2, black-forest-labs/flux), ComfyUI, and/or diffusers — and mirror it. Clone references into gitignored `tmp/` and `git pull` before consulting; justify behavioural changes with upstream `file:line` citations, never by inferring intent from mold's existing port.
  - The port is always pure Rust (candle): never call into Python, link Python runtimes, shell out to upstream scripts, or add non-Rust dependencies. Running upstream in a scratch venv to capture golden fixtures for parity tests is fine; shipping any of it is not.
  - **Prefer a reference you can run over one you can only read.** An executable oracle on the same hardware and checkpoint tells you in one command whether a difference is yours; where one exists it is the documented primary reference (Z-Image's is stable-diffusion.cpp; the LTX-2 rule in `inference.md` is the template for every family). Reach for it before theorising about the backend, the file, or the quantization.
  - Where mold deliberately tracks a different reference than the official repo, the choice is documented in the owning rules file (e.g. Wan's diffusers/Lightning flow-UniPC schedule and DMD ladder shifts in `inference.md`; Surface-nets and Qwen Image 2.1 divergences in `mesh-3d.md` / `qwen-image21.md`). Follow the documented reference; never silently switch it.
- **Keep user and agent docs in sync with every feature.** Model, CLI flag, env-var, endpoint, and native UI changes are incomplete until every affected surface agrees: a `changelog.d/<slug>.md` fragment, `README.md`, this file or the owning `.claude/rules/*.md`, the canonical skill renderer/template under `crates/mold-cli/src/skill/` (never a divergent second agent-skill copy), the prompting corpus under `crates/mold-core/src/prompting/`, the owning app README (e.g. `apps/mobile/README.md`), relevant `desktop/docs/`, and the VitePress `website/` pages/navigation.
- **Releases are automated by release-plz.** Never bump versions or edit `CHANGELOG.md`'s `[Unreleased]` section by hand — two open PRs inserting at that line made every PR conflict. Add one `changelog.d/<slug>.md` fragment per PR (one or more Keep-a-Changelog bullets; format in `changelog.d/README.md`); the release PR assembles and deletes them. The advisory `changelog` CI check flags direct edits (not a required status — a red check is a request to fix, not a hard block). Use the `skip-changelog` label for PRs that ship nothing user-visible. crates.io distribution is retired (GitHub artifacts, Nix, Docker, AUR, source). Details: `.claude/rules/release-ci.md`.
- **Never hand-edit generated files.** `website/guide/prompting.md`, `docs/generated/prompting-guides-v1.json`, `docs/generated/generation-profiles-v1.json`, `docs/model-resolution-matrix.md` and `studio/lib/generated/generationProfileV1.ts` are regenerated by the `generate_*` bins above and checked in CI with `--check`.

## Cross-cutting design rules

1. **Crate boundaries are clean** — `mold-cli` doesn't depend on candle; `mold-server` doesn't depend on clap; `mold-discord` talks to the server over HTTP through `mold-core` and links `mold-db` only for the config post-load hook every `main()` installs.
2. **candle, one fork identity.** Pure Rust, no libtorch. Application-owned models and public-API extensions live in `mold-ai-candle`; backend changes that cannot live outside Candle go in the `utensils/candle` fork and are removed as upstream accepts them. **Every candle crate — `candle-core`, `candle-nn`, `candle-transformers`, `candle-flash-attn`, `candle-onnx` — is a direct git dependency on ONE fork revision, in every cargo root (workspace, desktop, mobile).** `Tensor`/`Error` are nominal types, so one crate left on crates.io pulls upstream `candle-core` in beside the fork's `candle-core-mold` and every seam stops compiling (#1393 broke CUDA for four `main` merges that way; #1399). `[patch.crates-io]` cannot express this (the fork's packages are renamed), so the pin is the whole contract and `scripts/tests/candle-single-identity.sh` is the ONLY guard — the `--features flash-attn` compile gate was switched off in 48dbb266 (see `release-ci.md`). `cargo tree -d` must never report candle from two sources.
3. **Single binary** — `mold` includes `serve` via the `mold-server` library; GPU features forward `mold-cli` → `mold-server` → `mold-inference`.
4. **The generation profile is the capability authority.** `/api/capabilities` and each recipe's `capabilities.*` block (prompt, strength, mesh, reference images, transparency, …) are emitted by `mold_core::generation_profile` from the same functions admission validates against. Clients read the block and never carry a family/model allowlist; an ABSENT additive block means an OLDER SERVER, never a refusal. Details per area in `mesh-3d.md`, `reference-images.md`, `qwen-image21.md`, `studio-web.md`.
5. **Config is two stores plus env.** `$MOLD_HOME/config.toml` holds bootstrap settings, `mold.db` holds user prefs, `MOLD_*` env overrides both. See `config-stores.md`.

## Gotchas

- The tauri dev watcher does not rebuild on `crates/` changes — relaunch `desktop-dev`.
- In Vue stores, `reactive()`-wrap any object mutated from SSE/closure callbacks.
- Feature-gated modules (`mdns`, `pulid`, `h3`) never compile under `--workspace`; test them with explicit `--features`.
