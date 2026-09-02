# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

Keep lean: only things not obvious from the code, `--help`, or `git log`. Area-specific invariants live in `.claude/rules/*.md` (path-scoped; `AGENTS.md` symlinks here).

## What mold is

Local AI image/video generation CLI built on [candle](https://github.com/huggingface/candle). Supports FLUX, SD1.5, SDXL, SD3.5, Z-Image, Flux.2 Klein (distilled and base) and Dev, Qwen-Image, Wuerstchen v2, LTX-Video, LTX-2, and Wan 2.1/2.2 (T2V; the family's frame grid is `4k+1` and its sampler deliberately follows the diffusers/Lightning flow-UniPC schedule, not upstream Wan's `fm_solvers_unipc.py` — see `crates/mold-inference/src/wan/sampler.rs`). Runs locally on GPU or talks to a remote `mold serve` over HTTP. Single binary, everything feature-gated.

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
cargo test --workspace                       # PRs run a filtered deterministic subset; full suite only on main
cargo +1.93 check -p mold-ai --locked --features preview,discord,expand,tui,metrics,webp,mp4,mdns,pulid   # CI MSRV gate
cargo run -p mold-ai-core --bin generate_generation_profiles -- --check   # CI contract
bash scripts/tests/ci-routing-contract.sh                                 # CI contract
bash scripts/tests/candle-single-identity.sh                              # every candle crate on ONE fork rev
./scripts/coverage.sh [--html]

# Frontend (one Bun workspace at repo root; prettier scoped to studio/, desktop override in .prettierrc)
bun run check:frontend        # architecture check + tests + web/desktop builds
bun run check:architecture    # scripts/tests/frontend-architecture.sh
bun run check:dead-code       # knip
bun run fmt:check

# Local dev run (MUST prefix with ensure-web-dist so the embedded SPA isn't a stub)
./scripts/ensure-web-dist.sh && cargo run --profile dev-fast -p mold-ai \
  --features metal,preview,expand -- run "a cat"
```

Inside `nix develop` the devshell exposes ~60 shortcuts (`build`, `serve`, `mold`, `clippy`, `run-tests`, `coverage`, `fmt`, `ci-local`, `desktop-*`, `ios-*`, `android-*`, `frontend-bun-lock`, …). Run `type <cmd>` to see the underlying invocation.

**MSRV**: 1.93. **Rust 2024** only in `apps/mobile/src-tauri` (excluded from treefmt).

## Crates

```
crates/
├── mold-core/        Shared types, HTTP client, config, manifest, validation, download
├── mold-catalog/     Live HF + Civitai model-discovery proxy (5-min in-proc cache). Depended on by mold-cli, mold-server, and mold-inference — mold-discord MUST NOT depend on it (mold-tui reaches it transitively through mold-inference).
├── mold-db/          SQLite (rusqlite, bundled, WAL) — gallery, settings, model_prefs, prompt_history
├── mold-inference/   Candle engines per family
├── mold-candle/      Application-owned candle models + public-API extensions (backend changes go in the utensils/candle fork)
├── mold-scheduler/   Placement / admission planner
├── mold-server/      Axum HTTP server (consumed as lib by mold-cli)
├── mold-cli/         The `mold` binary (clap)
├── mold-discord/     Discord bot (poise + serenity), HTTP-only dep on mold-core
└── mold-tui/         Interactive TUI (ratatui)
ui/        @mold/ui      — visual tokens + low-level Vue primitives (lowest layer)
studio/    @mold/studio  — HTTP contracts, Pinia state, shared domain logic; must never import Tauri or a shell
web/       SPA embedded in the binary       desktop/  Tauri 2 app (own cargo root, excluded from workspace)
apps/mobile/  iPhone/Android thin Tauri crate (own cargo root, remote-only)
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
| `mold-tui/`       | `mold-ai-tui`              |

Gotcha: the tauri dev watcher does not rebuild on `crates/` changes — relaunch `desktop-dev`. In Vue stores, `reactive()`-wrap any object mutated from SSE/closure callbacks.

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
- **An auto-chained one-shot is not an authored sequence.** `ChainRequest.ephemeral` (additive, absent means authored) is the distinction, set on the ONE creation path: `mold run --frames 200` renders a long video as a chain because the model cannot do it in one pass, so its job is absent from authored `GET /api/chain-jobs` listings and emits no `chain_job_queued`, while `/api/activity` still exposes the live job on every client. Ephemeral means hidden from sequence history and swept only after settlement; it does **not** mean disposable while active. Graceful shutdown parks it as `paused`, preserves its manifest, source media, completed clips, and tail cache, and allows explicit resume after restart. Its final print carries `chain_job_id: None`, so Reuse settings restores a one-shot instead of opening the clip rail. It DOES publish its print with full per-clip provenance. Stage seeds are recorded either way — they describe how the pixels were made, not who authored the split.
- **A remote sequence is a DURABLE CHAIN JOB, on every surface.** `POST /api/chain-jobs` creates one and `GET /api/chain-jobs/{id}/events` streams its stage progress; `mold run --script` and the TUI both take that route and hydrate the stitched print from the host's gallery. The synchronous `POST /api/generate/chain` and SSE `POST /api/generate/chain/stream` shims are DELETED — they ran a chain as a hidden ephemeral job and deleted its artifacts after answering, so a dropped connection lost work that could not be resumed, retaken, or reattached. `POST /api/generate/chain/validate` survives and is the only thing left in `routes_chain.rs`, which is read-only planning. A `--script` run therefore leaves a job `mold chain list` can find, and the CLI no longer receives the host-side thumbnail or GIF preview inline.

## Prompting corpus (one source of truth)

`crates/mold-core/src/prompting/` holds every prompting guide: `shared.md`, one `families/<family>.md` per manifest family (the coverage test fails when a family is added without one), task leaves `<family>/<leaf>.md`, and model leaves `models/<name>.md`. `mod.rs` is the registry (word limits, identity matchers, `route()`); numbers live there and prose in markdown, joined by the `{{word_limit}}` placeholder. Three consumers read it and nothing else may carry a divergent copy: the skill renderer (`mold skill`, byte-identical `references/prompting/**` across every agent profile), the expander (`expand_prompts.rs` puts `PromptingRoute::expansion_excerpt()` into `{MODEL_NOTES}` and appends a `GENERATION CONTEXT` block rendered from `ExpandContext`), and MCP (`mold://prompting/` resources plus the `expand_prompt`/`remix_prompt` tools). Guides follow a fixed H2 order (Prompt style, Syntax, Generation context, Examples, Pitfalls, CLI, Sources); `CLI` and `Sources` and every `bash` fence are agent-only and never reach the LLM, and each route's excerpt must stay under `EXCERPT_WORD_BUDGET` so the 1.7B local expander and 2k-context OpenAI-compatible hosts still see the user prompt. Every `bash` block in the corpus is parsed against the clap CLI by a test, so an example with a bad flag fails the build. `website/guide/prompting.md` and `docs/generated/prompting-guides-v1.json` are generated by `cargo run -p mold-ai-core --bin generate_prompting_guides` and checked in CI with `--check`; never hand-edit them. Write guides from the official upstream prompting docs cited in each file's `Sources`, never from memory.

## 3-D generation

- **The generation profile is the single authority for the prompt, strength, and the mesh controls.** `capabilities.prompt` (`Required` | `Optional` | `Ignored`, `#[serde(default)]` Required so older JSON parses) is emitted from ONE core function, `generation_profile::prompt_requirement_for_family`, which `validation::prompt_required_with_conditioning` also calls — so admission, the CLI, and every client necessarily agree and nobody carries a family allowlist. The advertised mode answers for a CONDITIONED request, because that is the only case that can differ; a client resolves it against the request it is building. `hunyuan3d` is `Ignored` (no text encoder anywhere in the family), LTX with visual conditioning is `Optional`, everything else `Required`. `capabilities.supports_strength` and `capabilities.mesh` (octree allowlist + default, iso-threshold `FloatControl`, face bounds, `texture` feature control, built from the `validation::MESH_*` constants) follow the same rule: advertised once, validated against the same block by `validate_request_against_recipe`, and refused outright on a recipe with no `mesh` block. Discord is the one client that keeps a family pin: its request builder pins GLB for `hunyuan3d` (`is_mesh` in `mold-discord/src/commands/generate.rs`) because the bot builds requests from a model cache that is empty until the first refresh and a manifest fallback that carries no profile; the server's own `pin_output_format_for_family` is what makes that pin safe rather than a second authority.
- **A format the recipe does not advertise is a 422 at durable admission, not a Hold** — `validate_output_format_against_generation_profile` runs in `queue_media_admission` for every non-private request carrying an explicit format, so a client learns at submit time instead of watching a job hold and fail. Formats are PINNED, not refused, where the family has exactly one deliverable container: `GenerateRequest::pin_output_format_for_family` coerces a raster format to `Glb` for the mesh family at both doors, mirroring the CLI's own `default_output_format`, so an older client that always sends `png` still renders. That is the whole exception; everywhere else an unavailable format is a real client mistake.
- **Mesh exports are derived from the stored GLB, never generation targets.** GLB stays the only stored form (one file carrying geometry, UVs, normals and textures); `POST /api/gallery/export/:filename` reads it back through `hunyuan3d::glb::read_glb` and either TRANSCODES it to OBJ, STL, or PLY or RENDERS an animated GIF/APNG/WebP turntable from it (a render, not a transcode — it rasterizes the mesh, bounded at the rasterizer's 2048 `max_dimension`), and the same conversions are on `mold library export` and the `export_mesh` MCP tool. `MeshExportFormat` is its own enum rather than more `OutputFormat` variants precisely so a request can never name one as a generation target; `capabilities.mesh.export_formats` advertises the list, with the stored `glb` listed first so a client can see what it holds.
- **The GUI surfaces read the same authorities and nothing else.** The Create rail's **Mesh** group (octree ladder, iso threshold, target faces with an inline out-of-bounds advisory) is built from the recipe's `capabilities.mesh` block; a canvasless recipe hides Shape/Resolution and `toRequest` pins the size to 0×0, the format to GLB, and drops the fit policy at request time, so a stale persisted draft cannot ship raster leftovers. The shared `studio/components/MeshViewer.vue` (raw WebGL, GLSL ES 1.00 with a `webgl` fallback) renders the result; auto-rotate and fullscreen are prop-gated and passed only by the three Create result areas. The export menu is `splitMeshExportFormats(capabilities.mesh.export_formats)` from `studio/lib/meshExport.ts` — geometry files one entry each, animated containers collapsed into one **Export turntable…** entry that opens the video export sheet, `glb` dropped — never a client constant. Expand and Remix are hidden/refused for a recipe whose profile advertises `prompt.mode: ignored`, and the empty canvas's sentence is `promptGuidance()`'s, resolved once by the page.
- **A prompt-`Ignored` family is never handed to the expansion LLM.** `expand_prompts::ignored_prompt_advice` is the ONE decision behind every expand/remix door (`mold expand`, `mold remix`, `mold run --expand`, `/api/expand`, `/api/remix` — which is also where Discord's `/expand` and `/remix` are answered, they are HTTP-only — generate-time `maybe_expand_prompt` and the activation gate ahead of it in `prepare_generation_inner`, the MCP tools, the TUI, and the shared `expand_exact_with` driver): it asks `prompt_requirement_for_family` and, for `Ignored`, returns the family guide's `Generation context` section rendered from the corpus (`prompting::section_excerpt`, never a second copy in Rust) as the single result before any expansion model is created, activated, or pulled. Generation-time expansion is skipped (flag cleared) rather than answered, so provenance never records advice as a prompt. `ExpandContext.prompt_mode` carries the resolved contract and the `GENERATION CONTEXT` block states when the prompt is not read.
- **Geometry exports carry a server-advertised per-format defaults table, and the turntable sweep is fit once.** The stored GLB is Hunyuan3D's normalized unit-cube space, so a slicer or a Blender STL/PLY import reads it wrong; `obj`/`stl`/`ply` export take optional `size_mm`/`up_axis`/`origin`, resolved against `mold_core::validation::mesh_export_geometry_defaults` (OBJ unscaled/Y-up/floor; STL and PLY 100 mm/Z-up/floor) by `resolve_mesh_export_geometry`, and applied identically by the server's transcode and the TUI's local writer. The block's presence on `capabilities.mesh.export_geometry` is the only client gate — never a family allowlist — and the three keys are refused, not ignored, on `glb` and on a turntable. Separately, a turntable is framed ONCE for the whole sweep (`raster::FrameFit`, `turntable_frame_cameras`) rather than refit per frame, so a rotating mesh holds one size instead of breathing and popping; `FrameFit::Auto` stays the poster's own default so poster bytes are unchanged.

## Config

## Durable gallery source media

Durable queue uploads do not die with their queue row. Publication first pins
the encrypted media set under queue-media storage, commits that exact pin into
the gallery archive authority, projects it into `gallery_media_*`, and only
then settles the queue row. Restart replay performs the same handoff before it
recognizes an already-published job as complete. Trash preserves pins;
permanent deletion removes gallery authority first and releases only that
print's pins afterward, so sibling outputs remain independent.

`GET /api/gallery/source-media/:filename` and its opaque-member download route
require the CALLER to be authorized, which is the same question every other
privileged route asks: on a host with `MOLD_API_KEY` set that is an
authenticated request, and on a keyless host it is every request, exactly as
`DELETE /api/gallery/image/:filename` and device lifecycle already behave. The
gate must never read the server's own configuration as the answer —
`AuthState = None` means "open by policy", and treating it as a refusal made the
whole feature dead on a default server, answering `unavailable_auth` for prints
it never looked at. They never expose paths or store identities and
report explicit `unavailable_auth`, `unavailable_legacy`, and
`unavailable_missing_or_corrupt` states; an empty member list after a CLEAN
resolve is `unavailable_legacy`, never corruption, because `downloadable_role`
filters provenance-text roles by design. **Every client always asks, and the
server is the only authority on what it retained** — `OutputMetadata` records
no marker at all for inline `source_video`, `audio_file`, or `mask_image`
bytes, so a client that skipped the probe on missing markers would silently
lose those restorations. What the metadata decides is DISCLOSURE: the server
cannot tell a pre-feature print from one that never had source media (both
resolve with no pins), so an UNAVAILABLE answer is toasted only when
`retainedSourceMediaDisclosable` in `studio/api/gallerySourceMedia.ts` finds
the print's own recorded conditioning bytes, and a text-to-image print stays
silent. The same module maps a middleware `401` to `unavailable_auth` so a
keyed host reached without a key gets the API-key disclosure instead of a
swallowed error. Desktop's Lightbox
primary button and its right-click item both go through `reuseSettings`, which
is the only path that attaches retained authority (`composer.set` invalidates
it). A same-host reuse session is one-time, short-lived, and bound to the exact
target request — on a keyless host to one stable anonymous subject; cross-host
reuse remains a client download-and-upload relay.

Two stores, one logical `Config` view:

| Surface                                                              | Owns                                                                                     |
| -------------------------------------------------------------------- | ---------------------------------------------------------------------------------------- |
| `$MOLD_HOME/config.toml` (default `~/.mold/config.toml`; `$XDG_CONFIG_HOME/mold/home` is a bootstrap pointer file, never config) | Bootstrap: paths, ports, credentials, `[logging]`, `[runpod]`, per-model component paths |
| `mold.db` `settings` + `model_prefs`                                 | User prefs: `expand.*`, `generate.*`, `tui.*`, per-model defaults                        |
| `MOLD_*` env vars                                                    | Runtime override (highest precedence)                                                    |

Every `main()` calls `mold_db::config_sync::install_config_post_load_hook()`, which runs a one-shot idempotent `config.toml → DB` migration on first boot (renames original to `config.toml.migrated`) and overlays DB onto every `Config::load_or_default()`. Consumers still read `cfg.expand.*` unchanged.

`mold config set <key> <val>` routes by key prefix (`expand.*` → DB, `models_dir` → TOML). `mold config where <key>` prints the surface. `mold config list --json` tags each row `[db]` / `[file]` / `[env]`. Multi-profile: `settings` and `model_prefs` are keyed on `(profile, key)`; active profile resolves `MOLD_PROFILE` → `settings.profile.active` → `"default"`.

## Workflow

- **TDD.** Every bug fix and feature: failing test first, then code. Prefer unit tests on exported contracts (key→action maps, focus transitions, serialization round-trips, layout invariants) over E2E. Layout constants need a test that asserts the inner area fits the rendered row count — otherwise they drift.
- **Port from upstream reference implementations, never from memory.** When implementing or debugging any model family, sampler, scheduler, VAE, or pipeline in Rust, consult the authoritative upstream implementation first — the official model repo (e.g. Wan-Video/Wan2.2, Lightricks/LTX-2, black-forest-labs/flux), ComfyUI, and/or Hugging Face diffusers — and mirror it. Clone references into gitignored `tmp/` and `git pull` before consulting; justify behavioural changes with upstream `file:line` citations rather than inferring intent from mold's existing port. Upstream is read-only reference material: the port itself is always pure Rust (candle) — never call into Python, link Python runtimes, shell out to upstream scripts, or add non-Rust dependencies to make a port work. Running upstream code in a scratch venv to capture golden fixtures/expected tensors for parity tests is fine; shipping any of it is not. Where mold deliberately tracks a different reference than the official repo (e.g. the Wan sampler follows the diffusers/Lightning flow-UniPC schedule, not upstream's `fm_solvers_unipc.py`), that choice is documented in this file — follow the documented reference and never silently switch it. **Prefer a reference you can run over one you can only read.** A code read establishes that a port "looks equivalent"; an executable oracle on the same hardware, fed the same checkpoint, tells you in one command whether the difference is yours. Where such an oracle exists it is the documented primary reference — Z-Image's is stable-diffusion.cpp (see the Z-Image entry above), which renders the same GGUF files mold downloads, on Metal, correctly. Reach for it before theorising about the backend, the file, or the quantization. The LTX-2 rule below is the template; it applies to every family.
- **Keep user and agent docs in sync with every feature:** a release note as `changelog.d/<slug>.md` (one fragment file per PR, holding its Keep-a-Changelog bullet — or several bullets when one PR ships several notes — NEVER edit `CHANGELOG.md`'s `[Unreleased]` section by hand; two open PRs inserting at that line is what made every PR conflict, the release PR assembles and deletes the fragments, and the advisory `changelog` CI check flags direct edits (it is not a required status, so a red check is a request to fix, not a hard block) — `changelog.d/README.md` has the format; `skip-changelog` label for PRs that ship nothing user-visible), `README.md`, this file (`AGENTS.md` is its symlink), the canonical skill renderer/template under `crates/mold-cli/src/skill/` and the prompting corpus under `crates/mold-core/src/prompting/`, the owning app README such as `apps/mobile/README.md`, relevant `desktop/docs/`, and the VitePress `website/` pages/navigation. Model, CLI flag, env-var, endpoint, and native UI changes are incomplete until every affected surface agrees; do not maintain a divergent second agent-skill copy.
- **Releases are automated by release-plz.** Never bump versions or edit `CHANGELOG.md` `[Unreleased]` by hand — add a `changelog.d/<slug>.md` fragment. Merging the release PR ships the release. Details: `.claude/rules/release-ci.md`.

## Key design decisions

1. **Crate boundaries are clean** — `mold-cli` doesn't depend on candle; `mold-server` doesn't depend on clap; `mold-discord` only depends on `mold-core`.
2. **candle over tch/ort** — pure Rust, no libtorch. Application-owned models and public-API extensions live in `mold-ai-candle`; backend changes that cannot be implemented outside Candle live in the `utensils/candle` fork and are removed as upstream accepts them. **Every candle crate — `candle-core`, `candle-nn`, `candle-transformers`, `candle-flash-attn`, `candle-onnx` — is a direct git dependency on ONE fork revision, in every cargo root (workspace, desktop, mobile).** `candle_core::Tensor` and `Error` are nominal types, so a single crate left on crates.io pulls the upstream-named `candle-core` in beside the fork's `candle-core-mold` and every call site handing a tensor across that seam stops compiling — which is exactly how #1393 broke CUDA for four consecutive `main` merges by moving three crates and leaving `mold-candle`'s `candle-flash-attn` behind (#1399). `[patch.crates-io]` cannot express this (a patch must keep the patched package's name, and the fork's are renamed), so the pin is the whole contract and `scripts/tests/candle-single-identity.sh` enforces it on the PR-visible release-contract route — the `--features flash-attn` compile gate is push-only and cannot warn a PR. `cargo tree -d` must never report candle from two sources.
3. **Single binary** — `mold` includes `serve` via `mold-server` library; GPU flags forward `mold-cli` → `mold-server` → `mold-inference`.
4. **`tokio::sync::Mutex` + `spawn_blocking`** — single-model-at-a-time fits GPU workloads. `AppState.model_cache` is an LRU sized by `MOLD_MAX_CACHED_MODELS` (default 3, accepted range 1–16) with `ModelResidency { Gpu, Parked }` — eviction removes the entry rather than adding a third state; at most one engine is GPU-resident.
5. **Nix flake (flake-parts + crane)** — CUDA 12.8 on Linux (default sm_89 Ada; `mold-sm86` for RTX 3090/A40, `mold-sm100` for B200/B300, `mold-sm120` for RTX 50-series; `mkMold` for any), Metal on macOS. B200 is server-only and remains simulated, not hardware-qualified. Devshell sets `CPATH`/`LIBRARY_PATH`/`LD_LIBRARY_PATH` for CUDA compilation **and execution** — a devshell binary gets no RUNPATH, so every library the release feature set links (cuDNN included) must also be on `LD_LIBRARY_PATH`; the `devshell-cuda-load-path` check enforces it (#1510).
6. **Shell completions** — static via `clap_complete` + dynamic via `CompleteEnv` with `ArgValueCandidates` for model names.
7. **Lifecycle authority follows scheduler ownership** — `PATCH /api/devices/:id` and every client enable/disable control are available only when `/api/capabilities.devices.lifecycle` is true. Legacy, observe, CPU-fallback, and all-disabled maintenance runtimes remain read-only; never persist a live change they cannot enforce.
