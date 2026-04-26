# Phase-2 task 2.8 — CLI recipe-driven `mold pull cv:<id>` (kickoff handoff)

> Paste the prompt at the bottom of this file into a fresh Claude Code
> session to start task 2.8. Everything above is reference material for
> the author / for skimming.

## Where phase 2 stands on entry

Branch `feat/catalog-expansion`. The seven phase-2 commits are
**local-only** — phase 2 lands as one push at 2.10:

| Commit | Origin? | Scope |
|---|---|---|
| `520e69a` | local | feat(server,core,web): catalog companion auto-pull on download (phase 2.7) |
| `b221b67` | yes | docs(tasks): catalog-expansion phase-2 task 2.7 kickoff handoff |
| `14c6061` | local | feat(inference): single-file factory routing + load() (phase 2.6) |
| `c072967` | yes | docs(tasks): catalog-expansion phase-2 task 2.6 kickoff handoff |
| `a579a0c` | local | feat(inference): SDXL single-file engine constructor (phase 2.5) |
| `9e00bd3` | yes | docs(tasks): catalog-expansion phase-2 task 2.5 kickoff handoff |
| `f87687b` | local | feat(inference): SD1.5 single-file engine constructor (phase 2.4) |
| `e50128f` | yes | docs(tasks): catalog-expansion phase-2 task 2.4 kickoff handoff |
| `a6991b4` | local | feat(inference): single-file checkpoint dispatcher (phase 2.3) |
| `e7d4f4a` | yes | docs(tasks): catalog-expansion phase-2 task 2.3 kickoff handoff |
| `4970a92` | yes | docs(tasks): SD1.5 + SDXL tensor-prefix audit findings (phase 2.2) |
| `cb13e06` | yes | feat(inference): sd_singlefile_inspect dev-bin for tensor-prefix audit |

`tasks/catalog-expansion-phase-2-handoff.md` is your task-list source of
truth. The full single-file ingest stack — dispatcher → SD1.5 keys →
SDXL keys → engine constructors → factory routing → `SingleFileBackend`
+ server companion auto-pull — is shipped end-to-end. 2.8 wires the
**CLI** end: `mold pull cv:<id>` learns the recipe-driven download path
that 2.7 deferred for Civitai primaries.

### Done

- 2.1 pre-flight, 2.2 tensor audit, 2.3 dispatcher, 2.4 SD1.5
  constructor, 2.5 SDXL constructor (incl. CLIP-G fused QKV), 2.6 factory
  routing + `is_turbo` threading + `SingleFileBackend`, 2.7 server
  companion auto-pull (drops `engine_phase >= 2` gate, adds 6 synthetic
  companion manifests in `family: "companion"`, on-disk presence check
  treats `.pulling` markers as missing, response shape becomes
  `{ primary_job_id, companion_jobs: [{name, job_id}] }`).

### Carried forward from 2.7 — read this carefully

**The Civitai primary enqueue path is still a stub.** 2.7 returns
`primary_job_id: null` for `cv:` rows because `DownloadQueue::enqueue`
takes a manifest name and Civitai checkpoints have no manifest entry.
Companions enqueue cleanly (synthetic manifests handle them); the
primary safetensors still needs a recipe-driven path. From the 2.7
commit body:

```
Civitai entries surface `null` until the recipe-driven primary
download path lands in 2.8.
```

This is **explicitly deferred** to 2.8 — the 2.7 handoff sanctioned
"primary stays best-effort" so the server-side companion ordering
could land first without bleeding into recipe plumbing. 2.8's whole
job is making `cv:` primaries actually download.

**The candle-transformers-mold accessor bump is still pending too.**
The 2.6/2.7 handoffs flagged it as a "follow-up commit before 2.10
UAT" — neither 2.6 nor 2.7 bundled it. 2.8 should brainstorm with the
user whether to bundle the candle-mold bump (Option A) or keep it as a
separate 2.8.5 commit (Option B). Same trade-off as 2.7's Option A vs B
question.

### Not yet done

- 2.8 — **this handoff's task** (CLI recipe path with companion auto-pull).
- 2.8.5 (provisional) — candle-mold accessor bump + `load()` real-shape
  wiring if 2.8 doesn't include it.
- 2.9 web gate flip — `cat.canDownload(entry)` from `engine_phase === 1`
  to `<= 2`.
- 2.10 UAT — full killswitch run (Pony / Juggernaut XL / DreamShaper 8).

## What 2.8 produces

Per the canonical phase-2 spec
(`tasks/catalog-expansion-phase-2-handoff.md` § "Task list" 2.8 row):

> CLI `mold pull cv:<id>` integration — when the catalog row has
> `bundling: SingleFile` + `engine_phase ≤ 2`, dispatch through the new
> recipe path with companion auto-pull instead of the manifest-name
> fallback. Same companion-first ordering as 2.7.

Three concrete deliverables:

### A. Recipe-driven download for Civitai primaries

`crates/mold-cli/src/commands/pull.rs` already accepts `cv:` and `hf:`
catalog ids; today the catalog lookup runs but the actual download
still re-routes through the manifest path (which fails for Civitai).
2.8 makes the recipe path real:

1. Read the catalog row's `download_recipe.files: Vec<RecipeFile>`
   (each entry carries `url`, `dest`, `sha256`, `size_bytes`).
2. For each file, fetch the URL into the dest path. The destination is
   relative — resolve against `MOLD_MODELS_DIR/<sanitized-id>/`.
3. Verify SHA-256 if provided (mirror `mold_core::download::verify_sha256`).
4. Write a `.pulling` marker before the first byte and remove it after
   verification — same lifecycle the manifest path uses.

The Civitai HTTP fetcher needs auth: rows with
`download_recipe.needs_token: Some(Civitai)` require the user's Civitai
token (`CIVITAI_TOKEN` env var or config). Mirror the HF token lookup
pattern in `mold_core::download::resolve_hf_token`.

### B. Same companion-first ordering as 2.7

Before fetching the Civitai recipe files, resolve each canonical
companion via `mold_core::manifest::find_manifest`, check on-disk
presence (the same `pulling_marker_path_in` + file-list-walk check
2.7 uses), and pull missing companions through the existing
`mold_core::download::pull_and_configure_with_callback` path.

The companion-pull helper in 2.7 lives behind `pub(crate)` in
`crates/mold-server/src/catalog_api.rs::enqueue_missing_companions`.
2.8 might want to lift the on-disk presence check into `mold-core`
(`mold_core::download::companion_present_on_disk`?) so both server +
CLI share the same implementation. Pre-grep usage before pulling.

### C. Server-side recipe path (optional but useful)

If 2.8 also wires the server's `post_catalog_download` to accept
recipe-driven Civitai primaries, the web client's `primary_job_id`
stops being `null` for `cv:` entries. This is a bigger lift than the
CLI work because it needs a `DownloadQueue` extension or a parallel
recipe queue. Two paths:

- **Option B'** (cautious): 2.8 ships CLI only; server `primary_job_id`
  stays `null` for `cv:` entries until a 2.8.5 commit. Web users run
  `mold pull cv:<id>` from a terminal until then.
- **Option A'** (aggressive): 2.8 extends `DownloadQueue` to accept a
  recipe (`enqueue_recipe(name, files: Vec<RecipeFile>)`). Server now
  surfaces a real `primary_job_id` for Civitai entries. Bigger blast
  radius — the queue's idempotency contract has to extend to recipe
  identity.

The session driving 2.8 should brainstorm Option A' vs B' with the
user before starting. B' keeps 2.8 small; A' unblocks 2.10's UAT
through the web UI.

### TDD shape

Three rounds, mirroring 2.4–2.7:

#### Round 1 — recipe-driven file fetch unit tests

`recipe_fetcher_writes_files_under_models_dir` — synthetic recipe with
2 files, mock HTTP, assert files land at `models_dir/<id>/<dest>`.
`recipe_fetcher_verifies_sha256_when_present` — file with SHA mismatch
must error; matching SHA must succeed. `recipe_fetcher_writes_pulling_marker`
— marker present mid-download, removed post-success.

#### Round 2 — companion ordering tests

`mold_pull_cv_id_pulls_companions_first` — exercise the CLI flow with
a mocked recipe-and-companion driver; assert call order is
companions-then-primary. `mold_pull_cv_id_skips_present_companions` —
pre-stage clip-l, assert it's skipped.

#### Round 3 — civitai token threading

`recipe_fetch_uses_civitai_token_when_required` — recipe with
`needs_token: Civitai`, env var set, assert Authorization header
present on the mock HTTP request. `recipe_fetch_returns_clear_error_when_token_missing`
— same recipe without env var, assert error message points at the env
var name.

## Out of scope for 2.8

- **Server recipe path** — unless explicitly chosen as part of 2.8
  (Option A' above).
- **Web gate flip** — `cat.canDownload(entry)` predicate is 2.9.
- **End-to-end UAT** — real Civitai pulls + generations are 2.10.
- **The candle-mold accessor bump** — unless explicitly chosen as part
  of 2.8 (Option A above).
- **Glob expansion** — companion `files: ["tokenizer*.json"]` globs
  from `mold_catalog::companions::COMPANIONS` aren't materialised in
  the synthetic manifests (2.7 used the unified `tokenizer.json`).
  Phase 3+ flows that need vocab/merges separately can revisit then.

## Working conventions to preserve

- TDD — failing tests first per round.
- One scope per commit. 2.8 is one commit:
  `feat(cli): catalog cv:<id> recipe-driven pull (phase 2.8)` (plus
  optional second commit `chore(deps): bump candle-transformers-mold
  to 0.9.11 with SD config accessors` if Option A).
- **Phase 2 lands as one push when 2.10 is gate-green.** All phase-2
  commits stay local.
- `superpowers:subagent-driven-development` is appropriate here — the
  task is mechanical (HTTP fetch + companion ordering).
- `superpowers:verification-before-completion` before declaring done.
- **Pre-grep cross-crate before subagent dispatch.**
  `grep -rn 'pull_model\|pull_and_configure\|DownloadRecipe' crates/`
  to enumerate the surfaces 2.8 plumbs through.

## Verification commands

```bash
cd /Users/jeffreydilley/github/mold

# Pre-flight
git status                                      # clean
git log --oneline origin/main..HEAD | head -14  # phase-2 commit chain so far

# After implementation
cargo fmt --all -- --check
cargo clippy --workspace --all-targets -- -D warnings
cargo test -p mold-ai --lib
cargo test --workspace                          # retry once on TUI flake before blaming your changes
```

The TUI theme test flake
(`theme_save_then_load_round_trip_preserves_preset`) is documented in
user memory — retry once before blaming your changes. It did **not**
flake during 2.7 verification.

## Reference reading

- `tasks/catalog-expansion-phase-2-handoff.md` — overall phase-2 brief.
- `tasks/catalog-expansion-phase-2-task-2.7-handoff.md` — 2.7 brief +
  the candle-fork-accessor blocker section ("staged behind a
  candle-transformers-mold fork bump").
- `crates/mold-cli/src/commands/pull.rs` — the primary edit site for
  CLI catalog-id parsing + recipe routing.
- `crates/mold-catalog/src/entry.rs` — `DownloadRecipe` /
  `RecipeFile` shape.
- `crates/mold-core/src/download.rs` — `pull_and_configure_with_callback`,
  `pulling_marker_path_in`, `verify_sha256`, `resolve_hf_token`. The
  CLI's recipe fetcher should mirror these patterns.
- `crates/mold-server/src/catalog_api.rs::enqueue_missing_companions`
  — the 2.7 helper. Lift into `mold-core` for cross-crate reuse.
- `crates/mold-catalog/src/companions.rs` — the canonical companion
  registry (single source of truth).

---

## The prompt

Paste from here into a fresh Claude Code session:

---

I'm starting **task 2.8** of the mold catalog-expansion phase 2 — CLI
recipe-driven `mold pull cv:<id>` with companion auto-pull. Tasks
2.1–2.7 are done. **2.7 (commit `520e69a`, local-only) shipped server
companion auto-pull and the new response schema, but Civitai primary
entries still surface `primary_job_id: null` because `DownloadQueue`
takes a manifest name and there's no recipe-driven path yet.** 2.8's
whole job is making the recipe path real.

## Read first, in this order

1. `~/.claude-personal/CLAUDE.md` and
   `/Users/jeffreydilley/github/mold/CLAUDE.md` — coding conventions.
2. `tasks/catalog-expansion-phase-2-handoff.md` — overall phase-2 brief.
   **§ "Task list" 2.8 row** is the canonical spec.
3. `tasks/catalog-expansion-phase-2-task-2.8-handoff.md` — this file.
   Spells out the recipe-driven fetch path, companion-first ordering
   (mirror 2.7), and the Option A'/B' question for whether to extend
   `DownloadQueue` server-side.
4. `tasks/catalog-expansion-phase-2-task-2.7-handoff.md` — 2.7 context,
   in particular the candle-fork-accessor blocker that 2.8 may want to
   bundle.
5. `crates/mold-cli/src/commands/pull.rs` — the primary edit site.
   Grep for `cv:` to find the existing catalog-id branch.
6. `crates/mold-catalog/src/entry.rs::DownloadRecipe` — the recipe
   shape (`Vec<RecipeFile>`, `needs_token: Option<TokenKind>`).
7. `crates/mold-core/src/download.rs` — `pull_and_configure_with_callback`,
   marker lifecycle, SHA-256 verify, token resolution.
8. `crates/mold-server/src/catalog_api.rs::enqueue_missing_companions`
   — the 2.7 helper. Consider lifting into `mold-core` for CLI reuse.

## What you're doing

Implement task 2.8 per this handoff. Three deliverables:

1. **Recipe-driven download for Civitai primaries** — read
   `download_recipe.files`, fetch each URL into
   `models_dir/<sanitized-id>/<dest>`, verify SHA-256, manage `.pulling`
   marker. Mirror the manifest path's lifecycle.
2. **Companion-first ordering** — same on-disk presence check as 2.7,
   pulled through `mold_core::download::pull_and_configure_with_callback`
   for each missing companion before the primary. Lift the helper
   into `mold-core` so both server and CLI share an implementation.
3. **Civitai token threading** — `download_recipe.needs_token:
   Some(Civitai)` requires `CIVITAI_TOKEN`; surface a clear error
   message when missing.

Optional fourth deliverable (Option A'): extend `DownloadQueue` to
accept a recipe so the server's `post_catalog_download` returns a real
`primary_job_id` for Civitai entries. **Brainstorm Option A' vs B'
with the user before starting.**

TDD: three rounds, failing tests first per round, mirroring 2.4–2.7.

## How to work

1. Pre-flight: confirm `git status` clean and `cargo test --workspace`
   green before touching code.
2. **Pre-grep cross-crate.**
   `grep -rn 'pull_and_configure\|DownloadRecipe\|cv:' crates/mold-cli/ crates/mold-core/`
   to enumerate the surfaces.
3. Use `superpowers:test-driven-development`. Round 1 first — recipe
   fetcher is pure-data over a mocked HTTP client.

## Verification gate before committing

```bash
cargo fmt --all -- --check
cargo clippy --workspace --all-targets -- -D warnings
cargo test -p mold-ai --lib
cargo test --workspace                  # retry once on TUI flake (`theme_save_then_load_round_trip_preserves_preset`)
```

## Commit shape

Single commit when gate-green:

```
feat(cli): catalog cv:<id> recipe-driven pull (phase 2.8)

`mold pull cv:<id>` now reads `CatalogRow.download_recipe`, pulls
companion canonical files first (skipping ones already on disk), then
fetches each `RecipeFile` URL into `MOLD_MODELS_DIR/<sanitized-id>/`,
verifies SHA-256 when provided, and manages the `.pulling` marker
lifecycle. Civitai entries with `download_recipe.needs_token:
Civitai` resolve `CIVITAI_TOKEN` from env / config; missing tokens
surface a pointed error message. Companion presence check lifted
from `mold_server::catalog_api::enqueue_missing_companions` into
`mold_core::download::companion_present_on_disk` so the CLI and
server share one implementation.

[If Option A': also extends DownloadQueue::enqueue_recipe so the
server's post_catalog_download surfaces a real primary_job_id for
Civitai entries instead of null.]

[If Option A: also bumps candle-transformers-mold to 0.9.11 to expose
StableDiffusionConfig.unet/.autoencoder accessors, and wires
VarBuilder::from_backend(SingleFileBackend) into the
load_components_single_file stubs that 2.6 left behind.]

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
```

**Do not push.** Phase 2 lands as one push at 2.10. All 2.3-2.8
feat commits stay local.

## If you hit a surprise

- If Civitai's HTTPS endpoint refuses the request without a stricter
  User-Agent or extra query params (the catalog scanner's throttle
  layer might be tuned differently than a direct download client),
  surface the gap before working around it. The phase-1 scanner code
  in `mold-catalog` may have prior art worth grepping.
- If a recipe lists a file with `dest: "..."` containing a path
  traversal (`../etc/passwd`), the fetcher must reject it — paths
  must stay under the per-id models-dir subtree. Belt-and-braces:
  canonicalize and verify the parent prefix matches.
- If lifting `companion_present_on_disk` into `mold-core` introduces a
  cross-crate dependency cycle (mold-server already depends on
  mold-core), surface and ask. The clean path may be a `mold_core::companions`
  module that doesn't reach into `mold_catalog`.
- If the candle-mold fork bump path (Option A) hits crates.io
  publishing friction, fall back to Option B and document the gap in
  a 2.8.5 follow-up commit.

When 2.8 is gate-green and the test rounds pass, write
`tasks/catalog-expansion-phase-2-task-2.9-handoff.md` (template: this
file) and stop. Do not start 2.9 in the same session.
