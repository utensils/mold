# multi-prompt-chain v2 — session 4 resume handoff

> Paste the prompt at the bottom of this file into a fresh Claude Code session
> to continue sub-project A starting from **Phase 4**. Everything above the
> prompt is reference material.

## Status on handoff (2026-04-22, end of session 3)

### Sub-project A consolidation

Phases 1/2/3 are consolidated into a single combined PR, rebased against `main`.

| PR | State | Base | Head |
|---|---|---|---|
| [#268](https://github.com/utensils/mold/pull/268) | open | `main` | `feat/multi-prompt-chain-v2-phase3` |
| ~~#266~~ | closed (consolidated into #268) | — | — |
| ~~#267~~ | closed (consolidated into #268) | — | — |

**Branch note:** the head branch is named `feat/multi-prompt-chain-v2-phase3` for historical reasons (it was originally Phase 3's stacked branch). After consolidation it contains 35 commits covering all three phases (Phase 1 wire, Phase 2 engine, Phase 3 CLI) plus the design/plan docs. The name is a wart; don't let it confuse you.

### Combined gate (green on HEAD `a92d820`)

- `cargo fmt --all -- --check` ✓
- `cargo clippy --workspace --all-targets -- -D warnings` ✓
- `cargo test --workspace` ✓ — 2382 passed, 0 failed

The TUI theme test `theme_save_then_load_round_trip_preserves_preset` is a known flake; retry once if it trips.

### Phase 2 manual verification (still pending)

**Killswitch end-to-end smoke on `<gpu-host>` (CUDA <arch-tag>) is NOT yet done.** Three renders (smooth / cut / fade / mixed) with visual seam inspection. This is user work — the renders complete on the GPU, but "do the seams look right" is a human judgment.

To run it:

```bash
ssh <gpu-host>
cd ~/github/mold
git fetch origin feat/multi-prompt-chain-v2-phase3
git checkout feat/multi-prompt-chain-v2-phase3
CUDA_COMPUTE_CAP=86 nix build .#mold
./result/bin/mold serve --bind 0.0.0.0 --port 7680 &

# Hand-author chain_smooth/cut/fade/mixed.toml per the spec (or use curl
# with JSON bodies) and hit /api/generate/chain. scp the MP4s back.
# Acceptance criteria are in PR #268's "Test plan" section.
```

### Deferred minors (carried from Phase 2, still not blocking)

- `fade_frames.unwrap_or(8)` duplicates `DEFAULT_FADE_FRAMES` (which is private in `mold_core::chain`).
- `"stage render failed"` duplicated between construction site and `#[schema(example)]`.
- `ChainProgressEvent::Stitching { total_frames }` over-reports by `(N-1) * motion_tail` (cosmetic).
- `StitchPlan::assemble` on `Smooth` with `ClipTooShortForTrim` returns 500 today; arguably 422. Unreachable in practice because `normalise` validates.

## Phase 4 — TUI script mode

**Goal:** A `ScriptComposer` ratatui widget in the TUI for authoring chain scripts interactively.

**Commit scope:** `feat(tui)`. Plan section starts at `docs/superpowers/plans/2026-04-21-multi-prompt-chain-v2.md` line 3295.

**Tasks 4.1–4.8 (8 tasks).** Summary:

- 4.1: `ScriptComposer` widget scaffolding under `crates/mold-tui/src/ui/script_composer.rs` + `Mode::Script` variant in `app.rs`.
- 4.2: Stage list with `j/k` navigation, `J/K` reorder.
- 4.3: `a`/`A` add stage (after/before current), `d` delete.
- 4.4: `t` cycle transition (Smooth → Cut → Fade → Smooth), `f` editor for `fade_frames`.
- 4.5: `i` prompt editor (opens a multi-line text area).
- 4.6: `Ctrl-S`/`Ctrl-O` save/load TOML (via `mold_core::chain_toml`).
- 4.7: `Enter` submits — drives `/api/generate/chain/stream` with progress forwarded into the TUI's status bar.
- 4.8: `tui-uat.sh` scenario covering a 3-stage author-save-submit flow.

## Phase 5 — Web composer script mode

**Goal:** A Vue-side composer for chain authoring that round-trips the canonical TOML shape and posts via `/api/generate/chain/stream`.

**Commit scope:** `feat(web)`. Plan section starts at line 3907.

**Tasks 5.1–5.10 (10 tasks).** Summary:

- 5.1: Add `smol-toml` (or `@iarna/toml` fallback) dep. `bun run verify` must pass.
- 5.2: `web/src/chain/chainToml.ts` — parse/serialize TOML mirroring `mold_core::chain_toml`.
- 5.3: `fetchChainLimits()` in `web/src/api/chain.ts` — hits `/api/capabilities/chain-limits`.
- 5.4: `StageCard.vue` — renders one stage's prompt/frames/transition + fade_frames.
- 5.5: `ScriptComposer.vue` — list of StageCards + top-bar controls.
- 5.6: Mode toggle on `Composer.vue` — switches between single-prompt and script modes.
- 5.7: Chain submit path — posts `ChainRequest`, consumes SSE `progress`/`complete`/`error`.
- 5.8: Per-stage expand modal — calls `/api/expand` with stage prompt.
- 5.9: Drag-reorder via `vue-draggable-plus` — reorders `StageCard` list in-place.
- 5.10: Footer chain-limits clamp — read `ChainLimits.max_stages` / `max_total_frames` and disable "add stage" when at cap. `bun run verify && bun run build` gate.

## Phase 6 — Docs + release

**Goal:** Document the new composer UX for end users; update the changelog and skill file.

**Commit scope:** `docs(chain)`. Plan section starts at line 4603.

**Tasks 6.1–6.4 (4 tasks).**

- 6.1: `website/guide/video.md` — new section on multi-prompt chains + TOML authoring + transitions.
- 6.2: `CHANGELOG.md [Unreleased]` — bullet list of new API endpoints, CLI flags, TUI commands, web composer.
- 6.3: `.claude/skills/mold/SKILL.md` — update for the new `mold chain validate` subcommand, `--script` flag, and `TransitionMode` concept.
- 6.4: `CLAUDE.md` — add a "chain authoring" sub-section under the CLI reference + mention the canonical TOML schema.

## Dependency graph

```
#268 (phases 1/2/3, open)
  ↓
Phase 4 (TUI)  Phase 5 (web) — concurrent safe
  └────────────┘
         ↓
      Phase 6 (docs)
```

Phases 4 and 5 are independent after the consolidated PR is open: they each consume `ChainRequest` / `/api/generate/chain` / `chain_toml` but don't touch each other's crate trees. A single session can drive them sequentially, or two parallel worktrees can run concurrently.

**Branching for next phase(s):** since the user chose consolidation for phases 1–3, you can either:

- **Option A — one combined PR for 4+5+6.** Keep stacking: create `feat/multi-prompt-chain-v2-remaining` off the current phase3 branch, do all 22 remaining tasks, open one more PR. When it's ready, the user can decide whether to merge #268 first (clean two-PR history) or keep stacking.
- **Option B — one PR per phase.** New branch per phase off `feat/multi-prompt-chain-v2-phase3` (or off main once #268 merges). Easier to review, more overhead.

The user's stated preference in session 3 was "one branch one PR" for 1–3; mirror that unless they say otherwise.

## Working conventions to preserve

- **One scope per commit** — `feat(tui) / feat(web) / docs(chain)`.
- **No mid-phase pushes** — each phase is one PR when complete. Exceptions: intentional doc-only commits like this handoff, which land on the phase branch naturally.
- **TDD** — failing test first, then implementation, then verify pass, then commit.
- **`superpowers:subagent-driven-development`** — fresh general-purpose subagent per task. Sonnet for mechanical tasks, opus only if architectural judgment needed. Combined spec+quality review works well for plan-verbatim tasks; split into two-stage for complex/cross-crate work.
- **Prompt the subagent with full task text** — don't send "read the plan and implement". Paste the plan's text inline and flag pre-investigation findings.
- **Pre-grep before dispatching** — when a task changes a public struct, `grep -rn 'StructName {' crates/` and list every construction site in the brief.

## Gotchas accumulated through Phase 3

Carried forward from earlier sessions, still relevant:

- **`MoldError::Other`** takes `anyhow::Error`, not `String`.
- **`mold_core::manifest`** has `find_manifest(&name).map(|m| m.family.clone())` for family lookup.
- **`toml-rs` sorts keys alphabetically** within tables — `write_script` prepends the `schema = "..."` header manually.
- **`ChainSseMessage::Complete(Box<SseChainCompleteEvent>)`** — boxed for `large_enum_variant`. Don't un-box.
- **TUI theme test flake** — retry once if only `theme_save_then_load_round_trip_preserves_preset` fires.

From Phase 2:

- **`ChainOrchestratorError`** (in `mold-inference::ltx2`) is the typed error from `Ltx2ChainOrchestrator::run`. `Invalid(anyhow::Error)` for pre-flight failures; `StageFailed { stage_idx, elapsed_stages, elapsed_ms, inner: anyhow::Error }` for mid-chain failures.
- **`ChainFailure`** (in `mold-core::chain`) is the wire-type version of the above, returned as the 502 JSON body from `generate_chain` (non-SSE only; SSE collapses to a string).
- **`ChainRunError::StageFailed(ChainFailure)`** is the server-internal variant bridging the two. `From<ChainRunError> for ApiError` has a comment explaining the intentional SSE lossiness.
- **`generate_chain` signature** is `axum::response::Response` (not `Result<Json<ChainResponse>, ApiError>`) to support the structured 502 body. The SSE handler `generate_chain_stream` is unchanged.
- **`StitchPlan`** in `mold-inference::ltx2::stitch` is the per-boundary stitch assembler. Called from `routes_chain.rs::stitch_chain_output`.
- **`fade_frames`** defaults are `8` in two places: `mold_core::chain::DEFAULT_FADE_FRAMES` (private `const`) and the literal `8` in `stitch_chain_output` + `mixed_transitions_end_to_end`.

New in Phase 3 / session 3:

- **`commands/chain.rs` is the CLI chain hub** — both `run_from_script` (Task 3.2) and `run_from_sugar` (Task 3.3) live here, as does `run_chain`, `render_chain_progress`, and `StageLabel`. Don't fragment this.
- **`build_request_from_script` is `pub(crate)`** — `chain_validate.rs` (Task 3.4) imports it. Don't make it `pub`.
- **`read_script_resolving_paths` is a forward-compat wrapper** — currently delegates to `read_script` and ignores `script_dir`. Real path-string `source_image` resolve is a future schema bump.
- **`run_from_sugar`'s LTX-2 defaults are hardcoded** — 1216×704, 24 fps, 8 steps, 3.0 guidance. Sub-project C will read these from the manifest; don't rewire now.
- **`StageLabel`** is a private owned struct built once per run from the normalised request and moved into the render task so it never holds a `&ChainRequest` (which carries `source_image` bytes). This is why `render_chain_progress` takes `Vec<StageLabel>` rather than `&ChainRequest`.
- **Positional+flag rejection fires BEFORE the `prompt.len() > 1` sugar branch** — so `mold run foo "positional" --prompt "a" --prompt "b"` errors loudly instead of silently dropping the positional.
- **Adversarial corpus lives in `crates/mold-cli/tests/adversarial/`** — five `.toml` fixtures exercised by `crates/mold-cli/tests/chain_validate_corpus.rs`. The tests use `predicate::str::contains("X").or(contains("Y"))` to tolerate minor error-message rewording.
- **New integration-test pattern** — top-level `.rs` test files under `crates/mold-cli/tests/` share the `common/` harness (`mod common;` then `use common::TestEnv;`). The corpus test follows this, not the "append to cli_integration.rs" convention.

## Verification commands the new session will need

```bash
# Pre-flight: confirm branch state
cd /Users/jeffreydilley/github/mold
git fetch origin --prune
git checkout feat/multi-prompt-chain-v2-phase3
git log --oneline origin/main..HEAD | head -40
# Should show 35 commits (Phases 1/2/3 + spec + plan + handoff docs).

# Sanity check CI before writing any new code
cargo fmt --all -- --check
cargo clippy --workspace --all-targets -- -D warnings
cargo test --workspace  # 2382 pass expected; retry once on the TUI theme flake

# Phase 4 kickoff (Option A — keep stacking):
# Continue on feat/multi-prompt-chain-v2-phase3 and let Phase 4 commits land there.
# The PR #268 will auto-expand as new commits are pushed.

# Phase 4 kickoff (Option B — separate PR):
git checkout -b feat/multi-prompt-chain-v2-phase4
# ... work tasks 4.1-4.8 ...
git push -u origin feat/multi-prompt-chain-v2-phase4
gh pr create --base feat/multi-prompt-chain-v2-phase3 --head feat/multi-prompt-chain-v2-phase4 \
  --title "feat(tui): script mode for chain authoring (Phase 4/6)" --body "..."
```

## Spec & plan references

- Spec: `docs/superpowers/specs/2026-04-21-multi-prompt-chain-v2-design.md` (681 lines). §3 decisions NOT up for debate.
- Plan: `docs/superpowers/plans/2026-04-21-multi-prompt-chain-v2.md` (4805 lines, 51 tasks). Phase 4 starts at line 3295; Phase 5 at 3907; Phase 6 at 4603.
- Session-3 handoff: `tasks/multi-prompt-chain-v2-resume-phase3.md`.
- Session-2 handoff: `tasks/multi-prompt-chain-v2-resume-phase2.md`.
- Session-1 handoff: `tasks/multi-prompt-chain-v2-handoff.md`.

---

## The prompt

Paste from here into a fresh Claude Code session:

---

I'm resuming execution of **multi-prompt chain v2, sub-project A** for the mold repo, starting at **Phase 4 (TUI script mode)**. This is session 4 of a multi-session project. Phases 1, 2, and 3 are all consolidated into a single open PR ([#268](https://github.com/utensils/mold/pull/268)) targeting `main`.

## Read first, in this order

1. `~/.claude-personal/CLAUDE.md` (global) and `/Users/jeffreydilley/github/mold/CLAUDE.md` (project) — coding conventions.
2. `tasks/multi-prompt-chain-v2-resume-phase4.md` — **your primary briefing**. Read end-to-end. Phase 3 close-out status, what's next, working conventions, gotchas.
3. `docs/superpowers/specs/2026-04-21-multi-prompt-chain-v2-design.md` — approved design. §3 decisions not up for debate.
4. `docs/superpowers/plans/2026-04-21-multi-prompt-chain-v2.md` — 51-task plan. Phase 4 starts ~line 3295.
5. `tasks/multi-prompt-chain-v2-resume-phase3.md`, `tasks/multi-prompt-chain-v2-resume-phase2.md`, `tasks/multi-prompt-chain-v2-handoff.md` — prior session context in reverse-chronological.

## Status on entry

- Combined PR #268 targets `main` and contains 35 commits across phases 1/2/3 plus design/plan docs. Do not merge it from this session without explicit user confirmation.
- Current branch: `feat/multi-prompt-chain-v2-phase3` (name is historical; contains all of phases 1/2/3 after the session-3 consolidation — don't be thrown by it).
- Phase 2's <gpu-host> smoke is still pending manual verification. If the user hasn't run it yet and asks you to, see `tasks/multi-prompt-chain-v2-resume-phase4.md` §"Phase 2 manual verification".
- `cargo test --workspace` → 2382 passed on HEAD. TUI theme flake retries once.

## What you're doing

Execute Tasks **4.1 → 4.8** (Phase 4, TUI script mode) via `superpowers:subagent-driven-development`. Commit scope `feat(tui)`.

Decide branching strategy with the user on entry:
- **Option A:** keep stacking on `feat/multi-prompt-chain-v2-phase3`; let #268's diff expand as Phase 4 commits land. Simplest; matches session-3 precedent of consolidation.
- **Option B:** create `feat/multi-prompt-chain-v2-phase4` off the current branch and open a new PR stacked on #268.

Default to Option A unless the user requests B.

After Phase 4, consider whether to proceed with Phase 5 (web, 10 tasks) and Phase 6 (docs, 4 tasks) in the same session or hand off. Phases 4 and 5 are concurrent-safe; Phase 6 depends on 4/5.

Gate every phase with `cargo fmt --check && cargo clippy --workspace --all-targets -- -D warnings && cargo test --workspace`. Phase 5 also needs `bun run verify && bun run build` in `website/` AND `bun run verify && bun run build` in `web/`. Phase 6 is docs-only; the doc gate is still `cargo fmt/clippy/test` passing plus spot-checking that documented APIs still exist.

## Start here

1. Confirm state:
   ```bash
   cd /Users/jeffreydilley/github/mold
   git fetch origin --prune
   git checkout feat/multi-prompt-chain-v2-phase3
   git log --oneline origin/main..HEAD | head -40
   git status
   cargo test --workspace  # should be green; retry once on TUI flake
   ```

2. Pre-investigation for Task 4.1: understand the existing TUI modes.
   ```bash
   ls crates/mold-tui/src/ui/ crates/mold-tui/src/
   grep -rn 'enum Mode\|Mode::' crates/mold-tui/src/app.rs | head -20
   ```

3. Start Task 4.1 (`ScriptComposer` widget scaffolding). Plan at line 3303.

## If you hit a surprise

Read `tasks/multi-prompt-chain-v2-resume-phase4.md` §"Gotchas accumulated through Phase 3" before assuming anything about typed error surfaces, `StageLabel`, `build_request_from_script` visibility, or the branch's misleading name. If the surprise is beyond that list, stop and ask the user.
