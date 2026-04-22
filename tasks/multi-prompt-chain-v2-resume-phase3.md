# multi-prompt-chain v2 — session 3 resume handoff

> Paste the prompt at the bottom of this file into a fresh Claude Code session
> to continue sub-project A starting from **Phase 3**. Everything above the
> prompt is reference material.

## Status on handoff (2026-04-22)

### Branches & PRs

| Phase | Branch | PR | State | Base |
|---|---|---|---|---|
| 1 | `feat/multi-prompt-chain-v2` | [#266](https://github.com/utensils/mold/pull/266) | open | `main` |
| 2 | `feat/multi-prompt-chain-v2-phase2` | [#267](https://github.com/utensils/mold/pull/267) | open, stacked on #266 | `feat/multi-prompt-chain-v2` |
| 3/4/5/6 | — | — | not started | — |

**Stacking rule for future phases:** each phase gets its own branch `feat/multi-prompt-chain-v2-phaseN` stacked on the predecessor branch. When #266 merges to `main`, GitHub auto-retargets #267 to `main`. Same pattern for subsequent phases.

### Phase 2 automated gate (green on HEAD `10f435a`)

- `cargo fmt --all -- --check` ✓
- `cargo clippy --workspace --all-targets -- -D warnings` ✓
- `cargo test --workspace` ✓ — full suite green, no TUI flake this run

### Phase 2 manual verification (pending)

**Killswitch end-to-end smoke on `killswitch@192.168.1.67` (CUDA sm_86) is NOT yet done.** The plan called for three renders (smooth / cut / fade / mixed) with visual inspection of seam quality. This is user work — the renders complete on the GPU, but whether the seams *look right* is a human judgment.

To run it:

```bash
ssh killswitch@192.168.1.67
cd ~/github/mold
git fetch origin feat/multi-prompt-chain-v2-phase2
git checkout feat/multi-prompt-chain-v2-phase2
CUDA_COMPUTE_CAP=86 nix build .#mold
./result/bin/mold serve --bind 0.0.0.0 --port 7680 &

# Per chain_smooth/cut/fade/mixed.json, POST to /api/generate/chain and
# scp the MP4s back. Visual acceptance criteria are in PR #267's body.
```

If the smoke reveals a regression, fix on `feat/multi-prompt-chain-v2-phase2` and update #267. If it passes, flag that in the PR.

### Deferred minors (tracked in PR #267 body)

- Task 2.6: dedicated route-layer Cut/Fade test is missing (covered at adjacent layers).
- Task 2.6: `fade_frames.unwrap_or(8)` duplicates `DEFAULT_FADE_FRAMES` (which is private).
- Task 2.7: `"stage render failed"` duplicated between construction site and `#[schema(example)]`.
- `ChainProgressEvent::Stitching { total_frames }` over-reports by `(N-1) * motion_tail` (cosmetic).
- `StitchPlan::assemble` on `Smooth` with `ClipTooShortForTrim` → 500 today; arguably 422. Unreachable in practice because `normalise` validates.

None block Phase 2 merge.

## Phase 3 — CLI surface

**Goal:** `mold run --script shot.toml` canonical path + repeated `--prompt` sugar for uniform trivial chains + `mold chain validate` subcommand + `--dry-run`. Integrates with the existing chain endpoint from Phase 1.

**Commit scope:** `feat(cli)`. Plan section starts at `docs/superpowers/plans/2026-04-21-multi-prompt-chain-v2.md` line 2674.

**Tasks 3.1–3.7 (7 tasks).** Summary:

- 3.1: Add `--script` flag + `--dry-run` flag to `mold run` (clap parser).
- 3.2: `run_from_script` helper in `commands/chain.rs` + CLI integration test for `--dry-run`.
- 3.3: Repeated `--prompt` sugar — `mold run --prompt "..." --prompt "..." --prompt "..."` auto-expands into `ChainRequest.stages`.
- 3.4: `mold chain validate <path>` subcommand (schema gate, normalise, pretty-print stage summary).
- 3.5: `mold run --script` posts to `/api/generate/chain/stream` with SSE progress display, including transition tags per stage.
- 3.6: Progress-bar integration honoring `ChainStart.estimated_total_frames` and per-stage transitions.
- 3.7: Phase 3 gate — `cargo fmt && cargo clippy && cargo test` + CLI smoke (`mold chain validate` against a hand-authored TOML).

## Phase 4 — TUI script mode

**Goal:** A `ScriptComposer` ratatui widget in the TUI for authoring chain scripts interactively.

**Commit scope:** `feat(tui)`. Plan section starts at line 3295.

**Tasks 4.1–4.8 (8 tasks).** Summary:

- 4.1: `ScriptComposer` widget scaffolding under `crates/mold-tui/src/widgets/script_composer.rs`.
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
Phase 1 (merged before Phase 2 lands)
  ↓
Phase 2 (#267, pending merge)
  ↓
Phase 3 (CLI)  Phase 4 (TUI)  Phase 5 (web) — concurrent safe
  └────────────┴──────────────┘
                ↓
             Phase 6 (docs)
```

Phases 3/4/5 are independent after Phase 2 lands: they each consume `ChainRequest` / `/api/generate/chain` / `chain_toml` but don't touch each other's crate trees. A single session can drive all three sequentially, or three parallel worktrees can run concurrently.

Phase 6 depends on 3/4/5 having landed (or at least having their final shape locked) because the docs describe the final CLI/TUI/web surface.

## Working conventions to preserve

- **One scope per commit** — `feat(chain) / feat(ltx2) / feat(server) / feat(cli) / feat(tui) / feat(web) / test(chain) / docs(chain)`.
- **No mid-phase pushes** — each phase is one PR when complete. Exceptions: intentional doc-only commits like this handoff, which land on the phase branch naturally.
- **TDD** — failing test first, then implementation, then verify pass, then commit.
- **`superpowers:subagent-driven-development`** — fresh general-purpose subagent per task. Sonnet for mechanical tasks, opus only if the task requires architectural judgment. Combined spec+quality review works well for plan-verbatim tasks; split into two-stage for complex/cross-crate work (as done for Task 2.7).
- **Prompt the subagent with full task text** — don't send a "read the plan and implement" instruction. Paste the plan's text inline and flag pre-investigation findings.
- **Pre-grep before dispatching** — when a task changes a public struct, `grep -rn 'StructName {' crates/` and list every construction site in the brief.

## Gotchas accumulated through Phase 2

Carried forward from session-2 (still relevant):

- **`MoldError::Other`** takes `anyhow::Error`, not `String`.
- **`mold_core::manifest`** has `find_manifest(&name).map(|m| m.family.clone())` for family lookup.
- **`toml-rs` sorts keys alphabetically** within tables — `write_script` prepends the `schema = "..."` header manually.
- **`ChainSseMessage::Complete(Box<SseChainCompleteEvent>)`** — boxed for `large_enum_variant`. Don't un-box.
- **TUI theme test flake** — retry once if only `theme_save_then_load_round_trip_preserves_preset` fires.

New in session 2 / Phase 2:

- **`ChainOrchestratorError`** (in `mold-inference::ltx2`) is the typed error from `Ltx2ChainOrchestrator::run`. `Invalid(anyhow::Error)` for pre-flight failures; `StageFailed { stage_idx, elapsed_stages, elapsed_ms, inner: anyhow::Error }` for mid-chain failures.
- **`ChainFailure`** (in `mold-core::chain`) is the wire-type version of the above, returned as the 502 JSON body from `generate_chain` (non-SSE only; SSE collapses to a string).
- **`ChainRunError::StageFailed(ChainFailure)`** is the server-internal variant bridging the two. `From<ChainRunError> for ApiError` has a comment explaining the intentional SSE lossiness.
- **`generate_chain` signature** changed from `Result<Json<ChainResponse>, ApiError>` to `axum::response::Response` to support the structured 502 body. The SSE handler `generate_chain_stream` is unchanged.
- **`StitchPlan`** in `mold-inference::ltx2::stitch` is the per-boundary stitch assembler. Called from `routes_chain.rs::stitch_chain_output`.
- **`fade_frames`** defaults are `8` in two places: `mold_core::chain::DEFAULT_FADE_FRAMES` (private `const`) and the literal `8` in `stitch_chain_output` + `mixed_transitions_end_to_end`.
- **Branch naming** — `feat/multi-prompt-chain-v2-phaseN`, each stacked on predecessor. PRs target the predecessor branch; auto-retarget on merge.

## Verification commands the new session will need

```bash
# Pre-flight: confirm branch state
cd /Users/jeffreydilley/github/mold
git fetch origin
git checkout feat/multi-prompt-chain-v2-phase2  # or skip if starting a new phase-N
git log --oneline origin/main..HEAD | head -25
# Should show 24 commits (Phase 1 + Phase 2) on this branch.

# Sanity check CI before writing any new code
cargo fmt --all -- --check
cargo clippy --workspace --all-targets -- -D warnings
cargo test --workspace

# Phase N kickoff (example for Phase 3):
git checkout -b feat/multi-prompt-chain-v2-phase3
# ... work tasks 3.1-3.7 ...
git push -u origin feat/multi-prompt-chain-v2-phase3
gh pr create --base feat/multi-prompt-chain-v2-phase2 \
  --head feat/multi-prompt-chain-v2-phase3 \
  --title "feat(cli): script mode for chain authoring (Phase 3/6)" --body "..."
```

## Spec & plan references

- Spec: `docs/superpowers/specs/2026-04-21-multi-prompt-chain-v2-design.md` (681 lines). §3 decisions NOT up for debate.
- Plan: `docs/superpowers/plans/2026-04-21-multi-prompt-chain-v2.md` (4805 lines, 51 tasks). Phase 3 starts at line 2674.
- Session-2 handoff: `tasks/multi-prompt-chain-v2-resume-phase2.md`.
- Session-1 handoff: `tasks/multi-prompt-chain-v2-handoff.md`.

---

## The prompt

Paste from here into a fresh Claude Code session:

---

I'm resuming execution of **multi-prompt chain v2, sub-project A** for the mold repo, starting at **Phase 3 (CLI surface)**. This is session 3 of a multi-session project. Phase 1 and Phase 2 are complete; their PRs are open and stacked ([#266](https://github.com/utensils/mold/pull/266), [#267](https://github.com/utensils/mold/pull/267)).

## Read first, in this order

1. `~/.claude-personal/CLAUDE.md` (global) and `/Users/jeffreydilley/github/mold/CLAUDE.md` (project) — coding conventions.
2. `tasks/multi-prompt-chain-v2-resume-phase3.md` — **your primary briefing**. Read end-to-end. Phase 2 status, what's next, working conventions, gotchas.
3. `docs/superpowers/specs/2026-04-21-multi-prompt-chain-v2-design.md` — approved design. §3 decisions not up for debate.
4. `docs/superpowers/plans/2026-04-21-multi-prompt-chain-v2.md` — 51-task plan. Phase 3 starts ~line 2674.
5. `tasks/multi-prompt-chain-v2-resume-phase2.md` and `tasks/multi-prompt-chain-v2-handoff.md` — prior session context.

## Status on entry

- Branch strategy: `feat/multi-prompt-chain-v2-phase3` will be created from `feat/multi-prompt-chain-v2-phase2` (Phase 2 tip).
- Phase 1 PR #266 + Phase 2 PR #267 both open, stacked on `main`. Do not merge them from this session without explicit user confirmation.
- Phase 2's killswitch smoke is still pending manual verification. If the user hasn't run it yet and asks you to, see `tasks/multi-prompt-chain-v2-resume-phase3.md` §"Phase 2 manual verification".

## What you're doing

Execute Tasks **3.1 → 3.7** (Phase 3, CLI surface) via `superpowers:subagent-driven-development`. Then Phase 3 PR, stacked on Phase 2. Optionally follow with Phase 4 (TUI, 8 tasks) and Phase 5 (web, 10 tasks) — these are concurrent-safe, but sequential execution is easier to track. Phase 6 (docs, 4 tasks) after 3/4/5 are in.

Gate every phase with `cargo fmt --check && cargo clippy --workspace --all-targets -- -D warnings && cargo test --workspace`. Phase 5 also needs `bun run verify && bun run build` in `website/`. **No mid-phase pushes.** One PR per phase.

## Start here

1. Confirm state:
   ```bash
   cd /Users/jeffreydilley/github/mold
   git fetch origin
   git checkout feat/multi-prompt-chain-v2-phase2
   git log --oneline origin/main..HEAD | head -25
   git status
   cargo test --workspace  # should be green; retry once on TUI flake
   ```

2. Create the Phase 3 branch:
   ```bash
   git checkout -b feat/multi-prompt-chain-v2-phase3
   ```

3. Start Task 3.1 (`--script` + `--dry-run` flags). Plan at line 2682. Pre-investigation: `grep -n 'Commands::Run' crates/mold-cli/src/main.rs` for the current `Run` variant shape.

## If you hit a surprise

Read `tasks/multi-prompt-chain-v2-resume-phase3.md` §"Gotchas accumulated through Phase 2" before assuming anything about typed error surfaces, branch strategy, or the `ChainFailure` shape. If the surprise is beyond that list, stop and ask the user.
