# multi-prompt-chain v2 — session 5 resume handoff

> Paste the prompt at the bottom of this file into a fresh Claude Code session
> to continue sub-project A starting from **Phase 5**. Everything above the
> prompt is reference material.

## Status on handoff (2026-04-22, end of session 4)

### Sub-project A consolidation

Phases 1/2/3/4 are consolidated into a single combined PR, on one branch.

| PR | State | Base | Head |
|---|---|---|---|
| [#268](https://github.com/utensils/mold/pull/268) | open | `main` | `feat/multi-prompt-chain-v2-phase3` |

**Branch note:** the head branch is named `feat/multi-prompt-chain-v2-phase3` for historical reasons (it was originally Phase 3's stacked branch). After consolidation it contains 44 commits covering all four phases (Phase 1 wire, Phase 2 engine, Phase 3 CLI, Phase 4 TUI) plus design/plan docs. The name is a wart; don't let it confuse you.

### Combined gate (green on HEAD `9c3caed`)

- `cargo fmt --all -- --check` ✓
- `cargo clippy --workspace --all-targets -- -D warnings` ✓
- `cargo test --workspace` ✓ — 2413 passed, 0 failed

The TUI theme test `theme_save_then_load_round_trip_preserves_preset` and session `save_is_atomic` are known flakes; retry once if they trip.

### Phase 2 manual verification (still pending)

**Killswitch end-to-end smoke on `killswitch@192.168.1.67` (CUDA sm_86) is NOT yet done.** See `tasks/multi-prompt-chain-v2-resume-phase4.md` §"Phase 2 manual verification" for instructions.

### Phase 4 (TUI) — completed in session 4

8 tasks completed (4.1–4.8). The TUI now has a Script view (tab 6, `s` shortcut) with:

- Stage list with `j/k` navigation, `J/K` reorder
- `a`/`A` add stages, `d` delete with confirmation
- `t` cycle transition (Smooth→Cut→Fade, no-op on stage 0)
- `i` prompt editor (multi-line, Ctrl-S to save)
- `f` frames editor (8k+1 validation)
- `Ctrl-S`/`Ctrl-O` save/load TOML via `chain_toml`
- `Enter` submits chain via `/api/generate/chain/stream`
- UAT scenario `script_mode` in `tui-uat.sh`

### Deferred minors (carried from Phase 2, still not blocking)

- `fade_frames.unwrap_or(8)` duplicates `DEFAULT_FADE_FRAMES` (which is private in `mold_core::chain`).
- `"stage render failed"` duplicated between construction site and `#[schema(example)]`.
- `ChainProgressEvent::Stitching { total_frames }` over-reports by `(N-1) * motion_tail` (cosmetic).
- `StitchPlan::assemble` on `Smooth` with `ClipTooShortForTrim` returns 500 today; arguably 422.

## Phase 5 — Web composer script mode

**Goal:** A Vue-side composer for chain authoring that round-trips the canonical TOML shape and posts via `/api/generate/chain/stream`.

**Commit scope:** `feat(web)`. Plan section starts at `docs/superpowers/plans/2026-04-21-multi-prompt-chain-v2.md` line 3907.

**Tasks 5.1–5.10 (10 tasks).** Summary:

- 5.1: Add `smol-toml` dep + `web/src/lib/chainToml.ts` module with read/write + tests.
- 5.2: `fetchChainLimits()` API helper in `web/src/api.ts` (or `web/src/api/chain.ts`).
- 5.3: `StageCard.vue` — renders one stage's prompt/frames/transition + fade_frames.
- 5.4: `ScriptComposer.vue` — list of StageCards + top-bar controls.
- 5.5: Mode toggle on the main Composer — switches between single-prompt and script modes.
- 5.6: Chain submit path — posts `ChainRequest`, consumes SSE `progress`/`complete`/`error`.
- 5.7: Per-stage expand modal — calls `/api/expand` with stage prompt.
- 5.8: Drag-reorder via `vue-draggable-plus` — reorders `StageCard` list in-place.
- 5.9: TOML import/export — file picker for load, download for save.
- 5.10: Footer chain-limits clamp — read `ChainLimits.max_stages`/`max_total_frames` and disable "add stage" when at cap. `bun run verify && bun run build` gate.

## Phase 6 — Docs + release

**Goal:** Document the new composer UX for end users; update the changelog and skill file.

**Commit scope:** `docs(chain)`. Plan section starts at line 4603.

**Tasks 6.1–6.4 (4 tasks).**

- 6.1: `website/guide/video.md` — new section on multi-prompt chains + TOML authoring + transitions.
- 6.2: `CHANGELOG.md [Unreleased]` — bullet list of new API endpoints, CLI flags, TUI commands, web composer.
- 6.3: `.claude/skills/mold/SKILL.md` — update for `mold chain validate`, `--script` flag, `TransitionMode`.
- 6.4: `CLAUDE.md` — add chain authoring sub-section.

## Dependency graph

```
#268 (phases 1/2/3/4, open)
  ↓
Phase 5 (web)
  ↓
Phase 6 (docs) — depends on both 4 and 5
```

## Working conventions to preserve

- **One scope per commit** — `feat(web) / docs(chain)`.
- **TDD** — failing test first, then implementation, then verify pass, then commit.
- **`superpowers:subagent-driven-development`** — fresh general-purpose subagent per task. Sonnet for mechanical tasks, opus only if architectural judgment needed.
- **Prompt the subagent with full task text** — don't send "read the plan and implement". Paste the plan's text inline and flag pre-investigation findings.
- **Pre-grep before dispatching** — when a task changes a public struct, grep and list every construction site in the brief.

## Gotchas accumulated through Phases 1–4

- **`MoldError::Other`** takes `anyhow::Error`, not `String`.
- **`mold_core::manifest`** has `find_manifest(&name).map(|m| m.family.clone())` for family lookup.
- **`toml-rs` sorts keys alphabetically** within tables — `write_script` prepends the `schema = "..."` header manually.
- **`ChainSseMessage::Complete(Box<SseChainCompleteEvent>)`** — boxed for `large_enum_variant`. Don't un-box.
- **TUI theme test flake** — retry once if only `theme_save_then_load_round_trip_preserves_preset` fires.
- **TUI session save flake** — `save_is_atomic_and_does_not_leave_tempfiles_behind` also flakes under parallel test runs.
- **`ChainOrchestratorError`** (in `mold-inference::ltx2`) is the typed error from `Ltx2ChainOrchestrator::run`.
- **`ChainFailure`** (in `mold-core::chain`) is the wire-type version returned as the 502 JSON body.
- **`build_request_from_script` is `pub(crate)`** in `mold-cli`.
- **`read_script_resolving_paths` is a forward-compat wrapper** — currently delegates to `read_script`.
- **`run_from_sugar`'s LTX-2 defaults are hardcoded** — 1216×704, 24 fps, 8 steps, 3.0 guidance.
- **`ScriptComposerState.modal`** has 5 variants: `Closed`, `PromptEdit`, `FramesEdit`, `SavePath`, `LoadPath`. Modal intercepts all keys when open.
- **Chain generation in TUI is server-only** — `run_chain_generation` requires a server URL.

## Web codebase notes for Phase 5

- **Stack:** Vue 3 + Vite 7 + Tailwind CSS v4.2 in `web/`.
- **Dev:** `cd web && bun install && bun run dev` (proxies `/api` to `http://localhost:7680`).
- **Build gate:** `bun run verify && bun run build` in `web/`.
- **Nix lock:** `web/bun.nix` must be regenerated after `bun.lock` changes (add deps). The Nix flake uses `bun2nix` from `web/bun.lock → web/bun.nix`.
- **API types** live in `web/src/api.ts` (or nearby). `ChainRequest`, `ChainResponse`, `ChainProgressEvent` shapes need TypeScript mirrors.
- **SSE consumption** — check how the existing single-prompt generate flow handles SSE in the web SPA and mirror that pattern for chain SSE.

## Verification commands the new session will need

```bash
# Pre-flight: confirm branch state
cd /Users/jeffreydilley/github/mold
git fetch origin --prune
git checkout feat/multi-prompt-chain-v2-phase3
git log --oneline origin/main..HEAD | head -50
# Should show 44 commits (Phases 1/2/3/4 + spec + plan + handoff docs).

# Sanity check CI before writing any new code
cargo fmt --all -- --check
cargo clippy --workspace --all-targets -- -D warnings
cargo test --workspace  # 2413 pass expected; retry once on TUI flakes

# Phase 5 web pre-flight
cd web && bun install && bun run verify && bun run build
cd ..
```

## Spec & plan references

- Spec: `docs/superpowers/specs/2026-04-21-multi-prompt-chain-v2-design.md` (681 lines). §3 decisions NOT up for debate.
- Plan: `docs/superpowers/plans/2026-04-21-multi-prompt-chain-v2.md` (4805 lines, 51 tasks). Phase 5 starts at line 3907; Phase 6 at 4603.
- Session-4 handoff: this file.
- Session-3 handoff: `tasks/multi-prompt-chain-v2-resume-phase3.md`.
- Session-2 handoff: `tasks/multi-prompt-chain-v2-resume-phase2.md`.
- Session-1 handoff: `tasks/multi-prompt-chain-v2-handoff.md`.

---

## The prompt

Paste from here into a fresh Claude Code session:

---

I'm resuming execution of **multi-prompt chain v2, sub-project A** for the mold repo, starting at **Phase 5 (web composer)**. This is session 5 of a multi-session project. Phases 1, 2, 3, and 4 are all consolidated into a single open PR ([#268](https://github.com/utensils/mold/pull/268)) targeting `main`.

## Read first, in this order

1. `~/.claude-personal/CLAUDE.md` (global) and `/Users/jeffreydilley/github/mold/CLAUDE.md` (project) — coding conventions.
2. `tasks/multi-prompt-chain-v2-resume-phase5.md` — **your primary briefing**. Read end-to-end. Phase 4 close-out status, what's next, working conventions, gotchas.
3. `docs/superpowers/specs/2026-04-21-multi-prompt-chain-v2-design.md` — approved design. §3 decisions not up for debate.
4. `docs/superpowers/plans/2026-04-21-multi-prompt-chain-v2.md` — 51-task plan. Phase 5 starts ~line 3907.
5. `tasks/multi-prompt-chain-v2-resume-phase4.md`, `tasks/multi-prompt-chain-v2-resume-phase3.md` — prior session context.

## Status on entry

- Combined PR #268 targets `main` and contains 44 commits across phases 1/2/3/4 plus design/plan docs. Do not merge it from this session without explicit user confirmation.
- Current branch: `feat/multi-prompt-chain-v2-phase3` (name is historical; contains all phases after consolidation — don't be thrown by it).
- Phase 2's killswitch smoke is still pending manual verification.
- `cargo test --workspace` → 2413 passed on HEAD. TUI theme + session save flakes retry once.

## What you're doing

Execute Tasks **5.1 → 5.10** (Phase 5, web composer) via `superpowers:subagent-driven-development`. Commit scope `feat(web)`.

Keep stacking on `feat/multi-prompt-chain-v2-phase3` (Option A, matching all previous sessions).

After Phase 5, proceed with Phase 6 (docs, 4 tasks) in the same session if time permits — it depends on both 4 and 5 being done.

Gate Phase 5 with `cargo fmt --check && cargo clippy --workspace --all-targets -- -D warnings && cargo test --workspace` AND `cd web && bun run verify && bun run build`. Phase 6 is docs-only; gate is `cargo fmt/clippy/test` passing plus spot-checking that documented APIs still exist.

## Start here

1. Confirm state:
   ```bash
   cd /Users/jeffreydilley/github/mold
   git fetch origin --prune
   git checkout feat/multi-prompt-chain-v2-phase3
   git log --oneline origin/main..HEAD | head -50
   git status
   cargo test --workspace  # should be green; retry once on TUI flakes
   ```

2. Pre-investigation for Task 5.1: understand the web codebase.
   ```bash
   ls web/src/ web/src/components/ web/src/api* 2>/dev/null
   cat web/package.json | head -30
   grep -rn 'fetch.*api\|/api/' web/src/ | head -15
   ```

3. Start Task 5.1 (`smol-toml` dep + `chainToml.ts`). Plan at line 3915.

## If you hit a surprise

Read `tasks/multi-prompt-chain-v2-resume-phase5.md` §"Gotchas accumulated through Phases 1–4" before assuming anything. If the surprise is beyond that list, stop and ask the user.
