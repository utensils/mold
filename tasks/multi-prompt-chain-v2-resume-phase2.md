# multi-prompt-chain v2 — session 2 resume handoff

> Paste the prompt at the bottom of this file into a fresh Claude Code session
> to continue sub-project A starting from **Task 2.7**. Everything above the
> prompt is reference material.

## Status on handoff (2026-04-22)

### Branch & PR

- Working branch: `feat/multi-prompt-chain-v2` (local)
- **Phase 1 PR:** [utensils/mold#266](https://github.com/utensils/mold/pull/266) — **open, not merged**. Title: `feat(chain): wire format + TOML I/O + capabilities endpoint (Phase 1/6)`. Remote head: `adac9b1`.
- **Tasks 2.1–2.6 are local only** on top of Phase 1 — NOT pushed yet (no mid-phase pushes per handoff discipline). The fresh session should continue adding Phase 2 commits and push once Phase 2 is gate-green, opening the Phase 2 PR stacked on the Phase 1 branch.

### Local git state

Current HEAD: `de01ed8`. Local commits ahead of `origin/feat/multi-prompt-chain-v2`:

```
de01ed8 feat(server): chain route uses StitchPlan for per-boundary stitch
90cadd1 feat(ltx2): source_image honored on Cut/Fade continuation stages
81e1216 feat(ltx2): StitchPlan assembler with per-boundary rules
f986e4f feat(ltx2): fade_boundary helper for post-stitch crossfade
0908597 feat(ltx2): orchestrator passes None carry for cut/fade stages
13a78b9 feat(ltx2): expose per-stage frames from chain orchestrator
---  adac9b1 (origin/feat/multi-prompt-chain-v2, last pushed)  ---
adac9b1 chore(chain): sync Cargo.lock after adding tracing dep to mold-core
ddd39e4 feat(chain): echo ChainScript in SseChainCompleteEvent + consume in client
cfb2b6c feat(server): /api/capabilities/chain-limits endpoint
b9adb3d feat(server): ChainLimits shape + family-cap lookup
b6a9249 feat(chain): re-export new wire types from mold-core
8825f5c test(chain): TOML round-trip + normalisation invariants
7ec6efb feat(chain): TOML reader with schema version gate
29d0e15 feat(chain): chain_toml module with script writer
fb3d095 feat(chain): add estimated_total_frames with transition-aware math
79089ae feat(chain): normalise coerces stage 0 transition + rejects reserved fields
b68cbbb feat(chain): add ChainScript canonical echo + VramEstimate slot
6445cd4 feat(chain): extend ChainStage with transition/fade_frames + reserved fields
0e17516 feat(chain): reserve LoraSpec and NamedRef wire types for sub-project B
89035c1 feat(chain): add TransitionMode enum with smooth/cut/fade variants
3df5e34 docs(chain): multi-prompt chain v2 implementation plan
9cd8860 docs(chain): multi-prompt chain v2 authoring design spec
```

CI gate at each commit: `cargo fmt --all -- --check && cargo clippy --workspace --all-targets -- -D warnings && cargo test --workspace` — all green for every commit above (there's a **pre-existing flake** on `theme_save_then_load_round_trip_preserves_preset` in `mold-ai-tui` under parallel workspace tests; retry if it's the only failure — see `project_tui_theme_test_flake.md` in user memory).

### Phase progress

| Phase | Tasks | Status |
|---|---|---|
| 1 | 1.1–1.14 | ✅ complete, PR #266 open |
| 2 | 2.1–2.9 | **in progress** — 2.1–2.6 committed locally; 2.6 is unreviewed; 2.7–2.9 pending |
| 3 | 3.1–3.7 | pending (blocked on Phase 2) |
| 4 | 4.1–4.8 | pending (blocked on Phase 2) |
| 5 | 5.1–5.10 | pending (blocked on Phase 2) |
| 6 | 6.1–6.4 | pending (blocked on Phase 3/4/5) |

### Review notes for already-committed tasks

Tasks 2.1–2.5 had combined spec+quality reviews dispatched via subagent and came back APPROVED with only Minor issues flagged. Task 2.6 (`de01ed8`, StitchPlan wired into routes_chain) is **committed but not yet reviewed** — fmt/clippy/workspace-tests all pass. Fresh session should dispatch a combined review for 2.6 before starting 2.7.

Outstanding Minor issues deferred to later tasks (not blocking):

- **Stale test name in `ltx2::chain`** (`chain_runs_all_stages_and_drops_tail_prefix_from_continuations`) — orchestrator no longer trims since 2.1; test body still correct but name is misleading. Rename opportunity in 2.8.
- **`Stitching` progress event over-reports by `(N-1) * motion_tail`** — cosmetic; progress bar length uses `ChainStart.estimated_total_frames` not this. Worth fixing alongside 2.7.
- **`StitchPlan::assemble` on Smooth when next clip shorter than `motion_tail_frames`** returns `ClipTooShortForTrim` — this case should be prevented upstream by `ChainRequest::normalise`'s `motion_tail < stage.frames` check, but worth confirming the error propagates as a typed 500/502 in 2.7.

## What's left on Phase 2 (3 tasks)

### Task 2.7 — Enrich mid-chain 502 payload

- Add `ChainFailure` type to `crates/mold-core/src/chain.rs` (fields: `error`, `failed_stage_idx`, `elapsed_stages`, `elapsed_ms`, `stage_error`) + re-export from `lib.rs`.
- `Ltx2ChainOrchestrator::run` needs to return a typed error with the stage index embedded (new enum variant on `ChainRunError` or similar), so `routes_chain.rs` can map to 502 with `ChainFailure` JSON.
- Route test: inject `FakeRenderer` with `fail_on: vec![(1, "boom".into())]`, assert 502 response body shape.
- Plan file: `docs/superpowers/plans/2026-04-21-multi-prompt-chain-v2.md` Task 2.7 (~line 2504).
- Commit scope: `feat(chain)`.

### Task 2.8 — End-to-end mixed-transition test via FakeRenderer

- Add `mixed_transitions_end_to_end` test to `crates/mold-inference/src/ltx2/chain.rs` (inside `#[cfg(test)] mod tests`).
- Uses existing `FakeRenderer` + `sample_chain_request` helpers from Task 2.1/2.2.
- Builds a 4-stage chain (Smooth / Smooth / Cut / Fade(8)), drives orchestrator, pipes `stage_frames` through `StitchPlan::assemble`, asserts `frames.len() == 355` (= `97 + 72 + 97 + 89`).
- Plan file ~line 2563. Commit scope: `test(chain)`.

### Task 2.9 — Phase 2 gate

1. `cargo fmt --all`
2. `cargo clippy --workspace --all-targets -- -D warnings`
3. `cargo test --workspace`
4. **Killswitch box end-to-end renders** (requires SSH access to `killswitch@192.168.1.67`, mold repo at `~/github/mold`, sm_86 CUDA build):
   - 3-stage all-smooth chain
   - 3-stage all-cut chain
   - 3-stage all-fade chain
   - mixed (smooth/cut/fade)
   - Visual inspection of output MP4s for seam quality
5. Open Phase 2 PR titled `feat(ltx2): engine transitions (cut/fade/stitch) (Phase 2/6)`.

**If the killswitch box is unreachable** from the session, the fresh session should leave Phase 2 at the unit-test/integration-test level (green there) and flag the user to do the manual smoke before merging Phase 2.

## What's left after Phase 2

**Phase 3** (CLI surface, 7 tasks) — `--script shot.toml`, `--dry-run`, repeated `--prompt` sugar, `mold chain validate` subcommand, progress display with transition tags. Plan section starts at line 2674.

**Phase 4** (TUI script mode, 8 tasks) — `ScriptComposer` ratatui widget, stage list nav `j/k/J/K`, add/delete `a/A/d`, transition cycle `t`, prompt/frames editors `i/f`, TOML save/load `Ctrl-S/Ctrl-O`, submit `Enter`, `tui-uat.sh` scenario. Plan section starts at line 3295.

**Phase 5** (Web composer script mode, 10 tasks) — `smol-toml` (or `@iarna/toml` fallback) dep, `chainToml.ts`, `fetchChainLimits`, `StageCard.vue`, `ScriptComposer.vue`, mode toggle on `Composer.vue`, chain submit, per-stage expand modal, drag-reorder via `vue-draggable-plus`, footer chain-limits clamp, `bun run verify && bun run build` gate. Plan section starts at line 3907.

**Phase 6** (Docs + release, 4 tasks) — `website/guide/video.md`, `CHANGELOG.md [Unreleased]`, `.claude/skills/mold/SKILL.md`, `CLAUDE.md`. Each a separate `docs(chain)` commit. Plan section starts at line 4603.

Dependency graph: `Phase 2 → {3, 4, 5}` (concurrent) `→ 6`. After Phase 2 lands, the fresh session can dispatch Phases 3/4/5 in parallel using separate subagent-driven flows — though sequential is also fine and easier to track.

## Working conventions to preserve

- **One scope per commit** — `feat(chain) / feat(ltx2) / feat(server) / feat(cli) / feat(tui) / feat(web) / test(chain) / docs(chain)`.
- **No mid-phase pushes** — each phase is one PR when complete.
- **TDD** — failing test first, then implementation, then verify pass, then commit. Deviations (e.g. Task 2.3 skipped the explicit fail verify) should be rare and only for pure-math helpers where Step 2 adds no real signal.
- **`superpowers:subagent-driven-development`** — fresh general-purpose subagent per task. Sonnet for mechanical tasks, opus only if the task requires architectural judgment. Combined spec+quality review works well for plan-verbatim tasks; split into two-stage for complex/cross-crate work.
- **Prompt the subagent with full task text** — don't send a "read the plan and implement 2.7" instruction. Paste the plan's text inline and flag pre-investigation findings (manifest API differences, existing construction sites, etc.) so the subagent doesn't re-discover them.
- **Pre-grep before dispatching** — when a task changes a public struct (like `ChainStage` did), `grep -rn 'StructName {' crates/` across the workspace and list every construction site in the brief. CI's `cargo test --workspace` contract is what forces cross-crate scope expansion; the plan's file-list is often incomplete for that reason.

## Gotchas the new session should know

- **`MoldError::Other`** takes `anyhow::Error`, not `String` — use `MoldError::Other(anyhow::anyhow!(...))`.
- **`mold_core::manifest`** has `find_manifest(&name).map(|m| m.family.clone())` for family lookup; there's no `resolve_family` or `resolve_quant` despite what the plan hints at. Quant = `name.split_once(':').map(|(_, t)| t.to_string()).unwrap_or_default()`.
- **`toml-rs` sorts keys alphabetically** within tables — force the `schema = "..."` header to the top manually in `write_script` (Task 1.7 already handled this with a replace-and-prepend fallback).
- **`#[serde(default)]` on required struct fields** needs a `Default` impl. `ChainScript` + `ChainScriptChain` derive `Default` as of Task 1.13; `OutputFormat::Png` is the `#[default]` variant so `ChainScriptChain::default().output_format == Png` — semantically weird for a video context but only used for serde fallback deserialization. See commit `ddd39e4` for the "placeholder with odd default" note.
- **`ChainSseMessage::Complete(Box<SseChainCompleteEvent>)`** — Task 1.13 boxed this after adding `script`/`vram_estimate` to silence `large_enum_variant`. If you add more fields to `SseChainCompleteEvent`, keep the box; don't un-box.
- **`placeholder_for_sse_transition`** helper was deleted in Task 1.13. Don't resurrect it.
- **TUI theme test flake** — see user auto-memory `project_tui_theme_test_flake.md`. If only `theme_save_then_load_round_trip_preserves_preset` fails under `cargo test --workspace`, retry once. Not your regression.

## Verification commands the new session will need

```bash
# Pre-flight: confirm branch state
cd /Users/jeffreydilley/github/mold
git log --oneline origin/main..HEAD | head -25
# Should show 16 commits (Phase 1 + 2.1–2.6 + docs).

# Sanity check CI before writing any new code
cargo fmt --all -- --check
cargo clippy --workspace --all-targets -- -D warnings
cargo test --workspace  # retry once if only TUI flake fires

# Once Phase 2 is complete, push + open PR
git push
gh pr create --title "feat(ltx2): engine transitions (cut/fade/stitch) (Phase 2/6)" \
  --body "..."  # stack on Phase 1 branch
```

## Spec & plan references

- Spec: `docs/superpowers/specs/2026-04-21-multi-prompt-chain-v2-design.md` (681 lines). §3 decisions are NOT up for debate.
- Plan: `docs/superpowers/plans/2026-04-21-multi-prompt-chain-v2.md` (4805 lines, 51 tasks).
- Prior handoff: `tasks/multi-prompt-chain-v2-handoff.md` (the session-1 kickoff prompt).
- Prior-art handoff for style: `tasks/render-chain-v1-handoff.md` (the v1 chain plan executed with the same conventions).

---

## The prompt

Paste from here into a fresh Claude Code session:

---

I'm resuming execution of **multi-prompt chain v2, sub-project A** for the mold repo, starting at **Task 2.7** (mid-Phase 2). This is session 2 of a multi-session project.

## Read first, in this order

1. `~/.claude-personal/CLAUDE.md` (global) and `/Users/jeffreydilley/github/mold/CLAUDE.md` (project) — coding conventions.
2. `tasks/multi-prompt-chain-v2-resume-phase2.md` — **your primary briefing**. Read end-to-end. Status, gotchas, completion map, working conventions.
3. `docs/superpowers/specs/2026-04-21-multi-prompt-chain-v2-design.md` — approved design. §3 decisions not up for debate.
4. `docs/superpowers/plans/2026-04-21-multi-prompt-chain-v2.md` — 51-task plan. You'll be executing 2.7 onward.
5. `tasks/multi-prompt-chain-v2-handoff.md` — session-1 kickoff prompt (for context on how Phase 1 was run).

## Status on entry

- Branch: `feat/multi-prompt-chain-v2`, HEAD `de01ed8`. Phase 1 PR open at [utensils/mold#266](https://github.com/utensils/mold/pull/266). Tasks 2.1–2.6 local only (no mid-phase pushes).
- Task 2.6 (`de01ed8`, StitchPlan wired into routes_chain) is committed, tests pass, but **not yet dispatched through a spec+quality review**. Your first action should be to dispatch that review before starting 2.7.
- Design is approved. Plan is approved. No more brainstorming.

## What you're doing

Execute Tasks **2.7 → 2.8 → 2.9** (Phase 2 finish) via `superpowers:subagent-driven-development`. Then Phase 2 PR, then Phases 3/4/5 (concurrent ok) + 6. Use `superpowers:verification-before-completion` before declaring any phase done. TDD discipline per task.

Gate every phase with `cargo fmt --check && cargo clippy --workspace --all-targets -- -D warnings && cargo test --workspace` (plus `bun run verify` / `bun run build` for Phase 5). **No mid-phase pushes.** One PR per phase.

## How to work

- **Primary skill:** `superpowers:subagent-driven-development`. The plan is sized for it. One fresh subagent per task.
- **Verification skill:** `superpowers:verification-before-completion` before claiming any phase done.
- **Review discipline:** combined spec+quality review is fine for plan-verbatim mechanical tasks; split into two-stage for complex/cross-crate work. Session 1 used sonnet for implementers and reviewers — that worked well.
- **Pre-grep before dispatching** — when a task changes a public type, `grep -rn 'TypeName {' crates/` across the workspace and enumerate construction sites in the brief. CI `cargo test --workspace` will break if you miss any.
- **Paste full task text** to each subagent — don't make them read the plan file.

## Start here

1. Confirm branch state:
   ```bash
   cd /Users/jeffreydilley/github/mold
   git log --oneline origin/main..HEAD | head -20
   git status  # should be clean
   cargo test --workspace  # green; retry once on TUI flake if it fires
   ```
   You should see `de01ed8` at the top of the log.

2. Dispatch the **deferred spec+quality review for Task 2.6** (`de01ed8`). The implementer claimed DONE with all gates green but no reviewer has seen it. Acceptance criteria and context are in `tasks/multi-prompt-chain-v2-resume-phase2.md` under "Review notes for already-committed tasks".

3. Start Task 2.7 (`ChainFailure` typed error). Pre-investigation to do:
   - `grep -n 'ChainRunError' crates/mold-server/src/routes_chain.rs` — current error enum.
   - `grep -n 'bail!' crates/mold-inference/src/ltx2/chain.rs | head` — current orchestrator error surface (stringly-typed via `anyhow::Context`).
   - Decide: do you add a `StageFailed { stage_idx, elapsed_stages, elapsed_ms, inner }` variant on the orchestrator's error type, or propagate via a new `OrchestratorError` wrapper? Plan §2.7 sketches the former.

4. Task 2.8 (`mixed_transitions_end_to_end` test) is a straightforward test addition — use the existing `FakeRenderer` and `sample_chain_request` helpers already added in Tasks 2.1/2.2.

5. Task 2.9 (Phase 2 gate) — fmt + clippy + test + manual killswitch smoke. If killswitch is unreachable, green the unit/integration layer and flag to the user before opening the Phase 2 PR.

6. Open the Phase 2 PR: title `feat(ltx2): engine transitions (cut/fade/stitch) (Phase 2/6)`. It stacks on the Phase 1 branch (target: `main`); link Phase 1 PR #266 in the body.

## If you hit a surprise

If a type signature doesn't match, a test harness is harder than the plan assumed, a clippy lint the plan's code would trigger, a model-family hardcode that conflicts with live manifest state — **stop, capture the surprise, ask the user before pressing forward.** The handoff `tasks/multi-prompt-chain-v2-handoff.md` (session 1) documents the expected gotchas; anything beyond that list is a real surprise.
