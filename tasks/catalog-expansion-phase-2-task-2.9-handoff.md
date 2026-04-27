# Phase-2 task 2.9 — Web download gate flip (kickoff handoff)

> Paste the prompt at the bottom of this file into a fresh Claude Code
> session to start task 2.9. Everything above is reference material.

## Where phase 2 stands on entry

Branch `feat/catalog-expansion`. Eight phase-2 commits are
**local-only** — phase 2 lands as one push at 2.10:

| Commit | Origin? | Scope |
|---|---|---|
| `9322ffa` | local | feat(cli,server,core): catalog cv:<id> recipe-driven pull (phase 2.8) |
| `1c0b4a8` | yes | docs(tasks): catalog-expansion phase-2 task 2.8 kickoff handoff |
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
truth. After 2.8, the **server + CLI + core** ends are wired end-to-end:
companion-first ordering, recipe-driven primary fetch, real
`primary_job_id` for `cv:` entries, lifted helpers shared between CLI
and server. The candle-transformers-mold accessor bump (Option A from
2.7/2.8 brainstorming) is split into 2.8.5 because the user doesn't have
crates.io publish access for `candle-transformers-mold` —
[PR #1](https://github.com/utensils/candle/pull/1) is open against
`utensils/candle` waiting on James (jamesbrink) to merge + publish.

### Done

- 2.1 pre-flight, 2.2 tensor audit, 2.3 dispatcher, 2.4 SD1.5
  constructor, 2.5 SDXL constructor (incl. CLIP-G fused QKV), 2.6 factory
  routing + `is_turbo` threading + `SingleFileBackend`, 2.7 server
  companion auto-pull (response shape becomes `{ primary_job_id,
  companion_jobs: [{name, job_id}] }`), 2.8 CLI recipe path + Option A'
  server `enqueue_recipe`. **`cv:` entries now return real
  `primary_job_id` instead of `null`** as of 2.8.

### Carried forward from 2.8 — read this carefully

**The candle-mold accessor bump is split out as 2.8.5.** PR open at
https://github.com/utensils/candle/pull/1 with two `pub fn` accessors
on `StableDiffusionConfig` (`unet()` returns
`&unet_2d::UNet2DConditionModelConfig`, `autoencoder()` returns
`&vae::AutoEncoderKLConfig`) plus a version bump to 0.9.11. Once James
merges + publishes:

1. Bump `crates/mold-inference/Cargo.toml:51` from `"0.9.10"` →
   `"0.9.11"`.
2. Wire `VarBuilder::from_backend(SingleFileBackend)` into
   `crates/mold-inference/src/sd15/pipeline.rs::load_components_single_file`
   (UNet, VAE, CLIP-L) and the parallel SDXL site (UNet, VAE, CLIP-L,
   CLIP-G with fused-QKV `FusedSlice`).
3. Drop the `bail!()` sentinel; replace with the three (SD15) / four
   (SDXL) candle constructor calls.
4. Run `cargo test -p mold-ai-inference --lib pipeline` — the existing
   tests already assert the candle-fork-sentinel; update those to
   instead assert the constructor was called.

This unblocks **end-to-end SDXL/SD1.5 single-file generation** —
without it, the <gpu-host> UAT in 2.10 can pull SDXL Pony but the
inference path bails with the candle-mold sentinel.

**Track the PR**: `gh pr view utensils/candle/1 --json state,mergedAt`.
Once merged and `cargo info candle-transformers-mold` shows 0.9.11,
land 2.8.5.

### Not yet done

- 2.8.5 (provisional) — candle-mold pin bump + `load()` real-shape
  wiring. Blocked on https://github.com/utensils/candle/pull/1.
- **2.9 — this handoff's task** (web `cat.canDownload` gate flip).
- 2.10 UAT — full <gpu-host> run (Pony / Juggernaut XL / DreamShaper 8).

## What 2.9 produces

Per the canonical phase-2 spec
(`tasks/catalog-expansion-phase-2-handoff.md` § "Task list" 2.9 row):

> Web Download button — already wired to `POST /api/catalog/:id/download`;
> it just enables once 2.7 drops the 409. Confirm `cat.canDownload(entry)`
> in `web/src/composables/useCatalog.ts` switches from `engine_phase === 1`
> to `engine_phase <= 2`.

This is the smallest task in phase 2 — **single-line predicate change**
plus updated copy + a Vue component test. The whole CLI + server +
core stack already reports correctly; the SPA just needs to flip its
gate to match.

Three concrete deliverables:

### A. `cat.canDownload` gate flip

`web/src/composables/useCatalog.ts` (or wherever `canDownload` lives —
**grep first**, the spec mentions the file but verify) currently has:

```ts
const canDownload = (entry: CatalogEntry) => entry.engine_phase === 1;
```

Change to:

```ts
const canDownload = (entry: CatalogEntry) => entry.engine_phase <= 2;
```

The server already accepts `engine_phase <= 2` after 2.7's gate drop.
2.9 just stops the SPA from short-circuiting before the request goes
out.

### B. Badge + CTA copy update

`web/src/components/CatalogCard.vue` and
`web/src/components/CatalogDetailDrawer.vue` show a "phase 2 — coming
soon" badge or similar. After 2.9, phase-2 entries are downloadable —
the badge should disappear or change to indicate "single-file"
(SD1.5/SDXL) status. Verify the current copy by grep before editing —
the actual strings may differ.

The CTA (Download button) should change from a disabled state to an
active state for `engine_phase === 2` entries. Test in browser dev mode
against the running server.

### C. Vue component tests

The SPA has Vitest unit tests under `web/src/__tests__/` or similar
(`bun run test`). Add or update tests so:
- `canDownload({ engine_phase: 1 })` → `true`
- `canDownload({ engine_phase: 2 })` → `true` (was false)
- `canDownload({ engine_phase: 3 })` → `false`

Existing card / drawer tests may also need updating — grep for
`engine_phase` usage in `web/src/`.

### TDD shape

One round, mirroring 2.4–2.8 but smaller because the surface is tiny:

#### Round 1 — gate predicate + component tests

`canDownload_phase_1_returns_true`,
`canDownload_phase_2_returns_true_after_gate_flip` (was the regression
target),
`canDownload_phase_3_or_higher_returns_false`. If the badge / CTA copy
has unit-testable rendering, assert the right text appears for each
phase.

## Out of scope for 2.9

- **Server-side gate work** — already done in 2.7.
- **CLI gate work** — already done in 2.8.
- **Killswitch UAT** — that's 2.10. 2.9 is just the SPA predicate.
- **End-to-end download flow testing in browser** — fine to do as a
  smoke test, but the systematic UAT is 2.10.
- **2.8.5 candle-mold bump** — blocked on James, separate commit.

## Working conventions to preserve

- TDD — failing tests first.
- One scope per commit. 2.9 is one commit:
  `feat(web): catalog download gate flips at engine_phase <= 2 (phase 2.9)`.
- **Phase 2 lands as one push when 2.10 is gate-green.** All phase-2
  commits stay local.
- This task is small enough that the user can do it interactively
  (no subagent dispatch needed). `superpowers:test-driven-development`
  applies.
- `superpowers:verification-before-completion` before declaring done.

## Verification commands

```bash
cd /Users/jeffreydilley/github/mold

# Pre-flight
git status                                      # clean
git log --oneline origin/main..HEAD | head -16  # phase-2 commit chain so far

# Web tests
( cd web && bun install && bun run test )
( cd web && bun run build )
( cd web && bun run fmt:check )

# Cargo gates (should still be green from 2.8)
cargo fmt --all -- --check
cargo clippy --workspace --all-targets -- -D warnings
cargo test --workspace
```

The TUI theme test flake
(`theme_save_then_load_round_trip_preserves_preset`) is documented in
user memory — retry once before blaming your changes.

## Reference reading

- `tasks/catalog-expansion-phase-2-handoff.md` — overall phase-2 brief.
- `tasks/catalog-expansion-phase-2-task-2.8-handoff.md` — 2.8 brief.
- `web/src/composables/useCatalog.ts` (or whatever the actual file is —
  grep first) — predicate site.
- `web/src/components/CatalogCard.vue`,
  `web/src/components/CatalogDetailDrawer.vue` — badge + CTA sites.
- `web/src/types.ts` (or similar) — `CatalogEntry` shape, including
  `engine_phase` field.

---

## The prompt

Paste from here into a fresh Claude Code session:

---

I'm starting **task 2.9** of the mold catalog-expansion phase 2 — the
web `cat.canDownload(entry)` gate flip from `engine_phase === 1` to
`<= 2`. Tasks 2.1–2.8 are done; the server (2.7) + CLI + core (2.8)
already accept and surface phase-2 entries correctly. The SPA is the
last gate.

## Read first, in this order

1. `~/.claude-personal/CLAUDE.md` and
   `/Users/jeffreydilley/github/mold/CLAUDE.md` — coding conventions.
2. `tasks/catalog-expansion-phase-2-handoff.md` — overall phase-2 brief.
   **§ "Task list" 2.9 row** is the canonical spec.
3. `tasks/catalog-expansion-phase-2-task-2.9-handoff.md` — this file.
4. `tasks/catalog-expansion-phase-2-task-2.8-handoff.md` — 2.8 context,
   including the candle-fork-accessor blocker that's now in 2.8.5.
5. **Grep** `web/src/` for `canDownload`, `engine_phase`, "phase 2",
   "single-file", and any catalog-related badge text. The exact files
   and predicate location may have evolved since the spec was written.

## What you're doing

Implement task 2.9 per this handoff. Three small deliverables:

1. **`cat.canDownload(entry)` gate flip** — `engine_phase === 1` →
   `<= 2`. Single-line change in the composable.
2. **Badge / CTA copy update** — phase-2 entries are no longer "coming
   soon"; update text to reflect "single-file" / downloadable status.
3. **Component tests** — Vitest unit tests asserting the new gate
   behavior across phases 1, 2, 3.

TDD: write failing test first, flip the predicate, watch it go green.

## How to work

1. Pre-flight: confirm `git status` clean and `bun run test` green
   before touching code.
2. **Pre-grep `web/src/`.** The predicate location and badge copy may
   have moved since the 2.8 commit landed.
3. Use `superpowers:test-driven-development`. Round 1 is the entire
   task — there's no second round.

## Verification gate before committing

```bash
( cd web && bun run test && bun run build && bun run fmt:check )
cargo fmt --all -- --check
cargo clippy --workspace --all-targets -- -D warnings
cargo test --workspace                  # retry once on TUI flake
```

## Commit shape

Single commit when gate-green:

```
feat(web): catalog download gate flips at engine_phase <= 2 (phase 2.9)

cat.canDownload(entry) in web/src/composables/useCatalog.ts switches
from `engine_phase === 1` to `engine_phase <= 2`. SD1.5 + SDXL
single-file Civitai entries (Pony, Juggernaut XL, DreamShaper, etc.)
become downloadable through the SPA's Download button — server-side
gate dropped in 2.7, CLI in 2.8, this is the SPA catching up.

Badge + CTA copy on CatalogCard / CatalogDetailDrawer updated to
reflect that phase-2 entries are now downloadable. "phase 2 — coming
soon" → "single-file" or removed entirely depending on the visual
design.

Vitest unit tests assert canDownload returns true for phases 1 + 2,
false for phase 3+ (engine_phase 3 = FLUX/Z-Image/LTX, still gated
behind phases 3-5).

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
```

**Do not push.** Phase 2 lands as one push at 2.10.

## If you hit a surprise

- If `canDownload` lives somewhere other than `useCatalog.ts`, grep
  the actual location and update the handoff before fixing.
- If the badge text mentions "phase X" explicitly (e.g. "phase 3
  coming"), keep "phase 3" gated — that's still correct (FLUX
  single-file is phase 3).
- If `bun run test` reveals existing tests that assume phase-2 is
  ungated, those need updating too — flip them in the same commit.

When 2.9 is gate-green, write
`tasks/catalog-expansion-phase-2-task-2.10-handoff.md` (template: this
file). 2.10 is the <gpu-host> UAT — depends on 2.8.5 being merged
first because the SDXL/SD1.5 generation path needs candle-mold 0.9.11
to actually generate from single-file checkpoints.
