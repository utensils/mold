# Phase-2 task 2.10 — Killswitch UAT + phase-2 push (kickoff handoff)

> Paste the prompt at the bottom of this file into a fresh Claude Code
> session to start task 2.10. Everything above is reference material.

## Where phase 2 stands on entry

Branch `feat/catalog-expansion`. **Ten** phase-2 commits are
**local-only** — phase 2 lands as one push at the *end* of 2.10:

| Commit | Origin? | Scope |
|---|---|---|
| `3d0aa59` | local | feat(web): catalog download gate flips at engine_phase <= 2 (phase 2.9) |
| `5f7d0af` | local | feat(inference): wire SingleFileBackend into SD15/SDXL constructors (phase 2.8.5) |
| `fbd87cc` | yes | docs(tasks): catalog-expansion phase-2 task 2.9 kickoff handoff |
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
truth — § "Task list" 2.10 row is the canonical spec. The full stack is
now wired end-to-end:

- **inference** (2.3–2.6) — single-file dispatcher + SD15/SDXL
  constructors + factory routing + `SingleFileBackend`.
- **server** (2.7) — companion-first auto-pull, `engine_phase <= 2`
  accepted, response shape `{ primary_job_id, companion_jobs: [{name,
  job_id}] }`.
- **CLI** (2.8) — recipe-driven `mold pull cv:<id>` with companion
  ordering, real `primary_job_id` returned.
- **inference end-to-end** (2.8.5) — `candle-transformers-mold` 0.9.12
  bumped, `VarBuilder::from_backend(SingleFileBackend)` wired into both
  SD15 and SDXL `load_components_single_file`. SDXL/SD1.5 single-file
  generation actually runs end-to-end.
- **web** (2.9) — `cat.canDownload` flips to `<= 2`; phase-2 entries
  are downloadable through the SPA Download button; "Coming in phase N"
  badges only render for phase 3+.

### Done

2.1 pre-flight, 2.2 tensor audit, 2.3 dispatcher, 2.4 SD1.5 constructor,
2.5 SDXL constructor (CLIP-G fused QKV), 2.6 factory routing +
`is_turbo` threading + `SingleFileBackend`, 2.7 server companion
auto-pull, 2.8 CLI recipe path + `enqueue_recipe`, **2.8.5 candle-mold
0.9.12 + `from_backend` wired into both SD15 + SDXL constructors**, 2.9
SPA gate flip + badge-gate bump.

### Not yet done

- **2.10 — this handoff's task** — the <gpu-host> UAT + CHANGELOG
  entry + single phase-2 push.

### Important context that changed since the 2.9 handoff was written

The 2.9 brief said *"2.10 depends on 2.8.5 being merged first because the
SDXL/SD1.5 generation path needs candle-mold 0.9.11 to actually
generate from single-file checkpoints."*

**That blocker is resolved.** James merged
[utensils/candle PR #1](https://github.com/utensils/candle/pull/1) and
the user shipped 2.8.5 in commit `5f7d0af`. The actually-published
version is **`candle-transformers-mold` 0.9.12** (not 0.9.11):

> Release engineering note from 2.8.5's commit body: 0.9.11 was published
> from candle-mold's main branch and shipped behavior changes mold can't
> accept (`mode()` removed, `transformer.forward()` arity change). 0.9.11
> is yanked. 0.9.12 was published from a fresh `release-0.9.12` branch
> off the v0.9.10 tag with only the accessor commit cherry-picked on
> top — narrow accessor-only delta on the v0.9.10 API surface.

If `cargo info candle-transformers-mold` ever shows 0.9.11 as the
"recommended" version, ignore it — the workspace is intentionally
pinned to 0.9.12 in `crates/mold-inference/Cargo.toml:51`.

## What 2.10 produces

Per the canonical phase-2 spec
(`tasks/catalog-expansion-phase-2-handoff.md` § "Task list" 2.10 row):

> Phase 2 gate: `cargo fmt --all -- --check && cargo clippy --workspace
> --all-targets -- -D warnings && cargo test --workspace`; `bun run
> test && bun run build && bun run fmt:check`; **<gpu-host> UAT** —
> pull + generate one Pony entry, one Juggernaut XL entry, one SD1.5
> entry (e.g. epiCRealism). Visual inspection of outputs. CHANGELOG
> entry.

This task is **non-coding** in the normal sense. It is:

1. Confirm local gates are still green (5 minutes).
2. CHANGELOG entry under `[Unreleased]` (one commit).
3. SSH to <gpu-host> (`<gpu-host>`), build with CUDA
   `<arch-tag>`, restart `mold-server`, run the three-checkpoint UAT, eyeball
   the renders.
4. **Push the entire phase-2 commit chain as one push** and open the PR
   `feat(catalog): SD1.5 + SDXL single-file loaders (phase 2/5)`.

Three concrete deliverables:

### A. Local re-verify gate (no commits)

Quick re-run before touching <gpu-host> — phase-2 has been local for
2 weeks of development; one final check that nothing rotted:

```bash
cd /Users/jeffreydilley/github/mold
git status                                       # clean
cargo fmt --all -- --check
cargo clippy --workspace --all-targets -- -D warnings
cargo test --workspace                           # retry once on TUI flake
( cd web && bun run test && bun run build && bun run fmt:check )
```

If any gate fails, fix in a one-scope commit before proceeding. Don't
batch fixes with the CHANGELOG commit.

### B. CHANGELOG entry (one commit)

Add a `[Unreleased]` block to `CHANGELOG.md` summarising phase 2.
Group under Added / Changed / Fixed per Keep-a-Changelog convention.
Suggested shape (refine to match the project's prior entries):

```markdown
### Added

- **SD1.5 + SDXL single-file Civitai checkpoints** — `mold pull cv:<id>`
  and the web Download button now resolve Civitai single-file
  checkpoints (Pony, Juggernaut XL, DreamShaper, epiCRealism, etc.)
  through the catalog's `download_recipe`, auto-pulling required
  companions (`clip-l`, `clip-g`, `sdxl-vae`, `sd-vae-ft-mse`) before
  the primary file. End-to-end generation works on both backends.
- `crates/mold-inference/src/loader/single_file.rs` — single-file
  dispatcher + per-family SD15/SDXL constructors with custom
  `SingleFileBackend` for `VarBuilder::from_backend` integration.
- `sd_singlefile_inspect` dev-bin (`--features dev-bins`) for
  ad-hoc tensor-key audits of single-file checkpoints.

### Changed

- Catalog download gate (`POST /api/catalog/:id/download`) accepts
  `engine_phase <= 2` (was `>= 2 → 409`); CLI mirrors the gate; SPA
  `cat.canDownload` flips from `=== 1` to `<= 2`.
- Catalog download response shape now includes companion job ids:
  `{ primary_job_id, companion_jobs: [{name, job_id}] }`.
- `candle-transformers-mold` bumped to 0.9.12 with two new public
  accessors on `StableDiffusionConfig` (`unet()`, `autoencoder()`)
  required by the single-file path.
```

Commit:

```
docs(changelog): phase 2 — SD1.5 + SDXL single-file Civitai loaders
```

### C. Killswitch UAT (no code, but rigorous)

This is the actual verification gate. Do not declare phase 2 done
without three successful generations.

```bash
# From your laptop, push the phase-2 commit chain
git push origin feat/catalog-expansion

# On <gpu-host>
ssh <gpu-host>
cd ~/github/mold && git pull
cargo build --release -p mold-ai \
  --features cuda,preview,discord,expand,tui,webp,mp4,metrics
systemctl --user restart mold-server
journalctl --user -u mold-server -f &  # tail logs in another tmux pane

# Pick exemplar entries from the live catalog (the 2.1 task left this
# until UAT time). One of each:
mold catalog list --family sdxl --json | jq '.[] | select(.name | test("Pony"; "i")) | {id, name}' | head -3
mold catalog list --family sdxl --json | jq '.[] | select(.name | test("Juggernaut"; "i")) | {id, name}' | head -3
mold catalog list --family sd15 --json | jq '.[] | select(.name | test("epiCRealism|DreamShaper|Realistic Vision"; "i")) | {id, name}' | head -3

# UAT — three pulls + three generations
mold pull cv:<sdxl-pony-id>          # expect: clip-l + clip-g + sdxl-vae companions pulled first, then unet
mold pull cv:<sdxl-juggernaut-id>    # expect: companions cached, only unet downloads
mold pull cv:<sd15-epicrealism-id>   # expect: clip-l + sd-vae-ft-mse companions
mold run cv:<sdxl-pony-id> "a cottagecore landscape, anime style"
mold run cv:<sdxl-juggernaut-id> "a portrait of an astronaut on mars, photorealistic, golden hour"
mold run cv:<sd15-epicrealism-id> "a cinematic shot of a cat in a library"
ls -lh ~/.mold/output/ | tail -10    # sanity: three new images, sane sizes (>500KB)
```

For each generation, **eyeball the output**:
- Does it look like the model? (Pony = anime style, Juggernaut = photoreal,
  epiCRealism = SD1.5-tier photorealism)
- No NaN / black-image / rainbow-noise / scrambled-tile failure modes?
- Reasonable inference time? (SDXL ~15-30s on a 3090; SD1.5 ~5-10s)

Then test the same trio through the web UI as a smoke check:
- Browse to `/catalog`, filter `family=sdxl`, click a Pony entry, click
  Download → watch the DownloadsDrawer (top-bar button) show companions
  in the right order.
- Browse back to the model picker / generate page, pick the new model,
  generate. Confirm the output renders.

**If any step fails**, do NOT push or open the PR. Triage in this order:

1. Companion not found → check `crates/mold-catalog/src/companions.rs`
   registry has the expected HF repo path.
2. Tensor key missing → re-run `sd_singlefile_inspect` against the actual
   checkpoint; the audit findings in `tasks/catalog-expansion-phase-2-tensor-audit.md`
   may need updating for an unusual variant.
3. NaN / black image → likely a candle-mold integration bug. Try the
   same prompt with `--seed 42` for reproducibility and bisect.
4. Inference hang / OOM → SDXL UNets fit in 24GB without offload, but
   if VRAM is fragmented from a prior run, restart `mold-server` and
   retry.

### D. Open the phase-2 PR

After the UAT passes:

```bash
# From <gpu-host> or laptop, doesn't matter — branch is already pushed
gh pr create --title "feat(catalog): SD1.5 + SDXL single-file loaders (phase 2/5)" \
  --body "$(cat <<'EOF'
## Summary

Phase 2 of the catalog expansion: SD1.5 + SDXL single-file Civitai
checkpoints (Pony, Juggernaut XL, DreamShaper, epiCRealism, etc.) are
now downloadable through `mold pull cv:<id>` and the web Download
button, with full end-to-end generation working on candle.

- **Inference**: single-file loader dispatcher, SD15 + SDXL
  constructors, factory routing, custom `SingleFileBackend` for
  `VarBuilder::from_backend`. `candle-transformers-mold` bumped to
  0.9.12 (two new public accessors on `StableDiffusionConfig`).
- **Server**: `POST /api/catalog/:id/download` accepts `engine_phase <=
  2`, auto-pulls declared companions (`clip-l`, `clip-g`, `sdxl-vae`,
  `sd-vae-ft-mse`) before the primary file, returns
  `{ primary_job_id, companion_jobs: [{name, job_id}] }`.
- **CLI**: `mold pull cv:<id>` dispatches through the recipe path
  for single-file phase-2 entries.
- **Web**: `cat.canDownload` predicate + `CatalogCard` /
  `CatalogDetailDrawer` badge gates flip from `>= 2` to `>= 3`. SPA
  Download button enables for SD1.5/SDXL Civitai entries.

## Test plan

- [x] `cargo fmt --all -- --check && cargo clippy --workspace
      --all-targets -- -D warnings && cargo test --workspace`
- [x] `( cd web && bun run test && bun run build && bun run fmt:check )`
- [x] Killswitch UAT: pull + generate Pony / Juggernaut XL / SD1.5
      checkpoint; visual inspection clean.
- [x] Web UI smoke: `/catalog` Download button enabled for phase-2
      entries; companion ordering in DownloadsDrawer correct.

🤖 Generated with [Claude Code](https://claude.com/claude-code)
EOF
)"
```

The PR title follows the phase-1 PR shape ("phase N/5"). Body is a
condensed version of the CHANGELOG entry plus the UAT checklist.

## Out of scope for 2.10

- **FLUX / Z-Image / LTX single-file** — phases 3, 4, 5. Resist.
- **LoRA support** — sub-project D, not part of phase 2.
- **Refactoring the catalog scanner** — phase 1's scanner is shipped.
- **New badge copy in the SPA beyond what 2.9 already did** — the badge
  text already correctly says "Coming in phase {N}" and only renders
  for phase 3+. No further design work needed.
- **Squashing or rewriting the phase-2 commit chain** — push as-is.
  Each commit was scoped intentionally; the reviewer can `git log` per
  task. Don't `git rebase -i` to collapse them.

## Working conventions to preserve

- **No mid-phase pushes.** This is the *only* push for phase 2.
- **CHANGELOG entry is one scoped commit**, separate from the UAT.
  The UAT itself produces no commits — it produces evidence.
- `superpowers:verification-before-completion` applies: don't claim
  phase 2 done without UAT artifacts (image files, log lines from
  `journalctl --user -u mold-server`, the gh pr URL).
- This task is small enough that it can be done interactively; **do
  not dispatch a subagent for the <gpu-host> UAT** — it's
  hardware-dependent and needs the human in the loop for visual
  inspection of generated images.

## Verification commands

### Local (laptop)

```bash
cd /Users/jeffreydilley/github/mold

git status                                       # clean
git log --oneline origin/main..HEAD | head -20   # full phase-2 chain

cargo fmt --all -- --check
cargo clippy --workspace --all-targets -- -D warnings
cargo test --workspace                           # retry once on TUI flake

( cd web && bun run test && bun run build && bun run fmt:check )
```

### Killswitch (after push)

```bash
ssh <gpu-host>

cd ~/github/mold && git pull
cargo build --release -p mold-ai \
  --features cuda,preview,discord,expand,tui,webp,mp4,metrics

systemctl --user restart mold-server
systemctl --user status mold-server              # active (running)
journalctl --user -u mold-server -n 50           # no panics on startup

# Three pulls + three generations (see § C above for picking IDs)
```

The TUI theme test flake
(`theme_save_then_load_round_trip_preserves_preset`) is documented in
user memory — retry once before blaming your changes.

## Reference reading

- `tasks/catalog-expansion-phase-2-handoff.md` — overall phase-2 brief,
  § "Task list" 2.10 row (canonical spec), § "Killswitch deploy /
  verify recipe" (the UAT recipe this handoff inlines).
- `tasks/catalog-expansion-phase-2-task-2.9-handoff.md` — 2.9 brief
  (SPA gate flip), describes what 2.10 inherits.
- `tasks/catalog-expansion-phase-2-tensor-audit.md` — 2.2 audit
  findings; keep handy if a UAT pull surfaces an unexpected tensor key.
- `CHANGELOG.md` — find the prior phase boundaries to match style.
- `crates/mold-catalog/src/companions.rs` — canonical companion
  registry; reference if a UAT companion-pull misses.
- `contrib/mold-server.user.service` + `contrib/README.md` — <gpu-host>
  systemd unit + token setup.

## Important machine details

- **<gpu-host>** — GPU host, dual GPUs, repo
  at `~/github/mold`. Build with `--features
  cuda,preview,discord,expand,tui,webp,mp4,metrics`.
- The HF + Civitai tokens live in `~/.config/mold/server.env` on
  <gpu-host>; the systemd unit reads them. Tokens are *not* on the
  laptop — pulls run on <gpu-host>.
- `mold-server` listens on port 7680 by default. The web UI is the
  embedded SPA; if you want to test from your laptop, set
  `MOLD_HOST=http://<gpu-host>:7680`.

---

## The prompt

Paste from here into a fresh Claude Code session:

---

I'm starting **task 2.10** of the mold catalog-expansion phase 2 — the
final phase-2 task. Tasks 2.1–2.9 + 2.8.5 are done; the entire stack
(inference, server, CLI, web) is wired and locally green for SD1.5 +
SDXL single-file Civitai checkpoints. 2.10 is the <gpu-host> UAT plus
the single phase-2 push + PR.

This task is **non-coding** in the normal sense — there is no TDD
round. The work is:

1. Confirm local gates are still green.
2. Add a CHANGELOG entry under `[Unreleased]` (one commit).
3. SSH to <gpu-host>, build with CUDA `<arch-tag>`, restart `mold-server`,
   run the three-checkpoint UAT (Pony / Juggernaut XL / SD1.5), eyeball
   the renders.
4. Push the entire phase-2 commit chain (10 commits) as one push and
   open the PR `feat(catalog): SD1.5 + SDXL single-file loaders (phase
   2/5)`.

## Read first, in this order

1. `~/.claude-personal/CLAUDE.md` and
   `/Users/jeffreydilley/github/mold/CLAUDE.md` — coding conventions.
2. `tasks/catalog-expansion-phase-2-handoff.md` — overall phase-2 brief.
   **§ "Task list" 2.10 row** is the canonical spec. § "Killswitch
   deploy / verify recipe" is the UAT recipe.
3. `tasks/catalog-expansion-phase-2-task-2.10-handoff.md` — this file.
4. `tasks/catalog-expansion-phase-2-task-2.9-handoff.md` — 2.9 context;
   note the 2.8.5 candle-mold blocker has resolved (0.9.12 published +
   wired in commit `5f7d0af`).
5. **Check** `git log --oneline origin/main..HEAD | head -20` — confirm
   ten phase-2 commits are local-only and the chain is intact.

## What you're doing

Implement task 2.10 per this handoff. Four deliverables:

1. **Local re-verify gate** — `cargo fmt`, `cargo clippy`, `cargo
   test --workspace`, `bun run test && bun run build && bun run
   fmt:check`. Fix any rot in scoped commits before proceeding.
2. **CHANGELOG entry** — one commit, `[Unreleased]` block summarising
   phase 2 (Added / Changed). Suggested shape in this handoff § B.
3. **Killswitch UAT** — SSH to `<gpu-host>`, build with
   CUDA `<arch-tag>`, restart `mold-server`, pull + generate one Pony, one
   Juggernaut XL, one SD1.5 (e.g. epiCRealism). Visual inspection of
   each output. The recipe is in this handoff § C — the user picks
   the exemplar IDs at UAT time from the live catalog.
4. **Push + open PR** — single push of the phase-2 chain, `gh pr
   create` with the title `feat(catalog): SD1.5 + SDXL single-file
   loaders (phase 2/5)`. Body shape in § D.

## How to work

1. Pre-flight: `git status` clean, full local gate green.
2. Write the CHANGELOG entry, commit, **do not push yet**.
3. Walk the user through SSHing to <gpu-host> and running the UAT —
   this part needs the human in the loop because visual inspection of
   generated images cannot be automated.
4. After the UAT passes, push the chain and open the PR.

## If you hit a surprise

- **Local gate red** — fix in a one-scope commit before proceeding.
  Common rot points: candle-mold ABI drift if anything in the workspace
  pulled a different transitive version; web build can rot from a Vue
  / Vite minor bump.
- **Killswitch build fails** — check `nvcc --version` shows CUDA 12.x
  and `~/.config/mold/server.env` has both `HF_TOKEN` and
  `CIVITAI_TOKEN`.
- **Companion pull fails** — re-check
  `crates/mold-catalog/src/companions.rs` registry has the right HF
  repo path. Common miss: the companion file name in the registry
  doesn't match what's actually on the HF repo.
- **Generation fails (NaN, black, scrambled)** — bisect against
  commit `5f7d0af` (the 2.8.5 candle-mold wire-up). If it's the
  `from_backend` integration, the `SingleFileBackend` impl in
  `crates/mold-inference/src/loader/single_file.rs` is the suspect.
- **PR title / body needs adjustment** — the suggested shape in § D is a
  starting point; match the prior phase-1 PR shape for consistency.

## Verification gate before opening PR

The <gpu-host> UAT IS the verification gate. Do not open the PR
without:

- Three image files in `~/.mold/output/` on <gpu-host> (one per UAT
  generation), each visually inspected and not a failure mode.
- Web UI smoke check: SPA Download button works for one phase-2
  entry, DownloadsDrawer shows companion ordering correctly.
- Local gate (cargo + bun) all green at HEAD.

## When you're done

Phase 2 is complete. The PR is open. The CHANGELOG documents what
shipped. Tag the user — they may want to manually verify the PR before
merging, or move straight to phase 3 (FLUX single-file).

Phase 3 is more involved than phase 2 because FLUX single-file
checkpoints carry only the transformer (T5 is too big to bundle), so
they need a separate companion-first flow that pulls the full T5 + CLIP
encoder set. The companion registry already has the entries; phase 3
will write the FLUX-specific tensor-key audit + loader on top of the
phase-2 dispatcher.
