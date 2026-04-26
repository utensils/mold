# catalog-expansion phase 2 — kickoff handoff (SD1.5 + SDXL single-file loaders)

> Paste the prompt at the bottom of this file into a fresh Claude Code session
> to start phase 2. Everything above the prompt is reference material for the
> author / for skimming.

## What ships in phase 2

Single-file safetensors loaders for **SD1.5** and **SDXL**. After this phase
the catalog's "Download" button is no longer disabled on `engine_phase: 2`
entries — Pony, Illustrious, NoobAI, Juggernaut XL, epiCRealism, the
~95 % of Civitai inventory that's tensor-compatible with mold's existing
SD1.5 / SDXL engines, all become real generations.

Spec: `docs/superpowers/specs/2026-04-25-catalog-expansion-design.md` §2.

## Status on handoff (2026-04-25)

### Branch & PR

- Working branch: `feat/catalog-expansion` (the umbrella branch). Phase 1
  is shipped on this branch — see `git log origin/main..HEAD` for the
  6-commit sequence (`feat(catalog): cross-client refresh state` / `feat(web):
  unified TopBar` / live-progress refresh panel + systemd handoff /
  scanner throttle + 429 retry / etc.).
- Phase 2 work continues on the same branch. **No mid-phase pushes.**
  When phase 2 is gate-green, push as one push and open PR
  `feat(catalog): SD1.5 + SDXL single-file loaders (phase 2/5)`.

### Catalog state on killswitch

- killswitch (Arch box, dual RTX 3090, sm_86) runs `mold serve` under
  `systemctl --user mold-server`. Tokens (HF + Civitai) come from
  `~/.config/mold/server.env`. Logs: `journalctl --user -u mold-server`.
  See `contrib/mold-server.user.service` for the unit + `contrib/README.md`.
- The catalog DB on killswitch already contains Civitai entries for
  `engine_phase: 2` (SD1.5 + SDXL). They render in the `/catalog` UI with
  a "phase 2" badge and a disabled Download button. The 409 path is in
  `crates/mold-server/src/catalog_api.rs::post_catalog_download` —
  phase 2's first task is dropping that gate from `>= 2` to `>= 3`.
- **Run a fresh `mold catalog refresh` on killswitch before starting** if
  the catalog hasn't been re-scanned with the throttled binary. Empty
  families = nothing to test against.

### What's already in place that phase 2 builds on

- `crates/mold-catalog/src/companions.rs` — canonical companion registry
  (`clip-l`, `clip-g`, `sd-vae-ft-mse`, `sdxl-vae`, etc.) with HF repo
  paths. Phase 2 consumes this to enqueue companion downloads before the
  entry itself.
- `CatalogEntry.companions: Vec<CompanionRef>` already populated by the
  scanner's normaliser. Each Civitai SDXL entry already declares
  `["clip-l", "clip-g", "sdxl-vae"]` (verify in the live DB).
- `CatalogEntry.download_recipe: DownloadRecipe { files, needs_token }` —
  the URL + dest template phase 2 will actually download from.
- `crates/mold-server/src/downloads.rs::DownloadQueue` — existing
  single-writer download queue for HF manifest models. Phase 2 needs
  either an extension or a parallel "recipe-driven" enqueue path.
- `crates/mold-cli/src/commands/pull.rs` already accepts `cv:` and `hf:`
  catalog ids; the catalog lookup runs but the download itself just
  re-routes to the manifest path. Phase 2 makes the recipe path real.

### What phase 2 does NOT do

- **No FLUX single-file** — that's phase 3. FLUX checkpoints on Civitai
  carry the transformer only (T5 is too big to bundle), so they need a
  separate companion-first flow that's bigger than SD1.5/SDXL.
- **No Z-Image or LTX single-file** — phases 4 + 5.
- **No LoRA scanning or LoRA browsing** — that's sub-project D, not in
  scope for phase 2 (or any of A's phases).
- **No new catalog DB schema** — `kind`, `bundling`, `engine_phase`,
  `companions`, `download_recipe` are all already in v7.

## Task list (proposed)

| # | Task | Touches |
|---|---|---|
| 2.1 | Pre-flight: confirm killswitch DB has SD15/SDXL Civitai entries; fresh `mold catalog refresh` if not. Document exemplar entries (1 SD15, 1 SDXL, 1 Pony) by id for use as test targets. | (no code) |
| 2.2 | Tensor-prefix audit (TDD pre-step): write a one-shot `dev-bins` binary that opens an SD1.5 fixture safetensors and prints unique tensor-key prefixes. Repeat for SDXL. Use real Civitai checkpoints (small ones) downloaded into `tests/fixtures/`. | new `crates/mold-inference/src/bin/sd_singlefile_inspect.rs` (gated on `dev-bins` feature, like the existing `ltx2_review` helper) |
| 2.3 | `loader/single_file.rs` dispatcher (TDD) — small enum + `load(path, family)` that returns a `SingleFileBundle { unet, vae, clip_l, clip_g, t5 }` of `SafeTensors` slices. SD1.5/SDXL paths return `Some(...)`; FLUX/Z-Image/LTX return `Err(LoadError::Unsupported(family))` for now. | new `crates/mold-inference/src/loader/mod.rs`, `crates/mold-inference/src/loader/single_file.rs` |
| 2.4 | `sd15::single_file::load(SafeTensors)` (TDD) — extract `model.diffusion_model.*` → mold's UNET key layout, `first_stage_model.*` → VAE (optional), `cond_stage_model.transformer.text_model.*` → CLIP-L. Return owned `Vec<u8>`-backed `SafeTensors` per component or candle `Tensor` map — pick whichever round-trips into the existing SD15 engine path with the smallest delta. | new `crates/mold-inference/src/sd15/single_file.rs`; modify `crates/mold-inference/src/sd15/engine.rs` to add `Sd15Engine::from_single_file(path, vae_companion: Option<&Path>)` |
| 2.5 | `sdxl::single_file::load(SafeTensors)` (TDD) — same but with **two** text encoders. CLIP-L at `conditioner.embedders.0.transformer.text_model.*`; CLIP-G at `conditioner.embedders.1.model.*`. Pony / Illustrious / NoobAI use this loader unchanged. | new `crates/mold-inference/src/sdxl/single_file.rs`; modify `crates/mold-inference/src/sdxl/engine.rs` |
| 2.6 | Engine `create_engine` factory routing — when the catalog entry has `bundling: SingleFile` and family ∈ {Sd15, Sdxl}, call the new `from_single_file` constructor instead of the existing diffusers-style loader. Family detection from manifest may need to consult the catalog row. | `crates/mold-inference/src/factory.rs`; possibly thread a `SourceMode` enum through |
| 2.7 | Companion auto-pull on `POST /api/catalog/:id/download` (TDD) — drop the engine_phase gate from `>= 2` to `>= 3`; for entries with `companions: ["clip-l", "sdxl-vae", ...]`, enqueue each missing companion **before** the entry itself; surface as `Vec<job_id>` in response. Companion presence check: look on disk, not just in DB, since phase 1 may have left half-pulled state. | `crates/mold-server/src/catalog_api.rs::post_catalog_download`; possibly extend `crates/mold-server/src/downloads.rs::DownloadQueue::enqueue` to take a recipe instead of a manifest name |
| 2.8 | CLI `mold pull cv:<id>` integration — when the catalog row has `bundling: SingleFile` + `engine_phase ≤ 2`, dispatch through the new recipe path with companion auto-pull instead of the manifest-name fallback. Same companion-first ordering as 2.7. | `crates/mold-cli/src/commands/pull.rs` |
| 2.9 | Web Download button — already wired to `POST /api/catalog/:id/download`; it just enables once 2.7 drops the 409. Confirm `cat.canDownload(entry)` in `web/src/composables/useCatalog.ts` switches from `engine_phase === 1` to `engine_phase <= 2`. | `web/src/composables/useCatalog.ts`, `web/src/components/CatalogCard.vue` (badge text), `web/src/components/CatalogDetailDrawer.vue` (CTA copy) |
| 2.10 | Phase 2 gate: `cargo fmt --all -- --check && cargo clippy --workspace --all-targets -- -D warnings && cargo test --workspace`; `bun run test && bun run build && bun run fmt:check`; **killswitch UAT** — pull + generate one Pony entry, one Juggernaut XL entry, one SD1.5 entry (e.g. epiCRealism). Visual inspection of outputs. CHANGELOG entry. | (no code) |

The plan is intentionally smaller than multi-prompt-chain v2's 51 tasks
because the spec is more concentrated and the surface is already mostly
wired through phase 1.

## Pre-investigation the new session must do

Before dispatching any subagent, grep for and confirm:

```bash
# How does the existing SD15 engine construct itself today?
grep -rn 'impl Sd15Engine\|fn load\|from_repo\|from_diffusers' \
  crates/mold-inference/src/sd15/ | head

# How does SDXL?
grep -rn 'impl SdxlEngine\|fn load\|from_repo\|from_diffusers' \
  crates/mold-inference/src/sdxl/ | head

# Where does the engine factory branch on family?
grep -n 'create_engine\|Family::Sd15\|Family::Sdxl' \
  crates/mold-inference/src/factory.rs

# How does the existing DownloadQueue accept work?
grep -n 'pub async fn enqueue\|EnqueueError' \
  crates/mold-server/src/downloads.rs | head -20

# Catalog download handler — current 409 gate
grep -n 'engine_phase >= 2\|engine_phase >=2' \
  crates/mold-server/src/catalog_api.rs

# Companion shape
grep -n 'pub static COMPANIONS' crates/mold-catalog/src/companions.rs
```

Surprises to watch for:

- **The SD15 engine may already be diffusers-only**, with no notion of
  "load from a single safetensors file." If so, task 2.4 grows because
  it needs to refactor the engine constructor to accept either a
  diffusers tree or a pre-loaded bundle.
- **`mmap` lifetime on `SafeTensors`** — the borrow checker will not
  let you return both the `Mmap` and the `SafeTensors<'_>` from a
  function. Either store the `Mmap` in `SingleFileBundle` and use
  self-referencing structs (ouroboros / yoke), or load tensors eagerly
  into owned `candle::Tensor`s before dropping the mmap.
- **Civitai SDXL files vary in CLIP key prefix** — not every checkpoint
  uses A1111's `conditioner.embedders.*` exactly. Some use
  `cond_stage_model.*` (SD-style) for CLIP-L only. Tensor-prefix audit
  in task 2.2 must run against several representative checkpoints, not
  just one.
- **Companion presence check is racy** — a companion might be in the
  middle of being pulled by a previous request. Task 2.7 must handle
  the "enqueue while already enqueued" case (DownloadQueue probably
  returns the existing job id; verify).
- **`engine_phase >= 2` appears in multiple places** — at minimum the
  server gate + the web `canDownload` predicate + maybe a CLI guard.
  Update them all in one task or the UX gets weird (CLI pulls work,
  web button still disabled).

## Working conventions to preserve

- **TDD per task** — failing test first, implementation, gate-green,
  then commit. The phase 1 work followed this for the throttle layer
  (4 unit tests in `mold-catalog/src/stages/throttle.rs`) and the
  catalog scan progress (`run_scan_with_progress_advances_families_done`
  in `scanner_orchestrator.rs`).
- **One scope per commit** — `feat(inference) / feat(catalog) /
  feat(server) / feat(cli) / feat(web) / test(inference) /
  docs(catalog)`.
- **No mid-phase pushes.** Phase 2 lands as one push when 2.10 is green.
- **CHANGELOG.md entry under [Unreleased]** is part of 2.10, not a
  separate phase. Match the format the phase 1 entries use ("Live
  catalog-refresh progress." / `contrib/mold-server.user.service` —
  bold lead, prose body, no bullet sub-lists).
- **`superpowers:subagent-driven-development`** is appropriate for
  tasks 2.4/2.5/2.7/2.8 (mechanical). 2.2 (tensor audit) is too
  open-ended for a subagent — do it interactively.
- **`superpowers:verification-before-completion`** before declaring 2.10
  done. The killswitch UAT is the verification — don't mark phase 2
  done without a real generation succeeding.

## Killswitch deploy / verify recipe

After 2.10 gate is green locally:

```bash
# Push from your laptop
git push origin feat/catalog-expansion

# On killswitch
ssh killswitch@192.168.1.67
cd ~/github/mold && git pull
cargo build --release -p mold-ai \
  --features cuda,preview,discord,expand,tui,webp,mp4,metrics
systemctl --user restart mold-server
journalctl --user -u mold-server -f &  # tail logs

# UAT — phase 2 acceptance
mold pull cv:<sdxl-pony-id>          # should companion-pull clip-l + clip-g + sdxl-vae first
mold pull cv:<sdxl-juggernaut-id>    # companions are now cached, only the unet downloads
mold pull cv:<sd15-epicrealism-id>   # companions: clip-l + sd-vae-ft-mse
mold run cv:<sdxl-pony-id> "a cottagecore landscape, anime style"
# verify image lands in ~/.mold/output, render is sane

# Same trio via web UI
# - browse to /catalog, filter family=sdxl, click Pony entry, click Download
# - watch CatalogRefreshPanel-style progress... actually no — that's the
#   scanner. The download surface is DownloadsDrawer (top-bar button).
#   Verify each companion shows up as a separate job in the drawer in
#   the right order.
```

## Out-of-scope reminders

If the new session is tempted to:

- **Add a download-progress panel inline on `/catalog` like the refresh
  panel** — fine, but only if it's a quick win. The DownloadsDrawer
  already exists and works; don't redo it for phase 2.
- **Pre-implement FLUX/Z-Image/LTX single-file** — explicitly phases
  3/4/5. Resist.
- **Add LoRA support** — sub-project D. The `kind` column already has
  the `Lora` variant; do not start populating it.
- **Refactor the catalog scanner mid-phase** — phase 1's scanner is
  shipping. Bug fixes only.
- **Kill MOLD_OFFLOAD-style block-streaming for SDXL** — SDXL UNets are
  ~5 GB FP16 and fit comfortably; phase 2 doesn't need to touch the
  offload path.

## Verification commands the new session will need

```bash
cd /Users/jeffreydilley/github/mold

# Pre-flight: confirm phase 1 is at HEAD
git status                                    # clean
git log --oneline origin/main..HEAD | head -10

# CI gate (run before any new code, run after every task commit)
cargo fmt --all -- --check
cargo clippy --workspace --all-targets -- -D warnings
cargo test --workspace
# retry once if only `theme_save_then_load_round_trip_preserves_preset`
# fails — that's a known TUI test flake (see user memory).

# Web checks
( cd web && bun run test && bun run build && bun run fmt:check )
```

## Spec & plan references

- Spec: `docs/superpowers/specs/2026-04-25-catalog-expansion-design.md`
  — §2 is phase 2's authoritative shape. §1.3 (companion registry) and
  §1.10 (`POST /api/catalog/:id/download`) are the contracts phase 2
  fulfills.
- No detailed phase-2 plan file exists yet — the design doc is granular
  enough that a separate plan file would be pure duplication. The new
  session should treat the §2 sub-headings as the task list and adapt.
- Phase 1 history: `git log feat/catalog-expansion --oneline | head -30`
  — every phase 1 commit is one merge-able unit and the patterns there
  (atomic shard write, scanner per-family progress, single-writer
  CatalogScanQueue) are the working conventions for phase 2.
- Prior handoff style: `tasks/multi-prompt-chain-v2-resume-phase2.md`
  — same author, same shape, same TDD discipline.

---

## The prompt

Paste from here into a fresh Claude Code session:

---

I'm starting **phase 2** of the mold catalog-expansion sub-project — adding
**SD1.5 + SDXL single-file safetensors loaders**. This is the phase that
unlocks ~95 % of Civitai checkpoints for download + run via mold's existing
SD1.5 / SDXL engines. Phase 1 (catalog browse + scanner + UI + killswitch
systemd unit + live refresh progress) shipped on the same branch.

## Read first, in this order

1. `~/.claude-personal/CLAUDE.md` and `/Users/jeffreydilley/github/mold/CLAUDE.md` — coding conventions.
2. `tasks/catalog-expansion-phase-2-handoff.md` — **your primary briefing**. Read end-to-end. Status, task list (2.1–2.10), pre-investigation greps, gotchas, killswitch deploy recipe.
3. `docs/superpowers/specs/2026-04-25-catalog-expansion-design.md` — approved design. **§2 is your contract**; §1.3 and §1.10 are the contracts §2 fulfills. Do not reopen design decisions.
4. `crates/mold-catalog/src/companions.rs`, `crates/mold-catalog/src/entry.rs`, `crates/mold-server/src/catalog_api.rs::post_catalog_download` — the phase 1 surfaces phase 2 builds on.

## Status on entry

- Branch: `feat/catalog-expansion`. Phase 1 is shipped on this branch (last commit before phase 2 work begins is the systemd-handoff + live-progress commit).
- Killswitch (Arch, dual RTX 3090, sm_86) runs `mold serve` under `systemctl --user mold-server`. Tokens in `~/.config/mold/server.env`. Logs via `journalctl --user -u mold-server -f`. SSH: `killswitch@192.168.1.67`. Repo path: `~/github/mold`.
- Catalog DB on killswitch already has SDXL/SD15 Civitai entries with `engine_phase: 2`, currently 409'd at the download endpoint.

## What you're doing

Execute tasks **2.1 → 2.10** per `tasks/catalog-expansion-phase-2-handoff.md`. TDD per task. One scope per commit (`feat(inference) / feat(server) / feat(cli) / feat(web) / test(...)`). **No mid-phase pushes** — phase 2 lands as one push when 2.10 is gate-green. Open PR `feat(catalog): SD1.5 + SDXL single-file loaders (phase 2/5)` after the push.

Use `superpowers:subagent-driven-development` for the mechanical tasks (2.4 / 2.5 / 2.7 / 2.8). Tasks 2.2 (tensor-prefix audit) and 2.10 (killswitch UAT) need to be interactive — too open-ended for a subagent and the second is hardware-dependent.

Gate every commit: `cargo fmt --check && cargo clippy --workspace --all-targets -- -D warnings && cargo test --workspace`. Plus `bun run test && bun run build && bun run fmt:check` for any web-touching commit. Use `superpowers:verification-before-completion` before declaring 2.10 done — the killswitch UAT is the verification.

## How to work

- **Pre-investigation first.** Run every grep listed in `tasks/catalog-expansion-phase-2-handoff.md` § "Pre-investigation the new session must do" before touching code. The biggest risk in phase 2 is "the SD15/SDXL engines don't actually accept a pre-loaded tensor bundle today" — find out before you write the loader, not after.
- **Tensor-prefix audit (2.2) first.** Real Civitai checkpoints have key-naming variation A1111 docs don't fully cover. Download 2–3 representative SDXL files (Pony, Illustrious, NoobAI, generic SDXL fine-tune) and 2 representative SD1.5 files; print unique prefixes; reconcile against the spec's claimed paths. Without this, the loader will silently break on some entries.
- **Companion-first download ordering (2.7) is the user-visible behavior.** A user clicks Download on a Pony entry — they should see clip-l and clip-g start downloading immediately, then sdxl-vae, then the unet itself, in that order in the DownloadsDrawer. Test this end-to-end with a mocked download driver before declaring 2.7 done.
- **Pre-grep before subagent dispatch.** When 2.6 changes the engine factory, `grep -rn 'create_engine(' crates/` to enumerate every call site. Cross-crate ripples are what break `cargo test --workspace`.
- **Paste full task text** to each subagent — don't tell them to "read 2.4 from the handoff." Inline the task description, the file paths, and any pre-investigation findings.

## Start here

1. Confirm branch state:
   ```bash
   cd /Users/jeffreydilley/github/mold
   git status                                  # clean
   git log --oneline origin/main..HEAD | head -10
   cargo test --workspace                       # green; retry once on TUI flake if it fires
   ```
2. SSH to killswitch, confirm `mold-server.service` is active (`systemctl --user status mold-server`) and the catalog DB has SDXL/SD15 Civitai rows (`mold catalog list --family sdxl --json | jq '.[0].id'` — pick exemplars for the UAT). If the catalog has empty SDXL/SD15 families, run `mold catalog refresh` first; **wait for it to finish** (tail logs or watch the new live-progress panel in the web UI) before continuing.
3. Run the pre-investigation greps from the handoff. Capture findings inline in your working notes — they should change task 2.4 / 2.5 estimates if any of them surface a surprise.
4. Start task 2.1 (the killswitch state-of-DB confirmation), then 2.2 (tensor audit), then 2.3 forward.

## If you hit a surprise

If a tensor prefix doesn't match the spec, the SD15/SDXL engines need a constructor refactor that's bigger than 1 task, the DownloadQueue can't accept a URL recipe without a major rewrite, or a Civitai checkpoint uses an exotic VAE that isn't in `companions.rs` — **stop, document the surprise, ask the user before pressing forward.** The phase 1 patterns (atomic write, single-writer queue, embedded-shard fallback) are stable; surprises in phase 2 are most likely to surface at the **engine constructor boundary** and the **DownloadQueue input type**, both of which are pre-phase-1 code.

When phase 2 is gate-green and the killswitch UAT shows real generations from Civitai single-file SDXL/SD15 entries, push the branch, open the PR, and write `tasks/catalog-expansion-phase-3-handoff.md` for FLUX single-file. The phase 2 handoff (`tasks/catalog-expansion-phase-2-handoff.md`) is your template — copy its shape, swap the contents per design doc §3.
