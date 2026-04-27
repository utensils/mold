# Phase-2 task 2.10 — resume handoff (UAT continuation)

> Paste the prompt at the bottom of this file into a fresh Claude Code
> session to resume task 2.10. Everything above is reference material.

## What's already done

This handoff is a **resume**, not a kickoff. Phase 2.10 was started in a
prior session and is mostly complete; only the <gpu-host> UAT is pending.

### Concretely shipped already

- **Phase-2 chain pushed.** `feat/catalog-expansion` is at `1c7a6c7`
  on origin (15 phase-2 commits + CHANGELOG entry + cherry-pick from
  PR #270, all in one push). Local gate green at HEAD.
- **CHANGELOG entry** committed (`acf56f6` —
  `docs(changelog): phase 2 — SD1.5 + SDXL single-file Civitai loaders`).
  Adds three Added bullets + two Changed bullets to `[Unreleased]`. Also
  rewrote the existing phase-1 line on disabled Download badges so the
  `[Unreleased]` block doesn't self-contradict (gate is now `>= 3`,
  was `>= 2`).
- **Cherry-picked PR #270** (`1c7a6c7` —
  `fix(web): always-on delete, script-mode composer parity, tray-aware
  action bar`) onto the catalog branch, resolving two conflicts:
  - `crates/mold-server/src/routes.rs` —
    `server_capabilities` block: combined catalog branch's
    `catalog: CatalogCapabilities {…}` with PR #270's
    `gallery: GalleryCapabilities { can_delete: true }`. The
    `gallery_delete_allowed()` helper + its toggle test were already
    auto-removed by the cherry-pick; only the conflict zone needed
    hand resolution.
  - `web/src/components/TopBar.vue` — comment-only conflict. Combined
    PR #270's accurate post-`canDelete` prop description with HEAD's
    catalog-refresh history note.
- **PR #271** retitled to `feat(catalog): SD1.5 + SDXL single-file
  loaders + web gallery UX (phase 1 + 2/5)`, body rewritten to reflect
  bundled scope, marked ready for review (was Draft).
- **PR #270 closed** with a redirect comment pointing at #271. The
  user's account now has exactly one open PR.
- **Killswitch deployed** to the new binary
  (`mold 0.9.0 (1c7a6c7 2026-04-26)`). Build took ~9 min in release
  mode with `cuda,preview,discord,expand,tui,webp,mp4,metrics`. The
  user's PATH-via-login-shell trip wire (nvcc lives at `/opt/cuda/bin`,
  non-interactive ssh doesn't pick it up) is fixed by exporting
  `PATH=/opt/cuda/bin:$PATH` at the top of any cargo build invocation.
- **`systemctl --user restart mold-server`** done. `/health` returns
  200, version reports `1c7a6c7`.

### CI status on PR #271

All 6 checks green:
- `docs` — SUCCESS
- `web` — SUCCESS
- `rust` — SUCCESS
- `coverage` — SUCCESS
- `codecov/patch` — SUCCESS
- `codecov/project` — SUCCESS

## What's still pending

### Killswitch UAT — 3 generations + web smoke check

The PR body (`Test plan` section) has the exact checklist. The work is
**non-coding**; it validates that the deployed binary actually generates
images for SD1.5 / SDXL single-file Civitai checkpoints.

**Critical blocker discovered during the prior session:**

The <gpu-host>'s catalog DB had **no SDXL or SD15 entries**. Phase-1
shipped with empty seed shards (`crates/mold-catalog/data/catalog/*.json`
each contains `"entries": []`). Catalog rows are populated only by
`mold catalog refresh`, which scans HF + Civitai. The prior session
tried to refresh both families:

1. First attempt: `POST /api/catalog/refresh -d '{"families":["sdxl","sd15"]}'`
   was ignored (the body field is **`family` (singular)**, not `families`).
   Defaulted to a 9-family scan.
2. Second attempt: `{"family": "sdxl"}` (correct), then
   `{"family": "sdxl", "min_downloads": 50000}` (also correct).
3. **The HF scanner stage is genuinely slow** for SDXL — the prior
   session left it running on `current_family=sdxl, current_stage=hf,
   families_done=0` for 18+ minutes with no error in
   `journalctl --user -u mold-server`. SDXL has thousands of HF
   fine-tunes; the scanner appears to walk all of them before the
   `min_downloads` filter kicks in.

When you resume, **first re-check the scan state**:

```bash
ssh <gpu-host> 'curl -s http://127.0.0.1:7680/api/catalog/refresh | jq -c .active.status'
ssh <gpu-host> 'curl -s http://127.0.0.1:7680/api/catalog/families | jq ".families[] | select(.family==\"sdxl\" or .family==\"sd15\")"'
```

- If `state=null` and SDXL has `finetune>0` → scan finished, proceed
  to refresh SD15 then run the UAT.
- If `state=running` and `families_done=0` for >20 min → the scanner
  is hanging. Pivot to one of the fallback paths in **§ "If the HF
  scan won't finish"** below.
- If `state=null` and SDXL `finetune=0` → previous scan errored out;
  re-trigger with `{"family":"sdxl","min_downloads":50000}`.

### UAT recipe (when you have catalog rows)

```bash
ssh <gpu-host>

# Pick exemplars from the populated catalog
~/github/mold/target/release/mold catalog list --family sdxl --json \
  | jq -c '.[] | select(.name | test("Pony"; "i")) | {id, name}' | head -3
~/github/mold/target/release/mold catalog list --family sdxl --json \
  | jq -c '.[] | select(.name | test("Juggernaut"; "i")) | {id, name}' | head -3
~/github/mold/target/release/mold catalog list --family sd15 --json \
  | jq -c '.[] | select(.name | test("epiCRealism|DreamShaper|Realistic Vision"; "i")) | {id, name}' | head -3

# UAT: three pulls + three generations
~/github/mold/target/release/mold pull cv:<sdxl-pony-id>          # expect: clip-l + clip-g + sdxl-vae companions, then primary
~/github/mold/target/release/mold pull cv:<sdxl-juggernaut-id>    # expect: companions cached, only primary downloads
~/github/mold/target/release/mold pull cv:<sd15-epicrealism-id>   # expect: clip-l + sd-vae-ft-mse companions

~/github/mold/target/release/mold run cv:<sdxl-pony-id> "a cottagecore landscape, anime style"
~/github/mold/target/release/mold run cv:<sdxl-juggernaut-id> "a portrait of an astronaut on mars, photorealistic, golden hour"
~/github/mold/target/release/mold run cv:<sd15-epicrealism-id> "a cinematic shot of a cat in a library"

ls -lh ~/.mold/output/ | tail -10
```

Visual-inspect each output — does it look like the model? No NaN /
black / scrambled tiles? Reasonable inference time (SDXL ~15-30s on a
3090; SD1.5 ~5-10s)? If anything fails, see triage in
`tasks/catalog-expansion-phase-2-task-2.10-handoff.md` § C.

### Web UI smoke (PR #270 cherry-pick verification)

Browse to `http://<gpu-host>:7680/`:

- `/catalog` — Download button enabled for the freshly-pulled phase-2
  entries; DownloadsDrawer (top-bar button) shows companions in pull
  order.
- `/gallery` — bulk select + delete works without
  `MOLD_GALLERY_ALLOW_DELETE` set anywhere. Selection action bar floats
  above the GPU/CPU `ResourceTray` even when the tray is expanded.
- `/generate` — the 🖼️ source-image button + ⚙ settings button
  render in *both* Single mode and Script mode. In Script mode,
  clicking 🖼️ targets stage 0.

### When the UAT passes — post results to PR #271

```bash
gh pr comment 271 --body "$(cat <<'EOF'
## Killswitch UAT — passed ✓

Generated three images on `<gpu-host>` (dual GPUs):

- `cv:<sdxl-pony-id>` (Pony SDXL): <observation>, <inference time>
- `cv:<sdxl-juggernaut-id>` (Juggernaut XL): <observation>, <inference time>
- `cv:<sd15-epicrealism-id>` (epiCRealism SD1.5): <observation>, <inference time>

Output sizes range from <small>–<large> KB; visual inspection clean
(no NaN / black / scrambled tiles). Companion ordering in
DownloadsDrawer correct (clip-l, clip-g, sdxl-vae for SDXL;
clip-l, sd-vae-ft-mse for SD1.5).

Web smoke (PR #270 cherry-pick): bulk delete works without env var,
composer 🖼️ + ⚙ render in both Single and Script modes, selection
action bar floats above the expanded ResourceTray.

Phase 2 complete. Ready to merge.
EOF
)"
```

### If the HF scan won't finish

Two fallback paths:

**A. Insert SDXL/SD15 catalog rows by hand.** The `download_recipe`
JSON shape is in `crates/mold-catalog/src/entry.rs::DownloadRecipe`.
Companion file references are in
`crates/mold-catalog/src/companions.rs::CANONICAL_COMPANIONS`. The
SQLite `catalog` table schema is in
`crates/mold-db/src/migrations.rs` (v7). You can craft a row for a
known Civitai modelVersionId (Pony V6 XL = 290640, Juggernaut XL v9 =
782002, epiCRealism = 143906) by writing `INSERT INTO catalog ...`
with a hand-written recipe JSON. Validates the pull recipe path
end-to-end without a live HF scan.

**B. Skip the catalog browser, drive the HTTP layer directly.** The
recipe path is implemented in
`mold-server/src/downloads.rs::enqueue_recipe`. You can call
`POST /api/catalog/<id>/download` against a hand-crafted recipe by
inserting the row first (option A), then triggering download. Less
preferable because it's two layers of test scaffolding.

**C. Defer the UAT, accept the deferred-UAT contract.** PR #271's
body already explicitly says "<gpu-host> UAT: pending — gates the
merge, not the push." The PR can sit open with green CI and a
deferred-UAT note. Honest, non-greenwashed. The UAT happens before
merge by James or whoever runs it. Note this option only if A and B
are both unworkable.

## Reference reading

- `tasks/catalog-expansion-phase-2-handoff.md` — overall phase-2 brief.
- `tasks/catalog-expansion-phase-2-task-2.10-handoff.md` — original
  2.10 kickoff. The recipe in § C is canonical.
- `crates/mold-catalog/src/companions.rs` — companion registry.
- `crates/mold-catalog/src/scan.rs` (or wherever the HF stage lives)
  — if you want to understand why the HF scanner is slow.
- `CHANGELOG.md` — `[Unreleased]` block has the phase-2 + cherry-pick
  entries that summarise what shipped.

## Important machine details

- **<gpu-host>** — GPU host, dual GPUs, repo
  at `~/github/mold`. Build with `cargo build --release -p mold-ai
  --features cuda,preview,discord,expand,tui,webp,mp4,metrics`. Always
  `export PATH=/opt/cuda/bin:$PATH` at the top of a non-interactive
  ssh build invocation — login shell adds it but `ssh host 'cargo
  build'` doesn't.
- The HF + Civitai tokens live in `~/.config/mold/server.env`. The
  systemd unit `~/.config/systemd/user/mold-server.service` reads them.
  Tokens are *not* on the laptop — pulls run on <gpu-host>.
- `mold-server` listens on `0.0.0.0:7680`. Web UI is the embedded SPA;
  test from your laptop via `MOLD_HOST=http://<gpu-host>:7680` or
  the URL directly.
- Background mold-server logs:
  `journalctl --user -u mold-server -f`.

## Working conventions

- **No mid-phase pushes.** The single phase-2 push is already done.
  Resuming should NOT push anything new unless a UAT-discovered bug
  needs fixing.
- **`superpowers:verification-before-completion` applies.** Don't
  claim the UAT passed without the three image files in
  `~/.mold/output/`, the `journalctl` log lines, and a posted PR
  comment. Evidence before assertions.
- **The PR title and body are correct already.** Don't retitle on
  resume unless scope materially changes. The body already describes
  the bundled scope and explicit UAT-pending state.

---

## The prompt

Paste from here into a fresh Claude Code session:

---

I'm resuming **task 2.10** of the mold catalog-expansion phase 2 — the
<gpu-host> UAT step. Everything except the UAT itself is already done:
phase-2 chain pushed, PR #271 open with green CI, PR #270 cherry-picked
in and closed, <gpu-host> built + restarted on the new binary. The
prior session hit a blocker: the HF scanner stage of the catalog refresh
is slow for SDXL (probably paginating thousands of fine-tunes), and
without populated SDXL / SD15 catalog rows the recipe-driven `mold pull
cv:<id>` path can't be tested.

This task is **non-coding** in the normal sense — there is no TDD
round. The work is:

1. Re-check the <gpu-host> SDXL scan state. If complete, refresh SD15
   too. If still hanging, pick one of the fallback paths in this
   handoff's "If the HF scan won't finish" section.
2. Once SDXL + SD15 catalog rows exist, pick exemplar IDs (Pony,
   Juggernaut XL, epiCRealism / DreamShaper / Realistic Vision).
3. Three pulls + three generations on <gpu-host>. Visual-inspect each
   output. Web UI smoke check on the catalog Download flow + the
   PR #270 cherry-pick fixes (always-on delete, composer parity,
   tray-aware bar).
4. Post the UAT outcome as a comment on PR #271 — pass or fail with
   evidence (image paths, sizes, journalctl excerpts).

## Read first, in this order

1. `~/.claude-personal/CLAUDE.md` and
   `/Users/jeffreydilley/github/mold/CLAUDE.md` — coding conventions.
2. `tasks/catalog-expansion-phase-2-task-2.10-resume-handoff.md` — this
   file. The "What's already done" section is your scope; the "What's
   still pending" section is your work.
3. `tasks/catalog-expansion-phase-2-handoff.md` — overall phase-2
   brief. § "Killswitch deploy / verify recipe" is the canonical UAT
   recipe.
4. **Check** `git log --oneline origin/main..HEAD | head -20` — confirm
   the chain is intact and HEAD is `1c7a6c7`.
5. **Check** `gh pr view 271 --json state,statusCheckRollup` — confirm
   PR is OPEN, ready for review, all 6 CI checks SUCCESS.

## What you're doing

Implement task 2.10's UAT step per this handoff. Three deliverables:

1. Get the <gpu-host> catalog populated for SDXL + SD15. Either by
   waiting out the HF scan, or by one of the fallback paths in
   "If the HF scan won't finish".
2. Three pulls + three generations on <gpu-host>. Visual inspection
   of each output (no NaN / black / scrambled tiles). Web UI smoke
   check at `http://<gpu-host>:7680/`.
3. Post UAT pass/fail comment on PR #271 with evidence.

## How to work

1. Pre-flight: `git status` clean, on `feat/catalog-expansion` at
   `1c7a6c7`.
2. SSH to <gpu-host>, check scan state. Branch on what you find.
3. Once catalog is populated, run the UAT recipe. Eyeball outputs.
4. Post the comment. Update the PR body's UAT checkboxes if needed.

## If you hit a surprise

- **HF scan still hanging** — pick a fallback path (A/B/C in this
  handoff). Don't sit on a non-progressing scan for >30 min without
  pivoting.
- **Pull errors with "catalog row not found"** — the recipe path
  needs a populated row; double-check the scan completed for that
  family.
- **Generation NaN / black / scrambled** — likely a candle-mold
  integration bug. Bisect against commit `5f7d0af` (the 2.8.5
  candle-mold wire-up). The `SingleFileBackend` impl in
  `crates/mold-inference/src/loader/single_file_backend.rs` is
  the suspect.
- **PR #271's CI flips red** — investigate, but the local gate at
  HEAD is green and CI was green at last check. If a CI rerun has
  flaked, retry once before debugging.

## Verification gate

The <gpu-host> UAT IS the verification gate. PR comment requires:

- Three image files in `~/.mold/output/` on <gpu-host> (one per UAT
  generation), each visually inspected and not a failure mode.
- Web UI smoke check passed: SPA Download button works for one phase-2
  entry, DownloadsDrawer shows companion ordering correctly, gallery
  delete works without env var, composer 🖼️ + ⚙ visible in both modes.

## When you're done

Phase 2 is complete. Post the UAT pass comment on PR #271. The PR is
ready for merge.

Phase 3 (FLUX single-file) is the next sub-project off `main`. Cut a
new branch from `main` after #271 merges — phase 3 is more involved
than phase 2 because FLUX single-file checkpoints carry only the
transformer (T5 is too big to bundle), so they need a separate
companion-first flow that pulls the full T5 + CLIP encoder set.
