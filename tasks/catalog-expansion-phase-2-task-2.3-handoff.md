# Phase-2 task 2.3 — `loader/single_file.rs` dispatcher (kickoff handoff)

> Paste the prompt at the bottom of this file into a fresh Claude Code
> session to start task 2.3. Everything above is reference material for
> the author / for skimming.

## Where phase 2 stands on entry

Branch `feat/catalog-expansion`, all of the following on `origin`:

| Commit | Scope |
|---|---|
| `4970a92` | docs(tasks): SD1.5 + SDXL tensor-prefix audit findings (phase 2.2) |
| `cb13e06` | feat(inference): sd_singlefile_inspect dev-bin for tensor-prefix audit |
| `088ab9b` | docs(tasks): catalog-expansion phase-2 kickoff handoff |
| `5965d60` | chore: pre-commit + pre-push trufflehog secret scan |
| `3557ded` | chore(contrib): user-mode systemd unit + env example + README |
| `614d5f3` | feat(catalog): live scan progress + inline refresh panel |
| `ca072e8` | (head of phase-1 prior to this session) |

`tasks/catalog-expansion-phase-2-handoff.md` is your task-list source of
truth. `tasks/catalog-expansion-phase-2-tensor-audit.md` is the
findings doc that **narrows 2.3's scope**: what was thought to be a
"how does the loader detect CLIP layout per-checkpoint?" risk is now a
solved question — see "Audit findings that 2.3 builds on" below.

### Done

- **2.1 pre-flight** — <gpu-host> (`<gpu-host>`, dual-GPU <arch-tag>, `~/github/mold`) is on the same branch, rebuilt at `088ab9b`, `mold-server.service` running. A `mold catalog refresh` (job `d722af92-7f3a-4dca-bee8-4f00f2f32eb4`) was kicked off before phase 2 work began and is still walking the flux HF base_model graph in the background — it will eventually populate SD15 + SDXL Civitai entries needed for 2.10 UAT, but **not blocking** any of 2.3 / 2.4 / 2.5 / 2.6 / 2.7 / 2.8 / 2.9. Poll: `ssh <gpu-host> "curl -sS http://localhost:7680/api/catalog/refresh/d722af92-7f3a-4dca-bee8-4f00f2f32eb4 | jq ."`
- **2.2 tensor-prefix audit** — `sd_singlefile_inspect` dev-bin shipped, audit run against three Civitai checkpoints (DreamShaper 8 / SD1.5 — 1131 tensors, Juggernaut XL Ragnarok / SDXL — 2516, Pony Diffusion V6 / SDXL — 2515). Findings doc landed.

### Not yet done

- 2.3 — **this handoff's task**.
- 2.4 SD15 loader, 2.5 SDXL loader, 2.6 factory routing, 2.7 server companion auto-pull, 2.8 CLI integration, 2.9 web gate flip, 2.10 UAT — all downstream of 2.3.

## Audit findings that 2.3 builds on

Read `tasks/catalog-expansion-phase-2-tensor-audit.md` end-to-end before
writing code. Decisive points for 2.3:

1. **The UNet and VAE prefixes are universal across SD1.5 + SDXL Civitai single-files.** UNet is always `model.diffusion_model.*`. VAE is always `first_stage_model.*`. No A1111 / kohya / WebUI variant in the audited set deviates. **Don't write per-file detection logic.**
2. **CLIP location is family-determined, not checkpoint-determined.** SD15 → `cond_stage_model.transformer.text_model.*` only. SDXL → `conditioner.embedders.0.transformer.text_model.*` (CLIP-L) plus `conditioner.embedders.1.model.*` (CLIP-G). The handoff's worry that some SDXL files might mix in `cond_stage_model.*` for CLIP-L did **not** materialize. **Family alone tells you the prefix.**
3. **Pony is structurally indistinguishable from generic SDXL.** Same UNet count (1680), same VAE count (248), same CLIP-L (197), same CLIP-G (390). No sub-family branch needed in the dispatcher.
4. **VAE counts are identical between SD1.5 and SDXL (248 tensors).** Both families use the same VAE architecture; the loader can share VAE partition logic.
5. **Stray tensors must be tolerated, not erroneous.** Juggernaut carries an extra `denoiser.sigmas` (1 × F32) — a custom sigma noise schedule baked in for some downstream tool. Inert to the canonical SDXL engine. Loader contract: log + ignore unmapped tensors at the bundle level; never error.
6. **Inner block / attention / FF naming is identical** between A1111 and diffusers below the path-translation boundary. That is the **2.4 / 2.5** problem (rename rules), not 2.3's. 2.3 only partitions keys by component prefix.

## What 2.3 produces

A new module:

```
crates/mold-inference/src/loader/
├── mod.rs                  // re-exports + LoadError
└── single_file.rs          // SingleFileBundle, load(path, family)
```

Recommended type shape (adapt as TDD shapes it):

```rust
/// Result of partitioning a Civitai single-file safetensors into
/// recognised component buckets. Carries only the original key names,
/// not parsed tensors — phase 2.4 / 2.5 do the diffusers-rename pass
/// and hand the renamed keys to candle's existing
/// `MmapedSafetensors::multi(&[path])` to materialise tensors lazily.
/// Keeping 2.3 zero-copy avoids the `Mmap` + `SafeTensors<'_>` lifetime
/// gotcha the handoff flagged.
pub struct SingleFileBundle {
    pub path: PathBuf,
    pub unet_keys: Vec<String>,                 // model.diffusion_model.*
    pub vae_keys: Vec<String>,                  // first_stage_model.*
    pub clip_l_keys: Vec<String>,               // SD15: cond_stage_model.transformer.text_model.*
                                                // SDXL: conditioner.embedders.0.transformer.text_model.*
    pub clip_g_keys: Option<Vec<String>>,       // SDXL only: conditioner.embedders.1.model.*
    pub unknown_keys: Vec<String>,              // ex: denoiser.sigmas — log + ignore
}

#[derive(Debug, thiserror::Error)]
pub enum LoadError {
    #[error("io: {0}")]
    Io(#[from] std::io::Error),
    #[error("safetensors header: {0}")]
    Header(String),
    #[error("family {0:?} is not a single-file family yet (phase {1})")]
    UnsupportedFamily(Family, u8),
}

pub fn load(path: &Path, family: Family) -> Result<SingleFileBundle, LoadError>;
```

`load()` for FLUX / Flux2 / Z-Image / LtxVideo / Ltx2 / QwenImage / Wuerstchen returns `Err(LoadError::UnsupportedFamily(f, engine_phase_for(f, Bundling::SingleFile)))` — that integer is wired to the existing `engine_phase` taxonomy and tells callers which mold release will support it.

The header parse should reuse the same pattern as
`crates/mold-inference/src/bin/sd_singlefile_inspect.rs` (the
`read_safetensors_header` helper). Don't re-mmap the whole file in
2.3 — only the header is needed for partitioning.

### TDD shape

Three or four unit tests:

- `partition_sd15_dreamshaper_layout` — feed a synthesised tiny safetensors with one tensor at each of the SD15 prefixes (`model.diffusion_model.foo`, `first_stage_model.bar`, `cond_stage_model.transformer.text_model.baz`) plus an unknown key, assert the bundle's keys and `clip_g_keys: None`.
- `partition_sdxl_layout` — same, but with the four SDXL prefixes (`model.diffusion_model.*`, `first_stage_model.*`, `conditioner.embedders.0.transformer.text_model.*`, `conditioner.embedders.1.model.*`) plus a `denoiser.sigmas` to anchor that strays land in `unknown_keys`.
- `unsupported_family_returns_error` — `load(path, Family::Flux)` returns `Err(LoadError::UnsupportedFamily(Family::Flux, 3))`. Same for Z-Image (4), LtxVideo (5), Ltx2 (5), QwenImage (99), Wuerstchen (99). Use a parameterised loop or table-driven test.
- `partition_pony_uses_sdxl_path` — with `family == Sdxl` plus a Pony-shaped fixture (no metadata, all the SDXL prefixes), confirm the bundle is identical in shape to the generic-SDXL case. **No sub-family branching.**

Test fixtures: emit a minimal valid safetensors with one or two tiny F32 tensors per probe, written via the `safetensors::tensor::serialize` path (see `crates/mold-inference/src/weight_loader.rs` and `crates/mold-inference/src/ltx2/lora.rs:252` for examples already in the tree). 2.3 should not require the multi-GB Civitai fixtures at `~/Downloads/civitai-fixtures/`. Those are for human-driven re-runs of the inspector when a future checkpoint surprises us.

## Out of scope for 2.3

- **No diffusers-key rename.** Bundle holds original A1111 keys verbatim. Renames are 2.4 / 2.5.
- **No tensor materialisation.** Bundle does not parse tensor data, does not call `MmapedSafetensors::multi`, does not allocate `candle::Tensor`s.
- **No engine constructor changes.** `SD15Engine::new(...)` and `SDXLEngine::new(...)` are touched in 2.4 / 2.5, not here.
- **No factory routing.** Factory wiring is 2.6.
- **No `mmap` lifetime acrobatics.** Returning `Vec<String>` per component sidesteps the borrow checker entirely. Trust candle's `MmapedSafetensors::multi` to handle lifetimes when 2.4 / 2.5 actually materialise tensors.

## Working conventions to preserve

- TDD per task — failing test first, implementation, gate-green, then commit.
- One scope per commit. 2.3 is one commit: `feat(inference): single-file checkpoint dispatcher (phase 2.3)`.
- **Phase 2 lands as one push when 2.10 is gate-green.** This commit stays local. (The tensor-audit + dev-bin commits already on origin were exceptions because they had standalone diagnostic value.)
- `superpowers:test-driven-development` is the right skill for 2.3 (small, mechanical, well-defined contract).
- `superpowers:verification-before-completion` before declaring 2.3 done — gate must be green: `cargo fmt --all -- --check && cargo clippy --workspace --all-targets -- -D warnings && cargo test --workspace`.

## Verification commands

```bash
cd /Users/jeffreydilley/github/mold

# Pre-flight
git status                                      # clean
git log --oneline origin/main..HEAD | head -10  # 6 phase-1+2 commits ahead

# After implementation
cargo fmt --all -- --check
cargo clippy --workspace --all-targets -- -D warnings
cargo test -p mold-ai-inference --lib
cargo test --workspace                          # retry once on TUI flake
```

The TUI theme test flake (`theme_save_then_load_round_trip_preserves_preset`) is documented in user memory — retry once before blaming your changes.

---

## The prompt

Paste from here into a fresh Claude Code session:

---

I'm starting **task 2.3** of the mold catalog-expansion phase 2 — the `loader/single_file.rs` dispatcher that partitions a Civitai single-file safetensors into UNet / VAE / CLIP-L / CLIP-G key buckets. Tasks 2.1 (pre-flight) and 2.2 (tensor-prefix audit) are done; their findings narrow 2.3's scope substantially.

## Read first, in this order

1. `~/.claude-personal/CLAUDE.md` and `/Users/jeffreydilley/github/mold/CLAUDE.md` — coding conventions.
2. `tasks/catalog-expansion-phase-2-handoff.md` — overall phase-2 brief. **§ "Task list" 2.3 row** is the canonical spec. **§ "Pre-investigation"** + **§ "Surprises to watch for"** are still relevant.
3. `tasks/catalog-expansion-phase-2-tensor-audit.md` — **the audit findings 2.3 builds on**. Read end-to-end. Specifically: family alone determines CLIP layout, Pony is structurally identical to generic SDXL, stray tensors must be tolerated.
4. `tasks/catalog-expansion-phase-2-task-2.3-handoff.md` — this file. Has the recommended `SingleFileBundle` shape, TDD test list, and the out-of-scope rails.
5. `crates/mold-inference/src/bin/sd_singlefile_inspect.rs` — the dev-bin from 2.2. Reuse its `read_safetensors_header` pattern; **don't** mmap the whole file just for header parsing.
6. `crates/mold-catalog/src/families.rs` and `crates/mold-catalog/src/civitai_map.rs` — `Family` enum + `engine_phase_for(Family, Bundling::SingleFile)` table that drives `LoadError::UnsupportedFamily`.

## What you're doing

Implement task 2.3 per the handoff: new module at `crates/mold-inference/src/loader/{mod.rs, single_file.rs}` exporting a `SingleFileBundle` + `LoadError` + `load(path, family)` function. SD15 + SDXL paths populate the bundle from prefixed keys; FLUX / Flux2 / Z-Image / LtxVideo / Ltx2 / QwenImage / Wuerstchen return `Err(LoadError::UnsupportedFamily(family, phase))`.

TDD: failing test first, then implementation. Four tests target the contract:
- `partition_sd15_dreamshaper_layout`
- `partition_sdxl_layout`
- `unsupported_family_returns_error` (table-driven across all unsupported families)
- `partition_pony_uses_sdxl_path`

Build synthetic minimal safetensors fixtures via `safetensors::tensor::serialize` (reference: `crates/mold-inference/src/weight_loader.rs` and `crates/mold-inference/src/ltx2/lora.rs`). **No multi-GB fixtures.**

## Out of scope

- **No diffusers-key rename rules.** Bundle holds original A1111 keys. Renames are 2.4 / 2.5.
- **No tensor materialisation.** Bundle holds `Vec<String>` per component, not `candle::Tensor`s.
- **No engine constructor changes.** `SD15Engine::new` / `SDXLEngine::new` get the new constructors in 2.4 / 2.5.
- **No factory routing.** Factory is 2.6.

## How to work

1. Pre-flight: confirm `git status` clean and `cargo test -p mold-ai-inference --lib` green before touching code.
2. Use `superpowers:test-driven-development`. Write the four failing tests first; commit nothing until at least one is green.
3. Reuse `read_safetensors_header` from the 2.2 dev-bin — header-only parse, no mmap.
4. The `LoadError::UnsupportedFamily(family, phase)` arm wires to `mold_catalog::civitai_map::engine_phase_for(family, Bundling::SingleFile)` so the error's phase number tracks the canonical taxonomy. (Yes, this means depending on `mold-catalog` from `mold-inference` — verify the dep arrow is OK before adding it; the existing `factory.rs` file may give you a precedent.)

## Verification gate before committing

```bash
cargo fmt --all -- --check
cargo clippy --workspace --all-targets -- -D warnings
cargo test -p mold-ai-inference --lib
cargo test --workspace                  # retry once on TUI flake (`theme_save_then_load_round_trip_preserves_preset`) before blaming your changes
```

## Commit shape

Single commit when gate-green:

```
feat(inference): single-file checkpoint dispatcher (phase 2.3)

Adds `crates/mold-inference/src/loader/single_file.rs` partitioning a
Civitai single-file safetensors into UNet / VAE / CLIP-L / CLIP-G key
buckets by prefix. SD15 + SDXL paths return `Ok(SingleFileBundle)`;
FLUX / Flux2 / Z-Image / LtxVideo / Ltx2 / QwenImage / Wuerstchen
return `LoadError::UnsupportedFamily(family, engine_phase)`. Pony goes
through the SDXL path unchanged per the 2.2 audit. Stray tensors land
in `unknown_keys` and are logged + dropped — never erroneous.

Inner key renames (A1111 ↔ diffusers) and tensor materialisation are
deferred to tasks 2.4 / 2.5 by design.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
```

**Do not push.** Phase 2 lands as one push at 2.10. This commit stays local until then.

## If you hit a surprise

If `mold-inference` cannot depend on `mold-catalog` (cycle, feature gate, build order), or if `safetensors`'s header parser surfaces an SDXL checkpoint in the audit fixtures with structurally divergent keys (none did in 2.2, but a future fixture might), or if `engine_phase_for` is missing a `Family` variant we need — **stop, document the surprise, ask before pressing forward.** Do not invent a workaround that diverges from the audit's contract.

When 2.3 is gate-green and the four tests pass, write `tasks/catalog-expansion-phase-2-task-2.4-handoff.md` (template: this file) and stop. Do not start 2.4 in the same session — context budget is better spent fresh.
