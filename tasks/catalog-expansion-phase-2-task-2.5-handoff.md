# Phase-2 task 2.5 — SDXL single-file engine constructor (kickoff handoff)

> Paste the prompt at the bottom of this file into a fresh Claude Code
> session to start task 2.5. Everything above is reference material for
> the author / for skimming.

## Where phase 2 stands on entry

Branch `feat/catalog-expansion`. The four most recent commits are
**local-only** (phase 2 lands as one push at 2.10):

| Commit | Origin? | Scope |
|---|---|---|
| `f87687b` | local | feat(inference): SD1.5 single-file engine constructor (phase 2.4) |
| `a6991b4` | local | feat(inference): single-file checkpoint dispatcher (phase 2.3) |
| `e50128f` | yes | docs(tasks): catalog-expansion phase-2 task 2.4 kickoff handoff |
| `e7d4f4a` | yes | docs(tasks): catalog-expansion phase-2 task 2.3 kickoff handoff |
| `4970a92` | yes | docs(tasks): SD1.5 + SDXL tensor-prefix audit findings (phase 2.2) |
| `cb13e06` | yes | feat(inference): sd_singlefile_inspect dev-bin for tensor-prefix audit |
| `088ab9b` | yes | docs(tasks): catalog-expansion phase-2 kickoff handoff |
| `5965d60` | yes | chore: pre-commit + pre-push trufflehog secret scan |

`tasks/catalog-expansion-phase-2-handoff.md` is your task-list source of
truth. `tasks/catalog-expansion-phase-2-tensor-audit.md` anchors the
SDXL prefix layout (CLIP-L at
`conditioner.embedders.0.transformer.text_model.*`, CLIP-G at
`conditioner.embedders.1.model.*`, OpenCLIP layout under CLIP-G).

### Done

- **2.1 pre-flight** — same state as 2.3 / 2.4 handoffs. The
  `mold catalog refresh` job spawned in 2.1 (`d722af92-…`) is still
  walking the FLUX HF base_model graph in the background; not on the
  critical path for 2.5.
- **2.2 tensor-prefix audit** — finished (commits `cb13e06` / `4970a92`).
- **2.3 single-file dispatcher** — landed at `a6991b4`.
- **2.4 SD1.5 engine constructor** — landed at `f87687b`. New module
  `crates/mold-inference/src/loader/sd15_keys.rs` exposes
  `apply_sd15_unet_rename` / `apply_sd15_vae_rename` /
  `apply_sd15_clip_l_rename` (pure-data) and
  `build_sd15_remap(bundle: &SingleFileBundle) -> Result<Sd15Remap,
  RemapError>`. `Sd15Remap { unet, vae, clip_l, unmapped }` is the
  diffusers→a1111 lookup the future custom `SimpleBackend` will index
  by. `SD15Engine::from_single_file(model_name, single_file_path,
  clip_tokenizer, scheduler, load_strategy, gpu_ordinal)` constructor
  validates layout + remap and stashes the path on a new
  `single_file_path: Option<PathBuf>` field.
- **Decision recorded in 2.4:** **Shape A** (in-memory VarBuilder remap
  via `SimpleBackend`) — candle's per-component constructors take a
  `VarBuilder` directly, the `LoraBackend` precedent in
  `crates/mold-inference/src/flux/lora.rs:431-556` already proves
  interception works against candle in this codebase. 2.5 should do the
  same — no temp-dir extraction.

### Not yet done

- 2.5 — **this handoff's task**.
- 2.6 factory routing, 2.7 server companion auto-pull, 2.8 CLI
  integration, 2.9 web gate flip, 2.10 UAT — all downstream of 2.5.

## What 2.4 leaves on the table for 2.5

SDXL's UNet outer block layout differs from SD1.5 (the bottom down
stage *has* attentions, the down stages have 10 transformer layers
instead of 1, etc.) and SDXL adds a CLIP-G OpenCLIP encoder that is
**not** a pure prefix-strip. The audit's findings spell out exactly
what 2.5 must absorb:

1. **UNet outer block layout (SDXL)** — `block_out_channels = [320,
   640, 1280]` (3 stages, not 4). Down stage 0 has resnets only (no
   attentions), stages 1+2 have attentions. Same flip on the up side.
   Each `Transformer2DModel` block holds *10* transformer layers (not
   1), so the inner attention naming includes
   `transformer_blocks.{0..9}.…`. The outer rename surface is similar
   in spirit to 2.4 but the per-stage `[has_attn, num_resnets,
   has_downsampler]` table is different — see diffusers'
   `convert_diffusers_to_original_stable_diffusion.py` for the inverse
   direction. **Pony / Juggernaut share the same layout** (audit point
   4) — no sub-family branch.
2. **VAE** — identical to SD1.5 (audit point 3: 248 VAE keys in both
   families). 2.4's `apply_sd15_vae_rename` is wholly reusable;
   factor it out into a shared `apply_vae_rename` that 2.5 just calls.
3. **CLIP-L** — different *outer prefix* (`conditioner.embedders.0.transformer.text_model.*`
   vs SD1.5's `cond_stage_model.transformer.text_model.*`) but identical
   *inner* HF CLIP layout. Trivial — strip the new prefix and pass
   through.
4. **CLIP-G** — the load-bearing new piece. Lives at
   `conditioner.embedders.1.model.*` in OpenCLIP layout. Two distinct
   passes are required:
   - **Layout rename**:
     `transformer.resblocks.{i}.{ln_1, ln_2, attn.out_proj, mlp.c_fc,
     mlp.c_proj}` → diffusers' `text_model.encoder.layers.{i}.{layer_norm1,
     layer_norm2, self_attn.out_proj, mlp.fc1, mlp.fc2}`. Plus
     `text_projection` → `text_projection.weight` and a few embedding
     renames. Reference: diffusers `convert_open_clip_checkpoint.py`.
   - **Fused QKV split**: `attn.in_proj_weight` is a fused `[3*d, d]`
     slab that splits row-wise into `self_attn.{q,k,v}_proj.weight`,
     and `attn.in_proj_bias` (shape `[3*d]`) splits into the three
     biases. **This is a tensor-data transformation**, not just a
     rename — the simple `Option<String>` rename function shape we used
     in 2.4 is not enough. Two viable approaches:
     - **A:** Make the rename produce a `Vec<RenameOutput>` where
       each entry can be either `Direct(diffusers_key)` or
       `FusedSlice { diffusers_key, axis, component, num_components }`.
       The future `SimpleBackend` materialises the slice in `get(...)`.
     - **B:** Pre-split fused tensors at construction time and stash
       the three derived tensors on the `Sd15Remap`-equivalent struct.
     Shape A is more consistent with the rest of the loader (lazy /
     mmap-friendly). Either is fine — pick after a 5-minute spike.

## What 2.5 produces

A new module `crates/mold-inference/src/loader/sdxl_keys.rs` (sibling
of `sd15_keys.rs`) that exposes:

```rust
pub fn apply_sdxl_unet_rename(a1111_key: &str) -> Option<String>;
pub fn apply_sdxl_clip_l_rename(a1111_key: &str) -> Option<String>;
pub fn apply_sdxl_clip_g_rename(a1111_key: &str) -> Option<RenameOutput>;
// VAE reuses sd15_keys::apply_sd15_vae_rename (or a shared helper
// factored out of it during refactor).

pub enum RenameOutput {
    Direct(String),
    FusedSlice { diffusers_key: String, axis: usize, component: usize, num_components: usize },
}

pub struct SdxlRemap {
    pub unet: BTreeMap<String, String>,
    pub vae: BTreeMap<String, String>,
    pub clip_l: BTreeMap<String, String>,
    pub clip_g: BTreeMap<String, RenameOutput>,  // Direct or FusedSlice
    pub unmapped: Vec<String>,
}

pub fn build_sdxl_remap(bundle: &SingleFileBundle) -> Result<SdxlRemap, RemapError>;
```

And a sibling constructor on `SDXLEngine`:

```rust
impl SDXLEngine {
    pub fn from_single_file(
        model_name: String,
        single_file_path: PathBuf,
        clip_l_tokenizer: PathBuf,
        clip_g_tokenizer: PathBuf,        // SDXL needs both
        scheduler: Scheduler,
        load_strategy: LoadStrategy,
        gpu_ordinal: usize,
    ) -> Result<Self>;
}
```

Mirror 2.4's pattern: validate layout via `single_file::load`, validate
rename coverage via `build_sdxl_remap`, stash a
`single_file_path: Option<PathBuf>` on `SDXLEngine`. Don't materialise
the model.

## TDD shape

Three rounds, RED before GREEN per round, mirroring 2.4:

### Round 1 — rename rule unit tests (pure data)

5-7 unit tests on the new helpers. Concrete cases:

- `sdxl_unet_input_block_0_to_conv_in` — `model.diffusion_model.input_blocks.0.0.weight` →
  `conv_in.weight` (same as SD1.5).
- `sdxl_unet_input_block_with_attention_stage_1` — confirm the SDXL stage-1
  layout (which differs from SD1.5: stage 0 is resnet-only). E.g.
  `model.diffusion_model.input_blocks.4.1.transformer_blocks.5.attn1.to_q.weight` →
  `down_blocks.1.attentions.0.transformer_blocks.5.attn1.to_q.weight`
  (note: `transformer_blocks.5` — SDXL packs 10 layers per attention
  block, not 1).
- `sdxl_clip_l_strip_new_prefix` — `conditioner.embedders.0.transformer.text_model.encoder.layers.0.self_attn.q_proj.weight` →
  `text_model.encoder.layers.0.self_attn.q_proj.weight`.
- `sdxl_clip_g_resblock_layer_norm` — `conditioner.embedders.1.model.transformer.resblocks.0.ln_1.weight` →
  `RenameOutput::Direct("text_model.encoder.layers.0.layer_norm1.weight")`.
- `sdxl_clip_g_attn_in_proj_q_slice` — `conditioner.embedders.1.model.transformer.resblocks.0.attn.in_proj_weight` →
  `RenameOutput::FusedSlice { diffusers_key:
  "text_model.encoder.layers.0.self_attn.q_proj.weight", axis: 0,
  component: 0, num_components: 3 }`. Same for k, v, in_proj_bias.
- `sdxl_unrecognized_key_returns_none` — defensive test.

### Round 2 — `build_sdxl_remap` integration

One test using a synthesised tiny safetensors (reuse the
`write_fixture` pattern from `loader/single_file.rs::tests`) with 3-5
keys per component including a CLIP-G `attn.in_proj_weight` slab.
Assert `remap.clip_g.contains_key("text_model.encoder.layers.0.self_attn.q_proj.weight")`
and that the value is a `FusedSlice { component: 0, num_components: 3, ... }`.

### Round 3 — engine smoke test

`SDXLEngine::from_single_file(...)` against a synthetic checkpoint
asserts construction succeeds without panicking. Mirror 2.4's
`from_single_file_constructs_for_synthetic_sd15_checkpoint` and
`from_single_file_rejects_missing_file` tests.

## Out of scope for 2.5

- **No actual `SimpleBackend` implementation.** Both 2.4 and 2.5 ship
  the rename + constructor scaffolding only. The `SimpleBackend` that
  consumes `Sd15Remap` / `SdxlRemap` and bridges to candle's
  per-component `*::new(vb, ...)` constructors is intentionally
  deferred — it lands when 2.6 wires the factory or 2.10 UAT runs the
  first real Civitai checkpoint end-to-end.
- **No factory routing.** Factory wiring (`create_engine` choosing
  `from_single_file` vs `new`) is 2.6.
- **No companion auto-pull.** SDXL needs both CLIP-L and CLIP-G
  tokenizers from HF — 2.7 owns the pull, 2.5 just takes their paths
  as constructor args.
- **No CLI plumbing.** `mold run --single-file foo.safetensors` is 2.8.
- **No Pony / Illustrious sub-family branching.** Audit point 4
  confirmed they're structurally indistinguishable from SDXL — same
  loader path, no metadata sniffing.

## Working conventions to preserve

- TDD — failing tests first per round (3 RED-then-GREEN cycles).
- One scope per commit. 2.5 is one commit:
  `feat(inference): SDXL single-file engine constructor (phase 2.5)`.
- **Phase 2 lands as one push when 2.10 is gate-green.** This commit
  stays local. `a6991b4` (2.3) and `f87687b` (2.4) are also
  local-only.
- `superpowers:test-driven-development` is still the right skill.
- `superpowers:verification-before-completion` before declaring 2.5
  done — gate must be green:
  `cargo fmt --all -- --check && cargo clippy --workspace --all-targets -- -D warnings && cargo test --workspace`.

## Verification commands

```bash
cd /Users/jeffreydilley/github/mold

# Pre-flight
git status                                      # clean
git log --oneline origin/main..HEAD | head -10  # 8 phase-1+2 commits ahead, including a6991b4 + f87687b (local)

# After implementation
cargo fmt --all -- --check
cargo clippy --workspace --all-targets -- -D warnings
cargo test -p mold-ai-inference --lib
cargo test --workspace                          # retry once on TUI flake before blaming your changes
```

The TUI theme test flake
(`theme_save_then_load_round_trip_preserves_preset`) is documented in
user memory — retry once before blaming your changes. It did **not**
flake during 2.3 or 2.4 verification, but the race is real.

## Refactor opportunity

`apply_sd15_vae_rename` is wholly reusable for SDXL (audit point 3).
When you add `apply_sdxl_*` rename helpers, factor the VAE rename out
of `sd15_keys.rs` into a shared `loader/vae_keys.rs` (or rename the
function) and have both engines call it. Keep `sd15_keys` and
`sdxl_keys` thin and family-specific. Don't ship duplicate VAE rename
code.

---

## The prompt

Paste from here into a fresh Claude Code session:

---

I'm starting **task 2.5** of the mold catalog-expansion phase 2 — the
SDXL engine constructor that consumes the `SingleFileBundle` produced
by 2.3 and bridges A1111 → diffusers key naming so candle's existing
SDXL model can load Civitai-format checkpoints. Tasks 2.1–2.4 are done
(2.4 commit `f87687b`, **local-only**, ships the SD1.5 path that 2.5
mirrors).

## Read first, in this order

1. `~/.claude-personal/CLAUDE.md` and `/Users/jeffreydilley/github/mold/CLAUDE.md` — coding conventions. Note the LoRA-via-custom-VarBuilder pattern under "Key Design Decisions §12" — that's the precedent for the future `SimpleBackend` and informs the `RenameOutput::FusedSlice` shape this task introduces for CLIP-G.
2. `tasks/catalog-expansion-phase-2-handoff.md` — overall phase-2 brief. **§ "Task list" 2.5 row** is the canonical spec.
3. `tasks/catalog-expansion-phase-2-tensor-audit.md` — anchors which prefixes the SDXL rename rules must target. **Pay attention to point 6** (CLIP-G uses OpenCLIP layout) and **point 4** (Pony is structurally indistinguishable from SDXL — no sub-family branch).
4. `tasks/catalog-expansion-phase-2-task-2.5-handoff.md` — this file. Spells out the design (`RenameOutput` enum, `SdxlRemap` struct, fused QKV split for CLIP-G), the TDD round shape, and the refactor opportunity to share the VAE rename with `sd15_keys.rs`.
5. `crates/mold-inference/src/loader/sd15_keys.rs` — the 2.4 module you're mirroring. Note `apply_sd15_*_rename` shape (`Option<String>`), `Sd15Remap` struct, and the `build_sd15_remap` integration.
6. `crates/mold-inference/src/loader/single_file.rs` — the 2.3 module that produces the `SingleFileBundle`. Note that for SDXL, `clip_g_keys` is `Some(_)` and you'll route those through `apply_sdxl_clip_g_rename`.
7. `crates/mold-inference/src/sdxl/pipeline.rs` — current `SDXLEngine::new(...)` constructor; 2.5 adds a sibling `from_single_file(...)`.
8. `crates/mold-inference/src/flux/lora.rs::map_lora_key` — existing `LoraTarget::FusedSlice` precedent. The `RenameOutput::FusedSlice` you introduce in 2.5 mirrors that shape.
9. (Reference, do not edit) `~/.cargo/registry/src/index.crates.io-*/candle-transformers-mold-0.9.10/src/models/stable_diffusion/{unet_2d.rs, unet_2d_blocks.rs, vae.rs, clip.rs}` — `vs.pp("…")` chains are the authoritative source for diffusers key naming. SDXL uses the same SD1.5 candle infrastructure with a different `StableDiffusionConfig::sdxl()` (or similar — verify in `mod.rs`).

## What you're doing

Implement task 2.5 per this handoff. Spike CLIP-G fused QKV split shape
first (5 minutes max — pick `RenameOutput` enum vs pre-split-tensor
shape). State which shape you picked and why before writing
implementation code.

TDD: three rounds, failing tests first per round, mirroring 2.4.

- **Round 1 (rename rules):** 5-7 unit tests including the CLIP-G
  fused QKV slice case.
- **Round 2 (`build_sdxl_remap` integration):** one test using a
  synthesised tiny safetensors with 3-5 keys per component plus a
  CLIP-G `attn.in_proj_weight` slab.
- **Round 3 (engine smoke test):** `SDXLEngine::from_single_file(...)`
  constructs without panicking against a synthetic checkpoint, and
  rejects a missing-file path.

**Refactor:** factor `apply_sd15_vae_rename` out of `sd15_keys.rs` into
a shared helper (e.g. `loader/vae_keys.rs::apply_vae_rename`) when you
add `apply_sdxl_*_rename`. The audit confirmed VAE is identical between
SD1.5 and SDXL (point 3) — duplicating the rename table is wrong. Keep
`sd15_keys` and `sdxl_keys` thin and family-specific. Update 2.4's
`build_sd15_remap` to call the shared helper.

**No factory routing, no CLIP-G/CLIP-L companion auto-pull, no CLI
plumbing — those are 2.6 / 2.7 / 2.8 respectively.**

## How to work

1. Pre-flight: confirm `git status` clean and `cargo test -p mold-ai-inference --lib` green (624 tests should pass) before touching code.
2. Spike the CLIP-G fused QKV split shape (5 minutes) to pick the `RenameOutput` enum design. **State the choice before writing code.**
3. Use `superpowers:test-driven-development`. Round 1 first — rename rules are pure data, trivially testable, and surface the diffusers key naming questions early.
4. Cross-check at least 5 representative A1111 keys against the corresponding `candle_transformers::models::stable_diffusion` field names by `git grep`-ing into the candle crate (`~/.cargo/registry/src/index.crates.io-*/candle-transformers-mold-0.9.10/src/models/stable_diffusion/`). Don't trust diffusers documentation alone — candle may have its own quirks. CLIP-G in particular: SDXL uses `clip2` config (see `StableDiffusionConfig::sdxl()` in candle's `mod.rs`).
5. New module: `crates/mold-inference/src/loader/sdxl_keys.rs` (sibling of `sd15_keys.rs`, exported via `loader/mod.rs`). The engine constructor goes in `crates/mold-inference/src/sdxl/pipeline.rs` next to the existing `pub fn new(...)`.

## Verification gate before committing

```bash
cargo fmt --all -- --check
cargo clippy --workspace --all-targets -- -D warnings
cargo test -p mold-ai-inference --lib
cargo test --workspace                  # retry once on TUI flake (`theme_save_then_load_round_trip_preserves_preset`)
```

## Commit shape

Single commit when gate-green:

```
feat(inference): SDXL single-file engine constructor (phase 2.5)

Adds `SDXLEngine::from_single_file(...)` consuming the
`SingleFileBundle` produced by phase 2.3 and applying the SDXL
A1111 → diffusers rename pass, including a fused-QKV split for the
CLIP-G OpenCLIP encoder. Mirrors the SD1.5 path landed in 2.4.
Refactors VAE rename into a shared helper since the audit
confirmed VAE keys are identical between SD1.5 and SDXL (point 3).

[Note here whether you picked the RenameOutput enum approach or
pre-split-tensor approach for CLIP-G fused QKV, and one-line why.]

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
```

**Do not push.** Phase 2 lands as one push at 2.10. The 2.3
(`a6991b4`) and 2.4 (`f87687b`) commits are also local-only.

## If you hit a surprise

If candle's SDXL `StableDiffusionConfig::sdxl()` (or whatever the
constructor name is — verify in `mod.rs`) wires the UNet differently
from what the audit predicted; or the CLIP-G fused QKV split needs
sub-tensor disambiguation the `RenameOutput::FusedSlice` shape can't
express; or a Civitai checkpoint surfaces a CLIP-G key the audit didn't
anticipate — **stop, document the surprise in the handoff, ask before
pressing forward.** Prefer a clear note over inventing a workaround
that diverges from the audit's contract.

When 2.5 is gate-green and the three test rounds pass, write
`tasks/catalog-expansion-phase-2-task-2.6-handoff.md` (template: this
file) and stop. Do not start 2.6 in the same session.
