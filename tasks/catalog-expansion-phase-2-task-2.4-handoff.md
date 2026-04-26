# Phase-2 task 2.4 — SD1.5 single-file engine constructor (kickoff handoff)

> Paste the prompt at the bottom of this file into a fresh Claude Code
> session to start task 2.4. Everything above is reference material for
> the author / for skimming.

## Where phase 2 stands on entry

Branch `feat/catalog-expansion`. `a6991b4` is **local-only** (phase 2
lands as one push at 2.10):

| Commit | Origin? | Scope |
|---|---|---|
| `a6991b4` | local | feat(inference): single-file checkpoint dispatcher (phase 2.3) |
| `e7d4f4a` | yes | docs(tasks): catalog-expansion phase-2 task 2.3 kickoff handoff |
| `4970a92` | yes | docs(tasks): SD1.5 + SDXL tensor-prefix audit findings (phase 2.2) |
| `cb13e06` | yes | feat(inference): sd_singlefile_inspect dev-bin for tensor-prefix audit |
| `088ab9b` | yes | docs(tasks): catalog-expansion phase-2 kickoff handoff |
| `5965d60` | yes | chore: pre-commit + pre-push trufflehog secret scan |

`tasks/catalog-expansion-phase-2-handoff.md` is your task-list source of
truth. `tasks/catalog-expansion-phase-2-tensor-audit.md` is still the
findings doc that anchors which prefixes the renames must target.

### Done

- **2.1 pre-flight** — same state as 2.3's handoff. `mold catalog refresh`
  job `d722af92-…` is still walking the flux HF base_model graph in the
  background; no longer a critical-path dependency for 2.4.
- **2.2 tensor-prefix audit** — finished (commits `cb13e06`/`4970a92`).
- **2.3 single-file dispatcher** — landed at `a6991b4`. New module
  `crates/mold-inference/src/loader/single_file.rs` exposes
  `SingleFileBundle { path, unet_keys, vae_keys, clip_l_keys,
  clip_g_keys, unknown_keys }` and `load(path, family) ->
  Result<SingleFileBundle, LoadError>`. SD15+SDXL → `Ok(...)`; the seven
  other families → `LoadError::UnsupportedFamily(family,
  engine_phase_for(family, Bundling::SingleFile))`. **Header-only parse,
  no tensor data, no rename rules.**

### Not yet done

- 2.4 — **this handoff's task**.
- 2.5 SDXL loader, 2.6 factory routing, 2.7 server companion auto-pull,
  2.8 CLI integration, 2.9 web gate flip, 2.10 UAT — all downstream
  of 2.4.

## What 2.3 leaves on the table

The bundle hands you the original A1111 keys grouped by component, but
**not in a form candle can consume**. candle's
`stable_diffusion::build_unet` / `build_vae` /
`build_clip_transformer` expect tensors named per the diffusers
layout. The audit's depth-2 dump (DreamShaper 8 / SD 1.5) shows what
A1111 actually ships for the UNet:

```
model.diffusion_model.input_blocks.{0..11}.*           ← diffusers `down_blocks.{0..3}.*` (with downsamplers)
model.diffusion_model.middle_block.*                   ← diffusers `mid_block.*`
model.diffusion_model.output_blocks.{0..11}.*          ← diffusers `up_blocks.{0..3}.*` (with upsamplers)
model.diffusion_model.time_embed.{0,2}.{weight,bias}   ← diffusers `time_embedding.linear_{1,2}.*`
model.diffusion_model.{input,output,middle}_blocks…    ← inner block layout (`in_layers.{0,2}` ↔ `norm{1,2}`/`conv{1,2}`,
                                                          `emb_layers.1` ↔ `time_emb_proj`, `out_layers.*`,
                                                          `transformer_blocks.0` ↔ `attentions.0.transformer_blocks.0`, etc.)
```

VAE (`first_stage_model.*`) and CLIP-L
(`cond_stage_model.transformer.text_model.*`) have similar inner
renames. **2.4 owns the SD1.5 rename table.** SDXL gets the same
treatment in 2.5 (CLIP-G is the new variable, but the inner block
naming below the family path is the same as SD15 below
`down_blocks/mid_block/up_blocks`).

## What 2.4 produces

Two viable shapes — **pick after a five-minute spike on candle's
`build_unet`/`build_vae`/`build_clip_transformer` VarBuilder
expectations**:

### Shape A — `SingleFileBundle` → in-memory remap (preferred)

A new module `crates/mold-inference/src/loader/sd15_keys.rs` (or
`single_file_sd15.rs`) that exposes:

```rust
/// A1111 → diffusers rename table for SD1.5 components. Pure data.
pub fn sd15_unet_rename_rules() -> &'static [RenameRule];
pub fn sd15_vae_rename_rules() -> &'static [RenameRule];
pub fn sd15_clip_l_rename_rules() -> &'static [RenameRule];

/// Apply the SD1.5 rename rules to a SingleFileBundle, producing a
/// component-keyed map of `<diffusers_name> -> <original_a1111_name>`
/// the engine can hand candle's VarBuilder.
pub fn build_sd15_remap(bundle: &SingleFileBundle) -> Result<Sd15Remap, RemapError>;

pub struct Sd15Remap {
    pub unet: BTreeMap<String, String>,    // diffusers_key -> a1111_key
    pub vae: BTreeMap<String, String>,
    pub clip_l: BTreeMap<String, String>,
}
```

The engine then constructs a custom `VarBuilder` backend that mmaps the
original `.safetensors` once and intercepts `vb.get(diffusers_name)`
calls by routing them through the remap. The pattern is identical in
shape to `flux/lora.rs::LoraBackend` — see CLAUDE.md's
"LoRA via custom VarBuilder backend" point. **No copy, no temp files.**

### Shape B — extract per-component tensors to a temp dir (fallback)

If the VarBuilder remap turns out to be infeasible (e.g. candle's
`build_unet` calls `vb.contains_tensor` introspectively in a way the
remap can't fake), fall back to writing three small safetensors files
to `MOLD_HOME/cache/single-file/<sha>/...` and pointing the existing
`SD15Engine::new(model_name, paths, ...)` at them. This is one-time-cost
per checkpoint (cached by content hash), but trades disk for VRAM
simplicity. **Only do this if Shape A is materially harder.**

### Engine surface

Either shape adds **one** new constructor on `SD15Engine`:

```rust
impl SD15Engine {
    /// Construct from a Civitai single-file checkpoint. Internally calls
    /// `loader::single_file::load(path, Family::Sd15)` and applies the
    /// SD1.5 rename rules to bridge A1111 → diffusers naming.
    pub fn from_single_file(
        model_name: String,
        single_file_path: PathBuf,
        clip_tokenizer: PathBuf,        // tokenizer is companion-pulled (2.7), not in the SF
        scheduler: Scheduler,
        load_strategy: LoadStrategy,
        gpu_ordinal: usize,
    ) -> Result<Self>;
}
```

The existing `pub fn new(...)` stays — diffusers-layout (HF) checkpoints
keep their current code path. Factory routing (2.6) decides which
constructor to call based on `Bundling`.

## TDD shape

Three rounds of red-green-refactor, smallest to largest:

### Round 1 — rename-rule unit tests (pure data)

No I/O, no candle, no engine. Just exercise the rename function:

- `rename_unet_input_block_0_to_diffusers` — `model.diffusion_model.input_blocks.0.0.weight` →
  `down_blocks.0.resnets.0.in_layers.0.weight` *(or whichever diffusers
  key candle expects — verify against
  `candle_transformers::models::stable_diffusion::unet_2d`)*
- `rename_unet_middle_block_attention` — confirm `middle_block.1.transformer_blocks.0.…` →
  `mid_block.attentions.0.transformer_blocks.0.…`
- `rename_unet_output_block_with_upsampler` — output_blocks have a quirk where the upsampler
  lives at the end of the block; confirm A1111 `output_blocks.2.2.conv.weight` lands at the
  diffusers `up_blocks.0.upsamplers.0.conv.weight`.
- `rename_vae_encoder_down_block` — `first_stage_model.encoder.down.0.block.0.norm1.weight` →
  diffusers `encoder.down_blocks.0.resnets.0.norm1.weight`.
- `rename_clip_l_text_model_layer_attn` — `cond_stage_model.transformer.text_model.encoder.layers.0.self_attn.q_proj.weight` →
  diffusers `text_model.encoder.layers.0.self_attn.q_proj.weight` *(this
  one is mostly a strip prefix)*.

These tests can use synthetic key strings — no safetensors fixture
needed.

### Round 2 — `build_sd15_remap` integration

One test using a synthesised tiny safetensors (same trick as 2.3 — see
`crates/mold-inference/src/loader/single_file.rs` for the pattern) with
3-5 representative keys per component. Assert:

- `remap.unet.contains_key("down_blocks.0.resnets.0.in_layers.0.weight")`
- `remap.unet["down_blocks.0.resnets.0.in_layers.0.weight"] == "model.diffusion_model.input_blocks.0.0.weight"`
- `remap.vae.contains_key("encoder.down_blocks.0.resnets.0.norm1.weight")`
- `remap.clip_l.contains_key("text_model.encoder.layers.0.self_attn.q_proj.weight")`
- An A1111 key with no rename rule shows up in a `RemapError::UnknownKey`
  or in a `remap.unmapped: Vec<String>` (decide which based on the audit's
  "stray tensors must be tolerated" rule — `denoiser.sigmas` was already
  filtered into `unknown_keys` by 2.3, so anything reaching the remap
  should map; logging is enough).

### Round 3 — engine smoke test (skip if Shape A only)

If you went with Shape A: a test that calls
`SD15Engine::from_single_file(...)` against a synthesised tiny
checkpoint and asserts construction succeeds without panicking. Don't
run inference — that needs real weights and a GPU.

If you went with Shape B (temp-dir extraction): test that the cache
directory ends up with the three expected files (`unet.safetensors`,
`vae.safetensors`, `text_encoder.safetensors`) with the expected
renamed key names.

## Out of scope for 2.4

- **No SDXL.** Dual CLIP-L + CLIP-G renames are 2.5. Yes, the inner
  block naming below `down_blocks/mid_block/up_blocks` is shared, so
  factor the helpers wisely — but 2.4 ships the SD15 path only.
- **No factory routing.** Factory wiring (`create_engine` choosing
  `from_single_file` vs `new`) is 2.6.
- **No companion auto-pull.** The CLIP-L tokenizer (and CLIP-L weight
  for SDXL where applicable) is *not* inside the single file. 2.7 owns
  pulling those companions from HF; 2.4 just takes their paths as
  constructor args.
- **No CLI plumbing.** `mold run --single-file foo.safetensors` is 2.8.

## Working conventions to preserve

- TDD — failing tests first per round (especially Round 1, where the
  rename rules are pure data and trivially testable).
- One scope per commit. 2.4 is one commit:
  `feat(inference): SD1.5 single-file engine constructor (phase 2.4)`.
- **Phase 2 lands as one push when 2.10 is gate-green.** This commit
  stays local. 2.3's `a6991b4` is also local-only.
- `superpowers:test-driven-development` is still the right skill.
- `superpowers:verification-before-completion` before declaring 2.4
  done — gate must be green:
  `cargo fmt --all -- --check && cargo clippy --workspace --all-targets -- -D warnings && cargo test --workspace`.

## Verification commands

```bash
cd /Users/jeffreydilley/github/mold

# Pre-flight
git status                                      # clean
git log --oneline origin/main..HEAD | head -10  # 7 phase-1+2 commits ahead, including a6991b4 (local)

# After implementation
cargo fmt --all -- --check
cargo clippy --workspace --all-targets -- -D warnings
cargo test -p mold-ai-inference --lib
cargo test --workspace                          # retry once on TUI flake before blaming your changes
```

The TUI theme test flake
(`theme_save_then_load_round_trip_preserves_preset`) is documented in
user memory — retry once before blaming your changes. It did **not**
flake during 2.3 verification (2565/2565 passed first time), but the
race is real.

---

## The prompt

Paste from here into a fresh Claude Code session:

---

I'm starting **task 2.4** of the mold catalog-expansion phase 2 — the
SD1.5 engine constructor that consumes the `SingleFileBundle` produced
by 2.3 and bridges A1111 → diffusers key naming so candle's existing
SD15 model can load Civitai-format checkpoints. Tasks 2.1 (pre-flight),
2.2 (tensor-prefix audit), and 2.3 (single-file dispatcher, commit
`a6991b4`, **local-only**) are done.

## Read first, in this order

1. `~/.claude-personal/CLAUDE.md` and `/Users/jeffreydilley/github/mold/CLAUDE.md` — coding conventions. Note the LoRA-via-custom-VarBuilder pattern under "Key Design Decisions §12" — that pattern is the precedent for Shape A in this task.
2. `tasks/catalog-expansion-phase-2-handoff.md` — overall phase-2 brief. **§ "Task list" 2.4 row** is the canonical spec.
3. `tasks/catalog-expansion-phase-2-tensor-audit.md` — anchors which prefixes the SD1.5 rename rules must target. Skim the depth-2 prefix dumps for DreamShaper 8 / SD 1.5.
4. `tasks/catalog-expansion-phase-2-task-2.4-handoff.md` — this file. Has the two design shapes (A: VarBuilder remap, B: temp-dir extraction), the TDD round shape, and the rename-rule examples to spot-check against candle's `stable_diffusion::unet_2d`.
5. `crates/mold-inference/src/loader/single_file.rs` — the 2.3 module you're extending. Note the `read_tensor_keys` header-only parse pattern and the synthetic-fixture testing trick used in `tests::write_fixture` — reuse both.
6. `crates/mold-inference/src/sd15/pipeline.rs` — current `SD15Engine::new(...)` constructor; 2.4 adds a sibling `from_single_file(...)` constructor.
7. `crates/mold-inference/src/flux/lora.rs` — the `LoraBackend` `SimpleBackend` impl is the precedent for Shape A's "intercepting VarBuilder" approach. Read its `get(...)` flow.

## What you're doing

Implement task 2.4 per the handoff. Do a five-minute spike first to
confirm whether Shape A (VarBuilder remap, no temp files) is feasible
against candle's `stable_diffusion::build_unet` / `build_vae` /
`build_clip_transformer` — if their internals call introspective
methods on the VarBuilder backend that a thin remap can't fake, fall
back to Shape B (temp-dir extraction, content-hashed cache). State
which shape you picked and why before writing implementation code.

TDD: three rounds, failing tests first per round.

- **Round 1 (rename rules, pure data):** 5 unit tests on
  `apply_sd15_*_rename(...)` helpers, no I/O.
- **Round 2 (`build_sd15_remap` integration):** one test using a
  synthesised tiny safetensors (reuse the `write_fixture` pattern from
  `loader/single_file.rs::tests`).
- **Round 3 (engine smoke test):** Shape A → assert
  `SD15Engine::from_single_file(...)` constructs without panicking
  against a synthetic checkpoint. Shape B → assert the cache dir gets
  the three expected per-component safetensors.

**No SDXL, no factory routing, no companion auto-pull, no CLI plumbing
— those are 2.5 / 2.6 / 2.7 / 2.8 respectively.**

## How to work

1. Pre-flight: confirm `git status` clean and `cargo test -p mold-ai-inference --lib` green (615 tests should pass) before touching code.
2. Spike candle's `build_unet` / `build_vae` / `build_clip_transformer` (5 minutes max) to pick Shape A vs B. **State the choice before writing code.**
3. Use `superpowers:test-driven-development`. Round 1 first — rename rules are pure data, trivially testable, and surface the diffusers-key-naming questions early.
4. The rename rules are the load-bearing part. Cross-check at least 5 representative A1111 keys against the corresponding `candle_transformers::models::stable_diffusion::unet_2d::UNet2DConditionModel` field names by `git grep`-ing into the candle crate (it's at `~/.cargo/registry/src/index.crates.io-*/candle-transformers-mold-0.9.10/`). Don't trust diffusers documentation alone — candle may have its own quirks.
5. New module is `crates/mold-inference/src/loader/sd15_keys.rs` (sibling of `single_file.rs`, exported via `loader/mod.rs`). The engine constructor goes in `crates/mold-inference/src/sd15/pipeline.rs` next to the existing `pub fn new(...)`.

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
feat(inference): SD1.5 single-file engine constructor (phase 2.4)

Adds `SD15Engine::from_single_file(...)` consuming the
`SingleFileBundle` produced by phase 2.3 and applying the SD1.5
A1111→diffusers rename pass so candle's existing `build_unet` /
`build_vae` / `build_clip_transformer` machinery can materialise
tensors from a Civitai single-file checkpoint. SDXL gets the same
treatment in 2.5.

[Note here whether you picked Shape A (VarBuilder remap) or
Shape B (temp-dir extraction) and one-line why.]

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
```

**Do not push.** Phase 2 lands as one push at 2.10. `a6991b4` (the 2.3
commit) is also local-only.

## If you hit a surprise

If candle's `build_unet` introspects the VarBuilder in a way the remap
can't fake (Shape A blocked); or the rename rules turn out to need
sub-block-level disambiguation (e.g. attention vs resnet position
within a `down_blocks.X`); or a Civitai checkpoint surfaces a key the
audit didn't anticipate — **stop, document the surprise, ask before
pressing forward.** Prefer a clear note in the handoff over inventing
a workaround that diverges from the audit's contract.

When 2.4 is gate-green and the three test rounds pass, write
`tasks/catalog-expansion-phase-2-task-2.5-handoff.md` (template: this
file) and stop. Do not start 2.5 in the same session.
