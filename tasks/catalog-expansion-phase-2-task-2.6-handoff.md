# Phase-2 task 2.6 — Engine `create_engine` factory routing (kickoff handoff)

> Paste the prompt at the bottom of this file into a fresh Claude Code
> session to start task 2.6. Everything above is reference material for
> the author / for skimming.

## Where phase 2 stands on entry

Branch `feat/catalog-expansion`. The five most recent commits are
**local-only** (phase 2 lands as one push at 2.10):

| Commit | Origin? | Scope |
|---|---|---|
| `a579a0c` | local | feat(inference): SDXL single-file engine constructor (phase 2.5) |
| `9e00bd3` | yes | docs(tasks): catalog-expansion phase-2 task 2.5 kickoff handoff |
| `f87687b` | local | feat(inference): SD1.5 single-file engine constructor (phase 2.4) |
| `e50128f` | yes | docs(tasks): catalog-expansion phase-2 task 2.4 kickoff handoff |
| `a6991b4` | local | feat(inference): single-file checkpoint dispatcher (phase 2.3) |
| `e7d4f4a` | yes | docs(tasks): catalog-expansion phase-2 task 2.3 kickoff handoff |
| `4970a92` | yes | docs(tasks): SD1.5 + SDXL tensor-prefix audit findings (phase 2.2) |
| `cb13e06` | yes | feat(inference): sd_singlefile_inspect dev-bin for tensor-prefix audit |

`tasks/catalog-expansion-phase-2-handoff.md` is your task-list source of
truth. The 2.4 + 2.5 modules
(`crates/mold-inference/src/loader/{sd15_keys,sdxl_keys,vae_keys}.rs`
plus `SD15Engine::from_single_file` / `SDXLEngine::from_single_file`)
give 2.6 everything it needs to wire the factory.

### Done

- **2.1 pre-flight** — same state as 2.3-2.5 handoffs.
- **2.2 tensor-prefix audit** — finished (`cb13e06` / `4970a92`).
- **2.3 single-file dispatcher** — landed at `a6991b4`. Header-parses
  `.safetensors` and partitions keys into UNet / VAE / CLIP-L / CLIP-G /
  unknown. Family-gated: only `Family::Sd15` and `Family::Sdxl` succeed.
- **2.4 SD1.5 engine constructor** — landed at `f87687b`.
  `SD15Engine::from_single_file(model_name, single_file_path,
  clip_tokenizer, scheduler, load_strategy, gpu_ordinal) -> Result<Self>`.
- **2.5 SDXL engine constructor** — landed at `a579a0c`.
  `SDXLEngine::from_single_file(model_name, single_file_path,
  clip_l_tokenizer, clip_g_tokenizer, scheduler, load_strategy,
  gpu_ordinal) -> Result<Self>`. Defaults `is_turbo = false` — 2.6's
  factory routing is the right place to flip that based on manifest
  metadata. New `loader/sdxl_keys.rs` module exposes
  `apply_sdxl_*_rename` / `RenameOutput::{Direct, FusedSlice}` /
  `SdxlRemap` / `build_sdxl_remap`. VAE rename refactored to a shared
  `loader/vae_keys.rs::apply_vae_rename` (audit point 3 — VAE keys are
  identical between SD1.5 and SDXL).

### Not yet done

- 2.6 — **this handoff's task**.
- 2.7 server companion auto-pull, 2.8 CLI integration, 2.9 web gate
  flip, 2.10 UAT — all downstream of 2.6.

## What 2.5 leaves on the table for 2.6

Both `from_single_file` constructors stash the checkpoint path on a new
`pub(crate) single_file_path: Option<PathBuf>` field. Today that field
is **read only by tests** — the production `load()` paths don't consult
it yet (the future custom `SimpleBackend` that bridges
`Sd15Remap`/`SdxlRemap` to candle's per-component constructors lands
when 2.6 wires the factory or 2.10 UAT runs the first real Civitai
checkpoint end-to-end).

The 2.6 task has two distinguishable sub-questions:

1. **Routing question.** When `create_engine` is called for an SD1.5 or
   SDXL model, how does it decide between `SD15Engine::new(...)` /
   `SDXLEngine::new(...)` (today's diffusers-layout path) and
   `from_single_file(...)` (Civitai single-file path)?
2. **Materialisation question.** Once `from_single_file(...)` is the
   chosen path, what does `load()` actually do? Today the `from_single_file`
   constructors validate the rename surface but do not build the model
   — that intentional gap from 2.4/2.5 needs filling for 2.6 if (and
   only if) 2.10's UAT requires it.

The 2.10 UAT will pull a real Pony / Juggernaut XL / DreamShaper 8
checkpoint and try to generate. So the materialisation question is on
the critical path and 2.6 must answer it — at minimum with a stub that
returns a meaningful error so the failure mode is "your single-file
path isn't materialising weights yet" rather than "your single-file
path falls through to a diffusers loader that crashes on a missing
config.json".

## What 2.6 produces

Two concrete deliverables:

### A. Factory routing

Update `crates/mold-inference/src/factory.rs::create_engine_with_pool`
so the SD1.5 / SDXL match arms detect single-file checkpoints and call
`from_single_file(...)` instead of `new(...)`.

The cleanest detection signal is **the file extension on
`paths.transformer`** — when `transformer` ends in `.safetensors` and
no separate `vae`/`clip_encoder` paths are supplied (or they all point
at the same file), it's a single-file checkpoint. Alternatives:

- (1) New field on `ModelConfig`: `pub single_file: Option<PathBuf>`.
  Pro: explicit. Con: another config knob that has to be wired through
  every call site.
- (2) New `SourceMode` enum threaded into `create_engine`. Pro:
  type-safe. Con: every caller has to pick a variant.
- (3) Catalog row consultation from inside `create_engine`. Pro:
  authoritative. Con: introduces a DB read into a synchronous hot path.

**Recommendation:** sniff `paths.transformer.extension() == "safetensors"`
plus `paths.transformer == paths.vae` (or `paths.vae` is unset) as the
trigger. Civitai single-file checkpoints have always been a single
`.safetensors`; diffusers checkpoints are a directory with
`unet/diffusion_pytorch_model.safetensors` etc. The extension+identity
check is unambiguous and needs no schema churn.

For **`is_turbo`**, the factory already infers it (`model_cfg.is_turbo
.or(model_name.contains("turbo"))` at `factory.rs:97-100`). Thread
that into `from_single_file` as a new constructor arg — see § B below.

For **scheduler**, the factory already picks a default per family
(`model_cfg.scheduler.unwrap_or(Scheduler::Ddim)` etc.). Same threading.

### B. `is_turbo` arg on `SDXLEngine::from_single_file`

The 2.5 constructor defaulted `is_turbo = false` because the spec
didn't list it. Now that 2.6 has the manifest in hand, surface it as a
constructor arg so SDXL Turbo models get the right `VAE_SCALE_TURBO`
+ `EulerAncestral` defaults at load time. This pushes the constructor
from 7 args to 8 — clippy's `too_many_arguments` lints at >7, so add
`#[allow(clippy::too_many_arguments)]` on the impl with a one-line
rationale (or refactor into a small `SdxlSingleFileArgs` struct, your
call). The clippy suppression is the lower-friction option and matches
how candle itself handles wide constructors.

### C. `load()` materialisation (the load-bearing piece)

This is the actual unblock for 2.10's UAT. The
`from_single_file`-constructed engine must successfully `load()` and
`generate()` against a real Civitai checkpoint.

Two viable shapes:

- **Custom `SimpleBackend` per component** (the path 2.4 + 2.5 set
  up). Mirror `crates/mold-inference/src/flux/lora.rs::LoraBackend` —
  a `candle_nn::SimpleBackend` impl that wraps an `Mmap` + the
  `Sd15Remap` / `SdxlRemap` and translates each `vb.get(name)` into a
  source-tensor read (slicing for CLIP-G `FusedSlice`). Plug into
  candle's per-component `*::new(vb, ...)` constructors via a
  `VarBuilder::from_backend(...)`.
- **Pre-extract to a temporary diffusers layout.** Read the
  single-file checkpoint, write four diffusers-layout `.safetensors`
  to a scratch dir keyed by file hash, point existing `load()` at
  those. Pro: zero new code in the engines. Con: doubles disk I/O
  + scratch space; the cache key has to be invalidated on file mtime
  change.

**Recommendation:** custom `SimpleBackend`. The 2.4/2.5 design is
pointed straight at it (the `Sd15Remap` / `SdxlRemap` shape is
deliberately backend-friendly; the `RenameOutput::FusedSlice` axis +
component fields exist for exactly this slicing). The pre-extract
approach throws away the lazy-loading invariant that 2.3-2.5
preserved.

The slice math for CLIP-G `FusedSlice` mirrors
`crates/mold-inference/src/flux/lora.rs::fused_slice_range` — for
`num_components: 3, axis: 0, component: c`, the slice is
`base_rows / 3` rows starting at `c * base_rows / 3`.

### TDD shape

Three rounds, mirroring 2.4 + 2.5:

#### Round 1 — factory detection unit tests

- `factory_routes_sd15_diffusers_to_new_constructor` — when
  `paths.transformer` is a directory's `safetensors` shard *and*
  `paths.vae` is a separate path, route to `SD15Engine::new`.
- `factory_routes_sd15_single_file_to_from_single_file` — when
  `paths.transformer == paths.vae` and ends `.safetensors`, route to
  `SD15Engine::from_single_file`.
- Same pair for SDXL (with `clip_encoder` / `clip_encoder_2` identity
  check on top).
- `factory_threads_sdxl_is_turbo_into_constructor` — verify a model
  config with `is_turbo = true` lands in the constructor.

These tests need `from_single_file` to be invokable from the factory,
which means surfacing it on the engine's pub API (already done in
2.4/2.5).

#### Round 2 — `SimpleBackend` unit tests

- `sd15_simple_backend_resolves_diffusers_key_to_a1111_tensor` —
  build a synthetic 2-key safetensors with `model.diffusion_model.input_blocks.0.0.weight`,
  index via the backend with the diffusers key `conv_in.weight`, get
  the same tensor data back.
- `sdxl_simple_backend_slices_clip_g_fused_qkv` — synthesise a
  `[3*d, d]` fused weight, request the `q_proj` diffusers key, assert
  the returned tensor is the first `d` rows.
- Same for the bias slab.
- `simple_backend_unmapped_key_returns_error` — defensive.

#### Round 3 — engine `load()` smoke test

`SD15Engine::from_single_file(...).load()` against a synthetic
checkpoint must successfully construct UNet + VAE + CLIP-L without
panicking. This requires the synthetic to have **real-shape tensors**,
not 1-element placeholders — see `crates/mold-inference/src/sd15/`
for what shapes candle expects. Expect this to be the most expensive
test; a 1024×1024-resolution UNet with real shapes is too big for
in-process synthesis. Cap the test at the smallest dimensions candle
will accept (`down_blocks.0.resnets.0.norm1` is shape `[block_out_channels[0]]`,
i.e. `[320]` — synthesisable).

If real-shape synthesis blows the test budget, demote round 3 to a
`#[ignore]` end-to-end gated on the killswitch UAT in 2.10 and call
out the gap in the commit.

## Out of scope for 2.6

- **CLI plumbing.** `mold pull cv:<id>` recipe path with companion
  auto-pull is 2.7/2.8.
- **Server companion enqueue.** Wiring `companions: ["clip-l",
  "sdxl-vae", ...]` into `DownloadQueue::enqueue` is 2.7.
- **Web gate flip.** Toggling `cat.canDownload(entry)` from `engine_phase
  === 1` to `engine_phase <= 2` is 2.9.
- **End-to-end UAT.** Running real Pony / Juggernaut / DreamShaper
  generations through the wired-up factory is 2.10 — but 2.6 should
  produce something that *can* be UAT'd in 2.10.

## Working conventions to preserve

- TDD — failing tests first per round (3 RED-then-GREEN cycles).
- One scope per commit. 2.6 is one commit:
  `feat(inference): single-file factory routing + load() (phase 2.6)`.
- **Phase 2 lands as one push when 2.10 is gate-green.** This commit
  stays local. `a6991b4`, `f87687b`, `a579a0c` are also local-only.
- `superpowers:test-driven-development` for the rounds,
  `superpowers:verification-before-completion` before declaring done.
- **Pre-grep cross-crate before subagent dispatch.** `create_engine`
  has callers in `mold-server`, `mold-cli`, `mold-tui`, etc. —
  enumerate them with `grep -rn 'create_engine(' crates/` before
  changing the signature, or the workspace test will surface ripples.

## Verification commands

```bash
cd /Users/jeffreydilley/github/mold

# Pre-flight
git status                                      # clean
git log --oneline origin/main..HEAD | head -10  # 9 phase-1+2 commits ahead, including a6991b4 + f87687b + a579a0c (local)

# After implementation
cargo fmt --all -- --check
cargo clippy --workspace --all-targets -- -D warnings
cargo test -p mold-ai-inference --lib
cargo test --workspace                          # retry once on TUI flake before blaming your changes
```

The TUI theme test flake
(`theme_save_then_load_round_trip_preserves_preset`) is documented in
user memory — retry once before blaming your changes. It did **not**
flake during 2.3 / 2.4 / 2.5 verification.

## Refactor opportunity

`create_engine_with_pool`'s SD1.5 + SDXL match arms will now have two
sub-paths each (diffusers-layout `new` vs single-file `from_single_file`).
Resist the urge to abstract this into a `SourceMode` enum *threaded
into the function arg list* — the routing decision is purely a function
of `paths.{transformer,vae,clip_encoder,clip_encoder_2}` and lives at
exactly one site (the factory). A small private helper
`fn is_single_file(paths: &ModelPaths) -> bool` keeps the conditional
local. If both engines grow more single-file logic in the future,
revisit then.

## Reference reading

When the SimpleBackend implementation makes the keys puzzle real,
ComfyUI is checked out at `~/github/ComfyUI` — its
`comfy/sd1_clip.py`, `comfy/sdxl_clip.py`, `comfy/model_detection.py`,
`comfy/utils.py::state_dict_prefix_replace`, and especially
`comfy/sd.py::load_checkpoint_guess_config` are the production-grade
reference for how Civitai checkpoints actually behave under load. Read
for protocol; don't copy code (license + style mismatch).

---

## The prompt

Paste from here into a fresh Claude Code session:

---

I'm starting **task 2.6** of the mold catalog-expansion phase 2 — the
engine factory routing that picks `from_single_file` over `new` for
Civitai single-file SD1.5 / SDXL checkpoints, plus the `load()`
materialisation that bridges the `Sd15Remap` / `SdxlRemap` produced by
2.4 / 2.5 to candle's per-component constructors via a custom
`SimpleBackend`. Tasks 2.1–2.5 are done (2.5 commit `a579a0c`,
**local-only**, ships the SDXL `from_single_file` constructor with the
CLIP-G fused QKV split that 2.6 must materialise).

## Read first, in this order

1. `~/.claude-personal/CLAUDE.md` and `/Users/jeffreydilley/github/mold/CLAUDE.md` — coding conventions. Note the LoRA-via-custom-VarBuilder pattern under "Key Design Decisions §12" — that's the precedent for the `SimpleBackend` this task introduces.
2. `tasks/catalog-expansion-phase-2-handoff.md` — overall phase-2 brief. **§ "Task list" 2.6 row** is the canonical spec.
3. `tasks/catalog-expansion-phase-2-task-2.6-handoff.md` — this file. Spells out the routing detection (sniff `paths.transformer == paths.vae` + `.safetensors`), the `is_turbo` threading question, and the load-bearing `SimpleBackend` materialisation that 2.10 UAT depends on.
4. `crates/mold-inference/src/loader/{sd15_keys,sdxl_keys,vae_keys}.rs` — the rename modules from 2.4 + 2.5. The `Sd15Remap` and `SdxlRemap` outputs are what the new `SimpleBackend` indexes by.
5. `crates/mold-inference/src/loader/single_file.rs` — the 2.3 dispatcher.
6. `crates/mold-inference/src/sd15/pipeline.rs` and `crates/mold-inference/src/sdxl/pipeline.rs` — the existing `from_single_file` constructors that 2.6 wires into `create_engine`. Both stash a `pub(crate) single_file_path: Option<PathBuf>` that load() will branch on.
7. `crates/mold-inference/src/factory.rs` — the routing site. SD1.5 + SDXL match arms at `factory.rs:87-110`.
8. `crates/mold-inference/src/flux/lora.rs::LoraBackend` (lines 431-556) — **the** `SimpleBackend` precedent. Read carefully — it's the closest mold has to a "load-tensors-from-an-mmap-with-key-rewriting" backend, and the slice math at `fused_slice_range` is what CLIP-G's `FusedSlice` needs.
9. (Reference, do not edit) candle stable_diffusion sources at `~/.cargo/registry/src/index.crates.io-*/candle-transformers-mold-0.9.10/src/models/stable_diffusion/{unet_2d.rs, vae.rs, clip.rs}` — the `vs.pp("…")` chains are the diffusers keys the new SimpleBackend must answer.
10. `~/github/ComfyUI` — broad reference for how Civitai checkpoints behave under real loads. `comfy/sd.py::load_checkpoint_guess_config` is the production-grade analog of what this task's `load()` is doing. Read for protocol, don't copy code.

## What you're doing

Implement task 2.6 per this handoff. Three deliverables:

1. **Factory routing** — `create_engine_with_pool` detects single-file SD1.5 / SDXL via `paths.transformer == paths.vae && paths.transformer.extension() == "safetensors"` and dispatches to `from_single_file` instead of `new`. Thread `is_turbo` and `scheduler` through.
2. **`is_turbo` constructor arg on `SDXLEngine::from_single_file`** — adds the 8th arg; suppress `clippy::too_many_arguments` with a one-liner.
3. **Custom `SimpleBackend` + `load()` wiring** — SD15/SDXL engines branch on `single_file_path.is_some()` in `load()`, build a `SimpleBackend` over the mmap'd checkpoint that translates diffusers `vb.get(name)` calls through `Sd15Remap` / `SdxlRemap`, slicing CLIP-G `FusedSlice` entries via the same arithmetic as `flux/lora.rs::fused_slice_range`. Plug into candle's per-component constructors.

TDD: three rounds, failing tests first per round, mirroring 2.4 + 2.5.

- **Round 1 (factory detection):** 4-5 unit tests on `create_engine_with_pool` routing.
- **Round 2 (`SimpleBackend`):** 4-5 unit tests including the CLIP-G fused-QKV slice case.
- **Round 3 (engine `load()` smoke):** `SD15Engine::from_single_file(...).load()` against a synthetic checkpoint with real-shape tensors; if real-shape synthesis is too expensive, demote to `#[ignore]` and document the UAT-only verification path.

## How to work

1. Pre-flight: confirm `git status` clean and `cargo test -p mold-ai-inference --lib` green (644 tests pass) before touching code.
2. **Pre-grep cross-crate.** `grep -rn 'create_engine(' crates/` to enumerate callers — `create_engine` has callers in `mold-server`, `mold-cli`, `mold-tui` and the test ripples will surface in `cargo test --workspace`. The signature change in deliverable 2 (8 args on `SDXLEngine::from_single_file`) is internal, but the factory routing change (deliverable 1) may surface a call-site update if any tests construct paths inline.
3. Use `superpowers:test-driven-development`. Round 1 first — factory detection is pure-data, trivially testable, and surfaces the routing-trigger questions early.
4. For Round 2, the `SimpleBackend` math has the closest precedent in `flux/lora.rs:431-556` — read it carefully. The `fused_slice_range` helper at `flux/lora.rs:237` is the slice arithmetic; the `LoraBackend::get` impl is the per-key dispatch.
5. For Round 3, expect candle's `UNet2DConditionModel::new` to be the toughest customer. Synthetic shapes have to match candle's `BlockConfig` exactly — anything off and the constructor panics deep in matrix-shape validation. If you hit that wall, demote and document.

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
feat(inference): single-file factory routing + load() (phase 2.6)

Wires `create_engine_with_pool` to detect Civitai single-file
SD1.5/SDXL checkpoints (`paths.transformer == paths.vae && ends
.safetensors`) and dispatch to `from_single_file` instead of
`new`. Adds an `is_turbo` constructor arg to
`SDXLEngine::from_single_file` so SDXL Turbo gets the right
scheduler + VAE scale. Adds a custom `SimpleBackend` that translates
diffusers `vb.get(name)` calls into mmap'd Civitai-format reads via
the `Sd15Remap` / `SdxlRemap` lookup tables produced by 2.4 / 2.5,
including row-wise slicing for the CLIP-G `attn.in_proj_*` fused
QKV slabs. SD15/SDXL `load()` paths now branch on
`single_file_path.is_some()` and use the custom backend instead of
candle's diffusers-layout `build_*` helpers when the engine was
constructed via `from_single_file`.

[Note here whether Round 3's load() test runs real-shape or was
demoted to #[ignore], and one line why.]

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
```

**Do not push.** Phase 2 lands as one push at 2.10. The 2.3
(`a6991b4`), 2.4 (`f87687b`), and 2.5 (`a579a0c`) commits are also
local-only.

## If you hit a surprise

If candle's `UNet2DConditionModel::new` rejects synthetic shapes that
the `Sd15Remap` accepts; or the CLIP-G fused QKV split needs a
different axis or component layout than the audit predicted; or a
real Civitai checkpoint surfaces a key that `apply_sdxl_unet_rename`
doesn't anticipate (input_blocks.9+, label_emb beyond .0.{0,2}, …) —
**stop, document the surprise in the handoff, ask before pressing
forward.** Prefer a clear note over inventing a workaround that
diverges from the audit's contract.

When 2.6 is gate-green and the three test rounds pass, write
`tasks/catalog-expansion-phase-2-task-2.7-handoff.md` (template: this
file) and stop. Do not start 2.7 in the same session.
