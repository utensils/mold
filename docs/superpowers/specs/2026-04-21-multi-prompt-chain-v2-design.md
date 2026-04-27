# Multi-prompt chain authoring — design spec

- **Status:** Approved for implementation planning
- **Date:** 2026-04-21
- **Author:** Jeff Dilley + Claude (Opus 4.7)
- **Supersedes:** None (follows `tasks/render-chain-v1-plan.md`)
- **Sub-project scope:** A of a four-part decomposition (A now; B, C, D to follow)

## 1. Problem & goals

Render-chain v1 (`49ef35e`) shipped a stages-based wire format and a working
LTX-2 distilled orchestrator, but the user-facing surfaces only author a
single prompt that gets replicated across all stages. For any video longer
than one clip, that means the whole arc is driven by one prompt and every
transition is a visual morph.

We want to let users **direct any-length video, scene by scene, on all
surfaces**: each prompt becomes its own stage, each stage can specify a
transition mode (smooth morph, hard cut, or crossfade), and the whole chain
is authorable as a version-controllable TOML script that round-trips between
CLI, TUI, and web.

Success criteria for sub-project A:

1. A user can write a `shot.toml` with N prompts and `mold run --script
   shot.toml` renders a stitched video where each stage honors its own
   `prompt`, `frames`, and `transition`.
2. The web `/generate` composer has a "Script" mode with a card-per-stage
   editor, drag-reorder, per-stage expand, and bidirectional TOML
   import/export.
3. The TUI has a "Script" mode with a stage list + inline editor and TOML
   save/load.
4. v1 behavior is preserved byte-for-byte when `transition` is absent from
   the request and prompts are uniform.
5. The wire format has forward hooks for sub-projects B/C/D so they don't
   require another breaking change.

## 2. Decomposition: A, B, C, D

| # | Sub-project | What it unlocks | Depends on |
|---|---|---|---|
| **A** | **Multi-prompt chain authoring v2** (this spec) | "I can direct any-length video, scene by scene, with smooth/cut/fade transitions, on any surface." | Nothing. |
| B | Cross-stage identity carryover | LoRA stacking per stage, IC-LoRA, named reference library reused across stages. "My character/style/world stays consistent." | A (needs stages). |
| C | Per-stage model selection | Pick a model per stage; validate locked fields against each model's native shape; tier 1 (same-family LTX-2) only. "Use the cheap model for a beat, the heavy one for the money shot." | A (needs stages). |
| D | Chain-aware VRAM calculator | Pre-submit "this fits / doesn't fit" cue, worst-case stage analysis, multi-GPU + encoder-offload aware. "I know if I'm about to OOM before I press go." | A for chain-level cues; the inputs already exist in `mold-server/src/resources.rs`. |

Wire-format hooks baked into A so B/C/D ship without a new breaking change:

- `ChainStage.transition` — used by A.
- `ChainStage.fade_frames` — used by A.
- `ChainStage.model: Option<String>` — **rejected with 422 in A**, consumed by C.
- `ChainStage.loras: Vec<LoraSpec>` — **rejected with 422 in A**, consumed by B.
- `ChainStage.references: Vec<NamedRef>` — **rejected with 422 in A**, consumed by B.
- `ChainResponse.vram_estimate: Option<VramEstimate>` — response slot; `None` in A, populated by D.

## 3. Signed-off decisions (do not re-litigate)

Recorded here so the implementation plan can lock these in.

1. **Surfaces:** CLI + TUI + web, all three, designed together.
2. **Transition model:** Director's cut — per-stage `transition` enum with
   variants `smooth` (v1 behavior), `cut` (fresh latent, optional source
   image), `fade` (cut + post-stitch alpha blend). Stage 0's transition is
   ignored (nothing to transition from) and coerced to `smooth` at
   normalise with a warn.
3. **Per-prompt frame budget:** Fixed default with per-stage override. The
   default is the ceiling returned by `GET /api/capabilities/chain-limits`
   for the selected model; the UI's upper bound matches the ceiling
   (cannot be exceeded), the lower bound is `9` (smallest `8k+1`). Config
   file can pin a lower global preferred default; session overrides during
   authoring. For LTX-2 distilled, ceiling = 97 (model-hardcoded), margin
   inert.
4. **CLI input syntax:** TOML script file (`--script shot.toml`) is the
   canonical form. Sugar mode — repeated `--prompt` with uniform
   `--frames-per-clip` and uniform `--motion-tail` — covers trivial cases.
   Per-stage transitions or per-stage frames require a script file.
5. **Per-stage source image semantics (Option 2):** any stage can carry a
   `source_image`. `smooth` stages silently ignore it at the engine (warn
   logged; UI shows "ignored for smooth transition" hint). `cut` and
   `fade` stages use it as the i2v seed if present, else start from pure
   noise.
6. **Reserved fields reject with 422 in A.** Clients that populate `model`,
   `loras`, or `references` on a stage are misusing the wire format and
   must fail loud, not silently.
7. **Source images in TOML are relative to the script file.** Reader
   resolves them to absolute paths before submission. Supports git-friendly
   shot folders.
8. **Fail-closed stays.** Mid-chain failure → 502, discard all prior
   stages, no partial stitch. Error payload is richer (includes
   `failed_stage_idx`, `elapsed_stages`, `elapsed_ms`) so retry UX is
   actionable.
9. **Cancellation is stage-boundary only.** Mid-denoise interrupt is
   unsafe for GPU state. Worst-case wait = one stage's denoise time.
10. **Top-level locked fields:** `model`, `width`, `height`, `fps`,
    `output_format`, `seed`, `steps`, `guidance`, `strength`,
    `motion_tail_frames` stay on `ChainRequest`, not per-stage. Revisited
    when C lands (`model` moves to optional per-stage with top-level default).

## 4. Architecture

### 4.1 Wire format (`crates/mold-core/src/chain.rs`)

Additive changes to `ChainStage`:

```rust
pub struct ChainStage {
    // existing:
    pub prompt: String,
    pub frames: u32,
    pub source_image: Option<Vec<u8>>,
    pub negative_prompt: Option<String>,
    pub seed_offset: Option<u64>,

    // NEW in A:
    #[serde(default)]
    pub transition: TransitionMode,

    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub fade_frames: Option<u32>,

    // RESERVED (422 in A; consumed later):
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub model: Option<String>,           // C
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub loras: Vec<LoraSpec>,            // B
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub references: Vec<NamedRef>,      // B
}

#[derive(Default, Serialize, Deserialize, utoipa::ToSchema, ...)]
#[serde(rename_all = "snake_case")]
pub enum TransitionMode {
    #[default]
    Smooth,
    Cut,
    Fade,
}

// Placeholder shapes for reserved fields — defined so TOML parsing
// accepts well-formed scripts that populate them, but normalise
// rejects non-empty values in A.
pub struct LoraSpec {
    pub path: String,
    pub scale: f64,
    pub name: Option<String>,
}

pub struct NamedRef {
    pub name: String,
    pub image: Vec<u8>,  // base64 in JSON; relative path in TOML
}
```

`ChainResponse` gains two fields:

```rust
pub struct ChainResponse {
    // existing: video, stage_count, gpu
    pub script: ChainScript,                    // canonical echo
    pub vram_estimate: Option<VramEstimate>,    // None in A; D populates
}
```

`ChainScript` is a serialisation-oriented projection of the normalised
`ChainRequest` (same field set, no auto-expand sugar). Exists so the web
composer can populate "reload from this past render" without parsing the
request body.

### 4.2 `ChainRequest::normalise()` changes

Two new invariants:

- **Stage 0 transition coercion.** If `stages[0].transition != Smooth`,
  coerce to `Smooth` and `tracing::warn` ("stage 0 transition ignored").
  Rationale: we don't want scripts to fail lint when the user changes the
  transition on what *was* stage 1 and then reorders it to stage 0.
- **Reserved field rejection.** For every stage: if `model.is_some() ||
  !loras.is_empty() || !references.is_empty()`, return `422` with
  `field_path: "stages[N].<field>"` and a message pointing at the
  sub-project that will consume it (e.g., "per-stage model is reserved for
  sub-project C and not yet supported").
- **Smooth + source_image** passes normalise (the round-trip must survive),
  but the engine path logs a warn and ignores the image. The UI layer is
  responsible for showing the "ignored for smooth" hint before submit.
- **Existing invariants preserved:** `frames` is `8k+1`, stage count
  `<= MAX_CHAIN_STAGES`, `motion_tail_frames < stage.frames`.
- **New ceiling validation:** `stage.frames <= frames_per_clip_cap` for
  the selected model family. Previously only the `8k+1` constraint was
  checked; now we also enforce the per-model hard cap.

### 4.3 Capabilities endpoint

```
GET /api/capabilities/chain-limits?model=<name>
→ 200 {
    "model": "ltx-2-19b-distilled:fp8",
    "frames_per_clip_cap": 97,
    "frames_per_clip_recommended": 97,
    "max_stages": 16,
    "max_total_frames": 1552,
    "fade_frames_max": 32,
    "transition_modes": ["smooth", "cut", "fade"],
    "quantization_family": "fp8"
  }
```

Handler (new) lives alongside the existing `/api/capabilities` in
`mold-server/src/routes.rs`.

- `frames_per_clip_cap` = model-hardcoded per family (LTX-2 distilled = 97).
- `frames_per_clip_recommended` = `min(cap, hardware_derived_ceiling - 10% margin)`;
  hardware ceiling comes from `AppState.resources` free VRAM. For
  distilled, inert (cap is already the binding constraint).
- Response cached per-model for 30 s server-side so mode-change storms
  don't DoS the endpoint.
- Clients fetch on model-selection change, use the cap as the UI's upper
  bound and the recommended value as the initial default.

### 4.4 TOML script format (`mold.chain.v1`)

New module `crates/mold-core/src/chain_toml.rs` owns the
serialize/deserialize logic. Format:

```toml
schema = "mold.chain.v1"

[chain]
model = "ltx-2-19b-distilled:fp8"
width = 1216
height = 704
fps = 24
seed = 42
steps = 8
guidance = 3.0
strength = 1.0
motion_tail_frames = 25
output_format = "mp4"

[[stage]]
prompt = "a cat walks into the autumn forest"
frames = 97
source_image = "forest_entrance.png"   # relative to the script file

[[stage]]
prompt = "the forest opens to a clearing"
frames = 49
# transition defaults to smooth; field omitted

[[stage]]
prompt = "a spaceship lands"
frames = 97
transition = "cut"
source_image = "clearing.png"

[[stage]]
prompt = "the cat looks up in wonder"
frames = 97
transition = "fade"
fade_frames = 12
```

Round-trip contract: all three surfaces (CLI, TUI, web) must produce
bit-identical TOML for the same in-memory `ChainScript`. One round-trip
test per surface in `tests/`.

**Schema versioning:** `schema = "mold.chain.v1"` header. A missing header
defaults to v1. Future incompatible changes bump the version; the reader
refuses unknown versions with a clear error pointing at the mold version
that introduced them.

## 5. Surface UX

### 5.1 CLI

**Canonical:**

```bash
mold run --script shot.toml              # execute
mold run --script shot.toml --dry-run    # print normalised stages + total frames, exit
mold chain validate shot.toml            # parse + normalise without submitting
```

**Sugar** (trivial chains only — all `smooth`, uniform frames, no source images):

```bash
mold run ltx-2-19b-distilled:fp8 \
  --prompt "a cat walks into the autumn forest" \
  --prompt "the forest opens to a clearing" \
  --prompt "a spaceship lands" \
  --frames-per-clip 97 \
  --motion-tail 25
```

Sugar rules:

- Passing any `--prompt` flag routes to the chain endpoint. Single
  `--prompt` = single-stage chain (orchestrator handles this fine).
- Positional prompt argument is **disallowed when `--prompt` is used**
  (clap error — "use --prompt or a positional, not both").
- `--frames-per-clip` and `--motion-tail` apply uniformly to every stage.
- Per-stage transitions or per-stage frames are **not expressible in
  sugar**; clap error points the user at `--script`.

**Progress display** keeps the existing stacked-bar style, adds transition
tag in the per-stage line:

```
Chain: stage 3/5 ━━━━━━━━━━━━━━░░░░  60%
 → [stage 3/5 cut] "a spaceship lands"  ━━━━━━░░  6/8 steps
```

**Deliberately out of scope for A:** `--stage N` selective regen (C/v2.1),
`mold chain export <job-id>` (needs DB additions).

### 5.2 TUI

New mode: **Script**, reachable via `s` from the main TUI hub, or from the
composer when the selected model is chain-capable.

Layout:

```
┌─ Script ─────────────────────────────────────────────────────────┐
│ ▸ 1  smooth  97f  "a cat walks into the autumn forest"           │
│   2  smooth  49f  "the forest opens to a clearing"               │
│ ● 3  cut  🖼 97f  "a spaceship lands"                            │
│   4  fade  97f   "the cat looks up in wonder"                   │
│                                                                   │
│ ─ Editor: stage 3 ──────────────────────────────────────────────  │
│ Prompt:     a spaceship lands                                    │
│ Transition: [smooth] [▸cut] [fade]                               │
│ Frames:     97 (4.04s)                                           │
│ Source:     ✓ clearing.png                                       │
│                                                                   │
│ ─ Script total ─────────────────────────────────────────────────  │
│ 4 stages · 240 new frames (15 s) · ltx-2-19b-distilled · 1216×704│
└──────────────────────────────────────────────────────────────────┘
```

Keybindings:

| Key | Action |
|---|---|
| `j` / `k` | Navigate stage list |
| `a` / `A` | Add stage after current / at end |
| `d` | Delete current stage (confirm) |
| `J` / `K` | Move stage down / up |
| `t` | Cycle transition (smooth→cut→fade) |
| `i` | Edit prompt (opens modal textarea) |
| `f` | Edit frames (inline numeric; snaps to `8k+1`) |
| `p` | Attach/remove source image (file picker modal) |
| `⏎` | Submit chain |
| `Ctrl-S` | Save as TOML (prompts for path) |
| `Ctrl-O` | Load TOML (file picker) |
| `Esc` | Close editor, back to stage list |

Stage 0 hides the transition row (grayed with "n/a for first stage").

### 5.3 Web `/generate` composer

Mode toggle at the top: `◉ Single` / `◯ Script`. Single mode unchanged.
Script mode swaps the textarea for a card list.

```
┌──────────────────────────────────────────────────────────────┐
│ [ Single | ●Script ]       Export ▾  Import  Copy as command │
├──────────────────────────────────────────────────────────────┤
│ ┌──────────────────────────────────────────────────────────┐ │
│ │ ⋮⋮  1  Opening                                   [97 f ▾]│ │
│ │     a cat walks into the autumn forest                   │ │
│ │     🖼 forest_entrance.png   ✨ Expand   …more           │ │
│ └──────────────────────────────────────────────────────────┘ │
│ ┌──────────────────────────────────────────────────────────┐ │
│ │ ⋮⋮  2  [smooth|cut|fade]                         [49 f ▾]│ │
│ │     the forest opens to a clearing                       │ │
│ │     ✨ Expand   …more                                    │ │
│ └──────────────────────────────────────────────────────────┘ │
│ ┌──────────────────────────────────────────────────────────┐ │
│ │ ⋮⋮  3  [smooth|●cut|fade]  🖼 clearing.png       [97 f ▾]│ │
│ │     a spaceship lands                                    │ │
│ │     ✨ Expand   …more                                    │ │
│ └──────────────────────────────────────────────────────────┘ │
│                                                              │
│ [ + Add stage ]                                              │
│                                                              │
│ 3 stages · 243 new frames · ≈ 10.1s @ 24fps · ltx-2-19b      │
└──────────────────────────────────────────────────────────────┘
```

Per-card elements:

- Drag handle (`⋮⋮`) for reorder (vue-draggable-plus or similar).
- Stage index + transition selector (hidden on stage 0, three-state
  segmented control on others).
- Frame count dropdown: snapped to `8k+1`, clamped to
  `frames_per_clip_recommended`.
- Prompt textarea: auto-grows, Enter submits the whole chain, Shift+Enter
  newline.
- Optional source image chip (grayed with hover hint "ignored for smooth
  transition").
- Per-stage ✨ expand button (existing `ExpandModal`, writes back to this
  stage only).
- `…more` menu: Duplicate, Move up/down, Delete, Edit negative prompt,
  Edit seed offset.

Header actions:

- `Export ▾` → Save TOML / Copy TOML / Copy as CLI command / Copy as curl.
- `Import` → file picker (.toml).
- Mode toggle persists in `localStorage`.

Footer summary (live): stage count, total new frames after motion-tail trim,
estimated duration, model, resolution. Turns red with an explainer tooltip
when any cap is exceeded.

Persistence: draft state in `localStorage` keyed by `mold.chain.draft.v2`.
Survives refresh. Never auto-submit. Unsaved-changes indicator in the
header.

### 5.4 Per-stage prompt expansion

Each card's ✨ opens the existing `ExpandModal` scoped to that stage's
prompt only. No "expand all" in A. Story-to-script (one-line premise →
auto-generated stage list) is noted as a v2.1 follow-up.

## 6. Engine changes (`mold-inference`)

### 6.1 `Cut` transition

Orchestrator change only — no new engine code.

- `Ltx2ChainOrchestrator::run` inspects `stage.transition`. For `Cut`,
  it passes `carry: None` to `ChainStageRenderer::render` (same shape as
  stage 0 today).
- Renderer with `carry: None` + `source_image: Some(_)` → existing i2v
  path.
- Renderer with `carry: None` + `source_image: None` → existing t2v
  path (fresh noise).
- Motion-tail extraction after the cut stage still runs — the *next*
  stage (if smooth) can carry from it.

**Fade and subsequent motion-tail.** Fade is post-stitch and does not
alter the engine's latent output for the fade stage. Stage N+1's latents
are produced normally, so stage N+2 (if smooth) still carries from N+1's
motion-tail slice. In other words: transition affects the *incoming*
boundary only; outgoing motion-tail is always captured when the next
stage is smooth.

### 6.2 `Fade` transition

Engine = same as `Cut` (fresh latent, i2v if source image present). Plus a
new post-stitch step.

New helper in `crates/mold-inference/src/ltx2/media.rs`:

```rust
pub fn fade_boundary(
    tail: &[RgbImage],   // last fade_len frames of clip N
    head: &[RgbImage],   // first fade_len frames of clip N+1
    fade_len: u32,
) -> Vec<RgbImage>
```

Linear RGB interpolation per frame:
`out[i] = (1 - alpha) * tail[i] + alpha * head[i]`, `alpha = i / fade_len`.
Output length = `fade_len`.

Stitcher contract changes:

```rust
pub struct StitchPlan {
    pub clips: Vec<Vec<RgbImage>>,
    pub boundaries: Vec<TransitionMode>,  // len == clips.len() - 1
    pub fade_lens: Vec<u32>,              // len == clips.len() - 1
    pub motion_tail_frames: u32,
}
```

Stitch loop for each boundary `i`:

- `smooth`: drop leading `motion_tail_frames` of `clips[i+1]` (existing
  v1 behavior).
- `cut`: concatenate as-is.
- `fade`: take the trailing `fade_lens[i]` frames of `clips[i]` and the
  leading `fade_lens[i]` of `clips[i+1]`, blend with `fade_boundary`, emit
  one blended block instead of both boundary slices.

Frame-count accounting: fade consumes `fade_len` frames from both sides of
the boundary; a chain with a fade boundary emits
`sum(frames_i) - motion_tail_boundaries * motion_tail_frames - fade_boundaries * fade_len`
stitched frames. New helper `ChainRequest::estimated_total_frames()` returns
this; all three surfaces call it for the footer summary.

### 6.3 `source_image` on non-zero stages

Already supported structurally by the i2v code path — we just flip the
gate in `maybe_load_stage_video_conditioning` (and callers) to accept
source_image for any stage whose `carry` is `None` (i.e., stage 0 or any
`cut`/`fade` continuation).

For smooth continuations, the image is silently dropped (warn logged).
UI layer surfaces this pre-submit so the user isn't surprised.

### 6.4 Reserved-field enforcement

In `ChainRequest::normalise`, after the existing stage validation loop:

```rust
for (idx, stage) in self.stages.iter().enumerate() {
    if stage.model.is_some() {
        return Err(MoldError::Validation(format!(
            "stages[{idx}].model is reserved for sub-project C and not yet \
             supported in this mold version"
        )));
    }
    if !stage.loras.is_empty() {
        return Err(MoldError::Validation(format!(
            "stages[{idx}].loras is reserved for sub-project B"
        )));
    }
    if !stage.references.is_empty() {
        return Err(MoldError::Validation(format!(
            "stages[{idx}].references is reserved for sub-project B"
        )));
    }
}
```

These are hard `422`s. Silent acceptance would let v2.0 clients ship
scripts that work until C lands and then change behavior — the loud fail
is correct.

## 7. Error handling & cancellation

### 7.1 Fail-closed on stage failure

- Mid-chain failure → `502` response with structured body:
  ```json
  {
    "error": "stage render failed",
    "failed_stage_idx": 3,
    "elapsed_stages": 2,
    "elapsed_ms": 252000,
    "stage_error": "<inner error from engine>"
  }
  ```
- All prior stages are discarded (no partial stitch written to gallery).
- SSE stream: final frame is `event: error` with the same body.
- UI surfaces "Stage 3 of 5 failed after 4m 12s — try reducing its frames
  or switching to cut transition to drop carryover pressure."

### 7.2 Cancellation

- **Web/TUI:** "Cancel" button in RunningStrip. Fires `DELETE /api/jobs/<id>`
  or drops the SSE connection.
- **CLI:** `Ctrl-C` drops the SSE connection cleanly.
- **Server-side:** checked at **stage boundaries only**. Mid-denoise
  cancellation is unsafe for GPU state. Worst-case wait = one stage's
  denoise time (~30–60 s for LTX-2 distilled).
- **Canceled chain:** all partial output discarded; no gallery entry.

### 7.3 Validation UX

- Normalise is the single source of truth.
- Client-side mirrors (in `web/src/lib/chainRouting.ts` and equivalent
  CLI/TUI helpers) catch obvious cases for instant feedback but never
  substitute for the server check.
- Errors return `422` with `field_path: "stages[2].frames"` style pointers
  so UIs can highlight the offending card.

## 8. Testing strategy

| Layer | Scope | Key tests |
|---|---|---|
| `mold-core` | serde + normalise + TOML round-trip | `TransitionMode` serde; `ChainStage` serde with reserved fields; TOML read/write symmetry (including schema version); `ChainRequest::normalise` with every `transition × source_image` combo; reserved-field rejection; `estimated_total_frames()` arithmetic across all transition types |
| `mold-inference` | orchestrator + fade helper | Mixed-transition orchestrator via fake renderer; verify `Cut` passes `None` carry; fade boundary RGB correctness (alpha ramp, length); `StitchPlan` assembly |
| `mold-server` | routes | `/api/capabilities/chain-limits` response shape; chain route rejects reserved fields with 422; chain SSE emits per-stage events with correct `transition` tag; fail-closed 502 payload shape |
| `mold-cli` | flags + script I/O | Sugar-flag parsing (repeated `--prompt`, uniform `--frames-per-clip`); `--script` file loading; relative source-image path resolution; `--dry-run` output format |
| `mold-tui` | script mode | Keybindings (add/delete/reorder/transition-cycle); TOML save/load round-trip; editor pane focus discipline; stage 0 transition-row suppression |
| `web` | composer | Script-mode state transitions (add/remove/reorder cards); TOML import/export symmetric with Rust writer; ExpandModal per-stage wiring; `localStorage` persistence |
| **Integration** | end-to-end | 3-stage smooth, 3-stage cut, 3-stage fade, mixed — real GPU host (<gpu-host>) |
| **UAT** | acceptance | `tui-uat.sh` script mode scenario; manual browser checklist for web |
| **Regression** | v1 compat | v1 sugar `--prompt` still works byte-for-byte; v1 JSON without `transition` field normalises to all-smooth behavior |

**CI rule:** no real-weight tests in CI (same discipline as v1). Fake
engines via the trait seam for orchestrator tests. End-to-end only runs
on-demand on the <gpu-host> box (`<gpu-host>`, mold repo at
`~/github/mold`, <arch-tag>).

## 9. Team orchestration — six phases

One PR per phase. No mid-phase pushes. Gate review before the next phase
starts.

| # | Phase | Roles active | Gate |
|---|---|---|---|
| 1 | Wire format + capabilities endpoint + TOML I/O | code-explorer, code-architect, backend impl (2 parallel: core types/TOML, server routes), type-design analyzer, verification | `cargo test`, capabilities endpoint curl, TOML round-trip test passes |
| 2 | Engine transitions (cut/fade/stitch) | code-explorer, code-architect, backend impl (single, serial — tightly coupled), silent-failure hunter, verification | end-to-end renders of each transition on <gpu-host> box; fade boundary visual inspection |
| 3 | CLI surface (sugar + script) | code-architect, backend impl, adversarial, code-reviewer, verification | `cargo test`, adversarial's malformed-TOML corpus survives, manual run of `mold run --script ./shot.toml` |
| 4 | TUI surface (script mode) | UI/UX designer-engineer, backend impl (TUI specialist), adversarial, verification | `tui-uat.sh` script mode scenario passes, manual interaction verified |
| 5 | Web surface (composer script mode) | UI/UX designer-engineer via `frontend-design` skill, frontend impl, adversarial, code-reviewer, comment-analyzer, verification | `bun test`, import/export symmetric with CLI, manual browser test across 3 draft scenarios |
| 6 | Docs + release | docs-in-sync, verification | `CLAUDE.md`, `.claude/skills/mold/SKILL.md`, `website/guide/video.md` (new section), `CHANGELOG.md [Unreleased]` entry, VitePress build green |

**Phase dependency graph:**

- Phase 1 → Phase 2 (engine depends on wire format being frozen).
- Phase 2 → Phases 3, 4, 5 (each surface needs the engine to actually
  render cuts and fades end-to-end; their gate criteria require real
  renders on the <gpu-host> box).
- Phases 3, 4, 5 run concurrently (separate crates / directories, no
  shared-file contention after the Phase 1 wire freeze).
- Phase 6 last.

Parallelism sweet spot: three subagent-driven-development workers, one
per surface, fanned out after Phase 2 ships. Adversarial runs
continuously across all phases as a review role.

**Adversarial's standing job** — at least 5 ways to break what was shipped per phase:

- Phase 1: TOML with nested `[[stage.loras]]` populated; schema header
  with unknown version; script with circular `source_image` symlink.
- Phase 2: 16-stage chain with all-fade transitions; fade_len larger than
  stage frames; fade between a smooth-ended stage and a cut-start stage.
- Phase 3: `--prompt` with newlines; script file with absolute path
  outside repo; very large script file (stress TOML parser).
- Phase 4: terminal resize mid-edit; keybinding collision with screen
  readers; empty script submit; very long prompt wrapping.
- Phase 5: 50-stage draft in localStorage (quota); tab closed mid-drag-reorder;
  import of v2.1 schema version (forward compat); paste of malformed TOML.

**Commit scopes:**

- `feat(chain)` — core wire format, TOML I/O, normalise
- `feat(ltx2)` — engine orchestrator + fade stitcher
- `feat(server)` — capabilities endpoint, routes_chain updates
- `feat(cli)` — sugar flags, `--script`, `--dry-run`
- `feat(tui)` — script mode
- `feat(web)` — composer script mode
- `test(chain)` — test-only additions
- `docs(chain)` — docs sync

One scope per commit.

## 10. Out of scope for sub-project A

Hard boundary. Anything listed here does not ship in A:

- Per-stage model selection (sub-project C).
- LoRA stacking, IC-LoRA, named reference characters (sub-project B).
- Live VRAM feasibility cue in the composer (sub-project D).
- Partial-stitch recovery or `mold chain resume <job-id>`.
- Mid-denoise cancellation.
- Story-to-script LLM authoring (v2.1 follow-up).
- `mold chain export <job-id>` (needs DB schema additions).
- Per-stage `steps` / `guidance` / `strength` overrides.
- Motion-tail per-stage override (stays `ChainRequest`-level).
- Cross-model chains (e.g., LTX-2 + FLUX interludes) — waits for C.

## 11. Open questions carried to implementation

These don't block the spec, but the plan should resolve them:

1. **TOML `source_image` path resolution policy for submitted chains
   hosted on a remote server.** Web composer uploads images as base64 in
   the JSON request body; CLI with a local script resolves relative
   paths from the filesystem. TUI inherits the CLI path. No server-side
   file access for safety. Plan should spell out the CLI path-resolution
   rules (e.g., canonical-path check, symlink following policy, size
   cap for images loaded from disk).
2. **Draft-state migration** across mold releases. The
   `mold.chain.draft.v2` localStorage key — what happens when we bump
   the TOML schema to v2? Plan should sketch the migration.
3. **Progress-event rate for fade boundaries.** Fade is a post-stitch
   operation, not a denoise loop. Should it emit any SSE progress, or
   bundle into the existing `Stitching` event? Lean: fold into
   `Stitching` with a message update.
4. **TUI file picker implementation.** Either a custom modal or shell out
   to `$EDITOR` for the path. Plan picks one based on existing TUI
   patterns.
5. **Clippy + fmt pass on the new crates/modules.** CI enforces `-D
   warnings`; plan must hit clippy clean on every phase commit.
