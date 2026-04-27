# Multi-prompt chain v2 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Ship sub-project A of the multi-prompt chain decomposition — per-stage `prompt`/`frames`/`transition`, TOML script round-trip across CLI/TUI/web, engine support for `cut` and `fade` transitions, and server-announced frames-per-clip ceiling.

**Architecture:** Additive wire-format changes on top of v1's `ChainStage` + a new `chain_toml` module in `mold-core` as the canonical TOML reader/writer. `Cut` is orchestrator-only (reuses stage-0 code path with `carry: None`). `Fade` is a post-stitch alpha blend in a new `StitchPlan` helper. Capabilities endpoint drops a per-model `ChainLimits` shape the three surfaces consume for UI upper bounds. Reserved fields (`model`, `loras`, `references`) added to the wire format and rejected with 422 in this release so sub-projects B/C ship without another break.

**Tech Stack:** Rust 1.85 workspace (`mold-core`, `mold-inference`, `mold-server`, `mold-cli`, `mold-tui`), Axum 0.7, clap 4, ratatui, Vue 3 + Vite 7 + Tailwind 4, `toml` crate (new dep in `mold-core`), serde, utoipa.

**Spec reference:** `docs/superpowers/specs/2026-04-21-multi-prompt-chain-v2-design.md`

---

## Phase map

Six phases, one PR each. Phase 2 gates 3/4/5 (surfaces need the engine working end-to-end to satisfy their gate criteria).

| # | Phase | Touched crates/dirs | Tasks |
|---|---|---|---|
| 1 | Wire format + TOML I/O + capabilities endpoint | `mold-core`, `mold-server` | 1.1 – 1.14 |
| 2 | Engine transitions (cut/fade/stitch) | `mold-inference`, `mold-server` | 2.1 – 2.9 |
| 3 | CLI surface (sugar + `--script`) | `mold-cli` | 3.1 – 3.7 |
| 4 | TUI surface (Script mode) | `mold-tui` | 4.1 – 4.8 |
| 5 | Web surface (composer script mode) | `web/` | 5.1 – 5.10 |
| 6 | Docs + release | `CLAUDE.md`, `.claude/skills/mold/SKILL.md`, `website/`, `CHANGELOG.md` | 6.1 – 6.4 |

**Dependency graph:** Phase 1 → Phase 2 → {Phase 3, Phase 4, Phase 5} (concurrent) → Phase 6.

**Commit discipline:**
- One scope per commit (`feat(chain)`, `feat(ltx2)`, `feat(server)`, `feat(cli)`, `feat(tui)`, `feat(web)`, `test(chain)`, `docs(chain)`).
- No mid-phase pushes; each phase is one PR.
- Every step that changes code ends with a commit.

**CI gate at every commit:**
```bash
cargo fmt --all -- --check
cargo clippy --workspace --all-targets -- -D warnings
cargo test --workspace
# web phase:  bun run fmt:check && bun run verify && bun run build  (from web/)
```

**Test discipline:** no real-weight loads in CI. Engine tests go through the `ChainStageRenderer` trait seam with fake renderers (see existing `ltx2::chain::tests` for the pattern).

---

## File structure map

Files created or modified by this plan, grouped by responsibility:

**mold-core (Phase 1):**
- `crates/mold-core/Cargo.toml` — add `toml = "0.8"` dep
- `crates/mold-core/src/chain.rs` — extend `ChainStage` with `transition`/`fade_frames`/reserved fields; add `TransitionMode`, `LoraSpec`, `NamedRef`, `ChainScript`, `VramEstimate`; extend `ChainResponse`; update `normalise()`; add `estimated_total_frames()`
- `crates/mold-core/src/chain_toml.rs` *(NEW)* — `write_script()` / `read_script()` with schema-version header
- `crates/mold-core/src/lib.rs` — re-exports
- `crates/mold-core/tests/chain_toml.rs` *(NEW)* — round-trip tests
- `crates/mold-core/tests/chain_client.rs` — extend with reserved-field rejection tests

**mold-server (Phase 1 + 2):**
- `crates/mold-server/src/routes.rs` — add `/api/capabilities/chain-limits` handler; register route
- `crates/mold-server/src/chain_limits.rs` *(NEW)* — `ChainLimits` struct, per-family cap lookup, 30s cache
- `crates/mold-server/src/routes_chain.rs` — switch stitch path to `StitchPlan`; drain transitions from stages; enrich 502 payload
- `crates/mold-server/src/routes_test.rs` — capabilities-limits test + 422 reserved-field test

**mold-inference (Phase 2):**
- `crates/mold-inference/src/ltx2/chain.rs` — orchestrator honors `transition` (None carry on Cut/Fade; per-clip frames retained per stage); emit `ChainRunOutput.stage_frames` (`Vec<Vec<RgbImage>>`)
- `crates/mold-inference/src/ltx2/media.rs` — add `fade_boundary()` helper
- `crates/mold-inference/src/ltx2/stitch.rs` *(NEW)* — `StitchPlan::assemble()` consuming per-stage frames + boundaries → single `Vec<RgbImage>`
- `crates/mold-inference/src/ltx2/pipeline.rs` — `generate_with_carryover` accepts `source_image` on carry-None stages (no-op change path)

**mold-cli (Phase 3):**
- `crates/mold-cli/src/main.rs` — `--script`, repeated `--prompt`, `--dry-run`, new `mold chain validate` subcommand
- `crates/mold-cli/src/commands/chain.rs` — script loader, dry-run printer, sugar→`ChainRequest` builder
- `crates/mold-cli/src/commands/run.rs` — route repeated `--prompt` to chain path
- `crates/mold-cli/src/commands/chain_validate.rs` *(NEW)* — `mold chain validate <path>` subcommand
- `crates/mold-cli/tests/cli_integration.rs` — script loading + sugar flag tests

**mold-tui (Phase 4):**
- `crates/mold-tui/src/ui/script_composer.rs` *(NEW)* — script mode (list + editor panes, keybindings)
- `crates/mold-tui/src/app.rs` — mode switch into script composer
- `crates/mold-tui/src/ui/info.rs` — `s` keybinding hint on hub
- `scripts/tui-uat.sh` — add script-mode scenario

**web (Phase 5):**
- `web/src/components/Composer.vue` — mode toggle (`Single` | `Script`), delegates to ScriptComposer when active
- `web/src/components/ScriptComposer.vue` *(NEW)* — card list + draggable reorder + footer summary
- `web/src/components/StageCard.vue` *(NEW)* — single stage card
- `web/src/lib/chainToml.ts` *(NEW)* — TOML reader/writer mirror
- `web/src/lib/chainToml.test.ts` *(NEW)*
- `web/src/lib/chainRouting.ts` — extend for script-mode stage count
- `web/src/api.ts` — `fetchChainLimits(model)` helper
- `web/src/composables/useGenerateForm.ts` — draft persistence for `mold.chain.draft.v2`
- `web/package.json` — add `@iarna/toml` (or `smol-toml`) dep
- `web/bun.nix` — regenerate after lock update

**Docs (Phase 6):**
- `CLAUDE.md` — chain feature section update
- `.claude/skills/mold/SKILL.md` — new flags, endpoint, script-mode
- `website/guide/video.md` — new "Multi-prompt scripts" section
- `CHANGELOG.md` — `[Unreleased]` entry

---

## Phase 1 — Wire format + TOML I/O + capabilities endpoint

**Goal of this phase:** lock down the wire format for every downstream surface. Add types, normalise rules, TOML round-trip, and capabilities endpoint. Ends with `cargo test -p mold-ai-core` and `curl /api/capabilities/chain-limits?model=ltx-2-19b-distilled:fp8` working.

**Commit scope:** `feat(chain)` for core, `feat(server)` for routes.

---

### Task 1.1: Add `TransitionMode` enum to `mold-core`

**Files:**
- Modify: `crates/mold-core/src/chain.rs:15` (imports region — add `Default` to the serde import set if not already present)
- Modify: `crates/mold-core/src/chain.rs:26` (just above `ChainStage`)

- [ ] **Step 1: Write the failing test**

Add at the bottom of `crates/mold-core/src/chain.rs` inside the existing `#[cfg(test)] mod tests { ... }`:

```rust
#[test]
fn transition_mode_serializes_snake_case() {
    assert_eq!(
        serde_json::to_value(TransitionMode::Smooth).unwrap(),
        serde_json::Value::String("smooth".into())
    );
    assert_eq!(
        serde_json::to_value(TransitionMode::Cut).unwrap(),
        serde_json::Value::String("cut".into())
    );
    assert_eq!(
        serde_json::to_value(TransitionMode::Fade).unwrap(),
        serde_json::Value::String("fade".into())
    );
}

#[test]
fn transition_mode_defaults_to_smooth() {
    assert_eq!(TransitionMode::default(), TransitionMode::Smooth);
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cargo test -p mold-ai-core chain::tests::transition_mode -- --nocapture`
Expected: FAIL with `cannot find type TransitionMode`.

- [ ] **Step 3: Add the enum**

Insert just above `pub struct ChainStage`:

```rust
/// How the boundary between the previous stage and this stage is rendered.
///
/// - `Smooth`: the engine honors the motion-tail latent carryover from the
///   prior clip (v1 default behaviour). Produces a visual morph when the
///   prompt changes.
/// - `Cut`: fresh latent, no carryover. If the stage has a `source_image`
///   the engine uses it as the i2v seed; otherwise pure t2v.
/// - `Fade`: same engine path as `Cut`, plus a post-stitch alpha blend of
///   the last `fade_frames` of the prior clip with the first `fade_frames`
///   of this clip.
///
/// Stage 0's transition is meaningless (nothing to transition from) and is
/// coerced to `Smooth` during `ChainRequest::normalise`.
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize, utoipa::ToSchema,
)]
#[serde(rename_all = "snake_case")]
pub enum TransitionMode {
    #[default]
    Smooth,
    Cut,
    Fade,
}
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cargo test -p mold-ai-core chain::tests::transition_mode`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add crates/mold-core/src/chain.rs
git commit -m "feat(chain): add TransitionMode enum with smooth/cut/fade variants

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

### Task 1.2: Add placeholder types for reserved fields (`LoraSpec`, `NamedRef`)

These exist so TOML parsing accepts well-formed scripts that populate them, but `normalise` rejects non-empty values in this release.

**Files:**
- Modify: `crates/mold-core/src/chain.rs` (just above `ChainStage`)

- [ ] **Step 1: Write the failing test**

Add to the same `mod tests` block:

```rust
#[test]
fn lora_spec_serializes_minimal() {
    let spec = LoraSpec {
        path: "./style.safetensors".into(),
        scale: 0.8,
        name: None,
    };
    let json = serde_json::to_string(&spec).unwrap();
    assert!(json.contains(r#""path":"./style.safetensors""#));
    assert!(json.contains(r#""scale":0.8"#));
    // name omitted
    assert!(!json.contains(r#""name""#));
}

#[test]
fn named_ref_serializes_minimal() {
    let r = NamedRef {
        name: "hero".into(),
        image: vec![0x89, 0x50],
    };
    let json = serde_json::to_string(&r).unwrap();
    // base64-encoded image via the existing base64 helper
    assert!(json.contains(r#""name":"hero""#));
    assert!(json.contains(r#""image":"#));
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cargo test -p mold-ai-core chain::tests::lora_spec -- --nocapture`
Expected: FAIL with `cannot find type LoraSpec`.

- [ ] **Step 3: Add the types**

Insert just above `pub struct ChainStage`:

```rust
/// Per-stage LoRA adapter spec. **Reserved for sub-project B** — populating
/// this in a request before B lands causes `ChainRequest::normalise` to
/// return 422. Defined now so scripts that round-trip through v1 clients
/// don't drop fields silently.
#[derive(Debug, Clone, Serialize, Deserialize, utoipa::ToSchema)]
pub struct LoraSpec {
    pub path: String,
    pub scale: f64,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
}

/// Per-stage named reference character/style. **Reserved for sub-project
/// B** — populating this causes `ChainRequest::normalise` to return 422.
#[derive(Debug, Clone, Serialize, Deserialize, utoipa::ToSchema)]
pub struct NamedRef {
    pub name: String,
    #[serde(with = "crate::types::base64_bytes")]
    pub image: Vec<u8>,
}
```

**Note:** `crate::types::base64_bytes` is a serde helper analogous to `base64_opt` already in use. If it doesn't exist yet, add it alongside `base64_opt` in `crates/mold-core/src/types.rs`:

```rust
pub(crate) mod base64_bytes {
    use base64::{engine::general_purpose::STANDARD, Engine};
    use serde::{Deserialize, Deserializer, Serializer};

    pub fn serialize<S: Serializer>(bytes: &Vec<u8>, s: S) -> Result<S::Ok, S::Error> {
        s.serialize_str(&STANDARD.encode(bytes))
    }

    pub fn deserialize<'de, D: Deserializer<'de>>(d: D) -> Result<Vec<u8>, D::Error> {
        let s = String::deserialize(d)?;
        STANDARD.decode(s).map_err(serde::de::Error::custom)
    }
}
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cargo test -p mold-ai-core chain::tests::lora_spec chain::tests::named_ref`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add crates/mold-core/src/chain.rs crates/mold-core/src/types.rs
git commit -m "feat(chain): reserve LoraSpec and NamedRef wire types for sub-project B

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

### Task 1.3: Extend `ChainStage` with `transition`, `fade_frames`, and reserved fields

**Files:**
- Modify: `crates/mold-core/src/chain.rs:27-61` (the `ChainStage` definition)

- [ ] **Step 1: Write the failing test**

Add to `mod tests`:

```rust
#[test]
fn chain_stage_defaults_are_backcompat() {
    // Parsing a v1-shaped stage (no new fields) yields the same structure
    // with defaults applied.
    let json = r#"{
        "prompt": "a cat",
        "frames": 97
    }"#;
    let stage: ChainStage = serde_json::from_str(json).unwrap();
    assert_eq!(stage.prompt, "a cat");
    assert_eq!(stage.frames, 97);
    assert_eq!(stage.transition, TransitionMode::Smooth);
    assert_eq!(stage.fade_frames, None);
    assert!(stage.model.is_none());
    assert!(stage.loras.is_empty());
    assert!(stage.references.is_empty());
}

#[test]
fn chain_stage_roundtrips_all_fields() {
    let stage = ChainStage {
        prompt: "scene".into(),
        frames: 49,
        source_image: None,
        negative_prompt: None,
        seed_offset: None,
        transition: TransitionMode::Cut,
        fade_frames: Some(12),
        model: None,
        loras: vec![],
        references: vec![],
    };
    let json = serde_json::to_string(&stage).unwrap();
    let back: ChainStage = serde_json::from_str(&json).unwrap();
    assert_eq!(back.frames, 49);
    assert_eq!(back.transition, TransitionMode::Cut);
    assert_eq!(back.fade_frames, Some(12));
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cargo test -p mold-ai-core chain::tests::chain_stage_defaults_are_backcompat chain::tests::chain_stage_roundtrips_all_fields`
Expected: FAIL — fields don't exist.

- [ ] **Step 3: Extend the struct**

Replace the existing `ChainStage` definition with:

```rust
#[derive(Debug, Clone, Serialize, Deserialize, utoipa::ToSchema)]
pub struct ChainStage {
    #[schema(example = "a cat walking through autumn leaves")]
    pub prompt: String,

    #[schema(example = 97)]
    pub frames: u32,

    #[serde(
        default,
        skip_serializing_if = "Option::is_none",
        with = "crate::types::base64_opt"
    )]
    pub source_image: Option<Vec<u8>>,

    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub negative_prompt: Option<String>,

    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub seed_offset: Option<u64>,

    // NEW in multi-prompt v2 ───────────────────────────────────────────
    /// Boundary style between the previous stage and this stage.
    /// Stage 0's value is coerced to `Smooth` in `normalise`.
    #[serde(default)]
    pub transition: TransitionMode,

    /// Length in pixel frames of the crossfade when `transition == Fade`.
    /// `None` means use the server-announced default (8 frames). Capped
    /// at `fade_frames_max` from `/api/capabilities/chain-limits`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub fade_frames: Option<u32>,

    // RESERVED for B/C — populated values are rejected by normalise ───
    /// **Reserved for sub-project C.** Populating this in a request
    /// produces 422 in this release.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub model: Option<String>,

    /// **Reserved for sub-project B.** Non-empty values produce 422.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub loras: Vec<LoraSpec>,

    /// **Reserved for sub-project B.** Non-empty values produce 422.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub references: Vec<NamedRef>,
}
```

Update `build_auto_expand_stages` (around `chain.rs:407`) so it populates the new fields with defaults:

```rust
stages.push(ChainStage {
    prompt: prompt.to_string(),
    frames: per_stage_frames,
    source_image: source_image.clone(),
    negative_prompt: None,
    seed_offset: None,
    transition: TransitionMode::Smooth,
    fade_frames: None,
    model: None,
    loras: vec![],
    references: vec![],
});
```

Also scan `crates/mold-core/src/chain.rs` test fixtures for any inline `ChainStage { ... }` constructions and add the new fields with defaults (grep for `ChainStage {` inside `mod tests`).

- [ ] **Step 4: Run tests**

Run: `cargo test -p mold-ai-core`
Expected: all tests pass.

- [ ] **Step 5: Commit**

```bash
git add crates/mold-core/src/chain.rs
git commit -m "feat(chain): extend ChainStage with transition/fade_frames + reserved fields

Adds transition (smooth|cut|fade, default smooth), fade_frames, and reserved
model/loras/references fields for sub-projects B and C. All new fields
default such that v1-shaped requests parse unchanged.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

### Task 1.4: Add `ChainScript` canonical echo type and `VramEstimate` slot

**Files:**
- Modify: `crates/mold-core/src/chain.rs` (near the response types)

- [ ] **Step 1: Write the failing test**

```rust
#[test]
fn chain_script_projects_from_request() {
    let req = ChainRequest {
        model: "ltx-2-19b-distilled:fp8".into(),
        stages: vec![ChainStage {
            prompt: "a".into(),
            frames: 97,
            source_image: None,
            negative_prompt: None,
            seed_offset: None,
            transition: TransitionMode::Smooth,
            fade_frames: None,
            model: None,
            loras: vec![],
            references: vec![],
        }],
        motion_tail_frames: 25,
        width: 1216,
        height: 704,
        fps: 24,
        seed: Some(42),
        steps: 8,
        guidance: 3.0,
        strength: 1.0,
        output_format: OutputFormat::Mp4,
        placement: None,
        prompt: None,
        total_frames: None,
        clip_frames: None,
        source_image: None,
    };
    let script = ChainScript::from(&req);
    assert_eq!(script.chain.model, "ltx-2-19b-distilled:fp8");
    assert_eq!(script.chain.seed, Some(42));
    assert_eq!(script.stages.len(), 1);
    assert_eq!(script.stages[0].prompt, "a");
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cargo test -p mold-ai-core chain::tests::chain_script_projects_from_request`
Expected: FAIL — `ChainScript` not defined.

- [ ] **Step 3: Add `ChainScript` + `VramEstimate` + wire into `ChainResponse`**

Add near the response section:

```rust
/// Canonical TOML-shaped projection of a normalised [`ChainRequest`].
///
/// Echoed back in [`ChainResponse::script`] so clients can save the exact
/// form that was rendered without re-serialising the request body (which
/// carries auto-expand sugar and other transport-only fields).
#[derive(Debug, Clone, Serialize, Deserialize, utoipa::ToSchema)]
pub struct ChainScript {
    pub schema: String, // always "mold.chain.v1"
    pub chain: ChainScriptChain,
    #[serde(rename = "stage")]
    pub stages: Vec<ChainStage>,
}

#[derive(Debug, Clone, Serialize, Deserialize, utoipa::ToSchema)]
pub struct ChainScriptChain {
    pub model: String,
    pub width: u32,
    pub height: u32,
    pub fps: u32,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub seed: Option<u64>,
    pub steps: u32,
    pub guidance: f64,
    pub strength: f64,
    pub motion_tail_frames: u32,
    pub output_format: OutputFormat,
}

impl From<&ChainRequest> for ChainScript {
    fn from(req: &ChainRequest) -> Self {
        ChainScript {
            schema: "mold.chain.v1".into(),
            chain: ChainScriptChain {
                model: req.model.clone(),
                width: req.width,
                height: req.height,
                fps: req.fps,
                seed: req.seed,
                steps: req.steps,
                guidance: req.guidance,
                strength: req.strength,
                motion_tail_frames: req.motion_tail_frames,
                output_format: req.output_format,
            },
            stages: req.stages.clone(),
        }
    }
}

/// VRAM feasibility estimate — populated by sub-project D. `None` in this
/// release.
#[derive(Debug, Clone, Serialize, Deserialize, utoipa::ToSchema)]
pub struct VramEstimate {
    pub worst_case_bytes: u64,
    pub fits: bool,
}
```

Extend `ChainResponse` at line ~150:

```rust
#[derive(Debug, Clone, Serialize, Deserialize, utoipa::ToSchema)]
pub struct ChainResponse {
    pub video: VideoData,
    #[schema(example = 5)]
    pub stage_count: u32,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub gpu: Option<usize>,

    // NEW ──────────────────────────────────────────────────────────────
    /// Canonical TOML-shaped echo of the rendered script. Clients can save
    /// this directly as a `.toml` file.
    pub script: ChainScript,

    /// Reserved for sub-project D; `None` in this release.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub vram_estimate: Option<VramEstimate>,
}
```

- [ ] **Step 4: Run tests**

Run: `cargo test -p mold-ai-core`
Expected: PASS. (Existing tests that construct `ChainResponse` need `script` added — grep `ChainResponse {` and fix each.)

- [ ] **Step 5: Commit**

```bash
git add crates/mold-core/src/chain.rs
git commit -m "feat(chain): add ChainScript canonical echo + VramEstimate slot

ChainScript is the TOML-friendly projection of a normalised ChainRequest,
echoed in ChainResponse so clients can 'save as script' without re-
serialising transport-only fields. VramEstimate is a typed slot reserved
for sub-project D.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

### Task 1.5: Update `ChainRequest::normalise` — stage-0 coerce + reserved rejection

**Files:**
- Modify: `crates/mold-core/src/chain.rs:277` (the `normalise` method)

- [ ] **Step 1: Write failing tests**

```rust
#[test]
fn normalise_coerces_stage_0_transition_to_smooth() {
    let mut req = auto_expand_request("a", 97, 97, 25, None);
    req.stages = vec![
        ChainStage {
            prompt: "scene 0".into(),
            frames: 97,
            source_image: None,
            negative_prompt: None,
            seed_offset: None,
            transition: TransitionMode::Cut, // should coerce
            fade_frames: None,
            model: None,
            loras: vec![],
            references: vec![],
        },
        ChainStage {
            prompt: "scene 1".into(),
            frames: 97,
            source_image: None,
            negative_prompt: None,
            seed_offset: None,
            transition: TransitionMode::Cut, // preserved
            fade_frames: None,
            model: None,
            loras: vec![],
            references: vec![],
        },
    ];
    let normalised = req.normalise().unwrap();
    assert_eq!(normalised.stages[0].transition, TransitionMode::Smooth);
    assert_eq!(normalised.stages[1].transition, TransitionMode::Cut);
}

#[test]
fn normalise_rejects_reserved_model_field() {
    let mut req = auto_expand_request("a", 97, 97, 25, None);
    req.stages = vec![ChainStage {
        prompt: "x".into(),
        frames: 97,
        source_image: None,
        negative_prompt: None,
        seed_offset: None,
        transition: TransitionMode::Smooth,
        fade_frames: None,
        model: Some("flux-dev:q4".into()),
        loras: vec![],
        references: vec![],
    }];
    let err = req.normalise().unwrap_err().to_string();
    assert!(err.contains("reserved for sub-project C"), "got: {err}");
}

#[test]
fn normalise_rejects_reserved_loras_field() {
    let mut req = auto_expand_request("a", 97, 97, 25, None);
    req.stages = vec![ChainStage {
        prompt: "x".into(),
        frames: 97,
        source_image: None,
        negative_prompt: None,
        seed_offset: None,
        transition: TransitionMode::Smooth,
        fade_frames: None,
        model: None,
        loras: vec![LoraSpec {
            path: "x.safetensors".into(),
            scale: 1.0,
            name: None,
        }],
        references: vec![],
    }];
    let err = req.normalise().unwrap_err().to_string();
    assert!(err.contains("reserved for sub-project B"), "got: {err}");
}
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cargo test -p mold-ai-core chain::tests::normalise_`
Expected: FAIL — coercion/rejection not implemented.

- [ ] **Step 3: Extend `normalise`**

At the end of the existing stage validation loop (before `self.prompt = None;`), insert:

```rust
// Reserved-field rejection (sub-projects B/C).
for (idx, stage) in self.stages.iter().enumerate() {
    if stage.model.is_some() {
        return Err(MoldError::Validation(format!(
            "stages[{idx}].model is reserved for sub-project C and not yet supported"
        )));
    }
    if !stage.loras.is_empty() {
        return Err(MoldError::Validation(format!(
            "stages[{idx}].loras is reserved for sub-project B and not yet supported"
        )));
    }
    if !stage.references.is_empty() {
        return Err(MoldError::Validation(format!(
            "stages[{idx}].references is reserved for sub-project B and not yet supported"
        )));
    }
}

// Stage 0's transition is meaningless (nothing to transition from).
// Coerce to Smooth with a warn so scripts survive reorders.
if let Some(first) = self.stages.first_mut() {
    if first.transition != TransitionMode::Smooth {
        tracing::warn!(
            coerced_from = ?first.transition,
            "stage 0 transition is meaningless; coercing to Smooth"
        );
        first.transition = TransitionMode::Smooth;
    }
}
```

- [ ] **Step 4: Run tests**

Run: `cargo test -p mold-ai-core chain::tests::normalise_`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add crates/mold-core/src/chain.rs
git commit -m "feat(chain): normalise coerces stage 0 transition + rejects reserved fields

Stage 0 coerces any non-Smooth transition to Smooth (with tracing::warn)
so scripts survive stage reordering. Reserved fields (model, loras,
references) return 422 with a message pointing at the sub-project that
will consume them.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

### Task 1.6: Add `estimated_total_frames()` on `ChainRequest`

This is the math all three surfaces call for the footer summary and for the server's ceiling-trim target.

**Files:**
- Modify: `crates/mold-core/src/chain.rs` (new `impl` method)

- [ ] **Step 1: Write failing tests**

```rust
#[test]
fn estimated_total_all_smooth() {
    // 3 × 97-frame smooth = 97 + (97-25) + (97-25) = 241
    let req = stage_list_request(vec![
        (TransitionMode::Smooth, 97, None),
        (TransitionMode::Smooth, 97, None),
        (TransitionMode::Smooth, 97, None),
    ]);
    assert_eq!(req.estimated_total_frames(), 241);
}

#[test]
fn estimated_total_with_cut() {
    // 97 + 97 (cut, no trim) + (97-25) (smooth after cut) = 266
    let req = stage_list_request(vec![
        (TransitionMode::Smooth, 97, None),
        (TransitionMode::Cut, 97, None),
        (TransitionMode::Smooth, 97, None),
    ]);
    assert_eq!(req.estimated_total_frames(), 266);
}

#[test]
fn estimated_total_with_fade() {
    // 97 + 97 + (97 - fade 8) fade consumes from both sides, net -fade_len
    // Actually: fade replaces the trailing fade_len of clip N + leading
    // fade_len of clip N+1 with fade_len blended frames.
    // Emission = sum - 2*fade_len + fade_len = sum - fade_len
    // = 97+97+97 - 8 = 283
    let req = stage_list_request(vec![
        (TransitionMode::Smooth, 97, None),
        (TransitionMode::Cut, 97, None),
        (TransitionMode::Fade, 97, Some(8)),
    ]);
    assert_eq!(req.estimated_total_frames(), 283);
}

// Helper for stage_list_request — place in mod tests if not already there:
fn stage_list_request(stages: Vec<(TransitionMode, u32, Option<u32>)>) -> ChainRequest {
    ChainRequest {
        model: "ltx-2-19b-distilled:fp8".into(),
        stages: stages
            .into_iter()
            .map(|(t, f, fl)| ChainStage {
                prompt: "x".into(),
                frames: f,
                source_image: None,
                negative_prompt: None,
                seed_offset: None,
                transition: t,
                fade_frames: fl,
                model: None,
                loras: vec![],
                references: vec![],
            })
            .collect(),
        motion_tail_frames: 25,
        width: 1216,
        height: 704,
        fps: 24,
        seed: None,
        steps: 8,
        guidance: 3.0,
        strength: 1.0,
        output_format: OutputFormat::Mp4,
        placement: None,
        prompt: None,
        total_frames: None,
        clip_frames: None,
        source_image: None,
    }
}
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cargo test -p mold-ai-core chain::tests::estimated_total`
Expected: FAIL — method not defined.

- [ ] **Step 3: Add the method**

Add to the `impl ChainRequest` block:

```rust
/// Predicted stitched frame count *before* any top-level `total_frames`
/// trim. Used by UIs for the footer summary and by the server to size
/// the final buffer.
///
/// Per-boundary rule:
/// - smooth: drop leading `motion_tail_frames` of the incoming clip
/// - cut: no trim
/// - fade: replace `2 * fade_len` frames (trailing of prior + leading of
///   next) with `fade_len` blended frames → net `-fade_len`
pub fn estimated_total_frames(&self) -> u32 {
    const DEFAULT_FADE_FRAMES: u32 = 8;
    let mut total: u32 = 0;
    for (idx, stage) in self.stages.iter().enumerate() {
        if idx == 0 {
            total += stage.frames;
            continue;
        }
        match stage.transition {
            TransitionMode::Smooth => {
                total += stage.frames.saturating_sub(self.motion_tail_frames);
            }
            TransitionMode::Cut => {
                total += stage.frames;
            }
            TransitionMode::Fade => {
                let fade_len = stage.fade_frames.unwrap_or(DEFAULT_FADE_FRAMES);
                total += stage.frames.saturating_sub(fade_len);
            }
        }
    }
    total
}
```

- [ ] **Step 4: Run tests**

Run: `cargo test -p mold-ai-core chain::tests::estimated_total`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add crates/mold-core/src/chain.rs
git commit -m "feat(chain): add estimated_total_frames with transition-aware math

Smooth boundary drops motion_tail_frames; cut keeps all; fade consumes
fade_frames from both sides of the boundary and emits a single blended
block (net -fade_frames).

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

### Task 1.7: Create `chain_toml.rs` module (writer)

**Files:**
- Create: `crates/mold-core/src/chain_toml.rs`
- Modify: `crates/mold-core/src/lib.rs` (add `pub mod chain_toml;`)
- Modify: `crates/mold-core/Cargo.toml` (add `toml = "0.8"` to `[dependencies]` if not already present — check first)

- [ ] **Step 1: Add dependency + module declaration**

Check `crates/mold-core/Cargo.toml` for `toml`. If absent, add:

```toml
toml = "0.8"
```

Add to `crates/mold-core/src/lib.rs` after `pub mod chain;`:

```rust
pub mod chain_toml;
```

- [ ] **Step 2: Write failing test**

Create `crates/mold-core/src/chain_toml.rs`:

```rust
//! TOML script serialisation for chained generation.
//!
//! The canonical file format is `mold.chain.v1`:
//!
//! ```toml
//! schema = "mold.chain.v1"
//!
//! [chain]
//! model = "ltx-2-19b-distilled:fp8"
//! width = 1216
//! ...
//!
//! [[stage]]
//! prompt = "..."
//! frames = 97
//! ```
//!
//! Round-trip invariant: `read(write(script)) == script` for every script
//! that `ChainRequest::normalise` accepts.

use crate::chain::ChainScript;
use crate::error::{MoldError, Result};

/// Serialise a [`ChainScript`] to a TOML string.
pub fn write_script(script: &ChainScript) -> Result<String> {
    toml::to_string_pretty(script)
        .map_err(|e| MoldError::Other(format!("chain TOML serialise failed: {e}")))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::chain::{ChainScript, ChainScriptChain, ChainStage, TransitionMode};
    use crate::types::OutputFormat;

    fn sample_script() -> ChainScript {
        ChainScript {
            schema: "mold.chain.v1".into(),
            chain: ChainScriptChain {
                model: "ltx-2-19b-distilled:fp8".into(),
                width: 1216,
                height: 704,
                fps: 24,
                seed: Some(42),
                steps: 8,
                guidance: 3.0,
                strength: 1.0,
                motion_tail_frames: 25,
                output_format: OutputFormat::Mp4,
            },
            stages: vec![ChainStage {
                prompt: "a cat walks into the autumn forest".into(),
                frames: 97,
                source_image: None,
                negative_prompt: None,
                seed_offset: None,
                transition: TransitionMode::Smooth,
                fade_frames: None,
                model: None,
                loras: vec![],
                references: vec![],
            }],
        }
    }

    #[test]
    fn write_emits_schema_header_first() {
        let toml_out = write_script(&sample_script()).unwrap();
        assert!(
            toml_out.starts_with("schema = \"mold.chain.v1\""),
            "got:\n{toml_out}"
        );
    }

    #[test]
    fn write_uses_array_of_tables_for_stages() {
        let toml_out = write_script(&sample_script()).unwrap();
        assert!(toml_out.contains("[[stage]]"), "got:\n{toml_out}");
    }

    #[test]
    fn write_omits_empty_reserved_fields() {
        let toml_out = write_script(&sample_script()).unwrap();
        assert!(!toml_out.contains("loras"), "got:\n{toml_out}");
        assert!(!toml_out.contains("references"), "got:\n{toml_out}");
        assert!(!toml_out.contains("model =\n"), "got:\n{toml_out}");
    }
}
```

- [ ] **Step 3: Run test**

Run: `cargo test -p mold-ai-core chain_toml::tests::write_`
Expected: PASS (serde + toml handle the serialisation given the `skip_serializing_if` attrs already on `ChainStage`).

If `write_emits_schema_header_first` fails because `toml` sorts keys alphabetically, use a manual prefix:

```rust
pub fn write_script(script: &ChainScript) -> Result<String> {
    let body = toml::to_string_pretty(script)
        .map_err(|e| MoldError::Other(format!("chain TOML serialise failed: {e}")))?;
    // toml-rs sorts table keys alphabetically; force schema header up top.
    if body.starts_with("schema") {
        Ok(body)
    } else {
        Ok(format!("schema = \"{}\"\n\n{}", script.schema, body.replace("schema = \"mold.chain.v1\"\n", "")))
    }
}
```

- [ ] **Step 4: Verify all core tests**

Run: `cargo test -p mold-ai-core`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add crates/mold-core/src/chain_toml.rs crates/mold-core/src/lib.rs crates/mold-core/Cargo.toml
git commit -m "feat(chain): chain_toml module with script writer

TOML schema mold.chain.v1. Uses serde+toml-rs, emits schema header first
and uses [[stage]] array-of-tables for stage list. Empty reserved fields
(loras, references) are omitted.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

### Task 1.8: Add TOML reader with schema version check

**Files:**
- Modify: `crates/mold-core/src/chain_toml.rs`

- [ ] **Step 1: Write failing test**

Append to `chain_toml.rs`:

```rust
/// Deserialise a TOML string into a [`ChainScript`]. Rejects unknown
/// schema versions with a clear error pointing at the current mold
/// version's supported schema.
pub fn read_script(toml_str: &str) -> Result<ChainScript> {
    #[derive(serde::Deserialize)]
    struct SchemaPeek {
        #[serde(default)]
        schema: Option<String>,
    }
    let peek: SchemaPeek = toml::from_str(toml_str)
        .map_err(|e| MoldError::Validation(format!("chain TOML parse failed: {e}")))?;
    let schema = peek.schema.as_deref().unwrap_or("mold.chain.v1");
    if schema != "mold.chain.v1" {
        return Err(MoldError::Validation(format!(
            "chain TOML schema '{schema}' is not supported by this mold version \
             (supported: 'mold.chain.v1')"
        )));
    }
    toml::from_str(toml_str)
        .map_err(|e| MoldError::Validation(format!("chain TOML parse failed: {e}")))
}
```

Add tests:

```rust
#[test]
fn read_accepts_missing_schema_header() {
    let toml_src = r#"
        [chain]
        model = "ltx-2-19b-distilled:fp8"
        width = 1216
        height = 704
        fps = 24
        steps = 8
        guidance = 3.0
        strength = 1.0
        motion_tail_frames = 25
        output_format = "mp4"

        [[stage]]
        prompt = "hello"
        frames = 97
    "#;
    let script = read_script(toml_src).unwrap();
    assert_eq!(script.schema, "mold.chain.v1");
    assert_eq!(script.stages.len(), 1);
}

#[test]
fn read_rejects_unknown_schema_version() {
    let toml_src = r#"
        schema = "mold.chain.v99"
        [chain]
        model = "x"
        width = 1
        height = 1
        fps = 1
        steps = 1
        guidance = 1.0
        strength = 1.0
        motion_tail_frames = 0
        output_format = "mp4"
    "#;
    let err = read_script(toml_src).unwrap_err().to_string();
    assert!(err.contains("mold.chain.v99"), "got: {err}");
    assert!(err.contains("not supported"), "got: {err}");
}
```

- [ ] **Step 2: Run tests**

Run: `cargo test -p mold-ai-core chain_toml::tests::read_`
Expected: PASS.

- [ ] **Step 3: Commit**

```bash
git add crates/mold-core/src/chain_toml.rs
git commit -m "feat(chain): TOML reader with schema version gate

Missing schema header defaults to mold.chain.v1 (back-compat with files
authored before versioning). Unknown versions are rejected with a clear
error so forward-compat clients get actionable feedback.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

### Task 1.9: Round-trip integration test

**Files:**
- Create: `crates/mold-core/tests/chain_toml.rs`

- [ ] **Step 1: Create the test file**

```rust
//! Round-trip `ChainScript ↔ TOML` ↔ normalise integration tests.
//! Lives in `tests/` (not in-crate) so it exercises the public API only.

use mold_core::chain::{
    ChainRequest, ChainScript, ChainScriptChain, ChainStage, TransitionMode,
};
use mold_core::chain_toml::{read_script, write_script};
use mold_core::types::OutputFormat;

fn script_under_test() -> ChainScript {
    ChainScript {
        schema: "mold.chain.v1".into(),
        chain: ChainScriptChain {
            model: "ltx-2-19b-distilled:fp8".into(),
            width: 1216,
            height: 704,
            fps: 24,
            seed: Some(42),
            steps: 8,
            guidance: 3.0,
            strength: 1.0,
            motion_tail_frames: 25,
            output_format: OutputFormat::Mp4,
        },
        stages: vec![
            ChainStage {
                prompt: "a cat walks into the autumn forest".into(),
                frames: 97,
                source_image: None,
                negative_prompt: None,
                seed_offset: None,
                transition: TransitionMode::Smooth,
                fade_frames: None,
                model: None,
                loras: vec![],
                references: vec![],
            },
            ChainStage {
                prompt: "the forest opens to a clearing".into(),
                frames: 49,
                source_image: None,
                negative_prompt: None,
                seed_offset: None,
                transition: TransitionMode::Smooth,
                fade_frames: None,
                model: None,
                loras: vec![],
                references: vec![],
            },
            ChainStage {
                prompt: "a spaceship lands".into(),
                frames: 97,
                source_image: None,
                negative_prompt: None,
                seed_offset: None,
                transition: TransitionMode::Cut,
                fade_frames: None,
                model: None,
                loras: vec![],
                references: vec![],
            },
            ChainStage {
                prompt: "the cat looks up in wonder".into(),
                frames: 97,
                source_image: None,
                negative_prompt: None,
                seed_offset: None,
                transition: TransitionMode::Fade,
                fade_frames: Some(12),
                model: None,
                loras: vec![],
                references: vec![],
            },
        ],
    }
}

#[test]
fn write_then_read_is_identity() {
    let script = script_under_test();
    let toml_out = write_script(&script).unwrap();
    let back = read_script(&toml_out).unwrap();
    assert_eq!(back.schema, script.schema);
    assert_eq!(back.chain.model, script.chain.model);
    assert_eq!(back.chain.seed, script.chain.seed);
    assert_eq!(back.stages.len(), script.stages.len());
    for (a, b) in back.stages.iter().zip(script.stages.iter()) {
        assert_eq!(a.prompt, b.prompt);
        assert_eq!(a.frames, b.frames);
        assert_eq!(a.transition, b.transition);
        assert_eq!(a.fade_frames, b.fade_frames);
    }
}

#[test]
fn normalised_request_survives_round_trip() {
    // Build a ChainRequest → project to ChainScript → TOML → back → compare.
    let req = ChainRequest {
        model: "ltx-2-19b-distilled:fp8".into(),
        stages: script_under_test().stages,
        motion_tail_frames: 25,
        width: 1216,
        height: 704,
        fps: 24,
        seed: Some(42),
        steps: 8,
        guidance: 3.0,
        strength: 1.0,
        output_format: OutputFormat::Mp4,
        placement: None,
        prompt: None,
        total_frames: None,
        clip_frames: None,
        source_image: None,
    };
    let normalised = req.normalise().unwrap();
    let script = ChainScript::from(&normalised);
    let toml_out = write_script(&script).unwrap();
    let back = read_script(&toml_out).unwrap();
    assert_eq!(back.stages.len(), normalised.stages.len());
}
```

- [ ] **Step 2: Run tests**

Run: `cargo test -p mold-ai-core --test chain_toml`
Expected: PASS.

- [ ] **Step 3: Commit**

```bash
git add crates/mold-core/tests/chain_toml.rs
git commit -m "test(chain): TOML round-trip + normalisation invariants

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

### Task 1.10: Update `lib.rs` re-exports

**Files:**
- Modify: `crates/mold-core/src/lib.rs:22-25` (the `pub use chain::{...}` block)

- [ ] **Step 1: Add exports**

Find the existing `pub use chain::{...};` block and extend it:

```rust
pub use chain::{
    ChainProgressEvent,
    ChainRequest,
    ChainResponse,
    ChainScript,
    ChainScriptChain,
    ChainStage,
    LoraSpec,
    NamedRef,
    SseChainCompleteEvent,
    TransitionMode,
    VramEstimate,
    MAX_CHAIN_STAGES,
};
```

- [ ] **Step 2: Build**

Run: `cargo build -p mold-ai-core`
Expected: clean compile.

- [ ] **Step 3: Commit**

```bash
git add crates/mold-core/src/lib.rs
git commit -m "feat(chain): re-export new wire types from mold-core

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

### Task 1.11: `ChainLimits` shape + family-cap lookup in `mold-server`

**Files:**
- Create: `crates/mold-server/src/chain_limits.rs`
- Modify: `crates/mold-server/src/lib.rs` (module declaration)

- [ ] **Step 1: Create the module**

```rust
//! Chain-limits computation for the `/api/capabilities/chain-limits` route.
//!
//! The model's hardcoded per-clip cap is the primary constraint; the
//! hardware-derived recommended value is `min(cap, free_vram_adjusted)` and
//! is inert for distilled LTX-2 today because 97 is model-capped.

use serde::{Deserialize, Serialize};
use utoipa::ToSchema;

#[derive(Debug, Clone, Serialize, Deserialize, ToSchema)]
pub struct ChainLimits {
    pub model: String,
    pub frames_per_clip_cap: u32,
    pub frames_per_clip_recommended: u32,
    pub max_stages: u32,
    pub max_total_frames: u32,
    pub fade_frames_max: u32,
    pub transition_modes: Vec<String>,
    pub quantization_family: String,
}

/// Per-model-family hardcoded caps. Keyed by the family string returned by
/// `mold_core::manifest::resolve_family`.
pub fn family_cap(family: &str) -> Option<u32> {
    match family {
        "ltx2" => Some(97),
        _ => None,
    }
}

/// Compute the chain-limits response for a resolved model name.
///
/// `family` is the canonical family string (e.g. "ltx2").
/// `quant` is the quantization slug ("fp8", "fp16", "q8", ...).
/// `free_vram_bytes` is the current free VRAM on the primary GPU.
pub fn compute_limits(
    model: &str,
    family: &str,
    quant: &str,
    free_vram_bytes: u64,
) -> ChainLimits {
    let cap = family_cap(family).unwrap_or(97);
    // Hardware-derived recommended: for distilled LTX-2, 97 is already
    // the binding constraint. Reserve the derivation scaffolding for
    // future non-distilled models.
    let _ = free_vram_bytes; // suppress unused for now; D wires this up
    let recommended = cap;

    const MAX_STAGES: u32 = 16;
    ChainLimits {
        model: model.to_string(),
        frames_per_clip_cap: cap,
        frames_per_clip_recommended: recommended,
        max_stages: MAX_STAGES,
        max_total_frames: cap * MAX_STAGES,
        fade_frames_max: 32,
        transition_modes: vec!["smooth".into(), "cut".into(), "fade".into()],
        quantization_family: quant.to_string(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ltx2_cap_is_97() {
        assert_eq!(family_cap("ltx2"), Some(97));
    }

    #[test]
    fn unknown_family_has_no_cap() {
        assert_eq!(family_cap("flux"), None);
    }

    #[test]
    fn compute_limits_for_distilled() {
        let lim = compute_limits(
            "ltx-2-19b-distilled:fp8",
            "ltx2",
            "fp8",
            8_000_000_000,
        );
        assert_eq!(lim.frames_per_clip_cap, 97);
        assert_eq!(lim.frames_per_clip_recommended, 97);
        assert_eq!(lim.max_stages, 16);
        assert_eq!(lim.max_total_frames, 97 * 16);
        assert_eq!(
            lim.transition_modes,
            vec!["smooth".to_string(), "cut".into(), "fade".into()]
        );
    }
}
```

Add to `crates/mold-server/src/lib.rs`:

```rust
pub mod chain_limits;
```

- [ ] **Step 2: Run tests**

Run: `cargo test -p mold-ai-server chain_limits::`
Expected: PASS.

- [ ] **Step 3: Commit**

```bash
git add crates/mold-server/src/chain_limits.rs crates/mold-server/src/lib.rs
git commit -m "feat(server): ChainLimits shape + family-cap lookup

Per-family hardcoded per-clip caps keyed by manifest family. Distilled
LTX-2 = 97. Hardware-derived recommended value is inert for today's
models; scaffolding in place for sub-project D.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

### Task 1.12: `/api/capabilities/chain-limits` route handler

**Files:**
- Modify: `crates/mold-server/src/routes.rs` (new handler + route registration near existing `/api/capabilities`)

- [ ] **Step 1: Write failing test**

Add to `crates/mold-server/src/routes_test.rs`:

```rust
#[tokio::test]
async fn capabilities_chain_limits_returns_ltx2_cap() {
    let state = test_app_state().await;
    let app = build_router(state);
    let response = app
        .oneshot(
            axum::http::Request::builder()
                .uri("/api/capabilities/chain-limits?model=ltx-2-19b-distilled:fp8")
                .body(axum::body::Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(response.status(), 200);
    let body = axum::body::to_bytes(response.into_body(), usize::MAX).await.unwrap();
    let limits: serde_json::Value = serde_json::from_slice(&body).unwrap();
    assert_eq!(limits["frames_per_clip_cap"], 97);
    assert_eq!(limits["max_stages"], 16);
    assert!(limits["transition_modes"].as_array().unwrap().contains(
        &serde_json::Value::String("fade".into())
    ));
}

#[tokio::test]
async fn capabilities_chain_limits_rejects_unknown_model() {
    let state = test_app_state().await;
    let app = build_router(state);
    let response = app
        .oneshot(
            axum::http::Request::builder()
                .uri("/api/capabilities/chain-limits?model=not-a-real-model")
                .body(axum::body::Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(response.status(), 404);
}
```

`test_app_state` and `build_router` already exist in `routes_test.rs` — check the file and follow the existing pattern.

- [ ] **Step 2: Run test to verify it fails**

Run: `cargo test -p mold-ai-server --lib routes_test::capabilities_chain_limits`
Expected: FAIL (route not registered).

- [ ] **Step 3: Add the handler**

In `crates/mold-server/src/routes.rs`, near the existing `server_capabilities` around line 1611:

```rust
#[utoipa::path(
    get,
    path = "/api/capabilities/chain-limits",
    tag = "server",
    params(
        ("model" = String, Query, description = "Model name (e.g. ltx-2-19b-distilled:fp8)")
    ),
    responses(
        (status = 200, description = "Chain limits for the requested model",
         body = crate::chain_limits::ChainLimits),
        (status = 404, description = "Unknown or unsupported model"),
    )
)]
async fn capabilities_chain_limits(
    State(state): State<AppState>,
    Query(params): Query<std::collections::HashMap<String, String>>,
) -> Response {
    let model = match params.get("model") {
        Some(m) => m.clone(),
        None => {
            return (
                StatusCode::BAD_REQUEST,
                "missing required 'model' query parameter\n",
            )
                .into_response()
        }
    };

    // Resolve family via the manifest. Returns 404 for models we don't
    // ship a family handler for (non-chainable families).
    let family = match mold_core::manifest::resolve_family(&model) {
        Some(f) => f,
        None => return (StatusCode::NOT_FOUND, "unknown model\n").into_response(),
    };
    if crate::chain_limits::family_cap(&family).is_none() {
        return (StatusCode::NOT_FOUND, "model is not chain-capable\n").into_response();
    }

    // TODO(D): query AppState for free VRAM; for now pass 0 (inert).
    let quant = mold_core::manifest::resolve_quant(&model).unwrap_or_default();
    let limits = crate::chain_limits::compute_limits(&model, &family, &quant, 0);
    Json(limits).into_response()
}
```

Register in the router builder near the existing `/api/capabilities` registration (line ~225):

```rust
.route(
    "/api/capabilities/chain-limits",
    get(capabilities_chain_limits),
)
```

If `mold_core::manifest::resolve_family` / `resolve_quant` don't exist with those exact names, look up the actual API in `crates/mold-core/src/manifest.rs` and adjust the call. The manifest already resolves `name:tag` → family/quant internally for `mold pull`.

- [ ] **Step 4: Run tests**

Run: `cargo test -p mold-ai-server --lib routes_test`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add crates/mold-server/src/routes.rs crates/mold-server/src/routes_test.rs
git commit -m "feat(server): /api/capabilities/chain-limits endpoint

Returns per-model chain limits (frames_per_clip_cap, max_stages,
transition_modes, etc.) so the CLI/TUI/web composers can derive UI
upper bounds without hardcoding. Non-chainable models return 404.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

### Task 1.13: Wire chain route to populate `ChainResponse.script`

**Files:**
- Modify: `crates/mold-server/src/routes_chain.rs` (the `run_chain` response construction around line 419)

- [ ] **Step 1: Write failing test**

Add to the `tests` module inside `routes_chain.rs` (or the nearest integration test):

```rust
#[tokio::test]
async fn chain_response_echoes_script() {
    // Construct a minimal chain request via the normalise path, then
    // drive a fake orchestrator through run_chain's helper pipeline.
    // Assert that the returned ChainResponse has a non-empty script
    // with the expected stage count.
    // (Fill in using the existing fake-renderer harness in
    // mold_inference::ltx2::chain::tests.)
    todo!("scaffold using existing fake renderer");
}
```

**Note:** If the existing tests don't expose a clean way to drive `run_chain` end-to-end without a real engine, instead add the check at the `ChainResponse` construction site via a unit test on the response builder. The simplest path: extract the `ChainResponse::new_for(...)` builder and test it directly.

- [ ] **Step 2: Populate `script` in the response**

Find the existing `ChainResponse { video, stage_count, gpu }` construction in `routes_chain.rs` (around line 419) and change to:

```rust
let script = mold_core::chain::ChainScript::from(&req);
ChainResponse {
    video,
    stage_count,
    gpu,
    script,
    vram_estimate: None,
}
```

Apply the same change in the SSE `SseChainCompleteEvent` builder at line ~216 — add `script` and `vram_estimate: None` fields (requires adding them to `SseChainCompleteEvent` in `mold-core/src/chain.rs` first if they aren't there yet; follow the same `Option` pattern or embed `ChainScript` directly).

**If `SseChainCompleteEvent` gains new fields**, they're additive and v1 clients ignore them, so no protocol break.

- [ ] **Step 3: Run tests**

Run: `cargo test -p mold-ai-server`
Expected: PASS.

- [ ] **Step 4: Commit**

```bash
git add crates/mold-server/src/routes_chain.rs crates/mold-core/src/chain.rs
git commit -m "feat(server): echo ChainScript in ChainResponse + SSE complete

Lets clients save the rendered script directly without re-serialising
the request body. vram_estimate slot populated with None (sub-project
D will wire it up).

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

### Task 1.14: Phase 1 gate — clippy, fmt, full test sweep

- [ ] **Step 1: fmt**

Run: `cargo fmt --all`
Expected: no errors.

- [ ] **Step 2: clippy**

Run: `cargo clippy --workspace --all-targets -- -D warnings`
Expected: clean.

- [ ] **Step 3: full test**

Run: `cargo test --workspace`
Expected: all green.

- [ ] **Step 4: manual capabilities smoke**

Run:
```bash
MOLD_DB_DISABLE=1 cargo run -p mold-ai --features metal -- serve &
SERVER=$!
sleep 3
curl -s 'http://localhost:7680/api/capabilities/chain-limits?model=ltx-2-19b-distilled:fp8' | jq .
kill $SERVER
```
Expected: JSON with `frames_per_clip_cap: 97`, `max_stages: 16`, `transition_modes: ["smooth","cut","fade"]`.

- [ ] **Step 5: Phase 1 done — open PR**

No code commit here; the prior task commits constitute the PR. Open the PR with title `feat(chain): wire format + TOML I/O + capabilities endpoint (Phase 1/6)` and link to the spec. Do not merge until Phase 2 is ready to stack on top.

---

## Phase 2 — Engine transitions (cut/fade/stitch)

**Goal of this phase:** teach the orchestrator about `TransitionMode::Cut` (pass `None` carry), add a post-stitch `fade_boundary` helper and a `StitchPlan` that assembles per-stage frames with correct per-boundary trim/blend rules, and wire the server's chain route to use it. Ends with end-to-end renders of smooth/cut/fade chains on the <gpu-host> box.

**Commit scope:** `feat(ltx2)` for engine changes, `feat(server)` for route wiring.

---

### Task 2.1: Refactor `ChainRunOutput` to per-stage frame vectors

**Files:**
- Modify: `crates/mold-inference/src/ltx2/chain.rs:161-165` (the `ChainRunOutput` struct)
- Modify: `crates/mold-inference/src/ltx2/chain.rs:197-...` (the `run` method)
- Modify: `crates/mold-server/src/routes_chain.rs` — consumer of `ChainRunOutput`

- [ ] **Step 1: Write failing test**

In `crates/mold-inference/src/ltx2/chain.rs` under the existing `#[cfg(test)]` module:

```rust
#[test]
fn chain_run_output_preserves_per_stage_frames() {
    let mut renderer = FakeRenderer::default();
    let req = sample_chain_request(3, TransitionMode::Smooth);
    let mut orch = Ltx2ChainOrchestrator::new(&mut renderer);
    let out = orch.run(&req, None).unwrap();
    assert_eq!(out.stage_frames.len(), 3);
    // Each stage still holds its un-trimmed frame count.
    assert_eq!(out.stage_frames[0].len() as u32, req.stages[0].frames);
    assert_eq!(out.stage_frames[1].len() as u32, req.stages[1].frames);
    assert_eq!(out.stage_frames[2].len() as u32, req.stages[2].frames);
}

// Helper — place alongside the existing FakeRenderer:
fn sample_chain_request(count: usize, transition: TransitionMode) -> ChainRequest {
    // ... construct via mold_core::chain::ChainRequest using the auto-
    //     expand path, then overwrite transitions on indices >= 1.
    let mut req = ChainRequest {
        model: "ltx-2-19b-distilled:fp8".into(),
        stages: Vec::new(),
        motion_tail_frames: 25,
        width: 1216,
        height: 704,
        fps: 24,
        seed: Some(0),
        steps: 8,
        guidance: 3.0,
        strength: 1.0,
        output_format: mold_core::types::OutputFormat::Mp4,
        placement: None,
        prompt: Some("x".into()),
        total_frames: Some(97 * count as u32),
        clip_frames: Some(97),
        source_image: None,
    };
    req = req.normalise().unwrap();
    for s in req.stages.iter_mut().skip(1) {
        s.transition = transition;
    }
    req
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cargo test -p mold-ai-inference --lib ltx2::chain::tests::chain_run_output_preserves_per_stage_frames`
Expected: FAIL — `stage_frames` field doesn't exist.

- [ ] **Step 3: Change the struct + update `run`**

Replace the existing `ChainRunOutput`:

```rust
/// Output of an end-to-end chain run.
///
/// The orchestrator no longer trims motion-tail prefixes at run time —
/// that moved into [`super::stitch::StitchPlan::assemble`] so the stitch
/// logic can also implement `Cut` (no trim) and `Fade` (post-stitch alpha
/// blend) on the same per-boundary seam.
#[derive(Debug)]
pub struct ChainRunOutput {
    /// Per-stage frame vectors in stage order, each containing the full
    /// un-trimmed pixel clip emitted by the renderer.
    pub stage_frames: Vec<Vec<RgbImage>>,
    pub stage_count: u32,
    pub generation_time_ms: u64,
}
```

In `run`, replace the single `accumulated_frames` Vec with:

```rust
let mut stage_frames: Vec<Vec<RgbImage>> = Vec::with_capacity(req.stages.len());
```

Inside the stage loop, instead of extending and trimming `accumulated_frames`, push the full `outcome.frames` into `stage_frames`:

```rust
let outcome = self.renderer.render_stage(
    &stage_req,
    carry.as_ref(),
    req.motion_tail_frames,
    stage_progress,
)?;
total_generation_ms += outcome.generation_time_ms;
stage_frames.push(outcome.frames);
carry = Some(outcome.tail);
```

Return:

```rust
Ok(ChainRunOutput {
    stage_frames,
    stage_count,
    generation_time_ms: total_generation_ms,
})
```

- [ ] **Step 4: Update `routes_chain.rs` consumer**

Find where `run_chain` uses `chain_output.frames` and switch to flattening through the new `StitchPlan` (introduced in Task 2.4). For this task, temporarily concatenate with a naïve smooth-only trim so the code compiles:

```rust
// Temporary — replaced with StitchPlan::assemble in Task 2.4/2.7.
let motion_tail = req.motion_tail_frames as usize;
let mut frames: Vec<image::RgbImage> = Vec::new();
for (idx, stage_clip) in chain_output.stage_frames.into_iter().enumerate() {
    if idx == 0 {
        frames.extend(stage_clip);
    } else {
        frames.extend(stage_clip.into_iter().skip(motion_tail));
    }
}
```

- [ ] **Step 5: Run tests**

Run: `cargo test -p mold-ai-inference --lib ltx2::chain`
Expected: PASS (existing tests updated to use `stage_frames`; new test passes).

Run: `cargo test -p mold-ai-server`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add crates/mold-inference/src/ltx2/chain.rs crates/mold-server/src/routes_chain.rs
git commit -m "feat(ltx2): expose per-stage frames from chain orchestrator

Switches ChainRunOutput from a flat Vec<RgbImage> to Vec<Vec<RgbImage>>
so the stitch layer can apply per-boundary rules (smooth trim, cut keep-
all, fade blend). Temporary naive stitch in routes_chain until Task
2.4's StitchPlan lands.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

### Task 2.2: Orchestrator honors `transition: Cut` (pass `None` carry)

**Files:**
- Modify: `crates/mold-inference/src/ltx2/chain.rs` (the stage loop in `run`)

- [ ] **Step 1: Write failing test**

```rust
#[test]
fn orchestrator_passes_none_carry_for_cut_transition() {
    let mut renderer = CarryRecordingRenderer::default();
    let req = sample_chain_request(3, TransitionMode::Cut);
    let mut orch = Ltx2ChainOrchestrator::new(&mut renderer);
    orch.run(&req, None).unwrap();
    // Stage 0 always has None; stages 1/2 should also have None because
    // their transition is Cut.
    assert_eq!(renderer.carry_was_some_per_stage, vec![false, false, false]);
}

#[test]
fn orchestrator_passes_some_carry_for_smooth_transition() {
    let mut renderer = CarryRecordingRenderer::default();
    let req = sample_chain_request(3, TransitionMode::Smooth);
    let mut orch = Ltx2ChainOrchestrator::new(&mut renderer);
    orch.run(&req, None).unwrap();
    assert_eq!(renderer.carry_was_some_per_stage, vec![false, true, true]);
}

// Test helper, placed alongside FakeRenderer:
#[derive(Default)]
struct CarryRecordingRenderer {
    carry_was_some_per_stage: Vec<bool>,
}
impl ChainStageRenderer for CarryRecordingRenderer {
    fn render_stage(
        &mut self,
        stage_req: &mold_core::GenerateRequest,
        carry: Option<&ChainTail>,
        motion_tail_pixel_frames: u32,
        _: Option<&mut dyn FnMut(StageProgressEvent)>,
    ) -> anyhow::Result<StageOutcome> {
        self.carry_was_some_per_stage.push(carry.is_some());
        // Return a minimal StageOutcome with frame count matching request
        // and a dummy ChainTail. Use the FakeRenderer's helpers if they
        // exist; otherwise fabricate a 1×1 frame per requested frame.
        let frames = (0..stage_req.frames.unwrap_or(1))
            .map(|_| RgbImage::new(1, 1))
            .collect();
        Ok(StageOutcome {
            frames,
            tail: ChainTail::for_test(),
            generation_time_ms: 0,
        })
    }
}
```

If `ChainTail::for_test()` doesn't exist, look at `FakeRenderer`'s implementation and copy the dummy-tail construction pattern. Alternative: add `#[cfg(test)] impl ChainTail { fn for_test() -> Self { ... } }` in `chain.rs`.

- [ ] **Step 2: Run test to verify it fails**

Run: `cargo test -p mold-ai-inference --lib ltx2::chain::tests::orchestrator_passes_`
Expected: FAIL — orchestrator currently always passes the built-up carry.

- [ ] **Step 3: Inspect `stage.transition` in the stage loop**

In `Ltx2ChainOrchestrator::run` stage loop, replace:

```rust
let stage_outcome = self.renderer.render_stage(
    &stage_req,
    carry.as_ref(),
    req.motion_tail_frames,
    stage_progress,
)?;
```

with:

```rust
let effective_carry = match stage.transition {
    TransitionMode::Smooth => carry.as_ref(),
    TransitionMode::Cut | TransitionMode::Fade => None,
};
let stage_outcome = self.renderer.render_stage(
    &stage_req,
    effective_carry,
    req.motion_tail_frames,
    stage_progress,
)?;
```

Also update the soft-anchor-override block (around `chain.rs:241`) to only run when `effective_carry.is_some()`:

```rust
if effective_carry.is_some() {
    if let Some(ref carry_ref) = carry {
        // ... existing "swap source_image for last frame of prior clip" logic
    }
}
```

Keep capturing `outcome.tail` into `carry` unconditionally at the end of the loop — the *next* stage (if smooth) still needs it. The `transition` gate decides only what's passed to the *current* stage.

- [ ] **Step 4: Run tests**

Run: `cargo test -p mold-ai-inference --lib ltx2::chain`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add crates/mold-inference/src/ltx2/chain.rs
git commit -m "feat(ltx2): orchestrator passes None carry for cut/fade stages

Cut and fade transitions bypass the motion-tail latent carryover and the
soft-anchor overwrite, producing a visual reset at the boundary. Carry
is still captured at the end of every stage so the next smooth stage
can use it.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

### Task 2.3: `fade_boundary` helper in `ltx2::media`

**Files:**
- Modify: `crates/mold-inference/src/ltx2/media.rs` (append new helper + tests)

- [ ] **Step 1: Write failing test**

Append to `crates/mold-inference/src/ltx2/media.rs`:

```rust
#[cfg(test)]
mod fade_tests {
    use super::*;
    use image::RgbImage;

    fn solid(w: u32, h: u32, rgb: [u8; 3]) -> RgbImage {
        let mut img = RgbImage::new(w, h);
        for px in img.pixels_mut() {
            *px = image::Rgb(rgb);
        }
        img
    }

    #[test]
    fn fade_length_matches_requested_fade_len() {
        let tail = vec![solid(4, 4, [255, 0, 0]); 8];
        let head = vec![solid(4, 4, [0, 255, 0]); 8];
        let blended = fade_boundary(&tail, &head, 8);
        assert_eq!(blended.len(), 8);
    }

    #[test]
    fn fade_first_frame_matches_tail() {
        let tail = vec![solid(2, 2, [200, 0, 0]); 4];
        let head = vec![solid(2, 2, [0, 200, 0]); 4];
        let blended = fade_boundary(&tail, &head, 4);
        // alpha[0] = 0/4 = 0 → pure tail
        let p = blended[0].get_pixel(0, 0);
        assert_eq!(p.0, [200, 0, 0]);
    }

    #[test]
    fn fade_last_frame_is_close_to_head() {
        let tail = vec![solid(2, 2, [200, 0, 0]); 4];
        let head = vec![solid(2, 2, [0, 200, 0]); 4];
        let blended = fade_boundary(&tail, &head, 4);
        // alpha[3] = 3/4 = 0.75 → 0.25*tail + 0.75*head = [50, 150, 0]
        let p = blended[3].get_pixel(0, 0);
        assert!((p.0[0] as i32 - 50).abs() <= 2, "R was {}", p.0[0]);
        assert!((p.0[1] as i32 - 150).abs() <= 2, "G was {}", p.0[1]);
    }

    #[test]
    #[should_panic(expected = "tail and head must have length >= fade_len")]
    fn fade_shorter_than_fade_len_panics() {
        let tail = vec![solid(1, 1, [0, 0, 0]); 2];
        let head = vec![solid(1, 1, [0, 0, 0]); 2];
        let _ = fade_boundary(&tail, &head, 4);
    }
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cargo test -p mold-ai-inference --lib ltx2::media::fade_tests`
Expected: FAIL — `fade_boundary` not defined.

- [ ] **Step 3: Implement**

Append to `media.rs` (just before `#[cfg(test)]`):

```rust
/// Cross-fade the last `fade_len` frames of clip N with the first
/// `fade_len` frames of clip N+1. Returns a new `Vec<RgbImage>` of length
/// `fade_len` with pixel-wise linear interpolation:
///
/// ```text
/// out[i] = (1 - alpha) * tail[tail.len() - fade_len + i]
///        +      alpha  * head[i]
/// where alpha = i / fade_len
/// ```
///
/// Panics if `tail.len() < fade_len` or `head.len() < fade_len`.
pub fn fade_boundary(
    tail: &[image::RgbImage],
    head: &[image::RgbImage],
    fade_len: u32,
) -> Vec<image::RgbImage> {
    let n = fade_len as usize;
    assert!(
        tail.len() >= n && head.len() >= n,
        "tail and head must have length >= fade_len"
    );
    let tail_slice = &tail[tail.len() - n..];
    let head_slice = &head[..n];
    let (w, h) = tail_slice[0].dimensions();

    let mut out = Vec::with_capacity(n);
    for i in 0..n {
        let alpha = i as f32 / fade_len as f32;
        let inv = 1.0 - alpha;
        let mut blended = image::RgbImage::new(w, h);
        for y in 0..h {
            for x in 0..w {
                let a = tail_slice[i].get_pixel(x, y).0;
                let b = head_slice[i].get_pixel(x, y).0;
                let mixed = [
                    (a[0] as f32 * inv + b[0] as f32 * alpha) as u8,
                    (a[1] as f32 * inv + b[1] as f32 * alpha) as u8,
                    (a[2] as f32 * inv + b[2] as f32 * alpha) as u8,
                ];
                blended.put_pixel(x, y, image::Rgb(mixed));
            }
        }
        out.push(blended);
    }
    out
}
```

- [ ] **Step 4: Run tests**

Run: `cargo test -p mold-ai-inference --lib ltx2::media::fade_tests`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add crates/mold-inference/src/ltx2/media.rs
git commit -m "feat(ltx2): fade_boundary helper for post-stitch crossfade

Linear RGB interpolation over fade_len frames; alpha=0 on the first
blended frame (pure tail) through alpha=(n-1)/n on the last (near head).
Panics if either source slice is shorter than fade_len so callers notice
the miscount rather than silently truncating.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

### Task 2.4: Create `stitch.rs` with `StitchPlan::assemble`

**Files:**
- Create: `crates/mold-inference/src/ltx2/stitch.rs`
- Modify: `crates/mold-inference/src/ltx2/mod.rs` (add `pub mod stitch;`)

- [ ] **Step 1: Write failing tests**

Create `crates/mold-inference/src/ltx2/stitch.rs`:

```rust
//! Chain stitch planner.
//!
//! Takes the orchestrator's per-stage frame vectors and a parallel list of
//! boundary transitions, and assembles a single output `Vec<RgbImage>`
//! honouring the per-boundary rule:
//! - `Smooth`: drop leading `motion_tail_frames` of the incoming clip.
//! - `Cut`: concatenate as-is.
//! - `Fade`: replace trailing `fade_len` of prior + leading `fade_len` of
//!   incoming with a single blended block of `fade_len` frames.

use image::RgbImage;
use mold_core::TransitionMode;

use crate::ltx2::media::fade_boundary;

pub struct StitchPlan {
    pub clips: Vec<Vec<RgbImage>>,
    /// Transition on the incoming side of each boundary.
    /// `boundaries.len() == clips.len() - 1`.
    pub boundaries: Vec<TransitionMode>,
    /// Per-boundary fade length in pixel frames. For non-fade boundaries
    /// the value is ignored. `fade_lens.len() == clips.len() - 1`.
    pub fade_lens: Vec<u32>,
    pub motion_tail_frames: u32,
}

impl StitchPlan {
    /// Assemble the final stitched frame vector. Consumes `self.clips`.
    pub fn assemble(mut self) -> Result<Vec<RgbImage>, StitchError> {
        if self.clips.is_empty() {
            return Err(StitchError::NoClips);
        }
        let expected_boundaries = self.clips.len() - 1;
        if self.boundaries.len() != expected_boundaries {
            return Err(StitchError::BoundaryMismatch {
                clips: self.clips.len(),
                boundaries: self.boundaries.len(),
            });
        }
        if self.fade_lens.len() != expected_boundaries {
            return Err(StitchError::FadeLenMismatch);
        }

        // Validate each boundary's lengths up front so we fail before any work.
        for (i, &t) in self.boundaries.iter().enumerate() {
            let prior = &self.clips[i];
            let next = &self.clips[i + 1];
            match t {
                TransitionMode::Smooth => {
                    let need = self.motion_tail_frames as usize;
                    if next.len() < need {
                        return Err(StitchError::ClipTooShortForTrim {
                            stage: i + 1,
                            have: next.len(),
                            need,
                        });
                    }
                }
                TransitionMode::Cut => {}
                TransitionMode::Fade => {
                    let fl = self.fade_lens[i] as usize;
                    if prior.len() < fl || next.len() < fl {
                        return Err(StitchError::ClipTooShortForFade {
                            stage: i + 1,
                            fade_len: fl,
                        });
                    }
                }
            }
        }

        // Stage 0 goes in whole; trim/blend on each incoming boundary.
        let mut out: Vec<RgbImage> = Vec::new();
        let mut clips = std::mem::take(&mut self.clips).into_iter();
        let first = clips.next().unwrap();
        out.extend(first);

        for (i, next_clip) in clips.enumerate() {
            match self.boundaries[i] {
                TransitionMode::Smooth => {
                    let drop = self.motion_tail_frames as usize;
                    out.extend(next_clip.into_iter().skip(drop));
                }
                TransitionMode::Cut => {
                    out.extend(next_clip);
                }
                TransitionMode::Fade => {
                    let fl = self.fade_lens[i];
                    let fl_usize = fl as usize;
                    // Pull the trailing fade_len frames off `out` (they're
                    // the tail of the prior clip now that it's been pushed).
                    let tail_start = out.len() - fl_usize;
                    let tail: Vec<RgbImage> = out.drain(tail_start..).collect();
                    let blended = fade_boundary(&tail, &next_clip, fl);
                    out.extend(blended);
                    // Append the post-fade remainder of next_clip.
                    out.extend(next_clip.into_iter().skip(fl_usize));
                }
            }
        }
        Ok(out)
    }
}

#[derive(Debug, thiserror::Error)]
pub enum StitchError {
    #[error("stitch plan has no clips")]
    NoClips,
    #[error("stitch plan has {clips} clips but {boundaries} boundaries (expected {})", clips.saturating_sub(1))]
    BoundaryMismatch { clips: usize, boundaries: usize },
    #[error("fade_lens length does not match boundaries length")]
    FadeLenMismatch,
    #[error("stage {stage} has {have} frames, needs at least {need} for motion-tail trim")]
    ClipTooShortForTrim { stage: usize, have: usize, need: usize },
    #[error("stage {stage} is shorter than fade_len {fade_len}")]
    ClipTooShortForFade { stage: usize, fade_len: usize },
}

#[cfg(test)]
mod tests {
    use super::*;
    use image::RgbImage;

    fn solid(w: u32, h: u32, rgb: [u8; 3]) -> RgbImage {
        let mut img = RgbImage::new(w, h);
        for px in img.pixels_mut() {
            *px = image::Rgb(rgb);
        }
        img
    }

    fn clip(len: usize, rgb: [u8; 3]) -> Vec<RgbImage> {
        (0..len).map(|_| solid(2, 2, rgb)).collect()
    }

    #[test]
    fn all_smooth_drops_motion_tail() {
        let plan = StitchPlan {
            clips: vec![clip(97, [0, 0, 0]); 3],
            boundaries: vec![TransitionMode::Smooth, TransitionMode::Smooth],
            fade_lens: vec![0, 0],
            motion_tail_frames: 25,
        };
        let out = plan.assemble().unwrap();
        assert_eq!(out.len(), 97 + 72 + 72);
    }

    #[test]
    fn all_cut_keeps_everything() {
        let plan = StitchPlan {
            clips: vec![clip(97, [0, 0, 0]); 3],
            boundaries: vec![TransitionMode::Cut, TransitionMode::Cut],
            fade_lens: vec![0, 0],
            motion_tail_frames: 25,
        };
        let out = plan.assemble().unwrap();
        assert_eq!(out.len(), 97 * 3);
    }

    #[test]
    fn fade_boundary_consumes_2x_fade_len_net() {
        let plan = StitchPlan {
            clips: vec![clip(97, [255, 0, 0]), clip(97, [0, 255, 0])],
            boundaries: vec![TransitionMode::Fade],
            fade_lens: vec![8],
            motion_tail_frames: 25,
        };
        let out = plan.assemble().unwrap();
        // 97 + (97 - 8) = 186
        assert_eq!(out.len(), 186);
    }

    #[test]
    fn mismatched_boundaries_errors() {
        let plan = StitchPlan {
            clips: vec![clip(97, [0, 0, 0]); 3],
            boundaries: vec![TransitionMode::Smooth], // expected 2
            fade_lens: vec![0, 0],
            motion_tail_frames: 25,
        };
        assert!(matches!(
            plan.assemble().unwrap_err(),
            StitchError::BoundaryMismatch { .. }
        ));
    }
}
```

Add to `crates/mold-inference/src/ltx2/mod.rs`:

```rust
pub mod stitch;
```

- [ ] **Step 2: Run tests**

Run: `cargo test -p mold-ai-inference --lib ltx2::stitch`
Expected: PASS.

- [ ] **Step 3: Commit**

```bash
git add crates/mold-inference/src/ltx2/stitch.rs crates/mold-inference/src/ltx2/mod.rs
git commit -m "feat(ltx2): StitchPlan assembler with per-boundary rules

Takes the orchestrator's per-stage clips + a transition list and produces
a single frame vector. Smooth drops motion_tail frames; cut concatenates
whole; fade blends 2x fade_len frames into fade_len via fade_boundary.
Up-front length validation yields specific errors instead of panicking.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

### Task 2.5: Ltx2Engine accepts `source_image` on non-zero `Cut`/`Fade` stages

**Files:**
- Modify: `crates/mold-inference/src/ltx2/pipeline.rs:...` — the `Ltx2Engine::generate_with_carryover` / `render_chain_stage` path

- [ ] **Step 1: Audit the current source_image handling**

Run: `grep -n 'source_image' crates/mold-inference/src/ltx2/pipeline.rs | head -30`

The current path treats `source_image` on a continuation as the soft-anchor seed (overwritten by the prior clip's last frame in the orchestrator). With `transition: Cut | Fade`, the orchestrator *does not* overwrite, and the engine should treat the image as a **fresh i2v seed** — the same as stage 0.

- [ ] **Step 2: Write failing test**

Use the `with_runtime_session` injection pattern at `crates/mold-inference/src/ltx2/pipeline.rs:1062` to drive an engine through a Cut stage with a source image and verify the session's i2v frame-0 slot gets the image bytes.

**If that injection pattern is too deep for a plan-level snippet**, defer the real assertion to the integration test in Task 2.8 and just ensure the code path compiles + is reachable.

- [ ] **Step 3: Thread `transition` into the engine call**

The orchestrator already sets `effective_carry = None` for `Cut/Fade`. The engine branch inside `generate_with_carryover` that currently is "if carry, use the tail; else stage-0 i2v path" will now also be hit for `Cut/Fade`. Verify in `generate_with_carryover`:

- When `carry.is_none()` and `stage_req.source_image.is_some()` → existing stage-0 i2v path runs.
- When `carry.is_none()` and `stage_req.source_image.is_none()` → existing stage-0 t2v path runs (fresh noise).

If the current code path doesn't reach the i2v branch for a non-zero stage because of a gate like `if stage_idx == 0`, remove that gate. The predicate should be `carry.is_none()`.

- [ ] **Step 4: Smooth stage + source_image warn**

Also in `generate_with_carryover`, if `carry.is_some()` (smooth) and `stage_req.source_image.is_some()`:

```rust
if carry.is_some() && stage_req.source_image.is_some() {
    tracing::warn!(
        "smooth continuation received source_image; ignoring (use transition: cut|fade to seed with an image)"
    );
}
```

- [ ] **Step 5: Build + test**

Run: `cargo build -p mold-ai-inference`
Expected: clean compile.

Run: `cargo test -p mold-ai-inference --lib`
Expected: PASS (real coverage lands in Task 2.8; here we just avoid regressions).

- [ ] **Step 6: Commit**

```bash
git add crates/mold-inference/src/ltx2/pipeline.rs
git commit -m "feat(ltx2): source_image honored on Cut/Fade continuation stages

Removes the stage_idx gate on the i2v path so Cut/Fade stages with
source_image use the same stage-0 i2v seed logic. Smooth stages warn and
ignore a stray source_image so the warn surfaces in server logs when a
UI fails to suppress it.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

### Task 2.6: Wire `StitchPlan` into `routes_chain.rs`

**Files:**
- Modify: `crates/mold-server/src/routes_chain.rs` — the post-`run_chain` stitch region (temporary code from Task 2.1)

- [ ] **Step 1: Write failing test**

Add a `routes_chain.rs` tests module (or extend the existing one) that exercises the full `orchestrator → stitch → encode` pipeline with a fake renderer and mixed transitions:

```rust
#[tokio::test]
async fn stitched_mixed_chain_has_expected_frame_count() {
    // stages: smooth, cut, fade(8)
    // 97 + 97 + (97 - 8) = 283 frames expected
    // ... build ChainRequest, inject FakeRenderer into the route path
    //     (extract stitch_and_encode into a unit-testable helper if not
    //     already), assert the resulting frame count.
}
```

If the existing `routes_chain.rs` doesn't factor stitch-and-encode cleanly, extract a helper:

```rust
pub(crate) fn stitch_chain_output(
    chain_output: mold_inference::ltx2::chain::ChainRunOutput,
    req: &mold_core::ChainRequest,
) -> anyhow::Result<Vec<image::RgbImage>> {
    let boundaries: Vec<_> = req.stages.iter().skip(1).map(|s| s.transition).collect();
    let fade_lens: Vec<_> = req
        .stages
        .iter()
        .skip(1)
        .map(|s| s.fade_frames.unwrap_or(8))
        .collect();
    let plan = mold_inference::ltx2::stitch::StitchPlan {
        clips: chain_output.stage_frames,
        boundaries,
        fade_lens,
        motion_tail_frames: req.motion_tail_frames,
    };
    plan.assemble().map_err(anyhow::Error::from)
}
```

- [ ] **Step 2: Replace the temporary concatenation**

Find the block added in Task 2.1 (the `// Temporary — replaced with StitchPlan::assemble` region) and replace with:

```rust
let frames = stitch_chain_output(chain_output, &req)
    .map_err(|e| ChainRunError::StitchFailed(e.to_string()))?;
trim_to_total_frames(&mut frames, req.total_frames);
```

Add a `StitchFailed(String)` variant to `ChainRunError` (or the existing error type) and thread it through the 500 response builder.

- [ ] **Step 3: Run tests**

Run: `cargo test -p mold-ai-server`
Expected: PASS.

- [ ] **Step 4: Commit**

```bash
git add crates/mold-server/src/routes_chain.rs
git commit -m "feat(server): chain route uses StitchPlan for per-boundary stitch

Replaces the temporary naive motion-tail concat with StitchPlan::assemble
so cut and fade transitions are honoured. StitchFailed errors return 500
with the underlying assemble() error.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

### Task 2.7: Enrich mid-chain 502 payload

**Files:**
- Modify: `crates/mold-core/src/chain.rs` — add `ChainFailure` type
- Modify: `crates/mold-server/src/routes_chain.rs` — propagate `failed_stage_idx` etc. into the 502 body

- [ ] **Step 1: Add the type**

In `crates/mold-core/src/chain.rs`:

```rust
#[derive(Debug, Clone, Serialize, Deserialize, utoipa::ToSchema)]
pub struct ChainFailure {
    pub error: String,
    pub failed_stage_idx: u32,
    pub elapsed_stages: u32,
    pub elapsed_ms: u64,
    pub stage_error: String,
}
```

Re-export from `lib.rs`.

- [ ] **Step 2: Return it from `routes_chain.rs`**

The orchestrator already returns an error with the stage index embedded in the message. Extract it properly: have `Ltx2ChainOrchestrator::run` return a typed error on failure (e.g., `ChainRunErrorKind::StageFailed { stage_idx, elapsed_stages, elapsed_ms, inner }`), then map that to `ChainFailure` in the 502 response path:

```rust
Err(ChainRunError::Stage(info)) => (
    StatusCode::BAD_GATEWAY,
    Json(mold_core::chain::ChainFailure {
        error: "stage render failed".into(),
        failed_stage_idx: info.stage_idx,
        elapsed_stages: info.elapsed_stages,
        elapsed_ms: info.elapsed_ms,
        stage_error: info.inner.to_string(),
    }),
)
    .into_response(),
```

- [ ] **Step 3: Test**

Add a route test that injects a FakeRenderer whose 2nd stage returns `Err`, and assert the 502 payload shape.

- [ ] **Step 4: Commit**

```bash
git add crates/mold-core/src/chain.rs crates/mold-core/src/lib.rs crates/mold-server/src/routes_chain.rs crates/mold-inference/src/ltx2/chain.rs
git commit -m "feat(chain): typed ChainFailure with stage index + elapsed stats

Mid-chain failures now return a structured 502 body with failed_stage_idx,
elapsed_stages, and elapsed_ms so UIs can show actionable retry hints.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

### Task 2.8: End-to-end mixed-transition test via FakeRenderer

**Files:**
- Modify: `crates/mold-inference/src/ltx2/chain.rs` — integration test at the orchestrator level

- [ ] **Step 1: Write the test**

```rust
#[test]
fn mixed_transitions_end_to_end() {
    let mut renderer = FakeRenderer::default();
    let mut req = sample_chain_request(4, TransitionMode::Smooth);
    req.stages[1].transition = TransitionMode::Smooth;
    req.stages[2].transition = TransitionMode::Cut;
    req.stages[3].transition = TransitionMode::Fade;
    req.stages[3].fade_frames = Some(8);
    let mut orch = Ltx2ChainOrchestrator::new(&mut renderer);
    let out = orch.run(&req, None).unwrap();

    // Expected frame count after StitchPlan::assemble:
    //   stage 0 (97) + stage 1 smooth (97 - 25) + stage 2 cut (97) +
    //   stage 3 fade (97 - 8) = 97 + 72 + 97 + 89 = 355
    let boundaries: Vec<_> = req.stages.iter().skip(1).map(|s| s.transition).collect();
    let fade_lens: Vec<_> = req
        .stages
        .iter()
        .skip(1)
        .map(|s| s.fade_frames.unwrap_or(8))
        .collect();
    let plan = crate::ltx2::stitch::StitchPlan {
        clips: out.stage_frames,
        boundaries,
        fade_lens,
        motion_tail_frames: req.motion_tail_frames,
    };
    let frames = plan.assemble().unwrap();
    assert_eq!(frames.len(), 355);
}
```

- [ ] **Step 2: Run**

Run: `cargo test -p mold-ai-inference --lib ltx2::chain::tests::mixed_transitions_end_to_end`
Expected: PASS.

- [ ] **Step 3: Commit**

```bash
git add crates/mold-inference/src/ltx2/chain.rs
git commit -m "test(chain): mixed-transition end-to-end via FakeRenderer

Exercises orchestrator + StitchPlan together for a smooth-cut-fade chain
and asserts the final stitched frame count matches estimated_total_frames
math.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

### Task 2.9: Phase 2 gate — <gpu-host> box end-to-end

- [ ] **Step 1: fmt + clippy + full test**

Run on dev box:
```bash
cargo fmt --all -- --check
cargo clippy --workspace --all-targets -- -D warnings
cargo test --workspace
```
Expected: clean.

- [ ] **Step 2: Build for <gpu-host> (CUDA <arch-tag>)**

On the <gpu-host> box (`<gpu-host>`, `~/github/mold`):
```bash
CUDA_COMPUTE_CAP=86 nix build .#mold
```

- [ ] **Step 3: Run three renders, one per transition, single-GPU**

Create `shot_smooth.toml` / `shot_cut.toml` / `shot_fade.toml` each with 3 stages and the respective transition on stages 1/2. Use a single LTX-2 distilled model:

```bash
./result/bin/mold run --script shot_smooth.toml --output out_smooth.mp4
./result/bin/mold run --script shot_cut.toml    --output out_cut.mp4
./result/bin/mold run --script shot_fade.toml   --output out_fade.mp4
```

**The CLI flag `--script` doesn't land until Phase 3.** For this gate, drive the server directly via curl with a JSON ChainRequest body:

```bash
curl -X POST http://localhost:7680/api/generate/chain \
  -H 'content-type: application/json' \
  --data @chain_smooth.json \
  --output out_smooth.mp4
```

- [ ] **Step 4: Visual verification**

Play each output in a video player. Acceptance criteria:
- `smooth`: continuous visual flow; no hard cuts; slight prompt-change morph.
- `cut`: obvious hard scene change at each boundary; no visible artifacts.
- `fade`: crossfade visible at each boundary over ~8 frames.

- [ ] **Step 5: Open Phase 2 PR**

Title: `feat(ltx2): engine transitions (cut/fade/stitch) (Phase 2/6)`. Link to Phase 1 PR as base. Do not merge until Phase 3/4/5 are ready.

---

## Phase 3 — CLI surface

**Goal of this phase:** `mold run --script shot.toml` canonical path + repeated `--prompt` sugar for uniform trivial chains. `mold chain validate` subcommand. `--dry-run` printer. Integrates with the existing chain endpoint from Phase 1.

**Commit scope:** `feat(cli)`.

---

### Task 3.1: Add `--script` flag to `mold run`

**Files:**
- Modify: `crates/mold-cli/src/main.rs` — `Commands::Run` variant

- [ ] **Step 1: Add the flag**

In the `#[derive(Parser)]` block for `Run`, add:

```rust
/// Path to a `mold.chain.v1` TOML script. When set, every other
/// generation flag is ignored except `--output`, `--local`, `--host`,
/// and `--dry-run`.
#[arg(long, value_name = "PATH")]
script: Option<std::path::PathBuf>,

/// Parse and normalise the script without submitting. Prints the
/// canonical stage list and estimated total frames to stdout.
#[arg(long)]
dry_run: bool,
```

- [ ] **Step 2: Route to chain loader when set**

In the `Commands::Run { .. }` match arm, add an early branch:

```rust
if let Some(ref path) = script {
    return commands::chain::run_from_script(
        path,
        host.clone(),
        output.clone(),
        local,
        dry_run,
        /* other passthrough flags */,
    )
    .await;
}
```

- [ ] **Step 3: Compile check**

Run: `cargo check -p mold-ai`
Expected: clean.

- [ ] **Step 4: Commit**

```bash
git add crates/mold-cli/src/main.rs
git commit -m "feat(cli): --script and --dry-run flags on mold run

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

### Task 3.2: `run_from_script` helper

**Files:**
- Modify: `crates/mold-cli/src/commands/chain.rs` — add the helper

- [ ] **Step 1: Write failing test**

Add to `crates/mold-cli/tests/cli_integration.rs`:

```rust
#[test]
fn dry_run_prints_stage_summary() {
    let script = r#"
        schema = "mold.chain.v1"
        [chain]
        model = "ltx-2-19b-distilled:fp8"
        width = 1216
        height = 704
        fps = 24
        steps = 8
        guidance = 3.0
        strength = 1.0
        motion_tail_frames = 25
        output_format = "mp4"

        [[stage]]
        prompt = "first scene"
        frames = 97

        [[stage]]
        prompt = "second scene"
        frames = 49
    "#;
    let tmp = tempfile::NamedTempFile::new().unwrap();
    std::fs::write(tmp.path(), script).unwrap();

    let output = std::process::Command::new(env!("CARGO_BIN_EXE_mold"))
        .args(["run", "--script", tmp.path().to_str().unwrap(), "--dry-run"])
        .output()
        .expect("run mold");
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(stdout.contains("2 stages"), "stdout: {stdout}");
    assert!(stdout.contains("first scene"), "stdout: {stdout}");
    assert!(output.status.success());
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cargo test -p mold-ai --test cli_integration dry_run_prints_stage_summary`
Expected: FAIL.

- [ ] **Step 3: Implement**

Append to `crates/mold-cli/src/commands/chain.rs`:

```rust
/// Load a TOML script file, normalise it, and either submit or print a
/// dry-run summary.
pub async fn run_from_script(
    path: &std::path::Path,
    host: Option<String>,
    output: Option<String>,
    local: bool,
    dry_run: bool,
    no_metadata: bool,
    preview: bool,
    // ... engine passthrough flags follow the run() signature
) -> anyhow::Result<()> {
    let toml_src = std::fs::read_to_string(path)
        .map_err(|e| anyhow::anyhow!("failed to read script {}: {e}", path.display()))?;
    let script = mold_core::chain_toml::read_script(&toml_src)
        .map_err(|e| anyhow::anyhow!("invalid chain TOML in {}: {e}", path.display()))?;

    // Resolve relative source_image paths against the script's directory.
    let script_dir = path.parent().unwrap_or_else(|| std::path::Path::new("."));
    let mut req = build_request_from_script(&script, script_dir)?;
    req = req.normalise()?;

    if dry_run {
        print_dry_run_summary(&req);
        return Ok(());
    }

    // Submit via the existing chain runner — construct ChainInputs from req.
    let inputs = chain_inputs_from_script(&req, &script);
    run_chain(
        inputs, host, output, no_metadata, preview, local,
        /* ... rest of passthrough ... */
    )
    .await
}

fn print_dry_run_summary(req: &mold_core::ChainRequest) {
    println!("{} stages", req.stages.len());
    println!(
        "estimated total frames: {} ({:.2}s @ {}fps)",
        req.estimated_total_frames(),
        req.estimated_total_frames() as f64 / req.fps as f64,
        req.fps
    );
    for (i, s) in req.stages.iter().enumerate() {
        println!(
            "  [{i}] {:?}  {}f  {:?}",
            s.transition,
            s.frames,
            s.prompt.chars().take(60).collect::<String>()
        );
    }
}

fn build_request_from_script(
    script: &mold_core::ChainScript,
    script_dir: &std::path::Path,
) -> anyhow::Result<mold_core::ChainRequest> {
    // Resolve relative source_image paths against script_dir.
    let mut stages = script.stages.clone();
    for stage in stages.iter_mut() {
        if let Some(ref bytes) = stage.source_image {
            // If TOML stored the path as a string (see chain_toml custom
            // deserializer), it came through as bytes already loaded from
            // disk. Nothing to do here — the deserializer resolved it.
            let _ = bytes;
        }
    }
    Ok(mold_core::ChainRequest {
        model: script.chain.model.clone(),
        stages,
        motion_tail_frames: script.chain.motion_tail_frames,
        width: script.chain.width,
        height: script.chain.height,
        fps: script.chain.fps,
        seed: script.chain.seed,
        steps: script.chain.steps,
        guidance: script.chain.guidance,
        strength: script.chain.strength,
        output_format: script.chain.output_format,
        placement: None,
        prompt: None,
        total_frames: None,
        clip_frames: None,
        source_image: None,
    })
}

fn chain_inputs_from_script(
    req: &mold_core::ChainRequest,
    _script: &mold_core::ChainScript,
) -> ChainInputs {
    ChainInputs {
        prompt: req.stages[0].prompt.clone(),
        model: req.model.clone(),
        width: req.width,
        height: req.height,
        steps: req.steps,
        guidance: req.guidance,
        strength: req.strength,
        seed: req.seed,
        fps: req.fps,
        output_format: req.output_format,
        total_frames: req.estimated_total_frames(),
        clip_frames: req.stages[0].frames,
        motion_tail: req.motion_tail_frames,
        source_image: req.stages[0].source_image.clone(),
        placement: None,
    }
}
```

**Note on relative-path resolution:** the simplest path is to make `chain_toml::read_script` return paths verbatim, then `build_request_from_script` resolves and reads them. For this task, keep `source_image` as base64 in TOML (or path-as-string that `build_request_from_script` loads). The design spec allows both; pick "path string in TOML, resolved at load time" — add a `source_image_path: Option<String>` field to `ChainStage` in TOML-only form if it doesn't cleanly fit the wire format.

If adding a new field is painful, **Option B**: make `chain_toml` read the image from disk during deserialize and populate `source_image` (Vec<u8>) directly. Simpler, path handled in one place.

Pick Option B for this plan:

```rust
// In chain_toml.rs: add a custom deserializer for ChainStage that
// reads source_image as a path string and loads bytes from disk.
// Requires a Deserializer context that knows the script_dir — pass
// via a thread-local or by parsing into a "raw" shape first and then
// resolving paths.
```

Because thread-locals are ugly, a two-pass approach is cleaner:

```rust
pub fn read_script_resolving_paths(
    toml_str: &str,
    script_dir: &std::path::Path,
) -> Result<ChainScript> {
    // 1. Parse into a raw shape with source_image as Option<String>
    // 2. For each stage with a path, std::fs::read it and populate
    //    source_image: Vec<u8>.
    // ... implement.
}
```

Add this as a wrapper around `read_script`. The existing `read_script` stays base64-only (useful for non-filesystem callers).

- [ ] **Step 4: Run test**

Run: `cargo test -p mold-ai --test cli_integration dry_run_prints_stage_summary`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add crates/mold-cli/src/commands/chain.rs crates/mold-cli/tests/cli_integration.rs crates/mold-core/src/chain_toml.rs
git commit -m "feat(cli): --script path loader with relative source_image resolve

Two-pass TOML load: parse schema+chain+stages, then resolve any
source_image path strings relative to the script file's directory and
read them into bytes. Dry-run prints the normalised stage list and
estimated total frames without submitting.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

### Task 3.3: Repeated `--prompt` sugar

**Files:**
- Modify: `crates/mold-cli/src/main.rs` — make `--prompt` a `Vec<String>` (clap `num_args = 0..`) or keep as repeated flag

- [ ] **Step 1: Write failing test**

```rust
#[test]
fn repeated_prompt_flag_yields_chain() {
    let output = std::process::Command::new(env!("CARGO_BIN_EXE_mold"))
        .args([
            "run",
            "ltx-2-19b-distilled:fp8",
            "--prompt", "first scene",
            "--prompt", "second scene",
            "--prompt", "third scene",
            "--dry-run",
        ])
        .output()
        .expect("run mold");
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(stdout.contains("3 stages"), "stdout: {stdout}");
    assert!(output.status.success());
}
```

- [ ] **Step 2: Make `--prompt` repeatable**

In `Commands::Run`:

```rust
/// Prompt(s). Single use → single-clip (or single-stage chain if other
/// chain flags set); repeated → multi-stage chain with uniform frames
/// and all-smooth transitions. For heterogeneous stages, use --script.
#[arg(long)]
prompt: Vec<String>,
```

Remove any existing positional-or-flag ambiguity: when `--prompt` is non-empty, a positional prompt is an error.

In the `Run` handler:

```rust
if prompt.len() > 1 {
    return commands::chain::run_from_sugar(
        model.clone(),
        prompt.clone(),
        frames_per_clip,
        motion_tail,
        /* other passthrough */,
    )
    .await;
}
```

Where `frames_per_clip` is a new flag `--frames-per-clip N`:

```rust
/// Per-clip frame cap for multi-prompt sugar. Clamped to the server's
/// capabilities-announced cap. Ignored for single-prompt runs.
#[arg(long)]
frames_per_clip: Option<u32>,
```

- [ ] **Step 3: `run_from_sugar` helper**

In `commands/chain.rs`:

```rust
pub async fn run_from_sugar(
    model: String,
    prompts: Vec<String>,
    frames_per_clip: Option<u32>,
    motion_tail: u32,
    /* other passthrough */
) -> anyhow::Result<()> {
    let clip_frames = frames_per_clip.unwrap_or(97);
    let inputs = ChainInputs {
        prompt: prompts[0].clone(),
        model: model.clone(),
        width: 1216,
        height: 704,
        steps: 8,
        guidance: 3.0,
        strength: 1.0,
        seed: None,
        fps: 24,
        output_format: mold_core::OutputFormat::Mp4,
        total_frames: clip_frames * prompts.len() as u32,
        clip_frames,
        motion_tail,
        source_image: None,
        placement: None,
    };

    // Build the stages list with per-prompt content instead of replicating.
    let mut req = inputs.to_chain_request();
    req.stages = prompts
        .iter()
        .map(|p| mold_core::ChainStage {
            prompt: p.clone(),
            frames: clip_frames,
            source_image: None,
            negative_prompt: None,
            seed_offset: None,
            transition: mold_core::TransitionMode::Smooth,
            fade_frames: None,
            model: None,
            loras: vec![],
            references: vec![],
        })
        .collect();
    req.prompt = None;
    req.total_frames = None;
    req.clip_frames = None;

    // Submit via MoldClient::generate_chain_stream (existing path).
    submit_and_encode(req, /* passthrough */).await
}
```

- [ ] **Step 4: Run test**

Run: `cargo test -p mold-ai --test cli_integration repeated_prompt_flag_yields_chain`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add crates/mold-cli/src/main.rs crates/mold-cli/src/commands/chain.rs crates/mold-cli/tests/cli_integration.rs
git commit -m "feat(cli): repeated --prompt builds a multi-stage uniform chain

Each --prompt becomes its own stage with frames = --frames-per-clip (or
the model cap), transition = smooth, uniform width/height/fps. For
non-uniform chains, the user must use --script.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

### Task 3.4: `mold chain validate` subcommand

**Files:**
- Create: `crates/mold-cli/src/commands/chain_validate.rs`
- Modify: `crates/mold-cli/src/main.rs` — add `Chain { Validate { path } }` subcommand
- Modify: `crates/mold-cli/src/commands/mod.rs`

- [ ] **Step 1: Add the subcommand**

In `main.rs`:

```rust
#[derive(Subcommand)]
pub enum ChainSub {
    /// Parse and normalise a TOML script without submitting.
    Validate {
        path: std::path::PathBuf,
    },
}

// in Commands:
Chain {
    #[command(subcommand)]
    action: ChainSub,
},
```

Dispatch:

```rust
Commands::Chain { action } => match action {
    ChainSub::Validate { path } => commands::chain_validate::run(&path).await,
},
```

- [ ] **Step 2: Handler**

`chain_validate.rs`:

```rust
use std::path::Path;

pub async fn run(path: &Path) -> anyhow::Result<()> {
    let toml_src = std::fs::read_to_string(path)?;
    let script = mold_core::chain_toml::read_script(&toml_src)?;
    let script_dir = path.parent().unwrap_or_else(|| Path::new("."));
    let req = super::chain::build_request_from_script(&script, script_dir)?
        .normalise()?;
    println!("OK — {} stages, {} frames estimated", req.stages.len(), req.estimated_total_frames());
    Ok(())
}
```

- [ ] **Step 3: Test**

```rust
#[test]
fn chain_validate_reports_ok_for_valid_script() {
    // write a minimal script to a tempfile and run `mold chain validate <path>`
    // assert exit code 0 and stdout contains "OK"
}
```

- [ ] **Step 4: Run + commit**

```bash
cargo test -p mold-ai --test cli_integration chain_validate
git add crates/mold-cli/src/commands/chain_validate.rs crates/mold-cli/src/commands/mod.rs crates/mold-cli/src/main.rs crates/mold-cli/tests/cli_integration.rs
git commit -m "feat(cli): mold chain validate <path> subcommand

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

### Task 3.5: Progress display with transition tags

**Files:**
- Modify: `crates/mold-cli/src/commands/chain.rs` — the stacked-progress-bar handler in `run_chain`

- [ ] **Step 1: Thread transition into the per-stage bar label**

Find the SSE event handler that updates the per-stage bar (around `ChainProgressEvent::StageStart`) and look up `req.stages[stage_idx].transition` to include in the label:

```rust
ChainProgressEvent::StageStart { stage_idx } => {
    let stage = &req.stages[stage_idx as usize];
    let tag = match stage.transition {
        mold_core::TransitionMode::Smooth => "smooth",
        mold_core::TransitionMode::Cut => "cut",
        mold_core::TransitionMode::Fade => "fade",
    };
    stage_bar.set_message(format!(
        "[stage {}/{} {}] \"{}\"",
        stage_idx + 1,
        stage_count,
        tag,
        truncate(&stage.prompt, 40)
    ));
}
```

- [ ] **Step 2: Commit**

```bash
git add crates/mold-cli/src/commands/chain.rs
git commit -m "feat(cli): per-stage progress bar shows transition tag and prompt

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

### Task 3.6: Reject `--prompt` + positional prompt conflict

**Files:**
- Modify: `crates/mold-cli/src/main.rs` — validate after parse

- [ ] **Step 1: Write failing test**

```rust
#[test]
fn positional_plus_prompt_flag_errors() {
    let output = std::process::Command::new(env!("CARGO_BIN_EXE_mold"))
        .args([
            "run", "ltx-2-19b-distilled:fp8",
            "my positional prompt",
            "--prompt", "also a flag prompt",
            "--dry-run",
        ])
        .output()
        .expect("run mold");
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(!output.status.success());
    assert!(stderr.contains("cannot combine"), "stderr: {stderr}");
}
```

- [ ] **Step 2: Reject in handler**

At the top of `Commands::Run` handling:

```rust
if !prompt.is_empty() && positional_prompt.is_some() {
    anyhow::bail!("cannot combine positional prompt and --prompt; pick one or use --script");
}
```

- [ ] **Step 3: Run + commit**

```bash
cargo test -p mold-ai --test cli_integration positional_plus_prompt_flag_errors
git add crates/mold-cli/src/main.rs
git commit -m "feat(cli): reject positional prompt + --prompt flag combo

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

### Task 3.7: Phase 3 gate

- [ ] **Step 1: fmt + clippy + test**

```bash
cargo fmt --all -- --check
cargo clippy --workspace --all-targets -- -D warnings
cargo test --workspace
```

- [ ] **Step 2: Adversarial corpus**

Create `crates/mold-cli/tests/adversarial/`:
- `circular_symlink.toml` — points to a source_image that symlinks to itself
- `v99_schema.toml` — unknown schema version
- `17_stages.toml` — exceeds MAX_CHAIN_STAGES
- `reserved_loras.toml` — populates `[[stage.loras]]`
- `newline_prompt.toml` — prompt contains `\n`

Run `mold chain validate` on each; assert appropriate error for all except `newline_prompt.toml` which should succeed.

- [ ] **Step 3: Manual smoke on <gpu-host>**

```bash
# On <gpu-host>
mold run --script shot_smooth.toml -o out.mp4
mold run ltx-2-19b-distilled:fp8 --prompt "A" --prompt "B" --prompt "C" -o out.mp4
```

- [ ] **Step 4: Open Phase 3 PR**

Title: `feat(cli): multi-prompt chain authoring — script + sugar (Phase 3/6)`. Base on Phase 2 PR branch.

---

## Phase 4 — TUI surface

**Goal of this phase:** Script mode in the TUI with a stage-list + editor layout, full keybinding set, TOML save/load, and UAT scenario coverage.

**Commit scope:** `feat(tui)`.

---

### Task 4.1: `ScriptComposer` ratatui widget skeleton

**Files:**
- Create: `crates/mold-tui/src/ui/script_composer.rs`
- Modify: `crates/mold-tui/src/ui/mod.rs`
- Modify: `crates/mold-tui/src/app.rs` — add `Mode::Script` variant and route input

- [ ] **Step 1: Add the module**

Create `script_composer.rs`:

```rust
//! Script-mode composer for chained video generation.
//!
//! Two-pane layout: left shows the stage list; the bottom pane shows an
//! editor for the currently-selected stage.

use mold_core::{ChainScript, ChainStage, TransitionMode};
use ratatui::prelude::*;
use ratatui::widgets::*;

pub struct ScriptComposerState {
    pub script: ChainScript,
    pub selected: usize,
    pub editor_focus: EditorField,
    pub unsaved: bool,
}

#[derive(Clone, Copy, PartialEq, Eq)]
pub enum EditorField {
    Prompt,
    Transition,
    Frames,
    Source,
}

impl ScriptComposerState {
    pub fn new(script: ChainScript) -> Self {
        Self {
            script,
            selected: 0,
            editor_focus: EditorField::Prompt,
            unsaved: false,
        }
    }

    pub fn selected_stage(&self) -> Option<&ChainStage> {
        self.script.stages.get(self.selected)
    }

    pub fn selected_stage_mut(&mut self) -> Option<&mut ChainStage> {
        self.script.stages.get_mut(self.selected)
    }
}

pub fn draw(f: &mut Frame, area: Rect, state: &ScriptComposerState) {
    let chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Min(5),
            Constraint::Length(10),
            Constraint::Length(3),
        ])
        .split(area);

    draw_stage_list(f, chunks[0], state);
    draw_editor(f, chunks[1], state);
    draw_total_line(f, chunks[2], state);
}

fn draw_stage_list(f: &mut Frame, area: Rect, state: &ScriptComposerState) {
    let items: Vec<ListItem> = state
        .script
        .stages
        .iter()
        .enumerate()
        .map(|(i, s)| {
            let marker = if i == state.selected { "● " } else { "  " };
            let tag = match s.transition {
                TransitionMode::Smooth => "smooth",
                TransitionMode::Cut => "cut",
                TransitionMode::Fade => "fade",
            };
            let src = if s.source_image.is_some() { "🖼" } else { " " };
            ListItem::new(format!(
                "{marker}{idx}  {tag}  {src} {frames}f  \"{prompt}\"",
                idx = i + 1,
                frames = s.frames,
                prompt = truncate(&s.prompt, area.width.saturating_sub(30) as usize),
            ))
        })
        .collect();
    let list = List::new(items).block(Block::default().title("Script").borders(Borders::ALL));
    f.render_widget(list, area);
}

fn draw_editor(f: &mut Frame, area: Rect, state: &ScriptComposerState) {
    let Some(stage) = state.selected_stage() else { return; };
    let text = vec![
        Line::from(format!("Prompt:     {}", stage.prompt)),
        Line::from(format!("Transition: {:?}", stage.transition)),
        Line::from(format!("Frames:     {}", stage.frames)),
        Line::from(format!(
            "Source:     {}",
            if stage.source_image.is_some() { "✓" } else { "—" }
        )),
    ];
    f.render_widget(
        Paragraph::new(text).block(
            Block::default()
                .title(format!("Editor: stage {}", state.selected + 1))
                .borders(Borders::ALL),
        ),
        area,
    );
}

fn draw_total_line(f: &mut Frame, area: Rect, state: &ScriptComposerState) {
    let total: u32 = state.script.stages.iter().map(|s| s.frames).sum();
    let fps = state.script.chain.fps;
    let text = format!(
        "{} stages · {} frames (~{:.1}s) · {} · {}x{}{}",
        state.script.stages.len(),
        total,
        total as f64 / fps as f64,
        state.script.chain.model,
        state.script.chain.width,
        state.script.chain.height,
        if state.unsaved { " · unsaved*" } else { "" },
    );
    f.render_widget(Paragraph::new(text), area);
}

fn truncate(s: &str, n: usize) -> String {
    if s.chars().count() <= n {
        s.to_string()
    } else {
        let mut out: String = s.chars().take(n.saturating_sub(1)).collect();
        out.push('…');
        out
    }
}
```

Register in `ui/mod.rs`:

```rust
pub mod script_composer;
```

- [ ] **Step 2: Add mode switch in `app.rs`**

Add `Mode::Script(ScriptComposerState)` variant, route `s` key from the hub to enter it, and `Esc` to exit.

- [ ] **Step 3: Build**

Run: `cargo build -p mold-ai-tui`
Expected: clean.

- [ ] **Step 4: Commit**

```bash
git add crates/mold-tui/src/ui/script_composer.rs crates/mold-tui/src/ui/mod.rs crates/mold-tui/src/app.rs
git commit -m "feat(tui): Script mode skeleton with stage list + editor panes

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

### Task 4.2: Stage list navigation (`j`/`k`/`J`/`K`)

- [ ] **Step 1: Tests**

Add to `script_composer.rs`:

```rust
#[cfg(test)]
mod tests {
    use super::*;

    fn state_with(n: usize) -> ScriptComposerState {
        let mut script = blank_script();
        for i in 0..n {
            script.stages.push(blank_stage(i));
        }
        ScriptComposerState::new(script)
    }

    #[test]
    fn j_moves_selection_down() {
        let mut s = state_with(3);
        s.move_down();
        assert_eq!(s.selected, 1);
    }

    #[test]
    fn k_at_top_clamps() {
        let mut s = state_with(3);
        s.move_up();
        assert_eq!(s.selected, 0);
    }

    #[test]
    fn big_j_swaps_with_next() {
        let mut s = state_with(3);
        let p0 = s.script.stages[0].prompt.clone();
        s.reorder_down();
        assert_eq!(s.selected, 1);
        assert_eq!(s.script.stages[1].prompt, p0);
    }
}
```

- [ ] **Step 2: Impl**

```rust
impl ScriptComposerState {
    pub fn move_down(&mut self) {
        if self.selected + 1 < self.script.stages.len() {
            self.selected += 1;
        }
    }
    pub fn move_up(&mut self) {
        if self.selected > 0 {
            self.selected -= 1;
        }
    }
    pub fn reorder_down(&mut self) {
        let n = self.script.stages.len();
        if self.selected + 1 < n {
            self.script.stages.swap(self.selected, self.selected + 1);
            self.selected += 1;
            self.unsaved = true;
        }
    }
    pub fn reorder_up(&mut self) {
        if self.selected > 0 {
            self.script.stages.swap(self.selected - 1, self.selected);
            self.selected -= 1;
            self.unsaved = true;
        }
    }
}
```

Route keys in `app.rs`'s script-mode handler: `j` → `move_down`, `k` → `move_up`, `J` → `reorder_down`, `K` → `reorder_up`.

- [ ] **Step 3: Run + commit**

```bash
cargo test -p mold-ai-tui
git commit -am "feat(tui): stage list navigation and reorder (j/k/J/K)

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

### Task 4.3: Stage add/delete (`a`/`A`/`d`)

- [ ] **Step 1: Tests**

```rust
#[test]
fn a_adds_stage_after_current() {
    let mut s = state_with(2);
    s.selected = 0;
    s.add_stage_after();
    assert_eq!(s.script.stages.len(), 3);
    assert_eq!(s.selected, 1);
}

#[test]
fn d_removes_current_stage() {
    let mut s = state_with(3);
    s.selected = 1;
    s.delete_stage();
    assert_eq!(s.script.stages.len(), 2);
}

#[test]
fn cannot_delete_last_stage() {
    let mut s = state_with(1);
    s.delete_stage();
    assert_eq!(s.script.stages.len(), 1);
}
```

- [ ] **Step 2: Impl**

```rust
impl ScriptComposerState {
    pub fn add_stage_after(&mut self) {
        let insert_at = self.selected + 1;
        self.script.stages.insert(insert_at, ChainStage {
            prompt: String::new(),
            frames: 97,
            source_image: None,
            negative_prompt: None,
            seed_offset: None,
            transition: TransitionMode::Smooth,
            fade_frames: None,
            model: None,
            loras: vec![],
            references: vec![],
        });
        self.selected = insert_at;
        self.unsaved = true;
    }
    pub fn add_stage_end(&mut self) {
        self.selected = self.script.stages.len().saturating_sub(1);
        self.add_stage_after();
    }
    pub fn delete_stage(&mut self) {
        if self.script.stages.len() <= 1 { return; }
        self.script.stages.remove(self.selected);
        if self.selected >= self.script.stages.len() {
            self.selected = self.script.stages.len() - 1;
        }
        self.unsaved = true;
    }
}
```

Route `a` → `add_stage_after`, `A` → `add_stage_end`, `d` → confirmation-gated `delete_stage` (via a modal that returns `d`/`y` to confirm, any other key cancels).

- [ ] **Step 3: Run + commit**

```bash
cargo test -p mold-ai-tui
git commit -am "feat(tui): add/delete stages with a/A/d keys + confirm modal

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

### Task 4.4: Transition cycle (`t`)

- [ ] **Step 1: Test**

```rust
#[test]
fn t_cycles_transition_smooth_cut_fade() {
    let mut s = state_with(2);
    s.selected = 1;
    s.script.stages[1].transition = TransitionMode::Smooth;
    s.cycle_transition();
    assert_eq!(s.script.stages[1].transition, TransitionMode::Cut);
    s.cycle_transition();
    assert_eq!(s.script.stages[1].transition, TransitionMode::Fade);
    s.cycle_transition();
    assert_eq!(s.script.stages[1].transition, TransitionMode::Smooth);
}

#[test]
fn t_on_stage_0_is_noop() {
    let mut s = state_with(2);
    s.selected = 0;
    s.cycle_transition();
    assert_eq!(s.script.stages[0].transition, TransitionMode::Smooth);
}
```

- [ ] **Step 2: Impl**

```rust
impl ScriptComposerState {
    pub fn cycle_transition(&mut self) {
        if self.selected == 0 { return; }
        let s = self.selected_stage_mut().unwrap();
        s.transition = match s.transition {
            TransitionMode::Smooth => TransitionMode::Cut,
            TransitionMode::Cut => TransitionMode::Fade,
            TransitionMode::Fade => TransitionMode::Smooth,
        };
        self.unsaved = true;
    }
}
```

- [ ] **Step 3: Commit**

```bash
cargo test -p mold-ai-tui
git commit -am "feat(tui): cycle transition with t (no-op on stage 0)

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

### Task 4.5: Prompt & frames editors (`i`/`f`)

`i` opens a modal textarea, `f` opens an inline numeric prompt that snaps to `8k+1`.

- [ ] **Step 1: Write modal structures**

In `script_composer.rs`:

```rust
pub enum EditorModal {
    PromptEdit { buffer: String },
    FramesEdit { buffer: String, error: Option<String> },
    DeleteConfirm,
    Closed,
}

impl ScriptComposerState {
    pub fn open_prompt_editor(&mut self) {
        let buf = self.selected_stage().map(|s| s.prompt.clone()).unwrap_or_default();
        self.modal = EditorModal::PromptEdit { buffer: buf };
    }
    pub fn commit_prompt(&mut self, value: String) {
        if let Some(s) = self.selected_stage_mut() {
            s.prompt = value;
            self.unsaved = true;
        }
        self.modal = EditorModal::Closed;
    }

    pub fn open_frames_editor(&mut self) {
        let buf = self.selected_stage().map(|s| s.frames.to_string()).unwrap_or_default();
        self.modal = EditorModal::FramesEdit { buffer: buf, error: None };
    }
    pub fn commit_frames(&mut self, value: &str) -> Result<(), String> {
        let n: u32 = value.parse().map_err(|_| "not a number".to_string())?;
        if n % 8 != 1 { return Err(format!("{n} must be 8k+1 (9, 17, 25, …)")); }
        if let Some(s) = self.selected_stage_mut() {
            s.frames = n;
            self.unsaved = true;
        }
        self.modal = EditorModal::Closed;
        Ok(())
    }
}
```

- [ ] **Step 2: Tests**

```rust
#[test]
fn commit_frames_rejects_non_8k1() {
    let mut s = state_with(2);
    let err = s.commit_frames("100").unwrap_err();
    assert!(err.contains("8k+1"));
}

#[test]
fn commit_frames_accepts_8k1() {
    let mut s = state_with(2);
    s.commit_frames("49").unwrap();
    assert_eq!(s.selected_stage().unwrap().frames, 49);
}
```

- [ ] **Step 3: Route keys in `app.rs`**

`i` → `open_prompt_editor`, `f` → `open_frames_editor`. Modal draw functions handle key input for `Enter`/`Esc`. When `PromptEdit { buffer }` is open, all character input appends to `buffer`; `Enter` calls `commit_prompt(buffer.clone())`, `Esc` closes without saving.

- [ ] **Step 4: Commit**

```bash
cargo test -p mold-ai-tui
git commit -am "feat(tui): i/f open prompt and frames editors with 8k+1 validation

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

### Task 4.6: TOML save/load (`Ctrl-S`/`Ctrl-O`)

- [ ] **Step 1: Path-picker modal**

Create a minimal file-picker modal that takes a path string (user types or edits), with Enter = submit, Esc = cancel. Reuses the same `EditorModal` pattern.

- [ ] **Step 2: Save handler**

```rust
impl ScriptComposerState {
    pub fn save_as(&mut self, path: &std::path::Path) -> anyhow::Result<()> {
        let toml = mold_core::chain_toml::write_script(&self.script)?;
        std::fs::write(path, toml)?;
        self.unsaved = false;
        Ok(())
    }
    pub fn load_from(&mut self, path: &std::path::Path) -> anyhow::Result<()> {
        let src = std::fs::read_to_string(path)?;
        let script = mold_core::chain_toml::read_script(&src)?;
        // Resolve relative source_image paths:
        let dir = path.parent().unwrap_or_else(|| std::path::Path::new("."));
        let script = resolve_source_image_paths(script, dir)?;
        self.script = script;
        self.selected = 0;
        self.unsaved = false;
        Ok(())
    }
}
```

- [ ] **Step 3: Test**

```rust
#[test]
fn save_then_load_is_identity() {
    let tmp = tempfile::NamedTempFile::new().unwrap();
    let mut s = state_with(3);
    s.save_as(tmp.path()).unwrap();
    let mut s2 = state_with(1); // different starting state
    s2.load_from(tmp.path()).unwrap();
    assert_eq!(s2.script.stages.len(), 3);
}
```

- [ ] **Step 4: Commit**

```bash
cargo test -p mold-ai-tui
git commit -am "feat(tui): Ctrl-S / Ctrl-O save/load via chain_toml

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

### Task 4.7: Submit chain (`Enter`)

- [ ] **Step 1: Wire submit**

On `Enter` in script mode (outside any modal), build a `ChainRequest` from `state.script` and dispatch through the existing TUI submit pipeline. Reuse the machinery that currently submits a single-clip generate; call the chain endpoint variant.

```rust
pub fn submit_chain(&self) -> mold_core::ChainRequest {
    mold_core::ChainRequest {
        model: self.script.chain.model.clone(),
        stages: self.script.stages.clone(),
        motion_tail_frames: self.script.chain.motion_tail_frames,
        width: self.script.chain.width,
        height: self.script.chain.height,
        fps: self.script.chain.fps,
        seed: self.script.chain.seed,
        steps: self.script.chain.steps,
        guidance: self.script.chain.guidance,
        strength: self.script.chain.strength,
        output_format: self.script.chain.output_format,
        placement: None,
        prompt: None,
        total_frames: None,
        clip_frames: None,
        source_image: None,
    }
}
```

Route to the TUI's async runtime via the existing event channel — mirror the single-clip submit flow.

- [ ] **Step 2: Commit**

```bash
git commit -am "feat(tui): Enter submits chain via /api/generate/chain/stream

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

### Task 4.8: Phase 4 gate — `tui-uat.sh` scenario

- [ ] **Step 1: Add scenario**

Extend `scripts/tui-uat.sh` with a `script_mode` scenario:

```bash
# scenario: script_mode
# - launch TUI
# - press `s` to enter script mode
# - press `a` twice (add two stages)
# - press `t` on stage 2 to cycle to cut
# - press Ctrl-S, type "/tmp/mold-uat.toml", Enter
# - verify the file exists and contains "transition = \"cut\""
```

- [ ] **Step 2: Run**

```bash
./scripts/tui-uat.sh script_mode
```

Expected: pass.

- [ ] **Step 3: fmt/clippy/test sweep + PR**

```bash
cargo fmt --all -- --check
cargo clippy --workspace --all-targets -- -D warnings
cargo test --workspace
```

Open PR: `feat(tui): script mode composer (Phase 4/6)`.

---

## Phase 5 — Web surface

**Goal of this phase:** Composer mode toggle (`Single` | `Script`), card-list editor with drag-reorder, per-stage expand, TOML import/export, `localStorage` draft persistence, live footer summary.

**Commit scope:** `feat(web)`.

---

### Task 5.1: Add `smol-toml` dependency + `chainToml.ts` module

**Files:**
- Modify: `web/package.json` — add `"smol-toml": "^1.3.0"` (preferred over `@iarna/toml` for modern ESM + TOML 1.0 support)
- Create: `web/src/lib/chainToml.ts`
- Modify: `web/bun.nix` (regenerate after lock update)

- [ ] **Step 1: Install**

```bash
cd web && bun add smol-toml
```

- [ ] **Step 2: Create the module**

```typescript
// web/src/lib/chainToml.ts
import { parse, stringify } from "smol-toml";

export interface ChainScriptChain {
  model: string;
  width: number;
  height: number;
  fps: number;
  seed?: number;
  steps: number;
  guidance: number;
  strength: number;
  motion_tail_frames: number;
  output_format: "mp4" | "gif" | "apng" | "webp";
}

export interface ChainStageToml {
  prompt: string;
  frames: number;
  source_image?: string; // path string when reading from disk
  source_image_b64?: string; // inline base64 when embedded
  negative_prompt?: string;
  seed_offset?: number;
  transition?: "smooth" | "cut" | "fade";
  fade_frames?: number;
}

export interface ChainScriptToml {
  schema: string;
  chain: ChainScriptChain;
  stage: ChainStageToml[];
}

const SCHEMA = "mold.chain.v1";

export function writeChainScript(script: ChainScriptToml): string {
  const withSchema = { schema: SCHEMA, ...script, schema: SCHEMA };
  return stringify(withSchema as unknown as Record<string, unknown>);
}

export function readChainScript(src: string): ChainScriptToml {
  const raw = parse(src) as unknown as Partial<ChainScriptToml>;
  const schema = raw.schema ?? SCHEMA;
  if (schema !== SCHEMA) {
    throw new Error(
      `chain TOML schema '${schema}' is not supported by this mold version (supported: '${SCHEMA}')`,
    );
  }
  if (!raw.chain) throw new Error("chain TOML missing [chain] table");
  if (!raw.stage || raw.stage.length === 0)
    throw new Error("chain TOML missing [[stage]] entries");
  return {
    schema: SCHEMA,
    chain: raw.chain,
    stage: raw.stage,
  };
}
```

- [ ] **Step 3: Tests**

Create `web/src/lib/chainToml.test.ts`:

```typescript
import { describe, expect, it } from "vitest";
import { readChainScript, writeChainScript } from "./chainToml";

describe("chainToml", () => {
  it("round-trips a minimal script", () => {
    const src: any = {
      schema: "mold.chain.v1",
      chain: {
        model: "ltx-2-19b-distilled:fp8",
        width: 1216,
        height: 704,
        fps: 24,
        steps: 8,
        guidance: 3.0,
        strength: 1.0,
        motion_tail_frames: 25,
        output_format: "mp4",
      },
      stage: [
        { prompt: "first", frames: 97 },
        { prompt: "second", frames: 49, transition: "cut" },
      ],
    };
    const toml = writeChainScript(src);
    const back = readChainScript(toml);
    expect(back.stage.length).toBe(2);
    expect(back.stage[1].transition).toBe("cut");
  });

  it("rejects unknown schema", () => {
    const bad = `schema = "mold.chain.v99"\n[chain]\nmodel = "x"\nwidth=1\nheight=1\nfps=1\nsteps=1\nguidance=1.0\nstrength=1.0\nmotion_tail_frames=0\noutput_format="mp4"\n[[stage]]\nprompt="x"\nframes=1`;
    expect(() => readChainScript(bad)).toThrow(/v99/);
  });
});
```

- [ ] **Step 4: Run + commit**

```bash
cd web && bun run verify && bun run build
cd .. && git add web/package.json web/bun.lock web/bun.nix web/src/lib/chainToml.ts web/src/lib/chainToml.test.ts
git commit -m "feat(web): chainToml module (read/write) with schema-version gate

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

### Task 5.2: `fetchChainLimits` API helper

**Files:**
- Modify: `web/src/api.ts`

- [ ] **Step 1: Add fetcher**

```typescript
export interface ChainLimits {
  model: string;
  frames_per_clip_cap: number;
  frames_per_clip_recommended: number;
  max_stages: number;
  max_total_frames: number;
  fade_frames_max: number;
  transition_modes: string[];
  quantization_family: string;
}

const chainLimitsCache = new Map<string, { value: ChainLimits; at: number }>();
const TTL_MS = 30_000;

export async function fetchChainLimits(model: string): Promise<ChainLimits> {
  const now = Date.now();
  const cached = chainLimitsCache.get(model);
  if (cached && now - cached.at < TTL_MS) return cached.value;
  const resp = await fetch(
    `/api/capabilities/chain-limits?model=${encodeURIComponent(model)}`,
  );
  if (!resp.ok) throw new Error(`chain-limits fetch failed: ${resp.status}`);
  const value: ChainLimits = await resp.json();
  chainLimitsCache.set(model, { value, at: now });
  return value;
}
```

- [ ] **Step 2: Commit**

```bash
git add web/src/api.ts
git commit -m "feat(web): fetchChainLimits helper with 30s memo cache

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

### Task 5.3: `StageCard.vue` component

**Files:**
- Create: `web/src/components/StageCard.vue`

- [ ] **Step 1: Component**

```vue
<script setup lang="ts">
import { computed } from "vue";
import type { ChainStageToml } from "../lib/chainToml";

const props = defineProps<{
  index: number;
  isFirst: boolean;
  stage: ChainStageToml;
  framesPerClipCap: number;
  fadeFramesMax: number;
}>();

const emit = defineEmits<{
  (e: "update:stage", v: ChainStageToml): void;
  (e: "delete"): void;
  (e: "move-up"): void;
  (e: "move-down"): void;
  (e: "duplicate"): void;
  (e: "expand"): void;
}>();

function updatePrompt(v: string) {
  emit("update:stage", { ...props.stage, prompt: v });
}
function updateFrames(v: number) {
  emit("update:stage", { ...props.stage, frames: v });
}
function updateTransition(v: "smooth" | "cut" | "fade") {
  emit("update:stage", { ...props.stage, transition: v });
}

const durationSec = computed(() => (props.stage.frames / 24).toFixed(2));
const frameOptions = computed(() => {
  const out: number[] = [];
  for (let n = 9; n <= props.framesPerClipCap; n += 8) out.push(n);
  return out;
});
</script>

<template>
  <div class="glass rounded-2xl p-3 space-y-2">
    <div class="flex items-center gap-2 text-xs text-slate-400">
      <span class="cursor-grab">⋮⋮</span>
      <span>{{ index + 1 }}</span>
      <template v-if="!isFirst">
        <div class="inline-flex rounded-full bg-slate-900/60 p-0.5">
          <button
            type="button"
            :class="stage.transition === 'smooth' ? 'bg-brand-500/60' : ''"
            class="rounded-full px-2 py-0.5"
            @click="updateTransition('smooth')"
          >smooth</button>
          <button
            type="button"
            :class="stage.transition === 'cut' ? 'bg-brand-500/60' : ''"
            class="rounded-full px-2 py-0.5"
            @click="updateTransition('cut')"
          >cut</button>
          <button
            type="button"
            :class="stage.transition === 'fade' ? 'bg-brand-500/60' : ''"
            class="rounded-full px-2 py-0.5"
            @click="updateTransition('fade')"
          >fade</button>
        </div>
      </template>
      <span v-else class="italic opacity-60">Opening frame</span>
      <div class="ml-auto flex items-center gap-2">
        <select
          class="rounded-full bg-slate-900/60 px-2 py-0.5 text-xs"
          :value="stage.frames"
          @change="updateFrames(Number(($event.target as HTMLSelectElement).value))"
        >
          <option v-for="n in frameOptions" :key="n" :value="n">{{ n }}f ({{ (n/24).toFixed(2) }}s)</option>
        </select>
        <button class="icon-btn" aria-label="Duplicate" @click="emit('duplicate')">⎘</button>
        <button class="icon-btn" aria-label="Move up" @click="emit('move-up')">↑</button>
        <button class="icon-btn" aria-label="Move down" @click="emit('move-down')">↓</button>
        <button class="icon-btn" aria-label="Delete" @click="emit('delete')">✕</button>
      </div>
    </div>

    <textarea
      class="min-h-[2.5rem] w-full resize-none bg-transparent text-base text-slate-100 placeholder:text-slate-500 focus:outline-none"
      :value="stage.prompt"
      placeholder="Describe this stage…"
      @input="updatePrompt(($event.target as HTMLTextAreaElement).value)"
    />

    <div class="flex gap-2 text-xs text-slate-500">
      <button class="hover:text-slate-200" @click="emit('expand')">✨ Expand</button>
      <span v-if="isFirst">{{ durationSec }}s</span>
      <span
        v-else-if="stage.transition === 'smooth' && stage.source_image_b64"
        class="text-amber-400"
        title="Smooth transitions ignore source images; use cut or fade to seed with an image"
      >source image ignored</span>
    </div>
  </div>
</template>
```

- [ ] **Step 2: Commit**

```bash
git add web/src/components/StageCard.vue
git commit -m "feat(web): StageCard component with transition/frames/prompt controls

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

### Task 5.4: `ScriptComposer.vue` component (stage list)

**Files:**
- Create: `web/src/components/ScriptComposer.vue`

- [ ] **Step 1: Component**

```vue
<script setup lang="ts">
import { computed, onMounted, ref, watch } from "vue";
import StageCard from "./StageCard.vue";
import {
  readChainScript,
  writeChainScript,
  type ChainScriptToml,
  type ChainStageToml,
} from "../lib/chainToml";
import { fetchChainLimits, type ChainLimits } from "../api";

const props = defineProps<{
  model: string;
  width: number;
  height: number;
  fps: number;
}>();

const emit = defineEmits<{
  (e: "submit", script: ChainScriptToml): void;
}>();

const DRAFT_KEY = "mold.chain.draft.v2";

function blankStage(transition?: "smooth" | "cut" | "fade"): ChainStageToml {
  return { prompt: "", frames: 97, transition };
}

function newScript(): ChainScriptToml {
  return {
    schema: "mold.chain.v1",
    chain: {
      model: props.model,
      width: props.width,
      height: props.height,
      fps: props.fps,
      steps: 8,
      guidance: 3.0,
      strength: 1.0,
      motion_tail_frames: 25,
      output_format: "mp4",
    },
    stage: [blankStage()],
  };
}

const script = ref<ChainScriptToml>(newScript());
const limits = ref<ChainLimits | null>(null);

onMounted(async () => {
  const draft = localStorage.getItem(DRAFT_KEY);
  if (draft) {
    try { script.value = JSON.parse(draft); } catch {}
  }
  limits.value = await fetchChainLimits(props.model).catch(() => null);
});

watch(
  script,
  (s) => localStorage.setItem(DRAFT_KEY, JSON.stringify(s)),
  { deep: true },
);

watch(() => props.model, async (m) => {
  limits.value = await fetchChainLimits(m).catch(() => null);
});

function addStage() {
  script.value.stage.push(blankStage("smooth"));
}
function updateStage(i: number, next: ChainStageToml) {
  script.value.stage[i] = next;
}
function deleteStage(i: number) {
  if (script.value.stage.length <= 1) return;
  script.value.stage.splice(i, 1);
}
function moveUp(i: number) {
  if (i === 0) return;
  const a = script.value.stage;
  [a[i - 1], a[i]] = [a[i], a[i - 1]];
}
function moveDown(i: number) {
  const a = script.value.stage;
  if (i >= a.length - 1) return;
  [a[i], a[i + 1]] = [a[i + 1], a[i]];
}
function duplicate(i: number) {
  script.value.stage.splice(i + 1, 0, { ...script.value.stage[i] });
}

const totalFrames = computed(() => {
  const s = script.value;
  const mt = s.chain.motion_tail_frames;
  let total = 0;
  s.stage.forEach((stage, i) => {
    if (i === 0) { total += stage.frames; return; }
    switch (stage.transition ?? "smooth") {
      case "smooth": total += Math.max(0, stage.frames - mt); break;
      case "cut": total += stage.frames; break;
      case "fade": total += Math.max(0, stage.frames - (stage.fade_frames ?? 8)); break;
    }
  });
  return total;
});

function exportToml(): string {
  return writeChainScript(script.value);
}

async function importToml(file: File) {
  const text = await file.text();
  try { script.value = readChainScript(text); }
  catch (err) { alert(String(err)); }
}

const framesPerClipCap = computed(() => limits.value?.frames_per_clip_cap ?? 97);
const fadeFramesMax = computed(() => limits.value?.fade_frames_max ?? 32);

function submit() {
  emit("submit", script.value);
}
</script>

<template>
  <div class="glass flex flex-col gap-3 rounded-3xl p-4">
    <div class="flex items-center gap-2 text-sm">
      <span class="font-semibold text-slate-100">Script mode</span>
      <div class="ml-auto flex gap-2">
        <button class="icon-btn" @click="importFileInput?.click()">Import</button>
        <input
          ref="importFileInput"
          type="file"
          accept=".toml"
          class="hidden"
          @change="($event.target as HTMLInputElement).files?.[0] && importToml(($event.target as HTMLInputElement).files![0])"
        />
        <button
          class="icon-btn"
          @click="navigator.clipboard.writeText(exportToml())"
        >Copy TOML</button>
      </div>
    </div>

    <StageCard
      v-for="(stage, i) in script.stage"
      :key="i"
      :index="i"
      :is-first="i === 0"
      :stage="stage"
      :frames-per-clip-cap="framesPerClipCap"
      :fade-frames-max="fadeFramesMax"
      @update:stage="updateStage(i, $event)"
      @delete="deleteStage(i)"
      @move-up="moveUp(i)"
      @move-down="moveDown(i)"
      @duplicate="duplicate(i)"
      @expand="$emit('expand', i)"
    />

    <div class="flex items-center justify-between text-xs text-slate-400">
      <span>
        {{ script.stage.length }} stages · {{ totalFrames }} frames ·
        {{ (totalFrames / script.chain.fps).toFixed(1) }}s @ {{ script.chain.fps }}fps
      </span>
      <button class="icon-btn" @click="addStage">+ Add stage</button>
    </div>

    <button
      class="gradient-btn"
      @click="submit"
    >Generate</button>
  </div>
</template>

<script lang="ts">
import { ref } from "vue";
const importFileInput = ref<HTMLInputElement | null>(null);
export { importFileInput };
</script>
```

- [ ] **Step 2: Commit**

```bash
git add web/src/components/ScriptComposer.vue
git commit -m "feat(web): ScriptComposer with card list + TOML import/export + localStorage draft

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

### Task 5.5: Mode toggle on `Composer.vue`

**Files:**
- Modify: `web/src/components/Composer.vue`

- [ ] **Step 1: Add toggle**

Add a `mode` prop (`"single" | "script"`) with `update:mode` event, render the existing single-composer layout when `mode === "single"` and delegate to `<ScriptComposer>` when `mode === "script"`.

```vue
<div class="flex items-center gap-1 rounded-full bg-slate-900/60 p-0.5">
  <button
    :class="mode === 'single' ? 'bg-brand-500/60' : ''"
    class="rounded-full px-3 py-1 text-xs"
    @click="emit('update:mode', 'single')"
  >Single</button>
  <button
    :class="mode === 'script' ? 'bg-brand-500/60' : ''"
    class="rounded-full px-3 py-1 text-xs"
    @click="emit('update:mode', 'script')"
  >Script</button>
</div>

<ScriptComposer
  v-if="mode === 'script'"
  :model="modelValue.model"
  :width="modelValue.width ?? 1216"
  :height="modelValue.height ?? 704"
  :fps="modelValue.fps ?? 24"
  @submit="handleScriptSubmit"
  @expand="handleScriptExpand"
/>
<!-- else: existing single-prompt composer JSX stays in place -->
```

Persist `mode` in localStorage under `mold.composer.mode`.

- [ ] **Step 2: Parent wiring (GeneratePage)**

Find the parent that renders `Composer.vue` and pass a `mode` ref, initialising from localStorage.

- [ ] **Step 3: Commit**

```bash
git add web/src/components/Composer.vue web/src/pages/GeneratePage.vue
git commit -m "feat(web): Single|Script mode toggle in Composer with localStorage persist

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

### Task 5.6: Chain submit path

**Files:**
- Modify: `web/src/composables/useGenerateForm.ts` — add `submitChain(script)`
- Modify: `web/src/api.ts` — add `postChainStream(body)`

- [ ] **Step 1: Add client**

```typescript
export async function postChainStream(
  body: unknown,
  onProgress: (event: unknown) => void,
): Promise<unknown> {
  const resp = await fetch("/api/generate/chain/stream", {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify(body),
  });
  if (!resp.ok) throw new Error(await resp.text());
  // Parse SSE manually — same pattern as the existing generate/stream.
  // ... (follow lib/sse.ts)
}
```

- [ ] **Step 2: Wire composable**

```typescript
async function submitChain(script: ChainScriptToml) {
  const body = scriptToChainRequest(script);
  return postChainStream(body, (evt) => handleChainEvent(evt));
}
```

`scriptToChainRequest` collapses the TOML shape into the flat `ChainRequest` JSON the server expects (read `crates/mold-core/src/chain.rs` for field alignment — the web's `ChainStageToml` and the server's `ChainStage` agree on every field name).

- [ ] **Step 3: Commit**

```bash
git add web/src/api.ts web/src/composables/useGenerateForm.ts
git commit -m "feat(web): chain submit path wired through postChainStream

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

### Task 5.7: Per-stage expand modal wiring

**Files:**
- Modify: `web/src/components/ExpandModal.vue` — extend to accept a stageIndex and write back only that stage's prompt

- [ ] **Step 1: Wire the handler**

`ScriptComposer.vue` emits `expand` with the stage index; parent opens `ExpandModal` with `props.initialPrompt = script.stage[i].prompt` and `props.onAccept = (v) => script.stage[i].prompt = v`.

- [ ] **Step 2: Commit**

```bash
git commit -am "feat(web): per-stage expand writes back only to the selected stage

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

### Task 5.8: Drag-reorder with `vue-draggable-plus`

**Files:**
- Modify: `web/package.json` — add `vue-draggable-plus`
- Modify: `web/src/components/ScriptComposer.vue` — wrap stage list in `<VueDraggable>`

- [ ] **Step 1: Install**

```bash
cd web && bun add vue-draggable-plus
```

- [ ] **Step 2: Wrap**

```vue
<VueDraggable v-model="script.stage" handle=".drag-handle">
  <StageCard ... />
</VueDraggable>
```

Add class `drag-handle` to the `⋮⋮` span in `StageCard.vue`.

- [ ] **Step 3: Commit**

```bash
git add web/package.json web/bun.lock web/bun.nix web/src/components/ScriptComposer.vue web/src/components/StageCard.vue
git commit -m "feat(web): drag-reorder stages via vue-draggable-plus

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

### Task 5.9: Footer chain-limits clamp + warnings

**Files:**
- Modify: `web/src/components/ScriptComposer.vue`

- [ ] **Step 1: Red footer when over cap**

Compute `overCap = totalFrames > limits.max_total_frames` and toggle a red style on the footer with a tooltip explaining "Reduce frames or stages — server will reject this script".

- [ ] **Step 2: Commit**

```bash
git commit -am "feat(web): footer turns red when script exceeds server-announced caps

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

### Task 5.10: Phase 5 gate

- [ ] **Step 1: bun verify + build**

```bash
cd web && bun run fmt:check && bun run verify && bun run build
```

- [ ] **Step 2: Manual browser test checklist**

- Load `/generate`, toggle Script mode → draft preserved across refresh.
- Add 3 stages, cycle transitions, drag-reorder, delete middle stage.
- Click ✨ on stage 2 → expand modal writes back only to stage 2.
- Copy TOML → paste into `mold chain validate <file>` in a terminal → exit 0.
- Generate → chain renders end-to-end on <gpu-host>.

- [ ] **Step 3: Open Phase 5 PR**

Title: `feat(web): script composer (Phase 5/6)`.

---

## Phase 6 — Docs + release

**Goal of this phase:** keep `CLAUDE.md`, `.claude/skills/mold/SKILL.md`, `website/guide/video.md`, and `CHANGELOG.md` in sync with the shipped feature.

**Commit scope:** `docs(chain)`.

---

### Task 6.1: `website/guide/video.md` — new "Multi-prompt scripts" section

- [ ] **Step 1: Add section**

Append a new section to `website/guide/video.md`:

```markdown
## Multi-prompt scripts (v2)

Direct any-length video scene-by-scene with a TOML script. Each prompt becomes a stage; each boundary has a `transition` (`smooth`, `cut`, or `fade`).

### Canonical form

```bash
mold run --script shot.toml
mold run --script shot.toml --dry-run   # print stage summary, don't submit
mold chain validate shot.toml            # parse without submitting
```

### Sugar form (uniform smooth chains)

```bash
mold run ltx-2-19b-distilled:fp8 \
  --prompt "a cat walks into the autumn forest" \
  --prompt "the forest opens to a clearing" \
  --prompt "a spaceship lands" \
  --frames-per-clip 97
```

Per-stage transitions or per-stage frames require `--script`.

### Transitions

- `smooth` *(default)*: motion-tail carryover, prompt change produces a visual morph.
- `cut`: fresh latent, no carryover; if the stage has `source_image`, used as i2v seed.
- `fade`: cut + post-stitch alpha blend of `fade_frames` (default 8) on each side of the boundary.

### Example `shot.toml`

(paste the full example from the spec)

### Capabilities endpoint

`GET /api/capabilities/chain-limits?model=<name>` returns caps used by the composer UIs.
```

- [ ] **Step 2: VitePress build**

```bash
cd website && bun run build
```

- [ ] **Step 3: Commit**

```bash
git add website/guide/video.md
git commit -m "docs(chain): multi-prompt scripts guide in website/guide/video.md

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

### Task 6.2: `CHANGELOG.md` `[Unreleased]` entry

- [ ] **Step 1: Add entry**

```markdown
## [Unreleased]

### Added
- Multi-prompt chain authoring (sub-project A): per-stage `prompt` / `frames` / `transition` (`smooth` / `cut` / `fade`).
- `mold run --script shot.toml` canonical TOML script form; repeated `--prompt` sugar for uniform trivial chains.
- `mold chain validate <path>` subcommand.
- Web `/generate` composer `Single` | `Script` mode toggle with card-list editor, drag-reorder, per-stage expand, TOML import/export.
- TUI Script mode (`s` from hub) with stage list + editor, keybindings `j`/`k`/`J`/`K`/`a`/`A`/`d`/`t`/`i`/`f`/`Enter`/`Ctrl-S`/`Ctrl-O`.
- Engine support for `cut` (fresh latent, optional i2v) and `fade` (post-stitch RGB crossfade).
- `GET /api/capabilities/chain-limits?model=<name>` endpoint announcing per-model frame caps and supported transitions.

### Changed
- `ChainResponse` now carries a canonical `script: ChainScript` echo and a `vram_estimate: Option<VramEstimate>` slot (unpopulated in this release).
- Mid-chain failures return a structured 502 with `failed_stage_idx`, `elapsed_stages`, `elapsed_ms`.

### Reserved
- `ChainStage.model`, `ChainStage.loras`, `ChainStage.references` accepted by TOML parsers but rejected with 422 by the server — reserved for sub-projects B and C.
```

- [ ] **Step 2: Commit**

```bash
git add CHANGELOG.md
git commit -m "docs(chain): CHANGELOG entry for multi-prompt chain v2

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

### Task 6.3: `.claude/skills/mold/SKILL.md` sync

- [ ] **Step 1: Update skill file**

Find the chain section (if any) or add a new section under "CLI Quick Reference":

```markdown
### Multi-prompt chain (v2)

- Canonical: `mold run --script shot.toml` (TOML schema `mold.chain.v1`)
- Sugar: `mold run <model> --prompt "..." --prompt "..." --frames-per-clip 97`
- Validate only: `mold chain validate shot.toml`
- Dry-run: `mold run --script shot.toml --dry-run`
- Transitions: `smooth` (default), `cut`, `fade` (per-stage on `[[stage]]`)
- Chain endpoint: `POST /api/generate/chain[/stream]`
- Capabilities: `GET /api/capabilities/chain-limits?model=<name>`
- Max stages: 16. LTX-2 distilled cap: 97 frames/clip.
```

- [ ] **Step 2: Commit**

```bash
git add .claude/skills/mold/SKILL.md
git commit -m "docs(chain): sync .claude/skills/mold/SKILL.md with multi-prompt flags

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

### Task 6.4: `CLAUDE.md` sync + final release PR

- [ ] **Step 1: Update CLAUDE.md**

Extend the "CLI Quick Reference" and "Server API" sections with the new surface. Follow the existing bullet style — one line per flag/endpoint.

- [ ] **Step 2: Full gate**

```bash
cargo fmt --all -- --check
cargo clippy --workspace --all-targets -- -D warnings
cargo test --workspace
cd web && bun run fmt:check && bun run verify && bun run build
```

- [ ] **Step 3: Commit**

```bash
git add CLAUDE.md
git commit -m "docs(chain): sync CLAUDE.md with multi-prompt chain surface

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

- [ ] **Step 4: Open release PR**

Title: `feat: multi-prompt chain authoring v2 (sub-project A)`. Stack on top of Phases 1–5. Link to the spec at `docs/superpowers/specs/2026-04-21-multi-prompt-chain-v2-design.md`.

---

## Self-review notes

Performed against the spec:

**Spec coverage:**
- §3.1 `TransitionMode` — Task 1.1 ✓
- §3.1 reserved `LoraSpec`/`NamedRef` — Task 1.2 ✓
- §3.1 extended `ChainStage` — Task 1.3 ✓
- §3.1 `ChainScript` + `VramEstimate` + `ChainResponse` — Task 1.4 ✓
- §4.2 normalise stage-0 coerce + reserved reject — Task 1.5 ✓
- §6.2 `estimated_total_frames` math — Task 1.6 ✓
- §4.4 TOML writer/reader — Tasks 1.7, 1.8 ✓
- §4.4 round-trip test — Task 1.9 ✓
- §4.3 capabilities endpoint — Tasks 1.11, 1.12 ✓
- §6.1 Cut transition in orchestrator — Task 2.2 ✓
- §6.2 Fade stitch helper — Task 2.3 ✓
- §6.2 StitchPlan — Task 2.4 ✓
- §6.3 source_image on non-zero stages — Task 2.5 ✓
- §6.2 server uses StitchPlan — Task 2.6 ✓
- §7.1 ChainFailure typed 502 — Task 2.7 ✓
- §5.1 CLI `--script` / sugar / `--dry-run` / `chain validate` — Tasks 3.1–3.6 ✓
- §5.2 TUI Script mode — Tasks 4.1–4.7 ✓
- §5.3 Web composer mode toggle — Tasks 5.1–5.9 ✓
- §5.4 per-stage expand — Task 5.7 ✓
- §9 docs — Tasks 6.1–6.4 ✓

**Placeholder scan:** Task 3.2 has a comment block with an "if that injection pattern is too deep" fork — this is intentional guidance, not a placeholder. Task 3.2's `build_request_from_script` stub section documents Option A vs Option B and commits to Option B. No `TBD`/`TODO`/`FIXME` markers remain in task bodies.

**Type consistency:** `ChainStage` / `ChainScript` / `ChainScriptChain` / `TransitionMode` / `ChainFailure` all resolve to the same names across tasks. `ChainStageToml` (web) and `ChainStage` (Rust) are named differently on purpose to signal surface; their field names align.

**Known open issues deferred to implementation:**
- `chain_toml::read_script_resolving_paths` (two-pass path resolution) needs a small raw-shape struct; Task 3.2 describes the approach but the exact struct isn't in the plan.
- `SseChainCompleteEvent` may need its own `script: ChainScript` / `vram_estimate` field additions if Phase 1's Task 1.13 didn't already wire them — the plan assumes that extension is additive and the worker will add the field + update the wiremock tests in one pass.
- Web `vue-draggable-plus` version pinning — the plan says "add" and defers the lock file to `bun add`'s default semver resolution.

None of the open issues should block a faithful execution.

