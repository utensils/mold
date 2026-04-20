# Device Placement Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Let users override which device (CPU or a specific GPU ordinal) runs text encoders, transformer, and VAE for any model, with per-component placement exposed for FLUX / Flux.2 / Z-Image / Qwen-Image (SD3.5 stretch) and Tier 1 grouped-encoder placement for every other family.

**Architecture:** New `DevicePlacement` and `DeviceRef` serde types live in `mold-core::types`. A shared `resolve_device()` helper in `mold-inference::device` maps each `Option<DeviceRef>` to a concrete `candle_core::Device`, defaulting to the existing VRAM-aware auto-placement. Each engine's `load()` path reads the request's optional `placement` field and plumbs per-component overrides through to component loading sites. Config adds a `[models."name:tag".placement]` table plus `MOLD_PLACE_*` env overrides; a new `PUT /api/config/model/:name/placement` route persists the "Save as default" action. The SPA exposes a `PlacementPanel.vue` disclosure inside `Composer.vue`, driven by a `usePlacement.ts` composable that gates the Advanced section against a family allow-list.

**Tech Stack:** Rust, candle (CUDA/Metal), axum, serde, Vue 3, Vitest, bun

**Spec:** `docs/superpowers/specs/2026-04-19-model-ui-overhaul-design.md` — sections §3, §4.1, §4.2, §4.3

**Worktree:** This plan should be executed in `wt/device-placement` off branch `feat/model-ui-overhaul`. Create the worktree with `git worktree add ../mold-placement wt/device-placement feat/model-ui-overhaul` before starting Task 1.

**Agent boundary:** You own `DevicePlacement`, the per-engine plumbing, and `PlacementPanel.vue`. Other agents own downloads (§1) and resources (§2). Do NOT touch `/api/downloads*` or `/api/resources*`. Your UI consumes the GPU list via `useResources().gpuList` from Agent B — until that merges, stub with a hardcoded list (marked clearly with a `TODO-REMOVE-AFTER-MERGE` comment in `usePlacement.ts`). Keep changes to shared files (`types.rs`, `routes.rs`, `config.rs`, `Composer.vue`, `types.ts`) confined to clearly-labeled additive sections — see spec §4.2.

---

## File Map

### New Files

| File | Responsibility |
|------|----------------|
| `crates/mold-core/src/placement_test.rs` | Serde round-trip tests for `DeviceRef` / `DevicePlacement` / `AdvancedPlacement`. Included from `types.rs` via `#[cfg(test)] mod placement_test;`. |
| `crates/mold-inference/tests/device_resolve_test.rs` | Integration-style tests for `resolve_device()` (lives under `tests/` so the test binary only exercises the public API without pulling candle's full model-loading stack). |
| `web/src/composables/usePlacement.ts` | Per-model `DevicePlacement` state, `supportsAdvanced(family)` gate, saved-default merge. |
| `web/src/composables/usePlacement.test.ts` | Vitest coverage for the composable. |
| `web/src/components/PlacementPanel.vue` | Disclosure rendered inside `Composer.vue`: Tier 1 select plus Advanced fieldset (gated) plus "Save as default" button. |
| `web/src/components/PlacementPanel.test.ts` | Mount test: asserts Advanced is disabled for SDXL and enabled for FLUX; renders the GPU list; emits updates correctly. |

### Modified Files

| File | What Changes |
|------|--------------|
| `crates/mold-core/src/types.rs` | Add `DeviceRef`, `DevicePlacement`, `AdvancedPlacement`. Add optional `placement: Option<DevicePlacement>` to `GenerateRequest`. |
| `crates/mold-core/src/config.rs` | Add `placement: Option<DevicePlacement>` to `ModelConfig`. Add env overrides (`MOLD_PLACE_TEXT_ENCODERS`, `MOLD_PLACE_TRANSFORMER`, `MOLD_PLACE_VAE`). Expose `Config::set_model_placement(name, p)` helper. |
| `crates/mold-inference/src/device.rs` | Add `resolve_device(override, auto)` helper. |
| `crates/mold-inference/src/flux/pipeline.rs` | Accept placement overrides in `load()` (transformer, VAE, T5, CLIP-L). |
| `crates/mold-inference/src/flux2/pipeline.rs` | Same for Qwen3, transformer, VAE. |
| `crates/mold-inference/src/zimage/pipeline.rs` | Same for Qwen3, transformer, VAE. |
| `crates/mold-inference/src/qwen_image/pipeline.rs` | Same for Qwen2.5-VL, transformer, VAE. |
| `crates/mold-inference/src/sd3/pipeline.rs` | **Stretch** — CLIP-L, CLIP-G, T5-XXL, MMDiT, VAE. Cut Task 14 if behind. |
| `crates/mold-inference/src/sd15/pipeline.rs` | Tier 1 only: honor `text_encoders` override for CLIP-L. |
| `crates/mold-inference/src/sdxl/pipeline.rs` | Tier 1 only: honor `text_encoders` override for CLIP-L plus CLIP-G (as a group). |
| `crates/mold-inference/src/wuerstchen/pipeline.rs` | Tier 1 only: honor `text_encoders` for CLIP-G. |
| `crates/mold-inference/src/ltx_video/pipeline.rs` | Tier 1 only: honor `text_encoders` for T5. |
| `crates/mold-inference/src/ltx2/pipeline.rs` | Tier 1 only: honor `text_encoders` for Gemma3 (CUDA-only; on Metal it's still a no-op). |
| `crates/mold-server/src/routes.rs` | Add `PUT /api/config/model/:name/placement` handler and router wiring. Accept `placement` on `/api/generate` (already flows through because it's on `GenerateRequest`). |
| `crates/mold-server/src/routes_test.rs` | Coverage for the new PUT route (happy path plus 404). |
| `web/src/types.ts` | Mirror `DeviceRef` / `DevicePlacement` / `AdvancedPlacement`. Add `placement?: DevicePlacement | null` to `GenerateRequestWire` and `placement?` on `GenerateFormState`. |
| `web/src/composables/useGenerateForm.ts` | Carry `placement` on form state and include in `toRequest()`. |
| `web/src/components/Composer.vue` | Mount `PlacementPanel` below the existing settings affordance (clearly-labeled additive section). |

---

## Task 1: Add `DeviceRef` serde enum to `mold-core::types`

**File:** `crates/mold-core/src/types.rs` (additive, appended near the existing `GpuSelection` block around line 738). Test file: `crates/mold-core/src/placement_test.rs` (new, included under `#[cfg(test)]`).

- [ ] **Step 1: Write the failing round-trip test**

Create `crates/mold-core/src/placement_test.rs` with the full body below. Then register it in `crates/mold-core/src/types.rs` by appending at the very end of the file:

```rust
#[cfg(test)]
#[path = "placement_test.rs"]
mod placement_test;
```

Test body (new file):

```rust
//! Serde round-trip plus default tests for `DeviceRef`, `DevicePlacement`,
//! `AdvancedPlacement`. Kept in a sibling file because `types.rs` is already
//! 2100+ lines.
use super::{AdvancedPlacement, DevicePlacement, DeviceRef};

#[test]
fn device_ref_auto_default() {
    assert_eq!(DeviceRef::default(), DeviceRef::Auto);
}

#[test]
fn device_ref_auto_round_trip() {
    let json = serde_json::to_string(&DeviceRef::Auto).unwrap();
    assert_eq!(json, r#"{"kind":"auto"}"#);
    let back: DeviceRef = serde_json::from_str(&json).unwrap();
    assert_eq!(back, DeviceRef::Auto);
}

#[test]
fn device_ref_cpu_round_trip() {
    let json = serde_json::to_string(&DeviceRef::Cpu).unwrap();
    assert_eq!(json, r#"{"kind":"cpu"}"#);
    let back: DeviceRef = serde_json::from_str(&json).unwrap();
    assert_eq!(back, DeviceRef::Cpu);
}

#[test]
fn device_ref_gpu_round_trip() {
    let json = serde_json::to_string(&DeviceRef::gpu(2)).unwrap();
    assert_eq!(json, r#"{"kind":"gpu","ordinal":2}"#);
    let back: DeviceRef = serde_json::from_str(&json).unwrap();
    assert_eq!(back, DeviceRef::gpu(2));
}

#[test]
fn device_placement_defaults_to_all_auto() {
    let dp = DevicePlacement::default();
    assert_eq!(dp.text_encoders, DeviceRef::Auto);
    assert!(dp.advanced.is_none());
}

#[test]
fn device_placement_serializes_tier1_only_without_advanced() {
    let dp = DevicePlacement {
        text_encoders: DeviceRef::Cpu,
        advanced: None,
    };
    let json = serde_json::to_value(&dp).unwrap();
    assert_eq!(json["text_encoders"]["kind"], "cpu");
    assert!(json.get("advanced").is_none() || json["advanced"].is_null());
}

#[test]
fn device_placement_round_trip_with_advanced() {
    let dp = DevicePlacement {
        text_encoders: DeviceRef::gpu(0),
        advanced: Some(AdvancedPlacement {
            transformer: DeviceRef::gpu(1),
            vae: DeviceRef::Cpu,
            clip_l: Some(DeviceRef::Auto),
            clip_g: None,
            t5: Some(DeviceRef::gpu(0)),
            qwen: None,
        }),
    };
    let json = serde_json::to_string(&dp).unwrap();
    let back: DevicePlacement = serde_json::from_str(&json).unwrap();
    assert_eq!(back.text_encoders, DeviceRef::gpu(0));
    let adv = back.advanced.unwrap();
    assert_eq!(adv.transformer, DeviceRef::gpu(1));
    assert_eq!(adv.vae, DeviceRef::Cpu);
    assert_eq!(adv.clip_l, Some(DeviceRef::Auto));
    assert_eq!(adv.clip_g, None);
    assert_eq!(adv.t5, Some(DeviceRef::gpu(0)));
}

#[test]
fn advanced_placement_defaults_to_auto_pair() {
    let adv = AdvancedPlacement::default();
    assert_eq!(adv.transformer, DeviceRef::Auto);
    assert_eq!(adv.vae, DeviceRef::Auto);
    assert!(adv.clip_l.is_none());
    assert!(adv.clip_g.is_none());
    assert!(adv.t5.is_none());
    assert!(adv.qwen.is_none());
}
```

- [ ] **Step 2: Run the failing test**

```
cargo test -p mold-ai-core placement_test --lib
```

Expected: **fail** — `DeviceRef` / `DevicePlacement` / `AdvancedPlacement` do not yet exist. Actual error will be "cannot find type `DeviceRef` in module `super`" or similar.

- [ ] **Step 3: Implement `DeviceRef` plus `DevicePlacement` plus `AdvancedPlacement`**

Append the following block inside `crates/mold-core/src/types.rs` directly after the existing `GpuWorkerState` enum (around line 785, before the SSE streaming wire types comment). This is the "additive, separate section" called out in spec §4.2.

```rust
// ── Device placement (Agent C: model-ui-overhaul §3) ─────────────────────────

/// A user-facing request for where a component should run.
///
/// - `Auto` preserves the existing VRAM-aware auto-placement logic.
/// - `Cpu` pins the component to CPU regardless of available VRAM.
/// - `Gpu { ordinal }` pins to GPU ordinal `n` (CUDA-specific ordinal,
///   or `0` on Metal/unified memory).
///
/// Serialized as an externally-tagged enum: `{"kind":"auto"}`,
/// `{"kind":"cpu"}`, or `{"kind":"gpu","ordinal":1}`. A missing `DeviceRef`
/// field deserializes to `Auto`.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum DeviceRef {
    Auto,
    Cpu,
    Gpu { ordinal: usize },
}

impl Default for DeviceRef {
    fn default() -> Self {
        DeviceRef::Auto
    }
}

impl DeviceRef {
    /// Helper constructor mirroring the compact `Gpu(n)` form used in tests.
    pub const fn gpu(ordinal: usize) -> Self {
        DeviceRef::Gpu { ordinal }
    }
}

/// Top-level placement request attached to `GenerateRequest` and persisted
/// under `[models."name:tag".placement]` in config.
///
/// - `text_encoders` is the Tier 1 "group knob" — a single override applied
///   to all text-encoder components (T5 plus CLIP-L, Qwen3, Qwen2.5-VL, etc.).
/// - `advanced` is Tier 2, available only for families listed in spec §3.2
///   (FLUX, Flux.2, Z-Image, Qwen-Image; SD3.5 stretch). When `Some`, each
///   populated field overrides the Tier 1 group knob for that specific
///   component. When `None`, only the group knob is honored.
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq, Eq)]
pub struct DevicePlacement {
    #[serde(default)]
    pub text_encoders: DeviceRef,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub advanced: Option<AdvancedPlacement>,
}

/// Per-component placement overrides available only for Tier 2 families.
///
/// `transformer` and `vae` are required fields (default `Auto`). The
/// per-encoder fields are `Option<DeviceRef>` because not every family has
/// every encoder — FLUX has T5 plus CLIP-L, Flux.2 and Z-Image have Qwen3,
/// Qwen-Image has Qwen2.5-VL. `None` means "follow the Tier 1 group knob";
/// `Some(DeviceRef::Auto)` means "follow the engine's own auto logic".
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq, Eq)]
pub struct AdvancedPlacement {
    #[serde(default)]
    pub transformer: DeviceRef,
    #[serde(default)]
    pub vae: DeviceRef,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub clip_l: Option<DeviceRef>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub clip_g: Option<DeviceRef>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub t5: Option<DeviceRef>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub qwen: Option<DeviceRef>,
}
```

- [ ] **Step 4: Run the test — passes**

```
cargo test -p mold-ai-core placement_test --lib
```

Expected output:

```
running 8 tests
test placement_test::device_ref_auto_default ... ok
test placement_test::device_ref_auto_round_trip ... ok
test placement_test::device_ref_cpu_round_trip ... ok
test placement_test::device_ref_gpu_round_trip ... ok
test placement_test::device_placement_defaults_to_all_auto ... ok
test placement_test::device_placement_serializes_tier1_only_without_advanced ... ok
test placement_test::device_placement_round_trip_with_advanced ... ok
test placement_test::advanced_placement_defaults_to_auto_pair ... ok

test result: ok. 8 passed; 0 failed
```

- [ ] **Step 5: Commit**

```
git add crates/mold-core/src/types.rs crates/mold-core/src/placement_test.rs
git commit -m "feat(placement): add DeviceRef, DevicePlacement, AdvancedPlacement serde types

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 2: Attach optional `placement` to `GenerateRequest`

**File:** `crates/mold-core/src/types.rs`

- [ ] **Step 1: Write the failing wire test**

Append to `crates/mold-core/src/placement_test.rs`:

```rust
#[test]
fn generate_request_placement_round_trips() {
    use super::GenerateRequest;
    let req = GenerateRequest {
        prompt: "a cat".into(),
        negative_prompt: None,
        model: "flux-dev:q4".into(),
        width: 1024,
        height: 1024,
        steps: 20,
        guidance: 3.5,
        seed: Some(7),
        batch_size: 1,
        output_format: super::OutputFormat::Png,
        embed_metadata: None,
        scheduler: None,
        source_image: None,
        edit_images: None,
        strength: 0.75,
        mask_image: None,
        control_image: None,
        control_model: None,
        control_scale: 1.0,
        expand: None,
        original_prompt: None,
        lora: None,
        frames: None,
        fps: None,
        upscale_model: None,
        gif_preview: false,
        enable_audio: None,
        audio_file: None,
        source_video: None,
        keyframes: None,
        pipeline: None,
        loras: None,
        retake_range: None,
        spatial_upscale: None,
        temporal_upscale: None,
        placement: Some(DevicePlacement {
            text_encoders: DeviceRef::Cpu,
            advanced: Some(AdvancedPlacement {
                transformer: DeviceRef::gpu(1),
                vae: DeviceRef::Auto,
                t5: Some(DeviceRef::Cpu),
                ..Default::default()
            }),
        }),
    };
    let json = serde_json::to_string(&req).unwrap();
    let back: GenerateRequest = serde_json::from_str(&json).unwrap();
    let p = back.placement.unwrap();
    assert_eq!(p.text_encoders, DeviceRef::Cpu);
    let adv = p.advanced.unwrap();
    assert_eq!(adv.transformer, DeviceRef::gpu(1));
    assert_eq!(adv.t5, Some(DeviceRef::Cpu));
}

#[test]
fn generate_request_without_placement_is_none() {
    use super::GenerateRequest;
    let json = r#"{
        "prompt": "a cat",
        "model": "flux-dev:q4",
        "width": 1024,
        "height": 1024,
        "steps": 20,
        "guidance": 3.5,
        "batch_size": 1,
        "strength": 0.75
    }"#;
    let req: GenerateRequest = serde_json::from_str(json).unwrap();
    assert!(req.placement.is_none());
}
```

- [ ] **Step 2: Run — fail**

```
cargo test -p mold-ai-core placement_test::generate_request --lib
```

Expected: **fail** — `GenerateRequest` has no `placement` field.

- [ ] **Step 3: Add the field**

In `crates/mold-core/src/types.rs`, append to the `GenerateRequest` struct (immediately after the existing `temporal_upscale` field, roughly at line 331):

```rust
    /// Optional per-component device placement override. `None` preserves
    /// the engine's VRAM-aware auto-placement end-to-end. See §3 of the
    /// 2026-04-19 model-ui-overhaul design doc.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub placement: Option<DevicePlacement>,
```

Then update the existing `GenerateRequest` tests lower in the file that construct the struct — they currently list every field explicitly (search for `let req = GenerateRequest {` occurrences; there are about 10 in the tests module). Add `placement: None,` to every literal.

A concrete edit target list (from the existing grep of lines with `GenerateRequest {`): lines 972, 1128, 1174, 1217, 1262, 1305, 1516, 1565, 1627, 1674, 1742 — append `placement: None,` before the closing brace on each.

- [ ] **Step 4: Run — passes**

```
cargo test -p mold-ai-core placement_test --lib
cargo test -p mold-ai-core --lib
```

Expected: both all-green. The second command confirms no existing `GenerateRequest` test regressed.

- [ ] **Step 5: Commit**

```
git add crates/mold-core/src/types.rs crates/mold-core/src/placement_test.rs
git commit -m "feat(placement): add optional placement field to GenerateRequest

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 3: `resolve_device()` helper in `mold-inference::device`

**Files:**
- Modify: `crates/mold-inference/src/device.rs`
- Create: `crates/mold-inference/tests/device_resolve_test.rs`

> Why `tests/` rather than `#[cfg(test)]` inline? Integration tests under `tests/` compile as separate binaries and only exercise the public API — they don't pull candle's full model-loading stack the way a big inline unit test module could. (Note: CLAUDE.md claims `mold-inference` has `[lib] test = false`, but that flag has already been removed from `Cargo.toml`; either form of test now compiles, but `tests/` stays tidier.)

- [ ] **Step 1: Write the failing test**

Create `crates/mold-inference/tests/device_resolve_test.rs`:

```rust
//! Coverage for `mold_inference::device::resolve_device`. Runs as an
//! integration test — keeps the test binary small and avoids pulling candle's
//! full model-loading stack.

use mold_core::types::DeviceRef;
use mold_inference::device::resolve_device;

fn cpu_auto() -> anyhow::Result<candle_core::Device> {
    Ok(candle_core::Device::Cpu)
}

#[test]
fn resolve_device_none_calls_auto() {
    let dev = resolve_device(None, cpu_auto).unwrap();
    assert!(matches!(dev, candle_core::Device::Cpu));
}

#[test]
fn resolve_device_auto_calls_auto() {
    let dev = resolve_device(Some(DeviceRef::Auto), cpu_auto).unwrap();
    assert!(matches!(dev, candle_core::Device::Cpu));
}

#[test]
fn resolve_device_cpu_bypasses_auto() {
    let called = std::sync::atomic::AtomicBool::new(false);
    let dev = resolve_device(Some(DeviceRef::Cpu), || {
        called.store(true, std::sync::atomic::Ordering::SeqCst);
        cpu_auto()
    })
    .unwrap();
    assert!(matches!(dev, candle_core::Device::Cpu));
    assert!(
        !called.load(std::sync::atomic::Ordering::SeqCst),
        "Cpu override must not invoke the auto closure"
    );
}

#[test]
#[cfg(not(any(feature = "cuda", feature = "metal")))]
fn resolve_device_gpu_on_cpu_only_host_errors_clearly() {
    let err = resolve_device(Some(DeviceRef::gpu(0)), cpu_auto).unwrap_err();
    let msg = err.to_string();
    assert!(
        msg.contains("GPU") || msg.contains("cuda") || msg.contains("metal"),
        "error should mention the missing backend, got: {msg}"
    );
}
```

- [ ] **Step 2: Run — fail**

```
cargo test -p mold-ai-inference --test device_resolve_test
```

Expected: compilation failure — `resolve_device` does not yet exist.

- [ ] **Step 3: Implement `resolve_device()`**

In `crates/mold-inference/src/device.rs`, append a new section before the macOS memory query banner (roughly at line 160):

```rust
// ── Placement resolution ─────────────────────────────────────────────────────

/// Resolve a caller-supplied `DeviceRef` override into a concrete candle
/// `Device`, falling back to `auto` when the override is missing or `Auto`.
///
/// - `None`, `Some(Auto)` — call `auto()` (existing VRAM-aware logic).
/// - `Some(Cpu)`          — `Device::Cpu`, never invoke `auto()`.
/// - `Some(Gpu { ordinal })` — try CUDA first, then Metal. Each backend is
///   gated by its candle feature flag so a CPU-only build returns a clear
///   error message instead of a build failure.
pub fn resolve_device<F>(
    req: Option<mold_core::types::DeviceRef>,
    auto: F,
) -> anyhow::Result<candle_core::Device>
where
    F: FnOnce() -> anyhow::Result<candle_core::Device>,
{
    use mold_core::types::DeviceRef;
    match req {
        None | Some(DeviceRef::Auto) => auto(),
        Some(DeviceRef::Cpu) => Ok(candle_core::Device::Cpu),
        Some(DeviceRef::Gpu { ordinal }) => resolve_gpu_ordinal(ordinal),
    }
}

#[cfg(feature = "cuda")]
fn resolve_gpu_ordinal(ordinal: usize) -> anyhow::Result<candle_core::Device> {
    candle_core::Device::new_cuda(ordinal)
        .map_err(|e| anyhow::anyhow!("failed to open CUDA device {ordinal}: {e}"))
}

#[cfg(all(not(feature = "cuda"), feature = "metal"))]
fn resolve_gpu_ordinal(ordinal: usize) -> anyhow::Result<candle_core::Device> {
    candle_core::Device::new_metal(ordinal)
        .map_err(|e| anyhow::anyhow!("failed to open Metal device {ordinal}: {e}"))
}

#[cfg(all(not(feature = "cuda"), not(feature = "metal")))]
fn resolve_gpu_ordinal(ordinal: usize) -> anyhow::Result<candle_core::Device> {
    Err(anyhow::anyhow!(
        "GPU ordinal {ordinal} requested but this build has neither CUDA nor Metal enabled"
    ))
}
```

- [ ] **Step 4: Run — passes**

```
cargo test -p mold-ai-inference --test device_resolve_test
```

Expected:

```
running 3 tests
test resolve_device_none_calls_auto ... ok
test resolve_device_auto_calls_auto ... ok
test resolve_device_cpu_bypasses_auto ... ok
```

The `resolve_device_gpu_on_cpu_only_host_errors_clearly` test only compiles when both feature flags are off, which is not the default devshell configuration; that is fine — it exists as documentation that the clear-error path is tested by CI's `cargo check --workspace` build profile.

- [ ] **Step 5: Commit**

```
git add crates/mold-inference/src/device.rs crates/mold-inference/tests/device_resolve_test.rs
git commit -m "feat(placement): add resolve_device helper that honors DeviceRef overrides

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 4: Thread `placement` into the `GenerateRequest` path (Tier 1 FLUX)

**Files:** `crates/mold-inference/src/flux/pipeline.rs`

This is the first engine to consume the new field. Tier 2 plumbing (transformer plus VAE per-component overrides) is deferred to Task 10; Task 4 establishes the pattern and lands Tier 1 (`placement.text_encoders`) for FLUX — the simplest payoff.

- [ ] **Step 1: Failing signal**

Engine `load()` paths exercise candle's full model-loading stack, so an engine-level unit test here would drag in multi-GB weight-loading infra we don't want in CI. Behavior gets covered implicitly by the Tier 1 UI test (Task 18) and the `resolve_device` unit tests (Task 3). For this task the "failing test" is a compile-time check: the struct change in Step 2 below will not compile without Step 3's usage in `generate()`.

- [ ] **Step 2: Add placement plumbing to `FluxEngine`**

In `crates/mold-inference/src/flux/pipeline.rs`, add a field to the engine struct (find the struct `FluxEngine` declaration, look for the existing `shared_pool` field around line 651):

```rust
pub struct FluxEngine {
    // ... existing fields ...
    shared_pool: Option<Arc<Mutex<crate::shared_pool::SharedPool>>>,
    /// Per-request placement override. Set at the start of `generate()`,
    /// cleared on exit. `None` preserves the existing VRAM-aware auto logic.
    pending_placement: Option<mold_core::types::DevicePlacement>,
}
```

Initialize it in `new()` (line ~669) by adding `pending_placement: None,` to the struct literal.

Change `FluxEngine::generate()` to save and clear the field:

```rust
fn generate(&mut self, req: &GenerateRequest) -> Result<GenerateResponse> {
    self.pending_placement = req.placement.clone();
    let result = self.generate_impl(req);
    self.pending_placement = None;
    result
}
```

…where `generate_impl()` is the existing body renamed. If the existing body is compact, an alternative is to inline save/restore via a stack variable at each return site — but the two-call pattern above is the minimal diff.

- [ ] **Step 3: Honor `placement.text_encoders` when placing T5 plus CLIP-L**

Inside `FluxEngine::load()` (around line 965-1011 — where `t5_on_gpu` and `clip_on_gpu` are computed), replace:

```rust
let t5_device = if t5_on_gpu { &device } else { &cpu };
```

with:

```rust
// Tier 1: a `text_encoders` group override replaces the VRAM-aware decision.
// Tier 2 (Task 10) adds per-encoder overrides via `placement.advanced.t5`.
let tier1 = self
    .pending_placement
    .as_ref()
    .map(|p| p.text_encoders)
    .unwrap_or_default();
let auto_t5_device = if t5_on_gpu { device.clone() } else { cpu.clone() };
let t5_device_owned = crate::device::resolve_device(
    Some(tier1),
    || Ok(auto_t5_device.clone()),
)?;
let t5_device = &t5_device_owned;
let t5_device_label = if t5_device.is_cpu() { "CPU" } else { "GPU" };
```

Repeat the same pattern for `clip_device` a few lines below:

```rust
let auto_clip_device = if clip_on_gpu { device.clone() } else { cpu.clone() };
let clip_device_owned = crate::device::resolve_device(
    Some(tier1),
    || Ok(auto_clip_device.clone()),
)?;
let clip_device = &clip_device_owned;
let clip_device_label = if clip_device.is_cpu() { "CPU" } else { "GPU" };
```

Also repeat the same changes in `FluxEngine::generate_sequential()` — look for the second copy of the same T5/CLIP placement around line 1142-1203.

The `Device` type implements `Clone` on candle-core 0.8+, which the workspace already pins to. If clippy complains about unnecessary `.clone()` for the Metal `Device::Cpu` branch, keep it — the borrow checker needs owned values to bridge the `if` branches.

- [ ] **Step 4: Verify build and existing tests stay green**

```
cargo check -p mold-ai-inference
cargo test -p mold-ai-core --lib
cargo test -p mold-ai-inference --test device_resolve_test
```

Expected: all green. No Tier 1 behavior test yet — that lands with the UI composable in Task 18.

- [ ] **Step 5: Commit**

```
git add crates/mold-inference/src/flux/pipeline.rs
git commit -m "feat(placement): honor Tier 1 text_encoders override in FLUX engine

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 5: Tier 1 plumbing — Flux.2

**Files:** `crates/mold-inference/src/flux2/pipeline.rs`

- [ ] **Step 1: Add `pending_placement` field to `Flux2Engine`**

Same pattern as Task 4. Find the `pub struct Flux2Engine` declaration, append:

```rust
pending_placement: Option<mold_core::types::DevicePlacement>,
```

Initialize in `new()`.

- [ ] **Step 2: Save/clear in `generate()`**

Same two-call wrapper as Task 4:

```rust
fn generate(&mut self, req: &GenerateRequest) -> Result<GenerateResponse> {
    self.pending_placement = req.placement.clone();
    let result = self.generate_impl(req);
    self.pending_placement = None;
    result
}
```

- [ ] **Step 3: Honor Tier 1 on the Qwen3 encoder**

In `Flux2Engine::load()`, find the Qwen3 device-selection block. The file uses a pattern similar to:

```rust
let device = crate::device::create_device(self.base.gpu_ordinal, &self.base.progress)?;
```

followed by a Qwen3 load. The encoder's device is determined via `should_use_gpu(is_cuda, is_metal, free_vram, qwen3_vram_threshold(...))`. Replace the Qwen3 device-selection branch with:

```rust
let tier1 = self
    .pending_placement
    .as_ref()
    .map(|p| p.text_encoders)
    .unwrap_or_default();
let qwen3_device = crate::device::resolve_device(
    Some(tier1),
    || {
        let on_gpu = crate::device::should_use_gpu(
            device.is_cuda(),
            device.is_metal(),
            crate::device::free_vram_bytes(self.base.gpu_ordinal).unwrap_or(0),
            crate::device::qwen3_vram_threshold(qwen3_model_bytes),
        );
        Ok(if on_gpu { device.clone() } else { Device::Cpu })
    },
)?;
```

The exact variable names (`qwen3_model_bytes`, etc.) already exist in the pipeline — search for `qwen3_vram_threshold` to find the call site.

- [ ] **Step 4: Verify**

```
cargo check -p mold-ai-inference
```

- [ ] **Step 5: Commit**

```
git add crates/mold-inference/src/flux2/pipeline.rs
git commit -m "feat(placement): honor Tier 1 text_encoders override in Flux.2 engine

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 6: Tier 1 plumbing — Z-Image

**File:** `crates/mold-inference/src/zimage/pipeline.rs`

- [ ] **Step 1: Add `pending_placement` to `ZImageEngine`** — same pattern as Task 4.

- [ ] **Step 2: Wrap `generate()`** — same pattern.

- [ ] **Step 3: Honor Tier 1 on Qwen3**

In `ZImageEngine::load()` (the file's `load_qwen3` helper or the inline Qwen3 block around line 444-496 based on the grep showing `enc_device = if on_gpu { &device } else { &Device::Cpu };`), replace:

```rust
let enc_device = if on_gpu { &device } else { &Device::Cpu };
```

with:

```rust
let tier1 = self
    .pending_placement
    .as_ref()
    .map(|p| p.text_encoders)
    .unwrap_or_default();
let enc_device_owned = crate::device::resolve_device(
    Some(tier1),
    || Ok(if on_gpu { device.clone() } else { Device::Cpu }),
)?;
let enc_device = &enc_device_owned;
```

- [ ] **Step 4: Verify** — `cargo check -p mold-ai-inference`.

- [ ] **Step 5: Commit**

```
git add crates/mold-inference/src/zimage/pipeline.rs
git commit -m "feat(placement): honor Tier 1 text_encoders override in Z-Image engine

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 7: Tier 1 plumbing — Qwen-Image

**File:** `crates/mold-inference/src/qwen_image/pipeline.rs`

- [ ] **Step 1-3:** Same three micro-steps as Task 6, targeting the Qwen2.5-VL encoder device-selection site. Grep for `qwen2_vram_threshold` or `should_use_gpu` inside `qwen_image/pipeline.rs` to locate the call.

- [ ] **Step 4: Verify** — `cargo check -p mold-ai-inference`.

- [ ] **Step 5: Commit**

```
git add crates/mold-inference/src/qwen_image/pipeline.rs
git commit -m "feat(placement): honor Tier 1 text_encoders override in Qwen-Image engine

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 8: Tier 1 plumbing — SD1.5 plus SDXL plus SD3.5

**Files:**
- `crates/mold-inference/src/sd15/pipeline.rs`
- `crates/mold-inference/src/sdxl/pipeline.rs`
- `crates/mold-inference/src/sd3/pipeline.rs`

One engine per commit. For each:

- [ ] **Step 1:** Add `pending_placement: Option<mold_core::types::DevicePlacement>` field, initialize in `new()`.
- [ ] **Step 2:** Wrap `generate()` with save/clear.
- [ ] **Step 3:** Find CLIP-L (SD1.5), CLIP-L plus CLIP-G (SDXL), or CLIP-L plus CLIP-G plus T5-XXL (SD3.5) placement sites. Replace the auto-selection with `resolve_device(Some(tier1), || Ok(<auto branch>))`. In SDXL the two CLIPs share the Tier 1 knob (spec: "group knob"). In SD3.5 all three encoders share it.
- [ ] **Step 4:** `cargo check -p mold-ai-inference`.
- [ ] **Step 5:** Commit, e.g.:

```
git commit -m "feat(placement): honor Tier 1 text_encoders override in SD1.5 engine

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

Repeat for SDXL and SD3.5. Three commits total.

---

## Task 9: Tier 1 plumbing — Wuerstchen plus LTX-Video plus LTX-2

**Files:**
- `crates/mold-inference/src/wuerstchen/pipeline.rs`
- `crates/mold-inference/src/ltx_video/pipeline.rs`
- `crates/mold-inference/src/ltx2/pipeline.rs`

Same three-step pattern per engine. Wuerstchen has a CLIP-G encoder; LTX-Video has T5-XXL; LTX-2 has Gemma 3. LTX-2 is CUDA-only — the existing code already guards Metal; wrapping `resolve_device(Some(tier1), || ...)` is still safe on Metal because `Auto` will pass through to the existing guard.

Three commits:

```
feat(placement): honor Tier 1 text_encoders override in Wuerstchen engine
feat(placement): honor Tier 1 text_encoders override in LTX-Video engine
feat(placement): honor Tier 1 text_encoders override in LTX-2 engine
```

After this task, **every engine** accepts the Tier 1 knob. Downstream UI work (Tasks 17-19) can assume `placement.text_encoders` works everywhere.

---

## Task 10: Tier 2 plumbing — FLUX transformer plus VAE plus per-encoder

**File:** `crates/mold-inference/src/flux/pipeline.rs`

Tier 1 (`placement.text_encoders`) already lands in Task 4. Tier 2 adds per-component overrides: the transformer, the VAE, and fine-grained T5 and CLIP-L overrides that supersede the Tier 1 group.

- [ ] **Step 1: Define the precedence helper**

At the top of `flux/pipeline.rs`, add:

```rust
/// Resolve a component override given Tier 1 plus Tier 2 requests.
///
/// Precedence:
///   1. `advanced_override` (Tier 2 per-component) if `Some`.
///   2. Fall back to `tier1` (group knob) if `fallback_is_component_auto`.
///   3. Fall back to `Auto`.
fn effective_device_ref(
    placement: Option<&mold_core::types::DevicePlacement>,
    advanced_override: impl FnOnce(&mold_core::types::AdvancedPlacement) -> Option<mold_core::types::DeviceRef>,
    fallback_is_component_auto: bool,
) -> mold_core::types::DeviceRef {
    use mold_core::types::DeviceRef;
    let Some(placement) = placement else {
        return DeviceRef::Auto;
    };
    if let Some(adv) = placement.advanced.as_ref() {
        if let Some(r) = advanced_override(adv) {
            return r;
        }
        if fallback_is_component_auto {
            return placement.text_encoders;
        }
        DeviceRef::Auto
    } else {
        placement.text_encoders
    }
}
```

- [ ] **Step 2: Honor Tier 2 for T5 and CLIP-L**

Rework the Task 4 T5 block:

```rust
let t5_ref = effective_device_ref(
    self.pending_placement.as_ref(),
    |adv| adv.t5,
    true, // encoders follow Tier 1 when Tier 2 has no t5 entry
);
let auto_t5_device = if t5_on_gpu { device.clone() } else { cpu.clone() };
let t5_device_owned = crate::device::resolve_device(
    Some(t5_ref),
    || Ok(auto_t5_device.clone()),
)?;
```

And for CLIP-L:

```rust
let clip_ref = effective_device_ref(
    self.pending_placement.as_ref(),
    |adv| adv.clip_l,
    true,
);
```

- [ ] **Step 3: Honor Tier 2 for the transformer**

Find the `create_device(self.base.gpu_ordinal, ...)` call around line 844 and wrap with `resolve_device`:

```rust
let transformer_ref = effective_device_ref(
    self.pending_placement.as_ref(),
    |adv| Some(adv.transformer),
    false,
);
let device = crate::device::resolve_device(
    Some(transformer_ref),
    || crate::device::create_device(self.base.gpu_ordinal, &self.base.progress),
)?;
```

Do the same at the second call site (~line 1077 in `generate_sequential`).

- [ ] **Step 4: Honor Tier 2 for the VAE**

The VAE currently uses the same `device` variable as the transformer. For Tier 2, it should resolve independently:

```rust
let vae_ref = effective_device_ref(
    self.pending_placement.as_ref(),
    |adv| Some(adv.vae),
    false,
);
let vae_device = crate::device::resolve_device(
    Some(vae_ref),
    || Ok(device.clone()),
)?;
```

Then replace the `vae_vb` load site:

```rust
let vae_vb = crate::weight_loader::load_safetensors_with_progress(
    std::slice::from_ref(&self.base.paths.vae),
    gpu_dtype,
    &vae_device,
    "VAE",
    &self.base.progress,
)?;
```

…and use `vae_device` in the `autoencoder::AutoEncoder::new` call path. Confirm the VAE decode path still works when `vae_device` differs from `device` (the decoder input gets `.to_device(&vae_device)?` before use; if such a call is missing, add it at the decode site in `generate()`).

- [ ] **Step 5: Verify**

```
cargo check -p mold-ai-inference
cargo clippy -p mold-ai-inference -- -D warnings
```

- [ ] **Step 6: Commit**

```
git add crates/mold-inference/src/flux/pipeline.rs
git commit -m "feat(placement): FLUX Tier 2 — honor advanced per-component placement overrides

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 11: Tier 2 plumbing — Flux.2

**File:** `crates/mold-inference/src/flux2/pipeline.rs`

Flux.2 has three components: Qwen3 encoder, transformer, VAE.

- [ ] **Step 1: Copy the `effective_device_ref` helper** from `flux/pipeline.rs` to `flux2/pipeline.rs` (duplicate to keep the per-engine diff scoped).

- [ ] **Step 2: Tier 2 for Qwen3**

```rust
let qwen3_ref = effective_device_ref(
    self.pending_placement.as_ref(),
    |adv| adv.qwen,
    true,
);
let qwen3_device = crate::device::resolve_device(
    Some(qwen3_ref),
    || {
        let on_gpu = crate::device::should_use_gpu(
            device.is_cuda(),
            device.is_metal(),
            crate::device::free_vram_bytes(self.base.gpu_ordinal).unwrap_or(0),
            crate::device::qwen3_vram_threshold(qwen3_model_bytes),
        );
        Ok(if on_gpu { device.clone() } else { Device::Cpu })
    },
)?;
```

- [ ] **Step 3: Tier 2 for transformer plus VAE** — mirror Task 10 Steps 3-4 with the Flux.2 variable names.

- [ ] **Step 4: Verify plus commit**

```
cargo check -p mold-ai-inference
git add crates/mold-inference/src/flux2/pipeline.rs
git commit -m "feat(placement): Flux.2 Tier 2 — honor advanced per-component placement overrides

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 12: Tier 2 plumbing — Z-Image

**File:** `crates/mold-inference/src/zimage/pipeline.rs`

Z-Image's components: Qwen3 encoder (drop-and-reload), transformer, VAE.

- [ ] **Step 1:** Copy `effective_device_ref`.
- [ ] **Step 2:** Rework the Tier 1 block added in Task 6 to use `effective_device_ref(..., |adv| adv.qwen, true)`.
- [ ] **Step 3:** Apply the same Tier 2 treatment to the transformer `create_device` call (around line 333) and the VAE load path (`load_vae` helper around line 249).
- [ ] **Step 4:** Verify plus commit.

```
cargo check -p mold-ai-inference
git add crates/mold-inference/src/zimage/pipeline.rs
git commit -m "feat(placement): Z-Image Tier 2 — honor advanced per-component placement overrides

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 13: Tier 2 plumbing — Qwen-Image

**File:** `crates/mold-inference/src/qwen_image/pipeline.rs`

Qwen-Image's components: Qwen2.5-VL encoder, transformer (flow-matching), VAE (3D causal).

- [ ] **Step 1:** Copy `effective_device_ref`.
- [ ] **Step 2:** Tier 2 for Qwen2.5-VL (`|adv| adv.qwen`).
- [ ] **Step 3:** Tier 2 for transformer plus VAE.
- [ ] **Step 4:** Verify plus commit.

```
cargo check -p mold-ai-inference
git add crates/mold-inference/src/qwen_image/pipeline.rs
git commit -m "feat(placement): Qwen-Image Tier 2 — honor advanced per-component placement overrides

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 14 (STRETCH): Tier 2 plumbing — SD3.5

**File:** `crates/mold-inference/src/sd3/pipeline.rs`

> **CUT INSTRUCTIONS:** SD3.5 Tier 2 is a stretch goal per spec §3.2. If Agent C is behind at the midpoint (after Task 16), **skip this task entirely**. SD3.5 Tier 1 already lands in Task 8, which is all the spec promises for the baseline. Document the skip in the umbrella PR description.

SD3.5 components: CLIP-L, CLIP-G, T5-XXL, MMDiT transformer, VAE. The mapping from `AdvancedPlacement` fields is:

| SD3.5 component | `AdvancedPlacement` field |
|-----------------|---------------------------|
| CLIP-L          | `clip_l`                  |
| CLIP-G          | `clip_g`                  |
| T5-XXL          | `t5`                      |
| MMDiT           | `transformer`             |
| VAE             | `vae`                     |

- [ ] **Step 1-4:** Same four steps as Task 10-13. SD3.5's `load()` already places each encoder independently (look for `t5_on_gpu`, `clip_l_on_gpu`, `clip_g_on_gpu`) which makes the Tier 2 wiring a pure transformation.

- [ ] **Step 5:** Verify plus commit.

```
cargo check -p mold-ai-inference
git add crates/mold-inference/src/sd3/pipeline.rs
git commit -m "feat(placement): SD3.5 Tier 2 — honor advanced per-component placement overrides

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 15: Config schema plus env overrides

**Files:**
- `crates/mold-core/src/config.rs`
- `crates/mold-core/src/placement_test.rs` (extend)

- [ ] **Step 1: Write the failing tests**

Append to `crates/mold-core/src/placement_test.rs`:

```rust
#[test]
fn model_config_serializes_placement_section() {
    use super::config::{Config, ModelConfig};
    let mut mc = ModelConfig::default();
    mc.placement = Some(DevicePlacement {
        text_encoders: DeviceRef::Cpu,
        advanced: Some(AdvancedPlacement {
            transformer: DeviceRef::gpu(0),
            vae: DeviceRef::Cpu,
            t5: Some(DeviceRef::Cpu),
            ..Default::default()
        }),
    });
    let mut cfg = Config::default();
    cfg.models.insert("flux-dev:q4".to_string(), mc);

    let toml = toml::to_string(&cfg).unwrap();
    assert!(toml.contains(r#"[models."flux-dev:q4".placement]"#), "toml:\n{toml}");
    // round-trip
    let back: Config = toml::from_str(&toml).unwrap();
    let p = back.models["flux-dev:q4"].placement.as_ref().unwrap();
    assert_eq!(p.text_encoders, DeviceRef::Cpu);
    let adv = p.advanced.as_ref().unwrap();
    assert_eq!(adv.transformer, DeviceRef::gpu(0));
    assert_eq!(adv.t5, Some(DeviceRef::Cpu));
}

#[test]
fn env_override_text_encoders_cpu() {
    let cfg = super::config::Config::default();
    std::env::set_var("MOLD_PLACE_TEXT_ENCODERS", "cpu");
    let p = cfg.resolved_placement("flux-dev:q4").unwrap();
    assert_eq!(p.text_encoders, DeviceRef::Cpu);
    std::env::remove_var("MOLD_PLACE_TEXT_ENCODERS");
}

#[test]
fn env_override_transformer_gpu_ordinal() {
    let cfg = super::config::Config::default();
    std::env::set_var("MOLD_PLACE_TRANSFORMER", "gpu:1");
    let p = cfg.resolved_placement("flux-dev:q4").unwrap();
    let adv = p.advanced.expect("gpu env override should populate advanced");
    assert_eq!(adv.transformer, DeviceRef::gpu(1));
    std::env::remove_var("MOLD_PLACE_TRANSFORMER");
}

#[test]
fn set_model_placement_creates_entry_if_missing() {
    let mut cfg = super::config::Config::default();
    let p = DevicePlacement {
        text_encoders: DeviceRef::gpu(0),
        advanced: None,
    };
    cfg.set_model_placement("flux-dev:q4", Some(p.clone()));
    assert_eq!(
        cfg.models.get("flux-dev:q4").and_then(|m| m.placement.clone()),
        Some(p)
    );
}

#[test]
fn set_model_placement_clears_when_none() {
    let mut cfg = super::config::Config::default();
    cfg.set_model_placement(
        "flux-dev:q4",
        Some(DevicePlacement {
            text_encoders: DeviceRef::Cpu,
            advanced: None,
        }),
    );
    cfg.set_model_placement("flux-dev:q4", None);
    assert!(
        cfg.models
            .get("flux-dev:q4")
            .and_then(|m| m.placement.clone())
            .is_none()
    );
}
```

Note: the env-var tests use raw `std::env::set_var` / `remove_var`. If flakiness appears due to parallel test execution, wrap via the existing `env_lock()` pattern used in `routes_test.rs`.

- [ ] **Step 2: Run — fail**

```
cargo test -p mold-ai-core placement_test --lib
```

Expected: `ModelConfig` missing `placement` field, `Config` missing `resolved_placement` plus `set_model_placement`.

- [ ] **Step 3: Add `placement` to `ModelConfig`**

In `crates/mold-core/src/config.rs`, append to `ModelConfig` (after the `lora_scale` field, around line 88):

```rust
    /// Per-component device placement override. `None` preserves the
    /// engine's VRAM-aware auto-placement.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub placement: Option<crate::types::DevicePlacement>,
```

Make sure `ModelConfig` still derives `Default` — the new field is `Option`, so `None` is the default.

- [ ] **Step 4: Add `resolved_placement()` plus env parsing**

Append to `impl Config` in `config.rs`, right before `pub fn save(&self)`:

```rust
    /// Return the effective placement for a model: config entry plus env overrides.
    ///
    /// Precedence (higher wins):
    ///   1. `MOLD_PLACE_TRANSFORMER`, `MOLD_PLACE_VAE`, `MOLD_PLACE_TEXT_ENCODERS`,
    ///      `MOLD_PLACE_T5`, `MOLD_PLACE_CLIP_L`, `MOLD_PLACE_CLIP_G`,
    ///      `MOLD_PLACE_QWEN` (env overrides per-component).
    ///   2. Config file `[models."name:tag".placement]` table.
    ///   3. `None` (use engine auto).
    ///
    /// Each env var parses:
    ///   - `"auto"`    — `DeviceRef::Auto`
    ///   - `"cpu"`     — `DeviceRef::Cpu`
    ///   - `"gpu:N"`   — `DeviceRef::Gpu { ordinal: N }`
    ///   - `"gpu"`     — `DeviceRef::Gpu { ordinal: 0 }`
    pub fn resolved_placement(&self, model_name: &str) -> Option<crate::types::DevicePlacement> {
        use crate::types::DevicePlacement;

        let mut placement = self
            .lookup_model_config(model_name)
            .and_then(|mc| mc.placement);

        let env_tier1 = parse_device_ref_env("MOLD_PLACE_TEXT_ENCODERS");
        let env_transformer = parse_device_ref_env("MOLD_PLACE_TRANSFORMER");
        let env_vae = parse_device_ref_env("MOLD_PLACE_VAE");
        let env_t5 = parse_device_ref_env("MOLD_PLACE_T5");
        let env_clip_l = parse_device_ref_env("MOLD_PLACE_CLIP_L");
        let env_clip_g = parse_device_ref_env("MOLD_PLACE_CLIP_G");
        let env_qwen = parse_device_ref_env("MOLD_PLACE_QWEN");

        let any_env = env_tier1.is_some()
            || env_transformer.is_some()
            || env_vae.is_some()
            || env_t5.is_some()
            || env_clip_l.is_some()
            || env_clip_g.is_some()
            || env_qwen.is_some();

        if !any_env {
            return placement;
        }

        let mut effective: DevicePlacement = placement.unwrap_or_default();
        if let Some(r) = env_tier1 {
            effective.text_encoders = r;
        }
        let any_advanced = env_transformer.is_some() || env_vae.is_some()
            || env_t5.is_some() || env_clip_l.is_some()
            || env_clip_g.is_some() || env_qwen.is_some();
        if any_advanced {
            let mut adv = effective.advanced.unwrap_or_default();
            if let Some(r) = env_transformer { adv.transformer = r; }
            if let Some(r) = env_vae { adv.vae = r; }
            if let Some(r) = env_t5 { adv.t5 = Some(r); }
            if let Some(r) = env_clip_l { adv.clip_l = Some(r); }
            if let Some(r) = env_clip_g { adv.clip_g = Some(r); }
            if let Some(r) = env_qwen { adv.qwen = Some(r); }
            effective.advanced = Some(adv);
        }
        placement = Some(effective);
        placement
    }

    /// Persist a placement for `model_name`, creating the model entry if
    /// missing. `None` clears the placement (and leaves the rest of the
    /// entry intact).
    pub fn set_model_placement(
        &mut self,
        model_name: &str,
        placement: Option<crate::types::DevicePlacement>,
    ) {
        let entry = self.models.entry(model_name.to_string()).or_default();
        entry.placement = placement;
    }
```

Append the parser helper at the bottom of `config.rs` (sibling of the existing `default_*` helpers):

```rust
fn parse_device_ref_env(key: &str) -> Option<crate::types::DeviceRef> {
    use crate::types::DeviceRef;
    let raw = std::env::var(key).ok()?;
    let raw = raw.trim().to_lowercase();
    if raw == "auto" {
        Some(DeviceRef::Auto)
    } else if raw == "cpu" {
        Some(DeviceRef::Cpu)
    } else if raw == "gpu" {
        Some(DeviceRef::gpu(0))
    } else if let Some(rest) = raw.strip_prefix("gpu:") {
        rest.parse::<usize>().ok().map(DeviceRef::gpu)
    } else {
        tracing::warn!("ignoring invalid {key}={raw} (expected auto|cpu|gpu[:N])");
        None
    }
}
```

- [ ] **Step 5: Run — passes**

```
cargo test -p mold-ai-core placement_test --lib
```

Expected: all green.

- [ ] **Step 6: Commit**

```
git add crates/mold-core/src/config.rs crates/mold-core/src/placement_test.rs
git commit -m "feat(placement): config schema plus env overrides plus set_model_placement helper

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 16: `PUT /api/config/model/:name/placement` route

**Files:**
- `crates/mold-server/src/routes.rs`
- `crates/mold-server/src/routes_test.rs`

- [ ] **Step 1: Write the failing route test**

Append to `crates/mold-server/src/routes_test.rs`:

```rust
#[tokio::test]
async fn put_model_placement_updates_config_and_persists() {
    let _lock = env_lock().lock().unwrap();
    let tmp = tempfile::tempdir().unwrap();
    std::env::set_var("MOLD_HOME", tmp.path());
    let state = AppState::new_for_tests().await;
    let app = create_router(state.clone());

    let body = serde_json::json!({
        "text_encoders": { "kind": "cpu" },
        "advanced": {
            "transformer": { "kind": "gpu", "ordinal": 1 },
            "vae": { "kind": "auto" },
            "t5": { "kind": "cpu" }
        }
    });

    let resp = app
        .clone()
        .oneshot(
            Request::builder()
                .method("PUT")
                .uri("/api/config/model/flux-dev%3Aq4/placement")
                .header("content-type", "application/json")
                .body(Body::from(body.to_string()))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(resp.status(), StatusCode::OK);

    let cfg = state.config.read().await;
    let p = cfg
        .models
        .get("flux-dev:q4")
        .and_then(|m| m.placement.clone())
        .expect("placement not persisted");
    assert_eq!(
        p.text_encoders,
        mold_core::types::DeviceRef::Cpu
    );
    let adv = p.advanced.unwrap();
    assert_eq!(adv.transformer, mold_core::types::DeviceRef::gpu(1));
    std::env::remove_var("MOLD_HOME");
}

#[tokio::test]
async fn put_model_placement_rejects_malformed_body() {
    let _lock = env_lock().lock().unwrap();
    let state = AppState::new_for_tests().await;
    let app = create_router(state.clone());
    let resp = app
        .oneshot(
            Request::builder()
                .method("PUT")
                .uri("/api/config/model/flux-dev%3Aq4/placement")
                .header("content-type", "application/json")
                .body(Body::from(r#"{"text_encoders":"not-an-object"}"#))
                .unwrap(),
        )
        .await
        .unwrap();
    assert!(
        resp.status() == StatusCode::BAD_REQUEST
            || resp.status() == StatusCode::UNPROCESSABLE_ENTITY
    );
}
```

(`AppState::new_for_tests()` is an existing helper — confirm by searching `routes_test.rs`; it's referenced by neighboring tests.)

- [ ] **Step 2: Run — fail**

```
cargo test -p mold-ai-server put_model_placement --lib
```

Expected: route 404 because the handler doesn't exist yet.

- [ ] **Step 3: Implement the handler plus wire into router**

In `crates/mold-server/src/routes.rs`, add to the imports at top (skip if already imported):

```rust
use axum::extract::Path;
```

Add the handler (anywhere in the file — `server_status` is a reasonable neighbor):

```rust
async fn put_model_placement(
    State(state): State<AppState>,
    Path(name): Path<String>,
    Json(placement): Json<mold_core::types::DevicePlacement>,
) -> Result<Json<serde_json::Value>, ApiError> {
    {
        let mut cfg = state.config.write().await;
        cfg.set_model_placement(&name, Some(placement.clone()));
        if let Err(e) = cfg.save() {
            tracing::warn!("failed to persist placement to config.toml: {e}");
        }
    }
    Ok(Json(serde_json::json!({
        "ok": true,
        "model": name,
    })))
}

async fn delete_model_placement(
    State(state): State<AppState>,
    Path(name): Path<String>,
) -> Result<Json<serde_json::Value>, ApiError> {
    let mut cfg = state.config.write().await;
    cfg.set_model_placement(&name, None);
    let _ = cfg.save();
    Ok(Json(serde_json::json!({ "ok": true })))
}
```

Wire into `create_router` — in `routes.rs` around line 160, extend the `.route(...)` chain:

```rust
// Agent C (model-ui-overhaul §3): placement persistence.
.route(
    "/api/config/model/:name/placement",
    axum::routing::put(put_model_placement)
        .delete(delete_model_placement),
)
```

- [ ] **Step 4: Run — passes**

```
cargo test -p mold-ai-server put_model_placement --lib
```

Expected: both tests green.

- [ ] **Step 5: Commit**

```
git add crates/mold-server/src/routes.rs crates/mold-server/src/routes_test.rs
git commit -m "feat(placement): PUT /api/config/model/:name/placement route

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 17: Web types plus form wire-through

**Files:**
- `web/src/types.ts`
- `web/src/composables/useGenerateForm.ts`
- `web/src/composables/useGenerateForm.test.ts`

- [ ] **Step 1: Write the failing test**

Append to `web/src/composables/useGenerateForm.test.ts`:

```ts
it("carries placement into the outgoing request wire", () => {
  const form = useGenerateForm();
  form.state.value.placement = {
    text_encoders: { kind: "cpu" },
    advanced: {
      transformer: { kind: "gpu", ordinal: 1 },
      vae: { kind: "auto" },
      t5: { kind: "cpu" },
    },
  };
  const wire = form.toRequest();
  expect(wire.placement).toEqual({
    text_encoders: { kind: "cpu" },
    advanced: {
      transformer: { kind: "gpu", ordinal: 1 },
      vae: { kind: "auto" },
      t5: { kind: "cpu" },
    },
  });
});

it("omits placement from the request when null", () => {
  const form = useGenerateForm();
  form.state.value.placement = null;
  const wire = form.toRequest();
  expect(wire.placement).toBeUndefined();
});
```

- [ ] **Step 2: Run — fail**

```
cd web && bun run test useGenerateForm
```

Expected: fail — `placement` not on `GenerateFormState`.

- [ ] **Step 3: Add types**

In `web/src/types.ts`, append before the `GenerateFormState` interface:

```ts
// ── Device placement (Agent C: model-ui-overhaul §3) ──────────────────────
export type DeviceRef =
  | { kind: "auto" }
  | { kind: "cpu" }
  | { kind: "gpu"; ordinal: number };

export interface AdvancedPlacement {
  transformer: DeviceRef;
  vae: DeviceRef;
  clip_l?: DeviceRef | null;
  clip_g?: DeviceRef | null;
  t5?: DeviceRef | null;
  qwen?: DeviceRef | null;
}

export interface DevicePlacement {
  text_encoders: DeviceRef;
  advanced?: AdvancedPlacement | null;
}
```

Extend `GenerateRequestWire` (same file):

```ts
export interface GenerateRequestWire {
  // ... existing fields ...
  placement?: DevicePlacement | null;
}
```

Extend `GenerateFormState`:

```ts
export interface GenerateFormState {
  // ... existing fields ...
  placement: DevicePlacement | null;
}
```

- [ ] **Step 4: Update `useGenerateForm.ts`**

Add `placement: null` to `defaultForm()`:

```ts
function defaultForm(): GenerateFormState {
  return {
    // ... existing fields ...
    placement: null,
  };
}
```

Extend `toRequest()`:

```ts
toRequest: () => {
  const s = state.value;
  return {
    // ... existing fields ...
    placement: s.placement ?? undefined,
  };
},
```

- [ ] **Step 5: Run — passes**

```
cd web && bun run test useGenerateForm
```

Expected: all green.

- [ ] **Step 6: Commit**

```
git add web/src/types.ts web/src/composables/useGenerateForm.ts web/src/composables/useGenerateForm.test.ts
git commit -m "feat(web): carry DevicePlacement through form state and request wire

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 18: `usePlacement.ts` composable

**Files:**
- `web/src/composables/usePlacement.ts` (new)
- `web/src/composables/usePlacement.test.ts` (new)

- [ ] **Step 1: Write the failing test**

Create `web/src/composables/usePlacement.test.ts`:

```ts
import { describe, expect, it } from "vitest";
import { usePlacement } from "./usePlacement";

describe("usePlacement", () => {
  it("defaults to null placement", () => {
    const { placement } = usePlacement();
    expect(placement.value).toBeNull();
  });

  it("supportsAdvanced returns true for Tier 2 families", () => {
    const { supportsAdvanced } = usePlacement();
    expect(supportsAdvanced("flux")).toBe(true);
    expect(supportsAdvanced("flux2")).toBe(true);
    expect(supportsAdvanced("z-image")).toBe(true);
    expect(supportsAdvanced("qwen-image")).toBe(true);
  });

  it("supportsAdvanced respects the stretch flag for sd3", () => {
    const { supportsAdvanced } = usePlacement();
    // SD3.5 is in the Tier 2 family list (stretch goal, lands in Task 14).
    // If Task 14 is skipped the panel still renders because supportsAdvanced
    // reflects the intended allow-list — the server simply ignores the
    // advanced override for SD3.5.
    expect(supportsAdvanced("sd3")).toBe(true);
  });

  it("supportsAdvanced returns false for Tier 1 only families", () => {
    const { supportsAdvanced } = usePlacement();
    expect(supportsAdvanced("sdxl")).toBe(false);
    expect(supportsAdvanced("sd15")).toBe(false);
    expect(supportsAdvanced("wuerstchen")).toBe(false);
    expect(supportsAdvanced("ltx-video")).toBe(false);
    expect(supportsAdvanced("ltx2")).toBe(false);
  });

  it("sets text encoders Tier 1 without creating advanced", () => {
    const { placement, setTextEncoders } = usePlacement();
    setTextEncoders({ kind: "cpu" });
    expect(placement.value).toEqual({
      text_encoders: { kind: "cpu" },
      advanced: null,
    });
  });

  it("setAdvancedField promotes to Tier 2 automatically", () => {
    const { placement, setAdvancedField } = usePlacement();
    setAdvancedField("transformer", { kind: "gpu", ordinal: 1 });
    expect(placement.value?.advanced?.transformer).toEqual({
      kind: "gpu",
      ordinal: 1,
    });
    expect(placement.value?.text_encoders).toEqual({ kind: "auto" });
  });

  it("loadSaved overwrites current state", () => {
    const { placement, loadSaved } = usePlacement();
    loadSaved({
      text_encoders: { kind: "cpu" },
      advanced: null,
    });
    expect(placement.value?.text_encoders).toEqual({ kind: "cpu" });
  });

  it("clear resets placement to null", () => {
    const { placement, setTextEncoders, clear } = usePlacement();
    setTextEncoders({ kind: "cpu" });
    clear();
    expect(placement.value).toBeNull();
  });

  it("gpuList is a stub until Agent B merges useResources", () => {
    const { gpuList } = usePlacement();
    expect(Array.isArray(gpuList.value)).toBe(true);
  });
});
```

- [ ] **Step 2: Run — fail**

```
cd web && bun run test usePlacement
```

Expected: fail — file doesn't exist.

- [ ] **Step 3: Implement `usePlacement.ts`**

Create `web/src/composables/usePlacement.ts`:

```ts
import { computed, ref } from "vue";
import type {
  AdvancedPlacement,
  DevicePlacement,
  DeviceRef,
} from "../types";

// Families that support the Advanced (Tier 2) per-component disclosure.
// Matches spec §3.2 — update both in lock-step.
const TIER2_FAMILIES: ReadonlyArray<string> = [
  "flux",
  "flux2",
  "flux.2",
  "flux2-klein",
  "z-image",
  "qwen-image",
  "qwen_image",
  "sd3",
  "sd3.5",
  "stable-diffusion-3",
  "stable-diffusion-3.5",
];

export interface UsePlacement {
  placement: import("vue").Ref<DevicePlacement | null>;
  gpuList: import("vue").ComputedRef<Array<{ ordinal: number; name: string }>>;
  supportsAdvanced: (family: string) => boolean;
  setTextEncoders: (ref: DeviceRef) => void;
  setAdvancedField: (
    field: keyof AdvancedPlacement,
    ref: DeviceRef | null,
  ) => void;
  loadSaved: (p: DevicePlacement | null) => void;
  clear: () => void;
  saveAsDefault: (model: string) => Promise<void>;
}

function defaultAdvanced(): AdvancedPlacement {
  return {
    transformer: { kind: "auto" },
    vae: { kind: "auto" },
    clip_l: null,
    clip_g: null,
    t5: null,
    qwen: null,
  };
}

export function usePlacement(): UsePlacement {
  const placement = ref<DevicePlacement | null>(null);

  // TODO-REMOVE-AFTER-MERGE: replace this stub with
  //   const { gpuList: resourceGpus } = useResources();
  // once Agent B's resource-telemetry branch merges into the umbrella.
  // The stub keeps this composable testable in isolation and lets the UI
  // render a plausible list during agent-C development. Search for
  // `TODO-REMOVE-AFTER-MERGE` to find every site that needs updating.
  const gpuList = computed(() =>
    (globalThis as unknown as { __MOLD_GPU_STUB__?: Array<{ ordinal: number; name: string }> })
      .__MOLD_GPU_STUB__ ?? [
      { ordinal: 0, name: "GPU 0 (stub)" },
      { ordinal: 1, name: "GPU 1 (stub)" },
    ],
  );

  const supportsAdvanced = (family: string) =>
    TIER2_FAMILIES.includes(family);

  const setTextEncoders = (r: DeviceRef) => {
    placement.value = {
      text_encoders: r,
      advanced: placement.value?.advanced ?? null,
    };
  };

  const setAdvancedField = (
    field: keyof AdvancedPlacement,
    r: DeviceRef | null,
  ) => {
    const current =
      placement.value ?? {
        text_encoders: { kind: "auto" } as DeviceRef,
        advanced: null,
      };
    const adv = { ...(current.advanced ?? defaultAdvanced()) };
    if (field === "transformer" || field === "vae") {
      adv[field] = r ?? { kind: "auto" };
    } else {
      adv[field] = r;
    }
    placement.value = { ...current, advanced: adv };
  };

  const loadSaved = (p: DevicePlacement | null) => {
    placement.value = p;
  };

  const clear = () => {
    placement.value = null;
  };

  const saveAsDefault = async (model: string) => {
    if (!placement.value) return;
    const encoded = encodeURIComponent(model);
    const resp = await fetch(`/api/config/model/${encoded}/placement`, {
      method: "PUT",
      headers: { "content-type": "application/json" },
      body: JSON.stringify(placement.value),
    });
    if (!resp.ok) {
      throw new Error(
        `failed to save placement default: ${resp.status} ${resp.statusText}`,
      );
    }
  };

  return {
    placement,
    gpuList,
    supportsAdvanced,
    setTextEncoders,
    setAdvancedField,
    loadSaved,
    clear,
    saveAsDefault,
  };
}
```

- [ ] **Step 4: Run — passes**

```
cd web && bun run test usePlacement
```

Expected: all 9 tests green.

- [ ] **Step 5: Commit**

```
git add web/src/composables/usePlacement.ts web/src/composables/usePlacement.test.ts
git commit -m "feat(web): add usePlacement composable with Tier 2 family gate

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 19: `PlacementPanel.vue` component

**Files:**
- `web/src/components/PlacementPanel.vue` (new)
- `web/src/components/PlacementPanel.test.ts` (new)

- [ ] **Step 1: Write the failing test**

Create `web/src/components/PlacementPanel.test.ts`:

```ts
import { describe, expect, it } from "vitest";
import { mount } from "@vue/test-utils";
import PlacementPanel from "./PlacementPanel.vue";

function mountPanel(props: {
  family: string;
  placement?: import("../types").DevicePlacement | null;
  model?: string;
}) {
  return mount(PlacementPanel, {
    props: {
      modelValue: props.placement ?? null,
      family: props.family,
      model: props.model ?? "flux-dev:q4",
      gpus: [
        { ordinal: 0, name: "RTX 3090" },
        { ordinal: 1, name: "RTX 3090" },
      ],
    },
  });
}

describe("PlacementPanel", () => {
  it("renders the Tier 1 select with Auto/CPU/GPU options", () => {
    const wrapper = mountPanel({ family: "flux" });
    const opts = wrapper.findAll("select[data-test='tier1-select'] option");
    const labels = opts.map((o) => o.text());
    expect(labels).toContain("Auto");
    expect(labels).toContain("CPU");
    expect(labels.some((l) => l.includes("GPU 0"))).toBe(true);
    expect(labels.some((l) => l.includes("GPU 1"))).toBe(true);
  });

  it("hides Tier 1 select when GPU list is empty", () => {
    const wrapper = mount(PlacementPanel, {
      props: {
        modelValue: null,
        family: "flux",
        model: "flux-dev:q4",
        gpus: [],
      },
    });
    expect(wrapper.find("select[data-test='tier1-select']").exists()).toBe(
      false,
    );
  });

  it("enables Advanced disclosure for Tier 2 families", () => {
    const wrapper = mountPanel({ family: "flux" });
    const toggle = wrapper.find("button[data-test='advanced-toggle']");
    expect(toggle.exists()).toBe(true);
    expect(toggle.attributes("disabled")).toBeUndefined();
  });

  it("disables Advanced disclosure for Tier 1-only families with a tooltip", () => {
    const wrapper = mountPanel({ family: "sdxl" });
    const toggle = wrapper.find("button[data-test='advanced-toggle']");
    expect(toggle.attributes("disabled")).toBeDefined();
    expect(toggle.attributes("title")).toMatch(/not yet available/i);
  });

  it("emits update:modelValue when Tier 1 changes", async () => {
    const wrapper = mountPanel({ family: "flux" });
    const select = wrapper.find("select[data-test='tier1-select']");
    await select.setValue("cpu");
    const emitted = wrapper.emitted("update:modelValue");
    expect(emitted).toBeTruthy();
    const last = emitted!.at(-1)![0] as import("../types").DevicePlacement | null;
    expect(last?.text_encoders).toEqual({ kind: "cpu" });
  });

  it("renders Save-as-default button when placement differs from saved", () => {
    const wrapper = mountPanel({
      family: "flux",
      placement: {
        text_encoders: { kind: "cpu" },
        advanced: null,
      },
    });
    expect(
      wrapper.find("button[data-test='save-default']").exists(),
    ).toBe(true);
  });
});
```

- [ ] **Step 2: Run — fail**

```
cd web && bun run test PlacementPanel
```

Expected: fail — component doesn't exist.

- [ ] **Step 3: Implement `PlacementPanel.vue`**

Create `web/src/components/PlacementPanel.vue`:

```vue
<script setup lang="ts">
import { computed, ref } from "vue";
import type {
  AdvancedPlacement,
  DevicePlacement,
  DeviceRef,
} from "../types";
import { usePlacement } from "../composables/usePlacement";

interface GpuEntry {
  ordinal: number;
  name: string;
}

const props = defineProps<{
  modelValue: DevicePlacement | null;
  family: string;
  model: string;
  gpus: GpuEntry[];
}>();

const emit = defineEmits<{
  (e: "update:modelValue", v: DevicePlacement | null): void;
}>();

const { supportsAdvanced } = usePlacement();

const tier2 = computed(() => supportsAdvanced(props.family));
const advancedOpen = ref(false);

const tier1Value = computed<DeviceRef>(() =>
  props.modelValue?.text_encoders ?? { kind: "auto" },
);

function refToOption(r: DeviceRef): string {
  if (r.kind === "auto") return "auto";
  if (r.kind === "cpu") return "cpu";
  return `gpu:${r.ordinal}`;
}

function optionToRef(opt: string): DeviceRef {
  if (opt === "auto") return { kind: "auto" };
  if (opt === "cpu") return { kind: "cpu" };
  const m = /^gpu:(\d+)$/.exec(opt);
  return m ? { kind: "gpu", ordinal: Number(m[1]) } : { kind: "auto" };
}

function emitTier1(opt: string) {
  const next: DevicePlacement = {
    text_encoders: optionToRef(opt),
    advanced: props.modelValue?.advanced ?? null,
  };
  emit("update:modelValue", next);
}

function emitAdvanced<K extends keyof AdvancedPlacement>(
  field: K,
  opt: string,
) {
  const r = optionToRef(opt);
  const current: AdvancedPlacement = props.modelValue?.advanced ?? {
    transformer: { kind: "auto" },
    vae: { kind: "auto" },
    clip_l: null,
    clip_g: null,
    t5: null,
    qwen: null,
  };
  const nextAdv: AdvancedPlacement = { ...current };
  if (field === "transformer" || field === "vae") {
    nextAdv[field] = r;
  } else {
    nextAdv[field] = r;
  }
  emit("update:modelValue", {
    text_encoders: props.modelValue?.text_encoders ?? { kind: "auto" },
    advanced: nextAdv,
  });
}

async function saveAsDefault() {
  if (!props.modelValue) return;
  const encoded = encodeURIComponent(props.model);
  await fetch(`/api/config/model/${encoded}/placement`, {
    method: "PUT",
    headers: { "content-type": "application/json" },
    body: JSON.stringify(props.modelValue),
  });
}

const encoderRows = computed(() => {
  switch (props.family) {
    case "flux":
      return ["t5", "clip_l"] as const;
    case "flux2":
    case "flux.2":
    case "flux2-klein":
    case "z-image":
      return ["qwen"] as const;
    case "qwen-image":
    case "qwen_image":
      return ["qwen"] as const;
    case "sd3":
    case "sd3.5":
    case "stable-diffusion-3":
    case "stable-diffusion-3.5":
      return ["t5", "clip_l", "clip_g"] as const;
    default:
      return [] as const;
  }
});

function advancedValue(field: keyof AdvancedPlacement): string {
  const adv = props.modelValue?.advanced;
  if (!adv) return "auto";
  const v = adv[field];
  if (v === null || v === undefined) return "auto";
  return refToOption(v as DeviceRef);
}

const isDirty = computed(() => props.modelValue !== null);
</script>

<template>
  <section class="glass flex flex-col gap-2 rounded-2xl p-3 text-sm">
    <header class="flex items-center justify-between">
      <span class="font-medium text-slate-200">Device placement</span>
      <button
        v-if="isDirty"
        type="button"
        data-test="save-default"
        class="text-xs text-brand-400 hover:underline"
        @click="saveAsDefault"
      >
        Save as default
      </button>
    </header>

    <div v-if="gpus.length > 0" class="flex items-center gap-2">
      <label class="text-slate-400">Text encoders</label>
      <select
        data-test="tier1-select"
        :value="refToOption(tier1Value)"
        class="rounded bg-slate-900 px-2 py-1 text-slate-100"
        @change="emitTier1(($event.target as HTMLSelectElement).value)"
      >
        <option value="auto">Auto</option>
        <option value="cpu">CPU</option>
        <option
          v-for="g in gpus"
          :key="g.ordinal"
          :value="`gpu:${g.ordinal}`"
        >
          GPU {{ g.ordinal }} ({{ g.name }})
        </option>
      </select>
    </div>

    <div class="flex items-center gap-2">
      <button
        type="button"
        data-test="advanced-toggle"
        class="text-xs text-slate-400 hover:text-slate-200 disabled:cursor-not-allowed disabled:opacity-50"
        :disabled="!tier2 ? '' : undefined"
        :title="
          tier2
            ? undefined
            : `Advanced placement is not yet available for ${family} — Tier 1 controls all encoders as a group.`
        "
        @click="tier2 && (advancedOpen = !advancedOpen)"
      >
        {{ advancedOpen ? "▾" : "▸" }} Advanced
      </button>
    </div>

    <div v-if="tier2 && advancedOpen" class="flex flex-col gap-1 pl-4">
      <div class="flex items-center gap-2">
        <label class="w-24 text-slate-400">Transformer</label>
        <select
          :value="advancedValue('transformer')"
          class="rounded bg-slate-900 px-2 py-1 text-slate-100"
          @change="
            emitAdvanced(
              'transformer',
              ($event.target as HTMLSelectElement).value,
            )
          "
        >
          <option value="auto">Auto</option>
          <option value="cpu">CPU</option>
          <option
            v-for="g in gpus"
            :key="g.ordinal"
            :value="`gpu:${g.ordinal}`"
          >
            GPU {{ g.ordinal }}
          </option>
        </select>
      </div>

      <div class="flex items-center gap-2">
        <label class="w-24 text-slate-400">VAE</label>
        <select
          :value="advancedValue('vae')"
          class="rounded bg-slate-900 px-2 py-1 text-slate-100"
          @change="
            emitAdvanced('vae', ($event.target as HTMLSelectElement).value)
          "
        >
          <option value="auto">Auto</option>
          <option value="cpu">CPU</option>
          <option
            v-for="g in gpus"
            :key="g.ordinal"
            :value="`gpu:${g.ordinal}`"
          >
            GPU {{ g.ordinal }}
          </option>
        </select>
      </div>

      <div
        v-for="field in encoderRows"
        :key="field"
        class="flex items-center gap-2"
      >
        <label class="w-24 text-slate-400">{{ field }}</label>
        <select
          :value="advancedValue(field)"
          class="rounded bg-slate-900 px-2 py-1 text-slate-100"
          @change="
            emitAdvanced(field, ($event.target as HTMLSelectElement).value)
          "
        >
          <option value="auto">Auto (follow group)</option>
          <option value="cpu">CPU</option>
          <option
            v-for="g in gpus"
            :key="g.ordinal"
            :value="`gpu:${g.ordinal}`"
          >
            GPU {{ g.ordinal }}
          </option>
        </select>
      </div>
    </div>
  </section>
</template>
```

- [ ] **Step 4: Run — passes**

```
cd web && bun run test PlacementPanel
```

Expected: all 6 tests green.

- [ ] **Step 5: Commit**

```
git add web/src/components/PlacementPanel.vue web/src/components/PlacementPanel.test.ts
git commit -m "feat(web): PlacementPanel component with Tier 1 and Tier 2 fieldset

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 20: Mount `PlacementPanel` into `Composer.vue`

**File:** `web/src/components/Composer.vue`

Keep the diff confined to clearly-labeled additive sections per spec §4.2.

- [ ] **Step 1: Extend props plus emits**

In `Composer.vue` `<script setup>`:

```ts
const props = defineProps<{
  modelValue: GenerateFormState;
  queueDepth: number | null;
  queueCapacity: number | null;
  gpus: { ordinal: number; state: string }[] | null;
  expandActive: boolean;
  settingsDirty: boolean;
  // Agent C (model-ui-overhaul §3) ────────────────────────────────────
  family: string; // family of the currently-selected model
  placementGpus: { ordinal: number; name: string }[];
  // ──────────────────────────────────────────────────────────────────
}>();
```

Add a helper for placement updates:

```ts
function updatePlacement(p: import("../types").DevicePlacement | null) {
  emit("update:modelValue", { ...props.modelValue, placement: p });
}
```

- [ ] **Step 2: Import and render the panel**

Add at the top of `<script setup>`:

```ts
import PlacementPanel from "./PlacementPanel.vue";
```

In the `<template>`, add just before the closing `</div>` of the outermost `glass` container:

```vue
    <!-- Agent C (model-ui-overhaul §3): device placement -->
    <PlacementPanel
      :model-value="modelValue.placement"
      :family="family"
      :model="modelValue.model"
      :gpus="placementGpus"
      @update:model-value="updatePlacement"
    />
```

- [ ] **Step 3: Update the caller (GeneratePage.vue)**

Find `web/src/pages/GeneratePage.vue` (grep for `<Composer`). Pass the new required props:

```vue
<Composer
  v-model="form.state.value"
  :queue-depth="queueDepth"
  :queue-capacity="queueCapacity"
  :gpus="statusGpus"
  :expand-active="expandActive"
  :settings-dirty="settingsDirty"
  :family="currentModel?.family ?? ''"
  :placement-gpus="gpuListForPlacement"
  @submit="onSubmit"
  @open-settings="onOpenSettings"
  @open-expand="onOpenExpand"
  @open-image-picker="onOpenImagePicker"
  @clear-source="onClearSource"
/>
```

where `gpuListForPlacement` comes from the Agent B stub:

```ts
// TODO-REMOVE-AFTER-MERGE: use useResources().gpuList once Agent B merges.
const gpuListForPlacement = computed(() =>
  statusGpus.value?.map((g) => ({
    ordinal: g.ordinal,
    name: `GPU ${g.ordinal}`,
  })) ?? [],
);
```

`statusGpus` is the existing value sourced from `/api/status`.

- [ ] **Step 4: Run the full frontend suite**

```
cd web && bun run test
cd web && bun run fmt:check
cd web && bun run build
```

Expected: all three pass.

- [ ] **Step 5: Commit**

```
git add web/src/components/Composer.vue web/src/pages/GeneratePage.vue
git commit -m "feat(web): mount PlacementPanel in Composer with family plus GPU list props

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 21: Full gate plus open PR into umbrella

- [ ] **Step 1: Run the full local CI suite**

```
cargo fmt --check
cargo check --workspace
cargo clippy --workspace -- -D warnings
cargo test --workspace
cargo check -p mold-ai --features preview,discord,expand,tui,webp,mp4
cd web && bun run fmt:check && bun run test && bun run build
```

Every command must exit 0. If any fail, fix the root cause (per CLAUDE.md §6) — do NOT disable tests or bypass `-D warnings`.

- [ ] **Step 2: Re-read the agent-boundary checklist from spec §4.2**

Confirm the diff touches **only**:

- `crates/mold-core/src/types.rs` (additive Device types block plus `placement` field)
- `crates/mold-core/src/placement_test.rs` (new)
- `crates/mold-core/src/config.rs` (additive `placement`, `resolved_placement`, `set_model_placement`, `parse_device_ref_env`)
- `crates/mold-inference/src/device.rs` (additive `resolve_device`)
- `crates/mold-inference/tests/device_resolve_test.rs` (new)
- `crates/mold-inference/src/{flux,flux2,zimage,qwen_image,sd15,sdxl,sd3,wuerstchen,ltx_video,ltx2}/pipeline.rs` (per-engine plumbing)
- `crates/mold-server/src/routes.rs` (additive `put_model_placement` handler plus one route)
- `crates/mold-server/src/routes_test.rs` (additive)
- `web/src/types.ts` (additive `DeviceRef`/`DevicePlacement`/`AdvancedPlacement` block plus `placement` fields on existing types)
- `web/src/composables/{useGenerateForm,usePlacement}.{ts,test.ts}`
- `web/src/components/{Composer,PlacementPanel}.{vue,test.ts}`
- `web/src/pages/GeneratePage.vue` (prop wiring)

Confirm **no changes** to:

- `/api/downloads*` routes (Agent A)
- `/api/resources*` routes (Agent B)
- `crates/mold-server/src/state.rs` beyond incidental trait/import changes
- anything under `crates/mold-server/src/{downloads,resources}.rs` (those modules shouldn't even exist in this branch until Agents A/B merge)

- [ ] **Step 3: Push plus open PR**

```
git push -u origin wt/device-placement
gh pr create --base feat/model-ui-overhaul --head wt/device-placement \
  --title "feat: device placement (Agent C of model-ui-overhaul)" \
  --body "Agent C phase of the model-ui-overhaul umbrella. Closes Phase 3 of docs/superpowers/specs/2026-04-19-model-ui-overhaul-design.md."
```

Return the PR URL so the main agent can track it in the umbrella review.

---

## Appendix A — Open questions flagged for the main agent

1. **`Device::clone()` availability** — Task 4 relies on `candle_core::Device: Clone`. Candle 0.8+ ships this via `impl Clone for Device`. If the workspace pins an older revision, adjust Task 4 to use reference-only plumbing (the auto-closure can capture `&Device` directly).

2. **`scopeguard`-style cleanup of `pending_placement`** — the two-call wrapper pattern is panic-safe only if the inner `generate_impl` never panics (which matches the rest of the engine code, which consistently `?`s). If that assumption changes, wrap with an RAII guard via the existing `OptionRestoreGuard` pattern in `engine.rs`.

3. **SD3.5 stretch decision** — mid-plan the main agent should flip Task 14 to skipped if Agent C is behind. The spec explicitly sanctions this cut.

4. **Agent B merge order** — if Agent B lands before Agent C opens its PR, Task 18's `usePlacement.ts` stub should be replaced in-place (not deferred to a post-merge fixup). Grep for `TODO-REMOVE-AFTER-MERGE` and swap to `const { gpuList } = useResources();`.
