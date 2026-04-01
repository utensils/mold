# Architecture Research: Multi-Model Cache, LoRA Strategy, and Compute Boundaries

> Research for issues #102, #104, #108, and #109 — comparing ComfyUI, Automatic1111,
> and InvokeAI against mold's current architecture to define a clean implementation path.
>
> **Updated 2026-04-01** with codebase-validated findings from agent team research.

## Executive Summary

mold already has a multi-model LRU cache on the server (up to 3 models), but LoRA is still
locked to sequential (CLI-only) mode, expansion routing is partially implemented (server-side
exists but CLI defers when remote), and text encoders are not shared across engines. This
document synthesizes deep-dive research into three mature reference implementations and
codebase-validated findings to chart a path toward:

1. **Enhanced model cache** with shared component awareness and VRAM-pressure eviction
2. **Per-request LoRA** that works in server/eager mode (including GGUF quantized models)
3. **Complete server-side expansion** routing (CLI already defers to server when reachable)
4. **Shared component caching** for T5/CLIP/VAE/Qwen3 across model families
5. **Async CUDA stream weight streaming** for block offloading (deprioritized — see findings)

---

## 1. Current Mold Architecture

### Server State (mold-server)

```
crates/mold-server/src/state.rs (lines 87-146)

AppState {
    model_cache: Arc<Mutex<ModelCache>>,                     // LRU multi-model cache (cap: 3)
    engine_snapshot: Arc<RwLock<EngineSnapshot>>,             // Read-only view of GPU-loaded model
    model_load_lock: Arc<Mutex<()>>,                          // Serializes concurrent loads
    pull_lock: Arc<Mutex<()>>,                                // Serializes concurrent pulls
    active_generation: Arc<RwLock<Option<ActiveGenerationSnapshot>>>,
}
```

**ModelCache** (`model_cache.rs:1-179`): LRU cache of `CachedEngine` structs, max 3 models.
Each entry tracks `engine: Box<dyn InferenceEngine>`, `residency: ModelResidency` (Gpu or
Unloaded), `last_used: Instant`, `vram_bytes: u64`. Operations: `insert()`, `get_mut()` (with
LRU touch), `unload_active()`, `remove()`.

**Key invariant**: At most ONE engine has `residency == Gpu` at a time.

**Model lifecycle** (`model_manager.rs:87-175`): `ensure_model_ready()` checks cache first
(fast path if cached + on GPU), reloads from cache if unloaded (parks active model first),
or creates fresh engine if not cached. GPU memory reclaimed via `reclaim_gpu_memory()` only
when there was a prior active model.

### CUDA Context Reset (device.rs:151-192)

```rust
pub fn reclaim_gpu_memory() {
    result::ctx::synchronize();                    // Flush pending GPU ops
    let cu_device = result::device::get(0)?;       // Single-GPU assumption
    unsafe { sys::cuDevicePrimaryCtxReset_v2(cu_device) }  // Frees ALL allocations
}
```

Called between model switches. **Critical constraint for shared caching**: this reset
invalidates ALL GPU tensors, including any cached encoders. Shared GPU-resident components
would require selective cleanup instead of full context reset.

### LoRA (mold-inference)

Two paths, both requiring sequential mode:

| Path | File | Mechanism |
|------|------|-----------|
| **BF16** | `flux/lora.rs` `LoraBackend` | Implements `SimpleBackend` trait; intercepts `vb.get()` during model construction, loads from mmap, applies `W' = W + scale * (B @ A)` inline |
| **GGUF** | `flux/lora.rs` `gguf_lora_var_builder()` | Dequantizes affected tensors to F32 on CPU, merges LoRA deltas, re-quantizes to original GGML dtype |

**Why LoRA requires sequential mode**: Both paths apply deltas *during model construction*
(the `VarBuilder` phase). In eager mode the model is already constructed and weights are
loaded — there's no construction phase to hook into. The server runs eager, so LoRA requests
are rejected with "LoRA adapters require sequential loading mode."

LoRA resolution in CLI (`run.rs:473-484`): priority chain is CLI `--lora` flag → per-model
config default → global config → None. Path sent as-is in `GenerateRequest`; server assumes
path exists on its filesystem. Inference application in `flux/pipeline.rs:1235-1280`.

### Expansion (mold-cli)

**Current flow** (`run.rs:286-460`): Three paths based on `defer_expand_to_server = should_expand && !local`:

1. **Server-side (remote mode, lines 366-457)**: CLI calls `/api/expand` via HTTP BEFORE
   `generate_remote()`. Falls back to client-side if server unreachable (lines 421-454).
2. **Client-side (local mode, lines 300-365)**: `create_expander()` → LocalExpander initializes
   CUDA device (`expand.rs:119`), allocates GPU/CPU based on VRAM check.
3. **No expansion**: Prompt used as-is.

**Server side** (`routes.rs:318-435`): Two callsites — `maybe_expand_prompt()` during
`prepare_generation()` (for `req.expand == Some(true)`), and dedicated `/api/expand` endpoint.

**Key finding**: Expansion routing to server already works. The remaining gap is that the CLI
expands BEFORE the generate_remote/generate_local decision, so expansion always allocates
resources even when the server could handle everything. The fix is to defer expansion into
the generate request (`req.expand = Some(true)`) when targeting a server.

---

## 2. Reference Implementation Patterns

### 2.1 Model Cache Architecture

| Feature | ComfyUI | A1111 | InvokeAI |
|---------|---------|-------|----------|
| **Cache structure** | `current_loaded_models: Vec<LoadedModel>` | `checkpoints_loaded: OrderedDict` (state dicts) + `loaded_sd_models: Vec<Model>` | `HashMap<String, CacheRecord>` + `_cache_stack: VecDeque` (LRU ordering) |
| **What's cached** | Full model objects with patcher wrappers | State dicts (CPU tensors) separate from model instances | Full models, always in RAM, selectively in VRAM |
| **RAM eviction** | N/A (models unloaded fully) | `popitem(last=False)` on OrderedDict | LRU via `_cache_stack` pop from front |
| **VRAM eviction** | Composite score: `(-offloaded_mem, refcount, total_size)` | Manual `send_model_to_cpu()` / `send_model_to_trash()` | Smallest-first among unlocked models |
| **Concurrency guard** | Single-threaded queue | `threading.Lock` on `SdModelData` | `threading.RLock` + per-model lock counters |

**Key insight**: All three cache models, not just weights. But A1111 additionally caches
state dicts separately (CPU RAM), enabling fast model reconstruction without disk I/O.
InvokeAI's two-tier approach (always in RAM, selectively in VRAM) is the most sophisticated
and maps well to mold's architecture.

### 2.2 LoRA Strategies

| Feature | ComfyUI | A1111 | InvokeAI |
|---------|---------|-------|----------|
| **Application method** | In-place patch via `ModelPatcher` hooks; backup originals | Monkey-patched forward hooks; lazy merge on first forward | Direct weight merge OR sidecar (forward-pass residual) |
| **Eager mode LoRA** | Yes — `patch_hooks()` with `hook_backup` system | Yes — `network_apply_weights()` checks `network_current_names` | Yes — both direct and sidecar paths work with loaded models |
| **Quantized model LoRA** | Not native (uses dequant paths) | Not native | **Sidecar**: `output = quantized_forward(x) + lora_residual(x)` |
| **Weight caching** | `cached_hook_patches: dict[HookGroup, dict[key, Tensor]]` | `network_current_names` tuple comparison to skip re-merge | `OriginalWeightsStorage` with CPU state_dict for fast restore |
| **Unpatch mechanism** | Restore from `hook_backup` (CPU copies) | Restore from `network_weights_backup` (CPU copies) | Context manager: restore from `OriginalWeightsStorage` or `clear_patches()` |
| **Multiple LoRAs** | Additive: patches list per key, `calculate_weight()` applies all | Sequential: iterate `loaded_networks`, sum deltas | Sequential or combined via `_aggregate_patch_parameters()` |

**Critical finding — InvokeAI's sidecar approach for GGUF**:

```python
# Instead of: dequantize -> merge -> re-quantize (current mold approach)
# Do this:    output = quantized_matmul(x, W) + (x @ A) @ B * scale
```

This avoids all dequantization overhead and preserves the quantized weights untouched. The
LoRA A/B matrices are small (rank 16-64 typically) and computed in full precision as a
forward-pass residual. This is the single biggest architectural improvement available to mold.

### 2.3 VRAM Management

| Feature | ComfyUI | A1111 | InvokeAI |
|---------|---------|-------|----------|
| **Partial loading** | Per-module: sorted by offload cost, greedy packing | Per-module: `--lowvram` (UNet blocks), `--medvram` (major components) | **Per-tensor**: required vs optional, budget-constrained |
| **Lazy computation** | `weight_function` callbacks on offloaded modules | Forward pre-hooks swap modules GPU<->CPU | `CustomModuleMixin` with `set_device_autocasting_enabled()` |
| **Async transfers** | 2 CUDA streams, round-robin, per-stream cast buffers | None | None |
| **Pinned memory** | `cudaHostRegister`, budget: 90% RAM (Linux) | None | None |
| **Memory budget** | `EXTRA_RESERVED_VRAM`: 400-600MB; `MIN_WEIGHT_MEMORY_RATIO`: 40% | N/A (modes are boolean flags) | `50% RAM - 2GB`, capped at `1x VRAM`, min `4GB` |

**Key insight**: ComfyUI's async CUDA stream transfers and pinned memory are the most
advanced VRAM optimization. InvokeAI's per-tensor partial loading is the most granular.

**Codebase-validated assessment (issue #109)**: Async CUDA streams for mold's block offloading
(`flux/offload.rs:557-575`) would yield only ~5-10% speedup. The bottleneck is block data
dependencies (block N+1 input = block N output), limiting overlap to prefetching N+1 while
computing N. Per-block cost: ~5ms H2D transfer + ~10ms compute + ~2ms drop. Overlap saves
only the transfer time (~5ms) against a ~17ms total. Additionally, candle-core-mold 0.9.3
hardcodes a single CUDA stream per device (`CudaDevice.stream: Arc<CudaStream>`, line 39
of `cuda_backend/device.rs`). Adding secondary stream support would require forking candle's
`to_device()` to accept a stream parameter. **Recommendation: deprioritize until candle adds
native stream parameter support.**

### 2.4 Shared Components

| Feature | ComfyUI | A1111 | InvokeAI |
|---------|---------|-------|----------|
| **Text encoder sharing** | CLIP instances share `cond_stage_model`, separate patchers | Part of model object, not independently cached | Submodel keying: `model_key:text_encoder` |
| **VAE caching** | Independent tracking in `current_loaded_models` | Part of model, separate `sd_vae` module | Submodel keying: `model_key:vae` |
| **Conditioning cache** | `HierarchicalCache` / `LRUCache` for node outputs | None | `MemoryInvocationCache` keyed by invocation hash |

**Codebase-validated findings (issue #108)**:

- **All 25 FLUX variants** share identical T5-XXL (9.7GB), CLIP-L (246MB), and VAE (335MB)
  files (`manifest.rs:251-294`). Only the transformer differs per variant.
- **Flux.2 Klein-4B variants** share Qwen3-1.7B encoder and VAE; Klein-9B shares Qwen3-4B.
- **Per-engine ownership**: Each engine owns separate encoder instances — no sharing across
  engines. FluxEngine owns T5+CLIP directly (`flux/pipeline.rs:584-600`), SD3Engine wraps
  CLIP-L+CLIP-G+T5 in `SD3TripleEncoder`.
- **Drop-and-reload**: Sequential mode drops T5/CLIP after encoding to free VRAM for
  transformer (`pipeline.rs:1100-1170`). Eager mode keeps all resident.
- **CUDA context reset invalidates everything**: `reclaim_gpu_memory()` calls
  `cuDevicePrimaryCtxReset_v2` which frees ALL GPU allocations. Shared GPU-resident
  components would need selective cleanup instead of full reset.

InvokeAI's submodel keying pattern (`model_key:text_encoder`) maps cleanly to this. Caching
T5/CLIP/VAE independently would save ~10GB reload when switching FLUX variants.

---

## 3. Cross-Cutting Analysis

### What All Three Agree On

1. **Multi-model caching is essential** — single-model-at-a-time forces cold reloads on every switch
2. **CPU RAM as intermediate tier** — park idle models on CPU, don't drop entirely
3. **CPU backup of original weights** — enables fast LoRA unpatch/restore (memcpy vs disk I/O)
4. **Single-GPU inference serialization** — all three process one generation at a time
5. **LoRA as additive patches** — `W' = W + delta`, where the delta computation varies
6. **Lock-based eviction protection** — models in use must not be evicted

### Where They Diverge

| Decision | Best for mold | Rationale |
|----------|---------------|-----------|
| Eviction policy | InvokeAI (LRU RAM + smallest-first VRAM) | Maps well to Rust ownership; smallest-first VRAM maximizes cache capacity |
| LoRA for quantized | InvokeAI (sidecar) | Eliminates mold's expensive dequant→merge→re-quant cycle |
| LoRA for BF16 | ComfyUI (hook cache) | `cached_hook_patches` avoids recomputing same LoRA combo |
| Partial loading | ComfyUI (per-module + async streams) | mold already does block-level offloading; async streams are the natural next step |
| Shared components | InvokeAI (submodel keying) | Clean key scheme; candle already separates encoder/transformer/VAE |
| Memory budget | InvokeAI formula | Conservative, well-tested: `min(max(50% RAM - 2GB, 4GB), 1x VRAM)` |

---

## 4. Recommended Architecture for Mold

### 4.1 ModelCache (Enhance Existing)

The server already has `ModelCache` (`model_cache.rs:1-179`) with LRU eviction and
`Gpu`/`Unloaded` residency states. Enhance it with:

1. **CPU parking tier** — Add `Parked` state alongside existing `Gpu`/`Unloaded`. Parked
   engines hold weights in CPU RAM for fast reload (~1-2s vs ~30-60s from disk).
2. **RAM budget enforcement** — Track `ram_bytes` per entry, drop oldest Parked entries
   when budget exceeded. Use InvokeAI formula: `min(max(50% RAM - 2GB, 4GB), 1x VRAM)`.
3. **Lock-based eviction protection** — Add `lock_count: u32` to prevent evicting
   models currently in use by the generation queue.
4. **Selective GPU cleanup** — Replace `cuDevicePrimaryCtxReset_v2` (which nukes ALL GPU
   state) with per-engine `unload()` calls when shared components exist. Full context
   reset only when no shared components are cached.

**Shared components**: Cache T5, CLIP, and VAE independently with submodel keys
(`"flux-dev:t5"`, `"flux-dev:vae"`). When switching FLUX models, only the transformer
needs reloading. Requires `Arc`-wrapping shared tensors so multiple engines can reference
them.

### 4.2 Sidecar LoRA for GGUF (the big win)

Instead of the current dequantize→merge→re-quantize path, implement forward-pass residuals:

```rust
pub trait LayerPatch: Send + Sync {
    /// Compute the LoRA residual for this layer
    fn residual(&self, input: &Tensor, weight: f32) -> Result<Tensor>;

    /// Total bytes of patch tensors
    fn size_bytes(&self) -> usize;
}

pub struct LoRASidecar {
    up: Tensor,     // [out_features, rank] on GPU
    down: Tensor,   // [rank, in_features] on GPU
    alpha: f64,
    rank: usize,
}

impl LayerPatch for LoRASidecar {
    fn residual(&self, input: &Tensor, weight: f32) -> Result<Tensor> {
        let scale = weight * (self.alpha / self.rank as f64);
        // input @ down^T @ up^T * scale
        let x = input.matmul(&self.down.t()?)?;
        let x = x.matmul(&self.up.t()?)?;
        (x * scale)
    }
}
```

During inference:
```rust
// In the quantized linear forward:
let base_output = quantized_matmul(input, &self.weight)?;
let lora_output = self.sidecar.as_ref()
    .map(|s| s.residual(input, lora_scale))
    .transpose()?
    .unwrap_or_else(|| Tensor::zeros_like(&base_output));
base_output + lora_output
```

**Benefits**:
- No dequantization, no re-quantization, no VRAM spike
- Base weights stay quantized and read-only (mmap-friendly)
- LoRA A/B matrices are tiny (~5-50MB for rank 16-64)
- Multiple LoRAs stack naturally: sum residuals
- Works with block-level offloading (sidecars travel with blocks)

### 4.3 Direct Merge LoRA for BF16 (with caching)

For non-quantized models, keep the direct merge approach but make it work in eager mode:

```rust
pub struct PatchedWeights {
    /// CPU backup of original weights for fast unpatch
    original: HashMap<String, Tensor>,

    /// Cached patched weights keyed by (tensor_name, lora_path, scale)
    cache: HashMap<(String, PathBuf, OrderedFloat<f32>), Tensor>,
}
```

**Flow**:
1. Before generation: check if requested LoRA combo matches cache
2. If cached: apply cached weights (fast memcpy from CPU)
3. If not: backup originals to CPU, compute `W + B @ A`, cache result
4. After generation: restore originals if next request has different LoRA
5. Same LoRA combo on consecutive requests: no-op (weights already patched)

### 4.4 Server-Side Expansion

Move expansion from CLI to server when server is reachable:

```
CLI (new flow):
  1. Build GenerateRequest with expand=true, raw prompt
  2. Try server -> POST /api/generate/stream
     Server internally: maybe_expand_prompt() -> generate()
  3. If unreachable -> expand locally, then generate locally
```

**Changes required**:
- `run.rs`: Skip local expansion when not using `--local`; set `request.expand = Some(true)`
- `generate.rs`: Pass raw prompt + expand flag to server
- Server already handles this via `maybe_expand_prompt()` — no server changes needed
- CLI retains local expansion for `--local` mode

### 4.5 Memory Budget

```rust
pub fn calculate_cache_budget() -> CacheBudget {
    let total_ram = sys_info::mem_total();
    let total_vram = free_vram_bytes() + used_vram_bytes(); // from device.rs

    // RAM: 50% of system RAM minus 2GB baseline, min 4GB, capped at 1x VRAM
    let ram_budget = ((total_ram / 2).saturating_sub(2 * GB))
        .max(4 * GB)
        .min(total_vram);

    // VRAM: total minus 1.5GB working memory (activations, intermediates)
    let vram_budget = total_vram.saturating_sub(1_500_000_000);

    CacheBudget { ram_budget, vram_budget }
}
```

---

## 5. Phased Implementation Plan

### Phase 1: Complete Server-Side Expansion Routing (fixes #102, low risk)

**Goal**: CLI defers expansion into `GenerateRequest` when targeting a server, eliminating
client-side GPU allocation for expansion.

**Status**: Partially done — CLI already routes to `/api/expand` when remote (`run.rs:366-457`).
Remaining work: instead of expanding BEFORE the generate call, pass `expand: Some(true)` in
the request so the server handles it atomically during `prepare_generation()`.

**Files to change**:
- `crates/mold-cli/src/commands/run.rs` — defer expansion into request when not `--local`
- `crates/mold-cli/src/commands/generate.rs` — pass `expand: Some(true)` in request

**No server changes needed** — `maybe_expand_prompt()` in `routes.rs:319-362` already works.

**Estimated scope**: ~50 lines changed

### Phase 2: Enhance ModelCache with CPU Parking (core of #104)

**Goal**: Add CPU parking tier to existing `ModelCache` for fast model switching.

**Status**: `ModelCache` already exists (`model_cache.rs:1-179`) with LRU eviction and
`Gpu`/`Unloaded` residency. Missing: CPU parking tier (hold weights in RAM), RAM budget
enforcement, and lock-based eviction protection.

**Files to change**:
- `crates/mold-server/src/model_cache.rs` — add `Parked` residency state, RAM budget, locks
- `crates/mold-server/src/model_manager.rs` — use `park()` instead of `unload()` when switching
- `crates/mold-inference/src/engine.rs` — add `to_cpu()` / `to_gpu()` to trait
- `crates/mold-inference/src/device.rs` — selective GPU cleanup (skip full context reset when
  shared components exist)

**Estimated scope**: ~200 lines new, ~150 lines changed

### Phase 3: Sidecar LoRA for GGUF (biggest value, addresses #104)

**Goal**: Enable LoRA on the server for quantized models without sequential mode.

**New files**:
- `crates/mold-inference/src/flux/sidecar.rs` — `LoRASidecar`, `LayerPatch` trait
- Modifications to candle quantized model types to support sidecar attachment

**Files to change**:
- `crates/mold-inference/src/flux/lora.rs` — add sidecar path alongside existing merge path
- `crates/mold-inference/src/flux/pipeline.rs` — attach sidecars during generation, not construction
- `crates/mold-server/src/routes.rs` — remove "LoRA requires sequential" rejection

**Key challenge**: candle's quantized `Linear` doesn't support forward-pass hooks. We'll need
to either:
- (a) Wrap quantized layers with a `SidecarLinear` that chains base + residual, or
- (b) Modify the forward pass in each engine's denoising loop to apply residuals externally

Option (a) is cleaner but requires changes per model family. Option (b) is more surgical
but couples LoRA logic to the denoising loop.

**Estimated scope**: ~600 lines new, ~150 lines changed

### Phase 4: Direct Merge LoRA for BF16 in Eager Mode

**Goal**: Enable LoRA for BF16 models on the server without dropping to sequential.

**Files to change**:
- `crates/mold-inference/src/flux/lora.rs` — add `PatchedWeights` with CPU backup + cache
- `crates/mold-inference/src/flux/pipeline.rs` — apply/unpatch LoRA per-generation
- `crates/mold-inference/src/engine.rs` — add `apply_lora()` / `remove_lora()` to trait

**Approach**: After model is loaded in eager mode, apply LoRA by iterating model parameters,
backing up originals to CPU, and merging deltas. Reverse on next non-LoRA request (or
different LoRA).

**Estimated scope**: ~300 lines new, ~100 lines changed

### Phase 5: Shared Component Caching

**Goal**: Cache T5/CLIP/VAE independently so switching between FLUX models only reloads
the transformer.

**Files to change**:
- `crates/mold-inference/src/factory.rs` — accept optional pre-loaded encoders/VAE
- `crates/mold-inference/src/flux/pipeline.rs` — support injecting cached components
- `crates/mold-server/src/model_cache.rs` — submodel key scheme (`"flux-dev:t5"`)

**Key challenge**: candle models own their components via struct fields. Sharing requires
`Arc` wrapping or extracting components into a separate cache layer that engines borrow from.

**Estimated scope**: ~400 lines new, ~200 lines changed

### Phase 6: LoRA Weight Caching (optimization)

**Goal**: Cache computed `W + B @ A` results so repeated LoRA requests skip computation.

**Files to change**:
- `crates/mold-inference/src/flux/lora.rs` — add cache keyed by `(tensor_name, lora_hash, scale)`
- `crates/mold-inference/src/flux/pipeline.rs` — check cache before computing

**Estimated scope**: ~150 lines new, ~50 lines changed

### Phase 7: Async Weight Streaming (deprioritized — #109)

**Goal**: Overlap CPU->GPU transfers with GPU computation during block-level offloading.

**Status**: Deprioritized based on codebase research. Expected speedup is only ~5-10% due to:
- Block data dependencies (block N+1 input = block N output) limit overlap to prefetch only
- Per-block breakdown: ~5ms H2D + ~10ms compute + ~2ms drop = ~17ms; overlap saves ~5ms max
- candle-core-mold 0.9.3 hardcodes single CUDA stream per device (`device.rs:39`)
- Adding secondary stream support requires forking `to_device()` — maintenance burden

**Better alternatives**: GGUF quantization (already reduces offload overhead dramatically),
or upgrading to 24GB+ VRAM cards for eager mode (3-5x faster, no offloading needed).

**Revisit when**: candle upstream adds stream parameter to `Tensor::to_device()`.

**If pursued**: ComfyUI pattern — 2 CUDA streams, round-robin, `cuEventRecord`/`cuStreamWaitEvent`
for synchronization. ~300 lines plus candle fork changes.

---

## 6. Priority and Dependencies

```
Phase 1 (expansion)     -----> can ship independently (~50 LOC, fixes #102)
Phase 2 (cache parking) -----> enhances existing cache, foundation for Phases 3-6
Phase 3 (sidecar GGUF)  -----> depends on Phase 2 (needs cache for engine lifecycle)
Phase 4 (BF16 eager)    -----> depends on Phase 2
Phase 5 (shared comps)  -----> depends on Phase 2 + selective GPU cleanup
Phase 6 (LoRA cache)    -----> depends on Phase 3 or 4
Phase 7 (async streams) -----> DEPRIORITIZED (~5-10% gain, requires candle fork)
```

**Recommended order**: 1 → 2 → 3 → 4 → 5 → 6. Skip 7 until candle adds stream support.

Phase 1 is a quick win that completes #102. Phase 2 enhances the existing `ModelCache` with
CPU parking. Phase 3 is the biggest value — sidecar LoRA enables server-side LoRA for GGUF
without sequential mode. Phase 5 (shared components) is the most architecturally complex,
requiring changes to CUDA context management and `Arc`-wrapping of encoder tensors.

**Issues resolved per phase**:
- Phase 1: #102 (compute boundaries)
- Phases 2-4: #104 (multi-model cache + per-request LoRA)
- Phase 5: #108 (shared component caching)
- Phase 7: #109 (async CUDA streams — deprioritized)

---

## Appendix: Reference Code Locations

### ComfyUI (tmp/ComfyUI/)

| System | File | Key Lines |
|--------|------|-----------|
| Model cache | `comfy/model_management.py` | 499 (`current_loaded_models`), 531 (`LoadedModel`), 664 (`free_memory`), 718 (`load_models_gpu`) |
| Eviction | `comfy/model_management.py` | 674 (composite score), 682 (`DISABLE_SMART_MEMORY`) |
| ModelPatcher | `comfy/model_patcher.py` | 232 (class), 321 (`clone`), 684 (`patch_weight_to_device`), 766 (`load`) |
| Hook LoRA | `comfy/model_patcher.py` | 1333 (`patch_hooks`), 1384 (`patch_hook_weight_to_device`) |
| LoRA loading | `comfy/lora.py` | 37 (`load_lora`), 406 (`calculate_weight`) |
| Async streams | `comfy/model_management.py` | 1156-1261 (NUM_STREAMS, round-robin) |
| Pinned memory | `comfy/model_management.py` | 1323-1411 (`pin_memory`, budget) |

### A1111 (tmp/stable-diffusion-webui/)

| System | File | Key Lines |
|--------|------|-----------|
| State dict cache | `modules/sd_models.py` | 27 (`checkpoints_loaded`), 333-347 (`get_checkpoint_state_dict`), 524 (LRU eviction) |
| Model instances | `modules/sd_models.py` | 676-717 (`SdModelData`), 894-937 (`reuse_model_from_already_loaded`) |
| CPU parking | `modules/sd_models.py` | 738-765 (`send_model_to_cpu`, `send_model_to_trash`) |
| LoRA networks | `extensions-builtin/Lora/networks.py` | 281-366 (`load_networks`), 376-408 (backup/restore) |
| Lazy merge | `extensions-builtin/Lora/networks.py` | 411-543 (`network_apply_weights`) |
| Lowvram hooks | `modules/lowvram.py` | 34-162 (`setup_for_low_vram`) |

### InvokeAI (tmp/InvokeAI/)

| System | File | Key Lines |
|--------|------|-----------|
| ModelCache | `invokeai/backend/model_manager/load/model_cache/model_cache.py` | 106-868 |
| Two-tier eviction | Same | 702-733 (VRAM smallest-first), 823-836 (RAM LRU) |
| Partial loading | `model_cache/cached_model/cached_model_with_partial_load.py` | 250-321 (tensor classification) |
| Memory budget | `model_cache/model_cache.py` | 616-677 (RAM heuristics) |
| Sidecar LoRA | `backend/patches/layer_patcher.py` | 110-149 (smart decision), 220-240 (wrapper patch) |
| LoRA forward | `torch_module_autocast/custom_modules/custom_linear.py` | 14-27 (`linear_lora_forward`) |
| Original weights | `backend/patches/original_weights_storage.py` | 10-40 |
| CustomModuleMixin | `torch_module_autocast/torch_module_autocast.py` | 95-106 (module wrapping) |
