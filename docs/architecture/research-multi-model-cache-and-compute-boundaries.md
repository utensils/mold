# Architecture Research: Multi-Model Cache, LoRA Strategy, and Compute Boundaries

> Research for issues #102 and #104 — comparing ComfyUI, Automatic1111, and InvokeAI
> against mold's current architecture to define a clean implementation path.

## Executive Summary

mold currently operates with a single-model slot on the server, LoRA locked to sequential
(CLI-only) mode, and prompt expansion always running client-side. This document synthesizes
deep-dive research into three mature reference implementations to chart a path toward:

1. **Multi-model cache** with VRAM-pressure eviction
2. **Per-request LoRA** that works in server/eager mode (including GGUF quantized models)
3. **Server-side expansion** eliminating wasted client GPU allocation
4. **Shared component caching** for text encoders and VAEs across model families

---

## 1. Current Mold Architecture

### Server State (mold-server)

```
crates/mold-server/src/state.rs

AppState {
    engine: Arc<Mutex<Option<Box<dyn InferenceEngine>>>>,   // SINGLE engine slot
    engine_snapshot: Arc<RwLock<EngineSnapshot>>,            // Lightweight read-only view
    model_load_lock: Arc<Mutex<()>>,                         // Guards concurrent loads
    queue: QueueHandle,                                      // FIFO generation queue (cap: 16)
}
```

**Model lifecycle**: Full unload-before-load on every model switch. The engine is `.take()`'d
from the Option slot during `spawn_blocking` to avoid holding the async mutex, with an
`OptionRestoreGuard` for panic safety.

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

### Expansion (mold-cli)

```
CLI (run.rs):
  1. Expand prompt locally (allocates GPU!)     <-- always client-side
  2. Build GenerateRequest (expand=None)
  3. Try server -> POST /api/generate/stream
  4. If unreachable -> fallback to local
```

The server has `maybe_expand_prompt()` and `/api/expand` built in but the CLI never uses them.
The expand LLM competes for VRAM with the server's loaded model.

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
For mold's block-level offloading, ComfyUI's async stream pattern could overlap block N+1
transfer with block N computation — a meaningful speedup.

### 2.4 Shared Components

| Feature | ComfyUI | A1111 | InvokeAI |
|---------|---------|-------|----------|
| **Text encoder sharing** | CLIP instances share `cond_stage_model`, separate patchers | Part of model object, not independently cached | Submodel keying: `model_key:text_encoder` |
| **VAE caching** | Independent tracking in `current_loaded_models` | Part of model, separate `sd_vae` module | Submodel keying: `model_key:vae` |
| **Conditioning cache** | `HierarchicalCache` / `LRUCache` for node outputs | None | `MemoryInvocationCache` keyed by invocation hash |

**Key insight for mold**: T5-XXL (10GB), CLIP-L (250MB), and VAE (160MB) are shared across
all FLUX models. Caching them independently would make switching between flux-schnell and
flux-dev near-instant (only the transformer changes). InvokeAI's submodel keying pattern
maps cleanly to this.

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

### 4.1 ModelCache

Replace the single `Option<Box<dyn InferenceEngine>>` with a two-tier cache:

```rust
pub struct ModelCache {
    /// All cached engines, keyed by model name
    engines: HashMap<String, CacheEntry>,

    /// LRU ordering (front = oldest, back = most recent)
    lru_order: VecDeque<String>,

    /// Currently active (on GPU) engine name
    active: Option<String>,

    /// Memory budgets
    max_ram_bytes: usize,   // For parked engines (CPU state)
    max_vram_bytes: usize,  // For active + partially-loaded engines
}

pub struct CacheEntry {
    engine: Box<dyn InferenceEngine>,
    state: EngineState,
    last_used: Instant,
    vram_bytes: usize,
    ram_bytes: usize,
    lock_count: u32,
}

pub enum EngineState {
    /// Fully loaded on GPU, ready for inference
    Active,
    /// Weights in CPU RAM, fast to reload (~1-2s)
    Parked,
    /// Unloaded, must reconstruct from disk (~30-60s)
    Cold,
}
```

**Eviction policy**:
- **VRAM pressure**: When loading a new model, park (move to Parked) the least-recently-used
  unlocked Active engine. If still insufficient, unload Parked engines smallest-first.
- **RAM pressure**: Drop oldest Parked entries via LRU when RAM budget exceeded.

**Shared components**: Cache T5, CLIP, and VAE independently with submodel keys
(`"flux-dev:t5"`, `"flux-dev:vae"`). When switching FLUX models, only the transformer
needs reloading.

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

### Phase 1: Server-Side Expansion (fixes #102, low risk)

**Goal**: CLI delegates expansion to server when reachable, eliminating wasted GPU context.

**Files to change**:
- `crates/mold-cli/src/commands/run.rs` — skip local expansion when not `--local`
- `crates/mold-cli/src/commands/generate.rs` — pass `expand: Some(true)` in request

**No server changes needed** — `maybe_expand_prompt()` already works.

**Estimated scope**: ~50 lines changed

### Phase 2: ModelCache Foundation (core of #104)

**Goal**: Replace single engine slot with multi-model cache.

**New file**: `crates/mold-server/src/model_cache.rs`
- `ModelCache` struct with HashMap + VecDeque LRU
- `CacheEntry` with engine, state, timestamps, memory tracking
- `get_or_load()`, `park()`, `evict()`, `lock()`, `unlock()` methods

**Files to change**:
- `crates/mold-server/src/state.rs` — replace `engine: Option<...>` with `ModelCache`
- `crates/mold-server/src/model_manager.rs` — use cache instead of direct slot manipulation
- `crates/mold-server/src/routes.rs` — acquire engine from cache, release after generation
- `crates/mold-server/src/queue.rs` — lock model during generation, unlock after
- `crates/mold-inference/src/engine.rs` — add `vram_bytes()` and `to_cpu()` / `to_gpu()` to trait

**Estimated scope**: ~400 lines new, ~200 lines changed

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

### Phase 7: Async Weight Streaming (stretch goal)

**Goal**: Overlap CPU->GPU transfers with GPU computation during block-level offloading.

**Requires**: CUDA stream support in candle (may need fork changes).

**Approach** (from ComfyUI pattern):
- 2 CUDA streams, round-robin
- While computing with block N on default stream, transfer block N+1 on async stream
- `stream.wait_stream()` before reading transferred weights

**Estimated scope**: ~300 lines, plus potential candle fork work

---

## 6. Priority and Dependencies

```
Phase 1 (expansion)     -----> can ship independently
Phase 2 (cache)         -----> foundation for everything below
Phase 3 (sidecar GGUF)  -----> depends on Phase 2 (needs cache for engine lifecycle)
Phase 4 (BF16 eager)    -----> depends on Phase 2
Phase 5 (shared comps)  -----> depends on Phase 2
Phase 6 (LoRA cache)    -----> depends on Phase 3 or 4
Phase 7 (async streams) -----> independent, but benefits from Phase 2
```

**Recommended order**: 1 -> 2 -> 3 -> 4 -> 5 -> 6 -> 7

Phase 1 is a quick win that fixes #102 with minimal risk. Phase 2+3 together deliver the
core value of #104 (multi-model cache + server LoRA for GGUF). Phases 4-7 are progressive
enhancements.

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
