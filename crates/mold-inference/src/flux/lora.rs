use std::collections::{HashMap, VecDeque};
use std::hash::Hash;
use std::path::Path;
use std::sync::{Arc, Mutex, OnceLock};

use anyhow::{bail, Result};
use candle_core::{DType, Device, Tensor};

use crate::progress::ProgressReporter;

/// Key for cached LoRA delta tensors.
/// `patch_index` disambiguates multiple patches on the same fused tensor
/// (e.g., Q/K/V slices of a fused QKV weight each get a separate delta).
#[derive(Hash, Eq, PartialEq, Clone)]
struct LoraCacheKey {
    tensor_name: String,
    patch_index: usize,
    lora_path_hash: u64,
    scale_bits: u64,
}

/// CPU-resident cache of pre-computed LoRA delta tensors (B @ A * scale).
/// Avoids expensive matmul recomputation when the same LoRA is applied across rebuilds.
pub(crate) struct LoraDeltaCache {
    deltas: HashMap<LoraCacheKey, Tensor>,
}

impl LoraDeltaCache {
    pub fn new() -> Self {
        Self {
            deltas: HashMap::new(),
        }
    }

    fn get(&self, key: &LoraCacheKey) -> Option<&Tensor> {
        self.deltas.get(key)
    }

    fn insert(&mut self, key: LoraCacheKey, delta: Tensor) {
        self.deltas.insert(key, delta);
    }
}

/// Identity for a parsed `LoraAdapter` on disk. Combining the path
/// hash with the file's modification time means a user who edits a
/// `.safetensors` in place (e.g. re-exports from a trainer) gets a
/// fresh parse on the next load — no stale state.
#[derive(Hash, Eq, PartialEq, Clone, Debug)]
pub(crate) struct ParsedLoraCacheKey {
    path_hash: u64,
    file_mtime_nanos: i128,
}

impl ParsedLoraCacheKey {
    /// Build a cache key from a path on disk. Falls back gracefully
    /// when the file system can't report an mtime (read-only mounts,
    /// some FUSE backends): `i128::MIN` is used as the sentinel so
    /// every load on such a path becomes a miss, which is the safe
    /// behaviour.
    fn from_path(path: &Path) -> Result<Self> {
        use std::hash::Hasher;
        let canonical = std::fs::canonicalize(path).unwrap_or_else(|_| path.to_path_buf());

        let mut hasher = std::collections::hash_map::DefaultHasher::new();
        canonical.hash(&mut hasher);
        let path_hash = hasher.finish();

        let file_mtime_nanos = std::fs::metadata(&canonical)
            .and_then(|m| m.modified())
            .ok()
            .and_then(|t| t.duration_since(std::time::UNIX_EPOCH).ok())
            .map(|d| d.as_nanos() as i128)
            .unwrap_or(i128::MIN);

        Ok(Self {
            path_hash,
            file_mtime_nanos,
        })
    }
}

/// Tiny FIFO cache for parsed `LoraAdapter`s. 4 slots covers the
/// "user toggling a LoRA on/off" and "user scrubbing the strength
/// slider" cases without holding more than ~80 MB of CPU-resident
/// adapter weights in memory. We don't bother with a true LRU
/// (mostly-recently-used) policy at this size — FIFO with capacity 4
/// is correct enough and avoids the borrow contortions of an
/// `LruCache::get(&mut self, ...)` that touches order-tracking.
const PARSED_LORA_CACHE_CAPACITY: usize = 4;

struct ParsedLoraCache {
    order: VecDeque<ParsedLoraCacheKey>,
    entries: HashMap<ParsedLoraCacheKey, Arc<LoraAdapter>>,
}

impl ParsedLoraCache {
    fn new() -> Self {
        Self {
            order: VecDeque::with_capacity(PARSED_LORA_CACHE_CAPACITY),
            entries: HashMap::with_capacity(PARSED_LORA_CACHE_CAPACITY),
        }
    }

    fn get(&self, key: &ParsedLoraCacheKey) -> Option<Arc<LoraAdapter>> {
        self.entries.get(key).map(Arc::clone)
    }

    fn insert(&mut self, key: ParsedLoraCacheKey, adapter: Arc<LoraAdapter>) {
        if self.entries.contains_key(&key) {
            // Refresh: drop the existing entry from the FIFO order so
            // we don't double-count it and silently leak a slot.
            self.order.retain(|existing| existing != &key);
        }
        self.entries.insert(key.clone(), adapter);
        self.order.push_back(key);
        while self.entries.len() > PARSED_LORA_CACHE_CAPACITY {
            if let Some(oldest) = self.order.pop_front() {
                self.entries.remove(&oldest);
            } else {
                break;
            }
        }
    }
}

fn parsed_lora_cache() -> &'static Mutex<ParsedLoraCache> {
    static CACHE: OnceLock<Mutex<ParsedLoraCache>> = OnceLock::new();
    CACHE.get_or_init(|| Mutex::new(ParsedLoraCache::new()))
}

/// Load a LoRA adapter, returning a cached `Arc<LoraAdapter>` when
/// the same `(path, mtime)` was loaded recently. The cache survives
/// across transformer rebuilds — slider scrubbing on a single LoRA
/// hits this on every step after the first, saving the ~200-500 ms
/// of `safetensors::load` per rebuild.
///
/// On mtime change the previous entry is shadowed by a new one with
/// a different cache key; the old entry stays resident until FIFO
/// evicts it (acceptable — adapters are tens of MB on CPU).
pub(crate) fn get_or_load_adapter(path: &Path) -> Result<Arc<LoraAdapter>> {
    let key = ParsedLoraCacheKey::from_path(path)?;
    {
        let cache = parsed_lora_cache().lock().unwrap();
        if let Some(adapter) = cache.get(&key) {
            tracing::debug!(
                path = %path.display(),
                "parsed-LoRA cache hit"
            );
            return Ok(adapter);
        }
    }
    let adapter = Arc::new(LoraAdapter::load(path)?);
    {
        let mut cache = parsed_lora_cache().lock().unwrap();
        cache.insert(key, Arc::clone(&adapter));
    }
    Ok(adapter)
}

#[cfg(test)]
fn clear_parsed_lora_cache_for_test() {
    let mut cache = parsed_lora_cache().lock().unwrap();
    cache.entries.clear();
    cache.order.clear();
}

#[cfg(test)]
fn lock_parsed_lora_cache_tests() -> std::sync::MutexGuard<'static, ()> {
    static TEST_LOCK: OnceLock<Mutex<()>> = OnceLock::new();
    TEST_LOCK.get_or_init(|| Mutex::new(())).lock().unwrap()
}

/// A parsed LoRA adapter: pairs of (A, B) weight matrices keyed by layer name.
pub(crate) struct LoraAdapter {
    /// Map from diffusers layer name (without lora_A/lora_B suffix) to (A, B) tensors.
    pub layers: HashMap<String, LoraLayer>,
    pub rank: usize,
}

pub(crate) struct LoraLayer {
    pub a: Tensor,
    pub b: Tensor,
    /// Per-layer alpha (if present in the safetensors file).
    pub alpha: Option<f64>,
}

/// Direction of a single tensor inside a LoRA pair: `.lora_A.weight`
/// (the `(rank, in)` down-projection) or `.lora_B.weight` (the
/// `(out, rank)` up-projection). [`classify_lora_key`] returns this
/// alongside the layer stem so the loader can pair them up.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum LoraDirection {
    Down,
    Up,
}

/// Suffixes that mark the down-projection (`A`) tensor. Order matters:
/// the matcher returns on the first hit, so list more-specific suffixes
/// (e.g. `.lora_linear_layer.down.weight`) before generic ones
/// (`.lora_down.weight`) when they could shadow each other.
///
/// References:
/// - Diffusers / PEFT canonical: `.lora_A.weight`
/// - Kohya / sd-scripts: `.lora_down.weight`
/// - OneTrainer (note inverted naming — the *down* matrix is
///   `lora_linear_layer.down.weight`):
///   <https://github.com/comfyanonymous/ComfyUI/blob/master/comfy/weight_adapter/lora.py>
/// - PEFT default-adapter (when a model has multiple PEFT adapters
///   and the user picked the literally-named "default" one):
///   `.lora_A.default.weight`.
const LORA_DOWN_SUFFIXES: &[&str] = &[
    ".lora_linear_layer.down.weight",
    ".lora_A.default.weight",
    ".lora_A.weight",
    ".lora_down.weight",
];

/// Suffixes that mark the up-projection (`B`) tensor.
///
/// `.lora_B` (without `.weight`) is the Mochi / certain video-LoRA
/// edge case where the trainer dropped the `.weight` segment. We
/// keep it last so it never shadows `.lora_B.weight` or
/// `.lora_B.default.weight` for keys that end with the longer suffix.
const LORA_UP_SUFFIXES: &[&str] = &[
    ".lora_linear_layer.up.weight",
    ".lora_B.default.weight",
    ".lora_B.weight",
    ".lora_up.weight",
    ".lora_B",
];

/// Classify a tensor key from a LoRA safetensors file by its trailing
/// convention. Returns `Some((direction, stem))` where `stem` is the
/// layer name with the LoRA suffix stripped, or `None` for keys that
/// aren't LoRA pair tensors (e.g. `.alpha` scalars or unrelated state).
///
/// This is a pure function — no I/O, no allocation beyond the slice
/// reference — so it's exhaustively unit-tested against the suffix
/// matrix (Diffusers / Kohya / OneTrainer / PEFT default-adapter /
/// Mochi edge case) without needing a synthetic safetensors fixture.
/// Read a LoRA `.alpha` scalar whatever dtype the trainer stored it in.
///
/// kohya and diffusers write `F32`, but the Wan 2.2 Lightning LoRAs ship
/// **`I64`** alphas (value 8 against rank 64, so the intended scale is
/// `8/64 = 0.125`). The previous `tensor.to_scalar::<f32>()` fails on `I64`
/// storage inside `cpu_storage_as_slice`, and the surrounding `if let Ok`
/// swallowed the error — leaving `alpha: None`, which falls back to the
/// caller's raw scale and applies such a LoRA **8x too strong, silently**.
///
/// Also tolerates the shape-`[1]` spelling some trainers emit alongside the
/// canonical rank-0 scalar.
pub(crate) fn read_alpha_scalar(tensor: &Tensor) -> Option<f64> {
    let values = tensor
        .to_dtype(DType::F64)
        .ok()?
        .flatten_all()
        .ok()?
        .to_vec1::<f64>()
        .ok()?;
    match values.as_slice() {
        [value] if value.is_finite() => Some(*value),
        _ => None,
    }
}

pub(crate) fn classify_lora_key(key: &str) -> Option<(LoraDirection, &str)> {
    for suffix in LORA_DOWN_SUFFIXES {
        if let Some(stem) = key.strip_suffix(suffix) {
            return Some((LoraDirection::Down, stem));
        }
    }
    for suffix in LORA_UP_SUFFIXES {
        if let Some(stem) = key.strip_suffix(suffix) {
            return Some((LoraDirection::Up, stem));
        }
    }
    None
}

impl LoraAdapter {
    /// Load a LoRA safetensors file. Tensors are loaded on CPU.
    ///
    /// Accepts the suffix matrix in [`classify_lora_key`]:
    /// - **Diffusers / PEFT-canonical**: `<layer>.lora_A.weight` /
    ///   `<layer>.lora_B.weight`. `<layer>` is a dot-separated module
    ///   path like `transformer.transformer_blocks.0.attn.to_q`.
    /// - **Kohya / sd-scripts**: `<layer>.lora_down.weight` /
    ///   `<layer>.lora_up.weight`. `lora_down` is the (rank, in)
    ///   matrix (== `lora_A`); `lora_up` is the (out, rank) matrix
    ///   (== `lora_B`).
    /// - **OneTrainer**: `<layer>.lora_linear_layer.down.weight` /
    ///   `<layer>.lora_linear_layer.up.weight`.
    /// - **PEFT default-adapter**: `<layer>.lora_A.default.weight` /
    ///   `<layer>.lora_B.default.weight`.
    /// - **Mochi-style**: `<layer>.lora_A.weight` / `<layer>.lora_B`
    ///   (no trailing `.weight`), seen in some video-LoRA trainers.
    ///
    /// `map_lora_key` recognises both diffusers and Kohya stems
    /// downstream.
    pub fn load(path: &Path) -> Result<Self> {
        let tensors = candle_core::safetensors::load(path, &Device::Cpu)?;
        let mut a_tensors: HashMap<String, Tensor> = HashMap::new();
        let mut b_tensors: HashMap<String, Tensor> = HashMap::new();
        let mut alpha_values: HashMap<String, f64> = HashMap::new();
        let mut rank = 0usize;

        for (name, tensor) in &tensors {
            if let Some((direction, stem)) = classify_lora_key(name) {
                match direction {
                    LoraDirection::Down => {
                        rank = rank.max(tensor.dim(0)?);
                        a_tensors.insert(stem.to_string(), tensor.clone());
                    }
                    LoraDirection::Up => {
                        b_tensors.insert(stem.to_string(), tensor.clone());
                    }
                }
            } else if let Some(layer) = name.strip_suffix(".alpha") {
                if let Some(val) = read_alpha_scalar(tensor) {
                    alpha_values.insert(layer.to_string(), val);
                }
            }
        }

        let mut layers = HashMap::new();
        for (layer_name, a) in a_tensors {
            if let Some(b) = b_tensors.remove(&layer_name) {
                let alpha = alpha_values.get(&layer_name).copied();
                layers.insert(layer_name, LoraLayer { a, b, alpha });
            }
        }

        if layers.is_empty() {
            bail!("no LoRA A/B pairs found in {}", path.display());
        }

        Ok(Self { layers, rank })
    }
}

/// Describes how a diffusers-format LoRA key maps to a candle model tensor.
pub(crate) enum LoraTarget {
    /// Direct 1:1 mapping: LoRA delta applies to the entire candle tensor.
    Direct { candle_key: String },
    /// Fused mapping: LoRA delta applies to a row slice of the candle tensor.
    FusedSlice {
        candle_key: String,
        /// Which component within the fused tensor (0, 1, 2, ...).
        component: usize,
        /// Total number of equally-sized components in the fused tensor.
        num_components: usize,
    },
}

/// Map a LoRA layer key (diffusers- or Kohya-format) to a candle model target.
///
/// Returns None for unrecognized keys (logged as warning, skipped).
pub(crate) fn map_lora_key(diffusers_key: &str) -> Option<LoraTarget> {
    // Kohya / sd-scripts naming (`lora_unet_*`) — keys carry the FLUX module
    // path with `.` flattened to `_`. The transformer's fused tensors
    // (`img_attn.qkv`, `txt_attn.qkv`, `single_blocks.*.linear1`) match the
    // candle layout 1:1, so every Kohya key maps to a single Direct target —
    // no FusedSlice splitting needed (Kohya already trains a B@A delta of
    // the full fused output shape).
    if let Some(rest) = diffusers_key.strip_prefix("lora_unet_") {
        return map_kohya_unet_key(rest);
    }

    // Strip the "transformer." prefix that LoRA files use
    let key = diffusers_key
        .strip_prefix("transformer.")
        .unwrap_or(diffusers_key);

    // --- Double blocks (transformer_blocks.{i}) ---
    if let Some(rest) = key.strip_prefix("transformer_blocks.") {
        let (idx_str, layer) = rest.split_once('.')?;
        let _idx: usize = idx_str.parse().ok()?;
        let block = format!("double_blocks.{idx_str}");

        return match layer {
            // Image attention QKV (fused into img_attn.qkv): Q=0, K=1, V=2
            "attn.to_q" => Some(LoraTarget::FusedSlice {
                candle_key: format!("{block}.img_attn.qkv.weight"),
                component: 0,
                num_components: 3,
            }),
            "attn.to_k" => Some(LoraTarget::FusedSlice {
                candle_key: format!("{block}.img_attn.qkv.weight"),
                component: 1,
                num_components: 3,
            }),
            "attn.to_v" => Some(LoraTarget::FusedSlice {
                candle_key: format!("{block}.img_attn.qkv.weight"),
                component: 2,
                num_components: 3,
            }),
            // Text attention QKV (fused into txt_attn.qkv): Q=0, K=1, V=2
            "attn.add_q_proj" => Some(LoraTarget::FusedSlice {
                candle_key: format!("{block}.txt_attn.qkv.weight"),
                component: 0,
                num_components: 3,
            }),
            "attn.add_k_proj" => Some(LoraTarget::FusedSlice {
                candle_key: format!("{block}.txt_attn.qkv.weight"),
                component: 1,
                num_components: 3,
            }),
            "attn.add_v_proj" => Some(LoraTarget::FusedSlice {
                candle_key: format!("{block}.txt_attn.qkv.weight"),
                component: 2,
                num_components: 3,
            }),
            // Output projections
            "attn.to_out.0" => Some(LoraTarget::Direct {
                candle_key: format!("{block}.img_attn.proj.weight"),
            }),
            "attn.to_add_out" => Some(LoraTarget::Direct {
                candle_key: format!("{block}.txt_attn.proj.weight"),
            }),
            "ff.net.0.proj" => Some(LoraTarget::Direct {
                candle_key: format!("{block}.img_mlp.0.weight"),
            }),
            "ff.net.2" => Some(LoraTarget::Direct {
                candle_key: format!("{block}.img_mlp.2.weight"),
            }),
            "ff_context.net.0.proj" => Some(LoraTarget::Direct {
                candle_key: format!("{block}.txt_mlp.0.weight"),
            }),
            "ff_context.net.2" => Some(LoraTarget::Direct {
                candle_key: format!("{block}.txt_mlp.2.weight"),
            }),
            "norm1.linear" => Some(LoraTarget::Direct {
                candle_key: format!("{block}.img_mod.lin.weight"),
            }),
            "norm1_context.linear" => Some(LoraTarget::Direct {
                candle_key: format!("{block}.txt_mod.lin.weight"),
            }),
            _ => None,
        };
    }

    // --- Single blocks (single_transformer_blocks.{i}) ---
    if let Some(rest) = key.strip_prefix("single_transformer_blocks.") {
        let (idx_str, layer) = rest.split_once('.')?;
        let _idx: usize = idx_str.parse().ok()?;
        let block = format!("single_blocks.{idx_str}");

        // single_blocks.linear1 fuses: [Q, K, V, MLP_gate, MLP_up]
        // Q/K/V each have hidden_size rows, MLP has mlp_size rows.
        // We use component indices and derive sizes from the actual tensor.
        return match layer {
            "attn.to_q" => Some(LoraTarget::FusedSlice {
                candle_key: format!("{block}.linear1.weight"),
                component: 0,
                num_components: 0, // sentinel: use special single-block logic
            }),
            "attn.to_k" => Some(LoraTarget::FusedSlice {
                candle_key: format!("{block}.linear1.weight"),
                component: 1,
                num_components: 0,
            }),
            "attn.to_v" => Some(LoraTarget::FusedSlice {
                candle_key: format!("{block}.linear1.weight"),
                component: 2,
                num_components: 0,
            }),
            "proj_mlp" => Some(LoraTarget::FusedSlice {
                candle_key: format!("{block}.linear1.weight"),
                component: 3, // MLP starts after Q,K,V
                num_components: 0,
            }),
            "proj_out" => Some(LoraTarget::Direct {
                candle_key: format!("{block}.linear2.weight"),
            }),
            "norm.linear" => Some(LoraTarget::Direct {
                candle_key: format!("{block}.modulation.lin.weight"),
            }),
            _ => None,
        };
    }

    None
}

/// Map a Kohya/sd-scripts FLUX UNet LoRA key (with the `lora_unet_` prefix
/// already stripped) to a candle model target.
///
/// Kohya's flat-naming scheme is unambiguous after `lora_unet_<block>_<idx>_`
/// — what follows is one of a small fixed set of leaves per block kind. We
/// match on that suffix directly rather than try to reverse the
/// `.`→`_` collapse, which is ambiguous (`img_attn` vs `img.attn`).
fn map_kohya_unet_key(rest: &str) -> Option<LoraTarget> {
    if let Some(after) = rest.strip_prefix("double_blocks_") {
        let (idx_str, suffix) = after.split_once('_')?;
        idx_str.parse::<usize>().ok()?;
        let candle_key = match suffix {
            "img_attn_qkv" => format!("double_blocks.{idx_str}.img_attn.qkv.weight"),
            "img_attn_proj" => format!("double_blocks.{idx_str}.img_attn.proj.weight"),
            "img_mlp_0" => format!("double_blocks.{idx_str}.img_mlp.0.weight"),
            "img_mlp_2" => format!("double_blocks.{idx_str}.img_mlp.2.weight"),
            "img_mod_lin" => format!("double_blocks.{idx_str}.img_mod.lin.weight"),
            "txt_attn_qkv" => format!("double_blocks.{idx_str}.txt_attn.qkv.weight"),
            "txt_attn_proj" => format!("double_blocks.{idx_str}.txt_attn.proj.weight"),
            "txt_mlp_0" => format!("double_blocks.{idx_str}.txt_mlp.0.weight"),
            "txt_mlp_2" => format!("double_blocks.{idx_str}.txt_mlp.2.weight"),
            "txt_mod_lin" => format!("double_blocks.{idx_str}.txt_mod.lin.weight"),
            _ => return None,
        };
        return Some(LoraTarget::Direct { candle_key });
    }
    if let Some(after) = rest.strip_prefix("single_blocks_") {
        let (idx_str, suffix) = after.split_once('_')?;
        idx_str.parse::<usize>().ok()?;
        let candle_key = match suffix {
            "linear1" => format!("single_blocks.{idx_str}.linear1.weight"),
            "linear2" => format!("single_blocks.{idx_str}.linear2.weight"),
            "modulation_lin" => format!("single_blocks.{idx_str}.modulation.lin.weight"),
            _ => return None,
        };
        return Some(LoraTarget::Direct { candle_key });
    }
    None
}

/// Compute the row offset and size for a fused slice, handling both
/// equal-split (QKV with num_components=3) and single-block linear1
/// (Q,K,V each h_sz, then MLP is the remainder).
pub(crate) fn fused_slice_range(
    base_rows: usize,
    lora_out_dim: usize,
    component: usize,
    num_components: usize,
) -> (usize, usize) {
    if let Some(component_size) = base_rows.checked_div(num_components) {
        // Equal split (e.g. QKV fused: each is base_rows / 3)
        (component * component_size, component_size)
    } else {
        // Single-block linear1: [Q, K, V, MLP]
        // For Q/K/V (components 0-2): lora_out_dim = qkv_dim (e.g. 3072)
        // For MLP (component 3): lora_out_dim = mlp_dim (e.g. 12288)
        // Total: 3*qkv_dim + mlp_dim = base_rows
        if component < 3 {
            // Q/K/V: each has lora_out_dim rows
            (component * lora_out_dim, lora_out_dim)
        } else {
            // MLP: starts after 3*qkv_dim, size = lora_out_dim (= mlp_dim)
            // Derive qkv_dim from: base_rows = 3*qkv_dim + mlp_dim
            let qkv_dim = (base_rows - lora_out_dim) / 3;
            (3 * qkv_dim, lora_out_dim)
        }
    }
}

/// Apply LoRA deltas to base model tensors in-place.
/// Currently unused (superseded by `LoraBackend` for FLUX), but retained
/// for future SD1.5/SDXL LoRA support where the UNet loading path differs.
#[allow(dead_code)]
///
/// For direct mappings: `W' = W + scale * (B @ A)`
/// For fused slices: compute delta, then add to the corresponding row slice.
///
/// When a CUDA/Metal device is provided, the matmul (`B @ A`) runs on GPU for
/// speed — LoRA tensors are small (~50-200 MB total) and GPU matmul handles all
/// layers in seconds versus minutes on CPU.  The merged result is kept on CPU
/// (same as the base tensors) so the caller can build a VarBuilder normally.
pub(crate) fn merge_lora_into_tensors(
    base_tensors: &mut HashMap<String, Tensor>,
    adapter: &LoraAdapter,
    scale: f64,
    compute_device: &Device,
    progress: &ProgressReporter,
) -> Result<()> {
    let total = adapter.layers.len();
    let mut applied = 0usize;
    let mut skipped = 0usize;
    let on_gpu = compute_device.is_cuda() || compute_device.is_metal();

    if on_gpu {
        progress.info("Merging LoRA on GPU (fast path)");
    }

    for (i, (diffusers_key, lora_layer)) in adapter.layers.iter().enumerate() {
        if (i + 1) % 100 == 0 || i + 1 == total {
            progress.info(&format!("Merging LoRA layer {}/{total}", i + 1));
        }

        let target = match map_lora_key(diffusers_key) {
            Some(t) => t,
            None => {
                tracing::warn!(key = diffusers_key, "unrecognized LoRA key, skipping");
                skipped += 1;
                continue;
            }
        };

        // Effective scale: if alpha is present, scale = user_scale * alpha / layer_rank.
        // Use per-layer rank (A's dim 0) for correct normalization with non-uniform ranks.
        let layer_rank = lora_layer.a.dim(0)? as f64;
        let effective_scale = match lora_layer.alpha {
            Some(alpha) => scale * alpha / layer_rank,
            None => scale,
        };

        // Compute delta: B @ A
        // A shape: (rank, in_features), B shape: (out_features, rank)
        // delta shape: (out_features, in_features)
        // When a GPU is available, move A/B there for the matmul then bring delta back.
        let a = lora_layer
            .a
            .to_dtype(DType::F32)?
            .to_device(compute_device)?;
        let b = lora_layer
            .b
            .to_dtype(DType::F32)?
            .to_device(compute_device)?;
        let delta = b.matmul(&a)?;
        let delta = (delta * effective_scale)?.to_device(&Device::Cpu)?;

        match target {
            LoraTarget::Direct { candle_key } => {
                let base = base_tensors
                    .get(&candle_key)
                    .ok_or_else(|| anyhow::anyhow!("base model missing tensor: {candle_key}"))?;
                let original_dtype = base.dtype();
                let base_f32 = base.to_dtype(DType::F32)?;
                let merged = (base_f32 + delta)?;
                base_tensors.insert(candle_key, merged.to_dtype(original_dtype)?);
                applied += 1;
            }
            LoraTarget::FusedSlice {
                candle_key,
                component,
                num_components,
            } => {
                let base = base_tensors
                    .get(&candle_key)
                    .ok_or_else(|| anyhow::anyhow!("base model missing tensor: {candle_key}"))?;
                let original_dtype = base.dtype();
                let base_f32 = base.to_dtype(DType::F32)?;
                let base_rows = base_f32.dim(0)?;
                let lora_out_dim = delta.dim(0)?;

                let (offset, size) =
                    fused_slice_range(base_rows, lora_out_dim, component, num_components);

                if offset + size > base_rows {
                    tracing::warn!(
                        key = diffusers_key,
                        offset,
                        size,
                        base_rows,
                        "fused slice out of bounds, skipping"
                    );
                    skipped += 1;
                    continue;
                }

                // Extract slice, add delta, reconstruct
                let slice = base_f32.narrow(0, offset, size)?;
                let updated_slice = (slice + delta)?;

                let mut parts: Vec<Tensor> = Vec::new();
                if offset > 0 {
                    parts.push(base_f32.narrow(0, 0, offset)?);
                }
                parts.push(updated_slice);
                let after = offset + size;
                if after < base_rows {
                    parts.push(base_f32.narrow(0, after, base_rows - after)?);
                }
                let merged = Tensor::cat(&parts, 0)?;
                base_tensors.insert(candle_key, merged.to_dtype(original_dtype)?);
                applied += 1;
            }
        }
    }

    progress.info(&format!(
        "LoRA merged: {applied} layers applied, {skipped} skipped (rank {})",
        adapter.rank
    ));
    tracing::info!(applied, skipped, rank = adapter.rank, "LoRA merge complete");
    Ok(())
}

/// A `SimpleBackend` that wraps mmap'd safetensors and applies LoRA deltas
/// on-the-fly when the model constructor requests each tensor.
///
/// This is the ComfyUI/InvokeAI approach adapted for candle:
/// - Tensors are loaded lazily from mmap (identical memory profile to non-LoRA)
/// - LoRA deltas are computed and applied per-tensor as `Flux::new()` calls `vb.get()`
/// - Peak VRAM = final model size only (no pre-loaded HashMap)
///
/// The A×B matmul runs on the target device (GPU if available), and the merge
/// (F32 cast + add + cast back) also happens on the target device since we're
/// processing one tensor at a time with plenty of headroom.
struct LoraBackend {
    /// The mmap'd base safetensors.
    st: candle_core::safetensors::MmapedSafetensors,
    /// Key prefix to strip (e.g. "model.diffusion_model.").
    prefix: String,
    /// Pre-computed LoRA patches keyed by canonical tensor name. With
    /// multi-LoRA the inner `Vec` may contain patches from different
    /// adapters; each `LoraPatch` carries its own `lora_path_hash` so
    /// the delta-cache key stays unique per (tensor, adapter, slice).
    patches: HashMap<String, Vec<LoraPatch>>,
    /// Optional CPU-resident cache of pre-computed deltas (shared across rebuilds).
    delta_cache: Option<Arc<Mutex<LoraDeltaCache>>>,
}

/// A single LoRA patch to apply to a base tensor. Multiple patches on the
/// same tensor stack additively: `W' = W + Σ scale_i · B_i @ A_i` — and
/// because each patch carries its own `lora_path_hash`, the delta cache
/// can disambiguate "the cinematic LoRA's contribution to img_in.weight"
/// from "the lighting LoRA's contribution to img_in.weight" without
/// recomputing either.
struct LoraPatch {
    a: Tensor,
    b: Tensor,
    effective_scale: f64,
    target: LoraTarget,
    /// Per-LoRA hash of the source file path. Was previously stored once
    /// on `LoraBackend` (single-LoRA only); promoted here so each patch
    /// keys its own delta-cache slot.
    lora_path_hash: u64,
}

/// Loaded LoRA + its scale + a stable hash of its file path. The hash is
/// used as the cache key so a second build with the same path/scale hits
/// `LoraDeltaCache` and skips the matmul.
pub(crate) struct LoraSpec<'a> {
    pub adapter: &'a LoraAdapter,
    pub scale: f64,
    pub path_hash: u64,
}

/// Walk every (adapter, layer) pair across `specs` and turn it into a
/// `LoraPatch` keyed by its target candle tensor. Multiple specs that
/// touch the same tensor accumulate into the same `Vec` so the backend
/// applies them additively. The returned counter is the number of
/// `lora_*` keys we couldn't map (logged by the caller as `skipped`).
fn build_patches(specs: &[LoraSpec<'_>]) -> (HashMap<String, Vec<LoraPatch>>, usize) {
    let mut patches: HashMap<String, Vec<LoraPatch>> = HashMap::new();
    let mut skipped = 0usize;
    for spec in specs {
        for (diffusers_key, lora_layer) in &spec.adapter.layers {
            if let Some(target) = map_lora_key(diffusers_key) {
                let candle_key = match &target {
                    LoraTarget::Direct { candle_key } => candle_key.clone(),
                    LoraTarget::FusedSlice { candle_key, .. } => candle_key.clone(),
                };
                let layer_rank = lora_layer.a.dims()[0] as f64;
                let effective_scale = match lora_layer.alpha {
                    Some(alpha) => spec.scale * alpha / layer_rank,
                    None => spec.scale,
                };
                patches.entry(candle_key).or_default().push(LoraPatch {
                    a: lora_layer.a.clone(),
                    b: lora_layer.b.clone(),
                    effective_scale,
                    target,
                    lora_path_hash: spec.path_hash,
                });
            } else {
                tracing::warn!(
                    key = diffusers_key.as_str(),
                    "unrecognized LoRA key, skipping"
                );
                skipped += 1;
            }
        }
    }
    (patches, skipped)
}

impl candle_nn::var_builder::SimpleBackend for LoraBackend {
    fn get(
        &self,
        _s: candle_core::Shape,
        name: &str,
        _h: candle_nn::Init,
        dtype: DType,
        dev: &Device,
    ) -> candle_core::Result<Tensor> {
        self.get_unchecked(name, dtype, dev)
    }

    fn get_unchecked(&self, name: &str, dtype: DType, dev: &Device) -> candle_core::Result<Tensor> {
        // Resolve the raw key in the safetensors file
        let raw_key = if self.prefix.is_empty() {
            name.to_string()
        } else {
            format!("{}{name}", self.prefix)
        };

        // Load from mmap directly to target device (same as non-LoRA path)
        let tensor = self.st.load(&raw_key, dev)?;
        let tensor = if tensor.dtype() != dtype {
            tensor.to_dtype(dtype)?
        } else {
            tensor
        };

        // Apply LoRA patches if any target this tensor
        if let Some(patches) = self.patches.get(name) {
            let mut t = tensor;
            for (patch_idx, patch) in patches.iter().enumerate() {
                // Build cache key including patch index to disambiguate fused slices
                // (e.g., Q/K/V patches on the same qkv.weight tensor) AND
                // per-LoRA path hash so a stack of two LoRAs targeting the
                // same tensor doesn't collapse into one cache slot.
                let cache_key = LoraCacheKey {
                    tensor_name: name.to_string(),
                    patch_index: patch_idx,
                    lora_path_hash: patch.lora_path_hash,
                    scale_bits: patch.effective_scale.to_bits(),
                };

                // Try to retrieve from cache (CPU-resident delta)
                let cached_delta = self.delta_cache.as_ref().and_then(|c| {
                    c.lock()
                        .ok()
                        .and_then(|guard| guard.get(&cache_key).cloned())
                });

                let delta = if let Some(cpu_delta) = cached_delta {
                    // Cache hit: move to target device
                    cpu_delta.to_device(dev)?
                } else {
                    // Cache miss: compute delta on target device
                    let a = patch.a.to_dtype(DType::F32)?.to_device(dev)?;
                    let b = patch.b.to_dtype(DType::F32)?.to_device(dev)?;
                    let computed = b.matmul(&a)?;
                    let computed = (&computed * patch.effective_scale)?;

                    // Store on CPU for future rebuilds
                    if let Some(ref cache) = self.delta_cache {
                        if let Ok(mut guard) = cache.lock() {
                            let cpu_copy = computed.to_device(&Device::Cpu)?;
                            guard.insert(cache_key, cpu_copy);
                        }
                    }
                    computed
                };

                t = match &patch.target {
                    LoraTarget::Direct { .. } => {
                        let t_f32 = t.to_dtype(DType::F32)?;
                        let merged = (&t_f32 + &delta)?;
                        merged.to_dtype(dtype)?
                    }
                    LoraTarget::FusedSlice {
                        component,
                        num_components,
                        ..
                    } => {
                        let t_f32 = t.to_dtype(DType::F32)?;
                        let base_rows = t_f32.dim(0)?;
                        let lora_out_dim = delta.dim(0)?;
                        let (offset, size) =
                            fused_slice_range(base_rows, lora_out_dim, *component, *num_components);

                        if offset + size > base_rows {
                            tracing::warn!(
                                offset,
                                size,
                                base_rows,
                                "fused slice out of bounds, skipping"
                            );
                            t
                        } else {
                            let slice = t_f32.narrow(0, offset, size)?;
                            let updated_slice = (&slice + &delta)?;
                            let mut parts: Vec<Tensor> = Vec::new();
                            if offset > 0 {
                                parts.push(t_f32.narrow(0, 0, offset)?);
                            }
                            parts.push(updated_slice);
                            let after = offset + size;
                            if after < base_rows {
                                parts.push(t_f32.narrow(0, after, base_rows - after)?);
                            }
                            Tensor::cat(&parts, 0)?.to_dtype(dtype)?
                        }
                    }
                };
            }
            Ok(t)
        } else {
            Ok(tensor)
        }
    }

    fn contains_tensor(&self, name: &str) -> bool {
        let raw_key = if self.prefix.is_empty() {
            name.to_string()
        } else {
            format!("{}{name}", self.prefix)
        };
        // Check via trying to load metadata (tensors() lists all names)
        self.st.get(&raw_key).is_ok()
    }
}

/// Build a LoRA-patching VarBuilder that wraps mmap'd safetensors.
///
/// This uses candle's `SimpleBackend` trait to intercept every `vb.get()` call
/// during model construction.  Each tensor is loaded from mmap directly to the
/// target device (GPU), with LoRA deltas applied inline.  Memory profile is
/// identical to the non-LoRA mmap path — no HashMap, no pre-loading.
///
/// Multi-LoRA: pass multiple `LoraSpec`s and they merge additively. The
/// per-patch cache key (tensor, patch_idx, lora_path_hash, scale_bits)
/// keeps each adapter's delta independently cacheable.
pub(crate) fn lora_var_builder<'a>(
    transformer_path: &Path,
    specs: &[LoraSpec<'_>],
    dtype: DType,
    device: &Device,
    progress: &ProgressReporter,
    delta_cache: Option<Arc<Mutex<LoraDeltaCache>>>,
) -> Result<candle_nn::VarBuilder<'a>> {
    use candle_core::safetensors::MmapedSafetensors;

    if specs.is_empty() {
        bail!("lora_var_builder called with no LoraSpecs — caller must provide at least one");
    }

    // Open mmap (cheap, no I/O)
    let st = unsafe { MmapedSafetensors::multi(std::slice::from_ref(&transformer_path))? };

    // Detect key prefix
    let all_names: Vec<String> = st.tensors().into_iter().map(|(n, _)| n).collect();
    let prefix = if all_names.iter().any(|n| n == "img_in.weight") {
        ""
    } else if all_names
        .iter()
        .any(|n| n == "model.diffusion_model.img_in.weight")
    {
        "model.diffusion_model."
    } else if all_names
        .iter()
        .any(|n| n == "diffusion_model.img_in.weight")
    {
        "diffusion_model."
    } else {
        ""
    };

    // Build patch index: for each candle key, collect all LoRA patches
    // (across every adapter in `specs`).
    let (patches, skipped) = build_patches(specs);

    let patched_keys = patches.len();
    let total_patches: usize = patches.values().map(|v| v.len()).sum();
    let max_rank = specs.iter().map(|s| s.adapter.rank).max().unwrap_or(0);
    progress.info(&format!(
        "LoRA: {n} adapter(s), {total_patches} patches on {patched_keys} tensors, {skipped} skipped (max rank {max_rank})",
        n = specs.len(),
    ));

    let backend = LoraBackend {
        st,
        prefix: prefix.to_string(),
        patches,
        delta_cache,
    };

    Ok(candle_nn::VarBuilder::from_backend(
        Box::new(backend),
        dtype,
        device.clone(),
    ))
}

/// Build a quantized VarBuilder from a GGUF file with LoRA deltas applied.
///
/// Loads the GGUF file into a `HashMap<String, Arc<QTensor>>`, then for each
/// LoRA-targeted tensor: dequantizes to F32 on CPU, applies the LoRA delta,
/// and re-quantizes back to the original GGML dtype (e.g. Q8_0) on the target
/// device.  Non-LoRA tensors stay quantized and untouched.
///
/// By re-quantizing to the original dtype instead of storing as F16/BF16, each
/// patched tensor occupies the same VRAM as its original — no inflation.  The
/// LoRA rank is small (typically 32) so the re-quantization error is negligible.
pub(crate) fn gguf_lora_var_builder(
    transformer_path: &Path,
    specs: &[LoraSpec<'_>],
    device: &Device,
    progress: &ProgressReporter,
    delta_cache: Option<Arc<Mutex<LoraDeltaCache>>>,
) -> Result<mold_candle::quantized::VarBuilder> {
    use candle_core::quantized::{gguf_file, QTensor};
    use std::sync::Arc;

    if specs.is_empty() {
        bail!("gguf_lora_var_builder called with no LoraSpecs — caller must provide at least one");
    }

    // Load GGUF tensors
    let mut file = std::fs::File::open(transformer_path)?;
    let content = gguf_file::Content::read(&mut file)?;

    let total_tensors = content.tensor_infos.len();
    let mut data: HashMap<String, Arc<QTensor>> = HashMap::with_capacity(total_tensors);

    // Build patch index (same as safetensors LoRA path) — accumulate
    // patches from every adapter into the same map. The downstream
    // dequant→merge→requant loop already iterates per-tensor patches in
    // sequence, so multi-LoRA stacking is just additional entries.
    let (patches, skipped) = build_patches(specs);

    let patched_keys = patches.len();
    let total_patches: usize = patches.values().map(|v| v.len()).sum();
    let max_rank = specs.iter().map(|s| s.adapter.rank).max().unwrap_or(0);
    progress.info(&format!(
        "LoRA: {n} adapter(s), {total_patches} patches on {patched_keys} tensors, {skipped} skipped (max rank {max_rank})",
        n = specs.len(),
    ));

    // Phase 1: Load ALL tensors via normal GGUF path (same as from_gguf).
    // This uses the exact same CUDA allocation as the non-LoRA path.
    let gguf_bytes_total: u64 = std::fs::metadata(transformer_path)
        .map(|m| m.len())
        .unwrap_or(0);
    progress.weight_load("FLUX transformer (GGUF)", 0, gguf_bytes_total);
    for (i, tensor_name) in content.tensor_infos.keys().enumerate() {
        let qtensor = content.tensor(&mut file, tensor_name, device)?;
        data.insert(tensor_name.clone(), Arc::new(qtensor));
        // Approximate progress based on tensor count (GGUF has no per-tensor byte info)
        let approx_bytes = gguf_bytes_total * (i as u64 + 1) / total_tensors as u64;
        progress.weight_load(
            "FLUX transformer (GGUF)",
            approx_bytes.min(gguf_bytes_total),
            gguf_bytes_total,
        );
    }
    drop(file); // close GGUF file

    // Phase 2: Patch LoRA-affected tensors in-place.
    // For each target: dequantize the GPU QTensor to F32 on CPU, apply LoRA
    // delta, re-quantize back to the original GGML dtype (e.g. Q8_0), and
    // place the result on GPU.  This keeps each patched tensor at its original
    // quantized size — no VRAM inflation.
    let on_gpu = device.is_cuda() || device.is_metal();
    let mut applied = 0usize;
    let lora_keys: Vec<String> = patches.keys().cloned().collect();
    let lora_total = lora_keys.len();
    for (i, candle_key) in lora_keys.iter().enumerate() {
        let layer_patches = &patches[candle_key];

        // Find the matching tensor key (try with .weight suffix)
        let tensor_key = if data.contains_key(candle_key) {
            candle_key.clone()
        } else {
            // Shouldn't happen if map_lora_key produced correct candle keys
            tracing::warn!(
                key = candle_key.as_str(),
                "LoRA target tensor not found in GGUF, skipping"
            );
            continue;
        };

        // Remember the original quantized dtype so we can re-quantize to it.
        let orig_dtype = data[&tensor_key].dtype();

        // Dequantize to F32 on CPU — keeps GPU clean for other tensors.
        // The original Q8 GPU entry is removed to reclaim its VRAM.
        let qtensor = data.remove(&tensor_key).unwrap();
        let mut t = qtensor.dequantize(&Device::Cpu)?;
        drop(qtensor); // release GPU QTensor VRAM
        if on_gpu {
            device.synchronize()?; // ensure CUDA frees the Q8 allocation
        }

        for (patch_idx, patch) in layer_patches.iter().enumerate() {
            // Build cache key including patch index (disambiguates fused
            // slices on the same tensor) AND per-patch lora_path_hash
            // (disambiguates which adapter contributed this delta in a
            // multi-LoRA stack).
            let cache_key = LoraCacheKey {
                tensor_name: candle_key.clone(),
                patch_index: patch_idx,
                lora_path_hash: patch.lora_path_hash,
                scale_bits: patch.effective_scale.to_bits(),
            };

            // Try cache first, then compute
            let cached = delta_cache.as_ref().and_then(|c| {
                c.lock()
                    .ok()
                    .and_then(|guard| guard.get(&cache_key).cloned())
            });

            let delta = if let Some(cpu_delta) = cached {
                cpu_delta
            } else {
                let matmul_dev = if on_gpu { device } else { &Device::Cpu };
                let a = patch.a.to_dtype(DType::F32)?.to_device(matmul_dev)?;
                let b = patch.b.to_dtype(DType::F32)?.to_device(matmul_dev)?;
                let computed = b.matmul(&a)?;
                let computed = (&computed * patch.effective_scale)?.to_device(&Device::Cpu)?;

                // Store in cache for future rebuilds
                if let Some(ref cache) = delta_cache {
                    if let Ok(mut guard) = cache.lock() {
                        guard.insert(cache_key, computed.clone());
                    }
                }
                computed
            };

            t = match &patch.target {
                LoraTarget::Direct { .. } => (&t + &delta)?,
                LoraTarget::FusedSlice {
                    component,
                    num_components,
                    ..
                } => {
                    let base_rows = t.dim(0)?;
                    let lora_out_dim = delta.dim(0)?;
                    let (offset, size) =
                        fused_slice_range(base_rows, lora_out_dim, *component, *num_components);

                    if offset + size > base_rows {
                        tracing::warn!(
                            offset,
                            size,
                            base_rows,
                            "fused slice out of bounds, skipping"
                        );
                        t
                    } else {
                        let slice = t.narrow(0, offset, size)?;
                        let updated_slice = (&slice + &delta)?;
                        let mut parts: Vec<Tensor> = Vec::new();
                        if offset > 0 {
                            parts.push(t.narrow(0, 0, offset)?);
                        }
                        parts.push(updated_slice);
                        let after = offset + size;
                        if after < base_rows {
                            parts.push(t.narrow(0, after, base_rows - after)?);
                        }
                        Tensor::cat(&parts, 0)?
                    }
                }
            };
            applied += 1;
        }

        // Re-quantize back to the original GGML dtype (e.g. Q8_0) and place
        // on the target device.  `quantize_onto` quantizes the CPU F32 tensor
        // into CPU Q8_0 blocks, then copies the raw bytes to GPU — producing
        // the exact same storage size as the original GGUF-loaded tensor.
        // This avoids the 2x VRAM inflation that storing as F16 would cause.
        let patched = mold_candle::quantized::quantize_onto(&t, orig_dtype, device)?;
        drop(t); // free CPU F32 copy
        data.insert(tensor_key, Arc::new(patched));

        if (i + 1) % 50 == 0 || i + 1 == lora_total {
            progress.info(&format!(
                "Patching LoRA tensor {}/{}",
                i + 1,
                lora_keys.len()
            ));
        }
    }

    let total_layers: usize = specs.iter().map(|s| s.adapter.layers.len()).sum();
    progress.info(&format!(
        "LoRA: {applied} applied, {} skipped (max rank {max_rank}, {patched_keys} layers patched)",
        total_layers.saturating_sub(applied),
    ));

    // Drain pending cuMemFreeAsync from the per-tensor merge loop. Each
    // patched tensor allocates F32 A/B/B@A intermediates on GPU sized like the
    // full weight (~150 MB for a 3072×12288 MLP); their drops queue async
    // frees that don't actually return VRAM to the device until a sync. With
    // 50+ LoRA-affected tensors per adapter and a stack of 2 LoRAs, the queued
    // frees pile up to several GB. Without this sync, denoising starts with a
    // bloated working set and VAE decode at 1024² OOMs even though the kept
    // transformer is the same size as the no-LoRA case.
    if on_gpu {
        device.synchronize()?;
    }

    Ok(mold_candle::quantized::VarBuilder::from_qtensors(
        data, device,
    ))
}

/// Strip a known prefix from all tensor keys in a HashMap.
///
/// FLUX safetensors may store weights under `model.diffusion_model.` or
/// `diffusion_model.` — this normalizes them to root level.
#[allow(dead_code)]
pub(crate) fn strip_tensor_prefix(tensors: HashMap<String, Tensor>) -> HashMap<String, Tensor> {
    let prefix = if tensors.contains_key("img_in.weight") {
        ""
    } else if tensors.contains_key("model.diffusion_model.img_in.weight") {
        "model.diffusion_model."
    } else if tensors.contains_key("diffusion_model.img_in.weight") {
        "diffusion_model."
    } else {
        ""
    };

    if prefix.is_empty() {
        return tensors;
    }

    tensors
        .into_iter()
        .map(|(k, v)| {
            let stripped = k.strip_prefix(prefix).unwrap_or(&k).to_string();
            (stripped, v)
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn map_double_block_img_attn_qkv() {
        let key = "transformer.transformer_blocks.5.attn.to_q";
        let target = map_lora_key(key).unwrap();
        match target {
            LoraTarget::FusedSlice {
                candle_key,
                component,
                num_components,
            } => {
                assert_eq!(candle_key, "double_blocks.5.img_attn.qkv.weight");
                assert_eq!(component, 0);
                assert_eq!(num_components, 3);
            }
            _ => panic!("expected FusedSlice"),
        }

        let key = "transformer.transformer_blocks.5.attn.to_k";
        match map_lora_key(key).unwrap() {
            LoraTarget::FusedSlice { component, .. } => assert_eq!(component, 1),
            _ => panic!("expected FusedSlice"),
        }

        let key = "transformer.transformer_blocks.5.attn.to_v";
        match map_lora_key(key).unwrap() {
            LoraTarget::FusedSlice { component, .. } => assert_eq!(component, 2),
            _ => panic!("expected FusedSlice"),
        }
    }

    #[test]
    fn map_double_block_txt_attn_qkv() {
        let key = "transformer.transformer_blocks.0.attn.add_q_proj";
        match map_lora_key(key).unwrap() {
            LoraTarget::FusedSlice {
                candle_key,
                component,
                num_components,
            } => {
                assert_eq!(candle_key, "double_blocks.0.txt_attn.qkv.weight");
                assert_eq!(component, 0);
                assert_eq!(num_components, 3);
            }
            _ => panic!("expected FusedSlice"),
        }
    }

    #[test]
    fn map_double_block_direct() {
        let cases = [
            (
                "transformer.transformer_blocks.3.attn.to_out.0",
                "double_blocks.3.img_attn.proj.weight",
            ),
            (
                "transformer.transformer_blocks.3.attn.to_add_out",
                "double_blocks.3.txt_attn.proj.weight",
            ),
            (
                "transformer.transformer_blocks.3.ff.net.0.proj",
                "double_blocks.3.img_mlp.0.weight",
            ),
            (
                "transformer.transformer_blocks.3.ff.net.2",
                "double_blocks.3.img_mlp.2.weight",
            ),
            (
                "transformer.transformer_blocks.3.ff_context.net.0.proj",
                "double_blocks.3.txt_mlp.0.weight",
            ),
            (
                "transformer.transformer_blocks.3.ff_context.net.2",
                "double_blocks.3.txt_mlp.2.weight",
            ),
            (
                "transformer.transformer_blocks.3.norm1.linear",
                "double_blocks.3.img_mod.lin.weight",
            ),
            (
                "transformer.transformer_blocks.3.norm1_context.linear",
                "double_blocks.3.txt_mod.lin.weight",
            ),
        ];
        for (lora_key, expected) in cases {
            match map_lora_key(lora_key).unwrap() {
                LoraTarget::Direct { candle_key } => assert_eq!(candle_key, expected),
                _ => panic!("expected Direct for {lora_key}"),
            }
        }
    }

    #[test]
    fn map_single_block_fused() {
        let key = "transformer.single_transformer_blocks.7.attn.to_q";
        match map_lora_key(key).unwrap() {
            LoraTarget::FusedSlice {
                candle_key,
                component,
                ..
            } => {
                assert_eq!(candle_key, "single_blocks.7.linear1.weight");
                assert_eq!(component, 0);
            }
            _ => panic!("expected FusedSlice"),
        }

        let key = "transformer.single_transformer_blocks.7.proj_mlp";
        match map_lora_key(key).unwrap() {
            LoraTarget::FusedSlice { component, .. } => assert_eq!(component, 3),
            _ => panic!("expected FusedSlice"),
        }
    }

    #[test]
    fn map_single_block_direct() {
        let key = "transformer.single_transformer_blocks.7.proj_out";
        match map_lora_key(key).unwrap() {
            LoraTarget::Direct { candle_key } => {
                assert_eq!(candle_key, "single_blocks.7.linear2.weight")
            }
            _ => panic!("expected Direct"),
        }

        let key = "transformer.single_transformer_blocks.7.norm.linear";
        match map_lora_key(key).unwrap() {
            LoraTarget::Direct { candle_key } => {
                assert_eq!(candle_key, "single_blocks.7.modulation.lin.weight")
            }
            _ => panic!("expected Direct"),
        }
    }

    #[test]
    fn map_unknown_key_returns_none() {
        assert!(map_lora_key("totally.unknown.key").is_none());
        assert!(map_lora_key("transformer.transformer_blocks.0.unknown_layer").is_none());
    }

    #[test]
    fn fused_slice_range_equal_split() {
        // QKV fused: 9216 rows / 3 = 3072 each
        let (offset, size) = fused_slice_range(9216, 3072, 0, 3);
        assert_eq!((offset, size), (0, 3072));

        let (offset, size) = fused_slice_range(9216, 3072, 1, 3);
        assert_eq!((offset, size), (3072, 3072));

        let (offset, size) = fused_slice_range(9216, 3072, 2, 3);
        assert_eq!((offset, size), (6144, 3072));
    }

    #[test]
    fn fused_slice_range_single_block() {
        // linear1 fuses Q(3072), K(3072), V(3072), MLP(12288) = 21504 total
        // Q component:
        let (offset, size) = fused_slice_range(21504, 3072, 0, 0);
        assert_eq!((offset, size), (0, 3072));

        // K component:
        let (offset, size) = fused_slice_range(21504, 3072, 1, 0);
        assert_eq!((offset, size), (3072, 3072));

        // V component:
        let (offset, size) = fused_slice_range(21504, 3072, 2, 0);
        assert_eq!((offset, size), (6144, 3072));

        // MLP component (lora_out_dim = 12288):
        let (offset, size) = fused_slice_range(21504, 12288, 3, 0);
        assert_eq!((offset, size), (9216, 12288));
    }

    /// Build a synthetic adapter that touches one direct-mapped tensor
    /// (`transformer_blocks.0.attn.to_out.0`). Shape is (out=4, rank=2)
    /// for B and (rank=2, in=4) for A so the math is small but real.
    fn synthetic_single_layer_adapter(scale_a: f32, scale_b: f32) -> LoraAdapter {
        let device = candle_core::Device::Cpu;
        let a = Tensor::full(scale_a, (2, 4), &device).unwrap();
        let b = Tensor::full(scale_b, (4, 2), &device).unwrap();
        let mut layers = HashMap::new();
        layers.insert(
            "transformer_blocks.0.attn.to_out.0".to_string(),
            LoraLayer { a, b, alpha: None },
        );
        LoraAdapter { layers, rank: 2 }
    }

    /// Two adapters targeting the same tensor must produce two patches
    /// in the same Vec, each carrying its own `lora_path_hash` so the
    /// per-patch delta cache stays disambiguated.
    #[test]
    fn build_patches_stacks_multiple_specs_on_same_tensor() {
        let a1 = synthetic_single_layer_adapter(1.0, 1.0);
        let a2 = synthetic_single_layer_adapter(2.0, 3.0);
        let specs = [
            LoraSpec {
                adapter: &a1,
                scale: 0.5,
                path_hash: 0xAA,
            },
            LoraSpec {
                adapter: &a2,
                scale: 0.25,
                path_hash: 0xBB,
            },
        ];

        let (patches, skipped) = build_patches(&specs);
        assert_eq!(skipped, 0, "every test layer maps to a known target");
        let key = "double_blocks.0.img_attn.proj.weight";
        let stack = patches.get(key).expect("target tensor must be patched");
        assert_eq!(
            stack.len(),
            2,
            "both adapters must contribute a patch to the same tensor"
        );
        // Order is the order of the specs; lora_path_hash discriminates
        // them so the delta cache can never confuse one for the other.
        assert_eq!(stack[0].lora_path_hash, 0xAA);
        assert_eq!(stack[1].lora_path_hash, 0xBB);
        assert!(
            (stack[0].effective_scale - 0.5).abs() < 1e-9,
            "first patch keeps its caller-supplied scale (no alpha override)"
        );
        assert!(
            (stack[1].effective_scale - 0.25).abs() < 1e-9,
            "second patch keeps its caller-supplied scale"
        );
    }

    #[test]
    fn map_kohya_double_block_keys() {
        let cases = [
            (
                "lora_unet_double_blocks_0_img_attn_qkv",
                "double_blocks.0.img_attn.qkv.weight",
            ),
            (
                "lora_unet_double_blocks_5_img_attn_proj",
                "double_blocks.5.img_attn.proj.weight",
            ),
            (
                "lora_unet_double_blocks_10_img_mlp_0",
                "double_blocks.10.img_mlp.0.weight",
            ),
            (
                "lora_unet_double_blocks_10_img_mlp_2",
                "double_blocks.10.img_mlp.2.weight",
            ),
            (
                "lora_unet_double_blocks_3_img_mod_lin",
                "double_blocks.3.img_mod.lin.weight",
            ),
            (
                "lora_unet_double_blocks_7_txt_attn_qkv",
                "double_blocks.7.txt_attn.qkv.weight",
            ),
            (
                "lora_unet_double_blocks_7_txt_attn_proj",
                "double_blocks.7.txt_attn.proj.weight",
            ),
            (
                "lora_unet_double_blocks_2_txt_mlp_0",
                "double_blocks.2.txt_mlp.0.weight",
            ),
            (
                "lora_unet_double_blocks_2_txt_mlp_2",
                "double_blocks.2.txt_mlp.2.weight",
            ),
            (
                "lora_unet_double_blocks_18_txt_mod_lin",
                "double_blocks.18.txt_mod.lin.weight",
            ),
        ];
        for (kohya_key, expected) in cases {
            match map_lora_key(kohya_key).unwrap() {
                LoraTarget::Direct { candle_key } => {
                    assert_eq!(candle_key, expected, "kohya key {kohya_key}");
                }
                _ => panic!("expected Direct for kohya key {kohya_key}"),
            }
        }
    }

    #[test]
    fn map_kohya_single_block_keys() {
        let cases = [
            (
                "lora_unet_single_blocks_0_linear1",
                "single_blocks.0.linear1.weight",
            ),
            (
                "lora_unet_single_blocks_9_linear2",
                "single_blocks.9.linear2.weight",
            ),
            (
                "lora_unet_single_blocks_37_modulation_lin",
                "single_blocks.37.modulation.lin.weight",
            ),
        ];
        for (kohya_key, expected) in cases {
            match map_lora_key(kohya_key).unwrap() {
                LoraTarget::Direct { candle_key } => {
                    assert_eq!(candle_key, expected, "kohya key {kohya_key}");
                }
                _ => panic!("expected Direct for kohya key {kohya_key}"),
            }
        }
    }

    #[test]
    fn map_kohya_unknown_leaves_returns_none() {
        // Text-encoder LoRAs (`lora_te_*`) and unrecognized leaves are
        // skipped (caller logs a warning) rather than panicking.
        assert!(map_lora_key("lora_te_text_model_layer_0_attn_q").is_none());
        assert!(map_lora_key("lora_unet_double_blocks_0_unknown_leaf").is_none());
        assert!(map_lora_key("lora_unet_single_blocks_0_norm_query").is_none());
        assert!(map_lora_key("lora_unet_unrelated_block_0_x").is_none());
    }

    /// Round-trip `LoraAdapter::load` against a Kohya-shaped safetensors
    /// fixture to prove the suffix matcher accepts `lora_down`/`lora_up`/
    /// `alpha` and pairs them up correctly. Synthetic shapes — the
    /// numbers don't have to match a real FLUX layer, just the down/up
    /// convention.
    #[test]
    fn load_accepts_kohya_lora_down_up_alpha() {
        use safetensors::tensor::TensorView;
        let dir = tempfile::tempdir().expect("tempdir");
        let path = dir.path().join("kohya.safetensors");
        let layer = "lora_unet_double_blocks_0_img_attn_qkv";

        // (rank=2, in=4) for down, (out=6, rank=2) for up. f32 little-endian
        // raw bytes — `safetensors::serialize` round-trips them as F32
        // tensors that `candle_core::safetensors::load` then parses.
        let down: Vec<f32> = (0..2 * 4).map(|i| i as f32 * 0.1).collect();
        let up: Vec<f32> = (0..6 * 2).map(|i| i as f32 * 0.2).collect();
        let alpha: Vec<f32> = vec![16.0];

        let down_bytes: Vec<u8> = down.iter().flat_map(|f| f.to_le_bytes()).collect();
        let up_bytes: Vec<u8> = up.iter().flat_map(|f| f.to_le_bytes()).collect();
        let alpha_bytes: Vec<u8> = alpha.iter().flat_map(|f| f.to_le_bytes()).collect();

        let down_view = TensorView::new(safetensors::Dtype::F32, vec![2, 4], &down_bytes).unwrap();
        let up_view = TensorView::new(safetensors::Dtype::F32, vec![6, 2], &up_bytes).unwrap();
        let alpha_view = TensorView::new(safetensors::Dtype::F32, vec![], &alpha_bytes).unwrap();

        let entries: Vec<(String, TensorView)> = vec![
            (format!("{layer}.lora_down.weight"), down_view),
            (format!("{layer}.lora_up.weight"), up_view),
            (format!("{layer}.alpha"), alpha_view),
        ];
        safetensors::serialize_to_file(entries, &None, &path).expect("write safetensors");

        let adapter = LoraAdapter::load(&path).expect("kohya safetensors must load");
        assert_eq!(
            adapter.layers.len(),
            1,
            "lora_down/lora_up should be paired into one layer"
        );
        assert_eq!(adapter.rank, 2);
        let lora_layer = adapter.layers.get(layer).expect("layer present");
        assert_eq!(lora_layer.a.dims(), &[2, 4]);
        assert_eq!(lora_layer.b.dims(), &[6, 2]);
        assert_eq!(lora_layer.alpha, Some(16.0));
    }

    /// Regression: an `I64` alpha must be read, not silently dropped.
    ///
    /// The Wan 2.2 Lightning distills ship `.alpha` as `I64` (value 8 against
    /// rank 64). `to_scalar::<f32>()` errors on `I64` storage, and the old
    /// `if let Ok` swallowed it — the layer got `alpha: None`, the scale fell
    /// back to the caller's raw value, and the adapter applied 8x too strong
    /// with no diagnostic.
    #[test]
    fn load_reads_an_i64_alpha_rather_than_dropping_it() {
        use safetensors::tensor::TensorView;
        let dir = tempfile::tempdir().expect("tempdir");
        let path = dir.path().join("i64_alpha.safetensors");
        let layer = "lora_unet_double_blocks_0_img_attn_qkv";

        let down_bytes: Vec<u8> = (0..2 * 4).flat_map(|i| (i as f32).to_le_bytes()).collect();
        let up_bytes: Vec<u8> = (0..6 * 2).flat_map(|i| (i as f32).to_le_bytes()).collect();
        // The shipped spelling: a rank-0 I64 scalar.
        let alpha_bytes = 8i64.to_le_bytes().to_vec();

        let entries: Vec<(String, TensorView)> = vec![
            (
                format!("{layer}.lora_down.weight"),
                TensorView::new(safetensors::Dtype::F32, vec![2, 4], &down_bytes).unwrap(),
            ),
            (
                format!("{layer}.lora_up.weight"),
                TensorView::new(safetensors::Dtype::F32, vec![6, 2], &up_bytes).unwrap(),
            ),
            (
                format!("{layer}.alpha"),
                TensorView::new(safetensors::Dtype::I64, vec![], &alpha_bytes).unwrap(),
            ),
        ];
        safetensors::serialize_to_file(entries, &None, &path).expect("write safetensors");

        let adapter = LoraAdapter::load(&path).expect("I64-alpha safetensors must load");
        let lora_layer = adapter.layers.get(layer).expect("layer present");
        assert_eq!(
            lora_layer.alpha,
            Some(8.0),
            "an I64 alpha must survive the read"
        );
        // Rank 2 with alpha 8 means an effective scale of 4x the user's, not 1x.
        let specs = [LoraSpec {
            adapter: &adapter,
            scale: 1.0,
            path_hash: 0,
        }];
        let (patches, _) = build_patches(&specs);
        let patch = patches.values().next().expect("one patch").first().unwrap();
        assert!((patch.effective_scale - 4.0).abs() < 1e-9);
    }

    /// `read_alpha_scalar` must accept every spelling trainers emit and reject
    /// anything that is not one scalar.
    #[test]
    fn alpha_scalar_reads_are_dtype_and_shape_tolerant() {
        let device = Device::Cpu;
        for (label, tensor) in [
            ("f32 rank-0", Tensor::new(16f32, &device).unwrap()),
            ("f64 rank-0", Tensor::new(16f64, &device).unwrap()),
            ("i64 rank-0", Tensor::new(16i64, &device).unwrap()),
            ("u32 rank-0", Tensor::new(16u32, &device).unwrap()),
            ("f32 shape [1]", Tensor::new(&[16f32], &device).unwrap()),
            ("i64 shape [1]", Tensor::new(&[16i64], &device).unwrap()),
        ] {
            assert_eq!(read_alpha_scalar(&tensor), Some(16.0), "{label}");
        }
        // Not a scalar: refuse rather than silently taking the first element.
        let vector = Tensor::new(&[1f32, 2.0], &device).unwrap();
        assert_eq!(read_alpha_scalar(&vector), None);
    }

    #[test]
    fn build_patches_single_spec_matches_legacy_shape() {
        // Regression: a one-element specs slice must produce the same
        // patch shape as the original single-LoRA path, so existing
        // single-LoRA flows don't change behaviour.
        let a = synthetic_single_layer_adapter(1.0, 1.0);
        let specs = [LoraSpec {
            adapter: &a,
            scale: 0.75,
            path_hash: 0xCC,
        }];
        let (patches, skipped) = build_patches(&specs);
        assert_eq!(skipped, 0);
        let stack = patches
            .get("double_blocks.0.img_attn.proj.weight")
            .expect("present");
        assert_eq!(stack.len(), 1);
        assert_eq!(stack[0].lora_path_hash, 0xCC);
    }

    // ── classify_lora_key — multi-format suffix matrix ──────────────────

    #[test]
    fn classify_lora_key_diffusers() {
        assert_eq!(
            classify_lora_key("x.lora_A.weight"),
            Some((LoraDirection::Down, "x"))
        );
        assert_eq!(
            classify_lora_key("x.lora_B.weight"),
            Some((LoraDirection::Up, "x"))
        );
    }

    #[test]
    fn classify_lora_key_kohya() {
        assert_eq!(
            classify_lora_key("x.lora_down.weight"),
            Some((LoraDirection::Down, "x"))
        );
        assert_eq!(
            classify_lora_key("x.lora_up.weight"),
            Some((LoraDirection::Up, "x"))
        );
    }

    /// OneTrainer's flat-naming scheme inverts the down/up positions
    /// in the dotted key compared to Kohya: the *down* matrix is at
    /// `lora_linear_layer.down.weight`, NOT `lora_linear_layer.up.weight`.
    /// Pin the mapping so a future refactor doesn't silently swap them.
    #[test]
    fn classify_lora_key_onetrainer_inverted_naming() {
        assert_eq!(
            classify_lora_key("x.lora_linear_layer.down.weight"),
            Some((LoraDirection::Down, "x"))
        );
        assert_eq!(
            classify_lora_key("x.lora_linear_layer.up.weight"),
            Some((LoraDirection::Up, "x"))
        );
    }

    #[test]
    fn classify_lora_key_default_adapter_peft() {
        assert_eq!(
            classify_lora_key("x.lora_A.default.weight"),
            Some((LoraDirection::Down, "x"))
        );
        assert_eq!(
            classify_lora_key("x.lora_B.default.weight"),
            Some((LoraDirection::Up, "x"))
        );
    }

    /// Mochi-style trainer drops the trailing `.weight` segment from
    /// the up matrix only. Down stays as `.lora_A.weight`. The
    /// suffix-list ordering is load-bearing: `.lora_B` comes last so
    /// it doesn't shadow `.lora_B.weight` / `.lora_B.default.weight`.
    #[test]
    fn classify_lora_key_mochi_no_dot_weight() {
        assert_eq!(
            classify_lora_key("x.lora_B"),
            Some((LoraDirection::Up, "x"))
        );
        // Sanity: the down side keeps the canonical form.
        assert_eq!(
            classify_lora_key("x.lora_A.weight"),
            Some((LoraDirection::Down, "x"))
        );
    }

    #[test]
    fn classify_lora_key_unrelated_returns_none() {
        assert_eq!(classify_lora_key("x.weight"), None);
        assert_eq!(classify_lora_key("transformer.embed.weight"), None);
        assert_eq!(classify_lora_key("alpha"), None);
        // `.alpha` scalars are also not LoRA pair tensors — the
        // loader handles them on its own branch.
        assert_eq!(classify_lora_key("layer.alpha"), None);
    }

    /// Stems must come back unmodified — the loader pairs A/B by
    /// stem equality, so any silent normalisation here would break
    /// adapter loading on long module paths.
    #[test]
    fn classify_lora_key_preserves_dotted_stem() {
        assert_eq!(
            classify_lora_key("transformer.transformer_blocks.5.attn.to_q.lora_A.weight"),
            Some((
                LoraDirection::Down,
                "transformer.transformer_blocks.5.attn.to_q",
            ))
        );
    }

    // ── parsed_lora_cache — hit + mtime invalidate ──────────────────────

    /// Build the same fixture as `load_accepts_kohya_lora_down_up_alpha`
    /// but reusable so the cache tests can also drop a real safetensors
    /// file at a path of their choice. Returns a path the caller owns.
    fn write_synthetic_kohya_safetensors(path: &Path) {
        use safetensors::tensor::TensorView;
        let layer = "lora_unet_double_blocks_0_img_attn_qkv";

        let down: Vec<f32> = (0..2 * 4).map(|i| i as f32 * 0.1).collect();
        let up: Vec<f32> = (0..6 * 2).map(|i| i as f32 * 0.2).collect();
        let alpha: Vec<f32> = vec![16.0];

        let down_bytes: Vec<u8> = down.iter().flat_map(|f| f.to_le_bytes()).collect();
        let up_bytes: Vec<u8> = up.iter().flat_map(|f| f.to_le_bytes()).collect();
        let alpha_bytes: Vec<u8> = alpha.iter().flat_map(|f| f.to_le_bytes()).collect();

        let down_view = TensorView::new(safetensors::Dtype::F32, vec![2, 4], &down_bytes).unwrap();
        let up_view = TensorView::new(safetensors::Dtype::F32, vec![6, 2], &up_bytes).unwrap();
        let alpha_view = TensorView::new(safetensors::Dtype::F32, vec![], &alpha_bytes).unwrap();

        let entries: Vec<(String, TensorView)> = vec![
            (format!("{layer}.lora_down.weight"), down_view),
            (format!("{layer}.lora_up.weight"), up_view),
            (format!("{layer}.alpha"), alpha_view),
        ];
        safetensors::serialize_to_file(entries, &None, path).expect("write safetensors");
    }

    /// Loading the same path twice must hand back the exact same
    /// `Arc<LoraAdapter>` — `Arc::ptr_eq` is the strongest possible
    /// hit-test (no parsed-from-disk twin can ever satisfy it).
    #[test]
    fn parsed_lora_cache_is_a_hit_on_second_load() {
        let _cache_test_guard = lock_parsed_lora_cache_tests();
        clear_parsed_lora_cache_for_test();
        let dir = tempfile::tempdir().expect("tempdir");
        let path = dir.path().join("hit.safetensors");
        write_synthetic_kohya_safetensors(&path);

        let first = get_or_load_adapter(&path).expect("first load");
        let second = get_or_load_adapter(&path).expect("second load");
        assert!(
            Arc::ptr_eq(&first, &second),
            "second load must return the same Arc — proof the cache hit, no re-parse"
        );
    }

    /// When the file mtime changes (e.g. the user re-trains and saves
    /// over the same path), the cache key changes, so the second load
    /// must produce a NEW `Arc` even though the path string is
    /// identical.
    #[test]
    fn parsed_lora_cache_invalidates_on_mtime_change() {
        let _cache_test_guard = lock_parsed_lora_cache_tests();
        clear_parsed_lora_cache_for_test();
        let dir = tempfile::tempdir().expect("tempdir");
        let path = dir.path().join("invalidate.safetensors");
        write_synthetic_kohya_safetensors(&path);

        let first = get_or_load_adapter(&path).expect("first load");

        // Bump the file mtime by rewriting the file with fresh
        // contents. We sleep just long enough to exceed common
        // file-system mtime resolutions (HFS+ on older macOS rounds
        // to 1 s) so `metadata.modified()` actually changes.
        std::thread::sleep(std::time::Duration::from_millis(1100));
        write_synthetic_kohya_safetensors(&path);

        let second = get_or_load_adapter(&path).expect("second load");
        assert!(
            !Arc::ptr_eq(&first, &second),
            "mtime change must produce a fresh Arc — proof the cache key invalidated"
        );
    }
}
