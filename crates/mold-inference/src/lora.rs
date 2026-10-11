//! Family-neutral LoRA parsing, file revisions and bounded derived-weight caches.
//! Mapping model keys and merging tensors remain in their owning engines.

pub(crate) mod unet;

use std::collections::{HashMap, VecDeque};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex, OnceLock};
use std::time::SystemTime;

use anyhow::{bail, Result};
use candle_core::{DType, Device, Tensor};

/// Metadata identifies installed bytes without reading multi-GB adapters per request.
#[derive(Clone, Debug, Hash, PartialEq, Eq)]
pub(crate) struct FileRevision {
    canonical_path: PathBuf,
    bytes: u64,
    modified: SystemTime,
    #[cfg(unix)]
    unix: (u64, u64, i64, i64),
}

impl FileRevision {
    pub(crate) fn read(path: &Path) -> Option<Self> {
        let canonical_path = std::fs::canonicalize(path).ok()?;
        let metadata = std::fs::metadata(&canonical_path).ok()?;
        if !metadata.is_file() {
            return None;
        }
        #[cfg(unix)]
        let unix = {
            use std::os::unix::fs::MetadataExt;
            (
                metadata.dev(),
                metadata.ino(),
                metadata.ctime(),
                metadata.ctime_nsec(),
            )
        };
        Some(Self {
            canonical_path,
            bytes: metadata.len(),
            modified: metadata.modified().ok()?,
            #[cfg(unix)]
            unix,
        })
    }
}

#[derive(Clone, Debug, Hash, PartialEq, Eq)]
pub(crate) enum AdapterIdentity {
    File(FileRevision),
    /// Each failed observation is distinct and is never admitted to a cache.
    Unobservable(u64),
    #[cfg(test)]
    Synthetic(u64),
}

fn next_identity() -> u64 {
    static NEXT: AtomicU64 = AtomicU64::new(1);
    NEXT.fetch_add(1, Ordering::Relaxed)
}

impl AdapterIdentity {
    fn unobservable() -> Self {
        Self::Unobservable(next_identity())
    }
    fn read(path: &Path) -> Self {
        FileRevision::read(path)
            .map(Self::File)
            .unwrap_or_else(Self::unobservable)
    }
    fn cacheable(&self) -> bool {
        !matches!(self, Self::Unobservable(_))
    }
}

/// One resident adapter, in request order. Unknown revisions never permit reuse.
#[derive(Clone, Debug, Hash, PartialEq, Eq)]
pub(crate) struct LoraFingerprint {
    identity: AdapterIdentity,
    scale_bits: u64,
}

impl LoraFingerprint {
    pub(crate) fn new(path: &Path, scale: f64) -> Self {
        Self {
            identity: AdapterIdentity::read(path),
            scale_bits: scale.to_bits(),
        }
    }
}

pub(crate) fn fingerprint_stack(loras: &[mold_core::LoraWeight]) -> Vec<LoraFingerprint> {
    loras
        .iter()
        .map(|lora| LoraFingerprint::new(Path::new(&lora.path), lora.scale))
        .collect()
}

/// Fence the whole load as well as individual parsing. A changed or unobservable
/// file may be used for this request, but can never authorize resident reuse.
pub(crate) fn fence_fingerprint(
    before: Vec<LoraFingerprint>,
    after: Vec<LoraFingerprint>,
) -> Vec<LoraFingerprint> {
    if before == after {
        return before;
    }
    before
        .into_iter()
        .map(|mut item| {
            item.identity = AdapterIdentity::unobservable();
            item
        })
        .collect()
}

/// Source names, rather than randomized HashMap iteration positions, identify Q/K/V.
#[derive(Clone, Debug, Hash, PartialEq, Eq)]
pub(crate) struct LoraCacheKey {
    pub(crate) tensor_name: String,
    pub(crate) source_layer: String,
    pub(crate) adapter_identity: AdapterIdentity,
    pub(crate) scale_bits: u64,
}

pub(crate) const LORA_DELTA_CACHE_BYTES: usize = 256 * 1024 * 1024;

/// Scaled F32 deltas preserve the existing compute/multiply order on each backend.
/// The byte bound is per engine; unload releases all retained deltas.
pub(crate) struct LoraDeltaCache {
    deltas: HashMap<LoraCacheKey, Tensor>,
    order: VecDeque<LoraCacheKey>,
    bytes: usize,
    capacity_bytes: usize,
}

impl LoraDeltaCache {
    pub(crate) fn new() -> Self {
        Self::with_capacity_bytes(LORA_DELTA_CACHE_BYTES)
    }
    fn with_capacity_bytes(capacity_bytes: usize) -> Self {
        Self {
            deltas: HashMap::new(),
            order: VecDeque::new(),
            bytes: 0,
            capacity_bytes,
        }
    }
    pub(crate) fn get(&self, key: &LoraCacheKey) -> Option<&Tensor> {
        if !key.adapter_identity.cacheable() {
            return None;
        }
        self.deltas.get(key)
    }
    pub(crate) fn insert(&mut self, key: LoraCacheKey, delta: &Tensor) -> candle_core::Result<()> {
        let Some(bytes) = delta
            .elem_count()
            .checked_mul(delta.dtype().size_in_bytes())
        else {
            return Ok(());
        };
        // Refuse before making the host copy. Unknown metadata never earns cache authority.
        if !key.adapter_identity.cacheable() || bytes > self.capacity_bytes {
            return Ok(());
        }
        if let Some(previous) = self.deltas.remove(&key) {
            self.bytes -= previous.elem_count() * previous.dtype().size_in_bytes();
            self.order.retain(|existing| existing != &key);
        }
        while self.bytes > self.capacity_bytes - bytes {
            let Some(oldest) = self.order.pop_front() else {
                break;
            };
            if let Some(previous) = self.deltas.remove(&oldest) {
                self.bytes -= previous.elem_count() * previous.dtype().size_in_bytes();
            }
        }
        let cpu = delta.to_device(&Device::Cpu)?;
        self.deltas.insert(key.clone(), cpu);
        self.order.push_back(key);
        self.bytes += bytes;
        Ok(())
    }
    pub(crate) fn clear(&mut self) {
        self.deltas.clear();
        self.order.clear();
        self.bytes = 0;
    }
}

const PARSED_LORA_CACHE_CAPACITY: usize = 4;
struct ParsedLoraCache {
    order: VecDeque<FileRevision>,
    entries: HashMap<FileRevision, Arc<LoraAdapter>>,
}
impl ParsedLoraCache {
    fn new() -> Self {
        Self {
            order: VecDeque::new(),
            entries: HashMap::new(),
        }
    }
    fn insert(&mut self, revision: FileRevision, adapter: Arc<LoraAdapter>) {
        self.order.retain(|existing| existing != &revision);
        self.entries.insert(revision.clone(), adapter);
        self.order.push_back(revision);
        while self.entries.len() > PARSED_LORA_CACHE_CAPACITY {
            if let Some(oldest) = self.order.pop_front() {
                self.entries.remove(&oldest);
            }
        }
    }
}
fn parsed_lora_cache() -> &'static Mutex<ParsedLoraCache> {
    static CACHE: OnceLock<Mutex<ParsedLoraCache>> = OnceLock::new();
    CACHE.get_or_init(|| Mutex::new(ParsedLoraCache::new()))
}

pub(crate) fn get_or_load_adapter(path: &Path) -> Result<Arc<LoraAdapter>> {
    if let Some(revision) = FileRevision::read(path) {
        if let Some(adapter) = parsed_lora_cache()
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .entries
            .get(&revision)
            .cloned()
        {
            return Ok(adapter);
        }
    }
    let adapter = Arc::new(LoraAdapter::load(path)?);
    if let AdapterIdentity::File(revision) = &adapter.identity {
        parsed_lora_cache()
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .insert(revision.clone(), Arc::clone(&adapter));
    }
    Ok(adapter)
}

#[cfg(test)]
pub(crate) fn clear_parsed_lora_cache_for_test() {
    let mut cache = parsed_lora_cache()
        .lock()
        .unwrap_or_else(|e| e.into_inner());
    cache.entries.clear();
    cache.order.clear();
}
#[cfg(test)]
pub(crate) fn lock_parsed_lora_cache_tests() -> std::sync::MutexGuard<'static, ()> {
    static LOCK: Mutex<()> = Mutex::new(());
    LOCK.lock().unwrap_or_else(|e| e.into_inner())
}

/// A parsed LoRA adapter: pairs of (A, B) weight matrices keyed by layer name.
pub(crate) struct LoraAdapter {
    /// Map from diffusers layer name (without lora_A/lora_B suffix) to (A, B) tensors.
    pub layers: HashMap<String, LoraLayer>,
    pub rank: usize,
    pub(crate) identity: AdapterIdentity,
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

/// Name a LoRA layout mold does not parse, when the file's keys identify one.
///
/// An adapter matching none of the supported suffixes used to fail with the
/// file name and nothing else, which is indistinguishable from a corrupt
/// download. XLabs-AI's FLUX.1 adapters are the case worth naming: they carry
/// `double_blocks.N.processor.{proj,qkv}_lora{1,2}.{down,up}.weight`, and
/// mold's matcher accepts only the diffusers/PEFT, Kohya, OneTrainer and
/// PEFT-default conventions. Their `.down.weight` is preceded by `proj_lora1`
/// rather than by `.lora`, so it matches nothing.
///
/// This layout has NEVER been supported. Before the LoRA sealing fix such an
/// adapter was discarded before it reached the parser, so the render
/// succeeded with no adapter applied at all; now it reaches here and fails.
/// The failure is the honest answer — a silent no-op is worse — but it is a
/// user-visible change, so the message says which layout it saw.
pub(crate) fn unsupported_layout_hint<'a>(keys: impl Iterator<Item = &'a String>) -> String {
    let xlabs = keys
        .into_iter()
        .any(|key| key.contains(".processor.") && key.contains("_lora"));
    if xlabs {
        return " \u{2014} these are XLabs-AI `double_blocks.N.processor.*_lora*.{down,up}.weight` \
keys, a layout mold does not parse. Use a diffusers/PEFT (`lora_A`/`lora_B`) or Kohya \
(`lora_down`/`lora_up`) export of the same adapter."
            .to_string();
    }
    String::new()
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
// ComfyUI 7f7fd918 comfy/lora.py:41-44 reads alpha with .item(),
// independently of storage dtype; weight_adapter/lora.py:42-48 applies alpha/rank.
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
    /// Extend the parser's revision fence over a family-specific header read.
    pub(crate) fn fence_revision(&mut self, before: Option<FileRevision>) -> Result<()> {
        match (before, &self.identity) {
            (Some(before), AdapterIdentity::File(after)) => {
                anyhow::ensure!(before == *after, "LoRA changed while reading its metadata")
            }
            _ => self.identity = AdapterIdentity::unobservable(),
        }
        Ok(())
    }

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
        let before = FileRevision::read(path);
        let tensors = candle_core::safetensors::load(path, &Device::Cpu)?;
        let after = FileRevision::read(path);
        anyhow::ensure!(
            before.is_none() || after.is_none() || before == after,
            "LoRA changed while loading: {}",
            path.display()
        );
        let identity = match (before, after) {
            (Some(before), Some(after)) if before == after => AdapterIdentity::File(after),
            _ => AdapterIdentity::unobservable(),
        };
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
            bail!(
                "no LoRA A/B pairs found in {}{}",
                path.display(),
                unsupported_layout_hint(tensors.keys())
            );
        }

        Ok(Self {
            layers,
            rank,
            identity,
        })
    }
}

#[cfg(test)]
impl LoraAdapter {
    pub(crate) fn from_layers(layers: HashMap<String, LoraLayer>, rank: usize) -> Self {
        Self {
            layers,
            rank,
            identity: AdapterIdentity::Synthetic(next_identity()),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn a_load_fence_never_stamps_changed_or_unobservable_revisions() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("adapter");
        std::fs::write(&path, b"old").unwrap();
        let before = vec![LoraFingerprint::new(&path, 1.0)];
        assert_eq!(fence_fingerprint(before.clone(), before.clone()), before);
        let replacement = dir.path().join("new");
        std::fs::write(&replacement, b"new").unwrap();
        std::fs::rename(replacement, &path).unwrap();
        let after = vec![LoraFingerprint::new(&path, 1.0)];
        let fenced = fence_fingerprint(before.clone(), after.clone());
        assert_ne!(fenced, before);
        assert_ne!(fenced, after);
        assert!(matches!(
            fenced[0].identity,
            AdapterIdentity::Unobservable(_)
        ));
        std::fs::remove_file(&path).unwrap();
        let unknown = vec![LoraFingerprint::new(&path, 1.0)];
        assert_ne!(fence_fingerprint(before, unknown.clone()), unknown);
    }

    #[test]
    fn family_metadata_must_belong_to_the_loaded_tensor_revision() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("adapter.safetensors");
        let save = |value| {
            candle_core::safetensors::save(
                &HashMap::from([
                    (
                        "q.lora_A.weight",
                        Tensor::ones((1, 1), DType::F32, &Device::Cpu).unwrap(),
                    ),
                    (
                        "q.lora_B.weight",
                        Tensor::full(value, (1, 1), &Device::Cpu).unwrap(),
                    ),
                ]),
                &path,
            )
            .unwrap()
        };
        save(1.0f32);
        let before = FileRevision::read(&path);
        let mut adapter = LoraAdapter::load(&path).unwrap();
        adapter.fence_revision(before.clone()).unwrap();
        save(2.0f32);
        let mut replacement = LoraAdapter::load(&path).unwrap();
        assert!(replacement.fence_revision(before).is_err());
        replacement.fence_revision(None).unwrap();
        assert!(!replacement.identity.cacheable());
    }

    fn key(scale: f64) -> LoraCacheKey {
        LoraCacheKey {
            tensor_name: "qkv".into(),
            source_layer: "q".into(),
            adapter_identity: AdapterIdentity::Synthetic(1),
            scale_bits: scale.to_bits(),
        }
    }
    #[test]
    fn delta_cache_bounds_strength_sweeps_and_replacement_by_bytes() {
        let delta = Tensor::ones(2, DType::F32, &Device::Cpu).unwrap();
        let mut cache = LoraDeltaCache::with_capacity_bytes(16);
        for scale in 0..100 {
            cache.insert(key(scale as f64), &delta).unwrap();
            assert!(cache.bytes <= 16);
        }
        assert_eq!(cache.bytes, 16);
        assert!(cache.get(&key(0.0)).is_none());
        assert!(cache.get(&key(99.0)).is_some());
        cache.insert(key(99.0), &delta).unwrap();
        assert_eq!(cache.bytes, 16);
        let oversized = Tensor::ones(5, DType::F32, &Device::Cpu).unwrap();
        cache.insert(key(101.0), &oversized).unwrap();
        assert_eq!(cache.bytes, 16);
        assert!(cache.get(&key(101.0)).is_none());
        cache.clear();
        assert_eq!(cache.bytes, 0);
        assert!(cache.deltas.is_empty());
    }
    #[test]
    fn unknown_metadata_never_enters_delta_cache_or_reuses_a_fingerprint() {
        let dir = tempfile::tempdir().unwrap();
        let missing = dir.path().join("missing");
        assert!(FileRevision::read(&missing).is_none());
        assert_ne!(
            LoraFingerprint::new(&missing, 1.0),
            LoraFingerprint::new(&missing, 1.0)
        );
        let mut cache = LoraDeltaCache::with_capacity_bytes(16);
        let mut unknown = key(1.0);
        unknown.adapter_identity = AdapterIdentity::unobservable();
        cache
            .insert(
                unknown.clone(),
                &Tensor::ones(2, DType::F32, &Device::Cpu).unwrap(),
            )
            .unwrap();
        assert!(cache.get(&unknown).is_none());
        assert_eq!(cache.bytes, 0);
    }
}
