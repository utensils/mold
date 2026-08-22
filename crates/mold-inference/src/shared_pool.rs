use std::collections::HashMap;
use std::path::Path;
use std::sync::Arc;

use candle_core::Tensor;
use tokenizers::Tokenizer;

/// How many bytes of CPU-resident component weights the pool keeps alive for
/// reuse, in total, across every cached entry.
///
/// The cache exists to skip a re-read on model swap. The OS page cache already
/// provides that service *reclaimably*: a component read through
/// `weight_loader`'s mmap path costs file-backed pages the kernel can drop
/// under pressure, and `MemAvailable` — the input to the scheduler's host
/// memory ledger — already counts them as free. Parking a second, anonymous
/// copy in this map converts that reclaimable memory into memory nothing can
/// take back and host admission must charge in full.
///
/// For a 335 MB VAE that trade is cheap and the reuse is real. For FLUX's
/// 9.79 GB `t5xxl_fp16.safetensors` it is not: before #1273 one FLUX
/// generation left that whole file resident as anonymous host RAM for the
/// lifetime of the process — `DELETE /api/models/unload` did not touch it,
/// dropping the engine did not touch it, and a 64 GB host that had rendered a
/// few images could no longer admit MiniMax H3 without a restart.
///
/// So the budget is deliberately small: large enough for the VAE/CLIP-class
/// components the pool was introduced for, small enough that a text encoder or
/// a transformer is served from its own (reclaimable) path instead. Callers
/// that genuinely want an encoder resident in host RAM between requests have
/// `MOLD_KEEP_TE_RAM=1`, which parks it *inside the engine*, where `unload()`
/// releases it and admission can see it go.
pub const CPU_TENSOR_CACHE_BUDGET_BYTES: u64 = 2 << 30;

struct CachedCpuTensors {
    tensors: Arc<HashMap<String, Tensor>>,
    /// On-disk size of the component set, used as the retention weight. It is
    /// known before the read, so an over-budget component is never even
    /// materialized.
    bytes: u64,
    last_used: u64,
}

/// Cross-engine cache for tokenizers (and potentially prompt embeddings in the future).
/// Tokenizers are keyed by their canonical file path. Thread-safe when wrapped in `Arc<Mutex<>>`.
pub struct SharedPool {
    tokenizers: HashMap<String, Arc<Tokenizer>>,
    cpu_tensors: HashMap<String, CachedCpuTensors>,
    cpu_tensor_clock: u64,
    cpu_tensor_budget_bytes: u64,
}

impl Default for SharedPool {
    fn default() -> Self {
        Self {
            tokenizers: HashMap::new(),
            cpu_tensors: HashMap::new(),
            cpu_tensor_clock: 0,
            cpu_tensor_budget_bytes: CPU_TENSOR_CACHE_BUDGET_BYTES,
        }
    }
}

impl SharedPool {
    pub fn new() -> Self {
        Self::default()
    }

    /// A pool with a non-default CPU tensor retention budget. Tests use this to
    /// exercise the budget without writing multi-gigabyte fixtures.
    pub fn with_cpu_tensor_budget(budget_bytes: u64) -> Self {
        Self {
            cpu_tensor_budget_bytes: budget_bytes,
            ..Self::default()
        }
    }

    /// Get a cached tokenizer by file path, or None if not cached.
    pub fn get_tokenizer(&self, path: &str) -> Option<Arc<Tokenizer>> {
        self.tokenizers.get(path).cloned()
    }

    /// Cache a tokenizer by file path.
    pub fn insert_tokenizer(&mut self, path: String, tokenizer: Arc<Tokenizer>) {
        self.tokenizers.insert(path, tokenizer);
    }

    /// Load a tokenizer by file path, returning the already-cached handle when present.
    pub fn load_tokenizer(&mut self, path: &Path) -> anyhow::Result<Arc<Tokenizer>> {
        let key = path.to_string_lossy().into_owned();
        if let Some(tokenizer) = self.tokenizers.get(&key) {
            return Ok(tokenizer.clone());
        }

        let tokenizer =
            Arc::new(Tokenizer::from_file(path).map_err(|e| {
                anyhow::anyhow!("failed to load tokenizer {}: {e}", path.display())
            })?);
        self.tokenizers.insert(key, tokenizer.clone());
        Ok(tokenizer)
    }

    /// Load safetensors-backed weights into CPU RAM, returning the tensor map.
    ///
    /// The map is retained for reuse only when it fits
    /// [`CPU_TENSOR_CACHE_BUDGET_BYTES`]; an over-budget component is still
    /// returned, just not kept alive after the caller drops its handle.
    pub fn load_cpu_tensors(
        &mut self,
        paths: &[impl AsRef<Path>],
    ) -> anyhow::Result<Arc<HashMap<String, Tensor>>> {
        let (key, bytes) = cpu_tensor_cache_entry(paths)?;
        if let Some(tensors) = self.touch_cpu_tensors(&key) {
            return Ok(tensors);
        }

        let tensors = Arc::new(crate::encoders::park::load_tensors_to_cpu(paths)?);
        self.retain_cpu_tensors(key, tensors.clone(), bytes);
        Ok(tensors)
    }

    /// Load CPU tensors only when every component path is safetensors-backed
    /// *and* the component set is small enough to retain.
    ///
    /// `None` means "this pool is not the right home for these weights" — the
    /// caller falls back to its own loader, which for every current caller is
    /// the mmap-backed `weight_loader` path straight to the target device. That
    /// is the reclaimable option, and for anything encoder-sized it is the one
    /// we want (#1273).
    pub fn load_safetensors_cpu_tensors(
        &mut self,
        paths: &[impl AsRef<Path>],
    ) -> anyhow::Result<Option<Arc<HashMap<String, Tensor>>>> {
        if paths.iter().any(|path| {
            path.as_ref()
                .extension()
                .and_then(|ext| ext.to_str())
                .map(|ext| !ext.eq_ignore_ascii_case("safetensors"))
                .unwrap_or(true)
        }) {
            return Ok(None);
        }

        let (key, bytes) = cpu_tensor_cache_entry(paths)?;
        if let Some(tensors) = self.touch_cpu_tensors(&key) {
            return Ok(Some(tensors));
        }
        if bytes > self.cpu_tensor_budget_bytes {
            return Ok(None);
        }

        let tensors = Arc::new(crate::encoders::park::load_tensors_to_cpu(paths)?);
        self.retain_cpu_tensors(key, tensors.clone(), bytes);
        Ok(Some(tensors))
    }

    /// Bytes of CPU-resident component weights the pool is currently keeping
    /// alive for reuse.
    pub fn retained_cpu_tensor_bytes(&self) -> u64 {
        self.cpu_tensors
            .values()
            .map(|entry| entry.bytes)
            .fold(0u64, |total, bytes| total.saturating_add(bytes))
    }

    /// Drop every cached component map that nothing else still holds, and
    /// report how many bytes that released.
    ///
    /// An entry an engine is still streaming from — SD3's offloaded MMDiT reads
    /// its blocks out of the cached map for the life of the engine — has a
    /// strong count above one and is never taken from under it.
    pub fn release_unreferenced_cpu_tensors(&mut self) -> u64 {
        let mut released = 0u64;
        self.cpu_tensors.retain(|_, entry| {
            if Arc::strong_count(&entry.tensors) == 1 {
                released = released.saturating_add(entry.bytes);
                false
            } else {
                true
            }
        });
        released
    }

    fn touch_cpu_tensors(&mut self, key: &str) -> Option<Arc<HashMap<String, Tensor>>> {
        self.cpu_tensor_clock = self.cpu_tensor_clock.saturating_add(1);
        let clock = self.cpu_tensor_clock;
        let entry = self.cpu_tensors.get_mut(key)?;
        entry.last_used = clock;
        Some(entry.tensors.clone())
    }

    fn retain_cpu_tensors(
        &mut self,
        key: String,
        tensors: Arc<HashMap<String, Tensor>>,
        bytes: u64,
    ) {
        if bytes > self.cpu_tensor_budget_bytes {
            return;
        }
        while self.retained_cpu_tensor_bytes().saturating_add(bytes) > self.cpu_tensor_budget_bytes
        {
            if !self.evict_one_unreferenced_cpu_tensors() {
                // Everything still cached is in use; retaining anyway would
                // put the pool over budget, so this entry simply is not kept.
                return;
            }
        }
        self.cpu_tensor_clock = self.cpu_tensor_clock.saturating_add(1);
        let last_used = self.cpu_tensor_clock;
        self.cpu_tensors.insert(
            key,
            CachedCpuTensors {
                tensors,
                bytes,
                last_used,
            },
        );
    }

    fn evict_one_unreferenced_cpu_tensors(&mut self) -> bool {
        let victim = self
            .cpu_tensors
            .iter()
            .filter(|(_, entry)| Arc::strong_count(&entry.tensors) == 1)
            .min_by_key(|(_, entry)| entry.last_used)
            .map(|(key, _)| key.clone());
        match victim {
            Some(key) => {
                self.cpu_tensors.remove(&key);
                true
            }
            None => false,
        }
    }
}

/// The cache key for a component set, plus its total on-disk size.
///
/// Size comes from the directory entries rather than the loaded tensors so the
/// retention decision can be made *before* the read — an over-budget encoder is
/// never materialized as anonymous host RAM at all.
fn cpu_tensor_cache_entry(paths: &[impl AsRef<Path>]) -> anyhow::Result<(String, u64)> {
    let mut parts = Vec::with_capacity(paths.len());
    let mut bytes = 0u64;
    for path in paths {
        let path = path.as_ref();
        let canonical = path.canonicalize()?;
        let metadata = std::fs::metadata(&canonical)?;
        let modified = metadata
            .modified()?
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_nanos();
        bytes = bytes.saturating_add(metadata.len());
        parts.push(format!(
            "{}:{}:{}",
            canonical.display(),
            metadata.len(),
            modified
        ));
    }
    Ok((parts.join("|"), bytes))
}

#[cfg(test)]
mod tests {
    use super::*;
    use safetensors::tensor::{serialize_to_file, Dtype as SafeDtype, TensorView};
    use tokenizers::models::bpe::BPE;

    #[test]
    fn load_tokenizer_reuses_cached_handle_for_the_same_path() {
        let dir = tempfile::tempdir().unwrap();
        let tokenizer_path = dir.path().join("tokenizer.json");
        Tokenizer::new(BPE::default())
            .save(&tokenizer_path, false)
            .unwrap();

        let mut pool = SharedPool::new();
        let first = pool.load_tokenizer(&tokenizer_path).unwrap();
        let second = pool.load_tokenizer(&tokenizer_path).unwrap();

        assert!(Arc::ptr_eq(&first, &second));
    }

    #[test]
    fn load_cpu_tensors_reuses_cached_handle_for_the_same_component_paths() {
        let dir = tempfile::tempdir().unwrap();
        let weights_path = dir.path().join("vae.safetensors");
        let bytes = 1.0f32
            .to_le_bytes()
            .into_iter()
            .chain(2.0f32.to_le_bytes())
            .collect::<Vec<_>>();
        let mut tensors = HashMap::new();
        tensors.insert(
            "weight".to_string(),
            TensorView::new(SafeDtype::F32, vec![2], &bytes).unwrap(),
        );
        serialize_to_file(&tensors, &None, &weights_path).unwrap();

        let mut pool = SharedPool::new();
        let first = pool
            .load_cpu_tensors(std::slice::from_ref(&weights_path))
            .unwrap();
        let second = pool
            .load_cpu_tensors(std::slice::from_ref(&weights_path))
            .unwrap();

        assert!(Arc::ptr_eq(&first, &second));
        assert!(first.contains_key("weight"));
    }

    #[test]
    fn load_safetensors_cpu_tensors_skips_non_safetensors_paths() {
        let dir = tempfile::tempdir().unwrap();
        let gguf_path = dir.path().join("t5-q8.gguf");
        std::fs::write(&gguf_path, b"not safetensors").unwrap();

        let mut pool = SharedPool::new();

        assert!(pool
            .load_safetensors_cpu_tensors(std::slice::from_ref(&gguf_path))
            .unwrap()
            .is_none());
    }

    /// Write a safetensors file whose payload is at least `payload_bytes`
    /// long, so the cache's on-disk size accounting has something to weigh.
    fn temp_safetensors_of_size(
        dir: &Path,
        name: &str,
        payload_bytes: usize,
    ) -> std::path::PathBuf {
        let path = dir.join(name);
        let floats = payload_bytes.div_ceil(4).max(1);
        let bytes = vec![0u8; floats * 4];
        let mut tensors = HashMap::new();
        tensors.insert(
            "weight".to_string(),
            TensorView::new(SafeDtype::F32, vec![floats], &bytes).unwrap(),
        );
        serialize_to_file(&tensors, &None, &path).unwrap();
        path
    }

    /// #1273: an encoder that does not fit the retention budget must be served
    /// without being retained. Holding an anonymous copy of a multi-GB
    /// component converts reclaimable page cache into memory the kernel can
    /// never take back and host admission must charge in full — which is what
    /// made a 64 GB host refuse MiniMax H3 after ordinary image work.
    #[test]
    fn cpu_tensor_cache_declines_to_retain_a_component_over_its_budget() {
        let dir = tempfile::tempdir().unwrap();
        let big = temp_safetensors_of_size(dir.path(), "big.safetensors", 64 * 1024);

        let mut pool = SharedPool::with_cpu_tensor_budget(1024);

        // The optional accessor reports "not cached here" so the caller takes
        // its own (mmap-backed, reclaimable) load path.
        assert!(pool
            .load_safetensors_cpu_tensors(std::slice::from_ref(&big))
            .unwrap()
            .is_none());
        assert_eq!(pool.retained_cpu_tensor_bytes(), 0);

        // The unconditional accessor still returns the weights — it just does
        // not keep a copy alive after the caller drops its handle.
        let loaded = pool.load_cpu_tensors(std::slice::from_ref(&big)).unwrap();
        assert!(loaded.contains_key("weight"));
        assert_eq!(pool.retained_cpu_tensor_bytes(), 0);
    }

    /// A component that fits is still cached — the small VAE/CLIP reuse the
    /// pool was introduced for must keep working.
    #[test]
    fn cpu_tensor_cache_retains_components_that_fit_the_budget() {
        let dir = tempfile::tempdir().unwrap();
        let small = temp_safetensors_of_size(dir.path(), "small.safetensors", 1024);

        let mut pool = SharedPool::with_cpu_tensor_budget(1 << 20);
        let first = pool
            .load_safetensors_cpu_tensors(std::slice::from_ref(&small))
            .unwrap()
            .unwrap();
        assert!(pool.retained_cpu_tensor_bytes() > 0);
        let second = pool
            .load_safetensors_cpu_tensors(std::slice::from_ref(&small))
            .unwrap()
            .unwrap();
        assert!(Arc::ptr_eq(&first, &second));
    }

    /// Over budget, the least-recently-used entry that nobody else holds is
    /// the one that goes.
    #[test]
    fn cpu_tensor_cache_evicts_the_least_recently_used_unreferenced_entry() {
        let dir = tempfile::tempdir().unwrap();
        let a = temp_safetensors_of_size(dir.path(), "a.safetensors", 4096);
        let b = temp_safetensors_of_size(dir.path(), "b.safetensors", 4096);

        let budget = std::fs::metadata(&a).unwrap().len() + 16;
        let mut pool = SharedPool::with_cpu_tensor_budget(budget);

        drop(pool.load_cpu_tensors(std::slice::from_ref(&a)).unwrap());
        assert!(pool.retained_cpu_tensor_bytes() > 0);
        drop(pool.load_cpu_tensors(std::slice::from_ref(&b)).unwrap());

        // `b` displaced `a` rather than accumulating beside it.
        assert!(pool.retained_cpu_tensor_bytes() <= budget);
        let a_again = pool.load_cpu_tensors(std::slice::from_ref(&a)).unwrap();
        let b_again = pool.load_cpu_tensors(std::slice::from_ref(&b)).unwrap();
        assert!(!Arc::ptr_eq(&a_again, &b_again));
    }

    /// An entry an engine still holds (SD3's offloaded transformer streams its
    /// blocks straight out of the cached map) is never taken from under it.
    #[test]
    fn cpu_tensor_cache_never_evicts_an_entry_another_owner_still_holds() {
        let dir = tempfile::tempdir().unwrap();
        let a = temp_safetensors_of_size(dir.path(), "a.safetensors", 4096);
        let b = temp_safetensors_of_size(dir.path(), "b.safetensors", 4096);

        let budget = std::fs::metadata(&a).unwrap().len() + 16;
        let mut pool = SharedPool::with_cpu_tensor_budget(budget);

        let held = pool.load_cpu_tensors(std::slice::from_ref(&a)).unwrap();
        drop(pool.load_cpu_tensors(std::slice::from_ref(&b)).unwrap());

        let a_again = pool.load_cpu_tensors(std::slice::from_ref(&a)).unwrap();
        assert!(
            Arc::ptr_eq(&held, &a_again),
            "a live handle must survive budget pressure"
        );
    }

    /// #1273: `DELETE /api/models/unload` has to be able to give host RAM back.
    #[test]
    fn release_unreferenced_cpu_tensors_frees_only_what_nobody_holds() {
        let dir = tempfile::tempdir().unwrap();
        let a = temp_safetensors_of_size(dir.path(), "a.safetensors", 4096);
        let b = temp_safetensors_of_size(dir.path(), "b.safetensors", 4096);

        let mut pool = SharedPool::with_cpu_tensor_budget(1 << 20);
        let held = pool.load_cpu_tensors(std::slice::from_ref(&a)).unwrap();
        drop(pool.load_cpu_tensors(std::slice::from_ref(&b)).unwrap());
        let retained_before = pool.retained_cpu_tensor_bytes();

        let released = pool.release_unreferenced_cpu_tensors();
        assert!(released > 0, "the unheld entry must be released");
        assert_eq!(pool.retained_cpu_tensor_bytes(), retained_before - released);

        let a_again = pool.load_cpu_tensors(std::slice::from_ref(&a)).unwrap();
        assert!(
            Arc::ptr_eq(&held, &a_again),
            "a live handle must survive an explicit release"
        );
    }

    #[test]
    fn load_cpu_var_builder_reuses_cached_tensor_map_for_same_encoder_path() {
        let dir = tempfile::tempdir().unwrap();
        let weights_path = dir.path().join("encoder.safetensors");
        let weight = [1.0f32, 2.0, 3.0, 4.0];
        let bias = [0.5f32, -0.5];
        let mut weight_bytes = Vec::with_capacity(weight.len() * 4);
        for value in weight {
            weight_bytes.extend_from_slice(&value.to_le_bytes());
        }
        let mut bias_bytes = Vec::with_capacity(bias.len() * 4);
        for value in bias {
            bias_bytes.extend_from_slice(&value.to_le_bytes());
        }
        let mut tensors = HashMap::new();
        tensors.insert(
            "weight".to_string(),
            TensorView::new(SafeDtype::F32, vec![2, 2], &weight_bytes).unwrap(),
        );
        tensors.insert(
            "bias".to_string(),
            TensorView::new(SafeDtype::F32, vec![2], &bias_bytes).unwrap(),
        );
        serialize_to_file(&tensors, &None, &weights_path).unwrap();

        let mut pool = SharedPool::new();
        let first = pool
            .load_safetensors_cpu_tensors(std::slice::from_ref(&weights_path))
            .unwrap()
            .unwrap();
        let second = pool
            .load_safetensors_cpu_tensors(std::slice::from_ref(&weights_path))
            .unwrap()
            .unwrap();
        let vb = crate::encoders::park::varbuilder_from_parked(
            first.as_ref(),
            candle_core::DType::F32,
            &candle_core::Device::Cpu,
        );
        let linear = candle_nn::linear(2, 2, vb).unwrap();
        let input = candle_core::Tensor::new(&[10.0f32, 20.0], &candle_core::Device::Cpu)
            .unwrap()
            .unsqueeze(0)
            .unwrap();
        let output = candle_nn::Module::forward(&linear, &input).unwrap();
        let values = output.squeeze(0).unwrap().to_vec1::<f32>().unwrap();

        assert!(Arc::ptr_eq(&first, &second));
        assert_eq!(values, vec![50.5, 109.5]);
    }
}
