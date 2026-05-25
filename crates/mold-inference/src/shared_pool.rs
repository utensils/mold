use std::collections::HashMap;
use std::path::Path;
use std::sync::Arc;

use candle_core::Tensor;
use tokenizers::Tokenizer;

/// Cross-engine cache for tokenizers (and potentially prompt embeddings in the future).
/// Tokenizers are keyed by their canonical file path. Thread-safe when wrapped in `Arc<Mutex<>>`.
#[derive(Default)]
pub struct SharedPool {
    tokenizers: HashMap<String, Arc<Tokenizer>>,
    cpu_tensors: HashMap<String, Arc<HashMap<String, Tensor>>>,
}

impl SharedPool {
    pub fn new() -> Self {
        Self::default()
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

    /// Load safetensors-backed weights into CPU RAM, returning the cached tensor map.
    pub(crate) fn load_cpu_tensors(
        &mut self,
        paths: &[impl AsRef<Path>],
    ) -> anyhow::Result<Arc<HashMap<String, Tensor>>> {
        let key = cpu_tensor_cache_key(paths)?;
        if let Some(tensors) = self.cpu_tensors.get(&key) {
            return Ok(tensors.clone());
        }

        let tensors = Arc::new(crate::encoders::park::load_tensors_to_cpu(paths)?);
        self.cpu_tensors.insert(key, tensors.clone());
        Ok(tensors)
    }
}

fn cpu_tensor_cache_key(paths: &[impl AsRef<Path>]) -> anyhow::Result<String> {
    let mut parts = Vec::with_capacity(paths.len());
    for path in paths {
        let path = path.as_ref();
        let canonical = path.canonicalize()?;
        let metadata = std::fs::metadata(&canonical)?;
        let modified = metadata
            .modified()?
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_nanos();
        parts.push(format!(
            "{}:{}:{}",
            canonical.display(),
            metadata.len(),
            modified
        ));
    }
    Ok(parts.join("|"))
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
}
