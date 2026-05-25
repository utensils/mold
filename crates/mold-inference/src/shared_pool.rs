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

    /// Load CPU tensors only when every component path is safetensors-backed.
    pub(crate) fn load_safetensors_cpu_tensors(
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

        self.load_cpu_tensors(paths).map(Some)
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
