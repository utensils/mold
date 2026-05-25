use std::collections::HashMap;
use std::path::Path;
use std::sync::Arc;
use tokenizers::Tokenizer;

/// Cross-engine cache for tokenizers (and potentially prompt embeddings in the future).
/// Tokenizers are keyed by their canonical file path. Thread-safe when wrapped in `Arc<Mutex<>>`.
#[derive(Default)]
pub struct SharedPool {
    tokenizers: HashMap<String, Arc<Tokenizer>>,
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
}

#[cfg(test)]
mod tests {
    use super::*;
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
}
