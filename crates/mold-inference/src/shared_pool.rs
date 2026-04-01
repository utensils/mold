use std::collections::HashMap;
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
}
