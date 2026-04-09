use std::collections::HashMap;
use std::time::Instant;

use mold_inference::InferenceEngine;

/// Where a cached model's weights currently reside.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ModelResidency {
    /// Fully loaded on GPU, ready for immediate inference.
    Gpu,
    /// Engine exists but weights are unloaded. Can reload without recreating
    /// the engine (retains paths, config, caches).
    Unloaded,
    /// Engine was actively unloaded from GPU but retains tokenizers, caches,
    /// and config in memory for faster reload compared to `Unloaded`.
    Parked,
}

/// A model entry in the cache.
pub struct CachedEngine {
    pub engine: Box<dyn InferenceEngine>,
    pub model_name: String,
    pub residency: ModelResidency,
    pub last_used: Instant,
    /// Measured VRAM footprint (bytes). Set after loading by measuring delta.
    pub vram_bytes: u64,
}

/// Multi-model cache with LRU eviction under VRAM pressure.
///
/// Invariants:
/// - At most one engine has `residency == Gpu` at a time (single-GPU inference).
/// - `lru_order` tracks all entries from least-recently-used (front) to
///   most-recently-used (back).
/// - `max_cached` limits total entries (Gpu, Unloaded, and Parked).
pub struct ModelCache {
    entries: HashMap<String, CachedEngine>,
    /// Ordered from least-recently-used (index 0) to most-recently-used (last).
    lru_order: Vec<String>,
    /// Maximum number of models to keep cached (loaded + unloaded).
    max_cached: usize,
}

impl ModelCache {
    pub fn new(max_cached: usize) -> Self {
        Self {
            entries: HashMap::new(),
            lru_order: Vec::new(),
            max_cached: max_cached.max(1),
        }
    }

    /// Insert an engine into the cache. If the cache is full, the LRU entry
    /// is dropped entirely. Returns the evicted engine (if any) for cleanup.
    pub fn insert(
        &mut self,
        engine: Box<dyn InferenceEngine>,
        vram_bytes: u64,
    ) -> Option<Box<dyn InferenceEngine>> {
        let name = engine.model_name().to_string();
        let mut evicted = None;

        // Evict LRU if at capacity (skip if the model is already in cache)
        if self.entries.len() >= self.max_cached && !self.entries.contains_key(&name) {
            evicted = self.evict_lru();
        }

        let entry = CachedEngine {
            model_name: name.clone(),
            residency: if engine.is_loaded() {
                ModelResidency::Gpu
            } else {
                ModelResidency::Unloaded
            },
            last_used: Instant::now(),
            vram_bytes,
            engine,
        };

        self.entries.insert(name.clone(), entry);
        self.touch_order(&name);
        evicted
    }

    /// Get a reference to a cached engine entry (does not update LRU order).
    pub fn get(&self, model_name: &str) -> Option<&CachedEngine> {
        self.entries.get(model_name)
    }

    /// Get a mutable reference to the engine for a model, if cached.
    pub fn get_mut(&mut self, model_name: &str) -> Option<&mut CachedEngine> {
        if self.entries.contains_key(model_name) {
            self.touch_order(model_name);
            self.entries.get_mut(model_name)
        } else {
            None
        }
    }

    /// Remove an engine from the cache, returning the full entry.
    /// Used by the take-and-restore pattern: remove before inference, re-insert after.
    pub fn take(&mut self, model_name: &str) -> Option<CachedEngine> {
        self.lru_order.retain(|n| n != model_name);
        self.entries.remove(model_name)
    }

    /// Re-insert a taken engine after inference completes.
    pub fn restore(&mut self, cached: CachedEngine) {
        let name = cached.model_name.clone();
        self.lru_order.push(name.clone());
        self.entries.insert(name, cached);
    }

    /// Insert a loaded engine with a known VRAM footprint.
    /// Unlike `insert()`, this takes a name separately from the engine.
    pub fn insert_loaded(
        &mut self,
        model_name: String,
        engine: Box<dyn InferenceEngine>,
        vram_bytes: u64,
    ) -> Option<Box<dyn InferenceEngine>> {
        let mut evicted = None;

        // Evict LRU if at capacity (skip if the model is already in cache)
        if self.entries.len() >= self.max_cached && !self.entries.contains_key(&model_name) {
            evicted = self.evict_lru();
        }

        let entry = CachedEngine {
            model_name: model_name.clone(),
            residency: if engine.is_loaded() {
                ModelResidency::Gpu
            } else {
                ModelResidency::Unloaded
            },
            last_used: Instant::now(),
            vram_bytes,
            engine,
        };

        self.entries.insert(model_name.clone(), entry);
        self.touch_order(&model_name);
        evicted
    }

    /// Check if a model is in the cache.
    pub fn contains(&self, model_name: &str) -> bool {
        self.entries.contains_key(model_name)
    }

    /// Remove a model from the cache entirely, returning its engine.
    pub fn remove(&mut self, model_name: &str) -> Option<Box<dyn InferenceEngine>> {
        self.lru_order.retain(|n| n != model_name);
        self.entries.remove(model_name).map(|e| e.engine)
    }

    /// Unload all models from GPU. Returns names of models that were unloaded.
    /// Unloaded models are parked (retain tokenizers/caches for faster reload).
    pub fn unload_all(&mut self) -> Vec<String> {
        let mut unloaded = Vec::new();
        for entry in self.entries.values_mut() {
            if entry.residency == ModelResidency::Gpu {
                entry.engine.unload();
                entry.residency = ModelResidency::Parked;
                entry.vram_bytes = 0;
                unloaded.push(entry.model_name.clone());
            }
        }
        unloaded
    }

    /// Unload the current GPU-resident model (if any) to make room for a new one.
    /// The engine is parked (retains tokenizers/caches) for faster reload.
    /// Returns the name of the unloaded model.
    pub fn unload_active(&mut self) -> Option<String> {
        let active_name = self
            .entries
            .values()
            .find(|e| e.residency == ModelResidency::Gpu)
            .map(|e| e.model_name.clone());

        if let Some(ref name) = active_name {
            if let Some(entry) = self.entries.get_mut(name) {
                entry.engine.unload();
                entry.residency = ModelResidency::Parked;
                entry.vram_bytes = 0;
            }
        }
        active_name
    }

    /// Drop all entries, returning all engines for cleanup.
    pub fn clear(&mut self) -> Vec<Box<dyn InferenceEngine>> {
        self.lru_order.clear();
        self.entries.drain().map(|(_, e)| e.engine).collect()
    }

    /// VRAM footprint of the currently GPU-resident model (0 if none loaded).
    pub fn active_vram_bytes(&self) -> u64 {
        self.entries
            .values()
            .find(|e| e.residency == ModelResidency::Gpu)
            .map(|e| e.vram_bytes)
            .unwrap_or(0)
    }

    /// The currently GPU-loaded model name.
    pub fn active_model(&self) -> Option<&str> {
        self.entries
            .values()
            .find(|e| e.residency == ModelResidency::Gpu)
            .map(|e| e.model_name.as_str())
    }

    /// All cached model names (any residency).
    pub fn cached_model_names(&self) -> Vec<String> {
        self.lru_order.clone()
    }

    /// Number of cached entries.
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Evict the least-recently-used entry, returning its engine for cleanup.
    fn evict_lru(&mut self) -> Option<Box<dyn InferenceEngine>> {
        // Find the first LRU entry that can be evicted
        if let Some(name) = self.lru_order.first().cloned() {
            self.lru_order.remove(0);
            return self.entries.remove(&name).map(|e| e.engine);
        }
        None
    }

    /// Move a model name to the MRU position in the LRU order.
    fn touch_order(&mut self, model_name: &str) {
        self.lru_order.retain(|n| n != model_name);
        self.lru_order.push(model_name.to_string());
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use anyhow::Result;
    use mold_core::GenerateRequest;

    struct MockEngine {
        name: String,
        loaded: bool,
    }

    impl MockEngine {
        fn new(name: &str) -> Self {
            Self {
                name: name.to_string(),
                loaded: true,
            }
        }
    }

    impl InferenceEngine for MockEngine {
        fn generate(&mut self, _req: &GenerateRequest) -> Result<mold_core::GenerateResponse> {
            unimplemented!()
        }
        fn model_name(&self) -> &str {
            &self.name
        }
        fn is_loaded(&self) -> bool {
            self.loaded
        }
        fn load(&mut self) -> Result<()> {
            self.loaded = true;
            Ok(())
        }
        fn unload(&mut self) {
            self.loaded = false;
        }
    }

    #[test]
    fn insert_and_get() {
        let mut cache = ModelCache::new(3);
        cache.insert(Box::new(MockEngine::new("model-a")), 1000);
        assert!(cache.contains("model-a"));
        assert_eq!(cache.len(), 1);
        assert_eq!(cache.active_model(), Some("model-a"));
    }

    #[test]
    fn lru_eviction() {
        let mut cache = ModelCache::new(2);
        cache.insert(Box::new(MockEngine::new("model-a")), 1000);
        cache.insert(Box::new(MockEngine::new("model-b")), 1000);
        // Cache full (2), inserting model-c should evict model-a (LRU)
        let evicted = cache.insert(Box::new(MockEngine::new("model-c")), 1000);
        assert!(evicted.is_some());
        assert!(!cache.contains("model-a"));
        assert!(cache.contains("model-b"));
        assert!(cache.contains("model-c"));
    }

    #[test]
    fn touch_updates_lru_order() {
        let mut cache = ModelCache::new(2);
        cache.insert(Box::new(MockEngine::new("model-a")), 1000);
        cache.insert(Box::new(MockEngine::new("model-b")), 1000);
        // Touch model-a (makes model-b the LRU)
        cache.get_mut("model-a");
        let evicted = cache.insert(Box::new(MockEngine::new("model-c")), 1000);
        assert!(evicted.is_some());
        assert!(cache.contains("model-a")); // was touched, survived
        assert!(!cache.contains("model-b")); // LRU, evicted
        assert!(cache.contains("model-c"));
    }

    #[test]
    fn unload_active() {
        let mut cache = ModelCache::new(3);
        cache.insert(Box::new(MockEngine::new("model-a")), 1000);
        assert_eq!(cache.active_model(), Some("model-a"));

        let unloaded = cache.unload_active();
        assert_eq!(unloaded.as_deref(), Some("model-a"));
        assert_eq!(cache.active_model(), None);
        // Still in cache, just unloaded
        assert!(cache.contains("model-a"));
        let entry = cache.get_mut("model-a").unwrap();
        assert_eq!(entry.residency, ModelResidency::Parked);
    }

    #[test]
    fn remove_model() {
        let mut cache = ModelCache::new(3);
        cache.insert(Box::new(MockEngine::new("model-a")), 1000);
        let removed = cache.remove("model-a");
        assert!(removed.is_some());
        assert!(!cache.contains("model-a"));
        assert_eq!(cache.len(), 0);
    }

    #[test]
    fn reinserting_same_model_does_not_evict() {
        let mut cache = ModelCache::new(2);
        cache.insert(Box::new(MockEngine::new("model-a")), 1000);
        cache.insert(Box::new(MockEngine::new("model-b")), 1000);
        // Re-insert model-a (should replace, not trigger eviction)
        let evicted = cache.insert(Box::new(MockEngine::new("model-a")), 2000);
        assert!(evicted.is_none());
        assert_eq!(cache.len(), 2);
    }

    #[test]
    fn is_empty_and_clear() {
        let mut cache = ModelCache::new(3);
        assert!(cache.is_empty());
        cache.insert(Box::new(MockEngine::new("model-a")), 100);
        assert!(!cache.is_empty());
        let cleared = cache.clear();
        assert_eq!(cleared.len(), 1);
        assert!(cache.is_empty());
        assert_eq!(cache.len(), 0);
    }

    #[test]
    fn unload_all_parks_all_models() {
        let mut cache = ModelCache::new(3);
        cache.insert(Box::new(MockEngine::new("model-a")), 100);
        cache.insert(Box::new(MockEngine::new("model-b")), 200);

        let unloaded = cache.unload_all();
        // Only model-b has Gpu residency (model-a was replaced when model-b was inserted
        // — actually both are "loaded" since MockEngine::new starts loaded).
        // unload_all should park everything that's on GPU.
        assert!(!unloaded.is_empty());
        assert!(cache.active_model().is_none());
        // All entries still in cache
        assert_eq!(cache.len(), 2);
    }

    #[test]
    fn cached_model_names_reflects_lru_order() {
        let mut cache = ModelCache::new(3);
        cache.insert(Box::new(MockEngine::new("model-a")), 100);
        cache.insert(Box::new(MockEngine::new("model-b")), 200);
        cache.insert(Box::new(MockEngine::new("model-c")), 300);
        // LRU order: a, b, c (a is oldest)
        assert_eq!(
            cache.cached_model_names(),
            vec!["model-a", "model-b", "model-c"]
        );
        // Touch model-a, making it MRU
        cache.get_mut("model-a");
        assert_eq!(
            cache.cached_model_names(),
            vec!["model-b", "model-c", "model-a"]
        );
    }

    #[test]
    fn get_mut_nonexistent_returns_none() {
        let mut cache = ModelCache::new(3);
        assert!(cache.get_mut("nonexistent").is_none());
    }

    #[test]
    fn remove_nonexistent_returns_none() {
        let mut cache = ModelCache::new(3);
        assert!(cache.remove("nonexistent").is_none());
    }

    #[test]
    fn unload_active_when_empty_returns_none() {
        let mut cache = ModelCache::new(3);
        assert!(cache.unload_active().is_none());
    }

    #[test]
    fn max_cached_clamped_to_at_least_one() {
        let mut cache = ModelCache::new(0);
        cache.insert(Box::new(MockEngine::new("model-a")), 100);
        assert_eq!(cache.len(), 1);
        // Should still allow at least 1 entry
        assert!(cache.contains("model-a"));
    }
}
