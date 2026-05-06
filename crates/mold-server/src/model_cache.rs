use std::collections::HashMap;
use std::time::{Duration, Instant};

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
            evicted = self.evict_lru("capacity");
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
        self.report_size();
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
        let taken = self.entries.remove(model_name);
        if taken.is_some() {
            self.report_size();
        }
        taken
    }

    /// Re-insert a taken engine after inference completes.
    pub fn restore(&mut self, cached: CachedEngine) {
        let name = cached.model_name.clone();
        self.lru_order.push(name.clone());
        self.entries.insert(name, cached);
        self.report_size();
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
            evicted = self.evict_lru("capacity");
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
        self.report_size();
        evicted
    }

    /// Check if a model is in the cache.
    pub fn contains(&self, model_name: &str) -> bool {
        self.entries.contains_key(model_name)
    }

    /// Remove a model from the cache entirely, returning its engine.
    pub fn remove(&mut self, model_name: &str) -> Option<Box<dyn InferenceEngine>> {
        self.lru_order.retain(|n| n != model_name);
        let removed = self.entries.remove(model_name).map(|e| e.engine);
        if removed.is_some() {
            self.report_size();
        }
        removed
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
        let drained: Vec<_> = self.entries.drain().map(|(_, e)| e.engine).collect();
        self.report_size();
        drained
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
    /// `reason` is forwarded to the eviction log/metric (`"capacity"` from
    /// the insert paths, `"idle-ttl"` from the background sweeper).
    fn evict_lru(&mut self, reason: &'static str) -> Option<Box<dyn InferenceEngine>> {
        if let Some(name) = self.lru_order.first().cloned() {
            self.lru_order.remove(0);
            if let Some(entry) = self.entries.remove(&name) {
                let last_used_secs = entry.last_used.elapsed().as_secs();
                tracing::info!(
                    model = %name,
                    last_used_secs,
                    reason,
                    "cache eviction"
                );
                #[cfg(feature = "metrics")]
                crate::metrics::record_cache_eviction(reason);
                return Some(entry.engine);
            }
        }
        None
    }

    /// Reclaim cache entries whose `last_used` is older than `ttl`. Only
    /// entries that are not GPU-resident are eligible — the active model is
    /// always preserved. Skipped entirely when the cache holds at most one
    /// entry (so we never tear down the only warm engine after a quiet
    /// period).
    ///
    /// Returns evicted `(name, engine)` pairs so callers can drop the
    /// engines outside any cache mutex — `cuMemFree` and safetensor unmap on
    /// drop can block other cache users for non-trivial time.
    pub fn evict_idle(&mut self, ttl: Duration) -> Vec<(String, Box<dyn InferenceEngine>)> {
        if self.entries.len() <= 1 {
            return Vec::new();
        }
        let now = Instant::now();
        // Collect (name, last_used) for every stale, non-GPU entry. Sort by
        // `last_used` ascending (oldest first) so that when the "keep ≥1
        // warm engine" guard fires mid-loop we evict the LRU and the MRU
        // survives — without the sort, HashMap iteration order would pick
        // the survivor at random.
        let mut stale: Vec<(String, Instant)> = self
            .entries
            .iter()
            .filter_map(|(name, entry)| {
                if entry.residency == ModelResidency::Gpu {
                    return None;
                }
                let age = now.saturating_duration_since(entry.last_used);
                if age >= ttl {
                    Some((name.clone(), entry.last_used))
                } else {
                    None
                }
            })
            .collect();
        stale.sort_by_key(|(_, last_used)| *last_used);

        let mut out = Vec::with_capacity(stale.len());
        for (name, _) in stale {
            if self.entries.len() <= 1 {
                break;
            }
            self.lru_order.retain(|n| n != &name);
            if let Some(entry) = self.entries.remove(&name) {
                let last_used_secs = entry.last_used.elapsed().as_secs();
                tracing::info!(
                    model = %name,
                    last_used_secs,
                    reason = "idle-ttl",
                    "cache eviction"
                );
                #[cfg(feature = "metrics")]
                crate::metrics::record_cache_eviction("idle-ttl");
                out.push((name, entry.engine));
            }
        }
        if !out.is_empty() {
            self.report_size();
        }
        out
    }

    /// Move a model name to the MRU position in the LRU order.
    fn touch_order(&mut self, model_name: &str) {
        self.lru_order.retain(|n| n != model_name);
        self.lru_order.push(model_name.to_string());
    }

    /// Push the current entry count to the cache-size gauge. Cheap no-op
    /// when the metrics feature is off.
    fn report_size(&self) {
        #[cfg(feature = "metrics")]
        crate::metrics::set_cache_size(self.entries.len());
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

    /// Stronger LRU guarantee: the evicted engine returned by `insert` must
    /// be the LRU entry (model-a here), not any other entry. Callers in
    /// `model_manager` and `gpu_worker` drop this engine outside the cache
    /// lock — drop ordering depends on the right entry coming back.
    #[test]
    fn lru_eviction_returns_lru_engine() {
        let mut cache = ModelCache::new(2);
        cache.insert(Box::new(MockEngine::new("model-a")), 1000);
        cache.insert(Box::new(MockEngine::new("model-b")), 1000);
        let evicted = cache
            .insert(Box::new(MockEngine::new("model-c")), 1000)
            .expect("eviction must occur at capacity");
        assert_eq!(
            evicted.model_name(),
            "model-a",
            "evicted engine must be the LRU one (model-a), not any other"
        );
    }

    /// Same guarantee for `insert_loaded` (the GPU-worker path). When the
    /// cache is at capacity and a new load completes, the returned engine
    /// must be the LRU entry that was bumped — otherwise the dropped engine
    /// in `gpu_worker.rs` would be the wrong one and the cache would silently
    /// retain a stale entry.
    #[test]
    fn insert_loaded_returns_lru_engine_on_eviction() {
        let mut cache = ModelCache::new(2);
        cache.insert(Box::new(MockEngine::new("model-a")), 1000);
        cache.insert(Box::new(MockEngine::new("model-b")), 1000);
        let evicted = cache
            .insert_loaded(
                "model-c".to_string(),
                Box::new(MockEngine::new("model-c")),
                1000,
            )
            .expect("eviction must occur at capacity");
        assert_eq!(
            evicted.model_name(),
            "model-a",
            "insert_loaded must return the LRU engine on eviction"
        );
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

    /// Backdate an entry's `last_used` so we don't have to actually sleep.
    fn age_entry(cache: &mut ModelCache, name: &str, by: Duration) {
        let entry = cache
            .entries
            .get_mut(name)
            .expect("entry must exist for ageing");
        entry.last_used -= by;
    }

    #[test]
    fn evict_idle_drops_old_parked_entry_keeps_fresh_one() {
        let mut cache = ModelCache::new(3);
        cache.insert(Box::new(MockEngine::new("old")), 100);
        cache.insert(Box::new(MockEngine::new("fresh")), 100);
        // Park both so neither is GPU-resident.
        cache.unload_all();
        age_entry(&mut cache, "old", Duration::from_secs(120));

        let evicted = cache.evict_idle(Duration::from_secs(60));
        let names: Vec<&str> = evicted.iter().map(|(n, _)| n.as_str()).collect();
        assert_eq!(names, vec!["old"]);
        assert!(!cache.contains("old"));
        assert!(cache.contains("fresh"));
    }

    #[test]
    fn evict_idle_skips_when_only_one_entry() {
        let mut cache = ModelCache::new(3);
        cache.insert(Box::new(MockEngine::new("solo")), 100);
        cache.unload_all();
        age_entry(&mut cache, "solo", Duration::from_secs(3600));

        let evicted = cache.evict_idle(Duration::from_secs(60));
        assert!(
            evicted.is_empty(),
            "must keep at least one warm engine even past the TTL"
        );
        assert!(cache.contains("solo"));
    }

    #[test]
    fn evict_idle_never_evicts_gpu_resident_entry() {
        let mut cache = ModelCache::new(3);
        cache.insert(Box::new(MockEngine::new("gpu-active")), 100);
        cache.insert(Box::new(MockEngine::new("parked")), 100);
        // The most recent insert is the only Gpu-resident one (insert sets
        // residency from is_loaded(); MockEngine starts loaded but unload_all
        // parks everything). Park then re-mark `gpu-active` as on-GPU.
        cache.unload_all();
        cache.entries.get_mut("gpu-active").unwrap().residency = ModelResidency::Gpu;
        age_entry(&mut cache, "gpu-active", Duration::from_secs(3600));
        age_entry(&mut cache, "parked", Duration::from_secs(3600));

        let evicted = cache.evict_idle(Duration::from_secs(60));
        let names: Vec<&str> = evicted.iter().map(|(n, _)| n.as_str()).collect();
        assert_eq!(
            names,
            vec!["parked"],
            "Gpu-resident entries must be left alone regardless of age"
        );
        assert!(cache.contains("gpu-active"));
    }

    #[test]
    fn evict_idle_returns_engines_for_caller_drop() {
        let mut cache = ModelCache::new(3);
        cache.insert(Box::new(MockEngine::new("a")), 100);
        cache.insert(Box::new(MockEngine::new("b")), 100);
        cache.unload_all();
        // `a` is older than `b` — eviction-oldest-first should drop `a` and
        // leave the MRU (`b`) as the surviving warm engine when the
        // "≥ 1 warm entry" guard fires.
        age_entry(&mut cache, "a", Duration::from_secs(180));
        age_entry(&mut cache, "b", Duration::from_secs(120));

        let evicted = cache.evict_idle(Duration::from_secs(60));
        // Only one of the two is evicted — the "≥ 1 warm entry" guard kicks
        // in once the cache shrinks to a single entry. Determinism: the LRU
        // is dropped, the MRU survives.
        assert_eq!(evicted.len(), 1);
        let (evicted_name, engine) = evicted.into_iter().next().unwrap();
        assert_eq!(
            evicted_name, "a",
            "oldest-first sort must pick the LRU (`a`) for eviction"
        );
        assert_eq!(cache.len(), 1);
        assert!(cache.contains("b"), "MRU (`b`) must survive the guard");
        assert!(!cache.contains("a"), "LRU (`a`) must be gone");
        // Caller receives the engine box so it can drop outside the cache lock.
        drop(engine);
    }
}
