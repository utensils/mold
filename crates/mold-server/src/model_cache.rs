use std::collections::{HashMap, HashSet};
use std::time::{Duration, Instant};

use mold_inference::InferenceEngine;

use crate::state::EngineSnapshot;

/// Where a cached model's weights currently reside.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ModelResidency {
    /// Fully loaded on GPU, ready for immediate inference.
    Gpu,
    /// Engine exists but weights are not on GPU. Retains paths, config,
    /// tokenizers, and CPU-side caches (e.g. FLUX `lora_delta_cache`,
    /// shared tokenizers) so the next reload can skip recomputation. The
    /// engine struct can be reloaded in place without recreation.
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
/// - `max_cached` limits total entries across both residency states (Gpu and Parked).
/// - `in_flight` holds names temporarily checked out via `take()` until they
///   are returned via `restore()`. Such names are physically absent from
///   `entries`/`lru_order` but logically still part of the cache for the
///   purposes of `contains()` and `snapshot()` — without this, a parallel
///   reader during a take window would mistakenly conclude the model is
///   uncached.
pub struct ModelCache {
    entries: HashMap<String, CachedEngine>,
    /// Ordered from least-recently-used (index 0) to most-recently-used (last).
    lru_order: Vec<String>,
    /// Maximum number of models to keep cached (loaded + unloaded).
    max_cached: usize,
    /// Names checked out via `take()` and not yet `restore()`d. Treated as
    /// logically cached by `contains()`, `snapshot()`, and
    /// `cached_model_names()`. The single-GPU contract guarantees that at
    /// most one of these was the GPU-resident engine at the time of the
    /// take.
    in_flight: HashSet<String>,
    /// Name of the in-flight take that was the GPU-resident engine at the
    /// time of `take()`, if any. Surfaced as `snapshot().model_name` so
    /// readers can still see "this model is the active one" during the
    /// take/restore window.
    in_flight_active: Option<String>,
}

impl ModelCache {
    pub fn new(max_cached: usize) -> Self {
        Self {
            entries: HashMap::new(),
            lru_order: Vec::new(),
            max_cached: max_cached.max(1),
            in_flight: HashSet::new(),
            in_flight_active: None,
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
                // Engine struct exists but weights are off-GPU.
                ModelResidency::Parked
            },
            last_used: Instant::now(),
            vram_bytes,
            engine,
        };

        self.entries.insert(name.clone(), entry);
        // The freshly inserted entry supersedes any take-window placeholder
        // for this name (used by the reload path: take → load → insert).
        self.in_flight.remove(&name);
        if self.in_flight_active.as_deref() == Some(name.as_str()) {
            self.in_flight_active = None;
        }
        self.touch_order(&name);
        self.report_size();
        self.debug_check_invariants();
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
    /// While the engine is checked out, the name remains in `in_flight` so
    /// parallel readers (`contains`, `snapshot`, `cached_model_names`) still
    /// see the model as logically cached.
    pub fn take(&mut self, model_name: &str) -> Option<CachedEngine> {
        self.lru_order.retain(|n| n != model_name);
        let taken = self.entries.remove(model_name);
        if let Some(ref entry) = taken {
            self.in_flight.insert(model_name.to_string());
            if entry.residency == ModelResidency::Gpu {
                self.in_flight_active = Some(model_name.to_string());
            }
            self.report_size();
        }
        self.debug_check_invariants();
        taken
    }

    /// Re-insert a taken engine after inference completes.
    pub fn restore(&mut self, cached: CachedEngine) {
        let name = cached.model_name.clone();
        self.in_flight.remove(&name);
        if self.in_flight_active.as_deref() == Some(name.as_str()) {
            self.in_flight_active = None;
        }
        self.lru_order.push(name.clone());
        self.entries.insert(name, cached);
        self.report_size();
        self.debug_check_invariants();
    }

    /// Clear an in-flight marker without restoring an engine. Called when the
    /// take-and-restore window terminates abnormally (`JoinError`, panic that
    /// escapes `catch_unwind`, etc.) and we have no engine to put back —
    /// otherwise the name leaks into `in_flight` forever, making
    /// `contains()` permanently lie to `ensure_model_ready` while
    /// `take()`/`get()` keep returning `None`.
    pub fn clear_in_flight(&mut self, model_name: &str) {
        self.in_flight.remove(model_name);
        if self.in_flight_active.as_deref() == Some(model_name) {
            self.in_flight_active = None;
        }
        self.debug_check_invariants();
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
                // Engine struct exists but weights are off-GPU.
                ModelResidency::Parked
            },
            last_used: Instant::now(),
            vram_bytes,
            engine,
        };

        self.entries.insert(model_name.clone(), entry);
        // Freshly inserted entry supersedes any in-flight placeholder.
        self.in_flight.remove(&model_name);
        if self.in_flight_active.as_deref() == Some(model_name.as_str()) {
            self.in_flight_active = None;
        }
        self.touch_order(&model_name);
        self.report_size();
        self.debug_check_invariants();
        evicted
    }

    /// Check if a model is in the cache. Treats names taken-but-not-restored
    /// as still cached so concurrent readers don't see a transient hole
    /// during a take/restore window.
    pub fn contains(&self, model_name: &str) -> bool {
        self.entries.contains_key(model_name) || self.in_flight.contains(model_name)
    }

    /// Remove a model from the cache entirely, returning its engine. Also
    /// clears the name from `in_flight` so we never claim a model is cached
    /// after explicit removal.
    pub fn remove(&mut self, model_name: &str) -> Option<Box<dyn InferenceEngine>> {
        self.lru_order.retain(|n| n != model_name);
        self.in_flight.remove(model_name);
        if self.in_flight_active.as_deref() == Some(model_name) {
            self.in_flight_active = None;
        }
        let removed = self.entries.remove(model_name).map(|e| e.engine);
        if removed.is_some() {
            self.report_size();
        }
        self.debug_check_invariants();
        removed
    }

    /// Unload all models from GPU. Returns names of models that were unloaded.
    /// Each is transitioned to `Parked` (retain tokenizers/caches for faster
    /// reload).
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
        self.debug_check_invariants();
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
        self.debug_check_invariants();
        active_name
    }

    /// Drop all entries, returning all engines for cleanup. Also clears
    /// `in_flight` — any caller still holding a checked-out engine must
    /// drop it on their own (they own that `CachedEngine`); we just stop
    /// claiming it's logically present.
    pub fn clear(&mut self) -> Vec<Box<dyn InferenceEngine>> {
        self.lru_order.clear();
        self.in_flight.clear();
        self.in_flight_active = None;
        let drained: Vec<_> = self.entries.drain().map(|(_, e)| e.engine).collect();
        self.report_size();
        self.debug_check_invariants();
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

    /// All cached model names (any residency, including names temporarily
    /// taken-out for in-flight inference).
    pub fn cached_model_names(&self) -> Vec<String> {
        let mut names = self.lru_order.clone();
        for name in &self.in_flight {
            if !names.iter().any(|n| n == name) {
                names.push(name.clone());
            }
        }
        names
    }

    /// Snapshot of the cache's current state — what /api/models and
    /// /api/status report. Derived directly from the cache so there's no
    /// parallel field that can drift. During a take/restore window the
    /// engine that was GPU-resident is reflected via `in_flight_active`
    /// so readers don't see a transient "no model loaded" hole.
    pub fn snapshot(&self) -> EngineSnapshot {
        let active = self
            .active_model()
            .map(|s| s.to_string())
            .or_else(|| self.in_flight_active.clone());
        let is_loaded = active.is_some();
        EngineSnapshot {
            model_name: active,
            is_loaded,
            cached_models: self.cached_model_names(),
        }
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
        let mut evicted = None;
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
                evicted = Some(entry.engine);
            }
        }
        self.debug_check_invariants();
        evicted
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
        self.debug_check_invariants();
        out
    }

    /// Evict the LRU entry that is *not* GPU-resident and (optionally) not the
    /// named model. Returns `(name, engine)` so the caller can drop the engine
    /// outside the cache lock and then issue any GPU reclamation.
    ///
    /// Used by the load-time evict-to-fit recovery path: when a fresh load's
    /// preflight fails, we shrink the parked working set one entry at a time
    /// and retry. The `skip` parameter exists because the parked-reload branch
    /// must not evict the very entry it's about to reload.
    pub fn evict_lru_parked_except(
        &mut self,
        skip: Option<&str>,
    ) -> Option<(String, Box<dyn InferenceEngine>)> {
        let victim = self.lru_order.iter().find(|name| {
            if Some(name.as_str()) == skip {
                return false;
            }
            self.entries
                .get(name.as_str())
                .map(|e| e.residency != ModelResidency::Gpu)
                .unwrap_or(false)
        })?;
        let name = victim.clone();
        self.lru_order.retain(|n| n != &name);
        let entry = self.entries.remove(&name)?;
        tracing::info!(
            model = %name,
            last_used_secs = entry.last_used.elapsed().as_secs(),
            reason = "evict-to-fit",
            "cache eviction"
        );
        #[cfg(feature = "metrics")]
        crate::metrics::record_cache_eviction("evict-to-fit");
        self.report_size();
        self.debug_check_invariants();
        Some((name, entry.engine))
    }

    /// Move a model name to the MRU position in the LRU order.
    fn touch_order(&mut self, model_name: &str) {
        self.lru_order.retain(|n| n != model_name);
        self.lru_order.push(model_name.to_string());
        self.debug_check_invariants();
    }

    /// Push the current entry count to the cache-size gauge. Cheap no-op
    /// when the metrics feature is off.
    fn report_size(&self) {
        #[cfg(feature = "metrics")]
        crate::metrics::set_cache_size(self.entries.len());
    }

    /// Debug-only invariant check. Called at the end of every state-mutating
    /// method so a violation surfaces in tests rather than as a silent
    /// "wrong model came back" bug in production.
    ///
    /// Invariants enforced:
    /// 1. At most one entry has `residency == Gpu` (single-GPU contract).
    /// 2. `entries.len() == lru_order.len()` (LRU mirrors entries 1:1).
    #[cfg(debug_assertions)]
    fn debug_check_invariants(&self) {
        let gpu_count = self
            .entries
            .values()
            .filter(|e| e.residency == ModelResidency::Gpu)
            .count();
        debug_assert!(
            gpu_count <= 1,
            "ModelCache invariant violated: {gpu_count} engines have residency=Gpu (must be ≤1)"
        );
        debug_assert_eq!(
            self.entries.len(),
            self.lru_order.len(),
            "ModelCache invariant violated: entries len ({}) != lru_order len ({})",
            self.entries.len(),
            self.lru_order.len()
        );
    }

    #[cfg(not(debug_assertions))]
    fn debug_check_invariants(&self) {}
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

        /// Returns an engine that reports `is_loaded() == false` so the
        /// cache classifies it as `Parked` on insert. Used by tests that
        /// exercise LRU machinery and don't care which entry is GPU-resident
        /// — letting both insert-as-loaded would violate the invariant
        /// "at most one entry has residency == Gpu".
        fn parked(name: &str) -> Self {
            Self {
                name: name.to_string(),
                loaded: false,
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
        // Only one engine may be GPU-resident at a time. Use parked() for
        // the others so the cache invariant holds.
        cache.insert(Box::new(MockEngine::parked("model-a")), 1000);
        cache.insert(Box::new(MockEngine::parked("model-b")), 1000);
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
        cache.insert(Box::new(MockEngine::parked("model-a")), 1000);
        cache.insert(Box::new(MockEngine::parked("model-b")), 1000);
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
        cache.insert(Box::new(MockEngine::parked("model-a")), 1000);
        cache.insert(Box::new(MockEngine::parked("model-b")), 1000);
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
        cache.insert(Box::new(MockEngine::parked("model-a")), 1000);
        cache.insert(Box::new(MockEngine::parked("model-b")), 1000);
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
        cache.insert(Box::new(MockEngine::parked("model-a")), 1000);
        cache.insert(Box::new(MockEngine::new("model-b")), 1000);
        // Re-insert model-a as parked (should replace, not trigger eviction
        // and not violate the at-most-one-Gpu invariant).
        let evicted = cache.insert(Box::new(MockEngine::parked("model-a")), 2000);
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
        // Real prod inserts a single Gpu-resident engine at a time. Park
        // model-a explicitly so we don't violate the at-most-one-Gpu
        // invariant when model-b is also inserted.
        cache.insert(Box::new(MockEngine::parked("model-a")), 100);
        cache.insert(Box::new(MockEngine::new("model-b")), 200);

        let unloaded = cache.unload_all();
        // Only model-b is on GPU; unload_all parks it.
        assert_eq!(unloaded, vec!["model-b".to_string()]);
        assert!(cache.active_model().is_none());
        // All entries still in cache
        assert_eq!(cache.len(), 2);
    }

    #[test]
    fn cached_model_names_reflects_lru_order() {
        let mut cache = ModelCache::new(3);
        cache.insert(Box::new(MockEngine::parked("model-a")), 100);
        cache.insert(Box::new(MockEngine::parked("model-b")), 200);
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

    /// `clear_in_flight` is the "abnormal exit" companion to `restore`:
    /// when a `JoinError` (or other unrecoverable failure) breaks the
    /// take-and-restore window, callers must call this so the name
    /// doesn't leak into `in_flight` forever. After `take()` removes the
    /// entry from `entries` and `clear_in_flight` clears the marker, the
    /// cache must look fully clean — no entry, no in-flight residue,
    /// `contains()` false.
    #[test]
    fn clear_in_flight_removes_marker_after_take() {
        let mut cache = ModelCache::new(3);
        cache.insert(Box::new(MockEngine::new("model-a")), 1000);

        // Simulate a `take` that won't be `restore`d (e.g. spawn_blocking
        // returned a JoinError).
        let _engine = cache.take("model-a").expect("entry present before take");
        // While in-flight, contains() lies to readers (intentional — the
        // engine logically still belongs to the cache during the window).
        assert!(cache.contains("model-a"));
        // But the entry itself is physically gone, so a second take returns None.
        assert!(cache.take("model-a").is_none());
        // in_flight_active is set because we took the GPU-resident engine.
        assert_eq!(cache.in_flight_active.as_deref(), Some("model-a"));

        // Now simulate the abnormal-exit cleanup: caller has no engine
        // to put back, but must clear the marker.
        cache.clear_in_flight("model-a");

        assert!(
            !cache.contains("model-a"),
            "after clear_in_flight, the cache must not claim the model is present"
        );
        assert!(
            cache.in_flight.is_empty(),
            "in_flight set must be empty after clearing the only marker"
        );
        assert!(
            cache.in_flight_active.is_none(),
            "in_flight_active must be cleared when its name is cleared"
        );
        assert_eq!(cache.len(), 0);
        assert!(cache.is_empty());
    }

    /// `clear_in_flight` on a name that was never in flight is a no-op.
    /// This matters because the abnormal-exit cleanup paths run
    /// unconditionally (in the `Err(JoinError)` arm of a match), even
    /// when the `take` succeeded but produced `None` — we never want
    /// the cleanup itself to panic or poison the cache.
    #[test]
    fn clear_in_flight_is_noop_for_unknown_name() {
        let mut cache = ModelCache::new(3);
        cache.insert(Box::new(MockEngine::new("model-a")), 1000);

        cache.clear_in_flight("never-taken");

        // model-a is still fully cached.
        assert!(cache.contains("model-a"));
        assert_eq!(cache.len(), 1);
        assert!(cache.in_flight.is_empty());
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
        cache.insert(Box::new(MockEngine::parked("old")), 100);
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
        // Insert `parked` as Parked and `gpu-active` as Gpu so the cache's
        // single-Gpu invariant holds without requiring a manual residency
        // override after the fact.
        cache.insert(Box::new(MockEngine::parked("parked")), 100);
        cache.insert(Box::new(MockEngine::new("gpu-active")), 100);
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
        cache.insert(Box::new(MockEngine::parked("a")), 100);
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

    /// `evict_lru_parked_except` returns the LRU parked entry. With three
    /// parked entries inserted in order a → b → c, the LRU is `a` and that
    /// must come back. The cache shrinks by one and `a` is no longer
    /// `contains()`.
    #[test]
    fn evict_lru_parked_returns_lru_parked_entry() {
        let mut cache = ModelCache::new(3);
        cache.insert(Box::new(MockEngine::parked("a")), 100);
        cache.insert(Box::new(MockEngine::parked("b")), 100);
        cache.insert(Box::new(MockEngine::parked("c")), 100);

        let evicted = cache.evict_lru_parked_except(None).expect("must evict");
        assert_eq!(evicted.0, "a");
        assert!(!cache.contains("a"));
        assert!(cache.contains("b"));
        assert!(cache.contains("c"));
        assert_eq!(cache.len(), 2);
    }

    /// The `skip` argument exists so the parked-reload branch doesn't evict
    /// the very engine it's about to reload. With LRU order [a, b, c] and
    /// skip=Some("a"), the eviction must pick `b` (next-LRU non-skipped) and
    /// leave `a` untouched.
    #[test]
    fn evict_lru_parked_respects_skip() {
        let mut cache = ModelCache::new(3);
        cache.insert(Box::new(MockEngine::parked("a")), 100);
        cache.insert(Box::new(MockEngine::parked("b")), 100);
        cache.insert(Box::new(MockEngine::parked("c")), 100);

        let evicted = cache
            .evict_lru_parked_except(Some("a"))
            .expect("must evict next-LRU when LRU is skipped");
        assert_eq!(evicted.0, "b", "skip(a) must skip a and pick the next LRU");
        assert!(cache.contains("a"));
        assert!(!cache.contains("b"));
        assert!(cache.contains("c"));
    }

    /// Never evict a GPU-resident entry — the eviction is a budget-recovery
    /// path before a *new* load, and the active engine is still in use until
    /// the explicit `unload_active()` step.
    #[test]
    fn evict_lru_parked_never_picks_gpu_resident() {
        let mut cache = ModelCache::new(3);
        // `gpu` is GPU-resident (loaded), `parked` is not. Even though `gpu`
        // is the LRU here, it must not be the victim.
        cache.insert(Box::new(MockEngine::new("gpu")), 100);
        cache.insert(Box::new(MockEngine::parked("parked")), 100);

        let evicted = cache.evict_lru_parked_except(None).expect("must evict");
        assert_eq!(
            evicted.0, "parked",
            "Gpu-resident entries must be left alone — only parked are eligible"
        );
        assert!(cache.contains("gpu"));
        assert!(!cache.contains("parked"));
    }

    /// When the only candidates are skipped or GPU-resident, return `None`.
    /// The caller (preflight loop in gpu_worker) treats `None` as "nothing
    /// left to surrender" and surfaces the original OOM error.
    #[test]
    fn evict_lru_parked_returns_none_when_only_skip_or_gpu_remain() {
        let mut cache = ModelCache::new(3);
        cache.insert(Box::new(MockEngine::new("gpu")), 100);
        cache.insert(Box::new(MockEngine::parked("a")), 100);

        // skip=Some("a") leaves only `gpu` (ineligible) and `a` (skipped).
        assert!(cache.evict_lru_parked_except(Some("a")).is_none());
        // Cache is unchanged.
        assert!(cache.contains("gpu"));
        assert!(cache.contains("a"));
        assert_eq!(cache.len(), 2);
    }

    /// The returned engine is the actual engine box — caller-owned and
    /// droppable outside the cache lock. Verify the box's model_name matches
    /// the evicted name so caller-side drop ordering can't latch onto the
    /// wrong engine.
    #[test]
    fn evict_lru_parked_returns_matching_engine_box() {
        let mut cache = ModelCache::new(3);
        cache.insert(Box::new(MockEngine::parked("alpha")), 100);
        cache.insert(Box::new(MockEngine::parked("beta")), 100);

        let (name, engine) = cache.evict_lru_parked_except(None).unwrap();
        assert_eq!(name, "alpha");
        assert_eq!(engine.model_name(), "alpha");
    }

    /// Run a battery of legal operations in sequence and confirm none of
    /// them tripped the debug invariant assertions in `debug_check_invariants`.
    /// We can't directly test the negative case (calling private internals
    /// to corrupt state) without `unsafe` access, so this is the strongest
    /// reachable form: every public mutator should preserve `entries.len() ==
    /// lru_order.len()` and `gpu_count <= 1` end-to-end.
    #[test]
    fn invariants_hold_through_legal_op_sequence() {
        let mut cache = ModelCache::new(3);

        // insert one loaded engine → Gpu
        cache.insert(Box::new(MockEngine::new("a")), 100);

        // unload_active before inserting another loaded engine — mirrors
        // production where the active model is parked before a new load.
        cache.unload_active();
        cache.insert(Box::new(MockEngine::new("b")), 100);

        cache.unload_active();
        cache.insert_loaded("c".to_string(), Box::new(MockEngine::new("c")), 100);

        // touch_order via get_mut
        let _ = cache.get_mut("a");

        // take + restore round-trip on a parked entry — restore re-inserts
        // it as it was (parked).
        let taken = cache.take("b").expect("b is present");
        cache.restore(taken);

        // unload_active turns Gpu → Parked
        cache.unload_active();

        // reinsert same model as parked (no-op for entries length, no Gpu
        // promotion).
        cache.insert(Box::new(MockEngine::parked("a")), 200);

        // unload_all parks every Gpu entry
        cache.unload_all();

        // remove drops one entry
        cache.remove("c");

        // capacity-driven eviction via insert when full
        cache.insert(Box::new(MockEngine::parked("d")), 100);
        cache.insert(Box::new(MockEngine::parked("e")), 100);
        // cache is at max_cached=3; one more push triggers eviction. The
        // invariant must hold across that path too.
        cache.insert(Box::new(MockEngine::parked("f")), 100);

        // evict_idle (after backdating) with all parked entries
        if cache.contains("d") {
            age_entry(&mut cache, "d", Duration::from_secs(120));
        }
        let _drop = cache.evict_idle(Duration::from_secs(60));

        // clear drains everything
        let _all = cache.clear();
        assert!(cache.is_empty());
    }
}
