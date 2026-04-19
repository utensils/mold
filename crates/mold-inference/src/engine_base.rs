//! Shared base struct for inference engines.
//!
//! All engine families (FLUX, SD1.5, SDXL, SD3, Z-Image, Flux.2, Qwen-Image,
//! Wuerstchen) share the same set of bookkeeping fields: loaded state, model
//! name, paths, progress reporter, and load strategy. `EngineBase<L>` captures
//! these common fields so each engine can compose it instead of duplicating them.

use mold_core::ModelPaths;

use crate::engine::LoadStrategy;
use crate::progress::{ProgressCallback, ProgressReporter};

/// Common fields shared by all inference engine implementations.
///
/// `L` is the type of the loaded model state (e.g. `LoadedFlux`, `LoadedSD15`).
/// Each engine wraps `EngineBase<L>` and keeps engine-specific fields (caches,
/// config flags) alongside it.
pub(crate) struct EngineBase<L> {
    pub loaded: Option<L>,
    pub model_name: String,
    pub paths: ModelPaths,
    pub progress: ProgressReporter,
    pub load_strategy: LoadStrategy,
    /// GPU ordinal this engine is assigned to. Used by `create_device()` and VRAM queries.
    pub gpu_ordinal: usize,
}

impl<L> EngineBase<L> {
    /// Create a new engine base with no loaded state.
    pub fn new(
        model_name: String,
        paths: ModelPaths,
        load_strategy: LoadStrategy,
        gpu_ordinal: usize,
    ) -> Self {
        Self {
            loaded: None,
            model_name,
            paths,
            progress: ProgressReporter::default(),
            load_strategy,
            gpu_ordinal,
        }
    }

    /// Whether the engine has loaded model state, or operates in sequential
    /// (load-on-demand) mode.
    pub fn is_loaded(&self) -> bool {
        self.load_strategy == LoadStrategy::Sequential || self.loaded.is_some()
    }

    /// Return the model name.
    pub fn model_name(&self) -> &str {
        &self.model_name
    }

    /// Discard the loaded model state, freeing GPU memory.
    /// Callers are responsible for also clearing any engine-specific caches.
    pub fn unload(&mut self) {
        self.loaded = None;
    }

    /// Install a progress callback.
    pub fn set_on_progress(&mut self, callback: ProgressCallback) {
        self.progress.set_callback(callback);
    }

    /// Remove the progress callback.
    pub fn clear_on_progress(&mut self) {
        self.progress.clear_callback();
    }
}
