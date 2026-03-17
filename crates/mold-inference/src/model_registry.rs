use mold_core::ModelInfo;
use mold_core::manifest::known_manifests;

/// Returns the list of known FLUX model variants with their HuggingFace repos.
/// Delegates to the canonical manifest in `mold_core::manifest`.
pub fn known_models() -> Vec<ModelInfo> {
    known_manifests()
        .into_iter()
        .map(|m| {
            let hf_repo = m
                .files
                .iter()
                .find(|f| f.component == mold_core::manifest::ModelComponent::Transformer)
                .map(|f| f.hf_repo.clone())
                .unwrap_or_default();

            ModelInfo {
                name: m.name,
                family: m.family,
                size_gb: m.size_gb,
                is_loaded: false,
                last_used: None,
                hf_repo,
            }
        })
        .collect()
}
