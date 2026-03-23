use mold_core::manifest::known_manifests;
use mold_core::ModelInfo;

/// Returns the list of known FLUX model variants with their HuggingFace repos.
/// Delegates to the canonical manifest in `mold_core::manifest`.
pub fn known_models() -> Vec<ModelInfo> {
    known_manifests()
        .iter()
        .map(|m| {
            let hf_repo = m
                .files
                .iter()
                .find(|f| f.component == mold_core::manifest::ModelComponent::Transformer)
                .map(|f| f.hf_repo.clone())
                .unwrap_or_default();

            ModelInfo {
                name: m.name.clone(),
                family: m.family.clone(),
                size_gb: m.model_size_gb(),
                is_loaded: false,
                last_used: None,
                hf_repo,
            }
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn known_models_report_model_size_only() {
        let model = known_models()
            .into_iter()
            .find(|m| m.name == "flux-schnell:q8")
            .expect("flux-schnell:q8 should be in the registry");
        let manifest = mold_core::manifest::find_manifest("flux-schnell:q8").unwrap();

        assert!((model.size_gb - manifest.model_size_gb()).abs() < 0.001);
        assert!(manifest.total_size_gb() > model.size_gb);
    }
}
