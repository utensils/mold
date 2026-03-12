use mold_core::ModelInfo;

/// Returns the list of known FLUX model variants with their HuggingFace repos.
pub fn known_models() -> Vec<ModelInfo> {
    vec![
        ModelInfo {
            name: "flux-schnell".to_string(),
            family: "flux".to_string(),
            size_gb: 23.8,
            is_loaded: false,
            last_used: None,
            hf_repo: "black-forest-labs/FLUX.1-schnell".to_string(),
        },
        ModelInfo {
            name: "flux-dev".to_string(),
            family: "flux".to_string(),
            size_gb: 23.8,
            is_loaded: false,
            last_used: None,
            hf_repo: "black-forest-labs/FLUX.1-dev".to_string(),
        },
    ]
}

/// Look up a model by name.
pub fn find_model(name: &str) -> Option<ModelInfo> {
    known_models().into_iter().find(|m| m.name == name)
}
