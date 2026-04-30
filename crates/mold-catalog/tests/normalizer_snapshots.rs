use mold_catalog::entry::{Bundling, CatalogEntry, FamilyRole, FileFormat, Kind, Modality, Source};
use mold_catalog::families::Family;
use mold_catalog::normalizer::{from_civitai, from_hf, CivitaiItem, HfDetail, HfTreeEntry};

fn load(path: &str) -> String {
    std::fs::read_to_string(format!("tests/fixtures/{path}")).unwrap()
}

#[test]
fn hf_flux_dev_normalizes() {
    let detail: HfDetail = serde_json::from_str(&load("hf_flux_dev.json")).unwrap();
    let tree: Vec<HfTreeEntry> = serde_json::from_str(&load("hf_flux_dev_tree.json")).unwrap();
    let entry: CatalogEntry = from_hf(detail, tree, Family::Flux, FamilyRole::Foundation).unwrap();

    assert_eq!(entry.id.as_str(), "hf:black-forest-labs/FLUX.1-dev");
    assert_eq!(entry.source, Source::Hf);
    assert_eq!(entry.author.as_deref(), Some("black-forest-labs"));
    assert_eq!(entry.family, Family::Flux);
    assert_eq!(entry.family_role, FamilyRole::Foundation);
    assert_eq!(entry.modality, Modality::Image);
    assert_eq!(entry.kind, Kind::Checkpoint);
    assert_eq!(entry.file_format, FileFormat::Safetensors);
    // Diffusers layout (presence of model_index.json + text_encoder/ etc.)
    // → Bundling::Separated.
    assert_eq!(entry.bundling, Bundling::Separated);
    assert_eq!(entry.engine_phase, 1);
    assert_eq!(entry.likes, 12_300);
    assert!(!entry.nsfw);
    assert!(!entry.download_recipe.files.is_empty());
}

#[test]
fn civitai_juggernaut_normalizes_as_sdxl_single_file() {
    let item: CivitaiItem = serde_json::from_str(&load("civitai_juggernaut.json")).unwrap();
    let entry = from_civitai(item).expect("mapped");

    assert_eq!(entry.source, Source::Civitai);
    assert_eq!(entry.id.as_str(), "cv:618692");
    assert_eq!(entry.family, Family::Sdxl);
    assert_eq!(entry.bundling, Bundling::SingleFile);
    assert_eq!(entry.file_format, FileFormat::Safetensors);
    assert_eq!(entry.engine_phase, 2); // SDXL single-file → phase 2
                                       // Companions are required for single-file SDXL.
    assert!(entry.companions.contains(&"clip-l".to_string()));
    assert!(entry.companions.contains(&"clip-g".to_string()));
    assert!(entry.companions.contains(&"sdxl-vae".to_string()));
    assert!(!entry.download_recipe.files.is_empty());
    assert_eq!(
        entry.download_recipe.files[0].url,
        "https://civitai.com/api/download/models/618692"
    );
    assert_eq!(
        entry.download_recipe.files[0].sha256.as_deref(),
        Some("ABC123")
    );
    assert!(entry.thumbnail_url.as_deref().is_some());
}

#[test]
fn civitai_unknown_base_model_is_dropped() {
    let json = r#"{
        "id": 1,
        "name": "Random",
        "type": "Checkpoint",
        "nsfw": false,
        "creator": { "username": "x" },
        "stats": { "downloadCount": 100, "favoriteCount": 0 },
        "tags": [],
        "modelVersions": [{
            "id": 2,
            "name": "v1",
            "baseModel": "Some Future Model",
            "baseModelType": "Standard",
            "files": [{ "id": 3, "name": "x.safetensors", "sizeKB": 1, "downloadCount": 0,
                       "metadata": { "format": "SafeTensor" }, "downloadUrl": "u", "hashes": {} }],
            "images": []
        }]
    }"#;
    let item: CivitaiItem = serde_json::from_str(json).unwrap();
    assert!(from_civitai(item).is_none());
}
