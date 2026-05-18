use mold_catalog::entry::{
    Bundling, CatalogEntry, FamilyRole, FileFormat, Kind, Modality, RecipeFileRole, Source,
};
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
fn civitai_zimage_multifile_version_picks_model_file_not_text_encoder() {
    // cv:2442439 lists Text Encoder, VAE, then Model. The primary recipe
    // must choose the Model file; otherwise mold tries to load Qwen3 text
    // encoder weights as the Z-Image transformer and fails on t_embedder.
    let json = r#"{
        "id": 2442000,
        "name": "Z Image Turbo",
        "type": "Checkpoint",
        "nsfw": false,
        "creator": { "username": "z" },
        "stats": { "downloadCount": 10, "favoriteCount": 0 },
        "tags": [],
        "modelVersions": [{
            "id": 2442439,
            "name": "Turbo",
            "baseModel": "ZImageTurbo",
            "baseModelType": "Standard",
            "files": [
                { "id": 1, "name": "zImageTurbo_turbo_txt.safetensors",
                  "type": "Text Encoder", "sizeKB": 7856427.78125, "downloadCount": 0,
                  "metadata": { "format": "SafeTensor" },
                  "downloadUrl": "https://civitai.example/txt", "hashes": {} },
                { "id": 2, "name": "ae_zimgturbo.safetensors",
                  "type": "VAE", "sizeKB": 327445.69140625, "downloadCount": 0,
                  "metadata": { "format": "SafeTensor", "fp": "bf16" },
                  "downloadUrl": "https://civitai.example/vae", "hashes": {} },
                { "id": 3, "name": "zImageTurbo_turbo.safetensors",
                  "type": "Model", "sizeKB": 12021353.90625, "downloadCount": 0,
                  "metadata": { "format": "SafeTensor", "fp": "bf16", "size": "full" },
                  "downloadUrl": "https://civitai.example/model", "hashes": { "SHA256": "ABC" } }
            ],
            "images": []
        }]
    }"#;
    let item: CivitaiItem = serde_json::from_str(json).unwrap();
    let entry = from_civitai(item).expect("mapped");

    assert_eq!(entry.family, Family::ZImage);
    assert_eq!(entry.download_recipe.files.len(), 2);
    assert_eq!(
        entry.download_recipe.files[0].url,
        "https://civitai.example/model"
    );
    assert_eq!(
        entry.download_recipe.files[0].dest,
        "{family}/civitai/2442439/zImageTurbo_turbo.safetensors"
    );
    assert_eq!(
        entry.download_recipe.files[0].sha256.as_deref(),
        Some("ABC")
    );
    assert_eq!(
        entry.download_recipe.files[1].url,
        "https://civitai.example/txt"
    );
    assert_eq!(
        entry.download_recipe.files[1].dest,
        "{family}/civitai/2442439/zImageTurbo_turbo_txt.safetensors"
    );
    assert_eq!(
        entry.download_recipe.files[1].role,
        Some(RecipeFileRole::TextEncoder)
    );
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

fn civitai_lora_json(model_type: &str, base_model: &str) -> String {
    format!(
        r#"{{
            "id": 42,
            "name": "Cinematic Lora",
            "type": "{model_type}",
            "nsfw": false,
            "creator": {{ "username": "alice" }},
            "stats": {{ "downloadCount": 500, "favoriteCount": 8 }},
            "tags": [],
            "modelVersions": [{{
                "id": 99,
                "name": "v1",
                "baseModel": "{base_model}",
                "baseModelType": "Standard",
                "files": [{{ "id": 7, "name": "cinematic-lora.safetensors",
                           "sizeKB": 150000, "downloadCount": 1,
                           "metadata": {{ "format": "SafeTensor" }},
                           "downloadUrl": "u", "hashes": {{}} }}],
                "images": []
            }}]
        }}"#
    )
}

#[test]
fn civitai_lora_classifies_as_kind_lora() {
    let item: CivitaiItem =
        serde_json::from_str(&civitai_lora_json("LORA", "Flux.1 D")).expect("parse");
    let entry = from_civitai(item).expect("LORA must round-trip");
    assert_eq!(entry.kind, Kind::Lora);
    // LoRAs are self-contained — no T5/CLIP-L/flux-vae companions or the
    // installer wastes ~12 GB and the install-status check fails forever.
    assert!(
        entry.companions.is_empty(),
        "LoRA must have no companions: {:?}",
        entry.companions
    );
}

#[test]
fn civitai_locon_grouped_with_lora() {
    // LoCon adapters share the LoRA inference path (low-rank merge into
    // base weights). Catalog them as Kind::Lora so the FLUX backend can
    // load them via the same code path.
    let item: CivitaiItem =
        serde_json::from_str(&civitai_lora_json("LoCon", "Flux.1 D")).expect("parse");
    let entry = from_civitai(item).expect("LoCon must round-trip");
    assert_eq!(entry.kind, Kind::Lora);
}

#[test]
fn civitai_textual_inversion_is_dropped() {
    // The catalog has no slot for TI embeddings; misclassifying as
    // Checkpoint would mislead users into trying to generate from them.
    let item: CivitaiItem =
        serde_json::from_str(&civitai_lora_json("TextualInversion", "SDXL 1.0")).expect("parse");
    assert!(from_civitai(item).is_none());
}

#[test]
fn civitai_checkpoint_still_classifies_correctly() {
    // Regression: changing the kind classifier must not flip checkpoints
    // the catalog already serves.
    let item: CivitaiItem = serde_json::from_str(&load("civitai_juggernaut.json")).unwrap();
    let entry = from_civitai(item).expect("mapped");
    assert_eq!(entry.kind, Kind::Checkpoint);
}

fn hf_detail_with_tags(id: &str, tags: Vec<&str>) -> HfDetail {
    serde_json::from_value(serde_json::json!({
        "id": id,
        "author": id.split('/').next().unwrap_or("anon"),
        "downloads": 100,
        "likes": 5,
        "tags": tags,
        "pipeline_tag": null,
        "library_name": null,
        "createdAt": null,
        "lastModified": null,
        "cardData": null,
    }))
    .expect("synthesize HfDetail")
}

fn single_file_tree() -> Vec<HfTreeEntry> {
    serde_json::from_value(serde_json::json!([
        { "type": "file", "path": "lora.safetensors", "size": 150_000_000 }
    ]))
    .expect("synthesize tree")
}

#[test]
fn hf_lora_tag_is_classified_as_lora() {
    // The most reliable HF signal: curators tag LoRA repos with "lora".
    let detail = hf_detail_with_tags("ostris/face-helper-sdxl-lora", vec!["lora", "diffusers"]);
    let entry = from_hf(
        detail,
        single_file_tree(),
        Family::Sdxl,
        FamilyRole::Finetune,
    )
    .expect("normalizes");
    assert_eq!(entry.kind, Kind::Lora);
    assert!(entry.companions.is_empty(), "LoRA pulls no companions");
}

#[test]
fn hf_id_substring_classifies_as_lora() {
    // Repo-id substring is the fallback when the curator forgot tags.
    let detail = hf_detail_with_tags("ntc-ai/SDXL-LoRA-slider.cinematic-lighting", vec![]);
    let entry = from_hf(
        detail,
        single_file_tree(),
        Family::Sdxl,
        FamilyRole::Finetune,
    )
    .expect("normalizes");
    assert_eq!(entry.kind, Kind::Lora);
}

#[test]
fn hf_filename_signal_classifies_as_lora() {
    let detail = hf_detail_with_tags("anon/sdxl-fancy-style", vec![]);
    let tree: Vec<HfTreeEntry> = serde_json::from_value(serde_json::json!([
        { "type": "file", "path": "pytorch_lora_weights.safetensors", "size": 80_000_000 }
    ]))
    .unwrap();
    let entry = from_hf(detail, tree, Family::Sdxl, FamilyRole::Finetune).expect("normalizes");
    assert_eq!(entry.kind, Kind::Lora);
}

#[test]
fn hf_plain_checkpoint_stays_checkpoint() {
    // Regression: the heuristic must not over-match. A boring repo without
    // any LoRA signal stays a Checkpoint.
    let detail = hf_detail_with_tags("black-forest-labs/FLUX.1-dev", vec!["diffusers", "flux"]);
    let entry = from_hf(
        detail,
        single_file_tree(),
        Family::Flux,
        FamilyRole::Foundation,
    )
    .expect("normalizes");
    assert_eq!(entry.kind, Kind::Checkpoint);
}

#[test]
fn civitai_lora_carries_trained_words_through() {
    // The web UI surfaces trigger phrases as click-to-insert chips. The
    // catalog has to capture them at scrape time — Civitai's
    // `trainedWords` lives on the model version, not the model itself.
    let json = r#"{
        "id": 1,
        "name": "Trigger LoRA",
        "type": "LORA",
        "nsfw": false,
        "creator": { "username": "alice" },
        "stats": { "downloadCount": 10, "favoriteCount": 0 },
        "tags": [],
        "modelVersions": [{
            "id": 2,
            "name": "v1",
            "baseModel": "Flux.1 D",
            "baseModelType": "Standard",
            "trainedWords": ["cinematic style", "dramatic lighting"],
            "files": [{ "id": 3, "name": "x.safetensors", "sizeKB": 1, "downloadCount": 0,
                       "metadata": { "format": "SafeTensor" }, "downloadUrl": "u", "hashes": {} }],
            "images": []
        }]
    }"#;
    let item: CivitaiItem = serde_json::from_str(json).unwrap();
    let entry = from_civitai(item).expect("LORA must round-trip");
    assert_eq!(entry.kind, Kind::Lora);
    assert_eq!(
        entry.trained_words,
        vec!["cinematic style".to_string(), "dramatic lighting".into()]
    );
}

#[test]
fn civitai_checkpoint_with_no_trigger_words_yields_empty() {
    // Regression: the field must default to empty (not error) when the
    // upstream API omits or empties `trainedWords`.
    let item: CivitaiItem = serde_json::from_str(&load("civitai_juggernaut.json")).unwrap();
    let entry = from_civitai(item).expect("mapped");
    assert!(entry.trained_words.is_empty());
}
