use mold_catalog::entry::{
    Bundling, CatalogEntry, CatalogId, DownloadRecipe, FamilyRole, FileFormat, Kind, LicenseFlags,
    Modality, RecipeFile, Source, TokenKind,
};
use mold_catalog::families::Family;

fn sample_entry() -> CatalogEntry {
    CatalogEntry {
        id: CatalogId::from("hf:black-forest-labs/FLUX.1-dev"),
        source: Source::Hf,
        source_id: "black-forest-labs/FLUX.1-dev".into(),
        name: "FLUX.1-dev".into(),
        author: Some("black-forest-labs".into()),
        family: Family::Flux,
        family_role: FamilyRole::Foundation,
        sub_family: None,
        modality: Modality::Image,
        kind: Kind::Checkpoint,
        file_format: FileFormat::Safetensors,
        bundling: Bundling::Separated,
        size_bytes: Some(23_800_000_000),
        download_count: 5_400_000,
        rating: None,
        likes: 12_300,
        nsfw: false,
        thumbnail_url: Some("https://huggingface.co/.../thumb.png".into()),
        description: Some("Flux.1-dev base.".into()),
        license: Some("flux-1-dev-non-commercial-license".into()),
        license_flags: LicenseFlags {
            commercial: Some(false),
            derivatives: Some(true),
            different_license: Some(false),
        },
        tags: vec!["text-to-image".into(), "flux".into()],
        companions: vec!["t5-v1_1-xxl".into(), "clip-l".into(), "flux-vae".into()],
        download_recipe: DownloadRecipe {
            files: vec![RecipeFile {
                url: "https://huggingface.co/.../flux1-dev.safetensors".into(),
                dest: "{family}/{author}/{name}.safetensors".into(),
                sha256: Some("deadbeef".into()),
                size_bytes: Some(23_800_000_000),
                role: None,
            }],
            needs_token: Some(TokenKind::Hf),
        },
        supported: true,
        created_at: Some(1_700_000_000),
        updated_at: Some(1_710_000_000),
        added_at: 1_720_000_000,
        trained_words: vec!["cinematic".into(), "studio lighting".into()],
        page_url: Some("https://huggingface.co/black-forest-labs/FLUX.1-dev".into()),
    }
}

#[test]
fn catalog_entry_round_trips() {
    let entry = sample_entry();
    let s = serde_json::to_string_pretty(&entry).unwrap();
    let back: CatalogEntry = serde_json::from_str(&s).unwrap();
    assert_eq!(entry, back);
}

#[test]
fn entry_without_page_url_still_deserializes() {
    // Back-compat: shards and cached wire JSON written before `page_url`
    // existed must keep deserializing (same mechanism as `trained_words`).
    let mut value = serde_json::to_value(sample_entry()).unwrap();
    let obj = value.as_object_mut().unwrap();
    assert!(obj.remove("page_url").is_some(), "field must serialize");
    let back: CatalogEntry = serde_json::from_value(value).unwrap();
    assert_eq!(back.page_url, None);
    assert_eq!(back.id.as_str(), "hf:black-forest-labs/FLUX.1-dev");
}

#[test]
fn enum_serializations_use_kebab_case() {
    assert_eq!(serde_json::to_string(&Source::Hf).unwrap(), "\"hf\"");
    assert_eq!(
        serde_json::to_string(&Source::Civitai).unwrap(),
        "\"civitai\""
    );
    assert_eq!(
        serde_json::to_string(&FamilyRole::Foundation).unwrap(),
        "\"foundation\""
    );
    assert_eq!(
        serde_json::to_string(&Bundling::SingleFile).unwrap(),
        "\"single-file\""
    );
    assert_eq!(
        serde_json::to_string(&FileFormat::Safetensors).unwrap(),
        "\"safetensors\""
    );
    assert_eq!(
        serde_json::to_string(&Kind::TextEncoder).unwrap(),
        "\"text-encoder\""
    );
}

#[test]
fn token_kind_round_trips() {
    let none: Option<TokenKind> = None;
    assert_eq!(serde_json::to_string(&none).unwrap(), "null");
    let hf: Option<TokenKind> = Some(TokenKind::Hf);
    assert_eq!(serde_json::to_string(&hf).unwrap(), "\"hf\"");
    let cv: Option<TokenKind> = Some(TokenKind::Civitai);
    assert_eq!(serde_json::to_string(&cv).unwrap(), "\"civitai\"");
}
