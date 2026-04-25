use mold_catalog::entry::{
    Bundling, CatalogEntry, CatalogId, DownloadRecipe, FamilyRole, FileFormat, Kind, LicenseFlags,
    Modality, RecipeFile, Source,
};
use mold_catalog::families::Family;
use mold_catalog::filter::apply;
use mold_catalog::scanner::ScanOptions;

fn entry(source: Source, downloads: u64, nsfw: bool, with_files: bool) -> CatalogEntry {
    CatalogEntry {
        id: CatalogId::from(format!("{source:?}-{downloads}-{nsfw}-{with_files}")),
        source,
        source_id: "x".into(),
        name: "x".into(),
        author: None,
        family: Family::Flux,
        family_role: FamilyRole::Finetune,
        sub_family: None,
        modality: Modality::Image,
        kind: Kind::Checkpoint,
        file_format: FileFormat::Safetensors,
        bundling: Bundling::Separated,
        size_bytes: None,
        download_count: downloads,
        rating: None,
        likes: 0,
        nsfw,
        thumbnail_url: None,
        description: None,
        license: None,
        license_flags: LicenseFlags::default(),
        tags: vec![],
        companions: vec![],
        download_recipe: DownloadRecipe {
            files: if with_files {
                vec![RecipeFile {
                    url: "u".into(),
                    dest: "d".into(),
                    sha256: None,
                    size_bytes: None,
                }]
            } else {
                vec![]
            },
            needs_token: None,
        },
        engine_phase: 1,
        created_at: None,
        updated_at: None,
        added_at: 0,
    }
}

#[test]
fn drops_entries_with_no_recipe_files() {
    let entries = vec![
        entry(Source::Hf, 1000, false, true),
        entry(Source::Hf, 1000, false, false),
    ];
    let opts = ScanOptions::default();
    let kept = apply(entries, &opts);
    assert_eq!(kept.len(), 1);
}

#[test]
fn applies_min_downloads_to_civitai_only() {
    let entries = vec![
        entry(Source::Civitai, 50, false, true),
        entry(Source::Civitai, 200, false, true),
        // HF doesn't surface a per-repo download_count consistently —
        // don't penalize HF with the threshold.
        entry(Source::Hf, 0, false, true),
    ];
    let opts = ScanOptions {
        min_downloads: 100,
        ..ScanOptions::default()
    };
    let kept = apply(entries, &opts);
    assert_eq!(kept.len(), 2);
    assert!(kept.iter().any(|e| matches!(e.source, Source::Hf)));
    assert!(kept
        .iter()
        .any(|e| matches!(e.source, Source::Civitai) && e.download_count == 200));
}

#[test]
fn drops_nsfw_when_include_nsfw_false() {
    let entries = vec![
        entry(Source::Civitai, 1000, true, true),
        entry(Source::Civitai, 1000, false, true),
    ];
    let opts = ScanOptions {
        include_nsfw: false,
        ..ScanOptions::default()
    };
    let kept = apply(entries, &opts);
    assert_eq!(kept.len(), 1);
    assert!(!kept[0].nsfw);
}

#[test]
fn keeps_nsfw_when_include_nsfw_true() {
    let entries = vec![
        entry(Source::Civitai, 1000, true, true),
        entry(Source::Civitai, 1000, false, true),
    ];
    let opts = ScanOptions {
        include_nsfw: true,
        ..ScanOptions::default()
    };
    let kept = apply(entries, &opts);
    assert_eq!(kept.len(), 2);
}
