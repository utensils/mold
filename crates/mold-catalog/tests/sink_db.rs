use mold_catalog::entry::{
    Bundling, CatalogEntry, CatalogId, DownloadRecipe, FamilyRole, FileFormat, Kind, LicenseFlags,
    Modality, RecipeFile, Source, TokenKind,
};
use mold_catalog::families::Family;
use mold_catalog::sink::upsert_family;
use mold_db::catalog::{get_by_id, list, ListParams, SortBy};
use rusqlite::Connection;

fn open_db() -> Connection {
    let mut conn = Connection::open_in_memory().unwrap();
    mold_db::migrations::apply_pending(&mut conn).unwrap();
    conn
}

fn entry(id: &str, family: Family, downloads: u64) -> CatalogEntry {
    CatalogEntry {
        id: CatalogId::from(id),
        source: Source::Hf,
        source_id: id.trim_start_matches("hf:").to_string(),
        name: id.into(),
        author: None,
        family,
        family_role: FamilyRole::Finetune,
        sub_family: None,
        modality: Modality::Image,
        kind: Kind::Checkpoint,
        file_format: FileFormat::Safetensors,
        bundling: Bundling::Separated,
        size_bytes: Some(1),
        download_count: downloads,
        rating: None,
        likes: 0,
        nsfw: false,
        thumbnail_url: None,
        description: None,
        license: None,
        license_flags: LicenseFlags::default(),
        tags: vec![],
        companions: vec![],
        download_recipe: DownloadRecipe {
            files: vec![RecipeFile {
                url: "u".into(),
                dest: "d".into(),
                sha256: None,
                size_bytes: None,
            }],
            needs_token: Some(TokenKind::Hf),
        },
        engine_phase: 1,
        created_at: None,
        updated_at: None,
        added_at: 0,
    }
}

#[test]
fn upsert_family_replaces_only_that_family() {
    let conn = open_db();
    upsert_family(&conn, Family::Flux, &[entry("hf:f1", Family::Flux, 1)]).unwrap();
    upsert_family(&conn, Family::Sdxl, &[entry("hf:s1", Family::Sdxl, 1)]).unwrap();
    upsert_family(&conn, Family::Flux, &[entry("hf:f2", Family::Flux, 1)]).unwrap();

    assert!(
        get_by_id(&conn, "hf:f1").unwrap().is_none(),
        "old flux row replaced"
    );
    assert!(get_by_id(&conn, "hf:f2").unwrap().is_some());
    assert!(
        get_by_id(&conn, "hf:s1").unwrap().is_some(),
        "sdxl untouched"
    );
}

#[test]
fn upsert_family_preserves_engine_phase_and_companions() {
    let conn = open_db();
    let mut e = entry("hf:phase3", Family::Flux, 1);
    e.engine_phase = 3;
    e.companions = vec!["t5-v1_1-xxl".into(), "clip-l".into()];
    upsert_family(&conn, Family::Flux, &[e]).unwrap();
    let row = get_by_id(&conn, "hf:phase3").unwrap().unwrap();
    assert_eq!(row.engine_phase, 3);
    let comps: Vec<String> = serde_json::from_str(row.companions.as_deref().unwrap()).unwrap();
    assert_eq!(comps, vec!["t5-v1_1-xxl", "clip-l"]);
}

#[test]
fn list_filters_by_max_engine_phase() {
    let conn = open_db();
    let mut runnable = entry("hf:r", Family::Flux, 1);
    runnable.engine_phase = 1;
    let mut pending = entry("hf:p", Family::Flux, 1);
    pending.engine_phase = 3;
    upsert_family(&conn, Family::Flux, &[runnable, pending]).unwrap();
    let res = list(
        &conn,
        &ListParams {
            max_engine_phase: Some(1),
            sort: SortBy::Name,
            limit: 10,
            offset: 0,
            ..Default::default()
        },
    )
    .unwrap();
    assert_eq!(res.len(), 1);
    assert_eq!(res[0].id, "hf:r");
}
