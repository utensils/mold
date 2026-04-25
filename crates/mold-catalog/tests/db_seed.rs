use mold_catalog::entry::{
    Bundling, CatalogEntry, CatalogId, DownloadRecipe, FamilyRole, FileFormat, Kind, LicenseFlags,
    Modality, RecipeFile, Shard, Source, TokenKind, SHARD_SCHEMA,
};
use mold_catalog::families::Family;
use mold_catalog::shards::seed_db_from_embedded_if_empty;
use mold_catalog::sink::write_shard_atomic;
use mold_db::catalog::list;
use rusqlite::Connection;
use tempfile::TempDir;

fn open_db() -> Connection {
    let mut conn = Connection::open_in_memory().unwrap();
    mold_db::migrations::apply_pending(&mut conn).unwrap();
    conn
}

fn make_entry() -> CatalogEntry {
    CatalogEntry {
        id: CatalogId::from("hf:test/seed"),
        source: Source::Hf,
        source_id: "test/seed".into(),
        name: "seed".into(),
        author: None,
        family: Family::Flux,
        family_role: FamilyRole::Foundation,
        sub_family: None,
        modality: Modality::Image,
        kind: Kind::Checkpoint,
        file_format: FileFormat::Safetensors,
        bundling: Bundling::Separated,
        size_bytes: Some(1),
        download_count: 1,
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
fn seed_uses_disk_shards_when_dir_provided_and_db_empty() {
    let conn = open_db();
    let tmp = TempDir::new().unwrap();
    let shard = Shard {
        schema: SHARD_SCHEMA.into(),
        family: "flux".into(),
        generated_at: "2026-04-25T00:00:00Z".into(),
        scanner_version: "0.9.0".into(),
        entries: vec![make_entry()],
    };
    write_shard_atomic(tmp.path(), &shard).unwrap();
    seed_db_from_embedded_if_empty(&conn, Some(tmp.path())).unwrap();

    let res = list(
        &conn,
        &mold_db::catalog::ListParams {
            limit: 10,
            offset: 0,
            ..Default::default()
        },
    )
    .unwrap();
    assert_eq!(res.len(), 1);
    assert_eq!(res[0].id, "hf:test/seed");
}

#[test]
fn seed_is_idempotent() {
    let conn = open_db();
    let tmp = TempDir::new().unwrap();
    let shard = Shard {
        schema: SHARD_SCHEMA.into(),
        family: "flux".into(),
        generated_at: "2026-04-25T00:00:00Z".into(),
        scanner_version: "0.9.0".into(),
        entries: vec![make_entry()],
    };
    write_shard_atomic(tmp.path(), &shard).unwrap();
    seed_db_from_embedded_if_empty(&conn, Some(tmp.path())).unwrap();
    // Second call should be a no-op (DB no longer empty).
    seed_db_from_embedded_if_empty(&conn, Some(tmp.path())).unwrap();
    let res = list(
        &conn,
        &mold_db::catalog::ListParams {
            limit: 10,
            offset: 0,
            ..Default::default()
        },
    )
    .unwrap();
    assert_eq!(res.len(), 1);
}

#[test]
fn seed_skips_when_disk_dir_missing_and_embedded_stubs_are_empty() {
    let conn = open_db();
    seed_db_from_embedded_if_empty(&conn, None).unwrap();
    let res = list(
        &conn,
        &mold_db::catalog::ListParams {
            limit: 10,
            offset: 0,
            ..Default::default()
        },
    )
    .unwrap();
    // Embedded shards are stubs (zero entries) — DB stays empty.
    assert!(res.is_empty());
}
