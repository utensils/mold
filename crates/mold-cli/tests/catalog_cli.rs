use assert_cmd::Command;
use predicates::prelude::*;
use rusqlite::Connection;
use tempfile::TempDir;

fn seeded_home() -> TempDir {
    let tmp = TempDir::new().unwrap();
    let db = tmp.path().join("mold.db");
    let mut conn = Connection::open(&db).unwrap();
    mold_db::migrations::apply_pending(&mut conn).unwrap();
    mold_db::catalog::upsert_entries(
        &conn,
        "flux",
        &[mold_db::catalog::CatalogRow {
            id: "hf:bfl/FLUX.1-dev".into(),
            source: "hf".into(),
            source_id: "bfl/FLUX.1-dev".into(),
            name: "FLUX.1-dev".into(),
            author: Some("bfl".into()),
            family: "flux".into(),
            family_role: "foundation".into(),
            sub_family: None,
            modality: "image".into(),
            kind: "checkpoint".into(),
            file_format: "safetensors".into(),
            bundling: "separated".into(),
            size_bytes: Some(123),
            download_count: 1000,
            rating: None,
            likes: 0,
            nsfw: 0,
            thumbnail_url: None,
            description: None,
            license: None,
            license_flags: None,
            tags: Some("[]".into()),
            companions: Some("[]".into()),
            download_recipe: r#"{"files":[]}"#.into(),
            engine_phase: 1,
            created_at: None,
            updated_at: None,
            added_at: 0,
        }],
    )
    .unwrap();
    tmp
}

#[test]
fn catalog_list_json_returns_array() {
    let home = seeded_home();
    Command::cargo_bin("mold")
        .unwrap()
        .env("MOLD_HOME", home.path())
        .env("MOLD_DB_PATH", home.path().join("mold.db"))
        .args(["catalog", "list", "--family", "flux", "--json"])
        .assert()
        .success()
        .stdout(predicate::str::contains("FLUX.1-dev"));
}

#[test]
fn catalog_show_prints_entry() {
    let home = seeded_home();
    Command::cargo_bin("mold")
        .unwrap()
        .env("MOLD_HOME", home.path())
        .env("MOLD_DB_PATH", home.path().join("mold.db"))
        .args(["catalog", "show", "hf:bfl/FLUX.1-dev"])
        .assert()
        .success()
        .stdout(predicate::str::contains("FLUX.1-dev"));
}

#[test]
fn catalog_where_prints_not_downloaded() {
    let home = seeded_home();
    Command::cargo_bin("mold")
        .unwrap()
        .env("MOLD_HOME", home.path())
        .env("MOLD_DB_PATH", home.path().join("mold.db"))
        .args(["catalog", "where", "hf:bfl/FLUX.1-dev"])
        .assert()
        .success()
        .stdout(predicate::str::contains("<not downloaded>"));
}
