use mold_db::catalog::{
    delete_family, family_counts, get_by_id, list, search_fts, upsert_entries, CatalogRow,
    ListParams, SortBy,
};
use rusqlite::Connection;

fn open() -> Connection {
    let mut conn = Connection::open_in_memory().unwrap();
    mold_db::migrations::apply_pending(&mut conn).unwrap();
    conn
}

fn row(id: &str, source: &str, family: &str, name: &str, downloads: i64) -> CatalogRow {
    CatalogRow {
        id: id.into(),
        source: source.into(),
        source_id: id.split(':').nth(1).unwrap().into(),
        name: name.into(),
        author: Some("alice".into()),
        family: family.into(),
        family_role: "finetune".into(),
        sub_family: None,
        modality: "image".into(),
        kind: "checkpoint".into(),
        file_format: "safetensors".into(),
        bundling: "separated".into(),
        size_bytes: Some(123),
        download_count: downloads,
        rating: None,
        likes: 0,
        nsfw: 0,
        thumbnail_url: None,
        description: None,
        license: None,
        license_flags: None,
        tags: Some("[]".into()),
        companions: Some("[]".into()),
        download_recipe: "{\"files\":[]}".into(),
        engine_phase: 1,
        created_at: None,
        updated_at: None,
        added_at: 0,
    }
}

#[test]
fn upsert_then_get_round_trips() {
    let conn = open();
    let r = row("hf:a/b", "hf", "flux", "B", 100);
    upsert_entries(&conn, "flux", std::slice::from_ref(&r)).unwrap();
    let fetched = get_by_id(&conn, "hf:a/b").unwrap().unwrap();
    assert_eq!(fetched.id, r.id);
    assert_eq!(fetched.name, "B");
}

#[test]
fn delete_family_clears_only_that_family() {
    let conn = open();
    upsert_entries(&conn, "flux", &[row("hf:a/b", "hf", "flux", "B", 1)]).unwrap();
    upsert_entries(&conn, "sdxl", &[row("hf:c/d", "hf", "sdxl", "D", 1)]).unwrap();
    delete_family(&conn, "flux").unwrap();
    assert!(get_by_id(&conn, "hf:a/b").unwrap().is_none());
    assert!(get_by_id(&conn, "hf:c/d").unwrap().is_some());
}

#[test]
fn list_paginates_and_sorts() {
    let conn = open();
    let rows: Vec<CatalogRow> = (0..50)
        .map(|i| {
            row(
                &format!("hf:a/{i}"),
                "hf",
                "flux",
                &format!("Model {i}"),
                i as i64,
            )
        })
        .collect();
    upsert_entries(&conn, "flux", &rows).unwrap();

    let params = ListParams {
        family: Some("flux".into()),
        sort: SortBy::Downloads,
        limit: 10,
        offset: 0,
        ..Default::default()
    };
    let page = list(&conn, &params).unwrap();
    assert_eq!(page.len(), 10);
    assert!(page[0].download_count >= page[1].download_count);
}

#[test]
fn family_counts_groups_by_role() {
    let conn = open();
    upsert_entries(
        &conn,
        "flux",
        &[
            CatalogRow {
                family_role: "foundation".into(),
                ..row("hf:f1", "hf", "flux", "F1", 1)
            },
            CatalogRow {
                family_role: "foundation".into(),
                ..row("hf:f2", "hf", "flux", "F2", 1)
            },
            CatalogRow {
                family_role: "finetune".into(),
                ..row("hf:t1", "hf", "flux", "T1", 1)
            },
        ],
    )
    .unwrap();
    let counts = family_counts(&conn).unwrap();
    let flux = counts.iter().find(|c| c.family == "flux").unwrap();
    assert_eq!(flux.foundation, 2);
    assert_eq!(flux.finetune, 1);
}

#[test]
fn search_fts_matches_name() {
    let conn = open();
    upsert_entries(
        &conn,
        "sdxl",
        &[
            row("hf:a/Juggernaut", "hf", "sdxl", "Juggernaut XL", 1),
            row("hf:a/Other", "hf", "sdxl", "Other Model", 1),
        ],
    )
    .unwrap();
    let hits = search_fts(&conn, "juggernaut").unwrap();
    assert_eq!(hits.len(), 1);
    assert_eq!(hits[0].name, "Juggernaut XL");
}
