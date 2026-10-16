//! Tests for the gallery change feed: the feed replayed from nothing, and
//! incrementally from every previous cursor, must equal the full listing
//! (library ∪ trash) after every step of a mutation history.

use super::*;
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use axum::body::Body;
use axum::http::Request;
use mold_core::{GenerateRequest, OutputFormat, OutputMetadata};
use mold_db::{GenerationRecord, RecordSource};
use tower::ServiceExt as _;

use crate::gallery_rows::listing_images;
use crate::routes::GalleryView;

type Replica = BTreeMap<String, serde_json::Value>;

struct Fixture {
    state: AppState,
    _gallery: tempfile::TempDir,
    dir: PathBuf,
}

impl Fixture {
    fn db(&self) -> &MetadataDb {
        self.state.metadata_db.as_ref().as_ref().unwrap()
    }

    fn gate(&self) -> &GalleryPublicationGate {
        &self.state.gallery_publication_gate
    }

    async fn retention(&self) -> u32 {
        self.state
            .config
            .read()
            .await
            .gallery
            .effective_trash_retention_days()
    }
}

async fn fixture() -> Fixture {
    let gallery = tempfile::tempdir().unwrap();
    let dir = std::fs::canonicalize(gallery.path()).unwrap();
    let mut state = AppState::for_tests();
    state.metadata_db = std::sync::Arc::new(Some(MetadataDb::open_in_memory().unwrap()));
    {
        let mut config = state.config.write().await;
        config.output_dir = Some(dir.to_string_lossy().into());
        config.gallery.trash_retention_days = 30;
    }
    state.gallery_feed_readiness.finish(&dir);
    Fixture {
        state,
        _gallery: gallery,
        dir,
    }
}

fn record(name: &str, created_at_ms: i64) -> GenerationRecord {
    let request: GenerateRequest = serde_json::from_value(serde_json::json!({
        "prompt": format!("prompt for {name}"), "model": "test",
        "width": 64, "height": 64, "steps": 1, "guidance": 1.0
    }))
    .unwrap();
    GenerationRecord::from_save(
        Path::new("unused"),
        name,
        OutputFormat::Png,
        OutputMetadata::from_generate_request(&request, 1, None, "test"),
        RecordSource::Server,
        created_at_ms,
    )
}

/// Publish through the real import path: archive commit + DB row.
async fn publish(fx: &Fixture, name: &str, created_at_ms: i64) {
    let mut transaction = crate::batch_transaction::GalleryImportTransaction::begin(
        &fx.dir,
        &format!("import-{name}-{created_at_ms}"),
        0,
        serde_json::json!({"kind": "fixture"}),
        record(name, created_at_ms),
    )
    .unwrap();
    std::fs::write(transaction.staging_path(), name.as_bytes()).unwrap();
    transaction.seal_staged_file().unwrap();
    transaction.mark_prepared().unwrap();
    transaction
        .commit(fx.gate(), fx.state.metadata_db.clone())
        .await
        .unwrap();
}

async fn listing(fx: &Fixture) -> Replica {
    let retention = fx.retention().await;
    let mut all = Replica::new();
    for view in [GalleryView::Library, GalleryView::Trash] {
        let mut images = listing_images(
            fx.db(),
            || fx.gate().committed_archive_index(&fx.dir),
            &fx.dir,
            view,
            retention,
            None,
        )
        .unwrap()
        .unwrap_or_default();
        crate::thumbnails::stamp_poster_revision_all(&mut images);
        for image in images {
            let previous = all.insert(
                image.filename.clone(),
                serde_json::to_value(&image).unwrap(),
            );
            assert!(previous.is_none(), "{} is listed twice", image.filename);
        }
    }
    all
}

async fn page_raw(
    fx: &Fixture,
    since: Option<&str>,
    limit: Option<usize>,
) -> Result<GalleryChangesPage, ApiError> {
    gallery_changes(
        State(fx.state.clone()),
        Query(ChangesQuery {
            since: since.map(str::to_string),
            limit,
        }),
    )
    .await
    .map(|(headers, Json(page))| {
        assert_eq!(headers[header::CACHE_CONTROL], "private, no-store");
        page
    })
}

async fn page(fx: &Fixture, since: Option<&str>, limit: Option<usize>) -> GalleryChangesPage {
    page_raw(fx, since, limit).await.unwrap()
}

/// Pull until `more == false`, applying each page. Returns the cursor and
/// whether any page reset.
async fn pull(
    fx: &Fixture,
    replica: &mut Replica,
    since: Option<String>,
    limit: Option<usize>,
) -> (String, bool) {
    let mut cursor = since;
    let mut reset = false;
    for _ in 0..10_000 {
        let page = page(fx, cursor.as_deref(), limit).await;
        assert_eq!(page.protocol_version, 1);
        assert_eq!(page.instance_id, fx.state.instance_id.as_str());
        if let Some(limit) = limit {
            assert!(page.changes.len() <= limit);
        }
        reset |= page.reset;
        apply_page(replica, &page);
        cursor = Some(page.cursor.clone());
        if !page.more {
            return (page.cursor, reset);
        }
    }
    panic!("the feed never settled");
}

async fn snapshot(fx: &Fixture, limit: Option<usize>) -> (Replica, String) {
    let mut replica = Replica::new();
    let (cursor, reset) = pull(fx, &mut replica, None, limit).await;
    assert!(!reset, "a cursor-less snapshot never resets");
    (replica, cursor)
}

fn assert_same(label: &str, actual: &Replica, expected: &Replica) {
    if actual != expected {
        let a = actual.keys().collect::<Vec<_>>();
        let e = expected.keys().collect::<Vec<_>>();
        if a != e {
            panic!("{label}: names differ\n feed:    {a:?}\n listing: {e:?}");
        }
        for (name, value) in expected {
            assert_eq!(&actual[name], value, "{label}: row {name} differs");
        }
    }
}

/// A tiny deterministic generator, so a failure names its seed.
struct Lcg(u64);
impl Lcg {
    fn next(&mut self) -> u64 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        self.0 >> 33
    }
    fn pick<'a, T>(&mut self, items: &'a [T]) -> Option<&'a T> {
        (!items.is_empty()).then(|| &items[self.next() as usize % items.len()])
    }
}

#[derive(Debug, Clone, Copy)]
enum Op {
    Publish,
    Title,
    Tag,
    Favorite,
    Collect,
    Trash,
    Restore,
    PermanentDelete,
    ArchiveOnly,
    RetireArchiveOnly,
    Retention,
    ForeignArchiveCommit,
}

const OPS: [Op; 12] = [
    Op::Publish,
    Op::Title,
    Op::Tag,
    Op::Favorite,
    Op::Collect,
    Op::Trash,
    Op::Restore,
    Op::PermanentDelete,
    Op::ArchiveOnly,
    Op::RetireArchiveOnly,
    Op::Retention,
    Op::ForeignArchiveCommit,
];

/// Apply one mutation through the same functions the routes call. Returns
/// a description, or `None` when the op had no valid target.
async fn mutate(fx: &Fixture, rng: &mut Lcg, op: Op, counter: &mut i64) -> Option<String> {
    let db = fx.db();
    let dir = fx.dir.as_path();
    let gate = fx.gate();
    let live: Vec<String> = db
        .list_live(Some(dir))
        .unwrap()
        .into_iter()
        .map(|r| r.filename)
        .collect();
    // The keeper's row is never removed: a directory with no DB rows at all
    // is "not indexed" and the listing falls back to a filesystem scan.
    let removable: Vec<String> = live
        .iter()
        .filter(|name| name.as_str() != "keeper.png")
        .cloned()
        .collect();
    let trashed: Vec<String> = db
        .list_trashed(Some(dir))
        .unwrap()
        .into_iter()
        .map(|r| r.filename)
        .collect();
    let archive = gate.committed_archive_index(dir).unwrap();
    let archive_only: Vec<String> = archive
        .entries
        .keys()
        .filter(|name| db.get(dir, name).unwrap().is_none())
        .cloned()
        .collect();
    *counter += 1;
    let n = *counter;
    match op {
        Op::Publish => {
            let name = format!("print-{n:03}.png");
            publish(fx, &name, 1_000_000 + n * 1_000).await;
            Some(format!("publish {name}"))
        }
        Op::Title => {
            let name = rng.pick(&live)?.clone();
            db.set_title(dir, &name, Some(&format!("title {n}")))
                .unwrap();
            Some(format!("title {name}"))
        }
        Op::Tag => {
            let name = rng.pick(&live)?.clone();
            db.add_tags(dir, &name, &[format!("tag-{}", n % 3)])
                .unwrap();
            Some(format!("tag {name}"))
        }
        Op::Favorite => {
            let name = rng.pick(&live)?.clone();
            db.set_favorite(dir, &name, n % 2 == 0).unwrap();
            Some(format!("favorite {name}"))
        }
        Op::Collect => {
            let name = rng.pick(&live)?.clone();
            let collection = match db.list_collections().unwrap().into_iter().next() {
                Some(collection) => collection,
                None => db.create_collection("Shelf", None).unwrap(),
            };
            db.collection_add(&collection.id, dir, std::slice::from_ref(&name))
                .unwrap();
            Some(format!("collect {name}"))
        }
        Op::Trash => {
            let name = rng.pick(&removable)?.clone();
            crate::gallery_trash::trash_print_blocking(dir, &name, db, gate, 2_000_000 + n)
                .unwrap();
            Some(format!("trash {name}"))
        }
        Op::Restore => {
            let name = rng.pick(&trashed)?.clone();
            let retention = fx.retention().await;
            crate::gallery_trash::restore_print_blocking(dir, &name, db, gate, retention).unwrap();
            Some(format!("restore {name}"))
        }
        Op::PermanentDelete => {
            if rng.next().is_multiple_of(2) {
                let name = rng.pick(&trashed)?.clone();
                crate::gallery_trash::purge_trashed_print_blocking(dir, &name, db, gate, None)
                    .unwrap();
                Some(format!("purge {name}"))
            } else {
                let name = rng.pick(&removable)?.clone();
                crate::gallery_trash::hard_delete_live_print_blocking(
                    dir,
                    &name,
                    Some(db),
                    gate,
                    None,
                )
                .unwrap();
                Some(format!("delete {name}"))
            }
        }
        Op::ArchiveOnly => {
            // The DB row goes; the committed archive keeps listing it.
            let name = rng.pick(&removable)?.clone();
            archive.entries.get(&name)?;
            db.delete(dir, &name).unwrap();
            Some(format!("archive-only {name}"))
        }
        Op::RetireArchiveOnly => {
            // An archive-only print removed with no DB write at all: only
            // the archive projector can see it.
            let name = rng.pick(&archive_only)?.clone();
            crate::gallery_trash::hard_delete_live_print_blocking(dir, &name, Some(db), gate, None)
                .unwrap();
            Some(format!("retire archive-only {name}"))
        }
        Op::Retention => {
            let days = [7u32, 30, 0, 90][(n % 4) as usize];
            mold_db::Settings::new(db)
                .set_int("gallery.trash_retention_days", i64::from(days))
                .unwrap();
            fx.state.config.write().await.gallery.trash_retention_days = days;
            Some(format!("retention {days}"))
        }
        Op::ForeignArchiveCommit => {
            // A commit landed by another process: this one's log never saw
            // it, so the projector must re-mark everything. Retiring an
            // ARCHIVE-ONLY print writes nothing to the DB, so only that
            // re-mark can report it; a live print exercises the same gap
            // beside a DB delete.
            let name = match rng.pick(&archive_only) {
                Some(name) => name.clone(),
                None => rng.pick(&removable)?.clone(),
            };
            archive.entries.get(&name)?;
            crate::gallery_trash::hard_delete_live_print_blocking(dir, &name, Some(db), gate, None)
                .unwrap();
            crate::gallery_authority::forget_archive_commit_log(dir);
            Some(format!("foreign delete {name}"))
        }
    }
}

async fn run_history(seed: u64, steps: usize) -> Vec<String> {
    let fx = fixture().await;
    let mut rng = Lcg(seed);
    let mut counter = 0;
    // A keeper row, so the library listing never falls back to the
    // filesystem scan (an empty DB directory is "not indexed").
    publish(&fx, "keeper.png", 1).await;
    let (mut incremental, mut cursor) = snapshot(&fx, None).await;
    assert_same("initial", &incremental, &listing(&fx).await);
    let mut history = Vec::new();
    let mut applied = Vec::new();
    for step in 0..steps {
        // Publishing a quarter of the time keeps targets around for the rest.
        let op = if rng.next().is_multiple_of(4) {
            Op::Publish
        } else {
            *rng.pick(&OPS).unwrap()
        };
        let Some(done) = mutate(&fx, &mut rng, op, &mut counter).await else {
            continue;
        };
        history.push(done);
        applied.push(format!("{op:?}"));
        let expected = listing(&fx).await;
        let label = format!("seed {seed} step {step} after {history:?}");
        let (full, _) = snapshot(&fx, Some([3, 7, 500][step % 3])).await;
        assert_same(&format!("{label} (snapshot)"), &full, &expected);
        let (next, reset) = pull(&fx, &mut incremental, Some(cursor.clone()), None).await;
        assert!(!reset, "{label}: an ordinary delta never resets");
        assert_same(&format!("{label} (incremental)"), &incremental, &expected);
        cursor = next;
    }
    applied
}

#[tokio::test]
async fn feed_replay_equals_the_full_listing_after_every_mutation() {
    let mut covered = std::collections::BTreeSet::new();
    for seed in [1, 7, 42, 1234] {
        covered.extend(run_history(seed, 60).await);
    }
    // Every kind of mutation was exercised by at least one seed.
    for op in OPS {
        assert!(covered.contains(&format!("{op:?}")), "{op:?} never applied");
    }
}

#[tokio::test]
async fn the_snapshot_pages_at_every_boundary() {
    let fx = fixture().await;
    for n in 0..5 {
        publish(&fx, &format!("p{n}.png"), 1_000 + n).await;
    }
    let expected = listing(&fx).await;
    for limit in [1, 2, 500] {
        let first = page(&fx, None, Some(limit)).await;
        assert_eq!(first.more, limit < 5, "limit {limit}");
        let (replica, _) = snapshot(&fx, Some(limit)).await;
        assert_same(&format!("limit {limit}"), &replica, &expected);
    }
    // Limits are clamped, never refused.
    assert_eq!(page(&fx, None, Some(0)).await.changes.len(), 1);
    assert_eq!(page(&fx, None, Some(1_000_000)).await.changes.len(), 5);
    // An idle delta is empty and keeps its cursor.
    let (_, cursor) = snapshot(&fx, None).await;
    let idle = page(&fx, Some(&cursor), None).await;
    assert!(idle.changes.is_empty() && !idle.more && !idle.reset);
    assert_eq!(idle.cursor, cursor);
}

#[tokio::test]
async fn a_trashed_print_carries_its_trash_row_and_a_deleted_one_is_removed() {
    let fx = fixture().await;
    publish(&fx, "keeper.png", 1).await;
    publish(&fx, "binned.png", 2).await;
    publish(&fx, "gone.png", 3).await;
    let (_, cursor) = snapshot(&fx, None).await;
    crate::gallery_trash::trash_print_blocking(&fx.dir, "binned.png", fx.db(), fx.gate(), 5_000)
        .unwrap();
    crate::gallery_trash::hard_delete_live_print_blocking(
        &fx.dir,
        "gone.png",
        Some(fx.db()),
        fx.gate(),
        None,
    )
    .unwrap();
    let delta = page(&fx, Some(&cursor), None).await;
    let by_name = delta
        .changes
        .iter()
        .map(|c| (c.filename.as_str(), c))
        .collect::<BTreeMap<_, _>>();
    let binned = by_name["binned.png"].image.as_ref().unwrap();
    assert_eq!(binned.trashed_at, Some(5));
    assert!(binned.purge_at.is_some());
    assert!(!by_name["binned.png"].removed);
    assert!(by_name["gone.png"].removed && by_name["gone.png"].image.is_none());
    assert!(!by_name.contains_key("keeper.png"), "only changes travel");
}

fn forge(fx: &Fixture, cursor: &str, edit: impl FnOnce(&mut Cursor)) -> String {
    let mut decoded = Cursor::decode(cursor).expect("the server's own cursor decodes");
    edit(&mut decoded);
    let _ = fx;
    decoded.encode()
}

#[tokio::test]
async fn foreign_cursors_reset_to_a_fresh_snapshot() {
    let fx = fixture().await;
    publish(&fx, "a.png", 1).await;
    publish(&fx, "b.png", 2).await;
    let expected = listing(&fx).await;
    let (_, cursor) = snapshot(&fx, None).await;
    let max_seq = mold_db::gallery_changes::feed_state(fx.db())
        .unwrap()
        .max_seq;
    let cases = [
        ("garbage".to_string(), "malformed"),
        (
            base64::engine::general_purpose::URL_SAFE_NO_PAD.encode("g1.x.y"),
            "short",
        ),
        (
            forge(&fx, &cursor, |c| c.epoch = "deadbeef".into()),
            "other epoch",
        ),
        (
            forge(&fx, &cursor, |c| c.dir = "0".repeat(16)),
            "other directory",
        ),
        (
            forge(&fx, &cursor, |c| c.format = "ffffffff".into()),
            "other row format",
        ),
        (
            forge(&fx, &cursor, |c| c.seq = max_seq + 100),
            "since > max_seq",
        ),
    ];
    for (since, label) in cases {
        let mut replica = Replica::new();
        replica.insert("stale.png".into(), serde_json::json!({}));
        let (_, reset) = pull(&fx, &mut replica, Some(since), Some(1)).await;
        assert!(reset, "{label} must reset");
        assert_same(label, &replica, &expected);
    }
    // The server's own cursor does not reset.
    let mut replica = expected.clone();
    assert!(!pull(&fx, &mut replica, Some(cursor), None).await.1);
}

#[tokio::test]
async fn a_cursor_below_the_tombstone_horizon_resets() {
    let fx = fixture().await;
    publish(&fx, "keeper.png", 1).await;
    publish(&fx, "gone.png", 2).await;
    let (_, old) = snapshot(&fx, None).await;
    crate::gallery_trash::hard_delete_live_print_blocking(
        &fx.dir,
        "gone.png",
        Some(fx.db()),
        fx.gate(),
        None,
    )
    .unwrap();
    // Project the deletion, then age and collect its tombstone.
    let _ = page(&fx, Some(&old), None).await;
    fx.db()
        .with_conn(|conn| {
            conn.execute("UPDATE gallery_changes SET changed_at_ms = 0", [])?;
            Ok(())
        })
        .unwrap();
    let keep = fx
        .gate()
        .committed_archive_index(&fx.dir)
        .unwrap()
        .entries
        .keys()
        .cloned()
        .collect();
    let gc = mold_db::gallery_changes::gc_tombstones(fx.db(), &fx.dir, &keep, i64::MAX).unwrap();
    assert!(gc.deleted > 0 && gc.min_seq > 0);
    let reset = page(&fx, Some(&old), None).await;
    assert!(reset.reset, "since < min_seq resets");
}

#[tokio::test]
async fn a_changed_output_dir_resets_the_cursor() {
    let fx = fixture().await;
    publish(&fx, "a.png", 1).await;
    let (_, cursor) = snapshot(&fx, None).await;
    let other = tempfile::tempdir().unwrap();
    let other_dir = std::fs::canonicalize(other.path()).unwrap();
    fx.state.config.write().await.output_dir = Some(other_dir.to_string_lossy().into());
    fx.state.gallery_feed_readiness.finish(&other_dir);
    let moved = page(&fx, Some(&cursor), None).await;
    assert!(moved.reset);
    assert!(moved.changes.is_empty(), "the new directory has no prints");
}

#[tokio::test]
async fn the_feed_answers_503_until_the_directory_is_reconciled() {
    let gallery = tempfile::tempdir().unwrap();
    let dir = std::fs::canonicalize(gallery.path()).unwrap();
    let mut state = AppState::for_tests();
    state.metadata_db = std::sync::Arc::new(Some(MetadataDb::open_in_memory().unwrap()));
    state.config.write().await.output_dir = Some(dir.to_string_lossy().into());
    state.gallery_feed_readiness.begin(&dir);
    let fx = Fixture {
        state,
        _gallery: gallery,
        dir: dir.clone(),
    };
    let error = page_raw(&fx, None, None).await.unwrap_err();
    assert_eq!(error.code, GALLERY_FEED_NOT_READY);
    fx.state.gallery_feed_readiness.finish(&dir);
    assert!(page_raw(&fx, None, None).await.is_ok());
}

#[tokio::test]
async fn a_directory_nobody_reconciled_is_reconciled_on_demand() {
    let gallery = tempfile::tempdir().unwrap();
    let dir = std::fs::canonicalize(gallery.path()).unwrap();
    let mut state = AppState::for_tests();
    state.metadata_db = std::sync::Arc::new(Some(MetadataDb::open_in_memory().unwrap()));
    state.config.write().await.output_dir = Some(dir.to_string_lossy().into());
    let fx = Fixture {
        state,
        _gallery: gallery,
        dir: dir.clone(),
    };
    assert_eq!(
        page_raw(&fx, None, None).await.unwrap_err().code,
        GALLERY_FEED_NOT_READY
    );
    for _ in 0..200 {
        if fx.state.gallery_feed_readiness.is_ready(&dir) {
            return;
        }
        tokio::time::sleep(std::time::Duration::from_millis(10)).await;
    }
    panic!("the on-demand reconcile never finished");
}

#[tokio::test]
async fn the_feed_is_unsupported_without_the_db_or_output() {
    let fx = fixture().await;
    let mut no_db = fx.state.clone();
    no_db.metadata_db = std::sync::Arc::new(None);
    let error = gallery_changes(State(no_db), Query(ChangesQuery::default()))
        .await
        .unwrap_err();
    assert_eq!(error.code, GALLERY_FEED_UNSUPPORTED);
    fx.state.config.write().await.output_dir = Some("".into());
    assert_eq!(
        page_raw(&fx, None, None).await.unwrap_err().code,
        GALLERY_FEED_UNSUPPORTED
    );
}

fn router(state: AppState, keys: Option<&str>) -> axum::Router {
    let auth: crate::auth::AuthState = keys.map(|key| {
        std::sync::Arc::new(crate::auth::ApiKeySet::new(
            std::collections::HashSet::from([key.to_string()]),
        ))
    });
    crate::routes::create_router(state)
        .layer(axum::middleware::from_fn(crate::auth::require_api_key))
        .layer(crate::auth::test_loopback_peer())
        .layer(axum::middleware::from_fn_with_state(
            auth,
            crate::auth::inject_auth_state,
        ))
}

async fn get(app: &axum::Router, uri: &str, headers: &[(&str, &str)]) -> axum::response::Response {
    let mut request = Request::builder().uri(uri);
    for (name, value) in headers {
        request = request.header(*name, *value);
    }
    app.clone()
        .oneshot(request.body(Body::empty()).unwrap())
        .await
        .unwrap()
}

#[tokio::test]
async fn the_feed_takes_the_listing_auth_on_a_keyed_host() {
    let fx = fixture().await;
    publish(&fx, "a.png", 1).await;
    let app = router(fx.state.clone(), Some("feed-key"));
    for uri in ["/api/gallery", "/api/gallery/changes"] {
        assert_eq!(get(&app, uri, &[]).await.status(), StatusCode::UNAUTHORIZED);
        assert_eq!(
            get(&app, uri, &[("x-api-key", "feed-key")]).await.status(),
            StatusCode::OK,
            "{uri}"
        );
    }
}

#[tokio::test]
async fn capability_is_advertised_only_with_the_db_and_output() {
    async fn caps(state: AppState) -> serde_json::Value {
        let response = get(&router(state, None), "/api/capabilities", &[]).await;
        let bytes = axum::body::to_bytes(response.into_body(), usize::MAX)
            .await
            .unwrap();
        serde_json::from_slice::<serde_json::Value>(&bytes).unwrap()["gallery"].clone()
    }
    let fx = fixture().await;
    assert_eq!(
        caps(fx.state.clone()).await["changes_feed"],
        serde_json::json!({"protocol_version": 1})
    );
    let mut no_db = fx.state.clone();
    no_db.metadata_db = std::sync::Arc::new(None);
    assert!(caps(no_db).await.get("changes_feed").is_none());
    fx.state.config.write().await.output_dir = Some("".into());
    assert!(caps(fx.state.clone()).await.get("changes_feed").is_none());
}

#[tokio::test]
async fn the_legacy_listing_answers_304_without_building() {
    let fx = fixture().await;
    publish(&fx, "keeper.png", 1).await;
    publish(&fx, "other.png", 2).await;
    let app = router(fx.state.clone(), None);
    let etag_of = |response: &axum::response::Response| {
        response.headers()[header::ETAG]
            .to_str()
            .unwrap()
            .to_string()
    };
    // The first build indexes asset inventories lazily, which the validator
    // counts: the ETag it was answered with is already stale, and the next
    // request rebuilds once.
    let first = get(&app, "/api/gallery", &[]).await;
    assert_eq!(first.status(), StatusCode::OK);
    let indexed = get(&app, "/api/gallery", &[("if-none-match", &etag_of(&first))]).await;
    let etag = etag_of(&indexed);
    assert!(etag.starts_with("W/\"g1-"), "{etag}");

    let builds = crate::gallery_rows::full_builds_for_dir(&fx.dir);
    let cached = get(&app, "/api/gallery", &[("if-none-match", &etag)]).await;
    assert_eq!(cached.status(), StatusCode::NOT_MODIFIED);
    assert_eq!(etag_of(&cached), etag);
    assert_eq!(
        crate::gallery_rows::full_builds_for_dir(&fx.dir),
        builds,
        "an unchanged library is answered without building the listing"
    );

    // Every kind of visible change moves the validator.
    let mut seen = vec![etag];
    let mut expect_changed = |label: &str, etag: String| {
        assert!(!seen.contains(&etag), "{label} must change the ETag");
        seen.push(etag);
    };
    fx.db()
        .set_title(&fx.dir, "keeper.png", Some("Renamed"))
        .unwrap();
    expect_changed("a title", etag_of(&get(&app, "/api/gallery", &[]).await));
    // An archive-only change: no DB write at all.
    fx.db().delete(&fx.dir, "other.png").unwrap();
    let after_delete = etag_of(&get(&app, "/api/gallery", &[]).await);
    expect_changed("dropping a row", after_delete);
    crate::gallery_trash::hard_delete_live_print_blocking(
        &fx.dir,
        "other.png",
        Some(fx.db()),
        fx.gate(),
        None,
    )
    .unwrap();
    expect_changed(
        "retiring an archive-only print",
        etag_of(&get(&app, "/api/gallery", &[]).await),
    );
    // Retention moves every trashed print's purge_at.
    crate::gallery_trash::trash_print_blocking(&fx.dir, "keeper.png", fx.db(), fx.gate(), 9_000)
        .unwrap();
    let trash_etag = etag_of(&get(&app, "/api/gallery?view=trash", &[]).await);
    fx.state.config.write().await.gallery.trash_retention_days = 7;
    let retention_etag = etag_of(&get(&app, "/api/gallery?view=trash", &[]).await);
    assert_ne!(trash_etag, retention_etag, "a retention change");
    let stale = get(
        &app,
        "/api/gallery?view=trash",
        &[("if-none-match", &trash_etag)],
    )
    .await;
    assert_eq!(stale.status(), StatusCode::OK);
}

#[tokio::test]
async fn the_legacy_listing_keeps_the_body_hash_until_the_feed_is_ready() {
    let fx = fixture().await;
    publish(&fx, "a.png", 1).await;
    let mut unready = fx.state.clone();
    unready.gallery_feed_readiness = std::sync::Arc::default();
    let response = get(&router(unready, None), "/api/gallery", &[]).await;
    let etag = response.headers()[header::ETAG].to_str().unwrap();
    assert!(!etag.starts_with("W/"), "body hash: {etag}");
    // `?filename=` keeps the body hash too.
    let response = get(
        &router(fx.state.clone(), None),
        "/api/gallery?filename=a.png",
        &[],
    )
    .await;
    assert!(!response.headers()[header::ETAG]
        .to_str()
        .unwrap()
        .starts_with("W/"));
}

#[tokio::test]
async fn a_delta_of_one_change_builds_one_row() {
    let fx = fixture().await;
    for n in 0..30 {
        publish(&fx, &format!("p{n:02}.png"), 1_000 + n).await;
    }
    let (_, cursor) = snapshot(&fx, None).await;
    fx.db().set_favorite(&fx.dir, "p07.png", true).unwrap();
    let delta = page(&fx, Some(&cursor), None).await;
    assert_eq!(
        delta
            .changes
            .iter()
            .map(|c| c.filename.as_str())
            .collect::<Vec<_>>(),
        vec!["p07.png"]
    );
    assert_eq!(crate::gallery_rows::full_builds_for_dir(&fx.dir), 0);
}

/// `docs/contracts/gallery-changes-v1.json`: every page deserializes into
/// the wire type, and replaying each scenario's pages gives its expected
/// replica.
#[test]
fn the_contract_fixture_replays_to_its_expected_state() {
    let raw = include_str!("../../../docs/contracts/gallery-changes-v1.json");
    let fixture: serde_json::Value = serde_json::from_str(raw).unwrap();
    assert_eq!(fixture["protocol_version"], 1);
    let scenarios = fixture["scenarios"].as_array().unwrap();
    assert!(scenarios.len() >= 3);
    for scenario in scenarios {
        let name = scenario["name"].as_str().unwrap();
        let mut replica: Replica =
            serde_json::from_value::<Vec<String>>(scenario["initial"].clone())
                .unwrap()
                .into_iter()
                .map(|filename| (filename, serde_json::json!({})))
                .collect();
        let mut cursor = scenario["since"].as_str().map(str::to_string);
        for raw_page in scenario["pages"].as_array().unwrap() {
            let page: GalleryChangesPage = serde_json::from_value(raw_page.clone())
                .unwrap_or_else(|e| panic!("{name}: page does not deserialize: {e}"));
            apply_page(&mut replica, &page);
            cursor = Some(page.cursor);
        }
        let expected = serde_json::from_value::<BTreeMap<String, serde_json::Value>>(
            scenario["expected"].clone(),
        )
        .unwrap();
        assert_eq!(
            replica.keys().collect::<Vec<_>>(),
            expected.keys().collect::<Vec<_>>(),
            "{name}"
        );
        for (filename, fields) in expected {
            for (field, value) in fields.as_object().unwrap() {
                assert_eq!(
                    &replica[&filename][field], value,
                    "{name}: {filename}.{field}"
                );
            }
        }
        assert_eq!(
            cursor.as_deref(),
            scenario["final_cursor"].as_str(),
            "{name}: the client resumes from the last page's cursor"
        );
    }
}

#[test]
fn cursors_round_trip_and_reject_anything_else() {
    let cursor = Cursor {
        epoch: "abc".into(),
        dir: dir_token(Path::new("/tmp/x")),
        format: row_format_token(),
        seq: 42,
    };
    assert_eq!(Cursor::decode(&cursor.encode()), Some(cursor.clone()));
    for bad in ["", "!!!", "Zm9v"] {
        assert_eq!(Cursor::decode(bad), None, "{bad}");
    }
    let negative = base64::engine::general_purpose::URL_SAFE_NO_PAD.encode("g1.a.b.c.-1");
    assert_eq!(Cursor::decode(&negative), None);
}
