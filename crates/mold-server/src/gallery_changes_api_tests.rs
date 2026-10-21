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

fn epoch(db: &MetadataDb) -> String {
    mold_db::gallery_changes::feed_state(db).unwrap().epoch
}

/// What the startup reconcile records when it finishes.
fn mark_ready(state: &AppState, dir: &Path) {
    state.gallery_feed_readiness.finish(
        dir,
        GalleryFeedReadiness::epoch_of(state.metadata_db.as_ref().as_ref()),
    );
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
    mark_ready(&state, &dir);
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
    assert!(
        reset,
        "a cursor-less snapshot replaces the replica: its first page resets"
    );
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
    let mut previous_expected = listing(&fx).await;
    assert_same("initial", &incremental, &previous_expected);
    let mut previous_validators = validators(&fx).await;
    // Every (cursor, replica) a client could still be holding.
    let mut checkpoints = vec![(cursor.clone(), incremental.clone())];
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
        // The legacy validator is read BEFORE the listing is built, as the
        // route does.
        let current_validators = validators(&fx).await;
        let expected = listing(&fx).await;
        let label = format!("seed {seed} step {step} after {history:?}");
        if expected != previous_expected {
            assert_ne!(
                current_validators, previous_validators,
                "{label}: the listing changed but its cheap ETag did not"
            );
        }
        let (full, _) = snapshot(&fx, Some([3, 7, 500][step % 3])).await;
        assert_same(&format!("{label} (snapshot)"), &full, &expected);
        let (next, reset) = pull(&fx, &mut incremental, Some(cursor.clone()), None).await;
        assert!(!reset, "{label}: an ordinary delta never resets");
        assert_same(&format!("{label} (incremental)"), &incremental, &expected);
        // A client that last synced at a RANDOM earlier point converges too.
        let (old_cursor, mut old_replica) = rng.pick(&checkpoints).unwrap().clone();
        let (_, reset) = pull(&fx, &mut old_replica, Some(old_cursor), Some(4)).await;
        assert!(!reset, "{label}: an earlier cursor never resets");
        assert_same(
            &format!("{label} (from an earlier cursor)"),
            &old_replica,
            &expected,
        );
        cursor = next;
        checkpoints.push((cursor.clone(), incremental.clone()));
        previous_expected = expected;
        previous_validators = validators(&fx).await;
    }
    applied
}

/// The legacy listing's cheap validators for both views, as the route
/// computes them.
async fn validators(fx: &Fixture) -> (Option<String>, Option<String>) {
    let retention = fx.retention().await;
    let validator = |library| {
        listing_validator(
            fx.db(),
            fx.gate(),
            &fx.state.gallery_feed_readiness,
            &fx.dir,
            library,
            retention,
        )
        .unwrap()
    };
    (validator(true), validator(false))
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
    mark_ready(&fx.state, &other_dir);
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
    mark_ready(&fx.state, &dir);
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
        if fx
            .state
            .gallery_feed_readiness
            .is_ready(&dir, &epoch(fx.db()))
        {
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

/// Another process (the CLI, a second server sharing `mold.db`) commits an
/// archive change AND projects it in the window between this feed read's
/// projection and its page read. The page then carries that name's fresh
/// mark; built from the stale snapshot, the row would be wrong and the
/// cursor would carry the client past the mark for good.
#[tokio::test]
async fn a_foreign_commit_between_projection_and_page_is_never_skipped() {
    let fx = fixture().await;
    publish(&fx, "keeper.png", 1).await;
    publish(&fx, "x.png", 2).await;
    // Archive-only: retiring it later writes nothing to the DB.
    fx.db().delete(&fx.dir, "x.png").unwrap();
    let (mut replica, cursor) = snapshot(&fx, None).await;
    assert!(replica.contains_key("x.png"));

    let foreign = fx.state.clone();
    let dir = fx.dir.clone();
    install_feed_hook(&fx.dir, move || {
        let db = foreign.metadata_db.as_ref().as_ref().unwrap();
        let gate = &foreign.gallery_publication_gate;
        crate::gallery_trash::hard_delete_live_print_blocking(&dir, "x.png", Some(db), gate, None)
            .unwrap();
        // This process never saw that commit…
        crate::gallery_authority::forget_archive_commit_log(&dir);
        // …and the other process projects it before our page is read.
        let snapshot = gate.committed_archive_snapshot(&dir).unwrap();
        project_archive_changes(db, &dir, &snapshot).unwrap();
    });
    let (cursor, _) = pull(&fx, &mut replica, Some(cursor), None).await;
    let expected = listing(&fx).await;
    assert!(!expected.contains_key("x.png"));
    assert_same("after the foreign commit", &replica, &expected);
    // And nothing is left owed to the client.
    let (_, _) = pull(&fx, &mut replica, Some(cursor), None).await;
    assert_same("one pull later", &replica, &expected);
}

/// Projectors racing each other (several readers, commits landing between
/// them, this process's log forgotten midway) still converge on the listing.
#[tokio::test]
async fn concurrent_projectors_converge() {
    let fx = fixture().await;
    publish(&fx, "keeper.png", 1).await;
    for n in 0..8 {
        publish(&fx, &format!("c{n}.png"), 10 + n).await;
    }
    let (mut early, early_cursor) = snapshot(&fx, None).await;
    for n in 0..4 {
        fx.db().delete(&fx.dir, &format!("c{n}.png")).unwrap();
    }
    let retention = fx.retention().await;
    let instance = fx.state.instance_id.to_string();
    std::thread::scope(|scope| {
        for _ in 0..4 {
            scope.spawn(|| {
                for _ in 0..5 {
                    feed_page(fx.db(), fx.gate(), &fx.dir, None, 3, retention, &instance).unwrap();
                }
            });
        }
        scope.spawn(|| {
            for n in 0..4 {
                crate::gallery_trash::hard_delete_live_print_blocking(
                    &fx.dir,
                    &format!("c{n}.png"),
                    Some(fx.db()),
                    fx.gate(),
                    None,
                )
                .unwrap();
                if n == 1 {
                    crate::gallery_authority::forget_archive_commit_log(&fx.dir);
                }
            }
        });
    });
    let expected = listing(&fx).await;
    let (full, _) = snapshot(&fx, Some(2)).await;
    assert_same("snapshot", &full, &expected);
    pull(&fx, &mut early, Some(early_cursor), None).await;
    assert_same("from before the race", &early, &expected);
}

fn textured_glb() -> Vec<u8> {
    use mold_inference::hunyuan3d::{glb, mesh::Mesh};
    let png = |rgb| {
        let image = image::RgbImage::from_pixel(2, 2, image::Rgb(rgb));
        let mut bytes = Vec::new();
        image
            .write_to(
                &mut std::io::Cursor::new(&mut bytes),
                image::ImageFormat::Png,
            )
            .unwrap();
        bytes
    };
    glb::write_glb(
        &Mesh {
            vertices: vec![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
            faces: vec![[0, 1, 2]],
            normals: None,
            vertex_colors: None,
            uvs: Some(vec![[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]]),
        },
        &glb::GlbMaterial {
            base_color_texture: Some(png([12, 34, 56])),
            metallic_roughness_texture: Some(png([78, 90, 123])),
            normal_texture: Some(png([128, 128, 255])),
            ..Default::default()
        },
        None,
    )
    .unwrap()
}

fn mesh_row(dir: &Path, name: &str) -> GenerationRecord {
    let mut record = GenerationRecord::from_save(
        dir,
        name,
        OutputFormat::Glb,
        mold_db::metadata_io::synthesize_from_filename(name, 1_700_000_000),
        RecordSource::Server,
        1_700_000_000_000,
    );
    record.file_mtime_ms = Some(1_700_000_000_000);
    record.file_size_bytes = Some(4096);
    record
}

/// Pull until the feed has nothing more to say, then prove it says nothing
/// twice in a row: lazily indexed assets must not make reads write forever.
async fn settle(fx: &Fixture, replica: &mut Replica, mut cursor: String) -> (String, usize) {
    let mut quiet = 0;
    for pulls in 1..=10 {
        let page = page(fx, Some(&cursor), None).await;
        assert!(!page.more && !page.reset);
        let changed = !page.changes.is_empty();
        apply_page(replica, &page);
        cursor = page.cursor;
        quiet = if changed { 0 } else { quiet + 1 };
        if quiet == 2 {
            return (cursor, pulls);
        }
    }
    panic!("the feed never reached a fixed point");
}

#[tokio::test]
async fn a_mesh_whose_index_failed_is_redelivered_and_the_feed_settles() {
    let fx = fixture().await;
    publish(&fx, "keeper.png", 1).await;
    // A mesh row whose file is not there yet: its first index fails and the
    // row goes out with no assets.
    fx.db().upsert(&mesh_row(&fx.dir, "chair.glb")).unwrap();
    // A mesh that can never be read: it must not keep the feed busy.
    fx.db().upsert(&mesh_row(&fx.dir, "broken.glb")).unwrap();
    std::fs::write(fx.dir.join("broken.glb"), b"not a glb").unwrap();
    let (mut replica, cursor) = snapshot(&fx, None).await;
    assert_eq!(replica["chair.glb"].get("assets"), None);

    std::fs::write(fx.dir.join("chair.glb"), textured_glb()).unwrap();
    let (cursor, _) = settle(&fx, &mut replica, cursor).await;
    assert_eq!(
        replica["chair.glb"]["assets"].as_array().map(Vec::len),
        Some(3),
        "the inventory indexed after a failure is re-sent"
    );
    assert_same("settled", &replica, &listing(&fx).await);
    // The legacy validator is stable once settled, and listing builds no
    // longer write.
    let before = validators(&fx).await;
    listing(&fx).await;
    listing(&fx).await;
    assert_eq!(validators(&fx).await, before);
    let quiet = page(&fx, Some(&cursor), None).await;
    assert!(quiet.changes.is_empty());
}

#[tokio::test]
async fn a_fresh_snapshot_of_meshes_reaches_a_fixed_point() {
    let fx = fixture().await;
    publish(&fx, "keeper.png", 1).await;
    fx.db().upsert(&mesh_row(&fx.dir, "chair.glb")).unwrap();
    std::fs::write(fx.dir.join("chair.glb"), textured_glb()).unwrap();
    let (mut replica, cursor) = snapshot(&fx, None).await;
    let (_, pulls) = settle(&fx, &mut replica, cursor).await;
    assert!(pulls <= 3, "one extra delivery at most, took {pulls} pulls");
    assert_same("settled", &replica, &listing(&fx).await);
}

/// A corruption rebuild recreates `mold.db` with a new epoch and no rows;
/// the feed must not serve that as a library until it is reconciled again.
#[tokio::test]
async fn a_rebuilt_database_is_not_ready_until_reconciled_again() {
    let fx = fixture().await;
    publish(&fx, "a.png", 1).await;
    assert!(page_raw(&fx, None, None).await.is_ok());
    fx.db()
        .with_conn(|conn| {
            conn.execute(
                "UPDATE gallery_feed_meta SET epoch = 'rebuilt-after-corruption'",
                [],
            )?;
            Ok(())
        })
        .unwrap();
    assert_eq!(
        page_raw(&fx, None, None).await.unwrap_err().code,
        GALLERY_FEED_NOT_READY
    );
    assert_eq!(validators(&fx).await, (None, None), "nor the cheap ETag");
    for _ in 0..200 {
        if fx
            .state
            .gallery_feed_readiness
            .is_ready(&fx.dir, "rebuilt-after-corruption")
        {
            assert!(page_raw(&fx, None, None).await.is_ok());
            return;
        }
        tokio::time::sleep(std::time::Duration::from_millis(10)).await;
    }
    panic!("the re-claimed reconcile never finished");
}

#[tokio::test]
async fn a_fresh_host_with_no_output_dir_is_ready_and_empty() {
    let home = tempfile::tempdir().unwrap();
    let dir = std::fs::canonicalize(home.path()).unwrap().join("output");
    let mut state = AppState::for_tests();
    state.metadata_db = std::sync::Arc::new(Some(MetadataDb::open_in_memory().unwrap()));
    state.config.write().await.output_dir = Some(dir.to_string_lossy().into());
    let fx = Fixture {
        state,
        _gallery: home,
        dir: dir.clone(),
    };
    assert_eq!(
        page_raw(&fx, None, None).await.unwrap_err().code,
        GALLERY_FEED_NOT_READY
    );
    for _ in 0..200 {
        if let Ok(page) = page_raw(&fx, None, None).await {
            assert!(page.changes.is_empty() && page.reset && !page.more);
            assert!(!dir.exists(), "a read never creates the directory");
            return;
        }
        tokio::time::sleep(std::time::Duration::from_millis(10)).await;
    }
    panic!("a missing output directory stayed not-ready");
}

/// Pins how a gallery row is derived. If this changes for an existing print
/// (a new field, a different fallback, a new stamp), every replica holding
/// the old shape is stale with no seq to tell it: bump `ROW_FORMAT` and
/// update the expectation together.
#[test]
fn the_row_derivation_is_pinned_to_row_format() {
    assert_eq!(ROW_FORMAT, 2, "bump the golden row below with ROW_FORMAT");
    let dir = Path::new("/golden-gallery");
    let db = MetadataDb::open_in_memory().unwrap();
    let mut row = record("golden.png", 1_700_000_000_000);
    row.metadata.prompt = r"scene\npath".into();
    row.metadata.negative_prompt = Some(r"bad\ttext".into());
    row.metadata.seed = u64::MAX;
    row.output_dir = dir.to_string_lossy().into_owned();
    row.file_mtime_ms = Some(1_700_000_001_000);
    row.file_size_bytes = Some(2048);
    db.upsert(&row).unwrap();
    db.set_title(dir, "golden.png", Some("Golden")).unwrap();
    db.set_favorite(dir, "golden.png", true).unwrap();
    db.add_tags(dir, "golden.png", &["b".to_string(), "A".to_string()])
        .unwrap();
    db.mark_trashed(dir, "golden.png", 1_700_000_100_000)
        .unwrap();
    let rows = crate::gallery_rows::build_feed_rows(
        &db,
        &Default::default(),
        dir,
        &["golden.png".to_string()],
        30,
    )
    .unwrap();
    let actual = serde_json::to_value(rows["golden.png"].as_ref().unwrap()).unwrap();
    let expected: serde_json::Value = serde_json::from_str(GOLDEN_ROW).unwrap();
    assert_eq!(
        actual, expected,
        "the gallery row derivation changed: bump ROW_FORMAT\n{actual:#}"
    );
}

const GOLDEN_ROW: &str = r#"{
  "favorite": true,
  "filename": "golden.png",
  "format": "png",
  "media_version": "1700000001000:2048",
  "metadata": {
    "generation_height": 64,
    "generation_width": 64,
    "guidance": 1.0,
    "height": 64,
    "model": "test",
    "output_mode": "one-shot",
    "pipeline_requested": false,
    "prompt": "scene\\npath",
    "negative_prompt": "bad\\ttext",
    "seed": 18446744073709551615,
    "steps": 1,
    "version": "test",
    "width": 64
  },
  "purge_at": 1702592100,
  "size_bytes": 2048,
  "tags": ["A", "b"],
  "timestamp": 1700000001,
  "title": "Golden",
  "trashed_at": 1700000100
}"#;

/// The output directory does not exist when the read starts; another
/// process creates it, publishes an archive-only print and projects it
/// before the page is read. Built from the empty archive that print would
/// ship as a removal and the cursor would pass it.
#[tokio::test]
async fn a_directory_created_under_a_read_is_re_read_not_shipped_as_removals() {
    let home = tempfile::tempdir().unwrap();
    let dir = std::fs::canonicalize(home.path()).unwrap().join("output");
    let mut state = AppState::for_tests();
    state.metadata_db = std::sync::Arc::new(Some(MetadataDb::open_in_memory().unwrap()));
    state.config.write().await.output_dir = Some(dir.to_string_lossy().into());
    mark_ready(&state, &dir);
    let fx = Fixture {
        state,
        _gallery: home,
        dir: dir.clone(),
    };
    let foreign = fx.state.clone();
    let hook_dir = dir.clone();
    install_feed_hook(&dir, move || {
        std::fs::create_dir_all(&hook_dir).unwrap();
        let guard = crate::batch_transaction::acquire_gallery_bookkeeping_lock(&hook_dir).unwrap();
        let loaded = crate::gallery_authority::load_or_initialize(&hook_dir, &guard, || {
            Ok(Default::default())
        })
        .unwrap();
        let mut index = loaded.index;
        let mut ghost = record("ghost.png", 5);
        ghost.output_dir = hook_dir.to_string_lossy().into_owned();
        index.entries.insert(
            "ghost.png".into(),
            crate::batch_transaction::CommittedArchiveEntry {
                identity: crate::batch_transaction::ArchivedChildIdentity {
                    parent_id: "foreign".into(),
                    attempt_generation: 0,
                    child_index: 0,
                    final_name: "ghost.png".into(),
                    checksum_sha256: "0".repeat(64),
                    size_bytes: 1,
                },
                record: ghost,
                facts: None,
                retained_media: Vec::new(),
            },
        );
        crate::gallery_authority::commit_snapshot(
            &hook_dir,
            &guard,
            loaded.generation,
            &mut index,
            "publish_batch",
            Vec::new(),
        )
        .unwrap();
        drop(guard);
        crate::gallery_authority::forget_archive_commit_log(&hook_dir);
        let db = foreign.metadata_db.as_ref().as_ref().unwrap();
        let snapshot = foreign
            .gallery_publication_gate
            .committed_archive_snapshot(&hook_dir)
            .unwrap();
        project_archive_changes(db, &hook_dir, &snapshot).unwrap();
    });
    let mut replica = Replica::new();
    pull(&fx, &mut replica, None, None).await;
    assert!(
        replica.contains_key("ghost.png"),
        "the print published under the read is delivered, not removed"
    );
}

/// The epoch is captured BEFORE the reconcile: a rebuild in the middle of
/// it leaves the new database not ready.
#[test]
fn readiness_is_for_the_epoch_the_reconcile_started_against() {
    let db = MetadataDb::open_in_memory().unwrap();
    let readiness = GalleryFeedReadiness::default();
    let dir = Path::new("/tmp/readiness-epoch");
    let before = GalleryFeedReadiness::epoch_of(Some(&db));
    db.with_conn(|conn| {
        conn.execute(
            "UPDATE gallery_feed_meta SET epoch = 'rebuilt-mid-reconcile'",
            [],
        )?;
        Ok(())
    })
    .unwrap();
    readiness.finish(dir, before.clone());
    assert!(readiness.is_ready(dir, &before));
    assert!(!readiness.is_ready(dir, "rebuilt-mid-reconcile"));
    assert!(readiness.claim_reconcile(dir, "rebuilt-mid-reconcile"));
    assert!(
        !readiness.is_ready(dir, ""),
        "an unknown epoch is never ready"
    );
}
