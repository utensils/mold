// Env-var tests use `std::sync::Mutex<()>` to serialize process-global
// mutations; holding the guard across `.await` is intentional under the
// current-thread tokio test runtime.
#![allow(clippy::await_holding_lock)]

use std::sync::Arc;

use axum::extract::{Path, State};
use axum::http::StatusCode;
use axum::response::IntoResponse;
use mold_catalog::scanner::{ProgressHandle, ScanReport};
use mold_db::catalog::CatalogRow;
use mold_db::MetadataDb;
use tokio::sync::Mutex;

use super::{post_catalog_download, CatalogScanQueue, CatalogScanStatus, ScanDriver};
use crate::state::AppState;

struct FakeDriver {
    report: Arc<Mutex<Option<ScanReport>>>,
}

#[async_trait::async_trait]
impl ScanDriver for FakeDriver {
    async fn run(
        &self,
        _opts: mold_catalog::scanner::ScanOptions,
        _progress: ProgressHandle,
    ) -> ScanReport {
        self.report.lock().await.take().unwrap_or_default()
    }
}

#[tokio::test]
async fn enqueue_then_status_transitions_to_done() {
    let report = Arc::new(Mutex::new(Some(ScanReport {
        per_family: Default::default(),
        total_entries: 7,
    })));
    let driver = Arc::new(FakeDriver { report });
    let queue = CatalogScanQueue::new();
    let q2 = queue.clone();
    tokio::spawn(async move { q2.drive(driver.clone()).await });

    let id = queue
        .enqueue(mold_catalog::scanner::ScanOptions::default())
        .await
        .expect("enqueue");
    // Wait for completion.
    for _ in 0..50 {
        if let Some(CatalogScanStatus::Done { total_entries, .. }) = queue.status(&id).await {
            assert_eq!(total_entries, 7);
            return;
        }
        tokio::time::sleep(std::time::Duration::from_millis(20)).await;
    }
    panic!("scan never reached Done");
}

#[tokio::test]
async fn second_enqueue_while_running_is_rejected() {
    let report = Arc::new(Mutex::new(None)); // empty → driver hangs forever
    let driver = Arc::new(FakeDriver { report });
    let queue = CatalogScanQueue::new();
    let q2 = queue.clone();
    tokio::spawn(async move { q2.drive(driver.clone()).await });
    queue
        .enqueue(mold_catalog::scanner::ScanOptions::default())
        .await
        .expect("first enqueue");
    let second = queue
        .enqueue(mold_catalog::scanner::ScanOptions::default())
        .await;
    assert!(second.is_err(), "single-writer guarantee");
}

/// `active()` is what the web UI's `GET /api/catalog/refresh` handler
/// reads to decide whether to disable the refresh button. It must
/// return `None` for an idle queue and `Some((id, status))` for any
/// in-flight scan, regardless of who enqueued it.
#[tokio::test]
async fn active_returns_in_flight_scan_id_and_status() {
    let report = Arc::new(Mutex::new(None));
    let driver = Arc::new(FakeDriver { report });
    let queue = CatalogScanQueue::new();
    let q2 = queue.clone();
    tokio::spawn(async move { q2.drive(driver.clone()).await });

    assert!(queue.active().await.is_none(), "idle queue → no active");

    let id = queue
        .enqueue(mold_catalog::scanner::ScanOptions::default())
        .await
        .expect("enqueue");

    let (active_id, active_status) = queue.active().await.expect("active scan present");
    assert_eq!(active_id, id);
    assert!(
        matches!(
            active_status,
            CatalogScanStatus::Pending | CatalogScanStatus::Running { .. }
        ),
        "expected pending or running, got {:?}",
        active_status
    );
}

/// While a scan is running, `status()` must derive Running from the
/// live progress handle. The driver writes `families_done = 4`; the
/// queue must surface that — not a stale `families_done = 0`.
#[tokio::test]
async fn status_reflects_live_progress_during_running_scan() {
    let started = Arc::new(tokio::sync::Notify::new());
    let release = Arc::new(tokio::sync::Notify::new());

    struct ProgressDriver {
        started: Arc<tokio::sync::Notify>,
        release: Arc<tokio::sync::Notify>,
    }
    #[async_trait::async_trait]
    impl ScanDriver for ProgressDriver {
        async fn run(
            &self,
            _opts: mold_catalog::scanner::ScanOptions,
            progress: ProgressHandle,
        ) -> ScanReport {
            {
                let mut p = progress.lock().expect("progress");
                p.families_total = 9;
                p.families_done = 4;
                p.current_family = Some(mold_catalog::families::Family::Sdxl);
                p.current_stage = Some("hf");
            }
            self.started.notify_one();
            self.release.notified().await;
            ScanReport::default()
        }
    }

    let driver = Arc::new(ProgressDriver {
        started: started.clone(),
        release: release.clone(),
    });
    let queue = CatalogScanQueue::new();
    let q2 = queue.clone();
    tokio::spawn(async move { q2.drive(driver).await });

    let id = queue
        .enqueue(mold_catalog::scanner::ScanOptions::default())
        .await
        .expect("enqueue");
    started.notified().await;

    let status = queue.status(&id).await.expect("status while running");
    match status {
        CatalogScanStatus::Running {
            families_total,
            families_done,
            current_family,
            current_stage,
            started_at_ms,
        } => {
            assert_eq!(families_total, 9);
            assert_eq!(families_done, 4);
            assert_eq!(current_family.as_deref(), Some("sdxl"));
            assert_eq!(current_stage.as_deref(), Some("hf"));
            assert!(started_at_ms > 0);
        }
        other => panic!("expected Running with live counters, got {:?}", other),
    }

    release.notify_one();
}

// ── post_catalog_download — gate + companion auto-pull ──────────────────────

/// Build a catalog row with sane defaults so individual tests can override
/// only the fields under test. `download_recipe` ships a minimal but
/// schema-valid recipe (one synthetic file, no token) so the
/// `post_catalog_download` Civitai branch parses it without erroring; tests
/// that care about specific recipe content override `row.download_recipe`.
fn make_catalog_row(id: &str, source: &str, family: &str, engine_phase: i64) -> CatalogRow {
    let recipe = serde_json::json!({
        "files": [{
            "url": "http://test.invalid/primary.safetensors",
            "dest": "primary.safetensors",
            "sha256": null,
            "size_bytes": 1000,
        }],
        "needs_token": null,
    });
    CatalogRow {
        id: id.to_string(),
        source: source.to_string(),
        source_id: id.to_string(),
        name: id.to_string(),
        author: None,
        family: family.to_string(),
        family_role: "finetune".into(),
        sub_family: None,
        modality: "text-to-image".into(),
        kind: "checkpoint".into(),
        file_format: "safetensors".into(),
        bundling: "single-file".into(),
        size_bytes: Some(1_000),
        download_count: 0,
        rating: None,
        likes: 0,
        nsfw: 0,
        thumbnail_url: None,
        description: None,
        license: None,
        license_flags: None,
        tags: None,
        companions: None,
        download_recipe: recipe.to_string(),
        engine_phase,
        created_at: None,
        updated_at: None,
        added_at: 0,
    }
}

fn seeded_state(rows: &[CatalogRow]) -> AppState {
    let db = Arc::new(MetadataDb::open_in_memory().expect("in-memory DB"));
    // catalog_upsert is family-scoped; group rows by family before upserting.
    use std::collections::BTreeMap;
    let mut by_family: BTreeMap<String, Vec<CatalogRow>> = BTreeMap::new();
    for row in rows {
        by_family
            .entry(row.family.clone())
            .or_default()
            .push(row.clone());
    }
    for (family, batch) in by_family {
        db.catalog_upsert(&family, &batch).expect("upsert");
    }
    AppState::for_tests(db)
}

async fn invoke_download(state: AppState, id: &str) -> axum::http::Response<axum::body::Body> {
    post_catalog_download(State(state), Path(id.to_string()))
        .await
        .into_response()
}

/// Phase 2 (SD1.5 + SDXL single-file) entries become downloadable in 2.7.
/// Today the gate hard-rejects anything `>= 2` with 409. After the gate
/// drop they MUST flow past the gate — the response status must not be
/// 409 CONFLICT regardless of the downstream companion-pull logic.
#[tokio::test]
async fn post_catalog_download_no_longer_conflicts_for_engine_phase_2() {
    let row = make_catalog_row("cv:phase-2-test", "civitai", "sdxl", 2);
    let state = seeded_state(&[row]);
    let resp = invoke_download(state, "cv:phase-2-test").await;
    assert_ne!(
        resp.status(),
        StatusCode::CONFLICT,
        "engine_phase 2 must flow past the gate after 2.7",
    );
}

/// Phase 3+ (FLUX single-file, Z-Image, LTX-Video, ...) are still gated.
/// Dropping the gate to `>= 3` keeps phases 3 and beyond rejected.
#[tokio::test]
async fn post_catalog_download_still_conflicts_for_engine_phase_3() {
    let row = make_catalog_row("cv:phase-3-test", "civitai", "flux", 3);
    let state = seeded_state(&[row]);
    let resp = invoke_download(state, "cv:phase-3-test").await;
    assert_eq!(
        resp.status(),
        StatusCode::CONFLICT,
        "engine_phase 3 must remain gated until phase 3 lands",
    );
}

// ── Companion auto-pull (Round 2b) ──────────────────────────────────────────

use crate::catalog_api::enqueue_missing_companions;
use crate::downloads::DownloadQueue;
use mold_core::manifest::{find_manifest, storage_path};

/// Materialise every file declared by the companion's synthetic manifest
/// under `models_dir`, so the on-disk presence check sees a "fully pulled"
/// companion. Files are 1-byte stubs — the check only cares about presence.
fn stage_companion_on_disk(models_dir: &std::path::Path, name: &str) {
    let manifest = find_manifest(name).expect("companion manifest");
    for file in &manifest.files {
        let path = models_dir.join(storage_path(manifest, file));
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent).expect("mkdir");
        }
        std::fs::write(&path, b"x").expect("write stub");
    }
}

/// Write a `.pulling` marker for a companion to simulate a half-pulled
/// state — files might exist but the previous pull was interrupted before
/// the marker was cleaned up.
fn stage_pulling_marker(models_dir: &std::path::Path, name: &str) {
    let path = mold_core::download::pulling_marker_path_in(models_dir, name);
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent).expect("mkdir");
    }
    std::fs::write(&path, name).expect("write marker");
}

/// SDXL companions in the order they appear on a Civitai SDXL entry's
/// `companions` field — clip-l, clip-g, sdxl-vae. The companion-pull
/// helper must surface them in declaration order so the DownloadsDrawer
/// renders the same sequence the client requested.
#[tokio::test]
async fn companions_enqueued_in_declaration_order_when_none_present() {
    let tmp = tempfile::tempdir().expect("tempdir");
    let queue = DownloadQueue::new_for_test();
    let companions = serde_json::to_string(&["clip-l", "clip-g", "sdxl-vae"]).unwrap();

    let jobs = enqueue_missing_companions(Some(&companions), tmp.path(), &queue).await;

    let names: Vec<&str> = jobs.iter().map(|j| j.name.as_str()).collect();
    assert_eq!(names, vec!["clip-l", "clip-g", "sdxl-vae"]);
    for job in &jobs {
        assert!(!job.job_id.is_empty(), "{} must get a job id", job.name);
    }
}

/// On-disk presence check: when every file a companion declares is already
/// present under `models_dir`, the helper skips the enqueue entirely.
/// Companions that ARE missing still flow through.
#[tokio::test]
async fn companion_skipped_when_already_present_on_disk() {
    let tmp = tempfile::tempdir().expect("tempdir");
    stage_companion_on_disk(tmp.path(), "clip-l");

    let queue = DownloadQueue::new_for_test();
    let companions = serde_json::to_string(&["clip-l", "clip-g", "sdxl-vae"]).unwrap();

    let jobs = enqueue_missing_companions(Some(&companions), tmp.path(), &queue).await;

    let names: Vec<&str> = jobs.iter().map(|j| j.name.as_str()).collect();
    assert_eq!(
        names,
        vec!["clip-g", "sdxl-vae"],
        "clip-l is fully on disk and must be skipped",
    );
}

/// `.pulling` marker means a previous pull was interrupted. Treat as
/// "missing" even when the underlying files exist — the half-pulled
/// content might be corrupt and re-enqueuing is the safe call.
/// `DownloadQueue::enqueue` is idempotent against in-flight jobs, so a
/// concurrent retry won't double-pull.
#[tokio::test]
async fn companion_with_pulling_marker_is_re_enqueued() {
    let tmp = tempfile::tempdir().expect("tempdir");
    stage_companion_on_disk(tmp.path(), "clip-l");
    stage_pulling_marker(tmp.path(), "clip-l");

    let queue = DownloadQueue::new_for_test();
    let companions = serde_json::to_string(&["clip-l"]).unwrap();

    let jobs = enqueue_missing_companions(Some(&companions), tmp.path(), &queue).await;

    let names: Vec<&str> = jobs.iter().map(|j| j.name.as_str()).collect();
    assert_eq!(
        names,
        vec!["clip-l"],
        ".pulling marker must force a re-enqueue",
    );
}

/// Unknown canonical companion names (typos, future entries the build
/// doesn't recognise) are skipped without aborting the rest of the
/// auto-pull. The catalog scanner can ship new companion names ahead of
/// the binary supporting them.
#[tokio::test]
async fn unknown_companion_canonical_name_is_skipped() {
    let tmp = tempfile::tempdir().expect("tempdir");
    let queue = DownloadQueue::new_for_test();
    // `z-image-te` is reserved in the registry for phase 4 — there's no
    // synthetic manifest yet, so the helper should skip it gracefully.
    let companions = serde_json::to_string(&["clip-l", "z-image-te"]).unwrap();

    let jobs = enqueue_missing_companions(Some(&companions), tmp.path(), &queue).await;

    let names: Vec<&str> = jobs.iter().map(|j| j.name.as_str()).collect();
    assert_eq!(
        names,
        vec!["clip-l"],
        "unknown companion must not block the rest",
    );
}

/// Empty / missing companions field returns an empty job list — the
/// helper must not panic on `None` or an empty array. Phase-1 HF entries
/// land here.
#[tokio::test]
async fn empty_companions_returns_empty_job_list() {
    let tmp = tempfile::tempdir().expect("tempdir");
    let queue = DownloadQueue::new_for_test();

    let none_jobs = enqueue_missing_companions(None, tmp.path(), &queue).await;
    assert!(none_jobs.is_empty(), "None must yield empty list");

    let empty_jobs = enqueue_missing_companions(Some("[]"), tmp.path(), &queue).await;
    assert!(empty_jobs.is_empty(), "[] must yield empty list");
}

// ── Response schema (Round 3) ──────────────────────────────────────────────

async fn read_body_json(resp: axum::http::Response<axum::body::Body>) -> serde_json::Value {
    let body = axum::body::to_bytes(resp.into_body(), 1024 * 1024)
        .await
        .expect("body bytes");
    serde_json::from_slice(&body).unwrap_or_else(|e| panic!("body not JSON: {e}"))
}

/// 2.7 breaks the response shape: the bare `{ "job_ids": [...] }` form
/// is replaced with `{ "primary_job_id", "companion_jobs": [{name, job_id}] }`.
/// Phase-1 HF entries surface no companions but must still produce the
/// new shape so the web client can rely on the keys' existence.
#[tokio::test]
async fn download_response_has_companion_jobs_array() {
    let row = make_catalog_row("hf:phase-1-test", "hf", "flux", 1);
    let state = seeded_state(&[row]);
    let resp = invoke_download(state, "hf:phase-1-test").await;
    assert_eq!(resp.status(), StatusCode::ACCEPTED, "phase-1 must accept");
    let body = read_body_json(resp).await;
    assert!(
        body.get("companion_jobs").is_some(),
        "response must surface companion_jobs key, got {body}",
    );
    assert!(
        body.get("primary_job_id").is_some(),
        "response must surface primary_job_id key (may be null), got {body}",
    );
    assert!(
        body.get("job_ids").is_none(),
        "legacy job_ids key must be removed, got {body}",
    );
}

/// Civitai phase-2 entries (the whole point of 2.7's gate drop) must no
/// longer get the 501 NOT_IMPLEMENTED wall — they need to flow through
/// the companion auto-pull and surface companion_jobs in the response.
#[tokio::test]
async fn civitai_phase_2_no_longer_returns_not_implemented() {
    let mut row = make_catalog_row("cv:sdxl-pony-test", "civitai", "sdxl", 2);
    row.companions = Some(serde_json::to_string(&["clip-l", "clip-g", "sdxl-vae"]).expect("json"));
    let state = seeded_state(&[row]);
    let resp = invoke_download(state, "cv:sdxl-pony-test").await;
    assert_ne!(
        resp.status(),
        StatusCode::NOT_IMPLEMENTED,
        "civitai phase-2 must flow through after 2.7",
    );
    assert_eq!(
        resp.status(),
        StatusCode::ACCEPTED,
        "civitai phase-2 must return 202 Accepted",
    );
}

/// The companion-pull contract: every companion declared on the catalog
/// row must appear in the response's `companion_jobs` array (in order)
/// when none are present on disk yet. The web client renders these into
/// the DownloadsDrawer in the same order the user sees in the UI.
#[tokio::test]
async fn companion_jobs_in_response_match_row_companions_in_order() {
    let mut row = make_catalog_row("cv:sdxl-order-test", "civitai", "sdxl", 2);
    row.companions = Some(serde_json::to_string(&["clip-l", "clip-g", "sdxl-vae"]).expect("json"));
    let state = seeded_state(&[row]);
    let resp = invoke_download(state, "cv:sdxl-order-test").await;
    assert_eq!(resp.status(), StatusCode::ACCEPTED);
    let body = read_body_json(resp).await;
    let arr = body
        .get("companion_jobs")
        .and_then(|v| v.as_array())
        .cloned()
        .unwrap_or_default();
    let names: Vec<String> = arr
        .iter()
        .filter_map(|j| j.get("name").and_then(|n| n.as_str()).map(String::from))
        .collect();
    assert_eq!(names, vec!["clip-l", "clip-g", "sdxl-vae"]);
    for entry in &arr {
        assert!(
            entry
                .get("job_id")
                .and_then(|v| v.as_str())
                .map(|s| !s.is_empty())
                .unwrap_or(false),
            "each companion_jobs entry must include a non-empty job_id, got {entry}",
        );
    }
}

/// Option A' contract: `cv:` entries must surface a real `primary_job_id`
/// instead of `null`. Before 2.8 the queue couldn't accept a recipe so this
/// field was always `null` for civitai sources; the SPA's downloads drawer
/// had to skip rendering the primary job and instructions said "run mold
/// pull from a terminal." After 2.8 the recipe path makes this real.
#[tokio::test]
async fn civitai_post_catalog_download_returns_real_primary_job_id() {
    let row = make_catalog_row("cv:sdxl-primary-test", "civitai", "sdxl", 2);
    let state = seeded_state(&[row]);
    let resp = invoke_download(state, "cv:sdxl-primary-test").await;
    assert_eq!(resp.status(), StatusCode::ACCEPTED);
    let body = read_body_json(resp).await;
    let primary = body.get("primary_job_id");
    assert!(
        primary.is_some(),
        "primary_job_id field must be present, got {body}"
    );
    let v = primary.unwrap();
    assert!(
        !v.is_null(),
        "primary_job_id must NOT be null for civitai entries (Option A'), got {body}"
    );
    let id = v.as_str().expect("primary_job_id must be a string");
    assert!(
        !id.is_empty(),
        "primary_job_id must be non-empty, got {body}"
    );
}

/// Civitai recipes flagged `needs_token: Civitai` must surface a 401 when
/// `CIVITAI_TOKEN` is unset, with a remediation pointing at the env var.
/// Test serializes env-var access with a process-global mutex (matches the
/// `HF_TOKEN_LOCK` pattern in mold-core::download tests).
static CIVITAI_TOKEN_TEST_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

#[tokio::test]
async fn civitai_recipe_with_needs_token_returns_unauthorized_when_no_env_var() {
    let _guard = CIVITAI_TOKEN_TEST_LOCK.lock().unwrap();
    let original = std::env::var("CIVITAI_TOKEN").ok();
    std::env::remove_var("CIVITAI_TOKEN");

    let mut row = make_catalog_row("cv:gated-test", "civitai", "sdxl", 2);
    let recipe = serde_json::json!({
        "files": [{
            "url": "http://test.invalid/primary.safetensors",
            "dest": "primary.safetensors",
            "sha256": null,
            "size_bytes": 1000,
        }],
        "needs_token": "civitai",
    });
    row.download_recipe = recipe.to_string();
    let state = seeded_state(&[row]);
    let resp = invoke_download(state, "cv:gated-test").await;

    if let Some(v) = original {
        std::env::set_var("CIVITAI_TOKEN", v);
    }

    assert_eq!(
        resp.status(),
        StatusCode::UNAUTHORIZED,
        "needs_token:civitai without CIVITAI_TOKEN must surface 401",
    );
    let body_bytes = axum::body::to_bytes(resp.into_body(), 4096)
        .await
        .expect("read body")
        .to_vec();
    let body = String::from_utf8_lossy(&body_bytes);
    assert!(
        body.contains("CIVITAI_TOKEN"),
        "401 body should name the env var: {body}"
    );
}
