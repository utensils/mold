// Tests use `std::sync::Mutex<()>` to serialize process-global env-var
// mutations; holding the guard across `.await` is intentional under the
// current-thread tokio test runtime.
#![allow(clippy::await_holding_lock)]

#[cfg(test)]
mod tests {
    use axum::{
        body::Body,
        http::{Request, StatusCode},
    };
    use mold_core::chain::{ChainRequest, ChainStage, TransitionMode};
    use mold_core::{
        chain_job::{ChainJobManifest, ChainJobState, JobDirLayout, RetakeMode, StageState},
        GenerateRequest, GenerateResponse, ImageData, OutputFormat,
    };
    use mold_inference::progress::ProgressCallback;
    use mold_inference::InferenceEngine;
    use sha2::{Digest, Sha256};
    use std::net::IpAddr;
    use std::path::PathBuf;
    use std::sync::{
        atomic::{AtomicBool, AtomicUsize, Ordering},
        Arc, Condvar, Mutex, RwLock,
    };
    use std::time::Duration;
    use tower::ServiceExt;

    use crate::{routes::create_router, state::AppState};

    /// Serialize tests that mutate MOLD_MODELS_DIR env var.
    /// Uses std::sync::Mutex (not tokio) so it works across independent
    /// tokio runtimes that #[tokio::test] creates per test.
    fn env_lock() -> &'static std::sync::Mutex<()> {
        static ENV_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());
        &ENV_LOCK
    }

    /// Parse response body as JSON and return the value.
    async fn json_body(resp: axum::http::Response<Body>) -> serde_json::Value {
        let body = axum::body::to_bytes(resp.into_body(), 1024 * 1024)
            .await
            .unwrap();
        serde_json::from_slice(&body).unwrap()
    }

    #[derive(Default)]
    struct GenerateBlocker {
        entered: AtomicBool,
        released: Mutex<bool>,
        released_cv: Condvar,
    }

    impl GenerateBlocker {
        fn release(&self) {
            let mut released = self.released.lock().unwrap();
            *released = true;
            self.released_cv.notify_all();
        }
    }

    /// Minimal mock engine for testing routes without loading models.
    struct MockEngine {
        loaded: bool,
        fail: bool,
        empty_images: bool,
        load_count: Arc<AtomicUsize>,
        load_delay: Duration,
        progress_set_count: Arc<AtomicUsize>,
        progress_clear_count: Arc<AtomicUsize>,
        generate_blocker: Option<Arc<GenerateBlocker>>,
        /// When set, load() emits progress events through the stored callback.
        emit_load_progress: bool,
        progress_callback: Option<ProgressCallback>,
    }

    impl MockEngine {
        fn ready() -> Self {
            Self {
                loaded: true,
                fail: false,
                empty_images: false,
                load_count: Arc::new(AtomicUsize::new(0)),
                load_delay: Duration::from_millis(0),
                progress_set_count: Arc::new(AtomicUsize::new(0)),
                progress_clear_count: Arc::new(AtomicUsize::new(0)),
                generate_blocker: None,
                emit_load_progress: false,
                progress_callback: None,
            }
        }
        fn failing() -> Self {
            Self {
                loaded: true,
                fail: true,
                empty_images: false,
                load_count: Arc::new(AtomicUsize::new(0)),
                load_delay: Duration::from_millis(0),
                progress_set_count: Arc::new(AtomicUsize::new(0)),
                progress_clear_count: Arc::new(AtomicUsize::new(0)),
                generate_blocker: None,
                emit_load_progress: false,
                progress_callback: None,
            }
        }
        fn empty_images() -> Self {
            Self {
                loaded: true,
                fail: false,
                empty_images: true,
                load_count: Arc::new(AtomicUsize::new(0)),
                load_delay: Duration::from_millis(0),
                progress_set_count: Arc::new(AtomicUsize::new(0)),
                progress_clear_count: Arc::new(AtomicUsize::new(0)),
                generate_blocker: None,
                emit_load_progress: false,
                progress_callback: None,
            }
        }
        fn unloaded(load_count: Arc<AtomicUsize>, load_delay: Duration) -> Self {
            Self {
                loaded: false,
                fail: false,
                empty_images: false,
                load_count,
                load_delay,
                progress_set_count: Arc::new(AtomicUsize::new(0)),
                progress_clear_count: Arc::new(AtomicUsize::new(0)),
                generate_blocker: None,
                emit_load_progress: false,
                progress_callback: None,
            }
        }

        fn tracked_progress(
            progress_set_count: Arc<AtomicUsize>,
            progress_clear_count: Arc<AtomicUsize>,
        ) -> Self {
            Self {
                loaded: true,
                fail: false,
                empty_images: false,
                load_count: Arc::new(AtomicUsize::new(0)),
                load_delay: Duration::from_millis(0),
                progress_set_count,
                progress_clear_count,
                generate_blocker: None,
                emit_load_progress: false,
                progress_callback: None,
            }
        }

        fn blocking_generate(blocker: Arc<GenerateBlocker>) -> Self {
            Self {
                loaded: true,
                fail: false,
                empty_images: false,
                load_count: Arc::new(AtomicUsize::new(0)),
                load_delay: Duration::from_millis(0),
                progress_set_count: Arc::new(AtomicUsize::new(0)),
                progress_clear_count: Arc::new(AtomicUsize::new(0)),
                generate_blocker: Some(blocker),
                emit_load_progress: false,
                progress_callback: None,
            }
        }

        /// Create an unloaded engine that emits progress events during load(),
        /// simulating FP8→Q8 conversion status messages.
        fn unloaded_with_progress() -> Self {
            Self {
                loaded: false,
                fail: false,
                empty_images: false,
                load_count: Arc::new(AtomicUsize::new(0)),
                load_delay: Duration::from_millis(0),
                progress_set_count: Arc::new(AtomicUsize::new(0)),
                progress_clear_count: Arc::new(AtomicUsize::new(0)),
                generate_blocker: None,
                emit_load_progress: true,
                progress_callback: None,
            }
        }
    }

    impl InferenceEngine for MockEngine {
        fn generate(&mut self, req: &GenerateRequest) -> anyhow::Result<GenerateResponse> {
            if let Some(blocker) = &self.generate_blocker {
                blocker.entered.store(true, Ordering::SeqCst);
                let released = blocker.released.lock().unwrap();
                let _released = blocker
                    .released_cv
                    .wait_while(released, |released| !*released)
                    .unwrap();
            }
            if self.fail {
                anyhow::bail!("mock engine error");
            }
            let images = if self.empty_images {
                vec![]
            } else {
                vec![ImageData {
                    data: minimal_png(),
                    format: req.resolved_output_format(),
                    width: req.width,
                    height: req.height,
                    index: 0,
                }]
            };
            Ok(GenerateResponse {
                images,
                generation_time_ms: 1,
                model: req.model.clone(),
                seed_used: req.seed.unwrap_or(42),
                video: None,
                gpu: None,
            })
        }

        fn model_name(&self) -> &str {
            "mock-model"
        }

        fn is_loaded(&self) -> bool {
            self.loaded
        }

        fn load(&mut self) -> anyhow::Result<()> {
            self.load_count.fetch_add(1, Ordering::SeqCst);
            if self.emit_load_progress {
                if let Some(ref cb) = self.progress_callback {
                    cb(mold_inference::progress::ProgressEvent::Info {
                        message: "Converting FP8 checkpoint to Q8 GGUF cache (one-time, may take a few minutes)".to_string(),
                    });
                    cb(mold_inference::progress::ProgressEvent::StageStart {
                        name: "Loading transformer (GPU, quantized)".to_string(),
                    });
                }
            }
            if !self.load_delay.is_zero() {
                std::thread::sleep(self.load_delay);
            }
            self.loaded = true;
            Ok(())
        }

        fn set_on_progress(&mut self, callback: ProgressCallback) {
            self.progress_set_count.fetch_add(1, Ordering::SeqCst);
            self.progress_callback = Some(callback);
        }

        fn clear_on_progress(&mut self) {
            self.progress_clear_count.fetch_add(1, Ordering::SeqCst);
            self.progress_callback = None;
        }
    }

    /// Create an app with a running queue worker (needed for generate endpoints).
    fn app_with(engine: MockEngine) -> axum::Router {
        let (state, rx) = AppState::with_engine_and_queue(engine);
        let worker_state = state.clone();
        tokio::spawn(crate::queue::run_queue_worker(rx, worker_state));
        create_router(state)
    }

    /// Create an app from pre-built state. Caller must ensure queue worker is
    /// running if generate endpoints will be tested.
    fn app_with_state(state: AppState) -> axum::Router {
        create_router(state)
    }

    fn app_empty() -> axum::Router {
        let (tx, _rx) = tokio::sync::mpsc::channel(16);
        let queue = crate::state::QueueHandle::new(tx);
        let gpu_pool = std::sync::Arc::new(crate::gpu_pool::GpuPool {
            workers: Vec::new(),
        });
        app_with_state(AppState::empty(
            mold_core::Config::default(),
            queue,
            gpu_pool,
            200,
        ))
    }

    struct MoldHomeGuard {
        _lock: std::sync::MutexGuard<'static, ()>,
        previous: Option<std::ffi::OsString>,
    }

    impl MoldHomeGuard {
        fn set(path: &std::path::Path) -> Self {
            let lock = env_lock().lock().unwrap();
            let previous = std::env::var_os("MOLD_HOME");
            std::env::set_var("MOLD_HOME", path);
            Self {
                _lock: lock,
                previous,
            }
        }
    }

    impl Drop for MoldHomeGuard {
        fn drop(&mut self) {
            match &self.previous {
                Some(value) => std::env::set_var("MOLD_HOME", value),
                None => std::env::remove_var("MOLD_HOME"),
            }
        }
    }

    fn app_with_chain_db(db: mold_db::MetadataDb) -> axum::Router {
        app_with_chain_handle(
            db,
            Arc::new(crate::chain_job_runner::ChainJobRunnerHandle::inert_for_tests()),
        )
    }

    fn app_with_chain_handle(
        db: mold_db::MetadataDb,
        handle: Arc<crate::chain_job_runner::ChainJobRunnerHandle>,
    ) -> axum::Router {
        let mut state = AppState::for_tests();
        state.metadata_db = Arc::new(Some(db));
        state.chain_jobs = Some(handle);
        app_with_state(state)
    }

    fn route_chain_stage(prompt: &str, transition: TransitionMode) -> ChainStage {
        ChainStage {
            prompt: prompt.into(),
            frames: 9,
            source_image: None,
            negative_prompt: None,
            seed_offset: None,
            transition,
            fade_frames: Some(2),
            model: None,
            loras: vec![],
            references: vec![],
        }
    }

    fn route_chain_request() -> ChainRequest {
        ChainRequest {
            model: "ltx-2-19b-distilled:fp8".into(),
            stages: vec![route_chain_stage("first shot", TransitionMode::Smooth)],
            motion_tail_frames: 1,
            width: 64,
            height: 48,
            fps: 8,
            seed: Some(42),
            steps: 2,
            guidance: 1.0,
            strength: 1.0,
            output_format: OutputFormat::Mp4,
            placement: None,
            prompt: None,
            total_frames: None,
            clip_frames: None,
            source_image: None,
            enable_audio: None,
        }
    }

    fn seed_chain_job(
        db: &mold_db::MetadataDb,
        mold_home: &std::path::Path,
        id: &str,
        state: ChainJobState,
    ) -> PathBuf {
        let req = route_chain_request();
        seed_chain_job_with_request(db, mold_home, id, state, &req)
    }

    fn seed_chain_job_with_request(
        db: &mold_db::MetadataDb,
        mold_home: &std::path::Path,
        id: &str,
        state: ChainJobState,
        req: &ChainRequest,
    ) -> PathBuf {
        let job_dir = mold_home.join("jobs").join(id);
        std::fs::create_dir_all(&job_dir).unwrap();
        let mut manifest = ChainJobManifest::new(id.to_string(), 1_783_200_000_000, req).unwrap();
        match state {
            ChainJobState::Completed => {
                for stage in &mut manifest.stage_status {
                    stage.state = StageState::Completed;
                }
            }
            ChainJobState::Running => {
                manifest.stage_status[0].state = StageState::Running;
            }
            _ => {}
        }
        manifest.write_atomic(&job_dir).unwrap();
        let stage_count = req.stages.len() as u32;
        let current_stage = if state == ChainJobState::Completed {
            stage_count
        } else {
            0
        };

        let now = 1_783_200_000_000_i64;
        mold_db::chain_jobs::insert_job(
            db,
            &mold_db::chain_jobs::ChainJobRow {
                id: id.into(),
                state,
                model: req.model.clone(),
                request_json: serde_json::to_string(&req).unwrap(),
                job_dir: job_dir.clone(),
                stage_count,
                current_stage,
                error: None,
                created_at_ms: now,
                updated_at_ms: now,
                finalized_at_ms: None,
            },
        )
        .unwrap();
        for stage in &manifest.stage_status {
            mold_db::chain_jobs::upsert_stage(
                db,
                &mold_db::chain_jobs::ChainJobStageRow {
                    job_id: id.into(),
                    stage_idx: stage.idx,
                    state: stage.state,
                    seed: stage.seed,
                    frames_emitted: None,
                    generation_time_ms: None,
                    segment_rel_path: None,
                    error: None,
                    updated_at_ms: now,
                },
            )
            .unwrap();
        }
        JobDirLayout::new(job_dir.clone()).ensure_root().unwrap();
        job_dir
    }

    fn gpu_worker_stub(ordinal: usize) -> Arc<crate::gpu_pool::GpuWorker> {
        let (job_tx, _job_rx) = std::sync::mpsc::sync_channel(1);
        Arc::new(crate::gpu_pool::GpuWorker {
            gpu: mold_inference::device::DiscoveredGpu {
                ordinal,
                name: format!("gpu{ordinal}"),
                total_vram_bytes: 24_000_000_000,
                free_vram_bytes: 24_000_000_000,
            },
            model_cache: Arc::new(Mutex::new(crate::model_cache::ModelCache::new(3))),
            active_generation: Arc::new(RwLock::new(None)),
            model_load_lock: Arc::new(Mutex::new(())),
            shared_pool: Arc::new(Mutex::new(mold_inference::shared_pool::SharedPool::new())),
            in_flight: AtomicUsize::new(0),
            consecutive_failures: AtomicUsize::new(0),
            degraded_until: RwLock::new(None),
            job_tx,
        })
    }

    fn app_with_worker_pool(engine: MockEngine, ordinals: &[usize]) -> axum::Router {
        let mut state = AppState::with_engine(engine);
        state.gpu_pool = Arc::new(crate::gpu_pool::GpuPool {
            workers: ordinals.iter().copied().map(gpu_worker_stub).collect(),
        });
        create_router(state)
    }

    fn generate_body(prompt: &str, width: u32, height: u32) -> String {
        // Use "mock-model" to match MockEngine::model_name() — avoids hot-swap path.
        format!(
            r#"{{"prompt":"{prompt}","model":"mock-model","width":{width},"height":{height},"steps":4,"batch_size":1,"output_format":"png"}}"#
        )
    }

    fn test_models_dir(name: &str) -> PathBuf {
        let unique = format!(
            "mold-server-routes-{name}-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        );
        std::env::temp_dir().join(unique)
    }

    fn populate_manifest_files(root: &std::path::Path, model: &str) {
        let manifest = mold_core::manifest::find_manifest(model).unwrap();
        for file in &manifest.files {
            let path = root.join(mold_core::manifest::storage_path(manifest, file));
            if let Some(parent) = path.parent() {
                std::fs::create_dir_all(parent).unwrap();
            }
            std::fs::write(&path, b"test").unwrap();
            // Stamp a `.sha256-verified` marker so the post-B3 completeness
            // check accepts these 4-byte stubs as installed (size mismatch
            // would otherwise reject them as truncated).
            mold_core::download::write_sha256_marker(&path, "deadbeef").unwrap();
        }
    }

    /// Returns a valid 1x1 PNG (8-byte signature + IHDR + IDAT + IEND).
    fn minimal_png() -> Vec<u8> {
        vec![
            0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A, // signature
            0x00, 0x00, 0x00, 0x0D, 0x49, 0x48, 0x44, 0x52, // IHDR length + type
            0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0x01, // 1x1
            0x08, 0x02, 0x00, 0x00, 0x00, 0x90, 0x77, 0x53, 0xDE, // bit depth, color, CRC
            0x00, 0x00, 0x00, 0x0C, 0x49, 0x44, 0x41, 0x54, // IDAT length + type
            0x08, 0xD7, 0x63, 0xF8, 0xCF, 0xC0, 0x00, 0x00, 0x00, 0x02, 0x00,
            0x01, // compressed
            0xE2, 0x21, 0xBC, 0x33, // IDAT CRC
            0x00, 0x00, 0x00, 0x00, 0x49, 0x45, 0x4E, 0x44, 0xAE, 0x42, 0x60, 0x82, // IEND
        ]
    }

    // ── /health ──────────────────────────────────────────────────────────────

    #[tokio::test]
    async fn health_returns_200() {
        let app = app_with(MockEngine::ready());
        let resp = app
            .oneshot(Request::get("/health").body(Body::empty()).unwrap())
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
    }

    #[tokio::test]
    async fn health_when_no_model() {
        let app = app_empty();
        let resp = app
            .oneshot(Request::get("/health").body(Body::empty()).unwrap())
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
    }

    // ── /api/status ──────────────────────────────────────────────────────────

    #[tokio::test]
    async fn status_returns_json() {
        let app = app_with(MockEngine::ready());
        let resp = app
            .oneshot(Request::get("/api/status").body(Body::empty()).unwrap())
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
        let ct = resp
            .headers()
            .get("content-type")
            .unwrap()
            .to_str()
            .unwrap();
        assert!(ct.contains("application/json"));
    }

    // ── /api/queue ───────────────────────────────────────────────────────────

    #[tokio::test]
    async fn queue_returns_empty_listing_on_idle_server() {
        // No jobs in flight → snapshot is an empty array. Wire contract:
        // the response is `{ "entries": [] }`, NOT a bare array — extra
        // top-level fields can be added later without a breaking change.
        let app = app_empty();
        let resp = app
            .oneshot(Request::get("/api/queue").body(Body::empty()).unwrap())
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
        let body = json_body(resp).await;
        assert_eq!(body["entries"], serde_json::json!([]));
    }

    #[tokio::test]
    async fn queue_lists_registered_jobs_in_fifo_order_with_running_state_and_gpu() {
        // Hand-build state so we can poke the registry without standing up
        // a real generation flow. Locks in the wire shape that the SPA's
        // reconciliation poller depends on.
        let (state, _rx) = AppState::with_engine_and_queue(MockEngine::ready());
        state.job_registry.register("aaaa", "flux-dev:fp16");
        state.job_registry.register("bbbb", "sdxl:q8");
        state.job_registry.mark_running("aaaa", Some(0));

        let app = app_with_state(state);
        let resp = app
            .oneshot(Request::get("/api/queue").body(Body::empty()).unwrap())
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
        let body = json_body(resp).await;
        let entries = body["entries"].as_array().expect("entries array");
        assert_eq!(entries.len(), 2);

        // Position 0 — the running job. Carries `gpu`.
        assert_eq!(entries[0]["id"], "aaaa");
        assert_eq!(entries[0]["state"], "running");
        assert_eq!(entries[0]["position"], 0);
        assert_eq!(entries[0]["gpu"], 0);

        // Position 1 — still queued. `gpu` is omitted from the wire shape
        // (clients shouldn't see `"gpu": null` and infer GPU 0).
        assert_eq!(entries[1]["id"], "bbbb");
        assert_eq!(entries[1]["state"], "queued");
        assert_eq!(entries[1]["position"], 1);
        assert!(
            entries[1].get("gpu").is_none(),
            "queued rows must not emit a `gpu` field, got: {}",
            entries[1]
        );
    }

    #[tokio::test]
    async fn queue_lists_target_gpu_for_queued_jobs() {
        let (state, _rx) = AppState::with_engine_and_queue(MockEngine::ready());
        state
            .job_registry
            .register_with_target_gpu("aaaa", "flux-dev:fp16", Some(1));

        let app = app_with_state(state);
        let resp = app
            .oneshot(Request::get("/api/queue").body(Body::empty()).unwrap())
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
        let body = json_body(resp).await;
        let entries = body["entries"].as_array().expect("entries array");
        assert_eq!(entries[0]["state"], "queued");
        assert_eq!(entries[0]["target_gpu"], 1);
        assert!(
            entries[0].get("gpu").is_none(),
            "queued rows still must not emit running gpu: {}",
            entries[0]
        );
    }

    #[tokio::test]
    async fn patch_queue_target_gpu_updates_queued_job_and_allows_auto() {
        let (state, _rx) = AppState::with_engine_and_queue(MockEngine::ready());
        assert!(
            state.gpu_pool.workers.is_empty(),
            "test assumes explicit empty worker pool"
        );
        state.job_registry.register("aaaa", "flux-dev:fp16");

        let app = app_with_state(state);
        let resp = app
            .oneshot(
                Request::patch("/api/queue/aaaa")
                    .header("content-type", "application/json")
                    .body(Body::from(r#"{"target_gpu":null}"#))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
        let body = json_body(resp).await;
        assert_eq!(body["id"], "aaaa");
        assert!(body.get("target_gpu").is_none());
    }

    #[tokio::test]
    async fn patch_queue_target_gpu_rejects_already_running_jobs() {
        let (state, _rx) = AppState::with_engine_and_queue(MockEngine::ready());
        state.job_registry.register("aaaa", "flux-dev:fp16");
        state.job_registry.mark_running("aaaa", Some(0));

        let app = app_with_state(state);
        let resp = app
            .oneshot(
                Request::patch("/api/queue/aaaa")
                    .header("content-type", "application/json")
                    .body(Body::from(r#"{"target_gpu":null}"#))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::CONFLICT);
        let body = json_body(resp).await;
        assert_eq!(body["code"], "QUEUE_JOB_RUNNING");
    }

    #[tokio::test]
    async fn status_when_no_model() {
        let app = app_empty();
        let resp = app
            .oneshot(Request::get("/api/status").body(Body::empty()).unwrap())
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::OK);

        let body = axum::body::to_bytes(resp.into_body(), 1024 * 1024)
            .await
            .unwrap();
        let status: serde_json::Value = serde_json::from_slice(&body).unwrap();
        assert_eq!(status["models_loaded"], serde_json::json!([]));
        assert_eq!(status["busy"], serde_json::json!(false));
        assert_eq!(status["current_generation"], serde_json::Value::Null);
    }

    #[tokio::test]
    async fn status_multi_gpu_current_generation_includes_prompt_hash_and_timestamp() {
        let worker = gpu_worker_stub(1);
        *worker.active_generation.write().unwrap() = Some(crate::gpu_pool::ActiveGeneration {
            model: "flux-dev:q4".to_string(),
            prompt_sha256: "abc123".to_string(),
            started_at_unix_ms: 1_700_000_000_000,
            started_at: std::time::Instant::now(),
        });

        let mut state = AppState::with_engine(MockEngine::ready());
        state.gpu_pool = Arc::new(crate::gpu_pool::GpuPool {
            workers: vec![worker],
        });
        let app = app_with_state(state);

        let resp = app
            .oneshot(Request::get("/api/status").body(Body::empty()).unwrap())
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::OK);

        let body = axum::body::to_bytes(resp.into_body(), 1024 * 1024)
            .await
            .unwrap();
        let status: serde_json::Value = serde_json::from_slice(&body).unwrap();
        assert_eq!(status["current_generation"]["model"], "flux-dev:q4");
        assert_eq!(status["current_generation"]["prompt_sha256"], "abc123");
        assert_eq!(
            status["current_generation"]["started_at_unix_ms"],
            serde_json::json!(1_700_000_000_000_u64)
        );
        assert_eq!(status["gpus"][0]["ordinal"], serde_json::json!(1));
    }

    #[tokio::test]
    async fn status_includes_hostname_and_memory_status() {
        let app = app_empty();
        let resp = app
            .oneshot(Request::get("/api/status").body(Body::empty()).unwrap())
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::OK);

        let body = axum::body::to_bytes(resp.into_body(), 1024 * 1024)
            .await
            .unwrap();
        let status: mold_core::ServerStatus = serde_json::from_slice(&body).unwrap();
        // hostname should be populated from the OS (non-empty on any real machine)
        assert!(
            status.hostname.is_some(),
            "server should report its hostname"
        );
        assert!(
            !status.hostname.as_ref().unwrap().is_empty(),
            "hostname should not be empty"
        );
        // memory_status may be None on CI (no GPU, no macOS vm_stat) — just verify it
        // deserializes without error (the field exists in the response)
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn status_does_not_block_during_generation() {
        let blocker = Arc::new(GenerateBlocker::default());
        let app = app_with(MockEngine::blocking_generate(blocker.clone()));

        let generate_task = tokio::spawn({
            let app = app.clone();
            async move {
                app.oneshot(
                    Request::post("/api/generate")
                        .header("content-type", "application/json")
                        .body(Body::from(generate_body("a cat", 768, 768)))
                        .unwrap(),
                )
                .await
                .unwrap()
            }
        });

        tokio::time::timeout(Duration::from_secs(1), async {
            while !blocker.entered.load(Ordering::SeqCst) {
                tokio::time::sleep(Duration::from_millis(10)).await;
            }
        })
        .await
        .expect("generate should enter the mock engine");

        let resp = tokio::time::timeout(
            Duration::from_millis(200),
            app.clone()
                .oneshot(Request::get("/api/status").body(Body::empty()).unwrap()),
        )
        .await
        .expect("/api/status should not block on active generation")
        .unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
        let status = json_body(resp).await;
        assert_eq!(status["busy"], serde_json::json!(true));
        assert_eq!(status["current_generation"]["model"], "mock-model");
        assert_eq!(
            status["current_generation"]["prompt_sha256"],
            serde_json::json!(format!("{:x}", Sha256::digest("a cat".as_bytes())))
        );
        assert!(
            status["current_generation"]["started_at_unix_ms"]
                .as_u64()
                .unwrap()
                > 0
        );

        blocker.release();
        let generate_resp = generate_task.await.unwrap();
        assert_eq!(generate_resp.status(), StatusCode::OK);
    }

    // ── /api/models ──────────────────────────────────────────────────────────

    #[tokio::test]
    async fn list_models_returns_json_array() {
        let app = app_with(MockEngine::ready());
        let resp = app
            .oneshot(Request::get("/api/models").body(Body::empty()).unwrap())
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
    }

    #[tokio::test]
    async fn list_models_uses_manifest_defaults_for_unpulled() {
        let app = app_with(MockEngine::ready());
        let resp = app
            .oneshot(Request::get("/api/models").body(Body::empty()).unwrap())
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::OK);

        let body = axum::body::to_bytes(resp.into_body(), 1024 * 1024)
            .await
            .unwrap();
        let models: Vec<serde_json::Value> = serde_json::from_slice(&body).unwrap();

        // SD1.5 model should have manifest defaults (512x512, guidance 7.5, 25 steps)
        let sd15 = models.iter().find(|m| m["name"] == "sd15:fp16");
        if let Some(sd15) = sd15 {
            assert_eq!(sd15["default_width"], 512, "SD1.5 width should be 512");
            assert_eq!(sd15["default_height"], 512, "SD1.5 height should be 512");
            assert_eq!(sd15["default_steps"], 25, "SD1.5 steps should be 25");
            assert_eq!(
                sd15["default_guidance"], 7.5,
                "SD1.5 guidance should be 7.5"
            );
        }

        // FLUX schnell should have manifest defaults (1024x1024, guidance 0.0, 4 steps)
        let schnell = models.iter().find(|m| m["name"] == "flux-schnell:q8");
        if let Some(schnell) = schnell {
            assert_eq!(schnell["default_width"], 1024);
            assert_eq!(schnell["default_height"], 1024);
            assert_eq!(schnell["default_steps"], 4);
        }
    }

    #[tokio::test]
    #[allow(clippy::await_holding_lock)]
    async fn list_models_reports_server_disk_and_remaining_download_bytes() {
        let _lock = env_lock().lock().unwrap_or_else(|e| e.into_inner());
        let models_dir = test_models_dir("remote-catalog");
        populate_manifest_files(&models_dir, "flux-schnell:q8");
        std::env::set_var("MOLD_MODELS_DIR", &models_dir);

        let app = app_empty();
        let resp = app
            .oneshot(Request::get("/api/models").body(Body::empty()).unwrap())
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::OK);

        let body = axum::body::to_bytes(resp.into_body(), 1024 * 1024)
            .await
            .unwrap();
        let models: Vec<serde_json::Value> = serde_json::from_slice(&body).unwrap();

        let downloaded = models
            .iter()
            .find(|m| m["name"] == "flux-schnell:q8")
            .expect("flux-schnell:q8 should be present");
        assert_eq!(downloaded["downloaded"], serde_json::json!(true));
        assert!(
            downloaded["remaining_download_bytes"].is_number(),
            "downloaded model should expose remaining download bytes"
        );
        assert!(
            downloaded["disk_usage_bytes"].as_u64().unwrap() > 0,
            "downloaded model should report server disk usage"
        );

        let available = models
            .iter()
            .find(|m| m["name"] == "flux-dev:q8")
            .expect("flux-dev:q8 should be present");
        assert_eq!(available["downloaded"], serde_json::json!(false));
        assert!(
            available["remaining_download_bytes"].is_number(),
            "available model should expose server-side remaining bytes even when fully cached"
        );

        std::env::remove_var("MOLD_MODELS_DIR");
        let _ = std::fs::remove_dir_all(models_dir);
    }

    #[tokio::test]
    #[allow(clippy::await_holding_lock)]
    async fn generate_estimate_returns_server_memory_estimate() {
        let _lock = env_lock().lock().unwrap_or_else(|e| e.into_inner());
        let models_dir = test_models_dir("estimate");
        populate_manifest_files(&models_dir, "sdxl-base:fp16");
        std::env::set_var("MOLD_MODELS_DIR", &models_dir);

        let app = app_empty();
        let body = serde_json::json!({
            "prompt": "a cat",
            "model": "sdxl-base:fp16",
            "width": 1024,
            "height": 1024,
            "steps": 20,
            "guidance": 7.5,
            "batch_size": 1,
            "output_format": "png"
        });
        let resp = app
            .clone()
            .oneshot(
                Request::post("/api/generate/estimate")
                    .header("content-type", "application/json")
                    .body(Body::from(body.to_string()))
                    .unwrap(),
            )
            .await
            .unwrap();

        assert_eq!(resp.status(), StatusCode::OK);
        let json = json_body(resp).await;
        assert_eq!(json["model"], "sdxl-base:fp16");
        assert!(json["peak_memory_bytes"].as_u64().unwrap() > 0);
        assert!(json["activation_memory_bytes"].as_u64().unwrap() > 0);
        assert!(json["load_strategy"].as_str().is_some());
        let base_peak = json["peak_memory_bytes"].as_u64().unwrap();

        let larger_body = serde_json::json!({
            "prompt": "a cat",
            "model": "sdxl-base:fp16",
            "width": 1024,
            "height": 1024,
            "steps": 20,
            "guidance": 7.5,
            "negative_prompt": "blurry",
            "batch_size": 2,
            "source_image": "aW1hZ2U=",
            "mask_image": "bWFzaw==",
            "control_image": "Y29udHJvbA==",
            "control_model": "controlnet-canny-sd15",
            "upscale_model": "real-esrgan-x4plus:fp16",
            "loras": [{"path": "/tmp/style.safetensors", "scale": 0.8}],
            "output_format": "png"
        });
        let larger_resp = app
            .oneshot(
                Request::post("/api/generate/estimate")
                    .header("content-type", "application/json")
                    .body(Body::from(larger_body.to_string()))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(larger_resp.status(), StatusCode::OK);
        let larger_json = json_body(larger_resp).await;
        assert!(
            larger_json["peak_memory_bytes"].as_u64().unwrap() > base_peak,
            "request-sensitive knobs should raise the estimate: base={base_peak} larger={}",
            larger_json["peak_memory_bytes"]
        );

        std::env::remove_var("MOLD_MODELS_DIR");
    }

    #[tokio::test]
    #[allow(clippy::await_holding_lock)]
    async fn model_components_reports_present_and_missing_manifest_assets() {
        let _lock = env_lock().lock().unwrap_or_else(|e| e.into_inner());
        let models_dir = test_models_dir("components");
        let manifest = mold_core::manifest::find_manifest("sdxl-base:fp16").unwrap();
        for file in manifest.files.iter().take(1) {
            let path = models_dir.join(mold_core::manifest::storage_path(manifest, file));
            std::fs::create_dir_all(path.parent().unwrap()).unwrap();
            std::fs::write(&path, b"test").unwrap();
            mold_core::download::write_sha256_marker(&path, "deadbeef").unwrap();
        }
        std::env::set_var("MOLD_MODELS_DIR", &models_dir);

        let app = app_empty();
        let resp = app
            .oneshot(
                Request::get("/api/models/sdxl-base%3Afp16/components")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();

        assert_eq!(resp.status(), StatusCode::OK);
        let json = json_body(resp).await;
        assert_eq!(json["model"], "sdxl-base:fp16");
        let components = json["components"].as_array().unwrap();
        assert!(components.iter().any(|c| c["present"] == true));
        assert!(components.iter().any(|c| c["present"] == false));
        assert!(components
            .iter()
            .all(|c| c["repair_model"] == "sdxl-base:fp16"));
        assert!(components.iter().all(|c| c["options"].is_array()));
        assert!(components.iter().any(|c| c["options"]
            .as_array()
            .unwrap()
            .iter()
            .any(|opt| opt["present"] == true)));

        std::env::remove_var("MOLD_MODELS_DIR");
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn list_models_does_not_block_during_generation() {
        let blocker = Arc::new(GenerateBlocker::default());
        let app = app_with(MockEngine::blocking_generate(blocker.clone()));

        let generate_task = tokio::spawn({
            let app = app.clone();
            async move {
                app.oneshot(
                    Request::post("/api/generate")
                        .header("content-type", "application/json")
                        .body(Body::from(generate_body("a cat", 768, 768)))
                        .unwrap(),
                )
                .await
                .unwrap()
            }
        });

        tokio::time::timeout(Duration::from_secs(1), async {
            while !blocker.entered.load(Ordering::SeqCst) {
                tokio::time::sleep(Duration::from_millis(10)).await;
            }
        })
        .await
        .expect("generate should enter the mock engine");

        let resp = tokio::time::timeout(
            Duration::from_millis(200),
            app.clone()
                .oneshot(Request::get("/api/models").body(Body::empty()).unwrap()),
        )
        .await
        .expect("/api/models should not block on active generation")
        .unwrap();
        assert_eq!(resp.status(), StatusCode::OK);

        blocker.release();
        let generate_resp = generate_task.await.unwrap();
        assert_eq!(generate_resp.status(), StatusCode::OK);
    }

    // ── /api/generate — validation ───────────────────────────────────────────

    #[tokio::test]
    async fn generate_empty_prompt_returns_422() {
        let app = app_with(MockEngine::ready());
        let body = generate_body("", 768, 768);
        let resp = app
            .oneshot(
                Request::post("/api/generate")
                    .header("content-type", "application/json")
                    .body(Body::from(body))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::UNPROCESSABLE_ENTITY);
        let body = json_body(resp).await;
        assert_eq!(body["code"], "VALIDATION_ERROR");
        assert!(body["error"].as_str().unwrap().contains("prompt"));
    }

    #[tokio::test]
    async fn generate_zero_width_returns_422() {
        let app = app_with(MockEngine::ready());
        let body = generate_body("a cat", 0, 768);
        let resp = app
            .oneshot(
                Request::post("/api/generate")
                    .header("content-type", "application/json")
                    .body(Body::from(body))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::UNPROCESSABLE_ENTITY);
        let body = json_body(resp).await;
        assert_eq!(body["code"], "VALIDATION_ERROR");
    }

    #[tokio::test]
    async fn generate_non_multiple_of_16_returns_422() {
        let app = app_with(MockEngine::ready());
        let body = generate_body("a cat", 769, 768);
        let resp = app
            .oneshot(
                Request::post("/api/generate")
                    .header("content-type", "application/json")
                    .body(Body::from(body))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::UNPROCESSABLE_ENTITY);
        let body = json_body(resp).await;
        assert_eq!(body["code"], "VALIDATION_ERROR");
    }

    #[tokio::test]
    async fn generate_oversized_returns_422() {
        let app = app_with(MockEngine::ready());
        // 1408x1408 = ~1.98MP > 1.8MP limit
        let body = generate_body("a cat", 1408, 1408);
        let resp = app
            .oneshot(
                Request::post("/api/generate")
                    .header("content-type", "application/json")
                    .body(Body::from(body))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::UNPROCESSABLE_ENTITY);
        let body = json_body(resp).await;
        assert_eq!(body["code"], "VALIDATION_ERROR");
    }

    #[tokio::test]
    async fn generate_zero_steps_returns_422() {
        let app = app_with(MockEngine::ready());
        let body = r#"{"prompt":"a cat","model":"mock-model","width":768,"height":768,"steps":0,"batch_size":1,"output_format":"png"}"#;
        let resp = app
            .oneshot(
                Request::post("/api/generate")
                    .header("content-type", "application/json")
                    .body(Body::from(body))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::UNPROCESSABLE_ENTITY);
        let body = json_body(resp).await;
        assert_eq!(body["code"], "VALIDATION_ERROR");
    }

    #[tokio::test]
    async fn server_local_media_paths_require_configured_roots() {
        let state = AppState::with_engine(MockEngine::ready());
        let mut req: GenerateRequest = serde_json::from_value(serde_json::json!({
            "prompt": "a cat",
            "model": "mock-model",
            "width": 768,
            "height": 768,
            "steps": 4,
            "batch_size": 1,
            "output_format": "mp4",
            "source_video_path": "/tmp/clip.mp4"
        }))
        .unwrap();
        {
            let mut config = state.config.write().await;
            config.models.insert(
                "mock-model".to_string(),
                mold_core::ModelConfig {
                    family: Some("ltx2".to_string()),
                    ..Default::default()
                },
            );
        }

        let err = crate::routes::resolve_server_local_media_paths(&state, &mut req)
            .await
            .unwrap_err();

        assert_eq!(err.code, "VALIDATION_ERROR");
        assert!(err.error.contains("media_roots"), "got: {}", err.error);
    }

    #[tokio::test]
    async fn server_local_media_paths_are_canonicalized_before_queueing() {
        let root = tempfile::tempdir().unwrap();
        let clip = root.path().join("clip.mp4");
        let audio = root.path().join("voice.wav");
        std::fs::write(&clip, b"mp4").unwrap();
        std::fs::write(&audio, b"wav").unwrap();
        let nested = root.path().join("nested");
        std::fs::create_dir(&nested).unwrap();
        let clip_with_parent = nested.join("..").join("clip.mp4");

        let state = AppState::with_engine(MockEngine::ready());
        {
            let mut config = state.config.write().await;
            config.media_roots = Some(vec![root.path().to_string_lossy().to_string()]);
            config.models.insert(
                "mock-model".to_string(),
                mold_core::ModelConfig {
                    family: Some("ltx2".to_string()),
                    ..Default::default()
                },
            );
        }
        let mut req: GenerateRequest = serde_json::from_value(serde_json::json!({
            "prompt": "a cat",
            "model": "mock-model",
            "width": 768,
            "height": 768,
            "steps": 4,
            "batch_size": 1,
            "output_format": "mp4",
            "audio_file_path": audio,
            "source_video_path": clip_with_parent
        }))
        .unwrap();

        crate::routes::resolve_server_local_media_paths(&state, &mut req)
            .await
            .unwrap();

        assert_eq!(
            req.audio_file_path.as_deref(),
            Some(audio.canonicalize().unwrap().to_str().unwrap())
        );
        assert_eq!(
            req.source_video_path.as_deref(),
            Some(clip.canonicalize().unwrap().to_str().unwrap())
        );
    }

    // ── /api/generate — success path ─────────────────────────────────────────

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn generate_valid_request_returns_image_bytes() {
        let app = app_with(MockEngine::ready());
        let body = generate_body("a glowing robot", 768, 768);
        let resp = app
            .oneshot(
                Request::post("/api/generate")
                    .header("content-type", "application/json")
                    .body(Body::from(body))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
        let ct = resp
            .headers()
            .get("content-type")
            .unwrap()
            .to_str()
            .unwrap();
        assert_eq!(ct, "image/png");
        assert!(
            resp.headers().contains_key("x-mold-seed-used"),
            "response should include x-mold-seed-used header"
        );
    }

    // ── /api/generate — engine error ─────────────────────────────────────────

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn generate_engine_error_returns_500() {
        let app = app_with(MockEngine::failing());
        let body = generate_body("a cat", 768, 768);
        let resp = app
            .oneshot(
                Request::post("/api/generate")
                    .header("content-type", "application/json")
                    .body(Body::from(body))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::INTERNAL_SERVER_ERROR);
        let body = json_body(resp).await;
        assert_eq!(body["code"], "INFERENCE_ERROR");
        assert!(body["error"]
            .as_str()
            .unwrap()
            .contains("mock engine error"));
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn generate_empty_images_returns_500() {
        let app = app_with(MockEngine::empty_images());
        let body = generate_body("a cat", 768, 768);
        let resp = app
            .oneshot(
                Request::post("/api/generate")
                    .header("content-type", "application/json")
                    .body(Body::from(body))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::INTERNAL_SERVER_ERROR);
        let body = json_body(resp).await;
        assert_eq!(body["code"], "INFERENCE_ERROR");
        assert!(body["error"]
            .as_str()
            .unwrap()
            .contains("returned no images"));
    }

    // ── /api/generate — unknown model ────────────────────────────────────────

    #[tokio::test]
    async fn generate_unknown_model_returns_400() {
        let app = app_empty();
        let body = r#"{"prompt":"a cat","model":"nonexistent-model-xyz","width":768,"height":768,"steps":4,"batch_size":1,"output_format":"png"}"#;
        let resp = app
            .oneshot(
                Request::post("/api/generate")
                    .header("content-type", "application/json")
                    .body(Body::from(body))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
        let body = json_body(resp).await;
        assert_eq!(body["code"], "UNKNOWN_MODEL");
    }

    // ── /api/generate — known but not downloaded model returns 404 ───────────

    #[tokio::test]
    #[allow(clippy::await_holding_lock)]
    async fn generate_known_model_not_downloaded_returns_404() {
        let _lock = env_lock().lock().unwrap_or_else(|e| e.into_inner());
        let models_dir = test_models_dir("generate-not-downloaded");
        std::fs::create_dir_all(&models_dir).unwrap();
        std::env::set_var("MOLD_MODELS_DIR", &models_dir);

        let app = app_empty();
        // flux-schnell:q8 is a known manifest model but not configured/downloaded
        let body = r#"{"prompt":"a cat","model":"flux-schnell:q8","width":768,"height":768,"steps":4,"batch_size":1,"output_format":"png"}"#;
        let resp = app
            .oneshot(
                Request::post("/api/generate")
                    .header("content-type", "application/json")
                    .body(Body::from(body))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::NOT_FOUND);
        let body = json_body(resp).await;
        assert_eq!(body["code"], "MODEL_NOT_FOUND");

        std::env::remove_var("MOLD_MODELS_DIR");
        let _ = std::fs::remove_dir_all(models_dir);
    }

    // ── /api/openapi.json ────────────────────────────────────────────────────

    #[tokio::test]
    async fn openapi_json_returns_valid_spec() {
        let app = app_with(MockEngine::ready());
        let resp = app
            .oneshot(
                Request::get("/api/openapi.json")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::OK);

        let body = axum::body::to_bytes(resp.into_body(), 1024 * 1024)
            .await
            .unwrap();
        let spec: serde_json::Value = serde_json::from_slice(&body).unwrap();
        // Must have openapi version field
        assert!(
            spec["openapi"].is_string(),
            "spec should have openapi version"
        );
        // Must have paths
        assert!(spec["paths"].is_object(), "spec should have paths");
        // Must have our generate endpoint
        assert!(
            spec["paths"]["/api/generate"].is_object(),
            "spec should have /api/generate path"
        );
    }

    // ── /api/docs ────────────────────────────────────────────────────────────

    #[tokio::test]
    async fn docs_returns_html() {
        let app = app_with(MockEngine::ready());
        let resp = app
            .oneshot(Request::get("/api/docs").body(Body::empty()).unwrap())
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
        let ct = resp
            .headers()
            .get("content-type")
            .unwrap()
            .to_str()
            .unwrap();
        assert!(
            ct.contains("text/html"),
            "docs should return HTML, got: {ct}"
        );
    }

    // ── /api/generate/stream — SSE streaming ────────────────────────────────

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn stream_valid_request_returns_sse() {
        let app = app_with(MockEngine::ready());
        let body = generate_body("a robot", 768, 768);
        let resp = app
            .oneshot(
                Request::post("/api/generate/stream")
                    .header("content-type", "application/json")
                    .body(Body::from(body))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
        let ct = resp
            .headers()
            .get("content-type")
            .unwrap()
            .to_str()
            .unwrap();
        assert!(
            ct.contains("text/event-stream"),
            "stream should return text/event-stream, got: {ct}"
        );

        // Collect body and verify it contains a complete event with base64 image
        let body = axum::body::to_bytes(resp.into_body(), 1024 * 1024)
            .await
            .unwrap();
        let text = String::from_utf8_lossy(&body);
        assert!(
            text.contains("event: complete"),
            "stream should contain a complete event"
        );
        assert!(
            text.contains("\"image\""),
            "complete event should contain base64 image"
        );
    }

    #[tokio::test]
    async fn stream_empty_prompt_returns_422() {
        let app = app_with(MockEngine::ready());
        let body = generate_body("", 768, 768);
        let resp = app
            .oneshot(
                Request::post("/api/generate/stream")
                    .header("content-type", "application/json")
                    .body(Body::from(body))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::UNPROCESSABLE_ENTITY);
        let body = json_body(resp).await;
        assert_eq!(body["code"], "VALIDATION_ERROR");
    }

    #[tokio::test]
    async fn stream_unknown_model_returns_400() {
        let app = app_empty();
        let body = r#"{"prompt":"a cat","model":"nonexistent-model-xyz","width":768,"height":768,"steps":4,"batch_size":1,"output_format":"png"}"#;
        let resp = app
            .oneshot(
                Request::post("/api/generate/stream")
                    .header("content-type", "application/json")
                    .body(Body::from(body))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
        let body = json_body(resp).await;
        assert_eq!(body["code"], "UNKNOWN_MODEL");
    }

    #[tokio::test]
    #[allow(clippy::await_holding_lock)]
    async fn stream_known_model_not_downloaded_returns_404() {
        let _lock = env_lock().lock().unwrap_or_else(|e| e.into_inner());
        let models_dir = test_models_dir("stream-not-downloaded");
        std::fs::create_dir_all(&models_dir).unwrap();
        std::env::set_var("MOLD_MODELS_DIR", &models_dir);

        let app = app_empty();
        let body = r#"{"prompt":"a cat","model":"flux-schnell:q8","width":768,"height":768,"steps":4,"batch_size":1,"output_format":"png"}"#;
        let resp = app
            .oneshot(
                Request::post("/api/generate/stream")
                    .header("content-type", "application/json")
                    .body(Body::from(body))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::NOT_FOUND);
        let body = json_body(resp).await;
        assert_eq!(body["code"], "MODEL_NOT_FOUND");

        std::env::remove_var("MOLD_MODELS_DIR");
        let _ = std::fs::remove_dir_all(models_dir);
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn stream_engine_error_returns_sse_error() {
        let app = app_with(MockEngine::failing());
        let body = generate_body("a cat", 768, 768);
        let resp = app
            .oneshot(
                Request::post("/api/generate/stream")
                    .header("content-type", "application/json")
                    .body(Body::from(body))
                    .unwrap(),
            )
            .await
            .unwrap();
        // SSE stream starts with 200 — error is in the event stream
        assert_eq!(resp.status(), StatusCode::OK);

        let body = axum::body::to_bytes(resp.into_body(), 1024 * 1024)
            .await
            .unwrap();
        let text = String::from_utf8_lossy(&body);
        assert!(
            text.contains("event: error"),
            "stream should contain an error event"
        );
        assert!(
            text.contains("mock engine error"),
            "error event should contain the engine error message"
        );
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn stream_empty_images_returns_sse_error() {
        let app = app_with(MockEngine::empty_images());
        let body = generate_body("a cat", 768, 768);
        let resp = app
            .oneshot(
                Request::post("/api/generate/stream")
                    .header("content-type", "application/json")
                    .body(Body::from(body))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::OK);

        let body = axum::body::to_bytes(resp.into_body(), 1024 * 1024)
            .await
            .unwrap();
        let text = String::from_utf8_lossy(&body);
        assert!(text.contains("event: error"));
        assert!(text.contains("returned no images"));
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn reused_engine_clears_progress_callbacks_between_stream_and_generate() {
        let progress_set_count = Arc::new(AtomicUsize::new(0));
        let progress_clear_count = Arc::new(AtomicUsize::new(0));
        let app = app_with(MockEngine::tracked_progress(
            progress_set_count.clone(),
            progress_clear_count.clone(),
        ));

        let stream_resp = app
            .clone()
            .oneshot(
                Request::post("/api/generate/stream")
                    .header("content-type", "application/json")
                    .body(Body::from(generate_body("a robot", 768, 768)))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(stream_resp.status(), StatusCode::OK);
        let _ = axum::body::to_bytes(stream_resp.into_body(), 1024 * 1024)
            .await
            .unwrap();

        assert_eq!(progress_set_count.load(Ordering::SeqCst), 2);
        assert_eq!(progress_clear_count.load(Ordering::SeqCst), 1);

        let generate_resp = app
            .oneshot(
                Request::post("/api/generate")
                    .header("content-type", "application/json")
                    .body(Body::from(generate_body("a robot", 768, 768)))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(generate_resp.status(), StatusCode::OK);

        assert_eq!(progress_set_count.load(Ordering::SeqCst), 2);
        assert_eq!(progress_clear_count.load(Ordering::SeqCst), 3);
    }

    #[tokio::test]
    async fn unload_loaded_model_returns_200() {
        let app = app_with(MockEngine::ready());
        let resp = app
            .oneshot(
                Request::delete("/api/models/unload")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
        let body = axum::body::to_bytes(resp.into_body(), 1024 * 1024)
            .await
            .unwrap();
        assert!(String::from_utf8_lossy(&body).contains("unloaded mock-model"));
    }

    #[tokio::test]
    async fn unload_drops_engine_entirely() {
        let state = AppState::with_engine(MockEngine::ready());
        let app = app_with_state(state.clone());
        let resp = app
            .oneshot(
                Request::delete("/api/models/unload")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::OK);

        // Engine must be unloaded — in cache but not on GPU
        let cache = state.model_cache.lock().await;
        assert!(
            cache.active_model().is_none(),
            "no model should be active after unload"
        );
    }

    #[tokio::test]
    async fn unload_clears_snapshot_model_name() {
        let state = AppState::with_engine(MockEngine::ready());
        let app = app_with_state(state.clone());

        // Verify cache has an active model before unload
        {
            let snapshot = state.model_cache.lock().await.snapshot();
            assert!(snapshot.model_name.is_some());
            assert!(snapshot.is_loaded);
        }

        let resp = app
            .oneshot(
                Request::delete("/api/models/unload")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::OK);

        // Cache snapshot must reflect no active model after unload
        let snapshot = state.model_cache.lock().await.snapshot();
        assert!(
            snapshot.model_name.is_none(),
            "snapshot model_name should be None after unload"
        );
        assert!(!snapshot.is_loaded);
    }

    #[tokio::test]
    async fn unload_no_model_returns_200_with_message() {
        let (tx, _rx) = tokio::sync::mpsc::channel(16);
        let queue = crate::state::QueueHandle::new(tx);
        let gpu_pool = std::sync::Arc::new(crate::gpu_pool::GpuPool {
            workers: Vec::new(),
        });
        let app = app_with_state(AppState::empty(
            mold_core::Config::default(),
            queue,
            gpu_pool,
            200,
        ));
        let resp = app
            .oneshot(
                Request::delete("/api/models/unload")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
        let body = axum::body::to_bytes(resp.into_body(), 1024 * 1024)
            .await
            .unwrap();
        assert!(String::from_utf8_lossy(&body).contains("no model loaded"));
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn concurrent_requests_only_load_existing_engine_once() {
        let load_count = Arc::new(AtomicUsize::new(0));
        let (tx, rx) = tokio::sync::mpsc::channel(16);
        let queue = crate::state::QueueHandle::new(tx);
        let engine = MockEngine::unloaded(load_count.clone(), Duration::from_millis(50));
        let mut cache = crate::model_cache::ModelCache::new(3);
        cache.insert(Box::new(engine), 0);
        let state = AppState {
            gpu_pool: std::sync::Arc::new(crate::gpu_pool::GpuPool {
                workers: Vec::new(),
            }),
            queue_capacity: 200,
            model_cache: Arc::new(tokio::sync::Mutex::new(cache)),
            active_generation: Arc::new(std::sync::RwLock::new(None)),
            config: Arc::new(tokio::sync::RwLock::new(mold_core::Config::default())),
            start_time: std::time::Instant::now(),
            model_load_lock: Arc::new(tokio::sync::Mutex::new(())),
            pull_lock: Arc::new(tokio::sync::Mutex::new(())),
            chain_lock: Arc::new(tokio::sync::Mutex::new(())),
            queue,
            job_registry: crate::job_registry::JobRegistry::new(),
            shared_pool: Arc::new(std::sync::Mutex::new(
                mold_inference::shared_pool::SharedPool::new(),
            )),
            shutdown_tx: Arc::new(tokio::sync::Mutex::new(None)),
            upscaler_cache: Arc::new(std::sync::Mutex::new(None)),
            metadata_db: Arc::new(None),
            chain_jobs: None,
            downloads: crate::downloads::DownloadQueue::new(),
            resources: crate::resources::ResourceBroadcaster::new(),
            catalog_live_cache: mold_catalog::live::LiveCache::new(
                std::time::Duration::from_secs(300),
                64,
            ),
            catalog_live_civitai_base: std::sync::Arc::new(
                crate::state::CATALOG_LIVE_CIVITAI_BASE.to_string(),
            ),
            catalog_intents: std::sync::Arc::new(tokio::sync::RwLock::new(
                std::collections::HashMap::new(),
            )),
        };
        let worker_state = state.clone();
        tokio::spawn(crate::queue::run_queue_worker(rx, worker_state));
        let app = app_with_state(state);
        let req1 = Request::post("/api/generate")
            .header("content-type", "application/json")
            .body(Body::from(generate_body("a cat", 768, 768)))
            .unwrap();
        let req2 = Request::post("/api/generate")
            .header("content-type", "application/json")
            .body(Body::from(generate_body("a cat", 768, 768)))
            .unwrap();

        let (resp1, resp2) = tokio::join!(app.clone().oneshot(req1), app.oneshot(req2));
        assert_eq!(resp1.unwrap().status(), StatusCode::OK);
        assert_eq!(resp2.unwrap().status(), StatusCode::OK);
        assert_eq!(load_count.load(Ordering::SeqCst), 1);
    }

    /// Verify that progress events emitted during model loading (e.g. FP8→Q8
    /// conversion) are delivered through the SSE stream to the client.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn stream_delivers_load_progress_events() {
        let (tx, rx) = tokio::sync::mpsc::channel(16);
        let queue = crate::state::QueueHandle::new(tx);
        let engine = MockEngine::unloaded_with_progress();
        let mut cache = crate::model_cache::ModelCache::new(3);
        cache.insert(Box::new(engine), 0);
        let state = AppState {
            gpu_pool: std::sync::Arc::new(crate::gpu_pool::GpuPool {
                workers: Vec::new(),
            }),
            queue_capacity: 200,
            model_cache: Arc::new(tokio::sync::Mutex::new(cache)),
            active_generation: Arc::new(std::sync::RwLock::new(None)),
            config: Arc::new(tokio::sync::RwLock::new(mold_core::Config::default())),
            start_time: std::time::Instant::now(),
            model_load_lock: Arc::new(tokio::sync::Mutex::new(())),
            pull_lock: Arc::new(tokio::sync::Mutex::new(())),
            chain_lock: Arc::new(tokio::sync::Mutex::new(())),
            queue,
            job_registry: crate::job_registry::JobRegistry::new(),
            shared_pool: Arc::new(std::sync::Mutex::new(
                mold_inference::shared_pool::SharedPool::new(),
            )),
            shutdown_tx: Arc::new(tokio::sync::Mutex::new(None)),
            upscaler_cache: Arc::new(std::sync::Mutex::new(None)),
            metadata_db: Arc::new(None),
            chain_jobs: None,
            downloads: crate::downloads::DownloadQueue::new(),
            resources: crate::resources::ResourceBroadcaster::new(),
            catalog_live_cache: mold_catalog::live::LiveCache::new(
                std::time::Duration::from_secs(300),
                64,
            ),
            catalog_live_civitai_base: std::sync::Arc::new(
                crate::state::CATALOG_LIVE_CIVITAI_BASE.to_string(),
            ),
            catalog_intents: std::sync::Arc::new(tokio::sync::RwLock::new(
                std::collections::HashMap::new(),
            )),
        };
        let worker_state = state.clone();
        tokio::spawn(crate::queue::run_queue_worker(rx, worker_state));
        let app = app_with_state(state);
        let resp = app
            .oneshot(
                Request::post("/api/generate/stream")
                    .header("content-type", "application/json")
                    .body(Body::from(generate_body("a cat", 768, 768)))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::OK);

        let body = axum::body::to_bytes(resp.into_body(), 1024 * 1024)
            .await
            .unwrap();
        let text = String::from_utf8_lossy(&body);

        // The SSE stream must contain the FP8 conversion info event
        assert!(
            text.contains("Converting FP8 checkpoint"),
            "SSE stream should contain FP8 conversion progress info event, got: {text}"
        );
        // And the stage start event from model loading
        assert!(
            text.contains("Loading transformer"),
            "SSE stream should contain model loading stage event, got: {text}"
        );
        // Final complete event should also be present
        assert!(
            text.contains("event: complete"),
            "SSE stream should contain complete event, got: {text}"
        );
    }

    // ── Queue-specific tests ─────────────────────────────────────────────────

    /// Verify that two concurrent streaming requests both complete successfully
    /// when submitted to the generation queue. The first request blocks on
    /// generate, the second should queue behind it and complete after.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn concurrent_stream_requests_both_complete() {
        let blocker = Arc::new(GenerateBlocker::default());
        let (state, rx) =
            AppState::with_engine_and_queue(MockEngine::blocking_generate(blocker.clone()));
        let worker_state = state.clone();
        tokio::spawn(crate::queue::run_queue_worker(rx, worker_state));
        let app = app_with_state(state);

        let resp1_future = {
            let app = app.clone();
            tokio::spawn(async move {
                let resp = app
                    .oneshot(
                        Request::post("/api/generate/stream")
                            .header("content-type", "application/json")
                            .body(Body::from(generate_body("request one", 768, 768)))
                            .unwrap(),
                    )
                    .await
                    .unwrap();
                let body = axum::body::to_bytes(resp.into_body(), 1024 * 1024)
                    .await
                    .unwrap();
                String::from_utf8_lossy(&body).to_string()
            })
        };

        // Wait for the first request to enter generate (blocked)
        tokio::time::sleep(Duration::from_millis(50)).await;

        // Submit second request
        let resp2_future = {
            let app = app.clone();
            tokio::spawn(async move {
                let resp = app
                    .oneshot(
                        Request::post("/api/generate/stream")
                            .header("content-type", "application/json")
                            .body(Body::from(generate_body("request two", 768, 768)))
                            .unwrap(),
                    )
                    .await
                    .unwrap();
                let body = axum::body::to_bytes(resp.into_body(), 1024 * 1024)
                    .await
                    .unwrap();
                String::from_utf8_lossy(&body).to_string()
            })
        };

        // Release the blocker after a short delay
        tokio::time::sleep(Duration::from_millis(50)).await;
        blocker.release();

        let text1 = resp1_future.await.unwrap();
        let text2 = resp2_future.await.unwrap();

        assert!(
            text1.contains("event: complete"),
            "first request should complete, got: {text1}"
        );
        assert!(
            text2.contains("event: complete"),
            "second request should complete, got: {text2}"
        );
    }

    /// Verify that a queued streaming request receives a position event.
    ///
    /// Strategy: submit both requests BEFORE starting the queue worker.
    /// Without a worker, no job holds model_cache long-term, so both HTTP
    /// handlers complete submit() immediately with sequential positions
    /// (0 then 1). Starting the worker afterward lets both jobs process
    /// and close their SSE streams.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn queued_stream_receives_position_event() {
        let (state, rx) = AppState::with_engine_and_queue(MockEngine::ready());
        let queue = state.queue.clone();
        let worker_state = state.clone();
        let app = app_with_state(state);

        // Submit first request (worker not started — handler completes fast)
        let _resp1 = {
            let app = app.clone();
            tokio::spawn(async move {
                app.oneshot(
                    Request::post("/api/generate/stream")
                        .header("content-type", "application/json")
                        .body(Body::from(generate_body("first", 768, 768)))
                        .unwrap(),
                )
                .await
            })
        };

        // Wait for the first request to be queued before submitting the second,
        // guaranteeing the second request sees pending_count == 1 (position 1).
        while queue.pending() < 1 {
            tokio::time::sleep(Duration::from_millis(1)).await;
        }

        // Submit second request — should be queued at position 1
        let resp2 = {
            let app = app.clone();
            tokio::spawn(async move {
                let resp = app
                    .oneshot(
                        Request::post("/api/generate/stream")
                            .header("content-type", "application/json")
                            .body(Body::from(generate_body("second", 768, 768)))
                            .unwrap(),
                    )
                    .await
                    .unwrap();
                let body = axum::body::to_bytes(resp.into_body(), 1024 * 1024)
                    .await
                    .unwrap();
                String::from_utf8_lossy(&body).to_string()
            })
        };

        // Wait for both requests to be queued, then start the worker so
        // both jobs are processed and their SSE streams close.
        while queue.pending() < 2 {
            tokio::time::sleep(Duration::from_millis(1)).await;
        }
        tokio::spawn(crate::queue::run_queue_worker(rx, worker_state));

        let text2 = resp2.await.unwrap();
        assert!(
            text2.contains(r#""type":"queued""#),
            "second request should receive a queued event, got: {text2}"
        );
        // The second request should report position > 0 (queued behind the first)
        assert!(
            text2.contains(r#""position":1"#),
            "second request should be at position 1, got: {text2}"
        );
    }

    /// Verify that both streaming and non-streaming requests are properly
    /// serialized through the queue.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn non_streaming_generate_queues_correctly() {
        let app = app_with(MockEngine::ready());

        // Submit two non-streaming requests concurrently
        let resp1 = {
            let app = app.clone();
            tokio::spawn(async move {
                app.oneshot(
                    Request::post("/api/generate")
                        .header("content-type", "application/json")
                        .body(Body::from(generate_body("request one", 768, 768)))
                        .unwrap(),
                )
                .await
                .unwrap()
            })
        };
        let resp2 = {
            let app = app.clone();
            tokio::spawn(async move {
                app.oneshot(
                    Request::post("/api/generate")
                        .header("content-type", "application/json")
                        .body(Body::from(generate_body("request two", 768, 768)))
                        .unwrap(),
                )
                .await
                .unwrap()
            })
        };

        let (r1, r2) = tokio::join!(resp1, resp2);
        assert_eq!(r1.unwrap().status(), StatusCode::OK);
        assert_eq!(r2.unwrap().status(), StatusCode::OK);
    }

    /// Verify snapshot is consistent after model load through the queue.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn snapshot_consistent_after_queue_load() {
        let load_count = Arc::new(AtomicUsize::new(0));
        let (tx, rx) = tokio::sync::mpsc::channel(16);
        let queue = crate::state::QueueHandle::new(tx);
        let engine = MockEngine::unloaded(load_count, Duration::from_millis(10));
        let mut cache = crate::model_cache::ModelCache::new(3);
        cache.insert(Box::new(engine), 0);
        let state = AppState {
            gpu_pool: std::sync::Arc::new(crate::gpu_pool::GpuPool {
                workers: Vec::new(),
            }),
            queue_capacity: 200,
            model_cache: Arc::new(tokio::sync::Mutex::new(cache)),
            active_generation: Arc::new(std::sync::RwLock::new(None)),
            config: Arc::new(tokio::sync::RwLock::new(mold_core::Config::default())),
            start_time: std::time::Instant::now(),
            model_load_lock: Arc::new(tokio::sync::Mutex::new(())),
            pull_lock: Arc::new(tokio::sync::Mutex::new(())),
            chain_lock: Arc::new(tokio::sync::Mutex::new(())),
            queue,
            job_registry: crate::job_registry::JobRegistry::new(),
            shared_pool: Arc::new(std::sync::Mutex::new(
                mold_inference::shared_pool::SharedPool::new(),
            )),
            shutdown_tx: Arc::new(tokio::sync::Mutex::new(None)),
            upscaler_cache: Arc::new(std::sync::Mutex::new(None)),
            metadata_db: Arc::new(None),
            chain_jobs: None,
            downloads: crate::downloads::DownloadQueue::new(),
            resources: crate::resources::ResourceBroadcaster::new(),
            catalog_live_cache: mold_catalog::live::LiveCache::new(
                std::time::Duration::from_secs(300),
                64,
            ),
            catalog_live_civitai_base: std::sync::Arc::new(
                crate::state::CATALOG_LIVE_CIVITAI_BASE.to_string(),
            ),
            catalog_intents: std::sync::Arc::new(tokio::sync::RwLock::new(
                std::collections::HashMap::new(),
            )),
        };
        let worker_state = state.clone();
        tokio::spawn(crate::queue::run_queue_worker(rx, worker_state));
        let app = app_with_state(state.clone());

        let resp = app
            .oneshot(
                Request::post("/api/generate")
                    .header("content-type", "application/json")
                    .body(Body::from(generate_body("a cat", 768, 768)))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::OK);

        // After generation, snapshot should reflect the loaded model
        let snapshot = state.model_cache.lock().await.snapshot();
        assert_eq!(
            snapshot.model_name.as_deref(),
            Some("mock-model"),
            "snapshot should reflect the loaded model"
        );
        assert!(snapshot.is_loaded, "snapshot should show model as loaded");
    }

    // ── Durable chain-job route integration tests ──────────────────────────

    #[tokio::test]
    async fn chain_jobs_route_returns_503_when_runner_unavailable() {
        let app = app_empty();
        let resp = app
            .oneshot(
                Request::builder()
                    .uri("/api/chain-jobs")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::SERVICE_UNAVAILABLE);
        let body = json_body(resp).await;
        assert_eq!(body["code"], "CHAIN_JOBS_UNAVAILABLE");
    }

    #[tokio::test]
    async fn chain_job_detail_returns_404_for_unknown_job() {
        let home = tempfile::tempdir().unwrap();
        let _home = MoldHomeGuard::set(home.path());
        let app = app_with_chain_db(mold_db::MetadataDb::open_in_memory().unwrap());

        let resp = app
            .oneshot(
                Request::builder()
                    .uri("/api/chain-jobs/missing")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::NOT_FOUND);
        let body = json_body(resp).await;
        assert_eq!(body["code"], "CHAIN_JOB_NOT_FOUND");
    }

    #[tokio::test]
    async fn retake_splice_before_smooth_boundary_returns_409_code() {
        let home = tempfile::tempdir().unwrap();
        let _home = MoldHomeGuard::set(home.path());
        let db = mold_db::MetadataDb::open_in_memory().unwrap();
        let mut req = route_chain_request();
        req.stages
            .push(route_chain_stage("second shot", TransitionMode::Smooth));
        seed_chain_job_with_request(
            &db,
            home.path(),
            "smooth-retake",
            ChainJobState::Completed,
            &req,
        );
        let app = app_with_chain_db(db);
        let body = serde_json::to_string(&mold_core::chain_job::RetakeRequest {
            stage_idx: 0,
            mode: RetakeMode::Splice,
            seed_offset: Some(9),
            prompt: None,
        })
        .unwrap();

        let resp = app
            .oneshot(
                Request::post("/api/chain-jobs/smooth-retake/retake")
                    .header("content-type", "application/json")
                    .body(Body::from(body))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::CONFLICT);
        let body = json_body(resp).await;
        assert_eq!(body["code"], "RETAKE_SPLICE_REQUIRES_CUT_OR_FADE");
    }

    #[tokio::test]
    async fn chain_job_events_first_event_is_snapshot() {
        use futures::StreamExt as _;

        let home = tempfile::tempdir().unwrap();
        let _home = MoldHomeGuard::set(home.path());
        let db = mold_db::MetadataDb::open_in_memory().unwrap();
        seed_chain_job(&db, home.path(), "events", ChainJobState::Queued);
        let app = app_with_chain_db(db);

        let resp = app
            .oneshot(
                Request::get("/api/chain-jobs/events/events")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
        let mut body = resp.into_body().into_data_stream();
        let chunk = tokio::time::timeout(Duration::from_secs(1), body.next())
            .await
            .expect("first SSE event should arrive")
            .expect("SSE body should produce a chunk")
            .expect("SSE chunk should be readable");
        let text = String::from_utf8_lossy(&chunk);
        assert!(
            text.contains("event: chain_job") && text.contains(r#""type":"snapshot""#),
            "first SSE chunk must be Snapshot, got: {text}"
        );
    }

    #[cfg(not(feature = "mp4"))]
    #[tokio::test]
    async fn create_chain_job_rejects_audio_when_mp4_feature_is_disabled() {
        let home = tempfile::tempdir().unwrap();
        let _home = MoldHomeGuard::set(home.path());
        let app = app_with_chain_db(mold_db::MetadataDb::open_in_memory().unwrap());
        let mut req = route_chain_request();
        req.enable_audio = Some(true);

        let resp = app
            .oneshot(
                Request::post("/api/chain-jobs")
                    .header("content-type", "application/json")
                    .body(Body::from(serde_json::to_string(&req).unwrap()))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::UNPROCESSABLE_ENTITY);
        let body = json_body(resp).await;
        assert!(
            body["error"]
                .as_str()
                .is_some_and(|error| error.contains("mp4 feature")),
            "expected clear mp4 feature error, got {body}"
        );
    }

    #[tokio::test]
    async fn cancel_queued_chain_job_returns_202_and_cancelled_summary() {
        let home = tempfile::tempdir().unwrap();
        let _home = MoldHomeGuard::set(home.path());
        let db = mold_db::MetadataDb::open_in_memory().unwrap();
        seed_chain_job(&db, home.path(), "queued", ChainJobState::Queued);
        let app = app_with_chain_db(db);

        let resp = app
            .oneshot(
                Request::post("/api/chain-jobs/queued/cancel")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::ACCEPTED);
        let body = json_body(resp).await;
        assert_eq!(body["state"], "cancelled");
    }

    #[tokio::test]
    async fn delete_running_chain_job_returns_409() {
        let home = tempfile::tempdir().unwrap();
        let _home = MoldHomeGuard::set(home.path());
        let db = mold_db::MetadataDb::open_in_memory().unwrap();
        seed_chain_job(&db, home.path(), "running", ChainJobState::Running);
        let app = app_with_chain_db(db);

        let resp = app
            .oneshot(
                Request::delete("/api/chain-jobs/running")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::CONFLICT);
        let body = json_body(resp).await;
        assert_eq!(body["code"], "CHAIN_JOB_RUNNING");
    }

    #[tokio::test]
    async fn delete_non_running_chain_job_returns_204_and_removes_dir() {
        let home = tempfile::tempdir().unwrap();
        let _home = MoldHomeGuard::set(home.path());
        let db = mold_db::MetadataDb::open_in_memory().unwrap();
        let job_dir = seed_chain_job(&db, home.path(), "failed", ChainJobState::Failed);
        let app = app_with_chain_db(db);

        let resp = app
            .oneshot(
                Request::delete("/api/chain-jobs/failed")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::NO_CONTENT);
        assert!(!job_dir.exists(), "delete must remove the durable job dir");
    }

    #[tokio::test]
    async fn delete_subscribed_non_running_chain_job_removes_bus_entry() {
        let home = tempfile::tempdir().unwrap();
        let _home = MoldHomeGuard::set(home.path());
        let db = mold_db::MetadataDb::open_in_memory().unwrap();
        seed_chain_job(&db, home.path(), "failed-subscribed", ChainJobState::Failed);
        let handle = Arc::new(crate::chain_job_runner::ChainJobRunnerHandle::inert_for_tests());
        let _rx = handle
            .events_for_tests()
            .subscribe_persistent_for_tests("failed-subscribed");
        assert!(handle
            .events_for_tests()
            .contains_for_tests("failed-subscribed"));
        let app = app_with_chain_handle(db, handle.clone());

        let resp = app
            .oneshot(
                Request::delete("/api/chain-jobs/failed-subscribed")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();

        assert_eq!(resp.status(), StatusCode::NO_CONTENT);
        assert!(!handle
            .events_for_tests()
            .contains_for_tests("failed-subscribed"));
    }

    #[tokio::test]
    async fn auth_rejects_missing_api_key_on_chain_jobs_route() {
        let keys = std::collections::HashSet::from(["test-key".to_string()]);
        let auth = Some(std::sync::Arc::new(crate::auth::ApiKeySet::new(keys)));
        let app = app_with_auth(auth);

        let resp = app
            .oneshot(
                Request::builder()
                    .uri("/api/chain-jobs")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::UNAUTHORIZED);
    }

    // ── Auth & Rate Limiting integration tests ──────────────────────────────

    /// Build a router with auth middleware applied (mirrors lib.rs wiring).
    /// Uses .layer() (not .route_layer()) for inject so auth runs on ALL requests
    /// including unmatched 404 paths — preventing auth bypass.
    fn app_with_auth(auth_state: crate::auth::AuthState) -> axum::Router {
        let app = app_empty();
        app.layer(axum::middleware::from_fn(crate::auth::require_api_key))
            .layer(axum::middleware::from_fn_with_state(
                auth_state,
                crate::auth::inject_auth_state,
            ))
    }

    #[tokio::test]
    async fn auth_rejects_missing_api_key() {
        let keys = std::collections::HashSet::from(["test-key".to_string()]);
        let auth = Some(std::sync::Arc::new(crate::auth::ApiKeySet::new(keys)));
        let app = app_with_auth(auth);

        let resp = app
            .oneshot(
                Request::builder()
                    .uri("/api/status")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::UNAUTHORIZED);
        let body = json_body(resp).await;
        assert_eq!(body["code"], "UNAUTHORIZED");
    }

    #[tokio::test]
    async fn auth_rejects_invalid_api_key() {
        let keys = std::collections::HashSet::from(["test-key".to_string()]);
        let auth = Some(std::sync::Arc::new(crate::auth::ApiKeySet::new(keys)));
        let app = app_with_auth(auth);

        let resp = app
            .oneshot(
                Request::builder()
                    .uri("/api/status")
                    .header("x-api-key", "wrong-key")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::UNAUTHORIZED);
    }

    #[tokio::test]
    async fn auth_allows_valid_api_key() {
        let keys = std::collections::HashSet::from(["test-key".to_string()]);
        let auth = Some(std::sync::Arc::new(crate::auth::ApiKeySet::new(keys)));
        let app = app_with_auth(auth);

        let resp = app
            .oneshot(
                Request::builder()
                    .uri("/api/status")
                    .header("x-api-key", "test-key")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
    }

    #[tokio::test]
    async fn auth_health_exempt() {
        let keys = std::collections::HashSet::from(["test-key".to_string()]);
        let auth = Some(std::sync::Arc::new(crate::auth::ApiKeySet::new(keys)));
        let app = app_with_auth(auth);

        let resp = app
            .oneshot(
                Request::builder()
                    .uri("/health")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
    }

    #[tokio::test]
    async fn auth_docs_exempt() {
        let keys = std::collections::HashSet::from(["test-key".to_string()]);
        let auth = Some(std::sync::Arc::new(crate::auth::ApiKeySet::new(keys)));
        let app = app_with_auth(auth);

        let resp = app
            .oneshot(
                Request::builder()
                    .uri("/api/docs")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
    }

    #[tokio::test]
    async fn auth_openapi_exempt() {
        let keys = std::collections::HashSet::from(["test-key".to_string()]);
        let auth = Some(std::sync::Arc::new(crate::auth::ApiKeySet::new(keys)));
        let app = app_with_auth(auth);

        let resp = app
            .oneshot(
                Request::builder()
                    .uri("/api/openapi.json")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
    }

    #[tokio::test]
    async fn auth_disabled_when_none() {
        let app = app_with_auth(None);

        let resp = app
            .oneshot(
                Request::builder()
                    .uri("/api/status")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        // Should succeed without any API key
        assert_eq!(resp.status(), StatusCode::OK);
    }

    #[tokio::test]
    async fn auth_supports_multiple_keys() {
        let keys =
            std::collections::HashSet::from(["key-alpha".to_string(), "key-beta".to_string()]);
        let auth = Some(std::sync::Arc::new(crate::auth::ApiKeySet::new(keys)));
        let app = app_with_auth(auth);

        // First key works
        let resp = app
            .clone()
            .oneshot(
                Request::builder()
                    .uri("/api/status")
                    .header("x-api-key", "key-alpha")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::OK);

        // Second key works
        let resp = app
            .oneshot(
                Request::builder()
                    .uri("/api/status")
                    .header("x-api-key", "key-beta")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
    }

    #[tokio::test]
    async fn request_id_generated() {
        let app = app_empty().layer(axum::middleware::from_fn(
            crate::request_id::request_id_middleware,
        ));

        let resp = app
            .oneshot(
                Request::builder()
                    .uri("/health")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert!(resp.headers().contains_key("x-request-id"));
    }

    #[tokio::test]
    async fn request_id_preserved() {
        let app = app_empty().layer(axum::middleware::from_fn(
            crate::request_id::request_id_middleware,
        ));

        let resp = app
            .oneshot(
                Request::builder()
                    .uri("/health")
                    .header("x-request-id", "my-id-123")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(
            resp.headers()
                .get("x-request-id")
                .unwrap()
                .to_str()
                .unwrap(),
            "my-id-123"
        );
    }

    #[test]
    fn rate_limit_parse_specs() {
        use crate::rate_limit::RouteTier;
        use axum::http::Method;

        // Generation tier
        assert_eq!(
            crate::rate_limit::classify_route("/api/generate", &Method::POST),
            Some(RouteTier::Generation)
        );
        assert_eq!(
            crate::rate_limit::classify_route("/api/generate/stream", &Method::POST),
            Some(RouteTier::Generation)
        );

        // Read tier
        assert_eq!(
            crate::rate_limit::classify_route("/api/status", &Method::GET),
            Some(RouteTier::Read)
        );

        // Exempt
        assert_eq!(
            crate::rate_limit::classify_route("/health", &Method::GET),
            None
        );
    }

    #[tokio::test]
    async fn auth_enforced_on_unmatched_404_paths() {
        let keys = std::collections::HashSet::from(["test-key".to_string()]);
        let auth = Some(std::sync::Arc::new(crate::auth::ApiKeySet::new(keys)));
        let app = app_with_auth(auth);

        // Request to non-existent path without API key should get 401, not 404.
        let resp = app
            .oneshot(
                Request::builder()
                    .uri("/nonexistent")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(
            resp.status(),
            StatusCode::UNAUTHORIZED,
            "unmatched paths must still require auth"
        );
    }

    #[test]
    fn rate_limiter_map_bounded() {
        use crate::rate_limit::MAX_LIMITER_ENTRIES;

        let quota = governor::Quota::per_second(std::num::NonZeroU32::new(10).unwrap())
            .allow_burst(std::num::NonZeroU32::new(10).unwrap());
        let state = crate::rate_limit::RateLimitState::new(quota, quota);

        // Fill the map to the cap
        for i in 0..MAX_LIMITER_ENTRIES {
            let ip = IpAddr::V4(std::net::Ipv4Addr::from((i as u32).to_be_bytes()));
            state.get_generation_limiter(ip);
        }

        // Next insertion should trigger eviction (map cleared + new entry)
        let ip = IpAddr::V4(std::net::Ipv4Addr::new(255, 255, 255, 255));
        state.get_generation_limiter(ip);

        // Map should be small again (just the one new entry)
        let map = state.generation_limiters.lock().unwrap();
        assert!(map.len() <= 1, "map should be evicted, got {}", map.len());
    }

    /// `/api/gallery` should serve from the SQLite metadata DB when one is
    /// attached to AppState — bypassing the on-disk walk that fires when
    /// the DB is `None`.
    #[tokio::test]
    async fn gallery_list_prefers_metadata_db_when_populated() {
        use mold_db::{GenerationRecord, MetadataDb, RecordSource};

        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("real.png"), b"fake-bytes").unwrap();

        // Pre-populate the DB with a row that wouldn't otherwise survive
        // the on-disk validator (size below the 256-byte floor) — proves
        // the response came from the DB and not the filesystem walk.
        let db_path = dir.path().join("mold.db");
        let db = MetadataDb::open(&db_path).unwrap();
        let metadata = mold_core::OutputMetadata {
            prompt: "from db".into(),
            negative_prompt: None,
            original_prompt: None,
            model: "mock-model".into(),
            seed: 7,
            steps: 4,
            guidance: 1.0,
            width: 64,
            height: 64,
            strength: None,
            scheduler: None,
            output_format: Some(mold_core::OutputFormat::Png),
            cfg_plus: None,
            lora: None,
            lora_scale: None,
            loras: None,
            control_model: None,
            control_scale: None,
            upscale_model: None,
            gif_preview: None,
            enable_audio: None,
            audio_file_path: None,
            source_video_path: None,
            pipeline: None,
            retake_range: None,
            spatial_upscale: None,
            temporal_upscale: None,
            frames: None,
            fps: None,
            version: "test".into(),
        };
        let mut rec = GenerationRecord::from_save(
            dir.path(),
            "real.png",
            mold_core::OutputFormat::Png,
            metadata,
            RecordSource::Server,
            1_700_000_000_000,
        );
        rec.file_mtime_ms = Some(1_700_000_000_000);
        rec.file_size_bytes = Some(10);
        db.upsert(&rec).unwrap();

        // Build state with the DB attached and a config that points at our
        // gallery dir.
        let config = mold_core::Config {
            output_dir: Some(dir.path().to_string_lossy().into_owned()),
            ..Default::default()
        };
        let (tx, _rx) = tokio::sync::mpsc::channel(16);
        let queue = crate::state::QueueHandle::new(tx);
        let gpu_pool = std::sync::Arc::new(crate::gpu_pool::GpuPool {
            workers: Vec::new(),
        });
        let mut state = AppState::empty(config, queue, gpu_pool, 200);
        state.metadata_db = std::sync::Arc::new(Some(db));
        let app = app_with_state(state);

        let resp = app
            .oneshot(
                Request::builder()
                    .uri("/api/gallery")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
        let body = json_body(resp).await;
        let arr = body.as_array().expect("array response");
        assert_eq!(arr.len(), 1, "DB-backed listing should return our row");
        assert_eq!(arr[0]["filename"], "real.png");
        assert_eq!(arr[0]["metadata"]["prompt"], "from db");
        assert_eq!(arr[0]["metadata"]["seed"], 7);
    }

    /// Without a DB attached, the gallery list falls back to the on-disk
    /// walk + header validation. Files below the size floor / with bad
    /// headers should be filtered out, just like the historical behavior.
    #[tokio::test]
    async fn gallery_list_falls_back_to_filesystem_when_db_absent() {
        let dir = tempfile::tempdir().unwrap();
        // Below the 256 B floor → filtered.
        std::fs::write(dir.path().join("tiny.png"), b"x").unwrap();

        let config = mold_core::Config {
            output_dir: Some(dir.path().to_string_lossy().into_owned()),
            ..Default::default()
        };
        let (tx, _rx) = tokio::sync::mpsc::channel(16);
        let queue = crate::state::QueueHandle::new(tx);
        let gpu_pool = std::sync::Arc::new(crate::gpu_pool::GpuPool {
            workers: Vec::new(),
        });
        let state = AppState::empty(config, queue, gpu_pool, 200);
        let app = app_with_state(state);

        let resp = app
            .oneshot(
                Request::builder()
                    .uri("/api/gallery")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
        let body = json_body(resp).await;
        assert_eq!(
            body.as_array().unwrap().len(),
            0,
            "filesystem fallback must still apply size/header validation"
        );
    }

    /// `GET /api/gallery/preview/:filename` serves the cached `.preview.gif`
    /// the TUI's server-backed detail pane pulls when it wants to animate an
    /// MP4 entry. Happy path: the file exists → 200 with `image/gif` + the
    /// bytes. Missing file → 404 so the client can fall back to the full
    /// `/api/gallery/image/:filename` path.
    #[allow(clippy::await_holding_lock)]
    #[tokio::test]
    async fn gallery_preview_returns_gif_when_present_and_404_otherwise() {
        // Route is backed by `MOLD_HOME/cache/previews/<filename>.preview.gif`,
        // so pin MOLD_HOME at a tempdir for the duration of the test — and
        // hold env_lock so parallel tests can't race us.
        let _lock = env_lock().lock().unwrap();
        let mold_home = tempfile::tempdir().unwrap();
        let prev = std::env::var("MOLD_HOME").ok();
        unsafe {
            std::env::set_var("MOLD_HOME", mold_home.path());
        }

        // Plant a minimal valid GIF (header only) at the path the handler
        // will look for. The handler doesn't decode — it streams bytes back
        // verbatim — so this suffices as a regression fixture.
        const GIF: &[u8] = b"GIF89a\x01\x00\x01\x00\x00\x00\x00\x3b";
        let previews = mold_home.path().join("cache").join("previews");
        std::fs::create_dir_all(&previews).unwrap();
        std::fs::write(previews.join("ltx2-has-preview.mp4.preview.gif"), GIF).unwrap();
        // Also plant an orphaned preview whose source MP4 doesn't exist —
        // the endpoint must 404 it rather than leak the sidecar bytes.
        std::fs::write(previews.join("ltx2-orphan.mp4.preview.gif"), GIF).unwrap();

        let output_dir = tempfile::tempdir().unwrap();
        // Source MP4 must exist in the gallery dir for the endpoint to
        // serve its preview — the cache is tied to the file lifecycle.
        std::fs::write(output_dir.path().join("ltx2-has-preview.mp4"), b"fake-mp4").unwrap();
        let config = mold_core::Config {
            output_dir: Some(output_dir.path().to_string_lossy().into_owned()),
            ..Default::default()
        };
        let (tx, _rx) = tokio::sync::mpsc::channel(16);
        let queue = crate::state::QueueHandle::new(tx);
        let gpu_pool = std::sync::Arc::new(crate::gpu_pool::GpuPool {
            workers: Vec::new(),
        });
        let state = AppState::empty(config, queue, gpu_pool, 200);
        let app = crate::routes::create_router(state);

        // Source present + sidecar present → 200 with image/gif + bytes.
        let resp = app
            .clone()
            .oneshot(
                Request::builder()
                    .uri("/api/gallery/preview/ltx2-has-preview.mp4")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
        assert_eq!(
            resp.headers()
                .get(axum::http::header::CONTENT_TYPE)
                .and_then(|v| v.to_str().ok()),
            Some("image/gif")
        );
        let body = axum::body::to_bytes(resp.into_body(), 1024 * 1024)
            .await
            .unwrap();
        assert_eq!(body.as_ref(), GIF);

        // Missing entirely → 404.
        let resp = app
            .clone()
            .oneshot(
                Request::builder()
                    .uri("/api/gallery/preview/ltx2-missing.mp4")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::NOT_FOUND);

        // Orphaned sidecar (source MP4 deleted, GIF still on disk) → 404.
        // Regression guard: previously this returned the stale bytes.
        let resp = app
            .oneshot(
                Request::builder()
                    .uri("/api/gallery/preview/ltx2-orphan.mp4")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::NOT_FOUND);

        // Restore MOLD_HOME.
        unsafe {
            match prev {
                Some(v) => std::env::set_var("MOLD_HOME", v),
                None => std::env::remove_var("MOLD_HOME"),
            }
        }
        drop(_lock);
    }

    /// `DELETE /api/gallery/image/:filename` must remove the matching DB
    /// row in addition to the file on disk so the next list call doesn't
    /// resurrect a stale entry from cache.
    #[allow(clippy::await_holding_lock)]
    #[tokio::test]
    async fn gallery_delete_drops_metadata_row() {
        use mold_db::{GenerationRecord, MetadataDb, RecordSource};

        let dir = tempfile::tempdir().unwrap();
        let target = dir.path().join("doomed.png");
        std::fs::write(&target, vec![0u8; 1024]).unwrap();

        let db = MetadataDb::open(&dir.path().join("mold.db")).unwrap();
        let metadata = mold_core::OutputMetadata {
            prompt: "doomed".into(),
            negative_prompt: None,
            original_prompt: None,
            model: "m".into(),
            seed: 0,
            steps: 0,
            guidance: 0.0,
            width: 1,
            height: 1,
            strength: None,
            scheduler: None,
            output_format: Some(mold_core::OutputFormat::Png),
            cfg_plus: None,
            lora: None,
            lora_scale: None,
            loras: None,
            control_model: None,
            control_scale: None,
            upscale_model: None,
            gif_preview: None,
            enable_audio: None,
            audio_file_path: None,
            source_video_path: None,
            pipeline: None,
            retake_range: None,
            spatial_upscale: None,
            temporal_upscale: None,
            frames: None,
            fps: None,
            version: "t".into(),
        };
        let rec = GenerationRecord::from_save(
            dir.path(),
            "doomed.png",
            mold_core::OutputFormat::Png,
            metadata,
            RecordSource::Server,
            0,
        );
        db.upsert(&rec).unwrap();
        assert_eq!(db.count().unwrap(), 1);

        let config = mold_core::Config {
            output_dir: Some(dir.path().to_string_lossy().into_owned()),
            ..Default::default()
        };
        let (tx, _rx) = tokio::sync::mpsc::channel(16);
        let queue = crate::state::QueueHandle::new(tx);
        let gpu_pool = std::sync::Arc::new(crate::gpu_pool::GpuPool {
            workers: Vec::new(),
        });
        let mut state = AppState::empty(config, queue, gpu_pool, 200);
        state.metadata_db = std::sync::Arc::new(Some(db));
        let db_handle_for_assert = state.metadata_db.clone();
        let app = app_with_state(state);

        // Delete is always enabled — no env var gating.
        let resp = app
            .oneshot(
                Request::builder()
                    .method("DELETE")
                    .uri("/api/gallery/image/doomed.png")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::NO_CONTENT);
        assert!(!target.exists(), "file should be removed from disk");
        let db_after = db_handle_for_assert.as_ref().as_ref().unwrap();
        assert_eq!(db_after.count().unwrap(), 0, "DB row should be gone");
    }

    #[tokio::test]
    async fn put_model_placement_updates_config_and_persists() {
        let _lock = env_lock().lock().unwrap_or_else(|e| e.into_inner());
        let tmp = tempfile::tempdir().unwrap();
        std::env::set_var("MOLD_HOME", tmp.path());
        let app = app_empty();
        // Re-create state inside this test with mutable access.
        let (tx, _rx) = tokio::sync::mpsc::channel(16);
        let queue = crate::state::QueueHandle::new(tx);
        let gpu_pool = std::sync::Arc::new(crate::gpu_pool::GpuPool {
            workers: Vec::new(),
        });
        let state = AppState::empty(mold_core::Config::default(), queue, gpu_pool, 200);
        let app = {
            let _ = app;
            crate::routes::create_router(state.clone())
        };

        let body = serde_json::json!({
            "text_encoders": { "kind": "cpu" },
            "advanced": {
                "transformer": { "kind": "gpu", "ordinal": 1 },
                "vae": { "kind": "auto" },
                "t5": { "kind": "cpu" }
            }
        });

        let resp = app
            .clone()
            .oneshot(
                Request::builder()
                    .method("PUT")
                    .uri("/api/config/model/flux-dev%3Aq4/placement")
                    .header("content-type", "application/json")
                    .body(Body::from(body.to_string()))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::OK);

        let cfg = state.config.read().await;
        let p = cfg
            .models
            .get("flux-dev:q4")
            .and_then(|m| m.placement.clone())
            .expect("placement not persisted");
        assert_eq!(p.text_encoders, mold_core::types::DeviceRef::Cpu);
        let adv = p.advanced.unwrap();
        assert_eq!(adv.transformer, mold_core::types::DeviceRef::gpu(1));
        std::env::remove_var("MOLD_HOME");
    }

    #[tokio::test]
    async fn put_model_placement_returns_500_when_save_fails() {
        let _lock = env_lock().lock().unwrap_or_else(|e| e.into_inner());
        // Point MOLD_HOME at a regular file so `config.toml` cannot be created
        // underneath it — `Config::save()` must return `Err`.
        let tmp = tempfile::tempdir().unwrap();
        let blocker = tmp.path().join("not-a-dir");
        std::fs::write(&blocker, "blocker").unwrap();
        std::env::set_var("MOLD_HOME", &blocker);

        let (tx, _rx) = tokio::sync::mpsc::channel(16);
        let queue = crate::state::QueueHandle::new(tx);
        let gpu_pool = std::sync::Arc::new(crate::gpu_pool::GpuPool {
            workers: Vec::new(),
        });
        let state = AppState::empty(mold_core::Config::default(), queue, gpu_pool, 200);
        let app = crate::routes::create_router(state.clone());

        let body = serde_json::json!({
            "text_encoders": { "kind": "cpu" }
        });
        let resp = app
            .clone()
            .oneshot(
                Request::builder()
                    .method("PUT")
                    .uri("/api/config/model/flux-dev%3Aq4/placement")
                    .header("content-type", "application/json")
                    .body(Body::from(body.to_string()))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::INTERNAL_SERVER_ERROR);

        let del_resp = app
            .oneshot(
                Request::builder()
                    .method("DELETE")
                    .uri("/api/config/model/flux-dev%3Aq4/placement")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(del_resp.status(), StatusCode::INTERNAL_SERVER_ERROR);

        std::env::remove_var("MOLD_HOME");
    }

    #[tokio::test]
    async fn put_model_placement_rejects_malformed_body() {
        let _lock = env_lock().lock().unwrap_or_else(|e| e.into_inner());
        let app = app_empty();
        let resp = app
            .oneshot(
                Request::builder()
                    .method("PUT")
                    .uri("/api/config/model/flux-dev%3Aq4/placement")
                    .header("content-type", "application/json")
                    .body(Body::from(r#"{"text_encoders":"not-an-object"}"#))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert!(
            resp.status() == StatusCode::BAD_REQUEST
                || resp.status() == StatusCode::UNPROCESSABLE_ENTITY,
            "got status {}",
            resp.status()
        );
    }

    #[tokio::test]
    async fn put_model_placement_rejects_gpu_outside_worker_pool() {
        let _lock = env_lock().lock().unwrap_or_else(|e| e.into_inner());
        let (tx, _rx) = tokio::sync::mpsc::channel(16);
        let queue = crate::state::QueueHandle::new(tx);
        let gpu_pool = Arc::new(crate::gpu_pool::GpuPool {
            workers: vec![gpu_worker_stub(1)],
        });
        let state = AppState::empty(mold_core::Config::default(), queue, gpu_pool, 200);
        let app = crate::routes::create_router(state);

        let body = serde_json::json!({
            "text_encoders": { "kind": "auto" },
            "advanced": {
                "transformer": { "kind": "gpu", "ordinal": 0 },
                "vae": { "kind": "auto" }
            }
        });

        let resp = app
            .oneshot(
                Request::builder()
                    .method("PUT")
                    .uri("/api/config/model/flux-dev%3Aq4/placement")
                    .header("content-type", "application/json")
                    .body(Body::from(body.to_string()))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::UNPROCESSABLE_ENTITY);
        let body = json_body(resp).await;
        assert!(body["error"].as_str().unwrap().contains("gpu:0"));
    }

    #[tokio::test]
    async fn generate_rejects_gpu_outside_worker_pool() {
        let app = app_with_worker_pool(MockEngine::ready(), &[1]);
        let body = serde_json::json!({
            "prompt": "a cat",
            "model": "mock-model",
            "width": 512,
            "height": 512,
            "steps": 4,
            "batch_size": 1,
            "output_format": "png",
            "placement": {
                "text_encoders": { "kind": "auto" },
                "advanced": {
                    "transformer": { "kind": "gpu", "ordinal": 0 },
                    "vae": { "kind": "auto" }
                }
            }
        });

        let resp = app
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/api/generate")
                    .header("content-type", "application/json")
                    .body(Body::from(body.to_string()))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::UNPROCESSABLE_ENTITY);
        let body = json_body(resp).await;
        assert!(body["error"].as_str().unwrap().contains("gpu:0"));
    }
    // ─── Downloads UI (Agent A) ─────────────────────────────────────────────

    #[tokio::test]
    async fn post_api_downloads_enqueues_job() {
        let state = AppState::empty(
            mold_core::Config::default(),
            crate::state::QueueHandle::new(tokio::sync::mpsc::channel(1).0),
            AppState::empty_gpu_pool_for_test(),
            200,
        );
        let app = app_with_state(state.clone());

        let body = serde_json::json!({ "model": "flux-schnell:q4" });
        let req = Request::builder()
            .method("POST")
            .uri("/api/downloads")
            .header("content-type", "application/json")
            .body(Body::from(body.to_string()))
            .unwrap();

        let res = app.oneshot(req).await.unwrap();
        assert_eq!(res.status(), StatusCode::OK);

        let bytes = axum::body::to_bytes(res.into_body(), 64 * 1024)
            .await
            .unwrap();
        let v: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
        assert!(v.get("id").and_then(|x| x.as_str()).is_some());
        assert!(v.get("position").and_then(|x| x.as_u64()).is_some());

        let listing = state.downloads.listing().await;
        // No driver running in this test, so the job sits in `queued`.
        assert_eq!(listing.queued.len(), 1);
        assert_eq!(listing.queued[0].model, "flux-schnell:q4");
    }

    #[tokio::test]
    async fn post_api_downloads_unknown_model_400() {
        let state = AppState::empty(
            mold_core::Config::default(),
            crate::state::QueueHandle::new(tokio::sync::mpsc::channel(1).0),
            AppState::empty_gpu_pool_for_test(),
            200,
        );
        let app = app_with_state(state);
        let body = serde_json::json!({ "model": "not-a-real-model:xyz" });
        let req = Request::builder()
            .method("POST")
            .uri("/api/downloads")
            .header("content-type", "application/json")
            .body(Body::from(body.to_string()))
            .unwrap();
        let res = app.oneshot(req).await.unwrap();
        assert_eq!(res.status(), StatusCode::BAD_REQUEST);
    }

    #[tokio::test]
    async fn post_api_downloads_duplicate_is_idempotent_409() {
        let state = AppState::empty(
            mold_core::Config::default(),
            crate::state::QueueHandle::new(tokio::sync::mpsc::channel(1).0),
            AppState::empty_gpu_pool_for_test(),
            200,
        );
        let app = app_with_state(state.clone());

        let body = serde_json::json!({ "model": "flux-schnell:q4" });
        let make_req = || {
            Request::builder()
                .method("POST")
                .uri("/api/downloads")
                .header("content-type", "application/json")
                .body(Body::from(body.to_string()))
                .unwrap()
        };

        let res1 = app.clone().oneshot(make_req()).await.unwrap();
        assert_eq!(res1.status(), StatusCode::OK);
        let bytes1 = axum::body::to_bytes(res1.into_body(), 64 * 1024)
            .await
            .unwrap();
        let v1: serde_json::Value = serde_json::from_slice(&bytes1).unwrap();
        let id1 = v1["id"].as_str().unwrap().to_string();

        let res2 = app.oneshot(make_req()).await.unwrap();
        assert_eq!(res2.status(), StatusCode::CONFLICT);
        let bytes2 = axum::body::to_bytes(res2.into_body(), 64 * 1024)
            .await
            .unwrap();
        let v2: serde_json::Value = serde_json::from_slice(&bytes2).unwrap();
        let id2 = v2["id"].as_str().unwrap().to_string();

        assert_eq!(id1, id2, "duplicate enqueue must return the same id");
    }

    #[tokio::test]
    async fn delete_api_downloads_204_for_queued() {
        let state = AppState::empty(
            mold_core::Config::default(),
            crate::state::QueueHandle::new(tokio::sync::mpsc::channel(1).0),
            AppState::empty_gpu_pool_for_test(),
            200,
        );
        let app = app_with_state(state.clone());

        let (id, _, _) = state
            .downloads
            .enqueue("flux-schnell:q4".into())
            .await
            .unwrap();

        let req = Request::builder()
            .method("DELETE")
            .uri(format!("/api/downloads/{id}"))
            .body(Body::empty())
            .unwrap();
        let res = app.oneshot(req).await.unwrap();
        assert_eq!(res.status(), StatusCode::NO_CONTENT);

        let listing = state.downloads.listing().await;
        assert!(listing.queued.is_empty());
    }

    #[tokio::test]
    async fn delete_api_downloads_404_when_unknown() {
        let state = AppState::empty(
            mold_core::Config::default(),
            crate::state::QueueHandle::new(tokio::sync::mpsc::channel(1).0),
            AppState::empty_gpu_pool_for_test(),
            200,
        );
        let app = app_with_state(state);
        let req = Request::builder()
            .method("DELETE")
            .uri("/api/downloads/nonexistent-id")
            .body(Body::empty())
            .unwrap();
        let res = app.oneshot(req).await.unwrap();
        assert_eq!(res.status(), StatusCode::NOT_FOUND);
    }

    #[tokio::test]
    async fn get_api_downloads_returns_listing_shape() {
        let state = AppState::empty(
            mold_core::Config::default(),
            crate::state::QueueHandle::new(tokio::sync::mpsc::channel(1).0),
            AppState::empty_gpu_pool_for_test(),
            200,
        );
        let app = app_with_state(state.clone());

        let _ = state
            .downloads
            .enqueue("flux-schnell:q4".into())
            .await
            .unwrap();

        let req = Request::builder()
            .uri("/api/downloads")
            .body(Body::empty())
            .unwrap();
        let res = app.oneshot(req).await.unwrap();
        assert_eq!(res.status(), StatusCode::OK);

        let bytes = axum::body::to_bytes(res.into_body(), 64 * 1024)
            .await
            .unwrap();
        let v: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
        assert!(v["queued"].is_array());
        assert!(v["history"].is_array());
        assert_eq!(v["queued"].as_array().unwrap().len(), 1);
        assert_eq!(v["queued"][0]["model"], "flux-schnell:q4");
    }

    #[tokio::test]
    async fn sse_stream_emits_enqueued_event() {
        use futures::StreamExt as _;
        let state = AppState::empty(
            mold_core::Config::default(),
            crate::state::QueueHandle::new(tokio::sync::mpsc::channel(1).0),
            AppState::empty_gpu_pool_for_test(),
            200,
        );
        let app = app_with_state(state.clone());

        let req = Request::builder()
            .uri("/api/downloads/stream")
            .body(Body::empty())
            .unwrap();

        let res = app.oneshot(req).await.unwrap();
        assert_eq!(res.status(), StatusCode::OK);

        // Enqueue AFTER subscribing (SSE response already established).
        let state_for_send = state.clone();
        tokio::spawn(async move {
            tokio::time::sleep(std::time::Duration::from_millis(50)).await;
            let _ = state_for_send
                .downloads
                .enqueue("flux-schnell:q4".into())
                .await;
        });

        let mut body = res.into_body().into_data_stream();
        let mut saw_enqueued = false;
        let deadline = tokio::time::Instant::now() + std::time::Duration::from_secs(2);
        while tokio::time::Instant::now() < deadline {
            match tokio::time::timeout(std::time::Duration::from_millis(300), body.next()).await {
                Ok(Some(Ok(bytes))) => {
                    let text = String::from_utf8_lossy(&bytes).to_string();
                    if text.contains("\"type\":\"enqueued\"") {
                        saw_enqueued = true;
                        break;
                    }
                }
                _ => continue,
            }
        }
        assert!(saw_enqueued, "did not observe an 'enqueued' SSE event");
    }

    // ── Resource telemetry (Agent B) ────────────────────────────────────────

    #[tokio::test]
    async fn get_api_resources_returns_snapshot() {
        let _lock = env_lock().lock().unwrap_or_else(|e| e.into_inner());
        let state = AppState::empty(
            mold_core::Config::default(),
            crate::state::QueueHandle::new(tokio::sync::mpsc::channel(1).0),
            std::sync::Arc::new(crate::gpu_pool::GpuPool {
                workers: Vec::new(),
            }),
            200,
        );
        // Seed the broadcaster so the endpoint has something to return.
        state.resources.publish(mold_core::ResourceSnapshot {
            hostname: "unit-test".into(),
            timestamp: 12345,
            gpus: vec![],
            system_ram: mold_core::RamSnapshot {
                total: 1,
                used: 0,
                used_by_mold: 0,
                used_by_other: 0,
            },
            cpu: None,
        });

        let app = create_router(state);
        let req = Request::builder()
            .uri("/api/resources")
            .body(Body::empty())
            .unwrap();
        let resp = app.oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
        let body = json_body(resp).await;
        assert_eq!(body["hostname"], "unit-test");
        assert_eq!(body["timestamp"], 12345);
        assert!(body["system_ram"].is_object());
    }

    #[tokio::test]
    async fn get_api_resources_stream_sets_sse_content_type() {
        let _lock = env_lock().lock().unwrap_or_else(|e| e.into_inner());
        let state = AppState::empty(
            mold_core::Config::default(),
            crate::state::QueueHandle::new(tokio::sync::mpsc::channel(1).0),
            std::sync::Arc::new(crate::gpu_pool::GpuPool {
                workers: Vec::new(),
            }),
            200,
        );
        let app = create_router(state);
        let req = Request::builder()
            .uri("/api/resources/stream")
            .body(Body::empty())
            .unwrap();
        let resp = app.oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
        let ct = resp
            .headers()
            .get(axum::http::header::CONTENT_TYPE)
            .and_then(|v| v.to_str().ok())
            .unwrap_or("");
        assert!(
            ct.starts_with("text/event-stream"),
            "expected SSE content-type, got: {ct}"
        );
    }

    #[tokio::test]
    async fn get_api_resources_returns_503_before_first_tick() {
        let _lock = env_lock().lock().unwrap_or_else(|e| e.into_inner());
        let state = AppState::empty(
            mold_core::Config::default(),
            crate::state::QueueHandle::new(tokio::sync::mpsc::channel(1).0),
            std::sync::Arc::new(crate::gpu_pool::GpuPool {
                workers: Vec::new(),
            }),
            200,
        );
        // Do NOT publish — broadcaster has no cached snapshot.
        let app = create_router(state);
        let req = Request::builder()
            .uri("/api/resources")
            .body(Body::empty())
            .unwrap();
        let resp = app.oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::SERVICE_UNAVAILABLE);
    }

    // ── /api/capabilities/chain-limits ──────────────────────────────────────

    #[tokio::test]
    async fn capabilities_chain_limits_returns_ltx2_cap() {
        let app = app_empty();
        let response = app
            .oneshot(
                Request::get("/api/capabilities/chain-limits?model=ltx-2-19b-distilled:fp8")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::OK);
        let body = axum::body::to_bytes(response.into_body(), usize::MAX)
            .await
            .unwrap();
        let limits: serde_json::Value = serde_json::from_slice(&body).unwrap();
        assert_eq!(limits["frames_per_clip_cap"], 97);
        assert_eq!(limits["max_stages"], 16);
        assert!(limits["transition_modes"]
            .as_array()
            .unwrap()
            .contains(&serde_json::Value::String("fade".into())));
        assert_eq!(
            limits["supports_audio"], true,
            "ltx2 family advertises audio so the SPA can show the toggle",
        );
    }

    #[tokio::test]
    async fn capabilities_chain_limits_rejects_unknown_model() {
        let app = app_empty();
        let response = app
            .oneshot(
                Request::get("/api/capabilities/chain-limits?model=not-a-real-model")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::NOT_FOUND);
    }
}
