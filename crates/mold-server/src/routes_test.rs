// Tests use `std::sync::Mutex<()>` to serialize process-global env-var
// mutations; holding the guard across `.await` is intentional under the
// current-thread tokio test runtime.
#![allow(clippy::await_holding_lock)]

#[cfg(test)]
mod tests {
    use axum::{
        body::Body,
        http::{Method, Request, StatusCode},
    };
    use base64::Engine as _;
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

    /// Serialize tests that mutate process-global mold env vars.
    /// Uses std::sync::Mutex (not tokio) so it works across independent
    /// tokio runtimes that #[tokio::test] creates per test.
    fn env_lock() -> &'static std::sync::Mutex<()> {
        crate::test_support::env_lock()
    }

    /// Parse response body as JSON and return the value.
    async fn json_body(resp: axum::http::Response<Body>) -> serde_json::Value {
        let body = axum::body::to_bytes(resp.into_body(), 1024 * 1024)
            .await
            .unwrap();
        serde_json::from_slice(&body).unwrap()
    }

    /// Extract and parse the JSON payload for the first named event in an SSE body.
    fn sse_json_event(body: &str, event_name: &str) -> serde_json::Value {
        let mut lines = body.lines();
        let event_line = format!("event: {event_name}");
        while let Some(line) = lines.next() {
            if line == event_line {
                let data = lines
                    .next()
                    .and_then(|line| line.strip_prefix("data: "))
                    .unwrap_or_else(|| {
                        panic!("{event_name} event should have a data line: {body}")
                    });
                return serde_json::from_str(data).unwrap_or_else(|err| {
                    panic!("{event_name} data should be JSON: {err}: {data}")
                });
            }
        }
        panic!("SSE body should contain event {event_name}: {body}");
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
            workers: Vec::new().into(),
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
            original_prompt: None,
            batch_id: None,
            batch_index: None,
            batch_count: None,
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
        gpu_worker_stub_with_receiver(ordinal).0
    }

    fn gpu_worker_stub_with_receiver(
        ordinal: usize,
    ) -> (
        Arc<crate::gpu_pool::GpuWorker>,
        std::sync::mpsc::Receiver<crate::gpu_pool::GpuWorkerCommand>,
    ) {
        let (job_tx, job_rx) = std::sync::mpsc::sync_channel(1);
        let worker = Arc::new(crate::gpu_pool::GpuWorker {
            owner_epoch: 1,
            gpu: mold_inference::device::DiscoveredGpu {
                ordinal,
                stable_id: Some(format!("cuda:{ordinal:032x}")),
                raw_cuda_uuid: Some((ordinal as u128).to_be_bytes()),
                device_kind: Some(mold_inference::device::CudaDeviceKind::UnknownCuda),
                identity_error: None,
                backend: mold_core::types::GpuBackend::Cuda,
                name: format!("gpu{ordinal}"),
                compute_capability: Some((8, 6)),
                pci_bus_id: None,
                total_vram_bytes: 24_000_000_000,
                free_vram_bytes: 24_000_000_000,
            },
            model_cache: Arc::new(Mutex::new(crate::model_cache::ModelCache::new(3))),
            resident_model: Arc::new(RwLock::new(None)),
            resident_execution_fingerprint: Arc::new(RwLock::new(None)),
            active_generation: Arc::new(RwLock::new(None)),
            model_load_lock: Arc::new(Mutex::new(())),
            shared_pool: Arc::new(Mutex::new(mold_inference::shared_pool::SharedPool::new())),
            legacy_pending: AtomicUsize::new(0),
            in_flight: AtomicUsize::new(0),
            legacy_chain_waiters: Default::default(),
            consecutive_failures: AtomicUsize::new(0),
            poisoned: AtomicBool::new(false),
            fatal_cuda_error: Arc::new(AtomicBool::new(false)),
            fatal_cuda_shutdown: Arc::new(tokio::sync::Notify::new()),
            shutdown_requested: AtomicBool::new(false),
            drain_state: std::sync::atomic::AtomicU8::new(crate::gpu_pool::DRAIN_RUNNING),
            owner_thread_id: std::sync::OnceLock::new(),
            degraded_until: RwLock::new(None),
            job_tx,
        });
        (worker, job_rx)
    }

    fn install_worker_registry(state: &mut AppState) {
        let devices = state
            .gpu_pool
            .worker_snapshot()
            .into_iter()
            .map(|worker| crate::device_registry::DiscoveredDevice {
                stable_id: worker.gpu.stable_id.clone(),
                backend: worker.gpu.backend,
                visible_ordinal: Some(worker.gpu.ordinal),
                device_kind: mold_core::DeviceKind::UnknownCuda,
                nvml_uuid: None,
                physical_uuid: None,
                mig_uuid: None,
                mig_parent_uuid: None,
                mig_profile: None,
                pci_bus_id: worker.gpu.pci_bus_id.clone(),
                name: worker.gpu.name.clone(),
                compute_capability: worker.gpu.compute_capability,
                total_memory_bytes: Some(worker.gpu.total_vram_bytes),
                startup_allowed: true,
                telemetry_ordinal: Some(worker.gpu.ordinal),
            })
            .collect();
        state.device_registry = Arc::new(crate::device_registry::DeviceRegistry::new(
            Arc::new(crate::device_registry::StaticDeviceDiscovery::new(devices)),
            Arc::new(None),
        ));
    }

    fn install_authoritative_v2(state: &mut AppState) {
        let (scheduled_tx, _scheduled_rx) = tokio::sync::mpsc::channel(1);
        state.scheduled_work = crate::scheduler::ScheduledWorkHandle::for_mode(
            scheduled_tx,
            crate::dispatch_mode::DispatchMode::V2,
        );
    }

    fn app_with_worker_pool(engine: MockEngine, ordinals: &[usize]) -> axum::Router {
        let mut state = AppState::with_engine(engine);
        state.gpu_pool = Arc::new(crate::gpu_pool::GpuPool {
            workers: ordinals
                .iter()
                .copied()
                .map(gpu_worker_stub)
                .collect::<Vec<_>>()
                .into(),
        });
        install_worker_registry(&mut state);
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

    #[tokio::test]
    async fn devices_returns_registry_workers_with_nullable_telemetry() {
        let worker = gpu_worker_stub(0);
        let mut state = AppState::with_engine(MockEngine::ready());
        state.gpu_pool = Arc::new(crate::gpu_pool::GpuPool {
            workers: vec![worker].into(),
        });
        state.device_registry = Arc::new(crate::device_registry::DeviceRegistry::new(
            Arc::new(crate::device_registry::StaticDeviceDiscovery::new(vec![
                crate::device_registry::DiscoveredDevice {
                    stable_id: Some("cuda:0123456789abcdef0123456789abcdef".into()),
                    backend: mold_core::GpuBackend::Cuda,
                    visible_ordinal: Some(0),
                    device_kind: mold_core::DeviceKind::FullGpu,
                    nvml_uuid: Some("GPU-01234567-89ab-cdef-0123-456789abcdef".into()),
                    physical_uuid: Some("GPU-01234567-89ab-cdef-0123-456789abcdef".into()),
                    mig_uuid: None,
                    mig_parent_uuid: None,
                    mig_profile: None,
                    pci_bus_id: Some("00000000:01:00.0".into()),
                    name: "NVIDIA GeForce RTX 3090".into(),
                    compute_capability: Some((8, 6)),
                    total_memory_bytes: Some(24 * 1024 * 1024 * 1024),
                    startup_allowed: true,
                    telemetry_ordinal: Some(0),
                },
            ])),
            Arc::new(None),
        ));

        let response = app_with_state(state)
            .oneshot(Request::get("/api/devices").body(Body::empty()).unwrap())
            .await
            .unwrap();

        assert_eq!(response.status(), StatusCode::OK);
        let body = json_body(response).await;
        assert_eq!(body["plan_version"], 0);
        assert_eq!(body["devices"].as_array().unwrap().len(), 1);
        assert_eq!(
            body["devices"][0]["id"],
            "cuda:0123456789abcdef0123456789abcdef"
        );
        assert_eq!(body["devices"][0]["device_kind"], "full_gpu");
        assert_eq!(body["devices"][0]["desired_enabled"], true);
        assert_eq!(body["devices"][0]["schedulable"], true);
        assert_eq!(
            body["devices"][0]["telemetry"]["utilization_percent"],
            serde_json::Value::Null
        );
        assert_eq!(
            body["devices"][0]["memory"]["used_bytes"],
            serde_json::Value::Null
        );
    }

    #[tokio::test]
    async fn v2_disable_can_be_recovered_after_restart_into_legacy() {
        let worker = gpu_worker_stub(0);
        let pool = Arc::new(crate::gpu_pool::GpuPool {
            workers: vec![worker].into(),
        });
        let mut state = AppState::with_engine(MockEngine::ready());
        state.gpu_pool = pool.clone();
        install_worker_registry(&mut state);
        install_authoritative_v2(&mut state);
        let registry = state.device_registry.clone();
        let mut events = state.events.subscribe();
        let app = app_with_state(state);
        let id = "cuda:00000000000000000000000000000000";

        let response = app
            .clone()
            .oneshot(
                Request::patch(format!("/api/devices/{id}"))
                    .header("content-type", "application/json")
                    .body(Body::from(r#"{"enabled":false}"#))
                    .unwrap(),
            )
            .await
            .unwrap();

        assert_eq!(response.status(), StatusCode::OK);
        let body = json_body(response).await;
        assert_eq!(body["desired_enabled"], false);
        assert_eq!(body["admin_state"], "disabled");
        assert!(pool.workers.is_empty());
        assert!(!registry.desired_enabled(id));
        assert!(matches!(
            events.recv().await.unwrap(),
            mold_core::ServerEvent::DeviceStateChanged {
                device_id,
                desired_enabled: false,
                admin_state: mold_core::DeviceAdminState::Disabled,
            } if device_id == id
        ));

        let generation = app
            .oneshot(
                Request::post("/api/generate")
                    .header("content-type", "application/json")
                    .body(Body::from(generate_body("maintenance", 64, 64)))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(generation.status(), StatusCode::SERVICE_UNAVAILABLE);
        assert_eq!(json_body(generation).await["code"], "NO_SCHEDULABLE_DEVICE");

        let mut legacy = AppState::with_engine(MockEngine::ready());
        legacy.gpu_pool = Arc::new(crate::gpu_pool::GpuPool {
            workers: Vec::new().into(),
        });
        legacy.device_registry = registry.clone();
        legacy.scheduled_work = crate::scheduler::ScheduledWorkHandle::for_runtime(
            tokio::sync::mpsc::channel(1).0,
            crate::dispatch_mode::DispatchMode::Legacy,
            false,
            false,
        );
        let recovery = app_with_state(legacy)
            .oneshot(
                Request::patch(format!("/api/devices/{id}"))
                    .header("content-type", "application/json")
                    .body(Body::from(r#"{"enabled":true}"#))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(recovery.status(), StatusCode::ACCEPTED);
        assert_eq!(json_body(recovery).await["restart_required"], true);
        assert!(registry.desired_enabled(id));
    }

    #[tokio::test]
    async fn busy_disable_drains_and_reenable_cancels_pending_stop() {
        let worker = gpu_worker_stub(0);
        worker.in_flight.store(1, Ordering::SeqCst);
        let pool = Arc::new(crate::gpu_pool::GpuPool {
            workers: vec![worker.clone()].into(),
        });
        let mut state = AppState::with_engine(MockEngine::ready());
        state.gpu_pool = pool.clone();
        install_worker_registry(&mut state);
        install_authoritative_v2(&mut state);
        let registry = state.device_registry.clone();
        let app = app_with_state(state);
        let id = "cuda:00000000000000000000000000000000";

        let draining = app
            .clone()
            .oneshot(
                Request::patch(format!("/api/devices/{id}"))
                    .header("content-type", "application/json")
                    .body(Body::from(r#"{"enabled":false}"#))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(draining.status(), StatusCode::ACCEPTED);
        assert_eq!(json_body(draining).await["admin_state"], "draining");
        assert_eq!(
            worker.drain_state.load(Ordering::SeqCst),
            crate::gpu_pool::DRAIN_REQUESTED
        );
        assert!(!worker.shutdown_requested.load(Ordering::SeqCst));

        let enabled = app
            .oneshot(
                Request::patch(format!("/api/devices/{id}"))
                    .header("content-type", "application/json")
                    .body(Body::from(r#"{"enabled":true}"#))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(enabled.status(), StatusCode::OK);
        assert_eq!(json_body(enabled).await["admin_state"], "enabled");
        assert!(!worker.shutdown_requested.load(Ordering::SeqCst));
        assert_eq!(
            worker.drain_state.load(Ordering::SeqCst),
            crate::gpu_pool::DRAIN_RUNNING
        );
        assert!(registry.desired_enabled(id));
    }

    #[tokio::test]
    async fn patch_remains_responsive_while_owner_admin_load_lease_is_blocked() {
        let (worker, job_rx) = gpu_worker_stub_with_receiver(0);
        let (event_tx, mut event_rx) = tokio::sync::mpsc::unbounded_channel();
        let owner = crate::gpu_worker::spawn_gpu_thread(
            worker.clone(),
            job_rx,
            event_tx,
            Duration::from_secs(60),
        );
        assert!(matches!(
            event_rx.recv().await,
            Some(crate::scheduler::WorkerEvent::Ready {
                worker_generation: 1,
                ..
            })
        ));
        assert!(worker.try_claim_in_flight());
        let (load_started_tx, load_started_rx) = std::sync::mpsc::sync_channel(1);
        let (load_resume_tx, load_resume_rx) = std::sync::mpsc::sync_channel(1);
        worker
            .send_grant(crate::gpu_pool::LeaseGrant {
                fence: crate::scheduler::LeaseFence {
                    work_id: "blocked-admin-route".to_string(),
                    device_id: crate::scheduler::worker_device_id(&worker),
                    owner_epoch: worker.owner_epoch,
                    state_version: 1,
                    plan_version: 1,
                    worker_generation: 1,
                    memory_sample_generation: 1,
                    memory_ledger_sequence: 1,
                },
                work: crate::gpu_pool::OwnerWork::Probe {
                    id: "blocked-admin-route".to_string(),
                    kind: mold_scheduler::WorkKind::AdminModelLoad,
                    run: Box::new(move || {
                        load_started_tx.send(()).unwrap();
                        load_resume_rx.recv().unwrap();
                    }),
                },
                retry: None,
            })
            .unwrap();
        assert!(matches!(
            event_rx.recv().await,
            Some(crate::scheduler::WorkerEvent::Accepted { .. })
        ));
        assert!(matches!(
            event_rx.recv().await,
            Some(crate::scheduler::WorkerEvent::AllocationCommitted { .. })
        ));
        load_started_rx
            .recv_timeout(Duration::from_secs(1))
            .expect("admin lease must be executing");

        let pool = Arc::new(crate::gpu_pool::GpuPool {
            workers: vec![worker.clone()].into(),
        });
        let mut state = AppState::with_engine(MockEngine::ready());
        state.gpu_pool = pool;
        install_worker_registry(&mut state);
        install_authoritative_v2(&mut state);
        let app = app_with_state(state);
        let id = "cuda:00000000000000000000000000000000";
        let draining = tokio::time::timeout(
            Duration::from_millis(250),
            app.clone().oneshot(
                Request::patch(format!("/api/devices/{id}"))
                    .header("content-type", "application/json")
                    .body(Body::from(r#"{"enabled":false}"#))
                    .unwrap(),
            ),
        )
        .await
        .expect("PATCH must not join a busy GPU owner")
        .unwrap();
        assert_eq!(draining.status(), StatusCode::ACCEPTED);
        assert_eq!(json_body(draining).await["admin_state"], "draining");
        let devices = tokio::time::timeout(
            Duration::from_millis(250),
            app.oneshot(Request::get("/api/devices").body(Body::empty()).unwrap()),
        )
        .await
        .expect("device reads must remain responsive during drain")
        .unwrap();
        assert_eq!(devices.status(), StatusCode::OK);

        load_resume_tx.send(()).unwrap();
        assert!(matches!(
            event_rx.recv().await,
            Some(crate::scheduler::WorkerEvent::Completed { .. })
        ));
        assert!(matches!(
            event_rx.recv().await,
            Some(crate::scheduler::WorkerEvent::Stopped { .. })
        ));
        owner.join().unwrap();
        assert_eq!(worker.pending_or_executing(), 0);
    }

    #[tokio::test]
    async fn nonauthoritative_device_patch_rejects_before_state_or_event_mutation() {
        let cases = [
            (
                "legacy",
                crate::scheduler::ScheduledWorkHandle::for_runtime(
                    tokio::sync::mpsc::channel(1).0,
                    crate::dispatch_mode::DispatchMode::Legacy,
                    false,
                    false,
                ),
            ),
            (
                "observe",
                crate::scheduler::ScheduledWorkHandle::for_runtime(
                    tokio::sync::mpsc::channel(1).0,
                    crate::dispatch_mode::DispatchMode::Observe,
                    false,
                    true,
                ),
            ),
            (
                "v2-maintenance",
                crate::scheduler::ScheduledWorkHandle::for_runtime(
                    tokio::sync::mpsc::channel(1).0,
                    crate::dispatch_mode::DispatchMode::V2,
                    false,
                    false,
                ),
            ),
            (
                "v2-unavailable",
                crate::scheduler::ScheduledWorkHandle::default(),
            ),
        ];
        for (label, scheduled_work) in cases {
            let worker = gpu_worker_stub(0);
            worker.in_flight.store(1, Ordering::SeqCst);
            let pool = Arc::new(crate::gpu_pool::GpuPool {
                workers: vec![worker.clone()].into(),
            });
            let mut state = AppState::with_engine(MockEngine::ready());
            state.gpu_pool = pool.clone();
            install_worker_registry(&mut state);
            state.scheduled_work = scheduled_work;
            let registry = state.device_registry.clone();
            let mut events = state.events.subscribe();
            let app = app_with_state(state);
            let id = "cuda:00000000000000000000000000000000";

            let response = app
                .oneshot(
                    Request::patch(format!("/api/devices/{id}"))
                        .header("content-type", "application/json")
                        .body(Body::from(r#"{"enabled":false}"#))
                        .unwrap(),
                )
                .await
                .unwrap();

            assert_eq!(response.status(), StatusCode::CONFLICT, "{label}");
            assert_eq!(
                json_body(response).await["code"],
                "DEVICE_LIFECYCLE_MODE_CONFLICT"
            );
            assert!(registry.desired_enabled(id), "{label}");
            assert_eq!(pool.workers.len(), 1, "{label}");
            assert_eq!(
                worker.drain_state.load(Ordering::SeqCst),
                crate::gpu_pool::DRAIN_RUNNING,
                "{label}"
            );
            assert!(!worker.shutdown_requested.load(Ordering::SeqCst));
            let lifecycle_event = events.try_recv();
            assert!(
                matches!(
                    lifecycle_event,
                    Err(tokio::sync::broadcast::error::TryRecvError::Empty)
                        | Err(tokio::sync::broadcast::error::TryRecvError::Closed)
                ),
                "{label} published an event before rejecting: {lifecycle_event:?}"
            );
        }
    }

    #[tokio::test]
    async fn nonauthoritative_enable_persists_restart_recovery_without_live_mutation() {
        for (label, mode) in [
            ("legacy", crate::dispatch_mode::DispatchMode::Legacy),
            ("observe", crate::dispatch_mode::DispatchMode::Observe),
        ] {
            let worker = gpu_worker_stub(0);
            let pool = Arc::new(crate::gpu_pool::GpuPool {
                workers: vec![worker.clone()].into(),
            });
            let mut state = AppState::with_engine(MockEngine::ready());
            state.gpu_pool = pool.clone();
            install_worker_registry(&mut state);
            state.gpu_pool = Arc::new(crate::gpu_pool::GpuPool {
                workers: Vec::new().into(),
            });
            state.scheduled_work = crate::scheduler::ScheduledWorkHandle::for_runtime(
                tokio::sync::mpsc::channel(1).0,
                mode,
                false,
                mode == crate::dispatch_mode::DispatchMode::Observe,
            );
            let registry = state.device_registry.clone();
            let id = "cuda:00000000000000000000000000000000";
            registry.set_desired_enabled(id, false).unwrap();
            let mut events = state.events.subscribe();
            let app = app_with_state(state);

            let response = app
                .clone()
                .oneshot(
                    Request::patch(format!("/api/devices/{id}"))
                        .header("content-type", "application/json")
                        .body(Body::from(r#"{"enabled":true}"#))
                        .unwrap(),
                )
                .await
                .unwrap();

            assert_eq!(response.status(), StatusCode::ACCEPTED, "{label}");
            let body = json_body(response).await;
            assert_eq!(body["desired_enabled"], true, "{label}");
            assert_eq!(body["restart_required"], true, "{label}");
            assert!(registry.desired_enabled(id), "{label}");
            assert_eq!(pool.workers.len(), 1, "{label}");
            assert!(!worker.shutdown_requested.load(Ordering::SeqCst), "{label}");
            assert!(matches!(
                events.try_recv(),
                Err(tokio::sync::broadcast::error::TryRecvError::Empty)
                    | Err(tokio::sync::broadcast::error::TryRecvError::Closed)
            ));

            let listed = app
                .clone()
                .oneshot(Request::get("/api/devices").body(Body::empty()).unwrap())
                .await
                .unwrap();
            assert_eq!(
                json_body(listed).await["devices"][0]["restart_required"],
                true,
                "{label}"
            );

            let idempotent = app
                .oneshot(
                    Request::patch(format!("/api/devices/{id}"))
                        .header("content-type", "application/json")
                        .body(Body::from(r#"{"enabled":true}"#))
                        .unwrap(),
                )
                .await
                .unwrap();
            assert_eq!(idempotent.status(), StatusCode::OK, "{label}");
            assert_eq!(json_body(idempotent).await["restart_required"], true);
        }
    }

    #[tokio::test]
    async fn nonauthoritative_live_degraded_owner_does_not_claim_restart_recovery() {
        for (label, mode) in [
            ("legacy", crate::dispatch_mode::DispatchMode::Legacy),
            ("observe", crate::dispatch_mode::DispatchMode::Observe),
        ] {
            let worker = gpu_worker_stub(0);
            worker.consecutive_failures.store(3, Ordering::SeqCst);
            *worker.degraded_until.write().unwrap() =
                Some(std::time::Instant::now() + std::time::Duration::from_secs(60));
            let pool = Arc::new(crate::gpu_pool::GpuPool {
                workers: vec![worker].into(),
            });
            let mut state = AppState::with_engine(MockEngine::ready());
            state.gpu_pool = pool;
            install_worker_registry(&mut state);
            state.scheduled_work = crate::scheduler::ScheduledWorkHandle::for_runtime(
                tokio::sync::mpsc::channel(1).0,
                mode,
                false,
                mode == crate::dispatch_mode::DispatchMode::Observe,
            );

            let response = app_with_state(state)
                .oneshot(Request::get("/api/devices").body(Body::empty()).unwrap())
                .await
                .unwrap();

            assert_eq!(response.status(), StatusCode::OK, "{label}");
            let body = json_body(response).await;
            assert_eq!(body["devices"][0]["desired_enabled"], true, "{label}");
            assert_eq!(body["devices"][0]["health"], "degraded", "{label}");
            assert_eq!(body["devices"][0]["schedulable"], false, "{label}");
            assert_eq!(
                body["devices"][0]["restart_required"], false,
                "{label}: a live cooldown owner recovers without a process restart"
            );
        }
    }

    #[tokio::test]
    async fn persisted_all_disabled_boot_keeps_v2_lifecycle_and_starts_only_enabled_target() {
        const GPU_0: &str = "cuda:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa";
        const GPU_1: &str = "cuda:bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb";
        let runtime_gpu =
            |ordinal: usize, id: &str, name: &str| mold_inference::device::DiscoveredGpu {
                ordinal,
                stable_id: Some(id.into()),
                raw_cuda_uuid: Some([ordinal as u8; 16]),
                device_kind: Some(mold_inference::device::CudaDeviceKind::FullGpu),
                identity_error: None,
                backend: mold_core::GpuBackend::Cuda,
                name: name.into(),
                compute_capability: Some((8, 6)),
                pci_bus_id: None,
                total_vram_bytes: 24_000_000_000,
                free_vram_bytes: 24_000_000_000,
            };
        let selected = vec![
            runtime_gpu(0, GPU_0, "disabled startup GPU 0"),
            runtime_gpu(1, GPU_1, "disabled startup GPU 1"),
        ];
        let discovered = selected
            .iter()
            .map(|gpu| crate::device_registry::DiscoveredDevice {
                stable_id: gpu.stable_id.clone(),
                backend: gpu.backend,
                visible_ordinal: Some(gpu.ordinal),
                device_kind: mold_core::DeviceKind::FullGpu,
                nvml_uuid: None,
                physical_uuid: None,
                mig_uuid: None,
                mig_parent_uuid: None,
                mig_profile: None,
                pci_bus_id: None,
                name: gpu.name.clone(),
                compute_capability: gpu.compute_capability,
                total_memory_bytes: Some(gpu.total_vram_bytes),
                startup_allowed: true,
                telemetry_ordinal: None,
            })
            .collect();
        let db = mold_db::MetadataDb::open_in_memory().unwrap();
        let preferences = mold_db::DevicePreferences::new(&db);
        preferences.set(GPU_0, false).unwrap();
        preferences.set(GPU_1, false).unwrap();
        let registry = Arc::new(crate::device_registry::DeviceRegistry::new(
            Arc::new(crate::device_registry::StaticDeviceDiscovery::new(
                discovered,
            )),
            Arc::new(Some(db)),
        ));

        let startup_selection = crate::startup_device_selection(&selected, &registry);
        assert!(
            startup_selection.enabled.is_empty(),
            "persisted-disabled devices must not construct startup GPU owners"
        );
        assert_eq!(
            startup_selection
                .persisted_disabled
                .iter()
                .map(|gpu| gpu.name.as_str())
                .collect::<Vec<_>>(),
            vec!["disabled startup GPU 0", "disabled startup GPU 1"]
        );
        let plan = crate::startup_plan(
            &mold_core::GpuSelection::All,
            selected.len(),
            selected.len(),
            true,
            crate::dispatch_mode::DispatchMode::V2,
        );
        assert!(plan.start_gpu_workers);
        assert!(plan.start_v2_coordinator);
        assert!(!plan.start_legacy_dispatcher);

        let (scheduler_tx, mut scheduler_rx) = tokio::sync::mpsc::unbounded_channel();
        let pool = Arc::new(crate::gpu_pool::GpuPool {
            workers: Vec::new().into(),
        });
        pool.workers
            .install_factory(
                crate::gpu_pool::WorkerFactory {
                    devices: startup_selection.v2_factory_devices,
                    shared_pool: Arc::new(Mutex::new(
                        mold_inference::shared_pool::SharedPool::new(),
                    )),
                    fatal_cuda_error: Arc::new(AtomicBool::new(false)),
                    fatal_cuda_shutdown: Arc::new(tokio::sync::Notify::new()),
                    scheduler_tx,
                    owner_spawner: Arc::new(crate::gpu_pool::RuntimeOwnerThreadSpawner),
                    max_cached: 1,
                    cache_idle_ttl: Duration::from_secs(60),
                },
                Vec::new(),
            )
            .unwrap();
        assert!(pool.workers.is_empty());
        assert!(matches!(
            scheduler_rx.try_recv(),
            Err(tokio::sync::mpsc::error::TryRecvError::Empty)
        ));

        let (scheduled_tx, _scheduled_rx) = tokio::sync::mpsc::channel(1);
        let mut state = AppState::with_engine(MockEngine::ready());
        state.gpu_pool = pool.clone();
        state.device_registry = registry;
        state.scheduled_work = crate::scheduler::ScheduledWorkHandle::for_mode(
            scheduled_tx,
            crate::dispatch_mode::DispatchMode::V2,
        );
        let app = app_with_state(state);

        let devices = app
            .clone()
            .oneshot(Request::get("/api/devices").body(Body::empty()).unwrap())
            .await
            .unwrap();
        assert_eq!(devices.status(), StatusCode::OK);
        let body = json_body(devices).await;
        assert_eq!(body["devices"].as_array().unwrap().len(), 2);
        for device in body["devices"].as_array().unwrap() {
            assert_eq!(device["desired_enabled"], false);
            assert_eq!(device["admin_state"], "disabled");
            assert_eq!(device["schedulable"], false);
        }

        let enabled = app
            .oneshot(
                Request::patch(format!("/api/devices/{GPU_1}"))
                    .header("content-type", "application/json")
                    .body(Body::from(r#"{"enabled":true}"#))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(enabled.status(), StatusCode::ACCEPTED);
        let enabled = json_body(enabled).await;
        assert_eq!(enabled["id"], GPU_1);
        assert_eq!(enabled["desired_enabled"], true);
        assert_eq!(enabled["admin_state"], "starting");

        let ready = tokio::time::timeout(Duration::from_secs(1), scheduler_rx.recv())
            .await
            .expect("re-enabled target must publish Ready without hanging")
            .expect("re-enabled target owner channel must remain open");
        let (ready_id, ready_epoch) = match ready {
            crate::scheduler::WorkerEvent::Ready {
                device_id,
                owner_epoch,
                ..
            } => (device_id, owner_epoch),
            _ => panic!("re-enabled target must publish epoch-qualified Ready"),
        };
        assert_eq!(ready_id, GPU_1);
        assert!(ready_epoch > 0);
        let workers = pool.worker_snapshot();
        assert_eq!(workers.len(), 1);
        assert_eq!(crate::scheduler::worker_device_id(&workers[0]), GPU_1);
        workers[0].request_shutdown();
        assert!(pool.workers.wait_and_reap(GPU_1, ready_epoch));
        assert!(pool.workers.is_empty());
    }

    #[tokio::test]
    async fn device_patch_enable_returns_starting_without_waiting_for_owner_ready() {
        let id = "cuda:dddddddddddddddddddddddddddddddd";
        let gpu = mold_inference::device::DiscoveredGpu {
            ordinal: 0,
            stable_id: Some(id.into()),
            raw_cuda_uuid: Some([0xdd; 16]),
            device_kind: Some(mold_inference::device::CudaDeviceKind::FullGpu),
            identity_error: None,
            backend: mold_core::GpuBackend::Cuda,
            name: "replacement gpu".into(),
            compute_capability: Some((8, 6)),
            pci_bus_id: None,
            total_vram_bytes: 24_000_000_000,
            free_vram_bytes: 24_000_000_000,
        };
        let (scheduler_tx, mut scheduler_rx) = tokio::sync::mpsc::unbounded_channel();
        let pool = Arc::new(crate::gpu_pool::GpuPool {
            workers: Vec::new().into(),
        });
        pool.workers
            .install_factory(
                crate::gpu_pool::WorkerFactory {
                    devices: [(id.to_string(), gpu)].into_iter().collect(),
                    shared_pool: Arc::new(Mutex::new(
                        mold_inference::shared_pool::SharedPool::new(),
                    )),
                    fatal_cuda_error: Arc::new(AtomicBool::new(false)),
                    fatal_cuda_shutdown: Arc::new(tokio::sync::Notify::new()),
                    scheduler_tx,
                    owner_spawner: Arc::new(crate::gpu_pool::RuntimeOwnerThreadSpawner),
                    max_cached: 1,
                    cache_idle_ttl: Duration::from_secs(60),
                },
                Vec::new(),
            )
            .unwrap();
        let mut state = AppState::with_engine(MockEngine::ready());
        state.gpu_pool = pool.clone();
        state.device_registry = Arc::new(crate::device_registry::DeviceRegistry::new(
            Arc::new(crate::device_registry::StaticDeviceDiscovery::new(vec![
                crate::device_registry::DiscoveredDevice {
                    stable_id: Some(id.into()),
                    backend: mold_core::GpuBackend::Cuda,
                    visible_ordinal: Some(0),
                    device_kind: mold_core::DeviceKind::FullGpu,
                    nvml_uuid: None,
                    physical_uuid: None,
                    mig_uuid: None,
                    mig_parent_uuid: None,
                    mig_profile: None,
                    pci_bus_id: None,
                    name: "replacement gpu".into(),
                    compute_capability: Some((8, 6)),
                    total_memory_bytes: Some(24_000_000_000),
                    startup_allowed: true,
                    telemetry_ordinal: None,
                },
            ])),
            Arc::new(None),
        ));
        install_authoritative_v2(&mut state);

        let response = app_with_state(state)
            .oneshot(
                Request::patch(format!("/api/devices/{id}"))
                    .header("content-type", "application/json")
                    .body(Body::from(r#"{"enabled":true}"#))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::ACCEPTED);
        let body = json_body(response).await;
        assert_eq!(body["admin_state"], "starting");
        assert_eq!(body["schedulable"], false);

        let ready_epoch = match scheduler_rx.recv().await.unwrap() {
            crate::scheduler::WorkerEvent::Ready { owner_epoch, .. } => owner_epoch,
            _ => panic!("fresh owner must publish epoch-qualified Ready"),
        };
        let worker = pool.worker_snapshot().pop().unwrap();
        worker.request_shutdown();
        assert!(pool.workers.wait_and_reap(id, ready_epoch));
    }

    #[tokio::test]
    async fn device_patch_rejects_unknown_and_startup_excluded_ids() {
        let mut state = AppState::with_engine(MockEngine::ready());
        state.device_registry = Arc::new(crate::device_registry::DeviceRegistry::new(
            Arc::new(crate::device_registry::StaticDeviceDiscovery::new(vec![
                crate::device_registry::DiscoveredDevice {
                    stable_id: Some("cuda:eeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee".into()),
                    backend: mold_core::GpuBackend::Cuda,
                    visible_ordinal: Some(3),
                    device_kind: mold_core::DeviceKind::FullGpu,
                    nvml_uuid: None,
                    physical_uuid: None,
                    mig_uuid: None,
                    mig_parent_uuid: None,
                    mig_profile: None,
                    pci_bus_id: None,
                    name: "excluded".into(),
                    compute_capability: Some((8, 6)),
                    total_memory_bytes: Some(24_000_000_000),
                    startup_allowed: false,
                    telemetry_ordinal: None,
                },
            ])),
            Arc::new(None),
        ));
        install_authoritative_v2(&mut state);
        let app = app_with_state(state);

        let unknown = app
            .clone()
            .oneshot(
                Request::patch("/api/devices/cuda:ffffffffffffffffffffffffffffffff")
                    .header("content-type", "application/json")
                    .body(Body::from(r#"{"enabled":false}"#))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(unknown.status(), StatusCode::NOT_FOUND);

        let excluded = app
            .oneshot(
                Request::patch("/api/devices/cuda:eeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee")
                    .header("content-type", "application/json")
                    .body(Body::from(r#"{"enabled":true}"#))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(excluded.status(), StatusCode::CONFLICT);
    }

    #[tokio::test]
    async fn device_api_uses_cached_telemetry_and_legacy_status_keeps_shape() {
        let worker = gpu_worker_stub(0);
        let cache = worker.model_cache.clone();
        let _ = std::thread::spawn(move || {
            let _guard = cache.lock().unwrap();
            panic!("poison cache to prove status routes never acquire it");
        })
        .join();
        let mut state = AppState::with_engine(MockEngine::ready());
        state.gpu_pool = Arc::new(crate::gpu_pool::GpuPool {
            workers: vec![worker].into(),
        });
        state.device_registry = Arc::new(crate::device_registry::DeviceRegistry::new(
            Arc::new(crate::device_registry::StaticDeviceDiscovery::new(vec![
                crate::device_registry::DiscoveredDevice {
                    stable_id: Some("cuda:fedcba9876543210fedcba9876543210".into()),
                    backend: mold_core::GpuBackend::Cuda,
                    visible_ordinal: Some(0),
                    device_kind: mold_core::DeviceKind::FullGpu,
                    nvml_uuid: None,
                    physical_uuid: None,
                    mig_uuid: None,
                    mig_parent_uuid: None,
                    mig_profile: None,
                    pci_bus_id: None,
                    name: "test-gpu-0".into(),
                    compute_capability: Some((8, 6)),
                    total_memory_bytes: Some(24_000_000_000),
                    startup_allowed: true,
                    telemetry_ordinal: Some(0),
                },
            ])),
            Arc::new(None),
        ));
        state.resources.publish(mold_core::ResourceSnapshot {
            hostname: "gpu-host".into(),
            timestamp: 1,
            gpus: vec![mold_core::GpuSnapshot {
                ordinal: 0,
                name: "test-gpu-0".into(),
                backend: mold_core::GpuBackend::Cuda,
                vram_total: 24 * 1024 * 1024 * 1024,
                vram_used: 9 * 1024 * 1024 * 1024,
                vram_used_by_mold: Some(8 * 1024 * 1024 * 1024),
                vram_used_by_other: Some(1024 * 1024 * 1024),
                gpu_utilization: Some(41),
            }],
            system_ram: mold_core::RamSnapshot {
                total: 64_000_000_000,
                used: 20_000_000_000,
                available: None,
                used_by_mold: 2_000_000_000,
                used_by_other: 18_000_000_000,
            },
            cpu: None,
        });
        let app = app_with_state(state);

        let devices = json_body(
            app.clone()
                .oneshot(Request::get("/api/devices").body(Body::empty()).unwrap())
                .await
                .unwrap(),
        )
        .await;
        assert_eq!(
            devices["devices"][0]["memory"]["used_bytes"],
            9 * 1024 * 1024 * 1024_u64
        );
        assert_eq!(
            devices["devices"][0]["memory"]["mold_used_bytes"],
            8 * 1024 * 1024 * 1024_u64
        );
        assert_eq!(
            devices["devices"][0]["telemetry"]["utilization_percent"],
            41
        );

        let status = json_body(
            app.oneshot(Request::get("/api/status").body(Body::empty()).unwrap())
                .await
                .unwrap(),
        )
        .await;
        assert_eq!(status["gpu_info"]["name"], "test-gpu-0");
        assert_eq!(status["gpu_info"]["vram_total_mb"], 24_576);
        assert_eq!(status["gpu_info"]["vram_used_mb"], 9_216);
        assert_eq!(status["memory_status"], "VRAM: 16.1 GB free");
        assert_eq!(status["gpus"][0]["ordinal"], 0);
        assert_eq!(
            status["gpus"][0]["vram_used_bytes"],
            9 * 1024 * 1024 * 1024_u64
        );
        assert!(
            status["gpu_info"].get("id").is_none(),
            "legacy gpu_info shape must not gain device fields"
        );
    }

    #[test]
    fn status_handler_has_no_live_inference_device_queries() {
        let source = include_str!("routes.rs");
        let start = source.find("async fn server_status").unwrap();
        let end = source[start..].find("// ── /health").unwrap() + start;
        let handler = &source[start..end];

        assert!(!handler.contains("mold_inference::device"));
        assert!(!handler.contains("memory_status_string"));
        assert!(!handler.contains("free_vram_bytes"));
        assert!(!handler.contains("CudaContext"));
    }

    #[tokio::test]
    async fn status_omits_excluded_and_workerless_devices_but_devices_keeps_them() {
        let worker = gpu_worker_stub(1);
        let mut state = AppState::with_engine(MockEngine::ready());
        state.gpu_pool = Arc::new(crate::gpu_pool::GpuPool {
            workers: vec![worker].into(),
        });
        let discovered = |ordinal, id: &str, name: &str, startup_allowed| {
            crate::device_registry::DiscoveredDevice {
                stable_id: Some(id.into()),
                backend: mold_core::GpuBackend::Cuda,
                visible_ordinal: Some(ordinal),
                device_kind: mold_core::DeviceKind::FullGpu,
                nvml_uuid: None,
                physical_uuid: None,
                mig_uuid: None,
                mig_parent_uuid: None,
                mig_profile: None,
                pci_bus_id: None,
                name: name.into(),
                compute_capability: Some((8, 6)),
                total_memory_bytes: Some(24_000_000_000),
                startup_allowed,
                telemetry_ordinal: Some(ordinal),
            }
        };
        state.device_registry = Arc::new(crate::device_registry::DeviceRegistry::new(
            Arc::new(crate::device_registry::StaticDeviceDiscovery::new(vec![
                discovered(
                    0,
                    "cuda:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
                    "excluded GPU",
                    false,
                ),
                discovered(
                    1,
                    "cuda:bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb",
                    "active GPU",
                    true,
                ),
                discovered(
                    2,
                    "cuda:cccccccccccccccccccccccccccccccc",
                    "workerless GPU",
                    true,
                ),
            ])),
            Arc::new(None),
        ));
        let app = app_with_state(state);

        let devices = json_body(
            app.clone()
                .oneshot(Request::get("/api/devices").body(Body::empty()).unwrap())
                .await
                .unwrap(),
        )
        .await;
        assert_eq!(devices["devices"].as_array().unwrap().len(), 3);

        let status = json_body(
            app.oneshot(Request::get("/api/status").body(Body::empty()).unwrap())
                .await
                .unwrap(),
        )
        .await;
        assert_eq!(status["gpu_info"]["name"], "active GPU");
        assert_eq!(status["gpus"].as_array().unwrap().len(), 1);
        assert_eq!(status["gpus"][0]["ordinal"], 1);
    }

    #[tokio::test]
    async fn maintenance_mode_rejects_generation_before_queueing() {
        let state = AppState::with_engine(MockEngine::ready());
        state.set_generation_unavailable(
            "generation is unavailable while GPU selection is 'none' (maintenance mode)",
        );
        let app = app_with_state(state);

        let response = app
            .oneshot(
                Request::post("/api/generate")
                    .header("content-type", "application/json")
                    .body(Body::from(generate_body("must not run", 512, 512)))
                    .unwrap(),
            )
            .await
            .unwrap();

        assert_eq!(response.status(), StatusCode::SERVICE_UNAVAILABLE);
        let body = json_body(response).await;
        assert_eq!(body["code"], "GENERATION_UNAVAILABLE");
        assert!(body["error"].as_str().unwrap().contains("maintenance mode"));
    }

    #[tokio::test]
    async fn maintenance_mode_rejects_admin_model_load_before_resolution() {
        let state = AppState::with_engine(MockEngine::ready());
        state.set_generation_unavailable(
            "generation is unavailable while GPU selection is 'none' (maintenance mode)",
        );
        let app = app_with_state(state);

        let response = app
            .oneshot(
                Request::post("/api/models/load")
                    .header("content-type", "application/json")
                    .body(Body::from(r#"{"model":"does-not-exist"}"#))
                    .unwrap(),
            )
            .await
            .unwrap();

        assert_eq!(response.status(), StatusCode::SERVICE_UNAVAILABLE);
        let body = json_body(response).await;
        assert_eq!(body["code"], "GENERATION_UNAVAILABLE");
    }

    #[tokio::test]
    async fn maintenance_mode_rejects_chain_and_upscale_routes_before_gpu_work() {
        let state = AppState::with_engine(MockEngine::ready());
        state.set_generation_unavailable(
            "generation is unavailable while GPU selection is 'none' (maintenance mode)",
        );
        let app = app_with_state(state);
        let chain_body = serde_json::to_vec(&route_chain_request()).unwrap();
        // Deliberately invalid image bytes prove the availability fence runs
        // before upscale validation, model resolution/pull, or engine creation.
        let upscale_body = serde_json::to_vec(&serde_json::json!({
            "model": "does-not-exist",
            "image": "AQID",
            "output_format": "png"
        }))
        .unwrap();

        for (path, body) in [
            ("/api/generate/chain", chain_body.clone()),
            ("/api/generate/chain/stream", chain_body.clone()),
            ("/api/upscale", upscale_body.clone()),
            ("/api/upscale/stream", upscale_body.clone()),
        ] {
            let response = app
                .clone()
                .oneshot(
                    Request::post(path)
                        .header("content-type", "application/json")
                        .body(Body::from(body))
                        .unwrap(),
                )
                .await
                .unwrap();

            assert_eq!(
                response.status(),
                StatusCode::SERVICE_UNAVAILABLE,
                "{path} must be fenced in maintenance mode"
            );
            let body = json_body(response).await;
            assert_eq!(
                body["code"], "GENERATION_UNAVAILABLE",
                "{path} must return the typed maintenance error"
            );
            assert!(
                body["error"].as_str().unwrap().contains("maintenance mode"),
                "{path} must preserve the startup reason"
            );
        }
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
    async fn patch_queue_position_reorders_the_queued_job() {
        let (state, _rx) = AppState::with_engine_and_queue(MockEngine::ready());
        state.job_registry.register("aaaa", "flux-dev:fp16");
        state.job_registry.register("bbbb", "sdxl:q8");
        state.job_registry.register("cccc", "ltx-video:q8");

        let app = app_with_state(state);
        let resp = app
            .clone()
            .oneshot(
                Request::patch("/api/queue/cccc")
                    .header("content-type", "application/json")
                    .body(Body::from(r#"{"position":0}"#))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
        let body = json_body(resp).await;
        assert_eq!(body["id"], "cccc");
        assert_eq!(body["position"], 0);

        // The whole queue reflects the new order.
        let resp = app
            .oneshot(Request::get("/api/queue").body(Body::empty()).unwrap())
            .await
            .unwrap();
        let body = json_body(resp).await;
        let entries = body["entries"].as_array().expect("entries array");
        let order: Vec<&str> = entries.iter().map(|e| e["id"].as_str().unwrap()).collect();
        assert_eq!(order, ["cccc", "aaaa", "bbbb"]);
    }

    #[tokio::test]
    async fn patch_queue_position_only_leaves_target_gpu_untouched() {
        // A reorder that omits `target_gpu` must not reset the pinned lane to
        // Auto — the additive `position` edit is independent.
        let (state, _rx) = AppState::with_engine_and_queue(MockEngine::ready());
        state
            .job_registry
            .register_with_target_gpu("aaaa", "flux-dev:fp16", Some(2));
        state.job_registry.register("bbbb", "sdxl:q8");

        let app = app_with_state(state);
        let resp = app
            .clone()
            .oneshot(
                Request::patch("/api/queue/aaaa")
                    .header("content-type", "application/json")
                    .body(Body::from(r#"{"position":1}"#))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
        let body = json_body(resp).await;
        assert_eq!(body["id"], "aaaa");
        assert_eq!(body["position"], 1, "aaaa moved behind bbbb");
        assert_eq!(body["target_gpu"], 2, "pinned lane survived the reorder");
    }

    #[tokio::test]
    async fn patch_queue_applies_target_gpu_and_position_together() {
        let (state, _rx) = AppState::with_engine_and_queue(MockEngine::ready());
        state
            .job_registry
            .register_with_target_gpu("aaaa", "flux-dev:fp16", Some(0));
        state.job_registry.register("bbbb", "sdxl:q8");

        let app = app_with_state(state);
        // `target_gpu:null` clears the lane (no worker pool needed) and
        // `position:1` sends the job to the back — both in one PATCH.
        let resp = app
            .oneshot(
                Request::patch("/api/queue/aaaa")
                    .header("content-type", "application/json")
                    .body(Body::from(r#"{"target_gpu":null,"position":1}"#))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
        let body = json_body(resp).await;
        assert_eq!(body["id"], "aaaa");
        assert_eq!(body["position"], 1);
        assert!(
            body.get("target_gpu").is_none(),
            "lane cleared to Auto: {body}"
        );
    }

    #[tokio::test]
    async fn patch_queue_position_rejects_running_jobs() {
        let (state, _rx) = AppState::with_engine_and_queue(MockEngine::ready());
        state.job_registry.register("aaaa", "flux-dev:fp16");
        state.job_registry.mark_running("aaaa", Some(0));

        let app = app_with_state(state);
        let resp = app
            .oneshot(
                Request::patch("/api/queue/aaaa")
                    .header("content-type", "application/json")
                    .body(Body::from(r#"{"position":0}"#))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::CONFLICT);
        let body = json_body(resp).await;
        assert_eq!(body["code"], "QUEUE_JOB_RUNNING");
    }

    #[tokio::test]
    async fn patch_queue_position_unknown_id_returns_404() {
        let app = app_empty();
        let resp = app
            .oneshot(
                Request::patch("/api/queue/not-here")
                    .header("content-type", "application/json")
                    .body(Body::from(r#"{"position":0}"#))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::NOT_FOUND);
    }

    // ── DELETE /api/models/:model ────────────────────────────────────────────

    /// All clean storage paths for a manifest model under `models_dir`.
    fn manifest_clean_paths(models_dir: &std::path::Path, model: &str) -> Vec<std::path::PathBuf> {
        let manifest = mold_core::manifest::find_manifest(model).unwrap();
        manifest
            .files
            .iter()
            .map(|file| models_dir.join(mold_core::manifest::storage_path(manifest, file)))
            .collect()
    }

    #[tokio::test]
    #[allow(clippy::await_holding_lock)]
    async fn delete_model_removes_exclusively_owned_files() {
        let _lock = env_lock().lock().unwrap_or_else(|e| e.into_inner());
        let models_dir = test_models_dir("delete-model-solo");
        populate_manifest_files(&models_dir, "flux-schnell:q8");
        std::env::set_var("MOLD_MODELS_DIR", &models_dir);

        let app = app_empty();
        let resp = app
            .oneshot(
                Request::delete("/api/models/flux-schnell:q8")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
        let body = json_body(resp).await;

        // No other model references anything — every file is exclusively
        // owned, nothing is kept, and real bytes were freed.
        let removed = body["removed"].as_array().expect("removed array");
        assert!(!removed.is_empty(), "expected removed files: {body}");
        assert_eq!(body["kept"], serde_json::json!([]));
        assert!(
            body["freed_bytes"].as_u64().unwrap() > 0,
            "freed_bytes must be > 0: {body}"
        );

        for path in manifest_clean_paths(&models_dir, "flux-schnell:q8") {
            assert!(
                !path.exists(),
                "exclusively-owned file must be deleted: {}",
                path.display()
            );
        }

        std::env::remove_var("MOLD_MODELS_DIR");
        let _ = std::fs::remove_dir_all(models_dir);
    }

    #[tokio::test]
    #[allow(clippy::await_holding_lock)]
    async fn delete_model_keeps_components_shared_with_another_model() {
        let _lock = env_lock().lock().unwrap_or_else(|e| e.into_inner());
        let models_dir = test_models_dir("delete-model-shared");
        // flux-schnell:q8 and flux-dev:q8 share VAE/T5/CLIP under shared/.
        populate_manifest_files(&models_dir, "flux-schnell:q8");
        populate_manifest_files(&models_dir, "flux-dev:q8");
        std::env::set_var("MOLD_MODELS_DIR", &models_dir);

        let schnell_paths = manifest_clean_paths(&models_dir, "flux-schnell:q8");
        let dev_paths: std::collections::HashSet<_> =
            manifest_clean_paths(&models_dir, "flux-dev:q8")
                .into_iter()
                .collect();
        let shared: Vec<_> = schnell_paths
            .iter()
            .filter(|p| dev_paths.contains(*p))
            .collect();
        let unique: Vec<_> = schnell_paths
            .iter()
            .filter(|p| !dev_paths.contains(*p))
            .collect();
        assert!(
            !shared.is_empty() && !unique.is_empty(),
            "test premise: the two models must share some files and own others"
        );

        let app = app_empty();
        let resp = app
            .oneshot(
                Request::delete("/api/models/flux-schnell:q8")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
        let body = json_body(resp).await;

        // Exclusive files (the transformer) are gone; shared components stay.
        for path in &unique {
            assert!(
                !path.exists(),
                "exclusive file must be deleted: {}",
                path.display()
            );
        }
        for path in &shared {
            assert!(
                path.exists(),
                "shared component must survive: {}",
                path.display()
            );
        }

        // The kept list reports every shared component with the surviving
        // referencing model.
        let kept = body["kept"].as_array().expect("kept array");
        assert_eq!(kept.len(), shared.len(), "kept must cover shared: {body}");
        for entry in kept {
            let used_by: Vec<&str> = entry["used_by"]
                .as_array()
                .unwrap()
                .iter()
                .map(|v| v.as_str().unwrap())
                .collect();
            assert!(
                used_by.contains(&"flux-dev:q8"),
                "kept entry must name the surviving model: {entry}"
            );
            let component = entry["component"].as_str().unwrap();
            assert!(
                shared.iter().any(|p| p.to_string_lossy() == component),
                "kept component must be a shared path: {entry}"
            );
        }

        // The removed list must not contain any shared path.
        let removed: Vec<&str> = body["removed"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_str().unwrap())
            .collect();
        for path in &shared {
            assert!(
                !removed.contains(&path.to_string_lossy().as_ref()),
                "shared path must not be reported as removed: {}",
                path.display()
            );
        }

        // The sibling model is untouched.
        for path in &dev_paths {
            assert!(
                path.exists(),
                "sibling model file must survive: {}",
                path.display()
            );
        }

        std::env::remove_var("MOLD_MODELS_DIR");
        let _ = std::fs::remove_dir_all(models_dir);
    }

    #[tokio::test]
    async fn delete_model_unknown_returns_404_unknown_model() {
        let app = app_empty();
        let resp = app
            .oneshot(
                Request::delete("/api/models/definitely-not-a-model")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::NOT_FOUND);
        let body = json_body(resp).await;
        assert_eq!(body["code"], "UNKNOWN_MODEL");
    }

    #[tokio::test]
    async fn delete_model_gpu_resident_returns_409_model_loaded() {
        // MockEngine::ready() is GPU-resident in the cache as "mock-model" —
        // deletion must be refused until the model is unloaded.
        let app = app_with_state(AppState::with_engine(MockEngine::ready()));
        let resp = app
            .oneshot(
                Request::delete("/api/models/mock-model")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::CONFLICT);
        let body = json_body(resp).await;
        assert_eq!(body["code"], "MODEL_LOADED");
    }

    #[tokio::test]
    async fn delete_queue_cancels_queued_job_with_204_and_removes_it() {
        let (state, _rx) = AppState::with_engine_and_queue(MockEngine::ready());
        state.job_registry.register("aaaa", "flux-dev:fp16");

        let app = app_with_state(state);
        let resp = app
            .clone()
            .oneshot(
                Request::delete("/api/queue/aaaa")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::NO_CONTENT);

        let resp = app
            .oneshot(Request::get("/api/queue").body(Body::empty()).unwrap())
            .await
            .unwrap();
        let body = json_body(resp).await;
        assert_eq!(body["entries"], serde_json::json!([]));
    }

    #[tokio::test]
    async fn queue_mutation_response_waits_for_scheduler_transaction_fence() {
        let (state, _rx) = AppState::with_engine_and_queue(MockEngine::ready());
        state.job_registry.register("aaaa", "flux-dev:fp16");
        let fence = state.scheduler_mutation_fence.clone().lock_owned().await;
        let app = app_with_state(state);

        let mutation = tokio::spawn(async move {
            app.oneshot(
                Request::delete("/api/queue/aaaa")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap()
        });
        tokio::task::yield_now().await;
        assert!(
            !mutation.is_finished(),
            "mutation route must not acknowledge before the scheduler fence"
        );

        drop(fence);
        let response = tokio::time::timeout(Duration::from_secs(1), mutation)
            .await
            .expect("mutation must resume after the scheduler fence")
            .unwrap();
        assert_eq!(response.status(), StatusCode::NO_CONTENT);
    }

    #[tokio::test]
    async fn delete_queue_unknown_id_returns_404() {
        let (state, _rx) = AppState::with_engine_and_queue(MockEngine::ready());
        let app = app_with_state(state);
        let resp = app
            .oneshot(
                Request::delete("/api/queue/not-here")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::NOT_FOUND);
        let body = json_body(resp).await;
        assert_eq!(body["code"], "QUEUE_JOB_NOT_FOUND");
    }

    #[tokio::test]
    async fn delete_queue_running_job_returns_409() {
        let (state, _rx) = AppState::with_engine_and_queue(MockEngine::ready());
        state.job_registry.register("aaaa", "flux-dev:fp16");
        state.job_registry.mark_running("aaaa", Some(0));

        let app = app_with_state(state.clone());
        let resp = app
            .oneshot(
                Request::delete("/api/queue/aaaa")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::CONFLICT);
        let body = json_body(resp).await;
        assert_eq!(body["code"], "QUEUE_JOB_RUNNING");
        assert_eq!(
            state.job_registry.len(),
            1,
            "running job must survive the cancel attempt"
        );
    }

    #[tokio::test]
    async fn pause_and_resume_toggle_queue_paused_in_status() {
        let (state, _rx) = AppState::with_engine_and_queue(MockEngine::ready());
        let app = app_with_state(state);

        // Idle server reports not-paused.
        let resp = app
            .clone()
            .oneshot(Request::get("/api/status").body(Body::empty()).unwrap())
            .await
            .unwrap();
        assert_eq!(json_body(resp).await["queue_paused"], false);

        // Pause → 200 {"paused": true} → status reflects it.
        let resp = app
            .clone()
            .oneshot(
                Request::post("/api/queue/pause")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
        assert_eq!(json_body(resp).await["paused"], true);

        let resp = app
            .clone()
            .oneshot(Request::get("/api/status").body(Body::empty()).unwrap())
            .await
            .unwrap();
        assert_eq!(json_body(resp).await["queue_paused"], true);

        // Resume → 200 {"paused": false} → status cleared.
        let resp = app
            .clone()
            .oneshot(
                Request::post("/api/queue/resume")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
        assert_eq!(json_body(resp).await["paused"], false);

        let resp = app
            .oneshot(Request::get("/api/status").body(Body::empty()).unwrap())
            .await
            .unwrap();
        assert_eq!(json_body(resp).await["queue_paused"], false);
    }

    #[tokio::test]
    async fn pause_publishes_queue_paused_event_once_per_transition() {
        // Keep `state` in scope so its EventBroadcaster (and thus the
        // subscriber) outlives the router across both requests.
        let (state, _rx) = AppState::with_engine_and_queue(MockEngine::ready());
        let mut events = state.events.subscribe();
        let app = app_with_state(state.clone());

        // First pause flips state → one queue_paused event.
        let resp = app
            .clone()
            .oneshot(
                Request::post("/api/queue/pause")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
        match events.try_recv() {
            Ok(mold_core::ServerEvent::QueuePaused) => {}
            other => panic!("expected queue_paused, got {other:?}"),
        }

        // Idempotent second pause is a no-op → no duplicate event.
        let _ = app
            .clone()
            .oneshot(
                Request::post("/api/queue/pause")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert!(
            matches!(
                events.try_recv(),
                Err(tokio::sync::broadcast::error::TryRecvError::Empty)
            ),
            "second pause must not emit a duplicate queue_paused event"
        );
    }

    #[tokio::test]
    async fn delete_queue_cancels_all_queued_and_reports_count() {
        let (state, _rx) = AppState::with_engine_and_queue(MockEngine::ready());
        state.job_registry.register("aaaa", "flux-dev:fp16");
        state.job_registry.register("bbbb", "sdxl:q8");
        state.job_registry.register("cccc", "ltx-video:q8");
        // A running job must survive the bulk cancel.
        state.job_registry.mark_running("cccc", Some(0));

        let app = app_with_state(state);
        let resp = app
            .clone()
            .oneshot(Request::delete("/api/queue").body(Body::empty()).unwrap())
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
        assert_eq!(json_body(resp).await["cancelled"], 2);

        let resp = app
            .oneshot(Request::get("/api/queue").body(Body::empty()).unwrap())
            .await
            .unwrap();
        let body = json_body(resp).await;
        let entries = body["entries"].as_array().expect("entries array");
        assert_eq!(entries.len(), 1);
        assert_eq!(entries[0]["id"], "cccc");
        assert_eq!(entries[0]["state"], "running");
    }

    #[tokio::test]
    async fn discovery_peers_is_registered_and_excludes_the_serving_instance() {
        let mut state = AppState::for_tests();
        state.instance_id = std::sync::Arc::new("self-instance".to_string());
        state.discovery.set_can_browse(true);
        *state.discovery.peers.write().unwrap() = vec![
            mold_core::DiscoveryPeer {
                name: "this-server-7680".to_string(),
                url: "http://192.168.1.10:7680".to_string(),
                host: "192.168.1.10".to_string(),
                port: 7680,
                version: Some("0.20.2".to_string()),
                auth_required: false,
                instance_id: Some("self-instance".to_string()),
                is_this_machine: true,
            },
            mold_core::DiscoveryPeer {
                name: "studio-7680".to_string(),
                url: "http://192.168.1.20:7680".to_string(),
                host: "192.168.1.20".to_string(),
                port: 7680,
                version: Some("0.20.1".to_string()),
                auth_required: true,
                instance_id: Some("studio-instance".to_string()),
                is_this_machine: false,
            },
        ];

        let resp = app_with_state(state)
            .oneshot(
                Request::get("/api/discovery/peers")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
        let body = json_body(resp).await;
        let peers = body.as_array().expect("peer array");
        assert_eq!(peers.len(), 1);
        assert_eq!(peers[0]["instance_id"], "studio-instance");
        assert_eq!(peers[0]["auth_required"], true);
        assert_eq!(peers[0]["is_this_machine"], false);
    }

    #[tokio::test]
    async fn discovery_capability_and_endpoint_are_gated_when_browser_is_unavailable() {
        let app = app_empty();
        let capability = app
            .clone()
            .oneshot(
                Request::get("/api/capabilities")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(
            json_body(capability).await["discovery"]["can_browse"],
            false
        );

        let resp = app
            .oneshot(
                Request::get("/api/discovery/peers")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::SERVICE_UNAVAILABLE);
    }

    #[tokio::test]
    async fn capabilities_reports_queue_controls_available() {
        let (scheduled_tx, _scheduled_rx) = tokio::sync::mpsc::channel(1);
        let mut state = AppState::for_tests();
        state.scheduled_work = crate::scheduler::ScheduledWorkHandle::for_mode(
            scheduled_tx,
            crate::dispatch_mode::DispatchMode::V2,
        );
        let app = app_with_state(state);
        let resp = app
            .oneshot(
                Request::get("/api/capabilities")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
        let body = json_body(resp).await;
        assert_eq!(body["queue"]["can_pause"], true);
        assert_eq!(body["queue"]["can_cancel_all"], true);
        assert_eq!(body["queue"]["can_reorder"], true);
        assert_eq!(body["devices"]["available"], true);
        assert_eq!(body["devices"]["lifecycle"], true);
        assert_eq!(body["devices"]["restart_enable"], false);
    }

    #[tokio::test]
    async fn capabilities_reports_device_lifecycle_false_without_authoritative_v2() {
        let cases = [
            (
                "legacy",
                crate::scheduler::ScheduledWorkHandle::for_runtime(
                    tokio::sync::mpsc::channel(1).0,
                    crate::dispatch_mode::DispatchMode::Legacy,
                    false,
                    false,
                ),
            ),
            (
                "observe",
                crate::scheduler::ScheduledWorkHandle::for_runtime(
                    tokio::sync::mpsc::channel(1).0,
                    crate::dispatch_mode::DispatchMode::Observe,
                    false,
                    true,
                ),
            ),
            (
                "maintenance",
                crate::scheduler::ScheduledWorkHandle::for_runtime(
                    tokio::sync::mpsc::channel(1).0,
                    crate::dispatch_mode::DispatchMode::V2,
                    false,
                    false,
                ),
            ),
            (
                "unavailable",
                crate::scheduler::ScheduledWorkHandle::default(),
            ),
        ];

        for (label, scheduled_work) in cases {
            let mut state = AppState::for_tests();
            state.scheduled_work = scheduled_work;
            let response = app_with_state(state)
                .oneshot(
                    Request::get("/api/capabilities")
                        .body(Body::empty())
                        .unwrap(),
                )
                .await
                .unwrap();
            assert_eq!(response.status(), StatusCode::OK, "{label}");
            let body = json_body(response).await;
            assert_eq!(body["devices"]["lifecycle"], false, "{label}");
            assert_eq!(body["devices"]["restart_enable"], true, "{label}");
        }
    }

    /// Clients feature-detect server-side catalog sorting against this
    /// advertisement — older servers omit the field entirely.
    #[tokio::test]
    async fn capabilities_reports_catalog_sort_vocabulary() {
        let app = app_empty();
        let resp = app
            .oneshot(
                Request::get("/api/capabilities")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
        let body = json_body(resp).await;
        assert_eq!(
            body["catalog"]["sort"],
            serde_json::json!(["downloads", "recent", "rating"])
        );
    }

    /// Poll the registry until the submitted job shows up (the generate
    /// handler registers before submit, so this resolves almost instantly).
    async fn wait_for_registered_job(state: &AppState) -> String {
        for _ in 0..500 {
            if let Some(entry) = state.job_registry.snapshot().entries.first() {
                return entry.id.clone();
            }
            tokio::time::sleep(Duration::from_millis(5)).await;
        }
        panic!("job never appeared in the registry");
    }

    #[tokio::test]
    async fn delete_queue_resolves_blocking_generate_with_cancelled_error() {
        // No queue worker is spawned — the submitted job sits queued in the
        // channel exactly like a job stuck behind a long-running generation.
        let (state, _rx) = AppState::with_engine_and_queue(MockEngine::ready());
        let app = app_with_state(state.clone());

        let gen_app = app.clone();
        let gen_task = tokio::spawn(async move {
            gen_app
                .oneshot(
                    Request::post("/api/generate")
                        .header("content-type", "application/json")
                        .body(Body::from(generate_body("a cat", 512, 512)))
                        .unwrap(),
                )
                .await
                .unwrap()
        });

        let id = wait_for_registered_job(&state).await;
        let resp = app
            .oneshot(
                Request::delete(format!("/api/queue/{id}"))
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::NO_CONTENT);

        let gen_resp = tokio::time::timeout(Duration::from_secs(5), gen_task)
            .await
            .expect("blocking generate must resolve after cancel")
            .unwrap();
        assert_eq!(gen_resp.status().as_u16(), 499);
        let body = json_body(gen_resp).await;
        assert_eq!(body["code"], "CANCELLED");
    }

    #[tokio::test]
    async fn delete_queue_emits_sse_error_and_closes_the_stream() {
        // No queue worker — the streaming job stays queued until cancelled.
        let (state, _rx) = AppState::with_engine_and_queue(MockEngine::ready());
        let app = app_with_state(state.clone());

        let resp = app
            .clone()
            .oneshot(
                Request::post("/api/generate/stream")
                    .header("content-type", "application/json")
                    .body(Body::from(generate_body("a cat", 512, 512)))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::OK);

        let id = wait_for_registered_job(&state).await;
        let del = app
            .oneshot(
                Request::delete(format!("/api/queue/{id}"))
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(del.status(), StatusCode::NO_CONTENT);

        // The stream must emit an `error` event and then CLOSE — to_bytes
        // only returns once the body stream terminates, so the timeout is
        // the regression guard against a stream that stays open after
        // cancellation.
        let bytes = tokio::time::timeout(
            Duration::from_secs(5),
            axum::body::to_bytes(resp.into_body(), 1024 * 1024),
        )
        .await
        .expect("SSE stream must close after cancel")
        .unwrap();
        let text = String::from_utf8(bytes.to_vec()).unwrap();
        assert!(text.contains("event: error"), "missing error event: {text}");
        assert!(text.contains("cancelled"), "missing cancel message: {text}");
    }

    // ── /api/history ─────────────────────────────────────────────────────────

    fn history_entry(prompt: &str, model: &str, ts: i64) -> mold_db::HistoryEntry {
        mold_db::HistoryEntry {
            prompt: prompt.into(),
            negative: None,
            model: model.into(),
            created_at_ms: ts,
        }
    }

    /// State + router backed by an in-memory metadata DB. Returns the shared
    /// DB handle so tests can seed/inspect prompt history around requests.
    fn app_with_history_db() -> (axum::Router, Arc<Option<mold_db::MetadataDb>>) {
        let mut state = AppState::for_tests();
        let db = Arc::new(Some(mold_db::MetadataDb::open_in_memory().unwrap()));
        state.metadata_db = db.clone();
        (app_with_state(state), db)
    }

    fn seed_history(db: &Arc<Option<mold_db::MetadataDb>>, entries: &[(&str, &str, i64)]) {
        let db = db.as_ref().as_ref().expect("test DB present");
        let history = mold_db::PromptHistory::new(db);
        for (prompt, model, ts) in entries {
            history.push(&history_entry(prompt, model, *ts)).unwrap();
        }
    }

    #[tokio::test]
    async fn history_returns_recent_entries_newest_first() {
        let (app, db) = app_with_history_db();
        seed_history(
            &db,
            &[
                ("first", "flux-dev:q4", 1_000),
                ("second", "flux-dev:q4", 2_000),
                ("third", "sdxl:fp16", 3_000),
            ],
        );

        let resp = app
            .oneshot(Request::get("/api/history").body(Body::empty()).unwrap())
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
        let body = json_body(resp).await;
        let entries = body["entries"].as_array().expect("entries array");
        assert_eq!(entries.len(), 3);
        assert_eq!(
            entries[0],
            serde_json::json!({
                "prompt": "third",
                "model": "sdxl:fp16",
                "used_at": 3_000,
            })
        );
        assert_eq!(entries[2]["prompt"], "first");
    }

    #[tokio::test]
    async fn history_query_filters_by_substring() {
        let (app, db) = app_with_history_db();
        seed_history(
            &db,
            &[
                ("A Sunny Day", "m", 1),
                ("cloudy morning", "m", 2),
                ("SUNSET over sea", "m", 3),
            ],
        );

        let resp = app
            .oneshot(
                Request::get("/api/history?query=sun")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
        let body = json_body(resp).await;
        let prompts: Vec<&str> = body["entries"]
            .as_array()
            .unwrap()
            .iter()
            .map(|e| e["prompt"].as_str().unwrap())
            .collect();
        assert_eq!(prompts, vec!["SUNSET over sea", "A Sunny Day"]);
    }

    #[tokio::test]
    async fn history_limit_defaults_to_50_and_caps_at_500() {
        let (app, db) = app_with_history_db();
        let rows: Vec<(String, &str, i64)> = (0..510)
            .map(|i| (format!("p{i}"), "m", (i as i64 + 1) * 10))
            .collect();
        {
            let db = db.as_ref().as_ref().unwrap();
            let history = mold_db::PromptHistory::new(db);
            for (prompt, model, ts) in &rows {
                history.push(&history_entry(prompt, model, *ts)).unwrap();
            }
        }

        // Default limit is 50.
        let resp = app
            .clone()
            .oneshot(Request::get("/api/history").body(Body::empty()).unwrap())
            .await
            .unwrap();
        let body = json_body(resp).await;
        assert_eq!(body["entries"].as_array().unwrap().len(), 50);

        // Explicit limits are honored…
        let resp = app
            .clone()
            .oneshot(
                Request::get("/api/history?limit=2")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        let body = json_body(resp).await;
        assert_eq!(body["entries"].as_array().unwrap().len(), 2);

        // …but capped at 500.
        let resp = app
            .oneshot(
                Request::get("/api/history?limit=10000")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        let body = json_body(resp).await;
        assert_eq!(body["entries"].as_array().unwrap().len(), 500);
    }

    #[tokio::test]
    async fn history_returns_503_when_db_disabled() {
        // AppState::for_tests() has metadata_db = None — same state the
        // server boots into under MOLD_DB_DISABLE=1.
        let app = app_with_state(AppState::for_tests());
        let resp = app
            .oneshot(Request::get("/api/history").body(Body::empty()).unwrap())
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::SERVICE_UNAVAILABLE);
        let body = json_body(resp).await;
        assert_eq!(body["code"], "HISTORY_UNAVAILABLE");
        assert!(body["error"].as_str().unwrap().contains("metadata DB"));
    }

    #[tokio::test]
    async fn delete_history_clears_all_and_returns_204() {
        let (app, db) = app_with_history_db();
        seed_history(&db, &[("a", "m", 1), ("b", "m", 2)]);

        let resp = app
            .clone()
            .oneshot(Request::delete("/api/history").body(Body::empty()).unwrap())
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::NO_CONTENT);

        let resp = app
            .oneshot(Request::get("/api/history").body(Body::empty()).unwrap())
            .await
            .unwrap();
        let body = json_body(resp).await;
        assert_eq!(body["entries"], serde_json::json!([]));
    }

    #[tokio::test]
    async fn delete_history_keep_trims_to_most_recent_n() {
        let (app, db) = app_with_history_db();
        seed_history(
            &db,
            &[
                ("p0", "m", 100),
                ("p1", "m", 200),
                ("p2", "m", 300),
                ("p3", "m", 400),
            ],
        );

        let resp = app
            .clone()
            .oneshot(
                Request::delete("/api/history?keep=2")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::NO_CONTENT);

        let resp = app
            .oneshot(Request::get("/api/history").body(Body::empty()).unwrap())
            .await
            .unwrap();
        let body = json_body(resp).await;
        let prompts: Vec<&str> = body["entries"]
            .as_array()
            .unwrap()
            .iter()
            .map(|e| e["prompt"].as_str().unwrap())
            .collect();
        assert_eq!(prompts, vec!["p3", "p2"]);
    }

    #[tokio::test]
    async fn delete_history_returns_503_when_db_disabled() {
        let app = app_with_state(AppState::for_tests());
        let resp = app
            .oneshot(Request::delete("/api/history").body(Body::empty()).unwrap())
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::SERVICE_UNAVAILABLE);
        let body = json_body(resp).await;
        assert_eq!(body["code"], "HISTORY_UNAVAILABLE");
    }

    /// Engine+queue state (no worker) with an in-memory metadata DB — the
    /// setup for asserting that accepting a generation records history. The
    /// queue receiver is returned so the caller keeps the channel open.
    fn generating_app_with_history_db() -> (
        axum::Router,
        AppState,
        tokio::sync::mpsc::Receiver<crate::state::GenerationJob>,
    ) {
        let (mut state, rx) = AppState::with_engine_and_queue(MockEngine::ready());
        state.metadata_db = Arc::new(Some(mold_db::MetadataDb::open_in_memory().unwrap()));
        (app_with_state(state.clone()), state, rx)
    }

    async fn history_prompts(app: &axum::Router) -> Vec<String> {
        let resp = app
            .clone()
            .oneshot(Request::get("/api/history").body(Body::empty()).unwrap())
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
        json_body(resp).await["entries"]
            .as_array()
            .unwrap()
            .iter()
            .map(|e| e["prompt"].as_str().unwrap().to_string())
            .collect()
    }

    #[tokio::test]
    async fn generate_stream_records_prompt_history_on_accept() {
        let (app, _state, _rx) = generating_app_with_history_db();

        let resp = app
            .clone()
            .oneshot(
                Request::post("/api/generate/stream")
                    .header("content-type", "application/json")
                    .body(Body::from(generate_body("a cat in history", 512, 512)))
                    .unwrap(),
            )
            .await
            .unwrap();
        let status = resp.status();
        if status != StatusCode::OK {
            let body = json_body(resp).await;
            panic!("generate/stream answered {status}: {body}");
        }

        let resp = app
            .clone()
            .oneshot(Request::get("/api/history").body(Body::empty()).unwrap())
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
        let body = json_body(resp).await;
        let entries = body["entries"].as_array().unwrap();
        assert_eq!(entries.len(), 1, "accepted generation must land in history");
        assert_eq!(entries[0]["prompt"], "a cat in history");
        assert_eq!(entries[0]["model"], "mock-model");
        assert!(
            entries[0]["used_at"].as_i64().unwrap() > 0,
            "used_at must be stamped"
        );
    }

    #[tokio::test]
    async fn generate_stream_dedupes_consecutive_identical_prompts() {
        let (app, _state, _rx) = generating_app_with_history_db();

        for _ in 0..3 {
            let resp = app
                .clone()
                .oneshot(
                    Request::post("/api/generate/stream")
                        .header("content-type", "application/json")
                        .body(Body::from(generate_body("same prompt", 512, 512)))
                        .unwrap(),
                )
                .await
                .unwrap();
            assert_eq!(resp.status(), StatusCode::OK);
        }
        let resp = app
            .clone()
            .oneshot(
                Request::post("/api/generate/stream")
                    .header("content-type", "application/json")
                    .body(Body::from(generate_body("different prompt", 512, 512)))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::OK);

        // Batch siblings / retries collapse to one row; a new prompt appends.
        assert_eq!(
            history_prompts(&app).await,
            vec!["different prompt".to_string(), "same prompt".to_string()]
        );
    }

    #[tokio::test]
    async fn blocking_generate_records_prompt_history_on_accept() {
        // No queue worker: submit, verify the history row exists while the
        // job is still queued, then cancel to resolve the blocked request.
        let (app, state, _rx) = generating_app_with_history_db();

        let gen_app = app.clone();
        let gen_task = tokio::spawn(async move {
            gen_app
                .oneshot(
                    Request::post("/api/generate")
                        .header("content-type", "application/json")
                        .body(Body::from(generate_body("blocking prompt", 512, 512)))
                        .unwrap(),
                )
                .await
                .unwrap()
        });

        let id = wait_for_registered_job(&state).await;
        assert_eq!(
            history_prompts(&app).await,
            vec!["blocking prompt".to_string()],
            "history records at accept time, not completion"
        );

        let resp = app
            .oneshot(
                Request::delete(format!("/api/queue/{id}"))
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::NO_CONTENT);
        let _ = tokio::time::timeout(Duration::from_secs(5), gen_task).await;
    }

    // ── /api/config ──────────────────────────────────────────────────────────

    /// State + router backed by an in-memory metadata DB. Returns the shared
    /// DB handle so tests can seed/inspect settings rows around requests.
    fn app_with_settings_db() -> (axum::Router, Arc<Option<mold_db::MetadataDb>>) {
        let mut state = AppState::for_tests();
        let db = Arc::new(Some(mold_db::MetadataDb::open_in_memory().unwrap()));
        state.metadata_db = db.clone();
        (app_with_state(state), db)
    }

    async fn put_json(app: &axum::Router, uri: &str, body: &str) -> axum::http::Response<Body> {
        app.clone()
            .oneshot(
                Request::put(uri)
                    .header("content-type", "application/json")
                    .body(Body::from(body.to_string()))
                    .unwrap(),
            )
            .await
            .unwrap()
    }

    #[tokio::test]
    #[allow(clippy::await_holding_lock)]
    async fn config_list_reports_values_with_sources() {
        let _lock = env_lock().lock().unwrap_or_else(|e| e.into_inner());
        let (app, _db) = app_with_settings_db();
        let resp = app
            .oneshot(Request::get("/api/config").body(Body::empty()).unwrap())
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
        let body = json_body(resp).await;
        assert_eq!(body["profile"], "default");
        let entries = body["entries"].as_array().expect("entries array");
        assert!(!entries.is_empty());

        let find = |key: &str| {
            entries
                .iter()
                .find(|e| e["key"] == key)
                .unwrap_or_else(|| panic!("missing entry for {key}"))
                .clone()
        };
        // Bootstrap keys live in config.toml.
        assert_eq!(find("models_dir")["source"], "file");
        // User-preference keys live in the settings DB (post-#265 routing).
        assert_eq!(find("expand.enabled")["source"], "db");
        assert_eq!(find("default_steps")["source"], "db");
        // Values are typed JSON, mirroring `mold config list --json`.
        assert_eq!(
            find("server_port")["value"],
            serde_json::json!(mold_core::Config::default().server_port)
        );
        assert!(find("embed_metadata")["value"].is_boolean());
    }

    #[tokio::test]
    #[allow(clippy::await_holding_lock)]
    async fn config_list_marks_env_overridden_keys() {
        let _lock = env_lock().lock().unwrap_or_else(|e| e.into_inner());
        let prev = std::env::var("MOLD_EXPAND").ok();
        std::env::set_var("MOLD_EXPAND", "1");

        let (app, _db) = app_with_settings_db();
        let resp = app
            .oneshot(Request::get("/api/config").body(Body::empty()).unwrap())
            .await
            .unwrap();
        let body = json_body(resp).await;
        let entry = body["entries"]
            .as_array()
            .unwrap()
            .iter()
            .find(|e| e["key"] == "expand.enabled")
            .expect("expand.enabled entry")
            .clone();
        assert_eq!(entry["source"], "env");
        // env rows name the variable so UIs can say "Set by MOLD_EXPAND in
        // your environment" without guessing the mapping.
        assert_eq!(entry["env_var"], "MOLD_EXPAND");

        match prev {
            Some(v) => std::env::set_var("MOLD_EXPAND", v),
            None => std::env::remove_var("MOLD_EXPAND"),
        }
    }

    #[tokio::test]
    async fn config_get_key_returns_value_and_source() {
        let (app, _db) = app_with_settings_db();
        let resp = app
            .oneshot(
                Request::get("/api/config/server_port")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
        let body = json_body(resp).await;
        assert_eq!(body["key"], "server_port");
        assert_eq!(
            body["value"],
            serde_json::json!(mold_core::Config::default().server_port)
        );
        assert_eq!(body["source"], "file");
    }

    #[tokio::test]
    async fn config_get_unknown_key_returns_404() {
        let (app, _db) = app_with_settings_db();
        let resp = app
            .oneshot(
                Request::get("/api/config/definitely.not.a.key")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::NOT_FOUND);
        let body = json_body(resp).await;
        assert_eq!(body["code"], "UNKNOWN_CONFIG_KEY");
    }

    #[tokio::test]
    async fn config_put_db_key_persists_to_settings_db() {
        let (app, db) = app_with_settings_db();
        let resp = put_json(&app, "/api/config/default_steps", r#"{"value":12}"#).await;
        assert_eq!(resp.status(), StatusCode::OK);
        let body = json_body(resp).await;
        assert_eq!(body["key"], "default_steps");
        assert_eq!(body["value"], serde_json::json!(12));
        assert_eq!(body["source"], "db");

        // The row landed in the settings DB under the active profile…
        let handle = db.as_ref().as_ref().unwrap();
        let s = mold_db::Settings::for_profile(handle, "default");
        assert_eq!(
            s.get_int(mold_db::settings::GENERATE_DEFAULT_STEPS)
                .unwrap(),
            Some(12)
        );

        // …and the running server's config reflects it immediately.
        let resp = app
            .oneshot(
                Request::get("/api/config/default_steps")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        let body = json_body(resp).await;
        assert_eq!(body["value"], serde_json::json!(12));
    }

    #[tokio::test]
    #[allow(clippy::await_holding_lock)]
    async fn config_put_file_key_persists_to_toml() {
        let tmp = tempfile::tempdir().unwrap();
        let _home = MoldHomeGuard::set(tmp.path());

        // File-surface keys need no DB at all.
        let app = app_with_state(AppState::for_tests());
        let resp = put_json(&app, "/api/config/server_port", r#"{"value":8123}"#).await;
        assert_eq!(resp.status(), StatusCode::OK);
        let body = json_body(resp).await;
        assert_eq!(body["source"], "file");
        assert_eq!(body["value"], serde_json::json!(8123));

        let written = std::fs::read_to_string(tmp.path().join("config.toml")).unwrap();
        assert!(
            written.contains("server_port = 8123"),
            "config.toml must carry the new value: {written}"
        );
    }

    #[tokio::test]
    #[allow(clippy::await_holding_lock)]
    async fn config_put_env_overridden_key_returns_403() {
        let _lock = env_lock().lock().unwrap_or_else(|e| e.into_inner());
        let prev = std::env::var("MOLD_EXPAND").ok();
        std::env::set_var("MOLD_EXPAND", "1");

        let (app, _db) = app_with_settings_db();
        let resp = put_json(&app, "/api/config/expand.enabled", r#"{"value":false}"#).await;
        assert_eq!(resp.status(), StatusCode::FORBIDDEN);
        let body = json_body(resp).await;
        assert_eq!(body["code"], "ENV_OVERRIDDEN");
        assert!(
            body["error"].as_str().unwrap().contains("MOLD_EXPAND"),
            "error must name the env var: {body}"
        );

        match prev {
            Some(v) => std::env::set_var("MOLD_EXPAND", v),
            None => std::env::remove_var("MOLD_EXPAND"),
        }
    }

    #[tokio::test]
    async fn config_put_unknown_key_returns_422() {
        let (app, _db) = app_with_settings_db();
        let resp = put_json(&app, "/api/config/definitely.not.a.key", r#"{"value":1}"#).await;
        assert_eq!(resp.status(), StatusCode::UNPROCESSABLE_ENTITY);
        let body = json_body(resp).await;
        assert_eq!(body["code"], "UNKNOWN_CONFIG_KEY");
    }

    #[tokio::test]
    async fn config_put_invalid_value_returns_422() {
        let (app, _db) = app_with_settings_db();
        let resp = put_json(
            &app,
            "/api/config/server_port",
            r#"{"value":"not-a-number"}"#,
        )
        .await;
        assert_eq!(resp.status(), StatusCode::UNPROCESSABLE_ENTITY);
        let body = json_body(resp).await;
        assert_eq!(body["code"], "VALIDATION_ERROR");
    }

    #[tokio::test]
    async fn config_put_db_key_returns_503_when_db_disabled() {
        // default_steps routes to the settings DB — with the DB disabled the
        // write must fail loudly, mirroring history/chain-jobs.
        let app = app_with_state(AppState::for_tests());
        let resp = put_json(&app, "/api/config/default_steps", r#"{"value":12}"#).await;
        assert_eq!(resp.status(), StatusCode::SERVICE_UNAVAILABLE);
        let body = json_body(resp).await;
        assert_eq!(body["code"], "CONFIG_UNAVAILABLE");
    }

    #[tokio::test]
    #[allow(clippy::await_holding_lock)]
    async fn config_delete_resets_db_key_and_reports_fallback() {
        let tmp = tempfile::tempdir().unwrap();
        let _home = MoldHomeGuard::set(tmp.path());
        let (app, db) = app_with_settings_db();

        // Seed an override, then reset it.
        let resp = put_json(&app, "/api/config/default_steps", r#"{"value":12}"#).await;
        assert_eq!(resp.status(), StatusCode::OK);

        let resp = app
            .clone()
            .oneshot(
                Request::delete("/api/config/default_steps")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
        let body = json_body(resp).await;
        assert_eq!(body["key"], "default_steps");
        // The response reports the fallback value now that the row is gone.
        assert_eq!(
            body["value"],
            serde_json::json!(mold_core::Config::default().default_steps)
        );
        assert_eq!(body["source"], "default");

        // Row really dropped.
        let handle = db.as_ref().as_ref().unwrap();
        let s = mold_db::Settings::for_profile(handle, "default");
        assert_eq!(
            s.get_int(mold_db::settings::GENERATE_DEFAULT_STEPS)
                .unwrap(),
            None
        );
    }

    #[tokio::test]
    async fn config_delete_file_key_returns_422() {
        let (app, _db) = app_with_settings_db();
        let resp = app
            .oneshot(
                Request::delete("/api/config/models_dir")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::UNPROCESSABLE_ENTITY);
        let body = json_body(resp).await;
        assert_eq!(body["code"], "FILE_BACKED_KEY");
        assert!(body["error"].as_str().unwrap().contains("config.toml"));
    }

    #[tokio::test]
    async fn config_delete_returns_503_when_db_disabled() {
        let app = app_with_state(AppState::for_tests());
        let resp = app
            .oneshot(
                Request::delete("/api/config/default_steps")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::SERVICE_UNAVAILABLE);
        let body = json_body(resp).await;
        assert_eq!(body["code"], "CONFIG_UNAVAILABLE");
    }

    #[tokio::test]
    #[allow(clippy::await_holding_lock)]
    async fn config_profiles_lists_known_profiles_and_active() {
        let _lock = env_lock().lock().unwrap_or_else(|e| e.into_inner());
        let prev = std::env::var("MOLD_PROFILE").ok();
        std::env::remove_var("MOLD_PROFILE");

        let (app, db) = app_with_settings_db();
        // Seed a row under a second profile so it shows up in the listing.
        {
            let handle = db.as_ref().as_ref().unwrap();
            mold_db::Settings::for_profile(handle, "dev")
                .set_str("tui.theme", "nord")
                .unwrap();
        }
        let resp = app
            .oneshot(
                Request::get("/api/config/profiles")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
        let body = json_body(resp).await;
        assert_eq!(body["active"], "default");
        let profiles: Vec<&str> = body["profiles"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_str().unwrap())
            .collect();
        assert!(profiles.contains(&"default"), "got: {profiles:?}");
        assert!(profiles.contains(&"dev"), "got: {profiles:?}");

        match prev {
            Some(v) => std::env::set_var("MOLD_PROFILE", v),
            None => std::env::remove_var("MOLD_PROFILE"),
        }
    }

    #[tokio::test]
    #[allow(clippy::await_holding_lock)]
    async fn config_put_profile_switches_active_profile() {
        let _lock = env_lock().lock().unwrap_or_else(|e| e.into_inner());
        let prev = std::env::var("MOLD_PROFILE").ok();
        std::env::remove_var("MOLD_PROFILE");

        let (app, db) = app_with_settings_db();
        let resp = put_json(&app, "/api/config/profile", r#"{"name":"dev"}"#).await;
        assert_eq!(resp.status(), StatusCode::OK);
        let body = json_body(resp).await;
        assert_eq!(body["active"], "dev");

        // The switch is persisted as the profile.active meta-row under the
        // bootstrap-safe "default" profile.
        let handle = db.as_ref().as_ref().unwrap();
        let stored = mold_db::Settings::for_profile(handle, "default")
            .get_str(mold_db::settings::ACTIVE_PROFILE)
            .unwrap();
        assert_eq!(stored.as_deref(), Some("dev"));

        // Empty names are rejected.
        let resp = put_json(&app, "/api/config/profile", r#"{"name":"  "}"#).await;
        assert_eq!(resp.status(), StatusCode::UNPROCESSABLE_ENTITY);

        match prev {
            Some(v) => std::env::set_var("MOLD_PROFILE", v),
            None => std::env::remove_var("MOLD_PROFILE"),
        }
    }

    #[tokio::test]
    async fn config_profiles_return_503_when_db_disabled() {
        let app = app_with_state(AppState::for_tests());
        let resp = app
            .clone()
            .oneshot(
                Request::get("/api/config/profiles")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::SERVICE_UNAVAILABLE);
        let body = json_body(resp).await;
        assert_eq!(body["code"], "CONFIG_UNAVAILABLE");

        let resp = put_json(&app, "/api/config/profile", r#"{"name":"dev"}"#).await;
        assert_eq!(resp.status(), StatusCode::SERVICE_UNAVAILABLE);
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
            workers: vec![worker].into(),
        });
        install_worker_registry(&mut state);
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
    async fn status_uses_resource_telemetry_for_physical_gpu_memory() {
        let worker = gpu_worker_stub(1);
        let mut state = AppState::with_engine(MockEngine::ready());
        state.gpu_pool = Arc::new(crate::gpu_pool::GpuPool {
            workers: vec![worker].into(),
        });
        install_worker_registry(&mut state);
        state.resources.publish(mold_core::ResourceSnapshot {
            hostname: "gpu-host".to_string(),
            timestamp: 1,
            gpus: vec![mold_core::GpuSnapshot {
                ordinal: 1,
                name: "test-gpu-1".to_string(),
                backend: mold_core::GpuBackend::Cuda,
                vram_total: 48_000_000_000,
                vram_used: 35_431_000_000,
                vram_used_by_mold: Some(35_000_000_000),
                vram_used_by_other: Some(431_000_000),
                gpu_utilization: Some(0),
            }],
            system_ram: mold_core::RamSnapshot {
                total: 128_000_000_000,
                used: 32_000_000_000,
                available: None,
                used_by_mold: 2_000_000_000,
                used_by_other: 30_000_000_000,
            },
            cpu: None,
        });
        let app = app_with_state(state);

        let resp = app
            .oneshot(Request::get("/api/status").body(Body::empty()).unwrap())
            .await
            .unwrap();
        let status = json_body(resp).await;

        assert_eq!(status["gpus"][0]["vram_total_bytes"], 48_000_000_000_u64);
        assert_eq!(status["gpus"][0]["vram_used_bytes"], 35_431_000_000_u64);
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
        assert!(
            spec["paths"]["/api/devices"]["get"].is_object(),
            "spec should document GET /api/devices"
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

        // With no payload-selection header, preserve the legacy full-media wire
        // contract for desktop and older clients.
        let body = axum::body::to_bytes(resp.into_body(), 1024 * 1024)
            .await
            .unwrap();
        let text = String::from_utf8_lossy(&body);
        let complete = sse_json_event(&text, "complete");
        let encoded = complete["image"]
            .as_str()
            .expect("complete event should contain a base64 image string");
        assert!(!encoded.is_empty(), "legacy full payload must not be empty");
        assert_eq!(
            base64::engine::general_purpose::STANDARD
                .decode(encoded)
                .expect("full SSE image should be valid base64"),
            minimal_png(),
            "omitting X-Mold-SSE-Payload must preserve the full image bytes"
        );
    }

    #[tokio::test]
    async fn stream_invalid_payload_header_returns_422() {
        let app = app_with(MockEngine::ready());
        let resp = app
            .oneshot(
                Request::post("/api/generate/stream")
                    .header("content-type", "application/json")
                    .header("x-mold-sse-payload", "thumbnail-only")
                    .body(Body::from(generate_body("a robot", 768, 768)))
                    .unwrap(),
            )
            .await
            .unwrap();

        assert_eq!(resp.status(), StatusCode::UNPROCESSABLE_ENTITY);
        let body = json_body(resp).await;
        assert_eq!(body["code"], "VALIDATION_ERROR");
        assert_eq!(
            body["error"],
            "X-Mold-SSE-Payload must be 'metadata-only' when provided"
        );
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn stream_metadata_only_returns_saved_filename_without_base64_media() {
        let output_dir = tempfile::tempdir().unwrap();
        let (state, rx) = AppState::with_engine_and_queue(MockEngine::ready());
        state.config.write().await.output_dir =
            Some(output_dir.path().to_string_lossy().into_owned());
        let worker_state = state.clone();
        tokio::spawn(crate::queue::run_queue_worker(rx, worker_state));
        let app = create_router(state);

        let resp = app
            .oneshot(
                Request::post("/api/generate/stream")
                    .header("content-type", "application/json")
                    .header("x-mold-sse-payload", "metadata-only")
                    .body(Body::from(generate_body("a robot", 768, 768)))
                    .unwrap(),
            )
            .await
            .unwrap();

        assert_eq!(resp.status(), StatusCode::OK);
        let body = axum::body::to_bytes(resp.into_body(), 1024 * 1024)
            .await
            .unwrap();
        let text = String::from_utf8_lossy(&body);
        let complete = sse_json_event(&text, "complete");

        assert_eq!(complete["image"], "");
        assert!(complete.get("original_image").is_none());
        assert!(complete.get("video_thumbnail").is_none());
        assert!(complete.get("video_gif_preview").is_none());
        assert_eq!(complete["format"], "png");
        assert_eq!(complete["width"], 768);
        assert_eq!(complete["height"], 768);
        assert_eq!(complete["seed_used"], 42);
        assert_eq!(complete["model"], "mock-model");
        assert_eq!(complete["metadata"]["prompt"], "a robot");
        assert_eq!(complete["metadata"]["width"], 768);
        assert_eq!(complete["metadata"]["height"], 768);

        let filename = complete["filename"]
            .as_str()
            .expect("metadata-only completion should name the saved gallery output");
        assert!(
            filename.ends_with(".png"),
            "unexpected filename: {filename}"
        );
        assert!(
            output_dir.path().join(filename).is_file(),
            "metadata-only filename should identify the persisted output"
        );
        let encoded_png = base64::engine::general_purpose::STANDARD.encode(minimal_png());
        assert!(
            !text.contains(&encoded_png),
            "metadata-only response must not include base64 media bytes"
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
            workers: Vec::new().into(),
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
            instance_id: Arc::new(uuid::Uuid::new_v4().to_string()),
            discovery: Arc::new(crate::state::DiscoveryState::default()),
            gpu_pool: std::sync::Arc::new(crate::gpu_pool::GpuPool {
                workers: Vec::new().into(),
            }),
            generation_unavailable_reason: Arc::new(std::sync::RwLock::new(None)),
            device_registry: crate::device_registry::DeviceRegistry::empty(),
            queue_capacity: 200,
            model_cache: Arc::new(tokio::sync::Mutex::new(cache)),
            active_generation: Arc::new(std::sync::RwLock::new(None)),
            config: Arc::new(tokio::sync::RwLock::new(mold_core::Config::default())),
            start_time: std::time::Instant::now(),
            model_load_lock: Arc::new(tokio::sync::Mutex::new(())),
            pull_lock: Arc::new(tokio::sync::Mutex::new(())),
            queue,
            scheduled_work: crate::scheduler::ScheduledWorkHandle::default(),
            job_registry: crate::job_registry::JobRegistry::new(),
            scheduler_mutation_fence: Arc::new(tokio::sync::Mutex::new(())),
            queue_pause: crate::queue::QueuePause::new(),
            shared_pool: Arc::new(std::sync::Mutex::new(
                mold_inference::shared_pool::SharedPool::new(),
            )),
            shutdown_tx: Arc::new(tokio::sync::Mutex::new(None)),
            upscaler_cache: Arc::new(std::sync::Mutex::new(None)),
            metadata_db: Arc::new(None),
            gallery_publication_gate: crate::batch_transaction::GalleryPublicationGate::default(),
            chain_jobs: None,
            downloads: crate::downloads::DownloadQueue::new(),
            resources: crate::resources::ResourceBroadcaster::new(),
            events: crate::events::EventBroadcaster::new(),
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
            models_disk_cache: Arc::new(crate::state::ModelsDiskCache::default()),
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
            instance_id: Arc::new(uuid::Uuid::new_v4().to_string()),
            discovery: Arc::new(crate::state::DiscoveryState::default()),
            gpu_pool: std::sync::Arc::new(crate::gpu_pool::GpuPool {
                workers: Vec::new().into(),
            }),
            generation_unavailable_reason: Arc::new(std::sync::RwLock::new(None)),
            device_registry: crate::device_registry::DeviceRegistry::empty(),
            queue_capacity: 200,
            model_cache: Arc::new(tokio::sync::Mutex::new(cache)),
            active_generation: Arc::new(std::sync::RwLock::new(None)),
            config: Arc::new(tokio::sync::RwLock::new(mold_core::Config::default())),
            start_time: std::time::Instant::now(),
            model_load_lock: Arc::new(tokio::sync::Mutex::new(())),
            pull_lock: Arc::new(tokio::sync::Mutex::new(())),
            queue,
            scheduled_work: crate::scheduler::ScheduledWorkHandle::default(),
            job_registry: crate::job_registry::JobRegistry::new(),
            scheduler_mutation_fence: Arc::new(tokio::sync::Mutex::new(())),
            queue_pause: crate::queue::QueuePause::new(),
            shared_pool: Arc::new(std::sync::Mutex::new(
                mold_inference::shared_pool::SharedPool::new(),
            )),
            shutdown_tx: Arc::new(tokio::sync::Mutex::new(None)),
            upscaler_cache: Arc::new(std::sync::Mutex::new(None)),
            metadata_db: Arc::new(None),
            gallery_publication_gate: crate::batch_transaction::GalleryPublicationGate::default(),
            chain_jobs: None,
            downloads: crate::downloads::DownloadQueue::new(),
            resources: crate::resources::ResourceBroadcaster::new(),
            events: crate::events::EventBroadcaster::new(),
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
            models_disk_cache: Arc::new(crate::state::ModelsDiskCache::default()),
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
            instance_id: Arc::new(uuid::Uuid::new_v4().to_string()),
            discovery: Arc::new(crate::state::DiscoveryState::default()),
            gpu_pool: std::sync::Arc::new(crate::gpu_pool::GpuPool {
                workers: Vec::new().into(),
            }),
            generation_unavailable_reason: Arc::new(std::sync::RwLock::new(None)),
            device_registry: crate::device_registry::DeviceRegistry::empty(),
            queue_capacity: 200,
            model_cache: Arc::new(tokio::sync::Mutex::new(cache)),
            active_generation: Arc::new(std::sync::RwLock::new(None)),
            config: Arc::new(tokio::sync::RwLock::new(mold_core::Config::default())),
            start_time: std::time::Instant::now(),
            model_load_lock: Arc::new(tokio::sync::Mutex::new(())),
            pull_lock: Arc::new(tokio::sync::Mutex::new(())),
            queue,
            scheduled_work: crate::scheduler::ScheduledWorkHandle::default(),
            job_registry: crate::job_registry::JobRegistry::new(),
            scheduler_mutation_fence: Arc::new(tokio::sync::Mutex::new(())),
            queue_pause: crate::queue::QueuePause::new(),
            shared_pool: Arc::new(std::sync::Mutex::new(
                mold_inference::shared_pool::SharedPool::new(),
            )),
            shutdown_tx: Arc::new(tokio::sync::Mutex::new(None)),
            upscaler_cache: Arc::new(std::sync::Mutex::new(None)),
            metadata_db: Arc::new(None),
            gallery_publication_gate: crate::batch_transaction::GalleryPublicationGate::default(),
            chain_jobs: None,
            downloads: crate::downloads::DownloadQueue::new(),
            resources: crate::resources::ResourceBroadcaster::new(),
            events: crate::events::EventBroadcaster::new(),
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
            models_disk_cache: Arc::new(crate::state::ModelsDiskCache::default()),
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
    async fn discovery_peers_requires_the_configured_api_key() {
        let keys = std::collections::HashSet::from(["test-key".to_string()]);
        let auth = Some(std::sync::Arc::new(crate::auth::ApiKeySet::new(keys)));
        let app = app_with_auth(auth);

        let resp = app
            .oneshot(
                Request::get("/api/discovery/peers")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::UNAUTHORIZED);
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
    async fn gallery_media_token_endpoint_issues_scoped_streaming_credential() {
        const MEDIA_ROUTE: &str = "/api/gallery/image/clip.mp4";
        const TOKEN_REQUEST: &str = r#"{"path":"/api/gallery/image/clip.mp4"}"#;
        let output_dir = tempfile::tempdir().unwrap();
        let media_path = output_dir.path().join("clip.mp4");
        std::fs::write(&media_path, b"0123456789").unwrap();
        std::fs::write(output_dir.path().join("other.mp4"), b"abcdefghij").unwrap();

        let config = mold_core::Config {
            output_dir: Some(output_dir.path().to_string_lossy().into_owned()),
            ..Default::default()
        };
        let (tx, _rx) = tokio::sync::mpsc::channel(16);
        let state = AppState::empty(
            config,
            crate::state::QueueHandle::new(tx),
            AppState::empty_gpu_pool_for_test(),
            200,
        );
        let keys = std::collections::HashSet::from(["test-key".to_string()]);
        let key_set = std::sync::Arc::new(crate::auth::ApiKeySet::new_with_gallery_signing_secret(
            keys, [0x42; 32],
        ));
        let auth = Some(key_set.clone());
        let app = create_router(state)
            .layer(axum::middleware::from_fn(crate::auth::require_api_key))
            .layer(axum::middleware::from_fn_with_state(
                auth,
                crate::auth::inject_auth_state,
            ));

        // The issuing endpoint itself still requires the normal API key.
        let missing_key = app
            .clone()
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/api/gallery/media-token")
                    .header(axum::http::header::CONTENT_TYPE, "application/json")
                    .body(Body::from(TOKEN_REQUEST))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(missing_key.status(), StatusCode::UNAUTHORIZED);

        let before = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_secs();
        let response = app
            .clone()
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/api/gallery/media-token")
                    .header("x-api-key", "test-key")
                    .header(axum::http::header::CONTENT_TYPE, "application/json")
                    .body(Body::from(TOKEN_REQUEST))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::OK);
        assert_eq!(
            response
                .headers()
                .get(axum::http::header::CACHE_CONTROL)
                .and_then(|value| value.to_str().ok()),
            Some("no-store")
        );
        let ticket = json_body(response).await;
        assert_eq!(ticket["auth_required"], true);
        let token = ticket["token"].as_str().unwrap();
        let expires_at = ticket["expires_at"].as_u64().unwrap();
        let after = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_secs();
        assert!((before + crate::auth::GALLERY_MEDIA_TOKEN_TTL_SECS
            ..=after + crate::auth::GALLERY_MEDIA_TOKEN_TTL_SECS)
            .contains(&expires_at));
        assert!(!token.contains("test-key"));

        let full_response = app
            .clone()
            .oneshot(
                Request::builder()
                    .uri(format!(
                        "/api/gallery/image/clip.mp4?media_token={token}&expires={expires_at}"
                    ))
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(full_response.status(), StatusCode::OK);
        assert_eq!(
            full_response
                .headers()
                .get(axum::http::header::CACHE_CONTROL)
                .and_then(|value| value.to_str().ok()),
            Some("private, no-store")
        );
        let body = axum::body::to_bytes(full_response.into_body(), 1024)
            .await
            .unwrap();
        assert_eq!(body.as_ref(), b"0123456789");

        // A browser-style Range request succeeds without X-Api-Key and keeps
        // the existing video streaming semantics intact.
        let range_response = app
            .clone()
            .oneshot(
                Request::builder()
                    .uri(format!(
                        "/api/gallery/image/clip.mp4?media_token={token}&expires={expires_at}"
                    ))
                    .header(axum::http::header::RANGE, "bytes=2-5")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(range_response.status(), StatusCode::PARTIAL_CONTENT);
        assert_eq!(
            range_response
                .headers()
                .get(axum::http::header::CACHE_CONTROL)
                .and_then(|value| value.to_str().ok()),
            Some("private, no-store")
        );
        assert_eq!(
            range_response
                .headers()
                .get(axum::http::header::CONTENT_RANGE)
                .and_then(|value| value.to_str().ok()),
            Some("bytes 2-5/10")
        );
        let body = axum::body::to_bytes(range_response.into_body(), 1024)
            .await
            .unwrap();
        assert_eq!(body.as_ref(), b"2345");

        // Media stacks may probe with HEAD before opening a Range stream. It
        // uses the same read-only ticket but never broadens it to other verbs.
        let head_response = app
            .clone()
            .oneshot(
                Request::builder()
                    .method("HEAD")
                    .uri(format!(
                        "{MEDIA_ROUTE}?media_token={token}&expires={expires_at}"
                    ))
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(head_response.status(), StatusCode::OK);
        assert_eq!(
            head_response
                .headers()
                .get(axum::http::header::CACHE_CONTROL)
                .and_then(|value| value.to_str().ok()),
            Some("private, no-store")
        );
        let head_body = axum::body::to_bytes(head_response.into_body(), 1024)
            .await
            .unwrap();
        assert!(head_body.is_empty());

        // Tampered and correctly signed-but-expired tickets both remain 401.
        let invalid = app
            .clone()
            .oneshot(
                Request::builder()
                    .uri(format!(
                        "/api/gallery/image/clip.mp4?media_token=invalid&expires={expires_at}"
                    ))
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(invalid.status(), StatusCode::UNAUTHORIZED);

        let wrong_filename = app
            .clone()
            .oneshot(
                Request::builder()
                    .uri(format!(
                        "/api/gallery/image/other.mp4?media_token={token}&expires={expires_at}"
                    ))
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(wrong_filename.status(), StatusCode::UNAUTHORIZED);

        let expired_at = before.saturating_sub(1);
        let expired_token = key_set.sign_gallery_media_token_for_tests(MEDIA_ROUTE, expired_at);
        let expired = app
            .clone()
            .oneshot(
                Request::builder()
                    .uri(format!(
                        "/api/gallery/image/clip.mp4?media_token={expired_token}&expires={expired_at}"
                    ))
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(expired.status(), StatusCode::UNAUTHORIZED);

        // The ticket is scoped to GET /api/gallery/image/:filename. It cannot
        // authenticate another API route or a destructive gallery method.
        let wrong_path = app
            .clone()
            .oneshot(
                Request::builder()
                    .uri(format!(
                        "/api/status?media_token={token}&expires={expires_at}"
                    ))
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(wrong_path.status(), StatusCode::UNAUTHORIZED);

        let delete = app
            .clone()
            .oneshot(
                Request::builder()
                    .method("DELETE")
                    .uri(format!(
                        "/api/gallery/image/clip.mp4?media_token={token}&expires={expires_at}"
                    ))
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(delete.status(), StatusCode::UNAUTHORIZED);
        assert!(media_path.is_file());

        let invalid_issue_path = app
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/api/gallery/media-token")
                    .header("x-api-key", "test-key")
                    .header(axum::http::header::CONTENT_TYPE, "application/json")
                    .body(Body::from(r#"{"path":"/api/status"}"#))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(
            invalid_issue_path.status(),
            StatusCode::UNPROCESSABLE_ENTITY
        );
    }

    #[tokio::test]
    async fn gallery_media_token_endpoint_reports_when_auth_is_disabled() {
        let app = app_with_auth(None);
        let response = app
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/api/gallery/media-token")
                    // Simulate a stale key retained for a host that disabled auth.
                    .header("x-api-key", "stale-key")
                    .header(axum::http::header::CONTENT_TYPE, "application/json")
                    .body(Body::from(r#"{"path":"/api/gallery/image/clip.mp4"}"#))
                    .unwrap(),
            )
            .await
            .unwrap();

        assert_eq!(response.status(), StatusCode::OK);
        let body = json_body(response).await;
        assert_eq!(body["auth_required"], false);
        assert!(body["token"].is_null());
        assert!(body["expires_at"].is_null());
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
            batch_id: None,
            batch_index: None,
            batch_count: None,
            model: "mock-model".into(),
            seed: 7,
            steps: 4,
            guidance: 1.0,
            width: 64,
            height: 64,
            generation_width: None,
            generation_height: None,
            strength: None,
            source_image_name: None,
            source_image_sha256: None,
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
            workers: Vec::new().into(),
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
            workers: Vec::new().into(),
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

    #[tokio::test]
    async fn every_gallery_observer_and_mutator_waits_for_atomic_publication_writer() {
        let dir = tempfile::tempdir().unwrap();
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
        let gate = state.gallery_publication_gate.clone();
        let writer = gate.write().await;
        let app = app_with_state(state);
        let cases = [
            (Method::GET, "/api/gallery", StatusCode::OK),
            (
                Method::GET,
                "/api/gallery/image/missing.png",
                StatusCode::NOT_FOUND,
            ),
            (
                Method::GET,
                "/api/gallery/thumbnail/missing.png",
                StatusCode::NOT_FOUND,
            ),
            (
                Method::GET,
                "/api/gallery/preview/missing.mp4",
                StatusCode::NOT_FOUND,
            ),
            (
                Method::DELETE,
                "/api/gallery/image/missing.png",
                StatusCode::NO_CONTENT,
            ),
        ];
        let mut requests: Vec<_> = cases
            .into_iter()
            .map(|(method, uri, expected)| {
                let app = app.clone();
                (
                    expected,
                    tokio::spawn(async move {
                        app.oneshot(
                            Request::builder()
                                .method(method)
                                .uri(uri)
                                .body(Body::empty())
                                .unwrap(),
                        )
                        .await
                        .unwrap()
                    }),
                )
            })
            .collect();

        for (_, request) in &mut requests {
            assert!(
                tokio::time::timeout(Duration::from_millis(20), request)
                    .await
                    .is_err(),
                "a gallery route observed a transaction while its writer gate was held"
            );
        }
        drop(writer);
        for (expected, request) in requests {
            let response = tokio::time::timeout(Duration::from_secs(1), request)
                .await
                .unwrap()
                .unwrap();
            assert_eq!(response.status(), expected);
        }
    }

    #[tokio::test]
    async fn gallery_listings_share_the_publication_reader_side() {
        let dir = tempfile::tempdir().unwrap();
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
        let gate = state.gallery_publication_gate.clone();
        let reader = gate.read().await;
        let app = app_with_state(state);

        let response = tokio::time::timeout(
            Duration::from_secs(1),
            app.oneshot(
                Request::builder()
                    .uri("/api/gallery")
                    .body(Body::empty())
                    .unwrap(),
            ),
        )
        .await
        .expect("a second gallery reader must not serialize behind the first")
        .unwrap();

        drop(reader);
        assert_eq!(response.status(), StatusCode::OK);
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
            workers: Vec::new().into(),
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
            batch_id: None,
            batch_index: None,
            batch_count: None,
            model: "m".into(),
            seed: 0,
            steps: 0,
            guidance: 0.0,
            width: 1,
            height: 1,
            generation_width: None,
            generation_height: None,
            strength: None,
            source_image_name: None,
            source_image_sha256: None,
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
            workers: Vec::new().into(),
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

    /// Gallery delete must announce itself on the server-wide event stream
    /// so clients displaying the gallery drop the tile without a refetch.
    #[tokio::test]
    async fn gallery_delete_emits_gallery_removed_event() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("doomed.png"), vec![0u8; 16]).unwrap();

        let config = mold_core::Config {
            output_dir: Some(dir.path().to_string_lossy().into_owned()),
            ..Default::default()
        };
        let (tx, _rx) = tokio::sync::mpsc::channel(16);
        let state = AppState::empty(
            config,
            crate::state::QueueHandle::new(tx),
            AppState::empty_gpu_pool_for_test(),
            200,
        );
        let mut events_rx = state.events.subscribe();
        let app = app_with_state(state);

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

        match events_rx.try_recv().unwrap() {
            mold_core::ServerEvent::GalleryRemoved { filename } => {
                assert_eq!(filename, "doomed.png");
            }
            other => panic!("expected gallery_removed, got {other:?}"),
        }
    }

    // ── GET /api/events ─────────────────────────────────────────────────────

    #[tokio::test]
    async fn get_api_events_streams_published_server_events() {
        use futures::StreamExt as _;
        let state = AppState::empty(
            mold_core::Config::default(),
            crate::state::QueueHandle::new(tokio::sync::mpsc::channel(1).0),
            AppState::empty_gpu_pool_for_test(),
            200,
        );
        let app = app_with_state(state.clone());

        let res = app
            .oneshot(
                Request::builder()
                    .uri("/api/events")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(res.status(), StatusCode::OK);
        let ct = res
            .headers()
            .get(axum::http::header::CONTENT_TYPE)
            .and_then(|v| v.to_str().ok())
            .unwrap_or("");
        assert!(
            ct.starts_with("text/event-stream"),
            "expected SSE content-type, got: {ct}"
        );

        // Publish AFTER subscribing (SSE response already established) —
        // registering on the job registry must surface as a job_queued frame.
        let state_for_send = state.clone();
        tokio::spawn(async move {
            tokio::time::sleep(std::time::Duration::from_millis(50)).await;
            state_for_send.job_registry.register("j1", "flux-dev:q4");
        });

        let mut body = res.into_body().into_data_stream();
        let mut saw_queued = false;
        let deadline = tokio::time::Instant::now() + std::time::Duration::from_secs(2);
        while tokio::time::Instant::now() < deadline {
            match tokio::time::timeout(std::time::Duration::from_millis(300), body.next()).await {
                Ok(Some(Ok(bytes))) => {
                    let text = String::from_utf8_lossy(&bytes).to_string();
                    if text.contains("\"type\":\"job_queued\"") && text.contains("\"id\":\"j1\"") {
                        saw_queued = true;
                        break;
                    }
                }
                _ => continue,
            }
        }
        assert!(saw_queued, "did not observe a job_queued SSE event");
    }

    #[tokio::test]
    async fn capabilities_reports_events_available() {
        let app = app_empty();
        let resp = app
            .oneshot(
                Request::builder()
                    .uri("/api/capabilities")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
        let body = json_body(resp).await;
        assert_eq!(body["events"]["available"], true);
    }

    #[tokio::test]
    async fn put_model_placement_updates_config_and_persists() {
        let tmp = tempfile::tempdir().unwrap();
        let _home = MoldHomeGuard::set(tmp.path());
        let app = app_empty();
        // Re-create state inside this test with mutable access.
        let (tx, _rx) = tokio::sync::mpsc::channel(16);
        let queue = crate::state::QueueHandle::new(tx);
        let gpu_pool = std::sync::Arc::new(crate::gpu_pool::GpuPool {
            workers: vec![gpu_worker_stub(1)].into(),
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
    }

    #[tokio::test]
    async fn put_model_placement_returns_500_when_save_fails() {
        // Point MOLD_HOME at a regular file so `config.toml` cannot be created
        // underneath it — `Config::save()` must return `Err`.
        let tmp = tempfile::tempdir().unwrap();
        let blocker = tmp.path().join("not-a-dir");
        std::fs::write(&blocker, "blocker").unwrap();
        let _home = MoldHomeGuard::set(&blocker);

        let (tx, _rx) = tokio::sync::mpsc::channel(16);
        let queue = crate::state::QueueHandle::new(tx);
        let gpu_pool = std::sync::Arc::new(crate::gpu_pool::GpuPool {
            workers: Vec::new().into(),
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
            workers: vec![gpu_worker_stub(1)].into(),
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
    async fn get_model_placement_returns_saved_value() {
        let tmp = tempfile::tempdir().unwrap();
        let _home = MoldHomeGuard::set(tmp.path());
        let (tx, _rx) = tokio::sync::mpsc::channel(16);
        let queue = crate::state::QueueHandle::new(tx);
        let gpu_pool = std::sync::Arc::new(crate::gpu_pool::GpuPool {
            workers: vec![gpu_worker_stub(1)].into(),
        });
        let state = AppState::empty(mold_core::Config::default(), queue, gpu_pool, 200);
        let app = crate::routes::create_router(state);

        let body = serde_json::json!({
            "text_encoders": { "kind": "cpu" },
            "advanced": {
                "transformer": { "kind": "gpu", "ordinal": 1 },
                "vae": { "kind": "auto" },
                "t5": { "kind": "cpu" }
            }
        });
        let put = app
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
        assert_eq!(put.status(), StatusCode::OK);

        // The editor hydrates from this read; it must echo what was saved.
        let get = app
            .oneshot(
                Request::builder()
                    .method("GET")
                    .uri("/api/config/model/flux-dev%3Aq4/placement")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(get.status(), StatusCode::OK);
        let got = json_body(get).await;
        assert_eq!(got["text_encoders"]["kind"], "cpu");
        assert_eq!(got["advanced"]["transformer"]["kind"], "gpu");
        assert_eq!(got["advanced"]["transformer"]["ordinal"], 1);
        assert_eq!(got["advanced"]["t5"]["kind"], "cpu");
    }

    #[tokio::test]
    async fn get_model_placement_returns_404_when_none_saved() {
        let app = app_empty();
        let resp = app
            .oneshot(
                Request::builder()
                    .method("GET")
                    .uri("/api/config/model/flux-dev%3Aq4/placement")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::NOT_FOUND);
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
        assert!(v["active_jobs"].is_array());
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
                workers: Vec::new().into(),
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
                available: None,
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
    async fn resource_routes_expose_only_frozen_cuda_visible_inventory() {
        use futures::StreamExt as _;

        let _lock = env_lock().lock().unwrap_or_else(|e| e.into_inner());
        let visible_uuid = [0xaa; 16];
        let targets = vec![crate::resources::TelemetryTarget::cuda(
            0,
            visible_uuid,
            mold_inference::device::CudaDeviceKind::FullGpu,
            "visible GPU".into(),
            24 * 1024 * 1024 * 1024,
        )];
        let gpus = crate::resources::SmiSource::parse_visible_snapshot(
            "0, GPU-bbbbbbbb-bbbb-bbbb-bbbb-bbbbbbbbbbbb, hidden physical GPU, 24576, 900\n\
             7, GPU-aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa, visible UUID GPU, 24576, 300",
            &targets,
        );
        assert_eq!(gpus.len(), 1, "test setup must apply the frozen inventory");

        let state = AppState::empty(
            mold_core::Config::default(),
            crate::state::QueueHandle::new(tokio::sync::mpsc::channel(1).0),
            std::sync::Arc::new(crate::gpu_pool::GpuPool {
                workers: Vec::new().into(),
            }),
            200,
        );
        state.resources.publish(mold_core::ResourceSnapshot {
            hostname: "unit-test".into(),
            timestamp: 12345,
            gpus,
            system_ram: mold_core::RamSnapshot {
                total: 1,
                used: 0,
                available: None,
                used_by_mold: 0,
                used_by_other: 0,
            },
            cpu: None,
        });
        let app = create_router(state);

        let one_shot = json_body(
            app.clone()
                .oneshot(
                    Request::builder()
                        .uri("/api/resources")
                        .body(Body::empty())
                        .unwrap(),
                )
                .await
                .unwrap(),
        )
        .await;
        assert_eq!(one_shot["gpus"].as_array().unwrap().len(), 1);
        assert_eq!(one_shot["gpus"][0]["ordinal"], 0);
        assert_eq!(one_shot["gpus"][0]["name"], "visible UUID GPU");
        assert!(
            !one_shot.to_string().contains("hidden physical GPU"),
            "one-shot resources leaked a CUDA-hidden physical GPU"
        );

        let response = app
            .oneshot(
                Request::builder()
                    .uri("/api/resources/stream")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        let mut stream = response.into_body().into_data_stream();
        let bytes = tokio::time::timeout(std::time::Duration::from_secs(1), stream.next())
            .await
            .expect("cached resource SSE frame should arrive immediately")
            .expect("resource SSE stream should yield a frame")
            .expect("resource SSE frame should be readable");
        let frame = String::from_utf8_lossy(&bytes);
        assert!(frame.contains("visible UUID GPU"));
        assert!(
            !frame.contains("hidden physical GPU"),
            "resource stream leaked a CUDA-hidden physical GPU"
        );
    }

    #[tokio::test]
    async fn get_api_resources_stream_sets_sse_content_type() {
        let _lock = env_lock().lock().unwrap_or_else(|e| e.into_inner());
        let state = AppState::empty(
            mold_core::Config::default(),
            crate::state::QueueHandle::new(tokio::sync::mpsc::channel(1).0),
            std::sync::Arc::new(crate::gpu_pool::GpuPool {
                workers: Vec::new().into(),
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
                workers: Vec::new().into(),
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
        assert_eq!(limits["supports_sequence"], true);
        assert!(limits["sequence_unsupported_reason"].is_null());
    }

    #[tokio::test]
    async fn capabilities_chain_limits_rejects_ltx2_two_stage_pipeline_up_front() {
        let app = app_empty();
        let response = app
            .oneshot(
                Request::get("/api/capabilities/chain-limits?model=ltx-2.3-22b-dev:fp8")
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
        assert_eq!(limits["supports_sequence"], false);
        assert!(limits["sequence_unsupported_reason"]
            .as_str()
            .is_some_and(|reason| reason.contains("two-stage")));
    }

    #[tokio::test]
    async fn capabilities_chain_limits_accepts_installed_catalog_ltx2_sidecar() {
        let models_dir = tempfile::tempdir().unwrap();
        let install_dir = models_dir.path().join("cv-3143864");
        std::fs::create_dir_all(&install_dir).unwrap();
        std::fs::write(install_dir.join("model.safetensors"), []).unwrap();
        mold_catalog::sidecar::write_sidecar(
            &install_dir.join(mold_catalog::sidecar::SIDECAR_FILENAME),
            &mold_catalog::sidecar::CatalogSidecar {
                schema: 1,
                id: "cv:3143864".into(),
                source: "civitai".into(),
                source_id: "3143864".into(),
                name: "LTX 2.3 INT4 ConvRot".into(),
                author: None,
                family: "ltx2".into(),
                family_role: "finetune".into(),
                sub_family: Some("v2.3".into()),
                kind: "checkpoint".into(),
                modality: "video".into(),
                nsfw: None,
                description: None,
                tags: vec![],
                license: None,
                page_url: None,
                thumbnail_url: None,
                size_bytes: Some(0),
                supported: true,
                trained_words: vec![],
                primary_filename_rel: "model.safetensors".into(),
                written_at: 0,
            },
        )
        .unwrap();
        let config = mold_core::Config {
            models_dir: models_dir.path().to_string_lossy().into_owned(),
            ..Default::default()
        };
        let (tx, _rx) = tokio::sync::mpsc::channel(16);
        let state = AppState::empty(
            config,
            crate::state::QueueHandle::new(tx),
            std::sync::Arc::new(crate::gpu_pool::GpuPool {
                workers: Vec::new().into(),
            }),
            200,
        );
        let response = create_router(state)
            .oneshot(
                Request::get("/api/capabilities/chain-limits?model=cv:3143864")
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
        assert_eq!(limits["model"], "cv:3143864");
        assert_eq!(limits["frames_per_clip_cap"], 97);
        assert_eq!(
            limits["supports_audio"], false,
            "chain limits must preserve the checkpoint-specific audio capability",
        );
        assert_eq!(
            limits["supports_sequence"], true,
            "single-file catalog checkpoints without an upscaler use one-stage",
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
