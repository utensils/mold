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

    struct EnvVarGuard {
        _lock: std::sync::MutexGuard<'static, ()>,
        name: &'static str,
        previous: Option<std::ffi::OsString>,
    }

    impl EnvVarGuard {
        fn set(name: &'static str, value: &std::ffi::OsStr) -> Self {
            let lock = env_lock()
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner());
            let previous = std::env::var_os(name);
            std::env::set_var(name, value);
            Self {
                _lock: lock,
                name,
                previous,
            }
        }
    }

    impl Drop for EnvVarGuard {
        fn drop(&mut self) {
            match &self.previous {
                Some(value) => std::env::set_var(self.name, value),
                None => std::env::remove_var(self.name),
            }
        }
    }

    /// Parse response body as JSON and return the value.
    async fn json_body(resp: axum::http::Response<Body>) -> serde_json::Value {
        let body = axum::body::to_bytes(resp.into_body(), 1024 * 1024)
            .await
            .unwrap();
        serde_json::from_slice(&body).unwrap()
    }

    #[cfg(feature = "h3-private-uat")]
    fn private_h3_fl2va_body(batch_size: u32) -> serde_json::Value {
        serde_json::json!({
            "prompt": "private FL2VA route fence",
            "model": mold_core::minimax_h3::FL2VA_COMFY,
            "width": mold_core::minimax_h3::DEFAULT_WIDTH,
            "height": mold_core::minimax_h3::DEFAULT_HEIGHT,
            "steps": mold_core::minimax_h3::DEFAULT_STEPS,
            "guidance": 0.0,
            "batch_size": batch_size,
            "frames": mold_core::minimax_h3::MIN_FRAMES,
            "fps": mold_core::minimax_h3::FIXED_FPS,
            "output_format": "mp4"
        })
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
        model_name: &'static str,
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
            Self::ready_for_model("mock-model")
        }

        fn ready_for_model(model_name: &'static str) -> Self {
            Self {
                model_name,
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
                model_name: "mock-model",
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
                model_name: "mock-model",
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
                model_name: "mock-model",
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
                model_name: "mock-model",
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
                model_name: "mock-model",
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
                model_name: "mock-model",
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
                request_warnings: Vec::new(),
                audio: None,
                images,
                generation_time_ms: 1,
                model: req.model.clone(),
                seed_used: req.seed.unwrap_or(42),
                video: None,
                gpu: None,
            })
        }

        fn model_name(&self) -> &str {
            self.model_name
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

    #[tokio::test]
    async fn generation_fixture_disables_real_gallery_output_by_default() {
        let (state, _rx) = AppState::with_engine_and_queue(MockEngine::ready());
        let config = state.config.read().await;

        assert!(
            state.is_output_disabled(&config),
            "mock generation tests must opt into an isolated output directory"
        );
    }

    #[test]
    fn generation_fixture_ignores_process_output_override() {
        let _env = env_lock()
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        let previous = std::env::var_os("MOLD_OUTPUT_DIR");
        unsafe { std::env::set_var("MOLD_OUTPUT_DIR", "/tmp/mold-test-must-not-write") };

        let (state, _rx) = AppState::with_engine_and_queue(MockEngine::ready());
        let config = state.config.try_read().unwrap();
        let environment_would_enable_output = !config.is_output_disabled();
        let fixture_disables_output = state.is_output_disabled(&config);
        drop(config);

        match previous {
            Some(value) => unsafe { std::env::set_var("MOLD_OUTPUT_DIR", value) },
            None => unsafe { std::env::remove_var("MOLD_OUTPUT_DIR") },
        }

        assert!(environment_would_enable_output);
        assert!(
            fixture_disables_output,
            "mock generation tests must ignore process-level output overrides"
        );
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

    fn publish_test_gpu_resources(
        resources: &crate::resources::ResourceBroadcaster,
        total: u64,
        used: u64,
    ) {
        publish_test_gpu_resource_snapshots(resources, &[(0, total, used)]);
    }

    fn publish_test_gpu_resource_snapshots(
        resources: &crate::resources::ResourceBroadcaster,
        gpus: &[(usize, u64, u64)],
    ) {
        resources.publish(mold_core::ResourceSnapshot {
            hostname: "estimate-host".into(),
            timestamp: gpus
                .iter()
                .map(|(_, _, used)| *used)
                .max()
                .and_then(|used| i64::try_from(used).ok())
                .unwrap_or_default(),
            gpus: gpus
                .iter()
                .map(|(ordinal, total, used)| mold_core::GpuSnapshot {
                    ordinal: *ordinal,
                    name: format!("estimate-gpu-{ordinal}"),
                    backend: mold_core::GpuBackend::Cuda,
                    vram_total: *total,
                    vram_used: *used,
                    vram_used_by_mold: Some(*used),
                    vram_used_by_other: Some(0),
                    gpu_utilization: Some(0),
                })
                .collect(),
            system_ram: mold_core::RamSnapshot {
                total: 128_000_000_000,
                used: 32_000_000_000,
                available: None,
                used_by_mold: 1_000_000_000,
                used_by_other: 31_000_000_000,
            },
            cpu: None,
        });
    }

    fn app_with_test_gpu_resources(
        total: u64,
        used: u64,
    ) -> (axum::Router, Arc<crate::resources::ResourceBroadcaster>) {
        let (tx, _rx) = tokio::sync::mpsc::channel(16);
        let mut state = AppState::empty(
            mold_core::Config::default(),
            crate::state::QueueHandle::new(tx),
            Arc::new(crate::gpu_pool::GpuPool {
                workers: vec![gpu_worker_stub(0)].into(),
            }),
            200,
        );
        install_worker_registry(&mut state);
        let resources = state.resources.clone();
        publish_test_gpu_resources(&resources, total, used);
        (app_with_state(state), resources)
    }

    fn app_with_test_gpu_snapshots(
        gpus: &[(usize, u64, u64)],
        disabled_ordinals: &[usize],
    ) -> axum::Router {
        app_with_test_gpu_snapshots_and_config(
            gpus,
            disabled_ordinals,
            mold_core::Config::default(),
        )
    }

    fn app_with_test_gpu_snapshots_and_config(
        gpus: &[(usize, u64, u64)],
        disabled_ordinals: &[usize],
        config: mold_core::Config,
    ) -> axum::Router {
        let (tx, _rx) = tokio::sync::mpsc::channel(16);
        let mut state = AppState::empty(
            config,
            crate::state::QueueHandle::new(tx),
            Arc::new(crate::gpu_pool::GpuPool {
                workers: gpus
                    .iter()
                    .map(|(ordinal, _, _)| gpu_worker_stub(*ordinal))
                    .collect::<Vec<_>>()
                    .into(),
            }),
            200,
        );
        install_worker_registry(&mut state);
        for ordinal in disabled_ordinals {
            let id = format!("cuda:{ordinal:032x}");
            state
                .device_registry
                .set_desired_enabled(&id, false)
                .unwrap();
        }
        publish_test_gpu_resource_snapshots(&state.resources, gpus);
        app_with_state(state)
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
            collection: None,
            tags: None,
            title: None,
            model: "ltx-2-19b-distilled:fp8".into(),
            stages: vec![route_chain_stage("first shot", TransitionMode::Smooth)],
            motion_tail_frames: 1,
            width: 64,
            height: 64,
            fps: 8,
            seed: Some(42),
            steps: 2,
            guidance: 1.0,
            strength: 1.0,
            output_format: OutputFormat::Mp4,
            placement: None,
            original_prompt: None,
            prompt_transform: None,
            batch_id: None,
            batch_index: None,
            batch_count: None,
            output_mode: None,
            prompt: None,
            total_frames: None,
            clip_frames: None,
            source_image: None,
            enable_audio: None,
        }
    }

    #[tokio::test]
    #[allow(clippy::await_holding_lock)]
    async fn chain_validation_returns_normalized_plan_without_job_storage() {
        let _lock = env_lock().lock().unwrap_or_else(|e| e.into_inner());
        let models_dir = test_models_dir("chain-estimate");
        populate_manifest_files(&models_dir, "ltx-2-19b-distilled:fp8");
        std::env::set_var("MOLD_MODELS_DIR", &models_dir);
        let (app, resources) = app_with_test_gpu_resources(48_000_000_000, 8_000_000_000);
        let mut request = route_chain_request();
        request.stages[0].transition = TransitionMode::Cut;
        let mut continuation = route_chain_stage("second shot", TransitionMode::Fade);
        continuation.source_image = Some(vec![1, 2, 3]);
        continuation.negative_prompt = Some("camera shake".into());
        request.stages.push(continuation);

        let response = app
            .clone()
            .oneshot(
                Request::post("/api/generate/chain/validate")
                    .header("content-type", "application/json")
                    .body(Body::from(serde_json::to_vec(&request).unwrap()))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::OK);
        let body = json_body(response).await;

        assert_eq!(body["model"], "ltx-2-19b-distilled:fp8");
        assert_eq!(body["stage_count"], 2);
        assert_eq!(body["estimated_total_frames"], 16);
        assert_eq!(body["estimated_duration_ms"], 2_000);
        assert_eq!(body["stages"][0]["transition"], "smooth");
        assert_eq!(body["stages"][0]["output_frames"], 7);
        assert_eq!(body["stages"][1]["transition"], "fade");
        assert_eq!(body["stages"][1]["output_frames"], 9);
        assert_eq!(body["stages"][1]["has_source_image"], true);
        assert_eq!(body["stages"][1]["has_negative_prompt"], true);
        assert!(
            body["warnings"][0]
                .as_str()
                .unwrap()
                .contains("opening clip"),
            "normalization warning should explain the coerced first transition: {body}"
        );
        let stable_vram = body["vram_estimate"].clone();
        assert!(stable_vram["worst_case_bytes"].as_u64().unwrap() > 0);
        assert_eq!(stable_vram["fits"], true);

        // A busy GPU changes current free memory, not whether this normalized
        // sequence fits the host after earlier queued work releases VRAM.
        publish_test_gpu_resources(&resources, 48_000_000_000, 40_000_000_000);
        let busy_response = app
            .clone()
            .oneshot(
                Request::post("/api/generate/chain/validate")
                    .header("content-type", "application/json")
                    .body(Body::from(serde_json::to_vec(&request).unwrap()))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(busy_response.status(), StatusCode::OK);
        assert_eq!(json_body(busy_response).await["vram_estimate"], stable_vram);

        let queue = app
            .oneshot(Request::get("/api/queue").body(Body::empty()).unwrap())
            .await
            .unwrap();
        assert_eq!(queue.status(), StatusCode::OK);
        assert_eq!(
            json_body(queue).await["entries"].as_array().unwrap().len(),
            0
        );

        std::env::remove_var("MOLD_MODELS_DIR");
        let _ = std::fs::remove_dir_all(models_dir);
    }

    #[tokio::test]
    #[allow(clippy::await_holding_lock)]
    async fn chain_validation_surfaces_normalization_errors_without_queueing() {
        let response = {
            let models_dir = tempfile::tempdir().unwrap();
            let _models_root = EnvVarGuard::set("MOLD_MODELS_DIR", models_dir.path().as_os_str());
            populate_manifest_files(models_dir.path(), "ltx-2-19b-distilled:fp8");
            let app = app_empty();
            let mut request = route_chain_request();
            request.stages[0].frames = 10;

            app.oneshot(
                Request::post("/api/generate/chain/validate")
                    .header("content-type", "application/json")
                    .body(Body::from(serde_json::to_vec(&request).unwrap()))
                    .unwrap(),
            )
            .await
            .unwrap()
        };
        assert_eq!(response.status(), StatusCode::UNPROCESSABLE_ENTITY);
        let body = json_body(response).await;
        assert_eq!(body["code"], "VALIDATION_ERROR");
        assert!(
            body["error"].as_str().unwrap().contains("8k+1"),
            "response body: {body}"
        );
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
            queue_journal: Arc::new(crate::queue_journal::QueueJournal::disabled()),
            generation_cancel: Arc::new(crate::generation_cancel::CancelRegistry::new()),
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
        generate_body_for_model(prompt, "mock-model", width, height)
    }

    fn generate_body_for_model(prompt: &str, model: &str, width: u32, height: u32) -> String {
        format!(
            r#"{{"prompt":"{prompt}","model":"{model}","width":{width},"height":{height},"steps":4,"batch_size":1,"output_format":"png"}}"#
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
        let mut bytes = vec![
            0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A, // signature
            0x00, 0x00, 0x00, 0x0D, 0x49, 0x48, 0x44, 0x52, // IHDR length + type
            0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0x01, // 1x1
            0x08, 0x02, 0x00, 0x00, 0x00, 0x90, 0x77, 0x53, 0xDE, // bit depth, color, CRC
            0x00, 0x00, 0x00, 0x0C, 0x49, 0x44, 0x41, 0x54, // IDAT length + type
            0x08, 0xD7, 0x63, 0xF8, 0xCF, 0xC0, 0x00, 0x00, 0x00, 0x02, 0x00,
            0x01, // compressed
            0xE2, 0x21, 0xBC, 0x33, // IDAT CRC
            0x00, 0x00, 0x00, 0x00, 0x49, 0x45, 0x4E, 0x44, 0xAE, 0x42, 0x60, 0x82, // IEND
        ];
        // Gallery publication rejects undersized crash artifacts. PNG readers
        // ignore bytes after IEND, so pad this tiny valid fixture past the
        // ordinary gallery size floor without allocating a large image.
        bytes.resize(320, 0);
        bytes
    }

    fn minimal_mp4(marker: u8) -> Vec<u8> {
        let mut bytes = Vec::new();
        bytes.extend_from_slice(&[0x00, 0x00, 0x00, 0x20]);
        bytes.extend_from_slice(b"ftyp");
        bytes.extend_from_slice(b"isom\x00\x00\x02\x00");
        bytes.resize(8192, marker);
        bytes
    }

    fn minimal_webp() -> Vec<u8> {
        let image = image::RgbImage::from_fn(32, 32, |x, y| {
            image::Rgb([
                (x.wrapping_mul(17) ^ y.wrapping_mul(31)) as u8,
                (x.wrapping_mul(29) ^ y.wrapping_mul(13)) as u8,
                (x.wrapping_mul(7) ^ y.wrapping_mul(23)) as u8,
            ])
        });
        let mut bytes = Vec::new();
        image::DynamicImage::ImageRgb8(image)
            .write_to(
                &mut std::io::Cursor::new(&mut bytes),
                image::ImageFormat::WebP,
            )
            .expect("encode WebP fixture");
        bytes
    }

    fn output_metadata(prompt: &str) -> mold_core::OutputMetadata {
        mold_core::OutputMetadata {
            collection: None,
            tags: None,
            title: None,
            source_fit: None,
            guidance_overrides: None,
            sample_shift: None,
            distill_strength_high: None,
            distill_strength_low: None,
            job_id: None,
            prompt: prompt.into(),
            negative_prompt: None,
            original_prompt: None,
            prompt_transform: None,
            batch_id: None,
            batch_index: None,
            batch_count: None,
            output_mode: None,
            model: "test-model".into(),
            seed: 7,
            steps: 4,
            guidance: 1.0,
            width: 1,
            height: 1,
            generation_width: None,
            generation_height: None,
            strength: None,
            source_image_name: None,
            source_image_sha256: None,
            edit_image_sha256s: None,
            references: None,
            keyframes: None,
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
            extend_video_path: None,
            extend_overlap_frames: None,
            pipeline: None,
            pipeline_provenance_sha256: None,
            source_preprocessing: None,
            ic_lora_control: None,
            hdr_exr_dir: None,
            hdr_exr_full_float: false,
            retake_range: None,
            spatial_upscale: None,
            temporal_upscale: None,
            chain_job_id: None,
            chain: None,
            frames: None,
            fps: None,
            version: "test".into(),
        }
    }

    fn gallery_import_body(metadata: Option<&mold_core::OutputMetadata>, bytes: &[u8]) -> Vec<u8> {
        let fallback;
        let (metadata, metadata_synthetic) = match metadata {
            Some(metadata) => (metadata, false),
            None => {
                fallback = output_metadata("synthetic fallback");
                (&fallback, true)
            }
        };
        gallery_import_body_with_descriptor(metadata, metadata_synthetic, bytes)
    }

    fn gallery_import_body_with_descriptor(
        metadata: &mold_core::OutputMetadata,
        metadata_synthetic: bool,
        bytes: &[u8],
    ) -> Vec<u8> {
        let descriptor = serde_json::to_vec(&serde_json::json!({
            "metadata": metadata,
            "metadata_synthetic": metadata_synthetic,
        }))
        .unwrap();
        let mut body = Vec::with_capacity(12 + descriptor.len() + bytes.len());
        body.extend_from_slice(&(descriptor.len() as u32).to_be_bytes());
        body.extend_from_slice(&(bytes.len() as u64).to_be_bytes());
        body.extend_from_slice(&descriptor);
        body.extend_from_slice(bytes);
        body
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

    #[tokio::test]
    async fn ltx2_control_capabilities_return_only_the_effective_distilled_profile() {
        let app = app_empty();
        let response = app
            .clone()
            .oneshot(
                Request::get(
                    "/api/capabilities/ltx2-control-adapters?model=ltx-2-19b-distilled%3Afp8",
                )
                .body(Body::empty())
                .unwrap(),
            )
            .await
            .unwrap();
        let status = response.status();
        let body = json_body(response).await;
        assert_eq!(status, StatusCode::OK, "response body: {body}");
        let ids = body
            .as_array()
            .unwrap()
            .iter()
            .map(|entry| entry["id"].as_str().unwrap())
            .collect::<Vec<_>>();
        assert_eq!(ids, ["union", "pose", "detailer"]);

        let response = app
            .oneshot(
                Request::get("/api/capabilities/ltx2-control-adapters?model=ltx-2-19b-dev%3Afp8")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::UNPROCESSABLE_ENTITY);
        assert!(json_body(response).await["error"]
            .as_str()
            .unwrap()
            .contains("distilled"));
    }

    #[tokio::test]
    async fn ltx2_camera_capabilities_return_only_compatible_19b_presets() {
        let app = app_empty();
        for model in ["ltx-2-19b-distilled%3Afp8", "ltx-2-19b-dev%3Afp8"] {
            let response = app
                .clone()
                .oneshot(
                    Request::get(format!(
                        "/api/capabilities/ltx2-camera-controls?model={model}"
                    ))
                    .body(Body::empty())
                    .unwrap(),
                )
                .await
                .unwrap();
            let status = response.status();
            let body = json_body(response).await;
            assert_eq!(status, StatusCode::OK, "response body: {body}");
            let ids = body
                .as_array()
                .unwrap()
                .iter()
                .map(|entry| entry["id"].as_str().unwrap())
                .collect::<Vec<_>>();
            assert_eq!(
                ids,
                [
                    "dolly-in",
                    "dolly-left",
                    "dolly-out",
                    "dolly-right",
                    "jib-down",
                    "jib-up",
                    "static",
                ]
            );
            assert!(body
                .as_array()
                .unwrap()
                .iter()
                .all(|entry| entry["installed"] == false));
        }

        let response = app
            .oneshot(
                Request::get(
                    "/api/capabilities/ltx2-camera-controls?model=ltx-2.3-22b-distilled%3Afp8",
                )
                .body(Body::empty())
                .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::OK);
        assert_eq!(json_body(response).await, serde_json::json!([]));
    }

    /// `?detail=1` carries the server's own reason. Without it the response
    /// must stay a bare array, byte for byte, because desktop and iPhone talk
    /// to arbitrary-version remotes.
    #[tokio::test]
    async fn ltx2_camera_capabilities_detail_envelope_reports_why_presets_are_unavailable() {
        let app = app_empty();

        let response = app
            .clone()
            .oneshot(
                Request::get(
                    "/api/capabilities/ltx2-camera-controls?model=ltx-2.3-22b-distilled%3Afp8&detail=1",
                )
                .body(Body::empty())
                .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::OK);
        let body = json_body(response).await;
        assert_eq!(body["supported"], false);
        assert_eq!(body["controls"], serde_json::json!([]));
        assert!(
            body["unsupported_reason"]
                .as_str()
                .unwrap()
                .contains("LTX-2 19B only"),
            "response body: {body}"
        );

        let response = app
            .clone()
            .oneshot(
                Request::get(
                    "/api/capabilities/ltx2-camera-controls?model=ltx-2-19b-distilled%3Afp8&detail=1",
                )
                .body(Body::empty())
                .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::OK);
        let body = json_body(response).await;
        assert_eq!(body["supported"], true);
        assert!(body["unsupported_reason"].is_null());
        assert_eq!(body["controls"].as_array().unwrap().len(), 7);

        // Older clients keep the exact array they parse today.
        let response = app
            .oneshot(
                Request::get(
                    "/api/capabilities/ltx2-camera-controls?model=ltx-2.3-22b-distilled%3Afp8",
                )
                .body(Body::empty())
                .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(json_body(response).await, serde_json::json!([]));
    }

    /// An unknown architecture is "no presets here", not a client error. It
    /// used to 422, which every surface caught into an unexplained empty
    /// picker — the user saw nothing and was told nothing.
    #[tokio::test]
    async fn ltx2_camera_capabilities_detail_explains_an_unknown_architecture() {
        let mut config = mold_core::Config::default();
        config.models.insert(
            "mystery:fp8".to_string(),
            mold_core::config::ModelConfig {
                family: Some("ltx2".to_string()),
                transformer: Some("/models/mystery/weights.safetensors".to_string()),
                ..Default::default()
            },
        );
        let (tx, _rx) = tokio::sync::mpsc::channel(16);
        let queue = crate::state::QueueHandle::new(tx);
        let gpu_pool = std::sync::Arc::new(crate::gpu_pool::GpuPool {
            workers: Vec::new().into(),
        });
        let app = app_with_state(AppState::empty(config, queue, gpu_pool, 200));

        let response = app
            .oneshot(
                Request::get("/api/capabilities/ltx2-camera-controls?model=mystery%3Afp8&detail=1")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::OK);
        let body = json_body(response).await;
        assert_eq!(body["supported"], false);
        assert!(
            body["unsupported_reason"]
                .as_str()
                .unwrap()
                .contains("unknown"),
            "response body: {body}"
        );
    }

    #[tokio::test]
    #[allow(clippy::await_holding_lock)]
    async fn chain_validation_rejects_19b_camera_preset_on_ltx23_without_downloading() {
        let response = {
            let models_dir = tempfile::tempdir().unwrap();
            let _models_root = EnvVarGuard::set("MOLD_MODELS_DIR", models_dir.path().as_os_str());
            populate_manifest_files(models_dir.path(), "ltx-2.3-22b-distilled:fp8");
            app_empty()
                .oneshot(
                    Request::post("/api/generate/chain/validate")
                        .header("content-type", "application/json")
                        .body(Body::from(
                            serde_json::json!({
                                "model": "ltx-2.3-22b-distilled:fp8",
                                "stages": [
                                    {
                                        "prompt": "orbit the subject",
                                        "frames": 25,
                                        "loras": [
                                            {
                                                "path": "camera-control:dolly-left",
                                                "scale": 1.0,
                                                "name": "Dolly left"
                                            }
                                        ]
                                    }
                                ],
                                "motion_tail_frames": 17,
                                "width": 704,
                                "height": 416,
                                "fps": 24,
                                "steps": 8,
                                "guidance": 3.0,
                                "output_format": "mp4"
                            })
                            .to_string(),
                        ))
                        .unwrap(),
                )
                .await
                .unwrap()
        };
        assert_eq!(response.status(), StatusCode::UNPROCESSABLE_ENTITY);
        let body = json_body(response).await;
        assert!(
            body["error"]
                .as_str()
                .unwrap()
                .contains("published for LTX-2 19B only"),
            "response body: {body}"
        );
    }

    #[tokio::test]
    async fn invalid_built_in_control_pairing_is_rejected_before_media_or_queue_work() {
        let app = app_empty();
        let response = app
            .oneshot(
                Request::post("/api/generate")
                    .header("content-type", "application/json")
                    .body(Body::from(
                        serde_json::json!({
                            "prompt": "test",
                            "model": "ltx-2-19b-distilled:fp8",
                            "width": 960,
                            "height": 576,
                            "steps": 8,
                            "guidance": 1.0,
                            "batch_size": 1,
                            "output_format": "mp4",
                            "source_video_path": "/does/not/exist.mp4",
                            "ic_lora_control": "motion-track"
                        })
                        .to_string(),
                    ))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::UNPROCESSABLE_ENTITY);
        let body = json_body(response).await;
        assert!(body["error"]
            .as_str()
            .unwrap()
            .contains("not compatible with LTX-2 19B distilled"));
    }

    #[tokio::test]
    async fn guidance_overrides_are_rejected_for_non_ltx2_models() {
        let app = app_empty();
        let response = app
            .oneshot(
                Request::post("/api/generate")
                    .header("content-type", "application/json")
                    .body(Body::from(
                        serde_json::json!({
                            "prompt": "test",
                            "model": "flux-dev:q4",
                            "width": 512,
                            "height": 512,
                            "steps": 4,
                            "guidance": 3.0,
                            "batch_size": 1,
                            "guidance_overrides": { "stg_scale": 1.5 }
                        })
                        .to_string(),
                    ))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::UNPROCESSABLE_ENTITY);
        let body = json_body(response).await;
        let error = body["error"].as_str().unwrap();
        assert!(error.contains("guidance_overrides"), "got: {error}");
    }

    #[tokio::test]
    async fn guidance_overrides_out_of_range_are_rejected_before_queue_work() {
        let app = app_empty();
        let response = app
            .oneshot(
                Request::post("/api/generate")
                    .header("content-type", "application/json")
                    .body(Body::from(
                        serde_json::json!({
                            "prompt": "test",
                            "model": "ltx-2-19b-distilled:fp8",
                            "width": 960,
                            "height": 576,
                            "steps": 8,
                            "guidance": 1.0,
                            "batch_size": 1,
                            "output_format": "mp4",
                            "guidance_overrides": { "rescale_scale": 2.0 }
                        })
                        .to_string(),
                    ))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::UNPROCESSABLE_ENTITY);
        let body = json_body(response).await;
        let error = body["error"].as_str().unwrap();
        assert!(error.contains("rescale_scale"), "got: {error}");
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
                    stable_id: Some("cuda:00000000000000000000000000000000".into()),
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
            "cuda:00000000000000000000000000000000"
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
        assert!(registry.all_startup_allowed_devices_disabled());
        assert!(
            events.try_recv().is_err(),
            "the request response is authoritative; the coordinator owns semantic events"
        );

        let generation = app
            .clone()
            .oneshot(
                Request::post("/api/generate")
                    .header("content-type", "application/json")
                    .body(Body::from(generate_body("maintenance", 64, 64)))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(generation.status(), StatusCode::SERVICE_UNAVAILABLE);
        let generation = json_body(generation).await;
        assert_eq!(generation["code"], "GENERATION_UNAVAILABLE");
        assert!(
            generation["error"]
                .as_str()
                .unwrap()
                .contains("maintenance mode"),
            "disabling the last runtime device must enter maintenance mode"
        );

        let chain_body = serde_json::to_vec(&route_chain_request()).unwrap();
        let upscale_body = serde_json::to_vec(&serde_json::json!({
            "model": "does-not-exist",
            "image": "AQID",
            "output_format": "png"
        }))
        .unwrap();
        for (path, body) in [
            ("/api/generate/chain", chain_body.clone()),
            ("/api/generate/chain/stream", chain_body.clone()),
            ("/api/chain-jobs", chain_body),
            ("/api/upscale", upscale_body.clone()),
            ("/api/upscale/stream", upscale_body),
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
                "{path} must reject before generation work"
            );
            let body = json_body(response).await;
            assert_eq!(body["code"], "GENERATION_UNAVAILABLE", "{path}");
            assert!(
                body["error"].as_str().unwrap().contains("maintenance mode"),
                "{path} must report last-device maintenance"
            );
        }

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
        assert!(!registry.all_startup_allowed_devices_disabled());
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
        let db = mold_db::MetadataDb::open_in_memory().unwrap();
        let preferences = mold_db::DevicePreferences::new(&db);
        preferences.set(GPU_0, false).unwrap();
        preferences.set(GPU_1, false).unwrap();
        let registry = Arc::new(
            crate::device_registry::DeviceRegistry::from_runtime_inventory(
                selected.clone(),
                &selected,
                Arc::new(Some(db)),
            ),
        );

        assert!(
            registry.startup_worker_devices().is_empty(),
            "persisted-disabled devices must not construct startup GPU owners"
        );
        assert_eq!(
            registry
                .persisted_disabled_worker_devices()
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
                    registry: registry.clone(),
                    shared_pool: Arc::new(Mutex::new(
                        mold_inference::shared_pool::SharedPool::new(),
                    )),
                    fatal_cuda_error: Arc::new(AtomicBool::new(false)),
                    fatal_cuda_shutdown: Arc::new(tokio::sync::Notify::new()),
                    queue_journal: Arc::new(crate::queue_journal::QueueJournal::disabled()),
                    generation_cancel: Arc::new(crate::generation_cancel::CancelRegistry::new()),
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
        let registry = Arc::new(
            crate::device_registry::DeviceRegistry::from_runtime_inventory(
                vec![gpu.clone()],
                std::slice::from_ref(&gpu),
                Arc::new(None),
            ),
        );
        let (scheduler_tx, mut scheduler_rx) = tokio::sync::mpsc::unbounded_channel();
        let pool = Arc::new(crate::gpu_pool::GpuPool {
            workers: Vec::new().into(),
        });
        pool.workers
            .install_factory(
                crate::gpu_pool::WorkerFactory {
                    registry: registry.clone(),
                    shared_pool: Arc::new(Mutex::new(
                        mold_inference::shared_pool::SharedPool::new(),
                    )),
                    fatal_cuda_error: Arc::new(AtomicBool::new(false)),
                    fatal_cuda_shutdown: Arc::new(tokio::sync::Notify::new()),
                    queue_journal: Arc::new(crate::queue_journal::QueueJournal::disabled()),
                    generation_cancel: Arc::new(crate::generation_cancel::CancelRegistry::new()),
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
        state.device_registry = registry;
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
    async fn device_patch_leaves_semantic_event_publication_to_the_coordinator() {
        let worker = gpu_worker_stub(0);
        let mut state = AppState::with_engine(MockEngine::ready());
        state.gpu_pool = Arc::new(crate::gpu_pool::GpuPool {
            workers: vec![worker.clone()].into(),
        });
        install_worker_registry(&mut state);
        let (tx, _rx) = tokio::sync::mpsc::channel(1);
        state.scheduled_work = crate::scheduler::ScheduledWorkHandle::for_mode(
            tx,
            crate::dispatch_mode::DispatchMode::V2,
        );
        let id = worker.gpu.stable_id.as_deref().unwrap();
        let mut events = state.events.subscribe();

        let response = app_with_state(state)
            .oneshot(
                Request::patch(format!("/api/devices/{id}"))
                    .header("content-type", "application/json")
                    .body(Body::from(r#"{"enabled":false}"#))
                    .unwrap(),
            )
            .await
            .unwrap();

        assert_eq!(response.status(), StatusCode::OK);
        assert!(
            events.try_recv().is_err(),
            "the request response is authoritative; the coordinator owns semantic events"
        );
    }

    #[tokio::test]
    async fn device_lifecycle_is_unavailable_without_authoritative_v2_ownership() {
        use crate::dispatch_mode::DispatchMode;

        for (label, mode, authoritative, observes) in [
            ("legacy", DispatchMode::Legacy, false, false),
            ("observe", DispatchMode::Observe, false, true),
            ("maintenance", DispatchMode::V2, false, false),
        ] {
            let worker = gpu_worker_stub(0);
            let mut state = AppState::with_engine(MockEngine::ready());
            state.gpu_pool = Arc::new(crate::gpu_pool::GpuPool {
                workers: vec![worker.clone()].into(),
            });
            install_worker_registry(&mut state);
            state
                .device_registry
                .set_desired_enabled(worker.gpu.stable_id.as_deref().unwrap(), false)
                .unwrap();
            let (tx, _rx) = tokio::sync::mpsc::channel(1);
            state.scheduled_work = crate::scheduler::ScheduledWorkHandle::for_runtime(
                tx,
                mode,
                authoritative,
                observes,
            );
            let id = worker.gpu.stable_id.as_deref().unwrap();
            let app = app_with_state(state);

            let inventory = app
                .clone()
                .oneshot(Request::get("/api/devices").body(Body::empty()).unwrap())
                .await
                .unwrap();
            assert_eq!(inventory.status(), StatusCode::OK, "{label}");
            let inventory = json_body(inventory).await;
            assert_ne!(
                inventory["devices"][0]["admin_state"], "disabled",
                "{label} must not report a live worker disabled"
            );
            assert_eq!(
                inventory["devices"][0]["desired_enabled"], false,
                "{label} keeps the persisted preference visible"
            );
            assert_eq!(
                inventory["devices"][0]["schedulable"], false,
                "{label} must not rewrite registry routing eligibility"
            );
            assert_eq!(
                inventory["devices"][0]["unschedulable_reason"], "device_draining",
                "{label} must not clear the registry reason"
            );

            let patch = app
                .oneshot(
                    Request::patch(format!("/api/devices/{id}"))
                        .header("content-type", "application/json")
                        .body(Body::from(r#"{"enabled":false}"#))
                        .unwrap(),
                )
                .await
                .unwrap();
            assert_eq!(patch.status(), StatusCode::CONFLICT, "{label}");
            assert_eq!(
                json_body(patch).await["code"],
                "DEVICE_LIFECYCLE_MODE_CONFLICT",
                "{label}"
            );
        }
    }

    #[tokio::test]
    async fn legacy_device_projection_preserves_degraded_health_and_routing_exclusion() {
        let worker = gpu_worker_stub(0);
        worker.consecutive_failures.store(3, Ordering::SeqCst);
        *worker.degraded_until.write().unwrap() =
            Some(std::time::Instant::now() + Duration::from_secs(60));
        let mut state = AppState::with_engine(MockEngine::ready());
        state.gpu_pool = Arc::new(crate::gpu_pool::GpuPool {
            workers: vec![worker].into(),
        });
        install_worker_registry(&mut state);
        let (tx, _rx) = tokio::sync::mpsc::channel(1);
        state.scheduled_work = crate::scheduler::ScheduledWorkHandle::for_mode(
            tx,
            crate::dispatch_mode::DispatchMode::Legacy,
        );

        let response = app_with_state(state)
            .oneshot(Request::get("/api/devices").body(Body::empty()).unwrap())
            .await
            .unwrap();
        let inventory = json_body(response).await;

        assert_eq!(inventory["devices"][0]["admin_state"], "enabled");
        assert_eq!(inventory["devices"][0]["health"], "degraded");
        assert_eq!(inventory["devices"][0]["schedulable"], false);
        assert_eq!(
            inventory["devices"][0]["unschedulable_reason"],
            "device_degraded"
        );
    }

    #[tokio::test]
    async fn observe_device_projection_preserves_transient_unavailability() {
        let worker = gpu_worker_stub(0);
        let id = worker.gpu.stable_id.clone().unwrap();
        let mut state = AppState::with_engine(MockEngine::ready());
        state.gpu_pool = Arc::new(crate::gpu_pool::GpuPool {
            workers: vec![worker].into(),
        });
        install_worker_registry(&mut state);
        assert!(state.device_registry.mark_unavailable(&id));
        let (tx, _rx) = tokio::sync::mpsc::channel(1);
        state.scheduled_work = crate::scheduler::ScheduledWorkHandle::for_runtime(
            tx,
            crate::dispatch_mode::DispatchMode::Observe,
            false,
            true,
        );

        let response = app_with_state(state)
            .oneshot(Request::get("/api/devices").body(Body::empty()).unwrap())
            .await
            .unwrap();
        let inventory = json_body(response).await;

        assert_eq!(inventory["devices"][0]["admin_state"], "enabled");
        assert_eq!(inventory["devices"][0]["health"], "unavailable");
        assert_eq!(inventory["devices"][0]["schedulable"], false);
        assert_eq!(
            inventory["devices"][0]["unschedulable_reason"],
            "device_unavailable"
        );
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
                    stable_id: Some("cuda:00000000000000000000000000000000".into()),
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
                    "cuda:00000000000000000000000000000001",
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
            .clone()
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
    async fn activity_snapshot_exposes_only_server_owned_nonterminal_work() {
        let mut state = AppState::for_tests();
        let home = tempfile::tempdir().unwrap();
        let db = mold_db::MetadataDb::open_in_memory().unwrap();
        seed_chain_job(&db, home.path(), "sequence-c", ChainJobState::Running);
        state.metadata_db = Arc::new(Some(db));
        let (download_id, _, _) = state
            .downloads
            .enqueue_recipe(crate::downloads::RecipePayload {
                catalog_id: "cv:713".into(),
                files: vec![crate::downloads::OwnedRecipeFile {
                    url: "http://invalid.example/model.safetensors".into(),
                    dest: "model.safetensors".into(),
                    sha256: None,
                    size_bytes: Some(100),
                }],
                auth: mold_core::download::RecipeAuth::None,
            })
            .await
            .unwrap();
        state.job_registry.register("queued-a", "flux-dev:q4");
        state.job_registry.register("running-b", "sdxl:q8");
        state.job_registry.mark_running("running-b", Some(1));
        state.scheduled_work.set_queue_work_items_for_tests(vec![
            mold_core::QueueWorkItem {
                work_id: "running-b:child".into(),
                parent_id: "running-b".into(),
                work_kind: "generation".into(),
                activity_phase: mold_core::QueueActivityPhase::Dispatching,
                ..Default::default()
            },
            mold_core::QueueWorkItem {
                work_id: "expand:0".into(),
                parent_id: "expand-parent".into(),
                work_kind: "prompt_expand".into(),
                activity_phase: mold_core::QueueActivityPhase::Cpu,
                ..Default::default()
            },
            mold_core::QueueWorkItem {
                work_id: "expand:1".into(),
                parent_id: "expand-parent".into(),
                work_kind: "prompt_expand".into(),
                activity_phase: mold_core::QueueActivityPhase::Queued,
                ..Default::default()
            },
        ]);
        let registry = state.job_registry.clone();
        let instance_id = state.instance_id.as_ref().clone();
        let app = app_with_state(state);

        let response = app
            .clone()
            .oneshot(Request::get("/api/activity").body(Body::empty()).unwrap())
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::OK);
        let body = json_body(response).await;
        let items = body["items"].as_array().unwrap();
        assert_eq!(items.len(), 5);
        let item = |id: &str| items.iter().find(|item| item["id"] == id).unwrap();
        assert_eq!(item("queued-a")["kind"], "generation");
        assert_eq!(item("queued-a")["phase"], "queued");
        assert_eq!(item("queued-a")["can_cancel"], true);
        assert_eq!(item("running-b")["phase"], "loading");
        assert_eq!(item("running-b")["can_cancel"], true);
        assert_eq!(item("expand-parent")["kind"], "prompt_expand");
        assert_eq!(item("expand-parent")["phase"], "running");
        assert_eq!(item("sequence-c")["kind"], "sequence");
        assert_eq!(item("sequence-c")["phase"], "running");
        assert_eq!(item(&download_id)["kind"], "download");
        assert_eq!(item(&download_id)["phase"], "queued");
        assert_eq!(body["instance_id"], instance_id);
        assert!(body["observed_at_unix_ms"].as_u64().unwrap() > 0);

        registry.remove("queued-a");
        registry.remove("running-b");
        let after_generation_terminal = json_body(
            app.oneshot(Request::get("/api/activity").body(Body::empty()).unwrap())
                .await
                .unwrap(),
        )
        .await;
        let ids = after_generation_terminal["items"]
            .as_array()
            .unwrap()
            .iter()
            .map(|item| item["id"].as_str().unwrap())
            .collect::<Vec<_>>();
        assert!(!ids.contains(&"queued-a"));
        assert!(!ids.contains(&"running-b"));
        assert!(ids.contains(&"expand-parent"));
    }

    /// `/api/activity` must report the place in line, not the submission id.
    ///
    /// `queue_rank` is `Coordinator::synthetic_id`, a monotonic submission
    /// counter, so once V2 published a plan the route projected it as
    /// `position` and clients rendered "#1041 in line". The registry ordering
    /// that `GET /api/queue` reports is the single authority.
    #[tokio::test]
    async fn activity_positions_come_from_the_registry_not_the_synthetic_rank() {
        let state = AppState::for_tests();
        state.job_registry.register("queued-a", "flux-dev:q4");
        state.job_registry.register("queued-b", "flux-dev:q4");
        state.scheduled_work.set_queue_work_items_for_tests(vec![
            mold_core::QueueWorkItem {
                work_id: "queued-a:0".into(),
                parent_id: "queued-a".into(),
                work_kind: "generation".into(),
                activity_phase: mold_core::QueueActivityPhase::Queued,
                queue_rank: 1_041,
                ..Default::default()
            },
            mold_core::QueueWorkItem {
                work_id: "queued-b:0".into(),
                parent_id: "queued-b".into(),
                work_kind: "generation".into(),
                activity_phase: mold_core::QueueActivityPhase::Queued,
                queue_rank: 1_042,
                ..Default::default()
            },
        ]);
        let app = app_with_state(state);

        let body = json_body(
            app.oneshot(Request::get("/api/activity").body(Body::empty()).unwrap())
                .await
                .unwrap(),
        )
        .await;
        let items = body["items"].as_array().unwrap();
        let position = |id: &str| {
            items
                .iter()
                .find(|item| item["id"] == id)
                .unwrap_or_else(|| panic!("{id} present"))["position"]
                .clone()
        };
        assert_eq!(position("queued-a"), serde_json::json!(0));
        assert_eq!(position("queued-b"), serde_json::json!(1));
    }

    #[tokio::test]
    async fn activity_snapshot_marks_sequences_unavailable_without_metadata_db() {
        let app = app_with_state(AppState::for_tests());
        let response = app
            .oneshot(Request::get("/api/activity").body(Body::empty()).unwrap())
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::OK);
        let body = json_body(response).await;
        assert_eq!(body["unavailable_kinds"], serde_json::json!(["sequence"]));
    }

    #[tokio::test]
    async fn queue_v2_work_items_are_additive_and_legacy_patch_still_reorders_only_entries() {
        let state = AppState::for_tests();
        state.job_registry.register("ordinary-a", "flux");
        state.job_registry.register("ordinary-b", "flux");
        state
            .scheduled_work
            .set_queue_work_items_for_tests(vec![mold_core::QueueWorkItem {
                work_id: "chain:parent-1:attempt:4:stage:2".to_string(),
                parent_id: "parent-1".to_string(),
                work_kind: "chain_stage".to_string(),
                chain_stage: Some(2),
                planned_device_id: Some("cuda:test-gpu".to_string()),
                activity_phase: mold_core::QueueActivityPhase::Queued,
                ..mold_core::QueueWorkItem::default()
            }]);
        let app = app_with_state(state);

        let before = json_body(
            app.clone()
                .oneshot(Request::get("/api/queue").body(Body::empty()).unwrap())
                .await
                .unwrap(),
        )
        .await;
        assert_eq!(before["entries"].as_array().unwrap().len(), 2);
        assert_eq!(before["plan"]["plan_version"], 7);
        assert_eq!(before["plan"]["state_version"], 11);
        assert_eq!(before["plan"]["work_items"][0]["parent_id"], "parent-1");
        assert_eq!(before["plan"]["work_items"][0]["chain_stage"], 2);
        assert_eq!(before["plan"]["work_items"][0]["work_kind"], "chain_stage");
        assert_eq!(
            before["plan"]["work_items"][0]["planned_device_id"],
            "cuda:test-gpu"
        );

        let patch = app
            .clone()
            .oneshot(
                Request::patch("/api/queue/ordinary-b")
                    .header("content-type", "application/json")
                    .body(Body::from(r#"{"position":0}"#))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(patch.status(), StatusCode::OK);

        let after = json_body(
            app.oneshot(Request::get("/api/queue").body(Body::empty()).unwrap())
                .await
                .unwrap(),
        )
        .await;
        assert_eq!(after["entries"][0]["id"], "ordinary-b");
        assert_eq!(
            after["plan"]["work_items"][0]["work_id"],
            "chain:parent-1:attempt:4:stage:2"
        );
        assert_eq!(after["plan"], before["plan"]);
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
            .clone()
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
    async fn patch_queue_accepts_matching_pin_shapes_and_rejects_a_mismatch() {
        let worker = gpu_worker_stub(0);
        let mut state = AppState::with_engine_and_queue(MockEngine::ready()).0;
        state.gpu_pool = Arc::new(crate::gpu_pool::GpuPool {
            workers: vec![worker.clone()].into(),
        });
        install_worker_registry(&mut state);
        state.job_registry.register("aaaa", "flux-dev:fp16");
        let id = worker.gpu.stable_id.as_deref().unwrap();
        let app = app_with_state(state);

        let response = app
            .clone()
            .oneshot(
                Request::patch("/api/queue/aaaa")
                    .header("content-type", "application/json")
                    .body(Body::from(
                        serde_json::json!({ "hard_pinned_device_id": id }).to_string(),
                    ))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::OK);
        assert_eq!(json_body(response).await["target_gpu"], 0);

        let response = app
            .clone()
            .oneshot(
                Request::patch("/api/queue/aaaa")
                    .header("content-type", "application/json")
                    .body(Body::from(
                        serde_json::json!({
                            "hard_pinned_device_id": null,
                            "target_gpu": 0
                        })
                        .to_string(),
                    ))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::UNPROCESSABLE_ENTITY);

        let response = app
            .oneshot(
                Request::patch("/api/queue/aaaa")
                    .header("content-type", "application/json")
                    .body(Body::from(
                        serde_json::json!({
                            "hard_pinned_device_id": id,
                            "target_gpu": 0
                        })
                        .to_string(),
                    ))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::OK);
        assert_eq!(json_body(response).await["target_gpu"], 0);
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
    async fn delete_queue_running_job_revokes_inference_without_waiting_for_teardown() {
        let (state, _rx) = AppState::with_engine_and_queue(MockEngine::ready());
        state.job_registry.register("aaaa", "flux-dev:fp16");
        state.job_registry.mark_running("aaaa", Some(0));
        let attempt = mold_inference::InferenceCancellationToken::default();
        state
            .job_registry
            .install_running_cancellation("aaaa", attempt.clone());

        let app = app_with_state(state.clone());
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
        assert_eq!(
            state.job_registry.len(),
            1,
            "the worker owns cleanup after cooperative cancellation"
        );
        assert!(attempt.is_cancelled());
        assert!(state.job_registry.cancel_requested("aaaa"));

        let activity = app
            .oneshot(Request::get("/api/activity").body(Body::empty()).unwrap())
            .await
            .unwrap();
        let body = json_body(activity).await;
        let item = &body["items"][0];
        assert_eq!(item["phase"], "cancelling");
        assert_eq!(item["can_cancel"], false);
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
    async fn v2_pause_and_resume_wait_for_authoritative_plan_publication() {
        let (mut state, _rx) = AppState::with_engine_and_queue(MockEngine::ready());
        let (owner_tx, _owner_rx) = tokio::sync::mpsc::channel(1);
        let (control_tx, mut control_rx) = tokio::sync::mpsc::channel(1);
        let scheduled = crate::scheduler::ScheduledWorkHandle::for_runtime(
            owner_tx,
            crate::dispatch_mode::DispatchMode::V2,
            true,
            false,
        )
        .with_placement_preview(control_tx);
        state.scheduled_work = scheduled.clone();
        let queue_pause = state.queue_pause.clone();
        let app = app_with_state(state);

        let pause = {
            let app = app.clone();
            tokio::spawn(async move {
                app.oneshot(
                    Request::post("/api/queue/pause")
                        .body(Body::empty())
                        .unwrap(),
                )
                .await
                .unwrap()
            })
        };
        let pause_reply = match tokio::time::timeout(Duration::from_secs(1), control_rx.recv())
            .await
            .expect("pause must reach scheduler control")
            .expect("scheduler control channel must remain open")
        {
            crate::scheduler::PlacementPreviewQuery::SetQueuePaused {
                paused: true,
                reply_tx,
            } => reply_tx,
            _ => panic!("expected scheduler-owned pause control"),
        };
        assert!(
            !pause.is_finished(),
            "pause response must wait for authoritative plan publication"
        );
        assert!(queue_pause.pause());
        scheduled.set_queue_work_items_for_tests(vec![mold_core::QueueWorkItem {
            work_id: "paused-work".into(),
            parent_id: "paused-work".into(),
            blocked_reason: Some(mold_core::QueueBlockedReason::QueuePaused),
            activity_phase: mold_core::QueueActivityPhase::Blocked,
            ..Default::default()
        }]);
        pause_reply.send(Ok(true)).unwrap();
        let response = pause.await.unwrap();
        assert_eq!(response.status(), StatusCode::OK);
        let queue = json_body(
            app.clone()
                .oneshot(Request::get("/api/queue").body(Body::empty()).unwrap())
                .await
                .unwrap(),
        )
        .await;
        assert_eq!(
            queue["plan"]["work_items"][0]["blocked_reason"],
            "queue_paused"
        );

        let resume = {
            let app = app.clone();
            tokio::spawn(async move {
                app.oneshot(
                    Request::post("/api/queue/resume")
                        .body(Body::empty())
                        .unwrap(),
                )
                .await
                .unwrap()
            })
        };
        let resume_reply = match tokio::time::timeout(Duration::from_secs(1), control_rx.recv())
            .await
            .expect("resume must reach scheduler control")
            .expect("scheduler control channel must remain open")
        {
            crate::scheduler::PlacementPreviewQuery::SetQueuePaused {
                paused: false,
                reply_tx,
            } => reply_tx,
            _ => panic!("expected scheduler-owned resume control"),
        };
        assert!(
            !resume.is_finished(),
            "resume response must wait for authoritative plan publication"
        );
        assert!(queue_pause.resume());
        scheduled.set_queue_work_items_for_tests(vec![mold_core::QueueWorkItem {
            work_id: "queued-work".into(),
            parent_id: "queued-work".into(),
            activity_phase: mold_core::QueueActivityPhase::Queued,
            ..Default::default()
        }]);
        resume_reply.send(Ok(true)).unwrap();
        let response = resume.await.unwrap();
        assert_eq!(response.status(), StatusCode::OK);
        let queue = json_body(
            app.oneshot(Request::get("/api/queue").body(Body::empty()).unwrap())
                .await
                .unwrap(),
        )
        .await;
        assert_eq!(queue["plan"]["work_items"][0]["activity_phase"], "queued");
        assert_eq!(
            queue["plan"]["work_items"][0]["blocked_reason"],
            serde_json::Value::Null
        );
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
        // `server_batch` is intentionally gated by the process-global
        // `MOLD_OUTPUT_DIR` override. Serialize this positive assertion with
        // tests that temporarily disable output through that environment
        // variable.
        let _env = env_lock()
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
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
        assert_eq!(body["queue"]["server_batch"], true);
        assert_eq!(
            body["queue"]["server_batch_max_outputs"],
            crate::batch_runtime::MAX_LIVE_SERVER_BATCH_OUTPUTS
        );
        assert_eq!(body["devices"]["available"], true);
        assert_eq!(body["devices"]["lifecycle"], true);
        assert_eq!(body["devices"]["restart_enable"], false);
        assert_eq!(body["devices"]["stable_pins"], true);
        assert_eq!(body["devices"]["planned_lanes"], true);
        assert_eq!(body["devices"]["learned_eta"], true);
        assert_eq!(body["reference_uploads"]["available"], true);
        assert_eq!(body["reference_uploads"]["protocol_version"], 2);
        assert_eq!(body["reference_uploads"]["requires_api_key"], true);
        assert_eq!(
            body["reference_uploads"]["upload_handle_header"],
            crate::reference_uploads::UPLOAD_HANDLE_HEADER
        );
        assert_eq!(
            body["reference_uploads"]["max_file_bytes"],
            crate::reference_uploads::MAX_REFERENCE_UPLOAD_FILE_BYTES
        );
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
            assert_eq!(body["devices"]["stable_pins"], true, "{label}");
            assert_eq!(body["devices"]["planned_lanes"], false, "{label}");
            assert_eq!(body["devices"]["learned_eta"], false, "{label}");
            assert_eq!(body["queue"]["server_batch"], false, "{label}");
            assert!(
                body["queue"]["server_batch_max_outputs"].is_null(),
                "{label}"
            );
        }
    }

    #[tokio::test]
    async fn raw_batch_never_enters_legacy_single_result_worker() {
        let app = app_with(MockEngine::ready());
        let body = serde_json::json!({
            "prompt": "two cats",
            "model": "mock-model",
            "width": 64,
            "height": 64,
            "steps": 1,
            "batch_size": 2,
            "output_format": "png"
        });
        let response = app
            .oneshot(
                Request::post("/api/generate")
                    .header("content-type", "application/json")
                    .body(Body::from(body.to_string()))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::UNPROCESSABLE_ENTITY);
        let body = json_body(response).await;
        assert!(body["error"]
            .as_str()
            .unwrap()
            .contains("authoritative scheduler V2"));
    }

    /// The durable row is written before `submit()`, so a crash between
    /// admission and the worker still leaves something to replay.
    #[tokio::test]
    async fn an_admitted_gallery_bound_generation_is_journaled_before_it_is_queued() {
        let output_dir = tempfile::tempdir().unwrap();
        let (mut state, mut rx) = AppState::with_engine_and_queue(MockEngine::ready());
        state.output_disabled_override = false;
        state.config.write().await.output_dir =
            Some(output_dir.path().to_string_lossy().into_owned());
        state.metadata_db = Arc::new(Some(mold_db::MetadataDb::open_in_memory().unwrap()));
        let home = tempfile::tempdir().unwrap();
        state.queue_journal = Arc::new(crate::queue_journal::QueueJournal::new(
            state.metadata_db.clone(),
            Some(home.path()),
            "test-instance",
        ));
        let journal = state.queue_journal.clone();
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
        });

        let job = tokio::time::timeout(Duration::from_secs(5), rx.recv())
            .await
            .expect("the job must be queued")
            .expect("the queue channel must stay open");
        assert!(
            job.journal.is_some(),
            "a gallery-bound singleton must carry a durable queue ticket"
        );
        let rows = journal.list_all();
        assert_eq!(rows.len(), 1);
        assert_eq!(rows[0].id, job.id);
        assert_eq!(rows[0].output_dir, output_dir.path());

        gen_task.abort();
        let _ = gen_task.await;
    }

    /// No gallery target means the only delivery is the HTTP response, which
    /// by definition does not survive the restart. Replaying such a job would
    /// burn a full render whose result is discarded.
    #[tokio::test]
    async fn a_generation_with_no_gallery_target_is_not_journaled() {
        let (mut state, mut rx) = AppState::with_engine_and_queue(MockEngine::ready());
        state.metadata_db = Arc::new(Some(mold_db::MetadataDb::open_in_memory().unwrap()));
        let home = tempfile::tempdir().unwrap();
        state.queue_journal = Arc::new(crate::queue_journal::QueueJournal::new(
            state.metadata_db.clone(),
            Some(home.path()),
            "test-instance",
        ));
        let journal = state.queue_journal.clone();
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
        });

        let job = tokio::time::timeout(Duration::from_secs(5), rx.recv())
            .await
            .expect("the job must be queued")
            .expect("the queue channel must stay open");
        assert!(job.journal.is_none());
        assert!(journal.list_all().is_empty());

        gen_task.abort();
        let _ = gen_task.await;
    }

    /// Build a state whose journal is backed by `db` and whose gallery lands
    /// under `root` — the shape a durable generation is admitted under.
    ///
    /// `root` doubles as MOLD_HOME, so the queue identity's claim lock lives
    /// there. Calling this twice on one `root` is the restart case: the first
    /// state must be dropped first, exactly as a stopped server releases its
    /// claim, or the second boot mints a fresh identity and adopts nothing.
    fn durable_state(
        db: Arc<Option<mold_db::MetadataDb>>,
        root: &std::path::Path,
    ) -> (
        AppState,
        tokio::sync::mpsc::Receiver<crate::state::GenerationJob>,
    ) {
        let gallery = durable_gallery_dir(root);
        std::fs::create_dir_all(&gallery).unwrap();
        let (mut state, rx) = AppState::with_engine_and_queue(MockEngine::ready());
        state.output_disabled_override = false;
        state.metadata_db = db.clone();
        state.queue_journal = Arc::new(crate::queue_journal::QueueJournal::new(
            db,
            Some(root),
            "test-instance",
        ));
        state
            .config
            .try_write()
            .expect("fresh test config")
            .output_dir = Some(gallery.to_string_lossy().into_owned());
        (state, rx)
    }

    fn durable_gallery_dir(root: &std::path::Path) -> PathBuf {
        root.join("gallery")
    }

    /// The end-to-end shape: admit jobs, fence, drop the coordinator, rebuild
    /// `AppState` on the same DB, replay. The rows come back under their
    /// original ids, in submit order, through the ordinary queue.
    #[tokio::test]
    async fn retained_generations_replay_in_order_under_their_original_ids() {
        let output_dir = tempfile::tempdir().unwrap();
        let db = Arc::new(Some(mold_db::MetadataDb::open_in_memory().unwrap()));

        let submitted = {
            let (state, mut rx) = durable_state(db.clone(), output_dir.path());
            let app = app_with_state(state.clone());
            let mut submitted = Vec::new();
            // Held until the fence goes up: dropping a job before that is
            // ordinary completion, and deletes its row.
            let mut in_flight = Vec::new();
            for index in 0..3 {
                let app = app.clone();
                let task = tokio::spawn(async move {
                    app.oneshot(
                        Request::post("/api/generate")
                            .header("content-type", "application/json")
                            .body(Body::from(generate_body(
                                &format!("prompt {index}"),
                                512,
                                512,
                            )))
                            .unwrap(),
                    )
                    .await
                });
                let job = tokio::time::timeout(Duration::from_secs(5), rx.recv())
                    .await
                    .expect("submitted job")
                    .expect("open queue");
                submitted.push(job.id.clone());
                // The client goes away and the whole runtime is torn down.
                task.abort();
                let _ = task.await;
                // One of them was already claimed by a worker before the crash.
                if index == 0 {
                    assert_eq!(
                        job.journal.as_ref().unwrap().claim_dispatch(),
                        crate::queue_journal::DispatchClaim::Granted
                    );
                }
                in_flight.push(job);
            }
            state.queue_journal.retain_all();
            drop(in_flight);
            submitted
        };

        // A fresh server on the same database.
        let (state, mut rx) = durable_state(db.clone(), output_dir.path());
        let report = crate::queue_journal::replay(&state, true).await;

        assert_eq!(report.resumed, 3);
        assert_eq!(report.held, 0);
        assert_eq!(report.already_completed, 0);

        let mut replayed = Vec::new();
        for _ in 0..3 {
            let job = tokio::time::timeout(Duration::from_secs(5), rx.recv())
                .await
                .expect("replayed job")
                .expect("open queue");
            assert!(
                job.journal.is_some(),
                "a replayed job keeps owning its durable row"
            );
            assert!(
                job.progress_tx.is_none(),
                "a replayed job has no client to stream to"
            );
            replayed.push(job.id);
        }
        assert_eq!(replayed, submitted);
        assert_eq!(
            state
                .job_registry
                .snapshot()
                .entries
                .iter()
                .map(|entry| entry.id.clone())
                .collect::<Vec<_>>(),
            submitted,
            "replay re-registers under the original ids, so /api/queue and \
             /api/events see resumed jobs with no new event type"
        );
    }

    /// `PATCH /api/queue/:id` is authoritative over the lane and the order, so
    /// the durable row has to move with it. Otherwise a restart silently
    /// restores the admission-time lane — possibly the very GPU the user moved
    /// the job away from — and the original FIFO position.
    #[tokio::test]
    async fn a_reordered_and_relaned_job_replays_where_the_user_put_it() {
        let output_dir = tempfile::tempdir().unwrap();
        let db = Arc::new(Some(mold_db::MetadataDb::open_in_memory().unwrap()));

        let submitted = {
            let (state, mut rx) = durable_state(db.clone(), output_dir.path());
            let app = app_with_state(state.clone());
            let mut submitted = Vec::new();
            let mut in_flight = Vec::new();
            for index in 0..3 {
                let app = app.clone();
                let task = tokio::spawn(async move {
                    app.oneshot(
                        Request::post("/api/generate")
                            .header("content-type", "application/json")
                            .body(Body::from(generate_body(
                                &format!("prompt {index}"),
                                512,
                                512,
                            )))
                            .unwrap(),
                    )
                    .await
                });
                let job = tokio::time::timeout(Duration::from_secs(5), rx.recv())
                    .await
                    .expect("submitted job")
                    .expect("open queue");
                submitted.push(job.id.clone());
                task.abort();
                let _ = task.await;
                in_flight.push(job);
            }

            // Send the last job to the head of the line.
            let response = app_with_state(state.clone())
                .oneshot(
                    Request::patch(format!("/api/queue/{}", submitted[2]))
                        .header("content-type", "application/json")
                        .body(Body::from(r#"{"position":0}"#))
                        .unwrap(),
                )
                .await
                .unwrap();
            assert_eq!(response.status(), StatusCode::OK);

            let row = state
                .queue_journal
                .list_all()
                .into_iter()
                .find(|row| row.id == submitted[2])
                .expect("the reordered job keeps its durable row");
            assert_eq!(row.target_gpu, None, "admitted with no lane pin");

            state.queue_journal.retain_all();
            drop(in_flight);
            submitted
        };

        let (state, mut rx) = durable_state(db.clone(), output_dir.path());
        let report = crate::queue_journal::replay(&state, true).await;
        assert_eq!(report.resumed, 3);

        let mut replayed = Vec::new();
        for _ in 0..3 {
            let job = tokio::time::timeout(Duration::from_secs(5), rx.recv())
                .await
                .expect("replayed job")
                .expect("open queue");
            replayed.push(job.id);
        }
        assert_eq!(
            replayed,
            vec![
                submitted[2].clone(),
                submitted[0].clone(),
                submitted[1].clone()
            ],
            "replay must honour the reorder, not the admission order"
        );
    }

    /// The CPU-only worker is a separate publication path from the GPU one, so
    /// it needs the same two settlements: stamp the idempotence key before the
    /// save, and clear the row only once the output actually landed. Without
    /// the stamp, boot replay cannot recognise a print this path produced and
    /// re-renders it into a duplicate that no client-side dedupe can merge,
    /// because output filenames are wall-clock.
    #[tokio::test]
    async fn a_cpu_rendered_generation_stamps_its_job_id_and_clears_its_row() {
        let output_dir = tempfile::tempdir().unwrap();
        let db = Arc::new(Some(mold_db::MetadataDb::open_in_memory().unwrap()));
        let (state, rx) = durable_state(db.clone(), output_dir.path());
        let journal = state.queue_journal.clone();

        // The real CPU-only dispatch owner — `StartupMode::CpuFallback` spawns
        // exactly this.
        let worker = tokio::spawn(crate::queue::run_queue_worker(rx, state.clone()));

        let response = app_with_state(state.clone())
            .oneshot(
                Request::post("/api/generate")
                    .header("content-type", "application/json")
                    .body(Body::from(generate_body("a cat", 512, 512)))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::OK);

        assert!(
            journal.list_all().is_empty(),
            "a published generation clears its durable row"
        );

        // The saved print carries the queue job that produced it, which is
        // what makes replay idempotent.
        let saved: Vec<String> = db
            .as_ref()
            .as_ref()
            .unwrap()
            .with_conn(|conn| {
                let mut stmt = conn
                    .prepare("SELECT json_extract(metadata_json, '$.job_id') FROM generations")?;
                let rows = stmt.query_map([], |row| row.get::<_, Option<String>>(0))?;
                Ok(rows.filter_map(|row| row.ok().flatten()).collect())
            })
            .unwrap();
        assert_eq!(saved.len(), 1, "one print, carrying one job id: {saved:?}");
        assert!(!saved[0].is_empty());
        assert_eq!(
            mold_db::generation_queue::find_completed_job_ids(
                db.as_ref().as_ref().unwrap(),
                &saved
            )
            .unwrap()
            .len(),
            1,
            "replay's idempotence gate must recognise a CPU-rendered print"
        );

        worker.abort();
        let _ = worker.await;
    }

    /// A maintenance boot (`MOLD_GPUS=none`) has no dispatch owner at all, so
    /// there is nothing to replay INTO. Attempting it anyway sent every job
    /// into a dropped receiver, and the failed send dropped the ticket with a
    /// fresh boot's fence still down — deleting the whole queue on a routine
    /// maintenance restart, which is precisely what this feature exists to
    /// prevent.
    #[tokio::test]
    async fn a_boot_with_no_dispatch_owner_replays_nothing_and_keeps_every_row() {
        let output_dir = tempfile::tempdir().unwrap();
        let db = Arc::new(Some(mold_db::MetadataDb::open_in_memory().unwrap()));
        let submitted = seed_retained_jobs(db.clone(), output_dir.path(), 2).await;

        let (state, mut rx) = durable_state(db.clone(), output_dir.path());
        let report = crate::queue_journal::replay(&state, false).await;

        assert_eq!(report.resumed, 0);
        assert_eq!(report.held, 0);
        assert!(rx.try_recv().is_err());
        let rows = state.queue_journal.list_all();
        assert_eq!(
            rows.iter().map(|row| row.id.clone()).collect::<Vec<_>>(),
            submitted
        );
        assert!(
            rows.iter().all(|row| row.replay_seen == 0),
            "a boot that cannot replay must not spend the row's replay budget"
        );
    }

    /// The independent guard: whatever the reason a resubmission fails, the
    /// job never reached a worker, so its row must survive for the next boot
    /// rather than be deleted by the ticket's ordinary drop.
    #[tokio::test]
    async fn a_job_that_cannot_be_resubmitted_keeps_its_row() {
        let output_dir = tempfile::tempdir().unwrap();
        let db = Arc::new(Some(mold_db::MetadataDb::open_in_memory().unwrap()));
        let submitted = seed_retained_jobs(db.clone(), output_dir.path(), 2).await;

        let (state, rx) = durable_state(db.clone(), output_dir.path());
        // Exactly the maintenance shape: the dispatch owner is gone, so every
        // send fails — but we ask for a replay anyway.
        drop(rx);
        let report = crate::queue_journal::replay(&state, true).await;

        assert_eq!(report.resumed, 0);
        let rows = state.queue_journal.list_all();
        assert_eq!(
            rows.iter().map(|row| row.id.clone()).collect::<Vec<_>>(),
            submitted,
            "a job that never reached the queue must still be there to retry"
        );
        assert!(
            rows.iter()
                .all(|row| row.state == mold_db::generation_queue::QueueRowState::Queued),
            "it never ran, so it is not held — the replay budget bounds retries"
        );
    }

    /// Admit `count` durable jobs and retain them, as a crash would.
    async fn seed_retained_jobs(
        db: Arc<Option<mold_db::MetadataDb>>,
        output_dir: &std::path::Path,
        count: usize,
    ) -> Vec<String> {
        let (state, mut rx) = durable_state(db, output_dir);
        let app = app_with_state(state.clone());
        let mut submitted = Vec::new();
        let mut in_flight = Vec::new();
        for index in 0..count {
            let app = app.clone();
            let task = tokio::spawn(async move {
                app.oneshot(
                    Request::post("/api/generate")
                        .header("content-type", "application/json")
                        .body(Body::from(generate_body(
                            &format!("prompt {index}"),
                            512,
                            512,
                        )))
                        .unwrap(),
                )
                .await
            });
            let job = tokio::time::timeout(Duration::from_secs(5), rx.recv())
                .await
                .expect("submitted job")
                .expect("open queue");
            submitted.push(job.id.clone());
            task.abort();
            let _ = task.await;
            in_flight.push(job);
        }
        state.queue_journal.retain_all();
        drop(in_flight);
        submitted
    }

    /// A directory that is simply absent is not a directory that moved. The
    /// save helpers create it on demand, so holding every retained job because
    /// somebody tidied up the gallery — or a mount came back empty — parks
    /// work that would have run perfectly well.
    #[tokio::test]
    async fn an_absent_but_unchanged_gallery_is_recreated_rather_than_parking_every_job() {
        let output_dir = tempfile::tempdir().unwrap();
        let db = Arc::new(Some(mold_db::MetadataDb::open_in_memory().unwrap()));
        let submitted = seed_retained_jobs(db.clone(), output_dir.path(), 2).await;

        let (state, mut rx) = durable_state(db.clone(), output_dir.path());
        // The configured path is unchanged; the directory itself is gone.
        // Removed after the fixture builds, or it would just recreate it.
        std::fs::remove_dir_all(durable_gallery_dir(output_dir.path())).unwrap();
        let report = crate::queue_journal::replay(&state, true).await;

        assert_eq!(report.held, 0, "nothing should be parked");
        assert_eq!(report.resumed, 2);
        assert!(durable_gallery_dir(output_dir.path()).is_dir());
        let mut replayed = Vec::new();
        for _ in 0..2 {
            replayed.push(
                tokio::time::timeout(Duration::from_secs(5), rx.recv())
                    .await
                    .expect("replayed job")
                    .expect("open queue")
                    .id,
            );
        }
        assert_eq!(replayed, submitted);
    }

    /// A maintenance boot owes work it deliberately does not register, so
    /// without projecting the journal the operator sees an empty queue while
    /// the server is holding their jobs — exactly when they are most likely to
    /// be looking, and with no way to inspect or cancel them.
    #[tokio::test]
    async fn a_maintenance_boot_still_shows_the_jobs_it_owes() {
        let output_dir = tempfile::tempdir().unwrap();
        let db = Arc::new(Some(mold_db::MetadataDb::open_in_memory().unwrap()));
        let submitted = seed_retained_jobs(db.clone(), output_dir.path(), 2).await;

        let (state, _rx) = durable_state(db.clone(), output_dir.path());
        // No dispatch owner: replay returns without registering anything.
        let report = crate::queue_journal::replay(&state, false).await;
        assert_eq!(report.resumed, 0);
        assert!(state.job_registry.snapshot().entries.is_empty());

        let listing = json_body(
            app_with_state(state.clone())
                .oneshot(Request::get("/api/queue").body(Body::empty()).unwrap())
                .await
                .unwrap(),
        )
        .await;
        let entries = listing["entries"].as_array().unwrap();
        assert_eq!(
            entries
                .iter()
                .map(|entry| entry["id"].as_str().unwrap().to_string())
                .collect::<Vec<_>>(),
            submitted,
            "retained work must be visible even when this boot cannot run it"
        );
        for entry in entries {
            assert_eq!(entry["state"], "queued");
            assert_eq!(entry["durable"], true);
        }

        // And it can be cancelled, which is the other thing an operator needs.
        let response = app_with_state(state.clone())
            .oneshot(
                Request::delete(format!("/api/queue/{}", submitted[0]))
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::NO_CONTENT);
        assert_eq!(state.queue_journal.list_all().len(), 1);
    }

    /// A failure to CHECK the idempotence gate is not the same as "nothing was
    /// completed". Reading it as an empty result would re-render every job
    /// whose print already exists, and those duplicates are unmergeable
    /// because output filenames are wall-clock — so one malformed
    /// `metadata_json` would defeat the whole guarantee.
    #[tokio::test]
    async fn replay_renders_nothing_when_the_idempotence_gate_cannot_be_checked() {
        let output_dir = tempfile::tempdir().unwrap();
        let db = Arc::new(Some(mold_db::MetadataDb::open_in_memory().unwrap()));
        let submitted = seed_retained_jobs(db.clone(), output_dir.path(), 2).await;

        let (state, mut rx) = durable_state(db.clone(), output_dir.path());
        state.queue_journal.fail_completion_lookup_for_tests();
        let report = crate::queue_journal::replay(&state, true).await;

        assert_eq!(report.resumed, 0, "nothing may be rendered unverified");
        assert_eq!(report.skipped_unverified, 2);
        assert!(rx.try_recv().is_err());

        let rows = state.queue_journal.list_all();
        assert_eq!(
            rows.iter().map(|row| row.id.clone()).collect::<Vec<_>>(),
            submitted
        );
        assert!(
            rows.iter().all(|row| row.replay_seen == 0),
            "the next boot must get a full budget to try again"
        );
    }

    /// A job that finished between its last save and the crash must not be
    /// re-rendered — output filenames are wall-clock, so a duplicate print
    /// could never be merged afterwards.
    #[tokio::test]
    async fn a_retained_generation_whose_print_already_exists_is_never_replayed() {
        let output_dir = tempfile::tempdir().unwrap();
        let db = Arc::new(Some(mold_db::MetadataDb::open_in_memory().unwrap()));

        let finished_id = {
            let (state, mut rx) = durable_state(db.clone(), output_dir.path());
            let app = app_with_state(state.clone());
            let task = tokio::spawn(async move {
                app.oneshot(
                    Request::post("/api/generate")
                        .header("content-type", "application/json")
                        .body(Body::from(generate_body("a cat", 512, 512)))
                        .unwrap(),
                )
                .await
            });
            let job = tokio::time::timeout(Duration::from_secs(5), rx.recv())
                .await
                .expect("submitted job")
                .expect("open queue");
            task.abort();
            let _ = task.await;
            state.queue_journal.retain_all();
            job.id.clone()
        };

        // The print landed; only the journal delete was lost.
        db.as_ref()
            .as_ref()
            .unwrap()
            .with_conn(|conn| {
                conn.execute(
                    &format!(
                        "INSERT INTO generations
                            (filename, output_dir, created_at_ms, format, model, metadata_json)
                         VALUES ('done.png', '/gallery', 1, 'png', 'mock-model',
                                 '{{\"job_id\":\"{finished_id}\"}}')"
                    ),
                    [],
                )?;
                Ok(())
            })
            .unwrap();

        let (state, mut rx) = durable_state(db.clone(), output_dir.path());
        let report = crate::queue_journal::replay(&state, true).await;

        assert_eq!(report.already_completed, 1);
        assert_eq!(report.resumed, 0);
        assert!(rx.try_recv().is_err(), "nothing may be resubmitted");
        assert!(state.queue_journal.list_all().is_empty());
    }

    /// Fail closed: a request this build cannot read is parked for inspection,
    /// never guessed at and never silently dropped.
    #[tokio::test]
    async fn a_retained_generation_with_an_unreadable_request_is_held() {
        let output_dir = tempfile::tempdir().unwrap();
        let db = Arc::new(Some(mold_db::MetadataDb::open_in_memory().unwrap()));
        let (state, mut rx) = durable_state(db.clone(), output_dir.path());
        // The identity is claimed at boot, so the row has to be written under
        // whatever this server took.
        let owner = state.queue_journal.owner_uuid().unwrap().to_string();
        mold_db::generation_queue::insert(
            db.as_ref().as_ref().unwrap(),
            &mold_db::generation_queue::GenerationQueueRow {
                id: "unreadable".to_string(),
                owner_uuid: owner,
                state: mold_db::generation_queue::QueueRowState::Queued,
                model: "mock-model".to_string(),
                request_json: "{\"prompt\":".to_string(),
                output_dir: output_dir.path().to_path_buf(),
                target_gpu: None,
                target_device_id: None,
                completion_payload: "full".to_string(),
                seed_pinned: false,
                dispatch_attempts: 0,
                replay_seen: 0,
                held_reason: None,
                created_at_ms: 1,
                updated_at_ms: 1,
                started_at_ms: None,
            },
        )
        .unwrap();

        let report = crate::queue_journal::replay(&state, true).await;

        assert_eq!(report.held, 1);
        assert_eq!(report.resumed, 0);
        assert!(rx.try_recv().is_err());
        let row = state.queue_journal.list_all().pop().unwrap();
        assert_eq!(row.state, mold_db::generation_queue::QueueRowState::Held);
        assert!(row.held_reason.is_some());
    }

    /// Another installation sharing this MOLD_HOME owns its own rows.
    #[tokio::test]
    async fn a_foreign_owners_rows_are_never_replayed() {
        let output_dir = tempfile::tempdir().unwrap();
        let db = Arc::new(Some(mold_db::MetadataDb::open_in_memory().unwrap()));
        mold_db::generation_queue::insert(
            db.as_ref().as_ref().unwrap(),
            &mold_db::generation_queue::GenerationQueueRow {
                id: "theirs".to_string(),
                owner_uuid: "some-other-installation".to_string(),
                state: mold_db::generation_queue::QueueRowState::Queued,
                model: "mock-model".to_string(),
                request_json: r#"{"prompt":"a cat","model":"mock-model","width":512,"height":512,"steps":4,"guidance":3.5}"#
                    .to_string(),
                output_dir: output_dir.path().to_path_buf(),
                target_gpu: None,
                target_device_id: None,
                completion_payload: "full".to_string(),
                seed_pinned: false,
                dispatch_attempts: 0,
                replay_seen: 0,
                held_reason: None,
                created_at_ms: 1,
                updated_at_ms: 1,
                started_at_ms: None,
            },
        )
        .unwrap();

        let (state, mut rx) = durable_state(db.clone(), output_dir.path());
        let report = crate::queue_journal::replay(&state, true).await;

        assert_eq!(report.resumed, 0);
        assert!(rx.try_recv().is_err());
        assert!(state.queue_journal.list_all().is_empty());
    }

    /// `durable_queue` promises that a queued job survives a restart. A host
    /// with server gallery output disabled cannot promise that for ANY job —
    /// the only delivery is the HTTP response — so advertising the capability
    /// there would make every job on it a silent over-promise, not an edge
    /// case. Clients read the capability to decide whether to keep polling a
    /// job whose stream died.
    #[tokio::test]
    async fn a_host_with_no_gallery_output_does_not_promise_a_durable_queue() {
        let db = Arc::new(Some(mold_db::MetadataDb::open_in_memory().unwrap()));

        let (mut state, _rx) = AppState::with_engine_and_queue(MockEngine::ready());
        state.metadata_db = db.clone();
        let home = tempfile::tempdir().unwrap();
        state.queue_journal = Arc::new(crate::queue_journal::QueueJournal::new(
            db.clone(),
            Some(home.path()),
            "test-instance",
        ));
        assert!(
            state.queue_journal.is_enabled(),
            "the journal itself is available; only output is off"
        );
        let capabilities = json_body(
            app_with_state(state)
                .oneshot(
                    Request::get("/api/capabilities")
                        .body(Body::empty())
                        .unwrap(),
                )
                .await
                .unwrap(),
        )
        .await;
        assert_eq!(capabilities["queue"]["durable_queue"], false);

        let output_dir = tempfile::tempdir().unwrap();
        let (state, _rx) = durable_state(db, output_dir.path());
        let capabilities = json_body(
            app_with_state(state)
                .oneshot(
                    Request::get("/api/capabilities")
                        .body(Body::empty())
                        .unwrap(),
                )
                .await
                .unwrap(),
        )
        .await;
        assert_eq!(capabilities["queue"]["durable_queue"], true);
    }

    /// `durable_queue` is a promise about this host, and a held row is
    /// something only the journal knows about — invisible work that is never
    /// going to run is worse than work that failed.
    #[tokio::test]
    async fn the_queue_listing_reports_durability_and_surfaces_held_rows() {
        let output_dir = tempfile::tempdir().unwrap();
        let db = Arc::new(Some(mold_db::MetadataDb::open_in_memory().unwrap()));
        let (state, mut rx) = durable_state(db.clone(), output_dir.path());
        let app = app_with_state(state.clone());

        let capabilities = json_body(
            app.clone()
                .oneshot(
                    Request::get("/api/capabilities")
                        .body(Body::empty())
                        .unwrap(),
                )
                .await
                .unwrap(),
        )
        .await;
        assert_eq!(capabilities["queue"]["durable_queue"], true);
        assert_eq!(capabilities["queue"]["cooperative_cancellation"], true);

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
        });
        let job = tokio::time::timeout(Duration::from_secs(5), rx.recv())
            .await
            .expect("submitted job")
            .expect("open queue");

        let listing = json_body(
            app.clone()
                .oneshot(Request::get("/api/queue").body(Body::empty()).unwrap())
                .await
                .unwrap(),
        )
        .await;
        let live = &listing["entries"][0];
        assert_eq!(live["id"], serde_json::json!(job.id));
        assert_eq!(live["durable"], true);
        assert_eq!(live["replayed"], false);
        assert_eq!(live["dispatch_attempts"], 0);

        // Park it the way an exhausted attempt cap would.
        state
            .queue_journal
            .hold_id(&job.id, "dispatch attempts exhausted");
        gen_task.abort();
        let _ = gen_task.await;
        state.job_registry.remove(&job.id);

        let listing = json_body(
            app.oneshot(Request::get("/api/queue").body(Body::empty()).unwrap())
                .await
                .unwrap(),
        )
        .await;
        let held = &listing["entries"][0];
        assert_eq!(held["id"], serde_json::json!(job.id));
        assert_eq!(held["state"], "held");
        assert_eq!(held["held_reason"], "dispatch attempts exhausted");

        // A held job has no registry entry, so the documented way to clear one
        // has to reach the journal directly or the row is unreachable short of
        // editing the database.
        let response = app_with_state(state.clone())
            .oneshot(
                Request::delete(format!("/api/queue/{}", job.id))
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::NO_CONTENT);
        assert!(state.queue_journal.list_all().is_empty());

        let response = app_with_state(state.clone())
            .oneshot(
                Request::delete("/api/queue/never-existed")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(
            response.status(),
            StatusCode::NOT_FOUND,
            "an unknown id is still a 404"
        );
    }

    /// Once the retention fence is up the process is tearing down, so a new
    /// request is refused with a retryable 503 rather than admitted into a
    /// queue that immediately retains it.
    #[tokio::test]
    async fn a_restarting_host_refuses_new_generations_with_a_retry_hint() {
        let (state, _rx) = AppState::with_engine_and_queue(MockEngine::ready());
        state.queue_journal.retain_all();
        let app = app_with_state(state);

        for route in ["/api/generate", "/api/generate/stream"] {
            let response = app
                .clone()
                .oneshot(
                    Request::post(route)
                        .header("content-type", "application/json")
                        .body(Body::from(generate_body("a cat", 512, 512)))
                        .unwrap(),
                )
                .await
                .unwrap();
            assert_eq!(
                response.status(),
                StatusCode::SERVICE_UNAVAILABLE,
                "{route}"
            );
            assert_eq!(
                response
                    .headers()
                    .get(axum::http::header::RETRY_AFTER)
                    .and_then(|value| value.to_str().ok()),
                Some("1"),
                "{route}"
            );
            assert_eq!(json_body(response).await["code"], "SERVER_RESTARTING");
        }
    }

    #[tokio::test]
    async fn oversized_raw_batch_is_rejected_before_preparation_or_reservation() {
        let output_dir = tempfile::tempdir().unwrap();
        let (state, _rx) = AppState::with_engine_and_queue(MockEngine::ready());
        state.config.write().await.output_dir =
            Some(output_dir.path().to_string_lossy().into_owned());
        let registry = state.job_registry.clone();
        let app = app_with_state(state);
        let body = serde_json::json!({
            // Invalid downstream fields deliberately prove batch admission
            // runs before ordinary request preparation/model resolution.
            "prompt": "",
            "model": "must-not-be-resolved",
            "width": 64,
            "height": 64,
            "steps": 1,
            "batch_size": u32::MAX,
            "output_format": "png"
        });

        for route in ["/api/generate", "/api/generate/stream"] {
            let response = tokio::time::timeout(
                Duration::from_secs(5),
                app.clone().oneshot(
                    Request::post(route)
                        .header("content-type", "application/json")
                        .body(Body::from(body.to_string()))
                        .unwrap(),
                ),
            )
            .await
            .expect("u32::MAX batch admission must be bounded-time")
            .unwrap();
            assert_eq!(response.status(), StatusCode::UNPROCESSABLE_ENTITY);
            let body = json_body(response).await;
            assert_eq!(body["code"], "BATCH_OUTPUT_LIMIT_EXCEEDED");
            assert_eq!(
                body["error"],
                format!(
                    "batch_size ({}) exceeds the live server batch output limit ({})",
                    u32::MAX,
                    crate::batch_runtime::MAX_LIVE_SERVER_BATCH_OUTPUTS
                )
            );
        }

        assert!(
            registry.snapshot().entries.is_empty(),
            "rejected batch must not register a parent or enumerate children"
        );
        assert_eq!(
            std::fs::read_dir(output_dir.path()).unwrap().count(),
            0,
            "rejected batch must not reserve filenames or create transaction state"
        );
    }

    /// Clients feature-detect server-side catalog sorting against this
    /// advertisement — older servers omit the field entirely.
    #[tokio::test]
    async fn capabilities_reports_catalog_sort_vocabulary_without_h3_access_restriction() {
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
        assert!(body["catalog"]["families"]
            .as_array()
            .is_some_and(|families| families.iter().any(|family| family == "minimax-h3")));
        assert_eq!(body["model_access"]["restrictions"], serde_json::json!([]));
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

    /// Once a job is queued it runs, even if the client that asked for it
    /// goes away. The detached result supervisor owns `result_tx`'s receiver,
    /// so the fifteen `result_tx.is_closed()` dispatch gates keep reading
    /// `false` and the job still reaches a worker.
    #[tokio::test]
    async fn a_disconnected_blocking_generate_still_reaches_the_worker() {
        let (state, mut rx) = AppState::with_engine_and_queue(MockEngine::ready());
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
        });

        wait_for_registered_job(&state).await;
        // The client hung up: axum drops the handler future mid-await.
        gen_task.abort();
        let _ = gen_task.await;
        tokio::task::yield_now().await;

        let job = tokio::time::timeout(Duration::from_secs(5), rx.recv())
            .await
            .expect("the queued job must still be dispatchable")
            .expect("the queue channel must stay open");
        assert!(
            !job.result_tx.is_closed(),
            "a disconnected client must not make the worker skip an admitted job"
        );
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
        assert_eq!(
            find("scheduler.replan_debounce_ms")["restart_required"],
            true
        );
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
    async fn scheduler_config_is_validated_persisted_and_marked_restart_required() {
        let (app, db) = app_with_settings_db();
        let invalid = put_json(
            &app,
            "/api/config/scheduler.replan_max_delay_ms",
            r#"{"value":1999}"#,
        )
        .await;
        assert_eq!(invalid.status(), StatusCode::UNPROCESSABLE_ENTITY);

        let response = put_json(
            &app,
            "/api/config/scheduler.replan_debounce_ms",
            r#"{"value":1500}"#,
        )
        .await;
        assert_eq!(response.status(), StatusCode::OK);
        let body = json_body(response).await;
        assert_eq!(body["value"], 1500);
        assert_eq!(body["source"], "db");
        assert_eq!(body["restart_required"], true);

        let settings = mold_db::Settings::for_profile(db.as_ref().as_ref().unwrap(), "default");
        assert_eq!(
            settings
                .get_int(mold_db::settings::SCHEDULER_REPLAN_DEBOUNCE_MS)
                .unwrap(),
            Some(1500)
        );
        assert_eq!(
            settings
                .get_int(mold_db::settings::SCHEDULER_REPLAN_MAX_DELAY_MS)
                .unwrap(),
            Some(5000)
        );
        assert_eq!(
            settings
                .get_int(mold_db::settings::SCHEDULER_WARM_WAIT_MAX_MS)
                .unwrap(),
            Some(2000)
        );
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
    async fn config_put_output_dir_is_restart_only_and_changes_neither_runtime_nor_disk() {
        let tmp = tempfile::tempdir().unwrap();
        let _home = MoldHomeGuard::set(tmp.path());
        let boot_gallery = tmp.path().join("boot-gallery");
        let rejected_gallery = tmp.path().join("rejected-gallery");
        let config = mold_core::Config {
            output_dir: Some(boot_gallery.to_string_lossy().into_owned()),
            ..Default::default()
        };
        config.save_bootstrap_only().unwrap();

        let (tx, _rx) = tokio::sync::mpsc::channel(16);
        let state = AppState::empty(
            config,
            crate::state::QueueHandle::new(tx),
            AppState::empty_gpu_pool_for_test(),
            200,
        );
        let live_config = state.config.clone();
        let app = app_with_state(state);
        let response = put_json(
            &app,
            "/api/config/output_dir",
            &serde_json::json!({
                "value": rejected_gallery.to_string_lossy()
            })
            .to_string(),
        )
        .await;

        assert_eq!(response.status(), StatusCode::CONFLICT);
        let body = json_body(response).await;
        assert_eq!(body["code"], "RESTART_REQUIRED");
        assert!(body["error"]
            .as_str()
            .unwrap()
            .contains("mold config set output_dir"));
        assert_eq!(
            live_config.read().await.effective_output_dir(),
            boot_gallery,
            "the running server must stay on its boot gallery namespace"
        );
        let persisted = mold_core::Config::load_or_default();
        assert_eq!(
            persisted.effective_output_dir(),
            boot_gallery,
            "a rejected live PUT must not persist the next-start namespace"
        );
        assert!(
            !rejected_gallery.exists(),
            "a rejected namespace must not be touched"
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
    async fn list_models_surfaces_only_the_pinned_compact_h3_downloads() {
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
        let h3 = models
            .iter()
            .filter(|model| model["family"] == mold_core::minimax_h3::FAMILY)
            .map(|model| {
                (
                    model["name"].as_str().unwrap(),
                    model["downloaded"].as_bool().unwrap(),
                    model["hf_repo"].as_str().unwrap(),
                )
            })
            .collect::<std::collections::BTreeSet<_>>();
        assert_eq!(
            h3,
            std::collections::BTreeSet::from([
                (
                    mold_core::minimax_h3::FL2VA_COMFY,
                    false,
                    "Comfy-Org/MiniMax-H3",
                ),
                (
                    mold_core::minimax_h3::REF2VA_COMFY,
                    false,
                    "Comfy-Org/MiniMax-H3",
                ),
                (
                    mold_core::minimax_h3::FL2VA_COMFY_TURBO_8STEP,
                    false,
                    "Comfy-Org/MiniMax-H3",
                ),
                (
                    mold_core::minimax_h3::FL2VA_COMFY_TURBO_4STEP_768P,
                    false,
                    "Comfy-Org/MiniMax-H3",
                ),
            ])
        );
    }

    /// Sequence capability is advertised per model, so a picker never has to
    /// infer it from the checkpoint name. Every LTX-2 checkpoint qualifies,
    /// dev included — the old name heuristic hid dev checkpoints from the
    /// Sequence picker even though the server chains them.
    #[tokio::test]
    async fn list_models_advertise_per_model_sequence_support() {
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

        for name in [
            "ltx-2-19b-distilled:fp8",
            "ltx-2-19b-dev:fp8",
            "ltx-2.3-22b-dev:fp8",
        ] {
            let model = models
                .iter()
                .find(|m| m["name"] == name)
                .unwrap_or_else(|| panic!("{name} must be listed"));
            assert_eq!(
                model["supports_sequence"], true,
                "{name} must advertise sequence support"
            );
        }

        let still = models
            .iter()
            .find(|m| m["name"] == "sd15:fp16")
            .expect("sd15 must be listed");
        assert_eq!(
            still["supports_sequence"], false,
            "a still-image family must not advertise sequence support"
        );
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

        let (app, resources) = app_with_test_gpu_resources(48_000_000_000, 8_000_000_000);
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
        let capacity_peak = json["capacity_peak_memory_bytes"].as_u64().unwrap();
        assert_eq!(json["device_capacity_bytes"], 48_000_000_000_u64);
        assert_eq!(json["fits_device_capacity"], true);

        // Active work changes immediately free memory, but the hardware-fit
        // estimate must remain stable for an unchanged request and GPU.
        publish_test_gpu_resources(&resources, 48_000_000_000, 40_000_000_000);
        let busy_resp = app
            .clone()
            .oneshot(
                Request::post("/api/generate/estimate")
                    .header("content-type", "application/json")
                    .body(Body::from(body.to_string()))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(busy_resp.status(), StatusCode::OK);
        let busy_json = json_body(busy_resp).await;
        assert_eq!(busy_json["capacity_peak_memory_bytes"], capacity_peak);
        assert_eq!(busy_json["device_capacity_bytes"], 48_000_000_000_u64);
        assert_ne!(
            busy_json["available_memory_bytes"],
            json["available_memory_bytes"]
        );

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
    async fn generate_estimate_uses_only_eligible_and_explicitly_pinned_gpus() {
        let _lock = env_lock().lock().unwrap_or_else(|e| e.into_inner());
        let models_dir = test_models_dir("estimate-device-selection");
        populate_manifest_files(&models_dir, "sdxl-base:fp16");
        std::env::set_var("MOLD_MODELS_DIR", &models_dir);

        let request = serde_json::json!({
            "prompt": "a cat",
            "model": "sdxl-base:fp16",
            "width": 1024,
            "height": 1024,
            "steps": 20,
            "guidance": 7.5,
            "batch_size": 1,
            "output_format": "png"
        });
        let heterogeneous = [
            (0, 24_000_000_000, 2_000_000_000),
            (1, 80_000_000_000, 4_000_000_000),
        ];

        // A physically larger GPU that has been disabled is not a capacity
        // candidate for automatic routing.
        let disabled_app = app_with_test_gpu_snapshots(&heterogeneous, &[1]);
        let disabled_response = disabled_app
            .oneshot(
                Request::post("/api/generate/estimate")
                    .header("content-type", "application/json")
                    .body(Body::from(request.to_string()))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(disabled_response.status(), StatusCode::OK);
        let disabled_json = json_body(disabled_response).await;
        assert_eq!(disabled_json["device_capacity_bytes"], 24_000_000_000_u64);

        // When both devices are eligible, an explicit placement remains
        // authoritative instead of borrowing the roomier sibling's verdict.
        let pinned_app = app_with_test_gpu_snapshots(&heterogeneous, &[]);
        let mut pinned_request = request.clone();
        pinned_request["placement"] = serde_json::json!({
            "text_encoders": { "kind": "gpu", "ordinal": 0 }
        });
        let pinned_response = pinned_app
            .oneshot(
                Request::post("/api/generate/estimate")
                    .header("content-type", "application/json")
                    .body(Body::from(pinned_request.to_string()))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(pinned_response.status(), StatusCode::OK);
        let pinned_json = json_body(pinned_response).await;
        assert_eq!(pinned_json["device_capacity_bytes"], 24_000_000_000_u64);

        // Persisted/environment placement is the scheduler default when the
        // request omits placement, so the diagnostic must normalize through
        // the same Config authority before choosing a capacity candidate.
        let mut config = mold_core::Config::default();
        config.set_model_placement(
            "sdxl-base:fp16",
            Some(mold_core::DevicePlacement {
                text_encoders: mold_core::DeviceRef::gpu(0),
                advanced: None,
            }),
        );
        let configured_app = app_with_test_gpu_snapshots_and_config(&heterogeneous, &[], config);
        let configured_response = configured_app
            .oneshot(
                Request::post("/api/generate/estimate")
                    .header("content-type", "application/json")
                    .body(Body::from(request.to_string()))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(configured_response.status(), StatusCode::OK);
        let configured_json = json_body(configured_response).await;
        assert_eq!(configured_json["device_capacity_bytes"], 24_000_000_000_u64);

        std::env::remove_var("MOLD_MODELS_DIR");
        let _ = std::fs::remove_dir_all(models_dir);
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
    async fn generate_stream_qwen_edit_without_image_returns_actionable_422() {
        let app = app_with(MockEngine::ready());
        let body = serde_json::json!({
            "prompt": "make the coat blue",
            "model": "qwen-image-edit:q4",
            "width": 1024,
            "height": 1024,
            "steps": 4,
            "batch_size": 1,
            "output_format": "png"
        });
        let resp = app
            .oneshot(
                Request::post("/api/generate/stream")
                    .header("content-type", "application/json")
                    .body(Body::from(body.to_string()))
                    .unwrap(),
            )
            .await
            .unwrap();

        assert_eq!(resp.status(), StatusCode::UNPROCESSABLE_ENTITY);
        let body = json_body(resp).await;
        assert_eq!(body["code"], "VALIDATION_ERROR");
        assert_eq!(
            body["error"],
            "Qwen Image Edit needs at least one image. Add a Target image and try again."
        );
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
    async fn generate_rejects_client_supplied_hdr_exr_directory() {
        let app = app_with(MockEngine::ready());
        let body = serde_json::json!({
            "prompt": "an HDR sunset",
            "model": "mock-model",
            "width": 768,
            "height": 768,
            "steps": 4,
            "batch_size": 1,
            "output_format": "mp4",
            "hdr_exr_dir": "/tmp/client-chosen-server-path"
        });
        let resp = app
            .oneshot(
                Request::post("/api/generate")
                    .header("content-type", "application/json")
                    .body(Body::from(body.to_string()))
                    .unwrap(),
            )
            .await
            .unwrap();

        assert_eq!(resp.status(), StatusCode::UNPROCESSABLE_ENTITY);
        let body = json_body(resp).await;
        assert_eq!(body["code"], "VALIDATION_ERROR");
        assert!(
            body["error"]
                .as_str()
                .is_some_and(|error| error.contains("hdr_exr_dir") && error.contains("--local")),
            "got: {body}"
        );
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

    #[tokio::test]
    async fn batch_generate_response_uses_json_content_type() {
        let response = crate::routes::batch_generate_response(mold_core::BatchGenerateResponse {
            batch_id: "parent-1".to_string(),
            outputs: Vec::new(),
        });
        assert_eq!(
            response
                .headers()
                .get("content-type")
                .unwrap()
                .to_str()
                .unwrap(),
            "application/json"
        );
        let body = json_body(response).await;
        assert_eq!(body["batch_id"], "parent-1");
        assert_eq!(body["outputs"], serde_json::json!([]));
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

    #[tokio::test]
    async fn compact_h3_components_are_inspectable_before_runtime_qualification() {
        let app = app_empty();
        let resp = app
            .oneshot(
                Request::get("/api/models/minimax-h3-fl2va%3Acomfy-pruned-int8/components")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();

        assert_eq!(resp.status(), StatusCode::OK);
        let json = json_body(resp).await;
        assert_eq!(json["model"], mold_core::minimax_h3::FL2VA_COMFY);
        let components = json["components"].as_array().unwrap();
        assert!(!components.is_empty());
        assert!(components.iter().all(|component| {
            component["repair_model"] == mold_core::minimax_h3::FL2VA_COMFY
                && component["present"] == false
        }));
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
        assert!(
            spec["paths"]["/api/devices/{id}"]["patch"].is_object(),
            "spec should document PATCH /api/devices/{{id}}"
        );
        assert!(
            spec["paths"]["/api/generate/placement-preview"]["post"].is_object(),
            "spec should document POST /api/generate/placement-preview"
        );
        assert!(
            spec["paths"]["/api/generate/reference-upload-sessions"]["post"].is_object(),
            "spec should document reference upload session creation"
        );
        assert!(
            spec["paths"]["/api/generate/reference-upload-sessions"]["delete"].is_object(),
            "spec should document reference upload session cancellation"
        );
        assert!(
            spec["paths"]["/api/generate/reference-upload"]["put"].is_object(),
            "spec should document streaming reference upload"
        );
        assert!(
            spec["paths"]["/api/chain-jobs/placement-preview"]["post"].is_object(),
            "spec should document POST /api/chain-jobs/placement-preview"
        );
        assert!(
            spec["paths"]["/api/generate/chain/validate"]["post"].is_object(),
            "spec should document POST /api/generate/chain/validate"
        );
    }

    #[tokio::test]
    async fn generation_placement_preview_fails_closed_for_utility_stages() {
        let state = AppState::for_tests();
        let base = r#"{"prompt":"","model":"preview","width":512,"height":512,"steps":4,"guidance":1.0,"batch_size":1}"#;

        let mut expansion: GenerateRequest = serde_json::from_str(base).unwrap();
        expansion.expand = Some(true);
        let expansion_preview =
            crate::routes::placement_preview_for_request(&state, expansion, 2).await;
        assert!(!expansion_preview.authoritative);
        assert_eq!(expansion_preview.outcome, "unsupported");
        assert!(expansion_preview.candidate.is_none());
        assert!(expansion_preview.stage_candidates.is_empty());
        assert!(expansion_preview
            .reason
            .as_deref()
            .unwrap()
            .contains("utility CPU/GPU"));

        let mut upscale: GenerateRequest = serde_json::from_str(base).unwrap();
        upscale.upscale_model = Some("real-esrgan-x4plus:fp16".into());
        let upscale_preview =
            crate::routes::placement_preview_for_request(&state, upscale, 2).await;
        assert!(!upscale_preview.authoritative);
        assert_eq!(upscale_preview.outcome, "unsupported");
        assert!(upscale_preview.candidate.is_none());
        assert!(upscale_preview.stage_candidates.is_empty());
    }

    #[tokio::test]
    async fn generation_placement_preview_validates_copies_before_capability_fallback() {
        let state = AppState::for_tests();
        let request: GenerateRequest = serde_json::from_str(
            r#"{"prompt":"","model":"preview","width":512,"height":512,"steps":4,"guidance":1.0,"batch_size":1}"#,
        )
        .unwrap();

        let preview = crate::routes::placement_preview_for_request(&state, request, 0).await;

        assert!(preview.authoritative);
        assert_eq!(preview.outcome, "infeasible");
        assert!(preview.reason.unwrap().contains("between 1 and 64"));
    }

    #[tokio::test]
    async fn generation_placement_preview_is_authoritatively_infeasible_for_h3() {
        let state = AppState::for_tests();
        let request: GenerateRequest = serde_json::from_str(
            r#"{"prompt":"test","model":"hf:MiniMaxAI/MiniMax-H3","width":512,"height":512,"steps":4,"guidance":1.0,"batch_size":1}"#,
        )
        .unwrap();

        let preview = crate::routes::placement_preview_for_request(&state, request, 1).await;

        assert!(preview.authoritative);
        assert_eq!(preview.outcome, "infeasible");
        assert!(preview.candidate.is_none());
        assert!(preview
            .reason
            .unwrap()
            .contains(mold_core::MINIMAX_H3_AUTHORIZATION_REQUIRED));
    }

    #[cfg(feature = "h3-private-uat")]
    #[tokio::test]
    async fn private_h3_generation_requires_explicit_auth_before_queueing() {
        let (state, mut queue_rx) = AppState::with_engine_and_queue(MockEngine::ready());
        let response = app_with_state(state.clone())
            .oneshot(
                Request::post("/api/generate")
                    .header("content-type", "application/json")
                    .body(Body::from(private_h3_fl2va_body(1).to_string()))
                    .unwrap(),
            )
            .await
            .unwrap();

        assert_eq!(response.status(), StatusCode::UNAUTHORIZED);
        assert_eq!(json_body(response).await["code"], "UNAUTHORIZED");
        assert_eq!(state.job_registry.len(), 0);
        assert!(queue_rx.try_recv().is_err());
    }

    #[cfg(feature = "h3-private-uat")]
    #[tokio::test]
    async fn private_h3_wrong_partition_is_rejected_before_queueing() {
        let (state, mut queue_rx) = AppState::with_engine_and_queue(MockEngine::ready());
        let mut request = Request::post("/api/generate")
            .header("content-type", "application/json")
            .body(Body::from(private_h3_fl2va_body(2).to_string()))
            .unwrap();
        request
            .extensions_mut()
            .insert(crate::auth::ApiKeyAuthenticated {
                identity: "private-route-test".to_string(),
            });
        let response = app_with_state(state.clone())
            .oneshot(request)
            .await
            .unwrap();

        assert_eq!(response.status(), StatusCode::UNPROCESSABLE_ENTITY);
        assert_eq!(
            json_body(response).await["code"],
            crate::h3_private_bridge::H3_PRIVATE_PARTITION_REJECTED,
        );
        assert_eq!(state.job_registry.len(), 0);
        assert!(queue_rx.try_recv().is_err());
    }

    #[cfg(feature = "h3-private-uat")]
    #[tokio::test]
    async fn private_h3_placement_preview_requires_explicit_auth() {
        let state = AppState::for_tests();
        let response = app_with_state(state)
            .oneshot(
                Request::post("/api/generate/placement-preview")
                    .header("content-type", "application/json")
                    .body(Body::from(
                        serde_json::json!({
                            "request": private_h3_fl2va_body(1),
                            "copies": 1
                        })
                        .to_string(),
                    ))
                    .unwrap(),
            )
            .await
            .unwrap();

        assert_eq!(response.status(), StatusCode::UNAUTHORIZED);
        assert_eq!(json_body(response).await["code"], "UNAUTHORIZED");
    }

    #[tokio::test]
    async fn reviewed_reference_upload_session_requires_explicit_auth_before_staging() {
        let state = AppState::for_tests();
        let app = app_with_state(state.clone());
        let body = serde_json::json!({
            "request": {
                "prompt": "reference print",
                "model": mold_core::minimax_h3::REF2VA_COMFY,
                "width": mold_core::minimax_h3::DEFAULT_WIDTH,
                "height": mold_core::minimax_h3::DEFAULT_HEIGHT,
                "steps": mold_core::minimax_h3::DEFAULT_STEPS,
                "guidance": 0.0,
                "batch_size": 1,
                "frames": mold_core::minimax_h3::MIN_FRAMES,
                "fps": mold_core::minimax_h3::FIXED_FPS,
                "output_format": "mp4",
                "references": [{
                    "kind": "image",
                    "media": { "authority": "descriptor" },
                    "provenance": { "name": "anchor.png", "sha256": "11".repeat(32) },
                    "mime_type": "image/png",
                    "width": 1024,
                    "height": 1024
                }]
            },
            "upload_references": [1]
        })
        .to_string();

        let unauthenticated = app
            .clone()
            .oneshot(
                Request::post("/api/generate/reference-upload-sessions")
                    .header("content-type", "application/json")
                    .body(Body::from(body.clone()))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(unauthenticated.status(), StatusCode::UNAUTHORIZED);
        assert!(!state.reference_uploads.staging_exists());

        let mut request = Request::post("/api/generate/reference-upload-sessions")
            .header("content-type", "application/json")
            .body(Body::from(body))
            .unwrap();
        request
            .extensions_mut()
            .insert(crate::auth::ApiKeyAuthenticated {
                identity: "test-key".to_string(),
            });
        let created = app.oneshot(request).await.unwrap();
        assert_eq!(created.status(), StatusCode::OK);
        let created = json_body(created).await;
        assert_eq!(created["uploads"].as_array().unwrap().len(), 1);
        assert!(state.reference_uploads.staging_exists());
        state
            .reference_uploads
            .cancel_session("test-key", created["session_handle"].as_str().unwrap())
            .await
            .unwrap();
    }

    #[tokio::test]
    async fn config_only_h3_generation_is_rejected_before_queueing() {
        let (state, mut queue_rx) = AppState::with_engine_and_queue(MockEngine::ready());
        state.config.write().await.models.insert(
            "private-video-model".to_string(),
            mold_core::ModelConfig {
                family: Some("minimax-h3".to_string()),
                ..Default::default()
            },
        );
        let app = app_with_state(state.clone());
        let body = serde_json::json!({
            "prompt": "test",
            "model": "private-video-model",
            "width": 512,
            "height": 512,
            "steps": 4,
            "guidance": 1.0,
            "batch_size": 1
        });

        let response = app
            .oneshot(
                Request::post("/api/generate")
                    .header("content-type", "application/json")
                    .body(Body::from(body.to_string()))
                    .unwrap(),
            )
            .await
            .unwrap();

        assert_eq!(response.status(), StatusCode::UNAVAILABLE_FOR_LEGAL_REASONS);
        let body = json_body(response).await;
        assert_eq!(body["code"], mold_core::MINIMAX_H3_AUTHORIZATION_REQUIRED);
        assert_eq!(state.job_registry.len(), 0);
        assert!(
            queue_rx.try_recv().is_err(),
            "request must not reach the queue"
        );
    }

    #[tokio::test]
    async fn configured_h3_artifact_path_is_rejected_before_queueing() {
        let (state, mut queue_rx) = AppState::with_engine_and_queue(MockEngine::ready());
        state.config.write().await.models.insert(
            "renamed-private-model".to_string(),
            mold_core::ModelConfig {
                family: Some("flux".to_string()),
                transformer: Some("/models/MiniMax-H3/transformer.safetensors".to_string()),
                vae: Some("/models/ordinary-vae.safetensors".to_string()),
                ..Default::default()
            },
        );
        let app = app_with_state(state.clone());
        let body = serde_json::json!({
            "prompt": "test",
            "model": "renamed-private-model",
            "width": 512,
            "height": 512,
            "steps": 4,
            "guidance": 1.0,
            "batch_size": 1
        });

        let response = app
            .oneshot(
                Request::post("/api/generate")
                    .header("content-type", "application/json")
                    .body(Body::from(body.to_string()))
                    .unwrap(),
            )
            .await
            .unwrap();

        assert_eq!(response.status(), StatusCode::UNAVAILABLE_FOR_LEGAL_REASONS);
        let body = json_body(response).await;
        assert_eq!(body["code"], mold_core::MINIMAX_H3_AUTHORIZATION_REQUIRED);
        assert_eq!(state.job_registry.len(), 0);
        assert!(
            queue_rx.try_recv().is_err(),
            "configured artifact path must not reach the queue"
        );
    }

    #[tokio::test]
    async fn standalone_upscale_routes_reject_h3_before_pull_or_scheduling() {
        let state = AppState::for_tests();
        state.config.write().await.models.insert(
            "private-upscaler".to_string(),
            mold_core::ModelConfig {
                family: Some("minimax-h3".to_string()),
                transformer: Some("/models/ordinary-upscaler.safetensors".to_string()),
                ..Default::default()
            },
        );
        let app = app_with_state(state.clone());

        for model in ["MiniMaxH3Scheduler", "private-upscaler"] {
            let body = serde_json::json!({
                "model": model,
                "image": "AQID",
                "output_format": "png"
            })
            .to_string();
            for route in ["/api/upscale", "/api/upscale/stream"] {
                let response = app
                    .clone()
                    .oneshot(
                        Request::post(route)
                            .header("content-type", "application/json")
                            .body(Body::from(body.clone()))
                            .unwrap(),
                    )
                    .await
                    .unwrap();
                assert_eq!(
                    response.status(),
                    StatusCode::UNAVAILABLE_FOR_LEGAL_REASONS,
                    "{route} {model}"
                );
                let response = json_body(response).await;
                assert_eq!(
                    response["code"],
                    mold_core::MINIMAX_H3_AUTHORIZATION_REQUIRED,
                    "{route} {model}"
                );
            }
        }

        let listing = state.downloads.listing().await;
        assert!(listing.active.is_none());
        assert!(listing.queued.is_empty());
    }

    #[tokio::test]
    async fn config_only_h3_placement_preview_is_authoritatively_infeasible() {
        let state = AppState::for_tests();
        state.config.write().await.models.insert(
            "private-video-model".to_string(),
            mold_core::ModelConfig {
                family: Some("minimax-h3".to_string()),
                ..Default::default()
            },
        );
        let request: GenerateRequest = serde_json::from_str(
            r#"{"prompt":"test","model":"private-video-model","width":512,"height":512,"steps":4,"guidance":1.0,"batch_size":1}"#,
        )
        .unwrap();

        let preview = crate::routes::placement_preview_for_request(&state, request, 1).await;

        assert!(preview.authoritative);
        assert_eq!(preview.outcome, "infeasible");
        assert!(preview.candidate.is_none());
        assert!(preview
            .reason
            .unwrap()
            .contains(mold_core::MINIMAX_H3_AUTHORIZATION_REQUIRED));
    }

    #[tokio::test]
    async fn generation_placement_preview_names_missing_manifest_components() {
        let models = tempfile::tempdir().unwrap();
        let mut state = AppState::for_tests();
        state.config.write().await.models_dir = models.path().display().to_string();
        let (owner_tx, _owner_rx) = tokio::sync::mpsc::channel(1);
        let (preview_tx, _preview_rx) = tokio::sync::mpsc::channel(1);
        state.scheduled_work = crate::scheduler::ScheduledWorkHandle::for_runtime(
            owner_tx,
            crate::dispatch_mode::DispatchMode::V2,
            true,
            false,
        )
        .with_placement_preview(preview_tx);
        let request: GenerateRequest = serde_json::from_str(
            r#"{"prompt":"","model":"sd15:fp16","width":512,"height":512,"steps":4,"guidance":1.0,"batch_size":1}"#,
        )
        .unwrap();

        let preview = crate::routes::placement_preview_for_request(&state, request, 1).await;

        assert!(preview.authoritative);
        assert_eq!(preview.outcome, "infeasible");
        assert!(preview.candidate.is_none());
        let vae = preview
            .missing_components
            .iter()
            .find(|component| component.kind == "vae")
            .expect("missing manifest VAE must be named");
        assert!(!vae.present);
        assert_eq!(vae.name, "vae");
        assert_eq!(vae.repair_model.as_deref(), Some("sd15:fp16"));
    }

    /// #787 round 3: admission materializes wan's tuned default negative,
    /// which flips `request_sensitive_activation_memory`'s CFG factor to 2x
    /// for `guidance > 1` — an authoritative preview priced without the same
    /// materialization would promise a 1x plan admission cannot honor. This
    /// captures the exact request the scheduler is asked to price and pins it
    /// to the admission seam's output; the explicit `""` opt-out must pass
    /// through both paths untouched.
    #[tokio::test]
    async fn placement_preview_prices_the_same_wan_negative_as_admission() {
        let models = tempfile::tempdir().unwrap();
        let weights = models.path().join("studio-wan.safetensors");
        std::fs::write(&weights, b"stub").unwrap();
        let vae = models.path().join("studio-wan-vae.safetensors");
        std::fs::write(&vae, b"stub").unwrap();
        let (mut state, _rx) = AppState::with_engine_and_queue(MockEngine::ready());
        state.gpu_pool = Arc::new(crate::gpu_pool::GpuPool {
            workers: vec![gpu_worker_stub(0)].into(),
        });
        install_worker_registry(&mut state);
        let (owner_tx, _owner_rx) = tokio::sync::mpsc::channel(1);
        let (preview_tx, mut preview_rx) = tokio::sync::mpsc::channel(4);
        state.scheduled_work = crate::scheduler::ScheduledWorkHandle::for_runtime(
            owner_tx,
            crate::dispatch_mode::DispatchMode::V2,
            true,
            false,
        )
        .with_placement_preview(preview_tx);
        state.config.write().await.models.insert(
            "studio-wan".to_string(),
            mold_core::ModelConfig {
                family: Some("wan".to_string()),
                transformer: Some(weights.display().to_string()),
                vae: Some(vae.display().to_string()),
                ..Default::default()
            },
        );

        let base = serde_json::json!({
            "prompt": "a cat",
            "model": "studio-wan",
            "width": 256,
            "height": 256,
            "steps": 4,
            "guidance": 5.0,
            "batch_size": 1
        });
        for (sent_negative, expected) in [
            (
                None::<&str>,
                Some(mold_core::manifest::WAN_DEFAULT_NEGATIVE_PROMPT),
            ),
            (Some(""), Some("")),
            (Some("blurry"), Some("blurry")),
        ] {
            let mut request: GenerateRequest = serde_json::from_value(base.clone()).unwrap();
            request.negative_prompt = sent_negative.map(str::to_string);

            // Admission's own seam, for the parity half of the assertion.
            let mut admitted = request.clone();
            crate::routes::materialize_default_negative_prompt(&mut admitted, Some("wan"));

            let preview_call = {
                let state = state.clone();
                tokio::spawn(async move {
                    crate::routes::placement_preview_for_request(&state, request, 1).await
                })
            };
            let query = match tokio::time::timeout(Duration::from_secs(5), preview_rx.recv()).await
            {
                Ok(query) => query,
                Err(_) => {
                    let response = preview_call.await.unwrap();
                    panic!(
                        "preview never reached the scheduler pricing channel: \
                         outcome={} reason={:?}",
                        response.outcome, response.reason
                    );
                }
            };
            let priced = match query.expect("scheduler preview channel must remain open") {
                crate::scheduler::PlacementPreviewQuery::Generation {
                    request, reply_tx, ..
                } => {
                    let _ = reply_tx.send(mold_core::GenerationPlacementPreview {
                        version: 1,
                        authoritative: true,
                        state_version: 0,
                        plan_version: 0,
                        outcome: "planned".to_string(),
                        reason: None,
                        candidate: None,
                        stage_candidates: Vec::new(),
                        pending_downloads: Vec::new(),
                        missing_components: Vec::new(),
                    });
                    request
                }
                _ => panic!("expected a generation pricing query"),
            };
            let response = preview_call.await.unwrap();
            assert_eq!(response.outcome, "planned", "{sent_negative:?}");

            assert_eq!(
                priced.negative_prompt.as_deref(),
                expected,
                "preview must price the negative admission will materialize ({sent_negative:?})"
            );
            assert_eq!(
                priced.negative_prompt, admitted.negative_prompt,
                "preview and admission diverged for {sent_negative:?}"
            );
            // The CFG factor `request_sensitive_activation_memory` derives
            // from `guidance > 1 && negative_prompt.is_some()` must match.
            assert_eq!(
                priced.guidance > 1.0 && priced.negative_prompt.is_some(),
                admitted.guidance > 1.0 && admitted.negative_prompt.is_some(),
                "CFG pricing factor diverged for {sent_negative:?}"
            );
        }
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

    /// The first `queued` event must be on the same scale as every later one.
    ///
    /// It used to carry `Queue::submit`'s return — the pending count, which
    /// excludes the running job — while `/api/queue`, `/api/activity`, and the
    /// coordinator's re-announcements all report the registry index, which
    /// includes it. A job submitted behind one running generation was told 0
    /// and then re-announced as 1, so its place in line appeared to move
    /// backwards.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn the_first_queued_event_is_on_the_registry_scale() {
        let (state, rx) = AppState::with_engine_and_queue(MockEngine::ready());
        // One generation already occupies the machine, so the two scales
        // disagree: pending count 0, registry index 1.
        state
            .job_registry
            .register("already-running", "flux-dev:q4");
        state.job_registry.mark_running("already-running", Some(0));
        let worker_state = state.clone();
        let app = create_router(state);

        let response = app
            .oneshot(
                Request::post("/api/generate/stream")
                    .header("content-type", "application/json")
                    .body(Body::from(generate_body("a robot", 768, 768)))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::OK);
        // Keep the worker stopped until the handler has snapshotted the seed.
        // Under coverage instrumentation, starting it first can let the new
        // job reach its processing `position: 0` event before this assertion
        // observes the submit-time event this test is meant to pin.
        tokio::spawn(crate::queue::run_queue_worker(rx, worker_state));
        let body = axum::body::to_bytes(response.into_body(), 1024 * 1024)
            .await
            .unwrap();
        let text = String::from_utf8_lossy(&body);
        let first = sse_json_event(&text, "progress");
        assert_eq!(first["type"], "queued", "{text}");
        assert_eq!(
            first["position"], 1,
            "the seed must count the running job the registry counts"
        );
    }

    /// An idle host must still seed 0 — the CLI reads that as "no queue" and
    /// stays silent rather than announcing a position to a user who is first.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn an_idle_host_still_seeds_position_zero() {
        let app = app_with(MockEngine::ready());
        let response = app
            .oneshot(
                Request::post("/api/generate/stream")
                    .header("content-type", "application/json")
                    .body(Body::from(generate_body("a robot", 768, 768)))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::OK);
        let body = axum::body::to_bytes(response.into_body(), 1024 * 1024)
            .await
            .unwrap();
        let text = String::from_utf8_lossy(&body);
        let first = sse_json_event(&text, "progress");
        assert_eq!(first["type"], "queued", "{text}");
        assert_eq!(first["position"], 0);
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
        let (mut state, rx) = AppState::with_engine_and_queue(MockEngine::ready());
        state.output_disabled_override = false;
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
            queue_journal: Arc::new(crate::queue_journal::QueueJournal::disabled()),
            generation_cancel: Arc::new(crate::generation_cancel::CancelRegistry::new()),
            discovery: Arc::new(crate::state::DiscoveryState::default()),
            gpu_pool: std::sync::Arc::new(crate::gpu_pool::GpuPool {
                workers: Vec::new().into(),
            }),
            generation_unavailable_reason: Arc::new(std::sync::RwLock::new(None)),
            device_registry: crate::device_registry::DeviceRegistry::empty(),
            queue_capacity: 200,
            model_cache: Arc::new(tokio::sync::Mutex::new(cache)),
            active_generation: Arc::new(std::sync::RwLock::new(None)),
            config: Arc::new(tokio::sync::RwLock::new(AppState::test_config())),
            reference_uploads: crate::reference_uploads::ReferenceUploadStore::from_mold_home(),
            output_disabled_override: true,
            reload_config_from_disk: false,
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
            queue_journal: Arc::new(crate::queue_journal::QueueJournal::disabled()),
            generation_cancel: Arc::new(crate::generation_cancel::CancelRegistry::new()),
            discovery: Arc::new(crate::state::DiscoveryState::default()),
            gpu_pool: std::sync::Arc::new(crate::gpu_pool::GpuPool {
                workers: Vec::new().into(),
            }),
            generation_unavailable_reason: Arc::new(std::sync::RwLock::new(None)),
            device_registry: crate::device_registry::DeviceRegistry::empty(),
            queue_capacity: 200,
            model_cache: Arc::new(tokio::sync::Mutex::new(cache)),
            active_generation: Arc::new(std::sync::RwLock::new(None)),
            config: Arc::new(tokio::sync::RwLock::new(AppState::test_config())),
            reference_uploads: crate::reference_uploads::ReferenceUploadStore::from_mold_home(),
            output_disabled_override: true,
            reload_config_from_disk: false,
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
        let model = "sd15:fp16";
        let (state, rx) = AppState::with_engine_and_queue(MockEngine::ready_for_model(model));
        let queue = state.queue.clone();
        let worker_state = state.clone();
        let app = app_with_state(state);

        // The SSE route submits while its body is polled, so retain a task that
        // consumes the first response while waiting for the queue transition.
        let resp1 = app
            .clone()
            .oneshot(
                Request::post("/api/generate/stream")
                    .header("content-type", "application/json")
                    .body(Body::from(generate_body_for_model(
                        "first", model, 768, 768,
                    )))
                    .unwrap(),
            )
            .await
            .unwrap();
        if resp1.status() != StatusCode::OK {
            let status = resp1.status();
            let body = axum::body::to_bytes(resp1.into_body(), 1024 * 1024)
                .await
                .unwrap();
            panic!(
                "first streaming request returned {status}: {}",
                String::from_utf8_lossy(&body)
            );
        }
        let body1 =
            tokio::spawn(async move { axum::body::to_bytes(resp1.into_body(), 1024 * 1024).await });
        tokio::time::timeout(Duration::from_secs(10), async {
            while queue.pending() < 1 {
                tokio::task::yield_now().await;
            }
        })
        .await
        .expect("first streaming request should enter the queue");

        // Submit second request — should be queued at position 1
        let resp2 = app
            .oneshot(
                Request::post("/api/generate/stream")
                    .header("content-type", "application/json")
                    .body(Body::from(generate_body_for_model(
                        "second", model, 768, 768,
                    )))
                    .unwrap(),
            )
            .await
            .unwrap();
        let body2 =
            tokio::spawn(async move { axum::body::to_bytes(resp2.into_body(), 1024 * 1024).await });
        tokio::time::timeout(Duration::from_secs(10), async {
            while queue.pending() < 2 {
                tokio::task::yield_now().await;
            }
        })
        .await
        .expect("second streaming request should enter the queue");

        // Start the worker so both jobs are processed and the streams close.
        tokio::spawn(crate::queue::run_queue_worker(rx, worker_state));

        let body2 = tokio::time::timeout(Duration::from_secs(10), body2)
            .await
            .expect("second queued stream should close")
            .expect("second queued stream task should complete")
            .unwrap();
        let text2 = String::from_utf8_lossy(&body2).to_string();
        assert!(
            text2.contains(r#""type":"queued""#),
            "second request should receive a queued event, got: {text2}"
        );
        // The second request should report position > 0 (queued behind the first)
        assert!(
            text2.contains(r#""position":1"#),
            "second request should be at position 1, got: {text2}"
        );

        tokio::time::timeout(Duration::from_secs(10), body1)
            .await
            .expect("first queued stream should close")
            .expect("first queued stream task should complete")
            .unwrap();
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
            queue_journal: Arc::new(crate::queue_journal::QueueJournal::disabled()),
            generation_cancel: Arc::new(crate::generation_cancel::CancelRegistry::new()),
            discovery: Arc::new(crate::state::DiscoveryState::default()),
            gpu_pool: std::sync::Arc::new(crate::gpu_pool::GpuPool {
                workers: Vec::new().into(),
            }),
            generation_unavailable_reason: Arc::new(std::sync::RwLock::new(None)),
            device_registry: crate::device_registry::DeviceRegistry::empty(),
            queue_capacity: 200,
            model_cache: Arc::new(tokio::sync::Mutex::new(cache)),
            active_generation: Arc::new(std::sync::RwLock::new(None)),
            config: Arc::new(tokio::sync::RwLock::new(AppState::test_config())),
            reference_uploads: crate::reference_uploads::ReferenceUploadStore::from_mold_home(),
            output_disabled_override: true,
            reload_config_from_disk: false,
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
    async fn chain_jobs_listing_hides_one_shot_shims_unless_recovery_requests_them() {
        let home = tempfile::tempdir().unwrap();
        let _home = MoldHomeGuard::set(home.path());
        let db = mold_db::MetadataDb::open_in_memory().unwrap();
        seed_chain_job(
            &db,
            home.path(),
            "authored-sequence",
            ChainJobState::Running,
        );
        let shim_dir = seed_chain_job(&db, home.path(), "one-shot-shim", ChainJobState::Running);
        let mut shim = ChainJobManifest::read_from_dir(&shim_dir).unwrap();
        shim.ephemeral = true;
        shim.write_atomic(&shim_dir).unwrap();
        let app = app_with_chain_db(db);

        let public = app
            .clone()
            .oneshot(Request::get("/api/chain-jobs").body(Body::empty()).unwrap())
            .await
            .unwrap();
        assert_eq!(public.status(), StatusCode::OK);
        let public_body = json_body(public).await;
        let public_ids = public_body["jobs"]
            .as_array()
            .unwrap()
            .iter()
            .map(|job| job["id"].as_str().unwrap())
            .collect::<Vec<_>>();
        assert_eq!(public_ids, ["authored-sequence"]);

        let recovery = app
            .oneshot(
                Request::get("/api/chain-jobs?include_ephemeral=true")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(recovery.status(), StatusCode::OK);
        let recovery_body = json_body(recovery).await;
        let recovery_ids = recovery_body["jobs"]
            .as_array()
            .unwrap()
            .iter()
            .map(|job| job["id"].as_str().unwrap())
            .collect::<Vec<_>>();
        assert_eq!(recovery_ids, ["authored-sequence", "one-shot-shim"]);
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
    async fn cancel_running_chain_job_reports_cancelling_until_worker_settles() {
        let home = tempfile::tempdir().unwrap();
        let _home = MoldHomeGuard::set(home.path());
        let db = mold_db::MetadataDb::open_in_memory().unwrap();
        seed_chain_job(&db, home.path(), "running-cancel", ChainJobState::Running);
        let handle = Arc::new(crate::chain_job_runner::ChainJobRunnerHandle::inert_for_tests());
        handle.register_cancel_for_tests("running-cancel");
        let app = app_with_chain_handle(db, handle);

        let cancel_resp = app
            .clone()
            .oneshot(
                Request::post("/api/chain-jobs/running-cancel/cancel")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(cancel_resp.status(), StatusCode::ACCEPTED);
        let cancel_body = json_body(cancel_resp).await;
        assert_eq!(cancel_body["state"], "running");
        assert_eq!(cancel_body["cancelling"], true);

        let get_resp = app
            .oneshot(
                Request::get("/api/chain-jobs/running-cancel")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(get_resp.status(), StatusCode::OK);
        let get_body = json_body(get_resp).await;
        assert_eq!(get_body["state"], "running");
        assert_eq!(get_body["cancelling"], true);
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
    async fn gallery_video_export_advertises_formats_and_converts_gif_bounce() {
        let output_dir = tempfile::tempdir().unwrap();
        let mp4 = base64::engine::general_purpose::STANDARD
            .decode(include_str!("testdata/audio_muxed_final_mp4.b64").trim())
            .unwrap();
        std::fs::write(output_dir.path().join("rain dance.mp4"), mp4).unwrap();
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
        let app = create_router(state);

        let options = app
            .clone()
            .oneshot(
                Request::builder()
                    .uri("/api/gallery/export-options")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(options.status(), StatusCode::OK);
        let options = json_body(options).await;
        assert_eq!(
            options["gif_playback"],
            serde_json::json!(["loop", "bounce"])
        );
        assert!(options["formats"]
            .as_array()
            .unwrap()
            .iter()
            .any(|value| value == "gif"));

        let response = app
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/api/gallery/export/rain%20dance.mp4")
                    .header(axum::http::header::CONTENT_TYPE, "application/json")
                    .body(Body::from(
                        r#"{"format":"gif","playback":"bounce","repeat":"once","max_dimension":480,"fps":12}"#,
                    ))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::OK);
        assert_eq!(
            response.headers()[axum::http::header::CONTENT_TYPE],
            "image/gif"
        );
        assert_eq!(
            response.headers()[axum::http::header::CONTENT_DISPOSITION],
            "attachment; filename=\"rain_dance.gif\""
        );
        let bytes = axum::body::to_bytes(response.into_body(), 8 * 1024 * 1024)
            .await
            .unwrap();
        assert_eq!(&bytes[..6], b"GIF89a");
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
        assert_eq!(
            crate::rate_limit::classify_route("/api/generate/placement-preview", &Method::POST),
            Some(RouteTier::Generation)
        );
        assert_eq!(
            crate::rate_limit::classify_route("/api/chain-jobs/placement-preview", &Method::POST),
            Some(RouteTier::Generation)
        );
        assert_eq!(
            crate::rate_limit::classify_route("/api/devices/cuda:0123456789abcdef", &Method::PATCH,),
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
            collection: None,
            tags: None,
            title: None,
            source_fit: None,
            guidance_overrides: None,
            sample_shift: None,
            distill_strength_high: None,
            distill_strength_low: None,
            job_id: None,
            prompt: "from db".into(),
            negative_prompt: None,
            original_prompt: None,
            prompt_transform: None,
            batch_id: None,
            batch_index: None,
            batch_count: None,
            output_mode: None,
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
            edit_image_sha256s: None,
            references: None,
            keyframes: None,
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
            extend_video_path: None,
            extend_overlap_frames: None,
            pipeline: None,
            pipeline_provenance_sha256: None,
            source_preprocessing: None,
            ic_lora_control: None,
            hdr_exr_dir: None,
            hdr_exr_full_float: false,
            retake_range: None,
            spatial_upscale: None,
            temporal_upscale: None,
            frames: None,
            fps: None,
            chain_job_id: None,
            chain: None,
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
    async fn gallery_import_uses_no_replace_publication_metadata_and_events() {
        let dir = tempfile::tempdir().unwrap();
        let db = mold_db::MetadataDb::open_in_memory().unwrap();
        let config = mold_core::Config {
            output_dir: Some(dir.path().to_string_lossy().into_owned()),
            ..Default::default()
        };
        let (tx, _rx) = tokio::sync::mpsc::channel(16);
        let mut state = AppState::empty(
            config,
            crate::state::QueueHandle::new(tx),
            AppState::empty_gpu_pool_for_test(),
            200,
        );
        state.metadata_db = Arc::new(Some(db));
        let db = state.metadata_db.clone();
        let mut events = state.events.subscribe();
        let app = app_with_state(state);
        let bytes = minimal_png();
        let metadata = output_metadata("remote origin");

        let first = app
            .clone()
            .oneshot(
                Request::put("/api/gallery/import/print.png")
                    .header("content-type", "application/vnd.mold.gallery-import")
                    .body(Body::from(gallery_import_body(Some(&metadata), &bytes)))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(first.status(), StatusCode::CREATED);
        assert_eq!(json_body(first).await["filename"], "print.png");
        assert_eq!(std::fs::read(dir.path().join("print.png")).unwrap(), bytes);
        assert_eq!(
            db.as_ref()
                .as_ref()
                .unwrap()
                .get(dir.path(), "print.png")
                .unwrap()
                .unwrap()
                .metadata,
            metadata
        );
        assert!(matches!(
            events.try_recv().unwrap(),
            mold_core::ServerEvent::GalleryAdded { filename, .. } if filename == "print.png"
        ));

        // Identical bytes + identical metadata preserve cross-host filename
        // identity instead of minting print-2.png.
        let repeated = app
            .clone()
            .oneshot(
                Request::put("/api/gallery/import/print.png")
                    .header("content-type", "application/vnd.mold.gallery-import")
                    .body(Body::from(gallery_import_body(Some(&metadata), &bytes)))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(repeated.status(), StatusCode::OK);
        assert_eq!(json_body(repeated).await["filename"], "print.png");
        assert!(!dir.path().join("print-2.png").exists());
        let replay_event = events.try_recv();
        assert!(
            matches!(
                replay_event,
                Err(tokio::sync::broadcast::error::TryRecvError::Empty)
            ),
            "idempotent replay emitted an unexpected event: {replay_event:?}"
        );

        // Same bytes under the same identity may not silently rewrite
        // provenance.
        let conflict = app
            .oneshot(
                Request::put("/api/gallery/import/print.png")
                    .header("content-type", "application/vnd.mold.gallery-import")
                    .body(Body::from(gallery_import_body(
                        Some(&output_metadata("conflicting origin")),
                        &bytes,
                    )))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(conflict.status(), StatusCode::CONFLICT);
        assert_eq!(
            json_body(conflict).await["code"],
            "GALLERY_METADATA_CONFLICT"
        );
        let conflict_event = events.try_recv();
        assert!(
            matches!(
                conflict_event,
                Err(tokio::sync::broadcast::error::TryRecvError::Empty)
                    | Err(tokio::sync::broadcast::error::TryRecvError::Closed)
            ),
            "conflicting replay emitted an unexpected event: {conflict_event:?}"
        );
        assert_eq!(
            db.as_ref()
                .as_ref()
                .unwrap()
                .get(dir.path(), "print.png")
                .unwrap()
                .unwrap()
                .metadata,
            metadata
        );

        let (synthetic_app, mut synthetic_events) = {
            let config = mold_core::Config {
                output_dir: Some(dir.path().to_string_lossy().into_owned()),
                ..Default::default()
            };
            let (tx, _rx) = tokio::sync::mpsc::channel(16);
            let mut state = AppState::empty(
                config,
                crate::state::QueueHandle::new(tx),
                AppState::empty_gpu_pool_for_test(),
                200,
            );
            state.metadata_db = db.clone();
            let events = state.events.subscribe();
            (app_with_state(state), events)
        };
        let synthetic_conflict = synthetic_app
            .oneshot(
                Request::put("/api/gallery/import/print.png")
                    .header("content-type", "application/vnd.mold.gallery-import")
                    .body(Body::from(gallery_import_body_with_descriptor(
                        &metadata, true, &bytes,
                    )))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(synthetic_conflict.status(), StatusCode::CONFLICT);
        assert_eq!(
            json_body(synthetic_conflict).await["code"],
            "GALLERY_METADATA_CONFLICT"
        );
        assert!(
            !db.as_ref()
                .as_ref()
                .unwrap()
                .get(dir.path(), "print.png")
                .unwrap()
                .unwrap()
                .metadata_synthetic
        );
        assert!(matches!(
            synthetic_events.try_recv(),
            Err(tokio::sync::broadcast::error::TryRecvError::Empty)
                | Err(tokio::sync::broadcast::error::TryRecvError::Closed)
        ));
    }

    #[tokio::test]
    async fn db_disabled_import_archive_is_authoritative_across_replay_restart_delete_and_reuse() {
        // `Config::is_output_disabled` intentionally consults
        // `MOLD_OUTPUT_DIR` on every call. Keep the shared environment lock
        // for the full async lifecycle so the disabled-output route tests
        // cannot transiently turn this gallery off underneath a replay.
        let _env = env_lock()
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());

        fn state_for(dir: &std::path::Path) -> AppState {
            let config = mold_core::Config {
                output_dir: Some(dir.to_string_lossy().into_owned()),
                ..Default::default()
            };
            let (tx, _rx) = tokio::sync::mpsc::channel(16);
            AppState::empty(
                config,
                crate::state::QueueHandle::new(tx),
                AppState::empty_gpu_pool_for_test(),
                200,
            )
        }

        async fn import(
            app: &axum::Router,
            filename: &str,
            metadata: &mold_core::OutputMetadata,
            bytes: &[u8],
        ) -> StatusCode {
            app.clone()
                .oneshot(
                    Request::put(format!("/api/gallery/import/{filename}"))
                        .header("content-type", "application/vnd.mold.gallery-import")
                        .body(Body::from(gallery_import_body(Some(metadata), bytes)))
                        .unwrap(),
                )
                .await
                .unwrap()
                .status()
        }

        async fn listed(app: &axum::Router) -> serde_json::Value {
            let response = app
                .clone()
                .oneshot(Request::get("/api/gallery").body(Body::empty()).unwrap())
                .await
                .unwrap();
            assert_eq!(response.status(), StatusCode::OK);
            json_body(response).await
        }

        let dir = tempfile::tempdir().unwrap();
        let first_movie = minimal_mp4(0x11);
        let webp = minimal_webp();
        let mut movie_metadata = output_metadata("archived movie provenance");
        movie_metadata.output_format = Some(mold_core::OutputFormat::Mp4);
        let mut webp_metadata = output_metadata("archived WebP provenance");
        webp_metadata.output_format = Some(mold_core::OutputFormat::Webp);

        let state = state_for(dir.path());
        let app = app_with_state(state);
        assert_eq!(
            import(&app, "movie.mp4", &movie_metadata, &first_movie).await,
            StatusCode::CREATED
        );
        assert_eq!(
            import(&app, "still.webp", &webp_metadata, &webp).await,
            StatusCode::CREATED
        );

        let gallery = listed(&app).await;
        let rows = gallery.as_array().unwrap();
        assert_eq!(rows.len(), 2);
        assert!(
            rows.iter().any(|row| {
                row["filename"] == "movie.mp4"
                    && row["metadata"]["prompt"] == "archived movie provenance"
                    && !row["metadata_synthetic"].as_bool().unwrap_or(false)
            }),
            "movie archive metadata was not authoritative: {gallery}"
        );
        assert!(
            rows.iter().any(|row| {
                row["filename"] == "still.webp"
                    && row["metadata"]["prompt"] == "archived WebP provenance"
                    && !row["metadata_synthetic"].as_bool().unwrap_or(false)
            }),
            "WebP archive metadata was not authoritative: {gallery}"
        );
        assert_eq!(
            import(&app, "movie.mp4", &movie_metadata, &first_movie).await,
            StatusCode::OK
        );
        assert_eq!(
            import(&app, "still.webp", &webp_metadata, &webp).await,
            StatusCode::OK
        );

        // A fresh state models a DB-disabled process restart. Startup recovery
        // and the new router must retain the same committed archive authority.
        let restarted_state = state_for(dir.path());
        crate::batch_transaction::recover_transactions(
            dir.path(),
            &restarted_state.gallery_publication_gate,
            Arc::new(None),
        )
        .await
        .unwrap();
        let restarted = app_with_state(restarted_state);
        let gallery = listed(&restarted).await;
        assert!(gallery.as_array().unwrap().iter().any(|row| {
            row["filename"] == "movie.mp4"
                && row["metadata"]["prompt"] == "archived movie provenance"
                && !row["metadata_synthetic"].as_bool().unwrap_or(false)
        }));
        assert_eq!(
            import(&restarted, "movie.mp4", &movie_metadata, &first_movie).await,
            StatusCode::OK
        );

        let deleted = restarted
            .clone()
            .oneshot(
                Request::delete("/api/gallery/image/movie.mp4")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(deleted.status(), StatusCode::NO_CONTENT);

        let after_delete_state = state_for(dir.path());
        crate::batch_transaction::recover_transactions(
            dir.path(),
            &after_delete_state.gallery_publication_gate,
            Arc::new(None),
        )
        .await
        .unwrap();
        let after_delete = app_with_state(after_delete_state);
        let gallery = listed(&after_delete).await;
        assert!(!gallery
            .as_array()
            .unwrap()
            .iter()
            .any(|row| row["filename"] == "movie.mp4"));
        assert!(gallery.as_array().unwrap().iter().any(|row| {
            row["filename"] == "still.webp"
                && row["metadata"]["prompt"] == "archived WebP provenance"
                && !row["metadata_synthetic"].as_bool().unwrap_or(false)
        }));

        let replacement_movie = minimal_mp4(0x22);
        let mut replacement_metadata = output_metadata("replacement movie provenance");
        replacement_metadata.output_format = Some(mold_core::OutputFormat::Mp4);
        assert_eq!(
            import(
                &after_delete,
                "movie.mp4",
                &replacement_metadata,
                &replacement_movie,
            )
            .await,
            StatusCode::CREATED
        );
        let gallery = listed(&after_delete).await;
        assert!(gallery.as_array().unwrap().iter().any(|row| {
            row["filename"] == "movie.mp4"
                && row["metadata"]["prompt"] == "replacement movie provenance"
                && !row["metadata_synthetic"].as_bool().unwrap_or(false)
        }));
        assert_eq!(
            import(
                &after_delete,
                "movie.mp4",
                &replacement_metadata,
                &replacement_movie,
            )
            .await,
            StatusCode::OK
        );
    }

    #[tokio::test]
    async fn gallery_import_requires_configured_key_before_creating_staging() {
        let dir = tempfile::tempdir().unwrap();
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
        let keys = std::collections::HashSet::from(["desktop-local-key".to_string()]);
        let auth = Some(std::sync::Arc::new(crate::auth::ApiKeySet::new(keys)));
        let app = create_router(state)
            .layer(axum::middleware::from_fn(crate::auth::require_api_key))
            .layer(axum::middleware::from_fn_with_state(
                auth,
                crate::auth::inject_auth_state,
            ));
        let request_body = gallery_import_body(
            Some(&output_metadata("authenticated desktop import")),
            &minimal_png(),
        );

        let unauthorized = app
            .clone()
            .oneshot(
                Request::put("/api/gallery/import/auth.png")
                    .header("content-type", "application/vnd.mold.gallery-import")
                    .body(Body::from(request_body.clone()))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(unauthorized.status(), StatusCode::UNAUTHORIZED);
        assert!(!dir.path().join("auth.png").exists());
        assert!(
            !dir.path().join(".mold-batch").exists(),
            "auth must reject before transaction staging exists"
        );

        let authorized = app
            .oneshot(
                Request::put("/api/gallery/import/auth.png")
                    .header("content-type", "application/vnd.mold.gallery-import")
                    .header("x-api-key", "desktop-local-key")
                    .body(Body::from(request_body))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(authorized.status(), StatusCode::CREATED);
        assert!(dir.path().join("auth.png").is_file());
    }

    #[tokio::test]
    async fn gallery_import_accepts_chunked_declared_files_above_global_json_limit() {
        let dir = tempfile::tempdir().unwrap();
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
        let app = app_with_state(state);
        let descriptor = serde_json::to_vec(&serde_json::json!({
            "metadata": output_metadata("large streamed import"),
            "metadata_synthetic": false,
        }))
        .unwrap();
        let declared_len = 65_u64 * 1024 * 1024;
        let mut prefix = Vec::with_capacity(12 + descriptor.len() + 1);
        prefix.extend_from_slice(&(descriptor.len() as u32).to_be_bytes());
        prefix.extend_from_slice(&declared_len.to_be_bytes());
        prefix.extend_from_slice(&descriptor);
        prefix.push(0x89);
        let stream = futures::stream::iter([Ok::<_, std::convert::Infallible>(
            axum::body::Bytes::from(prefix),
        )]);

        let response = app
            .oneshot(
                Request::put("/api/gallery/import/large.mp4")
                    .header("content-type", "application/vnd.mold.gallery-import")
                    .body(Body::from_stream(stream))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(
            response.status(),
            StatusCode::UNPROCESSABLE_ENTITY,
            "route-specific streaming limit must admit a logical file above 64 MiB"
        );
        assert_eq!(json_body(response).await["code"], "INVALID_GALLERY_IMPORT");
        assert!(!dir.path().join("large.mp4").exists());
    }

    #[tokio::test]
    async fn gallery_import_framing_and_filename_are_bounded_and_fail_closed() {
        let dir = tempfile::tempdir().unwrap();
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
        let app = app_with_state(state);

        let traversal = app
            .clone()
            .oneshot(
                Request::put("/api/gallery/import/%2E%2E%2Fsecret.png")
                    .body(Body::from(gallery_import_body(None, &minimal_png())))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert!(
            matches!(
                traversal.status(),
                StatusCode::BAD_REQUEST | StatusCode::UNPROCESSABLE_ENTITY | StatusCode::NOT_FOUND
            ),
            "encoded traversal must never reach publication: {}",
            traversal.status()
        );

        let mut oversized_metadata = Vec::new();
        oversized_metadata.extend_from_slice(&(1024_u32 * 1024 + 1).to_be_bytes());
        oversized_metadata.extend_from_slice(&0_u64.to_be_bytes());
        let oversized_metadata = app
            .clone()
            .oneshot(
                Request::put("/api/gallery/import/print.png")
                    .header("content-type", "application/vnd.mold.gallery-import")
                    .body(Body::from(oversized_metadata))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(oversized_metadata.status(), StatusCode::PAYLOAD_TOO_LARGE);
        assert_eq!(
            json_body(oversized_metadata).await["code"],
            "GALLERY_IMPORT_TOO_LARGE"
        );

        for (filename, corrupt) in [
            ("corrupt.png", vec![0_u8; 320]),
            ("corrupt.mp4", vec![0_u8; 320]),
        ] {
            let response = app
                .clone()
                .oneshot(
                    Request::put(format!("/api/gallery/import/{filename}"))
                        .header("content-type", "application/vnd.mold.gallery-import")
                        .body(Body::from(gallery_import_body(
                            Some(&output_metadata("corrupt import")),
                            &corrupt,
                        )))
                        .unwrap(),
                )
                .await
                .unwrap();
            assert_eq!(response.status(), StatusCode::UNPROCESSABLE_ENTITY);
            assert_eq!(json_body(response).await["code"], "INVALID_GALLERY_MEDIA");
            assert!(!dir.path().join(filename).exists());
        }

        let short = app
            .oneshot(
                Request::put("/api/gallery/import/print.png")
                    .header("content-type", "application/vnd.mold.gallery-import")
                    .body(Body::from(vec![0, 0, 0, 8, 0, 0, 0, 0, 0, 0, 0, 1, b'{']))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(short.status(), StatusCode::UNPROCESSABLE_ENTITY);
        assert_eq!(json_body(short).await["code"], "INVALID_GALLERY_IMPORT");
        assert!(
            std::fs::read_dir(dir.path())
                .unwrap()
                .all(|entry| entry.unwrap().file_name() != "print.png"),
            "invalid framing must not publish a file"
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
            workers: Vec::new().into(),
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
            workers: Vec::new().into(),
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

    /// `DELETE /api/gallery/image/:filename?permanent=true` must remove the
    /// matching DB row in addition to the file on disk so the next list call
    /// doesn't resurrect a stale entry from cache. (Without `permanent` the
    /// DB-backed delete moves to the trash and keeps the row — see
    /// `gallery_delete_moves_to_trash_when_db_available`.)
    #[allow(clippy::await_holding_lock)]
    #[tokio::test]
    async fn gallery_permanent_delete_drops_metadata_row() {
        use mold_db::{GenerationRecord, MetadataDb, RecordSource};

        let dir = tempfile::tempdir().unwrap();
        let target = dir.path().join("doomed.png");
        std::fs::write(&target, vec![0u8; 1024]).unwrap();

        let db = MetadataDb::open(&dir.path().join("mold.db")).unwrap();
        let metadata = mold_core::OutputMetadata {
            collection: None,
            tags: None,
            title: None,
            source_fit: None,
            guidance_overrides: None,
            sample_shift: None,
            distill_strength_high: None,
            distill_strength_low: None,
            job_id: None,
            prompt: "doomed".into(),
            negative_prompt: None,
            original_prompt: None,
            prompt_transform: None,
            batch_id: None,
            batch_index: None,
            batch_count: None,
            output_mode: None,
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
            edit_image_sha256s: None,
            references: None,
            keyframes: None,
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
            extend_video_path: None,
            extend_overlap_frames: None,
            pipeline: None,
            pipeline_provenance_sha256: None,
            source_preprocessing: None,
            ic_lora_control: None,
            hdr_exr_dir: None,
            hdr_exr_full_float: false,
            retake_range: None,
            spatial_upscale: None,
            temporal_upscale: None,
            frames: None,
            fps: None,
            chain_job_id: None,
            chain: None,
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
                    .uri("/api/gallery/image/doomed.png?permanent=true")
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
    async fn post_api_downloads_enqueues_authorized_compact_h3() {
        let state = AppState::for_tests();
        let app = app_with_state(state.clone());
        let body = serde_json::json!({ "model": mold_core::minimax_h3::FL2VA_COMFY });
        let response = app
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/api/downloads")
                    .header("content-type", "application/json")
                    .body(Body::from(body.to_string()))
                    .unwrap(),
            )
            .await
            .unwrap();

        assert_eq!(response.status(), StatusCode::OK);
        let listing = state.downloads.listing().await;
        assert!(listing.active.is_none());
        assert_eq!(listing.queued.len(), 1);
        assert_eq!(listing.queued[0].model, mold_core::minimax_h3::FL2VA_COMFY);
    }

    #[tokio::test]
    async fn config_only_h3_download_and_runtime_admin_routes_remain_gated() {
        let state = AppState::for_tests();
        state.config.write().await.models.insert(
            "private-video-model".to_string(),
            mold_core::ModelConfig {
                family: Some("minimax-h3".to_string()),
                ..Default::default()
            },
        );
        let app = app_with_state(state.clone());
        let body = serde_json::json!({ "model": "private-video-model" }).to_string();

        for route in ["/api/downloads", "/api/models/pull", "/api/models/load"] {
            let response = app
                .clone()
                .oneshot(
                    Request::post(route)
                        .header("content-type", "application/json")
                        .body(Body::from(body.clone()))
                        .unwrap(),
                )
                .await
                .unwrap();
            assert_eq!(
                response.status(),
                StatusCode::UNAVAILABLE_FOR_LEGAL_REASONS,
                "{route}"
            );
            let response = json_body(response).await;
            assert_eq!(
                response["code"],
                mold_core::MINIMAX_H3_AUTHORIZATION_REQUIRED,
                "{route}"
            );
        }

        let listing = state.downloads.listing().await;
        assert!(listing.active.is_none());
        assert!(listing.queued.is_empty());
    }

    #[tokio::test]
    async fn prompt_transform_routes_reject_h3_local_or_hosted_models_before_execution() {
        for (backend, model, api_model) in [
            ("local", "MiniMax-H3", "ordinary-api-model"),
            (
                "http://127.0.0.1:9",
                "ordinary-local-model",
                "MiniMaxAI/MiniMax-H3",
            ),
        ] {
            let state = AppState::for_tests();
            {
                let mut config = state.config.write().await;
                config.expand.backend = backend.to_string();
                config.expand.model = model.to_string();
                config.expand.api_model = api_model.to_string();
            }
            let app = app_with_state(state);
            for (route, body) in [
                (
                    "/api/expand",
                    serde_json::json!({
                        "prompt": "a red apple",
                        "model_family": "flux",
                        "variations": 1
                    }),
                ),
                (
                    "/api/remix",
                    serde_json::json!({
                        "source_prompt": "a red apple",
                        "model_family": "flux",
                        "variations": 1
                    }),
                ),
            ] {
                let response = app
                    .clone()
                    .oneshot(
                        Request::post(route)
                            .header("content-type", "application/json")
                            .body(Body::from(body.to_string()))
                            .unwrap(),
                    )
                    .await
                    .unwrap();
                assert_eq!(
                    response.status(),
                    StatusCode::UNAVAILABLE_FOR_LEGAL_REASONS,
                    "{backend} {route}"
                );
                assert_eq!(
                    json_body(response).await["code"],
                    mold_core::MINIMAX_H3_AUTHORIZATION_REQUIRED,
                    "{backend} {route}"
                );
            }
        }
    }

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
        assert_eq!(limits["frames_per_clip_cap"], 481);
        assert_eq!(limits["fps"], 24);
        assert_eq!(limits["frames_per_clip_runtime_seconds"], 20);
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

    /// The endpoint must recommend the model's manifest default frames
    /// (25 for LTX-Video, 97 for LTX-2), not the family cap, so clients
    /// can default new clips to what the model actually ships with.
    #[tokio::test]
    async fn chain_limits_recommends_manifest_default_frames() {
        let app = app_empty();
        let response = app
            .oneshot(
                Request::get("/api/capabilities/chain-limits?model=ltx-video-0.9.6:bf16")
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
        assert!(
            limits.get("frames_per_clip_runtime_seconds").is_none(),
            "ltx-video publishes a flat frame ceiling, not a duration budget",
        );
        assert_eq!(
            limits["frames_per_clip_recommended"], 25,
            "LTX-Video manifests declare frames: 25",
        );

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
        assert_eq!(
            limits["frames_per_clip_recommended"], 97,
            "LTX-2 manifests declare frames: 97",
        );
    }

    /// /api/models must carry the additive per-model frame fields for video
    /// models and omit them for image models.
    #[tokio::test]
    async fn models_endpoint_advertises_frame_defaults_for_video_models() {
        let app = app_empty();
        let response = app
            .oneshot(Request::get("/api/models").body(Body::empty()).unwrap())
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::OK);
        let body = axum::body::to_bytes(response.into_body(), usize::MAX)
            .await
            .unwrap();
        let models: Vec<serde_json::Value> = serde_json::from_slice(&body).unwrap();

        let ltx2 = models
            .iter()
            .find(|m| m["family"] == "ltx2")
            .expect("an ltx2 model should be listed");
        assert_eq!(ltx2["default_frames"], 97);
        assert_eq!(ltx2["default_fps"], 24);
        assert_eq!(
            ltx2["max_frames"], 481,
            "the 20s temporal RoPE budget at the model's own 24 fps default",
        );
        assert_eq!(ltx2["max_runtime_seconds"], 20);
        assert_eq!(ltx2["frame_step"], 8);

        let flux = models
            .iter()
            .find(|m| m["family"] == "flux")
            .expect("a flux model should be listed");
        assert!(flux.get("default_frames").is_none());
        assert!(flux.get("max_frames").is_none());
        assert!(flux.get("frame_step").is_none());
    }

    #[tokio::test]
    async fn capabilities_chain_limits_accepts_ltx2_two_stage_pipeline() {
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
        assert_eq!(limits["supports_sequence"], true);
        assert!(limits["sequence_unsupported_reason"].is_null());
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
                primary_size_bytes: None,
                low_noise_filename_rel: None,
                low_noise_size_bytes: None,
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
        assert_eq!(limits["frames_per_clip_cap"], 481);
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
    /// The non-streaming `/api/generate` route returns one body. Audio prints
    /// carry a waveform PNG in `JobResult.image` so the queue and SSE pipeline
    /// have a raster to work with — a branch that only special-cases video
    /// therefore hands a direct HTTP caller `image/png` and silently drops the
    /// WAV the render actually produced.
    #[test]
    fn generate_media_headers_return_the_wav_for_an_audio_response() {
        let waveform = vec![0x89, 0x50, 0x4E, 0x47];
        let wav = b"RIFF....WAVEfmt ".to_vec();
        let response = GenerateResponse {
            request_warnings: Vec::new(),
            audio: Some(mold_core::AudioData {
                data: wav.clone(),
                format: OutputFormat::Wav,
                sample_rate: 24_000,
                channels: 2,
                duration_ms: 5_010,
                thumbnail: waveform.clone(),
                thumbnail_width: 640,
                thumbnail_height: 360,
            }),
            images: Vec::new(),
            video: None,
            generation_time_ms: 1,
            model: "ltx-2-19b-dev:fp8".into(),
            seed_used: 7,
            gpu: None,
        };
        let img = ImageData {
            data: waveform,
            format: OutputFormat::Png,
            width: 640,
            height: 360,
            index: 0,
        };

        let mut headers = axum::http::HeaderMap::new();
        let body = crate::routes::apply_media_headers(&response, img, &mut headers);

        assert_eq!(body, wav, "the caller must receive the encoded audio");
        assert_eq!(
            headers[axum::http::header::CONTENT_TYPE],
            "audio/wav",
            "an audio print must not be labelled as its waveform thumbnail",
        );
        assert_eq!(
            headers["x-mold-audio-format"], "wav",
            "the server states the container; the request may have omitted it",
        );
        assert_eq!(headers["x-mold-audio-sample-rate"], "24000");
        assert_eq!(headers["x-mold-audio-channels"], "2");
        assert_eq!(headers["x-mold-audio-duration-ms"], "5010");
        assert_eq!(headers["x-mold-audio-thumbnail-width"], "640");
        assert_eq!(headers["x-mold-audio-thumbnail-height"], "360");
        assert!(
            !headers.contains_key("x-mold-video-frames"),
            "audio has no frames; the video probe must not fire",
        );
    }

    /// The image and video branches must keep behaving exactly as they did.
    #[test]
    fn generate_media_headers_keep_the_video_and_image_shapes() {
        let img = || ImageData {
            data: b"raster".to_vec(),
            format: OutputFormat::Png,
            width: 64,
            height: 48,
            index: 0,
        };
        let still = GenerateResponse {
            request_warnings: Vec::new(),
            audio: None,
            images: vec![img()],
            video: None,
            generation_time_ms: 1,
            model: "flux-dev:q8".into(),
            seed_used: 1,
            gpu: None,
        };
        let mut headers = axum::http::HeaderMap::new();
        headers.insert(
            axum::http::header::CONTENT_TYPE,
            axum::http::HeaderValue::from_static("image/png"),
        );
        let body = crate::routes::apply_media_headers(&still, img(), &mut headers);
        assert_eq!(body, b"raster".to_vec());
        assert_eq!(headers[axum::http::header::CONTENT_TYPE], "image/png");

        let clip = GenerateResponse {
            request_warnings: Vec::new(),
            audio: None,
            images: Vec::new(),
            video: Some(mold_core::VideoData {
                data: b"mp4-bytes".to_vec(),
                format: OutputFormat::Mp4,
                width: 960,
                height: 576,
                frames: 97,
                fps: 24,
                pipeline: Some(mold_core::Ltx2PipelineMode::TwoStageHq),
                pipeline_provenance_sha256: None,
                source_preprocessing: None,
                thumbnail: Vec::new(),
                gif_preview: Vec::new(),
                has_audio: true,
                duration_ms: Some(4_000),
                audio_sample_rate: Some(48_000),
                audio_channels: Some(2),
            }),
            generation_time_ms: 1,
            model: "ltx-2-19b-dev:fp8".into(),
            seed_used: 1,
            gpu: None,
        };
        let mut headers = axum::http::HeaderMap::new();
        let body = crate::routes::apply_media_headers(&clip, img(), &mut headers);
        assert_eq!(body, b"mp4-bytes".to_vec());
        assert_eq!(headers[axum::http::header::CONTENT_TYPE], "video/mp4");
        assert_eq!(headers["x-mold-video-frames"], "97");
        assert_eq!(headers["x-mold-video-fps"], "24");
        assert_eq!(headers["x-mold-video-pipeline"], "two-stage-hq");
        assert_eq!(headers["x-mold-video-has-audio"], "1");
        assert_eq!(headers["x-mold-video-audio-sample-rate"], "48000");
    }

    // ── Library organization + trash ────────────────────────────────────────
    //
    // Fixtures: an in-memory metadata DB wired into an otherwise empty
    // AppState whose output dir is a tempdir. `seed_print` writes a valid
    // PNG and its `generations` row so the DB-backed listing path is taken.

    fn organized_state(dir: &std::path::Path) -> (AppState, Arc<Option<mold_db::MetadataDb>>) {
        let db = mold_db::MetadataDb::open_in_memory().unwrap();
        let config = mold_core::Config {
            output_dir: Some(dir.to_string_lossy().into_owned()),
            ..Default::default()
        };
        let (tx, _rx) = tokio::sync::mpsc::channel(16);
        let mut state = AppState::empty(
            config,
            crate::state::QueueHandle::new(tx),
            AppState::empty_gpu_pool_for_test(),
            200,
        );
        state.metadata_db = Arc::new(Some(db));
        let db = state.metadata_db.clone();
        (state, db)
    }

    fn seed_print(
        db: &Arc<Option<mold_db::MetadataDb>>,
        dir: &std::path::Path,
        name: &str,
        title: Option<&str>,
    ) {
        let path = dir.join(name);
        std::fs::write(&path, minimal_png()).unwrap();
        let mut metadata = output_metadata(&format!("prompt for {name}"));
        metadata.title = title.map(str::to_string);
        let mut record = mold_db::GenerationRecord::from_save(
            dir,
            name,
            mold_core::OutputFormat::Png,
            metadata,
            mold_db::RecordSource::Server,
            mold_core::time::now_epoch_ms(),
        );
        record.stat_from_disk(&path);
        db.as_ref().as_ref().unwrap().upsert(&record).unwrap();
    }

    fn json_request(method: &str, uri: &str, body: serde_json::Value) -> Request<Body> {
        Request::builder()
            .method(method)
            .uri(uri)
            .header("content-type", "application/json")
            .body(Body::from(body.to_string()))
            .unwrap()
    }

    fn empty_request(method: &str, uri: &str) -> Request<Body> {
        Request::builder()
            .method(method)
            .uri(uri)
            .body(Body::empty())
            .unwrap()
    }

    async fn gallery_rows(app: &axum::Router, uri: &str) -> Vec<serde_json::Value> {
        let resp = app
            .clone()
            .oneshot(empty_request("GET", uri))
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::OK, "{uri}");
        json_body(resp).await.as_array().unwrap().clone()
    }

    fn drain_events(
        rx: &mut tokio::sync::broadcast::Receiver<mold_core::ServerEvent>,
    ) -> Vec<mold_core::ServerEvent> {
        let mut out = Vec::new();
        while let Ok(event) = rx.try_recv() {
            out.push(event);
        }
        out
    }

    #[tokio::test]
    async fn gallery_list_view_excludes_trashed_rows_and_trash_view_lists_them() {
        let dir = tempfile::tempdir().unwrap();
        let (state, db) = organized_state(dir.path());
        seed_print(&db, dir.path(), "live.png", None);
        seed_print(&db, dir.path(), "binned.png", Some("Binned"));
        // Trash `binned.png` the way the primitive does: bytes into `.trash/`
        // plus the row flag — without going through HTTP, so the listing
        // filter is tested on its own.
        let trash = mold_db::trash_dir(dir.path());
        std::fs::create_dir_all(&trash).unwrap();
        std::fs::rename(dir.path().join("binned.png"), trash.join("binned.png")).unwrap();
        let trashed_at_ms = 1_700_000_000_000_i64;
        db.as_ref()
            .as_ref()
            .unwrap()
            .mark_trashed(dir.path(), "binned.png", trashed_at_ms)
            .unwrap();
        let app = app_with_state(state);

        let library = gallery_rows(&app, "/api/gallery").await;
        assert_eq!(
            library
                .iter()
                .map(|r| r["filename"].as_str().unwrap())
                .collect::<Vec<_>>(),
            vec!["live.png"],
            "the default (library) view must exclude trashed rows"
        );
        assert!(library[0].get("trashed_at").is_none());

        let explicit = gallery_rows(&app, "/api/gallery?view=library").await;
        assert_eq!(explicit.len(), 1);

        let trashed = gallery_rows(&app, "/api/gallery?view=trash").await;
        assert_eq!(trashed.len(), 1);
        assert_eq!(trashed[0]["filename"], "binned.png");
        assert_eq!(trashed[0]["title"], "Binned");
        assert_eq!(trashed[0]["trashed_at"], (trashed_at_ms / 1000) as u64);
        // Default retention is 30 days.
        assert_eq!(
            trashed[0]["purge_at"],
            (trashed_at_ms / 1000 + 30 * 24 * 60 * 60) as u64
        );

        let bogus = app
            .clone()
            .oneshot(empty_request("GET", "/api/gallery?view=attic"))
            .await
            .unwrap();
        assert_eq!(bogus.status(), StatusCode::UNPROCESSABLE_ENTITY);
    }

    #[tokio::test]
    async fn gallery_list_enriches_rows_with_title_tags_favorite_and_collections() {
        let dir = tempfile::tempdir().unwrap();
        let (state, db) = organized_state(dir.path());
        seed_print(&db, dir.path(), "owl.png", Some("Creation title"));
        seed_print(&db, dir.path(), "plain.png", None);
        let mdb = db.as_ref().as_ref().unwrap();
        mdb.set_title(dir.path(), "owl.png", Some("Edited title"))
            .unwrap();
        mdb.set_favorite(dir.path(), "owl.png", true).unwrap();
        mdb.add_tags(dir.path(), "owl.png", &["Night".into(), "birds".into()])
            .unwrap();
        let collection = mdb.create_collection("Owls", None).unwrap();
        mdb.collection_add(&collection.id, dir.path(), &["owl.png".into()])
            .unwrap();
        let app = app_with_state(state);

        let rows = gallery_rows(&app, "/api/gallery").await;
        let owl = rows.iter().find(|r| r["filename"] == "owl.png").unwrap();
        assert_eq!(
            owl["title"], "Edited title",
            "row title wins over metadata title"
        );
        assert_eq!(owl["favorite"], true);
        assert_eq!(owl["tags"], serde_json::json!(["birds", "Night"]));
        assert_eq!(owl["collections"], serde_json::json!([collection.id]));
        let plain = rows.iter().find(|r| r["filename"] == "plain.png").unwrap();
        assert!(plain.get("title").is_none());
        assert!(
            plain.get("tags").is_none(),
            "empty tags are omitted on the wire"
        );
        assert!(plain.get("favorite").is_none());
    }

    #[tokio::test]
    async fn gallery_scan_fallback_ignores_the_trash_subdirectory() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("live.png"), minimal_png()).unwrap();
        let trash = mold_db::trash_dir(dir.path());
        std::fs::create_dir_all(&trash).unwrap();
        std::fs::write(trash.join("binned.png"), minimal_png()).unwrap();
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
        let app = app_with_state(state);

        let rows = gallery_rows(&app, "/api/gallery").await;
        assert_eq!(
            rows.iter()
                .map(|r| r["filename"].as_str().unwrap())
                .collect::<Vec<_>>(),
            vec!["live.png"]
        );
        // No DB ⇒ no trash index: the trash view is empty rather than an error.
        assert!(gallery_rows(&app, "/api/gallery?view=trash")
            .await
            .is_empty());
    }

    #[tokio::test]
    async fn gallery_patch_edits_title_favorite_tags_and_publishes_updated_row() {
        let dir = tempfile::tempdir().unwrap();
        let (state, db) = organized_state(dir.path());
        seed_print(&db, dir.path(), "owl.png", None);
        let mut events = state.events.subscribe();
        let app = app_with_state(state);

        let resp = app
            .clone()
            .oneshot(json_request(
                "PATCH",
                "/api/gallery/image/owl.png",
                serde_json::json!({
                    "title": "  Smurf village  ",
                    "favorite": true,
                    "tags": ["teal", "Night"]
                }),
            ))
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
        let body = json_body(resp).await;
        assert_eq!(body["filename"], "owl.png");
        assert_eq!(body["title"], "Smurf village", "title is trimmed");
        assert_eq!(body["favorite"], true);
        assert_eq!(body["tags"], serde_json::json!(["Night", "teal"]));
        match events.try_recv().unwrap() {
            mold_core::ServerEvent::GalleryUpdated { filename, image } => {
                assert_eq!(filename, "owl.png");
                let image = image.expect("patch carries the refreshed row");
                assert_eq!(image.title.as_deref(), Some("Smurf village"));
                assert!(image.favorite);
            }
            other => panic!("expected gallery_updated, got {other:?}"),
        }

        // add/remove without `tags`, and an empty title clears it.
        let resp = app
            .clone()
            .oneshot(json_request(
                "PATCH",
                "/api/gallery/image/owl.png",
                serde_json::json!({"title": "", "add_tags": ["owls"], "remove_tags": ["teal"]}),
            ))
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
        let body = json_body(resp).await;
        assert!(
            body.get("title").is_none(),
            "empty title clears the row title"
        );
        assert_eq!(body["tags"], serde_json::json!(["Night", "owls"]));
        // Still a favorite: untouched fields stay put.
        assert_eq!(body["favorite"], true);

        // Invalid title → 422, nothing changed.
        let resp = app
            .clone()
            .oneshot(json_request(
                "PATCH",
                "/api/gallery/image/owl.png",
                serde_json::json!({"title": "bad\u{0007}title"}),
            ))
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::UNPROCESSABLE_ENTITY);

        // Unknown print → 404.
        let resp = app
            .clone()
            .oneshot(json_request(
                "PATCH",
                "/api/gallery/image/nope.png",
                serde_json::json!({"favorite": true}),
            ))
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::NOT_FOUND);

        // Listing reflects the edits.
        let rows = gallery_rows(&app, "/api/gallery").await;
        assert_eq!(rows[0]["tags"], serde_json::json!(["Night", "owls"]));
    }

    #[tokio::test]
    async fn gallery_organize_applies_bulk_edits_in_one_transaction() {
        let dir = tempfile::tempdir().unwrap();
        let (state, db) = organized_state(dir.path());
        seed_print(&db, dir.path(), "a.png", None);
        seed_print(&db, dir.path(), "b.png", None);
        let collection = db
            .as_ref()
            .as_ref()
            .unwrap()
            .create_collection("Bulk", None)
            .unwrap();
        let mut events = state.events.subscribe();
        let app = app_with_state(state);

        let resp = app
            .clone()
            .oneshot(json_request(
                "POST",
                "/api/gallery/organize",
                serde_json::json!({
                    "filenames": ["a.png", "b.png"],
                    "favorite": true,
                    "add_tags": ["batch"],
                    "add_to_collections": [collection.id]
                }),
            ))
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::NO_CONTENT);
        let published = drain_events(&mut events);
        let updated: Vec<_> = published
            .iter()
            .filter_map(|e| match e {
                mold_core::ServerEvent::GalleryUpdated { filename, image } => {
                    assert!(image.is_none(), "bulk edits publish without the row");
                    Some(filename.clone())
                }
                _ => None,
            })
            .collect();
        assert_eq!(updated, vec!["a.png", "b.png"]);
        assert!(published
            .iter()
            .any(|e| matches!(e, mold_core::ServerEvent::GalleryCollectionsChanged {})));
        let rows = gallery_rows(&app, "/api/gallery").await;
        for row in &rows {
            assert_eq!(row["favorite"], true);
            assert_eq!(row["tags"], serde_json::json!(["batch"]));
            assert_eq!(row["collections"], serde_json::json!([collection.id]));
        }

        // Unknown filename → 404 naming it, nothing applied.
        let resp = app
            .clone()
            .oneshot(json_request(
                "POST",
                "/api/gallery/organize",
                serde_json::json!({"filenames": ["a.png", "ghost.png"], "favorite": false}),
            ))
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::NOT_FOUND);
        assert!(json_body(resp).await["error"]
            .as_str()
            .unwrap()
            .contains("ghost.png"));
        let rows = gallery_rows(&app, "/api/gallery").await;
        assert!(rows.iter().all(|r| r["favorite"] == true));

        // Unknown collection → 404.
        let resp = app
            .clone()
            .oneshot(json_request(
                "POST",
                "/api/gallery/organize",
                serde_json::json!({"filenames": ["a.png"], "remove_from_collections": ["missing"]}),
            ))
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::NOT_FOUND);

        // Empty filename list → 422.
        let resp = app
            .oneshot(json_request(
                "POST",
                "/api/gallery/organize",
                serde_json::json!({"filenames": [], "favorite": true}),
            ))
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::UNPROCESSABLE_ENTITY);
    }

    #[tokio::test]
    async fn gallery_collections_crud_and_membership_round_trip() {
        let dir = tempfile::tempdir().unwrap();
        let (state, db) = organized_state(dir.path());
        seed_print(&db, dir.path(), "a.png", None);
        seed_print(&db, dir.path(), "b.png", None);
        let mut events = state.events.subscribe();
        let app = app_with_state(state);

        let resp = app
            .clone()
            .oneshot(json_request(
                "POST",
                "/api/gallery/collections",
                serde_json::json!({"name": "Smurf Village", "description": "blue"}),
            ))
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::CREATED);
        let created = json_body(resp).await;
        let id = created["id"].as_str().unwrap().to_string();
        assert_eq!(created["name"], "Smurf Village");
        assert_eq!(created["slug"], "smurf-village");
        assert_eq!(created["description"], "blue");
        assert_eq!(created["count"], 0);
        assert!(matches!(
            events.try_recv().unwrap(),
            mold_core::ServerEvent::GalleryCollectionsChanged {}
        ));

        // Same slug → 409; empty name → 422.
        let resp = app
            .clone()
            .oneshot(json_request(
                "POST",
                "/api/gallery/collections",
                serde_json::json!({"name": "smurf   village"}),
            ))
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::CONFLICT);
        let resp = app
            .clone()
            .oneshot(json_request(
                "POST",
                "/api/gallery/collections",
                serde_json::json!({"name": "   "}),
            ))
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::UNPROCESSABLE_ENTITY);

        // Items: add two, then remove one; unknown filename → 404 untouched.
        let resp = app
            .clone()
            .oneshot(json_request(
                "PUT",
                &format!("/api/gallery/collections/{id}/items"),
                serde_json::json!({"add": ["a.png", "b.png"]}),
            ))
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
        assert_eq!(json_body(resp).await["count"], 2);
        let resp = app
            .clone()
            .oneshot(json_request(
                "PUT",
                &format!("/api/gallery/collections/{id}/items"),
                serde_json::json!({"add": ["ghost.png"]}),
            ))
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::NOT_FOUND);
        let resp = app
            .clone()
            .oneshot(json_request(
                "PUT",
                &format!("/api/gallery/collections/{id}/items"),
                serde_json::json!({"remove": ["a.png"]}),
            ))
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
        assert_eq!(json_body(resp).await["count"], 1);

        // Detail carries the ordered filenames.
        let resp = app
            .clone()
            .oneshot(empty_request(
                "GET",
                &format!("/api/gallery/collections/{id}"),
            ))
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
        let detail = json_body(resp).await;
        assert_eq!(detail["collection"]["id"], id);
        assert_eq!(detail["filenames"], serde_json::json!(["b.png"]));

        // Rename + cover.
        let resp = app
            .clone()
            .oneshot(json_request(
                "PATCH",
                &format!("/api/gallery/collections/{id}"),
                serde_json::json!({"name": "Gargamel", "cover_filename": "b.png"}),
            ))
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
        let updated = json_body(resp).await;
        assert_eq!(updated["slug"], "gargamel");
        assert_eq!(updated["cover_filename"], "b.png");
        // A cover that is not a member → 422.
        let resp = app
            .clone()
            .oneshot(json_request(
                "PATCH",
                &format!("/api/gallery/collections/{id}"),
                serde_json::json!({"cover_filename": "a.png"}),
            ))
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::UNPROCESSABLE_ENTITY);

        // Listing shows the collection; the print's row names it.
        let listed = gallery_rows(&app, "/api/gallery/collections").await;
        assert_eq!(listed.len(), 1);
        assert_eq!(listed[0]["name"], "Gargamel");
        let rows = gallery_rows(&app, "/api/gallery").await;
        let b = rows.iter().find(|r| r["filename"] == "b.png").unwrap();
        assert_eq!(b["collections"], serde_json::json!([id]));

        // Delete never touches prints; a second delete is 404.
        let resp = app
            .clone()
            .oneshot(empty_request(
                "DELETE",
                &format!("/api/gallery/collections/{id}"),
            ))
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::NO_CONTENT);
        let resp = app
            .clone()
            .oneshot(empty_request(
                "DELETE",
                &format!("/api/gallery/collections/{id}"),
            ))
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::NOT_FOUND);
        assert!(gallery_rows(&app, "/api/gallery/collections")
            .await
            .is_empty());
        assert_eq!(gallery_rows(&app, "/api/gallery").await.len(), 2);
        assert!(dir.path().join("b.png").is_file());
        let changed = drain_events(&mut events)
            .into_iter()
            .filter(|e| matches!(e, mold_core::ServerEvent::GalleryCollectionsChanged {}))
            .count();
        assert!(
            changed >= 4,
            "every collection mutation announces itself: {changed}"
        );
    }

    #[tokio::test]
    async fn gallery_tags_list_rename_merge_and_delete() {
        let dir = tempfile::tempdir().unwrap();
        let (state, db) = organized_state(dir.path());
        seed_print(&db, dir.path(), "a.png", None);
        seed_print(&db, dir.path(), "b.png", None);
        let mdb = db.as_ref().as_ref().unwrap();
        mdb.add_tags(dir.path(), "a.png", &["teal".into(), "owls".into()])
            .unwrap();
        mdb.add_tags(dir.path(), "b.png", &["Teal".into(), "birds".into()])
            .unwrap();
        let mut events = state.events.subscribe();
        let app = app_with_state(state);

        let tags = gallery_rows(&app, "/api/gallery/tags").await;
        assert_eq!(
            tags.iter()
                .map(|t| (
                    t["name"].as_str().unwrap().to_string(),
                    t["count"].as_u64().unwrap()
                ))
                .collect::<Vec<_>>(),
            vec![("birds".into(), 1), ("owls".into(), 1), ("teal".into(), 2)]
        );

        // Rename owls → birds merges.
        let resp = app
            .clone()
            .oneshot(json_request(
                "PATCH",
                "/api/gallery/tags/owls",
                serde_json::json!({"name": "birds"}),
            ))
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
        let merged = json_body(resp).await;
        assert_eq!(merged["name"], "birds");
        assert_eq!(merged["count"], 2);
        let updated: Vec<_> = drain_events(&mut events)
            .into_iter()
            .filter_map(|e| match e {
                mold_core::ServerEvent::GalleryUpdated { filename, .. } => Some(filename),
                _ => None,
            })
            .collect();
        assert_eq!(
            updated,
            vec!["a.png"],
            "only prints that carried the tag are announced"
        );

        // Unknown tag → 404; empty new name → 422.
        let resp = app
            .clone()
            .oneshot(json_request(
                "PATCH",
                "/api/gallery/tags/ghost",
                serde_json::json!({"name": "x"}),
            ))
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::NOT_FOUND);
        let resp = app
            .clone()
            .oneshot(json_request(
                "PATCH",
                "/api/gallery/tags/teal",
                serde_json::json!({"name": "  "}),
            ))
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::UNPROCESSABLE_ENTITY);

        // Delete teal everywhere.
        let resp = app
            .clone()
            .oneshot(empty_request("DELETE", "/api/gallery/tags/teal"))
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::NO_CONTENT);
        let mut updated: Vec<_> = drain_events(&mut events)
            .into_iter()
            .filter_map(|e| match e {
                mold_core::ServerEvent::GalleryUpdated { filename, .. } => Some(filename),
                _ => None,
            })
            .collect();
        updated.sort();
        assert_eq!(updated, vec!["a.png", "b.png"]);
        let resp = app
            .clone()
            .oneshot(empty_request("DELETE", "/api/gallery/tags/teal"))
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::NOT_FOUND);
        let tags = gallery_rows(&app, "/api/gallery/tags").await;
        assert_eq!(tags.len(), 1);
        assert_eq!(tags[0]["name"], "birds");
    }

    #[tokio::test]
    async fn organization_routes_answer_501_without_the_metadata_db() {
        let dir = tempfile::tempdir().unwrap();
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
        let app = app_with_state(state);
        for (method, uri) in [
            ("GET", "/api/gallery/collections"),
            ("GET", "/api/gallery/tags"),
            ("POST", "/api/gallery/trash/sweep"),
            ("DELETE", "/api/gallery/trash"),
        ] {
            let resp = app
                .clone()
                .oneshot(empty_request(method, uri))
                .await
                .unwrap();
            assert_eq!(resp.status(), StatusCode::NOT_IMPLEMENTED, "{method} {uri}");
        }
        let resp = app
            .clone()
            .oneshot(json_request(
                "PATCH",
                "/api/gallery/image/x.png",
                serde_json::json!({"favorite": true}),
            ))
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::NOT_IMPLEMENTED);
        let caps = app
            .oneshot(empty_request("GET", "/api/capabilities"))
            .await
            .unwrap();
        let caps = json_body(caps).await;
        assert_eq!(caps["gallery"]["can_delete"], true);
        assert_eq!(caps["gallery"]["trash"]["enabled"], false);
        assert!(caps["gallery"].get("organize").is_none());
    }

    #[tokio::test]
    async fn capabilities_advertise_trash_and_organize_with_the_metadata_db() {
        let dir = tempfile::tempdir().unwrap();
        let (state, _db) = organized_state(dir.path());
        let app = app_with_state(state);
        let caps = app
            .oneshot(empty_request("GET", "/api/capabilities"))
            .await
            .unwrap();
        let caps = json_body(caps).await;
        assert_eq!(caps["gallery"]["can_delete"], true);
        assert_eq!(caps["gallery"]["trash"]["enabled"], true);
        assert_eq!(caps["gallery"]["trash"]["retention_days"], 30);
        assert_eq!(caps["gallery"]["organize"], true);
    }

    /// `DELETE /api/gallery/image/:filename` moves to the trash when the
    /// metadata DB is available: bytes land in `.trash/`, a tombstone sits
    /// beside them, the row is flagged (not dropped), and `gallery_trashed`
    /// is announced.
    #[tokio::test]
    async fn gallery_delete_moves_to_trash_when_db_available() {
        let dir = tempfile::tempdir().unwrap();
        let (state, db) = organized_state(dir.path());
        seed_print(&db, dir.path(), "doomed.png", Some("Doomed"));
        let mdb = db.as_ref().as_ref().unwrap();
        mdb.add_tags(dir.path(), "doomed.png", &["keep".into()])
            .unwrap();
        let mut events = state.events.subscribe();
        let app = app_with_state(state);

        let resp = app
            .clone()
            .oneshot(empty_request("DELETE", "/api/gallery/image/doomed.png"))
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::NO_CONTENT);
        let trash = mold_db::trash_dir(dir.path());
        assert!(!dir.path().join("doomed.png").exists(), "live bytes moved");
        assert!(trash.join("doomed.png").is_file(), "bytes are in .trash/");
        let tombstone =
            mold_db::trash::read_tombstone(&mold_db::tombstone_path(&trash, "doomed.png")).unwrap();
        assert_eq!(tombstone.title.as_deref(), Some("Doomed"));
        assert_eq!(tombstone.tags, vec!["keep".to_string()]);
        let row = mdb.get(dir.path(), "doomed.png").unwrap().unwrap();
        assert!(row.trashed_at_ms.is_some(), "row is flagged, not dropped");
        assert!(matches!(
            events.try_recv().unwrap(),
            mold_core::ServerEvent::GalleryTrashed { filename } if filename == "doomed.png"
        ));

        // Listing: gone from the library, present in the trash with purge_at.
        assert!(gallery_rows(&app, "/api/gallery").await.is_empty());
        let trashed = gallery_rows(&app, "/api/gallery?view=trash").await;
        assert_eq!(trashed.len(), 1);
        assert_eq!(trashed[0]["tags"], serde_json::json!(["keep"]));
        assert!(
            trashed[0]["purge_at"].as_u64().unwrap() > trashed[0]["trashed_at"].as_u64().unwrap()
        );

        // Trashing again is idempotent and silent.
        let resp = app
            .clone()
            .oneshot(empty_request("DELETE", "/api/gallery/image/doomed.png"))
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::NO_CONTENT);
        assert!(events.try_recv().is_err());

        // A print that exists nowhere → 404 (the DB-less path keeps 204).
        let resp = app
            .oneshot(empty_request("DELETE", "/api/gallery/image/ghost.png"))
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::NOT_FOUND);
    }

    #[tokio::test]
    async fn gallery_delete_permanent_purges_trashed_and_hard_deletes_live() {
        let dir = tempfile::tempdir().unwrap();
        let (state, db) = organized_state(dir.path());
        seed_print(&db, dir.path(), "binned.png", None);
        seed_print(&db, dir.path(), "live.png", None);
        let mdb = db.as_ref().as_ref().unwrap();
        let mut events = state.events.subscribe();
        let app = app_with_state(state);

        // Trash, then purge.
        let resp = app
            .clone()
            .oneshot(empty_request("DELETE", "/api/gallery/image/binned.png"))
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::NO_CONTENT);
        let trash = mold_db::trash_dir(dir.path());
        assert!(trash.join("binned.png").is_file());
        drain_events(&mut events);
        let resp = app
            .clone()
            .oneshot(empty_request(
                "DELETE",
                "/api/gallery/image/binned.png?permanent=true",
            ))
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::NO_CONTENT);
        assert!(!trash.join("binned.png").exists());
        assert!(!mold_db::tombstone_path(&trash, "binned.png").exists());
        assert!(mdb.get(dir.path(), "binned.png").unwrap().is_none());
        assert!(matches!(
            events.try_recv().unwrap(),
            mold_core::ServerEvent::GalleryRemoved { filename } if filename == "binned.png"
        ));

        // Permanent on a live print is today's hard delete.
        let resp = app
            .clone()
            .oneshot(empty_request(
                "DELETE",
                "/api/gallery/image/live.png?permanent=true",
            ))
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::NO_CONTENT);
        assert!(!dir.path().join("live.png").exists());
        assert!(!trash.join("live.png").exists());
        assert!(mdb.get(dir.path(), "live.png").unwrap().is_none());
        assert!(matches!(
            events.try_recv().unwrap(),
            mold_core::ServerEvent::GalleryRemoved { filename } if filename == "live.png"
        ));
        assert!(gallery_rows(&app, "/api/gallery?view=trash")
            .await
            .is_empty());
    }

    #[tokio::test]
    async fn gallery_trash_bulk_and_restore_round_trip() {
        let dir = tempfile::tempdir().unwrap();
        let (state, db) = organized_state(dir.path());
        seed_print(&db, dir.path(), "a.png", Some("Alpha"));
        seed_print(&db, dir.path(), "b.png", None);
        let mdb = db.as_ref().as_ref().unwrap();
        let mut events = state.events.subscribe();
        let app = app_with_state(state);

        let resp = app
            .clone()
            .oneshot(json_request(
                "POST",
                "/api/gallery/trash",
                serde_json::json!({"filenames": ["a.png", "b.png"]}),
            ))
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::NO_CONTENT);
        let trashed: Vec<_> = drain_events(&mut events)
            .into_iter()
            .filter_map(|e| match e {
                mold_core::ServerEvent::GalleryTrashed { filename } => Some(filename),
                _ => None,
            })
            .collect();
        assert_eq!(trashed, vec!["a.png", "b.png"]);
        assert_eq!(gallery_rows(&app, "/api/gallery?view=trash").await.len(), 2);
        assert!(gallery_rows(&app, "/api/gallery").await.is_empty());

        // Bulk with an unknown name stops there, naming it; earlier ones stand.
        seed_print(&db, dir.path(), "c.png", None);
        let resp = app
            .clone()
            .oneshot(json_request(
                "POST",
                "/api/gallery/trash",
                serde_json::json!({"filenames": ["c.png", "ghost.png"]}),
            ))
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::NOT_FOUND);
        assert!(json_body(resp).await["error"]
            .as_str()
            .unwrap()
            .starts_with("ghost.png"));
        assert!(mold_db::trash_dir(dir.path()).join("c.png").is_file());
        drain_events(&mut events);

        // Restore `a.png`: live again, row un-flagged, tombstone gone,
        // gallery_restored carries the enriched row.
        let resp = app
            .clone()
            .oneshot(json_request(
                "POST",
                "/api/gallery/trash/restore",
                serde_json::json!({"filenames": ["a.png"]}),
            ))
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::NO_CONTENT);
        assert!(dir.path().join("a.png").is_file());
        let trash = mold_db::trash_dir(dir.path());
        assert!(!trash.join("a.png").exists());
        assert!(!mold_db::tombstone_path(&trash, "a.png").exists());
        assert!(mdb
            .get(dir.path(), "a.png")
            .unwrap()
            .unwrap()
            .trashed_at_ms
            .is_none());
        match events.try_recv().unwrap() {
            mold_core::ServerEvent::GalleryRestored { filename, image } => {
                assert_eq!(filename, "a.png");
                let image = image.expect("restore carries the row");
                assert_eq!(image.title.as_deref(), Some("Alpha"));
                assert!(image.trashed_at.is_none());
            }
            other => panic!("expected gallery_restored, got {other:?}"),
        }
        let library = gallery_rows(&app, "/api/gallery").await;
        assert_eq!(library.len(), 1);
        assert_eq!(library[0]["filename"], "a.png");
        assert_eq!(library[0]["title"], "Alpha");

        // Restoring a live print → 409; restoring over a live file → 409.
        let resp = app
            .clone()
            .oneshot(json_request(
                "POST",
                "/api/gallery/trash/restore",
                serde_json::json!({"filenames": ["a.png"]}),
            ))
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::CONFLICT);
        std::fs::write(dir.path().join("b.png"), b"someone else's b").unwrap();
        let resp = app
            .clone()
            .oneshot(json_request(
                "POST",
                "/api/gallery/trash/restore",
                serde_json::json!({"filenames": ["b.png"]}),
            ))
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::CONFLICT);
        assert_eq!(json_body(resp).await["code"], "GALLERY_RESTORE_CONFLICT");
        assert!(
            trash.join("b.png").is_file(),
            "conflict leaves the trash untouched"
        );
    }

    /// A print published through the committed archive (the import route)
    /// keeps its identity through trash and restore: the archive retires the
    /// exact entry on trash, listings never resurrect it, and restore re-homes
    /// the same provenance (non-synthetic metadata) rather than re-scanning.
    #[tokio::test]
    async fn gallery_trash_and_restore_preserve_committed_archive_identity() {
        let dir = tempfile::tempdir().unwrap();
        let (state, db) = organized_state(dir.path());
        let gate = state.gallery_publication_gate.clone();
        let app = app_with_state(state);
        let bytes = minimal_png();
        let metadata = output_metadata("archived provenance");
        let resp = app
            .clone()
            .oneshot(
                Request::put("/api/gallery/import/print.png")
                    .header("content-type", "application/vnd.mold.gallery-import")
                    .body(Body::from(gallery_import_body(Some(&metadata), &bytes)))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::CREATED);
        let index = gate.committed_archive_index(dir.path()).unwrap();
        assert!(
            index.get("print.png").is_some(),
            "import publishes into the archive"
        );

        let resp = app
            .clone()
            .oneshot(empty_request("DELETE", "/api/gallery/image/print.png"))
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::NO_CONTENT);
        let index = gate.committed_archive_index(dir.path()).unwrap();
        assert!(index.get("print.png").is_none());
        assert!(
            index.is_retired("print.png"),
            "trash retires the archive entry"
        );
        assert!(
            index.retired_entries.contains_key("print.png"),
            "the retired identity is retained for restore"
        );
        assert!(mold_db::trash_dir(dir.path()).join("print.png").is_file());
        assert!(gallery_rows(&app, "/api/gallery").await.is_empty());
        let trashed = gallery_rows(&app, "/api/gallery?view=trash").await;
        assert_eq!(trashed.len(), 1);
        assert_eq!(trashed[0]["metadata"]["prompt"], "archived provenance");

        // A restart must not resurrect or unlink the trashed print.
        let (restarted, _) = organized_state(dir.path());
        let restarted_gate = restarted.gallery_publication_gate.clone();
        crate::batch_transaction::recover_transactions(dir.path(), &restarted_gate, Arc::new(None))
            .await
            .unwrap();
        let index = restarted_gate.committed_archive_index(dir.path()).unwrap();
        assert!(index.is_retired("print.png"));
        assert!(mold_db::trash_dir(dir.path()).join("print.png").is_file());

        let resp = app
            .clone()
            .oneshot(json_request(
                "POST",
                "/api/gallery/trash/restore",
                serde_json::json!({"filenames": ["print.png"]}),
            ))
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::NO_CONTENT);
        let index = gate.committed_archive_index(dir.path()).unwrap();
        assert!(
            index.get("print.png").is_some(),
            "restore re-homes the entry"
        );
        assert!(!index.is_retired("print.png"));
        assert!(!index.retired_entries.contains_key("print.png"));
        assert_eq!(std::fs::read(dir.path().join("print.png")).unwrap(), bytes);
        let library = gallery_rows(&app, "/api/gallery").await;
        assert_eq!(library.len(), 1);
        assert_eq!(library[0]["metadata"]["prompt"], "archived provenance");
        assert_ne!(library[0]["metadata_synthetic"], true);
        assert!(db
            .as_ref()
            .as_ref()
            .unwrap()
            .get(dir.path(), "print.png")
            .unwrap()
            .unwrap()
            .trashed_at_ms
            .is_none());

        // Trashed bytes that were tampered with are refused on restore.
        let resp = app
            .clone()
            .oneshot(empty_request("DELETE", "/api/gallery/image/print.png"))
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::NO_CONTENT);
        std::fs::write(
            mold_db::trash_dir(dir.path()).join("print.png"),
            b"tampered while trashed",
        )
        .unwrap();
        let resp = app
            .oneshot(json_request(
                "POST",
                "/api/gallery/trash/restore",
                serde_json::json!({"filenames": ["print.png"]}),
            ))
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::CONFLICT);
        assert_eq!(json_body(resp).await["code"], "GALLERY_RESTORE_CONFLICT");
    }

    #[tokio::test]
    async fn gallery_trash_sweep_and_empty_honor_retention() {
        let dir = tempfile::tempdir().unwrap();
        let (state, db) = organized_state(dir.path());
        seed_print(&db, dir.path(), "old.png", None);
        seed_print(&db, dir.path(), "fresh.png", None);
        seed_print(&db, dir.path(), "live.png", None);
        let mdb = db.as_ref().as_ref().unwrap();
        let mut events = state.events.subscribe();
        let app = app_with_state(state.clone());

        let resp = app
            .clone()
            .oneshot(json_request(
                "POST",
                "/api/gallery/trash",
                serde_json::json!({"filenames": ["old.png", "fresh.png"]}),
            ))
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::NO_CONTENT);
        // Age `old.png` past the 30-day default.
        let ancient = mold_core::time::now_epoch_ms() - 31 * mold_db::trash::DAY_MS;
        mdb.mark_trashed(dir.path(), "old.png", ancient).unwrap();
        drain_events(&mut events);

        let resp = app
            .clone()
            .oneshot(empty_request("POST", "/api/gallery/trash/sweep"))
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
        let sweep = json_body(resp).await;
        assert_eq!(sweep["purged"], 1);
        assert_eq!(sweep["remaining"], 1);
        let trash = mold_db::trash_dir(dir.path());
        assert!(!trash.join("old.png").exists());
        assert!(!mold_db::tombstone_path(&trash, "old.png").exists());
        assert!(trash.join("fresh.png").is_file());
        assert!(mdb.get(dir.path(), "old.png").unwrap().is_none());
        assert!(matches!(
            events.try_recv().unwrap(),
            mold_core::ServerEvent::GalleryRemoved { filename } if filename == "old.png"
        ));

        // Retention 0 keeps forever: even an ancient row survives a sweep.
        mdb.mark_trashed(dir.path(), "fresh.png", ancient).unwrap();
        state.config.write().await.gallery.trash_retention_days = 0;
        let result = crate::gallery_trash::sweep_trash_once(&state)
            .await
            .unwrap();
        assert_eq!(result.purged, 0);
        assert_eq!(result.remaining, 1);
        assert!(trash.join("fresh.png").is_file());
        // And the capability reflects the live value.
        let caps = json_body(
            app.clone()
                .oneshot(empty_request("GET", "/api/capabilities"))
                .await
                .unwrap(),
        )
        .await;
        assert_eq!(caps["gallery"]["trash"]["retention_days"], 0);
        assert!(
            gallery_rows(&app, "/api/gallery?view=trash").await[0]
                .get("purge_at")
                .is_none(),
            "keep-forever rows advertise no purge date"
        );

        // Empty trash purges everything trashed and nothing live.
        let resp = app
            .clone()
            .oneshot(empty_request("DELETE", "/api/gallery/trash"))
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
        assert_eq!(json_body(resp).await["purged"], 1);
        assert!(!trash.join("fresh.png").exists());
        assert!(dir.path().join("live.png").is_file());
        assert!(gallery_rows(&app, "/api/gallery?view=trash")
            .await
            .is_empty());
        assert_eq!(gallery_rows(&app, "/api/gallery").await.len(), 1);
    }

    /// The sweeper and reconcile must agree: after a trash + restart-style
    /// reconcile, the row is still trashed (not dropped, not resurrected) and
    /// the sweeper still purges it once expired.
    #[tokio::test]
    async fn trash_sweeper_agrees_with_reconcile_after_restart() {
        let dir = tempfile::tempdir().unwrap();
        let (state, db) = organized_state(dir.path());
        seed_print(&db, dir.path(), "old.png", None);
        let mdb = db.as_ref().as_ref().unwrap();
        let app = app_with_state(state.clone());
        let resp = app
            .clone()
            .oneshot(empty_request("DELETE", "/api/gallery/image/old.png"))
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::NO_CONTENT);
        let ancient = mold_core::time::now_epoch_ms() - 31 * mold_db::trash::DAY_MS;
        mdb.mark_trashed(dir.path(), "old.png", ancient).unwrap();

        let stats = mdb.reconcile(dir.path()).unwrap();
        assert_eq!(
            stats.removed, 0,
            "reconcile keeps trashed rows whose bytes sit in .trash/"
        );
        assert_eq!(stats.trashed_kept, 1);
        assert!(mdb
            .get(dir.path(), "old.png")
            .unwrap()
            .unwrap()
            .trashed_at_ms
            .is_some());

        let result = crate::gallery_trash::sweep_trash_once(&state)
            .await
            .unwrap();
        assert_eq!(result.purged, 1);
        assert_eq!(result.remaining, 0);
        assert!(mdb.get(dir.path(), "old.png").unwrap().is_none());
        let stats = mdb.reconcile(dir.path()).unwrap();
        assert_eq!(stats.trashed_imported, 0, "nothing orphaned in .trash/");
    }

    /// The sweeper task runs its first pass once `ready` resolves and stops
    /// when its token is cancelled.
    #[tokio::test]
    async fn trash_sweeper_task_runs_after_ready_and_stops_on_cancel() {
        let dir = tempfile::tempdir().unwrap();
        let (state, db) = organized_state(dir.path());
        seed_print(&db, dir.path(), "old.png", None);
        let mdb = db.as_ref().as_ref().unwrap();
        let app = app_with_state(state.clone());
        let resp = app
            .oneshot(empty_request("DELETE", "/api/gallery/image/old.png"))
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::NO_CONTENT);
        let ancient = mold_core::time::now_epoch_ms() - 31 * mold_db::trash::DAY_MS;
        mdb.mark_trashed(dir.path(), "old.png", ancient).unwrap();

        let shutdown = tokio_util::sync::CancellationToken::new();
        let (ready_tx, ready_rx) = tokio::sync::oneshot::channel();
        let handle = crate::gallery_trash::spawn_trash_sweeper(
            state.clone(),
            shutdown.clone(),
            Some(ready_rx),
        );
        tokio::time::sleep(Duration::from_millis(50)).await;
        assert!(
            mdb.get(dir.path(), "old.png").unwrap().is_some(),
            "no pass before reconcile reports ready"
        );
        ready_tx.send(()).unwrap();
        let deadline = tokio::time::Instant::now() + Duration::from_secs(5);
        while mdb.get(dir.path(), "old.png").unwrap().is_some() {
            assert!(
                tokio::time::Instant::now() < deadline,
                "startup pass never ran"
            );
            tokio::time::sleep(Duration::from_millis(10)).await;
        }
        shutdown.cancel();
        tokio::time::timeout(Duration::from_secs(2), handle)
            .await
            .expect("sweeper exits on cancel")
            .unwrap();
    }

    #[tokio::test]
    async fn gallery_media_routes_resolve_trashed_prints() {
        let _mold_home =
            EnvVarGuard::set("MOLD_HOME", tempfile::tempdir().unwrap().path().as_os_str());
        let dir = tempfile::tempdir().unwrap();
        let (state, db) = organized_state(dir.path());
        seed_print(&db, dir.path(), "binned.png", None);
        let bytes = std::fs::read(dir.path().join("binned.png")).unwrap();
        let app = app_with_state(state);
        let resp = app
            .clone()
            .oneshot(empty_request("DELETE", "/api/gallery/image/binned.png"))
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::NO_CONTENT);

        let resp = app
            .clone()
            .oneshot(empty_request("GET", "/api/gallery/image/binned.png"))
            .await
            .unwrap();
        assert_eq!(
            resp.status(),
            StatusCode::OK,
            "a trashed print still streams"
        );
        let body = axum::body::to_bytes(resp.into_body(), 1024 * 1024)
            .await
            .unwrap();
        assert_eq!(body.as_ref(), bytes.as_slice());

        let resp = app
            .clone()
            .oneshot(empty_request("GET", "/api/gallery/thumbnail/binned.png"))
            .await
            .unwrap();
        assert_eq!(
            resp.status(),
            StatusCode::OK,
            "a trashed print still has a thumbnail"
        );

        // Purged → gone everywhere.
        let resp = app
            .clone()
            .oneshot(empty_request(
                "DELETE",
                "/api/gallery/image/binned.png?permanent=true",
            ))
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::NO_CONTENT);
        let resp = app
            .oneshot(empty_request("GET", "/api/gallery/image/binned.png"))
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::NOT_FOUND);
    }

    /// Every new organization/trash mutator and listing must wait behind
    /// the atomic publication writer exactly like the historical routes.
    #[tokio::test]
    async fn every_organization_and_trash_route_waits_for_atomic_publication_writer() {
        let dir = tempfile::tempdir().unwrap();
        let (state, db) = organized_state(dir.path());
        seed_print(&db, dir.path(), "a.png", None);
        let gate = state.gallery_publication_gate.clone();
        let writer = gate.write().await;
        let app = app_with_state(state);
        let cases = [
            (
                json_request(
                    "PATCH",
                    "/api/gallery/image/a.png",
                    serde_json::json!({"favorite": true}),
                ),
                StatusCode::OK,
            ),
            (
                json_request(
                    "POST",
                    "/api/gallery/organize",
                    serde_json::json!({"filenames": ["a.png"], "favorite": true}),
                ),
                StatusCode::NO_CONTENT,
            ),
            (
                empty_request("GET", "/api/gallery/collections"),
                StatusCode::OK,
            ),
            (
                json_request(
                    "POST",
                    "/api/gallery/collections",
                    serde_json::json!({"name": "Gate"}),
                ),
                StatusCode::CREATED,
            ),
            (
                empty_request("GET", "/api/gallery/collections/nope"),
                StatusCode::NOT_FOUND,
            ),
            (
                json_request(
                    "PATCH",
                    "/api/gallery/collections/nope",
                    serde_json::json!({"name": "x"}),
                ),
                StatusCode::NOT_FOUND,
            ),
            (
                empty_request("DELETE", "/api/gallery/collections/nope"),
                StatusCode::NOT_FOUND,
            ),
            (
                json_request(
                    "PUT",
                    "/api/gallery/collections/nope/items",
                    serde_json::json!({"add": ["a.png"]}),
                ),
                StatusCode::NOT_FOUND,
            ),
            (empty_request("GET", "/api/gallery/tags"), StatusCode::OK),
            (
                json_request(
                    "PATCH",
                    "/api/gallery/tags/nope",
                    serde_json::json!({"name": "x"}),
                ),
                StatusCode::NOT_FOUND,
            ),
            (
                empty_request("DELETE", "/api/gallery/tags/nope"),
                StatusCode::NOT_FOUND,
            ),
            (
                empty_request("GET", "/api/gallery?view=trash"),
                StatusCode::OK,
            ),
            (
                json_request(
                    "POST",
                    "/api/gallery/trash",
                    serde_json::json!({"filenames": ["ghost.png"]}),
                ),
                StatusCode::NOT_FOUND,
            ),
            (
                json_request(
                    "POST",
                    "/api/gallery/trash/restore",
                    serde_json::json!({"filenames": ["ghost.png"]}),
                ),
                StatusCode::NOT_FOUND,
            ),
            (
                empty_request("DELETE", "/api/gallery/trash"),
                StatusCode::OK,
            ),
            (
                empty_request("POST", "/api/gallery/trash/sweep"),
                StatusCode::OK,
            ),
            (
                empty_request("DELETE", "/api/gallery/image/a.png?permanent=false"),
                StatusCode::NO_CONTENT,
            ),
        ];
        let mut requests: Vec<_> = cases
            .into_iter()
            .map(|(request, expected)| {
                let app = app.clone();
                (
                    expected,
                    tokio::spawn(async move { app.oneshot(request).await.unwrap() }),
                )
            })
            .collect();
        for (_, request) in &mut requests {
            assert!(
                tokio::time::timeout(Duration::from_millis(20), request)
                    .await
                    .is_err(),
                "an organization/trash route ran while the publication writer was held"
            );
        }
        drop(writer);
        for (expected, request) in requests {
            let response = tokio::time::timeout(Duration::from_secs(5), request)
                .await
                .unwrap()
                .unwrap();
            assert_eq!(response.status(), expected, "{:?}", response);
        }
    }

    /// A titled request refuses control characters before any model work.
    #[test]
    fn generate_request_title_is_validated_at_admission() {
        let mut req: GenerateRequest = serde_json::from_str(
            r#"{"prompt":"a cat","model":"flux-dev:q4","width":512,"height":512,"steps":4}"#,
        )
        .unwrap();
        req.title = Some("bad\u{0007}title".into());
        assert!(crate::routes::validate_generate_request(&req, None).is_err());
        req.title = Some("Smurf village".into());
        assert!(crate::routes::validate_generate_request(&req, None).is_ok());
    }

    /// The desktop auto-save mirror (`PUT /api/gallery/import/:filename`)
    /// seeds the row title from the imported metadata, so a mirrored titled
    /// print is titled on every host.
    #[tokio::test]
    async fn gallery_import_seeds_row_title_from_metadata() {
        let dir = tempfile::tempdir().unwrap();
        let (state, db) = organized_state(dir.path());
        let app = app_with_state(state);
        let mut metadata = output_metadata("mirrored");
        metadata.title = Some("Smurf village".into());
        let resp = app
            .clone()
            .oneshot(
                Request::put(
                    "/api/gallery/import/mold-flux-dev-q4-1700000000000~smurf-village.png",
                )
                .header("content-type", "application/vnd.mold.gallery-import")
                .body(Body::from(gallery_import_body(
                    Some(&metadata),
                    &minimal_png(),
                )))
                .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::CREATED);
        let row = db
            .as_ref()
            .as_ref()
            .unwrap()
            .get(
                dir.path(),
                "mold-flux-dev-q4-1700000000000~smurf-village.png",
            )
            .unwrap()
            .unwrap();
        assert_eq!(row.title.as_deref(), Some("Smurf village"));
        let rows = gallery_rows(&app, "/api/gallery").await;
        assert_eq!(rows[0]["title"], "Smurf village");
        assert_eq!(rows[0]["metadata"]["title"], "Smurf village");
    }

    /// Importing a filename that is currently in the trash republishes it
    /// live: the trashed copy is purged first so the row, the `.trash/` bytes,
    /// and the tombstone never disagree with the freshly published file.
    #[tokio::test]
    async fn gallery_import_over_trashed_name_republishes_live() {
        let dir = tempfile::tempdir().unwrap();
        let (state, db) = organized_state(dir.path());
        seed_print(&db, dir.path(), "again.png", Some("First"));
        let app = app_with_state(state);
        let resp = app
            .clone()
            .oneshot(empty_request("DELETE", "/api/gallery/image/again.png"))
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::NO_CONTENT);
        let trash = mold_db::trash_dir(dir.path());
        assert!(trash.join("again.png").is_file());

        let mut metadata = output_metadata("again");
        metadata.title = Some("Second".into());
        let resp = app
            .clone()
            .oneshot(
                Request::put("/api/gallery/import/again.png")
                    .header("content-type", "application/vnd.mold.gallery-import")
                    .body(Body::from(gallery_import_body(
                        Some(&metadata),
                        &minimal_png(),
                    )))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::CREATED);

        assert!(dir.path().join("again.png").is_file(), "published live");
        assert!(!trash.join("again.png").exists(), "trashed copy purged");
        assert!(
            !mold_db::tombstone_path(&trash, "again.png").exists(),
            "tombstone removed"
        );
        let row = db
            .as_ref()
            .as_ref()
            .unwrap()
            .get(dir.path(), "again.png")
            .unwrap()
            .unwrap();
        assert!(row.trashed_at_ms.is_none(), "row is live again");
        assert_eq!(row.title.as_deref(), Some("Second"));
        assert_eq!(gallery_rows(&app, "/api/gallery").await.len(), 1);
        assert!(gallery_rows(&app, "/api/gallery?view=trash")
            .await
            .is_empty());
    }
}
