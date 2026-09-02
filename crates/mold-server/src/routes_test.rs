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
    fn env_lock() -> std::sync::MutexGuard<'static, ()> {
        crate::test_support::env_lock()
    }

    struct EnvVarGuard {
        _lock: std::sync::MutexGuard<'static, ()>,
        name: &'static str,
        previous: Option<std::ffi::OsString>,
    }

    impl EnvVarGuard {
        fn set(name: &'static str, value: &std::ffi::OsStr) -> Self {
            let lock = env_lock();
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
        let body = axum::body::to_bytes(resp.into_body(), 8 * 1024 * 1024)
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
            "frames": mold_core::minimax_h3::REVIEWED_COMPACT_FRAMES,
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
        cancellation: Option<mold_inference::InferenceCancellationToken>,
        ignore_cancellation: bool,
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
                cancellation: None,
                ignore_cancellation: false,
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
                cancellation: None,
                ignore_cancellation: false,
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
                cancellation: None,
                ignore_cancellation: false,
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
                cancellation: None,
                ignore_cancellation: false,
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
                cancellation: None,
                ignore_cancellation: false,
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
                cancellation: None,
                ignore_cancellation: false,
                emit_load_progress: false,
                progress_callback: None,
            }
        }

        /// Simulates an engine family that finishes a result after its token
        /// was revoked. Publication fencing must still discard those bytes.
        fn blocking_generate_ignoring_cancellation(blocker: Arc<GenerateBlocker>) -> Self {
            let mut engine = Self::blocking_generate(blocker);
            engine.ignore_cancellation = true;
            engine
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
                cancellation: None,
                ignore_cancellation: false,
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
            if !self.ignore_cancellation {
                if let Some(cancellation) = &self.cancellation {
                    cancellation.checkpoint()?;
                }
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
                mesh: None,
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

        fn set_cancellation_token(&mut self, token: mold_inference::InferenceCancellationToken) {
            self.cancellation = Some(token);
        }

        fn clear_cancellation_token(&mut self) {
            self.cancellation = None;
        }
    }

    /// Create an app with a running queue worker (needed for generate endpoints).
    /// A DURABLE-ready host, which is the only kind that generates.
    ///
    /// The gallery lands in the returned `TempDir`, which the caller must keep
    /// alive for the life of the state — dropping it removes the output
    /// directory and the queue-owner claim underneath a running feeder.
    ///
    /// Every generation fixture in this module goes through here, because
    /// there is no non-durable path left to submit on: `/api/generate`,
    /// `/api/generate/stream`, and `/api/generation-batches` all admit through
    /// `admit_batch`, and a host missing any of the four readiness conjuncts
    /// answers `503 DURABLE_ADMISSION_UNAVAILABLE` before it reads the request.
    fn durable_test_state(
        engine: MockEngine,
    ) -> (
        AppState,
        tokio::sync::mpsc::Receiver<crate::state::GenerationJob>,
        tempfile::TempDir,
    ) {
        let root = tempfile::tempdir().unwrap();
        let db = Arc::new(Some(mold_db::MetadataDb::open_in_memory().unwrap()));
        let (mut state, rx) = durable_state_with_engine(db, root.path(), engine);
        install_authoritative_v2(&mut state);
        (state, rx, root)
    }

    /// Run the two tasks a durable admission needs to reach a worker: the
    /// feeder that hydrates the SQLite row into the runtime queue, and the
    /// queue worker itself. The direct routes block on the feeder claiming the
    /// row, so a generation test without a feeder hangs rather than fails.
    fn spawn_durable_runtime(
        state: &AppState,
        rx: tokio::sync::mpsc::Receiver<crate::state::GenerationJob>,
    ) {
        spawn_durable_feeder(state);
        tokio::spawn(crate::queue::run_queue_worker(rx, state.clone()));
    }

    /// The feeder alone, for a test that owns its own worker — or that wants
    /// the job to sit in the runtime queue rather than run.
    fn spawn_durable_feeder(state: &AppState) {
        tokio::spawn(crate::durable_queue_feeder::spawn(
            state.clone(),
            tokio_util::sync::CancellationToken::new(),
        ));
    }

    fn app_with(engine: MockEngine) -> (axum::Router, tempfile::TempDir) {
        let (state, rx, root) = durable_test_state(engine);
        spawn_durable_runtime(&state, rx);
        (create_router(state), root)
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
        let _env = env_lock();
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
                reclaimable_zfs_arc: None,
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
            let lock = env_lock();
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
            ephemeral: false,
        }
    }

    #[tokio::test]
    #[allow(clippy::await_holding_lock)]
    async fn chain_validation_returns_normalized_plan_without_job_storage() {
        let _lock = env_lock();
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

    fn gpu_worker_stub_with_stable_id(
        ordinal: usize,
        stable_id: &str,
    ) -> Arc<crate::gpu_pool::GpuWorker> {
        let (worker, _receiver) = gpu_worker_stub_with_receiver(ordinal);
        let Ok(mut worker) = Arc::try_unwrap(worker) else {
            unreachable!("fresh worker has one owner")
        };
        worker.gpu.stable_id = Some(stable_id.to_string());
        Arc::new(worker)
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

    fn install_worker_registry_with_metadata(
        state: &mut AppState,
        metadata_db: Arc<Option<mold_db::MetadataDb>>,
    ) {
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
            metadata_db,
        ));
    }

    fn install_worker_registry(state: &mut AppState) {
        install_worker_registry_with_metadata(state, Arc::new(None));
    }

    fn install_authoritative_v2(state: &mut AppState) {
        install_dispatch_mode(state, crate::dispatch_mode::DispatchMode::V2);
    }

    /// The non-V2 half of the same helper. Every durable route test installs
    /// V2, which left the legacy and observe hosts with zero coverage on any
    /// generation route — the blind axis that let `direct_durable_admission`
    /// omit the `v2_authoritative` conjunct entirely.
    fn install_dispatch_mode(state: &mut AppState, mode: crate::dispatch_mode::DispatchMode) {
        let (scheduled_tx, _scheduled_rx) = tokio::sync::mpsc::channel(1);
        state.scheduled_work = crate::scheduler::ScheduledWorkHandle::for_mode(scheduled_tx, mode);
    }

    fn app_with_worker_pool(
        engine: MockEngine,
        ordinals: &[usize],
    ) -> (axum::Router, tempfile::TempDir) {
        let (mut state, _rx, root) = durable_test_state(engine);
        state.gpu_pool = Arc::new(crate::gpu_pool::GpuPool {
            workers: ordinals
                .iter()
                .copied()
                .map(gpu_worker_stub)
                .collect::<Vec<_>>()
                .into(),
        });
        install_worker_registry(&mut state);
        (create_router(state), root)
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
            video_only: None,
            attention_path: None,
            int8_arm: None,
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
            mesh: None,
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
            pipeline_requested: None,
            duration_prediction_requested: None,
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
            id_image_name: None,
            id_image_sha256: None,
            id_weight: None,
            id_start_step: None,
            id_image_names: None,
            id_image_sha256s: None,
            true_cfg: None,
            cfg_start_step: None,
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
        let (app, _gallery_root) = app_with(MockEngine::ready());
        let resp = app
            .oneshot(Request::get("/health").body(Body::empty()).unwrap())
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
    }

    /// A default fixture has no durable queue owner at all, so media
    /// durability is not applicable rather than degraded. Reporting every
    /// `MOLD_DB_DISABLE` host as degraded would make the field useless.
    #[tokio::test]
    async fn health_reports_ok_when_no_durable_queue_owner_exists() {
        let (app, _gallery_root) = app_with(MockEngine::ready());
        let resp = app
            .oneshot(Request::get("/health").body(Body::empty()).unwrap())
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
        let body = json_body(resp).await;
        assert_eq!(body["status"], "ok");
        assert!(body.get("degraded").is_none(), "{body}");
    }

    /// A server that actually offers restart-safe request media: gallery
    /// output on, an authoritative V2 scheduler, and a claimed queue owner.
    /// Applicability is deliberately a precondition of the degraded state, so
    /// a fixture missing any of it would report "not applicable" and pass a
    /// degradation test for the wrong reason.
    fn durable_media_applicable_state(
        root: &std::path::Path,
    ) -> (
        AppState,
        tokio::sync::mpsc::Receiver<crate::state::GenerationJob>,
    ) {
        let db = std::sync::Arc::new(Some(mold_db::MetadataDb::open_in_memory().unwrap()));
        let (mut state, rx, _gallery_root) = durable_test_state(MockEngine::ready());
        spawn_durable_feeder(&state);
        state.output_disabled_override = false;
        state
            .config
            .try_write()
            .expect("fresh test config")
            .output_dir = Some(root.join("gallery").to_string_lossy().into_owned());
        state.scheduled_work = crate::scheduler::ScheduledWorkHandle::for_runtime(
            tokio::sync::mpsc::channel(1).0,
            crate::dispatch_mode::DispatchMode::V2,
            true,
            false,
        );
        state.queue_journal = Arc::new(crate::queue_journal::QueueJournal::new(
            db,
            Some(root),
            "test-instance",
        ));
        assert!(
            state.queue_journal.is_enabled(),
            "the fixture must claim a durable queue owner"
        );
        (state, rx)
    }

    /// The whole point of #1402: a widened `queue-media` mode turned durable
    /// admission off for the life of the process with a single startup log
    /// line as the only evidence. `/health` must keep saying so, and must
    /// still answer 200 — generation is unaffected, so failing the check would
    /// pull a working server out of a load balancer.
    #[tokio::test]
    async fn health_names_durable_media_while_it_is_degraded() {
        let root = tempfile::tempdir().unwrap();
        let (state, _rx) = durable_media_applicable_state(root.path());
        let app = app_with_state(state);

        let resp = app
            .clone()
            .oneshot(Request::get("/health").body(Body::empty()).unwrap())
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
        let body = json_body(resp).await;
        assert_eq!(body["status"], "degraded");
        assert_eq!(body["degraded"], serde_json::json!(["durable_media"]));
    }

    /// `/health` is auth-exempt, so the reasons — which name host filesystem
    /// paths — belong on authenticated `/api/status` and nowhere else.
    #[tokio::test]
    async fn status_carries_the_durable_media_reasons_that_health_withholds() {
        let root = tempfile::tempdir().unwrap();
        let (state, _rx) = durable_media_applicable_state(root.path());
        state.queue_journal.set_durable_media_status(
            false,
            vec!["owner media store unavailable: /srv/mold/queue-media has mode 0770".to_string()],
        );
        let app = app_with_state(state);

        let health = json_body(
            app.clone()
                .oneshot(Request::get("/health").body(Body::empty()).unwrap())
                .await
                .unwrap(),
        )
        .await;
        assert!(
            !health.to_string().contains("queue-media"),
            "an auth-exempt surface must not disclose host paths: {health}"
        );

        let status = json_body(
            app.oneshot(Request::get("/api/status").body(Body::empty()).unwrap())
                .await
                .unwrap(),
        )
        .await;
        assert_eq!(status["durable_media"]["available"], false);
        let reasons = status["durable_media"]["reasons"].as_array().unwrap();
        assert!(
            reasons
                .iter()
                .any(|reason| reason.as_str().unwrap().contains("mode 0770")),
            "{status}"
        );
    }

    /// `DurableMediaStatus.available` promises to mirror the presence of
    /// `capabilities.durable_media`. That capability carries a second runtime
    /// gate beyond the journal's own readiness, so the two must read the same
    /// applicability question or a client and its operator get contradictory
    /// answers on the same server.
    #[tokio::test]
    async fn status_availability_tracks_the_advertised_durable_media_capability() {
        let root = tempfile::tempdir().unwrap();
        let db = std::sync::Arc::new(Some(mold_db::MetadataDb::open_in_memory().unwrap()));
        let (state, _rx) = durable_state_with_engine(db, root.path(), MockEngine::ready());
        let app = app_with_state(state);

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
        let status = json_body(
            app.oneshot(Request::get("/api/status").body(Body::empty()).unwrap())
                .await
                .unwrap(),
        )
        .await;

        let advertised = capabilities
            .get("durable_media")
            .is_some_and(|entry| !entry.is_null());
        match status.get("durable_media") {
            Some(reported) if !reported.is_null() => {
                assert_eq!(
                    reported["available"], advertised,
                    "{capabilities}\n{status}"
                );
            }
            // Not applicable on this runtime, which is exactly when the
            // capability must be absent too.
            _ => assert!(!advertised, "{capabilities}"),
        }
    }

    /// A host that never advertises restart-safe media is configured that
    /// way, not broken: an observe-mode or output-disabled server must not
    /// read as degraded, or the field is noise on every such host.
    #[tokio::test]
    async fn health_stays_ok_where_durable_media_is_not_applicable() {
        let root = tempfile::tempdir().unwrap();
        let (mut state, _rx) = durable_media_applicable_state(root.path());
        state.scheduled_work = crate::scheduler::ScheduledWorkHandle::for_runtime(
            tokio::sync::mpsc::channel(1).0,
            crate::dispatch_mode::DispatchMode::Observe,
            false,
            true,
        );
        let app = app_with_state(state);

        let body = json_body(
            app.oneshot(Request::get("/health").body(Body::empty()).unwrap())
                .await
                .unwrap(),
        )
        .await;
        assert_eq!(body["status"], "ok", "{body}");
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
        // `installed == false` assertions read the store through
        // `resolved_models_dir()`; a direnv MOLD_MODELS_DIR points that at a
        // real store and flips them.
        let _env = crate::test_support::hermetic_store_env();
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
        // A validation refusal needs a host that generates: on one that does
        // not, every generation request is 503 before its shape is read.
        let (app, _gallery_root) = app_with(MockEngine::ready());
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
        // A validation refusal needs a host that generates: on one that does
        // not, every generation request is 503 before its shape is read.
        let (app, _gallery_root) = app_with(MockEngine::ready());
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
        // A validation refusal needs a host that generates: on one that does
        // not, every generation request is 503 before its shape is read.
        let (app, _gallery_root) = app_with(MockEngine::ready());
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
        let (app, _gallery_root) = app_with(MockEngine::ready());
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
    async fn status_queue_depth_reports_total_waiting_load_without_durable_overlay_duplicates() {
        let root = tempfile::tempdir().unwrap();
        let db = Arc::new(Some(mold_db::MetadataDb::open_in_memory().unwrap()));
        let (mut state, _rx) = durable_state(db.clone(), root.path());
        state.queue_capacity = 2;
        let owner = state.queue_journal.owner_uuid().unwrap().to_string();
        for index in 0..5 {
            seed_durable_projection_row(
                &db,
                &owner,
                &format!("durable-{index}"),
                mold_db::generation_queue::QueueRowState::Queued,
                index,
                0,
            );
        }
        seed_durable_projection_row(
            &db,
            &owner,
            "durable-running",
            mold_db::generation_queue::QueueRowState::Running,
            6,
            0,
        );
        seed_durable_projection_row(
            &db,
            &owner,
            "durable-held",
            mold_db::generation_queue::QueueRowState::Held,
            7,
            0,
        );
        state.job_registry.register("durable-0", "model-durable-0");
        state.job_registry.register("live-only", "model-live-only");
        state
            .job_registry
            .register("durable-running", "model-durable-running");
        state.job_registry.mark_running("durable-running", Some(0));

        let status = json_body(
            app_with_state(state)
                .oneshot(Request::get("/api/status").body(Body::empty()).unwrap())
                .await
                .unwrap(),
        )
        .await;
        assert_eq!(status["queue_depth"], 6);
        assert_eq!(status["queue_capacity"], 2);
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
    async fn blocked_device_preference_db_does_not_hold_scheduler_fence_and_patches_stay_ordered() {
        let worker = gpu_worker_stub(0);
        worker.in_flight.store(1, Ordering::SeqCst);
        let pool = Arc::new(crate::gpu_pool::GpuPool {
            workers: vec![worker.clone()].into(),
        });
        let db = Arc::new(Some(mold_db::MetadataDb::open_in_memory().unwrap()));
        let mut state = AppState::with_engine(MockEngine::ready());
        state.gpu_pool = pool;
        install_worker_registry_with_metadata(&mut state, db.clone());
        install_authoritative_v2(&mut state);
        let registry = state.device_registry.clone();
        let scheduler_fence = state.scheduler_mutation_fence.clone();
        let app = app_with_state(state);
        let id = "cuda:00000000000000000000000000000000";

        let (db_locked_tx, db_locked_rx) = std::sync::mpsc::sync_channel(1);
        let (release_db_tx, release_db_rx) = std::sync::mpsc::sync_channel(1);
        let locked_db = db.clone();
        let db_holder = std::thread::spawn(move || {
            locked_db
                .as_ref()
                .as_ref()
                .unwrap()
                .with_conn(|_| {
                    db_locked_tx.send(()).unwrap();
                    release_db_rx.recv().unwrap();
                    Ok(())
                })
                .unwrap();
        });
        db_locked_rx
            .recv_timeout(Duration::from_secs(1))
            .expect("test must hold the real metadata connection mutex");

        let (persist_started_tx, persist_started_rx) = std::sync::mpsc::sync_channel(1);
        let notified = Arc::new(AtomicBool::new(false));
        registry.set_preference_persistence_hook(Some(Arc::new({
            let notified = notified.clone();
            move |_, _| {
                if !notified.swap(true, Ordering::SeqCst) {
                    persist_started_tx.send(()).unwrap();
                }
                Ok(())
            }
        })));

        let disabling = tokio::spawn(
            app.clone().oneshot(
                Request::patch(format!("/api/devices/{id}"))
                    .header("content-type", "application/json")
                    .body(Body::from(r#"{"enabled":false}"#))
                    .unwrap(),
            ),
        );
        tokio::task::spawn_blocking(move || {
            persist_started_rx
                .recv_timeout(Duration::from_secs(1))
                .expect("PATCH must reach real metadata persistence");
        })
        .await
        .unwrap();
        let unrelated_scheduler_operation = scheduler_fence
            .try_lock_owned()
            .expect("blocked metadata must not hold the scheduler mutation fence");
        drop(unrelated_scheduler_operation);

        let enabling = tokio::spawn(
            app.oneshot(
                Request::patch(format!("/api/devices/{id}"))
                    .header("content-type", "application/json")
                    .body(Body::from(r#"{"enabled":true}"#))
                    .unwrap(),
            ),
        );
        release_db_tx.send(()).unwrap();
        db_holder.join().unwrap();

        let disabled = disabling.await.unwrap().unwrap();
        assert_eq!(disabled.status(), StatusCode::ACCEPTED);
        assert_eq!(json_body(disabled).await["desired_enabled"], false);
        let enabled = enabling.await.unwrap().unwrap();
        assert_eq!(enabled.status(), StatusCode::OK);
        assert_eq!(json_body(enabled).await["desired_enabled"], true);
        assert!(registry.desired_enabled(id));
        assert_eq!(
            mold_db::DevicePreferences::new(db.as_ref().as_ref().unwrap())
                .get(id)
                .unwrap(),
            Some(true),
            "same-device persistence and publication must retain request order"
        );
        assert_eq!(
            worker.drain_state.load(Ordering::SeqCst),
            crate::gpu_pool::DRAIN_RUNNING,
            "the later enable must cancel the earlier pending drain"
        );
    }

    #[tokio::test]
    async fn failed_concurrent_device_patch_does_not_publish_or_block_its_ordered_successor() {
        let worker = gpu_worker_stub(0);
        worker.in_flight.store(1, Ordering::SeqCst);
        let pool = Arc::new(crate::gpu_pool::GpuPool {
            workers: vec![worker.clone()].into(),
        });
        let db = Arc::new(Some(mold_db::MetadataDb::open_in_memory().unwrap()));
        let mut state = AppState::with_engine(MockEngine::ready());
        state.gpu_pool = pool;
        install_worker_registry_with_metadata(&mut state, db.clone());
        install_authoritative_v2(&mut state);
        let registry = state.device_registry.clone();
        let app = app_with_state(state);
        let id = "cuda:00000000000000000000000000000000";

        let (first_started_tx, first_started_rx) = std::sync::mpsc::sync_channel(1);
        let (release_first_tx, release_first_rx) = std::sync::mpsc::sync_channel(1);
        let release_first_rx = Arc::new(Mutex::new(release_first_rx));
        let calls = Arc::new(AtomicUsize::new(0));
        registry.set_preference_persistence_hook(Some(Arc::new({
            let calls = calls.clone();
            let release_first_rx = release_first_rx.clone();
            move |_, _| {
                if calls.fetch_add(1, Ordering::SeqCst) == 0 {
                    first_started_tx.send(()).unwrap();
                    release_first_rx.lock().unwrap().recv().unwrap();
                    anyhow::bail!("injected preference persistence failure");
                }
                Ok(())
            }
        })));

        let failing = tokio::spawn(
            app.clone().oneshot(
                Request::patch(format!("/api/devices/{id}"))
                    .header("content-type", "application/json")
                    .body(Body::from(r#"{"enabled":false}"#))
                    .unwrap(),
            ),
        );
        tokio::task::spawn_blocking(move || {
            first_started_rx
                .recv_timeout(Duration::from_secs(1))
                .expect("first PATCH must own same-device persistence ordering");
        })
        .await
        .unwrap();
        let succeeding = tokio::spawn(
            app.oneshot(
                Request::patch(format!("/api/devices/{id}"))
                    .header("content-type", "application/json")
                    .body(Body::from(r#"{"enabled":false}"#))
                    .unwrap(),
            ),
        );
        release_first_tx.send(()).unwrap();

        let failed = failing.await.unwrap().unwrap();
        assert_eq!(failed.status(), StatusCode::INTERNAL_SERVER_ERROR);
        assert!(json_body(failed).await["error"]
            .as_str()
            .unwrap()
            .contains("injected preference persistence failure"));
        let succeeded = succeeding.await.unwrap().unwrap();
        assert_eq!(succeeded.status(), StatusCode::ACCEPTED);
        assert_eq!(json_body(succeeded).await["desired_enabled"], false);
        assert_eq!(calls.load(Ordering::SeqCst), 2);
        assert!(!registry.desired_enabled(id));
        assert_eq!(
            mold_db::DevicePreferences::new(db.as_ref().as_ref().unwrap())
                .get(id)
                .unwrap(),
            Some(false),
            "the failed write must not publish, and its ordered successor must persist"
        );
        assert_eq!(
            worker.drain_state.load(Ordering::SeqCst),
            crate::gpu_pool::DRAIN_REQUESTED,
            "only the successfully persisted mutation may touch lifecycle state"
        );
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
                reclaimable_zfs_arc: None,
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
        let root = tempfile::tempdir().unwrap();
        let db = Arc::new(Some(mold_db::MetadataDb::open_in_memory().unwrap()));
        let (state, _rx) = durable_state(db, root.path());
        let journal = state.queue_journal.clone();
        state.set_generation_unavailable(
            "generation is unavailable while GPU selection is 'none' (maintenance mode)",
        );
        let app = app_with_state(state);

        for path in ["/api/generate", "/api/generate/stream"] {
            let response = app
                .clone()
                .oneshot(
                    Request::post(path)
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
        assert!(
            journal.list_all().is_empty(),
            "maintenance must reject before durable admission"
        );
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
            ("/api/chain-jobs", chain_body.clone()),
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
        assert_eq!(
            body,
            serde_json::json!({"entries": []}),
            "omitting pagination must preserve the legacy response shape"
        );
    }

    #[tokio::test]
    async fn queue_pagination_rejects_zero_and_malformed_inputs() {
        let app = app_empty();
        for uri in [
            "/api/queue?limit=0",
            "/api/queue?limit=1&cursor=not-a-valid-cursor",
            "/api/queue?cursor=not-a-valid-cursor",
        ] {
            let response = app
                .clone()
                .oneshot(Request::get(uri).body(Body::empty()).unwrap())
                .await
                .unwrap();
            assert_eq!(response.status(), StatusCode::BAD_REQUEST, "{uri}");
            let body = json_body(response).await;
            assert_eq!(body["code"], "INVALID_QUEUE_PAGE", "{uri}");
        }
    }

    #[tokio::test]
    async fn selected_queue_preview_is_scoped_to_a_live_job() {
        let state = AppState::for_tests();
        state.job_registry.register("rendering-a", "flux-dev:q4");
        let registry = state.job_registry.clone();
        let app = app_with_state(state);

        let response = app
            .clone()
            .oneshot(
                Request::get("/api/queue/rendering-a/preview")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::OK);
        assert_eq!(json_body(response).await, serde_json::Value::Null);

        // With previews disabled the row still reports what it is doing and
        // how far along it is: the snapshot is progress, not an image.
        registry.record_progress(
            "rendering-a",
            &mold_core::types::SseProgressEvent::StageStart {
                name: "Denoising".into(),
            },
        );
        registry.record_progress(
            "rendering-a",
            &mold_core::types::SseProgressEvent::DenoiseStep {
                step: 4,
                total: 20,
                elapsed_ms: 80,
            },
        );

        let response = app
            .clone()
            .oneshot(
                Request::get("/api/queue/rendering-a/preview")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::OK);
        let body = json_body(response).await;
        assert_eq!(body["step"], 4);
        assert_eq!(body["total"], 20);
        assert_eq!(body["stage"], "Denoising");
        assert!(body.get("preview_image").is_none());

        registry.record_progress(
            "rendering-a",
            &mold_core::types::SseProgressEvent::Preview {
                image: "UFJFVklFVw==".into(),
                step: 4,
                total: 20,
            },
        );

        let response = app
            .clone()
            .oneshot(
                Request::get("/api/queue/rendering-a/preview")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::OK);
        let body = json_body(response).await;
        assert_eq!(body["preview_image"], "UFJFVklFVw==");
        assert_eq!(body["step"], 4);
        assert_eq!(body["total"], 20);

        registry.remove("rendering-a");
        let response = app
            .oneshot(
                Request::get("/api/queue/rendering-a/preview")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::NOT_FOUND);
    }

    #[tokio::test]
    async fn activity_snapshot_exposes_only_server_owned_nonterminal_work() {
        let mut state = AppState::for_tests();
        let home = tempfile::tempdir().unwrap();
        let db = mold_db::MetadataDb::open_in_memory().unwrap();
        seed_chain_job(&db, home.path(), "sequence-c", ChainJobState::Running);
        let ephemeral_dir =
            seed_chain_job(&db, home.path(), "one-shot-chain", ChainJobState::Running);
        let mut ephemeral = ChainJobManifest::read_from_dir(&ephemeral_dir).unwrap();
        ephemeral.ephemeral = true;
        ephemeral.write_atomic(&ephemeral_dir).unwrap();
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
        state
            .job_registry
            .register("preparing-d", "flux2-klein-9b:bf16");
        state
            .job_registry
            .register("batch-child", "flux2-klein-9b:bf16");
        state.job_registry.register("running-b", "sdxl:q8");
        state.job_registry.mark_running("running-b", Some(1));
        state.job_registry.record_progress(
            "running-b",
            &mold_core::types::SseProgressEvent::StageStart {
                name: "Loading Flux.2 transformer".into(),
            },
        );
        state.job_registry.register("denoising-e", "sdxl:q8");
        state.job_registry.mark_running("denoising-e", Some(0));
        state.job_registry.record_progress(
            "denoising-e",
            &mold_core::types::SseProgressEvent::DenoiseStep {
                step: 2,
                total: 4,
                elapsed_ms: 1_000,
            },
        );
        state.scheduled_work.set_queue_work_items_for_tests(vec![
            mold_core::QueueWorkItem {
                work_id: "queued-a".into(),
                parent_id: "queued-a".into(),
                work_kind: "generation".into(),
                activity_phase: mold_core::QueueActivityPhase::Queued,
                ..Default::default()
            },
            mold_core::QueueWorkItem {
                work_id: "preparing-d".into(),
                parent_id: "preparing-d".into(),
                work_kind: "generation".into(),
                blocked_reason: Some(mold_core::QueueBlockedReason::Preparing),
                preparation_progress: Some(mold_core::QueuePreparationProgress {
                    component: "Verifying model files".into(),
                    bytes_done: 27,
                    bytes_total: 100,
                    phase_elapsed_ms: Some(4_200),
                }),
                activity_phase: mold_core::QueueActivityPhase::Blocked,
                ..Default::default()
            },
            mold_core::QueueWorkItem {
                work_id: "batch-child".into(),
                parent_id: "batch-parent".into(),
                work_kind: "prepared_sibling".into(),
                blocked_reason: Some(mold_core::QueueBlockedReason::Preparing),
                preparation_progress: Some(mold_core::QueuePreparationProgress {
                    component: "Loading reference images".into(),
                    bytes_done: 41,
                    bytes_total: 100,
                    phase_elapsed_ms: Some(900),
                }),
                activity_phase: mold_core::QueueActivityPhase::Blocked,
                ..Default::default()
            },
            mold_core::QueueWorkItem {
                work_id: "running-b".into(),
                parent_id: "running-b".into(),
                work_kind: "generation".into(),
                activity_phase: mold_core::QueueActivityPhase::Active,
                ..Default::default()
            },
            mold_core::QueueWorkItem {
                work_id: "denoising-e".into(),
                parent_id: "denoising-e".into(),
                work_kind: "generation".into(),
                activity_phase: mold_core::QueueActivityPhase::Active,
                ..Default::default()
            },
            mold_core::QueueWorkItem {
                work_id: "sequence-c:stage:0".into(),
                parent_id: "sequence-c".into(),
                work_kind: "chain_stage".into(),
                chain_stage: Some(0),
                activity_phase: mold_core::QueueActivityPhase::Active,
                ..Default::default()
            },
            mold_core::QueueWorkItem {
                work_id: "upscale-f".into(),
                parent_id: "upscale-f".into(),
                work_kind: "standalone_upscale".into(),
                activity_phase: mold_core::QueueActivityPhase::Active,
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
        assert_eq!(items.len(), 10);
        let item = |id: &str| items.iter().find(|item| item["id"] == id).unwrap();
        assert_eq!(item("queued-a")["kind"], "generation");
        assert_eq!(item("queued-a")["phase"], "queued");
        assert_eq!(item("queued-a")["can_cancel"], true);
        assert_eq!(item("running-b")["phase"], "loading");
        assert_eq!(item("running-b")["stage"], "Loading Flux.2 transformer");
        assert_eq!(item("running-b")["can_cancel"], true);
        assert_eq!(item("denoising-e")["phase"], "running");
        assert_eq!(item("denoising-e")["current"], 2);
        assert_eq!(item("denoising-e")["total"], 4);
        assert_eq!(item("preparing-d")["phase"], "preparing");
        assert_eq!(
            item("preparing-d")["preparation_progress"]["component"],
            "Verifying model files"
        );
        assert_eq!(item("preparing-d")["current"], 27);
        assert_eq!(item("preparing-d")["total"], 100);
        assert_eq!(item("batch-child")["phase"], "preparing");
        assert_eq!(item("batch-child")["current"], 41);
        assert_eq!(item("batch-child")["total"], 100);
        assert_eq!(item("expand-parent")["kind"], "prompt_expand");
        assert_eq!(item("expand-parent")["phase"], "running");
        assert_eq!(item("sequence-c")["kind"], "sequence");
        assert_eq!(item("sequence-c")["phase"], "running");
        assert!(item("sequence-c").get("stage").is_none());
        assert_eq!(item("upscale-f")["kind"], "standalone_upscale");
        assert_eq!(item("upscale-f")["phase"], "running");
        assert!(item("upscale-f").get("stage").is_none());
        assert_eq!(item("one-shot-chain")["kind"], "generation");
        assert_eq!(item("one-shot-chain")["execution"], "chain");
        assert_eq!(item("one-shot-chain")["phase"], "running");
        assert_eq!(item("one-shot-chain")["can_cancel"], true);
        assert_eq!(item(&download_id)["kind"], "download");
        assert_eq!(item(&download_id)["phase"], "queued");
        assert_eq!(body["instance_id"], instance_id);
        assert!(body["observed_at_unix_ms"].as_u64().unwrap() > 0);

        let queue_body = json_body(
            app.clone()
                .oneshot(
                    Request::get("/api/queue/running-b")
                        .body(Body::empty())
                        .unwrap(),
                )
                .await
                .unwrap(),
        )
        .await;
        assert_eq!(queue_body["work_item"]["runtime_phase"], "loading");
        assert_eq!(
            queue_body["work_item"]["runtime_stage"],
            "Loading Flux.2 transformer"
        );

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
        assert_eq!(
            body["unavailable_kinds"],
            serde_json::json!(["sequence", "chain_generation"])
        );
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
    async fn patch_queue_reorders_and_relanes_a_durable_row_beyond_the_runtime_window() {
        let root = tempfile::tempdir().unwrap();
        let db = Arc::new(Some(mold_db::MetadataDb::open_in_memory().unwrap()));
        let (mut state, _rx) = durable_state(db.clone(), root.path());
        state.queue_capacity = 2;
        state.gpu_pool = Arc::new(crate::gpu_pool::GpuPool {
            workers: vec![gpu_worker_stub(0)].into(),
        });
        install_worker_registry(&mut state);
        let owner = state.queue_journal.owner_uuid().unwrap().to_string();
        for index in 0..5 {
            seed_durable_projection_row(
                &db,
                &owner,
                &format!("deep-patch-{index}"),
                mold_db::generation_queue::QueueRowState::Queued,
                index,
                0,
            );
        }
        assert!(state.job_registry.snapshot().entries.is_empty());
        let app = app_with_state(state.clone());

        let response = app
            .oneshot(
                Request::patch("/api/queue/deep-patch-4")
                    .header("content-type", "application/json")
                    .body(Body::from(r#"{"target_gpu":0,"position":0}"#))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::OK);
        let body = json_body(response).await;
        assert_eq!(body["id"], "deep-patch-4");
        assert_eq!(body["position"], 0);
        assert_eq!(body["target_gpu"], 0);
        assert_eq!(body["durable"], true);

        let page = state.queue_journal.projection_page(None, 2).unwrap();
        assert_eq!(page.rows[0].id, "deep-patch-4");
        assert_eq!(page.rows[0].target_gpu, Some(0));
        assert_eq!(page.rows[1].id, "deep-patch-0");
    }

    #[tokio::test]
    async fn patch_queue_uses_live_authority_for_a_claimed_feeder_handoff() {
        let root = tempfile::tempdir().unwrap();
        let db = Arc::new(Some(mold_db::MetadataDb::open_in_memory().unwrap()));
        let (state, _rx) = durable_state(db.clone(), root.path());
        let owner = state.queue_journal.owner_uuid().unwrap().to_string();
        seed_durable_projection_row(
            &db,
            &owner,
            "live-running",
            mold_db::generation_queue::QueueRowState::Queued,
            0,
            0,
        );
        let _claim = state
            .queue_journal
            .claim_feeder_by_id("live-running")
            .unwrap()
            .expect("feeder claims the handoff row");
        state.job_registry.register("other-live", "model-other");
        state
            .job_registry
            .register_with_target_gpu("live-running", "model-live-running", Some(2));

        let response = app_with_state(state.clone())
            .oneshot(
                Request::patch("/api/queue/live-running")
                    .header("content-type", "application/json")
                    .body(Body::from(r#"{"target_gpu":null,"position":0}"#))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::OK);
        let body = json_body(response).await;
        assert_eq!(body["id"], "live-running");
        assert_eq!(body["position"], 0);
        assert!(body.get("target_gpu").is_none());

        let durable = mold_db::generation_queue::get(db.as_ref().as_ref().unwrap(), "live-running")
            .unwrap()
            .expect("claimed durable row remains present");
        assert_eq!(durable.target_gpu, None);
        assert_eq!(
            state.job_registry.queued_ids_in_order(),
            ["live-running", "other-live"]
        );
    }

    #[tokio::test]
    async fn patch_queue_returns_failure_without_mutating_runtime_when_sqlite_mutation_fails() {
        let root = tempfile::tempdir().unwrap();
        let db = Arc::new(Some(mold_db::MetadataDb::open_in_memory().unwrap()));
        let (state, _rx) = durable_state(db.clone(), root.path());
        let owner = state.queue_journal.owner_uuid().unwrap().to_string();
        seed_durable_projection_row(
            &db,
            &owner,
            "durable-failure",
            mold_db::generation_queue::QueueRowState::Queued,
            0,
            0,
        );
        state.job_registry.register_with_target_gpu(
            "durable-failure",
            "model-durable-failure",
            Some(7),
        );
        db.as_ref()
            .as_ref()
            .unwrap()
            .with_conn(|conn| {
                conn.execute_batch("DROP TABLE generation_queue")?;
                Ok(())
            })
            .unwrap();

        let response = app_with_state(state.clone())
            .oneshot(
                Request::patch("/api/queue/durable-failure")
                    .header("content-type", "application/json")
                    .body(Body::from(r#"{"target_gpu":null}"#))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::INTERNAL_SERVER_ERROR);
        assert_eq!(
            state
                .job_registry
                .entry("durable-failure")
                .unwrap()
                .target_gpu,
            Some(7),
            "SQLite-first failure must leave the runtime projection untouched"
        );
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn blocked_durable_patch_never_blocks_scheduler_grant_or_cancellation() {
        let root = tempfile::tempdir().unwrap();
        let db = Arc::new(Some(mold_db::MetadataDb::open_in_memory().unwrap()));
        let (state, _rx) = durable_state(db.clone(), root.path());
        let owner = state.queue_journal.owner_uuid().unwrap().to_string();
        seed_durable_projection_row(
            &db,
            &owner,
            "blocked-patch",
            mold_db::generation_queue::QueueRowState::Queued,
            0,
            0,
        );
        state.job_registry.register("grant-live", "model-grant");
        state.job_registry.register("cancel-live", "model-cancel");

        // Hold the journal's actual SQLite connection mutex so PATCH reaches
        // its spawn_blocking DB operation and remains there deterministically.
        let locked_db = db.clone();
        let (locked_tx, locked_rx) = tokio::sync::oneshot::channel();
        let (release_tx, release_rx) = std::sync::mpsc::sync_channel(0);
        let blocker = tokio::task::spawn_blocking(move || {
            locked_db.as_ref().as_ref().unwrap().with_conn(|_| {
                locked_tx.send(()).unwrap();
                release_rx.recv().unwrap();
                Ok(())
            })
        });
        locked_rx.await.unwrap();

        let journal = state.queue_journal.clone();
        let patch = tokio::spawn(
            app_with_state(state.clone()).oneshot(
                Request::patch("/api/queue/blocked-patch")
                    .header("content-type", "application/json")
                    .body(Body::from(r#"{"position":0}"#))
                    .unwrap(),
            ),
        );
        tokio::time::timeout(Duration::from_secs(2), async {
            while !journal.durable_transition_is_locked() {
                tokio::task::yield_now().await;
            }
        })
        .await
        .expect("PATCH enters the durable transition before waiting on SQLite");
        assert!(!patch.is_finished(), "the PATCH remains DB-blocked");

        let scheduler = tokio::time::timeout(
            Duration::from_secs(2),
            state.scheduler_mutation_fence.lock(),
        )
        .await
        .expect("blocked PATCH must not own the scheduler fence");
        state.job_registry.reorder_queued("grant-live", 0).unwrap();
        state.job_registry.cancel_queued("cancel-live").unwrap();
        state
            .job_registry
            .dispatch_if_queued("grant-live", 0, (), |_| Ok(()))
            .unwrap();
        drop(scheduler);

        release_tx.send(()).unwrap();
        blocker.await.unwrap().unwrap();
        let response = patch.await.unwrap().unwrap();
        assert_eq!(response.status(), StatusCode::OK);
        assert_eq!(
            state.job_registry.entry("grant-live").unwrap().state,
            crate::job_registry::JobLifecycle::Running
        );
        assert!(state.job_registry.entry("cancel-live").is_none());
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
        let _lock = env_lock();
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
        let _lock = env_lock();
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
    async fn durable_cancellation_lookup_never_holds_the_scheduler_mutation_fence() {
        let root = tempfile::tempdir().unwrap();
        let db = Arc::new(Some(mold_db::MetadataDb::open_in_memory().unwrap()));
        let (state, _rx) = durable_state(db.clone(), root.path());
        let owner = state.queue_journal.owner_uuid().unwrap().to_string();
        seed_durable_projection_row(
            &db,
            &owner,
            "deep-cancel",
            mold_db::generation_queue::QueueRowState::Queued,
            0,
            0,
        );

        let (locked_tx, locked_rx) = std::sync::mpsc::channel();
        let (release_tx, release_rx) = std::sync::mpsc::channel();
        let locked_db = db.clone();
        let holder = std::thread::spawn(move || {
            locked_db
                .as_ref()
                .as_ref()
                .unwrap()
                .with_conn(|_| {
                    locked_tx.send(()).unwrap();
                    release_rx.recv().unwrap();
                    Ok(())
                })
                .unwrap();
        });
        locked_rx.recv().unwrap();

        let app = app_with_state(state.clone());
        let mut cancellation = Box::pin(
            app.oneshot(
                Request::delete("/api/queue/deep-cancel")
                    .body(Body::empty())
                    .unwrap(),
            ),
        );
        assert!(
            futures::poll!(cancellation.as_mut()).is_pending(),
            "the cancellation lookup must wait behind the held SQLite connection"
        );
        let scheduler_guard = state
            .scheduler_mutation_fence
            .try_lock()
            .expect("blocking SQLite lookup must not own the scheduler mutation fence");
        drop(scheduler_guard);
        release_tx.send(()).unwrap();
        holder.join().unwrap();
        assert_eq!(cancellation.await.unwrap().status(), StatusCode::NO_CONTENT);
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
    async fn delete_queue_cannot_cross_a_live_owner_in_the_same_mold_home() {
        let root = tempfile::tempdir().unwrap();
        let db = Arc::new(Some(mold_db::MetadataDb::open_in_memory().unwrap()));
        let (first, _first_rx) = durable_state(db.clone(), root.path());
        let (second, _second_rx) = durable_state(db.clone(), root.path());
        let first_owner = first.queue_journal.owner_uuid().unwrap();
        let second_owner = second.queue_journal.owner_uuid().unwrap();
        assert_ne!(first_owner, second_owner);
        seed_durable_projection_row(
            &db,
            second_owner,
            "foreign-job",
            mold_db::generation_queue::QueueRowState::Queued,
            1,
            0,
        );

        let response = app_with_state(first)
            .oneshot(
                Request::delete("/api/queue/foreign-job")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();

        assert_eq!(response.status(), StatusCode::NOT_FOUND);
        assert_eq!(second.queue_journal.list_all().len(), 1);
    }

    #[tokio::test]
    async fn delete_queue_terminalizes_a_held_batch_child_before_acknowledging() {
        let root = tempfile::tempdir().unwrap();
        let db = Arc::new(Some(mold_db::MetadataDb::open_in_memory().unwrap()));
        let (state, _rx) = durable_state(db, root.path());
        admit_one_durable_batch(&state, "held-child", "held-batch");
        let claim = state.queue_journal.claim_next_feeder().unwrap().unwrap();
        state
            .queue_journal
            .attach_claimed(&claim.row.id, claim.claim_token)
            .hold("operator review");

        let response = app_with_state(state.clone())
            .oneshot(
                Request::delete("/api/queue/held-child")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();

        assert_eq!(response.status(), StatusCode::NO_CONTENT);
        assert!(state.queue_journal.list_all().is_empty());
        let child = &state
            .queue_journal
            .generation_batch("held-batch")
            .unwrap()
            .children[0];
        assert_eq!(child.state, "cancelled");
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
    async fn one_job_pause_leaves_dispatch_open_and_survives_a_global_pause_cycle() {
        let root = tempfile::tempdir().unwrap();
        let db = Arc::new(Some(mold_db::MetadataDb::open_in_memory().unwrap()));
        let (state, _rx) = durable_state(db.clone(), root.path());
        let owner = state.queue_journal.owner_uuid().unwrap().to_string();
        for (id, created_at) in [("selected", 1), ("sibling", 2)] {
            seed_durable_projection_row(
                &db,
                &owner,
                id,
                mold_db::generation_queue::QueueRowState::Queued,
                created_at,
                0,
            );
        }
        let app = app_with_state(state.clone());

        let response = app
            .clone()
            .oneshot(
                Request::post("/api/queue/selected/pause")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::NO_CONTENT);
        let status = app
            .clone()
            .oneshot(Request::get("/api/status").body(Body::empty()).unwrap())
            .await
            .unwrap();
        assert_eq!(json_body(status).await["queue_paused"], false);
        assert_eq!(
            state
                .queue_journal
                .list_all()
                .into_iter()
                .map(|row| (row.id, row.state))
                .collect::<Vec<_>>(),
            [
                (
                    "selected".to_string(),
                    mold_db::generation_queue::QueueRowState::Paused,
                ),
                (
                    "sibling".to_string(),
                    mold_db::generation_queue::QueueRowState::Queued,
                ),
            ]
        );

        for path in ["/api/queue/pause", "/api/queue/resume"] {
            let response = app
                .clone()
                .oneshot(Request::post(path).body(Body::empty()).unwrap())
                .await
                .unwrap();
            assert_eq!(response.status(), StatusCode::OK);
        }
        assert_eq!(
            mold_db::generation_queue::get(db.as_ref().as_ref().unwrap(), "selected")
                .unwrap()
                .unwrap()
                .state,
            mold_db::generation_queue::QueueRowState::Paused
        );

        let response = app
            .oneshot(
                Request::post("/api/queue/selected/resume")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::NO_CONTENT);
        assert_eq!(
            mold_db::generation_queue::get(db.as_ref().as_ref().unwrap(), "selected")
                .unwrap()
                .unwrap()
                .state,
            mold_db::generation_queue::QueueRowState::Queued
        );
    }

    #[tokio::test]
    async fn resume_releases_restart_paused_durable_work_and_activity_exposes_it() {
        let root = tempfile::tempdir().unwrap();
        let db = Arc::new(Some(mold_db::MetadataDb::open_in_memory().unwrap()));
        let (state, _rx) = durable_state(db.clone(), root.path());
        let owner = state.queue_journal.owner_uuid().unwrap().to_string();
        seed_durable_projection_row(
            &db,
            &owner,
            "restart-paused",
            mold_db::generation_queue::QueueRowState::Paused,
            1,
            0,
        );
        let app = app_with_state(state.clone());

        let activity = json_body(
            app.clone()
                .oneshot(Request::get("/api/activity").body(Body::empty()).unwrap())
                .await
                .unwrap(),
        )
        .await;
        assert_eq!(activity["items"][0]["id"], "restart-paused");
        assert_eq!(activity["items"][0]["phase"], "paused");
        assert_eq!(activity["items"][0]["can_cancel"], true);

        let response = app
            .oneshot(
                Request::post("/api/queue/resume")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::OK);
        assert_eq!(
            mold_db::generation_queue::get(db.as_ref().as_ref().unwrap(), "restart-paused")
                .unwrap()
                .unwrap()
                .state,
            mold_db::generation_queue::QueueRowState::Queued
        );
    }

    #[tokio::test]
    async fn restart_pauses_every_active_queue_and_resume_wakes_both_owners() {
        let root = tempfile::tempdir().unwrap();
        let db = Arc::new(Some(mold_db::MetadataDb::open_in_memory().unwrap()));
        let (mut state, mut generation_rx) = durable_state(db.clone(), root.path());

        admit_one_durable_batch(&state, "queued-generation", "queued-batch");
        admit_one_durable_batch(&state, "running-generation", "running-batch");
        let claimed = state.queue_journal.claim_next_feeder().unwrap().unwrap();
        let running = state
            .queue_journal
            .attach_claimed(&claimed.row.id, claimed.claim_token);
        assert_eq!(
            running.claim_dispatch(),
            crate::queue_journal::DispatchClaim::Granted
        );
        state.queue_journal.retain_all();
        drop(running);

        seed_chain_job(
            db.as_ref().as_ref().unwrap(),
            root.path(),
            "queued-sequence",
            ChainJobState::Queued,
        );
        seed_chain_job(
            db.as_ref().as_ref().unwrap(),
            root.path(),
            "running-sequence",
            ChainJobState::Running,
        );

        crate::durable_queue_feeder::recover_runtime(&state)
            .await
            .unwrap();
        crate::chain_job_runner::startup_reconcile(
            db.as_ref().as_ref().unwrap(),
            &root.path().join("jobs"),
        )
        .unwrap();
        assert!(state
            .queue_journal
            .list_all()
            .iter()
            .all(|row| row.state == mold_db::generation_queue::QueueRowState::Paused));
        assert!(["queued-sequence", "running-sequence"].iter().all(|id| {
            mold_db::chain_jobs::get_job(db.as_ref().as_ref().unwrap(), id)
                .unwrap()
                .unwrap()
                .state
                == ChainJobState::Paused
        }));

        let (chain_handle, mut chain_commands) =
            crate::chain_job_runner::ChainJobRunnerHandle::command_probe_for_tests();
        state.chain_jobs = Some(Arc::new(chain_handle));
        let feeder_shutdown = tokio_util::sync::CancellationToken::new();
        let feeder = crate::durable_queue_feeder::spawn(state.clone(), feeder_shutdown.clone());

        assert!(
            tokio::time::timeout(Duration::from_millis(100), generation_rx.recv())
                .await
                .is_err(),
            "restart-paused generations must not reach the worker"
        );
        assert!(chain_commands.try_recv().is_err());

        let response = app_with_state(state.clone())
            .oneshot(
                Request::post("/api/queue/resume")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::OK);

        let generation = tokio::time::timeout(Duration::from_secs(5), generation_rx.recv())
            .await
            .expect("resume wakes the durable generation feeder")
            .expect("generation queue remains open");
        assert!(matches!(
            generation.id.as_str(),
            "queued-generation" | "running-generation"
        ));
        assert!(matches!(
            tokio::time::timeout(Duration::from_secs(1), chain_commands.recv())
                .await
                .expect("resume wakes the chain runner"),
            Some(crate::chain_job_runner::RunnerCmd::Kick)
        ));
        assert!(["queued-sequence", "running-sequence"].iter().all(|id| {
            mold_db::chain_jobs::get_job(db.as_ref().as_ref().unwrap(), id)
                .unwrap()
                .unwrap()
                .state
                == ChainJobState::Queued
        }));

        feeder_shutdown.cancel();
        feeder.await.unwrap();
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
        assert_eq!(body["queue"]["can_pause_job"], true);
        assert_eq!(body["queue"]["can_cancel_all"], true);
        assert_eq!(body["queue"]["can_reorder"], true);
        assert!(
            body.get("durable_media").is_none(),
            "the server must keep durable request-media capability dark until activation is complete"
        );
        assert_eq!(body["devices"]["available"], true);
        assert_eq!(body["devices"]["lifecycle"], true);
        assert_eq!(body["devices"]["restart_enable"], false);
        assert_eq!(body["devices"]["stable_pins"], true);
        assert_eq!(body["devices"]["planned_lanes"], true);
        assert_eq!(body["devices"]["learned_eta"], true);
        assert_eq!(body["reference_uploads"]["available"], false);
        assert!(
            body["reference_uploads"].get("authless_inline").is_none(),
            "authless inline is the negation of `available`, not a second bit"
        );
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
    async fn capabilities_offer_reference_uploads_only_when_api_key_auth_is_enabled() {
        let keys = std::collections::HashSet::from(["test-key".to_string()]);
        let auth = Some(std::sync::Arc::new(crate::auth::ApiKeySet::new(keys)));
        let response = app_with_auth(auth)
            .oneshot(
                Request::get("/api/capabilities")
                    .header("x-api-key", "test-key")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::OK);
        let body = json_body(response).await;
        assert_eq!(body["reference_uploads"]["available"], true);
        assert_eq!(body["reference_uploads"]["requires_api_key"], true);
        assert_eq!(
            body["reference_uploads"]["max_active_sessions"],
            crate::reference_uploads::MAX_REFERENCE_UPLOAD_SESSIONS_PER_IDENTITY
        );
    }

    #[tokio::test]
    async fn capabilities_keep_durable_media_dark_without_installed_services() {
        let state = AppState::for_tests();
        state.queue_journal.set_durable_media_ready(true);
        assert_eq!(state.queue_journal.durable_media_capabilities(), None);

        let resp = app_with_state(state)
            .oneshot(
                Request::get("/api/capabilities")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
        assert!(
            json_body(resp).await.get("durable_media").is_none(),
            "readiness without lifecycle/admission services must remain dark"
        );
    }

    #[tokio::test]
    async fn capabilities_keep_canonical_admission_available_when_media_reconciliation_is_dark() {
        let root = tempfile::tempdir().unwrap();
        let db = Arc::new(Some(mold_db::MetadataDb::open_in_memory().unwrap()));
        let (mut state, _rx) = durable_state(db, root.path());
        install_authoritative_v2(&mut state);
        state.queue_journal.set_durable_media_ready(false);

        let response = app_with_state(state)
            .oneshot(
                Request::get("/api/capabilities")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::OK);
        let body = json_body(response).await;
        assert!(body.get("durable_media").is_none());
        assert_eq!(body["queue"]["heterogeneous_batch_max_outputs"], 64);
    }

    #[tokio::test(flavor = "current_thread")]
    async fn capabilities_advertise_exact_v2_for_single_and_multi_lane_runtime_matrices() {
        for ordinals in [vec![0], vec![0, 1, 2]] {
            let root = tempfile::tempdir().unwrap();
            let db = Arc::new(Some(mold_db::MetadataDb::open_in_memory().unwrap()));
            let (mut state, _rx) = durable_state(db, root.path());
            state.gpu_pool = Arc::new(crate::gpu_pool::GpuPool {
                workers: ordinals
                    .iter()
                    .copied()
                    .map(gpu_worker_stub)
                    .collect::<Vec<_>>()
                    .into(),
            });
            install_authoritative_v2(&mut state);

            let response = app_with_state(state)
                .oneshot(
                    Request::get("/api/capabilities")
                        .body(Body::empty())
                        .unwrap(),
                )
                .await
                .unwrap();
            assert_eq!(response.status(), StatusCode::OK);
            assert_eq!(
                json_body(response).await["durable_media"],
                serde_json::json!({
                    "protocol_version": 2,
                    "encrypted_at_rest": true,
                    "generate_request_media": true,
                    "identity": true,
                    "private_h3": cfg!(any(feature = "h3", feature = "h3-private-uat")),
                }),
                "lane count must not darken an otherwise complete runtime"
            );
        }
    }

    /// The identity block is what stops a client silently degrading a request
    /// an older server would drop: `id_images` becomes a render with no face,
    /// `true_cfg` becomes the distilled path. It must be advertised straight
    /// from `mold_core::identity`'s own constants, so it can never claim a
    /// bound the validator does not enforce.
    #[tokio::test]
    async fn capabilities_advertise_identity_shapes_from_the_contract_constants() {
        let app = app_with_state(AppState::for_tests());
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

        let available = mold_core::identity::identity_runtime_available();
        assert_eq!(body["identity"]["multi_photo"], available);
        assert_eq!(body["identity"]["true_cfg"], available);
        assert_eq!(
            body["identity"]["max_photos"],
            if available {
                mold_core::identity::ID_IMAGES_MAX
            } else {
                0
            }
        );

        // Absence is NO, never unknown: an older server's response omits the
        // block entirely and must deserialize to all-false rather than to a
        // permissive default. Modelled by removing the key from a real body,
        // which is exactly the shape such a server sends.
        let mut older = serde_json::to_value(mold_core::ServerCapabilities::default()).unwrap();
        older
            .as_object_mut()
            .expect("a capabilities object")
            .remove("identity")
            .expect("the block is serialized, so an older server is the one that omits it");
        let legacy: mold_core::ServerCapabilities = serde_json::from_value(older).unwrap();
        assert!(!legacy.identity.multi_photo);
        assert!(!legacy.identity.true_cfg);
        assert_eq!(legacy.identity.max_photos, 0);
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
        }
    }

    #[tokio::test]
    async fn raw_batch_never_enters_legacy_single_result_worker() {
        let (app, _gallery_root) = app_with(MockEngine::ready());
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
        assert_eq!(body["code"], "DIRECT_BATCH_UNSUPPORTED");
    }

    /// The durable row is written before `submit()`, so a crash between
    /// admission and the worker still leaves something to replay.
    #[tokio::test]
    async fn an_admitted_gallery_bound_generation_is_journaled_before_it_is_queued() {
        let (state, mut rx, _gallery_root) = durable_test_state(MockEngine::ready());
        spawn_durable_feeder(&state);
        let journal = state.queue_journal.clone();
        let output_dir = state.config.read().await.effective_output_dir();
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
        assert_eq!(rows[0].output_dir, output_dir);

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
        durable_state_with_engine(db, root, MockEngine::ready())
    }

    fn durable_state_with_engine(
        db: Arc<Option<mold_db::MetadataDb>>,
        root: &std::path::Path,
        engine: MockEngine,
    ) -> (
        AppState,
        tokio::sync::mpsc::Receiver<crate::state::GenerationJob>,
    ) {
        durable_state_with_engine_and_media_readiness(db, root, engine, true)
    }

    fn durable_state_with_engine_and_media_readiness(
        db: Arc<Option<mold_db::MetadataDb>>,
        root: &std::path::Path,
        engine: MockEngine,
        require_media_ready: bool,
    ) -> (
        AppState,
        tokio::sync::mpsc::Receiver<crate::state::GenerationJob>,
    ) {
        durable_state_with_admission_policy(
            db,
            root,
            engine,
            require_media_ready,
            true,
            "test-instance",
        )
    }

    fn durable_state_with_admission_policy(
        db: Arc<Option<mold_db::MetadataDb>>,
        root: &std::path::Path,
        engine: MockEngine,
        require_media_ready: bool,
        install_admission: bool,
        instance_id: &str,
    ) -> (
        AppState,
        tokio::sync::mpsc::Receiver<crate::state::GenerationJob>,
    ) {
        let gallery = durable_gallery_dir(root);
        std::fs::create_dir_all(&gallery).unwrap();
        let (mut state, rx) = AppState::with_engine_and_queue(engine);
        state.output_disabled_override = false;
        state.metadata_db = db.clone();
        state.queue_journal = Arc::new(crate::queue_journal::QueueJournal::new(
            db.clone(),
            Some(root),
            instance_id,
        ));
        // Production installs this in `lib.rs` before recovery; without it the
        // journal commits authoritative state silently and nothing that reads
        // `/api/events` — including the batch event stream — ever wakes.
        state
            .queue_journal
            .install_event_broadcaster(state.events.clone())
            .unwrap();
        if let Some(owner) = state.queue_journal.owner_uuid() {
            let lifecycle = Arc::new(crate::queue_media_lifecycle::QueueMediaLifecycle::new(
                db,
                root.to_path_buf(),
                owner.to_string(),
            ));
            state
                .queue_journal
                .install_queue_media_lifecycle(lifecycle.clone())
                .unwrap();
            let report = crate::queue_media_startup::reconcile_claimed_owner(
                &state.queue_journal,
                lifecycle.as_ref(),
            )
            .unwrap();
            assert!(!require_media_ready || report.durable_media_ready);
            if install_admission {
                let admission = crate::queue_media_admission::DurableMediaAdmission::new(
                    lifecycle,
                    state.queue_capacity,
                    false,
                )
                .unwrap();
                state
                    .queue_journal
                    .install_queue_media_admission(admission)
                    .unwrap();
            }
        }
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

    fn published_gallery_file_count(root: &std::path::Path) -> usize {
        std::fs::read_dir(durable_gallery_dir(root))
            .unwrap()
            .filter_map(Result::ok)
            .filter(|entry| entry.file_type().is_ok_and(|kind| kind.is_file()))
            .count()
    }

    fn admit_one_durable_batch(state: &AppState, id: &str, batch_id: &str) {
        let request: GenerateRequest =
            serde_json::from_str(&generate_body("publication fence", 64, 64)).unwrap();
        let output = state.config.try_read().unwrap().effective_output_dir();
        state
            .queue_journal
            .record_batch(crate::queue_journal::BatchJournalAdmission {
                id: batch_id,
                client_batch_id: &format!("client-{batch_id}"),
                request_sha256: "publication-fence-sha",
                children: &[crate::queue_journal::JournalAdmission {
                    id,
                    request: &request,
                    output_dir: Some(&output),
                    target_gpu: None,
                    target_device_id: None,
                    completion_payload: crate::state::SseCompletionPayload::MetadataOnly,
                    batch_child: false,
                }],
            })
            .unwrap();
    }

    async fn wait_for_attempt_cleanup(state: &AppState, id: &str) {
        tokio::time::timeout(Duration::from_secs(5), async {
            loop {
                if state.queue.pending() == 0 && !state.generation_cancel.is_registered(id) {
                    break;
                }
                tokio::task::yield_now().await;
            }
        })
        .await
        .expect("single-worker attempt must settle and unregister its token");
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn delete_racing_single_worker_completion_cannot_publish_or_overwrite_cancel() {
        let root = tempfile::tempdir().unwrap();
        let db = Arc::new(Some(mold_db::MetadataDb::open_in_memory().unwrap()));
        let blocker = Arc::new(GenerateBlocker::default());
        let (mut state, rx) = durable_state_with_engine(
            db,
            root.path(),
            MockEngine::blocking_generate_ignoring_cancellation(blocker.clone()),
        );
        state.queue_capacity = 1;
        admit_one_durable_batch(&state, "cancel-at-publication", "cancel-batch");

        let feeder_shutdown = tokio_util::sync::CancellationToken::new();
        let feeder = crate::durable_queue_feeder::spawn(state.clone(), feeder_shutdown.clone());
        let worker = tokio::spawn(crate::queue::run_queue_worker(rx, state.clone()));
        let app = app_with_state(state.clone());

        tokio::time::timeout(Duration::from_secs(5), async {
            while !blocker.entered.load(Ordering::SeqCst) {
                tokio::task::yield_now().await;
            }
        })
        .await
        .expect("single-worker fallback must enter inference");
        assert!(state
            .generation_cancel
            .is_registered("cancel-at-publication"));
        feeder_shutdown.cancel();
        feeder.await.unwrap();

        let cancelled = app
            .oneshot(
                Request::delete("/api/queue/cancel-at-publication")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(cancelled.status(), StatusCode::NO_CONTENT);
        blocker.release();
        wait_for_attempt_cleanup(&state, "cancel-at-publication").await;

        assert_eq!(
            published_gallery_file_count(root.path()),
            0,
            "a success returned after DELETE must not reach the gallery"
        );
        assert!(state.queue_journal.list_all().is_empty());
        let detail = state
            .queue_journal
            .generation_batch("cancel-batch")
            .unwrap();
        assert_eq!(detail.children[0].state, "cancelled");

        worker.abort();
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn delete_after_completion_claim_cannot_cancel_during_gallery_publication() {
        let root = tempfile::tempdir().unwrap();
        let db = Arc::new(Some(mold_db::MetadataDb::open_in_memory().unwrap()));
        let blocker = Arc::new(GenerateBlocker::default());
        let (mut state, rx) = durable_state_with_engine(
            db,
            root.path(),
            MockEngine::blocking_generate_ignoring_cancellation(blocker.clone()),
        );
        state.queue_capacity = 1;
        admit_one_durable_batch(&state, "complete-before-delete", "complete-batch");

        let feeder_shutdown = tokio_util::sync::CancellationToken::new();
        let feeder = crate::durable_queue_feeder::spawn(state.clone(), feeder_shutdown.clone());
        let worker = tokio::spawn(crate::queue::run_queue_worker(rx, state.clone()));
        let app = app_with_state(state.clone());

        tokio::time::timeout(Duration::from_secs(5), async {
            while !blocker.entered.load(Ordering::SeqCst) {
                tokio::task::yield_now().await;
            }
        })
        .await
        .expect("single-worker fallback must enter inference");
        feeder_shutdown.cancel();
        feeder.await.unwrap();

        let gallery_guard = state.gallery_publication_gate.write().await;
        blocker.release();
        tokio::time::timeout(Duration::from_secs(5), async {
            while !state
                .job_registry
                .completion_claimed_for_tests("complete-before-delete")
            {
                tokio::task::yield_now().await;
            }
        })
        .await
        .expect("worker must close cancellation admission before gallery publication");

        let rejected = app
            .oneshot(
                Request::delete("/api/queue/complete-before-delete")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(rejected.status(), StatusCode::NOT_FOUND);
        drop(gallery_guard);
        wait_for_attempt_cleanup(&state, "complete-before-delete").await;

        assert_eq!(published_gallery_file_count(root.path()), 1);
        assert!(state.queue_journal.list_all().is_empty());
        let detail = state
            .queue_journal
            .generation_batch("complete-batch")
            .unwrap();
        assert_eq!(detail.children[0].state, "complete");

        worker.abort();
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn shutdown_cancellation_retains_single_worker_attempt_without_publication() {
        let root = tempfile::tempdir().unwrap();
        let db = Arc::new(Some(mold_db::MetadataDb::open_in_memory().unwrap()));
        let blocker = Arc::new(GenerateBlocker::default());
        let (mut state, rx) = durable_state_with_engine(
            db,
            root.path(),
            MockEngine::blocking_generate(blocker.clone()),
        );
        state.queue_capacity = 1;
        admit_one_durable_batch(&state, "retain-on-shutdown", "retain-batch");

        let feeder_shutdown = tokio_util::sync::CancellationToken::new();
        let feeder = crate::durable_queue_feeder::spawn(state.clone(), feeder_shutdown.clone());
        let worker = tokio::spawn(crate::queue::run_queue_worker(rx, state.clone()));

        tokio::time::timeout(Duration::from_secs(5), async {
            while !blocker.entered.load(Ordering::SeqCst) {
                tokio::task::yield_now().await;
            }
        })
        .await
        .expect("single-worker fallback must enter inference");
        feeder_shutdown.cancel();
        feeder.await.unwrap();

        assert_eq!(state.generation_cancel.request_all(), 1);
        blocker.release();
        wait_for_attempt_cleanup(&state, "retain-on-shutdown").await;

        let rows = state.queue_journal.list_all();
        assert_eq!(rows.len(), 1, "shutdown must retain the durable attempt");
        assert_eq!(rows[0].id, "retain-on-shutdown");
        assert_eq!(
            rows[0].state,
            mold_db::generation_queue::QueueRowState::Queued
        );
        assert_eq!(
            published_gallery_file_count(root.path()),
            0,
            "shutdown-cancelled inference must not publish"
        );
        let detail = state
            .queue_journal
            .generation_batch("retain-batch")
            .unwrap();
        assert_ne!(detail.children[0].state, "complete");

        worker.abort();
    }

    fn seed_durable_projection_row(
        db: &Arc<Option<mold_db::MetadataDb>>,
        owner: &str,
        id: &str,
        state: mold_db::generation_queue::QueueRowState,
        created_at_ms: i64,
        payload_bytes: usize,
    ) {
        mold_db::generation_queue::insert(
            db.as_ref().as_ref().unwrap(),
            &mold_db::generation_queue::GenerationQueueRow {
                id: id.to_string(),
                owner_uuid: owner.to_string(),
                state,
                model: format!("model-{id}"),
                request_json: format!(
                    r#"{{"prompt":"{id}","source_image":"{}"}}"#,
                    "x".repeat(payload_bytes)
                ),
                media_set_id: None,
                admission_authority: None,
                output_dir: PathBuf::from(format!("/large-payload/{id}")),
                target_gpu: (id == "live-running").then_some(2),
                target_device_id: None,
                completion_payload: "full".repeat(1024),
                seed_pinned: id == "queued",
                dispatch_attempts: (id == "live-running") as u32,
                replay_seen: (id == "retained-running") as u32,
                held_reason: (state == mold_db::generation_queue::QueueRowState::Held)
                    .then(|| "held for review".to_string()),
                created_at_ms,
                updated_at_ms: created_at_ms,
                started_at_ms: (state == mold_db::generation_queue::QueueRowState::Running)
                    .then_some(created_at_ms),
            },
        )
        .unwrap();
    }

    /// A durably admitted job must be able to say what it is going to render.
    ///
    /// The queue projection is payload-free by construction — its SQL never
    /// selects `request_json` — so the listing hardcodes `metadata: None` and
    /// every durable job showed NO settings at all for its entire pre-dispatch
    /// window, which on this family is minutes. `GET /api/queue/:id` reads the
    /// one body the listing must not, and derives the same metadata shape the
    /// durable feeder derives at replay.
    #[tokio::test]
    async fn one_queue_job_reports_the_settings_the_payload_free_listing_cannot() {
        let root = tempfile::tempdir().unwrap();
        let db = Arc::new(Some(mold_db::MetadataDb::open_in_memory().unwrap()));
        let (state, _rx) = durable_state(db.clone(), root.path());
        let owner = state.queue_journal.owner_uuid().unwrap().to_string();
        mold_db::generation_queue::insert(
            db.as_ref().as_ref().unwrap(),
            &mold_db::generation_queue::GenerationQueueRow {
                id: "durable-only".to_string(),
                owner_uuid: owner,
                state: mold_db::generation_queue::QueueRowState::Queued,
                model: "flux-dev:q8".to_string(),
                request_json: serde_json::json!({
                    "prompt": "a lighthouse in a storm",
                    "model": "flux-dev:q8",
                    "width": 1024,
                    "height": 768,
                    "steps": 28,
                    "guidance": 3.5,
                    "seed": 4242,
                })
                .to_string(),
                media_set_id: None,
                admission_authority: None,
                output_dir: root.path().to_path_buf(),
                target_gpu: None,
                target_device_id: None,
                completion_payload: "full".to_string(),
                seed_pinned: true,
                dispatch_attempts: 0,
                replay_seen: 0,
                held_reason: None,
                created_at_ms: 1_000,
                updated_at_ms: 1_000,
                started_at_ms: None,
            },
        )
        .unwrap();

        // The listing still carries no settings — that is the property this
        // endpoint exists to work around, not a bug to fix there.
        let listing = app_with_state(state.clone())
            .oneshot(Request::get("/api/queue").body(Body::empty()).unwrap())
            .await
            .unwrap();
        assert_eq!(listing.status(), StatusCode::OK);
        let listed = json_body(listing).await;
        let row = listed["entries"]
            .as_array()
            .unwrap()
            .iter()
            .find(|entry| entry["id"] == "durable-only")
            .expect("the durable row is listed");
        assert!(
            row.get("metadata").is_none(),
            "the payload-free listing must not start reading request bodies: {row}"
        );

        let detail = app_with_state(state.clone())
            .oneshot(
                Request::get("/api/queue/durable-only")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(detail.status(), StatusCode::OK);
        let detail = json_body(detail).await;
        assert_eq!(detail["job"]["id"], "durable-only");
        assert_eq!(detail["job"]["model"], "flux-dev:q8");
        assert_eq!(detail["job"]["durable"], true);
        let metadata = &detail["job"]["metadata"];
        assert_eq!(metadata["prompt"], "a lighthouse in a storm");
        assert_eq!(metadata["width"], 1024);
        assert_eq!(metadata["height"], 768);
        assert_eq!(metadata["steps"], 28);
        assert_eq!(metadata["seed"], 4242);

        let missing = app_with_state(state)
            .oneshot(
                Request::get("/api/queue/no-such-job")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(missing.status(), StatusCode::NOT_FOUND);
    }

    /// The detail endpoint has to carry the PHASE, or the drawer it exists for
    /// still cannot say what a minutes-long `Preparing` window is doing.
    ///
    /// The work item is matched the way `studio/lib/queuePosition.ts` matches
    /// it: the entry whose `work_id` IS the job, or — for a batch parent, whose
    /// plan entries are its children — the first entry naming it as parent.
    /// Matching only `work_id` answers `null` for exactly the parent a client
    /// asks about.
    #[tokio::test]
    async fn one_queue_job_carries_its_planned_phase_including_for_a_batch_parent() {
        let root = tempfile::tempdir().unwrap();
        let db = Arc::new(Some(mold_db::MetadataDb::open_in_memory().unwrap()));
        let (state, _rx) = durable_state(db.clone(), root.path());
        let owner = state.queue_journal.owner_uuid().unwrap().to_string();
        for id in ["solo-job", "batch-parent"] {
            mold_db::generation_queue::insert(
                db.as_ref().as_ref().unwrap(),
                &mold_db::generation_queue::GenerationQueueRow {
                    id: id.to_string(),
                    owner_uuid: owner.clone(),
                    state: mold_db::generation_queue::QueueRowState::Queued,
                    model: "minimax-h3-fl2va:comfy-pruned-int8".to_string(),
                    request_json: serde_json::json!({
                        "prompt": "a lighthouse in a storm",
                        "model": "minimax-h3-fl2va:comfy-pruned-int8",
                        "width": 1344,
                        "height": 768,
                    })
                    .to_string(),
                    media_set_id: None,
                    admission_authority: None,
                    output_dir: root.path().to_path_buf(),
                    target_gpu: None,
                    target_device_id: None,
                    completion_payload: "full".to_string(),
                    seed_pinned: false,
                    dispatch_attempts: 0,
                    replay_seen: 0,
                    held_reason: None,
                    created_at_ms: 1_000,
                    updated_at_ms: 1_000,
                    started_at_ms: None,
                },
            )
            .unwrap();
        }
        let preparing = |work_id: &str, parent_id: &str| mold_core::QueueWorkItem {
            work_id: work_id.to_string(),
            parent_id: parent_id.to_string(),
            work_kind: "generation".to_string(),
            blocked_reason: Some(mold_core::QueueBlockedReason::Preparing),
            reason: Some("preparing".to_string()),
            preparation_elapsed_ms: Some(214_000),
            preparation_progress: Some(mold_core::QueuePreparationProgress {
                component: "Verifying MiniMax H3 artifacts".to_string(),
                bytes_done: 15_000_000_000,
                bytes_total: 37_000_000_000,
                phase_elapsed_ms: Some(96_000),
            }),
            ..Default::default()
        };
        state.scheduled_work.set_queue_work_items_for_tests(vec![
            preparing("solo-job", "solo-job"),
            preparing("batch-parent:0", "batch-parent"),
        ]);

        for id in ["solo-job", "batch-parent"] {
            let response = app_with_state(state.clone())
                .oneshot(
                    Request::get(format!("/api/queue/{id}"))
                        .body(Body::empty())
                        .unwrap(),
                )
                .await
                .unwrap();
            assert_eq!(response.status(), StatusCode::OK, "{id}");
            let body = json_body(response).await;
            let item = &body["work_item"];
            assert_eq!(item["blocked_reason"], "preparing", "{id}: {body}");
            assert_eq!(item["preparation_elapsed_ms"], 214_000, "{id}");
            assert_eq!(
                item["preparation_progress"]["component"], "Verifying MiniMax H3 artifacts",
                "{id}"
            );
            assert_eq!(
                item["preparation_progress"]["phase_elapsed_ms"], 96_000,
                "{id}: the phase's own age must survive the round trip"
            );
        }
    }

    #[tokio::test]
    async fn paginated_queue_reads_payload_free_pages_and_keeps_live_only_work_visible() {
        let root = tempfile::tempdir().unwrap();
        let db = Arc::new(Some(mold_db::MetadataDb::open_in_memory().unwrap()));
        let (state, _rx) = durable_state(db.clone(), root.path());
        let owner = state.queue_journal.owner_uuid().unwrap().to_string();
        for (id, row_state) in [
            (
                "live-running",
                mold_db::generation_queue::QueueRowState::Running,
            ),
            ("queued", mold_db::generation_queue::QueueRowState::Queued),
            (
                "retained-running",
                mold_db::generation_queue::QueueRowState::Running,
            ),
            ("held", mold_db::generation_queue::QueueRowState::Held),
            ("tail-a", mold_db::generation_queue::QueueRowState::Queued),
            ("tail-b", mold_db::generation_queue::QueueRowState::Queued),
        ] {
            // Six MiB of payload would dominate the legacy `list_all` result.
            // The paginated SQL/type regression test proves these columns are
            // absent; this route test proves the deep rows still page cleanly.
            seed_durable_projection_row(&db, &owner, id, row_state, 500, 1024 * 1024);
        }
        state
            .job_registry
            .register("live-running", "model-live-running");
        state.job_registry.mark_running("live-running", Some(2));
        // Models whose replay authority cannot be reconstructed (H3,
        // identity/reference requests) remain live in the registry only.
        state.job_registry.register("h3-live-only", "minimax-h3");
        let app = app_with_state(state.clone());

        let first = app
            .clone()
            .oneshot(
                Request::get("/api/queue?limit=2")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(first.status(), StatusCode::OK);
        let first = json_body(first).await;
        assert_eq!(first["page"]["limit"], 2);
        assert_eq!(first["page"]["offset"], 0);
        assert_eq!(first["page"]["returned"], 2);
        let cursor = first["page"]["next_cursor"]
            .as_str()
            .expect("deep queue has a next page")
            .to_string();
        assert!(!cursor.contains(':'));
        assert_eq!(first["entries"][0]["id"], "live-running");
        assert_eq!(first["entries"][0]["state"], "running");
        assert_eq!(first["entries"][0]["gpu"], 2);
        assert_eq!(first["entries"][0]["durable"], true);
        assert_eq!(first["entries"][0]["position"], 0);
        assert_eq!(first["entries"][1]["id"], "queued");
        assert_eq!(first["entries"][1]["position"], 1);
        assert_eq!(first["entries"][1]["seed_pinned"], true);
        assert_eq!(first["live_only_entries"].as_array().unwrap().len(), 1);
        assert_eq!(first["live_only_entries"][0]["id"], "h3-live-only");
        assert_eq!(first["live_only_entries"][0]["durable"], false);

        // The keyset is coordinates, not a foreign key. Deleting the row that
        // emitted the cursor cannot invalidate or rewind continuation.
        mold_db::generation_queue::delete(db.as_ref().as_ref().unwrap(), "queued").unwrap();
        let second = app
            .clone()
            .oneshot(
                Request::get(format!("/api/queue?limit=2&cursor={cursor}"))
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(second.status(), StatusCode::OK);
        let second = json_body(second).await;
        assert_eq!(second["page"]["offset"], 2);
        assert_eq!(second["entries"][0]["id"], "retained-running");
        assert_eq!(
            second["entries"][0]["state"], "queued",
            "an unowned interrupted running row keeps the legacy queued projection"
        );
        assert_eq!(second["entries"][0]["replayed"], true);
        assert_eq!(
            second["entries"][0]["position"], 2,
            "position continues from the runnable rows before this page"
        );
        assert_eq!(second["entries"][1]["id"], "held");
        assert_eq!(second["entries"][1]["state"], "held");
        assert_eq!(second["entries"][1]["held_reason"], "held for review");
        assert_eq!(second["live_only_entries"][0]["id"], "h3-live-only");

        // A held row on the previous page takes no place in line: the next
        // page's first runnable row is #3 (after live-running, queued,
        // retained-running), not #4 — the cursor carries the runnable count
        // separately from the traversal offset.
        let cursor = second["page"]["next_cursor"]
            .as_str()
            .expect("a third page follows the held row")
            .to_string();
        let third = app
            .clone()
            .oneshot(
                Request::get(format!("/api/queue?limit=2&cursor={cursor}"))
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(third.status(), StatusCode::OK);
        let third = json_body(third).await;
        assert_eq!(third["page"]["offset"], 4);
        assert_eq!(third["entries"][0]["id"], "tail-a");
        assert_eq!(third["entries"][0]["position"], 3);
        assert_eq!(third["entries"][1]["id"], "tail-b");
        assert_eq!(third["entries"][1]["position"], 4);
    }

    #[tokio::test]
    async fn a_queue_row_names_the_batch_that_can_retry_it() {
        // `POST /api/queue/{id}/retry` needs the whole authority. Everything
        // but the serving instance is a property of the row, so a listing
        // that omitted it made every client guess the parent batch — which is
        // why there was no CLI retry at all.
        let root = tempfile::tempdir().unwrap();
        let db = Arc::new(Some(mold_db::MetadataDb::open_in_memory().unwrap()));
        let (state, _rx) = durable_state(db.clone(), root.path());
        admit_one_durable_batch(&state, "child-job", "batch-1");
        let app = app_with_state(state);

        let listing = json_body(
            app.clone()
                .oneshot(Request::get("/api/queue").body(Body::empty()).unwrap())
                .await
                .unwrap(),
        )
        .await;
        let row = &listing["entries"][0];
        assert_eq!(row["id"], "child-job");
        assert_eq!(row["batch_id"], "batch-1");
        assert_eq!(row["client_batch_id"], "client-batch-1");
        assert_eq!(row["batch_index"], 1, "batch indices are one-based");

        // Asking about the one job answers the same identity, so a client
        // that holds only a job id can compose the retry without a listing.
        let detail = json_body(
            app.oneshot(
                Request::get("/api/queue/child-job")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap(),
        )
        .await;
        assert_eq!(detail["job"]["batch_id"], "batch-1");
        assert_eq!(detail["job"]["client_batch_id"], "client-batch-1");
        assert_eq!(detail["job"]["batch_index"], 1);
    }

    #[tokio::test]
    async fn a_single_job_read_reports_the_hold_s_own_retryable_bit() {
        // The payload row carries no `retryable` column, so the single-job
        // route used to synthesize one as `true` — telling an operator to
        // retry a row `POST /api/queue/{id}/retry` answers 409 for.
        let root = tempfile::tempdir().unwrap();
        let db = Arc::new(Some(mold_db::MetadataDb::open_in_memory().unwrap()));
        let (state, _rx) = durable_state(db.clone(), root.path());
        let owner = state.queue_journal.owner_uuid().unwrap().to_string();
        admit_one_durable_batch(&state, "needs-repair", "batch-repair");
        assert_eq!(
            mold_db::generation_batches::hold_owned(
                db.as_ref().as_ref().unwrap(),
                &owner,
                "needs-repair",
                None,
                "publication authority is invalid",
                None,
                false,
                600,
            )
            .unwrap(),
            mold_db::generation_batches::OwnedHold::Held
        );
        let app = app_with_state(state);

        let listing = json_body(
            app.clone()
                .oneshot(Request::get("/api/queue").body(Body::empty()).unwrap())
                .await
                .unwrap(),
        )
        .await;
        assert_eq!(listing["entries"][0]["retryable"], false);

        let detail = json_body(
            app.oneshot(
                Request::get("/api/queue/needs-repair")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap(),
        )
        .await;
        assert_eq!(detail["job"]["state"], "held");
        assert_eq!(
            detail["job"]["retryable"], false,
            "the single-job read must not promise a retry the retry route refuses"
        );
        assert_eq!(detail["job"]["error"], "publication authority is invalid");
        // The settings the payload-free listing cannot carry still arrive.
        assert!(detail["job"]["metadata"]["prompt"].is_string());
    }

    #[tokio::test]
    async fn retryable_dependency_hold_is_visible_and_resumes_through_the_retry_api() {
        let root = tempfile::tempdir().unwrap();
        let db = Arc::new(Some(mold_db::MetadataDb::open_in_memory().unwrap()));
        let (state, _rx) = durable_state(db.clone(), root.path());
        let owner = state.queue_journal.owner_uuid().unwrap().to_string();
        admit_one_durable_batch(&state, "retryable-preparation", "batch-retry");
        assert_eq!(
            mold_db::generation_batches::hold_owned(
                db.as_ref().as_ref().unwrap(),
                &owner,
                "retryable-preparation",
                None,
                "dependency download failed",
                None,
                true,
                600,
            )
            .unwrap(),
            mold_db::generation_batches::OwnedHold::Held
        );
        let app = app_with_state(state.clone());

        let listing = json_body(
            app.clone()
                .oneshot(Request::get("/api/queue").body(Body::empty()).unwrap())
                .await
                .unwrap(),
        )
        .await;
        let held = &listing["entries"][0];
        assert_eq!(held["state"], "held");
        assert_eq!(held["error"], "dependency download failed");
        assert_eq!(held["retryable"], true);

        let admitted: mold_core::GenerationBatchStatus = serde_json::from_value(
            json_body(
                app.clone()
                    .oneshot(
                        Request::get("/api/generation-batches/batch-retry")
                            .body(Body::empty())
                            .unwrap(),
                    )
                    .await
                    .unwrap(),
            )
            .await,
        )
        .unwrap();
        let authority =
            mold_core::GenerationBatchAuthority::from_admission(&admitted, "client-batch-retry")
                .unwrap();
        let retry =
            mold_core::GenerationRetryRequest::from_authority(&authority, "retryable-preparation");
        assert_eq!(retry.instance_id, *state.instance_id);

        let mut replacement_authority = retry.clone();
        replacement_authority.instance_id = "replacement".to_string();
        let replacement = app
            .clone()
            .oneshot(
                Request::post("/api/queue/retryable-preparation/retry")
                    .header("content-type", "application/json")
                    .body(Body::from(
                        serde_json::to_vec(&replacement_authority).unwrap(),
                    ))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(replacement.status(), StatusCode::CONFLICT);
        assert_eq!(
            json_body(replacement).await["code"],
            "QUEUE_JOB_AUTHORITY_MISMATCH"
        );

        let response = app
            .oneshot(
                Request::post("/api/queue/retryable-preparation/retry")
                    .header("content-type", "application/json")
                    .body(Body::from(serde_json::to_vec(&retry).unwrap()))
                    .unwrap(),
            )
            .await
            .unwrap();
        let status = response.status();
        assert_eq!(status, StatusCode::ACCEPTED);
        let row = state.queue_journal.list_all().pop().unwrap();
        assert_eq!(row.state, mold_db::generation_queue::QueueRowState::Queued);
        assert_eq!(row.held_reason, None);
    }

    #[tokio::test]
    async fn queue_default_and_explicit_pages_are_bounded_by_the_runtime_window() {
        let root = tempfile::tempdir().unwrap();
        let db = Arc::new(Some(mold_db::MetadataDb::open_in_memory().unwrap()));
        let (mut state, _rx) = durable_state(db.clone(), root.path());
        state.queue_capacity = 2;
        let owner = state.queue_journal.owner_uuid().unwrap().to_string();
        for index in 0..5 {
            seed_durable_projection_row(
                &db,
                &owner,
                &format!("deep-{index}"),
                mold_db::generation_queue::QueueRowState::Queued,
                index,
                1024 * 1024,
            );
        }
        let app = app_with_state(state);

        let default = json_body(
            app.clone()
                .oneshot(Request::get("/api/queue").body(Body::empty()).unwrap())
                .await
                .unwrap(),
        )
        .await;
        assert_eq!(default["entries"].as_array().unwrap().len(), 2);
        assert_eq!(default["page"]["limit"], 2);
        let cursor = default["page"]["next_cursor"]
            .as_str()
            .expect("bounded default exposes continuation")
            .to_string();

        let clamped = json_body(
            app.clone()
                .oneshot(
                    Request::get("/api/queue?limit=999999")
                        .body(Body::empty())
                        .unwrap(),
                )
                .await
                .unwrap(),
        )
        .await;
        assert_eq!(clamped["entries"].as_array().unwrap().len(), 2);
        assert_eq!(clamped["page"]["limit"], 2);

        let continued = json_body(
            app.oneshot(
                Request::get(format!("/api/queue?cursor={cursor}"))
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap(),
        )
        .await;
        assert_eq!(continued["entries"].as_array().unwrap().len(), 2);
        assert_eq!(continued["page"]["limit"], 2);
        assert_eq!(continued["page"]["offset"], 2);
        assert_eq!(continued["entries"][0]["id"], "deep-2");
    }

    #[tokio::test(flavor = "current_thread")]
    async fn heterogeneous_batch_acknowledges_before_model_resolution_or_catalog_work() {
        let root = tempfile::tempdir().unwrap();
        let db = Arc::new(Some(mold_db::MetadataDb::open_in_memory().unwrap()));
        let (mut state, _rx) = durable_state(db, root.path());
        install_authoritative_v2(&mut state);
        let journal = state.queue_journal.clone();
        let app = app_with_state(state.clone());
        let unavailable = generate_body_for_model(
            "must not become a stranded durable row",
            "not-installed-at-admission",
            512,
            512,
        );
        let body = serde_json::json!({
            "client_batch_id": uuid::Uuid::new_v4().to_string(),
            "requests": [serde_json::from_str::<serde_json::Value>(
                &unavailable
            ).unwrap()],
        });

        let response = app
            .oneshot(json_request("POST", "/api/generation-batches", body))
            .await
            .unwrap();

        let status = response.status();
        let response_body = json_body(response).await;
        assert_eq!(status, StatusCode::ACCEPTED, "{response_body}");
        assert_eq!(journal.list_all().len(), 1);
        assert_eq!(journal.list_all()[0].model, "not-installed-at-admission");
        assert_eq!(
            state.job_registry.len(),
            0,
            "preparation starts only in the feeder"
        );
    }

    /// An output format the recipe does not advertise will NEVER become
    /// valid, so it belongs at the door. Before this it was only checked in
    /// preparation, which runs after durable acknowledgement: the row was
    /// accepted, held, and then failed with an error the client could have
    /// been given at submit time.
    #[tokio::test(flavor = "current_thread")]
    async fn an_unavailable_output_format_is_a_422_at_admission_not_a_hold() {
        let (state, _rx, _root) = durable_test_state(MockEngine::ready());
        let journal = state.queue_journal.clone();
        let app = app_with_state(state.clone());
        let mut request_json = serde_json::from_str::<serde_json::Value>(&generate_body_for_model(
            "a cat in a gif nobody can render",
            "flux-dev:q8",
            1024,
            1024,
        ))
        .unwrap();
        request_json["output_format"] = serde_json::json!("gif");
        let body = serde_json::json!({
            "client_batch_id": uuid::Uuid::new_v4().to_string(),
            "requests": [request_json],
        });

        let response = app
            .oneshot(json_request("POST", "/api/generation-batches", body))
            .await
            .unwrap();

        let status = response.status();
        let response_body = json_body(response).await;
        assert_eq!(status, StatusCode::UNPROCESSABLE_ENTITY, "{response_body}");
        assert_eq!(response_body["code"], "VALIDATION_ERROR");
        assert_eq!(
            response_body["error"],
            "requests[1]: output format 'gif' is not available for this recipe"
        );
        assert!(
            journal.list_all().is_empty(),
            "a refused request must enqueue nothing"
        );
    }

    /// A mesh model stores binary glTF and nothing else, so an explicit
    /// raster format is COERCED rather than refused: an older client that
    /// always sends `png` must still get its mesh, exactly as the CLI already
    /// resolved it.
    #[tokio::test(flavor = "current_thread")]
    async fn a_mesh_request_pins_an_explicit_raster_format_to_glb() {
        let (state, _rx, _root) = durable_test_state(MockEngine::ready());
        let journal = state.queue_journal.clone();
        let app = app_with_state(state.clone());
        let mut request_json = serde_json::from_str::<serde_json::Value>(&generate_body_for_model(
            "an armchair",
            "hunyuan3d-mini-turbo:fp16",
            0,
            0,
        ))
        .unwrap();
        request_json["output_format"] = serde_json::json!("png");
        request_json["steps"] = serde_json::json!(5);
        request_json["source_image"] =
            serde_json::json!(base64::engine::general_purpose::STANDARD.encode(minimal_png()));
        let body = serde_json::json!({
            "client_batch_id": uuid::Uuid::new_v4().to_string(),
            "requests": [request_json],
        });

        let response = app
            .oneshot(json_request("POST", "/api/generation-batches", body))
            .await
            .unwrap();

        let status = response.status();
        let response_body = json_body(response).await;
        assert_eq!(status, StatusCode::ACCEPTED, "{response_body}");
        let rows = journal.list_all();
        assert_eq!(rows.len(), 1, "{response_body}");
        assert_eq!(rows[0].model, "hunyuan3d-mini-turbo:fp16");
        // The persisted request is what replay re-renders, so the pin has to
        // be in the row rather than applied again later.
        let admitted: mold_core::GenerateRequest =
            serde_json::from_str(&rows[0].request_json).unwrap();
        assert_eq!(
            admitted.output_format,
            Some(mold_core::OutputFormat::Glb),
            "an explicit png on a mesh model must be pinned before the row is written"
        );
    }

    /// An LTX-2.5 GGUF tier is an ordinary durable admission since the
    /// native quantized runtime landed (#1414): the batch is accepted, the
    /// row is journaled, and preparation starts only in the feeder.
    #[tokio::test(flavor = "current_thread")]
    async fn ltx25_gguf_batch_is_admitted_to_the_durable_queue() {
        let root = tempfile::tempdir().unwrap();
        let db = Arc::new(Some(mold_db::MetadataDb::open_in_memory().unwrap()));
        let (mut state, _rx) = durable_state(db, root.path());
        install_authoritative_v2(&mut state);
        let journal = state.queue_journal.clone();
        let app = app_with_state(state.clone());
        let gguf = generate_body_for_model(
            "a quantized transformer this build now runs",
            mold_core::ltx25_manifest::DISTILLED_Q4,
            512,
            512,
        );
        let mut request_json = serde_json::from_str::<serde_json::Value>(&gguf).unwrap();
        // The shared body builder is image-shaped; LTX-2 requires a video
        // container. WHICH video container depends on the encoders this
        // binary linked — admission now refuses a format the delivery-
        // qualified recipe does not advertise, and a test build without the
        // `mp4` feature genuinely cannot deliver MP4. Asking for one here
        // would test the encoder set rather than the subject of this test,
        // which is that a GGUF tier is an ordinary durable admission.
        request_json["output_format"] =
            serde_json::json!(if cfg!(feature = "mp4") { "mp4" } else { "apng" });
        let body = serde_json::json!({
            "client_batch_id": uuid::Uuid::new_v4().to_string(),
            "requests": [request_json],
        });

        let response = app
            .oneshot(json_request("POST", "/api/generation-batches", body))
            .await
            .unwrap();

        let status = response.status();
        let response_body = json_body(response).await;
        assert_eq!(status, StatusCode::ACCEPTED, "{response_body}");
        let rows = journal.list_all();
        assert_eq!(rows.len(), 1);
        assert_eq!(rows[0].model, mold_core::ltx25_manifest::DISTILLED_Q4);
        assert_eq!(
            state.job_registry.len(),
            0,
            "preparation starts only in the feeder"
        );
    }

    #[tokio::test(flavor = "current_thread")]
    async fn direct_stream_uses_queue_first_admission_without_media_or_operation_header() {
        let root = tempfile::tempdir().unwrap();
        let db = Arc::new(Some(mold_db::MetadataDb::open_in_memory().unwrap()));
        let (mut state, _rx) = durable_state(db, root.path());
        install_authoritative_v2(&mut state);
        let journal = state.queue_journal.clone();
        let app = app_with_state(state);

        let response = app
            .oneshot(json_request(
                "POST",
                "/api/generate/stream",
                serde_json::from_str::<serde_json::Value>(&generate_body_for_model(
                    "queue before resolving",
                    "not-installed-at-admission",
                    512,
                    512,
                ))
                .unwrap(),
            ))
            .await
            .unwrap();

        assert_eq!(response.status(), StatusCode::OK);
        let rows = journal.list_all();
        assert_eq!(rows.len(), 1);
        assert_eq!(rows[0].model, "not-installed-at-admission");
    }

    #[tokio::test(flavor = "current_thread")]
    async fn direct_routes_reject_multi_output_requests_before_admission() {
        let root = tempfile::tempdir().unwrap();
        let db = Arc::new(Some(mold_db::MetadataDb::open_in_memory().unwrap()));
        let (mut state, _rx) = durable_state(db, root.path());
        install_authoritative_v2(&mut state);
        let journal = state.queue_journal.clone();
        let app = app_with_state(state);
        let mut body: serde_json::Value =
            serde_json::from_str(&durable_direct_media_body("one media authority")).unwrap();
        body["batch_size"] = serde_json::json!(64);

        for path in ["/api/generate", "/api/generate/stream"] {
            let response = app
                .clone()
                .oneshot(json_request("POST", path, body.clone()))
                .await
                .unwrap();
            assert_eq!(response.status(), StatusCode::UNPROCESSABLE_ENTITY);
            assert_eq!(
                json_body(response).await["code"],
                "DIRECT_BATCH_UNSUPPORTED"
            );
        }
        assert!(journal.list_all().is_empty());
    }

    /// A host with no gallery does not generate.
    ///
    /// It used to render singletons on the attached path and simply not save
    /// them. There is no attached path any more — a queued print's only
    /// delivery is the gallery file, so a host that writes none cannot admit
    /// one — and every generation route says so with the same typed refusal.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn output_disabled_hosts_refuse_every_generation_route() {
        let root = tempfile::tempdir().unwrap();
        let db = Arc::new(Some(mold_db::MetadataDb::open_in_memory().unwrap()));
        let (mut state, rx) = durable_state(db, root.path());
        install_authoritative_v2(&mut state);
        state.output_disabled_override = true;
        let journal = state.queue_journal.clone();
        let worker = tokio::spawn(crate::queue::run_queue_worker(rx, state.clone()));
        let app = app_with_state(state);

        for path in [
            "/api/generate",
            "/api/generate/stream",
            "/api/generation-batches",
        ] {
            let single: serde_json::Value =
                serde_json::from_str(&generate_body("no gallery here", 64, 64)).unwrap();
            let body = if path == "/api/generation-batches" {
                serde_json::json!({
                    "client_batch_id": uuid::Uuid::new_v4().to_string(),
                    "requests": [single],
                })
            } else {
                single
            };
            let refused = app
                .clone()
                .oneshot(json_request("POST", path, body))
                .await
                .unwrap();
            assert_eq!(refused.status(), StatusCode::SERVICE_UNAVAILABLE, "{path}");
            assert_eq!(
                json_body(refused).await["code"],
                "DURABLE_ADMISSION_UNAVAILABLE",
                "{path}"
            );
        }
        assert!(journal.list_all().is_empty());
        worker.abort();
    }

    #[tokio::test(flavor = "current_thread")]
    async fn dark_media_services_refuse_media_before_direct_raw_or_sse_admission() {
        let root = tempfile::tempdir().unwrap();
        let db = Arc::new(Some(mold_db::MetadataDb::open_in_memory().unwrap()));
        let (mut state, _rx) = durable_state(db, root.path());
        install_authoritative_v2(&mut state);
        state.queue_journal.set_durable_media_ready(false);
        let journal = state.queue_journal.clone();
        let app = app_with_state(state);
        let body: serde_json::Value =
            serde_json::from_str(&durable_direct_media_body("dark media store")).unwrap();
        let h3_body = {
            let mut request: serde_json::Value =
                serde_json::from_str(&generate_body("dark H3 authority", 64, 64)).unwrap();
            request["model"] = serde_json::json!(mold_core::minimax_h3::FL2VA_COMFY);
            request
        };

        // DELIBERATE CHANGE, not a weakened assertion. This gate used to refuse
        // unconditionally while the gate below it consulted
        // `explicitly_requested`, so a degraded media store 503'd EVERY
        // media-carrying and H3 generation on both direct routes — behind a
        // status both client classifiers read as transient, on a host whose
        // capabilities were already telling clients to use the attached path.
        // A caller that explicitly demanded durability still gets the typed
        // refusal; a caller that did not now falls back, which is what
        // capability discovery promised it. The fallback half is asserted by
        // `a_degraded_media_store_falls_back_instead_of_refusing`, which drives
        // the gate directly rather than running a generation.
        for path in ["/api/generate", "/api/generate/stream"] {
            for request in [&body, &h3_body] {
                let response = app
                    .clone()
                    .oneshot(
                        Request::post(path)
                            .header("content-type", "application/json")
                            .header("x-mold-operation-id", uuid::Uuid::new_v4().to_string())
                            .body(Body::from(serde_json::to_vec(request).unwrap()))
                            .unwrap(),
                    )
                    .await
                    .unwrap();
                assert_eq!(response.status(), StatusCode::SERVICE_UNAVAILABLE);
                assert_eq!(
                    json_body(response).await["code"],
                    "DURABLE_MEDIA_UNAVAILABLE"
                );
            }
        }

        for request in [body, h3_body] {
            let response = app
                .clone()
                .oneshot(json_request(
                    "POST",
                    "/api/generation-batches",
                    serde_json::json!({
                        "client_batch_id": uuid::Uuid::new_v4().to_string(),
                        "requests": [request],
                    }),
                ))
                .await
                .unwrap();
            assert_eq!(response.status(), StatusCode::SERVICE_UNAVAILABLE);
            assert_eq!(
                json_body(response).await["code"],
                "DURABLE_MEDIA_UNAVAILABLE"
            );
        }
        assert!(journal.list_all().is_empty());
    }

    #[tokio::test(flavor = "current_thread")]
    async fn dark_media_services_still_admit_media_free_direct_and_batch_requests() {
        let root = tempfile::tempdir().unwrap();
        let db = Arc::new(Some(mold_db::MetadataDb::open_in_memory().unwrap()));
        let (mut state, _rx) = durable_state(db, root.path());
        install_authoritative_v2(&mut state);
        state.queue_journal.set_durable_media_ready(false);
        let mut request: GenerateRequest =
            serde_json::from_str(&generate_body("media-free admission", 64, 64)).unwrap();

        crate::routes::direct_durable_admission(&state, &mut request)
            .await
            .expect("encrypted-media readiness must not disable media-free direct admission");

        let journal = state.queue_journal.clone();
        let response = app_with_state(state)
            .oneshot(json_request(
                "POST",
                "/api/generation-batches",
                serde_json::json!({
                    "client_batch_id": uuid::Uuid::new_v4().to_string(),
                    "requests": [request],
                }),
            ))
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::ACCEPTED);
        assert_eq!(journal.list_all().len(), 1);
    }

    #[tokio::test(flavor = "current_thread")]
    async fn media_free_batch_replays_after_missing_or_corrupt_media_key_restart() {
        for corrupt in [false, true] {
            let root = tempfile::tempdir().unwrap();
            let db = Arc::new(Some(mold_db::MetadataDb::open_in_memory().unwrap()));
            let body = serde_json::json!({
                "client_batch_id": uuid::Uuid::new_v4().to_string(),
                "requests": [serde_json::from_str::<serde_json::Value>(
                    &generate_body("store-independent replay", 64, 64)
                ).unwrap()],
            });
            let (mut state, rx) = durable_state(db.clone(), root.path());
            install_authoritative_v2(&mut state);
            let admitted = app_with_state(state)
                .oneshot(json_request(
                    "POST",
                    "/api/generation-batches",
                    body.clone(),
                ))
                .await
                .unwrap();
            assert_eq!(admitted.status(), StatusCode::ACCEPTED);
            let admitted = json_body(admitted).await;
            drop(rx);

            // Leave undeniable encrypted-store evidence behind so a missing
            // key cannot be silently regenerated during restart.
            let store = crate::queue_media_store::QueueMediaStore::open(root.path())
                .unwrap()
                .store;
            store
                .seal(
                    "orphan-owner",
                    "orphan-job",
                    vec![crate::queue_media_store::SealMedia::bytes(
                        "source",
                        "orphan.bin",
                        vec![1],
                    )],
                )
                .unwrap();
            drop(store);
            let key = root.path().join("queue-media/master.key");
            if corrupt {
                std::fs::write(&key, [7_u8; 31]).unwrap();
            } else {
                std::fs::remove_file(&key).unwrap();
            }

            let (mut restarted, _rx) = durable_state_with_engine_and_media_readiness(
                db.clone(),
                root.path(),
                MockEngine::ready(),
                false,
            );
            install_authoritative_v2(&mut restarted);
            assert!(restarted
                .queue_journal
                .durable_media_capabilities()
                .is_none());
            let app = app_with_state(restarted);
            let replay = app
                .clone()
                .oneshot(json_request(
                    "POST",
                    "/api/generation-batches",
                    body.clone(),
                ))
                .await
                .unwrap();
            assert_eq!(replay.status(), StatusCode::OK);
            assert_eq!(json_body(replay).await["id"], admitted["id"]);

            let mut changed = body.clone();
            changed["requests"][0]["prompt"] = serde_json::json!("changed after restart");
            let conflict = app
                .oneshot(json_request("POST", "/api/generation-batches", changed))
                .await
                .unwrap();
            assert_eq!(conflict.status(), StatusCode::CONFLICT);
            assert_eq!(
                json_body(conflict).await["code"],
                "GENERATION_BATCH_IDEMPOTENCY_CONFLICT"
            );
        }
    }

    #[tokio::test(flavor = "current_thread")]
    async fn legacy_receipt_migrates_before_media_key_loss_and_replays_after_restart() {
        let root = tempfile::tempdir().unwrap();
        let db = Arc::new(Some(mold_db::MetadataDb::open_in_memory().unwrap()));
        let client_id = uuid::Uuid::new_v4().to_string();
        let body = serde_json::json!({
            "client_batch_id": client_id,
            "requests": [serde_json::from_str::<serde_json::Value>(
                &generate_body("legacy receipt migration", 64, 64)
            ).unwrap()],
        });
        let (mut state, rx) = durable_state(db.clone(), root.path());
        install_authoritative_v2(&mut state);
        let owner = state.queue_journal.owner_uuid().unwrap().to_string();
        let app = app_with_state(state);
        let admitted = app
            .clone()
            .oneshot(json_request(
                "POST",
                "/api/generation-batches",
                body.clone(),
            ))
            .await
            .unwrap();
        assert_eq!(admitted.status(), StatusCode::ACCEPTED);
        let admitted = json_body(admitted).await;

        let current_receipt = db
            .as_ref()
            .as_ref()
            .unwrap()
            .with_conn(|conn| {
                conn.query_row(
                    "SELECT request_sha256 FROM generation_batches WHERE client_batch_id = ?1",
                    [&client_id],
                    |row| row.get::<_, String>(0),
                )
                .map_err(Into::into)
            })
            .unwrap();
        assert!(current_receipt.starts_with("generation-v2."));
        let requests = serde_json::from_value(body["requests"].clone()).unwrap();
        let fingerprint =
            crate::queue_media_admission::test_operation_fingerprint(&client_id, requests);
        let store = crate::queue_media_store::QueueMediaStore::open_existing(root.path()).unwrap();
        let legacy = store
            .seal_operation_receipt_v1(&owner, &client_id, &fingerprint)
            .unwrap();
        store
            .seal(
                "orphan-owner",
                "orphan-job",
                vec![crate::queue_media_store::SealMedia::bytes(
                    "source",
                    "orphan.bin",
                    vec![1],
                )],
            )
            .unwrap();
        drop(store);
        db.as_ref()
            .as_ref()
            .unwrap()
            .with_conn(|conn| {
                conn.execute(
                    "UPDATE generation_batches SET request_sha256 = ?1
                      WHERE client_batch_id = ?2",
                    (legacy.as_str(), &client_id),
                )?;
                Ok(())
            })
            .unwrap();

        let migrated = app
            .clone()
            .oneshot(json_request(
                "POST",
                "/api/generation-batches",
                body.clone(),
            ))
            .await
            .unwrap();
        assert_eq!(migrated.status(), StatusCode::OK);
        assert_eq!(json_body(migrated).await["id"], admitted["id"]);
        let migrated_receipt = db
            .as_ref()
            .as_ref()
            .unwrap()
            .with_conn(|conn| {
                conn.query_row(
                    "SELECT request_sha256 FROM generation_batches WHERE client_batch_id = ?1",
                    [&client_id],
                    |row| row.get::<_, String>(0),
                )
                .map_err(Into::into)
            })
            .unwrap();
        assert!(migrated_receipt.starts_with("generation-v2."));
        assert_ne!(migrated_receipt, legacy.as_str());

        drop(app);
        drop(rx);
        std::fs::remove_file(root.path().join("queue-media/master.key")).unwrap();
        let (mut restarted, _rx) = durable_state_with_engine_and_media_readiness(
            db,
            root.path(),
            MockEngine::ready(),
            false,
        );
        install_authoritative_v2(&mut restarted);
        assert!(restarted
            .queue_journal
            .durable_media_capabilities()
            .is_none());
        let replay = app_with_state(restarted)
            .oneshot(json_request("POST", "/api/generation-batches", body))
            .await
            .unwrap();
        assert_eq!(replay.status(), StatusCode::OK);
        assert_eq!(json_body(replay).await["id"], admitted["id"]);
    }

    #[tokio::test(flavor = "current_thread")]
    async fn missing_or_corrupt_admission_key_degrades_capability_without_stopping_service() {
        for corrupt in [false, true] {
            let root = tempfile::tempdir().unwrap();
            let db = Arc::new(Some(mold_db::MetadataDb::open_in_memory().unwrap()));
            let body = serde_json::json!({
                "client_batch_id": uuid::Uuid::new_v4().to_string(),
                "requests": [serde_json::from_str::<serde_json::Value>(
                    &generate_body("receipt-key evidence", 64, 64)
                ).unwrap()],
            });
            let (mut state, rx) = durable_state(db.clone(), root.path());
            install_authoritative_v2(&mut state);
            let admitted = app_with_state(state)
                .oneshot(json_request(
                    "POST",
                    "/api/generation-batches",
                    body.clone(),
                ))
                .await
                .unwrap();
            assert_eq!(admitted.status(), StatusCode::ACCEPTED);
            drop(rx);

            let key = root.path().join("queue-media/generation-admission.key");
            if corrupt {
                std::fs::write(&key, [3_u8; 11]).unwrap();
            } else {
                std::fs::remove_file(&key).unwrap();
            }

            let (mut restarted, _rx) = durable_state_with_admission_policy(
                db,
                root.path(),
                MockEngine::ready(),
                true,
                false,
                "test-instance",
            );
            install_authoritative_v2(&mut restarted);
            let lifecycle = restarted.queue_journal.queue_media_lifecycle().unwrap();
            assert!(!crate::install_durable_admission_if_available(
                &restarted.queue_journal,
                lifecycle,
                restarted.queue_capacity,
            ));
            let app = app_with_state(restarted);
            let status = app
                .clone()
                .oneshot(empty_request("GET", "/api/status"))
                .await
                .unwrap();
            assert_eq!(status.status(), StatusCode::OK);
            let capabilities = app
                .clone()
                .oneshot(empty_request("GET", "/api/capabilities"))
                .await
                .unwrap();
            assert_eq!(capabilities.status(), StatusCode::OK);
            let capabilities = json_body(capabilities).await;
            assert!(capabilities["queue"]["heterogeneous_batch_max_outputs"].is_null());
            let rejected = app
                .oneshot(json_request("POST", "/api/generation-batches", body))
                .await
                .unwrap();
            assert_eq!(rejected.status(), StatusCode::SERVICE_UNAVAILABLE);
        }
    }

    #[tokio::test(flavor = "current_thread")]
    async fn orphan_owner_receipt_prevents_global_key_regeneration_for_fresh_owner() {
        let root = tempfile::tempdir().unwrap();
        let db = Arc::new(Some(mold_db::MetadataDb::open_in_memory().unwrap()));
        let body = serde_json::json!({
            "client_batch_id": uuid::Uuid::new_v4().to_string(),
            "requests": [serde_json::from_str::<serde_json::Value>(
                &generate_body("orphan receipt evidence", 64, 64)
            ).unwrap()],
        });
        let (mut first, first_rx) = durable_state(db.clone(), root.path());
        install_authoritative_v2(&mut first);
        let first_owner = first.queue_journal.owner_uuid().unwrap().to_string();
        let first_app = app_with_state(first);
        let admitted = first_app
            .clone()
            .oneshot(json_request("POST", "/api/generation-batches", body))
            .await
            .unwrap();
        assert_eq!(admitted.status(), StatusCode::ACCEPTED);

        // Claim a distinct owner while the first still holds its identity.
        // Once the first stops, its generation-v2 row is orphan evidence for
        // the one MOLD_HOME-global admission key.
        let (mut fresh, _fresh_rx) = durable_state_with_admission_policy(
            db,
            root.path(),
            MockEngine::ready(),
            true,
            false,
            "fresh-instance",
        );
        assert_ne!(fresh.queue_journal.owner_uuid().unwrap(), first_owner);
        drop(first_app);
        drop(first_rx);
        install_authoritative_v2(&mut fresh);

        let key = root.path().join("queue-media/generation-admission.key");
        std::fs::remove_file(&key).unwrap();
        let lifecycle = fresh.queue_journal.queue_media_lifecycle().unwrap();
        assert!(!crate::install_durable_admission_if_available(
            &fresh.queue_journal,
            lifecycle,
            fresh.queue_capacity,
        ));
        assert!(
            !key.exists(),
            "receipt evidence must prevent silent rekeying"
        );

        let app = app_with_state(fresh);
        assert_eq!(
            app.clone()
                .oneshot(empty_request("GET", "/api/status"))
                .await
                .unwrap()
                .status(),
            StatusCode::OK
        );
        let capabilities = app
            .oneshot(empty_request("GET", "/api/capabilities"))
            .await
            .unwrap();
        let capabilities = json_body(capabilities).await;
        assert!(capabilities["queue"]["heterogeneous_batch_max_outputs"].is_null());
    }

    /// A host can never refuse one route for a reason it accepts on another:
    /// with generation unavailable (every device disabled), the batch route
    /// refuses NEW work exactly as `/api/generate` does, instead of parking
    /// rows nothing will run.
    #[tokio::test(flavor = "current_thread")]
    async fn unavailable_generation_refuses_new_batch_work_on_the_batch_route_too() {
        let root = tempfile::tempdir().unwrap();
        let db = Arc::new(Some(mold_db::MetadataDb::open_in_memory().unwrap()));
        let (mut state, _rx) = durable_state(db, root.path());
        install_authoritative_v2(&mut state);
        state.set_generation_unavailable("every device is disabled");
        let journal = state.queue_journal.clone();
        let response = app_with_state(state)
            .oneshot(json_request(
                "POST",
                "/api/generation-batches",
                serde_json::json!({
                    "client_batch_id": uuid::Uuid::new_v4().to_string(),
                    "requests": [serde_json::from_str::<serde_json::Value>(
                        &generate_body("nothing can run this", 64, 64)
                    ).unwrap()],
                }),
            ))
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::SERVICE_UNAVAILABLE);
        assert!(journal.list_all().is_empty(), "nothing was parked");
    }

    #[tokio::test(flavor = "current_thread")]
    async fn output_disabled_batch_admission_is_a_typed_unavailable_response() {
        let root = tempfile::tempdir().unwrap();
        let db = Arc::new(Some(mold_db::MetadataDb::open_in_memory().unwrap()));
        let (mut state, _rx) = durable_state(db, root.path());
        install_authoritative_v2(&mut state);
        state.output_disabled_override = true;
        let response = app_with_state(state)
            .oneshot(json_request(
                "POST",
                "/api/generation-batches",
                serde_json::json!({
                    "client_batch_id": uuid::Uuid::new_v4().to_string(),
                    "requests": [serde_json::from_str::<serde_json::Value>(
                        &generate_body("no gallery authority", 64, 64)
                    ).unwrap()],
                }),
            ))
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::SERVICE_UNAVAILABLE);
        assert_eq!(
            json_body(response).await["code"],
            "DURABLE_ADMISSION_UNAVAILABLE"
        );
    }

    /// `hdr_exr_dir` is refused for a reason that predates durability and has
    /// nothing to do with it: it names an output directory on the machine
    /// doing inference, and an HTTP client may not choose one. The refusal
    /// keeps its own actionable wording and happens BEFORE acceptance, so a
    /// caller gets a `422` it can act on rather than a job that holds.
    #[tokio::test(flavor = "current_thread")]
    async fn a_client_supplied_hdr_directory_is_refused_before_acceptance() {
        let root = tempfile::tempdir().unwrap();
        let db = Arc::new(Some(mold_db::MetadataDb::open_in_memory().unwrap()));
        let (mut state, _rx) = durable_state(db, root.path());
        install_authoritative_v2(&mut state);
        let journal = state.queue_journal.clone();
        let mut body: serde_json::Value =
            serde_json::from_str(&generate_body("hdr from a client", 64, 64)).unwrap();
        body["hdr_exr_dir"] = serde_json::json!("/trusted/output");

        let refused = app_with_state(state)
            .oneshot(json_request("POST", "/api/generate", body))
            .await
            .unwrap();
        assert_eq!(refused.status(), StatusCode::UNPROCESSABLE_ENTITY);
        let refused = json_body(refused).await;
        assert_eq!(refused["code"], "VALIDATION_ERROR");
        assert!(
            refused["error"]
                .as_str()
                .unwrap_or_default()
                .contains("--local"),
            "{refused}"
        );
        assert!(journal.list_all().is_empty());
    }

    /// Conditioning media never reaches `mold.db`, and the mechanism is
    /// extraction rather than refusal.
    ///
    /// The media-free journal writers refused to persist media at all. They
    /// are gone; the live path extracts every media field into the encrypted
    /// queue-media store and serializes `request_json` from what is LEFT,
    /// which is what `capabilities.durable_media` advertises. This asserts the
    /// guarantee against the bytes on disk rather than against a refusal.
    ///
    /// Uses `source_image` because this binary is built without `pulid`, so an
    /// identity request is refused before admission here. The face-photograph
    /// half of the rule — biometric data about a real person, supplied for one
    /// render — is pinned by
    /// `queue_journal::tests::an_identity_request_never_writes_the_photograph_to_the_database`.
    #[tokio::test(flavor = "current_thread")]
    async fn conditioning_media_is_sealed_and_never_written_into_sqlite() {
        let root = tempfile::tempdir().unwrap();
        let db_path = root.path().join("mold.db");
        let db = Arc::new(Some(mold_db::MetadataDb::open(&db_path).unwrap()));
        let (mut state, _rx) = durable_state(db, root.path());
        install_authoritative_v2(&mut state);
        let journal = state.queue_journal.clone();

        // A base64 payload distinctive enough to find in the raw database.
        const PIXELS: &str = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg==";
        let mut body: serde_json::Value =
            serde_json::from_str(&generate_body("a repaint", 64, 64)).unwrap();
        body["source_image"] = serde_json::json!(PIXELS);

        let accepted = app_with_state(state)
            .oneshot(json_request(
                "POST",
                "/api/generation-batches",
                serde_json::json!({
                    "client_batch_id": uuid::Uuid::new_v4().to_string(),
                    "requests": [body],
                }),
            ))
            .await
            .unwrap();
        assert_eq!(accepted.status(), StatusCode::ACCEPTED);

        // The row exists — the print IS durable — and the photograph is not in
        // it.
        let rows = journal.list_all();
        assert_eq!(rows.len(), 1, "a media-carrying print is admitted durably");
        assert!(
            rows[0].media_set_id.is_some(),
            "its media belongs to the encrypted store"
        );
        let mut sqlite = std::fs::read(&db_path).unwrap();
        if let Ok(wal) = std::fs::read(format!("{}-wal", db_path.display())) {
            sqlite.extend(wal);
        }
        assert!(
            !sqlite
                .windows(PIXELS.len())
                .any(|window| window == PIXELS.as_bytes()),
            "conditioning media must never be written into mold.db"
        );
    }

    #[cfg(feature = "h3")]
    fn inline_ref2va_body(prompt: &str, image: &[u8]) -> serde_json::Value {
        serde_json::json!({
            "prompt": prompt,
            "model": mold_core::minimax_h3::REF2VA_COMFY,
            "width": mold_core::minimax_h3::DEFAULT_WIDTH,
            "height": mold_core::minimax_h3::DEFAULT_HEIGHT,
            "steps": mold_core::minimax_h3::DEFAULT_STEPS,
            "guidance": 0.0,
            "strength": 1.0,
            "batch_size": 1,
            "frames": mold_core::minimax_h3::REVIEWED_COMPACT_FRAMES,
            "fps": mold_core::minimax_h3::FIXED_FPS,
            "output_format": "mp4",
            "references": [{
                "kind": "image",
                "media": {
                    "authority": "inline",
                    "data": base64::engine::general_purpose::STANDARD.encode(image)
                },
                "provenance": {
                    "name": "anchor.png",
                    "sha256": format!("{:x}", Sha256::digest(image))
                },
                "mime_type": "image/png",
                "width": 1,
                "height": 1
            }]
        })
    }

    /// A router with API-key auth explicitly disabled: the auth layers are
    /// installed with no key set, which is what `ReferenceIdentity` reads as
    /// "inline references are admissible without a key".
    #[cfg(feature = "h3")]
    fn authless_app(state: AppState) -> axum::Router {
        app_with_state(state)
            .layer(axum::middleware::from_fn(crate::auth::require_api_key))
            .layer(axum::middleware::from_fn_with_state(
                None,
                crate::auth::inject_auth_state,
            ))
    }

    #[cfg(feature = "h3")]
    fn keyed_app(state: AppState) -> axum::Router {
        let keys = std::collections::HashSet::from(["test-key".to_string()]);
        app_with_state(state)
            .layer(axum::middleware::from_fn(crate::auth::require_api_key))
            .layer(axum::middleware::from_fn_with_state(
                Some(std::sync::Arc::new(crate::auth::ApiKeySet::new(keys))),
                crate::auth::inject_auth_state,
            ))
    }

    #[cfg(feature = "h3")]
    fn keyed_json_request(method: &str, uri: &str, body: serde_json::Value) -> Request<Body> {
        Request::builder()
            .method(method)
            .uri(uri)
            .header("content-type", "application/json")
            .header("x-api-key", "test-key")
            .body(Body::from(body.to_string()))
            .unwrap()
    }

    #[cfg(feature = "h3")]
    fn sqlite_bytes(db_path: &std::path::Path) -> Vec<u8> {
        let mut sqlite = std::fs::read(db_path).unwrap();
        if let Ok(wal) = std::fs::read(format!("{}-wal", db_path.display())) {
            sqlite.extend(wal);
        }
        sqlite
    }

    /// Ordered references are admitted durably: the descriptor is a request
    /// setting that stays in the row, the bytes are sealed into the encrypted
    /// media set, and nothing that could re-open the media — the inline
    /// payload, a handle, a staging path — is written into SQLite.
    #[cfg(feature = "h3")]
    #[tokio::test(flavor = "current_thread")]
    async fn inline_ref2va_references_admit_durably_and_seal_only_their_bytes() {
        let root = tempfile::tempdir().unwrap();
        let db_path = root.path().join("mold.db");
        let db = Arc::new(Some(mold_db::MetadataDb::open(&db_path).unwrap()));
        let (mut state, _rx) = durable_state_with_engine(
            db.clone(),
            root.path(),
            MockEngine::ready_for_model(mold_core::minimax_h3::REF2VA_COMFY),
        );
        install_authoritative_v2(&mut state);
        let journal = state.queue_journal.clone();
        let image = minimal_png();
        let inline_payload = base64::engine::general_purpose::STANDARD.encode(&image);

        let accepted = authless_app(state.clone())
            .oneshot(json_request(
                "POST",
                "/api/generation-batches",
                serde_json::json!({
                    "client_batch_id": uuid::Uuid::new_v4().to_string(),
                    "requests": [inline_ref2va_body("durable ordered references", &image)],
                }),
            ))
            .await
            .unwrap();
        if accepted.status() != StatusCode::ACCEPTED {
            let status = accepted.status();
            let error = json_body(accepted).await;
            panic!("inline Ref2VA was not admitted durably: {status}: {error}");
        }

        let rows = journal.list_all();
        assert_eq!(
            rows.len(),
            1,
            "a reference-carrying print is admitted durably"
        );
        let row = &rows[0];
        assert!(row.media_set_id.is_some(), "the reference bytes are sealed");
        assert!(
            row.admission_authority.is_some(),
            "the H3 grant is captured"
        );
        let stored: GenerateRequest = serde_json::from_str(&row.request_json).unwrap();
        let references = stored
            .references
            .as_deref()
            .expect("descriptors stay on the row");
        assert_eq!(references.len(), 1);
        assert!(matches!(
            references[0].media(),
            mold_core::GenerationReferenceAuthority::Descriptor
        ));
        assert_eq!(
            references[0].provenance().sha256.as_deref(),
            Some(format!("{:x}", Sha256::digest(&image)).as_str())
        );
        assert!(!row.request_json.contains("inline"));
        assert!(!row.request_json.contains("resolved-"));
        let owner = journal.owner_uuid().unwrap();
        assert_eq!(
            mold_db::generation_queue_media::list_obligations(
                db.as_ref().as_ref().unwrap(),
                owner,
                mold_db::generation_queue_media::QueueMediaObligationState::Active,
            )
            .unwrap()
            .len(),
            1
        );
        let sqlite = sqlite_bytes(&db_path);
        assert!(
            !sqlite
                .windows(inline_payload.len())
                .any(|window| window == inline_payload.as_bytes()),
            "reference bytes must never be written into mold.db"
        );
        // The admission staging released its quota and files at the seal; the
        // encrypted set is the only copy.
        assert_eq!(state.reference_uploads.resolved_bytes_for_test(), 0);
        assert_eq!(state.reference_uploads.staged_set_count_for_test(), 0);
    }

    /// A batch is atomic: a sibling refused by name must not have spent the
    /// upload session of the sibling before it. The refusal names
    /// `requests[2]`, the first child's handle is still redeemable, and the
    /// corrected retry admits it.
    #[cfg(feature = "h3")]
    #[tokio::test(flavor = "current_thread")]
    async fn a_refused_sibling_spends_no_upload_session() {
        let root = tempfile::tempdir().unwrap();
        let db_path = root.path().join("mold.db");
        let db = Arc::new(Some(mold_db::MetadataDb::open(&db_path).unwrap()));
        let (mut state, _rx) = durable_state_with_engine(
            db,
            root.path(),
            MockEngine::ready_for_model(mold_core::minimax_h3::REF2VA_COMFY),
        );
        install_authoritative_v2(&mut state);
        let app = keyed_app(state.clone());
        let image = minimal_png();
        let mut descriptor_request = inline_ref2va_body("uploaded ordered references", &image);
        descriptor_request["references"][0]["media"] =
            serde_json::json!({ "authority": "descriptor" });
        let session = json_body(
            app.clone()
                .oneshot(keyed_json_request(
                    "POST",
                    "/api/generate/reference-upload-sessions",
                    serde_json::json!({
                        "request": descriptor_request.clone(),
                        "upload_references": [1],
                    }),
                ))
                .await
                .unwrap(),
        )
        .await;
        let handle = session["uploads"][0]["handle"]
            .as_str()
            .unwrap()
            .to_string();
        let uploaded = json_body(
            app.clone()
                .oneshot(
                    Request::put("/api/generate/reference-upload")
                        .header("x-api-key", "test-key")
                        .header(crate::reference_uploads::UPLOAD_HANDLE_HEADER, &handle)
                        .header("content-type", "image/png")
                        .header("content-length", image.len().to_string())
                        .body(Body::from(image.clone()))
                        .unwrap(),
                )
                .await
                .unwrap(),
        )
        .await;
        let mut upload_request = descriptor_request.clone();
        upload_request["references"][0]["media"] =
            serde_json::json!({ "authority": "upload", "handle": handle });
        upload_request["references"][0]["provenance"]["sha256"] =
            uploaded["metadata"]["sha256"].clone();
        let mut invalid_sibling = inline_ref2va_body("a sibling nobody can render", &image);
        invalid_sibling["width"] = serde_json::json!(0);

        let refused = app
            .clone()
            .oneshot(keyed_json_request(
                "POST",
                "/api/generation-batches",
                serde_json::json!({
                    "client_batch_id": uuid::Uuid::new_v4().to_string(),
                    "requests": [upload_request.clone(), invalid_sibling],
                }),
            ))
            .await
            .unwrap();
        // A sibling with no canvas is a validation refusal of the batch.
        assert_eq!(refused.status(), StatusCode::UNPROCESSABLE_ENTITY);
        let error = json_body(refused).await;
        assert!(
            error["error"].as_str().unwrap().contains("requests[2]"),
            "{error}"
        );
        assert!(
            state.reference_uploads.staging_exists(),
            "the first sibling's session must survive a refusal of the second"
        );

        let admitted = app
            .clone()
            .oneshot(keyed_json_request(
                "POST",
                "/api/generation-batches",
                serde_json::json!({
                    "client_batch_id": uuid::Uuid::new_v4().to_string(),
                    "requests": [upload_request],
                }),
            ))
            .await
            .unwrap();
        assert_eq!(
            admitted.status(),
            StatusCode::ACCEPTED,
            "the unspent handle admits the corrected batch"
        );
    }

    /// Stream one PNG through a request-bound upload session, then admit the
    /// request through the batch route. The one-use handle is consumed inside
    /// admission and never journaled; the session, its staging, and its quota
    /// are all gone by the time the 202 is answered.
    /// The durable row carries the reference DESCRIPTOR; the feeder's deferred
    /// preparation must validate that form rather than the admission form —
    /// hal9000 held the first live Ref2VA print with
    /// `MINIMAX_H3_REFERENCE_DESCRIPTOR_ONLY` (2026-08-27).
    ///
    /// `h3` only: `authless_app` and `inline_ref2va_body` are `h3`-gated
    /// helpers, so the `h3-private-uat` test graph could not compile this.
    #[cfg(feature = "h3")]
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn deferred_preparation_validates_references_in_their_resolved_form() {
        let root = tempfile::tempdir().unwrap();
        let db_path = root.path().join("mold.db");
        let db = Arc::new(Some(mold_db::MetadataDb::open(&db_path).unwrap()));
        let (mut state, rx) = durable_state_with_engine(
            db.clone(),
            root.path(),
            MockEngine::ready_for_model(mold_core::minimax_h3::REF2VA_COMFY),
        );
        install_authoritative_v2(&mut state);
        let journal = state.queue_journal.clone();
        let image = minimal_png();

        let accepted = authless_app(state.clone())
            .oneshot(json_request(
                "POST",
                "/api/generation-batches",
                serde_json::json!({
                    "client_batch_id": uuid::Uuid::new_v4().to_string(),
                    "requests": [inline_ref2va_body("resolved-form validation", &image)],
                }),
            ))
            .await
            .unwrap();
        assert_eq!(accepted.status(), StatusCode::ACCEPTED);
        let status = json_body(accepted).await;
        let rows = journal.list_all();
        assert_eq!(rows.len(), 1, "the reference print is one durable row");
        let row = &rows[0];
        let mut request: mold_core::GenerateRequest =
            serde_json::from_str(&row.request_json).unwrap();
        assert!(
            request
                .references
                .as_ref()
                .is_some_and(|refs| refs.iter().all(|r| matches!(
                    r.media(),
                    mold_core::GenerationReferenceAuthority::Descriptor
                ))),
            "the row carries descriptors: {}",
            row.request_json
        );

        // The exact validation the feeder's preparation runs on that row.
        let outcome = crate::routes::validate_generate_request(
            &request,
            Some(mold_core::minimax_h3::FAMILY),
            mold_core::ReferenceForm::Resolved,
        );
        assert!(
            !outcome
                .as_ref()
                .err()
                .is_some_and(|e| e.contains("descriptor authority")),
            "deferred preparation must accept the descriptor form: {outcome:?}"
        );
        // And the admission form still refuses it, so a client can never
        // hand the server a payload-free reference.
        request.title = None;
        let admitted = crate::routes::validate_generate_request(
            &request,
            Some(mold_core::minimax_h3::FAMILY),
            mold_core::ReferenceForm::Admitted,
        )
        .unwrap_err();
        assert!(admitted.contains("descriptor authority"), "{admitted}");

        // Now the real thing: the feeder prepares that row. Whatever else a
        // mock H3 runtime refuses, it must never be the descriptor form —
        // that is the refusal hal9000 held the first live print with.
        let batch_id = status["id"].as_str().unwrap().to_string();
        spawn_durable_feeder(&state);
        tokio::spawn(crate::queue::run_queue_worker(rx, state.clone()));
        let deadline = std::time::Instant::now() + Duration::from_secs(30);
        let mut last;
        loop {
            let detail = json_body(
                authless_app(state.clone())
                    .oneshot(
                        Request::get(format!("/api/generation-batches/{batch_id}"))
                            .body(Body::empty())
                            .unwrap(),
                    )
                    .await
                    .unwrap(),
            )
            .await;
            let child = detail["children"][0].clone();
            let error = child["error"].as_str().unwrap_or_default().to_string();
            assert!(
                !error.contains("descriptor authority"),
                "deferred preparation re-ran the admission form: {detail}"
            );
            last = child;
            if matches!(
                last["state"].as_str(),
                Some("held" | "failed" | "cancelled" | "complete" | "running")
            ) || std::time::Instant::now() > deadline
            {
                break;
            }
            tokio::time::sleep(Duration::from_millis(200)).await;
        }
        assert_ne!(
            last["state"], "accepted",
            "the feeder never moved the row: {last}"
        );
    }

    #[cfg(feature = "h3")]
    #[tokio::test(flavor = "current_thread")]
    async fn upload_session_references_admit_durably_and_release_staging() {
        let root = tempfile::tempdir().unwrap();
        let db_path = root.path().join("mold.db");
        let db = Arc::new(Some(mold_db::MetadataDb::open(&db_path).unwrap()));
        let (mut state, _rx) = durable_state_with_engine(
            db,
            root.path(),
            MockEngine::ready_for_model(mold_core::minimax_h3::REF2VA_COMFY),
        );
        install_authoritative_v2(&mut state);
        let journal = state.queue_journal.clone();
        let app = keyed_app(state.clone());
        let image = minimal_png();
        let mut descriptor_request = inline_ref2va_body("uploaded ordered references", &image);
        descriptor_request["references"][0]["media"] =
            serde_json::json!({ "authority": "descriptor" });

        let session = app
            .clone()
            .oneshot(keyed_json_request(
                "POST",
                "/api/generate/reference-upload-sessions",
                serde_json::json!({
                    "request": descriptor_request.clone(),
                    "upload_references": [1],
                }),
            ))
            .await
            .unwrap();
        assert_eq!(session.status(), StatusCode::OK, "session");
        let session = json_body(session).await;
        let handle = session["uploads"][0]["handle"]
            .as_str()
            .unwrap()
            .to_string();
        let uploaded = app
            .clone()
            .oneshot(
                Request::put("/api/generate/reference-upload")
                    .header("x-api-key", "test-key")
                    .header(crate::reference_uploads::UPLOAD_HANDLE_HEADER, &handle)
                    .header("content-type", "image/png")
                    .header("content-length", image.len().to_string())
                    .body(Body::from(image.clone()))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(uploaded.status(), StatusCode::OK, "upload");
        let uploaded = json_body(uploaded).await;
        assert_eq!(uploaded["session_complete"], true);
        assert!(state.reference_uploads.staging_exists());

        let mut upload_request = descriptor_request.clone();
        upload_request["references"][0]["media"] =
            serde_json::json!({ "authority": "upload", "handle": handle });
        upload_request["references"][0]["provenance"]["sha256"] =
            uploaded["metadata"]["sha256"].clone();
        let client_batch_id = uuid::Uuid::new_v4().to_string();
        let batch = serde_json::json!({
            "client_batch_id": client_batch_id,
            "requests": [upload_request],
        });
        let accepted = app
            .clone()
            .oneshot(keyed_json_request(
                "POST",
                "/api/generation-batches",
                batch.clone(),
            ))
            .await
            .unwrap();
        if accepted.status() != StatusCode::ACCEPTED {
            let status = accepted.status();
            let error = json_body(accepted).await;
            panic!("uploaded Ref2VA was not admitted durably: {status}: {error}");
        }
        let admitted = json_body(accepted).await;
        let rows = journal.list_all();
        assert_eq!(rows.len(), 1);
        assert!(rows[0].media_set_id.is_some());
        assert!(!rows[0].request_json.contains(&handle));
        let stored: GenerateRequest = serde_json::from_str(&rows[0].request_json).unwrap();
        assert!(matches!(
            stored.references.as_deref().unwrap()[0].media(),
            mold_core::GenerationReferenceAuthority::Descriptor
        ));
        assert_eq!(state.reference_uploads.resolved_bytes_for_test(), 0);
        assert_eq!(state.reference_uploads.staged_set_count_for_test(), 0);
        assert!(
            !sqlite_bytes(&db_path)
                .windows(handle.len())
                .any(|window| window == handle.as_bytes()),
            "a one-use upload handle must never be written into mold.db"
        );

        // A retry of the same operation is answered from the journal before
        // anything is resolved: the handle it carries was already spent, and
        // the client gets its batch back rather than an unknown-upload error.
        let retried = app
            .clone()
            .oneshot(keyed_json_request("POST", "/api/generation-batches", batch))
            .await
            .unwrap();
        assert_eq!(retried.status(), StatusCode::OK);
        assert_eq!(json_body(retried).await["id"], admitted["id"]);
        assert_eq!(journal.list_all().len(), 1);

        // The same spent handle under a NEW operation is a refusal, and it
        // queues nothing.
        let mut reused = serde_json::from_value::<serde_json::Value>(
            serde_json::json!({ "client_batch_id": uuid::Uuid::new_v4().to_string() }),
        )
        .unwrap();
        reused["requests"] = serde_json::json!([descriptor_request]);
        reused["requests"][0]["references"][0]["media"] =
            serde_json::json!({ "authority": "upload", "handle": handle });
        let refused = app
            .oneshot(keyed_json_request(
                "POST",
                "/api/generation-batches",
                reused,
            ))
            .await
            .unwrap();
        assert!(refused.status().is_client_error(), "{}", refused.status());
        assert_eq!(journal.list_all().len(), 1);
    }

    /// A LoRA beside conditioning media is an ordinary durable request.
    ///
    /// It used to be refused by name: durable media protocol v1 could not
    /// carry the LoRA's server-local path, so `DURABLE_MEDIA_LORA_UNSUPPORTED`
    /// took out every img2img, inpaint, control and video-source render that
    /// used one. `lora.path` is a request field like `model` — it is persisted
    /// with the rest of the request and re-validated at dispatch.
    #[tokio::test(flavor = "current_thread")]
    async fn media_with_a_lora_is_admitted_durably() {
        let root = tempfile::tempdir().unwrap();
        let db = Arc::new(Some(mold_db::MetadataDb::open_in_memory().unwrap()));
        let (mut state, _rx) =
            durable_state_with_engine(db, root.path(), MockEngine::ready_for_model("flux-dev:q4"));
        install_authoritative_v2(&mut state);
        let adapter = root.path().join("adapter.safetensors");
        std::fs::write(&adapter, b"lora").unwrap();

        let mut request: GenerateRequest = serde_json::from_str(&generate_body_for_model(
            "img2img with a lora",
            "flux-dev:q4",
            64,
            64,
        ))
        .unwrap();
        request.source_image = Some(minimal_png());
        request.lora = Some(mold_core::LoraWeight {
            path: adapter.display().to_string(),
            scale: 0.8,
            expert: None,
        });

        assert!(
            crate::routes::direct_durable_admission(&state, &mut request)
                .await
                .is_ok(),
            "a LoRA beside conditioning media is an ordinary durable request"
        );
    }

    /// The persisted request keeps the LoRA, and dispatch re-validates the
    /// path it names. A LoRA that vanished between admission and replay is a
    /// HELD row that says so — never a silent drop, and never a render with
    /// the adapter missing.
    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn a_vanished_lora_holds_its_row_by_name() {
        // A LoRA needs a model whose family merges one, and admission infers
        // the family from the model name, so the fixture engine answers to a
        // real FLUX identity rather than `mock-model`.
        const MODEL: &str = "flux-dev:q4";
        let (state, rx, gallery_root) = durable_test_state(MockEngine::ready_for_model(MODEL));
        spawn_durable_runtime(&state, rx);
        let adapter = gallery_root.path().join("adapter.safetensors");
        std::fs::write(&adapter, b"lora").unwrap();
        let app = app_with_state(state.clone());

        let mut body: serde_json::Value = serde_json::from_str(&generate_body_for_model(
            "a print with a lora",
            MODEL,
            64,
            64,
        ))
        .unwrap();
        body["lora"] = serde_json::json!({
            "path": adapter.display().to_string(),
            "scale": 0.8,
        });
        // The adapter disappears after the request is composed but before the
        // feeder prepares it — the replay-after-restart shape, compressed.
        std::fs::remove_file(&adapter).unwrap();

        let response = tokio::time::timeout(
            Duration::from_secs(20),
            app.oneshot(json_request("POST", "/api/generate", body)),
        )
        .await
        .expect("a vanished adapter must settle the request")
        .unwrap();
        // The attached caller gets the hold as its own error — the same 404
        // shape a missing model takes, which is what the hold is filed as —
        // naming the adapter and the retryable job the row was parked as.
        assert_eq!(response.status(), StatusCode::NOT_FOUND);
        let error = json_body(response).await;
        assert_eq!(
            error["code"],
            mold_core::SSE_ERROR_CODE_MODEL_NOT_FOUND,
            "{error}"
        );
        let reason = error["error"].as_str().unwrap_or_default();
        assert!(
            reason.contains("adapter.safetensors"),
            "the hold must name the LoRA that vanished: {error}"
        );
        assert!(reason.contains("/retry"), "{error}");
    }

    #[tokio::test(flavor = "current_thread")]
    async fn heterogeneous_batch_persists_raw_filing_without_preack_db_resolution() {
        let root = tempfile::tempdir().unwrap();
        let db = Arc::new(Some(mold_db::MetadataDb::open_in_memory().unwrap()));
        let collection = db
            .as_ref()
            .as_ref()
            .unwrap()
            .create_collection("Durable Shelf", None)
            .unwrap();
        let (mut state, _rx) = durable_state(db, root.path());
        install_authoritative_v2(&mut state);
        let journal = state.queue_journal.clone();
        let app = app_with_state(state);
        let mut request: serde_json::Value =
            serde_json::from_str(&generate_body("file this once", 512, 512)).unwrap();
        request["output_format"] = serde_json::Value::Null;
        request["collection"] = serde_json::json!({ "id": collection.id });
        request["tags"] = serde_json::json!(["  Night Sky  ", "night sky", "Blue"]);

        let response = app
            .oneshot(json_request(
                "POST",
                "/api/generation-batches",
                serde_json::json!({
                    "client_batch_id": uuid::Uuid::new_v4().to_string(),
                    "requests": [request],
                }),
            ))
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::ACCEPTED);

        let rows = journal.list_all();
        assert_eq!(rows.len(), 1);
        let persisted: GenerateRequest = serde_json::from_str(&rows[0].request_json).unwrap();
        assert_eq!(persisted.output_format, None);
        assert_eq!(persisted.embed_metadata, Some(true));
        assert_eq!(
            persisted.tags,
            Some(vec![
                "  Night Sky  ".into(),
                "night sky".into(),
                "Blue".into()
            ])
        );
        assert_eq!(
            persisted.collection,
            Some(mold_core::CollectionRef {
                id: Some(collection.id),
                name: None,
            })
        );
    }

    #[tokio::test(flavor = "current_thread")]
    async fn heterogeneous_batch_replay_ignores_later_host_defaults_and_unavailability() {
        let root = tempfile::tempdir().unwrap();
        let db = Arc::new(Some(mold_db::MetadataDb::open_in_memory().unwrap()));
        let (mut state, _rx) = durable_state(db, root.path());
        install_authoritative_v2(&mut state);
        let app = app_with_state(state.clone());
        let body = serde_json::json!({
            "client_batch_id": uuid::Uuid::new_v4().to_string(),
            "requests": [serde_json::from_str::<serde_json::Value>(
                &generate_body("stable retry", 512, 512)
            ).unwrap()],
        });

        let admitted = app
            .clone()
            .oneshot(json_request(
                "POST",
                "/api/generation-batches",
                body.clone(),
            ))
            .await
            .unwrap();
        assert_eq!(admitted.status(), StatusCode::ACCEPTED);
        let admitted = json_body(admitted).await;

        {
            let mut config = state.config.write().await;
            config.embed_metadata = !config.embed_metadata;
        }
        state.set_generation_unavailable("test host is draining");
        let replay = app
            .oneshot(json_request("POST", "/api/generation-batches", body))
            .await
            .unwrap();
        assert_eq!(replay.status(), StatusCode::OK);
        let replay = json_body(replay).await;
        assert_eq!(replay["id"], admitted["id"]);
        assert_eq!(replay["children"], admitted["children"]);
    }

    #[tokio::test(flavor = "current_thread")]
    async fn heterogeneous_batch_seals_media_before_ack_without_sqlite_plaintext() {
        let root = tempfile::tempdir().unwrap();
        let db_path = root.path().join("mold.db");
        let db = Arc::new(Some(mold_db::MetadataDb::open(&db_path).unwrap()));
        let (mut state, _rx) = durable_state(db.clone(), root.path());
        install_authoritative_v2(&mut state);
        let journal = state.queue_journal.clone();
        let app = app_with_state(state);
        let mut request: serde_json::Value =
            serde_json::from_str(&generate_body("volatile source", 512, 512)).unwrap();
        let source = base64::engine::general_purpose::STANDARD.encode(minimal_png());
        request["source_image"] = serde_json::json!(source);
        request["source_image_name"] = serde_json::json!("sqlite-media-name-sentinel.png");

        let response = app
            .oneshot(json_request(
                "POST",
                "/api/generation-batches",
                serde_json::json!({
                    "client_batch_id": uuid::Uuid::new_v4().to_string(),
                    "requests": [request],
                }),
            ))
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::ACCEPTED);
        let response = json_body(response).await;
        let rows = journal.list_all();
        assert_eq!(rows.len(), 1);
        assert_eq!(rows[0].id, response["children"][0]["job_id"]);
        assert!(rows[0].media_set_id.is_some());
        assert!(!rows[0].request_json.contains("sqlite-media-name-sentinel"));
        assert!(!rows[0].request_json.contains(&source));

        let mut sqlite = std::fs::read(&db_path).unwrap();
        for suffix in ["-wal", "-shm"] {
            if let Ok(bytes) = std::fs::read(format!("{}{}", db_path.display(), suffix)) {
                sqlite.extend(bytes);
            }
        }
        assert!(!sqlite
            .windows(b"sqlite-media-name-sentinel".len())
            .any(|window| window == b"sqlite-media-name-sentinel"));
        assert!(!sqlite
            .windows(source.len())
            .any(|window| window == source.as_bytes()));
    }

    #[tokio::test(flavor = "current_thread")]
    async fn durable_media_receipt_survives_gc_and_rejects_changes_and_tampering() {
        let root = tempfile::tempdir().unwrap();
        let db = Arc::new(Some(mold_db::MetadataDb::open_in_memory().unwrap()));
        let (mut state, _rx) = durable_state(db.clone(), root.path());
        install_authoritative_v2(&mut state);
        let journal = state.queue_journal.clone();
        let app = app_with_state(state);
        let client_id = uuid::Uuid::new_v4().to_string();
        let source = base64::engine::general_purpose::STANDARD.encode(minimal_png());
        let mut request: serde_json::Value =
            serde_json::from_str(&generate_body("receipt survives media GC", 512, 512)).unwrap();
        request["source_image"] = serde_json::json!(source);
        request["source_image_name"] = serde_json::json!("first-name.png");
        let body = serde_json::json!({
            "client_batch_id": client_id,
            "requests": [request],
        });

        let admitted = app
            .clone()
            .oneshot(json_request(
                "POST",
                "/api/generation-batches",
                body.clone(),
            ))
            .await
            .unwrap();
        assert_eq!(admitted.status(), StatusCode::ACCEPTED);
        let admitted = json_body(admitted).await;
        let job_id = admitted["children"][0]["job_id"]
            .as_str()
            .unwrap()
            .to_string();
        assert!(journal.cancel_id(&job_id).unwrap());
        assert!(journal.list_all().is_empty());

        let retry = app
            .clone()
            .oneshot(json_request(
                "POST",
                "/api/generation-batches",
                body.clone(),
            ))
            .await
            .unwrap();
        assert_eq!(retry.status(), StatusCode::OK);
        let retry = json_body(retry).await;
        assert_eq!(retry["id"], admitted["id"]);
        assert_eq!(retry["children"][0]["job_id"], job_id);

        let mut changed = body.clone();
        changed["requests"][0]["source_image_name"] = serde_json::json!("changed-name.png");
        let conflict = app
            .clone()
            .oneshot(json_request("POST", "/api/generation-batches", changed))
            .await
            .unwrap();
        assert_eq!(conflict.status(), StatusCode::CONFLICT);
        assert_eq!(
            json_body(conflict).await["code"],
            "GENERATION_BATCH_IDEMPOTENCY_CONFLICT"
        );

        let receipt = db
            .as_ref()
            .as_ref()
            .unwrap()
            .with_conn(|conn| {
                conn.query_row(
                    "SELECT request_sha256 FROM generation_batches WHERE client_batch_id = ?1",
                    [&client_id],
                    |row| row.get::<_, String>(0),
                )
                .map_err(Into::into)
            })
            .unwrap();
        let mut tampered = receipt.into_bytes();
        let last = tampered.last_mut().unwrap();
        *last = if *last == b'a' { b'b' } else { b'a' };
        let tampered = String::from_utf8(tampered).unwrap();
        db.as_ref()
            .as_ref()
            .unwrap()
            .with_conn(|conn| {
                conn.execute(
                    "UPDATE generation_batches SET request_sha256 = ?1
                      WHERE client_batch_id = ?2",
                    (&tampered, client_id),
                )?;
                Ok(())
            })
            .unwrap();
        let undecidable = app
            .oneshot(json_request("POST", "/api/generation-batches", body))
            .await
            .unwrap();
        assert_eq!(undecidable.status(), StatusCode::SERVICE_UNAVAILABLE);
        assert_eq!(
            json_body(undecidable).await["code"],
            crate::queue_media_admission::DURABLE_MEDIA_IDENTITY_UNDECIDABLE
        );
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn durable_media_uuid_text_variants_converge_concurrently_and_conflict_once() {
        let root = tempfile::tempdir().unwrap();
        let db = Arc::new(Some(mold_db::MetadataDb::open_in_memory().unwrap()));
        let (mut state, _rx) = durable_state(db, root.path());
        install_authoritative_v2(&mut state);
        let journal = state.queue_journal.clone();
        let app = app_with_state(state);
        let canonical = uuid::Uuid::new_v4().to_string();
        let decorated = format!("  {}  ", canonical.to_ascii_uppercase());
        let mut request: serde_json::Value =
            serde_json::from_str(&generate_body("canonical durable media", 64, 64)).unwrap();
        request["source_image"] =
            serde_json::json!(base64::engine::general_purpose::STANDARD.encode(minimal_png()));
        request["source_image_name"] = serde_json::json!("canonical-source.png");
        let first_body = serde_json::json!({
            "client_batch_id": decorated,
            "requests": [request.clone()],
        });
        let second_body = serde_json::json!({
            "client_batch_id": canonical,
            "requests": [request.clone()],
        });

        let (first, second) = tokio::join!(
            app.clone().oneshot(json_request(
                "POST",
                "/api/generation-batches",
                first_body.clone(),
            )),
            app.clone()
                .oneshot(json_request("POST", "/api/generation-batches", second_body,)),
        );
        let first = first.unwrap();
        let second = second.unwrap();
        if !matches!(
            (first.status(), second.status()),
            (StatusCode::ACCEPTED, StatusCode::OK) | (StatusCode::OK, StatusCode::ACCEPTED)
        ) {
            let first_status = first.status();
            let second_status = second.status();
            let first_body = json_body(first).await;
            let second_body = json_body(second).await;
            panic!(
                "concurrent responses: {first_status} {first_body}; {second_status} {second_body}"
            );
        }
        let first = json_body(first).await;
        let second = json_body(second).await;
        assert_eq!(first["id"], second["id"]);
        assert_eq!(first["client_batch_id"], canonical);
        assert_eq!(second["client_batch_id"], canonical);
        assert_eq!(journal.list_all().len(), 1);
        let persisted: GenerateRequest =
            serde_json::from_str(&journal.list_all()[0].request_json).unwrap();
        // The operation converged on one canonical client id, but a plain
        // child is never stamped as a prepared sibling: `batch_id` is the
        // prepared-expansion contract and only a caller that prepared
        // variations supplies it.
        assert_eq!(persisted.batch_id, None);

        let retry = app
            .clone()
            .oneshot(json_request("POST", "/api/generation-batches", first_body))
            .await
            .unwrap();
        assert_eq!(retry.status(), StatusCode::OK);
        assert_eq!(json_body(retry).await["id"], first["id"]);

        let mut changed = request;
        changed["prompt"] = serde_json::json!("canonical conflict");
        let conflict = app
            .oneshot(json_request(
                "POST",
                "/api/generation-batches",
                serde_json::json!({
                    "client_batch_id": format!(" {} ", canonical.to_ascii_uppercase()),
                    "requests": [changed],
                }),
            ))
            .await
            .unwrap();
        assert_eq!(conflict.status(), StatusCode::CONFLICT);
        assert_eq!(
            json_body(conflict).await["code"],
            "GENERATION_BATCH_IDEMPOTENCY_CONFLICT"
        );
        assert_eq!(journal.list_all().len(), 1);
    }

    #[tokio::test(flavor = "current_thread")]
    async fn media_free_batch_uuid_text_variants_share_lookup_and_conflict_identity() {
        let root = tempfile::tempdir().unwrap();
        let db = Arc::new(Some(mold_db::MetadataDb::open_in_memory().unwrap()));
        let (mut state, _rx) = durable_state(db, root.path());
        install_authoritative_v2(&mut state);
        let journal = state.queue_journal.clone();
        let app = app_with_state(state);
        let canonical = uuid::Uuid::new_v4().to_string();
        let request: serde_json::Value =
            serde_json::from_str(&generate_body("canonical plain batch", 64, 64)).unwrap();

        let accepted = app
            .clone()
            .oneshot(json_request(
                "POST",
                "/api/generation-batches",
                serde_json::json!({
                    "client_batch_id": format!("  {}  ", canonical.to_ascii_uppercase()),
                    "requests": [request.clone()],
                }),
            ))
            .await
            .unwrap();
        assert_eq!(accepted.status(), StatusCode::ACCEPTED);
        let accepted = json_body(accepted).await;
        assert_eq!(accepted["client_batch_id"], canonical);
        let batch_id = accepted["id"].as_str().unwrap().to_string();

        let recovered = app
            .clone()
            .oneshot(
                Request::get(format!(
                    "/api/generation-batches/by-client/{}",
                    canonical.to_ascii_uppercase()
                ))
                .body(Body::empty())
                .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(recovered.status(), StatusCode::OK);
        assert_eq!(json_body(recovered).await["id"], batch_id);

        let reconciled = app
            .clone()
            .oneshot(json_request(
                "POST",
                "/api/generation-batches/status",
                serde_json::json!({
                    "client_batch_ids": [
                        canonical.to_ascii_uppercase(),
                        format!(" {canonical} "),
                    ],
                    "batch_ids": [],
                }),
            ))
            .await
            .unwrap();
        assert_eq!(reconciled.status(), StatusCode::OK);
        let reconciled = json_body(reconciled).await;
        assert_eq!(reconciled["batches"].as_array().unwrap().len(), 1);
        assert_eq!(reconciled["batches"][0]["id"], batch_id);
        assert!(reconciled["missing"]["client_batch_ids"]
            .as_array()
            .unwrap()
            .is_empty());

        let retry = app
            .clone()
            .oneshot(json_request(
                "POST",
                "/api/generation-batches",
                serde_json::json!({
                    "client_batch_id": canonical,
                    "requests": [request.clone()],
                }),
            ))
            .await
            .unwrap();
        assert_eq!(retry.status(), StatusCode::OK);
        assert_eq!(json_body(retry).await["id"], accepted["id"]);

        let mut changed = request;
        changed["prompt"] = serde_json::json!("plain conflict");
        let conflict = app
            .oneshot(json_request(
                "POST",
                "/api/generation-batches",
                serde_json::json!({
                    "client_batch_id": format!(" {} ", canonical.to_ascii_uppercase()),
                    "requests": [changed],
                }),
            ))
            .await
            .unwrap();
        assert_eq!(conflict.status(), StatusCode::CONFLICT);
        assert_eq!(
            json_body(conflict).await["code"],
            "GENERATION_BATCH_IDEMPOTENCY_CONFLICT"
        );
        assert_eq!(journal.list_all().len(), 1);
    }

    #[tokio::test(flavor = "current_thread")]
    async fn durable_media_mixed_siblings_commit_together_and_preflight_is_one_over_n_atomic() {
        let root = tempfile::tempdir().unwrap();
        let db = Arc::new(Some(mold_db::MetadataDb::open_in_memory().unwrap()));
        let (mut state, _rx) = durable_state(db.clone(), root.path());
        install_authoritative_v2(&mut state);
        let journal = state.queue_journal.clone();
        let app = app_with_state(state);
        let source = base64::engine::general_purpose::STANDARD.encode(minimal_png());
        let mut media: serde_json::Value =
            serde_json::from_str(&generate_body("mixed media", 64, 64)).unwrap();
        media["source_image"] = serde_json::json!(source);
        media["source_image_name"] = serde_json::json!("mixed-source.png");
        let plain: serde_json::Value =
            serde_json::from_str(&generate_body("mixed plain", 64, 64)).unwrap();

        let accepted = app
            .clone()
            .oneshot(json_request(
                "POST",
                "/api/generation-batches",
                serde_json::json!({
                    "client_batch_id": uuid::Uuid::new_v4().to_string(),
                    "requests": [media.clone(), plain],
                }),
            ))
            .await
            .unwrap();
        assert_eq!(accepted.status(), StatusCode::ACCEPTED);
        let rows = journal.list_all();
        assert_eq!(rows.len(), 2);
        assert!(rows[0].media_set_id.is_some());
        assert!(rows[1].media_set_id.is_none());
        for row in rows {
            assert!(journal.cancel_id(&row.id).unwrap());
        }

        // A LoRA beside the media is an ordinary img2img-with-adapter print:
        // it seals with the media and commits like any other sibling.
        let mut with_adapter = media.clone();
        // Ordinary validation still applies: an adapter needs a LoRA-capable
        // family, and the mock engine's model is not one.
        with_adapter["model"] = serde_json::json!("flux-dev");
        with_adapter["lora"] = serde_json::json!({
            "path": "adapter.safetensors",
            "scale": 0.8
        });
        let accepted = app
            .clone()
            .oneshot(json_request(
                "POST",
                "/api/generation-batches",
                serde_json::json!({
                    "client_batch_id": uuid::Uuid::new_v4().to_string(),
                    "requests": [
                        serde_json::from_str::<serde_json::Value>(
                            &generate_body("plain beside adapter", 64, 64)
                        ).unwrap(),
                        with_adapter
                    ],
                }),
            ))
            .await
            .unwrap();
        assert_eq!(accepted.status(), StatusCode::ACCEPTED);
        let rows = journal.list_all();
        assert_eq!(rows.len(), 2);
        assert!(rows[0].media_set_id.is_none());
        assert!(rows[1].media_set_id.is_some());
        for row in rows {
            assert!(journal.cancel_id(&row.id).unwrap());
        }

        // One-over-N atomicity is asked with a trait that refuses by
        // validation: ordered references on a model that is not MiniMax H3
        // Ref2VA. Nothing about them is a durability question any more.
        let mut refused = media;
        refused["references"] = serde_json::json!([]);
        let rejected = app
            .oneshot(json_request(
                "POST",
                "/api/generation-batches",
                serde_json::json!({
                    "client_batch_id": uuid::Uuid::new_v4().to_string(),
                    "requests": [
                        serde_json::from_str::<serde_json::Value>(
                            &generate_body("would otherwise admit", 64, 64)
                        ).unwrap(),
                        refused
                    ],
                }),
            ))
            .await
            .unwrap();
        assert_eq!(rejected.status(), StatusCode::UNPROCESSABLE_ENTITY);
        assert_eq!(json_body(rejected).await["code"], "VALIDATION_ERROR");
        assert!(journal.list_all().is_empty());
        let owner = journal.owner_uuid().unwrap();
        assert!(mold_db::generation_queue_media::list_obligations(
            db.as_ref().as_ref().unwrap(),
            owner,
            mold_db::generation_queue_media::QueueMediaObligationState::Active,
        )
        .unwrap()
        .is_empty());
    }

    fn durable_direct_media_body(prompt: &str) -> String {
        let mut request: serde_json::Value =
            serde_json::from_str(&generate_body(prompt, 64, 64)).unwrap();
        request["source_image"] =
            serde_json::json!(base64::engine::general_purpose::STANDARD.encode(minimal_png()));
        request["source_image_name"] = serde_json::json!("direct-source.png");
        request.to_string()
    }

    #[tokio::test(flavor = "current_thread")]
    async fn unreplayable_direct_authority_refusals_are_typed_and_never_reach_sqlite() {
        let root = tempfile::tempdir().unwrap();
        let db_path = root.path().join("mold.db");
        let db = Arc::new(Some(mold_db::MetadataDb::open(&db_path).unwrap()));
        let (mut state, _rx) = durable_state(db, root.path());
        install_authoritative_v2(&mut state);
        let journal = state.queue_journal.clone();
        let app = app_with_state(state);

        let mut references: serde_json::Value =
            serde_json::from_str(&generate_body("direct reference authority", 64, 64)).unwrap();
        references["references"] =
            serde_json::to_value(vec![mold_core::GenerationReference::Image {
                media: mold_core::GenerationReferenceAuthority::Inline {
                    data: minimal_png(),
                },
                provenance: mold_core::GenerationReferenceProvenance::default(),
                mime_type: "image/png".to_string(),
                width: 1,
                height: 1,
            }])
            .unwrap();

        let hdr_sentinel = "durable-hdr-must-never-enter-sqlite";
        let mut hdr: serde_json::Value =
            serde_json::from_str(&generate_body("direct HDR authority", 64, 64)).unwrap();
        hdr["hdr_exr_dir"] = serde_json::json!(hdr_sentinel);

        for path in ["/api/generate", "/api/generate/stream"] {
            // Both refuse by validation: references belong to MiniMax H3
            // Ref2VA alone, and `hdr_exr_dir` names an output directory on
            // the inference machine an HTTP client may not choose. The
            // guarantee this test exists for is that neither reaches SQLite.
            for (request, expected_code) in [
                (&references, "VALIDATION_ERROR"),
                (&hdr, "VALIDATION_ERROR"),
            ] {
                let response = app
                    .clone()
                    .oneshot(
                        Request::post(path)
                            .header("content-type", "application/json")
                            .body(Body::from(request.to_string()))
                            .unwrap(),
                    )
                    .await
                    .unwrap();
                assert_eq!(response.status(), StatusCode::UNPROCESSABLE_ENTITY);
                assert_eq!(json_body(response).await["code"], expected_code);
            }
        }

        assert!(journal.list_all().is_empty());
        let mut sqlite = std::fs::read(&db_path).unwrap();
        if let Ok(wal) = std::fs::read(format!("{}-wal", db_path.display())) {
            sqlite.extend(wal);
        }
        assert!(!sqlite
            .windows(hdr_sentinel.len())
            .any(|window| window == hdr_sentinel.as_bytes()));
    }

    #[tokio::test(flavor = "current_thread")]
    async fn direct_media_commits_before_it_waits_for_a_worker() {
        let root = tempfile::tempdir().unwrap();
        let db = Arc::new(Some(mold_db::MetadataDb::open_in_memory().unwrap()));
        let (mut state, mut rx) = durable_state(db, root.path());
        install_authoritative_v2(&mut state);
        let journal = state.queue_journal.clone();
        let app = app_with_state(state);
        let request = tokio::spawn(async move {
            app.oneshot(
                Request::post("/api/generate")
                    .header("content-type", "application/json")
                    .body(Body::from(durable_direct_media_body(
                        "media without a header",
                    )))
                    .unwrap(),
            )
            .await
        });

        tokio::time::timeout(Duration::from_secs(5), async {
            while journal.list_all().len() != 1 {
                tokio::task::yield_now().await;
            }
        })
        .await
        .expect("headerless media must commit before waiting for a worker");
        let rows = journal.list_all();
        assert_eq!(rows.len(), 1);
        assert!(uuid::Uuid::parse_str(&rows[0].id).is_ok());
        assert!(rx.try_recv().is_err(), "the feeder owns durable dispatch");
        request.abort();
        let _ = request.await;
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn durable_direct_raw_observer_preserves_the_legacy_media_response() {
        let root = tempfile::tempdir().unwrap();
        let db = Arc::new(Some(mold_db::MetadataDb::open_in_memory().unwrap()));
        let (mut state, rx) = durable_state(db, root.path());
        install_authoritative_v2(&mut state);
        let feeder_shutdown = tokio_util::sync::CancellationToken::new();
        let feeder = crate::durable_queue_feeder::spawn(state.clone(), feeder_shutdown.clone());
        let worker = tokio::spawn(crate::queue::run_queue_worker(rx, state.clone()));
        let response = tokio::time::timeout(
            Duration::from_secs(5),
            app_with_state(state.clone()).oneshot(
                Request::post("/api/generate")
                    .header("content-type", "application/json")
                    .body(Body::from(durable_direct_media_body("durable raw")))
                    .unwrap(),
            ),
        )
        .await
        .expect("durable raw observer must settle")
        .unwrap();

        assert_eq!(response.status(), StatusCode::OK);
        assert_eq!(response.headers().get("content-type").unwrap(), "image/png");
        assert_eq!(
            axum::body::to_bytes(response.into_body(), 1024 * 1024)
                .await
                .unwrap()
                .as_ref(),
            minimal_png()
        );
        feeder_shutdown.cancel();
        feeder.await.unwrap();
        worker.abort();
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn durable_direct_raw_failure_is_the_held_childs_error() {
        let root = tempfile::tempdir().unwrap();
        let db = Arc::new(Some(mold_db::MetadataDb::open_in_memory().unwrap()));
        let (mut state, rx) = durable_state_with_engine(db, root.path(), MockEngine::failing());
        install_authoritative_v2(&mut state);
        let feeder_shutdown = tokio_util::sync::CancellationToken::new();
        let feeder = crate::durable_queue_feeder::spawn(state.clone(), feeder_shutdown.clone());
        let worker = tokio::spawn(crate::queue::run_queue_worker(rx, state.clone()));

        let response = tokio::time::timeout(
            Duration::from_secs(5),
            app_with_state(state).oneshot(
                Request::post("/api/generate")
                    .header("content-type", "application/json")
                    .body(Body::from(durable_direct_media_body("durable raw failure")))
                    .unwrap(),
            ),
        )
        .await
        .expect("durable raw failure must settle as the caller's error")
        .unwrap();

        // The row is held and retryable; the attached caller is told so in
        // the singleton contract's own shape, naming the job to resume.
        assert_eq!(response.status(), StatusCode::INTERNAL_SERVER_ERROR);
        let body = json_body(response).await;
        assert_eq!(body["code"], "INFERENCE_ERROR", "{body}");
        let message = body["error"].as_str().unwrap_or_default();
        assert!(message.contains("mock engine error"), "{body}");
        assert!(message.contains("POST /api/queue/"), "{body}");
        assert!(message.contains("/retry"), "{body}");

        feeder_shutdown.cancel();
        feeder.await.unwrap();
        worker.abort();
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn durable_raw_failure_keeps_server_generated_identity_when_refresh_fails() {
        let root = tempfile::tempdir().unwrap();
        let db = Arc::new(Some(mold_db::MetadataDb::open_in_memory().unwrap()));
        let (mut state, rx) = durable_state_with_engine(db, root.path(), MockEngine::failing());
        install_authoritative_v2(&mut state);
        state.queue_journal.fail_batch_lookup_after_for_tests(1);
        let feeder_shutdown = tokio_util::sync::CancellationToken::new();
        let feeder = crate::durable_queue_feeder::spawn(state.clone(), feeder_shutdown.clone());
        let worker = tokio::spawn(crate::queue::run_queue_worker(rx, state.clone()));

        let response = tokio::time::timeout(
            Duration::from_secs(5),
            app_with_state(state).oneshot(
                Request::post("/api/generate")
                    .header("content-type", "application/json")
                    .body(Body::from(durable_direct_media_body(
                        "server generated reconciliation identity",
                    )))
                    .unwrap(),
            ),
        )
        .await
        .expect("accepted raw failure must answer from its original identity")
        .unwrap();

        // The refreshed row could not be read, so the error carries the
        // worker's sentence and the admission identity without claiming a
        // state this response never saw.
        assert_eq!(response.status(), StatusCode::INTERNAL_SERVER_ERROR);
        let body = json_body(response).await;
        let message = body["error"].as_str().unwrap_or_default();
        assert!(message.contains("mock engine error"), "{body}");
        assert!(message.contains("belongs to batch "), "{body}");
        assert!(!message.contains("/retry"), "{body}");

        feeder_shutdown.cancel();
        feeder.await.unwrap();
        worker.abort();
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn durable_direct_sse_observer_preserves_queued_and_complete_events() {
        let root = tempfile::tempdir().unwrap();
        let db = Arc::new(Some(mold_db::MetadataDb::open_in_memory().unwrap()));
        let (mut state, rx) = durable_state(db, root.path());
        install_authoritative_v2(&mut state);
        let feeder_shutdown = tokio_util::sync::CancellationToken::new();
        let feeder = crate::durable_queue_feeder::spawn(state.clone(), feeder_shutdown.clone());
        let worker = tokio::spawn(crate::queue::run_queue_worker(rx, state.clone()));
        let response = tokio::time::timeout(
            Duration::from_secs(5),
            app_with_state(state).oneshot(
                Request::post("/api/generate/stream")
                    .header("content-type", "application/json")
                    .header("x-mold-operation-id", uuid::Uuid::new_v4().to_string())
                    .body(Body::from(durable_direct_media_body("durable SSE")))
                    .unwrap(),
            ),
        )
        .await
        .expect("durable SSE observer must attach")
        .unwrap();
        assert_eq!(response.status(), StatusCode::OK);
        let body = tokio::time::timeout(
            Duration::from_secs(5),
            axum::body::to_bytes(response.into_body(), 1024 * 1024),
        )
        .await
        .expect("durable SSE stream must terminate")
        .unwrap();
        let body = String::from_utf8_lossy(&body);
        assert!(body.contains("\"type\":\"queued\""), "{body}");
        assert!(body.contains("event: complete"), "{body}");
        feeder_shutdown.cancel();
        feeder.await.unwrap();
        worker.abort();
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn durable_direct_sse_yields_committed_id_before_claim_and_disconnect_does_not_cancel() {
        use futures::StreamExt as _;

        let root = tempfile::tempdir().unwrap();
        let db = Arc::new(Some(mold_db::MetadataDb::open_in_memory().unwrap()));
        let (mut state, _rx) = durable_state(db, root.path());
        state.queue_capacity = 1;
        install_authoritative_v2(&mut state);
        let full_runtime = state
            .queue
            .try_reserve(state.queue_capacity)
            .expect("test owns the only runtime slot");
        let journal = state.queue_journal.clone();
        let admission = journal
            .queue_media_admission()
            .expect("admission installed");
        let feeder_shutdown = tokio_util::sync::CancellationToken::new();
        let feeder = crate::durable_queue_feeder::spawn(state.clone(), feeder_shutdown.clone());
        let app = app_with_state(state);

        let response = tokio::time::timeout(
            Duration::from_secs(5),
            app.clone().oneshot(
                Request::post("/api/generate/stream")
                    .header("content-type", "application/json")
                    .header("x-mold-operation-id", uuid::Uuid::new_v4().to_string())
                    .body(Body::from(durable_direct_media_body(
                        "committed before claim",
                    )))
                    .unwrap(),
            ),
        )
        .await
        .expect("HTTP admission must not wait for feeder capacity")
        .unwrap();
        assert_eq!(response.status(), StatusCode::OK);

        let rows = journal.list_all();
        assert_eq!(rows.len(), 1);
        let job_id = rows[0].id.clone();
        assert_eq!(
            rows[0].state,
            mold_db::generation_queue::QueueRowState::Queued
        );
        let mut stream = response.into_body().into_data_stream();
        let first = tokio::time::timeout(Duration::from_secs(1), stream.next())
            .await
            .expect("committed queued event must be available before feeder claim")
            .expect("SSE body remains open")
            .unwrap();
        let first = String::from_utf8_lossy(&first);
        assert!(first.contains("\"type\":\"queued\""), "{first}");
        assert!(first.contains(&job_id), "{first}");

        drop(stream);
        assert_eq!(admission.ingress().attached_len(), 0);
        assert_eq!(journal.list_all().len(), 1, "disconnect only detaches");

        let cancelled = app
            .oneshot(
                Request::delete(format!("/api/queue/{job_id}"))
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(cancelled.status(), StatusCode::NO_CONTENT);
        assert!(
            journal.list_all().is_empty(),
            "DELETE remains cancellation authority"
        );

        feeder_shutdown.cancel();
        feeder.await.unwrap();
        drop(full_runtime);
    }

    #[tokio::test(flavor = "current_thread")]
    async fn queued_cancellation_resolves_the_attached_durable_sse() {
        use futures::StreamExt as _;

        let root = tempfile::tempdir().unwrap();
        let db = Arc::new(Some(mold_db::MetadataDb::open_in_memory().unwrap()));
        let (mut state, _rx) = durable_state(db, root.path());
        state.queue_capacity = 1;
        install_authoritative_v2(&mut state);
        let full_runtime = state
            .queue
            .try_reserve(state.queue_capacity)
            .expect("test owns the only runtime slot");
        let journal = state.queue_journal.clone();
        let feeder_shutdown = tokio_util::sync::CancellationToken::new();
        let feeder = crate::durable_queue_feeder::spawn(state.clone(), feeder_shutdown.clone());
        let app = app_with_state(state);

        let response = app
            .clone()
            .oneshot(
                Request::post("/api/generate/stream")
                    .header("content-type", "application/json")
                    .header("x-mold-operation-id", uuid::Uuid::new_v4().to_string())
                    .body(Body::from(durable_direct_media_body("cancel before claim")))
                    .unwrap(),
            )
            .await
            .unwrap();
        let job_id = journal.list_all()[0].id.clone();
        let mut stream = response.into_body().into_data_stream();
        let queued = stream.next().await.unwrap().unwrap();
        assert!(String::from_utf8_lossy(&queued).contains(&job_id));

        let cancelled = app
            .oneshot(
                Request::delete(format!("/api/queue/{job_id}"))
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(cancelled.status(), StatusCode::NO_CONTENT);
        let terminal = tokio::time::timeout(Duration::from_secs(2), stream.next())
            .await
            .expect("cancellation must resolve the observer")
            .expect("SSE terminal frame")
            .unwrap();
        let terminal = String::from_utf8_lossy(&terminal);
        assert!(terminal.contains("queued_cancelled"), "{terminal}");
        assert!(journal.list_all().is_empty());

        feeder_shutdown.cancel();
        feeder.await.unwrap();
        drop(full_runtime);
    }

    #[tokio::test(flavor = "current_thread")]
    async fn full_observer_registry_refuses_before_durable_admission() {
        let root = tempfile::tempdir().unwrap();
        let db = Arc::new(Some(mold_db::MetadataDb::open_in_memory().unwrap()));
        let (mut state, _rx) = durable_state(db, root.path());
        install_authoritative_v2(&mut state);
        let admission = state
            .queue_journal
            .queue_media_admission()
            .expect("admission installed");
        let registrations = (0..state.queue_capacity)
            .map(|index| {
                admission
                    .ingress()
                    .reserve(
                        &format!("occupied-observer-{index}"),
                        crate::queue_media_ingress::ObserverMode::Raw,
                    )
                    .expect("registry has its authoritative runtime capacity")
            })
            .collect::<Vec<_>>();
        let journal = state.queue_journal.clone();
        let response = app_with_state(state)
            .oneshot(
                Request::post("/api/generate")
                    .header("content-type", "application/json")
                    .header("x-mold-operation-id", uuid::Uuid::new_v4().to_string())
                    .body(Body::from(durable_direct_media_body("detached overflow")))
                    .unwrap(),
            )
            .await
            .unwrap();

        assert_eq!(response.status(), StatusCode::SERVICE_UNAVAILABLE);
        assert_eq!(
            json_body(response).await["code"],
            "DIRECT_OBSERVER_CAPACITY_EXCEEDED"
        );
        assert!(journal.list_all().is_empty());
        drop(registrations);
    }

    #[tokio::test(flavor = "current_thread")]
    async fn heterogeneous_batch_admits_thirty_once_and_returns_same_ids_on_retry() {
        let root = tempfile::tempdir().unwrap();
        let db_path = root.path().join("mold.db");
        let first_ids = {
            let db = Arc::new(Some(mold_db::MetadataDb::open(&db_path).unwrap()));
            let (mut state, mut rx) = durable_state(db, root.path());
            state.queue_capacity = 64;
            let (scheduled_tx, _scheduled_rx) = tokio::sync::mpsc::channel(1);
            state.scheduled_work = crate::scheduler::ScheduledWorkHandle::new(scheduled_tx);
            let journal = state.queue_journal.clone();
            let app = app_with_state(state.clone());
            let client_batch_id = uuid::Uuid::new_v4().to_string();
            let requests: Vec<serde_json::Value> = (0..30)
                .map(|index| {
                    let mut request: serde_json::Value =
                        serde_json::from_str(&generate_body(&format!("moon {index}"), 512, 512))
                            .unwrap();
                    request["batch_size"] = serde_json::json!(1);
                    request
                })
                .collect();
            let body = serde_json::json!({
                "client_batch_id": client_batch_id,
                "requests": requests,
            });

            let admitted = app
                .clone()
                .oneshot(json_request(
                    "POST",
                    "/api/generation-batches",
                    body.clone(),
                ))
                .await
                .unwrap();
            assert_eq!(admitted.status(), StatusCode::ACCEPTED);
            let admitted = json_body(admitted).await;
            let batch_id = admitted["id"].as_str().unwrap().to_string();
            let first_ids: Vec<String> = admitted["children"]
                .as_array()
                .unwrap()
                .iter()
                .map(|child| child["job_id"].as_str().unwrap().to_string())
                .collect();
            assert_eq!(first_ids.len(), 30);
            assert_eq!(
                journal.list_all().len(),
                30,
                "all rows commit before response"
            );

            let retry = app
                .clone()
                .oneshot(json_request("POST", "/api/generation-batches", body))
                .await
                .unwrap();
            assert_eq!(retry.status(), StatusCode::OK);
            let retry = json_body(retry).await;
            let retry_ids: Vec<String> = retry["children"]
                .as_array()
                .unwrap()
                .iter()
                .map(|child| child["job_id"].as_str().unwrap().to_string())
                .collect();
            assert_eq!(retry_ids, first_ids);

            let cancelled_id = first_ids.last().unwrap().clone();
            let cancelled = app
                .clone()
                .oneshot(
                    Request::delete(format!("/api/queue/{cancelled_id}"))
                        .body(Body::empty())
                        .unwrap(),
                )
                .await
                .unwrap();
            assert_eq!(cancelled.status(), StatusCode::NO_CONTENT);
            let status = app
                .oneshot(
                    Request::get(format!("/api/generation-batches/{batch_id}"))
                        .body(Body::empty())
                        .unwrap(),
                )
                .await
                .unwrap();
            let status = json_body(status).await;
            assert_eq!(status["children"][29]["state"], "cancelled");
            assert_eq!(journal.list_all().len(), 29);

            assert!(
                rx.try_recv().is_err(),
                "admission must not construct or submit child jobs"
            );
            assert_eq!(state.job_registry.len(), 0, "the feeder owns hydration");
            let feeder_shutdown = tokio_util::sync::CancellationToken::new();
            let feeder = crate::durable_queue_feeder::spawn(state, feeder_shutdown.clone());
            let mut admitted_jobs = Vec::new();
            for _ in 0..29 {
                admitted_jobs.push(
                    tokio::time::timeout(Duration::from_secs(5), rx.recv())
                        .await
                        .expect("every admitted child reaches the ordinary queue")
                        .expect("queue remains open"),
                );
            }
            journal.retain_all();
            feeder_shutdown.cancel();
            feeder.await.unwrap();
            drop(admitted_jobs);
            drop(journal);
            first_ids[..29].to_vec()
        };

        let reopened = Arc::new(Some(mold_db::MetadataDb::open(&db_path).unwrap()));
        let (state, mut rx) = durable_state(reopened, root.path());
        crate::durable_queue_feeder::recover_runtime(&state)
            .await
            .expect("restart clears the prior runtime's retained claim tokens");
        state
            .queue_journal
            .resume_all_paused()
            .expect("test operator resumes restart-paused work");
        let feeder_shutdown = tokio_util::sync::CancellationToken::new();
        let feeder = crate::durable_queue_feeder::spawn(state.clone(), feeder_shutdown.clone());
        let mut replayed = Vec::new();
        for _ in 0..29 {
            replayed.push(
                tokio::time::timeout(Duration::from_secs(5), rx.recv())
                    .await
                    .expect("replayed batch child")
                    .expect("queue remains open")
                    .id,
            );
        }
        feeder_shutdown.cancel();
        feeder.await.unwrap();
        assert_eq!(replayed, first_ids);
        assert!(rx.try_recv().is_err(), "no child replays twice");
    }

    #[tokio::test(flavor = "current_thread")]
    async fn heterogeneous_batch_records_typed_prompt_history_after_admission() {
        let root = tempfile::tempdir().unwrap();
        let db = Arc::new(Some(mold_db::MetadataDb::open_in_memory().unwrap()));
        let (mut state, _rx) = durable_state(db.clone(), root.path());
        install_authoritative_v2(&mut state);
        let app = app_with_state(state);

        let mut first: serde_json::Value = serde_json::from_str(&generate_body_for_model(
            "  first prompt exactly as typed  ",
            "mock-model",
            512,
            512,
        ))
        .unwrap();
        first["negative_prompt"] = serde_json::json!("  literal negative prompt  ");
        let duplicate = first.clone();
        let second: serde_json::Value =
            serde_json::from_str(&generate_body("second prompt", 512, 512)).unwrap();
        let response = app
            .oneshot(json_request(
                "POST",
                "/api/generation-batches",
                serde_json::json!({
                    "client_batch_id": uuid::Uuid::new_v4().to_string(),
                    "requests": [first, duplicate, second],
                }),
            ))
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::ACCEPTED);

        let history = mold_db::PromptHistory::new(db.as_ref().as_ref().unwrap());
        let entries = history.recent(10).unwrap();
        assert_eq!(entries.len(), 2, "consecutive identical siblings dedupe");
        assert_eq!(entries[0].prompt, "second prompt");
        assert_eq!(entries[0].negative, None);
        assert_eq!(entries[0].model, "mock-model");
        assert_eq!(entries[1].prompt, "  first prompt exactly as typed  ");
        assert_eq!(
            entries[1].negative.as_deref(),
            Some("  literal negative prompt  ")
        );
        assert_eq!(entries[1].model, "mock-model");
    }

    #[tokio::test(flavor = "current_thread")]
    async fn heterogeneous_batch_idempotent_retry_does_not_record_history_again() {
        let root = tempfile::tempdir().unwrap();
        let db = Arc::new(Some(mold_db::MetadataDb::open_in_memory().unwrap()));
        let (mut state, _rx) = durable_state(db.clone(), root.path());
        install_authoritative_v2(&mut state);
        let app = app_with_state(state);
        let body = serde_json::json!({
            "client_batch_id": uuid::Uuid::new_v4().to_string(),
            "requests": [serde_json::from_str::<serde_json::Value>(
                &generate_body("durable batch prompt", 512, 512)
            ).unwrap()],
        });

        let admitted = app
            .clone()
            .oneshot(json_request(
                "POST",
                "/api/generation-batches",
                body.clone(),
            ))
            .await
            .unwrap();
        assert_eq!(admitted.status(), StatusCode::ACCEPTED);

        let history = mold_db::PromptHistory::new(db.as_ref().as_ref().unwrap());
        history
            .push(&mold_db::HistoryEntry::new(
                "intervening prompt",
                "mock-model",
            ))
            .unwrap();

        let retry = app
            .oneshot(json_request("POST", "/api/generation-batches", body))
            .await
            .unwrap();
        assert_eq!(retry.status(), StatusCode::OK);
        assert_eq!(
            history
                .recent(10)
                .unwrap()
                .into_iter()
                .map(|entry| entry.prompt)
                .collect::<Vec<_>>(),
            vec![
                "intervening prompt".to_string(),
                "durable batch prompt".to_string(),
            ]
        );
    }

    /// One id cancels the whole print run. Cancelling N children one at a
    /// time is the same work with N chances to be interrupted half way.
    #[tokio::test(flavor = "current_thread")]
    async fn cancelling_a_batch_cancels_every_child_that_had_not_settled() {
        let root = tempfile::tempdir().unwrap();
        let db = Arc::new(Some(mold_db::MetadataDb::open_in_memory().unwrap()));
        let (mut state, _rx) = durable_state(db.clone(), root.path());
        install_authoritative_v2(&mut state);
        let queue_owner = state.queue_journal.owner_uuid().unwrap().to_string();
        let app = app_with_state(state);

        let client_batch_id = uuid::Uuid::new_v4().to_string();
        let requests = ["settled", "still queued", "also queued"]
            .into_iter()
            .map(|prompt| {
                serde_json::from_str::<serde_json::Value>(&generate_body(prompt, 512, 512)).unwrap()
            })
            .collect::<Vec<_>>();
        let admitted = json_body(
            app.clone()
                .oneshot(json_request(
                    "POST",
                    "/api/generation-batches",
                    serde_json::json!({
                        "client_batch_id": client_batch_id,
                        "requests": requests,
                    }),
                ))
                .await
                .unwrap(),
        )
        .await;
        let batch_id = admitted["id"].as_str().unwrap().to_string();
        let settled_job = admitted["children"][0]["job_id"]
            .as_str()
            .unwrap()
            .to_string();

        // One child finishes before the cancellation lands; its terminal
        // outcome must survive.
        mold_db::generation_batches::finish_unclaimed_queued(
            db.as_ref().as_ref().unwrap(),
            &queue_owner,
            &settled_job,
            mold_db::generation_batches::GenerationBatchTerminal {
                state: mold_db::generation_batches::GenerationBatchTerminalState::Complete,
                error: None,
                terminal_error_json: None,
                result_json: Some(r#"{"filename":"done.png"}"#),
                completed_at_ms: 100,
            },
        )
        .unwrap();

        let cancelled = app
            .clone()
            .oneshot(
                Request::delete(format!("/api/generation-batches/{batch_id}"))
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(cancelled.status(), StatusCode::OK);
        let cancelled = json_body(cancelled).await;
        assert_eq!(cancelled["children"][0]["state"], "complete");
        assert_eq!(cancelled["children"][0]["result"]["filename"], "done.png");
        assert_eq!(cancelled["children"][1]["state"], "cancelled");
        assert_eq!(cancelled["children"][2]["state"], "cancelled");

        // The authoritative read agrees, and a second cancel is a no-op
        // rather than an error.
        let again = app
            .clone()
            .oneshot(
                Request::delete(format!("/api/generation-batches/{batch_id}"))
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(again.status(), StatusCode::OK);
        let read = json_body(
            app.clone()
                .oneshot(
                    Request::get(format!("/api/generation-batches/{batch_id}"))
                        .body(Body::empty())
                        .unwrap(),
                )
                .await
                .unwrap(),
        )
        .await;
        assert_eq!(read["children"][0]["state"], "complete");
        assert_eq!(read["children"][1]["state"], "cancelled");

        let missing = app
            .oneshot(
                Request::delete("/api/generation-batches/does-not-exist")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(missing.status(), StatusCode::NOT_FOUND);
    }

    #[tokio::test(flavor = "current_thread")]
    async fn durable_batch_outcomes_recover_by_client_and_bulk_with_instance_fencing() {
        let root = tempfile::tempdir().unwrap();
        let db = Arc::new(Some(mold_db::MetadataDb::open_in_memory().unwrap()));
        let (mut state, _rx) = durable_state(db.clone(), root.path());
        install_authoritative_v2(&mut state);
        state.instance_id = Arc::new("serving-instance-not-journal-owner".into());
        let queue_owner = state.queue_journal.owner_uuid().unwrap().to_string();
        let app = app_with_state(state);

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
        assert_eq!(capabilities["queue"]["heterogeneous_batch_max_outputs"], 64);

        let client_batch_id = uuid::Uuid::new_v4().to_string();
        let requests = ["complete", "failed"]
            .into_iter()
            .map(|prompt| {
                serde_json::from_str::<serde_json::Value>(&generate_body(prompt, 512, 512)).unwrap()
            })
            .collect::<Vec<_>>();
        let admitted = app
            .clone()
            .oneshot(json_request(
                "POST",
                "/api/generation-batches",
                serde_json::json!({
                    "client_batch_id": client_batch_id,
                    "requests": requests,
                }),
            ))
            .await
            .unwrap();
        assert_eq!(admitted.status(), StatusCode::ACCEPTED);
        let admitted = json_body(admitted).await;
        assert_eq!(
            admitted["instance_id"],
            "serving-instance-not-journal-owner"
        );
        assert_eq!(admitted["durable"], true);
        assert!(admitted["children"][0]["created_at_ms"].as_i64().unwrap() > 0);
        let batch_id = admitted["id"].as_str().unwrap().to_string();
        let complete_job = admitted["children"][0]["job_id"]
            .as_str()
            .unwrap()
            .to_string();
        let failed_job = admitted["children"][1]["job_id"]
            .as_str()
            .unwrap()
            .to_string();

        let db = db.as_ref().as_ref().unwrap();
        mold_db::generation_batches::finish_unclaimed_queued(
            db,
            &queue_owner,
            &complete_job,
            mold_db::generation_batches::GenerationBatchTerminal {
                state: mold_db::generation_batches::GenerationBatchTerminalState::Complete,
                error: None,
                terminal_error_json: None,
                result_json: Some(
                    r#"{"filename":"finished.png","original_filename":"original.png","seed":77,"generation_time_ms":4321,"gpu":1}"#,
                ),
                completed_at_ms: 200,
            },
        )
        .unwrap();
        mold_db::generation_batches::finish_unclaimed_queued(
            db,
            &queue_owner,
            &failed_job,
            mold_db::generation_batches::GenerationBatchTerminal {
                state: mold_db::generation_batches::GenerationBatchTerminalState::Failed,
                error: Some("render failed"),
                terminal_error_json: Some(r#"{"code":"RENDER_FAILED","message":"render failed"}"#),
                result_json: None,
                completed_at_ms: 201,
            },
        )
        .unwrap();

        let lookup_path = format!("/api/generation-batches/by-client/{client_batch_id}");
        let first = app
            .clone()
            .oneshot(Request::get(&lookup_path).body(Body::empty()).unwrap())
            .await
            .unwrap();
        assert_eq!(first.status(), StatusCode::OK);
        let first = json_body(first).await;
        let retry = json_body(
            app.clone()
                .oneshot(Request::get(&lookup_path).body(Body::empty()).unwrap())
                .await
                .unwrap(),
        )
        .await;
        assert_eq!(retry, first, "lookup is an idempotent authoritative read");
        assert_eq!(first["id"], batch_id);
        assert_eq!(first["instance_id"], "serving-instance-not-journal-owner");
        assert_eq!(first["children"][0]["state"], "complete");
        assert_eq!(first["children"][0]["completed_at_ms"], 200);
        assert_eq!(first["children"][0]["result"]["filename"], "finished.png");
        assert_eq!(
            first["children"][0]["result"]["original_filename"],
            "original.png"
        );
        // The terminal facts a caller cannot recover from the gallery at the
        // moment it needs them: the seed to advance from, the elapsed time,
        // and the accelerator that ran it.
        assert_eq!(first["children"][0]["result"]["seed"], 77);
        assert_eq!(first["children"][0]["result"]["generation_time_ms"], 4321);
        assert_eq!(first["children"][0]["result"]["gpu"], 1);
        assert_eq!(first["children"][1]["state"], "failed");
        assert_eq!(first["children"][1]["error"], "render failed");
        assert_eq!(
            first["children"][1]["terminal_error"]["code"],
            "RENDER_FAILED"
        );
        assert_eq!(first["children"][1]["updated_at_ms"], 201);
        let by_server_id = json_body(
            app.clone()
                .oneshot(
                    Request::get(format!("/api/generation-batches/{batch_id}"))
                        .body(Body::empty())
                        .unwrap(),
                )
                .await
                .unwrap(),
        )
        .await;
        assert_eq!(by_server_id, first);

        let unknown_client = uuid::Uuid::new_v4().to_string();
        let unknown_batch = uuid::Uuid::new_v4().to_string();
        let bulk = app
            .clone()
            .oneshot(json_request(
                "POST",
                "/api/generation-batches/status",
                serde_json::json!({
                    "client_batch_ids": [client_batch_id, unknown_client, unknown_client],
                    "batch_ids": [batch_id, unknown_batch, unknown_batch],
                }),
            ))
            .await
            .unwrap();
        assert_eq!(bulk.status(), StatusCode::OK);
        let bulk = json_body(bulk).await;
        assert_eq!(bulk["instance_id"], "serving-instance-not-journal-owner");
        assert_eq!(bulk["batches"].as_array().unwrap().len(), 1);
        assert_eq!(bulk["batches"][0], first);
        assert_eq!(bulk["missing"]["client_batch_ids"][0], unknown_client);
        assert_eq!(bulk["missing"]["batch_ids"][0], unknown_batch);

        let missing = app
            .oneshot(
                Request::get(format!(
                    "/api/generation-batches/by-client/{}",
                    uuid::Uuid::new_v4()
                ))
                .body(Body::empty())
                .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(missing.status(), StatusCode::NOT_FOUND);
    }

    #[tokio::test(flavor = "current_thread")]
    async fn generation_batch_status_bounds_and_validates_unique_uuid_identities() {
        let root = tempfile::tempdir().unwrap();
        let db = Arc::new(Some(mold_db::MetadataDb::open_in_memory().unwrap()));
        let (mut state, _rx) = durable_state(db, root.path());
        install_authoritative_v2(&mut state);
        let app = app_with_state(state);

        let too_many = (0..=256)
            .map(|_| uuid::Uuid::new_v4().to_string())
            .collect::<Vec<_>>();
        let over_limit = app
            .clone()
            .oneshot(json_request(
                "POST",
                "/api/generation-batches/status",
                serde_json::json!({ "client_batch_ids": too_many }),
            ))
            .await
            .unwrap();
        assert_eq!(over_limit.status(), StatusCode::UNPROCESSABLE_ENTITY);
        assert_eq!(
            json_body(over_limit).await["code"],
            "GENERATION_BATCH_STATUS_LIMIT_EXCEEDED"
        );

        let malformed = app
            .oneshot(json_request(
                "POST",
                "/api/generation-batches/status",
                serde_json::json!({
                    "client_batch_ids": [],
                    "batch_ids": ["not-a-server-batch-uuid"]
                }),
            ))
            .await
            .unwrap();
        assert_eq!(malformed.status(), StatusCode::UNPROCESSABLE_ENTITY);
    }

    #[tokio::test(flavor = "current_thread")]
    async fn bulk_cancel_counts_deep_batch_legacy_and_claimed_rows_once() {
        let root = tempfile::tempdir().unwrap();
        let db = Arc::new(Some(mold_db::MetadataDb::open_in_memory().unwrap()));
        let blocking_db = db.clone();
        let (mut state, _rx) = durable_state(db, root.path());
        let (scheduled_tx, _scheduled_rx) = tokio::sync::mpsc::channel(1);
        state.scheduled_work = crate::scheduler::ScheduledWorkHandle::new(scheduled_tx);
        let journal = state.queue_journal.clone();
        let app = app_with_state(state.clone());
        let client_batch_id = uuid::Uuid::new_v4().to_string();
        let requests = (0..12)
            .map(|index| {
                let mut request: serde_json::Value =
                    serde_json::from_str(&generate_body(&format!("cancel {index}"), 512, 512))
                        .unwrap();
                request["batch_size"] = serde_json::json!(1);
                request
            })
            .collect::<Vec<_>>();
        let admitted = app
            .clone()
            .oneshot(json_request(
                "POST",
                "/api/generation-batches",
                serde_json::json!({
                    "client_batch_id": client_batch_id,
                    "requests": requests,
                }),
            ))
            .await
            .unwrap();
        assert_eq!(admitted.status(), StatusCode::ACCEPTED);
        let admitted = json_body(admitted).await;
        let durable_id = admitted["id"].as_str().unwrap().to_string();
        let claimed = journal.claim_next_feeder().unwrap().unwrap();
        let claimed_ticket = journal.attach_claimed(&claimed.row.id, claimed.claim_token);
        state
            .job_registry
            .register(&claimed.row.id, &claimed.row.model);
        let owner = journal.owner_uuid().unwrap().to_string();
        for index in 0..5 {
            seed_durable_projection_row(
                &blocking_db,
                &owner,
                &format!("legacy-deep-{index}"),
                mold_db::generation_queue::QueueRowState::Queued,
                100 + index,
                0,
            );
        }

        let (locked_tx, locked_rx) = std::sync::mpsc::channel();
        let (release_tx, release_rx) = std::sync::mpsc::channel();
        let watchdog_released = Arc::new(AtomicBool::new(false));
        let watchdog_released_in_thread = watchdog_released.clone();
        let db_holder = std::thread::spawn(move || {
            blocking_db
                .as_ref()
                .as_ref()
                .unwrap()
                .with_conn(|_| {
                    locked_tx.send(()).unwrap();
                    if release_rx.recv_timeout(Duration::from_secs(2)).is_err() {
                        watchdog_released_in_thread.store(true, Ordering::SeqCst);
                    }
                    Ok(())
                })
                .unwrap();
        });
        locked_rx.recv().unwrap();

        let cancel_app = app.clone();
        let cancel_task = tokio::spawn(async move {
            cancel_app
                .oneshot(Request::delete("/api/queue").body(Body::empty()).unwrap())
                .await
                .unwrap()
        });
        tokio::task::yield_now().await;
        let live_health = app
            .clone()
            .oneshot(Request::get("/health").body(Body::empty()).unwrap())
            .await
            .unwrap();
        assert_eq!(live_health.status(), StatusCode::OK);
        assert!(
            !watchdog_released.load(Ordering::SeqCst),
            "bulk cancellation blocked the current-thread async executor"
        );
        release_tx.send(()).unwrap();
        db_holder.join().unwrap();

        let cancelled = cancel_task.await.unwrap();
        assert_eq!(cancelled.status(), StatusCode::OK);
        assert_eq!(
            json_body(cancelled).await["cancelled"],
            17,
            "twelve durable batch children plus five legacy rows, with the claimed live child counted once"
        );
        claimed_ticket.discard();
        let status = app
            .oneshot(
                Request::get(format!("/api/generation-batches/{durable_id}"))
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        let status = json_body(status).await;
        assert_eq!(status["children"].as_array().unwrap().len(), 12);
        assert!(status["children"]
            .as_array()
            .unwrap()
            .iter()
            .all(|child| child["state"] == "cancelled"));
        assert!(journal.list_all().is_empty());
    }

    /// Boot the durable feeder after simulating the operator's explicit
    /// resume of restart-paused work. The caller stops it with `stop_feeder`.
    async fn boot_feeder(
        state: &AppState,
    ) -> (
        tokio_util::sync::CancellationToken,
        tokio::task::JoinHandle<()>,
    ) {
        crate::durable_queue_feeder::recover_runtime(state)
            .await
            .expect("durable runtime recovery");
        state
            .queue_journal
            .resume_all_paused()
            .expect("test operator resumes restart-paused work");
        let shutdown = tokio_util::sync::CancellationToken::new();
        let feeder = crate::durable_queue_feeder::spawn(state.clone(), shutdown.clone());
        (shutdown, feeder)
    }

    async fn stop_feeder(
        shutdown: tokio_util::sync::CancellationToken,
        feeder: tokio::task::JoinHandle<()>,
    ) {
        shutdown.cancel();
        feeder.await.unwrap();
    }

    /// Receive the next hydrated job and settle it so the single runtime slot
    /// frees for the next one. With `queue_capacity = 1` the feeder hydrates
    /// strictly in durable order, which is what these tests assert.
    async fn receive_and_settle(
        state: &AppState,
        rx: &mut tokio::sync::mpsc::Receiver<crate::state::GenerationJob>,
    ) -> crate::state::GenerationJob {
        let mut job = tokio::time::timeout(Duration::from_secs(5), rx.recv())
            .await
            .expect("hydrated job")
            .expect("open queue");
        assert!(
            state
                .job_registry
                .snapshot()
                .entries
                .iter()
                .any(|entry| entry.id == job.id),
            "a hydrated job is registered under its original id, so /api/queue and \
             /api/events see resumed jobs with no new event type"
        );
        job.journal.take().unwrap().complete_before_dispatch();
        state.job_registry.remove(&job.id);
        state.queue.decrement();
        job
    }

    /// The end-to-end shape: admit jobs, fence, drop the coordinator, rebuild
    /// `AppState` on the same DB, boot the feeder. The rows come back under
    /// their original ids, in submit order, through the ordinary queue.
    #[tokio::test]
    async fn retained_generations_replay_in_order_under_their_original_ids() {
        let output_dir = tempfile::tempdir().unwrap();
        let db = Arc::new(Some(mold_db::MetadataDb::open_in_memory().unwrap()));
        let submitted = seed_retained_jobs(db.clone(), output_dir.path(), 3).await;

        // A fresh server on the same database.
        let (mut state, mut rx) = durable_state(db.clone(), output_dir.path());
        state.queue_capacity = 1;
        let (shutdown, feeder) = boot_feeder(&state).await;

        let mut replayed = Vec::new();
        for _ in 0..3 {
            let job = receive_and_settle(&state, &mut rx).await;
            assert!(
                job.progress_tx.is_some(),
                "a replayed job has no client to stream to, but it still carries the \
                 registry relay that feeds the snapshot every surface polls"
            );
            replayed.push(job.id);
        }
        assert_eq!(replayed, submitted);
        assert!(
            state.queue_journal.list_all().is_empty(),
            "every retained row was hydrated exactly once"
        );
        stop_feeder(shutdown, feeder).await;
    }

    #[tokio::test]
    async fn direct_admission_rebinds_a_stable_gpu_after_restart_and_renumbering() {
        let _env = env_lock();
        const STABLE_ID: &str = "cuda:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa";
        let output_dir = tempfile::tempdir().unwrap();
        let db = Arc::new(Some(mold_db::MetadataDb::open_in_memory().unwrap()));

        let submitted_id = {
            let (mut state, _rx) = durable_state(db.clone(), output_dir.path());
            install_authoritative_v2(&mut state);
            state.gpu_pool = Arc::new(crate::gpu_pool::GpuPool {
                workers: vec![gpu_worker_stub_with_stable_id(2, STABLE_ID)].into(),
            });
            install_worker_registry(&mut state);
            let journal = state.queue_journal.clone();
            let app = app_with_state(state.clone());
            let mut body: serde_json::Value =
                serde_json::from_str(&generate_body("stable direct", 64, 64)).unwrap();
            body["placement"] = serde_json::json!({
                "text_encoders": { "kind": "cpu" },
                "advanced": {
                    "transformer": { "kind": "gpu", "ordinal": 2 },
                    "vae": { "kind": "auto" },
                    "clip_l": { "kind": "cpu" }
                }
            });
            let response = tokio::time::timeout(
                Duration::from_secs(5),
                app.oneshot(json_request("POST", "/api/generate/stream", body)),
            )
            .await
            .expect("direct admission")
            .expect("direct route response");
            assert_eq!(response.status(), StatusCode::OK);
            let row = journal
                .list_all()
                .into_iter()
                .next()
                .expect("durable direct row");
            assert_eq!(row.target_gpu, Some(2));
            assert_eq!(row.target_device_id.as_deref(), Some(STABLE_ID));
            let id = row.id;
            journal.retain_all();
            id
        };

        let (mut state, mut rx) = durable_state(db, output_dir.path());
        state.gpu_pool = Arc::new(crate::gpu_pool::GpuPool {
            workers: vec![gpu_worker_stub_with_stable_id(7, STABLE_ID)].into(),
        });
        install_worker_registry(&mut state);
        let (shutdown, feeder) = boot_feeder(&state).await;
        let mut replayed = tokio::time::timeout(Duration::from_secs(5), rx.recv())
            .await
            .expect("direct replay")
            .expect("queue open");
        assert_eq!(replayed.id, submitted_id);
        assert_eq!(state.job_registry.target_gpu(&submitted_id), Some(Some(7)));
        assert_eq!(
            crate::scheduler::generation_hard_ordinal(&state, &submitted_id, &replayed.request),
            Some(7),
            "scheduler V2 resolves the current ordinal"
        );
        assert_eq!(
            crate::queue::legacy_generation_preferred_gpu(
                &state,
                &submitted_id,
                replayed.request.placement.as_ref(),
            ),
            Ok(Some(7)),
            "legacy dispatch resolves the current ordinal"
        );
        let placement = replayed.request.placement.as_ref().unwrap();
        assert_eq!(placement.text_encoders, mold_core::DeviceRef::Cpu);
        let advanced = placement.advanced.as_ref().unwrap();
        assert_eq!(
            advanced.transformer,
            mold_core::DeviceRef::device(STABLE_ID)
        );
        assert_eq!(advanced.vae, mold_core::DeviceRef::Auto);
        assert_eq!(advanced.clip_l, Some(mold_core::DeviceRef::Cpu));
        replayed.journal.take().unwrap().discard();
        state.job_registry.remove(&submitted_id);
        state.queue.decrement();
        stop_feeder(shutdown, feeder).await;
    }

    #[tokio::test]
    async fn batch_admission_rebinds_a_stable_gpu_after_restart_and_renumbering() {
        const STABLE_ID: &str = "cuda:bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb";
        let output_dir = tempfile::tempdir().unwrap();
        let db_path = output_dir.path().join("mold.db");
        let submitted_id = {
            let db = Arc::new(Some(mold_db::MetadataDb::open(&db_path).unwrap()));
            let (mut state, _rx) = durable_state(db, output_dir.path());
            state.gpu_pool = Arc::new(crate::gpu_pool::GpuPool {
                workers: vec![gpu_worker_stub_with_stable_id(2, STABLE_ID)].into(),
            });
            install_worker_registry(&mut state);
            let (scheduled_tx, _scheduled_rx) = tokio::sync::mpsc::channel(1);
            state.scheduled_work = crate::scheduler::ScheduledWorkHandle::new(scheduled_tx);
            let journal = state.queue_journal.clone();
            let app = app_with_state(state);
            let mut request: serde_json::Value =
                serde_json::from_str(&generate_body("stable batch", 64, 64)).unwrap();
            request["placement"] = serde_json::json!({
                "text_encoders": { "kind": "cpu" },
                "advanced": {
                    "transformer": { "kind": "gpu", "ordinal": 2 },
                    "vae": { "kind": "auto" }
                }
            });
            let response = app
                .oneshot(json_request(
                    "POST",
                    "/api/generation-batches",
                    serde_json::json!({
                        "client_batch_id": uuid::Uuid::new_v4().to_string(),
                        "requests": [request],
                    }),
                ))
                .await
                .unwrap();
            assert_eq!(response.status(), StatusCode::ACCEPTED);
            let body = json_body(response).await;
            let id = body["children"][0]["job_id"].as_str().unwrap().to_string();
            let row = journal
                .list_all()
                .into_iter()
                .find(|row| row.id == id)
                .expect("durable batch row");
            assert_eq!(row.target_gpu, Some(2));
            assert_eq!(row.target_device_id.as_deref(), Some(STABLE_ID));
            id
        };

        let db = Arc::new(Some(mold_db::MetadataDb::open(&db_path).unwrap()));
        let (mut state, mut rx) = durable_state(db, output_dir.path());
        state.gpu_pool = Arc::new(crate::gpu_pool::GpuPool {
            workers: vec![gpu_worker_stub_with_stable_id(7, STABLE_ID)].into(),
        });
        install_worker_registry(&mut state);
        crate::durable_queue_feeder::recover_runtime(&state)
            .await
            .unwrap();
        state
            .queue_journal
            .resume_all_paused()
            .expect("test operator resumes restart-paused work");
        let shutdown = tokio_util::sync::CancellationToken::new();
        let feeder = crate::durable_queue_feeder::spawn(state.clone(), shutdown.clone());
        let mut replayed = tokio::time::timeout(Duration::from_secs(5), rx.recv())
            .await
            .expect("batch replay")
            .expect("queue open");
        assert_eq!(replayed.id, submitted_id);
        assert_eq!(state.job_registry.target_gpu(&submitted_id), Some(Some(7)));
        assert_eq!(
            crate::scheduler::generation_hard_ordinal(&state, &submitted_id, &replayed.request),
            Some(7)
        );
        assert_eq!(
            crate::queue::legacy_generation_preferred_gpu(
                &state,
                &submitted_id,
                replayed.request.placement.as_ref(),
            ),
            Ok(Some(7))
        );
        replayed.journal.take().unwrap().discard();
        state.job_registry.remove(&submitted_id);
        state.queue.decrement();
        shutdown.cancel();
        feeder.await.unwrap();
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
            let (mut state, mut rx) = durable_state(db.clone(), output_dir.path());
            install_authoritative_v2(&mut state);
            let feeder_shutdown = tokio_util::sync::CancellationToken::new();
            let feeder = crate::durable_queue_feeder::spawn(state.clone(), feeder_shutdown.clone());
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
            feeder_shutdown.cancel();
            feeder.await.unwrap();
            submitted
        };

        let (mut state, mut rx) = durable_state(db.clone(), output_dir.path());
        state.queue_capacity = 1;
        let (shutdown, feeder) = boot_feeder(&state).await;

        let mut replayed = Vec::new();
        for _ in 0..3 {
            replayed.push(receive_and_settle(&state, &mut rx).await.id);
        }
        assert_eq!(
            replayed,
            vec![
                submitted[2].clone(),
                submitted[0].clone(),
                submitted[1].clone()
            ],
            "the feeder must honour the reorder, not the admission order"
        );
        stop_feeder(shutdown, feeder).await;
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
        let (mut state, rx) = durable_state(db.clone(), output_dir.path());
        install_authoritative_v2(&mut state);
        let journal = state.queue_journal.clone();

        // The real CPU-only dispatch pair — `StartupMode::CpuFallback` spawns
        // the durable feeder plus exactly this worker.
        let feeder_shutdown = tokio_util::sync::CancellationToken::new();
        let feeder = crate::durable_queue_feeder::spawn(state.clone(), feeder_shutdown.clone());
        let worker = tokio::spawn(crate::queue::run_queue_worker(rx, state.clone()));

        let response = tokio::time::timeout(
            Duration::from_secs(5),
            app_with_state(state.clone()).oneshot(
                Request::post("/api/generate")
                    .header("content-type", "application/json")
                    .body(Body::from(generate_body("a cat", 512, 512)))
                    .unwrap(),
            ),
        )
        .await
        .expect("CPU fallback generation must settle")
        .unwrap();
        assert_eq!(response.status(), StatusCode::OK);

        // The row is cleared on the worker as it settles, which is not ordered
        // against this response arriving, so wait for it rather than sampling
        // once. The assertion is unchanged — the row must still be gone; under
        // parallel load the immediate sample raced and made this test flaky in
        // CI. Same bounded-wait shape as
        // `headerless_direct_media_is_durable_without_a_client_operation_id`.
        tokio::time::timeout(Duration::from_secs(5), async {
            while !journal.list_all().is_empty() {
                tokio::task::yield_now().await;
            }
        })
        .await
        .expect("a published generation clears its durable row");

        // The saved print carries the queue job that produced it, projected
        // into the indexed columns the feeder's idempotence gate matches on
        // (`find_completed_output`), which is what keeps a restart from
        // rendering it twice.
        let saved: Vec<(Option<String>, Option<String>, Option<i64>)> = db
            .as_ref()
            .as_ref()
            .unwrap()
            .with_conn(|conn| {
                let mut stmt = conn.prepare(
                    "SELECT json_extract(metadata_json, '$.job_id'),
                            queue_job_id, queue_job_metadata_state
                       FROM generations",
                )?;
                let rows = stmt.query_map([], |row| Ok((row.get(0)?, row.get(1)?, row.get(2)?)))?;
                rows.collect::<Result<Vec<_>, _>>()
                    .map_err(anyhow::Error::from)
            })
            .unwrap();
        assert_eq!(saved.len(), 1, "one print, carrying one job id: {saved:?}");
        let (recorded, projected, projected_state) = &saved[0];
        assert!(recorded.as_deref().is_some_and(|id| !id.is_empty()));
        assert_eq!(
            projected, recorded,
            "the feeder's idempotence gate must recognise a CPU-rendered print"
        );
        assert_eq!(*projected_state, Some(1));

        feeder_shutdown.cancel();
        feeder.await.unwrap();
        worker.abort();
        let _ = worker.await;
    }

    /// A maintenance boot (`MOLD_GPUS=none`) has no dispatch owner at all, so
    /// `run_server` pauses the prior runtime's work and spawns no feeder.
    /// Nothing is hydrated, every row survives, and untouched backlog spends
    /// none of its replay budget on a boot that could not run it.
    #[tokio::test]
    async fn a_boot_with_no_dispatch_owner_replays_nothing_and_keeps_every_row() {
        let output_dir = tempfile::tempdir().unwrap();
        let db = Arc::new(Some(mold_db::MetadataDb::open_in_memory().unwrap()));
        let submitted = seed_retained_jobs(db.clone(), output_dir.path(), 2).await;

        let (state, mut rx) = durable_state(db.clone(), output_dir.path());
        crate::durable_queue_feeder::recover_runtime(&state)
            .await
            .unwrap();

        assert!(rx.try_recv().is_err());
        assert!(state.job_registry.snapshot().entries.is_empty());
        let rows = state.queue_journal.list_all();
        assert_eq!(
            rows.iter().map(|row| row.id.clone()).collect::<Vec<_>>(),
            submitted
        );
        assert!(
            rows.iter().all(|row| row.replay_seen == 0),
            "a boot that cannot replay must not spend the row's replay budget"
        );
        assert!(
            rows.iter()
                .all(|row| row.state == mold_db::generation_queue::QueueRowState::Paused),
            "restart recovery must park every retained row"
        );
    }

    /// The independent guard: whatever the reason a hydration fails, the job
    /// never reached a worker, so its row must survive for the next boot
    /// rather than be deleted by the ticket's ordinary drop.
    #[tokio::test]
    async fn a_job_that_cannot_be_resubmitted_keeps_its_row() {
        let output_dir = tempfile::tempdir().unwrap();
        let db = Arc::new(Some(mold_db::MetadataDb::open_in_memory().unwrap()));
        let submitted = seed_retained_jobs(db.clone(), output_dir.path(), 2).await;

        let (mut state, rx) = durable_state(db.clone(), output_dir.path());
        // The dispatch owner is gone, so every send fails — but the feeder
        // runs anyway. Its one worker stops on the closed transport.
        state.queue_capacity = 1;
        drop(rx);
        let (_shutdown, feeder) = boot_feeder(&state).await;
        tokio::time::timeout(Duration::from_secs(5), feeder)
            .await
            .expect("a closed transport stops the feeder")
            .unwrap();

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
        let (mut state, _rx) = durable_state(db, output_dir);
        let (scheduled_tx, _scheduled_rx) = tokio::sync::mpsc::channel(1);
        state.scheduled_work = crate::scheduler::ScheduledWorkHandle::new(scheduled_tx);
        let app = app_with_state(state);
        let mut submitted = Vec::new();
        for index in 0..count {
            let request: serde_json::Value =
                serde_json::from_str(&generate_body(&format!("prompt {index}"), 512, 512)).unwrap();
            let response = app
                .clone()
                .oneshot(json_request(
                    "POST",
                    "/api/generation-batches",
                    serde_json::json!({
                        "client_batch_id": uuid::Uuid::new_v4().to_string(),
                        "requests": [request],
                    }),
                ))
                .await
                .unwrap();
            assert_eq!(response.status(), StatusCode::ACCEPTED);
            let body = json_body(response).await;
            submitted.push(body["children"][0]["job_id"].as_str().unwrap().to_string());
        }
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

        let (mut state, mut rx) = durable_state(db.clone(), output_dir.path());
        state.queue_capacity = 1;
        // The configured path is unchanged; the directory itself is gone.
        // Removed after the fixture builds, or it would just recreate it.
        std::fs::remove_dir_all(durable_gallery_dir(output_dir.path())).unwrap();
        let (shutdown, feeder) = boot_feeder(&state).await;

        let mut replayed = Vec::new();
        for _ in 0..2 {
            let job = receive_and_settle(&state, &mut rx).await;
            assert_eq!(
                job.output_dir.as_deref(),
                Some(durable_gallery_dir(output_dir.path()).as_path())
            );
            replayed.push(job.id);
        }
        assert_eq!(replayed, submitted);
        assert!(durable_gallery_dir(output_dir.path()).is_dir());
        assert!(
            state.queue_journal.list_all().is_empty(),
            "nothing was parked"
        );
        stop_feeder(shutdown, feeder).await;
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
        // No dispatch owner: the runtime is recovered and no feeder spawns,
        // so nothing is registered.
        crate::durable_queue_feeder::recover_runtime(&state)
            .await
            .unwrap();
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
            assert_eq!(entry["state"], "paused");
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
    /// `metadata_json` would defeat the whole guarantee. The feeder releases
    /// the claim and retries later; until then nothing is rendered and the
    /// row's replay budget is untouched.
    #[tokio::test]
    async fn the_feeder_hydrates_nothing_when_the_idempotence_gate_cannot_be_checked() {
        let output_dir = tempfile::tempdir().unwrap();
        let db = Arc::new(Some(mold_db::MetadataDb::open_in_memory().unwrap()));
        let submitted = seed_retained_jobs(db.clone(), output_dir.path(), 2).await;

        let (mut state, mut rx) = durable_state(db.clone(), output_dir.path());
        // One feeder worker, so the injected failure is the only pass that
        // runs before the retry delay.
        state.queue_capacity = 1;
        state.queue_journal.fail_completion_lookup_for_tests();
        let (shutdown, feeder) = boot_feeder(&state).await;

        // The pass is over once the injected failure was consumed and the
        // claim it took has been handed back.
        let claim_db = db.clone();
        tokio::time::timeout(Duration::from_secs(5), async {
            loop {
                let claimed = claim_db
                    .as_ref()
                    .as_ref()
                    .unwrap()
                    .with_conn(|conn| {
                        conn.query_row(
                            "SELECT COUNT(*) FROM generation_queue WHERE claim_token IS NOT NULL",
                            [],
                            |row| row.get::<_, i64>(0),
                        )
                        .map_err(anyhow::Error::from)
                    })
                    .unwrap();
                if !state
                    .queue_journal
                    .completion_lookup_failure_pending_for_tests()
                    && claimed == 0
                {
                    break;
                }
                tokio::task::yield_now().await;
            }
        })
        .await
        .expect("the failed pass releases its claim");

        assert!(rx.try_recv().is_err(), "nothing may be rendered unverified");
        let rows = state.queue_journal.list_all();
        assert_eq!(
            rows.iter().map(|row| row.id.clone()).collect::<Vec<_>>(),
            submitted
        );
        assert!(rows
            .iter()
            .all(|row| row.state == mold_db::generation_queue::QueueRowState::Queued));
        assert!(
            rows.iter().all(|row| row.replay_seen == 0),
            "the retry must get a full budget to try again"
        );
        stop_feeder(shutdown, feeder).await;
    }

    /// A job that finished between its last save and the crash must not be
    /// re-rendered — output filenames are wall-clock, so a duplicate print
    /// could never be merged afterwards.
    #[tokio::test]
    async fn a_retained_generation_whose_print_already_exists_is_never_replayed() {
        let output_dir = tempfile::tempdir().unwrap();
        let db = Arc::new(Some(mold_db::MetadataDb::open_in_memory().unwrap()));

        let finished_id = seed_retained_jobs(db.clone(), output_dir.path(), 1)
            .await
            .pop()
            .unwrap();

        // The print landed; only the journal delete was lost.
        let gallery = mold_db::canonical_dir_string(&durable_gallery_dir(output_dir.path()));
        db.as_ref()
            .as_ref()
            .unwrap()
            .with_conn(|conn| {
                conn.execute(
                    "INSERT INTO generations
                        (filename, output_dir, created_at_ms, format, model, metadata_json,
                         queue_job_id, queue_job_metadata_state)
                     VALUES ('done.png', ?1, ?2, 'png', 'mock-model', ?3, ?4, 1)",
                    (
                        gallery.as_str(),
                        mold_core::time::now_epoch_ms(),
                        format!("{{\"job_id\":\"{finished_id}\"}}"),
                        finished_id.as_str(),
                    ),
                )?;
                Ok(())
            })
            .unwrap();

        let (state, mut rx) = durable_state(db.clone(), output_dir.path());
        let (shutdown, feeder) = boot_feeder(&state).await;

        tokio::time::timeout(Duration::from_secs(5), async {
            while !state.queue_journal.list_all().is_empty() {
                tokio::task::yield_now().await;
            }
        })
        .await
        .expect("the already-published row is settled without a render");
        assert!(rx.try_recv().is_err(), "nothing may be resubmitted");
        assert!(state.job_registry.snapshot().entries.is_empty());
        stop_feeder(shutdown, feeder).await;
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
                media_set_id: None,
                admission_authority: None,
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

        let (shutdown, feeder) = boot_feeder(&state).await;

        let row = tokio::time::timeout(Duration::from_secs(5), async {
            loop {
                let row = state.queue_journal.list_all().pop().unwrap();
                if row.state == mold_db::generation_queue::QueueRowState::Held {
                    break row;
                }
                tokio::task::yield_now().await;
            }
        })
        .await
        .expect("the unreadable row is held");
        assert!(rx.try_recv().is_err());
        assert!(row.held_reason.is_some());
        stop_feeder(shutdown, feeder).await;
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
                media_set_id: None,
                admission_authority: None,
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

        // Our own, younger row proves the feeder walked past the foreign one
        // rather than simply not having run yet.
        let submitted = seed_retained_jobs(db.clone(), output_dir.path(), 1).await;

        let (state, mut rx) = durable_state(db.clone(), output_dir.path());
        let (shutdown, feeder) = boot_feeder(&state).await;
        let mut ours = tokio::time::timeout(Duration::from_secs(5), rx.recv())
            .await
            .expect("our own row is hydrated")
            .expect("open queue");
        assert_eq!(ours.id, submitted[0]);
        assert!(rx.try_recv().is_err());

        let theirs = mold_db::generation_queue::get(db.as_ref().as_ref().unwrap(), "theirs")
            .unwrap()
            .expect("the foreign row is untouched");
        assert_eq!(
            theirs.state,
            mold_db::generation_queue::QueueRowState::Queued
        );
        assert_eq!(theirs.replay_seen, 0);
        assert_eq!(
            state.queue_journal.list_all().len(),
            1,
            "the foreign row is not ours to list either"
        );
        ours.journal.take().unwrap().discard();
        state.job_registry.remove(&ours.id);
        state.queue.decrement();
        stop_feeder(shutdown, feeder).await;
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
        let (mut state, _rx) = durable_state(db.clone(), output_dir.path());
        let (scheduled_tx, _scheduled_rx) = tokio::sync::mpsc::channel(1);
        state.scheduled_work = crate::scheduler::ScheduledWorkHandle::new(scheduled_tx);
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

        let request: serde_json::Value =
            serde_json::from_str(&generate_body("a cat", 512, 512)).unwrap();
        let response = app
            .clone()
            .oneshot(json_request(
                "POST",
                "/api/generation-batches",
                serde_json::json!({
                    "client_batch_id": uuid::Uuid::new_v4().to_string(),
                    "requests": [request],
                }),
            ))
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::ACCEPTED);
        let body = json_body(response).await;
        let job_id = body["children"][0]["job_id"].as_str().unwrap().to_string();

        let listing = json_body(
            app.clone()
                .oneshot(Request::get("/api/queue").body(Body::empty()).unwrap())
                .await
                .unwrap(),
        )
        .await;
        let live = &listing["entries"][0];
        assert_eq!(live["id"], serde_json::json!(job_id));
        assert_eq!(live["durable"], true);
        assert_eq!(live["replayed"], false);
        assert_eq!(live["dispatch_attempts"], 0);

        // Park it the way an exhausted attempt cap would.
        mold_db::generation_queue::hold(
            db.as_ref().as_ref().unwrap(),
            &job_id,
            "dispatch attempts exhausted",
            mold_core::time::now_epoch_ms(),
        )
        .unwrap();
        state.job_registry.remove(&job_id);

        let listing = json_body(
            app.oneshot(Request::get("/api/queue").body(Body::empty()).unwrap())
                .await
                .unwrap(),
        )
        .await;
        let held = &listing["entries"][0];
        assert_eq!(held["id"], serde_json::json!(job_id));
        assert_eq!(held["state"], "held");
        assert_eq!(held["held_reason"], "dispatch attempts exhausted");

        // A held job has no registry entry, so the documented way to clear one
        // has to reach the journal directly or the row is unreachable short of
        // editing the database.
        let response = app_with_state(state.clone())
            .oneshot(
                Request::delete(format!("/api/queue/{job_id}"))
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
    async fn any_direct_batch_is_rejected_before_preparation_or_reservation() {
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
            assert_eq!(body["code"], "DIRECT_BATCH_UNSUPPORTED");
            assert_eq!(
                body["error"],
                "direct generation accepts one output; submit durable singleton siblings through /api/generation-batches"
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
        let (state, mut rx, _gallery_root) = durable_test_state(MockEngine::ready());
        spawn_durable_feeder(&state);
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
        // The feeder still runs: it is what moves the durable row into that
        // channel in the first place.
        let (state, _rx, _gallery_root) = durable_test_state(MockEngine::ready());
        spawn_durable_feeder(&state);
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
        // The feeder still runs: without it the durable row never reaches the
        // runtime queue and the attached observer never resolves at all.
        let (state, _rx, _gallery_root) = durable_test_state(MockEngine::ready());
        spawn_durable_feeder(&state);
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
            axum::body::to_bytes(resp.into_body(), 8 * 1024 * 1024),
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
    fn generating_app_with_history_db() -> (axum::Router, AppState, tempfile::TempDir) {
        let (state, rx, root) = durable_test_state(MockEngine::ready());
        spawn_durable_runtime(&state, rx);
        (app_with_state(state.clone()), state, root)
    }

    /// As [`generating_app_with_history_db`], but with no queue worker, so an
    /// admitted job reaches the runtime queue and stays there. Tests that
    /// inspect a job WHILE it is queued need the feeder (which is what puts it
    /// there) and must not have the worker (which would finish it first).
    fn queueing_app_with_history_db() -> (
        axum::Router,
        AppState,
        tokio::sync::mpsc::Receiver<crate::state::GenerationJob>,
        tempfile::TempDir,
    ) {
        let (state, rx, root) = durable_test_state(MockEngine::ready());
        spawn_durable_feeder(&state);
        (app_with_state(state.clone()), state, rx, root)
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
        let (app, _state, _gallery_root) = generating_app_with_history_db();

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
        let (app, _state, _gallery_root) = generating_app_with_history_db();

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
        let (app, state, _rx, _gallery_root) = queueing_app_with_history_db();

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
        let _lock = env_lock();
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
        let _lock = env_lock();
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
        let _lock = env_lock();
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
        let _lock = env_lock();
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
        let _lock = env_lock();
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

        let body = axum::body::to_bytes(resp.into_body(), 8 * 1024 * 1024)
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

        let body = axum::body::to_bytes(resp.into_body(), 8 * 1024 * 1024)
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
                reclaimable_zfs_arc: None,
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

        let body = axum::body::to_bytes(resp.into_body(), 8 * 1024 * 1024)
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
        let (app, _gallery_root) = app_with(MockEngine::blocking_generate(blocker.clone()));

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
        let (app, _gallery_root) = app_with(MockEngine::ready());
        let resp = app
            .oneshot(Request::get("/api/models").body(Body::empty()).unwrap())
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
    }

    #[tokio::test]
    async fn list_models_uses_manifest_defaults_for_unpulled() {
        let (app, _gallery_root) = app_with(MockEngine::ready());
        let resp = app
            .oneshot(Request::get("/api/models").body(Body::empty()).unwrap())
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::OK);

        let body = axum::body::to_bytes(resp.into_body(), 8 * 1024 * 1024)
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
    async fn list_models_surfaces_every_pinned_h3_download_and_marks_runnability() {
        // Every pinned H3 manifest is a download, so every one lists. What
        // tells them apart on the wire is the additive `runtime_available`
        // plus, since #1276, the `runtime_unavailable_reason` naming the
        // obstacle — never their absence from the list. Runnability is a
        // property of THIS BUILD, not of the family: only a binary compiled
        // with the `h3` feature runs anything at all, Ref2VA runs on no
        // released build, and the official BF16 references and pruned NVFP4
        // transformers have no engine arm anywhere.
        let (app, _gallery_root) = app_with(MockEngine::ready());
        let resp = app
            .oneshot(Request::get("/api/models").body(Body::empty()).unwrap())
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::OK);

        let body = axum::body::to_bytes(resp.into_body(), 8 * 1024 * 1024)
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
                    model["runtime_available"].as_bool().unwrap(),
                    model["runtime_unavailable_reason"].as_str(),
                )
            })
            .collect::<std::collections::BTreeSet<_>>();
        use mold_core::minimax_h3::{
            RuntimeUnavailableReason, COMFY_REPO, FL2VA_COMFY, FL2VA_COMFY_NVFP4,
            FL2VA_COMFY_TURBO_4STEP_768P, FL2VA_COMFY_TURBO_8STEP, FL2VA_OFFICIAL, NVFP4_REPO,
            OFFICIAL_REPO, REF2VA_COMFY, REF2VA_COMFY_NVFP4, REF2VA_COMFY_TURBO_4STEP,
            REF2VA_OFFICIAL,
        };
        // Both compact task partitions execute since #825, so their answer
        // depends only on how this binary was compiled; the pinned layouts
        // below are unrunnable everywhere.
        let fl2va = if mold_core::minimax_h3::engine_is_built() {
            (true, None)
        } else {
            (
                false,
                Some(RuntimeUnavailableReason::EngineNotBuilt.message()),
            )
        };
        let ref2va = fl2va;
        let no_loader = (
            false,
            Some(RuntimeUnavailableReason::UnsupportedLayout.message()),
        );
        assert_eq!(
            h3,
            std::collections::BTreeSet::from([
                (FL2VA_COMFY, false, COMFY_REPO, fl2va.0, fl2va.1),
                (REF2VA_COMFY, false, COMFY_REPO, ref2va.0, ref2va.1),
                (FL2VA_COMFY_TURBO_8STEP, false, COMFY_REPO, fl2va.0, fl2va.1),
                (
                    FL2VA_COMFY_TURBO_4STEP_768P,
                    false,
                    COMFY_REPO,
                    fl2va.0,
                    fl2va.1
                ),
                (
                    REF2VA_COMFY_TURBO_4STEP,
                    false,
                    COMFY_REPO,
                    ref2va.0,
                    ref2va.1
                ),
                (
                    FL2VA_OFFICIAL,
                    false,
                    OFFICIAL_REPO,
                    no_loader.0,
                    no_loader.1
                ),
                (
                    REF2VA_OFFICIAL,
                    false,
                    OFFICIAL_REPO,
                    no_loader.0,
                    no_loader.1
                ),
                (
                    FL2VA_COMFY_NVFP4,
                    false,
                    NVFP4_REPO,
                    no_loader.0,
                    no_loader.1
                ),
                (
                    REF2VA_COMFY_NVFP4,
                    false,
                    NVFP4_REPO,
                    no_loader.0,
                    no_loader.1
                ),
            ])
        );
    }

    /// #1276: whatever the row says is unrunnable, generation refuses with
    /// HTTP 501 and the SAME sentence. A user who reads "Ref2VA execution is
    /// not available in any released build" on the model card must not then
    /// get a licensing refusal at submit.
    ///
    /// `prepare_generation` runs `classify_h3_private_ingress` BEFORE model
    /// activation, so on a build that compiles private H3 ingress the answer
    /// depends on which obstacle the row names. #1354 made the classifier
    /// defer for an identity whose obstacle is its weight LAYOUT — the pinned
    /// `official-bf16` references and pruned-NVFP4 tags — so those rows reach
    /// `model_runtime_availability` and answer 501 there exactly as they do on
    /// a public build.
    ///
    /// The other two obstacles keep the private boundary's answer, and
    /// deliberately: deferring on `UnsupportedTask` would open the private
    /// Ref2VA ingress seam, and deferring on `EngineNotBuilt` would take the
    /// whole `h3-private-uat` runtime off its own path. What such a build
    /// promises for those is that the private boundary refuses before anything
    /// is admitted, with a credential to satisfy first when public `h3` is
    /// off. The 401 is therefore not the assertion — the authenticated refusal
    /// below is.
    #[tokio::test]
    async fn every_unrunnable_h3_row_is_refused_at_generation_with_its_own_reason() {
        let (app, _gallery_root) = app_with(MockEngine::ready());
        let resp = app
            .clone()
            .oneshot(Request::get("/api/models").body(Body::empty()).unwrap())
            .await
            .unwrap();
        let body = axum::body::to_bytes(resp.into_body(), 8 * 1024 * 1024)
            .await
            .unwrap();
        let models: Vec<serde_json::Value> = serde_json::from_slice(&body).unwrap();

        let mut refused = 0;
        for model in models
            .iter()
            .filter(|model| model["family"] == mold_core::minimax_h3::FAMILY)
            .filter(|model| model["runtime_available"] == serde_json::json!(false))
        {
            let name = model["name"].as_str().unwrap();
            let reason = model["runtime_unavailable_reason"]
                .as_str()
                .unwrap_or_else(|| panic!("{name} reports no reason"));
            let body = serde_json::json!({
                "prompt": "a cat",
                "model": name,
                "width": mold_core::minimax_h3::DEFAULT_WIDTH,
                "height": mold_core::minimax_h3::DEFAULT_HEIGHT,
                "steps": mold_core::minimax_h3::DEFAULT_STEPS,
                "guidance": 0.0,
                "batch_size": 1,
                "frames": mold_core::minimax_h3::REVIEWED_COMPACT_FRAMES,
                "fps": mold_core::minimax_h3::FIXED_FPS,
                "output_format": "mp4",
                // Durable admission validates request fields before it asks
                // the runtime-availability question, so the probe must be a
                // request the family would otherwise accept.
                "strength": 1.0
            })
            .to_string();
            let submit = || {
                Request::post("/api/generate")
                    .header("content-type", "application/json")
                    .body(Body::from(body.clone()))
                    .unwrap()
            };

            // A layout with no engine arm anywhere is refused by the public
            // authority on every build; only the task and engine obstacles
            // stay behind the private boundary.
            let layout_obstacle = reason
                == mold_core::minimax_h3::RuntimeUnavailableReason::UnsupportedLayout.message();
            if cfg!(any(feature = "h3", feature = "h3-private-uat")) && !layout_obstacle {
                // The credential comes first on a build without public `h3`;
                // a public H3 build derives the ingress identity from the
                // server instance and goes straight to the partition.
                let resp = app.clone().oneshot(submit()).await.unwrap();
                assert!(
                    resp.status() == StatusCode::UNAUTHORIZED
                        || resp.status() == StatusCode::UNPROCESSABLE_ENTITY,
                    "{name}: {}",
                    resp.status()
                );
                // Authenticated, the private partition itself refuses: an
                // unrunnable row is never the reviewed compact partition, so
                // nothing reaches admission or the queue.
                let mut authenticated = submit();
                authenticated
                    .extensions_mut()
                    .insert(crate::auth::ApiKeyAuthenticated {
                        identity: "unrunnable-h3-row-test".to_string(),
                        durable_identity: "unrunnable-h3-row-test-stable".to_string(),
                    });
                let resp = app.clone().oneshot(authenticated).await.unwrap();
                assert_eq!(resp.status(), StatusCode::UNPROCESSABLE_ENTITY, "{name}");
                refused += 1;
                continue;
            }

            let resp = app.clone().oneshot(submit()).await.unwrap();
            assert_eq!(resp.status(), StatusCode::NOT_IMPLEMENTED, "{name}");
            let error = json_body(resp).await;
            assert_eq!(
                error["code"],
                mold_core::MINIMAX_H3_RUNTIME_UNAVAILABLE,
                "{name}"
            );
            let message = error["error"].as_str().unwrap();
            assert!(message.contains(reason), "{name}: {message} != {reason}");
            refused += 1;
        }
        assert!(refused > 0, "no unrunnable H3 row was listed");
    }

    /// #1354: a pinned H3 identity mold has no engine arm for answers with
    /// its own `/api/models` sentence on EVERY build, private ingress
    /// included.
    ///
    /// `prepare_generation` runs `classify_h3_private_ingress` before model
    /// activation, and that classifier used to claim every identity
    /// `capability_contract_for_model` resolves — which is every H3 row. So an
    /// `official-bf16` reference or a pruned-NVFP4 tag was answered
    /// `422 MINIMAX_H3_PRIVATE_PARTITION_REJECTED` ("accepts only its
    /// supported compact task partition") on a build that compiles H3, while
    /// its row promised `501 MINIMAX_H3_RUNTIME_UNAVAILABLE` and a sentence
    /// about a missing weight-layout loader. The private boundary is an
    /// authorization gate for identities mold can RUN; a layout with no
    /// engine arm anywhere is not one of them, so it defers.
    ///
    /// The deferral sits ahead of the credential check on purpose: a build
    /// without public `h3` demands an API key before it classifies anything,
    /// and answering 401 here would still hide the row's own reason behind a
    /// gate that protects nothing — there is no runtime to protect.
    #[tokio::test]
    async fn pinned_unrunnable_h3_identities_refuse_with_their_own_row_reason() {
        use mold_core::minimax_h3::{
            FL2VA_COMFY_NVFP4, FL2VA_OFFICIAL, REF2VA_COMFY_NVFP4, REF2VA_OFFICIAL,
        };
        let (app, _gallery_root) = app_with(MockEngine::ready());
        for name in [
            FL2VA_OFFICIAL,
            REF2VA_OFFICIAL,
            FL2VA_COMFY_NVFP4,
            REF2VA_COMFY_NVFP4,
        ] {
            let reason = mold_core::minimax_h3::model_runtime_availability(name)
                .reason()
                .unwrap_or_else(|| panic!("{name} must be pinned as unrunnable"))
                .message();
            let body = serde_json::json!({
                "prompt": "a cat",
                "model": name,
                "width": mold_core::minimax_h3::DEFAULT_WIDTH,
                "height": mold_core::minimax_h3::DEFAULT_HEIGHT,
                "steps": mold_core::minimax_h3::DEFAULT_STEPS,
                "guidance": 0.0,
                "batch_size": 1,
                "frames": mold_core::minimax_h3::REVIEWED_COMPACT_FRAMES,
                "fps": mold_core::minimax_h3::FIXED_FPS,
                "output_format": "mp4"
            })
            .to_string();
            let resp = app
                .clone()
                .oneshot(
                    Request::post("/api/generate")
                        .header("content-type", "application/json")
                        .body(Body::from(body))
                        .unwrap(),
                )
                .await
                .unwrap();
            assert_eq!(resp.status(), StatusCode::NOT_IMPLEMENTED, "{name}");
            let error = json_body(resp).await;
            assert_eq!(
                error["code"],
                mold_core::MINIMAX_H3_RUNTIME_UNAVAILABLE,
                "{name}"
            );
            let message = error["error"].as_str().unwrap();
            assert!(message.contains(reason), "{name}: {message} != {reason}");
        }
    }

    /// Sequence capability is advertised per model, so a picker never has to
    /// infer it from the checkpoint name. Every LTX-2 checkpoint qualifies,
    /// dev included — the old name heuristic hid dev checkpoints from the
    /// Sequence picker even though the server chains them.
    #[tokio::test]
    async fn list_models_advertise_per_model_sequence_support() {
        let (app, _gallery_root) = app_with(MockEngine::ready());
        let resp = app
            .oneshot(Request::get("/api/models").body(Body::empty()).unwrap())
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
        let body = axum::body::to_bytes(resp.into_body(), 8 * 1024 * 1024)
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
        let _lock = env_lock();
        let models_dir = test_models_dir("remote-catalog");
        populate_manifest_files(&models_dir, "flux-schnell:q8");
        std::env::set_var("MOLD_MODELS_DIR", &models_dir);

        let app = app_empty();
        let resp = app
            .oneshot(Request::get("/api/models").body(Body::empty()).unwrap())
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::OK);

        let body = axum::body::to_bytes(resp.into_body(), 8 * 1024 * 1024)
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
        let _lock = env_lock();
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
        let _lock = env_lock();
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
        let _lock = env_lock();
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
        let (app, _gallery_root) = app_with(MockEngine::blocking_generate(blocker.clone()));

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
        let (app, _gallery_root) = app_with(MockEngine::ready());
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
        let (app, _gallery_root) = app_with(MockEngine::ready());
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
            "requests[1]: Qwen Image Edit needs at least one image. Add a Target image and try again."
        );
    }

    #[tokio::test]
    async fn generate_zero_width_returns_422() {
        let (app, _gallery_root) = app_with(MockEngine::ready());
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
        let (app, _gallery_root) = app_with(MockEngine::ready());
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
        let (app, _gallery_root) = app_with(MockEngine::ready());
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
        let (app, _gallery_root) = app_with(MockEngine::ready());
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
        let (app, _gallery_root) = app_with(MockEngine::ready());
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
        let (app, _gallery_root) = app_with(MockEngine::ready());
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

    /// A render that fails while the caller is still attached is the
    /// caller's error, in the shape the singleton contract always had: an
    /// HTTP error carrying the engine's own sentence. The durable row is
    /// still HELD and retryable — the print is parked, not lost — so the
    /// error names the job the caller can resume, and the batch behind the
    /// client id it sent still answers with the same hold.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn generate_engine_error_is_the_childs_error_naming_its_retry() {
        let (app, _gallery_root) = app_with(MockEngine::failing());
        let body = generate_body("a cat", 768, 768);
        let client_batch_id = "0d4b1b1e-6f8e-4f52-9d4a-6b3e0c1f2a01";
        let resp = app
            .clone()
            .oneshot(
                Request::post("/api/generate")
                    .header("content-type", "application/json")
                    .header("x-mold-client-batch-id", client_batch_id)
                    .body(Body::from(body))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::INTERNAL_SERVER_ERROR);
        let error = json_body(resp).await;
        assert_eq!(error["code"], "INFERENCE_ERROR", "{error}");
        let message = error["error"].as_str().unwrap_or_default();
        assert!(message.contains("mock engine error"), "{error}");
        assert!(message.contains("/api/queue/"), "{error}");
        assert!(message.contains("/retry"), "{error}");

        let batch = app
            .oneshot(
                Request::get(format!(
                    "/api/generation-batches/by-client/{client_batch_id}"
                ))
                .body(Body::empty())
                .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(batch.status(), StatusCode::OK);
        let batch = json_body(batch).await;
        let child = &batch["children"][0];
        assert_eq!(child["state"], "held", "{batch}");
        assert_eq!(child["retryable"], true, "{batch}");
        assert!(
            message.contains(child["job_id"].as_str().unwrap()),
            "the error must name the held job: {message} vs {batch}"
        );
    }

    /// A client-chosen batch id makes `/api/generate` idempotent exactly as
    /// the batch route is: the retry of a lost response is answered with
    /// the batch the first attempt admitted, never a second print.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn generate_replays_a_client_batch_id_with_its_batch_status() {
        let (app, _gallery_root) = app_with(MockEngine::ready());
        let body = generate_body("a glowing robot", 768, 768);
        let client_batch_id = "6c2f3a7e-2d1b-4c58-8a0e-9f1d2b3c4d5e";
        let first = app
            .clone()
            .oneshot(
                Request::post("/api/generate")
                    .header("content-type", "application/json")
                    .header("x-mold-client-batch-id", client_batch_id)
                    .body(Body::from(body.clone()))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(first.status(), StatusCode::OK);
        assert_eq!(first.headers().get("content-type").unwrap(), "image/png");

        let replay = app
            .clone()
            .oneshot(
                Request::post("/api/generate")
                    .header("content-type", "application/json")
                    .header("x-mold-client-batch-id", client_batch_id)
                    .body(Body::from(body.clone()))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(replay.status(), StatusCode::OK);
        assert!(
            replay
                .headers()
                .get("content-type")
                .unwrap()
                .to_str()
                .unwrap()
                .starts_with("application/json"),
            "a replay answers with the batch, not a second render"
        );
        let status = json_body(replay).await;
        assert_eq!(status["client_batch_id"], client_batch_id, "{status}");
        let child = &status["children"][0];
        assert_eq!(child["state"], "complete", "{status}");
        assert!(child["result"]["filename"].is_string(), "{status}");

        // The streaming facade replays the same way.
        let stream_replay = app
            .oneshot(
                Request::post("/api/generate/stream")
                    .header("content-type", "application/json")
                    .header("x-mold-client-batch-id", client_batch_id)
                    .body(Body::from(body))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(stream_replay.status(), StatusCode::OK);
        let status = json_body(stream_replay).await;
        assert_eq!(status["children"][0]["state"], "complete", "{status}");
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn generate_refuses_a_malformed_client_batch_id() {
        let (app, _gallery_root) = app_with(MockEngine::ready());
        let resp = app
            .oneshot(
                Request::post("/api/generate")
                    .header("content-type", "application/json")
                    .header("x-mold-client-batch-id", "not-a-uuid")
                    .body(Body::from(generate_body("a cat", 768, 768)))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::UNPROCESSABLE_ENTITY);
        let error = json_body(resp).await;
        assert!(
            error["error"]
                .as_str()
                .unwrap_or_default()
                .contains("X-Mold-Client-Batch-Id"),
            "{error}"
        );
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn generate_empty_images_is_an_inference_error_naming_the_held_job() {
        let (app, _gallery_root) = app_with(MockEngine::empty_images());
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
        assert_eq!(body["code"], "INFERENCE_ERROR", "{body}");
        let message = body["error"].as_str().unwrap_or_default();
        assert!(message.contains("returned no images"), "{body}");
        assert!(message.contains("/retry"), "{body}");
    }

    // ── /api/generate — known but not downloaded model returns 404 ───────────

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    #[allow(clippy::await_holding_lock)]
    async fn generate_known_model_not_downloaded_is_a_404_naming_the_held_job() {
        let _lock = env_lock();
        let models_dir = test_models_dir("generate-not-downloaded");
        std::fs::create_dir_all(&models_dir).unwrap();
        std::env::set_var("MOLD_MODELS_DIR", &models_dir);

        let (app, _gallery_root) = app_with(MockEngine::ready());
        // flux-schnell:q8 is a known manifest model but not configured/downloaded
        let body = r#"{"prompt":"a cat","model":"flux-schnell:q8","width":768,"height":768,"steps":4,"batch_size":1,"output_format":"png"}"#;
        let resp = tokio::time::timeout(
            Duration::from_secs(10),
            app.oneshot(
                Request::post("/api/generate")
                    .header("content-type", "application/json")
                    .body(Body::from(body))
                    .unwrap(),
            ),
        )
        .await
        .expect("an undownloaded model must settle the request")
        .unwrap();
        // The singleton contract's 404 — the status the CLI's auto-pull reads
        // — carrying the held child's code and the job the pull resumes.
        assert_eq!(resp.status(), StatusCode::NOT_FOUND);
        let body = json_body(resp).await;
        assert_eq!(
            body["code"],
            mold_core::SSE_ERROR_CODE_MODEL_NOT_FOUND,
            "{body}"
        );
        let message = body["error"].as_str().unwrap_or_default();
        assert!(message.contains("flux-schnell:q8"), "{body}");
        assert!(message.contains("/retry"), "{body}");

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
        let (app, _gallery_root) = app_with(MockEngine::ready());
        let resp = app
            .oneshot(
                Request::get("/api/openapi.json")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::OK);

        let body = axum::body::to_bytes(resp.into_body(), 8 * 1024 * 1024)
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
        // A route installed on the router but missing from `ApiDoc`'s paths
        // list is invisible to generated clients and API discovery — the
        // `#[utoipa::path]` annotation alone does not register it.
        assert!(
            spec["paths"]["/api/queue/held/sweep"]["post"].is_object(),
            "spec should document POST /api/queue/held/sweep"
        );
        assert!(
            spec["paths"]["/api/generation-batches/sweep"]["post"].is_object(),
            "spec should document POST /api/generation-batches/sweep"
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
                durable_identity: "private-route-test-stable".to_string(),
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

    /// A placement preview is authoritative, so it must never answer
    /// `planned` for a request generation would refuse. The identity contract
    /// is checked at the same seam the source-image contract (#772) is, and
    /// for the same reason: preview reaches dependency preparation BEFORE the
    /// shared request validator runs, so without this an unqualified
    /// checkpoint carrying an `id_image` would have the PuLID bundle planned
    /// for it and then be 422'd at generate.
    #[tokio::test]
    async fn a_placement_preview_refuses_identity_on_an_unqualified_checkpoint() {
        let state = AppState::for_tests();
        let preview = |model: &str| {
            let state = state.clone();
            let model = model.to_string();
            async move {
                let response = app_with_state(state)
                    .oneshot(
                        Request::post("/api/generate/placement-preview")
                            .header("content-type", "application/json")
                            .body(Body::from(
                                serde_json::json!({
                                    "request": {
                                        "prompt": "a portrait",
                                        "model": model,
                                        "width": 1024,
                                        "height": 1024,
                                        "steps": 25,
                                        "guidance": 7.5,
                                        "id_image": "iVBORw0KGgo=",
                                    },
                                    "copies": 1
                                })
                                .to_string(),
                            ))
                            .unwrap(),
                    )
                    .await
                    .unwrap();
                assert_eq!(response.status(), StatusCode::OK);
                json_body(response).await
            }
        };

        // The sole refused SDXL tier remains an explicit family exception.
        let body = preview("sdxl-turbo:fp16").await;
        assert_eq!(body["outcome"], "infeasible");
        assert_eq!(body["authoritative"], true);
        // Either identity refusal is correct here and both are the request
        // contract's own wording: the model gate on a `pulid` build, the
        // missing-build-support refusal without the feature.
        assert!(
            body["reason"]
                .as_str()
                .unwrap_or_default()
                .contains("face-identity"),
            "{body}"
        );
        assert!(
            body["pending_downloads"]
                .as_array()
                .is_none_or(|downloads| downloads.is_empty()),
            "a refused preview must plan no PuLID bundle: {body}"
        );

        // A model with no identity support at all, for the same reason.
        let body = preview("z-image-turbo:q4").await;
        assert_eq!(body["outcome"], "infeasible");
        assert_eq!(body["authoritative"], true);
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
                "frames": mold_core::minimax_h3::REVIEWED_COMPACT_FRAMES,
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
                durable_identity: "test-key-stable".to_string(),
            });
        let created = app.oneshot(request).await.unwrap();
        // Authentication is the gate this test owns, and it is checked
        // before anything is staged. What happens next is this build's
        // runtime answer for the compact Ref2VA partition: since #825 it
        // executes wherever the engine is linked, and a build without one
        // refuses with its own sentence rather than staging media it could
        // never consume (#1276). Either way nothing is staged here, because
        // the session body names no upload yet.
        if mold_core::minimax_h3::engine_is_built() {
            assert_ne!(created.status(), StatusCode::UNAUTHORIZED);
            assert_ne!(created.status(), StatusCode::NOT_IMPLEMENTED);
        } else {
            assert_eq!(created.status(), StatusCode::NOT_IMPLEMENTED);
            let created = json_body(created).await;
            assert_eq!(created["code"], mold_core::MINIMAX_H3_RUNTIME_UNAVAILABLE);
            assert!(
                created["error"].as_str().unwrap().contains(
                    mold_core::minimax_h3::RuntimeUnavailableReason::EngineNotBuilt.message()
                ),
                "{created}"
            );
            assert!(!state.reference_uploads.staging_exists());
        }
    }

    /// A host with API-key auth disabled admits an inline Ref2VA set on the
    /// durable path and refuses an upload handle it has no identity to bind;
    /// a keyed host refuses the keyless submission outright.
    #[cfg(feature = "h3")]
    #[tokio::test(flavor = "current_thread")]
    async fn auth_disabled_host_accepts_inline_ref2va_but_auth_enabled_host_requires_a_key() {
        let (state, _rx, _root) = durable_test_state(MockEngine::ready_for_model(
            mold_core::minimax_h3::REF2VA_COMFY,
        ));
        let journal = state.queue_journal.clone();
        let image = minimal_png();
        let body = inline_ref2va_body("authless inline reference", &image).to_string();
        let authless = authless_app(state.clone());

        // The streaming route answers once the feeder claims the row; this
        // test owns admission, so it waits for the journal instead.
        let stream = tokio::spawn({
            let authless = authless.clone();
            let body = body.clone();
            async move {
                authless
                    .oneshot(
                        Request::post("/api/generate/stream")
                            .header("content-type", "application/json")
                            .body(Body::from(body))
                            .unwrap(),
                    )
                    .await
            }
        });
        tokio::time::timeout(Duration::from_secs(5), async {
            while journal.list_all().len() != 1 {
                tokio::task::yield_now().await;
            }
        })
        .await
        .expect("auth-disabled inline Ref2VA must be admitted durably");
        let rows = journal.list_all();
        assert!(rows[0].media_set_id.is_some());
        let stored: GenerateRequest = serde_json::from_str(&rows[0].request_json).unwrap();
        assert!(matches!(
            stored.references.as_deref().unwrap()[0].media(),
            mold_core::GenerationReferenceAuthority::Descriptor
        ));
        stream.abort();
        let _ = stream.await;

        let mut upload_body: serde_json::Value = serde_json::from_str(&body).unwrap();
        upload_body["references"][0]["media"] = serde_json::json!({
            "authority": "upload",
            "handle": "authless-upload-must-not-resolve"
        });
        let upload_rejected = authless
            .oneshot(
                Request::post("/api/generate/stream")
                    .header("content-type", "application/json")
                    .body(Body::from(upload_body.to_string()))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(upload_rejected.status(), StatusCode::UNAUTHORIZED);
        assert_eq!(journal.list_all().len(), 1);

        let rejected = keyed_app(state)
            .oneshot(
                Request::post("/api/generate/stream")
                    .header("content-type", "application/json")
                    .body(Body::from(body))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(rejected.status(), StatusCode::UNAUTHORIZED);
        assert_eq!(journal.list_all().len(), 1);
    }

    #[tokio::test]
    async fn config_only_h3_generation_is_rejected_before_queueing() {
        let (state, mut queue_rx, _gallery_root) = durable_test_state(MockEngine::ready());
        spawn_durable_feeder(&state);
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
        let (state, mut queue_rx, _gallery_root) = durable_test_state(MockEngine::ready());
        spawn_durable_feeder(&state);
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
        let (app, _gallery_root) = app_with(MockEngine::ready());
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
        let (app, _gallery_root) = app_with(MockEngine::ready());
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
        let body = axum::body::to_bytes(resp.into_body(), 8 * 1024 * 1024)
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
        let (state, rx, _gallery_root) = durable_test_state(MockEngine::ready());
        spawn_durable_feeder(&state);
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
        let (app, _gallery_root) = app_with(MockEngine::ready());
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
        let (app, _gallery_root) = app_with(MockEngine::ready());
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
        let (mut state, rx, _gallery_root) = durable_test_state(MockEngine::ready());
        spawn_durable_feeder(&state);
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
        let body = axum::body::to_bytes(resp.into_body(), 8 * 1024 * 1024)
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
        let (app, _gallery_root) = app_with(MockEngine::ready());
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

    /// Durable admission accepts before it resolves a checkpoint, so an
    /// unknown model is a CODED terminal frame rather than the `400` the
    /// attached path answered with. The code is what keeps every client's
    /// missing-model classifier — and therefore auto-pull — working.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn stream_unknown_model_reports_its_code_on_the_error_frame() {
        let (app, _gallery_root) = app_with(MockEngine::ready());
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
        assert_eq!(resp.status(), StatusCode::OK);
        let bytes = tokio::time::timeout(
            Duration::from_secs(10),
            axum::body::to_bytes(resp.into_body(), 8 * 1024 * 1024),
        )
        .await
        .expect("an unresolvable model must close the stream")
        .unwrap();
        let text = String::from_utf8_lossy(&bytes).into_owned();
        assert!(text.contains("event: error"), "{text}");
        assert!(
            text.contains(mold_core::SSE_ERROR_CODE_UNKNOWN_MODEL),
            "{text}"
        );
    }

    /// The held child carries the refusal's CODE beside its sentence. The
    /// feeder always had the code (it rides the SSE error frame) but only
    /// persisted the sentence, so every client's pull-and-resume offer —
    /// which classifies on `MODEL_NOT_FOUND` / `UNKNOWN_MODEL` — matched
    /// nothing against a real host.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn an_unknown_model_is_a_404_carrying_the_held_childs_code() {
        let (app, _gallery_root) = app_with(MockEngine::ready());
        let body = r#"{"prompt":"a cat","model":"nonexistent-model-xyz","width":768,"height":768,"steps":4,"batch_size":1,"output_format":"png"}"#;
        let client_batch_id = "3f9a0c2d-5b7e-4a1c-9e8d-7c6b5a4f3e2d";
        let response = tokio::time::timeout(
            Duration::from_secs(20),
            app.clone().oneshot(
                Request::post("/api/generate")
                    .header("content-type", "application/json")
                    .header("x-mold-client-batch-id", client_batch_id)
                    .body(Body::from(body))
                    .unwrap(),
            ),
        )
        .await
        .expect("an unresolvable model must settle the request")
        .unwrap();
        // The attached caller gets the refusal in the singleton contract's
        // own shape — a 404 carrying the held child's typed code — so the
        // CLI's missing-model auto-pull classifies on the status it always
        // read, while the durable row behind it keeps the same code.
        assert_eq!(response.status(), StatusCode::NOT_FOUND);
        let error = json_body(response).await;
        assert_eq!(
            error["code"],
            mold_core::SSE_ERROR_CODE_UNKNOWN_MODEL,
            "{error}"
        );
        assert!(
            error["error"]
                .as_str()
                .unwrap_or_default()
                .contains("nonexistent-model-xyz"),
            "{error}"
        );

        let batch = app
            .oneshot(
                Request::get(format!(
                    "/api/generation-batches/by-client/{client_batch_id}"
                ))
                .body(Body::empty())
                .unwrap(),
            )
            .await
            .unwrap();
        let settled = json_body(batch).await;
        let child = &settled["children"][0];
        assert_eq!(child["state"], "held", "{settled}");
        assert_eq!(
            child["error_code"],
            mold_core::SSE_ERROR_CODE_UNKNOWN_MODEL,
            "{settled}"
        );
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    #[allow(clippy::await_holding_lock)]
    async fn stream_known_model_not_downloaded_reports_its_code_on_the_error_frame() {
        let _lock = env_lock();
        let models_dir = test_models_dir("stream-not-downloaded");
        std::fs::create_dir_all(&models_dir).unwrap();
        std::env::set_var("MOLD_MODELS_DIR", &models_dir);

        let (app, _gallery_root) = app_with(MockEngine::ready());
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
        assert_eq!(resp.status(), StatusCode::OK);
        let bytes = tokio::time::timeout(
            Duration::from_secs(10),
            axum::body::to_bytes(resp.into_body(), 8 * 1024 * 1024),
        )
        .await
        .expect("an undownloaded model must close the stream")
        .unwrap();
        let text = String::from_utf8_lossy(&bytes).into_owned();
        assert!(text.contains("event: error"), "{text}");
        assert!(
            text.contains(mold_core::SSE_ERROR_CODE_MODEL_NOT_FOUND),
            "{text}"
        );

        std::env::remove_var("MOLD_MODELS_DIR");
        let _ = std::fs::remove_dir_all(models_dir);
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn stream_engine_error_returns_sse_error() {
        let (app, _gallery_root) = app_with(MockEngine::failing());
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

        let body = axum::body::to_bytes(resp.into_body(), 8 * 1024 * 1024)
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
        let (app, _gallery_root) = app_with(MockEngine::empty_images());
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

        let body = axum::body::to_bytes(resp.into_body(), 8 * 1024 * 1024)
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
        let (app, _gallery_root) = app_with(MockEngine::tracked_progress(
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

        // One install per generation, and never one left behind: the resident
        // engine is cleared when it is picked (nothing loads, so there is no
        // load progress to report), the generation installs its own callback,
        // and clears it before the engine goes back to the cache.
        assert_eq!(progress_set_count.load(Ordering::SeqCst), 1);
        assert_eq!(progress_clear_count.load(Ordering::SeqCst), 2);

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

        // A non-streaming generation installs the progress callback too — its
        // steps feed the snapshot `/api/queue/{id}/preview` serves through the
        // registry relay every job carries — under exactly the same discipline.
        assert_eq!(progress_set_count.load(Ordering::SeqCst), 2);
        assert_eq!(progress_clear_count.load(Ordering::SeqCst), 4);
    }

    #[tokio::test]
    async fn unload_loaded_model_returns_200() {
        let (app, _gallery_root) = app_with(MockEngine::ready());
        let resp = app
            .oneshot(
                Request::delete("/api/models/unload")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
        let body = axum::body::to_bytes(resp.into_body(), 8 * 1024 * 1024)
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

    /// Minimal single-tensor safetensors file, written without pulling the
    /// `safetensors` crate into the server's dev graph.
    fn write_tiny_safetensors(path: &std::path::Path) {
        let header = br#"{"weight":{"dtype":"F32","shape":[2],"data_offsets":[0,8]}}"#;
        let mut bytes = (header.len() as u64).to_le_bytes().to_vec();
        bytes.extend_from_slice(header);
        bytes.extend_from_slice(&1.0f32.to_le_bytes());
        bytes.extend_from_slice(&2.0f32.to_le_bytes());
        std::fs::write(path, bytes).unwrap();
    }

    /// #1273: unload has to give host RAM back. The shared CPU tensor pool
    /// outlives every engine, so parking the active model alone left component
    /// weight maps resident that nothing could reach and nothing would ever
    /// free — the reason a 64 GB host that had rendered images could no longer
    /// admit MiniMax H3 without a process restart.
    #[tokio::test]
    async fn unload_releases_unreferenced_shared_pool_cpu_tensors() {
        let state = AppState::with_engine(MockEngine::ready());
        let dir = tempfile::tempdir().unwrap();
        let weights = dir.path().join("vae.safetensors");
        write_tiny_safetensors(&weights);

        {
            let mut pool = state.shared_pool.lock().unwrap();
            drop(
                pool.load_cpu_tensors(std::slice::from_ref(&weights))
                    .unwrap(),
            );
            assert!(
                pool.retained_cpu_tensor_bytes() > 0,
                "pool should have retained the small component"
            );
        }

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

        assert_eq!(
            state
                .shared_pool
                .lock()
                .unwrap()
                .retained_cpu_tensor_bytes(),
            0,
            "unload must release pooled CPU weights nothing still holds"
        );
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
        let body = axum::body::to_bytes(resp.into_body(), 8 * 1024 * 1024)
            .await
            .unwrap();
        assert!(String::from_utf8_lossy(&body).contains("no model loaded"));
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn concurrent_requests_only_load_existing_engine_once() {
        let load_count = Arc::new(AtomicUsize::new(0));
        let (state, rx, _gallery_root) = durable_test_state(MockEngine::unloaded(
            load_count.clone(),
            Duration::from_millis(50),
        ));
        spawn_durable_feeder(&state);
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
        let (state, rx, _gallery_root) = durable_test_state(MockEngine::unloaded_with_progress());
        spawn_durable_feeder(&state);
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

        let body = axum::body::to_bytes(resp.into_body(), 8 * 1024 * 1024)
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
        let (state, rx, _gallery_root) =
            durable_test_state(MockEngine::blocking_generate(blocker.clone()));
        spawn_durable_feeder(&state);
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
                let body = axum::body::to_bytes(resp.into_body(), 8 * 1024 * 1024)
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
                let body = axum::body::to_bytes(resp.into_body(), 8 * 1024 * 1024)
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
        let (state, rx, _gallery_root) = durable_test_state(MockEngine::ready_for_model(model));
        spawn_durable_feeder(&state);
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
        // The position is a snapshot on the REGISTRY scale, taken when the
        // frame is emitted. Durable admission acknowledges before the feeder
        // claims the row, so a request admitted while its predecessor is still
        // being fed legitimately reports 0 — the exact value is a race and
        // `/api/queue` is the authority. What must hold is that the frame
        // carries the server id the client reconciles against.
        assert!(
            text2.contains(r#""position":"#) && text2.contains(r#""id":"#),
            "the queued frame must carry a position and the server id, got: {text2}"
        );

        tokio::time::timeout(Duration::from_secs(10), body1)
            .await
            .expect("first queued stream should close")
            .expect("first queued stream task should complete")
            .unwrap();
    }

    /// The batch event stream is the authoritative-state channel for Batch N.
    ///
    /// It opens with a whole status so a late or reconnecting client is
    /// correct immediately, emits a whole status again on every committed
    /// child transition, and closes once nothing is left for this host to do.
    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn generation_batch_events_stream_authoritative_status_until_settled() {
        let (state, rx, _gallery_root) = durable_test_state(MockEngine::ready());
        spawn_durable_runtime(&state, rx);
        let app = app_with_state(state);

        let single: serde_json::Value =
            serde_json::from_str(&generate_body("batch events", 64, 64)).unwrap();
        let accepted = app
            .clone()
            .oneshot(json_request(
                "POST",
                "/api/generation-batches",
                serde_json::json!({
                    "client_batch_id": uuid::Uuid::new_v4().to_string(),
                    "requests": [single.clone(), single],
                }),
            ))
            .await
            .unwrap();
        assert_eq!(accepted.status(), StatusCode::ACCEPTED);
        let batch_id = json_body(accepted).await["id"]
            .as_str()
            .unwrap()
            .to_string();

        let events = app
            .oneshot(
                Request::get(format!("/api/generation-batches/{batch_id}/events"))
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(events.status(), StatusCode::OK);
        let body = tokio::time::timeout(
            Duration::from_secs(30),
            axum::body::to_bytes(events.into_body(), 4 * 1024 * 1024),
        )
        .await
        .expect("the stream must close once every child settles")
        .unwrap();
        let text = String::from_utf8_lossy(&body).into_owned();
        assert!(text.contains("event: generation_batch"), "{text}");
        assert!(text.contains(&batch_id), "{text}");
        // The last frame is the settled one, and every frame is a whole
        // status rather than a delta.
        let last = text
            .rsplit("data: ")
            .next()
            .and_then(|frame| frame.lines().next())
            .expect("a data frame");
        let last: serde_json::Value = serde_json::from_str(last).unwrap();
        assert_eq!(last["children"].as_array().unwrap().len(), 2);
        for child in last["children"].as_array().unwrap() {
            assert!(
                ["complete", "failed", "cancelled", "held"]
                    .contains(&child["state"].as_str().unwrap()),
                "{last}"
            );
        }
    }

    /// Two concurrent prints must both come back as prints.
    ///
    /// The feeder prefers claiming rows that have an attached observer, and an
    /// exact claim that fails used to discard the hint AND the observer. Under
    /// concurrency the ordinary FIFO claim routinely wins a row first, so the
    /// second request's observer was destroyed and its caller received a `202`
    /// reconciliation body for a print that rendered perfectly well. Only a
    /// row that is genuinely gone may discard its observer.
    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn concurrent_prints_keep_their_own_observers() {
        let (app, _gallery_root) = app_with(MockEngine::ready());
        let responses = (0..4)
            .map(|index| {
                let app = app.clone();
                tokio::spawn(async move {
                    app.oneshot(
                        Request::post("/api/generate")
                            .header("content-type", "application/json")
                            .body(Body::from(generate_body(
                                &format!("concurrent print {index}"),
                                512,
                                512,
                            )))
                            .unwrap(),
                    )
                    .await
                    .unwrap()
                })
            })
            .collect::<Vec<_>>();
        for (index, response) in responses.into_iter().enumerate() {
            let response = tokio::time::timeout(Duration::from_secs(20), response)
                .await
                .expect("every concurrent print must settle")
                .unwrap();
            assert_eq!(
                response.status(),
                StatusCode::OK,
                "print {index} lost its observer"
            );
            assert_eq!(
                response
                    .headers()
                    .get("content-type")
                    .and_then(|value| value.to_str().ok()),
                Some("image/png"),
                "print {index} returned no media"
            );
        }
    }

    /// Verify that both streaming and non-streaming requests are properly
    /// serialized through the queue.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn non_streaming_generate_queues_correctly() {
        let (app, _gallery_root) = app_with(MockEngine::ready());

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
        let (state, rx, _gallery_root) =
            durable_test_state(MockEngine::unloaded(load_count, Duration::from_millis(10)));
        spawn_durable_feeder(&state);
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

    #[tokio::test]
    async fn shutdown_closes_idle_chain_job_event_stream() {
        use futures::StreamExt as _;

        let home = tempfile::tempdir().unwrap();
        let _home = MoldHomeGuard::set(home.path());
        let db = mold_db::MetadataDb::open_in_memory().unwrap();
        seed_chain_job(&db, home.path(), "shutdown-events", ChainJobState::Queued);
        let mut state = AppState::for_tests();
        state.metadata_db = Arc::new(Some(db));
        state.chain_jobs = Some(Arc::new(
            crate::chain_job_runner::ChainJobRunnerHandle::inert_for_tests(),
        ));
        let response = app_with_state(state.clone())
            .oneshot(
                Request::get("/api/chain-jobs/shutdown-events/events")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        let mut body = response.into_body().into_data_stream();
        let first = tokio::time::timeout(Duration::from_secs(1), body.next())
            .await
            .expect("snapshot should arrive before shutdown");
        assert!(first.is_some());

        state.events.shutdown();

        let end = tokio::time::timeout(Duration::from_secs(1), body.next())
            .await
            .expect("chain event stream did not close after shutdown");
        assert!(end.is_none());
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
    async fn generation_batch_recovery_routes_require_the_configured_api_key() {
        let keys = std::collections::HashSet::from(["test-key".to_string()]);
        let auth = Some(std::sync::Arc::new(crate::auth::ApiKeySet::new(keys)));
        let app = app_with_auth(auth);

        let lookup = app
            .clone()
            .oneshot(
                Request::get(format!(
                    "/api/generation-batches/by-client/{}",
                    uuid::Uuid::new_v4()
                ))
                .body(Body::empty())
                .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(lookup.status(), StatusCode::UNAUTHORIZED);

        let bulk = app
            .oneshot(json_request(
                "POST",
                "/api/generation-batches/status",
                serde_json::json!({"client_batch_ids": [], "batch_ids": []}),
            ))
            .await
            .unwrap();
        assert_eq!(bulk.status(), StatusCode::UNAUTHORIZED);
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

    /// A two-triangle GLB written straight into the output directory, so the
    /// export tests exercise the real reader and the real writers without
    /// running a 3-D model.
    fn gallery_glb_fixture() -> Vec<u8> {
        let mesh = mold_inference::hunyuan3d::mesh::Mesh {
            vertices: vec![
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [1.0, 1.0, 0.0],
                [0.0, 1.0, 0.0],
            ],
            faces: vec![[0, 1, 2], [0, 2, 3]],
            normals: Some(vec![[0.0, 0.0, 1.0]; 4]),
            uvs: None,
            vertex_colors: None,
        };
        mold_inference::hunyuan3d::glb::write_glb(
            &mesh,
            &mold_inference::hunyuan3d::glb::GlbMaterial::default(),
            None,
        )
        .unwrap()
    }

    fn gallery_export_app(files: &[(&str, Vec<u8>)]) -> (axum::Router, tempfile::TempDir) {
        let output_dir = tempfile::tempdir().unwrap();
        for (name, bytes) in files {
            std::fs::write(output_dir.path().join(name), bytes).unwrap();
        }
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
        (create_router(state), output_dir)
    }

    async fn export_gallery_file(
        app: &axum::Router,
        filename: &str,
        format: &str,
    ) -> axum::response::Response {
        app.clone()
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri(format!("/api/gallery/export/{filename}"))
                    .header(axum::http::header::CONTENT_TYPE, "application/json")
                    .body(Body::from(format!(r#"{{"format":"{format}"}}"#)))
                    .unwrap(),
            )
            .await
            .unwrap()
    }

    /// GLB is the stored form; OBJ, STL and PLY are transcodes of it. Each
    /// one is a DOWNLOAD with its own media type and filename, because the
    /// gallery keeps exactly one file per print.
    #[tokio::test]
    async fn exports_a_gallery_glb_as_obj_stl_and_ply() {
        let (app, _output_dir) =
            gallery_export_app(&[("armchair mesh.glb", gallery_glb_fixture())]);

        let options = json_body(
            app.clone()
                .oneshot(
                    Request::builder()
                        .uri("/api/gallery/export-options")
                        .body(Body::empty())
                        .unwrap(),
                )
                .await
                .unwrap(),
        )
        .await;
        for format in ["glb", "obj", "stl", "ply"] {
            assert!(
                options["formats"]
                    .as_array()
                    .unwrap()
                    .iter()
                    .any(|value| value == format),
                "export options must advertise {format}: {options}"
            );
        }

        for (format, content_type, filename) in [
            ("obj", "model/obj", "armchair_mesh.obj"),
            ("stl", "model/stl", "armchair_mesh.stl"),
            ("ply", "application/x-ply", "armchair_mesh.ply"),
        ] {
            let response = export_gallery_file(&app, "armchair%20mesh.glb", format).await;
            assert_eq!(response.status(), StatusCode::OK, "{format}");
            assert_eq!(
                response.headers()[axum::http::header::CONTENT_TYPE],
                content_type,
                "{format}"
            );
            assert_eq!(
                response.headers()[axum::http::header::CONTENT_DISPOSITION],
                format!("attachment; filename=\"{filename}\""),
                "{format}"
            );
            let bytes = axum::body::to_bytes(response.into_body(), 8 * 1024 * 1024)
                .await
                .unwrap();
            match format {
                "obj" => {
                    let text = String::from_utf8(bytes.to_vec()).unwrap();
                    assert_eq!(
                        text.lines().filter(|line| line.starts_with("v ")).count(),
                        4
                    );
                    assert_eq!(
                        text.lines().filter(|line| line.starts_with("f ")).count(),
                        2
                    );
                }
                "stl" => assert_eq!(bytes.len(), 84 + 2 * 50),
                _ => assert!(bytes.starts_with(b"ply\nformat binary_little_endian 1.0\n")),
            }
        }
    }

    /// Exporting a mesh AS GLB hands back the stored bytes unchanged. Parsing
    /// and rewriting would drop the material and any embedded texture.
    #[tokio::test]
    async fn exporting_a_glb_as_glb_returns_the_stored_bytes() {
        let stored = gallery_glb_fixture();
        let (app, _output_dir) = gallery_export_app(&[("mesh.glb", stored.clone())]);
        let response = export_gallery_file(&app, "mesh.glb", "glb").await;
        assert_eq!(response.status(), StatusCode::OK);
        assert_eq!(
            response.headers()[axum::http::header::CONTENT_TYPE],
            "model/gltf-binary"
        );
        let bytes = axum::body::to_bytes(response.into_body(), 8 * 1024 * 1024)
            .await
            .unwrap();
        assert_eq!(bytes.as_ref(), stored.as_slice());
    }

    async fn export_gallery_file_with(
        app: &axum::Router,
        filename: &str,
        body: serde_json::Value,
    ) -> axum::response::Response {
        app.clone()
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri(format!("/api/gallery/export/{filename}"))
                    .header(axum::http::header::CONTENT_TYPE, "application/json")
                    .body(Body::from(body.to_string()))
                    .unwrap(),
            )
            .await
            .unwrap()
    }

    fn gif_frames(bytes: &[u8]) -> Vec<image::RgbaImage> {
        use image::AnimationDecoder;
        image::codecs::gif::GifDecoder::new(std::io::Cursor::new(bytes))
            .expect("decode GIF")
            .into_frames()
            .map(|frame| frame.expect("decode frame").into_buffer())
            .collect()
    }

    /// An animation of a mesh is its TURNTABLE: the gallery poster's view
    /// rendered around the mesh, encoded by the same encoders a video export
    /// uses, delivered as a download named like every other export. Every
    /// format the build advertises for a mesh answers with its own media
    /// type and a real animation body.
    #[tokio::test]
    async fn exports_a_gallery_glb_as_a_turntable_gif_apng_and_webp() {
        let (app, _output_dir) =
            gallery_export_app(&[("armchair mesh.glb", gallery_glb_fixture())]);

        let advertised = crate::routes::mesh_export_formats();
        for format in [
            mold_core::MeshExportFormat::Gif,
            mold_core::MeshExportFormat::Apng,
        ] {
            assert!(advertised.contains(&format), "{advertised:?}");
        }
        assert_eq!(
            advertised.contains(&mold_core::MeshExportFormat::Webp),
            cfg!(feature = "webp"),
            "WebP is advertised exactly when this build can encode it"
        );
        let options = json_body(
            app.clone()
                .oneshot(
                    Request::builder()
                        .uri("/api/gallery/export-options")
                        .body(Body::empty())
                        .unwrap(),
                )
                .await
                .unwrap(),
        )
        .await;
        let options: Vec<&str> = options["formats"]
            .as_array()
            .unwrap()
            .iter()
            .map(|value| value.as_str().unwrap())
            .collect();
        for format in ["gif", "apng", "glb", "obj", "stl", "ply"] {
            assert!(options.contains(&format), "{options:?}");
        }
        assert_eq!(options.contains(&"webp"), cfg!(feature = "webp"));
        assert_eq!(
            options.iter().filter(|value| **value == "gif").count(),
            1,
            "one entry per format, whatever the source kind: {options:?}"
        );

        let mut formats = vec![
            ("gif", "image/gif", "armchair_mesh.gif"),
            ("apng", "image/apng", "armchair_mesh.png"),
        ];
        if cfg!(feature = "webp") {
            formats.push(("webp", "image/webp", "armchair_mesh.webp"));
        }
        for (format, content_type, filename) in formats {
            let response = export_gallery_file_with(
                &app,
                "armchair%20mesh.glb",
                serde_json::json!({ "format": format, "frames": 8, "max_dimension": 240 }),
            )
            .await;
            assert_eq!(response.status(), StatusCode::OK, "{format}");
            assert_eq!(
                response.headers()[axum::http::header::CONTENT_TYPE],
                content_type,
                "{format}"
            );
            assert_eq!(
                response.headers()[axum::http::header::CONTENT_DISPOSITION],
                format!("attachment; filename=\"{filename}\""),
                "{format}"
            );
            let bytes = axum::body::to_bytes(response.into_body(), 64 * 1024 * 1024)
                .await
                .unwrap();
            assert!(!bytes.is_empty(), "{format}");
            match format {
                "gif" => {
                    assert_eq!(&bytes[..6], b"GIF89a");
                    let frames = gif_frames(&bytes);
                    assert_eq!(frames.len(), 8);
                    assert_eq!((frames[0].width(), frames[0].height()), (240, 240));
                }
                "apng" => {
                    assert_eq!(&bytes[..8], b"\x89PNG\r\n\x1a\n");
                    assert!(bytes.windows(4).any(|chunk| chunk == b"acTL"));
                }
                _ => {
                    assert_eq!(&bytes[..4], b"RIFF");
                    assert_eq!(&bytes[8..12], b"WEBP");
                }
            }
        }
    }

    /// Playback and repeat mean for a turntable exactly what they mean for a
    /// video GIF: a loop is one seamless turn (its last frame is NOT its
    /// first), a bounce that repeats is the sweep plus the interior frames
    /// reversed, and a bounce played once also rests on the first frame.
    /// Bounce outside GIF is the video export's refusal, word for word.
    #[tokio::test]
    async fn turntable_playback_and_repeat_follow_the_video_gif_contract() {
        let (app, _output_dir) = gallery_export_app(&[("mesh.glb", gallery_glb_fixture())]);

        let looped = export_gallery_file_with(
            &app,
            "mesh.glb",
            serde_json::json!({ "format": "gif", "frames": 8, "max_dimension": 240 }),
        )
        .await;
        assert_eq!(looped.status(), StatusCode::OK);
        let frames = gif_frames(
            &axum::body::to_bytes(looped.into_body(), 64 * 1024 * 1024)
                .await
                .unwrap(),
        );
        assert_eq!(frames.len(), 8);
        assert_ne!(
            frames[0], frames[7],
            "a loop stops one step short of the first frame"
        );
        assert_ne!(frames[0], frames[4], "the mesh must actually turn");

        for (repeat, expected) in [("forever", 14), ("once", 15)] {
            let response = export_gallery_file_with(
                &app,
                "mesh.glb",
                serde_json::json!({
                    "format": "gif",
                    "playback": "bounce",
                    "repeat": repeat,
                    "frames": 8,
                    "max_dimension": 240
                }),
            )
            .await;
            assert_eq!(response.status(), StatusCode::OK, "{repeat}");
            let bytes = axum::body::to_bytes(response.into_body(), 64 * 1024 * 1024)
                .await
                .unwrap();
            assert_eq!(gif_frames(&bytes).len(), expected, "bounce + {repeat}");
        }

        let refused = export_gallery_file_with(
            &app,
            "mesh.glb",
            serde_json::json!({ "format": "apng", "playback": "bounce", "frames": 8, "max_dimension": 240 }),
        )
        .await;
        assert_eq!(refused.status(), StatusCode::UNPROCESSABLE_ENTITY);
        let body = json_body(refused).await;
        assert_eq!(
            body["error"].as_str().unwrap(),
            "bounce playback is only supported for GIF exports"
        );
    }

    /// Every turntable bound is a 422 at the door, in the `max_dimension`
    /// message style, and the frame budget is refused before a frame renders.
    #[tokio::test]
    async fn turntable_bounds_are_refused_with_the_bound_named() {
        let (app, _output_dir) = gallery_export_app(&[("mesh.glb", gallery_glb_fixture())]);
        for (body, message) in [
            (
                serde_json::json!({ "format": "gif", "frames": 7 }),
                "frames must be between 8 and 180",
            ),
            (
                serde_json::json!({ "format": "gif", "frames": 181 }),
                "frames must be between 8 and 180",
            ),
            (
                serde_json::json!({ "format": "gif", "fps": 0 }),
                "fps must be between 1 and 30 for a mesh turntable",
            ),
            (
                serde_json::json!({ "format": "gif", "fps": 31 }),
                "fps must be between 1 and 30 for a mesh turntable",
            ),
            (
                serde_json::json!({ "format": "gif", "max_dimension": 239 }),
                "max_dimension must be between 240 and 2160 pixels",
            ),
            (
                serde_json::json!({ "format": "gif", "max_dimension": 2161 }),
                "max_dimension must be between 240 and 2160 pixels",
            ),
        ] {
            let refused = export_gallery_file_with(&app, "mesh.glb", body.clone()).await;
            assert_eq!(refused.status(), StatusCode::UNPROCESSABLE_ENTITY, "{body}");
            let error = json_body(refused).await;
            assert_eq!(error["error"].as_str().unwrap(), message, "{body}");
        }

        let over_budget = export_gallery_file_with(
            &app,
            "mesh.glb",
            serde_json::json!({ "format": "gif", "frames": 180, "max_dimension": 2048 }),
        )
        .await;
        assert_eq!(over_budget.status(), StatusCode::UNPROCESSABLE_ENTITY);
        let error = json_body(over_budget).await;
        let message = error["error"].as_str().unwrap();
        assert!(
            message.contains("export budget") && message.contains("max_dimension"),
            "{message}"
        );
    }

    /// A video takes the animation group only, and a geometry container asked
    /// of it is refused with a sentence naming the other side rather than a
    /// generic "unsupported". (A mesh takes both groups: an animation of a
    /// mesh is its turntable, covered above.)
    #[tokio::test]
    async fn a_mesh_and_a_video_refuse_each_other_s_export_formats() {
        let mp4 = base64::engine::general_purpose::STANDARD
            .decode(include_str!("testdata/audio_muxed_final_mp4.b64").trim())
            .unwrap();
        let (app, _output_dir) =
            gallery_export_app(&[("clip.mp4", mp4), ("still.png", b"not a mesh".to_vec())]);

        let refused = export_gallery_file(&app, "clip.mp4", "stl").await;
        assert_eq!(refused.status(), StatusCode::UNPROCESSABLE_ENTITY);
        let body = json_body(refused).await;
        assert!(
            body["error"].as_str().unwrap().contains("mesh format"),
            "{body}"
        );

        let refused = export_gallery_file(&app, "still.png", "stl").await;
        assert_eq!(refused.status(), StatusCode::UNPROCESSABLE_ENTITY);
        let body = json_body(refused).await;
        assert!(
            body["error"]
                .as_str()
                .unwrap()
                .contains("only MP4 gallery videos and GLB meshes"),
            "{body}"
        );

        let missing = export_gallery_file(&app, "absent.glb", "obj").await;
        assert_eq!(missing.status(), StatusCode::NOT_FOUND);
    }

    /// A `.glb` that mold did not write is refused by NAME. The user is
    /// looking at a file in their own gallery, so "this reader does not cover
    /// that layout" has to be distinguishable from "that is not a mesh".
    #[tokio::test]
    async fn exporting_a_foreign_glb_names_what_is_unsupported() {
        let (app, _output_dir) =
            gallery_export_app(&[("foreign.glb", b"glTF not really".to_vec())]);
        let refused = export_gallery_file(&app, "foreign.glb", "stl").await;
        assert_eq!(refused.status(), StatusCode::UNPROCESSABLE_ENTITY);
        let body = json_body(refused).await;
        assert!(
            body["error"].as_str().unwrap().contains("cannot export"),
            "{body}"
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
            video_only: None,
            attention_path: None,
            int8_arm: None,
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
            mesh: None,
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
            pipeline_requested: None,
            duration_prediction_requested: None,
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
            id_image_name: None,
            id_image_sha256: None,
            id_weight: None,
            id_start_step: None,
            id_image_names: None,
            id_image_sha256s: None,
            true_cfg: None,
            cfg_start_step: None,
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
            .clone()
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
        let _env = env_lock();

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
        let _lock = env_lock();
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
        let body = axum::body::to_bytes(resp.into_body(), 8 * 1024 * 1024)
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
            video_only: None,
            attention_path: None,
            int8_arm: None,
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
            mesh: None,
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
            pipeline_requested: None,
            duration_prediction_requested: None,
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
            id_image_name: None,
            id_image_sha256: None,
            id_weight: None,
            id_start_step: None,
            id_image_names: None,
            id_image_sha256s: None,
            true_cfg: None,
            cfg_start_step: None,
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
    async fn get_api_events_starts_with_instance_authority() {
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

        let expected_instance_id = state.instance_id.as_str();
        let mut body = res.into_body().into_data_stream();
        let bytes = tokio::time::timeout(std::time::Duration::from_secs(1), body.next())
            .await
            .expect("authority frame must be immediately available")
            .expect("authority frame stream must remain open")
            .expect("authority frame body must be readable");
        let frame = String::from_utf8(bytes.to_vec()).unwrap();
        assert!(frame.starts_with("event: authority\n"), "frame: {frame:?}");
        assert!(
            frame.contains(&format!(r#""instance_id":"{expected_instance_id}""#)),
            "frame: {frame:?}"
        );
    }

    #[tokio::test]
    async fn get_api_events_preserves_published_server_event_frames() {
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

        let mut body = res.into_body().into_data_stream();
        let authority = tokio::time::timeout(std::time::Duration::from_secs(1), body.next())
            .await
            .expect("authority frame must be immediately available")
            .expect("authority frame stream must remain open")
            .expect("authority frame body must be readable");
        assert!(String::from_utf8_lossy(&authority).starts_with("event: authority\n"));

        // Publish only after consuming the opening authority frame. This
        // deterministically proves the additive frame did not change the
        // existing `event` name or ServerEvent JSON payload.
        state.job_registry.register("j1", "flux-dev:q4");

        let bytes = tokio::time::timeout(std::time::Duration::from_secs(1), body.next())
            .await
            .expect("published event must be immediately available")
            .expect("event stream must remain open")
            .expect("event frame body must be readable");
        let frame = String::from_utf8(bytes.to_vec()).unwrap();
        assert_eq!(
            frame,
            "event: event\ndata: {\"type\":\"job_queued\",\"id\":\"j1\",\"model\":\"flux-dev:q4\"}\n\n"
        );
    }

    #[tokio::test]
    async fn shutdown_closes_idle_server_subscription_streams() {
        use futures::StreamExt as _;

        for path in [
            "/api/events",
            "/api/downloads/stream",
            "/api/resources/stream",
        ] {
            let state = AppState::empty(
                mold_core::Config::default(),
                crate::state::QueueHandle::new(tokio::sync::mpsc::channel(1).0),
                AppState::empty_gpu_pool_for_test(),
                200,
            );
            let response = app_with_state(state.clone())
                .oneshot(Request::get(path).body(Body::empty()).unwrap())
                .await
                .unwrap();
            assert_eq!(response.status(), StatusCode::OK, "path: {path}");
            let mut body = response.into_body().into_data_stream();

            if path != "/api/resources/stream" {
                let initial = tokio::time::timeout(Duration::from_secs(1), body.next())
                    .await
                    .unwrap_or_else(|_| panic!("{path} did not emit its initial frame"));
                assert!(initial.is_some(), "{path} closed before becoming idle");
            }

            let (waiting_tx, waiting_rx) = tokio::sync::oneshot::channel();
            let waiter = tokio::spawn(async move {
                let _ = waiting_tx.send(());
                body.next().await
            });
            waiting_rx.await.unwrap();
            tokio::task::yield_now().await;
            state.events.shutdown();

            let end = tokio::time::timeout(Duration::from_secs(1), waiter)
                .await
                .unwrap_or_else(|_| panic!("{path} did not close after shutdown"))
                .expect("subscription waiter should not panic");
            assert!(end.is_none(), "{path} yielded another frame after shutdown");
        }
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
        let _lock = env_lock();
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
        let _lock = env_lock();
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
        let (app, _gallery_root) = app_with_worker_pool(MockEngine::ready(), &[1]);
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
    async fn expand_and_remix_answer_a_prompt_ignored_family_without_a_backend() {
        let state = AppState::for_tests();
        {
            // An unreachable API backend: any completion attempt fails loudly.
            let mut config = state.config.write().await;
            config.expand.backend = "http://127.0.0.1:9".to_string();
        }
        let app = app_with_state(state);
        let advice = mold_core::ignored_prompt_advice("hunyuan3d").unwrap();

        let response = app
            .clone()
            .oneshot(
                Request::post("/api/expand")
                    .header("content-type", "application/json")
                    .body(Body::from(
                        serde_json::json!({
                            "prompt": "a dining chair",
                            "model_family": "hunyuan3d",
                            "variations": 3,
                            "context": { "model": "hunyuan3d-mini-turbo:fp16" }
                        })
                        .to_string(),
                    ))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::OK);
        let body = json_body(response).await;
        assert_eq!(body["original"], "a dining chair");
        assert_eq!(body["expanded"], serde_json::json!([advice.text()]));

        let response = app
            .clone()
            .oneshot(
                Request::post("/api/remix")
                    .header("content-type", "application/json")
                    .body(Body::from(
                        serde_json::json!({
                            "source_prompt": "a dining chair",
                            "model_family": "hunyuan3d",
                            "variations": 3,
                            "dimensions": ["movement"]
                        })
                        .to_string(),
                    ))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::OK);
        let body = json_body(response).await;
        assert_eq!(body["task"], "text-to-image");
        assert_eq!(
            body["variants"],
            serde_json::json!([{ "prompt": advice.text(), "dimensions": [] }])
        );

        // Every other family still needs the backend, and says so.
        let response = app
            .oneshot(
                Request::post("/api/expand")
                    .header("content-type", "application/json")
                    .body(Body::from(
                        serde_json::json!({ "prompt": "a dining chair", "model_family": "flux" })
                            .to_string(),
                    ))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::INTERNAL_SERVER_ERROR);
        let body = json_body(response).await;
        let message = body["error"].as_str().unwrap_or_default();
        assert!(message.contains("prompt expansion failed"), "{body}");
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
    async fn post_api_downloads_accepts_every_cold_start_starter() {
        let state = AppState::empty(
            mold_core::Config::default(),
            crate::state::QueueHandle::new(tokio::sync::mpsc::channel(1).0),
            AppState::empty_gpu_pool_for_test(),
            200,
        );
        let app = app_with_state(state.clone());

        for model in ["flux2-klein:q4", "z-image-turbo:q8", "sdxl-base:fp16"] {
            let response = app
                .clone()
                .oneshot(
                    Request::post("/api/downloads")
                        .header("content-type", "application/json")
                        .body(Body::from(
                            serde_json::json!({ "model": model }).to_string(),
                        ))
                        .unwrap(),
                )
                .await
                .unwrap();
            assert_eq!(response.status(), StatusCode::OK, "starter {model}");
        }

        let listing = state.downloads.listing().await;
        assert_eq!(listing.queued.len(), 3);
    }

    /// Point `MOLD_HOME` at a throwaway root through the SHARED env guard.
    ///
    /// License acceptance is per Mold data root, so these tests write to it —
    /// and `MOLD_HOME` is process-global, so holding anything other than
    /// `env_lock()` lets them race the gallery tests that derive output dirs
    /// from the same variable.
    fn license_home() -> (tempfile::TempDir, MoldHomeGuard) {
        let home = tempfile::tempdir().unwrap();
        let guard = MoldHomeGuard::set(home.path());
        (home, guard)
    }

    fn licensed_state() -> AppState {
        AppState::empty(
            mold_core::Config::default(),
            crate::state::QueueHandle::new(tokio::sync::mpsc::channel(1).0),
            AppState::empty_gpu_pool_for_test(),
            200,
        )
    }

    fn download_request(body: serde_json::Value) -> Request<Body> {
        Request::builder()
            .method("POST")
            .uri("/api/downloads")
            .header("content-type", "application/json")
            .body(Body::from(body.to_string()))
            .unwrap()
    }

    /// The exact terms this build pins — what an honest client would have
    /// read from `GET /api/licenses` and displayed before accepting.
    fn accepted_terms() -> serde_json::Value {
        let license = &mold_core::license_acceptance::INSIGHTFACE_ANTELOPEV2;
        serde_json::json!({
            "id": license.id,
            "url": license.url,
            "sha256": license.sha256,
        })
    }

    /// A client on a different Mold release resolves the same id to a
    /// different pinned revision. Recording ITS consent against OUR text would
    /// store agreement to a document the user never read, so the server
    /// refuses and shows its own terms instead.
    #[tokio::test]
    async fn post_api_downloads_rejects_terms_the_server_does_not_pin() {
        let (home, _guard) = license_home();
        let state = licensed_state();
        let app = app_with_state(state.clone());

        let stale = serde_json::json!({
            "id": "insightface-antelopev2",
            "url": "https://raw.githubusercontent.com/deepinsight/insightface/0000000000000000000000000000000000000000/README.md",
            "sha256": "0000000000000000000000000000000000000000000000000000000000000000",
        });
        let res = app
            .oneshot(download_request(serde_json::json!({
                "model": "pulid-flux",
                "accept_licenses": [stale],
            })))
            .await
            .unwrap();

        assert_eq!(res.status(), StatusCode::CONFLICT);
        let body = json_body(res).await;
        assert_eq!(body["code"], mold_core::LICENSE_TERMS_MISMATCH);
        // The server's OWN terms ride the refusal so the client can display
        // them and retry without a second round trip.
        let ours = &mold_core::license_acceptance::INSIGHTFACE_ANTELOPEV2;
        assert_eq!(body["license"]["url"], ours.url);
        assert_eq!(body["license"]["sha256"], ours.sha256);
        assert_eq!(body["license"]["canonical"], ours.canonical);

        assert!(!mold_core::license_acceptance::is_accepted(
            home.path(),
            ours
        ));
        assert!(state.downloads.listing().await.queued.is_empty());
    }

    /// A right id with a wrong digest is still a mismatch — the URL alone is
    /// not the identity.
    #[tokio::test]
    async fn post_api_downloads_rejects_a_matching_url_with_a_different_digest() {
        let (home, _guard) = license_home();
        let state = licensed_state();
        let app = app_with_state(state.clone());

        let ours = &mold_core::license_acceptance::INSIGHTFACE_ANTELOPEV2;
        let res = app
            .oneshot(download_request(serde_json::json!({
                "model": "pulid-flux",
                "accept_licenses": [{
                    "id": ours.id,
                    "url": ours.url,
                    "sha256": "0000000000000000000000000000000000000000000000000000000000000000",
                }],
            })))
            .await
            .unwrap();

        assert_eq!(res.status(), StatusCode::CONFLICT);
        assert!(!mold_core::license_acceptance::is_accepted(
            home.path(),
            ours
        ));
        assert!(state.downloads.listing().await.queued.is_empty());
    }

    #[tokio::test]
    async fn get_api_licenses_reflects_this_servers_acceptance_state() {
        let (home, _guard) = license_home();
        let app = app_with_state(licensed_state());

        let res = app
            .clone()
            .oneshot(
                Request::builder()
                    .uri("/api/licenses")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(res.status(), StatusCode::OK);
        let listing = json_body(res).await;
        let entry = listing["licenses"]
            .as_array()
            .unwrap()
            .iter()
            .find(|entry| entry["id"] == "insightface-antelopev2")
            .expect("the antelopev2 license is listed");
        assert_eq!(entry["accepted"], serde_json::Value::Bool(false));
        assert_eq!(
            entry["url"],
            mold_core::license_acceptance::INSIGHTFACE_ANTELOPEV2.url
        );
        assert_eq!(
            entry["canonical"],
            mold_core::license_acceptance::INSIGHTFACE_ANTELOPEV2.canonical
        );
        assert!(entry["required_by"]
            .as_array()
            .unwrap()
            .iter()
            .any(|name| name == "pulid-flux"));

        mold_core::license_acceptance::record_acceptance(
            home.path(),
            &mold_core::license_acceptance::INSIGHTFACE_ANTELOPEV2,
        )
        .unwrap();

        let res = app
            .oneshot(
                Request::builder()
                    .uri("/api/licenses")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        let listing = json_body(res).await;
        let entry = listing["licenses"]
            .as_array()
            .unwrap()
            .iter()
            .find(|entry| entry["id"] == "insightface-antelopev2")
            .unwrap();
        assert_eq!(entry["accepted"], serde_json::Value::Bool(true));
    }

    /// The bug this endpoint exists to fix: a client that recorded acceptance
    /// in ITS OWN root told the wrong machine, so the server must refuse —
    /// and refuse in a shape a UI can act on rather than prose alone.
    #[tokio::test]
    async fn post_api_downloads_refuses_a_license_gated_model_structurally() {
        let (_home, _guard) = license_home();
        let app = app_with_state(licensed_state());

        let res = app
            .oneshot(download_request(
                serde_json::json!({ "model": "pulid-flux" }),
            ))
            .await
            .unwrap();
        assert_eq!(res.status(), StatusCode::FORBIDDEN);
        let body = json_body(res).await;
        assert_eq!(body["code"], mold_core::LICENSE_NOT_ACCEPTED);
        assert_eq!(body["license"]["id"], "insightface-antelopev2");
        assert!(body["license"]["summary"]
            .as_str()
            .unwrap()
            .contains("non-commercial research"));
        assert_eq!(
            body["license"]["url"],
            mold_core::license_acceptance::INSIGHTFACE_ANTELOPEV2.url
        );
        assert!(body["error"]
            .as_str()
            .unwrap()
            .contains("--accept-license insightface-antelopev2"));
    }

    #[tokio::test]
    async fn post_api_downloads_accept_licenses_records_on_the_server_and_proceeds() {
        let (home, _guard) = license_home();
        let state = licensed_state();
        let app = app_with_state(state.clone());

        let res = app
            .oneshot(download_request(serde_json::json!({
                "model": "pulid-flux",
                "accept_licenses": [accepted_terms()],
            })))
            .await
            .unwrap();
        assert_eq!(res.status(), StatusCode::OK);

        // Recorded in the SERVER's root, not the caller's.
        assert!(mold_core::license_acceptance::is_accepted(
            home.path(),
            &mold_core::license_acceptance::INSIGHTFACE_ANTELOPEV2
        ));
        let listing = state.downloads.listing().await;
        assert_eq!(listing.queued.len(), 1);
        assert_eq!(listing.queued[0].model, "pulid-flux");
    }

    #[tokio::test]
    async fn post_api_downloads_unknown_license_id_400_and_writes_nothing() {
        let (home, _guard) = license_home();
        let state = licensed_state();
        let app = app_with_state(state.clone());

        let res = app
            .oneshot(download_request(serde_json::json!({
                "model": "pulid-flux",
                "accept_licenses": [
                    accepted_terms(),
                    { "id": "not-a-license", "url": "https://example.invalid/x", "sha256": "0" },
                ],
            })))
            .await
            .unwrap();
        assert_eq!(res.status(), StatusCode::BAD_REQUEST);
        let body = json_body(res).await;
        assert_eq!(body["code"], "UNKNOWN_LICENSE");

        // Resolution happens for the whole list before the first write, so
        // the valid id in the same request must NOT have been recorded.
        assert!(!mold_core::license_acceptance::is_accepted(
            home.path(),
            &mold_core::license_acceptance::INSIGHTFACE_ANTELOPEV2
        ));
        assert!(state.downloads.listing().await.queued.is_empty());
    }

    /// An unrestricted model must be completely unaffected by the gate.
    #[tokio::test]
    async fn post_api_downloads_unlicensed_model_is_never_gated() {
        let (_home, _guard) = license_home();
        let state = licensed_state();
        let app = app_with_state(state.clone());

        let res = app
            .oneshot(download_request(
                serde_json::json!({ "model": "flux-schnell:q4" }),
            ))
            .await
            .unwrap();
        assert_eq!(res.status(), StatusCode::OK);
        assert_eq!(state.downloads.listing().await.queued.len(), 1);
    }

    #[tokio::test]
    async fn capabilities_advertise_license_support() {
        let (_home, _guard) = license_home();
        let app = app_with_state(licensed_state());
        let res = app
            .oneshot(
                Request::builder()
                    .uri("/api/capabilities")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(res.status(), StatusCode::OK);
        assert_eq!(
            json_body(res).await["licenses"],
            serde_json::Value::Bool(true)
        );
    }

    /// The terms this build pins for the Hunyuan3D 2.0 shape checkpoints.
    fn hunyuan3d_terms() -> serde_json::Value {
        let license = &mold_core::license_acceptance::TENCENT_HUNYUAN3D_2_0;
        serde_json::json!({
            "id": license.id,
            "url": license.url,
            "sha256": license.sha256,
        })
    }

    fn placement_preview_request(model: &str) -> Request<Body> {
        // The canvas is the model's own conditioning size, which is what a
        // real client sends from the manifest defaults.
        let request: serde_json::Value =
            serde_json::from_str(&generate_body_for_model("a chair", model, 512, 512)).unwrap();
        Request::builder()
            .method("POST")
            .uri("/api/generate/placement-preview")
            .header("content-type", "application/json")
            .body(Body::from(
                serde_json::json!({ "request": request, "copies": 1 }).to_string(),
            ))
            .unwrap()
    }

    /// A gated MAIN checkpoint is never a dependency-ladder artifact, so
    /// before this it reached no client at all: the preview refused it with
    /// `missing_components` carrying no terms, and the pull offer POSTed
    /// `/api/downloads` with no acceptances and got a 403 nobody rendered.
    #[tokio::test]
    async fn a_placement_preview_carries_the_requested_models_own_outstanding_terms() {
        let (_home, _guard) = license_home();
        let app = app_with_state(licensed_state());
        let res = app
            .oneshot(placement_preview_request("hunyuan3d:fp16"))
            .await
            .unwrap();
        assert_eq!(res.status(), StatusCode::OK);
        let body = json_body(res).await;
        let pending = body["pending_downloads"]
            .as_array()
            .expect("pending_downloads array");
        let row = pending
            .iter()
            .find(|row| !row["licenses"].as_array().is_none_or(|l| l.is_empty()))
            .expect("a row carrying outstanding terms");
        // The bundle the client posts back, not one of its files.
        assert_eq!(row["install_model"], "hunyuan3d:fp16");
        assert_eq!(row["licenses"][0]["id"], "tencent-hunyuan3d-2.0");
        assert_eq!(
            row["licenses"][0]["url"],
            mold_core::license_acceptance::TENCENT_HUNYUAN3D_2_0.url
        );
        // The territorial exclusion is the reason this is a gate and not a
        // footnote, so it must survive to the surface that renders it.
        assert!(row["licenses"][0]["summary"]
            .as_str()
            .unwrap()
            .contains("European Union"));
        assert!(row["bytes"].as_u64().unwrap() > 0);
        assert!(!row["repo"].as_str().unwrap().is_empty());
    }

    /// The accept-then-retry loop must terminate: once recorded, the same
    /// preview stops asking, or the dialog re-fires forever.
    #[tokio::test]
    async fn an_accepted_model_stops_carrying_terms_in_its_preview() {
        let (home, _guard) = license_home();
        mold_core::license_acceptance::record_acceptance(
            home.path(),
            &mold_core::license_acceptance::TENCENT_HUNYUAN3D_2_0,
        )
        .unwrap();
        let app = app_with_state(licensed_state());
        let res = app
            .oneshot(placement_preview_request("hunyuan3d:fp16"))
            .await
            .unwrap();
        let body = json_body(res).await;
        for row in body["pending_downloads"].as_array().unwrap_or(&Vec::new()) {
            assert!(
                row["licenses"].as_array().is_none_or(|l| l.is_empty()),
                "an accepted model must carry no outstanding terms: {row}"
            );
        }
    }

    /// Guards against the decoration firing for every model.
    #[tokio::test]
    async fn an_ungated_model_never_carries_terms_in_its_preview() {
        let (_home, _guard) = license_home();
        let app = app_with_state(licensed_state());
        let res = app
            .oneshot(placement_preview_request("flux-schnell:q4"))
            .await
            .unwrap();
        let body = json_body(res).await;
        for row in body["pending_downloads"].as_array().unwrap_or(&Vec::new()) {
            assert!(
                row["licenses"].as_array().is_none_or(|l| l.is_empty()),
                "an ungated model must never be license-gated: {row}"
            );
        }
    }

    /// The preview seam and the download seam must agree: whatever the preview
    /// asked consent for is exactly what the enqueue accepts.
    #[tokio::test]
    async fn accepting_the_previewed_terms_lets_the_same_model_enqueue() {
        let (_home, _guard) = license_home();
        let state = licensed_state();

        let refused = app_with_state(state.clone())
            .oneshot(download_request(
                serde_json::json!({ "model": "hunyuan3d:fp16" }),
            ))
            .await
            .unwrap();
        assert_eq!(refused.status(), StatusCode::FORBIDDEN);
        let body = json_body(refused).await;
        assert_eq!(body["code"], mold_core::LICENSE_NOT_ACCEPTED);
        assert_eq!(body["license"]["id"], "tencent-hunyuan3d-2.0");

        let accepted = app_with_state(state.clone())
            .oneshot(download_request(serde_json::json!({
                "model": "hunyuan3d:fp16",
                "accept_licenses": [hunyuan3d_terms()],
            })))
            .await
            .unwrap();
        assert_eq!(accepted.status(), StatusCode::OK);
    }

    /// Consent and acquisition are different acts. Accepting must record
    /// without starting a multi-gigabyte transfer.
    #[tokio::test]
    async fn post_api_licenses_accept_records_without_enqueueing_a_download() {
        let (home, _guard) = license_home();
        let app = app_with_state(licensed_state());
        let res = app
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/api/licenses/accept")
                    .header("content-type", "application/json")
                    .body(Body::from(
                        serde_json::json!({ "accept_licenses": [hunyuan3d_terms()] }).to_string(),
                    ))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(res.status(), StatusCode::OK);

        // It wrote through, and answered with the refreshed state so a caller
        // needs no second round trip.
        assert!(mold_core::license_acceptance::is_accepted(
            home.path(),
            &mold_core::license_acceptance::TENCENT_HUNYUAN3D_2_0
        ));
        let body = json_body(res).await;
        let row = body["licenses"]
            .as_array()
            .unwrap()
            .iter()
            .find(|row| row["id"] == "tencent-hunyuan3d-2.0")
            .expect("the accepted license");
        assert_eq!(row["accepted"], serde_json::Value::Bool(true));
        assert!(
            !row["required_by"].as_array().unwrap().is_empty(),
            "a license nothing requires can be read on every surface and accepted on none"
        );
    }

    #[tokio::test]
    async fn post_api_licenses_accept_rejects_terms_the_server_does_not_pin() {
        let (home, _guard) = license_home();
        let app = app_with_state(licensed_state());
        let mut stale = hunyuan3d_terms();
        stale["sha256"] =
            serde_json::json!("0000000000000000000000000000000000000000000000000000000000000000");
        let res = app
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/api/licenses/accept")
                    .header("content-type", "application/json")
                    .body(Body::from(
                        serde_json::json!({ "accept_licenses": [stale] }).to_string(),
                    ))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(res.status(), StatusCode::CONFLICT);
        assert_eq!(
            json_body(res).await["code"],
            mold_core::LICENSE_TERMS_MISMATCH
        );
        assert!(!mold_core::license_acceptance::is_accepted(
            home.path(),
            &mold_core::license_acceptance::TENCENT_HUNYUAN3D_2_0
        ));
    }

    /// A catalog row can name a gated built-in — `hf:tencent/Hunyuan3D-2` and
    /// `hunyuan3d:fp16` are the same weights reached two ways. The catalog
    /// route enqueues by manifest name and never checked acceptance, so that
    /// second way bypassed consent entirely and failed later in the worker.
    #[test]
    fn a_catalog_row_for_a_gated_built_in_is_refused_with_its_terms() {
        let (home, _guard) = license_home();
        let refusal = crate::catalog_api::catalog_license_refusal("tencent/Hunyuan3D-2")
            .expect("a gated built-in must be refused through the catalog route too");
        assert_eq!(refusal.status(), StatusCode::FORBIDDEN);
        assert_eq!(refusal.code, mold_core::LICENSE_NOT_ACCEPTED);
        assert_eq!(
            refusal.license.as_ref().unwrap().id,
            "tencent-hunyuan3d-2.0"
        );

        // …and stops once consent is on record for this root.
        mold_core::license_acceptance::record_acceptance(
            home.path(),
            &mold_core::license_acceptance::TENCENT_HUNYUAN3D_2_0,
        )
        .unwrap();
        assert!(crate::catalog_api::catalog_license_refusal("tencent/Hunyuan3D-2").is_none());
    }

    /// Guards against the catalog gate firing for everything.
    #[test]
    fn a_catalog_row_for_an_ungated_model_is_never_refused() {
        let (_home, _guard) = license_home();
        assert!(
            crate::catalog_api::catalog_license_refusal("black-forest-labs/FLUX.1-schnell")
                .is_none()
        );
        assert!(crate::catalog_api::catalog_license_refusal("not-a-repo/at-all").is_none());
    }

    /// A refusal is the one download failure a client CAN resolve. Reporting
    /// it as a 500 with no payload told the user it was the server's fault and
    /// stripped the terms a UI needs to offer acceptance.
    #[test]
    fn a_license_refusal_maps_to_a_structured_403_not_a_500() {
        let license = &mold_core::license_acceptance::TENCENT_HUNYUAN3D_2_0;
        let refusal = mold_core::download::DownloadError::LicenseNotAccepted {
            license_id: license.id.to_string(),
            message: mold_core::license_acceptance::acceptance_required_message(
                "hunyuan3d:fp16",
                license,
            ),
        };
        let mapped = crate::routes::ApiError::from_download_error("hunyuan3d:fp16", &refusal);
        assert_eq!(mapped.status(), StatusCode::FORBIDDEN);
        assert_eq!(mapped.code, mold_core::LICENSE_NOT_ACCEPTED);
        assert_eq!(mapped.license.as_ref().unwrap().url, license.url);
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
        let _lock = env_lock();
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
                reclaimable_zfs_arc: None,
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

        let _lock = env_lock();
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
                reclaimable_zfs_arc: None,
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
        let _lock = env_lock();
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
        let _lock = env_lock();
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
        // The advertised cap is the model's routing clip size — what ONE
        // generation renders — not the family's 481-frame single-request
        // ceiling at 24 fps, which stays derivable from the duration budget.
        assert_eq!(limits["frames_per_clip_cap"], 97);
        assert_eq!(limits["fps"], 24);
        assert_eq!(limits["frames_per_clip_runtime_seconds"], 20);
        assert_eq!(limits["max_stages"], 16);
        assert_eq!(limits["max_total_frames"], 97 * 16);
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
        let _env = crate::test_support::hermetic_store_env();
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
        // An opaque catalog LTX-2 checkpoint takes the family's routing clip
        // size, not the 481-frame duration budget at 24 fps.
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
            mesh: None,
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
            mesh: None,
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
            mesh: None,
            request_warnings: Vec::new(),
            audio: None,
            images: Vec::new(),
            video: Some(mold_core::VideoData {
                video_only: None,
                attention_path: None,
                int8_arm: None,
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
    async fn gallery_mutations_bulk_titles_and_replays_by_operation_id() {
        let dir = tempfile::tempdir().unwrap();
        let (state, db) = organized_state(dir.path());
        let filenames: Vec<String> = (0..30).map(|index| format!("{index}.png")).collect();
        for filename in &filenames {
            seed_print(&db, dir.path(), filename, None);
        }
        let app = app_with_state(state);
        let operation_id = uuid::Uuid::new_v4().to_string();
        let request = serde_json::json!({
            "operation_id": operation_id,
            "filenames": filenames,
            "titles": (0..30).map(|index| serde_json::json!({
                "filename": format!("{index}.png"),
                "title": format!("Moon {index}")
            })).collect::<Vec<_>>(),
            "favorite": true,
            "add_tags": ["bulk"],
            "add_to_collection": {"name": "Moon studies"}
        });
        let first = app
            .clone()
            .oneshot(json_request(
                "POST",
                "/api/gallery/mutations",
                request.clone(),
            ))
            .await
            .unwrap();
        assert_eq!(first.status(), StatusCode::OK);
        let first_body = json_body(first).await;
        assert_eq!(first_body["changed"], 30);

        let retry = app
            .clone()
            .oneshot(json_request(
                "POST",
                "/api/gallery/mutations",
                request.clone(),
            ))
            .await
            .unwrap();
        assert_eq!(retry.status(), StatusCode::OK);
        assert_eq!(json_body(retry).await, first_body);

        let rows = gallery_rows(&app, "/api/gallery").await;
        assert_eq!(rows.len(), 30);
        assert!(rows.iter().all(|row| row["favorite"] == true));
        assert!(rows
            .iter()
            .all(|row| row["tags"] == serde_json::json!(["bulk"])));
        let collections = app
            .clone()
            .oneshot(empty_request("GET", "/api/gallery/collections"))
            .await
            .unwrap();
        let collections = json_body(collections).await;
        assert_eq!(
            collections.as_array().unwrap().len(),
            1,
            "retry creates no duplicate"
        );

        let mut changed = request;
        changed["favorite"] = serde_json::json!(false);
        let conflict = app
            .clone()
            .oneshot(json_request("POST", "/api/gallery/mutations", changed))
            .await
            .unwrap();
        assert_eq!(conflict.status(), StatusCode::CONFLICT);

        // Conflicting concurrent payloads serialize at the DB transaction:
        // exactly one applies and the other observes its receipt.
        let race_id = uuid::Uuid::new_v4().to_string();
        let race_a = serde_json::json!({
            "operation_id": race_id.clone(),
            "filenames": ["0.png"],
            "add_tags": ["race-a"]
        });
        let race_b = serde_json::json!({
            "operation_id": race_id,
            "filenames": ["0.png"],
            "add_tags": ["race-b"]
        });
        let (a, b) = tokio::join!(
            app.clone()
                .oneshot(json_request("POST", "/api/gallery/mutations", race_a)),
            app.clone()
                .oneshot(json_request("POST", "/api/gallery/mutations", race_b)),
        );
        let mut statuses = [a.unwrap().status(), b.unwrap().status()];
        statuses.sort_by_key(|status| status.as_u16());
        assert_eq!(statuses, [StatusCode::OK, StatusCode::CONFLICT]);

        // Collection creation shares the print-validation transaction and is
        // rolled back when any selected filename is missing.
        let invalid = app
            .clone()
            .oneshot(json_request(
                "POST",
                "/api/gallery/mutations",
                serde_json::json!({
                    "operation_id": uuid::Uuid::new_v4().to_string(),
                    "filenames": ["ghost.png"],
                    "add_to_collection": {"name": "Must roll back"}
                }),
            ))
            .await
            .unwrap();
        assert_eq!(invalid.status(), StatusCode::NOT_FOUND);
        let collections = json_body(
            app.oneshot(empty_request("GET", "/api/gallery/collections"))
                .await
                .unwrap(),
        )
        .await;
        assert_eq!(collections.as_array().unwrap().len(), 1);
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
        assert!(created.get("hidden").is_none());
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

        // Rename + cover + hide from the default Library.
        let resp = app
            .clone()
            .oneshot(json_request(
                "PATCH",
                &format!("/api/gallery/collections/{id}"),
                serde_json::json!({"name": "Gargamel", "cover_filename": "b.png", "hidden": true}),
            ))
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
        let updated = json_body(resp).await;
        assert_eq!(updated["slug"], "gargamel");
        assert_eq!(updated["cover_filename"], "b.png");
        assert_eq!(updated["hidden"], true);
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
        assert_eq!(listed[0]["hidden"], true);
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
        assert_eq!(caps["gallery"]["bulk_mutations"], true);
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
        seed_print(&db, dir.path(), "bulk-binned.png", None);
        seed_print(&db, dir.path(), "bulk-live.png", None);
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

        let resp = app
            .clone()
            .oneshot(empty_request(
                "DELETE",
                "/api/gallery/image/bulk-binned.png",
            ))
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::NO_CONTENT);
        drain_events(&mut events);
        let resp = app
            .clone()
            .oneshot(json_request(
                "POST",
                "/api/gallery/trash/delete-forever",
                serde_json::json!({"filenames": ["bulk-binned.png", "bulk-live.png"]}),
            ))
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::NO_CONTENT);
        assert!(mdb.get(dir.path(), "bulk-binned.png").unwrap().is_none());
        assert!(mdb.get(dir.path(), "bulk-live.png").unwrap().is_none());
        assert_eq!(
            drain_events(&mut events)
                .into_iter()
                .filter(|event| matches!(event, mold_core::ServerEvent::GalleryRemoved { .. }))
                .count(),
            2
        );
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
        let body = axum::body::to_bytes(resp.into_body(), 8 * 1024 * 1024)
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
        assert!(crate::routes::validate_generate_request(
            &req,
            None,
            mold_core::ReferenceForm::Admitted
        )
        .is_err());
        req.title = Some("Smurf village".into());
        assert!(crate::routes::validate_generate_request(
            &req,
            None,
            mold_core::ReferenceForm::Admitted
        )
        .is_ok());
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

    // ── Durable-admission readiness ───────────────────────────────────────────
    //
    // Appended as one block on purpose. A semantic conflict in a Rust test file
    // compiles and passes on both sides of a rebase, so new coverage goes at the
    // end rather than interleaved into existing modules, where a three-way merge
    // could delete it with no signal.
    //
    // These drive `direct_durable_admission` directly rather than POSTing to
    // `/api/generate`: the attached fallback they assert on is exactly the path
    // that then runs a real generation, so a route-level test would block on a
    // worker instead of measuring the gate.

    fn readiness_request() -> mold_core::GenerateRequest {
        serde_json::from_str(&generate_body("a cat", 64, 64)).unwrap()
    }

    /// ONE admission path. A host that cannot admit durably refuses
    /// generation on every route rather than silently running a second,
    /// non-durable pipeline. Scheduler V2 is a conjunct like any other: the
    /// durable feeder's restart safety is only real when the authoritative
    /// dispatcher owns the row.
    #[tokio::test(flavor = "current_thread")]
    async fn a_non_authoritative_host_refuses_every_generation_route() {
        for mode in [
            crate::dispatch_mode::DispatchMode::Legacy,
            crate::dispatch_mode::DispatchMode::Observe,
        ] {
            let root = tempfile::tempdir().unwrap();
            let db = Arc::new(Some(mold_db::MetadataDb::open_in_memory().unwrap()));
            let (mut state, _rx) = durable_state(db, root.path());
            install_dispatch_mode(&mut state, mode);

            let mut request = readiness_request();
            let Err(refusal) = crate::routes::direct_durable_admission(&state, &mut request).await
            else {
                panic!("a non-authoritative host admits nothing");
            };
            assert_eq!(refusal.code, "DURABLE_ADMISSION_UNAVAILABLE", "{mode:?}");

            let app = app_with_state(state);
            let capabilities = json_body(
                app.clone()
                    .oneshot(empty_request("GET", "/api/capabilities"))
                    .await
                    .unwrap(),
            )
            .await;
            assert_eq!(
                capabilities["queue"]["heterogeneous_batch_max_outputs"],
                serde_json::Value::Null,
                "{mode:?}: a refusing host advertises no batch limit"
            );

            let single: serde_json::Value =
                serde_json::from_str(&generate_body("a cat", 64, 64)).unwrap();
            for (method, path, body) in [
                ("POST", "/api/generate", single.clone()),
                ("POST", "/api/generate/stream", single.clone()),
                (
                    "POST",
                    "/api/generation-batches",
                    serde_json::json!({
                        "client_batch_id": uuid::Uuid::new_v4().to_string(),
                        "requests": [single.clone()],
                    }),
                ),
            ] {
                let refused = app
                    .clone()
                    .oneshot(json_request(method, path, body))
                    .await
                    .unwrap();
                assert_eq!(
                    refused.status(),
                    StatusCode::SERVICE_UNAVAILABLE,
                    "{mode:?} {path}"
                );
                assert_eq!(
                    json_body(refused).await["code"],
                    "DURABLE_ADMISSION_UNAVAILABLE",
                    "{mode:?} {path}"
                );
            }
        }
    }

    /// One precedence, one code. The direct and batch routes used to keep
    /// opposite conjunct orders and two refusal codes for the same host state;
    /// a client cannot act differently on them, so they are one answer now.
    #[tokio::test(flavor = "current_thread")]
    async fn one_refusal_precedence_serves_every_generation_route() {
        let root = tempfile::tempdir().unwrap();
        let db = Arc::new(Some(mold_db::MetadataDb::open_in_memory().unwrap()));
        // Output disabled AND no admission service: two unmet conjuncts, so
        // the shared precedence is what decides the reported code.
        let (mut state, _rx) = durable_state_with_admission_policy(
            db,
            root.path(),
            MockEngine::ready(),
            false,
            false,
            "precedence-test",
        );
        install_authoritative_v2(&mut state);
        state.output_disabled_override = true;

        let mut request = readiness_request();
        let Err(refusal) = crate::routes::direct_durable_admission(&state, &mut request).await
        else {
            panic!("a durable direct request must be refused");
        };
        assert_eq!(refusal.code, "DURABLE_ADMISSION_UNAVAILABLE");

        let refused = app_with_state(state)
            .oneshot(json_request(
                "POST",
                "/api/generation-batches",
                serde_json::json!({
                    "client_batch_id": uuid::Uuid::new_v4().to_string(),
                    "requests": [serde_json::from_str::<serde_json::Value>(
                        &generate_body("a cat", 64, 64)
                    ).unwrap()],
                }),
            ))
            .await
            .unwrap();
        assert_eq!(refused.status(), StatusCode::SERVICE_UNAVAILABLE);
        assert_eq!(
            json_body(refused).await["code"],
            "DURABLE_ADMISSION_UNAVAILABLE"
        );
    }

    /// A degraded encrypted-media store refuses the request that needs it.
    /// There is no attached path to fall back to, and rendering a conditioning
    /// field the host cannot durably retain is exactly the silent
    /// non-durability this rule exists to remove. The table is the point: it
    /// is every conditioning field, not just `source_image`.
    #[tokio::test(flavor = "current_thread")]
    async fn a_degraded_media_store_refuses_a_media_carrying_request() {
        const PIXEL: &str = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg==";
        for field in ["source_image", "id_image", "mask_image", "control_image"] {
            let root = tempfile::tempdir().unwrap();
            let db = Arc::new(Some(mold_db::MetadataDb::open_in_memory().unwrap()));
            // Admission installed, media store NOT reconciled: exactly the
            // shape a corrupt or unreadable queue-media directory produces, and
            // the shape the four readiness conjuncts cannot see.
            let (mut state, _rx) = durable_state(db, root.path());
            install_authoritative_v2(&mut state);
            state.queue_journal.set_durable_media_ready(false);
            assert!(
                state.queue_journal.durable_media_capabilities().is_none(),
                "{field}: fixture must present a degraded media store"
            );

            let mut body: serde_json::Value =
                serde_json::from_str(&generate_body("a cat", 64, 64)).unwrap();
            body[field] = serde_json::Value::String(PIXEL.to_string());
            let mut request: mold_core::GenerateRequest = serde_json::from_value(body).unwrap();

            let Err(refusal) = crate::routes::direct_durable_admission(&state, &mut request).await
            else {
                panic!("{field}: a degraded media store must refuse");
            };
            assert_eq!(refusal.code, "DURABLE_MEDIA_UNAVAILABLE", "{field}");
        }
    }

    /// The overreach tripwire. `v2_authoritative()` serves five distinct roles
    /// and only the durability role belongs behind the readiness authority.
    /// A plain text-to-image request on an authoritative host must still be
    /// admitted with the media store degraded — the encrypted store gates
    /// media, not generation.
    #[tokio::test(flavor = "current_thread")]
    async fn a_degraded_media_store_still_admits_a_mediafree_request() {
        let root = tempfile::tempdir().unwrap();
        let db = Arc::new(Some(mold_db::MetadataDb::open_in_memory().unwrap()));
        let (mut state, _rx) = durable_state(db, root.path());
        install_authoritative_v2(&mut state);
        state.queue_journal.set_durable_media_ready(false);

        let mut request = readiness_request();
        assert!(
            crate::routes::direct_durable_admission(&state, &mut request)
                .await
                .is_ok(),
            "a media-free request does not need the encrypted store"
        );
    }

    // ── Single-print gallery reads ────────────────────────────────────────────
    // Reading one row's metadata used to mean serializing and transferring the
    // whole gallery index, once per artifact.

    #[tokio::test]
    async fn a_filename_filter_narrows_the_listing_to_one_print() {
        let dir = tempfile::tempdir().unwrap();
        // Two valid PNGs so the filesystem fallback lists both.
        for name in ["first.png", "second.png"] {
            let img = image::ImageBuffer::from_fn(64u32, 64u32, |x, y| {
                image::Rgb([(x % 256) as u8, (y % 256) as u8, 128u8])
            });
            img.save(dir.path().join(name)).unwrap();
        }
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

        let all = json_body(
            app_with_state(state.clone())
                .oneshot(
                    Request::builder()
                        .uri("/api/gallery")
                        .body(Body::empty())
                        .unwrap(),
                )
                .await
                .unwrap(),
        )
        .await;
        assert_eq!(all.as_array().unwrap().len(), 2, "both prints list");

        let one = json_body(
            app_with_state(state)
                .oneshot(
                    Request::builder()
                        .uri("/api/gallery?filename=second.png")
                        .body(Body::empty())
                        .unwrap(),
                )
                .await
                .unwrap(),
        )
        .await;
        let rows = one.as_array().unwrap();
        assert_eq!(rows.len(), 1);
        assert_eq!(rows[0]["filename"], "second.png");
    }

    #[tokio::test]
    async fn a_path_shaped_filename_filter_is_refused_not_silently_empty() {
        // An empty listing reads as "the print is gone". A caller bug must
        // say so instead.
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

        for probe in [
            "/api/gallery?filename=../etc/passwd",
            "/api/gallery?filename=",
        ] {
            let resp = app_with_state(state.clone())
                .oneshot(Request::builder().uri(probe).body(Body::empty()).unwrap())
                .await
                .unwrap();
            assert_eq!(
                resp.status(),
                StatusCode::UNPROCESSABLE_ENTITY,
                "{probe} must be refused"
            );
        }
    }

    // ── Held-row retention ────────────────────────────────────────────────────
    // A held row is durable so a human can come back to it; one nobody comes
    // back to pinned a queue row and its encrypted media forever.

    /// Commit one durable child straight through the DB and hold it. The
    /// sweeper's subject is a held ROW, so building one directly keeps this
    /// test about retention rather than about the admission path.
    async fn hold_one_durable_job(state: &AppState, index: usize, held_at_ms: i64) -> String {
        let db = state.metadata_db.clone();
        let owner = state
            .queue_journal
            .owner_uuid()
            .expect("claimed owner")
            .to_string();
        let job_id = format!("held-job-{index}");
        let batch_id = format!("held-batch-{index}");
        let returned = job_id.clone();
        tokio::task::spawn_blocking(move || {
            let db = db.as_ref().as_ref().expect("db");
            let batch = mold_db::generation_batches::GenerationBatchRow {
                id: batch_id.clone(),
                client_batch_id: format!("client-{index}"),
                owner_uuid: owner.clone(),
                request_sha256: format!("hash-{index}"),
                created_at_ms: 1,
            };
            let child = mold_db::generation_batches::GenerationBatchChildRow {
                batch_id,
                job_id: job_id.clone(),
                batch_index: 1,
                state: "accepted".into(),
                error: None,
                updated_at_ms: 1,
            };
            let queue_row = mold_db::generation_queue::GenerationQueueRow {
                id: job_id.clone(),
                owner_uuid: owner.clone(),
                state: mold_db::generation_queue::QueueRowState::Queued,
                model: "flux-dev".into(),
                request_json: r#"{"prompt":"a cat"}"#.into(),
                output_dir: std::path::PathBuf::from("/gallery"),
                target_gpu: None,
                target_device_id: None,
                completion_payload: "metadata_only".into(),
                seed_pinned: false,
                dispatch_attempts: 0,
                replay_seen: 0,
                held_reason: None,
                created_at_ms: 1,
                updated_at_ms: 1,
                started_at_ms: None,
                media_set_id: None,
                admission_authority: None,
            };
            mold_db::generation_batches::insert_or_get(db, &batch, &[(child, queue_row)])
                .expect("admit");
            mold_db::generation_batches::hold_owned(
                db,
                &owner,
                &job_id,
                None,
                "dependency failed",
                None,
                true,
                held_at_ms,
            )
            .expect("hold");
        })
        .await
        .unwrap();
        returned
    }

    #[tokio::test]
    async fn retention_purges_an_abandoned_hold_and_keeps_a_fresh_one() {
        let root = tempfile::tempdir().unwrap();
        let db = Arc::new(Some(mold_db::MetadataDb::open_in_memory().unwrap()));
        let (mut state, _rx) = durable_state(db, root.path());
        install_dispatch_mode(&mut state, crate::dispatch_mode::DispatchMode::V2);

        let day = 86_400_000_i64;
        let now = mold_core::time::now_epoch_ms();
        let stale = hold_one_durable_job(&state, 0, now - 45 * day).await;
        let fresh = hold_one_durable_job(&state, 1, now - day).await;

        state.config.write().await.queue.held_retention_days = 30;
        let result = crate::queue_retention::sweep_held_once(&state)
            .await
            .expect("sweep runs");

        assert_eq!(result.purged, 1, "only the abandoned hold is purged");
        assert_eq!(result.remaining, 1);

        let db = state.metadata_db.clone();
        let (stale_row, fresh_row) = tokio::task::spawn_blocking(move || {
            let db = db.as_ref().as_ref().expect("db");
            (
                mold_db::generation_queue::get(db, &stale).unwrap(),
                mold_db::generation_queue::get(db, &fresh).unwrap(),
            )
        })
        .await
        .unwrap();
        assert!(stale_row.is_none(), "the abandoned hold is gone");
        assert!(fresh_row.is_some(), "a hold inside its window survives");
    }

    #[tokio::test]
    async fn retention_zero_keeps_held_work_forever() {
        // `0` is the documented "keep forever" value on both retention keys.
        // Getting this backwards deletes a user's parked work on first boot.
        let root = tempfile::tempdir().unwrap();
        let db = Arc::new(Some(mold_db::MetadataDb::open_in_memory().unwrap()));
        let (mut state, _rx) = durable_state(db, root.path());
        install_dispatch_mode(&mut state, crate::dispatch_mode::DispatchMode::V2);

        let job = hold_one_durable_job(&state, 0, 0).await;
        state.config.write().await.queue.held_retention_days = 0;

        let result = crate::queue_retention::sweep_held_once(&state)
            .await
            .expect("sweep runs");
        assert_eq!(result.purged, 0);
        assert_eq!(result.remaining, 1);

        let db = state.metadata_db.clone();
        let row = tokio::task::spawn_blocking(move || {
            mold_db::generation_queue::get(db.as_ref().as_ref().unwrap(), &job).unwrap()
        })
        .await
        .unwrap();
        assert!(row.is_some());
    }

    const DAY_MS: i64 = 86_400_000;

    /// Admit one durable batch through the public route and settle its only
    /// child terminally at `settled_at_ms`, the way the worker does once the
    /// print lands. Returns `(batch_id, job_id)`.
    async fn settle_one_durable_batch(state: &AppState, settled_at_ms: i64) -> (String, String) {
        let request: serde_json::Value =
            serde_json::from_str(&generate_body("a settled cat", 512, 512)).unwrap();
        let response = app_with_state(state.clone())
            .oneshot(json_request(
                "POST",
                "/api/generation-batches",
                serde_json::json!({
                    "client_batch_id": uuid::Uuid::new_v4().to_string(),
                    "requests": [request],
                }),
            ))
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::ACCEPTED);
        let body = json_body(response).await;
        let batch_id = body["id"].as_str().unwrap().to_string();
        let job_id = body["children"][0]["job_id"].as_str().unwrap().to_string();

        let db = state.metadata_db.clone();
        let owner = state
            .queue_journal
            .owner_uuid()
            .expect("claimed owner")
            .to_string();
        let settled_job = job_id.clone();
        tokio::task::spawn_blocking(move || {
            let db = db.as_ref().as_ref().expect("db");
            let commit = mold_db::generation_batches::finish_unclaimed_queued(
                db,
                &owner,
                &settled_job,
                mold_db::generation_batches::GenerationBatchTerminal {
                    state: mold_db::generation_batches::GenerationBatchTerminalState::Complete,
                    error: None,
                    terminal_error_json: None,
                    result_json: Some(r#"{"filename":"settled.png"}"#),
                    completed_at_ms: settled_at_ms,
                },
            )
            .expect("settle");
            assert!(commit.queue_deleted && commit.batch_child_updated);
        })
        .await
        .unwrap();
        (batch_id, job_id)
    }

    async fn generation_batch_status(state: &AppState, batch_id: &str) -> StatusCode {
        app_with_state(state.clone())
            .oneshot(
                Request::get(format!("/api/generation-batches/{batch_id}"))
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap()
            .status()
    }

    /// The whole client-facing contract of a purge: the batch answers 404 by
    /// name, the bulk lookup files it under `missing.batch_ids`, and the
    /// print itself is untouched — a settled batch is only a receipt.
    #[tokio::test]
    async fn a_settled_batch_is_purged_after_retention_and_reads_as_missing() {
        let root = tempfile::tempdir().unwrap();
        let db = Arc::new(Some(mold_db::MetadataDb::open_in_memory().unwrap()));
        let (mut state, _rx) = durable_state(db, root.path());
        install_dispatch_mode(&mut state, crate::dispatch_mode::DispatchMode::V2);
        let now = mold_core::time::now_epoch_ms();
        let (stale, _) = settle_one_durable_batch(&state, now - 31 * DAY_MS).await;
        let (fresh, _) = settle_one_durable_batch(&state, now - DAY_MS).await;
        assert_eq!(
            generation_batch_status(&state, &stale).await,
            StatusCode::OK
        );

        state.config.write().await.queue.held_retention_days = 30;
        let response = app_with_state(state.clone())
            .oneshot(
                Request::post("/api/generation-batches/sweep")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::OK);
        let swept = json_body(response).await;
        assert_eq!(
            swept["purged"], 1,
            "only the batch past its window is purged"
        );
        assert_eq!(
            swept["remaining"], 1,
            "the fresh receipt waits for its own window"
        );

        let missing = app_with_state(state.clone())
            .oneshot(
                Request::get(format!("/api/generation-batches/{stale}"))
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(missing.status(), StatusCode::NOT_FOUND);
        assert_eq!(
            json_body(missing).await["code"],
            "GENERATION_BATCH_NOT_FOUND"
        );
        assert_eq!(
            generation_batch_status(&state, &fresh).await,
            StatusCode::OK
        );

        let lookup = json_body(
            app_with_state(state.clone())
                .oneshot(json_request(
                    "POST",
                    "/api/generation-batches/status",
                    serde_json::json!({ "batch_ids": [stale, fresh] }),
                ))
                .await
                .unwrap(),
        )
        .await;
        assert_eq!(lookup["missing"]["batch_ids"], serde_json::json!([stale]));
        assert_eq!(lookup["batches"].as_array().unwrap().len(), 1);
        assert_eq!(lookup["batches"][0]["id"], serde_json::json!(fresh));
    }

    /// A held child is not settled, however old the hold: the batch waits for
    /// the held sweep to settle that child, and only then starts its own clock.
    #[tokio::test]
    async fn a_batch_with_a_held_child_outlives_the_settled_sweep() {
        let root = tempfile::tempdir().unwrap();
        let db = Arc::new(Some(mold_db::MetadataDb::open_in_memory().unwrap()));
        let (mut state, _rx) = durable_state(db, root.path());
        install_dispatch_mode(&mut state, crate::dispatch_mode::DispatchMode::V2);
        let now = mold_core::time::now_epoch_ms();
        hold_one_durable_job(&state, 0, now - 45 * DAY_MS).await;

        state.config.write().await.queue.held_retention_days = 30;
        let result = crate::queue_retention::sweep_settled_batches_once(&state)
            .await
            .expect("sweep runs");
        assert_eq!(result, mold_core::SettledBatchSweepResult::default());
        assert_eq!(
            generation_batch_status(&state, "held-batch-0").await,
            StatusCode::OK
        );
    }

    /// The two passes compose: the held sweep settles the abandoned child as
    /// `failed`, which starts the batch's own retention clock, and the settled
    /// sweep reclaims the summary once THAT has elapsed — never in the same
    /// pass, because a client reconnecting after the hold expired still needs
    /// to read the terminal outcome.
    #[tokio::test]
    async fn held_sweep_then_settled_sweep_reclaims_an_expired_hold_entirely() {
        let root = tempfile::tempdir().unwrap();
        let db = Arc::new(Some(mold_db::MetadataDb::open_in_memory().unwrap()));
        let (mut state, _rx) = durable_state(db.clone(), root.path());
        install_dispatch_mode(&mut state, crate::dispatch_mode::DispatchMode::V2);
        let now = mold_core::time::now_epoch_ms();
        let job = hold_one_durable_job(&state, 0, now - 45 * DAY_MS).await;
        state.config.write().await.queue.held_retention_days = 30;

        let held = crate::queue_retention::sweep_held_once(&state)
            .await
            .expect("held sweep runs");
        assert_eq!(held.purged, 1);
        let settled = crate::queue_retention::sweep_settled_batches_once(&state)
            .await
            .expect("settled sweep runs");
        assert_eq!(
            settled,
            mold_core::SettledBatchSweepResult {
                purged: 0,
                remaining: 1,
            },
            "the child settled just now, so the summary is inside its own window"
        );
        assert_eq!(
            generation_batch_status(&state, "held-batch-0").await,
            StatusCode::OK
        );

        // Thirty-one days later.
        let aged_db = db.clone();
        tokio::task::spawn_blocking(move || {
            aged_db
                .as_ref()
                .as_ref()
                .unwrap()
                .with_conn(|conn| {
                    conn.execute(
                        "UPDATE generation_batch_children SET updated_at_ms = ?2 WHERE job_id = ?1",
                        (job.as_str(), now - 31 * DAY_MS),
                    )?;
                    Ok(())
                })
                .unwrap();
        })
        .await
        .unwrap();
        let settled = crate::queue_retention::sweep_settled_batches_once(&state)
            .await
            .expect("settled sweep runs");
        assert_eq!(
            settled,
            mold_core::SettledBatchSweepResult {
                purged: 1,
                remaining: 0,
            }
        );
        assert_eq!(
            generation_batch_status(&state, "held-batch-0").await,
            StatusCode::NOT_FOUND
        );
    }

    #[tokio::test]
    async fn sweep_route_returns_501_without_metadata_db() {
        // `MOLD_DB_DISABLE` hosts have no durable queue at all; the route says
        // so by name rather than reporting an empty sweep as success.
        let root = tempfile::tempdir().unwrap();
        let (state, _rx) = durable_state(Arc::new(None), root.path());
        let response = app_with_state(state.clone())
            .oneshot(
                Request::post("/api/generation-batches/sweep")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::NOT_IMPLEMENTED);
        assert_eq!(
            json_body(response).await["code"],
            "DURABLE_QUEUE_UNAVAILABLE"
        );
        assert_eq!(
            crate::queue_retention::sweep_settled_batches_once(&state)
                .await
                .expect("sweep runs"),
            mold_core::SettledBatchSweepResult::default()
        );
    }

    #[tokio::test]
    async fn retention_is_inert_without_a_metadata_db() {
        // `MOLD_DB_DISABLE` hosts have no durable queue at all; the sweeper
        // must be a no-op rather than an error every hour.
        let root = tempfile::tempdir().unwrap();
        let (state, _rx) = durable_state(Arc::new(None), root.path());
        let result = crate::queue_retention::sweep_held_once(&state)
            .await
            .expect("sweep runs");
        assert_eq!(result, mold_core::HeldSweepResult::default());
    }

    #[test]
    fn an_expansion_failure_never_journals_a_backend_url_credential() {
        use crate::routes::redact_url_userinfo;

        assert_eq!(
            redact_url_userinfo(
                "expand API request failed: https://svc:s3cret@llm.example.com/v1/chat/completions timed out"
            ),
            "expand API request failed: https://***@llm.example.com/v1/chat/completions timed out"
        );
        // A credential-free URL is returned untouched, path `@` included.
        for untouched in [
            "expand API request failed: http://127.0.0.1:7791/v1/chat/completions refused",
            "expansion backend returned 3 prompts when exactly 1 were requested",
            "expand API request failed: https://llm.example.com/v1/a@b failed",
        ] {
            assert_eq!(redact_url_userinfo(untouched), untouched);
        }
        // The client-facing message is deliberately NOT redacted.
        let api_error = crate::routes::expansion_failed(
            "expand API request failed: https://svc:s3cret@llm.example.com/v1 timed out",
        );
        assert!(
            format!("{api_error:?}").contains("s3cret"),
            "the authenticated caller keeps the verbatim reason"
        );
    }
}
