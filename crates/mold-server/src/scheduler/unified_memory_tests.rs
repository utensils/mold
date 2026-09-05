// Included in scheduler::tests so these exercise the real coordinator and
// transport fakes without constructing a GPU or reading model weights.

fn unified_memory_fixture(
    encoder_bytes: u64,
) -> (
    tempfile::TempDir,
    mold_core::Config,
    mold_core::GenerateRequest,
) {
    let root = tempfile::tempdir().unwrap();
    for (name, bytes) in [
        ("transformer.safetensors", 1 << 20),
        ("vae.safetensors", 1 << 20),
        ("t5.safetensors", encoder_bytes),
    ] {
        crate::execution_plan::sparse_admission_test_file(&root.path().join(name), bytes);
    }
    let mut config = mold_core::Config {
        models_dir: root.path().join("models").display().to_string(),
        t5_variant: Some("fp16".into()),
        ..Default::default()
    };
    config.models.insert(
        "unified-test".into(),
        mold_core::ModelConfig {
            family: Some("flux".into()),
            transformer: Some(
                root.path()
                    .join("transformer.safetensors")
                    .display()
                    .to_string(),
            ),
            vae: Some(root.path().join("vae.safetensors").display().to_string()),
            t5_encoder: Some(root.path().join("t5.safetensors").display().to_string()),
            ..Default::default()
        },
    );
    let mut request = fake_generation("unified-job").0.request;
    request.model = "unified-test".into();
    request.placement = Some(mold_core::DevicePlacement {
        text_encoders: mold_core::DeviceRef::Cpu,
        advanced: None,
    });
    (root, config, request)
}

fn publish_unified_headroom(state: &AppState, backend: mold_core::GpuBackend, available: u64) {
    state.resources.publish(mold_core::ResourceSnapshot {
        hostname: "test".into(),
        timestamp: 0,
        gpus: vec![mold_core::GpuSnapshot {
            ordinal: 0,
            name: "test".into(),
            backend,
            vram_total: 24 << 30,
            vram_used: (24u64 << 30).saturating_sub(available),
            vram_used_by_mold: None,
            vram_used_by_other: None,
            gpu_utilization: None,
        }],
        system_ram: mold_core::RamSnapshot {
            total: 128 << 30,
            used: 16 << 30,
            available: Some(112 << 30),
            reclaimable_zfs_arc: None,
            used_by_mold: 0,
            used_by_other: 16 << 30,
        },
        cpu: None,
    });
}

#[tokio::test]
async fn unified_memory_generation_preview_and_lease_use_the_same_phase_budget() {
    let _env = crate::test_support::hermetic_store_env();
    for backend in [mold_core::GpuBackend::Metal, mold_core::GpuBackend::Cuda] {
        for encoder_bytes in [1 << 20, 12 << 30] {
            let (_root, config, request) = unified_memory_fixture(encoder_bytes);
            let (worker, worker_rx) = if backend == mold_core::GpuBackend::Metal {
                metal_test_worker(0)
            } else {
                test_worker(0)
            };
            let device_id = worker_device_id(&worker);
            let (tx, _rx) = tokio::sync::mpsc::channel(1);
            let state = AppState::empty(
                config.clone(),
                QueueHandle::new(tx),
                Arc::new(GpuPool {
                    workers: vec![worker.clone()].into(),
                }),
                1,
            );
            let mut coordinator = Coordinator::with_preparer_and_memory(
                state.clone(),
                Arc::new(ImmediatePreparer),
                ample_memory(),
            );
            let prepared = crate::variant_dependencies::prepare_execution_inputs_existing_only(
                &state,
                &request,
                Default::default(),
            )
            .await
            .unwrap();
            let (mut job, _result) = fake_generation("unified-job");
            job.request = request.clone();
            state.job_registry.register(&job.id, &request.model);
            let mut immediate = false;
            coordinator.enqueue(job, &mut immediate);
            coordinator
                .pending
                .get_mut("unified-job")
                .unwrap()
                .preparation = PreparationState::Ready;
            coordinator.handle_worker_event(
                WorkerEvent::Ready {
                    device_id,
                    ordinal: 0,
                    owner_epoch: 1,
                    worker_generation: 1,
                },
                &mut immediate,
            );
            let execution = coordinator
                .generation_plans(&coordinator.pending["unified-job"])
                .unwrap()
                .remove(0);
            let demand = execution.admission_vram_demand_bytes();
            if backend == mold_core::GpuBackend::Metal {
                assert!(demand > execution.predicted_vram_peak_bytes);
                assert_eq!(
                    demand,
                    execution.predicted_vram_peak_bytes.max(encoder_bytes)
                        + execution.predicted_host_increment_bytes
                        - encoder_bytes
                );
            } else {
                assert_eq!(demand, execution.predicted_vram_peak_bytes);
            }

            // The raw GPU peak fits, but the unified phase peak plus concurrent
            // transients does not. Preview and the actual queue must both wait.
            publish_unified_headroom(&state, backend, demand - 1);
            let preview = coordinator.placement_preview(&request, 1, &prepared);
            assert_eq!(
                preview.outcome,
                if backend == mold_core::GpuBackend::Cuda {
                    "temporarily_unavailable"
                } else {
                    "infeasible"
                },
                "{backend:?}: {preview:?}"
            );
            coordinator.dispatch_ready().await;
            assert!(
                worker_rx.try_recv().is_err(),
                "an undersized pool must not grant {backend:?}"
            );
            assert!(coordinator.leases.is_empty());

            // Metal fits at the exact unified boundary; summing encoder and
            // denoise would refuse. CUDA keeps FLUX preflight's existing
            // 10% device margin, so exercise its grant with ample VRAM.
            // Host RAM is reserved only on CUDA.
            publish_unified_headroom(
                &state,
                backend,
                if backend == mold_core::GpuBackend::Metal {
                    demand
                } else {
                    24 << 30
                },
            );
            let preview = coordinator.placement_preview(&request, 1, &prepared);
            assert_eq!(preview.outcome, "planned", "{backend:?}: {preview:?}");
            coordinator.dispatch_ready().await;
            let grant = recv_grant(&worker_rx);
            assert_eq!(
                grant.execution_plan.unwrap().predicted_vram_peak_bytes,
                execution.predicted_vram_peak_bytes
            );
            assert_eq!(
                coordinator.memory.reservations["unified-job"].bytes,
                execution.admission_host_demand_bytes()
            );
        }
    }
}

#[tokio::test]
async fn unified_memory_chain_stage_transports_the_admitted_plan() {
    let _env = crate::test_support::hermetic_store_env();
    for backend in [mold_core::GpuBackend::Metal, mold_core::GpuBackend::Cuda] {
        let (_root, config, stage_req) = unified_memory_fixture(12 << 30);
        let (worker, worker_rx) = if backend == mold_core::GpuBackend::Metal {
            metal_test_worker(0)
        } else {
            test_worker(0)
        };
        let (tx, _rx) = tokio::sync::mpsc::channel(1);
        let state = AppState::empty(
            config.clone(),
            QueueHandle::new(tx),
            Arc::new(GpuPool {
                workers: vec![worker.clone()].into(),
            }),
            1,
        );
        let mut coordinator = Coordinator::with_preparer_and_memory(
            state.clone(),
            Arc::new(ImmediatePreparer),
            ample_memory(),
        );
        let mut immediate = false;
        let (result_tx, _result_rx) = tokio::sync::oneshot::channel();
        coordinator.enqueue_owner_work(
            ScheduledOwnerWork::new(
                "unified-stage",
                "unified-test",
                1,
                OwnerWork::ChainStage(Box::new(crate::chain_job_runner::ScheduledChainStageWork {
                    id: "unified-stage".into(),
                    model: stage_req.model.clone(),
                    cache_key: stage_req.model.clone(),
                    config,
                    stage_req,
                    carry: None,
                    motion_tail_frames: 1,
                    progress: Arc::new(|_, _| std::ops::ControlFlow::Continue(())),
                    cancelled: Arc::new(|| false),
                    cancellation: mold_inference::InferenceCancellationToken::default(),
                    on_leased: None,
                    execution_plan: None,
                    expected_model_fingerprint: None,
                    result_tx: Some(result_tx),
                    before_second_fence: None,
                })),
            ),
            &mut immediate,
        );
        coordinator.handle_worker_event(
            WorkerEvent::Ready {
                device_id: worker_device_id(&worker),
                ordinal: 0,
                owner_epoch: 1,
                worker_generation: 1,
            },
            &mut immediate,
        );
        let execution = coordinator
            .owner_plans(&coordinator.pending_owner_work["unified-stage"])
            .unwrap()
            .remove(0);
        let demand = execution.admission_vram_demand_bytes();
        publish_unified_headroom(&state, backend, demand - 1);
        coordinator.dispatch_ready().await;
        assert!(
            worker_rx.try_recv().is_err(),
            "stage must wait for its whole pool claim"
        );
        let blocked = coordinator.pending_owner_work["unified-stage"]
            .memory_block
            .as_ref()
            .expect("a refused stage must reach reclaim and bounded settlement");
        assert_eq!(blocked.required_bytes, demand);
        assert_eq!(blocked.headroom_bytes, demand - 1);
        coordinator
            .pending_owner_work
            .get_mut("unified-stage")
            .unwrap()
            .unschedulable_since_ms = Some(17);
        coordinator.dispatch_ready().await;
        assert_eq!(
            coordinator.pending_owner_work["unified-stage"].unschedulable_since_ms,
            Some(17)
        );
        // Preserve CUDA's existing FLUX preflight margin; only Metal
        // admits the exact unified demand tested here.
        publish_unified_headroom(
            &state,
            backend,
            if backend == mold_core::GpuBackend::Metal {
                demand
            } else {
                24 << 30
            },
        );
        coordinator.dispatch_ready().await;
        let command = worker_rx.try_recv().unwrap_or_else(|error| {
            let cache = coordinator.owner_plan_cache_and_settle_errors();
            let (snapshot, _) = coordinator.planner_snapshot(&cache);
            panic!(
                "a fitting {backend:?} stage must pass lease revalidation: {error:?}; snapshot: {snapshot:?}; plan: {:?}",
                coordinator.planner.plan(&snapshot)
            );
        });
        let crate::gpu_pool::GpuWorkerCommand::Grant(grant) = command else {
            panic!("expected a grant")
        };
        let OwnerWork::ChainStage(stage) = grant.work else {
            panic!("expected stage")
        };
        assert_eq!(
            stage.execution_plan.unwrap().predicted_vram_peak_bytes,
            execution.predicted_vram_peak_bytes
        );
        assert_eq!(
            coordinator.memory.reservations["unified-stage"].bytes,
            execution.admission_host_demand_bytes()
        );
    }
}
