//! Bounded bridge from SQLite's durable generation backlog into the runtime
//! scheduler queue.
//!
//! SQLite owns every not-yet-hydrated child. This task is the sole producer
//! for rows admitted through `/api/generation-batches`; legacy singleton rows
//! are deliberately excluded by the batch-child ownership join.

use crate::state::{AppState, GenerationJob, SubmitError};

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub(crate) struct FeederReport {
    pub submitted: usize,
    pub held: usize,
}

/// Clear ownership tokens left by the prior runtime before any new HTTP
/// admission can mint one. This is a serving precondition: callers must await
/// it and propagate failure before constructing the router.
pub(crate) async fn recover_runtime(
    state: &AppState,
) -> anyhow::Result<mold_db::generation_queue::RuntimeClaimRecovery> {
    let journal = state.queue_journal.clone();
    let report = tokio::task::spawn_blocking(move || journal.recover_feeder_runtime())
        .await
        .map_err(|error| anyhow::anyhow!("durable feeder recovery task failed: {error}"))??;
    if report.claims_cleared > 0 || report.running_requeued > 0 {
        tracing::info!(
            claims_cleared = report.claims_cleared,
            running_requeued = report.running_requeued,
            "durable feeder recovered prior runtime claims"
        );
    }
    Ok(report)
}

pub(crate) fn spawn(
    state: AppState,
    shutdown: tokio_util::sync::CancellationToken,
) -> tokio::task::JoinHandle<()> {
    tokio::spawn(async move {
        run(state, shutdown).await;
    })
}

async fn run(state: AppState, shutdown: tokio_util::sync::CancellationToken) {
    tracing::info!(
        capacity = state.queue_capacity,
        "durable generation queue feeder started"
    );
    let current_output_dir = {
        let config = state.config.read().await;
        (!state.is_output_disabled(&config)).then(|| config.effective_output_dir())
    };
    loop {
        // Construct both futures before the scan. Notify retains one permit,
        // so a commit or capacity release in the scan-to-wait gap is consumed
        // by the select below instead of stranding a row.
        let durable_wake = state.queue_journal.feeder_notified();
        let capacity_wake = state.queue.capacity_notified();
        tokio::pin!(durable_wake);
        tokio::pin!(capacity_wake);

        let report = match feed_available(&state, current_output_dir.as_deref()).await {
            Ok(report) => report,
            Err(()) => break,
        };
        if report.submitted > 0 || report.held > 0 {
            tracing::debug!(
                submitted = report.submitted,
                held = report.held,
                "durable feeder pass complete"
            );
        }
        if shutdown.is_cancelled() {
            break;
        }
        if state.queue.pending() < state.queue_capacity && report.submitted > 0 {
            continue;
        }

        tokio::select! {
            _ = shutdown.cancelled() => break,
            _ = &mut durable_wake => {},
            _ = &mut capacity_wake => {},
        }
    }
    tracing::info!("durable generation queue feeder stopped");
}

async fn feed_available(
    state: &AppState,
    current_output_dir: Option<&std::path::Path>,
) -> Result<FeederReport, ()> {
    let mut report = FeederReport::default();
    loop {
        let reservation = match state.queue.try_reserve(state.queue_capacity) {
            Ok(reservation) => reservation,
            Err(SubmitError::Full { .. }) => break,
            Err(_) => break,
        };
        let journal = state.queue_journal.clone();
        let claim = match tokio::task::spawn_blocking(move || journal.claim_next_feeder()).await {
            Ok(Ok(Some(claim))) => claim,
            Ok(Ok(None)) => {
                drop(reservation);
                break;
            }
            Ok(Err(error)) => {
                drop(reservation);
                tracing::warn!(error = %format!("{error:#}"), "durable feeder could not claim the next row");
                return Err(());
            }
            Err(error) => {
                drop(reservation);
                tracing::warn!(%error, "durable feeder claim task failed");
                return Err(());
            }
        };

        let mut row = claim.row;
        let ticket = state
            .queue_journal
            .attach_claimed(&row.id, claim.claim_token);
        let journal = state.queue_journal.clone();
        let completion_id = row.id.clone();
        let completion =
            tokio::task::spawn_blocking(move || journal.completed_output_exists(&completion_id))
                .await;
        match completion {
            Ok(Ok(true)) => {
                let _ =
                    tokio::task::spawn_blocking(move || ticket.complete_before_dispatch()).await;
                drop(reservation);
                continue;
            }
            Ok(Ok(false)) => {}
            Ok(Err(error)) => {
                let _ = tokio::task::spawn_blocking(move || ticket.retain()).await;
                drop(reservation);
                tracing::error!(job = %row.id, %error, "durable feeder idempotence check failed; stopping without rendering");
                return Err(());
            }
            Err(error) => {
                let _ = tokio::task::spawn_blocking(move || ticket.retain()).await;
                drop(reservation);
                tracing::error!(job = %row.id, %error, "durable feeder idempotence task failed; stopping without rendering");
                return Err(());
            }
        }
        if !row.output_dir.is_dir() {
            match current_output_dir {
                Some(replacement) if replacement != row.output_dir => {
                    let journal = state.queue_journal.clone();
                    let id = row.id.clone();
                    let replacement = replacement.to_path_buf();
                    let db_replacement = replacement.clone();
                    let repointed = tokio::task::spawn_blocking(move || {
                        journal.repoint_output(&id, &db_replacement)
                    })
                    .await;
                    if let Err(error) = repointed
                        .map_err(|error| anyhow::anyhow!(error))
                        .and_then(|result| result)
                    {
                        let _ = tokio::task::spawn_blocking(move || {
                            ticket.hold("the gallery directory could not be reconciled")
                        })
                        .await;
                        drop(reservation);
                        tracing::warn!(job = %row.id, %error, "held durable generation with an unusable output target");
                        report.held += 1;
                        continue;
                    }
                    row.output_dir = replacement;
                }
                Some(_) => {
                    let output_dir = row.output_dir.clone();
                    let created =
                        tokio::task::spawn_blocking(move || std::fs::create_dir_all(output_dir))
                            .await;
                    if let Err(error) = created
                        .map_err(|error| anyhow::anyhow!(error))
                        .and_then(|result| result.map_err(anyhow::Error::from))
                    {
                        let _ = tokio::task::spawn_blocking(move || {
                            ticket.hold("the gallery directory this job targets cannot be created")
                        })
                        .await;
                        drop(reservation);
                        tracing::warn!(job = %row.id, %error, "held durable generation with an unusable output target");
                        report.held += 1;
                        continue;
                    }
                }
                None => {
                    let _ = tokio::task::spawn_blocking(move || {
                        ticket.hold("server gallery output is disabled")
                    })
                    .await;
                    drop(reservation);
                    report.held += 1;
                    continue;
                }
            }
        }
        let mut request: mold_core::GenerateRequest = match serde_json::from_str(&row.request_json)
        {
            Ok(request) => request,
            Err(error) => {
                let _ = tokio::task::spawn_blocking(move || {
                    ticket.hold("the recorded request could not be deserialized")
                })
                .await;
                drop(reservation);
                tracing::warn!(job = %row.id, %error, "held unreadable durable generation");
                report.held += 1;
                continue;
            }
        };
        let target_gpu = match row.target_device_id.as_deref() {
            Some(device_id) => match state
                .gpu_pool
                .workers
                .iter()
                .find(|worker| crate::scheduler::worker_device_id(worker) == device_id)
                .map(|worker| worker.gpu.ordinal)
            {
                Some(ordinal) => Some(ordinal),
                None => {
                    scrub_accelerator_pins(&mut request);
                    tracing::warn!(
                        job = %row.id,
                        device = %device_id,
                        "the device this job was pinned to is not present; resuming on Auto"
                    );
                    None
                }
            },
            None => row.target_gpu,
        };
        let metadata = Box::new(mold_core::OutputMetadata::from_generate_request(
            &request,
            request.seed.unwrap_or(0),
            request.scheduler,
            mold_core::build_info::version_string(),
        ));
        // Cancellation and registry publication share the scheduler mutation
        // fence with DELETE /api/queue. If cancellation won before the row was
        // hydrated, its durable marker is settled by this exact claim. If
        // registration won, the route observes the live token and cancels it.
        let mutation = state.scheduler_mutation_fence.lock().await;
        let cancel = state.job_registry.register_job(
            &row.id,
            &row.model,
            target_gpu,
            Some(row.seed_pinned),
            Some(metadata),
        );
        let journal = state.queue_journal.clone();
        let cancel_id = row.id.clone();
        let cancel_requested =
            tokio::task::spawn_blocking(move || journal.feeder_cancel_requested(&cancel_id)).await;
        match cancel_requested {
            Ok(Ok(true)) => {
                state.job_registry.remove(&row.id);
                let _ = tokio::task::spawn_blocking(move || ticket.discard()).await;
                drop(mutation);
                drop(reservation);
                continue;
            }
            Ok(Ok(false)) => {}
            Ok(Err(error)) => {
                state.job_registry.remove(&row.id);
                let _ = tokio::task::spawn_blocking(move || ticket.retain()).await;
                drop(mutation);
                drop(reservation);
                tracing::error!(job = %row.id, %error, "durable feeder cancellation check failed");
                return Err(());
            }
            Err(error) => {
                state.job_registry.remove(&row.id);
                let _ = tokio::task::spawn_blocking(move || ticket.retain()).await;
                drop(mutation);
                drop(reservation);
                tracing::error!(job = %row.id, %error, "durable feeder cancellation task failed");
                return Err(());
            }
        }
        drop(mutation);
        let crate::job_supervisor::SupervisedJob {
            result_tx,
            outcome_rx,
        } = crate::job_supervisor::supervise_job(row.id.clone(), cancel);
        drop(outcome_rx);
        let id = row.id.clone();
        let job = GenerationJob {
            id: id.clone(),
            request,
            resolved_references: None,
            completion_payload: crate::queue_journal::completion_payload_from_str(
                &row.completion_payload,
            ),
            progress_tx: None,
            result_tx,
            output_dir: Some(row.output_dir),
            batch_child: None,
            journal: Some(ticket),
            #[cfg(any(feature = "h3", feature = "h3-private-uat"))]
            h3_private_ingress_grant: None,
        };
        match reservation.submit(job).await {
            Ok(_) => report.submitted += 1,
            Err((error, mut job)) => {
                if let Some(ticket) = job.journal.take() {
                    let _ = tokio::task::spawn_blocking(move || ticket.retain()).await;
                }
                state.job_registry.remove(&id);
                tracing::warn!(job = %id, ?error, "durable feeder transport stopped; row retained");
                return Err(());
            }
        }
    }
    Ok(report)
}

/// A missing stable lane invalidates every process-local accelerator reference
/// captured with it. CPU is a durable semantic choice, so preserve it while
/// turning only GPU ordinals and stable device IDs back into Auto.
pub(crate) fn scrub_accelerator_pins(request: &mut mold_core::GenerateRequest) {
    fn scrub(device: &mut mold_core::DeviceRef) {
        if matches!(
            device,
            mold_core::DeviceRef::Gpu { .. } | mold_core::DeviceRef::Device { .. }
        ) {
            *device = mold_core::DeviceRef::Auto;
        }
    }

    let Some(placement) = request.placement.as_mut() else {
        return;
    };
    scrub(&mut placement.text_encoders);
    let Some(advanced) = placement.advanced.as_mut() else {
        return;
    };
    scrub(&mut advanced.transformer);
    scrub(&mut advanced.vae);
    for device in [
        advanced.clip_l.as_mut(),
        advanced.clip_g.as_mut(),
        advanced.t5.as_mut(),
        advanced.qwen.as_mut(),
    ]
    .into_iter()
    .flatten()
    {
        scrub(device);
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;
    use std::time::Duration;

    use super::*;
    use crate::queue_journal::{BatchJournalAdmission, JournalAdmission, QueueJournal};
    use crate::state::{AppState, SseCompletionPayload};

    fn request(prompt: &str) -> mold_core::GenerateRequest {
        serde_json::from_value(serde_json::json!({
            "prompt": prompt,
            "model": "mock-model",
            "width": 64,
            "height": 64,
            "steps": 1,
            "batch_size": 1,
            "output_format": "png"
        }))
        .unwrap()
    }

    fn state(capacity: usize) -> (AppState, tokio::sync::mpsc::Receiver<GenerationJob>) {
        let root = tempfile::tempdir().unwrap().keep();
        let gallery = root.join("gallery");
        std::fs::create_dir_all(&gallery).unwrap();
        let mut state = AppState::for_tests();
        let (tx, rx) = tokio::sync::mpsc::channel(capacity.max(1));
        state.queue = crate::state::QueueHandle::new(tx);
        state.queue_capacity = capacity;
        state.metadata_db = Arc::new(Some(mold_db::MetadataDb::open_in_memory().unwrap()));
        state.queue_journal = Arc::new(QueueJournal::new(
            state.metadata_db.clone(),
            Some(&root),
            "feeder-test",
        ));
        state.config.try_write().unwrap().output_dir = Some(gallery.to_string_lossy().into());
        (state, rx)
    }

    fn admit(state: &AppState, count: usize) -> Vec<String> {
        let requests = (0..count)
            .map(|index| request(&format!("job {index}")))
            .collect::<Vec<_>>();
        let ids = (0..count)
            .map(|index| format!("job-{index}"))
            .collect::<Vec<_>>();
        let output = state.config.try_read().unwrap().effective_output_dir();
        let children = requests
            .iter()
            .zip(&ids)
            .map(|(request, id)| JournalAdmission {
                id,
                request,
                output_dir: Some(&output),
                target_gpu: None,
                completion_payload: SseCompletionPayload::MetadataOnly,
                batch_child: false,
                carries_reference_authority: false,
            })
            .collect::<Vec<_>>();
        state
            .queue_journal
            .record_batch(BatchJournalAdmission {
                id: "batch",
                client_batch_id: "client",
                request_sha256: "sha",
                children: &children,
            })
            .unwrap();
        ids
    }

    #[tokio::test]
    async fn deep_backlog_hydrates_no_more_than_runtime_capacity() {
        let (state, mut rx) = state(3);
        admit(&state, 20);
        let shutdown = tokio_util::sync::CancellationToken::new();
        let handle = spawn(state.clone(), shutdown.clone());
        let mut jobs = Vec::new();
        for _ in 0..3 {
            jobs.push(
                tokio::time::timeout(Duration::from_secs(2), rx.recv())
                    .await
                    .unwrap()
                    .unwrap(),
            );
        }
        assert_eq!(state.queue.pending(), 3);
        assert_eq!(state.job_registry.len(), 3);
        assert!(rx.try_recv().is_err());
        assert_eq!(state.queue_journal.list_all().len(), 20);
        state.queue_journal.retain_all();
        shutdown.cancel();
        handle.await.unwrap();
        drop(jobs);
    }

    #[tokio::test]
    async fn fifo_continues_after_each_capacity_release() {
        let (state, mut rx) = state(1);
        let ids = admit(&state, 4);
        let shutdown = tokio_util::sync::CancellationToken::new();
        let handle = spawn(state.clone(), shutdown.clone());
        for expected in ids {
            let mut job = tokio::time::timeout(Duration::from_secs(2), rx.recv())
                .await
                .unwrap()
                .unwrap();
            assert_eq!(job.id, expected);
            job.journal.take().unwrap().complete_before_dispatch();
            state.job_registry.remove(&job.id);
            state.queue.decrement();
        }
        state.queue_journal.retain_all();
        shutdown.cancel();
        handle.await.unwrap();
    }

    #[tokio::test]
    async fn startup_scan_and_post_commit_notification_both_feed_work() {
        let (state, mut rx) = state(2);
        let shutdown = tokio_util::sync::CancellationToken::new();
        let handle = spawn(state.clone(), shutdown.clone());
        tokio::task::yield_now().await;
        admit(&state, 1);
        let job = tokio::time::timeout(Duration::from_secs(2), rx.recv())
            .await
            .unwrap()
            .unwrap();
        assert_eq!(job.id, "job-0");
        state.queue_journal.retain_all();
        shutdown.cancel();
        handle.await.unwrap();
        drop(job);
    }

    #[tokio::test]
    async fn startup_recovery_feeds_an_interrupted_direct_generation() {
        let (state, mut rx) = state(1);
        let request = request("legacy stream interrupted");
        let output = state.config.try_read().unwrap().effective_output_dir();
        let direct_ticket = state
            .queue_journal
            .record(JournalAdmission {
                id: "legacy-direct",
                request: &request,
                output_dir: Some(&output),
                target_gpu: None,
                completion_payload: SseCompletionPayload::MetadataOnly,
                batch_child: false,
                carries_reference_authority: false,
            })
            .unwrap();
        assert!(state.queue_journal.claim_next_feeder().unwrap().is_none());

        state.queue_journal.retain_all();
        drop(direct_ticket);
        assert_eq!(
            state
                .queue_journal
                .recover_feeder_runtime()
                .unwrap()
                .claims_cleared,
            1
        );

        let shutdown = tokio_util::sync::CancellationToken::new();
        let handle = spawn(state.clone(), shutdown.clone());
        let mut replayed = tokio::time::timeout(Duration::from_secs(2), rx.recv())
            .await
            .unwrap()
            .unwrap();
        assert_eq!(replayed.id, "legacy-direct");
        replayed.journal.take().unwrap().complete_before_dispatch();
        assert!(state.queue_journal.list_all().is_empty());
        state.job_registry.remove(&replayed.id);
        state.queue.decrement();

        shutdown.cancel();
        handle.await.unwrap();
    }

    #[tokio::test]
    async fn post_recovery_direct_admission_keeps_its_live_runtime_token() {
        let (state, mut rx) = state(2);
        let request = request("interrupted before startup barrier");
        let output = state.config.try_read().unwrap().effective_output_dir();
        let interrupted_ticket = state
            .queue_journal
            .record(JournalAdmission {
                id: "interrupted-direct",
                request: &request,
                output_dir: Some(&output),
                target_gpu: None,
                completion_payload: SseCompletionPayload::MetadataOnly,
                batch_child: false,
                carries_reference_authority: false,
            })
            .unwrap();
        state.queue_journal.retain_all();
        drop(interrupted_ticket);

        let recovered = recover_runtime(&state).await.unwrap();
        assert_eq!(recovered.claims_cleared, 1);

        // This admission represents an HTTP request accepted only after the
        // startup barrier has completed. The feeder must never run recovery
        // again and erase this live submitter's ownership token.
        let live_ticket = state
            .queue_journal
            .record(JournalAdmission {
                id: "post-barrier-direct",
                request: &request,
                output_dir: Some(&output),
                target_gpu: None,
                completion_payload: SseCompletionPayload::MetadataOnly,
                batch_child: false,
                carries_reference_authority: false,
            })
            .unwrap();

        let shutdown = tokio_util::sync::CancellationToken::new();
        let handle = spawn(state.clone(), shutdown.clone());
        let mut replayed = tokio::time::timeout(Duration::from_secs(2), rx.recv())
            .await
            .unwrap()
            .unwrap();
        assert_eq!(replayed.id, "interrupted-direct");
        assert_eq!(
            live_ticket.claim_dispatch(),
            crate::queue_journal::DispatchClaim::Granted,
            "post-barrier admission must remain owned by its live submitter"
        );

        replayed.journal.take().unwrap().complete_before_dispatch();
        live_ticket.discard();
        state.job_registry.remove(&replayed.id);
        state.queue.decrement();
        shutdown.cancel();
        handle.await.unwrap();
    }

    #[tokio::test]
    async fn missing_stable_device_pin_resumes_on_auto_not_recorded_ordinal() {
        let (state, mut rx) = state(1);
        let mut pinned = request("mixed placement");
        pinned.placement = Some(mold_core::DevicePlacement {
            text_encoders: mold_core::DeviceRef::Cpu,
            advanced: Some(mold_core::AdvancedPlacement {
                transformer: mold_core::DeviceRef::device("cuda:missing-this-boot"),
                vae: mold_core::DeviceRef::gpu(7),
                clip_l: Some(mold_core::DeviceRef::Cpu),
                clip_g: Some(mold_core::DeviceRef::Auto),
                t5: Some(mold_core::DeviceRef::device("cuda:also-stale")),
                qwen: Some(mold_core::DeviceRef::gpu(9)),
            }),
        });
        let output = state.config.try_read().unwrap().effective_output_dir();
        state
            .queue_journal
            .record_batch(BatchJournalAdmission {
                id: "batch",
                client_batch_id: "client",
                request_sha256: "sha",
                children: &[JournalAdmission {
                    id: "job-0",
                    request: &pinned,
                    output_dir: Some(&output),
                    target_gpu: None,
                    completion_payload: SseCompletionPayload::MetadataOnly,
                    batch_child: false,
                    carries_reference_authority: false,
                }],
            })
            .unwrap();
        mold_db::generation_queue::set_target_gpu(
            state.metadata_db.as_ref().as_ref().unwrap(),
            "job-0",
            Some(7),
            Some("cuda:missing-this-boot"),
            2,
        )
        .unwrap();

        let shutdown = tokio_util::sync::CancellationToken::new();
        let handle = spawn(state.clone(), shutdown.clone());
        let job = tokio::time::timeout(Duration::from_secs(2), rx.recv())
            .await
            .unwrap()
            .unwrap();
        assert_eq!(job.id, "job-0");
        assert_eq!(state.job_registry.target_gpu("job-0"), Some(None));
        assert_eq!(
            state
                .gpu_pool
                .resolve_explicit_placement_gpu(job.request.placement.as_ref()),
            Ok(None),
            "the hydrated job must not retain an ordinal or stable accelerator pin"
        );
        let placement = job.request.placement.as_ref().unwrap();
        assert_eq!(placement.text_encoders, mold_core::DeviceRef::Cpu);
        let advanced = placement.advanced.as_ref().unwrap();
        assert_eq!(advanced.transformer, mold_core::DeviceRef::Auto);
        assert_eq!(advanced.vae, mold_core::DeviceRef::Auto);
        assert_eq!(advanced.clip_l, Some(mold_core::DeviceRef::Cpu));
        assert_eq!(advanced.clip_g, Some(mold_core::DeviceRef::Auto));
        assert_eq!(advanced.t5, Some(mold_core::DeviceRef::Auto));
        assert_eq!(advanced.qwen, Some(mold_core::DeviceRef::Auto));

        state.queue_journal.retain_all();
        shutdown.cancel();
        handle.await.unwrap();
        drop(job);
    }

    #[tokio::test]
    async fn cancellation_that_wins_before_registry_handoff_skips_the_exact_child() {
        let (state, mut rx) = state(1);
        admit(&state, 2);
        let claimed = state.queue_journal.claim_next_feeder().unwrap().unwrap();
        assert_eq!(claimed.row.id, "job-0");
        state.queue_journal.cancel_id("job-0");
        recover_runtime(&state).await.unwrap();

        let shutdown = tokio_util::sync::CancellationToken::new();
        let handle = spawn(state.clone(), shutdown.clone());
        let next = tokio::time::timeout(Duration::from_secs(2), rx.recv())
            .await
            .unwrap()
            .unwrap();
        assert_eq!(next.id, "job-1");
        let detail = state.queue_journal.generation_batch("batch").unwrap();
        assert_eq!(detail.children[0].state, "cancelled");
        assert_eq!(detail.children[1].state, "accepted");
        assert!(state
            .queue_journal
            .list_all()
            .iter()
            .all(|row| row.id != "job-0"));

        state.queue_journal.retain_all();
        shutdown.cancel();
        handle.await.unwrap();
        drop(next);
    }

    #[tokio::test]
    async fn cancellation_before_dispatch_settles_the_exact_claim_and_keeps_later_work() {
        let (state, mut rx) = state(1);
        admit(&state, 2);
        let shutdown = tokio_util::sync::CancellationToken::new();
        let handle = spawn(state.clone(), shutdown.clone());
        let first = tokio::time::timeout(Duration::from_secs(2), rx.recv())
            .await
            .unwrap()
            .unwrap();
        state.job_registry.cancel_queued(&first.id).unwrap();
        state.queue_journal.cancel_id(&first.id);
        let first_id = first.id.clone();
        drop(first);
        state.queue.decrement();

        let second = tokio::time::timeout(Duration::from_secs(2), rx.recv())
            .await
            .unwrap()
            .unwrap();
        assert_eq!(second.id, "job-1");
        let detail = state.queue_journal.generation_batch("batch").unwrap();
        assert_eq!(detail.children[0].job_id, first_id);
        assert_eq!(detail.children[0].state, "cancelled");
        assert!(state
            .queue_journal
            .list_all()
            .iter()
            .all(|row| row.id != first_id));

        state.queue_journal.retain_all();
        shutdown.cancel();
        handle.await.unwrap();
        drop(second);
    }

    #[tokio::test]
    async fn shutdown_keeps_claimed_and_unclaimed_rows_for_restart_recovery() {
        let (state, mut rx) = state(1);
        admit(&state, 3);
        let shutdown = tokio_util::sync::CancellationToken::new();
        let handle = spawn(state.clone(), shutdown.clone());
        let claimed = tokio::time::timeout(Duration::from_secs(2), rx.recv())
            .await
            .unwrap()
            .unwrap();
        state.queue_journal.retain_all();
        shutdown.cancel();
        handle.await.unwrap();
        drop(claimed);
        assert_eq!(state.queue_journal.list_all().len(), 3);
        let report = state.queue_journal.recover_feeder_runtime().unwrap();
        assert_eq!(report.claims_cleared, 1);
        assert_eq!(report.running_requeued, 0);
        let ids = state
            .queue_journal
            .list_all()
            .into_iter()
            .map(|row| row.id)
            .collect::<Vec<_>>();
        assert_eq!(ids, vec!["job-0", "job-1", "job-2"]);
    }
}
