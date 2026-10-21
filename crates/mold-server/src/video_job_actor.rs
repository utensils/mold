//! Bounded parent orchestration; the existing chain runner owns all inference.

use crate::video_job_manifest::VideoJobManifest;
use anyhow::{ensure, Context, Result};
use mold_core::{
    chain::{ChainIncomingSource, ChainRequest, TransitionMode},
    video_job_schedule::VideoJobPartPlan,
};

fn child_request(
    parent_id: &str,
    index: u64,
    manifest: &VideoJobManifest,
    mut template: ChainRequest,
    part: &VideoJobPartPlan,
    incoming: Option<ChainIncomingSource>,
) -> Result<ChainRequest> {
    ensure!(
        part.clips.len() == 1 && part.first_clip == index && index < manifest.plan.clip_count,
        "invalid bounded child plan"
    );
    ensure!(
        template.stages.len() == 1 && template.seed.is_some(),
        "child template has no exact original execution authority"
    );
    if let Some(source) = &incoming {
        ensure!(
            source.parent_job_id == parent_id && template.motion_tail_frames > 0,
            "incoming pixels belong to another parent or recipe"
        );
        source.validate()?;
    } else {
        ensure!(
            index == 0 || template.motion_tail_frames == 0,
            "continuous child is missing its exact published incoming pixels"
        );
    }
    let first = index == 0 && incoming.is_none();
    let final_clip =
        index.checked_add(1).context("child index overflow")? == manifest.plan.clip_count;
    let stage = &mut template.stages[0];
    stage.prompt = part.prompt.clone();
    stage.frames = part.clips[0].frames;
    stage.seed_offset = Some(index);
    stage.authored_media_sha256 = None;
    stage.transition = TransitionMode::Smooth;
    if !first {
        stage.source_image = None;
    }
    if let Some(keyframes) = &mut stage.keyframes {
        keyframes.retain(|keyframe| {
            if keyframe.frame == 0 {
                first
            } else {
                final_clip
            }
        });
        for keyframe in keyframes {
            if keyframe.frame != 0 {
                keyframe.frame = stage.frames - 1;
            }
        }
    }
    template.continue_from = None;
    template.incoming_source = incoming;
    template.output_mode = Some(mold_core::GenerationOutputMode::OneShot);
    template.output_format = mold_core::OutputFormat::Mp4;
    Ok(template)
}

pub(crate) struct VideoJobRunnerHandle {
    wake: std::sync::Arc<tokio::sync::Notify>,
    shutdown: tokio_util::sync::CancellationToken,
}
impl VideoJobRunnerHandle {
    pub(crate) fn kick(&self) {
        self.wake.notify_one();
    }
}

#[derive(Debug)]
pub(crate) struct VideoStopped;
impl std::fmt::Display for VideoStopped {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str("Video stopped before publication")
    }
}
impl std::error::Error for VideoStopped {}

#[cfg(feature = "mp4")]
mod runtime {
    use super::*;
    use crate::{state::AppState, video_job_output::VideoJobOutput};
    use anyhow::bail;
    use mold_core::{
        chain_job::{ChainJobManifest, ChainJobState},
        video_job::VideoJobState,
        video_job_schedule::VideoJobSchedule,
    };
    use mold_db::{chain_jobs, video_jobs as db};
    use std::{collections::HashMap, sync::Arc};
    use tokio_util::sync::CancellationToken;

    pub(crate) fn spawn(state: AppState, shutdown: CancellationToken) -> Arc<VideoJobRunnerHandle> {
        let handle = Arc::new(VideoJobRunnerHandle {
            wake: Arc::new(tokio::sync::Notify::new()),
            shutdown,
        });
        tokio::spawn(run(state, handle.clone()));
        handle
    }
    fn database(state: &AppState) -> Result<&mold_db::MetadataDb> {
        state
            .metadata_db
            .as_ref()
            .as_ref()
            .context("video storage is unavailable")
    }
    fn alive(state: &AppState, id: &str, shutdown: &CancellationToken) -> Result<()> {
        if shutdown.is_cancelled() || db::cancellation_requested(database(state)?, id)? {
            return Err(VideoStopped.into());
        }
        Ok(())
    }
    pub(crate) fn verify_recipe(
        config: &mold_core::Config,
        manifest: &VideoJobManifest,
    ) -> Result<()> {
        crate::routes_chain::require_chain_artifact_activation(
            config,
            &manifest.template,
            None,
            Some(&manifest.frozen),
        )
        .map_err(|error| anyhow::anyhow!(error.error))?;
        let mut frozen = config.clone();
        frozen
            .install_frozen_model_config(&manifest.template.model, manifest.frozen.config.clone());
        crate::video_job_loras::verify(
            &frozen,
            &manifest.request.generation,
            &manifest.frozen_loras,
        )
    }
    fn save(
        database: &mold_db::MetadataDb,
        row: &mut db::StoredVideoJob,
        part: Option<&db::StoredVideoJobPart>,
    ) -> Result<bool> {
        row.job.updated_at_ms = crate::video_jobs::now_ms();
        row.job.can_cancel = row.job.state.can_cancel();
        row.job.can_resume = row.job.state.can_resume();
        row.job.can_delete = !matches!(
            row.job.state,
            VideoJobState::Queued | VideoJobState::Running
        );
        if db::checkpoint(database, row, part)? {
            row.revision += 1;
            Ok(true)
        } else {
            Ok(false)
        }
    }
    fn settle(
        state: &AppState,
        id: &str,
        status: VideoJobState,
        error: Option<String>,
    ) -> Result<()> {
        let database = database(state)?;
        if let Some(mut row) = db::get(database, id)? {
            if !matches!(
                row.job.state,
                VideoJobState::Queued | VideoJobState::Running
            ) {
                return Ok(());
            }
            row.job.state = status;
            row.job.error = error;
            row.job.error_detail = None;
            save(database, &mut row, None)?;
        }
        Ok(())
    }
    fn child_id(row: &db::StoredVideoJob, index: u64) -> String {
        use sha2::{Digest, Sha256};
        let mut hash = Sha256::new();
        hash.update(b"mold.video.parent-child.v1\0");
        hash.update(row.job.id.as_bytes());
        hash.update(row.manifest_sha256.as_bytes());
        hash.update(index.to_le_bytes());
        let digest = hash.finalize();
        let mut bytes = [0u8; 16];
        bytes.copy_from_slice(&digest[..16]);
        bytes[6] = (bytes[6] & 0x0f) | 0x80;
        bytes[8] = (bytes[8] & 0x3f) | 0x80;
        uuid::Uuid::from_bytes(bytes).to_string()
    }
    fn exact_child(
        row: &chain_jobs::ChainJobRow,
        manifest: &ChainJobManifest,
        expected: &ChainRequest,
        parent: &VideoJobManifest,
    ) -> Result<()> {
        ensure!(
            manifest.frozen_model.as_ref() == Some(&parent.frozen),
            "child frozen model differs from its owning parent"
        );
        let mut actual =
            crate::chain_job_runner::effective_request_with_media(manifest, &row.job_dir)?;
        let mut expected = expected.clone();
        for stage in &mut actual.stages {
            stage.authored_media_sha256 = None;
        }
        for stage in &mut expected.stages {
            stage.authored_media_sha256 = None;
        }
        ensure!(
            actual == expected,
            "child request differs from its immutable parent claim"
        );
        if expected.has_incoming_boundary() {
            crate::chain_continuation::load(&row.job_dir, &actual, &parent.frozen)?;
        }
        Ok(())
    }
    async fn run(state: AppState, handle: Arc<VideoJobRunnerHandle>) {
        let slots = state.gpu_pool.schedulable_worker_count().max(1);
        let mut outputs: HashMap<String, VideoJobOutput> = HashMap::new();
        let mut cursor = None;
        let mut tick = tokio::time::interval(std::time::Duration::from_millis(300));
        loop {
            tokio::select! { _=handle.shutdown.cancelled()=>break, _=handle.wake.notified()=>{}, _=tick.tick()=>{} }
            let database = match database(&state) {
                Ok(database) => database,
                Err(error) => {
                    tracing::warn!(%error,"video actor storage unavailable");
                    break;
                }
            };
            let mut candidates = Vec::new();
            let mut missing = Vec::new();
            for id in outputs.keys() {
                match db::get(database, id) {
                    Ok(Some(row)) => candidates.push(row),
                    Ok(None) => missing.push(id.clone()),
                    Err(error) => tracing::warn!(%error,"Could not refresh active video parent"),
                }
            }
            for id in missing {
                outputs.remove(&id);
            }
            if outputs.len() < slots {
                match db::runnable_page(database, cursor.as_ref(), 32) {
                    Ok(rows) => {
                        if rows.is_empty() {
                            cursor = None;
                        } else {
                            cursor = rows.last().map(db::RunnableCursor::from);
                        }
                        candidates.extend(
                            rows.into_iter()
                                .filter(|row| !outputs.contains_key(&row.job.id)),
                        );
                    }
                    Err(error) => tracing::warn!(%error,"could not read runnable video parents"),
                }
            }
            for candidate in candidates {
                if !outputs.contains_key(&candidate.job.id)
                    && outputs.len() >= slots
                    && !candidate.cancel_requested
                {
                    continue;
                }
                let guard = match crate::video_jobs::try_lock(&candidate.job.id) {
                    Ok(Some(guard)) => guard,
                    Ok(None) => continue,
                    Err(error) => {
                        tracing::warn!(message=%error.error,"could not lock video parent");
                        continue;
                    }
                };
                let id = candidate.job.id.clone();
                let output = outputs.remove(&id);
                let config = state.config.read().await.clone();
                let turn_state = state.clone();
                let turn_shutdown = handle.shutdown.clone();
                let turn_id = id.clone();
                let result = tokio::task::spawn_blocking(move || {
                    let _guard = guard;
                    turn(&turn_state, &turn_id, output, &config, &turn_shutdown)
                })
                .await;
                match result {
                    Ok(Ok(Some(output))) => {
                        outputs.insert(id, output);
                    }
                    Ok(Ok(None)) => {}
                    Ok(Err(error)) => {
                        let stopped = error.downcast_ref::<VideoStopped>().is_some();
                        let status = if stopped && handle.shutdown.is_cancelled() {
                            VideoJobState::Interrupted
                        } else if stopped {
                            VideoJobState::Cancelled
                        } else {
                            VideoJobState::Failed
                        };
                        if let Err(storage) =
                            settle(&state, &id, status, Some(format!("{error:#}")))
                        {
                            tracing::warn!(error=%storage,"could not persist failed video parent");
                        }
                    }
                    Err(error) => {
                        let _ = settle(
                            &state,
                            &id,
                            VideoJobState::Failed,
                            Some(format!("Video orchestration task stopped: {error}")),
                        );
                    }
                }
            }
        }
    }
    fn prune_previous_child(state: &AppState, row: &db::StoredVideoJob) -> Result<bool> {
        if row.job.completed_clips == 0 {
            return Ok(true);
        }
        let database = database(state)?;
        let part = db::get_part(database, &row.job.id, row.job.completed_clips - 1)?
            .context("completed clip has no durable receipt")?;
        ensure!(
            part.output_sha256.is_some() && part.receipt_sha256.is_some(),
            "completed clip receipt is not committed"
        );
        let runner = state
            .chain_jobs
            .as_ref()
            .context("chain runner unavailable")?;
        let _guard = runner.blocking_lock_job(&part.child_id);
        let Some(child) = chain_jobs::get_job(database, &part.child_id)? else {
            return Ok(true);
        };
        ensure!(
            child.state == ChainJobState::Completed,
            "committed clip child is not completed"
        );
        let authority = crate::chain_execution::ChainExecutionAuthority::read_for_parent(
            &child.job_dir,
            &child.id,
        )?;
        if authority.state != crate::chain_execution::ChainExecutionState::Settled {
            return Ok(false);
        }
        let retained_runtime:bool=database.with_conn(|conn|Ok(conn.query_row("SELECT EXISTS(SELECT 1 FROM chain_stage_generations owner JOIN generation_queue queue ON queue.id=owner.generation_id WHERE owner.chain_id=?1)",[&child.id],|row|row.get(0))?))?;
        if retained_runtime {
            return Ok(false);
        }
        let root = database
            .path()
            .parent()
            .context("Missing captured data home")?
            .join("jobs");
        ensure!(
            child.job_dir == root.join(&child.id),
            "child storage differs from captured owner"
        );
        crate::chain_source_media::release_all(&root, &child.job_dir)?;
        if child.job_dir.try_exists()? {
            std::fs::remove_dir_all(&child.job_dir)?;
        }
        ensure!(
            chain_jobs::delete_job_not_running(database, &child.id)?,
            "completed child changed during retention cleanup"
        );
        runner.cleanup_deleted(&child.id);
        Ok(true)
    }

    fn turn(
        state: &AppState,
        id: &str,
        output: Option<VideoJobOutput>,
        config: &mold_core::Config,
        shutdown: &CancellationToken,
    ) -> Result<Option<VideoJobOutput>> {
        let database = database(state)?;
        let mut row = db::get(database, id)?.context("video parent disappeared")?;
        if !matches!(
            row.job.state,
            VideoJobState::Queued | VideoJobState::Running
        ) {
            return Ok(None);
        }
        let runner = state
            .chain_jobs
            .as_ref()
            .context("chain runner unavailable")?;
        if row.cancel_requested {
            if let Some(child) = &row.job.active_child_id {
                runner.request_cancel(child);
                let _ = chain_jobs::try_transition(
                    database,
                    child,
                    &[ChainJobState::Queued],
                    ChainJobState::Cancelled,
                    None,
                    crate::video_jobs::now_ms(),
                )?;
                if chain_jobs::get_job(database, child)?.is_some_and(|child| {
                    matches!(child.state, ChainJobState::Queued | ChainJobState::Running)
                }) {
                    return Ok(output);
                }
            }
            row.job.state = VideoJobState::Cancelled;
            save(database, &mut row, None)?;
            return Ok(None);
        }
        let resuming = row.job.state == VideoJobState::Queued && row.job.active_child_id.is_some();
        if row.job.state == VideoJobState::Queued {
            row.job.state = VideoJobState::Running;
            if !save(database, &mut row, None)? {
                return Ok(None);
            }
        }
        let manifest = crate::video_job_manifest::read_manifest(&row)?;
        let schedule = VideoJobSchedule::new(
            &manifest.request,
            manifest.envelope,
            manifest.plan.fps,
            manifest.plan.source_frames,
        )?;
        ensure!(
            schedule.clip_count() == manifest.plan.clip_count
                && schedule.part_count() == manifest.plan.part_count,
            "immutable duration schedule changed"
        );
        let mut checkpoint = || alive(state, id, shutdown);
        let mut output = match output {
            Some(output) => output,
            None => {
                verify_recipe(config, &manifest)?;
                VideoJobOutput::open(state, &row, &manifest, &mut checkpoint)?
            }
        };
        if !prune_previous_child(state, &row)? {
            return Ok(Some(output));
        }
        if row.job.completed_clips == manifest.plan.clip_count {
            let identity = output.finish(state, &row, &manifest, &mut checkpoint)?;
            // Publication is the commit point. A later Stop cannot unpublish
            // this movie; settle the exact archived receipt after reloading CAS.
            for _ in 0..3 {
                let mut current = db::get(database, id)?.context("published parent disappeared")?;
                current.job.state = VideoJobState::Completed;
                current.job.output_filename = Some(identity.final_name.clone());
                current.job.active_child_id = None;
                current.job.error = None;
                current.job.error_detail = None;
                if save(database, &mut current, None)? {
                    return Ok(None);
                }
            }
            anyhow::bail!("published video could not settle its durable receipt");
        }
        let index = row.job.completed_clips;
        let plan = schedule.part(index).context("missing bounded child plan")?;
        let expected_id = child_id(&row, index);
        // Shared with public create: reserve and create one UUID atomically.
        let _child_guard = runner.blocking_lock_job(&expected_id);
        let mut claim = if let Some(claim) = db::get_part(database, id, index)? {
            ensure!(
                claim.child_id == expected_id && claim.first_clip == index && claim.clip_count == 1,
                "child ledger identity changed"
            );
            claim
        } else {
            ensure!(
                chain_jobs::get_job(database, &expected_id)?.is_none(),
                "Video child operation ID is already occupied; start a new attempt"
            );
            db::StoredVideoJobPart {
                parent_id: id.into(),
                index,
                child_id: expected_id.clone(),
                first_clip: index,
                clip_count: 1,
                keep_frames: if manifest.plan.join_mode
                    == mold_core::video_job::VideoJoinMode::Continuous
                {
                    plan.clips[0].delivered_frames()
                } else {
                    u64::from(plan.clips[0].frames - plan.clips[0].end_trim_frames)
                },
                output_sha256: None,
                receipt_sha256: None,
            }
        };
        if row.job.active_child_id.as_deref() != Some(expected_id.as_str()) {
            ensure!(
                row.job.active_child_id.is_none(),
                "parent points to another child"
            );
            row.job.active_child_id = Some(expected_id.clone());
            row.job.state = VideoJobState::Running;
            if !save(database, &mut row, Some(&claim))? {
                return Ok(None);
            }
        }
        if let Some(child) = chain_jobs::get_job(database, &expected_id)? {
            if matches!(child.state, ChainJobState::Queued | ChainJobState::Running) {
                return Ok(Some(output));
            }
        }
        verify_recipe(config, &manifest)?;
        let incoming = output.incoming(state, &row, &manifest, &mut checkpoint)?;
        let template = crate::chain_source_media::hydrate(
            &crate::video_job_manifest::jobs_root(state)?,
            &crate::video_job_manifest::job_dir(state, id)?,
            manifest.template.clone(),
        )?;
        let request = child_request(
            id,
            index,
            &manifest,
            template,
            &plan,
            incoming.as_ref().map(|(binding, _)| binding.clone()),
        )?;
        alive(state, id, shutdown)?;
        let child = match chain_jobs::get_job(database, &expected_id)? {
            Some(child) => child,
            None => {
                let params = crate::chain_job_runner::CreateJobParams {
                    id: expected_id.clone(),
                    ephemeral: false,
                    request: request.clone(),
                    frozen_model: Some(manifest.frozen.clone()),
                };
                let root = database
                    .path()
                    .parent()
                    .context("chain storage has no captured home")?
                    .join("jobs");
                if let Some((_, frames)) = incoming {
                    let prepared = crate::chain_continuation::prepare_external(
                        &request,
                        &manifest.frozen,
                        frames,
                    )?;
                    crate::chain_job_runner::create_job_with_prepared_continuation(
                        database, &root, params, &prepared,
                    )?
                } else {
                    crate::chain_job_runner::create_job_with_params(database, &root, params)?
                }
            }
        };
        let mut child_manifest = ChainJobManifest::read_from_dir(&child.job_dir)?;
        exact_child(&child, &child_manifest, &request, &manifest)?;
        match child.state {
            ChainJobState::Queued | ChainJobState::Running => {
                runner.kick();
                Ok(Some(output))
            }
            ChainJobState::Completed => {
                ensure!(
                    child_manifest.current_revision_is_finalized(child.state),
                    "child completion has no current finalized take"
                );
                let receipt = output.accept_child(
                    state,
                    &row,
                    &manifest,
                    index,
                    &plan,
                    &child,
                    &child_manifest,
                    &mut checkpoint,
                )?;
                ensure!(
                    receipt.segment_frames == claim.keep_frames,
                    "private segment frame window differs from durable claim"
                );
                if let Some(expected) = &claim.output_sha256 {
                    ensure!(
                        expected == &receipt.sha256,
                        "private segment output identity changed"
                    );
                }
                if claim
                    .receipt_sha256
                    .as_ref()
                    .is_some_and(|sha| sha != &receipt.receipt_sha256)
                {
                    bail!("completed video receipt authority changed");
                }
                claim.output_sha256 = Some(receipt.sha256);
                claim.receipt_sha256 = Some(receipt.receipt_sha256);
                row.job.completed_clips = row
                    .job
                    .completed_clips
                    .checked_add(1)
                    .context("clip count overflow")?;
                row.job.active_child_id = None;
                if !save(database, &mut row, Some(&claim))? {
                    return Ok(None);
                }
                Ok(Some(output))
            }
            ChainJobState::Failed
            | ChainJobState::Interrupted
            | ChainJobState::Paused
            | ChainJobState::Cancelled => {
                if resuming {
                    crate::chain_job_runner::reset_running_stages_for_retry(
                        database,
                        &expected_id,
                        &child.job_dir,
                        &mut child_manifest,
                        crate::video_jobs::now_ms(),
                    )?;
                    ensure!(
                        chain_jobs::try_transition(
                            database,
                            &expected_id,
                            &[child.state],
                            ChainJobState::Queued,
                            None,
                            crate::video_jobs::now_ms()
                        )?,
                        "child changed before explicit parent Resume"
                    );
                    let stage = child_manifest
                        .stage_status
                        .iter()
                        .find(|stage| stage.state != mold_core::chain_job::StageState::Completed)
                        .map(|stage| stage.idx)
                        .unwrap_or(child_manifest.stage_status.len() as u32);
                    chain_jobs::set_current_stage(
                        database,
                        &expected_id,
                        stage,
                        crate::video_jobs::now_ms(),
                    )?;
                    runner.kick();
                    Ok(Some(output))
                } else {
                    anyhow::bail!(
                        "Video clip {} stopped: {}",
                        index + 1,
                        child
                            .error
                            .as_deref()
                            .unwrap_or("resume the parent to continue its exact checkpoint")
                    )
                }
            }
        }
    }
}
#[cfg(feature = "mp4")]
pub(crate) use runtime::{spawn, verify_recipe};

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn internal_children_keep_one_clip_and_preserve_only_global_endpoints() {
        let row = crate::video_job_manifest::tests::fixture();
        let manifest = crate::video_job_manifest::read_manifest(&row).unwrap();
        let mut template = manifest.template.clone();
        template.stages[0].source_image = Some(vec![1]);
        template.stages[0].keyframes = Some(vec![mold_core::KeyframeCondition {
            frame: 96,
            image: vec![2],
            name: Some("closing.png".into()),
        }]);
        let schedule = mold_core::video_job_schedule::VideoJobSchedule::new(
            &manifest.request,
            manifest.envelope,
            manifest.plan.fps,
            0,
        )
        .unwrap();
        let first = child_request(
            &row.job.id,
            0,
            &manifest,
            template.clone(),
            &schedule.part(0).unwrap(),
            None,
        )
        .unwrap();
        assert_eq!(first.stages.len(), 1);
        assert_eq!(first.stages[0].source_image, Some(vec![1]));
        assert!(first.stages[0].keyframes.as_ref().is_none_or(Vec::is_empty));
        assert_eq!(
            first.output_mode,
            Some(mold_core::GenerationOutputMode::OneShot)
        );
        let last_index = schedule.part_count() - 1;
        let part = schedule.part(last_index).unwrap();
        let last = child_request(
            &row.job.id,
            last_index,
            &manifest,
            template,
            &part,
            Some(mold_core::chain::ChainIncomingSource {
                parent_job_id: row.job.id.clone(),
                source_sha256: "a".repeat(64),
                source_frames: 657,
            }),
        )
        .unwrap();
        assert!(last.stages[0].source_image.is_none());
        assert_eq!(
            last.stages[0].keyframes.as_ref().unwrap()[0].frame,
            part.clips[0].frames - 1
        );
        assert_eq!(last.stages[0].seed_offset, Some(last_index));
    }
}
