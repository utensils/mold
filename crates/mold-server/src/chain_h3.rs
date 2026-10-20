//! H3 sequence stages use ordinary durable generation admission, one owned
//! attempt per stage. They never enter the reusable inference-engine cache.

use std::ops::ControlFlow;
use std::sync::Arc;
use std::time::{Duration, Instant};

use anyhow::{anyhow, bail, Context};
use mold_core::{GenerateRequest, GenerationBatchAdmissionRequest, GenerationBatchChildState};
use mold_inference::chain::{ChainTail, StageOutcome};
use sha2::{Digest, Sha256};

use crate::chain_job_runner::{ChainLeaseCallback, StageExecution, StageRenderOutcome};
use crate::state::{AppState, SseCompletionPayload};

fn child_client_id(job: &str, stage: u32, authored_take: u64, retry: u32) -> String {
    let digest = Sha256::digest(format!(
        "mold.chain.h3.child.v1\0{job}\0{stage}\0{authored_take}\0{retry}"
    ));
    let mut bytes = [0; 16];
    bytes.copy_from_slice(&digest[..16]);
    bytes[6] = (bytes[6] & 0x0f) | 0x50;
    bytes[8] = (bytes[8] & 0x3f) | 0x80;
    uuid::Uuid::from_bytes(bytes).to_string()
}

fn apply_carry(request: &mut GenerateRequest, carry: Option<&ChainTail>) -> anyhow::Result<()> {
    if mold_core::minimax_h3::task_for_model(&request.model)
        != Some(mold_core::minimax_h3::Task::Fl2va)
    {
        return Ok(());
    }
    let Some(last) = carry.and_then(|tail| tail.tail_rgb_frames.last()) else {
        return Ok(());
    };
    if request.source_image.is_some()
        || request
            .keyframes
            .as_ref()
            .is_some_and(|keys| keys.iter().any(|key| key.frame == 0))
    {
        bail!("H3 smooth continuation cannot replace an authored first frame; use cut or fade");
    }
    let mut png = std::io::Cursor::new(Vec::new());
    last.write_to(&mut png, image::ImageFormat::Png)?;
    request.source_image = Some(png.into_inner());
    Ok(())
}

/// Only the one durable child admitted for this exact stage attempt may be
/// hidden after retention. Titles and tags never confer ownership.
fn require_owned_child(
    detail: &mold_db::generation_batches::DurableGenerationBatchDetail,
    batch_id: &str,
    client_id: &str,
) -> anyhow::Result<()> {
    anyhow::ensure!(
        detail.batch.id == batch_id && detail.batch.client_batch_id == client_id,
        "H3 child batch differs from the owned stage attempt"
    );
    anyhow::ensure!(
        detail.children.len() == 1
            && detail.children[0].batch_id == batch_id
            && detail.children[0].batch_index == 1,
        "H3 stage must own exactly one child"
    );
    Ok(())
}

/// Caller holds the gallery publication writer for the entire copy-to-Trash
/// handoff. Fail closed before Trash; a Trash failure only leaves a live copy.
fn retain_and_trash_child(
    output: &std::path::Path,
    filename: &str,
    artifact: &std::path::Path,
    db: &mold_db::MetadataDb,
    gate: &crate::batch_transaction::GalleryPublicationGate,
    events: &crate::events::EventBroadcaster,
) -> anyhow::Result<()> {
    anyhow::ensure!(
        !filename.is_empty()
            && filename != "."
            && filename != ".."
            && !filename.contains('/')
            && !filename.contains('\\'),
        "invalid H3 child gallery filename"
    );
    match std::fs::symlink_metadata(artifact) {
        Ok(_) => {
            let existing = crate::batch_transaction::open_regular_file_no_follow(artifact)?;
            anyhow::ensure!(
                existing.metadata()?.len() > 0,
                "empty retained H3 child artifact"
            );
        }
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => {
            let parent = artifact
                .parent()
                .context("H3 artifact has no parent directory")?;
            std::fs::create_dir_all(parent)?;
            // Persist the new retention-directory entry before hiding its source.
            if let Some(ancestor) = parent.parent() {
                crate::dir_sync::sync_directory(ancestor)?;
            }
            let source = crate::gallery_trash::resolve_gallery_media_source(output, filename);
            let mut source = crate::batch_transaction::open_regular_file_no_follow(&source)?;
            let temporary = parent.join(format!(".{}.tmp", uuid::Uuid::new_v4()));
            let retained = (|| -> anyhow::Result<()> {
                let mut options = std::fs::OpenOptions::new();
                options.write(true).create_new(true);
                #[cfg(unix)]
                {
                    use std::os::unix::fs::OpenOptionsExt;
                    options.mode(0o600);
                }
                let mut staged = options.open(&temporary)?;
                anyhow::ensure!(
                    std::io::copy(&mut source, &mut staged)? > 0,
                    "empty H3 child output"
                );
                staged.sync_all()?;
                std::fs::rename(&temporary, artifact)?;
                crate::dir_sync::sync_directory(parent)?;
                Ok(())
            })();
            let _ = std::fs::remove_file(&temporary);
            retained?;
        }
        Err(error) => return Err(error.into()),
    }
    crate::gallery_trash::trash_published_outputs_blocking(
        output,
        &crate::queue::SavedOutputNames {
            output: Some(filename.into()),
            original: None,
        },
        Some(db),
        gate,
        Some(events),
    );
    Ok(())
}

/// Existing client IDs are public and may be preoccupied. Always pass the
/// canonical request through ordinary admission's authenticated receipt check
/// before reattaching, cancelling, or hiding an existing child.
async fn verify_or_admit_child(
    state: &AppState,
    client_id: &str,
    request: &GenerateRequest,
) -> anyhow::Result<String> {
    Ok(state
        .queue_journal
        .queue_media_admission()
        .context("H3 chain requires durable generation admission")?
        .admit_batch(
            state,
            None,
            Some(crate::reference_uploads::ReferenceIdentity::AuthDisabled {
                instance_id: state.instance_id.to_string(),
            }),
            GenerationBatchAdmissionRequest {
                client_batch_id: client_id.into(),
                requests: vec![request.clone()],
            },
            None,
            SseCompletionPayload::MetadataOnly,
        )
        .await
        .map_err(|error| anyhow!(error.error))?
        .status
        .id)
}

/// Reattach completed children after a crash. An authored retake/amend has a
/// different key; failed/cancelled children get a new retry only on a later
/// parent invocation, never an automatic retry in this polling loop.
#[allow(clippy::too_many_arguments)]
pub(crate) async fn render(
    state: &AppState,
    job_id: &str,
    stage_idx: u32,
    authored_take: u64,
    request: &GenerateRequest,
    carry: Option<&ChainTail>,
    motion_tail_frames: u32,
    frozen: Option<&mold_core::chain_job::FrozenChainModel>,
    mut on_leased: Option<ChainLeaseCallback>,
    progress: Arc<dyn Fn(u32, u32) -> ControlFlow<()> + Send + Sync>,
    cancelled: Arc<dyn Fn() -> bool + Send + Sync>,
) -> anyhow::Result<StageExecution> {
    let started = Instant::now();
    let mut state = state.clone();
    if let Some(frozen) = frozen {
        let current =
            crate::execution_plan::frozen_model_fingerprint(&request.model, &frozen.config)?;
        if current != frozen.model_fingerprint {
            bail!("frozen H3 chain model inputs changed before admission");
        }
        let mut config = state.config.read().await.clone();
        config.install_frozen_model_config(&request.model, frozen.config.clone());
        state.config = Arc::new(tokio::sync::RwLock::new(config));
    }
    let mut request = request.clone();
    apply_carry(&mut request, carry)?;
    request.save_to_gallery = Some(true);
    request.title = Some(format!("Chain {job_id} · stage {}", stage_idx + 1));
    request.tags = Some(vec![
        format!("chain:{job_id}"),
        format!("chain-stage:{stage_idx}"),
    ]);
    let mut retry = 0u32;
    let existing = loop {
        let client_id = child_client_id(job_id, stage_idx, authored_take, retry);
        let existing = state
            .queue_journal
            .durable_generation_batch_by_client(&client_id)
            .map_err(anyhow::Error::msg)?;
        let existing = if existing.is_some() {
            // Admission verifies the expected request's owner-bound receipt.
            // If GC raced the lookup, the returned batch is a newly admitted,
            // owned child and must be followed rather than orphaned.
            let verified = verify_or_admit_child(&state, &client_id, &request).await?;
            let detail = state
                .queue_journal
                .durable_generation_batch(&verified)
                .map_err(anyhow::Error::msg)?
                .context("verified H3 child batch disappeared")?;
            require_owned_child(&detail, &verified, &client_id)?;
            Some(detail)
        } else {
            None
        };
        match existing {
            Some(detail)
                if detail.children.first().is_some_and(|child| {
                    matches!(
                        child.state.as_str(),
                        "failed" | "cancelled" | "held" | "paused"
                    )
                }) =>
            {
                crate::routes::cancel_generation_batch_children(&state, &detail.batch.id)
                    .await
                    .map_err(|error| anyhow!(error.error))?;
                retry = retry.checked_add(1).context("H3 child retry overflow")?;
            }
            detail => break (client_id, detail),
        }
    };
    if cancelled() && existing.1.is_none() {
        return Ok(StageExecution {
            outcome: StageRenderOutcome::Cancelled,
            device_ordinal: None,
        });
    }
    let expected_client_id = existing.0.clone();
    let batch_id = if let Some(detail) = existing.1 {
        detail.batch.id
    } else {
        verify_or_admit_child(&state, &expected_client_id, &request).await?
    };
    let mut child_activity_recorded = false;
    let outcome = async {
        let mut cancelling = false;
        loop {
            if cancelled() && !cancelling {
                crate::routes::cancel_generation_batch_children(&state, &batch_id)
                    .await
                    .map_err(|error| anyhow!(error.error))?;
                cancelling = true;
            }
            let detail = state
                .queue_journal
                .durable_generation_batch(&batch_id)
                .map_err(anyhow::Error::msg)?
                .context("H3 chain child batch disappeared")?;
            require_owned_child(&detail, &batch_id, &expected_client_id)?;
            let status = crate::routes::generation_batch_status(&state.instance_id, detail);
            let child = status
                .children
                .first()
                .context("H3 chain child batch is empty")?;
            if !child_activity_recorded {
                let db = state
                    .metadata_db
                    .as_ref()
                    .as_ref()
                    .context("H3 chain database unavailable")?;
                let parent =
                    mold_db::chain_jobs::get_job(db, job_id)?.context("H3 chain disappeared")?;
                crate::routes_activity::record_stage_child_activity(
                    &parent.job_dir,
                    job_id,
                    &child.job_id,
                )?;
                child_activity_recorded = true;
            }
            if child.state == GenerationBatchChildState::Running {
                if let Some(ordinal) = state
                    .job_registry
                    .entry(&child.job_id)
                    .and_then(|entry| entry.gpu)
                {
                    if let Some(on_leased) = on_leased.take() {
                        on_leased(ordinal).map_err(anyhow::Error::msg)?;
                    }
                }
            }
            if let Some(snapshot) = state
                .job_registry
                .progress_snapshot(&child.job_id)
                .flatten()
            {
                if let (Some(step), Some(total)) = (snapshot.step, snapshot.total) {
                    if progress(step as u32, total as u32).is_break() && !cancelling {
                        crate::routes::cancel_generation_batch_children(&state, &batch_id)
                            .await
                            .map_err(|error| anyhow!(error.error))?;
                        cancelling = true;
                    }
                }
            }
            match child.state {
                GenerationBatchChildState::Accepted
                | GenerationBatchChildState::Running
                | GenerationBatchChildState::Cancelling => {}
                GenerationBatchChildState::Complete => {
                    let result = child
                        .result
                        .as_ref()
                        .context("completed H3 child has no result")?;
                    if let Some(on_leased) = on_leased.take() {
                        on_leased(
                            result
                                .gpu
                                .context("completed H3 child has no accelerator authority")?,
                        )
                        .map_err(anyhow::Error::msg)?;
                    }
                    if cancelling || cancelled() {
                        return Ok(StageExecution {
                            outcome: StageRenderOutcome::Cancelled,
                            device_ordinal: result.gpu,
                        });
                    }
                    let filename = result
                        .filename
                        .as_ref()
                        .context("completed H3 child has no gallery video")?;
                    let output = state.config.read().await.effective_output_dir();
                    if filename.contains('/') || filename.contains('\\') {
                        bail!("H3 child returned an invalid gallery filename");
                    }
                    let db = state
                        .metadata_db
                        .as_ref()
                        .as_ref()
                        .context("H3 chain database unavailable")?;
                    let job = mold_db::chain_jobs::get_job(db, job_id)?
                        .context("H3 chain disappeared")?;
                    let retained_dir = job.job_dir.join("h3-children");
                    let artifact = retained_dir.join(format!("{batch_id}.mp4"));
                    {
                        let _writer = state.gallery_publication_gate.write().await;
                        retain_and_trash_child(
                            &output,
                            filename,
                            &artifact,
                            db,
                            &state.gallery_publication_gate,
                            &state.events,
                        )?;
                    }
                    let (probe, frames) =
                        mold_inference::ltx2::media::decode_video_frames_from_path(&artifact)?;
                    if frames.len() != request.frames.unwrap_or_default() as usize
                        || probe.fps != request.fps.unwrap_or(24)
                    {
                        bail!("H3 child decoded frame/fps facts differ from the admitted stage");
                    }
                    let mut audio =
                        mold_inference::ltx2::media::decode_native_audio_from_mp4_file(&artifact)?
                            .context("H3 child is missing its synchronized soundtrack")?;
                    let sample_count =
                        frames.len() as u64 * u64::from(audio.sample_rate) / u64::from(probe.fps);
                    let wanted = sample_count as usize * usize::from(audio.channels);
                    if audio.interleaved_samples.len() < wanted {
                        bail!("H3 child soundtrack is shorter than its video");
                    }
                    audio.interleaved_samples.truncate(wanted);
                    let tail_count = (motion_tail_frames as usize).max(1).min(frames.len());
                    let tail_rgb_frames = frames[frames.len() - tail_count..].to_vec();
                    return Ok(StageExecution {
                        device_ordinal: result.gpu,
                        outcome: StageRenderOutcome::Done(StageOutcome {
                            frames,
                            tail: ChainTail {
                                frames: tail_count as u32,
                                tail_rgb_frames,
                            },
                            audio: Some(audio),
                            hdr_frames_written: None,
                            generation_time_ms: result
                                .generation_time_ms
                                .unwrap_or(started.elapsed().as_millis() as u64),
                            attention_path: None,
                            int8_arm: None,
                        }),
                    });
                }
                GenerationBatchChildState::Held
                | GenerationBatchChildState::Failed
                | GenerationBatchChildState::Cancelled
                | GenerationBatchChildState::Paused => {
                    if cancelling {
                        return Ok(StageExecution {
                            outcome: StageRenderOutcome::Cancelled,
                            device_ordinal: None,
                        });
                    }
                    bail!(
                        "H3 chain child {}: {}",
                        child.job_id,
                        child
                            .error
                            .as_deref()
                            .unwrap_or("generation did not complete")
                    );
                }
            }
            tokio::time::sleep(Duration::from_millis(300)).await;
        }
    }
    .await;
    if outcome.is_err() {
        // A failed parent callback or journal read must not orphan a child
        // that was already admitted. Keep ownership until cancellation settles.
        crate::routes::cancel_generation_batch_children(&state, &batch_id)
            .await
            .map_err(|error| anyhow!(error.error))?;
        loop {
            let detail = state
                .queue_journal
                .durable_generation_batch(&batch_id)
                .map_err(anyhow::Error::msg)?
                .context("H3 child disappeared during cancellation")?;
            let settled = detail.children.iter().all(|child| {
                matches!(
                    child.state.as_str(),
                    "complete" | "failed" | "cancelled" | "held" | "paused"
                )
            });
            if settled {
                break;
            }
            tokio::time::sleep(Duration::from_millis(300)).await;
        }
    }
    outcome
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn delivered_child_audio_decoder_preserves_sample_rate_channels_and_frame_count() {
        use base64::Engine;
        let bytes = base64::engine::general_purpose::STANDARD
            .decode(include_str!("testdata/audio_muxed_final_mp4.b64").trim())
            .unwrap();
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("child.mp4");
        std::fs::write(&path, bytes).unwrap();
        let probe = mold_inference::ltx2::media::probe_decoded_mp4_audio_file(&path)
            .unwrap()
            .unwrap();
        let track = mold_inference::ltx2::media::decode_native_audio_from_mp4_file(&path)
            .unwrap()
            .unwrap();
        assert_eq!(track.sample_rate, probe.sample_rate);
        assert_eq!(track.channels, probe.channels);
        assert_eq!(
            track.interleaved_samples.len() as u64,
            probe.samples_per_channel * u64::from(probe.channels)
        );
        assert!(track
            .interleaved_samples
            .iter()
            .all(|sample| sample.is_finite()));
    }

    #[test]
    fn child_identity_survives_recovery_but_changes_for_an_authored_take() {
        let first = child_client_id("chain", 2, 0, 0);
        assert_eq!(first, child_client_id("chain", 2, 0, 0));
        assert_ne!(first, child_client_id("chain", 3, 0, 0));
        assert_ne!(first, child_client_id("chain", 2, 1, 0));
        assert_ne!(first, child_client_id("chain", 2, 0, 1));
        uuid::Uuid::parse_str(&first).unwrap();
    }

    #[test]
    fn endpoint_handoff_preserves_authored_last_frame() {
        let mut request: mold_core::GenerateRequest = serde_json::from_value(
            serde_json::json!({"model": mold_core::minimax_h3::FL2VA_COMFY, "prompt": "test", "frames": 107, "width": 256, "height": 256, "steps": 21}),
        ).unwrap();
        request.keyframes = Some(vec![mold_core::KeyframeCondition {
            frame: 106,
            image: vec![7],
            name: Some("ending".into()),
        }]);
        let carry = mold_inference::chain::ChainTail {
            frames: 1,
            tail_rgb_frames: vec![image::RgbImage::from_pixel(32, 32, image::Rgb([1, 2, 3]))],
        };
        apply_carry(&mut request, Some(&carry)).unwrap();
        assert!(request.source_image.is_some());
        assert_eq!(request.keyframes.as_ref().unwrap()[0].image, vec![7]);
        assert!(apply_carry(&mut request, Some(&carry)).is_err());
    }
}

#[cfg(test)]
#[path = "chain_h3_retention_tests.rs"]
mod retention_tests;
