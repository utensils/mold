//! Revision-bound restart exports retain original inputs, never child progress.

use crate::{
    state::AppState,
    video_job_manifest::{read_manifest, VideoJobManifest},
};
use anyhow::{ensure, Context, Result};
use mold_core::{
    job_restart::{RestartAvailability, RestartExportRequest},
    video_job::{VideoJobState, VideoSourceUploadSpec},
    video_restart::{VideoRestartExport, VideoRestartOptions, VideoRestartPrefix},
};
use mold_db::video_jobs::StoredVideoJob;
use sha2::{Digest, Sha256};
use std::{
    fs::File,
    io::{Read, Seek, SeekFrom},
};

const CHUNK_BYTES: usize = 4 * 1024 * 1024;

pub(crate) fn options_for(instance: &str, row: &StoredVideoJob) -> Result<VideoRestartOptions> {
    let manifest = read_manifest(row)?;
    let reason = (!matches!(
        row.job.state,
        VideoJobState::Failed
            | VideoJobState::Paused
            | VideoJobState::Interrupted
            | VideoJobState::Cancelled
    ))
    .then_some("Only a failed, interrupted, paused, or stopped video can start a new attempt.");
    let availability = RestartAvailability {
        supported: reason.is_none(),
        reason: reason.map(str::to_owned),
    };
    let source_revision = format!(
        "{:x}",
        Sha256::digest(serde_json::to_vec(&(
            instance,
            &row.job,
            row.revision,
            &row.request_sha256,
            &row.manifest_sha256
        ))?)
    );
    let local_adapters = manifest.request.generation.lora.is_some()
        || manifest
            .request
            .generation
            .loras
            .as_ref()
            .is_some_and(|items| !items.is_empty());
    Ok(VideoRestartOptions {
        source_instance_id: instance.into(),
        source_revision,
        restart: availability.clone(),
        send: if availability.supported && local_adapters {
            RestartAvailability {supported:false, reason:Some("This video uses machine-local adapters. Start again on this machine or select adapters on the destination.".into())}
        } else {
            availability
        },
    })
}
fn validate_media_budget(request: &mold_core::GenerateRequest) -> Result<()> {
    let mut total = 0usize;
    let mut add = |length: usize| -> Result<()> {
        total = total
            .checked_add(length)
            .context("original media size overflow")?;
        ensure!(
            total <= 32 * 1024 * 1024,
            "original media exceeds portable input budget"
        );
        Ok(())
    };
    for media in [
        &request.source_image,
        &request.id_image,
        &request.mask_image,
        &request.control_image,
        &request.audio_file,
        &request.source_video,
        &request.extend_video,
    ]
    .into_iter()
    .flatten()
    {
        add(media.len())?;
    }
    for media in request
        .id_images
        .iter()
        .chain(request.edit_images.iter())
        .flatten()
    {
        add(media.len())?;
    }
    for frame in request.keyframes.iter().flatten() {
        add(frame.image.len())?;
    }
    for reference in request.references.iter().flatten() {
        match reference.media() {
            mold_core::GenerationReferenceAuthority::Inline { data } => add(data.len())?,
            _ => anyhow::bail!("original reference bytes are unavailable"),
        }
    }
    Ok(())
}
fn source_ref(revision: &str, source: &crate::video_source::RetainedVideoSource) -> Result<String> {
    let mut hash = Sha256::new();
    hash.update(b"mold-video-restart-prefix-v1\0");
    hash.update(serde_json::to_vec(&(
        revision,
        source.authority_kind,
        &source.filename,
        &source.archive_identity_sha256,
        &source.checksum_sha256,
        source.size_bytes,
    ))?);
    Ok(format!("{:x}", hash.finalize()))
}
fn prefix_file(
    state: &AppState,
    row: &StoredVideoJob,
    source: &crate::video_source::RetainedVideoSource,
) -> Result<File> {
    let file = mold_core::secure_file::open_regular_file_no_follow(
        &crate::video_job_manifest::job_dir(state, &row.job.id)?.join("prefix.mp4"),
    )?;
    ensure!(
        file.metadata()?.len() == source.size_bytes,
        "retained prefix size changed"
    );
    Ok(file)
}
fn authenticate_prefix(
    mut file: File,
    source: &crate::video_source::RetainedVideoSource,
    checkpoint: &mut dyn FnMut() -> Result<()>,
) -> Result<()> {
    let mut hash = Sha256::new();
    let mut input = (&mut file).take(
        source
            .size_bytes
            .checked_add(1)
            .context("prefix size overflow")?,
    );
    let mut count = 0u64;
    let mut buffer = [0u8; 64 * 1024];
    loop {
        checkpoint()?;
        let length = input.read(&mut buffer)?;
        if length == 0 {
            break;
        }
        count = count
            .checked_add(length as u64)
            .context("prefix byte count overflow")?;
        hash.update(&buffer[..length]);
    }
    ensure!(
        count == source.size_bytes
            && file.metadata()?.len() == source.size_bytes
            && format!("{:x}", hash.finalize()) == source.checksum_sha256,
        "retained prefix changed"
    );
    Ok(())
}
/// Caller holds the parent actor lock and rechecks the row revision afterwards.
pub(crate) fn export_for_with_checkpoint(
    state: &AppState,
    row: &StoredVideoJob,
    manifest: &VideoJobManifest,
    authority: &RestartExportRequest,
    checkpoint: &mut dyn FnMut() -> Result<()>,
) -> Result<VideoRestartExport> {
    checkpoint()?;
    let options = options_for(&state.instance_id, row)?;
    ensure!(
        authority.source_instance_id == options.source_instance_id
            && authority.source_revision == options.source_revision,
        "original video revision changed"
    );
    ensure!(
        uuid::Uuid::parse_str(&authority.destination_instance_id).is_ok(),
        "capture the destination installation before exporting"
    );
    let portable = authority.destination_instance_id != options.source_instance_id;
    let available = if portable {
        &options.send
    } else {
        &options.restart
    };
    ensure!(
        available.supported,
        "{}",
        available.reason.as_deref().unwrap_or("restart unavailable")
    );
    let authenticated = read_manifest(row)?;
    ensure!(
        serde_json::to_vec(&authenticated)? == serde_json::to_vec(manifest)?,
        "original manifest changed"
    );
    let mut request = crate::video_job_manifest::hydrate_original_request(state, row, manifest)?;
    validate_media_budget(&request.generation)?;
    let db = state
        .metadata_db
        .as_ref()
        .as_ref()
        .context("durable storage unavailable")?;
    crate::routes::resolve_collection_reference(db, &mut request.generation.collection);
    if let Some(collection) = &mut request.generation.collection {
        ensure!(
            collection.name.is_some() || collection.id.is_none(),
            "original collection is unavailable"
        );
        collection.id = None;
    }
    request.generation.placement = None;
    request.generation.batch_id = None;
    request.generation.batch_index = None;
    request.generation.batch_count = None;
    let prefix = manifest
        .retained_source
        .as_ref()
        .map(|source| -> Result<_> {
            let retained_upload = if portable {
                authenticate_prefix(prefix_file(state, row, source)?, source, checkpoint)?;
                None
            } else {
                let id = uuid::Uuid::new_v4().to_string();
                Some(crate::video_source_upload::install_retained(
                    &crate::video_source_upload_routes::root(state)
                        .map_err(|_| anyhow::anyhow!("durable video ingress unavailable"))?,
                    &state.instance_id,
                    &id,
                    &crate::video_job_manifest::job_dir(state, &row.job.id)?.join("prefix.mp4"),
                    source.clone(),
                    checkpoint,
                )?)
            };
            Ok(VideoRestartPrefix {
                retained_upload,
                filename: source.filename.clone(),
                archive_identity_sha256: source.archive_identity_sha256.clone(),
                sha256: source.checksum_sha256.clone(),
                size_bytes: source.size_bytes,
                source_ref: source_ref(&options.source_revision, source)?,
                upload_spec_json: serde_json::to_string(&VideoSourceUploadSpec {
                    authority_kind: source.authority_kind,
                    filename: source.filename.clone(),
                    archive_identity_sha256: source.archive_identity_sha256.clone(),
                    sha256: source.checksum_sha256.clone(),
                    size_bytes: source.size_bytes,
                    metadata: source.metadata.clone(),
                })?,
            })
        })
        .transpose()?;
    Ok(VideoRestartExport {
        destination_instance_id: authority.destination_instance_id.clone(),
        source_instance_id: options.source_instance_id,
        source_revision: options.source_revision,
        request_json: serde_json::to_string(&request)?,
        prefix,
    })
}
fn read_chunk(mut file: File, size: u64, offset: u64) -> Result<Vec<u8>> {
    ensure!(
        offset <= size && file.metadata()?.len() == size,
        "retained source offset or size changed"
    );
    file.seek(SeekFrom::Start(offset))?;
    let length = (size - offset).min(CHUNK_BYTES as u64) as usize;
    let mut bytes = vec![0; length];
    file.read_exact(&mut bytes)?;
    ensure!(
        file.metadata()?.len() == size,
        "retained source size changed during read"
    );
    Ok(bytes)
}
/// Locator is bound to the current revision and original prefix, never a path.
pub(crate) fn source_chunk(
    state: &AppState,
    row: &StoredVideoJob,
    manifest: &VideoJobManifest,
    locator: &str,
    offset: u64,
) -> Result<Vec<u8>> {
    let options = options_for(&state.instance_id, row)?;
    ensure!(
        options.restart.supported,
        "original video is no longer restartable"
    );
    let authenticated = read_manifest(row)?;
    ensure!(
        serde_json::to_vec(&authenticated)? == serde_json::to_vec(manifest)?,
        "original manifest changed"
    );
    let source = manifest
        .retained_source
        .as_ref()
        .context("original video has no prefix")?;
    ensure!(
        source_ref(&options.source_revision, source)? == locator,
        "retained source revision changed"
    );
    read_chunk(prefix_file(state, row, source)?, source.size_bytes, offset)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn historical_cancelled_videos_offer_new_attempts_without_rewriting_source() {
        let mut row = crate::video_job_manifest::tests::fixture();
        row.job.state = VideoJobState::Cancelled;
        row.job.completed_clips = 1;
        let before = (
            row.job.state,
            row.job.completed_clips,
            row.revision,
            row.manifest_sha256.clone(),
        );
        let options = options_for("source", &row).unwrap();
        assert!(options.restart.supported && options.send.supported);
        assert!(!options.source_revision.is_empty());
        assert_eq!(
            (
                row.job.state,
                row.job.completed_clips,
                row.revision,
                row.manifest_sha256
            ),
            before
        );
    }

    #[test]
    fn restart_revision_binds_instance_manifest_request_and_lifecycle() {
        let mut row = crate::video_job_manifest::tests::fixture();
        row.job.state = mold_core::video_job::VideoJobState::Failed;
        let first = options_for("source", &row).unwrap();
        assert!(first.restart.supported);
        row.revision += 1;
        assert_ne!(
            first.source_revision,
            options_for("source", &row).unwrap().source_revision
        );
        assert_ne!(
            first.source_revision,
            options_for("replacement", &row).unwrap().source_revision
        );
        row.job.state = mold_core::video_job::VideoJobState::Completed;
        assert!(!options_for("source", &row).unwrap().restart.supported);
    }
    #[test]
    fn export_counts_all_original_media_in_one_budget() {
        let mut request: mold_core::GenerateRequest = serde_json::from_value(serde_json::json!({"model":"ltx-video:fp16","prompt":"literal\\n", "width":32,"height":32,"steps":1})).unwrap();
        request.id_images = Some(vec![vec![0; 16 * 1024 * 1024]]);
        request.edit_images = Some(vec![vec![0; 16 * 1024 * 1024]]);
        assert!(validate_media_budget(&request).is_ok());
        request.audio_file = Some(vec![1]);
        assert!(validate_media_budget(&request).is_err());
    }
    #[test]
    fn original_prefix_export_can_be_cancelled_during_verification() {
        use std::io::Write;
        let mut file = tempfile::tempfile().unwrap();
        file.write_all(b"pixels").unwrap();
        let row = crate::video_job_manifest::tests::fixture();
        let request = read_manifest(&row).unwrap().request.generation;
        let source = crate::video_source::RetainedVideoSource {
            authority_kind: mold_core::video_job::VideoSourceAuthorityKind::PublishedArchive,
            filename: "source.mp4".into(),
            archive_identity_sha256: "a".repeat(64),
            checksum_sha256: format!("{:x}", Sha256::digest(b"pixels")),
            size_bytes: 6,
            metadata: mold_core::OutputMetadata::from_generate_request(&request, 1, None, "test"),
        };
        assert!(authenticate_prefix(file, &source, &mut || anyhow::bail!("cancelled")).is_err());
    }

    #[test]
    fn source_chunks_are_bounded_and_never_read_past_committed_size() {
        use std::io::Write;
        let mut input = tempfile::tempfile().unwrap();
        input.write_all(&vec![7; CHUNK_BYTES + 20]).unwrap();
        let first = read_chunk(input.try_clone().unwrap(), (CHUNK_BYTES + 20) as u64, 0).unwrap();
        assert_eq!(first.len(), CHUNK_BYTES);
        assert_eq!(
            read_chunk(
                input.try_clone().unwrap(),
                (CHUNK_BYTES + 20) as u64,
                CHUNK_BYTES as u64
            )
            .unwrap(),
            vec![7; 20]
        );
        assert!(read_chunk(input, 1, 2).is_err());
    }
}
