//! Exact published-video authority retained independently of Library deletion.
#[cfg(any(feature = "mp4", test))]
use anyhow::{ensure, Context};
use serde::{Deserialize, Serialize};
#[cfg(any(feature = "mp4", test))]
use sha2::{Digest, Sha256};
#[cfg(any(feature = "mp4", test))]
use std::{
    fs::File,
    io::{Read, Seek, SeekFrom, Write},
    path::Path,
};

#[cfg(feature = "mp4")]
use crate::{routes::ApiError, state::AppState};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub(crate) struct RetainedVideoSource {
    #[serde(default)]
    pub authority_kind: mold_core::video_job::VideoSourceAuthorityKind,
    pub filename: String,
    pub archive_identity_sha256: String,
    pub checksum_sha256: String,
    pub size_bytes: u64,
    #[serde(deserialize_with = "mold_core::video_job::deserialize_literal_metadata")]
    pub metadata: mold_core::OutputMetadata,
}

/// Open under the publication reader before releasing the original name.
/// The descriptor remains valid if the user subsequently trashes or purges it.
#[cfg(feature = "mp4")]
pub(crate) async fn open(
    state: &AppState,
    filename: &str,
    expected_identity: Option<&str>,
) -> Result<(File, RetainedVideoSource), ApiError> {
    open_mode(state, filename, expected_identity, false).await
}

/// Scalar previews must never hash a growing legacy movie. Empty legacy
/// digests are deliberately nonadmissible; full validation uses `open`.
#[cfg(feature = "mp4")]
pub(crate) async fn open_preview(
    state: &AppState,
    filename: &str,
    expected_identity: Option<&str>,
) -> Result<(File, RetainedVideoSource), ApiError> {
    open_mode(state, filename, expected_identity, true).await
}

#[cfg(feature = "mp4")]
async fn open_mode(
    state: &AppState,
    filename: &str,
    expected_identity: Option<&str>,
    preview: bool,
) -> Result<(File, RetainedVideoSource), ApiError> {
    crate::routes::validate_gallery_filename(filename)?;
    let _publication = state.gallery_publication_gate.read().await;
    let config = state.config.read().await;
    if state.is_output_disabled(&config) {
        return Err(ApiError::not_found("Library output is unavailable"));
    }
    let output = config.effective_output_dir();
    drop(config);
    let state = state.clone();
    let filename = filename.to_owned();
    let expected = expected_identity.map(str::to_owned);
    tokio::task::spawn_blocking(move || {
        let resolved = state
            .gallery_publication_gate
            .validated_retained_media_for_item(&output, &filename)
            .map_err(|_| ApiError::internal("Could not inspect Library video authority"))?;
        let (identity, metadata) = match resolved {
            crate::batch_transaction::ValidatedRetainedMedia::Present {
                identity,
                metadata,
                ..
            } => (identity, metadata),
            crate::batch_transaction::ValidatedRetainedMedia::Missing => {
                let index = state
                    .gallery_publication_gate
                    .committed_archive_index(&output)
                    .map_err(|_| ApiError::internal("Could not inspect Library ownership"))?;
                if index.is_retired(&filename) || index.is_quarantined(&filename) {
                    return Err(ApiError::not_found(
                        "The selected video is no longer in Library",
                    ));
                }
                let row = state
                    .metadata_db
                    .as_ref()
                    .as_ref()
                    .ok_or_else(|| ApiError::not_found("Library metadata is unavailable"))?
                    .get(&output, &filename)
                    .map_err(|_| ApiError::internal("Could not inspect selected Library movie"))?
                    .filter(|row| {
                        row.format == mold_core::OutputFormat::Mp4
                            && !row.metadata_synthetic
                            && row.trashed_at_ms.is_none()
                    })
                    .ok_or_else(|| {
                        ApiError::not_found("The selected video is no longer in Library")
                    })?;
                let file =
                    mold_core::secure_file::open_regular_file_no_follow(&output.join(&filename))
                        .map_err(|_| {
                            ApiError::not_found("The selected video file is unavailable")
                        })?;
                let capture = if preview {
                    legacy_metadata
                } else {
                    snapshot_legacy
                };
                let (file, source) = capture(file, filename, row.metadata).map_err(|_| {
                    ApiError::validation("Could not freeze the selected legacy video")
                })?;
                if !preview
                    && expected
                        .as_ref()
                        .is_some_and(|hash| hash != &source.archive_identity_sha256)
                {
                    return Err(source_changed());
                }
                return Ok((file, source));
            }
            crate::batch_transaction::ValidatedRetainedMedia::Invalid => {
                return Err(ApiError::validation(
                    "The selected video differs from its published authority",
                ));
            }
        };
        let archive_identity_sha256 =
            crate::gallery_source_media::archive_identity_sha256(&identity)?;
        if expected
            .as_ref()
            .is_some_and(|value| value != &archive_identity_sha256)
        {
            return Err(ApiError::with_code(
                "The selected video changed. Refresh Library before extending it.",
                "VIDEO_SOURCE_CHANGED",
                axum::http::StatusCode::CONFLICT,
            ));
        }
        let file = mold_core::secure_file::open_regular_file_no_follow(&output.join(&filename))
            .map_err(|_| ApiError::not_found("The selected video file is unavailable"))?;
        if file
            .metadata()
            .map_err(|_| ApiError::not_found("Video unavailable"))?
            .len()
            != identity.size_bytes
        {
            return Err(ApiError::validation(
                "The selected video differs from its published authority",
            ));
        }
        Ok((
            file,
            RetainedVideoSource {
                authority_kind: mold_core::video_job::VideoSourceAuthorityKind::PublishedArchive,
                filename,
                archive_identity_sha256,
                checksum_sha256: identity.checksum_sha256,
                size_bytes: identity.size_bytes,
                metadata: *metadata,
            },
        ))
    })
    .await
    .map_err(|_| ApiError::internal("Library video inspection failed"))?
}

#[cfg(feature = "mp4")]
fn source_changed() -> ApiError {
    ApiError::with_code(
        "The selected video changed. Refresh Library before extending it.",
        "VIDEO_SOURCE_CHANGED",
        axum::http::StatusCode::CONFLICT,
    )
}

/// A new current-pixel admission, never a promotion to historical archive authority.
#[cfg(any(feature = "mp4", test))]
fn snapshot_legacy(
    mut file: File,
    filename: String,
    metadata: mold_core::OutputMetadata,
) -> anyhow::Result<(File, RetainedVideoSource)> {
    let expected_size = file.metadata()?.len();
    ensure!(expected_size > 0, "source movie is empty");
    let checksum_sha256 = hash_exact(file.try_clone()?, expected_size)?;
    let size_bytes = expected_size;
    let mut identity = Sha256::new();
    identity.update(b"mold-video-current-pixel-snapshot-v1\0");
    identity.update(serde_json::to_vec(&(
        &filename,
        &checksum_sha256,
        size_bytes,
        &metadata,
    ))?);
    file.seek(SeekFrom::Start(0))?;
    Ok((
        file,
        RetainedVideoSource {
            authority_kind: mold_core::video_job::VideoSourceAuthorityKind::LegacySnapshot,
            filename,
            archive_identity_sha256: format!("{:x}", identity.finalize()),
            checksum_sha256,
            size_bytes,
            metadata,
        },
    ))
}

#[cfg(any(feature = "mp4", test))]
fn legacy_metadata(
    file: File,
    filename: String,
    metadata: mold_core::OutputMetadata,
) -> anyhow::Result<(File, RetainedVideoSource)> {
    let size_bytes = file.metadata()?.len();
    ensure!(size_bytes > 0, "source movie is empty");
    Ok((
        file,
        RetainedVideoSource {
            authority_kind: mold_core::video_job::VideoSourceAuthorityKind::LegacySnapshot,
            filename,
            archive_identity_sha256: String::new(),
            checksum_sha256: String::new(),
            size_bytes,
            metadata,
        },
    ))
}

#[cfg(any(feature = "mp4", test))]
fn hash_exact(mut file: File, expected: u64) -> anyhow::Result<String> {
    file.seek(SeekFrom::Start(0))?;
    let limit = expected.checked_add(1).context("source size overflow")?;
    let mut input = (&mut file).take(limit);
    let mut hash = Sha256::new();
    let mut bytes = 0u64;
    let mut buffer = [0u8; 64 * 1024];
    loop {
        let count = input.read(&mut buffer)?;
        if count == 0 {
            break;
        }
        bytes = bytes
            .checked_add(count as u64)
            .context("source size overflow")?;
        hash.update(&buffer[..count]);
    }
    ensure!(
        bytes == expected && file.metadata()?.len() == expected,
        "legacy video changed during snapshot"
    );
    Ok(format!("{:x}", hash.finalize()))
}

/// Stream the original into a private, immutable job-owned file. Never route
/// the growing published video through an inline-reference upload budget.
#[cfg(any(feature = "mp4", test))]
pub(crate) fn retain(
    mut source: File,
    authority: &RetainedVideoSource,
    job_directory: &Path,
    checkpoint: &mut dyn FnMut() -> anyhow::Result<()>,
) -> anyhow::Result<()> {
    source.seek(SeekFrom::Start(0))?;
    let mut staged = tempfile::NamedTempFile::new_in(job_directory)?;
    let mut hash = Sha256::new();
    let mut bytes = 0u64;
    let mut buffer = [0; 64 * 1024];
    loop {
        checkpoint()?;
        let count = source.read(&mut buffer)?;
        if count == 0 {
            break;
        }
        bytes = bytes
            .checked_add(count as u64)
            .context("video source size overflowed")?;
        ensure!(
            bytes <= authority.size_bytes,
            "video source changed during retention"
        );
        hash.update(&buffer[..count]);
        staged.write_all(&buffer[..count])?;
    }
    ensure!(
        bytes == authority.size_bytes
            && format!("{:x}", hash.finalize()) == authority.checksum_sha256,
        "video source changed from its published checksum"
    );
    checkpoint()?;
    staged.as_file().sync_all()?;
    staged.persist_noclobber(job_directory.join("prefix.mp4"))?;
    crate::dir_sync::sync_directory(job_directory)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn authority(bytes: &[u8]) -> RetainedVideoSource {
        let request = serde_json::from_value(serde_json::json!({"model":"ltx-2-19b-distilled:fp8",
            "prompt":"original", "width":32, "height":32, "steps":4}))
        .unwrap();
        RetainedVideoSource {
            authority_kind: mold_core::video_job::VideoSourceAuthorityKind::PublishedArchive,
            filename: "original.mp4".into(),
            archive_identity_sha256: "archive".into(),
            checksum_sha256: format!("{:x}", Sha256::digest(bytes)),
            size_bytes: bytes.len() as u64,
            metadata: mold_core::OutputMetadata::from_generate_request(&request, 1, None, "test"),
        }
    }

    #[test]
    fn preview_has_no_admissible_pixel_proof_and_snapshot_reads_are_size_bounded() {
        let file = tempfile::NamedTempFile::new().unwrap();
        std::fs::write(file.path(), b"selected pixels").unwrap();
        let metadata = authority(b"unused").metadata;
        let preview = legacy_metadata(File::open(file.path()).unwrap(), "old.mp4".into(), metadata)
            .unwrap()
            .1;
        assert_eq!(preview.size_bytes, 15);
        assert!(preview.archive_identity_sha256.is_empty() && preview.checksum_sha256.is_empty());
        assert!(hash_exact(File::open(file.path()).unwrap(), 14).is_err());
        assert!(hash_exact(File::open(file.path()).unwrap(), 16).is_err());
        assert_eq!(
            hash_exact(File::open(file.path()).unwrap(), 15).unwrap(),
            format!("{:x}", Sha256::digest(b"selected pixels"))
        );
    }

    #[test]
    fn legacy_snapshot_binds_current_pixels_name_and_metadata_without_claiming_history() {
        let input = tempfile::NamedTempFile::new().unwrap();
        std::fs::write(input.path(), b"current selected movie").unwrap();
        let meta = authority(b"unused").metadata;
        let capture = |filename: &str, metadata| {
            snapshot_legacy(File::open(input.path()).unwrap(), filename.into(), metadata)
                .unwrap()
                .1
        };
        let first = capture("old.mp4", meta.clone());
        assert_eq!(
            first.authority_kind,
            mold_core::video_job::VideoSourceAuthorityKind::LegacySnapshot
        );
        assert_eq!(
            first.archive_identity_sha256,
            capture("old.mp4", meta.clone()).archive_identity_sha256
        );
        assert_ne!(
            first.archive_identity_sha256,
            capture("other.mp4", meta.clone()).archive_identity_sha256
        );
        let mut changed = meta.clone();
        changed.prompt.push_str(" edited");
        assert_ne!(
            first.archive_identity_sha256,
            capture("old.mp4", changed).archive_identity_sha256
        );
        std::fs::write(input.path(), b"replacement movie").unwrap();
        assert_ne!(
            first.archive_identity_sha256,
            capture("old.mp4", meta).archive_identity_sha256
        );
    }

    #[test]
    fn exact_owned_copy_survives_original_purge_and_never_replaces_a_snapshot() {
        let original = tempfile::tempdir().unwrap();
        let job = tempfile::tempdir().unwrap();
        let path = original.path().join("original.mp4");
        std::fs::write(&path, b"published pixels").unwrap();
        let file = File::open(&path).unwrap();
        std::fs::remove_file(&path).unwrap();
        retain(
            file,
            &authority(b"published pixels"),
            job.path(),
            &mut || Ok(()),
        )
        .unwrap();
        assert_eq!(
            std::fs::read(job.path().join("prefix.mp4")).unwrap(),
            b"published pixels"
        );
        std::fs::write(&path, b"other pixels").unwrap();
        assert!(retain(
            File::open(path).unwrap(),
            &authority(b"other pixels"),
            job.path(),
            &mut || Ok(())
        )
        .is_err());
        assert_eq!(
            std::fs::read(job.path().join("prefix.mp4")).unwrap(),
            b"published pixels"
        );
    }

    #[test]
    fn changed_or_cancelled_source_publishes_no_partial_copy() {
        let source = tempfile::NamedTempFile::new().unwrap();
        std::fs::write(source.path(), b"changed pixels").unwrap();
        let job = tempfile::tempdir().unwrap();
        assert!(retain(
            File::open(source.path()).unwrap(),
            &authority(b"original pixels"),
            job.path(),
            &mut || Ok(())
        )
        .is_err());
        assert!(retain(
            File::open(source.path()).unwrap(),
            &authority(b"changed pixels"),
            job.path(),
            &mut || anyhow::bail!("cancelled")
        )
        .is_err());
        assert_eq!(std::fs::read_dir(job.path()).unwrap().count(), 0);
    }
}
