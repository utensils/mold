//! Exact published-video authority retained independently of Library deletion.
use anyhow::{ensure, Context};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::{
    fs::File,
    io::{Read, Seek, SeekFrom, Write},
    path::Path,
};

use crate::{routes::ApiError, state::AppState};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub(crate) struct RetainedVideoSource {
    pub filename: String,
    pub archive_identity_sha256: String,
    pub checksum_sha256: String,
    pub size_bytes: u64,
    pub metadata: mold_core::OutputMetadata,
}

/// Open under the publication reader before releasing the original name.
/// The descriptor remains valid if the user subsequently trashes or purges it.
pub(crate) async fn open(
    state: &AppState,
    filename: &str,
    expected_identity: Option<&str>,
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
        let crate::batch_transaction::ValidatedRetainedMedia::Present {
            identity, metadata, ..
        } = resolved
        else {
            return Err(ApiError::not_found(
                "The selected video is no longer in Library",
            ));
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

/// Stream the original into a private, immutable job-owned file. Never route
/// the growing published video through an inline-reference upload budget.
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
            filename: "original.mp4".into(),
            archive_identity_sha256: "archive".into(),
            checksum_sha256: format!("{:x}", Sha256::digest(bytes)),
            size_bytes: bytes.len() as u64,
            metadata: mold_core::OutputMetadata::from_generate_request(&request, 1, None, "test"),
        }
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
