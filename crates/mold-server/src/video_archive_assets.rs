//! Immutable file-backed provenance and exact PCM pinned to one published movie.

use crate::batch_transaction::ArchivedChildIdentity;
use anyhow::{ensure, Context, Result};
use mold_core::video_artifacts::VideoJobArtifacts;
#[cfg(any(feature = "mp4", test))]
use mold_core::video_artifacts::{VideoArtifactFile, VideoPcmArtifact};
#[cfg(any(feature = "mp4", test))]
use sha2::{Digest, Sha256};
use std::{
    io::Read,
    path::{Path, PathBuf},
};

#[cfg(any(feature = "mp4", test))]
use std::{
    fs::File,
    io::{Seek, Write},
};

pub(crate) fn root_from_db(db: &mold_db::MetadataDb) -> Result<PathBuf> {
    let home = db
        .path()
        .parent()
        .filter(|path| path.is_absolute())
        .context("video artifacts require a captured absolute Mold home")?;
    Ok(home.join("video-archive-assets"))
}

#[cfg(test)]
pub(crate) struct OpenArtifacts {
    // Pin verified provenance alongside PCM until this source read completes.
    _provenance: File,
    pub pcm: Option<File>,
}

#[cfg(any(feature = "mp4", test))]
fn private_directory(path: &Path) -> Result<()> {
    let mut builder = std::fs::DirBuilder::new();
    #[cfg(unix)]
    {
        use std::os::unix::fs::DirBuilderExt;
        builder.mode(0o700);
    }
    match builder.create(path) {
        Ok(()) => (),
        Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => (),
        Err(error) => return Err(error.into()),
    }
    verify_directory(path)
}
fn verify_directory(path: &Path) -> Result<()> {
    let metadata = std::fs::symlink_metadata(path)?;
    ensure!(
        metadata.is_dir() && !metadata.file_type().is_symlink(),
        "invalid artifact storage"
    );
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        ensure!(
            metadata.permissions().mode() & 0o077 == 0,
            "artifact storage is not private"
        );
    }
    Ok(())
}
fn directory(root: &Path, descriptor: &VideoJobArtifacts) -> Result<PathBuf> {
    descriptor.validate().map_err(anyhow::Error::msg)?;
    verify_directory(root)?;
    let path = root.join(&descriptor.bundle_id);
    verify_directory(&path)?;
    Ok(path)
}
fn read_json<T: serde::de::DeserializeOwned>(path: &Path) -> Result<T> {
    let input = mold_core::secure_file::open_regular_file_no_follow(path)?;
    ensure!(
        input.metadata()?.len() <= 16 * 1024,
        "artifact manifest exceeds metadata budget"
    );
    let mut bytes = Vec::new();
    input.take(16 * 1024 + 1).read_to_end(&mut bytes)?;
    ensure!(
        bytes.len() <= 16 * 1024,
        "artifact manifest grew beyond metadata budget"
    );
    Ok(serde_json::from_slice(&bytes)?)
}
#[cfg(any(feature = "mp4", test))]
fn write_json<T: serde::Serialize>(directory: &Path, name: &str, value: &T) -> Result<()> {
    let bytes = serde_json::to_vec(value)?;
    ensure!(
        bytes.len() <= 16 * 1024,
        "artifact manifest exceeds metadata budget"
    );
    let mut file = tempfile::NamedTempFile::new_in(directory)?;
    file.write_all(&bytes)?;
    file.as_file().sync_all()?;
    file.persist_noclobber(directory.join(name))?;
    crate::dir_sync::sync_directory(directory)?;
    Ok(())
}
#[cfg(any(feature = "mp4", test))]
fn copy_checked(
    mut input: File,
    mut output: Option<&mut File>,
    expected: &VideoArtifactFile,
    checkpoint: &mut dyn FnMut() -> Result<()>,
) -> Result<File> {
    input.rewind()?;
    let mut hash = Sha256::new();
    let mut size = 0u64;
    let mut buffer = [0u8; 64 * 1024];
    loop {
        checkpoint()?;
        let count = input.read(&mut buffer)?;
        if count == 0 {
            break;
        }
        size = size
            .checked_add(count as u64)
            .context("artifact size overflow")?;
        ensure!(
            size <= expected.size_bytes,
            "artifact grew beyond committed size"
        );
        hash.update(&buffer[..count]);
        if let Some(file) = output.as_mut() {
            file.write_all(&buffer[..count])?;
        }
    }
    ensure!(
        size == expected.size_bytes && format!("{:x}", hash.finalize()) == expected.sha256,
        "artifact differs from committed authority"
    );
    input.rewind()?;
    Ok(input)
}
#[cfg(any(feature = "mp4", test))]
fn checked_file(
    directory: &Path,
    name: &str,
    expected: &VideoArtifactFile,
    checkpoint: &mut dyn FnMut() -> Result<()>,
) -> Result<File> {
    copy_checked(
        mold_core::secure_file::open_regular_file_no_follow(&directory.join(name))?,
        None,
        expected,
        checkpoint,
    )
}
fn verify_descriptor(directory: &Path, descriptor: &VideoJobArtifacts) -> Result<()> {
    let recorded: VideoJobArtifacts = read_json(&directory.join("manifest.json"))?;
    ensure!(
        recorded == *descriptor,
        "artifact bundle belongs to another descriptor"
    );
    Ok(())
}
/// Stage immutable assets before publishing their descriptor in Gallery metadata.
#[cfg(any(feature = "mp4", test))]
pub(crate) fn prepare(
    root: &Path,
    job_id: &str,
    provenance: (File, VideoArtifactFile),
    pcm: Option<(File, VideoPcmArtifact)>,
    checkpoint: &mut dyn FnMut() -> Result<()>,
) -> Result<VideoJobArtifacts> {
    private_directory(root)?;
    let mut hash = Sha256::new();
    hash.update(b"mold-video-archive-assets-v1\0");
    hash.update(serde_json::to_vec(&(
        job_id,
        &provenance.1,
        pcm.as_ref().map(|(_, descriptor)| descriptor),
    ))?);
    let descriptor = VideoJobArtifacts {
        version: 1,
        bundle_id: format!("{:x}", hash.finalize()),
        provenance: provenance.1,
        pcm: pcm.as_ref().map(|(_, descriptor)| descriptor.clone()),
    };
    descriptor.validate().map_err(anyhow::Error::msg)?;
    let destination = root.join(&descriptor.bundle_id);
    if destination.exists() {
        let directory = directory(root, &descriptor)?;
        verify_descriptor(&directory, &descriptor)?;
        checked_file(
            &directory,
            "provenance.jsonl",
            &descriptor.provenance,
            checkpoint,
        )?;
        if let Some(pcm) = &descriptor.pcm {
            checked_file(&directory, "final-audio.pcm", &pcm.file, checkpoint)?;
        }
        return Ok(descriptor);
    }
    let staged = tempfile::Builder::new()
        .prefix(".assets-")
        .tempdir_in(root)?;
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        std::fs::set_permissions(staged.path(), std::fs::Permissions::from_mode(0o700))?;
    }
    let mut provenance_out = tempfile::NamedTempFile::new_in(staged.path())?;
    copy_checked(
        provenance.0,
        Some(provenance_out.as_file_mut()),
        &descriptor.provenance,
        checkpoint,
    )?;
    provenance_out.as_file().sync_all()?;
    provenance_out.persist_noclobber(staged.path().join("provenance.jsonl"))?;
    if let Some((file, pcm)) = pcm {
        let mut output = tempfile::NamedTempFile::new_in(staged.path())?;
        copy_checked(file, Some(output.as_file_mut()), &pcm.file, checkpoint)?;
        output.as_file().sync_all()?;
        output.persist_noclobber(staged.path().join("final-audio.pcm"))?;
    }
    write_json(staged.path(), "manifest.json", &descriptor)?;
    checkpoint()?;
    match std::fs::rename(staged.path(), &destination) {
        Ok(()) => crate::dir_sync::sync_directory(root)?,
        Err(error) => {
            if !destination.exists() {
                return Err(error.into());
            }
            let directory = directory(root, &descriptor)?;
            verify_descriptor(&directory, &descriptor)?;
            checked_file(
                &directory,
                "provenance.jsonl",
                &descriptor.provenance,
                checkpoint,
            )?;
            if let Some(pcm) = &descriptor.pcm {
                checked_file(&directory, "final-audio.pcm", &pcm.file, checkpoint)?;
            }
        }
    }
    Ok(descriptor)
}
/// Bind only after Gallery committed this exact output identity.
#[cfg(any(feature = "mp4", test))]
pub(crate) fn seal(
    root: &Path,
    descriptor: &VideoJobArtifacts,
    owner: &ArchivedChildIdentity,
) -> Result<()> {
    let directory = directory(root, descriptor)?;
    verify_descriptor(&directory, descriptor)?;
    match write_json(&directory, "owner.json", owner) {
        Ok(()) => Ok(()),
        Err(error) => {
            let existing: ArchivedChildIdentity = read_json(&directory.join("owner.json"))?;
            ensure!(
                existing == *owner,
                "artifact bundle is already sealed to another output: {error}"
            );
            Ok(())
        }
    }
}
#[cfg(test)]
pub(crate) fn open(
    root: &Path,
    descriptor: &VideoJobArtifacts,
    owner: &ArchivedChildIdentity,
) -> Result<OpenArtifacts> {
    open_with_checkpoint(root, descriptor, owner, &mut || Ok(()))
}

#[cfg(test)]
pub(crate) fn open_with_checkpoint(
    root: &Path,
    descriptor: &VideoJobArtifacts,
    owner: &ArchivedChildIdentity,
    checkpoint: &mut dyn FnMut() -> Result<()>,
) -> Result<OpenArtifacts> {
    let directory = directory(root, descriptor)?;
    verify_descriptor(&directory, descriptor)?;
    let existing: ArchivedChildIdentity = read_json(&directory.join("owner.json"))?;
    ensure!(
        existing == *owner,
        "artifact owner differs from validated Gallery identity"
    );
    Ok(OpenArtifacts {
        _provenance: checked_file(
            &directory,
            "provenance.jsonl",
            &descriptor.provenance,
            checkpoint,
        )?,
        pcm: descriptor
            .pcm
            .as_ref()
            .map(|pcm| checked_file(&directory, "final-audio.pcm", &pcm.file, checkpoint))
            .transpose()?,
    })
}
/// Retained prefix metadata binds this immutable archive identity even if its
/// original Library filename has moved to Trash since admission.
#[cfg(all(test, feature = "mp4"))]
pub(crate) fn open_for_source(
    root: &Path,
    descriptor: &VideoJobArtifacts,
    archive_identity_sha256: &str,
    checkpoint: &mut dyn FnMut() -> Result<()>,
) -> Result<OpenArtifacts> {
    let path = directory(root, descriptor)?;
    let owner: ArchivedChildIdentity = read_json(&path.join("owner.json"))?;
    ensure!(
        crate::gallery_source_media::archive_identity_sha256(&owner)
            .map_err(|error| anyhow::anyhow!(error.error))?
            == archive_identity_sha256,
        "PCM artifact belongs to another source archive"
    );
    open_with_checkpoint(root, descriptor, &owner, checkpoint)
}

/// Called only after permanent Gallery authority deletion. Trash retains assets.
pub(crate) fn release(
    root: &Path,
    descriptor: &VideoJobArtifacts,
    owner: &ArchivedChildIdentity,
) -> Result<bool> {
    let directory = directory(root, descriptor)?;
    verify_descriptor(&directory, descriptor)?;
    let existing: ArchivedChildIdentity = read_json(&directory.join("owner.json"))?;
    if existing != *owner {
        return Ok(false);
    }
    std::fs::remove_dir_all(directory)?;
    crate::dir_sync::sync_directory(root)?;
    Ok(true)
}

#[cfg(test)]
mod tests {
    use super::*;
    use sha2::{Digest, Sha256};
    use std::{fs::File, io::Write};

    fn input(bytes: &[u8]) -> (File, mold_core::video_artifacts::VideoArtifactFile) {
        let mut file = tempfile::tempfile().unwrap();
        file.write_all(bytes).unwrap();
        (
            file,
            mold_core::video_artifacts::VideoArtifactFile {
                sha256: format!("{:x}", Sha256::digest(bytes)),
                size_bytes: bytes.len() as u64,
            },
        )
    }
    fn owner(name: &str) -> crate::batch_transaction::ArchivedChildIdentity {
        crate::batch_transaction::ArchivedChildIdentity {
            parent_id: "parent".into(),
            attempt_generation: 1,
            child_index: 0,
            final_name: name.into(),
            checksum_sha256: "a".repeat(64),
            size_bytes: 20,
        }
    }
    #[test]
    fn published_artifacts_outlive_job_and_require_exact_archive_owner() {
        let home = tempfile::tempdir().unwrap();
        let root = home.path().join("assets");
        let descriptor = prepare(
            &root,
            "job",
            input(b"one bounded provenance line\n"),
            None,
            &mut || Ok(()),
        )
        .unwrap();
        assert!(open(&root, &descriptor, &owner("movie.mp4")).is_err());
        seal(&root, &descriptor, &owner("movie.mp4")).unwrap();
        seal(&root, &descriptor, &owner("movie.mp4")).unwrap();
        assert!(seal(&root, &descriptor, &owner("other.mp4")).is_err());
        assert!(open(&root, &descriptor, &owner("other.mp4")).is_err());
        let retained = open(&root, &descriptor, &owner("movie.mp4")).unwrap();
        assert!(!release(&root, &descriptor, &owner("other.mp4")).unwrap());
        assert!(release(&root, &descriptor, &owner("movie.mp4")).unwrap());
        assert!(retained._provenance.metadata().unwrap().len() > 0);
        assert!(open(&root, &descriptor, &owner("movie.mp4")).is_err());
    }
    #[test]
    fn replay_and_exact_asset_reads_honor_cancellation() {
        let home = tempfile::tempdir().unwrap();
        let root = home.path().join("assets");
        let descriptor = prepare(&root, "job", input(b"data"), None, &mut || Ok(())).unwrap();
        seal(&root, &descriptor, &owner("movie.mp4")).unwrap();
        assert!(
            prepare(&root, "job", input(b"data"), None, &mut || anyhow::bail!(
                "cancelled"
            ))
            .is_err()
        );
        assert!(open_with_checkpoint(
            &root,
            &descriptor,
            &owner("movie.mp4"),
            &mut || anyhow::bail!("cancelled")
        )
        .is_err());
    }

    #[test]
    fn exact_pcm_is_streamed_and_retained_with_its_format() {
        let home = tempfile::tempdir().unwrap();
        let (pcm_file, file) = input(&[0; 32]);
        let pcm = mold_core::video_artifacts::VideoPcmArtifact {
            file,
            sample_rate: 48000,
            channels: 2,
            frames: 4,
        };
        let descriptor = prepare(
            &home.path().join("assets"),
            "audio-job",
            input(b"provenance"),
            Some((pcm_file, pcm.clone())),
            &mut || Ok(()),
        )
        .unwrap();
        assert_eq!(descriptor.pcm.as_ref(), Some(&pcm));
        seal(
            &home.path().join("assets"),
            &descriptor,
            &owner("audio.mp4"),
        )
        .unwrap();
        let opened = open(
            &home.path().join("assets"),
            &descriptor,
            &owner("audio.mp4"),
        )
        .unwrap();
        assert_eq!(opened.pcm.unwrap().metadata().unwrap().len(), 32);
    }

    #[test]
    fn cancelled_or_corrupt_artifacts_never_become_usable() {
        let home = tempfile::tempdir().unwrap();
        let root = home.path().join("assets");
        assert!(
            prepare(&root, "job", input(b"data"), None, &mut || anyhow::bail!(
                "cancelled"
            ))
            .is_err()
        );
        let descriptor = prepare(&root, "job", input(b"data"), None, &mut || Ok(())).unwrap();
        seal(&root, &descriptor, &owner("movie.mp4")).unwrap();
        std::fs::write(
            root.join(&descriptor.bundle_id).join("provenance.jsonl"),
            b"evil",
        )
        .unwrap();
        assert!(open(&root, &descriptor, &owner("movie.mp4")).is_err());
    }
}
