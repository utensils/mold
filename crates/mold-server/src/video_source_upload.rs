//! Private, resumable destination ingress for a retained video prefix.
//! Uploads are byte streams, never inline reference blobs or caller paths.

use crate::video_source::RetainedVideoSource;
use anyhow::{ensure, Context, Result};
use mold_core::video_job::{VideoSourceUploadRef, VideoSourceUploadStatus};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::{
    fs::{File, OpenOptions},
    io::{Read, Seek, SeekFrom, Write},
    path::{Path, PathBuf},
};

pub(crate) const MAX_CHUNK_BYTES: usize = 4 * 1024 * 1024;
const MAX_MANIFEST_BYTES: u64 = 8 * 1024 * 1024;

#[derive(Serialize, Deserialize)]
struct Manifest {
    version: u32,
    instance: String,
    declaration_sha256: String,
    source: RetainedVideoSource,
    committed_bytes: u64,
    completed: bool,
    #[serde(default)]
    updated_at_ms: i64,
}

fn directory(root: &Path, id: &str) -> Result<PathBuf> {
    let uuid = uuid::Uuid::parse_str(id)?;
    ensure!(
        uuid.to_string() == id,
        "upload handle must be a canonical UUID"
    );
    Ok(root.join(id))
}

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
    verify_private_directory(path)
}

fn verify_private_directory(path: &Path) -> Result<()> {
    let metadata = std::fs::symlink_metadata(path)?;
    ensure!(
        metadata.is_dir() && !metadata.file_type().is_symlink(),
        "upload directory is not private storage"
    );
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        ensure!(
            metadata.permissions().mode() & 0o077 == 0,
            "upload directory permissions are not private"
        );
    }
    Ok(())
}

fn writable(path: &Path, create: bool) -> Result<File> {
    let mut options = OpenOptions::new();
    options.read(true).write(true);
    if create {
        options.create_new(true);
    }
    #[cfg(unix)]
    {
        use std::os::unix::fs::OpenOptionsExt;
        options.mode(0o600).custom_flags(libc::O_NOFOLLOW);
    }
    #[cfg(windows)]
    {
        use std::os::windows::fs::OpenOptionsExt;
        options.custom_flags(0x0020_0000);
    }
    let file = options.open(path)?;
    ensure!(file.metadata()?.is_file(), "upload is not a regular file");
    Ok(file)
}

fn save(directory: &Path, manifest: &Manifest) -> Result<()> {
    let bytes = serde_json::to_vec(manifest)?;
    ensure!(
        bytes.len() as u64 <= MAX_MANIFEST_BYTES,
        "source provenance exceeds metadata budget"
    );
    let mut staged = tempfile::NamedTempFile::new_in(directory)?;
    staged.write_all(&bytes)?;
    staged.as_file().sync_all()?;
    staged.persist(directory.join("manifest.json"))?;
    crate::dir_sync::sync_directory(directory)?;
    Ok(())
}

fn load_any(directory: &Path) -> Result<Manifest> {
    verify_private_directory(directory.parent().context("upload has no storage root")?)?;
    verify_private_directory(directory)?;
    let input =
        mold_core::secure_file::open_regular_file_no_follow(&directory.join("manifest.json"))?;
    ensure!(
        input.metadata()?.len() <= MAX_MANIFEST_BYTES,
        "source manifest exceeds metadata budget"
    );
    let mut bytes = Vec::new();
    input.take(MAX_MANIFEST_BYTES + 1).read_to_end(&mut bytes)?;
    ensure!(
        bytes.len() as u64 <= MAX_MANIFEST_BYTES,
        "source manifest exceeded metadata budget while reading"
    );
    let manifest: Manifest = serde_json::from_slice(&bytes)?;
    ensure!(
        manifest.version == 1,
        "upload destination installation changed"
    );
    ensure!(
        manifest.committed_bytes <= manifest.source.size_bytes,
        "invalid committed upload length"
    );
    Ok(manifest)
}

fn load(directory: &Path, instance: &str) -> Result<Manifest> {
    let manifest = load_any(directory)?;
    ensure!(
        manifest.instance == instance,
        "upload destination installation changed"
    );
    Ok(manifest)
}

pub(crate) fn now_ms() -> i64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|duration| duration.as_millis().try_into().unwrap_or(i64::MAX))
        .unwrap_or(0)
}

/// Caller holds the same upload stripe as ordinary ingress mutation.
pub(crate) fn expire(root: &Path, id: &str, inactive_before_ms: i64) -> Result<bool> {
    let directory = directory(root, id)?;
    let manifest = load_any(&directory)?;
    if manifest.updated_at_ms >= inactive_before_ms {
        return Ok(false);
    }
    std::fs::remove_dir_all(directory)?;
    crate::dir_sync::sync_directory(root)?;
    Ok(true)
}

/// Caller holds the per-upload actor lock for begin/append/complete/remove.
pub(crate) fn begin(
    root: &Path,
    instance: &str,
    id: &str,
    source: RetainedVideoSource,
) -> Result<VideoSourceUploadStatus> {
    crate::routes::validate_gallery_filename(&source.filename)
        .map_err(|_| anyhow::anyhow!("invalid source filename"))?;
    ensure!(
        source.size_bytes > 0
            && [&source.archive_identity_sha256, &source.checksum_sha256]
                .iter()
                .all(|hash| hash.len() == 64 && hash.bytes().all(|b| b.is_ascii_hexdigit())),
        "source requires positive size and SHA-256 authority"
    );
    private_directory(root)?;
    let declaration_sha256 = format!("{:x}", Sha256::digest(serde_json::to_vec(&source)?));
    let directory = directory(root, id)?;
    if directory.exists() {
        let manifest = load(&directory, instance)?;
        ensure!(
            manifest.declaration_sha256 == declaration_sha256,
            "upload UUID already belongs to another source"
        );
        return Ok(status(id, &manifest));
    }
    let staged = tempfile::Builder::new()
        .prefix(".upload-")
        .tempdir_in(root)?;
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        std::fs::set_permissions(staged.path(), std::fs::Permissions::from_mode(0o700))?;
    }
    let manifest = Manifest {
        version: 1,
        instance: instance.into(),
        declaration_sha256,
        source,
        committed_bytes: 0,
        completed: false,
        updated_at_ms: now_ms(),
    };
    writable(&staged.path().join("prefix.mp4"), true)?.sync_all()?;
    save(staged.path(), &manifest)?;
    std::fs::rename(staged.path(), &directory)?;
    crate::dir_sync::sync_directory(root)?;
    Ok(status(id, &manifest))
}

fn status(id: &str, manifest: &Manifest) -> VideoSourceUploadStatus {
    VideoSourceUploadStatus {
        handle: id.into(),
        offset_bytes: manifest.committed_bytes,
        completed: manifest.completed,
    }
}

pub(crate) fn append(
    root: &Path,
    instance: &str,
    id: &str,
    offset: u64,
    bytes: &[u8],
) -> Result<VideoSourceUploadStatus> {
    ensure!(
        !bytes.is_empty() && bytes.len() <= MAX_CHUNK_BYTES,
        "upload chunk must contain 1 to 4 MiB"
    );
    let directory = directory(root, id)?;
    let mut manifest = load(&directory, instance)?;
    ensure!(!manifest.completed, "completed upload is immutable");
    let end = offset
        .checked_add(bytes.len() as u64)
        .context("upload offset overflow")?;
    ensure!(
        end <= manifest.source.size_bytes,
        "upload exceeds declared size"
    );
    let mut file = writable(&directory.join("prefix.mp4"), false)?;
    ensure!(
        file.metadata()?.len() >= manifest.committed_bytes,
        "committed upload bytes disappeared"
    );
    file.set_len(manifest.committed_bytes)?;
    if offset < manifest.committed_bytes {
        ensure!(
            end <= manifest.committed_bytes,
            "upload retry overlaps uncommitted bytes"
        );
        file.seek(SeekFrom::Start(offset))?;
        let mut previous = vec![0; bytes.len()];
        file.read_exact(&mut previous)?;
        ensure!(
            previous == bytes,
            "upload retry differs from committed bytes"
        );
        return Ok(status(id, &manifest));
    }
    ensure!(
        offset == manifest.committed_bytes,
        "upload chunk is out of order"
    );
    file.seek(SeekFrom::Start(offset))?;
    file.write_all(bytes)?;
    file.sync_all()?;
    manifest.committed_bytes = end;
    manifest.updated_at_ms = now_ms();
    save(&directory, &manifest)?;
    Ok(status(id, &manifest))
}

pub(crate) fn complete(
    root: &Path,
    instance: &str,
    id: &str,
    checkpoint: &mut dyn FnMut() -> Result<()>,
) -> Result<VideoSourceUploadRef> {
    let directory = directory(root, id)?;
    let mut manifest = load(&directory, instance)?;
    ensure!(
        manifest.committed_bytes == manifest.source.size_bytes,
        "upload is incomplete"
    );
    let file = mold_core::secure_file::open_regular_file_no_follow(&directory.join("prefix.mp4"))?;
    ensure!(
        file.metadata()?.len() == manifest.source.size_bytes,
        "upload length changed"
    );
    let mut hash = Sha256::new();
    let limit = manifest
        .source
        .size_bytes
        .checked_add(1)
        .context("upload length cannot be represented")?;
    let mut input = file.try_clone()?.take(limit);
    let mut seen = 0u64;
    let mut buffer = [0; 64 * 1024];
    loop {
        checkpoint()?;
        let count = input.read(&mut buffer)?;
        if count == 0 {
            break;
        }
        seen = seen
            .checked_add(count as u64)
            .context("upload size overflow")?;
        ensure!(
            seen <= manifest.source.size_bytes,
            "upload grew during verification"
        );
        hash.update(&buffer[..count]);
    }
    ensure!(
        seen == manifest.source.size_bytes,
        "upload shrank during verification"
    );
    ensure!(
        format!("{:x}", hash.finalize()) == manifest.source.checksum_sha256,
        "upload checksum does not match its original source"
    );
    let facts = mold_inference::video_assembly_frames::inspect(file, checkpoint)?;
    // Pixel facts, not caller metadata, govern planning and carry windows.
    manifest.source.metadata.width = facts.width;
    manifest.source.metadata.height = facts.height;
    manifest.source.metadata.frames = Some(facts.frames);
    manifest.source.metadata.fps = (facts.fps_denominator == 1).then_some(facts.fps_numerator);
    manifest.completed = true;
    manifest.updated_at_ms = now_ms();
    save(&directory, &manifest)?;
    Ok(VideoSourceUploadRef {
        handle: id.into(),
        sha256: manifest.source.checksum_sha256,
        size_bytes: manifest.source.size_bytes,
    })
}

/// A same-installation restart can retain its immutable private prefix locally.
/// The caller supplies only a server-derived job path while holding its owner.
pub(crate) fn install_retained(
    root: &Path,
    instance: &str,
    id: &str,
    source_path: &Path,
    source: RetainedVideoSource,
    checkpoint: &mut dyn FnMut() -> Result<()>,
) -> Result<VideoSourceUploadRef> {
    private_directory(root)?;
    let destination = directory(root, id)?;
    if destination.exists() {
        begin(root, instance, id, source)?;
        return complete(root, instance, id, checkpoint);
    }
    let temporary_id = uuid::Uuid::new_v4().to_string();
    begin(root, instance, &temporary_id, source.clone())?;
    let staged = directory(root, &temporary_id)?;
    let result = (|| -> Result<VideoSourceUploadRef> {
        checkpoint()?;
        let input = mold_core::secure_file::open_regular_file_no_follow(source_path)?;
        ensure!(
            input.metadata()?.len() == source.size_bytes,
            "local prefix size changed"
        );
        let path = staged.join("prefix.mp4");
        std::fs::remove_file(&path)?;
        if std::fs::hard_link(source_path, &path).is_err() {
            let mut output = writable(&path, true)?;
            let mut reader = input.take(
                source
                    .size_bytes
                    .checked_add(1)
                    .context("prefix size overflow")?,
            );
            let mut buffer = [0u8; 64 * 1024];
            let mut size = 0u64;
            loop {
                checkpoint()?;
                let count = reader.read(&mut buffer)?;
                if count == 0 {
                    break;
                }
                size = size
                    .checked_add(count as u64)
                    .context("prefix size overflow")?;
                ensure!(
                    size <= source.size_bytes,
                    "local prefix grew during retention"
                );
                output.write_all(&buffer[..count])?;
            }
            ensure!(
                size == source.size_bytes,
                "local prefix shrank during retention"
            );
            output.sync_all()?;
        }
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            std::fs::set_permissions(&path, std::fs::Permissions::from_mode(0o600))?;
        }
        let mut manifest = load(&staged, instance)?;
        manifest.committed_bytes = source.size_bytes;
        save(&staged, &manifest)?;
        let mut reference = complete(root, instance, &temporary_id, checkpoint)?;
        match std::fs::rename(&staged, &destination) {
            Ok(()) => crate::dir_sync::sync_directory(root)?,
            Err(error) => {
                if !destination.exists() {
                    return Err(error.into());
                }
                begin(root, instance, id, source)?;
                return complete(root, instance, id, checkpoint);
            }
        }
        reference.handle = id.into();
        Ok(reference)
    })();
    if staged.exists() {
        let _ = std::fs::remove_dir_all(&staged);
        let _ = crate::dir_sync::sync_directory(root);
    }
    result
}

pub(crate) fn resolve(
    root: &Path,
    instance: &str,
    reference: &VideoSourceUploadRef,
    filename: &str,
    archive_identity: &str,
) -> Result<(File, RetainedVideoSource)> {
    let directory = directory(root, &reference.handle)?;
    let manifest = load(&directory, instance)?;
    ensure!(
        manifest.completed
            && manifest.source.filename == filename
            && manifest.source.archive_identity_sha256 == archive_identity
            && manifest.source.checksum_sha256 == reference.sha256
            && manifest.source.size_bytes == reference.size_bytes,
        "retained upload does not match the selected source"
    );
    let file = mold_core::secure_file::open_regular_file_no_follow(&directory.join("prefix.mp4"))?;
    ensure!(
        file.metadata()?.len() == reference.size_bytes,
        "retained upload length changed"
    );
    Ok((file, manifest.source))
}

#[cfg(test)]
mod tests {
    use super::*;
    use sha2::{Digest, Sha256};

    fn private_test_root() -> tempfile::TempDir {
        let root = tempfile::tempdir().unwrap();
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            std::fs::set_permissions(root.path(), std::fs::Permissions::from_mode(0o700)).unwrap();
        }
        root
    }

    fn authority(bytes: &[u8]) -> crate::video_source::RetainedVideoSource {
        let request = serde_json::from_value(serde_json::json!({"model":"ltx-2-19b-distilled:fp8", "prompt":"original", "width":32,"height":32,"steps":4})).unwrap();
        crate::video_source::RetainedVideoSource {
            authority_kind: mold_core::video_job::VideoSourceAuthorityKind::PublishedArchive,
            filename: "original.mp4".into(),
            archive_identity_sha256: "a".repeat(64),
            checksum_sha256: format!("{:x}", Sha256::digest(bytes)),
            size_bytes: bytes.len() as u64,
            metadata: mold_core::OutputMetadata::from_generate_request(&request, 1, None, "test"),
        }
    }

    #[test]
    fn fractional_cfr_prefixes_are_sealed_without_rounding_their_clock() {
        use mp4_rs::{
            AvcConfig, MediaConfig, Mp4Config, Mp4Sample, Mp4Writer, TrackConfig, TrackType,
        };
        let pictures = vec![image::RgbImage::from_pixel(32, 32, image::Rgb([40, 80, 100])); 3];
        let original = mold_inference::ltx_video::video_enc::encode_mp4(&pictures, 30).unwrap();
        let mut input =
            mp4_rs::Mp4Reader::read_header(std::io::Cursor::new(&original), original.len() as u64)
                .unwrap();
        let track = &input.tracks()[&1];
        let config = Mp4Config {
            major_brand: "isom".parse().unwrap(),
            minor_version: 0,
            compatible_brands: vec!["isom".parse().unwrap()],
            timescale: 30_000,
        };
        let mut writer = Mp4Writer::write_start(std::io::Cursor::new(Vec::new()), &config).unwrap();
        writer
            .add_track(&TrackConfig {
                track_type: TrackType::Video,
                timescale: 30_000,
                language: "und".into(),
                media_conf: MediaConfig::AvcConfig(AvcConfig {
                    width: 32,
                    height: 32,
                    seq_param_set: track.sequence_parameter_set().unwrap().to_vec(),
                    pic_param_set: track.picture_parameter_set().unwrap().to_vec(),
                }),
            })
            .unwrap();
        for index in 0..3 {
            let packet = input.read_sample(1, index + 1).unwrap().unwrap();
            writer
                .write_sample(
                    1,
                    &Mp4Sample {
                        start_time: u64::from(index) * 1001,
                        duration: 1001,
                        rendering_offset: 0,
                        is_sync: packet.is_sync,
                        bytes: packet.bytes,
                    },
                )
                .unwrap();
        }
        writer.write_end().unwrap();
        let bytes = writer.into_writer().into_inner();
        let root = private_test_root();
        let id = uuid::Uuid::new_v4().to_string();
        let descriptor = authority(&bytes);
        begin(root.path(), "owner", &id, descriptor.clone()).unwrap();
        append(root.path(), "owner", &id, 0, &bytes).unwrap();
        let reference = complete(root.path(), "owner", &id, &mut || Ok(())).unwrap();
        let (file, source) = resolve(
            root.path(),
            "owner",
            &reference,
            &descriptor.filename,
            &descriptor.archive_identity_sha256,
        )
        .unwrap();
        assert_eq!(source.metadata.fps, None);
        let facts = mold_inference::video_assembly_frames::inspect(file, &mut || Ok(())).unwrap();
        assert_eq!(
            (facts.frames, facts.fps_numerator, facts.fps_denominator),
            (3, 30_000, 1001)
        );
    }

    #[test]
    fn corrupted_bytes_and_unbounded_chunks_never_become_a_source() {
        let root = private_test_root();
        let id = uuid::Uuid::new_v4().to_string();
        begin(root.path(), "owner", &id, authority(b"original bytes")).unwrap();
        assert!(append(root.path(), "owner", &id, 0, &vec![0; MAX_CHUNK_BYTES + 1]).is_err());
        assert!(append(root.path(), "owner", &id, u64::MAX, b"overflow").is_err());
        append(root.path(), "owner", &id, 0, b"changed! bytes").unwrap();
        assert!(complete(root.path(), "owner", &id, &mut || Ok(())).is_err());
        assert!(!current(root.path(), "owner", &id).unwrap().completed);
        remove(root.path(), "owner", &id).unwrap();
        remove(root.path(), "owner", &id).unwrap();
    }

    #[cfg(unix)]
    #[test]
    fn upload_never_follows_a_redirected_directory_or_file() {
        use std::os::unix::fs::symlink;
        let root = private_test_root();
        let other = private_test_root();
        let id = uuid::Uuid::new_v4().to_string();
        symlink(other.path(), root.path().join(&id)).unwrap();
        assert!(begin(root.path(), "owner", &id, authority(b"original bytes")).is_err());
        assert!(std::fs::read_dir(other.path()).unwrap().next().is_none());
        std::fs::remove_file(root.path().join(&id)).unwrap();
        begin(root.path(), "owner", &id, authority(b"original bytes")).unwrap();
        let original = other.path().join("unrelated");
        std::fs::write(&original, b"leave unchanged").unwrap();
        let path = root.path().join(&id).join("prefix.mp4");
        std::fs::remove_file(&path).unwrap();
        symlink(&original, &path).unwrap();
        assert!(append(root.path(), "owner", &id, 0, b"original bytes").is_err());
        assert_eq!(std::fs::read(original).unwrap(), b"leave unchanged");
    }

    #[test]
    fn sequential_upload_reconciles_lost_replies_and_uncommitted_tail() {
        let root = private_test_root();
        let id = uuid::Uuid::new_v4().to_string();
        let descriptor = authority(b"0123456789");
        assert_eq!(
            begin(root.path(), "owner", &id, descriptor.clone())
                .unwrap()
                .offset_bytes,
            0
        );
        assert_eq!(
            append(root.path(), "owner", &id, 0, b"01234")
                .unwrap()
                .offset_bytes,
            5
        );
        assert_eq!(
            append(root.path(), "owner", &id, 0, b"01234")
                .unwrap()
                .offset_bytes,
            5
        );
        assert!(append(root.path(), "owner", &id, 0, b"WRONG").is_err());
        assert!(append(root.path(), "replacement", &id, 5, b"56789").is_err());
        let path = root.path().join(&id).join("prefix.mp4");
        use std::io::Write;
        std::fs::OpenOptions::new()
            .append(true)
            .open(&path)
            .unwrap()
            .write_all(b"crash")
            .unwrap();
        assert_eq!(
            append(root.path(), "owner", &id, 5, b"56789")
                .unwrap()
                .offset_bytes,
            10
        );
        assert_eq!(std::fs::read(path).unwrap(), b"0123456789");
        assert!(append(root.path(), "owner", &id, 10, b"overflow").is_err());
        assert!(begin(root.path(), "owner", &id, authority(b"replacement")).is_err());
        assert!(begin(root.path(), "owner", "../escape", descriptor).is_err());
    }

    #[test]
    fn source_ingress_expires_only_after_inactivity_and_never_follows_unknown_storage() {
        let root = private_test_root();
        let id = uuid::Uuid::new_v4().to_string();
        begin(root.path(), "owner", &id, authority(b"pixels")).unwrap();
        assert!(!expire(root.path(), &id, 0).unwrap());
        let dir = directory(root.path(), &id).unwrap();
        let mut manifest = load(&dir, "owner").unwrap();
        manifest.updated_at_ms = 1;
        save(&dir, &manifest).unwrap();
        assert!(expire(root.path(), &id, 2).unwrap());
        assert!(!dir.exists());
        assert!(expire(root.path(), "../unrelated", i64::MAX).is_err());
        let other = root.path().join("unrelated");
        std::fs::create_dir(&other).unwrap();
        assert!(expire(root.path(), "unrelated", i64::MAX).is_err());
        assert!(other.exists());
    }

    #[test]
    fn same_host_restart_retains_prefix_without_client_relay_or_original_lifetime() {
        let frames = vec![image::RgbImage::from_pixel(32, 32, image::Rgb([60, 80, 100])); 3];
        let bytes = mold_inference::ltx_video::video_enc::encode_mp4(&frames, 24).unwrap();
        let original = tempfile::tempdir().unwrap();
        let path = original.path().join("prefix.mp4");
        std::fs::write(&path, &bytes).unwrap();
        let root = private_test_root();
        let id = uuid::Uuid::new_v4().to_string();
        let source = authority(&bytes);
        let reference = install_retained(
            root.path(),
            "owner",
            &id,
            &path,
            source.clone(),
            &mut || Ok(()),
        )
        .unwrap();
        std::fs::remove_file(path).unwrap();
        let (mut file, _) = resolve(
            root.path(),
            "owner",
            &reference,
            &source.filename,
            &source.archive_identity_sha256,
        )
        .unwrap();
        let mut retained = Vec::new();
        file.read_to_end(&mut retained).unwrap();
        assert_eq!(retained, bytes);
        assert!(append(root.path(), "owner", &id, 0, b"mutated").is_err());
        assert_eq!(
            install_retained(
                root.path(),
                "owner",
                &id,
                &original.path().join("gone"),
                source,
                &mut || Ok(())
            )
            .unwrap(),
            reference
        );
    }

    #[test]
    fn completion_authenticates_every_byte_before_resolving_a_source() {
        let frames = vec![image::RgbImage::from_pixel(32, 32, image::Rgb([60, 80, 100])); 3];
        let bytes = mold_inference::ltx_video::video_enc::encode_mp4(&frames, 24).unwrap();
        let root = private_test_root();
        let id = uuid::Uuid::new_v4().to_string();
        let descriptor = authority(&bytes);
        begin(root.path(), "owner", &id, descriptor.clone()).unwrap();
        assert!(complete(root.path(), "owner", &id, &mut || Ok(())).is_err());
        append(root.path(), "owner", &id, 0, &bytes).unwrap();
        assert!(complete(root.path(), "owner", &id, &mut || anyhow::bail!(
            "cancelled"
        ))
        .is_err());
        assert!(!current(root.path(), "owner", &id).unwrap().completed);
        let reference = complete(root.path(), "owner", &id, &mut || Ok(())).unwrap();
        assert!(
            begin(root.path(), "owner", &id, descriptor.clone())
                .unwrap()
                .completed
        );
        let (mut file, retained) = resolve(
            root.path(),
            "owner",
            &reference,
            "original.mp4",
            &descriptor.archive_identity_sha256,
        )
        .unwrap();
        use std::io::Read;
        let mut actual = Vec::new();
        file.read_to_end(&mut actual).unwrap();
        assert_eq!(actual, bytes);
        assert_eq!(retained.checksum_sha256, descriptor.checksum_sha256);
        assert!(resolve(
            root.path(),
            "replacement",
            &reference,
            "original.mp4",
            &descriptor.archive_identity_sha256
        )
        .is_err());
        assert!(resolve(
            root.path(),
            "owner",
            &reference,
            "other.mp4",
            &descriptor.archive_identity_sha256
        )
        .is_err());
        assert!(append(root.path(), "owner", &id, 0, &bytes).is_err());
        assert_eq!(
            complete(root.path(), "owner", &id, &mut || Ok(())).unwrap(),
            reference
        );
    }
}

pub(crate) fn current(root: &Path, instance: &str, id: &str) -> Result<VideoSourceUploadStatus> {
    Ok(status(id, &load(&directory(root, id)?, instance)?))
}

pub(crate) fn remove(root: &Path, instance: &str, id: &str) -> Result<()> {
    let directory = directory(root, id)?;
    if !directory.exists() {
        return Ok(());
    }
    load(&directory, instance)?;
    std::fs::remove_dir_all(directory)?;
    crate::dir_sync::sync_directory(root)?;
    Ok(())
}

pub(crate) async fn open(
    state: &crate::state::AppState,
    reference: &VideoSourceUploadRef,
    source: &mold_core::video_job::VideoJobSource,
) -> std::result::Result<(File, RetainedVideoSource), crate::routes::ApiError> {
    let root = crate::video_source_upload_routes::root(state)?;
    let instance = state.instance_id.to_string();
    let reference = reference.clone();
    let source = source.clone();
    tokio::task::spawn_blocking(move || {
        ensure!(
            source.retained_upload.as_ref() == Some(&reference),
            "source upload binding changed"
        );
        resolve(
            &root,
            &instance,
            &reference,
            &source.filename,
            source
                .expected_archive_identity_sha256
                .as_deref()
                .context("missing original source identity")?,
        )
    })
    .await
    .map_err(|_| crate::routes::ApiError::internal("Could not open retained video upload"))?
    .map_err(crate::video_source_upload_routes::upload_error)
}
