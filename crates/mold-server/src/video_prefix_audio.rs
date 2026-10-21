//! Canonical source sound belongs to the accepted parent, never a mutable
//! Library bundle. Every new attempt derives it from the exact input MP4.
use anyhow::{ensure, Context, Result};
use mold_core::{
    video_artifacts::{VideoArtifactFile, VideoPcmArtifact},
    video_job::VideoJobPlan,
};
use mold_inference::video_assembly_audio::{PcmFormat, PcmSpool};
use sha2::{Digest, Sha256};
use std::{
    fs::File,
    io::{Read, Seek},
    path::Path,
};

pub(crate) const FORMAT: PcmFormat = PcmFormat {
    sample_rate: 48_000,
    channels: 2,
};

fn describe(
    file: &mut File,
    checkpoint: &mut dyn FnMut() -> Result<()>,
) -> Result<VideoArtifactFile> {
    let expected = file.metadata()?.len();
    file.rewind()?;
    let mut digest = Sha256::new();
    let mut count = 0u64;
    let mut buffer = [0; 64 * 1024];
    loop {
        checkpoint()?;
        let length = file.read(&mut buffer)?;
        if length == 0 {
            break;
        }
        count = count
            .checked_add(length as u64)
            .context("prefix audio size overflow")?;
        ensure!(count <= expected, "prefix audio grew during verification");
        digest.update(&buffer[..length]);
    }
    ensure!(
        count == expected && file.metadata()?.len() == expected,
        "prefix audio changed during verification"
    );
    file.rewind()?;
    Ok(VideoArtifactFile {
        sha256: format!("{:x}", digest.finalize()),
        size_bytes: count,
    })
}

pub(crate) fn validate(audio: &VideoPcmArtifact, source_frames: u64, fps: u32) -> Result<()> {
    let frames =
        mold_inference::av_media::aligned_sample_count(source_frames, fps, FORMAT.sample_rate)?;
    ensure!(
        audio.sample_rate == FORMAT.sample_rate
            && audio.channels == FORMAT.channels
            && audio.frames == frames
            && audio.file.size_bytes
                == frames
                    .checked_mul(8)
                    .context("prefix audio size overflow")?
            && audio.file.sha256.len() == 64
            && audio.file.sha256.bytes().all(|b| b.is_ascii_hexdigit()),
        "invalid sealed prefix audio descriptor"
    );
    Ok(())
}

pub(crate) fn seal(
    directory: &Path,
    plan: &VideoJobPlan,
    checkpoint: &mut dyn FnMut() -> Result<()>,
) -> Result<Option<VideoPcmArtifact>> {
    if plan.source_frames == 0 {
        return Ok(None);
    }
    let original =
        mold_core::secure_file::open_regular_file_no_follow(&directory.join("prefix.mp4"))?;
    if !mold_inference::video_assembly_audio::has_audio(original.try_clone()?, checkpoint)? {
        return Ok(None);
    }
    let frames = mold_inference::av_media::aligned_sample_count(
        plan.source_frames,
        plan.fps,
        FORMAT.sample_rate,
    )?;
    let staged = tempfile::NamedTempFile::new_in(directory)?;
    let mut spool = PcmSpool::new(staged.reopen()?, FORMAT)?;
    spool.append_mp4(original, tempfile::tempfile()?, 0, frames, 0, checkpoint)?;
    let mut file = spool.into_file()?;
    file.sync_all()?;
    let audio = VideoPcmArtifact {
        file: describe(&mut file, checkpoint)?,
        sample_rate: FORMAT.sample_rate,
        channels: FORMAT.channels,
        frames,
    };
    validate(&audio, plan.source_frames, plan.fps)?;
    checkpoint()?;
    staged.persist_noclobber(directory.join("prefix-audio.pcm"))?;
    crate::dir_sync::sync_directory(directory)?;
    Ok(Some(audio))
}

pub(crate) fn open(
    directory: &Path,
    audio: &VideoPcmArtifact,
    checkpoint: &mut dyn FnMut() -> Result<()>,
) -> Result<File> {
    let mut file =
        mold_core::secure_file::open_regular_file_no_follow(&directory.join("prefix-audio.pcm"))?;
    ensure!(
        describe(&mut file, checkpoint)? == audio.file,
        "accepted prefix audio differs from its immutable manifest"
    );
    Ok(file)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sealed_prefix_audio_refuses_same_size_byte_tampering() {
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("prefix-audio.pcm");
        let original = [0u8; 16];
        std::fs::write(&path, original).unwrap();
        let audio = VideoPcmArtifact {
            file: VideoArtifactFile {
                sha256: format!("{:x}", Sha256::digest(original)),
                size_bytes: original.len() as u64,
            },
            sample_rate: FORMAT.sample_rate,
            channels: FORMAT.channels,
            frames: 2,
        };
        open(directory.path(), &audio, &mut || Ok(())).unwrap();
        std::fs::write(path, [1u8; 16]).unwrap();
        let error = open(directory.path(), &audio, &mut || Ok(())).unwrap_err();
        assert!(error.to_string().contains("immutable manifest"));
    }
}
