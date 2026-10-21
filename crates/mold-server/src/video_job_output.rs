//! Append-once, file-backed output owned by one durable video parent.
use crate::{
    state::AppState,
    video_job_manifest::{job_dir, VideoJobManifest},
};
use anyhow::{ensure, Context, Result};
use image::RgbImage;
use mold_core::{
    chain::ChainIncomingSource, chain_job::ChainJobManifest, video_job::VideoJoinMode,
    video_job_schedule::VideoJobPartPlan,
};
use mold_db::{chain_jobs::ChainJobRow, video_jobs::StoredVideoJob};
use mold_inference::{
    video_assembly::Mp4FileAssembly,
    video_assembly_audio::{PcmFormat, PcmSpool},
    video_assembly_frames as frames,
};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::{
    fs::File,
    io::{Read, Seek, Write},
    path::{Path, PathBuf},
};

fn published_authoring(
    manifest: &VideoJobManifest,
) -> Result<mold_core::video_job::VideoJobAuthoring> {
    let boundary_frames = manifest
        .template
        .stages
        .first()
        .context("video template has no input stage")?
        .frames;
    let authoring = mold_core::video_job::VideoJobAuthoring {
        duration_ms: manifest.request.duration_ms,
        scenes: manifest.request.scenes.clone(),
        boundary_frames,
        input_frames: manifest
            .request
            .generation
            .frames
            .filter(|frames| *frames != boundary_frames),
        seed: manifest
            .template
            .seed
            .context("parent has no frozen seed")?,
    };
    authoring.validate().map_err(anyhow::Error::msg)?;
    Ok(authoring)
}

const PCM: PcmFormat = crate::video_prefix_audio::FORMAT;
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct PartReceipt {
    #[serde(skip)]
    pub(crate) receipt_sha256: String,
    pub sha256: String,
    pub segment_frames: u64,
    version: u32,
    manifest_sha256: String,
    index: u64,
    child_id: String,
    child_manifest_sha256: String,
    audio: Option<mold_core::video_artifacts::VideoPcmArtifact>,
    prompt: String,
    seed: u64,
    scene_index: usize,
    generated_frames: u32,
    incoming_overlap_frames: u32,
    end_trim_frames: u32,
    preserved_closing_frame: bool,
}

pub(crate) struct VideoJobOutput {
    assembly: Option<Mp4FileAssembly>,
    video: File,
    audio: Option<PcmSpool>,
    provenance: File,
    prefix: Option<File>,
    format_probe: File,
    appended: u64,
    visible_frames: u64,
    poisoned: bool,
}

fn db(state: &AppState) -> Result<&mold_db::MetadataDb> {
    state
        .metadata_db
        .as_ref()
        .as_ref()
        .context("video output requires durable storage")
}
fn open(path: &Path) -> Result<File> {
    mold_core::secure_file::open_regular_file_no_follow(path)
}
fn hash(
    file: &mut File,
    checkpoint: &mut dyn FnMut() -> Result<()>,
) -> Result<mold_core::video_artifacts::VideoArtifactFile> {
    let size = file.metadata()?.len();
    file.rewind()?;
    let mut input = (&mut *file).take(size.checked_add(1).context("file length overflow")?);
    let mut hash = Sha256::new();
    let mut count = 0u64;
    let mut bytes = [0u8; 64 * 1024];
    loop {
        checkpoint()?;
        let read = input.read(&mut bytes)?;
        if read == 0 {
            break;
        }
        count = count
            .checked_add(read as u64)
            .context("file length overflow")?;
        ensure!(count <= size, "private media grew during verification");
        hash.update(&bytes[..read]);
    }
    ensure!(
        count == size && file.metadata()?.len() == size,
        "private media changed during verification"
    );
    file.rewind()?;
    Ok(mold_core::video_artifacts::VideoArtifactFile {
        sha256: format!("{:x}", hash.finalize()),
        size_bytes: size,
    })
}
fn part_paths(directory: &Path, index: u64) -> (PathBuf, PathBuf, PathBuf) {
    let base = directory.join("parts").join(format!("{index:020}"));
    (
        base.with_extension("mp4"),
        base.with_extension("json"),
        base.with_extension("pcm"),
    )
}
fn relative(root: &Path, value: &str) -> Result<PathBuf> {
    ensure!(
        !value.contains('\\') && !value.contains(':') && !value.is_empty(),
        "invalid child artifact name"
    );
    ensure!(
        Path::new(value)
            .components()
            .all(|part| matches!(part, std::path::Component::Normal(_))),
        "invalid child artifact path"
    );
    Ok(root.join(value))
}
fn read_receipt(path: &Path, checkpoint: &mut dyn FnMut() -> Result<()>) -> Result<PartReceipt> {
    let mut file = open(path)?;
    ensure!(
        file.metadata()?.len() <= 8 * 1024 * 1024,
        "part receipt exceeds budget"
    );
    let descriptor = hash(&mut file, checkpoint)?;
    let mut value: PartReceipt = serde_json::from_reader(file.take(8 * 1024 * 1024 + 1))?;
    value.receipt_sha256 = descriptor.sha256;
    Ok(value)
}
fn save_receipt(
    path: &Path,
    value: &PartReceipt,
    checkpoint: &mut dyn FnMut() -> Result<()>,
) -> Result<String> {
    let directory = path.parent().context("missing part directory")?;
    let mut staged = tempfile::NamedTempFile::new_in(directory)?;
    serde_json::to_writer(&mut staged, value)?;
    staged.flush()?;
    staged.as_file().sync_all()?;
    let descriptor = hash(&mut staged.reopen()?, checkpoint)?;
    ensure!(
        descriptor.size_bytes <= 8 * 1024 * 1024,
        "part receipt exceeds budget"
    );
    staged.persist(path)?;
    crate::dir_sync::sync_directory(directory)?;
    Ok(descriptor.sha256)
}
fn persist(staged: tempfile::NamedTempFile, path: &Path) -> Result<()> {
    staged.as_file().sync_all()?;
    staged.persist(path)?;
    crate::dir_sync::sync_directory(path.parent().context("missing artifact directory")?)?;
    Ok(())
}
fn samples(frames: u64, fps: u32) -> Result<u64> {
    if frames == 0 {
        Ok(0)
    } else {
        mold_inference::av_media::aligned_sample_count(frames, fps, PCM.sample_rate)
    }
}
fn compatible(
    file: File,
    probe: &File,
    checkpoint: &mut dyn FnMut() -> Result<()>,
) -> Result<bool> {
    Mp4FileAssembly::compatible_parts(file, probe.try_clone()?, checkpoint)
}

impl VideoJobOutput {
    /// Call from blocking IO work, without holding a configuration lock.
    pub(crate) fn open(
        state: &AppState,
        row: &StoredVideoJob,
        manifest: &VideoJobManifest,
        checkpoint: &mut dyn FnMut() -> Result<()>,
    ) -> Result<Self> {
        checkpoint()?;
        let video = tempfile::tempfile()?;
        let mut probe = tempfile::tempfile()?;
        let mut encoder = mold_inference::ltx_video::video_enc::Mp4FileEncoder::new(
            probe.try_clone()?,
            manifest.plan.width,
            manifest.plan.height,
            manifest.plan.fps,
        )?;
        encoder.push(&RgbImage::new(manifest.plan.width, manifest.plan.height))?;
        encoder.finish()?;
        probe.rewind()?;
        let audio_enabled = manifest.prefix_audio.is_some()
            || (manifest.template.enable_audio == Some(true)
                && manifest.request.generation.video_only != Some(true));
        let mut result = Self {
            assembly: Some(Mp4FileAssembly::new(video.try_clone()?)),
            video,
            audio: audio_enabled
                .then(|| PcmSpool::new(tempfile::tempfile()?, PCM))
                .transpose()?,
            provenance: tempfile::tempfile()?,
            prefix: None,
            format_probe: probe,
            appended: 0,
            visible_frames: 0,
            poisoned: false,
        };
        let directory = job_dir(state, &row.job.id)?;
        if let Some(source) = &manifest.retained_source {
            let mut original = open(&directory.join("prefix.mp4"))?;
            let descriptor = hash(&mut original, checkpoint)?;
            ensure!(
                descriptor.sha256 == source.checksum_sha256
                    && descriptor.size_bytes == source.size_bytes,
                "retained prefix differs from accepted authority"
            );
            let facts = frames::inspect(original.try_clone()?, checkpoint)?;
            ensure!(
                mold_inference::video_assembly_audio::has_audio(original.try_clone()?, checkpoint)?
                    == manifest.prefix_audio.is_some(),
                "accepted source audio snapshot is unavailable; start a new attempt to seal its original sound"
            );
            let keep = u32::try_from(manifest.plan.source_frames)
                .context("prefix exceeds MP4 frame representation")?;
            let canonical = if facts.frames == keep
                && (facts.width, facts.height) == (manifest.plan.width, manifest.plan.height)
                && facts.fps_numerator == manifest.plan.fps
                && facts.fps_denominator == 1
                && compatible(original.try_clone()?, &result.format_probe, checkpoint)?
            {
                original.try_clone()?
            } else {
                let file = tempfile::tempfile()?;
                frames::normalize(
                    original.try_clone()?,
                    file.try_clone()?,
                    facts.frames,
                    keep,
                    manifest.plan.width,
                    manifest.plan.height,
                    manifest.plan.fps,
                    &[],
                    checkpoint,
                )?;
                file
            };
            let fade = if manifest.plan.join_mode == VideoJoinMode::BlendedShots {
                manifest.envelope.overlap_frames
            } else {
                0
            };
            result
                .assembly
                .as_mut()
                .context("missing assembly")?
                .append(
                    canonical.try_clone()?,
                    keep.checked_sub(fade)
                        .context("source too short for fade")?,
                    checkpoint,
                )?;
            if let Some(audio) = result.audio.as_mut() {
                let target = samples(manifest.plan.source_frames, manifest.plan.fps)?;
                if let Some(descriptor) = &manifest.prefix_audio {
                    let pcm = crate::video_prefix_audio::open(&directory, descriptor, checkpoint)?;
                    audio.append_pcm_conformed(pcm, PCM, 0, target, 0, checkpoint)?;
                } else {
                    audio.append_pcm(&mut std::io::empty(), target, 0, checkpoint)?;
                }
            }
            serde_json::to_writer(
                &mut result.provenance,
                &serde_json::json!({"source_archive_identity_sha256":source.archive_identity_sha256,"source_sha256":source.checksum_sha256,"source_frames":manifest.plan.source_frames.to_string()}),
            )?;
            result.provenance.write_all(b"\n")?;
            result.prefix = Some(canonical);
            result.visible_frames = manifest.plan.source_frames;
        }
        for index in 0..row.job.completed_clips {
            checkpoint()?;
            let ledger = mold_db::video_jobs::get_part(db(state)?, &row.job.id, index)?
                .context("completed parent part is missing")?;
            let (video, receipt, _) = part_paths(&directory, index);
            let receipt = read_receipt(&receipt, checkpoint)?;
            ensure!(
                receipt.manifest_sha256 == row.manifest_sha256 && receipt.index == index,
                "sealed part belongs to another parent"
            );
            ensure!(
                ledger.receipt_sha256.as_deref() == Some(receipt.receipt_sha256.as_str()),
                "sealed part receipt differs from committed authority"
            );
            ensure!(
                ledger.output_sha256.as_deref() == Some(receipt.sha256.as_str())
                    && ledger.child_id == receipt.child_id
                    && ledger.keep_frames == receipt.segment_frames,
                "part ledger differs from sealed output"
            );
            result.append_sealed(manifest, &directory, &video, &receipt, checkpoint)?;
        }
        Ok(result)
    }

    fn visible_tail(
        &self,
        state: &AppState,
        row: &StoredVideoJob,
        manifest: &VideoJobManifest,
        count: u32,
        checkpoint: &mut dyn FnMut() -> Result<()>,
    ) -> Result<Vec<RgbImage>> {
        ensure!(
            u64::from(count) <= self.visible_frames,
            "movie is shorter than its continuation boundary"
        );
        ensure!(
            u64::from(count) * u64::from(manifest.plan.width) * u64::from(manifest.plan.height) * 3
                <= 512 * 1024 * 1024,
            "boundary exceeds decoded frame budget"
        );
        let directory = job_dir(state, &row.job.id)?;
        let mut needed = count;
        let mut chunks = Vec::new();
        for index in (0..self.appended).rev() {
            if needed == 0 {
                break;
            }
            let (path, receipt, _) = part_paths(&directory, index);
            let receipt = read_receipt(&receipt, checkpoint)?;
            ensure!(
                receipt.manifest_sha256 == row.manifest_sha256 && receipt.index == index,
                "part belongs to another movie"
            );
            let ledger = mold_db::video_jobs::get_part(db(state)?, &row.job.id, index)?
                .context("visible boundary has no committed part authority")?;
            ensure!(
                ledger.output_sha256.as_deref() == Some(receipt.sha256.as_str())
                    && ledger.receipt_sha256.as_deref() == Some(receipt.receipt_sha256.as_str())
                    && ledger.child_id == receipt.child_id,
                "visible boundary differs from committed part authority"
            );
            let mut file = open(&path)?;
            ensure!(
                hash(&mut file, checkpoint)?.sha256 == receipt.sha256,
                "part pixels changed"
            );
            let withheld = if manifest.plan.join_mode == VideoJoinMode::BlendedShots
                && index + 1 < self.appended
            {
                manifest.envelope.overlap_frames
            } else {
                0
            };
            let visible = u32::try_from(receipt.segment_frames)?
                .checked_sub(withheld)
                .context("invalid blended segment")?;
            let take = needed.min(visible);
            if take > 0 {
                chunks.push(frames::tail(
                    file,
                    visible,
                    take,
                    manifest.plan.width,
                    manifest.plan.height,
                    checkpoint,
                )?);
                needed -= take;
            }
        }
        if needed > 0 {
            let prefix = self
                .prefix
                .as_ref()
                .context("missing visible prefix boundary")?;
            let withheld =
                if manifest.plan.join_mode == VideoJoinMode::BlendedShots && self.appended > 0 {
                    manifest.envelope.overlap_frames
                } else {
                    0
                };
            let visible = u32::try_from(manifest.plan.source_frames)?
                .checked_sub(withheld)
                .context("invalid prefix boundary")?;
            chunks.push(frames::tail(
                prefix.try_clone()?,
                visible,
                needed,
                manifest.plan.width,
                manifest.plan.height,
                checkpoint,
            )?);
        }
        chunks.reverse();
        Ok(chunks.into_iter().flatten().collect())
    }

    pub(crate) fn incoming(
        &self,
        state: &AppState,
        row: &StoredVideoJob,
        manifest: &VideoJobManifest,
        checkpoint: &mut dyn FnMut() -> Result<()>,
    ) -> Result<Option<(ChainIncomingSource, Vec<RgbImage>)>> {
        if manifest.template.motion_tail_frames == 0 || self.visible_frames == 0 {
            return Ok(None);
        }
        let context = manifest.template.motion_tail_frames;
        ensure!(
            u64::from(context)
                * u64::from(manifest.plan.width)
                * u64::from(manifest.plan.height)
                * 3
                <= 512 * 1024 * 1024,
            "padded boundary exceeds decoded frame budget"
        );
        let visible = context.min(u32::try_from(self.visible_frames.min(u64::from(context)))?);
        let mut tail = self.visible_tail(state, row, manifest, visible, checkpoint)?;
        if tail.len() < context as usize {
            let first = tail
                .first()
                .context("movie has no visible continuation picture")?
                .clone();
            let mut padded = vec![first; context as usize - tail.len()];
            padded.append(&mut tail);
            tail = padded;
        }
        let mut hash = Sha256::new();
        hash.update(b"mold-visible-parent-boundary-v1\0");
        hash.update(row.manifest_sha256.as_bytes());
        hash.update(self.visible_frames.to_le_bytes());
        for frame in &tail {
            hash.update(frame.as_raw());
        }
        Ok(Some((
            ChainIncomingSource {
                parent_job_id: row.job.id.clone(),
                source_sha256: format!("{:x}", hash.finalize()),
                source_frames: self.visible_frames,
            },
            tail,
        )))
    }

    fn append_sealed(
        &mut self,
        manifest: &VideoJobManifest,
        directory: &Path,
        path: &Path,
        receipt: &PartReceipt,
        checkpoint: &mut dyn FnMut() -> Result<()>,
    ) -> Result<()> {
        ensure!(
            !self.poisoned && receipt.version == 1 && receipt.index == self.appended,
            "invalid part append order"
        );
        let mut file = open(path)?;
        ensure!(
            hash(&mut file, checkpoint)?.sha256 == receipt.sha256,
            "sealed part pixels changed"
        );
        let facts = frames::inspect(file.try_clone()?, checkpoint)?;
        ensure!(
            u64::from(facts.frames) == receipt.segment_frames,
            "sealed part frame count changed"
        );
        let fade =
            if manifest.plan.join_mode == VideoJoinMode::BlendedShots && self.visible_frames > 0 {
                manifest.envelope.overlap_frames
            } else {
                0
            };
        let withheld = if manifest.plan.join_mode == VideoJoinMode::BlendedShots
            && receipt.index + 1 < manifest.plan.clip_count
        {
            manifest.envelope.overlap_frames
        } else {
            0
        };
        let append = u32::try_from(receipt.segment_frames)?
            .checked_sub(withheld)
            .context("part too short for pending fade")?;
        self.assembly
            .as_mut()
            .context("missing movie writer")?
            .append(file, append, checkpoint)?;
        if let Some(spool) = self.audio.as_mut() {
            let descriptor = receipt
                .audio
                .as_ref()
                .context("audio-enabled part has no native PCM authority")?;
            let (_, _, path) = part_paths(directory, receipt.index);
            let mut file = open(&path)?;
            ensure!(
                hash(&mut file, checkpoint)? == descriptor.file,
                "sealed part PCM changed"
            );
            let previous = samples(self.visible_frames, manifest.plan.fps)?;
            let start = samples(
                self.visible_frames
                    .checked_sub(u64::from(fade))
                    .context("invalid audio crossfade")?,
                manifest.plan.fps,
            )?;
            spool.append_pcm_conformed(
                file,
                PcmFormat {
                    sample_rate: descriptor.sample_rate,
                    channels: descriptor.channels,
                },
                0,
                descriptor.frames,
                u32::try_from(previous - start)?,
                checkpoint,
            )?;
        }
        self.visible_frames = self
            .visible_frames
            .checked_add(receipt.segment_frames)
            .and_then(|value| value.checked_sub(u64::from(fade)))
            .context("video timeline overflow")?;
        self.appended += 1;
        serde_json::to_writer(
            &mut self.provenance,
            &serde_json::json!({"receipt_sha256":receipt.receipt_sha256,"part":receipt}),
        )?;
        self.provenance.write_all(b"\n")?;
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn accept_child(
        &mut self,
        state: &AppState,
        row: &StoredVideoJob,
        manifest: &VideoJobManifest,
        index: u64,
        plan: &VideoJobPartPlan,
        child: &ChainJobRow,
        child_manifest: &ChainJobManifest,
        checkpoint: &mut dyn FnMut() -> Result<()>,
    ) -> Result<PartReceipt> {
        ensure!(
            !self.poisoned,
            "movie writer failed; resume from its durable ledger"
        );
        let result = self.accept_inner(
            state,
            row,
            manifest,
            index,
            plan,
            child,
            child_manifest,
            checkpoint,
        );
        if result.is_err() {
            self.poisoned = true;
        }
        result
    }
    #[allow(clippy::too_many_arguments)]
    fn accept_inner(
        &mut self,
        state: &AppState,
        row: &StoredVideoJob,
        manifest: &VideoJobManifest,
        index: u64,
        plan: &VideoJobPartPlan,
        child: &ChainJobRow,
        child_manifest: &ChainJobManifest,
        checkpoint: &mut dyn FnMut() -> Result<()>,
    ) -> Result<PartReceipt> {
        ensure!(
            plan.clips.len() == 1 && index < manifest.plan.clip_count,
            "parent part must own one bounded clip"
        );
        let clip = plan.clips[0];
        let directory = job_dir(state, &row.job.id)?;
        let parts = directory.join("parts");
        std::fs::create_dir_all(&parts)?;
        let (path, receipt_path, audio_path) = part_paths(&directory, index);
        let child_manifest_sha256 =
            format!("{:x}", Sha256::digest(child_manifest.to_toml()?.as_bytes()));
        if receipt_path.exists() {
            let receipt = read_receipt(&receipt_path, checkpoint)?;
            ensure!(
                receipt.version == 1
                    && receipt.manifest_sha256 == row.manifest_sha256
                    && receipt.index == index
                    && receipt.child_id == child.id
                    && receipt.child_manifest_sha256 == child_manifest_sha256,
                "existing part belongs to another child execution"
            );
            if self.appended == index {
                self.append_sealed(manifest, &directory, &path, &receipt, checkpoint)?;
            } else {
                ensure!(self.appended == index + 1, "part append sequence changed");
            }
            return Ok(receipt);
        }
        ensure!(self.appended == index, "part arrived out of order");
        let output = child_manifest
            .finalizes
            .last()
            .context("child has no completed take")?;
        let input = open(&relative(&child.job_dir, &output.output)?)?;
        let facts = frames::inspect(input.try_clone()?, checkpoint)?;
        let continuous = manifest.plan.join_mode == VideoJoinMode::Continuous;
        let incoming = if continuous {
            clip.incoming_overlap_frames
        } else {
            0
        };
        let expected = clip
            .frames
            .checked_sub(incoming)
            .context("invalid clip overlap")?;
        ensure!(
            facts.frames == expected,
            "child delivered a different frame window"
        );
        let keep = expected
            .checked_sub(clip.end_trim_frames)
            .context("invalid final clip trim")?;
        let fade = if !continuous && self.visible_frames > 0 {
            self.visible_tail(
                state,
                row,
                manifest,
                clip.incoming_overlap_frames,
                checkpoint,
            )?
        } else {
            Vec::new()
        };
        let closing = index + 1 == manifest.plan.clip_count
            && manifest.template.stages[0]
                .keyframes
                .as_ref()
                .is_some_and(|frames| frames.iter().any(|frame| frame.frame > 0));
        let mut staged = tempfile::NamedTempFile::new_in(&parts)?;
        if clip.end_trim_frames == 0
            && fade.is_empty()
            && compatible(input.try_clone()?, &self.format_probe, checkpoint)?
        {
            let mut input = input;
            let input_size = input.metadata()?.len();
            input.rewind()?;
            let mut remaining = input_size;
            let mut bytes = [0u8; 64 * 1024];
            loop {
                checkpoint()?;
                if remaining == 0 {
                    break;
                }
                let wanted = usize::try_from(remaining.min(bytes.len() as u64))?;
                let count = input.read(&mut bytes[..wanted])?;
                ensure!(count > 0, "finalized child shrank during retention");
                staged.write_all(&bytes[..count])?;
                remaining -= count as u64;
            }
            ensure!(
                input.metadata()?.len() == input_size,
                "finalized child length changed during retention"
            );
        } else if closing && clip.end_trim_frames > 0 {
            frames::normalize_preserving_endpoint(
                input,
                staged.reopen()?,
                expected,
                keep,
                manifest.plan.width,
                manifest.plan.height,
                manifest.plan.fps,
                &fade,
                checkpoint,
            )?;
        } else {
            frames::normalize(
                input,
                staged.reopen()?,
                keep,
                keep,
                manifest.plan.width,
                manifest.plan.height,
                manifest.plan.fps,
                &fade,
                checkpoint,
            )?;
        }
        let video_descriptor = hash(&mut staged.reopen()?, checkpoint)?;
        persist(staged, &path)?;
        let audio = if self.audio.is_some() {
            let fade_frames = if continuous {
                0
            } else {
                clip.incoming_overlap_frames
            };
            let start = samples(
                self.visible_frames
                    .checked_sub(u64::from(fade_frames))
                    .context("audio overlap exceeds movie")?,
                manifest.plan.fps,
            )?;
            let end = samples(
                self.visible_frames
                    .checked_add(clip.delivered_frames())
                    .context("audio timeline overflow")?,
                manifest.plan.fps,
            )?;
            let take = end.checked_sub(start).context("audio window overflow")?;
            let staged = tempfile::NamedTempFile::new_in(&parts)?;
            let mut conformed = PcmSpool::new(staged.reopen()?, PCM)?;
            if manifest.template.enable_audio == Some(true)
                && manifest.request.generation.video_only != Some(true)
            {
                let stage = child_manifest
                    .stage_status
                    .first()
                    .context("child has no stage PCM status")?;
                let sidecar = stage
                    .audio
                    .as_ref()
                    .context("audio-enabled child has no native PCM")?;
                let mut source = open(&relative(&child.job_dir, sidecar)?)?;
                let mut header = [0u8; 16];
                source.read_exact(&mut header)?;
                ensure!(
                    &header[..8] == b"MOLDPCM1" && header[14..] == [0, 0],
                    "unsupported native PCM header"
                );
                let format = PcmFormat {
                    sample_rate: u32::from_le_bytes(header[8..12].try_into()?),
                    channels: u16::from_le_bytes(header[12..14].try_into()?),
                };
                ensure!(
                    format.sample_rate > 0 && matches!(format.channels, 1 | 2),
                    "invalid native PCM format"
                );
                let bytes = source
                    .metadata()?
                    .len()
                    .checked_sub(16)
                    .context("truncated native PCM")?;
                ensure!(
                    bytes.is_multiple_of(u64::from(format.channels) * 4),
                    "truncated native PCM frame"
                );
                ensure!(
                    bytes <= 64 * 1024 * 1024,
                    "native single-clip PCM exceeds its retained media budget"
                );
                let mut raw = tempfile::tempfile()?;
                let mut buffer = [0u8; 64 * 1024];
                let mut remaining = bytes;
                while remaining > 0 {
                    checkpoint()?;
                    let wanted = usize::try_from(remaining.min(buffer.len() as u64))?;
                    let count = source.read(&mut buffer[..wanted])?;
                    ensure!(count > 0, "native PCM shrank during retention");
                    raw.write_all(&buffer[..count])?;
                    remaining -= count as u64;
                }
                ensure!(
                    source.metadata()?.len() == bytes + 16,
                    "native PCM length changed during retention"
                );
                let skip = if stage.raw_segment && continuous {
                    mold_inference::av_media::aligned_sample_count(
                        u64::from(incoming),
                        manifest.plan.fps,
                        PCM.sample_rate,
                    )
                    .or_else(|error| if incoming == 0 { Ok(0) } else { Err(error) })?
                } else {
                    0
                };
                conformed.append_pcm_conformed(raw, format, skip, take, 0, checkpoint)?;
            } else {
                conformed.append_pcm(&mut std::io::empty(), take, 0, checkpoint)?;
            }
            let mut file = conformed.into_file()?;
            let descriptor = hash(&mut file, checkpoint)?;
            persist(staged, &audio_path)?;
            Some(mold_core::video_artifacts::VideoPcmArtifact {
                file: descriptor,
                sample_rate: PCM.sample_rate,
                channels: PCM.channels,
                frames: take,
            })
        } else {
            None
        };
        let mut receipt = PartReceipt {
            receipt_sha256: String::new(),
            sha256: video_descriptor.sha256,
            segment_frames: u64::from(keep),
            version: 1,
            manifest_sha256: row.manifest_sha256.clone(),
            index,
            child_id: child.id.clone(),
            child_manifest_sha256,
            audio,
            prompt: plan.prompt.clone(),
            seed: *output
                .stage_seeds
                .first()
                .context("child has no effective seed")?,
            scene_index: plan.scene_index,
            generated_frames: clip.frames,
            incoming_overlap_frames: clip.incoming_overlap_frames,
            end_trim_frames: clip.end_trim_frames,
            preserved_closing_frame: closing && clip.end_trim_frames > 0,
        };
        receipt.receipt_sha256 = save_receipt(&receipt_path, &receipt, checkpoint)?;
        self.append_sealed(manifest, &directory, &path, &receipt, checkpoint)?;
        Ok(receipt)
    }

    /// All long reads are cancelable until the publication commit point. After
    /// publication, finish the short ownership handoffs before reporting success.
    pub(crate) fn finish(
        mut self,
        state: &AppState,
        row: &StoredVideoJob,
        manifest: &VideoJobManifest,
        checkpoint: &mut dyn FnMut() -> Result<()>,
    ) -> Result<crate::batch_transaction::ArchivedChildIdentity> {
        ensure!(
            !self.poisoned
                && self.appended == manifest.plan.clip_count
                && self.visible_frames == manifest.plan.total_frames,
            "movie is not fully assembled"
        );
        let receipt = self.assembly.take().context("missing assembly")?.finish()?;
        ensure!(
            receipt.video_frames == manifest.plan.total_frames,
            "assembled video length differs from plan"
        );
        let mut video = self.video;
        let mut pcm = None;
        if let Some(spool) = self.audio {
            let count = spool.frames();
            ensure!(
                count == samples(manifest.plan.total_frames, manifest.plan.fps)?,
                "audio length differs from video clock"
            );
            let mut file = spool.into_file()?;
            let descriptor = hash(&mut file, checkpoint)?;
            let muxed = tempfile::tempfile()?;
            mold_inference::video_assembly_audio::mux_pcm(
                video,
                file.try_clone()?,
                muxed.try_clone()?,
                PCM,
                count,
                checkpoint,
            )?;
            video = muxed;
            pcm = Some((
                file,
                mold_core::video_artifacts::VideoPcmArtifact {
                    file: descriptor,
                    sample_rate: PCM.sample_rate,
                    channels: PCM.channels,
                    frames: count,
                },
            ));
        }
        self.provenance.flush()?;
        self.provenance.sync_all()?;
        let provenance = hash(&mut self.provenance, checkpoint)?;
        let assets_root = crate::video_archive_assets::root_from_db(db(state)?)?;
        let assets = crate::video_archive_assets::prepare(
            &assets_root,
            &row.job.id,
            (self.provenance, provenance),
            pcm,
            checkpoint,
        )?;
        let original = crate::video_job_manifest::hydrate_original_request(state, row, manifest)?;
        let mut metadata = mold_core::OutputMetadata::from_generate_request(
            &original.generation,
            manifest
                .template
                .seed
                .context("parent has no frozen seed")?,
            None,
            env!("CARGO_PKG_VERSION"),
        );
        metadata.frames = u32::try_from(manifest.plan.total_frames).ok();
        metadata.fps = Some(manifest.plan.fps);
        metadata.width = manifest.plan.width;
        metadata.height = manifest.plan.height;
        metadata.chain = None;
        metadata.chain_job_id = None;
        metadata.video_job = Some(mold_core::video_artifacts::VideoJobProvenance {
            id: row.job.id.clone(),
            manifest_sha256: row.manifest_sha256.clone(),
            clip_count: manifest.plan.clip_count,
            part_count: manifest.plan.part_count,
            added_frames: manifest.plan.added_frames,
            source_frames: manifest.plan.source_frames,
            total_frames: manifest.plan.total_frames,
            join_mode: manifest.plan.join_mode,
            authoring: Some(published_authoring(manifest)?),
            source_archive_identity_sha256: manifest
                .retained_source
                .as_ref()
                .map(|source| source.archive_identity_sha256.clone()),
            artifacts: Some(assets.clone()),
        });
        let config = state.config.blocking_read();
        ensure!(
            !state.is_output_disabled(&config),
            "Library output is unavailable"
        );
        let output = config.effective_output_dir();
        drop(config);
        let name = format!("video-{}.mp4", row.job.id);
        let owner = crate::queue::publish_video_file_to_dir_named(
            &output,
            &name,
            video,
            &metadata,
            None,
            Some(db(state)?),
            Some(&state.events),
            &state.gallery_publication_gate,
            checkpoint,
        )?;
        crate::video_archive_assets::seal(&assets_root, &assets, &owner)?;
        crate::chain_source_media::handoff_current_to_gallery(
            db(state)?,
            &crate::video_job_manifest::jobs_root(state)?,
            &job_dir(state, &row.job.id)?,
            &owner,
            &output,
            &manifest.template,
            &state.gallery_publication_gate,
        )?;
        Ok(owner)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use mold_core::{
        chain_job::{ChainJobState, FinalizeRecord},
        video_job::VideoJobState,
    };
    fn fixture(
        home: &Path,
        audio: bool,
        blended: bool,
    ) -> (AppState, StoredVideoJob, VideoJobManifest) {
        let mut state = AppState::for_tests();
        state.metadata_db = std::sync::Arc::new(Some(
            mold_db::MetadataDb::open(&home.join("mold.db")).unwrap(),
        ));
        state.config.blocking_write().output_dir =
            Some(home.join("output").to_string_lossy().into());
        let mut row = crate::video_job_manifest::tests::fixture();
        let mut manifest = crate::video_job_manifest::read_manifest(&row).unwrap();
        manifest.request.duration_ms = 375;
        manifest.request.generation.width = 32;
        manifest.request.generation.height = 32;
        manifest.request.generation.enable_audio = Some(audio);
        manifest.template.width = 32;
        manifest.template.height = 32;
        manifest.template.motion_tail_frames = if blended { 0 } else { 1 };
        manifest.template.enable_audio = Some(audio);
        manifest.template.stages[0].frames = 5;
        manifest.envelope = mold_core::video_plan::ClipEnvelope {
            maximum_frames: 5,
            minimum_frames: 1,
            frame_step: 1,
            frame_offset: 0,
            overlap_frames: 1,
        };
        manifest.plan.width = 32;
        manifest.plan.height = 32;
        manifest.plan.single_clip_max_frames = 5;
        manifest.plan.overlap_frames = 1;
        manifest.plan.clip_count = 2;
        manifest.plan.part_count = 2;
        manifest.plan.added_frames = 9;
        manifest.plan.total_frames = 9;
        manifest.plan.added_duration_ms = 375;
        manifest.plan.total_duration_ms = 375;
        manifest.plan.join_mode = if blended {
            VideoJoinMode::BlendedShots
        } else {
            VideoJoinMode::Continuous
        };
        row.job.id = uuid::Uuid::new_v4().to_string();
        row.job.plan = manifest.plan.clone();
        row.job.state = VideoJobState::Running;
        row.manifest_json = serde_json::to_string(&manifest).unwrap();
        row.manifest_sha256 = format!("{:x}", Sha256::digest(row.manifest_json.as_bytes()));
        std::fs::create_dir_all(job_dir(&state, &row.job.id).unwrap()).unwrap();
        mold_db::video_jobs::insert(db(&state).unwrap(), &row).unwrap();
        (state, row, manifest)
    }
    fn child(
        home: &Path,
        manifest: &VideoJobManifest,
        index: u64,
    ) -> (ChainJobRow, ChainJobManifest) {
        let id = uuid::Uuid::new_v4().to_string();
        let directory = home.join(&id);
        std::fs::create_dir_all(directory.join("final")).unwrap();
        let request = manifest.template.clone();
        let mut child = ChainJobManifest::new(id.clone(), 1, &request).unwrap();
        let count = if index == 0 || manifest.plan.join_mode == VideoJoinMode::BlendedShots {
            5
        } else {
            4
        };
        let color = if index == 0 {
            image::Rgb([210, 0, 0])
        } else {
            image::Rgb([0, 210, 0])
        };
        let video = vec![RgbImage::from_pixel(32, 32, color); count];
        std::fs::write(
            directory.join("final/output-1.mp4"),
            mold_inference::ltx_video::video_enc::encode_mp4(&video, 24).unwrap(),
        )
        .unwrap();
        child.finalizes.push(FinalizeRecord {
            output: "final/output-1.mp4".into(),
            gallery_filename: None,
            at_unix_ms: 1,
            stage_seeds: vec![42 ^ index],
        });
        child.stage_status[0].raw_segment = true;
        if manifest.template.enable_audio == Some(true) {
            let path = directory.join("native.pcm");
            let mut file = File::create(&path).unwrap();
            file.write_all(b"MOLDPCM1").unwrap();
            file.write_all(&48_000u32.to_le_bytes()).unwrap();
            file.write_all(&2u16.to_le_bytes()).unwrap();
            file.write_all(&0u16.to_le_bytes()).unwrap();
            for _ in 0..20_000 {
                file.write_all(&0.25f32.to_le_bytes()).unwrap();
            }
            child.stage_status[0].audio = Some("native.pcm".into());
        }
        let row = ChainJobRow {
            id,
            state: ChainJobState::Completed,
            model: request.model.clone(),
            request_json: serde_json::to_string(&request).unwrap(),
            job_dir: directory,
            stage_count: 1,
            current_stage: 1,
            error: None,
            created_at_ms: 1,
            updated_at_ms: 1,
            finalized_at_ms: Some(1),
        };
        (row, child)
    }
    #[test]
    fn published_authoring_uses_original_added_duration_and_literal_scenes() {
        let home = tempfile::tempdir().unwrap();
        let (_, _, mut manifest) = fixture(home.path(), false, false);
        manifest.request.duration_ms = 30_000;
        manifest.request.generation.frames = Some(97);
        manifest.template.stages[0].frames = 107;
        manifest.template.seed = Some(u64::MAX);
        manifest.request.scenes = Some(vec![
            mold_core::video_job::VideoJobScene {
                prompt: "first\\n\r\nline".into(),
                duration_ms: 10_000,
            },
            mold_core::video_job::VideoJobScene {
                prompt: "second\\t".into(),
                duration_ms: 20_000,
            },
        ]);
        let fresh = published_authoring(&manifest).unwrap();
        assert_eq!(fresh.boundary_frames, 107); // H3's qualified 17n+5 grid.
        assert_eq!((fresh.boundary_frames - 5) % 17, 0);
        assert_eq!(fresh.input_frames, Some(97));
        assert_eq!(fresh.seed, u64::MAX);
        assert_eq!(fresh.scenes, manifest.request.scenes);
        manifest.plan.source_frames = 1440;
        manifest.plan.total_frames = 2160;
        manifest.request.source = Some(mold_core::video_job::VideoJobSource {
            filename: "source.mp4".into(),
            expected_archive_identity_sha256: Some("a".repeat(64)),
            retained_upload: None,
        });
        assert_eq!(
            published_authoring(&manifest).unwrap(),
            fresh,
            "Extend restores added footage as a fresh draft, without the original prefix"
        );
        manifest.request.scenes = None;
        assert!(published_authoring(&manifest).unwrap().scenes.is_none());
    }
    #[test]
    fn completed_parts_survive_child_removal_and_publish_one_exact_movie() {
        for audio in [false, true] {
            for blended in [false, true] {
                let home = tempfile::tempdir().unwrap();
                let (state, mut row, manifest) = fixture(home.path(), audio, blended);
                let schedule = mold_core::video_job_schedule::VideoJobSchedule::new(
                    &manifest.request,
                    manifest.envelope,
                    24,
                    0,
                )
                .unwrap();
                let mut output =
                    VideoJobOutput::open(&state, &row, &manifest, &mut || Ok(())).unwrap();
                assert!(output
                    .incoming(&state, &row, &manifest, &mut || Ok(()))
                    .unwrap()
                    .is_none());
                let (first, first_manifest) = child(home.path(), &manifest, 0);
                let part = output
                    .accept_child(
                        &state,
                        &row,
                        &manifest,
                        0,
                        &schedule.part(0).unwrap(),
                        &first,
                        &first_manifest,
                        &mut || Ok(()),
                    )
                    .unwrap();
                assert_eq!(
                    output
                        .accept_child(
                            &state,
                            &row,
                            &manifest,
                            0,
                            &schedule.part(0).unwrap(),
                            &first,
                            &first_manifest,
                            &mut || Ok(())
                        )
                        .unwrap()
                        .sha256,
                    part.sha256
                );
                row.job.completed_clips = 1;
                let ledger = mold_db::video_jobs::StoredVideoJobPart {
                    parent_id: row.job.id.clone(),
                    index: 0,
                    child_id: first.id.clone(),
                    first_clip: 0,
                    clip_count: 1,
                    keep_frames: part.segment_frames,
                    output_sha256: Some(part.sha256),
                    receipt_sha256: Some(part.receipt_sha256),
                };
                assert!(
                    mold_db::video_jobs::checkpoint(db(&state).unwrap(), &row, Some(&ledger))
                        .unwrap()
                );
                row.revision += 1;
                drop(output);
                std::fs::remove_dir_all(&first.job_dir).unwrap();
                let (_, receipt_path, _) = part_paths(&job_dir(&state, &row.job.id).unwrap(), 0);
                let original_receipt = std::fs::read(&receipt_path).unwrap();
                let mut changed: serde_json::Value =
                    serde_json::from_slice(&original_receipt).unwrap();
                changed["seed"] = serde_json::json!(999);
                std::fs::write(&receipt_path, serde_json::to_vec(&changed).unwrap()).unwrap();
                assert!(VideoJobOutput::open(&state, &row, &manifest, &mut || Ok(())).is_err());
                std::fs::write(&receipt_path, original_receipt).unwrap();
                let mut output =
                    VideoJobOutput::open(&state, &row, &manifest, &mut || Ok(())).unwrap();
                let incoming = output
                    .incoming(&state, &row, &manifest, &mut || Ok(()))
                    .unwrap();
                assert_eq!(incoming.is_some(), !blended);
                let (second, second_manifest) = child(home.path(), &manifest, 1);
                output
                    .accept_child(
                        &state,
                        &row,
                        &manifest,
                        1,
                        &schedule.part(1).unwrap(),
                        &second,
                        &second_manifest,
                        &mut || Ok(()),
                    )
                    .unwrap();
                let owner = output
                    .finish(&state, &row, &manifest, &mut || Ok(()))
                    .unwrap();
                let path = home.path().join("output").join(&owner.final_name);
                let facts = frames::inspect(open(&path).unwrap(), &mut || Ok(())).unwrap();
                assert_eq!(facts.frames, 9);
                let (_, decoded) =
                    mold_inference::ltx2::media::decode_video_frames_from_path(&path).unwrap();
                assert!(decoded[0].get_pixel(16, 16).0[0] > 180);
                assert!(decoded[8].get_pixel(16, 16).0[1] > 180);
                let record = db(&state)
                    .unwrap()
                    .get(&home.path().join("output"), &owner.final_name)
                    .unwrap()
                    .unwrap();
                let provenance = record.metadata.video_job.unwrap();
                assert_eq!(provenance.total_frames, 9);
                let authoring = serde_json::to_value(&provenance).unwrap();
                assert_eq!(authoring["authoring"]["duration_ms"], "375");
                assert_eq!(authoring["authoring"]["boundary_frames"], 5);
                row.job.state = VideoJobState::Completed;
                mold_db::video_jobs::checkpoint(db(&state).unwrap(), &row, None).unwrap();
                let settled = mold_db::video_jobs::get(db(&state).unwrap(), &row.job.id)
                    .unwrap()
                    .unwrap();
                assert!(mold_db::video_jobs::delete_settled(
                    db(&state).unwrap(),
                    &row.job.id,
                    settled.revision,
                    100
                )
                .unwrap());
                std::fs::remove_dir_all(job_dir(&state, &row.job.id).unwrap()).unwrap();
                let saved = db(&state)
                    .unwrap()
                    .get(&home.path().join("output"), &owner.final_name)
                    .unwrap()
                    .unwrap();
                assert_eq!(
                    serde_json::to_value(saved.metadata.video_job.unwrap()).unwrap()["authoring"],
                    authoring["authoring"]
                );

                let assets = provenance.artifacts.unwrap();
                assert_eq!(assets.pcm.is_some(), audio);
                if let Some(pcm) = assets.pcm.as_ref() {
                    assert_eq!(pcm.frames, 18_000);
                }
                let source_hash =
                    crate::gallery_source_media::archive_identity_sha256(&owner).unwrap();
                assert!(crate::video_archive_assets::open_for_source(
                    &crate::video_archive_assets::root_from_db(db(&state).unwrap()).unwrap(),
                    &assets,
                    &source_hash,
                    &mut || Ok(())
                )
                .is_ok());
                assert!(crate::video_archive_assets::open_for_source(
                    &crate::video_archive_assets::root_from_db(db(&state).unwrap()).unwrap(),
                    &assets,
                    &"a".repeat(64),
                    &mut || Ok(())
                )
                .is_err());
            }
        }
    }
    #[test]
    fn library_extension_keeps_prefix_audio_when_added_clip_is_silent() {
        let home = tempfile::tempdir().unwrap();
        let (state, mut row, mut manifest) = fixture(home.path(), false, false);
        let directory = job_dir(&state, &row.job.id).unwrap();
        let prefix = mold_inference::ltx_video::video_enc::encode_mp4(
            &vec![RgbImage::from_pixel(32, 32, image::Rgb([210, 0, 0])); 5],
            24,
        )
        .unwrap();
        let mut video = tempfile::tempfile().unwrap();
        video.write_all(&prefix).unwrap();
        let mut pcm = tempfile::tempfile().unwrap();
        for _ in 0..20_000 {
            pcm.write_all(&0.5f32.to_le_bytes()).unwrap();
        }
        let prefix_path = directory.join("prefix.mp4");
        mold_inference::video_assembly_audio::mux_pcm(
            video,
            pcm,
            std::fs::OpenOptions::new()
                .read(true)
                .write(true)
                .create_new(true)
                .open(&prefix_path)
                .unwrap(),
            PCM,
            10_000,
            &mut || Ok(()),
        )
        .unwrap();
        let descriptor = hash(&mut open(&prefix_path).unwrap(), &mut || Ok(())).unwrap();
        let mut metadata = mold_core::OutputMetadata::from_generate_request(
            &manifest.request.generation,
            42,
            None,
            "test",
        );
        // Actual tracks, rather than a historical enable_audio setting, own
        // the source's sound. The new clip deliberately has no PCM sidecar.
        metadata.enable_audio = Some(false);
        let mut native = tempfile::tempfile().unwrap();
        for _ in 0..20_000 {
            native.write_all(&0.5f32.to_le_bytes()).unwrap();
        }
        let native_descriptor = hash(&mut native, &mut || Ok(())).unwrap();
        let mut provenance = tempfile::tempfile().unwrap();
        provenance.write_all(b"source provenance").unwrap();
        let provenance_descriptor = hash(&mut provenance, &mut || Ok(())).unwrap();
        let assets_root = crate::video_archive_assets::root_from_db(db(&state).unwrap()).unwrap();
        let source_id = uuid::Uuid::new_v4().to_string();
        let source_owner = crate::batch_transaction::ArchivedChildIdentity {
            parent_id: source_id.clone(),
            attempt_generation: 1,
            child_index: 0,
            final_name: "original.mp4".into(),
            checksum_sha256: descriptor.sha256.clone(),
            size_bytes: descriptor.size_bytes,
        };
        let source_identity =
            crate::gallery_source_media::archive_identity_sha256(&source_owner).unwrap();
        let source_assets = crate::video_archive_assets::prepare(
            &assets_root,
            &source_id,
            (provenance, provenance_descriptor),
            Some((
                native,
                mold_core::video_artifacts::VideoPcmArtifact {
                    file: native_descriptor,
                    sample_rate: PCM.sample_rate,
                    channels: PCM.channels,
                    frames: 10_000,
                },
            )),
            &mut || Ok(()),
        )
        .unwrap();
        crate::video_archive_assets::seal(&assets_root, &source_assets, &source_owner).unwrap();
        metadata.video_job = Some(mold_core::video_artifacts::VideoJobProvenance {
            id: source_id,
            manifest_sha256: "b".repeat(64),
            clip_count: 1,
            part_count: 1,
            added_frames: 5,
            source_frames: 0,
            total_frames: 5,
            join_mode: VideoJoinMode::Continuous,
            authoring: None,
            source_archive_identity_sha256: None,
            artifacts: Some(source_assets.clone()),
        });
        manifest.retained_source = Some(crate::video_source::RetainedVideoSource {
            authority_kind: mold_core::video_job::VideoSourceAuthorityKind::PublishedArchive,
            filename: "original.mp4".into(),
            archive_identity_sha256: source_identity.clone(),
            checksum_sha256: descriptor.sha256,
            size_bytes: descriptor.size_bytes,
            metadata,
        });
        manifest.request.source = Some(mold_core::video_job::VideoJobSource {
            filename: "original.mp4".into(),
            expected_archive_identity_sha256: Some(source_identity),
            retained_upload: None,
        });
        manifest.request.duration_ms = 166;
        manifest.plan.source_frames = 5;
        manifest.plan.added_frames = 4;
        manifest.plan.total_frames = 9;
        manifest.plan.clip_count = 1;
        manifest.plan.part_count = 1;
        manifest.plan.added_duration_ms = 167;
        manifest.plan.total_duration_ms = 375;
        manifest.prefix_audio =
            crate::video_prefix_audio::seal(&directory, &manifest.plan, &mut || Ok(())).unwrap();
        row.job.plan = manifest.plan.clone();
        row.manifest_json = serde_json::to_string(&manifest).unwrap();
        row.manifest_sha256 = format!("{:x}", Sha256::digest(row.manifest_json.as_bytes()));
        let schedule = mold_core::video_job_schedule::VideoJobSchedule::new(
            &manifest.request,
            manifest.envelope,
            24,
            5,
        )
        .unwrap();
        let first = VideoJobOutput::open(&state, &row, &manifest, &mut || Ok(())).unwrap();
        let mut first_pcm = first.audio.unwrap().into_file().unwrap();
        let before = hash(&mut first_pcm, &mut || Ok(())).unwrap();
        assert!(
            crate::video_archive_assets::release(&assets_root, &source_assets, &source_owner)
                .unwrap()
        );
        let mut output = VideoJobOutput::open(&state, &row, &manifest, &mut || Ok(())).unwrap();
        let mut resumed_pcm = output.audio.take().unwrap().into_file().unwrap();
        let after = hash(&mut resumed_pcm, &mut || Ok(())).unwrap();
        assert_eq!(
            before, after,
            "accepted source sound must not depend on the original Library bundle surviving"
        );
        output.audio = Some(PcmSpool::resume(resumed_pcm, PCM, 10_000).unwrap());
        assert!(
            output.audio.is_some(),
            "the original movie's soundtrack must survive"
        );
        let (child, child_manifest) = child(home.path(), &manifest, 1);
        output
            .accept_child(
                &state,
                &row,
                &manifest,
                0,
                &schedule.part(0).unwrap(),
                &child,
                &child_manifest,
                &mut || Ok(()),
            )
            .unwrap();
        let audio = output.audio.take().unwrap();
        assert_eq!(audio.frames(), 18_000);
        let mut pcm = audio.into_file().unwrap();
        pcm.rewind().unwrap();
        let mut prefix = vec![0; 10_000 * 8];
        pcm.read_exact(&mut prefix).unwrap();
        assert!(
            prefix
                .as_chunks::<4>()
                .0
                .iter()
                .map(|b| f32::from_le_bytes(*b).abs())
                .sum::<f32>()
                > 100.0
        );
        let mut suffix = Vec::new();
        pcm.read_to_end(&mut suffix).unwrap();
        assert_eq!(suffix.len(), 8_000 * 8);
        assert!(
            suffix.iter().all(|b| *b == 0),
            "silent extension must not repeat the source soundtrack"
        );
    }
    #[test]
    fn short_visible_movie_pads_context_without_adding_published_frames() {
        let home = tempfile::tempdir().unwrap();
        let (state, mut row, mut manifest) = fixture(home.path(), false, false);
        manifest.template.motion_tail_frames = 3;
        manifest.envelope.overlap_frames = 3;
        manifest.plan.overlap_frames = 3;
        manifest.plan.clip_count = 1;
        manifest.plan.part_count = 1;
        manifest.plan.added_frames = 1;
        manifest.plan.total_frames = 1;
        manifest.plan.added_duration_ms = 42;
        manifest.plan.total_duration_ms = 42;
        manifest.request.duration_ms = 41;
        row.job.id = uuid::Uuid::new_v4().to_string();
        row.job.plan = manifest.plan.clone();
        row.manifest_json = serde_json::to_string(&manifest).unwrap();
        row.manifest_sha256 = format!("{:x}", Sha256::digest(row.manifest_json.as_bytes()));
        std::fs::create_dir_all(job_dir(&state, &row.job.id).unwrap()).unwrap();
        mold_db::video_jobs::insert(db(&state).unwrap(), &row).unwrap();
        let mut output = VideoJobOutput::open(&state, &row, &manifest, &mut || Ok(())).unwrap();
        let (child, child_manifest) = child(home.path(), &manifest, 0);
        let plan = VideoJobPartPlan {
            prompt: "short".into(),
            scene_index: 0,
            first_clip: 0,
            clips: vec![mold_core::video_plan::PlannedClip {
                frames: 5,
                incoming_overlap_frames: 0,
                end_trim_frames: 4,
            }],
        };
        let receipt = output
            .accept_child(
                &state,
                &row,
                &manifest,
                0,
                &plan,
                &child,
                &child_manifest,
                &mut || Ok(()),
            )
            .unwrap();
        row.job.completed_clips = 1;
        let ledger = mold_db::video_jobs::StoredVideoJobPart {
            parent_id: row.job.id.clone(),
            index: 0,
            child_id: child.id.clone(),
            first_clip: 0,
            clip_count: 1,
            keep_frames: receipt.segment_frames,
            output_sha256: Some(receipt.sha256),
            receipt_sha256: Some(receipt.receipt_sha256),
        };
        assert!(mold_db::video_jobs::checkpoint(db(&state).unwrap(), &row, Some(&ledger)).unwrap());
        row.revision += 1;
        let (source, tail) = output
            .incoming(&state, &row, &manifest, &mut || Ok(()))
            .unwrap()
            .unwrap();
        assert_eq!(source.source_frames, 1);
        assert_eq!(tail.len(), 3);
        assert_eq!(tail[0], tail[2]);
        let owner = output
            .finish(&state, &row, &manifest, &mut || Ok(()))
            .unwrap();
        let facts = frames::inspect(
            open(&home.path().join("output").join(owner.final_name)).unwrap(),
            &mut || Ok(()),
        )
        .unwrap();
        assert_eq!(facts.frames, 1);
    }
}
