//! Owned incoming motion tails for bounded durable continuation parts.
use std::io::{Read, Write};
use std::path::Path;

use anyhow::{bail, Context};
use mold_core::chain::ChainRequest;
use mold_core::chain_job::{
    ChainJobManifest, ChainJobState, FrozenChainModel, JobDirLayout, StageState,
};
use mold_db::chain_jobs::ChainJobRow;
use mold_inference::chain::ChainTail;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

const DIRECTORY: &str = "continuation";
const MANIFEST: &str = "tail.json";

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct Binding {
    version: u32,
    source_job_id: String,
    source_take: u32,
    model: String,
    model_fingerprint: String,
    width: u32,
    height: u32,
    fps: u32,
    motion_tail_frames: u32,
    audio: bool,
    frame_sha256: Vec<String>,
}

pub(crate) struct PreparedContinuation {
    binding: Binding,
    frames: Vec<Vec<u8>>,
}

impl PreparedContinuation {
    pub(crate) fn binding_sha256(&self) -> anyhow::Result<String> {
        Ok(format!(
            "{:x}",
            Sha256::digest(serde_json::to_vec(&self.binding)?)
        ))
    }
}

pub(crate) fn inherit_frozen(
    manifest: &ChainJobManifest,
    model: &str,
) -> anyhow::Result<FrozenChainModel> {
    if manifest.request()?.model != model {
        bail!("continuation must preserve the predecessor's model identity");
    }
    let frozen = manifest
        .frozen_model
        .as_ref()
        .context("predecessor has no frozen model artifact identity")?;
    // Revalidate the original recipe and files, rather than freezing today's
    // defaults. Sampling controls belong to each part; the artifact authority
    // remains exactly the predecessor's snapshot across config refreshes.
    let fingerprint = crate::execution_plan::frozen_model_fingerprint(model, &frozen.config)?;
    if fingerprint != frozen.model_fingerprint
        || (!frozen.runtime_model_id.is_empty()
            && frozen.runtime_model_id != format!("mold-frozen-chain:{fingerprint}"))
    {
        bail!("predecessor frozen model artifacts or internal identity changed");
    }
    Ok(frozen.clone())
}

pub(crate) fn prepare(
    source: &ChainJobRow,
    manifest: &ChainJobManifest,
    request: &ChainRequest,
    frozen: &FrozenChainModel,
) -> anyhow::Result<PreparedContinuation> {
    let link = request
        .continue_from
        .as_ref()
        .context("missing continuation predecessor")?;
    let previous = manifest.request()?;
    if source.id != link.job_id
        || manifest.job_id != source.id
        || source.state != ChainJobState::Completed
        || !manifest.current_revision_is_finalized(source.state)
        || manifest.finalizes.len() != link.finalize_take as usize
    {
        bail!("continuation requires the predecessor's latest completed finalize take");
    }
    if previous.model != request.model
        || source.model != previous.model
        || manifest.frozen_model.as_ref() != Some(frozen)
        || previous.width != request.width
        || previous.height != request.height
        || previous.fps != request.fps
        || previous.motion_tail_frames != request.motion_tail_frames
        || previous.enable_audio.unwrap_or(false) != request.enable_audio.unwrap_or(false)
    {
        bail!("continuation must preserve model artifacts, dimensions, fps, motion tail and audio setting");
    }
    validate_first_stage(request)?;
    let mut binding = binding(request, &frozen.model_fingerprint)?;
    let mut frames = Vec::new();
    let mut total_bytes = 0_u64;
    if request.motion_tail_frames > 0 {
        let status = manifest
            .stage_status
            .last()
            .context("predecessor has no completed stage")?;
        if status.state != StageState::Completed {
            bail!("predecessor's final stage is incomplete");
        }
        let available = status.tail_frames.unwrap_or(0);
        if available < request.motion_tail_frames {
            bail!("predecessor's motion tail is unavailable; its stage cache may have been pruned");
        }
        let directory = JobDirLayout::new(source.job_dir.clone()).tail_dir(status.idx);
        for index in available - request.motion_tail_frames..available {
            let bytes = read_bounded(
                &directory.join(format!("{index:03}.png")),
                png_limit(request),
            )?;
            total_bytes = total_bytes
                .checked_add(bytes.len() as u64)
                .filter(|total| *total <= 512 * 1024 * 1024)
                .context("continuation snapshot exceeds 512 MiB")?;
            decode_frame(&bytes, request)?;
            binding
                .frame_sha256
                .push(format!("{:x}", Sha256::digest(&bytes)));
            frames.push(bytes);
        }
    }
    Ok(PreparedContinuation { binding, frames })
}

pub(crate) fn persist(job_dir: &Path, prepared: &PreparedContinuation) -> anyhow::Result<()> {
    let destination = job_dir.join(DIRECTORY);
    if destination.symlink_metadata().is_ok() {
        bail!("continuation snapshot already exists and cannot be replaced");
    }
    struct Staging(std::path::PathBuf);
    impl Drop for Staging {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.0);
        }
    }
    let staged = Staging(job_dir.join(format!(".continuation-{}", uuid::Uuid::new_v4())));
    crate::batch_transaction::create_private_directory_no_follow(&staged.0)?;
    for (index, bytes) in prepared.frames.iter().enumerate() {
        write_private(&staged.0.join(format!("{index:03}.png")), bytes)?;
    }
    write_private(
        &staged.0.join(MANIFEST),
        &serde_json::to_vec(&prepared.binding)?,
    )?;
    crate::dir_sync::sync_directory(&staged.0)?;
    std::fs::rename(&staged.0, &destination)?;
    crate::dir_sync::sync_directory(job_dir)?;
    Ok(())
}

pub(crate) fn load(
    job_dir: &Path,
    request: &ChainRequest,
    frozen: &FrozenChainModel,
) -> anyhow::Result<Option<ChainTail>> {
    if request.continue_from.is_none() {
        return Ok(None);
    }
    let prepared = read_snapshot(job_dir, request, Some(frozen))?;
    if prepared.frames.is_empty() {
        return Ok(None);
    }
    let frames = prepared
        .frames
        .iter()
        .map(|bytes| decode_frame(bytes, request))
        .collect::<anyhow::Result<Vec<_>>>()?;
    Ok(Some(ChainTail {
        frames: request.motion_tail_frames,
        tail_rgb_frames: frames,
    }))
}

/// Read raw derived conditioning for encrypted gallery retention. These bytes
/// never become authored source_image/keyframes or runtime admission authority.
pub(crate) fn retained_frames(
    job_dir: &Path,
    request: &ChainRequest,
) -> anyhow::Result<Vec<Vec<u8>>> {
    if request.continue_from.is_none() {
        return Ok(Vec::new());
    }
    Ok(read_snapshot(job_dir, request, None)?.frames)
}

fn binding(request: &ChainRequest, fingerprint: &str) -> anyhow::Result<Binding> {
    let source = request
        .continue_from
        .as_ref()
        .context("missing continuation predecessor")?;
    Ok(Binding {
        version: 1,
        source_job_id: source.job_id.clone(),
        source_take: source.finalize_take,
        model: request.model.clone(),
        model_fingerprint: fingerprint.to_owned(),
        width: request.width,
        height: request.height,
        fps: request.fps,
        motion_tail_frames: request.motion_tail_frames,
        audio: request.enable_audio.unwrap_or(false),
        frame_sha256: Vec::new(),
    })
}

pub(crate) fn validate_first_stage(request: &ChainRequest) -> anyhow::Result<()> {
    let first = request
        .stages
        .first()
        .context("continuation requires an explicit first stage")?;
    if first.transition != mold_core::chain::TransitionMode::Smooth
        || (request.motion_tail_frames > 0
            && (first.source_image.is_some()
                || first
                    .keyframes
                    .as_ref()
                    .is_some_and(|keys| keys.iter().any(|key| key.frame == 0))))
    {
        bail!("continuation requires a smooth first stage; a carried tail forbids an authored first image or frame-zero keyframe");
    }
    Ok(())
}

fn read_snapshot(
    job_dir: &Path,
    request: &ChainRequest,
    frozen: Option<&FrozenChainModel>,
) -> anyhow::Result<PreparedContinuation> {
    validate_first_stage(request)?;
    let root = job_dir.join(DIRECTORY);
    crate::batch_transaction::verify_directory_no_follow(&root)?;
    let stored: Binding = serde_json::from_slice(&read_bounded(&root.join(MANIFEST), 64 * 1024)?)?;
    let fingerprint = frozen.map_or(stored.model_fingerprint.as_str(), |model| {
        model.model_fingerprint.as_str()
    });
    let mut expected = binding(request, fingerprint)?;
    expected.frame_sha256 = stored.frame_sha256.clone();
    if stored != expected || stored.frame_sha256.len() != request.motion_tail_frames as usize {
        bail!("continuation snapshot differs from the successor's frozen contract");
    }
    if std::fs::read_dir(&root)?.count() != stored.frame_sha256.len() + 1 {
        bail!("continuation snapshot has an unexpected member inventory");
    }
    let mut frames = Vec::new();
    let mut total_bytes = 0_u64;
    for (index, digest) in stored.frame_sha256.iter().enumerate() {
        let bytes = read_bounded(&root.join(format!("{index:03}.png")), png_limit(request))?;
        total_bytes = total_bytes
            .checked_add(bytes.len() as u64)
            .filter(|total| *total <= 512 * 1024 * 1024)
            .context("continuation snapshot exceeds 512 MiB")?;
        if format!("{:x}", Sha256::digest(&bytes)) != *digest {
            bail!("continuation tail digest mismatch");
        }
        decode_frame(&bytes, request)?;
        frames.push(bytes);
    }
    Ok(PreparedContinuation {
        binding: stored,
        frames,
    })
}

fn png_limit(request: &ChainRequest) -> u64 {
    u64::from(request.width)
        .saturating_mul(u64::from(request.height))
        .saturating_mul(8)
        .saturating_add(1024 * 1024)
        .min(512 * 1024 * 1024)
}

fn read_bounded(path: &Path, limit: u64) -> anyhow::Result<Vec<u8>> {
    let file = mold_core::secure_file::open_regular_file_no_follow(path)?;
    let mut bytes = Vec::new();
    file.take(limit + 1).read_to_end(&mut bytes)?;
    if bytes.len() as u64 > limit {
        bail!("continuation member exceeds its bounded reader");
    }
    Ok(bytes)
}

fn decode_frame(bytes: &[u8], request: &ChainRequest) -> anyhow::Result<image::RgbImage> {
    let dimensions = image::ImageReader::new(std::io::Cursor::new(bytes))
        .with_guessed_format()?
        .into_dimensions()?;
    if dimensions != (request.width, request.height) {
        bail!("continuation tail dimensions changed");
    }
    Ok(image::load_from_memory(bytes)?.to_rgb8())
}

fn write_private(path: &Path, bytes: &[u8]) -> anyhow::Result<()> {
    let mut options = std::fs::OpenOptions::new();
    options.write(true).create_new(true);
    #[cfg(unix)]
    {
        use std::os::unix::fs::OpenOptionsExt;
        options.mode(0o600);
    }
    let mut file = options.open(path)?;
    file.write_all(bytes)?;
    file.sync_all()?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use image::RgbImage;
    use mold_core::chain_job::FinalizeRecord;

    fn fixture(
        root: &Path,
        tail: u32,
    ) -> (
        ChainJobRow,
        ChainJobManifest,
        ChainRequest,
        FrozenChainModel,
    ) {
        let request: ChainRequest = serde_json::from_value(serde_json::json!({
            "model":"ltx-2-19b-distilled:fp8", "width":32,"height":32,"fps":8,
            "steps":2,"guidance":1.0,"motion_tail_frames":tail,"seed":42,
            "stages":[{"prompt":"previous","frames":9}]
        }))
        .unwrap();
        let frozen = FrozenChainModel {
            runtime_model_id: "fixture".into(),
            config: Default::default(),
            model_fingerprint: "same-artifacts".into(),
        };
        let job_dir = root.join("previous");
        let layout = JobDirLayout::new(job_dir.clone());
        std::fs::create_dir_all(layout.tail_dir(0)).unwrap();
        for index in 0..tail {
            RgbImage::from_pixel(32, 32, image::Rgb([index as u8 + 1, 2, 3]))
                .save(layout.tail_dir(0).join(format!("{index:03}.png")))
                .unwrap();
        }
        let mut manifest = ChainJobManifest::new("previous".into(), 1, &request).unwrap();
        manifest.frozen_model = Some(frozen.clone());
        manifest.stage_status[0].state = StageState::Completed;
        manifest.stage_status[0].tail_frames = Some(tail);
        manifest.needs_finalize = Some(false);
        manifest.finalizes.push(FinalizeRecord {
            output: "final/output-1.mp4".into(),
            gallery_filename: None,
            at_unix_ms: 1,
            stage_seeds: vec![42],
        });
        let row = ChainJobRow {
            id: "previous".into(),
            state: ChainJobState::Completed,
            model: request.model.clone(),
            request_json: manifest.request_json.clone(),
            job_dir,
            stage_count: 1,
            current_stage: 1,
            error: None,
            created_at_ms: 1,
            updated_at_ms: 1,
            finalized_at_ms: Some(1),
        };
        let mut child = request;
        child.continue_from = Some(mold_core::chain::ChainContinuation {
            job_id: row.id.clone(),
            finalize_take: 1,
        });
        (row, manifest, child, frozen)
    }

    #[test]
    fn continuation_owns_tail_after_predecessor_deletion_and_reopens_after_restart() {
        let root = tempfile::tempdir().unwrap();
        let (source, manifest, request, frozen) = fixture(root.path(), 1);
        let prepared = prepare(&source, &manifest, &request, &frozen).unwrap();
        let child = root.path().join("child");
        std::fs::create_dir(&child).unwrap();
        persist(&child, &prepared).unwrap();
        std::fs::remove_dir_all(&source.job_dir).unwrap();
        drop(prepared);
        let carry = load(&child, &request, &frozen).unwrap().unwrap();
        assert_eq!(carry.frames, 1);
        assert_eq!(carry.tail_rgb_frames[0].get_pixel(0, 0).0, [1, 2, 3]);
        std::fs::write(child.join(DIRECTORY).join("000.png"), b"tampered").unwrap();
        assert!(load(&child, &request, &frozen).is_err());
    }

    #[test]
    fn continuation_refuses_stale_take_incomplete_parent_and_changed_contracts() {
        let root = tempfile::tempdir().unwrap();
        let (source, manifest, request, frozen) = fixture(root.path(), 1);
        let mut stale = request.clone();
        stale.continue_from.as_mut().unwrap().finalize_take = 2;
        assert!(prepare(&source, &manifest, &stale, &frozen).is_err());
        let mut running = source.clone();
        running.state = ChainJobState::Running;
        assert!(prepare(&running, &manifest, &request, &frozen).is_err());
        for field in ["model", "width", "fps", "motion_tail_frames", "audio"] {
            let mut bad = request.clone();
            match field {
                "model" => bad.model = "different".into(),
                "width" => bad.width = 64,
                "fps" => bad.fps = 24,
                "audio" => bad.enable_audio = Some(true),
                _ => bad.motion_tail_frames = 9,
            }
            assert!(
                prepare(&source, &manifest, &bad, &frozen).is_err(),
                "{field}"
            );
        }
        let mut wrong = frozen.clone();
        wrong.model_fingerprint = "changed".into();
        assert!(prepare(&source, &manifest, &request, &wrong).is_err());
    }

    #[test]
    fn independent_link_still_owns_a_required_recovery_marker() {
        let root = tempfile::tempdir().unwrap();
        let (source, manifest, request, frozen) = fixture(root.path(), 0);
        let child = root.path().join("child");
        std::fs::create_dir(&child).unwrap();
        assert!(load(&child, &request, &frozen).is_err());
        let prepared = prepare(&source, &manifest, &request, &frozen).unwrap();
        persist(&child, &prepared).unwrap();
        std::fs::remove_dir_all(&source.job_dir).unwrap();
        assert!(load(&child, &request, &frozen).unwrap().is_none());
        assert!(
            persist(&child, &prepared).is_err(),
            "immutable snapshot cannot be replaced"
        );
    }

    #[test]
    fn independent_link_keeps_authored_inputs_while_carried_link_refuses_collision() {
        let root = tempfile::tempdir().unwrap();
        let (source, manifest, mut request, frozen) = fixture(root.path(), 0);
        request.stages[0].source_image = Some(vec![1, 2, 3]);
        let prepared = prepare(&source, &manifest, &request, &frozen).unwrap();
        let child = root.path().join("child");
        std::fs::create_dir(&child).unwrap();
        persist(&child, &prepared).unwrap();
        assert!(load(&child, &request, &frozen).unwrap().is_none());
        assert!(retained_frames(&child, &request).unwrap().is_empty());
        let scrubbed = crate::chain_source_media::persist_scrubbed(
            root.path(),
            &child,
            "child",
            request.clone(),
        )
        .unwrap();
        assert!(scrubbed.stages[0].source_image.is_none());
        let hydrated = crate::chain_source_media::hydrate(root.path(), &child, scrubbed).unwrap();
        assert_eq!(
            hydrated.stages[0].source_image.as_deref(),
            Some(&[1, 2, 3][..])
        );
        request.motion_tail_frames = 1;
        assert!(validate_first_stage(&request).is_err());
    }

    #[test]
    fn continuation_inherits_frozen_defaults_and_rejects_changed_artifact_bytes() {
        let root = tempfile::tempdir().unwrap();
        let (source, mut manifest, mut request, mut frozen) = fixture(root.path(), 1);
        let weights = root.path().join("weights.safetensors");
        std::fs::write(&weights, b"original weights").unwrap();
        frozen.config.transformer = Some(weights.display().to_string());
        frozen.config.family = Some("ltx2".into());
        frozen.config.default_steps = Some(2);
        frozen.model_fingerprint =
            crate::execution_plan::frozen_model_fingerprint(&request.model, &frozen.config)
                .unwrap();
        frozen.runtime_model_id = format!("mold-frozen-chain:{}", frozen.model_fingerprint);
        manifest.frozen_model = Some(frozen.clone());
        let mut changed_live_defaults = frozen.config.clone();
        changed_live_defaults.default_steps = Some(7);
        assert_ne!(changed_live_defaults, frozen.config);
        request.steps = 7;
        request.guidance = 2.0;
        request.seed = Some(123);
        let inherited = inherit_frozen(&manifest, &request.model).unwrap();
        assert_eq!(inherited, frozen);
        prepare(&source, &manifest, &request, &inherited).unwrap();
        std::fs::write(&weights, b"changed transformer weights").unwrap();
        assert!(inherit_frozen(&manifest, &request.model).is_err());
    }
}
