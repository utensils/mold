//! Private parent manifests and exact reconstruction of their sealed inputs.

use anyhow::{ensure, Context, Result};
use mold_core::{
    chain::ChainRequest,
    chain_job::FrozenChainModel,
    video_job::{VideoJobPlan, VideoJobRequest},
    video_plan::ClipEnvelope,
};
use mold_db::video_jobs::StoredVideoJob;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::path::PathBuf;

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct VideoJobManifest {
    pub version: u32,
    /// Logical scalar request. Exact typed media belongs to `template`'s
    /// encrypted source-media index, never this JSON object.
    pub request: VideoJobRequest,
    #[serde(deserialize_with = "literal_template")]
    pub template: ChainRequest,
    pub frozen: FrozenChainModel,
    #[serde(default)]
    pub frozen_loras: Vec<crate::video_job_loras::FrozenVideoLora>,
    pub envelope: ClipEnvelope,
    pub retained_source: Option<crate::video_source::RetainedVideoSource>,
    #[serde(default)]
    pub prefix_audio: Option<mold_core::video_artifacts::VideoPcmArtifact>,
    pub plan: VideoJobPlan,
    /// Sealed at admission: H3 FL2VA clips that carry the previous clip's
    /// last frame open with upstream's alignment instruction. Absent (false)
    /// on parents admitted before it existed, so their recorded children stay
    /// byte-identical on Resume.
    #[serde(default, skip_serializing_if = "std::ops::Not::not")]
    pub continuation_instruction: bool,
}

fn literal_template<'de, D: serde::Deserializer<'de>>(
    d: D,
) -> std::result::Result<ChainRequest, D::Error> {
    let raw = serde_json::Value::deserialize(d)?;
    mold_core::prompt_text::chain_request_from_literal_json(&raw.to_string())
        .map_err(serde::de::Error::custom)
}

pub(crate) fn read_manifest(row: &StoredVideoJob) -> Result<VideoJobManifest> {
    ensure!(
        row.manifest_json.len() <= 8 * 1024 * 1024,
        "parent manifest exceeds its scalar budget"
    );
    ensure!(
        format!("{:x}", Sha256::digest(row.manifest_json.as_bytes())) == row.manifest_sha256,
        "parent manifest identity changed"
    );
    let manifest: VideoJobManifest = serde_json::from_str(&row.manifest_json)?;
    ensure!(
        manifest.version == 1,
        "unsupported video parent manifest version"
    );
    manifest
        .request
        .validate_shape()
        .map_err(anyhow::Error::msg)?;
    ensure!(
        manifest.plan == row.job.plan,
        "parent plan differs from its immutable manifest"
    );
    ensure!(
        manifest.template.stages.len() == 1 && !manifest.template.has_incoming_boundary(),
        "parent template must contain one independent sealed input stage"
    );
    ensure!(
        manifest.template.model == manifest.plan.model
            && manifest.request.generation.model == manifest.plan.model
            && manifest.template.width == manifest.plan.width
            && manifest.template.height == manifest.plan.height
            && manifest.template.fps == manifest.plan.fps,
        "parent template recipe differs from its plan"
    );
    for lora in &manifest.frozen_loras {
        lora.validate()?;
    }
    if let Some(audio) = &manifest.prefix_audio {
        ensure!(
            manifest.retained_source.is_some(),
            "prefix audio has no source authority"
        );
        crate::video_prefix_audio::validate(audio, manifest.plan.source_frames, manifest.plan.fps)?;
    }
    let request = &manifest.request.generation;
    ensure!(
        [
            &request.source_image,
            &request.id_image,
            &request.mask_image,
            &request.control_image,
            &request.audio_file,
            &request.source_video,
            &request.extend_video
        ]
        .iter()
        .all(|value| value.is_none())
            && request.id_images.is_none()
            && request.edit_images.is_none()
            && request.references.is_none()
            && request.keyframes.is_none(),
        "parent scalar manifest contains unsealed authored media"
    );
    ensure!(
        request.audio_file_path.is_none()
            && request.source_video_path.is_none()
            && request.extend_video_path.is_none(),
        "parent scalar manifest contains machine-local input authority"
    );
    let stage = &manifest.template.stages[0];
    ensure!(
        stage.source_image.as_ref().is_none_or(Vec::is_empty)
            && stage
                .keyframes
                .as_ref()
                .is_none_or(|frames| frames.iter().all(|frame| frame.image.is_empty()))
            && stage
                .generation_references
                .as_ref()
                .is_none_or(|refs| refs.iter().all(|reference| matches!(
                    reference.media(),
                    mold_core::GenerationReferenceAuthority::Descriptor
                ))),
        "parent template contains unsealed authored media"
    );
    ensure!(
        manifest.request.source.is_some() == manifest.retained_source.is_some(),
        "parent source snapshot authority is incomplete"
    );
    if let (Some(source), Some(retained)) = (&manifest.request.source, &manifest.retained_source) {
        ensure!(
            source.retained_upload.is_none()
                && source.filename == retained.filename
                && source.expected_archive_identity_sha256.as_deref()
                    == Some(retained.archive_identity_sha256.as_str()),
            "parent source snapshot differs from its original publication"
        );
    }
    Ok(manifest)
}

/// Derive storage from the server's already-open metadata DB, never a mutable
/// bootstrap pointer or a caller-provided path.
pub(crate) fn jobs_root(state: &crate::state::AppState) -> Result<PathBuf> {
    let db = state
        .metadata_db
        .as_ref()
        .as_ref()
        .context("durable videos require a metadata database")?;
    let parent = db
        .path()
        .parent()
        .filter(|parent| !parent.as_os_str().is_empty())
        .context("durable videos require a file-backed metadata database")?;
    Ok(parent.join("video-jobs"))
}

pub(crate) fn job_dir(state: &crate::state::AppState, id: &str) -> Result<PathBuf> {
    let uuid = uuid::Uuid::parse_str(id)?;
    ensure!(
        uuid.to_string() == id,
        "video parent identifier must be a canonical UUID"
    );
    Ok(jobs_root(state)?.join(id))
}

/// Source exports and new attempts share this exact inverse of admission's
/// sealing map. Neither metadata filenames nor consumed upload IDs become bytes.
pub(crate) fn hydrate_original_request(
    state: &crate::state::AppState,
    row: &StoredVideoJob,
    manifest: &VideoJobManifest,
) -> Result<VideoJobRequest> {
    ensure!(
        manifest.plan == row.job.plan,
        "parent plan changed before input hydration"
    );
    let template = crate::chain_source_media::hydrate(
        &jobs_root(state)?,
        &job_dir(state, &row.job.id)?,
        manifest.template.clone(),
    )?;
    let first = template
        .stages
        .first()
        .context("parent has no original input template")?;
    let mut request = manifest.request.clone();
    request.generation.source_image = first.source_image.clone();
    request.generation.keyframes = first.keyframes.clone();
    request.generation.references = first.generation_references.clone();
    request.generation.seed = template.seed;
    // Retained keyframes still use the original authored endpoint locator.
    request.generation.frames = manifest.request.generation.frames.or(Some(first.frames));
    request.validation_revision = None;
    if let Some(source) = request.source.as_mut() {
        source.retained_upload = None;
    }
    Ok(request)
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use sha2::{Digest, Sha256};

    pub(crate) fn fixture() -> mold_db::video_jobs::StoredVideoJob {
        let plan = serde_json::json!({"validation_revision":"proof","model":"ltx-2-19b-distilled:fp8","width":512,"height":512,"fps":24,"single_clip_max_frames":97,"overlap_frames":17,"join_mode":"continuous","clip_count":"9","part_count":"9","added_frames":"720","source_frames":"0","total_frames":"720","added_duration_ms":"30000","total_duration_ms":"30000","warnings":[]});
        let manifest = serde_json::json!({"version":1,"request":{"generation":{"model":"ltx-2-19b-distilled:fp8","prompt":"walking","width":512,"height":512,"steps":4,"seed":42},"duration_ms":"30000"},"template":{"model":"ltx-2-19b-distilled:fp8","width":512,"height":512,"steps":4,"guidance":0,"fps":24,"motion_tail_frames":17,"seed":42,"stages":[{"prompt":"walking","frames":97}]},"frozen":{"runtime_model_id":"runtime","config":mold_core::ModelConfig::default(),"model_fingerprint":"frozen"},"envelope":{"maximum_frames":97,"minimum_frames":1,"frame_step":8,"frame_offset":1,"overlap_frames":17},"retained_source":null,"plan":plan});
        let json = serde_json::to_string(&manifest).unwrap();
        mold_db::video_jobs::StoredVideoJob{job:serde_json::from_value(serde_json::json!({"id":"parent","state":"queued","plan":plan,"completed_clips":"0","active_child_id":null,"output_filename":null,"error":null,"error_detail":null,"created_at_ms":1,"updated_at_ms":1,"can_cancel":true,"can_resume":false,"can_delete":true})).unwrap(),request_sha256:"request".into(),manifest_sha256:format!("{:x}",Sha256::digest(json.as_bytes())),manifest_json:json,revision:0,cancel_requested:false}
    }

    #[test]
    fn restart_hydration_preserves_original_endpoint_frame_locator() {
        let home = tempfile::tempdir().unwrap();
        let mut state = crate::state::AppState::for_tests();
        state.metadata_db = std::sync::Arc::new(Some(
            mold_db::MetadataDb::open(&home.path().join("mold.db")).unwrap(),
        ));
        let mut row = fixture();
        row.job.id = uuid::Uuid::new_v4().to_string();
        let mut manifest = read_manifest(&row).unwrap();
        manifest.request.generation.frames = Some(25);
        manifest.template.stages[0].keyframes = Some(vec![mold_core::KeyframeCondition {
            frame: 24,
            image: vec![1],
            name: None,
        }]);
        let restored = hydrate_original_request(&state, &row, &manifest).unwrap();
        assert_eq!(restored.generation.frames, Some(25));
        assert_eq!(restored.generation.keyframes.unwrap()[0].frame, 24);
    }
    #[test]
    fn parent_manifest_authenticates_exact_bytes_schema_and_public_plan() {
        let mut row = fixture();
        assert!(read_manifest(&row).is_ok());
        row.manifest_json = row.manifest_json.replace("walking", "running");
        assert!(read_manifest(&row).is_err());
        row.manifest_sha256 = format!("{:x}", Sha256::digest(row.manifest_json.as_bytes()));
        assert!(read_manifest(&row).is_ok());
        row.job.plan.model = "other-model".into();
        assert!(read_manifest(&row).is_err());
    }

    #[test]
    fn parent_manifest_never_adopts_unsealed_media() {
        let mut row = fixture();
        let mut value: serde_json::Value = serde_json::from_str(&row.manifest_json).unwrap();
        value["request"]["generation"]["source_image"] = serde_json::json!("AQID");
        row.manifest_json = serde_json::to_string(&value).unwrap();
        row.manifest_sha256 = format!("{:x}", Sha256::digest(row.manifest_json.as_bytes()));
        assert!(read_manifest(&row).is_err());
    }
}
