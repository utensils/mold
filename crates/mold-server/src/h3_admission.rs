//! MiniMax H3 admission authority.
//!
//! This module deliberately lands before the H3 runtime. It freezes every
//! input a future worker may consume, computes request-shape arithmetic, and
//! rejects plans unless landed header facts and a qualified CUDA runtime are
//! supplied. It does not make H3 runnable, parse gated model headers, or
//! manufacture measured workspace numbers.

use mold_core::manifest::ModelComponent;
use mold_core::minimax_h3::{self, GenerationReferencePreparedShape};
use mold_core::{GenerateRequest, GpuBackend};
use serde::Serialize;
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, BTreeSet};
use thiserror::Error;

const GIB: u64 = 1024 * 1024 * 1024;

pub(crate) const H3_ADMISSION_SCHEMA_VERSION: u32 = 2;
pub(crate) const H3_MAIN_BLOCK_COUNT: usize = 50;
pub(crate) const H3_QWEN_SELECTED_LANGUAGE_LAYERS: u32 = 50;
pub(crate) const H3_QWEN_MODEL_MAX_ROWS: u64 = 262_144;
pub(crate) const H3_MAX_PREFETCH_DEPTH: usize = 2;
pub(crate) const H3_TINY_MATH_MAX_PACKED_ROWS: u64 = 4_096;
pub(crate) const H3_CURATED_HOST_RAM_RECOMMENDATION_BYTES: u64 = 128 * GIB;

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub(crate) enum H3FrozenTask {
    Fl2va,
    Ref2va,
}

impl From<minimax_h3::Task> for H3FrozenTask {
    fn from(value: minimax_h3::Task) -> Self {
        match value {
            minimax_h3::Task::Fl2va => Self::Fl2va,
            minimax_h3::Task::Ref2va => Self::Ref2va,
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub(crate) enum H3PublishedLayout {
    OfficialFullBf16,
    ComfyPrunedInt8ConvrotNvfp4Awq,
}

impl From<minimax_h3::Layout> for H3PublishedLayout {
    fn from(value: minimax_h3::Layout) -> Self {
        match value {
            minimax_h3::Layout::OfficialBf16 => Self::OfficialFullBf16,
            minimax_h3::Layout::ComfyPrunedInt8ConvrotNvfp4Awq => {
                Self::ComfyPrunedInt8ConvrotNvfp4Awq
            }
        }
    }
}

#[derive(Clone, Debug, Eq, Ord, PartialEq, PartialOrd, Serialize)]
#[serde(rename_all = "snake_case")]
pub(crate) enum H3ArtifactRole {
    TransformerShard(u16),
    QwenShard(u16),
    ProcessorAsset(u16),
    VideoVaeShard(u16),
    AudioVae,
    SharedConfig(u16),
    TaskConfig(u16),
    VideoScheduler,
    AudioScheduler,
    /// A logical header-owned authority. It may be embedded in the checkpoint
    /// but still receives an independent fingerprint and byte range.
    QuantizationSidecar,
    /// Comfy's pruned AdaLN curves are not interchangeable with full AdaLN.
    PrunedAdaLnTable,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub(crate) enum H3ArtifactState {
    Pending,
    Landed,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub(crate) struct H3ArtifactFact {
    pub role: H3ArtifactRole,
    pub source_path: String,
    pub content_sha256: String,
    pub bytes: u64,
    pub state: H3ArtifactState,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub(crate) struct H3ArtifactInventory {
    pub model: String,
    pub task: H3FrozenTask,
    pub layout: H3PublishedLayout,
    pub source_revision: String,
    pub artifacts: Vec<H3ArtifactFact>,
}

impl H3ArtifactInventory {
    /// Exact source-pinned preview facts. They are intentionally pending: a
    /// manifest digest is not evidence that bytes have landed on this host.
    pub(crate) fn pending_from_manifest(model: &str) -> Result<Self, H3AdmissionError> {
        let contract = minimax_h3::capability_contract_for_model(model).ok_or_else(|| {
            H3AdmissionError::InvalidArtifactFacts(format!(
                "{model} has no pinned MiniMax H3 capability contract"
            ))
        })?;
        let manifest =
            mold_core::manifest::find_manifest(contract.canonical_model).ok_or_else(|| {
                H3AdmissionError::InvalidArtifactFacts(format!(
                    "{} has no pinned manifest",
                    contract.canonical_model
                ))
            })?;
        let manifest_contract = minimax_h3::manifest_contract(manifest).ok_or_else(|| {
            H3AdmissionError::InvalidArtifactFacts(format!(
                "{} is not an H3 manifest",
                manifest.name
            ))
        })?;
        let mut files = manifest.files.iter().collect::<Vec<_>>();
        files.sort_by(|left, right| left.hf_filename.cmp(&right.hf_filename));
        let mut transformer = 0_u16;
        let mut qwen = 0_u16;
        let mut processor = 0_u16;
        let mut video_vae = 0_u16;
        let mut shared_config = 0_u16;
        let mut task_config = 0_u16;
        let mut artifacts = Vec::with_capacity(files.len());
        for file in files {
            let role =
                match file.component {
                    ModelComponent::Transformer | ModelComponent::TransformerShard => {
                        let role = H3ArtifactRole::TransformerShard(transformer);
                        transformer = transformer.checked_add(1).ok_or(
                            H3AdmissionError::ArithmeticOverflow("transformer shard index"),
                        )?;
                        role
                    }
                    ModelComponent::TextEncoder => {
                        let role = H3ArtifactRole::QwenShard(qwen);
                        qwen = qwen
                            .checked_add(1)
                            .ok_or(H3AdmissionError::ArithmeticOverflow("Qwen shard index"))?;
                        role
                    }
                    ModelComponent::Processor => {
                        let role = H3ArtifactRole::ProcessorAsset(processor);
                        processor = processor.checked_add(1).ok_or(
                            H3AdmissionError::ArithmeticOverflow("processor asset index"),
                        )?;
                        role
                    }
                    ModelComponent::Vae => {
                        let role = H3ArtifactRole::VideoVaeShard(video_vae);
                        video_vae = video_vae.checked_add(1).ok_or(
                            H3AdmissionError::ArithmeticOverflow("video VAE shard index"),
                        )?;
                        role
                    }
                    ModelComponent::AudioVae => H3ArtifactRole::AudioVae,
                    ModelComponent::VideoScheduler => H3ArtifactRole::VideoScheduler,
                    ModelComponent::AudioScheduler => H3ArtifactRole::AudioScheduler,
                    ModelComponent::TaskConfig => {
                        let role = H3ArtifactRole::TaskConfig(task_config);
                        task_config = task_config
                            .checked_add(1)
                            .ok_or(H3AdmissionError::ArithmeticOverflow("task config index"))?;
                        role
                    }
                    ModelComponent::ModelConfig => {
                        let role = H3ArtifactRole::SharedConfig(shared_config);
                        shared_config = shared_config
                            .checked_add(1)
                            .ok_or(H3AdmissionError::ArithmeticOverflow("shared config index"))?;
                        role
                    }
                    _ => continue,
                };
            let artifact = minimax_h3::artifact_contract(manifest, file).ok_or_else(|| {
                H3AdmissionError::InvalidArtifactFacts(format!(
                    "{} is outside the pinned H3 artifact contract",
                    file.hf_filename
                ))
            })?;
            artifacts.push(H3ArtifactFact {
                role,
                source_path: file.hf_filename.clone(),
                content_sha256: artifact.identity.sha256.to_string(),
                bytes: file.size_bytes,
                state: H3ArtifactState::Pending,
            });
        }
        Ok(Self {
            model: contract.canonical_model.to_string(),
            task: contract.task.into(),
            layout: contract.layout.into(),
            source_revision: manifest_contract.source_revision.to_string(),
            artifacts,
        })
    }

    pub(crate) fn total_file_bytes(&self) -> Result<u64, H3AdmissionError> {
        checked_sum(
            self.artifacts.iter().map(|artifact| artifact.bytes),
            "artifact file bytes",
        )
    }

    fn validate_landed(&self) -> Result<(), H3AdmissionError> {
        let expected = Self::pending_from_manifest(&self.model)?;
        if self.model != expected.model
            || self.task != expected.task
            || self.layout != expected.layout
            || self.source_revision != expected.source_revision
        {
            return Err(H3AdmissionError::InvalidArtifactFacts(
                "H3 artifact inventory metadata differs from the pinned manifest".to_string(),
            ));
        }
        if self.artifacts.len() != expected.artifacts.len() {
            return Err(H3AdmissionError::InvalidArtifactFacts(format!(
                "H3 artifact inventory has {} entries, expected exactly {}",
                self.artifacts.len(),
                expected.artifacts.len()
            )));
        }
        let mut roles = BTreeSet::new();
        let mut pending = Vec::new();
        for (index, (artifact, expected)) in self
            .artifacts
            .iter()
            .zip(expected.artifacts.iter())
            .enumerate()
        {
            if !roles.insert(artifact.role.clone()) {
                return Err(H3AdmissionError::InvalidArtifactFacts(format!(
                    "duplicate H3 artifact role {:?}",
                    artifact.role
                )));
            }
            if artifact.role != expected.role
                || artifact.source_path != expected.source_path
                || artifact.content_sha256 != expected.content_sha256
                || artifact.bytes != expected.bytes
            {
                return Err(H3AdmissionError::InvalidArtifactFacts(format!(
                    "H3 artifact inventory entry {} differs from the pinned manifest",
                    index + 1
                )));
            }
            require_sha256(&artifact.content_sha256, "artifact content")?;
            if artifact.bytes == 0 || artifact.source_path.trim().is_empty() {
                return Err(H3AdmissionError::InvalidArtifactFacts(format!(
                    "H3 artifact {:?} has no path or bytes",
                    artifact.role
                )));
            }
            if artifact.state == H3ArtifactState::Pending {
                pending.push(artifact.source_path.clone());
            }
        }
        if !pending.is_empty() {
            return Err(H3AdmissionError::PendingArtifacts { paths: pending });
        }
        Ok(())
    }

    fn role_bytes(
        &self,
        predicate: impl Fn(&H3ArtifactRole) -> bool,
    ) -> Result<u64, H3AdmissionError> {
        checked_sum(
            self.artifacts
                .iter()
                .filter(|artifact| predicate(&artifact.role))
                .map(|artifact| artifact.bytes),
            "component file bytes",
        )
    }
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub(crate) struct H3BlockMemoryFact {
    pub index: u16,
    pub encoded_host_bytes: u64,
    pub resident_device_bytes: u64,
    pub dequantization_workspace_bytes: u64,
    pub content_fingerprint: String,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub(crate) enum H3AdaLnPolicy {
    FullProjection,
    PrunedCurve {
        grid_points: u32,
        basis_dim: u32,
        table_fingerprint: String,
    },
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub(crate) enum H3QuantizationPolicy {
    None,
    Int8ConvrotNvfp4Awq {
        transformer_policy_fingerprint: String,
        qwen_policy_fingerprint: String,
    },
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub(crate) struct H3CheckpointMemoryFacts {
    pub checkpoint_fingerprint: String,
    pub config_fingerprint: String,
    pub header_fingerprint: String,
    pub adaln: H3AdaLnPolicy,
    pub quantization: H3QuantizationPolicy,
    /// Top-level projections, refiners, heads, and other transformer tensors
    /// outside the fifty streamable main blocks.
    pub fixed_transformer_device_bytes: u64,
    pub qwen_device_bytes: u64,
    pub video_vae_device_bytes: u64,
    pub audio_vae_device_bytes: u64,
    pub blocks: Vec<H3BlockMemoryFact>,
}

impl H3CheckpointMemoryFacts {
    fn validate(&self, artifacts: &H3ArtifactInventory) -> Result<(), H3AdmissionError> {
        require_sha256(&self.checkpoint_fingerprint, "checkpoint")?;
        require_sha256(&self.config_fingerprint, "config")?;
        require_sha256(&self.header_fingerprint, "header")?;
        if self.blocks.len() != H3_MAIN_BLOCK_COUNT {
            return Err(H3AdmissionError::InvalidCheckpointFacts(format!(
                "H3 requires {H3_MAIN_BLOCK_COUNT} main block facts, got {}",
                self.blocks.len()
            )));
        }
        for (index, block) in self.blocks.iter().enumerate() {
            if usize::from(block.index) != index {
                return Err(H3AdmissionError::InvalidCheckpointFacts(format!(
                    "H3 block facts must be ordered 0..49; position {index} names {}",
                    block.index
                )));
            }
            if block.encoded_host_bytes == 0 || block.resident_device_bytes == 0 {
                return Err(H3AdmissionError::InvalidCheckpointFacts(format!(
                    "H3 block {index} has an empty encoded/device charge"
                )));
            }
            require_sha256(&block.content_fingerprint, "block")?;
        }
        let transformer_file_bytes =
            artifacts.role_bytes(|role| matches!(role, H3ArtifactRole::TransformerShard(_)))?;
        let encoded_blocks = checked_sum(
            self.blocks.iter().map(|block| block.encoded_host_bytes),
            "encoded transformer block bytes",
        )?;
        if encoded_blocks > transformer_file_bytes {
            return Err(H3AdmissionError::InvalidCheckpointFacts(format!(
                "H3 encoded block bytes {encoded_blocks} exceed transformer files {transformer_file_bytes}"
            )));
        }
        match (&self.adaln, &self.quantization, artifacts.layout) {
            (
                H3AdaLnPolicy::FullProjection,
                H3QuantizationPolicy::None,
                H3PublishedLayout::OfficialFullBf16,
            ) => {}
            (
                H3AdaLnPolicy::PrunedCurve {
                    table_fingerprint, ..
                },
                H3QuantizationPolicy::Int8ConvrotNvfp4Awq {
                    transformer_policy_fingerprint,
                    qwen_policy_fingerprint,
                },
                H3PublishedLayout::ComfyPrunedInt8ConvrotNvfp4Awq,
            ) => {
                require_sha256(table_fingerprint, "pruned AdaLN table")?;
                require_sha256(
                    transformer_policy_fingerprint,
                    "transformer quantization policy",
                )?;
                require_sha256(qwen_policy_fingerprint, "Qwen quantization policy")?;
                if self
                    .blocks
                    .iter()
                    .any(|block| block.dequantization_workspace_bytes == 0)
                {
                    return Err(H3AdmissionError::InvalidCheckpointFacts(
                        "quantized H3 blocks require exact dequantization workspace facts"
                            .to_string(),
                    ));
                }
            }
            _ => {
                return Err(H3AdmissionError::InvalidCheckpointFacts(
                    "H3 layout, AdaLN, and quantization authorities disagree".to_string(),
                ));
            }
        }
        Ok(())
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub(crate) enum H3QwenTruncationPolicy {
    RejectAboveModelMaximum { maximum_rows: u64 },
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub(crate) struct H3PreparedRows {
    pub qwen_output_text_rows: u64,
    pub qwen_vision_rows: u64,
    pub condition_visual_rows: u64,
    pub condition_audio_rows: u64,
    pub target_video_rows: u64,
    pub target_audio_rows: u64,
    pub total_packed_rows: u64,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub(crate) struct H3PreparedRequestShape {
    pub width: u32,
    pub height: u32,
    pub frames: u32,
    pub video_latent_frames: u64,
    pub audio_latents_per_channel: u64,
    pub audio_samples_per_channel: u64,
    pub reference_shapes: Vec<GenerationReferencePreparedShape>,
    pub reference_fingerprint: String,
    pub conditioning_fingerprint: String,
    pub rows: H3PreparedRows,
}

impl H3PreparedRequestShape {
    pub(crate) fn from_prepared_request(
        request: &GenerateRequest,
        qwen_output_text_rows: u64,
        qwen_vision_rows: u64,
    ) -> Result<(H3FrozenTask, Self), H3AdmissionError> {
        let contract =
            minimax_h3::capability_contract_for_model(&request.model).ok_or_else(|| {
                H3AdmissionError::RequestContract(format!(
                    "{} is not a pinned MiniMax H3 model",
                    request.model
                ))
            })?;
        minimax_h3::validate_request_contract(request, contract.task)
            .map_err(|error| H3AdmissionError::RequestContract(error.to_string()))?;
        if qwen_output_text_rows == 0 || qwen_output_text_rows > H3_QWEN_MODEL_MAX_ROWS {
            return Err(H3AdmissionError::RequestContract(format!(
                "H3 Qwen output rows must be 1..={H3_QWEN_MODEL_MAX_ROWS}, got {qwen_output_text_rows}"
            )));
        }
        let width = u64::from(request.width);
        let height = u64::from(request.height);
        let frames = u64::from(request.frames.unwrap_or(minimax_h3::MIN_FRAMES));
        let video_latent_frames = frames
            .checked_sub(u64::from(minimax_h3::FRAME_OFFSET))
            .ok_or(H3AdmissionError::ArithmeticOverflow("video latent frames"))?
            / u64::from(minimax_h3::FRAME_STEP)
            * 5
            + 2;
        let rows_per_video_latent = width
            .checked_div(32)
            .and_then(|value| value.checked_mul(height / 32))
            .ok_or(H3AdmissionError::ArithmeticOverflow(
                "video rows per latent",
            ))?;
        let target_video_rows = video_latent_frames
            .checked_mul(rows_per_video_latent)
            .ok_or(H3AdmissionError::ArithmeticOverflow("target video rows"))?;
        // round(frames / 24 * 40) using exact integer arithmetic. Valid H3
        // frame counts never land on a half tie.
        let audio_latents_per_channel = frames
            .checked_mul(5)
            .and_then(|value| value.checked_add(1))
            .ok_or(H3AdmissionError::ArithmeticOverflow("target audio latents"))?
            / 3;
        let target_audio_rows = audio_latents_per_channel
            .checked_mul(u64::from(minimax_h3::AUDIO_CHANNELS))
            .ok_or(H3AdmissionError::ArithmeticOverflow("target audio rows"))?;
        let audio_samples_per_channel = frames
            .checked_mul(4_000)
            .and_then(|value| value.checked_add(1))
            .ok_or(H3AdmissionError::ArithmeticOverflow("target audio samples"))?
            / 3;

        let reference_shapes = request
            .references
            .as_deref()
            .map(minimax_h3::reference_prepared_shapes)
            .transpose()
            .map_err(|error| H3AdmissionError::RequestContract(error.to_string()))?
            .unwrap_or_default();
        let reference_metadata = request
            .references
            .as_deref()
            .unwrap_or_default()
            .iter()
            .enumerate()
            .map(|(index, reference)| {
                reference.redacted_metadata(index).ok_or_else(|| {
                    H3AdmissionError::RequestContract(format!(
                        "reference {} lacks a validated digest or prepared shape",
                        index + 1
                    ))
                })
            })
            .collect::<Result<Vec<_>, _>>()?;
        let reference_fingerprint =
            mold_core::generation_reference_fingerprint(&reference_metadata);
        let conditioning_fingerprint = match contract.task {
            minimax_h3::Task::Fl2va => fl2va_conditioning_fingerprint(request),
            minimax_h3::Task::Ref2va => reference_fingerprint.clone(),
        };
        let condition_visual_rows = match contract.task {
            minimax_h3::Task::Fl2va => {
                let keyframe_count =
                    u64::try_from(request.keyframes.as_deref().unwrap_or_default().len())
                        .map_err(|_| H3AdmissionError::ArithmeticOverflow("keyframe count"))?
                        .checked_add(u64::from(request.source_image.is_some()))
                        .ok_or(H3AdmissionError::ArithmeticOverflow(
                            "FL2VA boundary frame count",
                        ))?;
                keyframe_count
                    .checked_mul(rows_per_video_latent)
                    .ok_or(H3AdmissionError::ArithmeticOverflow("FL2VA condition rows"))?
            }
            minimax_h3::Task::Ref2va => checked_sum(
                reference_shapes.iter().map(|shape| shape.visual_rows),
                "reference visual rows",
            )?,
        };
        let condition_audio_rows = checked_sum(
            reference_shapes.iter().map(|shape| shape.audio_rows),
            "reference audio rows",
        )?;
        let total_packed_rows = checked_sum(
            [
                qwen_output_text_rows,
                condition_visual_rows,
                condition_audio_rows,
                target_video_rows,
                target_audio_rows,
            ],
            "packed rows",
        )?;
        Ok((
            contract.task.into(),
            Self {
                width: request.width,
                height: request.height,
                frames: u32::try_from(frames)
                    .map_err(|_| H3AdmissionError::ArithmeticOverflow("frames"))?,
                video_latent_frames,
                audio_latents_per_channel,
                audio_samples_per_channel,
                reference_shapes,
                reference_fingerprint,
                conditioning_fingerprint,
                rows: H3PreparedRows {
                    qwen_output_text_rows,
                    qwen_vision_rows,
                    condition_visual_rows,
                    condition_audio_rows,
                    target_video_rows,
                    target_audio_rows,
                    total_packed_rows,
                },
            },
        ))
    }

    fn validate_for(&self, task: H3FrozenTask) -> Result<(), H3AdmissionError> {
        if self.width == 0
            || self.height == 0
            || !self.width.is_multiple_of(minimax_h3::DIMENSION_ALIGNMENT)
            || !self.height.is_multiple_of(minimax_h3::DIMENSION_ALIGNMENT)
            || u64::from(self.width) * u64::from(self.height) > minimax_h3::MAX_PIXELS
            || !(minimax_h3::MIN_ASPECT_RATIO..=minimax_h3::MAX_ASPECT_RATIO)
                .contains(&(self.width as f64 / self.height as f64))
            || !minimax_h3::valid_frame_count(self.frames)
        {
            return Err(H3AdmissionError::RequestContract(
                "prepared H3 geometry is outside the fixed request grid".to_string(),
            ));
        }
        require_sha256(&self.reference_fingerprint, "reference")?;
        require_sha256(&self.conditioning_fingerprint, "conditioning")?;
        if self.rows.qwen_output_text_rows == 0
            || self.rows.qwen_output_text_rows > H3_QWEN_MODEL_MAX_ROWS
        {
            return Err(H3AdmissionError::RequestContract(format!(
                "prepared H3 Qwen output rows must be 1..={H3_QWEN_MODEL_MAX_ROWS}"
            )));
        }

        let frames = u64::from(self.frames);
        let expected_video_latent_frames = frames
            .checked_sub(u64::from(minimax_h3::FRAME_OFFSET))
            .ok_or(H3AdmissionError::ArithmeticOverflow("video latent frames"))?
            / u64::from(minimax_h3::FRAME_STEP)
            * 5
            + 2;
        let rows_per_video_latent = u64::from(self.width / 32)
            .checked_mul(u64::from(self.height / 32))
            .ok_or(H3AdmissionError::ArithmeticOverflow(
                "video rows per latent",
            ))?;
        let expected_target_video_rows = expected_video_latent_frames
            .checked_mul(rows_per_video_latent)
            .ok_or(H3AdmissionError::ArithmeticOverflow("target video rows"))?;
        let expected_audio_latents = frames
            .checked_mul(5)
            .and_then(|value| value.checked_add(1))
            .ok_or(H3AdmissionError::ArithmeticOverflow("target audio latents"))?
            / 3;
        let expected_target_audio_rows = expected_audio_latents
            .checked_mul(u64::from(minimax_h3::AUDIO_CHANNELS))
            .ok_or(H3AdmissionError::ArithmeticOverflow("target audio rows"))?;
        let expected_audio_samples = frames
            .checked_mul(4_000)
            .and_then(|value| value.checked_add(1))
            .ok_or(H3AdmissionError::ArithmeticOverflow("target audio samples"))?
            / 3;
        if self.video_latent_frames != expected_video_latent_frames
            || self.audio_latents_per_channel != expected_audio_latents
            || self.audio_samples_per_channel != expected_audio_samples
            || self.rows.target_video_rows != expected_target_video_rows
            || self.rows.target_audio_rows != expected_target_audio_rows
        {
            return Err(H3AdmissionError::RequestContract(
                "prepared H3 target row counts do not match the fixed 17n+5 geometry".to_string(),
            ));
        }

        match task {
            H3FrozenTask::Fl2va => {
                let condition_count = self
                    .rows
                    .condition_visual_rows
                    .checked_div(rows_per_video_latent)
                    .ok_or(H3AdmissionError::ArithmeticOverflow(
                        "FL2VA condition count",
                    ))?;
                if !self.reference_shapes.is_empty()
                    || self.rows.condition_audio_rows != 0
                    || !self
                        .rows
                        .condition_visual_rows
                        .is_multiple_of(rows_per_video_latent)
                    || condition_count > 2
                {
                    return Err(H3AdmissionError::RequestContract(
                        "prepared FL2VA rows are not zero, one, or two boundary frames".to_string(),
                    ));
                }
            }
            H3FrozenTask::Ref2va => {
                if self.reference_shapes.is_empty()
                    || self.reference_shapes.iter().any(|shape| {
                        shape.version != minimax_h3::REFERENCE_PREPROCESS_VERSION
                            || (shape.visual_rows == 0 && shape.audio_rows == 0)
                    })
                {
                    return Err(H3AdmissionError::RequestContract(
                        "prepared Ref2VA shapes are missing or use another preprocessing version"
                            .to_string(),
                    ));
                }
                let visual = checked_sum(
                    self.reference_shapes.iter().map(|shape| shape.visual_rows),
                    "reference visual rows",
                )?;
                let audio = checked_sum(
                    self.reference_shapes.iter().map(|shape| shape.audio_rows),
                    "reference audio rows",
                )?;
                if self.rows.condition_visual_rows != visual
                    || self.rows.condition_audio_rows != audio
                    || self.conditioning_fingerprint != self.reference_fingerprint
                {
                    return Err(H3AdmissionError::RequestContract(
                        "prepared Ref2VA row or fingerprint authority is inconsistent".to_string(),
                    ));
                }
            }
        }
        let total = checked_sum(
            [
                self.rows.qwen_output_text_rows,
                self.rows.condition_visual_rows,
                self.rows.condition_audio_rows,
                self.rows.target_video_rows,
                self.rows.target_audio_rows,
            ],
            "packed rows",
        )?;
        if self.rows.total_packed_rows != total {
            return Err(H3AdmissionError::RequestContract(
                "prepared H3 packed-row total is inconsistent".to_string(),
            ));
        }
        Ok(())
    }
}

fn fl2va_conditioning_fingerprint(request: &GenerateRequest) -> String {
    let mut hash = Sha256::new();
    hash.update(b"mold.minimax-h3.fl2va-conditioning.v1\0");
    if let Some(source) = request.source_image.as_deref() {
        hash.update(b"source-image\0");
        hash.update((source.len() as u64).to_le_bytes());
        hash.update(Sha256::digest(source));
    }
    for keyframe in request.keyframes.as_deref().unwrap_or_default() {
        hash.update(b"keyframe\0");
        hash.update(keyframe.frame.to_le_bytes());
        hash.update((keyframe.image.len() as u64).to_le_bytes());
        hash.update(Sha256::digest(&keyframe.image));
    }
    format!("{:x}", hash.finalize())
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize)]
pub(crate) struct H3LinearWorkspace {
    pub fixed_bytes: u64,
    pub bytes_per_row: u64,
}

impl H3LinearWorkspace {
    fn charge(self, rows: u64, label: &'static str) -> Result<u64, H3AdmissionError> {
        self.bytes_per_row
            .checked_mul(rows)
            .and_then(|bytes| bytes.checked_add(self.fixed_bytes))
            .ok_or(H3AdmissionError::ArithmeticOverflow(label))
    }
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub(crate) struct H3QualifiedRuntimeFacts {
    pub qualification_fingerprint: String,
    pub kernel_identity: String,
    pub minimum_compute_capability: (u16, u16),
    pub maximum_packed_rows: u64,
    pub attention_head_count: u32,
    pub attention_head_dim: u32,
    pub full_noncausal_attention: bool,
    pub lossless: bool,
    pub qwen_cpu_supported: bool,
    pub fixed_vram_bytes: u64,
    pub fixed_host_bytes: u64,
    pub attention_workspace: H3LinearWorkspace,
    pub ffn_workspace: H3LinearWorkspace,
    pub qwen_text_workspace: H3LinearWorkspace,
    pub qwen_vision_workspace: H3LinearWorkspace,
    pub condition_vae_workspace: H3LinearWorkspace,
    pub condition_latent_workspace: H3LinearWorkspace,
    pub target_video_workspace: H3LinearWorkspace,
    pub target_audio_workspace: H3LinearWorkspace,
    pub video_vae_tile_workspace_bytes: u64,
    pub audio_decode_workspace_bytes: u64,
    pub aac_mux_workspace: H3LinearWorkspace,
    pub vae_tile_width: u32,
    pub vae_tile_height: u32,
    pub vae_tile_overlap: u32,
}

impl H3QualifiedRuntimeFacts {
    fn validate_for(
        &self,
        device: &H3AdmissionDevice,
        shape: &H3PreparedRequestShape,
    ) -> Result<(), H3AdmissionError> {
        require_sha256(&self.qualification_fingerprint, "runtime qualification")?;
        if self.kernel_identity.trim().is_empty()
            || !self.lossless
            || !self.full_noncausal_attention
            || self.attention_head_count != 56
            || self.attention_head_dim != 128
        {
            return Err(H3AdmissionError::QualifiedAttentionRequired {
                packed_rows: shape.rows.total_packed_rows,
            });
        }
        let capability = device.compute_capability.ok_or_else(|| {
            H3AdmissionError::DeviceUnsupported(
                "H3 CUDA admission requires an explicit compute capability".to_string(),
            )
        })?;
        if capability < self.minimum_compute_capability {
            return Err(H3AdmissionError::DeviceUnsupported(format!(
                "H3 runtime requires CUDA compute capability {}.{}, got {}.{}",
                self.minimum_compute_capability.0,
                self.minimum_compute_capability.1,
                capability.0,
                capability.1
            )));
        }
        if shape.rows.total_packed_rows > self.maximum_packed_rows {
            return Err(H3AdmissionError::QualifiedAttentionRequired {
                packed_rows: shape.rows.total_packed_rows,
            });
        }
        if self.vae_tile_width == 0 || self.vae_tile_height == 0 {
            return Err(H3AdmissionError::InvalidRuntimeFacts(
                "H3 VAE tiling dimensions must be nonzero".to_string(),
            ));
        }
        Ok(())
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct H3AdmissionDevice {
    pub id: String,
    pub ordinal: usize,
    pub backend: GpuBackend,
    pub compute_capability: Option<(u16, u16)>,
    pub available_vram_bytes: u64,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct H3HostMemory {
    pub total_bytes: u64,
    pub available_bytes: u64,
}

impl H3HostMemory {
    pub(crate) fn safety_floor_bytes(self) -> u64 {
        let whole = self.total_bytes / 100;
        let remainder = self.total_bytes % 100;
        whole
            .saturating_mul(15)
            .saturating_add(remainder.saturating_mul(15) / 100)
            .max(8 * GIB)
    }

    pub(crate) fn headroom_bytes(self) -> u64 {
        self.available_bytes
            .saturating_sub(self.safety_floor_bytes())
    }

    fn validate(self) -> Result<(), H3AdmissionError> {
        if self.total_bytes == 0 || self.available_bytes > self.total_bytes {
            return Err(H3AdmissionError::InvalidHostMemory {
                total_bytes: self.total_bytes,
                available_bytes: self.available_bytes,
            });
        }
        Ok(())
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub(crate) enum H3QwenPlacement {
    AssignedGpuThenDrop,
    HostCpuThenDrop,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub(crate) struct H3FrozenAttentionPolicy {
    pub kernel_identity: String,
    pub qualification_fingerprint: String,
    pub full_noncausal: bool,
    pub lossless: bool,
    pub head_count: u32,
    pub head_dim: u32,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub(crate) struct H3FrozenVaeTilingPolicy {
    pub width: u32,
    pub height: u32,
    pub overlap: u32,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub(crate) struct H3FrozenAudioPolicy {
    pub required: bool,
    pub sample_rate_hz: u32,
    pub channels: u32,
    pub latent_channels: u32,
    pub fp32_vae: bool,
    pub keep_vae_resident_during_denoise: bool,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub(crate) struct H3MemoryBreakdown {
    pub artifact_host_bytes: u64,
    pub runtime_host_workspace_bytes: u64,
    pub predicted_host_increment_bytes: u64,
    pub host_safety_floor_bytes: u64,
    pub host_headroom_bytes: u64,
    pub qwen_phase_vram_bytes: u64,
    pub denoise_phase_vram_bytes: u64,
    pub video_vae_phase_vram_bytes: u64,
    pub audio_decode_phase_vram_bytes: u64,
    pub predicted_vram_peak_bytes: u64,
    pub attention_workspace_bytes: u64,
    pub ffn_workspace_bytes: u64,
    pub qwen_activation_bytes: u64,
    pub condition_vae_workspace_bytes: u64,
    pub condition_latent_workspace_bytes: u64,
    pub target_video_workspace_bytes: u64,
    pub target_audio_workspace_bytes: u64,
    pub decoder_tile_workspace_bytes: u64,
    pub aac_mux_staging_bytes: u64,
    pub resident_block_bytes: u64,
    pub streamed_block_bytes: u64,
    pub prefetch_device_bytes: u64,
    pub dequantization_workspace_bytes: u64,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub(crate) struct H3FrozenExecutionPlan {
    pub schema_version: u32,
    pub canonical_model: String,
    pub task: H3FrozenTask,
    pub layout: H3PublishedLayout,
    pub device_id: String,
    pub device_ordinal: usize,
    pub compute_capability: (u16, u16),
    pub checkpoint_fingerprint: String,
    pub config_fingerprint: String,
    pub header_fingerprint: String,
    pub artifact_fingerprint: String,
    pub reference_fingerprint: String,
    pub conditioning_fingerprint: String,
    pub shape: H3PreparedRequestShape,
    pub qwen_language_layers: u32,
    pub qwen_truncation: H3QwenTruncationPolicy,
    pub qwen_placement: H3QwenPlacement,
    pub attention: H3FrozenAttentionPolicy,
    pub adaln: H3AdaLnPolicy,
    pub quantization: H3QuantizationPolicy,
    pub resident_block_count: u32,
    pub prefetch_depth: u32,
    pub vae_tiling: H3FrozenVaeTilingPolicy,
    pub audio: H3FrozenAudioPolicy,
    pub memory: H3MemoryBreakdown,
    pub execution_fingerprint: String,
}

impl H3FrozenExecutionPlan {
    /// Recheck immediately before worker dispatch. The fingerprint is computed
    /// over the complete frozen plan with this field cleared, so no hot/reload
    /// path may mutate placement or memory facts behind the scheduler grant.
    pub(crate) fn validate_execution_fingerprint(&self) -> Result<(), H3AdmissionError> {
        if self.execution_fingerprint.len() != 64
            || !self
                .execution_fingerprint
                .bytes()
                .all(|byte| byte.is_ascii_hexdigit())
        {
            return Err(H3AdmissionError::InvalidRuntimeFacts(
                "H3 execution fingerprint is not SHA-256".to_string(),
            ));
        }
        let expected = self.execution_fingerprint.clone();
        let mut canonical = self.clone();
        canonical.execution_fingerprint.clear();
        if fingerprint(&canonical)? != expected {
            return Err(H3AdmissionError::InvalidRuntimeFacts(
                "H3 frozen execution plan changed after admission".to_string(),
            ));
        }
        Ok(())
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct H3AdmissionPolicy {
    pub prefetch_depth: usize,
    pub prefer_resident_audio_vae: bool,
}

impl Default for H3AdmissionPolicy {
    fn default() -> Self {
        Self {
            prefetch_depth: 1,
            prefer_resident_audio_vae: true,
        }
    }
}

#[derive(Debug, Error, Eq, PartialEq)]
pub(crate) enum H3AdmissionError {
    #[error("MiniMax H3 request contract is invalid: {0}")]
    RequestContract(String),
    #[error("MiniMax H3 artifact facts are invalid: {0}")]
    InvalidArtifactFacts(String),
    #[error("MiniMax H3 checkpoint facts are invalid: {0}")]
    InvalidCheckpointFacts(String),
    #[error("MiniMax H3 runtime facts are invalid: {0}")]
    InvalidRuntimeFacts(String),
    #[error("MiniMax H3 placement is restricted to one Scheduler V2 GPU, got {device_count}")]
    SingleGpuRequired { device_count: usize },
    #[error("MiniMax H3 device is unsupported: {0}")]
    DeviceUnsupported(String),
    #[error(
        "MiniMax H3 host-memory facts are invalid: {available_bytes} available bytes exceeds {total_bytes} total bytes or total is zero"
    )]
    InvalidHostMemory {
        total_bytes: u64,
        available_bytes: u64,
    },
    #[error(
        "MiniMax H3 product-size packed attention ({packed_rows} rows) requires a qualified lossless CUDA runtime"
    )]
    QualifiedAttentionRequired { packed_rows: u64 },
    #[error("MiniMax H3 placement is pending landed artifacts: {paths:?}")]
    PendingArtifacts { paths: Vec<String> },
    #[error(
        "MiniMax H3 requires {required_bytes} host bytes but only {headroom_bytes} remain above the {safety_floor_bytes} safety floor"
    )]
    InsufficientHostRam {
        required_bytes: u64,
        headroom_bytes: u64,
        safety_floor_bytes: u64,
    },
    #[error("MiniMax H3 requires a {required_bytes}-byte VRAM peak but only {available_bytes} are available")]
    InsufficientVram {
        required_bytes: u64,
        available_bytes: u64,
    },
    #[error("MiniMax H3 admission arithmetic overflowed while computing {0}")]
    ArithmeticOverflow(&'static str),
}

impl H3AdmissionError {
    pub(crate) fn estimate_confidence(&self) -> Option<&'static str> {
        matches!(self, Self::PendingArtifacts { .. }).then_some("low")
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct StreamingChoice {
    resident_count: usize,
    resident_bytes: u64,
    streamed_bytes: u64,
    prefetch_bytes: u64,
    dequantization_bytes: u64,
    peak_bytes: u64,
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn plan_h3_admission(
    task: H3FrozenTask,
    shape: H3PreparedRequestShape,
    artifacts: &H3ArtifactInventory,
    checkpoint: &H3CheckpointMemoryFacts,
    runtime: Option<&H3QualifiedRuntimeFacts>,
    devices: &[H3AdmissionDevice],
    host: H3HostMemory,
    policy: H3AdmissionPolicy,
) -> Result<H3FrozenExecutionPlan, H3AdmissionError> {
    if devices.len() != 1 {
        return Err(H3AdmissionError::SingleGpuRequired {
            device_count: devices.len(),
        });
    }
    let device = &devices[0];
    if device.id.trim().is_empty() || device.backend != GpuBackend::Cuda {
        return Err(H3AdmissionError::DeviceUnsupported(
            "the initial H3 authority is CUDA-only".to_string(),
        ));
    }
    if task != artifacts.task {
        return Err(H3AdmissionError::InvalidArtifactFacts(
            "request task and artifact task differ".to_string(),
        ));
    }
    shape.validate_for(task)?;
    if policy.prefetch_depth > H3_MAX_PREFETCH_DEPTH {
        return Err(H3AdmissionError::InvalidRuntimeFacts(format!(
            "H3 prefetch depth {} exceeds bounded maximum {H3_MAX_PREFETCH_DEPTH}",
            policy.prefetch_depth
        )));
    }
    host.validate()?;
    artifacts.validate_landed()?;
    checkpoint.validate(artifacts)?;
    let runtime = runtime.ok_or(H3AdmissionError::QualifiedAttentionRequired {
        packed_rows: shape.rows.total_packed_rows,
    })?;
    runtime.validate_for(device, &shape)?;

    let attention_workspace = runtime
        .attention_workspace
        .charge(shape.rows.total_packed_rows, "attention workspace")?;
    let ffn_workspace = runtime
        .ffn_workspace
        .charge(shape.rows.total_packed_rows, "FFN workspace")?;
    let qwen_text_workspace = runtime
        .qwen_text_workspace
        .charge(shape.rows.qwen_output_text_rows, "Qwen text workspace")?;
    let qwen_vision_workspace = runtime
        .qwen_vision_workspace
        .charge(shape.rows.qwen_vision_rows, "Qwen vision workspace")?;
    let qwen_activation_bytes = qwen_text_workspace
        .checked_add(qwen_vision_workspace)
        .ok_or(H3AdmissionError::ArithmeticOverflow(
            "Qwen activation workspace",
        ))?;
    let condition_vae_workspace = runtime
        .condition_vae_workspace
        .charge(shape.rows.condition_visual_rows, "condition VAE workspace")?;
    let condition_latent_workspace = runtime.condition_latent_workspace.charge(
        shape.rows.condition_visual_rows,
        "condition latent workspace",
    )?;
    let target_video_workspace = runtime
        .target_video_workspace
        .charge(shape.rows.target_video_rows, "target video workspace")?;
    let target_audio_workspace = runtime.target_audio_workspace.charge(
        shape
            .rows
            .target_audio_rows
            .checked_add(shape.rows.condition_audio_rows)
            .ok_or(H3AdmissionError::ArithmeticOverflow("all audio rows"))?,
        "target audio workspace",
    )?;
    let aac_mux_staging = runtime
        .aac_mux_workspace
        .charge(shape.audio_samples_per_channel, "AAC mux staging")?;

    let qwen_gpu_peak = runtime
        .fixed_vram_bytes
        .checked_add(checkpoint.qwen_device_bytes)
        .and_then(|value| value.checked_add(qwen_activation_bytes))
        .ok_or(H3AdmissionError::ArithmeticOverflow("Qwen phase"))?;
    let (qwen_placement, qwen_phase_vram) = match qwen_gpu_peak {
        peak if peak <= device.available_vram_bytes => (H3QwenPlacement::AssignedGpuThenDrop, peak),
        _ if runtime.qwen_cpu_supported => {
            (H3QwenPlacement::HostCpuThenDrop, runtime.fixed_vram_bytes)
        }
        peak => {
            return Err(H3AdmissionError::InsufficientVram {
                required_bytes: peak,
                available_bytes: device.available_vram_bytes,
            });
        }
    };

    let denoise_workspace = checked_sum(
        [
            runtime.fixed_vram_bytes,
            checkpoint.fixed_transformer_device_bytes,
            attention_workspace,
            ffn_workspace,
            condition_latent_workspace,
            target_video_workspace,
            target_audio_workspace,
        ],
        "denoise base workspace",
    )?;
    let without_audio = choose_streaming(
        &checkpoint.blocks,
        denoise_workspace,
        device.available_vram_bytes,
        policy.prefetch_depth,
    )?;
    let with_audio = if policy.prefer_resident_audio_vae {
        denoise_workspace
            .checked_add(checkpoint.audio_vae_device_bytes)
            .and_then(|base| {
                choose_streaming(
                    &checkpoint.blocks,
                    base,
                    device.available_vram_bytes,
                    policy.prefetch_depth,
                )
                .ok()
            })
    } else {
        None
    };
    let (streaming, keep_audio_vae_resident) =
        with_audio.map_or((without_audio, false), |plan| (plan, true));
    if streaming.peak_bytes > device.available_vram_bytes {
        return Err(H3AdmissionError::InsufficientVram {
            required_bytes: streaming.peak_bytes,
            available_bytes: device.available_vram_bytes,
        });
    }
    let video_vae_phase_vram = checked_sum(
        [
            runtime.fixed_vram_bytes,
            checkpoint.video_vae_device_bytes,
            runtime.video_vae_tile_workspace_bytes,
            condition_vae_workspace,
            target_video_workspace,
        ],
        "video VAE phase",
    )?;
    let audio_decode_phase_vram = checked_sum(
        [
            runtime.fixed_vram_bytes,
            checkpoint.audio_vae_device_bytes,
            runtime.audio_decode_workspace_bytes,
            target_audio_workspace,
        ],
        "audio decode phase",
    )?;
    let predicted_vram_peak = [
        qwen_phase_vram,
        streaming.peak_bytes,
        video_vae_phase_vram,
        audio_decode_phase_vram,
    ]
    .into_iter()
    .max()
    .unwrap_or_default();
    if predicted_vram_peak > device.available_vram_bytes {
        return Err(H3AdmissionError::InsufficientVram {
            required_bytes: predicted_vram_peak,
            available_bytes: device.available_vram_bytes,
        });
    }

    // Every mapped/streamed artifact remains reserved in the host ledger.
    // This deliberately rejects the 134 GiB full stack on a 128 GiB host even
    // when the filesystem is sparse or mmap would initially fault few pages.
    let artifact_host_bytes = artifacts.total_file_bytes()?;
    let qwen_host_workspace = if qwen_placement == H3QwenPlacement::HostCpuThenDrop {
        checkpoint
            .qwen_device_bytes
            .checked_add(qwen_activation_bytes)
            .ok_or(H3AdmissionError::ArithmeticOverflow("Qwen host workspace"))?
    } else {
        0
    };
    let runtime_host_workspace = checked_sum(
        [
            runtime.fixed_host_bytes,
            aac_mux_staging,
            qwen_host_workspace,
        ],
        "runtime host workspace",
    )?;
    let predicted_host_increment = artifact_host_bytes
        .checked_add(runtime_host_workspace)
        .ok_or(H3AdmissionError::ArithmeticOverflow("host memory charge"))?;
    let host_floor = host.safety_floor_bytes();
    let host_headroom = host.headroom_bytes();
    if predicted_host_increment > host_headroom {
        return Err(H3AdmissionError::InsufficientHostRam {
            required_bytes: predicted_host_increment,
            headroom_bytes: host_headroom,
            safety_floor_bytes: host_floor,
        });
    }

    let artifact_fingerprint = fingerprint(artifacts)?;
    let mut plan = H3FrozenExecutionPlan {
        schema_version: H3_ADMISSION_SCHEMA_VERSION,
        canonical_model: artifacts.model.clone(),
        task,
        layout: artifacts.layout,
        device_id: device.id.clone(),
        device_ordinal: device.ordinal,
        compute_capability: device.compute_capability.ok_or_else(|| {
            H3AdmissionError::DeviceUnsupported(
                "H3 CUDA admission requires an exact compute capability".to_string(),
            )
        })?,
        checkpoint_fingerprint: checkpoint.checkpoint_fingerprint.clone(),
        config_fingerprint: checkpoint.config_fingerprint.clone(),
        header_fingerprint: checkpoint.header_fingerprint.clone(),
        artifact_fingerprint,
        reference_fingerprint: shape.reference_fingerprint.clone(),
        conditioning_fingerprint: shape.conditioning_fingerprint.clone(),
        shape,
        qwen_language_layers: H3_QWEN_SELECTED_LANGUAGE_LAYERS,
        qwen_truncation: H3QwenTruncationPolicy::RejectAboveModelMaximum {
            maximum_rows: H3_QWEN_MODEL_MAX_ROWS,
        },
        qwen_placement,
        attention: H3FrozenAttentionPolicy {
            kernel_identity: runtime.kernel_identity.clone(),
            qualification_fingerprint: runtime.qualification_fingerprint.clone(),
            full_noncausal: runtime.full_noncausal_attention,
            lossless: runtime.lossless,
            head_count: runtime.attention_head_count,
            head_dim: runtime.attention_head_dim,
        },
        adaln: checkpoint.adaln.clone(),
        quantization: checkpoint.quantization.clone(),
        resident_block_count: u32::try_from(streaming.resident_count)
            .map_err(|_| H3AdmissionError::ArithmeticOverflow("resident block count"))?,
        prefetch_depth: u32::try_from(policy.prefetch_depth)
            .map_err(|_| H3AdmissionError::ArithmeticOverflow("prefetch depth"))?,
        vae_tiling: H3FrozenVaeTilingPolicy {
            width: runtime.vae_tile_width,
            height: runtime.vae_tile_height,
            overlap: runtime.vae_tile_overlap,
        },
        audio: H3FrozenAudioPolicy {
            required: true,
            sample_rate_hz: minimax_h3::AUDIO_SAMPLE_RATE_HZ,
            channels: minimax_h3::AUDIO_CHANNELS,
            latent_channels: 32,
            fp32_vae: true,
            keep_vae_resident_during_denoise: keep_audio_vae_resident,
        },
        memory: H3MemoryBreakdown {
            artifact_host_bytes,
            runtime_host_workspace_bytes: runtime_host_workspace,
            predicted_host_increment_bytes: predicted_host_increment,
            host_safety_floor_bytes: host_floor,
            host_headroom_bytes: host_headroom,
            qwen_phase_vram_bytes: qwen_phase_vram,
            denoise_phase_vram_bytes: streaming.peak_bytes,
            video_vae_phase_vram_bytes: video_vae_phase_vram,
            audio_decode_phase_vram_bytes: audio_decode_phase_vram,
            predicted_vram_peak_bytes: predicted_vram_peak,
            attention_workspace_bytes: attention_workspace,
            ffn_workspace_bytes: ffn_workspace,
            qwen_activation_bytes,
            condition_vae_workspace_bytes: condition_vae_workspace,
            condition_latent_workspace_bytes: condition_latent_workspace,
            target_video_workspace_bytes: target_video_workspace,
            target_audio_workspace_bytes: target_audio_workspace,
            decoder_tile_workspace_bytes: runtime.video_vae_tile_workspace_bytes,
            aac_mux_staging_bytes: aac_mux_staging,
            resident_block_bytes: streaming.resident_bytes,
            streamed_block_bytes: streaming.streamed_bytes,
            prefetch_device_bytes: streaming.prefetch_bytes,
            dequantization_workspace_bytes: streaming.dequantization_bytes,
        },
        execution_fingerprint: String::new(),
    };
    plan.execution_fingerprint = fingerprint(&plan)?;
    plan.validate_execution_fingerprint()?;
    Ok(plan)
}

/// Fully synthetic/injected inputs for the ordinary generation admission
/// bridge. The shipping bridge has no source for these facts: authorization,
/// a runnable capability, exact landed artifact identities, and qualified
/// header/runtime evidence must all land before a real source is registered.
pub(crate) struct H3NormalAdmissionInputs {
    pub(crate) task: H3FrozenTask,
    pub(crate) shape: H3PreparedRequestShape,
    pub(crate) artifacts: H3ArtifactInventory,
    pub(crate) checkpoint: H3CheckpointMemoryFacts,
    pub(crate) runtime: H3QualifiedRuntimeFacts,
    pub(crate) devices: Vec<H3AdmissionDevice>,
    pub(crate) host: H3HostMemory,
    pub(crate) policy: H3AdmissionPolicy,
}

#[derive(Debug, Error, Eq, PartialEq)]
pub(crate) enum H3NormalAdmissionError {
    #[error("{0}")]
    Activation(String),
    #[error("MiniMax H3 public runnable capability remains disabled")]
    RuntimeUnavailable,
    #[error("{H3_REAL_ADMISSION_SOURCE_DISABLED}")]
    RealSourceDisabled,
    #[error(transparent)]
    Admission(#[from] H3AdmissionError),
}

pub(crate) const H3_REAL_ADMISSION_SOURCE_DISABLED: &str =
    "MiniMax H3 real artifact/header admission source is runtime-disabled pending authorization and exact artifact identities";

fn plan_normal_h3_admission_with_gates<G, S>(
    model: &str,
    family: &str,
    activation_gate: G,
    runtime_available: bool,
    source: S,
) -> Result<H3FrozenExecutionPlan, H3NormalAdmissionError>
where
    G: FnOnce() -> Result<(), mold_core::ModelActivationError>,
    S: FnOnce() -> Result<H3NormalAdmissionInputs, H3NormalAdmissionError>,
{
    activation_gate().map_err(|error| H3NormalAdmissionError::Activation(error.to_string()))?;
    if !runtime_available {
        return Err(H3NormalAdmissionError::RuntimeUnavailable);
    }
    let inputs = source()?;
    if inputs.artifacts.model
        != minimax_h3::capability_contract_for_model(model)
            .map(|contract| contract.canonical_model)
            .unwrap_or_default()
        || !minimax_h3::is_family(family)
    {
        return Err(H3NormalAdmissionError::Admission(
            H3AdmissionError::InvalidArtifactFacts(
                "normal H3 admission source differs from the requested model family".to_string(),
            ),
        ));
    }
    Ok(plan_h3_admission(
        inputs.task,
        inputs.shape,
        &inputs.artifacts,
        &inputs.checkpoint,
        Some(&inputs.runtime),
        &inputs.devices,
        inputs.host,
        inputs.policy,
    )?)
}

/// Ordinary placement entry point while H3 is disabled.
///
/// Ordering is load-bearing: the #831 gate runs first, then the public runtime
/// registry, and only then may a source gather paths, headers, or runtime
/// facts. Today the source is an unconditional error, so even a future legal
/// decision cannot accidentally start artifact work without a separate loader
/// registration and exact identities.
pub(crate) fn reject_normal_h3_admission(model: &str, family: &str) -> H3NormalAdmissionError {
    let result = plan_normal_h3_admission_with_gates(
        model,
        family,
        || mold_core::require_model_activation(model, Some(family)),
        minimax_h3::runnable_capability_contract_for_model(model).is_some()
            && mold_inference::production_family_capability_for_family(family).is_some(),
        || Err(H3NormalAdmissionError::RealSourceDisabled),
    );
    match result {
        Err(error) => error,
        Ok(_) => H3NormalAdmissionError::RealSourceDisabled,
    }
}

/// Bind a fully admitted server plan into the inference factory without
/// exposing paths or bytes and without activating an H3 loader.
///
/// Construction is transactional: `engine_config` is mutated only after the
/// execution fingerprint, exact CUDA route, component set, attention evidence,
/// streaming policy, and layout-specific quantization all agree.
pub(crate) fn bind_h3_factory_authority(
    plan: &H3FrozenExecutionPlan,
    artifacts: &H3ArtifactInventory,
    checkpoint: &H3CheckpointMemoryFacts,
    device: &H3AdmissionDevice,
    engine_config: &mut mold_inference::FrozenEngineConfig,
) -> Result<(), H3AdmissionError> {
    plan.validate_execution_fingerprint()?;
    artifacts.validate_landed()?;
    checkpoint.validate(artifacts)?;
    let compute_capability = device.compute_capability.ok_or_else(|| {
        H3AdmissionError::DeviceUnsupported(
            "H3 factory binding requires an exact CUDA compute capability".to_string(),
        )
    })?;
    if plan.schema_version != H3_ADMISSION_SCHEMA_VERSION
        || plan.canonical_model != artifacts.model
        || plan.task != artifacts.task
        || plan.layout != artifacts.layout
        || plan.device_id != device.id
        || plan.device_ordinal != device.ordinal
        || plan.compute_capability != compute_capability
        || device.backend != GpuBackend::Cuda
        || plan.checkpoint_fingerprint != checkpoint.checkpoint_fingerprint
        || plan.config_fingerprint != checkpoint.config_fingerprint
        || plan.header_fingerprint != checkpoint.header_fingerprint
        || plan.adaln != checkpoint.adaln
        || plan.quantization != checkpoint.quantization
        || plan.artifact_fingerprint != fingerprint(artifacts)?
    {
        return Err(H3AdmissionError::InvalidRuntimeFacts(
            "H3 server admission and factory authorities disagree".to_string(),
        ));
    }
    if plan.task != H3FrozenTask::Fl2va {
        return Err(H3AdmissionError::InvalidRuntimeFacts(
            "the current H3 Candle factory authority is FL2VA-only".to_string(),
        ));
    }
    if !minimax_h3::is_family(&engine_config.family) {
        return Err(H3AdmissionError::InvalidRuntimeFacts(
            "H3 admission cannot bind a non-H3 frozen engine family".to_string(),
        ));
    }
    let conditioner_placement = match plan.qwen_placement {
        H3QwenPlacement::AssignedGpuThenDrop => {
            mold_inference::H3FactoryConditionerPlacement::AssignedCudaThenDrop
        }
        H3QwenPlacement::HostCpuThenDrop => {
            mold_inference::H3FactoryConditionerPlacement::HostCpuThenDrop
        }
    };
    let quantization = match (&plan.quantization, &plan.adaln) {
        (H3QuantizationPolicy::None, H3AdaLnPolicy::FullProjection) => {
            mold_inference::H3FactoryQuantizationAuthority::OfficialBf16
        }
        (
            H3QuantizationPolicy::Int8ConvrotNvfp4Awq {
                transformer_policy_fingerprint,
                qwen_policy_fingerprint,
            },
            H3AdaLnPolicy::PrunedCurve {
                table_fingerprint, ..
            },
        ) => mold_inference::H3FactoryQuantizationAuthority::ComfyPrunedInt8ConvrotNvfp4Awq {
            transformer_policy_sha256: transformer_policy_fingerprint.clone(),
            qwen_policy_sha256: qwen_policy_fingerprint.clone(),
            pruned_adaln_table_sha256: table_fingerprint.clone(),
        },
        _ => {
            return Err(H3AdmissionError::InvalidCheckpointFacts(
                "H3 factory layout, AdaLN, and quantization authorities disagree".to_string(),
            ));
        }
    };
    let components = h3_factory_component_authorities(plan, artifacts, checkpoint)?;
    let authority = mold_inference::FrozenH3FactoryAuthority::new_contract_only(
        mold_inference::H3FactoryAuthorityInput {
            model: artifacts.model.clone(),
            device_id: plan.device_id.clone(),
            device_ordinal: plan.device_ordinal,
            compute_capability,
            execution_fingerprint: plan.execution_fingerprint.clone(),
            conditioner_placement,
            qwen_parameter_bytes: checkpoint.qwen_device_bytes,
            qwen_activation_workspace_bytes: plan.memory.qwen_activation_bytes,
            resident_block_count: plan.resident_block_count,
            prefetch_depth: plan.prefetch_depth,
            attention_backend: engine_config.attention_backend,
            attention_chunk: engine_config.attention_chunk,
            attention_kernel_identity: plan.attention.kernel_identity.clone(),
            attention_qualification_sha256: plan.attention.qualification_fingerprint.clone(),
            attention_full_noncausal: plan.attention.full_noncausal,
            attention_lossless: plan.attention.lossless,
            attention_head_count: plan.attention.head_count,
            attention_head_dim: plan.attention.head_dim,
            block_offload: true,
            quantization,
            components,
        },
    )
    .map_err(|error| H3AdmissionError::InvalidRuntimeFacts(error.to_string()))?;
    engine_config.h3_factory_authority = Some(authority);
    Ok(())
}

fn h3_factory_component_authorities(
    plan: &H3FrozenExecutionPlan,
    artifacts: &H3ArtifactInventory,
    checkpoint: &H3CheckpointMemoryFacts,
) -> Result<Vec<mold_inference::H3FactoryComponentAuthority>, H3AdmissionError> {
    use mold_inference::H3FactoryComponentRole as FactoryRole;

    let mut grouped = BTreeMap::<FactoryRole, Vec<&H3ArtifactFact>>::new();
    for artifact in &artifacts.artifacts {
        let role = match &artifact.role {
            H3ArtifactRole::QwenShard(_) | H3ArtifactRole::ProcessorAsset(_) => {
                FactoryRole::Conditioner
            }
            H3ArtifactRole::TransformerShard(_)
            | H3ArtifactRole::TaskConfig(_)
            | H3ArtifactRole::VideoScheduler
            | H3ArtifactRole::AudioScheduler
            | H3ArtifactRole::QuantizationSidecar
            | H3ArtifactRole::PrunedAdaLnTable => FactoryRole::Transformer,
            H3ArtifactRole::VideoVaeShard(_) => FactoryRole::VisualVae,
            H3ArtifactRole::AudioVae => FactoryRole::AudioVae,
            H3ArtifactRole::SharedConfig(_) => {
                if artifact.source_path.starts_with("text_encoder/") {
                    FactoryRole::Conditioner
                } else if artifact.source_path.starts_with("vae/") {
                    FactoryRole::VisualVae
                } else if artifact.source_path.starts_with("audio_vae/") {
                    FactoryRole::AudioVae
                } else {
                    FactoryRole::Transformer
                }
            }
        };
        grouped.entry(role).or_default().push(artifact);
    }

    [
        FactoryRole::Conditioner,
        FactoryRole::Transformer,
        FactoryRole::VisualVae,
        FactoryRole::AudioVae,
    ]
    .into_iter()
    .map(|role| {
        let members = grouped.get(&role).ok_or_else(|| {
            H3AdmissionError::InvalidArtifactFacts(format!(
                "H3 factory component {role:?} has no landed artifacts"
            ))
        })?;
        let role_id = match role {
            FactoryRole::Conditioner => "conditioner",
            FactoryRole::Transformer => "transformer",
            FactoryRole::VisualVae => "visual-vae",
            FactoryRole::AudioVae => "audio-vae",
        };
        let content_sha256 = fingerprint(&(
            "mold.minimax-h3.logical-component-content.v1",
            role_id,
            members,
        ))?;
        let validation_sha256 = fingerprint(&(
            "mold.minimax-h3.logical-component-validation.v1",
            role_id,
            plan,
            checkpoint,
            members,
        ))?;
        mold_inference::H3FactoryComponentAuthority::new(role, content_sha256, validation_sha256)
            .map_err(|error| H3AdmissionError::InvalidArtifactFacts(error.to_string()))
    })
    .collect()
}

fn choose_streaming(
    blocks: &[H3BlockMemoryFact],
    base_bytes: u64,
    available_bytes: u64,
    prefetch_depth: usize,
) -> Result<StreamingChoice, H3AdmissionError> {
    let total_device = checked_sum(
        blocks.iter().map(|block| block.resident_device_bytes),
        "total transformer device bytes",
    )?;
    for resident_count in (0..=blocks.len()).rev() {
        let resident_bytes = checked_sum(
            blocks
                .iter()
                .take(resident_count)
                .map(|block| block.resident_device_bytes),
            "resident block bytes",
        )?;
        let streamed = &blocks[resident_count..];
        let slots = prefetch_depth.saturating_add(1);
        let prefetch_bytes = streamed
            .windows(slots.min(streamed.len()).max(1))
            .map(|window| {
                checked_sum(
                    window.iter().map(|block| block.resident_device_bytes),
                    "prefetch window",
                )
            })
            .collect::<Result<Vec<_>, _>>()?
            .into_iter()
            .max()
            .unwrap_or(0);
        // Resident blocks also require their dequantization scratch while
        // they are materialized. Charging only the streamed suffix would make
        // the all-resident hot path report zero scratch and could understate
        // the preparation peak when an early resident block is the largest.
        let dequantization_bytes = blocks
            .iter()
            .map(|block| block.dequantization_workspace_bytes)
            .max()
            .unwrap_or(0);
        let peak = checked_sum(
            [
                base_bytes,
                resident_bytes,
                prefetch_bytes,
                dequantization_bytes,
            ],
            "streaming peak",
        )?;
        if peak <= available_bytes {
            return Ok(StreamingChoice {
                resident_count,
                resident_bytes,
                streamed_bytes: total_device.saturating_sub(resident_bytes),
                prefetch_bytes,
                dequantization_bytes,
                peak_bytes: peak,
            });
        }
    }
    let largest_block = blocks
        .iter()
        .map(|block| block.resident_device_bytes)
        .max()
        .unwrap_or_default();
    Err(H3AdmissionError::InsufficientVram {
        required_bytes: base_bytes.saturating_add(largest_block),
        available_bytes,
    })
}

pub(crate) fn preparation_authority_bytes(
    request: &GenerateRequest,
    family: Option<&str>,
) -> Option<Vec<u8>> {
    if !family.is_some_and(minimax_h3::is_family)
        && minimax_h3::capability_contract_for_model(&request.model).is_none()
    {
        return None;
    }
    #[derive(Serialize)]
    struct Authority<'a> {
        schema_version: u32,
        reference_preprocess_version: u32,
        qwen_language_layers: u32,
        qwen_model_max_rows: u64,
        attention_head_count: u32,
        attention_head_dim: u32,
        product_attention_requires_qualified_cuda: bool,
        tiny_math_maximum_packed_rows: u64,
        one_gpu_only: bool,
        maximum_prefetch_depth: usize,
        host_safety_floor_percent: u32,
        host_safety_floor_minimum_bytes: u64,
        model: &'a str,
        width: u32,
        height: u32,
        frames: Option<u32>,
        reference_shapes: Option<Vec<GenerationReferencePreparedShape>>,
    }
    let reference_shapes = request
        .references
        .as_deref()
        .and_then(|references| minimax_h3::reference_prepared_shapes(references).ok());
    serde_json::to_vec(&Authority {
        schema_version: H3_ADMISSION_SCHEMA_VERSION,
        reference_preprocess_version: minimax_h3::REFERENCE_PREPROCESS_VERSION,
        qwen_language_layers: H3_QWEN_SELECTED_LANGUAGE_LAYERS,
        qwen_model_max_rows: H3_QWEN_MODEL_MAX_ROWS,
        attention_head_count: 56,
        attention_head_dim: 128,
        product_attention_requires_qualified_cuda: true,
        tiny_math_maximum_packed_rows: H3_TINY_MATH_MAX_PACKED_ROWS,
        one_gpu_only: true,
        maximum_prefetch_depth: H3_MAX_PREFETCH_DEPTH,
        host_safety_floor_percent: 15,
        host_safety_floor_minimum_bytes: 8 * GIB,
        model: &request.model,
        width: request.width,
        height: request.height,
        frames: request.frames,
        reference_shapes,
    })
    .ok()
}

fn checked_sum(
    values: impl IntoIterator<Item = u64>,
    label: &'static str,
) -> Result<u64, H3AdmissionError> {
    values.into_iter().try_fold(0_u64, |total, value| {
        total
            .checked_add(value)
            .ok_or(H3AdmissionError::ArithmeticOverflow(label))
    })
}

fn require_sha256(value: &str, label: &'static str) -> Result<(), H3AdmissionError> {
    if value.len() == 64 && value.bytes().all(|byte| byte.is_ascii_hexdigit()) {
        Ok(())
    } else {
        Err(H3AdmissionError::InvalidArtifactFacts(format!(
            "{label} fingerprint is not SHA-256"
        )))
    }
}

fn fingerprint(value: &impl Serialize) -> Result<String, H3AdmissionError> {
    let bytes = serde_json::to_vec(value)
        .map_err(|error| H3AdmissionError::InvalidRuntimeFacts(error.to_string()))?;
    let mut hash = Sha256::new();
    hash.update(b"mold.minimax-h3.admission.v2\0");
    hash.update(bytes);
    Ok(format!("{:x}", hash.finalize()))
}

#[cfg(test)]
mod tests {
    use super::*;
    use mold_core::{
        GenerationReference, GenerationReferenceAuthority, GenerationReferenceProvenance,
    };

    const MIB: u64 = 1024 * 1024;

    fn sha(byte: u8) -> String {
        format!("{byte:02x}").repeat(32)
    }

    fn request(model: &str, frames: u32) -> GenerateRequest {
        serde_json::from_value(serde_json::json!({
            "prompt": "a lighthouse in a storm",
            "model": model,
            "width": 1344,
            "height": 768,
            "steps": 50,
            "guidance": 0.0,
            // H3 always denoises the complete trajectory and therefore has no
            // user-adjustable img2img-style strength control.
            "strength": 1.0,
            "seed": 42,
            "batch_size": 1,
            "frames": frames,
            "fps": 24,
            "output_format": "mp4"
        }))
        .unwrap()
    }

    fn landed_inventory(model: &str) -> H3ArtifactInventory {
        let mut inventory = H3ArtifactInventory::pending_from_manifest(model).unwrap();
        for artifact in &mut inventory.artifacts {
            artifact.state = H3ArtifactState::Landed;
        }
        inventory
    }

    fn checkpoint(inventory: &H3ArtifactInventory) -> H3CheckpointMemoryFacts {
        let transformer_bytes = inventory
            .role_bytes(|role| matches!(role, H3ArtifactRole::TransformerShard(_)))
            .unwrap();
        let qwen_bytes = inventory
            .role_bytes(|role| matches!(role, H3ArtifactRole::QwenShard(_)))
            .unwrap();
        let video_vae_bytes = inventory
            .role_bytes(|role| matches!(role, H3ArtifactRole::VideoVaeShard(_)))
            .unwrap();
        let audio_vae_bytes = inventory
            .role_bytes(|role| matches!(role, H3ArtifactRole::AudioVae))
            .unwrap();
        // Synthetic header facts use the real source-pinned aggregate bytes
        // but do not claim a measured production block distribution. The
        // production constructor remains blocked until #839 supplies audited
        // per-tensor ranges and #840 supplies qualified runtime workspaces.
        let encoded_per_block = transformer_bytes / 60;
        let quantized = inventory.layout == H3PublishedLayout::ComfyPrunedInt8ConvrotNvfp4Awq;
        let device_per_block = if quantized {
            encoded_per_block.saturating_mul(2)
        } else {
            encoded_per_block
        };
        let blocks = (0..H3_MAIN_BLOCK_COUNT)
            .map(|index| H3BlockMemoryFact {
                index: u16::try_from(index).unwrap(),
                encoded_host_bytes: encoded_per_block,
                resident_device_bytes: device_per_block,
                dequantization_workspace_bytes: if quantized { 64 * MIB } else { 0 },
                content_fingerprint: sha(u8::try_from(index + 1).unwrap()),
            })
            .collect();
        let (adaln, quantization) = if quantized {
            (
                H3AdaLnPolicy::PrunedCurve {
                    grid_points: 1_000,
                    basis_dim: 32,
                    table_fingerprint: sha(201),
                },
                H3QuantizationPolicy::Int8ConvrotNvfp4Awq {
                    transformer_policy_fingerprint: sha(202),
                    qwen_policy_fingerprint: sha(203),
                },
            )
        } else {
            (H3AdaLnPolicy::FullProjection, H3QuantizationPolicy::None)
        };
        H3CheckpointMemoryFacts {
            checkpoint_fingerprint: sha(210),
            config_fingerprint: sha(211),
            header_fingerprint: sha(212),
            adaln,
            quantization,
            fixed_transformer_device_bytes: if quantized { 2 * GIB } else { 10 * GIB },
            qwen_device_bytes: qwen_bytes,
            video_vae_device_bytes: video_vae_bytes,
            audio_vae_device_bytes: audio_vae_bytes,
            blocks,
        }
    }

    fn qualified_runtime() -> H3QualifiedRuntimeFacts {
        H3QualifiedRuntimeFacts {
            qualification_fingerprint: sha(220),
            kernel_identity: "synthetic-lossless-packed-attention-v1".to_string(),
            minimum_compute_capability: (8, 0),
            maximum_packed_rows: 200_000,
            attention_head_count: 56,
            attention_head_dim: 128,
            full_noncausal_attention: true,
            lossless: true,
            qwen_cpu_supported: true,
            fixed_vram_bytes: GIB,
            fixed_host_bytes: 256 * MIB,
            attention_workspace: H3LinearWorkspace {
                fixed_bytes: 128 * MIB,
                bytes_per_row: 1_024,
            },
            ffn_workspace: H3LinearWorkspace {
                fixed_bytes: 64 * MIB,
                bytes_per_row: 2_048,
            },
            qwen_text_workspace: H3LinearWorkspace {
                fixed_bytes: 32 * MIB,
                bytes_per_row: 20_480,
            },
            qwen_vision_workspace: H3LinearWorkspace {
                fixed_bytes: 32 * MIB,
                bytes_per_row: 9_216,
            },
            condition_vae_workspace: H3LinearWorkspace {
                fixed_bytes: 64 * MIB,
                bytes_per_row: 512,
            },
            condition_latent_workspace: H3LinearWorkspace {
                fixed_bytes: 0,
                bytes_per_row: 192,
            },
            target_video_workspace: H3LinearWorkspace {
                fixed_bytes: 0,
                bytes_per_row: 768,
            },
            target_audio_workspace: H3LinearWorkspace {
                fixed_bytes: 0,
                bytes_per_row: 512,
            },
            video_vae_tile_workspace_bytes: 2 * GIB,
            audio_decode_workspace_bytes: GIB,
            aac_mux_workspace: H3LinearWorkspace {
                fixed_bytes: 16 * MIB,
                bytes_per_row: 8,
            },
            vae_tile_width: 512,
            vae_tile_height: 512,
            vae_tile_overlap: 64,
        }
    }

    fn cuda(vram: u64) -> H3AdmissionDevice {
        H3AdmissionDevice {
            id: "gpu-0".to_string(),
            ordinal: 0,
            backend: GpuBackend::Cuda,
            compute_capability: Some((8, 9)),
            available_vram_bytes: vram,
        }
    }

    fn prepared(model: &str, frames: u32) -> (H3FrozenTask, H3PreparedRequestShape) {
        H3PreparedRequestShape::from_prepared_request(&request(model, frames), 128, 0).unwrap()
    }

    #[test]
    fn normal_admission_never_reads_facts_before_activation_and_runtime_gates() {
        let source_calls = std::cell::Cell::new(0_u8);
        let error = plan_normal_h3_admission_with_gates(
            minimax_h3::FL2VA_COMFY,
            minimax_h3::FAMILY,
            || Err(mold_core::ModelActivationError),
            true,
            || {
                source_calls.set(source_calls.get() + 1);
                Err(H3NormalAdmissionError::RealSourceDisabled)
            },
        )
        .unwrap_err();
        assert!(matches!(error, H3NormalAdmissionError::Activation(_)));
        assert_eq!(source_calls.get(), 0);

        let error = plan_normal_h3_admission_with_gates(
            minimax_h3::FL2VA_COMFY,
            minimax_h3::FAMILY,
            || Ok(()),
            false,
            || {
                source_calls.set(source_calls.get() + 1);
                Err(H3NormalAdmissionError::RealSourceDisabled)
            },
        )
        .unwrap_err();
        assert_eq!(error, H3NormalAdmissionError::RuntimeUnavailable);
        assert_eq!(source_calls.get(), 0);
    }

    #[test]
    fn normal_admission_bridge_delegates_to_the_canonical_h3_planner() {
        let inventory = landed_inventory(minimax_h3::FL2VA_COMFY);
        let checkpoint = checkpoint(&inventory);
        let runtime = qualified_runtime();
        let (task, shape) = prepared(minimax_h3::FL2VA_COMFY, 124);
        let plan = plan_normal_h3_admission_with_gates(
            minimax_h3::FL2VA_COMFY,
            minimax_h3::FAMILY,
            || Ok(()),
            true,
            || {
                Ok(H3NormalAdmissionInputs {
                    task,
                    shape,
                    artifacts: inventory,
                    checkpoint,
                    runtime,
                    devices: vec![cuda(24 * GIB)],
                    host: H3HostMemory {
                        total_bytes: 96 * GIB,
                        available_bytes: 96 * GIB,
                    },
                    policy: H3AdmissionPolicy::default(),
                })
            },
        )
        .unwrap();

        assert_eq!(plan.canonical_model, minimax_h3::FL2VA_COMFY);
        assert_eq!(plan.device_id, "gpu-0");
        assert_eq!(plan.compute_capability, (8, 9));
        assert_eq!(plan.execution_fingerprint.len(), 64);
        plan.validate_execution_fingerprint().unwrap();
    }

    #[test]
    fn target_rows_are_exact_for_124_and_362_frame_products() {
        let (_, short) = prepared(minimax_h3::FL2VA_COMFY, 124);
        assert_eq!(short.video_latent_frames, 37);
        assert_eq!(short.rows.target_video_rows, 37_296);
        assert_eq!(short.audio_latents_per_channel, 207);
        assert_eq!(short.rows.target_audio_rows, 414);
        assert_eq!(short.audio_samples_per_channel, 165_333);
        assert_eq!(short.rows.total_packed_rows, 37_838);

        let (_, long) = prepared(minimax_h3::FL2VA_COMFY, 362);
        assert_eq!(long.video_latent_frames, 107);
        assert_eq!(long.rows.target_video_rows, 107_856);
        assert_eq!(long.audio_latents_per_channel, 603);
        assert_eq!(long.rows.target_audio_rows, 1_206);
        assert_eq!(long.audio_samples_per_channel, 482_667);
        assert_eq!(long.rows.total_packed_rows, 109_190);
    }

    #[test]
    fn fl2va_source_image_is_charged_as_one_visual_condition() {
        let mut request = request(minimax_h3::FL2VA_COMFY, 124);
        request.source_image = Some(vec![1, 2, 3, 4]);
        let (task, shape) =
            H3PreparedRequestShape::from_prepared_request(&request, 128, 1_024).unwrap();
        assert_eq!(task, H3FrozenTask::Fl2va);
        assert_eq!(shape.rows.condition_visual_rows, 42 * 24);
        assert_eq!(shape.rows.condition_audio_rows, 0);
        assert_eq!(shape.rows.total_packed_rows, 37_838 + 42 * 24);

        request.source_image = Some(vec![1, 2, 3, 5]);
        let (_, changed) =
            H3PreparedRequestShape::from_prepared_request(&request, 128, 1_024).unwrap();
        assert_ne!(
            shape.conditioning_fingerprint,
            changed.conditioning_fingerprint
        );
    }

    #[test]
    fn prepared_ref2va_mix_prices_visual_audio_and_qwen_rows_in_order() {
        let mut request = request(minimax_h3::REF2VA_COMFY, 124);
        request.references = Some(vec![
            GenerationReference::Image {
                media: GenerationReferenceAuthority::ServerPath {
                    path: "/private/reference-image.png".to_string(),
                },
                provenance: GenerationReferenceProvenance {
                    name: Some("image.png".to_string()),
                    sha256: Some(sha(1)),
                },
                mime_type: "image/png".to_string(),
                width: 1024,
                height: 768,
            },
            GenerationReference::Video {
                media: GenerationReferenceAuthority::ServerPath {
                    path: "/private/reference-video.mp4".to_string(),
                },
                provenance: GenerationReferenceProvenance {
                    name: Some("video.mp4".to_string()),
                    sha256: Some(sha(2)),
                },
                mime_type: "video/mp4".to_string(),
                width: 1280,
                height: 720,
                frame_count: Some(120),
                duration_ms: 4_000,
                fps: 30.0,
                has_audio: true,
                audio_duration_ms: Some(4_000),
                audio_sample_count: Some(128_000),
                audio_sample_rate: Some(32_000),
                audio_channels: Some(2),
            },
            GenerationReference::Audio {
                media: GenerationReferenceAuthority::ServerPath {
                    path: "/private/reference-audio.wav".to_string(),
                },
                provenance: GenerationReferenceProvenance {
                    name: Some("audio.wav".to_string()),
                    sha256: Some(sha(3)),
                },
                mime_type: "audio/wav".to_string(),
                duration_ms: 4_000,
                sample_rate: 32_000,
                channels: 2,
                sample_count: Some(128_000),
            },
        ]);
        let (task, shape) =
            H3PreparedRequestShape::from_prepared_request(&request, 512, 4_608).unwrap();
        assert_eq!(task, H3FrozenTask::Ref2va);
        assert_eq!(shape.reference_shapes.len(), 3);
        assert!(shape.rows.condition_visual_rows > 0);
        assert!(shape.rows.condition_audio_rows > 0);
        assert_eq!(shape.rows.qwen_output_text_rows, 512);
        assert_eq!(shape.rows.qwen_vision_rows, 4_608);
        assert_eq!(shape.reference_fingerprint.len(), 64);
        assert_eq!(
            shape.rows.total_packed_rows,
            512 + shape.rows.condition_visual_rows
                + shape.rows.condition_audio_rows
                + shape.rows.target_video_rows
                + shape.rows.target_audio_rows
        );
    }

    #[test]
    fn pinned_full_and_curated_stacks_charge_every_physical_artifact() {
        let official =
            H3ArtifactInventory::pending_from_manifest(minimax_h3::FL2VA_OFFICIAL).unwrap();
        assert_eq!(
            official
                .role_bytes(|role| matches!(role, H3ArtifactRole::TransformerShard(_)))
                .unwrap(),
            66_280_504_216
        );
        assert_eq!(
            official
                .role_bytes(|role| matches!(role, H3ArtifactRole::QwenShard(_)))
                .unwrap(),
            66_714_912_872
        );
        assert_eq!(
            official
                .role_bytes(|role| matches!(role, H3ArtifactRole::VideoVaeShard(_)))
                .unwrap(),
            10_415_558_888
        );
        assert_eq!(
            official
                .role_bytes(|role| matches!(role, H3ArtifactRole::AudioVae))
                .unwrap(),
            605_429_340
        );
        assert!(official.total_file_bytes().unwrap() > 144_016_405_316);

        let curated = H3ArtifactInventory::pending_from_manifest(minimax_h3::FL2VA_COMFY).unwrap();
        assert_eq!(
            curated
                .role_bytes(|role| matches!(role, H3ArtifactRole::TransformerShard(_)))
                .unwrap(),
            20_970_379_616
        );
        assert_eq!(
            curated
                .role_bytes(|role| matches!(role, H3ArtifactRole::QwenShard(_)))
                .unwrap(),
            15_687_142_551
        );
        assert_eq!(
            curated
                .role_bytes(|role| matches!(role, H3ArtifactRole::VideoVaeShard(_)))
                .unwrap(),
            5_207_808_496
        );
        assert_eq!(
            curated
                .role_bytes(|role| matches!(role, H3ArtifactRole::AudioVae))
                .unwrap(),
            605_254_808
        );
        assert!(curated.total_file_bytes().unwrap() > 42_470_585_471);
    }

    #[test]
    fn host_floor_matches_scheduler_and_full_stack_cannot_fit_128_gib() {
        let forty = H3HostMemory {
            total_bytes: 40 * GIB,
            available_bytes: 20 * GIB,
        };
        assert_eq!(forty.safety_floor_bytes(), 8 * GIB);
        assert_eq!(forty.headroom_bytes(), 12 * GIB);
        let eighty = H3HostMemory {
            total_bytes: 80 * GIB,
            available_bytes: 80 * GIB,
        };
        assert_eq!(eighty.safety_floor_bytes(), 12 * GIB);
        assert_eq!(eighty.headroom_bytes(), 68 * GIB);
        let enormous = H3HostMemory {
            total_bytes: u64::MAX,
            available_bytes: u64::MAX,
        };
        assert_eq!(
            enormous.safety_floor_bytes(),
            (u64::MAX / 100) * 15 + ((u64::MAX % 100) * 15 / 100)
        );
        assert_eq!(H3_CURATED_HOST_RAM_RECOMMENDATION_BYTES, 128 * GIB);

        let inventory = landed_inventory(minimax_h3::FL2VA_OFFICIAL);
        let checkpoint = checkpoint(&inventory);
        let runtime = qualified_runtime();
        let (task, shape) = prepared(minimax_h3::FL2VA_OFFICIAL, 124);
        let error = plan_h3_admission(
            task,
            shape,
            &inventory,
            &checkpoint,
            Some(&runtime),
            &[cuda(80 * GIB)],
            H3HostMemory {
                total_bytes: 128 * GIB,
                available_bytes: 128 * GIB,
            },
            H3AdmissionPolicy::default(),
        )
        .unwrap_err();
        assert!(matches!(
            error,
            H3AdmissionError::InsufficientHostRam { .. }
        ));
    }

    #[test]
    fn malformed_host_and_incomplete_inventory_fail_before_admission() {
        let mut inventory = landed_inventory(minimax_h3::FL2VA_COMFY);
        let checkpoint = checkpoint(&inventory);
        let runtime = qualified_runtime();
        let (task, shape) = prepared(minimax_h3::FL2VA_COMFY, 124);
        let error = plan_h3_admission(
            task,
            shape.clone(),
            &inventory,
            &checkpoint,
            Some(&runtime),
            &[cuda(24 * GIB)],
            H3HostMemory {
                total_bytes: 96 * GIB,
                available_bytes: 96 * GIB + 1,
            },
            H3AdmissionPolicy::default(),
        )
        .unwrap_err();
        assert!(matches!(error, H3AdmissionError::InvalidHostMemory { .. }));

        inventory.artifacts.pop();
        let error = plan_h3_admission(
            task,
            shape,
            &inventory,
            &checkpoint,
            Some(&runtime),
            &[cuda(24 * GIB)],
            H3HostMemory {
                total_bytes: 96 * GIB,
                available_bytes: 96 * GIB,
            },
            H3AdmissionPolicy::default(),
        )
        .unwrap_err();
        assert!(matches!(error, H3AdmissionError::InvalidArtifactFacts(_)));
    }

    #[test]
    fn qwen_cpu_fallback_charges_parameters_and_never_masks_overflow() {
        let inventory = landed_inventory(minimax_h3::FL2VA_COMFY);
        let mut checkpoint = checkpoint(&inventory);
        checkpoint.qwen_device_bytes = 30 * GIB;
        let runtime = qualified_runtime();
        let (task, shape) = prepared(minimax_h3::FL2VA_COMFY, 124);
        let plan = plan_h3_admission(
            task,
            shape.clone(),
            &inventory,
            &checkpoint,
            Some(&runtime),
            &[cuda(24 * GIB)],
            H3HostMemory {
                total_bytes: 160 * GIB,
                available_bytes: 160 * GIB,
            },
            H3AdmissionPolicy::default(),
        )
        .unwrap();
        assert_eq!(plan.qwen_placement, H3QwenPlacement::HostCpuThenDrop);
        assert!(
            plan.memory.runtime_host_workspace_bytes
                >= checkpoint.qwen_device_bytes + plan.memory.qwen_activation_bytes
        );

        checkpoint.qwen_device_bytes = u64::MAX;
        let error = plan_h3_admission(
            task,
            shape,
            &inventory,
            &checkpoint,
            Some(&runtime),
            &[cuda(24 * GIB)],
            H3HostMemory {
                total_bytes: 160 * GIB,
                available_bytes: 160 * GIB,
            },
            H3AdmissionPolicy::default(),
        )
        .unwrap_err();
        assert_eq!(error, H3AdmissionError::ArithmeticOverflow("Qwen phase"));
    }

    #[test]
    fn linear_workspace_overflow_and_resident_dequant_scratch_fail_closed() {
        let inventory = landed_inventory(minimax_h3::FL2VA_COMFY);
        let checkpoint = checkpoint(&inventory);
        let mut runtime = qualified_runtime();
        let (task, shape) = prepared(minimax_h3::FL2VA_COMFY, 124);
        runtime.attention_workspace.bytes_per_row = u64::MAX;
        let error = plan_h3_admission(
            task,
            shape.clone(),
            &inventory,
            &checkpoint,
            Some(&runtime),
            &[cuda(24 * GIB)],
            H3HostMemory {
                total_bytes: 96 * GIB,
                available_bytes: 96 * GIB,
            },
            H3AdmissionPolicy::default(),
        )
        .unwrap_err();
        assert_eq!(
            error,
            H3AdmissionError::ArithmeticOverflow("attention workspace")
        );

        runtime = qualified_runtime();
        runtime.ffn_workspace.bytes_per_row = u64::MAX;
        let error = plan_h3_admission(
            task,
            shape,
            &inventory,
            &checkpoint,
            Some(&runtime),
            &[cuda(24 * GIB)],
            H3HostMemory {
                total_bytes: 96 * GIB,
                available_bytes: 96 * GIB,
            },
            H3AdmissionPolicy::default(),
        )
        .unwrap_err();
        assert_eq!(error, H3AdmissionError::ArithmeticOverflow("FFN workspace"));

        let blocks = (0..H3_MAIN_BLOCK_COUNT)
            .map(|index| H3BlockMemoryFact {
                index: u16::try_from(index).unwrap(),
                encoded_host_bytes: 1,
                resident_device_bytes: 1,
                dequantization_workspace_bytes: if index == 0 { 1_000 } else { 1 },
                content_fingerprint: sha(u8::try_from(index + 1).unwrap()),
            })
            .collect::<Vec<_>>();
        let streaming = choose_streaming(&blocks, 0, 1_050, 1).unwrap();
        assert_eq!(streaming.resident_count, H3_MAIN_BLOCK_COUNT);
        assert_eq!(streaming.dequantization_bytes, 1_000);
        assert_eq!(streaming.peak_bytes, 1_050);
    }

    #[test]
    fn curated_plan_freezes_every_policy_and_phase_charge_on_one_gpu() {
        let inventory = landed_inventory(minimax_h3::FL2VA_COMFY);
        let checkpoint = checkpoint(&inventory);
        let runtime = qualified_runtime();
        let (task, shape) = prepared(minimax_h3::FL2VA_COMFY, 124);
        let plan = plan_h3_admission(
            task,
            shape.clone(),
            &inventory,
            &checkpoint,
            Some(&runtime),
            &[cuda(24 * GIB)],
            H3HostMemory {
                total_bytes: 96 * GIB,
                available_bytes: 96 * GIB,
            },
            H3AdmissionPolicy::default(),
        )
        .unwrap();
        assert_eq!(plan.schema_version, H3_ADMISSION_SCHEMA_VERSION);
        assert_eq!(plan.device_id, "gpu-0");
        assert_eq!(plan.shape, shape);
        assert_eq!(plan.qwen_language_layers, 50);
        assert_eq!(
            plan.qwen_truncation,
            H3QwenTruncationPolicy::RejectAboveModelMaximum {
                maximum_rows: 262_144
            }
        );
        assert!(plan.attention.lossless && plan.attention.full_noncausal);
        assert!(matches!(plan.adaln, H3AdaLnPolicy::PrunedCurve { .. }));
        assert!(matches!(
            plan.quantization,
            H3QuantizationPolicy::Int8ConvrotNvfp4Awq { .. }
        ));
        assert!(plan.resident_block_count < 50);
        assert_eq!(plan.prefetch_depth, 1);
        assert_eq!(plan.audio.sample_rate_hz, 32_000);
        assert_eq!(plan.audio.channels, 2);
        assert!(plan.audio.fp32_vae);
        assert!(plan.memory.streamed_block_bytes > 0);
        assert!(plan.memory.prefetch_device_bytes > 0);
        assert!(plan.memory.dequantization_workspace_bytes > 0);
        assert_eq!(plan.execution_fingerprint.len(), 64);
        assert!(plan.memory.predicted_vram_peak_bytes <= 24 * GIB);
        assert!(plan.memory.predicted_host_increment_bytes <= plan.memory.host_headroom_bytes);

        let mut alternate_runtime = runtime.clone();
        alternate_runtime.kernel_identity = "alternate-qualified-kernel".to_string();
        alternate_runtime.qualification_fingerprint = sha(221);
        let alternate = plan_h3_admission(
            task,
            shape,
            &inventory,
            &checkpoint,
            Some(&alternate_runtime),
            &[cuda(24 * GIB)],
            H3HostMemory {
                total_bytes: 96 * GIB,
                available_bytes: 96 * GIB,
            },
            H3AdmissionPolicy::default(),
        )
        .unwrap();
        assert_ne!(plan.execution_fingerprint, alternate.execution_fingerprint);
        plan.validate_execution_fingerprint().unwrap();
        let mut tampered = plan.clone();
        tampered.memory.predicted_vram_peak_bytes -= 1;
        let error = tampered.validate_execution_fingerprint().unwrap_err();
        assert!(matches!(error, H3AdmissionError::InvalidRuntimeFacts(_)));
    }

    #[test]
    fn admitted_fl2va_plan_binds_exact_factory_route_and_component_authority() {
        let inventory = landed_inventory(minimax_h3::FL2VA_COMFY);
        let checkpoint = checkpoint(&inventory);
        let runtime = qualified_runtime();
        let device = cuda(24 * GIB);
        let (task, shape) = prepared(minimax_h3::FL2VA_COMFY, 124);
        let plan = plan_h3_admission(
            task,
            shape,
            &inventory,
            &checkpoint,
            Some(&runtime),
            std::slice::from_ref(&device),
            H3HostMemory {
                total_bytes: 96 * GIB,
                available_bytes: 96 * GIB,
            },
            H3AdmissionPolicy::default(),
        )
        .unwrap();
        let mut engine_config = mold_inference::FrozenEngineConfig::resolve(
            minimax_h3::FL2VA_COMFY,
            &mold_core::Config::default(),
        );
        engine_config.family = minimax_h3::FAMILY.to_string();

        bind_h3_factory_authority(&plan, &inventory, &checkpoint, &device, &mut engine_config)
            .unwrap();
        let authority = engine_config.h3_factory_authority.as_ref().unwrap();
        assert_eq!(plan.canonical_model, minimax_h3::FL2VA_COMFY);
        assert_eq!(authority.canonical_model(), minimax_h3::FL2VA_COMFY);
        assert_eq!(authority.device_id(), plan.device_id);
        assert_eq!(authority.device_ordinal(), plan.device_ordinal);
        assert_eq!(plan.compute_capability, (8, 9));
        assert_eq!(authority.compute_capability(), (8, 9));
        assert_eq!(
            authority.execution_fingerprint(),
            plan.execution_fingerprint
        );
        assert_eq!(
            authority.attention_qualification_sha256(),
            runtime.qualification_fingerprint
        );
        assert_eq!(
            authority.attention_backend(),
            engine_config.attention_backend
        );
        assert_eq!(authority.attention_chunk(), engine_config.attention_chunk);
        assert_eq!(
            authority.qwen_parameter_bytes(),
            checkpoint.qwen_device_bytes
        );
        assert_eq!(
            authority.qwen_activation_workspace_bytes(),
            plan.memory.qwen_activation_bytes
        );
        assert_eq!(
            authority.resident_block_count(),
            plan.resident_block_count as usize
        );
        assert_eq!(authority.prefetch_depth(), plan.prefetch_depth as usize);
        assert_eq!(authority.component_set_identity_sha256().len(), 64);
        assert!(authority.block_offload());
        assert!(matches!(
            authority.quantization(),
            mold_inference::H3FactoryQuantizationAuthority::ComfyPrunedInt8ConvrotNvfp4Awq { .. }
        ));
        let semantic =
            crate::execution_plan::ExecutionSemanticConfig::from_frozen(&engine_config).unwrap();
        assert_eq!(
            semantic.h3_factory_authority_sha256.as_deref(),
            Some(authority.identity_sha256())
        );
        assert!(!minimax_h3::capabilities(minimax_h3::Task::Fl2va).runtime_available);
    }

    #[test]
    fn official_bf16_admission_binds_without_inheriting_comfy_quantization() {
        let inventory = landed_inventory(minimax_h3::FL2VA_OFFICIAL);
        let checkpoint = checkpoint(&inventory);
        let runtime = qualified_runtime();
        let device = cuda(160 * GIB);
        let (task, shape) = prepared(minimax_h3::FL2VA_OFFICIAL, 124);
        let plan = plan_h3_admission(
            task,
            shape,
            &inventory,
            &checkpoint,
            Some(&runtime),
            std::slice::from_ref(&device),
            H3HostMemory {
                total_bytes: 512 * GIB,
                available_bytes: 512 * GIB,
            },
            H3AdmissionPolicy::default(),
        )
        .unwrap();
        let mut engine_config = mold_inference::FrozenEngineConfig::resolve(
            minimax_h3::FL2VA_OFFICIAL,
            &mold_core::Config::default(),
        );
        engine_config.family = minimax_h3::FAMILY.to_string();

        bind_h3_factory_authority(&plan, &inventory, &checkpoint, &device, &mut engine_config)
            .unwrap();
        let authority = engine_config.h3_factory_authority.as_ref().unwrap();
        assert_eq!(authority.canonical_model(), minimax_h3::FL2VA_OFFICIAL);
        assert_eq!(authority.compute_capability(), (8, 9));
        assert!(matches!(
            authority.quantization(),
            mold_inference::H3FactoryQuantizationAuthority::OfficialBf16
        ));
    }

    #[test]
    fn mutated_admission_or_component_facts_never_partially_bind_the_factory() {
        let inventory = landed_inventory(minimax_h3::FL2VA_COMFY);
        let checkpoint = checkpoint(&inventory);
        let runtime = qualified_runtime();
        let device = cuda(24 * GIB);
        let (task, shape) = prepared(minimax_h3::FL2VA_COMFY, 124);
        let plan = plan_h3_admission(
            task,
            shape,
            &inventory,
            &checkpoint,
            Some(&runtime),
            std::slice::from_ref(&device),
            H3HostMemory {
                total_bytes: 96 * GIB,
                available_bytes: 96 * GIB,
            },
            H3AdmissionPolicy::default(),
        )
        .unwrap();
        let mut engine_config = mold_inference::FrozenEngineConfig::resolve(
            minimax_h3::FL2VA_COMFY,
            &mold_core::Config::default(),
        );
        engine_config.family = minimax_h3::FAMILY.to_string();

        let mut rerouted = plan.clone();
        rerouted.device_ordinal += 1;
        assert!(bind_h3_factory_authority(
            &rerouted,
            &inventory,
            &checkpoint,
            &device,
            &mut engine_config,
        )
        .is_err());
        assert!(engine_config.h3_factory_authority.is_none());

        let mut changed_device = device.clone();
        changed_device.compute_capability = Some((9, 0));
        assert!(bind_h3_factory_authority(
            &plan,
            &inventory,
            &checkpoint,
            &changed_device,
            &mut engine_config,
        )
        .is_err());
        assert!(engine_config.h3_factory_authority.is_none());

        let mut changed_artifacts = inventory.clone();
        changed_artifacts.artifacts[0].content_sha256 = sha(254);
        assert!(bind_h3_factory_authority(
            &plan,
            &changed_artifacts,
            &checkpoint,
            &device,
            &mut engine_config,
        )
        .is_err());
        assert!(engine_config.h3_factory_authority.is_none());
    }

    #[test]
    fn pending_unqualified_multi_gpu_and_missing_quant_facts_fail_closed() {
        let pending = H3ArtifactInventory::pending_from_manifest(minimax_h3::FL2VA_COMFY).unwrap();
        let checkpoint = checkpoint(&pending);
        let runtime = qualified_runtime();
        let (task, shape) = prepared(minimax_h3::FL2VA_COMFY, 124);
        let error = plan_h3_admission(
            task,
            shape.clone(),
            &pending,
            &checkpoint,
            Some(&runtime),
            &[cuda(24 * GIB)],
            H3HostMemory {
                total_bytes: 96 * GIB,
                available_bytes: 96 * GIB,
            },
            H3AdmissionPolicy::default(),
        )
        .unwrap_err();
        assert_eq!(error.estimate_confidence(), Some("low"));

        let landed = landed_inventory(minimax_h3::FL2VA_COMFY);
        let error = plan_h3_admission(
            task,
            shape.clone(),
            &landed,
            &checkpoint,
            None,
            &[cuda(24 * GIB)],
            H3HostMemory {
                total_bytes: 96 * GIB,
                available_bytes: 96 * GIB,
            },
            H3AdmissionPolicy::default(),
        )
        .unwrap_err();
        assert_eq!(
            error,
            H3AdmissionError::QualifiedAttentionRequired {
                packed_rows: shape.rows.total_packed_rows
            }
        );

        let error = plan_h3_admission(
            task,
            shape.clone(),
            &landed,
            &checkpoint,
            Some(&runtime),
            &[cuda(24 * GIB), cuda(24 * GIB)],
            H3HostMemory {
                total_bytes: 96 * GIB,
                available_bytes: 96 * GIB,
            },
            H3AdmissionPolicy::default(),
        )
        .unwrap_err();
        assert_eq!(
            error,
            H3AdmissionError::SingleGpuRequired { device_count: 2 }
        );

        let mut missing_dequant = checkpoint;
        missing_dequant.blocks[0].dequantization_workspace_bytes = 0;
        let error = plan_h3_admission(
            task,
            shape,
            &landed,
            &missing_dequant,
            Some(&runtime),
            &[cuda(24 * GIB)],
            H3HostMemory {
                total_bytes: 96 * GIB,
                available_bytes: 96 * GIB,
            },
            H3AdmissionPolicy::default(),
        )
        .unwrap_err();
        assert!(matches!(error, H3AdmissionError::InvalidCheckpointFacts(_)));
    }

    #[test]
    fn preparation_fingerprint_marker_is_h3_only_and_payload_free() {
        let h3 = request(minimax_h3::FL2VA_COMFY, 124);
        let authority = preparation_authority_bytes(&h3, Some("minimax-h3")).unwrap();
        let text = String::from_utf8(authority).unwrap();
        assert!(text.contains("product_attention_requires_qualified_cuda"));
        assert!(!text.contains(&h3.prompt));

        let mut ordinary = h3;
        ordinary.model = "flux-dev".to_string();
        assert!(preparation_authority_bytes(&ordinary, Some("flux")).is_none());
    }
}
