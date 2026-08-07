use candle::{DType, Device};
use candle_nn::VarBuilder;
use memmap2::MmapOptions;
use safetensors::{Dtype as SafeDtype, SafeTensors};
use serde::Deserialize;
use std::collections::{BTreeMap, BTreeSet};
use std::fs;
use std::fs::File;
use std::path::PathBuf;
use std::sync::Arc;
use thiserror::Error;
use tokenizers::Tokenizer;

use super::artifacts::{
    canonical_checkpoint_name, expected_checkpoint_shapes, validate_checkpoint_keys_and_layout,
    ArtifactError, ArtifactRole, CheckpointKeyReport, ConditionerArtifacts,
    ConditionerWeightLayout,
};
use super::config::{
    ConfigError, H3ConditionerConfig, H3_BF16_PARAMETER_BYTES, H3_FULL_CHECKPOINT_BYTES,
};
use super::model::H3Layer50Conditioner;
use super::presentation::{
    H3_IMAGE_PAD_TOKEN_ID, H3_VIDEO_PAD_TOKEN_ID, H3_VISION_END_TOKEN_ID, H3_VISION_START_TOKEN_ID,
};

#[derive(Debug, Error)]
pub enum H3LoadError {
    #[error(transparent)]
    Artifact(#[from] ArtifactError),
    #[error(transparent)]
    Config(#[from] ConfigError),
    #[error("failed to read MiniMax H3 asset {path}: {source}")]
    Io {
        path: PathBuf,
        #[source]
        source: std::io::Error,
    },
    #[error("malformed MiniMax H3 processor assets: {0}")]
    Processor(String),
    #[error("wrong MiniMax H3 tokenizer assets: {0}")]
    Tokenizer(String),
    #[error("invalid MiniMax H3 checkpoint: {0}")]
    Checkpoint(String),
    #[error("failed to construct MiniMax H3 Qwen3-VL tensors: {0}")]
    Candle(#[from] candle::Error),
}

pub struct LoadedH3Conditioner {
    pub model: H3Layer50Conditioner,
    pub tokenizer: Arc<Tokenizer>,
    pub key_report: CheckpointKeyReport,
}

#[derive(Clone)]
pub struct PreparedH3ConditionerAssets {
    artifacts: ConditionerArtifacts,
    config: H3ConditionerConfig,
    tokenizer: Arc<Tokenizer>,
    key_report: CheckpointKeyReport,
    checkpoint_file_bytes: u64,
    weight_layout: ConditionerWeightLayout,
}

impl PreparedH3ConditionerAssets {
    pub fn tokenizer(&self) -> &Arc<Tokenizer> {
        &self.tokenizer
    }

    pub fn key_report(&self) -> &CheckpointKeyReport {
        &self.key_report
    }

    pub fn checkpoint_file_bytes(&self) -> u64 {
        self.checkpoint_file_bytes
    }
}

#[derive(Debug, Deserialize)]
struct ProcessorConfig {
    size: ProcessorSize,
    patch_size: usize,
    temporal_patch_size: usize,
    merge_size: usize,
    image_mean: [f64; 3],
    image_std: [f64; 3],
    processor_class: String,
    #[serde(default)]
    image_processor_type: Option<String>,
    #[serde(default)]
    video_processor_type: Option<String>,
}

#[derive(Debug, Deserialize)]
struct ProcessorSize {
    longest_edge: usize,
    shortest_edge: usize,
}

pub fn validate_processor_assets(image: &[u8], video: &[u8]) -> Result<(), H3LoadError> {
    let image: ProcessorConfig = serde_json::from_slice(image)
        .map_err(|error| H3LoadError::Processor(format!("image config JSON: {error}")))?;
    let video: ProcessorConfig = serde_json::from_slice(video)
        .map_err(|error| H3LoadError::Processor(format!("video config JSON: {error}")))?;
    validate_processor(
        &image,
        (65_536, 16_777_216),
        Some("Qwen2VLImageProcessorFast"),
        None,
        "image",
    )?;
    validate_processor(
        &video,
        (4_096, 25_165_824),
        None,
        Some("Qwen3VLVideoProcessor"),
        "video",
    )
}

fn validate_processor(
    config: &ProcessorConfig,
    edges: (usize, usize),
    image_type: Option<&str>,
    video_type: Option<&str>,
    label: &str,
) -> Result<(), H3LoadError> {
    let exact = config.patch_size == 16
        && config.temporal_patch_size == 2
        && config.merge_size == 2
        && config.image_mean == [0.5; 3]
        && config.image_std == [0.5; 3]
        && config.processor_class == "Qwen3VLProcessor"
        && config.size.shortest_edge == edges.0
        && config.size.longest_edge == edges.1
        && config.image_processor_type.as_deref() == image_type
        && config.video_processor_type.as_deref() == video_type;
    if !exact {
        return Err(H3LoadError::Processor(format!(
            "{label} config does not match the released patch-16/temporal-2/merge-2 Qwen3-VL processor: {config:?}"
        )));
    }
    Ok(())
}

/// Verify all frozen identities and checkpoint headers before mmaping tensors.
/// The function is unsafe for the same reason as Candle's safetensors mmap
/// loader: callers must not mutate or truncate checkpoint files while the
/// returned model is alive.
///
/// # Safety
///
/// Every checkpoint path must remain immutable and valid for the complete
/// lifetime of the returned conditioner.
pub unsafe fn load_bf16_conditioner(
    artifacts: &ConditionerArtifacts,
    device: &Device,
) -> Result<LoadedH3Conditioner, H3LoadError> {
    let prepared = prepare_conditioner_assets(artifacts)?;
    // SAFETY: forwarded from this function's caller.
    let model = unsafe { load_prepared_bf16_conditioner(&prepared, device)? };
    Ok(LoadedH3Conditioner {
        model,
        tokenizer: Arc::clone(&prepared.tokenizer),
        key_report: prepared.key_report.clone(),
    })
}

/// Validate and retain the lightweight Qwen tokenizer and processor state,
/// plus the frozen checkpoint contract, without allocating model tensors.
pub fn prepare_conditioner_assets(
    artifacts: &ConditionerArtifacts,
) -> Result<PreparedH3ConditionerAssets, H3LoadError> {
    artifacts.verify_all()?;
    let config_bytes = read_role(artifacts, ArtifactRole::ArchitectureConfig)?;
    let config = H3ConditionerConfig::from_json(&config_bytes)?;
    let tokenizer_config = read_role(artifacts, ArtifactRole::TokenizerConfig)?;
    validate_tokenizer_config(&tokenizer_config)?;
    validate_processor_assets(
        &read_role(artifacts, ArtifactRole::ProcessorConfig)?,
        &read_role(artifacts, ArtifactRole::VideoProcessorConfig)?,
    )?;

    let tokenizer_artifact = artifacts
        .get(&ArtifactRole::Tokenizer)
        .ok_or(ArtifactError::MissingRole(ArtifactRole::Tokenizer))?;
    let tokenizer = Tokenizer::from_file(&tokenizer_artifact.path)
        .map(Arc::new)
        .map_err(|error| H3LoadError::Tokenizer(error.to_string()))?;
    validate_tokenizer(&tokenizer)?;

    let checkpoint_paths = artifacts
        .checkpoint_shards()
        .map(|artifact| artifact.path.clone())
        .collect::<Vec<_>>();
    let (header_names, payload_bytes, key_report, weight_layout) =
        validate_safetensors_headers(&checkpoint_paths, &config)?;
    if !matches!(
        payload_bytes,
        H3_BF16_PARAMETER_BYTES | H3_FULL_CHECKPOINT_BYTES
    ) {
        return Err(H3LoadError::Checkpoint(format!(
            "BF16 tensor payload is {payload_bytes} bytes, expected truncated {H3_BF16_PARAMETER_BYTES} or full {H3_FULL_CHECKPOINT_BYTES}"
        )));
    }
    if let Some(index) = artifacts.get(&ArtifactRole::CheckpointIndex) {
        validate_index(
            &fs::read(&index.path).map_err(|source| H3LoadError::Io {
                path: index.path.clone(),
                source,
            })?,
            &header_names,
            weight_layout,
        )?;
    }
    let checkpoint_file_bytes =
        artifacts
            .checkpoint_shards()
            .try_fold(0_u64, |total, artifact| {
                total
                    .checked_add(artifact.fingerprint.size_bytes)
                    .ok_or_else(|| {
                        H3LoadError::Checkpoint("checkpoint file byte count overflow".into())
                    })
            })?;
    Ok(PreparedH3ConditionerAssets {
        artifacts: artifacts.clone(),
        config,
        tokenizer,
        key_report,
        checkpoint_file_bytes,
        weight_layout,
    })
}

/// Rebuild only dropped tensor state while retaining prepared tokenizer and
/// processor assets.
///
/// # Safety
///
/// Every checkpoint path must remain immutable and valid for the complete
/// lifetime of the returned conditioner.
pub unsafe fn load_prepared_bf16_conditioner(
    prepared: &PreparedH3ConditionerAssets,
    device: &Device,
) -> Result<H3Layer50Conditioner, H3LoadError> {
    prepared.artifacts.verify_all()?;
    let checkpoint_paths = prepared
        .artifacts
        .checkpoint_shards()
        .map(|artifact| artifact.path.clone())
        .collect::<Vec<_>>();
    let references = checkpoint_paths.iter().collect::<Vec<_>>();
    // SAFETY: upheld by this function's caller; identities were rechecked
    // immediately above, before constructing the mmap.
    let builder = unsafe { VarBuilder::from_mmaped_safetensors(&references, DType::BF16, device)? };
    H3Layer50Conditioner::new_with_weight_layout(&prepared.config, builder, prepared.weight_layout)
        .map_err(Into::into)
}

fn read_role(artifacts: &ConditionerArtifacts, role: ArtifactRole) -> Result<Vec<u8>, H3LoadError> {
    let artifact = artifacts
        .get(&role)
        .ok_or_else(|| ArtifactError::MissingRole(role.clone()))?;
    fs::read(&artifact.path).map_err(|source| H3LoadError::Io {
        path: artifact.path.clone(),
        source,
    })
}

fn validate_tokenizer_config(bytes: &[u8]) -> Result<(), H3LoadError> {
    let value: serde_json::Value = serde_json::from_slice(bytes)
        .map_err(|error| H3LoadError::Tokenizer(format!("tokenizer config JSON: {error}")))?;
    let exact = value
        .get("tokenizer_class")
        .and_then(|value| value.as_str())
        == Some("Qwen2Tokenizer")
        && value.get("add_bos_token").and_then(|value| value.as_bool()) == Some(false)
        && value
            .get("bos_token")
            .is_some_and(serde_json::Value::is_null)
        && value
            .get("model_max_length")
            .and_then(|value| value.as_u64())
            == Some(262_144)
        && value.get("pad_token").and_then(|value| value.as_str()) == Some("<|endoftext|>")
        && value.get("eos_token").and_then(|value| value.as_str()) == Some("<|im_end|>");
    if !exact {
        return Err(H3LoadError::Tokenizer(
            "config is not the released raw Qwen2 tokenizer contract".into(),
        ));
    }
    Ok(())
}

fn validate_tokenizer(tokenizer: &Tokenizer) -> Result<(), H3LoadError> {
    if tokenizer.get_vocab_size(true) != 151_936 {
        return Err(H3LoadError::Tokenizer(format!(
            "vocabulary has {} entries, expected 151936",
            tokenizer.get_vocab_size(true)
        )));
    }
    for (token, expected) in [
        ("<|vision_start|>", H3_VISION_START_TOKEN_ID),
        ("<|vision_end|>", H3_VISION_END_TOKEN_ID),
        ("<|image_pad|>", H3_IMAGE_PAD_TOKEN_ID),
        ("<|video_pad|>", H3_VIDEO_PAD_TOKEN_ID),
    ] {
        let found = tokenizer.token_to_id(token);
        if found != Some(expected) {
            return Err(H3LoadError::Tokenizer(format!(
                "{token} resolves to {found:?}, expected {expected}"
            )));
        }
    }
    Ok(())
}

fn validate_safetensors_headers(
    paths: &[PathBuf],
    config: &H3ConditionerConfig,
) -> Result<
    (
        BTreeSet<String>,
        u64,
        CheckpointKeyReport,
        ConditionerWeightLayout,
    ),
    H3LoadError,
> {
    let expected_shapes = expected_checkpoint_shapes(config);
    let mut names = BTreeSet::new();
    let mut shapes = Vec::new();
    let mut payload_bytes = 0_u64;
    for path in paths {
        let file = File::open(path).map_err(|source| H3LoadError::Io {
            path: path.clone(),
            source,
        })?;
        // SAFETY: this temporary read-only map is dropped before returning;
        // artifact fingerprints were checked by the caller.
        let map = unsafe { MmapOptions::new().map(&file) }.map_err(|source| H3LoadError::Io {
            path: path.clone(),
            source,
        })?;
        let tensors = SafeTensors::deserialize(&map).map_err(|error| {
            H3LoadError::Checkpoint(format!(
                "{} has an invalid safetensors header: {error}",
                path.display()
            ))
        })?;
        for name in tensors.names() {
            if !names.insert(name.to_string()) {
                return Err(H3LoadError::Checkpoint(format!(
                    "tensor {name} occurs in more than one shard"
                )));
            }
            let tensor = tensors.tensor(name).map_err(|error| {
                H3LoadError::Checkpoint(format!("cannot inspect tensor {name}: {error}"))
            })?;
            if tensor.dtype() != SafeDtype::BF16 {
                return Err(H3LoadError::Checkpoint(format!(
                    "tensor {name} has {:?}, expected BF16",
                    tensor.dtype()
                )));
            }
            shapes.push((name.to_string(), tensor.shape().to_vec()));
            payload_bytes = payload_bytes
                .checked_add(tensor.data().len() as u64)
                .ok_or_else(|| H3LoadError::Checkpoint("tensor byte count overflow".into()))?;
        }
    }
    let (key_report, weight_layout) = validate_checkpoint_keys_and_layout(names.iter().cloned())?;
    for (raw_name, shape) in shapes {
        let canonical_name = canonical_checkpoint_name(&raw_name, weight_layout);
        if let Some(expected) = expected_shapes.get(&canonical_name) {
            if &shape != expected {
                return Err(H3LoadError::Checkpoint(format!(
                    "tensor {raw_name} has shape {shape:?}, expected {expected:?}"
                )));
            }
        } else if !canonical_name.ends_with("rotary_emb.inv_freq") {
            return Err(H3LoadError::Checkpoint(format!(
                "tensor {raw_name} has no H3 Qwen3-VL shape contract"
            )));
        }
    }
    Ok((names, payload_bytes, key_report, weight_layout))
}

#[derive(Deserialize)]
struct CheckpointIndex {
    metadata: CheckpointIndexMetadata,
    weight_map: BTreeMap<String, String>,
}

#[derive(Deserialize)]
struct CheckpointIndexMetadata {
    total_size: u64,
}

fn validate_index(
    bytes: &[u8],
    header_names: &BTreeSet<String>,
    header_layout: ConditionerWeightLayout,
) -> Result<(), H3LoadError> {
    let index: CheckpointIndex = serde_json::from_slice(bytes)
        .map_err(|error| H3LoadError::Checkpoint(format!("checkpoint index JSON: {error}")))?;
    let index_names = index.weight_map.keys().cloned().collect::<BTreeSet<_>>();
    let (_, index_layout) = validate_checkpoint_keys_and_layout(index_names.iter().cloned())?;
    if index_layout != header_layout {
        return Err(H3LoadError::Checkpoint(
            "checkpoint index and shard headers use different weight namespaces".into(),
        ));
    }
    if &index_names != header_names {
        return Err(H3LoadError::Checkpoint(format!(
            "checkpoint index names do not match shard headers ({} versus {})",
            index_names.len(),
            header_names.len()
        )));
    }
    if !matches!(
        index.metadata.total_size,
        H3_BF16_PARAMETER_BYTES | H3_FULL_CHECKPOINT_BYTES
    ) {
        return Err(H3LoadError::Checkpoint(format!(
            "checkpoint index total_size is {}, expected truncated {} or full {}",
            index.metadata.total_size, H3_BF16_PARAMETER_BYTES, H3_FULL_CHECKPOINT_BYTES
        )));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    const IMAGE: &str = r#"{
      "size":{"shortest_edge":65536,"longest_edge":16777216},
      "patch_size":16,"temporal_patch_size":2,"merge_size":2,
      "image_mean":[0.5,0.5,0.5],"image_std":[0.5,0.5,0.5],
      "processor_class":"Qwen3VLProcessor",
      "image_processor_type":"Qwen2VLImageProcessorFast"
    }"#;
    const VIDEO: &str = r#"{
      "size":{"shortest_edge":4096,"longest_edge":25165824},
      "patch_size":16,"temporal_patch_size":2,"merge_size":2,
      "image_mean":[0.5,0.5,0.5],"image_std":[0.5,0.5,0.5],
      "processor_class":"Qwen3VLProcessor",
      "video_processor_type":"Qwen3VLVideoProcessor"
    }"#;

    #[test]
    fn exact_processor_assets_validate() {
        validate_processor_assets(IMAGE.as_bytes(), VIDEO.as_bytes()).unwrap();
    }

    #[test]
    fn malformed_or_wrong_processor_assets_fail_closed() {
        let malformed = validate_processor_assets(b"{", VIDEO.as_bytes()).unwrap_err();
        assert!(malformed.to_string().contains("image config JSON"));
        let wrong = IMAGE.replace("\"patch_size\":16", "\"patch_size\":14");
        let error = validate_processor_assets(wrong.as_bytes(), VIDEO.as_bytes()).unwrap_err();
        assert!(error.to_string().contains("patch-16"));
    }
}
