use candle::{DType, Device};
use candle_nn::VarBuilder;
use serde::Deserialize;
use std::collections::{BTreeMap, BTreeSet};
use std::fs;
use std::fs::File;
use std::io::Read;
use std::path::PathBuf;
use std::sync::Arc;
use thiserror::Error;
use tokenizers::Tokenizer;

use super::artifacts::{
    canonical_checkpoint_name, expected_checkpoint_shapes, validate_checkpoint_keys_and_layout,
    ArtifactError, ArtifactRole, ArtifactVerificationProgress, CheckpointKeyReport,
    ConditionerArtifacts, ConditionerWeightLayout, VerifiedArtifactMetadata,
};
use super::config::{
    ConfigError, H3ConditionerConfig, H3_BF16_PARAMETER_BYTES, H3_FULL_CHECKPOINT_BYTES,
};
use super::model::H3Layer50Conditioner;
#[cfg(feature = "h3-private-uat")]
use super::model::H3Layer50StreamingConditioner;
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

#[cfg(feature = "h3-private-uat")]
pub struct LoadedH3StreamingConditioner {
    pub model: H3Layer50StreamingConditioner,
    pub tokenizer: Arc<Tokenizer>,
    pub key_report: CheckpointKeyReport,
}

#[derive(Clone)]
pub struct PreparedH3ConditionerAssets {
    artifacts: ConditionerArtifacts,
    verified_metadata: VerifiedArtifactMetadata,
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

/// Load the private qualification conditioner with only one language layer
/// resident on the execution device at a time.
///
/// # Safety
///
/// Every checkpoint path must remain bound to the same authenticated file for
/// the returned conditioner's complete lifetime because each layer is mmaped
/// again immediately before use. The private adapter supplies retained
/// descriptor paths and holds those descriptors through final revalidation.
#[cfg(feature = "h3-private-uat")]
pub unsafe fn load_streamed_bf16_conditioner(
    artifacts: &ConditionerArtifacts,
    device: &Device,
) -> Result<LoadedH3StreamingConditioner, H3LoadError> {
    let prepared = prepare_conditioner_assets(artifacts)?;
    prepared
        .artifacts
        .verify_metadata(&prepared.verified_metadata)?;
    let checkpoint_paths = prepared
        .artifacts
        .checkpoint_shards()
        .map(|artifact| artifact.path().to_path_buf())
        .collect::<Vec<_>>();
    let references = checkpoint_paths.iter().collect::<Vec<_>>();
    // SAFETY: forwarded from the caller after the identities were rechecked.
    let builder = unsafe { VarBuilder::from_mmaped_safetensors(&references, DType::BF16, device)? };
    let model = H3Layer50StreamingConditioner::new(&prepared.config, builder, checkpoint_paths)?;
    Ok(LoadedH3StreamingConditioner {
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
    prepare_conditioner_assets_with_progress(artifacts, &mut |_| Ok(()))
}

pub fn prepare_conditioner_assets_with_progress(
    artifacts: &ConditionerArtifacts,
    progress: &mut dyn FnMut(ArtifactVerificationProgress) -> Result<(), ArtifactError>,
) -> Result<PreparedH3ConditionerAssets, H3LoadError> {
    let verified_metadata = artifacts.verify_all_with_progress(progress)?;
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
    let tokenizer = Tokenizer::from_file(tokenizer_artifact.path())
        .map(Arc::new)
        .map_err(|error| H3LoadError::Tokenizer(error.to_string()))?;
    validate_tokenizer(&tokenizer)?;

    let checkpoint_paths = artifacts
        .checkpoint_shards()
        .map(|artifact| artifact.path().to_path_buf())
        .collect::<Vec<_>>();
    let (header_names, payload_bytes, key_report, weight_layout) =
        validate_safetensors_headers(&checkpoint_paths, &config)?;
    validate_layout_payload_bytes(weight_layout, payload_bytes, "BF16 tensor payload")?;
    if let Some(index) = artifacts.get(&ArtifactRole::CheckpointIndex) {
        validate_index(
            &fs::read(index.path()).map_err(|source| H3LoadError::Io {
                path: index.path().to_path_buf(),
                source,
            })?,
            &header_names,
            weight_layout,
        )?;
    }
    // Authentication precedes parsing, but a path can still be replaced while
    // the small support files and bounded checkpoint headers are read. Close
    // that window before the prepared authority is returned.
    artifacts.verify_metadata(&verified_metadata)?;
    let checkpoint_file_bytes =
        artifacts
            .checkpoint_shards()
            .try_fold(0_u64, |total, artifact| {
                total
                    .checked_add(artifact.fingerprint().size_bytes)
                    .ok_or_else(|| {
                        H3LoadError::Checkpoint("checkpoint file byte count overflow".into())
                    })
            })?;
    Ok(PreparedH3ConditionerAssets {
        artifacts: artifacts.clone(),
        verified_metadata,
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
    // Content was authenticated once during preparation. Re-reading 51.5-66.7
    // GB here would make every warm reload another full-volume hash pass. The
    // cheap file identity check catches replacement/size/timestamp changes;
    // the unsafe caller still owns the stronger immutable-file lifetime that
    // Candle's mmap requires.
    prepared
        .artifacts
        .verify_metadata(&prepared.verified_metadata)?;
    let checkpoint_paths = prepared
        .artifacts
        .checkpoint_shards()
        .map(|artifact| artifact.path().to_path_buf())
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
    fs::read(artifact.path()).map_err(|source| H3LoadError::Io {
        path: artifact.path().to_path_buf(),
        source,
    })
}

const fn expected_payload_bytes(layout: ConditionerWeightLayout) -> u64 {
    match layout {
        ConditionerWeightLayout::Official => H3_FULL_CHECKPOINT_BYTES,
        ConditionerWeightLayout::ComfyLayer50 => H3_BF16_PARAMETER_BYTES,
    }
}

fn validate_layout_payload_bytes(
    layout: ConditionerWeightLayout,
    found: u64,
    label: &str,
) -> Result<(), H3LoadError> {
    let expected = expected_payload_bytes(layout);
    if found != expected {
        return Err(H3LoadError::Checkpoint(format!(
            "{layout:?} {label} is {found} bytes, expected exactly {expected}"
        )));
    }
    Ok(())
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
        let inspected = inspect_safetensors_header(path)?;
        payload_bytes = payload_bytes
            .checked_add(inspected.payload_bytes)
            .ok_or_else(|| H3LoadError::Checkpoint("tensor byte count overflow".into()))?;
        for tensor in inspected.tensors {
            if !names.insert(tensor.name.clone()) {
                return Err(H3LoadError::Checkpoint(format!(
                    "tensor {} occurs in more than one shard",
                    tensor.name
                )));
            }
            shapes.push((tensor.name, tensor.shape));
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

const MAX_SAFETENSORS_HEADER_BYTES: usize = 100_000_000;

#[derive(Debug)]
struct InspectedSafetensors {
    tensors: Vec<InspectedTensor>,
    payload_bytes: u64,
}

#[derive(Debug)]
struct InspectedTensor {
    name: String,
    shape: Vec<usize>,
}

/// Read only the bounded safetensors header. Using a temporary mmap in this
/// safe preparation API would inherit memmap2's external-file-mutation UB even
/// though no model tensor is being loaded yet.
fn inspect_safetensors_header(path: &PathBuf) -> Result<InspectedSafetensors, H3LoadError> {
    let mut file = File::open(path).map_err(|source| H3LoadError::Io {
        path: path.clone(),
        source,
    })?;
    let file_len = file
        .metadata()
        .map_err(|source| H3LoadError::Io {
            path: path.clone(),
            source,
        })?
        .len();
    let mut header_len_bytes = [0_u8; 8];
    file.read_exact(&mut header_len_bytes)
        .map_err(|source| H3LoadError::Io {
            path: path.clone(),
            source,
        })?;
    let header_len = usize::try_from(u64::from_le_bytes(header_len_bytes)).map_err(|_| {
        H3LoadError::Checkpoint(format!(
            "{} safetensors header length overflows usize",
            path.display()
        ))
    })?;
    if header_len == 0 || header_len > MAX_SAFETENSORS_HEADER_BYTES {
        return Err(H3LoadError::Checkpoint(format!(
            "{} safetensors header length {header_len} is invalid",
            path.display()
        )));
    }
    let mut header = vec![0_u8; header_len];
    file.read_exact(&mut header)
        .map_err(|source| H3LoadError::Io {
            path: path.clone(),
            source,
        })?;
    let root: serde_json::Map<String, serde_json::Value> = serde_json::from_slice(&header)
        .map_err(|error| {
            H3LoadError::Checkpoint(format!(
                "{} has an invalid safetensors JSON header: {error}",
                path.display()
            ))
        })?;
    let mut tensors = Vec::with_capacity(root.len());
    let mut intervals = Vec::with_capacity(root.len());
    for (name, value) in root {
        if name == "__metadata__" {
            let valid = value
                .as_object()
                .is_some_and(|metadata| metadata.values().all(serde_json::Value::is_string));
            if !valid {
                return Err(H3LoadError::Checkpoint(
                    "safetensors __metadata__ must map strings to strings".into(),
                ));
            }
            continue;
        }
        let object = value.as_object().ok_or_else(|| {
            H3LoadError::Checkpoint(format!("tensor {name} header entry is not an object"))
        })?;
        let dtype = object
            .get("dtype")
            .and_then(serde_json::Value::as_str)
            .ok_or_else(|| H3LoadError::Checkpoint(format!("tensor {name} has no dtype")))?
            .to_string();
        if dtype != "BF16" {
            return Err(H3LoadError::Checkpoint(format!(
                "tensor {name} has {dtype}, expected BF16"
            )));
        }
        let shape = object
            .get("shape")
            .and_then(serde_json::Value::as_array)
            .ok_or_else(|| H3LoadError::Checkpoint(format!("tensor {name} has no shape")))?
            .iter()
            .map(|dimension| {
                dimension
                    .as_u64()
                    .and_then(|value| usize::try_from(value).ok())
                    .ok_or_else(|| {
                        H3LoadError::Checkpoint(format!(
                            "tensor {name} has a non-usize shape dimension"
                        ))
                    })
            })
            .collect::<Result<Vec<_>, _>>()?;
        let offsets = object
            .get("data_offsets")
            .and_then(serde_json::Value::as_array)
            .filter(|offsets| offsets.len() == 2)
            .ok_or_else(|| {
                H3LoadError::Checkpoint(format!("tensor {name} has invalid data_offsets"))
            })?;
        let start = offsets[0].as_u64().ok_or_else(|| {
            H3LoadError::Checkpoint(format!("tensor {name} has invalid start offset"))
        })?;
        let end = offsets[1].as_u64().ok_or_else(|| {
            H3LoadError::Checkpoint(format!("tensor {name} has invalid end offset"))
        })?;
        if end < start {
            return Err(H3LoadError::Checkpoint(format!(
                "tensor {name} has descending data offsets [{start}, {end}]"
            )));
        }
        let element_count = shape.iter().try_fold(1_u64, |count, dimension| {
            count.checked_mul(*dimension as u64).ok_or_else(|| {
                H3LoadError::Checkpoint(format!("tensor {name} element count overflows"))
            })
        })?;
        let expected_bytes = element_count.checked_mul(2).ok_or_else(|| {
            H3LoadError::Checkpoint(format!("tensor {name} byte count overflows"))
        })?;
        if end - start != expected_bytes {
            return Err(H3LoadError::Checkpoint(format!(
                "tensor {name} data span is {} bytes, expected {expected_bytes} for {dtype} {shape:?}",
                end - start
            )));
        }
        intervals.push((start, end, name.clone()));
        tensors.push(InspectedTensor { name, shape });
    }
    intervals.sort_unstable_by_key(|(start, _, _)| *start);
    let mut payload_bytes = 0_u64;
    for (start, end, name) in intervals {
        if start != payload_bytes {
            return Err(H3LoadError::Checkpoint(format!(
                "tensor {name} starts at {start}, expected contiguous offset {payload_bytes}"
            )));
        }
        payload_bytes = end;
    }
    let expected_file_len = 8_u64
        .checked_add(header_len as u64)
        .and_then(|bytes| bytes.checked_add(payload_bytes))
        .ok_or_else(|| H3LoadError::Checkpoint("safetensors file length overflows".into()))?;
    if file_len != expected_file_len {
        return Err(H3LoadError::Checkpoint(format!(
            "{} is {file_len} bytes, but its header accounts for {expected_file_len}",
            path.display()
        )));
    }
    Ok(InspectedSafetensors {
        tensors,
        payload_bytes,
    })
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
    validate_layout_payload_bytes(
        header_layout,
        index.metadata.total_size,
        "checkpoint index total_size",
    )?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_path(label: &str) -> PathBuf {
        std::env::temp_dir().join(format!(
            "mold-h3-loader-{label}-{}-{:?}",
            std::process::id(),
            std::thread::current().id()
        ))
    }

    fn safetensors_fixture(label: &str, header: serde_json::Value, payload: &[u8]) -> PathBuf {
        let path = test_path(label);
        let mut header = serde_json::to_vec(&header).unwrap();
        while !header.len().is_multiple_of(8) {
            header.push(b' ');
        }
        let mut file = (header.len() as u64).to_le_bytes().to_vec();
        file.extend_from_slice(&header);
        file.extend_from_slice(payload);
        fs::write(&path, file).unwrap();
        path
    }

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

    #[test]
    fn checkpoint_payload_size_is_bound_to_the_detected_layout() {
        assert!(validate_layout_payload_bytes(
            ConditionerWeightLayout::Official,
            H3_FULL_CHECKPOINT_BYTES,
            "payload"
        )
        .is_ok());
        assert!(validate_layout_payload_bytes(
            ConditionerWeightLayout::ComfyLayer50,
            H3_BF16_PARAMETER_BYTES,
            "payload"
        )
        .is_ok());

        let official_truncated = validate_layout_payload_bytes(
            ConditionerWeightLayout::Official,
            H3_BF16_PARAMETER_BYTES,
            "payload",
        )
        .unwrap_err();
        assert!(official_truncated.to_string().contains("Official payload"));
        let comfy_full = validate_layout_payload_bytes(
            ConditionerWeightLayout::ComfyLayer50,
            H3_FULL_CHECKPOINT_BYTES,
            "payload",
        )
        .unwrap_err();
        assert!(comfy_full.to_string().contains("ComfyLayer50 payload"));
    }

    #[test]
    fn bounded_header_reader_validates_dtype_offsets_and_file_length() {
        let valid_path = safetensors_fixture(
            "valid-header",
            serde_json::json!({
                "weight": {"dtype": "BF16", "shape": [2], "data_offsets": [0, 4]}
            }),
            &[0; 4],
        );
        let inspected = inspect_safetensors_header(&valid_path).unwrap();
        assert_eq!(inspected.payload_bytes, 4);
        assert_eq!(inspected.tensors[0].shape, [2]);
        let _ = fs::remove_file(valid_path);

        let wrong_dtype_path = safetensors_fixture(
            "wrong-dtype",
            serde_json::json!({
                "weight": {"dtype": "F16", "shape": [2], "data_offsets": [0, 4]}
            }),
            &[0; 4],
        );
        let error = inspect_safetensors_header(&wrong_dtype_path).unwrap_err();
        assert!(error.to_string().contains("F16, expected BF16"));
        let _ = fs::remove_file(wrong_dtype_path);

        let gap_path = safetensors_fixture(
            "offset-gap",
            serde_json::json!({
                "weight": {"dtype": "BF16", "shape": [2], "data_offsets": [2, 6]}
            }),
            &[0; 6],
        );
        let error = inspect_safetensors_header(&gap_path).unwrap_err();
        assert!(error.to_string().contains("expected contiguous offset 0"));
        let _ = fs::remove_file(gap_path);
    }
}
