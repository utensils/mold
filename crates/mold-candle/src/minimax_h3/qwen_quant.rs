//! Artifact-free Qwen3-VL INT8 ConvRot loader and execution policy for H3.
//!
//! The policy is derived from ComfyUI
//! `a464ac33588ae182f81a090d910cfbf21e255b73` and comfy-kitchen 0.2.26
//! (`255a43879fe57bbcbecfdb273b46d772b00c5a90`). It validates caller-supplied
//! header/metadata descriptions only. It does not open, discover, download, or
//! register an H3 checkpoint, and it is not runtime activation authority.

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, BTreeSet};
use thiserror::Error;

use super::artifacts::{expected_checkpoint_shapes, expected_materialized_keys};
use super::config::{
    H3ConditionerConfig, H3_COMFY_INT8_CONVROT_PARAMETER_BYTES,
    H3_COMFY_INT8_CONVROT_SAFETENSORS_BYTES, H3_SELECTED_LANGUAGE_LAYERS,
};
use super::{H3_COMFY_CONVROT_GROUP_SIZE, H3_COMFY_PORTABLE_ROW_CHUNK};

pub const H3_QWEN_INT8_WEIGHT_ONLY_STABLE_ID: &str =
    "minimax-h3.qwen3-vl.layer50.int8-convrot-weight-only.v1";
pub const H3_QWEN_INT8_QUANTIZED_LINEAR_COUNT: usize = 350;

const LANGUAGE_LINEAR_SUFFIXES: &[&str] = &[
    "self_attn.q_proj",
    "self_attn.k_proj",
    "self_attn.v_proj",
    "self_attn.o_proj",
    "mlp.gate_proj",
    "mlp.up_proj",
    "mlp.down_proj",
];

#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd, Serialize, Deserialize)]
pub enum H3QwenTensorDType {
    Bf16,
    F8E4M3,
    F32,
    I8,
    U8,
}

impl H3QwenTensorDType {
    pub(super) const fn byte_width(self) -> u64 {
        match self {
            Self::F8E4M3 | Self::I8 | Self::U8 => 1,
            Self::Bf16 => 2,
            Self::F32 => 4,
        }
    }

    pub(super) const fn stable_id(self) -> &'static str {
        match self {
            Self::Bf16 => "bf16",
            Self::F8E4M3 => "f8-e4m3",
            Self::F32 => "f32",
            Self::I8 => "i8",
            Self::U8 => "u8",
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct H3QwenTensorSpec {
    pub dtype: H3QwenTensorDType,
    pub shape: Vec<usize>,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct H3QwenInt8LayerMetadata {
    pub format: String,
    pub full_precision_matrix_mult: Option<bool>,
    pub convrot: Option<bool>,
    pub convrot_groupsize: Option<usize>,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct H3QwenInt8CheckpointMetadata {
    pub format_version: String,
    pub layers: BTreeMap<String, H3QwenInt8LayerMetadata>,
}

#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd, Serialize, Deserialize)]
pub enum H3QwenFp32Boundary {
    TextMultimodalRotary,
    TextAttentionScoresAndSoftmax,
    VisionRotary,
    VisionAttentionScoresAndSoftmax,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct H3QwenInt8MemoryAccounting {
    pub artifact_file_bytes: u64,
    pub header_bytes: u64,
    pub encoded_tensor_bytes: u64,
    pub quantized_weight_bytes: u64,
    pub quantization_scale_bytes: u64,
    pub protected_bf16_bytes: u64,
    /// Exact peak scratch for default-chunk weight reconstruction: the signed
    /// F32 source rows, scaled rows, reconstructed rows, the concurrently live
    /// BF16/F16 conversion, per-row F32 scales, and one 256x256 F32 Hadamard
    /// matrix. Caller-owned activation and linear output workspaces remain
    /// request-shaped and are not hidden here.
    pub default_dequantization_workspace_bytes: u64,
    /// A future mmap loader would map the complete object, but mapped bytes are
    /// not claimed as resident RAM.
    pub host_mapped_file_bytes: u64,
    pub host_resident_bytes: Option<u64>,
    pub resident_device_weight_bytes: Option<u64>,
    pub streamed_weight_bytes: Option<u64>,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub enum H3QwenLinearExecution {
    DenseBf16,
    Int8ConvRotWeightOnly {
        weight_scale_tensor: String,
        group_size: usize,
    },
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct H3QwenInt8WeightOnlyPolicy {
    stable_id: String,
    /// Canonical official-namespace layer names without `.weight`.
    quantized_layers: Vec<String>,
    /// Canonical tensors that must remain dense BF16.
    protected_tensors: Vec<String>,
    fp32_compute_boundaries: Vec<H3QwenFp32Boundary>,
    memory: H3QwenInt8MemoryAccounting,
    policy_sha256: String,
    quantized_linear_shapes: Vec<[usize; 2]>,
}

impl H3QwenInt8WeightOnlyPolicy {
    pub fn stable_id(&self) -> &str {
        &self.stable_id
    }

    pub fn quantized_layers(&self) -> &[String] {
        &self.quantized_layers
    }

    pub fn protected_tensors(&self) -> &[String] {
        &self.protected_tensors
    }

    pub fn fp32_compute_boundaries(&self) -> &[H3QwenFp32Boundary] {
        &self.fp32_compute_boundaries
    }

    pub const fn memory(&self) -> &H3QwenInt8MemoryAccounting {
        &self.memory
    }

    pub fn policy_sha256(&self) -> &str {
        &self.policy_sha256
    }

    pub fn execution_for_weight(
        &self,
        canonical_weight: &str,
    ) -> Result<H3QwenLinearExecution, H3QwenInt8PolicyError> {
        let Some(layer) = canonical_weight.strip_suffix(".weight") else {
            return Err(H3QwenInt8PolicyError::Execution(format!(
                "Qwen execution lookup requires a canonical .weight tensor, got {canonical_weight:?}"
            )));
        };
        if self
            .quantized_layers
            .binary_search_by(|candidate| candidate.as_str().cmp(layer))
            .is_ok()
        {
            return Ok(H3QwenLinearExecution::Int8ConvRotWeightOnly {
                weight_scale_tensor: format!("{layer}.weight_scale"),
                group_size: H3_COMFY_CONVROT_GROUP_SIZE,
            });
        }
        if self
            .protected_tensors
            .binary_search_by(|candidate| candidate.as_str().cmp(canonical_weight))
            .is_ok()
        {
            return Ok(H3QwenLinearExecution::DenseBf16);
        }
        Err(H3QwenInt8PolicyError::Execution(format!(
            "Qwen tensor {canonical_weight:?} is outside the frozen layer-50 policy"
        )))
    }

    /// Maximum peak reconstruction scratch for any one quantized Qwen linear
    /// at the requested output-row chunk. The portable ConvRot path holds the
    /// signed F32 rows, scaled rows, and reconstructed rows concurrently. A
    /// BF16/F16 compute path also allocates its converted reconstructed rows
    /// before those temporaries leave scope. Per-row scales and the shared F32
    /// Hadamard matrix are included; caller-owned activations and the final
    /// linear output are excluded.
    pub fn max_dequantization_workspace_bytes(
        &self,
        rows_per_chunk: usize,
    ) -> Result<u64, H3QwenInt8PolicyError> {
        if rows_per_chunk == 0 {
            return Err(H3QwenInt8PolicyError::Accounting(
                "Qwen INT8 row chunk must be positive".into(),
            ));
        }
        let mut peak_reconstruction = 0_u64;
        for [out_features, in_features] in &self.quantized_linear_shapes {
            let rows = rows_per_chunk.min(*out_features) as u64;
            let elements = rows.checked_mul(*in_features as u64).ok_or_else(|| {
                H3QwenInt8PolicyError::Accounting("Qwen INT8 weight workspace overflows".into())
            })?;
            let row_bytes = elements.checked_mul(4).ok_or_else(|| {
                H3QwenInt8PolicyError::Accounting("Qwen INT8 weight workspace overflows".into())
            })?;
            let converted_row_bytes = elements.checked_mul(2).ok_or_else(|| {
                H3QwenInt8PolicyError::Accounting("Qwen INT8 converted workspace overflows".into())
            })?;
            let scale_bytes = rows.checked_mul(4).ok_or_else(|| {
                H3QwenInt8PolicyError::Accounting("Qwen INT8 scale workspace overflows".into())
            })?;
            let bytes = row_bytes
                .checked_mul(3)
                .and_then(|bytes| bytes.checked_add(converted_row_bytes))
                .and_then(|bytes| bytes.checked_add(scale_bytes))
                .ok_or_else(|| {
                    H3QwenInt8PolicyError::Accounting(
                        "Qwen INT8 reconstruction workspace overflows".into(),
                    )
                })?;
            peak_reconstruction = peak_reconstruction.max(bytes);
        }
        let hadamard = (H3_COMFY_CONVROT_GROUP_SIZE as u64)
            .checked_mul(H3_COMFY_CONVROT_GROUP_SIZE as u64)
            .and_then(|elements| elements.checked_mul(4))
            .ok_or_else(|| {
                H3QwenInt8PolicyError::Accounting("Qwen INT8 Hadamard workspace overflows".into())
            })?;
        peak_reconstruction.checked_add(hadamard).ok_or_else(|| {
            H3QwenInt8PolicyError::Accounting("Qwen INT8 workspace overflows".into())
        })
    }
}

#[derive(Debug, Error, Eq, PartialEq)]
pub enum H3QwenInt8PolicyError {
    #[error("invalid MiniMax H3 Qwen config: {0}")]
    Config(String),
    #[error("invalid MiniMax H3 Qwen INT8 namespace: {0}")]
    Namespace(String),
    #[error("invalid MiniMax H3 Qwen INT8 metadata: {0}")]
    Metadata(String),
    #[error("invalid MiniMax H3 Qwen INT8 tensor schema: {0}")]
    TensorSchema(String),
    #[error("invalid MiniMax H3 Qwen INT8 execution lookup: {0}")]
    Execution(String),
    #[error("invalid MiniMax H3 Qwen INT8 accounting: {0}")]
    Accounting(String),
}

fn canonical_quantized_layers() -> BTreeSet<String> {
    (0..H3_SELECTED_LANGUAGE_LAYERS)
        .flat_map(|layer| {
            LANGUAGE_LINEAR_SUFFIXES
                .iter()
                .map(move |suffix| format!("model.language_model.layers.{layer}.{suffix}"))
        })
        .collect()
}

fn comfy_name(canonical: &str) -> String {
    canonical
        .strip_prefix("model.language_model.")
        .map(|suffix| format!("model.{suffix}"))
        .or_else(|| {
            canonical
                .strip_prefix("model.visual.")
                .map(|suffix| format!("visual.{suffix}"))
        })
        .unwrap_or_else(|| canonical.to_owned())
}

fn checked_tensor_bytes(spec: &H3QwenTensorSpec) -> Result<u64, H3QwenInt8PolicyError> {
    spec.shape
        .iter()
        .try_fold(1_u64, |elements, dimension| {
            elements.checked_mul(*dimension as u64)
        })
        .and_then(|elements| elements.checked_mul(spec.dtype.byte_width()))
        .ok_or_else(|| H3QwenInt8PolicyError::Accounting("Qwen tensor byte count overflows".into()))
}

/// Build the exact expected Comfy layer-50 schema without reading a checkpoint.
/// The returned names use the published `model.*`/`visual.*` namespace.
pub fn expected_h3_qwen_int8_weight_only_schema(
    config: &H3ConditionerConfig,
) -> Result<
    (
        BTreeMap<String, H3QwenTensorSpec>,
        H3QwenInt8CheckpointMetadata,
    ),
    H3QwenInt8PolicyError,
> {
    config
        .validate()
        .map_err(|error| H3QwenInt8PolicyError::Config(error.to_string()))?;
    let shapes = expected_checkpoint_shapes(config);
    let quantized = canonical_quantized_layers();
    let mut tensors = BTreeMap::new();
    let mut layers = BTreeMap::new();
    for canonical in expected_materialized_keys() {
        let shape = shapes.get(&canonical).cloned().ok_or_else(|| {
            H3QwenInt8PolicyError::TensorSchema(format!(
                "missing production shape for {canonical:?}"
            ))
        })?;
        let layer = canonical.strip_suffix(".weight");
        let is_quantized = layer.is_some_and(|layer| quantized.contains(layer));
        let raw = comfy_name(&canonical);
        tensors.insert(
            raw.clone(),
            H3QwenTensorSpec {
                dtype: if is_quantized {
                    H3QwenTensorDType::I8
                } else {
                    H3QwenTensorDType::Bf16
                },
                shape: shape.clone(),
            },
        );
        if is_quantized {
            let layer = layer.expect("is_quantized requires a weight tensor");
            let raw_layer = comfy_name(layer);
            tensors.insert(
                format!("{raw_layer}.weight_scale"),
                H3QwenTensorSpec {
                    dtype: H3QwenTensorDType::F32,
                    shape: vec![shape[0], 1],
                },
            );
            layers.insert(
                raw_layer,
                H3QwenInt8LayerMetadata {
                    format: "int8_tensorwise".into(),
                    full_precision_matrix_mult: None,
                    convrot: Some(true),
                    convrot_groupsize: Some(H3_COMFY_CONVROT_GROUP_SIZE),
                },
            );
        }
    }
    Ok((
        tensors,
        H3QwenInt8CheckpointMetadata {
            format_version: "1.0".into(),
            layers,
        },
    ))
}

/// Validate a caller-supplied, header-derived schema and freeze the only
/// supported Qwen INT8 execution policy. Successful validation still grants no
/// filesystem, license, download, or runtime authority.
pub fn validate_h3_qwen_int8_weight_only_schema(
    config: &H3ConditionerConfig,
    tensors: &BTreeMap<String, H3QwenTensorSpec>,
    metadata: &H3QwenInt8CheckpointMetadata,
) -> Result<H3QwenInt8WeightOnlyPolicy, H3QwenInt8PolicyError> {
    let (expected_tensors, expected_metadata) = expected_h3_qwen_int8_weight_only_schema(config)?;
    if tensors.contains_key("model.language_model.embed_tokens.weight")
        || tensors.keys().any(|name| name.starts_with("model.visual."))
        || !tensors.contains_key("model.embed_tokens.weight")
        || !tensors.keys().any(|name| name.starts_with("visual."))
    {
        return Err(H3QwenInt8PolicyError::Namespace(
            "INT8 layer-50 checkpoints must use exactly the Comfy model.* and visual.* namespaces"
                .into(),
        ));
    }
    let expected_keys = expected_tensors.keys().collect::<BTreeSet<_>>();
    let actual_keys = tensors.keys().collect::<BTreeSet<_>>();
    if expected_keys != actual_keys {
        let missing = expected_keys
            .difference(&actual_keys)
            .take(4)
            .map(|name| (*name).clone())
            .collect::<Vec<_>>();
        let unexpected = actual_keys
            .difference(&expected_keys)
            .take(4)
            .map(|name| (*name).clone())
            .collect::<Vec<_>>();
        return Err(H3QwenInt8PolicyError::TensorSchema(format!(
            "tensor set mismatch: missing={missing:?} unexpected={unexpected:?}"
        )));
    }
    for (name, expected) in &expected_tensors {
        let actual = &tensors[name];
        if actual != expected {
            return Err(H3QwenInt8PolicyError::TensorSchema(format!(
                "tensor {name:?} expected {:?} {:?}, got {:?} {:?}",
                expected.dtype, expected.shape, actual.dtype, actual.shape
            )));
        }
    }
    if metadata.format_version != "1.0" {
        return Err(H3QwenInt8PolicyError::Metadata(format!(
            "format_version must be 1.0, got {:?}",
            metadata.format_version
        )));
    }
    let expected_layers = expected_metadata.layers.keys().collect::<BTreeSet<_>>();
    let actual_layers = metadata.layers.keys().collect::<BTreeSet<_>>();
    if expected_layers != actual_layers {
        return Err(H3QwenInt8PolicyError::Metadata(format!(
            "quantized layer set mismatch: expected {} layers, got {}",
            expected_layers.len(),
            actual_layers.len()
        )));
    }
    for (name, layer) in &metadata.layers {
        if layer.format != "int8_tensorwise"
            || layer.full_precision_matrix_mult == Some(true)
            || layer.convrot != Some(true)
            || layer.convrot_groupsize != Some(H3_COMFY_CONVROT_GROUP_SIZE)
        {
            return Err(H3QwenInt8PolicyError::Metadata(format!(
                "layer {name:?} must be INT8 tensorwise ConvRot group {} with weight-only activation policy",
                H3_COMFY_CONVROT_GROUP_SIZE
            )));
        }
    }

    let quantized_layers = canonical_quantized_layers().into_iter().collect::<Vec<_>>();
    if quantized_layers.len() != H3_QWEN_INT8_QUANTIZED_LINEAR_COUNT {
        return Err(H3QwenInt8PolicyError::Accounting(format!(
            "expected {H3_QWEN_INT8_QUANTIZED_LINEAR_COUNT} Qwen linears, derived {}",
            quantized_layers.len()
        )));
    }
    let materialized = expected_materialized_keys();
    let protected_tensors = materialized
        .iter()
        .filter(|tensor| {
            !tensor.strip_suffix(".weight").is_some_and(|layer| {
                quantized_layers
                    .binary_search_by(|candidate| candidate.as_str().cmp(layer))
                    .is_ok()
            })
        })
        .cloned()
        .collect::<Vec<_>>();
    let fp32_compute_boundaries = vec![
        H3QwenFp32Boundary::TextMultimodalRotary,
        H3QwenFp32Boundary::TextAttentionScoresAndSoftmax,
        H3QwenFp32Boundary::VisionRotary,
        H3QwenFp32Boundary::VisionAttentionScoresAndSoftmax,
    ];

    let mut encoded_tensor_bytes = 0_u64;
    let mut quantized_weight_bytes = 0_u64;
    let mut quantization_scale_bytes = 0_u64;
    let mut quantized_linear_shapes = Vec::new();
    for (name, spec) in &expected_tensors {
        let bytes = checked_tensor_bytes(spec)?;
        encoded_tensor_bytes = encoded_tensor_bytes.checked_add(bytes).ok_or_else(|| {
            H3QwenInt8PolicyError::Accounting("Qwen encoded bytes overflow".into())
        })?;
        if spec.dtype == H3QwenTensorDType::I8 {
            quantized_weight_bytes =
                quantized_weight_bytes.checked_add(bytes).ok_or_else(|| {
                    H3QwenInt8PolicyError::Accounting("Qwen quantized bytes overflow".into())
                })?;
            quantized_linear_shapes.push([spec.shape[0], spec.shape[1]]);
        } else if name.ends_with(".weight_scale") {
            quantization_scale_bytes =
                quantization_scale_bytes.checked_add(bytes).ok_or_else(|| {
                    H3QwenInt8PolicyError::Accounting("Qwen scale bytes overflow".into())
                })?;
        }
    }
    if encoded_tensor_bytes != H3_COMFY_INT8_CONVROT_PARAMETER_BYTES {
        return Err(H3QwenInt8PolicyError::Accounting(format!(
            "derived {encoded_tensor_bytes} encoded bytes, expected {H3_COMFY_INT8_CONVROT_PARAMETER_BYTES}"
        )));
    }
    let protected_bf16_bytes = encoded_tensor_bytes
        .checked_sub(quantized_weight_bytes)
        .and_then(|bytes| bytes.checked_sub(quantization_scale_bytes))
        .ok_or_else(|| {
            H3QwenInt8PolicyError::Accounting("Qwen protected byte count underflows".into())
        })?;
    let header_bytes = H3_COMFY_INT8_CONVROT_SAFETENSORS_BYTES
        .checked_sub(encoded_tensor_bytes)
        .ok_or_else(|| {
            H3QwenInt8PolicyError::Accounting("Qwen artifact header bytes underflow".into())
        })?;

    let mut policy = H3QwenInt8WeightOnlyPolicy {
        stable_id: H3_QWEN_INT8_WEIGHT_ONLY_STABLE_ID.into(),
        quantized_layers,
        protected_tensors,
        fp32_compute_boundaries,
        memory: H3QwenInt8MemoryAccounting {
            artifact_file_bytes: H3_COMFY_INT8_CONVROT_SAFETENSORS_BYTES,
            header_bytes,
            encoded_tensor_bytes,
            quantized_weight_bytes,
            quantization_scale_bytes,
            protected_bf16_bytes,
            default_dequantization_workspace_bytes: 0,
            host_mapped_file_bytes: H3_COMFY_INT8_CONVROT_SAFETENSORS_BYTES,
            host_resident_bytes: None,
            resident_device_weight_bytes: None,
            streamed_weight_bytes: None,
        },
        policy_sha256: String::new(),
        quantized_linear_shapes,
    };
    policy.memory.default_dequantization_workspace_bytes =
        policy.max_dequantization_workspace_bytes(H3_COMFY_PORTABLE_ROW_CHUNK)?;

    let mut digest = Sha256::new();
    digest.update(policy.stable_id.as_bytes());
    for (name, spec) in &expected_tensors {
        digest.update((name.len() as u64).to_le_bytes());
        digest.update(name.as_bytes());
        digest.update(spec.dtype.stable_id().as_bytes());
        for dimension in &spec.shape {
            digest.update((*dimension as u64).to_le_bytes());
        }
    }
    for boundary in &policy.fp32_compute_boundaries {
        digest.update(format!("{boundary:?}").as_bytes());
    }
    policy.policy_sha256 = digest
        .finalize()
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect();
    Ok(policy)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn released_config() -> H3ConditionerConfig {
        H3ConditionerConfig::from_json(
            br#"{
              "architectures":["Qwen3VLForConditionalGeneration"],
              "image_token_id":151655,"video_token_id":151656,
              "vision_start_token_id":151652,"vision_end_token_id":151653,
              "text_config":{"model_type":"qwen3_vl_text","vocab_size":151936,
                "hidden_size":5120,"intermediate_size":25600,"num_hidden_layers":64,
                "num_attention_heads":64,"num_key_value_heads":8,"head_dim":128,
                "rms_norm_eps":0.000001,"rope_theta":5000000.0,
                "max_position_embeddings":262144,"hidden_act":"silu","attention_bias":false,
                "rope_scaling":{"rope_type":"default","mrope_interleaved":true,
                  "mrope_section":[24,20,20]}},
              "vision_config":{"model_type":"qwen3_vl","depth":27,"hidden_size":1152,
                "intermediate_size":4304,"num_heads":16,"in_channels":3,"patch_size":16,
                "temporal_patch_size":2,"spatial_merge_size":2,"out_hidden_size":5120,
                "num_position_embeddings":2304,"deepstack_visual_indexes":[8,16,24],
                "hidden_act":"gelu_pytorch_tanh"}
            }"#,
        )
        .unwrap()
    }

    fn valid_fixture() -> (
        H3ConditionerConfig,
        BTreeMap<String, H3QwenTensorSpec>,
        H3QwenInt8CheckpointMetadata,
    ) {
        let config = released_config();
        let (tensors, metadata) = expected_h3_qwen_int8_weight_only_schema(&config).unwrap();
        (config, tensors, metadata)
    }

    #[test]
    fn exact_layer_50_policy_and_memory_are_frozen_without_artifacts() {
        let (config, tensors, metadata) = valid_fixture();
        let policy =
            validate_h3_qwen_int8_weight_only_schema(&config, &tensors, &metadata).unwrap();
        assert_eq!(policy.stable_id(), H3_QWEN_INT8_WEIGHT_ONLY_STABLE_ID);
        assert_eq!(policy.quantized_layers().len(), 350);
        assert!(!policy.protected_tensors().is_empty());
        assert_eq!(policy.fp32_compute_boundaries().len(), 4);
        assert_eq!(metadata.layers.len(), 350);
        assert_eq!(policy.memory().encoded_tensor_bytes, 27_141_135_840);
        assert_eq!(policy.memory().artifact_file_bytes, 27_141_342_152);
        assert_eq!(policy.memory().header_bytes, 206_312);
        assert_eq!(policy.memory().quantized_weight_bytes, 24_379_392_000);
        assert_eq!(policy.memory().quantization_scale_bytes, 14_336_000);
        assert_eq!(policy.memory().protected_bf16_bytes, 2_747_407_840);
        assert_eq!(
            policy.memory().default_dequantization_workspace_bytes,
            92_013_568
        );
        assert_eq!(policy.policy_sha256().len(), 64);
        assert_eq!(
            policy
                .execution_for_weight("model.language_model.layers.49.mlp.down_proj.weight")
                .unwrap(),
            H3QwenLinearExecution::Int8ConvRotWeightOnly {
                weight_scale_tensor: "model.language_model.layers.49.mlp.down_proj.weight_scale"
                    .into(),
                group_size: 256,
            }
        );
        assert_eq!(
            policy
                .execution_for_weight("model.language_model.embed_tokens.weight")
                .unwrap(),
            H3QwenLinearExecution::DenseBf16
        );
        assert_eq!(
            policy
                .execution_for_weight("model.visual.blocks.0.attn.qkv.weight")
                .unwrap(),
            H3QwenLinearExecution::DenseBf16
        );
        assert!(policy
            .execution_for_weight("model.language_model.layers.50.mlp.down_proj.weight")
            .is_err());
    }

    #[test]
    fn only_language_projection_weights_are_int8() {
        let (config, tensors, metadata) = valid_fixture();
        validate_h3_qwen_int8_weight_only_schema(&config, &tensors, &metadata).unwrap();
        assert_eq!(
            tensors["model.layers.0.self_attn.q_proj.weight"].dtype,
            H3QwenTensorDType::I8
        );
        assert_eq!(
            tensors["model.layers.0.self_attn.q_proj.weight_scale"],
            H3QwenTensorSpec {
                dtype: H3QwenTensorDType::F32,
                shape: vec![8192, 1],
            }
        );
        for protected in [
            "model.embed_tokens.weight",
            "model.layers.0.input_layernorm.weight",
            "model.layers.0.self_attn.q_norm.weight",
            "visual.blocks.0.attn.qkv.weight",
            "visual.merger.linear_fc2.weight",
        ] {
            assert_eq!(tensors[protected].dtype, H3QwenTensorDType::Bf16);
        }
        assert!(!tensors.keys().any(|name| name.ends_with(".input_scale")));
    }

    #[test]
    fn dtype_shape_and_metadata_mutations_fail_closed() {
        let (config, tensors, metadata) = valid_fixture();
        let scale = "model.layers.0.self_attn.q_proj.weight_scale";

        let mut wrong_dtype = tensors.clone();
        wrong_dtype.get_mut(scale).unwrap().dtype = H3QwenTensorDType::Bf16;
        assert!(matches!(
            validate_h3_qwen_int8_weight_only_schema(&config, &wrong_dtype, &metadata),
            Err(H3QwenInt8PolicyError::TensorSchema(_))
        ));

        let mut wrong_shape = tensors.clone();
        wrong_shape.get_mut(scale).unwrap().shape = Vec::new();
        assert!(matches!(
            validate_h3_qwen_int8_weight_only_schema(&config, &wrong_shape, &metadata),
            Err(H3QwenInt8PolicyError::TensorSchema(_))
        ));

        let mut missing = tensors.clone();
        missing.remove(scale);
        assert!(matches!(
            validate_h3_qwen_int8_weight_only_schema(&config, &missing, &metadata),
            Err(H3QwenInt8PolicyError::TensorSchema(_))
        ));

        let mut dynamic_activation = metadata.clone();
        dynamic_activation
            .layers
            .get_mut("model.layers.0.self_attn.q_proj")
            .unwrap()
            .convrot = Some(false);
        assert!(matches!(
            validate_h3_qwen_int8_weight_only_schema(&config, &tensors, &dynamic_activation),
            Err(H3QwenInt8PolicyError::Metadata(_))
        ));

        let mut fused_activation = metadata.clone();
        fused_activation
            .layers
            .get_mut("model.layers.0.self_attn.q_proj")
            .unwrap()
            .full_precision_matrix_mult = Some(true);
        assert!(matches!(
            validate_h3_qwen_int8_weight_only_schema(&config, &tensors, &fused_activation),
            Err(H3QwenInt8PolicyError::Metadata(_))
        ));

        let mut explicit_weight_only = metadata.clone();
        explicit_weight_only
            .layers
            .values_mut()
            .for_each(|layer| layer.full_precision_matrix_mult = Some(false));
        validate_h3_qwen_int8_weight_only_schema(&config, &tensors, &explicit_weight_only).unwrap();
    }

    #[test]
    fn serialized_checkpoint_descriptions_reject_unknown_fields() {
        let (_, tensors, metadata) = valid_fixture();

        let mut checkpoint = serde_json::to_value(&metadata).unwrap();
        checkpoint["future_execution_policy"] = serde_json::json!("fused-w8a8");
        assert!(
            serde_json::from_value::<H3QwenInt8CheckpointMetadata>(checkpoint).is_err(),
            "unknown checkpoint policy must not be silently discarded"
        );

        let mut layer = serde_json::to_value(
            metadata
                .layers
                .get("model.layers.0.self_attn.q_proj")
                .unwrap(),
        )
        .unwrap();
        layer["activation_quantization"] = serde_json::json!("dynamic");
        assert!(
            serde_json::from_value::<H3QwenInt8LayerMetadata>(layer).is_err(),
            "unknown per-layer execution policy must fail closed"
        );

        let mut tensor = serde_json::to_value(
            tensors
                .get("model.layers.0.self_attn.q_proj.weight")
                .unwrap(),
        )
        .unwrap();
        tensor["compression"] = serde_json::json!("opaque");
        assert!(
            serde_json::from_value::<H3QwenTensorSpec>(tensor).is_err(),
            "unknown tensor storage authority must fail closed"
        );
    }

    #[test]
    fn official_or_mixed_namespace_cannot_enter_the_comfy_policy() {
        let (config, mut tensors, metadata) = valid_fixture();
        let embed = tensors.remove("model.embed_tokens.weight").unwrap();
        tensors.insert("model.language_model.embed_tokens.weight".into(), embed);
        assert!(matches!(
            validate_h3_qwen_int8_weight_only_schema(&config, &tensors, &metadata),
            Err(H3QwenInt8PolicyError::Namespace(_))
        ));
    }
}
