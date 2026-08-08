//! Published Qwen3-VL NVFP4-AWQ artifact contract for MiniMax H3.
//!
//! The selected Comfy H3 conditioner is not the older INT8 ConvRot object.
//! It has one INT8 tensorwise token embedding, 350 NVFP4 language projection
//! matrices, and otherwise BF16 tensors. Every NVFP4 projection carries
//! `weight_scale`, `weight_scale_2`, and a `comfy_quant` marker which requires
//! full-precision matrix multiplication. ModelOpt AWQ smoothing is selective:
//! only the attention output and MLP down projections carry
//! `pre_quant_scale`; its absence on the other 250 projections means identity,
//! not an incomplete checkpoint.
//!
//! This module reads only the bounded safetensors header and the small
//! `comfy_quant` marker tensors through one identity-fenced regular-file
//! descriptor. It neither hashes the full artifact nor grants download,
//! license, factory, or runtime authority. Callers must separately authenticate
//! the complete pinned object before mapping tensor payloads.

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, BTreeSet};
use std::fs::{File, Metadata};
use std::io::{Read, Seek, SeekFrom};
use std::path::Path;
use thiserror::Error;

use super::artifacts::{expected_checkpoint_shapes, expected_materialized_keys};
use super::config::{H3ConditionerConfig, H3_SELECTED_LANGUAGE_LAYERS};
use super::qwen_quant::{H3QwenTensorDType, H3QwenTensorSpec};
use super::H3_COMFY_NVFP4_BLOCK_SIZE;

#[cfg(test)]
use super::visual_weights::inspect_safetensors_header;

pub const H3_QWEN_NVFP4_AWQ_STABLE_ID: &str = "minimax-h3.qwen3-vl.layer50.nvfp4-awq.v1";
pub const H3_QWEN_NVFP4_AWQ_FILENAME: &str = "qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors";
pub const H3_QWEN_NVFP4_AWQ_SHA256: &str =
    "35a88d51044231fe332301d7a62aa81e3f2cba62febeb446e2c1e3e0ef76f2c6";
pub const H3_QWEN_NVFP4_AWQ_POLICY_SHA256: &str =
    "b36894e044a993d114a37db743f9aec8cf1492073d1cb9e82f4b04ac92cb25b9";
pub const H3_QWEN_NVFP4_AWQ_FILE_BYTES: u64 = 15_687_142_551;
pub const H3_QWEN_NVFP4_AWQ_HEADER_BYTES: u64 = 231_400;
pub const H3_QWEN_NVFP4_AWQ_PAYLOAD_BYTES: u64 = 15_686_911_143;
pub const H3_QWEN_NVFP4_AWQ_TENSOR_COUNT: usize = 2_054;
pub const H3_QWEN_NVFP4_LINEAR_COUNT: usize = 350;
pub const H3_QWEN_NVFP4_PRE_QUANT_SCALE_COUNT: usize = 100;
pub const H3_QWEN_QUANT_MARKER_COUNT: usize = 351;

const INT8_MARKER_JSON: &[u8] = br#"{"format": "int8_tensorwise"}"#;
const NVFP4_MARKER_JSON: &[u8] = br#"{"format": "nvfp4", "full_precision_matrix_mult": true}"#;

const LANGUAGE_LINEAR_SUFFIXES: &[&str] = &[
    "self_attn.q_proj",
    "self_attn.k_proj",
    "self_attn.v_proj",
    "self_attn.o_proj",
    "mlp.gate_proj",
    "mlp.up_proj",
    "mlp.down_proj",
];

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct H3QwenQuantMarker {
    pub format: String,
    #[serde(default)]
    pub full_precision_matrix_mult: bool,
}

pub type H3QwenNvfp4AwqSchema = (
    BTreeMap<String, H3QwenTensorSpec>,
    BTreeMap<String, H3QwenQuantMarker>,
);

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct H3QwenNvfp4AwqMemoryAccounting {
    pub artifact_file_bytes: u64,
    pub header_bytes: u64,
    pub tensor_payload_bytes: u64,
    pub nvfp4_packed_weight_bytes: u64,
    pub nvfp4_block_scale_bytes: u64,
    pub nvfp4_tensor_scale_bytes: u64,
    pub awq_pre_quant_scale_bytes: u64,
    pub int8_embedding_weight_bytes: u64,
    pub int8_embedding_scale_bytes: u64,
    pub dense_bf16_bytes: u64,
    pub quant_marker_bytes: u64,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum H3QwenNvfp4AwqExecution {
    DenseBf16 {
        source_tensor: String,
    },
    Int8TensorwiseEmbedding {
        weight_tensor: String,
        weight_scale_tensor: String,
    },
    Nvfp4FullPrecision {
        packed_weight_tensor: String,
        block_scale_tensor: String,
        tensor_scale_tensor: String,
        pre_quant_scale_tensor: Option<String>,
    },
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct H3QwenNvfp4AwqPolicy {
    stable_id: String,
    quantized_layers: Vec<String>,
    awq_smoothed_layers: Vec<String>,
    protected_tensors: Vec<String>,
    memory: H3QwenNvfp4AwqMemoryAccounting,
    policy_sha256: String,
}

impl H3QwenNvfp4AwqPolicy {
    pub fn stable_id(&self) -> &str {
        &self.stable_id
    }

    pub fn quantized_layers(&self) -> &[String] {
        &self.quantized_layers
    }

    pub fn awq_smoothed_layers(&self) -> &[String] {
        &self.awq_smoothed_layers
    }

    pub fn protected_tensors(&self) -> &[String] {
        &self.protected_tensors
    }

    pub const fn memory(&self) -> &H3QwenNvfp4AwqMemoryAccounting {
        &self.memory
    }

    pub fn policy_sha256(&self) -> &str {
        &self.policy_sha256
    }

    /// Resolve one canonical H3 model weight to its exact published source
    /// tensors. Names returned here use the Comfy `model.*` / `visual.*`
    /// namespace and are loading instructions, not runtime activation.
    pub fn execution_for_weight(
        &self,
        canonical_weight: &str,
    ) -> Result<H3QwenNvfp4AwqExecution, H3QwenNvfp4AwqError> {
        if canonical_weight == "model.language_model.embed_tokens.weight" {
            return Ok(H3QwenNvfp4AwqExecution::Int8TensorwiseEmbedding {
                weight_tensor: "model.embed_tokens.weight".into(),
                weight_scale_tensor: "model.embed_tokens.weight_scale".into(),
            });
        }
        let Some(canonical_layer) = canonical_weight.strip_suffix(".weight") else {
            return Err(H3QwenNvfp4AwqError::Execution(format!(
                "Qwen execution lookup requires a canonical .weight tensor, got {canonical_weight:?}"
            )));
        };
        if self
            .quantized_layers
            .binary_search_by(|candidate| candidate.as_str().cmp(canonical_layer))
            .is_ok()
        {
            let source_layer = comfy_name(canonical_layer);
            let pre_quant_scale_tensor = self
                .awq_smoothed_layers
                .binary_search_by(|candidate| candidate.as_str().cmp(canonical_layer))
                .is_ok()
                .then(|| format!("{source_layer}.pre_quant_scale"));
            return Ok(H3QwenNvfp4AwqExecution::Nvfp4FullPrecision {
                packed_weight_tensor: format!("{source_layer}.weight"),
                block_scale_tensor: format!("{source_layer}.weight_scale"),
                tensor_scale_tensor: format!("{source_layer}.weight_scale_2"),
                pre_quant_scale_tensor,
            });
        }
        if self
            .protected_tensors
            .binary_search_by(|candidate| candidate.as_str().cmp(canonical_weight))
            .is_ok()
        {
            return Ok(H3QwenNvfp4AwqExecution::DenseBf16 {
                source_tensor: comfy_name(canonical_weight),
            });
        }
        Err(H3QwenNvfp4AwqError::Execution(format!(
            "Qwen tensor {canonical_weight:?} is outside the frozen layer-50 NVFP4-AWQ policy"
        )))
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct H3QwenNvfp4AwqInspection {
    pub stable_id: String,
    pub expected_artifact_sha256: String,
    pub artifact_file_bytes: u64,
    pub header_bytes: u64,
    pub tensor_payload_bytes: u64,
    pub tensor_count: usize,
    pub nvfp4_linear_count: usize,
    pub awq_pre_quant_scale_count: usize,
    pub int8_tensorwise_count: usize,
    pub header_identity_sha256: String,
    pub policy_sha256: String,
}

#[derive(Debug, Error, Eq, PartialEq)]
pub enum H3QwenNvfp4AwqError {
    #[error("invalid MiniMax H3 Qwen NVFP4-AWQ config: {0}")]
    Config(String),
    #[error("invalid MiniMax H3 Qwen NVFP4-AWQ header: {0}")]
    Header(String),
    #[error("invalid MiniMax H3 Qwen NVFP4-AWQ tensor schema: {0}")]
    TensorSchema(String),
    #[error("invalid MiniMax H3 Qwen NVFP4-AWQ metadata: {0}")]
    Metadata(String),
    #[error("invalid MiniMax H3 Qwen NVFP4-AWQ accounting: {0}")]
    Accounting(String),
    #[error("invalid MiniMax H3 Qwen NVFP4-AWQ execution lookup: {0}")]
    Execution(String),
    #[error("failed to read MiniMax H3 Qwen NVFP4-AWQ artifact: {0}")]
    Io(String),
}

fn released_config() -> Result<H3ConditionerConfig, H3QwenNvfp4AwqError> {
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
    .map_err(|error| H3QwenNvfp4AwqError::Config(error.to_string()))
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

fn canonical_awq_smoothed_layers() -> BTreeSet<String> {
    (0..H3_SELECTED_LANGUAGE_LAYERS)
        .flat_map(|layer| {
            ["self_attn.o_proj", "mlp.down_proj"]
                .into_iter()
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

fn round_up(value: usize, multiple: usize, role: &str) -> Result<usize, H3QwenNvfp4AwqError> {
    value
        .checked_add(multiple - 1)
        .map(|value| value / multiple * multiple)
        .ok_or_else(|| H3QwenNvfp4AwqError::Accounting(format!("{role} overflows")))
}

fn nvfp4_specs(
    logical_shape: &[usize],
) -> Result<(H3QwenTensorSpec, H3QwenTensorSpec), H3QwenNvfp4AwqError> {
    let [out_features, in_features] = logical_shape else {
        return Err(H3QwenNvfp4AwqError::TensorSchema(format!(
            "NVFP4 projection must have rank two, got {logical_shape:?}"
        )));
    };
    let padded_out = round_up(*out_features, 16, "NVFP4 output shape")?;
    let padded_in = round_up(*in_features, 16, "NVFP4 input shape")?;
    let scale_rows = round_up(padded_out, 128, "NVFP4 scale output shape")?;
    let scale_columns = round_up(
        padded_in / H3_COMFY_NVFP4_BLOCK_SIZE,
        4,
        "NVFP4 scale input shape",
    )?;
    Ok((
        H3QwenTensorSpec {
            dtype: H3QwenTensorDType::U8,
            shape: vec![padded_out, padded_in / 2],
        },
        H3QwenTensorSpec {
            dtype: H3QwenTensorDType::F8E4M3,
            shape: vec![scale_rows, scale_columns],
        },
    ))
}

/// Construct the exact published schema without opening a checkpoint.
/// Returned tensor names use the Comfy `model.*` / `visual.*` namespace.
pub fn expected_h3_qwen_nvfp4_awq_schema(
    config: &H3ConditionerConfig,
) -> Result<H3QwenNvfp4AwqSchema, H3QwenNvfp4AwqError> {
    config
        .validate()
        .map_err(|error| H3QwenNvfp4AwqError::Config(error.to_string()))?;
    let shapes = expected_checkpoint_shapes(config);
    let quantized = canonical_quantized_layers();
    let smoothed = canonical_awq_smoothed_layers();
    let mut tensors = BTreeMap::new();
    let mut markers = BTreeMap::new();
    for canonical in expected_materialized_keys() {
        let logical_shape = shapes.get(&canonical).ok_or_else(|| {
            H3QwenNvfp4AwqError::TensorSchema(format!("missing production shape for {canonical:?}"))
        })?;
        let source = comfy_name(&canonical);
        if canonical == "model.language_model.embed_tokens.weight" {
            tensors.insert(
                source.clone(),
                H3QwenTensorSpec {
                    dtype: H3QwenTensorDType::I8,
                    shape: logical_shape.clone(),
                },
            );
            let base = source.strip_suffix(".weight").expect("embedding weight");
            tensors.insert(
                format!("{base}.weight_scale"),
                H3QwenTensorSpec {
                    dtype: H3QwenTensorDType::F32,
                    shape: vec![logical_shape[0], 1],
                },
            );
            tensors.insert(
                format!("{base}.comfy_quant"),
                H3QwenTensorSpec {
                    dtype: H3QwenTensorDType::U8,
                    shape: vec![INT8_MARKER_JSON.len()],
                },
            );
            markers.insert(
                base.into(),
                H3QwenQuantMarker {
                    format: "int8_tensorwise".into(),
                    full_precision_matrix_mult: false,
                },
            );
            continue;
        }
        let Some(canonical_layer) = canonical.strip_suffix(".weight") else {
            tensors.insert(
                source,
                H3QwenTensorSpec {
                    dtype: H3QwenTensorDType::Bf16,
                    shape: logical_shape.clone(),
                },
            );
            continue;
        };
        if !quantized.contains(canonical_layer) {
            tensors.insert(
                source,
                H3QwenTensorSpec {
                    dtype: H3QwenTensorDType::Bf16,
                    shape: logical_shape.clone(),
                },
            );
            continue;
        }
        let source_layer = source
            .strip_suffix(".weight")
            .expect("projection weight")
            .to_owned();
        let (packed, block_scale) = nvfp4_specs(logical_shape)?;
        tensors.insert(source, packed);
        tensors.insert(format!("{source_layer}.weight_scale"), block_scale);
        tensors.insert(
            format!("{source_layer}.weight_scale_2"),
            H3QwenTensorSpec {
                dtype: H3QwenTensorDType::F32,
                shape: Vec::new(),
            },
        );
        tensors.insert(
            format!("{source_layer}.comfy_quant"),
            H3QwenTensorSpec {
                dtype: H3QwenTensorDType::U8,
                shape: vec![NVFP4_MARKER_JSON.len()],
            },
        );
        if smoothed.contains(canonical_layer) {
            tensors.insert(
                format!("{source_layer}.pre_quant_scale"),
                H3QwenTensorSpec {
                    dtype: H3QwenTensorDType::Bf16,
                    shape: vec![logical_shape[1]],
                },
            );
        }
        markers.insert(
            source_layer,
            H3QwenQuantMarker {
                format: "nvfp4".into(),
                full_precision_matrix_mult: true,
            },
        );
    }
    Ok((tensors, markers))
}

fn checked_tensor_bytes(spec: &H3QwenTensorSpec) -> Result<u64, H3QwenNvfp4AwqError> {
    spec.shape
        .iter()
        .try_fold(1_u64, |elements, dimension| {
            elements.checked_mul(*dimension as u64)
        })
        .and_then(|elements| elements.checked_mul(spec.dtype.byte_width()))
        .ok_or_else(|| H3QwenNvfp4AwqError::Accounting("tensor byte count overflows".into()))
}

/// Validate an already parsed schema and freeze the only supported loading
/// policy. This function is artifact-neutral and grants no filesystem or
/// runtime authority.
pub fn validate_h3_qwen_nvfp4_awq_schema(
    config: &H3ConditionerConfig,
    tensors: &BTreeMap<String, H3QwenTensorSpec>,
    markers: &BTreeMap<String, H3QwenQuantMarker>,
) -> Result<H3QwenNvfp4AwqPolicy, H3QwenNvfp4AwqError> {
    let (expected_tensors, expected_markers) = expected_h3_qwen_nvfp4_awq_schema(config)?;
    if tensors != &expected_tensors {
        let expected_keys = expected_tensors.keys().collect::<BTreeSet<_>>();
        let actual_keys = tensors.keys().collect::<BTreeSet<_>>();
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
        if !missing.is_empty() || !unexpected.is_empty() {
            return Err(H3QwenNvfp4AwqError::TensorSchema(format!(
                "tensor set mismatch: missing={missing:?} unexpected={unexpected:?}"
            )));
        }
        let (name, expected) = expected_tensors
            .iter()
            .find(|(name, expected)| tensors.get(*name) != Some(*expected))
            .expect("unequal maps have one unequal tensor");
        let actual = &tensors[name];
        return Err(H3QwenNvfp4AwqError::TensorSchema(format!(
            "tensor {name:?} expected {:?} {:?}, got {:?} {:?}",
            expected.dtype, expected.shape, actual.dtype, actual.shape
        )));
    }
    if markers != &expected_markers {
        let expected_keys = expected_markers.keys().collect::<BTreeSet<_>>();
        let actual_keys = markers.keys().collect::<BTreeSet<_>>();
        if expected_keys != actual_keys {
            return Err(H3QwenNvfp4AwqError::Metadata(format!(
                "quantized layer set mismatch: expected {} layers, got {}",
                expected_keys.len(),
                actual_keys.len()
            )));
        }
        let (name, expected) = expected_markers
            .iter()
            .find(|(name, expected)| markers.get(*name) != Some(*expected))
            .expect("unequal maps have one unequal marker");
        return Err(H3QwenNvfp4AwqError::Metadata(format!(
            "layer {name:?} expected marker {expected:?}, got {:?}",
            markers[name]
        )));
    }

    let quantized = canonical_quantized_layers();
    let smoothed = canonical_awq_smoothed_layers();
    if quantized.len() != H3_QWEN_NVFP4_LINEAR_COUNT
        || smoothed.len() != H3_QWEN_NVFP4_PRE_QUANT_SCALE_COUNT
    {
        return Err(H3QwenNvfp4AwqError::Accounting(format!(
            "derived {} NVFP4 layers and {} AWQ scales",
            quantized.len(),
            smoothed.len()
        )));
    }

    let mut memory = H3QwenNvfp4AwqMemoryAccounting {
        artifact_file_bytes: H3_QWEN_NVFP4_AWQ_FILE_BYTES,
        header_bytes: H3_QWEN_NVFP4_AWQ_HEADER_BYTES,
        tensor_payload_bytes: 0,
        nvfp4_packed_weight_bytes: 0,
        nvfp4_block_scale_bytes: 0,
        nvfp4_tensor_scale_bytes: 0,
        awq_pre_quant_scale_bytes: 0,
        int8_embedding_weight_bytes: 0,
        int8_embedding_scale_bytes: 0,
        dense_bf16_bytes: 0,
        quant_marker_bytes: 0,
    };
    for (name, spec) in &expected_tensors {
        let bytes = checked_tensor_bytes(spec)?;
        memory.tensor_payload_bytes = memory
            .tensor_payload_bytes
            .checked_add(bytes)
            .ok_or_else(|| H3QwenNvfp4AwqError::Accounting("payload bytes overflow".into()))?;
        if name == "model.embed_tokens.weight" {
            memory.int8_embedding_weight_bytes = bytes;
        } else if name == "model.embed_tokens.weight_scale" {
            memory.int8_embedding_scale_bytes = bytes;
        } else if name.ends_with(".comfy_quant") {
            memory.quant_marker_bytes = memory
                .quant_marker_bytes
                .checked_add(bytes)
                .ok_or_else(|| H3QwenNvfp4AwqError::Accounting("marker bytes overflow".into()))?;
        } else if name.ends_with(".pre_quant_scale") {
            memory.awq_pre_quant_scale_bytes = memory
                .awq_pre_quant_scale_bytes
                .checked_add(bytes)
                .ok_or_else(|| H3QwenNvfp4AwqError::Accounting("AWQ bytes overflow".into()))?;
        } else if name.ends_with(".weight_scale_2") {
            memory.nvfp4_tensor_scale_bytes = memory
                .nvfp4_tensor_scale_bytes
                .checked_add(bytes)
                .ok_or_else(|| {
                    H3QwenNvfp4AwqError::Accounting("tensor-scale bytes overflow".into())
                })?;
        } else if name.ends_with(".weight_scale") {
            memory.nvfp4_block_scale_bytes = memory
                .nvfp4_block_scale_bytes
                .checked_add(bytes)
                .ok_or_else(|| {
                    H3QwenNvfp4AwqError::Accounting("block-scale bytes overflow".into())
                })?;
        } else if spec.dtype == H3QwenTensorDType::U8 && name.ends_with(".weight") {
            memory.nvfp4_packed_weight_bytes = memory
                .nvfp4_packed_weight_bytes
                .checked_add(bytes)
                .ok_or_else(|| {
                    H3QwenNvfp4AwqError::Accounting("packed-weight bytes overflow".into())
                })?;
        } else if spec.dtype == H3QwenTensorDType::Bf16 {
            memory.dense_bf16_bytes = memory
                .dense_bf16_bytes
                .checked_add(bytes)
                .ok_or_else(|| H3QwenNvfp4AwqError::Accounting("dense bytes overflow".into()))?;
        } else {
            return Err(H3QwenNvfp4AwqError::Accounting(format!(
                "tensor {name:?} is not assigned to a memory category"
            )));
        }
    }
    if memory.tensor_payload_bytes != H3_QWEN_NVFP4_AWQ_PAYLOAD_BYTES {
        return Err(H3QwenNvfp4AwqError::Accounting(format!(
            "derived {} payload bytes, expected {H3_QWEN_NVFP4_AWQ_PAYLOAD_BYTES}",
            memory.tensor_payload_bytes
        )));
    }
    let protected_tensors = expected_materialized_keys()
        .into_iter()
        .filter(|name| {
            name != "model.language_model.embed_tokens.weight"
                && !name
                    .strip_suffix(".weight")
                    .is_some_and(|layer| quantized.contains(layer))
        })
        .collect::<Vec<_>>();

    let mut digest = Sha256::new();
    digest.update(H3_QWEN_NVFP4_AWQ_STABLE_ID.as_bytes());
    for (name, spec) in &expected_tensors {
        digest.update((name.len() as u64).to_le_bytes());
        digest.update(name.as_bytes());
        digest.update(spec.dtype.stable_id().as_bytes());
        for dimension in &spec.shape {
            digest.update((*dimension as u64).to_le_bytes());
        }
    }
    for (name, marker) in &expected_markers {
        digest.update((name.len() as u64).to_le_bytes());
        digest.update(name.as_bytes());
        digest.update(marker.format.as_bytes());
        digest.update([u8::from(marker.full_precision_matrix_mult)]);
    }
    let policy_sha256 = hex_digest(digest.finalize());
    if policy_sha256 != H3_QWEN_NVFP4_AWQ_POLICY_SHA256 {
        return Err(H3QwenNvfp4AwqError::Accounting(format!(
            "derived policy identity {policy_sha256}, expected {H3_QWEN_NVFP4_AWQ_POLICY_SHA256}"
        )));
    }
    Ok(H3QwenNvfp4AwqPolicy {
        stable_id: H3_QWEN_NVFP4_AWQ_STABLE_ID.into(),
        quantized_layers: quantized.into_iter().collect(),
        awq_smoothed_layers: smoothed.into_iter().collect(),
        protected_tensors,
        memory,
        policy_sha256,
    })
}

fn parse_dtype(dtype: &str) -> Result<H3QwenTensorDType, H3QwenNvfp4AwqError> {
    match dtype {
        "BF16" => Ok(H3QwenTensorDType::Bf16),
        "F8_E4M3" => Ok(H3QwenTensorDType::F8E4M3),
        "F32" => Ok(H3QwenTensorDType::F32),
        "I8" => Ok(H3QwenTensorDType::I8),
        "U8" => Ok(H3QwenTensorDType::U8),
        other => Err(H3QwenNvfp4AwqError::TensorSchema(format!(
            "unsupported published dtype {other:?}"
        ))),
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
struct FileIdentity {
    len: u64,
    modified: Option<std::time::SystemTime>,
    #[cfg(unix)]
    device: u64,
    #[cfg(unix)]
    inode: u64,
    #[cfg(unix)]
    changed_seconds: i64,
    #[cfg(unix)]
    changed_nanoseconds: i64,
}

fn file_identity(metadata: &Metadata) -> FileIdentity {
    #[cfg(unix)]
    use std::os::unix::fs::MetadataExt;
    FileIdentity {
        len: metadata.len(),
        modified: metadata.modified().ok(),
        #[cfg(unix)]
        device: metadata.dev(),
        #[cfg(unix)]
        inode: metadata.ino(),
        #[cfg(unix)]
        changed_seconds: metadata.ctime(),
        #[cfg(unix)]
        changed_nanoseconds: metadata.ctime_nsec(),
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
struct H3SafetensorsTensorHeader {
    dtype: String,
    shape: Vec<usize>,
    data_offsets: [u64; 2],
}

#[derive(Clone, Debug, Eq, PartialEq)]
struct H3SafetensorsHeader {
    tensors: BTreeMap<String, H3SafetensorsTensorHeader>,
    header_len: u64,
    file_len: u64,
    bytes: Vec<u8>,
    has_metadata: bool,
}

struct UniqueH3SafetensorsHeader(BTreeMap<String, serde_json::Value>);

impl<'de> Deserialize<'de> for UniqueH3SafetensorsHeader {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        struct UniqueHeaderVisitor;

        impl<'de> serde::de::Visitor<'de> for UniqueHeaderVisitor {
            type Value = UniqueH3SafetensorsHeader;

            fn expecting(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                formatter.write_str("a safetensors header with unique tensor names")
            }

            fn visit_map<A>(self, mut map: A) -> Result<Self::Value, A::Error>
            where
                A: serde::de::MapAccess<'de>,
            {
                let mut entries = BTreeMap::new();
                while let Some((name, value)) = map.next_entry::<String, serde_json::Value>()? {
                    if entries.insert(name.clone(), value).is_some() {
                        return Err(serde::de::Error::custom(format!(
                            "duplicate safetensors tensor {name}"
                        )));
                    }
                }
                Ok(UniqueH3SafetensorsHeader(entries))
            }
        }

        deserializer.deserialize_map(UniqueHeaderVisitor)
    }
}

fn checked_path_identity(
    path: &Path,
    expected: Option<&FileIdentity>,
    operation: &str,
) -> Result<FileIdentity, H3QwenNvfp4AwqError> {
    let metadata = std::fs::symlink_metadata(path)
        .map_err(|error| H3QwenNvfp4AwqError::Io(format!("{operation}: {error}")))?;
    if metadata.file_type().is_symlink() {
        return Err(H3QwenNvfp4AwqError::Io(format!(
            "{operation}: artifact path must not be a symlink"
        )));
    }
    if !metadata.file_type().is_file() {
        return Err(H3QwenNvfp4AwqError::Io(format!(
            "{operation}: artifact path must be a regular file"
        )));
    }
    let identity = file_identity(&metadata);
    if expected.is_some_and(|expected| *expected != identity) {
        return Err(H3QwenNvfp4AwqError::Io(format!(
            "{operation}: artifact path identity changed"
        )));
    }
    Ok(identity)
}

fn checked_open_identity(
    file: &File,
    expected: &FileIdentity,
    operation: &str,
) -> Result<(), H3QwenNvfp4AwqError> {
    let metadata = file
        .metadata()
        .map_err(|error| H3QwenNvfp4AwqError::Io(format!("{operation}: {error}")))?;
    if !metadata.file_type().is_file() || file_identity(&metadata) != *expected {
        return Err(H3QwenNvfp4AwqError::Io(format!(
            "{operation}: opened artifact identity changed"
        )));
    }
    Ok(())
}

fn read_h3_safetensors_header(
    file: &mut File,
    file_len: u64,
) -> Result<H3SafetensorsHeader, H3QwenNvfp4AwqError> {
    file.seek(SeekFrom::Start(0))
        .map_err(|error| H3QwenNvfp4AwqError::Io(error.to_string()))?;
    let mut len_bytes = [0_u8; 8];
    file.read_exact(&mut len_bytes)
        .map_err(|error| H3QwenNvfp4AwqError::Io(error.to_string()))?;
    let header_len = u64::from_le_bytes(len_bytes);
    if header_len != H3_QWEN_NVFP4_AWQ_HEADER_BYTES
        || header_len.checked_add(8).is_none_or(|end| end > file_len)
    {
        return Err(H3QwenNvfp4AwqError::Header(format!(
            "expected {H3_QWEN_NVFP4_AWQ_HEADER_BYTES} header bytes, got {header_len}"
        )));
    }
    let header_size = usize::try_from(header_len)
        .map_err(|_| H3QwenNvfp4AwqError::Header("header length overflows usize".into()))?;
    let mut json_bytes = vec![0_u8; header_size];
    file.read_exact(&mut json_bytes)
        .map_err(|error| H3QwenNvfp4AwqError::Io(error.to_string()))?;
    let UniqueH3SafetensorsHeader(mut raw) = serde_json::from_slice(&json_bytes)
        .map_err(|error| H3QwenNvfp4AwqError::Header(error.to_string()))?;
    let has_metadata = raw.remove("__metadata__").is_some();
    let data_len = file_len - header_len - 8;
    let mut tensors = BTreeMap::new();
    for (name, value) in raw {
        #[derive(Deserialize)]
        #[serde(deny_unknown_fields)]
        struct Entry {
            dtype: String,
            shape: Vec<usize>,
            data_offsets: [u64; 2],
        }

        let entry: Entry = serde_json::from_value(value).map_err(|error| {
            H3QwenNvfp4AwqError::Header(format!("invalid tensor {name:?}: {error}"))
        })?;
        if entry.data_offsets[0] > entry.data_offsets[1] || entry.data_offsets[1] > data_len {
            return Err(H3QwenNvfp4AwqError::Header(format!(
                "invalid safetensors offsets for {name:?}"
            )));
        }
        let dtype = parse_dtype(&entry.dtype)?;
        let elements = entry.shape.iter().try_fold(1_u64, |total, &dimension| {
            total.checked_mul(dimension as u64)
        });
        let expected_bytes = elements
            .and_then(|elements| elements.checked_mul(dtype.byte_width()))
            .ok_or_else(|| {
                H3QwenNvfp4AwqError::Header(format!("byte-size overflow for {name:?}"))
            })?;
        if entry.data_offsets[1] - entry.data_offsets[0] != expected_bytes {
            return Err(H3QwenNvfp4AwqError::Header(format!(
                "safetensors byte range does not match dtype/shape for {name:?}"
            )));
        }
        tensors.insert(
            name,
            H3SafetensorsTensorHeader {
                dtype: entry.dtype,
                shape: entry.shape,
                data_offsets: entry.data_offsets,
            },
        );
    }

    let mut cursor = 0_u64;
    let mut ranges = tensors
        .iter()
        .map(|(name, tensor)| (tensor.data_offsets, name.as_str()))
        .collect::<Vec<_>>();
    ranges.sort_by_key(|(offsets, _)| offsets[0]);
    for (offsets, name) in ranges {
        if offsets[0] != cursor {
            return Err(H3QwenNvfp4AwqError::Header(format!(
                "non-contiguous or overlapping safetensors data before {name:?}"
            )));
        }
        cursor = offsets[1];
    }
    if cursor != data_len {
        return Err(H3QwenNvfp4AwqError::Header(format!(
            "safetensors data section has {} unclaimed bytes",
            data_len - cursor
        )));
    }

    let mut bytes = Vec::with_capacity(8 + header_size);
    bytes.extend_from_slice(&len_bytes);
    bytes.extend_from_slice(&json_bytes);
    Ok(H3SafetensorsHeader {
        tensors,
        header_len,
        file_len,
        bytes,
        has_metadata,
    })
}

/// Inspect the exact selected H3 Qwen object using bounded reads.
///
/// Symlinks and non-regular files are rejected. One opened descriptor supplies
/// the prefix, header, header identity, and all marker bytes while descriptor
/// and path identities are fenced before and after bounded inspection. This
/// validates file/header size, every tensor dtype/shape/range, all 351 small
/// quantization marker payloads, and the frozen policy identity. It does
/// **not** hash the 15.7 GB tensor payload; compare a separately computed full
/// digest with [`H3_QWEN_NVFP4_AWQ_SHA256`] before treating weights as
/// authenticated.
pub fn inspect_h3_qwen_nvfp4_awq_header(
    path: &Path,
) -> Result<H3QwenNvfp4AwqInspection, H3QwenNvfp4AwqError> {
    let before = checked_path_identity(path, None, "before bounded inspection")?;
    let mut file = File::open(path).map_err(|error| H3QwenNvfp4AwqError::Io(error.to_string()))?;
    checked_open_identity(&file, &before, "after opening artifact")?;
    checked_path_identity(path, Some(&before), "after opening artifact")?;
    let header = read_h3_safetensors_header(&mut file, before.len)?;
    checked_open_identity(&file, &before, "after header inspection")?;
    checked_path_identity(path, Some(&before), "after header inspection")?;
    if header.file_len != H3_QWEN_NVFP4_AWQ_FILE_BYTES
        || header.header_len != H3_QWEN_NVFP4_AWQ_HEADER_BYTES
        || header.tensors.len() != H3_QWEN_NVFP4_AWQ_TENSOR_COUNT
    {
        return Err(H3QwenNvfp4AwqError::Header(format!(
            "expected {H3_QWEN_NVFP4_AWQ_FILE_BYTES} file bytes, {H3_QWEN_NVFP4_AWQ_HEADER_BYTES} header bytes, and {H3_QWEN_NVFP4_AWQ_TENSOR_COUNT} tensors; got {}, {}, and {}",
            header.file_len,
            header.header_len,
            header.tensors.len()
        )));
    }
    let tensors = header
        .tensors
        .iter()
        .map(|(name, tensor)| {
            Ok((
                name.clone(),
                H3QwenTensorSpec {
                    dtype: parse_dtype(&tensor.dtype)?,
                    shape: tensor.shape.clone(),
                },
            ))
        })
        .collect::<Result<BTreeMap<_, _>, H3QwenNvfp4AwqError>>()?;

    let data_start = 8_u64
        .checked_add(header.header_len)
        .ok_or_else(|| H3QwenNvfp4AwqError::Header("data offset overflows".into()))?;
    let mut markers = BTreeMap::new();
    for (name, tensor) in &header.tensors {
        let Some(base) = name.strip_suffix(".comfy_quant") else {
            continue;
        };
        if tensor.dtype != "U8" || tensor.shape.len() != 1 || tensor.shape[0] > 128 {
            return Err(H3QwenNvfp4AwqError::Metadata(format!(
                "quant marker {name:?} must be bounded rank-one U8"
            )));
        }
        let length = usize::try_from(tensor.data_offsets[1] - tensor.data_offsets[0])
            .map_err(|_| H3QwenNvfp4AwqError::Metadata("marker length overflows".into()))?;
        let mut bytes = vec![0_u8; length];
        file.seek(SeekFrom::Start(
            data_start
                .checked_add(tensor.data_offsets[0])
                .ok_or_else(|| H3QwenNvfp4AwqError::Metadata("marker offset overflows".into()))?,
        ))
        .and_then(|_| file.read_exact(&mut bytes))
        .map_err(|error| H3QwenNvfp4AwqError::Io(error.to_string()))?;
        let marker: H3QwenQuantMarker = serde_json::from_slice(&bytes).map_err(|error| {
            H3QwenNvfp4AwqError::Metadata(format!("invalid marker {name:?}: {error}"))
        })?;
        if markers.insert(base.into(), marker).is_some() {
            return Err(H3QwenNvfp4AwqError::Metadata(format!(
                "duplicate marker base {base:?}"
            )));
        }
    }
    let policy = validate_h3_qwen_nvfp4_awq_schema(&released_config()?, &tensors, &markers)?;

    if header.has_metadata {
        return Err(H3QwenNvfp4AwqError::Metadata(
            "published Qwen NVFP4-AWQ object must not add safetensors __metadata__".into(),
        ));
    }
    checked_open_identity(&file, &before, "after marker inspection")?;
    checked_path_identity(path, Some(&before), "after marker inspection")?;
    Ok(H3QwenNvfp4AwqInspection {
        stable_id: H3_QWEN_NVFP4_AWQ_STABLE_ID.into(),
        expected_artifact_sha256: H3_QWEN_NVFP4_AWQ_SHA256.into(),
        artifact_file_bytes: header.file_len,
        header_bytes: header.header_len,
        tensor_payload_bytes: policy.memory().tensor_payload_bytes,
        tensor_count: header.tensors.len(),
        nvfp4_linear_count: policy.quantized_layers().len(),
        awq_pre_quant_scale_count: policy.awq_smoothed_layers().len(),
        int8_tensorwise_count: markers
            .values()
            .filter(|marker| marker.format == "int8_tensorwise")
            .count(),
        header_identity_sha256: hex_digest(Sha256::digest(&header.bytes)),
        policy_sha256: policy.policy_sha256().into(),
    })
}

fn hex_digest(bytes: impl AsRef<[u8]>) -> String {
    bytes
        .as_ref()
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;
    use std::fs::OpenOptions;
    use std::io::Write;
    use std::path::PathBuf;

    fn fixture() -> (
        H3ConditionerConfig,
        BTreeMap<String, H3QwenTensorSpec>,
        BTreeMap<String, H3QwenQuantMarker>,
    ) {
        let config = released_config().unwrap();
        let (tensors, markers) = expected_h3_qwen_nvfp4_awq_schema(&config).unwrap();
        (config, tensors, markers)
    }

    fn temp_path(tag: &str) -> PathBuf {
        std::env::temp_dir().join(format!(
            "mold-h3-qwen-nvfp4-{tag}-{}-{}.safetensors",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ))
    }

    fn dtype_name(dtype: H3QwenTensorDType) -> &'static str {
        match dtype {
            H3QwenTensorDType::Bf16 => "BF16",
            H3QwenTensorDType::F8E4M3 => "F8_E4M3",
            H3QwenTensorDType::F32 => "F32",
            H3QwenTensorDType::I8 => "I8",
            H3QwenTensorDType::U8 => "U8",
        }
    }

    fn sparse_published_fixture_with_metadata(include_metadata: bool) -> PathBuf {
        let (_, tensors, markers) = fixture();
        // Safetensors payload order is independent of JSON key order. Put the
        // numerous small tensors first, like the published converter does, so
        // their decimal offsets keep the synthetic header below the frozen
        // published header size; the remaining bytes are legal JSON padding.
        let mut payload_order = tensors.iter().collect::<Vec<_>>();
        payload_order.sort_by_key(|(name, spec)| (checked_tensor_bytes(spec).unwrap(), *name));
        let mut ranges = BTreeMap::new();
        let mut cursor = 0_u64;
        for (name, spec) in payload_order {
            let bytes = checked_tensor_bytes(spec).unwrap();
            ranges.insert(name.clone(), [cursor, cursor + bytes]);
            cursor += bytes;
        }
        assert_eq!(cursor, H3_QWEN_NVFP4_AWQ_PAYLOAD_BYTES);

        let mut root = serde_json::Map::new();
        if include_metadata {
            root.insert("__metadata__".into(), json!({"format": "pt"}));
        }
        let mut marker_ranges = Vec::new();
        for (name, spec) in &tensors {
            let offsets = ranges[name];
            root.insert(
                name.clone(),
                json!({
                    "dtype": dtype_name(spec.dtype),
                    "shape": spec.shape,
                    "data_offsets": offsets
                }),
            );
            if let Some(base) = name.strip_suffix(".comfy_quant") {
                let marker = &markers[base];
                let bytes = if marker.format == "int8_tensorwise" {
                    INT8_MARKER_JSON
                } else {
                    NVFP4_MARKER_JSON
                };
                marker_ranges.push((offsets[0], bytes));
            }
        }
        let mut header = serde_json::to_vec(&root).unwrap();
        assert!(
            header.len() <= H3_QWEN_NVFP4_AWQ_HEADER_BYTES as usize,
            "synthetic header is {} bytes",
            header.len()
        );
        header.resize(H3_QWEN_NVFP4_AWQ_HEADER_BYTES as usize, b' ');
        let path = temp_path("published");
        let mut file = OpenOptions::new()
            .create_new(true)
            .read(true)
            .write(true)
            .open(&path)
            .unwrap();
        file.set_len(H3_QWEN_NVFP4_AWQ_FILE_BYTES).unwrap();
        file.write_all(&H3_QWEN_NVFP4_AWQ_HEADER_BYTES.to_le_bytes())
            .unwrap();
        file.write_all(&header).unwrap();
        let data_start = 8 + H3_QWEN_NVFP4_AWQ_HEADER_BYTES;
        for (offset, bytes) in marker_ranges {
            file.seek(SeekFrom::Start(data_start + offset)).unwrap();
            file.write_all(bytes).unwrap();
        }
        file.sync_all().unwrap();
        path
    }

    fn sparse_published_fixture() -> PathBuf {
        sparse_published_fixture_with_metadata(false)
    }

    #[test]
    fn published_schema_and_byte_accounting_are_frozen() {
        let (config, tensors, markers) = fixture();
        let policy = validate_h3_qwen_nvfp4_awq_schema(&config, &tensors, &markers).unwrap();
        assert_eq!(tensors.len(), H3_QWEN_NVFP4_AWQ_TENSOR_COUNT);
        assert_eq!(markers.len(), H3_QWEN_QUANT_MARKER_COUNT);
        assert_eq!(policy.quantized_layers().len(), H3_QWEN_NVFP4_LINEAR_COUNT);
        assert_eq!(
            policy.awq_smoothed_layers().len(),
            H3_QWEN_NVFP4_PRE_QUANT_SCALE_COUNT
        );
        assert_eq!(policy.memory().tensor_payload_bytes, 15_686_911_143);
        assert_eq!(policy.memory().nvfp4_packed_weight_bytes, 12_189_696_000);
        assert_eq!(policy.memory().nvfp4_block_scale_bytes, 1_523_712_000);
        assert_eq!(policy.memory().nvfp4_tensor_scale_bytes, 1_400);
        assert_eq!(policy.memory().awq_pre_quant_scale_bytes, 3_379_200);
        assert_eq!(policy.memory().int8_embedding_weight_bytes, 777_912_320);
        assert_eq!(policy.memory().int8_embedding_scale_bytes, 607_744);
        assert_eq!(policy.memory().dense_bf16_bytes, 1_191_583_200);
        assert_eq!(policy.memory().quant_marker_bytes, 19_279);
        assert_eq!(policy.policy_sha256(), H3_QWEN_NVFP4_AWQ_POLICY_SHA256);
    }

    #[test]
    fn selective_awq_and_int8_embedding_loading_contract_is_exact() {
        let (config, tensors, markers) = fixture();
        let policy = validate_h3_qwen_nvfp4_awq_schema(&config, &tensors, &markers).unwrap();
        assert_eq!(
            policy
                .execution_for_weight("model.language_model.embed_tokens.weight")
                .unwrap(),
            H3QwenNvfp4AwqExecution::Int8TensorwiseEmbedding {
                weight_tensor: "model.embed_tokens.weight".into(),
                weight_scale_tensor: "model.embed_tokens.weight_scale".into()
            }
        );
        assert_eq!(
            policy
                .execution_for_weight("model.language_model.layers.0.mlp.down_proj.weight")
                .unwrap(),
            H3QwenNvfp4AwqExecution::Nvfp4FullPrecision {
                packed_weight_tensor: "model.layers.0.mlp.down_proj.weight".into(),
                block_scale_tensor: "model.layers.0.mlp.down_proj.weight_scale".into(),
                tensor_scale_tensor: "model.layers.0.mlp.down_proj.weight_scale_2".into(),
                pre_quant_scale_tensor: Some("model.layers.0.mlp.down_proj.pre_quant_scale".into()),
            }
        );
        assert_eq!(
            policy
                .execution_for_weight("model.language_model.layers.0.mlp.gate_proj.weight")
                .unwrap(),
            H3QwenNvfp4AwqExecution::Nvfp4FullPrecision {
                packed_weight_tensor: "model.layers.0.mlp.gate_proj.weight".into(),
                block_scale_tensor: "model.layers.0.mlp.gate_proj.weight_scale".into(),
                tensor_scale_tensor: "model.layers.0.mlp.gate_proj.weight_scale_2".into(),
                pre_quant_scale_tensor: None,
            }
        );
        assert_eq!(
            policy
                .execution_for_weight("model.visual.blocks.0.attn.qkv.weight")
                .unwrap(),
            H3QwenNvfp4AwqExecution::DenseBf16 {
                source_tensor: "visual.blocks.0.attn.qkv.weight".into()
            }
        );
    }

    #[test]
    fn dtype_shape_sidecar_and_marker_mutations_fail_closed() {
        let (config, tensors, markers) = fixture();
        let base = "model.layers.0.mlp.down_proj";

        let mut missing = tensors.clone();
        missing.remove(&format!("{base}.weight_scale_2"));
        assert!(matches!(
            validate_h3_qwen_nvfp4_awq_schema(&config, &missing, &markers),
            Err(H3QwenNvfp4AwqError::TensorSchema(_))
        ));

        let mut wrong_dtype = tensors.clone();
        wrong_dtype
            .get_mut(&format!("{base}.weight_scale"))
            .unwrap()
            .dtype = H3QwenTensorDType::F32;
        assert!(matches!(
            validate_h3_qwen_nvfp4_awq_schema(&config, &wrong_dtype, &markers),
            Err(H3QwenNvfp4AwqError::TensorSchema(_))
        ));

        let mut missing_awq = tensors.clone();
        missing_awq.remove(&format!("{base}.pre_quant_scale"));
        assert!(validate_h3_qwen_nvfp4_awq_schema(&config, &missing_awq, &markers).is_err());

        let mut unexpected_awq = tensors.clone();
        unexpected_awq.insert(
            "model.layers.0.mlp.gate_proj.pre_quant_scale".into(),
            H3QwenTensorSpec {
                dtype: H3QwenTensorDType::Bf16,
                shape: vec![5120],
            },
        );
        assert!(validate_h3_qwen_nvfp4_awq_schema(&config, &unexpected_awq, &markers).is_err());

        let mut kernel_marker = markers.clone();
        kernel_marker
            .get_mut(base)
            .unwrap()
            .full_precision_matrix_mult = false;
        assert!(matches!(
            validate_h3_qwen_nvfp4_awq_schema(&config, &tensors, &kernel_marker),
            Err(H3QwenNvfp4AwqError::Metadata(_))
        ));
    }

    #[test]
    fn serialized_markers_reject_unknown_execution_fields() {
        let marker = br#"{"format":"nvfp4","full_precision_matrix_mult":true,"native":true}"#;
        assert!(serde_json::from_slice::<H3QwenQuantMarker>(marker).is_err());
    }

    #[cfg(unix)]
    #[test]
    fn bounded_inspector_rejects_symlinks_before_opening_weights() {
        use std::os::unix::fs::symlink;

        let target = temp_path("symlink-target");
        let link = temp_path("symlink");
        std::fs::write(&target, b"not opened through the link").unwrap();
        symlink(&target, &link).unwrap();
        let error = inspect_h3_qwen_nvfp4_awq_header(&link).unwrap_err();
        assert!(
            matches!(error, H3QwenNvfp4AwqError::Io(message) if message.contains("must not be a symlink"))
        );
        std::fs::remove_file(link).unwrap();
        std::fs::remove_file(target).unwrap();
    }

    #[cfg(unix)]
    #[test]
    fn fd_and_path_identity_fences_detect_replacement() {
        let path = temp_path("identity-original");
        let replacement = temp_path("identity-replacement");
        std::fs::write(&path, b"original").unwrap();
        std::fs::write(&replacement, b"replacement").unwrap();
        let expected = checked_path_identity(&path, None, "test setup").unwrap();
        let file = File::open(&path).unwrap();
        checked_open_identity(&file, &expected, "test setup").unwrap();

        std::fs::rename(&replacement, &path).unwrap();

        assert!(matches!(
            checked_open_identity(&file, &expected, "after replacement"),
            Err(H3QwenNvfp4AwqError::Io(message)) if message.contains("identity changed")
        ));
        assert!(matches!(
            checked_path_identity(&path, Some(&expected), "after replacement"),
            Err(H3QwenNvfp4AwqError::Io(message)) if message.contains("identity changed")
        ));
        std::fs::remove_file(path).unwrap();
    }

    #[test]
    fn bounded_inspector_accepts_exact_sparse_artifact_and_rejects_marker_corruption() {
        let path = sparse_published_fixture();
        let inspection = inspect_h3_qwen_nvfp4_awq_header(&path).unwrap();
        assert_eq!(inspection.tensor_count, H3_QWEN_NVFP4_AWQ_TENSOR_COUNT);
        assert_eq!(inspection.nvfp4_linear_count, H3_QWEN_NVFP4_LINEAR_COUNT);
        assert_eq!(inspection.awq_pre_quant_scale_count, 100);
        assert_eq!(inspection.int8_tensorwise_count, 1);
        assert_eq!(inspection.header_identity_sha256.len(), 64);
        assert_eq!(
            inspection.expected_artifact_sha256,
            H3_QWEN_NVFP4_AWQ_SHA256
        );

        let header = inspect_safetensors_header(&path).unwrap();
        let marker = &header.tensors["model.layers.0.mlp.down_proj.comfy_quant"];
        let mut file = OpenOptions::new().write(true).open(&path).unwrap();
        file.seek(SeekFrom::Start(
            8 + header.header_len + marker.data_offsets[0],
        ))
        .unwrap();
        file.write_all(br#"{"format": "nvfp5", "full_precision_matrix_mult": true}"#)
            .unwrap();
        file.sync_all().unwrap();
        assert!(matches!(
            inspect_h3_qwen_nvfp4_awq_header(&path),
            Err(H3QwenNvfp4AwqError::Metadata(_))
        ));
        std::fs::remove_file(path).unwrap();

        let metadata_path = sparse_published_fixture_with_metadata(true);
        assert!(matches!(
            inspect_h3_qwen_nvfp4_awq_header(&metadata_path),
            Err(H3QwenNvfp4AwqError::Metadata(_))
        ));
        std::fs::remove_file(metadata_path).unwrap();
    }

    #[test]
    #[ignore = "requires the authorized, separately authenticated private H3 artifact"]
    fn authorized_published_artifact_matches_bounded_contract() {
        let path = std::env::var_os("MOLD_H3_QWEN_NVFP4_PATH")
            .map(PathBuf::from)
            .expect("set MOLD_H3_QWEN_NVFP4_PATH to the private pinned artifact");
        let inspection = inspect_h3_qwen_nvfp4_awq_header(&path).unwrap();
        assert_eq!(inspection.artifact_file_bytes, H3_QWEN_NVFP4_AWQ_FILE_BYTES);
        assert_eq!(inspection.header_bytes, H3_QWEN_NVFP4_AWQ_HEADER_BYTES);
        assert_eq!(
            inspection.tensor_payload_bytes,
            H3_QWEN_NVFP4_AWQ_PAYLOAD_BYTES
        );
        assert_eq!(inspection.tensor_count, H3_QWEN_NVFP4_AWQ_TENSOR_COUNT);
        assert_eq!(inspection.nvfp4_linear_count, H3_QWEN_NVFP4_LINEAR_COUNT);
        assert_eq!(
            inspection.awq_pre_quant_scale_count,
            H3_QWEN_NVFP4_PRE_QUANT_SCALE_COUNT
        );
        assert_eq!(inspection.int8_tensorwise_count, 1);
    }
}
