//! Internal runtime for the authenticated H3 Qwen NVFP4-AWQ artifact.
//!
//! This module deliberately has no engine, factory, catalog, download, or
//! capability registration. The caller must cross the external authorization
//! boundary before opening the private object. One retained regular-file
//! descriptor supplies schema inspection, complete SHA-256 authentication,
//! and every tensor payload read.

use candle::{DType, Device, Tensor};
use candle_nn::VarBuilder;
use sha2::{Digest, Sha256};
use std::collections::HashMap;
use std::path::Path;
use std::sync::{Arc, Mutex};
use thiserror::Error;

use super::comfy_quant::{
    h3_nvfp4_weight_staging_bytes, H3ComfyInt8TensorwiseEmbedding, H3ComfyNvfp4AwqLinear,
    H3Nvfp4LinearKind,
};
use super::config::{H3ConditionerConfig, H3_SELECTED_LANGUAGE_LAYERS};
use super::model::{
    ConditionerCheckpoint, H3QwenNvfp4Layer50Conditioner, H3QwenNvfp4StreamingLayer50Conditioner,
};
use super::qwen_nvfp4::{
    expected_h3_qwen_nvfp4_awq_schema, open_h3_qwen_nvfp4_awq_artifact, released_config,
    validate_h3_qwen_nvfp4_awq_schema, H3QwenNvfp4AwqError, H3QwenNvfp4AwqExecution,
    H3QwenNvfp4AwqMemoryAccounting, H3SafetensorsTensorHeader, OpenedH3QwenNvfp4AwqArtifact,
    H3_QWEN_NVFP4_AWQ_FILE_BYTES, H3_QWEN_NVFP4_AWQ_HEADER_BYTES, H3_QWEN_NVFP4_AWQ_PAYLOAD_BYTES,
};
use super::qwen_quant::H3QwenTensorDType;
use super::text::{Qwen3VlNvfp4LayerLoader, Qwen3VlNvfp4LayerWeights, Qwen3VlNvfp4Weights};

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum H3QwenNvfp4LoadEvent {
    Authenticating {
        completed_bytes: u64,
        total_bytes: u64,
    },
    LoadingTensor {
        name: String,
        tensor_index: usize,
        tensor_count: usize,
        completed_bytes: u64,
        total_bytes: u64,
    },
}

pub trait H3QwenNvfp4LoadObserver {
    /// Return true to cancel at the next bounded read checkpoint.
    fn should_cancel(&mut self, event: &H3QwenNvfp4LoadEvent) -> bool;
}

#[derive(Default)]
pub struct NoopH3QwenNvfp4LoadObserver;

impl H3QwenNvfp4LoadObserver for NoopH3QwenNvfp4LoadObserver {
    fn should_cancel(&mut self, _event: &H3QwenNvfp4LoadEvent) -> bool {
        false
    }
}

#[derive(Debug, Error)]
pub enum H3QwenNvfp4RuntimeError {
    #[error(transparent)]
    Artifact(#[from] H3QwenNvfp4AwqError),
    #[error("invalid H3 Qwen NVFP4 runtime contract: {0}")]
    Contract(String),
    #[error("failed to construct H3 Qwen NVFP4 runtime: {0}")]
    Candle(#[from] candle::Error),
}

/// Exact storage and bounded-I/O facts for the released layer-50 runtime.
///
/// `effective_parameter_bytes` is the source-encoded sum of every tensor the
/// runtime loads; marker payloads are authenticated policy metadata and are
/// not parameters. Host/device residency describes the actual retained
/// representation after the per-tensor read buffer has been dropped. Since
/// #1316 the portable NVFP4 block scales are unswizzled at their authenticated
/// FP8 byte width and widened through a 256-entry table per lookup, so every
/// quantized component is retained at exactly its source width and the total
/// retained bytes equal `effective_parameter_bytes`. Inspection,
/// authentication, tensor, and aggregate I/O are exact byte counts;
/// authentication scratch and the maximum one-at-a-time source tensor buffer
/// are exact loader bounds. Allocator bookkeeping, transient quantization
/// conversion vectors, and backend workspaces are not measured by this
/// structure.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct H3QwenNvfp4RuntimeMemoryFacts {
    pub effective_parameter_bytes: u64,
    pub host_resident_parameter_bytes: u64,
    pub device_resident_parameter_bytes: u64,
    pub retained_parameter_bytes: u64,
    pub retained_raw_header_bytes: u64,
    pub maximum_tensor_staging_bytes: u64,
    pub authentication_scratch_bytes: u64,
    pub bounded_inspection_io_bytes: u64,
    pub authentication_io_bytes: u64,
    pub tensor_io_bytes: u64,
    pub aggregate_io_bytes: u64,
}

/// Device class used to derive the released runtime's retained parameter
/// representation without constructing a backend device. The accelerated
/// route retains portable quantized storage on the host and dense BF16
/// tensors on the selected device; the CPU route retains both on the host.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum H3QwenNvfp4RuntimePlacement {
    Accelerated,
    /// Metal retains the embedding and dense vision/norm tensors, then reads
    /// exactly one language layer's seven NVFP4 projections at a time.
    MetalStreamed,
    Cpu,
}

/// Fully authenticated, still-open Qwen authority established before any
/// model tensor is allocated.
///
/// This token is intentionally non-Clone and consumed by the loader. It binds
/// the retained descriptor, released schema/policy/content identities,
/// concrete host/device residency route, and every exact loader memory fact.
pub struct H3AuthenticatedQwenNvfp4Authority {
    artifact: OpenedH3QwenNvfp4AwqArtifact,
    placement: H3QwenNvfp4RuntimePlacement,
    memory_facts: H3QwenNvfp4RuntimeMemoryFacts,
    identity_sha256: String,
}

impl H3AuthenticatedQwenNvfp4Authority {
    pub const fn placement(&self) -> H3QwenNvfp4RuntimePlacement {
        self.placement
    }

    pub const fn memory_facts(&self) -> &H3QwenNvfp4RuntimeMemoryFacts {
        &self.memory_facts
    }

    pub fn artifact_identity_sha256(&self) -> &str {
        &self.artifact.inspection().expected_artifact_sha256
    }

    pub fn header_identity_sha256(&self) -> &str {
        &self.artifact.inspection().header_identity_sha256
    }

    pub fn policy_identity_sha256(&self) -> &str {
        &self.artifact.inspection().policy_sha256
    }

    pub fn artifact_file_bytes(&self) -> u64 {
        self.artifact.inspection().artifact_file_bytes
    }

    pub fn source_path(&self) -> &Path {
        self.artifact.path()
    }

    pub fn identity_sha256(&self) -> &str {
        &self.identity_sha256
    }

    pub fn revalidate(&self) -> Result<(), H3QwenNvfp4RuntimeError> {
        self.artifact
            .revalidate("while retaining authenticated H3 Qwen authority")?;
        if self.identity_sha256 != authenticated_qwen_authority_identity(self) {
            return Err(H3QwenNvfp4RuntimeError::Contract(
                "authenticated Qwen authority identity changed".into(),
            ));
        }
        Ok(())
    }
}

fn authenticated_qwen_authority_identity(authority: &H3AuthenticatedQwenNvfp4Authority) -> String {
    let mut digest = Sha256::new();
    digest.update(b"mold.minimax-h3.qwen-authenticated-open.v1\0");
    digest.update(match authority.placement {
        H3QwenNvfp4RuntimePlacement::Accelerated => b"accelerated".as_slice(),
        H3QwenNvfp4RuntimePlacement::MetalStreamed => b"metal-streamed".as_slice(),
        H3QwenNvfp4RuntimePlacement::Cpu => b"cpu".as_slice(),
    });
    digest.update(authority.artifact_identity_sha256().as_bytes());
    digest.update(authority.header_identity_sha256().as_bytes());
    digest.update(authority.policy_identity_sha256().as_bytes());
    let memory = &authority.memory_facts;
    for value in [
        memory.effective_parameter_bytes,
        memory.host_resident_parameter_bytes,
        memory.device_resident_parameter_bytes,
        memory.retained_parameter_bytes,
        memory.retained_raw_header_bytes,
        memory.maximum_tensor_staging_bytes,
        memory.authentication_scratch_bytes,
        memory.bounded_inspection_io_bytes,
        memory.authentication_io_bytes,
        memory.tensor_io_bytes,
        memory.aggregate_io_bytes,
    ] {
        digest.update(value.to_le_bytes());
    }
    format!("{:x}", digest.finalize())
}

/// Exact BF16 output-copy charge for the released `[1, sequence, hidden]`
/// conditioner states. The width comes from the same validated runtime config
/// used to construct the model, so admission tests cannot drift independently.
pub fn released_h3_qwen_nvfp4_output_tensor_bytes(
    sequence: u64,
) -> Result<u64, H3QwenNvfp4RuntimeError> {
    let config = released_config()?;
    let width = u64::try_from(config.text_config.hidden_size)
        .map_err(|_| H3QwenNvfp4RuntimeError::Contract("output width overflows u64".to_string()))?;
    let dtype_bytes = u64::try_from(DType::BF16.size_in_bytes()).map_err(|_| {
        H3QwenNvfp4RuntimeError::Contract("output dtype width overflows u64".to_string())
    })?;
    sequence
        .checked_mul(width)
        .and_then(|elements| elements.checked_mul(dtype_bytes))
        .ok_or_else(|| {
            H3QwenNvfp4RuntimeError::Contract("output tensor bytes overflow".to_string())
        })
}

/// Derive the released runtime's exact memory facts without opening an
/// artifact. This is safe to call at an authorization boundary before a
/// loader callback is allowed to inspect or allocate checkpoint storage.
pub fn released_h3_qwen_nvfp4_runtime_memory_facts(
    device: &Device,
) -> Result<H3QwenNvfp4RuntimeMemoryFacts, H3QwenNvfp4RuntimeError> {
    released_h3_qwen_nvfp4_runtime_memory_facts_for_placement(if device.is_cpu() {
        H3QwenNvfp4RuntimePlacement::Cpu
    } else if device.is_metal() {
        H3QwenNvfp4RuntimePlacement::MetalStreamed
    } else {
        H3QwenNvfp4RuntimePlacement::Accelerated
    })
}

/// Widest per-output-row-chunk weight-staging charge any one released Qwen
/// NVFP4 projection incurs, taken over BOTH dequantization arms.
///
/// A memory authority must charge the larger arm, because which one runs is a
/// property of the device the conditioner forward lands on and not of the
/// plan: the portable host loop uploads a dense `f32` row block, while CUDA's
/// device arm (#1317) uploads packed bytes and widens them through
/// `index_select` lookup tables, staging more device scratch for far less
/// traffic. `h3_nvfp4_weight_staging_bytes` owns the arithmetic; this walks
/// the released schema's own packed-weight shapes so it can never describe a
/// projection the artifact does not contain.
///
/// This is a per-linear transient, not a retained parameter, so it is
/// deliberately NOT a field of [`H3QwenNvfp4RuntimeMemoryFacts`]: those facts
/// are hashed into the authenticated-authority identity and compared field for
/// field by the private loader, and `H3QwenNvfp4RuntimeMemoryFacts`'s own
/// contract already excludes backend workspaces. The conditioner's transient
/// device demand is charged instead by the measured
/// `qwen_activation_workspace_bytes` grant, whose FL2VA observation
/// (4,168,069,120 bytes) bounds the whole `QwenEncode` phase; the released
/// figure below is 2.37% of it, so the device arm is absorbed with no change
/// to any `H3PrivateRuntimeBoundRecord` observation.
pub fn released_h3_qwen_nvfp4_max_weight_staging_bytes(
    rows_per_chunk: usize,
) -> Result<u64, H3QwenNvfp4RuntimeError> {
    let config = released_config()?;
    let (tensors, markers) = expected_h3_qwen_nvfp4_awq_schema(&config)?;
    validate_h3_qwen_nvfp4_awq_schema(&config, &tensors, &markers)?;
    let mut widest = 0u64;
    for (name, spec) in &tensors {
        // The packed NVFP4 projections: `U8`, rank two, and not the INT8
        // tensorwise embedding, which shares the dtype but not the arm.
        if spec.dtype != H3QwenTensorDType::U8
            || !name.ends_with(".weight")
            || name == "model.embed_tokens.weight"
            || spec.shape.len() != 2
        {
            continue;
        }
        let out_features = spec.shape[0];
        let in_features = spec.shape[1].checked_mul(2).ok_or_else(|| {
            H3QwenNvfp4RuntimeError::Contract(format!("Qwen packed width overflows for {name:?}"))
        })?;
        for kind in [
            H3Nvfp4LinearKind::PortableHostDequantize,
            H3Nvfp4LinearKind::DeviceLookupDequantize,
        ] {
            let bytes =
                h3_nvfp4_weight_staging_bytes(kind, out_features, in_features, rows_per_chunk)
                    .map_err(H3QwenNvfp4RuntimeError::Candle)?;
            widest = widest.max(u64::try_from(bytes).map_err(|_| {
                H3QwenNvfp4RuntimeError::Contract("Qwen staging bytes overflow u64".into())
            })?);
        }
    }
    if widest == 0 {
        return Err(H3QwenNvfp4RuntimeError::Contract(
            "the released Qwen schema contains no packed NVFP4 projection".into(),
        ));
    }
    Ok(widest)
}

/// Derive the exact released facts for admission before a concrete CUDA
/// device is constructed. This is the same authority used by the loader's
/// device-taking entrypoint above.
pub fn released_h3_qwen_nvfp4_runtime_memory_facts_for_placement(
    placement: H3QwenNvfp4RuntimePlacement,
) -> Result<H3QwenNvfp4RuntimeMemoryFacts, H3QwenNvfp4RuntimeError> {
    let config = released_config()?;
    let (tensors, markers) = expected_h3_qwen_nvfp4_awq_schema(&config)?;
    let policy = validate_h3_qwen_nvfp4_awq_schema(&config, &tensors, &markers)?;
    runtime_memory_facts(policy.memory(), tensors.iter(), placement)
}

fn runtime_memory_facts<'a>(
    memory: &H3QwenNvfp4AwqMemoryAccounting,
    tensors: impl Iterator<Item = (&'a String, &'a super::qwen_quant::H3QwenTensorSpec)>,
    placement: H3QwenNvfp4RuntimePlacement,
) -> Result<H3QwenNvfp4RuntimeMemoryFacts, H3QwenNvfp4RuntimeError> {
    let tensors = tensors.collect::<Vec<_>>();
    let effective_parameter_bytes = memory
        .tensor_payload_bytes
        .checked_sub(memory.quant_marker_bytes)
        .ok_or_else(|| {
            H3QwenNvfp4RuntimeError::Contract(
                "Qwen quantization marker bytes exceed the authenticated payload".into(),
            )
        })?;
    // The block scales are retained as the checkpoint's own FP8 bytes, not a
    // widened `f32` cache (#1316), so they are charged at source width like
    // every other quantized component.
    let quantized_host_bytes = [
        memory.nvfp4_packed_weight_bytes,
        memory.nvfp4_block_scale_bytes,
        memory.nvfp4_tensor_scale_bytes,
        memory.awq_pre_quant_scale_bytes,
        memory.int8_embedding_weight_bytes,
        memory.int8_embedding_scale_bytes,
    ]
    .into_iter()
    .try_fold(0_u64, |total, bytes| total.checked_add(bytes))
    .ok_or_else(|| {
        H3QwenNvfp4RuntimeError::Contract("Qwen host parameter bytes overflow".into())
    })?;
    let (host_resident_parameter_bytes, device_resident_parameter_bytes) = match placement {
        H3QwenNvfp4RuntimePlacement::Cpu => (
            quantized_host_bytes
                .checked_add(memory.dense_bf16_bytes)
                .ok_or_else(|| {
                    H3QwenNvfp4RuntimeError::Contract(
                        "Qwen CPU retained parameter bytes overflow".into(),
                    )
                })?,
            0,
        ),
        H3QwenNvfp4RuntimePlacement::Accelerated => (quantized_host_bytes, memory.dense_bf16_bytes),
        H3QwenNvfp4RuntimePlacement::MetalStreamed => {
            // Exact peak: the INT8 embedding remains live while one layer's
            // seven NVFP4 projections are loaded from the retained descriptor.
            // Dense vision, rotary and all language RMS norms stay on Metal.
            let encoded_bytes = |name: &str,
                                 spec: &super::qwen_quant::H3QwenTensorSpec|
             -> Result<u64, H3QwenNvfp4RuntimeError> {
                spec.shape
                    .iter()
                    .try_fold(1_u64, |elements, dimension| {
                        elements.checked_mul(*dimension as u64)
                    })
                    .and_then(|elements| elements.checked_mul(spec.dtype.byte_width()))
                    .ok_or_else(|| {
                        H3QwenNvfp4RuntimeError::Contract(format!(
                            "Qwen streamed tensor {name:?} byte count overflows"
                        ))
                    })
            };
            let embedding_bytes = tensors
                .iter()
                .filter(|(name, _)| {
                    name.starts_with("model.embed_tokens.") && !name.ends_with(".comfy_quant")
                })
                .try_fold(0_u64, |total, (name, spec)| {
                    total
                        .checked_add(encoded_bytes(name, spec)?)
                        .ok_or_else(|| {
                            H3QwenNvfp4RuntimeError::Contract(
                                "Qwen streamed embedding bytes overflow".into(),
                            )
                        })
                })?;
            let layer_zero_bytes = tensors
                .iter()
                .filter(|(name, _)| {
                    name.starts_with("model.layers.0.")
                        && !name.ends_with(".comfy_quant")
                        && [
                            "self_attn.q_proj",
                            "self_attn.k_proj",
                            "self_attn.v_proj",
                            "self_attn.o_proj",
                            "mlp.gate_proj",
                            "mlp.up_proj",
                            "mlp.down_proj",
                        ]
                        .iter()
                        .any(|projection| name.contains(projection))
                })
                .try_fold(0_u64, |total, (name, spec)| {
                    total
                        .checked_add(encoded_bytes(name, spec)?)
                        .ok_or_else(|| {
                            H3QwenNvfp4RuntimeError::Contract(
                                "Qwen streamed layer bytes overflow".into(),
                            )
                        })
                })?;
            let streamed_host_peak =
                embedding_bytes
                    .checked_add(layer_zero_bytes)
                    .ok_or_else(|| {
                        H3QwenNvfp4RuntimeError::Contract(
                            "Qwen streamed host peak overflows".into(),
                        )
                    })?;
            if embedding_bytes != 778_520_064
                || layer_zero_bytes != 274_335_772
                || streamed_host_peak != 1_052_855_836
            {
                return Err(H3QwenNvfp4RuntimeError::Contract(format!(
                    "released Metal streaming geometry changed: embedding/layer/peak {embedding_bytes}/{layer_zero_bytes}/{streamed_host_peak}"
                )));
            }
            (streamed_host_peak, memory.dense_bf16_bytes)
        }
    };
    let retained_parameter_bytes = host_resident_parameter_bytes
        .checked_add(device_resident_parameter_bytes)
        .ok_or_else(|| {
            H3QwenNvfp4RuntimeError::Contract("Qwen retained parameter bytes overflow".into())
        })?;
    let maximum_tensor_staging_bytes = tensors
        .iter()
        .copied()
        .filter(|(name, _)| !name.ends_with(".comfy_quant"))
        .map(|(_, spec)| {
            spec.shape
                .iter()
                .try_fold(1_u64, |elements, dimension| {
                    elements.checked_mul(*dimension as u64)
                })
                .and_then(|elements| elements.checked_mul(spec.dtype.byte_width()))
                .ok_or_else(|| {
                    H3QwenNvfp4RuntimeError::Contract(
                        "Qwen tensor staging byte count overflows".into(),
                    )
                })
        })
        .collect::<Result<Vec<_>, _>>()?
        .into_iter()
        .max()
        .unwrap_or_default();
    let retained_raw_header_bytes = H3_QWEN_NVFP4_AWQ_HEADER_BYTES
        .checked_add(8)
        .expect("published H3 Qwen header length is bounded");
    let bounded_inspection_io_bytes = retained_raw_header_bytes
        .checked_add(memory.quant_marker_bytes)
        .ok_or_else(|| {
            H3QwenNvfp4RuntimeError::Contract("Qwen inspection I/O bytes overflow".into())
        })?;
    let aggregate_io_bytes = bounded_inspection_io_bytes
        .checked_add(H3_QWEN_NVFP4_AWQ_FILE_BYTES)
        .and_then(|bytes| bytes.checked_add(effective_parameter_bytes))
        .ok_or_else(|| {
            H3QwenNvfp4RuntimeError::Contract("Qwen aggregate I/O bytes overflow".into())
        })?;
    Ok(H3QwenNvfp4RuntimeMemoryFacts {
        effective_parameter_bytes,
        host_resident_parameter_bytes,
        device_resident_parameter_bytes,
        retained_parameter_bytes,
        retained_raw_header_bytes,
        maximum_tensor_staging_bytes,
        authentication_scratch_bytes: 1024 * 1024,
        bounded_inspection_io_bytes,
        authentication_io_bytes: H3_QWEN_NVFP4_AWQ_FILE_BYTES,
        tensor_io_bytes: effective_parameter_bytes,
        aggregate_io_bytes,
    })
}

enum LoadedQwenModel {
    Resident(H3QwenNvfp4Layer50Conditioner),
    MetalStreamed(H3QwenNvfp4StreamingLayer50Conditioner),
}

struct StreamingArtifactState {
    artifact: OpenedH3QwenNvfp4AwqArtifact,
    tensor_index: usize,
    tensor_count: usize,
    completed_before: u64,
    total_bytes: u64,
    next_layer: usize,
}

struct ConditionerLoadHeartbeat<'a> {
    checkpoint: &'a mut dyn FnMut(ConditionerCheckpoint) -> candle::Result<()>,
}

impl H3QwenNvfp4LoadObserver for ConditionerLoadHeartbeat<'_> {
    fn should_cancel(&mut self, _event: &H3QwenNvfp4LoadEvent) -> bool {
        (self.checkpoint)(ConditionerCheckpoint::LanguageLayerLoadHeartbeat).is_err()
    }
}

impl Qwen3VlNvfp4LayerLoader for StreamingLayerLoader {
    fn load_layer(
        &self,
        index: usize,
        checkpoint: &mut dyn FnMut(ConditionerCheckpoint) -> candle::Result<()>,
    ) -> candle::Result<Qwen3VlNvfp4LayerWeights> {
        let mut state = self
            .state
            .lock()
            .map_err(|_| candle::Error::Msg("streamed Qwen artifact lock poisoned".into()))?;
        if index != state.next_layer {
            candle::bail!(
                "H3 streamed Qwen requested language layer {index}, expected exactly {}",
                state.next_layer
            )
        }
        state
            .artifact
            .revalidate("before streaming H3 Qwen language layer")
            .map_err(|error| candle::Error::Msg(error.to_string()))?;
        let mut heartbeat = ConditionerLoadHeartbeat { checkpoint };
        let mut progress = TensorLoadProgress {
            observer: &mut heartbeat,
            tensor_index: state.tensor_index,
            tensor_count: state.tensor_count,
            completed_before: state.completed_before,
            total_bytes: state.total_bytes,
        };
        let weights = Qwen3VlNvfp4LayerWeights {
            q_proj: load_linear(
                &mut state.artifact,
                &self.config,
                index,
                "self_attn.q_proj",
                &mut progress,
            )
            .map_err(|error| candle::Error::Msg(error.to_string()))?,
            k_proj: load_linear(
                &mut state.artifact,
                &self.config,
                index,
                "self_attn.k_proj",
                &mut progress,
            )
            .map_err(|error| candle::Error::Msg(error.to_string()))?,
            v_proj: load_linear(
                &mut state.artifact,
                &self.config,
                index,
                "self_attn.v_proj",
                &mut progress,
            )
            .map_err(|error| candle::Error::Msg(error.to_string()))?,
            o_proj: load_linear(
                &mut state.artifact,
                &self.config,
                index,
                "self_attn.o_proj",
                &mut progress,
            )
            .map_err(|error| candle::Error::Msg(error.to_string()))?,
            gate_proj: load_linear(
                &mut state.artifact,
                &self.config,
                index,
                "mlp.gate_proj",
                &mut progress,
            )
            .map_err(|error| candle::Error::Msg(error.to_string()))?,
            up_proj: load_linear(
                &mut state.artifact,
                &self.config,
                index,
                "mlp.up_proj",
                &mut progress,
            )
            .map_err(|error| candle::Error::Msg(error.to_string()))?,
            down_proj: load_linear(
                &mut state.artifact,
                &self.config,
                index,
                "mlp.down_proj",
                &mut progress,
            )
            .map_err(|error| candle::Error::Msg(error.to_string()))?,
        };
        state.tensor_index = progress.tensor_index;
        state.completed_before = progress.completed_before;
        state.next_layer += 1;
        state
            .artifact
            .revalidate("after streaming H3 Qwen language layer")
            .map_err(|error| candle::Error::Msg(error.to_string()))?;
        if state.next_layer == H3_SELECTED_LANGUAGE_LAYERS
            && (state.tensor_index - 1 != state.tensor_count
                || state.completed_before != state.total_bytes)
        {
            candle::bail!(
                "streamed runtime read {} of {} tensors and {} of {} bytes",
                state.tensor_index - 1,
                state.tensor_count,
                state.completed_before,
                state.total_bytes
            )
        }
        Ok(weights)
    }
}

#[derive(Clone)]
struct StreamingLayerLoader {
    state: Arc<Mutex<StreamingArtifactState>>,
    config: H3ConditionerConfig,
}

enum LoadedQwenArtifact {
    Resident(Box<OpenedH3QwenNvfp4AwqArtifact>),
    MetalStreamed(Arc<Mutex<StreamingArtifactState>>),
}

pub struct LoadedH3QwenNvfp4Conditioner {
    model: LoadedQwenModel,
    artifact: LoadedQwenArtifact,
    artifact_identity: String,
    header_identity: String,
    policy_identity: String,
    device: Device,
    memory_facts: H3QwenNvfp4RuntimeMemoryFacts,
}

impl LoadedH3QwenNvfp4Conditioner {
    pub fn encode(
        &self,
        input: &super::model::H3ConditionerInput,
        checkpoint: &mut dyn FnMut(super::model::ConditionerCheckpoint) -> candle::Result<()>,
    ) -> candle::Result<Tensor> {
        match &self.model {
            LoadedQwenModel::Resident(model) => model.encode(input, checkpoint),
            LoadedQwenModel::MetalStreamed(model) => model.encode(input, checkpoint),
        }
    }

    pub fn dtype_profile(&self) -> super::model::H3DTypeProfile {
        match &self.model {
            LoadedQwenModel::Resident(model) => model.dtype_profile(),
            LoadedQwenModel::MetalStreamed(model) => model.dtype_profile(),
        }
    }

    pub fn resident_language_layers(&self) -> usize {
        match &self.model {
            LoadedQwenModel::Resident(model) => model.resident_language_layers(),
            LoadedQwenModel::MetalStreamed(model) => model.language_layer_count(),
        }
    }

    pub fn peak_resident_language_layers(&self) -> usize {
        match &self.model {
            LoadedQwenModel::Resident(model) => model.resident_language_layers(),
            LoadedQwenModel::MetalStreamed(model) => model.peak_resident_language_layers(),
        }
    }

    pub const fn memory_facts(&self) -> &H3QwenNvfp4RuntimeMemoryFacts {
        &self.memory_facts
    }

    /// Device selected when the opened-file-bound runtime was constructed.
    ///
    /// NVFP4 payload storage may stage on the host, but every dense tensor and
    /// execution transfer is bound to this exact device authority.
    pub fn device(&self) -> &Device {
        &self.device
    }

    /// Full authenticated checkpoint identity retained by this runtime.
    pub fn artifact_identity_sha256(&self) -> &str {
        match &self.artifact {
            LoadedQwenArtifact::Resident(_) | LoadedQwenArtifact::MetalStreamed(_) => {
                self.artifact_identity.as_str()
            }
        }
    }

    /// Parsed safetensors-header identity used to reject cross-checkpoint
    /// pairing even when a caller supplies the same display name.
    pub fn header_identity_sha256(&self) -> &str {
        self.header_identity.as_str()
    }

    /// Exact NVFP4-AWQ execution-policy identity selected from the opened
    /// checkpoint rather than from a filename or caller assertion.
    pub fn policy_identity_sha256(&self) -> &str {
        self.policy_identity.as_str()
    }

    /// Revalidate the retained descriptor and its path identity before an
    /// inference callback reads the already-constructed tensor storage.
    pub fn revalidate_artifact(&self) -> Result<(), H3QwenNvfp4RuntimeError> {
        match &self.artifact {
            LoadedQwenArtifact::Resident(artifact) => {
                artifact.revalidate("while retaining loaded H3 Qwen runtime")?
            }
            LoadedQwenArtifact::MetalStreamed(state) => match state.try_lock() {
                Ok(state) => state
                    .artifact
                    .revalidate("while retaining streamed H3 Qwen runtime")?,
                // The layer reader owns this same singular state while its
                // 1 MiB heartbeat validates the outer attempt authorities.
                // It revalidates the descriptor immediately before and after
                // the layer; recursively locking here would deadlock the
                // heartbeat on the owner thread.
                Err(std::sync::TryLockError::WouldBlock) => {}
                Err(std::sync::TryLockError::Poisoned(_)) => {
                    return Err(H3QwenNvfp4RuntimeError::Contract(
                        "streamed Qwen artifact lock poisoned".into(),
                    ));
                }
            },
        }
        Ok(())
    }
}

struct TensorLoadProgress<'a> {
    observer: &'a mut dyn H3QwenNvfp4LoadObserver,
    tensor_index: usize,
    tensor_count: usize,
    completed_before: u64,
    total_bytes: u64,
}

impl TensorLoadProgress<'_> {
    fn read(
        &mut self,
        artifact: &mut OpenedH3QwenNvfp4AwqArtifact,
        name: &str,
    ) -> Result<Vec<u8>, H3QwenNvfp4RuntimeError> {
        let event_name = name.to_string();
        let observer = &mut self.observer;
        let tensor_index = self.tensor_index;
        let tensor_count = self.tensor_count;
        let completed_before = self.completed_before;
        let total_bytes = self.total_bytes;
        let bytes = artifact.read_tensor_bytes(name, &mut |completed, _| {
            let event = H3QwenNvfp4LoadEvent::LoadingTensor {
                name: event_name.clone(),
                tensor_index,
                tensor_count,
                completed_bytes: completed_before.saturating_add(completed),
                total_bytes,
            };
            if observer.should_cancel(&event) {
                return Err(H3QwenNvfp4AwqError::Io(
                    "H3 Qwen NVFP4 load cancelled".into(),
                ));
            }
            Ok(())
        })?;
        self.completed_before = self
            .completed_before
            .checked_add(bytes.len() as u64)
            .ok_or_else(|| {
                H3QwenNvfp4RuntimeError::Contract("loaded byte count overflows".into())
            })?;
        self.tensor_index += 1;
        Ok(bytes)
    }
}

/// Open and completely authenticate the private Qwen object without
/// allocating any model tensors.
///
/// # Safety
///
/// Before calling, the consumer must validate one frozen factory authority,
/// active conditioner/execution/artifact leases, the exact support identity,
/// route, and released quantization policy. Those authorities must be polled
/// by `observer` at every authentication checkpoint. This is an unsafe
/// cross-crate implementation seam, not an independently callable artifact
/// opener; the safe authority-first entrypoint lives in mold-inference.
pub unsafe fn open_h3_qwen_nvfp4_authority_after_authorization(
    path: &Path,
    placement: H3QwenNvfp4RuntimePlacement,
    observer: &mut dyn H3QwenNvfp4LoadObserver,
) -> Result<H3AuthenticatedQwenNvfp4Authority, H3QwenNvfp4RuntimeError> {
    let expected_memory_facts =
        released_h3_qwen_nvfp4_runtime_memory_facts_for_placement(placement)?;
    let mut artifact = open_h3_qwen_nvfp4_awq_artifact(path)?;
    artifact.authenticate_full_sha256(&mut |completed_bytes, total_bytes| {
        let event = H3QwenNvfp4LoadEvent::Authenticating {
            completed_bytes,
            total_bytes,
        };
        if observer.should_cancel(&event) {
            return Err(H3QwenNvfp4AwqError::Io(
                "H3 Qwen NVFP4 authentication cancelled".into(),
            ));
        }
        Ok(())
    })?;

    let loadable_names = artifact
        .tensors()
        .iter()
        .filter(|(name, _)| !name.ends_with(".comfy_quant"))
        .map(|(name, _)| name.clone())
        .collect::<Vec<_>>();
    let loadable_bytes = loadable_names.iter().try_fold(0_u64, |total, name| {
        let header = &artifact.tensors()[name];
        total.checked_add(header.data_offsets[1] - header.data_offsets[0])
    });
    let loadable_bytes = loadable_bytes.ok_or_else(|| {
        H3QwenNvfp4RuntimeError::Contract("loadable tensor bytes overflow".into())
    })?;
    if loadable_bytes > H3_QWEN_NVFP4_AWQ_PAYLOAD_BYTES {
        return Err(H3QwenNvfp4RuntimeError::Contract(
            "loadable tensor bytes exceed authenticated payload".into(),
        ));
    }
    if loadable_bytes != expected_memory_facts.effective_parameter_bytes {
        return Err(H3QwenNvfp4RuntimeError::Contract(format!(
            "authenticated loadable parameter bytes {loadable_bytes} differ from released runtime facts {}",
            expected_memory_facts.effective_parameter_bytes
        )));
    }
    artifact.revalidate("after authenticating H3 Qwen authority")?;
    let mut authority = H3AuthenticatedQwenNvfp4Authority {
        artifact,
        placement,
        memory_facts: expected_memory_facts,
        identity_sha256: String::new(),
    };
    authority.identity_sha256 = authenticated_qwen_authority_identity(&authority);
    authority.revalidate()?;
    Ok(authority)
}

/// Consume one authenticated Qwen authority and allocate the released runtime
/// on its frozen placement route.
///
/// # Safety
///
/// The caller must retain and poll the same complete attempt authority used to
/// open `authority`, and must have validated its exact memory facts at the
/// allocation boundary. This function consumes the non-Clone authority so one
/// authenticated descriptor cannot authorize multiple model allocations.
pub unsafe fn load_h3_qwen_nvfp4_conditioner_from_authority(
    mut authority: H3AuthenticatedQwenNvfp4Authority,
    config: &H3ConditionerConfig,
    device: &Device,
    observer: &mut dyn H3QwenNvfp4LoadObserver,
) -> Result<LoadedH3QwenNvfp4Conditioner, H3QwenNvfp4RuntimeError> {
    config
        .validate()
        .map_err(|error| H3QwenNvfp4RuntimeError::Contract(error.to_string()))?;
    if config != &released_config()? {
        return Err(H3QwenNvfp4RuntimeError::Contract(
            "runtime config differs from the frozen published layer-50 Qwen contract".into(),
        ));
    }
    authority.revalidate()?;
    let device_placement = if device.is_cpu() {
        H3QwenNvfp4RuntimePlacement::Cpu
    } else if device.is_metal() {
        H3QwenNvfp4RuntimePlacement::MetalStreamed
    } else {
        H3QwenNvfp4RuntimePlacement::Accelerated
    };
    if authority.placement != device_placement {
        return Err(H3QwenNvfp4RuntimeError::Contract(format!(
            "authenticated Qwen placement {:?} differs from selected device placement {:?}",
            authority.placement, device_placement
        )));
    }
    let expected_memory_facts = released_h3_qwen_nvfp4_runtime_memory_facts(device)?;
    if authority.memory_facts != expected_memory_facts {
        return Err(H3QwenNvfp4RuntimeError::Contract(
            "authenticated Qwen memory facts differ from selected device facts".into(),
        ));
    }
    if device_placement == H3QwenNvfp4RuntimePlacement::MetalStreamed {
        return load_metal_streamed_conditioner(authority, config, device, observer);
    }
    let loadable_names = authority
        .artifact
        .tensors()
        .iter()
        .filter(|(name, _)| !name.ends_with(".comfy_quant"))
        .map(|(name, _)| name.clone())
        .collect::<Vec<_>>();
    let loadable_bytes = expected_memory_facts.effective_parameter_bytes;
    let mut progress = TensorLoadProgress {
        observer,
        tensor_index: 1,
        tensor_count: loadable_names.len(),
        completed_before: 0,
        total_bytes: loadable_bytes,
    };

    let embed_weight = load_tensor(
        &mut authority.artifact,
        "model.embed_tokens.weight",
        &Device::Cpu,
        &mut progress,
    )?;
    let embed_scale = load_tensor(
        &mut authority.artifact,
        "model.embed_tokens.weight_scale",
        &Device::Cpu,
        &mut progress,
    )?;
    let embed_tokens = H3ComfyInt8TensorwiseEmbedding::new(embed_weight, embed_scale)?;

    let mut layers = Vec::with_capacity(H3_SELECTED_LANGUAGE_LAYERS);
    for layer in 0..H3_SELECTED_LANGUAGE_LAYERS {
        layers.push(Qwen3VlNvfp4LayerWeights {
            q_proj: load_linear(
                &mut authority.artifact,
                config,
                layer,
                "self_attn.q_proj",
                &mut progress,
            )?,
            k_proj: load_linear(
                &mut authority.artifact,
                config,
                layer,
                "self_attn.k_proj",
                &mut progress,
            )?,
            v_proj: load_linear(
                &mut authority.artifact,
                config,
                layer,
                "self_attn.v_proj",
                &mut progress,
            )?,
            o_proj: load_linear(
                &mut authority.artifact,
                config,
                layer,
                "self_attn.o_proj",
                &mut progress,
            )?,
            gate_proj: load_linear(
                &mut authority.artifact,
                config,
                layer,
                "mlp.gate_proj",
                &mut progress,
            )?,
            up_proj: load_linear(
                &mut authority.artifact,
                config,
                layer,
                "mlp.up_proj",
                &mut progress,
            )?,
            down_proj: load_linear(
                &mut authority.artifact,
                config,
                layer,
                "mlp.down_proj",
                &mut progress,
            )?,
        });
    }

    let dense_names = authority
        .artifact
        .tensors()
        .iter()
        .filter(|(name, header)| header.dtype == "BF16" && !name.ends_with(".pre_quant_scale"))
        .map(|(name, _)| name.clone())
        .collect::<Vec<_>>();
    let mut dense = HashMap::with_capacity(dense_names.len());
    for name in dense_names {
        let tensor = load_tensor(&mut authority.artifact, &name, device, &mut progress)?;
        if dense.insert(name.clone(), tensor).is_some() {
            return Err(H3QwenNvfp4RuntimeError::Contract(format!(
                "duplicate dense tensor {name:?}"
            )));
        }
    }
    if progress.tensor_index - 1 != progress.tensor_count
        || progress.completed_before != progress.total_bytes
    {
        return Err(H3QwenNvfp4RuntimeError::Contract(format!(
            "runtime loaded {} of {} tensors and {} of {} bytes",
            progress.tensor_index - 1,
            progress.tensor_count,
            progress.completed_before,
            progress.total_bytes
        )));
    }
    authority
        .artifact
        .revalidate("after constructing H3 Qwen tensor storage")?;
    let builder = VarBuilder::from_tensors(dense, DType::BF16, device);
    let model = H3QwenNvfp4Layer50Conditioner::new(
        config,
        builder,
        Qwen3VlNvfp4Weights {
            embed_tokens,
            layers,
        },
    )?;
    if model.resident_language_layers() != H3_SELECTED_LANGUAGE_LAYERS {
        return Err(H3QwenNvfp4RuntimeError::Contract(format!(
            "runtime constructed {} language layers, expected exactly {H3_SELECTED_LANGUAGE_LAYERS}",
            model.resident_language_layers()
        )));
    }
    authority
        .artifact
        .revalidate("after constructing H3 Qwen layer-50 model")?;
    let artifact_identity = authority
        .artifact
        .inspection()
        .expected_artifact_sha256
        .clone();
    let header_identity = authority
        .artifact
        .inspection()
        .header_identity_sha256
        .clone();
    let policy_identity = authority.artifact.inspection().policy_sha256.clone();
    Ok(LoadedH3QwenNvfp4Conditioner {
        model: LoadedQwenModel::Resident(model),
        artifact: LoadedQwenArtifact::Resident(Box::new(authority.artifact)),
        artifact_identity,
        header_identity,
        policy_identity,
        device: device.clone(),
        memory_facts: expected_memory_facts,
    })
}

fn load_metal_streamed_conditioner(
    mut authority: H3AuthenticatedQwenNvfp4Authority,
    config: &H3ConditionerConfig,
    device: &Device,
    observer: &mut dyn H3QwenNvfp4LoadObserver,
) -> Result<LoadedH3QwenNvfp4Conditioner, H3QwenNvfp4RuntimeError> {
    if !device.is_metal() || authority.placement != H3QwenNvfp4RuntimePlacement::MetalStreamed {
        return Err(H3QwenNvfp4RuntimeError::Contract(
            "streamed Qwen construction requires the frozen Metal placement".into(),
        ));
    }
    let loadable_names = authority
        .artifact
        .tensors()
        .iter()
        .filter(|(name, _)| !name.ends_with(".comfy_quant"))
        .map(|(name, _)| name.clone())
        .collect::<Vec<_>>();
    let mut progress = TensorLoadProgress {
        observer,
        tensor_index: 1,
        tensor_count: loadable_names.len(),
        completed_before: 0,
        total_bytes: authority.memory_facts.effective_parameter_bytes,
    };
    let embed_weight = load_tensor(
        &mut authority.artifact,
        "model.embed_tokens.weight",
        &Device::Cpu,
        &mut progress,
    )?;
    let embed_scale = load_tensor(
        &mut authority.artifact,
        "model.embed_tokens.weight_scale",
        &Device::Cpu,
        &mut progress,
    )?;
    let embed_tokens = H3ComfyInt8TensorwiseEmbedding::new(embed_weight, embed_scale)?;

    let dense_names = authority
        .artifact
        .tensors()
        .iter()
        .filter(|(name, header)| header.dtype == "BF16" && !name.ends_with(".pre_quant_scale"))
        .map(|(name, _)| name.clone())
        .collect::<Vec<_>>();
    let mut dense = HashMap::with_capacity(dense_names.len());
    for name in dense_names {
        let tensor = load_tensor(&mut authority.artifact, &name, device, &mut progress)?;
        if dense.insert(name.clone(), tensor).is_some() {
            return Err(H3QwenNvfp4RuntimeError::Contract(format!(
                "duplicate dense tensor {name:?}"
            )));
        }
    }
    authority
        .artifact
        .revalidate("after constructing streamed H3 Qwen fixed tensor storage")?;
    let artifact_identity = authority
        .artifact
        .inspection()
        .expected_artifact_sha256
        .clone();
    let header_identity = authority
        .artifact
        .inspection()
        .header_identity_sha256
        .clone();
    let policy_identity = authority.artifact.inspection().policy_sha256.clone();
    let memory_facts = authority.memory_facts.clone();
    let state = Arc::new(Mutex::new(StreamingArtifactState {
        artifact: authority.artifact,
        tensor_index: progress.tensor_index,
        tensor_count: progress.tensor_count,
        completed_before: progress.completed_before,
        total_bytes: progress.total_bytes,
        next_layer: 0,
    }));
    let builder = VarBuilder::from_tensors(dense, DType::BF16, device);
    let model = H3QwenNvfp4StreamingLayer50Conditioner::new(
        config,
        builder,
        embed_tokens,
        Box::new(StreamingLayerLoader {
            state: Arc::clone(&state),
            config: config.clone(),
        }),
    )?;
    Ok(LoadedH3QwenNvfp4Conditioner {
        model: LoadedQwenModel::MetalStreamed(model),
        artifact: LoadedQwenArtifact::MetalStreamed(state),
        artifact_identity,
        header_identity,
        policy_identity,
        device: device.clone(),
        memory_facts,
    })
}

/// Compatibility wrapper for callers that do not need to inspect the opened
/// evidence between authentication and allocation.
///
/// # Safety
///
/// The same authority requirements as
/// [`open_h3_qwen_nvfp4_authority_after_authorization`] and
/// [`load_h3_qwen_nvfp4_conditioner_from_authority`] apply.
pub unsafe fn load_h3_qwen_nvfp4_conditioner_after_authorization(
    path: &Path,
    config: &H3ConditionerConfig,
    device: &Device,
    observer: &mut dyn H3QwenNvfp4LoadObserver,
) -> Result<LoadedH3QwenNvfp4Conditioner, H3QwenNvfp4RuntimeError> {
    config
        .validate()
        .map_err(|error| H3QwenNvfp4RuntimeError::Contract(error.to_string()))?;
    if config != &released_config()? {
        return Err(H3QwenNvfp4RuntimeError::Contract(
            "runtime config differs from the frozen published layer-50 Qwen contract".into(),
        ));
    }
    let placement = if device.is_cpu() {
        H3QwenNvfp4RuntimePlacement::Cpu
    } else if device.is_metal() {
        H3QwenNvfp4RuntimePlacement::MetalStreamed
    } else {
        H3QwenNvfp4RuntimePlacement::Accelerated
    };
    let authority =
        unsafe { open_h3_qwen_nvfp4_authority_after_authorization(path, placement, observer)? };
    unsafe { load_h3_qwen_nvfp4_conditioner_from_authority(authority, config, device, observer) }
}

fn load_linear(
    artifact: &mut OpenedH3QwenNvfp4AwqArtifact,
    config: &H3ConditionerConfig,
    layer: usize,
    suffix: &str,
    progress: &mut TensorLoadProgress<'_>,
) -> Result<H3ComfyNvfp4AwqLinear, H3QwenNvfp4RuntimeError> {
    let canonical = format!("model.language_model.layers.{layer}.{suffix}.weight");
    let execution = artifact.policy().execution_for_weight(&canonical)?;
    let H3QwenNvfp4AwqExecution::Nvfp4FullPrecision {
        packed_weight_tensor,
        block_scale_tensor,
        tensor_scale_tensor,
        pre_quant_scale_tensor,
    } = execution
    else {
        return Err(H3QwenNvfp4RuntimeError::Contract(format!(
            "projection {canonical:?} did not resolve to NVFP4 execution"
        )));
    };
    let (out_features, in_features) = linear_dimensions(config, suffix)?;
    let packed_weight = load_tensor(artifact, &packed_weight_tensor, &Device::Cpu, progress)?;
    let block_scales = load_tensor(artifact, &block_scale_tensor, &Device::Cpu, progress)?;
    let tensor_scale = load_tensor(artifact, &tensor_scale_tensor, &Device::Cpu, progress)?;
    let pre_quant_scale = pre_quant_scale_tensor
        .as_deref()
        .map(|name| load_tensor(artifact, name, &Device::Cpu, progress))
        .transpose()?;
    H3ComfyNvfp4AwqLinear::new_with_optional_awq(
        packed_weight,
        block_scales,
        tensor_scale,
        pre_quant_scale,
        out_features,
        in_features,
    )
    .map_err(Into::into)
}

fn linear_dimensions(
    config: &H3ConditionerConfig,
    suffix: &str,
) -> Result<(usize, usize), H3QwenNvfp4RuntimeError> {
    let text = &config.text_config;
    let query_width = text
        .num_attention_heads
        .checked_mul(text.head_dim)
        .ok_or_else(|| H3QwenNvfp4RuntimeError::Contract("query width overflows".into()))?;
    let key_value_width = text
        .num_key_value_heads
        .checked_mul(text.head_dim)
        .ok_or_else(|| H3QwenNvfp4RuntimeError::Contract("key/value width overflows".into()))?;
    match suffix {
        "self_attn.q_proj" => Ok((query_width, text.hidden_size)),
        "self_attn.k_proj" | "self_attn.v_proj" => Ok((key_value_width, text.hidden_size)),
        "self_attn.o_proj" => Ok((text.hidden_size, query_width)),
        "mlp.gate_proj" | "mlp.up_proj" => Ok((text.intermediate_size, text.hidden_size)),
        "mlp.down_proj" => Ok((text.hidden_size, text.intermediate_size)),
        other => Err(H3QwenNvfp4RuntimeError::Contract(format!(
            "unsupported Qwen projection {other:?}"
        ))),
    }
}

fn load_tensor(
    artifact: &mut OpenedH3QwenNvfp4AwqArtifact,
    name: &str,
    device: &Device,
    progress: &mut TensorLoadProgress<'_>,
) -> Result<Tensor, H3QwenNvfp4RuntimeError> {
    let header = artifact.tensors().get(name).cloned().ok_or_else(|| {
        H3QwenNvfp4RuntimeError::Contract(format!("missing authenticated tensor {name:?}"))
    })?;
    let bytes = progress.read(artifact, name)?;
    tensor_from_bytes(&bytes, &header, device).map_err(Into::into)
}

fn tensor_from_bytes(
    bytes: &[u8],
    header: &H3SafetensorsTensorHeader,
    device: &Device,
) -> candle::Result<Tensor> {
    let dtype = match header.dtype.as_str() {
        "BF16" => DType::BF16,
        "F8_E4M3" => DType::F8E4M3,
        "F32" => DType::F32,
        // Candle has no signed I8 storage. Preserve exact two's-complement
        // bytes and let the embedding primitive widen them explicitly.
        "I8" | "U8" => DType::U8,
        other => candle::bail!("unsupported authenticated Qwen dtype {other:?}"),
    };
    Tensor::from_raw_buffer(bytes, dtype, &header.shape, device)
}

const _: () = assert!(H3_SELECTED_LANGUAGE_LAYERS == 50);

#[cfg(test)]
mod tests {
    use super::super::comfy_quant::H3_COMFY_PORTABLE_ROW_CHUNK;
    use super::super::qwen_nvfp4::tests::sparse_published_fixture;
    use super::*;

    #[test]
    fn released_cpu_runtime_facts_separate_source_residency_and_bounded_io() {
        let facts = released_h3_qwen_nvfp4_runtime_memory_facts(&Device::Cpu).unwrap();
        assert_eq!(facts.effective_parameter_bytes, 15_686_891_864);
        assert_eq!(facts.host_resident_parameter_bytes, 15_686_891_864);
        assert_eq!(facts.device_resident_parameter_bytes, 0);
        assert_eq!(facts.retained_parameter_bytes, 15_686_891_864);
        assert_eq!(facts.retained_raw_header_bytes, 231_408);
        assert_eq!(facts.maximum_tensor_staging_bytes, 777_912_320);
        assert_eq!(facts.authentication_scratch_bytes, 1_048_576);
        assert_eq!(facts.bounded_inspection_io_bytes, 250_687);
        assert_eq!(facts.authentication_io_bytes, 15_687_142_551);
        assert_eq!(facts.tensor_io_bytes, 15_686_891_864);
        assert_eq!(facts.aggregate_io_bytes, 31_374_285_102);
        // Every quantized component is now retained at its source width, so the
        // artifact is no longer inflated in RAM at all: retained bytes are
        // identically the source-encoded parameter total. Asserting the
        // equality is what catches a re-widened cache creeping back in.
        assert_eq!(
            facts.retained_parameter_bytes,
            facts.effective_parameter_bytes
        );
    }

    #[test]
    fn released_accelerated_runtime_facts_bind_exact_mixed_residency() {
        let facts = released_h3_qwen_nvfp4_runtime_memory_facts_for_placement(
            H3QwenNvfp4RuntimePlacement::Accelerated,
        )
        .unwrap();
        assert_eq!(facts.effective_parameter_bytes, 15_686_891_864);
        assert_eq!(facts.host_resident_parameter_bytes, 14_495_308_664);
        assert_eq!(facts.device_resident_parameter_bytes, 1_191_583_200);
        assert_eq!(
            released_h3_qwen_nvfp4_output_tensor_bytes(7).unwrap(),
            7 * 5_120 * 2
        );
        assert_eq!(facts.retained_parameter_bytes, 15_686_891_864);
        assert_eq!(
            facts.retained_parameter_bytes,
            facts.effective_parameter_bytes
        );
    }

    #[test]
    fn released_metal_runtime_streams_one_language_layer_and_keeps_cuda_exact() {
        let metal = released_h3_qwen_nvfp4_runtime_memory_facts_for_placement(
            H3QwenNvfp4RuntimePlacement::MetalStreamed,
        )
        .unwrap();
        let cuda = released_h3_qwen_nvfp4_runtime_memory_facts_for_placement(
            H3QwenNvfp4RuntimePlacement::Accelerated,
        )
        .unwrap();
        assert_eq!(metal.effective_parameter_bytes, 15_686_891_864);
        assert_eq!(metal.host_resident_parameter_bytes, 1_052_855_836);
        assert_eq!(metal.device_resident_parameter_bytes, 1_191_583_200);
        assert_eq!(metal.retained_parameter_bytes, 2_244_439_036);
        assert_eq!(
            cuda.retained_parameter_bytes - metal.retained_parameter_bytes,
            13_442_452_828
        );
        assert_eq!(metal.maximum_tensor_staging_bytes, 777_912_320);
        assert_eq!(metal.tensor_io_bytes, cuda.tensor_io_bytes);
        assert_eq!(cuda.host_resident_parameter_bytes, 14_495_308_664);
        assert_eq!(cuda.device_resident_parameter_bytes, 1_191_583_200);
    }

    struct CancelAuthentication {
        events: Vec<H3QwenNvfp4LoadEvent>,
    }

    impl H3QwenNvfp4LoadObserver for CancelAuthentication {
        fn should_cancel(&mut self, event: &H3QwenNvfp4LoadEvent) -> bool {
            self.events.push(event.clone());
            matches!(
                event,
                H3QwenNvfp4LoadEvent::Authenticating {
                    completed_bytes,
                    ..
                } if *completed_bytes > 0
            )
        }
    }

    struct CancelTensorRead {
        events: Vec<H3QwenNvfp4LoadEvent>,
    }

    impl H3QwenNvfp4LoadObserver for CancelTensorRead {
        fn should_cancel(&mut self, event: &H3QwenNvfp4LoadEvent) -> bool {
            self.events.push(event.clone());
            matches!(
                event,
                H3QwenNvfp4LoadEvent::LoadingTensor {
                    completed_bytes,
                    ..
                } if *completed_bytes > 0
            )
        }
    }

    #[test]
    fn authentication_is_cancellable_at_a_bounded_read_checkpoint() {
        let path = sparse_published_fixture();
        let mut observer = CancelAuthentication { events: Vec::new() };
        let error = unsafe {
            open_h3_qwen_nvfp4_authority_after_authorization(
                &path,
                H3QwenNvfp4RuntimePlacement::Cpu,
                &mut observer,
            )
        }
        .err()
        .expect("the observer must cancel authentication");
        assert!(error.to_string().contains("authentication cancelled"));
        assert_eq!(observer.events.len(), 1);
        assert!(matches!(
            observer.events[0],
            H3QwenNvfp4LoadEvent::Authenticating {
                completed_bytes: 1_048_576,
                total_bytes: super::super::qwen_nvfp4::H3_QWEN_NVFP4_AWQ_FILE_BYTES,
            }
        ));
        std::fs::remove_file(path).unwrap();
    }

    #[test]
    fn tensor_staging_is_cancellable_at_a_bounded_read_checkpoint() {
        let path = sparse_published_fixture();
        let mut artifact = open_h3_qwen_nvfp4_awq_artifact(&path).unwrap();
        let mut observer = CancelTensorRead { events: Vec::new() };
        let mut progress = TensorLoadProgress {
            observer: &mut observer,
            tensor_index: 1,
            tensor_count: 1,
            completed_before: 0,
            total_bytes: 777_912_320,
        };
        let error = progress
            .read(&mut artifact, "model.embed_tokens.weight")
            .expect_err("the observer must cancel tensor staging");
        assert!(error.to_string().contains("load cancelled"));
        assert_eq!(observer.events.len(), 1);
        assert!(matches!(
            observer.events[0],
            H3QwenNvfp4LoadEvent::LoadingTensor {
                ref name,
                tensor_index: 1,
                tensor_count: 1,
                completed_bytes: 1_048_576,
                total_bytes: 777_912_320,
            } if name == "model.embed_tokens.weight"
        ));
        std::fs::remove_file(path).unwrap();
    }

    #[test]
    fn runtime_rejects_config_drift_before_opening_the_artifact() {
        let missing = std::env::temp_dir().join("mold-h3-qwen-deliberately-missing.safetensors");
        let mut config = released_config().unwrap();
        config.text_config.max_position_embeddings -= 1;
        let error = unsafe {
            load_h3_qwen_nvfp4_conditioner_after_authorization(
                &missing,
                &config,
                &Device::Cpu,
                &mut NoopH3QwenNvfp4LoadObserver,
            )
        }
        .err()
        .expect("frozen config drift must fail");
        assert!(matches!(error, H3QwenNvfp4RuntimeError::Contract(_)));
        assert!(!missing.exists());
    }

    /// Pin the released weight-staging charge for both arms at the default row
    /// chunk, and the relationship between them.
    ///
    /// The widest released projection is `mlp.down_proj`, `[5120, 25600]`. At
    /// `H3_COMFY_PORTABLE_ROW_CHUNK` rows the portable arm stages
    /// `256 * 25600 * 4` = 26,214,400 bytes; the device arm's sum of transients
    /// is 98,716,676. The larger is what a memory authority charges, and it is
    /// 2.37% of the 4,168,069,120-byte measured `qwen_activation_workspace`
    /// grant that bounds the whole `QwenEncode` phase — so #1317's device arm
    /// is absorbed by the existing observation rather than moving it.
    #[test]
    fn released_qwen_nvfp4_weight_staging_charges_the_larger_arm() {
        let portable = h3_nvfp4_weight_staging_bytes(
            H3Nvfp4LinearKind::PortableHostDequantize,
            5_120,
            25_600,
            H3_COMFY_PORTABLE_ROW_CHUNK,
        )
        .unwrap();
        let device = h3_nvfp4_weight_staging_bytes(
            H3Nvfp4LinearKind::DeviceLookupDequantize,
            5_120,
            25_600,
            H3_COMFY_PORTABLE_ROW_CHUNK,
        )
        .unwrap();
        assert_eq!(portable, 26_214_400);
        assert_eq!(device, 98_716_676);
        let widest =
            released_h3_qwen_nvfp4_max_weight_staging_bytes(H3_COMFY_PORTABLE_ROW_CHUNK).unwrap();
        assert_eq!(widest, device as u64);
        assert!(widest > portable as u64);
        // The measured FL2VA `qwen_activation_workspace_bytes` grant.
        assert!(widest * 40 < 4_168_069_120);
        assert!(released_h3_qwen_nvfp4_max_weight_staging_bytes(0).is_err());
    }
}
