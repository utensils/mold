//! Request-aware, concrete execution plans used by scheduler admission.
//!
//! The scheduler never guesses component placement from a model name. This
//! module resolves the already-normalized request plus concrete model paths
//! into one immutable plan per eligible device. Plans are validated again on
//! the owner thread before model loading touches CUDA.

use mold_core::{Config, DevicePlacement, DeviceRef, GenerateRequest, ModelPaths};
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, BTreeSet};
#[cfg(not(any(unix, windows)))]
use std::io::Read;
use std::path::{Path, PathBuf};
use std::sync::{Mutex, OnceLock};

const MIB: u64 = 1024 * 1024;
const BASE_HOST_TRANSIENT: u64 = 256 * MIB;
const UNKNOWN_ARTIFACT_HOST_CHARGE: u64 = 64 * MIB;

#[derive(Clone, Debug, Eq, Ord, PartialEq, PartialOrd)]
pub enum ComponentRole {
    Transformer,
    TransformerShard(u8),
    Vae,
    T5,
    T5Tokenizer,
    ClipL,
    ClipLTokenizer,
    ClipG,
    ClipGTokenizer,
    QwenShard(u16),
    GemmaShard(u16),
    GenericTextEncoderShard(u16),
    TextTokenizer,
    Lora(u16),
    SpatialUpscaler,
    TemporalUpscaler,
    Decoder,
    DistilledLora,
}

impl ComponentRole {
    fn is_text_encoder(&self) -> bool {
        matches!(
            self,
            Self::T5
                | Self::ClipL
                | Self::ClipG
                | Self::QwenShard(_)
                | Self::GemmaShard(_)
                | Self::GenericTextEncoderShard(_)
        )
    }

    fn is_host_only(&self) -> bool {
        matches!(
            self,
            Self::T5Tokenizer
                | Self::ClipLTokenizer
                | Self::ClipGTokenizer
                | Self::TextTokenizer
                | Self::Lora(_)
        )
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum ResolvedComponentConstraint {
    Auto,
    Cpu,
    Device(String),
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum ResolvedComponentPlacement {
    Cpu,
    Device(String),
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ComponentLoadStrategy {
    Resident,
    DropReload,
    ParkedCpu,
    StreamedBlocks,
    TiledVae,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum QuantizationVariant {
    Q4,
    Q8,
    Fp8,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum PlannedDType {
    Bf16,
    F16,
    F32,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum AttentionBackend {
    Math,
    Flash,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum OffloadMode {
    None,
    Block,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum DeterminismClass {
    CpuSeededCrossBackend,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ContentFingerprint(pub String);

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ComponentExecutionPlan {
    pub role: ComponentRole,
    pub artifact_path: PathBuf,
    pub content_fingerprint: ContentFingerprint,
    pub dtype: Option<PlannedDType>,
    pub quantization: Option<QuantizationVariant>,
    pub placement: ResolvedComponentPlacement,
    pub load_strategy: ComponentLoadStrategy,
    pub predicted_vram_bytes: u64,
    pub predicted_host_bytes: u64,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct EffectivePlacement {
    pub components: BTreeMap<ComponentRole, ResolvedComponentConstraint>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct PlacementCapabilities {
    pub supports_text_encoder_cpu: bool,
    pub supports_vae_cpu: bool,
    pub supports_audio_components_cpu: bool,
    pub supports_block_offload: bool,
    pub supports_tiled_vae: bool,
    pub native_batch_sizes: Vec<u32>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ResolvedExecutionPlan {
    pub device_id: String,
    pub device_ordinal: usize,
    /// Semantic family resolved from authoritative model metadata.
    pub model_family: String,
    pub model_fingerprint: String,
    pub effective_placement: EffectivePlacement,
    pub components: BTreeMap<ComponentRole, ComponentExecutionPlan>,
    /// Exact paths and factory inputs consumed after the lease grant. Worker
    /// dispatch must not re-resolve these from mutable config or environment.
    pub engine_paths: ModelPaths,
    pub engine_config: mold_inference::FrozenEngineConfig,
    /// Mutable inputs observed before dependency preparation. These remain
    /// separate from the materialized engine inputs because an auto encoder
    /// preference is intentionally replaced by one concrete per-device
    /// variant during preparation.
    pub admission_paths: ModelPaths,
    pub admission_engine_config: mold_inference::FrozenEngineConfig,
    /// Ordered effective request/default LoRA stack. Scale uses IEEE bits so
    /// cache identity and equality preserve every finite wire value exactly.
    pub effective_loras: Vec<PlannedLora>,
    pub attention_backend: AttentionBackend,
    pub engine_load_strategy: mold_inference::LoadStrategy,
    pub offload_mode: OffloadMode,
    pub predicted_vram_peak_bytes: u64,
    /// Exact effective capacity against which this candidate was admitted:
    /// current sampled free VRAM plus only owner-reclaimable resident bytes.
    pub admitted_available_vram_bytes: u64,
    pub predicted_host_increment_bytes: u64,
    pub determinism_class: DeterminismClass,
    pub execution_fingerprint: String,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct PlannedLora {
    pub path: PathBuf,
    pub content_fingerprint: ContentFingerprint,
    pub scale_bits: u64,
}

impl PlannedLora {
    pub fn scale(&self) -> f64 {
        f64::from_bits(self.scale_bits)
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct DeviceFact {
    pub id: String,
    pub ordinal: usize,
    pub available_vram_bytes: u64,
}

/// Concrete engine inputs produced by asynchronous dependency preparation.
///
/// The map is keyed by stable runtime device id because mixed-capacity hosts
/// can legitimately select different encoder variants for different GPUs.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct PreparedExecutionInputs {
    /// Hash of the immutable request/config authority that selected these
    /// concrete dependency variants. An empty value is reserved for synthetic
    /// tests that do not exercise production dependency preparation.
    pub authority_fingerprint: String,
    pub by_device: BTreeMap<String, PreparedDeviceExecutionInputs>,
    /// Request-eligible devices whose dependency materialization failed while
    /// at least one sibling succeeded. These omissions are retryable; they
    /// must not silently become the device set for the lifetime of the job.
    pub retryable_device_failures: BTreeMap<String, String>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct PreparedDeviceExecutionInputs {
    pub engine_paths: ModelPaths,
    pub engine_config: mold_inference::FrozenEngineConfig,
    /// Free VRAM used to resolve an `auto` dependency choice.
    pub prepared_available_vram_bytes: u64,
    /// False for explicit variants, whose choice must not churn with
    /// telemetry.
    pub capacity_sensitive: bool,
}

#[derive(Clone, Debug, thiserror::Error, Eq, PartialEq)]
pub enum ExecutionPlanError {
    #[error("model '{model}' has no concrete local artifact paths")]
    MissingArtifacts { model: String },
    #[error("component {role:?} is pinned to CPU, but family '{family}' does not support that placement")]
    UnsupportedCpuPlacement { family: String, role: ComponentRole },
    #[error("block offload was requested, but family '{family}' does not support it")]
    UnsupportedOffload { family: String },
    #[error("component placement references unavailable device '{0}'")]
    UnavailableDevice(String),
    #[error("component placement spans multiple devices: {0}")]
    CrossDevicePlacement(String),
    #[error("no device has enough effective VRAM capacity for a safe execution plan")]
    InsufficientVram,
    #[error("execution plan was invalidated before CUDA work: {0}")]
    PlanInvalidated(String),
    #[error("prepared execution inputs are stale: {0}")]
    PreparedInputsStale(String),
}

pub fn capabilities_for_family(family: &str) -> PlacementCapabilities {
    // Keep this list capability-based and deliberately conservative. Each
    // `true` corresponds to an implemented engine path, not a scheduler
    // heuristic. Unknown/catalog families receive no automatic CPU/offload.
    match family {
        "flux" => PlacementCapabilities {
            supports_text_encoder_cpu: true,
            supports_vae_cpu: false,
            supports_audio_components_cpu: false,
            supports_block_offload: true,
            supports_tiled_vae: true,
            native_batch_sizes: vec![1],
        },
        "flux2" | "flux.2" | "flux2-klein" => PlacementCapabilities {
            supports_text_encoder_cpu: true,
            supports_vae_cpu: true,
            supports_audio_components_cpu: false,
            supports_block_offload: true,
            supports_tiled_vae: true,
            native_batch_sizes: vec![1],
        },
        "ltx2" | "ltx-2" | "ltx2.3" => PlacementCapabilities {
            // Gemma CPU execution is tested; the transformer and video VAE
            // remain on the leased CUDA device.
            supports_text_encoder_cpu: true,
            supports_vae_cpu: false,
            supports_audio_components_cpu: false,
            // The native runtime consumes MOLD_OFFLOAD as a forced full-
            // streaming policy even though its factory does not take the
            // generic FLUX block-offload boolean.
            supports_block_offload: true,
            supports_tiled_vae: true,
            native_batch_sizes: vec![1],
        },
        "z-image" | "qwen-image" | "qwen-image-edit" | "sd3" | "sd3.5" => PlacementCapabilities {
            supports_text_encoder_cpu: false,
            supports_vae_cpu: false,
            supports_audio_components_cpu: false,
            supports_block_offload: true,
            supports_tiled_vae: matches!(family, "z-image" | "qwen-image" | "qwen-image-edit"),
            native_batch_sizes: vec![1],
        },
        _ => PlacementCapabilities {
            supports_text_encoder_cpu: false,
            supports_vae_cpu: false,
            supports_audio_components_cpu: false,
            supports_block_offload: false,
            supports_tiled_vae: false,
            native_batch_sizes: vec![1],
        },
    }
}

/// Resolve hard request/config placement before dependency preparation.
///
/// This is intentionally artifact-only: it filters irrelevant sibling GPUs
/// without consulting CUDA or performing downloads. Full memory admission
/// still happens after dependencies are concrete.
pub fn eligible_devices_for_request(
    config: &Config,
    request: &GenerateRequest,
    devices: &[DeviceFact],
) -> Result<Vec<DeviceFact>, ExecutionPlanError> {
    let paths = ModelPaths::resolve(&request.model, config).ok_or_else(|| {
        ExecutionPlanError::MissingArtifacts {
            model: request.model.clone(),
        }
    })?;
    let family = config
        .resolved_model_config(&request.model)
        .family
        .or_else(|| {
            mold_core::manifest::find_manifest(&request.model)
                .map(|manifest| manifest.family.clone())
        })
        .unwrap_or_else(|| "unknown".to_string());
    let capabilities = capabilities_for_family(&family);
    let engine_config = mold_inference::FrozenEngineConfig::resolve(&request.model, config);
    let loras = effective_loras(config, request);
    let artifacts = concrete_artifacts_for_family(&paths, &family, &loras, &engine_config);
    let normalized = config.effective_placement(&request.model, request.placement.as_ref());
    let effective = effective_constraints(&normalized, &artifacts);
    validate_cpu_constraints(&family, &capabilities, &effective)?;
    let hard = hard_device_ids(&effective, devices)?;
    if hard.len() > 1 {
        return Err(ExecutionPlanError::CrossDevicePlacement(
            hard.into_iter().collect::<Vec<_>>().join(", "),
        ));
    }
    let hard = hard.into_iter().next();
    Ok(devices
        .iter()
        .filter(|device| hard.as_ref().is_none_or(|hard| hard == &device.id))
        .cloned()
        .collect())
}

pub fn resolve_execution_plans(
    config: &Config,
    request: &GenerateRequest,
    devices: &[DeviceFact],
    offload_requested: bool,
) -> Result<Vec<ResolvedExecutionPlan>, ExecutionPlanError> {
    resolve_execution_plans_with_prepared(config, request, devices, offload_requested, None)
}

pub fn resolve_execution_plans_with_prepared(
    config: &Config,
    request: &GenerateRequest,
    devices: &[DeviceFact],
    offload_requested: bool,
    prepared: Option<&PreparedExecutionInputs>,
) -> Result<Vec<ResolvedExecutionPlan>, ExecutionPlanError> {
    let paths = ModelPaths::resolve(&request.model, config).ok_or_else(|| {
        ExecutionPlanError::MissingArtifacts {
            model: request.model.clone(),
        }
    })?;
    let family = config
        .resolved_model_config(&request.model)
        .family
        .or_else(|| {
            mold_core::manifest::find_manifest(&request.model)
                .map(|manifest| manifest.family.clone())
        })
        .unwrap_or_else(|| "unknown".to_string());
    let capabilities = capabilities_for_family(&family);
    if offload_requested && !capabilities.supports_block_offload {
        return Err(ExecutionPlanError::UnsupportedOffload { family });
    }

    let normalized = config.effective_placement(&request.model, request.placement.as_ref());
    let effective_loras = effective_loras(config, request);
    let admission_engine_config =
        mold_inference::FrozenEngineConfig::resolve(&request.model, config);
    if let Some(prepared) = prepared {
        let current_authority =
            preparation_authority_fingerprint(config, request, &paths, &admission_engine_config);
        if !prepared.authority_fingerprint.is_empty()
            && prepared.authority_fingerprint != current_authority
        {
            return Err(ExecutionPlanError::PreparedInputsStale(
                "request or configuration changed after dependency preparation".into(),
            ));
        }
    }
    let admission_artifacts =
        concrete_artifacts_for_family(&paths, &family, &effective_loras, &admission_engine_config);
    let constraints = effective_constraints(&normalized, &admission_artifacts);
    validate_cpu_constraints(&family, &capabilities, &constraints)?;
    let hard_devices = hard_device_ids(&constraints, devices)?;
    if hard_devices.len() > 1 {
        return Err(ExecutionPlanError::CrossDevicePlacement(
            hard_devices.into_iter().collect::<Vec<_>>().join(", "),
        ));
    }
    let hard_device = hard_devices.into_iter().next();
    let candidates = devices
        .iter()
        .filter(|device| hard_device.as_ref().is_none_or(|hard| hard == &device.id))
        .filter_map(|device| {
            let inputs = match prepared {
                Some(prepared) => prepared.by_device.get(&device.id).map(|prepared| {
                    (
                        prepared.engine_paths.clone(),
                        prepared.engine_config.clone(),
                    )
                })?,
                None => (paths.clone(), admission_engine_config.clone()),
            };
            let artifacts =
                concrete_artifacts_for_family(&inputs.0, &family, &effective_loras, &inputs.1);
            let effective = effective_constraints(&normalized, &artifacts);
            if let Err(error) = validate_cpu_constraints(&family, &capabilities, &effective) {
                return Some(Err(error));
            }
            let context = PlanContext {
                model: &request.model,
                family: &family,
                capabilities: &capabilities,
                request,
                paths: &inputs.0,
                engine_config: &inputs.1,
                admission_paths: &paths,
                admission_engine_config: &admission_engine_config,
                effective_loras: &effective_loras,
                artifacts: &artifacts,
                effective: &effective,
                offload_requested,
            };
            build_plan(&context, device)
        })
        .collect::<Result<Vec<_>, _>>()?;
    if candidates.is_empty() {
        return Err(ExecutionPlanError::InsufficientVram);
    }
    Ok(candidates)
}

pub(crate) fn preparation_authority_fingerprint(
    config: &Config,
    request: &GenerateRequest,
    paths: &ModelPaths,
    engine_config: &mold_inference::FrozenEngineConfig,
) -> String {
    let mut normalized_request = request.clone();
    // Local prompt expansion is part of the same dependency-preparation
    // transaction and intentionally changes only these fields.
    normalized_request.prompt.clear();
    normalized_request.original_prompt = None;
    normalized_request.expand = None;

    let mut hash = Sha256::new();
    hash.update(format!("{paths:?}").as_bytes());
    hash.update(format!("{engine_config:?}").as_bytes());
    hash.update(
        format!(
            "{:?}",
            config.effective_placement(&request.model, request.placement.as_ref())
        )
        .as_bytes(),
    );
    hash.update(format!("{:?}", effective_loras(config, request)).as_bytes());
    hash.update(
        serde_json::to_vec(&normalized_request)
            .expect("GenerateRequest serialization is infallible")
            .as_slice(),
    );
    format!("{:x}", hash.finalize())
}

pub fn validate_before_cuda(
    plan: &ResolvedExecutionPlan,
    worker_device_id: &str,
    worker_ordinal: usize,
    config: &Config,
    request: &GenerateRequest,
) -> Result<(), ExecutionPlanError> {
    if plan.device_id != worker_device_id || plan.device_ordinal != worker_ordinal {
        return Err(ExecutionPlanError::PlanInvalidated(format!(
            "lease targets {worker_device_id}/gpu:{worker_ordinal}, plan targets {}/gpu:{}",
            plan.device_id, plan.device_ordinal
        )));
    }
    let model = request.model.as_str();
    let current_paths = ModelPaths::resolve(model, config).ok_or_else(|| {
        ExecutionPlanError::PlanInvalidated("model paths are no longer resolvable".into())
    })?;
    let current_loras = effective_loras(config, request);
    let current_engine_config = mold_inference::FrozenEngineConfig::resolve(model, config);
    if current_paths != plan.admission_paths
        || current_engine_config != plan.admission_engine_config
        || current_loras != plan.effective_loras
    {
        return Err(ExecutionPlanError::PlanInvalidated(
            "frozen engine paths, config, or LoRA stack changed after admission".into(),
        ));
    }
    for component in plan.components.values() {
        let current = fingerprint_path(&component.artifact_path);
        if current != component.content_fingerprint {
            return Err(ExecutionPlanError::PlanInvalidated(format!(
                "artifact '{}' changed after admission",
                component.artifact_path.display()
            )));
        }
        if let ResolvedComponentPlacement::Device(device_id) = &component.placement {
            if device_id != worker_device_id {
                return Err(ExecutionPlanError::PlanInvalidated(format!(
                    "component {:?} targets sibling device {device_id}",
                    component.role
                )));
            }
        }
    }
    Ok(())
}

/// Convert the selected concrete component plan back into the current engine
/// placement contract. This removes runtime Auto decisions: the engine sees
/// exactly CPU or the leased device for every exposed component knob.
pub fn materialized_placement(plan: &ResolvedExecutionPlan) -> DevicePlacement {
    let component_ref = |role: &ComponentRole| {
        plan.components
            .get(role)
            .map(|component| match &component.placement {
                ResolvedComponentPlacement::Cpu => DeviceRef::Cpu,
                ResolvedComponentPlacement::Device(id) => DeviceRef::device(id.clone()),
            })
            .unwrap_or(DeviceRef::Auto)
    };
    let role_ref = |predicate: &dyn Fn(&ComponentRole) -> bool| {
        plan.components
            .iter()
            .find(|(role, _)| predicate(role))
            .map(|(_, component)| match &component.placement {
                ResolvedComponentPlacement::Cpu => DeviceRef::Cpu,
                ResolvedComponentPlacement::Device(id) => DeviceRef::device(id.clone()),
            })
    };
    let text = role_ref(&|role| {
        matches!(
            role,
            ComponentRole::GemmaShard(_) | ComponentRole::GenericTextEncoderShard(_)
        )
    })
    .or_else(|| role_ref(&ComponentRole::is_text_encoder))
    .unwrap_or(DeviceRef::Auto);
    DevicePlacement {
        text_encoders: text.clone(),
        advanced: Some(mold_core::types::AdvancedPlacement {
            transformer: component_ref(&ComponentRole::Transformer),
            vae: component_ref(&ComponentRole::Vae),
            clip_l: role_ref(&|role| matches!(role, ComponentRole::ClipL)),
            clip_g: role_ref(&|role| matches!(role, ComponentRole::ClipG)),
            t5: role_ref(&|role| matches!(role, ComponentRole::T5)),
            qwen: role_ref(&|role| matches!(role, ComponentRole::QwenShard(_))),
        }),
    }
}

/// Apply the request-shaping portion of a selected plan. This freezes the
/// ordered default/request LoRA stack in the payload actually consumed by the
/// engine; later config edits cannot inject or reorder adapters.
pub fn materialize_request(plan: &ResolvedExecutionPlan, request: &mut GenerateRequest) {
    request.placement = Some(materialized_placement(plan));
    let loras = plan
        .effective_loras
        .iter()
        .map(|lora| mold_core::LoraWeight {
            path: lora.path.to_string_lossy().into_owned(),
            scale: lora.scale(),
        })
        .collect::<Vec<_>>();
    request.lora = None;
    request.loras = (!loras.is_empty()).then_some(loras);
}

fn concrete_artifacts_for_family(
    paths: &ModelPaths,
    family: &str,
    effective_loras: &[PlannedLora],
    engine_config: &mold_inference::FrozenEngineConfig,
) -> BTreeMap<ComponentRole, PathBuf> {
    let mut artifacts = BTreeMap::new();
    // LTX-2 audio VAE and vocoder tensors are namespaces inside the primary
    // transformer/checkpoint safetensors, not independently addressable
    // ModelPaths. They therefore share this artifact, device placement, and
    // content fingerprint. Do not invent duplicate component paths: the
    // runtime currently cannot place either audio namespace on CPU
    // (`supports_audio_components_cpu == false`).
    artifacts.insert(ComponentRole::Transformer, paths.transformer.clone());
    for (index, shard) in paths.transformer_shards.iter().enumerate() {
        artifacts.insert(ComponentRole::TransformerShard(index as u8), shard.clone());
    }
    artifacts.insert(ComponentRole::Vae, paths.vae.clone());
    if let Some(path) = engine_config
        .selected_t5_path
        .as_ref()
        .or(paths.t5_encoder.as_ref())
    {
        artifacts.insert(ComponentRole::T5, path.clone());
    }
    if let Some(path) = &paths.t5_tokenizer {
        artifacts.insert(ComponentRole::T5Tokenizer, path.clone());
    }
    if let Some(path) = &paths.clip_encoder {
        artifacts.insert(ComponentRole::ClipL, path.clone());
    }
    if let Some(path) = &paths.clip_tokenizer {
        artifacts.insert(ComponentRole::ClipLTokenizer, path.clone());
    }
    if let Some(path) = &paths.clip_encoder_2 {
        artifacts.insert(ComponentRole::ClipG, path.clone());
    }
    if let Some(path) = &paths.clip_tokenizer_2 {
        artifacts.insert(ComponentRole::ClipGTokenizer, path.clone());
    }
    let selected_text_paths = if !engine_config.selected_qwen3_paths.is_empty() {
        engine_config.selected_qwen3_paths.clone()
    } else if let Some(path) = engine_config.selected_qwen2_path.as_ref() {
        // Qwen-Image-Edit still consumes the native multimodal shards for
        // vision even when its language path is the selected GGUF.
        std::iter::once(path.clone())
            .chain(
                paths
                    .text_encoder_files
                    .iter()
                    .filter(|candidate| *candidate != path)
                    .cloned(),
            )
            .collect()
    } else if !engine_config.selected_gemma_paths.is_empty() {
        // `text_encoder_files` also carries the Gemma tokenizer anchor and
        // optional LTX-2.3 text projection. Keep those host artifacts beside
        // the exact selected Gemma weight files.
        engine_config
            .selected_gemma_paths
            .iter()
            .cloned()
            .chain(paths.text_encoder_files.iter().cloned())
            .collect()
    } else {
        paths.text_encoder_files.clone()
    };
    for (index, path) in selected_text_paths.iter().enumerate() {
        let index = index as u16;
        let role = match family {
            "ltx2" | "ltx-2" | "ltx2.3" => ComponentRole::GemmaShard(index),
            "qwen-image" | "qwen-image-edit" | "z-image" | "flux2" | "flux.2" | "flux2-klein" => {
                ComponentRole::QwenShard(index)
            }
            _ => ComponentRole::GenericTextEncoderShard(index),
        };
        artifacts.insert(role, path.clone());
    }
    if let Some(path) = &paths.text_tokenizer {
        artifacts.insert(ComponentRole::TextTokenizer, path.clone());
    }
    if let Some(path) = &paths.spatial_upscaler {
        artifacts.insert(ComponentRole::SpatialUpscaler, path.clone());
    }
    if let Some(path) = &paths.temporal_upscaler {
        artifacts.insert(ComponentRole::TemporalUpscaler, path.clone());
    }
    if let Some(path) = &paths.decoder {
        artifacts.insert(ComponentRole::Decoder, path.clone());
    }
    if let Some(path) = &paths.distilled_lora {
        artifacts.insert(ComponentRole::DistilledLora, path.clone());
    }
    for (index, lora) in effective_loras.iter().enumerate() {
        artifacts.insert(ComponentRole::Lora(index as u16), lora.path.clone());
    }
    artifacts
}

fn effective_loras(config: &Config, request: &GenerateRequest) -> Vec<PlannedLora> {
    const ZERO_SCALE_EPS: f64 = 1e-8;
    let requested = request
        .loras
        .as_ref()
        .filter(|stack| !stack.is_empty())
        .cloned()
        .or_else(|| request.lora.clone().map(|lora| vec![lora]))
        .or_else(|| {
            config
                .resolved_model_config(&request.model)
                .effective_lora()
                .map(|(path, scale)| mold_core::LoraWeight { path, scale })
                .map(|lora| vec![lora])
        })
        .unwrap_or_default();
    requested
        .into_iter()
        .filter(|lora| lora.scale.abs() > ZERO_SCALE_EPS)
        .map(|lora| {
            let path = PathBuf::from(lora.path);
            PlannedLora {
                content_fingerprint: fingerprint_path(&path),
                path,
                scale_bits: lora.scale.to_bits(),
            }
        })
        .collect()
}

fn effective_constraints(
    placement: &DevicePlacement,
    artifacts: &BTreeMap<ComponentRole, PathBuf>,
) -> EffectivePlacement {
    let advanced = placement.advanced.as_ref();
    let mut components = BTreeMap::new();
    for role in artifacts.keys() {
        let requested = match role {
            ComponentRole::Transformer | ComponentRole::TransformerShard(_) => {
                advanced.map(|value| &value.transformer)
            }
            ComponentRole::Vae => advanced.map(|value| &value.vae),
            ComponentRole::T5 => advanced
                .and_then(|value| value.t5.as_ref())
                .or(Some(&placement.text_encoders)),
            ComponentRole::ClipL => advanced
                .and_then(|value| value.clip_l.as_ref())
                .or(Some(&placement.text_encoders)),
            ComponentRole::ClipG => advanced
                .and_then(|value| value.clip_g.as_ref())
                .or(Some(&placement.text_encoders)),
            ComponentRole::QwenShard(_) => advanced
                .and_then(|value| value.qwen.as_ref())
                .or(Some(&placement.text_encoders)),
            ComponentRole::GemmaShard(_) | ComponentRole::GenericTextEncoderShard(_) => {
                Some(&placement.text_encoders)
            }
            role if role.is_host_only() => Some(&DeviceRef::Cpu),
            _ => None,
        }
        .unwrap_or(&DeviceRef::Auto);
        components.insert(role.clone(), constraint_from_ref(requested));
    }
    EffectivePlacement { components }
}

fn constraint_from_ref(reference: &DeviceRef) -> ResolvedComponentConstraint {
    match reference {
        DeviceRef::Auto => ResolvedComponentConstraint::Auto,
        DeviceRef::Cpu => ResolvedComponentConstraint::Cpu,
        DeviceRef::Gpu { ordinal } => {
            ResolvedComponentConstraint::Device(format!("ordinal:{ordinal}"))
        }
        DeviceRef::Device { id } => ResolvedComponentConstraint::Device(id.clone()),
    }
}

fn validate_cpu_constraints(
    family: &str,
    capabilities: &PlacementCapabilities,
    placement: &EffectivePlacement,
) -> Result<(), ExecutionPlanError> {
    for (role, constraint) in &placement.components {
        if constraint != &ResolvedComponentConstraint::Cpu {
            continue;
        }
        let supported = match role {
            role if role.is_host_only() => true,
            role if role.is_text_encoder() => capabilities.supports_text_encoder_cpu,
            ComponentRole::Vae => capabilities.supports_vae_cpu,
            ComponentRole::Transformer | ComponentRole::TransformerShard(_) => {
                matches!(family, "flux2" | "flux.2" | "flux2-klein")
            }
            _ => false,
        };
        if !supported {
            return Err(ExecutionPlanError::UnsupportedCpuPlacement {
                family: family.to_string(),
                role: role.clone(),
            });
        }
    }
    Ok(())
}

fn hard_device_ids(
    placement: &EffectivePlacement,
    devices: &[DeviceFact],
) -> Result<BTreeSet<String>, ExecutionPlanError> {
    placement
        .components
        .values()
        .filter_map(|constraint| match constraint {
            ResolvedComponentConstraint::Device(id) => Some(id),
            _ => None,
        })
        .map(|id| {
            if let Some(ordinal) = id.strip_prefix("ordinal:") {
                let ordinal = ordinal
                    .parse::<usize>()
                    .map_err(|_| ExecutionPlanError::UnavailableDevice(id.clone()))?;
                devices
                    .iter()
                    .find(|device| device.ordinal == ordinal)
                    .map(|device| device.id.clone())
                    .ok_or_else(|| ExecutionPlanError::UnavailableDevice(id.clone()))
            } else {
                devices
                    .iter()
                    .find(|device| device.id == *id)
                    .map(|device| device.id.clone())
                    .ok_or_else(|| ExecutionPlanError::UnavailableDevice(id.clone()))
            }
        })
        .collect()
}

struct PlanContext<'a> {
    model: &'a str,
    family: &'a str,
    capabilities: &'a PlacementCapabilities,
    request: &'a GenerateRequest,
    paths: &'a ModelPaths,
    engine_config: &'a mold_inference::FrozenEngineConfig,
    admission_paths: &'a ModelPaths,
    admission_engine_config: &'a mold_inference::FrozenEngineConfig,
    effective_loras: &'a [PlannedLora],
    artifacts: &'a BTreeMap<ComponentRole, PathBuf>,
    effective: &'a EffectivePlacement,
    offload_requested: bool,
}

fn build_plan(
    context: &PlanContext<'_>,
    device: &DeviceFact,
) -> Option<Result<ResolvedExecutionPlan, ExecutionPlanError>> {
    let quantization = infer_quantization(context.model, context.artifacts);
    let dtype = infer_dtype(context.model);
    let hint = Some(crate::memory_preflight::ActivationHint::from_request(
        context.request,
        context.family,
    ));
    let request_has_lora = !context.effective_loras.is_empty();
    let gemma_placement = mold_inference::device::resolve_ltx2_gemma_device_override_from_values(
        context
            .engine_config
            .runtime_environment
            .value("MOLD_LTX2_GEMMA_DEVICE"),
        context
            .engine_config
            .runtime_environment
            .value("MOLD_LTX2_DEBUG_FORCE_CPU_PROMPT_ENCODER"),
        device.ordinal,
    );
    let gemma_competes = matches!(
        gemma_placement,
        Some(mold_inference::device::LtxGemmaPlacement::Gpu(ordinal))
            if ordinal == device.ordinal
    );
    let initial_memory = crate::memory_preflight::estimate_generation_memory_for_request(
        context.request,
        context.paths,
        hint,
        Some(device.available_vram_bytes),
        context.offload_requested,
        request_has_lora,
        gemma_competes,
    );
    // A process-wide offload preference is advisory for concrete formats
    // which cannot honor it (for example Flux.2 GGUF/NVFP4 or a LoRA merge).
    // The family capability gate above remains a typed error; this path-level
    // policy exactly matches what engine construction will consume.
    let auto_cpu_text =
        initial_memory.under_memory_pressure && context.capabilities.supports_text_encoder_cpu;

    let placements = context
        .artifacts
        .keys()
        .map(|role| {
            let constraint = context
                .effective
                .components
                .get(role)
                .cloned()
                .unwrap_or(ResolvedComponentConstraint::Auto);
            let cpu = role.is_host_only()
                || constraint == ResolvedComponentConstraint::Cpu
                || (auto_cpu_text && role.is_text_encoder());
            (role.clone(), cpu)
        })
        .collect::<BTreeMap<_, _>>();
    let transformer_on_cpu = placements
        .iter()
        .filter(|(role, _)| {
            matches!(
                role,
                ComponentRole::Transformer | ComponentRole::TransformerShard(_)
            )
        })
        .all(|(_, cpu)| *cpu);
    let gpu_paths = gpu_resident_paths(context.paths, &placements);
    let memory = crate::memory_preflight::estimate_generation_memory_for_request(
        context.request,
        &gpu_paths,
        hint,
        Some(device.available_vram_bytes),
        initial_memory.block_offload && !transformer_on_cpu,
        request_has_lora,
        gemma_competes,
    );
    if memory.fits_available_memory != Some(true) {
        return None;
    }

    let mut components = BTreeMap::new();
    let mut host_paths = BTreeSet::new();
    for (role, path) in context.artifacts {
        let place_cpu = placements.get(role).copied().unwrap_or(false);
        let bytes = artifact_size(path);
        let (placement, load_strategy, vram, host) = if place_cpu {
            host_paths.insert(path.clone());
            (
                ResolvedComponentPlacement::Cpu,
                ComponentLoadStrategy::ParkedCpu,
                0,
                bytes,
            )
        } else {
            let strategy = if memory.block_offload
                && matches!(
                    role,
                    ComponentRole::Transformer | ComponentRole::TransformerShard(_)
                ) {
                host_paths.insert(path.clone());
                ComponentLoadStrategy::StreamedBlocks
            } else if role.is_text_encoder() {
                ComponentLoadStrategy::DropReload
            } else {
                ComponentLoadStrategy::Resident
            };
            (
                ResolvedComponentPlacement::Device(device.id.clone()),
                strategy,
                bytes,
                0,
            )
        };
        components.insert(
            role.clone(),
            ComponentExecutionPlan {
                role: role.clone(),
                artifact_path: path.clone(),
                content_fingerprint: fingerprint_path(path),
                dtype: (!role.is_host_only()).then_some(dtype).flatten(),
                quantization: (!role.is_host_only()).then_some(quantization).flatten(),
                placement,
                load_strategy,
                predicted_vram_bytes: vram,
                predicted_host_bytes: host,
            },
        );
    }

    let predicted_vram = memory.peak_memory_bytes;
    let predicted_host = host_paths.iter().fold(BASE_HOST_TRANSIENT, |total, path| {
        total.saturating_add(artifact_size(path))
    });
    let fingerprint = execution_fingerprint(
        context.model,
        device,
        context.effective,
        &components,
        context.engine_config,
        context.effective_loras,
        memory.block_offload,
    );
    Some(Ok(ResolvedExecutionPlan {
        device_id: device.id.clone(),
        device_ordinal: device.ordinal,
        model_family: context.family.to_string(),
        model_fingerprint: model_fingerprint(context.model, context.artifacts),
        effective_placement: context.effective.clone(),
        components,
        engine_paths: context.paths.clone(),
        engine_config: context.engine_config.clone(),
        admission_paths: context.admission_paths.clone(),
        admission_engine_config: context.admission_engine_config.clone(),
        effective_loras: context.effective_loras.to_vec(),
        attention_backend: match context.engine_config.attention_backend {
            mold_inference::attention::AttentionBackend::Math => AttentionBackend::Math,
            mold_inference::attention::AttentionBackend::Flash => AttentionBackend::Flash,
        },
        engine_load_strategy: memory.load_strategy,
        offload_mode: if memory.block_offload {
            OffloadMode::Block
        } else {
            OffloadMode::None
        },
        predicted_vram_peak_bytes: predicted_vram,
        admitted_available_vram_bytes: device.available_vram_bytes,
        predicted_host_increment_bytes: predicted_host,
        determinism_class: DeterminismClass::CpuSeededCrossBackend,
        execution_fingerprint: fingerprint,
    }))
}

fn gpu_resident_paths(
    paths: &ModelPaths,
    placements: &BTreeMap<ComponentRole, bool>,
) -> ModelPaths {
    let mut gpu = paths.clone();
    let on_cpu = |role: &ComponentRole| placements.get(role).copied().unwrap_or(false);

    if on_cpu(&ComponentRole::Transformer) {
        gpu.transformer = PathBuf::new();
    }
    if !gpu.transformer_shards.is_empty() {
        gpu.transformer_shards = gpu
            .transformer_shards
            .into_iter()
            .enumerate()
            .filter_map(|(index, path)| {
                (!on_cpu(&ComponentRole::TransformerShard(index as u8))).then_some(path)
            })
            .collect();
    }
    if on_cpu(&ComponentRole::Vae) {
        gpu.vae = PathBuf::new();
    }

    if on_cpu(&ComponentRole::T5) {
        gpu.t5_encoder = None;
    }
    if on_cpu(&ComponentRole::ClipL) {
        gpu.clip_encoder = None;
    }
    if on_cpu(&ComponentRole::ClipG) {
        gpu.clip_encoder_2 = None;
    }
    gpu.text_encoder_files = paths
        .text_encoder_files
        .iter()
        .enumerate()
        .filter_map(|(index, path)| {
            let index = index as u16;
            let on_cpu_for_family = [
                ComponentRole::QwenShard(index),
                ComponentRole::GemmaShard(index),
                ComponentRole::GenericTextEncoderShard(index),
            ]
            .iter()
            .any(&on_cpu);
            (!on_cpu_for_family).then_some(path.clone())
        })
        .collect();
    gpu
}

fn artifact_size(path: &Path) -> u64 {
    std::fs::metadata(path)
        .map(|metadata| metadata.len())
        .unwrap_or(UNKNOWN_ARTIFACT_HOST_CHARGE)
}

fn infer_quantization(
    model: &str,
    artifacts: &BTreeMap<ComponentRole, PathBuf>,
) -> Option<QuantizationVariant> {
    let joined = format!(
        "{} {}",
        model.to_ascii_lowercase(),
        artifacts
            .iter()
            .filter(|(role, _)| !role.is_host_only())
            .map(|(_, path)| path.to_string_lossy().to_ascii_lowercase())
            .collect::<Vec<_>>()
            .join(" ")
    );
    if joined.contains("q4") {
        Some(QuantizationVariant::Q4)
    } else if joined.contains("q8") {
        Some(QuantizationVariant::Q8)
    } else if joined.contains("fp8") {
        Some(QuantizationVariant::Fp8)
    } else {
        None
    }
}

fn infer_dtype(model: &str) -> Option<PlannedDType> {
    let lower = model.to_ascii_lowercase();
    if lower.contains("bf16") {
        Some(PlannedDType::Bf16)
    } else if lower.contains("fp16") {
        Some(PlannedDType::F16)
    } else {
        None
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
struct ArtifactMetadataIdentity {
    len: u64,
    platform_identity: Vec<u64>,
}

fn artifact_metadata_identity(
    _path: &Path,
    metadata: &std::fs::Metadata,
) -> ArtifactMetadataIdentity {
    #[cfg(unix)]
    {
        use std::os::unix::fs::MetadataExt;
        ArtifactMetadataIdentity {
            len: metadata.len(),
            // inode/file-id plus ctime detects replacement and in-place
            // mutation even when a caller restores size and mtime.
            platform_identity: vec![
                metadata.dev(),
                metadata.ino(),
                metadata.ctime() as u64,
                metadata.ctime_nsec() as u64,
            ],
        }
    }
    #[cfg(windows)]
    {
        use std::os::windows::fs::MetadataExt;
        use std::os::windows::io::AsRawHandle;
        use windows_sys::Win32::Storage::FileSystem::{
            FileBasicInfo, GetFileInformationByHandleEx, FILE_BASIC_INFO,
        };
        let change_time = std::fs::File::open(_path)
            .ok()
            .and_then(|file| {
                let mut info = std::mem::MaybeUninit::<FILE_BASIC_INFO>::uninit();
                // SAFETY: `file` owns a valid handle for the duration of the
                // call and `info` is correctly sized/aligned for FileBasicInfo.
                let result = unsafe {
                    GetFileInformationByHandleEx(
                        file.as_raw_handle() as _,
                        FileBasicInfo,
                        info.as_mut_ptr().cast(),
                        std::mem::size_of::<FILE_BASIC_INFO>() as u32,
                    )
                };
                // SAFETY: Win32 initializes the entire FILE_BASIC_INFO on a
                // nonzero result.
                (result != 0).then(|| unsafe { info.assume_init().ChangeTime as u64 })
            })
            .unwrap_or(0);
        ArtifactMetadataIdentity {
            len: metadata.file_size(),
            // File ID catches replacement; NTFS ChangeTime catches an
            // in-place overwrite even if last-write time and size are
            // restored by the caller.
            platform_identity: vec![
                u64::from(metadata.volume_serial_number().unwrap_or(0)),
                metadata.file_index().unwrap_or(0),
                metadata.creation_time(),
                change_time,
            ],
        }
    }
    #[cfg(not(any(unix, windows)))]
    {
        let modified = metadata
            .modified()
            .ok()
            .and_then(|time| time.duration_since(std::time::UNIX_EPOCH).ok())
            .map_or(0, |duration| duration.as_nanos() as u64);
        ArtifactMetadataIdentity {
            len: metadata.len(),
            platform_identity: vec![modified],
        }
    }
}

type ArtifactFingerprintCache = BTreeMap<PathBuf, (ArtifactMetadataIdentity, ContentFingerprint)>;

fn artifact_fingerprint_cache() -> &'static Mutex<ArtifactFingerprintCache> {
    static CACHE: OnceLock<Mutex<ArtifactFingerprintCache>> = OnceLock::new();
    CACHE.get_or_init(|| Mutex::new(BTreeMap::new()))
}

#[cfg(not(any(unix, windows)))]
fn hash_artifact_contents(path: &Path) -> std::io::Result<ContentFingerprint> {
    let mut file = std::fs::File::open(path)?;
    let mut hash = Sha256::new();
    let mut buffer = vec![0_u8; 1024 * 1024];
    loop {
        let read = file.read(&mut buffer)?;
        if read == 0 {
            break;
        }
        hash.update(&buffer[..read]);
    }
    Ok(ContentFingerprint(format!("{:x}", hash.finalize())))
}

fn fingerprint_path(path: &Path) -> ContentFingerprint {
    let Ok(before_metadata) = std::fs::metadata(path) else {
        let mut hash = Sha256::new();
        hash.update(path.as_os_str().as_encoded_bytes());
        hash.update(b"missing");
        return ContentFingerprint(format!("{:x}", hash.finalize()));
    };
    let before = artifact_metadata_identity(path, &before_metadata);
    if let Some((_, fingerprint)) = artifact_fingerprint_cache()
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
        .get(path)
        .filter(|(identity, _)| identity == &before)
    {
        return fingerprint.clone();
    }

    // Unix inode+ctime and Windows creation/file metadata are immutable
    // replacement identities and avoid reading multi-gigabyte checkpoints.
    // Exotic platforms fall back to one cached content hash per metadata
    // identity.
    #[cfg(any(unix, windows))]
    let fingerprint = {
        let mut hash = Sha256::new();
        hash.update(path.as_os_str().as_encoded_bytes());
        hash.update(before.len.to_le_bytes());
        for value in &before.platform_identity {
            hash.update(value.to_le_bytes());
        }
        ContentFingerprint(format!("{:x}", hash.finalize()))
    };
    #[cfg(not(any(unix, windows)))]
    let fingerprint = hash_artifact_contents(path).unwrap_or_else(|error| {
        let mut hash = Sha256::new();
        hash.update(path.as_os_str().as_encoded_bytes());
        hash.update(error.kind().to_string().as_bytes());
        ContentFingerprint(format!("{:x}", hash.finalize()))
    });
    if std::fs::metadata(path)
        .ok()
        .map(|metadata| artifact_metadata_identity(path, &metadata))
        .as_ref()
        == Some(&before)
    {
        artifact_fingerprint_cache()
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .insert(path.to_path_buf(), (before, fingerprint.clone()));
    }
    fingerprint
}

fn model_fingerprint(model: &str, artifacts: &BTreeMap<ComponentRole, PathBuf>) -> String {
    let mut hash = Sha256::new();
    hash.update(model.as_bytes());
    for (role, path) in artifacts {
        hash.update(format!("{role:?}").as_bytes());
        hash.update(fingerprint_path(path).0.as_bytes());
    }
    format!("{:x}", hash.finalize())
}

fn execution_fingerprint(
    model: &str,
    device: &DeviceFact,
    effective: &EffectivePlacement,
    components: &BTreeMap<ComponentRole, ComponentExecutionPlan>,
    engine_config: &mold_inference::FrozenEngineConfig,
    effective_loras: &[PlannedLora],
    offload: bool,
) -> String {
    let mut hash = Sha256::new();
    hash.update(model.as_bytes());
    hash.update(device.id.as_bytes());
    hash.update(format!("{effective:?}").as_bytes());
    hash.update(format!("{components:?}").as_bytes());
    hash.update(format!("{engine_config:?}").as_bytes());
    hash.update(format!("{effective_loras:?}").as_bytes());
    hash.update([u8::from(offload)]);
    format!("{:x}", hash.finalize())
}

#[cfg(test)]
mod tests {
    use super::*;
    use mold_core::{AdvancedPlacement, ModelConfig};
    use tempfile::TempDir;

    const GIB: u64 = 1024 * 1024 * 1024;

    fn sparse_file(path: &Path, bytes: u64) {
        let file = std::fs::File::create(path).unwrap();
        file.set_len(bytes).unwrap();
    }

    fn config(root: &Path, family: &str, persisted: Option<DevicePlacement>) -> Config {
        let mut config = Config::default();
        config.models.insert(
            "test:q4".into(),
            ModelConfig {
                transformer: Some(root.join("transformer-q4.gguf").display().to_string()),
                vae: Some(root.join("vae.safetensors").display().to_string()),
                t5_encoder: Some(root.join("t5.safetensors").display().to_string()),
                family: Some(family.into()),
                placement: persisted,
                ..ModelConfig::default()
            },
        );
        config
    }

    fn request(placement: Option<DevicePlacement>) -> GenerateRequest {
        let mut request: GenerateRequest = serde_json::from_str(
            r#"{"prompt":"x","model":"test:q4","width":512,"height":512,"steps":4,"guidance":1.0}"#,
        )
        .unwrap();
        request.placement = placement;
        request
    }

    fn sized_config(
        root: &Path,
        family: &str,
        transformer_gib: u64,
        vae_gib: u64,
        encoder_gib: u64,
    ) -> (Config, GenerateRequest) {
        let transformer = root.join(format!("{family}-transformer.safetensors"));
        let vae = root.join(format!("{family}-vae.safetensors"));
        let encoder = root.join(format!("{family}-encoder.safetensors"));
        sparse_file(&transformer, transformer_gib * GIB);
        sparse_file(&vae, vae_gib * GIB);
        sparse_file(&encoder, encoder_gib * GIB);
        let mut config = Config::default();
        config.models.insert(
            "case:bf16".to_string(),
            ModelConfig {
                transformer: Some(transformer.display().to_string()),
                vae: Some(vae.display().to_string()),
                t5_encoder: Some(encoder.display().to_string()),
                family: Some(family.to_string()),
                ..ModelConfig::default()
            },
        );
        let request = serde_json::from_str(
            r#"{"prompt":"x","model":"case:bf16","width":512,"height":512,"steps":4,"guidance":1.0}"#,
        )
        .unwrap();
        (config, request)
    }

    fn devices(free: &[u64]) -> Vec<DeviceFact> {
        free.iter()
            .enumerate()
            .map(|(ordinal, bytes)| DeviceFact {
                id: format!("cuda:{ordinal}"),
                ordinal,
                available_vram_bytes: *bytes,
            })
            .collect()
    }

    #[test]
    fn request_placement_wins_wholly_over_persisted() {
        let root = TempDir::new().unwrap();
        let persisted = DevicePlacement {
            text_encoders: DeviceRef::Cpu,
            advanced: None,
        };
        let requested = DevicePlacement {
            text_encoders: DeviceRef::Auto,
            advanced: Some(AdvancedPlacement {
                transformer: DeviceRef::device("cuda:1"),
                ..AdvancedPlacement::default()
            }),
        };
        let plans = resolve_execution_plans(
            &config(root.path(), "flux2", Some(persisted)),
            &request(Some(requested)),
            &devices(&[24 * GIB, 24 * GIB]),
            false,
        )
        .unwrap();
        assert_eq!(plans.len(), 1);
        assert_eq!(plans[0].device_id, "cuda:1");
        assert!(plans[0]
            .components
            .values()
            .all(|component| { component.placement != ResolvedComponentPlacement::Cpu }));
    }

    #[test]
    fn explicit_cpu_is_charged_to_host_ram() {
        let root = TempDir::new().unwrap();
        sparse_file(&root.path().join("t5.safetensors"), MIB);
        let placement = DevicePlacement {
            text_encoders: DeviceRef::Cpu,
            advanced: None,
        };
        let plan = resolve_execution_plans(
            &config(root.path(), "flux2", None),
            &request(Some(placement)),
            &devices(&[24 * GIB]),
            false,
        )
        .unwrap()
        .remove(0);
        assert_eq!(
            plan.predicted_host_increment_bytes,
            BASE_HOST_TRANSIENT + MIB
        );
        assert!(plan.components.iter().any(|(role, component)| {
            role.is_text_encoder() && component.placement == ResolvedComponentPlacement::Cpu
        }));
    }

    #[test]
    fn hard_cpu_transformer_pin_is_credited_before_the_fit_gate() {
        let root = TempDir::new().unwrap();
        let (config, mut request) = sized_config(root.path(), "flux2", 32, 1, 2);
        request.placement = Some(DevicePlacement {
            text_encoders: DeviceRef::Auto,
            advanced: Some(AdvancedPlacement {
                transformer: DeviceRef::Cpu,
                ..AdvancedPlacement::default()
            }),
        });

        let plan = resolve_execution_plans(&config, &request, &devices(&[8 * GIB]), false)
            .expect("the CPU-resident transformer must be removed from GPU admission")
            .remove(0);

        assert_eq!(
            plan.components[&ComponentRole::Transformer].placement,
            ResolvedComponentPlacement::Cpu
        );
        assert!(plan.predicted_vram_peak_bytes < 8 * GIB);
        assert!(
            plan.predicted_host_increment_bytes >= BASE_HOST_TRANSIENT + 32 * GIB,
            "CPU-resident weights must be reserved against host RAM"
        );
    }

    #[test]
    fn block_offload_reserves_the_streamed_transformer_in_host_ram() {
        let root = TempDir::new().unwrap();
        let (config, request) = sized_config(root.path(), "flux", 24, 1, 4);
        let plan = resolve_execution_plans(&config, &request, &devices(&[12 * GIB]), false)
            .expect("large FLUX should fit through automatic block offload")
            .remove(0);

        assert_eq!(plan.offload_mode, OffloadMode::Block);
        assert!(
            plan.predicted_host_increment_bytes >= BASE_HOST_TRANSIENT + 24 * GIB,
            "streamed transformer weights remain resident in host memory"
        );
    }

    #[test]
    fn different_gpu_component_pins_are_rejected() {
        let root = TempDir::new().unwrap();
        let placement = DevicePlacement {
            text_encoders: DeviceRef::device("cuda:0"),
            advanced: Some(AdvancedPlacement {
                transformer: DeviceRef::device("cuda:1"),
                ..AdvancedPlacement::default()
            }),
        };
        assert!(matches!(
            resolve_execution_plans(
                &config(root.path(), "flux2", None),
                &request(Some(placement)),
                &devices(&[24 * GIB, 24 * GIB]),
                false,
            ),
            Err(ExecutionPlanError::CrossDevicePlacement(_))
        ));
    }

    #[test]
    fn auto_cpu_exists_only_under_pressure_for_supported_family() {
        let root = TempDir::new().unwrap();
        sparse_file(&root.path().join("transformer-q4.gguf"), 4 * GIB);
        sparse_file(&root.path().join("vae.safetensors"), GIB / 2);
        sparse_file(&root.path().join("t5.safetensors"), 4 * GIB);
        let flux2_config = config(root.path(), "flux2", None);
        let roomy =
            resolve_execution_plans(&flux2_config, &request(None), &devices(&[24 * GIB]), false)
                .unwrap()
                .remove(0);
        assert!(roomy
            .components
            .values()
            .all(|component| component.placement != ResolvedComponentPlacement::Cpu));

        let pressured =
            resolve_execution_plans(&flux2_config, &request(None), &devices(&[9 * GIB]), false)
                .unwrap()
                .remove(0);
        assert!(pressured.components.iter().any(|(role, component)| {
            role.is_text_encoder() && component.placement == ResolvedComponentPlacement::Cpu
        }));

        let unsupported = config(root.path(), "unknown-family", None);
        assert_eq!(
            resolve_execution_plans(&unsupported, &request(None), &devices(&[3 * GIB]), false,),
            Err(ExecutionPlanError::InsufficientVram)
        );
    }

    #[test]
    fn unsupported_family_offload_is_rejected_but_supported_runtime_fallbacks_are_plannable() {
        let root = TempDir::new().unwrap();
        assert!(matches!(
            resolve_execution_plans(
                &config(root.path(), "sdxl", None),
                &request(None),
                &devices(&[24 * GIB]),
                true,
            ),
            Err(ExecutionPlanError::UnsupportedOffload { .. })
        ));
        let ltx2 = resolve_execution_plans(
            &config(root.path(), "ltx2", None),
            &request(None),
            &devices(&[24 * GIB]),
            true,
        )
        .expect("LTX-2 honors MOLD_OFFLOAD through its native streaming runtime");
        assert_eq!(ltx2[0].offload_mode, OffloadMode::Block);

        sparse_file(&root.path().join("transformer-q4.gguf"), 4 * GIB);
        sparse_file(&root.path().join("vae.safetensors"), GIB);
        sparse_file(&root.path().join("t5.safetensors"), GIB);
        let flux2 = resolve_execution_plans(
            &config(root.path(), "flux2", None),
            &request(None),
            &devices(&[24 * GIB]),
            true,
        )
        .expect("a global offload preference must not reject an incompatible quantized path");
        assert_eq!(flux2[0].offload_mode, OffloadMode::None);
    }

    #[test]
    fn candidate_count_scales_without_two_gpu_assumption() {
        let root = TempDir::new().unwrap();
        let config = config(root.path(), "flux2", None);
        for count in [1, 2, 8, 16, 64] {
            let facts = devices(&vec![24 * GIB; count]);
            let plans = resolve_execution_plans(&config, &request(None), &facts, false).unwrap();
            assert_eq!(plans.len(), count);
        }
        assert_eq!(
            resolve_execution_plans(&config, &request(None), &[], false),
            Err(ExecutionPlanError::InsufficientVram)
        );
    }

    #[test]
    fn each_candidate_uses_its_own_effective_vram_capacity() {
        let root = TempDir::new().unwrap();
        let (config, request) = sized_config(root.path(), "flux2", 12, 1, 2);
        let plans = resolve_execution_plans(
            &config,
            &request,
            &devices(&[12 * GIB, 24 * GIB, 48 * GIB]),
            false,
        )
        .unwrap();
        assert_eq!(
            plans
                .iter()
                .map(|plan| plan.device_ordinal)
                .collect::<Vec<_>>(),
            vec![1, 2],
            "a roomy sibling must not make the 12 GiB candidate admissible"
        );
        assert!(plans
            .iter()
            .all(|plan| plan.predicted_vram_peak_bytes <= 24 * GIB));
    }

    #[test]
    fn missing_free_vram_never_falls_back_to_total_capacity() {
        let root = TempDir::new().unwrap();
        let (config, request) = sized_config(root.path(), "sd15", 1, 1, 1);
        assert_eq!(
            resolve_execution_plans(&config, &request, &devices(&[0]), false),
            Err(ExecutionPlanError::InsufficientVram)
        );
    }

    #[test]
    fn request_aware_family_regressions_cover_12_24_and_48_gib() {
        let cases = [
            ("flux", 20, 1, 4, vec![0, 1, 2]),
            ("flux2", 12, 1, 2, vec![1, 2]),
            ("sd15", 2, 1, 1, vec![0, 1, 2]),
            ("sdxl", 6, 1, 2, vec![0, 1, 2]),
            ("sd3", 14, 1, 3, vec![1, 2]),
            ("qwen-image", 34, 1, 6, vec![2]),
            ("z-image", 14, 1, 3, vec![1, 2]),
            ("ltx-video", 8, 1, 2, vec![1, 2]),
            ("ltx2", 40, 1, 10, vec![0, 1, 2]),
        ];
        for (family, transformer, vae, encoder, expected_ordinals) in cases {
            let root = TempDir::new().unwrap();
            let (config, request) = sized_config(root.path(), family, transformer, vae, encoder);
            let plans = resolve_execution_plans(
                &config,
                &request,
                &devices(&[12 * GIB, 24 * GIB, 48 * GIB]),
                false,
            )
            .unwrap_or_else(|error| panic!("{family} planning failed: {error}"));
            assert_eq!(
                plans
                    .iter()
                    .map(|plan| plan.device_ordinal)
                    .collect::<Vec<_>>(),
                expected_ordinals,
                "{family} admission drifted at 12/24/48 GiB"
            );
            if family == "ltx2" {
                assert!(plans.iter().all(|plan| {
                    plan.engine_load_strategy == mold_inference::LoadStrategy::Eager
                        && plan.offload_mode == OffloadMode::None
                }));
            }
        }
    }

    #[test]
    fn missing_artifact_metadata_keeps_a_conservative_host_charge() {
        let missing = PathBuf::from("/definitely/missing/mold-artifact.safetensors");
        assert_eq!(artifact_size(&missing), 64 * MIB);
    }

    #[test]
    fn owner_boundary_rejects_changed_artifact_and_materializes_auto() {
        let root = TempDir::new().unwrap();
        for name in ["transformer-q4.gguf", "vae.safetensors", "t5.safetensors"] {
            std::fs::write(root.path().join(name), vec![0_u8; 1024]).unwrap();
        }
        let config = config(root.path(), "flux2", None);
        let request = request(None);
        let plan = resolve_execution_plans(&config, &request, &devices(&[24 * GIB]), false)
            .unwrap()
            .remove(0);
        let placement = materialized_placement(&plan);
        assert!(matches!(
            placement.advanced.unwrap().transformer,
            DeviceRef::Device { .. }
        ));
        validate_before_cuda(&plan, "cuda:0", 0, &config, &request).unwrap();

        std::fs::write(root.path().join("transformer-q4.gguf"), vec![1_u8; 2048]).unwrap();
        assert!(matches!(
            validate_before_cuda(&plan, "cuda:0", 0, &config, &request),
            Err(ExecutionPlanError::PlanInvalidated(_))
        ));
    }

    #[test]
    fn plan_freezes_tokenizers_factory_config_and_ordered_default_lora_stack() {
        let root = TempDir::new().unwrap();
        for name in [
            "transformer-q4.gguf",
            "vae.safetensors",
            "t5.safetensors",
            "t5-tokenizer.json",
            "first-lora.safetensors",
            "second-lora.safetensors",
        ] {
            std::fs::write(root.path().join(name), name.as_bytes()).unwrap();
        }
        let mut config = config(root.path(), "flux", None);
        let model = config.models.get_mut("test:q4").unwrap();
        model.t5_tokenizer = Some(root.path().join("t5-tokenizer.json").display().to_string());
        model.is_schnell = Some(true);
        model.lora = Some(
            root.path()
                .join("first-lora.safetensors")
                .display()
                .to_string(),
        );
        model.lora_scale = Some(0.75);
        config.t5_variant = Some("q4".into());

        let mut request = request(None);
        let default_plan = resolve_execution_plans(&config, &request, &devices(&[24 * GIB]), false)
            .unwrap()
            .remove(0);
        assert!(default_plan
            .components
            .contains_key(&ComponentRole::T5Tokenizer));
        assert_eq!(default_plan.effective_loras.len(), 1);
        assert_eq!(default_plan.effective_loras[0].scale(), 0.75);
        assert_eq!(default_plan.engine_config.is_schnell, Some(true));
        assert_eq!(default_plan.engine_config.t5_variant.as_deref(), Some("q4"));

        let flux2_transformer = root.path().join("flux2-transformer.safetensors");
        std::fs::write(&flux2_transformer, b"weights").unwrap();
        let model = config.models.get_mut("test:q4").unwrap();
        model.family = Some("flux2".into());
        model.transformer = Some(flux2_transformer.display().to_string());
        let default_lora_plan =
            resolve_execution_plans(&config, &request, &devices(&[24 * GIB]), true)
                .unwrap()
                .remove(0);
        assert_eq!(
            default_lora_plan.offload_mode,
            OffloadMode::None,
            "model-default LoRA must participate in offload/memory admission"
        );

        request.loras = Some(vec![
            mold_core::LoraWeight {
                path: root
                    .path()
                    .join("second-lora.safetensors")
                    .display()
                    .to_string(),
                scale: -0.5,
            },
            mold_core::LoraWeight {
                path: root
                    .path()
                    .join("first-lora.safetensors")
                    .display()
                    .to_string(),
                scale: 1.25,
            },
        ]);
        let request_plan = resolve_execution_plans(&config, &request, &devices(&[24 * GIB]), false)
            .unwrap()
            .remove(0);
        assert_ne!(
            request_plan.execution_fingerprint,
            default_plan.execution_fingerprint
        );
        assert_eq!(
            request_plan
                .effective_loras
                .iter()
                .map(PlannedLora::scale)
                .collect::<Vec<_>>(),
            vec![-0.5, 1.25]
        );
        let mut materialized = request.clone();
        materialize_request(&request_plan, &mut materialized);
        assert!(materialized.lora.is_none());
        assert_eq!(
            materialized
                .loras
                .unwrap()
                .iter()
                .map(|lora| lora.scale)
                .collect::<Vec<_>>(),
            vec![-0.5, 1.25]
        );

        config.models.get_mut("test:q4").unwrap().is_schnell = Some(false);
        assert!(matches!(
            validate_before_cuda(&request_plan, "cuda:0", 0, &config, &request),
            Err(ExecutionPlanError::PlanInvalidated(_))
        ));
    }

    #[test]
    fn semantic_encoder_roles_preserve_sparse_and_mixed_topologies() {
        let root = TempDir::new().unwrap();
        for name in [
            "transformer.safetensors",
            "vae.safetensors",
            "clip-l.safetensors",
            "clip-g.safetensors",
            "t5.safetensors",
            "qwen-0.safetensors",
            "qwen-1.safetensors",
        ] {
            std::fs::write(root.path().join(name), b"x").unwrap();
        }
        let paths = ModelPaths {
            transformer: root.path().join("transformer.safetensors"),
            transformer_shards: Vec::new(),
            vae: root.path().join("vae.safetensors"),
            spatial_upscaler: None,
            temporal_upscaler: None,
            distilled_lora: None,
            t5_encoder: Some(root.path().join("t5.safetensors")),
            clip_encoder: Some(root.path().join("clip-l.safetensors")),
            t5_tokenizer: None,
            clip_tokenizer: None,
            clip_encoder_2: Some(root.path().join("clip-g.safetensors")),
            clip_tokenizer_2: None,
            text_encoder_files: vec![
                root.path().join("qwen-0.safetensors"),
                root.path().join("qwen-1.safetensors"),
            ],
            text_tokenizer: None,
            decoder: None,
        };
        let frozen = mold_inference::FrozenEngineConfig::resolve("unused", &Config::default());
        let artifacts = concrete_artifacts_for_family(&paths, "qwen-image", &[], &frozen);
        assert!(artifacts.contains_key(&ComponentRole::T5));
        assert!(artifacts.contains_key(&ComponentRole::ClipL));
        assert!(artifacts.contains_key(&ComponentRole::ClipG));
        assert!(artifacts.contains_key(&ComponentRole::QwenShard(0)));
        assert!(artifacts.contains_key(&ComponentRole::QwenShard(1)));
        for family in ["flux2", "flux.2", "flux2-klein", "z-image"] {
            let family_artifacts = concrete_artifacts_for_family(&paths, family, &[], &frozen);
            assert!(
                family_artifacts.contains_key(&ComponentRole::QwenShard(0)),
                "{family} must retain semantic Qwen topology"
            );
        }

        let sparse_ltx = ModelPaths {
            t5_encoder: None,
            clip_encoder: None,
            clip_encoder_2: None,
            text_encoder_files: vec![root.path().join("qwen-0.safetensors")],
            ..paths
        };
        let ltx_artifacts = concrete_artifacts_for_family(&sparse_ltx, "ltx2", &[], &frozen);
        assert!(ltx_artifacts.contains_key(&ComponentRole::GemmaShard(0)));
        assert!(!ltx_artifacts.keys().any(|role| matches!(
            role,
            ComponentRole::T5 | ComponentRole::ClipL | ComponentRole::ClipG
        )));
    }

    #[test]
    fn materialized_placement_preserves_clip_only_and_mixed_encoder_choices() {
        let root = TempDir::new().unwrap();
        for name in [
            "transformer.safetensors",
            "vae.safetensors",
            "clip-l.safetensors",
            "t5.safetensors",
        ] {
            std::fs::write(root.path().join(name), b"x").unwrap();
        }

        let mut clip_only_config = Config::default();
        clip_only_config.models.insert(
            "clip-only:bf16".into(),
            ModelConfig {
                transformer: Some(
                    root.path()
                        .join("transformer.safetensors")
                        .display()
                        .to_string(),
                ),
                vae: Some(root.path().join("vae.safetensors").display().to_string()),
                clip_encoder: Some(root.path().join("clip-l.safetensors").display().to_string()),
                family: Some("sdxl".into()),
                ..ModelConfig::default()
            },
        );
        let clip_request: GenerateRequest = serde_json::from_str(
            r#"{"prompt":"x","model":"clip-only:bf16","width":512,"height":512,"steps":4,"guidance":1.0}"#,
        )
        .unwrap();
        let clip_plan = resolve_execution_plans(
            &clip_only_config,
            &clip_request,
            &devices(&[24 * GIB]),
            false,
        )
        .unwrap()
        .remove(0);
        let clip_materialized = materialized_placement(&clip_plan);
        assert!(matches!(
            clip_materialized.text_encoders,
            DeviceRef::Device { .. }
        ));
        assert!(matches!(
            clip_materialized.advanced.unwrap().clip_l,
            Some(DeviceRef::Device { .. })
        ));

        let mut mixed_config = config(root.path(), "flux", None);
        mixed_config.models.get_mut("test:q4").unwrap().clip_encoder =
            Some(root.path().join("clip-l.safetensors").display().to_string());
        let mixed_request = request(Some(DevicePlacement {
            text_encoders: DeviceRef::Auto,
            advanced: Some(AdvancedPlacement {
                t5: Some(DeviceRef::Cpu),
                clip_l: Some(DeviceRef::device("cuda:0")),
                ..AdvancedPlacement::default()
            }),
        }));
        let mixed_plan =
            resolve_execution_plans(&mixed_config, &mixed_request, &devices(&[24 * GIB]), false)
                .unwrap()
                .remove(0);
        let mixed_materialized = materialized_placement(&mixed_plan);
        let advanced = mixed_materialized.advanced.unwrap();
        assert_eq!(advanced.t5, Some(DeviceRef::Cpu));
        assert!(matches!(advanced.clip_l, Some(DeviceRef::Device { .. })));
    }

    #[test]
    fn prepared_inputs_are_per_device_and_preserve_admission_fences() {
        let root = TempDir::new().unwrap();
        for name in [
            "transformer-q4.gguf",
            "vae.safetensors",
            "t5.safetensors",
            "selected-t5.safetensors",
        ] {
            std::fs::write(root.path().join(name), b"prepared").unwrap();
        }
        let config = config(root.path(), "flux", None);
        let request = request(None);
        let admission_paths = ModelPaths::resolve(&request.model, &config).unwrap();
        let mut selected_paths = admission_paths.clone();
        let selected_t5 = root.path().join("selected-t5.safetensors");
        selected_paths.t5_encoder = Some(selected_t5.clone());
        let mut selected_config =
            mold_inference::FrozenEngineConfig::resolve(&request.model, &config);
        selected_config.t5_variant = Some("fp16".to_string());
        selected_config.selected_t5_path = Some(selected_t5.clone());
        let prepared = PreparedExecutionInputs {
            authority_fingerprint: preparation_authority_fingerprint(
                &config,
                &request,
                &admission_paths,
                &mold_inference::FrozenEngineConfig::resolve(&request.model, &config),
            ),
            by_device: BTreeMap::from([(
                "cuda:0".to_string(),
                PreparedDeviceExecutionInputs {
                    engine_paths: selected_paths,
                    engine_config: selected_config,
                    prepared_available_vram_bytes: 24 * GIB,
                    capacity_sensitive: false,
                },
            )]),
            retryable_device_failures: BTreeMap::new(),
        };

        let plans = resolve_execution_plans_with_prepared(
            &config,
            &request,
            &devices(&[24 * GIB, 24 * GIB]),
            false,
            Some(&prepared),
        )
        .unwrap();
        assert_eq!(plans.len(), 1, "unprepared sibling must not be admitted");
        let plan = &plans[0];
        assert_eq!(plan.device_id, "cuda:0");
        assert_eq!(plan.engine_paths.t5_encoder.as_ref(), Some(&selected_t5));
        assert_eq!(plan.admission_paths, admission_paths);
        assert_eq!(
            plan.components
                .get(&ComponentRole::T5)
                .map(|component| &component.artifact_path),
            Some(&selected_t5)
        );
        assert!(
            plan.components
                .values()
                .filter(|component| {
                    matches!(component.placement, ResolvedComponentPlacement::Device(_))
                })
                .all(|component| component.predicted_vram_bytes > 0),
            "every concrete GPU component must expose a non-zero weight estimate"
        );
        validate_before_cuda(plan, "cuda:0", 0, &config, &request).unwrap();
    }

    #[test]
    fn prepared_inputs_are_rejected_when_config_or_request_authority_changes() {
        let root = TempDir::new().unwrap();
        for name in ["transformer-q4.gguf", "vae.safetensors", "t5.safetensors"] {
            std::fs::write(root.path().join(name), b"prepared").unwrap();
        }
        let config = config(root.path(), "flux", None);
        let request = request(None);
        let paths = ModelPaths::resolve(&request.model, &config).unwrap();
        let engine_config = mold_inference::FrozenEngineConfig::resolve(&request.model, &config);
        let prepared = PreparedExecutionInputs {
            authority_fingerprint: preparation_authority_fingerprint(
                &config,
                &request,
                &paths,
                &engine_config,
            ),
            by_device: BTreeMap::from([(
                "cuda:0".to_string(),
                PreparedDeviceExecutionInputs {
                    engine_paths: paths.clone(),
                    engine_config,
                    prepared_available_vram_bytes: 24 * GIB,
                    capacity_sensitive: false,
                },
            )]),
            retryable_device_failures: BTreeMap::new(),
        };

        let mut changed_config = config.clone();
        changed_config.t5_variant = Some("q3".to_string());
        assert!(matches!(
            resolve_execution_plans_with_prepared(
                &changed_config,
                &request,
                &devices(&[24 * GIB]),
                false,
                Some(&prepared),
            ),
            Err(ExecutionPlanError::PreparedInputsStale(_))
        ));

        let mut changed_request = request.clone();
        changed_request.width += 64;
        assert!(matches!(
            resolve_execution_plans_with_prepared(
                &config,
                &changed_request,
                &devices(&[24 * GIB]),
                false,
                Some(&prepared),
            ),
            Err(ExecutionPlanError::PreparedInputsStale(_))
        ));
    }

    #[test]
    fn same_size_in_place_overwrite_with_restored_mtime_changes_content_identity() {
        use filetime::{set_file_mtime, FileTime};

        let root = TempDir::new().unwrap();
        let path = root.path().join("weights.safetensors");
        std::fs::write(&path, b"aaaa").unwrap();
        let original_mtime = FileTime::from_last_modification_time(&path.metadata().unwrap());
        let before = fingerprint_path(&path);

        std::fs::write(&path, b"bbbb").unwrap();
        set_file_mtime(&path, original_mtime).unwrap();

        assert_ne!(before, fingerprint_path(&path));
    }
}
