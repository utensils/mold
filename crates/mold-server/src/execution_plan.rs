//! Request-aware, concrete execution plans used by scheduler admission.
//!
//! The scheduler never guesses component placement from a model name. This
//! module resolves the already-normalized request plus concrete model paths
//! into one immutable plan per eligible device. Plans are validated again on
//! the owner thread before model loading touches CUDA.

use mold_core::{Config, DevicePlacement, DeviceRef, GenerateRequest, ModelPaths};
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, BTreeSet};
use std::path::{Path, PathBuf};

const MIB: u64 = 1024 * 1024;
const BASE_HOST_TRANSIENT: u64 = 256 * MIB;
const UNKNOWN_ARTIFACT_HOST_CHARGE: u64 = 64 * MIB;

#[derive(Clone, Debug, Eq, Ord, PartialEq, PartialOrd)]
pub enum ComponentRole {
    Transformer,
    TransformerShard(u8),
    Vae,
    TextEncoder(u8),
    SpatialUpscaler,
    TemporalUpscaler,
    Decoder,
    DistilledLora,
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
    Auto,
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
    pub model_fingerprint: String,
    pub effective_placement: EffectivePlacement,
    pub components: BTreeMap<ComponentRole, ComponentExecutionPlan>,
    pub attention_backend: AttentionBackend,
    pub engine_load_strategy: mold_inference::LoadStrategy,
    pub offload_mode: OffloadMode,
    pub predicted_vram_peak_bytes: u64,
    pub predicted_host_increment_bytes: u64,
    pub determinism_class: DeterminismClass,
    pub execution_fingerprint: String,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct DeviceFact {
    pub id: String,
    pub ordinal: usize,
    pub available_vram_bytes: u64,
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
    #[error("no device has enough sampled free VRAM for a safe execution plan")]
    InsufficientVram,
    #[error("execution plan was invalidated before CUDA work: {0}")]
    PlanInvalidated(String),
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

pub fn resolve_execution_plans(
    config: &Config,
    request: &GenerateRequest,
    devices: &[DeviceFact],
    offload_requested: bool,
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
    let artifacts = concrete_artifacts(&paths);
    let constraints = effective_constraints(&normalized, &artifacts);
    validate_cpu_constraints(&family, &capabilities, &constraints)?;
    let hard_devices = hard_device_ids(&constraints, devices)?;
    if hard_devices.len() > 1 {
        return Err(ExecutionPlanError::CrossDevicePlacement(
            hard_devices.into_iter().collect::<Vec<_>>().join(", "),
        ));
    }
    let hard_device = hard_devices.into_iter().next();
    let context = PlanContext {
        model: &request.model,
        family: &family,
        capabilities: &capabilities,
        request,
        paths: &paths,
        artifacts: &artifacts,
        effective: &constraints,
        offload_requested,
    };
    let candidates = devices
        .iter()
        .filter(|device| hard_device.as_ref().is_none_or(|hard| hard == &device.id))
        .filter_map(|device| build_plan(&context, device))
        .collect::<Result<Vec<_>, _>>()?;
    if candidates.is_empty() {
        return Err(ExecutionPlanError::InsufficientVram);
    }
    Ok(candidates)
}

pub fn validate_before_cuda(
    plan: &ResolvedExecutionPlan,
    worker_device_id: &str,
    worker_ordinal: usize,
    config: &Config,
    model: &str,
) -> Result<(), ExecutionPlanError> {
    if plan.device_id != worker_device_id || plan.device_ordinal != worker_ordinal {
        return Err(ExecutionPlanError::PlanInvalidated(format!(
            "lease targets {worker_device_id}/gpu:{worker_ordinal}, plan targets {}/gpu:{}",
            plan.device_id, plan.device_ordinal
        )));
    }
    let current_paths = ModelPaths::resolve(model, config).ok_or_else(|| {
        ExecutionPlanError::PlanInvalidated("model paths are no longer resolvable".into())
    })?;
    let current_artifacts = concrete_artifacts(&current_paths);
    let planned_artifacts = plan
        .components
        .iter()
        .map(|(role, component)| (role.clone(), component.artifact_path.clone()))
        .collect::<BTreeMap<_, _>>();
    if current_artifacts != planned_artifacts {
        return Err(ExecutionPlanError::PlanInvalidated(
            "resolved component artifacts changed after admission".into(),
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
    let text = plan
        .components
        .iter()
        .find(|(role, _)| matches!(role, ComponentRole::TextEncoder(_)))
        .map(|(_, component)| match &component.placement {
            ResolvedComponentPlacement::Cpu => DeviceRef::Cpu,
            ResolvedComponentPlacement::Device(id) => DeviceRef::device(id.clone()),
        })
        .unwrap_or(DeviceRef::Auto);
    DevicePlacement {
        text_encoders: text.clone(),
        advanced: Some(mold_core::types::AdvancedPlacement {
            transformer: component_ref(&ComponentRole::Transformer),
            vae: component_ref(&ComponentRole::Vae),
            clip_l: Some(text.clone()),
            clip_g: Some(text.clone()),
            t5: Some(text.clone()),
            qwen: Some(text),
        }),
    }
}

fn concrete_artifacts(paths: &ModelPaths) -> BTreeMap<ComponentRole, PathBuf> {
    let mut artifacts = BTreeMap::new();
    artifacts.insert(ComponentRole::Transformer, paths.transformer.clone());
    for (index, shard) in paths.transformer_shards.iter().enumerate() {
        artifacts.insert(ComponentRole::TransformerShard(index as u8), shard.clone());
    }
    artifacts.insert(ComponentRole::Vae, paths.vae.clone());
    for path in [
        paths.t5_encoder.as_ref(),
        paths.clip_encoder.as_ref(),
        paths.clip_encoder_2.as_ref(),
    ]
    .into_iter()
    .flatten()
    .chain(paths.text_encoder_files.iter())
    .enumerate()
    {
        artifacts.insert(ComponentRole::TextEncoder(path.0 as u8), path.1.clone());
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
    artifacts
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
            ComponentRole::TextEncoder(index) => {
                if let Some(advanced) = advanced {
                    match *index {
                        0 => advanced.t5.as_ref().or(Some(&placement.text_encoders)),
                        1 => advanced.clip_l.as_ref().or(Some(&placement.text_encoders)),
                        2 => advanced.clip_g.as_ref().or(Some(&placement.text_encoders)),
                        _ => advanced.qwen.as_ref().or(Some(&placement.text_encoders)),
                    }
                } else {
                    Some(&placement.text_encoders)
                }
            }
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
            ComponentRole::TextEncoder(_) => capabilities.supports_text_encoder_cpu,
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
    let request_has_lora = crate::model_manager::request_has_effective_lora(context.request);
    let initial_memory = crate::memory_preflight::estimate_generation_memory_for_request(
        context.request,
        context.paths,
        hint,
        Some(device.available_vram_bytes),
        context.offload_requested,
        request_has_lora,
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
            let cpu = constraint == ResolvedComponentConstraint::Cpu
                || (auto_cpu_text && matches!(role, ComponentRole::TextEncoder(_)));
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
            } else if matches!(role, ComponentRole::TextEncoder(_)) {
                ComponentLoadStrategy::DropReload
            } else {
                ComponentLoadStrategy::Resident
            };
            (
                ResolvedComponentPlacement::Device(device.id.clone()),
                strategy,
                0,
                0,
            )
        };
        components.insert(
            role.clone(),
            ComponentExecutionPlan {
                role: role.clone(),
                artifact_path: path.clone(),
                content_fingerprint: fingerprint_path(path),
                dtype,
                quantization,
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
        memory.block_offload,
    );
    Some(Ok(ResolvedExecutionPlan {
        device_id: device.id.clone(),
        device_ordinal: device.ordinal,
        model_fingerprint: model_fingerprint(context.model, context.artifacts),
        effective_placement: context.effective.clone(),
        components,
        attention_backend: AttentionBackend::Auto,
        engine_load_strategy: memory.load_strategy,
        offload_mode: if memory.block_offload {
            OffloadMode::Block
        } else {
            OffloadMode::None
        },
        predicted_vram_peak_bytes: predicted_vram,
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

    let mut encoder_index = 0_u8;
    if on_cpu(&ComponentRole::TextEncoder(encoder_index)) {
        gpu.t5_encoder = None;
    }
    if paths.t5_encoder.is_some() {
        encoder_index = encoder_index.saturating_add(1);
    }
    if on_cpu(&ComponentRole::TextEncoder(encoder_index)) {
        gpu.clip_encoder = None;
    }
    if paths.clip_encoder.is_some() {
        encoder_index = encoder_index.saturating_add(1);
    }
    if on_cpu(&ComponentRole::TextEncoder(encoder_index)) {
        gpu.clip_encoder_2 = None;
    }
    if paths.clip_encoder_2.is_some() {
        encoder_index = encoder_index.saturating_add(1);
    }
    gpu.text_encoder_files = paths
        .text_encoder_files
        .iter()
        .enumerate()
        .filter_map(|(offset, path)| {
            (!on_cpu(&ComponentRole::TextEncoder(
                encoder_index.saturating_add(offset as u8),
            )))
            .then_some(path.clone())
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
            .values()
            .map(|path| path.to_string_lossy().to_ascii_lowercase())
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

fn fingerprint_path(path: &Path) -> ContentFingerprint {
    let mut hash = Sha256::new();
    hash.update(path.as_os_str().as_encoded_bytes());
    match std::fs::metadata(path) {
        Ok(metadata) => {
            hash.update(metadata.len().to_le_bytes());
            if let Ok(modified) = metadata.modified() {
                if let Ok(since_epoch) = modified.duration_since(std::time::UNIX_EPOCH) {
                    hash.update(since_epoch.as_nanos().to_le_bytes());
                }
            }
        }
        Err(error) => hash.update(error.kind().to_string().as_bytes()),
    }
    ContentFingerprint(format!("{:x}", hash.finalize()))
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
    offload: bool,
) -> String {
    let mut hash = Sha256::new();
    hash.update(model.as_bytes());
    hash.update(device.id.as_bytes());
    hash.update(format!("{effective:?}").as_bytes());
    hash.update(format!("{components:?}").as_bytes());
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
            matches!(role, ComponentRole::TextEncoder(_))
                && component.placement == ResolvedComponentPlacement::Cpu
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
            matches!(role, ComponentRole::TextEncoder(_))
                && component.placement == ResolvedComponentPlacement::Cpu
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
    fn each_candidate_uses_its_own_sampled_free_vram() {
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
        let plan = resolve_execution_plans(&config, &request(None), &devices(&[24 * GIB]), false)
            .unwrap()
            .remove(0);
        let placement = materialized_placement(&plan);
        assert!(matches!(
            placement.advanced.unwrap().transformer,
            DeviceRef::Device { .. }
        ));
        validate_before_cuda(&plan, "cuda:0", 0, &config, "test:q4").unwrap();

        std::fs::write(root.path().join("transformer-q4.gguf"), vec![1_u8; 2048]).unwrap();
        assert!(matches!(
            validate_before_cuda(&plan, "cuda:0", 0, &config, "test:q4"),
            Err(ExecutionPlanError::PlanInvalidated(_))
        ));
    }
}
