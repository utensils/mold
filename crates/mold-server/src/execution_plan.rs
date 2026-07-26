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
const GIB: u64 = 1024 * MIB;
const BASE_HOST_TRANSIENT: u64 = 256 * MIB;
const GPU_SAFETY_HEADROOM: u64 = 512 * MIB;
const GPU_RUNTIME_FLOOR: u64 = GIB;

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
    let candidates = devices
        .iter()
        .filter(|device| hard_device.as_ref().is_none_or(|hard| hard == &device.id))
        .filter_map(|device| {
            build_plan(
                &request.model,
                &family,
                &capabilities,
                &artifacts,
                &constraints,
                device,
                offload_requested,
            )
        })
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

fn build_plan(
    model: &str,
    family: &str,
    capabilities: &PlacementCapabilities,
    artifacts: &BTreeMap<ComponentRole, PathBuf>,
    effective: &EffectivePlacement,
    device: &DeviceFact,
    offload_requested: bool,
) -> Option<Result<ResolvedExecutionPlan, ExecutionPlanError>> {
    let quantization = infer_quantization(model, artifacts);
    let dtype = infer_dtype(model);
    let artifact_bytes = artifacts
        .iter()
        .map(|(role, path)| (role.clone(), artifact_size_floor(path)))
        .collect::<BTreeMap<_, _>>();
    let static_peak =
        static_vram_floor(family, quantization).max(artifact_bytes.values().copied().sum::<u64>());
    let artifact_total = artifact_bytes.values().copied().sum::<u64>().max(1);
    let pressured = static_peak.saturating_add(GPU_SAFETY_HEADROOM) > device.available_vram_bytes;
    let auto_cpu_text = pressured && capabilities.supports_text_encoder_cpu;

    let mut components = BTreeMap::new();
    let mut cpu_bytes = 0_u64;
    let mut gpu_artifact_bytes = 0_u64;
    for (role, path) in artifacts {
        let constraint = effective
            .components
            .get(role)
            .cloned()
            .unwrap_or(ResolvedComponentConstraint::Auto);
        let place_cpu = constraint == ResolvedComponentConstraint::Cpu
            || (auto_cpu_text && matches!(role, ComponentRole::TextEncoder(_)));
        let bytes = artifact_bytes.get(role).copied().unwrap_or(0);
        let component_peak = ((bytes as u128 * static_peak as u128) / artifact_total as u128)
            .min(u64::MAX as u128) as u64;
        let (placement, load_strategy, vram, host) = if place_cpu {
            let host_bytes = bytes.max(component_peak);
            cpu_bytes = cpu_bytes.saturating_add(host_bytes);
            (
                ResolvedComponentPlacement::Cpu,
                ComponentLoadStrategy::ParkedCpu,
                0,
                host_bytes,
            )
        } else {
            gpu_artifact_bytes = gpu_artifact_bytes.saturating_add(component_peak);
            let strategy = if offload_requested && matches!(role, ComponentRole::Transformer) {
                ComponentLoadStrategy::StreamedBlocks
            } else if matches!(role, ComponentRole::TextEncoder(_)) {
                ComponentLoadStrategy::DropReload
            } else {
                ComponentLoadStrategy::Resident
            };
            (
                ResolvedComponentPlacement::Device(device.id.clone()),
                strategy,
                component_peak,
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

    let predicted_vram = if offload_requested {
        static_peak.min(
            device
                .available_vram_bytes
                .saturating_sub(GPU_SAFETY_HEADROOM),
        )
    } else {
        gpu_artifact_bytes.max(GPU_RUNTIME_FLOOR)
    };
    if predicted_vram.saturating_add(GPU_SAFETY_HEADROOM) > device.available_vram_bytes {
        return None;
    }
    let predicted_host = BASE_HOST_TRANSIENT
        .saturating_add(cpu_bytes)
        .saturating_add(if offload_requested {
            static_peak.saturating_sub(predicted_vram)
        } else {
            0
        });
    let fingerprint =
        execution_fingerprint(model, device, effective, &components, offload_requested);
    Some(Ok(ResolvedExecutionPlan {
        device_id: device.id.clone(),
        device_ordinal: device.ordinal,
        model_fingerprint: model_fingerprint(model, artifacts),
        effective_placement: effective.clone(),
        components,
        attention_backend: AttentionBackend::Auto,
        offload_mode: if offload_requested {
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

fn static_vram_floor(family: &str, quantization: Option<QuantizationVariant>) -> u64 {
    let base = match family {
        "ltx2" | "ltx-2" | "ltx2.3" => 20 * GIB,
        "flux2" | "flux.2" | "flux2-klein" => 16 * GIB,
        "flux" => 12 * GIB,
        "qwen-image" | "qwen-image-edit" | "z-image" | "zimage" => 12 * GIB,
        "sdxl" | "sd3" | "sd3.5" => 8 * GIB,
        "sd15" | "sd1.5" => 4 * GIB,
        _ => 8 * GIB,
    };
    match quantization {
        Some(QuantizationVariant::Q4) => base / 2,
        Some(QuantizationVariant::Q8 | QuantizationVariant::Fp8) => base * 3 / 4,
        None => base,
    }
}

fn artifact_size_floor(path: &Path) -> u64 {
    std::fs::metadata(path)
        .map(|metadata| metadata.len())
        .unwrap_or(64 * MIB)
        .max(64 * MIB)
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
        assert!(plan.predicted_host_increment_bytes >= BASE_HOST_TRANSIENT + 64 * MIB);
        assert!(plan.components.iter().any(|(role, component)| {
            matches!(role, ComponentRole::TextEncoder(_))
                && component.placement == ResolvedComponentPlacement::Cpu
        }));
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
        let flux2_config = config(root.path(), "flux2", None);
        let roomy =
            resolve_execution_plans(&flux2_config, &request(None), &devices(&[24 * GIB]), false)
                .unwrap()
                .remove(0);
        assert!(roomy
            .components
            .values()
            .all(|component| component.placement != ResolvedComponentPlacement::Cpu));

        let pressured = resolve_execution_plans(
            &flux2_config,
            &request(None),
            &devices(&[8 * GIB + 480 * MIB]),
            false,
        )
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
    fn unsupported_offload_is_rejected() {
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
