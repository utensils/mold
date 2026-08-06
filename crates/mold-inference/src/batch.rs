use crate::BatchExecutionCapability;

/// How completely one backend is supported by a production inference family.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BackendQualification {
    Supported,
    CorrectnessOnly,
    Unsupported,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct BackendApplicability {
    pub cuda: BackendQualification,
    pub metal: BackendQualification,
    pub cpu: BackendQualification,
}

const PORTABLE_BACKENDS: BackendApplicability = BackendApplicability {
    cuda: BackendQualification::Supported,
    metal: BackendQualification::Supported,
    cpu: BackendQualification::Supported,
};

/// Request-controlled component placement. Runtime-owned emergency fallbacks
/// are deliberately not advertised here.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct ComponentPlacementCapability {
    pub text_encoder_cpu: bool,
    pub vae_cpu: bool,
    pub audio_components_cpu: bool,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TiledVaeCapability {
    Unsupported,
    /// Uses the shared `MOLD_VAE_TILED` policy and OOM fallback.
    GenericPolicy,
    /// Uses a family-specific CUDA tiler rather than the shared policy.
    NativeCuda,
    /// Uses family-specific temporal/framewise video decode chunks.
    NativeTemporalChunks,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DeterminismGuarantee {
    /// Exact output is claimed only when the normalized request, immutable
    /// artifacts, execution-equivalence fingerprint, and backend guarantee
    /// are equal.
    ExactWithinExecutionFingerprint,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SeedContract {
    /// Initial noise is produced from the request seed on CPU, then moved to
    /// the execution device. This is the current production-family contract.
    CpuSeededNoiseTransferredToExecutionDevice,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MediaKind {
    Image,
    Video,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct WorkflowCapabilities {
    /// Standard source-image img2img/inpaint or image-to-video input.
    pub source: bool,
    /// Ordered reference images consumed by a dedicated edit-family path.
    pub edit_references: bool,
    pub lora: bool,
    /// Generated audio is checkpoint-specific even for a capable family.
    pub generated_audio: bool,
    pub chain: bool,
}

/// Checked-in owner and concrete executable/test reference for the layered
/// qualification matrix. These strings are duplicated in the human-readable
/// matrix so reviewers can audit claims without loading model weights.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct QualificationReference {
    pub owner: &'static str,
    pub reference: &'static str,
}

/// Static, pre-load capability contract for one production inference family.
///
/// This extends the original batch registry rather than introducing a second
/// family registry. Server execution planning, parent batching, and factory
/// drift checks all resolve through this table.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct FamilyBatchCapability {
    pub family: &'static str,
    pub aliases: &'static [&'static str],
    pub backends: BackendApplicability,
    pub placement: ComponentPlacementCapability,
    pub block_offload: bool,
    pub tiled_vae: TiledVaeCapability,
    pub execution: BatchExecutionCapability,
    pub determinism: DeterminismGuarantee,
    pub seed_contract: SeedContract,
    pub media: MediaKind,
    pub workflows: WorkflowCapabilities,
    pub tier1: QualificationReference,
    pub tier2: QualificationReference,
}

const SINGLETON: BatchExecutionCapability = BatchExecutionCapability::SINGLETON_COOPERATIVE;
const EXACT: DeterminismGuarantee = DeterminismGuarantee::ExactWithinExecutionFingerprint;
const CPU_SEED: SeedContract = SeedContract::CpuSeededNoiseTransferredToExecutionDevice;
const TIER1: QualificationReference = QualificationReference {
    owner: "runtime-qualification",
    reference: "scripts/regression-matrix.sh::installed_family_base_case",
};
const QWEN_EDIT_TIER1: QualificationReference = QualificationReference {
    owner: "runtime-qualification",
    reference: "scripts/qwen-edit-parity-smoke.sh::explicit_source_edit_cuda_smoke",
};

const fn deep(reference: &'static str) -> QualificationReference {
    QualificationReference {
        owner: "mold-inference",
        reference,
    }
}

/// Authoritative production family registry.
///
/// Every factory family is explicit here, even when two families currently
/// share one runtime engine. All generation families remain singleton until a
/// measured engine contract returns multiple ordered outputs.
const PRODUCTION_FAMILY_CAPABILITIES: &[FamilyBatchCapability] = &[
    FamilyBatchCapability {
        family: "flux",
        aliases: &[],
        backends: PORTABLE_BACKENDS,
        placement: ComponentPlacementCapability {
            text_encoder_cpu: true,
            vae_cpu: false,
            audio_components_cpu: false,
        },
        block_offload: true,
        tiled_vae: TiledVaeCapability::GenericPolicy,
        execution: SINGLETON,
        determinism: EXACT,
        seed_contract: CPU_SEED,
        media: MediaKind::Image,
        workflows: WorkflowCapabilities {
            source: true,
            edit_references: false,
            lora: true,
            generated_audio: false,
            chain: false,
        },
        tier1: TIER1,
        tier2: deep(
            "crates/mold-inference/src/flux/pipeline.rs::forced_offload_uses_sequential_generation_path_for_bf16_flux",
        ),
    },
    FamilyBatchCapability {
        family: "flux2",
        aliases: &["flux.2", "flux2-klein"],
        backends: PORTABLE_BACKENDS,
        placement: ComponentPlacementCapability {
            text_encoder_cpu: true,
            vae_cpu: true,
            audio_components_cpu: false,
        },
        block_offload: true,
        tiled_vae: TiledVaeCapability::GenericPolicy,
        execution: SINGLETON,
        determinism: EXACT,
        seed_contract: CPU_SEED,
        media: MediaKind::Image,
        workflows: WorkflowCapabilities {
            source: true,
            edit_references: false,
            lora: true,
            generated_audio: false,
            chain: false,
        },
        tier1: TIER1,
        tier2: deep(
            "crates/mold-inference/src/flux2/pipeline.rs::flux2_selected_bf16_offload_reaches_runtime_loader",
        ),
    },
    FamilyBatchCapability {
        family: "sd15",
        aliases: &["sd1.5", "stable-diffusion-1.5"],
        backends: PORTABLE_BACKENDS,
        placement: ComponentPlacementCapability {
            text_encoder_cpu: false,
            vae_cpu: false,
            audio_components_cpu: false,
        },
        block_offload: false,
        tiled_vae: TiledVaeCapability::GenericPolicy,
        execution: SINGLETON,
        determinism: EXACT,
        seed_contract: CPU_SEED,
        media: MediaKind::Image,
        workflows: WorkflowCapabilities {
            source: true,
            edit_references: false,
            lora: true,
            generated_audio: false,
            chain: false,
        },
        tier1: TIER1,
        tier2: deep(
            "crates/mold-inference/src/sd15/pipeline.rs::from_single_file_real_shape_load_smoke",
        ),
    },
    FamilyBatchCapability {
        family: "sdxl",
        aliases: &[],
        backends: PORTABLE_BACKENDS,
        placement: ComponentPlacementCapability {
            text_encoder_cpu: false,
            vae_cpu: false,
            audio_components_cpu: false,
        },
        block_offload: false,
        tiled_vae: TiledVaeCapability::GenericPolicy,
        execution: SINGLETON,
        determinism: EXACT,
        seed_contract: CPU_SEED,
        media: MediaKind::Image,
        workflows: WorkflowCapabilities {
            source: true,
            edit_references: false,
            lora: true,
            generated_audio: false,
            chain: false,
        },
        tier1: TIER1,
        tier2: deep(
            "crates/mold-inference/src/sdxl/pipeline.rs::from_single_file_real_shape_load_smoke",
        ),
    },
    FamilyBatchCapability {
        family: "sd3",
        aliases: &["sd3.5", "stable-diffusion-3", "stable-diffusion-3.5"],
        backends: PORTABLE_BACKENDS,
        placement: ComponentPlacementCapability {
            text_encoder_cpu: false,
            vae_cpu: false,
            audio_components_cpu: false,
        },
        block_offload: true,
        tiled_vae: TiledVaeCapability::GenericPolicy,
        execution: SINGLETON,
        determinism: EXACT,
        seed_contract: CPU_SEED,
        media: MediaKind::Image,
        workflows: WorkflowCapabilities {
            source: true,
            edit_references: false,
            lora: true,
            generated_audio: false,
            chain: false,
        },
        tier1: TIER1,
        tier2: deep(
            "crates/mold-inference/src/sd3/pipeline.rs::sd3_selected_bf16_offload_reaches_runtime_loader",
        ),
    },
    FamilyBatchCapability {
        family: "z-image",
        aliases: &[],
        backends: PORTABLE_BACKENDS,
        placement: ComponentPlacementCapability {
            text_encoder_cpu: false,
            vae_cpu: false,
            audio_components_cpu: false,
        },
        block_offload: true,
        tiled_vae: TiledVaeCapability::Unsupported,
        execution: SINGLETON,
        determinism: EXACT,
        seed_contract: CPU_SEED,
        media: MediaKind::Image,
        workflows: WorkflowCapabilities {
            source: true,
            edit_references: false,
            lora: true,
            generated_audio: false,
            chain: false,
        },
        tier1: TIER1,
        tier2: deep(
            "crates/mold-inference/src/zimage/pipeline.rs::zimage_selected_bf16_offload_reaches_runtime_loader",
        ),
    },
    FamilyBatchCapability {
        family: "qwen-image",
        aliases: &["qwen_image"],
        backends: PORTABLE_BACKENDS,
        placement: ComponentPlacementCapability {
            text_encoder_cpu: false,
            vae_cpu: false,
            audio_components_cpu: false,
        },
        block_offload: true,
        tiled_vae: TiledVaeCapability::NativeCuda,
        execution: SINGLETON,
        determinism: EXACT,
        seed_contract: CPU_SEED,
        media: MediaKind::Image,
        workflows: WorkflowCapabilities {
            source: true,
            edit_references: false,
            lora: true,
            generated_audio: false,
            chain: false,
        },
        tier1: TIER1,
        tier2: deep(
            "crates/mold-inference/src/qwen_image/pipeline.rs::qwen_proactive_tiled_decode_skips_primary_full_decode",
        ),
    },
    FamilyBatchCapability {
        family: "qwen-image-edit",
        aliases: &[],
        backends: PORTABLE_BACKENDS,
        placement: ComponentPlacementCapability {
            text_encoder_cpu: false,
            vae_cpu: false,
            audio_components_cpu: false,
        },
        block_offload: true,
        tiled_vae: TiledVaeCapability::NativeCuda,
        execution: SINGLETON,
        determinism: EXACT,
        seed_contract: CPU_SEED,
        media: MediaKind::Image,
        workflows: WorkflowCapabilities {
            source: false,
            edit_references: true,
            lora: true,
            generated_audio: false,
            chain: false,
        },
        tier1: QWEN_EDIT_TIER1,
        tier2: deep(
            "crates/mold-inference/src/qwen_image/pipeline.rs::qwen_quantized_edit_always_uses_split_cfg_on_high_vram_cuda",
        ),
    },
    FamilyBatchCapability {
        family: "ltx-video",
        aliases: &["ltx_video"],
        backends: PORTABLE_BACKENDS,
        placement: ComponentPlacementCapability {
            text_encoder_cpu: false,
            vae_cpu: false,
            audio_components_cpu: false,
        },
        block_offload: false,
        tiled_vae: TiledVaeCapability::Unsupported,
        execution: SINGLETON,
        determinism: EXACT,
        seed_contract: CPU_SEED,
        media: MediaKind::Video,
        workflows: WorkflowCapabilities {
            source: false,
            edit_references: false,
            lora: false,
            generated_audio: false,
            chain: true,
        },
        tier1: TIER1,
        tier2: deep(
            "crates/mold-inference/src/ltx_video/pipeline.rs::decode_apng_round_trips_rgb_frames",
        ),
    },
    FamilyBatchCapability {
        family: "ltx2",
        aliases: &["ltx-2", "ltx2.3"],
        backends: BackendApplicability {
            cuda: BackendQualification::Supported,
            metal: BackendQualification::Unsupported,
            cpu: BackendQualification::CorrectnessOnly,
        },
        placement: ComponentPlacementCapability {
            text_encoder_cpu: true,
            vae_cpu: false,
            audio_components_cpu: false,
        },
        block_offload: true,
        tiled_vae: TiledVaeCapability::NativeTemporalChunks,
        execution: SINGLETON,
        determinism: EXACT,
        seed_contract: CPU_SEED,
        media: MediaKind::Video,
        workflows: WorkflowCapabilities {
            source: true,
            edit_references: false,
            lora: true,
            generated_audio: true,
            chain: true,
        },
        tier1: TIER1,
        tier2: deep(
            "crates/mold-inference/src/ltx2/pipeline.rs::from_transformer_only_single_file_preserves_external_vae_for_chains",
        ),
    },
    FamilyBatchCapability {
        family: "wan",
        aliases: &[],
        backends: BackendApplicability {
            cuda: BackendQualification::Supported,
            metal: BackendQualification::Unsupported,
            cpu: BackendQualification::CorrectnessOnly,
        },
        placement: ComponentPlacementCapability {
            // UMT5-XXL is 11.4 GB at fp16 and is dropped before denoise, so it
            // can be encoded on CPU when the GPU is tight.
            text_encoder_cpu: true,
            vae_cpu: false,
            audio_components_cpu: false,
        },
        // Block streaming is not wired for the Wan DiT yet; the 1.3B and 5B
        // checkpoints fit a 24 GB card whole once the encoder is gone.
        block_offload: false,
        tiled_vae: TiledVaeCapability::Unsupported,
        execution: SINGLETON,
        determinism: EXACT,
        seed_contract: CPU_SEED,
        media: MediaKind::Video,
        workflows: WorkflowCapabilities {
            // Text-to-video only in this layer; I2V/TI2V conditioning, chained
            // clips, generated audio, and LoRA all arrive later.
            source: false,
            edit_references: false,
            lora: false,
            generated_audio: false,
            chain: false,
        },
        tier1: TIER1,
        tier2: deep(
            "crates/mold-inference/src/wan/pipeline.rs::tiny_end_to_end_denoise_and_decode_produces_frames",
        ),
    },
    FamilyBatchCapability {
        family: "wuerstchen",
        aliases: &["wuerstchen-v2"],
        backends: PORTABLE_BACKENDS,
        placement: ComponentPlacementCapability {
            text_encoder_cpu: false,
            vae_cpu: false,
            audio_components_cpu: false,
        },
        block_offload: false,
        tiled_vae: TiledVaeCapability::Unsupported,
        execution: SINGLETON,
        determinism: EXACT,
        seed_contract: CPU_SEED,
        media: MediaKind::Image,
        workflows: WorkflowCapabilities {
            source: true,
            edit_references: false,
            lora: false,
            generated_audio: false,
            chain: false,
        },
        tier1: TIER1,
        tier2: deep(
            "crates/mold-inference/src/wuerstchen/pipeline.rs::wuerstchen_loads_vqgan_tensors_through_shared_pool",
        ),
    },
];

pub fn production_family_capabilities() -> &'static [FamilyBatchCapability] {
    PRODUCTION_FAMILY_CAPABILITIES
}

pub fn production_batch_capabilities() -> &'static [FamilyBatchCapability] {
    production_family_capabilities()
}

pub fn production_family_capability_for_family(
    family: &str,
) -> Option<&'static FamilyBatchCapability> {
    PRODUCTION_FAMILY_CAPABILITIES
        .iter()
        .find(|entry| entry.family == family || entry.aliases.contains(&family))
}

pub fn batch_execution_capability_for_family(family: &str) -> Option<BatchExecutionCapability> {
    production_family_capability_for_family(family).map(|entry| entry.execution)
}

/// Fail closed when an instantiated engine disagrees with the capability used
/// before load by execution and parent planning.
pub fn validate_runtime_batch_capability(
    family: &str,
    runtime: BatchExecutionCapability,
) -> anyhow::Result<()> {
    runtime.validate()?;
    let expected = batch_execution_capability_for_family(family)
        .ok_or_else(|| anyhow::anyhow!("model family '{family}' has no static batch capability"))?;
    anyhow::ensure!(
        runtime == expected,
        "runtime batch capability for family '{family}' does not match the static registry: \
         expected {expected:?}, got {runtime:?}"
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::BTreeSet;

    #[test]
    fn production_family_batch_registry_is_explicit_valid_and_singleton_only() {
        let expected = [
            "flux",
            "flux2",
            "sd15",
            "sdxl",
            "sd3",
            "z-image",
            "qwen-image",
            "qwen-image-edit",
            "ltx-video",
            "ltx2",
            "wan",
            "wuerstchen",
        ];
        assert_eq!(
            production_batch_capabilities()
                .iter()
                .map(|entry| entry.family)
                .collect::<Vec<_>>(),
            expected
        );
        let mut names = BTreeSet::new();
        for entry in production_batch_capabilities() {
            assert!(
                names.insert(entry.family),
                "duplicate family {}",
                entry.family
            );
            for alias in entry.aliases {
                assert!(names.insert(alias), "duplicate family alias {alias}");
            }
            entry.execution.validate().unwrap();
            assert_eq!(
                entry.execution.native_batch_sizes,
                &[1],
                "{} must remain singleton until its engine returns multiple outputs",
                entry.family
            );
            assert!(entry.execution.cooperative_cancellation);
        }
    }

    #[test]
    fn family_batch_registry_resolves_factory_aliases_and_rejects_runtime_drift() {
        for alias in [
            "flux.2",
            "flux2-klein",
            "sd1.5",
            "stable-diffusion-1.5",
            "sd3.5",
            "stable-diffusion-3",
            "stable-diffusion-3.5",
            "qwen_image",
            "ltx_video",
            "ltx-2",
            "ltx2.3",
            "wuerstchen-v2",
        ] {
            assert_eq!(
                batch_execution_capability_for_family(alias)
                    .unwrap()
                    .native_batch_sizes,
                &[1],
                "{alias}"
            );
        }
        assert!(batch_execution_capability_for_family("unknown").is_none());
        assert!(validate_runtime_batch_capability(
            "flux",
            BatchExecutionCapability {
                native_batch_sizes: &[1, 2],
                cooperative_cancellation: true,
            }
        )
        .is_err());
    }
}
