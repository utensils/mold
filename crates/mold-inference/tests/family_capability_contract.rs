use std::collections::BTreeSet;

use mold_inference::{
    production_family_capabilities, production_family_capability_for_family, BackendQualification,
    ComponentPlacementCapability, DeterminismGuarantee, MediaKind, SeedContract,
    TiledVaeCapability, WorkflowCapabilities,
};

const FACTORY_FAMILIES: &[&str] = &[
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
    "hunyuan3d",
];

#[test]
fn every_factory_family_has_one_complete_weight_free_capability_contract() {
    let capabilities = production_family_capabilities();
    assert_eq!(
        capabilities
            .iter()
            .map(|capability| capability.family)
            .collect::<Vec<_>>(),
        FACTORY_FAMILIES
    );

    let mut names = BTreeSet::new();
    for capability in capabilities {
        assert!(names.insert(capability.family));
        for alias in capability.aliases {
            assert!(names.insert(alias), "duplicate alias {alias}");
            assert_eq!(
                production_family_capability_for_family(alias),
                Some(capability),
                "alias {alias} must resolve the entire family contract"
            );
        }

        assert_eq!(capability.execution.native_batch_sizes, &[1]);
        assert!(capability.execution.cooperative_cancellation);
        assert_eq!(
            capability.determinism,
            DeterminismGuarantee::ExactWithinExecutionFingerprint
        );
        assert_eq!(
            capability.seed_contract,
            SeedContract::CpuSeededNoiseTransferredToExecutionDevice
        );
        assert!(!capability.tier1.owner.is_empty());
        assert!(!capability.tier1.reference.is_empty());
        assert!(!capability.tier2.owner.is_empty());
        assert!(!capability.tier2.reference.is_empty());
    }
}

#[test]
fn backend_and_deep_path_claims_match_current_runtime_boundaries() {
    let capabilities = production_family_capabilities();
    let ltx2 = capabilities
        .iter()
        .find(|capability| capability.family == "ltx2")
        .unwrap();
    assert_eq!(ltx2.backends.cuda, BackendQualification::Supported);
    assert_eq!(ltx2.backends.cpu, BackendQualification::CorrectnessOnly);
    // Metal promoted to `Supported` (#597): the #1030 perf campaign measured
    // and optimized the path on real Apple Silicon (FP8 LUT widening #1032,
    // single-decode VAE chunking #1034), and checkpoint-backed renders back
    // both the 19B (LTX-2) and 22B (LTX-2.3) distilled FP8 tiers.
    assert_eq!(ltx2.backends.metal, BackendQualification::Supported);
    assert_eq!(ltx2.media, MediaKind::Video);
    assert!(ltx2.workflows.source);
    assert!(ltx2.workflows.generated_audio);
    assert!(ltx2.workflows.chain);

    let wan = capabilities
        .iter()
        .find(|capability| capability.family == "wan")
        .unwrap();
    assert_eq!(wan.backends.cuda, BackendQualification::Supported);
    assert_eq!(wan.backends.cpu, BackendQualification::CorrectnessOnly);
    // The Metal correctness path (#800): promoted from `Unsupported` with the
    // folded VAE reductions, Metal-chunked math attention, family-scoped BF16,
    // and the named fp8 refusal. `Supported` waits on perf UAT.
    assert_eq!(wan.backends.metal, BackendQualification::CorrectnessOnly);

    let hunyuan3d = capabilities
        .iter()
        .find(|capability| capability.family == "hunyuan3d")
        .unwrap();
    // Metal is the one qualified backend: `hunyuan3d-mini-turbo:fp16` on an
    // M4 Max at octree 192/256/320 against ComfyUI on the same checkpoint,
    // image and seed (`scripts/capture-hunyuan3d-metal-uat.sh`). CUDA is
    // portable by construction but unmeasured, and the CPU never runs this
    // family for memory reasons.
    assert_eq!(hunyuan3d.backends.metal, BackendQualification::Supported);
    assert_eq!(
        hunyuan3d.backends.cuda,
        BackendQualification::CorrectnessOnly
    );
    assert_eq!(
        hunyuan3d.backends.cpu,
        BackendQualification::CorrectnessOnly
    );

    let qwen_edit = capabilities
        .iter()
        .find(|capability| capability.family == "qwen-image-edit")
        .unwrap();
    assert!(!qwen_edit.workflows.source);
    assert!(qwen_edit.workflows.edit_references);
    assert!(qwen_edit.workflows.lora);

    for family in ["flux", "flux2", "sd15", "sdxl", "sd3"] {
        let capability = capabilities
            .iter()
            .find(|capability| capability.family == family)
            .unwrap();
        assert_eq!(
            capability.tiled_vae,
            TiledVaeCapability::GenericPolicy,
            "{family}"
        );
    }
    for family in ["qwen-image", "qwen-image-edit"] {
        let capability = capabilities
            .iter()
            .find(|capability| capability.family == family)
            .unwrap();
        assert_eq!(
            capability.tiled_vae,
            TiledVaeCapability::NativeCuda,
            "{family}"
        );
    }
    assert_eq!(ltx2.tiled_vae, TiledVaeCapability::NativeTemporalChunks);

    let expected = [
        (
            "flux",
            ComponentPlacementCapability {
                text_encoder_cpu: true,
                vae_cpu: false,
                audio_components_cpu: false,
            },
            true,
            MediaKind::Image,
            WorkflowCapabilities {
                source: true,
                edit_references: false,
                lora: true,
                generated_audio: false,
                chain: false,
            },
        ),
        (
            "flux2",
            ComponentPlacementCapability {
                text_encoder_cpu: true,
                vae_cpu: true,
                audio_components_cpu: false,
            },
            true,
            MediaKind::Image,
            WorkflowCapabilities {
                source: true,
                edit_references: false,
                lora: true,
                generated_audio: false,
                chain: false,
            },
        ),
        (
            "sd15",
            ComponentPlacementCapability::default(),
            false,
            MediaKind::Image,
            WorkflowCapabilities {
                source: true,
                edit_references: false,
                lora: true,
                generated_audio: false,
                chain: false,
            },
        ),
        (
            "sdxl",
            ComponentPlacementCapability::default(),
            false,
            MediaKind::Image,
            WorkflowCapabilities {
                source: true,
                edit_references: false,
                lora: true,
                generated_audio: false,
                chain: false,
            },
        ),
        (
            "sd3",
            ComponentPlacementCapability::default(),
            true,
            MediaKind::Image,
            WorkflowCapabilities {
                source: true,
                edit_references: false,
                lora: true,
                generated_audio: false,
                chain: false,
            },
        ),
        (
            "z-image",
            ComponentPlacementCapability::default(),
            true,
            MediaKind::Image,
            WorkflowCapabilities {
                source: true,
                edit_references: false,
                lora: true,
                generated_audio: false,
                chain: false,
            },
        ),
        (
            "qwen-image",
            ComponentPlacementCapability::default(),
            true,
            MediaKind::Image,
            WorkflowCapabilities {
                source: true,
                edit_references: false,
                lora: true,
                generated_audio: false,
                chain: false,
            },
        ),
        (
            "qwen-image-edit",
            ComponentPlacementCapability::default(),
            true,
            MediaKind::Image,
            WorkflowCapabilities {
                source: false,
                edit_references: true,
                lora: true,
                generated_audio: false,
                chain: false,
            },
        ),
        (
            "ltx-video",
            ComponentPlacementCapability::default(),
            false,
            MediaKind::Video,
            WorkflowCapabilities {
                source: false,
                edit_references: false,
                lora: false,
                generated_audio: false,
                chain: true,
            },
        ),
        (
            "ltx2",
            ComponentPlacementCapability {
                text_encoder_cpu: true,
                vae_cpu: false,
                audio_components_cpu: false,
            },
            true,
            MediaKind::Video,
            WorkflowCapabilities {
                source: true,
                edit_references: false,
                lora: true,
                generated_audio: true,
                chain: true,
            },
        ),
        (
            "wuerstchen",
            ComponentPlacementCapability::default(),
            false,
            MediaKind::Image,
            WorkflowCapabilities {
                source: true,
                edit_references: false,
                lora: false,
                generated_audio: false,
                chain: false,
            },
        ),
        (
            "wan",
            ComponentPlacementCapability {
                text_encoder_cpu: true,
                vae_cpu: false,
                audio_components_cpu: false,
            },
            // Partial block offload: trailing DiT blocks park in host RAM and
            // each returns for the duration of its own forward (#776 item 3).
            true,
            MediaKind::Video,
            WorkflowCapabilities {
                source: true,
                edit_references: false,
                // Merged at load on the safetensors paths, applied as a
                // parallel low-rank branch on GGUF — which is what the A14B
                // fast tier's four-step distill pair rides on.
                lora: true,
                generated_audio: false,
                // `WanEngine` implements `ChainStageRenderer` and
                // `chain::capability_for_family("wan")` answers — the static
                // table said otherwise and contradicted its own runtime (#783).
                chain: true,
            },
        ),
    ];
    for (family, placement, block_offload, media, workflows) in expected {
        let capability = capabilities
            .iter()
            .find(|capability| capability.family == family)
            .unwrap();
        assert_eq!(capability.placement, placement, "{family} placement");
        assert_eq!(capability.block_offload, block_offload, "{family} offload");
        assert_eq!(capability.media, media, "{family} media");
        assert_eq!(capability.workflows, workflows, "{family} workflows");
    }
}

/// The static table is a pre-load *description* of the runtime, so its `chain`
/// flag has exactly one authority: whether `chain::capability_for_family`
/// answers for that family. Wan's entry said `false` while its engine
/// implemented `ChainStageRenderer` and the chain registry routed to it (#783);
/// pinning the two together is what keeps the table from drifting again.
#[test]
fn static_chain_capability_agrees_with_the_chain_registry() {
    for capability in production_family_capabilities() {
        assert_eq!(
            capability.workflows.chain,
            mold_inference::chain::capability_for_family(capability.family).is_some(),
            "{} chain",
            capability.family
        );
    }
}

/// The checked-in qualification matrix is a *description* of the registry, so
/// its columns are pinned to it rather than reviewed by eye. Wan's row claimed
/// no block offload while `batch.rs` declared `block_offload: true` for the
/// partial-park path the A14B pair depends on (#783).
#[test]
fn the_qualification_matrix_block_offload_column_matches_the_registry() {
    let matrix = include_str!("../../../docs/qualification/multi-gpu-family-matrix.md");
    // `| family | aliases | backends | cpu placement | block offload | …`
    const BLOCK_OFFLOAD_COLUMN: usize = 4;

    for capability in production_family_capabilities() {
        let row = matrix
            .lines()
            .find(|line| {
                line.trim_start()
                    .starts_with(&format!("| `{}` |", capability.family))
            })
            .unwrap_or_else(|| panic!("qualification matrix has no row for {}", capability.family));
        let cells: Vec<&str> = row
            .trim()
            .trim_matches('|')
            .split('|')
            .map(str::trim)
            .collect();
        let declared = match cells[BLOCK_OFFLOAD_COLUMN] {
            "yes" => true,
            "no" => false,
            other => panic!(
                "{} block-offload cell is neither yes nor no: {other:?}",
                capability.family
            ),
        };
        assert_eq!(
            declared, capability.block_offload,
            "{} block offload: matrix says {declared}, registry says {}",
            capability.family, capability.block_offload
        );
    }
}

#[test]
fn qualification_references_and_checked_in_matrix_are_concrete() {
    let repo_root = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(std::path::Path::parent)
        .unwrap();
    let matrix = include_str!("../../../docs/qualification/multi-gpu-family-matrix.md");

    for capability in production_family_capabilities() {
        for name in std::iter::once(capability.family).chain(capability.aliases.iter().copied()) {
            assert!(
                matrix.contains(&format!("`{name}`")),
                "qualification matrix is missing {name}"
            );
        }

        let (tier1_path, _) = capability
            .tier1
            .reference
            .split_once("::")
            .expect("Tier-1 reference must be path::case");
        assert!(
            repo_root.join(tier1_path).is_file(),
            "missing Tier-1 path for {}: {tier1_path}",
            capability.family
        );

        let (tier2_path, tier2_test) = capability
            .tier2
            .reference
            .rsplit_once("::")
            .expect("Tier-2 reference must be path::test");
        let tier2_source = std::fs::read_to_string(repo_root.join(tier2_path))
            .unwrap_or_else(|error| panic!("cannot read {tier2_path}: {error}"));
        assert!(
            tier2_source.contains(&format!("fn {tier2_test}")),
            "Tier-2 reference for {} is not a concrete test: {}",
            capability.family,
            capability.tier2.reference
        );
    }

    assert!(matrix.contains("Deferred; not hardware-qualified"));
    assert!(matrix.contains("not observed hardware"));
}
