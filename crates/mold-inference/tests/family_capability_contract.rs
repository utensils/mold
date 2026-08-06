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
    assert_eq!(ltx2.backends.metal, BackendQualification::Unsupported);
    assert_eq!(ltx2.media, MediaKind::Video);
    assert!(ltx2.workflows.source);
    assert!(ltx2.workflows.generated_audio);
    assert!(ltx2.workflows.chain);

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
            false,
            MediaKind::Video,
            WorkflowCapabilities {
                source: true,
                edit_references: false,
                lora: false,
                generated_audio: false,
                chain: false,
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
