use mold_core::{Config, GenerateRequest, GpuBackend, ModelConfig, OutputFormat};
use mold_server::execution_plan::{
    resolve_execution_plans, AttentionKernelClass, ComponentLoadStrategy, DeterminismClass,
    DeviceArchitectureClass, DeviceFact, EngineLoadStrategyClass, OffloadMode, PlannedDType,
    QuantizationVariant, SemanticComponentPlacement,
};
use std::path::Path;
use tempfile::TempDir;

const GIB: u64 = 1024 * 1024 * 1024;

fn fixture() -> (TempDir, Config, GenerateRequest) {
    let root = TempDir::new().unwrap();
    for name in ["transformer-q4.gguf", "vae.safetensors", "t5.safetensors"] {
        std::fs::write(root.path().join(name), name.as_bytes()).unwrap();
    }
    let mut config = Config::default();
    config.models.insert(
        "test:q4".into(),
        ModelConfig {
            transformer: Some(path(root.path(), "transformer-q4.gguf")),
            vae: Some(path(root.path(), "vae.safetensors")),
            t5_encoder: Some(path(root.path(), "t5.safetensors")),
            family: Some("flux2".into()),
            ..ModelConfig::default()
        },
    );
    let request = serde_json::from_str(
        r#"{"prompt":"x","model":"test:q4","width":512,"height":512,"steps":4,"guidance":1.0}"#,
    )
    .unwrap();
    (root, config, request)
}

fn path(root: &Path, name: &str) -> String {
    root.join(name).display().to_string()
}

fn device(id: &str, backend: GpuBackend, architecture: Option<(u16, u16)>) -> DeviceFact {
    DeviceFact {
        id: id.into(),
        ordinal: id.bytes().last().unwrap_or(b'0').saturating_sub(b'0') as usize,
        backend,
        compute_capability: architecture,
        available_vram_bytes: 24 * GIB,
    }
}

#[test]
fn compatible_devices_share_parent_equivalence_but_not_lease_identity() {
    let (_root, config, request) = fixture();
    let plans = resolve_execution_plans(
        &config,
        &request,
        &[
            device("cuda:0", GpuBackend::Cuda, Some((8, 6))),
            device("cuda:1", GpuBackend::Cuda, Some((8, 6))),
        ],
        false,
    )
    .unwrap();

    assert_ne!(
        plans[0].execution_fingerprint,
        plans[1].execution_fingerprint
    );
    assert_eq!(
        plans[0].execution_equivalence_fingerprint,
        plans[1].execution_equivalence_fingerprint
    );
    assert_eq!(
        plans[0].execution_environment.architecture,
        DeviceArchitectureClass::CudaComputeCapability { major: 8, minor: 6 }
    );
}

#[test]
fn every_determinism_dimension_participates_in_equivalence_identity() {
    let (_root, config, request) = fixture();
    let base = resolve_execution_plans(
        &config,
        &request,
        &[device("cuda:0", GpuBackend::Cuda, Some((8, 6)))],
        false,
    )
    .unwrap()
    .remove(0)
    .execution_environment;
    let fingerprint = base.fingerprint();

    let mut changed = base.clone();
    changed.backend = GpuBackend::Metal;
    assert_ne!(fingerprint, changed.fingerprint());

    let mut changed = base.clone();
    changed.architecture = DeviceArchitectureClass::CudaComputeCapability {
        major: 10,
        minor: 0,
    };
    assert_ne!(fingerprint, changed.fingerprint());

    let mut changed = base.clone();
    changed.attention_kernel_class = AttentionKernelClass::Flash;
    assert_ne!(fingerprint, changed.fingerprint());

    let mut changed = base.clone();
    changed.code.source_revision = Some("different-code".into());
    assert_ne!(fingerprint, changed.fingerprint());

    let mut changed = base.clone();
    changed.model_fingerprint = "different-model".into();
    assert_ne!(fingerprint, changed.fingerprint());

    let mut changed = base.clone();
    changed
        .components
        .iter_mut()
        .next()
        .unwrap()
        .content_fingerprint
        .0 = "different-assets".into();
    assert_ne!(fingerprint, changed.fingerprint());

    let mut changed = base.clone();
    changed.components.iter_mut().next().unwrap().dtype = Some(PlannedDType::F32);
    assert_ne!(fingerprint, changed.fingerprint());

    let mut changed = base.clone();
    changed
        .components
        .iter_mut()
        .find(|component| component.quantization.is_some())
        .unwrap()
        .quantization = Some(QuantizationVariant::Q8);
    assert_ne!(fingerprint, changed.fingerprint());

    let mut changed = base.clone();
    changed
        .components
        .iter_mut()
        .find(|component| component.placement == SemanticComponentPlacement::AssignedDevice)
        .unwrap()
        .placement = SemanticComponentPlacement::Cpu;
    assert_ne!(fingerprint, changed.fingerprint());

    let mut changed = base.clone();
    changed.components.iter_mut().next().unwrap().load_strategy = ComponentLoadStrategy::ParkedCpu;
    assert_ne!(fingerprint, changed.fingerprint());

    let mut changed = base.clone();
    changed.engine_load_strategy = EngineLoadStrategyClass::Sequential;
    assert_ne!(fingerprint, changed.fingerprint());

    let mut changed = base.clone();
    changed.offload_mode = OffloadMode::Block;
    assert_ne!(fingerprint, changed.fingerprint());

    let mut changed = base.clone();
    changed.output_format = OutputFormat::Jpeg;
    assert_ne!(fingerprint, changed.fingerprint());

    let mut changed = base;
    changed.determinism_class = DeterminismClass::BackendSeeded;
    assert_ne!(fingerprint, changed.fingerprint());
}

#[test]
fn equivalence_identity_scales_across_synthetic_compatible_fleets() {
    let (_root, config, request) = fixture();
    for count in [1, 2, 8, 64] {
        let devices = (0..count)
            .map(|index| device(&format!("cuda:{index}"), GpuBackend::Cuda, Some((8, 6))))
            .collect::<Vec<_>>();
        let plans = resolve_execution_plans(&config, &request, &devices, false).unwrap();
        assert_eq!(plans.len(), count);
        assert_eq!(
            plans
                .iter()
                .map(|plan| &plan.execution_equivalence_fingerprint)
                .collect::<std::collections::BTreeSet<_>>()
                .len(),
            1
        );
        assert_eq!(
            plans
                .iter()
                .map(|plan| &plan.execution_fingerprint)
                .collect::<std::collections::BTreeSet<_>>()
                .len(),
            count
        );
    }
}

#[test]
fn unknown_architecture_fails_closed_per_device() {
    let (_root, config, request) = fixture();
    let plans = resolve_execution_plans(
        &config,
        &request,
        &[
            device("cuda:unknown-0", GpuBackend::Cuda, None),
            device("cuda:unknown-1", GpuBackend::Cuda, None),
        ],
        false,
    )
    .unwrap();

    assert_ne!(
        plans[0].execution_equivalence_fingerprint,
        plans[1].execution_equivalence_fingerprint
    );
    assert!(matches!(
        plans[0].execution_environment.architecture,
        DeviceArchitectureClass::Unknown { .. }
    ));
}
