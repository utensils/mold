use mold_core::{
    AdvancedPlacement, Config, DevicePlacement, DeviceRef, GenerateRequest, GpuBackend,
    ModelConfig, OutputFormat, Scheduler,
};
use mold_server::execution_plan::{
    resolve_execution_plans, AttentionKernelClass, CanonicalRuntimeValue, ComponentLoadStrategy,
    ComponentRole, DeterminismClass, DeviceArchitectureClass, DeviceFact, EngineLoadStrategyClass,
    ExecutionSemanticConfig, OffloadMode, PlannedDType, QuantizationVariant,
    SemanticAttentionBackend, SemanticAttentionChunk, SemanticComponentPlacement, SemanticVaeDType,
    SemanticVaeTiling,
};
use std::io::Write;
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

fn write_single_tensor_safetensors(path: &Path, dtype: &str, bytes: usize) {
    let header = serde_json::to_vec(&serde_json::json!({
        "opaque.weight": {
            "dtype": dtype,
            "shape": [bytes],
            "data_offsets": [0, bytes]
        }
    }))
    .unwrap();
    let mut file = std::fs::File::create(path).unwrap();
    file.write_all(&(header.len() as u64).to_le_bytes())
        .unwrap();
    file.write_all(&header).unwrap();
    file.write_all(&vec![0_u8; bytes]).unwrap();
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
        .content_fingerprint =
        mold_server::execution_plan::EquivalenceContentIdentity::Sha256("different-assets".into());
    assert_ne!(fingerprint, changed.fingerprint());

    let mut changed = base.clone();
    changed.components.iter_mut().next().unwrap().dtype = Some(PlannedDType::F32);
    assert_ne!(fingerprint, changed.fingerprint());

    let mut changed = base.clone();
    changed
        .components
        .iter_mut()
        .find(|component| component.role == mold_server::execution_plan::ComponentRole::Transformer)
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

#[test]
fn frozen_engine_semantics_change_equivalence_identity() {
    let (_root, mut config, request) = fixture();
    let base = resolve_execution_plans(
        &config,
        &request,
        &[device("cuda:0", GpuBackend::Cuda, Some((8, 6)))],
        false,
    )
    .unwrap()
    .remove(0);

    config.models.get_mut("test:q4").unwrap().is_schnell = Some(true);
    let schnell = resolve_execution_plans(
        &config,
        &request,
        &[device("cuda:0", GpuBackend::Cuda, Some((8, 6)))],
        false,
    )
    .unwrap()
    .remove(0);

    assert_ne!(
        base.execution_equivalence_fingerprint, schnell.execution_equivalence_fingerprint,
        "a frozen engine semantic must participate in equivalence"
    );
}

#[test]
fn byte_identical_artifacts_at_different_paths_share_content_identity() {
    let (root, config, request) = fixture();
    let base = resolve_execution_plans(
        &config,
        &request,
        &[device("cuda:0", GpuBackend::Cuda, Some((8, 6)))],
        false,
    )
    .unwrap()
    .remove(0);

    let alias_root = root.path().join("alias");
    std::fs::create_dir(&alias_root).unwrap();
    for name in ["transformer-q4.gguf", "vae.safetensors", "t5.safetensors"] {
        std::fs::copy(root.path().join(name), alias_root.join(name)).unwrap();
    }
    let mut aliased_config = config;
    let aliased = aliased_config.models.get_mut("test:q4").unwrap();
    aliased.transformer = Some(path(&alias_root, "transformer-q4.gguf"));
    aliased.vae = Some(path(&alias_root, "vae.safetensors"));
    aliased.t5_encoder = Some(path(&alias_root, "t5.safetensors"));
    let alias = resolve_execution_plans(
        &aliased_config,
        &request,
        &[device("cuda:0", GpuBackend::Cuda, Some((8, 6)))],
        false,
    )
    .unwrap()
    .remove(0);

    assert_eq!(
        base.execution_equivalence_fingerprint, alias.execution_equivalence_fingerprint,
        "equivalence content identity must not depend on path or inode"
    );
}

#[test]
fn every_frozen_semantic_field_and_runtime_input_is_differential() {
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
    let base_fingerprint = base.fingerprint();

    macro_rules! assert_semantic_change {
        ($mutation:expr, $label:literal) => {{
            let mut changed = base.clone();
            let mutation: fn(&mut ExecutionSemanticConfig) = $mutation;
            mutation(&mut changed.semantic_config);
            assert_ne!(base_fingerprint, changed.fingerprint(), $label);
        }};
    }
    assert_semantic_change!(|value| value.family = "other".into(), "family");
    assert_semantic_change!(|value| value.is_schnell = Some(true), "schnell");
    assert_semantic_change!(|value| value.is_turbo = Some(true), "turbo");
    assert_semantic_change!(
        |value| value.scheduler = Some(Scheduler::UniPc),
        "scheduler"
    );
    assert_semantic_change!(|value| value.t5_variant = Some("q3".into()), "t5");
    assert_semantic_change!(|value| value.qwen3_variant = Some("q5".into()), "qwen3");
    assert_semantic_change!(|value| value.qwen2_variant = Some("q6".into()), "qwen2");
    assert_semantic_change!(
        |value| value.qwen2_text_encoder_mode = Some("legacy".into()),
        "qwen2 mode"
    );
    assert_semantic_change!(
        |value| value.ltx2_gemma_variant = Some("q4".into()),
        "gemma"
    );
    assert_semantic_change!(
        |value| value.attention_backend = SemanticAttentionBackend::Flash,
        "attention backend"
    );
    assert_semantic_change!(
        |value| value.attention_chunk = SemanticAttentionChunk::Size(17),
        "attention chunk"
    );
    assert_semantic_change!(
        |value| value.vae_tiling = SemanticVaeTiling::Force,
        "vae tiling"
    );
    assert_semantic_change!(|value| value.vae_dtype = SemanticVaeDType::F32, "vae dtype");

    assert_eq!(
        base.semantic_config.runtime.len(),
        mold_inference::runtime_env::ENGINE_SHAPING_VARIABLES.len(),
        "the runtime audit guard must classify every frozen variable"
    );
    let variables = base
        .semantic_config
        .runtime
        .iter()
        .map(|setting| setting.variable)
        .collect::<std::collections::BTreeSet<_>>();
    assert_eq!(variables.len(), base.semantic_config.runtime.len());
    for index in 0..base.semantic_config.runtime.len() {
        let mut changed = base.clone();
        changed.semantic_config.runtime[index].value =
            CanonicalRuntimeValue::Text(format!("differential-{index}"));
        assert_ne!(
            base_fingerprint,
            changed.fingerprint(),
            "runtime semantic input {index} was omitted"
        );
    }
}

#[test]
fn opaque_model_aliases_with_identical_bytes_share_equivalence_only() {
    let (root, mut config, mut request) = fixture();
    let model = config.models.remove("test:q4").unwrap();
    config.models.insert("cv:3143864".into(), model.clone());
    config
        .models
        .insert("hf:opaque-owner/opaque-repo".into(), model);

    request.model = "cv:3143864".into();
    let cv = resolve_execution_plans(
        &config,
        &request,
        &[device("cuda:0", GpuBackend::Cuda, Some((8, 6)))],
        false,
    )
    .unwrap()
    .remove(0);
    request.model = "hf:opaque-owner/opaque-repo".into();
    let hf = resolve_execution_plans(
        &config,
        &request,
        &[device("cuda:0", GpuBackend::Cuda, Some((8, 6)))],
        false,
    )
    .unwrap()
    .remove(0);

    assert_ne!(cv.execution_fingerprint, hf.execution_fingerprint);
    assert_eq!(
        cv.execution_equivalence_fingerprint,
        hf.execution_equivalence_fingerprint
    );
    assert_eq!(
        cv.execution_environment.components,
        hf.execution_environment.components
    );
    drop(root);
}

#[test]
fn opaque_headers_resolve_per_component_storage_and_effective_dtype() {
    use mold_inference::artifact_format::{
        ArtifactStorageFormat, SafetensorsEncoding, TensorDType,
    };
    use mold_server::execution_plan::{ComponentStorageFormat, EffectiveComponentDType};

    let root = TempDir::new().unwrap();
    write_single_tensor_safetensors(&root.path().join("transformer.asset"), "F8_E4M3", 1);
    write_single_tensor_safetensors(&root.path().join("vae.asset"), "BF16", 2);
    write_single_tensor_safetensors(&root.path().join("encoder.asset"), "F16", 2);
    let mut config = Config::default();
    config.models.insert(
        "cv:3143864".into(),
        ModelConfig {
            transformer: Some(path(root.path(), "transformer.asset")),
            vae: Some(path(root.path(), "vae.asset")),
            t5_encoder: Some(path(root.path(), "encoder.asset")),
            family: Some("flux".into()),
            ..ModelConfig::default()
        },
    );
    let request: GenerateRequest = serde_json::from_str(
        r#"{"prompt":"x","model":"cv:3143864","width":512,"height":512,"steps":4,"guidance":1.0}"#,
    )
    .unwrap();
    let plan = resolve_execution_plans(
        &config,
        &request,
        &[device("cuda:0", GpuBackend::Cuda, Some((8, 6)))],
        false,
    )
    .unwrap()
    .remove(0);

    let transformer = plan
        .execution_environment
        .components
        .iter()
        .find(|component| component.role == ComponentRole::Transformer)
        .unwrap();
    assert_eq!(transformer.quantization, Some(QuantizationVariant::Fp8));
    assert_eq!(
        transformer.precision.compute_dtype,
        EffectiveComponentDType::F16
    );
    assert_eq!(
        transformer.precision.storage,
        ComponentStorageFormat::Known(ArtifactStorageFormat::Safetensors {
            encoding: SafetensorsEncoding::Standard,
            tensor_dtypes: vec![TensorDType::F8E4M3],
        })
    );

    let vae = plan
        .execution_environment
        .components
        .iter()
        .find(|component| component.role == ComponentRole::Vae)
        .unwrap();
    assert_eq!(vae.quantization, None);
    assert_eq!(vae.precision.compute_dtype, EffectiveComponentDType::Bf16);

    let t5 = plan
        .execution_environment
        .components
        .iter()
        .find(|component| component.role == ComponentRole::T5)
        .unwrap();
    assert_eq!(t5.quantization, None);
    assert_eq!(t5.precision.compute_dtype, EffectiveComponentDType::Bf16);
}

#[test]
fn cpu_vae_effective_dtype_is_f32_and_differs_from_cuda() {
    use mold_server::execution_plan::EffectiveComponentDType;

    let root = TempDir::new().unwrap();
    write_single_tensor_safetensors(&root.path().join("transformer.asset"), "BF16", 2);
    write_single_tensor_safetensors(&root.path().join("vae.asset"), "BF16", 2);
    let mut config = Config::default();
    config.models.insert(
        "cv:opaque-z-image".into(),
        ModelConfig {
            transformer: Some(path(root.path(), "transformer.asset")),
            vae: Some(path(root.path(), "vae.asset")),
            family: Some("flux2".into()),
            ..ModelConfig::default()
        },
    );
    let mut request: GenerateRequest = serde_json::from_str(
        r#"{"prompt":"x","model":"cv:opaque-z-image","width":512,"height":512,"steps":4,"guidance":1.0}"#,
    )
    .unwrap();
    let gpu = resolve_execution_plans(
        &config,
        &request,
        &[device("cuda:0", GpuBackend::Cuda, Some((8, 6)))],
        false,
    )
    .unwrap()
    .remove(0);
    request.placement = Some(DevicePlacement {
        text_encoders: DeviceRef::Auto,
        advanced: Some(AdvancedPlacement {
            vae: DeviceRef::Cpu,
            ..AdvancedPlacement::default()
        }),
    });
    let cpu = resolve_execution_plans(
        &config,
        &request,
        &[device("cuda:0", GpuBackend::Cuda, Some((8, 6)))],
        false,
    )
    .unwrap()
    .remove(0);
    let vae_dtype = |plan: &mold_server::execution_plan::ResolvedExecutionPlan| {
        plan.execution_environment
            .components
            .iter()
            .find(|component| component.role == ComponentRole::Vae)
            .unwrap()
            .precision
            .compute_dtype
    };
    assert_eq!(vae_dtype(&gpu), EffectiveComponentDType::Bf16);
    assert_eq!(vae_dtype(&cpu), EffectiveComponentDType::F32);
    assert_ne!(
        gpu.execution_equivalence_fingerprint,
        cpu.execution_equivalence_fingerprint
    );
}
