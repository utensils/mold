//! Pre-admission materialization of the PuLID face-identity asset bundle.
//!
//! A face-identity render needs four artifacts that are not part of any
//! checkpoint's `ModelPaths`: PuLID's identity adapter, the EVA02-CLIP vision
//! tower it encodes the reference face with, and InsightFace's SCRFD detector
//! and ArcFace recognizer. They are resolved here, beside the encoder ladders
//! in [`crate::variant_dependencies`], because they answer the same question
//! at the same moment — what must be on disk before the scheduler may admit
//! this job — and freeze into the same [`mold_inference::FrozenEngineConfig`].
//!
//! Everything in this module is inert unless the request actually conditions
//! on a face. `id_weight` 0 plans nothing, downloads nothing, freezes nothing,
//! and charges nothing.

use std::path::{Path, PathBuf};

use mold_core::manifest::{ModelComponent, ModelFile, ModelManifest};
use mold_core::pulid_assets::PulidPaths;
use mold_core::GenerateRequest;

use crate::execution_plan::PendingArtifactContainer;
use crate::variant_dependencies::{
    ensure_downloaded, DependencyContext, DependencyMaterializationPolicy, DependencySpec,
    MissingDependency,
};

/// The one family that can condition on an identity.
///
/// `mold_core::identity` already restricts identity conditioning to two
/// qualified FLUX checkpoints at the request boundary, so reaching this with
/// anything else means the gate was bypassed. Refusing is deliberate: an
/// identity that is accepted and then silently ignored is the failure slice A
/// exists to prevent.
const IDENTITY_FAMILY: &str = "flux";

/// Whether this request will actually condition on a face.
///
/// Presence of the fields is not enough — an explicit `id_weight` of 0 applies
/// no identity at all, so it must plan no dependency, start no download,
/// report no pending artifact, and freeze no paths.
pub(crate) fn request_needs_identity_assets(request: &GenerateRequest) -> bool {
    mold_core::identity::request_mentions_identity(request)
        && mold_core::identity::effective_id_weight(request) > 0.0
}

/// What a client sees in `pending_downloads` for each asset.
///
/// Derived from the manifest component rather than from the filename so a
/// re-pinned release cannot rename a kind out from under a client.
fn pending_kind(component: ModelComponent) -> Option<&'static str> {
    Some(match component {
        ModelComponent::IdentityAdapter => "identity_adapter",
        ModelComponent::IdentityVisionEncoder => "identity_vision_encoder",
        ModelComponent::FaceDetector => "face_detector",
        ModelComponent::FaceRecognizer => "face_recognizer",
        _ => return None,
    })
}

/// The container a preview may honestly claim without having read the file.
fn pending_container(component: ModelComponent) -> Option<PendingArtifactContainer> {
    Some(match component {
        ModelComponent::IdentityAdapter => PendingArtifactContainer::Safetensors,
        // A PyTorch pickle carried as a conversion input, never loaded as-is.
        ModelComponent::IdentityVisionEncoder => PendingArtifactContainer::TorchArchive,
        ModelComponent::FaceDetector | ModelComponent::FaceRecognizer => {
            PendingArtifactContainer::Onnx
        }
        _ => return None,
    })
}

/// The bundle's storage directory, read off the manifest so this can never
/// drift from what [`mold_core::pulid_assets`] resolves and what removal
/// deletes.
pub(crate) fn identity_storage_subdir(manifest: &ModelManifest, file: &ModelFile) -> String {
    mold_core::manifest::storage_path(manifest, file)
        .parent()
        .map(|parent| parent.to_string_lossy().into_owned())
        .unwrap_or_default()
}

/// Refuse an acquisition the user has not licensed, before a byte moves.
///
/// Only files that are actually missing are gated: a bundle already on disk
/// was acquired through a path that already asked. The check runs once for the
/// whole bundle so a refusal never leaves a 1.14 GB adapter downloaded beside
/// two files mold will not fetch.
fn require_identity_licenses(
    manifest: &ModelManifest,
    models_root: &Path,
    mold_home: Option<&Path>,
) -> Result<(), String> {
    for file in &manifest.files {
        let subdir = identity_storage_subdir(manifest, file);
        let present = mold_core::download::cached_file_path_in(
            models_root,
            &file.hf_repo,
            &file.hf_filename,
            Some(&subdir),
        )
        .is_some();
        if present {
            continue;
        }
        mold_core::download::require_license_accepted(&manifest.name, &file.hf_filename, mold_home)
            .map_err(|error| error.to_string())?;
    }
    Ok(())
}

/// Resolve the identity bundle for one device's prepared inputs.
///
/// Mirrors the encoder ladders exactly: under
/// [`DependencyMaterializationPolicy::Admission`] the files are downloaded;
/// under `ExistingOnly` (read-only placement preview) a missing file becomes a
/// pending download with its real bytes and nothing is started or refused. The
/// planned path is frozen either way, because that is the path admission will
/// land the file at.
pub(crate) async fn materialize_identity_assets(
    context: &DependencyContext<'_>,
    request: &GenerateRequest,
    family: &str,
    frozen: &mut mold_inference::FrozenEngineConfig,
    pending: &mut Vec<MissingDependency>,
) -> Result<(), String> {
    if !request_needs_identity_assets(request) {
        return Ok(());
    }
    if family != IDENTITY_FAMILY {
        return Err(format!(
            "identity conditioning is not supported by model family '{family}'"
        ));
    }

    let manifest = mold_core::pulid_assets::pulid_manifest();
    if context.policy == DependencyMaterializationPolicy::Admission {
        require_identity_licenses(
            manifest,
            context.models_root,
            mold_core::Config::mold_dir().as_deref(),
        )?;
    }

    let mut adapter: Option<PathBuf> = None;
    let mut vision_encoder_source: Option<PathBuf> = None;
    let mut face_detector: Option<PathBuf> = None;
    let mut face_recognizer: Option<PathBuf> = None;
    for file in &manifest.files {
        let kind = pending_kind(file.component).ok_or_else(|| {
            format!(
                "identity asset '{}' has no client-facing component kind",
                file.hf_filename
            )
        })?;
        let container = pending_container(file.component).ok_or_else(|| {
            format!(
                "identity asset '{}' has no declared container",
                file.hf_filename
            )
        })?;
        let subdir = identity_storage_subdir(manifest, file);
        let path = ensure_downloaded(
            context.state,
            context.work_id,
            DependencySpec {
                models_root: context.models_root,
                repo: &file.hf_repo,
                filename: &file.hf_filename,
                expected_bytes: Some(file.size_bytes),
                kind,
                container,
                // None of the four is quantized: the adapter and the vision
                // tower ship at their trained precision and the ONNX models
                // carry no GGUF-style variant at all.
                quantization: None,
                subdir: &subdir,
            },
            context.progress,
            context.policy,
        )
        .await?
        .into_path(pending);
        match file.component {
            ModelComponent::IdentityAdapter => adapter = Some(path),
            ModelComponent::IdentityVisionEncoder => vision_encoder_source = Some(path),
            ModelComponent::FaceDetector => face_detector = Some(path),
            ModelComponent::FaceRecognizer => face_recognizer = Some(path),
            other => {
                return Err(format!(
                    "the PuLID manifest carries an unexpected component {other:?}"
                ))
            }
        }
    }

    let missing = |what: &str| format!("the PuLID manifest is missing its {what}");
    frozen.identity_assets = Some(PulidPaths {
        adapter: adapter.ok_or_else(|| missing("identity adapter"))?,
        vision_encoder_source: vision_encoder_source
            .ok_or_else(|| missing("identity vision encoder"))?,
        face_detector: face_detector.ok_or_else(|| missing("face detector"))?,
        face_recognizer: face_recognizer.ok_or_else(|| missing("face recognizer"))?,
    });
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::execution_plan::DeviceFact;
    use crate::variant_dependencies::{prepare_inputs_for_devices, DependencyPreparationContext};
    use mold_core::{Config, ModelConfig};
    use tempfile::TempDir;

    const MODEL: &str = "prepared-flux";

    struct EnvGuard {
        _lock: std::sync::MutexGuard<'static, ()>,
        previous_home: Option<String>,
        previous_models: Option<String>,
    }

    impl EnvGuard {
        fn new(mold_home: &Path, models_dir: &Path) -> Self {
            let lock = crate::test_support::env_lock()
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner());
            let guard = Self {
                _lock: lock,
                previous_home: std::env::var("MOLD_HOME").ok(),
                previous_models: std::env::var("MOLD_MODELS_DIR").ok(),
            };
            std::env::set_var("MOLD_HOME", mold_home);
            std::env::set_var("MOLD_MODELS_DIR", models_dir);
            guard
        }
    }

    impl Drop for EnvGuard {
        fn drop(&mut self) {
            match &self.previous_home {
                Some(value) => std::env::set_var("MOLD_HOME", value),
                None => std::env::remove_var("MOLD_HOME"),
            }
            match &self.previous_models {
                Some(value) => std::env::set_var("MOLD_MODELS_DIR", value),
                None => std::env::remove_var("MOLD_MODELS_DIR"),
            }
        }
    }

    fn flux_case(models_dir: &Path) -> (TempDir, Config) {
        let root = TempDir::new().unwrap();
        for name in [
            "transformer.safetensors",
            "vae.safetensors",
            "t5.safetensors",
            "clip.safetensors",
            "t5_tokenizer.json",
            "clip_tokenizer.json",
        ] {
            std::fs::write(root.path().join(name), b"prepared").unwrap();
        }
        let file = |name: &str| root.path().join(name).display().to_string();
        let mut config = Config {
            models_dir: models_dir.display().to_string(),
            // Pin FP16 so the shared flux/T5 arm resolves from the manifest
            // encoder and this test measures the identity bundle alone.
            t5_variant: Some("fp16".to_string()),
            ..Config::default()
        };
        config.models.insert(
            MODEL.to_string(),
            ModelConfig {
                transformer: Some(file("transformer.safetensors")),
                vae: Some(file("vae.safetensors")),
                t5_encoder: Some(file("t5.safetensors")),
                clip_encoder: Some(file("clip.safetensors")),
                t5_tokenizer: Some(file("t5_tokenizer.json")),
                clip_tokenizer: Some(file("clip_tokenizer.json")),
                family: Some("flux".to_string()),
                ..ModelConfig::default()
            },
        );
        (root, config)
    }

    fn request(id_weight: Option<f64>, with_image: bool) -> GenerateRequest {
        let mut request: GenerateRequest = serde_json::from_str(
            r#"{"prompt":"a portrait","model":"prepared-flux","width":512,"height":512,"steps":20,"guidance":3.5}"#,
        )
        .unwrap();
        if with_image {
            request.id_image = Some(vec![0x89, 0x50, 0x4e, 0x47]);
        }
        request.id_weight = id_weight;
        request
    }

    fn device() -> DeviceFact {
        DeviceFact {
            id: "cuda:0".to_string(),
            ordinal: 0,
            backend: mold_core::GpuBackend::Cuda,
            compute_capability: Some((8, 6)),
            available_vram_bytes: 24_000_000_000,
        }
    }

    fn expected_paths(models_dir: &Path) -> PulidPaths {
        let manifest = mold_core::pulid_assets::pulid_manifest();
        let resolve = |component: ModelComponent| {
            let file = manifest
                .files
                .iter()
                .find(|file| file.component == component)
                .expect("the PuLID manifest declares every component");
            models_dir.join(mold_core::manifest::storage_path(manifest, file))
        };
        PulidPaths {
            adapter: resolve(ModelComponent::IdentityAdapter),
            vision_encoder_source: resolve(ModelComponent::IdentityVisionEncoder),
            face_detector: resolve(ModelComponent::FaceDetector),
            face_recognizer: resolve(ModelComponent::FaceRecognizer),
        }
    }

    /// The bundle's planned destinations must be the same paths
    /// `mold_core::pulid_assets` reports as installed and removal deletes. Two
    /// answers here means preparation downloads to one place while `mold list`
    /// and repair look at another.
    #[test]
    fn planned_paths_agree_with_the_installed_state_authority() {
        let models = TempDir::new().unwrap();
        let home = TempDir::new().unwrap();
        let _env = EnvGuard::new(home.path(), models.path());
        let config = Config {
            models_dir: models.path().display().to_string(),
            ..Config::default()
        };

        let manifest = mold_core::pulid_assets::pulid_manifest();
        let planned = manifest
            .files
            .iter()
            .map(|file| {
                mold_core::download::planned_single_file_path_in(
                    models.path(),
                    &file.hf_filename,
                    &identity_storage_subdir(manifest, file),
                )
            })
            .collect::<Vec<_>>();

        assert_eq!(
            planned,
            mold_core::pulid_assets::pulid_storage_paths(&config)
        );
        assert!(manifest
            .files
            .iter()
            .all(|file| identity_storage_subdir(manifest, file) == "shared/pulid"));
    }

    /// A read-only placement preview must report the whole bundle, with its
    /// pinned identities and real bytes, and must not touch the disk.
    #[tokio::test]
    #[allow(clippy::await_holding_lock)]
    async fn a_preview_plans_the_whole_bundle_without_downloading_it() {
        let models = TempDir::new().unwrap();
        let home = TempDir::new().unwrap();
        let _env = EnvGuard::new(home.path(), models.path());
        let (_root, config) = flux_case(models.path());

        let prepared = prepare_inputs_for_devices(
            None,
            "placement-preview",
            &request(None, true),
            &config,
            vec![device()],
            None,
            DependencyMaterializationPolicy::ExistingOnly,
            DependencyPreparationContext::default(),
        )
        .await
        .unwrap();

        let downloads = prepared.pending_downloads_for_device("cuda:0");
        assert_eq!(downloads.len(), 4, "{downloads:?}");
        let by_kind = downloads
            .iter()
            .map(|download| {
                (
                    download.kind.as_str(),
                    (
                        download.repo.as_str(),
                        download.name.as_str(),
                        download.bytes,
                    ),
                )
            })
            .collect::<std::collections::BTreeMap<_, _>>();
        assert_eq!(
            by_kind["identity_adapter"],
            (
                "guozinan/PuLID",
                "pulid_flux_v0.9.1.safetensors",
                1_142_099_520
            )
        );
        assert_eq!(
            by_kind["identity_vision_encoder"],
            (
                "QuanSun/EVA-CLIP",
                "EVA02_CLIP_L_336_psz14_s6B.pt",
                856_461_210
            )
        );
        assert_eq!(
            by_kind["face_detector"],
            (
                "DIAMONIK7777/antelopev2",
                "scrfd_10g_bnkps.onnx",
                16_923_827
            )
        );
        assert_eq!(
            by_kind["face_recognizer"],
            ("DIAMONIK7777/antelopev2", "glintr100.onnx", 260_665_334)
        );

        let device_inputs = &prepared.by_device["cuda:0"];
        assert!(device_inputs
            .pending_artifacts
            .keys()
            .all(|path| !path.exists()));
        assert!(
            !models.path().join("shared/pulid").exists(),
            "a read-only preview must not create the bundle's storage root"
        );
        // The planned paths are frozen even while pending, exactly as the
        // selected encoder path is: it is where admission will land the file.
        assert_eq!(
            device_inputs.engine_config.identity_assets,
            Some(expected_paths(models.path()))
        );

        // A pending safetensors/`.pt`/`.onnx` dependency must never be
        // described as a quantized GGUF the preview has not read.
        for identity in device_inputs.pending_artifacts.values() {
            assert_eq!(identity.quantization, None, "{identity:?}");
            assert!(
                !matches!(
                    identity.container,
                    crate::execution_plan::PendingArtifactContainer::Gguf
                ),
                "{identity:?}"
            );
        }

        // The device-resident half of the bundle is budgeted the way a pending
        // encoder is: bytes plus the shared preparation headroom.
        let plans = crate::execution_plan::resolve_execution_plans_with_prepared(
            &config,
            &request(None, true),
            &[device()],
            false,
            Some(&prepared),
        )
        .unwrap();
        assert_eq!(plans.len(), 1);
        assert!(
            plans[0].predicted_vram_peak_bytes
                >= 1_142_099_520 + crate::execution_plan::ENCODER_DEPENDENCY_HEADROOM_BYTES,
            "the pending identity adapter must be charged against the GPU peak"
        );
        let by_role = &plans[0].components;
        assert!(matches!(
            by_role[&crate::execution_plan::ComponentRole::FaceDetector].placement,
            crate::execution_plan::ResolvedComponentPlacement::Cpu
        ));
        assert!(matches!(
            by_role[&crate::execution_plan::ComponentRole::FaceRecognizer].placement,
            crate::execution_plan::ResolvedComponentPlacement::Cpu
        ));
        assert!(matches!(
            by_role[&crate::execution_plan::ComponentRole::IdentityAdapter].placement,
            crate::execution_plan::ResolvedComponentPlacement::Device(_)
        ));
    }

    /// An installed bundle plans no download at all and freezes the concrete
    /// paths the worker will construct the engine from.
    #[tokio::test]
    #[allow(clippy::await_holding_lock)]
    async fn an_installed_bundle_is_frozen_and_plans_nothing() {
        let models = TempDir::new().unwrap();
        let home = TempDir::new().unwrap();
        let _env = EnvGuard::new(home.path(), models.path());
        let (_root, config) = flux_case(models.path());
        for path in mold_core::pulid_assets::pulid_storage_paths(&config) {
            std::fs::create_dir_all(path.parent().unwrap()).unwrap();
            std::fs::write(&path, b"identity asset").unwrap();
        }

        let prepared = prepare_inputs_for_devices(
            None,
            "installed",
            &request(None, true),
            &config,
            vec![device()],
            None,
            DependencyMaterializationPolicy::ExistingOnly,
            DependencyPreparationContext::default(),
        )
        .await
        .unwrap();

        assert!(prepared.pending_downloads_for_device("cuda:0").is_empty());
        let device_inputs = &prepared.by_device["cuda:0"];
        assert!(device_inputs.pending_artifacts.is_empty());
        assert_eq!(
            device_inputs.engine_config.identity_assets,
            Some(expected_paths(models.path()))
        );
    }

    /// `id_weight` 0 applies no identity, so it must be completely inert — and
    /// so must a request that never mentioned identity at all. The two have to
    /// produce byte-identical prepared inputs.
    #[tokio::test]
    #[allow(clippy::await_holding_lock)]
    async fn weight_zero_is_indistinguishable_from_no_identity_at_all() {
        let models = TempDir::new().unwrap();
        let home = TempDir::new().unwrap();
        let _env = EnvGuard::new(home.path(), models.path());
        let (_root, config) = flux_case(models.path());

        let prepare = |request: GenerateRequest| {
            let config = config.clone();
            async move {
                prepare_inputs_for_devices(
                    None,
                    "inert",
                    &request,
                    &config,
                    vec![device()],
                    None,
                    DependencyMaterializationPolicy::ExistingOnly,
                    DependencyPreparationContext::default(),
                )
                .await
                .unwrap()
            }
        };

        let zero = prepare(request(Some(0.0), true)).await;
        let none = prepare(request(None, false)).await;

        for prepared in [&zero, &none] {
            assert!(prepared.pending_downloads_for_device("cuda:0").is_empty());
            let device_inputs = &prepared.by_device["cuda:0"];
            assert!(device_inputs.pending_artifacts.is_empty());
            assert_eq!(device_inputs.engine_config.identity_assets, None);
        }
        assert_eq!(
            zero.by_device["cuda:0"].engine_config,
            none.by_device["cuda:0"].engine_config
        );
        assert!(
            !models.path().join("shared/pulid").exists(),
            "an inert identity request must not create the bundle's storage root"
        );
    }

    /// The InsightFace models are non-commercial-research-only weights. An
    /// admission that would fetch them without a recorded acceptance must fail
    /// with the actionable message, before a byte of the whole bundle moves —
    /// including the two files that are not themselves gated.
    #[tokio::test]
    #[allow(clippy::await_holding_lock)]
    async fn admission_refuses_the_bundle_until_the_insightface_license_is_accepted() {
        let models = TempDir::new().unwrap();
        let home = TempDir::new().unwrap();
        let _env = EnvGuard::new(home.path(), models.path());
        let (_root, config) = flux_case(models.path());

        let error = prepare_inputs_for_devices(
            None,
            "admission",
            &request(None, true),
            &config,
            vec![device()],
            None,
            DependencyMaterializationPolicy::Admission,
            DependencyPreparationContext::default(),
        )
        .await
        .expect_err("an unaccepted license must refuse admission");

        assert!(error.contains("insightface-antelopev2"), "{error}");
        assert!(error.contains("--accept-license"), "{error}");
        assert!(
            !models.path().join("shared/pulid").exists(),
            "a refused acquisition must not have started a download"
        );

        // Recording the acceptance is what unblocks it. The bundle is present
        // here so the test stays offline; what it proves is that the gate
        // itself no longer refuses.
        mold_core::license_acceptance::record_acceptance(
            home.path(),
            &mold_core::license_acceptance::INSIGHTFACE_ANTELOPEV2,
        )
        .unwrap();
        for path in mold_core::pulid_assets::pulid_storage_paths(&config) {
            std::fs::create_dir_all(path.parent().unwrap()).unwrap();
            std::fs::write(&path, b"identity asset").unwrap();
        }
        let prepared = prepare_inputs_for_devices(
            None,
            "admission",
            &request(None, true),
            &config,
            vec![device()],
            None,
            DependencyMaterializationPolicy::Admission,
            DependencyPreparationContext::default(),
        )
        .await
        .expect("an accepted license admits the request");
        assert_eq!(
            prepared.by_device["cuda:0"].engine_config.identity_assets,
            Some(expected_paths(models.path()))
        );
    }

    /// A read-only preview is not the refusal point: it must report the gated
    /// files as ordinary pending downloads with their real bytes rather than
    /// erroring or silently omitting them. Admission is where the license is
    /// enforced.
    #[tokio::test]
    #[allow(clippy::await_holding_lock)]
    async fn a_preview_reports_gated_files_without_refusing_or_accepting_anything() {
        let models = TempDir::new().unwrap();
        let home = TempDir::new().unwrap();
        let _env = EnvGuard::new(home.path(), models.path());
        let (_root, config) = flux_case(models.path());

        let prepared = prepare_inputs_for_devices(
            None,
            "placement-preview",
            &request(None, true),
            &config,
            vec![device()],
            None,
            DependencyMaterializationPolicy::ExistingOnly,
            DependencyPreparationContext::default(),
        )
        .await
        .unwrap();

        let downloads = prepared.pending_downloads_for_device("cuda:0");
        assert!(downloads
            .iter()
            .any(|download| download.name == "scrfd_10g_bnkps.onnx"));
        assert!(downloads
            .iter()
            .any(|download| download.name == "glintr100.onnx"));
        assert!(
            !mold_core::license_acceptance::acceptance_path(home.path()).exists(),
            "a preview must never record an acceptance on the user's behalf"
        );
    }
}
