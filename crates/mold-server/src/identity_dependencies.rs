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
    MissingDependency, PinnedDigest,
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
        let pin = file.sha256.ok_or_else(|| {
            format!(
                "identity asset '{}' has no pinned SHA-256; refusing to acquire an unpinned \
                 identity artifact",
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
                // Every PuLID file is SHA-256 pinned in the manifest, and this
                // is the only place that pin is enforced for them: the
                // single-file downloader resolves the repo's mutable `main`
                // revision, so without it a replaced upstream file — or a
                // compromised mirror — would be frozen into the plan and
                // executed. The pin is required, never optional: an entry
                // without one is refused below rather than fetched unpinned.
                expected_sha256: Some(PinnedDigest {
                    sha256: pin,
                    repair_model: &manifest.name,
                }),
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

    /// Place a stand-in for every bundle file, each carrying a
    /// `.sha256-verified` marker that names the manifest's pinned digest while
    /// the bytes beside it are something else entirely.
    ///
    /// This is not a fixture for "correctly installed" — no test can produce
    /// a 1.14 GB preimage of a fixed digest. It is the ATTACK: a group-writable
    /// models root, which the model-storage invariant explicitly supports, lets
    /// anyone who can drop weights also drop the sidecar that vouches for them.
    /// Every use below asserts that mold refuses it.
    fn install_forged_bundle(config: &Config) {
        let manifest = mold_core::pulid_assets::pulid_manifest();
        let models_dir = config.resolved_models_dir();
        for file in &manifest.files {
            let path = models_dir.join(mold_core::manifest::storage_path(manifest, file));
            std::fs::create_dir_all(path.parent().unwrap()).unwrap();
            std::fs::write(&path, b"identity asset").unwrap();
            mold_core::download::write_sha256_marker(&path, file.sha256.unwrap()).unwrap();
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

    /// A bundle that is merely PRESENT is not installed. Presence plus a
    /// self-served attestation is what an attacker with write access to a
    /// shared models root can manufacture; only bytes that hash to the
    /// manifest pin count.
    ///
    /// The read-only preview says so by planning the download anyway, and
    /// admission says so by refusing. Neither is allowed to read the sidecar
    /// and call it installed.
    #[tokio::test]
    #[allow(clippy::await_holding_lock)]
    async fn a_forged_attestation_never_makes_a_bundle_count_as_installed() {
        let models = TempDir::new().unwrap();
        let home = TempDir::new().unwrap();
        let _env = EnvGuard::new(home.path(), models.path());
        let (_root, config) = flux_case(models.path());
        install_forged_bundle(&config);

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

        assert_eq!(
            prepared.pending_downloads_for_device("cuda:0").len(),
            4,
            "unproven bytes are not evidence that nothing needs downloading"
        );
        // The preview stays read-only about them: nothing deleted, nothing
        // attested, nothing refused.
        for path in mold_core::pulid_assets::pulid_storage_paths(&config) {
            assert!(path.exists(), "{}", path.display());
        }
        // The planned paths are still frozen — they are where admission will
        // land the real bytes.
        assert_eq!(
            prepared.by_device["cuda:0"].engine_config.identity_assets,
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

        // Recording the acceptance is what clears the gate. This half is a
        // direct call rather than another admission: with the gate open and
        // no files on disk, admission's next move is a real download, and a
        // forged bundle can no longer stand in for one (which is the point of
        // the pinned check).
        let manifest = mold_core::pulid_assets::pulid_manifest();
        assert!(require_identity_licenses(manifest, models.path(), Some(home.path())).is_err());
        mold_core::license_acceptance::record_acceptance(
            home.path(),
            &mold_core::license_acceptance::INSIGHTFACE_ANTELOPEV2,
        )
        .unwrap();
        require_identity_licenses(manifest, models.path(), Some(home.path()))
            .expect("a recorded acceptance clears the gate");
        // An unresolvable Mold data root stays closed: unverifiable is not
        // accepted.
        assert!(require_identity_licenses(manifest, models.path(), None).is_err());
    }

    /// Every identity asset is SHA-256 pinned in the manifest, and this is the
    /// only place that pin is enforced for them — the single-file downloader
    /// resolves the repo's mutable `main` revision, and the frozen plan proves
    /// only that the path is local. A file on disk that is not the pinned
    /// bytes must therefore fail admission by name, be removed, and never be
    /// frozen into an execution plan.
    #[tokio::test]
    #[allow(clippy::await_holding_lock)]
    async fn a_tampered_identity_asset_fails_admission_and_is_removed() {
        let models = TempDir::new().unwrap();
        let home = TempDir::new().unwrap();
        let _env = EnvGuard::new(home.path(), models.path());
        let (_root, config) = flux_case(models.path());
        mold_core::license_acceptance::record_acceptance(
            home.path(),
            &mold_core::license_acceptance::INSIGHTFACE_ANTELOPEV2,
        )
        .unwrap();
        install_forged_bundle(&config);

        // The adapter's bytes are not its pin, and its forged marker is left
        // deliberately IN PLACE — the whole point is that the sidecar buys the
        // attacker nothing.
        let manifest = mold_core::pulid_assets::pulid_manifest();
        let adapter_file = manifest
            .files
            .iter()
            .find(|file| file.component == ModelComponent::IdentityAdapter)
            .unwrap();
        let adapter = models
            .path()
            .join(mold_core::manifest::storage_path(manifest, adapter_file));
        assert_eq!(
            mold_core::download::recorded_sha256_marker(&adapter).as_deref(),
            adapter_file.sha256,
            "the fixture must present a marker that vouches for the pin"
        );

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
        .expect_err("a tampered identity asset must fail admission");

        assert!(error.contains("pulid_flux_v0.9.1.safetensors"), "{error}");
        assert!(error.contains(adapter_file.sha256.unwrap()), "{error}");
        assert!(error.contains("mold pull pulid-flux"), "{error}");
        assert!(
            !adapter.exists(),
            "the rejected asset must be removed so a repair re-downloads it"
        );

        // The other three assets are untouched: only the file that failed its
        // own pin is rejected.
        for file in &manifest.files {
            if file.component == ModelComponent::IdentityAdapter {
                continue;
            }
            let path = models
                .path()
                .join(mold_core::manifest::storage_path(manifest, file));
            assert!(path.exists(), "{}", file.hf_filename);
        }
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

    /// A second device, so the fan-out cases below are genuinely multi-device.
    fn second_device() -> DeviceFact {
        DeviceFact {
            id: "cuda:1".to_string(),
            ordinal: 1,
            backend: mold_core::GpuBackend::Cuda,
            compute_capability: Some((8, 6)),
            available_vram_bytes: 24_000_000_000,
        }
    }

    /// The core of #1223's extraction lifetime, at the one place it could
    /// break: the scheduler re-prepares dependencies for EVERY pending job,
    /// batch children included, so a child arriving with its parent's frozen
    /// identity must reuse it verbatim and extract nothing — on every device
    /// it fans out to.
    #[tokio::test]
    #[allow(clippy::await_holding_lock)]
    async fn a_batch_child_reuses_the_parents_identity_across_every_device() {
        let models = TempDir::new().unwrap();
        let home = TempDir::new().unwrap();
        let _env = EnvGuard::new(home.path(), models.path());
        let (_root, config) = flux_case(models.path());

        let frozen = crate::identity_extraction::stub_embedding(b"parent-face");
        let stubbed = crate::identity_extraction::StubbedExtractor::install(|_, image| {
            Ok(crate::identity_extraction::ResolvedIdentity {
                embedding: Some(crate::identity_extraction::stub_embedding(image)),
                warning: None,
            })
        });

        let prepared = prepare_inputs_for_devices(
            None,
            "batch-child",
            &request(None, true),
            &config,
            vec![device(), second_device()],
            None,
            DependencyMaterializationPolicy::ExistingOnly,
            // Without an `h3` feature this struct has exactly one field, so the
            // update base is a no-op there; it only carries the cfg-gated fields
            // when the build actually has them.
            #[allow(clippy::needless_update)]
            DependencyPreparationContext {
                frozen_identity: Some(frozen.clone()),
                ..Default::default()
            },
        )
        .await
        .unwrap();

        assert_eq!(
            stubbed.extractions(),
            0,
            "a child must never re-extract the identity its parent already froze"
        );
        assert_eq!(
            prepared
                .identity_embedding
                .as_ref()
                .map(|e| e.fingerprint()),
            Some(frozen.fingerprint()),
            "the child must carry the parent's exact identity, not an equivalent one"
        );
        assert_eq!(
            prepared.by_device.len(),
            2,
            "the fan-out under test must actually be multi-device"
        );
    }

    /// Placement preview is a read-only probe. Running the extractor there
    /// would cost seconds and ~1.4 GB of host RAM for an answer that does not
    /// depend on the embedding — `memory_preflight` charges identity from the
    /// request, never from the extracted value.
    #[tokio::test]
    #[allow(clippy::await_holding_lock)]
    async fn a_placement_preview_extracts_nothing() {
        let models = TempDir::new().unwrap();
        let home = TempDir::new().unwrap();
        let _env = EnvGuard::new(home.path(), models.path());
        let (_root, config) = flux_case(models.path());

        let stubbed = crate::identity_extraction::StubbedExtractor::install(|_, image| {
            Ok(crate::identity_extraction::ResolvedIdentity {
                embedding: Some(crate::identity_extraction::stub_embedding(image)),
                warning: None,
            })
        });

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

        assert_eq!(stubbed.extractions(), 0);
        assert!(prepared.identity_embedding.is_none());
        // The bundle is still PLANNED — the probe reports what admission will
        // fetch; it simply does not run the extractor over it.
        assert_eq!(
            prepared.by_device["cuda:0"].engine_config.identity_assets,
            Some(expected_paths(models.path()))
        );
    }

    /// #1223's headline requirement, in the shape it actually occurs: ONE
    /// parent request, `batch_size = 4`, two devices — exactly ONE extraction,
    /// and all four children conditioned on the identical value.
    ///
    /// This walks the real sequence rather than asserting that a clone equals
    /// itself. `freeze_batch_plan` prepares the parent once, and the scheduler
    /// then re-prepares EVERY child; the child preparations are the ones that
    /// would re-extract, so each is driven here with the frozen identity the
    /// parent produced, exactly as `start_needed_preparations` supplies it.
    #[tokio::test]
    #[allow(clippy::await_holding_lock)]
    async fn one_parent_request_extracts_once_for_four_children_on_two_devices() {
        let models = TempDir::new().unwrap();
        let home = TempDir::new().unwrap();
        let _env = EnvGuard::new(home.path(), models.path());
        let (_root, config) = flux_case(models.path());

        let stubbed = crate::identity_extraction::StubbedExtractor::install(|_, image| {
            Ok(crate::identity_extraction::ResolvedIdentity {
                embedding: Some(crate::identity_extraction::stub_embedding(image)),
                warning: None,
            })
        });

        // The parent. `freeze_batch_plan` validates a `batch_size = 1` clone,
        // which is what reaches preparation, so the request here matches.
        let parent = crate::identity_extraction::resolve_identity_embedding(
            &request(None, true),
            Some(&expected_paths(models.path())),
        )
        .await
        .expect("the stub extractor answers")
        .embedding
        .expect("a conditioned parent resolves an identity");
        assert_eq!(stubbed.extractions(), 1);

        // The four children, each re-prepared by the scheduler across both
        // devices.
        let mut fingerprints = std::collections::BTreeSet::new();
        for index in 1..=4u32 {
            let mut child = request(None, true);
            child.batch_id = Some("parent".to_string());
            child.batch_index = Some(index);
            child.batch_count = Some(4);
            let prepared = prepare_inputs_for_devices(
                None,
                &format!("child-{index}"),
                &child,
                &config,
                vec![device(), second_device()],
                None,
                DependencyMaterializationPolicy::ExistingOnly,
                // Without an `h3` feature this struct has exactly one field, so the
                // update base is a no-op there; it only carries the cfg-gated fields
                // when the build actually has them.
                #[allow(clippy::needless_update)]
                DependencyPreparationContext {
                    frozen_identity: Some(parent.clone()),
                    ..Default::default()
                },
            )
            .await
            .unwrap();
            fingerprints.insert(
                prepared
                    .identity_embedding
                    .expect("every child carries the parent's identity")
                    .fingerprint()
                    .to_string(),
            );
        }

        assert_eq!(
            stubbed.extractions(),
            1,
            "four children across two devices must not add a single extraction"
        );
        assert_eq!(
            fingerprints,
            std::collections::BTreeSet::from([parent.fingerprint().to_string()]),
            "every sibling must condition on the parent's exact identity"
        );
    }

    /// A re-prepared batch child must not invent a second copy of its
    /// parent's advisory. The child never extracts, so it has nothing of its
    /// own to report; the note reaches it by cloning the parent's prepared
    /// inputs, and `merge_prepared` carries it across re-preparation.
    #[tokio::test]
    #[allow(clippy::await_holding_lock)]
    async fn a_reused_identity_reports_no_advisory_of_its_own() {
        let models = TempDir::new().unwrap();
        let home = TempDir::new().unwrap();
        let _env = EnvGuard::new(home.path(), models.path());
        let (_root, config) = flux_case(models.path());

        let _stubbed = crate::identity_extraction::StubbedExtractor::install(|_, image| {
            Ok(crate::identity_extraction::ResolvedIdentity {
                embedding: Some(crate::identity_extraction::stub_embedding(image)),
                warning: Some("3 faces were detected".to_string()),
            })
        });

        let prepared = prepare_inputs_for_devices(
            None,
            "child",
            &request(None, true),
            &config,
            vec![device()],
            None,
            DependencyMaterializationPolicy::ExistingOnly,
            // Without an `h3` feature this struct has exactly one field, so the
            // update base is a no-op there; it only carries the cfg-gated fields
            // when the build actually has them.
            #[allow(clippy::needless_update)]
            DependencyPreparationContext {
                frozen_identity: Some(crate::identity_extraction::stub_embedding(b"parent-face")),
                ..Default::default()
            },
        )
        .await
        .unwrap();

        assert!(prepared.identity_embedding.is_some());
        assert!(
            prepared.identity_warning.is_none(),
            "a sibling that did not extract has no advisory of its own to add"
        );
    }

    /// The zero-weight rule, end to end through preparation: no assets, no
    /// embedding, and byte-identical prepared inputs to a request that never
    /// mentioned identity.
    #[tokio::test]
    #[allow(clippy::await_holding_lock)]
    async fn weight_zero_freezes_no_identity_embedding() {
        let models = TempDir::new().unwrap();
        let home = TempDir::new().unwrap();
        let _env = EnvGuard::new(home.path(), models.path());
        let (_root, config) = flux_case(models.path());

        let stubbed = crate::identity_extraction::StubbedExtractor::install(|_, image| {
            Ok(crate::identity_extraction::ResolvedIdentity {
                embedding: Some(crate::identity_extraction::stub_embedding(image)),
                warning: None,
            })
        });

        let prepared = prepare_inputs_for_devices(
            None,
            "inert",
            &request(Some(0.0), true),
            &config,
            vec![device()],
            None,
            DependencyMaterializationPolicy::ExistingOnly,
            DependencyPreparationContext::default(),
        )
        .await
        .unwrap();

        assert_eq!(stubbed.extractions(), 0);
        assert!(prepared.identity_embedding.is_none());
    }
}
