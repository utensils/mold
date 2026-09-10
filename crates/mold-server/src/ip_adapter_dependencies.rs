//! Pre-admission materialization of the IP-Adapter image-prompt asset bundles.
//!
//! A reference-conditioned SD1.5 or SDXL render needs three artifacts that are
//! not part of any checkpoint's `ModelPaths`: the image-prompt adapter, the
//! OpenCLIP ViT-H/14 tower that turns the reference picture into a CLIP
//! embedding, and that tower's published `config.json`. They are resolved here,
//! beside the encoder ladders in [`crate::variant_dependencies`] and beside the
//! PuLID bundle in [`crate::identity_dependencies`], because they answer the
//! same question at the same moment — what must be on disk before the scheduler
//! may admit this job — and freeze into the same
//! [`mold_inference::FrozenEngineConfig`].
//!
//! It is a SIBLING of [`crate::identity_dependencies`] rather than a
//! generalization of it, for the reason `mold_core::ip_adapter_assets`' own
//! header records: the two bundles share a *pattern*, not a type. What differs
//! here, and the only material difference, is that **there is no licence gate**.
//! `h94/IP-Adapter` is Apache-2.0 and laion's CLIP-ViT-H-14 is MIT, so unlike
//! PuLID — whose antelopev2 graphs are research-only and are refused until a
//! human records an acceptance — nothing in this bundle needs one. The absence
//! is asserted rather than assumed: see
//! `the_bundle_needs_no_recorded_licence_acceptance` below.
//!
//! Everything in this module is inert unless the request actually conditions on
//! a reference picture. No `edit_images`, or `reference_weight` 0, plans
//! nothing, downloads nothing, freezes nothing, and charges nothing.

use std::path::PathBuf;

use mold_core::ip_adapter_assets::{ImagePromptFamily, IpAdapterPaths};
use mold_core::manifest::{ModelComponent, ModelFile, ModelManifest};
use mold_core::GenerateRequest;

use crate::execution_plan::PendingArtifactContainer;
use crate::variant_dependencies::{
    ensure_downloaded, DependencyContext, DependencySpec, MissingDependency, PinnedDigest,
};

/// The bundle this request's engine will condition with, or `None` when the
/// family has no image-prompt adapter at all.
///
/// Keyed on the resolved model FAMILY and on nothing else, because the family
/// is what selects the engine and therefore what decides which adapter the
/// checkpoint can accept: `image_proj.proj` emits `tokens * cross_attention_dim`
/// values, and SD1.5's 768 and SDXL's 2048 are not whole multiples of one
/// another, so the wrong file does not merely degrade — it fails to load.
///
/// Unlike the identity module's `identity_family_for` this is not a
/// `Result`. PuLID refuses an unqualified family because reaching it at all
/// means a request gate was bypassed; here the caller is
/// [`crate::variant_dependencies`]'s per-device loop, which runs for EVERY
/// family, and a FLUX render that happens to carry `edit_images` is an ordinary
/// FLUX.2-style reference request that this module has no opinion about.
/// `None` is "not mine", not "refused" — the refusal for a family that can
/// take neither is `validate_edit_images_against`'s, at the admission door.
fn image_prompt_family_for(family: &str) -> Option<ImagePromptFamily> {
    ImagePromptFamily::from_generation_family(family)
}

/// Whether this request will actually condition on a reference picture.
///
/// Presence of `edit_images` is not enough — an explicit `reference_weight` of
/// 0 injects nothing at all, so it must plan no dependency, start no download,
/// report no pending artifact, and freeze no paths. The same falsification case
/// `id_weight` 0 has, and it delegates to the request contract
/// (`mold_core::validation::request_conditions_on_reference`) so admission, the
/// estimate, and the engine cannot disagree about which requests are reference
/// requests.
#[cfg(test)]
pub(crate) fn request_needs_ip_adapter_assets(request: &GenerateRequest) -> bool {
    request_needs_ip_adapter_assets_with_projection(request, None)
}

/// The durable-queue form of `request_needs_ip_adapter_assets`.
///
/// A queued request's reference bytes were sealed into the encrypted media set
/// at admission and scrubbed from the row, so a re-prepared job carries an
/// empty `edit_images` and the projection is the only surviving evidence that
/// it had any. Reading the inline field alone would silently drop the bundle
/// from every replayed reference render.
pub(crate) fn request_needs_ip_adapter_assets_with_projection(
    request: &GenerateRequest,
    projection: Option<&crate::queue_media_store::QueueMediaProjection>,
) -> bool {
    mold_core::validation::request_conditions_on_reference(request)
        || (projection.is_some_and(|projection| projection.edit_image_count > 0)
            && mold_core::validation::effective_reference_weight(request) != 0.0)
}

/// What a client sees in `pending_downloads` for each asset.
///
/// Derived from the manifest component rather than from the filename so a
/// re-pinned release cannot rename a kind out from under a client. The names
/// are the components' own, so a client can tell an image-prompt adapter from a
/// face adapter and from a prompt encoder without parsing a path.
fn pending_kind(component: ModelComponent) -> Option<&'static str> {
    Some(match component {
        ModelComponent::ImagePromptAdapter => "image_prompt_adapter",
        ModelComponent::ImagePromptVisionEncoder => "image_prompt_vision_encoder",
        ModelComponent::ImagePromptVisionConfig => "image_prompt_vision_config",
        _ => return None,
    })
}

/// The container a preview may honestly claim without having read the file.
///
/// Both weight files are real safetensors — the tower is republished in
/// Hugging Face's `CLIPVisionModelWithProjection` layout and is loaded
/// directly, so unlike PuLID's EVA02-CLIP `.pt` there is no `TorchArchive`
/// conversion input anywhere in this bundle.
fn pending_container(component: ModelComponent) -> Option<PendingArtifactContainer> {
    Some(match component {
        ModelComponent::ImagePromptAdapter | ModelComponent::ImagePromptVisionEncoder => {
            PendingArtifactContainer::Safetensors
        }
        // A 560-byte `config.json`, the same class of small non-weight
        // artifact a tokenizer file is.
        ModelComponent::ImagePromptVisionConfig => PendingArtifactContainer::Raw,
        _ => return None,
    })
}

/// The bundle's storage directory, read off the manifest so this can never
/// drift from what [`mold_core::ip_adapter_assets`] resolves and what removal
/// deletes.
///
/// Unlike PuLID's four flat filenames, every file here carries an upstream
/// directory prefix (`models/`, `sdxl_models/`, `models/image_encoder/`), so
/// the three assets of one bundle land in two or three different subdirectories
/// under `shared/ip-adapter/`. That is the manifest's arrangement, not this
/// module's, and taking the parent of `storage_path` is what keeps the two
/// agreeing.
pub(crate) fn ip_adapter_storage_subdir(manifest: &ModelManifest, file: &ModelFile) -> String {
    mold_core::manifest::storage_path(manifest, file)
        .parent()
        .map(|parent| parent.to_string_lossy().into_owned())
        .unwrap_or_default()
}

/// Resolve the IP-Adapter bundle for one device's prepared inputs.
///
/// Mirrors the encoder ladders and the PuLID bundle exactly: under
/// [`DependencyMaterializationPolicy::Admission`] the files are downloaded;
/// under `ExistingOnly` (read-only placement preview) a missing file becomes a
/// pending download with its real bytes and nothing is started or refused. The
/// planned path is frozen either way, because that is the path admission will
/// land the file at.
///
/// There is deliberately no licence step between the family choice and the
/// download loop. That is not an omission — see the module header.
pub(crate) async fn materialize_ip_adapter_assets(
    context: &DependencyContext<'_>,
    request: &GenerateRequest,
    projection: Option<&crate::queue_media_store::QueueMediaProjection>,
    family: &str,
    frozen: &mut mold_inference::FrozenEngineConfig,
    pending: &mut Vec<MissingDependency>,
) -> Result<(), String> {
    if !request_needs_ip_adapter_assets_with_projection(request, projection) {
        return Ok(());
    }
    let Some(image_prompt_family) = image_prompt_family_for(family) else {
        return Ok(());
    };
    let manifest = mold_core::ip_adapter_assets::ip_adapter_manifest_for(image_prompt_family);

    let mut adapter: Option<PathBuf> = None;
    let mut vision_encoder: Option<PathBuf> = None;
    let mut vision_config: Option<PathBuf> = None;
    for file in &manifest.files {
        let kind = pending_kind(file.component).ok_or_else(|| {
            format!(
                "IP-Adapter asset '{}' has no client-facing component kind",
                file.hf_filename
            )
        })?;
        let container = pending_container(file.component).ok_or_else(|| {
            format!(
                "IP-Adapter asset '{}' has no declared container",
                file.hf_filename
            )
        })?;
        let pin = file.sha256.ok_or_else(|| {
            format!(
                "IP-Adapter asset '{}' has no pinned SHA-256; refusing to acquire an unpinned \
                 image-prompt artifact",
                file.hf_filename
            )
        })?;
        let subdir = ip_adapter_storage_subdir(manifest, file);
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
                // Neither weight file is quantized: upstream publishes the
                // adapter at its trained precision and the tower as the f32
                // half of laion's release. The `config.json` is not a weight
                // file at all.
                quantization: None,
                // Every IP-Adapter file is SHA-256 pinned in the manifest, and
                // this is the only place that pin is enforced for them: the
                // single-file downloader resolves the repo's mutable `main`
                // revision, so without it a replaced upstream file — or a
                // compromised mirror — would be frozen into the plan and
                // executed. The pin is required, never optional: an entry
                // without one is refused above rather than fetched unpinned.
                // It carries more weight here than it does for PuLID, because
                // this bundle has no licence gate to make a human look at the
                // acquisition first.
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
            ModelComponent::ImagePromptAdapter => adapter = Some(path),
            ModelComponent::ImagePromptVisionEncoder => vision_encoder = Some(path),
            ModelComponent::ImagePromptVisionConfig => vision_config = Some(path),
            other => {
                return Err(format!(
                    "the IP-Adapter manifest carries an unexpected component {other:?}"
                ))
            }
        }
    }

    let missing = |what: &str| format!("the IP-Adapter manifest is missing its {what}");
    frozen.ip_adapter_assets = Some(IpAdapterPaths {
        family: image_prompt_family,
        adapter: adapter.ok_or_else(|| missing("image-prompt adapter"))?,
        vision_encoder: vision_encoder.ok_or_else(|| missing("vision encoder"))?,
        vision_config: vision_config.ok_or_else(|| missing("vision encoder config"))?,
    });
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::execution_plan::DeviceFact;
    use crate::variant_dependencies::{
        prepare_inputs_for_devices, DependencyMaterializationPolicy, DependencyPreparationContext,
    };
    use mold_core::{Config, ModelConfig};
    use std::path::Path;
    use tempfile::TempDir;

    const SD15_MODEL: &str = "prepared-sd15";
    const SDXL_MODEL: &str = "prepared-sdxl";

    struct EnvGuard {
        _lock: std::sync::MutexGuard<'static, ()>,
        previous_home: Option<String>,
        previous_models: Option<String>,
    }

    impl EnvGuard {
        fn new(mold_home: &Path, models_dir: &Path) -> Self {
            let lock = crate::test_support::env_lock();
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

    /// An SD1.5 model config: one CLIP encoder, and a family that selects the
    /// 768-wide adapter.
    fn sd15_case(models_dir: &Path) -> (TempDir, Config) {
        let root = TempDir::new().unwrap();
        for name in [
            "unet.safetensors",
            "vae.safetensors",
            "clip_l.safetensors",
            "clip_l_tokenizer.json",
        ] {
            std::fs::write(root.path().join(name), b"prepared").unwrap();
        }
        let file = |name: &str| root.path().join(name).display().to_string();
        let mut config = Config {
            models_dir: models_dir.display().to_string(),
            ..Config::default()
        };
        config.models.insert(
            SD15_MODEL.to_string(),
            ModelConfig {
                transformer: Some(file("unet.safetensors")),
                vae: Some(file("vae.safetensors")),
                clip_encoder: Some(file("clip_l.safetensors")),
                clip_tokenizer: Some(file("clip_l_tokenizer.json")),
                family: Some("sd15".to_string()),
                ..ModelConfig::default()
            },
        );
        (root, config)
    }

    /// An SDXL model config: the dual encoders, and a family that selects the
    /// 2048-wide adapter.
    fn sdxl_case(models_dir: &Path) -> (TempDir, Config) {
        let root = TempDir::new().unwrap();
        for name in [
            "unet.safetensors",
            "vae.safetensors",
            "clip_l.safetensors",
            "clip_g.safetensors",
            "clip_l_tokenizer.json",
            "clip_g_tokenizer.json",
        ] {
            std::fs::write(root.path().join(name), b"prepared").unwrap();
        }
        let file = |name: &str| root.path().join(name).display().to_string();
        let mut config = Config {
            models_dir: models_dir.display().to_string(),
            ..Config::default()
        };
        config.models.insert(
            SDXL_MODEL.to_string(),
            ModelConfig {
                transformer: Some(file("unet.safetensors")),
                vae: Some(file("vae.safetensors")),
                clip_encoder: Some(file("clip_l.safetensors")),
                clip_encoder_2: Some(file("clip_g.safetensors")),
                clip_tokenizer: Some(file("clip_l_tokenizer.json")),
                clip_tokenizer_2: Some(file("clip_g_tokenizer.json")),
                family: Some("sdxl".to_string()),
                ..ModelConfig::default()
            },
        );
        (root, config)
    }

    /// A FLUX.2-style model config, so the "carries `edit_images` but has no
    /// image-prompt adapter" case is a real family rather than a string.
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
            t5_variant: Some("fp16".to_string()),
            ..Config::default()
        };
        config.models.insert(
            "prepared-flux".to_string(),
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

    fn request(
        model: &str,
        reference_weight: Option<f64>,
        with_reference: bool,
    ) -> GenerateRequest {
        let mut request: GenerateRequest = serde_json::from_str(
            r#"{"prompt":"a lighthouse","model":"placeholder","width":512,"height":512,"steps":20,"guidance":7.5}"#,
        )
        .unwrap();
        request.model = model.to_string();
        if with_reference {
            request.edit_images = Some(vec![vec![0x89, 0x50, 0x4e, 0x47]]);
        }
        request.reference_weight = reference_weight;
        request
    }

    fn device() -> DeviceFact {
        DeviceFact {
            cuda_peak_baseline: None,
            id: "cuda:0".to_string(),
            ordinal: 0,
            backend: mold_core::GpuBackend::Cuda,
            compute_capability: Some((8, 6)),
            available_vram_bytes: 24_000_000_000,
        }
    }

    fn expected_paths(models_dir: &Path, family: ImagePromptFamily) -> IpAdapterPaths {
        let manifest = mold_core::ip_adapter_assets::ip_adapter_manifest_for(family);
        let resolve = |component: ModelComponent| {
            let file = manifest
                .files
                .iter()
                .find(|file| file.component == component)
                .expect("the IP-Adapter manifest declares every component");
            models_dir.join(mold_core::manifest::storage_path(manifest, file))
        };
        IpAdapterPaths {
            family,
            adapter: resolve(ModelComponent::ImagePromptAdapter),
            vision_encoder: resolve(ModelComponent::ImagePromptVisionEncoder),
            vision_config: resolve(ModelComponent::ImagePromptVisionConfig),
        }
    }

    fn by_kind(
        downloads: &[mold_core::PendingModelDownload],
    ) -> std::collections::BTreeMap<&str, (&str, &str, u64)> {
        downloads
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
            .collect()
    }

    /// A read-only placement preview must report the whole bundle, with its
    /// pinned identities and real bytes, and must not touch the disk.
    #[tokio::test]
    #[allow(clippy::await_holding_lock)]
    async fn a_preview_plans_the_whole_bundle_without_downloading_it() {
        let models = TempDir::new().unwrap();
        let home = TempDir::new().unwrap();
        let _env = EnvGuard::new(home.path(), models.path());
        let (_root, config) = sdxl_case(models.path());

        let prepared = prepare_inputs_for_devices(
            None,
            "placement-preview",
            &request(SDXL_MODEL, None, true),
            &config,
            vec![device()],
            None,
            DependencyMaterializationPolicy::ExistingOnly,
            DependencyPreparationContext::default(),
        )
        .await
        .unwrap();

        let downloads = prepared.pending_downloads_for_device("cuda:0");
        assert_eq!(downloads.len(), 3, "{downloads:?}");
        assert!(downloads
            .iter()
            .all(|download| download.install_model.as_deref() == Some("ip-adapter-sdxl")));
        let kinds = by_kind(&downloads);
        assert_eq!(
            kinds["image_prompt_adapter"],
            (
                "h94/IP-Adapter",
                "sdxl_models/ip-adapter_sdxl_vit-h.safetensors",
                698_391_064
            )
        );
        assert_eq!(
            kinds["image_prompt_vision_encoder"],
            (
                "h94/IP-Adapter",
                "models/image_encoder/model.safetensors",
                2_528_373_448
            )
        );
        assert_eq!(
            kinds["image_prompt_vision_config"],
            ("h94/IP-Adapter", "models/image_encoder/config.json", 560)
        );

        let device_inputs = &prepared.by_device["cuda:0"];
        assert!(device_inputs
            .pending_artifacts
            .keys()
            .all(|path| !path.exists()));
        assert!(
            !models.path().join("shared/ip-adapter").exists(),
            "a read-only preview must not create the bundle's storage root"
        );
        // The planned paths are frozen even while pending, exactly as the
        // selected encoder path is: it is where admission will land the file.
        assert_eq!(
            device_inputs.engine_config.ip_adapter_assets,
            Some(expected_paths(models.path(), ImagePromptFamily::Sdxl))
        );

        // A pending safetensors/JSON dependency must never be described as a
        // quantized GGUF the preview has not read.
        for artifact in device_inputs.pending_artifacts.values() {
            assert_eq!(artifact.quantization, None, "{artifact:?}");
            assert!(
                !matches!(
                    artifact.container,
                    crate::execution_plan::PendingArtifactContainer::Gguf
                ),
                "{artifact:?}"
            );
        }
    }

    /// The bundle follows the family, and the SD1.5 one differs from the SDXL
    /// one by exactly its adapter — which is what makes "a machine that already
    /// holds one bundle pulls only the other's adapter" true rather than
    /// aspirational.
    #[tokio::test]
    #[allow(clippy::await_holding_lock)]
    async fn an_sd15_request_plans_its_own_adapter_and_the_shared_tower() {
        let models = TempDir::new().unwrap();
        let home = TempDir::new().unwrap();
        let _env = EnvGuard::new(home.path(), models.path());
        let (_root, config) = sd15_case(models.path());

        let prepared = prepare_inputs_for_devices(
            None,
            "placement-preview",
            &request(SD15_MODEL, None, true),
            &config,
            vec![device()],
            None,
            DependencyMaterializationPolicy::ExistingOnly,
            DependencyPreparationContext::default(),
        )
        .await
        .unwrap();

        let downloads = prepared.pending_downloads_for_device("cuda:0");
        let kinds = by_kind(&downloads);
        assert_eq!(
            kinds["image_prompt_adapter"],
            (
                "h94/IP-Adapter",
                "models/ip-adapter_sd15.safetensors",
                44_642_768
            )
        );
        // The tower is the SDXL bundle's, byte for byte and path for path.
        assert_eq!(
            kinds["image_prompt_vision_encoder"],
            (
                "h94/IP-Adapter",
                "models/image_encoder/model.safetensors",
                2_528_373_448
            )
        );

        let assets = prepared.by_device["cuda:0"]
            .engine_config
            .ip_adapter_assets
            .as_ref()
            .expect("the SD1.5 bundle is frozen into the plan");
        assert_eq!(assets.family, ImagePromptFamily::Sd15);
        assert!(assets
            .adapter
            .ends_with("shared/ip-adapter/models/ip-adapter_sd15.safetensors"));
        assert_eq!(
            assets.vision_encoder,
            expected_paths(models.path(), ImagePromptFamily::Sdxl).vision_encoder,
            "both bundles must share one tower on disk"
        );
    }

    /// `reference_weight` 0 injects nothing, so it must be completely inert —
    /// and so must a request that attached no reference at all. The two have to
    /// produce byte-identical prepared inputs.
    #[tokio::test]
    #[allow(clippy::await_holding_lock)]
    async fn weight_zero_is_indistinguishable_from_no_reference_at_all() {
        let models = TempDir::new().unwrap();
        let home = TempDir::new().unwrap();
        let _env = EnvGuard::new(home.path(), models.path());
        let (_root, config) = sdxl_case(models.path());

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

        let zero = prepare(request(SDXL_MODEL, Some(0.0), true)).await;
        let none = prepare(request(SDXL_MODEL, None, false)).await;

        for prepared in [&zero, &none] {
            assert!(prepared.pending_downloads_for_device("cuda:0").is_empty());
            let device_inputs = &prepared.by_device["cuda:0"];
            assert!(device_inputs.pending_artifacts.is_empty());
            assert_eq!(device_inputs.engine_config.ip_adapter_assets, None);
        }
        assert_eq!(
            zero.by_device["cuda:0"].engine_config,
            none.by_device["cuda:0"].engine_config
        );
        assert!(
            !models.path().join("shared/ip-adapter").exists(),
            "an inert reference request must not create the bundle's storage root"
        );
    }

    /// `edit_images` is not this module's field alone: FLUX.2 and
    /// Qwen-Image-Edit carry references natively and load none of this. A
    /// family with no image-prompt adapter must plan nothing rather than be
    /// refused — the refusal for a family that can take neither belongs to
    /// `validate_edit_images_against`, at the admission door.
    #[tokio::test]
    #[allow(clippy::await_holding_lock)]
    async fn a_family_with_no_image_prompt_adapter_plans_nothing_and_is_not_refused() {
        let models = TempDir::new().unwrap();
        let home = TempDir::new().unwrap();
        let _env = EnvGuard::new(home.path(), models.path());
        let (_root, config) = flux_case(models.path());

        let prepared = prepare_inputs_for_devices(
            None,
            "placement-preview",
            &request("prepared-flux", None, true),
            &config,
            vec![device()],
            None,
            DependencyMaterializationPolicy::ExistingOnly,
            DependencyPreparationContext::default(),
        )
        .await
        .expect("a reference on a family with no adapter is not this module's refusal");

        assert_eq!(
            prepared.by_device["cuda:0"].engine_config.ip_adapter_assets,
            None
        );
        assert!(!models.path().join("shared/ip-adapter").exists());
        assert_eq!(image_prompt_family_for("flux"), None);
        assert_eq!(image_prompt_family_for("qwen-image-edit"), None);
        assert_eq!(
            image_prompt_family_for("sd15"),
            Some(ImagePromptFamily::Sd15)
        );
        assert_eq!(
            image_prompt_family_for("sdxl"),
            Some(ImagePromptFamily::Sdxl)
        );
    }

    /// A durable request's reference bytes are sealed at admission and scrubbed
    /// from the row, so the projection is the only surviving evidence a
    /// replayed job had any. The hydrated and the projected forms must decide
    /// identically, or every replayed reference render loses its bundle.
    #[test]
    fn projected_reference_presence_matches_hydrated_dependency_need() {
        let hydrated = request(SDXL_MODEL, None, true);
        let mut sanitized = hydrated.clone();
        sanitized.edit_images = None;
        let projection = crate::queue_media_store::QueueMediaProjection {
            edit_image_count: 1,
            ..Default::default()
        };
        assert!(request_needs_ip_adapter_assets(&hydrated));
        assert_eq!(
            request_needs_ip_adapter_assets_with_projection(&hydrated, None),
            request_needs_ip_adapter_assets_with_projection(&sanitized, Some(&projection)),
        );

        // Weight zero is inert on both forms.
        let mut zero = sanitized.clone();
        zero.reference_weight = Some(0.0);
        assert!(!request_needs_ip_adapter_assets_with_projection(
            &zero,
            Some(&projection)
        ));
        // And an empty projection is inert even with a weight named.
        let mut weighted = sanitized.clone();
        weighted.reference_weight = Some(0.8);
        assert!(!request_needs_ip_adapter_assets_with_projection(
            &weighted, None
        ));
    }

    /// The bundle's planned destinations must be the same paths
    /// `mold_core::ip_adapter_assets` reports as installed and removal deletes.
    /// Two answers here means preparation downloads to one place while
    /// `mold list` and repair look at another.
    #[test]
    fn planned_paths_agree_with_the_installed_state_authority() {
        let models = TempDir::new().unwrap();
        let home = TempDir::new().unwrap();
        let _env = EnvGuard::new(home.path(), models.path());
        let config = Config {
            models_dir: models.path().display().to_string(),
            ..Config::default()
        };

        for family in ImagePromptFamily::ALL.iter().copied() {
            let manifest = mold_core::ip_adapter_assets::ip_adapter_manifest_for(family);
            let planned = manifest
                .files
                .iter()
                .map(|file| {
                    mold_core::download::planned_single_file_path_in(
                        models.path(),
                        &file.hf_filename,
                        &ip_adapter_storage_subdir(manifest, file),
                    )
                })
                .collect::<Vec<_>>();
            assert_eq!(
                planned,
                mold_core::ip_adapter_assets::ip_adapter_storage_paths_for(&config, family),
                "{family:?}"
            );
            assert!(
                manifest.files.iter().all(|file| {
                    ip_adapter_storage_subdir(manifest, file).starts_with("shared/ip-adapter")
                }),
                "{family:?} must live under the shared bundle root"
            );
        }
    }

    /// **The licence step is absent on purpose, and this is the pin.**
    ///
    /// `h94/IP-Adapter` is Apache-2.0 and laion's CLIP-ViT-H-14 is MIT, so
    /// unlike PuLID's antelopev2 graphs nothing in either bundle needs a
    /// recorded acceptance. That `license_acceptance`'s own
    /// `antelopev2_files_are_gated_and_nothing_else_is` keeps passing
    /// unchanged after this bundle landed is the correct outcome — and an
    /// invisible one, so it is asserted here from the other end: no
    /// IP-Adapter file is gated, no `pending_downloads` entry carries a
    /// licence, and a preview records no acceptance on the user's behalf.
    #[tokio::test]
    #[allow(clippy::await_holding_lock)]
    async fn the_bundle_needs_no_recorded_licence_acceptance() {
        for family in ImagePromptFamily::ALL.iter().copied() {
            let manifest = mold_core::ip_adapter_assets::ip_adapter_manifest_for(family);
            for file in &manifest.files {
                assert!(
                    !file.gated,
                    "{} must not require Hugging Face authentication",
                    file.hf_filename
                );
                assert!(
                    mold_core::license_acceptance::licenses_for_manifest_file(
                        &manifest.name,
                        &file.hf_filename
                    )
                    .next()
                    .is_none(),
                    "{} must carry no acceptance gate",
                    file.hf_filename
                );
            }
        }

        let models = TempDir::new().unwrap();
        let home = TempDir::new().unwrap();
        let _env = EnvGuard::new(home.path(), models.path());
        let (_root, config) = sdxl_case(models.path());

        let prepared = prepare_inputs_for_devices(
            None,
            "placement-preview",
            &request(SDXL_MODEL, None, true),
            &config,
            vec![device()],
            None,
            DependencyMaterializationPolicy::ExistingOnly,
            DependencyPreparationContext::default(),
        )
        .await
        .unwrap();

        assert!(prepared
            .pending_downloads_for_device("cuda:0")
            .iter()
            .all(|download| download.licenses.is_empty()));
        assert!(
            !mold_core::license_acceptance::acceptance_path(home.path()).exists(),
            "a preview must never record an acceptance on the user's behalf"
        );
    }

    /// Admission is the download path, and with no licence gate in front of it
    /// the SHA-256 pin is the only thing standing between a mutable upstream
    /// `main` and a frozen plan. Every file must carry one.
    #[test]
    fn every_bundle_file_is_content_pinned() {
        for family in ImagePromptFamily::ALL.iter().copied() {
            let manifest = mold_core::ip_adapter_assets::ip_adapter_manifest_for(family);
            for file in &manifest.files {
                assert!(
                    file.sha256.is_some(),
                    "{} must be SHA-256 pinned",
                    file.hf_filename
                );
                assert!(
                    pending_kind(file.component).is_some(),
                    "{} must have a client-facing kind",
                    file.hf_filename
                );
                assert!(
                    pending_container(file.component).is_some(),
                    "{} must declare a container",
                    file.hf_filename
                );
            }
        }
    }
}
