//! On-disk resolution for the PuLID auxiliary asset bundles.
//!
//! A bundle is five unrelated artifacts — an identity adapter, a vision tower,
//! a face detector, a face recognizer, and a face parser — and none of them is
//! a transformer or a VAE. [`crate::manifest::paths_from_downloads`] therefore
//! cannot represent it: it exists to build a [`crate::ModelPaths`], which
//! requires a generator. This module resolves the bundle on its own terms
//! instead, and answers only one question: are all five files present, and
//! where?
//!
//! There are two bundles, one per [`IdentityFamily`], and they differ in
//! exactly one file. The adapter is family-specific — PuLID-FLUX v0.9.1 for
//! FLUX, PuLID v1.1 for SDXL — while the four EXTRACTION artifacts are shared,
//! because upstream's `pipeline_v1_1.py:get_id_embedding` runs the same
//! detector, recognizer, parser, and vision tower as `pipeline_flux.py`'s.
//! Both manifests carry the same `pulid` family and the same four
//! non-model-specific components, so [`crate::manifest::storage_path`] lands
//! every one of them at the identical `shared/pulid/` path: a machine holding
//! one bundle pulls only the other's adapter.

use std::path::PathBuf;

use crate::config::Config;
use crate::identity::IdentityFamily;
use crate::manifest::{find_manifest, ModelComponent, ModelFile, ModelManifest};

/// Concrete, verified-complete paths to every PuLID asset.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PulidPaths {
    /// Which adapter [`Self::adapter`] is, and therefore which IDFormer prefix
    /// reads it and which injection sites consume it.
    ///
    /// Carried rather than inferred from the filename: the extractor has to
    /// pick `id_adapter` or `pulid_encoder` before it opens the file, and the
    /// cache key has to pin the family's own manifest digest. A path string is
    /// not a contract.
    pub family: IdentityFamily,
    /// PuLID's identity adapter (IDFormer + cross-attention weights).
    pub adapter: PathBuf,
    /// The EVA02-CLIP-L-14-336 `.pt` release. A conversion INPUT — callers
    /// must not hand this to a safetensors loader.
    pub vision_encoder_source: PathBuf,
    /// InsightFace antelopev2 SCRFD detector.
    pub face_detector: PathBuf,
    /// InsightFace antelopev2 ArcFace recognizer.
    pub face_recognizer: PathBuf,
    /// facexlib's BiSeNet face parser. A conversion INPUT, like
    /// [`Self::vision_encoder_source`] — callers must not hand this `.pth` to
    /// a safetensors loader.
    pub face_parser_source: PathBuf,
}

/// The manifest for one family's bundle.
///
/// Panics only if the manifest registry lost the entry, which a completeness
/// test in `manifest.rs` makes impossible.
pub fn pulid_manifest_for(family: IdentityFamily) -> &'static ModelManifest {
    find_manifest(family.manifest()).expect("every PuLID manifest is registered")
}

/// Every registered PuLID bundle manifest, in [`IdentityFamily::ALL`] order.
pub fn pulid_manifests() -> Vec<&'static ModelManifest> {
    IdentityFamily::ALL
        .iter()
        .copied()
        .map(pulid_manifest_for)
        .collect()
}

fn file_for(
    manifest: &'static ModelManifest,
    component: ModelComponent,
) -> Option<&'static ModelFile> {
    manifest
        .files
        .iter()
        .find(|file| file.component == component)
}

/// Resolve one family's bundle, returning `Some` only when **all five** files
/// are completely on disk.
///
/// A partially present bundle is deliberately `None` rather than a struct with
/// holes: PuLID cannot condition on an identity without every one of these
/// artifacts, so a caller holding a `PulidPaths` holds a runnable bundle.
/// Use [`missing_pulid_files_for`] to report what a repair still needs.
pub fn pulid_paths_for(config: &Config, family: IdentityFamily) -> Option<PulidPaths> {
    let manifest = pulid_manifest_for(family);
    let resolve = |component: ModelComponent| -> Option<PathBuf> {
        config.complete_manifest_file_path(manifest, file_for(manifest, component)?)
    };
    Some(PulidPaths {
        family,
        adapter: resolve(ModelComponent::IdentityAdapter)?,
        vision_encoder_source: resolve(ModelComponent::IdentityVisionEncoder)?,
        face_detector: resolve(ModelComponent::FaceDetector)?,
        face_recognizer: resolve(ModelComponent::FaceRecognizer)?,
        face_parser_source: resolve(ModelComponent::FaceParser)?,
    })
}

/// True when every asset of `family`'s bundle is present and complete.
pub fn pulid_is_installed_for(config: &Config, family: IdentityFamily) -> bool {
    pulid_paths_for(config, family).is_some()
}

/// The manifest files a repair pull of `family`'s bundle still has to fetch.
///
/// Empty means installed; a non-empty result on an otherwise-present bundle is
/// exactly the "needs repair" signal `mold pull` acts on.
pub fn missing_pulid_files_for(config: &Config, family: IdentityFamily) -> Vec<&'static ModelFile> {
    let manifest = pulid_manifest_for(family);
    manifest
        .files
        .iter()
        .filter(|file| config.complete_manifest_file_path(manifest, file).is_none())
        .collect()
}

/// The EVA02-CLIP vision tower mold DERIVES from the `.pt` source on first use.
///
/// The name lives here rather than beside the converter in `mold-inference`
/// because removal has to delete it and `mold-core` cannot see that crate. It
/// is the converter's authority all the same — `encoders::pickle_convert`
/// reads it from here — so the two can never name different files.
pub const DERIVED_VISION_FILENAME: &str = "eva02_clip_l_336_vision.safetensors";

/// Provenance sidecar written beside [`DERIVED_VISION_FILENAME`]. Never read
/// back to decide anything; deleted with the artifact it describes.
pub const DERIVED_VISION_SIDECAR_FILENAME: &str = "eva02_clip_l_336_vision.json";

/// The BiSeNet face parser mold DERIVES from facexlib's `.pth` on first use.
///
/// Here for the same reason [`DERIVED_VISION_FILENAME`] is: removal has to
/// delete it, and `mold-core` cannot see `mold-inference`.
pub const DERIVED_PARSER_FILENAME: &str = "bisenet_face_parser.safetensors";

/// Provenance sidecar written beside [`DERIVED_PARSER_FILENAME`].
pub const DERIVED_PARSER_SIDECAR_FILENAME: &str = "bisenet_face_parser.json";

/// The artifacts mold derived rather than downloaded, present or not.
///
/// These are NOT manifest files, so nothing that enumerates the manifest can
/// find them — which is exactly how a `mold rm pulid-flux` would leave 609 MB
/// of converted weights and their sidecar behind. They live beside the `.pt`
/// they were converted from.
///
/// They are derived from the SHARED extraction artifacts, so both bundles own
/// them and both name them: removal's ref-counting is what keeps the surviving
/// bundle's copy, exactly as it keeps the four downloads themselves.
pub fn derived_pulid_paths(config: &Config) -> Vec<PathBuf> {
    let manifest = pulid_manifest_for(IdentityFamily::Flux);
    let models_dir = config.resolved_models_dir();
    let Some(source) = file_for(manifest, ModelComponent::IdentityVisionEncoder) else {
        return Vec::new();
    };
    let root = models_dir
        .join(crate::manifest::storage_path(manifest, source))
        .parent()
        .map(std::path::Path::to_path_buf);
    root.map(|root| {
        vec![
            root.join(DERIVED_VISION_FILENAME),
            root.join(DERIVED_VISION_SIDECAR_FILENAME),
            root.join(DERIVED_PARSER_FILENAME),
            root.join(DERIVED_PARSER_SIDECAR_FILENAME),
        ]
    })
    .unwrap_or_default()
}

/// Canonical on-disk destinations for every DOWNLOADED asset of `family`'s
/// bundle, present or not.
///
/// Deliberately the five manifest files only. Removal also has to delete what
/// mold DERIVED from them, which is [`derived_pulid_paths`] and is kept
/// separate because a caller asking "where do the bundle's downloads land" is
/// not asking the same question.
pub fn pulid_storage_paths_for(config: &Config, family: IdentityFamily) -> Vec<PathBuf> {
    let manifest = pulid_manifest_for(family);
    let models_dir = config.resolved_models_dir();
    manifest
        .files
        .iter()
        .map(|file| models_dir.join(crate::manifest::storage_path(manifest, file)))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::manifest::{storage_path, PULID_FLUX_MANIFEST, PULID_SDXL_MANIFEST};
    use crate::test_support::ENV_LOCK;

    /// `resolved_models_dir()` consults `MOLD_MODELS_DIR` / `MOLD_HOME`, which
    /// sibling tests mutate. Serialize on the shared guard so a throwaway
    /// models dir stays the one that answers.
    fn env_guard() -> std::sync::MutexGuard<'static, ()> {
        ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner())
    }

    fn config_for(models_dir: &std::path::Path) -> Config {
        Config {
            models_dir: models_dir.to_string_lossy().to_string(),
            ..Default::default()
        }
    }

    const ALL: &[ModelComponent] = &[
        ModelComponent::IdentityAdapter,
        ModelComponent::IdentityVisionEncoder,
        ModelComponent::FaceDetector,
        ModelComponent::FaceRecognizer,
        ModelComponent::FaceParser,
    ];

    fn write_files(
        models_dir: &std::path::Path,
        manifest: &ModelManifest,
        components: &[ModelComponent],
    ) {
        for file in &manifest.files {
            if !components.contains(&file.component) {
                continue;
            }
            let path = models_dir.join(storage_path(manifest, file));
            std::fs::create_dir_all(path.parent().unwrap()).unwrap();
            std::fs::write(&path, b"stub").unwrap();
            crate::download::write_sha256_marker(&path, file.sha256.unwrap()).unwrap();
        }
    }

    #[test]
    fn every_family_manifest_shares_one_pulid_storage_root() {
        for manifest in pulid_manifests() {
            for file in &manifest.files {
                let path = storage_path(manifest, file);
                assert_eq!(
                    path.parent().map(|p| p.to_string_lossy().to_string()),
                    Some("shared/pulid".to_string()),
                    "{} landed at {}",
                    file.hf_filename,
                    path.display()
                );
            }
        }
    }

    /// The whole point of the two-bundle split: a machine that already holds
    /// one family's extractor pulls exactly the other family's adapter.
    #[test]
    fn the_two_bundles_differ_by_exactly_the_adapter() {
        let flux = pulid_manifest_for(IdentityFamily::Flux);
        let sdxl = pulid_manifest_for(IdentityFamily::Sdxl);
        assert_eq!(flux.files.len(), sdxl.files.len());

        let shared_paths = |manifest: &'static ModelManifest| -> Vec<String> {
            manifest
                .files
                .iter()
                .filter(|file| file.component != ModelComponent::IdentityAdapter)
                .map(|file| storage_path(manifest, file).to_string_lossy().to_string())
                .collect()
        };
        assert_eq!(
            shared_paths(flux),
            shared_paths(sdxl),
            "the four extraction artifacts must resolve to identical on-disk paths"
        );

        let adapter = |manifest: &'static ModelManifest| {
            storage_path(
                manifest,
                file_for(manifest, ModelComponent::IdentityAdapter).unwrap(),
            )
        };
        assert_ne!(adapter(flux), adapter(sdxl));
        assert!(adapter(flux).ends_with("pulid_flux_v0.9.1.safetensors"));
        assert!(adapter(sdxl).ends_with("pulid_v1.1.safetensors"));
    }

    #[test]
    fn resolution_requires_every_asset() {
        let _lock = env_guard();
        for family in IdentityFamily::ALL.iter().copied() {
            // Sizes are declared in the manifest, so write sparse stand-ins
            // instead: a `.sha256-verified` marker is the other acceptance
            // signal. One models dir PER family, because the four extraction
            // artifacts are shared — writing them for one bundle would make
            // the next family's "nothing installed" case a lie.
            let dir = tempfile::tempdir().unwrap();
            let models_dir = dir.path();
            let config = config_for(models_dir);
            let manifest = pulid_manifest_for(family);
            assert!(pulid_paths_for(&config, family).is_none());
            assert_eq!(missing_pulid_files_for(&config, family).len(), 5);

            write_files(
                models_dir,
                manifest,
                &[
                    ModelComponent::IdentityAdapter,
                    ModelComponent::IdentityVisionEncoder,
                    ModelComponent::FaceDetector,
                    ModelComponent::FaceRecognizer,
                ],
            );
            assert!(
                pulid_paths_for(&config, family).is_none(),
                "four of five assets is not an install"
            );
            let missing = missing_pulid_files_for(&config, family);
            assert_eq!(missing.len(), 1);
            assert_eq!(missing[0].component, ModelComponent::FaceParser);

            write_files(models_dir, manifest, ALL);
            let paths = pulid_paths_for(&config, family).expect("complete bundle resolves");
            assert_eq!(paths.family, family);
            assert!(paths
                .vision_encoder_source
                .ends_with("shared/pulid/EVA02_CLIP_L_336_psz14_s6B.pt"));
            assert!(paths
                .face_detector
                .ends_with("shared/pulid/scrfd_10g_bnkps.onnx"));
            assert!(paths
                .face_recognizer
                .ends_with("shared/pulid/glintr100.onnx"));
            assert!(paths
                .face_parser_source
                .ends_with("shared/pulid/parsing_bisenet.pth"));
            assert!(missing_pulid_files_for(&config, family).is_empty());
            assert!(pulid_is_installed_for(&config, family));

            let expected_adapter = match family {
                IdentityFamily::Flux => "shared/pulid/pulid_flux_v0.9.1.safetensors",
                IdentityFamily::Sdxl => "shared/pulid/pulid_v1.1.safetensors",
            };
            assert!(
                paths.adapter.ends_with(expected_adapter),
                "{}",
                paths.adapter.display()
            );
        }
    }

    /// Installing one bundle leaves the other needing exactly its adapter —
    /// the 984 MB / 1.14 GB claim, checked rather than asserted in prose.
    #[test]
    fn installing_one_bundle_leaves_only_the_other_adapter_missing() {
        let _lock = env_guard();
        let dir = tempfile::tempdir().unwrap();
        let config = config_for(dir.path());

        write_files(dir.path(), pulid_manifest_for(IdentityFamily::Flux), ALL);
        assert!(pulid_is_installed_for(&config, IdentityFamily::Flux));

        let missing = missing_pulid_files_for(&config, IdentityFamily::Sdxl);
        assert_eq!(missing.len(), 1, "{missing:?}");
        assert_eq!(missing[0].component, ModelComponent::IdentityAdapter);
        assert_eq!(missing[0].hf_filename, "pulid_v1.1.safetensors");
        assert_eq!(missing[0].size_bytes, 984_405_232);
    }

    /// Installed state must come from the files, not from a `ModelPaths` the
    /// bundle can never produce — `mold list` and `mold pull`'s repair check
    /// both read `manifest_model_is_downloaded`.
    #[test]
    fn installed_state_tracks_the_files_not_a_model_paths() {
        let _lock = env_guard();
        for name in [PULID_FLUX_MANIFEST, PULID_SDXL_MANIFEST] {
            // One models dir per bundle: the shared extraction artifacts would
            // otherwise carry over and make the second bundle look half
            // installed before anything is written for it.
            let dir = tempfile::tempdir().unwrap();
            let config = config_for(dir.path());
            let manifest = find_manifest(name).unwrap();
            assert!(!config.manifest_model_is_downloaded(name));
            assert!(config.manifest_model_needs_download(name));

            let write = |file: &ModelFile| {
                let path = dir.path().join(storage_path(manifest, file));
                std::fs::create_dir_all(path.parent().unwrap()).unwrap();
                std::fs::write(&path, b"stub").unwrap();
                crate::download::write_sha256_marker(&path, file.sha256.unwrap()).unwrap();
            };

            for file in manifest.files.iter().take(manifest.files.len() - 1) {
                write(file);
            }
            assert!(
                config.manifest_model_needs_download(name),
                "a partially present bundle needs repair"
            );

            write(manifest.files.last().unwrap());
            assert!(config.manifest_model_is_downloaded(name));
            assert!(!config.manifest_model_needs_download(name));
        }
    }

    /// An installed auxiliary bundle is not a checkpoint: it must never
    /// become the "only downloaded model" default.
    #[test]
    fn an_installed_bundle_is_not_a_default_model_candidate() {
        for manifest in pulid_manifests() {
            assert!(!manifest.is_generation_model());
            assert!(manifest.is_files_only_bundle());
            assert!(manifest.hidden);
        }
    }

    #[test]
    fn storage_paths_are_listed_even_when_absent() {
        let _lock = env_guard();
        let dir = tempfile::tempdir().unwrap();
        let config = config_for(dir.path());
        for family in IdentityFamily::ALL.iter().copied() {
            let paths = pulid_storage_paths_for(&config, family);
            assert_eq!(paths.len(), 5);
            assert!(paths.iter().all(|path| !path.exists()));
        }
    }

    /// The derived artifacts live beside the `.pt` they were converted from,
    /// and removal has to be able to name them before they exist.
    #[test]
    fn derived_paths_sit_beside_the_conversion_source() {
        let _lock = env_guard();
        let dir = tempfile::tempdir().unwrap();
        let config = config_for(dir.path());
        let derived = derived_pulid_paths(&config);
        assert_eq!(derived.len(), 4);

        let source = pulid_storage_paths_for(&config, IdentityFamily::Flux)
            .into_iter()
            .find(|path| path.extension().is_some_and(|ext| ext == "pt"))
            .expect("the vision encoder source is a .pt");
        for path in &derived {
            assert_eq!(path.parent(), source.parent(), "{}", path.display());
        }
        assert!(derived
            .iter()
            .any(|path| path.file_name().unwrap() == DERIVED_VISION_FILENAME));
        assert!(derived
            .iter()
            .any(|path| path.file_name().unwrap() == DERIVED_VISION_SIDECAR_FILENAME));
        assert!(derived
            .iter()
            .any(|path| path.file_name().unwrap() == DERIVED_PARSER_FILENAME));
        assert!(derived
            .iter()
            .any(|path| path.file_name().unwrap() == DERIVED_PARSER_SIDECAR_FILENAME));
    }
}
