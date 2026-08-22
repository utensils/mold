//! On-disk resolution for the PuLID-FLUX auxiliary asset bundle.
//!
//! The bundle is five unrelated artifacts — an identity adapter, a vision
//! tower, a face detector, a face recognizer, and a face parser — and none of
//! them is a transformer or a VAE. [`crate::manifest::paths_from_downloads`] therefore
//! cannot represent it: it exists to build a [`crate::ModelPaths`], which
//! requires a generator. This module resolves the bundle on its own terms
//! instead, and answers only one question: are all five files present, and
//! where?

use std::path::PathBuf;

use crate::config::Config;
use crate::manifest::{
    find_manifest, ModelComponent, ModelFile, ModelManifest, PULID_FLUX_MANIFEST,
};

/// Concrete, verified-complete paths to every PuLID asset.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PulidPaths {
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

/// The PuLID-FLUX manifest.
///
/// Panics only if the manifest registry lost the entry, which a completeness
/// test in `manifest.rs` makes impossible.
pub fn pulid_manifest() -> &'static ModelManifest {
    find_manifest(PULID_FLUX_MANIFEST).expect("pulid-flux manifest is registered")
}

fn file_for(component: ModelComponent) -> Option<&'static ModelFile> {
    pulid_manifest()
        .files
        .iter()
        .find(|file| file.component == component)
}

/// Resolve the bundle, returning `Some` only when **all five** files are
/// completely on disk.
///
/// A partially present bundle is deliberately `None` rather than a struct with
/// holes: PuLID cannot condition on an identity without every one of these
/// artifacts, so a caller holding a `PulidPaths` holds a runnable bundle.
/// Use [`missing_pulid_files`] to report what a repair still needs.
pub fn pulid_paths(config: &Config) -> Option<PulidPaths> {
    let manifest = pulid_manifest();
    let resolve = |component: ModelComponent| -> Option<PathBuf> {
        config.complete_manifest_file_path(manifest, file_for(component)?)
    };
    Some(PulidPaths {
        adapter: resolve(ModelComponent::IdentityAdapter)?,
        vision_encoder_source: resolve(ModelComponent::IdentityVisionEncoder)?,
        face_detector: resolve(ModelComponent::FaceDetector)?,
        face_recognizer: resolve(ModelComponent::FaceRecognizer)?,
        face_parser_source: resolve(ModelComponent::FaceParser)?,
    })
}

/// True when every PuLID asset is present and complete.
pub fn pulid_is_installed(config: &Config) -> bool {
    pulid_paths(config).is_some()
}

/// The manifest files a repair pull still has to fetch.
///
/// Empty means installed; a non-empty result on an otherwise-present bundle is
/// exactly the "needs repair" signal `mold pull` acts on.
pub fn missing_pulid_files(config: &Config) -> Vec<&'static ModelFile> {
    let manifest = pulid_manifest();
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
pub fn derived_pulid_paths(config: &Config) -> Vec<PathBuf> {
    let manifest = pulid_manifest();
    let models_dir = config.resolved_models_dir();
    let Some(source) = file_for(ModelComponent::IdentityVisionEncoder) else {
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

/// Canonical on-disk destinations for every DOWNLOADED PuLID asset, present or
/// not.
///
/// Deliberately the four manifest files only. Removal also has to delete what
/// mold DERIVED from them, which is [`derived_pulid_paths`] and is kept
/// separate because a caller asking "where do the bundle's downloads land" is
/// not asking the same question.
pub fn pulid_storage_paths(config: &Config) -> Vec<PathBuf> {
    let manifest = pulid_manifest();
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
    use crate::manifest::storage_path;
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

    #[test]
    fn manifest_files_share_one_pulid_storage_root() {
        let manifest = pulid_manifest();
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

    #[test]
    fn resolution_requires_every_asset() {
        let _lock = env_guard();
        // Sizes are declared in the manifest, so write sparse stand-ins
        // instead: a `.sha256-verified` marker is the other acceptance signal.
        let dir = tempfile::tempdir().unwrap();
        let models_dir = dir.path();
        let config = config_for(models_dir);
        let manifest = pulid_manifest();

        let mark = |components: &[ModelComponent]| {
            for file in &manifest.files {
                if !components.contains(&file.component) {
                    continue;
                }
                let path = models_dir.join(storage_path(manifest, file));
                std::fs::create_dir_all(path.parent().unwrap()).unwrap();
                std::fs::write(&path, b"stub").unwrap();
                crate::download::write_sha256_marker(&path, file.sha256.unwrap()).unwrap();
            }
        };

        assert!(pulid_paths(&config).is_none());
        assert_eq!(missing_pulid_files(&config).len(), 5);

        mark(&[
            ModelComponent::IdentityAdapter,
            ModelComponent::IdentityVisionEncoder,
            ModelComponent::FaceDetector,
            ModelComponent::FaceRecognizer,
        ]);
        assert!(
            pulid_paths(&config).is_none(),
            "four of five assets is not an install"
        );
        let missing = missing_pulid_files(&config);
        assert_eq!(missing.len(), 1);
        assert_eq!(missing[0].component, ModelComponent::FaceParser);

        mark(ALL);
        let paths = pulid_paths(&config).expect("complete bundle resolves");
        assert!(paths
            .adapter
            .ends_with("shared/pulid/pulid_flux_v0.9.1.safetensors"));
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
        assert!(missing_pulid_files(&config).is_empty());
        assert!(pulid_is_installed(&config));
    }

    /// Installed state must come from the files, not from a `ModelPaths` the
    /// bundle can never produce — `mold list` and `mold pull`'s repair check
    /// both read `manifest_model_is_downloaded`.
    #[test]
    fn installed_state_tracks_the_files_not_a_model_paths() {
        let _lock = env_guard();
        let dir = tempfile::tempdir().unwrap();
        let config = config_for(dir.path());
        let manifest = pulid_manifest();

        assert!(!config.manifest_model_is_downloaded(PULID_FLUX_MANIFEST));
        assert!(config.manifest_model_needs_download(PULID_FLUX_MANIFEST));

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
            config.manifest_model_needs_download(PULID_FLUX_MANIFEST),
            "a partially present bundle needs repair"
        );

        write(manifest.files.last().unwrap());
        assert!(config.manifest_model_is_downloaded(PULID_FLUX_MANIFEST));
        assert!(!config.manifest_model_needs_download(PULID_FLUX_MANIFEST));
    }

    /// An installed auxiliary bundle is not a checkpoint: it must never
    /// become the "only downloaded model" default.
    #[test]
    fn an_installed_bundle_is_not_a_default_model_candidate() {
        assert!(!pulid_manifest().is_generation_model());
    }

    #[test]
    fn storage_paths_are_listed_even_when_absent() {
        let _lock = env_guard();
        let dir = tempfile::tempdir().unwrap();
        let config = config_for(dir.path());
        let paths = pulid_storage_paths(&config);
        assert_eq!(paths.len(), 5);
        assert!(paths.iter().all(|path| !path.exists()));
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

        let source = pulid_storage_paths(&config)
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
