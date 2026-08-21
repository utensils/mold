//! On-disk resolution for the PuLID-FLUX auxiliary asset bundle.
//!
//! The bundle is four unrelated artifacts — an identity adapter, a vision
//! tower, a face detector, and a face recognizer — and none of them is a
//! transformer or a VAE. [`crate::manifest::paths_from_downloads`] therefore
//! cannot represent it: it exists to build a [`crate::ModelPaths`], which
//! requires a generator. This module resolves the bundle on its own terms
//! instead, and answers only one question: are all four files present, and
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

/// Resolve the bundle, returning `Some` only when **all four** files are
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

/// Canonical on-disk destinations for every PuLID asset, present or not.
///
/// Used by removal, which must delete the files it planned to write rather
/// than only the ones it can currently see.
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
        assert_eq!(missing_pulid_files(&config).len(), 4);

        mark(&[
            ModelComponent::IdentityAdapter,
            ModelComponent::IdentityVisionEncoder,
            ModelComponent::FaceDetector,
        ]);
        assert!(
            pulid_paths(&config).is_none(),
            "three of four assets is not an install"
        );
        let missing = missing_pulid_files(&config);
        assert_eq!(missing.len(), 1);
        assert_eq!(missing[0].component, ModelComponent::FaceRecognizer);

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
        assert_eq!(paths.len(), 4);
        assert!(paths.iter().all(|path| !path.exists()));
    }
}
