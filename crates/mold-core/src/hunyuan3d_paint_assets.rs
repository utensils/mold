//! On-disk resolution for the Hunyuan3D PBR paint auxiliary bundle.

use std::path::PathBuf;

use crate::{
    config::Config,
    manifest::{
        find_manifest, ModelFile, ModelManifest, HUNYUAN3D_PAINT_MANIFEST,
        HUNYUAN3D_PAINT_UPSCALER_MANIFEST,
    },
};

/// Concrete runtime weights for one frozen Hunyuan3D paint execution.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Hunyuan3dPaintPaths {
    pub unet: PathBuf,
    pub vae: PathBuf,
    pub dino: PathBuf,
    pub upscaler: PathBuf,
}

pub fn paint_manifest() -> &'static ModelManifest {
    find_manifest(HUNYUAN3D_PAINT_MANIFEST).expect("Hunyuan3D paint manifest is registered")
}

fn file_named(name: &str) -> &'static ModelFile {
    paint_manifest()
        .files
        .iter()
        .find(|file| file.hf_filename.ends_with(name))
        .unwrap_or_else(|| panic!("Hunyuan3D paint manifest lost {name}"))
}

fn dino_file() -> &'static ModelFile {
    paint_manifest()
        .files
        .iter()
        .find(|file| file.hf_repo == "facebook/dinov2-giant")
        .expect("Hunyuan3D paint manifest lost DINOv2 Giant")
}

/// Resolve runtime paths only when every declared bundle file is complete.
pub fn paint_paths(config: &Config) -> Option<Hunyuan3dPaintPaths> {
    let manifest = paint_manifest();
    if manifest
        .files
        .iter()
        .any(|file| config.complete_manifest_file_path(manifest, file).is_none())
    {
        return None;
    }
    let resolve = |name: &str| config.complete_manifest_file_path(manifest, file_named(name));
    let upscaler_manifest = find_manifest(HUNYUAN3D_PAINT_UPSCALER_MANIFEST)?;
    let upscaler_file = upscaler_manifest.files.first()?;
    Some(Hunyuan3dPaintPaths {
        unet: resolve("unet/diffusion_pytorch_model.bin")?,
        vae: resolve("vae/diffusion_pytorch_model.bin")?,
        dino: config.complete_manifest_file_path(manifest, dino_file())?,
        upscaler: config.complete_manifest_file_path(upscaler_manifest, upscaler_file)?,
    })
}

pub fn missing_paint_files(config: &Config) -> Vec<&'static ModelFile> {
    let manifest = paint_manifest();
    manifest
        .files
        .iter()
        .filter(|file| config.complete_manifest_file_path(manifest, file).is_none())
        .collect()
}

pub fn paint_storage_paths(config: &Config) -> Vec<PathBuf> {
    let manifest = paint_manifest();
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
    use crate::{
        config::Config,
        manifest::{storage_path, HUNYUAN3D_PAINT_MANIFEST},
        test_support::ENV_LOCK,
    };

    fn config_for(models_dir: &std::path::Path) -> Config {
        Config {
            models_dir: models_dir.to_string_lossy().to_string(),
            ..Default::default()
        }
    }

    fn write_file(
        models_dir: &std::path::Path,
        manifest: &crate::manifest::ModelManifest,
        file: &crate::manifest::ModelFile,
    ) {
        let path = models_dir.join(storage_path(manifest, file));
        std::fs::create_dir_all(path.parent().unwrap()).unwrap();
        let handle = std::fs::File::create(&path).unwrap();
        handle.set_len(file.size_bytes).unwrap();
        if let Some(digest) = file.sha256 {
            crate::download::write_sha256_marker(&path, digest).unwrap();
        }
    }

    #[test]
    fn resolution_requires_the_complete_bundle_and_names_runtime_weights() {
        let _guard = ENV_LOCK.lock().unwrap_or_else(|error| error.into_inner());
        let root = tempfile::tempdir().unwrap();
        let config = config_for(root.path());
        let manifest = paint_manifest();
        assert_eq!(manifest.name, HUNYUAN3D_PAINT_MANIFEST);
        assert!(paint_paths(&config).is_none());
        assert_eq!(missing_paint_files(&config).len(), manifest.files.len());

        for file in manifest.files.iter().take(manifest.files.len() - 1) {
            write_file(root.path(), manifest, file);
        }
        assert!(paint_paths(&config).is_none());
        assert_eq!(missing_paint_files(&config).len(), 1);

        write_file(root.path(), manifest, manifest.files.last().unwrap());
        assert!(
            paint_paths(&config).is_none(),
            "paint also requires its upscaler"
        );
        let upscaler = crate::manifest::find_manifest(HUNYUAN3D_PAINT_UPSCALER_MANIFEST).unwrap();
        write_file(root.path(), upscaler, &upscaler.files[0]);
        let paths = paint_paths(&config).expect("complete paint bundle resolves");
        assert!(paths.unet.ends_with("unet/diffusion_pytorch_model.bin"));
        assert!(paths.vae.ends_with("vae/diffusion_pytorch_model.bin"));
        assert!(paths.dino.ends_with("dinov2-giant/model.safetensors"));
        assert!(paths
            .upscaler
            .ends_with("diffusion_pytorch_model.fp16.safetensors"));
        assert!(missing_paint_files(&config).is_empty());
    }

    #[test]
    fn paint_dino_is_the_external_giant_tower_not_the_bundled_clip_encoder() {
        let dino = paint_manifest()
            .files
            .iter()
            .find(|file| file.hf_repo == "facebook/dinov2-giant")
            .expect("paint manifest declares DINOv2 Giant");
        assert_eq!(dino.hf_repo, "facebook/dinov2-giant");
        assert_eq!(dino.size_bytes, 4_546_005_432);
        assert_eq!(
            dino.sha256,
            Some("917d3c470db999d32a312f8542149be91c7cbac61ee8fb4b67ae3d82b79ce21f")
        );
        let bundled_clip = paint_manifest()
            .files
            .iter()
            .find(|file| {
                file.hf_filename
                    .ends_with("image_encoder/model.safetensors")
            })
            .expect("upstream bundle remains complete");
        assert_ne!(dino.hf_repo, bundled_clip.hf_repo);
    }

    #[test]
    fn storage_paths_cover_every_manifest_file_without_resolving_a_model() {
        let _guard = ENV_LOCK.lock().unwrap_or_else(|error| error.into_inner());
        let root = tempfile::tempdir().unwrap();
        let config = config_for(root.path());
        assert_eq!(
            paint_storage_paths(&config).len(),
            paint_manifest().files.len()
        );
        assert!(paint_storage_paths(&config)
            .iter()
            .all(|path| !path.exists()));
        assert!(!paint_manifest().is_generation_model());
        assert!(paint_manifest().is_files_only_bundle());
    }
}
