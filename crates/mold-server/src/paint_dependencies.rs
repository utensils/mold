//! Pre-admission materialization of Hunyuan3D paint runtime artifacts.

use std::path::{Path, PathBuf};

use mold_core::{
    hunyuan3d_paint_assets::Hunyuan3dPaintPaths,
    manifest::{
        find_manifest, storage_path, ModelFile, HUNYUAN3D_FAMILY, HUNYUAN3D_PAINT_MANIFEST,
        HUNYUAN3D_PAINT_UPSCALER_MANIFEST,
    },
    GenerateRequest,
};

use crate::{
    execution_plan::PendingArtifactContainer,
    variant_dependencies::{
        ensure_downloaded, DependencyContext, DependencyMaterializationPolicy, DependencySpec,
        MissingDependency, PinnedDigest,
    },
};

fn requested(request: &GenerateRequest, family: &str) -> bool {
    family == HUNYUAN3D_FAMILY
        && request
            .mesh
            .as_ref()
            .is_some_and(|mesh| mesh.texture == Some(true))
}

fn storage_subdir(manifest: &mold_core::manifest::ModelManifest, file: &ModelFile) -> String {
    storage_path(manifest, file)
        .parent()
        .map(|path| path.to_string_lossy().into_owned())
        .unwrap_or_default()
}

fn require_missing_licenses(
    manifest: &mold_core::manifest::ModelManifest,
    files: &[&ModelFile],
    models_root: &Path,
) -> Result<(), String> {
    for file in files {
        let subdir = storage_subdir(manifest, file);
        if mold_core::download::cached_file_path_in(
            models_root,
            &file.hf_repo,
            &file.hf_filename,
            Some(&subdir),
        )
        .is_some()
        {
            continue;
        }
        mold_core::download::require_license_accepted(
            &manifest.name,
            &file.hf_filename,
            mold_core::Config::mold_dir().as_deref(),
        )
        .map_err(|error| error.to_string())?;
    }
    Ok(())
}

pub(crate) async fn materialize_paint_assets(
    context: &DependencyContext<'_>,
    request: &GenerateRequest,
    family: &str,
    frozen: &mut mold_inference::FrozenEngineConfig,
    pending: &mut Vec<MissingDependency>,
) -> Result<(), String> {
    if !requested(request, family) {
        return Ok(());
    }
    let paint = find_manifest(HUNYUAN3D_PAINT_MANIFEST)
        .ok_or_else(|| "Hunyuan3D paint manifest is not registered".to_string())?;
    let upscaler = find_manifest(HUNYUAN3D_PAINT_UPSCALER_MANIFEST)
        .ok_or_else(|| "Hunyuan3D paint upscaler is not registered".to_string())?;
    let select = |suffix: &str| {
        paint
            .files
            .iter()
            .find(|file| file.hf_filename.ends_with(suffix))
            .ok_or_else(|| format!("Hunyuan3D paint manifest lost {suffix}"))
    };
    let runtime = [
        (
            select("unet/diffusion_pytorch_model.bin")?,
            "paint_unet",
            PendingArtifactContainer::TorchArchive,
        ),
        (
            select("vae/diffusion_pytorch_model.bin")?,
            "paint_vae",
            PendingArtifactContainer::TorchArchive,
        ),
        (
            paint
                .files
                .iter()
                .find(|file| file.hf_repo == "facebook/dinov2-giant")
                .ok_or_else(|| "Hunyuan3D paint manifest lost DINOv2 Giant".to_string())?,
            "paint_dino",
            PendingArtifactContainer::Safetensors,
        ),
    ];
    if context.policy == DependencyMaterializationPolicy::Admission {
        require_missing_licenses(
            paint,
            &runtime.map(|(file, _, _)| file),
            context.models_root,
        )?;
    }

    let mut resolved = Vec::<PathBuf>::new();
    for (file, kind, container) in runtime {
        let subdir = storage_subdir(paint, file);
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
                quantization: None,
                expected_sha256: file.sha256.map(|sha256| PinnedDigest {
                    sha256,
                    repair_model: &paint.name,
                }),
                subdir: &subdir,
            },
            context.progress,
            context.policy,
        )
        .await?
        .into_path(pending);
        resolved.push(path);
    }
    let upscaler_file = upscaler
        .files
        .first()
        .ok_or_else(|| "Hunyuan3D paint upscaler has no weights".to_string())?;
    let subdir = storage_subdir(upscaler, upscaler_file);
    let upscaler_path = ensure_downloaded(
        context.state,
        context.work_id,
        DependencySpec {
            models_root: context.models_root,
            repo: &upscaler_file.hf_repo,
            filename: &upscaler_file.hf_filename,
            expected_bytes: Some(upscaler_file.size_bytes),
            kind: "paint_upscaler",
            container: PendingArtifactContainer::Safetensors,
            quantization: None,
            expected_sha256: upscaler_file.sha256.map(|sha256| PinnedDigest {
                sha256,
                repair_model: &upscaler.name,
            }),
            subdir: &subdir,
        },
        context.progress,
        context.policy,
    )
    .await?
    .into_path(pending);

    frozen.paint_assets = Some(Hunyuan3dPaintPaths {
        unet: resolved.remove(0),
        vae: resolved.remove(0),
        dino: resolved.remove(0),
        upscaler: upscaler_path,
    });
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn only_an_explicit_textured_hunyuan3d_request_needs_paint() {
        let mut request: GenerateRequest = serde_json::from_value(serde_json::json!({
            "prompt": "",
            "model": "hunyuan3d:fp16",
            "width": 0,
            "height": 0,
            "steps": 30
        }))
        .unwrap();
        request.mesh = Some(mold_core::MeshRequestOptions::default());
        assert!(!requested(&request, HUNYUAN3D_FAMILY));
        request.mesh.as_mut().unwrap().texture = Some(true);
        assert!(requested(&request, HUNYUAN3D_FAMILY));
        assert!(!requested(&request, "flux"));
    }

    #[test]
    fn runtime_files_use_the_manifest_storage_layout() {
        let manifest = find_manifest(HUNYUAN3D_PAINT_MANIFEST).unwrap();
        let unet = manifest
            .files
            .iter()
            .find(|file| {
                file.hf_filename
                    .ends_with("unet/diffusion_pytorch_model.bin")
            })
            .unwrap();
        assert_eq!(
            storage_subdir(manifest, unet),
            "hunyuan3d-paint/hunyuan3d-paintpbr-v2-1/unet"
        );
    }
}
