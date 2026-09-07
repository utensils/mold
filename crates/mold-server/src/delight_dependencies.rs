//! Pre-admission materialization of Hunyuan3D's lighting-removal pipeline.

use mold_core::manifest::{
    find_manifest, paths_from_downloads, storage_path, HUNYUAN3D_DELIGHT_MANIFEST, HUNYUAN3D_FAMILY,
};
use mold_core::GenerateRequest;

use crate::execution_plan::PendingArtifactContainer;
use crate::variant_dependencies::{
    ensure_downloaded, DependencyContext, DependencySpec, MissingDependency, PinnedDigest,
};

fn requested(request: &GenerateRequest, family: &str) -> bool {
    family == HUNYUAN3D_FAMILY
        && request
            .mesh
            .as_ref()
            .is_some_and(|mesh| mesh.delight == Some(true))
}

pub(crate) async fn materialize_delight_paths(
    context: &DependencyContext<'_>,
    request: &GenerateRequest,
    family: &str,
    frozen: &mut mold_inference::FrozenEngineConfig,
    pending: &mut Vec<MissingDependency>,
) -> Result<(), String> {
    if !requested(request, family) {
        return Ok(());
    }
    let manifest = find_manifest(HUNYUAN3D_DELIGHT_MANIFEST)
        .ok_or_else(|| "Hunyuan3D delight manifest is not registered".to_string())?;
    let mut downloads = Vec::with_capacity(manifest.files.len());
    for file in &manifest.files {
        let relative = storage_path(manifest, file);
        let subdir = relative
            .parent()
            .map(|path| path.to_string_lossy().into_owned())
            .unwrap_or_default();
        let container = if file.hf_filename.ends_with(".safetensors") {
            PendingArtifactContainer::Safetensors
        } else {
            PendingArtifactContainer::Raw
        };
        let path = ensure_downloaded(
            context.state,
            context.work_id,
            DependencySpec {
                models_root: context.models_root,
                repo: &file.hf_repo,
                filename: &file.hf_filename,
                expected_bytes: Some(file.size_bytes),
                kind: "mesh_delight",
                container,
                quantization: None,
                expected_sha256: file.sha256.map(|sha256| PinnedDigest {
                    sha256,
                    repair_model: &manifest.name,
                }),
                subdir: &subdir,
            },
            context.progress,
            context.policy,
        )
        .await?
        .into_path(pending);
        downloads.push((file.component, path));
    }
    frozen.delight_paths = Some(
        paths_from_downloads(&downloads, &manifest.family)
            .ok_or_else(|| "Hunyuan3D delight files did not resolve to a pipeline".to_string())?,
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn only_an_explicit_hunyuan3d_delight_request_needs_the_pipeline() {
        let mut request: GenerateRequest = serde_json::from_value(serde_json::json!({
            "prompt": "", "model": "hunyuan3d:fp16", "width": 0,
            "height": 0, "steps": 30, "guidance": 5.0
        }))
        .unwrap();
        request.mesh = Some(mold_core::MeshRequestOptions::default());
        assert!(!requested(&request, HUNYUAN3D_FAMILY));
        request.mesh.as_mut().unwrap().delight = Some(true);
        assert!(requested(&request, HUNYUAN3D_FAMILY));
        assert!(!requested(&request, "flux"));
    }
}
