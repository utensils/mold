//! Pre-admission materialization of the pinned U²-Net matting graph.

use mold_core::{
    manifest::{find_manifest, storage_path, HUNYUAN3D_FAMILY, HUNYUAN3D_MATTING_MANIFEST},
    GenerateRequest, MeshMattingMode,
};

use crate::{
    execution_plan::PendingArtifactContainer,
    variant_dependencies::{
        ensure_downloaded, DependencyContext, DependencySpec, MissingDependency, PinnedDigest,
    },
};

fn requested(request: &GenerateRequest, family: &str) -> bool {
    family == HUNYUAN3D_FAMILY
        && request
            .mesh
            .as_ref()
            .and_then(|mesh| mesh.matting)
            .unwrap_or_default()
            != MeshMattingMode::Off
}

pub(crate) async fn materialize_matting_asset(
    context: &DependencyContext<'_>,
    request: &GenerateRequest,
    family: &str,
    frozen: &mut mold_inference::FrozenEngineConfig,
    pending: &mut Vec<MissingDependency>,
) -> Result<(), String> {
    if !requested(request, family) {
        return Ok(());
    }
    let manifest = find_manifest(HUNYUAN3D_MATTING_MANIFEST)
        .ok_or_else(|| "Hunyuan3D matting manifest is not registered".to_string())?;
    let file = manifest
        .files
        .first()
        .ok_or_else(|| "Hunyuan3D matting manifest has no graph".to_string())?;
    let relative = storage_path(manifest, file);
    let subdir = relative
        .parent()
        .map(|path| path.to_string_lossy().into_owned())
        .unwrap_or_default();
    let path = ensure_downloaded(
        context.state,
        context.work_id,
        DependencySpec {
            models_root: context.models_root,
            repo: &file.hf_repo,
            filename: &file.hf_filename,
            expected_bytes: Some(file.size_bytes),
            kind: "background_matting",
            container: PendingArtifactContainer::Onnx,
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
    frozen.matting_asset = Some(path);
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn auto_and_on_materialize_while_off_never_does() {
        let mut request: GenerateRequest = serde_json::from_value(serde_json::json!({
            "prompt": "",
            "model": "hunyuan3d:fp16",
            "width": 0,
            "height": 0,
            "steps": 30
        }))
        .unwrap();
        request.mesh = Some(mold_core::MeshRequestOptions::default());
        assert!(requested(&request, HUNYUAN3D_FAMILY));
        request.mesh.as_mut().unwrap().matting = Some(MeshMattingMode::On);
        assert!(requested(&request, HUNYUAN3D_FAMILY));
        request.mesh.as_mut().unwrap().matting = Some(MeshMattingMode::Off);
        assert!(!requested(&request, HUNYUAN3D_FAMILY));
        assert!(!requested(&request, "flux"));
    }
}
