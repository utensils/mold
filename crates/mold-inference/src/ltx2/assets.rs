use anyhow::{anyhow, bail, Result};
use mold_core::{Ltx2SpatialUpscale, Ltx2TemporalUpscale, ModelPaths};
use std::path::{Path, PathBuf};

pub(crate) fn gemma_root(paths: &ModelPaths) -> Result<PathBuf> {
    paths
        .text_encoder_files
        .first()
        .and_then(|path| path.parent().map(Path::to_path_buf))
        .ok_or_else(|| anyhow!("LTX-2 requires Gemma text encoder files to be available"))
}

pub(crate) fn request_quantization(model_name: &str) -> Option<String> {
    model_name.contains(":fp8").then(|| {
        // Use the no-extra-deps FP8 path by default. Hopper-specific
        // `fp8-scaled-mm` requires TensorRT-LLM and does not fit the
        // normal local 4090 workflow.
        "fp8-cast".to_string()
    })
}

pub(crate) fn resolve_spatial_upscaler_path(
    model_name: &str,
    paths: &ModelPaths,
    mode: Option<Ltx2SpatialUpscale>,
) -> Result<Option<PathBuf>> {
    match mode {
        None | Some(Ltx2SpatialUpscale::X2) => Ok(paths.spatial_upscaler.clone()),
        Some(Ltx2SpatialUpscale::X1_5) => {
            if !model_name.contains("ltx-2.3") {
                bail!("x1.5 spatial upscaling is currently published for LTX-2.3 only");
            }
            mold_core::download::download_single_file_sync(
                "Lightricks/LTX-2.3",
                "ltx-2.3-spatial-upscaler-x1.5-1.0.safetensors",
                Some("shared/LTX-2.3"),
            )
            .map(Some)
            .map_err(|err| anyhow!("failed to download LTX-2.3 x1.5 spatial upscaler: {err}"))
        }
    }
}

pub(crate) fn resolve_temporal_upscaler_path(
    paths: &ModelPaths,
    mode: Option<Ltx2TemporalUpscale>,
) -> Result<Option<PathBuf>> {
    match mode {
        None => Ok(None),
        Some(Ltx2TemporalUpscale::X2) => paths
            .temporal_upscaler
            .clone()
            .ok_or_else(|| {
                anyhow!("LTX-2 temporal upscaling requires a configured temporal upsampler asset")
            })
            .map(Some),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn dummy_paths() -> ModelPaths {
        ModelPaths {
            transformer: PathBuf::from("/tmp/transformer"),
            transformer_shards: vec![],
            vae: PathBuf::from("/tmp/unused"),
            spatial_upscaler: Some(PathBuf::from("/tmp/spatial-x2.safetensors")),
            temporal_upscaler: Some(PathBuf::from("/tmp/temporal.safetensors")),
            distilled_lora: Some(PathBuf::from("/tmp/distilled-lora.safetensors")),
            t5_encoder: None,
            clip_encoder: None,
            t5_tokenizer: None,
            clip_tokenizer: None,
            clip_encoder_2: None,
            clip_tokenizer_2: None,
            text_encoder_files: vec![PathBuf::from("/tmp/gemma/tokenizer.model")],
            text_tokenizer: None,
            decoder: None,
        }
    }

    #[test]
    fn fp8_models_use_fp8_cast_quantization() {
        assert_eq!(
            request_quantization("ltx-2-19b-distilled:fp8"),
            Some("fp8-cast".to_string())
        );
        assert_eq!(request_quantization("ltx-2-19b-dev:bf16"), None);
    }

    #[test]
    fn gemma_root_uses_parent_directory() {
        let root = gemma_root(&dummy_paths()).unwrap();
        assert_eq!(root, PathBuf::from("/tmp/gemma"));
    }

    #[test]
    fn x2_spatial_upscaling_uses_configured_asset() {
        let path = resolve_spatial_upscaler_path(
            "ltx-2-19b-distilled:fp8",
            &dummy_paths(),
            Some(Ltx2SpatialUpscale::X2),
        )
        .unwrap();
        assert_eq!(path, Some(PathBuf::from("/tmp/spatial-x2.safetensors")));
    }

    #[test]
    fn x1_5_spatial_upscaling_is_rejected_for_19b_models() {
        let err = resolve_spatial_upscaler_path(
            "ltx-2-19b-distilled:fp8",
            &dummy_paths(),
            Some(Ltx2SpatialUpscale::X1_5),
        )
        .unwrap_err();
        assert!(err.to_string().contains("LTX-2.3 only"));
    }

    #[test]
    fn temporal_upscaling_uses_configured_asset() {
        let path =
            resolve_temporal_upscaler_path(&dummy_paths(), Some(Ltx2TemporalUpscale::X2)).unwrap();
        assert_eq!(path, Some(PathBuf::from("/tmp/temporal.safetensors")));
    }

    #[test]
    fn temporal_upscaling_requires_configured_asset() {
        let mut paths = dummy_paths();
        paths.temporal_upscaler = None;
        let err =
            resolve_temporal_upscaler_path(&paths, Some(Ltx2TemporalUpscale::X2)).unwrap_err();
        assert!(err.to_string().contains("temporal upsampler asset"));
    }
}
