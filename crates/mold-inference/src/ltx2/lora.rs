use anyhow::{anyhow, bail, Result};
use mold_core::{GenerateRequest, LoraWeight};
use std::path::PathBuf;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct CameraControlPreset {
    pub(crate) repo: &'static str,
    pub(crate) filename: &'static str,
}

pub(crate) fn normalize_loras(req: &GenerateRequest) -> Vec<LoraWeight> {
    req.loras
        .clone()
        .or_else(|| req.lora.clone().map(|lora| vec![lora]))
        .unwrap_or_default()
}

pub(crate) fn camera_control_preset(name: &str) -> Option<CameraControlPreset> {
    let normalized = name.trim().to_ascii_lowercase().replace('_', "-");
    match normalized.as_str() {
        "dolly-in" => Some(CameraControlPreset {
            repo: "Lightricks/LTX-2-19b-LoRA-Camera-Control-Dolly-In",
            filename: "ltx-2-19b-lora-camera-control-dolly-in.safetensors",
        }),
        "dolly-left" => Some(CameraControlPreset {
            repo: "Lightricks/LTX-2-19b-LoRA-Camera-Control-Dolly-Left",
            filename: "ltx-2-19b-lora-camera-control-dolly-left.safetensors",
        }),
        "dolly-out" => Some(CameraControlPreset {
            repo: "Lightricks/LTX-2-19b-LoRA-Camera-Control-Dolly-Out",
            filename: "ltx-2-19b-lora-camera-control-dolly-out.safetensors",
        }),
        "dolly-right" => Some(CameraControlPreset {
            repo: "Lightricks/LTX-2-19b-LoRA-Camera-Control-Dolly-Right",
            filename: "ltx-2-19b-lora-camera-control-dolly-right.safetensors",
        }),
        "jib-down" => Some(CameraControlPreset {
            repo: "Lightricks/LTX-2-19b-LoRA-Camera-Control-Jib-Down",
            filename: "ltx-2-19b-lora-camera-control-jib-down.safetensors",
        }),
        "jib-up" => Some(CameraControlPreset {
            repo: "Lightricks/LTX-2-19b-LoRA-Camera-Control-Jib-Up",
            filename: "ltx-2-19b-lora-camera-control-jib-up.safetensors",
        }),
        "static" => Some(CameraControlPreset {
            repo: "Lightricks/LTX-2-19b-LoRA-Camera-Control-Static",
            filename: "ltx-2-19b-lora-camera-control-static.safetensors",
        }),
        _ => None,
    }
}

pub(crate) fn resolve_camera_control_preset_path(model_name: &str, name: &str) -> Result<PathBuf> {
    if model_name.contains("ltx-2.3") {
        bail!(
            "camera-control presets are currently published for LTX-2 19B only; pass an explicit .safetensors path for LTX-2.3"
        );
    }

    let preset = camera_control_preset(name).ok_or_else(|| {
        anyhow!(
            "unknown camera-control preset '{name}' (expected one of: dolly-in, dolly-left, dolly-out, dolly-right, jib-down, jib-up, static)"
        )
    })?;

    mold_core::download::download_single_file_sync(
        preset.repo,
        preset.filename,
        Some("shared/ltx2-camera-control"),
    )
    .map_err(|err| anyhow!("failed to download camera-control preset '{name}': {err}"))
}

pub(crate) fn resolve_loras(model_name: &str, req: &GenerateRequest) -> Result<Vec<LoraWeight>> {
    let mut loras = normalize_loras(req);
    for lora in &mut loras {
        if let Some(name) = lora.path.strip_prefix("camera-control:") {
            let resolved = resolve_camera_control_preset_path(model_name, name)?;
            lora.path = resolved.to_string_lossy().to_string();
        }
    }
    Ok(loras)
}

#[cfg(test)]
mod tests {
    use super::*;
    use mold_core::{GenerateRequest, OutputFormat};

    fn dummy_request() -> GenerateRequest {
        GenerateRequest {
            prompt: "test".to_string(),
            negative_prompt: None,
            model: "ltx-2-19b-distilled:fp8".to_string(),
            width: 960,
            height: 576,
            steps: 8,
            guidance: 3.0,
            seed: Some(42),
            batch_size: 1,
            output_format: OutputFormat::Mp4,
            embed_metadata: None,
            scheduler: None,
            source_image: None,
            edit_images: None,
            strength: 0.75,
            mask_image: None,
            control_image: None,
            control_model: None,
            control_scale: 1.0,
            expand: None,
            original_prompt: None,
            lora: None,
            frames: Some(17),
            fps: Some(12),
            upscale_model: None,
            gif_preview: false,
            enable_audio: Some(true),
            audio_file: None,
            source_video: None,
            keyframes: None,
            pipeline: None,
            loras: None,
            retake_range: None,
            spatial_upscale: None,
            temporal_upscale: None,
        }
    }

    #[test]
    fn legacy_single_lora_is_normalized_to_stack() {
        let mut req = dummy_request();
        req.lora = Some(LoraWeight {
            path: "/tmp/a.safetensors".to_string(),
            scale: 0.75,
        });
        let loras = normalize_loras(&req);
        assert_eq!(loras.len(), 1);
        assert_eq!(loras[0].path, "/tmp/a.safetensors");
    }

    #[test]
    fn explicit_lora_stack_preserves_order() {
        let mut req = dummy_request();
        req.loras = Some(vec![
            LoraWeight {
                path: "/tmp/one.safetensors".to_string(),
                scale: 0.5,
            },
            LoraWeight {
                path: "/tmp/two.safetensors".to_string(),
                scale: 1.0,
            },
        ]);
        let loras = normalize_loras(&req);
        assert_eq!(loras[0].path, "/tmp/one.safetensors");
        assert_eq!(loras[1].path, "/tmp/two.safetensors");
    }

    #[test]
    fn camera_control_preset_aliases_are_supported() {
        let preset = camera_control_preset("dolly-in").unwrap();
        assert_eq!(
            preset.filename,
            "ltx-2-19b-lora-camera-control-dolly-in.safetensors"
        );
        assert!(camera_control_preset("unknown").is_none());
    }
}
