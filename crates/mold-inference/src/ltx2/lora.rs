use anyhow::{anyhow, bail, Context, Result};
use candle_core::{Device, Tensor};
use mold_core::{GenerateRequest, LoraWeight};
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct CameraControlPreset {
    pub(crate) repo: &'static str,
    pub(crate) filename: &'static str,
}

#[derive(Clone, Debug)]
pub(crate) struct LinearLoraAdapter {
    pub(crate) a: Tensor,
    pub(crate) b: Tensor,
    pub(crate) scale: f64,
}

#[derive(Clone, Debug, Default)]
pub(crate) struct Ltx2LoraRegistry {
    layers: HashMap<String, Vec<LinearLoraAdapter>>,
}

impl Ltx2LoraRegistry {
    pub(crate) fn adapters_for(&self, key: &str) -> Vec<LinearLoraAdapter> {
        self.layers.get(key).cloned().unwrap_or_default()
    }

    pub(crate) fn is_empty(&self) -> bool {
        self.layers.is_empty()
    }

    #[cfg(test)]
    fn contains_layer(&self, key: &str) -> bool {
        self.layers.contains_key(key)
    }
}

fn strip_optional_model_prefix(key: &str) -> &str {
    key.strip_prefix("model.").unwrap_or(key)
}

fn canonical_lora_layer_key(name: &str) -> Option<String> {
    let base = name
        .strip_suffix(".lora_A.weight")
        .or_else(|| name.strip_suffix(".lora_B.weight"))
        .or_else(|| name.strip_suffix(".alpha"))?;
    let base = strip_optional_model_prefix(base);
    if base.starts_with("diffusion_model.") {
        Some(base.to_string())
    } else if base.starts_with("transformer_blocks.")
        || base.starts_with("patchify_proj")
        || base.starts_with("adaln_single")
        || base.starts_with("prompt_adaln_single")
        || base.starts_with("caption_projection")
        || base.starts_with("proj_out")
        || base.starts_with("audio_")
        || base.starts_with("av_ca_")
        || base.starts_with("scale_shift_table")
    {
        Some(format!("diffusion_model.{base}"))
    } else {
        None
    }
}

fn effective_lora_scale(user_scale: f64, rank: usize, alpha: Option<f64>) -> f64 {
    match alpha {
        Some(alpha) if rank > 0 => user_scale * alpha / rank as f64,
        _ => user_scale,
    }
}

pub(crate) fn load_lora_registry(loras: &[LoraWeight]) -> Result<Option<Arc<Ltx2LoraRegistry>>> {
    if loras.is_empty() {
        return Ok(None);
    }

    let mut registry = Ltx2LoraRegistry::default();
    for lora in loras {
        let tensors = candle_core::safetensors::load(&lora.path, &Device::Cpu)
            .with_context(|| format!("failed to load LTX-2 LoRA {}", lora.path))?;
        let mut a_tensors: HashMap<String, Tensor> = HashMap::new();
        let mut b_tensors: HashMap<String, Tensor> = HashMap::new();
        let mut alpha_values: HashMap<String, f64> = HashMap::new();

        for (name, tensor) in tensors {
            if let Some(key) = name
                .strip_suffix(".lora_A.weight")
                .and_then(|_| canonical_lora_layer_key(&name))
            {
                a_tensors.insert(key, tensor);
            } else if let Some(key) = name
                .strip_suffix(".lora_B.weight")
                .and_then(|_| canonical_lora_layer_key(&name))
            {
                b_tensors.insert(key, tensor);
            } else if let Some(key) = name
                .strip_suffix(".alpha")
                .and_then(|_| canonical_lora_layer_key(&name))
            {
                if let Ok(value) = tensor.to_dtype(candle_core::DType::F32)?.to_scalar::<f32>() {
                    alpha_values.insert(key, value as f64);
                }
            }
        }

        let mut found_pairs = 0usize;
        for (key, a) in a_tensors {
            let Some(b) = b_tensors.remove(&key) else {
                continue;
            };
            let rank = a.dim(0)?;
            let scale = effective_lora_scale(lora.scale, rank, alpha_values.get(&key).copied());
            registry
                .layers
                .entry(key)
                .or_default()
                .push(LinearLoraAdapter { a, b, scale });
            found_pairs += 1;
        }

        if found_pairs == 0 {
            bail!(
                "no LTX-2 LoRA A/B pairs found in {}",
                PathBuf::from(&lora.path).display()
            );
        }
    }

    if registry.is_empty() {
        Ok(None)
    } else {
        Ok(Some(Arc::new(registry)))
    }
}

fn read_reference_downscale_factor(path: &Path) -> usize {
    let Ok(data) = std::fs::read(path) else {
        return 1;
    };
    let Ok((_header_len, metadata)) = safetensors::tensor::SafeTensors::read_metadata(&data) else {
        return 1;
    };
    metadata
        .metadata()
        .as_ref()
        .and_then(|metadata| metadata.get("reference_downscale_factor"))
        .and_then(|value| value.trim().parse::<usize>().ok())
        .filter(|value| *value > 0)
        .unwrap_or(1)
}

pub(crate) fn reference_video_downscale_factor(loras: &[LoraWeight]) -> Result<usize> {
    let mut resolved = 1usize;
    for lora in loras {
        let scale = read_reference_downscale_factor(Path::new(&lora.path));
        if scale == 1 {
            continue;
        }
        if resolved != 1 && resolved != scale {
            bail!(
                "conflicting reference_downscale_factor values in LoRAs: already have {resolved}, but {} specifies {scale}",
                PathBuf::from(&lora.path).display()
            );
        }
        resolved = scale;
    }
    Ok(resolved)
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
    use safetensors::tensor::{serialize_to_file, Dtype as SafeDtype, TensorView};
    use std::collections::HashMap;

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

    fn temp_file(name: &str) -> std::path::PathBuf {
        let mut path = std::env::temp_dir();
        path.push(format!(
            "mold-ltx2-lora-{}-{}-{}.safetensors",
            name,
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        path
    }

    #[test]
    fn canonical_lora_layer_key_normalizes_expected_prefixes() {
        assert_eq!(
            canonical_lora_layer_key(
                "diffusion_model.transformer_blocks.0.attn1.to_q.lora_A.weight"
            )
            .as_deref(),
            Some("diffusion_model.transformer_blocks.0.attn1.to_q")
        );
        assert_eq!(
            canonical_lora_layer_key(
                "model.diffusion_model.transformer_blocks.0.attn1.to_q.lora_B.weight"
            )
            .as_deref(),
            Some("diffusion_model.transformer_blocks.0.attn1.to_q")
        );
        assert_eq!(
            canonical_lora_layer_key("transformer_blocks.0.attn1.to_q.alpha").as_deref(),
            Some("diffusion_model.transformer_blocks.0.attn1.to_q")
        );
        assert!(canonical_lora_layer_key("tokenizer.foo").is_none());
    }

    #[test]
    fn load_lora_registry_parses_camera_control_style_pairs() {
        let path = temp_file("registry");
        let a_data = vec![0u8; 2 * 4 * std::mem::size_of::<f32>()];
        let b_data = vec![0u8; 8 * 2 * std::mem::size_of::<f32>()];
        let alpha_data = 4.0f32.to_le_bytes().to_vec();
        let mut tensors = HashMap::new();
        tensors.insert(
            "diffusion_model.transformer_blocks.0.attn1.to_q.lora_A.weight".to_string(),
            TensorView::new(SafeDtype::F32, vec![2, 4], &a_data).unwrap(),
        );
        tensors.insert(
            "diffusion_model.transformer_blocks.0.attn1.to_q.lora_B.weight".to_string(),
            TensorView::new(SafeDtype::F32, vec![8, 2], &b_data).unwrap(),
        );
        tensors.insert(
            "diffusion_model.transformer_blocks.0.attn1.to_q.alpha".to_string(),
            TensorView::new(SafeDtype::F32, vec![], &alpha_data).unwrap(),
        );
        serialize_to_file(&tensors, &None, &path).unwrap();

        let registry = load_lora_registry(&[LoraWeight {
            path: path.to_string_lossy().to_string(),
            scale: 0.5,
        }])
        .unwrap()
        .unwrap();
        assert!(registry.contains_layer("diffusion_model.transformer_blocks.0.attn1.to_q"));
        let adapters = registry.adapters_for("diffusion_model.transformer_blocks.0.attn1.to_q");
        assert_eq!(adapters.len(), 1);
        assert!((adapters[0].scale - 1.0).abs() < 1e-6);

        let _ = std::fs::remove_file(path);
    }

    #[test]
    fn reference_video_downscale_factor_reads_metadata() {
        let path = temp_file("ref-scale");
        let data = vec![0u8; 4 * std::mem::size_of::<f32>()];
        let mut tensors = HashMap::new();
        tensors.insert(
            "diffusion_model.transformer_blocks.0.attn1.to_q.lora_A.weight".to_string(),
            TensorView::new(SafeDtype::F32, vec![1, 4], &data).unwrap(),
        );
        let metadata = Some(HashMap::from([(
            "reference_downscale_factor".to_string(),
            "2".to_string(),
        )]));
        serialize_to_file(&tensors, &metadata, &path).unwrap();

        let scale = reference_video_downscale_factor(&[LoraWeight {
            path: path.to_string_lossy().to_string(),
            scale: 1.0,
        }])
        .unwrap();

        assert_eq!(scale, 2);
        let _ = std::fs::remove_file(path);
    }

    #[test]
    fn reference_video_downscale_factor_rejects_conflicting_values() {
        let path_one = temp_file("ref-scale-one");
        let path_two = temp_file("ref-scale-two");
        let data = vec![0u8; 4 * std::mem::size_of::<f32>()];
        let mut tensors = HashMap::new();
        tensors.insert(
            "diffusion_model.transformer_blocks.0.attn1.to_q.lora_A.weight".to_string(),
            TensorView::new(SafeDtype::F32, vec![1, 4], &data).unwrap(),
        );
        serialize_to_file(
            &tensors,
            &Some(HashMap::from([(
                "reference_downscale_factor".to_string(),
                "2".to_string(),
            )])),
            &path_one,
        )
        .unwrap();
        serialize_to_file(
            &tensors,
            &Some(HashMap::from([(
                "reference_downscale_factor".to_string(),
                "4".to_string(),
            )])),
            &path_two,
        )
        .unwrap();

        let err = reference_video_downscale_factor(&[
            LoraWeight {
                path: path_one.to_string_lossy().to_string(),
                scale: 1.0,
            },
            LoraWeight {
                path: path_two.to_string_lossy().to_string(),
                scale: 1.0,
            },
        ])
        .unwrap_err();

        assert!(err
            .to_string()
            .contains("conflicting reference_downscale_factor"));
        let _ = std::fs::remove_file(path_one);
        let _ = std::fs::remove_file(path_two);
    }
}
