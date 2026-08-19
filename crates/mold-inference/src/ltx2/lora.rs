use anyhow::{anyhow, bail, Context, Result};
use candle_core::{Device, Tensor};
use mold_core::{GenerateRequest, LoraWeight};
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;

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

    /// How many transformer layers this stack actually resolved onto. Zero
    /// layers with a non-empty stack means the adapters named nothing the
    /// checkpoint has.
    pub(crate) fn layer_count(&self) -> usize {
        self.layers.len()
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

pub(crate) fn camera_control_preset(
    name: &str,
) -> Option<&'static mold_core::ltx2_camera::Ltx2CameraControlPreset> {
    mold_core::ltx2_camera::resolve_camera_control_preset(name).ok()
}

pub(crate) fn resolve_camera_control_preset_path(
    paths: &mold_core::ModelPaths,
    name: &str,
) -> Result<PathBuf> {
    // Sniff the resolved artifacts, not the model name: an opaque `cv:` /
    // `hf:` catalog ID for an LTX-2.3 checkpoint contains no architecture at
    // all, so a name-substring test both accepted 2.3 and rejected 19B.
    mold_core::ltx2_camera::camera_profile_for_artifact_paths([
        paths.transformer.to_str(),
        paths.vae.to_str(),
        paths.spatial_upscaler.as_deref().and_then(|p| p.to_str()),
    ])
    .map_err(|reason| anyhow!("{reason}; pass an explicit .safetensors path instead"))?;

    let preset =
        mold_core::ltx2_camera::resolve_camera_control_preset(name).map_err(anyhow::Error::msg)?;

    mold_core::download::download_single_file_sync(
        preset.hf_repo,
        preset.hf_filename,
        Some(preset.download_model),
    )
    .map_err(|err| anyhow!("failed to download camera-control preset '{name}': {err}"))
}

pub(crate) fn resolve_loras(
    paths: &mold_core::ModelPaths,
    req: &GenerateRequest,
) -> Result<Vec<LoraWeight>> {
    let mut loras = normalize_loras(req);
    for lora in &mut loras {
        if let Some(name) = lora.path.strip_prefix("camera-control:") {
            let resolved = resolve_camera_control_preset_path(paths, name)?;
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
            title: None,
            source_fit: None,
            hdr_exr_dir: None,
            hdr_exr_full_float: false,
            guidance_overrides: None,
            sample_shift: None,
            distill_strength_high: None,
            distill_strength_low: None,
            prompt: "test".to_string(),
            negative_prompt: None,
            model: "ltx-2-19b-distilled:fp8".to_string(),
            width: 960,
            height: 576,
            steps: 8,
            guidance: 3.0,
            seed: Some(42),
            batch_size: 1,
            output_format: Some(OutputFormat::Mp4),
            embed_metadata: None,
            scheduler: None,
            cfg_plus: None,
            source_image: None,
            source_image_name: None,
            edit_images: None,
            references: None,
            strength: 0.75,
            mask_image: None,
            control_image: None,
            control_model: None,
            control_scale: 1.0,
            expand: None,
            original_prompt: None,
            prompt_transform: None,
            batch_id: None,
            batch_index: None,
            batch_count: None,
            lora: None,
            frames: Some(17),
            fps: Some(12),
            upscale_model: None,
            gif_preview: false,
            enable_audio: Some(true),
            audio_file: None,
            audio_file_path: None,
            source_video: None,
            source_video_path: None,
            extend_video: None,
            extend_video_path: None,
            extend_overlap_frames: None,
            keyframes: None,
            pipeline: None,
            ic_lora_control: None,
            loras: None,
            retake_range: None,
            spatial_upscale: None,
            temporal_upscale: None,
            placement: None,
        }
    }

    #[test]
    fn legacy_single_lora_is_normalized_to_stack() {
        let mut req = dummy_request();
        req.lora = Some(LoraWeight {
            path: "/tmp/a.safetensors".to_string(),
            scale: 0.75,

            expert: None,
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

                expert: None,
            },
            LoraWeight {
                path: "/tmp/two.safetensors".to_string(),
                scale: 1.0,

                expert: None,
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
            preset.hf_filename,
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

            expert: None,
        }])
        .unwrap()
        .unwrap();
        assert!(registry.contains_layer("diffusion_model.transformer_blocks.0.attn1.to_q"));
        let adapters = registry.adapters_for("diffusion_model.transformer_blocks.0.attn1.to_q");
        assert_eq!(adapters.len(), 1);
        assert!((adapters[0].scale - 1.0).abs() < 1e-6);

        let _ = std::fs::remove_file(path);
    }

    /// Two camera-control adapters that touch the same layer must both apply,
    /// in request order. Keying the registry by layer alone would let the
    /// second silently replace the first, so a two-LoRA stack would quietly
    /// render as one.
    #[test]
    fn load_lora_registry_stacks_two_adapters_on_one_layer() {
        let key = "diffusion_model.transformer_blocks.0.attn1.to_q";
        let mut paths = Vec::new();
        for (index, rank) in [2usize, 4usize].into_iter().enumerate() {
            let path = temp_file(&format!("stack-{index}"));
            let a_data = vec![0u8; rank * 4 * std::mem::size_of::<f32>()];
            let b_data = vec![0u8; 8 * rank * std::mem::size_of::<f32>()];
            let mut tensors = HashMap::new();
            tensors.insert(
                format!("{key}.lora_A.weight"),
                TensorView::new(SafeDtype::F32, vec![rank, 4], &a_data).unwrap(),
            );
            tensors.insert(
                format!("{key}.lora_B.weight"),
                TensorView::new(SafeDtype::F32, vec![8, rank], &b_data).unwrap(),
            );
            serialize_to_file(&tensors, &None, &path).unwrap();
            paths.push(path);
        }

        let registry = load_lora_registry(&[
            LoraWeight {
                path: paths[0].to_string_lossy().to_string(),
                scale: 0.8,

                expert: None,
            },
            LoraWeight {
                path: paths[1].to_string_lossy().to_string(),
                scale: 0.5,

                expert: None,
            },
        ])
        .unwrap()
        .unwrap();

        let adapters = registry.adapters_for(key);
        assert_eq!(adapters.len(), 2, "both adapters must stack on the layer");
        // No `alpha` tensor, so the user scale passes through untouched and
        // pins the order.
        assert!((adapters[0].scale - 0.8).abs() < 1e-6);
        assert!((adapters[1].scale - 0.5).abs() < 1e-6);
        assert_eq!(registry.layer_count(), 1);

        for path in paths {
            let _ = std::fs::remove_file(path);
        }
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

            expert: None,
        }])
        .unwrap();

        assert_eq!(scale, 2);
        let _ = std::fs::remove_file(path);
    }

    #[test]
    fn reference_video_downscale_factor_defaults_to_one() {
        let path = temp_file("ref-scale-default");
        let data = vec![0u8; 4 * std::mem::size_of::<f32>()];
        let tensors = HashMap::from([(
            "diffusion_model.transformer_blocks.0.attn1.to_q.lora_A.weight".to_string(),
            TensorView::new(SafeDtype::F32, vec![1, 4], &data).unwrap(),
        )]);
        serialize_to_file(&tensors, &None, &path).unwrap();

        let scale = reference_video_downscale_factor(&[LoraWeight {
            path: path.to_string_lossy().to_string(),
            scale: 1.0,

            expert: None,
        }])
        .unwrap();

        assert_eq!(scale, 1);
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

                expert: None,
            },
            LoraWeight {
                path: path_two.to_string_lossy().to_string(),
                scale: 1.0,

                expert: None,
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
