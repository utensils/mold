//! Original scalar recipe projection for newly added Library footage.

use crate::{GenerateRequest, LoraWeight, OutputFormat, OutputMetadata};

/// Sampling settings for newly added footage, never authority for the original
/// movie or retained inputs. Callers must bind an authenticated Library source
/// and obtain a fresh server validation proof before admitting this request.
/// Historical endpoints, local paths, seed, placement, filing and prompt/batch
/// provenance are deliberately absent. The source archive retains them itself.
pub fn extension_generation(
    metadata: &OutputMetadata,
    prompt: &str,
) -> Result<GenerateRequest, String> {
    let (width, height) = if metadata.spatial_upscale.is_some() {
        (
            metadata.generation_width.unwrap_or(metadata.width),
            metadata.generation_height.unwrap_or(metadata.height),
        )
    } else {
        (metadata.width, metadata.height)
    };
    let mut request: GenerateRequest = serde_json::from_value(serde_json::json!({
        "model": metadata.model, "prompt":"", "width":width,"height":height,
        "steps":metadata.steps,"guidance":metadata.guidance,"output_format":"mp4"
    }))
    .map_err(|error| format!("Cannot restore this video's scalar recipe: {error}"))?;
    // Typed assignment preserves literal backslashes and CRLF; this new protocol
    // never sends source/new prompts through the legacy escape decoder again.
    request.prompt = prompt.to_owned();
    request.negative_prompt = metadata.negative_prompt.clone();
    request.scheduler = metadata.scheduler;
    request.cfg_plus = metadata.cfg_plus;
    request.strength = metadata.strength.unwrap_or(request.strength);
    request.reference_weight = metadata.reference_weight;
    request.true_cfg = metadata.true_cfg;
    request.cfg_start_step = metadata.cfg_start_step;
    request.control_model = metadata.control_model.clone();
    request.control_scale = metadata.control_scale.unwrap_or(request.control_scale);
    request.loras = metadata.loras.clone().or_else(|| {
        metadata.lora.as_ref().map(|path| {
            vec![LoraWeight {
                path: path.clone(),
                scale: metadata.lora_scale.unwrap_or(1.0),
                expert: None,
            }]
        })
    });
    // The original movie length is not a new bounded sampling window.
    request.frames = None;
    request.fps = metadata.fps;
    request.enable_audio = metadata.enable_audio;
    request.video_only = metadata.video_only;
    request.pipeline = metadata.pipeline;
    request.ic_lora_control = metadata.ic_lora_control.clone();
    request.spatial_upscale = metadata.spatial_upscale;
    request.temporal_upscale = metadata.temporal_upscale;
    request.guidance_overrides = metadata.guidance_overrides.clone();
    request.sample_shift = metadata.sample_shift;
    request.distill_strength_high = metadata.distill_strength_high;
    request.distill_strength_low = metadata.distill_strength_low;
    request.output_format = Some(OutputFormat::Mp4);
    request.save_to_gallery = Some(true);
    request.embed_metadata = Some(true);
    Ok(request)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{GenerateRequest, OutputMetadata};

    fn metadata() -> OutputMetadata {
        let request: GenerateRequest = serde_json::from_value(serde_json::json!({
            "model":"installed-video","prompt":"original","width":768,"height":512,
            "steps":9,"guidance":1.5,"seed":18446744073709551615u64,"fps":24,"frames":107,
            "negative_prompt":"avoid","cfg_plus":true,"enable_audio":true,
            "spatial_upscale":"x2","sample_shift":3.2,"distill_strength_high":0.7,
            "batch_size":4,"batch_id":"old","batch_index":2,"batch_count":4,
            "source_image":"AQID","keyframes":[{"frame":106,"image":"BAUG"}],
            "audio_file_path":"/private/old.wav","extend_video_path":"/private/old.mp4",
            "extend_overlap_frames":17,"title":"Old title"
        }))
        .unwrap();
        OutputMetadata::from_generate_request(&request, u64::MAX, None, "test")
    }
    #[test]
    fn preserves_recipe_but_not_old_input_seed_batch_or_prompt_provenance() {
        let mut original = metadata();
        original.negative_prompt = Some(r"literal\new".into());
        let before = serde_json::to_value(&original).unwrap();
        let request = extension_generation(&original, r"new\night\scene").unwrap();
        assert_eq!(request.prompt, r"new\night\scene");
        assert_eq!(request.negative_prompt.as_deref(), Some(r"literal\new"));
        assert_eq!(
            (
                request.model.as_str(),
                request.width,
                request.height,
                request.fps
            ),
            ("installed-video", 768, 512, Some(24))
        );
        assert_eq!(request.steps, 9);
        assert_eq!(request.enable_audio, Some(true));
        assert_eq!(request.spatial_upscale, original.spatial_upscale);
        assert_eq!(request.sample_shift, Some(3.2));
        assert_eq!(request.distill_strength_high, Some(0.7));
        assert!(request.seed.is_none());
        assert!(request.frames.is_none());
        assert_eq!(request.batch_size, 1);
        assert!(
            request.batch_id.is_none()
                && request.batch_index.is_none()
                && request.batch_count.is_none()
        );
        assert!(
            request.source_image.is_none()
                && request.keyframes.is_none()
                && request.references.is_none()
        );
        assert!(
            request.audio_file_path.is_none()
                && request.extend_video_path.is_none()
                && request.extend_overlap_frames.is_none()
        );
        assert!(
            request.original_prompt.is_none()
                && request.prompt_transform.is_none()
                && request.placement.is_none()
        );
        assert!(request.title.is_none());
        assert_eq!(serde_json::to_value(&original).unwrap(), before);
    }
    #[test]
    fn uses_original_not_postprocessed_canvas_for_new_sampling() {
        let mut original = metadata();
        original.width = 1536;
        original.height = 1024;
        original.generation_width = Some(768);
        original.generation_height = Some(512);
        let request = extension_generation(&original, "Next").unwrap();
        // The parent owns exact prefix/output normalization; sampling retains the original recipe.
        assert_eq!((request.width, request.height), (768, 512));
    }
    #[test]
    fn preserves_lora_stack_and_expert_controls_without_making_input_authority() {
        let mut original = metadata();
        original.loras = Some(vec![LoraWeight {
            path: "adapter.safetensors".into(),
            scale: 0.6,
            expert: Some(crate::LoraExpert::High),
        }]);
        original.control_model = Some("original-control".into());
        original.control_scale = Some(0.75);
        original.frames = Some(86400);
        let request = extension_generation(&original, "Next").unwrap();
        assert_eq!(request.loras, original.loras);
        assert_eq!(request.control_model, original.control_model);
        assert_eq!(request.control_scale, 0.75);
        assert!(request.control_image.is_none());
        assert!(request.frames.is_none());
    }
}
