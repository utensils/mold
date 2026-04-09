use mold_core::{Config, Scheduler};

/// Capabilities and defaults derived from a model's family.
#[derive(Debug, Clone)]
pub struct ModelCapabilities {
    /// Whether the model supports negative prompts (CFG-based models).
    pub supports_negative_prompt: bool,
    /// Whether the model supports scheduler selection.
    pub supports_scheduler: bool,
    /// Whether the model supports img2img.
    pub supports_img2img: bool,
    /// Whether the model accepts a source/reference image.
    pub supports_source_image: bool,
    /// Whether the model uses a denoising strength control.
    pub supports_strength: bool,
    /// Whether the model supports a mask image.
    pub supports_mask: bool,
    /// Whether the model supports ControlNet.
    pub supports_controlnet: bool,
    /// Whether the model supports LoRA adapters.
    pub supports_lora: bool,
    /// Whether the model is a video model (supports frames/fps params).
    pub supports_video: bool,
    /// Default scheduler for UNet-based models.
    pub default_scheduler: Option<Scheduler>,
}

/// Determine model capabilities from family name.
pub fn capabilities_for_family(family: &str) -> ModelCapabilities {
    match family {
        "sd15" | "sd1.5" | "stable-diffusion-1.5" => ModelCapabilities {
            supports_negative_prompt: true,
            supports_scheduler: true,
            supports_img2img: true,
            supports_source_image: true,
            supports_strength: true,
            supports_mask: true,
            supports_controlnet: true,
            supports_lora: false,
            supports_video: false,
            default_scheduler: Some(Scheduler::Ddim),
        },
        "sdxl" => ModelCapabilities {
            supports_negative_prompt: true,
            supports_scheduler: true,
            supports_img2img: true,
            supports_source_image: true,
            supports_strength: true,
            supports_mask: true,
            supports_controlnet: false,
            supports_lora: false,
            supports_video: false,
            default_scheduler: Some(Scheduler::Ddim),
        },
        "sd3" | "sd3.5" | "stable-diffusion-3" => ModelCapabilities {
            supports_negative_prompt: true,
            supports_scheduler: false,
            supports_img2img: false,
            supports_source_image: false,
            supports_strength: false,
            supports_mask: false,
            supports_controlnet: false,
            supports_lora: false,
            supports_video: false,
            default_scheduler: None,
        },
        "wuerstchen" | "wuerstchen-v2" => ModelCapabilities {
            supports_negative_prompt: true,
            supports_scheduler: false,
            supports_img2img: false,
            supports_source_image: false,
            supports_strength: false,
            supports_mask: false,
            supports_controlnet: false,
            supports_lora: false,
            supports_video: false,
            default_scheduler: None,
        },
        "flux" => ModelCapabilities {
            supports_negative_prompt: false,
            supports_scheduler: false,
            supports_img2img: true,
            supports_source_image: true,
            supports_strength: true,
            supports_mask: true,
            supports_controlnet: false, // ControlNet only supported on SD1.5
            supports_lora: true,
            supports_video: false,
            default_scheduler: None,
        },
        "flux2" | "flux.2" | "flux2-klein" => ModelCapabilities {
            supports_negative_prompt: false,
            supports_scheduler: false,
            supports_img2img: false,
            supports_source_image: false,
            supports_strength: false,
            supports_mask: false,
            supports_controlnet: false,
            supports_lora: false,
            supports_video: false,
            default_scheduler: None,
        },
        "z-image" => ModelCapabilities {
            supports_negative_prompt: false,
            supports_scheduler: false,
            supports_img2img: false,
            supports_source_image: false,
            supports_strength: false,
            supports_mask: false,
            supports_controlnet: false,
            supports_lora: false,
            supports_video: false,
            default_scheduler: None,
        },
        "qwen-image" | "qwen_image" => ModelCapabilities {
            supports_negative_prompt: true,
            supports_scheduler: false,
            supports_img2img: false,
            supports_source_image: false,
            supports_strength: false,
            supports_mask: false,
            supports_controlnet: false,
            supports_lora: false,
            supports_video: false,
            default_scheduler: None,
        },
        "qwen-image-edit" => ModelCapabilities {
            supports_negative_prompt: true,
            supports_scheduler: false,
            supports_img2img: false,
            supports_source_image: true,
            supports_strength: false,
            supports_mask: false,
            supports_controlnet: false,
            supports_lora: false,
            supports_video: false,
            default_scheduler: None,
        },
        "ltx-video" => ModelCapabilities {
            supports_negative_prompt: false,
            supports_scheduler: false,
            supports_img2img: false,
            supports_source_image: false,
            supports_strength: false,
            supports_mask: false,
            supports_controlnet: false,
            supports_lora: false,
            supports_video: true,
            default_scheduler: None,
        },
        _ => ModelCapabilities {
            supports_negative_prompt: false,
            supports_scheduler: false,
            supports_img2img: false,
            supports_source_image: false,
            supports_strength: false,
            supports_mask: false,
            supports_controlnet: false,
            supports_lora: false,
            supports_video: false,
            default_scheduler: None,
        },
    }
}

/// Resolve the family string for a given model name using the config and manifest.
pub fn family_for_model(model_name: &str, config: &Config) -> String {
    let model_cfg = config.resolved_model_config(model_name);
    model_cfg
        .family
        .clone()
        .or_else(|| mold_core::manifest::find_manifest(model_name).map(|m| m.family.clone()))
        .unwrap_or_else(|| "flux".to_string())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn flux_does_not_support_controlnet() {
        let caps = capabilities_for_family("flux");
        assert!(!caps.supports_controlnet);
    }

    #[test]
    fn sd15_supports_controlnet() {
        let caps = capabilities_for_family("sd15");
        assert!(caps.supports_controlnet);
    }

    #[test]
    fn sd15_supports_negative_and_scheduler() {
        let caps = capabilities_for_family("sd15");
        assert!(caps.supports_negative_prompt);
        assert!(caps.supports_scheduler);
        assert!(caps.default_scheduler.is_some());
    }

    #[test]
    fn flux_supports_lora_and_img2img() {
        let caps = capabilities_for_family("flux");
        assert!(caps.supports_lora);
        assert!(caps.supports_img2img);
    }

    #[test]
    fn flux2_has_minimal_capabilities() {
        let caps = capabilities_for_family("flux2");
        assert!(!caps.supports_negative_prompt);
        assert!(!caps.supports_scheduler);
        assert!(!caps.supports_img2img);
        assert!(!caps.supports_controlnet);
        assert!(!caps.supports_lora);
    }

    #[test]
    fn unknown_family_defaults_to_minimal() {
        let caps = capabilities_for_family("unknown-model");
        assert!(!caps.supports_negative_prompt);
        assert!(!caps.supports_scheduler);
        assert!(!caps.supports_controlnet);
    }

    #[test]
    fn sdxl_supports_negative_and_scheduler_but_not_controlnet() {
        let caps = capabilities_for_family("sdxl");
        assert!(caps.supports_negative_prompt);
        assert!(caps.supports_scheduler);
        assert!(!caps.supports_controlnet);
        assert!(!caps.supports_lora);
    }

    #[test]
    fn qwen_image_supports_negative_only() {
        let caps = capabilities_for_family("qwen-image");
        assert!(caps.supports_negative_prompt);
        assert!(!caps.supports_scheduler);
        assert!(!caps.supports_controlnet);
    }

    #[test]
    fn qwen_image_edit_supports_source_image_without_img2img_controls() {
        let caps = capabilities_for_family("qwen-image-edit");
        assert!(caps.supports_negative_prompt);
        assert!(caps.supports_source_image);
        assert!(!caps.supports_strength);
        assert!(!caps.supports_mask);
    }

    #[test]
    fn ltx_video_supports_video() {
        let caps = capabilities_for_family("ltx-video");
        assert!(caps.supports_video);
        assert!(!caps.supports_negative_prompt);
    }
}
