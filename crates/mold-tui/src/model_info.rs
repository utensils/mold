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
    /// Whether the model supports ControlNet.
    pub supports_controlnet: bool,
    /// Whether the model supports LoRA adapters.
    pub supports_lora: bool,
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
            supports_controlnet: true,
            supports_lora: false,
            default_scheduler: Some(Scheduler::Ddim),
        },
        "sdxl" => ModelCapabilities {
            supports_negative_prompt: true,
            supports_scheduler: true,
            supports_img2img: true,
            supports_controlnet: false,
            supports_lora: false,
            default_scheduler: Some(Scheduler::Ddim),
        },
        "sd3" | "sd3.5" | "stable-diffusion-3" => ModelCapabilities {
            supports_negative_prompt: true,
            supports_scheduler: false,
            supports_img2img: false,
            supports_controlnet: false,
            supports_lora: false,
            default_scheduler: None,
        },
        "wuerstchen" | "wuerstchen-v2" => ModelCapabilities {
            supports_negative_prompt: true,
            supports_scheduler: false,
            supports_img2img: false,
            supports_controlnet: false,
            supports_lora: false,
            default_scheduler: None,
        },
        "flux" => ModelCapabilities {
            supports_negative_prompt: false,
            supports_scheduler: false,
            supports_img2img: true,
            supports_controlnet: true,
            supports_lora: true,
            default_scheduler: None,
        },
        "flux2" | "flux.2" | "flux2-klein" => ModelCapabilities {
            supports_negative_prompt: false,
            supports_scheduler: false,
            supports_img2img: false,
            supports_controlnet: false,
            supports_lora: false,
            default_scheduler: None,
        },
        "z-image" => ModelCapabilities {
            supports_negative_prompt: false,
            supports_scheduler: false,
            supports_img2img: false,
            supports_controlnet: false,
            supports_lora: false,
            default_scheduler: None,
        },
        "qwen-image" | "qwen_image" => ModelCapabilities {
            supports_negative_prompt: true,
            supports_scheduler: false,
            supports_img2img: false,
            supports_controlnet: false,
            supports_lora: false,
            default_scheduler: None,
        },
        _ => ModelCapabilities {
            supports_negative_prompt: false,
            supports_scheduler: false,
            supports_img2img: false,
            supports_controlnet: false,
            supports_lora: false,
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
        .or_else(|| {
            mold_core::manifest::find_manifest(model_name).map(|m| m.family.clone())
        })
        .unwrap_or_else(|| "flux".to_string())
}
