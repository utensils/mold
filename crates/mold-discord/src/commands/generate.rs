use crate::handler;
use crate::state::Context;
use anyhow::Result;
use mold_core::{GenerateRequest, OutputFormat};
use std::time::Duration;

/// Autocomplete function for model names.
async fn autocomplete_model(ctx: Context<'_>, partial: &str) -> Vec<String> {
    let models = ctx.data().cached_models().await;
    let lower = partial.to_lowercase();
    models
        .iter()
        .filter(|m| m.downloaded)
        .filter(|m| m.info.name.to_lowercase().contains(&lower))
        .map(|m| m.info.name.clone())
        .take(25) // Discord autocomplete limit
        .collect()
}

/// Build a GenerateRequest from slash command parameters, using model defaults.
#[allow(clippy::too_many_arguments)]
pub fn build_generate_request(
    prompt: &str,
    model: &str,
    width: Option<u32>,
    height: Option<u32>,
    steps: Option<u32>,
    guidance: Option<f64>,
    seed: Option<u64>,
    defaults: Option<&mold_core::ModelDefaults>,
) -> GenerateRequest {
    let (def_w, def_h, def_steps, def_guidance) = match defaults {
        Some(d) => (
            d.default_width,
            d.default_height,
            d.default_steps,
            d.default_guidance,
        ),
        None => (1024, 1024, 20, 3.5),
    };

    GenerateRequest {
        prompt: prompt.to_string(),
        model: model.to_string(),
        width: width.unwrap_or(def_w),
        height: height.unwrap_or(def_h),
        steps: steps.unwrap_or(def_steps),
        guidance: guidance.unwrap_or(def_guidance),
        seed,
        batch_size: 1,
        output_format: OutputFormat::Png,
        embed_metadata: None,
        scheduler: None,
        source_image: None,
        strength: 0.75,
        mask_image: None,
        control_image: None,
        control_model: None,
        control_scale: 1.0,
    }
}

/// Resolve the default model name from the cached model list.
/// Prefers: loaded model > smallest downloaded model > "flux-schnell:q8" fallback.
fn resolve_default_model(models: &[mold_core::ModelInfoExtended]) -> String {
    // Prefer the currently loaded model
    if let Some(loaded) = models.iter().find(|m| m.info.is_loaded) {
        return loaded.info.name.clone();
    }
    // Fall back to the smallest downloaded generative model (avoids accidentally
    // picking a 23GB BF16 variant or a ControlNet auxiliary model).
    if let Some(downloaded) = models
        .iter()
        .filter(|m| m.downloaded && m.info.family != "controlnet")
        .min_by(|a, b| {
            a.info
                .size_gb
                .partial_cmp(&b.info.size_gb)
                .unwrap_or(std::cmp::Ordering::Equal)
        })
    {
        return downloaded.info.name.clone();
    }
    // Last resort
    "flux-schnell:q8".to_string()
}

/// Generate an AI image from a text prompt.
#[allow(clippy::too_many_arguments)]
#[poise::command(slash_command)]
pub async fn generate(
    ctx: Context<'_>,
    #[description = "Text prompt describing the image to generate"] prompt: String,
    #[description = "Model to use (e.g. flux-schnell:q8)"]
    #[autocomplete = "autocomplete_model"]
    model: Option<String>,
    #[description = "Image width in pixels"] width: Option<u32>,
    #[description = "Image height in pixels"] height: Option<u32>,
    #[description = "Number of inference steps"] steps: Option<u32>,
    #[description = "Guidance scale (0.0 for schnell, ~3.5 for dev)"] guidance: Option<f64>,
    #[description = "Random seed for reproducibility"] seed: Option<u64>,
) -> Result<()> {
    // Validate prompt before deferring (avoids wasting the interaction)
    if prompt.trim().is_empty() {
        ctx.send(
            poise::CreateReply::default()
                .content("Prompt cannot be empty.")
                .ephemeral(true),
        )
        .await?;
        return Ok(());
    }

    // Check cooldown
    let user_id = ctx.author().id.get();
    let cooldown = Duration::from_secs(ctx.data().config.cooldown_seconds);
    if let Err(remaining) = ctx.data().cooldowns.check(user_id, cooldown) {
        let secs = remaining.as_secs() + 1;
        ctx.send(
            poise::CreateReply::default()
                .content(format!(
                    "Please wait {secs} seconds before generating again."
                ))
                .ephemeral(true),
        )
        .await?;
        return Ok(());
    }

    // Defer the response (shows "Bot is thinking...")
    ctx.defer().await?;

    // Resolve model name — use server's loaded/downloaded model if none specified
    let models = ctx.data().cached_models().await;
    let model_name = model.unwrap_or_else(|| resolve_default_model(&models));

    // Look up model defaults from cache
    let model_defaults = models
        .iter()
        .find(|m| m.info.name == model_name)
        .map(|m| &m.defaults);

    let req = build_generate_request(
        &prompt,
        &model_name,
        width,
        height,
        steps,
        guidance,
        seed,
        model_defaults,
    );

    match handler::run_generation(ctx, req).await {
        Ok(()) => {
            ctx.data().cooldowns.record(user_id);
        }
        Err(e) => {
            let msg = if mold_core::MoldClient::is_connection_error(&e) {
                "Could not connect to the mold server. Is it running?".to_string()
            } else if mold_core::MoldClient::is_model_not_found(&e) {
                format!("Model '{model_name}' is not downloaded. Use `/models` to see available models.")
            } else {
                format!("Generation failed: {e}")
            };
            handler::send_error(ctx, &msg).await?;
        }
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn build_request_with_defaults() {
        let defaults = mold_core::ModelDefaults {
            default_steps: 4,
            default_guidance: 0.0,
            default_width: 1024,
            default_height: 1024,
            description: "test".to_string(),
        };
        let req = build_generate_request(
            "a cat",
            "flux-schnell:q8",
            None,
            None,
            None,
            None,
            None,
            Some(&defaults),
        );
        assert_eq!(req.prompt, "a cat");
        assert_eq!(req.model, "flux-schnell:q8");
        assert_eq!(req.width, 1024);
        assert_eq!(req.height, 1024);
        assert_eq!(req.steps, 4);
        assert_eq!(req.guidance, 0.0);
        assert!(req.seed.is_none());
        assert_eq!(req.batch_size, 1);
    }

    #[test]
    fn build_request_overrides() {
        let defaults = mold_core::ModelDefaults {
            default_steps: 4,
            default_guidance: 0.0,
            default_width: 1024,
            default_height: 1024,
            description: "test".to_string(),
        };
        let req = build_generate_request(
            "a dog",
            "flux-dev:q4",
            Some(768),
            Some(512),
            Some(28),
            Some(7.5),
            Some(42),
            Some(&defaults),
        );
        assert_eq!(req.width, 768);
        assert_eq!(req.height, 512);
        assert_eq!(req.steps, 28);
        assert_eq!(req.guidance, 7.5);
        assert_eq!(req.seed, Some(42));
    }

    #[test]
    fn build_request_no_defaults() {
        let req =
            build_generate_request("test", "unknown-model", None, None, None, None, None, None);
        assert_eq!(req.width, 1024);
        assert_eq!(req.height, 1024);
        assert_eq!(req.steps, 20);
        assert_eq!(req.guidance, 3.5);
    }

    #[test]
    fn build_request_partial_overrides() {
        let defaults = mold_core::ModelDefaults {
            default_steps: 4,
            default_guidance: 0.0,
            default_width: 512,
            default_height: 512,
            description: "test".to_string(),
        };
        // Override only width, rest from defaults
        let req = build_generate_request(
            "test",
            "sd15:fp16",
            Some(768),
            None,
            None,
            None,
            None,
            Some(&defaults),
        );
        assert_eq!(req.width, 768);
        assert_eq!(req.height, 512); // from defaults
        assert_eq!(req.steps, 4); // from defaults
    }

    #[test]
    fn resolve_default_prefers_loaded() {
        let models = vec![
            mold_core::ModelInfoExtended {
                info: mold_core::ModelInfo {
                    name: "flux-dev:q4".to_string(),
                    family: "flux".to_string(),
                    size_gb: 4.0,
                    is_loaded: false,
                    last_used: None,
                    hf_repo: "test/repo".to_string(),
                },
                defaults: mold_core::ModelDefaults {
                    default_steps: 25,
                    default_guidance: 3.5,
                    default_width: 1024,
                    default_height: 1024,
                    description: "test".to_string(),
                },
                downloaded: true,
                disk_usage_bytes: None,
                remaining_download_bytes: None,
            },
            mold_core::ModelInfoExtended {
                info: mold_core::ModelInfo {
                    name: "flux-schnell:q8".to_string(),
                    family: "flux".to_string(),
                    size_gb: 4.5,
                    is_loaded: true,
                    last_used: None,
                    hf_repo: "test/repo".to_string(),
                },
                defaults: mold_core::ModelDefaults {
                    default_steps: 4,
                    default_guidance: 0.0,
                    default_width: 1024,
                    default_height: 1024,
                    description: "test".to_string(),
                },
                downloaded: true,
                disk_usage_bytes: None,
                remaining_download_bytes: None,
            },
        ];
        assert_eq!(resolve_default_model(&models), "flux-schnell:q8");
    }

    #[test]
    fn resolve_default_falls_back_to_downloaded() {
        let models = vec![mold_core::ModelInfoExtended {
            info: mold_core::ModelInfo {
                name: "flux-dev:q4".to_string(),
                family: "flux".to_string(),
                size_gb: 4.0,
                is_loaded: false,
                last_used: None,
                hf_repo: "test/repo".to_string(),
            },
            defaults: mold_core::ModelDefaults {
                default_steps: 25,
                default_guidance: 3.5,
                default_width: 1024,
                default_height: 1024,
                description: "test".to_string(),
            },
            downloaded: true,
            disk_usage_bytes: None,
            remaining_download_bytes: None,
        }];
        assert_eq!(resolve_default_model(&models), "flux-dev:q4");
    }

    #[test]
    fn resolve_default_empty_list() {
        assert_eq!(resolve_default_model(&[]), "flux-schnell:q8");
    }

    #[test]
    fn resolve_default_picks_smallest_when_multiple_downloaded() {
        let models = vec![
            mold_core::ModelInfoExtended {
                info: mold_core::ModelInfo {
                    name: "flux-schnell:bf16".to_string(),
                    family: "flux".to_string(),
                    size_gb: 22.1,
                    is_loaded: false,
                    last_used: None,
                    hf_repo: "test/repo".to_string(),
                },
                defaults: mold_core::ModelDefaults {
                    default_steps: 4,
                    default_guidance: 0.0,
                    default_width: 1024,
                    default_height: 1024,
                    description: "test".to_string(),
                },
                downloaded: true,
                disk_usage_bytes: None,
                remaining_download_bytes: None,
            },
            mold_core::ModelInfoExtended {
                info: mold_core::ModelInfo {
                    name: "flux-schnell:q8".to_string(),
                    family: "flux".to_string(),
                    size_gb: 4.5,
                    is_loaded: false,
                    last_used: None,
                    hf_repo: "test/repo".to_string(),
                },
                defaults: mold_core::ModelDefaults {
                    default_steps: 4,
                    default_guidance: 0.0,
                    default_width: 1024,
                    default_height: 1024,
                    description: "test".to_string(),
                },
                downloaded: true,
                disk_usage_bytes: None,
                remaining_download_bytes: None,
            },
        ];
        // Should pick q8 (4.5GB) over bf16 (22.1GB)
        assert_eq!(resolve_default_model(&models), "flux-schnell:q8");
    }

    #[test]
    fn resolve_default_skips_controlnet() {
        let models = vec![
            mold_core::ModelInfoExtended {
                info: mold_core::ModelInfo {
                    name: "controlnet-canny-sd15:fp16".to_string(),
                    family: "controlnet".to_string(),
                    size_gb: 0.7,
                    is_loaded: false,
                    last_used: None,
                    hf_repo: "test/repo".to_string(),
                },
                defaults: mold_core::ModelDefaults {
                    default_steps: 25,
                    default_guidance: 7.5,
                    default_width: 512,
                    default_height: 512,
                    description: "test".to_string(),
                },
                downloaded: true,
                disk_usage_bytes: None,
                remaining_download_bytes: None,
            },
            mold_core::ModelInfoExtended {
                info: mold_core::ModelInfo {
                    name: "flux-schnell:q8".to_string(),
                    family: "flux".to_string(),
                    size_gb: 4.5,
                    is_loaded: false,
                    last_used: None,
                    hf_repo: "test/repo".to_string(),
                },
                defaults: mold_core::ModelDefaults {
                    default_steps: 4,
                    default_guidance: 0.0,
                    default_width: 1024,
                    default_height: 1024,
                    description: "test".to_string(),
                },
                downloaded: true,
                disk_usage_bytes: None,
                remaining_download_bytes: None,
            },
        ];
        // Should pick flux-schnell:q8, not the smaller controlnet
        assert_eq!(resolve_default_model(&models), "flux-schnell:q8");
    }
}
