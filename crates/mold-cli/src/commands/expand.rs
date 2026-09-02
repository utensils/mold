//! `mold expand` — preview prompt expansion without generating images.

use anyhow::Result;
use colored::Colorize;
use mold_core::{
    Config, ExpandContext, ExpandReference, ExpandReferenceRole, ExpandSettings,
    GenerationReferenceKind, PromptExpander,
};

use crate::output::status;
use crate::theme;

/// Create a progress callback for the local expand model that prints status
/// messages matching the existing engine progress output style.
#[cfg(feature = "expand")]
fn expand_progress_callback() -> mold_inference::progress::ProgressCallback {
    use mold_inference::progress::ProgressEvent;
    Box::new(move |event: ProgressEvent| match event {
        ProgressEvent::StageDone { name, elapsed } => {
            let secs = elapsed.as_secs_f64();
            status!(
                "  {} {} {}",
                theme::icon_done(),
                name,
                format!("[{:.1}s]", secs).dimmed()
            );
        }
        ProgressEvent::Info { message } => {
            status!("  {} {}", theme::icon_bullet(), message.dimmed());
        }
        _ => {}
    })
}

#[allow(clippy::too_many_arguments)]
pub async fn run(
    prompt: &str,
    model: Option<&str>,
    variations: usize,
    json_output: bool,
    backend_override: Option<&str>,
    model_override: Option<&str>,
    task_override: Option<&str>,
    context: Option<ExpandContext>,
) -> Result<()> {
    mold_core::expand::validate_expansion_variation_count(variations)?;
    let mut config = Config::load_or_default();
    let mut expand_settings = config.expand.clone().with_env_overrides();

    // Apply overrides
    if let Some(backend) = backend_override {
        expand_settings.backend = backend.to_string();
    }
    if let Some(m) = model_override {
        if expand_settings.is_local() {
            expand_settings.model = m.to_string();
        } else {
            expand_settings.api_model = m.to_string();
        }
    }

    // Validate custom templates if present
    let template_errors = expand_settings.validate_templates();
    if !template_errors.is_empty() {
        for err in &template_errors {
            eprintln!("{} {err}", theme::prefix_warning());
        }
    }

    // Determine model family for prompt style
    let model_family = if let Some(model_name) = model {
        crate::catalog_bridge::ensure_catalog_model(&mut config, model_name).await?;
        resolve_family(model_name, &config)
    } else {
        "flux".to_string() // Default to FLUX-style prompts
    };

    // A family that reads no prompt has nothing to expand. Answer before an
    // expansion model is created (or pulled): the guide's image advice is
    // the whole answer, and no language model runs.
    if let Some(advice) = mold_core::ignored_prompt_advice(&model_family) {
        if json_output {
            println!("{}", serde_json::to_string_pretty(&[advice.text()])?);
        } else {
            print_ignored_prompt_advice(&advice);
        }
        return Ok(());
    }

    let mut expand_config = expand_settings.to_expand_config(&model_family, variations);
    if let Some(task) = task_override {
        expand_config.task = task
            .parse::<mold_core::ExpandTask>()
            .map_err(anyhow::Error::msg)?;
    }
    expand_config.context = context;

    // Get expander (auto-pulls expand model if needed)
    let expander = create_expander(&expand_settings, &config).await?;

    if !json_output {
        status!(
            "{} Expanding prompt for {} family...",
            theme::icon_info(),
            model_family.bold()
        );
    }

    let result = expander.expand(prompt, &expand_config)?;
    mold_core::expand::validate_expanded_prompts(&result.expanded, variations)?;

    if json_output {
        let json =
            serde_json::to_string_pretty(&result.expanded).unwrap_or_else(|_| "[]".to_string());
        println!("{json}");
    } else {
        if variations == 1 {
            println!("{}", result.expanded[0]);
        } else {
            for (i, expanded) in result.expanded.iter().enumerate() {
                status!("{} Variation {}:", theme::icon_ok(), i + 1);
                println!("{expanded}");
                println!();
            }
        }
    }

    Ok(())
}

/// `mold expand` / `mold remix` on a family whose profile ignores the prompt:
/// the one-line reason goes to stderr like every other status line, and the
/// guide's image-preparation advice is the stdout answer.
pub(crate) fn print_ignored_prompt_advice(advice: &mold_core::IgnoredPromptAdvice) {
    status!("{} {}", theme::icon_info(), advice.headline);
    println!("{}", advice.preparation);
}

/// Create the appropriate expander based on settings.
///
/// When the local backend is selected and the expand model hasn't been pulled
/// yet, this will auto-pull it (same pattern as diffusion model auto-pull in
/// `generate.rs`).
pub(crate) async fn create_expander(
    settings: &ExpandSettings,
    config: &Config,
) -> Result<Box<dyn PromptExpander>> {
    if let Some(api_expander) = settings.create_api_expander()? {
        return Ok(Box::new(api_expander));
    }

    // Local expander
    #[cfg(feature = "expand")]
    {
        if let Some(mut local) =
            mold_inference::expand::LocalExpander::from_config(config, Some(&settings.model))
        {
            local.set_on_progress(expand_progress_callback());
            return Ok(Box::new(local));
        }

        // Auto-pull: if a manifest exists for the expand model, download it
        let expand_model = &settings.model;
        if let Some(manifest) = mold_core::manifest::find_manifest(expand_model) {
            status!(
                "{} Expand model '{}' not found locally, pulling...",
                theme::icon_info(),
                manifest.name.bold(),
            );
            super::pull::pull_and_configure(
                expand_model,
                &mold_core::download::PullOptions::default(),
            )
            .await?;

            // Reload config after pull and retry
            let updated_config = Config::load_or_default();
            if let Some(mut local) = mold_inference::expand::LocalExpander::from_config(
                &updated_config,
                Some(&settings.model),
            ) {
                local.set_on_progress(expand_progress_callback());
                return Ok(Box::new(local));
            }
        }

        anyhow::bail!(
            "local expand model not found and auto-pull failed.\n\
             Try manually: mold pull qwen3-expand\n\
             Or use an API backend: --expand-backend http://localhost:11434"
        );
    }

    #[cfg(not(feature = "expand"))]
    {
        let _ = config; // suppress unused warning
        anyhow::bail!(
            "local prompt expansion not available — this binary was built without the `expand` feature.\n\
             Use an API backend instead: --expand-backend http://localhost:11434"
        );
    }
}

/// Build the expander's generation context from `mold expand` / `mold remix`
/// flags. Returns `None` when no fact was given so old behaviour is kept.
pub(crate) fn context_from_flags(
    model: Option<&str>,
    width: Option<u32>,
    height: Option<u32>,
    frames: Option<u32>,
    fps: Option<u32>,
    clip_frames: Option<u32>,
    references: &[String],
) -> Result<Option<ExpandContext>> {
    let mut parsed = Vec::new();
    for spec in references {
        let (kind, role) = spec
            .split_once(':')
            .map_or((spec.as_str(), None), |(kind, role)| (kind, Some(role)));
        let kind = match kind.trim().to_ascii_lowercase().as_str() {
            "image" | "picture" => GenerationReferenceKind::Image,
            "video" => GenerationReferenceKind::Video,
            "audio" => GenerationReferenceKind::Audio,
            other => anyhow::bail!(
                "unknown reference kind '{other}' in --reference {spec}. Valid: image, video, audio"
            ),
        };
        let role = match role.map(|role| role.trim().to_ascii_lowercase()) {
            None => None,
            Some(role) => Some(match role.as_str() {
                "first-frame" | "first" => ExpandReferenceRole::FirstFrame,
                "last-frame" | "last" => ExpandReferenceRole::LastFrame,
                "keyframe" => ExpandReferenceRole::Keyframe,
                "source" => ExpandReferenceRole::Source,
                "identity" | "id" => ExpandReferenceRole::Identity,
                "edit" => ExpandReferenceRole::Edit,
                "reference" => ExpandReferenceRole::Reference,
                other => anyhow::bail!(
                    "unknown reference role '{other}' in --reference {spec}. Valid: first-frame, last-frame, keyframe, source, identity, edit, reference"
                ),
            }),
        };
        parsed.push(ExpandReference {
            kind,
            has_audio: false,
            role,
        });
    }
    if model.is_none()
        && width.is_none()
        && height.is_none()
        && frames.is_none()
        && fps.is_none()
        && clip_frames.is_none()
        && parsed.is_empty()
    {
        return Ok(None);
    }
    Ok(Some(ExpandContext {
        model: model.map(mold_core::manifest::resolve_model_name),
        width,
        height,
        frames,
        fps,
        clip_frames,
        negative_prompt_supported: None,
        audio: None,
        references: parsed,
        loras: Vec::new(),
        prompt_mode: None,
    }))
}

/// Resolve the model family string from a model name (public for use from run.rs).
pub(crate) fn resolve_family_from_config(model_name: &str, config: &Config) -> String {
    resolve_family(model_name, config)
}

/// Resolve the model family string from a model name.
fn resolve_family(model_name: &str, config: &Config) -> String {
    let model_cfg = config.resolved_model_config(model_name);
    if let Some(family) = model_cfg.family {
        return family;
    }
    if let Some(manifest) = mold_core::manifest::find_manifest(model_name) {
        return manifest.family.clone();
    }
    // Default to flux
    "flux".to_string()
}
