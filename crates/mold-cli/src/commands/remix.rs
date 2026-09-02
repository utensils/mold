//! `mold remix` — preview subject-preserving prompt alternatives.

use anyhow::{Context, Result};
use mold_core::{
    Config, PromptTransformOperation, RemixDimension, RemixResponse, RemixSourceKind, RemixVariant,
};

use crate::{output::status, theme};

#[allow(clippy::too_many_arguments)]
pub async fn run(
    source_prompt: &str,
    model: Option<&str>,
    variations: usize,
    json_output: bool,
    backend_override: Option<&str>,
    model_override: Option<&str>,
    task_override: Option<&str>,
    source_kind: &str,
    root_prompt: Option<&str>,
    dimension_values: &[String],
    style: Option<&str>,
    context: Option<mold_core::ExpandContext>,
) -> Result<()> {
    mold_core::expand::validate_expansion_variation_count(variations)?;
    anyhow::ensure!(
        !source_prompt.trim().is_empty(),
        "source prompt must not be empty"
    );
    let mut config = Config::load_or_default();
    let mut settings = config.expand.clone().with_env_overrides();
    if let Some(backend) = backend_override {
        settings.backend = backend.to_string();
    }
    if let Some(name) = model_override {
        if settings.is_local() {
            settings.model = name.to_string();
        } else {
            settings.api_model = name.to_string();
        }
    }
    let family = if let Some(model_name) = model {
        crate::catalog_bridge::ensure_catalog_model(&mut config, model_name).await?;
        super::expand::resolve_family_from_config(model_name, &config)
    } else {
        "flux".to_string()
    };
    let task = match task_override {
        Some(value) => value.parse().map_err(anyhow::Error::msg)?,
        None => mold_core::ExpandTask::for_family(&family),
    };
    let source_kind = match source_kind.trim().to_ascii_lowercase().as_str() {
        "original" => RemixSourceKind::Original,
        "current" => RemixSourceKind::Current,
        "direct" => RemixSourceKind::Direct,
        value => anyhow::bail!("unknown remix source '{value}'. Valid: original, current, direct"),
    };
    // A family that reads no prompt has nothing to remix either: the same
    // answer `mold expand` gives, as one variant with no varied dimension.
    if let Some(advice) = mold_core::ignored_prompt_advice(&family) {
        if json_output {
            let response = RemixResponse {
                source_prompt: source_prompt.to_string(),
                root_prompt: root_prompt.map(str::to_string),
                source_kind,
                task,
                variants: vec![RemixVariant {
                    prompt: advice.text(),
                    dimensions: Vec::new(),
                }],
            };
            println!("{}", serde_json::to_string_pretty(&response)?);
        } else {
            super::expand::print_ignored_prompt_advice(&advice);
        }
        return Ok(());
    }
    let requested_dimensions = dimension_values
        .iter()
        .flat_map(|value| value.split(','))
        .filter(|value| !value.trim().is_empty())
        .map(|value| value.parse::<RemixDimension>().map_err(anyhow::Error::msg))
        .collect::<Result<Vec<_>>>()?;
    let style = style.map(str::trim).filter(|value| !value.is_empty());
    let dimensions =
        mold_core::expand::resolve_remix_dimensions(&requested_dimensions, task, style.is_some())?;
    let mut remix_config = settings.to_expand_config(&family, variations);
    remix_config.task = task;
    remix_config.operation = PromptTransformOperation::Remix;
    remix_config.remix_dimensions = dimensions.clone();
    remix_config.style = style.map(str::to_string);
    remix_config.context = context;
    remix_config.system_prompt = None;
    remix_config.batch_prompt = None;

    if !json_output {
        status!("{} Remixing prompt for {family}...", theme::icon_info());
    }
    let expander = super::expand::create_expander(&settings, &config).await?;
    let result = expander
        .expand(source_prompt, &remix_config)
        .context("prompt remix failed")?;
    mold_core::expand::validate_expanded_prompts(&result.expanded, variations)?;
    let variants = result
        .expanded
        .into_iter()
        .enumerate()
        .map(|(index, prompt)| RemixVariant {
            prompt,
            dimensions: mold_core::expand::remix_dimensions_for_position(&dimensions, index + 1),
        })
        .collect::<Vec<_>>();
    let response = RemixResponse {
        source_prompt: source_prompt.to_string(),
        root_prompt: root_prompt.map(str::to_string),
        source_kind,
        task,
        variants,
    };
    if json_output {
        println!("{}", serde_json::to_string_pretty(&response)?);
    } else {
        for (index, variant) in response.variants.iter().enumerate() {
            let labels = variant
                .dimensions
                .iter()
                .map(ToString::to_string)
                .collect::<Vec<_>>()
                .join(", ");
            status!("{} Variation {} ({labels}):", theme::icon_ok(), index + 1);
            println!("{}\n", variant.prompt);
        }
    }
    Ok(())
}
