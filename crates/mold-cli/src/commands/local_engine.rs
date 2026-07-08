//! Shared local-engine preamble for `mold run` (single clip) and
//! `mold run --frames N` chains in `--local` mode: resolve-or-pull the
//! model, apply encoder-variant env overrides, resolve eager/offload,
//! pick a GPU, and construct the engine. One home so the generate and
//! chain paths can't drift.

#[cfg(any(feature = "cuda", feature = "metal"))]
use anyhow::Result;
#[cfg(any(feature = "cuda", feature = "metal"))]
use colored::Colorize;
#[cfg(any(feature = "cuda", feature = "metal"))]
use mold_core::{Config, ModelPaths};

#[cfg(any(feature = "cuda", feature = "metal"))]
use crate::output::status;
#[cfg(any(feature = "cuda", feature = "metal"))]
use crate::theme;

/// Engine-construction knobs collected from CLI flags.
#[cfg(any(feature = "cuda", feature = "metal"))]
#[derive(Debug, Default)]
pub(crate) struct EngineOverrides {
    pub gpus: Option<String>,
    pub t5_variant: Option<String>,
    pub qwen3_variant: Option<String>,
    pub qwen2_variant: Option<String>,
    pub qwen2_text_encoder_mode: Option<String>,
    pub eager: bool,
    pub offload: bool,
}

/// Export encoder-variant overrides as `MOLD_*` env vars so the engine
/// factory's auto-select picks them up during construction.
#[cfg(any(feature = "cuda", feature = "metal", test))]
pub(crate) fn apply_local_engine_env_overrides(
    t5_variant_override: Option<&str>,
    qwen3_variant_override: Option<&str>,
    qwen2_variant_override: Option<&str>,
    qwen2_text_encoder_mode_override: Option<&str>,
) {
    if let Some(variant) = t5_variant_override {
        std::env::set_var("MOLD_T5_VARIANT", variant);
    }
    if let Some(variant) = qwen3_variant_override {
        std::env::set_var("MOLD_QWEN3_VARIANT", variant);
    }
    if let Some(variant) = qwen2_variant_override {
        std::env::set_var("MOLD_QWEN2_VARIANT", variant);
    }
    if let Some(mode) = qwen2_text_encoder_mode_override {
        std::env::set_var("MOLD_QWEN2_TEXT_ENCODER_MODE", mode);
    }
}

/// Resolve model paths, auto-pulling when the model is missing locally or
/// has missing assets (repair pull). Returns the paths, the effective
/// config (post-pull when a pull happened), and whether a pull ran — a
/// `true` tells callers to re-derive request defaults from the refreshed
/// model config.
#[cfg(any(feature = "cuda", feature = "metal"))]
pub(crate) async fn resolve_or_pull_model(
    model: &str,
    config: &Config,
) -> Result<(ModelPaths, Config, bool)> {
    use mold_core::manifest::find_manifest;

    if config.manifest_model_needs_download(model) {
        status!(
            "{} Model '{}' is missing local assets, pulling repair...",
            theme::icon_info(),
            model.bold(),
        );
        let updated =
            super::pull::pull_and_configure(model, &mold_core::download::PullOptions::default())
                .await?;
        let paths = ModelPaths::resolve(model, &updated).ok_or_else(|| {
            anyhow::anyhow!("model '{model}' was pulled but paths could not be resolved")
        })?;
        Ok((paths, updated, true))
    } else if let Some(paths) = ModelPaths::resolve(model, config) {
        Ok((paths, config.clone(), false))
    } else if find_manifest(model).is_some() {
        status!(
            "{} Model '{}' not found locally, pulling...",
            theme::icon_info(),
            model.bold(),
        );
        let updated =
            super::pull::pull_and_configure(model, &mold_core::download::PullOptions::default())
                .await?;
        let paths = ModelPaths::resolve(model, &updated).ok_or_else(|| {
            anyhow::anyhow!("model '{model}' was pulled but paths could not be resolved")
        })?;
        Ok((paths, updated, true))
    } else {
        anyhow::bail!(
            "no model paths configured for '{model}'. Add [models.{model}] to \
             ~/.mold/config.toml, pull via `mold pull {model}`, or set \
             MOLD_TRANSFORMER_PATH / MOLD_VAE_PATH / MOLD_T5_PATH / MOLD_CLIP_PATH \
             / MOLD_T5_TOKENIZER_PATH / MOLD_CLIP_TOKENIZER_PATH env vars.",
        );
    }
}

/// Apply env overrides, resolve eager/offload, select the best GPU from
/// the allowed set (most free VRAM), and construct the engine.
#[cfg(any(feature = "cuda", feature = "metal"))]
pub(crate) fn build_local_engine(
    model: &str,
    paths: ModelPaths,
    config: &Config,
    ov: &EngineOverrides,
) -> Result<Box<dyn mold_inference::InferenceEngine>> {
    use mold_inference::LoadStrategy;

    apply_local_engine_env_overrides(
        ov.t5_variant.as_deref(),
        ov.qwen3_variant.as_deref(),
        ov.qwen2_variant.as_deref(),
        ov.qwen2_text_encoder_mode.as_deref(),
    );

    let is_eager = ov.eager || std::env::var("MOLD_EAGER").is_ok_and(|v| v == "1");
    let load_strategy = if is_eager {
        LoadStrategy::Eager
    } else {
        LoadStrategy::Sequential
    };
    if is_eager {
        std::env::set_var("MOLD_EAGER", "1");
    }
    let is_offload = ov.offload || std::env::var("MOLD_OFFLOAD").is_ok_and(|v| v == "1");

    let gpu_selection = match &ov.gpus {
        Some(s) => mold_core::types::GpuSelection::parse(s)?,
        None => config.gpu_selection(),
    };
    let discovered = mold_inference::device::discover_gpus();
    let available = mold_inference::device::filter_gpus(&discovered, &gpu_selection);
    let gpu_ordinal = mold_inference::device::select_best_gpu(&available)
        .map(|g| g.ordinal)
        .unwrap_or(0);

    Ok(mold_inference::create_engine(
        model.to_string(),
        paths,
        config,
        load_strategy,
        gpu_ordinal,
        is_offload,
    )?)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_support::ENV_LOCK;

    #[test]
    fn apply_local_engine_env_overrides_sets_qwen2_overrides() {
        let _lock = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        let prior_variant = std::env::var("MOLD_QWEN2_VARIANT").ok();
        let prior_mode = std::env::var("MOLD_QWEN2_TEXT_ENCODER_MODE").ok();

        std::env::remove_var("MOLD_QWEN2_VARIANT");
        std::env::remove_var("MOLD_QWEN2_TEXT_ENCODER_MODE");

        apply_local_engine_env_overrides(None, None, Some("q6"), Some("cpu-stage"));

        assert_eq!(
            std::env::var("MOLD_QWEN2_VARIANT").ok().as_deref(),
            Some("q6")
        );
        assert_eq!(
            std::env::var("MOLD_QWEN2_TEXT_ENCODER_MODE")
                .ok()
                .as_deref(),
            Some("cpu-stage")
        );

        match prior_variant {
            Some(value) => std::env::set_var("MOLD_QWEN2_VARIANT", value),
            None => std::env::remove_var("MOLD_QWEN2_VARIANT"),
        }
        match prior_mode {
            Some(value) => std::env::set_var("MOLD_QWEN2_TEXT_ENCODER_MODE", value),
            None => std::env::remove_var("MOLD_QWEN2_TEXT_ENCODER_MODE"),
        }
    }
}
