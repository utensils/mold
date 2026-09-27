//! `/transparent` — render a subject on a transparent background.
//!
//! ## Why this is its own command
//!
//! `/generate` sits at Discord's 25-option cap (pinned by
//! `generate_stays_within_discords_option_limit`), so a transparency toggle
//! and a container choice cannot be added there. The transparent render is
//! also a narrow shape: a still image, PNG or WebP (the only containers that
//! carry alpha), optionally edited from up to three ordered references — none
//! of `/generate`'s video, keyframe or retake options apply to it.
//!
//! Everything semantic is `mold_core`'s. The capability is the model's
//! `capabilities.transparency` block (`transparency_for_recipe` while the
//! model cache is cold), the refusals are `validate_transparency_choice`'s
//! own sentences, and references route through the same capability rule
//! `/generate` uses. The RGBA prompt recipe is applied by the engine, never
//! here: the prompt the user typed is the prompt that is recorded.

use crate::checks::{self, AuthResult};
use crate::commands::generate::{
    build_generate_request, defaults_from_manifest, fetch_reference_image, last_reference_canvas,
    route_references, validate_edit_reference_request, BuildParams, ReferenceRoute,
};
use crate::handler;
use crate::state::Context;
use anyhow::Result;
use mold_core::{ModelInfoExtended, OutputFormat, TransparencyCapabilitiesProfile};
use poise::serenity_prelude as serenity;

/// The containers `/transparent` offers — the two that carry an alpha
/// channel. JPEG is deliberately absent rather than offered and refused.
#[derive(Debug, Clone, Copy, PartialEq, Eq, poise::ChoiceParameter)]
pub enum TransparentFormat {
    #[name = "PNG (default)"]
    Png,
    #[name = "WebP"]
    Webp,
}

impl TransparentFormat {
    pub fn to_output_format(self) -> OutputFormat {
        match self {
            TransparentFormat::Png => OutputFormat::Png,
            TransparentFormat::Webp => OutputFormat::Webp,
        }
    }
}

/// The transparency contract for one model, as the bot can best know it.
///
/// A model the server lists answers with ITS advertised block; one whose
/// profile carries no block is an older server, which cannot render the
/// toggle, so the answer is `None` (hidden). While the cache is cold the
/// shared `mold-core` decision for the model's manifest family answers.
pub fn transparency_contract(
    entry: Option<&ModelInfoExtended>,
    family: Option<&str>,
    model: &str,
) -> Option<TransparencyCapabilitiesProfile> {
    match entry {
        Some(entry) => entry
            .generation_profile
            .as_ref()
            .and_then(|profile| profile.default_recipe())
            .and_then(|recipe| recipe.capabilities.transparency.clone()),
        None => family.map(|family| mold_core::transparency_for_recipe(family, model)),
    }
}

fn is_capable(contract: Option<&TransparencyCapabilitiesProfile>) -> bool {
    contract.is_some_and(|contract| contract.mode == mold_core::ControlMode::Adjustable)
}

fn entry_is_capable(entry: &ModelInfoExtended) -> bool {
    entry.runtime_available != Some(false)
        && is_capable(
            transparency_contract(Some(entry), Some(&entry.info.family), &entry.info.name).as_ref(),
        )
}

/// Built-in manifests whose recipe renders transparent backgrounds — the
/// cold-cache answer for autocomplete and the default model.
fn manifest_transparent_names() -> Vec<String> {
    mold_core::manifest::visible_manifests()
        .filter(|manifest| {
            is_capable(Some(&mold_core::transparency_for_recipe(
                &manifest.family,
                &manifest.name,
            )))
        })
        .map(|manifest| manifest.name.clone())
        .collect()
}

/// Rank transparency-capable models for autocomplete: downloaded first, then
/// the rest; the built-in manifests while the cache is cold.
pub fn rank_transparent_suggestions(cached: &[ModelInfoExtended], partial: &str) -> Vec<String> {
    let lower = partial.to_lowercase();
    let matches = |name: &str| lower.is_empty() || name.to_lowercase().contains(&lower);
    if cached.is_empty() {
        return manifest_transparent_names()
            .into_iter()
            .filter(|name| matches(name))
            .take(25)
            .collect();
    }
    let capable = cached
        .iter()
        .filter(|entry| entry_is_capable(entry) && matches(&entry.info.name));
    capable
        .clone()
        .filter(|entry| entry.downloaded)
        .chain(capable.filter(|entry| !entry.downloaded))
        .take(25)
        .map(|entry| entry.info.name.clone())
        .collect()
}

/// Default model when the user names none: the first downloaded
/// transparency-capable model, else the first capable one, else (cold cache)
/// the first built-in manifest that is. `None` is an answer the user must
/// see: a toggle on a model that cannot honour it would only be refused.
pub fn resolve_transparent_model(models: &[ModelInfoExtended]) -> Option<String> {
    if models.is_empty() {
        return manifest_transparent_names().into_iter().next();
    }
    let capable = || models.iter().filter(|entry| entry_is_capable(entry));
    capable()
        .find(|entry| entry.downloaded)
        .or_else(|| capable().next())
        .map(|entry| entry.info.name.clone())
}

/// Refusal shown when no listed model renders transparent backgrounds.
pub const NO_TRANSPARENT_MODEL: &str =
    "This server advertises no model that renders transparent backgrounds. Qwen Image 2.1 \
     does, on a server new enough to advertise capabilities.transparency.";

/// Refuse a model/container pairing the recipe cannot render, in
/// admission's own words. An absent contract is an older server or a model
/// without the capability.
pub fn transparent_gate(
    contract: Option<&TransparencyCapabilitiesProfile>,
    model: &str,
    format: OutputFormat,
) -> Result<(), String> {
    match contract {
        Some(contract) => mold_core::validate_transparency_choice(contract, Some(true), format),
        None => Err(format!(
            "{} '{model}' advertises no transparency contract on this server; pick a \
             transparency-capable model or update the server.",
            mold_core::TRANSPARENCY_UNSUPPORTED_REASON
        )),
    }
}

/// The first gap in an ordered reference list (`reference_2` without
/// `reference_1`, ...). References are named by position in the prompt, so a
/// hole would silently renumber them.
pub fn reference_order_gap(present: &[bool]) -> Option<String> {
    present.windows(2).enumerate().find_map(|(index, pair)| {
        (!pair[0] && pair[1]).then(|| {
            format!(
                "Add reference_{} before reference_{} so the reference order is unambiguous.",
                index + 1,
                index + 2
            )
        })
    })
}

async fn autocomplete_transparent_model(ctx: Context<'_>, partial: &str) -> Vec<String> {
    let cached = ctx.data().cached_models().await;
    rank_transparent_suggestions(&cached, partial)
}

/// Render a subject on a transparent background (PNG or WebP with alpha).
#[allow(clippy::too_many_arguments)]
#[poise::command(slash_command)]
pub async fn transparent(
    ctx: Context<'_>,
    #[description = "The subject to cut out — describe it alone, with no scenery"] prompt: String,
    #[description = "Transparency-capable model (defaults to one this server advertises)"]
    #[autocomplete = "autocomplete_transparent_model"]
    model: Option<String>,
    #[description = "Container with an alpha channel (PNG default, or WebP)"] format: Option<
        TransparentFormat,
    >,
    #[description = "Ordered reference image 1 (e.g. a picture to cut the subject out of)"]
    reference_1: Option<serenity::Attachment>,
    #[description = "Ordered reference image 2"] reference_2: Option<serenity::Attachment>,
    #[description = "Ordered reference image 3"] reference_3: Option<serenity::Attachment>,
    #[description = "Random seed for reproducibility"] seed: Option<u64>,
    #[description = "Image width in pixels (default: the last reference's aspect, else the model's)"]
    width: Option<u32>,
    #[description = "Image height in pixels"] height: Option<u32>,
    #[description = "Number of inference steps"] steps: Option<u32>,
) -> Result<()> {
    let refuse = |message: String| async move {
        ctx.send(
            poise::CreateReply::default()
                .content(message)
                .ephemeral(true),
        )
        .await
        .map(|_| ())
    };
    if prompt.trim().is_empty() {
        return Ok(refuse("Prompt cannot be empty.".to_string()).await?);
    }
    let slots = [
        reference_1.as_ref(),
        reference_2.as_ref(),
        reference_3.as_ref(),
    ];
    if let Some(message) = reference_order_gap(&slots.map(|slot| slot.is_some())) {
        return Ok(refuse(message).await?);
    }
    let references: Vec<&serenity::Attachment> = slots.into_iter().flatten().collect();

    // Everything below is knowable before deferring, so an impossible request
    // never costs a quota slot or a download.
    let models = ctx.data().cached_models().await;
    let Some(model_name) = model.or_else(|| resolve_transparent_model(&models)) else {
        return Ok(refuse(NO_TRANSPARENT_MODEL.to_string()).await?);
    };
    let model_entry = models.iter().find(|entry| entry.info.name == model_name);
    let fallback_manifest = model_entry
        .is_none()
        .then(|| mold_core::manifest::find_manifest(&model_name))
        .flatten();
    let fallback_defaults = fallback_manifest.map(defaults_from_manifest);
    let model_defaults = model_entry
        .map(|entry| &entry.defaults)
        .or(fallback_defaults.as_ref());
    let family = model_entry
        .map(|entry| entry.info.family.as_str())
        .or_else(|| fallback_manifest.map(|manifest| manifest.family.as_str()));
    let output_format = format.unwrap_or(TransparentFormat::Png).to_output_format();
    if let Err(message) = transparent_gate(
        transparency_contract(model_entry, family, &model_name).as_ref(),
        &model_name,
        output_format,
    ) {
        return Ok(refuse(message).await?);
    }
    let reference_profile = if references.is_empty() {
        None
    } else {
        match route_references(model_entry, family, &model_name) {
            ReferenceRoute::EditImages(profile) => {
                if let Err(message) = validate_edit_reference_request(
                    &profile,
                    family,
                    &model_name,
                    references.len(),
                    false,
                ) {
                    return Ok(refuse(message).await?);
                }
                Some(profile)
            }
            ReferenceRoute::Refused(message) => return Ok(refuse(message).await?),
            ReferenceRoute::H3Ref2va => {
                return Ok(
                    refuse(mold_core::REFERENCE_IMAGES_UNSUPPORTED_REASON.to_string()).await?,
                )
            }
        }
    };

    let user_id = ctx.author().id.get();
    if let AuthResult::Denied(msg) = checks::check_generate_auth(&ctx).await {
        return Ok(refuse(msg).await?);
    }
    ctx.defer().await?;

    let mut edit_images = Vec::with_capacity(references.len());
    if let Some(profile) = reference_profile.as_ref() {
        for (index, attachment) in references.iter().enumerate() {
            match fetch_reference_image(attachment, index + 1, profile).await {
                Ok(bytes) => edit_images.push(bytes),
                Err(message) => {
                    ctx.data().quotas.refund(user_id);
                    handler::send_error(ctx, &message).await?;
                    return Ok(());
                }
            }
        }
    }
    let (width, height) = match (width, height, reference_profile.as_ref()) {
        (None, None, Some(profile)) => {
            last_reference_canvas(profile, &edit_images, model_defaults, family, &model_name)
                .map_or((None, None), |(w, h)| (Some(w), Some(h)))
        }
        (width, height, _) => (width, height),
    };

    let req = build_generate_request(BuildParams {
        prompt: &prompt,
        model: &model_name,
        family,
        width,
        height,
        steps,
        seed,
        defaults: model_defaults,
        edit_images: (!edit_images.is_empty()).then_some(edit_images),
        transparent_background: Some(true),
        still_format: Some(output_format),
        ..Default::default()
    });

    match handler::run_generation(ctx, req).await {
        Ok(()) => ctx.data().cooldowns.record(user_id),
        Err(error) => {
            ctx.data().quotas.refund(user_id);
            let message = if mold_core::MoldClient::is_connection_error(&error) {
                "Could not connect to the mold server. Is it running?".to_string()
            } else if mold_core::MoldClient::is_model_not_found(&error) {
                format!(
                    "Model '{model_name}' is not downloaded. Use `/models` to see available models."
                )
            } else {
                format!("Transparent generation failed: {error}")
            };
            handler::send_error(ctx, &message).await?;
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn model(name: &str, family: &str, downloaded: bool) -> ModelInfoExtended {
        ModelInfoExtended {
            runtime_available: None,
            runtime_unavailable_reason: None,
            info: mold_core::ModelInfo {
                name: name.to_string(),
                family: family.to_string(),
                size_gb: 1.0,
                is_loaded: false,
                last_used: None,
                hf_repo: "test/repo".to_string(),
            },
            defaults: mold_core::ModelDefaults {
                default_steps: 40,
                default_guidance: 4.0,
                default_width: 1024,
                default_height: 1024,
                dimension_alignment: Some(32),
                ..Default::default()
            },
            downloaded,
            disk_usage_bytes: None,
            remaining_download_bytes: None,
            display_name: None,
            kind: None,
            modality: None,
            nsfw: None,
            supports_audio: None,
            supports_extend: None,
            supports_sequence: None,
            extend_default_overlap_frames: None,
            guidance_capabilities: None,
            source_image: None,
            generation_profile: Some(mold_core::generation_profile_for_manifest(
                mold_core::manifest::find_manifest(name).expect("a built-in manifest"),
            )),
            supports_identity: None,
            supports_duration_prediction: None,
            runtime_ready: None,
            runtime_readiness_error: None,
        }
    }

    fn png(width: u32, height: u32) -> Vec<u8> {
        let mut bytes = std::io::Cursor::new(Vec::new());
        image::DynamicImage::ImageRgba8(image::RgbaImage::new(width, height))
            .write_to(&mut bytes, image::ImageFormat::Png)
            .unwrap();
        bytes.into_inner()
    }

    #[test]
    fn the_command_stays_within_discords_option_limit() {
        let command = transparent();
        assert!(command.parameters.len() <= 25);
        let names: Vec<&str> = command
            .parameters
            .iter()
            .map(|parameter| parameter.name.as_str())
            .collect();
        assert_eq!(
            names,
            [
                "prompt",
                "model",
                "format",
                "reference_1",
                "reference_2",
                "reference_3",
                "seed",
                "width",
                "height",
                "steps"
            ]
        );
    }

    #[test]
    fn only_alpha_containers_are_offered() {
        assert_eq!(TransparentFormat::Png.to_output_format(), OutputFormat::Png);
        assert_eq!(
            TransparentFormat::Webp.to_output_format(),
            OutputFormat::Webp
        );
    }

    /// The capability is the server's advertised block; a cold cache falls
    /// back to the shared core decision by manifest family.
    #[test]
    fn autocomplete_and_default_read_the_transparency_capability() {
        let models = vec![
            model("flux-dev:q8", "flux", true),
            model("qwen-image-2.1:q8", "qwen-image21", false),
            model("qwen-image-2.1-turbo:bf16", "qwen-image21", true),
            model("sdxl-base:fp16", "sdxl", true),
        ];
        assert_eq!(
            rank_transparent_suggestions(&models, ""),
            ["qwen-image-2.1-turbo:bf16", "qwen-image-2.1:q8"],
            "downloaded capable models first, and never a model without the contract"
        );
        assert_eq!(
            rank_transparent_suggestions(&models, "q8"),
            ["qwen-image-2.1:q8"]
        );
        assert_eq!(
            resolve_transparent_model(&models).as_deref(),
            Some("qwen-image-2.1-turbo:bf16")
        );
        assert_eq!(
            resolve_transparent_model(&[model("flux-dev:q8", "flux", true)]),
            None
        );

        // Cold cache: the built-in manifests that carry the contract.
        let cold = rank_transparent_suggestions(&[], "");
        assert!(!cold.is_empty());
        assert!(
            cold.iter().all(|name| name.starts_with("qwen-image-2.1")),
            "{cold:?}"
        );
        assert!(
            resolve_transparent_model(&[]).is_some_and(|name| name.starts_with("qwen-image-2.1"))
        );
    }

    #[test]
    fn an_older_server_without_the_block_is_hidden_not_guessed() {
        let mut old = model("qwen-image-2.1:bf16", "qwen-image21", true);
        for recipe in &mut old.generation_profile.as_mut().unwrap().recipes {
            recipe.capabilities.transparency = None;
        }
        assert_eq!(
            transparency_contract(Some(&old), Some("qwen-image21"), "qwen-image-2.1:bf16"),
            None
        );
        assert!(resolve_transparent_model(&[old]).is_none());
        // Cold cache: the family decides.
        assert!(is_capable(
            transparency_contract(None, Some("qwen-image21"), "qwen-image-2.1:bf16").as_ref()
        ));
    }

    #[test]
    fn the_gate_speaks_with_admissions_voice() {
        let qwen = transparency_contract(None, Some("qwen-image21"), "qwen-image-2.1:bf16");
        transparent_gate(qwen.as_ref(), "qwen-image-2.1:bf16", OutputFormat::Png).unwrap();
        transparent_gate(qwen.as_ref(), "qwen-image-2.1:bf16", OutputFormat::Webp).unwrap();
        let flux = transparency_contract(None, Some("flux"), "flux-dev:q8");
        assert_eq!(
            transparent_gate(flux.as_ref(), "flux-dev:q8", OutputFormat::Png).unwrap_err(),
            mold_core::TRANSPARENCY_UNSUPPORTED_REASON
        );
        assert!(
            transparent_gate(None, "qwen-image-2.1:bf16", OutputFormat::Png)
                .unwrap_err()
                .starts_with(mold_core::TRANSPARENCY_UNSUPPORTED_REASON)
        );
    }

    #[test]
    fn reference_gaps_name_the_missing_slot() {
        assert_eq!(reference_order_gap(&[true, true, true]), None);
        assert_eq!(reference_order_gap(&[true, false, false]), None);
        assert_eq!(reference_order_gap(&[false, false, false]), None);
        assert_eq!(
            reference_order_gap(&[false, true, false]).as_deref(),
            Some(crate::commands::generate::REFERENCE_ORDER_GAP)
        );
        assert_eq!(
            reference_order_gap(&[true, false, true]).as_deref(),
            Some("Add reference_2 before reference_3 so the reference order is unambiguous.")
        );
    }

    /// The request the command builds: the toggle, the chosen alpha
    /// container, ordered `edit_images`, and a last-reference canvas.
    #[test]
    fn the_request_carries_the_toggle_the_container_and_the_references() {
        let entry = model("qwen-image-2.1:bf16", "qwen-image21", true);
        let ReferenceRoute::EditImages(profile) =
            route_references(Some(&entry), Some("qwen-image21"), "qwen-image-2.1:bf16")
        else {
            panic!("Qwen Image 2.1 reads references as edit_images");
        };
        let references = vec![png(64, 64), png(1920, 1080)];
        let (width, height) = last_reference_canvas(
            &profile,
            &references,
            Some(&entry.defaults),
            Some("qwen-image21"),
            "qwen-image-2.1:bf16",
        )
        .unwrap();
        assert_eq!((width, height), (1376, 768));
        let req = build_generate_request(BuildParams {
            prompt: "a red paper lantern",
            model: "qwen-image-2.1:bf16",
            family: Some("qwen-image21"),
            width: Some(width),
            height: Some(height),
            defaults: Some(&entry.defaults),
            edit_images: Some(references.clone()),
            transparent_background: Some(true),
            still_format: Some(OutputFormat::Webp),
            ..Default::default()
        });
        assert_eq!(req.transparent_background, Some(true));
        assert_eq!(req.output_format, Some(OutputFormat::Webp));
        assert_eq!(req.edit_images, Some(references));
        assert_eq!(
            req.prompt, "a red paper lantern",
            "the engine wraps the prompt, never the bot"
        );
        mold_core::validate_transparency_against(
            &mold_core::transparency_for_recipe("qwen-image21", "qwen-image-2.1:bf16"),
            &req,
        )
        .unwrap();
    }
}
