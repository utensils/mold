//! `/identity` — generate an image conditioned on a face reference photo
//! (PuLID).
//!
//! ## Why this is not an option on `/generate`
//!
//! Discord caps a chat-input command at **25 options** and `/generate` is
//! already at exactly 25 (`generate_stays_within_discords_option_limit` pins
//! it). Three identity options cannot be added there without deleting three
//! existing controls, so identity gets its own command — which is the better
//! shape anyway: `mold_core::identity` qualifies identity conditioning only
//! for the two FLUX dev tiers, so none of `/generate`'s fifteen
//! video/conditioning options apply to it.
//!
//! Everything semantic is `mold_core::identity`'s: the accepted containers
//! and the encoded ceiling (`ID_IMAGE_LIMITS`, `validate_id_image_bytes`),
//! the `id_weight` range, the `id_start_step` rule, and the model-gate
//! refusal. The bot re-states none of them. The capability gate is the
//! server's advertised `/api/models[].supports_identity` — never the local
//! build, which for an HTTP-only bot says nothing about the renderer.

use crate::checks::{self, AuthResult};
use crate::commands::generate::{build_generate_request, BuildParams};
use crate::handler;
use crate::state::Context;
use anyhow::Result;
use mold_core::{identity, ModelInfoExtended};
use poise::serenity_prelude as serenity;

/// Everything about the identity request that is known before the attachment
/// bytes are fetched. Split out so the whole precondition set is testable
/// without a live interaction or a network round trip.
#[derive(Debug, Clone, Default)]
pub struct IdentityOptions<'a> {
    /// Attachment size as Discord reports it, in bytes.
    pub attachment_bytes: u64,
    /// Attachment MIME type when Discord supplied one.
    pub content_type: Option<&'a str>,
    pub strength: Option<f64>,
    pub start_step: Option<u32>,
}

/// Refuse an identity request on what is knowable before the download.
///
/// Checked in cost order: the declared size first (so an oversized upload is
/// never fetched), then the container, then the two knobs. The size ceiling
/// and the ranges are `mold_core::identity`'s, so this bot and the server
/// draw the same line.
pub fn validate_identity_options(options: &IdentityOptions<'_>) -> Result<(), String> {
    let limit = identity::ID_IMAGE_LIMITS.max_encoded_bytes as u64;
    if options.attachment_bytes > limit {
        return Err(format!(
            "Identity photo is too large ({:.1} MiB). Keep it under {} MiB.",
            options.attachment_bytes as f64 / (1024.0 * 1024.0),
            limit / (1024 * 1024)
        ));
    }
    if let Some(content_type) = options.content_type {
        if !(content_type.starts_with("image/png") || content_type.starts_with("image/jpeg")) {
            return Err(format!(
                "Identity photo must be PNG or JPEG, got `{content_type}`."
            ));
        }
    }
    if let Some(strength) = options.strength {
        identity::validate_id_weight(strength)?;
    }
    Ok(())
}

/// The model gate: this concrete checkpoint, on this concrete server, must
/// advertise `supports_identity`.
///
/// An absent field is "no". That covers both a server too old to know about
/// identity conditioning and one whose binary cannot execute it — in either
/// case the request would be refused after taking a queue slot, and the
/// refusal is more useful here.
pub fn identity_model_gate(entry: Option<&ModelInfoExtended>, model: &str) -> Option<String> {
    if entry.and_then(|entry| entry.supports_identity) == Some(true) {
        return None;
    }
    Some(format!(
        "{} Pick a model this server advertises as identity-capable, or update the server.",
        identity::identity_model_gate_message(model)
    ))
}

/// Refuse a start step the resolved step count cannot honour. Deferred until
/// the model's advertised defaults are known, because an omitted `steps`
/// takes the checkpoint's own default rather than a constant.
pub fn validate_identity_start_step(start_step: Option<u32>, steps: u32) -> Result<(), String> {
    match start_step {
        Some(value) => identity::validate_id_start_step(value, steps),
        None => Ok(()),
    }
}

/// Refuse fetched bytes that are not a bounded PNG or JPEG. Header-only —
/// nothing decodes the payload here or on the server.
pub fn validate_identity_bytes(bytes: &[u8]) -> Result<(), String> {
    identity::validate_id_image_bytes(bytes)
}

/// Default model for `/identity` when the user names none: the first
/// downloaded checkpoint this server advertises as identity-capable, else the
/// first advertised one at all.
///
/// There is deliberately no hard-coded fallback. `/generate` can fall back to
/// `flux2-klein:q8` because any checkpoint renders something; an identity
/// request against a checkpoint that cannot take the photo would be refused,
/// so "no candidate" is an answer the user needs to see.
pub fn resolve_identity_model(models: &[ModelInfoExtended]) -> Option<String> {
    let capable = || {
        models
            .iter()
            .filter(|entry| entry.supports_identity == Some(true))
    };
    capable()
        .find(|entry| entry.downloaded)
        .or_else(|| capable().next())
        .map(|entry| entry.info.name.clone())
}

/// Refusal shown when the fleet advertises no identity-capable checkpoint.
pub const NO_IDENTITY_MODEL: &str =
    "This server advertises no identity-capable model. Face-identity conditioning is qualified \
     only for the FLUX dev tiers and a few SDXL checkpoints, on a server built with PuLID \
     support.";

/// Display-safe attachment basename for `id_image_name`. Discord filenames
/// are user-controlled, so the same sanitizer the H3 reference path uses
/// applies here — the label is recorded into saved metadata.
fn identity_image_name(attachment: &serenity::Attachment) -> Option<String> {
    crate::h3_references::safe_name(&attachment.filename)
}

/// Download the identity photo, bounds-checking before and after the fetch.
async fn fetch_identity_image(attachment: &serenity::Attachment) -> Result<Vec<u8>, String> {
    let bytes = attachment
        .download()
        .await
        .map_err(|error| format!("Failed to download identity photo: {error}"))?;
    validate_identity_bytes(&bytes)?;
    Ok(bytes)
}

async fn autocomplete_identity_model(ctx: Context<'_>, partial: &str) -> Vec<String> {
    let partial = partial.to_lowercase();
    ctx.data()
        .cached_models()
        .await
        .into_iter()
        .filter(|entry| entry.supports_identity == Some(true))
        .filter(|entry| partial.is_empty() || entry.info.name.to_lowercase().contains(&partial))
        .take(25)
        .map(|entry| entry.info.name)
        .collect()
}

/// Generate an image conditioned on a face reference photo (PuLID).
#[allow(clippy::too_many_arguments)]
#[poise::command(slash_command)]
pub async fn identity(
    ctx: Context<'_>,
    #[description = "Text prompt describing the image to render"] prompt: String,
    #[description = "Face reference photo (PNG or JPEG)"] identity: serenity::Attachment,
    #[description = "Identity-capable model (defaults to one this server advertises)"]
    #[autocomplete = "autocomplete_identity_model"]
    model: Option<String>,
    #[description = "Identity strength, 0.0-3.0 (default 1.0)"] identity_strength: Option<f64>,
    #[description = "First denoise step the face is applied at (default 0; must be under steps)"]
    identity_start_step: Option<u32>,
    #[description = "Image width in pixels"] width: Option<u32>,
    #[description = "Image height in pixels"] height: Option<u32>,
    #[description = "Number of inference steps"] steps: Option<u32>,
    #[description = "Guidance scale"] guidance: Option<f64>,
    #[description = "Random seed for reproducibility"] seed: Option<u64>,
) -> Result<()> {
    if prompt.trim().is_empty() {
        ctx.send(
            poise::CreateReply::default()
                .content("Prompt cannot be empty.")
                .ephemeral(true),
        )
        .await?;
        return Ok(());
    }

    // Everything knowable before the interaction is deferred, so an obviously
    // impossible request never costs a quota slot or a download.
    if let Err(message) = validate_identity_options(&IdentityOptions {
        attachment_bytes: identity.size as u64,
        content_type: identity.content_type.as_deref(),
        strength: identity_strength,
        start_step: identity_start_step,
    }) {
        ctx.send(
            poise::CreateReply::default()
                .content(message)
                .ephemeral(true),
        )
        .await?;
        return Ok(());
    }

    let user_id = ctx.author().id.get();
    if let AuthResult::Denied(msg) = checks::check_generate_auth(&ctx).await {
        ctx.send(poise::CreateReply::default().content(msg).ephemeral(true))
            .await?;
        return Ok(());
    }

    ctx.defer().await?;

    let models = ctx.data().cached_models().await;
    let Some(model_name) = model.or_else(|| resolve_identity_model(&models)) else {
        ctx.data().quotas.refund(user_id);
        handler::send_error(ctx, NO_IDENTITY_MODEL).await?;
        return Ok(());
    };
    let model_entry = models.iter().find(|entry| entry.info.name == model_name);

    if let Some(message) = identity_model_gate(model_entry, &model_name) {
        ctx.data().quotas.refund(user_id);
        handler::send_error(ctx, &message).await?;
        return Ok(());
    }

    let model_defaults = model_entry.map(|entry| &entry.defaults);
    let resolved_steps = steps.unwrap_or_else(|| {
        model_defaults
            .map(|defaults| defaults.default_steps)
            .unwrap_or(20)
    });
    if let Err(message) = validate_identity_start_step(identity_start_step, resolved_steps) {
        ctx.data().quotas.refund(user_id);
        handler::send_error(ctx, &message).await?;
        return Ok(());
    }

    let bytes = match fetch_identity_image(&identity).await {
        Ok(bytes) => bytes,
        Err(message) => {
            ctx.data().quotas.refund(user_id);
            handler::send_error(ctx, &message).await?;
            return Ok(());
        }
    };

    let req = build_generate_request(BuildParams {
        prompt: &prompt,
        model: &model_name,
        family: model_entry.map(|entry| entry.info.family.as_str()),
        width,
        height,
        steps: Some(resolved_steps),
        guidance,
        seed,
        defaults: model_defaults,
        id_image: Some(bytes),
        id_image_name: identity_image_name(&identity),
        id_weight: identity_strength,
        id_start_step: identity_start_step,
        ..Default::default()
    });

    match handler::run_generation(ctx, req).await {
        Ok(()) => {
            ctx.data().cooldowns.record(user_id);
        }
        Err(error) => {
            ctx.data().quotas.refund(user_id);
            let message = if mold_core::MoldClient::is_connection_error(&error) {
                "Could not connect to the mold server. Is it running?".to_string()
            } else if mold_core::MoldClient::is_model_not_found(&error) {
                format!(
                    "Model '{model_name}' is not downloaded. Use `/models` to see available models."
                )
            } else {
                format!("Identity generation failed: {error}")
            };
            handler::send_error(ctx, &message).await?;
        }
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A genuine 1x1 RGBA PNG — the smallest payload
    /// `identity::validate_id_image_bytes` accepts.
    const PNG_1X1: [u8; 67] = [
        0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A, 0x00, 0x00, 0x00, 0x0D, 0x49, 0x48, 0x44,
        0x52, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0x01, 0x08, 0x06, 0x00, 0x00, 0x00, 0x1F,
        0x15, 0xC4, 0x89, 0x00, 0x00, 0x00, 0x0A, 0x49, 0x44, 0x41, 0x54, 0x78, 0x9C, 0x63, 0x00,
        0x01, 0x00, 0x00, 0x05, 0x00, 0x01, 0x0D, 0x0A, 0x2D, 0xB4, 0x00, 0x00, 0x00, 0x00, 0x49,
        0x45, 0x4E, 0x44, 0xAE, 0x42, 0x60, 0x82,
    ];

    fn model(name: &str, supports_identity: Option<bool>, downloaded: bool) -> ModelInfoExtended {
        ModelInfoExtended {
            runtime_available: None,
            runtime_unavailable_reason: None,
            info: mold_core::ModelInfo {
                name: name.to_string(),
                family: "flux".to_string(),
                size_gb: 1.0,
                is_loaded: false,
                last_used: None,
                hf_repo: "test/repo".to_string(),
            },
            defaults: mold_core::ModelDefaults {
                default_steps: 20,
                default_guidance: 3.5,
                default_width: 1024,
                default_height: 1024,
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
            generation_profile: None,
            supports_identity,
        }
    }

    #[test]
    fn oversized_and_wrong_container_uploads_are_refused_before_the_download() {
        let over = identity::ID_IMAGE_LIMITS.max_encoded_bytes as u64 + 1;
        let error = validate_identity_options(&IdentityOptions {
            attachment_bytes: over,
            ..Default::default()
        })
        .unwrap_err();
        assert!(error.contains("16 MiB"), "{error}");

        // Exactly at the ceiling is allowed — the byte check is the authority.
        assert!(validate_identity_options(&IdentityOptions {
            attachment_bytes: identity::ID_IMAGE_LIMITS.max_encoded_bytes as u64,
            ..Default::default()
        })
        .is_ok());

        let error = validate_identity_options(&IdentityOptions {
            content_type: Some("image/webp"),
            ..Default::default()
        })
        .unwrap_err();
        assert!(error.contains("PNG or JPEG"), "{error}");

        for content_type in ["image/png", "image/jpeg", "image/png; charset=binary"] {
            assert!(
                validate_identity_options(&IdentityOptions {
                    content_type: Some(content_type),
                    ..Default::default()
                })
                .is_ok(),
                "{content_type} must be accepted"
            );
        }
        // Discord may omit the type entirely; the byte check still runs.
        assert!(validate_identity_options(&IdentityOptions::default()).is_ok());
    }

    #[test]
    fn the_strength_range_is_mold_cores() {
        for strength in [0.0, 1.0, identity::ID_WEIGHT_MAX] {
            assert!(validate_identity_options(&IdentityOptions {
                strength: Some(strength),
                ..Default::default()
            })
            .is_ok());
        }
        for strength in [-0.1, identity::ID_WEIGHT_MAX + 0.1, f64::NAN] {
            let error = validate_identity_options(&IdentityOptions {
                strength: Some(strength),
                ..Default::default()
            })
            .unwrap_err();
            assert_eq!(
                error,
                identity::validate_id_weight(strength).unwrap_err(),
                "the refusal must be mold-core's own sentence"
            );
        }
    }

    /// The start step is checked against the *resolved* step count, so an
    /// omitted `steps` is judged against the checkpoint's own default rather
    /// than a constant.
    #[test]
    fn the_start_step_is_checked_against_the_resolved_step_count() {
        assert!(validate_identity_start_step(None, 4).is_ok());
        assert!(validate_identity_start_step(Some(0), 1).is_ok());
        assert!(validate_identity_start_step(Some(3), 4).is_ok());
        assert_eq!(
            validate_identity_start_step(Some(4), 4).unwrap_err(),
            identity::validate_id_start_step(4, 4).unwrap_err()
        );
    }

    #[test]
    fn the_model_gate_reads_only_the_servers_advertisement() {
        let capable = model("flux-dev:q8", Some(true), true);
        assert_eq!(identity_model_gate(Some(&capable), "flux-dev:q8"), None);

        // An unadvertised, an explicitly false, and an unknown model are all
        // refused with mold-core's sentence plus what to do about it.
        for entry in [
            Some(model("flux-dev:q8", None, true)),
            Some(model("flux-dev:q8", Some(false), true)),
            None,
        ] {
            let message = identity_model_gate(entry.as_ref(), "flux-dev:q8")
                .expect("an unadvertised checkpoint must be refused");
            assert!(
                message.starts_with(&identity::identity_model_gate_message("flux-dev:q8")),
                "{message}"
            );
        }
    }

    #[test]
    fn the_default_model_prefers_a_downloaded_identity_capable_checkpoint() {
        assert_eq!(resolve_identity_model(&[]), None);
        assert_eq!(
            resolve_identity_model(&[model("flux2-klein:q8", None, true)]),
            None,
            "a model that does not advertise identity is never the default"
        );
        assert_eq!(
            resolve_identity_model(&[
                model("flux2-klein:q8", None, true),
                model("flux-dev:q4", Some(true), false),
                model("flux-dev:q8", Some(true), true),
            ])
            .as_deref(),
            Some("flux-dev:q8"),
            "a downloaded capable model wins over an undownloaded one"
        );
        assert_eq!(
            resolve_identity_model(&[model("flux-dev:q4", Some(true), false)]).as_deref(),
            Some("flux-dev:q4"),
            "an undownloaded capable model still beats no answer — the server auto-pulls"
        );
    }

    #[test]
    fn fetched_bytes_are_bounds_checked_by_mold_core() {
        assert!(validate_identity_bytes(&PNG_1X1).is_ok());
        assert_eq!(
            validate_identity_bytes(b"not an image").unwrap_err(),
            identity::validate_id_image_bytes(b"not an image").unwrap_err()
        );
        assert!(validate_identity_bytes(&[]).is_err());
    }

    /// The four identity fields ship together, with `mold_core`'s defaults
    /// materialized so the saved provenance records what actually rendered.
    #[test]
    fn the_request_carries_the_whole_identity_group() {
        let req = build_generate_request(BuildParams {
            prompt: "a portrait",
            model: "flux-dev:q8",
            family: Some("flux"),
            steps: Some(20),
            id_image: Some(PNG_1X1.to_vec()),
            id_image_name: Some("ada.png".into()),
            id_weight: Some(1.4),
            id_start_step: Some(3),
            ..Default::default()
        });
        assert_eq!(req.id_image.as_deref(), Some(&PNG_1X1[..]));
        assert_eq!(req.id_image_name.as_deref(), Some("ada.png"));
        assert_eq!(req.id_weight, Some(1.4));
        assert_eq!(req.id_start_step, Some(3));

        // Omitted knobs materialize mold-core's published defaults.
        let req = build_generate_request(BuildParams {
            prompt: "a portrait",
            model: "flux-dev:q8",
            family: Some("flux"),
            steps: Some(20),
            id_image: Some(PNG_1X1.to_vec()),
            id_image_name: Some("ada.png".into()),
            ..Default::default()
        });
        assert_eq!(req.id_weight, Some(identity::ID_WEIGHT_DEFAULT));
        assert_eq!(req.id_start_step, Some(identity::ID_START_STEP_DEFAULT));
        assert_eq!(
            identity::effective_id_weight(&req),
            identity::ID_WEIGHT_DEFAULT
        );
    }

    /// A knob without a photo is the incomplete form the server refuses, so
    /// the builder must never produce it — including for `/generate`, which
    /// has no identity options at all.
    #[test]
    fn knobs_without_a_photo_never_reach_the_wire() {
        let req = build_generate_request(BuildParams {
            prompt: "a portrait",
            model: "flux-dev:q8",
            family: Some("flux"),
            id_weight: Some(2.0),
            id_start_step: Some(1),
            id_image_name: Some("ada.png".into()),
            ..Default::default()
        });
        assert!(!identity::request_mentions_identity(&req));

        let ordinary = build_generate_request(BuildParams {
            prompt: "a cat",
            model: "flux2-klein:q8",
            family: Some("flux2"),
            ..Default::default()
        });
        assert!(!identity::request_mentions_identity(&ordinary));
    }

    #[test]
    fn the_command_stays_well_under_discords_option_limit() {
        let command = identity();
        assert!(
            command.parameters.len() <= 25,
            "identity has {} options",
            command.parameters.len()
        );
        for name in [
            "prompt",
            "identity",
            "identity_strength",
            "identity_start_step",
        ] {
            assert!(
                command
                    .parameters
                    .iter()
                    .any(|parameter| parameter.name == name),
                "missing option {name}"
            );
        }
        // `/generate` is at the hard cap, which is exactly why this command
        // exists; a regression there would silently drop options at
        // registration time.
        assert_eq!(
            crate::commands::generate::generate().parameters.len(),
            25,
            "identity lives here because /generate has no room left"
        );
    }
}
