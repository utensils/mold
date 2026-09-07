use crate::checks::{self, AuthResult};
use crate::commands::generate::{self, BuildParams};
use crate::handler;
use crate::state::Context;
use anyhow::Result;
use image::{ImageDecoder, ImageReader};
use mold_core::{
    GenerationImageReferenceRole, GenerationReference, GenerationReferenceAuthority,
    GenerationReferenceProvenance, MeshRequestOptions,
};
use poise::serenity_prelude as serenity;
use std::io::Cursor;

/// Generate a Hunyuan3D mesh from one image or semantic front/left/back/right views.
#[allow(clippy::too_many_arguments)]
#[poise::command(slash_command)]
pub async fn mesh(
    ctx: Context<'_>,
    #[description = "Hunyuan3D shape model"]
    #[autocomplete = "autocomplete_mesh_model"]
    model: Option<String>,
    #[description = "Single-view source image (PNG or JPEG)"] source: Option<serenity::Attachment>,
    #[description = "Front view for a 2mv model"] front: Option<serenity::Attachment>,
    #[description = "Left view for a 2mv model"] left: Option<serenity::Attachment>,
    #[description = "Back view for a 2mv model"] back: Option<serenity::Attachment>,
    #[description = "Right view for a 2mv model"] right: Option<serenity::Attachment>,
    #[description = "Random seed for reproducibility"] seed: Option<u64>,
    #[description = "Run the PBR paint stage"] texture: Option<bool>,
    #[description = "Shape query-grid resolution"] octree: Option<u32>,
    #[description = "Surface iso threshold"] threshold: Option<f64>,
    #[description = "Approximate triangle target"] target_faces: Option<u32>,
) -> Result<()> {
    let named = [
        (GenerationImageReferenceRole::Front, front.as_ref()),
        (GenerationImageReferenceRole::Left, left.as_ref()),
        (GenerationImageReferenceRole::Back, back.as_ref()),
        (GenerationImageReferenceRole::Right, right.as_ref()),
    ];
    let named_count = named.iter().filter(|(_, item)| item.is_some()).count();
    if source.is_some() == (named_count > 0) {
        ctx.send(
            poise::CreateReply::default()
                .content("Attach either one source image or at least one named view, but not both.")
                .ephemeral(true),
        )
        .await?;
        return Ok(());
    }

    let model_name = model.unwrap_or_else(|| {
        if named_count > 0 {
            "hunyuan3d-2mv-turbo:fp16".to_string()
        } else {
            mold_core::manifest::HUNYUAN3D_DEFAULT_MODEL.to_string()
        }
    });
    let models = ctx.data().cached_models().await;
    let model_entry = models.iter().find(|entry| entry.info.name == model_name);
    let fallback = model_entry
        .is_none()
        .then(|| mold_core::manifest::find_manifest(&model_name))
        .flatten();
    let family = model_entry
        .map(|entry| entry.info.family.as_str())
        .or_else(|| fallback.map(|manifest| manifest.family.as_str()));
    if family != Some(mold_core::manifest::HUNYUAN3D_FAMILY) {
        ctx.send(
            poise::CreateReply::default()
                .content("`/mesh` requires a Hunyuan3D model.")
                .ephemeral(true),
        )
        .await?;
        return Ok(());
    }
    if let Some(message) = conditioning_error(&model_name, source.is_some(), named_count) {
        ctx.send(
            poise::CreateReply::default()
                .content(message)
                .ephemeral(true),
        )
        .await?;
        return Ok(());
    }
    let fallback_defaults = fallback.map(generate::defaults_from_manifest);
    let defaults = model_entry
        .map(|entry| &entry.defaults)
        .or(fallback_defaults.as_ref());

    let user_id = ctx.author().id.get();
    if let AuthResult::Denied(message) = checks::check_generate_auth(&ctx).await {
        ctx.send(
            poise::CreateReply::default()
                .content(message)
                .ephemeral(true),
        )
        .await?;
        return Ok(());
    }
    ctx.defer().await?;

    let result = async {
        let source_bytes = match source.as_ref() {
            Some(attachment) => Some(
                generate::fetch_source_image(attachment)
                    .await
                    .map_err(anyhow::Error::msg)?,
            ),
            None => None,
        };
        let mut references = Vec::with_capacity(named_count);
        for (role, attachment) in named {
            if let Some(attachment) = attachment {
                let bytes = generate::fetch_source_image(attachment)
                    .await
                    .map_err(anyhow::Error::msg)?;
                references
                    .push(prepare_named_view(role, attachment, bytes).map_err(anyhow::Error::msg)?);
            }
        }
        let mut request = generate::build_generate_request(BuildParams {
            prompt: "",
            model: &model_name,
            family,
            defaults,
            source_image: source_bytes,
            source_image_name: source
                .as_ref()
                .and_then(|attachment| crate::h3_references::safe_name(&attachment.filename)),
            seed,
            ..Default::default()
        });
        request.references = (!references.is_empty()).then_some(references);
        request.mesh = (texture.is_some()
            || octree.is_some()
            || threshold.is_some()
            || target_faces.is_some())
        .then_some(MeshRequestOptions {
            octree_resolution: octree,
            threshold: threshold.map(|value| value as f32),
            target_faces,
            texture,
            texture_resolution: None,
        });
        handler::run_generation(ctx, request).await
    }
    .await;

    match result {
        Ok(()) => ctx.data().cooldowns.record(user_id),
        Err(error) => {
            ctx.data().quotas.refund(user_id);
            handler::send_error(ctx, &format!("Mesh generation failed: {error}")).await?;
        }
    }
    Ok(())
}

async fn autocomplete_mesh_model(ctx: Context<'_>, partial: &str) -> Vec<String> {
    let cached = ctx
        .data()
        .cached_models()
        .await
        .into_iter()
        .filter(|entry| entry.info.family == mold_core::manifest::HUNYUAN3D_FAMILY)
        .collect::<Vec<_>>();
    let fallback = mold_core::manifest::visible_manifests()
        .filter(|manifest| manifest.family == mold_core::manifest::HUNYUAN3D_FAMILY)
        .map(|manifest| manifest.name.as_str())
        .collect::<Vec<_>>();
    generate::rank_model_suggestions(&cached, &fallback, partial)
}

fn prepare_named_view(
    role: GenerationImageReferenceRole,
    attachment: &serenity::Attachment,
    bytes: Vec<u8>,
) -> Result<GenerationReference, String> {
    let reader = ImageReader::new(Cursor::new(&bytes))
        .with_guessed_format()
        .map_err(|error| format!("Named view could not be identified: {error}"))?;
    let format = reader
        .format()
        .ok_or_else(|| "Named view has an unknown image format".to_string())?;
    let mime_type = match format {
        image::ImageFormat::Png => "image/png",
        image::ImageFormat::Jpeg => "image/jpeg",
        _ => return Err("Named views must be PNG or JPEG.".to_string()),
    };
    let mut decoder = reader
        .into_decoder()
        .map_err(|error| format!("Named view dimensions could not be read: {error}"))?;
    let (width, height) = decoder.dimensions();
    let orientation = decoder
        .orientation()
        .map_err(|error| format!("Named view orientation could not be read: {error}"))?;
    let (width, height) = oriented_dimensions(width, height, orientation);
    drop(decoder);
    Ok(GenerationReference::NamedImage {
        role,
        media: GenerationReferenceAuthority::Inline { data: bytes },
        provenance: GenerationReferenceProvenance {
            name: crate::h3_references::safe_name(&attachment.filename),
            sha256: None,
            crop: None,
        },
        mime_type: mime_type.to_string(),
        width,
        height,
    })
}

fn conditioning_error(model: &str, has_source: bool, named_count: usize) -> Option<&'static str> {
    if mold_core::manifest::hunyuan3d_multiview_model(model) {
        has_source.then_some("Hunyuan3D 2mv requires named views instead of a single source image.")
    } else {
        (named_count > 0).then_some("Named views require a Hunyuan3D 2mv model.")
    }
}

fn oriented_dimensions(
    width: u32,
    height: u32,
    orientation: image::metadata::Orientation,
) -> (u32, u32) {
    match orientation {
        image::metadata::Orientation::Rotate90
        | image::metadata::Orientation::Rotate270
        | image::metadata::Orientation::Rotate90FlipH
        | image::metadata::Orientation::Rotate270FlipH => (height, width),
        _ => (width, height),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mesh_command_has_all_semantic_slots_and_stays_under_the_option_cap() {
        let command = mesh();
        assert!(command.parameters.len() <= 25);
        for name in ["source", "front", "left", "back", "right"] {
            assert!(command.parameters.iter().any(|item| item.name == name));
        }
    }

    #[test]
    fn conditioning_form_must_match_the_selected_checkpoint() {
        assert_eq!(
            conditioning_error("hunyuan3d-2mv-turbo:fp16", true, 0),
            Some("Hunyuan3D 2mv requires named views instead of a single source image.")
        );
        assert_eq!(
            conditioning_error("hunyuan3d-2.1:fp16", false, 2),
            Some("Named views require a Hunyuan3D 2mv model.")
        );
        assert_eq!(conditioning_error("hunyuan3d-2mv:fp16", false, 1), None);
        assert_eq!(conditioning_error("hunyuan3d-2.1:fp16", true, 0), None);
    }

    #[test]
    fn exif_quarter_turns_swap_advertised_dimensions() {
        assert_eq!(
            oriented_dimensions(640, 480, image::metadata::Orientation::Rotate90),
            (480, 640)
        );
        assert_eq!(
            oriented_dimensions(640, 480, image::metadata::Orientation::Rotate270FlipH),
            (480, 640)
        );
        assert_eq!(
            oriented_dimensions(640, 480, image::metadata::Orientation::FlipHorizontal),
            (640, 480)
        );
    }
}
