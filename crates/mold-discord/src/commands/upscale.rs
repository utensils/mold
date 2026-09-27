use crate::checks::{self, AuthResult};
use crate::state::Context;
use anyhow::Result;
use mold_core::{OutputFormat, UpscaleRequest};
use poise::serenity_prelude as serenity;

const MAX_INPUT_BYTES: u64 = 10 * 1024 * 1024;
const MAX_OUTPUT_BYTES: usize = 24 * 1024 * 1024;
const DEFAULT_MODEL: &str = "real-esrgan-x4plus:fp16";

pub fn validate_upscale_attachment(size: u64, content_type: Option<&str>) -> Result<(), String> {
    if size > MAX_INPUT_BYTES {
        return Err("The source image must be 10 MiB or smaller.".into());
    }
    if let Some(kind) = content_type {
        if !matches!(
            kind.split(';').next(),
            Some("image/png" | "image/jpeg" | "image/webp")
        ) {
            return Err("The source image must be PNG, JPEG, or WebP.".into());
        }
    }
    Ok(())
}

pub fn validate_upscale_bytes(bytes: &[u8]) -> Result<(), String> {
    if bytes.len() as u64 > MAX_INPUT_BYTES {
        return Err("The downloaded source image exceeds 10 MiB.".into());
    }
    let format = image::guess_format(bytes)
        .map_err(|_| "The attachment is not a valid image.".to_string())?;
    if !matches!(
        format,
        image::ImageFormat::Png | image::ImageFormat::Jpeg | image::ImageFormat::WebP
    ) {
        return Err("The source image must be PNG, JPEG, or WebP.".into());
    }
    let (width, height) = mold_core::reference_image::oriented_dimensions(bytes)?;
    mold_core::reference_image::validate_reference_image_dimensions(
        "The source image",
        width,
        height,
    )?;
    Ok(())
}

/// Upscale a PNG, JPEG, or WebP attachment with the server's Real-ESRGAN model.
#[poise::command(slash_command)]
pub async fn upscale(
    ctx: Context<'_>,
    #[description = "PNG, JPEG, or WebP image to enlarge"] image: serenity::Attachment,
    #[description = "Upscaler model advertised by the Mold server"] model: Option<String>,
    #[description = "Tile size (omit for the server default)"] tile_size: Option<u32>,
) -> Result<()> {
    if let Err(message) =
        validate_upscale_attachment(image.size as u64, image.content_type.as_deref())
    {
        ctx.send(
            poise::CreateReply::default()
                .content(message)
                .ephemeral(true),
        )
        .await?;
        return Ok(());
    }
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
    ctx.defer_ephemeral().await?;
    let result: Result<(mold_core::UpscaleResponse, String), String> = async {
        let bytes = image
            .download()
            .await
            .map_err(|error| format!("Could not download the source image: {error}"))?;
        validate_upscale_bytes(&bytes)?;
        let filename = image.filename.clone();
        let response = ctx
            .data()
            .client
            .upscale(&UpscaleRequest {
                model: model.unwrap_or_else(|| DEFAULT_MODEL.into()),
                image: bytes,
                output_format: OutputFormat::Png,
                tile_size,
                metadata: None,
            })
            .await
            .map_err(|error| format!("Upscale failed: {error}"))?;
        if response.image.data.len() > MAX_OUTPUT_BYTES {
            return Err("The upscaled image exceeds Discord's 24 MiB delivery limit.".into());
        }
        Ok((response, filename))
    }
    .await;
    match result {
        Ok((response, filename)) => {
            let stem = std::path::Path::new(&filename)
                .file_stem()
                .and_then(|s| s.to_str())
                .unwrap_or("image");
            let output_name = format!("{stem}-upscaled.png");
            let delivery = ctx
                .send(
                    poise::CreateReply::default()
                        .content(format!(
                            "Upscaled {}×{} → {}×{} with `{}`.",
                            response.original_width,
                            response.original_height,
                            response.image.width,
                            response.image.height,
                            response.model
                        ))
                        .attachment(serenity::CreateAttachment::bytes(
                            response.image.data,
                            output_name,
                        ))
                        .ephemeral(true),
                )
                .await;
            match delivery {
                Ok(_) => ctx.data().cooldowns.record(user_id),
                Err(error) => {
                    ctx.data().quotas.refund(user_id);
                    return Err(error.into());
                }
            }
        }
        Err(message) => {
            ctx.data().quotas.refund(user_id);
            ctx.send(
                poise::CreateReply::default()
                    .content(message)
                    .ephemeral(true),
            )
            .await?;
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn attachment_preflight_rejects_large_and_non_image_inputs() {
        for content_type in [
            "image/png",
            "image/jpeg",
            "image/webp",
            "image/png; charset=binary",
        ] {
            assert!(validate_upscale_attachment(MAX_INPUT_BYTES, Some(content_type)).is_ok());
        }
        assert!(validate_upscale_attachment(MAX_INPUT_BYTES + 1, Some("image/png")).is_err());
        assert!(validate_upscale_attachment(1, Some("video/mp4")).is_err());
    }

    #[test]
    fn downloaded_bytes_must_be_a_real_supported_image() {
        assert!(validate_upscale_bytes(b"not an image").is_err());
        assert!(validate_upscale_bytes(&vec![0; MAX_INPUT_BYTES as usize + 1]).is_err());
        let mut png = Vec::new();
        image::DynamicImage::new_rgba8(1, 1)
            .write_to(&mut std::io::Cursor::new(&mut png), image::ImageFormat::Png)
            .unwrap();
        assert!(validate_upscale_bytes(&png).is_ok());
    }
}
