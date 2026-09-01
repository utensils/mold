use crate::checks::{self, AuthResult};
use crate::format;
use crate::handler;
use crate::state::Context;
use anyhow::Result;
use mold_core::{RemixDimension, RemixRequest, RemixSourceKind};

/// Rewrite an existing prompt into subject-preserving alternatives.
#[poise::command(slash_command)]
pub async fn remix(
    ctx: Context<'_>,
    #[description = "Prompt to remix (the subject and constraints are kept)"] prompt: String,
    #[description = "Target model family for prompt style (flux, sdxl, sd15, wan, ltx2)"]
    model_family: Option<String>,
    #[description = "Number of alternatives (1-5, capped for Discord embeds)"] variations: Option<
        usize,
    >,
    #[description = "Dimensions to vary, comma-separated: composition, camera, lighting, setting, mood, movement, style"]
    dimensions: Option<String>,
    #[description = "Locked style kept in every alternative"] style: Option<String>,
) -> Result<()> {
    if let AuthResult::Denied(msg) = checks::check_access_only(&ctx).await {
        ctx.send(poise::CreateReply::default().content(msg).ephemeral(true))
            .await?;
        return Ok(());
    }
    if prompt.trim().is_empty() {
        ctx.send(
            poise::CreateReply::default()
                .content("Prompt cannot be empty.")
                .ephemeral(true),
        )
        .await?;
        return Ok(());
    }
    let requested = match parse_dimensions(dimensions.as_deref()) {
        Ok(dimensions) => dimensions,
        Err(message) => {
            ctx.send(
                poise::CreateReply::default()
                    .content(message)
                    .ephemeral(true),
            )
            .await?;
            return Ok(());
        }
    };

    ctx.defer().await?;

    let family = model_family.unwrap_or_else(|| "flux".to_string());
    let variations = variations
        .unwrap_or(3)
        .clamp(1, mold_core::expand::DISCORD_MAX_VARIATIONS);
    let style = style
        .map(|value| value.trim().to_string())
        .filter(|value| !value.is_empty());
    let req = RemixRequest {
        source_prompt: prompt.clone(),
        root_prompt: None,
        source_kind: RemixSourceKind::Direct,
        model_family: family.clone(),
        variations,
        style,
        task: None,
        dimensions: requested,
        context: None,
    };

    match ctx.data().client.remix_prompt(&req).await {
        Ok(resp) => {
            let embed_data = format::format_remix_result(&resp, &family);
            let embed = handler::embed_data_to_create_embed(&embed_data);
            ctx.send(poise::CreateReply::default().embed(embed)).await?;
        }
        Err(e) => {
            let msg = if mold_core::MoldClient::is_connection_error(&e) {
                "Could not connect to the mold server. Is it running?".to_string()
            } else {
                format!("Prompt remix failed: {e}")
            };
            handler::send_error(ctx, &msg).await?;
        }
    }

    Ok(())
}

/// Parse a comma-separated dimension list; an empty list means the
/// server's task-safe default set.
pub(crate) fn parse_dimensions(
    value: Option<&str>,
) -> std::result::Result<Vec<RemixDimension>, String> {
    let mut parsed = Vec::new();
    for token in value
        .unwrap_or_default()
        .split(',')
        .map(str::trim)
        .filter(|token| !token.is_empty())
    {
        let dimension = token
            .to_ascii_lowercase()
            .parse::<RemixDimension>()
            .map_err(|_| {
                format!(
                    "Unknown remix dimension '{token}'. Valid: composition, camera, lighting, setting, mood, movement, style."
                )
            })?;
        if !parsed.contains(&dimension) {
            parsed.push(dimension);
        }
    }
    Ok(parsed)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dimensions_parse_case_insensitively_and_dedupe() {
        assert_eq!(
            parse_dimensions(Some("Camera, lighting,camera")).unwrap(),
            vec![RemixDimension::Camera, RemixDimension::Lighting]
        );
        assert!(parse_dimensions(None).unwrap().is_empty());
        assert!(parse_dimensions(Some("colour")).is_err());
    }
}
