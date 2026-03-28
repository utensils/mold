use crate::format;
use crate::handler;
use crate::state::Context;
use anyhow::Result;
use mold_core::ExpandRequest;

/// Expand a short prompt into a detailed image generation prompt using LLM.
#[poise::command(slash_command)]
pub async fn expand(
    ctx: Context<'_>,
    #[description = "Short prompt to expand (e.g. 'a cat')"] prompt: String,
    #[description = "Target model family for prompt style (flux, sdxl, sd15, sd3)"]
    model_family: Option<String>,
    #[description = "Number of prompt variations (1-5)"] variations: Option<usize>,
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

    ctx.defer().await?;

    let family = model_family.unwrap_or_else(|| "flux".to_string());
    let variations = variations.unwrap_or(1).clamp(1, 5);

    let req = ExpandRequest {
        prompt: prompt.clone(),
        model_family: family.clone(),
        variations,
    };

    match ctx.data().client.expand_prompt(&req).await {
        Ok(resp) => {
            let embed_data = format::format_expand_result(&resp, &prompt, &family);
            let embed = handler::embed_data_to_create_embed(&embed_data);
            ctx.send(poise::CreateReply::default().embed(embed)).await?;
        }
        Err(e) => {
            let msg = if mold_core::MoldClient::is_connection_error(&e) {
                "Could not connect to the mold server. Is it running?".to_string()
            } else {
                format!("Prompt expansion failed: {e}")
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
    fn expand_request_defaults() {
        let req = ExpandRequest {
            prompt: "a cat".to_string(),
            model_family: "flux".to_string(),
            variations: 1,
        };
        assert_eq!(req.prompt, "a cat");
        assert_eq!(req.model_family, "flux");
        assert_eq!(req.variations, 1);
    }

    #[test]
    fn expand_request_with_variations() {
        let req = ExpandRequest {
            prompt: "sunset".to_string(),
            model_family: "sdxl".to_string(),
            variations: 3,
        };
        assert_eq!(req.variations, 3);
        assert_eq!(req.model_family, "sdxl");
    }
}
