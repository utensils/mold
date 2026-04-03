use crate::state::Context;
use std::time::Duration;

/// Result of a pre-generation authorization check.
pub enum AuthResult {
    /// User is authorized to proceed.
    Allowed,
    /// User is denied with a reason to show (ephemeral).
    Denied(String),
}

/// Check whether the invoking user is allowed to generate.
/// Checks in order: block list, role access, cooldown, quota.
pub async fn check_generate_auth(ctx: &Context<'_>) -> AuthResult {
    let user_id = ctx.author().id.get();
    let data = ctx.data();

    // 1. Block list
    if data.block_list.is_blocked(user_id) {
        return AuthResult::Denied(
            "You have been temporarily blocked from generating images.".into(),
        );
    }

    // 2. Role access (only enforced when roles are configured)
    if !data.config.allowed_roles.unrestricted {
        match extract_member_roles(ctx).await {
            Some(roles) => {
                if !data.config.allowed_roles.check(&roles) {
                    let allowed = data.config.allowed_roles.display_names();
                    return AuthResult::Denied(format!(
                        "You need one of these roles to generate images: {allowed}"
                    ));
                }
            }
            None => {
                return AuthResult::Denied(
                    "This command requires specific roles and can only be used in a server.".into(),
                );
            }
        }
    }

    // 3. Cooldown
    let cooldown = Duration::from_secs(data.config.cooldown_seconds);
    if let Err(remaining) = data.cooldowns.check(user_id, cooldown) {
        let secs = remaining.as_secs() + 1;
        return AuthResult::Denied(format!(
            "Please wait {secs} seconds before generating again."
        ));
    }

    // 4. Quota — atomically consume a slot (refund on generation failure)
    if data
        .quotas
        .consume(user_id, data.config.daily_quota)
        .is_none()
    {
        return AuthResult::Denied(
            "You've reached your daily generation limit. Try again tomorrow!".into(),
        );
    }

    AuthResult::Allowed
}

/// Check block list and role access only (no cooldown or quota).
/// Used for commands like `/expand` that don't consume GPU resources.
pub async fn check_access_only(ctx: &Context<'_>) -> AuthResult {
    let user_id = ctx.author().id.get();
    let data = ctx.data();

    if data.block_list.is_blocked(user_id) {
        return AuthResult::Denied("You have been temporarily blocked from using this bot.".into());
    }

    if !data.config.allowed_roles.unrestricted {
        match extract_member_roles(ctx).await {
            Some(roles) => {
                if !data.config.allowed_roles.check(&roles) {
                    let allowed = data.config.allowed_roles.display_names();
                    return AuthResult::Denied(format!(
                        "You need one of these roles to use this command: {allowed}"
                    ));
                }
            }
            None => {
                return AuthResult::Denied(
                    "This command requires specific roles and can only be used in a server.".into(),
                );
            }
        }
    }

    AuthResult::Allowed
}

/// Extract `(role_id, role_name)` pairs for the invoking user.
/// Returns `None` if not in a guild (DM context).
async fn extract_member_roles(ctx: &Context<'_>) -> Option<Vec<(u64, String)>> {
    // Clone guild roles map and drop the non-Send CacheRef before any await.
    let guild_roles = {
        let guild = ctx.guild()?;
        guild.roles.clone()
    };

    // Get member from interaction (always present for guild slash commands)
    let member = ctx.author_member().await?;

    let roles = member
        .roles
        .iter()
        .map(|role_id| {
            let name = guild_roles
                .get(role_id)
                .map(|r| r.name.clone())
                .unwrap_or_default();
            (role_id.get(), name)
        })
        .collect();

    Some(roles)
}
