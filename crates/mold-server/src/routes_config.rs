//! `/api/config` — the HTTP counterpart of the `mold config` CLI verbs.
//!
//! Shares the CLI's key registry, typed get/set, env-override detection,
//! and DB-vs-TOML surface routing via `mold_core::config_keys`, and its
//! DB persistence via `mold_db::config_sync`. Reads come from the server's
//! in-memory `Config`; writes mutate it in place (so the running server
//! reflects changes immediately) and persist to the owning surface.

use axum::{
    extract::{Path, State},
    http::StatusCode,
    Json,
};
use mold_core::config_keys as keys;
use mold_core::{Config, ConfigEntry, ConfigListing, ConfigProfiles};
use serde::Deserialize;

use crate::routes::ApiError;
use crate::state::AppState;

/// 503 error code when the metadata DB is disabled (`MOLD_DB_DISABLE=1`).
const CONFIG_UNAVAILABLE: &str = "CONFIG_UNAVAILABLE";
/// 403 error code when an env var owns the key at runtime.
const ENV_OVERRIDDEN: &str = "ENV_OVERRIDDEN";
/// 404/422 error code for keys outside the registry.
const UNKNOWN_CONFIG_KEY: &str = "UNKNOWN_CONFIG_KEY";
/// 422 error code for reset attempts on config.toml-owned keys.
const FILE_BACKED_KEY: &str = "FILE_BACKED_KEY";
/// 409 error code for bootstrap namespaces that a live server cannot switch
/// without splitting long-lived workers from HTTP observers.
const RESTART_REQUIRED: &str = "RESTART_REQUIRED";

fn settings_db(state: &AppState) -> Result<&mold_db::MetadataDb, ApiError> {
    state.metadata_db.as_ref().as_ref().ok_or_else(|| {
        ApiError::with_code(
            "config settings are unavailable because the metadata DB is disabled",
            CONFIG_UNAVAILABLE,
            StatusCode::SERVICE_UNAVAILABLE,
        )
    })
}

/// `(source, env_var)` for a key — `"env"` (with the variable name) when an
/// environment variable overrides it at runtime, otherwise the owning
/// storage surface (`"db"` / `"file"`). Matches the CLI's
/// `surface_annotated_value` classification exactly.
fn source_for(key: &str) -> (String, Option<String>) {
    if let Some((var, _)) = keys::env_override_for(key) {
        ("env".to_string(), Some(var.to_string()))
    } else {
        (keys::effective_surface(key).as_str().to_string(), None)
    }
}

fn entry_for(key: &str, value: serde_json::Value) -> ConfigEntry {
    let (source, env_var) = source_for(key);
    ConfigEntry {
        key: key.to_string(),
        value,
        source,
        env_var,
    }
}

fn unknown_key(err: impl std::fmt::Display, status: StatusCode) -> ApiError {
    ApiError::with_code(err.to_string(), UNKNOWN_CONFIG_KEY, status)
}

// ── GET /api/config ──────────────────────────────────────────────────────────

/// Every effective config row — static keys plus `models.<name>.<field>`
/// rows for configured models — exactly like `mold config list --json`.
#[utoipa::path(
    get,
    path = "/api/config",
    tag = "config",
    responses(
        (status = 200, description = "Effective config rows with sources", body = mold_core::ConfigListing),
    )
)]
pub async fn list_config(State(state): State<AppState>) -> Json<ConfigListing> {
    let cfg = state.config.read().await;
    let mut entries = Vec::new();
    for info in keys::ALL_KEYS {
        if let Ok(val) = keys::get_static_value(&cfg, info.key) {
            entries.push(entry_for(info.key, val.to_json()));
        }
    }
    for model_name in cfg.models.keys() {
        for (field, _) in keys::MODEL_FIELDS {
            let full_key = format!("models.{model_name}.{field}");
            if let Ok(val) = keys::get_model_value(&cfg, &full_key) {
                entries.push(entry_for(&full_key, val.to_json()));
            }
        }
    }
    let profile = state
        .metadata_db
        .as_ref()
        .as_ref()
        .map(mold_db::resolve_active_profile);
    Json(ConfigListing { profile, entries })
}

// ── GET /api/config/:key ─────────────────────────────────────────────────────

/// One effective config row (mirrors `mold config get` + `mold config where`).
#[utoipa::path(
    get,
    path = "/api/config/{key}",
    tag = "config",
    params(("key" = String, Path, description = "Config key (e.g. expand.enabled, models.flux-dev:q4.lora)")),
    responses(
        (status = 200, description = "Config row", body = mold_core::ConfigEntry),
        (status = 404, description = "Unknown config key"),
    )
)]
pub async fn get_config_key(
    State(state): State<AppState>,
    Path(key): Path<String>,
) -> Result<Json<ConfigEntry>, ApiError> {
    let cfg = state.config.read().await;
    let value = keys::get_value(&cfg, &key)
        .map_err(|e| unknown_key(e, StatusCode::NOT_FOUND))?
        .to_json();
    Ok(Json(entry_for(&key, value)))
}

// ── PUT /api/config/:key ─────────────────────────────────────────────────────

#[derive(Debug, Deserialize, utoipa::ToSchema)]
pub struct ConfigSetRequest {
    /// New value — string, number, bool, or null (null clears optional keys).
    #[schema(value_type = Object)]
    pub value: serde_json::Value,
}

/// Coerce the JSON body value into the raw string form the shared
/// `set_value` parser expects (the CLI receives values as strings).
fn raw_value(value: &serde_json::Value) -> Result<String, ApiError> {
    match value {
        serde_json::Value::Null => Ok(String::new()),
        serde_json::Value::Bool(b) => Ok(b.to_string()),
        serde_json::Value::Number(n) => Ok(n.to_string()),
        serde_json::Value::String(s) => Ok(s.clone()),
        _ => Err(ApiError::validation(
            "value must be a scalar (string, number, bool, or null)",
        )),
    }
}

/// Set a config key, routed by surface exactly like `mold config set`:
/// DB-backed keys land in the settings DB for the active profile, file
/// keys are written back to config.toml.
#[utoipa::path(
    put,
    path = "/api/config/{key}",
    tag = "config",
    params(("key" = String, Path, description = "Config key")),
    request_body = ConfigSetRequest,
    responses(
        (status = 200, description = "Updated config row", body = mold_core::ConfigEntry),
        (status = 403, description = "Key is overridden by an environment variable"),
        (status = 422, description = "Unknown key or invalid value"),
        (status = 503, description = "Metadata DB disabled (DB-backed key)"),
    )
)]
pub async fn put_config_key(
    State(state): State<AppState>,
    Path(key): Path<String>,
    Json(req): Json<ConfigSetRequest>,
) -> Result<Json<ConfigEntry>, ApiError> {
    if !keys::is_known_key(&key) {
        return Err(unknown_key(
            keys::unknown_key_error(&key),
            StatusCode::UNPROCESSABLE_ENTITY,
        ));
    }
    // output_dir is a server-lifetime namespace. The durable chain runner,
    // queued generation attempts, gallery publication gate, reconciliation,
    // and HTTP routes must all agree on the directory captured at boot.
    // Persisting a next-start value while the listener remains live would
    // split those authorities, so the live API rejects the write before
    // parsing, mutating, or persisting anything. Offline `mold config set`
    // remains the supported editor.
    if key == "output_dir" {
        return Err(ApiError::with_code(
            "'output_dir' is fixed for the lifetime of mold serve; stop the server, run \
             `mold config set output_dir <path>`, then restart it",
            RESTART_REQUIRED,
            StatusCode::CONFLICT,
        ));
    }
    if let Some((var, _)) = keys::env_override_for(&key) {
        return Err(ApiError::with_code(
            format!("'{key}' is set by {var} in the environment — unset it to edit"),
            ENV_OVERRIDDEN,
            StatusCode::FORBIDDEN,
        ));
    }
    let raw = raw_value(&req.value)?;

    // Route by surface BEFORE mutating anything so a DB-backed write with
    // the DB disabled fails without leaving the in-memory config changed.
    let surface = keys::effective_surface(&key);
    if surface == keys::Surface::Db {
        settings_db(&state)?;
    }

    let mut cfg = state.config.write().await;
    keys::set_value(&mut cfg, &key, &raw).map_err(|e| ApiError::validation(e.to_string()))?;

    match surface {
        keys::Surface::Db => {
            let db = settings_db(&state)?;
            let persisted = if key.starts_with("models.") {
                mold_db::config_sync::persist_model_field(db, &cfg, &key)
            } else {
                mold_db::config_sync::persist_config_key(db, &cfg, &key)
            };
            persisted
                .map_err(|e| ApiError::internal(format!("failed to persist '{key}': {e:#}")))?;
        }
        keys::Surface::File => {
            // Bootstrap-only writer, mirroring the CLI — DB-owned fields
            // that the hydrate hook mixed in must not leak back into TOML.
            cfg.save_bootstrap_only().map_err(|e| {
                ApiError::internal(format!("failed to write config.toml for '{key}': {e:#}"))
            })?;
        }
    }

    let value = keys::get_value(&cfg, &key)
        .map(|v| v.to_json())
        .unwrap_or(serde_json::Value::Null);
    Ok(Json(ConfigEntry {
        key,
        value,
        source: surface.as_str().to_string(),
        env_var: None,
    }))
}

// ── DELETE /api/config/:key ──────────────────────────────────────────────────

/// Reset a DB-backed key (mirrors `mold config reset`): drops the settings
/// or model_prefs row for the active profile so the next read falls back
/// to config.toml / env / the compiled default. The response reports the
/// post-reset fallback value. File-backed keys are rejected — edit them
/// via `PUT /api/config/:key` instead.
#[utoipa::path(
    delete,
    path = "/api/config/{key}",
    tag = "config",
    params(("key" = String, Path, description = "Config key")),
    responses(
        (status = 200, description = "Key reset; body carries the fallback value", body = mold_core::ConfigEntry),
        (status = 422, description = "Unknown key or file-backed key"),
        (status = 503, description = "Metadata DB disabled"),
    )
)]
pub async fn delete_config_key(
    State(state): State<AppState>,
    Path(key): Path<String>,
) -> Result<Json<ConfigEntry>, ApiError> {
    if !keys::is_known_key(&key) {
        return Err(unknown_key(
            keys::unknown_key_error(&key),
            StatusCode::UNPROCESSABLE_ENTITY,
        ));
    }
    if keys::effective_surface(&key) != keys::Surface::Db {
        return Err(ApiError::with_code(
            format!("'{key}' is stored in config.toml — edit it via PUT /api/config/{key}"),
            FILE_BACKED_KEY,
            StatusCode::UNPROCESSABLE_ENTITY,
        ));
    }
    let db = settings_db(&state)?;
    let profile = mold_db::resolve_active_profile(db);
    let dropped = if key.starts_with("models.") {
        mold_db::config_sync::reset_model_field(db, &profile, &key)
    } else {
        mold_db::config_sync::reset_global_key(db, &profile, &key)
    };
    dropped.map_err(|e| ApiError::internal(format!("failed to reset '{key}': {e:#}")))?;

    // Reload from the surviving surfaces (TOML + compiled defaults; the
    // process-wide DB hydrate hook, when installed, no longer sees the
    // dropped row) to learn what the key falls back to, and mirror that
    // into the running server's config.
    let fresh = Config::load_or_default();
    let fallback = keys::get_value(&fresh, &key).ok();
    {
        let mut cfg = state.config.write().await;
        let raw = fallback
            .as_ref()
            .map(|v| v.raw())
            .unwrap_or_else(|| "none".to_string());
        // Best-effort: optional keys clear on "", per-model keys clear on
        // "none"; required keys keep their current runtime value if the
        // fallback raw form doesn't parse.
        let _ = keys::set_value(&mut cfg, &key, &raw);
    }

    let (source, env_var) = match keys::env_override_for(&key) {
        Some((var, _)) => ("env".to_string(), Some(var.to_string())),
        None => ("default".to_string(), None),
    };
    Ok(Json(ConfigEntry {
        key,
        value: fallback
            .map(|v| v.to_json())
            .unwrap_or(serde_json::Value::Null),
        source,
        env_var,
    }))
}

// ── GET /api/config/profiles ─────────────────────────────────────────────────

/// List settings profiles and the active one. The active profile resolves
/// `MOLD_PROFILE` (env) → the stored `profile.active` row → `"default"`.
#[utoipa::path(
    get,
    path = "/api/config/profiles",
    tag = "config",
    responses(
        (status = 200, description = "Profiles and the active profile", body = mold_core::ConfigProfiles),
        (status = 503, description = "Metadata DB disabled"),
    )
)]
pub async fn list_config_profiles(
    State(state): State<AppState>,
) -> Result<Json<ConfigProfiles>, ApiError> {
    let db = settings_db(&state)?;
    Ok(Json(profiles_snapshot(db)?))
}

fn profiles_snapshot(db: &mold_db::MetadataDb) -> Result<ConfigProfiles, ApiError> {
    let active = mold_db::resolve_active_profile(db);
    let mut profiles = mold_db::settings::list_profiles(db)
        .map_err(|e| ApiError::internal(format!("failed to list profiles: {e:#}")))?;
    if !profiles.iter().any(|p| p == &active) {
        profiles.push(active.clone());
        profiles.sort();
    }
    Ok(ConfigProfiles { active, profiles })
}

// ── PUT /api/config/profile ──────────────────────────────────────────────────

#[derive(Debug, Deserialize, utoipa::ToSchema)]
pub struct ProfileSetRequest {
    /// Profile name to activate (created implicitly on first write).
    pub name: String,
}

/// Switch the active settings profile by writing the `profile.active`
/// meta-row (the same mechanism the CLI's `--profile` resolution reads).
/// A `MOLD_PROFILE` environment variable still wins at runtime — the
/// response reports the profile that is actually active.
#[utoipa::path(
    put,
    path = "/api/config/profile",
    tag = "config",
    request_body = ProfileSetRequest,
    responses(
        (status = 200, description = "Active profile after the switch", body = mold_core::ConfigProfiles),
        (status = 422, description = "Invalid profile name"),
        (status = 503, description = "Metadata DB disabled"),
    )
)]
pub async fn put_config_profile(
    State(state): State<AppState>,
    Json(req): Json<ProfileSetRequest>,
) -> Result<Json<ConfigProfiles>, ApiError> {
    let name = req.name.trim();
    if name.is_empty() {
        return Err(ApiError::validation("profile name cannot be empty"));
    }
    let db = settings_db(&state)?;
    mold_db::settings::set_active_profile(db, name)
        .map_err(|e| ApiError::internal(format!("failed to switch profile: {e:#}")))?;
    Ok(Json(profiles_snapshot(db)?))
}
