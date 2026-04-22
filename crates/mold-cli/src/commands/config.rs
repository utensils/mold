use anyhow::{anyhow, bail, Result};
use clap_complete::engine::CompletionCandidate;
use colored::Colorize;
use mold_core::config::Config;
use mold_core::Scheduler;

use crate::theme;
use crate::AlreadyReported;

// ── Key registry ────────────────────────────────────────────────────

#[derive(Debug, Clone, Copy, PartialEq)]
enum ValueType {
    String,
    OptionalString,
    Bool,
    U16,
    U32,
    F64,
}

#[allow(dead_code)]
struct ConfigKeyInfo {
    key: &'static str,
    value_type: ValueType,
    env_var: Option<&'static str>,
    section: &'static str,
}

const ALL_KEYS: &[ConfigKeyInfo] = &[
    // General
    ConfigKeyInfo {
        key: "default_model",
        value_type: ValueType::String,
        env_var: Some("MOLD_DEFAULT_MODEL"),
        section: "General",
    },
    ConfigKeyInfo {
        key: "models_dir",
        value_type: ValueType::String,
        env_var: Some("MOLD_MODELS_DIR"),
        section: "General",
    },
    ConfigKeyInfo {
        key: "output_dir",
        value_type: ValueType::OptionalString,
        env_var: Some("MOLD_OUTPUT_DIR"),
        section: "General",
    },
    ConfigKeyInfo {
        key: "server_port",
        value_type: ValueType::U16,
        env_var: None,
        section: "General",
    },
    ConfigKeyInfo {
        key: "default_width",
        value_type: ValueType::U32,
        env_var: None,
        section: "General",
    },
    ConfigKeyInfo {
        key: "default_height",
        value_type: ValueType::U32,
        env_var: None,
        section: "General",
    },
    ConfigKeyInfo {
        key: "default_steps",
        value_type: ValueType::U32,
        env_var: None,
        section: "General",
    },
    ConfigKeyInfo {
        key: "embed_metadata",
        value_type: ValueType::Bool,
        env_var: Some("MOLD_EMBED_METADATA"),
        section: "General",
    },
    ConfigKeyInfo {
        key: "t5_variant",
        value_type: ValueType::OptionalString,
        env_var: Some("MOLD_T5_VARIANT"),
        section: "General",
    },
    ConfigKeyInfo {
        key: "qwen3_variant",
        value_type: ValueType::OptionalString,
        env_var: Some("MOLD_QWEN3_VARIANT"),
        section: "General",
    },
    ConfigKeyInfo {
        key: "default_negative_prompt",
        value_type: ValueType::OptionalString,
        env_var: None,
        section: "General",
    },
    // Expand
    ConfigKeyInfo {
        key: "expand.enabled",
        value_type: ValueType::Bool,
        env_var: Some("MOLD_EXPAND"),
        section: "Expand",
    },
    ConfigKeyInfo {
        key: "expand.backend",
        value_type: ValueType::String,
        env_var: Some("MOLD_EXPAND_BACKEND"),
        section: "Expand",
    },
    ConfigKeyInfo {
        key: "expand.model",
        value_type: ValueType::String,
        env_var: Some("MOLD_EXPAND_MODEL"),
        section: "Expand",
    },
    ConfigKeyInfo {
        key: "expand.api_model",
        value_type: ValueType::String,
        env_var: Some("MOLD_EXPAND_MODEL"),
        section: "Expand",
    },
    ConfigKeyInfo {
        key: "expand.temperature",
        value_type: ValueType::F64,
        env_var: Some("MOLD_EXPAND_TEMPERATURE"),
        section: "Expand",
    },
    ConfigKeyInfo {
        key: "expand.top_p",
        value_type: ValueType::F64,
        env_var: None,
        section: "Expand",
    },
    ConfigKeyInfo {
        key: "expand.max_tokens",
        value_type: ValueType::U32,
        env_var: None,
        section: "Expand",
    },
    ConfigKeyInfo {
        key: "expand.thinking",
        value_type: ValueType::Bool,
        env_var: Some("MOLD_EXPAND_THINKING"),
        section: "Expand",
    },
    // Logging
    ConfigKeyInfo {
        key: "logging.level",
        value_type: ValueType::String,
        env_var: None,
        section: "Logging",
    },
    ConfigKeyInfo {
        key: "logging.file",
        value_type: ValueType::Bool,
        env_var: None,
        section: "Logging",
    },
    ConfigKeyInfo {
        key: "logging.dir",
        value_type: ValueType::OptionalString,
        env_var: None,
        section: "Logging",
    },
    ConfigKeyInfo {
        key: "logging.max_days",
        value_type: ValueType::U32,
        env_var: None,
        section: "Logging",
    },
    // RunPod
    ConfigKeyInfo {
        key: "runpod.api_key",
        value_type: ValueType::OptionalString,
        env_var: Some("RUNPOD_API_KEY"),
        section: "RunPod",
    },
    ConfigKeyInfo {
        key: "runpod.default_gpu",
        value_type: ValueType::OptionalString,
        env_var: None,
        section: "RunPod",
    },
    ConfigKeyInfo {
        key: "runpod.default_datacenter",
        value_type: ValueType::OptionalString,
        env_var: None,
        section: "RunPod",
    },
    ConfigKeyInfo {
        key: "runpod.default_network_volume_id",
        value_type: ValueType::OptionalString,
        env_var: None,
        section: "RunPod",
    },
    ConfigKeyInfo {
        key: "runpod.auto_teardown",
        value_type: ValueType::Bool,
        env_var: None,
        section: "RunPod",
    },
    ConfigKeyInfo {
        key: "runpod.auto_teardown_idle_mins",
        value_type: ValueType::U32,
        env_var: None,
        section: "RunPod",
    },
    ConfigKeyInfo {
        key: "runpod.cost_alert_usd",
        value_type: ValueType::F64,
        env_var: None,
        section: "RunPod",
    },
    ConfigKeyInfo {
        key: "runpod.endpoint",
        value_type: ValueType::OptionalString,
        env_var: None,
        section: "RunPod",
    },
];

/// Per-model field names and their value types.
const MODEL_FIELDS: &[(&str, ValueType)] = &[
    ("default_steps", ValueType::U32),
    ("default_guidance", ValueType::F64),
    ("default_width", ValueType::U32),
    ("default_height", ValueType::U32),
    ("scheduler", ValueType::OptionalString),
    ("negative_prompt", ValueType::OptionalString),
    ("lora", ValueType::OptionalString),
    ("lora_scale", ValueType::F64),
];

// ── Value representation ────────────────────────────────────────────

#[derive(Debug, Clone)]
enum ConfigValue {
    String(String),
    U16(u16),
    U32(u32),
    F64(f64),
    Bool(bool),
    None,
}

impl ConfigValue {
    fn display(&self) -> String {
        match self {
            Self::String(s) => s.clone(),
            Self::U16(v) => v.to_string(),
            Self::U32(v) => v.to_string(),
            Self::F64(v) => format!("{v:.}"),
            Self::Bool(v) => if *v { "true" } else { "false" }.to_string(),
            Self::None => "(not set)".to_string(),
        }
    }

    fn raw(&self) -> String {
        match self {
            Self::String(s) => s.clone(),
            Self::U16(v) => v.to_string(),
            Self::U32(v) => v.to_string(),
            Self::F64(v) => format!("{v:.}"),
            Self::Bool(v) => if *v { "true" } else { "false" }.to_string(),
            Self::None => String::new(),
        }
    }

    fn to_json(&self) -> serde_json::Value {
        match self {
            Self::String(s) => serde_json::Value::String(s.clone()),
            Self::U16(v) => serde_json::json!(*v),
            Self::U32(v) => serde_json::json!(*v),
            Self::F64(v) => serde_json::json!(*v),
            Self::Bool(v) => serde_json::Value::Bool(*v),
            Self::None => serde_json::Value::Null,
        }
    }
}

// ── Key lookup helpers ──────────────────────────────────────────────

fn find_static_key(key: &str) -> Option<&'static ConfigKeyInfo> {
    ALL_KEYS.iter().find(|k| k.key == key)
}

/// Parse a `models.<name>.<field>` key, returning (model_name, field_name, value_type).
fn parse_model_key(key: &str) -> Result<(&str, &str, ValueType)> {
    let rest = key
        .strip_prefix("models.")
        .ok_or_else(|| anyhow!("not a model key: {key}"))?;
    // Find the last dot — everything before it is the model name (may contain dots/colons),
    // everything after is the field name.
    let last_dot = rest
        .rfind('.')
        .ok_or_else(|| anyhow!("invalid model key (expected models.<name>.<field>): {key}"))?;
    let model_name = &rest[..last_dot];
    let field_name = &rest[last_dot + 1..];
    if model_name.is_empty() || field_name.is_empty() {
        bail!("invalid model key: {key}");
    }
    let vt = MODEL_FIELDS
        .iter()
        .find(|(f, _)| *f == field_name)
        .map(|(_, vt)| *vt)
        .ok_or_else(|| {
            let valid: Vec<&str> = MODEL_FIELDS.iter().map(|(f, _)| *f).collect();
            anyhow!(
                "unknown model field '{}'. Valid fields: {}",
                field_name,
                valid.join(", ")
            )
        })?;
    Ok((model_name, field_name, vt))
}

// ── Read a config value ─────────────────────────────────────────────

fn get_value(config: &Config, key: &str) -> Result<ConfigValue> {
    // Try static keys first
    if find_static_key(key).is_some() {
        return get_static_value(config, key);
    }
    // Try model keys
    if key.starts_with("models.") {
        return get_model_value(config, key);
    }
    // Unknown key
    Err(unknown_key_error(key))
}

fn get_static_value(config: &Config, key: &str) -> Result<ConfigValue> {
    Ok(match key {
        // General
        "default_model" => ConfigValue::String(config.default_model.clone()),
        "models_dir" => ConfigValue::String(config.models_dir.clone()),
        "output_dir" => match &config.output_dir {
            Some(s) => ConfigValue::String(s.clone()),
            None => ConfigValue::None,
        },
        "server_port" => ConfigValue::U16(config.server_port),
        "default_width" => ConfigValue::U32(config.default_width),
        "default_height" => ConfigValue::U32(config.default_height),
        "default_steps" => ConfigValue::U32(config.default_steps),
        "embed_metadata" => ConfigValue::Bool(config.embed_metadata),
        "t5_variant" => match &config.t5_variant {
            Some(s) => ConfigValue::String(s.clone()),
            None => ConfigValue::String("auto".into()),
        },
        "qwen3_variant" => match &config.qwen3_variant {
            Some(s) => ConfigValue::String(s.clone()),
            None => ConfigValue::String("auto".into()),
        },
        "default_negative_prompt" => match &config.default_negative_prompt {
            Some(s) => ConfigValue::String(s.clone()),
            None => ConfigValue::None,
        },
        // Expand
        "expand.enabled" => ConfigValue::Bool(config.expand.enabled),
        "expand.backend" => ConfigValue::String(config.expand.backend.clone()),
        "expand.model" => ConfigValue::String(config.expand.model.clone()),
        "expand.api_model" => ConfigValue::String(config.expand.api_model.clone()),
        "expand.temperature" => ConfigValue::F64(config.expand.temperature),
        "expand.top_p" => ConfigValue::F64(config.expand.top_p),
        "expand.max_tokens" => ConfigValue::U32(config.expand.max_tokens),
        "expand.thinking" => ConfigValue::Bool(config.expand.thinking),
        // Logging
        "logging.level" => ConfigValue::String(config.logging.level.clone()),
        "logging.file" => ConfigValue::Bool(config.logging.file),
        "logging.dir" => match &config.logging.dir {
            Some(s) => ConfigValue::String(s.clone()),
            None => ConfigValue::None,
        },
        "logging.max_days" => ConfigValue::U32(config.logging.max_days),
        // RunPod
        "runpod.api_key" => match &config.runpod.api_key {
            Some(_) => ConfigValue::String("<set>".to_string()),
            None => ConfigValue::None,
        },
        "runpod.default_gpu" => match &config.runpod.default_gpu {
            Some(s) => ConfigValue::String(s.clone()),
            None => ConfigValue::None,
        },
        "runpod.default_datacenter" => match &config.runpod.default_datacenter {
            Some(s) => ConfigValue::String(s.clone()),
            None => ConfigValue::None,
        },
        "runpod.default_network_volume_id" => match &config.runpod.default_network_volume_id {
            Some(s) => ConfigValue::String(s.clone()),
            None => ConfigValue::None,
        },
        "runpod.auto_teardown" => ConfigValue::Bool(config.runpod.auto_teardown),
        "runpod.auto_teardown_idle_mins" => ConfigValue::U32(config.runpod.auto_teardown_idle_mins),
        "runpod.cost_alert_usd" => ConfigValue::F64(config.runpod.cost_alert_usd),
        "runpod.endpoint" => match &config.runpod.endpoint {
            Some(s) => ConfigValue::String(s.clone()),
            None => ConfigValue::None,
        },
        _ => return Err(unknown_key_error(key)),
    })
}

fn get_model_value(config: &Config, key: &str) -> Result<ConfigValue> {
    let (model_name, field_name, _) = parse_model_key(key)?;
    let mc = config.models.get(model_name).ok_or_else(|| {
        anyhow!(
            "no model '{}' in config. Run {} to see configured models.",
            model_name,
            "mold config list".bold()
        )
    })?;
    Ok(match field_name {
        "default_steps" => mc
            .default_steps
            .map(ConfigValue::U32)
            .unwrap_or(ConfigValue::None),
        "default_guidance" => mc
            .default_guidance
            .map(ConfigValue::F64)
            .unwrap_or(ConfigValue::None),
        "default_width" => mc
            .default_width
            .map(ConfigValue::U32)
            .unwrap_or(ConfigValue::None),
        "default_height" => mc
            .default_height
            .map(ConfigValue::U32)
            .unwrap_or(ConfigValue::None),
        "scheduler" => mc
            .scheduler
            .map(|s| ConfigValue::String(s.to_string()))
            .unwrap_or(ConfigValue::None),
        "negative_prompt" => mc
            .negative_prompt
            .as_ref()
            .map(|s| ConfigValue::String(s.clone()))
            .unwrap_or(ConfigValue::None),
        "lora" => mc
            .lora
            .as_ref()
            .map(|s| ConfigValue::String(s.clone()))
            .unwrap_or(ConfigValue::None),
        "lora_scale" => mc
            .lora_scale
            .map(ConfigValue::F64)
            .unwrap_or(ConfigValue::None),
        _ => return Err(unknown_key_error(key)),
    })
}

// ── Write a config value ────────────────────────────────────────────

fn set_value(config: &mut Config, key: &str, raw: &str) -> Result<()> {
    if find_static_key(key).is_some() {
        return set_static_value(config, key, raw);
    }
    if key.starts_with("models.") {
        return set_model_value(config, key, raw);
    }
    Err(unknown_key_error(key))
}

fn set_static_value(config: &mut Config, key: &str, raw: &str) -> Result<()> {
    match key {
        // General
        "default_model" => config.default_model = parse_string(raw)?,
        "models_dir" => config.models_dir = parse_string(raw)?,
        "output_dir" => config.output_dir = parse_optional_string(raw),
        "server_port" => config.server_port = parse_u16(raw, 1, 65535, key)?,
        "default_width" => config.default_width = parse_u32(raw, 64, 8192, key)?,
        "default_height" => config.default_height = parse_u32(raw, 64, 8192, key)?,
        "default_steps" => config.default_steps = parse_u32(raw, 1, 1000, key)?,
        "embed_metadata" => config.embed_metadata = parse_bool(raw, key)?,
        "t5_variant" => {
            let val = parse_optional_string(raw);
            if let Some(ref v) = val {
                validate_enum(v, &["auto", "fp16", "q8", "q6", "q5", "q4", "q3"], key)?;
                config.t5_variant = if v == "auto" { None } else { val };
            } else {
                config.t5_variant = None;
            }
        }
        "qwen3_variant" => {
            let val = parse_optional_string(raw);
            if let Some(ref v) = val {
                validate_enum(v, &["auto", "bf16", "q8", "q6", "iq4", "q3"], key)?;
                config.qwen3_variant = if v == "auto" { None } else { val };
            } else {
                config.qwen3_variant = None;
            }
        }
        "default_negative_prompt" => config.default_negative_prompt = parse_optional_string(raw),
        // Expand
        "expand.enabled" => config.expand.enabled = parse_bool(raw, key)?,
        "expand.backend" => config.expand.backend = parse_string(raw)?,
        "expand.model" => config.expand.model = parse_string(raw)?,
        "expand.api_model" => config.expand.api_model = parse_string(raw)?,
        "expand.temperature" => config.expand.temperature = parse_f64(raw, 0.0, 2.0, key)?,
        "expand.top_p" => config.expand.top_p = parse_f64(raw, 0.0, 1.0, key)?,
        "expand.max_tokens" => config.expand.max_tokens = parse_u32(raw, 1, 65535, key)?,
        "expand.thinking" => config.expand.thinking = parse_bool(raw, key)?,
        // Logging
        "logging.level" => {
            validate_enum(raw, &["trace", "debug", "info", "warn", "error"], key)?;
            config.logging.level = raw.to_string();
        }
        "logging.file" => config.logging.file = parse_bool(raw, key)?,
        "logging.dir" => config.logging.dir = parse_optional_string(raw),
        "logging.max_days" => config.logging.max_days = parse_u32(raw, 1, 3650, key)?,
        // RunPod
        "runpod.api_key" => config.runpod.api_key = parse_optional_string(raw),
        "runpod.default_gpu" => config.runpod.default_gpu = parse_optional_string(raw),
        "runpod.default_datacenter" => {
            config.runpod.default_datacenter = parse_optional_string(raw)
        }
        "runpod.default_network_volume_id" => {
            config.runpod.default_network_volume_id = parse_optional_string(raw)
        }
        "runpod.auto_teardown" => config.runpod.auto_teardown = parse_bool(raw, key)?,
        "runpod.auto_teardown_idle_mins" => {
            config.runpod.auto_teardown_idle_mins = parse_u32(raw, 0, 10_080, key)?
        }
        "runpod.cost_alert_usd" => config.runpod.cost_alert_usd = parse_f64(raw, 0.0, 1000.0, key)?,
        "runpod.endpoint" => config.runpod.endpoint = parse_optional_string(raw),
        _ => return Err(unknown_key_error(key)),
    }
    Ok(())
}

fn set_model_value(config: &mut Config, key: &str, raw: &str) -> Result<()> {
    let (model_name, field_name, _) = parse_model_key(key)?;
    // Create model entry if it doesn't exist
    let mc = config.models.entry(model_name.to_string()).or_default();
    match field_name {
        "default_steps" => mc.default_steps = parse_optional_u32(raw, 1, 1000, key)?,
        "default_guidance" => mc.default_guidance = parse_optional_f64(raw, 0.0, 100.0, key)?,
        "default_width" => mc.default_width = parse_optional_u32(raw, 64, 8192, key)?,
        "default_height" => mc.default_height = parse_optional_u32(raw, 64, 8192, key)?,
        "scheduler" => {
            let val = parse_optional_string(raw);
            mc.scheduler = match val.as_deref() {
                Some("ddim") => Some(Scheduler::Ddim),
                Some("euler-ancestral") => Some(Scheduler::EulerAncestral),
                Some("uni-pc") => Some(Scheduler::UniPc),
                Some(v) => {
                    bail!(
                        "invalid value for {key}: '{}'. Valid: none, ddim, euler-ancestral, uni-pc",
                        v
                    );
                }
                None => None,
            };
        }
        "negative_prompt" => mc.negative_prompt = parse_optional_string(raw),
        "lora" => mc.lora = parse_optional_string(raw),
        "lora_scale" => mc.lora_scale = parse_optional_f64(raw, 0.0, 2.0, key)?,
        _ => return Err(unknown_key_error(key)),
    }
    Ok(())
}

// ── Env override detection ──────────────────────────────────────────

/// Returns (var_name, current_value) if an env var overrides the given key.
fn env_override_for(key: &str) -> Option<(&'static str, String)> {
    let info = find_static_key(key)?;
    let var = info.env_var?;
    std::env::var(var).ok().map(|v| (var, v))
}

// ── Parsing helpers ─────────────────────────────────────────────────

fn parse_string(raw: &str) -> Result<String> {
    if raw.is_empty() {
        bail!("value cannot be empty");
    }
    Ok(raw.to_string())
}

fn parse_optional_string(raw: &str) -> Option<String> {
    match raw.to_lowercase().as_str() {
        "none" | "" => None,
        _ => Some(raw.to_string()),
    }
}

fn parse_bool(raw: &str, key: &str) -> Result<bool> {
    match raw.to_lowercase().as_str() {
        "true" | "on" | "1" | "yes" => Ok(true),
        "false" | "off" | "0" | "no" => Ok(false),
        _ => bail!(
            "invalid value for {key}: '{}'. Use true/false, on/off, or 1/0.",
            raw
        ),
    }
}

fn parse_u16(raw: &str, min: u16, max: u16, key: &str) -> Result<u16> {
    let val: u16 = raw
        .parse()
        .map_err(|_| anyhow!("invalid value for {key}: '{}'. Must be a number.", raw))?;
    if val < min || val > max {
        bail!("invalid value for {key}: {val}. Must be between {min} and {max}.");
    }
    Ok(val)
}

fn parse_u32(raw: &str, min: u32, max: u32, key: &str) -> Result<u32> {
    let val: u32 = raw
        .parse()
        .map_err(|_| anyhow!("invalid value for {key}: '{}'. Must be a number.", raw))?;
    if val < min || val > max {
        bail!("invalid value for {key}: {val}. Must be between {min} and {max}.");
    }
    Ok(val)
}

fn parse_f64(raw: &str, min: f64, max: f64, key: &str) -> Result<f64> {
    let val: f64 = raw
        .parse()
        .map_err(|_| anyhow!("invalid value for {key}: '{}'. Must be a number.", raw))?;
    if val < min || val > max {
        bail!("invalid value for {key}: {val}. Must be between {min} and {max}.");
    }
    Ok(val)
}

fn parse_optional_u32(raw: &str, min: u32, max: u32, key: &str) -> Result<Option<u32>> {
    if raw.eq_ignore_ascii_case("none") || raw.is_empty() {
        return Ok(None);
    }
    parse_u32(raw, min, max, key).map(Some)
}

fn parse_optional_f64(raw: &str, min: f64, max: f64, key: &str) -> Result<Option<f64>> {
    if raw.eq_ignore_ascii_case("none") || raw.is_empty() {
        return Ok(None);
    }
    parse_f64(raw, min, max, key).map(Some)
}

fn validate_enum(raw: &str, valid: &[&str], key: &str) -> Result<()> {
    if !valid.contains(&raw) {
        bail!(
            "invalid value for {key}: '{}'. Valid: {}",
            raw,
            valid.join(", ")
        );
    }
    Ok(())
}

fn unknown_key_error(key: &str) -> anyhow::Error {
    anyhow!(
        "unknown config key: '{}'. Run {} to see all keys.",
        key,
        "mold config list"
    )
}

// ── Surface classification (DB vs TOML) ─────────────────────────────

/// Where a given config key is persisted after issue #265.
///
/// Reads always go through `Config::load_or_default()` which overlays DB
/// values on top of TOML, so callers of `get` don't care. Writes do —
/// `Db` keys must round-trip through `mold-db` instead of rewriting
/// `config.toml`, and `where` prints this so operators can tell which
/// surface owns a given key.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Surface {
    /// Persisted in `config.toml` (paths, bootstrap, credentials).
    File,
    /// Persisted in the SQLite `settings` / `model_prefs` tables.
    Db,
}

impl Surface {
    pub(crate) fn as_str(self) -> &'static str {
        match self {
            Surface::File => "file",
            Surface::Db => "db",
        }
    }
}

/// Classify a key into its storage surface. Unknown keys return `File`
/// as a conservative default (matches pre-issue-#265 behavior).
pub(crate) fn surface_for_key(key: &str) -> Surface {
    // Prefix-based: user-preference slices that moved to DB.
    if key.starts_with("tui.")
        || key.starts_with("expand.")
        || key.starts_with("generate.")
        || key.starts_with("model_prefs.")
    {
        return Surface::Db;
    }
    // Global generation defaults that live on Config but now persist in
    // the DB. Keep the old flat names for backwards-compatible CLI UX.
    matches!(
        key,
        "default_width"
            | "default_height"
            | "default_steps"
            | "embed_metadata"
            | "default_negative_prompt"
            | "t5_variant"
            | "qwen3_variant"
    )
    .then_some(Surface::Db)
    .unwrap_or(Surface::File)
}

/// For `models.<name>.<field>` keys: user-preference fields (generation
/// defaults, LoRA) live in the DB `model_prefs` table; path fields stay
/// in TOML. Returns `None` for non-`models.*` keys.
fn model_field_surface(key: &str) -> Option<Surface> {
    let rest = key.strip_prefix("models.")?;
    let field = rest.rsplit('.').next()?;
    let db_fields = [
        "default_steps",
        "default_guidance",
        "default_width",
        "default_height",
        "scheduler",
        "negative_prompt",
        "lora",
        "lora_scale",
    ];
    Some(if db_fields.contains(&field) {
        Surface::Db
    } else {
        Surface::File
    })
}

// ── Command implementations ─────────────────────────────────────────

pub fn run_list(json: bool) -> Result<()> {
    let config = Config::load_or_default();

    if json {
        return run_list_json(&config);
    }

    let mut current_section = "";

    for info in ALL_KEYS {
        if info.section != current_section {
            if !current_section.is_empty() {
                println!();
            }
            println!("{}", format!("[{}]", info.section).bold());
            current_section = info.section;
        }

        let value = get_static_value(&config, info.key)
            .map(|v| v.display())
            .unwrap_or_else(|_| "?".into());

        let env_note = if let Some((var, _)) = env_override_for(info.key) {
            format!("  (env: {})", var).dimmed().to_string()
        } else {
            String::new()
        };

        println!("  {:<26} = {}{}", info.key, value, env_note);
    }

    // Per-model sections
    for model_name in config.models.keys() {
        println!();
        println!("{}", format!("[Model: {model_name}]").bold());
        for (field, _) in MODEL_FIELDS {
            let full_key = format!("models.{model_name}.{field}");
            let value = get_model_value(&config, &full_key)
                .map(|v| v.display())
                .unwrap_or_else(|_| "?".into());
            println!("  {:<26} = {}", full_key, value);
        }
    }

    Ok(())
}

fn run_list_json(config: &Config) -> Result<()> {
    let mut map = serde_json::Map::new();

    for info in ALL_KEYS {
        if let Ok(val) = get_static_value(config, info.key) {
            map.insert(info.key.to_string(), val.to_json());
        }
    }

    for model_name in config.models.keys() {
        for (field, _) in MODEL_FIELDS {
            let full_key = format!("models.{model_name}.{field}");
            if let Ok(val) = get_model_value(config, &full_key) {
                map.insert(full_key, val.to_json());
            }
        }
    }

    let json = serde_json::to_string_pretty(&serde_json::Value::Object(map))?;
    println!("{json}");
    Ok(())
}

pub fn run_get(key: &str, raw: bool) -> Result<()> {
    let config = Config::load_or_default();

    let value = match get_value(&config, key) {
        Ok(v) => v,
        Err(e) => {
            eprintln!("{} {e}", theme::icon_fail());
            return Err(AlreadyReported.into());
        }
    };

    if raw {
        println!("{}", value.raw());
        return Ok(());
    }

    println!("{} {} = {}", theme::icon_ok(), key.bold(), value.display());

    // Show env override info
    if let Some((var, env_val)) = env_override_for(key) {
        println!(
            "  {} overridden by {} = {}",
            theme::icon_bullet(),
            var.dimmed(),
            env_val.dimmed()
        );
    }

    Ok(())
}

pub fn run_set(key: &str, value: &str) -> Result<()> {
    // Validate the key exists before loading fresh config
    if find_static_key(key).is_none() && !key.starts_with("models.") {
        eprintln!("{} {}", theme::icon_fail(), unknown_key_error(key));
        return Err(AlreadyReported.into());
    }

    // For model keys, validate the field name
    if key.starts_with("models.") {
        if let Err(e) = parse_model_key(key) {
            eprintln!("{} {e}", theme::icon_fail());
            return Err(AlreadyReported.into());
        }
    }

    let mut config = Config::load_or_default();

    if let Err(e) = set_value(&mut config, key, value) {
        eprintln!("{} {e}", theme::icon_fail());
        return Err(AlreadyReported.into());
    }

    // Route persistence by surface: user-preference slices (expand.*,
    // generation defaults, per-model user prefs) go to the DB after
    // issue #265. Path/bootstrap/credential keys still land in the TOML.
    let persisted_surface = persist_value(&config, key)?;

    let display_value = get_value(&config, key)
        .map(|v| v.display())
        .unwrap_or_else(|_| value.to_string());

    println!(
        "{} Set {} = {} [{}]",
        theme::icon_done(),
        key.bold(),
        display_value,
        persisted_surface.as_str().dimmed(),
    );

    // Warn about env override
    if let Some((var, env_val)) = env_override_for(key) {
        eprintln!(
            "{} {} = {} will override this setting at runtime",
            theme::prefix_warning(),
            var.bold(),
            env_val,
        );
    }

    Ok(())
}

/// Persist the post-mutation `config` for this key. Returns the surface
/// that actually took the write.
fn persist_value(config: &Config, key: &str) -> Result<Surface> {
    // Per-model fields split by field name.
    if key.starts_with("models.") {
        if let Some(Surface::Db) = model_field_surface(key) {
            persist_model_field_to_db(config, key)?;
            return Ok(Surface::Db);
        }
        // Path fields still live in TOML.
        config.save()?;
        return Ok(Surface::File);
    }

    match surface_for_key(key) {
        Surface::Db => {
            persist_key_to_db(config, key)?;
            Ok(Surface::Db)
        }
        Surface::File => {
            config.save()?;
            Ok(Surface::File)
        }
    }
}

fn persist_key_to_db(config: &Config, key: &str) -> Result<()> {
    let Some(db) = crate::metadata_db::handle() else {
        bail!(
            "settings DB is unavailable; key '{key}' requires MOLD_DB_DISABLE=0 \
             and a writable MOLD_HOME/mold.db"
        );
    };
    // expand.* is written as a whole blob so env-var precedence and
    // families stay coherent with the load path.
    if key.starts_with("expand.") {
        mold_db::config_sync::save_expand_to_db(db, &config.expand)?;
        return Ok(());
    }
    // All the global generation defaults ride one writer — cheap and
    // keeps the two halves of a `--width --height` pair consistent.
    mold_db::config_sync::save_generate_globals_to_db(db, config)?;
    Ok(())
}

fn persist_model_field_to_db(config: &Config, key: &str) -> Result<()> {
    let Some(db) = crate::metadata_db::handle() else {
        bail!(
            "settings DB is unavailable; key '{key}' requires MOLD_DB_DISABLE=0 \
             and a writable MOLD_HOME/mold.db"
        );
    };
    let (model_name, _, _) = parse_model_key(key)?;
    let mc = config
        .models
        .get(model_name)
        .ok_or_else(|| anyhow!("no model '{model_name}' in config after set"))?;
    // Resolve to canonical so `flux-dev` and `flux-dev:q4` share a row.
    let canonical = mold_core::manifest::resolve_model_name(model_name);
    // Load-merge-save so we don't clobber fields owned by the TUI
    // (seed_mode, batch, last_prompt, etc.) when only a CFG default
    // changed. Assignments are unconditional so that `mold config set
    // models.<name>.<field> none` clears the DB row instead of leaving
    // a stale value to be rehydrated on next load.
    let mut prefs = mold_db::ModelPrefs::load(db, &canonical)?.unwrap_or_default();
    prefs.width = mc.default_width;
    prefs.height = mc.default_height;
    prefs.steps = mc.default_steps;
    prefs.guidance = mc.default_guidance;
    prefs.scheduler = mc.scheduler.map(|s| s.to_string());
    prefs.lora_path = mc.lora.clone();
    prefs.lora_scale = mc.lora_scale;
    prefs.last_negative = mc.negative_prompt.clone();
    prefs.save(db, &canonical)?;
    Ok(())
}

/// `mold config where <key>` — print which surface owns the key so users
/// can disambiguate without reading source. Answers: "did my `mold config
/// set expand.enabled true` go to config.toml, mold.db, or is an env var
/// overriding it at runtime?"
pub fn run_where(key: &str) -> Result<()> {
    // Validate that this is a known key.
    if find_static_key(key).is_none() && !key.starts_with("models.") {
        eprintln!("{} {}", theme::icon_fail(), unknown_key_error(key));
        return Err(AlreadyReported.into());
    }

    let surface = if key.starts_with("models.") {
        if let Err(e) = parse_model_key(key) {
            eprintln!("{} {e}", theme::icon_fail());
            return Err(AlreadyReported.into());
        }
        model_field_surface(key).unwrap_or(Surface::File)
    } else {
        surface_for_key(key)
    };

    println!("{} {} = {}", theme::icon_ok(), key.bold(), surface.as_str());

    // Callout for env-var override at runtime — even DB/TOML values lose
    // to these, so it matters for disambiguation.
    if let Some((var, env_val)) = env_override_for(key) {
        println!(
            "  {} env {} = {} overrides at runtime",
            theme::icon_bullet(),
            var.dimmed(),
            env_val.dimmed()
        );
    }

    Ok(())
}

pub fn run_path() -> Result<()> {
    match Config::config_path() {
        Some(path) => {
            let exists = path.exists();
            if exists {
                println!("{} {}", theme::icon_ok(), path.display());
            } else {
                println!(
                    "{} {} (not yet created)",
                    theme::icon_info(),
                    path.display()
                );
            }
        }
        None => {
            eprintln!(
                "{} Cannot determine config path (HOME not set)",
                theme::icon_fail()
            );
            return Err(AlreadyReported.into());
        }
    }
    Ok(())
}

pub fn run_edit() -> Result<()> {
    let path = Config::config_path()
        .ok_or_else(|| anyhow!("cannot determine config path (HOME not set)"))?;

    // Create default config if it doesn't exist
    if !path.exists() {
        let config = Config::load_or_default();
        config.save()?;
        println!(
            "{} Created default config at {}",
            theme::icon_info(),
            path.display()
        );
    }

    let editor = std::env::var("EDITOR")
        .or_else(|_| std::env::var("VISUAL"))
        .unwrap_or_else(|_| "vi".to_string());

    // Split to handle EDITOR="code --wait" without shell injection risk
    let parts: Vec<&str> = editor.split_whitespace().collect();
    let (prog, args) = parts
        .split_first()
        .ok_or_else(|| anyhow!("EDITOR is empty"))?;
    let status = std::process::Command::new(prog)
        .args(args)
        .arg(&path)
        .status()?;

    if !status.success() {
        eprintln!("{} Editor exited with {}", theme::icon_fail(), status);
        return Err(AlreadyReported.into());
    }

    Ok(())
}

// ── Shell completion ────────────────────────────────────────────────

/// Return completion candidates for config key names.
pub fn complete_config_key() -> Vec<CompletionCandidate> {
    let mut candidates: Vec<CompletionCandidate> = ALL_KEYS
        .iter()
        .map(|k| CompletionCandidate::new(k.key))
        .collect();

    // Add per-model keys from the current config
    if let Some(config) = Config::config_path().and_then(|p| {
        if p.exists() {
            Some(Config::load_or_default())
        } else {
            None
        }
    }) {
        for model_name in config.models.keys() {
            for (field, _) in MODEL_FIELDS {
                candidates.push(CompletionCandidate::new(format!(
                    "models.{model_name}.{field}"
                )));
            }
        }
    }

    candidates
}

// ── Tests ───────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_support::ENV_LOCK;
    use std::collections::HashMap;

    fn test_config() -> Config {
        Config {
            config_version: 1,
            default_model: "flux2-klein".to_string(),
            models_dir: "~/.mold/models".to_string(),
            server_port: 7680,
            default_width: 768,
            default_height: 768,
            default_steps: 4,
            embed_metadata: true,
            t5_variant: None,
            qwen3_variant: None,
            output_dir: None,
            default_negative_prompt: None,
            expand: mold_core::ExpandSettings::default(),
            logging: mold_core::LoggingConfig::default(),
            runpod: mold_core::runpod::RunPodSettings::default(),
            gpus: None,
            queue_size: None,
            models: HashMap::new(),
        }
    }

    fn test_config_with_model() -> Config {
        let mut config = test_config();
        config.models.insert(
            "test-model:q8".to_string(),
            mold_core::config::ModelConfig {
                default_steps: Some(20),
                default_guidance: Some(3.5),
                default_width: Some(1024),
                default_height: Some(1024),
                scheduler: Some(Scheduler::EulerAncestral),
                negative_prompt: Some("blurry".into()),
                lora: Some("/path/to/lora.safetensors".into()),
                lora_scale: Some(0.8),
                ..Default::default()
            },
        );
        config
    }

    // ── Key lookup tests ────────────────────────────────

    #[test]
    fn known_static_key_resolves() {
        for info in ALL_KEYS {
            assert!(
                find_static_key(info.key).is_some(),
                "key '{}' not found",
                info.key
            );
        }
    }

    #[test]
    fn unknown_key_errors() {
        let config = test_config();
        assert!(get_value(&config, "foo.bar").is_err());
        assert!(get_value(&config, "nonexistent").is_err());
    }

    #[test]
    fn model_key_parses_correctly() {
        let (model, field, _) = parse_model_key("models.flux-dev:q4.default_steps").unwrap();
        assert_eq!(model, "flux-dev:q4");
        assert_eq!(field, "default_steps");
    }

    #[test]
    fn model_key_unknown_field_errors() {
        assert!(parse_model_key("models.flux-dev:q4.bogus_field").is_err());
    }

    #[test]
    fn all_keys_count() {
        // 11 General + 8 Expand + 4 Logging + 8 RunPod = 31 static keys
        assert_eq!(ALL_KEYS.len(), 31);
    }

    #[test]
    fn runpod_keys_registered() {
        for key in [
            "runpod.api_key",
            "runpod.default_gpu",
            "runpod.default_datacenter",
            "runpod.default_network_volume_id",
            "runpod.auto_teardown",
            "runpod.auto_teardown_idle_mins",
            "runpod.cost_alert_usd",
            "runpod.endpoint",
        ] {
            assert!(
                find_static_key(key).is_some(),
                "key {key} not registered in ALL_KEYS",
            );
        }
    }

    #[test]
    fn runpod_api_key_roundtrip() {
        let mut config = test_config();
        set_value(&mut config, "runpod.api_key", "my-secret-key").unwrap();
        assert_eq!(config.runpod.api_key.as_deref(), Some("my-secret-key"));
        // Getter redacts the actual value.
        let v = get_value(&config, "runpod.api_key").unwrap();
        assert_eq!(v.raw(), "<set>");
        // "none" clears it.
        set_value(&mut config, "runpod.api_key", "none").unwrap();
        assert!(config.runpod.api_key.is_none());
    }

    #[test]
    fn runpod_auto_teardown_bool() {
        let mut config = test_config();
        set_value(&mut config, "runpod.auto_teardown", "true").unwrap();
        assert!(config.runpod.auto_teardown);
        set_value(&mut config, "runpod.auto_teardown", "off").unwrap();
        assert!(!config.runpod.auto_teardown);
    }

    #[test]
    fn runpod_cost_alert_bounds() {
        let mut config = test_config();
        set_value(&mut config, "runpod.cost_alert_usd", "2.5").unwrap();
        assert_eq!(config.runpod.cost_alert_usd, 2.5);
        assert!(set_value(&mut config, "runpod.cost_alert_usd", "-1").is_err());
    }

    // ── Value get tests ─────────────────────────────────

    #[test]
    fn get_default_model() {
        let config = test_config();
        let val = get_value(&config, "default_model").unwrap();
        assert_eq!(val.raw(), "flux2-klein");
    }

    #[test]
    fn get_server_port() {
        let config = test_config();
        let val = get_value(&config, "server_port").unwrap();
        assert_eq!(val.raw(), "7680");
    }

    #[test]
    fn get_embed_metadata() {
        let config = test_config();
        let val = get_value(&config, "embed_metadata").unwrap();
        assert_eq!(val.raw(), "true");
    }

    #[test]
    fn get_output_dir_unset() {
        let config = test_config();
        let val = get_value(&config, "output_dir").unwrap();
        assert_eq!(val.display(), "(not set)");
        assert_eq!(val.raw(), "");
    }

    #[test]
    fn get_output_dir_set() {
        let mut config = test_config();
        config.output_dir = Some("/tmp/output".into());
        let val = get_value(&config, "output_dir").unwrap();
        assert_eq!(val.raw(), "/tmp/output");
    }

    #[test]
    fn get_expand_temperature() {
        let config = test_config();
        let val = get_value(&config, "expand.temperature").unwrap();
        assert_eq!(val.raw(), "0.7");
    }

    #[test]
    fn get_expand_top_p() {
        let config = test_config();
        let val = get_value(&config, "expand.top_p").unwrap();
        assert_eq!(val.raw(), "0.9");
    }

    #[test]
    fn get_logging_level() {
        let config = test_config();
        let val = get_value(&config, "logging.level").unwrap();
        assert_eq!(val.raw(), "info");
    }

    #[test]
    fn get_logging_max_days() {
        let config = test_config();
        let val = get_value(&config, "logging.max_days").unwrap();
        assert_eq!(val.raw(), "7");
    }

    #[test]
    fn get_t5_variant_default() {
        let config = test_config();
        let val = get_value(&config, "t5_variant").unwrap();
        assert_eq!(val.raw(), "auto");
    }

    #[test]
    fn get_model_field() {
        let config = test_config_with_model();
        let val = get_value(&config, "models.test-model:q8.default_steps").unwrap();
        assert_eq!(val.raw(), "20");
    }

    #[test]
    fn get_model_field_guidance() {
        let config = test_config_with_model();
        let val = get_value(&config, "models.test-model:q8.default_guidance").unwrap();
        assert_eq!(val.raw(), "3.5");
    }

    #[test]
    fn get_model_field_scheduler() {
        let config = test_config_with_model();
        let val = get_value(&config, "models.test-model:q8.scheduler").unwrap();
        assert_eq!(val.raw(), "euler-ancestral");
    }

    #[test]
    fn get_model_field_lora() {
        let config = test_config_with_model();
        let val = get_value(&config, "models.test-model:q8.lora").unwrap();
        assert_eq!(val.raw(), "/path/to/lora.safetensors");
    }

    #[test]
    fn get_model_field_lora_scale() {
        let config = test_config_with_model();
        let val = get_value(&config, "models.test-model:q8.lora_scale").unwrap();
        assert_eq!(val.raw(), "0.8");
    }

    #[test]
    fn get_model_field_missing_model() {
        let config = test_config();
        assert!(get_value(&config, "models.nonexistent:q8.default_steps").is_err());
    }

    // ── Value set tests ─────────────────────────────────

    #[test]
    fn set_string_field() {
        let mut config = test_config();
        set_value(&mut config, "default_model", "sd15:fp16").unwrap();
        assert_eq!(config.default_model, "sd15:fp16");
    }

    #[test]
    fn set_bool_true() {
        let mut config = test_config();
        config.expand.enabled = false;
        set_value(&mut config, "expand.enabled", "true").unwrap();
        assert!(config.expand.enabled);
    }

    #[test]
    fn set_bool_false() {
        let mut config = test_config();
        set_value(&mut config, "embed_metadata", "false").unwrap();
        assert!(!config.embed_metadata);
    }

    #[test]
    fn set_bool_on() {
        let mut config = test_config();
        config.expand.enabled = false;
        set_value(&mut config, "expand.enabled", "on").unwrap();
        assert!(config.expand.enabled);
    }

    #[test]
    fn set_bool_off() {
        let mut config = test_config();
        set_value(&mut config, "embed_metadata", "off").unwrap();
        assert!(!config.embed_metadata);
    }

    #[test]
    fn set_bool_invalid() {
        let mut config = test_config();
        assert!(set_value(&mut config, "embed_metadata", "maybe").is_err());
    }

    #[test]
    fn set_u16_valid() {
        let mut config = test_config();
        set_value(&mut config, "server_port", "8080").unwrap();
        assert_eq!(config.server_port, 8080);
    }

    #[test]
    fn set_u16_out_of_range() {
        let mut config = test_config();
        assert!(set_value(&mut config, "server_port", "0").is_err());
        assert!(set_value(&mut config, "server_port", "99999").is_err());
    }

    #[test]
    fn set_u32_valid() {
        let mut config = test_config();
        set_value(&mut config, "default_steps", "50").unwrap();
        assert_eq!(config.default_steps, 50);
    }

    #[test]
    fn set_u32_out_of_range() {
        let mut config = test_config();
        assert!(set_value(&mut config, "default_steps", "0").is_err());
        assert!(set_value(&mut config, "default_steps", "9999").is_err());
    }

    #[test]
    fn set_f64_valid() {
        let mut config = test_config();
        set_value(&mut config, "expand.temperature", "1.5").unwrap();
        assert!((config.expand.temperature - 1.5).abs() < 0.001);
    }

    #[test]
    fn set_f64_out_of_range() {
        let mut config = test_config();
        assert!(set_value(&mut config, "expand.temperature", "3.0").is_err());
        assert!(set_value(&mut config, "expand.temperature", "-1.0").is_err());
    }

    #[test]
    fn set_optional_none_clears() {
        let mut config = test_config();
        config.output_dir = Some("/tmp".into());
        set_value(&mut config, "output_dir", "none").unwrap();
        assert!(config.output_dir.is_none());
    }

    #[test]
    fn set_optional_value() {
        let mut config = test_config();
        set_value(&mut config, "output_dir", "/tmp/images").unwrap();
        assert_eq!(config.output_dir, Some("/tmp/images".into()));
    }

    #[test]
    fn set_model_field_existing() {
        let mut config = test_config_with_model();
        set_value(&mut config, "models.test-model:q8.default_steps", "30").unwrap();
        assert_eq!(
            config.models.get("test-model:q8").unwrap().default_steps,
            Some(30)
        );
    }

    #[test]
    fn set_model_field_creates_entry() {
        let mut config = test_config();
        assert!(!config.models.contains_key("new-model:q4"));
        set_value(&mut config, "models.new-model:q4.default_steps", "25").unwrap();
        assert_eq!(
            config.models.get("new-model:q4").unwrap().default_steps,
            Some(25)
        );
    }

    #[test]
    fn set_model_scheduler() {
        let mut config = test_config_with_model();
        set_value(&mut config, "models.test-model:q8.scheduler", "uni-pc").unwrap();
        assert_eq!(
            config.models.get("test-model:q8").unwrap().scheduler,
            Some(Scheduler::UniPc)
        );
    }

    #[test]
    fn set_model_scheduler_none() {
        let mut config = test_config_with_model();
        set_value(&mut config, "models.test-model:q8.scheduler", "none").unwrap();
        assert!(config
            .models
            .get("test-model:q8")
            .unwrap()
            .scheduler
            .is_none());
    }

    #[test]
    fn set_model_scheduler_invalid() {
        let mut config = test_config_with_model();
        assert!(set_value(&mut config, "models.test-model:q8.scheduler", "invalid").is_err());
    }

    #[test]
    fn set_model_lora_scale() {
        let mut config = test_config_with_model();
        set_value(&mut config, "models.test-model:q8.lora_scale", "1.5").unwrap();
        assert!(
            (config
                .models
                .get("test-model:q8")
                .unwrap()
                .lora_scale
                .unwrap()
                - 1.5)
                .abs()
                < 0.001
        );
    }

    // ── Validation tests ────────────────────────────────

    #[test]
    fn validate_port_range() {
        assert!(parse_u16("1", 1, 65535, "port").is_ok());
        assert!(parse_u16("65535", 1, 65535, "port").is_ok());
        assert!(parse_u16("0", 1, 65535, "port").is_err());
    }

    #[test]
    fn validate_bool_values() {
        assert!(parse_bool("true", "k").is_ok());
        assert!(parse_bool("false", "k").is_ok());
        assert!(parse_bool("on", "k").is_ok());
        assert!(parse_bool("off", "k").is_ok());
        assert!(parse_bool("1", "k").is_ok());
        assert!(parse_bool("0", "k").is_ok());
        assert!(parse_bool("yes", "k").is_ok());
        assert!(parse_bool("no", "k").is_ok());
        assert!(parse_bool("maybe", "k").is_err());
    }

    #[test]
    fn validate_log_level_enum() {
        let mut config = test_config();
        set_value(&mut config, "logging.level", "debug").unwrap();
        assert_eq!(config.logging.level, "debug");
        assert!(set_value(&mut config, "logging.level", "verbose").is_err());
    }

    #[test]
    fn validate_t5_variant_enum() {
        let mut config = test_config();
        set_value(&mut config, "t5_variant", "q8").unwrap();
        assert_eq!(config.t5_variant, Some("q8".into()));
        set_value(&mut config, "t5_variant", "auto").unwrap();
        assert!(config.t5_variant.is_none()); // "auto" = None
        assert!(set_value(&mut config, "t5_variant", "invalid").is_err());
    }

    #[test]
    fn validate_qwen3_variant_enum() {
        let mut config = test_config();
        set_value(&mut config, "qwen3_variant", "bf16").unwrap();
        assert_eq!(config.qwen3_variant, Some("bf16".into()));
        assert!(set_value(&mut config, "qwen3_variant", "fp16").is_err()); // not valid for qwen3
    }

    #[test]
    fn validate_temperature_range() {
        assert!(parse_f64("0.0", 0.0, 2.0, "t").is_ok());
        assert!(parse_f64("2.0", 0.0, 2.0, "t").is_ok());
        assert!(parse_f64("2.1", 0.0, 2.0, "t").is_err());
        assert!(parse_f64("-0.1", 0.0, 2.0, "t").is_err());
    }

    // ── Env override tests ──────────────────────────────

    #[test]
    fn env_override_absent() {
        // Most env vars won't be set in test environment
        assert!(env_override_for("server_port").is_none()); // no env var mapped
        assert!(env_override_for("logging.level").is_none());
    }

    // ── Completion tests ────────────────────────────────

    #[test]
    fn complete_returns_candidates() {
        let candidates = complete_config_key();
        assert!(!candidates.is_empty());
    }

    #[test]
    fn complete_includes_static_keys() {
        let candidates = complete_config_key();
        let keys: Vec<String> = candidates
            .iter()
            .map(|c| c.get_value().to_string_lossy().to_string())
            .collect();
        assert!(keys.contains(&"expand.backend".to_string()));
        assert!(keys.contains(&"server_port".to_string()));
        assert!(keys.contains(&"logging.level".to_string()));
    }

    // ── ConfigValue formatting tests ────────────────────

    #[test]
    fn config_value_display_none() {
        assert_eq!(ConfigValue::None.display(), "(not set)");
        assert_eq!(ConfigValue::None.raw(), "");
    }

    // ── Surface classification tests ────────────────────

    #[test]
    fn surface_for_tui_and_expand_and_generate_is_db() {
        assert_eq!(surface_for_key("tui.theme"), Surface::Db);
        assert_eq!(surface_for_key("expand.enabled"), Surface::Db);
        assert_eq!(surface_for_key("expand.backend"), Surface::Db);
        assert_eq!(surface_for_key("generate.default_width"), Surface::Db);
        assert_eq!(
            surface_for_key("model_prefs.flux-dev:q4.width"),
            Surface::Db
        );
    }

    #[test]
    fn surface_for_flat_generation_globals_is_db() {
        assert_eq!(surface_for_key("default_width"), Surface::Db);
        assert_eq!(surface_for_key("default_height"), Surface::Db);
        assert_eq!(surface_for_key("default_steps"), Surface::Db);
        assert_eq!(surface_for_key("embed_metadata"), Surface::Db);
        assert_eq!(surface_for_key("default_negative_prompt"), Surface::Db);
        assert_eq!(surface_for_key("t5_variant"), Surface::Db);
        assert_eq!(surface_for_key("qwen3_variant"), Surface::Db);
    }

    #[test]
    fn surface_for_bootstrap_keys_is_file() {
        assert_eq!(surface_for_key("default_model"), Surface::File);
        assert_eq!(surface_for_key("models_dir"), Surface::File);
        assert_eq!(surface_for_key("output_dir"), Surface::File);
        assert_eq!(surface_for_key("server_port"), Surface::File);
        assert_eq!(surface_for_key("logging.level"), Surface::File);
        assert_eq!(surface_for_key("runpod.api_key"), Surface::File);
    }

    #[test]
    fn model_field_surface_splits_db_vs_path() {
        // User preference fields → DB.
        for field in [
            "default_steps",
            "default_guidance",
            "default_width",
            "default_height",
            "scheduler",
            "negative_prompt",
            "lora",
            "lora_scale",
        ] {
            let key = format!("models.flux-dev:q4.{field}");
            assert_eq!(model_field_surface(&key), Some(Surface::Db), "key={key}");
        }
        // Path fields stay in TOML.
        assert_eq!(
            model_field_surface("models.flux-dev:q4.transformer"),
            Some(Surface::File)
        );
        // Non-model keys return None.
        assert_eq!(model_field_surface("expand.enabled"), None);
    }

    #[test]
    fn config_value_json_types() {
        assert_eq!(ConfigValue::Bool(true).to_json(), serde_json::json!(true));
        assert_eq!(ConfigValue::U32(42).to_json(), serde_json::json!(42));
        assert_eq!(
            ConfigValue::String("hello".into()).to_json(),
            serde_json::json!("hello")
        );
        assert!(ConfigValue::None.to_json().is_null());
    }

    // ── Integration tests ───────────────────────────────

    /// Codex P2 regression: clearing a per-model negative prompt via
    /// `mold config set models.<name>.negative_prompt none` must null
    /// out the stored `last_negative` row, not leave it untouched.
    /// Exercises the unconditional assignment in
    /// `persist_model_field_to_db`.
    #[test]
    fn model_config_sync_clear_nulls_last_negative() {
        // Pre-seed prefs with a negative prompt, then simulate the
        // "user cleared it via CLI" path by mutating the in-memory
        // ModelConfig and re-snapshotting.
        let db = mold_db::MetadataDb::open_in_memory().unwrap();
        mold_db::ModelPrefs {
            last_negative: Some("stale".into()),
            width: Some(1024),
            ..Default::default()
        }
        .save(&db, "flux-dev:q4")
        .unwrap();

        // Mimic persist_model_field_to_db's load-merge-save with a
        // cleared negative_prompt on ModelConfig.
        let mc = mold_core::config::ModelConfig {
            negative_prompt: None,
            default_width: Some(1024),
            ..Default::default()
        };
        let mut prefs = mold_db::ModelPrefs::load(&db, "flux-dev:q4")
            .unwrap()
            .unwrap_or_default();
        prefs.width = mc.default_width;
        prefs.last_negative = mc.negative_prompt.clone();
        prefs.save(&db, "flux-dev:q4").unwrap();

        let loaded = mold_db::ModelPrefs::load(&db, "flux-dev:q4")
            .unwrap()
            .unwrap();
        assert_eq!(loaded.last_negative, None, "clear must null the row");
        assert_eq!(loaded.width, Some(1024), "unrelated fields preserved");
    }

    #[test]
    fn set_persists_to_disk() {
        let _lock = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        let tmp = std::env::temp_dir().join(format!(
            "mold-config-test-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        std::fs::create_dir_all(&tmp).unwrap();

        let prev_home = std::env::var("MOLD_HOME").ok();
        unsafe { std::env::set_var("MOLD_HOME", &tmp) };

        let mut config = Config::load_or_default();
        set_value(&mut config, "server_port", "9999").unwrap();
        config.save().unwrap();

        let config_path = tmp.join("config.toml");
        let written = std::fs::read_to_string(&config_path).unwrap();
        assert!(written.contains("server_port = 9999"));

        // Cleanup
        match prev_home {
            Some(v) => unsafe { std::env::set_var("MOLD_HOME", v) },
            None => unsafe { std::env::remove_var("MOLD_HOME") },
        }
        let _ = std::fs::remove_dir_all(&tmp);
    }

    // ── Full round-trip for every global field ──────────

    #[test]
    fn roundtrip_all_global_fields() {
        let mut config = test_config();

        // String fields
        set_value(&mut config, "default_model", "sd15:fp16").unwrap();
        assert_eq!(
            get_value(&config, "default_model").unwrap().raw(),
            "sd15:fp16"
        );

        set_value(&mut config, "models_dir", "/new/path").unwrap();
        assert_eq!(get_value(&config, "models_dir").unwrap().raw(), "/new/path");

        // Optional string
        set_value(&mut config, "output_dir", "/out").unwrap();
        assert_eq!(get_value(&config, "output_dir").unwrap().raw(), "/out");
        set_value(&mut config, "output_dir", "none").unwrap();
        assert_eq!(get_value(&config, "output_dir").unwrap().raw(), "");

        set_value(&mut config, "default_negative_prompt", "ugly").unwrap();
        assert_eq!(
            get_value(&config, "default_negative_prompt").unwrap().raw(),
            "ugly"
        );

        // Numeric
        set_value(&mut config, "server_port", "9090").unwrap();
        assert_eq!(get_value(&config, "server_port").unwrap().raw(), "9090");

        set_value(&mut config, "default_width", "1024").unwrap();
        assert_eq!(get_value(&config, "default_width").unwrap().raw(), "1024");

        set_value(&mut config, "default_height", "512").unwrap();
        assert_eq!(get_value(&config, "default_height").unwrap().raw(), "512");

        set_value(&mut config, "default_steps", "50").unwrap();
        assert_eq!(get_value(&config, "default_steps").unwrap().raw(), "50");

        // Bool
        set_value(&mut config, "embed_metadata", "false").unwrap();
        assert_eq!(get_value(&config, "embed_metadata").unwrap().raw(), "false");

        // Variant enums
        set_value(&mut config, "t5_variant", "q4").unwrap();
        assert_eq!(get_value(&config, "t5_variant").unwrap().raw(), "q4");

        set_value(&mut config, "qwen3_variant", "q8").unwrap();
        assert_eq!(get_value(&config, "qwen3_variant").unwrap().raw(), "q8");
    }

    #[test]
    fn roundtrip_all_expand_fields() {
        let mut config = test_config();

        set_value(&mut config, "expand.enabled", "true").unwrap();
        assert_eq!(get_value(&config, "expand.enabled").unwrap().raw(), "true");

        set_value(&mut config, "expand.backend", "http://localhost:11434").unwrap();
        assert_eq!(
            get_value(&config, "expand.backend").unwrap().raw(),
            "http://localhost:11434"
        );

        set_value(&mut config, "expand.model", "qwen3:q4").unwrap();
        assert_eq!(
            get_value(&config, "expand.model").unwrap().raw(),
            "qwen3:q4"
        );

        set_value(&mut config, "expand.api_model", "gpt-4o").unwrap();
        assert_eq!(
            get_value(&config, "expand.api_model").unwrap().raw(),
            "gpt-4o"
        );

        set_value(&mut config, "expand.temperature", "0.5").unwrap();
        assert_eq!(
            get_value(&config, "expand.temperature").unwrap().raw(),
            "0.5"
        );

        set_value(&mut config, "expand.top_p", "0.8").unwrap();
        assert_eq!(get_value(&config, "expand.top_p").unwrap().raw(), "0.8");

        set_value(&mut config, "expand.max_tokens", "500").unwrap();
        assert_eq!(
            get_value(&config, "expand.max_tokens").unwrap().raw(),
            "500"
        );

        set_value(&mut config, "expand.thinking", "true").unwrap();
        assert_eq!(get_value(&config, "expand.thinking").unwrap().raw(), "true");
    }

    #[test]
    fn roundtrip_all_logging_fields() {
        let mut config = test_config();

        set_value(&mut config, "logging.level", "debug").unwrap();
        assert_eq!(get_value(&config, "logging.level").unwrap().raw(), "debug");

        set_value(&mut config, "logging.file", "true").unwrap();
        assert_eq!(get_value(&config, "logging.file").unwrap().raw(), "true");

        set_value(&mut config, "logging.dir", "/var/log/mold").unwrap();
        assert_eq!(
            get_value(&config, "logging.dir").unwrap().raw(),
            "/var/log/mold"
        );

        set_value(&mut config, "logging.max_days", "30").unwrap();
        assert_eq!(get_value(&config, "logging.max_days").unwrap().raw(), "30");
    }

    #[test]
    fn roundtrip_all_model_fields() {
        let mut config = test_config_with_model();
        let prefix = "models.test-model:q8";

        set_value(&mut config, &format!("{prefix}.default_steps"), "30").unwrap();
        assert_eq!(
            get_value(&config, &format!("{prefix}.default_steps"))
                .unwrap()
                .raw(),
            "30"
        );

        set_value(&mut config, &format!("{prefix}.default_guidance"), "7.5").unwrap();
        assert_eq!(
            get_value(&config, &format!("{prefix}.default_guidance"))
                .unwrap()
                .raw(),
            "7.5"
        );

        set_value(&mut config, &format!("{prefix}.default_width"), "512").unwrap();
        assert_eq!(
            get_value(&config, &format!("{prefix}.default_width"))
                .unwrap()
                .raw(),
            "512"
        );

        set_value(&mut config, &format!("{prefix}.default_height"), "512").unwrap();
        assert_eq!(
            get_value(&config, &format!("{prefix}.default_height"))
                .unwrap()
                .raw(),
            "512"
        );

        set_value(&mut config, &format!("{prefix}.scheduler"), "ddim").unwrap();
        assert_eq!(
            get_value(&config, &format!("{prefix}.scheduler"))
                .unwrap()
                .raw(),
            "ddim"
        );

        set_value(
            &mut config,
            &format!("{prefix}.negative_prompt"),
            "bad quality",
        )
        .unwrap();
        assert_eq!(
            get_value(&config, &format!("{prefix}.negative_prompt"))
                .unwrap()
                .raw(),
            "bad quality"
        );

        set_value(
            &mut config,
            &format!("{prefix}.lora"),
            "/new/lora.safetensors",
        )
        .unwrap();
        assert_eq!(
            get_value(&config, &format!("{prefix}.lora")).unwrap().raw(),
            "/new/lora.safetensors"
        );

        set_value(&mut config, &format!("{prefix}.lora_scale"), "1.2").unwrap();
        assert_eq!(
            get_value(&config, &format!("{prefix}.lora_scale"))
                .unwrap()
                .raw(),
            "1.2"
        );
    }
}
