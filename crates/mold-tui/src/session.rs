//! TUI session state — persisted via the SQLite metadata DB (`mold-db`).
//!
//! `TuiSession` is a DTO. Its public shape is what the app's call sites
//! see; the underlying storage moved from `~/.mold/tui-session.json` to
//! the `settings` + `model_prefs` tables. The legacy JSON file is
//! imported once at startup (see [`import_legacy_json_once`]) and then
//! renamed to `tui-session.json.migrated` so a downgrade still has a
//! recoverable copy for one release.

use std::path::{Path, PathBuf};

use mold_db::{settings as keys, MetadataDb, ModelPrefs, Settings};
use serde::{Deserialize, Serialize};

/// Persisted TUI session state — restored on next launch.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct TuiSession {
    #[serde(default)]
    pub last_prompt: String,
    #[serde(default)]
    pub last_negative: String,
    #[serde(default)]
    pub last_model: String,
    // Generation parameters
    #[serde(default)]
    pub width: Option<u32>,
    #[serde(default)]
    pub height: Option<u32>,
    #[serde(default)]
    pub steps: Option<u32>,
    #[serde(default)]
    pub guidance: Option<f64>,
    #[serde(default)]
    pub seed_mode: Option<String>,
    #[serde(default)]
    pub batch: Option<u32>,
    #[serde(default)]
    pub format: Option<String>,
    #[serde(default)]
    pub scheduler: Option<String>,
    // Advanced
    #[serde(default)]
    pub lora_path: Option<String>,
    #[serde(default)]
    pub lora_scale: Option<f64>,
    #[serde(default)]
    pub expand: Option<bool>,
    #[serde(default)]
    pub offload: Option<bool>,
    // img2img
    #[serde(default)]
    pub strength: Option<f64>,
    #[serde(default)]
    pub control_scale: Option<f64>,
    /// Theme preset slug (e.g. "mocha", "latte"). Missing = Mocha.
    #[serde(default)]
    pub theme: Option<String>,
    /// Whether the Negative prompt panel was collapsed at exit. Missing = false.
    #[serde(default)]
    pub negative_collapsed: Option<bool>,
}

fn legacy_session_path() -> Option<PathBuf> {
    mold_core::Config::mold_dir().map(|d| d.join("tui-session.json"))
}

fn open_db() -> Option<MetadataDb> {
    match mold_db::open_default() {
        Ok(Some(db)) => Some(db),
        Ok(None) => None,
        Err(e) => {
            tracing::warn!(error = %e, "tui session: metadata DB open failed; falling back to in-memory");
            None
        }
    }
}

impl TuiSession {
    /// Load session from the metadata DB. Returns default if the DB is
    /// disabled, unreachable, or empty — the TUI still boots in those
    /// cases but won't persist across restarts.
    pub fn load() -> Self {
        // One-shot migration: if the legacy JSON file is still sitting on
        // disk and hasn't been imported yet, slurp it into the DB before
        // we read. Idempotent (sentinel key guards the second pass).
        if let Some(db) = open_db() {
            import_legacy_json_once(&db);
            return load_from_db(&db);
        }

        // DB unavailable — return default. The TUI still functions; the
        // cost is that settings don't persist across restarts. Legacy
        // JSON is not consulted when the DB is disabled, because users
        // who opt out (`MOLD_DB_DISABLE=1`) have explicitly said "don't
        // write files from me".
        Self::default()
    }

    /// Persist the session into the DB. Writes the global-ish fields to
    /// the `settings` table and the per-model generation parameters to
    /// the `model_prefs` table under [`Self::last_model`]. Silent no-op
    /// when the DB is unavailable.
    pub fn save(&self) {
        let Some(db) = open_db() else {
            return;
        };
        save_to_db(&db, self);
    }

    /// Check if session has any meaningful content.
    pub fn has_prompt(&self) -> bool {
        !self.last_prompt.is_empty()
    }

    /// Build a session from current app state.
    pub fn from_params(prompt: &str, negative: &str, params: &super::app::GenerateParams) -> Self {
        Self {
            last_prompt: prompt.to_string(),
            last_negative: negative.to_string(),
            last_model: params.model.clone(),
            width: Some(params.width),
            height: Some(params.height),
            steps: Some(params.steps),
            guidance: Some(params.guidance),
            seed_mode: Some(params.seed_mode.label().to_string()),
            batch: Some(params.batch),
            format: Some(format!("{:?}", params.format).to_lowercase()),
            scheduler: params.scheduler.map(|s| format!("{s:?}").to_lowercase()),
            lora_path: params.lora_path.clone(),
            lora_scale: Some(params.lora_scale),
            expand: Some(params.expand),
            offload: Some(params.offload),
            strength: Some(params.strength),
            control_scale: Some(params.control_scale),
            theme: None,
            negative_collapsed: None,
        }
    }

    /// Attach a theme slug for persistence. Chainable so call sites can append
    /// to `from_params` without adding a positional argument.
    pub fn with_theme(mut self, preset: super::ui::theme::ThemePreset) -> Self {
        self.theme = Some(preset.slug().to_string());
        self
    }

    /// Record whether the negative-prompt panel was collapsed at save time.
    pub fn with_negative_collapsed(mut self, collapsed: bool) -> Self {
        self.negative_collapsed = Some(collapsed);
        self
    }

    /// Apply saved settings to params (keeps model as-is, caller handles model).
    pub fn apply_to_params(&self, params: &mut super::app::GenerateParams) {
        if let Some(w) = self.width {
            params.width = w;
        }
        if let Some(h) = self.height {
            params.height = h;
        }
        if let Some(s) = self.steps {
            params.steps = s;
        }
        if let Some(g) = self.guidance {
            params.guidance = g;
        }
        if let Some(ref sm) = self.seed_mode {
            params.seed_mode = match sm.as_str() {
                "fixed" => super::app::SeedMode::Fixed,
                "increment" => super::app::SeedMode::Increment,
                _ => super::app::SeedMode::Random,
            };
        }
        if let Some(b) = self.batch {
            params.batch = b;
        }
        if let Some(ref f) = self.format {
            params.format = match f.as_str() {
                "jpeg" => mold_core::OutputFormat::Jpeg,
                _ => mold_core::OutputFormat::Png,
            };
        }
        if let Some(ref s) = self.scheduler {
            params.scheduler = match s.as_str() {
                "ddim" => Some(mold_core::Scheduler::Ddim),
                "eulerancestral" => Some(mold_core::Scheduler::EulerAncestral),
                "unipc" => Some(mold_core::Scheduler::UniPc),
                _ => None,
            };
        }
        if let Some(ref lp) = self.lora_path {
            params.lora_path = Some(lp.clone());
        }
        if let Some(ls) = self.lora_scale {
            params.lora_scale = ls;
        }
        if let Some(e) = self.expand {
            params.expand = e;
        }
        if let Some(o) = self.offload {
            params.offload = o;
        }
        if let Some(s) = self.strength {
            params.strength = s;
        }
        if let Some(cs) = self.control_scale {
            params.control_scale = cs;
        }
    }

    /// Apply only non-model-specific settings. Skips width, height, steps,
    /// guidance, and scheduler since those belong to the saved model and would
    /// be wrong for a different default model.
    pub fn apply_non_model_params(&self, params: &mut super::app::GenerateParams) {
        if let Some(ref sm) = self.seed_mode {
            params.seed_mode = match sm.as_str() {
                "fixed" => super::app::SeedMode::Fixed,
                "increment" => super::app::SeedMode::Increment,
                _ => super::app::SeedMode::Random,
            };
        }
        if let Some(b) = self.batch {
            params.batch = b;
        }
        if let Some(ref f) = self.format {
            params.format = match f.as_str() {
                "jpeg" => mold_core::OutputFormat::Jpeg,
                _ => mold_core::OutputFormat::Png,
            };
        }
        if let Some(ref lp) = self.lora_path {
            params.lora_path = Some(lp.clone());
        }
        if let Some(ls) = self.lora_scale {
            params.lora_scale = ls;
        }
        if let Some(e) = self.expand {
            params.expand = e;
        }
        if let Some(o) = self.offload {
            params.offload = o;
        }
        if let Some(s) = self.strength {
            params.strength = s;
        }
        if let Some(cs) = self.control_scale {
            params.control_scale = cs;
        }
    }
}

// ------------------------------------------------------------------
// Internal helpers — splitting the DB IO keeps the impl blocks focused
// on the DTO surface and makes the migration path testable.
// ------------------------------------------------------------------

fn load_from_db(db: &MetadataDb) -> TuiSession {
    let s = Settings::new(db);
    let last_model = s
        .get_str(keys::TUI_LAST_MODEL)
        .unwrap_or(None)
        .unwrap_or_default();
    let last_prompt = s
        .get_str(keys::TUI_LAST_PROMPT)
        .unwrap_or(None)
        .unwrap_or_default();
    let last_negative = s
        .get_str(keys::TUI_LAST_NEGATIVE)
        .unwrap_or(None)
        .unwrap_or_default();
    let theme = s.get_str(keys::TUI_THEME).unwrap_or(None);
    let negative_collapsed = s.get_bool(keys::TUI_NEGATIVE_COLLAPSED).unwrap_or(None);

    let mut session = TuiSession {
        last_prompt,
        last_negative,
        last_model: last_model.clone(),
        theme,
        negative_collapsed,
        ..Default::default()
    };

    if !last_model.is_empty() {
        if let Ok(Some(prefs)) = ModelPrefs::load(db, &last_model) {
            overlay_prefs(&mut session, &prefs);
        }
    }
    session
}

fn overlay_prefs(session: &mut TuiSession, prefs: &ModelPrefs) {
    session.width = prefs.width.or(session.width);
    session.height = prefs.height.or(session.height);
    session.steps = prefs.steps.or(session.steps);
    session.guidance = prefs.guidance.or(session.guidance);
    session.scheduler = prefs
        .scheduler
        .clone()
        .or_else(|| session.scheduler.clone());
    session.seed_mode = prefs
        .seed_mode
        .clone()
        .or_else(|| session.seed_mode.clone());
    session.batch = prefs.batch.or(session.batch);
    session.format = prefs.format.clone().or_else(|| session.format.clone());
    session.lora_path = prefs
        .lora_path
        .clone()
        .or_else(|| session.lora_path.clone());
    session.lora_scale = prefs.lora_scale.or(session.lora_scale);
    session.expand = prefs.expand.or(session.expand);
    session.offload = prefs.offload.or(session.offload);
    session.strength = prefs.strength.or(session.strength);
    session.control_scale = prefs.control_scale.or(session.control_scale);
}

fn save_to_db(db: &MetadataDb, session: &TuiSession) {
    let s = Settings::new(db);
    if let Err(e) = s.set_str(keys::TUI_LAST_PROMPT, &session.last_prompt) {
        tracing::warn!(error = %e, "tui session: set last_prompt failed");
    }
    if let Err(e) = s.set_str(keys::TUI_LAST_NEGATIVE, &session.last_negative) {
        tracing::warn!(error = %e, "tui session: set last_negative failed");
    }
    if !session.last_model.is_empty() {
        let _ = s.set_str(keys::TUI_LAST_MODEL, &session.last_model);
    }
    if let Some(ref theme) = session.theme {
        let _ = s.set_str(keys::TUI_THEME, theme);
    }
    if let Some(collapsed) = session.negative_collapsed {
        let _ = s.set_bool(keys::TUI_NEGATIVE_COLLAPSED, collapsed);
    }

    // Per-model row. We overwrite the row keyed on `last_model` with all
    // currently-set generation parameters — the caller populates them
    // via `from_params` right before `save()`, so this captures the
    // user's latest choices for the active model.
    if !session.last_model.is_empty() {
        let prefs = ModelPrefs {
            width: session.width,
            height: session.height,
            steps: session.steps,
            guidance: session.guidance,
            scheduler: session.scheduler.clone(),
            seed_mode: session.seed_mode.clone(),
            batch: session.batch,
            format: session.format.clone(),
            lora_path: session.lora_path.clone(),
            lora_scale: session.lora_scale,
            expand: session.expand,
            offload: session.offload,
            strength: session.strength,
            control_scale: session.control_scale,
            frames: None,
            fps: None,
            last_prompt: Some(session.last_prompt.clone()),
            last_negative: Some(session.last_negative.clone()),
        };
        if let Err(e) = prefs.save(db, &session.last_model) {
            tracing::warn!(error = %e, model = %session.last_model, "tui session: model_prefs save failed");
        }
    }
}

/// Run the legacy `tui-session.json` + `prompt-history.jsonl` import
/// into the DB exactly once per install. After a successful import the
/// source files are renamed to `.migrated` — a downgrade still has a
/// recoverable copy for one release, and the sentinel key prevents a
/// second pass from clobbering fresh DB state with stale JSON.
pub(crate) fn import_legacy_json_once(db: &MetadataDb) {
    let settings = Settings::new(db);
    let already = settings
        .get_bool(keys::TUI_MIGRATED_FROM_JSON)
        .unwrap_or(None)
        .unwrap_or(false);
    if already {
        return;
    }

    // Session JSON
    if let Some(path) = legacy_session_path() {
        if path.exists() {
            match std::fs::read_to_string(&path) {
                Ok(contents) => match serde_json::from_str::<TuiSession>(&contents) {
                    Ok(session) => {
                        save_to_db(db, &session);
                        rename_to_migrated(&path);
                        tracing::info!(
                            path = %path.display(),
                            "imported legacy tui-session.json into metadata DB"
                        );
                    }
                    Err(e) => {
                        tracing::warn!(path = %path.display(), error = %e,
                            "tui session: legacy JSON parse failed; leaving file in place");
                    }
                },
                Err(e) => {
                    tracing::warn!(path = %path.display(), error = %e,
                        "tui session: legacy JSON read failed");
                }
            }
        }
    }

    // Prompt history JSONL
    crate::history::import_legacy_jsonl(db);

    let _ = settings.set_bool(keys::TUI_MIGRATED_FROM_JSON, true);
}

fn rename_to_migrated(path: &Path) {
    if let Some(fname) = path.file_name().and_then(|n| n.to_str()) {
        if let Some(parent) = path.parent() {
            let dst = parent.join(format!("{fname}.migrated"));
            if let Err(e) = std::fs::rename(path, &dst) {
                tracing::warn!(src = %path.display(), dst = %dst.display(), error = %e,
                    "rename legacy file to .migrated failed");
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use crate::test_env::with_isolated_env;
    use serial_test::serial;

    #[test]
    fn default_session_has_no_prompt() {
        let session = TuiSession::default();
        assert!(!session.has_prompt());
        assert!(session.last_prompt.is_empty());
        assert!(session.last_model.is_empty());
    }

    #[test]
    fn with_theme_and_with_negative_collapsed_are_chainable() {
        use crate::ui::theme::ThemePreset;
        let params = crate::app::GenerateParams::from_config(&mold_core::Config::default());
        let session = TuiSession::from_params("p", "n", &params)
            .with_theme(ThemePreset::Dracula)
            .with_negative_collapsed(true);
        assert_eq!(session.theme.as_deref(), Some("dracula"));
        assert_eq!(session.negative_collapsed, Some(true));
    }

    #[test]
    #[serial(mold_env)]
    fn save_then_load_roundtrip_through_db() {
        with_isolated_env(|_home| {
            let seed = TuiSession {
                last_prompt: "a cat".into(),
                last_negative: "blurry".into(),
                last_model: "flux-dev:q4".into(),
                width: Some(1024),
                height: Some(1024),
                steps: Some(20),
                guidance: Some(3.5),
                seed_mode: Some("random".into()),
                batch: Some(2),
                format: Some("png".into()),
                scheduler: Some("ddim".into()),
                lora_path: Some("/lora.safetensors".into()),
                lora_scale: Some(0.75),
                expand: Some(true),
                offload: Some(false),
                strength: Some(0.8),
                control_scale: Some(1.0),
                theme: Some("dracula".into()),
                negative_collapsed: Some(true),
            };
            seed.save();
            let loaded = TuiSession::load();
            assert_eq!(loaded.last_prompt, "a cat");
            assert_eq!(loaded.last_model, "flux-dev:q4");
            assert_eq!(loaded.width, Some(1024));
            assert_eq!(loaded.steps, Some(20));
            assert_eq!(loaded.theme.as_deref(), Some("dracula"));
            assert_eq!(loaded.negative_collapsed, Some(true));
        });
    }

    #[test]
    #[serial(mold_env)]
    fn save_load_save_load_preserves_theme_across_many_cycles() {
        // Regression (preserved from the old JSON implementation): theme
        // must survive repeated save → load cycles, *including* when the
        // loaded session is re-saved without touching the theme.
        with_isolated_env(|_home| {
            let seed = TuiSession {
                last_model: "flux-dev:q4".into(),
                theme: Some("mocha".into()),
                ..Default::default()
            };
            seed.save();

            for i in 0..10 {
                let loaded = TuiSession::load();
                assert_eq!(
                    loaded.theme.as_deref(),
                    Some("mocha"),
                    "iteration {i}: theme must round-trip as 'mocha', got {:?}",
                    loaded.theme
                );
                loaded.save();
            }
        });
    }

    #[test]
    #[serial(mold_env)]
    fn legacy_json_is_imported_once_and_file_is_renamed() {
        with_isolated_env(|home| {
            // Drop a legacy session file in MOLD_HOME.
            let src = home.join("tui-session.json");
            let legacy = TuiSession {
                last_model: "sdxl:fp16".into(),
                last_prompt: "from legacy".into(),
                width: Some(768),
                theme: Some("nord".into()),
                ..Default::default()
            };
            std::fs::write(&src, serde_json::to_string(&legacy).unwrap()).unwrap();
            assert!(src.exists());

            // First load triggers the import.
            let loaded = TuiSession::load();
            assert_eq!(loaded.last_prompt, "from legacy");
            assert_eq!(loaded.width, Some(768));
            assert_eq!(loaded.theme.as_deref(), Some("nord"));

            // Legacy file was renamed, not deleted.
            assert!(!src.exists(), "legacy file should have been renamed");
            assert!(
                home.join("tui-session.json.migrated").exists(),
                "legacy file should live under .migrated"
            );

            // A second load is a no-op: nothing to import, nothing to rename.
            let again = TuiSession::load();
            assert_eq!(again.last_prompt, "from legacy");
        });
    }

    #[test]
    #[serial(mold_env)]
    fn legacy_json_import_is_idempotent() {
        with_isolated_env(|home| {
            let src = home.join("tui-session.json");
            std::fs::write(
                &src,
                r#"{"last_prompt":"first","last_model":"flux-dev:q4"}"#,
            )
            .unwrap();

            TuiSession::load();

            // Simulate a new file appearing *after* the first import
            // (e.g. user restored a backup). Because the sentinel is
            // set, the importer leaves it alone.
            std::fs::write(&src, r#"{"last_prompt":"SHOULD-NOT-OVERWRITE"}"#).unwrap();
            let loaded = TuiSession::load();
            assert_eq!(
                loaded.last_prompt, "first",
                "sentinel must prevent re-importing stale JSON"
            );
        });
    }

    #[test]
    #[serial(mold_env)]
    fn db_disabled_returns_default_without_persistence() {
        with_isolated_env(|_home| {
            std::env::set_var("MOLD_DB_DISABLE", "1");
            let seed = TuiSession {
                last_model: "flux-dev:q4".into(),
                theme: Some("nord".into()),
                ..Default::default()
            };
            seed.save(); // silently no-ops
            let loaded = TuiSession::load();
            assert!(loaded.last_model.is_empty());
            assert!(loaded.theme.is_none());
            std::env::remove_var("MOLD_DB_DISABLE");
        });
    }

    #[test]
    fn session_deserialize_missing_fields() {
        // Legacy JSON round-trip is still needed for the one-shot import.
        let json = r#"{"last_prompt": "test"}"#;
        let session: TuiSession = serde_json::from_str(json).unwrap();
        assert_eq!(session.last_prompt, "test");
        assert!(session.last_model.is_empty());
        assert_eq!(session.width, None);
        assert_eq!(session.batch, None);
    }

    #[test]
    fn from_params_captures_all_fields() {
        use crate::app::{GenerateParams, SeedMode};

        let params = GenerateParams {
            model: "sdxl-turbo:fp16".to_string(),
            width: 512,
            height: 512,
            steps: 8,
            guidance: 2.0,
            seed: Some(42),
            seed_mode: SeedMode::Fixed,
            batch: 3,
            format: mold_core::OutputFormat::Jpeg,
            scheduler: Some(mold_core::Scheduler::EulerAncestral),
            inference_mode: crate::app::InferenceMode::Auto,
            host: None,
            lora_path: Some("/path/to/lora.safetensors".to_string()),
            lora_scale: 0.7,
            expand: true,
            offload: true,
            source_image_path: None,
            strength: 0.6,
            mask_image_path: None,
            frames: 25,
            fps: 24,
            control_image_path: None,
            control_model: None,
            control_scale: 0.8,
        };

        let session = TuiSession::from_params("a sunset", "blurry", &params);
        assert_eq!(session.last_prompt, "a sunset");
        assert_eq!(session.last_negative, "blurry");
        assert_eq!(session.last_model, "sdxl-turbo:fp16");
        assert_eq!(session.width, Some(512));
        assert_eq!(session.height, Some(512));
        assert_eq!(session.steps, Some(8));
        assert_eq!(session.guidance, Some(2.0));
        assert_eq!(session.seed_mode, Some("fixed".to_string()));
        assert_eq!(session.batch, Some(3));
        assert_eq!(session.format, Some("jpeg".to_string()));
        assert_eq!(session.scheduler, Some("eulerancestral".to_string()));
        assert_eq!(
            session.lora_path,
            Some("/path/to/lora.safetensors".to_string())
        );
        assert_eq!(session.lora_scale, Some(0.7));
        assert_eq!(session.expand, Some(true));
        assert_eq!(session.offload, Some(true));
        assert_eq!(session.strength, Some(0.6));
        assert_eq!(session.control_scale, Some(0.8));
    }

    #[test]
    fn apply_to_params_restores_all_fields() {
        use crate::app::{GenerateParams, SeedMode};

        let session = TuiSession {
            last_prompt: "a cat".to_string(),
            last_negative: "ugly".to_string(),
            last_model: "sd15:fp16".to_string(),
            width: Some(512),
            height: Some(768),
            steps: Some(30),
            guidance: Some(7.5),
            seed_mode: Some("increment".to_string()),
            batch: Some(4),
            format: Some("jpeg".to_string()),
            scheduler: Some("unipc".to_string()),
            lora_path: Some("/lora.safetensors".to_string()),
            lora_scale: Some(0.5),
            expand: Some(true),
            offload: Some(false),
            strength: Some(0.3),
            control_scale: Some(1.5),
            theme: None,
            negative_collapsed: None,
        };

        let mut params = GenerateParams::from_config(&mold_core::Config::default());
        session.apply_to_params(&mut params);

        assert_eq!(params.width, 512);
        assert_eq!(params.height, 768);
        assert_eq!(params.steps, 30);
        assert_eq!(params.guidance, 7.5);
        assert_eq!(params.seed_mode, SeedMode::Increment);
        assert_eq!(params.batch, 4);
        assert_eq!(params.format, mold_core::OutputFormat::Jpeg);
        assert_eq!(params.scheduler, Some(mold_core::Scheduler::UniPc));
        assert_eq!(params.lora_path, Some("/lora.safetensors".to_string()));
        assert_eq!(params.lora_scale, 0.5);
        assert!(params.expand);
        assert!(!params.offload);
        assert_eq!(params.strength, 0.3);
        assert_eq!(params.control_scale, 1.5);
    }

    #[test]
    fn apply_non_model_params_skips_dimensions_and_guidance() {
        use crate::app::GenerateParams;

        let session = TuiSession {
            last_model: "sd15:fp16".to_string(),
            width: Some(512),
            height: Some(512),
            steps: Some(25),
            guidance: Some(7.5),
            scheduler: Some("ddim".to_string()),
            batch: Some(3),
            expand: Some(true),
            offload: Some(true),
            ..Default::default()
        };

        let mut params = GenerateParams::from_config(&mold_core::Config::default());
        let original_width = params.width;
        let original_height = params.height;
        let original_steps = params.steps;
        let original_guidance = params.guidance;

        session.apply_non_model_params(&mut params);

        assert_eq!(params.width, original_width);
        assert_eq!(params.height, original_height);
        assert_eq!(params.steps, original_steps);
        assert_eq!(params.guidance, original_guidance);
        assert_eq!(params.scheduler, None);

        assert_eq!(params.batch, 3);
        assert!(params.expand);
        assert!(params.offload);
    }
}
