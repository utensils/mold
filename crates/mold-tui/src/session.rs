use std::path::PathBuf;

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
}

fn session_path() -> Option<PathBuf> {
    mold_core::Config::mold_dir().map(|d| d.join("tui-session.json"))
}

impl TuiSession {
    /// Load session from disk. Returns default if file missing or malformed.
    pub fn load() -> Self {
        let path = match session_path() {
            Some(p) => p,
            None => return Self::default(),
        };
        match std::fs::read_to_string(&path) {
            Ok(contents) => serde_json::from_str(&contents).unwrap_or_default(),
            Err(_) => Self::default(),
        }
    }

    /// Save session to disk (best-effort, non-fatal).
    pub fn save(&self) {
        let path = match session_path() {
            Some(p) => p,
            None => return,
        };
        if let Some(parent) = path.parent() {
            let _ = std::fs::create_dir_all(parent);
        }
        if let Ok(json) = serde_json::to_string_pretty(self) {
            let _ = std::fs::write(&path, json);
        }
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
        }
    }

    /// Attach a theme slug for persistence. Chainable so call sites can append
    /// to `from_params` without adding a positional argument.
    pub fn with_theme(mut self, preset: super::ui::theme::ThemePreset) -> Self {
        self.theme = Some(preset.slug().to_string());
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_session_has_no_prompt() {
        let session = TuiSession::default();
        assert!(!session.has_prompt());
        assert!(session.last_prompt.is_empty());
        assert!(session.last_model.is_empty());
    }

    #[test]
    fn session_roundtrip_json() {
        let session = TuiSession {
            last_prompt: "a cat in a hat".to_string(),
            last_negative: "blurry".to_string(),
            last_model: "flux2-klein:q8".to_string(),
            width: Some(1024),
            height: Some(1024),
            steps: Some(4),
            guidance: Some(0.0),
            seed_mode: Some("random".to_string()),
            batch: Some(2),
            format: Some("png".to_string()),
            scheduler: None,
            lora_path: None,
            lora_scale: Some(1.0),
            expand: Some(false),
            offload: Some(true),
            strength: Some(0.75),
            control_scale: Some(1.0),
            theme: Some("mocha".to_string()),
        };
        let json = serde_json::to_string(&session).unwrap();
        let restored: TuiSession = serde_json::from_str(&json).unwrap();
        assert_eq!(restored.last_prompt, "a cat in a hat");
        assert_eq!(restored.last_model, "flux2-klein:q8");
        assert_eq!(restored.width, Some(1024));
        assert_eq!(restored.batch, Some(2));
        assert_eq!(restored.theme.as_deref(), Some("mocha"));
        assert_eq!(restored.offload, Some(true));
        assert_eq!(restored.seed_mode, Some("random".to_string()));
    }

    #[test]
    fn session_deserialize_missing_fields() {
        let json = r#"{"last_prompt": "test"}"#;
        let session: TuiSession = serde_json::from_str(json).unwrap();
        assert_eq!(session.last_prompt, "test");
        assert!(session.last_model.is_empty());
        assert_eq!(session.width, None);
        assert_eq!(session.batch, None);
    }

    #[test]
    fn session_backward_compat_old_format() {
        // Old format only had last_width/last_height etc
        let json = r#"{"last_prompt":"old","last_model":"flux","last_width":512}"#;
        let session: TuiSession = serde_json::from_str(json).unwrap();
        assert_eq!(session.last_prompt, "old");
        // Old field names don't match new ones — they'll be None
        assert_eq!(session.width, None);
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
    fn from_params_then_apply_roundtrips() {
        use crate::app::{GenerateParams, SeedMode};

        let original = GenerateParams {
            model: "flux-dev:q4".to_string(),
            width: 768,
            height: 1024,
            steps: 25,
            guidance: 3.5,
            seed: None,
            seed_mode: SeedMode::Increment,
            batch: 2,
            format: mold_core::OutputFormat::Png,
            scheduler: Some(mold_core::Scheduler::Ddim),
            inference_mode: crate::app::InferenceMode::Auto,
            host: None,
            lora_path: None,
            lora_scale: 1.0,
            expand: false,
            offload: false,
            source_image_path: None,
            strength: 0.75,
            mask_image_path: None,
            frames: 25,
            fps: 24,
            control_image_path: None,
            control_model: None,
            control_scale: 1.0,
        };

        // Save to session
        let session = TuiSession::from_params("test prompt", "bad quality", &original);

        // Apply to fresh params
        let mut restored = GenerateParams::from_config(&mold_core::Config::default());
        session.apply_to_params(&mut restored);

        // All persisted fields should match
        assert_eq!(restored.width, 768);
        assert_eq!(restored.height, 1024);
        assert_eq!(restored.steps, 25);
        assert_eq!(restored.guidance, 3.5);
        assert_eq!(restored.seed_mode, SeedMode::Increment);
        assert_eq!(restored.batch, 2);
        assert_eq!(restored.format, mold_core::OutputFormat::Png);
        assert_eq!(restored.scheduler, Some(mold_core::Scheduler::Ddim));
        assert_eq!(restored.lora_scale, 1.0);
        assert!(!restored.expand);
        assert!(!restored.offload);
        assert_eq!(restored.strength, 0.75);
        assert_eq!(restored.control_scale, 1.0);
    }

    #[test]
    fn bare_model_name_resolves_on_load() {
        // Session files from older versions may store bare names like "flux2-klein"
        // instead of "flux2-klein:q8". The app should resolve these on load.
        let json = r#"{"last_prompt":"test","last_model":"flux2-klein","width":512,"height":512}"#;
        let session: TuiSession = serde_json::from_str(json).unwrap();

        // The session itself stores exactly what was saved
        assert_eq!(session.last_model, "flux2-klein");

        // But when the app resolves it, it should become the tagged name
        let resolved = mold_core::manifest::resolve_model_name(&session.last_model);
        assert_eq!(resolved, "flux2-klein:q8");
    }

    #[test]
    fn tagged_model_name_survives_resolution() {
        let session = TuiSession {
            last_model: "flux-dev:q4".to_string(),
            ..Default::default()
        };
        let resolved = mold_core::manifest::resolve_model_name(&session.last_model);
        assert_eq!(resolved, "flux-dev:q4");
    }

    #[test]
    fn session_with_custom_dimensions_roundtrips() {
        // Regression: user sets 512x512 in TUI, quits without generating,
        // next launch should restore 512x512
        use crate::app::{GenerateParams, SeedMode};

        let params = GenerateParams {
            model: "flux2-klein:q8".to_string(),
            width: 512,
            height: 512,
            steps: 4,
            guidance: 0.0,
            seed: None,
            seed_mode: SeedMode::Random,
            batch: 1,
            format: mold_core::OutputFormat::Png,
            scheduler: None,
            inference_mode: crate::app::InferenceMode::Auto,
            host: None,
            lora_path: None,
            lora_scale: 1.0,
            expand: false,
            offload: false,
            source_image_path: None,
            strength: 0.75,
            mask_image_path: None,
            frames: 25,
            fps: 24,
            control_image_path: None,
            control_model: None,
            control_scale: 1.0,
        };

        // Simulate save on quit
        let session = TuiSession::from_params("", "", &params);
        assert_eq!(session.width, Some(512));
        assert_eq!(session.height, Some(512));
        assert_eq!(session.last_model, "flux2-klein:q8");

        // Simulate restore on next launch
        let mut fresh = GenerateParams::from_config(&mold_core::Config::default());
        // fresh would have model defaults (1024x1024 for flux2-klein)
        session.apply_to_params(&mut fresh);
        assert_eq!(fresh.width, 512);
        assert_eq!(fresh.height, 512);
    }

    #[test]
    fn apply_non_model_params_skips_dimensions_and_guidance() {
        // When the saved model is unavailable, the fallback should NOT apply
        // model-specific params (width, height, steps, guidance, scheduler)
        // because they belong to the missing model and would be wrong for the
        // current default.
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

        // Model-specific settings should NOT have changed
        assert_eq!(params.width, original_width);
        assert_eq!(params.height, original_height);
        assert_eq!(params.steps, original_steps);
        assert_eq!(params.guidance, original_guidance);
        assert_eq!(params.scheduler, None); // should stay as default, not "ddim"

        // Non-model settings SHOULD have been applied
        assert_eq!(params.batch, 3);
        assert!(params.expand);
        assert!(params.offload);
    }

    #[test]
    fn custom_config_model_name_preserved() {
        // Config-only models like [models."my-flux"] should NOT be rewritten
        // by resolve_model_name to "my-flux:q8"
        let resolved = mold_core::manifest::resolve_model_name("my-custom-model");
        // resolve_model_name appends :q8 when no manifest match is found,
        // but the catalog lookup in the app uses the original name first
        assert_eq!(resolved, "my-custom-model:q8");
        // The app code tries the exact session name first, then resolved —
        // so "my-custom-model" would match the catalog entry directly.
    }
}
