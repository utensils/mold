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
        }
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
        };
        let json = serde_json::to_string(&session).unwrap();
        let restored: TuiSession = serde_json::from_str(&json).unwrap();
        assert_eq!(restored.last_prompt, "a cat in a hat");
        assert_eq!(restored.last_model, "flux2-klein:q8");
        assert_eq!(restored.width, Some(1024));
        assert_eq!(restored.batch, Some(2));
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
}
