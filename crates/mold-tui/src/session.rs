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
    #[serde(default)]
    pub last_width: Option<u32>,
    #[serde(default)]
    pub last_height: Option<u32>,
    #[serde(default)]
    pub last_steps: Option<u32>,
    #[serde(default)]
    pub last_guidance: Option<f64>,
    #[serde(default)]
    pub last_seed_mode: Option<String>,
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
            last_width: Some(1024),
            last_height: Some(1024),
            last_steps: Some(4),
            last_guidance: Some(0.0),
            last_seed_mode: Some("random".to_string()),
        };
        let json = serde_json::to_string(&session).unwrap();
        let restored: TuiSession = serde_json::from_str(&json).unwrap();
        assert_eq!(restored.last_prompt, "a cat in a hat");
        assert_eq!(restored.last_model, "flux2-klein:q8");
        assert_eq!(restored.last_width, Some(1024));
        assert_eq!(restored.last_seed_mode, Some("random".to_string()));
    }

    #[test]
    fn session_deserialize_missing_fields() {
        let json = r#"{"last_prompt": "test"}"#;
        let session: TuiSession = serde_json::from_str(json).unwrap();
        assert_eq!(session.last_prompt, "test");
        assert!(session.last_model.is_empty());
        assert_eq!(session.last_width, None);
    }
}
