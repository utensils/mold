use serde::Deserialize;
use std::path::PathBuf;

#[derive(Debug, Clone, Deserialize)]
pub struct Config {
    #[serde(default = "default_model")]
    pub default_model: String,

    #[serde(default = "default_models_dir")]
    pub models_dir: String,

    #[serde(default = "default_port")]
    pub server_port: u16,

    #[serde(default = "default_output_dir")]
    pub output_dir: String,

    #[serde(default = "default_dimension")]
    pub default_width: u32,

    #[serde(default = "default_dimension")]
    pub default_height: u32,
}

fn default_model() -> String {
    "flux-schnell".to_string()
}

fn default_models_dir() -> String {
    "~/.mold/models".to_string()
}

fn default_port() -> u16 {
    7680
}

fn default_output_dir() -> String {
    ".".to_string()
}

fn default_dimension() -> u32 {
    1024
}

impl Default for Config {
    fn default() -> Self {
        Self {
            default_model: default_model(),
            models_dir: default_models_dir(),
            server_port: default_port(),
            output_dir: default_output_dir(),
            default_width: default_dimension(),
            default_height: default_dimension(),
        }
    }
}

impl Config {
    pub fn load_or_default() -> Self {
        let config_path = Self::config_path();
        if config_path.exists() {
            match std::fs::read_to_string(&config_path) {
                Ok(contents) => toml::from_str(&contents).unwrap_or_default(),
                Err(_) => Config::default(),
            }
        } else {
            Config::default()
        }
    }

    pub fn config_path() -> PathBuf {
        dirs::home_dir()
            .unwrap_or_else(|| PathBuf::from("."))
            .join(".mold")
            .join("config.toml")
    }

    pub fn data_dir() -> PathBuf {
        dirs::home_dir()
            .unwrap_or_else(|| PathBuf::from("."))
            .join(".mold")
    }

    pub fn resolved_models_dir(&self) -> PathBuf {
        if let Ok(env_dir) = std::env::var("MOLD_MODELS_DIR") {
            PathBuf::from(env_dir)
        } else {
            let expanded = self.models_dir.replace(
                "~",
                &dirs::home_dir()
                    .unwrap_or_else(|| PathBuf::from("."))
                    .to_string_lossy(),
            );
            PathBuf::from(expanded)
        }
    }
}
