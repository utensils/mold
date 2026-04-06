//! Shared test harness for CLI integration tests.
//!
//! Provides an isolated environment with temp directories for `MOLD_HOME`
//! and `MOLD_MODELS_DIR`, ensuring tests don't read from the host machine's
//! real config or model files.

use std::path::PathBuf;

use assert_cmd::Command;
use tempfile::TempDir;

/// An isolated test environment with temp `MOLD_HOME` and `MOLD_MODELS_DIR`.
///
/// All commands created via [`TestEnv::cmd`] inherit these env vars, so the
/// mold binary operates entirely within the temp directory.
pub struct TestEnv {
    _dir: TempDir,
    pub home: PathBuf,
    pub models: PathBuf,
    pub output: PathBuf,
}

impl TestEnv {
    /// Create a new isolated test environment.
    pub fn new() -> Self {
        let dir = TempDir::new().expect("failed to create temp dir");
        let home = dir.path().join(".mold");
        let models = home.join("models");
        let output = home.join("output");
        std::fs::create_dir_all(&models).unwrap();
        std::fs::create_dir_all(&output).unwrap();
        Self {
            _dir: dir,
            home,
            models,
            output,
        }
    }

    /// All `MOLD_*` env vars that could leak from the developer's shell.
    /// These are cleared before setting our isolated values so tests are
    /// deterministic regardless of the host environment.
    const MOLD_ENV_VARS: &'static [&'static str] = &[
        "MOLD_HOME",
        "MOLD_MODELS_DIR",
        "MOLD_OUTPUT_DIR",
        "MOLD_HOST",
        "MOLD_PORT",
        "MOLD_DEFAULT_MODEL",
        "MOLD_LOG",
        "MOLD_EAGER",
        "MOLD_OFFLOAD",
        "MOLD_EMBED_METADATA",
        "MOLD_PREVIEW",
        "MOLD_T5_VARIANT",
        "MOLD_QWEN3_VARIANT",
        "MOLD_SCHEDULER",
        "MOLD_API_KEY",
        "MOLD_RATE_LIMIT",
        "MOLD_RATE_LIMIT_BURST",
        "MOLD_CORS_ORIGIN",
        "MOLD_EXPAND",
        "MOLD_EXPAND_BACKEND",
        "MOLD_EXPAND_MODEL",
        "MOLD_EXPAND_TEMPERATURE",
        "MOLD_EXPAND_THINKING",
        "MOLD_EXPAND_SYSTEM_PROMPT",
        "MOLD_EXPAND_BATCH_PROMPT",
        "MOLD_DISCORD_TOKEN",
        "MOLD_UPSCALE_MODEL",
        "MOLD_UPSCALE_TILE_SIZE",
        "MOLD_DEVICE",
        "MOLD_TRANSFORMER_PATH",
        "MOLD_VAE_PATH",
        "MOLD_T5_PATH",
        "MOLD_CLIP_PATH",
        "MOLD_T5_TOKENIZER_PATH",
        "MOLD_CLIP_TOKENIZER_PATH",
        "MOLD_CLIP2_PATH",
        "MOLD_CLIP2_TOKENIZER_PATH",
    ];

    /// Create a `Command` for the mold binary with isolated env vars.
    ///
    /// Clears all `MOLD_*` env vars from the inherited environment first,
    /// then sets only the vars needed for test isolation. This prevents
    /// developer shell exports (e.g. `MOLD_DEFAULT_MODEL`) from leaking
    /// into test subprocesses.
    pub fn cmd(&self) -> Command {
        let mut cmd = Command::cargo_bin("mold").expect("mold binary not found");
        // Clear all MOLD_* vars to prevent host environment leakage
        for var in Self::MOLD_ENV_VARS {
            cmd.env_remove(var);
        }
        cmd.env("MOLD_HOME", &self.home);
        cmd.env("MOLD_MODELS_DIR", &self.models);
        cmd.env("MOLD_OUTPUT_DIR", &self.output);
        cmd.env("NO_COLOR", "1");
        // Prevent tests from connecting to a running server
        cmd.env("MOLD_HOST", "http://127.0.0.1:1");
        cmd
    }

    /// Write a config.toml file in the test environment.
    #[allow(dead_code)]
    pub fn write_config(&self, toml_content: &str) {
        let config_dir = self.home.join("config");
        std::fs::create_dir_all(&config_dir).ok();
        // mold reads from MOLD_HOME/config.toml
        std::fs::write(self.home.join("config.toml"), toml_content).unwrap();
    }

    /// Create stub files on disk for a manifest-backed model so it appears
    /// as "installed" to `mold list`, `mold rm`, etc.
    pub fn populate_manifest_model(&self, name: &str) {
        let manifest = mold_core::manifest::find_manifest(name)
            .unwrap_or_else(|| panic!("unknown manifest model: {name}"));
        for file in &manifest.files {
            let path = self
                .models
                .join(mold_core::manifest::storage_path(manifest, file));
            if let Some(parent) = path.parent() {
                std::fs::create_dir_all(parent).unwrap();
            }
            // Write a small stub — enough for existence checks, not for inference
            std::fs::write(&path, b"stub").unwrap();
        }
    }
}
