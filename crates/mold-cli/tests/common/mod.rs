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

    /// Create a `Command` for the mold binary with isolated env vars.
    pub fn cmd(&self) -> Command {
        let mut cmd = Command::cargo_bin("mold").expect("mold binary not found");
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
