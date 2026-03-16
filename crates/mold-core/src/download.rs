use std::path::PathBuf;

use console::Term;
use hf_hub::api::tokio::{Api, ApiBuilder, ApiError, Progress};
use hf_hub::{Repo, RepoType};
use indicatif::{MultiProgress, ProgressBar, ProgressDrawTarget, ProgressStyle};
use thiserror::Error;

use crate::manifest::{paths_from_downloads, ModelComponent, ModelFile, ModelManifest};
use crate::ModelPaths;

#[derive(Debug, Error)]
pub enum DownloadError {
    #[error("Model requires access approval on HuggingFace.\n\n  1. Visit: https://huggingface.co/{repo}\n  2. Accept the license agreement\n  3. Create a token at: https://huggingface.co/settings/tokens\n  4. Set: export HF_TOKEN=hf_...\n  5. Retry: mold pull {model}")]
    GatedModel { repo: String, model: String },

    #[error("Download failed for {filename} from {repo}: {source}")]
    DownloadFailed {
        repo: String,
        filename: String,
        source: ApiError,
    },

    #[error("Failed to build HuggingFace API client: {0}")]
    ApiSetup(#[from] ApiError),

    #[error("Failed to build sync HuggingFace API client: {0}")]
    SyncApiSetup(String),

    #[error("Sync download failed for {filename} from {repo}: {message}")]
    SyncDownloadFailed {
        repo: String,
        filename: String,
        message: String,
    },

    #[error("Missing component after download — this is a bug")]
    MissingComponent,
}

/// Truncate a string to fit within `max_len`, replacing the middle with "..." if needed.
fn truncate_filename(name: &str, max_len: usize) -> String {
    if name.len() <= max_len || max_len < 8 {
        return name.to_string();
    }
    // Keep the end of the filename (the unique part) and trim the start
    let suffix_len = max_len - 3; // "..." prefix
    let start = name.len() - suffix_len;
    format!("...{}", &name[start..])
}

/// Maximum characters for the filename column in progress bars.
/// Derived from terminal width minus the fixed overhead of the bar template:
/// 2 (indent) + 1 (space) + 1 ([) + 30 (bar) + 1 (]) + ~40 (bytes/speed/eta) = ~75 chars overhead.
fn filename_column_width() -> usize {
    let term_width = Term::stderr().size().1 as usize;
    term_width.saturating_sub(75).max(12)
}

/// Progress adapter bridging hf-hub's `Progress` trait to an `indicatif::ProgressBar`.
#[derive(Clone)]
struct DownloadProgress {
    bar: ProgressBar,
    max_msg_len: usize,
}

impl DownloadProgress {
    fn new(bar: ProgressBar, max_msg_len: usize) -> Self {
        Self { bar, max_msg_len }
    }
}

impl Progress for DownloadProgress {
    async fn init(&mut self, size: usize, filename: &str) {
        self.bar.set_length(size as u64);
        self.bar
            .set_message(truncate_filename(filename, self.max_msg_len));
    }

    async fn update(&mut self, size: usize) {
        self.bar.inc(size as u64);
    }

    async fn finish(&mut self) {
        self.bar.finish_with_message("done");
    }
}

/// Download all files for a model manifest, returning resolved paths.
pub async fn pull_model(manifest: &ModelManifest) -> Result<ModelPaths, DownloadError> {
    let api = ApiBuilder::from_env().build()?;

    let multi = MultiProgress::with_draw_target(ProgressDrawTarget::stderr());
    let msg_width = filename_column_width();
    let bar_style = ProgressStyle::with_template(&format!(
        "  {{msg:<{msg_width}}} [{{bar:30.cyan/dim}}] {{bytes}}/{{total_bytes}} ({{bytes_per_sec}}, {{eta}})"
    ))
    .unwrap()
    .progress_chars("━╸─");

    let mut downloads: Vec<(ModelComponent, PathBuf)> = Vec::new();

    for file in &manifest.files {
        let bar = multi.add(ProgressBar::new(file.size_bytes));
        bar.set_style(bar_style.clone());
        bar.set_message(truncate_filename(&file.hf_filename, msg_width));

        let path = download_file(
            &api,
            file,
            DownloadProgress::new(bar, msg_width),
            &manifest.name,
        )
        .await?;
        downloads.push((file.component, path));
    }

    paths_from_downloads(&downloads).ok_or(DownloadError::MissingComponent)
}

async fn download_file(
    api: &Api,
    file: &ModelFile,
    progress: DownloadProgress,
    model_name: &str,
) -> Result<PathBuf, DownloadError> {
    let repo = api.repo(Repo::new(file.hf_repo.clone(), RepoType::Model));

    match repo
        .download_with_progress(&file.hf_filename, progress)
        .await
    {
        Ok(path) => Ok(path),
        Err(e) => {
            let err_str = e.to_string();
            if err_str.contains("403")
                || err_str.contains("Forbidden")
                || err_str.contains("gated")
                || err_str.contains("Access denied")
            {
                Err(DownloadError::GatedModel {
                    repo: file.hf_repo.clone(),
                    model: model_name.to_string(),
                })
            } else {
                Err(DownloadError::DownloadFailed {
                    repo: file.hf_repo.clone(),
                    filename: file.hf_filename.clone(),
                    source: e,
                })
            }
        }
    }
}

// ── Synchronous single-file download (for use from spawn_blocking) ───────────

/// Download a single file from HuggingFace, returning its cached path.
/// Uses the sync hf-hub API — safe to call from `spawn_blocking`.
/// Returns immediately if already cached.
pub fn download_single_file_sync(
    hf_repo: &str,
    hf_filename: &str,
) -> Result<PathBuf, DownloadError> {
    use hf_hub::api::sync::ApiBuilder;

    let api = ApiBuilder::from_env()
        .build()
        .map_err(|e| DownloadError::SyncApiSetup(e.to_string()))?;
    let repo = api.repo(Repo::new(hf_repo.to_string(), RepoType::Model));
    repo.get(hf_filename)
        .map_err(|e| DownloadError::SyncDownloadFailed {
            repo: hf_repo.to_string(),
            filename: hf_filename.to_string(),
            message: e.to_string(),
        })
}

/// Check if a file is already cached locally (no download).
pub fn cached_file_path(hf_repo: &str, hf_filename: &str) -> Option<PathBuf> {
    use hf_hub::Cache;

    let cache = Cache::from_env();
    let repo = cache.repo(Repo::new(hf_repo.to_string(), RepoType::Model));
    repo.get(hf_filename)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn truncate_short_name_unchanged() {
        assert_eq!(truncate_filename("ae.safetensors", 45), "ae.safetensors");
    }

    #[test]
    fn truncate_exact_fit_unchanged() {
        let name = "x".repeat(30);
        assert_eq!(truncate_filename(&name, 30), name);
    }

    #[test]
    fn truncate_long_name_keeps_suffix() {
        let result = truncate_filename("unet/diffusion_pytorch_model.fp16.safetensors", 30);
        assert_eq!(result.len(), 30);
        assert!(result.starts_with("..."));
        assert!(result.ends_with(".fp16.safetensors"));
    }

    #[test]
    fn truncate_very_small_max_returns_original() {
        // max_len < 8 returns unchanged to avoid degenerate "..." output
        let name = "something.safetensors";
        assert_eq!(truncate_filename(name, 5), name);
    }

    #[test]
    fn download_error_gated_message() {
        let err = DownloadError::GatedModel {
            repo: "black-forest-labs/FLUX.1-dev".to_string(),
            model: "flux-dev:q8".to_string(),
        };
        let msg = err.to_string();
        assert!(msg.contains("huggingface.co/black-forest-labs/FLUX.1-dev"));
        assert!(msg.contains("HF_TOKEN"));
        assert!(msg.contains("mold pull flux-dev:q8"));
    }
}
