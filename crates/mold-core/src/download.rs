use std::path::PathBuf;
use std::sync::{Arc, OnceLock};
use std::time::Instant;

use console::Term;
use hf_hub::api::tokio::{Api, ApiBuilder, ApiError, Progress};
use hf_hub::{Cache, Repo, RepoType};
use indicatif::{MultiProgress, ProgressBar, ProgressDrawTarget, ProgressStyle};
use thiserror::Error;

use crate::manifest::{paths_from_downloads, ModelComponent, ModelFile, ModelManifest};
use crate::ModelPaths;

/// Callback-based download progress event.
#[derive(Debug, Clone)]
pub enum DownloadProgressEvent {
    /// A file download has started.
    FileStart {
        filename: String,
        file_index: usize,
        total_files: usize,
        size_bytes: u64,
    },
    /// Bytes downloaded for the current file.
    FileProgress {
        filename: String,
        file_index: usize,
        bytes_downloaded: u64,
        bytes_total: u64,
    },
    /// A file download completed.
    FileDone {
        filename: String,
        file_index: usize,
        total_files: usize,
    },
}

/// Callback type for download progress reporting.
pub type DownloadProgressCallback = Arc<dyn Fn(DownloadProgressEvent) + Send + Sync>;

#[derive(Debug, Error)]
pub enum DownloadError {
    #[error(
        "Model requires access approval on HuggingFace.\n\n  1. Visit: https://huggingface.co/{repo}\n  2. Accept the license agreement\n  3. Create a token at: https://huggingface.co/settings/tokens\n  4. Set: export HF_TOKEN=hf_...\n  5. Retry: mold pull {model}"
    )]
    GatedModel { repo: String, model: String },

    #[error(
        "Authentication required for repository {repo}.\n\n  1. Create a token at: https://huggingface.co/settings/tokens\n     (select at least \"Read\" access)\n  2. Set: export HF_TOKEN=hf_...\n     Or run: huggingface-cli login\n  3. Retry: mold pull {model}\n\n  If HF_TOKEN is already set, it may be invalid or expired."
    )]
    Unauthorized { repo: String, model: String },

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

    #[error("IO error during file placement: {0}")]
    FilePlacement(String),

    #[error("Unknown model '{model}'. No manifest found.")]
    UnknownModel { model: String },

    #[error("Failed to save config: {0}")]
    ConfigSave(String),
}

/// Resolve HuggingFace token: `HF_TOKEN` env var takes precedence over
/// the token file (`~/.cache/huggingface/token` from `huggingface-cli login`).
fn resolve_hf_token() -> Option<String> {
    if let Ok(token) = std::env::var("HF_TOKEN") {
        let token = token.trim().to_string();
        if !token.is_empty() {
            return Some(token);
        }
    }
    Cache::new(hf_cache_dir())
        .token()
        .or_else(|| Cache::from_env().token())
}

/// Resolve the mold models directory. Computed once from config on first access.
/// Resolution order: `MOLD_MODELS_DIR` env var → config `models_dir` → `~/.mold/models`.
///
/// This is the clean model storage root. Actual model files live at clean paths like
/// `models/flux-schnell-q8/transformer.gguf` and `models/shared/flux/ae.safetensors`.
///
/// **OnceLock caching**: The directory is resolved once on the first call and cached
/// for the entire process lifetime. Changing `MOLD_MODELS_DIR` or the config file
/// after the first call has no effect. This is by design — model paths recorded in
/// config must remain stable within a single process run.
fn models_dir() -> PathBuf {
    static DIR: OnceLock<PathBuf> = OnceLock::new();
    DIR.get_or_init(|| {
        let dir = crate::Config::load_or_default().resolved_models_dir();
        let _ = std::fs::create_dir_all(&dir);
        dir
    })
    .clone()
}

/// Internal hf-hub cache directory: `<models_dir>/.hf-cache/`.
/// Hidden from users; files get hardlinked to clean paths after download.
fn hf_cache_dir() -> PathBuf {
    static DIR: OnceLock<PathBuf> = OnceLock::new();
    DIR.get_or_init(|| {
        let dir = models_dir().join(".hf-cache");
        let _ = std::fs::create_dir_all(&dir);
        dir
    })
    .clone()
}

/// Hardlink `src` to `dst`, falling back to copy if hardlink fails (cross-filesystem).
/// Idempotent: skips if `dst` already exists with the same size as `src`.
///
/// The source path is canonicalized to resolve hf-hub's symlink chain
/// (`snapshots/<sha>/file → ../../blobs/<hash>`) before any filesystem ops.
fn hardlink_or_copy(src: &std::path::Path, dst: &std::path::Path) -> Result<(), DownloadError> {
    // Resolve symlinks — hf-hub cache returns symlink paths that can cause
    // ENOENT on some filesystems when passed directly to hard_link or copy.
    let real_src = src.canonicalize().map_err(|e| {
        DownloadError::FilePlacement(format!(
            "source file not found after download: {} ({e})",
            src.display()
        ))
    })?;

    // Check if dst already has the correct content (idempotent skip).
    // Use metadata() which follows symlinks — only skip if the real target matches.
    if dst.exists() {
        if let (Ok(src_meta), Ok(dst_meta)) = (real_src.metadata(), dst.metadata()) {
            if src_meta.len() == dst_meta.len() {
                return Ok(());
            }
        }
    }

    // Remove stale destination before placement. A previous hard_link on an
    // hf-hub symlink creates a relative symlink that dangles from the new
    // location (e.g. shared/sd3/file → ../../blobs/hash, which doesn't exist
    // relative to shared/sd3/). symlink_metadata() sees these even though
    // exists() returns false for dangling symlinks.
    if dst.symlink_metadata().is_ok() {
        let _ = std::fs::remove_file(dst);
    }

    if let Some(parent) = dst.parent() {
        std::fs::create_dir_all(parent).map_err(|e| {
            DownloadError::FilePlacement(format!(
                "failed to create directory {}: {e}",
                parent.display()
            ))
        })?;
    }
    // Try hardlink first (zero extra disk space, instant)
    match std::fs::hard_link(&real_src, dst) {
        Ok(()) => return Ok(()),
        Err(_e) => {
            // Expected on cross-filesystem setups; fall through to copy
        }
    }
    // Fall back to copy (cross-filesystem or hard_link unsupported)
    std::fs::copy(&real_src, dst).map_err(|e| {
        DownloadError::FilePlacement(format!(
            "failed to copy {} → {}: {e}",
            real_src.display(),
            dst.display()
        ))
    })?;
    Ok(())
}

/// Verify the SHA-256 digest of a file against an expected hex string.
///
/// Returns `Ok(true)` when the digest matches, `Ok(false)` on mismatch.
/// Errors only on I/O failures (e.g. file not found).
pub fn verify_sha256(path: &std::path::Path, expected: &str) -> anyhow::Result<bool> {
    use sha2::{Digest, Sha256};

    let mut file = std::fs::File::open(path)?;
    let mut hasher = Sha256::new();
    std::io::copy(&mut file, &mut hasher)?;
    let digest = format!("{:x}", hasher.finalize());
    Ok(digest == expected)
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
    filename: String,
}

impl DownloadProgress {
    fn new(bar: ProgressBar, max_msg_len: usize) -> Self {
        Self {
            bar,
            max_msg_len,
            filename: String::new(),
        }
    }
}

impl Progress for DownloadProgress {
    async fn init(&mut self, size: usize, filename: &str) {
        self.bar.set_length(size as u64);
        self.filename = truncate_filename(filename, self.max_msg_len);
        self.bar.set_message(self.filename.clone());
    }

    async fn update(&mut self, size: usize) {
        self.bar.inc(size as u64);
    }

    async fn finish(&mut self) {
        self.bar.finish_with_message(self.filename.clone());
    }
}

/// Progress adapter that dispatches to a callback instead of indicatif.
/// Throttles `FileProgress` events to ~4/sec per file to avoid flooding SSE.
#[derive(Clone)]
struct CallbackProgress {
    callback: DownloadProgressCallback,
    file_index: usize,
    total_files: usize,
    accumulated: u64,
    total: u64,
    filename: String,
    last_emit: Instant,
}

impl CallbackProgress {
    fn new(callback: DownloadProgressCallback, file_index: usize, total_files: usize) -> Self {
        Self {
            callback,
            file_index,
            total_files,
            accumulated: 0,
            total: 0,
            filename: String::new(),
            last_emit: Instant::now(),
        }
    }
}

impl Progress for CallbackProgress {
    async fn init(&mut self, size: usize, filename: &str) {
        self.total = size as u64;
        self.accumulated = 0;
        self.filename = filename.to_string();
        (self.callback)(DownloadProgressEvent::FileStart {
            filename: self.filename.clone(),
            file_index: self.file_index,
            total_files: self.total_files,
            size_bytes: self.total,
        });
    }

    async fn update(&mut self, size: usize) {
        self.accumulated += size as u64;
        // Throttle to ~4 events/sec
        let now = Instant::now();
        if now.duration_since(self.last_emit).as_millis() >= 250 || self.accumulated >= self.total {
            self.last_emit = now;
            (self.callback)(DownloadProgressEvent::FileProgress {
                filename: self.filename.clone(),
                file_index: self.file_index,
                bytes_downloaded: self.accumulated,
                bytes_total: self.total,
            });
        }
    }

    async fn finish(&mut self) {
        (self.callback)(DownloadProgressEvent::FileDone {
            filename: self.filename.clone(),
            file_index: self.file_index,
            total_files: self.total_files,
        });
    }
}

/// Download all files for a model manifest, returning resolved paths.
///
/// Downloads go to a hidden hf-hub cache (`.hf-cache/`) for resume/dedup support,
/// then files are hardlinked to clean paths:
/// - Transformers → `<model-name>/<filename>`
/// - Shared components → `shared/<family>/<filename>`
pub async fn pull_model(manifest: &ModelManifest) -> Result<ModelPaths, DownloadError> {
    let mut builder = ApiBuilder::from_env().with_cache_dir(hf_cache_dir());
    if let Some(token) = resolve_hf_token() {
        builder = builder.with_token(Some(token));
    }
    let api = builder.build()?;

    let multi = MultiProgress::with_draw_target(ProgressDrawTarget::stderr());
    let msg_width = filename_column_width();
    let bar_style = ProgressStyle::with_template(&format!(
        "  {{msg:<{msg_width}}} [{{bar:30.cyan/dim}}] {{bytes}}/{{total_bytes}} ({{bytes_per_sec}}, {{eta}})"
    ))
    .unwrap()
    .progress_chars("━╸─");

    let mdir = models_dir();
    let mut downloads: Vec<(ModelComponent, PathBuf)> = Vec::new();

    for file in &manifest.files {
        let bar = multi.add(ProgressBar::new(file.size_bytes));
        bar.set_style(bar_style.clone());
        bar.set_message(truncate_filename(&file.hf_filename, msg_width));

        let hf_path = download_file(
            &api,
            file,
            DownloadProgress::new(bar, msg_width),
            &manifest.name,
        )
        .await?;

        // Place at clean path via hardlink (or copy as fallback)
        let clean_rel = crate::manifest::storage_path(manifest, file);
        let clean_path = mdir.join(&clean_rel);
        hardlink_or_copy(&hf_path, &clean_path)?;

        // Verify SHA-256 when expected digest is known
        if let Some(expected) = file.sha256 {
            match verify_sha256(&clean_path, expected) {
                Ok(true) => {}
                Ok(false) => {
                    eprintln!(
                        "warning: SHA-256 mismatch for {} (file may have been updated on HuggingFace)",
                        file.hf_filename
                    );
                }
                Err(e) => {
                    eprintln!(
                        "warning: failed to verify SHA-256 for {}: {e}",
                        file.hf_filename
                    );
                }
            }
        }

        downloads.push((file.component, clean_path));
    }

    paths_from_downloads(&downloads).ok_or(DownloadError::MissingComponent)
}

/// Download all files for a model manifest, reporting progress via callback.
///
/// Same as `pull_model` but uses a callback instead of indicatif progress bars.
/// Suitable for server-side downloads where terminal bars are not appropriate.
pub async fn pull_model_with_callback(
    manifest: &ModelManifest,
    callback: DownloadProgressCallback,
) -> Result<ModelPaths, DownloadError> {
    let mut builder = ApiBuilder::from_env().with_cache_dir(hf_cache_dir());
    if let Some(token) = resolve_hf_token() {
        builder = builder.with_token(Some(token));
    }
    let api = builder.build()?;

    let mdir = models_dir();
    let mut downloads: Vec<(ModelComponent, PathBuf)> = Vec::new();
    let total_files = manifest.files.len();

    for (idx, file) in manifest.files.iter().enumerate() {
        let progress = CallbackProgress::new(callback.clone(), idx, total_files);

        let hf_path = download_file(&api, file, progress, &manifest.name).await?;

        let clean_rel = crate::manifest::storage_path(manifest, file);
        let clean_path = mdir.join(&clean_rel);
        hardlink_or_copy(&hf_path, &clean_path)?;

        // Verify SHA-256 when expected digest is known
        if let Some(expected) = file.sha256 {
            match verify_sha256(&clean_path, expected) {
                Ok(true) => {}
                Ok(false) => {
                    eprintln!(
                        "warning: SHA-256 mismatch for {} (file may have been updated on HuggingFace)",
                        file.hf_filename
                    );
                }
                Err(e) => {
                    eprintln!(
                        "warning: failed to verify SHA-256 for {}: {e}",
                        file.hf_filename
                    );
                }
            }
        }

        downloads.push((file.component, clean_path));
    }

    paths_from_downloads(&downloads).ok_or(DownloadError::MissingComponent)
}

/// Extract HTTP status code from an async `ApiError`, if available.
fn extract_http_status(err: &ApiError) -> Option<u16> {
    if let ApiError::RequestError(reqwest_err) = err {
        reqwest_err.status().map(|s| s.as_u16())
    } else {
        None
    }
}

async fn download_file<P: Progress + Clone + Send + Sync + 'static>(
    api: &Api,
    file: &ModelFile,
    progress: P,
    model_name: &str,
) -> Result<PathBuf, DownloadError> {
    let repo = api.repo(Repo::new(file.hf_repo.clone(), RepoType::Model));

    match repo
        .download_with_progress(&file.hf_filename, progress)
        .await
    {
        Ok(path) => Ok(path),
        Err(e) => {
            let status = extract_http_status(&e);
            let err_str = e.to_string();
            if status == Some(401) || err_str.contains("401") || err_str.contains("Unauthorized") {
                Err(DownloadError::Unauthorized {
                    repo: file.hf_repo.clone(),
                    model: model_name.to_string(),
                })
            } else if status == Some(403)
                || err_str.contains("403")
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

/// Download a single file from HuggingFace, returning its path.
/// Uses the sync hf-hub API — safe to call from `spawn_blocking`.
/// Returns immediately if already cached.
///
/// If `target_subdir` is provided (e.g., `"shared/t5-gguf"`), the file is hardlinked
/// from the hf-cache to `<models_dir>/<target_subdir>/<leaf_filename>` and that clean
/// path is returned. If `None`, the raw hf-cache path is returned.
pub fn download_single_file_sync(
    hf_repo: &str,
    hf_filename: &str,
    target_subdir: Option<&str>,
) -> Result<PathBuf, DownloadError> {
    use hf_hub::api::sync::ApiBuilder;

    let mut builder = ApiBuilder::from_env().with_cache_dir(hf_cache_dir());
    if let Some(token) = resolve_hf_token() {
        builder = builder.with_token(Some(token));
    }
    let api = builder
        .build()
        .map_err(|e| DownloadError::SyncApiSetup(e.to_string()))?;
    let repo = api.repo(Repo::new(hf_repo.to_string(), RepoType::Model));
    let hf_path = repo.get(hf_filename).map_err(|e| {
        let err_str = e.to_string();
        if err_str.contains("401") || err_str.contains("Unauthorized") {
            DownloadError::Unauthorized {
                repo: hf_repo.to_string(),
                model: String::new(),
            }
        } else if err_str.contains("403")
            || err_str.contains("Forbidden")
            || err_str.contains("gated")
            || err_str.contains("Access denied")
        {
            DownloadError::GatedModel {
                repo: hf_repo.to_string(),
                model: String::new(),
            }
        } else {
            DownloadError::SyncDownloadFailed {
                repo: hf_repo.to_string(),
                filename: hf_filename.to_string(),
                message: err_str,
            }
        }
    })?;

    // Place at clean path if target_subdir specified
    if let Some(subdir) = target_subdir {
        let leaf = hf_filename.rsplit('/').next().unwrap_or(hf_filename);
        let clean_path = models_dir().join(subdir).join(leaf);
        hardlink_or_copy(&hf_path, &clean_path)?;
        Ok(clean_path)
    } else {
        Ok(hf_path)
    }
}

/// Check if a file is already cached locally (no download).
///
/// If `target_subdir` is provided, checks the clean path first
/// (`<models_dir>/<target_subdir>/<leaf_filename>`). Then checks the hf-cache,
/// old mold models dir (backward compat), and default HF cache.
pub fn cached_file_path(
    hf_repo: &str,
    hf_filename: &str,
    target_subdir: Option<&str>,
) -> Option<PathBuf> {
    // 1. Check clean path (if target_subdir specified)
    if let Some(subdir) = target_subdir {
        let leaf = hf_filename.rsplit('/').next().unwrap_or(hf_filename);
        let clean_path = models_dir().join(subdir).join(leaf);
        if clean_path.exists() {
            return Some(clean_path);
        }
    }

    // 2. Check new hf-cache location (~/.mold/models/.hf-cache/)
    let new_cache = Cache::new(hf_cache_dir());
    let new_repo = new_cache.repo(Repo::new(hf_repo.to_string(), RepoType::Model));
    if let Some(path) = new_repo.get(hf_filename) {
        return Some(path);
    }

    // 3. Check old mold models dir (backward compat — HF cached here before .hf-cache/)
    let old_cache = Cache::new(models_dir());
    let old_repo = old_cache.repo(Repo::new(hf_repo.to_string(), RepoType::Model));
    if let Some(path) = old_repo.get(hf_filename) {
        return Some(path);
    }

    // 4. Check default HF cache (~/.cache/huggingface/hub/)
    let default_cache = Cache::from_env();
    let default_repo = default_cache.repo(Repo::new(hf_repo.to_string(), RepoType::Model));
    default_repo.get(hf_filename)
}

// ── Pull and configure (shared between CLI and server) ───────────────────────

/// Download a model and save its paths to config. Returns the updated config
/// and resolved model paths. Used by both the CLI `pull` command and the
/// server's auto-pull logic.
pub async fn pull_and_configure(model: &str) -> Result<(crate::Config, ModelPaths), DownloadError> {
    use crate::config::Config;
    use crate::manifest::{find_manifest, resolve_model_name};

    let canonical = resolve_model_name(model);

    let manifest = find_manifest(&canonical).ok_or_else(|| DownloadError::UnknownModel {
        model: model.to_string(),
    })?;

    let paths = pull_model(manifest).await?;

    let mut config = Config::load_or_default();
    let model_config = manifest.to_model_config(&paths);

    // Auto-set default_model if no config existed before
    if !Config::exists_on_disk() {
        config.default_model = manifest.name.clone();
    }

    config.upsert_model(manifest.name.clone(), model_config);
    config
        .save()
        .map_err(|e| DownloadError::ConfigSave(e.to_string()))?;

    Ok((config, paths))
}

/// Download a model and save its paths to config, reporting progress via callback.
/// Same as `pull_and_configure` but uses a callback instead of indicatif bars.
pub async fn pull_and_configure_with_callback(
    model: &str,
    callback: DownloadProgressCallback,
) -> Result<(crate::Config, ModelPaths), DownloadError> {
    use crate::config::Config;
    use crate::manifest::{find_manifest, resolve_model_name};

    let canonical = resolve_model_name(model);

    let manifest = find_manifest(&canonical).ok_or_else(|| DownloadError::UnknownModel {
        model: model.to_string(),
    })?;

    let paths = pull_model_with_callback(manifest, callback).await?;

    let mut config = Config::load_or_default();
    let model_config = manifest.to_model_config(&paths);

    if !Config::exists_on_disk() {
        config.default_model = manifest.name.clone();
    }

    config.upsert_model(manifest.name.clone(), model_config);
    config
        .save()
        .map_err(|e| DownloadError::ConfigSave(e.to_string()))?;

    Ok((config, paths))
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

    #[test]
    fn download_error_unauthorized_message() {
        let err = DownloadError::Unauthorized {
            repo: "black-forest-labs/FLUX.1-schnell".to_string(),
            model: "flux-schnell:q8".to_string(),
        };
        let msg = err.to_string();
        assert!(msg.contains("Authentication required"));
        assert!(msg.contains("black-forest-labs/FLUX.1-schnell"));
        assert!(msg.contains("HF_TOKEN"));
        assert!(msg.contains("huggingface-cli login"));
        assert!(msg.contains("mold pull flux-schnell:q8"));
    }

    /// Mutex to serialize tests that mutate `HF_TOKEN` — `set_var`/`remove_var`
    /// are process-global and not thread-safe, so parallel tests race.
    static HF_TOKEN_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

    #[test]
    fn resolve_hf_token_reads_env_var() {
        let _guard = HF_TOKEN_LOCK.lock().unwrap();
        let original = std::env::var("HF_TOKEN").ok();
        std::env::set_var("HF_TOKEN", "hf_test_token_123");
        let token = resolve_hf_token();
        // Restore before asserting so we don't leak on panic
        match &original {
            Some(v) => std::env::set_var("HF_TOKEN", v),
            None => std::env::remove_var("HF_TOKEN"),
        }
        assert_eq!(token, Some("hf_test_token_123".to_string()));
    }

    #[test]
    fn resolve_hf_token_ignores_empty_env() {
        let _guard = HF_TOKEN_LOCK.lock().unwrap();
        let original = std::env::var("HF_TOKEN").ok();
        std::env::set_var("HF_TOKEN", "  ");
        let token = resolve_hf_token();
        // Restore before asserting
        match &original {
            Some(v) => std::env::set_var("HF_TOKEN", v),
            None => std::env::remove_var("HF_TOKEN"),
        }
        // Should fall through to file-based token (which may or may not exist)
        assert_ne!(token, Some("  ".to_string()));
    }

    #[test]
    fn verify_sha256_matches() {
        let dir = std::env::temp_dir().join("mold_test_sha256_match");
        let _ = std::fs::create_dir_all(&dir);
        let path = dir.join("test_file.bin");
        std::fs::write(&path, b"hello world").unwrap();
        // SHA-256 of "hello world"
        let expected = "b94d27b9934d3e08a52e52d7da7dabfac484efe37a5380ee9088f7ace2efcde9";
        assert!(verify_sha256(&path, expected).unwrap());
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn verify_sha256_mismatch() {
        let dir = std::env::temp_dir().join("mold_test_sha256_mismatch");
        let _ = std::fs::create_dir_all(&dir);
        let path = dir.join("test_file.bin");
        std::fs::write(&path, b"hello world").unwrap();
        let wrong = "0000000000000000000000000000000000000000000000000000000000000000";
        assert!(!verify_sha256(&path, wrong).unwrap());
        let _ = std::fs::remove_dir_all(&dir);
    }
}
