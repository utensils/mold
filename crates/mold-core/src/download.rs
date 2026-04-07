use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex, OnceLock};
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
        batch_bytes_downloaded: u64,
        batch_bytes_total: u64,
        batch_elapsed_ms: u64,
    },
    /// Bytes downloaded for the current file.
    FileProgress {
        filename: String,
        file_index: usize,
        bytes_downloaded: u64,
        bytes_total: u64,
        batch_bytes_downloaded: u64,
        batch_bytes_total: u64,
        batch_elapsed_ms: u64,
    },
    /// Status message (e.g. "Verifying cached files...").
    Status { message: String },
    /// A file download completed.
    FileDone {
        filename: String,
        file_index: usize,
        total_files: usize,
        batch_bytes_downloaded: u64,
        batch_bytes_total: u64,
        batch_elapsed_ms: u64,
    },
}

/// Callback type for download progress reporting.
pub type DownloadProgressCallback = Arc<dyn Fn(DownloadProgressEvent) + Send + Sync>;

/// Options controlling model pull behavior.
#[derive(Debug, Clone, Default)]
pub struct PullOptions {
    /// Skip SHA-256 verification after download (use when HF updated a file).
    pub skip_verify: bool,
}

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

    #[error("SHA-256 mismatch for {filename}\n  Expected: {expected}\n  Got:      {actual}\n\nThe corrupted file has been removed. Re-run: mold pull {model}\nIf the file was intentionally updated on HuggingFace, use: mold pull {model} --skip-verify")]
    Sha256Mismatch {
        filename: String,
        expected: String,
        actual: String,
        model: String,
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

    #[error("{0}")]
    Other(String),

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

/// Compute the SHA-256 hex digest of a file.
pub fn compute_sha256(path: &std::path::Path) -> anyhow::Result<String> {
    use sha2::{Digest, Sha256};

    let mut file = std::fs::File::open(path)?;
    let mut hasher = Sha256::new();
    std::io::copy(&mut file, &mut hasher)?;
    Ok(format!("{:x}", hasher.finalize()))
}

/// Verify the SHA-256 digest of a file against an expected hex string.
///
/// Returns `Ok(true)` when the digest matches, `Ok(false)` on mismatch.
/// Errors only on I/O failures (e.g. file not found).
pub fn verify_sha256(path: &std::path::Path, expected: &str) -> anyhow::Result<bool> {
    Ok(compute_sha256(path)? == expected)
}

// ── Pull marker file (.pulling) ──────────────────────────────────────────────

/// Relative path to a model's `.pulling` marker: `<sanitized-name>/.pulling`.
pub fn pulling_marker_rel_path(model_name: &str) -> PathBuf {
    let canonical = crate::manifest::resolve_model_name(model_name);
    PathBuf::from(canonical.replace(':', "-")).join(".pulling")
}

/// Path to the `.pulling` marker for a model under an explicit models dir.
pub fn pulling_marker_path_in(models_dir: &Path, model_name: &str) -> PathBuf {
    models_dir.join(pulling_marker_rel_path(model_name))
}

/// Path to the `.pulling` marker for a model: `<models_dir>/<sanitized-name>/.pulling`.
fn pulling_marker_path(model_name: &str) -> PathBuf {
    pulling_marker_path_in(&models_dir(), model_name)
}

/// Write a `.pulling` marker to signal an in-progress download.
fn write_pulling_marker(model_name: &str) -> Result<(), DownloadError> {
    let path = pulling_marker_path(model_name);
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent).map_err(|e| {
            DownloadError::FilePlacement(format!(
                "failed to create directory for pull marker {}: {e}",
                parent.display()
            ))
        })?;
    }
    std::fs::write(&path, model_name).map_err(|e| {
        DownloadError::FilePlacement(format!(
            "failed to write pull marker {}: {e}",
            path.display()
        ))
    })
}

/// Remove the `.pulling` marker (best-effort, ignores errors).
pub fn remove_pulling_marker(model_name: &str) {
    let path = pulling_marker_path(model_name);
    let _ = std::fs::remove_file(path);
}

/// Check whether a model has an active `.pulling` marker (incomplete download).
pub fn has_pulling_marker(model_name: &str) -> bool {
    let canonical = crate::manifest::resolve_model_name(model_name);
    pulling_marker_path(&canonical).exists()
}

/// Verify SHA-256 integrity of a downloaded file. On mismatch, deletes the
/// corrupted file and returns `Sha256Mismatch`. Respects `skip_verify`.
fn verify_file_integrity(
    clean_path: &std::path::Path,
    file: &ModelFile,
    model_name: &str,
    skip_verify: bool,
) -> Result<(), DownloadError> {
    let expected = match file.sha256 {
        Some(h) => h,
        None => return Ok(()),
    };
    if skip_verify {
        return Ok(());
    }
    match compute_sha256(clean_path) {
        Ok(actual) if actual == expected => Ok(()),
        Ok(actual) => {
            let _ = std::fs::remove_file(clean_path);
            Err(DownloadError::Sha256Mismatch {
                filename: file.hf_filename.clone(),
                expected: expected.to_string(),
                actual,
                model: model_name.to_string(),
            })
        }
        Err(e) => {
            eprintln!(
                "warning: failed to verify SHA-256 for {}: {e}",
                file.hf_filename
            );
            Ok(())
        }
    }
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
    batch_bytes_before_current: u64,
    batch_bytes_total: u64,
    batch_started_at: Instant,
    shared: Arc<Mutex<CallbackProgressState>>,
}

struct CallbackProgressState {
    accumulated: u64,
    total: u64,
    filename: String,
    last_emit: Instant,
}

impl CallbackProgress {
    fn new(
        callback: DownloadProgressCallback,
        file_index: usize,
        total_files: usize,
        batch_bytes_before_current: u64,
        batch_bytes_total: u64,
        batch_started_at: Instant,
    ) -> Self {
        Self {
            callback,
            file_index,
            total_files,
            batch_bytes_before_current,
            batch_bytes_total,
            batch_started_at,
            shared: Arc::new(Mutex::new(CallbackProgressState {
                accumulated: 0,
                total: 0,
                filename: String::new(),
                last_emit: Instant::now(),
            })),
        }
    }
}

impl Progress for CallbackProgress {
    async fn init(&mut self, size: usize, filename: &str) {
        let (fname, total) = {
            let mut shared = self
                .shared
                .lock()
                .expect("download progress mutex poisoned");
            shared.total = size as u64;
            shared.accumulated = 0;
            shared.filename = filename.to_string();
            shared.last_emit = Instant::now();
            (shared.filename.clone(), shared.total)
        };
        (self.callback)(DownloadProgressEvent::FileStart {
            filename: fname,
            file_index: self.file_index,
            total_files: self.total_files,
            size_bytes: total,
            batch_bytes_downloaded: self.batch_bytes_before_current,
            batch_bytes_total: self.batch_bytes_total,
            batch_elapsed_ms: self.batch_started_at.elapsed().as_millis() as u64,
        });
    }

    async fn update(&mut self, size: usize) {
        let mut shared = self
            .shared
            .lock()
            .expect("download progress mutex poisoned");
        shared.accumulated += size as u64;

        let now = Instant::now();
        let should_emit = now.duration_since(shared.last_emit).as_millis() >= 250
            || shared.accumulated >= shared.total;
        if !should_emit {
            return;
        }

        shared.last_emit = now;
        let filename = shared.filename.clone();
        let accumulated = shared.accumulated;
        let total = shared.total;
        drop(shared);

        (self.callback)(DownloadProgressEvent::FileProgress {
            filename,
            file_index: self.file_index,
            bytes_downloaded: accumulated,
            bytes_total: total,
            batch_bytes_downloaded: self.batch_bytes_before_current + accumulated,
            batch_bytes_total: self.batch_bytes_total,
            batch_elapsed_ms: self.batch_started_at.elapsed().as_millis() as u64,
        });
    }

    async fn finish(&mut self) {
        let (fname, total) = {
            let shared = self
                .shared
                .lock()
                .expect("download progress mutex poisoned");
            (shared.filename.clone(), shared.total)
        };
        (self.callback)(DownloadProgressEvent::FileDone {
            filename: fname,
            file_index: self.file_index,
            total_files: self.total_files,
            batch_bytes_downloaded: self.batch_bytes_before_current + total,
            batch_bytes_total: self.batch_bytes_total,
            batch_elapsed_ms: self.batch_started_at.elapsed().as_millis() as u64,
        });
    }
}

/// Sync progress adapter bridging hf-hub's sync `Progress` trait to our
/// local `indicatif::ProgressBar`.
struct SyncDownloadProgress {
    bar: ProgressBar,
    max_msg_len: usize,
    filename: String,
}

impl SyncDownloadProgress {
    fn new(bar: ProgressBar, max_msg_len: usize) -> Self {
        Self {
            bar,
            max_msg_len,
            filename: String::new(),
        }
    }
}

impl hf_hub::api::Progress for SyncDownloadProgress {
    fn init(&mut self, size: usize, filename: &str) {
        self.bar.set_length(size as u64);
        self.filename = truncate_filename(filename, self.max_msg_len);
        self.bar.set_message(self.filename.clone());
    }

    fn update(&mut self, size: usize) {
        self.bar.inc(size as u64);
    }

    fn finish(&mut self) {
        self.bar.finish_with_message(self.filename.clone());
    }
}

/// Returns `true` if the file already exists at `clean_path` with the correct
/// size and (if a SHA-256 is available) the correct digest.
///
/// **Side-effect**: if the file exists with matching size but failing integrity,
/// `verify_file_integrity` will delete the corrupted file before returning `false`.
fn is_already_placed(
    clean_path: &std::path::Path,
    file: &ModelFile,
    model_name: &str,
    skip_verify: bool,
) -> bool {
    let size_ok = clean_path
        .metadata()
        .map(|m| m.len() == file.size_bytes)
        .unwrap_or(false);
    if !size_ok {
        return false;
    }
    // Verify integrity — a same-size but corrupted file must not be accepted
    verify_file_integrity(clean_path, file, model_name, skip_verify).is_ok()
}

/// Download all files for a model manifest, returning resolved paths.
///
/// Downloads go to a hidden hf-hub cache (`.hf-cache/`) for resume/dedup support,
/// then files are hardlinked to clean paths:
/// - Transformers → `<model-name>/<filename>`
/// - Shared components → `shared/<family>/<filename>`
///
/// A `.pulling` marker file is written before downloads begin and removed on
/// success. If the pull is interrupted, the marker signals an incomplete state.
pub async fn pull_model(
    manifest: &ModelManifest,
    opts: &PullOptions,
) -> Result<ModelPaths, DownloadError> {
    write_pulling_marker(&manifest.name)?;

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
        // Skip files already at their clean path with correct size (resume after partial failure)
        let clean_rel = crate::manifest::storage_path(manifest, file);
        let clean_path = mdir.join(&clean_rel);
        if is_already_placed(&clean_path, file, &manifest.name, opts.skip_verify) {
            downloads.push((file.component, clean_path));
            continue;
        }

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
        hardlink_or_copy(&hf_path, &clean_path)?;

        verify_file_integrity(&clean_path, file, &manifest.name, opts.skip_verify)?;

        downloads.push((file.component, clean_path));
    }

    remove_pulling_marker(&manifest.name);
    paths_from_downloads(&downloads, &manifest.family).ok_or(DownloadError::MissingComponent)
}

/// Download all files for a model manifest, reporting progress via callback.
///
/// Same as `pull_model` but uses a callback instead of indicatif progress bars.
/// Suitable for server-side downloads where terminal bars are not appropriate.
pub async fn pull_model_with_callback(
    manifest: &ModelManifest,
    callback: DownloadProgressCallback,
    opts: &PullOptions,
) -> Result<ModelPaths, DownloadError> {
    write_pulling_marker(&manifest.name)?;

    let mut builder = ApiBuilder::from_env().with_cache_dir(hf_cache_dir());
    if let Some(token) = resolve_hf_token() {
        builder = builder.with_token(Some(token));
    }
    let api = builder.build()?;

    let mdir = models_dir();
    let mut downloads: Vec<(ModelComponent, PathBuf)> = Vec::new();

    // Pre-compute which files need downloading vs already cached.
    // Run in spawn_blocking because SHA-256 verification of multi-GB cached
    // files blocks the async runtime and prevents SSE event delivery.
    let manifest_clone = manifest.clone();
    let skip_verify = opts.skip_verify;
    let mdir_clone = mdir.clone();
    let cb = callback.clone();
    let file_status: Vec<bool> = tokio::task::spawn_blocking(move || {
        let total = manifest_clone.files.len();
        manifest_clone
            .files
            .iter()
            .enumerate()
            .map(|(i, file)| {
                cb(DownloadProgressEvent::Status {
                    message: format!(
                        "Verifying file [{}/{}] {}...",
                        i + 1,
                        total,
                        file.hf_filename
                    ),
                });
                let clean_path =
                    mdir_clone.join(crate::manifest::storage_path(&manifest_clone, file));
                is_already_placed(&clean_path, file, &manifest_clone.name, skip_verify)
            })
            .collect()
    })
    .await
    .map_err(|e| DownloadError::Other(format!("pre-scan task failed: {e}")))?;

    let total_bytes_to_download: u64 = manifest
        .files
        .iter()
        .zip(file_status.iter())
        .filter(|(_, &placed)| !placed)
        .map(|(file, _)| file.size_bytes)
        .sum();
    let total_files_count = manifest.files.len();
    let mut completed_bytes = 0u64;
    let batch_started_at = Instant::now();

    for (file_pos, (file, &already_placed)) in
        manifest.files.iter().zip(file_status.iter()).enumerate()
    {
        let clean_rel = crate::manifest::storage_path(manifest, file);
        let clean_path = mdir.join(&clean_rel);

        if already_placed {
            // Emit events for cached files so the TUI shows checkmarks.
            let elapsed = batch_started_at.elapsed().as_millis() as u64;
            (callback)(DownloadProgressEvent::FileStart {
                filename: file.hf_filename.clone(),
                file_index: file_pos,
                total_files: total_files_count,
                size_bytes: file.size_bytes,
                batch_bytes_downloaded: completed_bytes,
                batch_bytes_total: total_bytes_to_download,
                batch_elapsed_ms: elapsed,
            });
            (callback)(DownloadProgressEvent::FileDone {
                filename: file.hf_filename.clone(),
                file_index: file_pos,
                total_files: total_files_count,
                batch_bytes_downloaded: completed_bytes,
                batch_bytes_total: total_bytes_to_download,
                batch_elapsed_ms: elapsed,
            });
            downloads.push((file.component, clean_path));
            continue;
        }

        let progress = CallbackProgress::new(
            callback.clone(),
            file_pos,
            total_files_count,
            completed_bytes,
            total_bytes_to_download,
            batch_started_at,
        );
        let hf_path = download_file(&api, file, progress, &manifest.name).await?;

        hardlink_or_copy(&hf_path, &clean_path)?;

        verify_file_integrity(&clean_path, file, &manifest.name, opts.skip_verify)?;

        downloads.push((file.component, clean_path));
        completed_bytes += file.size_bytes;
    }

    remove_pulling_marker(&manifest.name);
    paths_from_downloads(&downloads, &manifest.family).ok_or(DownloadError::MissingComponent)
}

/// Download all files for a utility model (no ModelPaths, no config writing).
///
/// Used for models like qwen3-expand that are not diffusion models and don't
/// have a VAE. Files are downloaded and placed at their standard storage paths.
async fn pull_model_files_only(
    manifest: &ModelManifest,
    opts: &PullOptions,
) -> Result<(), DownloadError> {
    write_pulling_marker(&manifest.name)?;

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

    for file in &manifest.files {
        // Skip files already at their clean path with correct size (resume after partial failure)
        let clean_rel = crate::manifest::storage_path(manifest, file);
        let clean_path = mdir.join(&clean_rel);
        if is_already_placed(&clean_path, file, &manifest.name, opts.skip_verify) {
            continue;
        }

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

        hardlink_or_copy(&hf_path, &clean_path)?;

        verify_file_integrity(&clean_path, file, &manifest.name, opts.skip_verify)?;
    }

    remove_pulling_marker(&manifest.name);
    Ok(())
}

/// Download all files for a utility model, reporting progress via callback.
async fn pull_model_files_only_with_callback(
    manifest: &ModelManifest,
    callback: DownloadProgressCallback,
    opts: &PullOptions,
) -> Result<(), DownloadError> {
    write_pulling_marker(&manifest.name)?;

    let mut builder = ApiBuilder::from_env().with_cache_dir(hf_cache_dir());
    if let Some(token) = resolve_hf_token() {
        builder = builder.with_token(Some(token));
    }
    let api = builder.build()?;

    let mdir = models_dir();

    let manifest_clone = manifest.clone();
    let skip_verify = opts.skip_verify;
    let mdir_clone = mdir.clone();
    let cb = callback.clone();
    let file_status: Vec<bool> = tokio::task::spawn_blocking(move || {
        let total = manifest_clone.files.len();
        manifest_clone
            .files
            .iter()
            .enumerate()
            .map(|(i, file)| {
                cb(DownloadProgressEvent::Status {
                    message: format!(
                        "Verifying file [{}/{}] {}...",
                        i + 1,
                        total,
                        file.hf_filename
                    ),
                });
                let clean_path =
                    mdir_clone.join(crate::manifest::storage_path(&manifest_clone, file));
                is_already_placed(&clean_path, file, &manifest_clone.name, skip_verify)
            })
            .collect()
    })
    .await
    .map_err(|e| DownloadError::Other(format!("pre-scan task failed: {e}")))?;
    let total_bytes_to_download: u64 = manifest
        .files
        .iter()
        .zip(file_status.iter())
        .filter(|(_, &placed)| !placed)
        .map(|(file, _)| file.size_bytes)
        .sum();
    let total_files_count = manifest.files.len();
    let mut completed_bytes = 0u64;
    let batch_started_at = Instant::now();

    for (file_pos, (file, &already_placed)) in
        manifest.files.iter().zip(file_status.iter()).enumerate()
    {
        let clean_rel = crate::manifest::storage_path(manifest, file);
        let clean_path = mdir.join(&clean_rel);

        if already_placed {
            let elapsed = batch_started_at.elapsed().as_millis() as u64;
            (callback)(DownloadProgressEvent::FileStart {
                filename: file.hf_filename.clone(),
                file_index: file_pos,
                total_files: total_files_count,
                size_bytes: file.size_bytes,
                batch_bytes_downloaded: completed_bytes,
                batch_bytes_total: total_bytes_to_download,
                batch_elapsed_ms: elapsed,
            });
            (callback)(DownloadProgressEvent::FileDone {
                filename: file.hf_filename.clone(),
                file_index: file_pos,
                total_files: total_files_count,
                batch_bytes_downloaded: completed_bytes,
                batch_bytes_total: total_bytes_to_download,
                batch_elapsed_ms: elapsed,
            });
            continue;
        }

        let progress = CallbackProgress::new(
            callback.clone(),
            file_pos,
            total_files_count,
            completed_bytes,
            total_bytes_to_download,
            batch_started_at,
        );

        let hf_path = download_file(&api, file, progress, &manifest.name).await?;

        hardlink_or_copy(&hf_path, &clean_path)?;

        verify_file_integrity(&clean_path, file, &manifest.name, opts.skip_verify)?;
        completed_bytes += file.size_bytes;
    }

    remove_pulling_marker(&manifest.name);
    Ok(())
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

    let mut builder = ApiBuilder::from_env()
        .with_cache_dir(hf_cache_dir())
        .with_progress(false);
    if let Some(token) = resolve_hf_token() {
        builder = builder.with_token(Some(token));
    }
    let api = builder
        .build()
        .map_err(|e| DownloadError::SyncApiSetup(e.to_string()))?;
    let repo = api.repo(Repo::new(hf_repo.to_string(), RepoType::Model));
    let msg_width = filename_column_width();
    let bar_style = ProgressStyle::with_template(&format!(
        "  {{msg:<{msg_width}}} [{{bar:30.cyan/dim}}] {{bytes}}/{{total_bytes}} ({{bytes_per_sec}}, {{eta}})"
    ))
    .unwrap()
    .progress_chars("━╸─");
    let bar = ProgressBar::new(0);
    bar.set_style(bar_style);
    bar.set_message(truncate_filename(hf_filename, msg_width));
    let progress = SyncDownloadProgress::new(bar, msg_width);
    let hf_path = repo
        .download_with_progress(hf_filename, progress)
        .map_err(|e| {
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
pub async fn pull_and_configure(
    model: &str,
    opts: &PullOptions,
) -> Result<(crate::Config, Option<ModelPaths>), DownloadError> {
    use crate::config::Config;
    use crate::manifest::{find_manifest, resolve_model_name};

    let canonical = resolve_model_name(model);

    let manifest = find_manifest(&canonical).ok_or_else(|| DownloadError::UnknownModel {
        model: model.to_string(),
    })?;

    // Utility models (e.g., qwen3-expand) have no VAE and don't need config entries.
    if manifest.is_utility() {
        pull_model_files_only(manifest, opts).await?;
        let config = Config::load_or_default();
        return Ok((config, None));
    }

    // Upscaler models have a single weights file (no VAE, no encoders).
    // Download files and create a minimal config entry with the weights path.
    if manifest.is_upscaler() {
        pull_model_files_only(manifest, opts).await?;

        // Resolve the weights path from the manifest storage path
        let mdir = models_dir();
        let weights_file = manifest
            .files
            .iter()
            .find(|f| f.component == crate::manifest::ModelComponent::Upscaler)
            .ok_or(DownloadError::MissingComponent)?;
        let weights_path = mdir.join(crate::manifest::storage_path(manifest, weights_file));

        let mut config = Config::load_or_default();
        let model_config = crate::config::ModelConfig {
            transformer: Some(weights_path.to_string_lossy().to_string()),
            family: Some("upscaler".to_string()),
            ..Default::default()
        };
        config.upsert_model(manifest.name.clone(), model_config);
        config
            .save()
            .map_err(|e| DownloadError::ConfigSave(e.to_string()))?;

        return Ok((config, None));
    }

    let paths = pull_model(manifest, opts).await?;

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

    Ok((config, Some(paths)))
}

/// Download a model and save its paths to config, reporting progress via callback.
/// Same as `pull_and_configure` but uses a callback instead of indicatif bars.
pub async fn pull_and_configure_with_callback(
    model: &str,
    callback: DownloadProgressCallback,
    opts: &PullOptions,
) -> Result<(crate::Config, Option<ModelPaths>), DownloadError> {
    use crate::config::Config;
    use crate::manifest::{find_manifest, resolve_model_name};

    let canonical = resolve_model_name(model);

    let manifest = find_manifest(&canonical).ok_or_else(|| DownloadError::UnknownModel {
        model: model.to_string(),
    })?;

    // Utility models (e.g., qwen3-expand) have no VAE and don't need config entries.
    if manifest.is_utility() {
        pull_model_files_only_with_callback(manifest, callback, opts).await?;
        let config = Config::load_or_default();
        return Ok((config, None));
    }

    // Upscaler models: download files, create minimal config with weights path.
    if manifest.is_upscaler() {
        pull_model_files_only_with_callback(manifest, callback, opts).await?;

        let mdir = models_dir();
        let weights_file = manifest
            .files
            .iter()
            .find(|f| f.component == crate::manifest::ModelComponent::Upscaler)
            .ok_or(DownloadError::MissingComponent)?;
        let weights_path = mdir.join(crate::manifest::storage_path(manifest, weights_file));

        let mut config = Config::load_or_default();
        let model_config = crate::config::ModelConfig {
            transformer: Some(weights_path.to_string_lossy().to_string()),
            family: Some("upscaler".to_string()),
            ..Default::default()
        };
        config.upsert_model(manifest.name.clone(), model_config);
        config
            .save()
            .map_err(|e| DownloadError::ConfigSave(e.to_string()))?;

        return Ok((config, None));
    }

    let paths = pull_model_with_callback(manifest, callback, opts).await?;

    let mut config = Config::load_or_default();
    let model_config = manifest.to_model_config(&paths);

    if !Config::exists_on_disk() {
        config.default_model = manifest.name.clone();
    }

    config.upsert_model(manifest.name.clone(), model_config);
    config
        .save()
        .map_err(|e| DownloadError::ConfigSave(e.to_string()))?;

    Ok((config, Some(paths)))
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

    #[tokio::test]
    async fn callback_progress_clones_share_accumulated_bytes() {
        let events = Arc::new(Mutex::new(Vec::new()));
        let events_for_cb = events.clone();
        let callback: DownloadProgressCallback = Arc::new(move |event| {
            events_for_cb
                .lock()
                .expect("events mutex poisoned")
                .push(event);
        });

        let mut progress = CallbackProgress::new(callback, 1, 3, 1_000, 10_000, Instant::now());
        progress.init(1_024, "weights.safetensors").await;

        let mut chunk_a = progress.clone();
        let mut chunk_b = progress.clone();
        chunk_a.update(512).await;
        chunk_b.update(512).await;
        progress.finish().await;

        let events = events.lock().expect("events mutex poisoned");
        assert!(events.iter().any(|event| matches!(
            event,
            DownloadProgressEvent::FileProgress {
                bytes_downloaded: 1_024,
                bytes_total: 1_024,
                batch_bytes_downloaded: 2_024,
                ..
            }
        )));
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
    fn compute_sha256_correct_digest() {
        let dir = std::env::temp_dir().join("mold_test_sha256_compute");
        let _ = std::fs::create_dir_all(&dir);
        let path = dir.join("test_file.bin");
        std::fs::write(&path, b"hello world").unwrap();
        let digest = compute_sha256(&path).unwrap();
        assert_eq!(
            digest,
            "b94d27b9934d3e08a52e52d7da7dabfac484efe37a5380ee9088f7ace2efcde9"
        );
        let _ = std::fs::remove_dir_all(&dir);
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

    #[test]
    fn verify_file_integrity_deletes_on_mismatch() {
        use crate::manifest::{ModelComponent, ModelFile};
        let dir = std::env::temp_dir().join("mold_test_integrity_mismatch");
        let _ = std::fs::create_dir_all(&dir);
        let path = dir.join("corrupted.bin");
        std::fs::write(&path, b"corrupted data").unwrap();

        let file = ModelFile {
            hf_repo: "test/repo".to_string(),
            hf_filename: "corrupted.bin".to_string(),
            component: ModelComponent::Transformer,
            size_bytes: 14,
            gated: false,
            sha256: Some("0000000000000000000000000000000000000000000000000000000000000000"),
        };

        let result = verify_file_integrity(&path, &file, "test-model:q8", false);
        assert!(result.is_err());
        assert!(matches!(
            result.unwrap_err(),
            DownloadError::Sha256Mismatch { .. }
        ),);
        // File should be deleted
        assert!(!path.exists());
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn verify_file_integrity_skip_verify_ignores_mismatch() {
        use crate::manifest::{ModelComponent, ModelFile};
        let dir = std::env::temp_dir().join("mold_test_integrity_skip");
        let _ = std::fs::create_dir_all(&dir);
        let path = dir.join("file.bin");
        std::fs::write(&path, b"some data").unwrap();

        let file = ModelFile {
            hf_repo: "test/repo".to_string(),
            hf_filename: "file.bin".to_string(),
            component: ModelComponent::Transformer,
            size_bytes: 9,
            gated: false,
            sha256: Some("0000000000000000000000000000000000000000000000000000000000000000"),
        };

        let result = verify_file_integrity(&path, &file, "test-model:q8", true);
        assert!(result.is_ok());
        // File should still exist
        assert!(path.exists());
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn verify_file_integrity_no_hash_is_ok() {
        use crate::manifest::{ModelComponent, ModelFile};
        let dir = std::env::temp_dir().join("mold_test_integrity_nohash");
        let _ = std::fs::create_dir_all(&dir);
        let path = dir.join("file.bin");
        std::fs::write(&path, b"data").unwrap();

        let file = ModelFile {
            hf_repo: "test/repo".to_string(),
            hf_filename: "file.bin".to_string(),
            component: ModelComponent::Transformer,
            size_bytes: 4,
            gated: false,
            sha256: None,
        };

        assert!(verify_file_integrity(&path, &file, "test:q8", false).is_ok());
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn pulling_marker_roundtrip() {
        let dir = std::env::temp_dir().join("mold_test_marker_roundtrip");
        let _ = std::fs::create_dir_all(&dir);
        let marker = dir.join(".pulling");

        // Write
        std::fs::write(&marker, "test-model:q8").unwrap();
        assert!(marker.exists());

        // Remove
        let _ = std::fs::remove_file(&marker);
        assert!(!marker.exists());

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn sha256_mismatch_error_message() {
        let err = DownloadError::Sha256Mismatch {
            filename: "transformer.gguf".to_string(),
            expected: "aaa".to_string(),
            actual: "bbb".to_string(),
            model: "flux-dev:q8".to_string(),
        };
        let msg = err.to_string();
        assert!(msg.contains("SHA-256 mismatch"));
        assert!(msg.contains("transformer.gguf"));
        assert!(msg.contains("mold pull flux-dev:q8"));
        assert!(msg.contains("--skip-verify"));
    }
}
