use std::collections::HashMap;
use std::future::Future;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex, OnceLock, Weak};
use std::time::Instant;

use console::Term;
use hf_hub::api::tokio::{Api, ApiBuilder, ApiError, Progress};
use hf_hub::{Cache, Repo, RepoType};
use indicatif::{MultiProgress, ProgressBar, ProgressDrawTarget, ProgressStyle};
use thiserror::Error;

use crate::manifest::{paths_from_downloads, ModelComponent, ModelFile, ModelManifest};
use crate::ModelPaths;

fn hf_file_repo(repo_id: &str, filename: &str) -> Repo {
    if let Some(revision) = crate::minimax_h3::file_revision(repo_id, filename) {
        Repo::with_revision(repo_id.to_string(), RepoType::Model, revision.to_string())
    } else {
        Repo::new(repo_id.to_string(), RepoType::Model)
    }
}

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
    #[error(transparent)]
    ModelActivation(#[from] crate::ModelActivationError),

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

    #[error(
        "Insufficient disk space for {model}: {required_bytes} bytes remain to download but only {available_bytes} bytes are available at {path}"
    )]
    InsufficientDiskSpace {
        model: String,
        required_bytes: u64,
        available_bytes: u64,
        path: PathBuf,
    },

    #[error("Downloaded {model}, but its LTX-2.5 asset contract failed qualification: {message}")]
    QualificationFailed { model: String, message: String },

    /// A file in the manifest is published under terms mold cannot accept on
    /// the user's behalf. Carries the full actionable message so automatic and
    /// server-side pulls surface the same wording the CLI does.
    #[error("{message}")]
    LicenseNotAccepted { license_id: String, message: String },

    #[error("{0}")]
    Other(String),

    #[error("IO error during file placement: {0}")]
    FilePlacement(String),

    #[error("Unknown model '{model}'. No manifest found.")]
    UnknownModel { model: String },

    #[error("Failed to save config: {0}")]
    ConfigSave(String),

    #[error("Recipe destination path '{dest}' escapes the per-recipe subdirectory")]
    RecipePathTraversal { dest: String },

    #[error("Civitai download requires CIVITAI_TOKEN.\n\n  1. Create a token at: https://civitai.com/user/account (Add API Key)\n  2. Set: export CIVITAI_TOKEN=...\n  3. Retry: mold pull {id}")]
    MissingCivitaiToken { id: String },

    #[error("Recipe HTTP fetch failed for {url}: status {status}{}", .body.as_ref().map(|b| format!(" — {b}")).unwrap_or_default())]
    RecipeHttp {
        url: String,
        status: u16,
        body: Option<String>,
    },

    #[error("Recipe transport error for {url}: {source}")]
    RecipeTransport {
        url: String,
        #[source]
        source: reqwest::Error,
    },
}

/// Does a GGUF file's header contain the given tensor name?
///
/// Scans the first 4 MiB of the file — enough to cover tensor_infos for any
/// real FLUX GGUF (~800 tensors × ~100 B per entry). Tensor names are stored
/// as UTF-8 in the header, so a substring search is reliable: the needle is
/// length-prefixed by a u64, so accidental coincidences in the scanned region
/// would need to match a 31+ character needle exactly.
fn gguf_header_contains_tensor(path: &std::path::Path, needle: &str) -> bool {
    use std::io::Read;
    let Ok(mut f) = std::fs::File::open(path) else {
        return false;
    };
    let mut buf = vec![0u8; 4 * 1024 * 1024];
    let Ok(n) = f.read(&mut buf) else {
        return false;
    };
    buf.truncate(n);
    if buf.len() < 4 || &buf[..4] != b"GGUF" {
        return false;
    }
    buf.windows(needle.len()).any(|w| w == needle.as_bytes())
}

/// Decide whether to emit the pull-time "city96-format, needs reference" warning.
///
/// Pure logic, no process-global state — `models_dir` is always passed in so
/// tests can use a temp dir. Returns `Some(message)` when the warning should
/// fire, `None` otherwise.
fn flux_reference_warning(manifest: &ModelManifest, models_dir: &Path) -> Option<String> {
    if manifest.family != "flux" {
        return None;
    }
    let xformer_file = manifest.files.iter().find(|f| {
        f.component == ModelComponent::Transformer
            && f.hf_filename.to_lowercase().ends_with(".gguf")
    })?;
    let xformer_path = models_dir.join(crate::manifest::storage_path(manifest, xformer_file));
    if !xformer_path.exists() {
        return None;
    }
    // img_in is present in schnell and in complete dev GGUFs; missing from city96-format
    if gguf_header_contains_tensor(&xformer_path, "img_in.weight") {
        return None;
    }

    let needs_guidance = !manifest.defaults.is_schnell;
    let reference_candidates: &[&str] = if needs_guidance {
        &["flux-dev:q8", "flux-dev:q6", "flux-dev:q4"]
    } else {
        &[
            "flux-dev:q8",
            "flux-dev:q6",
            "flux-dev:q4",
            "flux-schnell:q8",
            "flux-schnell:q4",
        ]
    };
    let have_reference = reference_candidates.iter().any(|name| {
        let Some(m) = crate::manifest::find_manifest(name) else {
            return false;
        };
        let Some(xf) = m
            .files
            .iter()
            .find(|f| f.component == ModelComponent::Transformer)
        else {
            return false;
        };
        let path = models_dir.join(crate::manifest::storage_path(m, xf));
        path.exists()
            && gguf_header_contains_tensor(&path, "img_in.weight")
            && (!needs_guidance
                || gguf_header_contains_tensor(&path, "guidance_in.in_layer.weight"))
    });
    if have_reference {
        return None;
    }

    let fix_cmd = if needs_guidance {
        "mold pull flux-dev:q8"
    } else {
        "mold pull flux-dev:q8 (or flux-schnell:q8)"
    };
    Some(format!(
        "Heads up: {} is a city96-format GGUF — it ships only the diffusion blocks. \
         FLUX input embedding layers{} must be patched from a separate reference \
         model at load time, and none is downloaded yet. Run `{fix_cmd}` before \
         generating with {}.",
        xformer_file.hf_filename,
        if needs_guidance {
            " (including dev-only guidance_in)"
        } else {
            ""
        },
        manifest.name,
    ))
}

/// Warn the operator if the downloaded transformer is a city96-format GGUF
/// that will need an additional reference pull before inference will run.
///
/// Community FLUX fine-tune GGUFs ship only the diffusion blocks; their input
/// embedding layers (img_in / time_in / vector_in / guidance_in) are inherited
/// from base flux-dev and must be patched in from a locally-downloaded
/// reference. This check surfaces the dependency at pull time so users don't
/// discover it on the first generation attempt.
fn warn_if_flux_gguf_needs_reference(
    manifest: &ModelManifest,
    callback: Option<&DownloadProgressCallback>,
) {
    let Some(msg) = flux_reference_warning(manifest, &models_dir()) else {
        return;
    };
    if let Some(cb) = callback {
        cb(DownloadProgressEvent::Status {
            message: format!("⚠ {msg}"),
        });
    } else {
        let _ = console::Term::stderr().write_line(&format!("\n⚠ {msg}\n"));
    }
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

fn resolve_hf_token_for(explicit_token: Option<&str>) -> Option<String> {
    explicit_token
        .map(str::trim)
        .filter(|token| !token.is_empty())
        .map(str::to_string)
        .or_else(resolve_hf_token)
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
/// Comparison is hex-case-insensitive — Civitai's API publishes uppercase
/// hashes and `compute_sha256` produces lowercase, so a literal `==`
/// would false-mismatch on bit-identical files.
///
/// Returns `Ok(true)` when the digest matches, `Ok(false)` on mismatch.
/// Errors only on I/O failures (e.g. file not found).
pub fn verify_sha256(path: &std::path::Path, expected: &str) -> anyhow::Result<bool> {
    Ok(compute_sha256(path)?.eq_ignore_ascii_case(expected))
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

/// Filename suffix for the "this file is fully written and integrity-checked"
/// sidecar marker: `model.safetensors` → `model.safetensors.sha256-verified`.
///
/// The marker is written by [`verify_file_integrity`] on a successful pull
/// (or by the post-startup backfill sweep for pre-marker installs). Two
/// downstream consumers depend on it:
///
/// 1. `cleanup_partials_in_dir` (in `mold-server`) preserves any file that
///    has a sibling marker — those are known-good and survive cancel/retry.
/// 2. `Config::manifest_files_exist` requires the marker before reporting a
///    model as "downloaded" — eliminates the existence-only race that let
///    truncated files masquerade as complete installs.
pub const SHA256_VERIFIED_SUFFIX: &str = ".sha256-verified";

/// Minimum interval between `FileProgress` events emitted by the recipe-pull
/// path (`fetch_recipe_inner`). The manifest-pull path's `CallbackProgress`
/// throttles to the same cadence; this constant keeps them in sync. 250ms
/// matches a comfortable UI refresh rate (~4 Hz) without flooding SSE
/// subscribers when downloads run at multi-MB/s chunk rates.
pub const RECIPE_PROGRESS_THROTTLE_MS: u64 = 250;

/// Build the marker path for a downloaded file. `model.safetensors` →
/// `model.safetensors.sha256-verified` in the same directory.
pub fn sha256_marker_path(path: &Path) -> PathBuf {
    let mut marker = path.as_os_str().to_os_string();
    marker.push(SHA256_VERIFIED_SUFFIX);
    PathBuf::from(marker)
}

/// True iff `<path>.sha256-verified` exists.
pub fn has_sha256_marker(path: &Path) -> bool {
    sha256_marker_path(path).exists()
}

/// Atomically write the `.sha256-verified` marker for `path` recording the
/// computed digest. Atomic via tempfile-then-rename so a crash mid-write
/// never leaves a half-populated marker (which would otherwise read as a
/// successfully-installed file).
pub fn write_sha256_marker(path: &Path, digest: &str) -> std::io::Result<()> {
    let marker = sha256_marker_path(path);
    let tmp = marker.with_extension(format!("sha256-verified.tmp.{}", std::process::id()));
    std::fs::write(&tmp, format!("{digest}\n"))?;
    std::fs::rename(&tmp, &marker)
}

/// Verify SHA-256 integrity of a downloaded file and write the
/// `.sha256-verified` marker on success.
///
/// - Manifest declares `sha256`: compute, compare, on match write marker
///   (containing the verified digest); on mismatch delete the corrupted
///   file and return `Sha256Mismatch`.
/// - Manifest does not declare a hash: still compute and write the marker
///   so the file is positively attested as "fully written." This is the
///   load-bearing change for the gallery race — `Config::manifest_files_exist`
///   consults marker presence, so unmarked-but-present files no longer
///   appear in the available-models list.
/// - `skip_verify = true`: respected from the original contract — no read,
///   no marker. The caller has explicitly asked us to trust the bytes.
fn verify_file_integrity(
    clean_path: &std::path::Path,
    file: &ModelFile,
    model_name: &str,
    skip_verify: bool,
) -> Result<(), DownloadError> {
    if skip_verify {
        return Ok(());
    }
    let actual = match pinned_file_digest(clean_path) {
        Ok(d) => d,
        Err(e) => {
            // I/O failure during hashing — log and move on without a marker.
            // The downstream `manifest_files_exist` check will report the
            // file incomplete, prompting a retry rather than a silent pass.
            eprintln!(
                "warning: failed to verify SHA-256 for {}: {e}",
                file.hf_filename
            );
            return Ok(());
        }
    };
    if let Some(expected) = file.sha256 {
        if !actual.eq_ignore_ascii_case(expected) {
            let _ = std::fs::remove_file(clean_path);
            return Err(DownloadError::Sha256Mismatch {
                filename: file.hf_filename.clone(),
                expected: expected.to_string(),
                actual,
                model: model_name.to_string(),
            });
        }
    }
    if let Err(e) = write_sha256_marker(clean_path, &actual) {
        // Marker-write failure isn't fatal to this attempt — the file is
        // good. But it does mean the next `manifest_files_exist` check will
        // report incomplete. Log loudly so users can see why.
        eprintln!(
            "warning: failed to write .sha256-verified marker for {}: {e}",
            file.hf_filename
        );
    }
    Ok(())
}

/// Identity of an open regular file, as read from the descriptor that is about
/// to be hashed.
///
/// Keying the digest memo on this — and reading it via `fstat` on the retained
/// descriptor rather than by re-`stat`ing the path — is what makes the memo
/// sound: a cache hit means "this exact inode, at this exact size and these
/// exact timestamps, hashed to D earlier in this process", not "a file with
/// this name did".
///
/// `ctime` is in the tuple deliberately. `mtime` alone is forgeable with
/// `utimes(2)` by anyone who can write the file, which in the group-writable
/// models roots the model-storage invariant supports is not the owner alone.
/// `ctime` updates on every inode change and cannot be set directly at all, so
/// an in-place rewrite that restores size and mtime still misses the cache.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq, serde::Deserialize, serde::Serialize)]
struct PinnedFileIdentity {
    len: u64,
    platform: [u64; 9],
}

fn pinned_file_identity(file: &std::fs::File) -> std::io::Result<PinnedFileIdentity> {
    let metadata = file.metadata()?;
    #[cfg(unix)]
    {
        use std::os::unix::fs::MetadataExt;
        Ok(PinnedFileIdentity {
            len: metadata.len(),
            platform: [
                metadata.dev(),
                metadata.ino(),
                u64::from(metadata.mode()),
                u64::from(metadata.uid()),
                u64::from(metadata.gid()),
                metadata.mtime() as u64,
                metadata.mtime_nsec() as u64,
                metadata.ctime() as u64,
                metadata.ctime_nsec() as u64,
            ],
        })
    }
    #[cfg(not(unix))]
    {
        // No inode identity is available here, so the memo degrades to
        // (size, mtime). That is weaker, never wrong-in-the-unsafe-direction
        // for the pinned check itself — the digest is still of real bytes read
        // through a retained descriptor — but it can hold a stale entry for a
        // same-size same-mtime replacement. Unix is every shipped server
        // target; if Windows ever becomes one, key this on the file index from
        // `GetFileInformationByHandleEx` as `execution_plan` already does.
        let mtime = metadata
            .modified()
            .ok()
            .and_then(|time| time.duration_since(std::time::UNIX_EPOCH).ok());
        Ok(PinnedFileIdentity {
            len: metadata.len(),
            platform: [
                mtime.map_or(0, |value| value.as_secs()),
                mtime.map_or(0, |value| u64::from(value.subsec_nanos())),
                0,
                0,
                0,
                0,
                0,
                0,
                0,
            ],
        })
    }
}

type PinnedDigestCache = std::sync::Mutex<std::collections::HashMap<PinnedFileIdentity, String>>;
type PinnedDigestFlights = std::sync::Mutex<
    std::collections::HashMap<PinnedFileIdentity, std::sync::Arc<std::sync::Mutex<()>>>,
>;

fn pinned_digest_cache() -> &'static PinnedDigestCache {
    static CACHE: std::sync::OnceLock<PinnedDigestCache> = std::sync::OnceLock::new();
    CACHE.get_or_init(|| std::sync::Mutex::new(std::collections::HashMap::new()))
}

fn pinned_digest_flights() -> &'static PinnedDigestFlights {
    static FLIGHTS: std::sync::OnceLock<PinnedDigestFlights> = std::sync::OnceLock::new();
    FLIGHTS.get_or_init(|| std::sync::Mutex::new(std::collections::HashMap::new()))
}

struct PinnedDigestFlightCleanup {
    identity: PinnedFileIdentity,
    flight: std::sync::Arc<std::sync::Mutex<()>>,
}

impl Drop for PinnedDigestFlightCleanup {
    fn drop(&mut self) {
        let mut flights = pinned_digest_flights()
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        let removable = flights.get(&self.identity).is_some_and(|held| {
            std::sync::Arc::ptr_eq(held, &self.flight)
                // map + caller-local `flight` + this cleanup guard
                && std::sync::Arc::strong_count(held) == 3
        });
        if removable {
            flights.remove(&self.identity);
        }
    }
}

#[derive(serde::Deserialize, serde::Serialize)]
struct DurablePinnedDigest {
    schema: u32,
    path: PathBuf,
    identity: PinnedFileIdentity,
    sha256: String,
}

const DURABLE_PINNED_DIGEST_SCHEMA: u32 = 1;
const DURABLE_PINNED_DIGEST_DIR: &str = ".artifact-attestations-v1";
const DURABLE_PINNED_DIGEST_DIR_ENV: &str = "MOLD_ARTIFACT_ATTESTATIONS_DIR";
const MAX_DURABLE_PINNED_DIGEST_BYTES: u64 = 16 * 1024;

#[cfg(unix)]
fn directory_protects_entries(path: &Path) -> bool {
    use std::os::unix::fs::MetadataExt;
    let Ok(metadata) = std::fs::symlink_metadata(path) else {
        return false;
    };
    if !metadata.is_dir() || metadata.file_type().is_symlink() {
        return false;
    }
    // SAFETY: geteuid takes no arguments and cannot fail.
    let euid = unsafe { libc::geteuid() };
    let mode = metadata.mode();
    if mode & 0o1000 != 0 {
        metadata.uid() == euid || metadata.uid() == 0
    } else {
        (metadata.uid() == euid || metadata.uid() == 0) && mode & 0o022 == 0
    }
}

#[cfg(unix)]
fn private_attestation_dir(create: bool) -> Option<PathBuf> {
    let configured = std::env::var_os(DURABLE_PINNED_DIGEST_DIR_ENV)
        .filter(|path| !path.is_empty())
        .map(PathBuf::from);
    let dir = if let Some(dir) = configured.as_deref() {
        private_attestation_dir_exact_at(dir, create)
    } else {
        private_attestation_dir_at(&crate::Config::mold_dir()?, create)
    };
    if create && dir.is_none() {
        static WARNED: std::sync::Once = std::sync::Once::new();
        WARNED.call_once(|| {
            let attempted = configured
                .as_deref()
                .map(Path::to_path_buf)
                .or_else(|| {
                    crate::Config::mold_dir().map(|home| home.join(DURABLE_PINNED_DIGEST_DIR))
                });
            tracing::warn!(
                path = ?attempted,
                env = DURABLE_PINNED_DIGEST_DIR_ENV,
                "persistent artifact attestations are unavailable; unchanged pinned models will be rehashed after restart"
            );
        });
    }
    dir
}

#[cfg(unix)]
fn private_attestation_dir_at(mold_dir: &Path, create: bool) -> Option<PathBuf> {
    private_attestation_dir_exact_at(&mold_dir.join(DURABLE_PINNED_DIGEST_DIR), create)
}

#[cfg(unix)]
fn private_attestation_dir_exact_at(dir: &Path, create: bool) -> Option<PathBuf> {
    use std::os::unix::fs::{DirBuilderExt, MetadataExt};

    // The containing directory is the authority boundary. A 0700 attestation
    // store in a group-writable, non-sticky parent can be renamed and replaced.
    // The store itself need not live under MOLD_HOME: shared installations can
    // point at service-private state with MOLD_ARTIFACT_ATTESTATIONS_DIR.
    if !dir.parent().is_some_and(directory_protects_entries) {
        return None;
    }
    if !dir.exists() && create {
        let mut builder = std::fs::DirBuilder::new();
        builder.mode(0o700);
        if builder.create(dir).is_err() && !dir.is_dir() {
            return None;
        }
    }
    let metadata = std::fs::symlink_metadata(dir).ok()?;
    // SAFETY: geteuid takes no arguments and cannot fail.
    let euid = unsafe { libc::geteuid() };
    (metadata.is_dir()
        && !metadata.file_type().is_symlink()
        && metadata.uid() == euid
        && metadata.mode() & 0o077 == 0)
        .then(|| dir.to_path_buf())
}

#[cfg(not(unix))]
fn private_attestation_dir(_create: bool) -> Option<PathBuf> {
    // A DACL proof equivalent to the Unix owner/mode policy is not implemented.
    // Falling back to the process cache preserves authentication correctness.
    None
}

fn absolute_pinned_path(path: &Path) -> anyhow::Result<PathBuf> {
    if path.is_absolute() {
        Ok(path.to_path_buf())
    } else {
        Ok(std::env::current_dir()?.join(path))
    }
}

fn durable_pinned_digest_path(dir: &Path, path: &Path) -> PathBuf {
    use sha2::{Digest, Sha256};
    let mut hash = Sha256::new();
    hash.update(path.as_os_str().as_encoded_bytes());
    dir.join(format!("{:x}.json", hash.finalize()))
}

fn read_durable_pinned_digest(path: &Path, identity: PinnedFileIdentity) -> Option<String> {
    let dir = private_attestation_dir(false)?;
    read_durable_pinned_digest_from_dir(&dir, path, identity)
}

fn read_durable_pinned_digest_from_dir(
    dir: &Path,
    path: &Path,
    identity: PinnedFileIdentity,
) -> Option<String> {
    let absolute = absolute_pinned_path(path).ok()?;
    let record_path = durable_pinned_digest_path(dir, &absolute);
    let file = crate::secure_file::open_regular_file_no_follow(&record_path).ok()?;
    if file.metadata().ok()?.len() > MAX_DURABLE_PINNED_DIGEST_BYTES {
        return None;
    }
    #[cfg(unix)]
    {
        use std::os::unix::fs::MetadataExt;
        let metadata = file.metadata().ok()?;
        // SAFETY: geteuid takes no arguments and cannot fail.
        let euid = unsafe { libc::geteuid() };
        if metadata.uid() != euid || metadata.mode() & 0o077 != 0 || metadata.nlink() != 1 {
            return None;
        }
    }
    let record: DurablePinnedDigest = serde_json::from_reader(file).ok()?;
    (record.schema == DURABLE_PINNED_DIGEST_SCHEMA
        && record.path == absolute
        && record.identity == identity
        && record.sha256.len() == 64
        && record.sha256.bytes().all(|byte| byte.is_ascii_hexdigit()))
    .then(|| record.sha256.to_ascii_lowercase())
}

#[cfg(unix)]
fn write_durable_pinned_digest(
    path: &Path,
    identity: PinnedFileIdentity,
    sha256: &str,
) -> std::io::Result<()> {
    use std::io::Write;
    use std::os::unix::fs::OpenOptionsExt;

    let Some(dir) = private_attestation_dir(true) else {
        return Ok(());
    };
    let absolute = absolute_pinned_path(path).map_err(std::io::Error::other)?;
    let target = durable_pinned_digest_path(&dir, &absolute);
    let temp = dir.join(format!(
        ".tmp-{}-{}",
        std::process::id(),
        uuid::Uuid::new_v4()
    ));
    let mut file = std::fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .mode(0o600)
        .open(&temp)?;
    let record = DurablePinnedDigest {
        schema: DURABLE_PINNED_DIGEST_SCHEMA,
        path: absolute,
        identity,
        sha256: sha256.to_ascii_lowercase(),
    };
    serde_json::to_writer(&mut file, &record).map_err(std::io::Error::other)?;
    file.write_all(b"\n")?;
    file.sync_all()?;
    drop(file);
    let result = std::fs::rename(&temp, &target);
    if result.is_err() {
        let _ = std::fs::remove_file(&temp);
    }
    result
}

#[cfg(not(unix))]
fn write_durable_pinned_digest(
    _path: &Path,
    _identity: PinnedFileIdentity,
    _sha256: &str,
) -> std::io::Result<()> {
    Ok(())
}

static PINNED_DIGEST_HASHES: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

#[cfg(test)]
fn pinned_digest_hashes_by_identity(
) -> &'static std::sync::Mutex<std::collections::HashMap<PinnedFileIdentity, u64>> {
    static HASHES: std::sync::OnceLock<
        std::sync::Mutex<std::collections::HashMap<PinnedFileIdentity, u64>>,
    > = std::sync::OnceLock::new();
    HASHES.get_or_init(Default::default)
}

#[cfg(test)]
fn pinned_digest_hash_count_for(path: &Path) -> u64 {
    let Ok(file) = crate::secure_file::open_regular_file_no_follow(path) else {
        return 0;
    };
    let Ok(identity) = pinned_file_identity(&file) else {
        return 0;
    };
    pinned_digest_hash_count_for_identity(identity)
}

#[cfg(test)]
fn pinned_digest_hash_count_for_identity(identity: PinnedFileIdentity) -> u64 {
    pinned_digest_hashes_by_identity()
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
        .get(&identity)
        .copied()
        .unwrap_or(0)
}

/// How many times this process has actually read a file end-to-end to answer a
/// pinned-digest question.
///
/// Exported so tests can prove the verifier is doing its job: a 2.3 GB bundle
/// must be hashed at most once for one unchanged identity, not once per
/// admission. A valid durable attestation can make the count zero after a
/// process restart.
pub fn pinned_digest_hash_count() -> u64 {
    PINNED_DIGEST_HASHES.load(std::sync::atomic::Ordering::Relaxed)
}

/// SHA-256 of a file's CURRENT bytes, read through a retained no-follow
/// descriptor and memoized on that descriptor's identity.
///
/// The descriptor is the point. Opening no-follow, checking that the target is
/// a regular file, and hashing that same open file means the bytes proven are
/// the bytes at the inode the check resolved — a symlink swap or a path
/// replacement between the check and the read cannot substitute different
/// content.
pub fn pinned_file_digest(path: &Path) -> anyhow::Result<String> {
    pinned_file_digest_with_progress(path, |_, _| Ok(()))
}

/// [`pinned_file_digest`] with bounded read progress. A callback receiving
/// `(0, total)` is also used as a cancellable heartbeat while another thread
/// owns the same file-identity flight.
pub fn pinned_file_digest_with_progress(
    path: &Path,
    progress: impl FnMut(u64, u64) -> anyhow::Result<()>,
) -> anyhow::Result<String> {
    let file = crate::secure_file::open_regular_file_no_follow(path)?;
    pinned_file_digest_from_open_file(path, &file, progress)
}

/// Digest the exact retained descriptor supplied by a caller and bind any
/// cache or durable attestation to that descriptor's identity. `path` is used
/// only to key the durable record and to prove the same identity still
/// occupies the pathname after hashing.
pub fn pinned_file_digest_from_open_file(
    path: &Path,
    file: &std::fs::File,
    mut progress: impl FnMut(u64, u64) -> anyhow::Result<()>,
) -> anyhow::Result<String> {
    anyhow::ensure!(
        file.metadata()?.is_file(),
        "pinned artifact is not a regular file"
    );
    let identity = pinned_file_identity(file)?;
    if let Some(digest) = pinned_digest_cache()
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
        .get(&identity)
        .cloned()
    {
        return Ok(digest);
    }
    let flight = {
        let mut flights = pinned_digest_flights()
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        flights.entry(identity).or_default().clone()
    };
    // Declared before the mutex guard so return/unwind drops the guard first;
    // the last participant then removes this identity from the flight map.
    let _flight_cleanup = PinnedDigestFlightCleanup {
        identity,
        flight: flight.clone(),
    };
    let _flight = loop {
        match flight.try_lock() {
            Ok(guard) => break guard,
            Err(std::sync::TryLockError::Poisoned(poisoned)) => break poisoned.into_inner(),
            Err(std::sync::TryLockError::WouldBlock) => {
                progress(0, identity.len)?;
                std::thread::sleep(std::time::Duration::from_millis(100));
            }
        }
    };
    if let Some(digest) = pinned_digest_cache()
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
        .get(&identity)
        .cloned()
    {
        return Ok(digest);
    }
    if let Some(digest) = read_durable_pinned_digest(path, identity) {
        pinned_digest_cache()
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .insert(identity, digest.clone());
        progress(identity.len, identity.len)?;
        return Ok(digest);
    }
    use sha2::{Digest, Sha256};
    use std::io::{Read, Seek, SeekFrom};
    let mut reader = file.try_clone()?;
    reader.seek(SeekFrom::Start(0))?;
    let mut hash = Sha256::new();
    let mut buffer = vec![0_u8; 1024 * 1024];
    let mut read_total = 0_u64;
    progress(0, identity.len)?;
    loop {
        let read = reader.read(&mut buffer)?;
        if read == 0 {
            break;
        }
        hash.update(&buffer[..read]);
        read_total = read_total
            .checked_add(read as u64)
            .ok_or_else(|| anyhow::anyhow!("pinned file byte count overflow"))?;
        progress(read_total, identity.len)?;
    }
    anyhow::ensure!(
        read_total == identity.len,
        "pinned file changed length while hashing"
    );
    anyhow::ensure!(
        pinned_file_identity(file)? == identity,
        "pinned file changed while hashing"
    );
    let current = crate::secure_file::open_regular_file_no_follow(path)?;
    anyhow::ensure!(
        pinned_file_identity(&current)? == identity,
        "pinned file path changed while hashing"
    );
    let digest = format!("{:x}", hash.finalize());
    PINNED_DIGEST_HASHES.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    #[cfg(test)]
    {
        *pinned_digest_hashes_by_identity()
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .entry(identity)
            .or_default() += 1;
    }
    pinned_digest_cache()
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
        .insert(identity, digest.clone());
    if let Err(error) = write_durable_pinned_digest(path, identity, &digest) {
        tracing::warn!(path = %path.display(), %error, "failed to persist artifact digest attestation");
    }
    Ok(digest)
}

/// Read-only pinned check: do this file's current bytes hash to `expected`?
///
/// Deletes nothing, writes nothing, and answers `false` for any file it cannot
/// open or hash. This is the form a read-only placement preview may ask.
pub fn pinned_file_matches(path: &Path, expected_sha256: &str) -> bool {
    pinned_file_digest(path)
        .map(|digest| digest.eq_ignore_ascii_case(expected_sha256))
        .unwrap_or(false)
}

/// Prove one already-placed file against a manifest-pinned SHA-256, writing
/// the shared `.sha256-verified` marker on success.
///
/// This is [`verify_file_integrity`]'s contract for callers that hold a pin
/// but not a [`ModelFile`] — per-file dependency materialization, which never
/// routes through `pull_model` and so never reaches the manifest pull path's
/// verification. Hugging Face `main` revisions are mutable, so a dependency
/// downloaded by hash-free bytes-and-size alone could be a different file than
/// the one the manifest pinned; nothing downstream would notice, because the
/// frozen plan proves only that the path is local.
///
/// **The `.sha256-verified` marker is never accepted as proof here.** It is an
/// ordinary file sitting beside the artifact, so in a group-writable models
/// root — which the model-storage invariant explicitly supports, `0664` and
/// all — anyone who can write the weights can also write a sidecar naming the
/// expected digest, and a stale marker can outlive a replace. A writable
/// attestation is not content authentication. Instead, the digest is cached by
/// exact descriptor identity and persisted only in Mold's owner-only,
/// parent-protected attestation directory. Changed identities hash again; an
/// unchanged identity survives a process restart without rereading the model.
/// The marker is still WRITTEN, because
/// `Config::manifest_files_exist` and the partial-cleanup sweep read it as an
/// "this file is fully written" signal.
///
/// On mismatch both the file and its marker are removed, so a retry
/// re-downloads instead of resurrecting the rejected bytes.
pub fn verify_pinned_file(
    path: &Path,
    expected_sha256: &str,
    filename: &str,
    model: &str,
) -> Result<(), DownloadError> {
    // Unverifiable is not verified: a pinned dependency whose bytes cannot be
    // read fails closed rather than being accepted unproven, because nothing
    // downstream re-asks — the frozen plan proves only that the path is local.
    let actual = pinned_file_digest(path).map_err(|error| {
        DownloadError::Other(format!(
            "cannot verify the pinned SHA-256 of {filename} at {}: {error:#}",
            path.display()
        ))
    })?;
    if !actual.eq_ignore_ascii_case(expected_sha256) {
        let _ = std::fs::remove_file(path);
        let _ = std::fs::remove_file(sha256_marker_path(path));
        return Err(DownloadError::Sha256Mismatch {
            filename: filename.to_string(),
            expected: expected_sha256.to_string(),
            actual,
            model: model.to_string(),
        });
    }
    if let Err(error) = write_sha256_marker(path, &actual) {
        // Same policy as the manifest pull path: the bytes are proven, so the
        // attempt succeeds, but a missing marker means the next installed-state
        // check reports the file incomplete. Log loudly rather than fail.
        eprintln!("warning: failed to write .sha256-verified marker for {filename}: {error}");
    }
    Ok(())
}

/// The digest a `.sha256-verified` marker records, if one is readable.
pub fn recorded_sha256_marker(path: &Path) -> Option<String> {
    std::fs::read_to_string(sha256_marker_path(path))
        .ok()
        .map(|recorded| recorded.trim().to_string())
        .filter(|recorded| !recorded.is_empty())
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

/// Synchronous hf-hub progress adapter used by pre-admission dependency
/// preparation running on Tokio's blocking pool.
struct SyncCallbackProgress {
    callback: DownloadProgressCallback,
    started_at: Instant,
    filename: String,
    accumulated: u64,
    total: u64,
    last_emit: Instant,
}

impl SyncCallbackProgress {
    fn new(callback: DownloadProgressCallback) -> Self {
        Self {
            callback,
            started_at: Instant::now(),
            filename: String::new(),
            accumulated: 0,
            total: 0,
            last_emit: Instant::now(),
        }
    }
}

impl hf_hub::api::Progress for SyncCallbackProgress {
    fn init(&mut self, size: usize, filename: &str) {
        self.filename = filename.to_string();
        self.accumulated = 0;
        self.total = size as u64;
        self.last_emit = Instant::now();
        (self.callback)(DownloadProgressEvent::FileStart {
            filename: self.filename.clone(),
            file_index: 0,
            total_files: 1,
            size_bytes: self.total,
            batch_bytes_downloaded: 0,
            batch_bytes_total: self.total,
            batch_elapsed_ms: self.started_at.elapsed().as_millis() as u64,
        });
    }

    fn update(&mut self, size: usize) {
        self.accumulated = self.accumulated.saturating_add(size as u64);
        let now = Instant::now();
        if now.duration_since(self.last_emit).as_millis() < 250 && self.accumulated < self.total {
            return;
        }
        self.last_emit = now;
        (self.callback)(DownloadProgressEvent::FileProgress {
            filename: self.filename.clone(),
            file_index: 0,
            bytes_downloaded: self.accumulated,
            bytes_total: self.total,
            batch_bytes_downloaded: self.accumulated,
            batch_bytes_total: self.total,
            batch_elapsed_ms: self.started_at.elapsed().as_millis() as u64,
        });
    }

    fn finish(&mut self) {
        (self.callback)(DownloadProgressEvent::FileDone {
            filename: self.filename.clone(),
            file_index: 0,
            total_files: 1,
            batch_bytes_downloaded: self.total,
            batch_bytes_total: self.total,
            batch_elapsed_ms: self.started_at.elapsed().as_millis() as u64,
        });
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

/// Return an existing valid clean path for a manifest file, migrating from a
/// legacy location when needed.
fn find_existing_placed_file(
    models_dir: &std::path::Path,
    manifest: &ModelManifest,
    file: &ModelFile,
    skip_verify: bool,
) -> Result<Option<PathBuf>, DownloadError> {
    let canonical_rel = crate::manifest::storage_path(manifest, file);
    let canonical_path = models_dir.join(&canonical_rel);

    for candidate_rel in crate::manifest::storage_path_candidates(manifest, file) {
        let candidate_path = models_dir.join(candidate_rel);
        if !is_already_placed(&candidate_path, file, &manifest.name, skip_verify) {
            continue;
        }
        if candidate_path != canonical_path {
            hardlink_or_copy(&candidate_path, &canonical_path)?;
            verify_file_integrity(&canonical_path, file, &manifest.name, skip_verify)?;
        }
        return Ok(Some(canonical_path));
    }

    Ok(None)
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
fn require_manifest_acquisition(manifest: &ModelManifest) -> Result<(), DownloadError> {
    let contains_gated_identity =
        crate::require_model_activation(&manifest.name, Some(&manifest.family)).is_err()
            || manifest.files.iter().any(|file| {
                crate::require_model_activation(&file.hf_repo, Some(&manifest.family)).is_err()
                    || crate::require_model_activation(&file.hf_filename, Some(&manifest.family))
                        .is_err()
            });
    if contains_gated_identity {
        crate::require_model_acquisition(&manifest.name, Some(&manifest.family))?;
        if !crate::is_exact_registered_manifest(manifest) {
            crate::require_model_activation("minimax-h3", Some("minimax-h3"))?;
        }
    } else {
        crate::require_model_acquisition(&manifest.name, Some(&manifest.family))?;
    }
    require_manifest_licenses_accepted(manifest)
}

pub(crate) fn validate_available_download_space(
    manifest: &ModelManifest,
    required_bytes: u64,
    available_bytes: u64,
    path: &Path,
) -> Result<(), DownloadError> {
    if required_bytes <= available_bytes {
        return Ok(());
    }
    Err(DownloadError::InsufficientDiskSpace {
        model: manifest.name.clone(),
        required_bytes,
        available_bytes,
        path: path.to_path_buf(),
    })
}

/// Refuse before the first byte when the selected volume cannot hold every
/// uncached file in the manifest. The closest existing ancestor keeps this
/// useful for a fresh `models/` directory without creating it as a side effect.
fn required_download_bytes_in(
    manifest: &ModelManifest,
    models_root: &Path,
    skip_verify: bool,
) -> Result<u64, DownloadError> {
    let mut required = 0u64;
    let managed_cache = Cache::new(models_root.join(".hf-cache"));
    for file in &manifest.files {
        if find_existing_placed_file(models_root, manifest, file, skip_verify)?.is_some() {
            continue;
        }
        let cached = managed_cache
            .repo(hf_file_repo(&file.hf_repo, &file.hf_filename))
            .get(&file.hf_filename)
            .and_then(|path| {
                path.metadata()
                    .ok()
                    .filter(|metadata| metadata.len() == file.size_bytes)
                    .map(|_| path)
            })
            .is_some();
        if !cached {
            required = required.saturating_add(file.size_bytes);
        }
    }
    Ok(required)
}

fn require_download_space(
    manifest: &ModelManifest,
    skip_verify: bool,
) -> Result<(), DownloadError> {
    let target = models_dir();
    let required_bytes = required_download_bytes_in(manifest, &target, skip_verify)?;
    if required_bytes == 0 {
        return Ok(());
    }
    let probe = target
        .ancestors()
        .find(|path| path.exists())
        .unwrap_or_else(|| Path::new("."));
    let available_bytes = fs2::available_space(probe).map_err(|error| {
        DownloadError::Other(format!(
            "failed to inspect free space at {} before pulling {}: {error}",
            probe.display(),
            manifest.name
        ))
    })?;
    validate_available_download_space(manifest, required_bytes, available_bytes, probe)
}

/// Refuse to download any manifest carrying a file under a license the user
/// has not explicitly accepted.
///
/// Checked at the manifest level, before the first byte moves, and from the
/// one choke point every pull path already calls — an automatic or
/// server-side auto-pull must fail here rather than acquire restricted
/// weights on the user's behalf. A Mold data root that cannot be resolved
/// fails closed: unverifiable is not accepted.
fn require_manifest_licenses_accepted(manifest: &ModelManifest) -> Result<(), DownloadError> {
    require_manifest_licenses_accepted_in(manifest, crate::Config::mold_dir().as_deref())
}

/// The pure half of the gate: decide against an explicit Mold data root.
///
/// `None` is a root that could not be resolved, which fails closed —
/// unverifiable is not accepted.
fn require_manifest_licenses_accepted_in(
    manifest: &ModelManifest,
    mold_home: Option<&std::path::Path>,
) -> Result<(), DownloadError> {
    for file in &manifest.files {
        require_license_accepted(&manifest.name, &file.hf_filename, mold_home)?;
    }
    Ok(())
}

/// Refuse one manifest file whose license the user has not accepted.
///
/// This is the single gate implementation. `require_manifest_licenses_accepted`
/// asks it once per file before a `mold pull`, and per-file dependency
/// materialization (identity assets, which never route through `pull_model`)
/// asks it directly for exactly the files it is about to fetch. Two gates would
/// be two policies; a dependency path with no gate at all would acquire
/// restricted weights on the user's behalf.
///
/// `mold_home` is `None` when the Mold data root cannot be resolved, which
/// fails closed — unverifiable is not accepted.
pub fn require_license_accepted(
    manifest_name: &str,
    hf_filename: &str,
    mold_home: Option<&std::path::Path>,
) -> Result<(), DownloadError> {
    for license in crate::license_acceptance::licenses_for_manifest_file(manifest_name, hf_filename)
    {
        if mold_home.is_some_and(|home| crate::license_acceptance::is_accepted(home, license)) {
            continue;
        }
        return Err(DownloadError::LicenseNotAccepted {
            license_id: license.id.to_string(),
            message: crate::license_acceptance::acceptance_required_message(manifest_name, license),
        });
    }
    Ok(())
}

pub async fn pull_model(
    manifest: &ModelManifest,
    opts: &PullOptions,
) -> Result<ModelPaths, DownloadError> {
    pull_model_with_hf_token(manifest, opts, None).await
}

async fn pull_model_with_hf_token(
    manifest: &ModelManifest,
    opts: &PullOptions,
    hf_token: Option<&str>,
) -> Result<ModelPaths, DownloadError> {
    require_manifest_acquisition(manifest)?;
    require_download_space(manifest, opts.skip_verify)?;
    write_pulling_marker(&manifest.name)?;

    let mut builder = ApiBuilder::from_env().with_cache_dir(hf_cache_dir());
    if let Some(token) = resolve_hf_token_for(hf_token) {
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
        if let Some(clean_path) =
            find_existing_placed_file(&mdir, manifest, file, opts.skip_verify)?
        {
            downloads.push((file.component, clean_path));
            continue;
        }

        let clean_path = mdir.join(crate::manifest::storage_path(manifest, file));

        let bar = multi.add(ProgressBar::new(file.size_bytes));
        bar.set_style(bar_style.clone());
        bar.set_message(truncate_filename(&file.hf_filename, msg_width));

        let clean_path = download_and_place_file(
            &api,
            file,
            DownloadProgress::new(bar, msg_width),
            &manifest.name,
            &clean_path,
            opts.skip_verify,
        )
        .await?;

        downloads.push((file.component, clean_path));
    }

    warn_if_flux_gguf_needs_reference(manifest, None);

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
    pull_model_with_callback_and_hf_token(manifest, callback, opts, None).await
}

async fn pull_model_with_callback_and_hf_token(
    manifest: &ModelManifest,
    callback: DownloadProgressCallback,
    opts: &PullOptions,
    hf_token: Option<&str>,
) -> Result<ModelPaths, DownloadError> {
    require_manifest_acquisition(manifest)?;
    require_download_space(manifest, opts.skip_verify)?;
    write_pulling_marker(&manifest.name)?;

    let mut builder = ApiBuilder::from_env().with_cache_dir(hf_cache_dir());
    if let Some(token) = resolve_hf_token_for(hf_token) {
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
                find_existing_placed_file(&mdir_clone, &manifest_clone, file, skip_verify)
                    .map(|p| p.is_some())
                    .unwrap_or(false)
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
        let clean_path = mdir.join(crate::manifest::storage_path(manifest, file));

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
        let clean_path = download_and_place_file(
            &api,
            file,
            progress,
            &manifest.name,
            &clean_path,
            opts.skip_verify,
        )
        .await?;

        downloads.push((file.component, clean_path));
        completed_bytes += file.size_bytes;
    }

    warn_if_flux_gguf_needs_reference(manifest, Some(&callback));

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
    pull_model_files_only_with_hf_token(manifest, opts, None).await
}

async fn pull_model_files_only_with_hf_token(
    manifest: &ModelManifest,
    opts: &PullOptions,
    hf_token: Option<&str>,
) -> Result<(), DownloadError> {
    require_manifest_acquisition(manifest)?;
    require_download_space(manifest, opts.skip_verify)?;
    write_pulling_marker(&manifest.name)?;

    let mut builder = ApiBuilder::from_env().with_cache_dir(hf_cache_dir());
    if let Some(token) = resolve_hf_token_for(hf_token) {
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
        if find_existing_placed_file(&mdir, manifest, file, opts.skip_verify)?.is_some() {
            continue;
        }

        let clean_path = mdir.join(crate::manifest::storage_path(manifest, file));

        let bar = multi.add(ProgressBar::new(file.size_bytes));
        bar.set_style(bar_style.clone());
        bar.set_message(truncate_filename(&file.hf_filename, msg_width));

        download_and_place_file(
            &api,
            file,
            DownloadProgress::new(bar, msg_width),
            &manifest.name,
            &clean_path,
            opts.skip_verify,
        )
        .await?;
    }

    Ok(())
}

async fn pull_model_files_only_with_callback_and_hf_token(
    manifest: &ModelManifest,
    callback: DownloadProgressCallback,
    opts: &PullOptions,
    hf_token: Option<&str>,
) -> Result<(), DownloadError> {
    require_manifest_acquisition(manifest)?;
    require_download_space(manifest, opts.skip_verify)?;
    write_pulling_marker(&manifest.name)?;

    let mut builder = ApiBuilder::from_env().with_cache_dir(hf_cache_dir());
    if let Some(token) = resolve_hf_token_for(hf_token) {
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
                find_existing_placed_file(&mdir_clone, &manifest_clone, file, skip_verify)
                    .map(|p| p.is_some())
                    .unwrap_or(false)
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
        let clean_path = mdir.join(crate::manifest::storage_path(manifest, file));

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

        download_and_place_file(
            &api,
            file,
            progress,
            &manifest.name,
            &clean_path,
            opts.skip_verify,
        )
        .await?;
        completed_bytes += file.size_bytes;
    }

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

type HfFileDownloadFlight = tokio::sync::Mutex<()>;
type HfFileDownloadFlights =
    tokio::sync::Mutex<HashMap<(String, String), Weak<HfFileDownloadFlight>>>;

/// Return the process-wide flight for one Hugging Face repository file.
///
/// Different model variants often reuse the same large encoder or VAE blob.
/// The server deliberately runs unrelated pulls in parallel, so coordinate
/// only identical HF files here before entering hf-hub's short-lived file
/// lock. Weak entries keep completed identities from accumulating forever.
async fn hf_file_download_flight(repo: &str, filename: &str) -> Arc<HfFileDownloadFlight> {
    static FLIGHTS: OnceLock<HfFileDownloadFlights> = OnceLock::new();

    let flights = FLIGHTS.get_or_init(|| tokio::sync::Mutex::new(HashMap::new()));
    let mut flights = flights.lock().await;
    flights.retain(|_, flight| flight.strong_count() > 0);

    let key = (repo.to_string(), filename.to_string());
    if let Some(flight) = flights.get(&key).and_then(Weak::upgrade) {
        return flight;
    }

    let flight = Arc::new(tokio::sync::Mutex::new(()));
    flights.insert(key, Arc::downgrade(&flight));
    flight
}

async fn with_hf_file_download_flight<T, Fut>(repo: &str, filename: &str, operation: Fut) -> T
where
    Fut: Future<Output = T>,
{
    let flight = hf_file_download_flight(repo, filename).await;
    let _flight_guard = flight.lock().await;
    operation.await
}

async fn download_and_place_file<P: Progress + Clone + Send + Sync + 'static>(
    api: &Api,
    file: &ModelFile,
    progress: P,
    model_name: &str,
    clean_path: &Path,
    skip_verify: bool,
) -> Result<PathBuf, DownloadError> {
    with_hf_file_download_flight(&file.hf_repo, &file.hf_filename, async move {
        let repo = api.repo(hf_file_repo(&file.hf_repo, &file.hf_filename));
        let hf_path = match repo
            .download_with_progress(&file.hf_filename, progress)
            .await
        {
            Ok(path) => path,
            Err(e) => {
                let status = extract_http_status(&e);
                let err_str = e.to_string();
                if status == Some(401)
                    || err_str.contains("401")
                    || err_str.contains("Unauthorized")
                {
                    return Err(DownloadError::Unauthorized {
                        repo: file.hf_repo.clone(),
                        model: model_name.to_string(),
                    });
                } else if status == Some(403)
                    || err_str.contains("403")
                    || err_str.contains("Forbidden")
                    || err_str.contains("gated")
                    || err_str.contains("Access denied")
                {
                    return Err(DownloadError::GatedModel {
                        repo: file.hf_repo.clone(),
                        model: model_name.to_string(),
                    });
                } else {
                    return Err(DownloadError::DownloadFailed {
                        repo: file.hf_repo.clone(),
                        filename: file.hf_filename.clone(),
                        source: e,
                    });
                }
            }
        };

        hardlink_or_copy(&hf_path, clean_path)?;
        verify_file_integrity(clean_path, file, model_name, skip_verify)?;
        Ok(clean_path.to_path_buf())
    })
    .await
}

// ── Synchronous single-file download (for use from spawn_blocking) ───────────

fn require_single_file_acquisition(
    hf_repo: &str,
    hf_filename: &str,
    target_subdir: Option<&str>,
) -> Result<(), DownloadError> {
    crate::require_model_activation(hf_repo, None)?;
    crate::require_model_activation(hf_filename, None)?;
    if let Some(target_subdir) = target_subdir {
        crate::require_model_activation(target_subdir, None)?;
    }
    Ok(())
}

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
    require_single_file_acquisition(hf_repo, hf_filename, target_subdir)?;

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
    download_single_file_sync_with_adapter(
        &models_dir(),
        hf_repo,
        hf_filename,
        target_subdir,
        progress,
    )
}

/// Callback-reporting counterpart to [`download_single_file_sync`].
///
/// It remains a blocking function by design; callers must use
/// `tokio::task::spawn_blocking`.
pub fn download_single_file_sync_with_progress(
    hf_repo: &str,
    hf_filename: &str,
    target_subdir: Option<&str>,
    callback: DownloadProgressCallback,
) -> Result<PathBuf, DownloadError> {
    require_single_file_acquisition(hf_repo, hf_filename, target_subdir)?;

    download_single_file_sync_with_adapter(
        &models_dir(),
        hf_repo,
        hf_filename,
        target_subdir,
        SyncCallbackProgress::new(callback),
    )
}

/// Explicit-root counterpart used when a caller owns an immutable config
/// snapshot. Both the managed Hugging Face cache and clean target stay under
/// `models_root`.
pub fn download_single_file_sync_with_progress_in(
    models_root: &Path,
    hf_repo: &str,
    hf_filename: &str,
    target_subdir: Option<&str>,
    callback: DownloadProgressCallback,
) -> Result<PathBuf, DownloadError> {
    require_single_file_acquisition(hf_repo, hf_filename, target_subdir)?;

    download_single_file_sync_with_adapter(
        models_root,
        hf_repo,
        hf_filename,
        target_subdir,
        SyncCallbackProgress::new(callback),
    )
}

/// Deterministic clean path that [`download_single_file_sync`] will populate
/// for a dependency with a target subdirectory.
///
/// This performs no I/O and does not imply that the file is present. It is
/// used by read-only placement previews to build the same engine input shape
/// that admission will materialize later.
pub fn planned_single_file_path(hf_filename: &str, target_subdir: &str) -> PathBuf {
    planned_single_file_path_in(&models_dir(), hf_filename, target_subdir)
}

/// No-I/O counterpart used by read-only previews with their exact config
/// snapshot. The root is never created.
pub fn planned_single_file_path_in(
    models_root: &Path,
    hf_filename: &str,
    target_subdir: &str,
) -> PathBuf {
    let leaf = hf_filename.rsplit('/').next().unwrap_or(hf_filename);
    models_root.join(target_subdir).join(leaf)
}

fn download_single_file_sync_with_adapter<P>(
    models_root: &Path,
    hf_repo: &str,
    hf_filename: &str,
    target_subdir: Option<&str>,
    progress: P,
) -> Result<PathBuf, DownloadError>
where
    P: hf_hub::api::Progress,
{
    // Keep the policy at the lowest download boundary as a defense against a
    // future internal caller bypassing the public wrappers. The wrappers also
    // check before constructing progress adapters or resolving managed paths.
    require_single_file_acquisition(hf_repo, hf_filename, target_subdir)?;

    use hf_hub::api::sync::ApiBuilder;

    let mut builder = ApiBuilder::from_env()
        .with_cache_dir(models_root.join(".hf-cache"))
        .with_progress(false);
    if let Some(token) = resolve_hf_token() {
        builder = builder.with_token(Some(token));
    }
    let api = builder
        .build()
        .map_err(|e| DownloadError::SyncApiSetup(e.to_string()))?;
    let repo = api.repo(hf_file_repo(hf_repo, hf_filename));
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
        let clean_path = planned_single_file_path_in(models_root, hf_filename, subdir);
        hardlink_or_copy(&hf_path, &clean_path)?;
        Ok(clean_path)
    } else {
        Ok(hf_path)
    }
}

/// Check whether a file is present in mold's managed hf-hub cache
/// (`<models_dir>/.hf-cache/`). Narrower than [`cached_file_path`] — does
/// not consult the system-wide `~/.cache/huggingface/`, the legacy mold
/// models cache, or any clean-path location. Used as a layout-agnostic
/// fallback by `Config::discovered_manifest_paths` so a single shard set
/// downloaded by a manifest install can also satisfy a catalog companion
/// that expects the same files under a different canonical layout (e.g.
/// the Gemma TE shared by `ltx-2.3-22b-distilled:fp8` and the catalog
/// `ltx2-te` companion). Tests that intentionally set up a "model not
/// downloaded" world are unaffected because they only override
/// `MOLD_MODELS_DIR`, not the user's home HF cache.
pub fn cached_file_path_in_mold_cache(hf_repo: &str, hf_filename: &str) -> Option<PathBuf> {
    let cache = Cache::new(hf_cache_dir());
    let repo = cache.repo(hf_file_repo(hf_repo, hf_filename));
    repo.get(hf_filename)
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
    cached_file_path_in(&models_dir(), hf_repo, hf_filename, target_subdir)
}

/// Explicit-root cache lookup for admission. This preserves the historical
/// fallback search while keeping the managed cache and clean target bound to
/// the caller's config snapshot.
pub fn cached_file_path_in(
    models_root: &Path,
    hf_repo: &str,
    hf_filename: &str,
    target_subdir: Option<&str>,
) -> Option<PathBuf> {
    // 1. Check clean path (if target_subdir specified)
    if let Some(subdir) = target_subdir {
        let clean_path = planned_single_file_path_in(models_root, hf_filename, subdir);
        if clean_path.exists() {
            return Some(clean_path);
        }
    }

    // 2. Check new hf-cache location (~/.mold/models/.hf-cache/)
    let new_cache = Cache::new(models_root.join(".hf-cache"));
    let new_repo = new_cache.repo(hf_file_repo(hf_repo, hf_filename));
    if let Some(path) = new_repo.get(hf_filename) {
        return Some(path);
    }

    // 3. Check old mold models dir (backward compat — HF cached here before .hf-cache/)
    let old_cache = Cache::new(models_root.to_path_buf());
    let old_repo = old_cache.repo(hf_file_repo(hf_repo, hf_filename));
    if let Some(path) = old_repo.get(hf_filename) {
        return Some(path);
    }

    // 4. Check default HF cache (~/.cache/huggingface/hub/)
    let default_cache = Cache::from_env();
    let default_repo = default_cache.repo(hf_file_repo(hf_repo, hf_filename));
    default_repo.get(hf_filename)
}

/// Strictly read-only cache inspection for placement previews.
///
/// Every cache constructor is gated on an already-existing root, and the
/// clean destination is derived from the caller's immutable config snapshot.
/// This function never creates the models root, `.hf-cache`, or the default
/// Hugging Face cache.
pub fn cached_file_path_existing_only(
    models_root: &Path,
    hf_repo: &str,
    hf_filename: &str,
    target_subdir: Option<&str>,
) -> Option<PathBuf> {
    if let Some(subdir) = target_subdir {
        let clean_path = planned_single_file_path_in(models_root, hf_filename, subdir);
        if clean_path.exists() {
            return Some(clean_path);
        }
    }

    let lookup = |root: &Path| {
        root.is_dir().then(|| {
            Cache::new(root.to_path_buf())
                .repo(hf_file_repo(hf_repo, hf_filename))
                .get(hf_filename)
        })?
    };
    lookup(&models_root.join(".hf-cache"))
        .or_else(|| lookup(models_root))
        .or_else(|| {
            let cache = Cache::from_env();
            cache.path().is_dir().then(|| {
                cache
                    .repo(hf_file_repo(hf_repo, hf_filename))
                    .get(hf_filename)
            })?
        })
}

// ── Pull and configure (shared between CLI and server) ───────────────────────

/// Hidden LTX-2 adapters are complete, runnable assets by themselves. They
/// must use the files-only pull path: treating their single LoRA tensor as a
/// standalone diffusion checkpoint makes `paths_from_downloads` require a
/// VAE after the verified file has already landed. Upstream publishes the
/// camera controls as individual `.safetensors` LoRAs (LTX-2 README:72-83).
fn manifest_uses_files_only_pull(manifest: &ModelManifest) -> bool {
    manifest.is_files_only_bundle()
}

fn qualify_downloaded_contract(manifest: &ModelManifest) -> Result<(), DownloadError> {
    qualify_downloaded_contract_in(manifest, &models_dir())
}

fn qualify_downloaded_contract_in(
    manifest: &ModelManifest,
    models_root: &Path,
) -> Result<(), DownloadError> {
    if !crate::ltx25_manifest::is_contract_manifest(&manifest.name) {
        return Ok(());
    }
    let paths = crate::ltx25_manifest::Ltx25ModelPaths::resolve_in(models_root, &manifest.name)
        .ok_or(DownloadError::MissingComponent)?;
    paths
        .qualify()
        .map_err(|error| DownloadError::QualificationFailed {
            model: manifest.name.clone(),
            message: error.to_string(),
        })
}

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

    // Utility models and hidden LTX-2 control adapters have no standalone
    // runtime config entry. The selected control is frozen onto the request
    // as a concrete LoRA path after this download completes.
    if manifest_uses_files_only_pull(manifest) {
        pull_model_files_only(manifest, opts).await?;
        qualify_downloaded_contract(manifest)?;
        let config = Config::load_or_default();
        remove_pulling_marker(&manifest.name);
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

        remove_pulling_marker(&manifest.name);
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
    pull_and_configure_with_callback_and_hf_token(model, callback, opts, None).await
}

/// Download a model with a request-scoped Hugging Face token. The explicit
/// token takes precedence over environment and token-file credentials for this
/// call only, without expanding the stable public [`PullOptions`] struct.
pub async fn pull_and_configure_with_callback_and_hf_token(
    model: &str,
    callback: DownloadProgressCallback,
    opts: &PullOptions,
    hf_token: Option<&str>,
) -> Result<(crate::Config, Option<ModelPaths>), DownloadError> {
    use crate::config::Config;
    use crate::manifest::{find_manifest, resolve_model_name};

    let canonical = resolve_model_name(model);

    let manifest = find_manifest(&canonical).ok_or_else(|| DownloadError::UnknownModel {
        model: model.to_string(),
    })?;

    // Utility models and hidden LTX-2 control adapters have no standalone
    // runtime config entry. The selected control is frozen onto the request
    // as a concrete LoRA path after this download completes.
    if manifest_uses_files_only_pull(manifest) {
        pull_model_files_only_with_callback_and_hf_token(manifest, callback, opts, hf_token)
            .await?;
        qualify_downloaded_contract(manifest)?;
        let config = Config::load_or_default();
        remove_pulling_marker(&manifest.name);
        return Ok((config, None));
    }

    // Upscaler models: download files, create minimal config with weights path.
    if manifest.is_upscaler() {
        pull_model_files_only_with_callback_and_hf_token(manifest, callback, opts, hf_token)
            .await?;

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

        remove_pulling_marker(&manifest.name);
        return Ok((config, None));
    }

    let paths = pull_model_with_callback_and_hf_token(manifest, callback, opts, hf_token).await?;

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

// ── Civitai token resolution ────────────────────────────────────────────────

/// Resolve `CIVITAI_TOKEN` from the environment. Mirrors `resolve_hf_token`'s
/// shape, but Civitai has no token-file convention — just the env var. An
/// empty / whitespace-only env var resolves to `None` so a stale shell can't
/// silently send blank `Authorization: Bearer ` headers.
pub fn resolve_civitai_token() -> Option<String> {
    std::env::var("CIVITAI_TOKEN").ok().and_then(|t| {
        let trimmed = t.trim().to_string();
        if trimmed.is_empty() {
            None
        } else {
            Some(trimmed)
        }
    })
}

/// Build the [`RecipeAuth`] required for a Civitai-gated recipe. Returns
/// [`DownloadError::MissingCivitaiToken`] when no token is set; the error
/// message names the env var so the CLI/server can surface a clear remediation.
pub fn civitai_auth_or_error(id: &str) -> Result<RecipeAuth, DownloadError> {
    match resolve_civitai_token() {
        Some(t) => Ok(RecipeAuth::Bearer(t)),
        None => Err(DownloadError::MissingCivitaiToken { id: id.to_string() }),
    }
}

// ── Companion presence helpers ──────────────────────────────────────────────
//
// Civitai single-file checkpoints ship without their text encoders / VAE,
// so the catalog scanner records `companions: ["clip-l", "sdxl-vae", ...]`
// on those entries. Both server (`POST /api/catalog/:id/download`) and CLI
// (`mold pull cv:<id>`) need to enqueue/pull missing companions before the
// primary entry. The on-disk presence check + name-resolution loop lives
// here so they share one implementation; the server's
// `enqueue_missing_companions` consumes this through `DownloadQueue`, the
// CLI through `pull_and_configure_with_callback`.

/// True when every file the companion's synthetic manifest declares is
/// present under `models_dir` AND no `.pulling` marker for the manifest's
/// canonical name exists. A leftover marker means a previous pull was
/// interrupted and the on-disk content can't be trusted yet.
pub fn companion_present_on_disk(
    models_dir: &Path,
    manifest: &crate::manifest::ModelManifest,
) -> bool {
    if pulling_marker_path_in(models_dir, &manifest.name).exists() {
        return false;
    }
    manifest.files.iter().all(|f| {
        let storage = crate::manifest::storage_path(manifest, f);
        let path = models_dir.join(storage);
        if !path.exists() {
            return false;
        }
        if f.sha256.is_some() {
            return sha256_marker_path(&path).exists();
        }
        if f.size_bytes > 0 {
            return std::fs::metadata(&path)
                .map(|m| m.len() == f.size_bytes)
                .unwrap_or(false);
        }
        true
    })
}

/// True iff the recipe file at `dest` should be considered already
/// placed — used by both the catalog API's `installed: bool` predicate
/// AND the `fetch_recipe_inner` skip path so they cannot drift apart.
///
/// Acceptance rule:
/// - `sha256` declared → `.sha256-verified` marker is the sole criterion.
///   The marker is written only after cryptographic verification at download
///   time, so it is more authoritative than `size_bytes` (which can be stale
///   in the catalog DB when a model is re-uploaded under the same sha256 with
///   a different compressed size).  A file at the exact declared size but
///   without the marker is still rejected.
/// - `sha256` absent, `size_bytes` known → on-disk length must equal declared.
/// - Neither declared → marker is the only attestation; require it.
fn recipe_file_is_placed(dest: &Path, file: &RecipeFetchFile<'_>) -> bool {
    if !dest.exists() {
        return false;
    }
    if has_sha256_marker(dest) {
        return true;
    }
    match (file.sha256, file.size_bytes) {
        (Some(_), _) => sha256_marker_path(dest).exists(),
        (None, Some(expected)) => std::fs::metadata(dest)
            .map(|m| m.len() == expected)
            .unwrap_or(false),
        (None, None) => sha256_marker_path(dest).exists(),
    }
}

/// True iff every file in the recipe is present at its declared size (or,
/// when the recipe omits the size, has a `.sha256-verified` marker from
/// a prior verified pull) AND no `.pulling` marker for the catalog id is
/// present. Used by the catalog API to set `installed: bool` on each
/// wire entry so the SPA can hide the Download button and show Repair
/// instead.
///
/// Empty file slice returns `false` — callers (see `catalog_row_to_wire`
/// in mold-server) use this for Civitai-style recipe rows. HF rows
/// without a recipe go through `Config::manifest_model_is_downloaded`
/// instead, so an empty input here means "no recipe to walk" and we
/// refuse to claim install.
///
/// `id` is the catalog id (`cv:1234` / `hf:author/name`) — same string
/// the recipe-pull path uses to derive its marker and subdir name.
pub fn catalog_entry_installed(models_dir: &Path, id: &str, files: &[RecipeFetchFile<'_>]) -> bool {
    if files.is_empty() {
        return false;
    }
    if pulling_marker_path_in(models_dir, id).exists() {
        return false;
    }
    let sanitized = sanitize_recipe_id(id);
    let subdir_root = models_dir.join(&sanitized);
    files.iter().all(|f| {
        let Ok(dest) = resolve_recipe_dest(&subdir_root, f.dest) else {
            return false;
        };
        recipe_file_is_placed(&dest, f)
    })
}

/// Parse a `Vec<String>` of companion names out of `companions_json` and
/// return the ones that (a) resolve to a known synthetic manifest and (b)
/// aren't already fully present under `models_dir`.
///
/// Order is preserved — callers depend on companion-first ordering. Unknown
/// companions (no synthetic manifest in this build) are silently skipped:
/// catalog scanners may ship new canonical names ahead of the binary, and
/// surfacing those as errors would break catalog rows older builds can
/// never satisfy. `None` / unparsable JSON returns an empty vec.
pub fn missing_companions_from_json(
    companions_json: Option<&str>,
    models_dir: &Path,
) -> Vec<&'static crate::manifest::ModelManifest> {
    let Some(json) = companions_json else {
        return Vec::new();
    };
    let names: Vec<String> = match serde_json::from_str(json) {
        Ok(n) => n,
        Err(_) => return Vec::new(),
    };
    missing_companions(&names, models_dir)
}

/// `Vec<String>`-shaped variant for callers that already have a typed
/// companion list (e.g. live-fetched `CatalogEntry::companions`).
pub fn missing_companions(
    names: &[String],
    models_dir: &Path,
) -> Vec<&'static crate::manifest::ModelManifest> {
    let mut out = Vec::with_capacity(names.len());
    for name in names {
        let Some(manifest) = crate::manifest::find_manifest(name) else {
            tracing::warn!(
                companion = %name,
                "skipping companion with no synthetic manifest in this build",
            );
            continue;
        };
        if companion_present_on_disk(models_dir, manifest) {
            continue;
        }
        out.push(manifest);
    }
    out
}

// ── Recipe-driven downloads (Civitai single-file checkpoints) ───────────────
//
// Catalog rows for `cv:<id>` entries carry a `download_recipe.files` list of
// `(url, dest, sha256, size_bytes)` tuples that the manifest path can't
// express (the manifest assumes HF repos). The recipe fetcher lives here so
// it can share `compute_sha256`, `verify_sha256`, and the `.pulling` marker
// lifecycle with the manifest path. mold-core takes a plain
// `&[RecipeFetchFile]` slice + `RecipeAuth`; CLI/server callers translate
// from `mold_catalog::DownloadRecipe` at the boundary so this crate stays
// catalog-free (`mold-catalog` already depends on `mold-core`).

/// Plain recipe-input shape for [`fetch_recipe`]. Callers translate from
/// `mold_catalog::DownloadRecipe` to a slice of these.
#[derive(Debug, Clone)]
pub struct RecipeFetchFile<'a> {
    /// HTTP URL the file is fetched from.
    pub url: &'a str,
    /// Destination path relative to the per-recipe subdirectory
    /// (`<models_dir>/<sanitized-id>/`). May contain forward slashes for
    /// nested layouts; `..` and absolute paths are rejected.
    pub dest: &'a str,
    /// Optional SHA-256 hex digest. Verified after download when present.
    pub sha256: Option<&'a str>,
    /// Optional declared file size, used for progress reporting before the
    /// `Content-Length` header arrives.
    pub size_bytes: Option<u64>,
}

/// Authentication required for the recipe's URLs.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RecipeAuth {
    /// No bearer token required.
    None,
    /// Send the given bearer token as `Authorization: Bearer <token>`.
    /// Used for Civitai (`needs_token: Civitai`); callers resolve the
    /// token from `CIVITAI_TOKEN` / config before calling.
    Bearer(String),
}

/// Sanitize a catalog id (e.g. `cv:618692`) into a filesystem-safe subdir
/// name (`cv-618692`). Mirrors the manifest path's `replace(':', "-")`
/// rule so both sides land under the same models-dir subtree convention.
pub fn sanitize_recipe_id(id: &str) -> String {
    id.replace(':', "-")
}

/// Verify that a recipe `dest` stays under the per-recipe subdir. Rejects
/// absolute paths and any segment that traverses upward (`..`). Returns the
/// resolved per-file path under `subdir_root` on success.
fn resolve_recipe_dest(subdir_root: &Path, dest: &str) -> Result<PathBuf, DownloadError> {
    let candidate = Path::new(dest);
    if candidate.is_absolute() {
        return Err(DownloadError::RecipePathTraversal {
            dest: dest.to_string(),
        });
    }
    for component in candidate.components() {
        match component {
            std::path::Component::Normal(_) => {}
            // ParentDir / Prefix / RootDir / CurDir all escape or are
            // pointless. CurDir (`./`) is harmless but signals a malformed
            // recipe — reject for consistency.
            _ => {
                return Err(DownloadError::RecipePathTraversal {
                    dest: dest.to_string(),
                });
            }
        }
    }
    Ok(subdir_root.join(candidate))
}

/// Fetch a recipe-driven download. Writes each file under
/// `models_dir/<sanitized-id>/<dest>`, verifies SHA-256 when present, and
/// manages the `.pulling` marker lifecycle.
///
/// The marker is written before the first byte and removed only after every
/// file has been integrity-checked. On any error the marker is removed
/// best-effort so callers can retry; partial files are NOT cleaned up here
/// (callers wire that into their failure path the same way the manifest
/// path does, via `cleanup_partials_in_dir`).
pub async fn fetch_recipe(
    id: &str,
    files: &[RecipeFetchFile<'_>],
    auth: RecipeAuth,
    models_dir: &Path,
    progress: Option<DownloadProgressCallback>,
    opts: &PullOptions,
) -> Result<Vec<PathBuf>, DownloadError> {
    // The recipe path is a low-level public ingress used by both the CLI and
    // server. Apply the activation policy before deriving paths, creating the
    // recipe directory/marker, constructing an HTTP client, or reporting any
    // progress so callers cannot bypass a catalog-level gate with an opaque id.
    crate::require_model_activation(id, None)?;
    for file in files {
        crate::require_model_activation(file.url, None)?;
        crate::require_model_activation(file.dest, None)?;
    }

    let sanitized = sanitize_recipe_id(id);
    let subdir_root = models_dir.join(&sanitized);

    // Pre-flight: validate every dest before touching the network. A bad
    // `..` in any file aborts the whole recipe with no side-effects.
    let resolved: Vec<PathBuf> = files
        .iter()
        .map(|f| resolve_recipe_dest(&subdir_root, f.dest))
        .collect::<Result<Vec<_>, _>>()?;

    std::fs::create_dir_all(&subdir_root).map_err(|e| {
        DownloadError::FilePlacement(format!(
            "failed to create recipe subdir {}: {e}",
            subdir_root.display()
        ))
    })?;

    let marker = pulling_marker_path_in(models_dir, id);
    if let Some(parent) = marker.parent() {
        let _ = std::fs::create_dir_all(parent);
    }
    std::fs::write(&marker, id).map_err(|e| {
        DownloadError::FilePlacement(format!(
            "failed to write recipe marker {}: {e}",
            marker.display()
        ))
    })?;

    let result = fetch_recipe_inner(id, files, &resolved, auth, progress, opts).await;
    // Marker removed on success and best-effort on error. Cleanup of
    // partial files is the caller's responsibility (matches manifest path).
    let _ = std::fs::remove_file(&marker);
    result
}

async fn fetch_recipe_inner(
    id: &str,
    files: &[RecipeFetchFile<'_>],
    resolved: &[PathBuf],
    auth: RecipeAuth,
    progress: Option<DownloadProgressCallback>,
    opts: &PullOptions,
) -> Result<Vec<PathBuf>, DownloadError> {
    use std::io::Write;

    let client = reqwest::Client::builder()
        .user_agent(concat!("mold/", env!("CARGO_PKG_VERSION")))
        .build()
        .map_err(|e| DownloadError::Other(format!("failed to build HTTP client: {e}")))?;

    let total_files = files.len();
    let batch_bytes_total: u64 = files.iter().filter_map(|f| f.size_bytes).sum();
    let mut batch_bytes_downloaded: u64 = 0;
    let started = Instant::now();

    for (file_index, (file, dest_path)) in files.iter().zip(resolved.iter()).enumerate() {
        if let Some(parent) = dest_path.parent() {
            std::fs::create_dir_all(parent).map_err(|e| {
                DownloadError::FilePlacement(format!(
                    "failed to create directory {}: {e}",
                    parent.display()
                ))
            })?;
        }

        // Idempotency: skip the HTTP fetch when the file is already on disk
        // with the declared size, or (when no size is declared) when the
        // post-download .sha256-verified marker is present from a prior
        // run. Mirrors `is_already_placed` from the manifest path so a
        // recipe re-pull (Repair, double-clicked Download, retry-after-
        // partial-companion-failure) costs zero bytes when nothing's missing.
        //
        // The acceptance rule is centralized in `recipe_file_is_placed` so
        // that this skip path and `catalog_entry_installed` (the catalog
        // API's `installed: bool` predicate) cannot drift apart — otherwise
        // the SPA's Repair button would silently re-pull a model the
        // predicate just claimed was installed.
        let already_placed = recipe_file_is_placed(dest_path, file);
        if already_placed {
            let size_bytes = file
                .size_bytes
                .unwrap_or_else(|| std::fs::metadata(dest_path).map(|m| m.len()).unwrap_or(0));
            if let Some(cb) = progress.as_deref() {
                cb(DownloadProgressEvent::FileStart {
                    filename: file.dest.to_string(),
                    file_index,
                    total_files,
                    size_bytes,
                    batch_bytes_downloaded,
                    batch_bytes_total,
                    batch_elapsed_ms: started.elapsed().as_millis() as u64,
                });
            }
            batch_bytes_downloaded = batch_bytes_downloaded.saturating_add(size_bytes);
            if let Some(cb) = progress.as_deref() {
                cb(DownloadProgressEvent::FileDone {
                    filename: file.dest.to_string(),
                    file_index,
                    total_files,
                    batch_bytes_downloaded,
                    batch_bytes_total,
                    batch_elapsed_ms: started.elapsed().as_millis() as u64,
                });
            }
            continue;
        }

        let mut req = client.get(file.url);
        if let RecipeAuth::Bearer(token) = &auth {
            req = req.bearer_auth(token);
        }
        let resp = req
            .send()
            .await
            .map_err(|e| DownloadError::RecipeTransport {
                url: file.url.to_string(),
                source: e,
            })?;
        if !resp.status().is_success() {
            let status = resp.status().as_u16();
            let body = resp.text().await.ok().map(|b| {
                let mut t = b.trim().to_string();
                if t.len() > 200 {
                    t.truncate(200);
                }
                t
            });
            return Err(DownloadError::RecipeHttp {
                url: file.url.to_string(),
                status,
                body,
            });
        }

        let content_length = resp.content_length();
        let size_bytes = file.size_bytes.or(content_length).unwrap_or(0);

        if let Some(cb) = progress.as_deref() {
            cb(DownloadProgressEvent::FileStart {
                filename: file.dest.to_string(),
                file_index,
                total_files,
                size_bytes,
                batch_bytes_downloaded,
                batch_bytes_total,
                batch_elapsed_ms: started.elapsed().as_millis() as u64,
            });
        }

        let mut bytes_downloaded: u64 = 0;
        let mut out = std::fs::File::create(dest_path).map_err(|e| {
            DownloadError::FilePlacement(format!("failed to create {}: {e}", dest_path.display()))
        })?;
        let mut resp = resp;
        // Throttle FileProgress to once per RECIPE_PROGRESS_THROTTLE_MS so SSE
        // subscribers and reactive UIs aren't drowned in chunk-rate events
        // (a multi-GB Civitai pull emits hundreds of thousands of chunks).
        // Mirrors the throttle in the manifest-pull `CallbackProgress::update`.
        let mut last_emit = Instant::now();
        let mut last_emit_bytes: u64 = 0;
        while let Some(chunk) = resp
            .chunk()
            .await
            .map_err(|e| DownloadError::RecipeTransport {
                url: file.url.to_string(),
                source: e,
            })?
        {
            out.write_all(&chunk).map_err(|e| {
                DownloadError::FilePlacement(format!(
                    "failed to write to {}: {e}",
                    dest_path.display()
                ))
            })?;
            bytes_downloaded += chunk.len() as u64;
            batch_bytes_downloaded += chunk.len() as u64;
            if let Some(cb) = progress.as_deref() {
                let now = Instant::now();
                let elapsed = now.duration_since(last_emit).as_millis();
                if elapsed >= RECIPE_PROGRESS_THROTTLE_MS as u128 {
                    last_emit = now;
                    last_emit_bytes = bytes_downloaded;
                    cb(DownloadProgressEvent::FileProgress {
                        filename: file.dest.to_string(),
                        file_index,
                        bytes_downloaded,
                        bytes_total: size_bytes,
                        batch_bytes_downloaded,
                        batch_bytes_total,
                        batch_elapsed_ms: started.elapsed().as_millis() as u64,
                    });
                }
            }
        }
        // Final progress emit so the file's last few chunks aren't swallowed
        // by the throttle (FileDone fires below, but it doesn't carry the
        // intermediate bytes_downloaded value — drawers that key off
        // FileProgress for their byte counter would otherwise stall short).
        if let Some(cb) = progress.as_deref() {
            if bytes_downloaded > last_emit_bytes {
                cb(DownloadProgressEvent::FileProgress {
                    filename: file.dest.to_string(),
                    file_index,
                    bytes_downloaded,
                    bytes_total: size_bytes,
                    batch_bytes_downloaded,
                    batch_bytes_total,
                    batch_elapsed_ms: started.elapsed().as_millis() as u64,
                });
            }
        }
        // Drop file handle so the SHA-256 read sees a flushed file.
        drop(out);

        // Hash-and-mark on success. Mirror of the manifest-pull path: when
        // the recipe declares an expected hash we compare and bail on
        // mismatch; either way we end up writing the `.sha256-verified`
        // marker so `Config::manifest_files_exist` recognises this file
        // as a positively-attested install (not just "exists on disk").
        // Skipped under `skip_verify` — the user has explicitly asked us
        // not to read the file, so we have nothing to attest.
        if !opts.skip_verify {
            let actual = pinned_file_digest(dest_path).map_err(|e| {
                DownloadError::Other(format!(
                    "failed to compute SHA-256 for {}: {e}",
                    dest_path.display()
                ))
            })?;
            if let Some(expected) = file.sha256 {
                if !actual.eq_ignore_ascii_case(expected) {
                    let _ = std::fs::remove_file(dest_path);
                    return Err(DownloadError::Sha256Mismatch {
                        filename: file.dest.to_string(),
                        expected: expected.to_string(),
                        actual,
                        model: id.to_string(),
                    });
                }
            }
            if let Err(e) = write_sha256_marker(dest_path, &actual) {
                eprintln!(
                    "warning: failed to write .sha256-verified marker for {}: {e}",
                    file.dest
                );
            }
        }

        if let Some(cb) = progress.as_deref() {
            cb(DownloadProgressEvent::FileDone {
                filename: file.dest.to_string(),
                file_index,
                total_files,
                batch_bytes_downloaded,
                batch_bytes_total,
                batch_elapsed_ms: started.elapsed().as_millis() as u64,
            });
        }
    }

    Ok(resolved.to_vec())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn identical_hf_files_serialize_acquisition_and_placement() {
        use std::sync::atomic::{AtomicUsize, Ordering};

        let filename = format!("shared-{}.safetensors", uuid::Uuid::new_v4());
        let temp = tempfile::tempdir().unwrap();
        let source = temp.path().join("blob");
        let destination = temp.path().join("shared").join("encoder.safetensors");
        std::fs::write(&source, b"shared encoder").unwrap();

        let active = Arc::new(AtomicUsize::new(0));
        let peak = Arc::new(AtomicUsize::new(0));
        let operation = || {
            let active = active.clone();
            let peak = peak.clone();
            let source = source.clone();
            let destination = destination.clone();
            async move {
                let now_active = active.fetch_add(1, Ordering::SeqCst) + 1;
                peak.fetch_max(now_active, Ordering::SeqCst);
                tokio::time::sleep(std::time::Duration::from_millis(25)).await;
                let result = hardlink_or_copy(&source, &destination);
                active.fetch_sub(1, Ordering::SeqCst);
                result
            }
        };

        let first = with_hf_file_download_flight("Qwen/Qwen-Image-2512", &filename, operation());
        let second = with_hf_file_download_flight("Qwen/Qwen-Image-2512", &filename, operation());
        let (first_result, second_result) = tokio::join!(first, second);

        first_result.unwrap();
        second_result.unwrap();
        assert_eq!(peak.load(Ordering::SeqCst), 1);
        assert_eq!(std::fs::read(destination).unwrap(), b"shared encoder");
    }

    #[test]
    fn hidden_ltx2_adapters_use_files_only_pulls() {
        let adapters = crate::manifest::known_manifests()
            .iter()
            .filter(|manifest| {
                matches!(
                    manifest.family.as_str(),
                    "ltx2-control" | "ltx2-camera-control"
                )
            })
            .collect::<Vec<_>>();
        assert!(!adapters.is_empty());
        assert!(adapters
            .iter()
            .any(|manifest| manifest.name == "ltx2-camera-control-dolly-right-19b"));
        for manifest in adapters {
            assert!(manifest.is_auxiliary());
            assert!(manifest_uses_files_only_pull(manifest), "{}", manifest.name);
        }

        assert!(!manifest_uses_files_only_pull(
            crate::manifest::find_manifest("ltx-2-19b-distilled:fp8").unwrap()
        ));
        assert!(!manifest_uses_files_only_pull(
            crate::manifest::find_manifest("controlnet-canny-sd15:fp16").unwrap()
        ));
    }

    #[test]
    fn pulid_bundle_uses_the_files_only_pull() {
        let manifest =
            crate::manifest::find_manifest(crate::manifest::PULID_FLUX_MANIFEST).unwrap();
        assert!(manifest_uses_files_only_pull(manifest));
    }

    #[test]
    fn restricted_files_are_refused_until_the_license_is_accepted() {
        use crate::license_acceptance::{
            record_acceptance, ThirdPartyLicense, INSIGHTFACE_ANTELOPEV2,
        };

        let manifest =
            crate::manifest::find_manifest(crate::manifest::PULID_FLUX_MANIFEST).unwrap();
        let home = tempfile::tempdir().unwrap();
        let gate =
            |home: Option<&std::path::Path>| require_manifest_licenses_accepted_in(manifest, home);

        let error = gate(Some(home.path())).expect_err("an unaccepted license refuses the pull");
        match &error {
            DownloadError::LicenseNotAccepted {
                license_id,
                message,
            } => {
                assert_eq!(license_id, INSIGHTFACE_ANTELOPEV2.id);
                assert!(message.contains("non-commercial research"));
                assert!(message.contains("--accept-license insightface-antelopev2"));
            }
            other => panic!("unexpected error: {other:?}"),
        }

        // A stale record — accepted against terms that have since changed —
        // must not unlock the download.
        const CHANGED_TERMS: &str =
            "0000000000000000000000000000000000000000000000000000000000000000";
        let stale = ThirdPartyLicense {
            sha256: CHANGED_TERMS,
            ..INSIGHTFACE_ANTELOPEV2
        };
        record_acceptance(home.path(), &stale).unwrap();
        assert!(gate(Some(home.path())).is_err());

        record_acceptance(home.path(), &INSIGHTFACE_ANTELOPEV2).unwrap();
        gate(Some(home.path())).expect("an accepted license permits the pull");

        // An unresolvable Mold data root fails closed rather than open.
        assert!(gate(None).is_err());
    }

    #[test]
    fn unrestricted_manifests_are_never_license_gated() {
        for name in ["flux2-klein:q8", "controlnet-canny-sd15:fp16"] {
            let manifest = crate::manifest::find_manifest(name).unwrap();
            // Not even an unresolvable data root gates an unrestricted model.
            require_manifest_licenses_accepted_in(manifest, None).unwrap_or_else(|error| {
                panic!("{name} must not be license gated: {error}");
            });
        }
    }

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
    fn sync_callback_progress_reports_real_accumulated_bytes() {
        let events = Arc::new(Mutex::new(Vec::new()));
        let events_for_callback = events.clone();
        let callback: DownloadProgressCallback = Arc::new(move |event| {
            events_for_callback.lock().unwrap().push(event);
        });
        let mut progress = SyncCallbackProgress::new(callback);
        hf_hub::api::Progress::init(&mut progress, 100, "encoder.gguf");
        hf_hub::api::Progress::update(&mut progress, 40);
        hf_hub::api::Progress::update(&mut progress, 60);
        hf_hub::api::Progress::finish(&mut progress);

        let events = events.lock().unwrap();
        assert!(matches!(
            &events[0],
            DownloadProgressEvent::FileStart {
                filename,
                size_bytes: 100,
                ..
            } if filename == "encoder.gguf"
        ));
        assert!(events.iter().any(|event| matches!(
            event,
            DownloadProgressEvent::FileProgress {
                bytes_downloaded: 100,
                bytes_total: 100,
                ..
            }
        )));
        assert!(matches!(
            events.last(),
            Some(DownloadProgressEvent::FileDone {
                batch_bytes_downloaded: 100,
                batch_bytes_total: 100,
                ..
            })
        ));
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

    fn compliance_gated_manifest(name: &str, family: &str, repo: &str) -> ModelManifest {
        use crate::manifest::{ManifestDefaults, ModelFile};

        ModelManifest {
            name: name.to_string(),
            family: family.to_string(),
            description: "policy fixture".to_string(),
            files: vec![ModelFile {
                hf_repo: repo.to_string(),
                hf_filename: "weights.safetensors".to_string(),
                component: ModelComponent::Transformer,
                size_bytes: 1,
                gated: false,
                sha256: None,
            }],
            defaults: ManifestDefaults {
                steps: 1,
                guidance: 1.0,
                width: 32,
                height: 32,
                is_schnell: false,
                scheduler: None,
                negative_prompt: None,
                frames: None,
                fps: None,
                source_image: None,
            },
            hidden: true,
        }
    }

    #[test]
    fn reviewed_h3_manifest_is_accepted_for_upstream_acquisition() {
        let manifest = crate::manifest::find_manifest(crate::minimax_h3::FL2VA_COMFY).unwrap();
        require_manifest_acquisition(manifest).unwrap();
        // Acquisition, deliberately not activation: this build has no engine
        // (#1276), and the whole point of the reviewed manifest is that it
        // still downloads, verifies, and stores.
        crate::require_model_acquisition(&manifest.name, Some(&manifest.family)).unwrap();
    }

    #[test]
    fn h3_repo_identity_cannot_bypass_the_pinned_manifest() {
        let manifest = compliance_gated_manifest("renamed-model", "custom", "Comfy-Org/MiniMax-H3");
        assert!(require_manifest_acquisition(&manifest).is_err());
    }

    #[test]
    fn disk_preflight_rejects_truncated_clean_files() {
        let mut manifest = compliance_gated_manifest("space-test", "flux", "example/model");
        manifest.files[0].size_bytes = 4;
        let temp = tempfile::tempdir().unwrap();
        let clean = temp
            .path()
            .join(crate::manifest::storage_path(&manifest, &manifest.files[0]));
        std::fs::create_dir_all(clean.parent().unwrap()).unwrap();
        std::fs::write(&clean, [0u8; 2]).unwrap();
        assert_eq!(
            required_download_bytes_in(&manifest, temp.path(), false).unwrap(),
            4
        );
        std::fs::write(&clean, [0u8; 4]).unwrap();
        assert_eq!(
            required_download_bytes_in(&manifest, temp.path(), false).unwrap(),
            0
        );
    }

    #[test]
    fn failed_ltx25_qualification_keeps_the_repair_marker() {
        let temp = tempfile::tempdir().unwrap();
        let marker = pulling_marker_path_in(temp.path(), crate::ltx25_manifest::DISTILLED);
        std::fs::create_dir_all(marker.parent().unwrap()).unwrap();
        std::fs::write(&marker, crate::ltx25_manifest::DISTILLED).unwrap();
        let manifest = crate::manifest::find_manifest(crate::ltx25_manifest::DISTILLED).unwrap();
        assert!(qualify_downloaded_contract_in(manifest, temp.path()).is_err());
        assert!(marker.exists());
    }

    #[test]
    fn arbitrary_h3_single_files_remain_outside_pinned_manifest_acquisition() {
        for (repo, filename, target) in [
            ("MiniMaxAI/MiniMax-H3", "weights.safetensors", None),
            (
                "example/renamed-model",
                "MiniMax-H3/weights.safetensors",
                None,
            ),
            (
                "example/renamed-model",
                "weights.safetensors",
                Some("shared/MiniMax-H3"),
            ),
        ] {
            assert!(require_single_file_acquisition(repo, filename, target).is_err());
        }
    }

    #[test]
    fn sync_download_policy_keeps_h3_lookalikes_available() {
        for (repo, filename, target) in [
            ("example/minimax-h30", "weights.safetensors", None),
            ("example/model", "minimaxh30.safetensors", None),
            ("example/model", "weights.safetensors", Some("shared/h3")),
        ] {
            require_single_file_acquisition(repo, filename, target)
                .unwrap_or_else(|_| panic!("lookalike must remain available: {repo}/{filename}"));
        }
    }

    /// Mutex to serialize tests that mutate `HF_TOKEN` — `set_var`/`remove_var`
    /// are process-global and not thread-safe, so parallel tests race.
    static HF_TOKEN_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

    // ---------------------------------------------------------------------
    // FLUX city96-format GGUF pull-time warning tests.
    // `gguf_header_contains_tensor` just does a bounded substring scan after
    // validating the "GGUF" magic — no real GGUF parsing — so tests can write
    // synthetic files that only satisfy those two properties.
    // `flux_reference_warning` is pure over (manifest, models_dir), so we can
    // drive every branch without touching process-global state.
    // ---------------------------------------------------------------------

    fn write_fake_gguf(path: &std::path::Path, tensor_names: &[&str]) {
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent).unwrap();
        }
        let mut buf = Vec::with_capacity(4096);
        buf.extend_from_slice(b"GGUF");
        // Pad a couple hundred bytes of synthetic header bytes, then include
        // every tensor name as a plain UTF-8 substring so the scanner finds it.
        buf.extend(std::iter::repeat_n(0u8, 256));
        for name in tensor_names {
            buf.extend_from_slice(name.as_bytes());
            buf.push(0);
        }
        std::fs::write(path, &buf).unwrap();
    }

    fn tmp_dir(tag: &str) -> std::path::PathBuf {
        let dir = std::env::temp_dir().join(format!(
            "mold-dl-{tag}-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    fn fake_flux_gguf_manifest(name: &str, filename: &str, is_schnell: bool) -> ModelManifest {
        use crate::manifest::{ManifestDefaults, ModelFile};
        ModelManifest {
            name: name.to_string(),
            family: "flux".to_string(),
            description: "test".to_string(),
            files: vec![ModelFile {
                hf_repo: "test/repo".to_string(),
                hf_filename: filename.to_string(),
                component: ModelComponent::Transformer,
                size_bytes: 0,
                gated: false,
                sha256: None,
            }],
            defaults: ManifestDefaults {
                steps: 20,
                guidance: 3.5,
                width: 1024,
                height: 1024,
                is_schnell,
                scheduler: None,
                negative_prompt: None,
                frames: None,
                fps: None,
                source_image: None,
            },
            hidden: false,
        }
    }

    #[test]
    fn gguf_header_contains_tensor_false_for_missing_file() {
        let path = std::env::temp_dir().join(format!(
            "mold-dl-nofile-{}-{}.gguf",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        assert!(!gguf_header_contains_tensor(&path, "img_in.weight"));
    }

    #[test]
    fn gguf_header_contains_tensor_false_for_non_gguf_magic() {
        let dir = tmp_dir("nonmagic");
        let path = dir.join("not-a-gguf.gguf");
        std::fs::write(&path, b"SAFE\0\0\0\0img_in.weight\0").unwrap();
        assert!(!gguf_header_contains_tensor(&path, "img_in.weight"));
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn gguf_header_contains_tensor_finds_needle_after_magic() {
        let dir = tmp_dir("finds");
        let path = dir.join("has.gguf");
        write_fake_gguf(&path, &["img_in.weight", "time_in.in_layer.weight"]);
        assert!(gguf_header_contains_tensor(&path, "img_in.weight"));
        assert!(gguf_header_contains_tensor(
            &path,
            "time_in.in_layer.weight"
        ));
        assert!(!gguf_header_contains_tensor(
            &path,
            "guidance_in.in_layer.weight"
        ));
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn flux_reference_warning_noop_for_non_flux_family() {
        use crate::manifest::{ManifestDefaults, ModelFile};
        let dir = tmp_dir("non-flux");
        let manifest = ModelManifest {
            name: "sd15:fp16".to_string(),
            family: "sd15".to_string(),
            description: "test".to_string(),
            files: vec![ModelFile {
                hf_repo: "test/repo".to_string(),
                hf_filename: "model.gguf".to_string(),
                component: ModelComponent::Transformer,
                size_bytes: 0,
                gated: false,
                sha256: None,
            }],
            defaults: ManifestDefaults {
                steps: 25,
                guidance: 7.5,
                width: 512,
                height: 512,
                is_schnell: false,
                scheduler: None,
                negative_prompt: None,
                frames: None,
                fps: None,
                source_image: None,
            },
            hidden: false,
        };
        assert!(flux_reference_warning(&manifest, &dir).is_none());
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn flux_reference_warning_noop_for_safetensors_transformer() {
        let dir = tmp_dir("safetensors");
        // Non-GGUF filename is ignored even when everything else matches.
        let manifest = fake_flux_gguf_manifest("ultra-test:bf16", "model.safetensors", false);
        assert!(flux_reference_warning(&manifest, &dir).is_none());
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn flux_reference_warning_noop_when_file_absent() {
        let dir = tmp_dir("absent");
        let manifest = fake_flux_gguf_manifest("ultra-absent:q8", "ultra-absent-q8.gguf", false);
        // Transformer file not written — function should silently return None.
        assert!(flux_reference_warning(&manifest, &dir).is_none());
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn flux_reference_warning_noop_when_transformer_is_complete() {
        let dir = tmp_dir("complete");
        let manifest =
            fake_flux_gguf_manifest("ultra-complete:q8", "ultra-complete-q8.gguf", false);
        // A "complete" GGUF has img_in.weight, so no patching needed.
        let xformer = dir.join(crate::manifest::storage_path(&manifest, &manifest.files[0]));
        write_fake_gguf(&xformer, &["img_in.weight", "guidance_in.in_layer.weight"]);
        assert!(flux_reference_warning(&manifest, &dir).is_none());
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn flux_reference_warning_fires_for_city96_dev_without_reference() {
        let dir = tmp_dir("city96-dev");
        let manifest = fake_flux_gguf_manifest("ultra-v4:q8", "ultra-v4-q8.gguf", false);
        let xformer = dir.join(crate::manifest::storage_path(&manifest, &manifest.files[0]));
        // city96-format: diffusion blocks but no embedding layers.
        write_fake_gguf(&xformer, &["double_blocks.0.img_mod.lin.weight"]);

        let msg = flux_reference_warning(&manifest, &dir)
            .expect("city96-format dev GGUF without reference must emit warning");
        assert!(msg.contains("ultra-v4-q8.gguf"));
        assert!(msg.contains("ultra-v4:q8"));
        assert!(msg.contains("mold pull flux-dev:q8"));
        assert!(
            msg.contains("guidance_in"),
            "dev target message must mention guidance_in: {msg}"
        );
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn flux_reference_warning_fires_for_city96_schnell_without_reference() {
        let dir = tmp_dir("city96-schnell");
        let manifest = fake_flux_gguf_manifest("ultra-schnell:q8", "ultra-schnell-q8.gguf", true);
        let xformer = dir.join(crate::manifest::storage_path(&manifest, &manifest.files[0]));
        write_fake_gguf(&xformer, &["double_blocks.0.img_mod.lin.weight"]);

        let msg = flux_reference_warning(&manifest, &dir)
            .expect("city96-format schnell GGUF without reference must emit warning");
        // Schnell target: message accepts flux-schnell OR flux-dev as reference.
        assert!(msg.contains("ultra-schnell-q8.gguf"));
        assert!(msg.contains("mold pull flux-dev:q8"));
        assert!(!msg.contains("guidance_in"));
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn flux_reference_warning_silenced_when_dev_reference_exists() {
        let dir = tmp_dir("has-dev-ref");
        let manifest = fake_flux_gguf_manifest("ultra-v4:q8", "ultra-v4-q8.gguf", false);
        let xformer = dir.join(crate::manifest::storage_path(&manifest, &manifest.files[0]));
        write_fake_gguf(&xformer, &["double_blocks.0.img_mod.lin.weight"]);

        // Place a fake "downloaded" complete flux-dev:q8 alongside.
        let dev_manifest = crate::manifest::find_manifest("flux-dev:q8")
            .expect("flux-dev:q8 must exist in the static manifest catalog");
        let dev_xformer_file = dev_manifest
            .files
            .iter()
            .find(|f| f.component == ModelComponent::Transformer)
            .expect("flux-dev:q8 must declare a Transformer file");
        let dev_path = dir.join(crate::manifest::storage_path(
            dev_manifest,
            dev_xformer_file,
        ));
        write_fake_gguf(&dev_path, &["img_in.weight", "guidance_in.in_layer.weight"]);

        assert!(
            flux_reference_warning(&manifest, &dir).is_none(),
            "warning must be silenced when a complete flux-dev reference is downloaded"
        );
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn flux_reference_warning_rejects_schnell_as_reference_for_dev_target() {
        // Regression: schnell has img_in but not guidance_in. Pre-fix, it was
        // accepted as a reference; then ensure_gguf_embeddings failed mid-patch.
        let dir = tmp_dir("schnell-only-for-dev");
        let manifest = fake_flux_gguf_manifest("ultra-v4:q8", "ultra-v4-q8.gguf", false);
        let xformer = dir.join(crate::manifest::storage_path(&manifest, &manifest.files[0]));
        write_fake_gguf(&xformer, &["double_blocks.0.img_mod.lin.weight"]);

        // Drop a schnell GGUF that looks "valid" (has img_in, lacks guidance_in).
        let schnell_manifest = crate::manifest::find_manifest("flux-schnell:q8")
            .expect("flux-schnell:q8 must exist in the static manifest catalog");
        let schnell_xformer_file = schnell_manifest
            .files
            .iter()
            .find(|f| f.component == ModelComponent::Transformer)
            .expect("flux-schnell:q8 must declare a Transformer file");
        let schnell_path = dir.join(crate::manifest::storage_path(
            schnell_manifest,
            schnell_xformer_file,
        ));
        write_fake_gguf(&schnell_path, &["img_in.weight"]);

        let msg = flux_reference_warning(&manifest, &dir)
            .expect("dev target must not accept schnell as reference; warning should fire");
        assert!(msg.contains("mold pull flux-dev:q8"));
        std::fs::remove_dir_all(&dir).ok();
    }

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

    /// Civitai's API returns SHA-256 hashes in uppercase hex
    /// (`DD08FA32...`), while `compute_sha256` formats with `{:x}` so it
    /// produces lowercase. A literal string comparison treats these as
    /// distinct, so every Civitai pull bailed out with a "mismatch" even
    /// when the file was bit-identical to what was advertised. The
    /// verifier must be hex-case-insensitive.
    #[test]
    fn verify_sha256_is_hex_case_insensitive() {
        let dir = std::env::temp_dir().join("mold_test_sha256_case");
        let _ = std::fs::create_dir_all(&dir);
        let path = dir.join("test_file.bin");
        std::fs::write(&path, b"hello world").unwrap();
        let lower = "b94d27b9934d3e08a52e52d7da7dabfac484efe37a5380ee9088f7ace2efcde9";
        let upper = "B94D27B9934D3E08A52E52D7DA7DABFAC484EFE37A5380EE9088F7ACE2EFCDE9";
        let mixed = "B94d27b9934D3e08a52E52d7Da7dabfac484EFE37A5380ee9088f7Ace2efcDE9";
        assert!(
            verify_sha256(&path, lower).unwrap(),
            "lowercase digest must match"
        );
        assert!(
            verify_sha256(&path, upper).unwrap(),
            "uppercase digest must match (Civitai-style)",
        );
        assert!(
            verify_sha256(&path, mixed).unwrap(),
            "mixed-case must match"
        );
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// The pinned-digest read counter is process-global, so the tests that
    /// assert on it have to be the only ones hashing at the time.
    fn pinned_digest_lock() -> std::sync::MutexGuard<'static, ()> {
        static LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());
        LOCK.lock().unwrap_or_else(|poisoned| poisoned.into_inner())
    }

    /// A `.sha256-verified` marker is an ordinary file beside the artifact.
    /// The model-storage invariant explicitly supports group-writable model
    /// roots (`0664` and collaborative umasks are valid), so anyone who can
    /// drop weights there can also drop a sidecar naming the expected digest —
    /// and a stale marker can outlive a replace. A pinned check that trusted
    /// it would authenticate nothing.
    #[test]
    fn a_forged_marker_is_never_accepted_as_proof_of_content() {
        let _serial = pinned_digest_lock();
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("adapter.safetensors");
        std::fs::write(&path, b"attacker bytes").unwrap();
        let expected = {
            let honest = dir.path().join("honest.bin");
            std::fs::write(&honest, b"the real pinned bytes").unwrap();
            compute_sha256(&honest).unwrap()
        };
        // The forgery: bytes that are not the pin, attested as if they were.
        write_sha256_marker(&path, &expected).unwrap();
        assert_eq!(recorded_sha256_marker(&path).as_deref(), Some(&*expected));

        let error = verify_pinned_file(&path, &expected, "adapter.safetensors", "pinned-bundle")
            .expect_err("a forged marker must not authenticate the bytes beside it");

        assert!(
            matches!(error, DownloadError::Sha256Mismatch { .. }),
            "{error}"
        );
        let rendered = error.to_string();
        assert!(rendered.contains("adapter.safetensors"), "{rendered}");
        assert!(rendered.contains(&expected), "{rendered}");
        assert!(!path.exists(), "the rejected bytes must be removed");
        assert!(
            !has_sha256_marker(&path),
            "the forged attestation must not outlive the file it lied about"
        );

        // Read-only form: same verdict, and it deletes nothing.
        std::fs::write(&path, b"attacker bytes").unwrap();
        write_sha256_marker(&path, &expected).unwrap();
        assert!(!pinned_file_matches(&path, &expected));
        assert!(path.exists());
    }

    /// Hashing every admission would re-read a 2.3 GB bundle per job. The memo
    /// is keyed on the descriptor's own identity, so an unchanged file costs
    /// one read per process.
    #[test]
    fn an_unchanged_pinned_file_is_read_once_per_process() {
        let _serial = pinned_digest_lock();
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("stable.bin");
        std::fs::write(&path, b"the real pinned bytes").unwrap();
        let expected = compute_sha256(&path).unwrap();

        let before = pinned_digest_hash_count_for(&path);
        verify_pinned_file(&path, &expected, "stable.bin", "pinned-bundle").unwrap();
        let after_first = pinned_digest_hash_count_for(&path);
        assert_eq!(
            after_first,
            before + 1,
            "the first check must read the file"
        );

        for _ in 0..3 {
            verify_pinned_file(&path, &expected, "stable.bin", "pinned-bundle").unwrap();
            assert!(pinned_file_matches(&path, &expected));
        }
        assert_eq!(
            pinned_digest_hash_count_for(&path),
            after_first,
            "an unchanged file must not be re-read once it is memoized"
        );
        // The marker is still written — `manifest_files_exist` and the
        // partial-cleanup sweep read it as a "fully written" signal. It is
        // just never read back as proof of content.
        assert_eq!(recorded_sha256_marker(&path).as_deref(), Some(&*expected));
    }

    #[test]
    fn concurrent_pinned_checks_share_one_physical_hash() {
        let _serial = pinned_digest_lock();
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("single-flight.bin");
        std::fs::write(&path, vec![0x5a; 4 * 1024 * 1024]).unwrap();
        let expected = compute_sha256(&path).unwrap();
        let before = pinned_digest_hash_count_for(&path);
        let barrier = std::sync::Arc::new(std::sync::Barrier::new(5));
        let mut threads = Vec::new();
        for _ in 0..4 {
            let path = path.clone();
            let expected = expected.clone();
            let barrier = barrier.clone();
            threads.push(std::thread::spawn(move || {
                barrier.wait();
                assert!(pinned_file_matches(&path, &expected));
            }));
        }
        barrier.wait();
        for thread in threads {
            thread.join().unwrap();
        }
        assert_eq!(
            pinned_digest_hash_count_for(&path),
            before + 1,
            "concurrent misses for one file identity must share one body read"
        );
        let identity =
            pinned_file_identity(&crate::secure_file::open_regular_file_no_follow(&path).unwrap())
                .unwrap();
        assert!(
            !pinned_digest_flights()
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner())
                .contains_key(&identity),
            "the last flight participant must evict the completed identity"
        );
    }

    #[test]
    fn opened_digest_never_authenticates_a_path_replacement() {
        let _serial = pinned_digest_lock();
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("artifact.bin");
        let original = dir.path().join("original.bin");
        std::fs::write(&path, b"reviewed bytes").unwrap();
        let file = crate::secure_file::open_regular_file_no_follow(&path).unwrap();
        std::fs::rename(&path, &original).unwrap();
        std::fs::write(&path, b"attacker bytes").unwrap();

        let error = pinned_file_digest_from_open_file(&path, &file, |_, _| Ok(()))
            .expect_err("the retained descriptor and current path must name one identity");
        assert!(error.to_string().contains("path changed"), "{error:#}");
    }

    #[cfg(unix)]
    #[test]
    fn durable_attestation_survives_process_cache_loss_and_rejects_mutation() {
        use std::os::unix::fs::{OpenOptionsExt, PermissionsExt};

        let _serial = pinned_digest_lock();
        let home = tempfile::tempdir().unwrap();
        std::fs::set_permissions(home.path(), std::fs::Permissions::from_mode(0o700)).unwrap();
        let path = home.path().join("model.bin");
        std::fs::write(&path, b"verified model bytes").unwrap();
        let file = crate::secure_file::open_regular_file_no_follow(&path).unwrap();
        let identity = pinned_file_identity(&file).unwrap();
        let digest = crate::secure_file::sha256_open_file(&file).unwrap();
        let attestation_dir = private_attestation_dir_at(home.path(), true).unwrap();
        let absolute = absolute_pinned_path(&path).unwrap();
        let target = durable_pinned_digest_path(&attestation_dir, &absolute);
        let record = DurablePinnedDigest {
            schema: DURABLE_PINNED_DIGEST_SCHEMA,
            path: absolute,
            identity,
            sha256: digest.clone(),
        };
        let record_file = std::fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .mode(0o600)
            .open(&target)
            .unwrap();
        serde_json::to_writer(record_file, &record).unwrap();

        pinned_digest_cache()
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .remove(&identity);
        assert_eq!(
            read_durable_pinned_digest_from_dir(&attestation_dir, &path, identity).as_deref(),
            Some(digest.as_str()),
            "a valid private attestation must replace the process-lifetime body read"
        );

        std::fs::write(&path, b"tampered model bytes").unwrap();
        let changed = crate::secure_file::open_regular_file_no_follow(&path).unwrap();
        let changed_identity = pinned_file_identity(&changed).unwrap();
        assert_ne!(changed_identity, identity);
        assert_eq!(
            read_durable_pinned_digest_from_dir(&attestation_dir, &path, changed_identity),
            None,
            "a durable digest is valid only for the exact attested file identity"
        );
    }

    #[cfg(unix)]
    #[test]
    fn durable_attestation_is_disabled_in_renamable_parent() {
        use std::os::unix::fs::PermissionsExt;

        let home = tempfile::tempdir().unwrap();
        std::fs::set_permissions(home.path(), std::fs::Permissions::from_mode(0o770)).unwrap();
        assert!(private_attestation_dir_at(home.path(), true).is_none());
        assert!(!home.path().join(DURABLE_PINNED_DIGEST_DIR).exists());
    }

    #[cfg(unix)]
    #[test]
    fn dedicated_private_attestation_store_works_with_a_shared_mold_home() {
        use std::os::unix::fs::{MetadataExt, PermissionsExt};

        let shared_home = tempfile::tempdir().unwrap();
        std::fs::set_permissions(shared_home.path(), std::fs::Permissions::from_mode(0o770))
            .unwrap();
        assert!(private_attestation_dir_at(shared_home.path(), true).is_none());

        let private_state = tempfile::tempdir().unwrap();
        std::fs::set_permissions(private_state.path(), std::fs::Permissions::from_mode(0o700))
            .unwrap();
        let configured = private_state.path().join("attestations");
        let actual = private_attestation_dir_exact_at(&configured, true)
            .expect("a dedicated owner-private store must not depend on MOLD_HOME permissions");
        assert_eq!(actual, configured);
        assert_eq!(std::fs::metadata(&actual).unwrap().mode() & 0o777, 0o700);
    }

    /// The memo must not become the hole the marker was: a file whose bytes
    /// change has a new identity (`ctime` alone guarantees it — `utimes` can
    /// forge `mtime`, nothing can set `ctime`), so it is re-read and refused.
    #[test]
    fn a_rewritten_pinned_file_is_reread_and_refused() {
        let _serial = pinned_digest_lock();
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("swapped.bin");
        std::fs::write(&path, b"the real pinned bytes").unwrap();
        let expected = compute_sha256(&path).unwrap();
        verify_pinned_file(&path, &expected, "swapped.bin", "pinned-bundle").unwrap();
        // Same path, same length, different content — the shape of an
        // in-place substitution in a shared models root.
        std::fs::write(&path, b"the fake pinned bytes").unwrap();
        let changed_identity =
            pinned_file_identity(&crate::secure_file::open_regular_file_no_follow(&path).unwrap())
                .unwrap();
        let before_changed = pinned_digest_hash_count_for_identity(changed_identity);
        let error = verify_pinned_file(&path, &expected, "swapped.bin", "pinned-bundle")
            .expect_err("replaced bytes must be caught, not served from the memo");

        assert!(
            matches!(error, DownloadError::Sha256Mismatch { .. }),
            "{error}"
        );
        assert_eq!(
            pinned_digest_hash_count_for_identity(changed_identity),
            before_changed + 1,
            "a changed file identity must force a fresh read"
        );
        assert!(!path.exists());
    }

    /// A symlink is not a regular file, and the check must refuse it rather
    /// than hash whatever it happens to point at.
    #[cfg(unix)]
    #[test]
    fn a_symlinked_pinned_artifact_is_refused_rather_than_followed() {
        let _serial = pinned_digest_lock();
        let dir = tempfile::tempdir().unwrap();
        let real = dir.path().join("real.bin");
        std::fs::write(&real, b"the real pinned bytes").unwrap();
        let expected = compute_sha256(&real).unwrap();
        let link = dir.path().join("link.bin");
        std::os::unix::fs::symlink(&real, &link).unwrap();

        assert!(!pinned_file_matches(&link, &expected));
        let error = verify_pinned_file(&link, &expected, "link.bin", "pinned-bundle")
            .expect_err("a pinned artifact reached through a symlink is not proven");
        assert!(matches!(error, DownloadError::Other(_)), "{error}");
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

    // ── .sha256-verified marker helpers (B1) ─────────────────────────────

    #[test]
    fn sha256_marker_path_appends_suffix() {
        let p = std::path::Path::new("/tmp/foo/model.safetensors");
        let marker = sha256_marker_path(p);
        assert_eq!(
            marker,
            std::path::PathBuf::from("/tmp/foo/model.safetensors.sha256-verified")
        );
    }

    #[test]
    fn sha256_marker_path_handles_dotted_filenames() {
        let p = std::path::Path::new("/tmp/.hidden.bin");
        let marker = sha256_marker_path(p);
        assert_eq!(
            marker,
            std::path::PathBuf::from("/tmp/.hidden.bin.sha256-verified")
        );
    }

    #[test]
    fn write_sha256_marker_creates_file_with_digest() {
        let dir = std::env::temp_dir().join("mold_test_marker_write");
        let _ = std::fs::create_dir_all(&dir);
        let path = dir.join("file.bin");
        std::fs::write(&path, b"hello world").unwrap();
        let digest = "b94d27b9934d3e08a52e52d7da7dabfac484efe37a5380ee9088f7ace2efcde9";
        write_sha256_marker(&path, digest).unwrap();

        let marker = sha256_marker_path(&path);
        assert!(marker.exists(), "marker should exist next to file");
        let content = std::fs::read_to_string(&marker).unwrap();
        assert!(
            content.contains(digest),
            "marker content should contain the digest, got: {content:?}"
        );
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn write_sha256_marker_is_idempotent() {
        let dir = std::env::temp_dir().join("mold_test_marker_idempotent");
        let _ = std::fs::create_dir_all(&dir);
        let path = dir.join("file.bin");
        std::fs::write(&path, b"x").unwrap();
        let digest = "2d711642b726b04401627ca9fbac32f5c8530fb1903cc4db02258717921a4881";
        write_sha256_marker(&path, digest).unwrap();
        // Second call must not fail.
        write_sha256_marker(&path, digest).unwrap();
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn has_sha256_marker_reflects_existence() {
        let dir = std::env::temp_dir().join("mold_test_marker_has");
        let _ = std::fs::create_dir_all(&dir);
        let path = dir.join("file.bin");
        std::fs::write(&path, b"x").unwrap();
        assert!(!has_sha256_marker(&path), "no marker yet");
        write_sha256_marker(&path, "deadbeef").unwrap();
        assert!(has_sha256_marker(&path), "marker should exist");
        let _ = std::fs::remove_dir_all(&dir);
    }

    // ── verify_file_integrity now writes a marker on success (B2) ────────

    #[test]
    fn verify_file_integrity_writes_marker_on_match() {
        use crate::manifest::{ModelComponent, ModelFile};
        let dir = std::env::temp_dir().join("mold_test_integrity_writes_marker");
        let _ = std::fs::create_dir_all(&dir);
        let path = dir.join("ok.bin");
        std::fs::write(&path, b"hello world").unwrap();
        let file = ModelFile {
            hf_repo: "test/repo".to_string(),
            hf_filename: "ok.bin".to_string(),
            component: ModelComponent::Transformer,
            size_bytes: 11,
            gated: false,
            sha256: Some("b94d27b9934d3e08a52e52d7da7dabfac484efe37a5380ee9088f7ace2efcde9"),
        };
        verify_file_integrity(&path, &file, "test:q8", false).unwrap();
        assert!(
            has_sha256_marker(&path),
            "marker should be written after a successful verify"
        );
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn verify_file_integrity_writes_marker_when_no_hash_declared() {
        use crate::manifest::{ModelComponent, ModelFile};
        let dir = std::env::temp_dir().join("mold_test_integrity_no_hash_marker");
        let _ = std::fs::create_dir_all(&dir);
        let path = dir.join("ok.bin");
        std::fs::write(&path, b"data").unwrap();
        let file = ModelFile {
            hf_repo: "test/repo".to_string(),
            hf_filename: "ok.bin".to_string(),
            component: ModelComponent::Transformer,
            size_bytes: 4,
            gated: false,
            sha256: None,
        };
        verify_file_integrity(&path, &file, "test:q8", false).unwrap();
        assert!(
            has_sha256_marker(&path),
            "marker must be written even when manifest declares no expected hash \
             (the marker still proves the file finished writing)"
        );
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn verify_file_integrity_no_marker_on_mismatch() {
        use crate::manifest::{ModelComponent, ModelFile};
        let dir = std::env::temp_dir().join("mold_test_integrity_no_marker_on_miss");
        let _ = std::fs::create_dir_all(&dir);
        let path = dir.join("bad.bin");
        std::fs::write(&path, b"corrupted").unwrap();
        let file = ModelFile {
            hf_repo: "test/repo".to_string(),
            hf_filename: "bad.bin".to_string(),
            component: ModelComponent::Transformer,
            size_bytes: 9,
            gated: false,
            sha256: Some("0000000000000000000000000000000000000000000000000000000000000000"),
        };
        let result = verify_file_integrity(&path, &file, "test:q8", false);
        assert!(result.is_err(), "mismatch should error");
        // The corrupted file is removed by verify_file_integrity, but more
        // importantly: there must be no marker pointing at the bad bytes.
        assert!(
            !has_sha256_marker(&path),
            "no marker may exist after a hash mismatch"
        );
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn verify_file_integrity_skip_verify_does_not_write_marker() {
        use crate::manifest::{ModelComponent, ModelFile};
        let dir = std::env::temp_dir().join("mold_test_integrity_skip_no_marker");
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
        // skip_verify = true: we don't know the file is good, so no marker.
        verify_file_integrity(&path, &file, "test:q8", true).unwrap();
        assert!(
            !has_sha256_marker(&path),
            "skip_verify must not produce a marker — we have no integrity guarantee"
        );
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

    // ── Civitai token resolution (round 3) ──────────────────────────────

    static CIVITAI_TOKEN_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

    #[test]
    fn resolve_civitai_token_reads_env_var() {
        let _guard = CIVITAI_TOKEN_LOCK.lock().unwrap();
        let original = std::env::var("CIVITAI_TOKEN").ok();
        std::env::set_var("CIVITAI_TOKEN", "cv_test_token_abc");
        let token = resolve_civitai_token();
        match &original {
            Some(v) => std::env::set_var("CIVITAI_TOKEN", v),
            None => std::env::remove_var("CIVITAI_TOKEN"),
        }
        assert_eq!(token, Some("cv_test_token_abc".to_string()));
    }

    #[test]
    fn resolve_civitai_token_ignores_empty() {
        let _guard = CIVITAI_TOKEN_LOCK.lock().unwrap();
        let original = std::env::var("CIVITAI_TOKEN").ok();
        std::env::set_var("CIVITAI_TOKEN", "  ");
        let token = resolve_civitai_token();
        match &original {
            Some(v) => std::env::set_var("CIVITAI_TOKEN", v),
            None => std::env::remove_var("CIVITAI_TOKEN"),
        }
        assert_eq!(token, None);
    }

    #[test]
    fn civitai_auth_or_error_returns_bearer_when_set() {
        let _guard = CIVITAI_TOKEN_LOCK.lock().unwrap();
        let original = std::env::var("CIVITAI_TOKEN").ok();
        std::env::set_var("CIVITAI_TOKEN", "cv_secret_xyz");
        let auth = civitai_auth_or_error("cv:123");
        match &original {
            Some(v) => std::env::set_var("CIVITAI_TOKEN", v),
            None => std::env::remove_var("CIVITAI_TOKEN"),
        }
        match auth {
            Ok(RecipeAuth::Bearer(t)) => assert_eq!(t, "cv_secret_xyz"),
            other => panic!("expected Bearer, got {other:?}"),
        }
    }

    #[test]
    fn civitai_auth_or_error_returns_missing_token_error_when_unset() {
        let _guard = CIVITAI_TOKEN_LOCK.lock().unwrap();
        let original = std::env::var("CIVITAI_TOKEN").ok();
        std::env::remove_var("CIVITAI_TOKEN");
        let err = civitai_auth_or_error("cv:618692").unwrap_err();
        if let Some(v) = &original {
            std::env::set_var("CIVITAI_TOKEN", v);
        }
        match err {
            DownloadError::MissingCivitaiToken { id } => {
                assert_eq!(id, "cv:618692");
            }
            other => panic!("expected MissingCivitaiToken, got {other:?}"),
        }
    }

    #[test]
    fn missing_civitai_token_error_message_points_at_env_var() {
        let err = DownloadError::MissingCivitaiToken {
            id: "cv:618692".to_string(),
        };
        let msg = err.to_string();
        assert!(
            msg.contains("CIVITAI_TOKEN"),
            "msg should name the env var: {msg}"
        );
        assert!(
            msg.contains("mold pull cv:618692"),
            "msg should suggest the retry command verbatim: {msg}"
        );
        assert!(msg.contains("https://civitai.com"));
    }

    // ── Companion presence helpers (round 2) ────────────────────────────

    fn stage_complete_companion(models_dir: &std::path::Path, name: &str) {
        let manifest = crate::manifest::find_manifest(name)
            .unwrap_or_else(|| panic!("companion manifest {name} must exist"));
        for f in &manifest.files {
            let dest = models_dir.join(crate::manifest::storage_path(manifest, f));
            if let Some(parent) = dest.parent() {
                std::fs::create_dir_all(parent).unwrap();
            }
            std::fs::File::create(&dest)
                .unwrap()
                .set_len(f.size_bytes)
                .unwrap();
            if f.sha256.is_some() {
                std::fs::write(sha256_marker_path(&dest), "verified").unwrap();
            }
        }
    }

    #[test]
    fn companion_present_returns_false_when_files_missing() {
        let models_dir = recipe_tmp_dir("companion_missing");
        let manifest =
            crate::manifest::find_manifest("clip-l").expect("clip-l manifest must exist");
        assert!(!companion_present_on_disk(&models_dir, manifest));
        let _ = std::fs::remove_dir_all(&models_dir);
    }

    #[test]
    fn companion_present_returns_true_when_files_present() {
        let models_dir = recipe_tmp_dir("companion_present");
        stage_complete_companion(&models_dir, "clip-l");
        let manifest = crate::manifest::find_manifest("clip-l").unwrap();
        assert!(companion_present_on_disk(&models_dir, manifest));
        let _ = std::fs::remove_dir_all(&models_dir);
    }

    #[test]
    fn companion_present_returns_false_for_unverified_sha_file() {
        let models_dir = recipe_tmp_dir("companion_unverified_sha");
        let manifest = crate::manifest::find_manifest("sdxl-vae").unwrap();
        let file = &manifest.files[0];
        let dest = models_dir.join(crate::manifest::storage_path(manifest, file));
        std::fs::create_dir_all(dest.parent().unwrap()).unwrap();
        std::fs::File::create(&dest)
            .unwrap()
            .set_len(file.size_bytes)
            .unwrap();
        assert!(
            !companion_present_on_disk(&models_dir, manifest),
            "SHA-declared companion files need the verification marker before repair skips them"
        );
        std::fs::write(sha256_marker_path(&dest), "verified").unwrap();
        assert!(companion_present_on_disk(&models_dir, manifest));
        let _ = std::fs::remove_dir_all(&models_dir);
    }

    #[test]
    fn companion_present_returns_false_when_pulling_marker_present() {
        let models_dir = recipe_tmp_dir("companion_marker");
        stage_complete_companion(&models_dir, "clip-l");
        let marker = pulling_marker_path_in(&models_dir, "clip-l");
        if let Some(parent) = marker.parent() {
            std::fs::create_dir_all(parent).unwrap();
        }
        std::fs::write(&marker, "in-progress").unwrap();
        let manifest = crate::manifest::find_manifest("clip-l").unwrap();
        assert!(
            !companion_present_on_disk(&models_dir, manifest),
            "marker must override on-disk completeness"
        );
        let _ = std::fs::remove_dir_all(&models_dir);
    }

    #[test]
    fn missing_companions_skips_unknown_names() {
        let models_dir = recipe_tmp_dir("companion_unknown");
        // "clip-l" is real; "future-encoder-9000" doesn't exist.
        let json = r#"["clip-l","future-encoder-9000"]"#;
        let missing = missing_companions_from_json(Some(json), &models_dir);
        assert_eq!(missing.len(), 1);
        assert_eq!(missing[0].name, "clip-l");
        let _ = std::fs::remove_dir_all(&models_dir);
    }

    #[test]
    fn missing_companions_resolves_zimage_text_encoder() {
        let models_dir = recipe_tmp_dir("companion_zimage_te");
        let json = r#"["z-image-te"]"#;
        let missing = missing_companions_from_json(Some(json), &models_dir);
        assert_eq!(missing.len(), 1);
        assert_eq!(missing[0].name, "z-image-te");
        let _ = std::fs::remove_dir_all(&models_dir);
    }

    #[test]
    fn missing_companions_skips_present_returns_only_missing() {
        let models_dir = recipe_tmp_dir("companion_skip_present");
        stage_complete_companion(&models_dir, "clip-l");
        // clip-l is staged, sdxl-vae is not.
        let json = r#"["clip-l","sdxl-vae"]"#;
        let missing = missing_companions_from_json(Some(json), &models_dir);
        assert_eq!(missing.len(), 1);
        assert_eq!(missing[0].name, "sdxl-vae");
        let _ = std::fs::remove_dir_all(&models_dir);
    }

    #[test]
    fn missing_companions_preserves_input_order() {
        let models_dir = recipe_tmp_dir("companion_order");
        let json = r#"["sdxl-vae","clip-l","clip-g"]"#;
        let missing = missing_companions_from_json(Some(json), &models_dir);
        let names: Vec<&str> = missing.iter().map(|m| m.name.as_str()).collect();
        assert_eq!(names, vec!["sdxl-vae", "clip-l", "clip-g"]);
        let _ = std::fs::remove_dir_all(&models_dir);
    }

    #[test]
    fn missing_companions_returns_empty_for_none_or_invalid() {
        let models_dir = recipe_tmp_dir("companion_empty");
        assert!(missing_companions_from_json(None, &models_dir).is_empty());
        assert!(missing_companions_from_json(Some("not json"), &models_dir).is_empty());
        assert!(missing_companions_from_json(Some("[]"), &models_dir).is_empty());
        let _ = std::fs::remove_dir_all(&models_dir);
    }

    // ── Recipe fetcher (round 1) ────────────────────────────────────────

    fn recipe_tmp_dir(label: &str) -> std::path::PathBuf {
        let dir = std::env::temp_dir().join(format!(
            "mold_recipe_{label}_{}",
            uuid::Uuid::new_v4().simple()
        ));
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    #[tokio::test]
    async fn h3_recipe_fetch_rejects_id_url_and_dest_before_any_side_effect() {
        use std::sync::atomic::{AtomicUsize, Ordering};
        use wiremock::matchers::method;
        use wiremock::{Mock, MockServer, ResponseTemplate};

        let server = MockServer::start().await;
        Mock::given(method("GET"))
            .respond_with(ResponseTemplate::new(200).set_body_bytes(b"must not be fetched"))
            .expect(0)
            .mount(&server)
            .await;

        let ordinary_url = format!("{}/weights.safetensors", server.uri());
        let gated_url = format!("{}/MiniMax-H3/weights.safetensors", server.uri());
        let cases = [
            (
                "id",
                "hf:MiniMaxAI/MiniMax-H3".to_string(),
                ordinary_url.clone(),
                "weights.safetensors".to_string(),
            ),
            (
                "url",
                "cv:opaque".to_string(),
                gated_url,
                "weights.safetensors".to_string(),
            ),
            (
                "dest",
                "cv:opaque".to_string(),
                ordinary_url,
                "MiniMax-H3/weights.safetensors".to_string(),
            ),
        ];

        for (field, id, url, dest) in cases {
            let models_dir = std::env::temp_dir().join(format!(
                "mold_recipe_h3_{field}_{}",
                uuid::Uuid::new_v4().simple()
            ));
            assert!(!models_dir.exists(), "test path must begin absent");

            let files = [RecipeFetchFile {
                url: &url,
                dest: &dest,
                sha256: None,
                size_bytes: Some(1),
            }];
            let progress_count = Arc::new(AtomicUsize::new(0));
            let observed = progress_count.clone();
            let progress: DownloadProgressCallback = Arc::new(move |_| {
                observed.fetch_add(1, Ordering::SeqCst);
            });

            let error = fetch_recipe(
                &id,
                &files,
                RecipeAuth::None,
                &models_dir,
                Some(progress),
                &PullOptions::default(),
            )
            .await
            .expect_err("H3 recipe input must be compliance-gated");

            assert!(
                matches!(error, DownloadError::ModelActivation(_)),
                "{field}"
            );
            assert_eq!(progress_count.load(Ordering::SeqCst), 0, "{field}");
            assert!(
                !models_dir.exists(),
                "{field} rejection must not create the models directory"
            );
        }

        server.verify().await;
    }

    #[tokio::test]
    async fn recipe_fetcher_writes_files_under_models_dir() {
        use wiremock::matchers::{method, path};
        use wiremock::{Mock, MockServer, ResponseTemplate};

        let server = MockServer::start().await;
        Mock::given(method("GET"))
            .and(path("/file1.safetensors"))
            .respond_with(ResponseTemplate::new(200).set_body_bytes(b"hello".as_ref()))
            .mount(&server)
            .await;
        Mock::given(method("GET"))
            .and(path("/sub/file2.safetensors"))
            .respond_with(ResponseTemplate::new(200).set_body_bytes(b"world".as_ref()))
            .mount(&server)
            .await;

        let models_dir = recipe_tmp_dir("writes");
        let url1 = format!("{}/file1.safetensors", server.uri());
        let url2 = format!("{}/sub/file2.safetensors", server.uri());
        let files = vec![
            RecipeFetchFile {
                url: &url1,
                dest: "file1.safetensors",
                sha256: None,
                size_bytes: None,
            },
            RecipeFetchFile {
                url: &url2,
                dest: "sub/file2.safetensors",
                sha256: None,
                size_bytes: None,
            },
        ];

        let written = fetch_recipe(
            "cv:42",
            &files,
            RecipeAuth::None,
            &models_dir,
            None,
            &PullOptions::default(),
        )
        .await
        .expect("fetch_recipe ok");

        let f1 = models_dir.join("cv-42").join("file1.safetensors");
        let f2 = models_dir
            .join("cv-42")
            .join("sub")
            .join("file2.safetensors");
        assert_eq!(written, vec![f1.clone(), f2.clone()]);
        assert_eq!(std::fs::read(&f1).unwrap(), b"hello");
        assert_eq!(std::fs::read(&f2).unwrap(), b"world");

        let _ = std::fs::remove_dir_all(&models_dir);
    }

    #[tokio::test]
    async fn recipe_fetcher_verifies_sha256_when_present_match() {
        use wiremock::matchers::{method, path};
        use wiremock::{Mock, MockServer, ResponseTemplate};

        let server = MockServer::start().await;
        let body = b"hello world";
        // SHA-256 of "hello world"
        let expected = "b94d27b9934d3e08a52e52d7da7dabfac484efe37a5380ee9088f7ace2efcde9";
        Mock::given(method("GET"))
            .and(path("/m.safetensors"))
            .respond_with(ResponseTemplate::new(200).set_body_bytes(body.as_ref()))
            .mount(&server)
            .await;

        let models_dir = recipe_tmp_dir("sha_match");
        let url = format!("{}/m.safetensors", server.uri());
        let files = vec![RecipeFetchFile {
            url: &url,
            dest: "m.safetensors",
            sha256: Some(expected),
            size_bytes: None,
        }];
        fetch_recipe(
            "cv:1",
            &files,
            RecipeAuth::None,
            &models_dir,
            None,
            &PullOptions::default(),
        )
        .await
        .expect("matching SHA must succeed");

        let _ = std::fs::remove_dir_all(&models_dir);
    }

    #[tokio::test]
    async fn recipe_fetcher_verifies_sha256_when_present_mismatch() {
        use wiremock::matchers::{method, path};
        use wiremock::{Mock, MockServer, ResponseTemplate};

        let server = MockServer::start().await;
        Mock::given(method("GET"))
            .and(path("/bad.safetensors"))
            .respond_with(ResponseTemplate::new(200).set_body_bytes(b"hello".as_ref()))
            .mount(&server)
            .await;

        let models_dir = recipe_tmp_dir("sha_mismatch");
        let url = format!("{}/bad.safetensors", server.uri());
        // Wrong digest — file content is "hello".
        let files = vec![RecipeFetchFile {
            url: &url,
            dest: "bad.safetensors",
            sha256: Some("0000000000000000000000000000000000000000000000000000000000000000"),
            size_bytes: None,
        }];

        let err = fetch_recipe(
            "cv:2",
            &files,
            RecipeAuth::None,
            &models_dir,
            None,
            &PullOptions::default(),
        )
        .await
        .expect_err("mismatched SHA must error");

        match err {
            DownloadError::Sha256Mismatch { filename, .. } => {
                assert_eq!(filename, "bad.safetensors");
            }
            other => panic!("expected Sha256Mismatch, got {other:?}"),
        }
        // Corrupted file should be deleted.
        let bad = models_dir.join("cv-2").join("bad.safetensors");
        assert!(
            !bad.exists(),
            "corrupted file should be removed on mismatch"
        );

        let _ = std::fs::remove_dir_all(&models_dir);
    }

    #[tokio::test]
    async fn recipe_fetcher_marker_lifecycle() {
        use wiremock::matchers::{method, path};
        use wiremock::{Mock, MockServer, ResponseTemplate};

        let server = MockServer::start().await;
        Mock::given(method("GET"))
            .and(path("/x.safetensors"))
            .respond_with(ResponseTemplate::new(200).set_body_bytes(b"x".as_ref()))
            .mount(&server)
            .await;

        let models_dir = recipe_tmp_dir("marker");
        let url = format!("{}/x.safetensors", server.uri());
        let files = vec![RecipeFetchFile {
            url: &url,
            dest: "x.safetensors",
            sha256: None,
            size_bytes: None,
        }];
        let marker = pulling_marker_path_in(&models_dir, "cv:7");
        assert!(!marker.exists(), "marker should not exist before fetch");

        fetch_recipe(
            "cv:7",
            &files,
            RecipeAuth::None,
            &models_dir,
            None,
            &PullOptions::default(),
        )
        .await
        .expect("ok");

        assert!(
            !marker.exists(),
            "marker should be removed after successful fetch"
        );
        let _ = std::fs::remove_dir_all(&models_dir);
    }

    #[tokio::test]
    async fn recipe_fetcher_skips_files_with_matching_size() {
        use wiremock::matchers::{method, path};
        use wiremock::{Mock, MockServer, ResponseTemplate};

        let server = MockServer::start().await;
        let body = b"hello world";
        Mock::given(method("GET"))
            .and(path("/m.safetensors"))
            .respond_with(ResponseTemplate::new(200).set_body_bytes(body.as_ref()))
            // First call serves the body; any second call is an unexpected re-fetch.
            .expect(1)
            .mount(&server)
            .await;

        let models_dir = recipe_tmp_dir("idempotent_size");
        let url = format!("{}/m.safetensors", server.uri());
        let files = vec![RecipeFetchFile {
            url: &url,
            dest: "m.safetensors",
            sha256: None,
            size_bytes: Some(body.len() as u64),
        }];

        fetch_recipe(
            "cv:idemp",
            &files,
            RecipeAuth::None,
            &models_dir,
            None,
            &PullOptions::default(),
        )
        .await
        .expect("first fetch ok");

        // Second call must skip the HTTP fetch entirely because the file is on
        // disk with the declared size.
        fetch_recipe(
            "cv:idemp",
            &files,
            RecipeAuth::None,
            &models_dir,
            None,
            &PullOptions::default(),
        )
        .await
        .expect("second fetch ok (skip path)");

        // wiremock's `.expect(1)` is verified on `MockServer::drop`; explicit
        // verify here gives a clearer failure message at the assertion site.
        server.verify().await;

        let _ = std::fs::remove_dir_all(&models_dir);
    }

    #[tokio::test]
    async fn recipe_fetcher_skips_files_with_sha256_marker_when_size_unknown() {
        use wiremock::matchers::{method, path};
        use wiremock::{Mock, MockServer, ResponseTemplate};

        let server = MockServer::start().await;
        Mock::given(method("GET"))
            .and(path("/m.safetensors"))
            .respond_with(ResponseTemplate::new(200).set_body_bytes(b"x".as_ref()))
            .expect(1)
            .mount(&server)
            .await;

        let models_dir = recipe_tmp_dir("idempotent_marker");
        let url = format!("{}/m.safetensors", server.uri());
        let files = vec![RecipeFetchFile {
            url: &url,
            dest: "m.safetensors",
            sha256: None,
            // size_bytes intentionally None — fall through to marker check.
            size_bytes: None,
        }];

        // First call writes the marker via the existing post-download codepath.
        fetch_recipe(
            "cv:idemp_marker",
            &files,
            RecipeAuth::None,
            &models_dir,
            None,
            &PullOptions::default(),
        )
        .await
        .expect("first fetch ok");

        // Confirm marker is in place (sanity check for the test setup).
        let dest = models_dir.join("cv-idemp_marker").join("m.safetensors");
        assert!(
            sha256_marker_path(&dest).exists(),
            "first fetch should have written the .sha256-verified marker"
        );

        fetch_recipe(
            "cv:idemp_marker",
            &files,
            RecipeAuth::None,
            &models_dir,
            None,
            &PullOptions::default(),
        )
        .await
        .expect("second fetch ok");

        server.verify().await;
        let _ = std::fs::remove_dir_all(&models_dir);
    }

    #[tokio::test]
    async fn recipe_fetcher_refetches_when_sha256_declared_but_marker_missing() {
        use wiremock::matchers::{method, path};
        use wiremock::{Mock, MockServer, ResponseTemplate};

        let server = MockServer::start().await;
        let body = b"correct";
        // SHA-256 of "correct"
        let expected = "15a596e3c98c407e043751ff3b21ff0358a1bdfdf3fe948b1523893a8e5de2e8";
        Mock::given(method("GET"))
            .and(path("/m.safetensors"))
            .respond_with(ResponseTemplate::new(200).set_body_bytes(body.as_ref()))
            // The pre-staged file has the right size but no marker, so the
            // skip path must refuse it and re-fetch exactly once.
            .expect(1)
            .mount(&server)
            .await;

        let models_dir = recipe_tmp_dir("idempotent_no_marker");
        let subdir = models_dir.join("cv-idemp_no_marker");
        std::fs::create_dir_all(&subdir).unwrap();
        let dest = subdir.join("m.safetensors");
        // Pre-stage a file at the declared size — but NO marker, and bytes
        // don't actually match the declared sha256. A size-only skip would
        // accept this; the tightened predicate must not.
        std::fs::write(&dest, b"BADBYTE").unwrap();
        assert!(!sha256_marker_path(&dest).exists());

        let url = format!("{}/m.safetensors", server.uri());
        let files = vec![RecipeFetchFile {
            url: &url,
            dest: "m.safetensors",
            sha256: Some(expected),
            size_bytes: Some(body.len() as u64),
        }];

        fetch_recipe(
            "cv:idemp_no_marker",
            &files,
            RecipeAuth::None,
            &models_dir,
            None,
            &PullOptions::default(),
        )
        .await
        .expect("fetch ok");

        // After re-fetch the bytes match the server response and the marker exists.
        assert_eq!(std::fs::read(&dest).unwrap(), body);
        assert!(sha256_marker_path(&dest).exists());
        server.verify().await;
        let _ = std::fs::remove_dir_all(&models_dir);
    }

    #[test]
    fn catalog_entry_installed_returns_true_for_complete_recipe() {
        let models_dir = recipe_tmp_dir("installed_complete");
        let subdir = models_dir.join("cv-installed_a");
        std::fs::create_dir_all(&subdir).unwrap();
        let dest = subdir.join("m.safetensors");
        std::fs::write(&dest, b"hello").unwrap();

        let files = vec![RecipeFetchFile {
            url: "https://example.invalid/m.safetensors",
            dest: "m.safetensors",
            sha256: None,
            size_bytes: Some(5),
        }];

        assert!(catalog_entry_installed(
            &models_dir,
            "cv:installed_a",
            &files
        ));
        let _ = std::fs::remove_dir_all(&models_dir);
    }

    #[test]
    fn catalog_entry_installed_returns_false_when_any_file_missing() {
        let models_dir = recipe_tmp_dir("installed_partial");
        let subdir = models_dir.join("cv-installed_b");
        std::fs::create_dir_all(&subdir).unwrap();
        std::fs::write(subdir.join("a.safetensors"), b"present").unwrap();
        // b.safetensors is intentionally missing.

        let files = vec![
            RecipeFetchFile {
                url: "https://example.invalid/a.safetensors",
                dest: "a.safetensors",
                sha256: None,
                size_bytes: Some(7),
            },
            RecipeFetchFile {
                url: "https://example.invalid/b.safetensors",
                dest: "b.safetensors",
                sha256: None,
                size_bytes: Some(7),
            },
        ];

        assert!(!catalog_entry_installed(
            &models_dir,
            "cv:installed_b",
            &files
        ));
        let _ = std::fs::remove_dir_all(&models_dir);
    }

    #[test]
    fn catalog_entry_installed_returns_false_on_size_mismatch() {
        let models_dir = recipe_tmp_dir("installed_mismatch");
        let subdir = models_dir.join("cv-installed_c");
        std::fs::create_dir_all(&subdir).unwrap();
        std::fs::write(subdir.join("m.safetensors"), b"WRONG").unwrap();

        let files = vec![RecipeFetchFile {
            url: "https://example.invalid/m.safetensors",
            dest: "m.safetensors",
            sha256: None,
            size_bytes: Some(99),
        }];

        assert!(!catalog_entry_installed(
            &models_dir,
            "cv:installed_c",
            &files
        ));
        let _ = std::fs::remove_dir_all(&models_dir);
    }

    #[test]
    fn catalog_entry_installed_accepts_marker_when_declared_size_is_stale() {
        let models_dir = recipe_tmp_dir("installed_stale_size_marker");
        let subdir = models_dir.join("cv-installed_c2");
        std::fs::create_dir_all(&subdir).unwrap();
        let dest = subdir.join("m.safetensors");
        std::fs::write(&dest, b"new larger bytes").unwrap();
        write_sha256_marker(&dest, "deadbeef").unwrap();

        let files = vec![RecipeFetchFile {
            url: "https://example.invalid/m.safetensors",
            dest: "m.safetensors",
            sha256: None,
            size_bytes: Some(5),
        }];

        assert!(catalog_entry_installed(
            &models_dir,
            "cv:installed_c2",
            &files
        ));
        let _ = std::fs::remove_dir_all(&models_dir);
    }

    #[test]
    fn catalog_entry_installed_uses_marker_when_size_unknown() {
        let models_dir = recipe_tmp_dir("installed_marker");
        let subdir = models_dir.join("cv-installed_d");
        std::fs::create_dir_all(&subdir).unwrap();
        let dest = subdir.join("m.safetensors");
        std::fs::write(&dest, b"hello").unwrap();
        write_sha256_marker(&dest, "deadbeef").unwrap();

        let files = vec![RecipeFetchFile {
            url: "https://example.invalid/m.safetensors",
            dest: "m.safetensors",
            sha256: None,
            size_bytes: None,
        }];

        assert!(catalog_entry_installed(
            &models_dir,
            "cv:installed_d",
            &files
        ));
        let _ = std::fs::remove_dir_all(&models_dir);
    }

    #[test]
    fn catalog_entry_installed_returns_false_without_marker_and_without_size() {
        let models_dir = recipe_tmp_dir("installed_nomarker");
        let subdir = models_dir.join("cv-installed_e");
        std::fs::create_dir_all(&subdir).unwrap();
        std::fs::write(subdir.join("m.safetensors"), b"hello").unwrap();
        // No marker, no declared size — refuse to claim install.

        let files = vec![RecipeFetchFile {
            url: "https://example.invalid/m.safetensors",
            dest: "m.safetensors",
            sha256: None,
            size_bytes: None,
        }];

        assert!(!catalog_entry_installed(
            &models_dir,
            "cv:installed_e",
            &files
        ));
        let _ = std::fs::remove_dir_all(&models_dir);
    }

    #[test]
    fn catalog_entry_installed_returns_false_when_pulling_marker_present() {
        let models_dir = recipe_tmp_dir("installed_pulling");
        let subdir = models_dir.join("cv-installed_f");
        std::fs::create_dir_all(&subdir).unwrap();
        std::fs::write(subdir.join("m.safetensors"), b"hello").unwrap();

        let marker = pulling_marker_path_in(&models_dir, "cv:installed_f");
        if let Some(parent) = marker.parent() {
            std::fs::create_dir_all(parent).unwrap();
        }
        std::fs::write(&marker, "in-progress").unwrap();

        let files = vec![RecipeFetchFile {
            url: "https://example.invalid/m.safetensors",
            dest: "m.safetensors",
            sha256: None,
            size_bytes: Some(5),
        }];

        assert!(
            !catalog_entry_installed(&models_dir, "cv:installed_f", &files),
            "active .pulling marker must override on-disk completeness"
        );
        let _ = std::fs::remove_dir_all(&models_dir);
    }

    #[test]
    fn catalog_entry_installed_rejects_path_traversal() {
        let models_dir = recipe_tmp_dir("installed_traversal");

        let files = vec![RecipeFetchFile {
            url: "https://example.invalid/m.safetensors",
            dest: "../escape.safetensors",
            sha256: None,
            size_bytes: Some(5),
        }];

        assert!(
            !catalog_entry_installed(&models_dir, "cv:installed_g", &files),
            "path traversal must be treated as not-installed, not as a panic"
        );
        let _ = std::fs::remove_dir_all(&models_dir);
    }

    #[test]
    fn catalog_entry_installed_returns_false_for_empty_files() {
        let models_dir = recipe_tmp_dir("installed_empty");
        assert!(
            !catalog_entry_installed(&models_dir, "cv:installed_h", &[]),
            "empty file slice means no recipe to verify; must refuse to claim install"
        );
        let _ = std::fs::remove_dir_all(&models_dir);
    }

    #[test]
    fn catalog_entry_installed_returns_true_for_multi_file_complete_recipe() {
        let models_dir = recipe_tmp_dir("installed_multi");
        let subdir = models_dir.join("cv-installed_i");
        std::fs::create_dir_all(&subdir).unwrap();
        std::fs::write(subdir.join("a.safetensors"), b"present").unwrap();
        std::fs::write(subdir.join("b.safetensors"), b"present_too").unwrap();
        std::fs::write(subdir.join("c.safetensors"), b"third").unwrap();

        let files = vec![
            RecipeFetchFile {
                url: "https://example.invalid/a.safetensors",
                dest: "a.safetensors",
                sha256: None,
                size_bytes: Some(7),
            },
            RecipeFetchFile {
                url: "https://example.invalid/b.safetensors",
                dest: "b.safetensors",
                sha256: None,
                size_bytes: Some(11),
            },
            RecipeFetchFile {
                url: "https://example.invalid/c.safetensors",
                dest: "c.safetensors",
                sha256: None,
                size_bytes: Some(5),
            },
        ];

        assert!(
            catalog_entry_installed(&models_dir, "cv:installed_i", &files),
            "every file present at declared size — must report installed"
        );
        let _ = std::fs::remove_dir_all(&models_dir);
    }

    #[test]
    fn catalog_entry_installed_returns_false_when_file_larger_than_declared() {
        // Mutation guard: pins == (not >=) for the size comparison. A
        // 99-byte file declared as 5 bytes is just as wrong as a 5-byte file
        // declared as 99 bytes — the existing `_size_mismatch` test only
        // exercises the file-too-small direction.
        let models_dir = recipe_tmp_dir("installed_too_big");
        let subdir = models_dir.join("cv-installed_j");
        std::fs::create_dir_all(&subdir).unwrap();
        std::fs::write(
            subdir.join("m.safetensors"),
            b"this is much longer than five bytes",
        )
        .unwrap();

        let files = vec![RecipeFetchFile {
            url: "https://example.invalid/m.safetensors",
            dest: "m.safetensors",
            sha256: None,
            size_bytes: Some(5),
        }];

        assert!(!catalog_entry_installed(
            &models_dir,
            "cv:installed_j",
            &files
        ));
        let _ = std::fs::remove_dir_all(&models_dir);
    }

    #[test]
    fn catalog_entry_installed_requires_marker_when_sha256_declared() {
        // Cross-consistency: catalog_entry_installed and the inline skip
        // path inside fetch_recipe_inner must agree on what "placed" means.
        // A file at the right size with no marker — and a declared sha256
        // — would otherwise be reported `installed=true` by the catalog
        // API while the fetch path re-downloads it on Repair. Pins the
        // shared `recipe_file_is_placed` rule.
        let models_dir = recipe_tmp_dir("installed_no_marker");
        let subdir = models_dir.join("cv-installed_k");
        std::fs::create_dir_all(&subdir).unwrap();
        std::fs::write(subdir.join("m.safetensors"), b"hello").unwrap();
        // No marker.

        let files = vec![RecipeFetchFile {
            url: "https://example.invalid/m.safetensors",
            dest: "m.safetensors",
            sha256: Some("deadbeef00000000000000000000000000000000000000000000000000000000"),
            size_bytes: Some(5),
        }];

        assert!(
            !catalog_entry_installed(&models_dir, "cv:installed_k", &files),
            "size matches but no marker AND sha256 declared — must refuse to claim install",
        );
        let _ = std::fs::remove_dir_all(&models_dir);
    }

    #[test]
    fn catalog_entry_installed_trusts_marker_over_stale_size_bytes() {
        // Regression guard: catalog DB can have stale size_bytes (e.g. model
        // re-uploaded with same sha256 but different compressed size).  When a
        // sha256 is declared and the marker exists, the file is verified —
        // reject it only on size would cause installed models to disappear from
        // the settings modal.
        let models_dir = recipe_tmp_dir("installed_stale_size");
        let subdir = models_dir.join("cv-installed_stale");
        std::fs::create_dir_all(&subdir).unwrap();
        let dest = subdir.join("m.safetensors");
        // File is 5 bytes, but we'll declare size as 99 (stale) in the recipe.
        std::fs::write(&dest, b"hello").unwrap();
        write_sha256_marker(
            &dest,
            "deadbeef00000000000000000000000000000000000000000000000000000000",
        )
        .unwrap();

        let files = vec![RecipeFetchFile {
            url: "https://example.invalid/m.safetensors",
            dest: "m.safetensors",
            sha256: Some("deadbeef00000000000000000000000000000000000000000000000000000000"),
            size_bytes: Some(99), // stale — actual file is 5 bytes
        }];

        assert!(
            catalog_entry_installed(&models_dir, "cv:installed_stale", &files),
            "sha256 marker present → installed despite stale size_bytes",
        );
        let _ = std::fs::remove_dir_all(&models_dir);
    }

    #[tokio::test]
    async fn recipe_fetcher_pulls_when_size_mismatch() {
        use wiremock::matchers::{method, path};
        use wiremock::{Mock, MockServer, ResponseTemplate};

        let server = MockServer::start().await;
        Mock::given(method("GET"))
            .and(path("/m.safetensors"))
            .respond_with(ResponseTemplate::new(200).set_body_bytes(b"correct".as_ref()))
            // Size mismatch must trigger the fetch.
            .expect(1)
            .mount(&server)
            .await;

        let models_dir = recipe_tmp_dir("idempotent_mismatch");
        let subdir = models_dir.join("cv-idemp_mismatch");
        std::fs::create_dir_all(&subdir).unwrap();
        let dest = subdir.join("m.safetensors");
        // Pre-stage a wrong-size file (4 bytes vs. the recipe's declared 7).
        std::fs::write(&dest, b"WRNG").unwrap();

        let url = format!("{}/m.safetensors", server.uri());
        let files = vec![RecipeFetchFile {
            url: &url,
            dest: "m.safetensors",
            sha256: None,
            size_bytes: Some(7),
        }];

        fetch_recipe(
            "cv:idemp_mismatch",
            &files,
            RecipeAuth::None,
            &models_dir,
            None,
            &PullOptions::default(),
        )
        .await
        .expect("ok");

        // File should now match the server response.
        assert_eq!(std::fs::read(&dest).unwrap(), b"correct");
        server.verify().await;
        let _ = std::fs::remove_dir_all(&models_dir);
    }

    #[tokio::test]
    async fn recipe_fetcher_rejects_path_traversal_in_dest() {
        let models_dir = recipe_tmp_dir("traversal");
        let files = vec![RecipeFetchFile {
            url: "http://example.invalid/should-not-be-fetched",
            dest: "../etc/passwd",
            sha256: None,
            size_bytes: None,
        }];
        let err = fetch_recipe(
            "cv:8",
            &files,
            RecipeAuth::None,
            &models_dir,
            None,
            &PullOptions::default(),
        )
        .await
        .expect_err("traversal must be rejected");
        match err {
            DownloadError::RecipePathTraversal { dest } => {
                assert_eq!(dest, "../etc/passwd");
            }
            other => panic!("expected RecipePathTraversal, got {other:?}"),
        }
        // Sanity: nothing should have been created outside the per-id subdir.
        assert!(
            !models_dir.join("cv-8").exists()
                || std::fs::read_dir(models_dir.join("cv-8"))
                    .map(|d| d.count())
                    .unwrap_or(0)
                    == 0
        );
        let _ = std::fs::remove_dir_all(&models_dir);
    }

    #[tokio::test]
    async fn recipe_fetcher_rejects_absolute_dest() {
        let models_dir = recipe_tmp_dir("absolute");
        let files = vec![RecipeFetchFile {
            url: "http://example.invalid/should-not-be-fetched",
            dest: "/etc/passwd",
            sha256: None,
            size_bytes: None,
        }];
        let err = fetch_recipe(
            "cv:9",
            &files,
            RecipeAuth::None,
            &models_dir,
            None,
            &PullOptions::default(),
        )
        .await
        .expect_err("absolute dest must be rejected");
        assert!(matches!(err, DownloadError::RecipePathTraversal { .. }));
        let _ = std::fs::remove_dir_all(&models_dir);
    }

    #[tokio::test]
    async fn recipe_fetcher_sends_bearer_token_when_auth_set() {
        use wiremock::matchers::{header, method, path};
        use wiremock::{Mock, MockServer, ResponseTemplate};

        let server = MockServer::start().await;
        Mock::given(method("GET"))
            .and(path("/civitai.safetensors"))
            .and(header("authorization", "Bearer secret-cv-token"))
            .respond_with(ResponseTemplate::new(200).set_body_bytes(b"ok".as_ref()))
            .mount(&server)
            .await;

        let models_dir = recipe_tmp_dir("bearer");
        let url = format!("{}/civitai.safetensors", server.uri());
        let files = vec![RecipeFetchFile {
            url: &url,
            dest: "civitai.safetensors",
            sha256: None,
            size_bytes: None,
        }];
        fetch_recipe(
            "cv:618692",
            &files,
            RecipeAuth::Bearer("secret-cv-token".to_string()),
            &models_dir,
            None,
            &PullOptions::default(),
        )
        .await
        .expect("authenticated request must succeed");

        let _ = std::fs::remove_dir_all(&models_dir);
    }
}
