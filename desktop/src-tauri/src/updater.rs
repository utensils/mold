//! Signed desktop updates with stable/nightly channels and a strict preflight.
//!
//! Tauri downloads the complete archive and verifies its Minisign signature.
//! Mold then extracts the entire payload into a temporary directory, binds the
//! bundle identifier/version to the manifest, verifies the Apple signature and
//! Gatekeeper assessment, and proves that the install location is replaceable.
//! Only after every check passes does Mold atomically swap the app bundle.

use std::path::{Component, Path, PathBuf};
use std::process::Command;
use std::time::Duration;

use serde::Serialize;
use tauri::{AppHandle, Emitter};
use tauri_plugin_updater::{Update, UpdaterExt};
use time::format_description::well_known::Rfc3339;
use time::OffsetDateTime;

use crate::commands::SettingsStore;
use crate::settings::UpdateChannel;

const STABLE_ENDPOINT: &str =
    "https://github.com/utensils/mold/releases/latest/download/mold-desktop-stable.json";
const NIGHTLY_ENDPOINT: &str =
    "https://github.com/utensils/mold/releases/download/latest/mold-desktop-nightly.json";
const MANIFEST_TIMEOUT: Duration = Duration::from_secs(15);
const DOWNLOAD_TIMEOUT: Duration = Duration::from_secs(15 * 60);
const EXPECTED_BUNDLE_IDENTIFIER: &str = "com.utensils.mold";
const LEGACY_HEALTH_TOKEN: &str = "MOLD_UPDATE_HEALTH_TOKEN";

/// The release that introduces preflight-only updates can itself be launched
/// by the previous backup supervisor. Retire only that exact parent process
/// before its first poll so it cannot roll this candidate back for omitting the
/// now-deleted frontend handshake.
#[cfg(target_os = "macos")]
pub fn retire_legacy_supervisor_if_present() {
    if std::env::var_os(LEGACY_HEALTH_TOKEN).is_none() {
        return;
    }
    std::env::remove_var(LEGACY_HEALTH_TOKEN);

    let parent = unsafe { libc::getppid() };
    let mut buffer = vec![0_u8; libc::PROC_PIDPATHINFO_MAXSIZE as usize];
    let length = unsafe {
        libc::proc_pidpath(
            parent,
            buffer.as_mut_ptr().cast(),
            buffer.len().try_into().unwrap_or(u32::MAX),
        )
    };
    if length <= 0 {
        return;
    }
    let path = Path::new(std::ffi::OsStr::from_bytes(&buffer[..length as usize]));
    if is_legacy_supervisor_path(path) {
        unsafe {
            libc::kill(parent, libc::SIGTERM);
        }
    }
}

#[cfg(target_os = "macos")]
use std::os::unix::ffi::OsStrExt;

#[cfg(not(target_os = "macos"))]
pub fn retire_legacy_supervisor_if_present() {}

fn is_legacy_supervisor_path(path: &Path) -> bool {
    let display = path.to_string_lossy();
    display.contains("/updater/backup/Mold.app/Contents/MacOS/")
}

pub fn endpoint_for(channel: UpdateChannel) -> &'static str {
    match channel {
        UpdateChannel::Stable => STABLE_ENDPOINT,
        UpdateChannel::Nightly => NIGHTLY_ENDPOINT,
    }
}

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct UpdateCandidate {
    pub id: String,
    pub version: String,
    pub published_at: Option<String>,
    pub notes: Option<String>,
}

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct UpdateCheckResult {
    pub supported: bool,
    pub channel: UpdateChannel,
    pub current_version: String,
    pub checked_at: String,
    pub candidate: Option<UpdateCandidate>,
}

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct UpdateProgress {
    pub candidate_id: String,
    pub phase: &'static str,
    pub downloaded_bytes: Option<u64>,
    pub total_bytes: Option<u64>,
}

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct UpdateCommandError {
    pub code: String,
    pub message: String,
    pub disposition: &'static str,
    pub retryable: bool,
}

impl UpdateCommandError {
    fn unchanged(code: &str, message: impl Into<String>, retryable: bool) -> Self {
        Self {
            code: code.into(),
            message: message.into(),
            disposition: "unchanged",
            retryable,
        }
    }
}

#[derive(Clone)]
struct PendingUpdate {
    candidate: UpdateCandidate,
    channel: UpdateChannel,
    update: Update,
}

#[derive(Default)]
struct UpdateSession {
    busy: bool,
    pending: Option<PendingUpdate>,
}

#[derive(Default)]
pub struct UpdaterState {
    inner: tokio::sync::Mutex<UpdateSession>,
}

fn now_rfc3339() -> String {
    OffsetDateTime::now_utc()
        .format(&Rfc3339)
        .unwrap_or_else(|_| "unknown".into())
}

fn format_date(date: Option<OffsetDateTime>) -> Option<String> {
    date.and_then(|value| value.format(&Rfc3339).ok())
}

fn updater_supported() -> bool {
    cfg!(target_os = "macos") && !cfg!(debug_assertions)
}

#[tauri::command]
pub async fn check_for_updates(
    app: AppHandle,
    store: tauri::State<'_, SettingsStore>,
    state: tauri::State<'_, UpdaterState>,
    channel: UpdateChannel,
) -> Result<UpdateCheckResult, UpdateCommandError> {
    let current_version = app.package_info().version.to_string();
    let checked_at = now_rfc3339();

    if !updater_supported() {
        return Ok(UpdateCheckResult {
            supported: false,
            channel,
            current_version,
            checked_at,
            candidate: None,
        });
    }
    if store.current.lock().expect("settings mutex").update_channel != channel {
        return Err(UpdateCommandError::unchanged(
            "candidate-stale",
            "The update channel changed. Check again on the selected channel.",
            true,
        ));
    }

    {
        let mut session = state.inner.lock().await;
        if session.busy {
            return Err(UpdateCommandError::unchanged(
                "busy",
                "An update operation is already in progress.",
                true,
            ));
        }
        session.busy = true;
        session.pending = None;
    }

    let checked = async {
        let endpoint = endpoint_for(channel)
            .parse::<reqwest::Url>()
            .map_err(|error| {
                UpdateCommandError::unchanged(
                    "manifest",
                    format!("The update endpoint is invalid: {error}"),
                    false,
                )
            })?;
        let updater = app
            .updater_builder()
            .endpoints(vec![endpoint])
            .map_err(|error| {
                UpdateCommandError::unchanged(
                    "manifest",
                    format!("The update endpoint could not be configured: {error}"),
                    false,
                )
            })?
            .timeout(DOWNLOAD_TIMEOUT)
            .build()
            .map_err(|error| {
                UpdateCommandError::unchanged(
                    "manifest",
                    format!("The updater could not start: {error}"),
                    false,
                )
            })?;
        match tokio::time::timeout(MANIFEST_TIMEOUT, updater.check()).await {
            Ok(result) => result.map_err(|error| {
                UpdateCommandError::unchanged(
                    "offline",
                    format!("Mold could not check for updates: {error}"),
                    true,
                )
            }),
            Err(_) => Err(UpdateCommandError::unchanged(
                "timeout",
                "The update check timed out after 15 seconds.",
                true,
            )),
        }
    }
    .await;

    let mut session = state.inner.lock().await;
    session.busy = false;
    let update = checked?;
    if store.current.lock().expect("settings mutex").update_channel != channel {
        return Err(UpdateCommandError::unchanged(
            "candidate-stale",
            "The update channel changed while the check was running. Check again.",
            true,
        ));
    }

    let candidate = update.map(|update| {
        let candidate = UpdateCandidate {
            id: uuid::Uuid::new_v4().to_string(),
            version: update.version.clone(),
            published_at: format_date(update.date),
            notes: update.body.clone(),
        };
        session.pending = Some(PendingUpdate {
            candidate: candidate.clone(),
            channel,
            update,
        });
        candidate
    });

    Ok(UpdateCheckResult {
        supported: true,
        channel,
        current_version,
        checked_at,
        candidate,
    })
}

fn emit_progress(app: &AppHandle, progress: UpdateProgress) {
    let _ = app.emit("updater-progress", progress);
}

fn classify_download_error(error: &tauri_plugin_updater::Error) -> &'static str {
    let message = error.to_string().to_ascii_lowercase();
    if message.contains("signature") || message.contains("minisign") {
        "signature"
    } else {
        "offline"
    }
}

async fn restore_pending(state: &UpdaterState, pending: PendingUpdate) {
    let mut session = state.inner.lock().await;
    session.busy = false;
    session.pending = Some(pending);
}

async fn clear_update_session(state: &UpdaterState) {
    let mut session = state.inner.lock().await;
    session.busy = false;
    session.pending = None;
}

#[tauri::command]
pub async fn install_pending_update(
    app: AppHandle,
    store: tauri::State<'_, SettingsStore>,
    state: tauri::State<'_, UpdaterState>,
    candidate_id: String,
) -> Result<(), UpdateCommandError> {
    if !updater_supported() {
        return Err(UpdateCommandError::unchanged(
            "unsupported",
            "Automatic updates are available in signed desktop builds.",
            false,
        ));
    }

    let pending = {
        let mut session = state.inner.lock().await;
        if session.busy {
            return Err(UpdateCommandError::unchanged(
                "busy",
                "An update operation is already in progress.",
                true,
            ));
        }
        let Some(pending) = session.pending.take() else {
            return Err(UpdateCommandError::unchanged(
                "candidate-stale",
                "This update is no longer pending. Check again.",
                true,
            ));
        };
        if pending.candidate.id != candidate_id {
            session.pending = Some(pending);
            return Err(UpdateCommandError::unchanged(
                "candidate-stale",
                "This update check is stale. Check again before installing.",
                true,
            ));
        }
        if store.current.lock().expect("settings mutex").update_channel != pending.channel {
            session.pending = Some(pending);
            return Err(UpdateCommandError::unchanged(
                "candidate-stale",
                "The update channel changed. Check again before installing.",
                true,
            ));
        }
        session.busy = true;
        pending
    };
    let retry = pending.clone();

    let candidate_for_chunks = candidate_id.clone();
    let candidate_for_finish = candidate_id.clone();
    let app_for_chunks = app.clone();
    let app_for_finish = app.clone();
    let mut downloaded = 0_u64;
    let download = pending.update.download(
        move |chunk, total| {
            downloaded = downloaded.saturating_add(chunk as u64);
            emit_progress(
                &app_for_chunks,
                UpdateProgress {
                    candidate_id: candidate_for_chunks.clone(),
                    phase: "downloading",
                    downloaded_bytes: Some(downloaded),
                    total_bytes: total,
                },
            );
        },
        move || {
            emit_progress(
                &app_for_finish,
                UpdateProgress {
                    candidate_id: candidate_for_finish,
                    phase: "verifying",
                    downloaded_bytes: None,
                    total_bytes: None,
                },
            );
        },
    );
    let bytes = match tokio::time::timeout(DOWNLOAD_TIMEOUT, download).await {
        Ok(Ok(bytes)) => bytes,
        Ok(Err(error)) => {
            let code = classify_download_error(&error);
            restore_pending(&state, retry).await;
            return Err(UpdateCommandError::unchanged(
                code,
                format!("The update was not installed: {error}"),
                true,
            ));
        }
        Err(_) => {
            restore_pending(&state, retry).await;
            return Err(UpdateCommandError::unchanged(
                "timeout",
                "The update download timed out after 15 minutes. No app files were changed.",
                true,
            ));
        }
    };

    emit_progress(
        &app,
        UpdateProgress {
            candidate_id: candidate_id.clone(),
            phase: "staging",
            downloaded_bytes: None,
            total_bytes: None,
        },
    );
    let prepared = match preflight_update(
        &bytes,
        &pending.update.current_version,
        &pending.update.version,
    ) {
        Ok(prepared) => prepared,
        Err(error) => {
            restore_pending(&state, retry).await;
            return Err(UpdateCommandError::unchanged(
                "preflight",
                format!("The update failed verification before installation: {error:#}"),
                false,
            ));
        }
    };

    emit_progress(
        &app,
        UpdateProgress {
            candidate_id,
            phase: "installing",
            downloaded_bytes: None,
            total_bytes: None,
        },
    );
    if let Err(error) = prepared.install() {
        restore_pending(&state, retry).await;
        return Err(UpdateCommandError::unchanged(
            "install",
            format!("The verified update could not be installed: {error}"),
            true,
        ));
    }

    clear_update_session(&state).await;
    let app_to_restart = app.clone();
    std::thread::spawn(move || {
        std::thread::sleep(Duration::from_millis(350));
        app_to_restart.request_restart();
    });
    Ok(())
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct BundleIdentity {
    identifier: String,
    version: String,
}

fn identity_from_plist(value: plist::Value) -> anyhow::Result<BundleIdentity> {
    let dictionary = value
        .into_dictionary()
        .ok_or_else(|| anyhow::anyhow!("Info.plist is not a dictionary"))?;
    let identifier = dictionary
        .get("CFBundleIdentifier")
        .and_then(plist::Value::as_string)
        .ok_or_else(|| anyhow::anyhow!("Info.plist has no CFBundleIdentifier"))?;
    let version = dictionary
        .get("CFBundleShortVersionString")
        .and_then(plist::Value::as_string)
        .ok_or_else(|| anyhow::anyhow!("Info.plist has no CFBundleShortVersionString"))?;
    Ok(BundleIdentity {
        identifier: identifier.into(),
        version: version.into(),
    })
}

fn read_bundle_identity(app_path: &Path) -> anyhow::Result<BundleIdentity> {
    let path = app_path.join("Contents/Info.plist");
    identity_from_plist(
        plist::Value::from_file(&path)
            .map_err(|error| anyhow::anyhow!("could not read {}: {error}", path.display()))?,
    )
}

fn validate_identity(
    identity: &BundleIdentity,
    expected_identifier: &str,
    expected_version: &str,
) -> anyhow::Result<()> {
    if identity.identifier != expected_identifier {
        anyhow::bail!(
            "bundle identifier {} does not match {}",
            identity.identifier,
            expected_identifier
        );
    }
    if identity.version != expected_version {
        anyhow::bail!(
            "bundle version {} does not match manifest version {}",
            identity.version,
            expected_version
        );
    }
    Ok(())
}

fn app_bundle_from_executable(executable: &Path) -> Option<PathBuf> {
    executable
        .ancestors()
        .find(|path| path.extension().is_some_and(|extension| extension == "app"))
        .map(Path::to_path_buf)
}

fn reject_unsafe_install_location(app_path: &Path) -> anyhow::Result<()> {
    let path = app_path.to_string_lossy();
    if path.starts_with("/Volumes/") {
        anyhow::bail!("Move Mold from the disk image into Applications before updating.");
    }
    if path.contains("/AppTranslocation/") || path.starts_with("/private/var/folders/") {
        anyhow::bail!("Move Mold into Applications and reopen it before updating.");
    }
    Ok(())
}

fn validate_archive_path(path: &Path) -> anyhow::Result<()> {
    if path.is_absolute()
        || path
            .components()
            .any(|component| !matches!(component, Component::Normal(_)))
    {
        anyhow::bail!("update archive contains an unsafe path: {}", path.display());
    }
    Ok(())
}

fn extracted_bundle_from_archive(bytes: &[u8], destination: &Path) -> anyhow::Result<PathBuf> {
    let decoder = flate2::read::GzDecoder::new(bytes);
    let mut archive = tar::Archive::new(decoder);
    let mut bundle: Option<PathBuf> = None;
    let mut entries = 0_usize;

    for entry in archive.entries()? {
        let mut entry = entry?;
        let path = entry.path()?.into_owned();
        validate_archive_path(&path)?;
        entries += 1;
        if entries > 200_000 {
            anyhow::bail!("update archive contains too many entries");
        }
        if path.ends_with("Contents/Info.plist") {
            let app_path = path
                .parent()
                .and_then(Path::parent)
                .ok_or_else(|| anyhow::anyhow!("invalid app bundle path in update archive"))?;
            if app_path
                .extension()
                .is_some_and(|extension| extension == "app")
                && bundle.replace(app_path.to_path_buf()).is_some()
            {
                anyhow::bail!("update archive contains more than one app bundle");
            }
        }
        if !entry.unpack_in(destination)? {
            anyhow::bail!("update archive entry escaped the staging directory");
        }
    }
    let bundle = bundle.ok_or_else(|| anyhow::anyhow!("update archive contains no app bundle"))?;
    Ok(destination.join(bundle))
}

#[cfg(target_os = "macos")]
fn verify_apple_bundle(app_path: &Path) -> anyhow::Result<()> {
    for (tool, args, label) in [
        (
            "/usr/bin/codesign",
            vec!["--verify", "--deep", "--strict"],
            "Apple code signature",
        ),
        (
            "/usr/sbin/spctl",
            vec!["--assess", "--type", "execute"],
            "Gatekeeper assessment",
        ),
    ] {
        let output = Command::new(tool).args(args).arg(app_path).output()?;
        if !output.status.success() {
            let detail = String::from_utf8_lossy(&output.stderr);
            anyhow::bail!("{label} failed: {}", detail.trim());
        }
    }
    Ok(())
}

#[cfg(not(target_os = "macos"))]
fn verify_apple_bundle(_app_path: &Path) -> anyhow::Result<()> {
    Ok(())
}

fn prove_install_location_replaceable(app_path: &Path) -> anyhow::Result<()> {
    let parent = app_path
        .parent()
        .ok_or_else(|| anyhow::anyhow!("the app bundle has no parent directory"))?;
    let probe = parent.join(format!(".mold-update-preflight-{}", uuid::Uuid::new_v4()));
    std::fs::create_dir(&probe).map_err(|error| {
        anyhow::anyhow!(
            "Mold cannot replace {} without modifying permissions: {error}",
            app_path.display()
        )
    })?;
    std::fs::remove_dir(&probe)?;
    Ok(())
}

struct PreparedUpdate {
    app_path: PathBuf,
    candidate_path: PathBuf,
    staging: tempfile::TempDir,
}

impl PreparedUpdate {
    fn install(self) -> anyhow::Result<()> {
        atomic_swap_bundles(&self.app_path, &self.candidate_path)?;
        // `candidate_path` now names the old bundle. TempDir removes it; a
        // cleanup failure cannot undo or invalidate the completed atomic swap.
        if let Err(error) = self.staging.close() {
            tracing::warn!(%error, "could not remove the previous app bundle after update");
        }
        Ok(())
    }
}

#[cfg(target_os = "macos")]
fn atomic_swap_bundles(current: &Path, replacement: &Path) -> anyhow::Result<()> {
    use std::ffi::CString;
    use std::os::unix::ffi::OsStrExt;

    let current = CString::new(current.as_os_str().as_bytes())?;
    let replacement = CString::new(replacement.as_os_str().as_bytes())?;
    // Both paths are siblings on the same filesystem. RENAME_SWAP makes the
    // visible replacement indivisible: failure leaves both bundles untouched.
    let result = unsafe {
        libc::renameatx_np(
            libc::AT_FDCWD,
            current.as_ptr(),
            libc::AT_FDCWD,
            replacement.as_ptr(),
            libc::RENAME_SWAP,
        )
    };
    if result == 0 {
        Ok(())
    } else {
        Err(std::io::Error::last_os_error().into())
    }
}

#[cfg(not(target_os = "macos"))]
fn atomic_swap_bundles(_current: &Path, _replacement: &Path) -> anyhow::Result<()> {
    anyhow::bail!("atomic app replacement is available only on macOS")
}

fn preflight_update(
    bytes: &[u8],
    current_version: &str,
    target_version: &str,
) -> anyhow::Result<PreparedUpdate> {
    let executable = std::env::current_exe()?.canonicalize()?;
    let app_path = app_bundle_from_executable(&executable)
        .ok_or_else(|| anyhow::anyhow!("this executable is not inside a macOS .app bundle"))?;
    reject_unsafe_install_location(&app_path)?;
    validate_identity(
        &read_bundle_identity(&app_path)?,
        EXPECTED_BUNDLE_IDENTIFIER,
        current_version,
    )?;
    prove_install_location_replaceable(&app_path)?;

    let parent = app_path
        .parent()
        .ok_or_else(|| anyhow::anyhow!("the app bundle has no parent directory"))?;
    let staging = tempfile::Builder::new()
        .prefix(".mold-update-stage-")
        .tempdir_in(parent)?;
    let candidate = extracted_bundle_from_archive(bytes, staging.path())?;
    validate_identity(
        &read_bundle_identity(&candidate)?,
        EXPECTED_BUNDLE_IDENTIFIER,
        target_version,
    )?;
    verify_apple_bundle(&candidate)?;
    Ok(PreparedUpdate {
        app_path,
        candidate_path: candidate,
        staging,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use flate2::write::GzEncoder;
    use flate2::Compression;
    use std::io::Write;

    #[test]
    fn endpoints_are_fixed_https_allowlist() {
        for channel in [UpdateChannel::Stable, UpdateChannel::Nightly] {
            let endpoint = endpoint_for(channel);
            assert!(endpoint.starts_with("https://github.com/utensils/mold/releases/"));
            assert!(endpoint.ends_with(".json"));
        }
    }

    #[test]
    fn only_the_old_backup_supervisor_path_is_recognized() {
        assert!(is_legacy_supervisor_path(Path::new(
            "/Users/me/Library/Application Support/com.utensils.mold/updater/backup/Mold.app/Contents/MacOS/mold-desktop"
        )));
        assert!(!is_legacy_supervisor_path(Path::new(
            "/Applications/Mold.app/Contents/MacOS/mold-desktop"
        )));
        assert!(!is_legacy_supervisor_path(Path::new("/bin/zsh")));
    }

    #[test]
    fn archive_is_fully_extracted_and_bound_to_manifest_identity() {
        let archive = test_update_archive(EXPECTED_BUNDLE_IDENTIFIER, "0.18.0");
        let staging = tempfile::tempdir().unwrap();
        let bundle = extracted_bundle_from_archive(&archive, staging.path()).unwrap();
        let identity = read_bundle_identity(&bundle).unwrap();
        validate_identity(&identity, EXPECTED_BUNDLE_IDENTIFIER, "0.18.0").unwrap();
        assert!(bundle.join("Contents/MacOS/mold-desktop").exists());
    }

    #[test]
    fn archive_identity_must_match_manifest_target() {
        let archive = test_update_archive(EXPECTED_BUNDLE_IDENTIFIER, "0.17.9");
        let staging = tempfile::tempdir().unwrap();
        let bundle = extracted_bundle_from_archive(&archive, staging.path()).unwrap();
        let identity = read_bundle_identity(&bundle).unwrap();
        assert!(validate_identity(&identity, EXPECTED_BUNDLE_IDENTIFIER, "0.18.0").is_err());
    }

    #[test]
    fn unsafe_install_locations_are_rejected_before_install() {
        assert!(reject_unsafe_install_location(Path::new("/Volumes/Mold/Mold.app")).is_err());
        assert!(reject_unsafe_install_location(Path::new(
            "/private/var/folders/x/AppTranslocation/Mold.app"
        ))
        .is_err());
        assert!(reject_unsafe_install_location(Path::new("/Applications/Mold.app")).is_ok());
    }

    #[test]
    fn install_location_must_be_replaceable_during_preflight() {
        let root = tempfile::tempdir().unwrap();
        prove_install_location_replaceable(&root.path().join("Mold.app")).unwrap();
        assert!(prove_install_location_replaceable(Path::new("/missing/Mold.app")).is_err());
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn bundle_swap_is_atomic_and_exchanges_both_paths() {
        let root = tempfile::tempdir().unwrap();
        let current = root.path().join("Mold.app");
        let replacement = root.path().join("Mold.next.app");
        std::fs::create_dir(&current).unwrap();
        std::fs::create_dir(&replacement).unwrap();
        std::fs::write(current.join("version"), "old").unwrap();
        std::fs::write(replacement.join("version"), "new").unwrap();

        atomic_swap_bundles(&current, &replacement).unwrap();

        assert_eq!(
            std::fs::read_to_string(current.join("version")).unwrap(),
            "new"
        );
        assert_eq!(
            std::fs::read_to_string(replacement.join("version")).unwrap(),
            "old"
        );
    }

    fn test_update_archive(identifier: &str, version: &str) -> Vec<u8> {
        let mut plist = plist::Dictionary::new();
        plist.insert("CFBundleIdentifier".into(), identifier.into());
        plist.insert("CFBundleShortVersionString".into(), version.into());
        let mut plist_bytes = Vec::new();
        plist::Value::Dictionary(plist)
            .to_writer_xml(&mut plist_bytes)
            .unwrap();

        let encoder = GzEncoder::new(Vec::new(), Compression::default());
        let mut archive = tar::Builder::new(encoder);
        append_bytes(&mut archive, "Mold.app/Contents/Info.plist", &plist_bytes);
        append_bytes(
            &mut archive,
            "Mold.app/Contents/MacOS/mold-desktop",
            b"binary",
        );
        archive.into_inner().unwrap().finish().unwrap()
    }

    fn append_bytes<W: Write>(archive: &mut tar::Builder<W>, path: &str, bytes: &[u8]) {
        let mut header = tar::Header::new_gnu();
        header.set_size(bytes.len() as u64);
        header.set_mode(0o755);
        header.set_cksum();
        archive.append_data(&mut header, path, bytes).unwrap();
    }
}
