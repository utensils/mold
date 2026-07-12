//! Signed desktop updates with stable/nightly channels and a macOS rollback
//! watchdog. Tauri owns manifest parsing, download, and Minisign verification;
//! Mold keeps the trusted `Update` object in Rust and wraps installation in a
//! persistent backup transaction.

use std::io::{Read, Write};
use std::path::{Component, Path, PathBuf};
use std::process::{Child, Command, Stdio};
use std::time::{Duration, Instant};

use serde::{Deserialize, Serialize};
use tauri::{AppHandle, Emitter, Manager};
use tauri_plugin_updater::{Update, UpdaterExt};
use time::format_description::well_known::Rfc3339;
use time::OffsetDateTime;

use crate::commands::SettingsStore;
use crate::settings::UpdateChannel;

const STABLE_ENDPOINT: &str =
    "https://github.com/utensils/mold/releases/latest/download/mold-desktop-stable.json";
const NIGHTLY_ENDPOINT: &str =
    "https://github.com/utensils/mold/releases/download/latest/mold-desktop-nightly.json";
const SUPERVISOR_ARG: &str = "--mold-update-supervisor";
const HEALTH_TIMEOUT: Duration = Duration::from_secs(60);
const HEALTH_PROBATION: Duration = Duration::from_secs(10);
const MANIFEST_TIMEOUT: Duration = Duration::from_secs(15);
const DOWNLOAD_TIMEOUT: Duration = Duration::from_secs(15 * 60);
const PARENT_POLL: Duration = Duration::from_millis(200);
const EXPECTED_BUNDLE_IDENTIFIER: &str = "com.utensils.mold";

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
pub struct UpdateRecovery {
    pub restored_version: String,
    pub failed_version: String,
    pub message: String,
    pub rollback_failed: bool,
    pub backup_path: Option<String>,
}

#[derive(Debug, Clone, Copy, Serialize)]
#[serde(rename_all = "kebab-case")]
pub enum UpdateDisposition {
    Unchanged,
    RolledBack,
    RollbackFailed,
}

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct UpdateCommandError {
    pub code: String,
    pub message: String,
    pub disposition: UpdateDisposition,
    pub retryable: bool,
}

impl UpdateCommandError {
    fn unchanged(code: &str, message: impl Into<String>, retryable: bool) -> Self {
        Self {
            code: code.into(),
            message: message.into(),
            disposition: UpdateDisposition::Unchanged,
            retryable,
        }
    }

    fn rolled_back(message: impl Into<String>, retryable: bool) -> Self {
        Self {
            code: "install".into(),
            message: message.into(),
            disposition: UpdateDisposition::RolledBack,
            retryable,
        }
    }

    fn rollback_failed(message: impl Into<String>) -> Self {
        Self {
            code: "rollback".into(),
            message: message.into(),
            disposition: UpdateDisposition::RollbackFailed,
            retryable: false,
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

    let root = updater_root(&app).map_err(|error| {
        UpdateCommandError::unchanged(
            "rollback",
            format!("Mold could not inspect updater recovery state: {error:#}"),
            false,
        )
    })?;
    if let Some(transaction) = read_rollback_failed_transaction(&root).map_err(|error| {
        UpdateCommandError::unchanged(
            "rollback",
            format!("Mold could not inspect updater recovery state: {error:#}"),
            false,
        )
    })? {
        return Err(UpdateCommandError::rollback_failed(format!(
            "Resolve the previous failed rollback before checking again. Quit Mold, then copy the recovery app at {} to Applications.",
            transaction.backup_path.display()
        )));
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
        let builder = app
            .updater_builder()
            .endpoints(vec![endpoint])
            .map_err(|error| {
                UpdateCommandError::unchanged(
                    "manifest",
                    format!("The update endpoint could not be configured: {error}"),
                    false,
                )
            })?
            // This timeout is retained by the returned Update and bounds the
            // payload stream. The manifest request has a shorter outer bound
            // below so a dead endpoint never stalls startup for 15 minutes.
            .timeout(DOWNLOAD_TIMEOUT);
        let updater = builder.build().map_err(|error| {
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
        Ok(result) => result,
        Err(_) => {
            restore_pending(&state, retry).await;
            return Err(UpdateCommandError::unchanged(
                "timeout",
                "The update download timed out after 15 minutes. No app files were changed.",
                true,
            ));
        }
    };

    let bytes = match bytes {
        Ok(bytes) => bytes,
        Err(error) => {
            let code = classify_download_error(&error);
            restore_pending(&state, retry).await;
            return Err(UpdateCommandError::unchanged(
                code,
                format!("The update was not installed: {error}"),
                true,
            ));
        }
    };

    if let Err(error) =
        validate_update_archive(&bytes, EXPECTED_BUNDLE_IDENTIFIER, &pending.update.version)
    {
        clear_update_session(&state).await;
        return Err(UpdateCommandError::unchanged(
            "archive-identity",
            format!("The verified update did not match its manifest: {error:#}"),
            false,
        ));
    }

    emit_progress(
        &app,
        UpdateProgress {
            candidate_id: candidate_id.clone(),
            phase: "staging",
            downloaded_bytes: None,
            total_bytes: None,
        },
    );

    let mut prepared = match prepare_transaction(
        &app,
        &pending.update.current_version,
        &pending.update.version,
    ) {
        Ok(transaction) => transaction,
        Err(error) => {
            restore_pending(&state, retry).await;
            return Err(UpdateCommandError::unchanged(
                "permission",
                format!("Mold could not prepare a rollback copy: {error:#}"),
                true,
            ));
        }
    };

    let transaction = &mut prepared.transaction;

    emit_progress(
        &app,
        UpdateProgress {
            candidate_id: candidate_id.clone(),
            phase: "installing",
            downloaded_bytes: None,
            total_bytes: None,
        },
    );

    let install_failure = match pending.update.install(&bytes) {
        Err(error) => Some(error.to_string()),
        Ok(()) => validate_bundle_identity(
            &transaction.app_path,
            EXPECTED_BUNDLE_IDENTIFIER,
            &transaction.target_version,
        )
        .err()
        .map(|error| format!("the installed app did not match the manifest: {error:#}")),
    };
    if let Some(error) = install_failure {
        emit_progress(
            &app,
            UpdateProgress {
                candidate_id,
                phase: "rolling-back",
                downloaded_bytes: None,
                total_bytes: None,
            },
        );
        let restored = restore_bundle(&transaction.backup_path, &transaction.app_path);
        transaction.status = if restored.is_ok() {
            TransactionStatus::RolledBack
        } else {
            TransactionStatus::RollbackFailed
        };
        transaction.message = Some(match &restored {
            Ok(()) => format!(
                "Mold {} was restored after the update to {} failed.",
                transaction.current_version, transaction.target_version
            ),
            Err(restore_error) => format!(
                "The update failed ({error}) and automatic rollback also failed ({restore_error:#}). The backup remains at {}.",
                transaction.backup_path.display()
            ),
        });
        if write_transaction(transaction).is_err() && restored.is_ok() {
            let _ = prepared.supervisor.kill();
            let _ = prepared.supervisor.wait();
            cleanup_transaction(transaction);
        }
        return match restored {
            Ok(()) => {
                restore_pending(&state, retry).await;
                Err(UpdateCommandError::rolled_back(
                    format!(
                        "The update failed and Mold {} was restored.",
                        transaction.current_version
                    ),
                    true,
                ))
            }
            Err(restore_error) => {
                clear_update_session(&state).await;
                Err(UpdateCommandError::rollback_failed(format!(
                    "The update failed and automatic rollback also failed: {restore_error:#}. Quit Mold, then copy the recovery app at {} to Applications.",
                    transaction.backup_path.display()
                )))
            }
        };
    }

    transaction.status = TransactionStatus::Installed;
    if let Err(marker_error) = write_transaction(transaction) {
        let restored = restore_bundle(&transaction.backup_path, &transaction.app_path);
        return match restored {
            Ok(()) => {
                let _ = prepared.supervisor.kill();
                let _ = prepared.supervisor.wait();
                restore_pending(&state, retry).await;
                cleanup_transaction(transaction);
                Err(UpdateCommandError::rolled_back(
                    format!(
                        "The update restart could not be committed ({marker_error:#}); Mold {} was restored.",
                        transaction.current_version
                    ),
                    true,
                ))
            }
            Err(restore_error) => {
                transaction.status = TransactionStatus::RollbackFailed;
                transaction.message = Some(format!(
                    "The update installed, but its restart marker and automatic rollback failed: {marker_error:#}; {restore_error:#}. The backup remains at {}.",
                    transaction.backup_path.display()
                ));
                let _ = write_transaction(transaction);
                clear_update_session(&state).await;
                Err(UpdateCommandError::rollback_failed(format!(
                    "The update restart marker and automatic rollback both failed. Quit Mold, then copy the recovery app at {} to Applications. Details: {marker_error:#}; {restore_error:#}",
                    transaction.backup_path.display()
                )))
            }
        };
    }
    clear_update_session(&state).await;

    // Return to the webview so it can paint "Restarting", then let the
    // already-running backup supervisor launch and health-check the new app.
    let app_to_exit = app.clone();
    std::thread::spawn(move || {
        std::thread::sleep(Duration::from_millis(350));
        app_to_exit.exit(0);
    });
    Ok(())
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
enum TransactionStatus {
    Prepared,
    Installing,
    Installed,
    RolledBack,
    RollbackFailed,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
struct UpdateTransaction {
    current_version: String,
    target_version: String,
    app_path: PathBuf,
    backup_path: PathBuf,
    executable_relative: PathBuf,
    transaction_path: PathBuf,
    health_path: PathBuf,
    token: String,
    parent_pid: u32,
    #[serde(default)]
    candidate_pid: Option<u32>,
    #[serde(default)]
    external_supervisor_ready: bool,
    status: TransactionStatus,
    message: Option<String>,
}

struct PreparedTransaction {
    transaction: UpdateTransaction,
    supervisor: Child,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum InterruptedInstall {
    PreviousVersionIntact,
    CandidateRunning,
    UnknownBundle,
}

fn classify_interrupted_install(
    running_version: Option<&str>,
    current_version: &str,
    target_version: &str,
) -> InterruptedInstall {
    match running_version {
        Some(version) if version == current_version => InterruptedInstall::PreviousVersionIntact,
        Some(version) if version == target_version => InterruptedInstall::CandidateRunning,
        _ => InterruptedInstall::UnknownBundle,
    }
}

#[cfg(test)]
impl UpdateTransaction {
    fn test_fixture(root: &Path) -> Self {
        Self {
            current_version: "0.16.0".into(),
            target_version: "0.16.1".into(),
            app_path: root.join("Mold.app"),
            backup_path: root.join("Mold.backup.app"),
            executable_relative: PathBuf::from("Contents/MacOS/mold-desktop"),
            transaction_path: root.join("transaction.json"),
            health_path: root.join("healthy"),
            token: "expected-token".into(),
            parent_pid: std::process::id(),
            candidate_pid: None,
            external_supervisor_ready: false,
            status: TransactionStatus::Prepared,
            message: None,
        }
    }
}

fn updater_root(app: &AppHandle) -> anyhow::Result<PathBuf> {
    Ok(app.path().app_data_dir()?.join("updater"))
}

fn read_rollback_failed_transaction(root: &Path) -> anyhow::Result<Option<UpdateTransaction>> {
    let path = root.join("transaction.json");
    if !path.exists() {
        return Ok(None);
    }
    let transaction = read_transaction(&path)?;
    Ok((transaction.status == TransactionStatus::RollbackFailed).then_some(transaction))
}

fn app_bundle_from_executable(executable: &Path) -> Option<PathBuf> {
    executable
        .ancestors()
        .find(|path| path.extension().is_some_and(|extension| extension == "app"))
        .map(Path::to_path_buf)
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

fn validate_bundle_identity(
    app_path: &Path,
    expected_identifier: &str,
    expected_version: &str,
) -> anyhow::Result<()> {
    let identity = read_bundle_identity(app_path)?;
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

fn update_info_plist_path(path: &Path) -> bool {
    let components: Vec<_> = path.components().collect();
    matches!(
        components.as_slice(),
        [Component::Normal(app), Component::Normal(contents), Component::Normal(info)]
            if Path::new(app).extension().is_some_and(|extension| extension == "app")
                && *contents == "Contents"
                && *info == "Info.plist"
    )
}

fn validate_update_archive(
    bytes: &[u8],
    expected_identifier: &str,
    expected_version: &str,
) -> anyhow::Result<()> {
    let decoder = flate2::read::GzDecoder::new(bytes);
    let mut archive = tar::Archive::new(decoder);
    let mut identity = None;
    for entry in archive.entries()? {
        let mut entry = entry?;
        let path = entry.path()?.into_owned();
        if !update_info_plist_path(&path) {
            continue;
        }
        if identity.is_some() {
            anyhow::bail!("update archive contains more than one app Info.plist");
        }
        if entry.size() > 1024 * 1024 {
            anyhow::bail!("update Info.plist is unexpectedly large");
        }
        let mut bytes = Vec::with_capacity(entry.size() as usize);
        entry.read_to_end(&mut bytes)?;
        identity = Some(identity_from_plist(plist::Value::from_reader(
            std::io::Cursor::new(bytes),
        )?)?);
    }
    let identity = identity
        .ok_or_else(|| anyhow::anyhow!("update archive has no <App>.app/Contents/Info.plist"))?;
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

fn prepare_transaction(
    app: &AppHandle,
    current_version: &str,
    target_version: &str,
) -> anyhow::Result<PreparedTransaction> {
    let executable = std::env::current_exe()?.canonicalize()?;
    let app_path = app_bundle_from_executable(&executable)
        .ok_or_else(|| anyhow::anyhow!("this executable is not inside a macOS .app bundle"))?;
    reject_unsafe_install_location(&app_path)?;
    validate_bundle_identity(&app_path, EXPECTED_BUNDLE_IDENTIFIER, current_version)?;
    let executable_relative = executable.strip_prefix(&app_path)?.to_path_buf();

    let root = updater_root(app)?;
    let transaction_path = root.join("transaction.json");
    if transaction_path.exists() {
        let existing = read_transaction(&transaction_path)?;
        if matches!(
            existing.status,
            TransactionStatus::Installed | TransactionStatus::Installing
        ) {
            anyhow::bail!("a previous update is still awaiting a healthy launch");
        }
        if existing.status == TransactionStatus::RollbackFailed {
            anyhow::bail!(
                "a failed rollback still has its recovery copy at {}",
                existing.backup_path.display()
            );
        }
        let _ = std::fs::remove_dir_all(&root);
    }
    std::fs::create_dir_all(&root)?;

    let backup_path = root.join("backup/Mold.app");
    if let Some(parent) = backup_path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    copy_bundle(&app_path, &backup_path)?;

    let transaction = UpdateTransaction {
        current_version: current_version.into(),
        target_version: target_version.into(),
        app_path,
        backup_path: backup_path.clone(),
        executable_relative,
        transaction_path,
        health_path: root.join("healthy"),
        token: uuid::Uuid::new_v4().to_string(),
        parent_pid: std::process::id(),
        candidate_pid: None,
        external_supervisor_ready: false,
        status: TransactionStatus::Installing,
        message: None,
    };
    write_transaction(&transaction)?;

    let supervisor = match spawn_supervisor(&transaction) {
        Ok(child) => child,
        Err(error) => {
            let _ = std::fs::remove_dir_all(&root);
            return Err(error);
        }
    };

    Ok(PreparedTransaction {
        transaction,
        supervisor,
    })
}

fn spawn_supervisor(transaction: &UpdateTransaction) -> anyhow::Result<Child> {
    let supervisor = transaction
        .backup_path
        .join(&transaction.executable_relative);
    Command::new(&supervisor)
        .arg(SUPERVISOR_ARG)
        .arg(&transaction.transaction_path)
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .spawn()
        .map_err(|error| {
            anyhow::anyhow!(
                "could not start the rollback watchdog at {}: {error}",
                supervisor.display()
            )
        })
}

fn write_transaction(transaction: &UpdateTransaction) -> anyhow::Result<()> {
    if let Some(parent) = transaction.transaction_path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    let temporary = transaction.transaction_path.with_extension("json.tmp");
    let bytes = serde_json::to_vec_pretty(transaction)?;
    let mut file = std::fs::OpenOptions::new()
        .create(true)
        .truncate(true)
        .write(true)
        .open(&temporary)?;
    file.write_all(&bytes)?;
    file.sync_all()?;
    drop(file);
    std::fs::rename(&temporary, &transaction.transaction_path)?;
    if let Some(parent) = transaction.transaction_path.parent() {
        sync_directory(parent)?;
    }
    Ok(())
}

fn sync_directory(path: &Path) -> std::io::Result<()> {
    #[cfg(unix)]
    {
        std::fs::File::open(path)?.sync_all()
    }
    #[cfg(not(unix))]
    {
        let _ = path;
        Ok(())
    }
}

fn read_transaction(path: &Path) -> anyhow::Result<UpdateTransaction> {
    Ok(serde_json::from_slice(&std::fs::read(path)?)?)
}

fn copy_bundle(source: &Path, destination: &Path) -> anyhow::Result<()> {
    if destination.exists() {
        std::fs::remove_dir_all(destination)?;
    }
    if let Some(parent) = destination.parent() {
        std::fs::create_dir_all(parent)?;
    }
    #[cfg(target_os = "macos")]
    {
        let output = Command::new("/usr/bin/ditto")
            .arg(source)
            .arg(destination)
            .output()?;
        if !output.status.success() {
            anyhow::bail!(
                "ditto failed: {}",
                String::from_utf8_lossy(&output.stderr).trim()
            );
        }
        Ok(())
    }
    #[cfg(not(target_os = "macos"))]
    {
        copy_directory_for_tests(source, destination)
    }
}

#[cfg(not(target_os = "macos"))]
fn copy_directory_for_tests(source: &Path, destination: &Path) -> anyhow::Result<()> {
    std::fs::create_dir_all(destination)?;
    for entry in std::fs::read_dir(source)? {
        let entry = entry?;
        let target = destination.join(entry.file_name());
        if entry.file_type()?.is_dir() {
            copy_directory_for_tests(&entry.path(), &target)?;
        } else {
            std::fs::copy(entry.path(), target)?;
        }
    }
    Ok(())
}

/// Restore with the same current→failed, backup→current shape as the CLI's
/// proven binary updater. If the second rename fails, the current bundle is
/// put back before the error escapes.
fn restore_bundle_unprivileged(backup: &Path, current: &Path) -> std::io::Result<()> {
    let failed = current.with_extension(format!("failed-{}", std::process::id()));
    if failed.exists() {
        std::fs::remove_dir_all(&failed)?;
    }
    let had_current = current.exists();
    if had_current {
        std::fs::rename(current, &failed)?;
    }

    if let Err(error) = std::fs::rename(backup, current) {
        if had_current {
            let _ = std::fs::rename(&failed, current);
        }
        return Err(error);
    }

    if had_current {
        let _ = std::fs::remove_dir_all(failed);
    }
    Ok(())
}

fn restore_bundle(backup: &Path, current: &Path) -> anyhow::Result<()> {
    match restore_bundle_unprivileged(backup, current) {
        Ok(()) => Ok(()),
        #[cfg(target_os = "macos")]
        Err(error) if error.kind() == std::io::ErrorKind::PermissionDenied => {
            restore_bundle_with_authorization(backup, current)
        }
        Err(error) => Err(error.into()),
    }
}

#[cfg(target_os = "macos")]
fn restore_bundle_with_authorization(backup: &Path, current: &Path) -> anyhow::Result<()> {
    let candidate = current.with_extension(format!("restore-{}", std::process::id()));
    let failed = current.with_extension(format!("failed-{}", std::process::id()));
    let script = r#"
on run argv
  set backupApp to item 1 of argv
  set currentApp to item 2 of argv
  set candidateApp to item 3 of argv
  set failedApp to item 4 of argv
  set commandText to "/bin/rm -rf " & quoted form of candidateApp & " " & quoted form of failedApp & " && /usr/bin/ditto " & quoted form of backupApp & " " & quoted form of candidateApp & " && if test -e " & quoted form of currentApp & "; then /bin/mv " & quoted form of currentApp & " " & quoted form of failedApp & "; fi && if /bin/mv " & quoted form of candidateApp & " " & quoted form of currentApp & "; then /bin/rm -rf " & quoted form of failedApp & "; else if test -e " & quoted form of failedApp & "; then /bin/mv " & quoted form of failedApp & " " & quoted form of currentApp & "; fi; exit 1; fi"
  do shell script commandText with administrator privileges
end run
"#;
    let output = Command::new("/usr/bin/osascript")
        .arg("-e")
        .arg(script)
        .arg(backup)
        .arg(current)
        .arg(&candidate)
        .arg(&failed)
        .output()?;
    if !output.status.success() {
        anyhow::bail!(
            "authorized rollback failed: {}",
            String::from_utf8_lossy(&output.stderr).trim()
        );
    }
    Ok(())
}

fn confirm_health(transaction: &UpdateTransaction, token: &str) -> anyhow::Result<bool> {
    if token != transaction.token {
        return Ok(false);
    }
    let temporary = transaction.health_path.with_extension("tmp");
    std::fs::write(&temporary, token)?;
    std::fs::rename(temporary, &transaction.health_path)?;
    Ok(true)
}

#[tauri::command]
pub fn confirm_update_healthy(app: AppHandle) -> Result<(), UpdateCommandError> {
    let root = updater_root(&app)
        .map_err(|error| UpdateCommandError::unchanged("install", format!("{error:#}"), false))?;
    let path = root.join("transaction.json");
    if !path.exists() {
        return Ok(());
    }
    let transaction = read_transaction(&path).map_err(|error| {
        UpdateCommandError::unchanged(
            "install",
            format!("The update health record is unreadable: {error:#}"),
            false,
        )
    })?;
    if transaction.status != TransactionStatus::Installed {
        return Ok(());
    }

    if app.package_info().version.to_string() != transaction.target_version {
        return Err(UpdateCommandError::unchanged(
            "install",
            "The running app version does not match the pending update transaction.",
            false,
        ));
    }
    validate_bundle_identity(
        &transaction.app_path,
        EXPECTED_BUNDLE_IDENTIFIER,
        &transaction.target_version,
    )
    .map_err(|error| {
        UpdateCommandError::unchanged(
            "archive-identity",
            format!("The launched app does not match the verified update: {error:#}"),
            false,
        )
    })?;

    let inherited_token = std::env::var("MOLD_UPDATE_HEALTH_TOKEN").ok();
    let token = match inherited_token {
        Some(token) => token,
        None if tokenless_health_allowed(&transaction, std::process::id()) => {
            // Covers a machine restart between install and relaunch: reaching
            // this frontend command starts probation, but only after the
            // recovered watchdog has durably claimed this exact process.
            transaction.token.clone()
        }
        None => {
            return Err(UpdateCommandError::rollback_failed(format!(
                "Mold cannot confirm this update because no rollback watchdog owns the running process. Quit Mold, then copy the recovery app at {} to Applications.",
                transaction.backup_path.display()
            )))
        }
    };
    let confirmed = confirm_health(&transaction, &token).map_err(|error| {
        UpdateCommandError::unchanged(
            "install",
            format!("Mold could not confirm the updated app is healthy: {error:#}"),
            false,
        )
    })?;
    if !confirmed {
        return Err(UpdateCommandError::unchanged(
            "install",
            "The update health token did not match the pending transaction.",
            false,
        ));
    }
    Ok(())
}

fn tokenless_health_allowed(transaction: &UpdateTransaction, current_pid: u32) -> bool {
    transaction.external_supervisor_ready && transaction.candidate_pid == Some(current_pid)
}

fn running_bundle_version(app: &AppHandle, transaction: &UpdateTransaction) -> Option<String> {
    let identity = read_bundle_identity(&transaction.app_path).ok()?;
    if identity.identifier != EXPECTED_BUNDLE_IDENTIFIER
        || app.package_info().version.to_string() != identity.version
    {
        return None;
    }
    Some(identity.version)
}

fn mark_interrupted_rollback(
    transaction: &mut UpdateTransaction,
    reason: &str,
) -> anyhow::Result<()> {
    transaction.status = TransactionStatus::RolledBack;
    transaction.candidate_pid = None;
    transaction.message = Some(format!(
        "The update to {} was interrupted ({reason}). Mold {} remains installed.",
        transaction.target_version, transaction.current_version
    ));
    write_transaction(transaction)
}

fn restore_interrupted_transaction(
    transaction: &mut UpdateTransaction,
    reason: &str,
) -> anyhow::Result<()> {
    if let Err(identity_error) = validate_bundle_identity(
        &transaction.backup_path,
        EXPECTED_BUNDLE_IDENTIFIER,
        &transaction.current_version,
    ) {
        transaction.status = TransactionStatus::RollbackFailed;
        transaction.candidate_pid = None;
        transaction.message = Some(format!(
            "The interrupted update could not be restored because the recovery app is invalid: {identity_error:#}. The recovery path is {}.",
            transaction.backup_path.display()
        ));
        write_transaction(transaction)?;
        return Ok(());
    }
    match restore_bundle(&transaction.backup_path, &transaction.app_path) {
        Ok(()) => mark_interrupted_rollback(transaction, reason),
        Err(error) => {
            transaction.status = TransactionStatus::RollbackFailed;
            transaction.candidate_pid = None;
            transaction.message = Some(format!(
                "The update was interrupted ({reason}) and automatic rollback failed: {error:#}. The recovery app remains at {}.",
                transaction.backup_path.display()
            ));
            write_transaction(transaction)
        }
    }
}

fn start_recovered_candidate_supervisor(transaction: &mut UpdateTransaction) -> anyhow::Result<()> {
    start_recovered_candidate_supervisor_with(transaction, spawn_supervisor)
}

fn start_recovered_candidate_supervisor_with<F>(
    transaction: &mut UpdateTransaction,
    spawn: F,
) -> anyhow::Result<()>
where
    F: FnOnce(&UpdateTransaction) -> anyhow::Result<Child>,
{
    transaction.status = TransactionStatus::Installed;
    transaction.parent_pid = std::process::id();
    transaction.candidate_pid = Some(std::process::id());
    transaction.external_supervisor_ready = false;
    let _ = std::fs::remove_file(&transaction.health_path);
    write_transaction(transaction)?;
    let mut supervisor = match spawn(transaction) {
        Ok(supervisor) => supervisor,
        Err(error) => {
            transaction.status = TransactionStatus::RollbackFailed;
            transaction.candidate_pid = None;
            transaction.external_supervisor_ready = false;
            transaction.message = Some(format!(
                "Mold could not start the rollback watchdog: {error:#}. Quit Mold, then copy the recovery app at {} to Applications.",
                transaction.backup_path.display()
            ));
            write_transaction(transaction)?;
            return Ok(());
        }
    };
    transaction.external_supervisor_ready = true;
    if let Err(ready_error) = write_transaction(transaction) {
        let _ = supervisor.kill();
        let _ = supervisor.wait();
        transaction.status = TransactionStatus::RollbackFailed;
        transaction.candidate_pid = None;
        transaction.external_supervisor_ready = false;
        transaction.message = Some(format!(
            "Mold started a rollback watchdog but could not durably claim it: {ready_error:#}. Quit Mold, then copy the recovery app at {} to Applications.",
            transaction.backup_path.display()
        ));
        write_transaction(transaction)?;
    }
    Ok(())
}

fn reconcile_interrupted_transaction(
    app: &AppHandle,
    transaction: &mut UpdateTransaction,
) -> anyhow::Result<()> {
    if !matches!(
        transaction.status,
        TransactionStatus::Installing | TransactionStatus::Installed
    ) {
        return Ok(());
    }

    // A token means the original backup supervisor launched this candidate
    // and already owns its probation. Starting another watchdog would race it.
    if transaction.status == TransactionStatus::Installed
        && std::env::var_os("MOLD_UPDATE_HEALTH_TOKEN").is_some()
    {
        return Ok(());
    }

    let running_version = running_bundle_version(app, transaction);
    match classify_interrupted_install(
        running_version.as_deref(),
        &transaction.current_version,
        &transaction.target_version,
    ) {
        InterruptedInstall::PreviousVersionIntact => mark_interrupted_rollback(
            transaction,
            "the machine stopped before the replacement was committed",
        ),
        InterruptedInstall::CandidateRunning => start_recovered_candidate_supervisor(transaction),
        InterruptedInstall::UnknownBundle => restore_interrupted_transaction(
            transaction,
            "the installed app identity did not match either transaction version",
        ),
    }
}

#[tauri::command]
pub fn take_update_recovery(app: AppHandle) -> Result<Option<UpdateRecovery>, UpdateCommandError> {
    let root = updater_root(&app)
        .map_err(|error| UpdateCommandError::unchanged("rollback", format!("{error:#}"), false))?;
    let path = root.join("transaction.json");
    if !path.exists() {
        return Ok(None);
    }
    let mut transaction = read_transaction(&path).map_err(|error| {
        UpdateCommandError::unchanged(
            "rollback",
            format!("The rollback record is unreadable: {error:#}"),
            false,
        )
    })?;
    reconcile_interrupted_transaction(&app, &mut transaction).map_err(|error| {
        UpdateCommandError::rollback_failed(format!(
            "Mold could not reconcile an interrupted update: {error:#}. Quit Mold, then copy the recovery app at {} to Applications.",
            transaction.backup_path.display()
        ))
    })?;
    if !matches!(
        transaction.status,
        TransactionStatus::RolledBack | TransactionStatus::RollbackFailed
    ) {
        return Ok(None);
    }
    let recovery = UpdateRecovery {
        restored_version: transaction.current_version.clone(),
        failed_version: transaction.target_version.clone(),
        message: transaction.message.clone().unwrap_or_else(|| {
            format!(
                "Mold restored {} after {} did not start cleanly.",
                transaction.current_version, transaction.target_version
            )
        }),
        rollback_failed: transaction.status == TransactionStatus::RollbackFailed,
        backup_path: (transaction.status == TransactionStatus::RollbackFailed)
            .then(|| transaction.backup_path.display().to_string()),
    };
    if transaction.status == TransactionStatus::RolledBack {
        cleanup_transaction(&transaction);
    }
    Ok(Some(recovery))
}

fn cleanup_transaction_with<F>(
    transaction: &UpdateTransaction,
    remove_marker: F,
) -> std::io::Result<()>
where
    F: FnOnce() -> std::io::Result<()>,
{
    // The transaction marker is the source of truth. Commit its removal to
    // disk before deleting the only known-good backup; cleanup after that is
    // orphan collection and may safely be retried by the OS/user later.
    remove_marker()?;
    if let Some(parent) = transaction.transaction_path.parent() {
        sync_directory(parent)?;
    }
    let _ = std::fs::remove_file(&transaction.health_path);
    let _ = std::fs::remove_dir_all(&transaction.backup_path);
    Ok(())
}

fn cleanup_transaction(transaction: &UpdateTransaction) {
    let result = cleanup_transaction_with(transaction, || {
        match std::fs::remove_file(&transaction.transaction_path) {
            Ok(()) => Ok(()),
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => Ok(()),
            Err(error) => Err(error),
        }
    });
    if let Err(error) = result {
        tracing::warn!(
            path = %transaction.transaction_path.display(),
            %error,
            "preserving updater backup because the transaction commit marker could not be durably removed"
        );
    }
}

fn process_is_alive(pid: u32) -> bool {
    let result = unsafe { libc::kill(pid as libc::pid_t, 0) };
    if result == 0 {
        return true;
    }
    std::io::Error::last_os_error().raw_os_error() != Some(libc::ESRCH)
}

fn terminate_process(pid: u32) {
    unsafe {
        libc::kill(pid as libc::pid_t, libc::SIGTERM);
    }
    let deadline = Instant::now() + Duration::from_secs(3);
    while process_is_alive(pid) && Instant::now() < deadline {
        std::thread::sleep(PARENT_POLL);
    }
    if process_is_alive(pid) {
        unsafe {
            libc::kill(pid as libc::pid_t, libc::SIGKILL);
        }
    }
}

fn launch_app(
    transaction: &UpdateTransaction,
    app_path: &Path,
    health_token: Option<&str>,
) -> anyhow::Result<Child> {
    let expected_version = if health_token.is_some() {
        &transaction.target_version
    } else {
        &transaction.current_version
    };
    validate_bundle_identity(app_path, EXPECTED_BUNDLE_IDENTIFIER, expected_version)?;
    let executable = app_path.join(&transaction.executable_relative);
    let mut command = Command::new(&executable);
    command
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(Stdio::null());
    if let Some(token) = health_token {
        command.env("MOLD_UPDATE_HEALTH_TOKEN", token);
    } else {
        command.env("MOLD_UPDATE_ROLLBACK", &transaction.target_version);
    }
    command
        .spawn()
        .map_err(|error| anyhow::anyhow!("could not launch {}: {error}", executable.display()))
}

fn health_matches(transaction: &UpdateTransaction) -> bool {
    std::fs::read_to_string(&transaction.health_path)
        .map(|value| value == transaction.token)
        .unwrap_or(false)
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum SupervisorPoll {
    Rollback,
    Wait(Option<Instant>),
    Commit,
}

fn supervisor_poll(
    child_exited: bool,
    health_confirmed: bool,
    health_since: Option<Instant>,
    now: Instant,
) -> SupervisorPoll {
    // Process failure always wins a same-poll health marker. The marker only
    // starts probation; it never commits the candidate immediately.
    if child_exited {
        return SupervisorPoll::Rollback;
    }
    if !health_confirmed {
        return SupervisorPoll::Wait(health_since);
    }
    let started = health_since.unwrap_or(now);
    if now.saturating_duration_since(started) >= HEALTH_PROBATION {
        SupervisorPoll::Commit
    } else {
        SupervisorPoll::Wait(Some(started))
    }
}

fn rollback_after_failed_launch(
    transaction: &mut UpdateTransaction,
    launch_error: &str,
) -> anyhow::Result<()> {
    transaction.status = TransactionStatus::RolledBack;
    transaction.message = Some(format!(
        "Mold {} did not start cleanly ({launch_error}). Mold {} was restored.",
        transaction.target_version, transaction.current_version
    ));
    match restore_bundle(&transaction.backup_path, &transaction.app_path) {
        Ok(()) => {
            let marker = write_transaction(transaction);
            let launch = launch_app(transaction, &transaction.app_path, None);
            marker?;
            let _ = launch?;
            Ok(())
        }
        Err(error) => {
            transaction.status = TransactionStatus::RollbackFailed;
            transaction.message = Some(format!(
                "Mold {} did not start cleanly and automatic rollback failed: {error:#}. The backup remains at {}.",
                transaction.target_version,
                transaction.backup_path.display()
            ));
            let marker = write_transaction(transaction);
            // Even if restoring into Applications failed, run the signed
            // backup in place so the user is not stranded without Mold.
            let launch = transaction
                .backup_path
                .exists()
                .then(|| launch_app(transaction, &transaction.backup_path, None));
            marker?;
            if let Some(launch) = launch {
                let _ = launch?;
            }
            Err(error)
        }
    }
}

fn supervise_child(transaction: &mut UpdateTransaction, child: &mut Child) -> anyhow::Result<()> {
    let handshake_deadline = Instant::now() + HEALTH_TIMEOUT;
    let mut health_since = None;
    loop {
        let status = child.try_wait()?;
        let now = Instant::now();
        match supervisor_poll(
            status.is_some(),
            health_matches(transaction),
            health_since,
            now,
        ) {
            SupervisorPoll::Rollback => {
                return rollback_after_failed_launch(
                    transaction,
                    &format!(
                        "the updated process exited with {}",
                        status.expect("rollback requires an exit status")
                    ),
                );
            }
            SupervisorPoll::Commit => {
                cleanup_transaction(transaction);
                return Ok(());
            }
            SupervisorPoll::Wait(started) => health_since = started,
        }
        if health_since.is_none() && now >= handshake_deadline {
            let _ = child.kill();
            let _ = child.wait();
            return rollback_after_failed_launch(
                transaction,
                "the interface did not become healthy within 60 seconds",
            );
        }
        std::thread::sleep(PARENT_POLL);
    }
}

fn supervise_external_candidate(
    transaction: &mut UpdateTransaction,
    pid: u32,
) -> anyhow::Result<()> {
    let handshake_deadline = Instant::now() + HEALTH_TIMEOUT;
    let mut health_since = None;
    loop {
        let exited = !process_is_alive(pid);
        let now = Instant::now();
        match supervisor_poll(exited, health_matches(transaction), health_since, now) {
            SupervisorPoll::Rollback => {
                return rollback_after_failed_launch(
                    transaction,
                    "the updated process exited during startup probation",
                );
            }
            SupervisorPoll::Commit => {
                cleanup_transaction(transaction);
                return Ok(());
            }
            SupervisorPoll::Wait(started) => health_since = started,
        }
        if health_since.is_none() && now >= handshake_deadline {
            terminate_process(pid);
            return rollback_after_failed_launch(
                transaction,
                "the interface did not become healthy within 60 seconds",
            );
        }
        std::thread::sleep(PARENT_POLL);
    }
}

fn supervise_update(path: &Path) -> anyhow::Result<()> {
    let mut transaction = read_transaction(path)?;

    if let Some(candidate_pid) = transaction.candidate_pid {
        return supervise_external_candidate(&mut transaction, candidate_pid);
    }

    while process_is_alive(transaction.parent_pid) {
        std::thread::sleep(PARENT_POLL);
        transaction = match read_transaction(path) {
            Ok(transaction) => transaction,
            Err(_) => return Ok(()),
        };
        if transaction.status == TransactionStatus::RolledBack {
            return Ok(());
        }
    }

    transaction = read_transaction(path)?;
    if transaction.status == TransactionStatus::RollbackFailed {
        if transaction.backup_path.exists() {
            let _ = launch_app(&transaction, &transaction.backup_path, None)?;
        }
        return Ok(());
    }
    if transaction.status != TransactionStatus::Installed {
        return rollback_after_failed_launch(
            &mut transaction,
            "the installer stopped before committing the update",
        );
    }

    let mut child = match launch_app(
        &transaction,
        &transaction.app_path,
        Some(&transaction.token),
    ) {
        Ok(child) => child,
        Err(error) => {
            return rollback_after_failed_launch(&mut transaction, &error.to_string());
        }
    };

    supervise_child(&mut transaction, &mut child)
}

/// The copied, previously healthy app binary runs this mode without creating
/// a Tauri webview. It survives replacement of the primary `.app`, launches
/// the candidate, and restores the backup if the health handshake never lands.
pub fn run_supervisor_if_requested() -> bool {
    let mut args = std::env::args_os();
    let _ = args.next();
    let Some(first) = args.next() else {
        return false;
    };
    if first != SUPERVISOR_ARG {
        return false;
    }
    let Some(path) = args.next() else {
        eprintln!("mold update supervisor: missing transaction path");
        return true;
    };
    if let Err(error) = supervise_update(Path::new(&path)) {
        eprintln!("mold update supervisor failed: {error:#}");
    }
    true
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn channels_map_to_allowlisted_https_manifests() {
        assert_eq!(
            endpoint_for(UpdateChannel::Stable),
            "https://github.com/utensils/mold/releases/latest/download/mold-desktop-stable.json"
        );
        assert_eq!(
            endpoint_for(UpdateChannel::Nightly),
            "https://github.com/utensils/mold/releases/download/latest/mold-desktop-nightly.json"
        );
    }

    #[test]
    fn failed_swap_restores_the_previous_bundle() {
        let dir = tempfile::tempdir().unwrap();
        let current = dir.path().join("Mold.app");
        let backup = dir.path().join("Mold.backup.app");
        std::fs::create_dir_all(&current).unwrap();
        std::fs::write(current.join("version"), "0.16.0").unwrap();

        copy_bundle(&current, &backup).unwrap();
        std::fs::remove_dir_all(&current).unwrap();
        std::fs::create_dir_all(&current).unwrap();
        std::fs::write(current.join("partial"), "broken").unwrap();

        restore_bundle(&backup, &current).unwrap();

        assert_eq!(
            std::fs::read_to_string(current.join("version")).unwrap(),
            "0.16.0"
        );
        assert!(!current.join("partial").exists());
    }

    #[test]
    fn failed_second_rename_restores_the_current_bundle() {
        let dir = tempfile::tempdir().unwrap();
        let current = dir.path().join("Mold.app");
        let missing_backup = dir.path().join("missing.app");
        std::fs::create_dir_all(&current).unwrap();
        std::fs::write(current.join("version"), "current").unwrap();

        assert!(restore_bundle(&missing_backup, &current).is_err());
        assert_eq!(
            std::fs::read_to_string(current.join("version")).unwrap(),
            "current"
        );
    }

    #[test]
    fn health_confirmation_requires_the_transaction_token() {
        let dir = tempfile::tempdir().unwrap();
        let transaction = UpdateTransaction::test_fixture(dir.path());

        assert!(!confirm_health(&transaction, "wrong-token").unwrap());
        assert!(!transaction.health_path.exists());
        assert!(confirm_health(&transaction, &transaction.token).unwrap());
        assert_eq!(
            std::fs::read_to_string(&transaction.health_path).unwrap(),
            transaction.token
        );
    }

    #[test]
    fn tokenless_health_requires_a_ready_watchdog_for_this_process() {
        let dir = tempfile::tempdir().unwrap();
        let mut transaction = UpdateTransaction::test_fixture(dir.path());
        transaction.status = TransactionStatus::Installed;
        transaction.candidate_pid = Some(std::process::id());

        assert!(!tokenless_health_allowed(&transaction, std::process::id()));
        transaction.external_supervisor_ready = true;
        assert!(tokenless_health_allowed(&transaction, std::process::id()));
        assert!(!tokenless_health_allowed(
            &transaction,
            std::process::id().saturating_add(1)
        ));
    }

    #[test]
    fn recovered_watchdog_spawn_failure_becomes_visible_manual_recovery() {
        let dir = tempfile::tempdir().unwrap();
        let mut transaction = UpdateTransaction::test_fixture(dir.path());

        start_recovered_candidate_supervisor_with(&mut transaction, |_| {
            anyhow::bail!("watchdog unavailable")
        })
        .unwrap();

        let persisted = read_transaction(&transaction.transaction_path).unwrap();
        assert_eq!(persisted.status, TransactionStatus::RollbackFailed);
        assert_eq!(persisted.candidate_pid, None);
        assert!(!persisted.external_supervisor_ready);
        assert!(persisted
            .message
            .as_deref()
            .unwrap()
            .contains("watchdog unavailable"));
        assert!(persisted
            .message
            .as_deref()
            .unwrap()
            .contains(&persisted.backup_path.display().to_string()));
        assert_eq!(
            read_rollback_failed_transaction(dir.path())
                .unwrap()
                .unwrap()
                .status,
            TransactionStatus::RollbackFailed
        );
    }

    #[test]
    fn cleanup_preserves_backup_when_commit_marker_cannot_be_removed() {
        let dir = tempfile::tempdir().unwrap();
        let transaction = UpdateTransaction::test_fixture(dir.path());
        std::fs::create_dir_all(&transaction.backup_path).unwrap();
        std::fs::write(&transaction.health_path, &transaction.token).unwrap();
        write_transaction(&transaction).unwrap();

        let result = cleanup_transaction_with(&transaction, || {
            Err(std::io::Error::new(
                std::io::ErrorKind::PermissionDenied,
                "marker locked",
            ))
        });

        assert!(result.is_err());
        assert!(transaction.transaction_path.exists());
        assert!(transaction.backup_path.exists());
        assert!(transaction.health_path.exists());
    }

    #[test]
    fn health_confirmation_starts_probation_and_exit_wins_the_same_poll() {
        let started = Instant::now();

        assert_eq!(
            supervisor_poll(true, true, None, started),
            SupervisorPoll::Rollback
        );
        assert_eq!(
            supervisor_poll(false, true, None, started),
            SupervisorPoll::Wait(Some(started))
        );
        assert_eq!(
            supervisor_poll(
                false,
                true,
                Some(started),
                started + HEALTH_PROBATION - Duration::from_millis(1),
            ),
            SupervisorPoll::Wait(Some(started))
        );
        assert_eq!(
            supervisor_poll(false, true, Some(started), started + HEALTH_PROBATION,),
            SupervisorPoll::Commit
        );
    }

    #[test]
    fn interrupted_install_classification_does_not_wedge_the_next_launch() {
        assert_eq!(
            classify_interrupted_install(Some("0.16.0"), "0.16.0", "0.17.0"),
            InterruptedInstall::PreviousVersionIntact
        );
        assert_eq!(
            classify_interrupted_install(Some("0.17.0"), "0.16.0", "0.17.0"),
            InterruptedInstall::CandidateRunning
        );
        assert_eq!(
            classify_interrupted_install(Some("0.16.5"), "0.16.0", "0.17.0"),
            InterruptedInstall::UnknownBundle
        );
    }

    #[test]
    fn verified_archive_identity_must_match_manifest_target() {
        let matching = test_update_archive("com.utensils.mold", "0.17.0");
        validate_update_archive(&matching, "com.utensils.mold", "0.17.0").unwrap();

        let replayed = test_update_archive("com.utensils.mold", "0.16.0");
        assert!(
            validate_update_archive(&replayed, "com.utensils.mold", "0.17.0")
                .unwrap_err()
                .to_string()
                .contains("version")
        );

        let wrong_app = test_update_archive("com.example.not-mold", "0.17.0");
        assert!(
            validate_update_archive(&wrong_app, "com.utensils.mold", "0.17.0")
                .unwrap_err()
                .to_string()
                .contains("identifier")
        );
    }

    fn test_update_archive(identifier: &str, version: &str) -> Vec<u8> {
        use std::io::Write;

        let mut plist = Vec::new();
        plist::Value::Dictionary(plist::Dictionary::from_iter([
            (
                "CFBundleIdentifier".to_string(),
                plist::Value::String(identifier.into()),
            ),
            (
                "CFBundleShortVersionString".to_string(),
                plist::Value::String(version.into()),
            ),
        ]))
        .to_writer_xml(&mut plist)
        .unwrap();

        let mut compressed = Vec::new();
        {
            let encoder =
                flate2::write::GzEncoder::new(&mut compressed, flate2::Compression::default());
            let mut archive = tar::Builder::new(encoder);
            let mut header = tar::Header::new_gnu();
            header.set_size(plist.len() as u64);
            header.set_mode(0o644);
            header.set_cksum();
            archive
                .append_data(
                    &mut header,
                    "Mold.app/Contents/Info.plist",
                    plist.as_slice(),
                )
                .unwrap();
            let encoder = archive.into_inner().unwrap();
            encoder.finish().unwrap();
        }
        compressed.flush().unwrap();
        compressed
    }

    #[test]
    fn app_bundle_detection_rejects_dev_executables() {
        assert_eq!(
            app_bundle_from_executable(Path::new("/tmp/mold-desktop")),
            None
        );
        assert_eq!(
            app_bundle_from_executable(Path::new(
                "/Applications/Mold.app/Contents/MacOS/mold-desktop"
            )),
            Some(PathBuf::from("/Applications/Mold.app"))
        );
    }

    #[test]
    fn disk_images_and_app_translocation_are_rejected() {
        assert!(reject_unsafe_install_location(Path::new("/Volumes/Mold/Mold.app")).is_err());
        assert!(reject_unsafe_install_location(Path::new(
            "/private/var/folders/x/AppTranslocation/y/Mold.app"
        ))
        .is_err());
        assert!(reject_unsafe_install_location(Path::new("/Applications/Mold.app")).is_ok());
    }

    #[test]
    fn transaction_writes_atomically_and_round_trips() {
        let dir = tempfile::tempdir().unwrap();
        let transaction = UpdateTransaction::test_fixture(dir.path());
        write_transaction(&transaction).unwrap();

        let loaded = read_transaction(&transaction.transaction_path).unwrap();
        assert_eq!(loaded.token, transaction.token);
        assert_eq!(loaded.status, TransactionStatus::Prepared);
        assert!(!transaction
            .transaction_path
            .with_extension("json.tmp")
            .exists());
    }

    #[test]
    fn progress_and_errors_keep_the_frontend_wire_shape() {
        let progress = serde_json::to_value(UpdateProgress {
            candidate_id: "candidate-1".into(),
            phase: "downloading",
            downloaded_bytes: Some(25),
            total_bytes: Some(100),
        })
        .unwrap();
        assert_eq!(progress["candidateId"], "candidate-1");
        assert_eq!(progress["downloadedBytes"], 25);
        assert_eq!(progress["totalBytes"], 100);

        let error =
            serde_json::to_value(UpdateCommandError::rolled_back("restored", true)).unwrap();
        assert_eq!(error["disposition"], "rolled-back");
        assert_eq!(error["retryable"], true);
    }
}
