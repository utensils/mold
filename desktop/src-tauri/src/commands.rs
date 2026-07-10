use std::path::PathBuf;
use std::sync::Mutex;
use std::time::Duration;

use serde::Serialize;
use tauri::Manager;

use crate::connection::{normalize_host_url, Conn, ConnectionInfo};
use crate::server;
use crate::settings::{self, AppSettings, ConnectionMode};

pub struct SettingsStore {
    pub path: PathBuf,
    pub current: Mutex<AppSettings>,
}

impl SettingsStore {
    pub fn load(app: &tauri::AppHandle) -> anyhow::Result<Self> {
        let path = app.path().app_data_dir()?.join("settings.json");
        let current = Mutex::new(settings::load(&path));
        Ok(Self { path, current })
    }
}

pub struct AppState {
    pub conn: tokio::sync::Mutex<Conn>,
    /// Ephemeral per-launch key for the embedded engine; exported as
    /// MOLD_API_KEY at startup so only this app can drive the loopback server.
    pub local_api_key: String,
    pub secrets: crate::secrets::SecretStore,
}

/// Env knobs the Settings Performance section may set on the embedded
/// engine. Applied (or removed when unset) right before the engine thread
/// spawns — restart the engine to apply changes.
pub const ENGINE_ENV_KEYS: &[&str] = &[
    "MOLD_STEP_PREVIEW",
    "MOLD_KEEP_TE_RAM",
    "MOLD_VAE_TILED",
    "MOLD_OFFLOAD",
    "MOLD_ATTN",
    "MOLD_QUEUE_SIZE",
];

/// Apply the Performance env knobs plus HF/Civitai tokens to this process's
/// environment so the embedded engine (and its downloads) see them.
fn apply_engine_environment(
    engine_env: &std::collections::HashMap<String, String>,
    secrets: &crate::secrets::SecretStore,
) {
    for key in ENGINE_ENV_KEYS {
        match engine_env.get(*key).map(String::as_str) {
            Some(v) if !v.is_empty() => std::env::set_var(key, v),
            _ => std::env::remove_var(key),
        }
    }
    for (secret, env) in [("hf-token", "HF_TOKEN"), ("civitai-token", "CIVITAI_TOKEN")] {
        if let Ok(Some(v)) = secrets.get(secret) {
            if !v.is_empty() {
                std::env::set_var(env, v);
            }
        }
    }
}

#[tauri::command]
pub fn app_settings_get(store: tauri::State<'_, SettingsStore>) -> AppSettings {
    store.current.lock().expect("settings mutex").clone()
}

#[tauri::command]
pub fn app_settings_set(
    store: tauri::State<'_, SettingsStore>,
    settings: AppSettings,
) -> Result<(), String> {
    settings::save(&store.path, &settings).map_err(|e| e.to_string())?;
    *store.current.lock().expect("settings mutex") = settings;
    Ok(())
}

#[tauri::command]
pub async fn get_connection(state: tauri::State<'_, AppState>) -> Result<ConnectionInfo, String> {
    Ok(state.conn.lock().await.info(&state.local_api_key))
}

/// Where the local engine writes gallery files. `None` for remote hosts —
/// reveal-in-Finder and drag-out only make sense for on-disk media.
#[tauri::command]
pub async fn get_output_dir(state: tauri::State<'_, AppState>) -> Result<Option<String>, String> {
    let conn = state.conn.lock().await;
    Ok(match &*conn {
        Conn::Local(_) | Conn::External { .. } => {
            let config = mold_core::Config::load_or_default();
            if config.is_output_disabled() {
                None
            } else {
                Some(config.effective_output_dir().to_string_lossy().into_owned())
            }
        }
        _ => None,
    })
}

/// Dock badge: queue depth while jobs wait, cleared when idle (macOS).
#[tauri::command]
pub fn set_dock_badge(app: tauri::AppHandle, count: Option<i64>) -> Result<(), String> {
    let window = app
        .get_webview_window("main")
        .ok_or_else(|| "no main window".to_string())?;
    window
        .set_badge_count(count.filter(|c| *c > 0))
        .map_err(|e| e.to_string())
}

/// Reveal an engine-written file in Finder (local mode only).
#[tauri::command]
pub async fn reveal_output_file(
    state: tauri::State<'_, AppState>,
    app: tauri::AppHandle,
    filename: String,
) -> Result<(), String> {
    use tauri_plugin_opener::OpenerExt;
    // Gallery filenames are server-generated basenames; refuse traversal.
    if filename.contains('/') || filename.contains("..") {
        return Err("Invalid filename.".into());
    }
    let dir = {
        let conn = state.conn.lock().await;
        match &*conn {
            Conn::Local(_) | Conn::External { .. } => {
                mold_core::Config::load_or_default().effective_output_dir()
            }
            _ => return Err("Reveal in Finder works with the built-in engine only.".into()),
        }
    };
    let path = dir.join(&filename);
    if !path.exists() {
        return Err("The file is no longer on disk.".into());
    }
    app.opener()
        .reveal_item_in_dir(&path)
        .map_err(|e| e.to_string())
}

/// Bring a local engine online. Prefers a mold server the user already runs
/// on the well-known port (avoids two engines sharing mold.db and the models
/// dir); otherwise embeds one on an ephemeral loopback port.
#[tauri::command]
pub async fn start_local_engine(
    state: tauri::State<'_, AppState>,
    store: tauri::State<'_, SettingsStore>,
) -> Result<ConnectionInfo, String> {
    let mut conn = state.conn.lock().await;
    match &*conn {
        // A dead engine thread (crash or shutdown) falls through to a
        // fresh start instead of returning a base URL nothing answers.
        Conn::Local(engine) if engine.is_alive() => {
            return Ok(conn.info(&state.local_api_key));
        }
        Conn::Local(_) => {
            tracing::warn!("embedded engine thread is gone; restarting");
            *conn = Conn::Off;
        }
        Conn::External { .. } => return Ok(conn.info(&state.local_api_key)),
        _ => {}
    }

    let well_known = format!("http://127.0.0.1:{}", server::WELL_KNOWN_PORT);
    if server::is_mold_server(&well_known).await {
        tracing::info!("using existing mold server at {well_known}");
        *conn = Conn::External {
            base_url: well_known,
        };
        return Ok(conn.info(&state.local_api_key));
    }

    let config = mold_core::Config::load_or_default();
    let models_dir = config.resolved_models_dir();
    let gpu_selection = config.gpu_selection();
    apply_engine_environment(
        &store
            .current
            .lock()
            .expect("settings mutex")
            .engine_env
            .clone(),
        &state.secrets,
    );
    let engine = server::start_engine(models_dir, gpu_selection).map_err(|e| format!("{e:#}"))?;
    let base_url = engine.base_url();
    if !server::wait_healthy(&base_url, Duration::from_secs(30)).await {
        return Err("The engine didn't start. Check the logs (~/.mold/logs).".into());
    }
    *conn = Conn::Local(engine);
    Ok(conn.info(&state.local_api_key))
}

#[tauri::command]
pub async fn stop_local_engine(
    state: tauri::State<'_, AppState>,
) -> Result<ConnectionInfo, String> {
    let mut conn = state.conn.lock().await;
    if let Conn::Local(_) = &*conn {
        let Conn::Local(engine) = std::mem::replace(&mut *conn, Conn::Off) else {
            unreachable!()
        };
        let shutdown_url = format!("{}/api/shutdown", engine.base_url());
        let _ = reqwest::Client::new()
            .post(&shutdown_url)
            .header("X-Api-Key", &state.local_api_key)
            .timeout(Duration::from_secs(3))
            .send()
            .await;
        tauri::async_runtime::spawn_blocking(move || engine.join(Duration::from_secs(5)))
            .await
            .map_err(|e| e.to_string())?;
    } else {
        *conn = Conn::Off;
    }
    Ok(conn.info(&state.local_api_key))
}

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct HostTest {
    pub ok: bool,
    pub version: Option<String>,
    pub error: Option<String>,
}

async fn probe_host(url: &str, api_key: Option<&str>) -> HostTest {
    let client = reqwest::Client::new();
    let health = client
        .get(format!("{url}/health"))
        .timeout(Duration::from_secs(4))
        .send()
        .await;
    match health {
        Ok(resp) if resp.status().is_success() => {}
        Ok(resp) => {
            return HostTest {
                ok: false,
                version: None,
                error: Some(format!("{url} answered {} on /health.", resp.status())),
            }
        }
        Err(e) => {
            return HostTest {
                ok: false,
                version: None,
                error: Some(format!("Can't reach {url}: {e}")),
            }
        }
    }
    let mut status = client
        .get(format!("{url}/api/status"))
        .timeout(Duration::from_secs(4));
    if let Some(key) = api_key {
        status = status.header("X-Api-Key", key);
    }
    match status.send().await {
        Ok(resp) if resp.status() == reqwest::StatusCode::UNAUTHORIZED => HostTest {
            ok: false,
            version: None,
            error: Some("This host requires an API key.".into()),
        },
        Ok(resp) if resp.status().is_success() => {
            let version = resp
                .json::<serde_json::Value>()
                .await
                .ok()
                .and_then(|v| v.get("version")?.as_str().map(str::to_string));
            HostTest {
                ok: true,
                version,
                error: None,
            }
        }
        Ok(resp) => HostTest {
            ok: false,
            version: None,
            error: Some(format!("{url} answered {} on /api/status.", resp.status())),
        },
        Err(e) => HostTest {
            ok: false,
            version: None,
            error: Some(format!("Can't reach {url}: {e}")),
        },
    }
}

#[tauri::command]
pub async fn test_remote_host(url: String, api_key: Option<String>) -> Result<HostTest, String> {
    let url = normalize_host_url(&url)?;
    Ok(probe_host(&url, api_key.as_deref()).await)
}

/// Switch to a remote host (validates first) and persist it in app settings.
#[tauri::command]
pub async fn set_remote_host(
    state: tauri::State<'_, AppState>,
    store: tauri::State<'_, SettingsStore>,
    url: String,
    api_key: Option<String>,
) -> Result<ConnectionInfo, String> {
    let url = normalize_host_url(&url)?;
    let test = probe_host(&url, api_key.as_deref()).await;
    if !test.ok {
        return Err(test.error.unwrap_or_else(|| "Connection failed.".into()));
    }

    let mut conn = state.conn.lock().await;
    *conn = Conn::Remote {
        url: url.clone(),
        api_key: api_key.clone(),
    };

    // The key lives in the Keychain; the legacy plaintext slot is wiped so
    // old settings.json files converge on first save.
    match api_key.as_deref().filter(|k| !k.is_empty()) {
        Some(key) => state
            .secrets
            .set("remote-api-key", key)
            .map_err(|e| e.to_string())?,
        None => state
            .secrets
            .clear("remote-api-key")
            .map_err(|e| e.to_string())?,
    }
    let updated = {
        let mut current = store.current.lock().expect("settings mutex");
        current.mode = ConnectionMode::Remote;
        current.remote_url = Some(url);
        current.remote_api_key = None;
        current.clone()
    };
    settings::save(&store.path, &updated).map_err(|e| e.to_string())?;
    Ok(conn.info(&state.local_api_key))
}

/// Open the engine's log directory in Finder.
#[tauri::command]
pub fn open_logs_dir(app: tauri::AppHandle) -> Result<(), String> {
    use tauri_plugin_opener::OpenerExt;
    let dir = mold_core::Config::load_or_default().resolved_log_dir();
    app.opener()
        .open_path(dir.to_string_lossy(), None::<&str>)
        .map_err(|e| e.to_string())
}

#[tauri::command]
pub fn secret_get(
    state: tauri::State<'_, AppState>,
    name: String,
) -> Result<Option<String>, String> {
    state.secrets.get(&name).map_err(|e| e.to_string())
}

#[tauri::command]
pub fn secret_set(
    state: tauri::State<'_, AppState>,
    name: String,
    value: String,
) -> Result<(), String> {
    state.secrets.set(&name, &value).map_err(|e| e.to_string())
}

#[tauri::command]
pub fn secret_clear(state: tauri::State<'_, AppState>, name: String) -> Result<(), String> {
    state.secrets.clear(&name).map_err(|e| e.to_string())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn engine_environment_sets_and_clears_knobs_and_tokens() {
        let dir = tempfile::tempdir().unwrap();
        let secrets = crate::secrets::SecretStore::file_only(dir.path().to_path_buf());
        secrets.set("hf-token", "hf_test_token").unwrap();

        let mut env = std::collections::HashMap::new();
        env.insert("MOLD_VAE_TILED".to_string(), "force".to_string());
        env.insert("MOLD_STEP_PREVIEW".to_string(), String::new()); // empty = unset
        apply_engine_environment(&env, &secrets);
        assert_eq!(std::env::var("MOLD_VAE_TILED").as_deref(), Ok("force"));
        assert!(std::env::var("MOLD_STEP_PREVIEW").is_err());
        assert_eq!(std::env::var("HF_TOKEN").as_deref(), Ok("hf_test_token"));

        // Removing the knob from the map clears the variable on next apply.
        env.remove("MOLD_VAE_TILED");
        apply_engine_environment(&env, &secrets);
        assert!(std::env::var("MOLD_VAE_TILED").is_err());
        std::env::remove_var("HF_TOKEN");
    }
}
