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

/// Bring a local engine online. Prefers a mold server the user already runs
/// on the well-known port (avoids two engines sharing mold.db and the models
/// dir); otherwise embeds one on an ephemeral loopback port.
#[tauri::command]
pub async fn start_local_engine(
    state: tauri::State<'_, AppState>,
) -> Result<ConnectionInfo, String> {
    let mut conn = state.conn.lock().await;
    if matches!(&*conn, Conn::Local(_) | Conn::External { .. }) {
        return Ok(conn.info(&state.local_api_key));
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

    let updated = {
        let mut current = store.current.lock().expect("settings mutex");
        current.mode = ConnectionMode::Remote;
        current.remote_url = Some(url);
        current.remote_api_key = api_key;
        current.clone()
    };
    settings::save(&store.path, &updated).map_err(|e| e.to_string())?;
    Ok(conn.info(&state.local_api_key))
}
