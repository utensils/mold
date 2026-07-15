use std::path::PathBuf;
use std::sync::Mutex;
use std::time::Duration;

use serde::Serialize;
use tauri::Manager;

use crate::connection::{host_id, normalize_host_url, Conn, ConnectionInfo};
use crate::server;
use crate::settings::{
    self, remember_connected_host, upsert_saved_host, AppSettings, ConnectionMode, SavedHost,
};

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
    pub local_server: tokio::sync::Mutex<LocalServer>,
    /// Persistent per-install key for the desktop-owned LAN server; exported
    /// as MOLD_API_KEY before the embedded engine starts.
    pub local_api_key: String,
    pub secrets: crate::secrets::SecretStore,
}

pub enum LocalServer {
    Off,
    Embedded(server::EngineHandle),
    External { base_url: String },
}

#[derive(Debug, Clone, Serialize, PartialEq, Eq)]
#[serde(rename_all = "camelCase")]
pub struct LocalServerInfo {
    pub kind: &'static str,
    pub base_url: String,
    pub api_key: Option<String>,
    pub port: u16,
}

impl LocalServer {
    fn info(&self, api_key: &str) -> Option<LocalServerInfo> {
        let (kind, base_url, port) = match self {
            LocalServer::Off => return None,
            LocalServer::Embedded(engine) => ("embedded", engine.base_url(), engine.port),
            LocalServer::External { base_url } => {
                let port = reqwest::Url::parse(base_url)
                    .ok()
                    .and_then(|url| url.port_or_known_default())
                    .unwrap_or(server::WELL_KNOWN_PORT);
                ("external", base_url.clone(), port)
            }
        };
        Some(LocalServerInfo {
            kind,
            base_url,
            api_key: Some(api_key.to_string()),
            port,
        })
    }
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
        match secrets.get(secret) {
            Ok(Some(v)) if !v.is_empty() => std::env::set_var(env, v),
            // Cleared/missing tokens must not linger across engine restarts.
            _ => std::env::remove_var(env),
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

/// Where this Mac writes gallery files. Remote engine selection does not
/// hide the local gallery from the user.
#[tauri::command]
pub async fn get_output_dir(state: tauri::State<'_, AppState>) -> Result<Option<String>, String> {
    let _ = state;
    let config = mold_core::Config::load_or_default();
    Ok((!config.is_output_disabled())
        .then(|| config.effective_output_dir().to_string_lossy().into_owned()))
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
    let _ = state;
    let dir = mold_core::Config::load_or_default().effective_output_dir();
    let path = dir.join(&filename);
    if !path.exists() {
        return Err("The file is no longer on disk.".into());
    }
    app.opener()
        .reveal_item_in_dir(&path)
        .map_err(|e| e.to_string())
}

async fn ensure_local_server_inner(
    state: &AppState,
    store: &SettingsStore,
) -> Result<LocalServerInfo, String> {
    let mut local = state.local_server.lock().await;
    let external_url = match &*local {
        LocalServer::Embedded(engine) if engine.is_alive() => {
            return Ok(local.info(&state.local_api_key).expect("embedded info"));
        }
        LocalServer::Embedded(_) => {
            tracing::warn!("embedded engine thread is gone; restarting");
            *local = LocalServer::Off;
            None
        }
        LocalServer::External { base_url } => Some(base_url.clone()),
        LocalServer::Off => None,
    };
    if let Some(base_url) = external_url {
        if server::is_mold_server(&base_url).await {
            ensure_local_server_auth(&base_url, &state.local_api_key).await?;
            return Ok(local.info(&state.local_api_key).expect("external info"));
        }
        tracing::warn!(%base_url, "external local server disappeared; replacing it");
        *local = LocalServer::Off;
    }

    // A user-run server owns the shared model directory and metadata DB, so it
    // wins over embedding another process on this Mac.
    let well_known = format!("http://127.0.0.1:{}", server::WELL_KNOWN_PORT);
    if server::is_mold_server(&well_known).await {
        ensure_local_server_auth(&well_known, &state.local_api_key).await?;
        tracing::info!("using existing mold server at {well_known}");
        *local = LocalServer::External {
            base_url: well_known,
        };
        return Ok(local.info(&state.local_api_key).expect("external info"));
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
    let port = server::available_server_port(server::LAN_BIND).map_err(|e| format!("{e:#}"))?;
    let engine = server::start_engine(server::LAN_BIND, port, models_dir, gpu_selection)
        .map_err(|e| format!("{e:#}"))?;
    let base_url = engine.base_url();
    if !server::wait_healthy(&base_url, Duration::from_secs(30)).await {
        engine.join(Duration::from_secs(1));
        return Err("The engine didn't start. Check the logs (~/.mold/logs).".into());
    }
    *local = LocalServer::Embedded(engine);
    Ok(local.info(&state.local_api_key).expect("embedded info"))
}

async fn ensure_local_server_auth(base_url: &str, api_key: &str) -> Result<(), String> {
    if server::accepts_api_key(base_url, api_key).await {
        return Ok(());
    }
    Err(format!(
        "A Mold server is already running at {base_url}, but it does not accept This Mac API key. Stop that server or restart it with the same MOLD_API_KEY."
    ))
}

/// Ensure this Mac is serving Mold independently of whichever primary host the
/// UI currently uses. The embedded server binds to the LAN and advertises over
/// mDNS; the returned URL remains loopback for the local webview.
#[tauri::command]
pub async fn ensure_local_server(
    state: tauri::State<'_, AppState>,
    store: tauri::State<'_, SettingsStore>,
) -> Result<LocalServerInfo, String> {
    ensure_local_server_inner(&state, &store).await
}

/// Bring the local server online and select it as the primary connection.
#[tauri::command]
pub async fn start_local_engine(
    state: tauri::State<'_, AppState>,
    store: tauri::State<'_, SettingsStore>,
) -> Result<ConnectionInfo, String> {
    let local = ensure_local_server_inner(&state, &store).await?;
    let mut conn = state.conn.lock().await;
    *conn = if local.kind == "embedded" {
        Conn::Local {
            base_url: local.base_url,
        }
    } else {
        Conn::External {
            base_url: local.base_url,
        }
    };
    Ok(conn.info(&state.local_api_key))
}

#[tauri::command]
pub async fn stop_local_engine(
    state: tauri::State<'_, AppState>,
) -> Result<ConnectionInfo, String> {
    let engine = {
        let mut local = state.local_server.lock().await;
        match std::mem::replace(&mut *local, LocalServer::Off) {
            LocalServer::Embedded(engine) => Some(engine),
            LocalServer::External { .. } | LocalServer::Off => None,
        }
    };
    if let Some(engine) = engine {
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
    }
    let mut conn = state.conn.lock().await;
    if primary_uses_local_server(&conn) {
        *conn = Conn::Off;
    }
    Ok(conn.info(&state.local_api_key))
}

fn primary_uses_local_server(conn: &Conn) -> bool {
    matches!(conn, Conn::Local { .. } | Conn::External { .. })
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

/// A mold server discovered on the LAN via mDNS, shaped for the frontend.
#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct DiscoveredHost {
    pub name: String,
    pub url: String,
    pub host: String,
    pub port: u16,
    pub version: Option<String>,
    pub auth_required: bool,
    /// True when the advertisement resolves to one of this machine's own
    /// interface addresses (or its hostname) — i.e. our embedded/local server.
    pub is_this_machine: bool,
}

/// Whether any of a discovered server's addresses (or hostname) belong to this
/// machine. Pure over its inputs so it is unit-testable without a live network.
fn host_is_this_machine(
    addresses: &[String],
    server_name: &str,
    local_addrs: &[std::net::IpAddr],
    local_hostname: &str,
) -> bool {
    let addr_match = addresses.iter().any(|a| {
        a.parse::<std::net::IpAddr>()
            .map(|ip| local_addrs.contains(&ip))
            .unwrap_or(false)
    });
    if addr_match {
        return true;
    }
    // Fall back to a hostname match (instance names are `<host>-<port>`).
    // Require the `-` delimiter so local host `box` doesn't claim `box2-7680`.
    let short = local_hostname.split('.').next().unwrap_or(local_hostname);
    if short.is_empty() {
        return false;
    }
    let short = short.to_ascii_lowercase();
    let name = server_name.to_ascii_lowercase();
    name == short || name.starts_with(&format!("{short}-"))
}

/// Browse the local network for advertised mold servers.
#[cfg(target_os = "macos")]
async fn discover_servers_native(
    timeout: Duration,
) -> Result<Vec<mold_server::mdns::DiscoveredServer>, String> {
    use mdns_sd_discovery::{BrowseEvent, ServiceBrowserBuilder};

    let mut builder = ServiceBrowserBuilder::new();
    builder.service_type("_mold._tcp");
    let mut browser = builder.browse().await.map_err(|e| e.to_string())?;
    let deadline = tokio::time::Instant::now() + timeout;
    let mut found = std::collections::BTreeMap::new();

    loop {
        let remaining = deadline.saturating_duration_since(tokio::time::Instant::now());
        if remaining.is_zero() {
            break;
        }
        match tokio::time::timeout(remaining, browser.recv()).await {
            Ok(Some(Ok(BrowseEvent::Found(service)))) => {
                let fullname = format!("{}.{}", service.name, mold_server::mdns::SERVICE_TYPE);
                let txt = service
                    .txt_records
                    .into_iter()
                    .map(|record| {
                        let value = record
                            .value
                            .map(|value| String::from_utf8_lossy(&value).into_owned())
                            .unwrap_or_default();
                        (record.key, value)
                    })
                    .collect();
                let server = mold_server::mdns::from_service_parts(
                    &fullname,
                    &service.host_name,
                    service.port,
                    service.addresses,
                    txt,
                );
                found.insert(service.name, server);
            }
            Ok(Some(Ok(BrowseEvent::Removed(service)))) => {
                found.remove(&service.name);
            }
            Ok(Some(Err(error))) => {
                tracing::debug!(%error, "native DNS-SD service resolution failed");
            }
            Ok(None) | Err(_) => break,
        }
    }

    Ok(found.into_values().collect())
}

#[tauri::command]
pub async fn discover_servers(timeout_ms: Option<u64>) -> Result<Vec<DiscoveredHost>, String> {
    let timeout = Duration::from_millis(timeout_ms.unwrap_or(3000).clamp(500, 10_000));

    #[cfg(target_os = "macos")]
    let servers = discover_servers_native(timeout).await?;

    #[cfg(not(target_os = "macos"))]
    let servers =
        tauri::async_runtime::spawn_blocking(move || mold_server::mdns::discover(timeout))
            .await
            .map_err(|e| e.to_string())?
            .map_err(|e| format!("{e:#}"))?;

    let local_addrs: Vec<std::net::IpAddr> = if_addrs::get_if_addrs()
        .map(|ifs| ifs.into_iter().map(|i| i.ip()).collect())
        .unwrap_or_default();
    let local_hostname = hostname::get()
        .ok()
        .and_then(|h| h.into_string().ok())
        .unwrap_or_default();

    Ok(servers
        .into_iter()
        .map(|s| {
            let is_this_machine =
                host_is_this_machine(&s.addresses, &s.name, &local_addrs, &local_hostname);
            DiscoveredHost {
                name: s.name,
                url: s.url,
                host: s.host,
                port: s.port,
                version: s.version,
                auth_required: s.auth_required,
                is_this_machine,
            }
        })
        .collect())
}

/// Switch to a remote host (validates first) and persist it in app settings,
/// remembering it in the most-recently-used host list.
#[tauri::command]
pub async fn set_remote_host(
    state: tauri::State<'_, AppState>,
    store: tauri::State<'_, SettingsStore>,
    url: String,
    api_key: Option<String>,
    name: Option<String>,
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

    // Keys live in the secret store: the shared slot feeds the current
    // connection, the per-host slot lets saved hosts reconnect later. The
    // legacy plaintext settings slot is wiped so old files converge.
    let id = host_id(&url);
    match api_key.as_deref().filter(|k| !k.is_empty()) {
        Some(key) => {
            state
                .secrets
                .set("remote-api-key", key)
                .map_err(|e| e.to_string())?;
            state
                .secrets
                .set(&format!("remote-api-key.{id}"), key)
                .map_err(|e| e.to_string())?;
        }
        None => {
            state
                .secrets
                .clear("remote-api-key")
                .map_err(|e| e.to_string())?;
            state
                .secrets
                .clear(&format!("remote-api-key.{id}"))
                .map_err(|e| e.to_string())?;
        }
    }
    let now_ms = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_millis() as u64)
        .unwrap_or(0);
    let updated = {
        let mut current = store.current.lock().expect("settings mutex");
        current.mode = ConnectionMode::Remote;
        current.remote_url = Some(url.clone());
        current.remote_api_key = None;
        upsert_saved_host(&mut current.saved_hosts, &id, &url, name, now_ms);
        // The primary also joins the boot-reconnect set: switching to another
        // host (or falling back to the built-in engine) must not make this
        // one vanish on the next launch.
        remember_connected_host(&mut current, &id);
        current.clone()
    };
    settings::save(&store.path, &updated).map_err(|e| e.to_string())?;
    Ok(conn.info(&state.local_api_key))
}

/// Drop a host from the saved list and delete its stored API key. Does not
/// touch the live connection, but if the host is also the persisted primary
/// remote, the remote preference (and its shared key) is cleared too — the
/// next launch must not resurrect a host the user just forgot.
#[tauri::command]
pub async fn forget_remote_host(
    state: tauri::State<'_, AppState>,
    store: tauri::State<'_, SettingsStore>,
    id: String,
) -> Result<Vec<SavedHost>, String> {
    state
        .secrets
        .clear(&format!("remote-api-key.{id}"))
        .map_err(|e| e.to_string())?;
    let (updated, was_primary) = {
        let mut current = store.current.lock().expect("settings mutex");
        let was_primary = settings::forget_host(&mut current, &id);
        (current.clone(), was_primary)
    };
    if was_primary {
        state
            .secrets
            .clear("remote-api-key")
            .map_err(|e| e.to_string())?;
    }
    settings::save(&store.path, &updated).map_err(|e| e.to_string())?;
    Ok(updated.saved_hosts)
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
    fn remote_primary_does_not_own_the_local_server_lifecycle() {
        assert!(!primary_uses_local_server(&Conn::Remote {
            url: "http://hal9000:7680".into(),
            api_key: None,
        }));
        assert!(primary_uses_local_server(&Conn::Local {
            base_url: "http://127.0.0.1:7680".into(),
        }));
        assert!(primary_uses_local_server(&Conn::External {
            base_url: "http://127.0.0.1:7680".into(),
        }));
    }

    #[test]
    fn engine_environment_sets_and_clears_knobs_and_tokens() {
        let dir = tempfile::tempdir().unwrap();
        let secrets = crate::secrets::SecretStore::new(dir.path().to_path_buf());
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

        // Clearing the secret clears the token env on the next engine start —
        // stale credentials must not linger across restarts.
        secrets.clear("hf-token").unwrap();
        apply_engine_environment(&env, &secrets);
        assert!(std::env::var("HF_TOKEN").is_err());
    }

    #[test]
    fn discovered_host_serializes_camel_case() {
        let host = DiscoveredHost {
            name: "hal9000-7680".into(),
            url: "http://192.168.1.10:7680".into(),
            host: "192.168.1.10".into(),
            port: 7680,
            version: Some("0.14.0".into()),
            auth_required: true,
            is_this_machine: false,
        };
        let json = serde_json::to_value(&host).unwrap();
        assert_eq!(json["authRequired"], true);
        assert_eq!(json["isThisMachine"], false);
        assert_eq!(json["url"], "http://192.168.1.10:7680");
        // Snake-case keys must not leak through.
        assert!(json.get("auth_required").is_none());
        assert!(json.get("is_this_machine").is_none());
    }

    #[test]
    fn host_is_this_machine_matches_local_address() {
        let local: Vec<std::net::IpAddr> = vec!["192.168.1.10".parse().unwrap()];
        assert!(host_is_this_machine(
            &["192.168.1.10".into()],
            "somehost-7680",
            &local,
            "unrelated"
        ));
        assert!(!host_is_this_machine(
            &["192.168.1.99".into()],
            "somehost-7680",
            &local,
            "unrelated"
        ));
    }

    #[test]
    fn host_is_this_machine_matches_hostname_prefix() {
        assert!(host_is_this_machine(
            &["10.0.0.5".into()],
            "hal9000-7680",
            &[],
            "hal9000.local"
        ));
        assert!(!host_is_this_machine(
            &["10.0.0.5".into()],
            "box-7680",
            &[],
            "hal9000.local"
        ));
    }

    #[test]
    fn host_is_this_machine_requires_delimiter_and_ignores_case() {
        // `box` must not claim `box2-7680` (shared prefix, different host).
        assert!(!host_is_this_machine(
            &["10.0.0.5".into()],
            "box2-7680",
            &[],
            "box.local"
        ));
        // Case-insensitive match on the short hostname.
        assert!(host_is_this_machine(
            &["10.0.0.5".into()],
            "hal9000-7680",
            &[],
            "HAL9000.local"
        ));
        // Bare instance name equal to the short hostname still matches.
        assert!(host_is_this_machine(
            &["10.0.0.5".into()],
            "hal9000",
            &[],
            "hal9000.local"
        ));
    }
}
