//! Embedded mold engine: runs `mold-ai-server` in-process on a dedicated
//! thread with its own tokio runtime. The webview talks plain HTTP + SSE to
//! `127.0.0.1:<port>` — the identical wire contract used for remote hosts.

use std::path::PathBuf;
use std::time::Duration;

use mold_core::types::GpuSelection;

pub const DEFAULT_QUEUE_SIZE: usize = 200;
/// Port a user-run `mold serve` listens on by default.
pub const WELL_KNOWN_PORT: u16 = 7680;
pub const LAN_BIND: &str = "0.0.0.0";

pub struct EngineHandle {
    pub port: u16,
    pub models_dir: PathBuf,
    thread: Option<std::thread::JoinHandle<()>>,
}

impl EngineHandle {
    pub fn base_url(&self) -> String {
        format!("http://127.0.0.1:{}", self.port)
    }

    /// False once the engine thread has exited (crash, shutdown, or panic) —
    /// the connection state machine restarts a dead engine instead of
    /// handing out a base URL nothing listens on.
    pub fn is_alive(&self) -> bool {
        self.thread.as_ref().is_some_and(|t| !t.is_finished())
    }

    /// Join the engine thread after shutdown has been requested over HTTP.
    pub fn join(mut self, timeout: Duration) {
        if let Some(thread) = self.thread.take() {
            let deadline = std::time::Instant::now() + timeout;
            while !thread.is_finished() && std::time::Instant::now() < deadline {
                std::thread::sleep(Duration::from_millis(50));
            }
            if thread.is_finished() {
                let _ = thread.join();
            } else {
                tracing::warn!("embedded engine did not stop within {timeout:?}; detaching");
            }
        }
    }
}

/// Reserve a port on `bind` by binding to :0 and reading the assignment.
///
/// TOCTOU caveat: the listener is dropped before `run_server` rebinds it.
/// Upstream fix tracked as U1 (`run_server_with_listener`) in
/// desktop/docs/architecture.md.
pub fn allocate_port(bind: &str) -> anyhow::Result<u16> {
    let probe = std::net::TcpListener::bind((bind, 0))?;
    Ok(probe.local_addr()?.port())
}

/// Prefer the conventional LAN port. If something other than Mold owns it,
/// fall back to an ephemeral port that mDNS will advertise.
pub fn available_server_port(bind: &str) -> anyhow::Result<u16> {
    available_server_port_for(bind, WELL_KNOWN_PORT)
}

fn available_server_port_for(bind: &str, preferred: u16) -> anyhow::Result<u16> {
    match std::net::TcpListener::bind((bind, preferred)) {
        Ok(listener) => {
            drop(listener);
            Ok(preferred)
        }
        Err(_) => allocate_port(bind),
    }
}

fn queue_size() -> usize {
    std::env::var("MOLD_QUEUE_SIZE")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(DEFAULT_QUEUE_SIZE)
}

/// Spawn the embedded engine. The API key must already be exported as
/// `MOLD_API_KEY` (done once at app startup, before any threads exist).
pub fn start_engine(
    bind: &str,
    port: u16,
    models_dir: PathBuf,
    gpu_selection: GpuSelection,
) -> anyhow::Result<EngineHandle> {
    std::fs::create_dir_all(&models_dir)?;
    let bind = bind.to_string();
    let dir = models_dir.clone();
    let size = queue_size();
    let thread = std::thread::Builder::new()
        .name("mold-engine".into())
        .spawn(move || {
            let rt = tokio::runtime::Builder::new_multi_thread()
                .enable_all()
                .thread_name("mold-engine-worker")
                .build()
                .expect("engine tokio runtime");
            match rt.block_on(mold_server::run_server(
                &bind,
                port,
                dir,
                gpu_selection,
                size,
            )) {
                Ok(()) => {
                    // Also to stderr: the file subscriber may not be active.
                    eprintln!("embedded mold engine stopped (clean shutdown)");
                    tracing::info!("embedded mold engine stopped");
                }
                Err(e) => {
                    eprintln!("embedded mold engine exited with error: {e:#}");
                    tracing::error!("embedded mold engine exited: {e:#}");
                }
            }
        })?;
    Ok(EngineHandle {
        port,
        models_dir,
        thread: Some(thread),
    })
}

/// Poll `base_url` until /health answers 200 or the deadline passes.
pub async fn wait_healthy(base_url: &str, timeout: Duration) -> bool {
    let client = reqwest::Client::new();
    let deadline = tokio::time::Instant::now() + timeout;
    let url = format!("{base_url}/health");
    while tokio::time::Instant::now() < deadline {
        if matches!(
            client
                .get(&url)
                .timeout(Duration::from_secs(2))
                .send()
                .await,
            Ok(resp) if resp.status().is_success()
        ) {
            return true;
        }
        tokio::time::sleep(Duration::from_millis(100)).await;
    }
    false
}

/// True when `base_url` serves mold's OpenAPI document (auth-exempt, unlike
/// /api/capabilities) — distinguishes a mold server from some other process.
pub async fn is_mold_server(base_url: &str) -> bool {
    let url = format!("{base_url}/api/openapi.json");
    match reqwest::Client::new()
        .get(&url)
        .timeout(Duration::from_secs(2))
        .send()
        .await
    {
        Ok(resp) if resp.status().is_success() => resp
            .json::<serde_json::Value>()
            .await
            .map(|v| {
                v.pointer("/info/title")
                    .and_then(|t| t.as_str())
                    .is_some_and(|t| t.to_ascii_lowercase().contains("mold"))
            })
            .unwrap_or(false),
        _ => false,
    }
}

/// True when the server accepts the API key the desktop will hand to its
/// local HTTP clients. Unauthenticated servers also return success here when
/// a key is supplied, while a server owned by another key returns 401.
pub async fn accepts_api_key(base_url: &str, api_key: &str) -> bool {
    matches!(
        reqwest::Client::new()
            .get(format!("{base_url}/api/status"))
            .header("X-Api-Key", api_key)
            .timeout(Duration::from_secs(2))
            .send()
            .await,
        Ok(resp) if resp.status().is_success()
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn allocates_distinct_nonzero_ports() {
        let a = allocate_port("127.0.0.1").unwrap();
        let b = allocate_port("127.0.0.1").unwrap();
        assert_ne!(a, 0);
        assert_ne!(b, 0);
        // Not guaranteed distinct by the OS, but freshly-released ephemeral
        // ports are not immediately reused on macOS/Linux.
        assert_ne!(a, b);
    }

    #[test]
    fn prefers_well_known_port_when_available() {
        let preferred = allocate_port("127.0.0.1").unwrap();
        let port = available_server_port_for("127.0.0.1", preferred).unwrap();
        assert_eq!(port, preferred);
    }

    #[test]
    fn falls_back_when_well_known_port_is_occupied() {
        let listener = std::net::TcpListener::bind(("127.0.0.1", 0)).unwrap();
        let occupied = listener.local_addr().unwrap().port();
        let port = available_server_port_for("127.0.0.1", occupied).unwrap();
        assert_ne!(port, occupied);
        assert_ne!(port, 0);
        drop(listener);
    }

    #[test]
    fn queue_size_defaults_and_honors_env() {
        std::env::remove_var("MOLD_QUEUE_SIZE");
        assert_eq!(queue_size(), DEFAULT_QUEUE_SIZE);
        std::env::set_var("MOLD_QUEUE_SIZE", "17");
        assert_eq!(queue_size(), 17);
        std::env::remove_var("MOLD_QUEUE_SIZE");
    }
}
