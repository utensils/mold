use std::path::PathBuf;

use anyhow::Result;

use crate::procinfo;

// ── PID file management ──────────────────────────────────────────────────────

fn pid_file_path() -> Option<PathBuf> {
    mold_core::Config::mold_dir().map(|d| d.join("mold-server.pid"))
}

/// Managed server metadata from PID file.
struct ManagedServer {
    pid: u32,
    port: u16,
    bind: String,
}

impl ManagedServer {
    /// The address to use for health/status probes.
    fn probe_host(&self) -> &str {
        // 0.0.0.0 and :: bind to all interfaces — probe via loopback
        match self.bind.as_str() {
            "0.0.0.0" | "::" | "" => "127.0.0.1",
            "::1" => "::1",
            other => other,
        }
    }

    /// Format host for use in socket addresses — wraps IPv6 in brackets.
    fn socket_host(&self) -> String {
        let host = self.probe_host();
        if host.contains(':') {
            format!("[{host}]")
        } else {
            host.to_string()
        }
    }

    fn base_url(&self) -> String {
        format!("http://{}:{}", self.socket_host(), self.port)
    }
}

/// Read and validate PID file. Returns None if missing, malformed, or stale.
fn read_pid_file() -> Option<ManagedServer> {
    let path = pid_file_path()?;
    let contents = std::fs::read_to_string(&path).ok()?;
    let val: serde_json::Value = serde_json::from_str(&contents).ok()?;
    let pid = val.get("pid")?.as_u64()? as u32;
    let port = val.get("port")?.as_u64()? as u16;
    let bind = val
        .get("bind")
        .and_then(|v| v.as_str())
        .unwrap_or("0.0.0.0")
        .to_string();
    if process_alive(pid) && is_mold_serve_process(pid) {
        Some(ManagedServer { pid, port, bind })
    } else {
        // Stale PID file or PID reused by unrelated process — clean up
        let _ = std::fs::remove_file(&path);
        None
    }
}

fn write_pid_file(pid: u32, port: u16, bind: &str) -> Result<()> {
    let path = match pid_file_path() {
        Some(p) => p,
        None => anyhow::bail!("cannot determine mold home directory"),
    };
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    let json = serde_json::json!({
        "pid": pid,
        "port": port,
        "bind": bind,
    });
    let data = serde_json::to_string_pretty(&json)?;
    // Atomic write: write to .tmp, rename
    let tmp = path.with_extension("pid.tmp");
    std::fs::write(&tmp, &data)?;
    std::fs::rename(&tmp, &path)?;
    Ok(())
}

fn remove_pid_file() {
    if let Some(path) = pid_file_path() {
        let _ = std::fs::remove_file(&path);
    }
}

/// Check if a process is alive.
#[cfg(unix)]
pub fn process_alive(pid: u32) -> bool {
    unsafe { libc::kill(pid as libc::pid_t, 0) == 0 }
}

#[cfg(not(unix))]
pub fn process_alive(_pid: u32) -> bool {
    use sysinfo::{Pid, System};
    let mut sys = System::new();
    sys.refresh_processes(sysinfo::ProcessesToUpdate::All, true);
    sys.process(Pid::from_u32(_pid)).is_some()
}

/// Check if a PID belongs to a mold serve process (not a reused PID).
fn is_mold_serve_process(pid: u32) -> bool {
    let procs = procinfo::find_mold_processes();
    procs
        .iter()
        .any(|p| p.pid == pid && p.subcommand == "serve")
}

/// Check server health via TCP connect + HTTP GET.
fn check_health(host: &str, port: u16) -> bool {
    use std::io::{Read, Write};
    use std::net::{TcpStream, ToSocketAddrs};

    // Wrap IPv6 hosts in brackets for socket address parsing
    let addr_str = if host.contains(':') {
        format!("[{host}]:{port}")
    } else {
        format!("{host}:{port}")
    };
    let Ok(addr) = addr_str
        .to_socket_addrs()
        .map(|mut addrs| addrs.next())
        .ok()
        .flatten()
        .ok_or(())
    else {
        return false;
    };
    let Ok(mut stream) = TcpStream::connect_timeout(&addr, std::time::Duration::from_secs(2))
    else {
        return false;
    };
    stream
        .set_read_timeout(Some(std::time::Duration::from_secs(2)))
        .ok();
    let host_header = if host.contains(':') {
        format!("[{host}]:{port}")
    } else {
        format!("{host}:{port}")
    };
    let req = format!("GET /health HTTP/1.0\r\nHost: {host_header}\r\n\r\n");
    if stream.write_all(req.as_bytes()).is_err() {
        return false;
    }
    let mut buf = [0u8; 64];
    match stream.read(&mut buf) {
        Ok(n) if n > 0 => {
            let resp = String::from_utf8_lossy(&buf[..n]);
            resp.contains("200")
        }
        _ => false,
    }
}

/// Check if a child process has exited.
fn child_exited(child: &mut std::process::Child) -> bool {
    matches!(child.try_wait(), Ok(Some(_)))
}

// ── Commands ─────────────────────────────────────────────────────────────────

pub async fn run_start(
    port: u16,
    bind: &str,
    models_dir: Option<String>,
    log_file: bool,
    #[cfg(feature = "mdns")] no_mdns: bool,
) -> Result<()> {
    // Check for existing managed server
    if let Some(srv) = read_pid_file() {
        eprintln!(
            "Server already running (PID {} on port {})",
            srv.pid, srv.port
        );
        std::process::exit(1);
    }

    let exe = std::env::current_exe()?;
    let port_str = port.to_string();
    let mut args = vec!["serve".to_string(), "--port".to_string(), port_str.clone()];
    args.extend(["--bind".to_string(), bind.to_string()]);
    if let Some(ref dir) = models_dir {
        args.extend(["--models-dir".to_string(), dir.clone()]);
    }
    if log_file {
        args.push("--log-file".to_string());
    }
    #[cfg(feature = "mdns")]
    if no_mdns {
        args.push("--no-mdns".to_string());
    }

    let mut cmd = std::process::Command::new(&exe);
    cmd.args(&args)
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::null());

    // On Unix, call setsid() so the child survives terminal close
    #[cfg(unix)]
    {
        use std::os::unix::process::CommandExt;
        unsafe {
            cmd.pre_exec(|| {
                libc::setsid();
                Ok(())
            });
        }
    }

    let mut child = cmd
        .spawn()
        .map_err(|e| anyhow::anyhow!("failed to start server: {e}"))?;
    let pid = child.id();

    write_pid_file(pid, port, bind)?;

    eprint!("Starting server (PID {pid}) on port {port}...");

    // Poll for health, checking if the child died early (port conflict, etc.)
    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(15);
    let mut healthy = false;
    while std::time::Instant::now() < deadline {
        if child_exited(&mut child) {
            eprintln!(" failed");
            remove_pid_file();
            anyhow::bail!(
                "server exited immediately — port {port} may already be in use. \
                 Check logs at ~/.mold/logs/"
            );
        }
        let probe = match bind {
            "0.0.0.0" | "::" | "" => "127.0.0.1",
            "::1" => "::1",
            other => other,
        };
        if check_health(probe, port) {
            healthy = true;
            break;
        }
        std::thread::sleep(std::time::Duration::from_millis(250));
    }

    if healthy {
        eprintln!(" ready");
        eprintln!("  PID:  {pid}");
        eprintln!("  Port: {port}");
        eprintln!("  Logs: ~/.mold/logs/");
        eprintln!("  Stop: mold server stop");
    } else {
        eprintln!(" timeout (server may still be loading models)");
        eprintln!("  PID:  {pid}");
        eprintln!("  Stop: mold server stop");
    }

    Ok(())
}

pub async fn run_status() -> Result<()> {
    match read_pid_file() {
        Some(srv) => {
            let client = mold_core::MoldClient::new(&srv.base_url());
            match client.server_status().await {
                Ok(status) => {
                    eprintln!("Server running (PID {})", srv.pid);
                    eprintln!("  Version: {}", status.version);
                    eprintln!("  Port:    {}", srv.port);
                    eprintln!("  Uptime:  {}s", status.uptime_secs);
                    eprintln!(
                        "  Models:  {}",
                        if status.models_loaded.is_empty() {
                            "none".to_string()
                        } else {
                            status.models_loaded.join(", ")
                        }
                    );
                    if let Some(gpu) = &status.gpu_info {
                        eprintln!("  GPU:     {}", gpu.name);
                        eprintln!("  VRAM:    {}/{}MB", gpu.vram_used_mb, gpu.vram_total_mb);
                    }
                    if status.busy {
                        eprintln!("  Status:  busy (generating)");
                    }
                }
                Err(_) => {
                    eprintln!(
                        "Server process running (PID {}) but not responding on port {}",
                        srv.pid, srv.port
                    );
                }
            }
        }
        None => {
            // Check for unmanaged mold processes
            let procs = procinfo::find_mold_processes();
            let serve_procs: Vec<_> = procs.iter().filter(|p| p.subcommand == "serve").collect();
            if serve_procs.is_empty() {
                eprintln!("No server running");
            } else {
                eprintln!("No managed server found, but detected unmanaged mold processes:");
                for p in &serve_procs {
                    eprintln!(
                        "  PID {} — mold serve {} ({:.0}s)",
                        p.pid,
                        p.args.join(" "),
                        p.run_time_secs
                    );
                }
                eprintln!("\nThese were not started with 'mold server start'.");
            }
            std::process::exit(1);
        }
    }
    Ok(())
}

pub async fn run_stop() -> Result<()> {
    let srv = match read_pid_file() {
        Some(s) => s,
        None => {
            eprintln!("No managed server running");
            std::process::exit(1);
        }
    };
    let pid = srv.pid;

    eprint!("Stopping server (PID {pid})...");

    // Try graceful HTTP shutdown first
    let client = mold_core::MoldClient::new(&srv.base_url());
    let http_ok = tokio::time::timeout(std::time::Duration::from_secs(5), client.shutdown_server())
        .await
        .map(|r| r.is_ok())
        .unwrap_or(false);

    if http_ok {
        // Wait for process to exit
        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(10);
        while std::time::Instant::now() < deadline && process_alive(pid) {
            std::thread::sleep(std::time::Duration::from_millis(250));
        }
    }

    // Fallback: SIGTERM
    if process_alive(pid) {
        #[cfg(unix)]
        unsafe {
            libc::kill(pid as libc::pid_t, libc::SIGTERM);
        }

        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(10);
        while std::time::Instant::now() < deadline && process_alive(pid) {
            std::thread::sleep(std::time::Duration::from_millis(250));
        }
    }

    // Last resort: SIGKILL
    if process_alive(pid) {
        #[cfg(unix)]
        unsafe {
            libc::kill(pid as libc::pid_t, libc::SIGKILL);
        }
        std::thread::sleep(std::time::Duration::from_millis(500));
    }

    remove_pid_file();
    eprintln!(" stopped");
    Ok(())
}

// ── mDNS discovery ───────────────────────────────────────────────────────────

/// One row of the discovery table: a discovered server plus optional probe
/// latency in milliseconds (`None` when `--probe` is off or the probe failed).
#[cfg(feature = "mdns")]
pub struct DiscoverRow {
    pub server: mold_server::mdns::DiscoveredServer,
    pub latency_ms: Option<u64>,
}

/// Render the discovery results as an aligned table. Pure so it is unit-tested
/// without touching the network. `show_latency` adds the LATENCY column.
#[cfg(feature = "mdns")]
pub fn render_table(rows: &[DiscoverRow], show_latency: bool) -> String {
    use std::fmt::Write as _;

    if rows.is_empty() {
        return "No mold servers found on the local network.".to_string();
    }

    let mut headers = vec!["NAME", "URL", "VERSION", "AUTH", "GPU"];
    if show_latency {
        headers.push("LATENCY");
    }

    // Build each row's cells as owned strings.
    let mut cells: Vec<Vec<String>> = Vec::with_capacity(rows.len());
    for row in rows {
        let s = &row.server;
        let mut cols = vec![
            s.name.clone(),
            s.url.clone(),
            s.version.clone().unwrap_or_else(|| "?".to_string()),
            if s.auth_required { "key" } else { "-" }.to_string(),
            s.txt.get("gpu").cloned().unwrap_or_else(|| "-".to_string()),
        ];
        if show_latency {
            cols.push(match row.latency_ms {
                Some(ms) => format!("{ms}ms"),
                None => "-".to_string(),
            });
        }
        cells.push(cols);
    }

    // Column widths = max of header and any cell.
    let mut widths: Vec<usize> = headers.iter().map(|h| h.len()).collect();
    for cols in &cells {
        for (i, c) in cols.iter().enumerate() {
            widths[i] = widths[i].max(c.chars().count());
        }
    }

    let mut out = String::new();
    for (i, h) in headers.iter().enumerate() {
        let pad = widths[i] - h.len();
        let _ = write!(out, "{h}{}", " ".repeat(pad));
        if i + 1 < headers.len() {
            out.push_str("  ");
        }
    }
    out.push('\n');
    for cols in &cells {
        for (i, c) in cols.iter().enumerate() {
            let pad = widths[i] - c.chars().count();
            let _ = write!(out, "{c}{}", " ".repeat(pad));
            if i + 1 < cols.len() {
                out.push_str("  ");
            }
        }
        out.push('\n');
    }
    out
}

/// `mold server discover` — browse the LAN for `_mold._tcp` advertisements.
#[cfg(feature = "mdns")]
pub async fn run_discover(timeout_secs: u64, json: bool, probe: bool) -> Result<()> {
    let timeout = std::time::Duration::from_secs(timeout_secs.max(1));
    // The mdns browse is blocking; keep the async runtime free while it runs.
    let servers =
        tokio::task::spawn_blocking(move || mold_server::mdns::discover(timeout)).await??;

    // Optionally probe each server's /health for a rough latency signal.
    let mut rows: Vec<DiscoverRow> = Vec::with_capacity(servers.len());
    for server in servers {
        let latency_ms = if probe {
            probe_latency_ms(&server.url).await
        } else {
            None
        };
        rows.push(DiscoverRow { server, latency_ms });
    }

    if json {
        let payload: Vec<&mold_server::mdns::DiscoveredServer> =
            rows.iter().map(|r| &r.server).collect();
        println!("{}", serde_json::to_string_pretty(&payload)?);
        return Ok(());
    }

    print!("{}", render_table(&rows, probe));

    if let Some(first) = rows.first() {
        println!();
        println!("Connect: export MOLD_HOST={}", first.server.url);
    }
    Ok(())
}

/// Best-effort latency probe: time a GET to `/health`, falling back to
/// `/api/status` (a 401 there still confirms a mold server). Caps at ~2s.
#[cfg(feature = "mdns")]
async fn probe_latency_ms(base_url: &str) -> Option<u64> {
    let client = reqwest::Client::builder()
        .timeout(std::time::Duration::from_secs(2))
        .build()
        .ok()?;
    let start = std::time::Instant::now();
    // /health is auth-exempt; if it 404s on an old build, /api/status confirms
    // mold even behind auth (401 is a positive signal).
    for path in ["/health", "/api/status"] {
        if let Ok(resp) = client.get(format!("{base_url}{path}")).send().await {
            let ok = resp.status().is_success() || resp.status().as_u16() == 401;
            if ok {
                return Some(start.elapsed().as_millis() as u64);
            }
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pid_file_roundtrip() {
        let dir = std::env::temp_dir().join(format!("mold-server-test-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();

        let path = dir.join("mold-server.pid");
        let json = serde_json::json!({"pid": 12345, "port": 7680});
        std::fs::write(&path, serde_json::to_string_pretty(&json).unwrap()).unwrap();

        let contents = std::fs::read_to_string(&path).unwrap();
        let val: serde_json::Value = serde_json::from_str(&contents).unwrap();
        assert_eq!(val["pid"], 12345);
        assert_eq!(val["port"], 7680);

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn malformed_pid_file_returns_none() {
        let val: Option<serde_json::Value> = serde_json::from_str("not json").ok();
        assert!(val.is_none());
    }

    #[test]
    fn stale_pid_returns_false() {
        assert!(!process_alive(999_999_999));
    }

    #[test]
    fn process_alive_self() {
        assert!(process_alive(std::process::id()));
    }

    #[test]
    fn check_health_no_server() {
        // Port 1 should never have a server
        assert!(!check_health("127.0.0.1", 1));
    }

    #[test]
    fn check_health_ipv6_no_server() {
        // IPv6 loopback on port 1 should not panic
        assert!(!check_health("::1", 1));
    }

    #[test]
    fn managed_server_probe_host_ipv4_wildcard() {
        let srv = ManagedServer {
            pid: 1,
            port: 7680,
            bind: "0.0.0.0".to_string(),
        };
        assert_eq!(srv.probe_host(), "127.0.0.1");
    }

    #[test]
    fn managed_server_probe_host_ipv6_wildcard() {
        let srv = ManagedServer {
            pid: 1,
            port: 7680,
            bind: "::".to_string(),
        };
        assert_eq!(srv.probe_host(), "127.0.0.1");
    }

    #[test]
    fn managed_server_probe_host_ipv6_loopback() {
        let srv = ManagedServer {
            pid: 1,
            port: 7680,
            bind: "::1".to_string(),
        };
        assert_eq!(srv.probe_host(), "::1");
    }

    #[test]
    fn managed_server_base_url_ipv6() {
        let srv = ManagedServer {
            pid: 1,
            port: 7680,
            bind: "::1".to_string(),
        };
        assert_eq!(srv.base_url(), "http://[::1]:7680");
    }

    #[test]
    fn managed_server_base_url_ipv4() {
        let srv = ManagedServer {
            pid: 1,
            port: 8080,
            bind: "192.168.1.10".to_string(),
        };
        assert_eq!(srv.base_url(), "http://192.168.1.10:8080");
    }

    #[test]
    fn is_mold_serve_process_bogus_pid() {
        // A non-existent PID should not be a mold serve process
        assert!(!is_mold_serve_process(999_999_999));
    }

    #[cfg(feature = "mdns")]
    mod discover {
        use super::*;
        use mold_server::mdns::DiscoveredServer;
        use std::collections::BTreeMap;

        fn sample(name: &str, url: &str, version: Option<&str>, auth: bool) -> DiscoveredServer {
            let mut txt = BTreeMap::new();
            txt.insert("gpu".to_string(), "1xRTX 4090".to_string());
            DiscoveredServer {
                name: name.to_string(),
                host: "192.168.1.10".to_string(),
                addresses: vec!["192.168.1.10".to_string()],
                port: 7680,
                url: url.to_string(),
                version: version.map(String::from),
                auth_required: auth,
                instance_id: None,
                txt,
            }
        }

        #[test]
        fn empty_result_message() {
            assert_eq!(
                render_table(&[], false),
                "No mold servers found on the local network."
            );
        }

        #[test]
        fn table_has_header_and_aligned_columns() {
            let rows = vec![
                DiscoverRow {
                    server: sample(
                        "hal9000-7680",
                        "http://192.168.1.10:7680",
                        Some("0.14.0"),
                        true,
                    ),
                    latency_ms: None,
                },
                DiscoverRow {
                    server: sample(
                        "box-7681",
                        "http://192.168.1.11:7681",
                        Some("0.14.0"),
                        false,
                    ),
                    latency_ms: None,
                },
            ];
            let out = render_table(&rows, false);
            let lines: Vec<&str> = out.lines().collect();
            assert!(lines[0].starts_with("NAME"));
            assert!(lines[0].contains("AUTH"));
            assert!(lines[0].contains("GPU"));
            assert!(!lines[0].contains("LATENCY"));
            // AUTH badge: "key" when required, "-" otherwise.
            assert!(lines[1].contains("key"));
            assert!(lines[2].contains(" - "));
            // Columns line up: NAME column width matches the longest name.
            assert!(lines[1].starts_with("hal9000-7680"));
        }

        #[test]
        fn table_latency_column_when_probing() {
            let rows = vec![DiscoverRow {
                server: sample(
                    "hal9000-7680",
                    "http://192.168.1.10:7680",
                    Some("0.14.0"),
                    false,
                ),
                latency_ms: Some(12),
            }];
            let out = render_table(&rows, true);
            assert!(out.lines().next().unwrap().contains("LATENCY"));
            assert!(out.contains("12ms"));
        }

        #[test]
        fn table_missing_version_renders_placeholder() {
            let rows = vec![DiscoverRow {
                server: sample("box-7680", "http://192.168.1.11:7680", None, false),
                latency_ms: None,
            }];
            let out = render_table(&rows, false);
            assert!(out.contains('?'));
        }

        #[test]
        fn json_shape_is_array_of_servers() {
            let servers = vec![sample(
                "hal9000-7680",
                "http://192.168.1.10:7680",
                Some("0.14.0"),
                true,
            )];
            let json = serde_json::to_string(&servers).unwrap();
            let parsed: serde_json::Value = serde_json::from_str(&json).unwrap();
            assert!(parsed.is_array());
            assert_eq!(parsed[0]["name"], "hal9000-7680");
            assert_eq!(parsed[0]["auth_required"], true);
            assert_eq!(parsed[0]["url"], "http://192.168.1.10:7680");
        }
    }
}
