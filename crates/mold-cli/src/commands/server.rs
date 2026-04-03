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
            other => other,
        }
    }

    fn base_url(&self) -> String {
        format!("http://{}:{}", self.probe_host(), self.port)
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
    if process_alive(pid) {
        Some(ManagedServer { pid, port, bind })
    } else {
        // Stale PID file — clean up
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

/// Check server health via TCP connect + HTTP GET.
fn check_health(host: &str, port: u16) -> bool {
    use std::io::{Read, Write};
    use std::net::TcpStream;
    let addr = format!("{host}:{port}");
    let Ok(mut stream) =
        TcpStream::connect_timeout(&addr.parse().unwrap(), std::time::Duration::from_secs(2))
    else {
        return false;
    };
    stream
        .set_read_timeout(Some(std::time::Duration::from_secs(2)))
        .ok();
    let req = format!("GET /health HTTP/1.0\r\nHost: {host}:{port}\r\n\r\n");
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
}
