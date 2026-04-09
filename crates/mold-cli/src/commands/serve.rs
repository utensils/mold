use anyhow::Result;
use mold_core::types::GpuSelection;
use mold_core::Config;
use std::path::PathBuf;

use crate::theme;

#[cfg(feature = "discord")]
/// Resolve a listen address to a client-reachable loopback address.
///
/// Wildcard listen addresses (`0.0.0.0`, `::`, `[::]`) are valid for binding
/// but not for connecting. Replace them with the corresponding loopback.
/// IPv6 addresses are wrapped in brackets for use in HTTP URLs.
fn client_host(bind: &str, port: u16) -> String {
    let host = match bind {
        "0.0.0.0" => "127.0.0.1".to_string(),
        "::" | "[::]" => "[::1]".to_string(),
        addr if addr.contains(':') && !addr.starts_with('[') => {
            // Bare IPv6 address — wrap in brackets for URL
            format!("[{addr}]")
        }
        addr => addr.to_string(),
    };
    format!("http://{host}:{port}")
}

pub async fn run(
    port: u16,
    bind: &str,
    models_dir: Option<String>,
    gpus: Option<String>,
    queue_size: usize,
    discord: bool,
) -> Result<()> {
    let config = Config::load_or_default();

    let models_path = match models_dir {
        Some(dir) => PathBuf::from(dir),
        None => config.resolved_models_dir(),
    };

    // Ensure models directory exists
    std::fs::create_dir_all(&models_path)?;

    // Resolve GPU selection: CLI flag > env var > config > default (all).
    let gpu_selection = match &gpus {
        Some(s) => GpuSelection::parse(s)?,
        None => config.gpu_selection(),
    };

    println!(
        "{} Starting mold server on {}:{}",
        theme::icon_ok(),
        bind,
        port,
    );
    println!(
        "{} Models directory: {}",
        theme::icon_ok(),
        models_path.display(),
    );
    match &gpu_selection {
        GpuSelection::All => {
            println!("{} GPUs: all available", theme::icon_ok());
        }
        GpuSelection::Specific(ordinals) => {
            let list: Vec<String> = ordinals.iter().map(|o| o.to_string()).collect();
            println!("{} GPUs: {}", theme::icon_ok(), list.join(", "));
        }
    }
    println!("{} Queue size: {}", theme::icon_ok(), queue_size);

    // Optionally spawn the Discord bot alongside the server.
    #[cfg(feature = "discord")]
    if discord {
        // Resolve the client URL for the bot to reach this server.
        // Set MOLD_HOST before spawning any tasks to avoid thread-safety
        // issues with std::env::set_var in a multi-threaded runtime.
        let mold_host = std::env::var("MOLD_HOST").unwrap_or_else(|_| client_host(bind, port));
        // SAFETY: set_var is called here on the main thread before the
        // discord bot task is spawned, so there is no data race.
        unsafe { std::env::set_var("MOLD_HOST", &mold_host) };

        println!(
            "{} Discord bot enabled (connecting to {})",
            theme::icon_ok(),
            mold_host,
        );

        // Use a oneshot channel so the bot can signal startup failure.
        let (startup_tx, startup_rx) = tokio::sync::oneshot::channel::<Result<(), String>>();

        tokio::spawn(async move {
            // Brief delay so the HTTP listener is ready before the bot connects.
            tokio::time::sleep(std::time::Duration::from_secs(1)).await;
            match mold_discord::run().await {
                Ok(()) => {
                    let _ = startup_tx.send(Ok(()));
                }
                Err(e) => {
                    let msg = format!("{e:#}");
                    let _ = startup_tx.send(Err(msg.clone()));
                    eprintln!("{} Discord bot failed: {msg}", crate::theme::prefix_error());
                }
            }
        });

        // Give the bot a few seconds to either connect or fail fast
        // (e.g. missing token). If it's still starting, proceed — the
        // server shouldn't block on a slow Discord gateway handshake.
        match tokio::time::timeout(std::time::Duration::from_secs(5), startup_rx).await {
            Ok(Ok(Err(e))) => {
                // Bot reported a startup error — abort the combined mode.
                anyhow::bail!("Discord bot failed to start: {e}");
            }
            Ok(Err(_)) => {
                // Channel dropped — bot task panicked.
                anyhow::bail!("Discord bot task panicked during startup");
            }
            // Ok(Ok(Ok(()))) — bot started successfully (unlikely within 5s,
            // since run() blocks on the gateway). Timeout is the normal path.
            _ => {}
        }
    }
    #[cfg(not(feature = "discord"))]
    if discord {
        anyhow::bail!(
            "Discord support is not compiled in. Rebuild with --features discord to enable."
        );
    }

    mold_server::run_server(bind, port, models_path, gpu_selection, queue_size).await
}

#[cfg(all(test, feature = "discord"))]
mod tests {
    use super::client_host;

    #[test]
    fn wildcard_ipv4_resolves_to_loopback() {
        assert_eq!(client_host("0.0.0.0", 7680), "http://127.0.0.1:7680");
    }

    #[test]
    fn wildcard_ipv6_resolves_to_loopback() {
        assert_eq!(client_host("::", 7680), "http://[::1]:7680");
        assert_eq!(client_host("[::]", 7680), "http://[::1]:7680");
    }

    #[test]
    fn bare_ipv6_gets_brackets() {
        assert_eq!(client_host("::1", 7680), "http://[::1]:7680");
        assert_eq!(client_host("fe80::1", 8080), "http://[fe80::1]:8080");
    }

    #[test]
    fn bracketed_ipv6_unchanged() {
        assert_eq!(client_host("[::1]", 7680), "http://[::1]:7680");
    }

    #[test]
    fn normal_ipv4_unchanged() {
        assert_eq!(client_host("127.0.0.1", 7680), "http://127.0.0.1:7680");
        assert_eq!(client_host("10.0.0.5", 8080), "http://10.0.0.5:8080");
    }

    #[test]
    fn hostname_unchanged() {
        assert_eq!(client_host("localhost", 7680), "http://localhost:7680");
    }
}
