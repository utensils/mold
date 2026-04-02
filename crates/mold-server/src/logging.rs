use std::path::{Path, PathBuf};

use mold_core::LoggingConfig;
use tracing_appender::non_blocking::WorkerGuard;
use tracing_subscriber::prelude::*;
use tracing_subscriber::EnvFilter;

/// Holds the non-blocking file writer guard. Must live for the duration of the program.
pub struct LogGuard {
    _file_guard: Option<WorkerGuard>,
}

/// Initialize tracing with optional file logging.
///
/// - `log_file`: CLI `--log-file` flag (overrides config).
/// - `json`: use JSON format for stderr output (ignored when file logging is active).
/// - `config`: the `[logging]` section from config.toml.
/// - `default_level`: fallback level if neither env var nor config is set.
/// - `log_dir`: resolved log directory path.
pub fn init_tracing(
    log_file: bool,
    json: bool,
    config: &LoggingConfig,
    default_level: &str,
    log_dir: PathBuf,
) -> LogGuard {
    // Priority: MOLD_LOG env > config level > default_level
    let filter_str = std::env::var("MOLD_LOG").unwrap_or_else(|_| {
        if config.level.is_empty() {
            default_level.to_string()
        } else {
            config.level.clone()
        }
    });
    let file_enabled = log_file || config.file;

    let mut file_guard: Option<WorkerGuard> = None;

    if file_enabled {
        let _ = std::fs::create_dir_all(&log_dir);
        cleanup_old_logs(&log_dir, config.max_days);
        let appender = tracing_appender::rolling::daily(&log_dir, "mold-server");
        let (non_blocking, guard) = tracing_appender::non_blocking(appender);
        file_guard = Some(guard);

        let stderr_filter = make_filter(&filter_str, default_level);
        let file_filter = make_filter(&filter_str, default_level);

        // Both layers use text format for type compatibility.
        // File layer disables ANSI escape codes.
        let stderr_layer = tracing_subscriber::fmt::layer().with_writer(std::io::stderr);
        let file_layer = tracing_subscriber::fmt::layer()
            .with_writer(non_blocking)
            .with_ansi(false);

        tracing_subscriber::registry()
            .with(stderr_layer.with_filter(stderr_filter))
            .with(file_layer.with_filter(file_filter))
            .init();
    } else if json {
        let filter = make_filter(&filter_str, default_level);
        tracing_subscriber::fmt()
            .with_env_filter(filter)
            .with_writer(std::io::stderr)
            .json()
            .init();
    } else {
        let filter = make_filter(&filter_str, default_level);
        tracing_subscriber::fmt()
            .with_env_filter(filter)
            .with_writer(std::io::stderr)
            .init();
    }

    LogGuard {
        _file_guard: file_guard,
    }
}

fn make_filter(filter_str: &str, default_level: &str) -> EnvFilter {
    EnvFilter::try_new(filter_str).unwrap_or_else(|_| EnvFilter::new(default_level))
}

/// Delete log files older than `max_days` from the log directory.
fn cleanup_old_logs(log_dir: &Path, max_days: u32) {
    let now = std::time::SystemTime::now();
    let max_age = std::time::Duration::from_secs(max_days as u64 * 86400);

    let entries = match std::fs::read_dir(log_dir) {
        Ok(e) => e,
        Err(_) => return,
    };

    for entry in entries.filter_map(|e| e.ok()) {
        let path = entry.path();
        if !path.is_file() {
            continue;
        }

        let filename = path
            .file_name()
            .and_then(|f| f.to_str())
            .unwrap_or_default();

        if !filename.starts_with("mold-server.") {
            continue;
        }

        if let Ok(metadata) = path.metadata() {
            if let Ok(modified) = metadata.modified() {
                if let Ok(age) = now.duration_since(modified) {
                    if age > max_age {
                        let _ = std::fs::remove_file(&path);
                    }
                }
            }
        }
    }
}
