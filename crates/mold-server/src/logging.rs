use std::path::{Path, PathBuf};

use mold_core::LoggingConfig;
use tracing_appender::non_blocking::WorkerGuard;
use tracing_subscriber::prelude::*;
use tracing_subscriber::EnvFilter;

/// Holds the non-blocking file writer guard. Must live for the duration of the program.
pub struct LogGuard {
    _file_guard: Option<WorkerGuard>,
}

/// Initialize tracing with optional file logging and stderr output.
pub fn init_tracing(
    log_file: bool,
    json: bool,
    config: &LoggingConfig,
    default_level: &str,
    log_dir: PathBuf,
) -> LogGuard {
    let filter_str = resolve_filter(config, default_level);
    let file_enabled = log_file || config.file;

    let mut file_guard: Option<WorkerGuard> = None;

    if file_enabled {
        let _ = std::fs::create_dir_all(&log_dir);
        cleanup_old_logs(&log_dir, config.max_days);
        let appender = match tracing_appender::rolling::RollingFileAppender::builder()
            .rotation(tracing_appender::rolling::Rotation::DAILY)
            .filename_prefix("mold-server")
            .filename_suffix("log")
            .build(&log_dir)
        {
            Ok(a) => a,
            Err(e) => {
                eprintln!("warning: failed to create log file appender: {e}");
                eprintln!("warning: file logging disabled, continuing with stderr only");
                let filter = make_filter(&filter_str, default_level);
                tracing_subscriber::fmt()
                    .with_env_filter(filter)
                    .with_writer(std::io::stderr)
                    .init();
                return LogGuard { _file_guard: None };
            }
        };
        let (non_blocking, guard) = tracing_appender::non_blocking(appender);
        file_guard = Some(guard);

        let stderr_filter = make_filter(&filter_str, default_level);
        let file_filter = make_filter(&filter_str, default_level);

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

/// Initialize tracing with file-only output (no stderr).
///
/// Used by the TUI which owns the terminal and cannot have tracing
/// output on stderr.
pub fn init_tracing_file_only(
    config: &LoggingConfig,
    default_level: &str,
    log_dir: PathBuf,
) -> LogGuard {
    let filter_str = resolve_filter(config, default_level);

    let _ = std::fs::create_dir_all(&log_dir);
    cleanup_old_logs(&log_dir, config.max_days);
    let appender = match tracing_appender::rolling::RollingFileAppender::builder()
        .rotation(tracing_appender::rolling::Rotation::DAILY)
        .filename_prefix("mold-tui")
        .filename_suffix("log")
        .build(&log_dir)
    {
        Ok(a) => a,
        Err(e) => {
            eprintln!("warning: failed to create TUI log file appender: {e}");
            return LogGuard { _file_guard: None };
        }
    };
    let (non_blocking, guard) = tracing_appender::non_blocking(appender);

    let filter = make_filter(&filter_str, default_level);
    tracing_subscriber::fmt()
        .with_env_filter(filter)
        .with_writer(non_blocking)
        .with_ansi(false)
        .init();

    LogGuard {
        _file_guard: Some(guard),
    }
}

fn resolve_filter(config: &LoggingConfig, default_level: &str) -> String {
    std::env::var("MOLD_LOG").unwrap_or_else(|_| {
        if config.level.is_empty() {
            default_level.to_string()
        } else {
            config.level.clone()
        }
    })
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

        // Match both old (no .log) and new (.log suffix) patterns
        let is_mold_log = filename.starts_with("mold-server.") || filename.starts_with("mold-tui.");
        if !is_mold_log {
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
