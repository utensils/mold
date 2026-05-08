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
            // Register a no-op subscriber so tracing macros are silently
            // discarded rather than leaving the global subscriber unset.
            let _ = tracing::subscriber::set_global_default(tracing_subscriber::registry());
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

pub(crate) fn resolve_filter(config: &LoggingConfig, default_level: &str) -> String {
    std::env::var("MOLD_LOG").unwrap_or_else(|_| {
        if config.level.is_empty() {
            default_level.to_string()
        } else {
            config.level.clone()
        }
    })
}

pub(crate) fn make_filter(filter_str: &str, default_level: &str) -> EnvFilter {
    EnvFilter::try_new(filter_str).unwrap_or_else(|_| EnvFilter::new(default_level))
}

/// Delete log files older than `max_days` from the log directory.
pub(crate) fn cleanup_old_logs(log_dir: &Path, max_days: u32) {
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

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use std::time::{Duration, SystemTime};

    // Process-global lock so the three resolve_filter cases below don't
    // race each other on `MOLD_LOG` when cargo test runs them in parallel.
    // Single bundled test walks the env→config→default precedence chain;
    // splitting into separate tests would expose the env-var mutation to
    // interleaving inside the same process.
    #[test]
    fn resolve_filter_precedence_chain() {
        let prev = std::env::var("MOLD_LOG").ok();

        // 1. MOLD_LOG env var wins over both config and default.
        std::env::set_var("MOLD_LOG", "trace");
        let cfg = LoggingConfig {
            level: "warn".to_string(),
            file: false,
            dir: None,
            max_days: 7,
        };
        assert_eq!(resolve_filter(&cfg, "info"), "trace");

        // 2. With env unset, config.level is used.
        std::env::remove_var("MOLD_LOG");
        let cfg = LoggingConfig {
            level: "debug".to_string(),
            file: false,
            dir: None,
            max_days: 7,
        };
        assert_eq!(resolve_filter(&cfg, "info"), "debug");

        // 3. With env unset and config.level empty, the default fires.
        let cfg = LoggingConfig {
            level: String::new(),
            file: false,
            dir: None,
            max_days: 7,
        };
        assert_eq!(resolve_filter(&cfg, "info"), "info");

        match prev {
            Some(v) => std::env::set_var("MOLD_LOG", v),
            None => std::env::remove_var("MOLD_LOG"),
        }
    }

    #[test]
    fn make_filter_accepts_both_valid_and_invalid_input_without_panicking() {
        // Valid directive — try_new succeeds, no fallback.
        let _ = make_filter("mold=trace,hyper=warn", "info");
        // EnvFilter is forgiving, but truly malformed input still fails
        // try_new — exercising the unwrap_or_else branch.
        let _ = make_filter("@@@@@@", "warn");
        let _ = make_filter("invalid=garbage=syntax", "info");
    }

    #[test]
    fn cleanup_old_logs_deletes_aged_mold_files_and_keeps_fresh_ones() {
        let dir = tempfile::tempdir().expect("tempdir");
        let dir_path = dir.path();

        let aged = dir_path.join("mold-server.2020-01-01");
        let aged_with_suffix = dir_path.join("mold-tui.2020-01-01.log");
        let fresh = dir_path.join("mold-server.fresh.log");
        let unrelated = dir_path.join("not-our-business.log");

        for path in [&aged, &aged_with_suffix, &fresh, &unrelated] {
            fs::write(path, b"x").expect("write fixture");
        }

        // Backdate the two "old" files so they exceed max_days.
        let ancient = SystemTime::now() - Duration::from_secs(60 * 60 * 24 * 30); // 30 days ago
        for path in [&aged, &aged_with_suffix] {
            let f = fs::File::open(path).expect("open fixture");
            f.set_modified(ancient).expect("backdate fixture");
        }

        cleanup_old_logs(dir_path, 7);

        assert!(!aged.exists(), "30-day-old mold-server log must be removed");
        assert!(
            !aged_with_suffix.exists(),
            "30-day-old mold-tui log must be removed",
        );
        assert!(fresh.exists(), "fresh mold-server log must be retained");
        assert!(
            unrelated.exists(),
            "unrelated *.log file must be left untouched",
        );
    }

    #[test]
    fn cleanup_old_logs_returns_silently_for_missing_dir() {
        let dir = tempfile::tempdir().expect("tempdir");
        let missing = dir.path().join("does-not-exist");
        // Should not panic — read_dir errors are absorbed by an early return.
        cleanup_old_logs(&missing, 1);
    }

    #[test]
    fn cleanup_old_logs_skips_subdirectories() {
        let dir = tempfile::tempdir().expect("tempdir");
        let dir_path = dir.path();
        let subdir = dir_path.join("mold-server.subdir");
        fs::create_dir(&subdir).expect("create subdir");
        // Backdate so the age check would otherwise match.
        let ancient = SystemTime::now() - Duration::from_secs(60 * 60 * 24 * 30);
        let f = fs::File::open(&subdir).expect("open subdir");
        // set_modified on a directory may fail on some filesystems — best-effort.
        let _ = f.set_modified(ancient);

        cleanup_old_logs(dir_path, 7);
        assert!(
            subdir.exists(),
            "subdirectories must be skipped even when they pattern-match",
        );
    }
}
