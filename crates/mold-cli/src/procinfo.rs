//! Detect running mold processes via OS-level process inspection.

use std::collections::HashMap;
use std::ffi::OsStr;
use sysinfo::System;

/// Known mold subcommands — used to distinguish our `mold` binary from
/// the GNU `mold` linker or other unrelated processes with the same name.
const KNOWN_SUBCOMMANDS: &[&str] = &[
    "run",
    "serve",
    "pull",
    "list",
    "ls", // alias for list
    "info",
    "rm",
    "remove", // alias for rm
    "unload",
    "default",
    "expand",
    "version",
    "completions",
];

/// A detected mold process (deduplicated — threads are counted, not listed).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MoldProcess {
    pub pid: u32,
    pub subcommand: String,
    pub args: Vec<String>,
    pub run_time_secs: u64,
    pub memory_bytes: u64,
    pub thread_count: usize,
}

/// Scan the system for running mold processes (excluding the current PID).
///
/// Threads share their parent's cmdline on Linux, so we deduplicate by
/// command line — each unique cmdline becomes one entry with a thread count.
pub fn find_mold_processes() -> Vec<MoldProcess> {
    let current_pid = std::process::id();
    let mut sys = System::new();
    sys.refresh_processes_specifics(
        sysinfo::ProcessesToUpdate::All,
        false,
        sysinfo::ProcessRefreshKind::nothing()
            .with_cmd(sysinfo::UpdateKind::OnlyIfNotSet)
            .with_exe(sysinfo::UpdateKind::OnlyIfNotSet)
            .with_memory(),
    );

    // Collect raw matches, keyed by cmdline for dedup.
    #[allow(clippy::type_complexity)]
    let mut groups: HashMap<Vec<String>, (u32, String, Vec<String>, u64, u64, usize)> =
        HashMap::new();

    for (pid, process) in sys.processes() {
        let pid_u32 = pid.as_u32();
        if pid_u32 == current_pid {
            continue;
        }

        if !is_mold_process(process.name(), process.exe()) {
            continue;
        }

        let cmd: Vec<String> = process
            .cmd()
            .iter()
            .map(|s| s.to_string_lossy().into_owned())
            .collect();

        if cmd.is_empty() {
            continue;
        }

        let (subcommand, args) = parse_mold_cmd(&cmd);

        // Filter: recognized subcommand, not "ps" (that's us).
        if !KNOWN_SUBCOMMANDS.contains(&subcommand.as_str()) || subcommand == "ps" {
            continue;
        }

        let run_time = process.run_time();
        let memory = process.memory();

        groups
            .entry(cmd)
            .and_modify(|(lowest_pid, _, _, max_time, max_mem, count)| {
                if pid_u32 < *lowest_pid {
                    *lowest_pid = pid_u32;
                }
                if run_time > *max_time {
                    *max_time = run_time;
                }
                if memory > *max_mem {
                    *max_mem = memory;
                }
                *count += 1;
            })
            .or_insert((pid_u32, subcommand, args, run_time, memory, 1));
    }

    let mut results: Vec<MoldProcess> = groups
        .into_values()
        .map(
            |(pid, subcommand, args, run_time, memory, count)| MoldProcess {
                pid,
                subcommand,
                args,
                run_time_secs: run_time,
                memory_bytes: memory,
                thread_count: count,
            },
        )
        .collect();

    results.sort_by_key(|p| p.pid);
    results
}

/// Check if a process is a mold binary by name or exe path.
fn is_mold_process(name: &OsStr, exe: Option<&std::path::Path>) -> bool {
    let name_str = name.to_string_lossy();
    if name_str == "mold" {
        return true;
    }
    if let Some(path) = exe {
        if let Some(file_name) = path.file_name() {
            return file_name.to_string_lossy() == "mold";
        }
    }
    false
}

/// Parse the command-line args to extract the subcommand and its arguments.
fn parse_mold_cmd(cmd: &[String]) -> (String, Vec<String>) {
    // Skip argv[0] (binary path), then find the first non-flag token as subcommand
    let rest: Vec<&String> = cmd.iter().skip(1).collect();
    for (i, arg) in rest.iter().enumerate() {
        if !arg.starts_with('-') {
            let subcommand = (*arg).clone();
            let args: Vec<String> = rest[i + 1..].iter().map(|s| (*s).clone()).collect();
            return (subcommand, args);
        }
    }
    ("unknown".to_string(), Vec::new())
}

/// Format a duration in seconds to a human-readable string.
pub fn format_duration(secs: u64) -> String {
    if secs < 60 {
        format!("{secs}s")
    } else if secs < 3600 {
        format!("{}m {}s", secs / 60, secs % 60)
    } else {
        format!("{}h {}m", secs / 3600, (secs % 3600) / 60)
    }
}

/// Format bytes to a human-readable MB string.
pub fn format_memory_mb(bytes: u64) -> String {
    let mb = bytes / (1024 * 1024);
    if mb == 0 && bytes > 0 {
        return "< 1 MB".to_string();
    }
    format!("{mb} MB")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_cmd_run_with_model() {
        let cmd = vec![
            "/nix/store/xxx/bin/mold".into(),
            "run".into(),
            "flux-schnell:q8".into(),
            "a cat".into(),
            "--local".into(),
        ];
        let (sub, args) = parse_mold_cmd(&cmd);
        assert_eq!(sub, "run");
        assert_eq!(args, vec!["flux-schnell:q8", "a cat", "--local"]);
    }

    #[test]
    fn parse_cmd_serve() {
        let cmd = vec![
            "mold".into(),
            "serve".into(),
            "--port".into(),
            "7680".into(),
        ];
        let (sub, args) = parse_mold_cmd(&cmd);
        assert_eq!(sub, "serve");
        assert_eq!(args, vec!["--port", "7680"]);
    }

    #[test]
    fn parse_cmd_no_args() {
        let cmd = vec!["mold".into()];
        let (sub, args) = parse_mold_cmd(&cmd);
        assert_eq!(sub, "unknown");
        assert!(args.is_empty());
    }

    #[test]
    fn parse_cmd_with_leading_flags() {
        let cmd = vec![
            "mold".into(),
            "--verbose".into(),
            "run".into(),
            "a cat".into(),
        ];
        let (sub, args) = parse_mold_cmd(&cmd);
        assert_eq!(sub, "run");
        assert_eq!(args, vec!["a cat"]);
    }

    #[test]
    fn format_duration_seconds() {
        assert_eq!(format_duration(42), "42s");
    }

    #[test]
    fn format_duration_minutes() {
        assert_eq!(format_duration(125), "2m 5s");
    }

    #[test]
    fn format_duration_hours() {
        assert_eq!(format_duration(7320), "2h 2m");
    }

    #[test]
    fn format_memory_mb_basic() {
        assert_eq!(format_memory_mb(512 * 1024 * 1024), "512 MB");
    }

    #[test]
    fn format_memory_mb_sub_mb() {
        assert_eq!(format_memory_mb(500_000), "< 1 MB");
        assert_eq!(format_memory_mb(0), "0 MB");
    }

    #[test]
    fn is_mold_by_name() {
        assert!(is_mold_process(OsStr::new("mold"), None));
        assert!(!is_mold_process(OsStr::new("cargo"), None));
        assert!(!is_mold_process(OsStr::new("moldy"), None));
    }

    #[test]
    fn is_mold_by_exe_path() {
        use std::path::Path;
        assert!(is_mold_process(
            OsStr::new("mold-wrapped"),
            Some(Path::new("/nix/store/xxx/bin/mold"))
        ));
        assert!(!is_mold_process(
            OsStr::new("cargo"),
            Some(Path::new("/usr/bin/cargo"))
        ));
    }

    #[test]
    fn find_mold_processes_excludes_self() {
        let procs = find_mold_processes();
        let self_pid = std::process::id();
        assert!(
            procs.iter().all(|p| p.pid != self_pid),
            "should exclude current process"
        );
    }

    #[test]
    fn gnu_mold_linker_filtered_out() {
        let cmd = vec![
            "/usr/bin/mold".into(),
            "-run".into(),
            "gcc".into(),
            "main.c".into(),
        ];
        let (sub, _) = parse_mold_cmd(&cmd);
        assert!(!KNOWN_SUBCOMMANDS.contains(&sub.as_str()));
    }

    #[test]
    fn discord_not_in_known_subcommands() {
        // "discord" was removed to avoid matching unrelated Discord processes.
        assert!(!KNOWN_SUBCOMMANDS.contains(&"discord"));
    }

    #[test]
    fn known_subcommands_detected() {
        for &cmd_name in KNOWN_SUBCOMMANDS {
            let cmd = vec!["mold".into(), cmd_name.into()];
            let (sub, _) = parse_mold_cmd(&cmd);
            assert_eq!(sub, cmd_name);
            assert!(KNOWN_SUBCOMMANDS.contains(&sub.as_str()));
        }
    }
}
