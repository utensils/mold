//! External watchdog core for the H3 Metal default-resolution campaign.
//!
//! The `h3_metal_campaign_watch` binary (available only with `dev-bins`) is a
//! thin front end for this module. It supervises ONE campaign attempt from
//! OUTSIDE the GPU process, enforcing the campaign document's
//! host-level gates that the in-process guard cannot enforce against itself:
//! a 12 GiB available floor, a 256 MiB attempt swap-growth limit, normal
//! kernel pressure, a per-case wall-clock deadline, and fail-closed behavior
//! on a missing or stale sample. A violation kills the child's whole process
//! group, verifies no descendant survives, and records the refusal.
//!
//! It also owns the coordinated `/tmp/mold-metal-qualification.lock` for the
//! attempt and injects the campaign environment into the child, so the
//! watchdog limits and the in-process guard limits are configured at exactly
//! one place. Sampling is deliberately independent of the child: the
//! in-process capture (`campaign_capture`) records evidence rows, this
//! process decides.

use std::io::Write;
use std::os::unix::process::CommandExt;
use std::path::PathBuf;
use std::process::Command;
use std::sync::LazyLock;
use std::time::{Duration, Instant};

use anyhow::{anyhow, bail, Context, Result};

pub const H3_METAL_CAMPAIGN_WATCH_SCHEMA: &str = "mold.minimax-h3.metal-campaign-watch.v1";

const SAMPLE_INTERVAL: Duration = Duration::from_millis(250);
const DESCENDANT_SETTLE: Duration = Duration::from_secs(2);

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Sample {
    /// `None` marks a failed sample: the watchdog fails closed on it.
    pub available_bytes: Option<u64>,
    pub swap_used_bytes: Option<u64>,
    pub kernel_pressure: Option<u32>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct WatchdogLimits {
    pub available_floor_bytes: u64,
    pub maximum_swap_growth_bytes: u64,
    pub deadline: Duration,
}

impl WatchdogLimits {
    /// The campaign limits. The swap baseline is the FIRST successful sample,
    /// taken before the child does meaningful work; growth beyond it is what
    /// the attempt is charged for.
    pub fn violation(
        &self,
        baseline_swap_bytes: Option<u64>,
        sample: Sample,
        elapsed: Duration,
    ) -> Option<String> {
        let (Some(available), Some(swap), Some(pressure)) = (
            sample.available_bytes,
            sample.swap_used_bytes,
            sample.kernel_pressure,
        ) else {
            return Some(
                "campaign watchdog stopped the attempt because a host memory sample failed".into(),
            );
        };
        if elapsed > self.deadline {
            return Some(format!(
                "campaign watchdog stopped the attempt after {:.0}s, past the {:.0}s deadline",
                elapsed.as_secs_f32(),
                self.deadline.as_secs_f32(),
            ));
        }
        if available < self.available_floor_bytes {
            return Some(format!(
                "campaign watchdog stopped the attempt: {:.1} GiB available is below the {:.1} GiB floor",
                available as f64 / (1_u64 << 30) as f64,
                self.available_floor_bytes as f64 / (1_u64 << 30) as f64,
            ));
        }
        if let Some(baseline) = baseline_swap_bytes {
            let growth = swap.saturating_sub(baseline);
            if growth > self.maximum_swap_growth_bytes {
                return Some(format!(
                    "campaign watchdog stopped the attempt: swap grew by {:.1} MiB, above the {:.0} MiB limit",
                    growth as f64 / (1 << 20) as f64,
                    self.maximum_swap_growth_bytes as f64 / (1 << 20) as f64,
                ));
            }
        }
        if pressure != 1 {
            return Some(format!(
                "campaign watchdog stopped the attempt: kernel memory pressure level is {pressure}, not normal"
            ));
        }
        None
    }
}

fn real_sample() -> Sample {
    Sample {
        available_bytes: crate::device::available_system_memory_bytes(),
        swap_used_bytes: crate::device::used_system_swap_bytes(),
        kernel_pressure: crate::minimax_h3::campaign_capture::kernel_pressure_level().ok(),
    }
}

static STARTED: LazyLock<Instant> = LazyLock::new(Instant::now);

fn monotonic_ns() -> u64 {
    STARTED.elapsed().as_nanos().min(u64::MAX as u128) as u64
}

struct EventLog {
    file: Option<std::io::BufWriter<std::fs::File>>,
    pub case_id: String,
}

impl EventLog {
    fn create_new(path: &std::path::Path, case_id: &str) -> Result<Self> {
        let file = std::fs::OpenOptions::new()
            .create_new(true)
            .write(true)
            .open(path)
            .with_context(|| {
                format!(
                    "campaign watchdog could not open its event log {}",
                    path.display()
                )
            })?;
        Ok(Self {
            file: Some(std::io::BufWriter::new(file)),
            case_id: case_id.to_string(),
        })
    }

    fn record(&mut self, event: &str, detail: &str, sample: Sample) {
        let Some(file) = self.file.as_mut() else {
            return;
        };
        let row = serde_json::json!({
            "schema": H3_METAL_CAMPAIGN_WATCH_SCHEMA,
            "case_id": self.case_id,
            "monotonic_ns": monotonic_ns(),
            "event": event,
            "detail": detail,
            "available_bytes": sample.available_bytes,
            "swap_used_bytes": sample.swap_used_bytes,
            "kernel_pressure": sample.kernel_pressure,
        });
        if let Ok(line) = serde_json::to_string(&row) {
            let _ = writeln!(file, "{line}").and_then(|()| file.flush());
        }
    }
}

/// Take the coordinated campaign lock exclusively and without blocking. A
/// held lock refuses the launch: the document serializes GPU work through
/// this file and a stale lock that belongs to a live process is never
/// cleared silently.
fn acquire_lock(path: &std::path::Path, case_id: &str) -> Result<std::fs::File> {
    use std::os::unix::io::AsRawFd;
    let mut file = std::fs::OpenOptions::new()
        .create(true)
        .write(true)
        .truncate(false)
        .open(path)
        .with_context(|| {
            format!(
                "campaign watchdog could not open the lock {}",
                path.display()
            )
        })?;
    // SAFETY: `flock` on a live fd with LOCK_EX | LOCK_NB is the standard
    // non-blocking exclusive advisory lock; the fd outlives the call.
    let status = unsafe { libc::flock(file.as_raw_fd(), libc::LOCK_EX | libc::LOCK_NB) };
    if status != 0 {
        bail!(
            "campaign watchdog refuses to launch: the coordinated lock {} is held by another task",
            path.display()
        );
    }
    file.set_len(0)?;
    writeln!(file, "{case_id} pid={}", std::process::id())?;
    file.flush()?;
    Ok(file)
}

/// Verify the owned process group has no surviving member, waiting a bounded
/// time for kernels to finish reaping SIGKILLed descendants.
fn process_group_is_empty(pgid: i32) -> bool {
    let deadline = Instant::now() + DESCENDANT_SETTLE;
    loop {
        // SAFETY: kill with signal 0 performs an existence check only.
        let status = unsafe { libc::kill(-pgid, 0) };
        if status == -1 && std::io::Error::last_os_error().raw_os_error() == Some(libc::ESRCH) {
            return true;
        }
        if Instant::now() >= deadline {
            return false;
        }
        std::thread::sleep(Duration::from_millis(25));
    }
}

fn kill_process_group(pgid: i32) {
    // SAFETY: SIGKILL to a negative pgid targets the whole group; the group
    // is the one this watchdog created for its child.
    unsafe {
        libc::kill(-pgid, libc::SIGKILL);
    }
}

pub struct ParsedArgs {
    pub case_id: String,
    pub capture_path: PathBuf,
    pub log_path: PathBuf,
    pub lock_path: PathBuf,
    pub ceiling_mib: u64,
    pub limits: WatchdogLimits,
    pub child: Vec<String>,
}

pub fn parse_args(args: &[String]) -> Result<ParsedArgs> {
    let mut case_id = None;
    let mut capture_path = None;
    let mut log_path = None;
    let mut lock_path = PathBuf::from("/tmp/mold-metal-qualification.lock");
    let mut ceiling_mib = None;
    let mut available_floor_gib = 12_u64;
    let mut max_swap_growth_mib = 256_u64;
    let mut deadline_secs = None;
    let mut child = Vec::new();
    let mut in_child = false;
    let mut index = 0;
    while index < args.len() {
        let arg = &args[index];
        if in_child {
            child.push(arg.clone());
            index += 1;
            continue;
        }
        let mut take_value = |name: &str| -> Result<String> {
            index += 1;
            args.get(index)
                .cloned()
                .ok_or_else(|| anyhow!("{name} requires a value"))
        };
        if arg == "--" {
            in_child = true;
        } else if let Some(value) = arg.strip_prefix("--case-id=") {
            case_id = Some(value.to_string());
        } else if arg == "--case-id" {
            case_id = Some(take_value("--case-id")?);
        } else if let Some(value) = arg.strip_prefix("--capture-path=") {
            capture_path = Some(PathBuf::from(value));
        } else if arg == "--capture-path" {
            capture_path = Some(PathBuf::from(take_value("--capture-path")?));
        } else if let Some(value) = arg.strip_prefix("--log=") {
            log_path = Some(PathBuf::from(value));
        } else if arg == "--log" {
            log_path = Some(PathBuf::from(take_value("--log")?));
        } else if let Some(value) = arg.strip_prefix("--lock=") {
            lock_path = PathBuf::from(value);
        } else if arg == "--lock" {
            lock_path = PathBuf::from(take_value("--lock")?);
        } else if let Some(value) = arg.strip_prefix("--ceiling-mib=") {
            ceiling_mib = Some(value.parse().context("--ceiling-mib is not an integer")?);
        } else if arg == "--ceiling-mib" {
            ceiling_mib = Some(
                take_value("--ceiling-mib")?
                    .parse()
                    .context("--ceiling-mib is not an integer")?,
            );
        } else if let Some(value) = arg.strip_prefix("--available-floor-gib=") {
            available_floor_gib = value
                .parse()
                .context("--available-floor-gib is not an integer")?;
        } else if arg == "--available-floor-gib" {
            available_floor_gib = take_value("--available-floor-gib")?
                .parse()
                .context("--available-floor-gib is not an integer")?;
        } else if let Some(value) = arg.strip_prefix("--max-swap-growth-mib=") {
            max_swap_growth_mib = value
                .parse()
                .context("--max-swap-growth-mib is not an integer")?;
        } else if arg == "--max-swap-growth-mib" {
            max_swap_growth_mib = take_value("--max-swap-growth-mib")?
                .parse()
                .context("--max-swap-growth-mib is not an integer")?;
        } else if let Some(value) = arg.strip_prefix("--deadline-secs=") {
            deadline_secs = Some(value.parse().context("--deadline-secs is not an integer")?);
        } else if arg == "--deadline-secs" {
            deadline_secs = Some(
                take_value("--deadline-secs")?
                    .parse()
                    .context("--deadline-secs is not an integer")?,
            );
        } else {
            bail!("unrecognized campaign watchdog argument {arg:?}");
        }
        index += 1;
    }
    if child.is_empty() {
        bail!("campaign watchdog requires a child command after --");
    }
    let ceiling_mib =
        ceiling_mib.ok_or_else(|| anyhow!("campaign watchdog requires --ceiling-mib"))?;
    if ceiling_mib == 0 {
        bail!("--ceiling-mib must be positive");
    }
    let deadline_secs =
        deadline_secs.ok_or_else(|| anyhow!("campaign watchdog requires --deadline-secs"))?;
    if deadline_secs == 0 {
        bail!("--deadline-secs must be positive");
    }
    Ok(ParsedArgs {
        case_id: case_id.ok_or_else(|| anyhow!("campaign watchdog requires --case-id"))?,
        capture_path: capture_path
            .ok_or_else(|| anyhow!("campaign watchdog requires --capture-path"))?,
        log_path: log_path.ok_or_else(|| anyhow!("campaign watchdog requires --log"))?,
        lock_path,
        ceiling_mib,
        limits: WatchdogLimits {
            available_floor_bytes: available_floor_gib << 30,
            maximum_swap_growth_bytes: max_swap_growth_mib << 20,
            deadline: Duration::from_secs(deadline_secs),
        },
        child,
    })
}

pub fn run(parsed: ParsedArgs) -> Result<i32> {
    if parsed.capture_path.exists() {
        bail!(
            "campaign watchdog refuses to launch: capture evidence {} already exists",
            parsed.capture_path.display()
        );
    }
    let mut log = EventLog::create_new(&parsed.log_path, &parsed.case_id)?;
    let initial = real_sample();
    if let Some(message) = parsed.limits.violation(None, initial, Duration::ZERO) {
        log.record("refused", &message, initial);
        bail!("campaign watchdog refused the launch: {message}");
    }
    let _lock = acquire_lock(&parsed.lock_path, &parsed.case_id)?;

    let (command, args) = parsed.child.split_first().expect("child argv is nonempty");
    let mut child_command = Command::new(command);
    child_command
        .args(args)
        .env("MOLD_H3_METAL_CAMPAIGN", "1")
        .env("MOLD_H3_METAL_CAMPAIGN_CAPTURE", &parsed.capture_path)
        .env("MOLD_H3_METAL_CAMPAIGN_CASE_ID", &parsed.case_id)
        .env(
            "MOLD_H3_METAL_CAMPAIGN_CEILING_MB",
            parsed.ceiling_mib.to_string(),
        )
        .env_remove("MOLD_H3_METAL_CAMPAIGN_BUDGET_ONLY")
        .process_group(0);
    let mut child = child_command.spawn().with_context(|| {
        format!(
            "campaign watchdog could not spawn {}",
            parsed.child.join(" ")
        )
    })?;
    let pgid = child.id() as i32;
    log.record(
        "start",
        &format!("child pid {} argv {:?}", child.id(), parsed.child),
        initial,
    );

    // The swap baseline is the first sample AFTER the child exists, so the
    // attempt is charged only for swap it causes.
    let mut baseline_swap = real_sample().swap_used_bytes;
    let started = Instant::now();
    let outcome = loop {
        std::thread::sleep(SAMPLE_INTERVAL);
        if let Some(status) = child.try_wait()? {
            break Ok(status);
        }
        let sample = real_sample();
        if baseline_swap.is_none() {
            baseline_swap = sample.swap_used_bytes;
        }
        if let Some(message) = parsed
            .limits
            .violation(baseline_swap, sample, started.elapsed())
        {
            break Err(message);
        }
        log.record("sample", "attempt in progress", sample);
    };

    let exit_code = match outcome {
        Ok(status) => {
            let code = status.code().unwrap_or(-1);
            log.record("child_exit", &format!("exit code {code}"), real_sample());
            code
        }
        Err(message) => {
            log.record("violation", &message, real_sample());
            kill_process_group(pgid);
            let _ = child.wait();
            2
        }
    };
    if process_group_is_empty(pgid) {
        log.record(
            "cleanup",
            "the owned process group has no descendants",
            real_sample(),
        );
    } else {
        log.record(
            "cleanup_failed",
            "descendants survived the kill",
            real_sample(),
        );
    }
    log.record(
        "released",
        "the coordinated lock is released",
        real_sample(),
    );
    Ok(exit_code)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn gib(value: u64) -> u64 {
        value << 30
    }

    fn sample(available: Option<u64>, swap: Option<u64>, pressure: Option<u32>) -> Sample {
        Sample {
            available_bytes: available,
            swap_used_bytes: swap,
            kernel_pressure: pressure,
        }
    }

    fn limits(deadline_secs: u64) -> WatchdogLimits {
        WatchdogLimits {
            available_floor_bytes: gib(12),
            maximum_swap_growth_bytes: 256 << 20,
            deadline: Duration::from_secs(deadline_secs),
        }
    }

    #[test]
    fn violation_covers_every_fail_closed_gate() {
        let limits = limits(600);
        // A failed or partial sample refuses regardless of the healthy fields.
        assert!(limits
            .violation(
                Some(gib(1)),
                sample(None, Some(gib(1)), Some(1)),
                Duration::ZERO
            )
            .unwrap()
            .contains("sample failed"));
        assert!(limits
            .violation(
                Some(gib(1)),
                sample(Some(gib(20)), None, Some(1)),
                Duration::ZERO
            )
            .unwrap()
            .contains("sample failed"));
        assert!(limits
            .violation(
                Some(gib(1)),
                sample(Some(gib(20)), Some(gib(1)), None),
                Duration::ZERO
            )
            .unwrap()
            .contains("sample failed"));
        // Floor breach.
        assert!(limits
            .violation(
                Some(gib(1)),
                sample(Some(gib(11)), Some(gib(1)), Some(1)),
                Duration::ZERO
            )
            .unwrap()
            .contains("below the"));
        // Swap growth over the attempt baseline.
        assert!(limits
            .violation(
                Some(gib(1)),
                sample(Some(gib(20)), Some(gib(1) + (300 << 20)), Some(1)),
                Duration::ZERO
            )
            .unwrap()
            .contains("swap grew"));
        // Non-normal pressure.
        assert!(limits
            .violation(
                Some(gib(1)),
                sample(Some(gib(20)), Some(gib(1)), Some(2)),
                Duration::ZERO
            )
            .unwrap()
            .contains("not normal"));
        // Deadline.
        assert!(limits
            .violation(
                Some(gib(1)),
                sample(Some(gib(20)), Some(gib(1)), Some(1)),
                Duration::from_secs(601)
            )
            .unwrap()
            .contains("deadline"));
        // A healthy sample within the baseline and deadline passes.
        assert_eq!(
            limits.violation(
                Some(gib(1)),
                sample(Some(gib(20)), Some(gib(1) + (64 << 20)), Some(1)),
                Duration::from_secs(30)
            ),
            None
        );
    }

    #[test]
    fn violation_without_a_baseline_yet_tolerates_swap() {
        // Before the post-spawn baseline exists, swap is only observed.
        let limits = limits(600);
        assert_eq!(
            limits.violation(
                None,
                sample(Some(gib(20)), Some(gib(30)), Some(1)),
                Duration::ZERO
            ),
            None
        );
    }

    #[test]
    fn watchdog_kills_a_runaway_child_and_verifies_cleanup() {
        let dir = std::env::temp_dir().join(format!(
            "h3-campaign-watch-{}-{}",
            std::process::id(),
            monotonic_ns()
        ));
        std::fs::create_dir_all(&dir).expect("test directory");
        let parsed = ParsedArgs {
            case_id: "watch-test".into(),
            capture_path: dir.join("rows.jsonl"),
            log_path: dir.join("watch.jsonl"),
            lock_path: dir.join("lock"),
            ceiling_mib: 8192,
            limits: WatchdogLimits {
                available_floor_bytes: gib(12),
                maximum_swap_growth_bytes: 256 << 20,
                // A one-second deadline trips the run loop against the REAL
                // child long before `sleep 60` could exit.
                deadline: Duration::from_secs(1),
            },
            child: vec!["sleep".into(), "60".into()],
        };
        let code = run(parsed).expect("the watchdog run completes");
        assert_eq!(code, 2, "a violated attempt exits with the refusal code");
        let log = std::fs::read_to_string(dir.join("watch.jsonl")).expect("event log");
        assert!(log.contains("\"violation\""));
        assert!(log.contains("deadline"));
        assert!(log.contains("\"cleanup\""));
        assert!(log.contains("the owned process group has no descendants"));
        assert!(!log.contains("cleanup_failed"));
        std::fs::remove_dir_all(dir).ok();
    }

    #[test]
    fn watchdog_refuses_an_existing_capture_file() {
        let dir = std::env::temp_dir().join(format!(
            "h3-campaign-watch-refuse-{}-{}",
            std::process::id(),
            monotonic_ns()
        ));
        std::fs::create_dir_all(&dir).expect("test directory");
        std::fs::write(dir.join("rows.jsonl"), "stale").expect("stale evidence");
        let parsed = ParsedArgs {
            case_id: "watch-refuse".into(),
            capture_path: dir.join("rows.jsonl"),
            log_path: dir.join("watch.jsonl"),
            lock_path: dir.join("lock"),
            ceiling_mib: 8192,
            limits: limits(600),
            child: vec!["true".into()],
        };
        let error = run(parsed).expect_err("existing evidence refuses the launch");
        assert!(error.to_string().contains("already exists"));
        std::fs::remove_dir_all(dir).ok();
    }

    #[test]
    fn parse_args_requires_the_campaign_configuration() {
        let child = ["--".to_string(), "true".to_string()].to_vec();
        assert!(parse_args(&child).is_err(), "no case id");
        let with_case = [
            "--case-id".to_string(),
            "a".to_string(),
            "--".to_string(),
            "true".to_string(),
        ]
        .to_vec();
        assert!(parse_args(&with_case).is_err(), "no capture path");
        let full = [
            "--case-id".to_string(),
            "a".to_string(),
            "--capture-path".to_string(),
            "/tmp/rows.jsonl".to_string(),
            "--log".to_string(),
            "/tmp/watch.jsonl".to_string(),
            "--ceiling-mib".to_string(),
            "8192".to_string(),
            "--deadline-secs".to_string(),
            "600".to_string(),
            "--".to_string(),
            "true".to_string(),
        ]
        .to_vec();
        let parsed = parse_args(&full).expect("complete arguments parse");
        assert_eq!(parsed.case_id, "a");
        assert_eq!(parsed.limits.available_floor_bytes, gib(12));
        assert_eq!(parsed.limits.maximum_swap_growth_bytes, 256 << 20);
        assert_eq!(parsed.child, ["true"]);
    }
}
