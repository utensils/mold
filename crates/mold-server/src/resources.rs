//! Always-on VRAM + system-RAM telemetry aggregator.
//!
//! A single `tokio::spawn`ed task builds a `ResourceSnapshot` every 1 s and
//! broadcasts it through `ResourceBroadcaster`. The HTTP layer in
//! `routes.rs` exposes both a one-shot `GET /api/resources` endpoint (reads
//! the most recently published snapshot) and an SSE stream
//! `GET /api/resources/stream` that replays the broadcast channel.

use mold_core::ResourceSnapshot;
use std::sync::{Arc, Mutex};
use std::time::{Duration, SystemTime, UNIX_EPOCH};
use tokio::sync::broadcast;
use tokio::task::JoinHandle;

/// Broadcast buffer size. Per spec 2.3 — small because downstream consumers
/// only care about the latest tick and lagging receivers (slow SSE clients)
/// recover by reading the `latest` cache on reconnect.
const BROADCAST_BUFFER: usize = 4;

/// Wraps a `tokio::sync::broadcast::Sender<ResourceSnapshot>` and a
/// `Mutex<Option<ResourceSnapshot>>` that caches the most recently published
/// snapshot for the REST endpoint and for new subscribers that connect
/// between ticks.
#[derive(Clone)]
pub struct ResourceBroadcaster {
    tx: broadcast::Sender<ResourceSnapshot>,
    latest: Arc<Mutex<Option<ResourceSnapshot>>>,
}

impl ResourceBroadcaster {
    pub fn new() -> Arc<Self> {
        let (tx, _rx) = broadcast::channel(BROADCAST_BUFFER);
        Arc::new(Self {
            tx,
            latest: Arc::new(Mutex::new(None)),
        })
    }

    /// Publish a new snapshot. Failures (no subscribers yet) are deliberately
    /// ignored — the cache still updates, so the next `GET /api/resources`
    /// call will see it.
    pub fn publish(&self, snapshot: ResourceSnapshot) {
        // Cache first, then fan out. The critical section is a pointer write
        // so a `std::sync::Mutex` is the right primitive — no async scheduler
        // overhead, no silent-drop-on-contention from `try_lock`.
        *self.latest.lock().expect("resource cache mutex poisoned") = Some(snapshot.clone());
        let _ = self.tx.send(snapshot);
    }

    pub fn subscribe(&self) -> broadcast::Receiver<ResourceSnapshot> {
        self.tx.subscribe()
    }

    /// Returns the most recent published snapshot. Used by `GET /api/resources`.
    pub fn latest(&self) -> Option<ResourceSnapshot> {
        self.latest
            .lock()
            .expect("resource cache mutex poisoned")
            .clone()
    }
}

#[cfg(feature = "nvml")]
pub(crate) mod nvml_source {
    use mold_core::{GpuBackend, GpuSnapshot};
    use nvml_wrapper::enums::device::UsedGpuMemory;
    use nvml_wrapper::Nvml;

    pub struct NvmlSource {
        nvml: Nvml,
    }

    impl NvmlSource {
        pub fn try_new() -> anyhow::Result<Self> {
            let nvml = Nvml::init()?;
            Ok(Self { nvml })
        }

        /// Produce a per-GPU snapshot. `pid` is `std::process::id()` of this
        /// server process; we filter `running_compute_processes()` against it
        /// to attribute `vram_used_by_mold`.
        pub fn snapshot(&self, pid: u32) -> Vec<GpuSnapshot> {
            let count = match self.nvml.device_count() {
                Ok(c) => c,
                Err(e) => {
                    tracing::debug!(err = %e, "NVML device_count failed");
                    return Vec::new();
                }
            };
            let mut out = Vec::with_capacity(count as usize);
            for ordinal in 0..count {
                let Ok(dev) = self.nvml.device_by_index(ordinal) else {
                    continue;
                };
                let name = dev
                    .name()
                    .unwrap_or_else(|_| format!("CUDA Device {ordinal}"));
                let mem = match dev.memory_info() {
                    Ok(m) => m,
                    Err(e) => {
                        tracing::debug!(ordinal, err = %e, "NVML memory_info failed");
                        continue;
                    }
                };
                let used_by_mold = dev.running_compute_processes().ok().map(|procs| {
                    procs
                        .iter()
                        .filter(|p| p.pid == pid)
                        .map(|p| match p.used_gpu_memory {
                            UsedGpuMemory::Used(b) => b,
                            UsedGpuMemory::Unavailable => 0,
                        })
                        .sum::<u64>()
                });
                let used_by_other = used_by_mold.map(|m| mem.used.saturating_sub(m));
                out.push(GpuSnapshot {
                    ordinal: ordinal as usize,
                    name,
                    backend: GpuBackend::Cuda,
                    vram_total: mem.total,
                    vram_used: mem.used,
                    vram_used_by_mold: used_by_mold,
                    vram_used_by_other: used_by_other,
                });
            }
            out
        }
    }
}

#[cfg(feature = "nvml")]
pub use nvml_source::NvmlSource;

use mold_core::{GpuBackend, GpuSnapshot};

/// Locate the `nvidia-smi` binary. Matches the existing resolver in
/// `routes.rs::query_gpu_info` so NixOS hosts still work.
pub(crate) fn resolve_nvidia_smi() -> &'static str {
    if std::path::Path::new("/run/current-system/sw/bin/nvidia-smi").exists() {
        "/run/current-system/sw/bin/nvidia-smi"
    } else {
        "nvidia-smi"
    }
}

/// Parse a single `nvidia-smi --query-gpu=index,name,memory.total,memory.used --format=csv,noheader,nounits`
/// line. Returns `(ordinal, name, total_bytes, used_bytes)` or `None` if the
/// line doesn't have the expected shape.
pub fn parse_nvidia_smi_line(line: &str) -> Option<(usize, String, u64, u64)> {
    let parts: Vec<&str> = line.split(',').map(str::trim).collect();
    if parts.len() < 4 {
        return None;
    }
    let ordinal: usize = parts[0].parse().ok()?;
    let name = parts[1].to_string();
    let total_mb: u64 = parts[2].parse().ok()?;
    let used_mb: u64 = parts[3].parse().ok()?;
    // nvidia-smi with `nounits` reports MiB; we expose bytes. Upstream uses
    // 1 MiB = 1_000_000 for display consistency with the rest of mold.
    Some((ordinal, name, total_mb * 1_000_000, used_mb * 1_000_000))
}

use mold_core::RamSnapshot;
use sysinfo::{Pid, ProcessRefreshKind, RefreshKind, System};

/// Metal unified-memory snapshot — macOS only. Off-Darwin returns an empty
/// Vec so callers on Linux/CUDA hosts can unconditionally call this.
///
/// Unified memory means there's no distinct VRAM total; we report the
/// system RAM total so the SPA's GPU row still communicates "this is how
/// much the GPU can address." Per-process attribution is unavailable on
/// macOS (IOKit doesn't expose it in userspace), so both per-process fields
/// are `None` and the SPA hides those rows.
pub fn metal_snapshot() -> Vec<GpuSnapshot> {
    #[cfg(target_os = "macos")]
    {
        let mut sys = sysinfo::System::new_with_specifics(
            sysinfo::RefreshKind::nothing().with_memory(sysinfo::MemoryRefreshKind::everything()),
        );
        sys.refresh_memory();
        let total = sys.total_memory();
        let used = sys.used_memory();
        vec![GpuSnapshot {
            ordinal: 0,
            name: "Apple Metal GPU".to_string(),
            backend: GpuBackend::Metal,
            vram_total: total,
            vram_used: used,
            vram_used_by_mold: None,
            vram_used_by_other: None,
        }]
    }
    #[cfg(not(target_os = "macos"))]
    {
        Vec::new()
    }
}

/// Build a single `RamSnapshot` using `sysinfo`. Refreshes only memory and
/// the current process — cheap enough to run at 1 Hz (~200 µs).
pub fn ram_snapshot() -> RamSnapshot {
    let mut sys = System::new_with_specifics(
        RefreshKind::nothing()
            .with_memory(sysinfo::MemoryRefreshKind::everything())
            .with_processes(ProcessRefreshKind::nothing().with_memory()),
    );
    sys.refresh_memory();
    let pid = Pid::from_u32(std::process::id());
    sys.refresh_processes_specifics(
        sysinfo::ProcessesToUpdate::Some(&[pid]),
        true,
        ProcessRefreshKind::nothing().with_memory(),
    );
    let total = sys.total_memory();
    let used = sys.used_memory();
    let used_by_mold = sys.process(pid).map(|p| p.memory()).unwrap_or(0);
    let used_by_other = used.saturating_sub(used_by_mold);
    RamSnapshot {
        total,
        used,
        used_by_mold,
        used_by_other,
    }
}

pub struct SmiSource;

impl SmiSource {
    /// Invoke `nvidia-smi` and parse the output. Returns an empty Vec if the
    /// binary isn't present or returns non-zero.
    ///
    /// Cost note: this fork/execs `nvidia-smi`, which takes on the order of
    /// tens of milliseconds — not microseconds. Call from a blocking task
    /// (e.g. `tokio::task::spawn_blocking`) if invoked from an async context.
    pub fn snapshot() -> Vec<GpuSnapshot> {
        let bin = resolve_nvidia_smi();
        let output = match std::process::Command::new(bin)
            .args([
                "--query-gpu=index,name,memory.total,memory.used",
                "--format=csv,noheader,nounits",
            ])
            .output()
        {
            Ok(o) if o.status.success() => o,
            Ok(_) => return Vec::new(),
            Err(_) => return Vec::new(),
        };
        let text = match String::from_utf8(output.stdout) {
            Ok(s) => s,
            Err(_) => return Vec::new(),
        };
        Self::parse_snapshot(&text)
    }

    /// Pure parser — split out for testability.
    pub fn parse_snapshot(text: &str) -> Vec<GpuSnapshot> {
        text.lines()
            .filter_map(|l| {
                let (ordinal, name, total, used) = parse_nvidia_smi_line(l)?;
                Some(GpuSnapshot {
                    ordinal,
                    name,
                    backend: GpuBackend::Cuda,
                    vram_total: total,
                    vram_used: used,
                    vram_used_by_mold: None,
                    vram_used_by_other: None,
                })
            })
            .collect()
    }
}

/// Assemble a single `ResourceSnapshot` from whichever data sources are
/// available on the current host. Cheap enough to run at 1 Hz (~200 µs).
///
/// Source priority on CUDA: NVML (if linked) → `nvidia-smi` subprocess → empty.
/// On macOS: `metal_snapshot()`.
pub fn build_snapshot() -> ResourceSnapshot {
    let hostname = hostname::get()
        .ok()
        .and_then(|h| h.into_string().ok())
        .unwrap_or_else(|| "unknown".to_string());
    let timestamp = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_millis() as i64)
        .unwrap_or(0);

    let gpus = collect_gpus();
    let system_ram = ram_snapshot();

    ResourceSnapshot {
        hostname,
        timestamp,
        gpus,
        system_ram,
    }
}

#[allow(clippy::needless_return)]
fn collect_gpus() -> Vec<GpuSnapshot> {
    // Darwin: Metal is the only GPU path.
    #[cfg(target_os = "macos")]
    {
        return metal_snapshot();
    }
    // Linux / other: try NVML first, fall back to nvidia-smi.
    #[cfg(all(not(target_os = "macos"), feature = "nvml"))]
    {
        if let Ok(src) = NvmlSource::try_new() {
            let gpus = src.snapshot(std::process::id());
            if !gpus.is_empty() {
                return gpus;
            }
        }
    }
    #[cfg(not(target_os = "macos"))]
    {
        SmiSource::snapshot()
    }
}

/// Spawn the 1 Hz aggregator task. Returns the `JoinHandle` so `run_server`
/// can drop it on shutdown. The task fires once immediately on startup so
/// `GET /api/resources` succeeds without waiting a full second.
pub fn spawn_aggregator(bcast: Arc<ResourceBroadcaster>) -> JoinHandle<()> {
    tokio::spawn(async move {
        // Immediate first tick so `latest()` is populated before any HTTP
        // request arrives.
        bcast.publish(build_snapshot());
        let mut interval = tokio::time::interval(Duration::from_secs(1));
        interval.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Skip);
        // Consume the first tick (it fires immediately) so we don't double-emit.
        interval.tick().await;
        loop {
            interval.tick().await;
            let snap = tokio::task::spawn_blocking(build_snapshot)
                .await
                .unwrap_or_else(|_| ResourceSnapshot {
                    hostname: "unknown".to_string(),
                    timestamp: 0,
                    gpus: Vec::new(),
                    system_ram: mold_core::RamSnapshot {
                        total: 0,
                        used: 0,
                        used_by_mold: 0,
                        used_by_other: 0,
                    },
                });
            bcast.publish(snap);
        }
    })
}
