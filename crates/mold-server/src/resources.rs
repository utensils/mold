//! Always-on VRAM + system-RAM telemetry aggregator.
//!
//! A single `tokio::spawn`ed task builds a `ResourceSnapshot` every 1 s and
//! broadcasts it through `ResourceBroadcaster`. The HTTP layer in
//! `routes.rs` exposes both a one-shot `GET /api/resources` endpoint (reads
//! the most recently published snapshot) and an SSE stream
//! `GET /api/resources/stream` that replays the broadcast channel.

use mold_core::ResourceSnapshot;
use std::sync::Arc;
use tokio::sync::{broadcast, Mutex};

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
        // Cache first, then fan out.
        // `try_lock` is used because the aggregator never contends with other
        // writers and publish may be called from sync/async contexts.
        if let Ok(mut guard) = self.latest.try_lock() {
            *guard = Some(snapshot.clone());
        }
        let _ = self.tx.send(snapshot);
    }

    pub fn subscribe(&self) -> broadcast::Receiver<ResourceSnapshot> {
        self.tx.subscribe()
    }

    /// Returns the most recent published snapshot. Used by `GET /api/resources`.
    pub fn latest(&self) -> Option<ResourceSnapshot> {
        self.latest.try_lock().ok().and_then(|g| g.clone())
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

pub struct SmiSource;

impl SmiSource {
    /// Invoke `nvidia-smi` and parse the output. Returns an empty Vec if the
    /// binary isn't present or returns non-zero.
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
