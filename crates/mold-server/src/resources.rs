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
