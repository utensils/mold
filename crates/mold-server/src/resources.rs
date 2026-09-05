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

/// One CUDA/Metal device from the runtime-visible discovery inventory.
///
/// CUDA UUID bytes, rather than NVML ordinals, are the join key. The ordinal
/// is the process-local CUDA display ordinal that clients already use. This
/// distinction is load-bearing when `CUDA_VISIBLE_DEVICES` reorders cards or
/// contains UUID/MIG selectors.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct TelemetryTarget {
    pub logical_ordinal: usize,
    pub backend: mold_core::GpuBackend,
    pub raw_cuda_uuid: Option<[u8; 16]>,
    pub cuda_kind: Option<mold_inference::device::CudaDeviceKind>,
    pub name: String,
    pub total_memory_bytes: u64,
    pub nvml_uuid: Option<String>,
    pub physical_uuid: Option<String>,
    pub mig_uuid: Option<String>,
    pub pci_bus_id: Option<String>,
}

impl TelemetryTarget {
    #[cfg(test)]
    pub(crate) fn cuda(
        logical_ordinal: usize,
        raw_cuda_uuid: [u8; 16],
        cuda_kind: mold_inference::device::CudaDeviceKind,
        name: String,
        total_memory_bytes: u64,
    ) -> Self {
        Self {
            logical_ordinal,
            backend: mold_core::GpuBackend::Cuda,
            raw_cuda_uuid: Some(raw_cuda_uuid),
            cuda_kind: Some(cuda_kind),
            name,
            total_memory_bytes,
            nvml_uuid: None,
            physical_uuid: None,
            mig_uuid: None,
            pci_bus_id: None,
        }
    }

    pub(crate) fn from_discovered(gpu: &mold_inference::device::DiscoveredGpu) -> Self {
        Self {
            logical_ordinal: gpu.ordinal,
            backend: gpu.backend,
            raw_cuda_uuid: gpu.raw_cuda_uuid,
            cuda_kind: gpu.device_kind,
            name: gpu.name.clone(),
            total_memory_bytes: gpu.total_vram_bytes,
            nvml_uuid: None,
            physical_uuid: None,
            mig_uuid: None,
            pci_bus_id: gpu.pci_bus_id.clone(),
        }
    }
}

/// Resolve the one-time telemetry metadata join used to populate the
/// canonical [`crate::device_registry::DeviceRegistry`]. The returned values
/// are not retained as a second inventory; the 1 Hz sampler asks the registry
/// for its driver-free target projection on each tick.
pub(crate) fn discover_telemetry_targets(
    discovered: &[mold_inference::device::DiscoveredGpu],
) -> Vec<TelemetryTarget> {
    #[allow(unused_mut)]
    let mut targets: Vec<_> = discovered
        .iter()
        .map(TelemetryTarget::from_discovered)
        .collect();
    #[cfg(feature = "nvml")]
    if let Ok(source) = NvmlSource::try_new() {
        for target in &mut targets {
            if let Some(metadata) = source.metadata(target) {
                target.nvml_uuid = Some(metadata.nvml_uuid.clone());
                target.physical_uuid = metadata.physical_uuid;
                target.mig_uuid = metadata.mig_uuid;
                target.pci_bus_id = metadata.pci_bus_id.or(target.pci_bus_id.take());
            }
        }
    }
    targets
}

#[cfg(any(test, feature = "nvml", not(target_os = "macos")))]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum NvidiaUuidKind {
    FullGpu,
    Mig,
}

#[cfg(feature = "nvml")]
fn nvidia_uuid_text(prefix: &str, uuid: [u8; 16]) -> String {
    format!(
        "{prefix}-{:02x}{:02x}{:02x}{:02x}-{:02x}{:02x}-{:02x}{:02x}-{:02x}{:02x}-{:02x}{:02x}{:02x}{:02x}{:02x}{:02x}",
        uuid[0],
        uuid[1],
        uuid[2],
        uuid[3],
        uuid[4],
        uuid[5],
        uuid[6],
        uuid[7],
        uuid[8],
        uuid[9],
        uuid[10],
        uuid[11],
        uuid[12],
        uuid[13],
        uuid[14],
        uuid[15],
    )
}

#[cfg(any(test, feature = "nvml", not(target_os = "macos")))]
fn parse_nvidia_uuid(value: &str) -> Option<(NvidiaUuidKind, [u8; 16])> {
    let (kind, body) = if value
        .get(..4)
        .is_some_and(|prefix| prefix.eq_ignore_ascii_case("GPU-"))
    {
        (NvidiaUuidKind::FullGpu, &value[4..])
    } else if value
        .get(..4)
        .is_some_and(|prefix| prefix.eq_ignore_ascii_case("MIG-"))
    {
        (NvidiaUuidKind::Mig, &value[4..])
    } else {
        return None;
    };
    // Deliberately reject the legacy `MIG-GPU-.../gi/ci` spelling. It does
    // not contain the CUDA v2 compute-instance UUID, so accepting it would
    // risk overlaying parent/full-GPU memory onto a MIG worker.
    let compact: String = body.chars().filter(|character| *character != '-').collect();
    if compact.len() != 32 || !compact.bytes().all(|byte| byte.is_ascii_hexdigit()) {
        return None;
    }
    let mut bytes = [0_u8; 16];
    for (index, byte) in bytes.iter_mut().enumerate() {
        *byte = u8::from_str_radix(&compact[index * 2..index * 2 + 2], 16).ok()?;
    }
    Some((kind, bytes))
}

#[cfg(any(test, feature = "nvml", not(target_os = "macos")))]
fn target_accepts_nvidia_uuid(target: &TelemetryTarget, value: &str) -> bool {
    let Some(expected) = target.raw_cuda_uuid else {
        return false;
    };
    let Some((actual_kind, actual)) = parse_nvidia_uuid(value) else {
        return false;
    };
    if actual != expected {
        return false;
    }
    match target.cuda_kind {
        Some(mold_inference::device::CudaDeviceKind::FullGpu) => {
            actual_kind == NvidiaUuidKind::FullGpu
        }
        Some(mold_inference::device::CudaDeviceKind::Mig) => actual_kind == NvidiaUuidKind::Mig,
        Some(mold_inference::device::CudaDeviceKind::UnknownCuda) | None => true,
    }
}

#[cfg(feature = "nvml")]
fn nvidia_uuid_candidates(target: &TelemetryTarget) -> Vec<String> {
    if let Some(uuid) = &target.nvml_uuid {
        return vec![uuid.clone()];
    }
    let Some(raw) = target.raw_cuda_uuid else {
        return Vec::new();
    };
    match target.cuda_kind {
        Some(mold_inference::device::CudaDeviceKind::FullGpu) => {
            vec![nvidia_uuid_text("GPU", raw)]
        }
        Some(mold_inference::device::CudaDeviceKind::Mig) => {
            vec![nvidia_uuid_text("MIG", raw)]
        }
        Some(mold_inference::device::CudaDeviceKind::UnknownCuda) | None => {
            vec![nvidia_uuid_text("GPU", raw), nvidia_uuid_text("MIG", raw)]
        }
    }
}

#[cfg(feature = "nvml")]
pub(crate) mod nvml_source {
    use super::{
        nvidia_uuid_candidates, target_accepts_nvidia_uuid, NvidiaUuidKind, TelemetryTarget,
    };
    use mold_core::{GpuBackend, GpuSnapshot};
    use nvml_wrapper::enums::device::UsedGpuMemory;
    use nvml_wrapper::Device;
    use nvml_wrapper::Nvml;

    pub(crate) struct NvmlMetadata {
        pub nvml_uuid: String,
        pub physical_uuid: Option<String>,
        pub mig_uuid: Option<String>,
        pub pci_bus_id: Option<String>,
    }

    pub(crate) struct NvmlSource {
        nvml: Nvml,
    }

    impl NvmlSource {
        pub(crate) fn try_new() -> anyhow::Result<Self> {
            let nvml = Nvml::init()?;
            Ok(Self { nvml })
        }

        fn matching_device<'a>(&'a self, target: &TelemetryTarget) -> Option<Device<'a>> {
            for candidate in nvidia_uuid_candidates(target) {
                let Ok(device) = self.nvml.device_by_uuid(candidate.as_str()) else {
                    continue;
                };
                let Ok(actual_uuid) = device.uuid() else {
                    continue;
                };
                if target_accepts_nvidia_uuid(target, &actual_uuid) {
                    return Some(device);
                }
            }
            None
        }

        pub(crate) fn metadata(&self, target: &TelemetryTarget) -> Option<NvmlMetadata> {
            if target.backend != GpuBackend::Cuda {
                return None;
            }
            let device = self.matching_device(target)?;
            let nvml_uuid = device.uuid().ok()?;
            let kind = super::parse_nvidia_uuid(&nvml_uuid)?.0;
            let pci_bus_id = device.pci_info().ok().map(|info| info.bus_id);
            Some(NvmlMetadata {
                physical_uuid: (kind == NvidiaUuidKind::FullGpu).then(|| nvml_uuid.clone()),
                mig_uuid: (kind == NvidiaUuidKind::Mig).then(|| nvml_uuid.clone()),
                nvml_uuid,
                pci_bus_id,
            })
        }

        pub(crate) fn snapshot_visible(
            &self,
            pid: u32,
            targets: &[TelemetryTarget],
        ) -> Vec<GpuSnapshot> {
            targets
                .iter()
                .filter(|target| target.backend == GpuBackend::Cuda)
                .filter_map(|target| {
                    let dev = self.matching_device(target)?;
                    let name = dev.name().unwrap_or_else(|_| target.name.clone());
                    let mem = match dev.memory_info() {
                        Ok(memory) => memory,
                        Err(error) => {
                            tracing::debug!(
                                ordinal = target.logical_ordinal,
                                err = %error,
                                "NVML memory_info failed"
                            );
                            return None;
                        }
                    };
                    let used_by_mold = dev.running_compute_processes().ok().map(|processes| {
                        processes
                            .iter()
                            .filter(|process| process.pid == pid)
                            .map(|process| match process.used_gpu_memory {
                                UsedGpuMemory::Used(bytes) => bytes,
                                UsedGpuMemory::Unavailable => 0,
                            })
                            .sum::<u64>()
                    });
                    let used_by_other = used_by_mold.map(|mold| mem.used.saturating_sub(mold));
                    let gpu_utilization = dev
                        .utilization_rates()
                        .ok()
                        .map(|usage| usage.gpu.min(100) as u8);
                    Some(GpuSnapshot {
                        ordinal: target.logical_ordinal,
                        name,
                        backend: GpuBackend::Cuda,
                        vram_total: mem.total,
                        vram_used: mem.used,
                        vram_used_by_mold: used_by_mold,
                        vram_used_by_other: used_by_other,
                        gpu_utilization,
                    })
                })
                .collect()
        }

        /// Produce a per-GPU snapshot. `pid` is `std::process::id()` of this
        /// server process; we filter `running_compute_processes()` against it
        /// to attribute `vram_used_by_mold`.
        pub(crate) fn snapshot(&self, pid: u32) -> Vec<GpuSnapshot> {
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
                // NVML's GPU-core utilization over the last sample period.
                // Cheap — this is just a driver query, not a counter reset.
                let gpu_util = dev.utilization_rates().ok().map(|u| u.gpu.min(100) as u8);
                out.push(GpuSnapshot {
                    ordinal: ordinal as usize,
                    name,
                    backend: GpuBackend::Cuda,
                    vram_total: mem.total,
                    vram_used: mem.used,
                    vram_used_by_mold: used_by_mold,
                    vram_used_by_other: used_by_other,
                    gpu_utilization: gpu_util,
                });
            }
            out
        }
    }
}

#[cfg(feature = "nvml")]
pub(crate) use nvml_source::NvmlSource;

#[cfg(any(feature = "nvml", test))]
pub(crate) fn nonzero_process_vram(bytes: Option<u64>) -> Option<u64> {
    bytes.filter(|bytes| *bytes > 0)
}

/// Fresh process-attributed VRAM for one runtime CUDA device.
///
/// Cache load deltas are global before/after measurements and may become stale
/// after an engine drops components. Hot-cache admission therefore clamps any
/// claimed reusable footprint to NVML's current accounting for this process.
/// Missing or zero attribution is ambiguous in PID namespaces, WSL2, and MIG:
/// it can mean "unreportable" rather than "unused". Return `None` in that
/// case so callers do not turn telemetry absence into an authoritative zero.
#[cfg(feature = "nvml")]
pub(crate) fn current_process_vram_bytes(
    gpu: &mold_inference::device::DiscoveredGpu,
) -> Option<u64> {
    if gpu.backend != mold_core::GpuBackend::Cuda {
        return None;
    }
    let target = TelemetryTarget::from_discovered(gpu);
    nonzero_process_vram(
        NvmlSource::try_new()
            .ok()?
            .snapshot_visible(std::process::id(), std::slice::from_ref(&target))
            .into_iter()
            .find(|snapshot| snapshot.ordinal == gpu.ordinal)
            .and_then(|snapshot| snapshot.vram_used_by_mold),
    )
}

#[cfg(not(feature = "nvml"))]
pub(crate) fn current_process_vram_bytes(
    _gpu: &mold_inference::device::DiscoveredGpu,
) -> Option<u64> {
    None
}

use mold_core::{GpuBackend, GpuSnapshot};

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
    // nvidia-smi with `nounits` reports MiB (2^20 bytes), not decimal MB.
    Some((
        ordinal,
        name,
        total_mb.checked_mul(1024 * 1024)?,
        used_mb.checked_mul(1024 * 1024)?,
    ))
}

use mold_core::{CpuSnapshot, RamSnapshot};
use sysinfo::{CpuRefreshKind, Pid, ProcessRefreshKind, RefreshKind, System};

/// Project one host sample onto Metal's shared physical pool. The scheduler
/// derives headroom from `total - used`, so that subtraction must recover
/// exactly the available bytes the host snapshot reports.
#[cfg(any(test, target_os = "macos"))]
pub(crate) fn metal_snapshot_from_ram(ram: &RamSnapshot) -> GpuSnapshot {
    GpuSnapshot {
        ordinal: 0,
        name: "Apple Metal GPU".into(),
        backend: GpuBackend::Metal,
        vram_total: ram.total,
        vram_used: ram.total.saturating_sub(ram.available_or_estimate()),
        vram_used_by_mold: None,
        vram_used_by_other: None,
        gpu_utilization: None,
    }
}

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
        vec![metal_snapshot_from_ram(&ram_snapshot())]
    }
    #[cfg(not(target_os = "macos"))]
    {
        Vec::new()
    }
}

/// Build a single `RamSnapshot` for host admission: macOS uses the worker's
/// free + inactive authority; other hosts use `sysinfo`, with evictable ZFS
/// ARC credit recorded beside `MemAvailable` (#1439).
///
/// This is the ONE place the credit enters a sample. Every consumer that
/// spends host memory — the scheduler ledger, H3 admission, the reclaim
/// re-sample, the forced-local CLI — reads
/// [`RamSnapshot::available_with_evictable_arc`] off this snapshot, so the
/// credit is added exactly once and `available` keeps meaning `MemAvailable`.
pub fn ram_snapshot() -> RamSnapshot {
    ram_snapshot_from_system().with_zfs_arc_credit(crate::zfs_arc::evictable_arc_credit())
}

/// Build a single `RamSnapshot` using `sysinfo`. Refreshes only memory and
/// the current process — cheap enough to run at 1 Hz (~200 µs).
///
/// `reclaimable_zfs_arc` is `None` here on purpose: this is the reading for
/// RSS-only probes (`used_by_mold` before/after an unload), which have no
/// business reading arcstats. Admission goes through [`ram_snapshot`].
pub(crate) fn ram_snapshot_from_system() -> RamSnapshot {
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
    // Metal admission and worker preflight spend the same free + inactive
    // authority. Do not add sysinfo's broader reclaimable-page estimate.
    #[cfg(target_os = "macos")]
    let available = mold_inference::device::available_system_memory_bytes().unwrap_or(0);
    #[cfg(not(target_os = "macos"))]
    let available = sys.available_memory();
    let used_by_mold = sys.process(pid).map(|p| p.memory()).unwrap_or(0);
    let used_by_other = used.saturating_sub(used_by_mold);
    RamSnapshot {
        total,
        used,
        available: Some(available.min(total)),
        reclaimable_zfs_arc: None,
        used_by_mold,
        used_by_other,
    }
}

pub struct SmiSource;

#[cfg(any(test, not(target_os = "macos")))]
#[derive(Debug)]
struct SmiPhysicalSample {
    uuid: String,
    name: String,
    vram_total: u64,
    vram_used: u64,
}

#[cfg(any(test, not(target_os = "macos")))]
fn parse_visible_nvidia_smi_line(line: &str) -> Option<SmiPhysicalSample> {
    let parts: Vec<&str> = line.split(',').map(str::trim).collect();
    if parts.len() < 5 {
        return None;
    }
    let _physical_ordinal: usize = parts[0].parse().ok()?;
    let uuid = parts[1].to_string();
    let name = parts[2].to_string();
    let total_mib: u64 = parts[3].parse().ok()?;
    let used_mib: u64 = parts[4].parse().ok()?;
    Some(SmiPhysicalSample {
        uuid,
        name,
        vram_total: total_mib.checked_mul(1024 * 1024)?,
        vram_used: used_mib.checked_mul(1024 * 1024)?,
    })
}

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

    /// Invoke `nvidia-smi` and project only CUDA-runtime-visible devices.
    ///
    /// The subprocess can see physical GPUs hidden by
    /// `CUDA_VISIBLE_DEVICES`, so filtering by CUDA's UUID inventory is a
    /// security and correctness boundary, not merely a display preference.
    #[cfg(not(target_os = "macos"))]
    pub(crate) fn snapshot_visible(targets: &[TelemetryTarget]) -> Vec<GpuSnapshot> {
        let bin = resolve_nvidia_smi();
        let output = match std::process::Command::new(bin)
            .args([
                "--query-gpu=index,uuid,name,memory.total,memory.used",
                "--format=csv,noheader,nounits",
            ])
            .output()
        {
            Ok(output) if output.status.success() => output,
            Ok(_) | Err(_) => return Vec::new(),
        };
        let Ok(text) = String::from_utf8(output.stdout) else {
            return Vec::new();
        };
        Self::parse_visible_snapshot(&text, targets)
    }

    /// Pure UUID join used by tests and by the subprocess fallback.
    #[cfg(any(test, not(target_os = "macos")))]
    pub(crate) fn parse_visible_snapshot(
        text: &str,
        targets: &[TelemetryTarget],
    ) -> Vec<GpuSnapshot> {
        let physical: Vec<_> = text
            .lines()
            .filter_map(parse_visible_nvidia_smi_line)
            .collect();
        targets
            .iter()
            .filter(|target| target.backend == GpuBackend::Cuda)
            .filter_map(|target| {
                let sample = physical
                    .iter()
                    .find(|sample| target_accepts_nvidia_uuid(target, &sample.uuid))?;
                Some(GpuSnapshot {
                    ordinal: target.logical_ordinal,
                    name: sample.name.clone(),
                    backend: GpuBackend::Cuda,
                    vram_total: sample.vram_total,
                    vram_used: sample.vram_used,
                    vram_used_by_mold: None,
                    vram_used_by_other: None,
                    gpu_utilization: None,
                })
            })
            .collect()
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
                    gpu_utilization: None,
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
///
/// CPU utilization is `None` — call `build_snapshot_with_cpu` with a
/// persistent `System` to populate it (sysinfo computes CPU usage from
/// deltas between refreshes, so the aggregator needs to hold state).
pub fn build_snapshot() -> ResourceSnapshot {
    build_snapshot_inner(None, None)
}

fn build_snapshot_inner(
    cpu: Option<CpuSnapshot>,
    inventory: Option<&[TelemetryTarget]>,
) -> ResourceSnapshot {
    let hostname = hostname::get()
        .ok()
        .and_then(|h| h.into_string().ok())
        .unwrap_or_else(|| "unknown".to_string());
    let timestamp = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_millis() as i64)
        .unwrap_or(0);

    let system_ram = ram_snapshot();
    let gpus = collect_gpus(inventory, &system_ram);

    ResourceSnapshot {
        hostname,
        timestamp,
        gpus,
        system_ram,
        cpu,
    }
}

/// Holds the persistent `System` sysinfo needs for CPU delta computation.
pub struct CpuSampler {
    sys: System,
    cores: u16,
}

impl CpuSampler {
    pub fn new() -> Self {
        let mut sys = System::new_with_specifics(
            RefreshKind::nothing().with_cpu(CpuRefreshKind::everything().with_cpu_usage()),
        );
        // Prime the sampler. The first `global_cpu_usage()` read always
        // returns 0 — the real number shows up on the second refresh.
        sys.refresh_cpu_usage();
        let cores = sys.cpus().len().min(u16::MAX as usize) as u16;
        Self { sys, cores }
    }

    pub fn sample(&mut self) -> CpuSnapshot {
        self.sys.refresh_cpu_usage();
        CpuSnapshot {
            cores: self.cores,
            usage_percent: self.sys.global_cpu_usage().clamp(0.0, 100.0),
        }
    }
}

impl Default for CpuSampler {
    fn default() -> Self {
        Self::new()
    }
}

#[allow(clippy::needless_return)]
fn collect_gpus(inventory: Option<&[TelemetryTarget]>, ram: &RamSnapshot) -> Vec<GpuSnapshot> {
    #[cfg(not(target_os = "macos"))]
    let _ = ram;
    // Darwin: Metal is the only GPU path.
    #[cfg(target_os = "macos")]
    {
        // One sample feeds both host and device telemetry, not two reads of
        // a shared physical pool taken at different moments.
        let snapshots = vec![metal_snapshot_from_ram(ram)];
        return match inventory {
            Some(inventory)
                if !inventory
                    .iter()
                    .any(|target| target.backend == GpuBackend::Metal) =>
            {
                Vec::new()
            }
            Some(inventory) => snapshots
                .into_iter()
                .filter(|snapshot| {
                    inventory.iter().any(|target| {
                        target.backend == GpuBackend::Metal
                            && target.logical_ordinal == snapshot.ordinal
                    })
                })
                .collect(),
            None => snapshots,
        };
    }
    // Linux / other: try NVML first, fall back to nvidia-smi.
    #[cfg(all(not(target_os = "macos"), feature = "nvml"))]
    {
        if let Ok(src) = NvmlSource::try_new() {
            let gpus = match inventory {
                Some(inventory) => src.snapshot_visible(std::process::id(), inventory),
                None => src.snapshot(std::process::id()),
            };
            if !gpus.is_empty() {
                return gpus;
            }
        }
    }
    #[cfg(not(target_os = "macos"))]
    {
        match inventory {
            Some(inventory) => SmiSource::snapshot_visible(inventory),
            None => SmiSource::snapshot(),
        }
    }
}

/// Spawn the 1 Hz aggregator task. Returns the `JoinHandle` so `run_server`
/// can drop it on shutdown. The task fires once immediately on startup so
/// `GET /api/resources` succeeds without waiting a full second.
pub(crate) fn spawn_aggregator(
    bcast: Arc<ResourceBroadcaster>,
    registry: Arc<crate::device_registry::DeviceRegistry>,
) -> JoinHandle<()> {
    tokio::spawn(async move {
        // Immediate first tick so `latest()` is populated before any HTTP
        // request arrives. CPU usage is None on this first sample (no delta
        // to compute against yet).
        let targets = registry.telemetry_targets();
        bcast.publish(build_snapshot_inner(None, Some(&targets)));
        let mut interval = tokio::time::interval(Duration::from_secs(1));
        interval.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Skip);
        // Consume the first tick (it fires immediately) so we don't double-emit.
        interval.tick().await;

        // The sampler lives on the blocking thread across ticks — sysinfo
        // computes CPU usage from deltas, so we can't rebuild it every tick.
        let mut sampler: Option<CpuSampler> = None;
        loop {
            interval.tick().await;
            let taken = sampler.take();
            let tick_registry = registry.clone();
            let (snap, returned) = tokio::task::spawn_blocking(move || {
                let mut s = taken.unwrap_or_default();
                let cpu = s.sample();
                let targets = tick_registry.telemetry_targets();
                let snap = build_snapshot_inner(Some(cpu), Some(&targets));
                (snap, s)
            })
            .await
            .unwrap_or_else(|_| {
                (
                    ResourceSnapshot {
                        hostname: "unknown".to_string(),
                        timestamp: 0,
                        gpus: Vec::new(),
                        system_ram: mold_core::RamSnapshot {
                            total: 0,
                            used: 0,
                            available: None,
                            reclaimable_zfs_arc: None,
                            used_by_mold: 0,
                            used_by_other: 0,
                        },
                        cpu: None,
                    },
                    CpuSampler::new(),
                )
            });
            sampler = Some(returned);
            bcast.publish(snap);
        }
    })
}
