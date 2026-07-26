//! Authoritative read projection for runtime-visible devices.
//!
//! Phase A deliberately does not own dispatch or worker lifecycle. The
//! registry joins a narrow discovery inventory with the current worker pool,
//! the background telemetry cache, and machine-wide desired preferences. That
//! keeps `/api/devices` and legacy `/api/status` on one source without moving
//! jobs or querying CUDA on an HTTP request.

use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::sync::{Arc, Mutex, RwLock};

use mold_core::{
    DeviceActivity, DeviceAdminState, DeviceHealth, DeviceInfo, DeviceKind, DeviceMemoryInfo,
    DeviceState, DeviceTelemetry, GpuBackend, GpuInfo, GpuWorkerState, GpuWorkerStatus,
    ResourceSnapshot,
};

use crate::gpu_pool::GpuPool;
use crate::job_registry::SharedJobRegistry;

/// Discovery-owned facts. CUDA discovery can populate this without depending
/// on Axum, the worker pool, telemetry, or SQLite.
#[derive(Debug, Clone, PartialEq)]
pub struct DiscoveredDevice {
    pub stable_id: Option<String>,
    pub backend: GpuBackend,
    pub visible_ordinal: Option<usize>,
    pub device_kind: DeviceKind,
    pub nvml_uuid: Option<String>,
    pub physical_uuid: Option<String>,
    pub mig_uuid: Option<String>,
    pub mig_parent_uuid: Option<String>,
    pub mig_profile: Option<String>,
    pub pci_bus_id: Option<String>,
    pub name: String,
    pub compute_capability: Option<(u16, u16)>,
    pub total_memory_bytes: Option<u64>,
    pub startup_allowed: bool,
    /// Ordinal used by the current telemetry snapshot. `None` is required
    /// when CUDA visibility used a UUID/MIG selector that cannot safely be
    /// ordinal-joined.
    pub telemetry_ordinal: Option<usize>,
}

impl DiscoveredDevice {
    pub(crate) fn from_runtime_gpu(
        gpu: &mold_inference::device::DiscoveredGpu,
        startup_allowed: bool,
        telemetry: Option<&crate::resources::TelemetryTarget>,
    ) -> Self {
        let device_kind = if telemetry.is_some_and(|target| target.mig_uuid.is_some()) {
            DeviceKind::Mig
        } else if telemetry.is_some_and(|target| target.physical_uuid.is_some()) {
            DeviceKind::FullGpu
        } else {
            match gpu.device_kind {
                Some(mold_inference::device::CudaDeviceKind::FullGpu) => DeviceKind::FullGpu,
                Some(mold_inference::device::CudaDeviceKind::Mig) => DeviceKind::Mig,
                Some(mold_inference::device::CudaDeviceKind::UnknownCuda) => {
                    DeviceKind::UnknownCuda
                }
                None => DeviceKind::Metal,
            }
        };
        Self {
            stable_id: gpu.stable_id.clone(),
            backend: gpu.backend,
            visible_ordinal: Some(gpu.ordinal),
            device_kind,
            nvml_uuid: telemetry.and_then(|target| target.nvml_uuid.clone()),
            physical_uuid: telemetry.and_then(|target| target.physical_uuid.clone()),
            mig_uuid: telemetry.and_then(|target| target.mig_uuid.clone()),
            mig_parent_uuid: None,
            mig_profile: None,
            pci_bus_id: telemetry
                .and_then(|target| target.pci_bus_id.clone())
                .or_else(|| gpu.pci_bus_id.clone()),
            name: gpu.name.clone(),
            compute_capability: gpu.compute_capability,
            total_memory_bytes: Some(gpu.total_vram_bytes),
            startup_allowed,
            telemetry_ordinal: telemetry.map(|target| target.logical_ordinal),
        }
    }
}

/// Adapter boundary for CUDA/Metal identity discovery. The registry remains
/// independent of cudarc and NVML APIs.
pub trait DeviceDiscoveryAdapter: Send + Sync {
    fn devices(&self) -> Vec<DiscoveredDevice>;
}

/// Immutable adapter used by server startup and deterministic tests.
#[derive(Debug, Clone, Default)]
pub struct StaticDeviceDiscovery {
    devices: Vec<DiscoveredDevice>,
}

impl StaticDeviceDiscovery {
    pub fn new(devices: Vec<DiscoveredDevice>) -> Self {
        Self { devices }
    }
}

impl DeviceDiscoveryAdapter for StaticDeviceDiscovery {
    fn devices(&self) -> Vec<DiscoveredDevice> {
        self.devices.clone()
    }
}

pub struct DeviceRegistry {
    discovery: Arc<dyn DeviceDiscoveryAdapter>,
    explicit_preferences: RwLock<BTreeMap<String, bool>>,
    metadata_db: Arc<Option<mold_db::MetadataDb>>,
    transient_ids: Mutex<HashMap<String, String>>,
}

impl DeviceRegistry {
    pub fn new(
        discovery: Arc<dyn DeviceDiscoveryAdapter>,
        metadata_db: Arc<Option<mold_db::MetadataDb>>,
    ) -> Self {
        let explicit_preferences = metadata_db
            .as_ref()
            .as_ref()
            .and_then(|db| match mold_db::DevicePreferences::new(db).list() {
                Ok(rows) => Some(
                    rows.into_iter()
                        .map(|(id, preference)| (id, preference.desired_enabled))
                        .collect(),
                ),
                Err(error) => {
                    tracing::warn!(
                        error = %format!("{error:#}"),
                        "device preferences unavailable; using enabled-by-default"
                    );
                    None
                }
            })
            .unwrap_or_default();

        Self {
            discovery,
            explicit_preferences: RwLock::new(explicit_preferences),
            metadata_db,
            transient_ids: Mutex::new(HashMap::new()),
        }
    }

    pub fn empty() -> Arc<Self> {
        Arc::new(Self::new(
            Arc::new(StaticDeviceDiscovery::default()),
            Arc::new(None),
        ))
    }

    /// Foundation for Phase C lifecycle mutation. Phase A has no route that
    /// calls this method, but DB-disabled mode already behaves correctly:
    /// changes remain process-local and log that they will not persist.
    pub fn set_desired_enabled(&self, device_id: &str, enabled: bool) -> anyhow::Result<()> {
        if let Some(db) = self.metadata_db.as_ref().as_ref() {
            mold_db::DevicePreferences::new(db).set(device_id, enabled)?;
        } else {
            tracing::warn!(
                device_id,
                desired_enabled = enabled,
                "metadata DB disabled; device preference will not persist"
            );
        }
        self.explicit_preferences
            .write()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .insert(device_id.to_string(), enabled);
        Ok(())
    }

    pub fn discovered_device(&self, device_id: &str) -> Option<DiscoveredDevice> {
        self.discovery
            .devices()
            .into_iter()
            .find(|device| device.stable_id.as_deref() == Some(device_id))
    }

    pub fn desired_enabled(&self, device_id: &str) -> bool {
        self.explicit_preferences
            .read()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .get(device_id)
            .copied()
            .unwrap_or(true)
    }

    pub fn has_devices(&self) -> bool {
        !self.discovery.devices().is_empty()
    }

    pub fn snapshot(
        &self,
        pool: &GpuPool,
        resources: Option<&ResourceSnapshot>,
        jobs: &SharedJobRegistry,
    ) -> DeviceState {
        let mut discovered = self.discovery.devices();
        let known_workers: BTreeSet<_> = discovered
            .iter()
            .filter_map(|device| {
                device
                    .visible_ordinal
                    .map(|ordinal| (device.backend, ordinal))
            })
            .collect();

        // During integration and tests, a worker may predate the UUID-first
        // adapter. Keep it visible, but mark it unavailable because an ordinal
        // is not a persistable identity.
        for worker in pool.worker_snapshot() {
            let telemetry_device = resources.and_then(|snapshot| {
                snapshot
                    .gpus
                    .iter()
                    .find(|gpu| gpu.ordinal == worker.gpu.ordinal)
            });
            let backend = discovered
                .iter()
                .find(|device| device.visible_ordinal == Some(worker.gpu.ordinal))
                .map(|device| device.backend)
                .or_else(|| telemetry_device.map(|gpu| gpu.backend))
                .unwrap_or_else(runtime_backend);
            if known_workers.contains(&(backend, worker.gpu.ordinal)) {
                continue;
            }
            discovered.push(DiscoveredDevice {
                stable_id: None,
                backend,
                visible_ordinal: Some(worker.gpu.ordinal),
                device_kind: if backend == GpuBackend::Metal {
                    DeviceKind::Metal
                } else {
                    DeviceKind::UnknownCuda
                },
                nvml_uuid: None,
                physical_uuid: None,
                mig_uuid: None,
                mig_parent_uuid: None,
                mig_profile: None,
                pci_bus_id: None,
                name: worker.gpu.name.clone(),
                compute_capability: None,
                total_memory_bytes: Some(worker.gpu.total_vram_bytes),
                startup_allowed: true,
                telemetry_ordinal: telemetry_device
                    .map(|gpu| gpu.ordinal)
                    .or_else(|| worker_telemetry_ordinal(backend, worker.gpu.ordinal)),
            });
        }

        discovered.sort_by(|left, right| {
            left.backend
                .wire_name()
                .cmp(right.backend.wire_name())
                .then(left.visible_ordinal.cmp(&right.visible_ordinal))
                .then(left.name.cmp(&right.name))
        });

        let worker_statuses = pool.gpu_status();
        let workers = pool.worker_snapshot();
        let queue = jobs.snapshot();
        let preferences = self
            .explicit_preferences
            .read()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        let devices = discovered
            .into_iter()
            .map(|device| {
                let stable_identity = device.stable_id.is_some();
                let id = device
                    .stable_id
                    .clone()
                    .unwrap_or_else(|| self.transient_id_for(&device));
                let desired_enabled = preferences.get(&id).copied().unwrap_or(true);
                let worker = device.visible_ordinal.and_then(|ordinal| {
                    worker_statuses
                        .iter()
                        .find(|status| status.ordinal == ordinal)
                });
                let telemetry = device.telemetry_ordinal.and_then(|ordinal| {
                    resources.and_then(|snapshot| {
                        snapshot
                            .gpus
                            .iter()
                            .find(|gpu| gpu.backend == device.backend && gpu.ordinal == ordinal)
                    })
                });

                let worker_ref = workers.iter().find(|worker| {
                    device.visible_ordinal == Some(worker.gpu.ordinal)
                        && device.backend == worker.gpu.backend
                });
                let admin_state = if !device.startup_allowed {
                    DeviceAdminState::StartupExcluded
                } else if pool.workers.is_starting(&id)
                    || (desired_enabled
                        && worker_ref.is_some_and(|worker| {
                            worker.drain_state.load(std::sync::atomic::Ordering::SeqCst)
                                == crate::gpu_pool::DRAIN_COMMITTED
                        }))
                {
                    DeviceAdminState::Starting
                } else if !desired_enabled && worker_ref.is_some() {
                    DeviceAdminState::Draining
                } else if desired_enabled {
                    DeviceAdminState::Enabled
                } else {
                    DeviceAdminState::Disabled
                };
                let health = if !stable_identity {
                    DeviceHealth::Unavailable
                } else if device.visible_ordinal.is_some() {
                    if let Some(worker_ref) = worker_ref {
                        if worker_ref
                            .poisoned
                            .load(std::sync::atomic::Ordering::SeqCst)
                        {
                            DeviceHealth::Poisoned
                        } else if worker_ref.is_degraded() {
                            DeviceHealth::Degraded
                        } else {
                            DeviceHealth::Healthy
                        }
                    } else if device.startup_allowed && desired_enabled {
                        DeviceHealth::Unavailable
                    } else {
                        DeviceHealth::Healthy
                    }
                } else {
                    DeviceHealth::Unavailable
                };
                let activity = worker
                    .map(|status| match status.state {
                        GpuWorkerState::Generating => DeviceActivity::Generating,
                        GpuWorkerState::Loading => DeviceActivity::Loading,
                        GpuWorkerState::Idle | GpuWorkerState::Degraded => DeviceActivity::Idle,
                    })
                    .unwrap_or(DeviceActivity::Idle);
                let activity = if admin_state == DeviceAdminState::Draining
                    && activity == DeviceActivity::Idle
                {
                    DeviceActivity::Stopping
                } else {
                    activity
                };
                let schedulable = admin_state == DeviceAdminState::Enabled
                    && health == DeviceHealth::Healthy
                    && worker_ref.is_some()
                    && worker_ref.is_some_and(|worker| {
                        !worker
                            .shutdown_requested
                            .load(std::sync::atomic::Ordering::SeqCst)
                            && worker.drain_state.load(std::sync::atomic::Ordering::SeqCst)
                                == crate::gpu_pool::DRAIN_RUNNING
                    });
                let unschedulable_reason = (!schedulable).then(|| {
                    if admin_state == DeviceAdminState::StartupExcluded {
                        "device_startup_excluded"
                    } else if admin_state == DeviceAdminState::Starting {
                        "device_starting"
                    } else if admin_state == DeviceAdminState::Draining {
                        "device_draining"
                    } else if admin_state == DeviceAdminState::Disabled {
                        "device_disabled"
                    } else if health == DeviceHealth::Degraded {
                        "device_degraded"
                    } else if let Some(error) = pool.workers.last_start_error(&id) {
                        return format!("device_start_failed: {error}");
                    } else {
                        "device_unavailable"
                    }
                    .to_string()
                });
                let loaded_models = worker
                    .and_then(|status| status.loaded_model.clone())
                    .into_iter()
                    .collect();
                let active_work_id = device.visible_ordinal.and_then(|ordinal| {
                    queue
                        .entries
                        .iter()
                        .find(|entry| {
                            entry.state == crate::job_registry::JobLifecycle::Running
                                && entry.gpu == Some(ordinal)
                        })
                        .map(|entry| entry.id.clone())
                });

                DeviceInfo {
                    id,
                    backend: device.backend,
                    ordinal: device.visible_ordinal,
                    device_kind: device.device_kind,
                    nvml_uuid: device.nvml_uuid,
                    physical_uuid: device.physical_uuid,
                    mig_uuid: device.mig_uuid,
                    mig_parent_uuid: device.mig_parent_uuid,
                    mig_profile: device.mig_profile,
                    name: device.name,
                    pci_bus_id: device.pci_bus_id,
                    compute_capability: device
                        .compute_capability
                        .map(|(major, minor)| format!("{major}.{minor}")),
                    memory: DeviceMemoryInfo {
                        total_bytes: telemetry
                            .map(|snapshot| snapshot.vram_total)
                            .or(device.total_memory_bytes),
                        used_bytes: telemetry.map(|snapshot| snapshot.vram_used),
                        mold_used_bytes: telemetry.and_then(|snapshot| snapshot.vram_used_by_mold),
                        other_used_bytes: telemetry
                            .and_then(|snapshot| snapshot.vram_used_by_other),
                    },
                    telemetry: DeviceTelemetry {
                        utilization_percent: telemetry
                            .and_then(|snapshot| snapshot.gpu_utilization),
                        temperature_c: None,
                        power_w: None,
                    },
                    desired_enabled,
                    restart_required: false,
                    admin_state,
                    health,
                    activity,
                    schedulable,
                    unschedulable_reason,
                    loaded_models,
                    active_work_id,
                    planned_work_ids: Vec::new(),
                }
            })
            .collect();

        DeviceState {
            devices,
            plan_version: 0,
        }
    }

    pub fn legacy_gpu_info(devices: &DeviceState) -> Option<GpuInfo> {
        let device = devices.devices.iter().find(|device| device.schedulable)?;
        Some(GpuInfo {
            name: device.name.clone(),
            vram_total_mb: device.memory.total_bytes.unwrap_or(0) / (1024 * 1024),
            vram_used_mb: device.memory.used_bytes.unwrap_or(0) / (1024 * 1024),
            backend: Some(device.backend),
        })
    }

    pub fn legacy_gpu_status_from_snapshot(devices: &DeviceState) -> Vec<GpuWorkerStatus> {
        devices
            .devices
            .iter()
            .filter(|device| device.schedulable)
            .filter_map(|device| {
                let ordinal = device.ordinal?;
                let state = if device.health == DeviceHealth::Degraded
                    || device.health == DeviceHealth::Poisoned
                {
                    GpuWorkerState::Degraded
                } else {
                    match device.activity {
                        DeviceActivity::Generating | DeviceActivity::Upscaling => {
                            GpuWorkerState::Generating
                        }
                        DeviceActivity::Loading
                        | DeviceActivity::AdminLoading
                        | DeviceActivity::Stopping => GpuWorkerState::Loading,
                        DeviceActivity::Idle => GpuWorkerState::Idle,
                    }
                };
                Some(GpuWorkerStatus {
                    ordinal,
                    name: device.name.clone(),
                    vram_total_bytes: device.memory.total_bytes.unwrap_or(0),
                    vram_used_bytes: device.memory.used_bytes.unwrap_or(0),
                    loaded_model: device.loaded_models.first().cloned(),
                    state,
                })
            })
            .collect()
    }

    /// Human-readable compatibility text derived exclusively from the cached
    /// registry projection. This must never query CUDA from an HTTP handler.
    pub fn legacy_memory_status(devices: &DeviceState) -> Option<String> {
        let device = devices.devices.iter().find(|device| device.schedulable)?;
        let free = device
            .memory
            .total_bytes?
            .saturating_sub(device.memory.used_bytes?);
        let label = if device.backend == GpuBackend::Metal {
            "Memory"
        } else {
            "VRAM"
        };
        Some(format!(
            "{label}: {:.1} GB free",
            free as f64 / 1_000_000_000.0
        ))
    }

    fn transient_id_for(&self, device: &DiscoveredDevice) -> String {
        let key = format!(
            "{}:{}:{}",
            device.backend.wire_name(),
            device
                .visible_ordinal
                .map_or_else(|| "none".to_string(), |ordinal| ordinal.to_string()),
            device.name
        );
        let mut ids = self
            .transient_ids
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        ids.entry(key)
            .or_insert_with(|| {
                format!(
                    "{}:unavailable-{}",
                    device.backend.wire_name(),
                    uuid::Uuid::new_v4().simple()
                )
            })
            .clone()
    }
}

trait GpuBackendWireName {
    fn wire_name(self) -> &'static str;
}

impl GpuBackendWireName for GpuBackend {
    fn wire_name(self) -> &'static str {
        match self {
            GpuBackend::Cuda => "cuda",
            GpuBackend::Metal => "metal",
        }
    }
}

fn runtime_backend() -> GpuBackend {
    if cfg!(feature = "cuda") {
        GpuBackend::Cuda
    } else {
        GpuBackend::Metal
    }
}

fn worker_telemetry_ordinal(backend: GpuBackend, ordinal: usize) -> Option<usize> {
    let _ = backend;
    // Resource snapshots are projected back onto process-local ordinals by
    // UUID before publication. Never translate to a physical NVML ordinal.
    Some(ordinal)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicBool, AtomicUsize};

    fn worker(ordinal: usize) -> Arc<crate::gpu_pool::GpuWorker> {
        let (job_tx, _job_rx) = std::sync::mpsc::sync_channel(1);
        Arc::new(crate::gpu_pool::GpuWorker {
            owner_epoch: 1,
            gpu: mold_inference::device::DiscoveredGpu {
                ordinal,
                stable_id: Some(format!("cuda:{ordinal:032x}")),
                raw_cuda_uuid: Some((ordinal as u128).to_be_bytes()),
                device_kind: Some(mold_inference::device::CudaDeviceKind::UnknownCuda),
                identity_error: None,
                backend: GpuBackend::Cuda,
                name: format!("gpu{ordinal}"),
                compute_capability: Some((8, 6)),
                pci_bus_id: None,
                total_vram_bytes: 24_000_000_000,
                free_vram_bytes: 24_000_000_000,
            },
            model_cache: Arc::new(Mutex::new(crate::model_cache::ModelCache::new(3))),
            resident_model: Arc::new(RwLock::new(None)),
            resident_execution_fingerprint: Arc::new(RwLock::new(None)),
            active_generation: Arc::new(RwLock::new(None)),
            model_load_lock: Arc::new(Mutex::new(())),
            shared_pool: Arc::new(Mutex::new(mold_inference::shared_pool::SharedPool::new())),
            legacy_pending: AtomicUsize::new(0),
            in_flight: AtomicUsize::new(0),
            legacy_chain_waiters: Default::default(),
            consecutive_failures: AtomicUsize::new(0),
            poisoned: AtomicBool::new(false),
            fatal_cuda_error: Arc::new(AtomicBool::new(false)),
            fatal_cuda_shutdown: Arc::new(tokio::sync::Notify::new()),
            shutdown_requested: AtomicBool::new(false),
            drain_state: std::sync::atomic::AtomicU8::new(crate::gpu_pool::DRAIN_RUNNING),
            owner_thread_id: std::sync::OnceLock::new(),
            degraded_until: RwLock::new(None),
            job_tx,
        })
    }

    #[test]
    fn runtime_cuda_discovery_preserves_stable_identity_and_metadata() {
        let gpu = mold_inference::device::DiscoveredGpu {
            ordinal: 3,
            stable_id: Some("cuda:0123456789abcdef0123456789abcdef".into()),
            raw_cuda_uuid: Some([
                0x01, 0x23, 0x45, 0x67, 0x89, 0xab, 0xcd, 0xef, 0x01, 0x23, 0x45, 0x67, 0x89, 0xab,
                0xcd, 0xef,
            ]),
            device_kind: Some(mold_inference::device::CudaDeviceKind::FullGpu),
            identity_error: None,
            backend: GpuBackend::Cuda,
            name: "NVIDIA RTX 3090".into(),
            compute_capability: Some((8, 6)),
            pci_bus_id: Some("00000000:01:00.0".into()),
            total_vram_bytes: 24_000_000_000,
            free_vram_bytes: 20_000_000_000,
        };

        let telemetry = crate::resources::TelemetryTarget::cuda(
            7,
            gpu.raw_cuda_uuid.unwrap(),
            mold_inference::device::CudaDeviceKind::FullGpu,
            gpu.name.clone(),
            gpu.total_vram_bytes,
        );
        let mapped = DiscoveredDevice::from_runtime_gpu(&gpu, true, Some(&telemetry));

        assert_eq!(mapped.stable_id, gpu.stable_id);
        assert_eq!(mapped.backend, GpuBackend::Cuda);
        assert_eq!(mapped.visible_ordinal, Some(3));
        assert_eq!(mapped.device_kind, DeviceKind::FullGpu);
        assert_eq!(mapped.compute_capability, Some((8, 6)));
        assert_eq!(mapped.pci_bus_id.as_deref(), Some("00000000:01:00.0"));
        assert!(mapped.startup_allowed);
        assert_eq!(mapped.telemetry_ordinal, Some(7));
    }

    #[test]
    fn mig_registry_fixture_keeps_unavailable_parent_and_profile_metadata_null() {
        let raw_uuid = [0xaa; 16];
        let gpu = mold_inference::device::DiscoveredGpu {
            ordinal: 0,
            stable_id: Some(mold_inference::device::stable_cuda_id(raw_uuid)),
            raw_cuda_uuid: Some(raw_uuid),
            device_kind: Some(mold_inference::device::CudaDeviceKind::Mig),
            identity_error: None,
            backend: GpuBackend::Cuda,
            name: "NVIDIA B200 MIG 1g.23gb".into(),
            compute_capability: Some((10, 0)),
            pci_bus_id: Some("00000000:01:00.0".into()),
            total_vram_bytes: 23 * 1024 * 1024 * 1024,
            free_vram_bytes: 20 * 1024 * 1024 * 1024,
        };
        let mut telemetry = crate::resources::TelemetryTarget::cuda(
            0,
            raw_uuid,
            mold_inference::device::CudaDeviceKind::Mig,
            gpu.name.clone(),
            gpu.total_vram_bytes,
        );
        telemetry.nvml_uuid = Some("MIG-aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa".into());
        telemetry.mig_uuid = telemetry.nvml_uuid.clone();

        let mapped = DiscoveredDevice::from_runtime_gpu(&gpu, true, Some(&telemetry));

        assert_eq!(mapped.device_kind, DeviceKind::Mig);
        assert_eq!(
            mapped.mig_uuid.as_deref(),
            Some("MIG-aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa")
        );
        assert_eq!(mapped.nvml_uuid, mapped.mig_uuid);
        assert!(
            mapped.physical_uuid.is_none(),
            "a MIG UUID must not be presented as its physical parent UUID"
        );
        assert!(
            mapped.mig_parent_uuid.is_none(),
            "nvml-wrapper 0.10 cannot resolve the MIG parent; never guess it"
        );
        assert!(
            mapped.mig_profile.is_none(),
            "nvml-wrapper 0.10 cannot resolve the MIG profile; never guess it"
        );
    }

    fn discovered(stable_id: Option<&str>, startup_allowed: bool) -> DiscoveredDevice {
        DiscoveredDevice {
            stable_id: stable_id.map(str::to_string),
            backend: GpuBackend::Cuda,
            visible_ordinal: Some(0),
            device_kind: DeviceKind::FullGpu,
            nvml_uuid: None,
            physical_uuid: None,
            mig_uuid: None,
            mig_parent_uuid: None,
            mig_profile: None,
            pci_bus_id: None,
            name: "test device".into(),
            compute_capability: Some((8, 6)),
            total_memory_bytes: Some(24_000_000_000),
            startup_allowed,
            telemetry_ordinal: Some(0),
        }
    }

    #[test]
    fn db_disabled_preferences_are_process_local() {
        let registry =
            DeviceRegistry::new(Arc::new(StaticDeviceDiscovery::default()), Arc::new(None));
        let id = "cuda:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa";

        registry.set_desired_enabled(id, false).unwrap();
        assert_eq!(
            registry
                .explicit_preferences
                .read()
                .unwrap()
                .get(id)
                .copied(),
            Some(false)
        );
    }

    #[test]
    fn legacy_projection_contains_only_schedulable_devices() {
        let registry = DeviceRegistry::new(
            Arc::new(StaticDeviceDiscovery::new(vec![
                discovered(Some("cuda:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"), false),
                DiscoveredDevice {
                    stable_id: Some("cuda:bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb".into()),
                    visible_ordinal: Some(1),
                    name: "active GPU".into(),
                    telemetry_ordinal: Some(1),
                    ..discovered(Some("cuda:bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb"), true)
                },
                DiscoveredDevice {
                    stable_id: Some("cuda:cccccccccccccccccccccccccccccccc".into()),
                    visible_ordinal: Some(2),
                    name: "missing worker".into(),
                    telemetry_ordinal: Some(2),
                    ..discovered(Some("cuda:cccccccccccccccccccccccccccccccc"), true)
                },
            ])),
            Arc::new(None),
        );
        let pool = crate::gpu_pool::GpuPool {
            workers: vec![worker(1)].into(),
        };
        let state = registry.snapshot(
            &pool,
            None,
            &crate::job_registry::JobRegistry::with_events(crate::events::EventBroadcaster::new()),
        );

        assert_eq!(state.devices.len(), 3, "full inventory remains visible");
        assert_eq!(
            DeviceRegistry::legacy_gpu_info(&state)
                .as_ref()
                .map(|gpu| gpu.name.as_str()),
            Some("active GPU")
        );
        let workers = DeviceRegistry::legacy_gpu_status_from_snapshot(&state);
        assert_eq!(workers.len(), 1);
        assert_eq!(workers[0].ordinal, 1);
    }

    #[test]
    fn loads_explicit_preferences_but_missing_devices_default_enabled() {
        let db = mold_db::MetadataDb::open_in_memory().unwrap();
        mold_db::DevicePreferences::new(&db)
            .set("cuda:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa", false)
            .unwrap();
        let registry = DeviceRegistry::new(
            Arc::new(StaticDeviceDiscovery::default()),
            Arc::new(Some(db)),
        );
        let preferences = registry.explicit_preferences.read().unwrap();

        assert_eq!(
            preferences
                .get("cuda:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa")
                .copied(),
            Some(false)
        );
        assert!(preferences
            .get("cuda:bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb")
            .copied()
            .unwrap_or(true));
    }

    #[test]
    fn startup_exclusion_wins_over_enabled_by_default() {
        let registry = DeviceRegistry::new(
            Arc::new(StaticDeviceDiscovery::new(vec![discovered(
                Some("cuda:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"),
                false,
            )])),
            Arc::new(None),
        );
        let pool = GpuPool {
            workers: Vec::new().into(),
        };
        let jobs = crate::job_registry::JobRegistry::new();

        let state = registry.snapshot(&pool, None, &jobs);
        assert_eq!(state.devices.len(), 1);
        assert!(state.devices[0].desired_enabled);
        assert_eq!(
            state.devices[0].admin_state,
            DeviceAdminState::StartupExcluded
        );
        assert!(!state.devices[0].schedulable);
        assert_eq!(
            state.devices[0].unschedulable_reason.as_deref(),
            Some("device_startup_excluded")
        );
    }

    #[test]
    fn missing_stable_identity_is_visible_unavailable_and_process_stable() {
        let registry = DeviceRegistry::new(
            Arc::new(StaticDeviceDiscovery::new(vec![discovered(None, true)])),
            Arc::new(None),
        );
        let pool = GpuPool {
            workers: Vec::new().into(),
        };
        let jobs = crate::job_registry::JobRegistry::new();

        let first = registry.snapshot(&pool, None, &jobs);
        let second = registry.snapshot(&pool, None, &jobs);
        assert_eq!(first.devices[0].id, second.devices[0].id);
        assert_eq!(first.devices[0].health, DeviceHealth::Unavailable);
        assert!(!first.devices[0].schedulable);
    }

    #[test]
    fn legacy_gpu_info_preserves_mebibytes_for_real_24_gib_device() {
        let state = DeviceState {
            devices: vec![DeviceInfo {
                id: "cuda:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa".into(),
                backend: GpuBackend::Cuda,
                ordinal: Some(0),
                device_kind: DeviceKind::FullGpu,
                nvml_uuid: Some("GPU-aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa".into()),
                physical_uuid: Some("GPU-aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa".into()),
                mig_uuid: None,
                mig_parent_uuid: None,
                mig_profile: None,
                name: "NVIDIA GeForce RTX 3090".into(),
                pci_bus_id: None,
                compute_capability: Some("8.6".into()),
                memory: DeviceMemoryInfo {
                    total_bytes: Some(24 * 1024 * 1024 * 1024),
                    used_bytes: Some(8 * 1024 * 1024 * 1024),
                    mold_used_bytes: None,
                    other_used_bytes: None,
                },
                telemetry: DeviceTelemetry {
                    utilization_percent: None,
                    temperature_c: None,
                    power_w: None,
                },
                desired_enabled: true,
                restart_required: false,
                admin_state: DeviceAdminState::Enabled,
                health: DeviceHealth::Healthy,
                activity: DeviceActivity::Idle,
                schedulable: true,
                unschedulable_reason: None,
                loaded_models: Vec::new(),
                active_work_id: None,
                planned_work_ids: Vec::new(),
            }],
            plan_version: 0,
        };

        let legacy = DeviceRegistry::legacy_gpu_info(&state).unwrap();
        assert_eq!(legacy.vram_total_mb, 24_576);
        assert_eq!(legacy.vram_used_mb, 8_192);
    }
}
