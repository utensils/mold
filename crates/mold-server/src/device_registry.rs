//! Canonical runtime device registry.
//!
//! CUDA/Metal discovery is consumed exactly once at server startup. The
//! registry retains the immutable identity, worker-construction, and telemetry
//! join facts and owns the versioned desired/administrative/health/activity
//! projection. Worker objects and join handles remain on their owner-thread
//! boundary, but worker construction, scheduler candidates, telemetry, the
//! device API, and legacy status all derive from this registry. No request
//! path performs CUDA or NVML discovery.

use std::collections::{BTreeMap, BTreeSet};
use std::sync::atomic::{AtomicU64, Ordering};
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

#[derive(Clone)]
struct CanonicalDeviceRecord {
    id: String,
    discovered: DiscoveredDevice,
    construction: Option<mold_inference::device::DiscoveredGpu>,
    telemetry: Option<crate::resources::TelemetryTarget>,
}

#[derive(Debug, Clone, PartialEq)]
pub(crate) struct SchedulerDeviceProjection {
    pub id: String,
    pub backend: GpuBackend,
    pub ordinal: usize,
    pub compute_capability: Option<(u16, u16)>,
    pub admin_state: DeviceAdminState,
    pub health: DeviceHealth,
    pub activity: DeviceActivity,
    pub schedulable: bool,
    pub sampled_free_vram_bytes: u64,
    pub sampled_mold_vram_bytes: Option<u64>,
    pub discovered_free_vram_bytes: u64,
}

#[derive(Debug, Clone, PartialEq)]
pub(crate) struct CanonicalDeviceSnapshot {
    pub revision: u64,
    pub device_state: DeviceState,
    pub scheduler_devices: Vec<SchedulerDeviceProjection>,
}

pub struct DeviceRegistry {
    records: Vec<CanonicalDeviceRecord>,
    explicit_preferences: RwLock<BTreeMap<String, bool>>,
    transient_unavailable: RwLock<BTreeSet<String>>,
    metadata_db: Arc<Option<mold_db::MetadataDb>>,
    projection_guard: Mutex<()>,
    last_device_state: RwLock<Option<DeviceState>>,
    mutation_sequence: AtomicU64,
}

impl DeviceRegistry {
    pub fn new(
        discovery: Arc<dyn DeviceDiscoveryAdapter>,
        metadata_db: Arc<Option<mold_db::MetadataDb>>,
    ) -> Self {
        let records = discovery
            .devices()
            .into_iter()
            .map(|discovered| {
                let id = discovered.stable_id.clone().unwrap_or_else(|| {
                    format!(
                        "{}:unavailable-{}",
                        discovered.backend.wire_name(),
                        uuid::Uuid::new_v4().simple()
                    )
                });
                let telemetry = discovered.visible_ordinal.map(|logical_ordinal| {
                    crate::resources::TelemetryTarget {
                        logical_ordinal,
                        backend: discovered.backend,
                        raw_cuda_uuid: None,
                        cuda_kind: match discovered.device_kind {
                            DeviceKind::FullGpu => {
                                Some(mold_inference::device::CudaDeviceKind::FullGpu)
                            }
                            DeviceKind::Mig => Some(mold_inference::device::CudaDeviceKind::Mig),
                            DeviceKind::UnknownCuda => {
                                Some(mold_inference::device::CudaDeviceKind::UnknownCuda)
                            }
                            DeviceKind::Metal => None,
                        },
                        name: discovered.name.clone(),
                        total_memory_bytes: discovered.total_memory_bytes.unwrap_or(0),
                        nvml_uuid: discovered.nvml_uuid.clone(),
                        physical_uuid: discovered.physical_uuid.clone(),
                        mig_uuid: discovered.mig_uuid.clone(),
                        pci_bus_id: discovered.pci_bus_id.clone(),
                    }
                });
                CanonicalDeviceRecord {
                    id,
                    discovered,
                    construction: None,
                    telemetry,
                }
            })
            .collect();

        Self::with_records(records, metadata_db)
    }

    fn with_records(
        records: Vec<CanonicalDeviceRecord>,
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
            records,
            explicit_preferences: RwLock::new(explicit_preferences),
            transient_unavailable: RwLock::new(BTreeSet::new()),
            metadata_db,
            projection_guard: Mutex::new(()),
            last_device_state: RwLock::new(None),
            mutation_sequence: AtomicU64::new(0),
        }
    }

    /// Consume the process-visible discovery result and startup selection into
    /// the single immutable registry inventory. CUDA/NVML enrichment happens
    /// here, before any owner thread or request handler exists.
    pub(crate) fn from_runtime_inventory(
        discovered: Vec<mold_inference::device::DiscoveredGpu>,
        selected: &[mold_inference::device::DiscoveredGpu],
        metadata_db: Arc<Option<mold_db::MetadataDb>>,
    ) -> Self {
        let telemetry = crate::resources::discover_telemetry_targets(&discovered);
        Self::from_inventory_with_targets(discovered, selected, telemetry, metadata_db)
    }

    /// Build a registry around an already-created worker pool without
    /// repeating NVML enrichment. Production startup passes its original
    /// registry explicitly; this adapter is for embedding/tests that supply a
    /// preconstructed pool.
    pub(crate) fn from_worker_inventory(
        discovered: Vec<mold_inference::device::DiscoveredGpu>,
        metadata_db: Arc<Option<mold_db::MetadataDb>>,
    ) -> Self {
        let telemetry = discovered
            .iter()
            .map(crate::resources::TelemetryTarget::from_discovered)
            .collect();
        let selected = discovered.clone();
        Self::from_inventory_with_targets(discovered, &selected, telemetry, metadata_db)
    }

    fn from_inventory_with_targets(
        discovered: Vec<mold_inference::device::DiscoveredGpu>,
        selected: &[mold_inference::device::DiscoveredGpu],
        telemetry: Vec<crate::resources::TelemetryTarget>,
        metadata_db: Arc<Option<mold_db::MetadataDb>>,
    ) -> Self {
        let selected = selected
            .iter()
            .map(runtime_device_key)
            .collect::<BTreeSet<_>>();
        let inventory = discovered
            .into_iter()
            .map(|gpu| {
                let target = telemetry
                    .iter()
                    .find(|target| {
                        target.backend == gpu.backend && target.logical_ordinal == gpu.ordinal
                    })
                    .cloned();
                let startup_allowed = selected.contains(&runtime_device_key(&gpu));
                let discovered =
                    DiscoveredDevice::from_runtime_gpu(&gpu, startup_allowed, target.as_ref());
                let id = discovered.stable_id.clone().unwrap_or_else(|| {
                    format!(
                        "{}:unavailable-{}",
                        discovered.backend.wire_name(),
                        uuid::Uuid::new_v4().simple()
                    )
                });
                CanonicalDeviceRecord {
                    id,
                    discovered,
                    construction: Some(gpu),
                    telemetry: target,
                }
            })
            .collect::<Vec<_>>();
        Self::with_records(inventory, metadata_db)
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
    /// Persist a device preference and return whether its effective value
    /// changed. The write lock spans persistence so concurrent callers cannot
    /// both publish the same logical transition.
    pub fn set_desired_enabled(&self, device_id: &str, enabled: bool) -> anyhow::Result<bool> {
        let _projection = self
            .projection_guard
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        let mut preferences = self
            .explicit_preferences
            .write()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        if preferences.get(device_id).copied().unwrap_or(true) == enabled {
            return Ok(false);
        }
        if let Some(db) = self.metadata_db.as_ref().as_ref() {
            mold_db::DevicePreferences::new(db).set(device_id, enabled)?;
        } else {
            tracing::warn!(
                device_id,
                desired_enabled = enabled,
                "metadata DB disabled; device preference will not persist"
            );
        }
        preferences.insert(device_id.to_string(), enabled);
        self.mutation_sequence.fetch_add(1, Ordering::SeqCst);
        Ok(true)
    }

    pub fn discovered_device(&self, device_id: &str) -> Option<DiscoveredDevice> {
        self.records
            .iter()
            .find(|record| record.id == device_id)
            .map(|record| record.discovered.clone())
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
        !self.records.is_empty()
    }

    pub fn mutation_sequence(&self) -> u64 {
        self.mutation_sequence.load(Ordering::SeqCst)
    }

    pub(crate) fn visible_device_count(&self) -> usize {
        self.records.len()
    }

    pub(crate) fn startup_allowed_count(&self) -> usize {
        self.records
            .iter()
            .filter(|record| record.discovered.startup_allowed)
            .count()
    }

    /// Complete startup-selected construction catalog. Persisted-disabled
    /// devices remain here so V2 can start a fresh owner without rediscovery.
    #[cfg(test)]
    pub(crate) fn worker_constructions(
        &self,
    ) -> BTreeMap<String, mold_inference::device::DiscoveredGpu> {
        self.records
            .iter()
            .filter(|record| record.discovered.startup_allowed)
            .filter_map(|record| {
                record
                    .construction
                    .clone()
                    .map(|gpu| (record.id.clone(), gpu))
            })
            .collect()
    }

    pub(crate) fn worker_construction(
        &self,
        device_id: &str,
    ) -> Option<mold_inference::device::DiscoveredGpu> {
        self.records
            .iter()
            .find(|record| record.id == device_id && record.discovered.startup_allowed)
            .and_then(|record| record.construction.clone())
    }

    pub(crate) fn startup_worker_devices(&self) -> Vec<mold_inference::device::DiscoveredGpu> {
        self.records
            .iter()
            .filter(|record| record.discovered.startup_allowed)
            .filter(|record| self.desired_enabled(&record.id))
            .filter_map(|record| record.construction.clone())
            .collect()
    }

    pub(crate) fn persisted_disabled_worker_devices(
        &self,
    ) -> Vec<mold_inference::device::DiscoveredGpu> {
        self.records
            .iter()
            .filter(|record| record.discovered.startup_allowed)
            .filter(|record| !self.desired_enabled(&record.id))
            .filter_map(|record| record.construction.clone())
            .collect()
    }

    /// Driver-free telemetry join targets cloned from canonical startup facts.
    pub(crate) fn telemetry_targets(&self) -> Vec<crate::resources::TelemetryTarget> {
        self.records
            .iter()
            .filter_map(|record| record.telemetry.clone())
            .collect()
    }

    /// Record a scheduler transport/start failure in the authoritative device
    /// projection. Returns true only for a real health transition.
    pub fn mark_unavailable(&self, device_id: &str) -> bool {
        let _projection = self
            .projection_guard
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        let changed = self
            .transient_unavailable
            .write()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .insert(device_id.to_string());
        if changed {
            self.mutation_sequence.fetch_add(1, Ordering::SeqCst);
        }
        changed
    }

    /// Clear a transient scheduler failure when the worker announces Ready.
    /// Returns true only for a real health transition.
    pub fn mark_available(&self, device_id: &str) -> bool {
        let _projection = self
            .projection_guard
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        let changed = self
            .transient_unavailable
            .write()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .remove(device_id);
        if changed {
            self.mutation_sequence.fetch_add(1, Ordering::SeqCst);
        }
        changed
    }

    pub fn snapshot(
        &self,
        pool: &GpuPool,
        resources: Option<&ResourceSnapshot>,
        jobs: &SharedJobRegistry,
    ) -> DeviceState {
        self.canonical_snapshot(pool, resources, jobs).device_state
    }

    pub(crate) fn canonical_snapshot(
        &self,
        pool: &GpuPool,
        resources: Option<&ResourceSnapshot>,
        jobs: &SharedJobRegistry,
    ) -> CanonicalDeviceSnapshot {
        let _projection = self
            .projection_guard
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        let workers = pool.worker_snapshot();
        let queue = jobs.snapshot();
        let preferences = self
            .explicit_preferences
            .read()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        let transient_unavailable = self
            .transient_unavailable
            .read()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        let mut records = self.records.iter().collect::<Vec<_>>();
        records.sort_by(|left, right| {
            left.discovered
                .backend
                .wire_name()
                .cmp(right.discovered.backend.wire_name())
                .then(
                    left.discovered
                        .visible_ordinal
                        .cmp(&right.discovered.visible_ordinal),
                )
                .then(left.discovered.name.cmp(&right.discovered.name))
                .then(left.id.cmp(&right.id))
        });
        let projected = records
            .into_iter()
            .map(|record| {
                let device = &record.discovered;
                let stable_identity = device.stable_id.is_some();
                let id = record.id.clone();
                let desired_enabled = preferences.get(&id).copied().unwrap_or(true);
                let worker_ref = workers.iter().find(|worker| {
                    if let Some(stable_id) = device.stable_id.as_deref() {
                        worker.gpu.stable_id.as_deref() == Some(stable_id)
                    } else {
                        device.visible_ordinal == Some(worker.gpu.ordinal)
                            && device.backend == worker.gpu.backend
                    }
                });
                let worker = worker_ref.map(|worker| worker.status());
                let telemetry = device.telemetry_ordinal.and_then(|ordinal| {
                    resources.and_then(|snapshot| {
                        snapshot
                            .gpus
                            .iter()
                            .find(|gpu| gpu.backend == device.backend && gpu.ordinal == ordinal)
                    })
                });
                let active = worker.as_ref().is_some_and(|status| {
                    !matches!(
                        status.state,
                        GpuWorkerState::Idle | GpuWorkerState::Degraded
                    )
                });
                let drain_state = worker_ref
                    .map(|worker| worker.drain_state.load(std::sync::atomic::Ordering::SeqCst));
                let admin_state = if !device.startup_allowed {
                    DeviceAdminState::StartupExcluded
                } else if pool.workers.is_starting(&id)
                    || (desired_enabled && drain_state == Some(crate::gpu_pool::DRAIN_COMMITTED))
                {
                    DeviceAdminState::Starting
                } else if drain_state.is_some_and(|state| state != crate::gpu_pool::DRAIN_RUNNING)
                    || (!desired_enabled && worker_ref.is_some())
                {
                    DeviceAdminState::Draining
                } else if desired_enabled {
                    DeviceAdminState::Enabled
                } else if active {
                    DeviceAdminState::Draining
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
                        } else if transient_unavailable.contains(&id) {
                            DeviceHealth::Unavailable
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
                    .as_ref()
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
                    .as_ref()
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

                let info = DeviceInfo {
                    id: id.clone(),
                    backend: device.backend,
                    ordinal: device.visible_ordinal,
                    device_kind: device.device_kind,
                    nvml_uuid: device.nvml_uuid.clone(),
                    physical_uuid: device.physical_uuid.clone(),
                    mig_uuid: device.mig_uuid.clone(),
                    mig_parent_uuid: device.mig_parent_uuid.clone(),
                    mig_profile: device.mig_profile.clone(),
                    name: device.name.clone(),
                    pci_bus_id: device.pci_bus_id.clone(),
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
                };
                let scheduler = device.visible_ordinal.map(|ordinal| {
                    let total = info.memory.total_bytes.unwrap_or(0);
                    let discovered_free_vram_bytes = record
                        .construction
                        .as_ref()
                        .map_or(0, |gpu| gpu.free_vram_bytes);
                    SchedulerDeviceProjection {
                        id,
                        backend: device.backend,
                        ordinal,
                        compute_capability: device.compute_capability,
                        admin_state: info.admin_state,
                        health: info.health,
                        activity: info.activity,
                        schedulable: info.schedulable,
                        sampled_free_vram_bytes: info
                            .memory
                            .used_bytes
                            .map_or(discovered_free_vram_bytes, |used| {
                                total.saturating_sub(used)
                            }),
                        sampled_mold_vram_bytes: info.memory.mold_used_bytes,
                        discovered_free_vram_bytes,
                    }
                });
                (info, scheduler)
            })
            .collect::<Vec<_>>();
        let device_state = DeviceState {
            devices: projected.iter().map(|(device, _)| device.clone()).collect(),
            plan_version: 0,
        };
        let scheduler_devices = projected
            .into_iter()
            .filter_map(|(_, scheduler)| scheduler)
            .collect();
        let changed = self
            .last_device_state
            .read()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .as_ref()
            != Some(&device_state);
        if changed {
            *self
                .last_device_state
                .write()
                .unwrap_or_else(|poisoned| poisoned.into_inner()) = Some(device_state.clone());
            self.mutation_sequence.fetch_add(1, Ordering::SeqCst);
        }
        CanonicalDeviceSnapshot {
            revision: self.mutation_sequence.load(Ordering::SeqCst),
            device_state,
            scheduler_devices,
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

fn runtime_device_key(
    gpu: &mold_inference::device::DiscoveredGpu,
) -> (GpuBackend, usize, Option<String>) {
    (gpu.backend, gpu.ordinal, gpu.stable_id.clone())
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

        assert_eq!(registry.mutation_sequence(), 0);
        registry.set_desired_enabled(id, false).unwrap();
        assert_eq!(registry.mutation_sequence(), 1);
        assert_eq!(
            registry
                .explicit_preferences
                .read()
                .unwrap()
                .get(id)
                .copied(),
            Some(false)
        );

        registry.set_desired_enabled(id, false).unwrap();
        assert_eq!(
            registry.mutation_sequence(),
            1,
            "repeating the same preference is not a semantic mutation"
        );
    }

    #[test]
    fn legacy_projection_contains_only_schedulable_devices() {
        let registry = DeviceRegistry::new(
            Arc::new(StaticDeviceDiscovery::new(vec![
                discovered(Some("cuda:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"), false),
                DiscoveredDevice {
                    stable_id: Some("cuda:00000000000000000000000000000001".into()),
                    visible_ordinal: Some(1),
                    name: "active GPU".into(),
                    telemetry_ordinal: Some(1),
                    ..discovered(Some("cuda:00000000000000000000000000000001"), true)
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
    fn transient_worker_unavailability_is_authoritative_until_ready() {
        let id = "cuda:00000000000000000000000000000000";
        let registry = DeviceRegistry::new(
            Arc::new(StaticDeviceDiscovery::new(vec![discovered(Some(id), true)])),
            Arc::new(None),
        );
        let pool = GpuPool {
            workers: vec![worker(0)].into(),
        };
        let jobs = crate::job_registry::JobRegistry::new();

        assert_eq!(
            registry.snapshot(&pool, None, &jobs).devices[0].health,
            DeviceHealth::Healthy
        );
        assert!(registry.mark_unavailable(id));
        let unavailable = registry.snapshot(&pool, None, &jobs);
        assert_eq!(unavailable.devices[0].health, DeviceHealth::Unavailable);
        assert!(!unavailable.devices[0].schedulable);
        assert!(
            !registry.mark_unavailable(id),
            "idempotent transitions stay silent"
        );
        assert!(registry.mark_available(id));
        assert_eq!(
            registry.snapshot(&pool, None, &jobs).devices[0].health,
            DeviceHealth::Healthy
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

    fn runtime_gpu(
        ordinal: usize,
        uuid_byte: u8,
        kind: mold_inference::device::CudaDeviceKind,
    ) -> mold_inference::device::DiscoveredGpu {
        let raw_uuid = [uuid_byte; 16];
        mold_inference::device::DiscoveredGpu {
            ordinal,
            stable_id: Some(mold_inference::device::stable_cuda_id(raw_uuid)),
            raw_cuda_uuid: Some(raw_uuid),
            device_kind: Some(kind),
            identity_error: None,
            backend: GpuBackend::Cuda,
            name: format!("synthetic GPU {uuid_byte}"),
            compute_capability: Some((10, 0)),
            pci_bus_id: Some(format!("00000000:{ordinal:02x}:00.0")),
            total_vram_bytes: 192 << 30,
            free_vram_bytes: 180 << 30,
        }
    }

    #[test]
    fn one_runtime_inventory_drives_worker_factory_telemetry_and_sixty_four_devices() {
        let discovered = (0..64)
            .map(|ordinal| {
                runtime_gpu(
                    ordinal,
                    (ordinal + 1) as u8,
                    match ordinal % 3 {
                        0 => mold_inference::device::CudaDeviceKind::Mig,
                        1 => mold_inference::device::CudaDeviceKind::FullGpu,
                        _ => mold_inference::device::CudaDeviceKind::UnknownCuda,
                    },
                )
            })
            .collect::<Vec<_>>();
        let disabled_id = discovered[42].stable_id.clone().unwrap();
        let db = mold_db::MetadataDb::open_in_memory().unwrap();
        mold_db::DevicePreferences::new(&db)
            .set(&disabled_id, false)
            .unwrap();

        let registry = DeviceRegistry::from_runtime_inventory(
            discovered.clone(),
            &discovered,
            Arc::new(Some(db)),
        );
        let construction = registry.worker_constructions();
        let telemetry = registry.telemetry_targets();

        assert_eq!(registry.visible_device_count(), 64);
        assert_eq!(registry.startup_allowed_count(), 64);
        assert_eq!(construction.len(), 64);
        assert_eq!(telemetry.len(), 64);
        assert_eq!(registry.startup_worker_devices().len(), 63);
        assert!(construction.contains_key(&disabled_id));
        for gpu in &discovered {
            let id = gpu.stable_id.as_deref().unwrap();
            assert_eq!(construction[id].ordinal, gpu.ordinal);
            let target = telemetry
                .iter()
                .find(|target| target.logical_ordinal == gpu.ordinal)
                .expect("every canonical construction has one telemetry target");
            assert_eq!(target.raw_cuda_uuid, gpu.raw_cuda_uuid);
            assert_eq!(target.cuda_kind, gpu.device_kind);
        }
    }

    #[test]
    fn stable_uuid_keeps_construction_identity_when_visible_ordinals_reorder() {
        let gpu_a = runtime_gpu(1, 0xaa, mold_inference::device::CudaDeviceKind::FullGpu);
        let gpu_b = runtime_gpu(0, 0xbb, mold_inference::device::CudaDeviceKind::UnknownCuda);
        let selected = vec![gpu_b.clone(), gpu_a.clone()];
        let registry = DeviceRegistry::from_runtime_inventory(
            vec![gpu_a.clone(), gpu_b.clone()],
            &selected,
            Arc::new(None),
        );
        let construction = registry.worker_constructions();

        assert_eq!(construction[gpu_a.stable_id.as_deref().unwrap()].ordinal, 1);
        assert_eq!(construction[gpu_b.stable_id.as_deref().unwrap()].ordinal, 0);
    }

    #[test]
    fn metal_default_uses_the_same_construction_and_telemetry_registry() {
        let metal = mold_inference::device::DiscoveredGpu {
            ordinal: 0,
            stable_id: Some("metal:default".to_string()),
            raw_cuda_uuid: None,
            device_kind: None,
            identity_error: None,
            backend: GpuBackend::Metal,
            name: "Apple GPU".to_string(),
            compute_capability: None,
            pci_bus_id: None,
            total_vram_bytes: 32 << 30,
            free_vram_bytes: 28 << 30,
        };
        let registry = DeviceRegistry::from_runtime_inventory(
            vec![metal.clone()],
            std::slice::from_ref(&metal),
            Arc::new(None),
        );

        assert_eq!(
            registry.worker_constructions()["metal:default"].backend,
            GpuBackend::Metal
        );
        let target = registry.telemetry_targets().pop().unwrap();
        assert_eq!(target.backend, GpuBackend::Metal);
        assert_eq!(target.raw_cuda_uuid, None);
        assert_eq!(
            registry
                .discovered_device("metal:default")
                .unwrap()
                .device_kind,
            DeviceKind::Metal
        );
    }

    #[test]
    fn lifecycle_mutation_publishes_one_coherent_api_and_scheduler_projection() {
        let gpu = runtime_gpu(0, 0, mold_inference::device::CudaDeviceKind::FullGpu);
        let registry = DeviceRegistry::from_runtime_inventory(
            vec![gpu.clone()],
            std::slice::from_ref(&gpu),
            Arc::new(None),
        );
        let worker = worker(0);
        let pool = GpuPool {
            workers: vec![worker.clone()].into(),
        };
        let jobs = crate::job_registry::JobRegistry::new();

        let initial = registry.canonical_snapshot(&pool, None, &jobs);
        assert_eq!(initial.device_state.devices.len(), 1);
        assert_eq!(initial.scheduler_devices.len(), 1);
        assert_eq!(
            initial.device_state.devices[0].id,
            initial.scheduler_devices[0].id
        );
        assert_eq!(
            initial.device_state.devices[0].admin_state,
            initial.scheduler_devices[0].admin_state
        );
        assert_eq!(
            initial.device_state.devices[0].health,
            initial.scheduler_devices[0].health
        );

        registry
            .set_desired_enabled(gpu.stable_id.as_deref().unwrap(), false)
            .unwrap();
        worker.in_flight.store(1, Ordering::SeqCst);
        let draining = registry.canonical_snapshot(&pool, None, &jobs);
        assert!(draining.revision > initial.revision);
        assert_eq!(
            draining.device_state.devices[0].admin_state,
            DeviceAdminState::Draining
        );
        assert_eq!(
            draining.scheduler_devices[0].admin_state,
            DeviceAdminState::Draining
        );
        assert!(!draining.device_state.devices[0].schedulable);
        assert!(!draining.scheduler_devices[0].schedulable);
    }
}
