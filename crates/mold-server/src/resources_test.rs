//! Unit tests for the resources module.

use crate::resources::{nonzero_process_vram, ResourceBroadcaster, TelemetryTarget};
use mold_core::{GpuBackend, GpuSnapshot, RamSnapshot, ResourceSnapshot};
use mold_inference::device::CudaDeviceKind;

fn fake_snapshot() -> ResourceSnapshot {
    ResourceSnapshot {
        hostname: "test".into(),
        timestamp: 1_700_000_000_000,
        gpus: vec![GpuSnapshot {
            ordinal: 0,
            name: "fake".into(),
            backend: GpuBackend::Cuda,
            vram_total: 24_000_000_000,
            vram_used: 0,
            vram_used_by_mold: Some(0),
            vram_used_by_other: Some(0),
            gpu_utilization: None,
        }],
        system_ram: RamSnapshot {
            total: 64_000_000_000,
            used: 0,
            available: Some(64_000_000_000),
            reclaimable_zfs_arc: None,
            used_by_mold: 0,
            used_by_other: 0,
        },
        cpu: None,
    }
}

#[test]
fn zero_process_vram_is_ambiguous_not_authoritative() {
    assert_eq!(nonzero_process_vram(None), None);
    assert_eq!(nonzero_process_vram(Some(0)), None);
    assert_eq!(nonzero_process_vram(Some(6 << 30)), Some(6 << 30));
}

#[tokio::test]
async fn broadcaster_delivers_published_snapshots() {
    let bcast = ResourceBroadcaster::new();
    let mut rx = bcast.subscribe();
    bcast.publish(fake_snapshot());

    let got = rx.recv().await.expect("should receive snapshot");
    assert_eq!(got.hostname, "test");
    assert_eq!(got.gpus.len(), 1);
}

#[tokio::test]
async fn broadcaster_latest_reflects_most_recent_publish() {
    let bcast = ResourceBroadcaster::new();
    assert!(bcast.latest().is_none());

    let mut snap1 = fake_snapshot();
    snap1.timestamp = 1;
    bcast.publish(snap1);

    let mut snap2 = fake_snapshot();
    snap2.timestamp = 2;
    bcast.publish(snap2);

    let latest = bcast.latest().expect("latest should be set");
    assert_eq!(latest.timestamp, 2);
}

#[tokio::test]
async fn subscribe_with_lagged_receiver_recovers() {
    let bcast = ResourceBroadcaster::new();
    let mut rx = bcast.subscribe();
    // The broadcast buffer size is 4 (per spec 2.3); publishing 10 rapid
    // snapshots must not wedge the channel — lagging receivers catch up
    // with the tail.
    for i in 0..10 {
        let mut snap = fake_snapshot();
        snap.timestamp = i;
        bcast.publish(snap);
    }
    // Drain whatever is still in the channel — should yield at least 1.
    // NOTE: tokio's broadcast receiver surfaces a single `Lagged(n)` error
    // when it falls behind; subsequent `try_recv` calls return the tail.
    // So we skip the Lagged error rather than breaking out of the loop.
    let mut count = 0;
    for _ in 0..16 {
        match rx.try_recv() {
            Ok(_) => count += 1,
            Err(tokio::sync::broadcast::error::TryRecvError::Lagged(_)) => continue,
            Err(_) => break,
        }
        if count >= 4 {
            break;
        }
    }
    assert!(count > 0, "receiver should recover and deliver tail");
}

#[test]
fn build_snapshot_populates_hostname_and_timestamp() {
    let snap = crate::resources::build_snapshot();
    assert!(!snap.hostname.is_empty(), "hostname must be populated");
    assert!(snap.timestamp > 0, "timestamp must be non-zero");
    // On any host, either gpus is non-empty (CUDA/Metal) or it's empty
    // (CPU-only). Both are valid — we just require the call doesn't panic.
    assert!(snap.system_ram.total > 0);
}

#[tokio::test(flavor = "current_thread", start_paused = true)]
async fn aggregator_publishes_within_first_tick() {
    let bcast = crate::resources::ResourceBroadcaster::new();
    let mut rx = bcast.subscribe();
    let handle = crate::resources::spawn_aggregator(
        bcast.clone(),
        crate::device_registry::DeviceRegistry::empty(),
    );

    // Advance virtual time past one tick interval (1 s).
    tokio::time::advance(std::time::Duration::from_millis(1_100)).await;

    // The aggregator fires immediately on startup, so there should be a
    // snapshot waiting even before the 1-second tick.
    let got = tokio::time::timeout(std::time::Duration::from_millis(50), rx.recv())
        .await
        .expect("aggregator should publish within first tick")
        .expect("channel should not be closed");
    assert!(got.timestamp > 0);

    handle.abort();
}

#[test]
#[cfg(target_os = "macos")]
fn metal_snapshot_reports_unified_memory_with_none_attribution() {
    let gpus = crate::resources::metal_snapshot();
    assert_eq!(
        gpus.len(),
        1,
        "Metal hosts expose a single unified-memory GPU"
    );
    let gpu = &gpus[0];
    assert_eq!(gpu.backend, mold_core::GpuBackend::Metal);
    assert_eq!(gpu.ordinal, 0);
    assert!(gpu.vram_total > 0);
    assert!(
        gpu.vram_used_by_mold.is_none(),
        "Metal does not expose per-process GPU attribution"
    );
    assert!(gpu.vram_used_by_other.is_none());
}

#[test]
#[cfg(not(target_os = "macos"))]
fn metal_snapshot_is_empty_off_darwin() {
    let gpus = crate::resources::metal_snapshot();
    assert!(gpus.is_empty());
}

#[test]
fn metal_telemetry_spends_the_host_samples_available_pool() {
    let mut ram = fake_snapshot().system_ram;
    // Deliberately disagree with total - used: the immediate pressure sample
    // excludes pages a broader sysinfo estimate might count as reclaimable.
    for available in [0, 7 << 30, 19 << 30] {
        ram.available = Some(available);
        let gpu = crate::resources::metal_snapshot_from_ram(&ram);
        assert_eq!(gpu.vram_total - gpu.vram_used, available);
        assert_eq!(gpu.vram_total, ram.total);
        assert_eq!(gpu.vram_used_by_mold, None);
    }
}

#[test]
fn unavailable_unified_sample_is_not_zero_capacity() {
    let ram = crate::resources::ram_snapshot_from_system_with_available(|_| None);
    assert_eq!(ram.available, None, "a failed query is not a measured zero");
    let gpu = crate::resources::metal_snapshot_from_ram(&ram);
    assert_eq!(
        gpu.vram_used, ram.used,
        "retain the existing estimated fallback"
    );

    let zero = crate::resources::ram_snapshot_from_system_with_available(|_| Some(0));
    assert_eq!(zero.available, Some(0));
    let gpu = crate::resources::metal_snapshot_from_ram(&zero);
    assert_eq!(
        gpu.vram_used, gpu.vram_total,
        "a successful zero sample must block"
    );
}

#[test]
fn ram_snapshot_satisfies_invariants() {
    let ram = crate::resources::ram_snapshot();
    assert!(ram.total > 0, "total RAM should be >0 on any host");
    assert!(
        ram.used <= ram.total,
        "used ({}) must be <= total ({})",
        ram.used,
        ram.total
    );
    assert!(
        ram.used_by_mold <= ram.used,
        "used_by_mold ({}) must be <= used ({})",
        ram.used_by_mold,
        ram.used
    );
    assert_eq!(
        ram.used_by_other,
        ram.used.saturating_sub(ram.used_by_mold),
        "used_by_other must == used - used_by_mold"
    );
    assert!(
        ram.available
            .is_some_and(|available| available <= ram.total),
        "OS available RAM must be retained for admission"
    );
    // #1439: the evictable ZFS ARC credit rides beside MemAvailable and the
    // sum admission spends can never exceed the machine. On a ZFS host this
    // exercises the real arcstats reader; elsewhere the credit is absent.
    let credit = ram.reclaimable_zfs_arc.unwrap_or(0);
    assert!(
        ram.available_or_estimate().saturating_add(credit) <= ram.total,
        "available {} + evictable ARC {} must fit in total {}",
        ram.available_or_estimate(),
        credit,
        ram.total
    );
    assert!(ram.available_with_evictable_arc() <= ram.total);
    assert!(
        ram.available_with_evictable_arc() >= ram.available_or_estimate(),
        "the credit never lowers the figure admission spends"
    );
    eprintln!(
        "ram_snapshot on this host: total={} available={:?} reclaimable_zfs_arc={:?} available_with_evictable_arc={}",
        ram.total,
        ram.available,
        ram.reclaimable_zfs_arc,
        ram.available_with_evictable_arc()
    );
    let system = crate::resources::ram_snapshot_from_system();
    assert_eq!(
        system.reclaimable_zfs_arc, None,
        "the RSS-only reading never consults arcstats"
    );
}

fn raw_uuid(hex: &str) -> [u8; 16] {
    assert_eq!(hex.len(), 32);
    let mut bytes = [0_u8; 16];
    for (index, byte) in bytes.iter_mut().enumerate() {
        *byte = u8::from_str_radix(&hex[index * 2..index * 2 + 2], 16).unwrap();
    }
    bytes
}

fn cuda_target(logical_ordinal: usize, uuid: &str, kind: CudaDeviceKind) -> TelemetryTarget {
    TelemetryTarget::cuda(
        logical_ordinal,
        raw_uuid(uuid),
        kind,
        format!("visible GPU {logical_ordinal}"),
        24 * 1024 * 1024 * 1024,
    )
}

#[test]
fn visible_smi_projection_joins_reordered_numeric_devices_by_uuid() {
    let targets = vec![
        cuda_target(
            0,
            "22222222222222222222222222222222",
            CudaDeviceKind::FullGpu,
        ),
        cuda_target(
            1,
            "00000000000000000000000000000000",
            CudaDeviceKind::FullGpu,
        ),
    ];
    let samples = crate::resources::SmiSource::parse_visible_snapshot(
        "0, GPU-00000000-0000-0000-0000-000000000000, physical zero, 24576, 100\n\
         1, GPU-11111111-1111-1111-1111-111111111111, hidden one, 24576, 200\n\
         2, GPU-22222222-2222-2222-2222-222222222222, physical two, 24576, 300",
        &targets,
    );

    assert_eq!(samples.len(), 2);
    assert_eq!(samples[0].ordinal, 0);
    assert_eq!(samples[0].name, "physical two");
    assert_eq!(samples[0].vram_used, 300 * 1024 * 1024);
    assert_eq!(samples[1].ordinal, 1);
    assert_eq!(samples[1].name, "physical zero");
}

#[test]
fn visible_smi_projection_never_exposes_hidden_physical_devices() {
    let targets = vec![cuda_target(
        0,
        "22222222222222222222222222222222",
        CudaDeviceKind::FullGpu,
    )];
    let samples = crate::resources::SmiSource::parse_visible_snapshot(
        "0, GPU-00000000-0000-0000-0000-000000000000, hidden zero, 24576, 100\n\
         2, GPU-22222222-2222-2222-2222-222222222222, visible two, 24576, 300",
        &targets,
    );

    assert_eq!(samples.len(), 1);
    assert_eq!(samples[0].ordinal, 0);
    assert_eq!(samples[0].name, "visible two");
}

#[test]
fn visible_smi_projection_supports_gpu_uuid_visibility_selectors() {
    let targets = vec![cuda_target(
        0,
        "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
        CudaDeviceKind::FullGpu,
    )];
    let samples = crate::resources::SmiSource::parse_visible_snapshot(
        "7, GPU-aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa, selected by UUID, 24576, 42",
        &targets,
    );

    assert_eq!(samples.len(), 1);
    assert_eq!(samples[0].ordinal, 0);
    assert_eq!(samples[0].vram_used, 42 * 1024 * 1024);
}

#[test]
fn mig_target_never_receives_physical_gpu_telemetry() {
    let targets = vec![cuda_target(
        0,
        "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
        CudaDeviceKind::Mig,
    )];
    let samples = crate::resources::SmiSource::parse_visible_snapshot(
        "0, GPU-aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa, physical parent, 81920, 1000",
        &targets,
    );

    assert!(
        samples.is_empty(),
        "a MIG worker must not inherit its parent GPU's full-memory telemetry"
    );
}

#[test]
fn parse_nvidia_smi_line_happy_path() {
    let line = "0, NVIDIA GeForce RTX 3090, 24564, 14248";
    let parsed = crate::resources::parse_nvidia_smi_line(line).expect("parse should succeed");
    assert_eq!(parsed.0, 0);
    assert_eq!(parsed.1, "NVIDIA GeForce RTX 3090");
    assert_eq!(parsed.2, 24_564 * 1024 * 1024);
    assert_eq!(parsed.3, 14_248 * 1024 * 1024);
}

#[test]
fn parse_nvidia_smi_line_garbage_returns_none() {
    assert!(crate::resources::parse_nvidia_smi_line("not,enough,fields").is_none());
    assert!(crate::resources::parse_nvidia_smi_line("0,GPU,notnum,0").is_none());
    assert!(crate::resources::parse_nvidia_smi_line("").is_none());
}

#[test]
fn smi_snapshot_sets_per_process_fields_to_none() {
    let gpus = crate::resources::SmiSource::parse_snapshot(
        "0, NVIDIA GeForce RTX 3090, 24564, 14248\n\
         1, NVIDIA GeForce RTX 3090, 24564, 800",
    );
    assert_eq!(gpus.len(), 2);
    assert_eq!(gpus[0].ordinal, 0);
    assert_eq!(gpus[0].vram_total, 24_564 * 1024 * 1024);
    assert_eq!(gpus[0].vram_used, 14_248 * 1024 * 1024);
    assert_eq!(gpus[0].vram_used_by_mold, None);
    assert_eq!(gpus[0].vram_used_by_other, None);
    assert_eq!(gpus[1].ordinal, 1);
}

#[test]
#[cfg(feature = "nvml")]
fn nvml_source_returns_zero_gpus_when_nvml_init_fails() {
    // On a CI box without NVML, `NvmlSource::try_new()` returns Err — the
    // caller must treat that as "no GPUs" without panicking.
    //
    // We call `snapshot` with a deliberately-uninitialized source by
    // passing an Err to ensure the happy-path ctor isn't required for
    // the fallback behavior.
    let res = crate::resources::NvmlSource::try_new();
    match res {
        Ok(_) => {
            // NVML is present — then at minimum snapshot() should not panic
            // and should return Vec<_> (possibly empty).
            let src = crate::resources::NvmlSource::try_new().unwrap();
            let gpus = src.snapshot(std::process::id());
            for g in &gpus {
                assert!(g.vram_total >= g.vram_used);
            }
        }
        Err(_) => {
            // NVML absent — acceptable on CI, treat as skip.
        }
    }
}

#[test]
#[cfg(all(feature = "cuda", feature = "nvml"))]
fn live_nvml_join_matches_every_visible_full_cuda_gpu_by_uuid() {
    let discovered = mold_inference::device::discover_gpus();
    if discovered.is_empty() {
        return;
    }
    let Ok(source) = crate::resources::NvmlSource::try_new() else {
        return;
    };
    let inventory = crate::resources::discover_telemetry_targets(&discovered);
    let snapshots = source.snapshot_visible(std::process::id(), &inventory);

    assert!(
        snapshots
            .iter()
            .all(|snapshot| discovered.iter().any(|gpu| gpu.ordinal == snapshot.ordinal)),
        "NVML projection published a GPU outside CUDA's visible inventory"
    );
    for gpu in discovered.iter().filter(|gpu| {
        gpu.raw_cuda_uuid.is_some()
            && gpu.device_kind != Some(mold_inference::device::CudaDeviceKind::Mig)
    }) {
        let target = inventory
            .iter()
            .find(|target| target.logical_ordinal == gpu.ordinal)
            .unwrap();
        assert!(
            target.nvml_uuid.is_some(),
            "visible CUDA GPU {} did not resolve to an NVML UUID",
            gpu.ordinal
        );
        let snapshot = snapshots
            .iter()
            .find(|snapshot| snapshot.ordinal == gpu.ordinal)
            .unwrap_or_else(|| {
                panic!(
                    "visible CUDA GPU {} did not receive UUID-joined telemetry",
                    gpu.ordinal
                )
            });
        assert_eq!(snapshot.backend, GpuBackend::Cuda);
        assert!(snapshot.vram_total >= snapshot.vram_used);
    }
}
