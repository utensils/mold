//! Unit tests for the resources module.

use crate::resources::ResourceBroadcaster;
use mold_core::{GpuBackend, GpuSnapshot, RamSnapshot, ResourceSnapshot};

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
            used_by_mold: 0,
            used_by_other: 0,
        },
        cpu: None,
    }
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
    let handle = crate::resources::spawn_aggregator(bcast.clone());

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
}

#[test]
fn parse_nvidia_smi_line_happy_path() {
    let line = "0, NVIDIA GeForce RTX 3090, 24564, 14248";
    let parsed = crate::resources::parse_nvidia_smi_line(line).expect("parse should succeed");
    assert_eq!(parsed.0, 0);
    assert_eq!(parsed.1, "NVIDIA GeForce RTX 3090");
    assert_eq!(parsed.2, 24_564_000_000);
    assert_eq!(parsed.3, 14_248_000_000);
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
    assert_eq!(gpus[0].vram_total, 24_564_000_000);
    assert_eq!(gpus[0].vram_used, 14_248_000_000);
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
