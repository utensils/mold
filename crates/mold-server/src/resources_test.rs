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
        }],
        system_ram: RamSnapshot {
            total: 64_000_000_000,
            used: 0,
            used_by_mold: 0,
            used_by_other: 0,
        },
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
