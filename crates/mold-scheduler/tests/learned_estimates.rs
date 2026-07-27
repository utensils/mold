use mold_scheduler::{
    EstimateConfidence, EstimateKey, EstimateObservation, EstimateOutcome, EstimatePhaseTimings,
    EstimateStore, StaticEstimate,
};

fn key(suffix: &str) -> EstimateKey {
    EstimateKey {
        device_class: "cuda:sm86:24gb".into(),
        model_family: "flux".into(),
        model_fingerprint: format!("flux-dev:{suffix}"),
        work_kind: "generation".into(),
        shape_bucket: "1024x1024:s30".into(),
        execution_fingerprint: "bf16:resident".into(),
    }
}

#[test]
fn learned_timing_never_lowers_the_static_memory_floor() {
    let mut store = EstimateStore::default();
    store.observe(
        key("q8"),
        EstimateObservation {
            total_ms: Some(1_000),
            phases: EstimatePhaseTimings {
                cold_load_ms: Some(100),
                ..Default::default()
            },
            vram_high_water_bytes: Some(2_000),
            host_incremental_high_water_bytes: Some(3_000),
            observed_at_unix_s: 100,
            ..Default::default()
        },
    );

    let estimate = store.estimate(
        &key("q8"),
        StaticEstimate {
            total_ms: 8_000,
            vram_bytes: 12_000,
            host_bytes: 16_000,
        },
    );
    assert_eq!(estimate.total_ms, 1_000);
    assert_eq!(estimate.vram_bytes, 12_000);
    assert_eq!(estimate.host_bytes, 16_000);
    assert_eq!(estimate.confidence, EstimateConfidence::Low);
}

#[test]
fn samples_are_winsorized_then_ewma_updated_with_documented_confidence() {
    let mut store = EstimateStore::default();
    for observed_at_unix_s in 1..=10 {
        store.observe(
            key("q8"),
            EstimateObservation {
                total_ms: Some(match observed_at_unix_s {
                    1 => 1_000,
                    2 => 100_000,
                    _ => 1_750,
                }),
                observed_at_unix_s,
                ..Default::default()
            },
        );
    }

    let bucket = store.exact(&key("q8")).unwrap();
    assert_eq!(bucket.sample_count, 10);
    assert_eq!(bucket.ewma_total_ms, 1_750.0);
    assert_eq!(bucket.confidence(), EstimateConfidence::High);
}

#[test]
fn a_zero_first_sample_does_not_pin_the_ewma_at_zero() {
    let mut store = EstimateStore::default();
    let key = key("zero-clock");
    for (total_ms, observed_at_unix_s) in [(0, 1), (1_000, 2)] {
        store.observe(
            key.clone(),
            EstimateObservation {
                total_ms: Some(total_ms),
                observed_at_unix_s,
                ..Default::default()
            },
        );
    }

    let bucket = store.exact(&key).unwrap();
    assert!(bucket.ewma_total_ms > 0.0);
    assert_eq!(bucket.sample_count, 2);
}

#[test]
fn conservative_measured_memory_high_water_decays_after_an_outlier() {
    let mut store = EstimateStore::default();
    let key = key("memory-envelope");
    for (sample, observed_at_unix_s) in [(10_000, 1), (1_000, 2), (1_000, 3)] {
        store.observe(
            key.clone(),
            EstimateObservation {
                total_ms: Some(1_000),
                vram_high_water_bytes: Some(sample),
                observed_at_unix_s,
                ..Default::default()
            },
        );
    }

    assert_eq!(
        store.exact(&key).unwrap().vram_conservative_bytes,
        Some(9_025)
    );
}

#[test]
fn old_buckets_and_capacity_overflow_are_pruned_deterministically() {
    let mut store = EstimateStore::with_limits(2, 180 * 24 * 60 * 60);
    for (suffix, observed_at_unix_s) in [("old", 1), ("newer", 20), ("newest", 30)] {
        store.observe(
            key(suffix),
            EstimateObservation {
                total_ms: Some(1_000),
                observed_at_unix_s,
                ..Default::default()
            },
        );
    }
    assert!(store.exact(&key("old")).is_none());
    assert_eq!(store.len(), 2);

    store.prune(180 * 24 * 60 * 60 + 31);
    assert!(store.is_empty());
}

#[test]
fn opaque_catalog_ids_do_not_collapse_to_their_source_prefix() {
    let mut cv = key("unused");
    cv.model_family.clear();
    cv.model_fingerprint = "cv:3143864".into();
    let mut hf = cv.clone();
    hf.model_fingerprint = "hf:org/ltx-video".into();

    assert_ne!(cv.normalized(), hf.normalized());
    assert_ne!(cv.persistence_key(), hf.persistence_key());
}

#[test]
fn failures_and_invalidations_are_persisted_without_poisoning_eta_samples() {
    let mut store = EstimateStore::default();
    let key = key("outcomes");
    store.observe(
        key.clone(),
        EstimateObservation {
            total_ms: Some(1_000),
            phases: EstimatePhaseTimings {
                denoise_ms: Some(800),
                ..Default::default()
            },
            observed_at_unix_s: 1,
            ..Default::default()
        },
    );
    store.observe(
        key.clone(),
        EstimateObservation {
            outcome: EstimateOutcome::Failure,
            fallback_reason: Some("vae_cpu".into()),
            observed_at_unix_s: 2,
            ..Default::default()
        },
    );
    store.observe(
        key.clone(),
        EstimateObservation {
            outcome: EstimateOutcome::Invalidated,
            invalidated_plan_reason: Some("memory sample changed".into()),
            observed_at_unix_s: 3,
            ..Default::default()
        },
    );

    let bucket = store.exact(&key).unwrap();
    assert_eq!(bucket.sample_count, 1);
    assert_eq!(bucket.failure_count, 1);
    assert_eq!(bucket.invalidated_count, 1);
    assert_eq!(bucket.ewma_total_ms, 1_000.0);
    assert_eq!(bucket.ewma_denoise_ms, Some(800.0));
    assert_eq!(bucket.last_outcome, EstimateOutcome::Invalidated);
    assert_eq!(
        bucket.last_invalidated_plan_reason.as_deref(),
        Some("memory sample changed")
    );
}
