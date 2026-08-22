use mold_scheduler::{
    EstimateBucket, EstimateConfidence, EstimateKey, EstimateObservation, EstimateOutcome,
    EstimatePhaseTimings, EstimateStore, StaticEstimate,
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
fn static_runtime_is_explicit_and_not_rederived_from_total() {
    let estimate = EstimateStore::default().estimate(
        &key("explicit-static-runtime"),
        StaticEstimate {
            total_ms: 9_999,
            cold_setup_ms: 900,
            warm_setup_ms: 90,
            predicted_run_ms: 1_234,
            vram_bytes: 1,
            host_bytes: 1,
        },
    );

    assert_eq!(estimate.total_ms, 9_999);
    assert_eq!(estimate.predicted_run_ms, 1_234);
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
            ..Default::default()
        },
    );
    assert_eq!(estimate.total_ms, 1_000);
    assert_eq!(estimate.vram_bytes, 12_000);
    assert_eq!(estimate.host_bytes, 16_000);
    assert_eq!(estimate.confidence, EstimateConfidence::Low);
}

#[test]
fn resolved_estimate_exposes_learned_setup_and_run_phases_for_planning() {
    let mut store = EstimateStore::default();
    store.observe(
        key("phase-truth"),
        EstimateObservation {
            total_ms: Some(1_000),
            phases: EstimatePhaseTimings {
                cold_load_ms: Some(100),
                warm_reload_ms: Some(25),
                identity_extract_ms: None,
                prompt_encode_ms: Some(200),
                denoise_ms: Some(500),
                vae_ms: Some(100),
                visual_decode_ms: Some(75),
                audio_decode_ms: Some(25),
                mux_ms: Some(10),
                upscale_ms: Some(50),
            },
            observed_at_unix_s: 100,
            ..Default::default()
        },
    );

    let estimate = store.estimate(
        &key("phase-truth"),
        StaticEstimate {
            total_ms: 9_000,
            cold_setup_ms: 2_000,
            warm_setup_ms: 500,
            predicted_run_ms: 7_000,
            vram_bytes: 1,
            host_bytes: 1,
        },
    );
    assert!(estimate.learned);
    assert_eq!(estimate.cold_setup_ms, 100);
    assert_eq!(estimate.warm_setup_ms, 25);
    // Runtime excludes both independently observed setup phases.
    assert_eq!(estimate.predicted_run_ms, 875);
    assert_eq!(estimate.confidence, EstimateConfidence::Low);
    let phases = store.exact(&key("phase-truth")).unwrap();
    assert_eq!(phases.ewma_prompt_encode_ms, Some(200.0));
    assert_eq!(phases.ewma_denoise_ms, Some(500.0));
    assert_eq!(phases.ewma_vae_ms, Some(100.0));
    assert_eq!(phases.ewma_visual_decode_ms, Some(75.0));
    assert_eq!(phases.ewma_audio_decode_ms, Some(25.0));
    assert_eq!(phases.ewma_mux_ms, Some(10.0));
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
fn alternating_setup_dispositions_learn_one_runtime_and_distinct_setup_costs() {
    let mut store = EstimateStore::default();
    let key = key("setup-independent-runtime");
    for (total_ms, cold_load_ms, warm_reload_ms, observed_at_unix_s) in [
        (1_100, Some(100), None, 1),
        (1_025, None, Some(25), 2),
        (1_000, None, None, 3),
        (1_100, Some(100), None, 4),
        (1_025, None, Some(25), 5),
        (1_000, None, None, 6),
    ] {
        store.observe(
            key.clone(),
            EstimateObservation {
                total_ms: Some(total_ms),
                phases: EstimatePhaseTimings {
                    cold_load_ms,
                    warm_reload_ms,
                    ..Default::default()
                },
                observed_at_unix_s,
                ..Default::default()
            },
        );
    }

    let estimate = store.estimate(
        &key,
        StaticEstimate {
            total_ms: 9_000,
            cold_setup_ms: 900,
            warm_setup_ms: 90,
            predicted_run_ms: 8_100,
            vram_bytes: 1,
            host_bytes: 1,
        },
    );
    assert_eq!(estimate.predicted_run_ms, 1_000);
    assert_eq!(estimate.cold_setup_ms, 100);
    assert_eq!(estimate.warm_setup_ms, 25);
}

#[test]
fn runtime_subtracts_both_typed_setup_dispositions_when_both_are_reported() {
    let mut store = EstimateStore::default();
    let key = key("both-setup-dispositions");
    store.observe(
        key.clone(),
        EstimateObservation {
            total_ms: Some(1_200),
            phases: EstimatePhaseTimings {
                cold_load_ms: Some(250),
                warm_reload_ms: Some(50),
                ..Default::default()
            },
            observed_at_unix_s: 1,
            ..Default::default()
        },
    );

    let estimate = store.estimate(
        &key,
        StaticEstimate {
            total_ms: 9_000,
            cold_setup_ms: 900,
            warm_setup_ms: 90,
            predicted_run_ms: 8_100,
            vram_bytes: 1,
            host_bytes: 1,
        },
    );
    assert_eq!(estimate.predicted_run_ms, 900);
}

#[test]
fn legacy_total_without_learned_load_remains_a_nonzero_run_estimate() {
    let key = key("legacy-null-load");
    let store = EstimateStore::from_buckets([EstimateBucket {
        key: key.clone(),
        sample_count: 1,
        ewma_total_ms: 1_200.0,
        ewma_runtime_ms: None,
        ewma_load_ms: None,
        ewma_warm_reload_ms: None,
        ewma_prompt_encode_ms: None,
        ewma_denoise_ms: None,
        ewma_vae_ms: None,
        ewma_visual_decode_ms: None,
        ewma_audio_decode_ms: None,
        ewma_mux_ms: None,
        ewma_upscale_ms: None,
        ewma_identity_extract_ms: None,
        vram_conservative_bytes: None,
        host_conservative_bytes: None,
        failure_count: 0,
        invalidated_count: 0,
        last_outcome: EstimateOutcome::Success,
        last_fallback_reason: None,
        last_invalidated_plan_reason: None,
        last_observed_at_unix_s: 1,
    }]);

    let estimate = store.estimate(
        &key,
        StaticEstimate {
            total_ms: 38_000,
            cold_setup_ms: 8_000,
            warm_setup_ms: 250,
            predicted_run_ms: 30_000,
            vram_bytes: 1,
            host_bytes: 1,
        },
    );

    assert_eq!(estimate.predicted_run_ms, 1_200);
}

#[test]
fn failed_observation_with_accidental_total_never_seeds_success_timing() {
    let key = key("failure-total");
    let mut store = EstimateStore::default();
    store.observe(
        key.clone(),
        EstimateObservation {
            total_ms: Some(20_000),
            outcome: EstimateOutcome::Failure,
            observed_at_unix_s: 1,
            ..Default::default()
        },
    );
    store.observe(
        key.clone(),
        EstimateObservation {
            total_ms: Some(1_000),
            outcome: EstimateOutcome::Success,
            observed_at_unix_s: 2,
            ..Default::default()
        },
    );

    let bucket = store.exact(&key).unwrap();
    assert_eq!(bucket.sample_count, 1);
    assert_eq!(bucket.ewma_total_ms, 1_000.0);
    assert_eq!(bucket.ewma_runtime_ms, Some(1_000.0));
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
fn v14_lookup_uses_an_exact_v13_bucket_with_an_empty_model_family() {
    let mut legacy = key("legacy-v13");
    legacy.model_family.clear();
    legacy.model_fingerprint = "cv:3143864".into();
    let mut current = legacy.clone();
    current.model_family = "ltx2".into();

    let mut store = EstimateStore::default();
    store.observe(
        legacy,
        EstimateObservation {
            total_ms: Some(1_234),
            observed_at_unix_s: 1,
            ..Default::default()
        },
    );

    let resolved = store.estimate(
        &current,
        StaticEstimate {
            total_ms: 9_999,
            vram_bytes: 1,
            host_bytes: 1,
            ..Default::default()
        },
    );
    assert!(resolved.learned);
    assert_eq!(resolved.total_ms, 1_234);
}

#[test]
fn first_v14_observation_promotes_v13_history_instead_of_resetting_it() {
    let mut legacy = key("legacy-v13-promotion");
    legacy.model_family.clear();
    legacy.model_fingerprint = "cv:3143864".into();
    let mut current = legacy.clone();
    current.model_family = "ltx2".into();

    let mut store = EstimateStore::default();
    store.observe(
        legacy.clone(),
        EstimateObservation {
            total_ms: Some(1_000),
            observed_at_unix_s: 1,
            ..Default::default()
        },
    );
    store.observe(
        current.clone(),
        EstimateObservation {
            total_ms: Some(2_000),
            observed_at_unix_s: 2,
            ..Default::default()
        },
    );

    assert!(store.exact(&legacy).is_none());
    let promoted = store.exact(&current).expect("promoted v14 bucket");
    assert_eq!(promoted.sample_count, 2);
    assert_eq!(promoted.ewma_total_ms, 1_250.0);
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
                visual_decode_ms: Some(90),
                audio_decode_ms: Some(40),
                mux_ms: Some(10),
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
            phases: EstimatePhaseTimings {
                visual_decode_ms: Some(9_000),
                audio_decode_ms: Some(9_000),
                mux_ms: Some(9_000),
                ..Default::default()
            },
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
    assert_eq!(bucket.ewma_visual_decode_ms, Some(90.0));
    assert_eq!(bucket.ewma_audio_decode_ms, Some(40.0));
    assert_eq!(bucket.ewma_mux_ms, Some(10.0));
    assert_eq!(bucket.last_outcome, EstimateOutcome::Invalidated);
    assert_eq!(
        bucket.last_invalidated_plan_reason.as_deref(),
        Some("memory sample changed")
    );
}
