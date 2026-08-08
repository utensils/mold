use mold_db::{MetadataDb, SchedulerEstimateRecord, SchedulerEstimates};

#[test]
fn scheduler_estimates_round_trip_and_prune() {
    let dir = tempfile::tempdir().unwrap();
    let db = MetadataDb::open(&dir.path().join("mold.db")).unwrap();
    let estimates = SchedulerEstimates::new(&db);
    estimates
        .upsert(&SchedulerEstimateRecord {
            estimate_key: "cuda:sm86|flux|generation|1024|bf16".into(),
            device_class: "cuda:sm86:24gb".into(),
            model_family: "flux".into(),
            model_fingerprint: "flux".into(),
            work_kind: "generation".into(),
            shape_bucket: "1024x1024:s30".into(),
            execution_fingerprint: "bf16".into(),
            sample_count: 3,
            ewma_total_ms: 1_250.0,
            ewma_runtime_ms: Some(950.0),
            ewma_load_ms: Some(250.0),
            ewma_warm_reload_ms: Some(50.0),
            ewma_prompt_encode_ms: Some(100.0),
            ewma_denoise_ms: Some(800.0),
            ewma_vae_ms: Some(100.0),
            ewma_visual_decode_ms: Some(90.0),
            ewma_audio_decode_ms: Some(40.0),
            ewma_mux_ms: Some(10.0),
            ewma_upscale_ms: None,
            vram_high_water_bytes: Some(12_000),
            host_high_water_bytes: Some(16_000),
            failure_count: 1,
            invalidated_count: 2,
            last_outcome: "invalidated".into(),
            last_fallback_reason: Some("vae_cpu".into()),
            last_invalidated_plan_reason: Some("capacity changed".into()),
            last_observed_at: 100,
        })
        .unwrap();

    let loaded = estimates.list().unwrap();
    assert_eq!(loaded.len(), 1);
    assert_eq!(loaded[0].sample_count, 3);
    assert_eq!(loaded[0].ewma_load_ms, Some(250.0));
    assert_eq!(loaded[0].ewma_runtime_ms, Some(950.0));
    assert_eq!(loaded[0].model_family, "flux");
    assert_eq!(loaded[0].ewma_denoise_ms, Some(800.0));
    assert_eq!(loaded[0].ewma_vae_ms, Some(100.0));
    assert_eq!(loaded[0].ewma_visual_decode_ms, Some(90.0));
    assert_eq!(loaded[0].ewma_audio_decode_ms, Some(40.0));
    assert_eq!(loaded[0].ewma_mux_ms, Some(10.0));
    assert_eq!(loaded[0].failure_count, 1);
    assert_eq!(loaded[0].invalidated_count, 2);

    assert_eq!(estimates.prune_before(101, 10_000).unwrap(), 1);
    assert!(estimates.list().unwrap().is_empty());
}
