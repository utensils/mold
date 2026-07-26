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
            model_fingerprint: "flux".into(),
            work_kind: "generation".into(),
            shape_bucket: "1024x1024:s30".into(),
            execution_fingerprint: "bf16".into(),
            sample_count: 3,
            ewma_total_ms: 1_250.0,
            ewma_load_ms: Some(250.0),
            vram_high_water_bytes: Some(12_000),
            host_high_water_bytes: Some(16_000),
            last_observed_at: 100,
        })
        .unwrap();

    let loaded = estimates.list().unwrap();
    assert_eq!(loaded.len(), 1);
    assert_eq!(loaded[0].sample_count, 3);
    assert_eq!(loaded[0].ewma_load_ms, Some(250.0));

    assert_eq!(estimates.prune_before(101, 10_000).unwrap(), 1);
    assert!(estimates.list().unwrap().is_empty());
}
