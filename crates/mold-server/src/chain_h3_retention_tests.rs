use super::*;

fn fixture() -> (
    tempfile::TempDir,
    mold_db::MetadataDb,
    crate::batch_transaction::GalleryPublicationGate,
) {
    (
        tempfile::tempdir().unwrap(),
        mold_db::MetadataDb::open_in_memory().unwrap(),
        Default::default(),
    )
}
fn publish(db: &mold_db::MetadataDb, output: &std::path::Path, name: &str) {
    let req: GenerateRequest = serde_json::from_value(serde_json::json!({"prompt":"child","model":"h3","width":64,"height":64,"steps":1,"guidance":1.0})).unwrap();
    db.upsert(&mold_db::GenerationRecord::from_save(
        output,
        name,
        mold_core::OutputFormat::Mp4,
        mold_core::OutputMetadata::from_generate_request(&req, 1, None, "test"),
        mold_db::RecordSource::Server,
        1,
    ))
    .unwrap();
    std::fs::write(output.join(name), b"delivered-child").unwrap();
}
#[test]
fn owned_child_is_retained_before_trash_and_survives_public_purge() {
    let (dir, db, gate) = fixture();
    let output = dir.path();
    let name = "owned.mp4";
    publish(&db, output, name);
    let artifact = output.join("private/h3-children/batch.mp4");
    let events = crate::events::EventBroadcaster::new();
    let mut receiver = events.subscribe();
    retain_and_trash_child(output, name, &artifact, &db, &gate, &events).unwrap();
    assert_eq!(std::fs::read(&artifact).unwrap(), b"delivered-child");
    assert!(!output.join(name).exists());
    assert!(db
        .get(output, name)
        .unwrap()
        .unwrap()
        .trashed_at_ms
        .is_some());
    assert!(
        matches!(receiver.try_recv().unwrap(), mold_core::ServerEvent::GalleryTrashed { filename } if filename == name)
    );
    std::fs::remove_file(crate::gallery_trash::resolve_gallery_media_source(
        output, name,
    ))
    .unwrap();
    retain_and_trash_child(output, name, &artifact, &db, &gate, &events).unwrap();
    assert_eq!(std::fs::read(&artifact).unwrap(), b"delivered-child");
}
#[test]
fn recovery_copies_the_exact_child_from_trash_when_private_artifact_missing() {
    let (dir, db, gate) = fixture();
    let output = dir.path();
    let name = "owned.mp4";
    publish(&db, output, name);
    crate::gallery_trash::trash_print_blocking(output, name, &db, &gate, 1).unwrap();
    let artifact = output.join("private/recovered.mp4");
    retain_and_trash_child(
        output,
        name,
        &artifact,
        &db,
        &gate,
        &crate::events::EventBroadcaster::new(),
    )
    .unwrap();
    assert_eq!(std::fs::read(artifact).unwrap(), b"delivered-child");
}
#[test]
fn failed_private_copy_never_discards_the_only_public_child() {
    let (dir, db, gate) = fixture();
    let output = dir.path();
    let name = "owned.mp4";
    publish(&db, output, name);
    std::fs::write(output.join("not-a-directory"), b"x").unwrap();
    let artifact = output.join("not-a-directory/child.mp4");
    assert!(retain_and_trash_child(
        output,
        name,
        &artifact,
        &db,
        &gate,
        &crate::events::EventBroadcaster::new()
    )
    .is_err());
    assert!(output.join(name).is_file());
    assert!(db
        .get(output, name)
        .unwrap()
        .unwrap()
        .trashed_at_ms
        .is_none());
}

#[test]
fn retention_refuses_foreign_batch_or_more_than_one_child() {
    use mold_db::generation_batches::*;
    let child = DurableGenerationBatchChildRow {
        batch_id: "batch".into(),
        job_id: "child".into(),
        batch_index: 1,
        state: "complete".into(),
        error: None,
        error_code: None,
        retryable: false,
        updated_at_ms: 1,
        revision: 1,
        terminal_error_json: None,
        result_json: None,
        completed_at_ms: Some(1),
    };
    let mut detail = DurableGenerationBatchDetail {
        batch: GenerationBatchRow {
            id: "batch".into(),
            client_batch_id: "owned".into(),
            owner_uuid: "host".into(),
            request_sha256: "hash".into(),
            created_at_ms: 1,
        },
        children: vec![child.clone()],
    };
    assert!(require_owned_child(&detail, "batch", "owned").is_ok());
    let mut zero_based = detail.clone();
    zero_based.children[0].batch_index = 0;
    assert!(require_owned_child(&zero_based, "batch", "owned").is_err());
    assert!(require_owned_child(&detail, "other", "owned").is_err());
    assert!(require_owned_child(&detail, "batch", "foreign").is_err());
    detail.children.push(child);
    assert!(require_owned_child(&detail, "batch", "owned").is_err());
    detail.children.pop();
    detail.children[0].batch_id = "foreign".into();
    assert!(require_owned_child(&detail, "batch", "owned").is_err());
}

#[test]
fn trash_failure_keeps_retained_artifact_and_does_not_fail_generation() {
    let (dir, db, gate) = fixture();
    let output = dir.path();
    let name = "owned.mp4";
    publish(&db, output, name);
    std::fs::write(
        crate::batch_transaction::gallery_trash_dir(output),
        b"blocked trash",
    )
    .unwrap();
    let artifact = output.join("private/batch.mp4");
    retain_and_trash_child(
        output,
        name,
        &artifact,
        &db,
        &gate,
        &crate::events::EventBroadcaster::new(),
    )
    .unwrap();
    assert_eq!(std::fs::read(artifact).unwrap(), b"delivered-child");
    assert!(output.join(name).is_file());
}

#[cfg(unix)]
#[test]
fn source_symlinks_cannot_be_retained_or_trashed() {
    let (dir, db, gate) = fixture();
    let output = dir.path();
    std::fs::write(output.join("foreign.mp4"), b"not-owned").unwrap();
    std::os::unix::fs::symlink(output.join("foreign.mp4"), output.join("owned.mp4")).unwrap();
    let artifact = output.join("private/batch.mp4");
    assert!(retain_and_trash_child(
        output,
        "owned.mp4",
        &artifact,
        &db,
        &gate,
        &crate::events::EventBroadcaster::new()
    )
    .is_err());
    assert!(!artifact.exists());
    assert_eq!(
        std::fs::read(output.join("foreign.mp4")).unwrap(),
        b"not-owned"
    );
}

#[tokio::test]
async fn preoccupied_client_id_requires_exact_admitted_request_before_any_reattach() {
    let root = tempfile::tempdir().unwrap();
    let db = Arc::new(Some(mold_db::MetadataDb::open_in_memory().unwrap()));
    let mut state = AppState::for_tests();
    state.metadata_db = db.clone();
    state.queue_journal = Arc::new(crate::queue_journal::QueueJournal::new(
        db.clone(),
        Some(root.path()),
        "h3-retention-test",
    ));
    let owner = state.queue_journal.owner_uuid().unwrap().to_string();
    let lifecycle = Arc::new(crate::queue_media_lifecycle::QueueMediaLifecycle::new(
        db,
        root.path().into(),
        owner.clone(),
    ));
    state
        .queue_journal
        .install_queue_media_lifecycle(lifecycle.clone())
        .unwrap();
    crate::queue_media_startup::reconcile_claimed_owner(&state.queue_journal, lifecycle.as_ref())
        .unwrap();
    let admission =
        crate::queue_media_admission::DurableMediaAdmission::new(lifecycle, 4, false).unwrap();
    state
        .queue_journal
        .install_queue_media_admission(admission)
        .unwrap();
    let request: GenerateRequest = serde_json::from_value(serde_json::json!({"prompt":"owned","model":"mock-model","width":64,"height":64,"steps":1,"guidance":1.0})).unwrap();
    let client = child_client_id("parent", 0, 1, 0);
    let fingerprint =
        crate::queue_media_admission::test_operation_fingerprint(&client, vec![request.clone()]);
    let receipt = crate::queue_media_store::QueueMediaStore::open(root.path())
        .unwrap()
        .store
        .seal_operation_receipt_v1(&owner, &client, &fingerprint)
        .unwrap();
    let serialized = serde_json::to_string(&request).unwrap();
    state.metadata_db.as_ref().as_ref().unwrap().with_conn(|conn| {
        conn.execute("INSERT INTO chain_jobs(id,state,model,request_json,job_dir,stage_count,created_at,updated_at) VALUES('parent','running','mock-model','{}','/jobs/parent',1,0,0)",[])?;
        Ok(())
    }).unwrap();
    let stage_owner = mold_db::chain_stage_generations::ChainStageOwner {
        chain_id: "parent".into(),
        stage_index: 0,
    };
    state
        .queue_journal
        .record_batch_with_media_owned(
            crate::queue_journal::MediaBatchJournalAdmission {
                id: "stored-batch",
                client_batch_id: &client,
                operation_receipt: receipt.as_str(),
                children: &[crate::queue_journal::MediaJournalAdmission {
                    id: "stored-child",
                    model: &request.model,
                    request_json: &serialized,
                    media_set: None,
                    output_dir: root.path(),
                    target_gpu: None,
                    target_device_id: None,
                    completion_payload: SseCompletionPayload::MetadataOnly,
                    seed_pinned: false,
                    admission_authority: None,
                }],
                observer_job_ids: &[],
            },
            Some(&stage_owner),
        )
        .unwrap();
    let before = state
        .queue_journal
        .durable_generation_batch("stored-batch")
        .unwrap()
        .unwrap();
    let mut foreign = request.clone();
    foreign.prompt = "someone else's operation".into();
    let conflict = verify_or_admit_child(&state, &client, &foreign, "parent", 0)
        .await
        .unwrap_err();
    assert!(
        conflict.to_string().contains("different request"),
        "{conflict:#}"
    );
    assert_eq!(
        state
            .queue_journal
            .durable_generation_batch("stored-batch")
            .unwrap()
            .unwrap(),
        before,
        "foreign collision must not cancel or mutate the existing child"
    );
    assert_eq!(
        verify_or_admit_child(&state, &client, &request, "parent", 0)
            .await
            .unwrap(),
        "stored-batch"
    );
    let after = state
        .queue_journal
        .durable_generation_batch("stored-batch")
        .unwrap()
        .unwrap();
    assert_eq!(after.children, before.children);
    assert_eq!(
        after.children[0].batch_index, 1,
        "ordinary durable admission indexes its first child from one"
    );
    require_owned_child(&after, "stored-batch", &client).unwrap();
}
