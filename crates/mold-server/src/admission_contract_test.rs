//! Cross-cutting durable-admission regressions.
//!
//! These tests intentionally stay below the HTTP and inference layers. They
//! pin the durability, observer-detachment, preparation-failure, ownership,
//! and media-integrity contracts that every model-family adapter must share.

use std::path::Path;
use std::sync::Arc;

use mold_core::GenerateRequest;

use crate::queue_journal::{BatchJournalAdmission, JournalAdmission, QueueJournal};
use crate::queue_media_ingress::{ObserverMode, QueueMediaIngress};
use crate::state::SseCompletionPayload;

fn request(prompt: &str) -> GenerateRequest {
    serde_json::from_value(serde_json::json!({
        "prompt": prompt,
        "model": "mock-model",
        "width": 64,
        "height": 64,
        "steps": 1,
        "guidance": 1.0,
        "output_format": "png"
    }))
    .unwrap()
}

fn record_batch(journal: &Arc<QueueJournal>, output: &Path, job_id: &str, batch_id: &str) {
    let request = request("durable admission contract");
    journal
        .record_batch(BatchJournalAdmission {
            id: batch_id,
            client_batch_id: &format!("client-{batch_id}"),
            request_sha256: "admission-contract-fingerprint",
            children: &[JournalAdmission {
                id: job_id,
                request: &request,
                output_dir: Some(output),
                target_gpu: None,
                target_device_id: None,
                completion_payload: SseCompletionPayload::MetadataOnly,
                batch_child: false,
            }],
        })
        .unwrap();
}

#[tokio::test]
async fn accepted_job_survives_observer_disconnect_and_process_restart() {
    let root = tempfile::tempdir().unwrap();
    let output = root.path().join("gallery");
    std::fs::create_dir_all(&output).unwrap();
    let db_path = root.path().join("mold.db");

    {
        let db = Arc::new(Some(mold_db::MetadataDb::open(&db_path).unwrap()));
        let journal = Arc::new(QueueJournal::new(
            db,
            Some(root.path()),
            "admission-contract-instance",
        ));
        let ingress = QueueMediaIngress::new(1);
        let observer = ingress
            .reserve("survives-disconnect", ObserverMode::Raw)
            .unwrap();

        record_batch(&journal, &output, "survives-disconnect", "disconnect-batch");
        ingress.publish_committed("survives-disconnect");
        assert_eq!(
            ingress.next_committed_id().as_deref(),
            Some("survives-disconnect")
        );

        // Dropping the response observer is not cancellation authority.
        drop(observer);
        assert_eq!(ingress.next_committed_id(), None);
        let rows = journal.list_all();
        assert_eq!(rows.len(), 1);
        assert_eq!(rows[0].id, "survives-disconnect");
    }

    // Re-open both the database and owner claim. The accepted row must still
    // be claimable even though its original response channel no longer exists.
    let db = Arc::new(Some(mold_db::MetadataDb::open(&db_path).unwrap()));
    let journal = Arc::new(QueueJournal::new(
        db,
        Some(root.path()),
        "admission-contract-instance",
    ));
    let rows = journal.list_all();
    assert_eq!(rows.len(), 1);
    assert_eq!(rows[0].id, "survives-disconnect");
    let claimed = journal.claim_next_feeder().unwrap().unwrap();
    assert_eq!(claimed.row.id, "survives-disconnect");
}

#[test]
fn preparation_failure_is_visible_and_a_stale_preparer_is_fenced() {
    let root = tempfile::tempdir().unwrap();
    let output = root.path().join("gallery");
    std::fs::create_dir_all(&output).unwrap();
    let db = Arc::new(Some(mold_db::MetadataDb::open_in_memory().unwrap()));
    let journal = Arc::new(QueueJournal::new(
        db.clone(),
        Some(root.path()),
        "preparation-failure-instance",
    ));
    record_batch(&journal, &output, "prepare-failed", "prepare-batch");

    let claim = journal.claim_next_feeder().unwrap().unwrap();
    let stale_token = claim.claim_token.clone();
    journal
        .attach_claimed(&claim.row.id, claim.claim_token)
        .hold("preparation failed: artifact digest mismatch");

    let row = journal.list_all().pop().unwrap();
    assert_eq!(row.state, mold_db::generation_queue::QueueRowState::Held);
    assert_eq!(
        row.held_reason.as_deref(),
        Some("preparation failed: artifact digest mismatch")
    );
    let batch = journal.generation_batch("prepare-batch").unwrap();
    assert_eq!(batch.children[0].state, "held");
    assert_eq!(
        batch.children[0].error.as_deref(),
        Some("preparation failed: artifact digest mismatch")
    );

    // The attempt that reported the failure cannot later publish Running.
    assert_eq!(
        mold_db::generation_queue::mark_dispatched_claimed(
            db.as_ref().as_ref().unwrap(),
            "prepare-failed",
            &stale_token,
            100,
        )
        .unwrap(),
        None
    );
    assert!(journal.claim_next_feeder().unwrap().is_none());
}

#[test]
fn retried_queue_claim_fences_the_old_attempt_and_starts_only_once() {
    let root = tempfile::tempdir().unwrap();
    let output = root.path().join("gallery");
    std::fs::create_dir_all(&output).unwrap();
    let db = Arc::new(Some(mold_db::MetadataDb::open_in_memory().unwrap()));
    let journal = Arc::new(QueueJournal::new(
        db.clone(),
        Some(root.path()),
        "preparation-retry-instance",
    ));
    record_batch(&journal, &output, "retry-job", "retry-batch");
    let owner = journal.owner_uuid().unwrap().to_string();
    let db = db.as_ref().as_ref().unwrap();

    let first = mold_db::generation_queue::claim_by_id(db, &owner, "retry-job", "attempt-1", 10)
        .unwrap()
        .unwrap();
    assert_eq!(first.row.id, "retry-job");
    assert!(mold_db::generation_queue::release_claim(db, "retry-job", "attempt-1", 11).unwrap());

    let retry = mold_db::generation_queue::claim_by_id(db, &owner, "retry-job", "attempt-2", 12)
        .unwrap()
        .unwrap();
    assert_eq!(retry.row.id, "retry-job");
    assert_eq!(
        mold_db::generation_queue::mark_dispatched_claimed(db, "retry-job", "attempt-1", 13)
            .unwrap(),
        None
    );
    assert_eq!(
        mold_db::generation_queue::mark_dispatched_claimed(db, "retry-job", "attempt-2", 14)
            .unwrap(),
        Some(1)
    );
    assert_eq!(
        mold_db::generation_queue::mark_dispatched_claimed(db, "retry-job", "attempt-2", 15)
            .unwrap(),
        None,
        "one retry token may start the job only once"
    );
}

#[cfg(unix)]
#[test]
fn source_media_survives_store_reopen_and_rehydrates_exactly_once() {
    use crate::queue_media::{
        extract_request_media, into_seal_media, project_request_media, ProcessPrivateAuthorities,
    };
    use crate::queue_media_runtime::DeferredQueueMedia;
    use crate::queue_media_store::{QueueMediaOperationFingerprint, QueueMediaStore};

    let root = tempfile::tempdir().unwrap();
    let original: GenerateRequest = serde_json::from_value(serde_json::json!({
        "prompt": "source must survive",
        "model": "mock-model",
        "width": 64,
        "height": 64,
        "steps": 1,
        "guidance": 1.0,
        "source_image": "c291cmNlLWJ5dGVz",
        "source_image_name": "source-sentinel.png"
    }))
    .unwrap();
    let expected_source = original.source_image.clone();
    let extracted = extract_request_media(
        "source-restart-job",
        original,
        &ProcessPrivateAuthorities::none(),
        None,
    )
    .unwrap();
    let projection = project_request_media(extracted.media()).unwrap();
    assert!(projection.source_image);
    let (request_json, media) = extracted.into_parts();
    assert!(!request_json.contains("source-sentinel"));
    assert!(!request_json.contains("c291cmNlLWJ5dGVz"));

    let media_set = {
        let store = QueueMediaStore::open(root.path()).unwrap().store;
        store
            .seal_v2_with_operation_fingerprint(
                "source-owner",
                "source-restart-job",
                &QueueMediaOperationFingerprint::sha256_v1(b"source restart operation"),
                &projection,
                into_seal_media(media).unwrap(),
            )
            .unwrap()
    };

    let store = Arc::new(QueueMediaStore::open_existing(root.path()).unwrap());
    let deferred = DeferredQueueMedia::new(store, media_set, projection);
    let mut restored: GenerateRequest = serde_json::from_str(&request_json).unwrap();
    let _lease = deferred
        .hydrate_into("source-restart-job", &mut restored)
        .unwrap();
    assert_eq!(restored.source_image, expected_source);
    assert_eq!(
        restored.source_image_name.as_deref(),
        Some("source-sentinel.png")
    );
    assert_eq!(restored.model, "mock-model");
}

#[test]
fn h3_with_source_media_is_refused_explicitly_instead_of_losing_conditioning() {
    use crate::queue_media::{
        extract_request_media, ProcessPrivateAuthorities, ProcessPrivateAuthority, QueueMediaError,
    };

    let request: GenerateRequest = serde_json::from_value(serde_json::json!({
        "prompt": "h3 source must not disappear",
        "model": "minimax-h3-fl2va:comfy-pruned-int8",
        "width": 768,
        "height": 768,
        "steps": 9,
        "guidance": 1.0,
        "source_image": "c291cmNlLWJ5dGVz",
        "source_image_name": "h3-source-sentinel.png"
    }))
    .unwrap();

    assert!(matches!(
        extract_request_media(
            "h3-source-job",
            request,
            &ProcessPrivateAuthorities::none(),
            None
        ),
        Err(QueueMediaError::UnsupportedProcessPrivateAuthority(
            ProcessPrivateAuthority::H3PrivateIngressGrant
        ))
    ));
}

/// A GGUF LTX-2.5 row in the journal replays into ordinary deferred
/// preparation: model activation admits it (the native quantized runtime
/// shipped, #1414), and planning can only refuse it for real resource or
/// artifact reasons — never with the retired `LTX25_GGUF_RUNTIME_UNAVAILABLE`
/// policy code.
#[test]
fn ltx25_gguf_row_replayed_from_journal_is_admitted_at_deferred_preparation() {
    use crate::execution_plan::{eligible_devices_for_request, DeviceFact, ExecutionPlanError};

    let root = tempfile::tempdir().unwrap();
    let output = root.path().join("gallery");
    std::fs::create_dir_all(&output).unwrap();
    let db = Arc::new(Some(mold_db::MetadataDb::open_in_memory().unwrap()));
    let journal = Arc::new(QueueJournal::new(
        db,
        Some(root.path()),
        "ltx25-gguf-replay-instance",
    ));
    let mut request = request("replayed GGUF row");
    request.model = mold_core::ltx25_manifest::DISTILLED_Q4.to_string();
    journal
        .record_batch(BatchJournalAdmission {
            id: "gguf-batch",
            client_batch_id: "client-gguf-batch",
            request_sha256: "gguf-replay-fingerprint",
            children: &[JournalAdmission {
                id: "gguf-replay",
                request: &request,
                output_dir: Some(&output),
                target_gpu: None,
                target_device_id: None,
                completion_payload: SseCompletionPayload::MetadataOnly,
                batch_child: false,
            }],
        })
        .unwrap();

    let claimed = journal.claim_next_feeder().unwrap().unwrap();
    let replayed: GenerateRequest = serde_json::from_str(&claimed.row.request_json).unwrap();
    assert_eq!(replayed.model, mold_core::ltx25_manifest::DISTILLED_Q4);

    // Policy no longer refuses the identity anywhere in the pipeline.
    mold_core::require_model_activation(&replayed.model, Some("ltx2"))
        .expect("GGUF activation is admitted since the quantized runtime landed");

    let devices = vec![DeviceFact {
        id: "cuda:0".to_string(),
        ordinal: 0,
        backend: mold_core::GpuBackend::Cuda,
        compute_capability: Some((8, 9)),
        available_vram_bytes: 24 * 1024 * 1024 * 1024,
    }];
    // This test root installs no artifacts, so planning may refuse for
    // missing components — an ordinary retryable plan failure the feeder
    // retries — but never with a model-activation policy refusal.
    if let Err(error) =
        eligible_devices_for_request(&mold_core::Config::default(), &replayed, &devices)
    {
        assert!(
            !matches!(error, ExecutionPlanError::ModelActivation(_)),
            "policy must not refuse a GGUF row: {error:?}"
        );
    }
}
