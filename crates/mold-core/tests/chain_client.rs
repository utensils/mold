//! Integration tests for the durable chain-job HTTP surface, using `wiremock`
//! to simulate a server.
//!
//! These pin method, path and error translation. Chained video generation is
//! `POST /api/chain-jobs` plus `GET /api/chain-jobs/{id}/events`; the
//! synchronous and SSE compatibility endpoints they replaced are gone.

use mold_core::chain_job::{RetakeMode, RetakeRequest};
use mold_core::MoldClient;
use wiremock::matchers::{method, path};
use wiremock::{Mock, MockServer, ResponseTemplate};

#[tokio::test]
async fn list_chain_jobs_hides_legacy_server_one_shot_shims() {
    let server = MockServer::start().await;
    let summary = |id: &str, ephemeral: bool| {
        serde_json::json!({
            "id": id,
            "state": "running",
            "model": "ltx-2-19b-distilled:fp8",
            "stage_count": 3,
            "current_stage": 1,
            "created_at_unix_ms": 100,
            "updated_at_unix_ms": 101,
            "error": null,
            "ephemeral": ephemeral
        })
    };
    Mock::given(method("GET"))
        .and(path("/api/chain-jobs"))
        .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
            "jobs": [summary("authored", false), summary("one-shot", true)]
        })))
        .expect(1)
        .mount(&server)
        .await;

    let jobs = MoldClient::new(&server.uri())
        .list_chain_jobs()
        .await
        .expect("chain-job listing should parse");

    assert_eq!(jobs.jobs.len(), 1);
    assert_eq!(jobs.jobs[0].id, "authored");
}

// ── /api/generate/chain (non-streaming) ────────────────────────────────

// ── /api/generate/chain/stream (SSE) ───────────────────────────────────

#[tokio::test]
async fn retake_chain_job_409_surfaces_splice_rejection_body() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/api/chain-jobs/job-1/retake"))
        .respond_with(
            ResponseTemplate::new(409)
                .set_body_string("splice retake cannot precede a smooth transition"),
        )
        .mount(&server)
        .await;

    let client = MoldClient::new(&server.uri());
    let err = client
        .retake_chain_job(
            "job-1",
            &RetakeRequest {
                stage_idx: 0,
                mode: RetakeMode::Splice,
                seed_offset: None,
                prompt: None,
            },
        )
        .await
        .expect_err("409 splice rejection must error");
    let msg = format!("{err:#}");

    assert!(
        msg.contains("server error 409 Conflict"),
        "error should include status, got: {msg}",
    );
    assert!(
        msg.contains("splice retake cannot precede a smooth transition"),
        "error should include server body, got: {msg}",
    );
}

/// `finalized.output` is the job-relative MP4 amend/retake decode; the print a
/// client fetches is `gallery_filename`. Every chain run hydrates from this
/// value, so reading the artifact path here 404s on the gallery route.
#[tokio::test]
async fn stream_chain_job_events_hands_back_the_gallery_filename() {
    let server = MockServer::start().await;
    let body = concat!(
        "event: chain_job\n",
        "data: {\"type\":\"finalizing\",\"total_frames\":97}\n\n",
        "event: chain_job\n",
        "data: {\"type\":\"finalized\",\"output\":\"final/output-1.mp4\",\"take\":1,\"gallery_filename\":\"mold-chain-abc-take-1.gif\"}\n\n",
        "event: chain_job\n",
        "data: {\"type\":\"state_changed\",\"state\":\"completed\",\"error\":null}\n\n",
    );
    Mock::given(method("GET"))
        .and(path("/api/chain-jobs/job-1/events"))
        .respond_with(
            ResponseTemplate::new(200)
                .insert_header("content-type", "text/event-stream")
                .set_body_string(body),
        )
        .mount(&server)
        .await;
    let (tx, _rx) = tokio::sync::mpsc::unbounded_channel();
    let outcome = MoldClient::new(&server.uri())
        .stream_chain_job_events("job-1", tx)
        .await
        .unwrap();
    assert_eq!(
        outcome.state,
        mold_core::chain_job::ChainJobState::Completed
    );
    assert_eq!(outcome.output.as_deref(), Some("mold-chain-abc-take-1.gif"));
}
