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
