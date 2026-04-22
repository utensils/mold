//! Integration tests for `MoldClient::generate_chain{,_stream}` using
//! `wiremock` to simulate the `/api/generate/chain` server endpoints.
//!
//! These tests pin the HTTP surface (method, path, JSON request body) and
//! verify error translation (422 → Validation, 404 empty → None on stream,
//! 404 with body → ModelNotFound). They do NOT exercise real LTX-2 work —
//! the server side lands in Phase 2.

use base64::Engine as _;
use mold_core::chain::{ChainProgressEvent, ChainRequest, ChainStage, SseChainCompleteEvent};
use mold_core::error::MoldError;
use mold_core::types::OutputFormat;
use mold_core::MoldClient;
use wiremock::matchers::{body_json_schema, method, path};
use wiremock::{Mock, MockServer, ResponseTemplate};

fn mold_error(err: &anyhow::Error) -> &MoldError {
    err.downcast_ref::<MoldError>()
        .unwrap_or_else(|| panic!("not a MoldError: {err}"))
}

fn sample_request() -> ChainRequest {
    ChainRequest {
        model: "ltx-2-19b-distilled:fp8".into(),
        stages: vec![ChainStage {
            prompt: "a cat walking".into(),
            frames: 97,
            source_image: None,
            negative_prompt: None,
            seed_offset: None,
        }],
        motion_tail_frames: 4,
        width: 1216,
        height: 704,
        fps: 24,
        seed: Some(42),
        steps: 8,
        guidance: 3.0,
        strength: 1.0,
        output_format: OutputFormat::Mp4,
        placement: None,
        prompt: None,
        total_frames: None,
        clip_frames: None,
        source_image: None,
    }
}

fn minimal_chain_response_json() -> serde_json::Value {
    serde_json::json!({
        "video": {
            "data": [],
            "format": "mp4",
            "width": 1216,
            "height": 704,
            "frames": 97,
            "fps": 24,
            "thumbnail": []
        },
        "stage_count": 1
    })
}

// ── /api/generate/chain (non-streaming) ────────────────────────────────

#[tokio::test]
async fn generate_chain_posts_to_correct_endpoint_and_parses_response() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/api/generate/chain"))
        .and(body_json_schema::<ChainRequest>)
        .respond_with(ResponseTemplate::new(200).set_body_json(minimal_chain_response_json()))
        .expect(1)
        .mount(&server)
        .await;

    let client = MoldClient::new(&server.uri());
    let resp = client
        .generate_chain(&sample_request())
        .await
        .expect("non-streaming chain should succeed on 200");
    assert_eq!(resp.stage_count, 1);
    assert_eq!(resp.video.frames, 97);
    assert_eq!(resp.video.format, OutputFormat::Mp4);
}

#[tokio::test]
async fn generate_chain_surfaces_422_as_validation_error() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/api/generate/chain"))
        .respond_with(ResponseTemplate::new(422).set_body_string("frames must be 8k+1"))
        .mount(&server)
        .await;

    let client = MoldClient::new(&server.uri());
    let err = client
        .generate_chain(&sample_request())
        .await
        .expect_err("422 must error");
    assert!(
        matches!(mold_error(&err), MoldError::Validation(msg) if msg.contains("8k+1")),
        "422 must translate to MoldError::Validation carrying the body",
    );
}

#[tokio::test]
async fn generate_chain_translates_404_with_body_to_model_not_found() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/api/generate/chain"))
        .respond_with(ResponseTemplate::new(404).set_body_string("model 'ltx-2-foo' not found"))
        .mount(&server)
        .await;

    let client = MoldClient::new(&server.uri());
    let err = client
        .generate_chain(&sample_request())
        .await
        .expect_err("404 with body must error");
    assert!(
        matches!(mold_error(&err), MoldError::ModelNotFound(msg) if msg.contains("ltx-2-foo")),
        "404-with-body must translate to MoldError::ModelNotFound",
    );
}

#[tokio::test]
async fn generate_chain_empty_404_fails_loudly_instead_of_silently() {
    // Non-streaming callers have no fallback path — an empty 404 means the
    // server predates render-chain v1, which is a hard error (unlike the
    // streaming case where Ok(None) signals "try the non-streaming path").
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/api/generate/chain"))
        .respond_with(ResponseTemplate::new(404).set_body_string(""))
        .mount(&server)
        .await;

    let client = MoldClient::new(&server.uri());
    let err = client
        .generate_chain(&sample_request())
        .await
        .expect_err("empty 404 must error on non-streaming path");
    let msg = format!("{err}");
    assert!(
        msg.contains("chain endpoint not found"),
        "error must name the missing endpoint, got: {msg}",
    );
}

// ── /api/generate/chain/stream (SSE) ───────────────────────────────────

#[tokio::test]
async fn generate_chain_stream_returns_none_on_empty_404() {
    // An empty 404 on the streaming endpoint means the server doesn't
    // support chain SSE yet — callers are expected to fall back to the
    // non-streaming path.
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/api/generate/chain/stream"))
        .respond_with(ResponseTemplate::new(404).set_body_string(""))
        .mount(&server)
        .await;

    let client = MoldClient::new(&server.uri());
    let (tx, _rx) = tokio::sync::mpsc::unbounded_channel::<ChainProgressEvent>();
    let out = client
        .generate_chain_stream(&sample_request(), tx)
        .await
        .expect("empty 404 should resolve to Ok(None)");
    assert!(out.is_none(), "empty 404 must signal unsupported endpoint");
}

#[tokio::test]
async fn generate_chain_stream_parses_progress_and_complete_events() {
    let b64 = base64::engine::general_purpose::STANDARD;
    let video_bytes = b"FAKE_MP4_BYTES";
    let thumb_bytes = b"THUMB";
    let complete = SseChainCompleteEvent {
        video: b64.encode(video_bytes),
        format: OutputFormat::Mp4,
        width: 1216,
        height: 704,
        frames: 97,
        fps: 24,
        thumbnail: Some(b64.encode(thumb_bytes)),
        gif_preview: None,
        has_audio: false,
        duration_ms: Some(4040),
        audio_sample_rate: None,
        audio_channels: None,
        stage_count: 1,
        gpu: Some(0),
        generation_time_ms: Some(45_000),
    };
    let progress = ChainProgressEvent::DenoiseStep {
        stage_idx: 0,
        step: 4,
        total: 8,
    };
    // Build a chunk-encoded SSE body carrying one progress event then
    // complete. `\n\n` terminates each SSE event.
    let body = format!(
        "event: progress\ndata: {}\n\nevent: complete\ndata: {}\n\n",
        serde_json::to_string(&progress).unwrap(),
        serde_json::to_string(&complete).unwrap(),
    );

    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/api/generate/chain/stream"))
        .respond_with(
            ResponseTemplate::new(200)
                .insert_header("content-type", "text/event-stream")
                .set_body_string(body),
        )
        .mount(&server)
        .await;

    let client = MoldClient::new(&server.uri());
    let (tx, mut rx) = tokio::sync::mpsc::unbounded_channel::<ChainProgressEvent>();
    let resp = client
        .generate_chain_stream(&sample_request(), tx)
        .await
        .expect("SSE stream should succeed")
        .expect("complete event should yield a response");

    assert_eq!(resp.stage_count, 1);
    assert_eq!(resp.video.data, video_bytes);
    assert_eq!(resp.video.thumbnail, thumb_bytes);
    assert_eq!(resp.gpu, Some(0));
    let ev = rx.recv().await.expect("progress event should be forwarded");
    assert_eq!(ev, progress);
}
