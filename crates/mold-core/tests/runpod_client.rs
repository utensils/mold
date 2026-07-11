//! Integration tests for `RunPodClient` using `wiremock` to simulate the
//! RunPod REST + GraphQL API.
//!
//! These tests verify error translation (401 → RunPodAuth, 404 → RunPodNotFound,
//! 500 "no resources" → RunPodNoStock), JSON parsing, and GraphQL→REST
//! adapter behaviour in `gpu_types` / `datacenters`.

use mold_core::error::MoldError;
use mold_core::runpod::{
    CreateNetworkVolumeRequest, CreatePodRequest, RunPodClient, RunPodSettings,
    UpdateNetworkVolumeRequest,
};
use wiremock::matchers::{bearer_token, body_json, method, path};
use wiremock::{Mock, MockServer, ResponseTemplate};

// ── helpers ────────────────────────────────────────────────────────────

async fn client_with_mock() -> (RunPodClient, MockServer) {
    let server = MockServer::start().await;
    let settings = RunPodSettings {
        api_key: Some("test-key".into()),
        endpoint: Some(server.uri()),
        ..Default::default()
    };
    let client = RunPodClient::from_settings(&settings).expect("client");
    (client, server)
}

fn mold_error(err: &anyhow::Error) -> &MoldError {
    err.downcast_ref::<MoldError>()
        .unwrap_or_else(|| panic!("not MoldError: {err}"))
}

// ── auth ───────────────────────────────────────────────────────────────

#[tokio::test]
async fn missing_api_key_yields_auth_error() {
    std::env::remove_var("RUNPOD_API_KEY");
    let err = RunPodClient::from_settings(&RunPodSettings::default()).unwrap_err();
    assert!(matches!(err, MoldError::RunPodAuth(_)), "got {err:?}");
}

#[tokio::test]
async fn bearer_token_is_sent() {
    let (client, server) = client_with_mock().await;
    Mock::given(method("GET"))
        .and(path("/pods"))
        .and(bearer_token("test-key"))
        .respond_with(ResponseTemplate::new(200).set_body_string("[]"))
        .mount(&server)
        .await;
    let pods = client.list_pods().await.expect("pods");
    assert!(pods.is_empty());
}

#[tokio::test]
async fn unauthorized_maps_to_runpod_auth_error() {
    let (client, server) = client_with_mock().await;
    Mock::given(method("GET"))
        .and(path("/pods"))
        .respond_with(ResponseTemplate::new(401).set_body_string(r#"{"error":"bad key"}"#))
        .mount(&server)
        .await;
    let err = client.list_pods().await.unwrap_err();
    assert!(matches!(mold_error(&err), MoldError::RunPodAuth(_)));
}

#[tokio::test]
async fn forbidden_maps_to_runpod_auth_error() {
    let (client, server) = client_with_mock().await;
    Mock::given(method("GET"))
        .and(path("/pods"))
        .respond_with(ResponseTemplate::new(403))
        .mount(&server)
        .await;
    let err = client.list_pods().await.unwrap_err();
    assert!(matches!(mold_error(&err), MoldError::RunPodAuth(_)));
}

// ── not found ──────────────────────────────────────────────────────────

#[tokio::test]
async fn get_missing_pod_yields_not_found() {
    let (client, server) = client_with_mock().await;
    Mock::given(method("GET"))
        .and(path("/pods/missing"))
        .respond_with(ResponseTemplate::new(404).set_body_string(r#"{"error":"not found"}"#))
        .mount(&server)
        .await;
    let err = client.get_pod("missing").await.unwrap_err();
    assert!(matches!(mold_error(&err), MoldError::RunPodNotFound(_)));
}

// ── no stock ───────────────────────────────────────────────────────────

#[tokio::test]
async fn create_no_stock_maps_to_runpod_no_stock() {
    let (client, server) = client_with_mock().await;
    Mock::given(method("POST"))
        .and(path("/pods"))
        .respond_with(ResponseTemplate::new(500).set_body_string(
            r#"{"error":"This machine does not have the resources to deploy your pod"}"#,
        ))
        .mount(&server)
        .await;
    let req = CreatePodRequest {
        name: "t".into(),
        image_name: "x".into(),
        gpu_type_ids: vec!["NVIDIA GeForce RTX 4090".into()],
        cloud_type: "SECURE".into(),
        gpu_count: 1,
        container_disk_in_gb: 20,
        volume_in_gb: 50,
        volume_mount_path: "/workspace".into(),
        ports: vec!["7680/http".into()],
        env: serde_json::Map::new(),
        ..Default::default()
    };
    let err = client.create_pod(&req).await.unwrap_err();
    let me = mold_error(&err);
    assert!(matches!(me, MoldError::RunPodNoStock(_)));
    assert!(format!("{me}").to_lowercase().contains("does not have"));
}

// ── happy paths ────────────────────────────────────────────────────────

#[tokio::test]
async fn list_pods_parses_minimal_json() {
    let (client, server) = client_with_mock().await;
    Mock::given(method("GET"))
        .and(path("/pods"))
        .respond_with(
            ResponseTemplate::new(200)
                .set_body_string(r#"[{"id":"abc","desiredStatus":"RUNNING","costPerHr":0.99}]"#),
        )
        .mount(&server)
        .await;
    let pods = client.list_pods().await.unwrap();
    assert_eq!(pods.len(), 1);
    assert_eq!(pods[0].id, "abc");
    assert_eq!(pods[0].desired_status, "RUNNING");
    assert_eq!(pods[0].cost_per_hr, 0.99);
}

#[tokio::test]
async fn create_pod_returns_parsed_pod() {
    let (client, server) = client_with_mock().await;
    Mock::given(method("POST"))
        .and(path("/pods"))
        .respond_with(ResponseTemplate::new(200).set_body_string(
            r#"{"id":"new-pod","desiredStatus":"RUNNING","costPerHr":0.69,"name":"t"}"#,
        ))
        .mount(&server)
        .await;
    let req = CreatePodRequest {
        name: "t".into(),
        image_name: "x".into(),
        gpu_type_ids: vec!["NVIDIA GeForce RTX 5090".into()],
        cloud_type: "SECURE".into(),
        gpu_count: 1,
        container_disk_in_gb: 20,
        volume_in_gb: 50,
        volume_mount_path: "/workspace".into(),
        ports: vec!["7680/http".into()],
        env: serde_json::Map::new(),
        ..Default::default()
    };
    let pod = client.create_pod(&req).await.unwrap();
    assert_eq!(pod.id, "new-pod");
    assert_eq!(pod.name.as_deref(), Some("t"));
    assert_eq!(pod.cost_per_hr, 0.69);
}

#[tokio::test]
async fn delete_pod_requires_success_status() {
    let (client, server) = client_with_mock().await;
    Mock::given(method("DELETE"))
        .and(path("/pods/abc"))
        .respond_with(ResponseTemplate::new(204))
        .mount(&server)
        .await;
    assert!(client.delete_pod("abc").await.is_ok());
}

#[tokio::test]
async fn network_volume_crud_uses_rest_contract() {
    let (client, server) = client_with_mock().await;
    let volume_json = r#"{"id":"nv-1","name":"models","dataCenterId":"US-KS-2","size":100}"#;

    Mock::given(method("GET"))
        .and(path("/networkvolumes/nv-1"))
        .respond_with(ResponseTemplate::new(200).set_body_string(volume_json))
        .mount(&server)
        .await;
    Mock::given(method("POST"))
        .and(path("/networkvolumes"))
        .and(body_json(serde_json::json!({
            "name": "models",
            "size": 100,
            "dataCenterId": "US-KS-2"
        })))
        .respond_with(ResponseTemplate::new(200).set_body_string(volume_json))
        .mount(&server)
        .await;
    Mock::given(method("PATCH"))
        .and(path("/networkvolumes/nv-1"))
        .and(body_json(
            serde_json::json!({"name": "models-v2", "size": 150}),
        ))
        .respond_with(ResponseTemplate::new(200).set_body_string(
            r#"{"id":"nv-1","name":"models-v2","dataCenterId":"US-KS-2","size":150}"#,
        ))
        .mount(&server)
        .await;
    Mock::given(method("DELETE"))
        .and(path("/networkvolumes/nv-1"))
        .respond_with(ResponseTemplate::new(204))
        .mount(&server)
        .await;

    let found = client.get_network_volume("nv-1").await.unwrap();
    assert_eq!(found.data_center_id, "US-KS-2");
    let created = client
        .create_network_volume(&CreateNetworkVolumeRequest {
            name: "models".into(),
            size: 100,
            data_center_id: "US-KS-2".into(),
        })
        .await
        .unwrap();
    assert_eq!(created.size, 100);
    let updated = client
        .update_network_volume(
            "nv-1",
            &UpdateNetworkVolumeRequest {
                name: Some("models-v2".into()),
                size: Some(150),
            },
        )
        .await
        .unwrap();
    assert_eq!(updated.name, "models-v2");
    assert!(client.delete_network_volume("nv-1").await.is_ok());
}

#[tokio::test]
async fn network_volume_delete_fails_closed_when_attachment_check_fails() {
    let (client, server) = client_with_mock().await;
    Mock::given(method("GET"))
        .and(path("/pods"))
        .respond_with(ResponseTemplate::new(503).set_body_string("temporarily unavailable"))
        .mount(&server)
        .await;

    assert!(client
        .delete_network_volume_if_detached("nv-1")
        .await
        .is_err());
    let requests = server.received_requests().await.unwrap();
    assert!(requests
        .iter()
        .all(|request| !(request.method.as_str() == "DELETE"
            && request.url.path() == "/networkvolumes/nv-1")));
}

#[tokio::test]
async fn stop_and_start_pod_happy_paths() {
    let (client, server) = client_with_mock().await;
    Mock::given(method("POST"))
        .and(path("/pods/abc/stop"))
        .respond_with(ResponseTemplate::new(200))
        .mount(&server)
        .await;
    Mock::given(method("POST"))
        .and(path("/pods/abc/start"))
        .respond_with(ResponseTemplate::new(200))
        .mount(&server)
        .await;
    assert!(client.stop_pod("abc").await.is_ok());
    assert!(client.start_pod("abc").await.is_ok());
}

#[tokio::test]
async fn graphql_endpoint_override_routes_to_mock() {
    // The endpoint override in RunPodSettings should also redirect
    // GraphQL calls (user/, gpu_types/, datacenters) to the mock server,
    // not to production. Verify by watching for a POST on root-ish path.
    let (client, server) = client_with_mock().await;
    Mock::given(method("POST"))
        .and(path("/"))
        .respond_with(ResponseTemplate::new(200).set_body_string(
            r#"{"data":{"myself":{"id":"u1","email":"x@y.com","clientBalance":10.0,"currentSpendPerHr":0.5,"spendLimit":null}}}"#,
        ))
        .mount(&server)
        .await;
    let info = client.user().await.expect("user info via mocked graphql");
    assert_eq!(info.id, "u1");
    assert_eq!(info.email, "x@y.com");
}
