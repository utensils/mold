use mold_catalog::scanner::{ScanError, ScanOptions};
use mold_catalog::stages::civitai;
use wiremock::matchers::{method, path, query_param};
use wiremock::{Mock, MockServer, ResponseTemplate};

const RESPONSE: &str = r#"{
    "items": [
        {
            "id": 100,
            "name": "Real Photo XL",
            "type": "Checkpoint",
            "nsfw": false,
            "creator": { "username": "alice" },
            "stats": { "downloadCount": 950000, "rating": 4.8, "favoriteCount": 5 },
            "tags": [],
            "modelVersions": [{
                "id": 200,
                "name": "v1",
                "baseModel": "SDXL 1.0",
                "baseModelType": "Standard",
                "files": [
                    { "id": 1, "name": "x.safetensors", "sizeKB": 100,
                      "downloadCount": 1, "metadata": { "format": "SafeTensor" },
                      "downloadUrl": "u", "hashes": {} }
                ],
                "images": []
            }]
        },
        {
            "id": 101,
            "name": "Pickle Trap",
            "type": "Checkpoint",
            "nsfw": false,
            "creator": { "username": "bob" },
            "stats": { "downloadCount": 1, "favoriteCount": 0 },
            "tags": [],
            "modelVersions": [{
                "id": 201,
                "name": "v1",
                "baseModel": "SDXL 1.0",
                "baseModelType": "Standard",
                "files": [
                    { "id": 2, "name": "x.pt", "sizeKB": 100,
                      "downloadCount": 1, "metadata": { "format": "PickleTensor" },
                      "downloadUrl": "u", "hashes": {} }
                ],
                "images": []
            }]
        }
    ],
    "metadata": { "totalPages": 1 }
}"#;

#[tokio::test]
async fn scan_drops_pickle_files_and_keeps_safetensors() {
    let server = MockServer::start().await;
    Mock::given(method("GET"))
        .and(path("/api/v1/models"))
        .and(query_param("baseModels", "SDXL 1.0"))
        .respond_with(ResponseTemplate::new(200).set_body_string(RESPONSE))
        .mount(&server)
        .await;

    let opts = ScanOptions {
        hf_request_delay: std::time::Duration::ZERO,
        civitai_request_delay: std::time::Duration::ZERO,
        max_429_retries: 0,
        ..ScanOptions::default()
    };
    let (entries, err) = civitai::scan(&server.uri(), &opts, &["SDXL 1.0"]).await;
    assert!(err.is_none(), "no rate-limit on this fixture: {err:?}");
    assert_eq!(entries.len(), 1, "pickle entry must be dropped");
    assert_eq!(entries[0].source_id, "200");
}

#[tokio::test]
async fn scan_passes_token_via_bearer_when_present() {
    let server = MockServer::start().await;
    Mock::given(method("GET"))
        .and(path("/api/v1/models"))
        .and(query_param("baseModels", "SDXL 1.0"))
        .and(wiremock::matchers::header(
            "Authorization",
            "Bearer civitai-secret",
        ))
        .respond_with(
            ResponseTemplate::new(200)
                .set_body_string(r#"{"items":[],"metadata":{"totalPages":1}}"#),
        )
        .mount(&server)
        .await;

    let opts = ScanOptions {
        civitai_token: Some("civitai-secret".into()),
        hf_request_delay: std::time::Duration::ZERO,
        civitai_request_delay: std::time::Duration::ZERO,
        max_429_retries: 0,
        ..ScanOptions::default()
    };
    let (entries, err) = civitai::scan(&server.uri(), &opts, &["SDXL 1.0"]).await;
    assert!(err.is_none());
    assert!(entries.is_empty());
}

/// Regression: when the Civitai page-walk hits a 429 mid-scan (e.g. their
/// per-IP cooldown kicks in on page 11 after 10 successful pages), the
/// entries already collected on earlier pages MUST be returned, not
/// discarded. The orchestrator-side observation was that a real refresh
/// against `civitai.com` would walk 10 pages, 429 on page 11 with
/// retries exhausted, and the catalog would end up with ZERO Civitai
/// rows for that family.
#[tokio::test]
async fn scan_preserves_partial_results_when_later_page_429s() {
    let server = MockServer::start().await;

    // Page 1: succeeds with one entry.
    Mock::given(method("GET"))
        .and(path("/api/v1/models"))
        .and(query_param("baseModels", "SDXL 1.0"))
        .and(query_param("page", "1"))
        .respond_with(ResponseTemplate::new(200).set_body_string(
            r#"{
                "items": [{
                    "id": 100, "name": "Page1 Model",
                    "type": "Checkpoint", "nsfw": false,
                    "creator": { "username": "alice" },
                    "stats": { "downloadCount": 950000, "rating": 4.8, "favoriteCount": 5 },
                    "tags": [],
                    "modelVersions": [{
                        "id": 200, "name": "v1",
                        "baseModel": "SDXL 1.0", "baseModelType": "Standard",
                        "files": [{
                            "id": 1, "name": "x.safetensors", "sizeKB": 100,
                            "downloadCount": 1, "metadata": { "format": "SafeTensor" },
                            "downloadUrl": "u", "hashes": {}
                        }],
                        "images": []
                    }]
                }],
                "metadata": { "totalPages": 5, "nextPage": "page=2" }
            }"#,
        ))
        .mount(&server)
        .await;

    // Page 2: returns 429. With max_429_retries=0 the throttle layer
    // surfaces the rate-limit error immediately.
    Mock::given(method("GET"))
        .and(path("/api/v1/models"))
        .and(query_param("baseModels", "SDXL 1.0"))
        .and(query_param("page", "2"))
        .respond_with(ResponseTemplate::new(429))
        .mount(&server)
        .await;

    let opts = ScanOptions {
        hf_request_delay: std::time::Duration::ZERO,
        civitai_request_delay: std::time::Duration::ZERO,
        max_429_retries: 0,
        ..ScanOptions::default()
    };

    let (entries, err) = civitai::scan(&server.uri(), &opts, &["SDXL 1.0"]).await;
    assert!(
        matches!(
            err,
            Some(ScanError::RateLimited {
                host: "civitai.com"
            })
        ),
        "must surface rate-limit error so orchestrator can mark family RateLimited: got {err:?}",
    );
    assert_eq!(
        entries.len(),
        1,
        "page-1 entry MUST be preserved on partial-success rate limit, got {entries:?}",
    );
    assert_eq!(entries[0].source_id, "200");
}
