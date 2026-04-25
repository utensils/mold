use mold_catalog::scanner::ScanOptions;
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
        // include_nsfw is irrelevant here; both fixtures are SFW. The
        // assertion checks the safetensor/pt distinction.
        ..ScanOptions::default()
    };
    let entries = civitai::scan(&server.uri(), &opts, &["SDXL 1.0"])
        .await
        .expect("scan");
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
        ..ScanOptions::default()
    };
    let entries = civitai::scan(&server.uri(), &opts, &["SDXL 1.0"])
        .await
        .expect("scan");
    assert!(entries.is_empty());
}
