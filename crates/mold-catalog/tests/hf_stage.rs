use mold_catalog::families::Family;
use mold_catalog::scanner::ScanOptions;
use mold_catalog::stages::hf;
use wiremock::matchers::{method, path, query_param};
use wiremock::{Mock, MockServer, ResponseTemplate};

#[tokio::test]
async fn scan_family_yields_seed_plus_finetunes() {
    let server = MockServer::start().await;

    // Detail for the seed.
    Mock::given(method("GET"))
        .and(path("/api/models/black-forest-labs/FLUX.1-dev"))
        .respond_with(
            ResponseTemplate::new(200).set_body_string(include_str!("fixtures/hf_flux_dev.json")),
        )
        .mount(&server)
        .await;

    // Tree for the seed.
    Mock::given(method("GET"))
        .and(path("/api/models/black-forest-labs/FLUX.1-dev/tree/main"))
        .respond_with(
            ResponseTemplate::new(200)
                .set_body_string(include_str!("fixtures/hf_flux_dev_tree.json")),
        )
        .mount(&server)
        .await;

    // Empty finetune page (no finetunes for this fixture).
    Mock::given(method("GET"))
        .and(path("/api/models"))
        .and(query_param(
            "filter",
            "base_model:black-forest-labs/FLUX.1-dev",
        ))
        .respond_with(ResponseTemplate::new(200).set_body_string("[]"))
        .mount(&server)
        .await;

    let opts = ScanOptions::default();
    let entries = hf::scan_family(
        &server.uri(),
        &opts,
        Family::Flux,
        &["black-forest-labs/FLUX.1-dev"],
    )
    .await
    .expect("scan");

    assert!(!entries.is_empty(), "expected at least the seed entry");
    let seed = entries
        .iter()
        .find(|e| e.source_id == "black-forest-labs/FLUX.1-dev")
        .expect("seed present");
    assert_eq!(seed.family, Family::Flux);
}
