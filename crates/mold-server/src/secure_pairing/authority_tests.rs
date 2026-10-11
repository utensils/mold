use super::*;
use crate::auth::{inject_auth_state, require_api_key, ApiKeySet, AuthState};
use axum::{
    body::Body,
    extract::ConnectInfo,
    http::{Request, StatusCode},
    routing::{get, post},
    Extension, Router,
};
use base64::{engine::general_purpose::URL_SAFE_NO_PAD, Engine as _};
use std::net::SocketAddr;
use tower::ServiceExt;
fn fixture() -> (Arc<ApiKeySet>, Identity, String, DeviceGrant) {
    fixture_with_db(Arc::new(Some(
        mold_db::MetadataDb::open_in_memory().unwrap(),
    )))
}
fn fixture_with_db(
    db: Arc<Option<mold_db::MetadataDb>>,
) -> (Arc<ApiKeySet>, Identity, String, DeviceGrant) {
    let mut keys = ApiKeySet::new_with_metadata_db(HashSet::from(["operator".into()]), db, "host");
    keys.secure = Some(Arc::new(SecurePairing::new(
        Identity::generate().unwrap(),
        Vec::new(),
    )));
    let secure = keys.secure.as_ref().unwrap();
    let issued = secure.issue_invite().unwrap();
    let device = Identity::generate().unwrap();
    let mut secret = [0; 64];
    secret[..32].copy_from_slice(&mold_secure::decode_hex::<32>(&issued.invite_seed).unwrap());
    let invitation = Identity::from_bytes(&secret).unwrap();
    let context = RequestContext {
        binding: device.public_binding(),
        grant: None,
        invite: Some(issued.invite.clone()),
        handshake_hash: hex(&[8; 32]),
        stream_id: 1,
        issued_at: unix_time(),
        request_nonce: hex(&[9; 16]),
        signature: String::new(),
    };
    let proof = hex(&invitation.sign(
        &mold_secure::claim_bytes(
            &secure.namespace(),
            &issued.invite.id,
            &[8; 32],
            &device.signing_public(),
            &device.noise_public(),
        )
        .unwrap(),
    ));
    let (credential, grant) = keys
        .claim_secure_pairing(&context, &proof, "phone", "iphone")
        .unwrap();
    (Arc::new(keys), device, credential, grant)
}
fn context(
    keys: &ApiKeySet,
    device: &Identity,
    grant: &DeviceGrant,
    method: &str,
    path: &str,
) -> RequestContext {
    let mut nonce = [0; 16];
    getrandom::fill(&mut nonce).unwrap();
    let mut context = RequestContext {
        binding: device.public_binding(),
        grant: Some(grant.clone()),
        invite: None,
        handshake_hash: hex(&[7; 32]),
        stream_id: 1,
        issued_at: unix_time(),
        request_nonce: hex(&nonce),
        signature: String::new(),
    };
    context
        .sign(&keys.secure.as_ref().unwrap().identity, method, path)
        .unwrap();
    context
}
fn app(keys: Arc<ApiKeySet>, peer: &str) -> Router {
    let auth: AuthState = Some(keys);
    Router::new()
        .route("/api/status", get(|| async { "private status" }))
        .route(
            "/api/pairing/sessions",
            post(
                |authority: Option<Extension<crate::auth::PairingAuthority>>| async move {
                    if authority.is_some() {
                        StatusCode::OK
                    } else {
                        StatusCode::FORBIDDEN
                    }
                },
            ),
        )
        .route("/api/other", get(|| async { "other" }))
        .route("/api/pairing/secure/check", get(check))
        .route("/api/pairing/secure/renew", post(renew))
        .layer(axum::middleware::from_fn(require_api_key))
        .layer(axum::middleware::from_fn_with_state(
            auth,
            inject_auth_state,
        ))
        .layer(Extension(ConnectInfo(peer.parse::<SocketAddr>().unwrap())))
}
fn request(path: &str, key: &str, context: Option<&RequestContext>) -> Request<Body> {
    let mut request = Request::get(path).header("x-api-key", key);
    if let Some(context) = context {
        request = request.header(
            "x-mold-secure-peer",
            URL_SAFE_NO_PAD.encode(serde_json::to_vec(context).unwrap()),
        );
    }
    request.body(Body::empty()).unwrap()
}
#[test]
fn pairing_keeps_each_client_surface_kind() {
    for kind in [
        "iphone", "ipad", "android", "browser", "web", "desktop", "cli",
    ] {
        assert_eq!(crate::auth::normalized_client_kind(kind), kind);
    }
}
#[tokio::test]
async fn plaintext_api_without_socket_peer_fails_closed() {
    let (keys, _, _, _) = fixture();
    let app = Router::new()
        .route("/api/status", get(|| async { "private" }))
        .layer(axum::middleware::from_fn(require_api_key))
        .layer(axum::middleware::from_fn_with_state(
            Some(keys),
            inject_auth_state,
        ));
    assert_eq!(
        app.oneshot(request("/api/status", "operator", None))
            .await
            .unwrap()
            .status(),
        StatusCode::UPGRADE_REQUIRED
    );
}
#[tokio::test]
async fn remote_browser_bootstrap_redirects_to_independent_trusted_code() {
    let app = Router::new()
        .fallback(|| async { "untrusted LAN shell" })
        .layer(axum::middleware::from_fn(require_api_key))
        .layer(Extension(ConnectInfo(
            "192.168.1.44:1000".parse::<SocketAddr>().unwrap(),
        )));
    for method in ["GET", "HEAD"] {
        for path in [
            "/?pairing=private",
            "/machines/host?token=private",
            "/assets/index.js",
        ] {
            let response = app
                .clone()
                .oneshot(
                    Request::builder()
                        .method(method)
                        .uri(path)
                        .body(Body::empty())
                        .unwrap(),
                )
                .await
                .unwrap();
            assert_eq!(response.status(), StatusCode::SEE_OTHER);
            assert_eq!(
                response.headers()["location"],
                "https://utensils.io/mold/studio/"
            );
            assert_eq!(response.headers()["cache-control"], "no-store");
            assert!(axum::body::to_bytes(response.into_body(), 1024)
                .await
                .unwrap()
                .is_empty());
        }
    }
}
#[tokio::test]
async fn plaintext_remote_api_is_refused_even_with_operator_key_or_auth_disabled() {
    let (keys, _, _, _) = fixture();
    for peer in ["192.168.1.44:1000", "[2001:db8::1]:1000"] {
        for auth in [Some(keys.clone()), None] {
            let app = Router::new()
                .route("/api/status", get(|| async { "private" }))
                .route("/health", get(|| async { "health" }))
                .layer(axum::middleware::from_fn(require_api_key))
                .layer(axum::middleware::from_fn_with_state(
                    auth,
                    inject_auth_state,
                ))
                .layer(Extension(ConnectInfo(peer.parse::<SocketAddr>().unwrap())));
            for path in ["/api/status", "/health"] {
                let mut request = request(path, "operator", None);
                request
                    .headers_mut()
                    .insert("x-forwarded-for", "127.0.0.1".parse().unwrap());
                request
                    .headers_mut()
                    .insert("host", "localhost".parse().unwrap());
                assert_eq!(
                    app.clone().oneshot(request).await.unwrap().status(),
                    StatusCode::UPGRADE_REQUIRED
                );
            }
        }
    }
    for peer in ["127.0.0.1:1000", "[::1]:1000"] {
        assert_eq!(
            app(keys.clone(), peer)
                .oneshot(request("/api/status", "operator", None))
                .await
                .unwrap()
                .status(),
            StatusCode::OK
        );
    }
}
#[tokio::test]
async fn middleware_binds_encrypted_device_credential_to_fresh_local_request_proof() {
    let (keys, device, credential, grant) = fixture();
    let app = app(keys.clone(), "127.0.0.1:1000");
    assert_eq!(
        app.clone()
            .oneshot(request("/api/status", &credential, None))
            .await
            .unwrap()
            .status(),
        StatusCode::UNAUTHORIZED
    );
    let proof = context(&keys, &device, &grant, "GET", "/api/status");
    assert_eq!(
        app.clone()
            .oneshot(request("/api/status", &credential, Some(&proof)))
            .await
            .unwrap()
            .status(),
        StatusCode::OK
    );
    assert_eq!(
        app.clone()
            .oneshot(request("/api/status", &credential, Some(&proof)))
            .await
            .unwrap()
            .status(),
        StatusCode::UNAUTHORIZED
    );
    let proof = context(&keys, &device, &grant, "GET", "/api/status");
    assert_eq!(
        app.clone()
            .oneshot(request("/api/other", &credential, Some(&proof)))
            .await
            .unwrap()
            .status(),
        StatusCode::UNAUTHORIZED
    );
    let proof = context(&keys, &device, &grant, "GET", "/api/status");
    assert_eq!(
        app.clone()
            .oneshot(request("/api/status", "operator", Some(&proof)))
            .await
            .unwrap()
            .status(),
        StatusCode::UNAUTHORIZED
    );
    let proof = context(&keys, &device, &grant, "GET", "/api/status");
    assert_eq!(
        super::authority_tests::app(keys.clone(), "192.168.1.44:1000")
            .oneshot(request("/api/status", &credential, Some(&proof)))
            .await
            .unwrap()
            .status(),
        StatusCode::UPGRADE_REQUIRED
    );
    let client = keys.paired_clients().pop().unwrap();
    keys.revoke_paired_client(&client.id).unwrap();
    let proof = context(&keys, &device, &grant, "GET", "/api/status");
    assert_eq!(
        app.oneshot(request("/api/status", &credential, Some(&proof)))
            .await
            .unwrap()
            .status(),
        StatusCode::UNAUTHORIZED
    );
}
#[tokio::test]
async fn signed_control_poll_is_read_only_local_one_use_and_reflects_revocation_immediately() {
    let (keys, _device, _credential, grant) = fixture();
    let app = app(keys.clone(), "127.0.0.1:1000");
    let mut context = ControlContext {
        issued_at: unix_time(),
        nonce: hex(&[1; 16]),
        signature: String::new(),
    };
    context
        .sign(&keys.secure.as_ref().unwrap().identity)
        .unwrap();
    let header = URL_SAFE_NO_PAD.encode(serde_json::to_vec(&context).unwrap());
    let make = || {
        Request::get("/api/pairing/secure/check")
            .header("x-mold-secure-control", &header)
            .body(Body::empty())
            .unwrap()
    };
    let response = app.clone().oneshot(make()).await.unwrap();
    assert_eq!(response.status(), StatusCode::OK);
    let json: serde_json::Value = serde_json::from_slice(
        &axum::body::to_bytes(response.into_body(), 65536)
            .await
            .unwrap(),
    )
    .unwrap();
    assert_eq!(json["active"], serde_json::json!([grant.id]));
    assert_eq!(
        app.clone().oneshot(make()).await.unwrap().status(),
        StatusCode::UNAUTHORIZED
    );
    let client = keys.paired_clients().pop().unwrap();
    keys.revoke_paired_client(&client.id).unwrap();
    context.nonce = hex(&[2; 16]);
    context
        .sign(&keys.secure.as_ref().unwrap().identity)
        .unwrap();
    let request = Request::get("/api/pairing/secure/check")
        .header(
            "x-mold-secure-control",
            URL_SAFE_NO_PAD.encode(serde_json::to_vec(&context).unwrap()),
        )
        .body(Body::empty())
        .unwrap();
    let response = app.oneshot(request).await.unwrap();
    let json: serde_json::Value = serde_json::from_slice(
        &axum::body::to_bytes(response.into_body(), 65536)
            .await
            .unwrap(),
    )
    .unwrap();
    assert_eq!(json["active"], serde_json::json!([]));
}
#[tokio::test]
async fn only_live_matching_device_can_renew_a_certificate() {
    let (keys, device, credential, grant) = fixture();
    let app = app(keys.clone(), "127.0.0.1:1000");
    let proof = context(&keys, &device, &grant, "POST", "/api/pairing/secure/renew");
    let request = Request::post("/api/pairing/secure/renew")
        .header("x-api-key", credential)
        .header(
            "x-mold-secure-peer",
            URL_SAFE_NO_PAD.encode(serde_json::to_vec(&proof).unwrap()),
        )
        .body(Body::empty())
        .unwrap();
    let response = app.clone().oneshot(request).await.unwrap();
    assert_eq!(response.status(), StatusCode::OK);
    let json: serde_json::Value = serde_json::from_slice(
        &axum::body::to_bytes(response.into_body(), 65536)
            .await
            .unwrap(),
    )
    .unwrap();
    assert_eq!(json["grant"]["id"], grant.id);
    assert_eq!(
        app.oneshot(
            Request::post("/api/pairing/secure/renew")
                .header("x-api-key", "operator")
                .body(Body::empty())
                .unwrap()
        )
        .await
        .unwrap()
        .status(),
        StatusCode::UNAUTHORIZED
    );
}

#[test]
fn serialized_claim_rechecks_capacity_without_consuming_a_waiting_invitation() {
    let (keys, device, _credential, grant) = fixture();
    let secure = keys.secure.as_ref().unwrap();
    let issued = secure.issue_invite().unwrap();
    let existing = context(&keys, &device, &grant, "GET", "/api/status");
    let template = secure.device_for_context(&existing).unwrap();
    for index in 0..4095 {
        secure.insert_device(mold_db::secure_devices::SecureDevice {
            client_id: format!("capacity-{index}"),
            grant_id: format!("{index:032x}"),
            ..template.clone()
        });
    }
    let joining = Identity::generate().unwrap();
    let mut secret = [0; 64];
    secret[..32].copy_from_slice(&mold_secure::decode_hex::<32>(&issued.invite_seed).unwrap());
    let invitation = Identity::from_bytes(&secret).unwrap();
    let claim = RequestContext {
        binding: joining.public_binding(),
        grant: None,
        invite: Some(issued.invite.clone()),
        handshake_hash: hex(&[6; 32]),
        stream_id: 1,
        issued_at: unix_time(),
        request_nonce: hex(&[8; 16]),
        signature: String::new(),
    };
    let proof = hex(&invitation.sign(
        &mold_secure::claim_bytes(
            &secure.namespace(),
            &issued.invite.id,
            &[6; 32],
            &joining.signing_public(),
            &joining.noise_public(),
        )
        .unwrap(),
    ));
    assert!(keys
        .claim_secure_pairing(&claim, &proof, "pending", "cli")
        .is_err());
    assert_eq!(secure.active_ids().len(), 4096);
    secure.revoke("capacity-0");
    assert!(keys
        .claim_secure_pairing(&claim, &proof, "pending", "cli")
        .is_ok());
    assert_eq!(secure.active_ids().len(), 4096);
}

#[tokio::test]
async fn secure_paired_device_cannot_mint_other_devices() {
    let (keys, device, credential, grant) = fixture();
    let proof = context(&keys, &device, &grant, "POST", "/api/pairing/sessions");
    let response = app(keys, "127.0.0.1:1000")
        .oneshot(
            Request::post("/api/pairing/sessions")
                .header("x-api-key", credential)
                .header(
                    "x-mold-secure-peer",
                    URL_SAFE_NO_PAD.encode(serde_json::to_vec(&proof).unwrap()),
                )
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::FORBIDDEN);
}

#[test]
fn secure_credentials_survive_restart_only_on_their_host_until_revoked() {
    let db = Arc::new(Some(mold_db::MetadataDb::open_in_memory().unwrap()));
    let (keys, _, credential, _) = fixture_with_db(db.clone());
    let restart = |instance: &str| {
        ApiKeySet::new_with_metadata_db(HashSet::from(["operator".into()]), db.clone(), instance)
    };
    assert!(restart("host").contains(&credential));
    assert!(!restart("other-host").contains(&credential));
    let id = keys.paired_clients()[0].id.clone();
    keys.revoke_paired_client(&id).unwrap();
    assert!(!restart("host").contains(&credential));
}
