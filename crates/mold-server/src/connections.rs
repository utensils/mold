//! Trusted connection advertisements and credential-free instance proofs.
//! A proof detects accidental address reuse; plaintext HTTP still trusts the network.
use axum::{
    extract::{Request, State},
    http::{header, StatusCode},
    response::{IntoResponse, Response},
    Extension, Json,
};
use serde::{Deserialize, Serialize};
use std::net::{IpAddr, SocketAddr};
use std::sync::RwLock;

#[derive(Debug, Clone, Serialize, utoipa::ToSchema, PartialEq, Eq)]
pub(crate) struct ConnectionEndpoint {
    pub url: String,
    pub kind: String,
}
#[derive(Default)]
pub struct ConnectionAddresses {
    configuration: RwLock<Option<(SocketAddr, Option<String>)>>,
}
impl ConnectionAddresses {
    pub(crate) fn configure(
        &self,
        bound: SocketAddr,
        public_url: Option<&str>,
    ) -> anyhow::Result<()> {
        let public_url = public_url.map(validate_public_url).transpose()?;
        *self
            .configuration
            .write()
            .unwrap_or_else(|p| p.into_inner()) = Some((bound, public_url));
        Ok(())
    }
    pub(crate) fn endpoints(&self) -> Vec<ConnectionEndpoint> {
        let configured = self
            .configuration
            .read()
            .unwrap_or_else(|p| p.into_inner())
            .clone();
        let Some((bound, public)) = configured else {
            return Vec::new();
        };
        let interfaces = if_addrs::get_if_addrs()
            .unwrap_or_default()
            .into_iter()
            .map(|i| i.ip())
            .collect::<Vec<_>>();
        advertised_endpoints(bound, &interfaces, public.as_deref())
    }
}
pub(crate) fn hex(bytes: &[u8]) -> String {
    bytes.iter().map(|byte| format!("{byte:02x}")).collect()
}
pub(crate) fn lower_hex(value: &str, length: usize) -> bool {
    value.len() == length
        && value
            .bytes()
            .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b))
}
fn validate_public_url(value: &str) -> anyhow::Result<String> {
    let url = url::Url::parse(value)
        .map_err(|_| anyhow::anyhow!("MOLD_PUBLIC_URL must be an HTTPS origin"))?;
    if value.len() > 2048
        || url.scheme() != "https"
        || url.host_str().is_none()
        || !url.username().is_empty()
        || url.password().is_some()
        || url.path() != "/"
        || url.query().is_some()
        || url.fragment().is_some()
        || url.port() == Some(0)
        || value
            .split('/')
            .nth(2)
            .is_some_and(|authority| authority.contains('@'))
    {
        anyhow::bail!(
            "MOLD_PUBLIC_URL must be an HTTPS origin without credentials, path, query or fragment"
        );
    }
    Ok(url.origin().ascii_serialization())
}
fn advertised_endpoints(
    bound: SocketAddr,
    interfaces: &[IpAddr],
    public: Option<&str>,
) -> Vec<ConnectionEndpoint> {
    let mut endpoints = Vec::new();
    if let Some(url) = public {
        endpoints.push(ConnectionEndpoint {
            url: url.into(),
            kind: "relay".into(),
        });
    }
    let scoped = matches!(bound, SocketAddr::V6(address) if address.scope_id() != 0);
    let mut local = interfaces
        .iter()
        .copied()
        .filter(|ip| {
            !scoped
                && bound.port() != 0
                && bound.is_ipv4() == ip.is_ipv4()
                && (bound.ip().is_unspecified() || bound.ip() == *ip)
                && usable(*ip)
        })
        .map(|ip| ConnectionEndpoint {
            url: format!("http://{}", SocketAddr::new(ip, bound.port())),
            kind: if tailscale(ip) { "tailscale" } else { "lan" }.into(),
        })
        .collect::<Vec<_>>();
    local.sort_by_key(|e| (e.kind != "tailscale", e.url.clone()));
    local.dedup_by(|a, b| a.url == b.url);
    endpoints.extend(local);
    endpoints.truncate(8);
    endpoints
}
fn tailscale(ip: IpAddr) -> bool {
    match ip {
        IpAddr::V4(ip) => ip.octets()[0] == 100 && (64..128).contains(&ip.octets()[1]),
        IpAddr::V6(ip) => ip.segments()[..3] == [0xfd7a, 0x115c, 0xa1e0],
    }
}
fn usable(ip: IpAddr) -> bool {
    if ip.is_loopback() || ip.is_unspecified() || ip.is_multicast() {
        return false;
    }
    match ip {
        IpAddr::V4(ip) => !ip.is_link_local() && !ip.is_broadcast() && ip.octets()[0] != 0,
        IpAddr::V6(ip) => {
            !ip.is_unicast_link_local()
                && ip.to_ipv4_mapped().is_none_or(|v4| usable(IpAddr::V4(v4)))
        }
    }
}

#[derive(Debug, Serialize, utoipa::ToSchema)]
pub(crate) struct ConnectionAddressResponse {
    version: u8,
    instance_id: String,
    endpoints: Vec<ConnectionEndpoint>,
}
#[derive(Debug, Deserialize, utoipa::ToSchema)]
#[serde(deny_unknown_fields)]
pub(crate) struct ConnectionProbeRequest {
    kind: String,
    key_tag: String,
    nonce: String,
}
#[derive(Debug, Serialize, utoipa::ToSchema)]
pub(crate) struct ConnectionProbeResponse {
    instance_id: String,
    proof: String,
}
#[utoipa::path(get, path = "/api/connection-addresses", tag = "server", responses((status = 200, body = ConnectionAddressResponse), (status = 401, description = "Authentication required")))]
pub(crate) async fn addresses(
    State(state): State<crate::state::AppState>,
    authority: Option<Extension<crate::auth::PairingAuthority>>,
) -> impl IntoResponse {
    (
        [(header::CACHE_CONTROL, "no-store")],
        Json(ConnectionAddressResponse {
            version: 1,
            instance_id: state.instance_id.to_string(),
            endpoints: if authority.is_some() {
                Vec::new()
            } else {
                state.connection_addresses.endpoints()
            },
        }),
    )
}
#[utoipa::path(post, path = "/api/connection-probe", tag = "server", request_body = ConnectionProbeRequest, responses((status = 200, body = ConnectionProbeResponse), (status = 401, description = "Connection proof unavailable")))]
pub(crate) async fn probe(
    State(state): State<crate::state::AppState>,
    auth: Option<Extension<crate::auth::AuthState>>,
    request: Request,
) -> Response {
    let invalid = || {
        (
            StatusCode::UNAUTHORIZED,
            [(header::CACHE_CONTROL, "no-store")],
            Json(serde_json::json!({"error":"connection proof unavailable"})),
        )
            .into_response()
    };
    if request
        .headers()
        .get(header::CONTENT_TYPE)
        .and_then(|v| v.to_str().ok())
        .is_none_or(|v| {
            v.split(';')
                .next()
                .is_none_or(|v| !v.trim().eq_ignore_ascii_case("application/json"))
        })
    {
        return invalid();
    }
    let Ok(body) = axum::body::to_bytes(request.into_body(), 512).await else {
        return invalid();
    };
    let Ok(request) = serde_json::from_slice::<ConnectionProbeRequest>(&body) else {
        return invalid();
    };
    let Some(Extension(Some(keys))) = auth else {
        return invalid();
    };
    let Some(proof) = keys.connection_proof(
        &request.kind,
        &request.key_tag,
        &request.nonce,
        &state.instance_id,
    ) else {
        return invalid();
    };
    (
        [(header::CACHE_CONTROL, "no-store")],
        Json(ConnectionProbeResponse {
            instance_id: state.instance_id.to_string(),
            proof,
        }),
    )
        .into_response()
}

#[cfg(test)]
mod tests {
    use super::*;
    use sha2::Digest;
    #[tokio::test]
    async fn connection_routes_enforce_auth_and_validate_anonymous_proofs() {
        use axum::{
            body::{to_bytes, Body},
            http::{Request, StatusCode},
            middleware,
        };
        use tower::ServiceExt;
        let state = crate::state::AppState::for_tests();
        state
            .connection_addresses
            .configure(
                "127.0.0.1:7680".parse().unwrap(),
                Some("https://relay.example"),
            )
            .unwrap();
        let instance = state.instance_id.to_string();
        let keys = std::sync::Arc::new(crate::auth::ApiKeySet::new_with_metadata_db(
            std::collections::HashSet::from(["operator".into()]),
            std::sync::Arc::new(Some(mold_db::MetadataDb::open_in_memory().unwrap())),
            instance.clone(),
        ));
        let (token, _) = keys.issue_pairing_token().unwrap();
        let paired = keys
            .claim_pairing_token(&token, "fixture", "iphone")
            .unwrap()
            .unwrap();
        let app = crate::routes::create_router(state)
            .layer(middleware::from_fn(crate::auth::require_api_key))
            .layer(middleware::from_fn_with_state(
                Some(keys),
                crate::auth::inject_auth_state,
            ));
        let missing = app
            .clone()
            .oneshot(
                Request::get("/api/connection-addresses")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(missing.status(), StatusCode::UNAUTHORIZED);
        let valid = app
            .clone()
            .oneshot(
                Request::get("/api/connection-addresses")
                    .header("x-api-key", &paired)
                    .header("host", "evil.example")
                    .header("forwarded", "host=evil.example;proto=https")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(valid.status(), StatusCode::OK);
        assert_eq!(valid.headers()["cache-control"], "no-store");
        let json: serde_json::Value =
            serde_json::from_slice(&to_bytes(valid.into_body(), 4096).await.unwrap()).unwrap();
        assert_eq!(json["version"], 1);
        assert_eq!(json["instance_id"], instance);
        assert_eq!(
            json["endpoints"],
            serde_json::json!([{ "url":"https://relay.example", "kind":"relay" }])
        );
        let tag = hex(&sha2::Sha256::digest(paired.as_bytes())[..8]);
        let nonce = "01".repeat(32);
        let body = serde_json::json!({"kind":"api","key_tag":tag,"nonce":nonce}).to_string();
        let proof = app
            .clone()
            .oneshot(
                Request::post("/api/connection-probe")
                    .header("content-type", "application/json")
                    .body(Body::from(body.clone()))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(proof.status(), StatusCode::OK);
        assert_eq!(proof.headers()["cache-control"], "no-store");
        let json: serde_json::Value =
            serde_json::from_slice(&to_bytes(proof.into_body(), 4096).await.unwrap()).unwrap();
        assert_eq!(json["instance_id"], instance);
        assert_eq!(json["proof"].as_str().unwrap().len(), 64);
        for bad in [
            "{}".into(),
            "x".repeat(513),
            body.replace("api", "wrong"),
            body.replace(&nonce, &nonce.to_uppercase().replace("01", "FF")),
            body.replace(&tag, "0000000000000000"),
        ] {
            let invalid = app
                .clone()
                .oneshot(
                    Request::post("/api/connection-probe")
                        .header("content-type", "application/json")
                        .body(Body::from(bad))
                        .unwrap(),
                )
                .await
                .unwrap();
            assert_eq!(invalid.status(), StatusCode::UNAUTHORIZED);
            assert_eq!(invalid.headers()["cache-control"], "no-store");
        }
    }
    #[tokio::test]
    async fn keyless_and_operator_routes_refuse_proofs_without_breaking_original_auth() {
        use axum::{
            body::{to_bytes, Body},
            http::{Request, StatusCode},
            middleware,
        };
        use tower::ServiceExt;
        for operator in [true, false] {
            let state = crate::state::AppState::for_tests();
            state
                .connection_addresses
                .configure(
                    "127.0.0.1:7680".parse().unwrap(),
                    Some("https://relay.example"),
                )
                .unwrap();
            let auth = operator.then(|| {
                std::sync::Arc::new(crate::auth::ApiKeySet::new(
                    std::collections::HashSet::from(["password".to_string()]),
                ))
            });
            let app = crate::routes::create_router(state)
                .layer(middleware::from_fn(crate::auth::require_api_key))
                .layer(middleware::from_fn_with_state(
                    auth,
                    crate::auth::inject_auth_state,
                ));
            let response = app.clone().oneshot(Request::post("/api/connection-probe").header("content-type","application/json").body(Body::from(serde_json::json!({"kind":"api","key_tag":hex(&sha2::Sha256::digest(b"password")[..8]),"nonce":"a".repeat(64)}).to_string())).unwrap()).await.unwrap();
            assert_eq!(response.status(), StatusCode::UNAUTHORIZED);
            assert_eq!(response.headers()["cache-control"], "no-store");
            let response = app
                .oneshot(
                    Request::get("/api/connection-addresses")
                        .header("x-api-key", "password")
                        .body(Body::empty())
                        .unwrap(),
                )
                .await
                .unwrap();
            assert_eq!(response.status(), StatusCode::OK);
            let json: serde_json::Value =
                serde_json::from_slice(&to_bytes(response.into_body(), 4096).await.unwrap())
                    .unwrap();
            if operator {
                assert_eq!(json["endpoints"], serde_json::json!([]));
            }
        }
    }
    #[test]
    fn public_url_requires_https_origin_without_credentials_or_path() {
        assert_eq!(
            validate_public_url("https://mold-link.urandom.io").unwrap(),
            "https://mold-link.urandom.io"
        );
        for bad in [
            "http://host",
            "https://user:password@host",
            "https://host/path",
            "https://host/?secret=x",
            "https://host/#x",
            "https://host:0",
            "https://host/../x",
        ] {
            assert!(validate_public_url(bad).is_err(), "{bad}");
        }
    }
    #[test]
    fn candidates_match_actual_bind_and_filter_unusable_interfaces() {
        let ips: Vec<IpAddr> = [
            "127.0.0.1",
            "0.0.0.0",
            "169.254.1.1",
            "224.0.0.1",
            "192.168.1.20",
            "100.80.1.2",
            "::1",
            "fe80::1",
            "ff02::1",
            "fd7a:115c:a1e0::123",
            "fd00::1",
        ]
        .iter()
        .map(|ip| ip.parse().unwrap())
        .collect();
        let v4 = advertised_endpoints(
            "0.0.0.0:7680".parse().unwrap(),
            &ips,
            Some("https://relay.example"),
        );
        assert_eq!(
            v4.iter().map(|e| (&*e.url, &*e.kind)).collect::<Vec<_>>(),
            [
                ("https://relay.example", "relay"),
                ("http://100.80.1.2:7680", "tailscale"),
                ("http://192.168.1.20:7680", "lan")
            ]
        );
        let specific = advertised_endpoints("192.168.1.20:7680".parse().unwrap(), &ips, None);
        assert_eq!(specific.len(), 1);
        assert_eq!(specific[0].url, "http://192.168.1.20:7680");
        assert!(advertised_endpoints("127.0.0.1:7680".parse().unwrap(), &ips, None).is_empty());
        assert!(advertised_endpoints("10.1.1.1:7680".parse().unwrap(), &ips, None).is_empty());
        let v6 = advertised_endpoints("[::]:7680".parse().unwrap(), &ips, None);
        assert_eq!(v6.len(), 2);
        assert!(v6
            .iter()
            .any(|e| e.url == "http://[fd7a:115c:a1e0::123]:7680" && e.kind == "tailscale"));
        let lots: Vec<_> = (1..30).map(|n| IpAddr::from([10, 0, 0, n])).collect();
        assert_eq!(
            advertised_endpoints(
                "0.0.0.0:7680".parse().unwrap(),
                &lots,
                Some("https://relay.example")
            )
            .len(),
            8
        );
    }
}
