//! Development-only protocol fixture. Never uses production grants or models.
use axum::{
    body::{to_bytes, Body},
    extract::{Request, State},
    http::{Response, StatusCode},
    routing::{any, get},
    Router,
};
use base64::{engine::general_purpose::URL_SAFE_NO_PAD, Engine};
use mold_relay::secure::{lan_router, unix_now, HostConfig, HostPolicy};
use mold_secure::*;
use std::sync::Arc;
#[derive(Clone)]
struct App {
    host: Arc<Identity>,
    policy: tokio::sync::watch::Sender<HostPolicy>,
}
fn json(value: serde_json::Value) -> Response<Body> {
    Response::builder()
        .header("content-type", "application/json")
        .body(Body::from(value.to_string()))
        .unwrap()
}
fn refuse(status: StatusCode) -> Response<Body> {
    Response::builder()
        .status(status)
        .body(Body::from("fixture authorization refused"))
        .unwrap()
}
async fn api(State(app): State<App>, req: Request) -> Response<Body> {
    if req.uri().path() == "/api/pairing/secure/check" {
        let control = req
            .headers()
            .get("x-mold-secure-control")
            .and_then(|v| v.to_str().ok())
            .and_then(|v| URL_SAFE_NO_PAD.decode(v).ok())
            .and_then(|v| serde_json::from_slice::<ControlContext>(&v).ok());
        if control.is_none_or(|c| c.verify(&app.host.signing_public(), unix_now()).is_err()) {
            return refuse(StatusCode::FORBIDDEN);
        }
        let policy = app.policy.borrow();
        return json(serde_json::json!({"epoch":policy.epoch,"active":policy.active}));
    }
    let claim = req.uri().path() == "/api/pairing/claim";
    if !claim
        && req.headers().get("x-api-key").and_then(|v| v.to_str().ok())
            != Some("mold_pair_fixture_api_key")
    {
        return refuse(StatusCode::UNAUTHORIZED);
    }
    let context = req
        .headers()
        .get("x-mold-secure-peer")
        .and_then(|v| v.to_str().ok())
        .and_then(|v| URL_SAFE_NO_PAD.decode(v).ok())
        .and_then(|v| serde_json::from_slice::<RequestContext>(&v).ok());
    let Some(context) = context.filter(|c| {
        c.verify(
            &app.host.signing_public(),
            req.method().as_str(),
            &req.uri().to_string(),
            unix_now(),
        )
        .is_ok()
    }) else {
        return refuse(StatusCode::FORBIDDEN);
    };
    if claim {
        let Some(invite) = &context.invite else {
            return refuse(StatusCode::FORBIDDEN);
        };
        if invite
            .verify(&app.host.signing_public(), unix_now(), 0)
            .is_err()
        {
            return refuse(StatusCode::FORBIDDEN);
        }
        let body = to_bytes(req.into_body(), 16384).await.unwrap();
        let body: serde_json::Value = serde_json::from_slice(&body).unwrap();
        let bytes = claim_bytes(
            &namespace(&app.host.signing_public()),
            &invite.id,
            &decode_hex(&context.handshake_hash).unwrap(),
            &decode_hex(&context.binding.signing_key).unwrap(),
            &decode_hex(&context.binding.noise_key).unwrap(),
        )
        .unwrap();
        if !body["proof"]
            .as_str()
            .and_then(|v| decode_hex::<64>(v).ok())
            .is_some_and(|proof| {
                verify_signature(
                    &decode_hex::<32>(&invite.signing_key).unwrap(),
                    &bytes,
                    &proof,
                )
            })
        {
            return refuse(StatusCode::FORBIDDEN);
        }
        let id = uuid::Uuid::new_v4().simple().to_string();
        let grant = grant(&app.host, &context.binding, id.clone());
        app.policy.send_modify(|p| p.active.push(id));
        return json(
            serde_json::json!({"protocol":3,"api_key":"mold_pair_fixture_api_key","grant":grant,"hostname":"Secure Fixture","instance_id":"00000000-0000-0000-0000-000000000001","endpoints":[],"host_key":hex::encode(app.host.signing_public()),"host_noise_key":hex::encode(app.host.noise_public()),"host_binding_signature":app.host.public_binding().signature,"namespace":namespace(&app.host.signing_public())}),
        );
    }
    if req.uri().path() == "/api/pairing/secure/renew" {
        let Some(old) = context.grant else {
            return refuse(StatusCode::FORBIDDEN);
        };
        return json(serde_json::json!({"grant":grant(&app.host,&context.binding,old.id)}));
    }
    if req.uri().path() == "/api/gallery/image/fixture.mp4" {
        return media(req).await;
    }
    if req.uri().path() == "/bulk" {
        let count = req
            .uri()
            .query()
            .and_then(|q| q.strip_prefix("bytes="))
            .and_then(|n| n.parse::<usize>().ok())
            .unwrap_or(209715200)
            .min(268435456);
        let stream = futures_util::stream::unfold(count, |left| async move {
            if left == 0 {
                None
            } else {
                let count = left.min(65536);
                Some((Ok::<_, std::io::Error>(vec![42; count]), left - count))
            }
        });
        return Response::builder()
            .header("content-type", "application/octet-stream")
            .header("content-length", count)
            .body(Body::from_stream(stream))
            .unwrap();
    }
    match req.uri().path() {
        "/echo" => {
            let body = to_bytes(req.into_body(), MAX_UPLOAD as usize)
                .await
                .unwrap();
            Response::builder()
                .header("content-type", "application/octet-stream")
                .body(Body::from(body))
                .unwrap()
        }
        "/sse" => {
            let stream = futures_util::stream::unfold(0, |n| async move {
                if n == 4 {
                    return None;
                }
                tokio::time::sleep(std::time::Duration::from_millis(50)).await;
                Some((
                    Ok::<_, std::io::Error>(format!("data: {{\"tick\":{n}}}\n\n")),
                    n + 1,
                ))
            });
            Response::builder()
                .header("content-type", "text/event-stream")
                .body(Body::from_stream(stream))
                .unwrap()
        }
        "/range" => Response::builder()
            .status(206)
            .header("content-range", "bytes 0-196607/196608")
            .header("content-type", "application/octet-stream")
            .body(Body::from(vec![42; 196608]))
            .unwrap(),
        _ => json(
            serde_json::json!({"status":"ok","fixture":true,"version":"fixture-v3","models_loaded":[],"gpu_info":null,"uptime_secs":1,"instance_id":"00000000-0000-0000-0000-000000000001","hostname":"Secure Fixture"}),
        ),
    }
}
async fn media(req: Request) -> Response<Body> {
    let path = std::env::var_os("MOLD_SECURE_FIXTURE_MEDIA")
        .map(std::path::PathBuf::from)
        .unwrap_or_else(|| "/tmp/mold-secure-uat/fixture.mp4".into());
    let Ok(bytes) = tokio::fs::read(path).await else {
        return refuse(StatusCode::NOT_FOUND);
    };
    let total = bytes.len();
    if total == 0 {
        return refuse(StatusCode::NOT_FOUND);
    }
    let mut start = 0usize;
    let mut end = total - 1;
    let mut status = StatusCode::OK;
    if let Some(range) = req.headers().get("range") {
        let parsed = range
            .to_str()
            .ok()
            .and_then(|r| r.strip_prefix("bytes="))
            .and_then(|r| r.split_once('-'))
            .and_then(|(a, b)| {
                if a.is_empty() {
                    let n = b.parse::<usize>().ok()?;
                    if n == 0 {
                        return None;
                    }
                    Some((total.saturating_sub(n), total - 1))
                } else {
                    Some((
                        a.parse().ok()?,
                        if b.is_empty() {
                            total - 1
                        } else {
                            b.parse::<usize>().ok()?.min(total - 1)
                        },
                    ))
                }
            });
        let Some((a, b)) = parsed.filter(|(a, b)| a <= b && *a < total) else {
            return Response::builder()
                .status(416)
                .header("content-range", format!("bytes */{total}"))
                .body(Body::empty())
                .unwrap();
        };
        start = a;
        end = b;
        status = StatusCode::PARTIAL_CONTENT;
    }
    let mut response = Response::builder()
        .status(status)
        .header("content-type", "video/mp4")
        .header("accept-ranges", "bytes")
        .header("content-length", end - start + 1);
    if status == StatusCode::PARTIAL_CONTENT {
        response = response.header("content-range", format!("bytes {start}-{end}/{total}"));
    }
    response
        .body(if req.method() == "HEAD" {
            Body::empty()
        } else {
            Body::from(bytes[start..=end].to_vec())
        })
        .unwrap()
}
fn grant(host: &Identity, device: &Binding, id: String) -> DeviceGrant {
    let mut grant = DeviceGrant {
        id,
        expires_at: unix_now() + 30 * 86400,
        epoch: 0,
        signature: String::new(),
    };
    grant.signature = hex::encode(
        host.sign(
            &grant_bytes(
                &namespace(&host.signing_public()),
                &decode_hex(&device.signing_key).unwrap(),
                &decode_hex(&device.noise_key).unwrap(),
                &grant.id,
                grant.expires_at,
                0,
            )
            .unwrap(),
        ),
    );
    grant
}
#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let host = Arc::new(Identity::from_bytes(&[1; 64])?);
    let device = Identity::from_bytes(&[2; 64])?;
    let grant = grant(&host, &device.public_binding(), "11".repeat(16));
    let (policy, rx) = tokio::sync::watch::channel(HostPolicy {
        epoch: 0,
        active: vec![grant.id.clone()],
    });
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await?;
    let target = listener.local_addr()?;
    let app = Router::new()
        .route("/api/status", get(api))
        .fallback(any(api))
        .with_state(App {
            host: host.clone(),
            policy,
        });
    tokio::spawn(async move {
        axum::serve(listener, app).await.unwrap();
    });
    let ns = namespace(&host.signing_public());
    let invitation_key = Identity::from_bytes(&[3; 64])?;
    let mut invite = Invitation {
        id: uuid::Uuid::new_v4().simple().to_string(),
        signing_key: hex::encode(invitation_key.signing_public()),
        expires_at: unix_now() + 600,
        epoch: 0,
        signature: String::new(),
    };
    invite.signature = hex::encode(host.sign(&invite.canonical(&ns)?));
    let listener = tokio::net::TcpListener::bind(
        std::env::var("MOLD_SECURE_FIXTURE_BIND").unwrap_or_else(|_| "127.0.0.1:0".into()),
    )
    .await?;
    let address = listener.local_addr()?;
    let fixture = serde_json::json!({"protocol":3,"version":2,"host_key":hex::encode(host.signing_public()),"host_noise_key":hex::encode(host.noise_public()),"host_binding_signature":host.public_binding().signature,"namespace":ns,"websocket":std::env::var("MOLD_SECURE_FIXTURE_RELAY").unwrap_or_else(|_|format!("ws://{address}/_mold/secure")),"origin":format!("http://{address}"),"device_identity":hex::encode(device.to_bytes()),"api_key":"mold_pair_fixture_api_key","grant":grant});
    let invitation = serde_json::json!({"type":"mold.mobile-pairing","name":"Secure Fixture","instance_id":"00000000-0000-0000-0000-000000000001","version":2,"base_url":format!("http://{address}"),"websocket_url":std::env::var("MOLD_SECURE_FIXTURE_RELAY").ok(),"endpoints":[{"url":format!("http://{address}"),"kind":"lan"}],"host_key":hex::encode(host.signing_public()),"host_noise_key":hex::encode(host.noise_public()),"host_binding_signature":host.public_binding().signature,"namespace":ns,"invite_seed":"03".repeat(32),"invite":invite});
    let path = std::env::args_os()
        .nth(1)
        .map(std::path::PathBuf::from)
        .unwrap_or_else(|| std::env::temp_dir().join("mold-secure-fixture/config.json"));
    mold_secure::identity_file::write_private_file(
        &path,
        serde_json::to_string_pretty(&fixture)?.as_bytes(),
        true,
    )?;
    let invite_path = path.with_file_name("invite.json");
    mold_secure::identity_file::write_private_file(
        &invite_path,
        serde_json::to_string_pretty(&invitation)?.as_bytes(),
        true,
    )?;
    println!(
        "Secure fixture listening on {address}; config: {}; invitation: {}",
        path.display(),
        invite_path.display()
    );
    let stop = tokio_util::sync::CancellationToken::new();
    if let Ok(endpoint) = std::env::var("MOLD_SECURE_FIXTURE_RELAY") {
        let cloud_config = HostConfig {
            identity: host.clone(),
            target,
            policy: rx.clone(),
        };
        let cloud_stop = stop.clone();
        let (status, mut ready) = tokio::sync::watch::channel(false);
        tokio::spawn(async move {
            if let Err(error) = mold_relay::secure::connect_with_status(
                &endpoint,
                cloud_config,
                cloud_stop,
                false,
                status,
            )
            .await
            {
                eprintln!("Fixture cloud connector stopped: {error}");
            }
        });
        tokio::spawn(async move {
            while ready.changed().await.is_ok() {
                println!("Fixture cloud admitted: {}", *ready.borrow());
            }
        });
    }

    let signal = stop.clone();
    tokio::spawn(async move {
        let _ = tokio::signal::ctrl_c().await;
        signal.cancel();
    });
    axum::serve(
        listener,
        lan_router(
            HostConfig {
                identity: host,
                target,
                policy: rx,
            },
            stop.clone(),
        ),
    )
    .with_graceful_shutdown(stop.cancelled_owned())
    .await?;
    Ok(())
}
