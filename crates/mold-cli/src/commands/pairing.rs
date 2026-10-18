//! Pairing invitations: create one on this host (`mold host invite`), or ask
//! a host for one with its operator key (`mold host add URL`). Private
//! invitation material never enters arguments or logs; on this host it goes
//! only to stdout as the one pairing link, or to an owner-only file.
use anyhow::{bail, Context, Result};
use std::{net::SocketAddr, path::Path};
use zeroize::Zeroizing;

/// What `POST /api/pairing/sessions` answered.
pub(crate) enum SessionReply {
    /// A one-use invitation (the raw session response).
    Created(serde_json::Value),
    /// A refusal, with the API error `code` when the body carried one.
    Refused { status: u16, code: String },
}

/// Ask `base_url` for a pairing session with `key` (sent as `X-API-Key`
/// when non-empty). `regular_direct` asks for a Regular invitation bound to
/// that explicit direct origin. A transport failure is an error.
pub(crate) async fn request_session(
    base_url: &str,
    key: &str,
    regular_direct: Option<&str>,
) -> Result<SessionReply> {
    let client = reqwest::Client::builder()
        .no_proxy()
        .redirect(reqwest::redirect::Policy::none())
        .timeout(std::time::Duration::from_secs(10))
        .build()?;
    let mut request = client.post(format!(
        "{}/api/pairing/sessions",
        base_url.trim_end_matches('/')
    ));
    if !key.is_empty() {
        request = request.header("x-api-key", key);
    }
    let request = match regular_direct {
        Some(origin) => request.json(&serde_json::json!({"mode":"regular","direct_url":origin})),
        None => request.body(""),
    };
    let mut response = request
        .send()
        .await
        .with_context(|| format!("cannot contact the Mold host at {base_url}"))?;
    let status = response.status();
    let mut bytes = Zeroizing::new(Vec::new());
    while let Some(chunk) = response.chunk().await? {
        if bytes.len() + chunk.len() > 32768 {
            bail!("invalid pairing response size")
        }
        bytes.extend_from_slice(&chunk);
    }
    if status.is_success() {
        return Ok(SessionReply::Created(serde_json::from_slice(&bytes)?));
    }
    let code = serde_json::from_slice::<serde_json::Value>(&bytes)
        .ok()
        .and_then(|body| body["code"].as_str().map(str::to_owned))
        .unwrap_or_default();
    Ok(SessionReply::Refused {
        status: status.as_u16(),
        code,
    })
}

/// The first usable key in an owner-only key file: the first line that is
/// neither blank nor a `#` comment.
pub(crate) fn read_key_file(path: &Path) -> Result<Zeroizing<String>> {
    let bytes = mold_secure_client::read_private(path)?;
    Ok(Zeroizing::new(
        std::str::from_utf8(&bytes)?
            .lines()
            .map(str::trim)
            .find(|v| !v.is_empty() && !v.starts_with('#'))
            .context("key file is empty")?
            .to_string(),
    ))
}

/// This host's operator key: `--api-key-file`, else `MOLD_API_KEY` (a key,
/// a comma list whose first entry is used, or `@owner-only-key-file`), else
/// the per-home operator credential.
pub(crate) fn operator_key(api_key_file: Option<&Path>) -> Result<Zeroizing<String>> {
    if let Some(path) = api_key_file {
        return read_key_file(path);
    }
    let raw = Zeroizing::new(match std::env::var("MOLD_API_KEY") {
        Ok(value) => value,
        Err(_) => mold_secure_client::operator_key::read(
            &mold_core::Config::mold_dir().context("Mold home is unavailable")?,
        )?
        .context("Set MOLD_API_KEY to the host operator key or @owner-only-key-file, or pass --api-key-file")?
        .to_string(),
    });
    if let Some(path) = raw.strip_prefix('@') {
        return read_key_file(Path::new(path));
    }
    Ok(Zeroizing::new(
        raw.split(',')
            .map(str::trim)
            .find(|v| !v.is_empty())
            .context("operator key is empty")?
            .to_string(),
    ))
}

/// A canonical invitation from this authenticated local host, as the value
/// `mold host add` / [`mold_secure_client::pairing_link`] take.
pub(crate) async fn create_invitation(
    target: SocketAddr,
    relay_url: Option<&str>,
    mode: mold_secure_client::ConnectionMode,
    direct_url: Option<&str>,
    api_key_file: Option<&Path>,
) -> Result<serde_json::Value> {
    anyhow::ensure!(
        mode != mold_secure_client::ConnectionMode::Regular || relay_url.is_none(),
        "Regular pairing cannot use a proxy endpoint"
    );
    let direct_url = match mode {
        mold_secure_client::ConnectionMode::Regular => Some(mold_secure_client::regular_origin(
            direct_url.context("Regular pairing requires --direct-url")?,
        )?),
        mold_secure_client::ConnectionMode::Secure => {
            anyhow::ensure!(direct_url.is_none(), "--direct-url requires --regular");
            None
        }
    };
    mold_relay::validate_target(target)?;
    if let Some(url) = relay_url {
        validate_relay_url(url)?;
    }
    let key = operator_key(api_key_file)?;
    let value = match request_session(&format!("http://{target}"), &key, direct_url.as_deref())
        .await
        .context("cannot contact the local Mold host")?
    {
        SessionReply::Created(value) => value,
        SessionReply::Refused { status, code } => bail!(
            "host refused pairing ({status}{}); use its operator credential",
            if code.is_empty() {
                String::new()
            } else {
                format!(" {code}")
            }
        ),
    };
    if mode == mold_secure_client::ConnectionMode::Regular {
        regular_invitation_payload(value)
    } else {
        invitation_payload(value, relay_url)
    }
}

/// Write an invitation to a new owner-only file (the `--output` form).
pub(crate) fn write_invitation_file(output: &Path, value: &serde_json::Value) -> Result<()> {
    mold_secure_client::write_invitation(output, value)
}

pub(crate) fn invitation_payload(
    mut value: serde_json::Value,
    relay_url: Option<&str>,
) -> Result<serde_json::Value> {
    if let Some(url) = relay_url {
        validate_relay_url(url)?;
        value["websocket_url"] = url.into();
    }
    if value["protocol"] != 3 {
        bail!("upgrade the host before pairing")
    }
    let endpoints = value["endpoints"]
        .as_array()
        .context("host did not advertise secure endpoints")?;
    let origin = endpoints
        .iter()
        .find(|e| e["kind"] == "relay")
        .or_else(|| endpoints.first())
        .and_then(|e| e["url"].as_str())
        .context("enable Remote Access or configure MOLD_PUBLIC_URL before pairing")?
        .to_string();
    let url = reqwest::Url::parse(&origin)?;
    if !matches!(url.scheme(), "http" | "https") || url.origin().ascii_serialization() != origin {
        bail!("invalid host origin")
    }
    value
        .as_object_mut()
        .context("invalid pairing response")?
        .remove("protocol");
    value["version"] = 2.into();
    value["type"] = "mold.mobile-pairing".into();
    value["base_url"] = origin.into();
    value["name"] = value["hostname"].clone();
    Ok(value)
}
fn validate_relay_url(value: &str) -> Result<()> {
    let url = reqwest::Url::parse(value)?;
    if url.scheme() != "wss"
        || !url.username().is_empty()
        || url.password().is_some()
        || url.query().is_some()
        || url.fragment().is_some()
    {
        bail!("relay endpoint must be credential-free WSS without a query or fragment");
    }
    Ok(())
}

pub(crate) fn regular_invitation_payload(
    mut value: serde_json::Value,
) -> Result<serde_json::Value> {
    anyhow::ensure!(
        value["mode"] == "regular",
        "invalid Regular invitation response"
    );
    let origin = value["base_url"]
        .as_str()
        .context("missing Regular origin")?;
    anyhow::ensure!(
        mold_secure_client::regular_origin(origin)? == origin,
        "invalid Regular origin"
    );
    value["version"] = 2.into();
    value["type"] = "mold.mobile-pairing".into();
    value["name"] = value["hostname"].as_str().unwrap_or("Mold host").into();
    let invitation = value
        .as_object_mut()
        .context("invalid Regular invitation")?;
    invitation.remove("hostname");
    invitation.remove("auth_required");
    Ok(value)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn regular_owner_export_round_trips_through_the_real_client_parser() {
        let response = serde_json::json!({
            "mode": "regular",
            "auth_required": true,
            "base_url": "http://192.168.1.5:7680",
            "instance_id": "host-instance",
            "hostname": "Host",
            "pairing_token": "a".repeat(64),
            "expires_at": 4_000_000_000_u64
        });
        let output = regular_invitation_payload(response.clone()).unwrap();
        let serialized = serde_json::to_vec(&output).unwrap();
        let _: mold_secure_client::RegularInvite = serde_json::from_slice(&serialized).unwrap();
        assert_eq!(output["base_url"], response["base_url"]);
        assert_eq!(output["pairing_token"], response["pairing_token"]);
        assert_eq!(output["mode"], response["mode"]);
        assert_eq!(output["instance_id"], "host-instance");
        assert_eq!(output["name"], "Host");
        assert!(output.get("auth_required").is_none());
    }
    /// The exported Secure invitation is the shared pairing link's payload:
    /// it encodes and parses back unchanged.
    #[test]
    fn a_secure_export_becomes_the_shared_pairing_link() {
        let fixture: serde_json::Value =
            serde_json::from_str(include_str!("../../../../studio/api/pairing.fixtures.json"))
                .unwrap();
        let mut session = fixture["links"][3]["payload"].clone();
        session["protocol"] = 3.into();
        session["hostname"] = "Roaming Studio".into();
        session["auth_required"] = true.into();
        let invitation = invitation_payload(session, None).unwrap();
        let link = mold_secure_client::pairing_link(&invitation).unwrap();
        assert!(link.starts_with("https://utensils.io/mold/pair#version=2&"));
        let parsed = mold_secure_client::parse_pairing_link(&link).unwrap();
        assert_eq!(parsed["base_url"], "https://relay.example");
        assert_eq!(parsed["name"], "Roaming Studio");
        let _: mold_secure_client::PairingInvite = serde_json::from_value(parsed).unwrap();
    }

    #[test]
    fn standalone_invitation_binds_an_explicit_credential_free_relay_endpoint() {
        let value = serde_json::json!({"protocol":3,"hostname":"host","endpoints":[{"kind":"relay","url":"https://host.example"}]});
        let output =
            invitation_payload(value.clone(), Some("wss://relay.example/production")).unwrap();
        assert_eq!(output["websocket_url"], "wss://relay.example/production");
        for bad in [
            "ws://relay.example",
            "wss://secret@relay.example",
            "wss://relay.example?token=secret",
        ] {
            assert!(invitation_payload(value.clone(), Some(bad)).is_err());
        }
    }
    #[test]
    fn owner_export_uses_shared_qr_contract_and_never_private_engine_origin() {
        let value = serde_json::json!({"protocol":3,"hostname":"host","instance_id":"inst-1","endpoints":[{"kind":"lan","url":"http://192.168.1.5:4321"},{"kind":"relay","url":"https://host.example"}],"websocket_url":"wss://relay.example"});
        let output = invitation_payload(value, None).unwrap();
        // The client names the pairing and recognises a re-pair from these.
        assert_eq!(output["hostname"], "host");
        assert_eq!(output["instance_id"], "inst-1");
        assert_eq!(output["version"], 2);
        assert!(output.get("protocol").is_none());
        assert_eq!(output["type"], "mold.mobile-pairing");
        assert_eq!(output["base_url"], "https://host.example");
        assert_eq!(output["name"], "host");
        assert!(
            invitation_payload(serde_json::json!({"protocol":3,"endpoints":[]}), None).is_err()
        );
    }
}
