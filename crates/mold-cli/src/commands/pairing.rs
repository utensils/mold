//! Owner-only headless invitation export. Private invitation material never
//! enters arguments, stdout, logs, or a remote plaintext HTTP request.
use anyhow::{bail, Context, Result};
use std::{net::SocketAddr, path::Path};

pub async fn create(
    target: SocketAddr,
    output: &Path,
    relay_url: Option<&str>,
    mode: mold_secure_client::ConnectionMode,
    direct_url: Option<&str>,
) -> Result<()> {
    anyhow::ensure!(
        mode != mold_secure_client::ConnectionMode::Regular || relay_url.is_none(),
        "Regular pairing cannot use a proxy endpoint"
    );
    let direct_url = match mode {
        mold_secure_client::ConnectionMode::Regular => Some(mold_secure_client::regular_origin(
            direct_url.context("Regular pairing requires --direct-url")?,
        )?),
        mold_secure_client::ConnectionMode::Secure => {
            anyhow::ensure!(direct_url.is_none(), "--direct-url requires --mode regular");
            None
        }
    };
    mold_relay::validate_target(target)?;
    if let Some(url) = relay_url {
        validate_relay_url(url)?;
    }
    // Check the output destination before consuming a short-lived invitation.
    if output.try_exists()? {
        bail!("invitation output already exists; choose a new private filename")
    }
    let raw = match std::env::var("MOLD_API_KEY") {
        Ok(value) => value,
        Err(_) => mold_secure_client::operator_key::read(
            &mold_core::Config::mold_dir().context("Mold home is unavailable")?,
        )?
        .context("Set MOLD_API_KEY to the host operator key or @owner-only-key-file")?
        .to_string(),
    };
    let key = if let Some(path) = raw.strip_prefix('@') {
        let bytes = mold_secure_client::read_private(Path::new(path))?;
        std::str::from_utf8(&bytes)?
            .lines()
            .map(str::trim)
            .find(|v| !v.is_empty() && !v.starts_with('#'))
            .context("operator key file is empty")?
            .to_string()
    } else {
        raw.split(',')
            .map(str::trim)
            .find(|v| !v.is_empty())
            .context("operator key is empty")?
            .to_string()
    };
    let client = reqwest::Client::builder()
        .no_proxy()
        .redirect(reqwest::redirect::Policy::none())
        .timeout(std::time::Duration::from_secs(10))
        .build()?;
    let request = client
        .post(format!("http://{target}/api/pairing/sessions"))
        .header("x-api-key", key);
    let request = if let Some(origin) = direct_url {
        request.json(&serde_json::json!({"mode":"regular","direct_url":origin}))
    } else {
        request.body("")
    };
    let mut response = request
        .send()
        .await
        .context("cannot contact the local Mold host")?
        .error_for_status()
        .context("host refused pairing; use its operator credential")?;
    let mut bytes = Vec::new();
    while let Some(chunk) = response.chunk().await? {
        if bytes.len() + chunk.len() > 32768 {
            bail!("invalid pairing response size")
        }
        bytes.extend_from_slice(&chunk);
    }
    let value: serde_json::Value = serde_json::from_slice(&bytes)?;
    let value = if mode == mold_secure_client::ConnectionMode::Regular {
        regular_invitation_payload(value)?
    } else {
        invitation_payload(value, relay_url)?
    };
    mold_secure_client::write_invitation(output, &value)?;
    println!("Pairing invitation saved to {}. It expires in two minutes; transfer it privately and run mold pair --invite-file <file> on the client.",output.display());
    println!(
        "Host public signing key: {}",
        value["host_key"].as_str().unwrap_or_default()
    );
    Ok(())
}
fn invitation_payload(
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

fn regular_invitation_payload(mut value: serde_json::Value) -> Result<serde_json::Value> {
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
