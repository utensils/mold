//! Owner-only headless invitation export. Private invitation material never
//! enters arguments, stdout, logs, or a remote plaintext HTTP request.
use anyhow::{bail, Context, Result};
use std::{net::SocketAddr, path::Path};

pub async fn create(target: SocketAddr, output: &Path, relay_url: Option<&str>) -> Result<()> {
    mold_relay::validate_target(target)?;
    if let Some(url) = relay_url {
        validate_relay_url(url)?;
    }
    // Check the output destination before consuming a short-lived invitation.
    if output.try_exists()? {
        bail!("invitation output already exists; choose a new private filename")
    }
    let raw = std::env::var("MOLD_API_KEY")
        .context("Set MOLD_API_KEY to the host operator key or @owner-only-key-file")?;
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
    let mut response = client
        .post(format!("http://{target}/api/pairing/sessions"))
        .header("x-api-key", key)
        .body("")
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
    let value = invitation_payload(serde_json::from_slice(&bytes)?, relay_url)?;
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
#[cfg(test)]
mod tests {
    use super::*;
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
        let value = serde_json::json!({"protocol":3,"hostname":"host","endpoints":[{"kind":"lan","url":"http://192.168.1.5:4321"},{"kind":"relay","url":"https://host.example"}],"websocket_url":"wss://relay.example"});
        let output = invitation_payload(value, None).unwrap();
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
