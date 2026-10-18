//! The pairing link every Mold app shares:
//! `https://utensils.io/mold/pair#version=2&base_url=…` (a form-encoded
//! fragment, never a query, so the invitation never reaches the website), and
//! the native `moldstudio://pair#…` handoff.
//!
//! A Rust port of `studio/api/pairing.ts` (`mobilePairingUrl`,
//! `pairingLinkFields`, `parseMobilePairingPayload`), held to the same
//! `studio/api/pairing.fixtures.json` as the TypeScript and Swift ports. The
//! encoder must stay byte-identical to `mobilePairingUrl`: the same link has
//! to pair a phone, Studio, or the CLI.
use anyhow::{bail, ensure, Context, Result};
use serde_json::{Map, Value};

/// Where a pairing link points.
pub const PAIRING_LINK: &str = "https://utensils.io/mold/pair";
const LINK_HOST: &str = "utensils.io";
const LINK_PATH: &str = "/mold/pair";
/// A pairing code is a QR payload; anything longer is not one.
const MAX_INPUT: usize = 32768;
const NOT_A_CODE: &str = "that is not a Mold pairing link";
const UNSUPPORTED: &str = "that is not a supported Mold pairing link";

/// Read a pairing link (`https://utensils.io/mold/pair[/]#…`,
/// `moldstudio://pair#…`) or a raw JSON pairing payload into the invitation
/// value [`crate::pair`] / [`crate::pair_regular`] take: `type` is added,
/// `version` and `expires_at` are numbers, and `invite` / `endpoints` are
/// decoded from their JSON strings. Refused like `parseMobilePairingPayload`:
/// not a pairing code at all, or one this version cannot use.
pub fn parse_pairing_link(input: &str) -> Result<Value> {
    let input = input.trim();
    ensure!(input.len() <= MAX_INPUT, NOT_A_CODE);
    let value = match serde_json::from_str::<Value>(input) {
        Ok(value) => value,
        Err(_) => link_payload(&link_fields(input).context(NOT_A_CODE)?).context(NOT_A_CODE)?,
    };
    validate(&value).context(UNSUPPORTED)?;
    Ok(value)
}

/// The form-encoded fragment of a pairing link, or `None` for anything else.
fn link_fields(input: &str) -> Option<Vec<(String, String)>> {
    let url = reqwest::Url::parse(input).ok()?;
    if !url.username().is_empty() || url.password().is_some() {
        return None;
    }
    let fragment = match url.scheme() {
        "https" => {
            let path = url.path().strip_suffix('/').unwrap_or(url.path());
            (url.host_str() == Some(LINK_HOST)
                && url.port().is_none()
                && path == LINK_PATH
                && url.query().is_none_or(str::is_empty))
            .then(|| url.fragment().unwrap_or_default())?
        }
        "moldstudio" => (url.host_str() == Some("pair")
            && url.port().is_none()
            && url.path().is_empty()
            && url.query().is_none())
        .then(|| url.fragment().unwrap_or_default())?,
        _ => return None,
    };
    Some(form_decode(fragment))
}

/// The payload object `parseMobilePairingPayload` builds from link fields.
fn link_payload(fields: &[(String, String)]) -> Result<Value> {
    let get = |name: &str| {
        fields
            .iter()
            .find(|(key, _)| key == name)
            .map(|(_, v)| v.as_str())
    };
    let text = |name: &str| get(name).map_or(Value::Null, |v| Value::String(v.into()));
    let number = |name: &str| get(name).map_or(Value::Null, js_number);
    let regular = get("mode") == Some("regular");
    let mut payload = Map::new();
    payload.insert("type".into(), "mold.mobile-pairing".into());
    payload.insert("version".into(), number("version"));
    payload.insert("base_url".into(), text("base_url"));
    for key in [
        "namespace",
        "host_key",
        "host_noise_key",
        "host_binding_signature",
        "websocket_url",
        "invite_seed",
    ] {
        // A Regular link carries none of the Secure fields, and the strict
        // Regular invitation refuses any that are present.
        if !regular || get(key).is_some() {
            payload.insert(key.into(), text(key));
        }
    }
    if !regular || get("invite").is_some() {
        let invite = match get("invite") {
            Some(raw) => serde_json::from_str(raw)?,
            None => Value::Null,
        };
        payload.insert("invite".into(), invite);
    }
    payload.insert("instance_id".into(), text("instance_id"));
    payload.insert("name".into(), text("name"));
    if let Some(mode) = get("mode") {
        payload.insert("mode".into(), mode.into());
    }
    if regular {
        payload.insert("pairing_token".into(), text("pairing_token"));
        payload.insert("expires_at".into(), number("expires_at"));
    }
    if let Some(raw) = get("endpoints") {
        payload.insert("endpoints".into(), serde_json::from_str(raw)?);
    }
    Ok(Value::Object(payload))
}

/// `Number(text)` for the decimal integers producers write; anything else
/// becomes a value the validator refuses.
fn js_number(text: &str) -> Value {
    text.parse::<u64>()
        .map(Value::from)
        .unwrap_or_else(|_| Value::String(text.into()))
}

fn validate(value: &Value) -> Result<()> {
    let object = value.as_object().context("not an object")?;
    let text = |key: &str| object.get(key).and_then(Value::as_str);
    let hex = |key: &str, len: usize| {
        text(key).is_some_and(|v| {
            v.len() == len
                && v.bytes()
                    .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b))
        })
    };
    ensure!(
        text("type") == Some("mold.mobile-pairing")
            && object.get("version").and_then(Value::as_u64) == Some(2)
            && text("instance_id").is_some()
            && text("name").is_some(),
        "invalid pairing header"
    );
    let base = text("base_url").context("missing base_url")?;
    if text("mode") == Some("regular") {
        ensure!(
            !text("instance_id").unwrap_or_default().is_empty()
                && hex("pairing_token", 64)
                && object
                    .get("expires_at")
                    .and_then(Value::as_u64)
                    .is_some_and(|v| v > 0)
                && [
                    "namespace",
                    "host_key",
                    "host_noise_key",
                    "host_binding_signature",
                    "websocket_url",
                    "invite_seed",
                    "invite",
                    "endpoints",
                ]
                .iter()
                .all(|key| !object.contains_key(*key)),
            "invalid Regular pairing"
        );
        crate::regular_origin(base)?;
        return Ok(());
    }
    if let Some(mode) = object.get("mode") {
        ensure!(mode == "secure", "unknown pairing mode");
    }
    ensure!(valid_address(base, false), "invalid base_url");
    ensure!(
        hex("namespace", 32)
            && hex("host_key", 64)
            && hex("host_noise_key", 64)
            && hex("host_binding_signature", 128)
            && hex("invite_seed", 64),
        "invalid Secure pairing"
    );
    match object.get("websocket_url") {
        None | Some(Value::Null) => {}
        Some(Value::String(url)) => ensure!(valid_address(url, true), "invalid websocket_url"),
        Some(_) => bail!("invalid websocket_url"),
    }
    let invite: mold_secure::Invitation = serde_json::from_value(
        object
            .get("invite")
            .cloned()
            .context("missing invitation")?,
    )?;
    let invite_hex = |v: &str, len: usize| {
        v.len() == len
            && v.bytes()
                .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b))
    };
    ensure!(
        invite_hex(&invite.id, 32)
            && invite_hex(&invite.signing_key, 64)
            && invite_hex(&invite.signature, 128)
            && invite.expires_at > 0,
        "invalid invitation"
    );
    if let Some(endpoints) = object.get("endpoints") {
        let endpoints: Vec<crate::Endpoint> = serde_json::from_value(endpoints.clone())?;
        ensure!(endpoints.len() <= 8, "too many endpoints");
    }
    Ok(())
}

/// A credential-free http(s) (or ws(s)) URL with no query or fragment.
fn valid_address(value: &str, websocket: bool) -> bool {
    let Ok(url) = reqwest::Url::parse(value) else {
        return false;
    };
    let schemes: &[&str] = if websocket {
        &["ws", "wss"]
    } else {
        &["http", "https"]
    };
    value == value.trim()
        && !value.contains(['\r', '\n', '\t'])
        && url.username().is_empty()
        && url.password().is_none()
        && url.query().is_none()
        && url.fragment().is_none()
        && schemes.contains(&url.scheme())
        && url.host_str().is_some()
}

/// The pairing link for an invitation value, byte-identical to
/// `mobilePairingUrl`: the same fields in the same order, form-encoded the
/// way `URLSearchParams` encodes them.
pub fn pairing_link(payload: &Value) -> Result<String> {
    let object = payload.as_object().context("invalid pairing payload")?;
    let text = |key: &str| object.get(key).and_then(Value::as_str);
    let required = |key: &str| text(key).with_context(|| format!("pairing payload has no {key}"));
    let version = object
        .get("version")
        .and_then(Value::as_u64)
        .context("pairing payload has no version")?;
    let mut fields: Vec<(&str, String)> = vec![
        ("version", version.to_string()),
        ("base_url", required("base_url")?.into()),
        ("instance_id", required("instance_id")?.into()),
        ("name", text("name").unwrap_or("Mold host").into()),
    ];
    if text("mode") == Some("regular") {
        let expires_at = object
            .get("expires_at")
            .and_then(Value::as_u64)
            .context("pairing payload has no expires_at")?;
        fields.push(("mode", "regular".into()));
        fields.push(("pairing_token", required("pairing_token")?.into()));
        fields.push(("expires_at", expires_at.to_string()));
        return Ok(encode(&fields));
    }
    for key in [
        "namespace",
        "host_key",
        "host_noise_key",
        "host_binding_signature",
    ] {
        fields.push((key, required(key)?.into()));
    }
    if let Some(url) = text("websocket_url").filter(|v| !v.is_empty()) {
        fields.push(("websocket_url", url.into()));
    }
    fields.push(("invite_seed", required("invite_seed")?.into()));
    // The struct fixes JSON.stringify's key order (id, signingKey,
    // expiresAt, epoch, signature) whatever map order the value has.
    let invite: mold_secure::Invitation = serde_json::from_value(
        object
            .get("invite")
            .cloned()
            .context("pairing payload has no invite")?,
    )?;
    fields.push(("invite", serde_json::to_string(&invite)?));
    if let Some(endpoints) = object
        .get("endpoints")
        .and_then(Value::as_array)
        .filter(|list| !list.is_empty())
    {
        let endpoints: Vec<crate::Endpoint> =
            serde_json::from_value(Value::Array(endpoints.clone()))?;
        // JSON.stringify of `{kind, url}`, in that order.
        #[derive(serde::Serialize)]
        struct Ordered<'a> {
            kind: &'a str,
            url: &'a str,
        }
        let ordered: Vec<Ordered> = endpoints
            .iter()
            .map(|e| Ordered {
                kind: &e.kind,
                url: &e.url,
            })
            .collect();
        fields.push(("endpoints", serde_json::to_string(&ordered)?));
    }
    Ok(encode(&fields))
}

fn encode(fields: &[(&str, String)]) -> String {
    let body: Vec<String> = fields
        .iter()
        .map(|(key, value)| format!("{}={}", form_encode(key), form_encode(value)))
        .collect();
    format!("{PAIRING_LINK}#{}", body.join("&"))
}

/// The WHATWG `application/x-www-form-urlencoded` byte serializer
/// (`URLSearchParams.toString`): ASCII alphanumerics and `*-._` stay, a
/// space is `+`, every other byte is `%XX`.
fn form_encode(value: &str) -> String {
    let mut out = String::with_capacity(value.len());
    for byte in value.bytes() {
        match byte {
            b'a'..=b'z' | b'A'..=b'Z' | b'0'..=b'9' | b'*' | b'-' | b'.' | b'_' => {
                out.push(byte as char)
            }
            b' ' => out.push('+'),
            _ => out.push_str(&format!("%{byte:02X}")),
        }
    }
    out
}

fn form_decode(input: &str) -> Vec<(String, String)> {
    input
        .split('&')
        .filter(|pair| !pair.is_empty())
        .map(|pair| {
            let (key, value) = pair.split_once('=').unwrap_or((pair, ""));
            (percent_decode(key), percent_decode(value))
        })
        .collect()
}

fn percent_decode(input: &str) -> String {
    let bytes = input.as_bytes();
    let mut out = Vec::with_capacity(bytes.len());
    let mut i = 0;
    while i < bytes.len() {
        match bytes[i] {
            b'+' => out.push(b' '),
            b'%' => {
                let hex = bytes
                    .get(i + 1..i + 3)
                    .and_then(|h| std::str::from_utf8(h).ok())
                    .and_then(|h| u8::from_str_radix(h, 16).ok());
                match hex {
                    Some(byte) => {
                        out.push(byte);
                        i += 2;
                    }
                    None => out.push(b'%'),
                }
            }
            byte => out.push(byte),
        }
        i += 1;
    }
    String::from_utf8_lossy(&out).into_owned()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fixture() -> Value {
        serde_json::from_str(include_str!("../../../studio/api/pairing.fixtures.json")).unwrap()
    }

    /// The fixture's payloads are the `MobilePairingPayload` shape minus the
    /// `type`/`version` header every producer adds.
    fn with_header(payload: &Value) -> Value {
        let mut payload = payload.clone();
        payload["type"] = "mold.mobile-pairing".into();
        payload["version"] = 2.into();
        payload
    }

    #[test]
    fn every_fixture_payload_encodes_to_its_exact_link() {
        for case in fixture()["links"].as_array().unwrap() {
            assert_eq!(
                pairing_link(&with_header(&case["payload"])).unwrap(),
                case["link"].as_str().unwrap()
            );
        }
    }

    #[test]
    fn every_fixture_link_and_alias_parses_to_its_payload() {
        for case in fixture()["links"].as_array().unwrap() {
            let expected = with_header(&case["payload"]);
            let mut inputs = vec![case["link"].as_str().unwrap().to_string()];
            for alias in case["aliases"].as_array().unwrap() {
                inputs.push(alias.as_str().unwrap().to_string());
            }
            // The native handoff carries the same fragment.
            let fragment = inputs[0].split_once('#').unwrap().1.to_string();
            inputs.push(format!("moldstudio://pair#{fragment}"));
            inputs.push(serde_json::to_string(&expected).unwrap());
            for input in inputs {
                let parsed = parse_pairing_link(&input).unwrap_or_else(|e| panic!("{input}: {e}"));
                for (key, value) in expected.as_object().unwrap() {
                    assert_eq!(&parsed[key], value, "{key} in {input}");
                }
                // And it round-trips byte for byte.
                assert_eq!(
                    pairing_link(&parsed).unwrap(),
                    case["link"].as_str().unwrap()
                );
                if parsed["mode"] == "regular" {
                    let _: crate::RegularInvite = serde_json::from_value(parsed).unwrap();
                } else {
                    let invite: crate::PairingInvite = serde_json::from_value(parsed).unwrap();
                    assert_eq!(invite.origin, expected["base_url"]);
                    assert_eq!(
                        invite.instance_id.as_deref(),
                        expected["instance_id"].as_str()
                    );
                }
            }
        }
    }

    #[test]
    fn every_fixture_rejection_is_refused() {
        for input in fixture()["rejected"].as_array().unwrap() {
            let input = input.as_str().unwrap();
            assert!(parse_pairing_link(input).is_err(), "{input}");
        }
        let link = fixture()["links"][0]["link"].as_str().unwrap().to_string();
        let (base, fragment) = link.split_once('#').unwrap();
        for bad in [
            format!("{base}?x=1#{fragment}"),
            format!("https://utensils.io:8443/mold/pair#{fragment}"),
            format!("moldstudio://pair:1#{fragment}"),
            format!("moldstudio://pair/other#{fragment}"),
            format!("moldstudio://pair?token=private#{fragment}"),
            format!("{link}&pad={}", "a".repeat(MAX_INPUT)),
        ] {
            assert!(parse_pairing_link(&bad).is_err(), "{bad}");
        }
    }

    #[test]
    fn form_encoding_matches_url_search_params() {
        assert_eq!(form_encode("Café ☕ box"), "Caf%C3%A9+%E2%98%95+box");
        assert_eq!(form_encode("a*-._~!"), "a*-._%7E%21");
        assert_eq!(percent_decode("Caf%C3%A9+%E2%98%95+box"), "Café ☕ box");
        assert_eq!(percent_decode("100%"), "100%");
    }
}
