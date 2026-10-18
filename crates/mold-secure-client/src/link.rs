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
    normalized(value).context(UNSUPPORTED)
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

/// What JavaScript's `Number(text)` reads from a link field, as a JSON
/// number (an integer when it is one); `null` for `NaN` or an infinity.
fn js_number(text: &str) -> Value {
    let number = js_number_value(text.trim());
    if number.fract() == 0.0 && (0.0..=u64::MAX as f64).contains(&number) {
        return Value::from(number as u64);
    }
    serde_json::Number::from_f64(number).map_or(Value::Null, Value::Number)
}

fn js_number_value(text: &str) -> f64 {
    if text.is_empty() {
        return 0.0;
    }
    for (prefix, radix) in [
        ("0x", 16),
        ("0X", 16),
        ("0o", 8),
        ("0O", 8),
        ("0b", 2),
        ("0B", 2),
    ] {
        if let Some(digits) = text.strip_prefix(prefix) {
            return u64::from_str_radix(digits, radix).map_or(f64::NAN, |v| v as f64);
        }
    }
    match text {
        "Infinity" | "+Infinity" => return f64::INFINITY,
        "-Infinity" => return f64::NEG_INFINITY,
        _ => {}
    }
    // Rust also reads "inf" and "NaN"; JavaScript's decimal grammar does not.
    if text
        .bytes()
        .any(|b| !(b.is_ascii_digit() || matches!(b, b'.' | b'e' | b'E' | b'+' | b'-')))
    {
        return f64::NAN;
    }
    text.parse::<f64>().unwrap_or(f64::NAN)
}

/// A JSON number that is exactly the whole number `n` the way `===` sees it.
fn whole(value: Option<&Value>) -> Option<u64> {
    let number = value?.as_f64()?;
    (number.fract() == 0.0 && (0.0..=9_007_199_254_740_991.0).contains(&number))
        .then_some(number as u64)
}

/// `parseMobilePairingPayload`'s checks, returning the payload with
/// `version`/`expires_at` as integers and endpoints as `parseConnectionEndpoints`
/// leaves them (normalized origins, duplicates removed).
fn normalized(mut value: Value) -> Result<Value> {
    let object = value.as_object_mut().context("not an object")?;
    let text = |object: &Map<String, Value>, key: &str| {
        object.get(key).and_then(Value::as_str).map(str::to_owned)
    };
    let hex = |object: &Map<String, Value>, key: &str, len: usize| {
        text(object, key).is_some_and(|v| is_hex(&v, len))
    };
    ensure!(
        text(object, "type").as_deref() == Some("mold.mobile-pairing")
            && whole(object.get("version")) == Some(2)
            && text(object, "instance_id").is_some()
            && text(object, "name").is_some(),
        "invalid pairing header"
    );
    object.insert("version".into(), 2.into());
    let base = text(object, "base_url").context("missing base_url")?;
    if text(object, "mode").as_deref() == Some("regular") {
        let expires_at = whole(object.get("expires_at")).filter(|v| *v > 0);
        ensure!(
            !text(object, "instance_id").unwrap_or_default().is_empty()
                && hex(object, "pairing_token", 64)
                && expires_at.is_some()
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
        object.insert("expires_at".into(), expires_at.into());
        crate::regular_origin(&base)?;
        return Ok(value);
    }
    if let Some(mode) = object.get("mode") {
        ensure!(mode == "secure", "unknown pairing mode");
    }
    ensure!(valid_address(&base, false), "invalid base_url");
    ensure!(
        hex(object, "namespace", 32)
            && hex(object, "host_key", 64)
            && hex(object, "host_noise_key", 64)
            && hex(object, "host_binding_signature", 128)
            && hex(object, "invite_seed", 64),
        "invalid Secure pairing"
    );
    // Present, and null or a credential-free wss:// URL.
    match object.get("websocket_url") {
        Some(Value::Null) => {}
        Some(Value::String(url)) => ensure!(valid_address(url, true), "invalid websocket_url"),
        _ => bail!("invalid websocket_url"),
    }
    let invite = object.get("invite").context("missing invitation")?;
    ensure!(
        invite["id"].as_str().is_some_and(|v| is_hex(v, 32))
            && invite["signingKey"].as_str().is_some_and(|v| is_hex(v, 64))
            && invite["signature"].as_str().is_some_and(|v| is_hex(v, 128))
            && whole(invite.get("expiresAt")).is_some_and(|v| v > 0)
            && whole(invite.get("epoch")).is_some(),
        "invalid invitation"
    );
    let _: mold_secure::Invitation = serde_json::from_value(invite.clone())?;
    if let Some(endpoints) = object.get("endpoints") {
        let endpoints = connection_endpoints(endpoints)?;
        object.insert("endpoints".into(), serde_json::to_value(endpoints)?);
    }
    Ok(value)
}

fn is_hex(value: &str, len: usize) -> bool {
    value.len() == len
        && value
            .bytes()
            .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b))
}

/// `parseConnectionEndpoints` (`studio/api/connectionRoutes.ts`): at most
/// eight `lan`/`tailscale`/`relay` endpoints, each a plain http(s) origin
/// (relay: https only) that is not loopback, link-local or unspecified;
/// returned as origins, duplicates dropped.
fn connection_endpoints(value: &Value) -> Result<Vec<crate::Endpoint>> {
    let list = value.as_array().context("invalid connection addresses")?;
    ensure!(list.len() <= 8, "too many connection addresses");
    let mut out: Vec<crate::Endpoint> = Vec::new();
    for item in list {
        let raw = item["url"].as_str().context("invalid connection address")?;
        let kind = item["kind"].as_str().unwrap_or_default();
        ensure!(
            raw.len() <= 2048
                && !raw.contains(['?', '#'])
                && ["lan", "tailscale", "relay"].contains(&kind),
            "invalid connection address"
        );
        let url = reqwest::Url::parse(raw).context("invalid connection address")?;
        let host = url
            .host_str()
            .unwrap_or_default()
            .to_ascii_lowercase()
            .trim_start_matches('[')
            .trim_end_matches(']')
            .trim_end_matches('.')
            .to_string();
        let link_local_v6 = host.len() > 4
            && host.starts_with("fe")
            && matches!(host.as_bytes()[2], b'8' | b'9' | b'a' | b'b')
            && host.as_bytes()[3].is_ascii_hexdigit()
            && host.as_bytes()[4] == b':';
        let forbidden = host.is_empty()
            || host == "localhost"
            || host.ends_with(".localhost")
            || host == "::"
            || host == "::1"
            || host == "0.0.0.0"
            || host.starts_with("127.")
            || host.starts_with("169.254.")
            || link_local_v6
            || host.contains('%');
        ensure!(
            !forbidden
                && matches!(url.scheme(), "http" | "https")
                && url.username().is_empty()
                && url.password().is_none()
                && url.path() == "/"
                && (kind != "relay" || url.scheme() == "https"),
            "connection addresses must be plain HTTP(S) origins"
        );
        let origin = url.origin().ascii_serialization();
        if !out.iter().any(|e| e.url == origin) {
            out.push(crate::Endpoint {
                url: origin,
                kind: kind.into(),
            });
        }
    }
    Ok(out)
}

/// `validPairingAddress`: a credential-free URL with no query or fragment;
/// an API origin is http(s) with path `/`, a WebSocket is wss.
fn valid_address(value: &str, websocket: bool) -> bool {
    let Ok(url) = reqwest::Url::parse(value) else {
        return false;
    };
    value == value.trim()
        && !value.contains(['\r', '\n', '\t'])
        && url.username().is_empty()
        && url.password().is_none()
        && url.query().is_none()
        && url.fragment().is_none()
        && url.host_str().is_some()
        && if websocket {
            url.scheme() == "wss"
        } else {
            matches!(url.scheme(), "http" | "https") && url.path() == "/"
        }
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
        // mobilePairingUrl writes what parseConnectionEndpoints leaves.
        let endpoints = connection_endpoints(&Value::Array(endpoints.clone()))?;
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
    use serde_json::json;

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

    fn secure_json() -> Value {
        with_header(&fixture()["links"][0]["payload"])
    }

    fn parses(value: &Value) -> Result<Value> {
        parse_pairing_link(&serde_json::to_string(value).unwrap())
    }

    /// `validPairingAddress`: a Secure API origin has path `/`; the relay
    /// WebSocket is wss only; a JSON payload names `websocket_url` (null is
    /// fine) like the link always does.
    #[test]
    fn secure_addresses_follow_valid_pairing_address() {
        let mut value = secure_json();
        value["websocket_url"] = Value::Null;
        assert!(parses(&value).is_ok());
        value.as_object_mut().unwrap().remove("websocket_url");
        assert!(parses(&value).is_err(), "websocket_url must be present");
        for websocket in ["ws://relay.example/ws", "https://relay.example/ws"] {
            let mut value = secure_json();
            value["websocket_url"] = websocket.into();
            assert!(parses(&value).is_err(), "{websocket}");
        }
        let mut value = secure_json();
        value["base_url"] = "http://studio.local:7680/api".into();
        assert!(parses(&value).is_err());
    }

    /// Link numbers are read the way `Number()` reads them.
    #[test]
    fn link_numbers_follow_javascript_number() {
        let link = fixture()["links"][4]["link"].as_str().unwrap().to_string();
        for version in ["2", "2.0", "+2", "0x2", "2e0", "%202%20"] {
            let input = link.replace("version=2", &format!("version={version}"));
            let parsed = parse_pairing_link(&input).unwrap_or_else(|e| panic!("{version}: {e}"));
            assert_eq!(parsed["version"], 2, "{version}");
            let _: crate::RegularInvite = serde_json::from_value(parsed).unwrap();
        }
        for version in ["two", "inf", "NaN", "2.5", "Infinity", ""] {
            let input = link.replace("version=2", &format!("version={version}"));
            assert!(parse_pairing_link(&input).is_err(), "{version}");
        }
        let expiry = link.replace("expires_at=4000000000", "expires_at=4e9");
        assert_eq!(
            parse_pairing_link(&expiry).unwrap()["expires_at"],
            4_000_000_000_u64
        );
        let fractional = link.replace("expires_at=4000000000", "expires_at=4000000000.5");
        assert!(parse_pairing_link(&fractional).is_err());
        assert_eq!(js_number(""), Value::from(0_u64));
        assert_eq!(js_number(" 0b10 "), Value::from(2_u64));
    }

    /// `parseConnectionEndpoints`. These refusals are Rust- and TS-only: the
    /// TypeScript throws them with its own message (not the shared
    /// fixture's), and Swift drops a bad endpoint and then refuses the count
    /// mismatch, so they cannot live in pairing.fixtures.json.
    #[test]
    fn endpoints_follow_parse_connection_endpoints() {
        let with = |endpoints: Value| {
            let mut value = secure_json();
            value["endpoints"] = endpoints;
            parses(&value)
        };
        for bad in [
            json!([{"kind": "wifi", "url": "http://192.168.1.20:7680"}]),
            json!([{"kind": "lan", "url": "http://localhost:7680"}]),
            json!([{"kind": "lan", "url": "http://box.localhost:7680"}]),
            json!([{"kind": "lan", "url": "http://127.0.0.1:7680"}]),
            json!([{"kind": "lan", "url": "http://169.254.1.1:7680"}]),
            json!([{"kind": "lan", "url": "http://[fe80::1]:7680"}]),
            json!([{"kind": "lan", "url": "http://[::1]:7680"}]),
            json!([{"kind": "lan", "url": "http://0.0.0.0:7680"}]),
            json!([{"kind": "lan", "url": "http://192.168.1.20:7680/api"}]),
            json!([{"kind": "lan", "url": "http://192.168.1.20:7680?x=1"}]),
            json!([{"kind": "lan", "url": "http://192.168.1.20:7680#x"}]),
            json!([{"kind": "lan", "url": "http://u:p@192.168.1.20:7680"}]),
            json!([{"kind": "lan", "url": "ftp://192.168.1.20"}]),
            json!([{"kind": "relay", "url": "http://relay.example"}]),
            json!([{"kind": "lan", "url": format!("http://{}.example", "a".repeat(2048))}]),
            json!((0..9)
                .map(|i| json!({"kind": "lan", "url": format!("http://10.0.0.{i}:7680")}))
                .collect::<Vec<_>>()),
        ] {
            assert!(with(bad.clone()).is_err(), "{bad}");
        }
        // Normalized to origins and deduplicated, as TypeScript returns them
        // (Swift refuses the duplicate outright).
        let parsed = with(json!([
            {"kind": "lan", "url": "HTTP://192.168.1.20:7680/"},
            {"kind": "lan", "url": "http://192.168.1.20:7680"},
            {"kind": "relay", "url": "https://relay.example:443"},
        ]))
        .unwrap();
        assert_eq!(
            parsed["endpoints"],
            json!([
                {"kind": "lan", "url": "http://192.168.1.20:7680"},
                {"kind": "relay", "url": "https://relay.example"},
            ])
        );
        let mut value = secure_json();
        value["endpoints"] = parsed["endpoints"].clone();
        value["endpoints"]
            .as_array_mut()
            .unwrap()
            .push(json!({"kind": "lan", "url": "http://192.168.1.20:7680/"}));
        let link = pairing_link(&value).unwrap();
        assert!(link.ends_with(
            "endpoints=%5B%7B%22kind%22%3A%22lan%22%2C%22url%22%3A%22http%3A%2F%2F192.168.1.20%3A7680%22%7D%2C%7B%22kind%22%3A%22relay%22%2C%22url%22%3A%22https%3A%2F%2Frelay.example%22%7D%5D"
        ), "{link}");
    }

    #[test]
    fn form_encoding_matches_url_search_params() {
        assert_eq!(form_encode("Café ☕ box"), "Caf%C3%A9+%E2%98%95+box");
        assert_eq!(form_encode("a*-._~!"), "a*-._%7E%21");
        assert_eq!(percent_decode("Caf%C3%A9+%E2%98%95+box"), "Café ☕ box");
        assert_eq!(percent_decode("100%"), "100%");
    }
}
