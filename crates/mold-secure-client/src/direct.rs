//! Explicit direct transport contracts. A proxy is never a Regular endpoint.
use anyhow::{ensure, Result};
use serde::{Deserialize, Serialize};

#[derive(Clone, Copy, Debug, Default, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub enum ConnectionMode {
    #[default]
    Secure,
    Regular,
}

pub fn regular_origin(value: &str) -> Result<String> {
    ensure!(
        value.len() <= 2048 && !value.chars().any(|c| c.is_whitespace() || c.is_control()),
        "invalid direct origin"
    );
    let url = reqwest::Url::parse(value)?;
    ensure!(
        matches!(url.scheme(), "http" | "https")
            && url.host_str().is_some()
            && url.username().is_empty()
            && url.password().is_none()
            && url.path() == "/"
            && url.query().is_none()
            && url.fragment().is_none()
            && url.port() != Some(0),
        "Regular mode requires a credential-free HTTP(S) origin without a path, query or fragment"
    );
    ensure!(
        !is_managed_proxy(&url),
        "the Mold proxy always requires a Secure connection"
    );
    Ok(url.origin().ascii_serialization())
}

pub fn is_managed_proxy(url: &reqwest::Url) -> bool {
    let host = url.host_str().unwrap_or_default().trim_end_matches('.');
    host == "mold-link.urandom.io" || host.ends_with(".mold-link.urandom.io")
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn regular_origins_match_cross_client_contract() {
        let fixture: serde_json::Value = serde_json::from_str(include_str!(
            "../../../studio/api/fixtures/direct-origins-v1.json"
        ))
        .unwrap();
        for case in fixture["accepted"].as_array().unwrap() {
            assert_eq!(
                regular_origin(case["input"].as_str().unwrap()).unwrap(),
                case["origin"].as_str().unwrap()
            );
        }
        for case in fixture["rejected"].as_array().unwrap() {
            let input = case.as_str().unwrap();
            assert!(regular_origin(input).is_err(), "{input:?}");
        }
    }
}
