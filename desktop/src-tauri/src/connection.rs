//! Connection state machine: which engine the webview talks to. The built-in
//! engine is always the primary — remote hosts are additive list entries owned
//! by the frontend's hosts store, never a primary this ever switches to.
//! `Local` selects the desktop-owned embedded server; `External` selects a mold
//! server the user already runs on this machine. The embedded handle lives in
//! `commands::AppState::local_server`, so this Mac stays online for multi-host
//! routing and LAN discovery regardless.

use serde::Serialize;

pub enum Conn {
    Off,
    Local { base_url: String },
    External { base_url: String },
}

#[derive(Debug, Clone, Serialize, PartialEq, Eq)]
#[serde(rename_all = "camelCase")]
pub struct ConnectionInfo {
    pub mode: &'static str,
    pub base_url: Option<String>,
    pub api_key: Option<String>,
}

impl Conn {
    pub fn info(&self, local_api_key: &str) -> ConnectionInfo {
        match self {
            Conn::Off => ConnectionInfo {
                mode: "off",
                base_url: None,
                api_key: None,
            },
            Conn::Local { base_url } => ConnectionInfo {
                mode: "local",
                base_url: Some(base_url.clone()),
                api_key: Some(local_api_key.to_string()),
            },
            Conn::External { base_url } => ConnectionInfo {
                mode: "external",
                base_url: Some(base_url.clone()),
                api_key: Some(local_api_key.to_string()),
            },
        }
    }
}

/// Stable slug identifying a host, derived from its normalized URL. Used as
/// the saved-host id and as the per-host secret-name suffix
/// (`remote-api-key.<id>`), so it must only contain allowlist-safe bytes.
pub fn host_id(url: &str) -> String {
    let stripped = url
        .strip_prefix("https://")
        .or_else(|| url.strip_prefix("http://"))
        .unwrap_or(url);
    let mut id = String::with_capacity(stripped.len());
    let mut last_dash = true; // trim leading separators
    for c in stripped.chars() {
        if c.is_ascii_alphanumeric() {
            id.push(c.to_ascii_lowercase());
            last_dash = false;
        } else if !last_dash {
            id.push('-');
            last_dash = true;
        }
    }
    while id.ends_with('-') {
        id.pop();
    }
    id
}

/// Normalize a user-entered host: default scheme/port, strip trailing slashes.
pub fn normalize_host_url(input: &str) -> Result<String, String> {
    let trimmed = input.trim().trim_end_matches('/');
    if trimmed.is_empty() {
        return Err("Enter a host, like hal9000".into());
    }
    let has_explicit_scheme = trimmed.starts_with("http://") || trimmed.starts_with("https://");
    let with_scheme = if has_explicit_scheme {
        trimmed.to_string()
    } else {
        format!("http://{trimmed}")
    };
    let mut url =
        reqwest::Url::parse(&with_scheme).map_err(|_| "Enter a valid host.".to_string())?;
    if url.host_str().is_none() {
        return Err("Enter a valid host.".into());
    }
    if !has_explicit_scheme && url.port().is_none() {
        url.set_port(Some(7680))
            .map_err(|_| "Enter a valid host.".to_string())?;
    }
    Ok(url.to_string().trim_end_matches('/').to_string())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn info_reports_each_mode() {
        assert_eq!(Conn::Off.info("k").mode, "off");
        let external = Conn::External {
            base_url: "http://127.0.0.1:7680".into(),
        };
        let info = external.info("k");
        assert_eq!(info.mode, "external");
        assert_eq!(info.base_url.as_deref(), Some("http://127.0.0.1:7680"));
        assert_eq!(info.api_key.as_deref(), Some("k"));
        let local = Conn::Local {
            base_url: "http://127.0.0.1:49152".into(),
        };
        assert_eq!(local.info("k").mode, "local");
        assert_eq!(local.info("k").api_key.as_deref(), Some("k"));
    }

    #[test]
    fn host_ids_are_stable_slugs() {
        assert_eq!(host_id("http://hal9000:7680"), "hal9000-7680");
        assert_eq!(host_id("https://mold.example.com"), "mold-example-com");
        assert_eq!(
            host_id("https://abc123-7680.proxy.runpod.net"),
            "abc123-7680-proxy-runpod-net"
        );
        // Same host, same id — scheme and case don't matter.
        assert_eq!(host_id("http://Studio.local:7680"), "studio-local-7680");
        assert_eq!(host_id("studio.local:7680"), "studio-local-7680");
    }

    #[test]
    fn normalizes_host_urls() {
        assert_eq!(
            normalize_host_url("hal9000").unwrap(),
            "http://hal9000:7680"
        );
        assert_eq!(
            normalize_host_url("studio.local:7680").unwrap(),
            "http://studio.local:7680"
        );
        assert_eq!(
            normalize_host_url("http://studio.local:7680///").unwrap(),
            "http://studio.local:7680"
        );
        assert_eq!(
            normalize_host_url("https://mold.example.com/").unwrap(),
            "https://mold.example.com"
        );
        assert_eq!(
            normalize_host_url("http://mold.example.com/").unwrap(),
            "http://mold.example.com"
        );
        assert!(normalize_host_url("   ").is_err());
    }
}
