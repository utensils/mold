//! Connection state machine: which engine the webview talks to.
//!
//! `Local` is the embedded in-process engine; `External` is a mold server the
//! user already runs on this machine (detected on the well-known port, used
//! instead of embedding to avoid two download drivers/model caches sharing
//! mold.db); `Remote` is a network host.

use serde::Serialize;

use crate::server::EngineHandle;

pub enum Conn {
    Off,
    Local(EngineHandle),
    External {
        base_url: String,
    },
    Remote {
        url: String,
        api_key: Option<String>,
    },
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
            Conn::Local(engine) => ConnectionInfo {
                mode: "local",
                base_url: Some(engine.base_url()),
                api_key: Some(local_api_key.to_string()),
            },
            Conn::External { base_url } => ConnectionInfo {
                mode: "external",
                base_url: Some(base_url.clone()),
                api_key: None,
            },
            Conn::Remote { url, api_key } => ConnectionInfo {
                mode: "remote",
                base_url: Some(url.clone()),
                api_key: api_key.clone(),
            },
        }
    }
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
        let remote = Conn::Remote {
            url: "http://h:1".into(),
            api_key: Some("secret".into()),
        };
        let info = remote.info("local-key");
        assert_eq!(info.mode, "remote");
        assert_eq!(info.base_url.as_deref(), Some("http://h:1"));
        assert_eq!(info.api_key.as_deref(), Some("secret"));
        let external = Conn::External {
            base_url: "http://127.0.0.1:7680".into(),
        };
        assert_eq!(external.info("k").api_key, None);
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
