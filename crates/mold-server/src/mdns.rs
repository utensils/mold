//! mDNS / DNS-SD advertising and discovery for mold servers.
//!
//! A running `mold serve` advertises itself on the local network as a
//! `_mold._tcp.local.` service so desktop clients and the `mold server
//! discover` CLI can find it without knowing its address. TXT records carry a
//! small identity payload (version, git sha, whether auth is required, a GPU
//! summary, queue capacity, protocol) so a browser can render a useful list
//! without probing every host.
//!
//! The module splits cleanly into pure, unit-testable helpers (name sanitising,
//! env toggles, TXT assembly, resolved-service → [`DiscoveredServer`] mapping)
//! and the two network entry points [`register`] and [`discover`] that talk to
//! the `mdns-sd` daemon. Only the latter touch multicast, so the pure helpers
//! carry the bulk of the test coverage.

use std::collections::BTreeMap;
use std::net::IpAddr;
use std::time::{Duration, Instant};

use anyhow::{Context, Result};
use mdns_sd::{IfKind, ServiceDaemon, ServiceEvent, ServiceInfo};
use serde::Serialize;

/// DNS-SD service type advertised and browsed by mold.
pub const SERVICE_TYPE: &str = "_mold._tcp.local.";

/// Longest instance label we will emit. DNS labels cap at 63 bytes; we reserve
/// headroom for the `-{port}` suffix so the sanitised host never overflows.
const MAX_INSTANCE_BYTES: usize = 63;

/// Handle that keeps an advertisement alive. Dropping it (or calling
/// [`MdnsGuard::shutdown`]) unregisters the service and stops the daemon so the
/// server disappears from the network promptly on shutdown.
pub struct MdnsGuard {
    daemon: ServiceDaemon,
    fullname: String,
}

impl MdnsGuard {
    /// Unregister the service and shut the daemon down. Best-effort: errors are
    /// logged, never propagated, because this runs on the server's exit path.
    pub fn shutdown(self) {
        match self.daemon.unregister(&self.fullname) {
            Ok(_) => tracing::info!(fullname = %self.fullname, "mDNS: unregistered service"),
            Err(e) => tracing::warn!(error = %e, "mDNS: unregister failed"),
        }
        if let Err(e) = self.daemon.shutdown() {
            tracing::warn!(error = %e, "mDNS: daemon shutdown failed");
        }
    }
}

/// A mold server discovered on the local network.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct DiscoveredServer {
    /// Instance label without the service-type suffix (e.g. `hal9000-7680`).
    pub name: String,
    /// Best host to connect to — the preferred IPv4 address when present,
    /// otherwise an IPv6 address, otherwise the advertised hostname.
    pub host: String,
    /// Every resolved address, IPv4 first, as strings.
    pub addresses: Vec<String>,
    /// Advertised TCP port.
    pub port: u16,
    /// Ready-to-use base URL (`http://host:port`).
    pub url: String,
    /// mold version string from the TXT record, if present.
    pub version: Option<String>,
    /// Whether the server requires an API key (`auth=1`).
    pub auth_required: bool,
    /// All decoded TXT key/value pairs.
    pub txt: BTreeMap<String, String>,
}

/// Whether mDNS advertising is enabled from the environment.
///
/// `MOLD_MDNS` unset / `1` / `true` / `yes` / `on` → enabled; `0` / `false` /
/// `no` / `off` → disabled (case-insensitive). Any other value is treated as
/// enabled — advertising is the safe, expected default when compiled in.
pub fn enabled_from_env() -> bool {
    match std::env::var("MOLD_MDNS") {
        Ok(v) => !matches!(
            v.trim().to_ascii_lowercase().as_str(),
            "0" | "false" | "no" | "off"
        ),
        Err(_) => true,
    }
}

/// Whether a `bind` address is worth advertising. Loopback and unspecified-but-
/// clearly-local binds are pointless to announce on the LAN. A wildcard bind
/// (`0.0.0.0` / `::`) IS advertisable — `enable_addr_auto` resolves it to the
/// real interface addresses.
pub fn is_advertisable(bind: &str, _port: u16) -> bool {
    let host = bind.trim();
    if host.is_empty() {
        return false;
    }
    // Wildcard binds are fine — the daemon fills in real interface addresses.
    if matches!(host, "0.0.0.0" | "::" | "[::]") {
        return true;
    }
    match host
        .trim_start_matches('[')
        .trim_end_matches(']')
        .parse::<IpAddr>()
    {
        Ok(ip) => !ip.is_loopback(),
        // A hostname (e.g. "localhost") — treat literal loopback names as
        // non-advertisable, anything else as advertisable.
        Err(_) => !host.eq_ignore_ascii_case("localhost"),
    }
}

/// Build a stable, DNS-safe instance label from the hostname and port.
///
/// Non-alphanumeric runs collapse to a single dash, leading/trailing dashes are
/// trimmed, and the label is truncated so that `-{port}` fits within
/// [`MAX_INSTANCE_BYTES`]. Empty input falls back to `mold`.
pub fn instance_name(hostname: &str, port: u16) -> String {
    let mut sanitized = String::with_capacity(hostname.len());
    let mut last_dash = false;
    for ch in hostname.chars() {
        if ch.is_ascii_alphanumeric() {
            sanitized.push(ch);
            last_dash = false;
        } else if !last_dash {
            sanitized.push('-');
            last_dash = true;
        }
    }
    let base = sanitized.trim_matches('-');
    let base = if base.is_empty() { "mold" } else { base };

    let suffix = format!("-{port}");
    let budget = MAX_INSTANCE_BYTES.saturating_sub(suffix.len());
    // Truncate on a char boundary within the byte budget (host chars are ASCII
    // here, but stay defensive).
    let mut trimmed = base;
    while trimmed.len() > budget {
        match trimmed.char_indices().next_back() {
            Some((idx, _)) => trimmed = &trimmed[..idx],
            None => break,
        }
    }
    let trimmed = trimmed.trim_end_matches('-');
    format!("{trimmed}{suffix}")
}

/// Assemble the TXT records advertised alongside the service.
///
/// Keys: `version`, `sha`, `auth` (`1`/`0`), `gpu`, `queue`, `proto`. Values are
/// clamped defensively — a single TXT entry must stay well under 255 bytes.
pub fn build_txt_records(
    version: &str,
    git_sha: &str,
    auth_required: bool,
    gpu_summary: &str,
    queue_capacity: usize,
) -> Vec<(String, String)> {
    fn clamp(s: &str, max: usize) -> String {
        if s.len() <= max {
            s.to_string()
        } else {
            let mut end = max;
            while end > 0 && !s.is_char_boundary(end) {
                end -= 1;
            }
            s[..end].to_string()
        }
    }
    vec![
        ("version".to_string(), clamp(version, 200)),
        ("sha".to_string(), clamp(git_sha, 64)),
        (
            "auth".to_string(),
            if auth_required { "1" } else { "0" }.to_string(),
        ),
        ("gpu".to_string(), clamp(gpu_summary, 200)),
        ("queue".to_string(), queue_capacity.to_string()),
        ("proto".to_string(), "http".to_string()),
    ]
}

/// Map the primitive parts of a resolved service into a [`DiscoveredServer`].
///
/// Kept free of `mdns-sd` types so it is unit-testable. Addresses are ordered
/// IPv4-first; the connect `host`/`url` prefer the first IPv4, then IPv6
/// (bracketed), then the advertised hostname (trailing dot stripped).
pub fn from_service_parts(
    fullname: &str,
    hostname: &str,
    port: u16,
    mut addresses: Vec<IpAddr>,
    txt: Vec<(String, String)>,
) -> DiscoveredServer {
    // Drop loopback addresses whenever at least one routable address exists —
    // a remote client that resolved the loopback record first must never be
    // handed a URL that points back at itself. If loopback is all we have
    // (e.g. discovering our own single-homed box), keep it.
    if addresses.iter().any(|ip| !ip.is_loopback()) {
        addresses.retain(|ip| !ip.is_loopback());
    }

    // IPv4 first, then IPv6, each group sorted for determinism.
    addresses.sort_by(|a, b| match (a, b) {
        (IpAddr::V4(_), IpAddr::V6(_)) => std::cmp::Ordering::Less,
        (IpAddr::V6(_), IpAddr::V4(_)) => std::cmp::Ordering::Greater,
        _ => a.cmp(b),
    });
    addresses.dedup();

    let name = fullname
        .strip_suffix(&format!(".{SERVICE_TYPE}"))
        .unwrap_or(fullname)
        .to_string();

    let txt_map: BTreeMap<String, String> = txt.into_iter().collect();

    let first_v4 = addresses.iter().find(|ip| ip.is_ipv4());
    let host = if let Some(ip) = first_v4 {
        ip.to_string()
    } else if let Some(ip) = addresses.first() {
        match ip {
            IpAddr::V6(_) => format!("[{ip}]"),
            IpAddr::V4(_) => ip.to_string(),
        }
    } else {
        hostname.trim_end_matches('.').to_string()
    };

    DiscoveredServer {
        name,
        url: format!("http://{host}:{port}"),
        host,
        addresses: addresses.iter().map(|ip| ip.to_string()).collect(),
        port,
        version: txt_map.get("version").cloned(),
        auth_required: txt_map.get("auth").map(|v| v == "1").unwrap_or(false),
        txt: txt_map,
    }
}

/// Register a mold advertisement on the local network.
///
/// `txt` is the output of [`build_txt_records`]. The service binds to all
/// interface addresses via `enable_addr_auto`, so a wildcard server bind still
/// advertises the machine's real LAN addresses.
pub fn register(port: u16, txt: Vec<(String, String)>) -> Result<MdnsGuard> {
    let daemon = ServiceDaemon::new().context("failed to start mDNS daemon")?;

    // Never announce on loopback interfaces. `enable_addr_auto` otherwise
    // includes 127.0.0.1/::1, and a remote client that resolves that record
    // first would be handed a URL pointing back at itself. Best-effort: if the
    // daemon rejects the call we still fall back to the browse-side filter.
    if let Err(e) = daemon.disable_interface(vec![IfKind::LoopbackV4, IfKind::LoopbackV6]) {
        tracing::warn!(error = %e, "mDNS: could not disable loopback interfaces");
    }

    let raw_host = hostname::get()
        .ok()
        .and_then(|h| h.into_string().ok())
        .unwrap_or_else(|| "mold".to_string());
    let instance = instance_name(&raw_host, port);
    // `enable_addr_auto` requires a hostname ending in `.local.`.
    let host_label = {
        let base = raw_host.split('.').next().unwrap_or("mold");
        let base = if base.is_empty() { "mold" } else { base };
        format!("{base}.local.")
    };

    let props: Vec<(&str, &str)> = txt.iter().map(|(k, v)| (k.as_str(), v.as_str())).collect();

    let service = ServiceInfo::new(
        SERVICE_TYPE,
        &instance,
        &host_label,
        "", // addresses filled in by enable_addr_auto
        port,
        &props[..],
    )
    .context("failed to build mDNS service info")?
    .enable_addr_auto();

    let fullname = service.get_fullname().to_string();
    daemon
        .register(service)
        .context("failed to register mDNS service")?;
    tracing::info!(fullname = %fullname, port, "mDNS: advertising service");

    Ok(MdnsGuard { daemon, fullname })
}

/// Browse the local network for mold servers for up to `timeout`.
///
/// Collects `ServiceResolved` events, dedupes by fullname (last wins), and
/// returns the results sorted by name. Blocking — call from a blocking context
/// (CLI, or `spawn_blocking` in async).
pub fn discover(timeout: Duration) -> Result<Vec<DiscoveredServer>> {
    let daemon = ServiceDaemon::new().context("failed to start mDNS daemon")?;
    let receiver = daemon
        .browse(SERVICE_TYPE)
        .context("failed to start mDNS browse")?;

    let mut found: BTreeMap<String, DiscoveredServer> = BTreeMap::new();
    let deadline = Instant::now() + timeout;
    loop {
        let remaining = deadline.saturating_duration_since(Instant::now());
        if remaining.is_zero() {
            break;
        }
        match receiver.recv_timeout(remaining) {
            Ok(ServiceEvent::ServiceResolved(info)) => {
                let addresses: Vec<IpAddr> =
                    info.addresses.iter().map(|a| a.to_ip_addr()).collect();
                let txt: Vec<(String, String)> = info
                    .txt_properties
                    .iter()
                    .map(|p| (p.key().to_string(), p.val_str().to_string()))
                    .collect();
                let server =
                    from_service_parts(&info.fullname, &info.host, info.port, addresses, txt);
                found.insert(info.fullname.clone(), server);
            }
            Ok(_) => {}
            Err(_) => break, // timeout or channel closed
        }
    }

    let _ = daemon.shutdown();

    let mut servers: Vec<DiscoveredServer> = found.into_values().collect();
    servers.sort_by(|a, b| a.name.cmp(&b.name));
    Ok(servers)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::net::{Ipv4Addr, Ipv6Addr};

    #[test]
    fn is_advertisable_rejects_loopback_and_localhost() {
        assert!(!is_advertisable("127.0.0.1", 7680));
        assert!(!is_advertisable("::1", 7680));
        assert!(!is_advertisable("[::1]", 7680));
        assert!(!is_advertisable("localhost", 7680));
        assert!(!is_advertisable("LocalHost", 7680));
        assert!(!is_advertisable("", 7680));
    }

    #[test]
    fn is_advertisable_accepts_wildcard_and_lan() {
        assert!(is_advertisable("0.0.0.0", 7680));
        assert!(is_advertisable("::", 7680));
        assert!(is_advertisable("[::]", 7680));
        assert!(is_advertisable("192.168.1.10", 7680));
        assert!(is_advertisable("10.0.0.5", 7680));
        assert!(is_advertisable("hal9000", 7680));
    }

    #[test]
    fn instance_name_sanitizes_and_suffixes() {
        assert_eq!(instance_name("hal9000", 7680), "hal9000-7680");
        assert_eq!(instance_name("my.host.local", 7680), "my-host-local-7680");
        assert_eq!(instance_name("weird__name!!", 8080), "weird-name-8080");
        assert_eq!(instance_name("--edge--", 7680), "edge-7680");
    }

    #[test]
    fn instance_name_empty_falls_back() {
        assert_eq!(instance_name("", 7680), "mold-7680");
        assert_eq!(instance_name("!!!", 7680), "mold-7680");
    }

    #[test]
    fn instance_name_truncates_within_budget() {
        let long = "a".repeat(200);
        let name = instance_name(&long, 7680);
        assert!(name.len() <= MAX_INSTANCE_BYTES, "got {} bytes", name.len());
        assert!(name.ends_with("-7680"));
        assert!(!name.contains("--"));
    }

    #[test]
    fn build_txt_records_shape() {
        let txt = build_txt_records("0.14.0", "abc1234", true, "1xRTX 4090", 200);
        let map: BTreeMap<_, _> = txt.into_iter().collect();
        assert_eq!(map["version"], "0.14.0");
        assert_eq!(map["sha"], "abc1234");
        assert_eq!(map["auth"], "1");
        assert_eq!(map["gpu"], "1xRTX 4090");
        assert_eq!(map["queue"], "200");
        assert_eq!(map["proto"], "http");
    }

    #[test]
    fn build_txt_records_auth_off_and_clamps() {
        let txt = build_txt_records(&"v".repeat(300), "sha", false, "gpu", 0);
        let map: BTreeMap<_, _> = txt.into_iter().collect();
        assert_eq!(map["auth"], "0");
        assert!(map["version"].len() <= 200);
    }

    #[test]
    fn enabled_from_env_matrix() {
        // Guard the shared process env; these run serially within one test.
        let key = "MOLD_MDNS";
        // SAFETY: single-threaded test mutating env before reading it.
        unsafe {
            std::env::remove_var(key);
            assert!(enabled_from_env());
            for on in ["1", "true", "TRUE", "yes", "on", "anything"] {
                std::env::set_var(key, on);
                assert!(enabled_from_env(), "{on} should enable");
            }
            for off in ["0", "false", "FALSE", "no", "off", " off "] {
                std::env::set_var(key, off);
                assert!(!enabled_from_env(), "{off} should disable");
            }
            std::env::remove_var(key);
        }
    }

    #[test]
    fn from_service_parts_prefers_ipv4_for_url() {
        let v4 = IpAddr::V4(Ipv4Addr::new(192, 168, 1, 20));
        let v6 = IpAddr::V6(Ipv6Addr::LOCALHOST);
        let s = from_service_parts(
            "hal9000-7680._mold._tcp.local.",
            "hal9000.local.",
            7680,
            vec![v6, v4],
            vec![
                ("version".to_string(), "0.14.0".to_string()),
                ("auth".to_string(), "1".to_string()),
            ],
        );
        assert_eq!(s.name, "hal9000-7680");
        assert_eq!(s.host, "192.168.1.20");
        assert_eq!(s.url, "http://192.168.1.20:7680");
        assert_eq!(s.version.as_deref(), Some("0.14.0"));
        assert!(s.auth_required);
        // IPv4 sorts ahead of IPv6.
        assert_eq!(s.addresses.first().unwrap(), "192.168.1.20");
    }

    #[test]
    fn from_service_parts_excludes_loopback_when_routable_exists() {
        let loop4 = IpAddr::V4(Ipv4Addr::LOCALHOST);
        let lan = IpAddr::V4(Ipv4Addr::new(192, 168, 1, 142));
        let s = from_service_parts(
            "hal9000-7690._mold._tcp.local.",
            "hal9000.local.",
            7690,
            vec![loop4, lan],
            vec![],
        );
        // The connect URL must not point at loopback when a LAN address exists.
        assert_eq!(s.url, "http://192.168.1.142:7690");
        assert_eq!(s.host, "192.168.1.142");
        // And loopback is dropped from the address list entirely.
        assert!(!s.addresses.iter().any(|a| a == "127.0.0.1"));
        assert_eq!(s.addresses, vec!["192.168.1.142".to_string()]);
    }

    #[test]
    fn from_service_parts_keeps_loopback_when_only_address() {
        let loop4 = IpAddr::V4(Ipv4Addr::LOCALHOST);
        let s = from_service_parts(
            "self-7680._mold._tcp.local.",
            "self.local.",
            7680,
            vec![loop4],
            vec![],
        );
        // Loopback is the only thing we have — keep it rather than emit nothing.
        assert_eq!(s.addresses, vec!["127.0.0.1".to_string()]);
        assert_eq!(s.url, "http://127.0.0.1:7680");
    }

    #[test]
    fn from_service_parts_ipv6_only_brackets_url() {
        let v6 = IpAddr::V6(Ipv6Addr::new(0xfe80, 0, 0, 0, 0, 0, 0, 1));
        let s = from_service_parts(
            "box._mold._tcp.local.",
            "box.local.",
            8080,
            vec![v6],
            vec![],
        );
        assert_eq!(s.url, "http://[fe80::1]:8080");
        assert!(!s.auth_required);
        assert!(s.version.is_none());
    }

    #[test]
    fn from_service_parts_no_addresses_uses_hostname() {
        let s = from_service_parts("box._mold._tcp.local.", "box.local.", 7680, vec![], vec![]);
        // No addresses: fall back to the mDNS hostname (trailing dot stripped).
        assert_eq!(s.host, "box.local");
        assert_eq!(s.url, "http://box.local:7680");
        assert!(s.addresses.is_empty());
    }

    // Full register→browse round-trip. Ignored in CI: multicast loopback is
    // unreliable on shared/containerized runners. Run locally with
    // `cargo test -p mold-ai-server --features mdns -- --ignored`.
    #[test]
    #[ignore]
    fn register_then_discover_roundtrip() {
        let txt = build_txt_records("9.9.9", "deadbee", false, "cpu", 42);
        let guard = register(0, txt).expect("register");
        // Give the responder a moment, then browse.
        let found = discover(Duration::from_secs(3)).expect("discover");
        guard.shutdown();
        assert!(
            found.iter().any(|s| s.version.as_deref() == Some("9.9.9")),
            "expected to rediscover our own advertisement, got {found:?}"
        );
    }
}
