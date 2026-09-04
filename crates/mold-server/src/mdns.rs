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
//! and the network entry points [`register`], [`start_browser`], and
//! [`discover`] that talk to the `mdns-sd` daemon. Only those entry points
//! touch multicast, so the pure helpers carry the bulk of the test coverage.

use std::collections::BTreeMap;
use std::net::IpAddr;
use std::sync::Arc;
use std::time::{Duration, Instant};

use anyhow::{Context, Result};
use mdns_sd::{IfKind, ServiceDaemon, ServiceEvent, ServiceInfo};
use serde::Serialize;

/// DNS-SD service type advertised and browsed by mold.
pub const SERVICE_TYPE: &str = "_mold._tcp.local.";

/// Longest instance label we will emit. DNS labels cap at 63 bytes; we reserve
/// headroom for the `-{port}` suffix so the sanitised host never overflows.
const MAX_INSTANCE_BYTES: usize = 63;

/// Whether an address is an IPv6 link-local (`fe80::/10`). These carry a zone
/// id that mdns-sd drops, so they can't be used verbatim in a URL.
fn is_ipv6_link_local(ip: &IpAddr) -> bool {
    matches!(ip, IpAddr::V6(v6) if (v6.segments()[0] & 0xffc0) == 0xfe80)
}

/// Handle that keeps an advertisement alive. Dropping it (or calling
/// [`MdnsGuard::shutdown`]) unregisters the service and stops the daemon so the
/// server disappears from the network promptly on shutdown.
pub struct MdnsGuard {
    daemon: ServiceDaemon,
    fullname: String,
}

/// Handle for the server's long-lived DNS-SD browser. Resolved peers are kept
/// in a shared cache so HTTP requests never wait on multicast resolution.
pub struct MdnsBrowserGuard {
    daemon: ServiceDaemon,
    thread: Option<std::thread::JoinHandle<()>>,
    stop: Arc<std::sync::atomic::AtomicBool>,
}

/// How often the browse loop looks up from the channel to notice a stop
/// request. The browse receiver's sender belongs to the mDNS daemon, so a
/// blocking `recv()` is only woken by the daemon choosing to close it; on a
/// host where multicast never works that never happens and the join below
/// waits forever. Shutdown must be bounded by mold, not by the daemon.
const BROWSE_STOP_POLL: Duration = Duration::from_millis(250);

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

impl MdnsBrowserGuard {
    /// Stop browsing and join the cache updater thread. Best-effort on the
    /// server shutdown path, matching [`MdnsGuard::shutdown`].
    pub fn shutdown(mut self) {
        // Set BEFORE asking the daemon to stop: the join below is unbounded,
        // and the flag is the only wake-up that does not depend on the daemon
        // answering. `stop_browse`/`shutdown` remain the fast path — they
        // close the channel and the loop leaves immediately — but a daemon
        // that never answers now costs `BROWSE_STOP_POLL`, not the process.
        self.stop.store(true, std::sync::atomic::Ordering::SeqCst);
        let _ = self.daemon.stop_browse(SERVICE_TYPE);
        let _ = self.daemon.shutdown();
        if let Some(thread) = self.thread.take() {
            if thread.join().is_err() {
                tracing::warn!("mDNS: discovery browser thread panicked");
            }
        }
    }
}

/// A mold server discovered on the local network.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct DiscoveredServer {
    /// Instance label without the service-type suffix (e.g. `hal9000-7680`).
    pub name: String,
    /// Best host to connect to — a routable IPv4 when present, else a routable
    /// (non-link-local) IPv6, else the advertised `.local` hostname.
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
    /// Stable server-installation UUID from the `id` TXT record. `None` on
    /// older servers that don't advertise one (empty values are treated as
    /// absent).
    pub instance_id: Option<String>,
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

/// How a server's addresses should be announced, decided from its bind address.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Advertised {
    /// Wildcard or hostname bind — let `enable_addr_auto` announce every
    /// interface address (loopback is disabled separately).
    Auto,
    /// A concrete non-loopback bind — announce only this address, so discovery
    /// can never hand out an interface the listener isn't actually bound on.
    Fixed(IpAddr),
}

/// Decide how to advertise given the server's `--bind` address.
///
/// A wildcard bind (`0.0.0.0` / `::` / empty) fans out to every interface via
/// auto-addressing. A specific non-loopback IP is advertised verbatim so a
/// multi-homed host (LAN + VPN NIC) never announces an address the listener
/// isn't serving on. Loopback and hostname binds fall back to `Auto` (loopback
/// is already excluded by [`is_advertisable`] and the interface disable).
pub fn advertise_addr(bind: &str) -> Advertised {
    let host = bind.trim();
    if host.is_empty() || matches!(host, "0.0.0.0" | "::" | "[::]") {
        return Advertised::Auto;
    }
    match host
        .trim_start_matches('[')
        .trim_end_matches(']')
        .parse::<IpAddr>()
    {
        Ok(ip) if !ip.is_loopback() => Advertised::Fixed(ip),
        _ => Advertised::Auto,
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

/// Concise GPU summary for the `gpu` TXT record.
///
/// `[]` → `"cpu"`; homogeneous names → `"<n>x<name>"`; mixed names → a generic
/// `"<n> GPUs"` so a 1xA100 + 1x4090 host isn't advertised as `2xA100`.
pub fn gpu_summary(names: &[String]) -> String {
    match names {
        [] => "cpu".to_string(),
        [only] => format!("1x{only}"),
        [first, rest @ ..] if rest.iter().all(|n| n == first) => {
            format!("{}x{first}", names.len())
        }
        _ => format!("{} GPUs", names.len()),
    }
}

/// Assemble the TXT records advertised alongside the service.
///
/// Keys: `version`, `sha`, `auth` (`1`/`0`), `gpu`, `queue`, `proto`, `id`.
/// Values are clamped defensively — a single TXT entry must stay well under
/// 255 bytes.
pub fn build_txt_records(
    version: &str,
    git_sha: &str,
    auth_required: bool,
    gpu_summary: &str,
    queue_capacity: usize,
    instance_id: &str,
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
        ("id".to_string(), clamp(instance_id, 64)),
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

    // Connect-host preference: routable IPv4, then routable (non-link-local)
    // IPv6, then the advertised `.local` hostname. mdns-sd strips the zone id
    // from scoped IPv6, so a bare `[fe80::…]` URL is unroutable — we keep such
    // addresses in the list but never build a URL from them; the `.local`
    // hostname resolves via mDNS on the client instead.
    let host = if let Some(ip) = addresses.iter().find(|ip| ip.is_ipv4()) {
        ip.to_string()
    } else if let Some(ip) = addresses.iter().find(|ip| !is_ipv6_link_local(ip)) {
        format!("[{ip}]")
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
        instance_id: txt_map.get("id").filter(|v| !v.is_empty()).cloned(),
        txt: txt_map,
    }
}

/// Register a mold advertisement on the local network.
///
/// `txt` is the output of [`build_txt_records`]. A wildcard `bind` advertises
/// every non-loopback interface (via `enable_addr_auto`); a concrete
/// non-loopback `bind` advertises only that address so a multi-homed host never
/// announces an interface the listener isn't serving on ([`advertise_addr`]).
pub fn register(bind: &str, port: u16, txt: Vec<(String, String)>) -> Result<MdnsGuard> {
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
    // The advertised hostname must end in `.local.`.
    let host_label = {
        let base = raw_host.split('.').next().unwrap_or("mold");
        let base = if base.is_empty() { "mold" } else { base };
        format!("{base}.local.")
    };

    let props: Vec<(&str, &str)> = txt.iter().map(|(k, v)| (k.as_str(), v.as_str())).collect();

    // A specific-IP bind is announced verbatim; a wildcard/hostname bind uses
    // auto-addressing over every (non-loopback) interface.
    let fixed_addr = match advertise_addr(bind) {
        Advertised::Fixed(ip) => Some(ip.to_string()),
        Advertised::Auto => None,
    };
    let service = ServiceInfo::new(
        SERVICE_TYPE,
        &instance,
        &host_label,
        fixed_addr.as_deref().unwrap_or(""),
        port,
        &props[..],
    )
    .context("failed to build mDNS service info")?;
    let service = if fixed_addr.is_some() {
        service
    } else {
        service.enable_addr_auto()
    };

    let fullname = service.get_fullname().to_string();
    daemon
        .register(service)
        .context("failed to register mDNS service")?;
    tracing::info!(fullname = %fullname, port, "mDNS: advertising service");

    Ok(MdnsGuard { daemon, fullname })
}

/// Start the server's long-lived DNS-SD browser and cache updater.
///
/// The background thread tracks resolved and removed services by fullname.
/// If browsing stops, it marks discovery unavailable and clears cached peers.
/// Drain browse events into the shared peer cache until the channel closes,
/// the daemon reports `SearchStopped`, or `stop` is set. Split out of the
/// thread body so a test can prove the loop leaves on the flag alone, with a
/// live sender and no events — the shape a daemon that never closes its
/// channel presents on shutdown.
fn browse_until_stopped(
    receiver: &mdns_sd::Receiver<ServiceEvent>,
    discovery: &crate::state::DiscoveryState,
    own_instance_id: &str,
    stop: &std::sync::atomic::AtomicBool,
) {
    let mut found: BTreeMap<String, mold_core::DiscoveryPeer> = BTreeMap::new();
    loop {
        if stop.load(std::sync::atomic::Ordering::SeqCst) {
            return;
        }
        // `is_disconnected` rather than matching flume's error type: the
        // receiver reaches us through mdns-sd's re-export, and mold does not
        // otherwise depend on flume.
        let event = match receiver.recv_timeout(BROWSE_STOP_POLL) {
            Ok(event) => event,
            Err(_) if receiver.is_disconnected() && receiver.is_empty() => return,
            Err(_) => continue,
        };
        match event {
            ServiceEvent::ServiceResolved(info) => {
                let addresses: Vec<IpAddr> =
                    info.addresses.iter().map(|a| a.to_ip_addr()).collect();
                let txt = info
                    .txt_properties
                    .iter()
                    .map(|p| (p.key().to_string(), p.val_str().to_string()))
                    .collect();
                let server =
                    from_service_parts(&info.fullname, &info.host, info.port, addresses, txt);
                let is_this_machine = server.instance_id.as_deref() == Some(own_instance_id);
                found.insert(
                    info.fullname.clone(),
                    mold_core::DiscoveryPeer {
                        name: server.name,
                        url: server.url,
                        host: server.host,
                        port: server.port,
                        version: server.version,
                        auth_required: server.auth_required,
                        instance_id: server.instance_id,
                        is_this_machine,
                    },
                );
            }
            ServiceEvent::ServiceRemoved(_, fullname) => {
                found.remove(&fullname);
            }
            ServiceEvent::SearchStopped(_) => break,
            _ => continue,
        }

        let peers = found.values().cloned().collect();
        *discovery
            .peers
            .write()
            .unwrap_or_else(|poisoned| poisoned.into_inner()) = peers;
    }
}

pub fn start_browser(
    discovery: Arc<crate::state::DiscoveryState>,
    own_instance_id: Arc<String>,
) -> Result<MdnsBrowserGuard> {
    let daemon = ServiceDaemon::new().context("failed to start mDNS discovery daemon")?;
    let receiver = daemon
        .browse(SERVICE_TYPE)
        .context("failed to start mDNS discovery browse")?;

    discovery.set_can_browse(true);
    let stop = Arc::new(std::sync::atomic::AtomicBool::new(false));
    let thread_discovery = discovery.clone();
    let thread_instance_id = own_instance_id.clone();
    let thread_stop = stop.clone();
    let thread = std::thread::Builder::new()
        .name("mold-mdns-browser".to_string())
        .spawn(move || {
            browse_until_stopped(
                &receiver,
                &thread_discovery,
                &thread_instance_id,
                &thread_stop,
            );
            thread_discovery.mark_unavailable();
        });
    let thread = match thread {
        Ok(thread) => thread,
        Err(error) => {
            discovery.mark_unavailable();
            return Err(error).context("failed to spawn mDNS discovery browser thread");
        }
    };

    tracing::info!(
        service_type = SERVICE_TYPE,
        "mDNS: browsing for peer servers"
    );
    Ok(MdnsBrowserGuard {
        daemon,
        thread: Some(thread),
        stop,
    })
}

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
    use std::sync::atomic::{AtomicBool, Ordering};

    /// The browse channel's sender belongs to the mDNS daemon. On a host where
    /// multicast never works the daemon can fail to close it during shutdown,
    /// and the old blocking `recv()` then parked the browser thread forever —
    /// `MdnsBrowserGuard::shutdown`'s join never returned, so an embedded
    /// server never finished stopping. Prove the loop leaves on the flag with
    /// a LIVE sender and no events.
    #[test]
    fn the_browse_loop_stops_on_the_flag_even_when_the_channel_stays_open() {
        let (sender, receiver) = flume_channel();
        let discovery = crate::state::DiscoveryState::default();
        let stop = Arc::new(AtomicBool::new(false));

        let loop_stop = stop.clone();
        let worker = std::thread::spawn(move || {
            browse_until_stopped(&receiver, &discovery, "self-id", &loop_stop);
        });

        // Nothing is ever sent and the sender stays alive for the whole test.
        std::thread::sleep(BROWSE_STOP_POLL / 2);
        assert!(!worker.is_finished(), "the loop waits while it may browse");
        stop.store(true, Ordering::SeqCst);

        let deadline = Instant::now() + BROWSE_STOP_POLL * 8;
        while !worker.is_finished() && Instant::now() < deadline {
            std::thread::sleep(Duration::from_millis(10));
        }
        assert!(
            worker.is_finished(),
            "the browse loop must leave within a poll of the stop flag"
        );
        worker.join().expect("browse loop thread");
        drop(sender);
    }

    /// A daemon that DOES close the channel still ends the loop immediately;
    /// the flag is the fallback, not the only exit.
    #[test]
    fn the_browse_loop_stops_when_the_daemon_closes_the_channel() {
        let (sender, receiver) = flume_channel();
        let discovery = crate::state::DiscoveryState::default();
        let stop = Arc::new(AtomicBool::new(false));

        let worker = std::thread::spawn(move || {
            browse_until_stopped(&receiver, &discovery, "self-id", &stop);
        });
        drop(sender);

        let deadline = Instant::now() + BROWSE_STOP_POLL * 8;
        while !worker.is_finished() && Instant::now() < deadline {
            std::thread::sleep(Duration::from_millis(10));
        }
        assert!(worker.is_finished(), "a closed channel ends the loop");
        worker.join().expect("browse loop thread");
    }

    /// `mdns_sd::Receiver` IS `flume::Receiver` (mdns-sd re-exports it), so a
    /// test channel comes from flume directly — pinned to the same 0.12 line
    /// mdns-sd depends on, or the two `Receiver` types would not unify.
    fn flume_channel() -> (flume::Sender<ServiceEvent>, flume::Receiver<ServiceEvent>) {
        flume::bounded(8)
    }

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
    fn advertise_addr_wildcard_is_auto() {
        assert_eq!(advertise_addr("0.0.0.0"), Advertised::Auto);
        assert_eq!(advertise_addr("::"), Advertised::Auto);
        assert_eq!(advertise_addr("[::]"), Advertised::Auto);
        assert_eq!(advertise_addr(""), Advertised::Auto);
        // A hostname bind can't be resolved to one address here — auto.
        assert_eq!(advertise_addr("hal9000"), Advertised::Auto);
    }

    #[test]
    fn advertise_addr_specific_ip_is_fixed() {
        assert_eq!(
            advertise_addr("192.168.1.5"),
            Advertised::Fixed(IpAddr::V4(Ipv4Addr::new(192, 168, 1, 5)))
        );
        assert_eq!(
            advertise_addr("[fe80::1]"),
            Advertised::Fixed(IpAddr::V6(Ipv6Addr::new(0xfe80, 0, 0, 0, 0, 0, 0, 1)))
        );
    }

    #[test]
    fn advertise_addr_loopback_falls_back_to_auto() {
        // Loopback binds are filtered by is_advertisable before register, but
        // guard against a stray call handing us a self-pointing address.
        assert_eq!(advertise_addr("127.0.0.1"), Advertised::Auto);
        assert_eq!(advertise_addr("::1"), Advertised::Auto);
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
    fn gpu_summary_shapes() {
        assert_eq!(gpu_summary(&[]), "cpu");
        assert_eq!(gpu_summary(&["RTX 4090".into()]), "1xRTX 4090");
        assert_eq!(
            gpu_summary(&["RTX 4090".into(), "RTX 4090".into()]),
            "2xRTX 4090"
        );
        // Mixed GPUs fall back to a generic count instead of the first name.
        assert_eq!(gpu_summary(&["A100".into(), "RTX 4090".into()]), "2 GPUs");
    }

    #[test]
    fn build_txt_records_shape() {
        let txt = build_txt_records(
            "0.14.0",
            "abc1234",
            true,
            "1xRTX 4090",
            200,
            "0b5c1a4e-9f3d-4c8a-b2e7-6d1f0a9c3e58",
        );
        let map: BTreeMap<_, _> = txt.into_iter().collect();
        assert_eq!(map["version"], "0.14.0");
        assert_eq!(map["sha"], "abc1234");
        assert_eq!(map["auth"], "1");
        assert_eq!(map["gpu"], "1xRTX 4090");
        assert_eq!(map["queue"], "200");
        assert_eq!(map["proto"], "http");
        assert_eq!(map["id"], "0b5c1a4e-9f3d-4c8a-b2e7-6d1f0a9c3e58");
    }

    #[test]
    fn build_txt_records_auth_off_and_clamps() {
        let txt = build_txt_records(&"v".repeat(300), "sha", false, "gpu", 0, &"i".repeat(100));
        let map: BTreeMap<_, _> = txt.into_iter().collect();
        assert_eq!(map["auth"], "0");
        assert!(map["version"].len() <= 200);
        assert!(map["id"].len() <= 64);
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
    fn from_service_parts_routable_ipv6_only_brackets_url() {
        // A global-unicast IPv6 (2001:db8::/32 doc range) has no scope id, so
        // it is safe to use verbatim in a bracketed URL.
        let v6 = IpAddr::V6(Ipv6Addr::new(0x2001, 0xdb8, 0, 0, 0, 0, 0, 1));
        let s = from_service_parts(
            "box._mold._tcp.local.",
            "box.local.",
            8080,
            vec![v6],
            vec![],
        );
        assert_eq!(s.url, "http://[2001:db8::1]:8080");
        assert!(!s.auth_required);
        assert!(s.version.is_none());
    }

    #[test]
    fn from_service_parts_link_local_v6_only_uses_hostname() {
        // Link-local IPv6 loses its zone id → unroutable as a bare URL. Fall
        // back to the mDNS `.local` hostname, but keep the address in the list.
        let ll = IpAddr::V6(Ipv6Addr::new(0xfe80, 0, 0, 0, 0, 0, 0, 1));
        let s = from_service_parts(
            "box._mold._tcp.local.",
            "box.local.",
            8080,
            vec![ll],
            vec![],
        );
        assert_eq!(s.url, "http://box.local:8080");
        assert_eq!(s.host, "box.local");
        assert_eq!(s.addresses, vec!["fe80::1".to_string()]);
    }

    #[test]
    fn from_service_parts_ipv4_beats_link_local_v6() {
        let ll = IpAddr::V6(Ipv6Addr::new(0xfe80, 0, 0, 0, 0, 0, 0, 1));
        let v4 = IpAddr::V4(Ipv4Addr::new(192, 168, 1, 5));
        let s = from_service_parts(
            "box-7680._mold._tcp.local.",
            "box.local.",
            7680,
            vec![ll, v4],
            vec![],
        );
        assert_eq!(s.url, "http://192.168.1.5:7680");
        assert_eq!(s.host, "192.168.1.5");
    }

    #[test]
    fn from_service_parts_parses_instance_id() {
        let s = from_service_parts(
            "box._mold._tcp.local.",
            "box.local.",
            7680,
            vec![],
            vec![(
                "id".to_string(),
                "0b5c1a4e-9f3d-4c8a-b2e7-6d1f0a9c3e58".to_string(),
            )],
        );
        assert_eq!(
            s.instance_id.as_deref(),
            Some("0b5c1a4e-9f3d-4c8a-b2e7-6d1f0a9c3e58")
        );
    }

    #[test]
    fn from_service_parts_tolerates_missing_or_empty_instance_id() {
        // Older servers don't advertise `id` at all.
        let s = from_service_parts("box._mold._tcp.local.", "box.local.", 7680, vec![], vec![]);
        assert_eq!(s.instance_id, None);
        // An empty value is treated as absent, not as an empty identity.
        let s = from_service_parts(
            "box._mold._tcp.local.",
            "box.local.",
            7680,
            vec![],
            vec![("id".to_string(), String::new())],
        );
        assert_eq!(s.instance_id, None);
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
        let txt = build_txt_records("9.9.9", "deadbee", false, "cpu", 42, "test-instance-id");
        let guard = register("0.0.0.0", 0, txt).expect("register");
        // Give the responder a moment, then browse.
        let found = discover(Duration::from_secs(3)).expect("discover");
        guard.shutdown();
        assert!(
            found.iter().any(|s| s.version.as_deref() == Some("9.9.9")),
            "expected to rediscover our own advertisement, got {found:?}"
        );
    }
}
