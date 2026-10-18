//! `mold host` — pair with hosts and manage the saved hosts `--host` and
//! `MOLD_HOST` accept by name.
//!
//! The registry is `mold_secure_client::hosts`, which reads the credential
//! profiles pairing writes. `add` and `invite` are the only verbs that talk
//! to a host; `list`, `use` and `rm` never touch the network.
use anyhow::{bail, Context, Result};
use clap_complete::engine::CompletionCandidate;
use mold_secure_client::hosts::{self, SavedHost};
use mold_secure_client::ConnectionMode;
use std::io::{BufRead, IsTerminal, Write};
use std::path::{Path, PathBuf};
use zeroize::Zeroizing;

use crate::commands::pairing::{self, SessionReply};
use crate::theme;
use crate::HostAction;

pub async fn run(action: HostAction) -> Result<()> {
    // Running a host command is the one time derived names are written
    // down, so they stop depending on what else is saved.
    hosts::persist_derived_names()?;
    match action {
        HostAction::Add {
            input,
            name,
            api_key_file,
            api_key_stdin,
            regular,
            default,
            device_name,
        } => {
            add(AddOptions {
                input,
                name,
                api_key_file,
                api_key_stdin,
                regular,
                default,
                device_name,
            })
            .await
        }
        HostAction::Invite {
            regular,
            direct_url,
            qr,
            output,
            target,
            relay_url,
            api_key_file,
        } => {
            invite(InviteOptions {
                target,
                mode: if regular {
                    ConnectionMode::Regular
                } else {
                    ConnectionMode::Secure
                },
                direct_url,
                relay_url,
                api_key_file,
                qr,
                output,
            })
            .await
        }
        HostAction::List { json } => list(json),
        HostAction::Use { name, clear } => {
            if clear {
                let cleared = hosts::clear_default()?;
                println!(
                    "{} {}; commands target {} unless --host or MOLD_HOST names a host.",
                    theme::icon_ok(),
                    if cleared {
                        "Default host cleared"
                    } else {
                        "No default host was set"
                    },
                    hosts::LOCAL_HOST
                );
                return Ok(());
            }
            let name = name.expect("clap requires NAME unless --clear");
            let host = hosts::set_default(&name)?;
            println!(
                "{} Default host is now '{}' ({}).",
                theme::icon_ok(),
                host.name,
                host.origin
            );
            if std::env::var("MOLD_HOST").is_ok_and(|v| !v.trim().is_empty()) {
                eprintln!(
                    "{} MOLD_HOST is set in this shell and still takes precedence over the default.",
                    theme::icon_warn()
                );
            }
            Ok(())
        }
        HostAction::Rm { name } => {
            let host = hosts::remove(&name)?;
            println!(
                "{} Removed '{}' ({}) and its saved credentials. Revoke this device on the host to end its grant there too.",
                theme::icon_ok(),
                host.name,
                host.origin
            );
            Ok(())
        }
    }
}

fn list(json: bool) -> Result<()> {
    let saved = hosts::list()?;
    let default = hosts::default()?.map(|h| h.origin);
    if json {
        let rows: Vec<serde_json::Value> = saved
            .iter()
            .map(|h| row_json(h, default.as_deref() == Some(h.origin.as_str())))
            .collect();
        println!("{}", serde_json::to_string_pretty(&rows)?);
        return Ok(());
    }
    if saved.is_empty() {
        println!(
            "No saved hosts. On the host run `mold host invite`, then here `mold host add` and paste its link (or `mold host add URL` with an API key)."
        );
        return Ok(());
    }
    print!("{}", render_table(&saved, default.as_deref()));
    Ok(())
}

fn row_json(host: &SavedHost, is_default: bool) -> serde_json::Value {
    serde_json::json!({
        "name": host.name,
        "mode": host.mode.as_str(),
        "origin": host.origin,
        "default": is_default,
        "named": host.named,
        "ambiguous": host.ambiguous,
        "instance_id": host.instance_id,
        "hostname": host.hostname,
        "endpoints": host.endpoints,
    })
}

fn render_table(saved: &[SavedHost], default: Option<&str>) -> String {
    let label = |h: &SavedHost| {
        if h.ambiguous {
            format!("{} (ambiguous)", h.name)
        } else {
            h.name.clone()
        }
    };
    let name_width = saved
        .iter()
        .map(|h| label(h).len())
        .max()
        .unwrap_or(0)
        .max(4);
    let origin_width = saved
        .iter()
        .map(|h| h.origin.len())
        .max()
        .unwrap_or(0)
        .max(6);
    let mut out = format!(
        "{:<name_width$}  {:<7}  {:<origin_width$}  DEFAULT\n",
        "NAME", "MODE", "ORIGIN"
    );
    for host in saved {
        let marker = if default == Some(host.origin.as_str()) {
            "*"
        } else {
            ""
        };
        out.push_str(
            format!(
                "{:<name_width$}  {:<7}  {:<origin_width$}  {marker}",
                label(host),
                host.mode.as_str(),
                host.origin
            )
            .trim_end(),
        );
        out.push('\n');
    }
    if saved.iter().any(|h| h.ambiguous) {
        out.push_str(
            "\nAn ambiguous name belongs to more than one host and resolves to an error; re-pair one of them so it gets its own name.\n",
        );
    }
    out
}

pub struct AddOptions {
    pub input: Option<String>,
    pub name: Option<String>,
    pub api_key_file: Option<PathBuf>,
    pub api_key_stdin: bool,
    pub regular: bool,
    pub default: bool,
    pub device_name: Option<String>,
}

/// `mold host add`: pair from a pairing link (or a file holding one, or the
/// JSON invitation `mold host invite --output` writes), or from a URL plus
/// an API key. An operator key is EXCHANGED for this client's own grant and
/// never stored; an app/device key is saved as a Regular credential.
pub async fn add(options: AddOptions) -> Result<()> {
    let input = match options.input.as_deref() {
        Some("-") => read_line(&mut std::io::stdin().lock())?,
        Some(input) => Zeroizing::new(input.trim().to_string()),
        None if std::io::stdin().is_terminal() => prompt("Pairing link or host URL: ", false)?,
        None => bail!("pass a pairing link or host URL, or `-` to read one from stdin"),
    };
    anyhow::ensure!(!input.is_empty(), "pass a pairing link or host URL");
    let device = options
        .device_name
        .clone()
        .unwrap_or_else(default_device_name);
    let invitation = match invitation_input(&input)? {
        Some(value) => value,
        None => return add_with_key(&input, &options, &device).await,
    };
    let added = pair_invitation(invitation, options.name.as_deref(), &device).await?;
    finish(added, options.default).await
}

/// A pairing link, a file holding one, or the JSON invitation; `None` for
/// anything else (a plain host URL).
fn invitation_input(input: &str) -> Result<Option<serde_json::Value>> {
    let text = match file_argument(input) {
        Some(path) => {
            let bytes = pairing::read_private_explained(&path, "invitation file")?;
            Zeroizing::new(
                String::from_utf8(bytes.to_vec()).context("invitation file is not text")?,
            )
        }
        None => Zeroizing::new(input.to_string()),
    };
    let looks_like_invitation = text.trim_start().starts_with('{')
        || text.contains('#')
        || text.trim_start().starts_with("moldstudio:");
    if !looks_like_invitation {
        anyhow::ensure!(
            file_argument(input).is_none(),
            "{input} does not hold a Mold pairing link or invitation"
        );
        return Ok(None);
    }
    Ok(Some(mold_secure_client::parse_pairing_link(&text)?))
}

/// The file an argument names: only a spelled path (absolute, `./`, `~/`,
/// or one with a separator) and never a URL, so `mold host add workstation`
/// is the host `workstation`, never a file of that name here.
fn file_argument(input: &str) -> Option<PathBuf> {
    if input.contains("://") || input.starts_with('{') || input.starts_with("moldstudio:") {
        return None;
    }
    let spelled = input.starts_with('.')
        || input.starts_with('~')
        || input.contains(std::path::MAIN_SEPARATOR)
        || input.contains('/')
        || Path::new(input).is_absolute();
    if !spelled {
        return None;
    }
    Some(match input.strip_prefix("~/") {
        Some(rest) => std::env::var_os("HOME")
            .map(PathBuf::from)
            .unwrap_or_default()
            .join(rest),
        None => PathBuf::from(input),
    })
}

/// A pairing that was just saved.
pub(crate) struct Added {
    name: String,
    origin: String,
    mode: ConnectionMode,
}

/// Claim an invitation value (Secure or Regular by its mode) and save the
/// resulting device grant under its name. Shared by `host add` and the
/// `mold pair --invite-file` alias.
pub(crate) async fn pair_invitation(
    invitation: serde_json::Value,
    requested: Option<&str>,
    device: &str,
) -> Result<Added> {
    pair_invitation_with(
        invitation,
        requested,
        device,
        &Transport::default().client()?,
    )
    .await
}

/// [`pair_invitation`] with the Regular claim sent through `client` (pinned
/// by the caller). A Secure claim travels inside Noise to the invitation's
/// own endpoints, authenticated by the pinned host key.
async fn pair_invitation_with(
    invitation: serde_json::Value,
    requested: Option<&str>,
    device: &str,
    client: &reqwest::Client,
) -> Result<Added> {
    let text = |key: &str| invitation[key].as_str().map(str::to_owned);
    let reported = text("name")
        .or_else(|| text("hostname"))
        .unwrap_or_default();
    let instance = text("instance_id").filter(|v| !v.is_empty());
    let base = text("base_url").context("invitation has no base_url")?;
    if invitation["mode"] == "regular" {
        let origin = mold_secure_client::regular_origin(&base)?;
        // Decide the name before the claim spends the one-use invitation.
        let (name, superseded) = choose_name(
            requested,
            &reported,
            &origin,
            instance.as_deref(),
            ConnectionMode::Regular,
        )?;
        let mut host = mold_secure_client::pair_regular_with(invitation, device, client).await?;
        host.name = Some(name.clone());
        mold_secure_client::save_regular_host(&host)?;
        retire_superseded(superseded, &host.origin, ConnectionMode::Regular)?;
        return Ok(Added {
            name,
            origin: host.origin.clone(),
            mode: ConnectionMode::Regular,
        });
    }
    let invite: mold_secure_client::PairingInvite = serde_json::from_value(invitation)?;
    let (name, superseded) = choose_name(
        requested,
        &reported,
        &invite.origin,
        instance.as_deref(),
        ConnectionMode::Secure,
    )?;
    let mut host = mold_secure_client::pair(invite, device).await?;
    host.name = Some(name.clone());
    mold_secure_client::save_pairing(&host)?;
    retire_superseded(superseded, &host.origin, ConnectionMode::Secure)?;
    mold_secure_client::select_connection_mode(&host.origin, ConnectionMode::Secure)?;
    Ok(Added {
        name,
        origin: host.origin.clone(),
        mode: ConnectionMode::Secure,
    })
}

/// The saved name: an explicit `--name` (free, or the name this same
/// server already has), else the host's own (see `hosts::name_for_pairing`).
fn choose_name(
    requested: Option<&str>,
    reported: &str,
    origin: &str,
    instance: Option<&str>,
    mode: ConnectionMode,
) -> Result<(String, Option<SavedHost>)> {
    let (automatic, superseded) = hosts::name_for_pairing(reported, origin, instance, mode)?;
    let Some(requested) = requested else {
        return Ok((automatic, superseded));
    };
    let requested = requested.trim().to_ascii_lowercase();
    if superseded.as_ref().is_some_and(|old| old.name == requested) {
        return Ok((requested, superseded));
    }
    Ok((hosts::require_free_name(&requested, origin)?, superseded))
}

/// After re-pairing a server that was saved under another origin, delete the
/// old entry (its name and default already moved to the new one).
fn retire_superseded(
    superseded: Option<SavedHost>,
    origin: &str,
    mode: ConnectionMode,
) -> Result<()> {
    if let Some(old) = superseded {
        hosts::retire(&old, origin, mode)?;
        eprintln!(
            "Replaced the earlier pairing of '{}' at {}.",
            old.name, old.origin
        );
    }
    Ok(())
}

/// What a host's answer to an API-key pairing request means.
#[derive(Debug, PartialEq, Eq)]
enum KeyOutcome {
    /// The key is an operator key: claim the invitation it bought.
    Claim,
    /// An app/device key: save it as a Regular credential.
    SaveKey,
    /// Authentication is off: save a keyless Regular connection.
    SaveKeyless,
}

fn key_outcome(status: u16, code: &str, regular: bool) -> Result<KeyOutcome> {
    Ok(match (status, code) {
        (200..=299, _) => KeyOutcome::Claim,
        (403, "PAIRING_OPERATOR_REQUIRED") => KeyOutcome::SaveKey,
        (503, "PAIRING_UNAVAILABLE") => KeyOutcome::SaveKeyless,
        (401, _) => bail!("the host rejected that API key"),
        (403, "REGULAR_ACCESS_DISABLED") => bail!(
            "the host has Regular direct access turned off; enable it there (`mold serve --allow-direct-api`) or pair with a link: run `mold host invite` on the host and `mold host add` here"
        ),
        _ => bail!(
            "the host could not start pairing (HTTP {status}{}){}; run `mold host invite` on the host and paste its link into `mold host add`",
            if code.is_empty() {
                String::new()
            } else {
                format!(" {code}")
            },
            if regular {
                ""
            } else {
                " — a remote host may also need `--regular`"
            }
        ),
    })
}

/// What the plain-HTTP rule decided for one `host add`: whether to warn,
/// and the addresses a verified tailnet name resolved to. Every request the
/// run makes is pinned to those, so the name is never resolved again
/// between verification and use.
#[derive(Debug, Default, PartialEq, Eq)]
struct Transport {
    warn: bool,
    pin: Option<(String, Vec<std::net::SocketAddr>)>,
}

impl Transport {
    /// A pairing/probe client: no proxy and no redirects, like the session
    /// request, so credentials only ever reach the origin named, and pinned
    /// to the verified addresses when there are any.
    fn client(&self) -> Result<reqwest::Client> {
        let mut builder = reqwest::Client::builder()
            .no_proxy()
            .redirect(reqwest::redirect::Policy::none())
            .timeout(std::time::Duration::from_secs(10));
        if let Some((host, addrs)) = &self.pin {
            builder = builder.resolve_to_addrs(host, addrs);
        }
        Ok(builder.build()?)
    }
}

type IpSet = std::collections::HashSet<std::net::IpAddr>;

/// Where the plain-HTTP rule looks things up: a resolver, Tailscale's peer
/// list, and the cheap address-range pre-filter. Injected so tests use no
/// real DNS or Tailscale.
struct Net<'a> {
    resolve: &'a dyn Fn(&str) -> Option<Vec<std::net::IpAddr>>,
    peers: &'a dyn Fn() -> Option<IpSet>,
    prefilter: fn(&std::net::IpAddr) -> bool,
}

impl Net<'static> {
    fn system() -> Self {
        Net {
            resolve: &resolve_bounded,
            peers: &tailnet_peer_ips,
            prefilter: |ip| mold_secure_client::hosts::is_tailscale_address(&ip.to_string()),
        }
    }
}

/// Whether pairing with `origin` may go ahead, and whether to warn: https,
/// loopback and a tailnet peer Tailscale confirms are private paths; plain
/// HTTP to anything else is refused for Secure (the invitation would be
/// readable on the network) and allowed, with a warning, for an explicit
/// Regular direct connection.
fn plain_http_policy_with(origin: &str, regular: bool, net: &Net) -> Result<Transport> {
    let Ok(url) = reqwest::Url::parse(origin) else {
        return Ok(Transport::default());
    };
    let machine = url.host_str().unwrap_or_default();
    if url.scheme() != "http" || crate::control::is_loopback_host(origin) {
        return Ok(Transport::default());
    }
    if let Some(verified) = verified_tailnet_peer(machine, net) {
        let port = url.port_or_known_default().unwrap_or(80);
        let pin = (!verified.is_empty()).then(|| {
            (
                machine.to_string(),
                verified
                    .into_iter()
                    .map(|ip| std::net::SocketAddr::new(ip, port))
                    .collect(),
            )
        });
        return Ok(Transport { warn: false, pin });
    }
    anyhow::ensure!(
        regular,
        "refusing to start a Secure pairing over plain HTTP to {origin}: the invitation would cross the network unencrypted. Use the host's https:// address, add --regular for a direct HTTP connection, or run `mold host invite` on the host and pair with its link"
    );
    Ok(Transport {
        warn: true,
        pin: None,
    })
}

/// Whether Tailscale itself confirms `machine` is a tailnet peer: the
/// literal address, or EVERY address the name resolves to, must be in the
/// IP lists `tailscale status` reports. The address ranges are only a cheap
/// pre-filter (100.64.0.0/10 is also carrier-grade NAT); a missing or
/// failing CLI verifies nothing. Returns the verified addresses of a name
/// (empty for a literal address, which needs no pinning).
fn verified_tailnet_peer(machine: &str, net: &Net) -> Option<Vec<std::net::IpAddr>> {
    let bare = machine.trim_matches(['[', ']']);
    let (candidates, literal) = match bare.parse::<std::net::IpAddr>() {
        Ok(ip) => (vec![ip], true),
        Err(_) => match (net.resolve)(&machine.trim_end_matches('.').to_ascii_lowercase()) {
            Some(ips) if !ips.is_empty() => (ips, false),
            _ => return None,
        },
    };
    if !candidates.iter().all(net.prefilter) {
        return None;
    }
    let known = (net.peers)()?;
    if !candidates.iter().all(|ip| known.contains(ip)) {
        return None;
    }
    Some(if literal { Vec::new() } else { candidates })
}

/// Every Tailscale IP of this machine and its peers, from
/// `tailscale status --json` (bounded to three seconds). `None` when the
/// CLI is missing, slow, fails, or prints something unreadable.
fn tailnet_peer_ips() -> Option<IpSet> {
    let cli = tailscale_cli()?;
    let output = run_bounded(
        &cli,
        &["status", "--json"],
        std::time::Duration::from_secs(3),
    )?;
    parse_tailnet_ips(&output)
}

/// The first `tailscale` on PATH, else the macOS app's bundled CLI.
fn tailscale_cli() -> Option<PathBuf> {
    let on_path = std::env::var_os("PATH").and_then(|path| {
        std::env::split_paths(&path)
            .map(|dir| dir.join("tailscale"))
            .find(|candidate| candidate.is_file())
    });
    on_path.or_else(|| {
        let app = PathBuf::from("/Applications/Tailscale.app/Contents/MacOS/Tailscale");
        (cfg!(target_os = "macos") && app.is_file()).then_some(app)
    })
}

/// Only `Self.TailscaleIPs` and each `Peer.*.TailscaleIPs` are read; the
/// rest of the status (names, keys, routes) is ignored and never logged.
fn parse_tailnet_ips(json: &[u8]) -> Option<IpSet> {
    let status: serde_json::Value = serde_json::from_slice(json).ok()?;
    let ips_of = |node: &serde_json::Value| -> Vec<std::net::IpAddr> {
        node["TailscaleIPs"]
            .as_array()
            .map(|list| {
                list.iter()
                    .filter_map(|ip| ip.as_str()?.parse().ok())
                    .collect()
            })
            .unwrap_or_default()
    };
    let mut ips: IpSet = ips_of(&status["Self"]).into_iter().collect();
    if let Some(peers) = status["Peer"].as_object() {
        for peer in peers.values() {
            ips.extend(ips_of(peer));
        }
    }
    Some(ips)
}

/// Run `program args` and return its stdout, or `None` when it cannot
/// start, exits non-zero, or outlives `timeout` (then it is killed).
fn run_bounded(program: &Path, args: &[&str], timeout: std::time::Duration) -> Option<Vec<u8>> {
    use std::io::Read;
    let mut child = std::process::Command::new(program)
        .args(args)
        .stdin(std::process::Stdio::null())
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::null())
        .spawn()
        .ok()?;
    let mut stdout = child.stdout.take()?;
    // Drain concurrently so a large status cannot fill the pipe and stall.
    let reader = std::thread::spawn(move || {
        let mut bytes = Vec::new();
        stdout.read_to_end(&mut bytes).map(|_| bytes)
    });
    let deadline = std::time::Instant::now() + timeout;
    let status = loop {
        match child.try_wait() {
            Ok(Some(status)) => break status,
            Ok(None) if std::time::Instant::now() < deadline => {
                std::thread::sleep(std::time::Duration::from_millis(20))
            }
            _ => {
                let _ = child.kill();
                let _ = child.wait();
                return None;
            }
        }
    };
    let bytes = reader.join().ok()?.ok()?;
    status.success().then_some(bytes)
}

/// The system resolver, given at most two seconds; `None` on failure or
/// timeout.
fn resolve_bounded(name: &str) -> Option<Vec<std::net::IpAddr>> {
    use std::net::ToSocketAddrs;
    let (sender, receiver) = std::sync::mpsc::channel();
    let name = name.to_string();
    std::thread::spawn(move || {
        let ips = (name.as_str(), 0)
            .to_socket_addrs()
            .ok()
            .map(|addrs| addrs.map(|addr| addr.ip()).collect::<Vec<_>>());
        let _ = sender.send(ips);
    });
    receiver
        .recv_timeout(std::time::Duration::from_secs(2))
        .ok()
        .flatten()
}

async fn add_with_key(input: &str, options: &AddOptions, device: &str) -> Result<()> {
    let base = mold_secure_client::hosts::normalize_address(input);
    let origin = mold_secure_client::regular_origin(&base).with_context(|| {
        format!("{input} is neither a pairing link nor a credential-free http(s) host URL")
    })?;
    // Settle everything that can be refused locally before the host is asked.
    if let Some(name) = options.name.as_deref() {
        let name = name.trim().to_ascii_lowercase();
        hosts::validate_name(&name)?;
    }
    add_with_key_using(origin, options, device, &Net::system()).await
}

async fn add_with_key_using(
    origin: String,
    options: &AddOptions,
    device: &str,
    net: &Net<'_>,
) -> Result<()> {
    // Read the key first: then the only gap after verifying the route is
    // the requests themselves, which are pinned to what was verified.
    let key = api_key(options)?;
    let transport = plain_http_policy_with(&origin, options.regular, net)?;
    let client = transport.client()?;
    if transport.warn {
        eprintln!(
            "{} the key and the invitation it buys travel unencrypted to {origin}; prefer https, or `mold host invite` on the host",
            theme::icon_warn()
        );
    }
    let regular_direct = options.regular.then_some(origin.as_str());
    let reply = pairing::request_session_with(&client, &origin, &key, regular_direct)
        .await
        .map_err(|error| {
            error.context(
                "run `mold host invite` on the host and paste its link into `mold host add`",
            )
        })?;
    let (status, code, message, session) = match reply {
        SessionReply::Created(value) => (200, String::new(), String::new(), Some(value)),
        SessionReply::Refused {
            status,
            code,
            message,
        } => (status, code, message, None),
    };
    let outcome = key_outcome(status, &code, options.regular)?;
    if outcome == KeyOutcome::SaveKeyless && !answers_without_a_key(&client, &origin).await {
        // PAIRING_UNAVAILABLE also means "auth on, Secure pairing not set
        // up"; only a host that answers keyless has authentication off.
        bail!(
            "the host can't create a pairing right now ({code}{}); run `mold host invite` on the host and pair with its link",
            if message.is_empty() {
                String::new()
            } else {
                format!(": {message}")
            }
        );
    }
    let added = match outcome {
        KeyOutcome::Claim => {
            // Claimed at once: this client gets its own revocable grant and
            // the operator key is dropped, never written anywhere.
            drop(key);
            let session = session.context("empty pairing session")?;
            let invitation = if session["mode"] == "regular" {
                pairing::regular_invitation_payload(session)?
            } else {
                pairing::invitation_payload(session, None)?
            };
            pair_invitation_with(invitation, options.name.as_deref(), device, &client).await?
        }
        KeyOutcome::SaveKey => {
            save_key(&client, &origin, Some(key), options.name.as_deref()).await?
        }
        KeyOutcome::SaveKeyless => {
            save_key(&client, &origin, None, options.name.as_deref()).await?
        }
    };
    finish(added, options.default).await
}

/// Whether `GET /api/status` succeeds with no key at all.
async fn answers_without_a_key(client: &reqwest::Client, origin: &str) -> bool {
    client
        .get(format!("{origin}/api/status"))
        .send()
        .await
        .is_ok_and(|response| response.status().is_success())
}

/// Save a Regular connection with an app/device key (or none, for a host
/// with authentication off), after one authenticated read proves it.
async fn save_key(
    client: &reqwest::Client,
    origin: &str,
    key: Option<Zeroizing<String>>,
    requested: Option<&str>,
) -> Result<Added> {
    let key = key.filter(|k| !k.is_empty());
    // Directly, as the saved Regular connection will talk: no pairing for
    // this origin exists yet, so the normal client cannot be used here.
    let mut request = client.get(format!("{origin}/api/status"));
    if let Some(key) = key.as_deref() {
        request = request.header("x-api-key", key.as_str());
    }
    let response = request
        .send()
        .await
        .with_context(|| format!("cannot contact {origin}"))?;
    match response.status().as_u16() {
        200..=299 => {}
        401 | 403 => bail!("{origin} refused that API key for direct access"),
        status => bail!(
            "{origin} answered HTTP {status}; direct (Regular) access may be off there — run `mold host invite` on the host instead"
        ),
    }
    let (name, superseded) = choose_name(requested, "", origin, None, ConnectionMode::Regular)?;
    let host = mold_secure_client::RegularHost {
        origin: origin.to_string(),
        api_key: key.map(|k| k.to_string()),
        name: Some(name.clone()),
        ..Default::default()
    };
    mold_secure_client::save_regular_host(&host)?;
    retire_superseded(superseded, origin, ConnectionMode::Regular)?;
    Ok(Added {
        name,
        origin: origin.to_string(),
        mode: ConnectionMode::Regular,
    })
}

/// `mold pair --mode regular --direct-url URL [--api-key-file F]`: save a
/// direct connection with a key the user already holds, unverified.
pub(crate) async fn add_manual_regular(
    url: &str,
    api_key: Option<Zeroizing<String>>,
) -> Result<()> {
    let origin = mold_secure_client::regular_origin(url)?;
    let (name, superseded) = choose_name(None, "", &origin, None, ConnectionMode::Regular)?;
    mold_secure_client::save_regular_host(&mold_secure_client::RegularHost {
        origin: origin.clone(),
        api_key: api_key.map(|k| k.to_string()),
        name: Some(name.clone()),
        ..Default::default()
    })?;
    retire_superseded(superseded, &origin, ConnectionMode::Regular)?;
    finish(
        Added {
            name,
            origin,
            mode: ConnectionMode::Regular,
        },
        false,
    )
    .await
}

/// The key for `host add URL`: `--api-key-file`, `--api-key-stdin`, else a
/// hidden prompt — never `MOLD_API_KEY`, which on a host is its own operator
/// key. A typed key may be `@owner-only-key-file`. Empty is allowed: it asks
/// a host with authentication off to say so.
fn api_key(options: &AddOptions) -> Result<Zeroizing<String>> {
    if let Some(path) = &options.api_key_file {
        return pairing::read_key_file(path);
    }
    let typed = if options.api_key_stdin {
        read_line(&mut std::io::stdin().lock())?
    } else if std::io::stdin().is_terminal() {
        prompt("API key (empty if the host has authentication off): ", true)?
    } else {
        return Ok(Zeroizing::new(String::new()));
    };
    match typed.strip_prefix('@') {
        Some(path) => pairing::read_key_file(Path::new(path)),
        None => Ok(typed),
    }
}

fn read_line(input: &mut impl BufRead) -> Result<Zeroizing<String>> {
    let mut line = Zeroizing::new(String::new());
    input.read_line(&mut line)?;
    Ok(Zeroizing::new(line.trim().to_string()))
}

/// Ask on stderr; `hidden` turns terminal echo off while the answer is
/// typed, and refuses to ask at all where echo cannot be turned off.
fn prompt(question: &str, hidden: bool) -> Result<Zeroizing<String>> {
    let echo = if hidden {
        Some(EchoOff::new().ok_or_else(echo_unavailable)?)
    } else {
        None
    };
    eprint!("{question}");
    std::io::stderr().flush()?;
    let _echo = echo;
    let answer = read_line(&mut std::io::stdin().lock());
    if hidden {
        eprintln!();
    }
    answer
}

/// Terminal echo off for the life of the guard (Unix); a no-op elsewhere.
struct EchoOff {
    #[cfg(unix)]
    saved: libc::termios,
}

impl EchoOff {
    #[cfg(unix)]
    fn new() -> Option<Self> {
        // SAFETY: plain termios calls on this process's stdin descriptor.
        unsafe {
            let mut saved: libc::termios = std::mem::zeroed();
            if libc::tcgetattr(libc::STDIN_FILENO, &mut saved) != 0 {
                return None;
            }
            let mut quiet = saved;
            quiet.c_lflag &= !libc::ECHO;
            (libc::tcsetattr(libc::STDIN_FILENO, libc::TCSANOW, &quiet) == 0)
                .then_some(Self { saved })
        }
    }
    #[cfg(not(unix))]
    fn new() -> Option<Self> {
        None
    }
}

impl Drop for EchoOff {
    fn drop(&mut self) {
        #[cfg(unix)]
        // SAFETY: restores the attributes read in `new`.
        unsafe {
            libc::tcsetattr(libc::STDIN_FILENO, libc::TCSANOW, &self.saved);
        }
    }
}

fn echo_unavailable() -> anyhow::Error {
    anyhow::anyhow!(
        "this terminal cannot hide a typed key; pass it with --api-key-file FILE or --api-key-stdin"
    )
}

fn default_device_name() -> String {
    let machine = hostname::get()
        .ok()
        .and_then(|name| name.into_string().ok())
        .filter(|name| !name.trim().is_empty())
        .unwrap_or_else(|| "this machine".into());
    format!("Mold CLI on {machine}")
}

/// Report a saved pairing, check it once, and apply `--default`.
pub(crate) async fn finish(added: Added, make_default: bool) -> Result<()> {
    let mode = added.mode.as_str();
    println!(
        "{} Added {} ({mode}) → {}",
        theme::icon_ok(),
        added.name,
        added.origin
    );
    match mold_core::MoldClient::for_host(Some(&added.name)) {
        Ok(client) => match client.server_status().await {
            Ok(status) => eprintln!("  reachable: mold server v{}", status.version),
            Err(error) => eprintln!(
                "{} saved, but the host did not answer just now: {error}",
                theme::icon_warn()
            ),
        },
        Err(error) => eprintln!("{} saved, but {error:#}", theme::icon_warn()),
    }
    println!("  Try: mold ps --host {}", added.name);
    if make_default {
        hosts::set_default(&added.name)?;
        println!("  Default host is now '{}'.", added.name);
    } else if hosts::list()?.len() > 1 {
        println!("  Make it the default: mold host use {}", added.name);
    }
    Ok(())
}

pub struct InviteOptions {
    pub target: std::net::SocketAddr,
    pub mode: ConnectionMode,
    pub direct_url: Option<String>,
    pub relay_url: Option<String>,
    pub api_key_file: Option<PathBuf>,
    pub qr: bool,
    pub output: Option<PathBuf>,
}

/// `mold host invite`: a one-use, two-minute invitation from this host. The
/// link alone goes to stdout (so it can be piped or copied); everything else
/// goes to stderr. `--output` writes the owner-only invitation file instead.
pub async fn invite(options: InviteOptions) -> Result<()> {
    if let Some(output) = &options.output {
        // Check the destination before consuming a short-lived invitation.
        if output.try_exists()? {
            bail!("invitation output already exists; choose a new private filename")
        }
    }
    let value = pairing::create_invitation(
        options.target,
        options.relay_url.as_deref(),
        options.mode,
        options.direct_url.as_deref(),
        options.api_key_file.as_deref(),
    )
    .await?;
    // The link is built only when it is shown; --output never depends on
    // the encoder.
    let link = (options.output.is_none() || options.qr)
        .then(|| mold_secure_client::pairing_link(&value).map(Zeroizing::new))
        .transpose()?;
    match (&options.output, &link) {
        (Some(output), _) => {
            pairing::write_invitation_file(output, &value)?;
            eprintln!(
                "Pairing invitation saved to {}. Transfer it privately; on the client run `mold host add {}` within two minutes.",
                output.display(),
                output.display()
            );
        }
        (None, Some(link)) => {
            println!("{}", link.as_str());
            eprintln!("One-use pairing link; it expires in two minutes. On the client run `mold host add` and paste it.");
        }
        (None, None) => unreachable!("a link is built whenever there is no --output"),
    }
    if let Some(key) = value["host_key"].as_str() {
        eprintln!("Host public signing key: {key}");
    }
    if let Some(link) = link.filter(|_| options.qr) {
        eprintln!("{}", render_qr(&link)?);
    }
    Ok(())
}

/// A terminal QR code of the link, two modules per character row.
fn render_qr(link: &str) -> Result<String> {
    let code = qrcode::QrCode::new(link.as_bytes())?;
    Ok(code
        .render::<qrcode::render::unicode::Dense1x2>()
        .quiet_zone(true)
        .build())
}

/// Saved host names, for `mold host use|rm <TAB>`. Never fails.
pub fn complete_saved_name() -> Vec<CompletionCandidate> {
    hosts::list()
        .unwrap_or_default()
        .into_iter()
        .map(|h| CompletionCandidate::new(h.name))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use mold_secure_client::ConnectionMode;

    fn host(name: &str, origin: &str, mode: ConnectionMode) -> SavedHost {
        SavedHost {
            name: name.into(),
            origin: origin.into(),
            mode,
            profiles: vec![mode],
            instance_id: None,
            hostname: None,
            endpoints: vec![],
            named: true,
            ambiguous: false,
        }
    }

    #[test]
    fn the_table_marks_the_default_and_aligns_columns() {
        let saved = [
            host(
                "workstation",
                "https://abc.mold-link.urandom.io",
                ConnectionMode::Secure,
            ),
            host("lab", "http://10.0.0.8:7680", ConnectionMode::Regular),
        ];
        let table = render_table(&saved, Some("http://10.0.0.8:7680"));
        let lines: Vec<&str> = table.lines().collect();
        assert_eq!(
            lines,
            [
                "NAME         MODE     ORIGIN                            DEFAULT",
                "workstation  secure   https://abc.mold-link.urandom.io",
                "lab          regular  http://10.0.0.8:7680              *",
            ]
        );
    }

    #[test]
    fn ambiguous_rows_are_marked_and_explained() {
        let mut first = host("lab", "http://10.0.0.8:7680", ConnectionMode::Regular);
        first.ambiguous = true;
        first.named = false;
        let mut second = first.clone();
        second.origin = "http://10.0.0.9:7680".into();
        let table = render_table(&[first, second], None);
        assert_eq!(table.matches("lab (ambiguous)").count(), 2, "{table}");
        assert!(table.contains("resolves to an error"), "{table}");
    }

    #[test]
    fn what_a_hosts_answer_to_a_key_means() {
        assert_eq!(key_outcome(200, "", false).unwrap(), KeyOutcome::Claim);
        assert_eq!(
            key_outcome(403, "PAIRING_OPERATOR_REQUIRED", false).unwrap(),
            KeyOutcome::SaveKey
        );
        assert_eq!(
            key_outcome(503, "PAIRING_UNAVAILABLE", true).unwrap(),
            KeyOutcome::SaveKeyless
        );
        let message =
            |status, code: &str| key_outcome(status, code, false).unwrap_err().to_string();
        assert!(message(401, "UNAUTHORIZED").contains("rejected that API key"));
        assert!(message(403, "REGULAR_ACCESS_DISABLED").contains("--allow-direct-api"));
        for (status, code) in [
            (404, ""),
            (426, "UPGRADE_REQUIRED"),
            (500, "INTERNAL_ERROR"),
        ] {
            assert!(
                message(status, code).contains("mold host invite"),
                "{status}"
            );
        }
    }

    #[test]
    fn a_link_a_file_or_json_is_an_invitation_and_a_url_is_not() {
        let link = mold_secure_client::pairing_link(&serde_json::json!({
            "type": "mold.mobile-pairing", "version": 2, "mode": "regular",
            "base_url": "http://10.0.0.8:7680", "instance_id": "i", "name": "n",
            "pairing_token": "f".repeat(64), "expires_at": 4_000_000_000_u64
        }))
        .unwrap();
        assert!(invitation_input(&link).unwrap().is_some());
        for url in ["http://10.0.0.8:7680", "10.0.0.8", "workstation:7680"] {
            assert!(invitation_input(url).unwrap().is_none(), "{url}");
        }
        assert!(invitation_input("https://utensils.io/mold/pair#version=1").is_err());
    }

    #[test]
    fn an_explicit_name_must_be_free_unless_the_same_server_holds_it() {
        let _lock = crate::test_support::ENV_LOCK
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        let home = tempfile::tempdir().unwrap();
        let previous = std::env::var_os("MOLD_HOME");
        std::env::set_var("MOLD_HOME", home.path());
        mold_secure_client::save_regular_host(&mold_secure_client::RegularHost {
            origin: "http://10.0.0.8:7680".into(),
            name: Some("workstation".into()),
            instance_id: Some("inst-1".into()),
            ..Default::default()
        })
        .unwrap();
        let taken = choose_name(
            Some("workstation"),
            "",
            "http://10.0.0.9:7680",
            None,
            ConnectionMode::Regular,
        );
        let same_server = choose_name(
            Some("Workstation"),
            "",
            "http://10.0.0.9:7680",
            Some("inst-1"),
            ConnectionMode::Regular,
        );
        let automatic = choose_name(
            None,
            "Lab Box",
            "http://10.0.0.10:7680",
            None,
            ConnectionMode::Regular,
        );
        match previous {
            Some(value) => std::env::set_var("MOLD_HOME", value),
            None => std::env::remove_var("MOLD_HOME"),
        }
        assert!(taken
            .unwrap_err()
            .to_string()
            .contains("already named 'workstation'"));
        let (name, superseded) = same_server.unwrap();
        assert_eq!(name, "workstation");
        assert_eq!(superseded.unwrap().origin, "http://10.0.0.8:7680");
        assert_eq!(automatic.unwrap().0, "lab-box");
    }

    /// A resolver for tests: no real DNS. Names map to fixed addresses.
    fn fake_dns(host: &str) -> Option<Vec<std::net::IpAddr>> {
        let ips = |list: &[&str]| Some(list.iter().map(|ip| ip.parse().unwrap()).collect());
        match host {
            "workstation" => ips(&["100.101.102.103", "fd7a:115c:a1e0::5"]),
            "workstation.tail1234.ts.net" => ips(&["100.101.102.103"]),
            "elsewhere.ts.net" => ips(&["100.64.9.9"]),
            "mixed" => ips(&["100.101.102.103", "192.168.1.5"]),
            "lan-box" => ips(&["192.168.1.5"]),
            _ => None,
        }
    }

    /// What `tailscale status --json` would report: this machine and two
    /// peers (the shape matches the real output; only the IP lists matter).
    const STATUS: &str = r#"{
        "Self": {"HostName": "laptop", "TailscaleIPs": ["100.88.1.2", "fd7a:115c:a1e0::2"]},
        "Peer": {
            "nodekey:aa": {"HostName": "workstation", "TailscaleIPs": ["100.101.102.103", "fd7a:115c:a1e0::5"]},
            "nodekey:bb": {"HostName": "render", "TailscaleIPs": ["100.105.134.43"]}
        }
    }"#;

    fn peers() -> Option<std::collections::HashSet<std::net::IpAddr>> {
        parse_tailnet_ips(STATUS.as_bytes())
    }

    fn no_tailscale() -> Option<std::collections::HashSet<std::net::IpAddr>> {
        None
    }

    /// Plain-HTTP Secure pairing is allowed over Tailscale only when the
    /// `tailscale` CLI confirms the destination is a tailnet peer: a literal
    /// address, or EVERY address a name resolves to (`.ts.net` included).
    /// 100.64.0.0/10 is also carrier-grade NAT, so the range alone proves
    /// nothing; a missing CLI or an address it does not list is refused.
    #[test]
    fn plain_http_is_refused_for_secure_and_warned_for_regular() {
        type Peers = fn() -> Option<std::collections::HashSet<std::net::IpAddr>>;
        let policy = |origin: &str, regular, peers: Peers| {
            let net = Net {
                resolve: &fake_dns,
                peers: &peers,
                prefilter: Net::system().prefilter,
            };
            plain_http_policy_with(origin, regular, &net).map(|t| t.warn)
        };
        for peers in [peers as Peers, no_tailscale] {
            for origin in [
                "https://192.168.1.5",
                "http://127.0.0.1:7680",
                "http://localhost:7680",
            ] {
                assert!(!policy(origin, false, peers).unwrap(), "{origin}");
            }
        }
        for origin in [
            "http://100.105.134.43:7680",
            "http://[fd7a:115c:a1e0::5]:7680",
            "http://workstation:7680",
            "http://workstation.tail1234.ts.net:7680",
        ] {
            assert!(!policy(origin, false, peers).unwrap(), "{origin}");
            assert!(!policy(origin, true, peers).unwrap(), "{origin}");
            let error = policy(origin, false, no_tailscale).unwrap_err().to_string();
            assert!(error.contains("plain HTTP"), "{origin}: {error}");
            assert!(policy(origin, true, no_tailscale).unwrap(), "{origin}");
        }
        for origin in [
            "http://100.64.9.9:7680",
            "http://[fd7a:115c:a1e0::99]:7680",
            "http://192.168.1.5:7680",
            "http://172.17.0.1:7680",
            "http://mixed:7680",
            "http://lan-box:7680",
            "http://unknown-name:7680",
            "http://elsewhere.ts.net:7680",
        ] {
            let error = policy(origin, false, peers).unwrap_err().to_string();
            assert!(
                error.contains("plain HTTP") && error.contains("--regular"),
                "{error}"
            );
            assert!(policy(origin, true, peers).unwrap(), "{origin}");
        }
    }

    /// A verified name is pinned to the addresses that were verified; a
    /// literal address needs no pin.
    #[test]
    fn a_verified_name_is_pinned_to_what_was_verified() {
        let net = Net {
            resolve: &fake_dns,
            peers: &peers,
            prefilter: Net::system().prefilter,
        };
        let transport = plain_http_policy_with("http://workstation:7680", false, &net).unwrap();
        let (host, mut addrs) = transport.pin.unwrap();
        addrs.sort();
        assert_eq!(host, "workstation");
        assert_eq!(
            addrs,
            vec![
                "100.101.102.103:7680".parse().unwrap(),
                "[fd7a:115c:a1e0::5]:7680".parse().unwrap()
            ]
        );
        let literal = plain_http_policy_with("http://100.105.134.43:7680", false, &net).unwrap();
        assert_eq!(literal, Transport::default());
    }

    /// The name is resolved once, for the check, and every request of the
    /// run goes to what was verified: a second answer (a different address)
    /// is never asked for and never reached.
    #[allow(clippy::await_holding_lock)]
    #[tokio::test(flavor = "multi_thread")]
    async fn every_request_goes_to_the_verified_address() {
        use wiremock::matchers::{method, path};
        use wiremock::{Mock, MockServer, ResponseTemplate};
        let verified = MockServer::start().await;
        let elsewhere = MockServer::start().await;
        for server in [&verified, &elsewhere] {
            Mock::given(method("POST"))
                .and(path("/api/pairing/sessions"))
                .respond_with(ResponseTemplate::new(503).set_body_json(
                    serde_json::json!({"error": "off", "code": "PAIRING_UNAVAILABLE"}),
                ))
                .mount(server)
                .await;
            Mock::given(method("GET"))
                .and(path("/api/status"))
                .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
                    "version": "0.36.0", "git_sha": null, "build_date": null,
                    "models_loaded": [], "busy": false, "uptime_secs": 1
                })))
                .mount(server)
                .await;
        }
        let port = verified.address().port();
        let calls = std::sync::atomic::AtomicUsize::new(0);
        let elsewhere_ip = elsewhere.address().ip();
        let resolve = |_: &str| -> Option<Vec<std::net::IpAddr>> {
            let first = calls.fetch_add(1, std::sync::atomic::Ordering::SeqCst) == 0;
            Some(vec![if first {
                "127.0.0.1".parse().unwrap()
            } else {
                elsewhere_ip
            }])
        };
        let peer_set = || Some(IpSet::from(["127.0.0.1".parse().unwrap()]));
        let net = Net {
            resolve: &resolve,
            peers: &peer_set,
            prefilter: |_| true,
        };
        let home = tempfile::tempdir().unwrap();
        let _lock = crate::test_support::ENV_LOCK
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        let previous = std::env::var_os("MOLD_HOME");
        std::env::set_var("MOLD_HOME", home.path());
        // An unresolvable name: reaching the listener at all proves the pin.
        let origin = format!("http://workstation.invalid:{port}");
        let options = AddOptions {
            input: None,
            name: Some("workstation".into()),
            api_key_file: None,
            api_key_stdin: false,
            regular: false,
            default: false,
            device_name: None,
        };
        let outcome = add_with_key_using(origin, &options, "test", &net).await;
        match previous {
            Some(value) => std::env::set_var("MOLD_HOME", value),
            None => std::env::remove_var("MOLD_HOME"),
        }
        outcome.unwrap();
        assert_eq!(calls.load(std::sync::atomic::Ordering::SeqCst), 1);
        let reached = verified.received_requests().await.unwrap();
        assert!(reached
            .iter()
            .any(|r| r.url.path() == "/api/pairing/sessions"));
        assert!(reached.iter().any(|r| r.url.path() == "/api/status"));
        assert!(elsewhere.received_requests().await.unwrap().is_empty());
    }

    #[test]
    fn tailscale_status_is_read_for_ip_lists_only() {
        let ips = peers().unwrap();
        for ip in [
            "100.88.1.2",
            "100.101.102.103",
            "fd7a:115c:a1e0::5",
            "100.105.134.43",
        ] {
            assert!(ips.contains(&ip.parse().unwrap()), "{ip}");
        }
        assert_eq!(ips.len(), 5);
        assert_eq!(parse_tailnet_ips(b"not json"), None);
        assert_eq!(parse_tailnet_ips(b"{}").map(|s| s.len()), Some(0));
    }

    #[cfg(unix)]
    #[test]
    fn the_tailscale_cli_is_bounded_and_may_be_missing() {
        let short = std::time::Duration::from_millis(200);
        assert_eq!(
            run_bounded(Path::new("/nonexistent/tailscale"), &["status"], short),
            None
        );
        let started = std::time::Instant::now();
        assert_eq!(run_bounded(Path::new("/bin/sleep"), &["5"], short), None);
        assert!(started.elapsed() < std::time::Duration::from_secs(3));
        assert_eq!(
            run_bounded(Path::new("/bin/sh"), &["-c", "exit 3"], short),
            None
        );
        assert_eq!(
            run_bounded(
                Path::new("/bin/sh"),
                &["-c", "printf ok"],
                std::time::Duration::from_secs(3)
            ),
            Some(b"ok".to_vec())
        );
    }

    #[test]
    fn only_a_spelled_path_is_read_as_a_file() {
        for host in ["workstation", "workstation:7680", "http://x/y", "10.0.0.8"] {
            assert_eq!(file_argument(host), None, "{host}");
        }
        for path in [
            "./invite.txt",
            "../invite.txt",
            "/private/invite.txt",
            "dir/invite.txt",
        ] {
            assert_eq!(file_argument(path), Some(PathBuf::from(path)), "{path}");
        }
        let home = std::env::var_os("HOME")
            .map(PathBuf::from)
            .unwrap_or_default();
        assert_eq!(file_argument("~/invite.txt"), Some(home.join("invite.txt")));
    }

    #[test]
    fn a_key_prompt_that_would_echo_names_the_alternatives() {
        let message = echo_unavailable().to_string();
        assert!(message.contains("--api-key-file") && message.contains("--api-key-stdin"));
    }

    #[test]
    fn the_qr_code_is_drawn_in_text() {
        let qr = render_qr("https://utensils.io/mold/pair#version=2").unwrap();
        assert!(qr.lines().count() > 10);
        assert!(qr.contains('█') || qr.contains('▀') || qr.contains('▄'));
    }

    #[test]
    fn json_rows_carry_the_default_flag() {
        let row = row_json(&host("lab", "http://x:7680", ConnectionMode::Regular), true);
        assert_eq!(row["name"], "lab");
        assert_eq!(row["mode"], "regular");
        assert_eq!(row["default"], true);
    }
}
