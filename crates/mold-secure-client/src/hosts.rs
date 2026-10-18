//! Named hosts over the CLI's existing credential profiles.
//!
//! A saved host IS its credential profile in `$MOLD_HOME/secure/clients`:
//! `<sha256(origin)>.json` for Secure, `regular-<sha256(origin)>.json` for
//! Regular, and `regular-<sha256(origin)>.mode` choosing between the two when
//! both exist. There is no second registry to drift from it: the name is an
//! additive field on those profiles, and the default host is one owner-only
//! `default-host` file holding an origin.
//!
//! A profile written before names existed lists under a derived name, so it
//! is addressable without being rewritten: the slug of the host's reported
//! hostname, else `runpod-<pod>` for a RunPod proxy origin, `relay-<8 hex>`
//! for a managed-proxy origin, or the slug of the origin's first DNS label
//! (`host-<ip>` for an address). Derived names are never suffixed, because a
//! suffix would move to another host after a removal: a derived name two
//! hosts share, or one a stored name already holds, is listed as ambiguous
//! and resolves to an error. [`persist_derived_names`] (run by the explicit
//! `mold host` commands) writes each unambiguous derived name into its
//! profile, so from then on it is a stored name, and stored names never move.
//!
//! Listing reads metadata only; credentials are never parsed out of a
//! profile here. A profile that cannot be read is skipped and reported, and
//! a bare name that finds nothing while one was skipped is an error rather
//! than a DNS lookup that might reach a different machine.
use crate::{ConnectionMode, Endpoint};
use anyhow::{bail, Context, Result};
use serde::Deserialize;
use serde_json::{Map, Value};
use std::{
    collections::{BTreeMap, BTreeSet},
    path::{Path, PathBuf},
};

/// The default target when nothing names a host.
pub const LOCAL_HOST: &str = "http://localhost:7680";
const DEFAULT_FILE: &str = "default-host";
const MAX_NAME: usize = 63;

/// One saved host, as `mold host list` shows it.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SavedHost {
    /// Unique, lowercase; what `--host` and `MOLD_HOST` accept.
    pub name: String,
    /// The exact origin its credentials are filed under.
    pub origin: String,
    /// The selected mode (the `.mode` file, else the only profile's).
    pub mode: ConnectionMode,
    /// Which profiles are on disk for this origin.
    pub profiles: Vec<ConnectionMode>,
    pub instance_id: Option<String>,
    pub hostname: Option<String>,
    /// Endpoints the host advertised when it was paired (Secure only).
    pub endpoints: Vec<Endpoint>,
    /// True when `name` is stored in the profile and no other host holds it.
    pub named: bool,
    /// True when another saved host lists under the same name, so the name
    /// resolves to an error until one is renamed or re-paired.
    pub ambiguous: bool,
}

/// How [`resolve`] matched its input.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Match {
    /// A saved host name.
    Name,
    /// The exact origin the host's credentials are filed under.
    Origin,
    /// A Tailscale address the host advertised: the caller typed an address
    /// and is being sent to a different origin, so it should say so.
    Tailscale,
}

/// A resolved saved host.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Resolved {
    pub origin: String,
    pub name: String,
    pub via: Match,
}

/// Every saved host plus the profile files that could not be read.
struct Scan {
    hosts: Vec<SavedHost>,
    skipped: Vec<PathBuf>,
}

#[derive(Deserialize)]
struct ProfileMeta {
    origin: String,
    #[serde(default)]
    name: Option<String>,
    #[serde(default)]
    hostname: Option<String>,
    #[serde(default)]
    instance_id: Option<String>,
    #[serde(default)]
    endpoints: Vec<Endpoint>,
}

#[derive(Default)]
struct Entry {
    secure: Option<ProfileMeta>,
    regular: Option<ProfileMeta>,
    mode: Option<ConnectionMode>,
}

/// Normalize a host address into a base URL: a bare `host` or `host:port`
/// gets `http://` and, without a port, `:7680`; an explicit scheme keeps its
/// own default port.
pub fn normalize_address(input: &str) -> String {
    let trimmed = input.trim().trim_end_matches('/');
    let has_scheme = has_http_scheme(trimmed);
    let bare_ipv6 = !has_scheme
        && !trimmed.starts_with('[')
        && trimmed.bytes().filter(|byte| *byte == b':').count() > 1;
    let candidate = if has_scheme {
        trimmed.to_string()
    } else if bare_ipv6 {
        format!("http://[{trimmed}]")
    } else {
        format!("http://{trimmed}")
    };
    if let Ok(mut url) = reqwest::Url::parse(&candidate) {
        if !has_scheme && url.port().is_none() {
            let _ = url.set_port(Some(7680));
        }
        url.to_string().trim_end_matches('/').to_string()
    } else {
        candidate
    }
}

fn has_http_scheme(input: &str) -> bool {
    input
        .get(..7)
        .is_some_and(|prefix| prefix.eq_ignore_ascii_case("http://"))
        || input
            .get(..8)
            .is_some_and(|prefix| prefix.eq_ignore_ascii_case("https://"))
}

/// Lowercase slug for a host label: ASCII letters, digits and single dashes,
/// starting with a letter. `None` when nothing usable is left.
pub fn slug(raw: &str) -> Option<String> {
    let mut out = String::new();
    for c in raw.trim().chars() {
        if c.is_ascii_alphanumeric() {
            out.push(c.to_ascii_lowercase());
        } else if !out.is_empty() && !out.ends_with('-') {
            out.push('-');
        }
    }
    let out = out.trim_end_matches('-');
    if out.is_empty() {
        return None;
    }
    let mut out = if out.as_bytes()[0].is_ascii_digit() {
        format!("host-{out}")
    } else {
        out.to_string()
    };
    out.truncate(MAX_NAME);
    let out = out.trim_end_matches('-').to_string();
    (validate_name(&out).is_ok()).then_some(out)
}

/// A host name can never be mistaken for an address: no dots or colons, it
/// starts with a letter, and `localhost` is reserved.
pub fn validate_name(name: &str) -> Result<()> {
    let bytes = name.as_bytes();
    anyhow::ensure!(
        !bytes.is_empty()
            && bytes.len() <= MAX_NAME
            && bytes[0].is_ascii_lowercase()
            && !name.ends_with('-')
            && !name.contains("--")
            && bytes
                .iter()
                .all(|b| b.is_ascii_lowercase() || b.is_ascii_digit() || *b == b'-')
            && name != "localhost",
        "host names are lowercase letters, digits and dashes, start with a letter, and are at most {MAX_NAME} characters"
    );
    Ok(())
}

/// Every saved host in the default store.
pub fn list() -> Result<Vec<SavedHost>> {
    list_at(&crate::default_store())
}

pub fn list_at(root: &Path) -> Result<Vec<SavedHost>> {
    Ok(scan_at(root)?.hosts)
}

fn scan_at(root: &Path) -> Result<Scan> {
    let mut entries: BTreeMap<String, Entry> = BTreeMap::new();
    let mut skipped = Vec::new();
    if !root
        .try_exists()
        .with_context(|| format!("cannot read saved hosts in {}", root.display()))?
    {
        return Ok(Scan {
            hosts: Vec::new(),
            skipped,
        });
    }
    let mut modes = BTreeMap::new();
    let unreadable = || format!("cannot read saved hosts in {}", root.display());
    for item in std::fs::read_dir(root).with_context(unreadable)? {
        let path = item.with_context(unreadable)?.path();
        let Some(file) = path.file_name().and_then(|v| v.to_str()) else {
            continue;
        };
        if let Some(digest) = file
            .strip_prefix("regular-")
            .and_then(|v| v.strip_suffix(".mode"))
        {
            match crate::read_private(&path)
                .ok()
                .and_then(|bytes| serde_json::from_slice::<ConnectionMode>(&bytes).ok())
            {
                Some(mode) => {
                    modes.insert(digest.to_string(), mode);
                }
                None => skipped.push(path),
            }
            continue;
        }
        let Some(stem) = file.strip_suffix(".json") else {
            continue;
        };
        let (regular, digest) = match stem.strip_prefix("regular-") {
            Some(digest) => (true, digest),
            None => (false, stem),
        };
        // A profile is filed under the digest of its own origin; one that
        // claims another origin is not this host's credential.
        let Some(meta) =
            read_meta(&path).filter(|m| crate::store::origin_digest(&m.origin) == digest)
        else {
            skipped.push(path);
            continue;
        };
        let entry = entries.entry(meta.origin.clone()).or_default();
        if regular {
            entry.regular = Some(meta);
        } else {
            entry.secure = Some(meta);
        }
    }
    for (origin, entry) in entries.iter_mut() {
        entry.mode = modes.get(&crate::store::origin_digest(origin)).copied();
    }
    skipped.sort();
    Ok(Scan {
        hosts: assign_names(entries),
        skipped,
    })
}

fn read_meta(path: &Path) -> Option<ProfileMeta> {
    serde_json::from_slice(&crate::read_private(path).ok()?).ok()
}

fn assign_names(entries: BTreeMap<String, Entry>) -> Vec<SavedHost> {
    let mut hosts: Vec<(SavedHost, Option<String>)> = entries
        .into_iter()
        .filter_map(|(origin, entry)| {
            let mode = match (&entry.secure, &entry.regular, entry.mode) {
                (Some(_), Some(_), Some(mode)) => mode,
                (Some(_), _, _) => ConnectionMode::Secure,
                (None, Some(_), _) => ConnectionMode::Regular,
                (None, None, _) => return None,
            };
            let (primary, other) = match mode {
                ConnectionMode::Secure => (entry.secure.as_ref(), entry.regular.as_ref()),
                ConnectionMode::Regular => (entry.regular.as_ref(), entry.secure.as_ref()),
            };
            let pick = |field: fn(&ProfileMeta) -> &Option<String>| {
                primary
                    .and_then(|m| field(m).clone())
                    .or_else(|| other.and_then(|m| field(m).clone()))
                    .filter(|v| !v.is_empty())
            };
            let stored = pick(|m| &m.name)
                .map(|name| name.to_ascii_lowercase())
                .filter(|name| validate_name(name).is_ok());
            let hostname = pick(|m| &m.hostname);
            let endpoints = entry
                .secure
                .as_ref()
                .map(|m| m.endpoints.clone())
                .unwrap_or_default();
            let profiles = [
                entry.secure.as_ref().map(|_| ConnectionMode::Secure),
                entry.regular.as_ref().map(|_| ConnectionMode::Regular),
            ]
            .into_iter()
            .flatten()
            .collect();
            Some((
                SavedHost {
                    name: String::new(),
                    instance_id: pick(|m| &m.instance_id),
                    hostname,
                    endpoints,
                    mode,
                    profiles,
                    origin,
                    named: false,
                    ambiguous: false,
                },
                stored,
            ))
        })
        .collect();
    // A name belongs to its one stored holder; a derived name sharing it,
    // or a name two hosts hold the same way, is ambiguous.
    let mut stored_holders: BTreeMap<String, usize> = BTreeMap::new();
    let mut holders: BTreeMap<String, usize> = BTreeMap::new();
    for (host, stored) in hosts.iter_mut() {
        host.name = stored
            .clone()
            .unwrap_or_else(|| derived_name(&host.origin, host.hostname.as_deref()));
        *holders.entry(host.name.clone()).or_default() += 1;
        if stored.is_some() {
            *stored_holders.entry(host.name.clone()).or_default() += 1;
        }
    }
    for (host, stored) in hosts.iter_mut() {
        let owners = stored_holders.get(&host.name).copied().unwrap_or(0);
        host.named = stored.is_some() && owners == 1;
        host.ambiguous = if stored.is_some() {
            owners > 1
        } else {
            holders[&host.name] > 1
        };
    }
    hosts.into_iter().map(|(host, _)| host).collect()
}

/// The one saved host `name` names. An ambiguous name is an error that says
/// how to make it unique.
fn lookup<'a>(hosts: &'a [SavedHost], name: &str) -> Result<Option<&'a SavedHost>> {
    let name = name.trim();
    let matches: Vec<&SavedHost> = hosts
        .iter()
        .filter(|h| h.name.eq_ignore_ascii_case(name))
        .collect();
    if let [one] = matches.as_slice() {
        return Ok(Some(one));
    }
    if let Some(owner) = matches.iter().find(|h| h.named) {
        return Ok(Some(owner));
    }
    if matches.is_empty() {
        return Ok(None);
    }
    bail!(
        "'{name}' names more than one saved host ({}); re-pair one of them so it gets its own name, then `mold host use` it",
        matches
            .iter()
            .map(|h| h.origin.as_str())
            .collect::<Vec<_>>()
            .join(", ")
    )
}

/// Write each unambiguous derived name into its host's profile(s), keeping
/// every other field, so the name no longer depends on what else is saved.
/// Returns the hosts that were named. Run only by explicit host commands.
pub fn persist_derived_names() -> Result<Vec<SavedHost>> {
    persist_derived_names_at(&crate::default_store())
}

pub fn persist_derived_names_at(root: &Path) -> Result<Vec<SavedHost>> {
    let mut named = Vec::new();
    for host in list_at(root)?
        .into_iter()
        .filter(|h| !h.named && !h.ambiguous)
    {
        let digest = crate::store::origin_digest(&host.origin);
        for file in [format!("{digest}.json"), format!("regular-{digest}.json")] {
            let path = root.join(file);
            if !path.try_exists()? {
                continue;
            }
            let name = host.name.clone();
            rewrite_profile(&path, |profile| {
                profile.insert("name".into(), name.into());
            })?;
        }
        named.push(SavedHost {
            named: true,
            ..host
        });
    }
    Ok(named)
}

fn next_free(base: &str, taken: &BTreeSet<String>) -> String {
    if !taken.contains(base) {
        return base.to_string();
    }
    (2..)
        .map(|n| {
            let suffix = format!("-{n}");
            let mut stem = base.to_string();
            stem.truncate(MAX_NAME - suffix.len());
            format!("{}{suffix}", stem.trim_end_matches('-'))
        })
        .find(|name| !taken.contains(name))
        .expect("an unbounded suffix search always finds a free name")
}

/// The name a profile without one lists under. See the module docs.
fn derived_name(origin: &str, hostname: Option<&str>) -> String {
    if let Some(name) = hostname.and_then(slug) {
        return name;
    }
    let url = reqwest::Url::parse(origin).ok();
    let host = url
        .as_ref()
        .and_then(|u| u.host_str())
        .unwrap_or_default()
        .trim_matches(['[', ']'])
        .to_string();
    if let Some(pod) = host.strip_suffix("-7680.proxy.runpod.net") {
        if let Some(name) = slug(&format!("runpod-{pod}")) {
            return name;
        }
    }
    if url.as_ref().is_some_and(crate::is_managed_proxy) {
        let label: String = host.chars().take_while(|c| *c != '.').take(8).collect();
        if let Some(name) = slug(&format!("relay-{label}")) {
            return name;
        }
    }
    let label = if host.parse::<std::net::IpAddr>().is_ok() {
        host.as_str()
    } else {
        host.split('.').next().unwrap_or_default()
    };
    slug(label).unwrap_or_else(|| "host".into())
}

/// Zeroize every string in a parsed profile (credentials and identity keys
/// among them) before the value is dropped.
fn wipe_secrets(value: &mut serde_json::Value) {
    use zeroize::Zeroize;
    match value {
        serde_json::Value::String(text) => text.zeroize(),
        serde_json::Value::Array(items) => items.iter_mut().for_each(wipe_secrets),
        serde_json::Value::Object(map) => map.values_mut().for_each(wipe_secrets),
        _ => {}
    }
}

/// The saved host with this name, case-insensitively.
pub fn find(name: &str) -> Result<Option<SavedHost>> {
    find_at(&crate::default_store(), name)
}

pub fn find_at(root: &Path, name: &str) -> Result<Option<SavedHost>> {
    let hosts = list_at(root)?;
    Ok(lookup(&hosts, name)?.cloned())
}

pub fn find_by_origin(origin: &str) -> Result<Option<SavedHost>> {
    find_by_origin_at(&crate::default_store(), origin)
}

pub fn find_by_origin_at(root: &Path, origin: &str) -> Result<Option<SavedHost>> {
    Ok(list_at(root)?.into_iter().find(|h| h.origin == origin))
}

/// The saved host a re-pair of the same server should replace.
pub fn find_by_instance_id(instance_id: &str) -> Result<Option<SavedHost>> {
    find_by_instance_id_at(&crate::default_store(), instance_id)
}

pub fn find_by_instance_id_at(root: &Path, instance_id: &str) -> Result<Option<SavedHost>> {
    Ok(list_at(root)?
        .into_iter()
        .find(|h| !instance_id.is_empty() && h.instance_id.as_deref() == Some(instance_id)))
}

/// A unique name for the host saved at `origin`, from a reported label: the
/// unambiguous name `origin` already lists under (re-pairing keeps it), else
/// the label's slug when free, else the first free `-N` suffix. Callers that
/// took the name from the user should refuse a collision instead; see
/// [`require_free_name`].
pub fn available_name(base: &str, origin: &str) -> Result<String> {
    available_name_at(&crate::default_store(), base, origin)
}

pub fn available_name_at(root: &Path, base: &str, origin: &str) -> Result<String> {
    let hosts = list_at(root)?;
    if let Some(saved) = hosts.iter().find(|h| h.origin == origin && !h.ambiguous) {
        return Ok(saved.name.clone());
    }
    let base = slug(base).unwrap_or_else(|| derived_name(origin, None));
    let taken: BTreeSet<String> = hosts
        .into_iter()
        .filter(|h| h.origin != origin)
        .map(|h| h.name)
        .collect();
    Ok(next_free(&base, &taken))
}

/// Whether `name` is unused by every saved host other than `origin`'s.
pub fn name_is_free(name: &str, origin: &str) -> Result<bool> {
    name_is_free_at(&crate::default_store(), name, origin)
}

pub fn name_is_free_at(root: &Path, name: &str, origin: &str) -> Result<bool> {
    Ok(!list_at(root)?
        .iter()
        .any(|h| h.origin != origin && h.name.eq_ignore_ascii_case(name)))
}

/// Delete a saved host's Secure and Regular credentials and its mode
/// selection, and clear the default when it pointed here.
pub fn remove(name: &str) -> Result<SavedHost> {
    remove_at(&crate::default_store(), name)
}

pub fn remove_at(root: &Path, name: &str) -> Result<SavedHost> {
    let host = find_at(root, name)?.with_context(|| unknown(name))?;
    delete_profiles(root, &host.origin)?;
    if read_default_origin(root)?.as_deref() == Some(host.origin.as_str()) {
        clear_default_at(root)?;
    }
    Ok(host)
}

fn remove_if_present(path: &Path) -> Result<()> {
    match std::fs::remove_file(path) {
        Err(e) if e.kind() != std::io::ErrorKind::NotFound => Err(e.into()),
        _ => Ok(()),
    }
}

/// Rewrite one owner-only profile in place through `edit`, keeping every
/// other field and wiping the parsed copy's secrets afterwards.
fn rewrite_profile(path: &Path, edit: impl FnOnce(&mut Map<String, Value>)) -> Result<()> {
    let bytes = crate::read_private(path)?;
    let mut profile: Value = serde_json::from_slice(&bytes)?;
    edit(
        profile
            .as_object_mut()
            .with_context(|| format!("invalid saved host profile {}", path.display()))?,
    );
    let serialized = serde_json::to_vec(&profile);
    wipe_secrets(&mut profile);
    drop(profile);
    let bytes = zeroize::Zeroizing::new(serialized?);
    mold_secure::identity_file::write_private_file(path, &bytes, true)?;
    Ok(())
}

fn delete_profiles(root: &Path, origin: &str) -> Result<()> {
    let digest = crate::store::origin_digest(origin);
    for file in [
        format!("{digest}.json"),
        format!("regular-{digest}.json"),
        format!("regular-{digest}.mode"),
    ] {
        match std::fs::remove_file(root.join(file)) {
            Err(e) if e.kind() != std::io::ErrorKind::NotFound => return Err(e.into()),
            _ => {}
        }
    }
    Ok(())
}

/// Make `name` the host every command targets when neither `--host` nor
/// `MOLD_HOST` names one.
pub fn set_default(name: &str) -> Result<SavedHost> {
    set_default_at(&crate::default_store(), name)
}

pub fn set_default_at(root: &Path, name: &str) -> Result<SavedHost> {
    let host = find_at(root, name)?.with_context(|| unknown(name))?;
    mold_secure::identity_file::write_private_file(
        &root.join(DEFAULT_FILE),
        host.origin.as_bytes(),
        true,
    )?;
    Ok(host)
}

/// Forget the default host; commands go back to `http://localhost:7680`.
/// Returns whether one was set.
pub fn clear_default() -> Result<bool> {
    clear_default_at(&crate::default_store())
}

pub fn clear_default_at(root: &Path) -> Result<bool> {
    match std::fs::remove_file(root.join(DEFAULT_FILE)) {
        Ok(()) => Ok(true),
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(false),
        Err(e) => Err(e.into()),
    }
}

/// The default host, if one is set and still saved.
pub fn default() -> Result<Option<SavedHost>> {
    default_at(&crate::default_store())
}

/// `Ok(None)` when no default is set, or its host has since been removed. A
/// default whose profile is still there but cannot be read is an error, so
/// a broken file never quietly sends commands to localhost instead.
pub fn default_at(root: &Path) -> Result<Option<SavedHost>> {
    let Some(origin) = read_default_origin(root)? else {
        return Ok(None);
    };
    let scan = scan_at(root)?;
    if let Some(host) = scan.hosts.into_iter().find(|h| h.origin == origin) {
        return Ok(Some(host));
    }
    let digest = crate::store::origin_digest(&origin);
    let broken = scan.skipped.iter().find(|path| {
        path.file_name()
            .and_then(|n| n.to_str())
            .is_some_and(|name| {
                name == format!("{digest}.json") || name == format!("regular-{digest}.json")
            })
    });
    match broken {
        Some(path) => bail!(
            "the default host {origin} is saved in {}, which cannot be read: {}; fix it, re-pair the host, or run `mold host use --clear`",
            path.display(),
            crate::explain_private_file(path)
                .unwrap_or_else(|| "it is not a valid saved host profile".into())
        ),
        None => Ok(None),
    }
}

fn read_default_origin(root: &Path) -> Result<Option<String>> {
    let path = root.join(DEFAULT_FILE);
    if !path.try_exists()? {
        return Ok(None);
    }
    let bytes = crate::read_private(&path)?;
    let origin = std::str::from_utf8(&bytes)
        .context("invalid default host")?
        .trim()
        .to_string();
    Ok((!origin.is_empty()).then_some(origin))
}

fn unknown(name: &str) -> String {
    format!("no saved host named '{name}'; run `mold host list`")
}

/// The saved host `input` names, or `None` to use the address as typed.
///
/// In order: a saved host name (case-insensitive; an ambiguous one is an
/// error); a saved origin; a Tailscale address (100.64.0.0/10 or
/// fd7a:115c:a1e0::/48) one saved host advertised as a `tailscale`
/// endpoint; and, when the input has neither a scheme nor a port, the one
/// saved host with a Tailscale endpoint on that machine. LAN endpoints never
/// redirect an address, loopback never does, and an ambiguous match is no
/// match.
///
/// A bare name (one [`validate_name`] accepts) that matches nothing is an
/// error when the store could not be read completely, since the host it
/// meant may be in the unreadable profile; with a clean store it falls
/// through to DNS (`hal9000` over MagicDNS is a supported use).
pub fn resolve(input: &str) -> Result<Option<Resolved>> {
    resolve_at(&crate::default_store(), input)
}

pub fn resolve_at(root: &Path, input: &str) -> Result<Option<Resolved>> {
    let input = input.trim();
    if input.is_empty() {
        return Ok(None);
    }
    let bare_name = validate_name(&input.to_ascii_lowercase()).is_ok();
    let scan = match scan_at(root) {
        Ok(scan) => scan,
        Err(error) if bare_name => return Err(error),
        Err(_) => return Ok(None),
    };
    let found = |host: &SavedHost, via| {
        Ok(Some(Resolved {
            origin: host.origin.clone(),
            name: host.name.clone(),
            via,
        }))
    };
    if let Some(host) = lookup(&scan.hosts, input)? {
        return found(host, Match::Name);
    }
    if bare_name && !scan.skipped.is_empty() {
        bail!(
            "'{input}' is not a saved host, but some saved hosts could not be read ({}); fix or remove them before using a bare name",
            scan.skipped
                .iter()
                .map(|p| p.display().to_string())
                .collect::<Vec<_>>()
                .join(", ")
        )
    }
    let address = normalize_address(input);
    if let Some(host) = scan.hosts.iter().find(|h| h.origin == address) {
        return found(host, Match::Origin);
    }
    let Some(machine) = machine_of(&address) else {
        return Ok(None);
    };
    if is_loopback(&machine) || !is_tailscale_address(&machine) {
        return Ok(None);
    }
    let tailscale = |h: &&SavedHost| -> Vec<String> {
        h.endpoints
            .iter()
            .filter(|e| e.kind == "tailscale")
            .map(|e| normalize_address(&e.url))
            .filter(|url| machine_of(url).is_some_and(|m| is_tailscale_address(&m)))
            .collect()
    };
    let unique = |matches: Vec<&SavedHost>| match matches.as_slice() {
        [one] => found(one, Match::Tailscale),
        _ => Ok(None),
    };
    let exact: Vec<&SavedHost> = scan
        .hosts
        .iter()
        .filter(|h| tailscale(h).contains(&address))
        .collect();
    if !exact.is_empty() {
        return unique(exact);
    }
    if has_http_scheme(input) || typed_port(input) {
        return Ok(None);
    }
    let same_machine = scan
        .hosts
        .iter()
        .filter(|h| {
            tailscale(h)
                .iter()
                .any(|url| machine_of(url).as_deref() == Some(machine.as_str()))
        })
        .collect();
    unique(same_machine)
}

fn machine_of(address: &str) -> Option<String> {
    let url = reqwest::Url::parse(address).ok()?;
    Some(
        url.host_str()?
            .trim_matches(['[', ']'])
            .to_ascii_lowercase(),
    )
}

/// Tailscale's CGNAT IPv4 range (100.64.0.0/10) and its IPv6 ULA prefix
/// (fd7a:115c:a1e0::/48): a WireGuard-encrypted path even over plain HTTP.
pub fn is_tailscale_address(machine: &str) -> bool {
    match machine.trim_matches(['[', ']']).parse::<std::net::IpAddr>() {
        Ok(std::net::IpAddr::V4(v4)) => {
            let [a, b, ..] = v4.octets();
            a == 100 && (64..128).contains(&b)
        }
        Ok(std::net::IpAddr::V6(v6)) => v6.segments()[..3] == [0xfd7a, 0x115c, 0xa1e0],
        Err(_) => false,
    }
}

/// Whether the user spelled a port.
fn typed_port(input: &str) -> bool {
    let trimmed = input.trim().trim_end_matches('/');
    let bare_ipv6 = !has_http_scheme(trimmed)
        && !trimmed.starts_with('[')
        && trimmed.bytes().filter(|byte| *byte == b':').count() > 1;
    let candidate = if has_http_scheme(trimmed) {
        trimmed.to_string()
    } else if bare_ipv6 {
        format!("http://[{trimmed}]")
    } else {
        format!("http://{trimmed}")
    };
    reqwest::Url::parse(&candidate).is_ok_and(|url| url.port().is_some())
}

fn is_loopback(machine: &str) -> bool {
    machine == "localhost"
        || machine
            .parse::<std::net::IpAddr>()
            .is_ok_and(|ip| ip.is_loopback())
}

/// Bail with a clear message when a requested name is unusable or taken.
pub fn require_free_name(name: &str, origin: &str) -> Result<String> {
    require_free_name_at(&crate::default_store(), name, origin)
}

pub fn require_free_name_at(root: &Path, name: &str, origin: &str) -> Result<String> {
    let name = name.trim().to_ascii_lowercase();
    validate_name(&name)?;
    if !name_is_free_at(root, &name, origin)? {
        bail!("a saved host is already named '{name}'; choose another name")
    }
    Ok(name)
}

/// The name to save a new `mode` pairing of `origin` under, and the saved
/// host it supersedes: when the same server (`instance_id`) is already saved
/// in the same mode under a different origin, the new pairing takes that
/// host's name, and the caller [`retire`]s the old entry after saving. A
/// pairing in the other mode is a second, deliberate route and is kept.
/// Otherwise see [`available_name`].
pub fn name_for_pairing(
    reported: &str,
    origin: &str,
    instance_id: Option<&str>,
    mode: ConnectionMode,
) -> Result<(String, Option<SavedHost>)> {
    name_for_pairing_at(&crate::default_store(), reported, origin, instance_id, mode)
}

pub fn name_for_pairing_at(
    root: &Path,
    reported: &str,
    origin: &str,
    instance_id: Option<&str>,
    mode: ConnectionMode,
) -> Result<(String, Option<SavedHost>)> {
    // Every entry of this server, not just the first: the one to replace is
    // the one holding a profile in this mode at another origin.
    if let Some(previous) = instance_id.filter(|id| !id.is_empty()).and_then(|id| {
        list_at(root).ok()?.into_iter().find(|h| {
            h.instance_id.as_deref() == Some(id)
                && h.origin != origin
                && h.profiles.contains(&mode)
                && !h.ambiguous
        })
    }) {
        return Ok((previous.name.clone(), Some(previous)));
    }
    Ok((available_name_at(root, reported, origin)?, None))
}

/// Delete the `mode` profile a pairing superseded after its server was saved
/// again at `origin`, moving the default along with it. A profile in the
/// other mode stays: it becomes that origin's selected mode and gives up
/// its name, which the new pairing now holds.
pub fn retire(superseded: &SavedHost, origin: &str, mode: ConnectionMode) -> Result<()> {
    retire_at(&crate::default_store(), superseded, origin, mode)
}

pub fn retire_at(
    root: &Path,
    superseded: &SavedHost,
    origin: &str,
    mode: ConnectionMode,
) -> Result<()> {
    anyhow::ensure!(
        superseded.origin != origin,
        "a pairing cannot supersede itself"
    );
    let was_default = read_default_origin(root)?.as_deref() == Some(superseded.origin.as_str());
    let digest = crate::store::origin_digest(&superseded.origin);
    let (gone, kept, kept_mode) = match mode {
        ConnectionMode::Secure => (
            format!("{digest}.json"),
            format!("regular-{digest}.json"),
            ConnectionMode::Regular,
        ),
        ConnectionMode::Regular => (
            format!("regular-{digest}.json"),
            format!("{digest}.json"),
            ConnectionMode::Secure,
        ),
    };
    if root.join(&kept).try_exists()? {
        remove_if_present(&root.join(&gone))?;
        mold_secure::identity_file::write_private_file(
            &root.join(format!("regular-{digest}.mode")),
            &serde_json::to_vec(&kept_mode)?,
            true,
        )?;
        rewrite_profile(&root.join(&kept), |profile| {
            profile.remove("name");
        })?;
    } else {
        delete_profiles(root, &superseded.origin)?;
    }
    if was_default {
        mold_secure::identity_file::write_private_file(
            &root.join(DEFAULT_FILE),
            origin.as_bytes(),
            true,
        )?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn store() -> (tempfile::TempDir, std::path::PathBuf) {
        let dir = tempfile::tempdir().unwrap();
        let root = dir.path().join("secure/clients");
        (dir, root)
    }

    /// Endpoints as `(kind, url)`.
    fn secure(root: &Path, origin: &str, name: Option<&str>, endpoints: &[(&str, &str)]) {
        let mut host = crate::prepare_bootstrap("hosts", "cli")
            .unwrap()
            .profile(origin)
            .unwrap();
        host.name = name.map(str::to_owned);
        host.endpoints = endpoints
            .iter()
            .map(|(kind, url)| Endpoint {
                url: (*url).into(),
                kind: (*kind).into(),
            })
            .collect();
        crate::save_pairing_at(root, &host).unwrap();
    }

    fn regular_path(root: &Path, origin: &str) -> PathBuf {
        root.join(format!(
            "regular-{}.json",
            crate::store::origin_digest(origin)
        ))
    }

    /// Writes a profile the way a binary from before names did.
    fn legacy_regular(root: &Path, origin: &str, hostname: Option<&str>) {
        let mut profile = serde_json::json!({"origin": origin, "api_key": "k"});
        if let Some(hostname) = hostname {
            profile["hostname"] = hostname.into();
        }
        mold_secure::identity_file::write_private_file(
            &regular_path(root, origin),
            &serde_json::to_vec(&profile).unwrap(),
            false,
        )
        .unwrap();
    }

    fn resolved(root: &Path, input: &str) -> Option<(String, Match)> {
        resolve_at(root, input)
            .unwrap()
            .map(|found| (found.origin, found.via))
    }

    #[test]
    fn a_profile_from_before_names_still_loads_and_lists() {
        let (_dir, root) = store();
        legacy_regular(&root, "http://100.64.0.9:7680", None);
        let raw = crate::read_private(&regular_path(&root, "http://100.64.0.9:7680")).unwrap();
        let host: crate::RegularHost = serde_json::from_slice(&raw).unwrap();
        assert_eq!(host.name, None);
        let hosts = list_at(&root).unwrap();
        assert_eq!(hosts.len(), 1);
        assert_eq!(hosts[0].name, "host-100-64-0-9");
        assert_eq!(hosts[0].mode, ConnectionMode::Regular);
        assert!(!hosts[0].named && !hosts[0].ambiguous);
    }

    #[test]
    fn legacy_names_derive_from_the_origin() {
        assert_eq!(
            derived_name("https://abc123-7680.proxy.runpod.net", None),
            "runpod-abc123"
        );
        assert_eq!(
            derived_name("https://73b0aa11ffee.mold-link.urandom.io", None),
            "relay-73b0aa11"
        );
        assert_eq!(
            derived_name("https://gpu-box.tailnet.ts.net", None),
            "gpu-box"
        );
        assert_eq!(
            derived_name("https://x.invalid", Some("Studio Mac")),
            "studio-mac"
        );
    }

    /// A derived name that collides is never suffixed (a suffix would move
    /// to another host after `rm`): both rows are ambiguous, and resolving
    /// the name is an error naming both origins.
    #[test]
    fn colliding_derived_names_are_ambiguous_not_suffixed() {
        let (_dir, root) = store();
        legacy_regular(&root, "http://10.0.0.8:7680", Some("lab"));
        legacy_regular(&root, "http://10.0.0.9:7680", Some("lab"));
        let hosts = list_at(&root).unwrap();
        assert!(hosts
            .iter()
            .all(|h| h.name == "lab" && h.ambiguous && !h.named));
        let error = resolve_at(&root, "lab").unwrap_err().to_string();
        assert!(error.contains("http://10.0.0.8:7680"), "{error}");
        assert!(error.contains("http://10.0.0.9:7680"), "{error}");
        assert!(error.contains("mold host use"), "{error}");
        assert!(set_default_at(&root, "lab").is_err());
        assert!(remove_at(&root, "lab").is_err());
        // Persisting skips ambiguous names; nothing is rewritten.
        assert!(persist_derived_names_at(&root).unwrap().is_empty());
    }

    /// A stored name never moves: a derived one sharing it is the ambiguous
    /// row, and the name resolves to the stored holder.
    #[test]
    fn a_stored_name_owns_its_name_over_a_derived_one() {
        let (_dir, root) = store();
        secure(&root, "https://zz.invalid", Some("workstation"), &[]);
        secure(&root, "https://workstation.invalid", None, &[]);
        let hosts = list_at(&root).unwrap();
        let stored = hosts
            .iter()
            .find(|h| h.origin == "https://zz.invalid")
            .unwrap();
        assert!(stored.named && !stored.ambiguous);
        let derived = hosts
            .iter()
            .find(|h| h.origin == "https://workstation.invalid")
            .unwrap();
        assert_eq!(derived.name, "workstation");
        assert!(derived.ambiguous && !derived.named);
        assert_eq!(
            resolved(&root, "workstation"),
            Some(("https://zz.invalid".into(), Match::Name))
        );
    }

    /// Explicit host commands persist unambiguous derived names, keeping
    /// every other field, so a later removal cannot rename anything.
    #[test]
    fn persisting_derived_names_makes_them_stable() {
        let (_dir, root) = store();
        legacy_regular(&root, "http://10.0.0.8:7680", Some("lab"));
        legacy_regular(&root, "http://10.0.0.9:7680", Some("other"));
        let named = persist_derived_names_at(&root).unwrap();
        assert_eq!(named.len(), 2);
        let raw = crate::read_private(&regular_path(&root, "http://10.0.0.8:7680")).unwrap();
        let profile: serde_json::Value = serde_json::from_slice(&raw).unwrap();
        assert_eq!(
            profile,
            serde_json::json!({"origin": "http://10.0.0.8:7680", "api_key": "k", "hostname": "lab", "name": "lab"})
        );
        // A newcomer reporting the same hostname cannot take the name.
        legacy_regular(&root, "http://10.0.0.10:7680", Some("lab"));
        assert_eq!(
            resolved(&root, "lab"),
            Some(("http://10.0.0.8:7680".into(), Match::Name))
        );
        assert!(list_at(&root)
            .unwrap()
            .iter()
            .all(|h| h.named || h.origin == "http://10.0.0.10:7680"));
    }

    #[test]
    fn slugs_never_look_like_addresses() {
        assert_eq!(slug("Workstation").as_deref(), Some("workstation"));
        assert_eq!(slug("my  box.local").as_deref(), Some("my-box-local"));
        assert_eq!(slug("10.0.0.8").as_deref(), Some("host-10-0-0-8"));
        assert_eq!(slug("--"), None);
        assert_eq!(slug("localhost"), None);
        assert!(validate_name("a.b").is_err());
        assert!(validate_name("UPPER").is_err());
        assert!(validate_name("9lives").is_err());
        assert!(validate_name("workstation").is_ok());
    }

    #[test]
    fn list_use_and_remove_round_trip() {
        let (_dir, root) = store();
        secure(&root, "https://one.invalid", Some("one"), &[]);
        legacy_regular(&root, "http://10.0.0.8:7680", None);
        assert_eq!(default_at(&root).unwrap(), None);
        assert!(set_default_at(&root, "missing").is_err());
        assert_eq!(
            set_default_at(&root, "ONE").unwrap().origin,
            "https://one.invalid"
        );
        assert_eq!(default_at(&root).unwrap().unwrap().name, "one");
        assert!(clear_default_at(&root).unwrap());
        assert!(!clear_default_at(&root).unwrap());
        set_default_at(&root, "one").unwrap();

        let removed = remove_at(&root, "one").unwrap();
        assert_eq!(removed.origin, "https://one.invalid");
        assert_eq!(
            default_at(&root).unwrap(),
            None,
            "removing clears its default"
        );
        assert!(!root.join(DEFAULT_FILE).exists());
        let left = list_at(&root).unwrap();
        assert_eq!(left.len(), 1);
        assert_eq!(left[0].origin, "http://10.0.0.8:7680");
        remove_at(&root, &left[0].name).unwrap();
        assert!(list_at(&root).unwrap().is_empty());
        assert!(remove_at(&root, "one").is_err());
    }

    #[test]
    fn remove_deletes_both_modes_and_the_selection() {
        let (_dir, root) = store();
        let origin = "https://both.invalid";
        secure(&root, origin, Some("both"), &[]);
        legacy_regular(&root, origin, None);
        let digest = crate::store::origin_digest(origin);
        mold_secure::identity_file::write_private_file(
            &root.join(format!("regular-{digest}.mode")),
            &serde_json::to_vec(&ConnectionMode::Regular).unwrap(),
            false,
        )
        .unwrap();
        let hosts = list_at(&root).unwrap();
        assert_eq!(hosts.len(), 1);
        assert_eq!(hosts[0].mode, ConnectionMode::Regular);
        assert_eq!(hosts[0].name, "both", "a name on either profile counts");
        remove_at(&root, "both").unwrap();
        assert_eq!(std::fs::read_dir(&root).unwrap().count(), 0);
    }

    const RELAY: &str = "https://73b0aa11ffee.mold-link.urandom.io";

    fn workstation(root: &Path) {
        secure(
            root,
            RELAY,
            Some("workstation"),
            &[
                ("tailscale", "http://100.105.134.43:7680"),
                ("tailscale", "http://[fd7a:115c:a1e0::43]:7680"),
                ("lan", "http://192.168.1.5:7680"),
                ("lan", "http://172.17.0.1:7680"),
                // Mislabelled: a "tailscale" kind outside Tailscale's ranges.
                ("tailscale", "http://10.0.0.5:7680"),
            ],
        );
    }

    #[test]
    fn names_and_origins_resolve_and_say_how() {
        let (_dir, root) = store();
        workstation(&root);
        legacy_regular(&root, "http://10.0.0.8:7680", None);
        for input in ["workstation", "Workstation"] {
            assert_eq!(resolved(&root, input), Some((RELAY.into(), Match::Name)));
        }
        assert_eq!(resolved(&root, RELAY), Some((RELAY.into(), Match::Origin)));
        assert_eq!(
            resolved(&root, "10.0.0.8"),
            Some(("http://10.0.0.8:7680".into(), Match::Origin))
        );
    }

    /// Only an advertised Tailscale address redirects to the host's paired
    /// origin; the CLI is told so it can print the redirect.
    #[test]
    fn a_tailscale_address_finds_the_host_paired_under_its_relay_origin() {
        let (_dir, root) = store();
        workstation(&root);
        for input in [
            "100.105.134.43",
            "100.105.134.43:7680",
            "http://100.105.134.43:7680",
            "fd7a:115c:a1e0::43",
        ] {
            assert_eq!(
                resolved(&root, input),
                Some((RELAY.into(), Match::Tailscale)),
                "{input}"
            );
        }
    }

    /// A LAN or Docker address a host once advertised is somebody's machine
    /// on this network now, never a reason to send the request elsewhere.
    #[test]
    fn lan_and_unlisted_addresses_are_used_as_typed() {
        let (_dir, root) = store();
        workstation(&root);
        for input in [
            "192.168.1.5",
            "192.168.1.5:7680",
            "172.17.0.1",
            "10.0.0.5",
            "100.105.134.44",
            "localhost",
            "http://127.0.0.1:7680",
            "",
        ] {
            assert_eq!(resolved(&root, input), None, "{input:?}");
        }
    }

    /// Same-machine matching needs neither a scheme nor a port: an explicit
    /// `https://` or port means exactly that origin.
    #[test]
    fn a_scheme_or_port_disables_same_machine_matching() {
        let (_dir, root) = store();
        workstation(&root);
        for input in [
            "https://100.105.134.43",
            "http://100.105.134.43",
            "100.105.134.43:9999",
        ] {
            assert_eq!(resolved(&root, input), None, "{input}");
        }
        let (_dir, root) = store();
        secure(
            &root,
            "https://a.invalid",
            Some("a"),
            &[("tailscale", "http://100.64.1.1:7690")],
        );
        // No port: the one Tailscale endpoint on that machine.
        assert_eq!(
            resolved(&root, "100.64.1.1"),
            Some(("https://a.invalid".into(), Match::Tailscale))
        );
    }

    #[test]
    fn an_ambiguous_machine_is_no_match() {
        let (_dir, root) = store();
        secure(
            &root,
            "https://a.invalid",
            Some("a"),
            &[("tailscale", "http://100.64.1.1:7680")],
        );
        secure(
            &root,
            "https://b.invalid",
            Some("b"),
            &[("tailscale", "http://100.64.1.1:7690")],
        );
        assert_eq!(
            resolved(&root, "100.64.1.1"),
            Some(("https://a.invalid".into(), Match::Tailscale))
        );
        assert_eq!(
            resolved(&root, "100.64.1.1:7690"),
            Some(("https://b.invalid".into(), Match::Tailscale))
        );
        secure(
            &root,
            "https://c.invalid",
            Some("c"),
            &[("tailscale", "http://100.64.1.2:7680")],
        );
        secure(
            &root,
            "https://d.invalid",
            Some("d"),
            &[("tailscale", "http://100.64.1.2:7680")],
        );
        assert_eq!(resolved(&root, "100.64.1.2"), None);
    }

    /// With an unreadable profile, a bare name that matches nothing might be
    /// that profile's host: refuse rather than look it up in DNS. Addresses
    /// and clean stores keep the plain fall-through.
    #[test]
    fn an_unreadable_profile_stops_a_bare_name_falling_through_to_dns() {
        let (_dir, root) = store();
        assert_eq!(resolved(&root, "hal9000"), None, "no store: plain DNS");
        workstation(&root);
        assert_eq!(resolved(&root, "hal9000"), None, "clean store: plain DNS");
        let broken = root.join(format!("{}.json", "ab".repeat(32)));
        mold_secure::identity_file::write_private_file(&broken, b"not json", false).unwrap();
        let error = resolve_at(&root, "hal9000").unwrap_err().to_string();
        assert!(error.contains(&broken.display().to_string()), "{error}");
        // A readable host still resolves, and an address is used as typed.
        assert_eq!(
            resolved(&root, "workstation"),
            Some((RELAY.into(), Match::Name))
        );
        assert_eq!(resolved(&root, "hal9000.tailnet.ts.net"), None);
        assert_eq!(resolved(&root, "http://hal9000:7680"), None);
    }

    /// Re-pairing the same server (same instance id) under a new origin
    /// replaces the old entry: it keeps the old name and default, and the
    /// stale origin's credentials are deleted.
    #[test]
    fn a_re_pair_under_a_new_origin_replaces_the_old_entry() {
        let (_dir, root) = store();
        let mut old = crate::prepare_bootstrap("hosts", "cli")
            .unwrap()
            .profile("https://old.invalid")
            .unwrap();
        old.name = Some("workstation".into());
        old.instance_id = Some("instance-1".into());
        crate::save_pairing_at(&root, &old).unwrap();
        set_default_at(&root, "workstation").unwrap();
        let (name, superseded) = name_for_pairing_at(
            &root,
            "Other Label",
            "https://new.invalid",
            Some("instance-1"),
            ConnectionMode::Secure,
        )
        .unwrap();
        assert_eq!(name, "workstation");
        let superseded = superseded.expect("same instance at another origin");
        assert_eq!(superseded.origin, "https://old.invalid");
        let mut new = crate::prepare_bootstrap("hosts", "cli")
            .unwrap()
            .profile("https://new.invalid")
            .unwrap();
        new.name = Some(name);
        new.instance_id = Some("instance-1".into());
        crate::save_pairing_at(&root, &new).unwrap();
        retire_at(
            &root,
            &superseded,
            "https://new.invalid",
            ConnectionMode::Secure,
        )
        .unwrap();
        let hosts = list_at(&root).unwrap();
        assert_eq!(hosts.len(), 1);
        assert_eq!(hosts[0].origin, "https://new.invalid");
        assert_eq!(hosts[0].name, "workstation");
        assert_eq!(
            default_at(&root).unwrap().unwrap().origin,
            "https://new.invalid",
            "the default follows the host"
        );
        // Same origin, or no instance id: an ordinary name, nothing replaced.
        let (_, none) = name_for_pairing_at(
            &root,
            "x",
            "https://new.invalid",
            Some("instance-1"),
            ConnectionMode::Secure,
        )
        .unwrap();
        assert!(none.is_none());
        let (name, none) = name_for_pairing_at(
            &root,
            "Lab",
            "https://lab.invalid",
            None,
            ConnectionMode::Secure,
        )
        .unwrap();
        // A Regular route to the same server is a second pairing, kept.
        let (_, none_regular) = name_for_pairing_at(
            &root,
            "x",
            "http://10.1.2.3:7680",
            Some("instance-1"),
            ConnectionMode::Regular,
        )
        .unwrap();
        assert!(none_regular.is_none());
        assert_eq!((name.as_str(), none), ("lab", None));
    }

    /// A default whose profile cannot be read is an error naming the file,
    /// never a silent fall back to localhost; a default whose host is gone
    /// is simply unset.
    #[cfg(unix)]
    #[test]
    fn an_unreadable_default_is_an_error_not_localhost() {
        use std::os::unix::fs::PermissionsExt;
        let (_dir, root) = store();
        secure(&root, "https://one.invalid", Some("one"), &[]);
        set_default_at(&root, "one").unwrap();
        let profile = root.join(format!(
            "{}.json",
            crate::store::origin_digest("https://one.invalid")
        ));
        std::fs::set_permissions(&profile, std::fs::Permissions::from_mode(0o644)).unwrap();
        let error = format!("{:#}", default_at(&root).unwrap_err());
        assert!(error.contains(&profile.display().to_string()), "{error}");
        assert!(error.contains("chmod 600"), "{error}");
        assert!(error.contains("mold host use --clear"), "{error}");
        std::fs::set_permissions(&profile, std::fs::Permissions::from_mode(0o600)).unwrap();
        assert_eq!(default_at(&root).unwrap().unwrap().name, "one");
        std::fs::remove_file(&profile).unwrap();
        assert_eq!(default_at(&root).unwrap(), None);
    }

    /// Re-pairing one mode at a new origin retires only that mode: the
    /// other mode's profile stays, becomes the selected mode, and gives up
    /// the name the new pairing took.
    #[test]
    fn retiring_one_mode_keeps_the_other() {
        let (_dir, root) = store();
        let old = "https://both.invalid";
        let mut secure_host = crate::prepare_bootstrap("hosts", "cli")
            .unwrap()
            .profile(old)
            .unwrap();
        secure_host.name = Some("workstation".into());
        secure_host.instance_id = Some("inst-1".into());
        crate::save_pairing_at(&root, &secure_host).unwrap();
        let digest = crate::store::origin_digest(old);
        mold_secure::identity_file::write_private_file(
            &root.join(format!("regular-{digest}.json")),
            &serde_json::to_vec(&serde_json::json!({
                "origin": old, "api_key": "k", "name": "workstation", "instance_id": "inst-1"
            }))
            .unwrap(),
            false,
        )
        .unwrap();
        mold_secure::identity_file::write_private_file(
            &root.join(format!("regular-{digest}.mode")),
            &serde_json::to_vec(&ConnectionMode::Secure).unwrap(),
            false,
        )
        .unwrap();
        set_default_at(&root, "workstation").unwrap();
        let (name, superseded) = name_for_pairing_at(
            &root,
            "x",
            "https://new.invalid",
            Some("inst-1"),
            ConnectionMode::Secure,
        )
        .unwrap();
        let superseded = superseded.unwrap();
        assert_eq!(
            (name.as_str(), superseded.origin.as_str()),
            ("workstation", old)
        );
        let mut new = crate::prepare_bootstrap("hosts", "cli")
            .unwrap()
            .profile("https://new.invalid")
            .unwrap();
        new.name = Some(name);
        new.instance_id = Some("inst-1".into());
        crate::save_pairing_at(&root, &new).unwrap();
        retire_at(
            &root,
            &superseded,
            "https://new.invalid",
            ConnectionMode::Secure,
        )
        .unwrap();

        assert!(
            !root.join(format!("{digest}.json")).exists(),
            "the Secure profile went"
        );
        let remaining = list_at(&root).unwrap();
        let old_entry = remaining.iter().find(|h| h.origin == old).unwrap();
        assert_eq!(
            old_entry.mode,
            ConnectionMode::Regular,
            "selector follows what is left"
        );
        assert_eq!(old_entry.profiles, vec![ConnectionMode::Regular]);
        assert_ne!(
            old_entry.name, "workstation",
            "the new pairing owns the name"
        );
        let raw = crate::read_private(&root.join(format!("regular-{digest}.json"))).unwrap();
        let profile: serde_json::Value = serde_json::from_slice(&raw).unwrap();
        assert_eq!(profile["api_key"], "k", "the Regular credential is kept");
        assert_eq!(
            resolved(&root, "workstation"),
            Some(("https://new.invalid".into(), Match::Name))
        );
        assert_eq!(
            default_at(&root).unwrap().unwrap().origin,
            "https://new.invalid"
        );
    }

    /// The replacement is found by (instance, mode) across every entry, not
    /// by whichever entry with that instance happens to list first.
    #[test]
    fn the_superseded_entry_matches_instance_and_mode_together() {
        let (_dir, root) = store();
        let mut first = crate::prepare_bootstrap("hosts", "cli")
            .unwrap()
            .profile("https://a-secure.invalid")
            .unwrap();
        first.name = Some("secure-route".into());
        first.instance_id = Some("inst-1".into());
        crate::save_pairing_at(&root, &first).unwrap();
        let regular = "http://10.0.0.8:7680";
        mold_secure::identity_file::write_private_file(
            &root.join(format!(
                "regular-{}.json",
                crate::store::origin_digest(regular)
            )),
            &serde_json::to_vec(&serde_json::json!({
                "origin": regular, "api_key": "k", "name": "direct-route", "instance_id": "inst-1"
            }))
            .unwrap(),
            false,
        )
        .unwrap();
        let (name, superseded) = name_for_pairing_at(
            &root,
            "x",
            "http://10.0.0.9:7680",
            Some("inst-1"),
            ConnectionMode::Regular,
        )
        .unwrap();
        assert_eq!(name, "direct-route");
        assert_eq!(superseded.unwrap().origin, regular);
    }

    #[test]
    fn a_taken_name_error_does_not_name_a_flag() {
        let (_dir, root) = store();
        secure(&root, "https://one.invalid", Some("one"), &[]);
        let error = require_free_name_at(&root, "one", "https://two.invalid")
            .unwrap_err()
            .to_string();
        assert!(error.contains("already named 'one'"), "{error}");
        assert!(!error.contains("--"), "{error}");
        assert_eq!(
            require_free_name_at(&root, "One", "https://one.invalid").unwrap(),
            "one"
        );
    }

    /// A store directory that cannot be read at all stops a bare name.
    #[cfg(unix)]
    #[test]
    fn an_unreadable_store_stops_a_bare_name() {
        use std::os::unix::fs::PermissionsExt;
        let (_dir, root) = store();
        workstation(&root);
        std::fs::set_permissions(&root, std::fs::Permissions::from_mode(0o000)).unwrap();
        let result = resolve_at(&root, "hal9000");
        let address = resolve_at(&root, "http://hal9000:7680");
        std::fs::set_permissions(&root, std::fs::Permissions::from_mode(0o700)).unwrap();
        let error = result.unwrap_err().to_string();
        assert!(error.contains(&root.display().to_string()), "{error}");
        assert_eq!(address.unwrap(), None, "an address is used as typed");
    }

    /// Derived names do not depend on what else is saved, so removing one
    /// host never renames another.
    #[test]
    fn removing_a_host_never_renames_another() {
        let (_dir, root) = store();
        legacy_regular(&root, "http://10.0.0.8:7680", Some("lab"));
        legacy_regular(&root, "http://10.0.0.9:7680", Some("lab"));
        legacy_regular(&root, "http://10.0.0.7:7680", Some("bench"));
        legacy_regular(&root, "http://10.0.0.6:7680", Some("desk"));
        let before: Vec<(String, String)> = list_at(&root)
            .unwrap()
            .into_iter()
            .map(|h| (h.origin, h.name))
            .collect();
        remove_at(&root, "bench").unwrap();
        let after: Vec<(String, String)> = list_at(&root)
            .unwrap()
            .into_iter()
            .map(|h| (h.origin, h.name))
            .collect();
        let expected: Vec<_> = before
            .into_iter()
            .filter(|(origin, _)| origin != "http://10.0.0.7:7680")
            .collect();
        assert_eq!(after, expected);
        assert!(list_at(&root)
            .unwrap()
            .iter()
            .filter(|h| h.name == "lab")
            .all(|h| h.ambiguous));
    }

    /// `https://IP` with no port is that exact origin, not the saved Regular
    /// `http://IP:7680` on the same machine.
    #[test]
    fn https_without_a_port_is_not_the_saved_http_origin() {
        let (_dir, root) = store();
        legacy_regular(&root, "http://100.64.5.5:7680", None);
        legacy_regular(&root, "http://10.0.0.5:7680", None);
        for input in ["https://100.64.5.5", "https://10.0.0.5"] {
            assert_eq!(resolved(&root, input), None, "{input}");
        }
        assert_eq!(
            resolved(&root, "10.0.0.5"),
            Some(("http://10.0.0.5:7680".into(), Match::Origin))
        );
    }

    #[test]
    fn instance_ids_and_free_names() {
        let (_dir, root) = store();
        let mut host = crate::prepare_bootstrap("hosts", "cli")
            .unwrap()
            .profile("https://inst.invalid")
            .unwrap();
        host.name = Some("workstation".into());
        host.instance_id = Some("instance-1".into());
        crate::save_pairing_at(&root, &host).unwrap();
        assert_eq!(
            find_by_instance_id_at(&root, "instance-1")
                .unwrap()
                .unwrap()
                .origin,
            "https://inst.invalid"
        );
        assert_eq!(find_by_instance_id_at(&root, "").unwrap(), None);
        assert_eq!(
            available_name_at(&root, "Workstation", "https://other.invalid").unwrap(),
            "workstation-2"
        );
        assert_eq!(
            available_name_at(&root, "Renamed Box", "https://inst.invalid").unwrap(),
            "workstation",
            "re-pairing keeps the saved name"
        );
        assert_eq!(
            available_name_at(&root, "", "https://10.9.8.7").unwrap(),
            "host-10-9-8-7"
        );
        assert!(!name_is_free_at(&root, "workstation", "https://other.invalid").unwrap());
        assert!(name_is_free_at(&root, "workstation", "https://inst.invalid").unwrap());
    }

    #[test]
    fn normalize_address_matches_the_cli_rule() {
        for (input, expected) in [
            ("workstation", "http://workstation:7680"),
            ("workstation:8080", "http://workstation:8080"),
            ("http://workstation", "http://workstation"),
            ("https://workstation/", "https://workstation"),
            ("::1", "http://[::1]:7680"),
        ] {
            assert_eq!(normalize_address(input), expected, "{input}");
        }
        assert!(typed_port("10.0.0.1:7680"));
        assert!(typed_port("http://10.0.0.1:7680"));
        assert!(!typed_port("10.0.0.1"));
        assert!(!typed_port("http://10.0.0.1"));
        assert!(!typed_port("fd00::1"));
        assert!(is_tailscale_address("100.64.0.1"));
        assert!(is_tailscale_address("100.127.255.255"));
        assert!(!is_tailscale_address("100.128.0.1"));
        assert!(is_tailscale_address("fd7a:115c:a1e0::1"));
        assert!(!is_tailscale_address("fd7a:115c:a1e1::1"));
        assert!(!is_tailscale_address("workstation"));
    }
}
