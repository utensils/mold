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
//! (`host-<ip>` for an address). Stored names win over derived ones; any
//! remaining collision takes a `-2`, `-3`, ... suffix in origin order.
//!
//! Listing reads metadata only. Credentials are never parsed out of the
//! profile here, and a profile that cannot be read is skipped rather than
//! failing every command that resolves a host.
use crate::{ConnectionMode, Endpoint};
use anyhow::{bail, Context, Result};
use serde::Deserialize;
use std::{
    collections::{BTreeMap, BTreeSet},
    path::Path,
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
    pub mode: ConnectionMode,
    pub instance_id: Option<String>,
    pub hostname: Option<String>,
    /// Endpoint URLs the host advertised when it was paired (Secure only).
    pub endpoints: Vec<String>,
    /// False when `name` was derived because the profile stores none (or a
    /// stored name lost a collision).
    pub named: bool,
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
    let mut entries: BTreeMap<String, Entry> = BTreeMap::new();
    if !root.try_exists()? {
        return Ok(Vec::new());
    }
    let mut modes = BTreeMap::new();
    for item in std::fs::read_dir(root)? {
        let path = item?.path();
        let Some(file) = path.file_name().and_then(|v| v.to_str()) else {
            continue;
        };
        if let Some(digest) = file
            .strip_prefix("regular-")
            .and_then(|v| v.strip_suffix(".mode"))
        {
            if let Ok(bytes) = crate::read_private(&path) {
                if let Ok(mode) = serde_json::from_slice::<ConnectionMode>(&bytes) {
                    modes.insert(digest.to_string(), mode);
                }
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
        let Some(meta) = read_meta(&path) else {
            continue;
        };
        // A profile is filed under the digest of its own origin; one that
        // claims another origin is not this host's credential.
        if crate::store::origin_digest(&meta.origin) != digest {
            continue;
        }
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
    Ok(assign_names(entries))
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
                .map(|m| m.endpoints.iter().map(|e| e.url.clone()).collect())
                .unwrap_or_default();
            Some((
                SavedHost {
                    name: String::new(),
                    instance_id: pick(|m| &m.instance_id),
                    hostname,
                    endpoints,
                    mode,
                    origin,
                    named: false,
                },
                stored,
            ))
        })
        .collect();
    let mut taken = BTreeSet::new();
    for (host, stored) in hosts.iter_mut() {
        if let Some(name) = stored {
            if taken.insert(name.clone()) {
                host.name = name.clone();
                host.named = true;
            }
        }
    }
    for (host, _) in hosts.iter_mut().filter(|(h, _)| h.name.is_empty()) {
        let base = derived_name(&host.origin, host.hostname.as_deref());
        host.name = next_free(&base, &taken);
        taken.insert(host.name.clone());
    }
    hosts.into_iter().map(|(host, _)| host).collect()
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

/// The saved host with this name, case-insensitively.
pub fn find(name: &str) -> Result<Option<SavedHost>> {
    find_at(&crate::default_store(), name)
}

pub fn find_at(root: &Path, name: &str) -> Result<Option<SavedHost>> {
    let name = name.trim();
    Ok(list_at(root)?
        .into_iter()
        .find(|h| h.name.eq_ignore_ascii_case(name)))
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
/// name `origin` already has when one is stored (re-pairing keeps it), else
/// the label's slug when free, else the first free `-N` suffix. Callers that
/// took the name from the user should refuse a collision instead; see
/// [`require_free_name`].
pub fn available_name(base: &str, origin: &str) -> Result<String> {
    available_name_at(&crate::default_store(), base, origin)
}

pub fn available_name_at(root: &Path, base: &str, origin: &str) -> Result<String> {
    let hosts = list_at(root)?;
    if let Some(saved) = hosts.iter().find(|h| h.origin == origin && h.named) {
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
    let digest = crate::store::origin_digest(&host.origin);
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
    if read_default_origin(root)?.as_deref() == Some(host.origin.as_str()) {
        clear_default_at(root)?;
    }
    Ok(host)
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

pub fn default_at(root: &Path) -> Result<Option<SavedHost>> {
    match read_default_origin(root)? {
        Some(origin) => find_by_origin_at(root, &origin),
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

/// The saved origin `input` names, or `None` to use the address as typed.
///
/// In order: a saved host name (case-insensitive); a saved origin; a saved
/// Secure host one of whose advertised endpoints is exactly this address;
/// and, when no port was typed, the one saved host whose origin or endpoint
/// is on this machine name or IP. So a Tailscale IP finds the host paired
/// under its relay origin. Loopback addresses are never redirected, and an
/// ambiguous match is no match.
pub fn resolve(input: &str) -> Option<String> {
    resolve_at(&crate::default_store(), input)
}

pub fn resolve_at(root: &Path, input: &str) -> Option<String> {
    let input = input.trim();
    if input.is_empty() {
        return None;
    }
    let hosts = list_at(root).ok()?;
    if let Some(host) = hosts.iter().find(|h| h.name.eq_ignore_ascii_case(input)) {
        return Some(host.origin.clone());
    }
    let address = normalize_address(input);
    if let Some(host) = hosts.iter().find(|h| h.origin == address) {
        return Some(host.origin.clone());
    }
    let url = reqwest::Url::parse(&address).ok()?;
    let machine = url
        .host_str()?
        .trim_matches(['[', ']'])
        .to_ascii_lowercase();
    if is_loopback(&machine) {
        return None;
    }
    let unique = |matches: Vec<&SavedHost>| match matches.as_slice() {
        [one] => Some(one.origin.clone()),
        _ => None,
    };
    let exact = hosts
        .iter()
        .filter(|h| h.endpoints.iter().any(|e| normalize_address(e) == address))
        .collect();
    if let Some(origin) = unique(exact) {
        return Some(origin);
    }
    if typed_port(input) {
        return None;
    }
    let same_machine = hosts
        .iter()
        .filter(|h| {
            std::iter::once(&h.origin)
                .chain(&h.endpoints)
                .any(|e| machine_of(e).as_deref() == Some(machine.as_str()))
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

/// Whether the user spelled a port, which then has to match exactly.
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
    let name = name.trim().to_ascii_lowercase();
    validate_name(&name)?;
    if !name_is_free(&name, origin)? {
        bail!("a saved host is already named '{name}'; choose another with --name")
    }
    Ok(name)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn store() -> (tempfile::TempDir, std::path::PathBuf) {
        let dir = tempfile::tempdir().unwrap();
        let root = dir.path().join("secure/clients");
        (dir, root)
    }

    fn secure(root: &Path, origin: &str, name: Option<&str>, endpoints: &[&str]) {
        let mut host = crate::prepare_bootstrap("hosts", "cli")
            .unwrap()
            .profile(origin)
            .unwrap();
        host.name = name.map(str::to_owned);
        host.endpoints = endpoints
            .iter()
            .map(|url| Endpoint {
                url: (*url).into(),
                kind: "lan".into(),
            })
            .collect();
        crate::save_pairing_at(root, &host).unwrap();
    }

    /// Writes a profile the way a binary from before names did.
    fn legacy_regular(root: &Path, origin: &str) {
        let digest = crate::store::origin_digest(origin);
        mold_secure::identity_file::write_private_file(
            &root.join(format!("regular-{digest}.json")),
            &serde_json::to_vec(&serde_json::json!({"origin": origin, "api_key": "k"})).unwrap(),
            false,
        )
        .unwrap();
    }

    #[test]
    fn a_profile_from_before_names_still_loads_and_lists() {
        let (_dir, root) = store();
        legacy_regular(&root, "http://100.64.0.9:7680");
        let raw = crate::read_private(&root.join(format!(
            "regular-{}.json",
            crate::store::origin_digest("http://100.64.0.9:7680")
        )))
        .unwrap();
        let host: crate::RegularHost = serde_json::from_slice(&raw).unwrap();
        assert_eq!(host.name, None);
        let hosts = list_at(&root).unwrap();
        assert_eq!(hosts.len(), 1);
        assert_eq!(hosts[0].name, "host-100-64-0-9");
        assert_eq!(hosts[0].mode, ConnectionMode::Regular);
        assert!(!hosts[0].named);
    }

    #[test]
    fn legacy_names_derive_from_the_origin_and_never_steal_a_stored_name() {
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
        let (_dir, root) = store();
        secure(&root, "https://workstation.invalid", None, &[]);
        secure(&root, "https://zz.invalid", Some("workstation"), &[]);
        let hosts = list_at(&root).unwrap();
        let named = hosts.iter().find(|h| h.named).unwrap();
        assert_eq!(
            (named.name.as_str(), named.origin.as_str()),
            ("workstation", "https://zz.invalid")
        );
        let derived = hosts.iter().find(|h| !h.named).unwrap();
        assert_eq!(derived.name, "workstation-2");
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
        legacy_regular(&root, "http://10.0.0.8:7680");
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
        legacy_regular(&root, origin);
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

    #[test]
    fn resolve_finds_names_origins_and_advertised_endpoints() {
        let (_dir, root) = store();
        let relay = "https://73b0aa11ffee.mold-link.urandom.io";
        secure(
            &root,
            relay,
            Some("workstation"),
            &["http://100.105.134.43:7680", "http://192.168.1.5:4321"],
        );
        legacy_regular(&root, "http://10.0.0.8:7680");
        for input in [
            "workstation",
            "Workstation",
            relay,
            "100.105.134.43",
            "100.105.134.43:7680",
            "http://100.105.134.43:7680",
            "http://100.105.134.43",
            "192.168.1.5",
            "192.168.1.5:4321",
        ] {
            assert_eq!(resolve_at(&root, input).as_deref(), Some(relay), "{input}");
        }
        assert_eq!(
            resolve_at(&root, "10.0.0.8").as_deref(),
            Some("http://10.0.0.8:7680")
        );
        for input in [
            "192.168.1.5:9999",
            "100.105.134.44",
            "localhost",
            "http://127.0.0.1:7680",
            "unknown-name",
            "",
        ] {
            assert_eq!(resolve_at(&root, input), None, "{input}");
        }
    }

    #[test]
    fn an_ambiguous_machine_is_no_match() {
        let (_dir, root) = store();
        secure(
            &root,
            "https://a.invalid",
            Some("a"),
            &["http://10.1.1.1:7680"],
        );
        secure(
            &root,
            "https://b.invalid",
            Some("b"),
            &["http://10.1.1.1:7690"],
        );
        assert_eq!(
            resolve_at(&root, "10.1.1.1"),
            Some("https://a.invalid".into())
        );
        assert_eq!(resolve_at(&root, "http://10.1.1.1"), None);
        assert_eq!(
            resolve_at(&root, "10.1.1.1:7690"),
            Some("https://b.invalid".into())
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
    }
}
