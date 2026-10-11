use crate::{PairedHost, PairingInvite};
use anyhow::{bail, Context, Result};
use sha2::{Digest, Sha256};
use std::path::{Path, PathBuf};
pub fn default_store() -> PathBuf {
    std::env::var_os("MOLD_HOME")
        .map(PathBuf::from)
        .unwrap_or_else(|| {
            std::env::var_os("HOME")
                .or_else(|| std::env::var_os("USERPROFILE"))
                .map(PathBuf::from)
                .unwrap_or_default()
                .join(".mold")
        })
        .join("secure/clients")
}
fn filename(origin: &str) -> String {
    format!("{}.json", hex::encode(Sha256::digest(origin.as_bytes())))
}
pub fn load_pairing(origin: &str) -> Result<PairedHost> {
    if let Some(host) = ephemeral_pairing(origin) {
        host.validate()?;
        return Ok(host);
    }
    load_pairing_at(&default_store(), origin)
}
pub fn load_pairing_at(root: &Path, origin: &str) -> Result<PairedHost> {
    let bytes=read_private(&root.join(filename(origin))) .context("No secure pairing for this host. Use mold pair with an invitation from the host; legacy proxy credentials cannot be upgraded automatically")?;
    let host: PairedHost = serde_json::from_slice(&bytes)?;
    if host.origin != origin {
        bail!("stored pairing origin mismatch")
    }
    host.validate()?;
    Ok(host)
}
pub fn read_invitation(path: &Path) -> Result<PairingInvite> {
    Ok(serde_json::from_slice(&read_private(path)?)?)
}
pub fn save_pairing(host: &PairedHost) -> Result<PathBuf> {
    save_pairing_at(&default_store(), host)
}
pub fn save_pairing_at(root: &Path, host: &PairedHost) -> Result<PathBuf> {
    host.validate()?;
    let dest = root.join(filename(&host.origin));
    let bytes = zeroize::Zeroizing::new(serde_json::to_vec(host)?);
    mold_secure::identity_file::write_private_file(&dest, &bytes, true)?;
    Ok(dest)
}
pub fn read_private(path: &Path) -> Result<zeroize::Zeroizing<Vec<u8>>> {
    Ok(mold_secure::identity_file::read_private_file(path, 16384)?)
}

/// Detect an existing pairing without treating unreadable state as unpaired.
pub fn pairing_exists(origin: &str) -> bool {
    if ephemeral_pairing(origin).is_some() {
        return true;
    }
    match std::fs::symlink_metadata(default_store().join(filename(origin))) {
        Ok(_) => true,
        Err(e) => e.kind() != std::io::ErrorKind::NotFound,
    }
}

fn ephemeral(
) -> std::sync::MutexGuard<'static, std::collections::BTreeMap<(String, String), PairedHost>> {
    static STORE: std::sync::OnceLock<
        std::sync::Mutex<std::collections::BTreeMap<(String, String), PairedHost>>,
    > = std::sync::OnceLock::new();
    STORE
        .get_or_init(Default::default)
        .lock()
        .unwrap_or_else(|e| e.into_inner())
}
/// Import a trusted WebView's existing pairing without creating another store.
pub fn register_ephemeral(owner: &str, host: PairedHost) -> Result<()> {
    host.validate()?;
    let mut store = ephemeral();
    let key = (owner.to_string(), host.origin.clone());
    if !store.contains_key(&key) && store.len() >= 128 {
        bail!("native paired host limit exceeded")
    }
    if store.iter().any(|((other, origin), saved)| {
        other != owner && origin == &host.origin && saved.device_identity != host.device_identity
    }) {
        bail!("conflicting device identity for this origin")
    }
    store.insert(key, host);
    Ok(())
}
pub fn forget_ephemeral(owner: &str, origin: Option<&str>) {
    let mut store = ephemeral();
    let removed: Vec<String> = store
        .keys()
        .filter(|(o, u)| o == owner && origin.is_none_or(|v| v == u))
        .map(|(_, u)| u.clone())
        .collect();
    store.retain(|(o, u), _| o != owner || origin.is_some_and(|v| v != u));
    drop(store);
    for origin in removed {
        crate::forget_connection(&origin);
    }
}
fn ephemeral_pairing(origin: &str) -> Option<PairedHost> {
    ephemeral()
        .iter()
        .find(|((_, u), _)| u == origin)
        .map(|(_, h)| h.clone())
}

/// Refresh only the authority that already holds this exact device. WebView
/// credentials must never silently become a second persistent CLI store.
pub(crate) fn save_renewed_pairing(host: &PairedHost) -> Result<()> {
    let mut store = ephemeral();
    for ((_, origin), saved) in store.iter_mut() {
        if origin == &host.origin
            && saved.device_identity == host.device_identity
            && saved.grant.id == host.grant.id
        {
            saved.grant = host.grant.clone();
        }
    }
    drop(store);
    let root = default_store();
    if root.join(filename(&host.origin)).try_exists()? {
        let saved = load_pairing_at(&root, &host.origin)?;
        if saved.device_identity == host.device_identity && saved.grant.id == host.grant.id {
            save_pairing_at(&root, host)?;
        }
    }
    Ok(())
}

pub fn write_invitation(path: &Path, value: &serde_json::Value) -> Result<()> {
    let bytes = zeroize::Zeroizing::new(serde_json::to_vec(value)?);
    mold_secure::identity_file::write_private_file(path, &bytes, false)?;
    Ok(())
}
