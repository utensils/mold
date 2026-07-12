//! Secrets: an owner-only (0600) `secrets.json` under the app data dir.
//! Plain files instead of the macOS Keychain — Keychain access prompts on
//! every ad-hoc rebuild in dev, and repeatedly interrupts users in release
//! builds after updates. Names are constrained to a small allowlist so the
//! IPC surface can't be used as an arbitrary secret browser.

use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::Mutex;

/// Secrets the webview may read/write. Anything else is rejected.
pub const ALLOWED: &[&str] = &[
    "hf-token",
    "civitai-token",
    "remote-api-key",
    "runpod-api-key",
];

/// Per-host remote keys use `remote-api-key.<host-id>`. The suffix is a slug
/// derived from the host URL (see `connection::host_id`).
const PER_HOST_PREFIX: &str = "remote-api-key.";

struct Loaded {
    map: HashMap<String, String>,
    /// The on-disk file existed but didn't parse. It is renamed to
    /// `secrets.json.corrupt` on the next write instead of being clobbered —
    /// a transient parse failure must never silently destroy credentials.
    corrupt: bool,
}

pub struct SecretStore {
    path: PathBuf,
    /// One lock around the whole read-modify-write cycle: `secret_set`
    /// commands run concurrently on Tauri's thread pool, and an unserialized
    /// load→modify→save loses whichever write lands first.
    state: Mutex<Option<Loaded>>,
}

impl SecretStore {
    pub fn new(app_data_dir: PathBuf) -> Self {
        Self {
            path: app_data_dir.join("secrets.json"),
            state: Mutex::new(None),
        }
    }

    fn check_name(name: &str) -> anyhow::Result<()> {
        if ALLOWED.contains(&name) {
            return Ok(());
        }
        if let Some(suffix) = name.strip_prefix(PER_HOST_PREFIX) {
            anyhow::ensure!(
                !suffix.is_empty()
                    && suffix
                        .bytes()
                        .all(|b| b.is_ascii_alphanumeric() || b == b'-' || b == b'_' || b == b'.'),
                "invalid per-host secret name: {name}"
            );
            return Ok(());
        }
        anyhow::bail!("unknown secret name: {name}")
    }

    pub fn get(&self, name: &str) -> anyhow::Result<Option<String>> {
        Self::check_name(name)?;
        let mut guard = self.state.lock().unwrap_or_else(|e| e.into_inner());
        Ok(Self::loaded(&self.path, &mut guard).map.get(name).cloned())
    }

    pub fn set(&self, name: &str, value: &str) -> anyhow::Result<()> {
        Self::check_name(name)?;
        let mut guard = self.state.lock().unwrap_or_else(|e| e.into_inner());
        let loaded = Self::loaded(&self.path, &mut guard);
        loaded.map.insert(name.to_string(), value.to_string());
        Self::save(&self.path, loaded)
    }

    pub fn clear(&self, name: &str) -> anyhow::Result<()> {
        Self::check_name(name)?;
        let mut guard = self.state.lock().unwrap_or_else(|e| e.into_inner());
        let loaded = Self::loaded(&self.path, &mut guard);
        loaded.map.remove(name);
        Self::save(&self.path, loaded)
    }

    fn loaded<'a>(path: &PathBuf, guard: &'a mut Option<Loaded>) -> &'a mut Loaded {
        if guard.is_none() {
            let loaded = match std::fs::read_to_string(path) {
                Err(_) => Loaded {
                    map: HashMap::new(),
                    corrupt: false,
                },
                Ok(raw) => match serde_json::from_str(&raw) {
                    Ok(map) => Loaded {
                        map,
                        corrupt: false,
                    },
                    Err(e) => {
                        tracing::warn!(
                            "secrets.json is unreadable ({e}); the original will be kept as \
                             secrets.json.corrupt"
                        );
                        Loaded {
                            map: HashMap::new(),
                            corrupt: true,
                        }
                    }
                },
            };
            *guard = Some(loaded);
        }
        guard.as_mut().expect("just initialized")
    }

    fn save(path: &PathBuf, loaded: &mut Loaded) -> anyhow::Result<()> {
        if let Some(dir) = path.parent() {
            std::fs::create_dir_all(dir)?;
        }
        if loaded.corrupt {
            // Move the unparseable original aside exactly once.
            let _ = std::fs::rename(path, path.with_extension("json.corrupt"));
            loaded.corrupt = false;
        }
        let tmp = path.with_extension("json.tmp");
        std::fs::write(&tmp, serde_json::to_vec_pretty(&loaded.map)?)?;
        // Owner-only: these are plaintext credentials.
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            std::fs::set_permissions(&tmp, std::fs::Permissions::from_mode(0o600))?;
        }
        std::fs::rename(&tmp, path)?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn store() -> (SecretStore, tempfile::TempDir) {
        let dir = tempfile::tempdir().unwrap();
        (SecretStore::new(dir.path().to_path_buf()), dir)
    }

    #[test]
    fn round_trips_and_clears() {
        let (s, _dir) = store();
        assert_eq!(s.get("hf-token").unwrap(), None);
        s.set("hf-token", "hf_abc").unwrap();
        assert_eq!(s.get("hf-token").unwrap().as_deref(), Some("hf_abc"));
        s.set("hf-token", "hf_new").unwrap();
        assert_eq!(s.get("hf-token").unwrap().as_deref(), Some("hf_new"));
        s.clear("hf-token").unwrap();
        assert_eq!(s.get("hf-token").unwrap(), None);
    }

    #[test]
    fn rejects_unknown_names() {
        let (s, _dir) = store();
        assert!(s.get("ssh-private-key").is_err());
        assert!(s.set("anything", "x").is_err());
    }

    #[test]
    fn allows_per_host_remote_key_names() {
        let (s, _dir) = store();
        s.set("remote-api-key.hal9000-7680", "k1").unwrap();
        assert_eq!(
            s.get("remote-api-key.hal9000-7680").unwrap().as_deref(),
            Some("k1")
        );
        s.clear("remote-api-key.hal9000-7680").unwrap();
        assert_eq!(s.get("remote-api-key.hal9000-7680").unwrap(), None);
    }

    #[test]
    fn rejects_malformed_per_host_names() {
        let (s, _dir) = store();
        // Empty suffix, path traversal, and other prefixes must all fail.
        assert!(s.set("remote-api-key.", "x").is_err());
        assert!(s.set("remote-api-key./etc/passwd", "x").is_err());
        assert!(s.set("hf-token.evil", "x").is_err());
    }

    #[test]
    fn persists_across_instances() {
        let dir = tempfile::tempdir().unwrap();
        SecretStore::new(dir.path().to_path_buf())
            .set("civitai-token", "cv_1")
            .unwrap();
        let again = SecretStore::new(dir.path().to_path_buf());
        assert_eq!(again.get("civitai-token").unwrap().as_deref(), Some("cv_1"));
    }

    #[test]
    fn corrupt_file_is_preserved_not_clobbered() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("secrets.json");
        std::fs::write(&path, "not json {").unwrap();
        let s = SecretStore::new(dir.path().to_path_buf());
        // Reads degrade to empty…
        assert_eq!(s.get("hf-token").unwrap(), None);
        // …and the first write moves the original aside instead of erasing it.
        s.set("hf-token", "hf_new").unwrap();
        assert_eq!(
            std::fs::read_to_string(dir.path().join("secrets.json.corrupt")).unwrap(),
            "not json {"
        );
        assert_eq!(s.get("hf-token").unwrap().as_deref(), Some("hf_new"));
    }

    #[test]
    fn concurrent_writers_do_not_lose_updates() {
        let dir = tempfile::tempdir().unwrap();
        let s = std::sync::Arc::new(SecretStore::new(dir.path().to_path_buf()));
        let handles: Vec<_> = [
            ("hf-token", "a"),
            ("civitai-token", "b"),
            ("runpod-api-key", "c"),
        ]
        .into_iter()
        .map(|(name, value)| {
            let s = s.clone();
            std::thread::spawn(move || s.set(name, value).unwrap())
        })
        .collect();
        for h in handles {
            h.join().unwrap();
        }
        assert_eq!(s.get("hf-token").unwrap().as_deref(), Some("a"));
        assert_eq!(s.get("civitai-token").unwrap().as_deref(), Some("b"));
        assert_eq!(s.get("runpod-api-key").unwrap().as_deref(), Some("c"));
    }

    #[cfg(unix)]
    #[test]
    fn secrets_file_is_owner_only() {
        use std::os::unix::fs::PermissionsExt;
        let dir = tempfile::tempdir().unwrap();
        let s = SecretStore::new(dir.path().to_path_buf());
        s.set("hf-token", "hf_abc").unwrap();
        let mode = std::fs::metadata(dir.path().join("secrets.json"))
            .unwrap()
            .permissions()
            .mode();
        assert_eq!(mode & 0o777, 0o600);
    }
}
