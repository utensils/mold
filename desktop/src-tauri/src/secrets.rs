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

pub struct SecretStore {
    path: PathBuf,
    cache: Mutex<Option<HashMap<String, String>>>,
}

impl SecretStore {
    pub fn new(app_data_dir: PathBuf) -> Self {
        Self {
            path: app_data_dir.join("secrets.json"),
            cache: Mutex::new(None),
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
        Ok(self.load().get(name).cloned())
    }

    pub fn set(&self, name: &str, value: &str) -> anyhow::Result<()> {
        Self::check_name(name)?;
        let mut map = self.load();
        map.insert(name.to_string(), value.to_string());
        self.save(map)
    }

    pub fn clear(&self, name: &str) -> anyhow::Result<()> {
        Self::check_name(name)?;
        let mut map = self.load();
        map.remove(name);
        self.save(map)
    }

    fn load(&self) -> HashMap<String, String> {
        let mut cache = self.cache.lock().expect("secrets cache");
        if let Some(map) = cache.as_ref() {
            return map.clone();
        }
        let map: HashMap<String, String> = std::fs::read_to_string(&self.path)
            .ok()
            .and_then(|raw| serde_json::from_str(&raw).ok())
            .unwrap_or_default();
        *cache = Some(map.clone());
        map
    }

    fn save(&self, map: HashMap<String, String>) -> anyhow::Result<()> {
        if let Some(dir) = self.path.parent() {
            std::fs::create_dir_all(dir)?;
        }
        let tmp = self.path.with_extension("json.tmp");
        std::fs::write(&tmp, serde_json::to_vec_pretty(&map)?)?;
        // Owner-only: these are plaintext credentials.
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            std::fs::set_permissions(&tmp, std::fs::Permissions::from_mode(0o600))?;
        }
        std::fs::rename(&tmp, &self.path)?;
        *self.cache.lock().expect("secrets cache") = Some(map);
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
