//! Secrets: macOS Keychain via `keyring`, with a JSON-file fallback for
//! unsigned dev builds (ad-hoc signatures make the Keychain re-prompt on
//! every rebuild) and for tests. Names are constrained to a small allowlist
//! so the IPC surface can't be used as an arbitrary Keychain browser.

use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::Mutex;

const SERVICE: &str = "com.utensils.mold";

/// Secrets the webview may read/write. Anything else is rejected.
pub const ALLOWED: &[&str] = &["hf-token", "civitai-token", "remote-api-key"];

pub struct SecretStore {
    /// Try the OS keychain first; on any error fall back to the file store.
    use_keyring: bool,
    fallback_path: PathBuf,
    cache: Mutex<Option<HashMap<String, String>>>,
}

impl SecretStore {
    pub fn new(app_data_dir: PathBuf) -> Self {
        Self {
            use_keyring: true,
            fallback_path: app_data_dir.join("secrets.json"),
            cache: Mutex::new(None),
        }
    }

    /// File-only store (tests, headless CI).
    pub fn file_only(app_data_dir: PathBuf) -> Self {
        Self {
            use_keyring: false,
            ..Self::new(app_data_dir)
        }
    }

    fn check_name(name: &str) -> anyhow::Result<()> {
        anyhow::ensure!(ALLOWED.contains(&name), "unknown secret name: {name}");
        Ok(())
    }

    pub fn get(&self, name: &str) -> anyhow::Result<Option<String>> {
        Self::check_name(name)?;
        if self.use_keyring {
            match keyring::Entry::new(SERVICE, name).and_then(|e| e.get_password()) {
                Ok(v) => return Ok(Some(v)),
                Err(keyring::Error::NoEntry) => return Ok(self.file_get(name)),
                Err(e) => {
                    tracing::warn!("keychain read failed for {name} ({e}); using file store");
                }
            }
        }
        Ok(self.file_get(name))
    }

    pub fn set(&self, name: &str, value: &str) -> anyhow::Result<()> {
        Self::check_name(name)?;
        if self.use_keyring {
            match keyring::Entry::new(SERVICE, name).and_then(|e| e.set_password(value)) {
                Ok(()) => {
                    // A value may linger in the fallback from a dev build.
                    let _ = self.file_clear(name);
                    return Ok(());
                }
                Err(e) => {
                    tracing::warn!("keychain write failed for {name} ({e}); using file store");
                }
            }
        }
        self.file_set(name, value)
    }

    pub fn clear(&self, name: &str) -> anyhow::Result<()> {
        Self::check_name(name)?;
        if self.use_keyring {
            match keyring::Entry::new(SERVICE, name).and_then(|e| e.delete_credential()) {
                Ok(()) | Err(keyring::Error::NoEntry) => {}
                Err(e) => tracing::warn!("keychain delete failed for {name}: {e}"),
            }
        }
        self.file_clear(name)
    }

    // ── file fallback ─────────────────────────────────────────────────────

    fn load_file(&self) -> HashMap<String, String> {
        let mut cache = self.cache.lock().expect("secrets cache");
        if let Some(map) = cache.as_ref() {
            return map.clone();
        }
        let map: HashMap<String, String> = std::fs::read_to_string(&self.fallback_path)
            .ok()
            .and_then(|raw| serde_json::from_str(&raw).ok())
            .unwrap_or_default();
        *cache = Some(map.clone());
        map
    }

    fn save_file(&self, map: HashMap<String, String>) -> anyhow::Result<()> {
        if let Some(dir) = self.fallback_path.parent() {
            std::fs::create_dir_all(dir)?;
        }
        let tmp = self.fallback_path.with_extension("json.tmp");
        std::fs::write(&tmp, serde_json::to_vec_pretty(&map)?)?;
        std::fs::rename(&tmp, &self.fallback_path)?;
        *self.cache.lock().expect("secrets cache") = Some(map);
        Ok(())
    }

    fn file_get(&self, name: &str) -> Option<String> {
        self.load_file().get(name).cloned()
    }

    fn file_set(&self, name: &str, value: &str) -> anyhow::Result<()> {
        let mut map = self.load_file();
        map.insert(name.to_string(), value.to_string());
        self.save_file(map)
    }

    fn file_clear(&self, name: &str) -> anyhow::Result<()> {
        let mut map = self.load_file();
        map.remove(name);
        self.save_file(map)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn store() -> (SecretStore, tempfile::TempDir) {
        let dir = tempfile::tempdir().unwrap();
        (SecretStore::file_only(dir.path().to_path_buf()), dir)
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
    fn persists_across_instances() {
        let dir = tempfile::tempdir().unwrap();
        SecretStore::file_only(dir.path().to_path_buf())
            .set("civitai-token", "cv_1")
            .unwrap();
        let again = SecretStore::file_only(dir.path().to_path_buf());
        assert_eq!(again.get("civitai-token").unwrap().as_deref(), Some("cv_1"));
    }
}
