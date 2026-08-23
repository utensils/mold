//! Server-owned Hugging Face and Civitai credentials.
//!
//! The web UI must not make a browser profile the authority for a remote
//! server's downloads. Credentials are persisted beside the server config,
//! returned only as masked status, and resolved ahead of environment defaults.
//!
//! On-disk protection is `0600` on unix, set below. Windows has no equivalent
//! here and does not get one: the file lands under the per-user `$MOLD_HOME`,
//! whose inherited ACL already excludes other standard users, but mold writes
//! no explicit per-user DACL, so a local administrator can read it — as root
//! can on unix. Stated rather than implied, because a `#[cfg(unix)]` chmod
//! reads as a promise the other platform silently does not keep.

use std::fs::{self, OpenOptions};
use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::{Mutex, OnceLock, RwLock};

use axum::extract::Path as AxumPath;
use axum::Json;
use serde::{Deserialize, Serialize};

use crate::routes::ApiError;

const CREDENTIALS_FILE: &str = "catalog-credentials.json";
static CREDENTIALS_LOCK: OnceLock<Mutex<()>> = OnceLock::new();
static SERVER_CREDENTIALS_CACHE: OnceLock<RwLock<CatalogCredentialsCache>> = OnceLock::new();

#[derive(Clone, Debug, Default, Deserialize, PartialEq, Eq, Serialize)]
pub struct CatalogCredentials {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub hf_token: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub civitai_token: Option<String>,
}

#[derive(Debug, Default)]
struct CatalogCredentialsCache {
    path: Option<PathBuf>,
    credentials: CatalogCredentials,
}

impl CatalogCredentialsCache {
    fn get_or_load(&mut self, path: &Path) -> anyhow::Result<CatalogCredentials> {
        if self.path.as_deref() != Some(path) {
            self.credentials = load_from_path(path)?;
            self.path = Some(path.to_path_buf());
        }
        Ok(self.credentials.clone())
    }

    fn replace(&mut self, path: PathBuf, credentials: CatalogCredentials) {
        self.path = Some(path);
        self.credentials = credentials;
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
pub struct CatalogCredentialState {
    pub configured: bool,
    pub source: Option<&'static str>,
    pub masked: Option<String>,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
pub struct CatalogCredentialStatus {
    pub hf: CatalogCredentialState,
    pub civitai: CatalogCredentialState,
}

#[derive(Debug, Deserialize)]
pub struct SetCatalogCredential {
    token: String,
}

#[derive(Clone, Copy)]
enum Provider {
    Hf,
    Civitai,
}

impl Provider {
    fn parse(value: &str) -> Result<Self, ApiError> {
        match value {
            "hf" => Ok(Self::Hf),
            "civitai" => Ok(Self::Civitai),
            _ => Err(ApiError::validation("provider must be `hf` or `civitai`")),
        }
    }

    fn get_mut(self, credentials: &mut CatalogCredentials) -> &mut Option<String> {
        match self {
            Self::Hf => &mut credentials.hf_token,
            Self::Civitai => &mut credentials.civitai_token,
        }
    }
}

fn nonempty_env(name: &str) -> Option<String> {
    std::env::var(name)
        .ok()
        .map(|value| value.trim().to_string())
        .filter(|value| !value.is_empty())
}

fn resolved_token(stored: Option<String>, environment: Option<String>) -> Option<String> {
    stored
        .map(|value| value.trim().to_string())
        .filter(|value| !value.is_empty())
        .or(environment)
}

fn credentials_path() -> Result<PathBuf, ApiError> {
    mold_core::Config::mold_dir()
        .map(|dir| dir.join(CREDENTIALS_FILE))
        .ok_or_else(|| ApiError::internal("could not resolve the Mold data directory"))
}

pub(crate) fn load_from_path(path: &Path) -> anyhow::Result<CatalogCredentials> {
    match fs::read(path) {
        Ok(bytes) => Ok(serde_json::from_slice(&bytes)?),
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => {
            Ok(CatalogCredentials::default())
        }
        Err(error) => Err(error.into()),
    }
}

#[cfg(test)]
fn save_to_path(path: &Path, credentials: &CatalogCredentials) -> anyhow::Result<()> {
    let _guard = CREDENTIALS_LOCK
        .get_or_init(|| Mutex::new(()))
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    save_to_path_unlocked(path, credentials)
}

fn save_to_path_unlocked(path: &Path, credentials: &CatalogCredentials) -> anyhow::Result<()> {
    let parent = path
        .parent()
        .ok_or_else(|| anyhow::anyhow!("credential path has no parent"))?;
    fs::create_dir_all(parent)?;

    let tmp = path.with_extension(format!("json.tmp-{}", uuid::Uuid::new_v4()));
    let result = (|| -> anyhow::Result<()> {
        let mut options = OpenOptions::new();
        options.write(true).create_new(true);
        #[cfg(unix)]
        {
            use std::os::unix::fs::OpenOptionsExt;
            options.mode(0o600);
        }
        let mut file = options.open(&tmp)?;
        file.write_all(&serde_json::to_vec_pretty(credentials)?)?;
        file.write_all(b"\n")?;
        file.sync_all()?;
        fs::rename(&tmp, path)?;
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            fs::set_permissions(path, fs::Permissions::from_mode(0o600))?;
        }
        Ok(())
    })();
    if result.is_err() {
        let _ = fs::remove_file(&tmp);
    }
    result
}

fn mask_token(token: &str) -> String {
    let (prefix, secret) = match token.split_once('_') {
        Some((prefix, secret)) => (Some(prefix), secret),
        None => (None, token),
    };
    let suffix = if secret.chars().count() <= 4 {
        String::new()
    } else {
        secret
            .chars()
            .rev()
            .take(4)
            .collect::<Vec<_>>()
            .into_iter()
            .rev()
            .collect()
    };
    match prefix {
        Some(prefix) => format!("{prefix}_••••{suffix}"),
        None => format!("••••{suffix}"),
    }
}

fn state(stored: Option<&str>, environment: Option<&str>) -> CatalogCredentialState {
    let stored = stored.map(str::trim).filter(|value| !value.is_empty());
    let environment = environment.map(str::trim).filter(|value| !value.is_empty());
    let (token, source) = if let Some(token) = stored {
        (Some(token), Some("server"))
    } else if let Some(token) = environment {
        (Some(token), Some("environment"))
    } else {
        (None, None)
    };
    CatalogCredentialState {
        configured: token.is_some(),
        source,
        masked: token.map(mask_token),
    }
}

pub(crate) fn credential_status(
    credentials: &CatalogCredentials,
    hf_environment: Option<&str>,
    civitai_environment: Option<&str>,
) -> CatalogCredentialStatus {
    CatalogCredentialStatus {
        hf: state(credentials.hf_token.as_deref(), hf_environment),
        civitai: state(credentials.civitai_token.as_deref(), civitai_environment),
    }
}

fn current_status(credentials: &CatalogCredentials) -> CatalogCredentialStatus {
    credential_status(
        credentials,
        nonempty_env("HF_TOKEN").as_deref(),
        nonempty_env("CIVITAI_TOKEN").as_deref(),
    )
}

fn load_server_credentials() -> CatalogCredentials {
    let Some(path) = mold_core::Config::mold_dir().map(|dir| dir.join(CREDENTIALS_FILE)) else {
        return CatalogCredentials::default();
    };
    let cache =
        SERVER_CREDENTIALS_CACHE.get_or_init(|| RwLock::new(CatalogCredentialsCache::default()));
    {
        let guard = cache
            .read()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        if guard.path.as_deref() == Some(path.as_path()) {
            return guard.credentials.clone();
        }
    }
    let mut guard = cache
        .write()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    match guard.get_or_load(&path) {
        Ok(credentials) => credentials,
        Err(error) => {
            tracing::warn!(
                target: "catalog.credentials",
                path = %path.display(),
                %error,
                "failed to read server catalog credentials",
            );
            CatalogCredentials::default()
        }
    }
}

fn update_server_credentials_cache(path: PathBuf, credentials: &CatalogCredentials) {
    SERVER_CREDENTIALS_CACHE
        .get_or_init(|| RwLock::new(CatalogCredentialsCache::default()))
        .write()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
        .replace(path, credentials.clone());
}

pub(crate) fn resolved_hf_token() -> Option<String> {
    resolved_token(load_server_credentials().hf_token, nonempty_env("HF_TOKEN"))
}

pub(crate) fn resolved_civitai_token() -> Option<String> {
    resolved_token(
        load_server_credentials().civitai_token,
        nonempty_env("CIVITAI_TOKEN"),
    )
}

pub async fn get_catalog_credentials() -> Result<Json<CatalogCredentialStatus>, ApiError> {
    let path = credentials_path()?;
    let credentials = load_from_path(&path).map_err(|error| {
        ApiError::internal(format!("failed to read catalog credentials: {error:#}"))
    })?;
    update_server_credentials_cache(path, &credentials);
    Ok(Json(current_status(&credentials)))
}

pub async fn put_catalog_credential(
    AxumPath(provider): AxumPath<String>,
    Json(body): Json<SetCatalogCredential>,
) -> Result<Json<CatalogCredentialStatus>, ApiError> {
    let provider = Provider::parse(&provider)?;
    let token = body.token.trim();
    if token.is_empty() {
        return Err(ApiError::validation("token must not be empty"));
    }

    let path = credentials_path()?;
    let _guard = CREDENTIALS_LOCK
        .get_or_init(|| Mutex::new(()))
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    let mut credentials = load_from_path(&path).map_err(|error| {
        ApiError::internal(format!("failed to read catalog credentials: {error:#}"))
    })?;
    *provider.get_mut(&mut credentials) = Some(token.to_string());
    save_to_path_unlocked(&path, &credentials).map_err(|error| {
        ApiError::internal(format!("failed to persist catalog credentials: {error:#}"))
    })?;
    update_server_credentials_cache(path, &credentials);
    Ok(Json(current_status(&credentials)))
}

pub async fn delete_catalog_credential(
    AxumPath(provider): AxumPath<String>,
) -> Result<Json<CatalogCredentialStatus>, ApiError> {
    let provider = Provider::parse(&provider)?;
    let path = credentials_path()?;
    let _guard = CREDENTIALS_LOCK
        .get_or_init(|| Mutex::new(()))
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    let mut credentials = load_from_path(&path).map_err(|error| {
        ApiError::internal(format!("failed to read catalog credentials: {error:#}"))
    })?;
    *provider.get_mut(&mut credentials) = None;
    save_to_path_unlocked(&path, &credentials).map_err(|error| {
        ApiError::internal(format!("failed to persist catalog credentials: {error:#}"))
    })?;
    update_server_credentials_cache(path, &credentials);
    Ok(Json(current_status(&credentials)))
}

#[cfg(test)]
mod tests {
    use std::fs;

    use super::*;

    #[test]
    fn credentials_round_trip_without_exposing_raw_values_in_status() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("catalog-credentials.json");
        let credentials = CatalogCredentials {
            hf_token: Some("hf_secret1234".into()),
            civitai_token: Some("cv_secret5678".into()),
        };

        save_to_path(&path, &credentials).unwrap();
        assert_eq!(load_from_path(&path).unwrap(), credentials);
        let status = credential_status(&credentials, None, None);
        assert_eq!(status.hf.masked.as_deref(), Some("hf_••••1234"));
        assert_eq!(status.civitai.masked.as_deref(), Some("cv_••••5678"));
        assert!(!serde_json::to_string(&status).unwrap().contains("secret"));

        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            assert_eq!(
                fs::metadata(path).unwrap().permissions().mode() & 0o777,
                0o600
            );
        }
    }

    #[test]
    fn saved_server_credentials_override_environment_defaults() {
        let credentials = CatalogCredentials {
            hf_token: Some("stored-hf".into()),
            civitai_token: Some("stored-cv".into()),
        };
        let status = credential_status(&credentials, Some("env-hf"), None);
        assert_eq!(status.hf.source, Some("server"));
        assert_eq!(status.hf.masked.as_deref(), Some("••••d-hf"));
        assert_eq!(status.civitai.source, Some("server"));
    }

    #[test]
    fn environment_credentials_are_fallbacks_when_no_saved_override_exists() {
        let credentials = CatalogCredentials::default();
        let status = credential_status(&credentials, Some("env-hf"), Some("env-cv"));
        assert_eq!(status.hf.source, Some("environment"));
        assert_eq!(status.hf.masked.as_deref(), Some("••••v-hf"));
        assert_eq!(status.civitai.source, Some("environment"));
        assert_eq!(status.civitai.masked.as_deref(), Some("••••v-cv"));
    }

    #[test]
    fn whitespace_only_saved_value_reports_the_environment_fallback() {
        let credentials = CatalogCredentials {
            hf_token: Some("  ".into()),
            civitai_token: None,
        };
        let status = credential_status(&credentials, Some("env-hf"), None);
        assert_eq!(status.hf.source, Some("environment"));
        assert_eq!(status.hf.masked.as_deref(), Some("••••v-hf"));
    }

    #[test]
    fn runtime_resolution_uses_saved_override_then_environment_fallback() {
        assert_eq!(
            resolved_token(Some(" saved ".into()), Some("environment".into())),
            Some("saved".into())
        );
        assert_eq!(
            resolved_token(None, Some("environment".into())),
            Some("environment".into())
        );
        assert_eq!(
            resolved_token(Some("  ".into()), Some("environment".into())),
            Some("environment".into())
        );
    }

    #[test]
    fn short_tokens_never_expose_their_secret_characters() {
        assert_eq!(mask_token("abc"), "••••");
        assert_eq!(mask_token("1234"), "••••");
        assert_eq!(mask_token("hf_x"), "hf_••••");
        assert_eq!(mask_token("cv_abcd"), "cv_••••");
    }

    #[test]
    fn credential_cache_reuses_a_loaded_path_until_an_api_write_updates_it() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("catalog-credentials.json");
        let first = CatalogCredentials {
            hf_token: Some("first".into()),
            civitai_token: None,
        };
        let second = CatalogCredentials {
            hf_token: Some("second".into()),
            civitai_token: None,
        };
        save_to_path(&path, &first).unwrap();

        let mut cache = CatalogCredentialsCache::default();
        assert_eq!(cache.get_or_load(&path).unwrap(), first);
        save_to_path(&path, &second).unwrap();
        assert_eq!(
            cache.get_or_load(&path).unwrap(),
            first,
            "hot-path reads should use the in-memory value"
        );

        cache.replace(path, second.clone());
        assert_eq!(cache.credentials, second);
    }

    #[test]
    fn provider_mutation_does_not_touch_the_other_credential() {
        let mut credentials = CatalogCredentials {
            hf_token: Some("stored-hf".into()),
            civitai_token: Some("stored-cv".into()),
        };
        *Provider::Hf.get_mut(&mut credentials) = Some("new-hf".into());
        assert_eq!(credentials.hf_token.as_deref(), Some("new-hf"));
        assert_eq!(credentials.civitai_token.as_deref(), Some("stored-cv"));
    }
}
