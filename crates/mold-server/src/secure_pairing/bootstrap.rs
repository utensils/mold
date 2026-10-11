//! Owner-only initial provisioning. Never an HTTP enrollment endpoint.
use super::{hex, unix_time, GRANT_TTL};
use anyhow::{ensure, Result};
use mold_secure::{identity_file, Binding, DeviceGrant, Identity};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::path::Path;
use subtle::ConstantTimeEq;
use zeroize::{Zeroize, Zeroizing};

/// Secret provisioning envelope; intentionally has no Debug implementation.
#[derive(Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct BootstrapInput {
    version: u8,
    id: String,
    instance_id: String,
    host_identity: String,
    client: BootstrapClient,
}
#[derive(Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct BootstrapClient {
    binding: Binding,
    grant: DeviceGrant,
    api_key: String,
    name: String,
    kind: String,
}
impl Drop for BootstrapInput {
    fn drop(&mut self) {
        self.host_identity.zeroize();
        self.client.api_key.zeroize();
    }
}
impl BootstrapInput {
    pub fn parse(raw: &str) -> Result<Self> {
        ensure!(
            raw.len() <= 16 * 1024,
            "secure bootstrap exceeds its size limit"
        );
        serde_json::from_str(raw).map_err(|_| anyhow::anyhow!("invalid secure bootstrap document"))
    }
    /// Capture and remove the provisioning environment before creating a runtime.
    ///
    /// # Safety
    /// Call once, before starting any threads that could read the process environment.
    pub unsafe fn take_environment_before_threads() -> Result<Option<Self>> {
        let raw = std::env::var_os("MOLD_SECURE_BOOTSTRAP");
        // SAFETY: caller guarantees exclusive access to process environment.
        unsafe { std::env::remove_var("MOLD_SECURE_BOOTSTRAP") };
        raw.map(|value| {
            let raw = Zeroizing::new(
                value
                    .into_string()
                    .map_err(|_| anyhow::anyhow!("secure bootstrap must be UTF-8"))?,
            );
            Self::parse(&raw)
        })
        .transpose()
    }
    pub(crate) fn apply(
        &self,
        home: &Path,
        db: &mold_db::MetadataDb,
        port: u16,
        output: &Path,
    ) -> Result<()> {
        ensure!(self.version == 1, "unsupported secure bootstrap version");
        mold_secure::decode_hex::<16>(&self.id)?;
        let parsed_instance = uuid::Uuid::parse_str(&self.instance_id)?;
        ensure!(
            parsed_instance.to_string() == self.instance_id
                && parsed_instance.get_version_num() == 4,
            "invalid bootstrap instance UUID"
        );
        let instance = self.instance_id.as_str();
        let bytes = Zeroizing::new(mold_secure::decode_hex::<64>(&self.host_identity)?);
        let identity = Identity::from_bytes(bytes.as_ref())?;
        self.client.binding.bytes()?;
        let grant = &self.client.grant;
        let namespace = mold_secure::namespace(&identity.signing_public());
        ensure!(
            grant.epoch == 0 && grant.expires_at > 0 && grant.expires_at <= i64::MAX as u64,
            "invalid bootstrap grant lifetime"
        );
        // Exact replay remains valid after the original certificate expires;
        // only a first import may install authorization, and it checks time below.
        grant.verify(
            &identity.signing_public(),
            &self.client.binding,
            grant.expires_at - 1,
            0,
        )?;
        ensure!(
            self.client
                .api_key
                .strip_prefix("mold_pair_")
                .is_some_and(|k| mold_secure::decode_hex::<32>(k).is_ok()),
            "invalid bootstrap paired credential"
        );
        ensure!(
            matches!(
                self.client.kind.as_str(),
                "cli" | "desktop" | "ios" | "android" | "browser"
            ) && !self.client.name.trim().is_empty()
                && self.client.name.len() <= 128
                && !self.client.name.chars().any(char::is_control),
            "invalid bootstrap client description"
        );
        let canonical = Zeroizing::new(
            serde_json::to_vec(self)
                .map_err(|_| anyhow::anyhow!("invalid secure bootstrap document"))?,
        );
        let digest = hex(&Sha256::digest(canonical.as_slice()));
        let hash: [u8; 32] = Sha256::digest(self.client.api_key.as_bytes()).into();
        let client = mold_db::paired_clients::PairedClient {
            id: grant.id.clone(),
            server_instance_id: instance.into(),
            name: self.client.name.clone(),
            client_kind: self.client.kind.clone(),
            credential_hash: hash,
            created_at_ms: (unix_time() * 1000) as i64,
            last_used_at_ms: None,
        };
        let device = mold_db::secure_devices::SecureDevice {
            client_id: client.id.clone(),
            server_instance_id: instance.into(),
            grant_id: grant.id.clone(),
            signing_key: self.client.binding.signing_key.clone(),
            noise_key: self.client.binding.noise_key.clone(),
            credential_hash: hash,
            expires_at: grant.expires_at,
            epoch: 0,
        };
        let path = home.join("secure/host.identity");
        let journal = home.join("secure/bootstrap.intent");
        let intent = format!("{}:{digest}:{namespace}", self.id);
        mold_db::secure_bootstrap::import(
            db,
            &self.id,
            &digest,
            &crate::instance::instance_id_key(port),
            &client,
            &device,
            |first| {
                if first {
                    match std::fs::read_dir(output) {
                        Ok(mut entries) => ensure!(
                            entries.next().is_none(),
                            "secure bootstrap requires an empty media directory"
                        ),
                        Err(e) if e.kind() == std::io::ErrorKind::NotFound => {}
                        Err(e) => return Err(e.into()),
                    }
                    let now = unix_time();
                    ensure!(
                        grant.expires_at > now && grant.expires_at <= now + GRANT_TTL,
                        "bootstrap grant expired or exceeds thirty days"
                    );
                    match identity_file::read_private_file(&journal, 256) {
                        Ok(existing) => ensure!(
                            existing.as_slice() == intent.as_bytes(),
                            "secure bootstrap journal conflict"
                        ),
                        Err(e) if e.kind() == std::io::ErrorKind::NotFound => {
                            ensure!(
                                std::fs::symlink_metadata(&path)
                                    .is_err_and(|e| e.kind() == std::io::ErrorKind::NotFound),
                                "secure bootstrap refuses an existing host identity"
                            );
                            identity_file::write_private_file(&journal, intent.as_bytes(), false)?;
                        }
                        Err(e) => return Err(e.into()),
                    }
                    if std::fs::symlink_metadata(&path)
                        .is_err_and(|e| e.kind() == std::io::ErrorKind::NotFound)
                    {
                        identity_file::write_private_file(&path, bytes.as_ref(), false)?;
                    }
                }
                match identity_file::read_private_file(&journal, 256) {
                    Ok(existing) => ensure!(
                        existing.as_slice() == intent.as_bytes(),
                        "secure bootstrap journal conflict"
                    ),
                    Err(e) if e.kind() == std::io::ErrorKind::NotFound => {}
                    Err(e) => return Err(e.into()),
                }
                let actual = identity_file::read_host_identity(&path)?;
                let actual = Zeroizing::new(actual.to_bytes());
                ensure!(
                    bool::from(actual.as_slice().ct_eq(bytes.as_ref())),
                    "secure bootstrap host identity conflict"
                );
                Ok(())
            },
        )?;
        // Only after the DB commit. A crash before here is an exact replay,
        // never a second grant; removing the journal fences a lost/replaced DB.
        match std::fs::remove_file(journal) {
            Ok(()) => {}
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => {}
            Err(e) => return Err(e.into()),
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests;
