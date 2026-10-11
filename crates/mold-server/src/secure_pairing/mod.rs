//! Host-owned E2EE pairing and device authorization. Cloud admission is not API authority.
mod identity;
use anyhow::{ensure, Result};
pub use identity::load_host_identity;
use mold_db::secure_devices::SecureDevice;
use mold_secure::{Binding, ControlContext, DeviceGrant, Identity, Invitation, RequestContext};
use serde::Serialize;
use std::{
    collections::{BTreeMap, HashMap, HashSet},
    sync::{Arc, Mutex},
    time::{SystemTime, UNIX_EPOCH},
};
const MAX_NONCES: usize = 65536;
pub(crate) const GRANT_TTL: u64 = 30 * 86400;
fn unix_time() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs()
}
fn hex(bytes: &[u8]) -> String {
    crate::connections::hex(bytes)
}
#[derive(Serialize)]
pub(crate) struct IssuedInvite {
    pub invite_seed: String,
    pub invite: Invitation,
}
#[derive(Default)]
struct NonceCache {
    seen: HashSet<String>,
    expires: BTreeMap<u64, Vec<String>>,
}
pub(crate) struct SecurePairing {
    pub identity: Arc<Identity>,
    invites: Mutex<HashMap<String, Invitation>>,
    nonces: Mutex<NonceCache>,
    devices: Mutex<HashMap<String, SecureDevice>>,
    policy: tokio::sync::watch::Sender<mold_relay::secure::HostPolicy>,
}
impl SecurePairing {
    pub fn new(identity: Identity, devices: Vec<SecureDevice>) -> Self {
        let policy = tokio::sync::watch::channel(mold_relay::secure::HostPolicy {
            epoch: 0,
            active: devices.iter().filter(|device| device.expires_at > unix_time()).map(|device| device.grant_id.clone()).collect(),
        }).0;
        Self {
            policy,
            identity: Arc::new(identity),
            invites: Mutex::new(HashMap::new()),
            nonces: Mutex::new(NonceCache::default()),
            devices: Mutex::new(
                devices
                    .into_iter()
                    .map(|d| (d.grant_id.clone(), d))
                    .collect(),
            ),
        }
    }
    pub fn namespace(&self) -> String {
        mold_secure::namespace(&self.identity.signing_public())
    }
    pub fn epoch(&self) -> u64 {
        0
    }
    pub fn public_fields(&self) -> serde_json::Value {
        let binding = self.identity.public_binding();
        serde_json::json!({"protocol":3,"namespace":self.namespace(),"host_key":binding.signing_key,"host_noise_key":binding.noise_key,"host_binding_signature":binding.signature})
    }
    pub fn issue_invite(&self) -> Result<IssuedInvite> {
        ensure!(
            self.has_device_capacity(),
            "paired device capacity reached"
        );
        let identity = Identity::generate()?;
        let secret = zeroize::Zeroizing::new(identity.to_bytes());
        let now = unix_time();
        let mut random = [0; 16];
        getrandom::fill(&mut random)?;
        let mut invite = Invitation {
            id: hex(&random),
            signing_key: hex(&identity.signing_public()),
            expires_at: now + 120,
            epoch: self.epoch(),
            signature: String::new(),
        };
        invite.signature = hex(&self.identity.sign(&invite.canonical(&self.namespace())?));
        let mut invites = self.invites.lock().unwrap_or_else(|p| p.into_inner());
        invites.retain(|_, i| i.expires_at > now);
        ensure!(invites.len() < 32, "pairing invitation capacity reached");
        invites.insert(invite.id.clone(), invite.clone());
        Ok(IssuedInvite {
            invite_seed: hex(&secret[..32]),
            invite,
        })
    }
    pub fn has_device_capacity(&self) -> bool {
        self.devices.lock().unwrap_or_else(|p| p.into_inner()).len() < 4096
    }
    pub fn consume_invite(
        &self,
        invite: &Invitation,
        binding: &Binding,
        handshake_hash: &[u8; 32],
        proof: &str,
    ) -> Result<bool> {
        binding.bytes()?;
        invite.verify(&self.identity.signing_public(), unix_time(), self.epoch())?;
        let bytes = mold_secure::claim_bytes(
            &self.namespace(),
            &invite.id,
            handshake_hash,
            &mold_secure::decode_hex(&binding.signing_key)?,
            &mold_secure::decode_hex(&binding.noise_key)?,
        )?;
        if !mold_secure::verify_signature(
            &mold_secure::decode_hex::<32>(&invite.signing_key)?,
            &bytes,
            &mold_secure::decode_hex::<64>(proof)?,
        ) {
            return Ok(false);
        }
        let mut invites = self.invites.lock().unwrap_or_else(|p| p.into_inner());
        let matches = invites.get(&invite.id).is_some_and(|issued| {
            issued.signature == invite.signature && issued.expires_at > unix_time()
        });
        if matches {
            invites.remove(&invite.id);
        }
        Ok(matches)
    }
    pub fn consume_nonce(&self, nonce: &str, now: u64) -> bool {
        let mut nonces = self.nonces.lock().unwrap_or_else(|p| p.into_inner());
        while nonces
            .expires
            .first_key_value()
            .is_some_and(|(until, _)| *until <= now)
        {
            if let Some((_, expired)) = nonces.expires.pop_first() {
                for value in expired {
                    nonces.seen.remove(&value);
                }
            }
        }
        if nonces.seen.contains(nonce) || nonces.seen.len() >= MAX_NONCES {
            return false;
        }
        nonces.seen.insert(nonce.into());
        nonces
            .expires
            .entry(now + 61)
            .or_default()
            .push(nonce.into());
        true
    }
    pub fn verify_request(&self, context: &RequestContext, method: &str, path: &str) -> Result<()> {
        let now = unix_time();
        context.verify(&self.identity.signing_public(), method, path, now)?;
        match (&context.grant, &context.invite) {
            (Some(grant), None) => {
                grant.verify(
                    &self.identity.signing_public(),
                    &context.binding,
                    now,
                    self.epoch(),
                )?;
                ensure!(
                    self.device_for_context(context).is_some(),
                    "paired grant revoked or mismatched"
                );
            }
            (None, Some(invite)) => {
                invite.verify(&self.identity.signing_public(), now, self.epoch())?;
                ensure!(
                    method == "POST" && path == "/api/pairing/claim",
                    "invitation may only claim pairing"
                );
                ensure!(
                    self.invites
                        .lock()
                        .unwrap_or_else(|p| p.into_inner())
                        .contains_key(&invite.id),
                    "invitation is unavailable"
                );
            }
            _ => anyhow::bail!("invalid secure admission"),
        }
        ensure!(
            self.consume_nonce(&context.request_nonce, now),
            "request proof already used or capacity exhausted"
        );
        Ok(())
    }
    pub fn verify_control(&self, context: &ControlContext) -> Result<()> {
        let now = unix_time();
        context.verify(&self.identity.signing_public(), now)?;
        ensure!(
            self.consume_nonce(&format!("control:{}", context.nonce), now),
            "control proof already used or capacity exhausted"
        );
        Ok(())
    }
    pub fn device_for_context(&self, context: &RequestContext) -> Option<SecureDevice> {
        let grant = context.grant.as_ref()?;
        self.devices
            .lock()
            .unwrap_or_else(|p| p.into_inner())
            .get(&grant.id)
            .filter(|device| {
                device.signing_key == context.binding.signing_key
                    && device.noise_key == context.binding.noise_key
                    && device.epoch == grant.epoch
                    && device.expires_at >= grant.expires_at
                    && device.expires_at > unix_time()
            })
            .cloned()
    }
    pub fn credential_bound(&self, hash: &[u8; 32]) -> bool {
        self.devices
            .lock()
            .unwrap_or_else(|p| p.into_inner())
            .values()
            .any(|d| &d.credential_hash == hash)
    }
    pub fn credential_matches(&self, context: &RequestContext, hash: &[u8; 32]) -> bool {
        self.device_for_context(context)
            .is_some_and(|d| d.credential_hash == *hash)
    }
    pub fn insert_device(&self, device: SecureDevice) {
        self.devices
            .lock()
            .unwrap_or_else(|p| p.into_inner())
            .insert(device.grant_id.clone(), device);
        self.publish_policy();
    }
    pub fn revoke(&self, client_id: &str) {
        self.devices
            .lock()
            .unwrap_or_else(|p| p.into_inner())
            .retain(|_, d| d.client_id != client_id);
        self.publish_policy();
    }
    fn publish_policy(&self) {
        self.policy.send_replace(mold_relay::secure::HostPolicy { epoch: self.epoch(), active: self.active_ids() });
    }
    pub fn watch_policy(&self) -> tokio::sync::watch::Receiver<mold_relay::secure::HostPolicy> {
        self.policy.subscribe()
    }
    pub fn active_ids(&self) -> Vec<String> {
        let now = unix_time();
        let mut ids = self
            .devices
            .lock()
            .unwrap_or_else(|p| p.into_inner())
            .values()
            .filter(|d| d.expires_at > now)
            .map(|d| d.grant_id.clone())
            .collect::<Vec<_>>();
        ids.sort();
        ids
    }
    pub fn certificate(&self, device: &SecureDevice) -> Result<DeviceGrant> {
        let bytes = mold_secure::grant_bytes(
            &self.namespace(),
            &mold_secure::decode_hex(&device.signing_key)?,
            &mold_secure::decode_hex(&device.noise_key)?,
            &device.grant_id,
            device.expires_at,
            device.epoch,
        )?;
        Ok(DeviceGrant {
            id: device.grant_id.clone(),
            expires_at: device.expires_at,
            epoch: device.epoch,
            signature: hex(&self.identity.sign(&bytes)),
        })
    }
}

#[cfg(test)]
mod authority_tests;
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn invitation_claim_requires_channel_bound_proof_and_is_single_use() {
        let secure = SecurePairing::new(mold_secure::Identity::generate().unwrap(), Vec::new());
        let issued = secure.issue_invite().unwrap();
        let device = mold_secure::Identity::generate().unwrap();
        let binding = device.public_binding();
        let mut private = [0; 64];
        private[..32].copy_from_slice(&mold_secure::decode_hex::<32>(&issued.invite_seed).unwrap());
        let invitation = mold_secure::Identity::from_bytes(&private).unwrap();
        let hash = [8; 32];
        let message = mold_secure::claim_bytes(
            &secure.namespace(),
            &issued.invite.id,
            &hash,
            &device.signing_public(),
            &device.noise_public(),
        )
        .unwrap();
        let proof = crate::connections::hex(&invitation.sign(&message));
        assert!(!secure
            .consume_invite(&issued.invite, &binding, &[9; 32], &proof)
            .unwrap());
        assert!(secure
            .consume_invite(&issued.invite, &binding, &hash, &proof)
            .unwrap());
        assert!(!secure
            .consume_invite(&issued.invite, &binding, &hash, &proof)
            .unwrap());
    }
    #[test]
    fn replay_cache_never_evicts_unexpired_proofs_to_admit_more_requests() {
        let secure = SecurePairing::new(mold_secure::Identity::generate().unwrap(), Vec::new());
        assert!(secure.consume_nonce("first", 100));
        assert!(!secure.consume_nonce("first", 100));
        for n in 1..MAX_NONCES {
            assert!(secure.consume_nonce(&n.to_string(), 100));
        }
        assert!(!secure.consume_nonce("overflow", 100));
        assert!(secure.consume_nonce("first", 162));
    }
}

pub(crate) async fn check(
    auth: Option<axum::Extension<crate::auth::AuthState>>,
) -> Result<impl axum::response::IntoResponse, crate::routes::ApiError> {
    let keys = auth.and_then(|axum::Extension(a)| a).ok_or_else(|| {
        crate::routes::ApiError::with_code(
            "secure host unavailable",
            "SECURE_UNAVAILABLE",
            axum::http::StatusCode::UNAUTHORIZED,
        )
    })?;
    let secure = keys.secure.as_ref().ok_or_else(|| {
        crate::routes::ApiError::with_code(
            "secure host unavailable",
            "SECURE_UNAVAILABLE",
            axum::http::StatusCode::UNAUTHORIZED,
        )
    })?;
    Ok((
        [(axum::http::header::CACHE_CONTROL, "no-store")],
        axum::Json(serde_json::json!({"epoch":secure.epoch(),"active":secure.active_ids()})),
    ))
}
pub(crate) async fn renew(
    auth: Option<axum::Extension<crate::auth::AuthState>>,
    context: Option<axum::Extension<RequestContext>>,
) -> Result<impl axum::response::IntoResponse, crate::routes::ApiError> {
    let invalid = || {
        crate::routes::ApiError::with_code(
            "live encrypted device grant required",
            "SECURE_GRANT_REQUIRED",
            axum::http::StatusCode::UNAUTHORIZED,
        )
    };
    let keys = auth.and_then(|axum::Extension(a)| a).ok_or_else(invalid)?;
    let axum::Extension(context) = context.ok_or_else(invalid)?;
    let grant = keys.renew_secure_pairing(&context).map_err(|_| invalid())?;
    Ok((
        [(axum::http::header::CACHE_CONTROL, "no-store")],
        axum::Json(serde_json::json!({"grant":grant})),
    ))
}
