use crate::{Error, Result};
use ed25519_dalek::{Signature, Signer, SigningKey, VerifyingKey};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use zeroize::Zeroizing;

pub struct Identity {
    secret: Zeroizing<[u8; 64]>,
}
impl Identity {
    pub fn generate() -> Result<Self> {
        let mut secret = [0; 64];
        getrandom::fill(&mut secret).map_err(|_| Error::Random)?;
        Ok(Self {
            secret: Zeroizing::new(secret),
        })
    }
    pub fn from_bytes(bytes: &[u8]) -> Result<Self> {
        Ok(Self {
            secret: Zeroizing::new(bytes.try_into().map_err(|_| Error::Input)?),
        })
    }
    pub fn to_bytes(&self) -> Vec<u8> {
        self.secret.to_vec()
    }
    pub fn signing_public(&self) -> [u8; 32] {
        self.signing_key().verifying_key().to_bytes()
    }
    pub fn noise_public(&self) -> [u8; 32] {
        x25519_dalek::PublicKey::from(&x25519_dalek::StaticSecret::from(self.noise_secret()))
            .to_bytes()
    }
    pub(crate) fn noise_secret(&self) -> [u8; 32] {
        self.secret[32..].try_into().expect("fixed secret")
    }
    fn signing_key(&self) -> SigningKey {
        SigningKey::from_bytes(self.secret[..32].try_into().expect("fixed seed"))
    }
    pub fn sign(&self, bytes: &[u8]) -> [u8; 64] {
        self.signing_key().sign(bytes).to_bytes()
    }
    pub fn binding(&self) -> [u8; 128] {
        let mut bytes = [0; 128];
        bytes[..32].copy_from_slice(&self.signing_public());
        bytes[32..64].copy_from_slice(&self.noise_public());
        let mut signed = b"MOLD3-BIND\0".to_vec();
        signed.extend_from_slice(&bytes[..64]);
        bytes[64..].copy_from_slice(&self.sign(&signed));
        bytes
    }
    pub fn public_binding(&self) -> Binding {
        let binding = self.binding();
        Binding {
            signing_key: hex::encode(&binding[..32]),
            noise_key: hex::encode(&binding[32..64]),
            signature: hex::encode(&binding[64..]),
        }
    }
}
pub fn verify_signature(public: &[u8], message: &[u8], signature: &[u8]) -> bool {
    let Ok(public) = <&[u8; 32]>::try_from(public) else {
        return false;
    };
    let Ok(key) = VerifyingKey::from_bytes(public) else {
        return false;
    };
    let Ok(signature) = Signature::from_slice(signature) else {
        return false;
    };
    key.verify_strict(message, &signature).is_ok()
}
pub fn namespace(public: &[u8]) -> String {
    hex::encode(&Sha256::digest(public)[..16])
}
pub fn decode_hex<const N: usize>(value: &str) -> Result<[u8; N]> {
    if value.len() != N * 2
        || !value
            .bytes()
            .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b))
    {
        return Err(Error::Input);
    }
    let mut out = [0; N];
    hex::decode_to_slice(value, &mut out).map_err(|_| Error::Input)?;
    Ok(out)
}
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct Binding {
    pub signing_key: String,
    pub noise_key: String,
    pub signature: String,
}
impl Binding {
    pub fn bytes(&self) -> Result<[u8; 128]> {
        let mut bytes = [0; 128];
        bytes[..32].copy_from_slice(&decode_hex::<32>(&self.signing_key)?);
        bytes[32..64].copy_from_slice(&decode_hex::<32>(&self.noise_key)?);
        bytes[64..].copy_from_slice(&decode_hex::<64>(&self.signature)?);
        Self::verify_bytes(&bytes)?;
        Ok(bytes)
    }
    pub fn verify_bytes(bytes: &[u8]) -> Result<()> {
        if bytes.len() != 128 {
            return Err(Error::Input);
        }
        let mut signed = b"MOLD3-BIND\0".to_vec();
        signed.extend_from_slice(&bytes[..64]);
        if !verify_signature(&bytes[..32], &signed, &bytes[64..]) {
            return Err(Error::Authentication);
        }
        Ok(())
    }
}
/// Exact cloud admission signing bytes. The signing identity is independent
/// from the pinned endpoint keys and never grants application API permissions.
pub fn join_bytes(
    namespace: &str,
    role: u8,
    connection_id: &str,
    nonce: &str,
    joining_key: &[u8; 32],
    expires_at: u64,
) -> Result<Vec<u8>> {
    if !(1..=3).contains(&role)
        || connection_id.is_empty()
        || connection_id.len() > 128
        || !connection_id.bytes().all(|b| b.is_ascii_graphic())
    {
        return Err(Error::Input);
    }
    let mut bytes = b"MOLD3-JOIN\0".to_vec();
    bytes.push(3);
    bytes.extend_from_slice(&decode_hex::<16>(namespace)?);
    bytes.push(role);
    bytes.extend_from_slice(&(connection_id.len() as u16).to_be_bytes());
    bytes.extend_from_slice(connection_id.as_bytes());
    bytes.extend_from_slice(&decode_hex::<32>(nonce)?);
    bytes.extend_from_slice(joining_key);
    bytes.extend_from_slice(&expires_at.to_be_bytes());
    Ok(bytes)
}
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct DeviceGrant {
    pub id: String,
    pub expires_at: u64,
    pub epoch: u64,
    pub signature: String,
}
pub fn grant_bytes(
    namespace: &str,
    device_signing: &[u8; 32],
    device_noise: &[u8; 32],
    id: &str,
    expires_at: u64,
    epoch: u64,
) -> Result<Vec<u8>> {
    let mut bytes = b"MOLD3-GRANT\0".to_vec();
    bytes.extend_from_slice(&decode_hex::<16>(namespace)?);
    bytes.extend_from_slice(device_signing);
    bytes.extend_from_slice(device_noise);
    bytes.extend_from_slice(&decode_hex::<16>(id)?);
    bytes.extend_from_slice(&expires_at.to_be_bytes());
    bytes.extend_from_slice(&epoch.to_be_bytes());
    Ok(bytes)
}
impl DeviceGrant {
    pub fn verify(
        &self,
        host_key: &[u8; 32],
        device: &Binding,
        now: u64,
        epoch: u64,
    ) -> Result<()> {
        device.bytes()?;
        if self.expires_at <= now || self.epoch != epoch {
            return Err(Error::Authentication);
        }
        let message = grant_bytes(
            &namespace(host_key),
            &decode_hex(&device.signing_key)?,
            &decode_hex(&device.noise_key)?,
            &self.id,
            self.expires_at,
            self.epoch,
        )?;
        if !verify_signature(host_key, &message, &decode_hex::<64>(&self.signature)?) {
            return Err(Error::Authentication);
        }
        Ok(())
    }
}
pub fn invite_bytes(
    namespace: &str,
    id: &str,
    invite_signing: &[u8; 32],
    expires_at: u64,
    epoch: u64,
) -> Result<Vec<u8>> {
    let mut bytes = b"MOLD3-INVITE\0".to_vec();
    bytes.extend_from_slice(&decode_hex::<16>(namespace)?);
    bytes.extend_from_slice(&decode_hex::<16>(id)?);
    bytes.extend_from_slice(invite_signing);
    bytes.extend_from_slice(&expires_at.to_be_bytes());
    bytes.extend_from_slice(&epoch.to_be_bytes());
    Ok(bytes)
}

/// Invitation proof is bound to a completed, authenticated Noise transcript.
pub fn claim_bytes(
    namespace: &str,
    id: &str,
    handshake_hash: &[u8; 32],
    device_signing: &[u8; 32],
    device_noise: &[u8; 32],
) -> Result<Vec<u8>> {
    let mut bytes = b"MOLD3-CLAIM\0".to_vec();
    bytes.extend_from_slice(&decode_hex::<16>(namespace)?);
    bytes.extend_from_slice(&decode_hex::<16>(id)?);
    bytes.extend_from_slice(handshake_hash);
    bytes.extend_from_slice(device_signing);
    bytes.extend_from_slice(device_noise);
    Ok(bytes)
}
