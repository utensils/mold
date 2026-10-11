//! Host-to-loopback API attestations. These never authorize the cloud router.
use crate::*;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct Invitation {
    pub id: String,
    pub signing_key: String,
    pub expires_at: u64,
    pub epoch: u64,
    pub signature: String,
}
impl Invitation {
    pub fn canonical(&self, namespace: &str) -> Result<Vec<u8>> {
        invite_bytes(
            namespace,
            &self.id,
            &decode_hex(&self.signing_key)?,
            self.expires_at,
            self.epoch,
        )
    }
    pub fn verify(&self, host: &[u8; 32], now: u64, epoch: u64) -> Result<()> {
        if self.expires_at <= now || self.epoch != epoch {
            return Err(Error::Authentication);
        }
        if !verify_signature(
            host,
            &self.canonical(&namespace(host))?,
            &decode_hex::<64>(&self.signature)?,
        ) {
            return Err(Error::Authentication);
        }
        Ok(())
    }
}
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RequestContext {
    pub binding: Binding,
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub grant: Option<DeviceGrant>,
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub invite: Option<Invitation>,
    pub handshake_hash: String,
    pub stream_id: u32,
    pub issued_at: u64,
    pub request_nonce: String,
    pub signature: String,
}
fn lp(bytes: &mut Vec<u8>, s: &str) -> Result<()> {
    if s.len() > u16::MAX as usize {
        return Err(Error::Limit);
    }
    bytes.extend_from_slice(&(s.len() as u16).to_be_bytes());
    bytes.extend_from_slice(s.as_bytes());
    Ok(())
}
impl RequestContext {
    pub fn certificate_bytes(&self, namespace: &str) -> Result<Vec<u8>> {
        let mut bytes = match (&self.grant, &self.invite) {
            (Some(g), None) => grant_bytes(
                namespace,
                &decode_hex(&self.binding.signing_key)?,
                &decode_hex(&self.binding.noise_key)?,
                &g.id,
                g.expires_at,
                g.epoch,
            )?,
            (None, Some(i)) => i.canonical(namespace)?,
            _ => return Err(Error::Input),
        };
        bytes.extend_from_slice(&decode_hex::<64>(if let Some(g) = &self.grant {
            &g.signature
        } else {
            &self.invite.as_ref().ok_or(Error::Input)?.signature
        })?);
        Ok(bytes)
    }
    pub fn canonical(&self, namespace: &str, method: &str, path: &str) -> Result<Vec<u8>> {
        if self.stream_id == 0 || self.stream_id.is_multiple_of(2) {
            return Err(Error::Input);
        }
        // Validate the same method/target rules used by encrypted Open records.
        Record::Open {
            stream_id: self.stream_id,
            request: Open {
                method: method.into(),
                path: path.into(),
                headers: vec![],
                length: None,
            },
        }
        .validate()?;
        self.binding.bytes()?;
        let mut bytes = b"MOLD3-REQUEST\0".to_vec();
        bytes.extend_from_slice(&decode_hex::<16>(namespace)?);
        bytes.extend_from_slice(&decode_hex::<32>(&self.handshake_hash)?);
        bytes.extend_from_slice(&decode_hex::<32>(&self.binding.signing_key)?);
        bytes.extend_from_slice(&decode_hex::<32>(&self.binding.noise_key)?);
        bytes.extend_from_slice(&self.stream_id.to_be_bytes());
        bytes.extend_from_slice(&self.issued_at.to_be_bytes());
        bytes.extend_from_slice(&decode_hex::<16>(&self.request_nonce)?);
        lp(&mut bytes, method)?;
        lp(&mut bytes, path)?;
        bytes.extend_from_slice(&Sha256::digest(self.certificate_bytes(namespace)?));
        Ok(bytes)
    }
    pub fn sign(&mut self, host: &Identity, method: &str, path: &str) -> Result<()> {
        self.signature = hex::encode(host.sign(&self.canonical(
            &namespace(&host.signing_public()),
            method,
            path,
        )?));
        Ok(())
    }
    /// Replay nonce consumption and grant/database policy remain server-owned.
    pub fn verify(&self, host: &[u8; 32], method: &str, path: &str, now: u64) -> Result<()> {
        if now.abs_diff(self.issued_at) > 30 {
            return Err(Error::Authentication);
        }
        if !verify_signature(
            host,
            &self.canonical(&namespace(host), method, path)?,
            &decode_hex::<64>(&self.signature)?,
        ) {
            return Err(Error::Authentication);
        }
        Ok(())
    }
}
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ControlContext {
    pub issued_at: u64,
    pub nonce: String,
    pub signature: String,
}
pub fn control_bytes(namespace: &str, issued_at: u64, nonce: &str) -> Result<Vec<u8>> {
    let mut bytes = b"MOLD3-CONTROL\0".to_vec();
    bytes.extend_from_slice(&decode_hex::<16>(namespace)?);
    bytes.extend_from_slice(&issued_at.to_be_bytes());
    bytes.extend_from_slice(&decode_hex::<16>(nonce)?);
    Ok(bytes)
}
impl ControlContext {
    pub fn sign(&mut self, host: &Identity) -> Result<()> {
        self.signature = hex::encode(host.sign(&control_bytes(
            &namespace(&host.signing_public()),
            self.issued_at,
            &self.nonce,
        )?));
        Ok(())
    }
    pub fn verify(&self, host: &[u8; 32], now: u64) -> Result<()> {
        if now.abs_diff(self.issued_at) > 30
            || !verify_signature(
                host,
                &control_bytes(&namespace(host), self.issued_at, &self.nonce)?,
                &decode_hex::<64>(&self.signature)?,
            )
        {
            return Err(Error::Authentication);
        }
        Ok(())
    }
}
