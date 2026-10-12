//! One-use direct pairing handoffs. Durable credentials live in the metadata DB.
use anyhow::{ensure, Result};
use sha2::{Digest, Sha256};
use std::{collections::HashMap, sync::Mutex};

#[derive(Clone)]
pub(crate) struct Invitation {
    pub origin: String,
    pub expires_at: u64,
}
#[derive(Default)]
pub(crate) struct Invitations(Mutex<HashMap<[u8; 32], Invitation>>);
impl Invitations {
    pub fn issue(&self, origin: String, now: u64) -> Result<(String, u64)> {
        let mut entries = self.0.lock().unwrap_or_else(|p| p.into_inner());
        entries.retain(|_, v| v.expires_at > now);
        ensure!(entries.len() < 32, "regular invitation capacity reached");
        let mut secret = [0u8; 32];
        getrandom::fill(&mut secret)?;
        let token = crate::connections::hex(&secret);
        let expires_at = now.saturating_add(120);
        entries.insert(
            Sha256::digest(token.as_bytes()).into(),
            Invitation { origin, expires_at },
        );
        Ok((token, expires_at))
    }
    pub fn get(&self, token: &str, now: u64) -> Option<Invitation> {
        if token.len() != 64
            || !token
                .bytes()
                .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b))
        {
            return None;
        }
        let mut entries = self.0.lock().unwrap_or_else(|p| p.into_inner());
        entries.retain(|_, v| v.expires_at > now);
        entries
            .get(&<[u8; 32]>::from(Sha256::digest(token.as_bytes())))
            .cloned()
    }
    /// The owning authority serializes claim validation, DB commit and consume.
    pub fn consume(&self, token: &str) {
        self.0
            .lock()
            .unwrap_or_else(|p| p.into_inner())
            .remove(&<[u8; 32]>::from(Sha256::digest(token.as_bytes())));
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn handoff_is_hashed_bounded_expiring_and_one_use() {
        let invitations = Invitations::default();
        let (token, expiry) = invitations
            .issue("https://direct.example".into(), 100)
            .unwrap();
        assert_eq!(expiry, 220);
        assert_eq!(
            invitations.get(&token, 100).unwrap().origin,
            "https://direct.example"
        );
        assert!(!invitations
            .0
            .lock()
            .unwrap()
            .contains_key(&mold_secure::decode_hex::<32>(&token).unwrap()));
        invitations.consume(&token);
        assert!(invitations.get(&token, 100).is_none());
        for _ in 0..32 {
            invitations
                .issue("https://direct.example".into(), 100)
                .unwrap();
        }
        assert!(invitations
            .issue("https://direct.example".into(), 219)
            .is_err());
        assert!(invitations
            .issue("https://direct.example".into(), 220)
            .is_ok());
        let (token, _) = invitations
            .issue("https://direct.example".into(), 220)
            .unwrap();
        assert!(invitations.get(&token, 340).is_none());
        assert!(Invitations::default().get(&token, 220).is_none());
    }
}
