//! Device keys and relay certificates bound to independently revocable API grants.
use crate::{paired_clients::PairedClient, MetadataDb};
use anyhow::{ensure, Result};
use rusqlite::params;
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SecureDevice {
    pub client_id: String,
    pub server_instance_id: String,
    pub grant_id: String,
    pub signing_key: String,
    pub noise_key: String,
    pub credential_hash: [u8; 32],
    pub expires_at: u64,
    pub epoch: u64,
}
pub struct SecureDevices<'a>(&'a MetadataDb);
impl<'a> SecureDevices<'a> {
    pub fn new(db: &'a MetadataDb) -> Self {
        Self(db)
    }
    pub fn list(&self, instance: &str) -> Result<Vec<SecureDevice>> {
        self.0.with_conn(|conn| {
            let mut statement=conn.prepare("SELECT s.client_id, p.server_instance_id, s.grant_id, s.signing_key, s.noise_key, p.credential_hash, s.expires_at, s.epoch FROM secure_devices s JOIN paired_clients p ON p.id=s.client_id WHERE p.server_instance_id=?1 ORDER BY s.grant_id")?;
            let rows=statement.query_map([instance],|row| {
                let bytes:Vec<u8>=row.get(5)?;
                let hash=bytes.try_into().map_err(|bytes:Vec<u8>|rusqlite::Error::FromSqlConversionFailure(bytes.len(),rusqlite::types::Type::Blob,"invalid device credential digest".into()))?;
                Ok(SecureDevice{client_id:row.get(0)?,server_instance_id:row.get(1)?,grant_id:row.get(2)?,signing_key:row.get(3)?,noise_key:row.get(4)?,credential_hash:hash,expires_at:row.get(6)?,epoch:row.get(7)?})
            })?;
            rows.collect::<rusqlite::Result<Vec<_>>>().map_err(Into::into)
        })
    }
    pub fn insert(&self, client: &PairedClient, device: &SecureDevice) -> Result<()> {
        ensure!(
            device.client_id == client.id
                && device.server_instance_id == client.server_instance_id
                && device.credential_hash == client.credential_hash,
            "device credential identity mismatch"
        );
        self.0.with_conn(|conn| {
            let tx=conn.unchecked_transaction()?;
            tx.execute("INSERT INTO paired_clients(id,server_instance_id,name,client_kind,credential_hash,created_at_ms,last_used_at_ms) VALUES(?1,?2,?3,?4,?5,?6,?7)",params![client.id,client.server_instance_id,client.name,client.client_kind,client.credential_hash.as_slice(),client.created_at_ms,client.last_used_at_ms])?;
            tx.execute("INSERT INTO secure_devices(client_id,grant_id,signing_key,noise_key,expires_at,epoch) VALUES(?1,?2,?3,?4,?5,?6)",params![device.client_id,device.grant_id,device.signing_key,device.noise_key,device.expires_at,device.epoch])?;
            tx.commit()?;Ok(())
        })
    }
    pub fn renew(&self, instance: &str, grant_id: &str, expires_at: u64) -> Result<bool> {
        self.0.with_conn(|conn|Ok(conn.execute("UPDATE secure_devices SET expires_at=?3 WHERE grant_id=?2 AND client_id IN (SELECT id FROM paired_clients WHERE server_instance_id=?1)",params![instance,grant_id,expires_at])?>0))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{paired_clients::PairedClient, MetadataDb};
    #[test]
    fn secure_device_grant_is_atomic_scoped_and_deleted_with_client() {
        let db = MetadataDb::open_in_memory().unwrap();
        let client = PairedClient {
            id: "device".into(),
            server_instance_id: "host".into(),
            name: "phone".into(),
            client_kind: "iphone".into(),
            credential_hash: [1; 32],
            created_at_ms: 1,
            last_used_at_ms: None,
        };
        let device = SecureDevice {
            client_id: client.id.clone(),
            server_instance_id: "host".into(),
            grant_id: "ab".repeat(16),
            signing_key: "cd".repeat(32),
            noise_key: "ef".repeat(32),
            credential_hash: client.credential_hash,
            expires_at: 1000,
            epoch: 0,
        };
        let store = SecureDevices::new(&db);
        store.insert(&client, &device).unwrap();
        assert_eq!(store.list("host").unwrap(), vec![device.clone()]);
        assert!(store.list("other").unwrap().is_empty());
        assert!(!store.renew("other", &device.grant_id, 2000).unwrap());
        assert!(store.renew("host", &device.grant_id, 2000).unwrap());
        assert_eq!(store.list("host").unwrap()[0].expires_at, 2000);
        crate::paired_clients::PairedClients::new(&db)
            .revoke("host", "device")
            .unwrap();
        assert!(store.list("host").unwrap().is_empty());
        let invalid = SecureDevice {
            client_id: "different".into(),
            ..device
        };
        assert!(store.insert(&client, &invalid).is_err());
        assert!(crate::paired_clients::PairedClients::new(&db)
            .list("host")
            .unwrap()
            .is_empty());
    }
}
