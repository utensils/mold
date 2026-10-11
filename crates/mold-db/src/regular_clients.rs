//! Explicit ordinary HTTP authority, separate from device-bound Noise grants.
use crate::{paired_clients::PairedClient, MetadataDb};
use anyhow::Result;
use rusqlite::params;

pub struct RegularClients<'a>(&'a MetadataDb);
impl<'a> RegularClients<'a> {
    pub fn new(db: &'a MetadataDb) -> Self {
        Self(db)
    }
    pub fn list(&self, instance: &str) -> Result<Vec<String>> {
        self.0.with_conn(|conn| {
            let mut statement = conn.prepare("SELECT r.client_id FROM regular_clients r JOIN paired_clients p ON p.id=r.client_id WHERE p.server_instance_id=?1 ORDER BY r.client_id")?;
            let rows = statement.query_map([instance], |r| r.get(0))?;
            rows.collect::<rusqlite::Result<Vec<_>>>().map_err(Into::into)
        })
    }
    pub fn insert(&self, client: &PairedClient) -> Result<()> {
        self.0.with_conn(|conn| {
            let tx = conn.unchecked_transaction()?;
            tx.execute("INSERT INTO paired_clients(id,server_instance_id,name,client_kind,credential_hash,created_at_ms,last_used_at_ms) VALUES(?1,?2,?3,?4,?5,?6,?7)",params![client.id,client.server_instance_id,client.name,client.client_kind,client.credential_hash.as_slice(),client.created_at_ms,client.last_used_at_ms])?;
            tx.execute("INSERT INTO regular_clients(client_id) VALUES(?1)",[&client.id])?;
            tx.commit()?;
            Ok(())
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn regular_authority_is_atomic_scoped_exclusive_and_revoked_with_credential() {
        let db = MetadataDb::open_in_memory().unwrap();
        let client = PairedClient {
            id: "direct-phone".into(),
            server_instance_id: "host".into(),
            name: "Phone".into(),
            client_kind: "iphone".into(),
            credential_hash: [1; 32],
            created_at_ms: 1,
            last_used_at_ms: None,
        };
        let store = RegularClients::new(&db);
        store.insert(&client).unwrap();
        assert_eq!(store.list("host").unwrap(), vec!["direct-phone"]);
        assert!(store.list("other").unwrap().is_empty());
        assert!(store.insert(&client).is_err());
        assert_eq!(
            crate::paired_clients::PairedClients::new(&db)
                .list("host")
                .unwrap()
                .len(),
            1
        );
        let forbidden = db.with_conn(|conn| Ok(conn.execute("INSERT INTO secure_devices(client_id,grant_id,signing_key,noise_key,expires_at,epoch) VALUES(?1,?2,?3,?4,1,0)",params![client.id,"ab".repeat(16),"cd".repeat(32),"ef".repeat(32)])?));
        assert!(forbidden.is_err());
        crate::paired_clients::PairedClients::new(&db)
            .revoke("host", &client.id)
            .unwrap();
        assert!(store.list("host").unwrap().is_empty());
    }
}
