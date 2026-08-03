//! Durable, individually revocable credentials issued by the pairing API.

use anyhow::{Context, Result};
use rusqlite::params;

use crate::MetadataDb;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PairedClient {
    pub id: String,
    pub server_instance_id: String,
    pub name: String,
    pub client_kind: String,
    pub credential_hash: [u8; 32],
    pub created_at_ms: i64,
    pub last_used_at_ms: Option<i64>,
}

pub struct PairedClients<'a> {
    db: &'a MetadataDb,
}

impl<'a> PairedClients<'a> {
    pub fn new(db: &'a MetadataDb) -> Self {
        Self { db }
    }

    pub fn list(&self, server_instance_id: &str) -> Result<Vec<PairedClient>> {
        self.db.with_conn(|conn| {
            let mut statement = conn.prepare(
                "SELECT id, server_instance_id, name, client_kind, credential_hash, created_at_ms, last_used_at_ms
                 FROM paired_clients
                 WHERE server_instance_id = ?1
                 ORDER BY created_at_ms DESC, id ASC",
            )?;
            let rows = statement.query_map([server_instance_id], |row| {
                let hash: Vec<u8> = row.get(4)?;
                let credential_hash: [u8; 32] = hash.try_into().map_err(|value: Vec<u8>| {
                    rusqlite::Error::FromSqlConversionFailure(
                        value.len(),
                        rusqlite::types::Type::Blob,
                        "paired credential hash must be 32 bytes".into(),
                    )
                })?;
                Ok(PairedClient {
                    id: row.get(0)?,
                    server_instance_id: row.get(1)?,
                    name: row.get(2)?,
                    client_kind: row.get(3)?,
                    credential_hash,
                    created_at_ms: row.get(5)?,
                    last_used_at_ms: row.get(6)?,
                })
            })?;
            rows.collect::<rusqlite::Result<Vec<_>>>()
                .map_err(Into::into)
        })
    }

    pub fn insert(&self, client: &PairedClient) -> Result<()> {
        self.db.with_conn(|conn| {
            conn.execute(
                "INSERT INTO paired_clients (
                    id, server_instance_id, name, client_kind, credential_hash, created_at_ms, last_used_at_ms
                 ) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7)",
                params![
                    client.id,
                    client.server_instance_id,
                    client.name,
                    client.client_kind,
                    client.credential_hash.as_slice(),
                    client.created_at_ms,
                    client.last_used_at_ms,
                ],
            )
            .context("inserting paired client")?;
            Ok(())
        })
    }

    pub fn touch(&self, server_instance_id: &str, id: &str, last_used_at_ms: i64) -> Result<()> {
        self.db.with_conn(|conn| {
            conn.execute(
                "UPDATE paired_clients SET last_used_at_ms = ?3 WHERE server_instance_id = ?1 AND id = ?2",
                params![server_instance_id, id, last_used_at_ms],
            )?;
            Ok(())
        })
    }

    pub fn revoke(&self, server_instance_id: &str, id: &str) -> Result<bool> {
        self.db.with_conn(|conn| {
            Ok(conn.execute(
                "DELETE FROM paired_clients WHERE server_instance_id = ?1 AND id = ?2",
                params![server_instance_id, id],
            )? > 0)
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn grants_round_trip_touch_and_revoke() {
        let db = MetadataDb::open_in_memory().unwrap();
        let clients = PairedClients::new(&db);
        let client = PairedClient {
            id: "phone-1".into(),
            server_instance_id: "server-a".into(),
            name: "James's iPhone".into(),
            client_kind: "mobile".into(),
            credential_hash: [7; 32],
            created_at_ms: 100,
            last_used_at_ms: None,
        };

        clients.insert(&client).unwrap();
        assert_eq!(clients.list("server-a").unwrap(), vec![client.clone()]);
        assert!(clients.list("server-b").unwrap().is_empty());
        clients.touch("server-a", &client.id, 250).unwrap();
        assert_eq!(
            clients.list("server-a").unwrap()[0].last_used_at_ms,
            Some(250)
        );
        assert!(!clients.revoke("server-b", &client.id).unwrap());
        assert!(clients.revoke("server-a", &client.id).unwrap());
        assert!(!clients.revoke("server-a", &client.id).unwrap());
        assert!(clients.list("server-a").unwrap().is_empty());
    }
}
