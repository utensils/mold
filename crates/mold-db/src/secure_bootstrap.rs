//! Atomic one-time import of an owner-provisioned empty server authority.
use crate::{paired_clients::PairedClient, secure_devices::SecureDevice, MetadataDb};
use anyhow::{ensure, Result};
use rusqlite::{params, OptionalExtension, Transaction, TransactionBehavior};

pub fn import(
    db: &MetadataDb,
    id: &str,
    digest: &str,
    instance_key: &str,
    client: &PairedClient,
    device: &SecureDevice,
    prepare_identity: impl FnOnce(bool) -> Result<()>,
) -> Result<bool> {
    db.with_conn(|conn| {
        let tx = Transaction::new_unchecked(conn, TransactionBehavior::Immediate)?;
        let prior: Option<(String, String, String)> = tx.query_row(
            "SELECT bootstrap_id,digest,server_instance_id FROM secure_bootstrap WHERE singleton=1",
            [], |r| Ok((r.get(0)?, r.get(1)?, r.get(2)?)),
        ).optional()?;
        if let Some(prior) = prior {
            ensure!(prior == (id.into(), digest.into(), client.server_instance_id.clone()), "secure bootstrap conflicts with existing authority");
            let stored: Option<String> = tx.query_row("SELECT value FROM settings WHERE profile=?1 AND key=?2", params![crate::DEFAULT_PROFILE,instance_key], |r| r.get(0)).optional()?;
            ensure!(stored.as_deref() == Some(client.server_instance_id.as_str()), "secure bootstrap instance identity conflict");
            prepare_identity(false)?;
            // Revocation and renewal after first import remain authoritative.
            return Ok(false);
        }
        for table in ["paired_clients", "secure_devices", "generations", "generation_queue", "chain_jobs", "video_upscale_jobs", "mesh_workflow_jobs", "collections", "tags"] {
            let populated: bool = tx.query_row(&format!("SELECT EXISTS(SELECT 1 FROM {table})"), [], |r| r.get(0))?;
            ensure!(!populated, "secure bootstrap requires an empty authority and content store");
        }
        ensure!(client.id == device.client_id && client.server_instance_id == device.server_instance_id && client.credential_hash == device.credential_hash, "secure bootstrap device binding mismatch");
        let existing_instance: bool = tx.query_row("SELECT EXISTS(SELECT 1 FROM settings WHERE key = 'server.instance_id' OR key LIKE 'server.instance_id.%')", [], |r| r.get(0))?;
        ensure!(!existing_instance, "secure bootstrap refuses an existing server instance");
        prepare_identity(true)?;
        tx.execute("INSERT INTO settings(profile,key,value,value_type,updated_at_ms) VALUES(?1,?2,?3,'string',?4)",params![crate::DEFAULT_PROFILE,instance_key,client.server_instance_id,client.created_at_ms])?;
        tx.execute("INSERT INTO paired_clients(id,server_instance_id,name,client_kind,credential_hash,created_at_ms,last_used_at_ms) VALUES(?1,?2,?3,?4,?5,?6,NULL)", params![client.id,client.server_instance_id,client.name,client.client_kind,client.credential_hash.as_slice(),client.created_at_ms])?;
        tx.execute("INSERT INTO secure_devices(client_id,grant_id,signing_key,noise_key,expires_at,epoch) VALUES(?1,?2,?3,?4,?5,?6)",params![device.client_id,device.grant_id,device.signing_key,device.noise_key,device.expires_at,device.epoch])?;
        tx.execute("INSERT INTO secure_bootstrap(singleton,bootstrap_id,digest,server_instance_id) VALUES(1,?1,?2,?3)",params![id,digest,client.server_instance_id])?;
        tx.commit()?;
        Ok(true)
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    fn fixture() -> (MetadataDb, PairedClient, SecureDevice) {
        let db = MetadataDb::open_in_memory().unwrap();
        let client = PairedClient {
            id: "client".into(),
            server_instance_id: "instance".into(),
            name: "CLI".into(),
            client_kind: "cli".into(),
            credential_hash: [7; 32],
            created_at_ms: 1,
            last_used_at_ms: None,
        };
        let device = SecureDevice {
            client_id: client.id.clone(),
            server_instance_id: client.server_instance_id.clone(),
            grant_id: "a".repeat(32),
            signing_key: "b".repeat(64),
            noise_key: "c".repeat(64),
            credential_hash: client.credential_hash,
            expires_at: 2000,
            epoch: 0,
        };
        (db, client, device)
    }
    #[test]
    fn exact_replay_never_resurrects_revoked_or_overwrites_renewed_authority() {
        let (db, c, d) = fixture();
        assert!(import(
            &db,
            "id",
            "digest",
            "server.instance_id.7680",
            &c,
            &d,
            |new| {
                assert!(new);
                Ok(())
            }
        )
        .unwrap());
        crate::secure_devices::SecureDevices::new(&db)
            .renew("instance", &d.grant_id, 9000)
            .unwrap();
        assert!(!import(
            &db,
            "id",
            "digest",
            "server.instance_id.7680",
            &c,
            &d,
            |new| {
                assert!(!new);
                Ok(())
            }
        )
        .unwrap());
        assert_eq!(
            crate::secure_devices::SecureDevices::new(&db)
                .list("instance")
                .unwrap()[0]
                .expires_at,
            9000
        );
        crate::paired_clients::PairedClients::new(&db)
            .revoke("instance", "client")
            .unwrap();
        assert!(!import(
            &db,
            "id",
            "digest",
            "server.instance_id.7680",
            &c,
            &d,
            |_| Ok(())
        )
        .unwrap());
        assert!(crate::secure_devices::SecureDevices::new(&db)
            .list("instance")
            .unwrap()
            .is_empty());
        assert!(import(
            &db,
            "other",
            "digest",
            "server.instance_id.7680",
            &c,
            &d,
            |_| panic!("conflict must not touch identity")
        )
        .is_err());
    }
    #[test]
    fn preparation_failure_rolls_back_and_existing_authority_is_never_overwritten() {
        let (db, c, d) = fixture();
        assert!(import(
            &db,
            "id",
            "digest",
            "server.instance_id.7680",
            &c,
            &d,
            |_| anyhow::bail!("disk failure")
        )
        .is_err());
        assert!(crate::secure_devices::SecureDevices::new(&db)
            .list("instance")
            .unwrap()
            .is_empty());
        crate::secure_devices::SecureDevices::new(&db)
            .insert(&c, &d)
            .unwrap();
        assert!(import(
            &db,
            "id",
            "digest",
            "server.instance_id.7680",
            &c,
            &d,
            |_| panic!("existing authority must not touch identity")
        )
        .is_err());
    }
}
