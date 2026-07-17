//! Persistent server instance identity.
//!
//! Every `mold serve` installation gets a stable UUIDv4, generated on first
//! boot and persisted in the metadata DB's settings table. Clients (desktop
//! multi-host, discovery) use it to recognise the same server across
//! hostname/IP/port changes — it is surfaced in `GET /api/status` and the
//! mDNS `id` TXT record.

use mold_db::{MetadataDb, Settings, DEFAULT_PROFILE};

/// Settings key holding the installation UUID. Like
/// [`mold_db::settings::ACTIVE_PROFILE`], this is identity metadata — it is
/// always read and written under [`DEFAULT_PROFILE`] so the id never varies
/// with `MOLD_PROFILE`.
pub const SERVER_INSTANCE_ID_KEY: &str = "server.instance_id";

/// Resolve the persistent instance id, generating and storing one on first
/// boot. When the metadata DB is unavailable (`MOLD_DB_DISABLE=1`, open
/// failure), falls back to a fresh ephemeral UUID — callers resolve once at
/// startup and hold the result, making the fallback per-process.
pub fn resolve_instance_id(db: Option<&MetadataDb>) -> String {
    if let Some(db) = db {
        let settings = Settings::for_profile(db, DEFAULT_PROFILE);
        match settings.get_str(SERVER_INSTANCE_ID_KEY) {
            Ok(Some(id)) if !id.trim().is_empty() => return id,
            Ok(_) => {
                let id = uuid::Uuid::new_v4().to_string();
                if let Err(e) = settings.set_str(SERVER_INSTANCE_ID_KEY, &id) {
                    tracing::warn!(
                        "failed to persist server instance id: {e:#} — using ephemeral id"
                    );
                }
                return id;
            }
            Err(e) => {
                tracing::warn!("failed to read server instance id: {e:#} — using ephemeral id");
            }
        }
    }
    uuid::Uuid::new_v4().to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn instance_id_persists_across_resolver_calls() {
        let db = MetadataDb::open_in_memory().unwrap();
        let first = resolve_instance_id(Some(&db));
        let second = resolve_instance_id(Some(&db));
        assert_eq!(first, second);
        uuid::Uuid::parse_str(&first).expect("instance id must be a valid UUID");
    }

    #[test]
    fn instance_id_is_stored_under_the_default_profile() {
        // Identity must be pinned to the default profile: a view onto any
        // other profile must not see (or shadow) the id.
        let db = MetadataDb::open_in_memory().unwrap();
        let id = resolve_instance_id(Some(&db));
        let default_view = Settings::for_profile(&db, DEFAULT_PROFILE);
        assert_eq!(
            default_view.get_str(SERVER_INSTANCE_ID_KEY).unwrap(),
            Some(id)
        );
        let other_view = Settings::for_profile(&db, "staging");
        assert_eq!(other_view.get_str(SERVER_INSTANCE_ID_KEY).unwrap(), None);
    }

    #[test]
    fn instance_id_ephemeral_fallback_without_db() {
        let id = resolve_instance_id(None);
        uuid::Uuid::parse_str(&id).expect("ephemeral id must be a valid UUID");
        // No storage → each resolution mints a fresh id; run_server resolves
        // once and holds it, which is what makes the fallback per-process.
        assert_ne!(id, resolve_instance_id(None));
    }
}
