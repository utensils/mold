//! Persistent server instance identity.
//!
//! Every `mold serve` endpoint gets a stable UUIDv4, generated on first boot
//! and persisted in the metadata DB's settings table. Clients (desktop
//! multi-host, discovery) use it to recognise the same server across
//! hostname/IP changes — it is surfaced in `GET /api/status` and the mDNS
//! `id` TXT record.
//!
//! The identity is scoped per (data dir, serving port): two servers sharing
//! one `mold.db` (e.g. the desktop's embedded server next to a standalone
//! `mold serve` on the same box) listen on different ports and must report
//! distinct ids, while one server keeps its id across restarts and address
//! changes because its port is stable.

use mold_db::{MetadataDb, Settings, DEFAULT_PROFILE};

/// Legacy port-less settings key from before identity was port-scoped. Never
/// written anymore; its value is adopted (renamed) by the first port that
/// resolves an id so existing installs keep their identity.
pub const SERVER_INSTANCE_ID_KEY: &str = "server.instance_id";

/// Port-scoped settings key holding the installation UUID. Like
/// [`mold_db::settings::ACTIVE_PROFILE`], this is identity metadata — it is
/// always read and written under [`DEFAULT_PROFILE`] so the id never varies
/// with `MOLD_PROFILE`.
pub fn instance_id_key(port: u16) -> String {
    format!("{SERVER_INSTANCE_ID_KEY}.{port}")
}

/// Resolve the persistent instance id for the server listening on `port`,
/// generating and storing one on first boot. A legacy port-less id (written
/// by older versions) is adopted by the first port that asks — renamed to the
/// port-scoped key so a second server on the same DB mints its own id instead
/// of colliding. When the metadata DB is unavailable (`MOLD_DB_DISABLE=1`,
/// open failure), falls back to a fresh ephemeral UUID — callers resolve once
/// at startup and hold the result, making the fallback per-process.
pub fn resolve_instance_id(db: Option<&MetadataDb>, port: u16) -> String {
    if let Some(db) = db {
        let settings = Settings::for_profile(db, DEFAULT_PROFILE);
        let key = instance_id_key(port);
        match settings.get_str(&key) {
            Ok(Some(id)) if !id.trim().is_empty() => return id,
            Ok(_) => {
                // Adopt the legacy port-less id when present so pre-scoping
                // installs keep their identity across the upgrade.
                let adopted = match settings.get_str(SERVER_INSTANCE_ID_KEY) {
                    Ok(Some(id)) if !id.trim().is_empty() => Some(id),
                    Ok(_) => None,
                    Err(e) => {
                        tracing::warn!("failed to read legacy server instance id: {e:#}");
                        None
                    }
                };
                let id = adopted
                    .clone()
                    .unwrap_or_else(|| uuid::Uuid::new_v4().to_string());
                if let Err(e) = settings.set_str(&key, &id) {
                    tracing::warn!(
                        "failed to persist server instance id: {e:#} — using ephemeral id"
                    );
                } else if adopted.is_some() {
                    // Delete the legacy key only after the port-scoped copy is
                    // durable — a second port must mint its own id, not adopt
                    // the same legacy value.
                    if let Err(e) = settings.delete(SERVER_INSTANCE_ID_KEY) {
                        tracing::warn!("failed to retire legacy server instance id key: {e:#}");
                    }
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
        let first = resolve_instance_id(Some(&db), 7680);
        let second = resolve_instance_id(Some(&db), 7680);
        assert_eq!(first, second);
        uuid::Uuid::parse_str(&first).expect("instance id must be a valid UUID");
    }

    #[test]
    fn two_ports_on_one_db_get_distinct_ids() {
        // Two live servers sharing one mold.db (desktop embedded server +
        // standalone `mold serve`, or one process per GPU) must not report
        // the same identity — clients dedupe hosts by this id.
        let db = MetadataDb::open_in_memory().unwrap();
        let a = resolve_instance_id(Some(&db), 7680);
        let b = resolve_instance_id(Some(&db), 7681);
        assert_ne!(a, b);
        // Each port's id stays stable across restarts.
        assert_eq!(resolve_instance_id(Some(&db), 7680), a);
        assert_eq!(resolve_instance_id(Some(&db), 7681), b);
    }

    #[test]
    fn legacy_portless_id_is_adopted_by_the_first_port() {
        // Pre-scoping installs persisted a single port-less id. The first
        // port that resolves must keep that identity (desktop saved-host
        // dedupe depends on it surviving the upgrade)…
        let db = MetadataDb::open_in_memory().unwrap();
        let settings = Settings::for_profile(&db, DEFAULT_PROFILE);
        settings
            .set_str(SERVER_INSTANCE_ID_KEY, "legacy-id-1234")
            .unwrap();

        let first = resolve_instance_id(Some(&db), 7680);
        assert_eq!(first, "legacy-id-1234");
        assert_eq!(
            settings.get_str(&instance_id_key(7680)).unwrap().as_deref(),
            Some("legacy-id-1234")
        );
        // …and the legacy key is retired (renamed, not copied) so a second
        // port mints a fresh id instead of adopting the same one.
        assert_eq!(settings.get_str(SERVER_INSTANCE_ID_KEY).unwrap(), None);
        let second = resolve_instance_id(Some(&db), 7681);
        assert_ne!(second, "legacy-id-1234");
    }

    #[test]
    fn instance_id_is_stored_under_the_default_profile() {
        // Identity must be pinned to the default profile: a view onto any
        // other profile must not see (or shadow) the id.
        let db = MetadataDb::open_in_memory().unwrap();
        let id = resolve_instance_id(Some(&db), 7680);
        let default_view = Settings::for_profile(&db, DEFAULT_PROFILE);
        assert_eq!(
            default_view
                .get_str(&instance_id_key(7680))
                .unwrap()
                .as_deref(),
            Some(id.as_str())
        );
        let other_view = Settings::for_profile(&db, "staging");
        assert_eq!(other_view.get_str(&instance_id_key(7680)).unwrap(), None);
    }

    #[test]
    fn instance_id_ephemeral_fallback_without_db() {
        let id = resolve_instance_id(None, 7680);
        uuid::Uuid::parse_str(&id).expect("ephemeral id must be a valid UUID");
        // No storage → each resolution mints a fresh id; run_server resolves
        // once and holds it, which is what makes the fallback per-process.
        assert_ne!(id, resolve_instance_id(None, 7680));
    }
}
