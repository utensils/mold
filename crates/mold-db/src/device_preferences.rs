//! Machine-wide desired device enablement.
//!
//! Absence is meaningful: a device without a row is enabled by default.
//! Discovery therefore performs reads only. Rows are created or updated only
//! by an explicit administrative change.

use std::collections::BTreeMap;
use std::time::{SystemTime, UNIX_EPOCH};

use anyhow::{bail, Result};
use rusqlite::params;

use crate::MetadataDb;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DevicePreference {
    pub device_id: String,
    pub desired_enabled: bool,
    pub updated_at: i64,
}

/// Borrowed machine-wide view over `device_preferences`.
pub struct DevicePreferences<'a> {
    db: &'a MetadataDb,
}

impl<'a> DevicePreferences<'a> {
    pub fn new(db: &'a MetadataDb) -> Self {
        Self { db }
    }

    /// Return the explicit preference, or `None` when enabled-by-default
    /// applies.
    pub fn get(&self, device_id: &str) -> Result<Option<bool>> {
        self.db.with_conn(|conn| {
            let mut stmt = conn
                .prepare("SELECT desired_enabled FROM device_preferences WHERE device_id = ?1")?;
            let mut rows = stmt.query(params![device_id])?;
            Ok(rows
                .next()?
                .map(|row| row.get::<_, i64>(0))
                .transpose()?
                .map(|value| value != 0))
        })
    }

    /// Load every explicit preference, including entries for devices that are
    /// currently absent and may return later.
    pub fn list(&self) -> Result<BTreeMap<String, DevicePreference>> {
        self.db.with_conn(|conn| {
            let mut stmt = conn.prepare(
                "SELECT device_id, desired_enabled, updated_at
                 FROM device_preferences
                 ORDER BY device_id",
            )?;
            let rows = stmt.query_map([], |row| {
                Ok(DevicePreference {
                    device_id: row.get(0)?,
                    desired_enabled: row.get::<_, i64>(1)? != 0,
                    updated_at: row.get(2)?,
                })
            })?;
            let mut preferences = BTreeMap::new();
            for row in rows {
                let preference = row?;
                preferences.insert(preference.device_id.clone(), preference);
            }
            Ok(preferences)
        })
    }

    /// Persist one explicit administrative choice.
    pub fn set(&self, device_id: &str, desired_enabled: bool) -> Result<()> {
        if device_id.trim().is_empty() {
            bail!("device id must not be empty");
        }
        let updated_at = now_ms();
        self.db.with_conn(|conn| {
            conn.execute(
                "INSERT INTO device_preferences (device_id, desired_enabled, updated_at)
                 VALUES (?1, ?2, ?3)
                 ON CONFLICT(device_id) DO UPDATE SET
                     desired_enabled = excluded.desired_enabled,
                     updated_at = excluded.updated_at",
                params![device_id, i64::from(desired_enabled), updated_at],
            )?;
            Ok(())
        })
    }
}

fn now_ms() -> i64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|duration| duration.as_millis() as i64)
        .unwrap_or(0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn missing_row_means_enabled_by_default_without_writing() {
        let db = MetadataDb::open_in_memory().unwrap();
        let preferences = DevicePreferences::new(&db);

        assert_eq!(
            preferences
                .get("cuda:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa")
                .unwrap(),
            None
        );
        assert!(preferences.list().unwrap().is_empty());
    }

    #[test]
    fn explicit_choices_round_trip_machine_wide() {
        let db = MetadataDb::open_in_memory().unwrap();
        let preferences = DevicePreferences::new(&db);
        let id = "cuda:bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb";

        preferences.set(id, false).unwrap();
        assert_eq!(preferences.get(id).unwrap(), Some(false));
        preferences.set(id, true).unwrap();
        assert_eq!(preferences.get(id).unwrap(), Some(true));

        let rows = preferences.list().unwrap();
        assert_eq!(rows.len(), 1);
        assert!(rows[id].desired_enabled);
        assert!(rows[id].updated_at > 0);
    }

    #[test]
    fn preferences_are_not_profile_scoped() {
        let db = MetadataDb::open_in_memory().unwrap();
        let id = "cuda:cccccccccccccccccccccccccccccccc";
        DevicePreferences::new(&db).set(id, false).unwrap();

        crate::Settings::for_profile(&db, "one")
            .set_bool("unrelated", true)
            .unwrap();
        crate::Settings::for_profile(&db, "two")
            .set_bool("unrelated", false)
            .unwrap();

        assert_eq!(DevicePreferences::new(&db).get(id).unwrap(), Some(false));
    }
}
