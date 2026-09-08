//! Repairable projection of downloadable files associated with gallery rows.

use anyhow::{ensure, Result};
use rusqlite::{params, OptionalExtension};

use crate::MetadataDb;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AssetStorage {
    EmbeddedGlb { role: String },
    Sidecar { filename: String },
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GenerationAssetRecord {
    pub asset: mold_core::GenerationAsset,
    pub storage: AssetStorage,
}

impl AssetStorage {
    fn columns(&self) -> (&'static str, &str) {
        match self {
            Self::EmbeddedGlb { role } => ("embedded_glb", role),
            Self::Sidecar { filename } => ("sidecar", filename),
        }
    }

    fn from_columns(kind: &str, locator: String) -> rusqlite::Result<Self> {
        match kind {
            "embedded_glb" => Ok(Self::EmbeddedGlb { role: locator }),
            "sidecar" => Ok(Self::Sidecar { filename: locator }),
            other => Err(rusqlite::Error::FromSqlConversionFailure(
                0,
                rusqlite::types::Type::Text,
                format!("unknown generation asset storage kind {other}").into(),
            )),
        }
    }
}

fn validate(record: &GenerationAssetRecord) -> Result<()> {
    let asset = &record.asset;
    ensure!(
        !asset.asset_id.is_empty() && asset.asset_id.len() <= 128,
        "invalid asset id"
    );
    ensure!(
        !asset.role.is_empty() && asset.role.len() <= 64,
        "invalid asset role"
    );
    ensure!(
        !asset.display_name.is_empty() && asset.display_name.len() <= 255,
        "invalid asset display name"
    );
    ensure!(
        !asset.media_type.is_empty() && asset.media_type.len() <= 128,
        "invalid asset media type"
    );
    ensure!(
        asset.width.is_none_or(|value| value > 0),
        "invalid asset width"
    );
    ensure!(
        asset.height.is_none_or(|value| value > 0),
        "invalid asset height"
    );
    ensure!(
        asset.sha256.len() == 64 && asset.sha256.bytes().all(|byte| byte.is_ascii_hexdigit()),
        "asset sha256 must be a hexadecimal digest"
    );
    let locator = record.storage.columns().1;
    ensure!(
        !locator.is_empty() && locator.len() <= 1024,
        "invalid asset locator"
    );
    Ok(())
}

/// Replace the complete inventory for one generation in a transaction.
pub fn replace_for_generation(
    db: &MetadataDb,
    generation_id: i64,
    media_version: &str,
    records: &[GenerationAssetRecord],
) -> Result<()> {
    ensure!(generation_id > 0, "generation id must be positive");
    ensure!(
        !media_version.is_empty() && media_version.len() <= 128,
        "invalid generation asset media version"
    );
    for record in records {
        validate(record)?;
    }
    db.with_conn(|conn| {
        let tx = conn.unchecked_transaction()?;
        tx.execute(
            "DELETE FROM generation_assets WHERE generation_id = ?1",
            params![generation_id],
        )?;
        for record in records {
            let asset = &record.asset;
            let (storage_kind, locator) = record.storage.columns();
            tx.execute(
                "INSERT INTO generation_assets
                    (generation_id, asset_id, role, display_name, media_type,
                     size_bytes, sha256, width, height, storage_kind, locator)
                 VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11)",
                params![
                    generation_id,
                    asset.asset_id,
                    asset.role,
                    asset.display_name,
                    asset.media_type,
                    i64::try_from(asset.size_bytes)?,
                    asset.sha256,
                    asset.width.map(i64::from),
                    asset.height.map(i64::from),
                    storage_kind,
                    locator,
                ],
            )?;
        }
        tx.execute(
            "INSERT INTO generation_asset_scans (generation_id, media_version)
             VALUES (?1, ?2)
             ON CONFLICT(generation_id) DO UPDATE SET media_version = excluded.media_version",
            params![generation_id, media_version],
        )?;
        tx.commit()?;
        Ok(())
    })
}

pub fn scan_is_current(db: &MetadataDb, generation_id: i64, media_version: &str) -> Result<bool> {
    db.with_conn(|conn| {
        Ok(conn
            .query_row(
                "SELECT media_version FROM generation_asset_scans WHERE generation_id = ?1",
                params![generation_id],
                |row| row.get::<_, String>(0),
            )
            .optional()?
            .is_some_and(|stored| stored == media_version))
    })
}

pub fn list_for_generation(
    db: &MetadataDb,
    generation_id: i64,
) -> Result<Vec<GenerationAssetRecord>> {
    db.with_conn(|conn| {
        let mut statement = conn.prepare(
            "SELECT asset_id, role, display_name, media_type, size_bytes,
                    sha256, width, height, storage_kind, locator
               FROM generation_assets
              WHERE generation_id = ?1
              ORDER BY role COLLATE NOCASE, asset_id",
        )?;
        let rows = statement.query_map(params![generation_id], |row| {
            let kind: String = row.get(8)?;
            let locator: String = row.get(9)?;
            Ok(GenerationAssetRecord {
                asset: mold_core::GenerationAsset {
                    asset_id: row.get(0)?,
                    role: row.get(1)?,
                    display_name: row.get(2)?,
                    media_type: row.get(3)?,
                    size_bytes: row.get::<_, i64>(4)?.try_into().map_err(|error| {
                        rusqlite::Error::FromSqlConversionFailure(
                            4,
                            rusqlite::types::Type::Integer,
                            Box::new(error),
                        )
                    })?,
                    sha256: row.get(5)?,
                    width: row.get::<_, Option<i64>>(6)?.map(|value| value as u32),
                    height: row.get::<_, Option<i64>>(7)?.map(|value| value as u32),
                },
                storage: AssetStorage::from_columns(&kind, locator)?,
            })
        })?;
        rows.collect::<rusqlite::Result<Vec<_>>>()
            .map_err(Into::into)
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{GenerationRecord, RecordSource};
    use mold_core::OutputFormat;

    fn generation(db: &MetadataDb) -> i64 {
        db.upsert(&GenerationRecord::from_save(
            std::path::Path::new("/gallery"),
            "chair.glb",
            OutputFormat::Glb,
            crate::metadata_io::synthesize_from_filename("chair.glb", 1),
            RecordSource::Server,
            1,
        ))
        .unwrap()
    }

    fn asset(id: &str, role: &str) -> GenerationAssetRecord {
        GenerationAssetRecord {
            asset: mold_core::GenerationAsset {
                asset_id: id.into(),
                role: role.into(),
                display_name: format!("chair-{role}.png"),
                media_type: "image/png".into(),
                size_bytes: 123,
                sha256: "a".repeat(64),
                width: Some(1024),
                height: Some(1024),
            },
            storage: AssetStorage::EmbeddedGlb { role: role.into() },
        }
    }

    #[test]
    fn replacement_is_complete_ordered_and_cascades_with_generation() {
        let db = MetadataDb::open_in_memory().unwrap();
        let id = generation(&db);
        replace_for_generation(
            &db,
            id,
            "1:123",
            &[
                asset("mr", "metallic_roughness"),
                asset("albedo", "base_color"),
            ],
        )
        .unwrap();
        assert!(scan_is_current(&db, id, "1:123").unwrap());
        let listed = list_for_generation(&db, id).unwrap();
        assert_eq!(
            listed
                .iter()
                .map(|r| r.asset.asset_id.as_str())
                .collect::<Vec<_>>(),
            ["albedo", "mr"]
        );

        replace_for_generation(&db, id, "2:123", &[asset("albedo", "base_color")]).unwrap();
        assert_eq!(list_for_generation(&db, id).unwrap().len(), 1);
        db.with_conn(|conn| {
            conn.execute("DELETE FROM generations WHERE id = ?1", params![id])?;
            Ok(())
        })
        .unwrap();
        assert!(list_for_generation(&db, id).unwrap().is_empty());
    }
}
