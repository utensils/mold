//! Two-phase commit per family.
//!
//! Phase 1 — shard file:
//! 1. Serialize entries to canonical JSON (sorted by `(family_role,
//!    download_count desc, name)`, indented 2 spaces, trailing newline).
//! 2. Write to `<dest>/.staging/<family>.json`.
//! 3. `fs::rename` to `<dest>/<family>.json` (POSIX atomic on the same fs).
//!
//! Phase 2 — DB upsert (added in Task 18).
//!
//! Sort ordering is canonical: a re-scan that finds zero changes produces
//! a byte-identical shard, so `git diff` shows nothing — `mold catalog
//! refresh` weekly does not pollute commit history.

use std::cmp::Ordering;
use std::fs;
use std::path::{Path, PathBuf};

use mold_db::catalog::{upsert_entries, CatalogRow};
use rusqlite::Connection;

use crate::entry::{CatalogEntry, FamilyRole, Shard, SHARD_SCHEMA};
use crate::families::Family;

#[derive(Debug, thiserror::Error)]
pub enum SinkError {
    #[error("io: {0}")]
    Io(#[from] std::io::Error),
    #[error("serialize: {0}")]
    Serialize(#[from] serde_json::Error),
}

pub fn canonicalize_entries(entries: &mut [CatalogEntry]) {
    entries.sort_by(|a, b| {
        let role = match (a.family_role, b.family_role) {
            (FamilyRole::Foundation, FamilyRole::Foundation) => Ordering::Equal,
            (FamilyRole::Foundation, _) => Ordering::Less,
            (_, FamilyRole::Foundation) => Ordering::Greater,
            _ => Ordering::Equal,
        };
        role.then_with(|| b.download_count.cmp(&a.download_count))
            .then_with(|| a.name.cmp(&b.name))
            .then_with(|| a.id.as_str().cmp(b.id.as_str()))
    });
}

pub fn write_shard_atomic(dir: &Path, shard: &Shard) -> Result<PathBuf, SinkError> {
    debug_assert_eq!(
        shard.schema, SHARD_SCHEMA,
        "shard.schema must be {SHARD_SCHEMA}"
    );
    fs::create_dir_all(dir)?;
    let staging = dir.join(".staging");
    fs::create_dir_all(&staging)?;
    let staged = staging.join(format!("{}.json", shard.family));
    let body = serde_json::to_string_pretty(shard)? + "\n";
    fs::write(&staged, &body)?;

    // Validate round-trip before flipping the rename.
    let _: Shard = serde_json::from_str(&body)?;

    let final_path = dir.join(format!("{}.json", shard.family));
    fs::rename(&staged, &final_path)?;
    // Best-effort cleanup if the staging dir is empty.
    let _ = fs::remove_dir(&staging);
    Ok(final_path)
}

pub fn build_shard(
    family: &str,
    scanner_version: &str,
    generated_at: &str,
    mut entries: Vec<CatalogEntry>,
) -> Shard {
    canonicalize_entries(&mut entries);
    Shard {
        schema: SHARD_SCHEMA.to_string(),
        family: family.to_string(),
        generated_at: generated_at.to_string(),
        scanner_version: scanner_version.to_string(),
        entries,
    }
}

/// Convert a [`CatalogEntry`] into a [`CatalogRow`] suitable for DB upsert.
pub fn entry_to_row(e: &CatalogEntry) -> Result<CatalogRow, SinkError> {
    Ok(CatalogRow {
        id: e.id.as_str().to_string(),
        source: serde_json::to_value(e.source)?
            .as_str()
            .unwrap_or_default()
            .to_string(),
        source_id: e.source_id.clone(),
        name: e.name.clone(),
        author: e.author.clone(),
        family: e.family.as_str().to_string(),
        family_role: serde_json::to_value(e.family_role)?
            .as_str()
            .unwrap_or_default()
            .to_string(),
        sub_family: e.sub_family.clone(),
        modality: serde_json::to_value(e.modality)?
            .as_str()
            .unwrap_or_default()
            .to_string(),
        kind: serde_json::to_value(e.kind)?
            .as_str()
            .unwrap_or_default()
            .to_string(),
        file_format: serde_json::to_value(e.file_format)?
            .as_str()
            .unwrap_or_default()
            .to_string(),
        bundling: serde_json::to_value(e.bundling)?
            .as_str()
            .unwrap_or_default()
            .to_string(),
        size_bytes: e.size_bytes.map(|v| v as i64),
        download_count: e.download_count as i64,
        rating: e.rating.map(|v| v as f64),
        likes: e.likes as i64,
        nsfw: if e.nsfw { 1 } else { 0 },
        thumbnail_url: e.thumbnail_url.clone(),
        description: e.description.clone(),
        license: e.license.clone(),
        license_flags: Some(serde_json::to_string(&e.license_flags)?),
        tags: Some(serde_json::to_string(&e.tags)?),
        companions: Some(serde_json::to_string(&e.companions)?),
        download_recipe: serde_json::to_string(&e.download_recipe)?,
        engine_phase: e.engine_phase as i64,
        created_at: e.created_at,
        updated_at: e.updated_at,
        added_at: e.added_at,
    })
}

/// Replace every row for `family` in the DB with the given entries.
///
/// Wraps [`mold_db::catalog::upsert_entries`] — the delete + insert + FTS
/// rebuild all happen in one transaction.
pub fn upsert_family(
    conn: &Connection,
    family: Family,
    entries: &[CatalogEntry],
) -> Result<(), SinkError> {
    let rows: Vec<CatalogRow> = entries
        .iter()
        .map(entry_to_row)
        .collect::<Result<Vec<_>, _>>()?;
    upsert_entries(conn, family.as_str(), &rows)
        .map_err(|e| SinkError::Io(std::io::Error::other(e)))?;
    Ok(())
}
