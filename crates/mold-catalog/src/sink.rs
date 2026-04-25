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

use crate::entry::{CatalogEntry, FamilyRole, Shard, SHARD_SCHEMA};

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
