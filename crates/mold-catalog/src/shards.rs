//! Embedded catalog shards. The `MOLD_CATALOG_DIR` env var is stamped at
//! build time by `build.rs`, so `rust-embed` always finds something —
//! either the committed shards or stubs.

use std::path::Path;

use rust_embed::RustEmbed;
use serde::de::Error as _;

use crate::entry::Shard;
use crate::families::Family;
use crate::sink::upsert_family;

#[derive(RustEmbed)]
#[folder = "$MOLD_CATALOG_DIR"]
#[exclude = ".*"]
pub struct EmbeddedShards;

/// Yield every embedded shard parsed into the typed form. Errors on
/// individual shards are returned alongside the family name so callers
/// can log+continue rather than abort.
pub fn iter_shards() -> impl Iterator<Item = (String, Result<Shard, serde_json::Error>)> {
    EmbeddedShards::iter().map(|name| {
        let bytes = EmbeddedShards::get(&name).expect("embedded").data;
        let parsed: Result<Shard, _> = serde_json::from_slice(&bytes);
        (name.to_string(), parsed)
    })
}

pub fn shard_count() -> usize {
    EmbeddedShards::iter().count()
}

#[derive(Debug, thiserror::Error)]
pub enum SeedError {
    #[error("io: {0}")]
    Io(#[from] std::io::Error),
    #[error("decode: {0}")]
    Decode(#[from] serde_json::Error),
    #[error("rusqlite: {0}")]
    Sql(#[from] rusqlite::Error),
    #[error("sink: {0}")]
    Sink(#[from] crate::sink::SinkError),
}

/// Idempotent: if the catalog table already has any rows, returns Ok(())
/// without touching anything. Otherwise:
///
/// - If `disk_dir` is provided AND contains shards, seed from disk.
/// - Else, seed from rust-embedded shards.
///
/// Stub shards (zero entries) are silently skipped, so a fresh checkout
/// that hasn't yet run `mold catalog refresh` quietly stays empty.
pub fn seed_db_from_embedded_if_empty(
    conn: &rusqlite::Connection,
    disk_dir: Option<&Path>,
) -> Result<(), SeedError> {
    let count: i64 = conn.query_row("SELECT COUNT(*) FROM catalog", [], |r| r.get(0))?;
    if count > 0 {
        return Ok(());
    }

    let mut shards: Vec<Shard> = Vec::new();
    if let Some(dir) = disk_dir {
        if dir.exists() {
            for entry in std::fs::read_dir(dir)? {
                let entry = entry?;
                let path = entry.path();
                if path.extension().and_then(|s| s.to_str()) != Some("json") {
                    continue;
                }
                let body = std::fs::read_to_string(&path)?;
                let shard: Shard = serde_json::from_str(&body)?;
                shards.push(shard);
            }
        }
    }
    if shards.is_empty() {
        for (_name, parsed) in iter_shards() {
            shards.push(parsed?);
        }
    }

    for shard in shards {
        if shard.entries.is_empty() {
            continue;
        }
        let family = Family::from_str(&shard.family)
            .map_err(|e| serde_json::Error::custom(format!("unknown family in shard: {e}")))?;
        upsert_family(conn, family, &shard.entries)?;
    }
    Ok(())
}
