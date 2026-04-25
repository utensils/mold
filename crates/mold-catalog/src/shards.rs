//! Embedded catalog shards. The `MOLD_CATALOG_DIR` env var is stamped at
//! build time by `build.rs`, so `rust-embed` always finds something —
//! either the committed shards or stubs.

use rust_embed::RustEmbed;

use crate::entry::Shard;

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
