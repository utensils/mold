use mold_catalog::shards::{iter_shards, shard_count};

#[test]
fn nine_families_embedded() {
    assert_eq!(shard_count(), 9, "expected one shard per family");
}

#[test]
fn every_shard_parses_as_typed_form() {
    for (name, result) in iter_shards() {
        result.unwrap_or_else(|e| panic!("shard {name} failed to parse: {e}"));
    }
}
