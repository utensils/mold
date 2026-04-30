use mold_catalog::entry::Shard;
use std::path::PathBuf;

fn shard_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("data/catalog")
}

#[test]
fn every_committed_shard_round_trips_byte_identical() {
    let dir = shard_dir();
    let mut count = 0usize;
    for entry in std::fs::read_dir(&dir).expect("read shard dir") {
        let entry = entry.unwrap();
        let path = entry.path();
        if path.extension().and_then(|e| e.to_str()) != Some("json") {
            continue;
        }
        let original = std::fs::read_to_string(&path).unwrap();
        let parsed: Shard = serde_json::from_str(&original).unwrap();
        let re = serde_json::to_string_pretty(&parsed).unwrap() + "\n";
        assert_eq!(
            re.trim_end(),
            original.trim_end(),
            "shard {} is not in canonical form — run `mold catalog refresh --commit-to-repo --dry-run` to normalize",
            path.display()
        );
        count += 1;
    }
    assert!(count >= 9, "expected 9+ shards, found {count}");
}
