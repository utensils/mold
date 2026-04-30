use mold_catalog::entry::Shard;
use mold_catalog::sink::write_shard_atomic;
use tempfile::TempDir;

#[test]
fn write_shard_atomic_creates_canonical_file() {
    let dir = TempDir::new().unwrap();
    let shard = Shard {
        schema: "mold.catalog.v1".into(),
        family: "flux".into(),
        generated_at: "2026-04-25T00:00:00Z".into(),
        scanner_version: "0.9.0".into(),
        entries: vec![],
    };
    let path = write_shard_atomic(dir.path(), &shard).expect("write");
    assert!(path.exists());
    assert!(path.ends_with("flux.json"));

    // Round-trip.
    let body = std::fs::read_to_string(&path).unwrap();
    let parsed: Shard = serde_json::from_str(&body).unwrap();
    assert_eq!(parsed, shard);

    // No staging file leaked.
    let staging = dir.path().join(".staging");
    if staging.exists() {
        let leftovers: Vec<_> = std::fs::read_dir(&staging)
            .unwrap()
            .map(|e| e.unwrap().path())
            .collect();
        assert!(
            leftovers.is_empty(),
            "leftover staging files: {leftovers:?}"
        );
    }
}

#[test]
fn write_shard_overwrites_existing() {
    let dir = TempDir::new().unwrap();
    let shard1 = Shard {
        schema: "mold.catalog.v1".into(),
        family: "flux".into(),
        generated_at: "2026-04-25T00:00:00Z".into(),
        scanner_version: "0.9.0".into(),
        entries: vec![],
    };
    let path = write_shard_atomic(dir.path(), &shard1).unwrap();
    let shard2 = Shard {
        generated_at: "2026-05-01T00:00:00Z".into(),
        ..shard1
    };
    let path2 = write_shard_atomic(dir.path(), &shard2).unwrap();
    assert_eq!(path, path2);
    let body = std::fs::read_to_string(&path).unwrap();
    let parsed: Shard = serde_json::from_str(&body).unwrap();
    assert_eq!(parsed.generated_at, "2026-05-01T00:00:00Z");
}
