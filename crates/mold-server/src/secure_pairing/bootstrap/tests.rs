use super::*;

fn fixture() -> (tempfile::TempDir, mold_db::MetadataDb, BootstrapInput) {
    let home = tempfile::tempdir().unwrap();
    let db = mold_db::MetadataDb::open_in_memory().unwrap();
    let host = Identity::from_bytes(&[1; 64]).unwrap();
    let binding = Identity::from_bytes(&[2; 64]).unwrap().public_binding();
    let mut grant = DeviceGrant {
        id: "a".repeat(32),
        expires_at: unix_time() + 3600,
        epoch: 0,
        signature: String::new(),
    };
    grant.signature = hex(&host.sign(
        &mold_secure::grant_bytes(
            &mold_secure::namespace(&host.signing_public()),
            &mold_secure::decode_hex::<32>(&binding.signing_key).unwrap(),
            &mold_secure::decode_hex::<32>(&binding.noise_key).unwrap(),
            &grant.id,
            grant.expires_at,
            grant.epoch,
        )
        .unwrap(),
    ));
    let input = BootstrapInput {
        version: 1,
        id: "b".repeat(32),
        instance_id: uuid::Uuid::new_v4().to_string(),
        host_identity: hex(&host.to_bytes()),
        client: BootstrapClient {
            binding,
            grant,
            api_key: format!("mold_pair_{}", "c".repeat(64)),
            name: "RunPod CLI".into(),
            kind: "cli".into(),
        },
    };
    (home, db, input)
}
fn apply(home: &Path, db: &mold_db::MetadataDb, input: &BootstrapInput) -> Result<()> {
    input.apply(home, db, 7680, &home.join("output"))
}

#[test]
fn creates_pinned_identity_instance_and_bound_device_once() {
    let (home, db, input) = fixture();
    apply(home.path(), &db, &input).unwrap();
    assert_eq!(
        crate::instance::resolve_instance_id(Some(&db), 7680),
        input.instance_id
    );
    let identity =
        identity_file::read_host_identity(&home.path().join("secure/host.identity")).unwrap();
    assert_eq!(hex(&identity.to_bytes()), input.host_identity);
    let devices = mold_db::secure_devices::SecureDevices::new(&db);
    let device = devices.list(&input.instance_id).unwrap().pop().unwrap();
    assert_eq!(device.signing_key, input.client.binding.signing_key);
    assert_eq!(device.noise_key, input.client.binding.noise_key);
    assert_eq!(
        device.credential_hash,
        <[u8; 32]>::from(Sha256::digest(input.client.api_key.as_bytes()))
    );
    assert!(!home.path().join("secure/bootstrap.intent").exists());
    devices
        .renew(
            &input.instance_id,
            &input.client.grant.id,
            input.client.grant.expires_at + 8000,
        )
        .unwrap();
    apply(home.path(), &db, &input).unwrap();
    assert_eq!(
        devices.list(&input.instance_id).unwrap()[0].expires_at,
        input.client.grant.expires_at + 8000
    );
    mold_db::paired_clients::PairedClients::new(&db)
        .revoke(&input.instance_id, &input.client.grant.id)
        .unwrap();
    apply(home.path(), &db, &input).unwrap();
    assert!(devices.list(&input.instance_id).unwrap().is_empty());
}

#[test]
fn journal_recovers_file_written_before_database_commit() {
    let (home, db, input) = fixture();
    let canonical = serde_json::to_vec(&input).unwrap();
    let digest = hex(&Sha256::digest(canonical));
    let identity =
        Identity::from_bytes(&mold_secure::decode_hex::<64>(&input.host_identity).unwrap())
            .unwrap();
    let intent = format!(
        "{}:{digest}:{}",
        input.id,
        mold_secure::namespace(&identity.signing_public())
    );
    identity_file::write_private_file(
        &home.path().join("secure/bootstrap.intent"),
        intent.as_bytes(),
        false,
    )
    .unwrap();
    identity_file::write_private_file(
        &home.path().join("secure/host.identity"),
        &identity.to_bytes(),
        false,
    )
    .unwrap();
    apply(home.path(), &db, &input).unwrap();
    assert_eq!(
        mold_db::secure_devices::SecureDevices::new(&db)
            .list(&input.instance_id)
            .unwrap()
            .len(),
        1
    );
    assert!(!home.path().join("secure/bootstrap.intent").exists());
    // A lost DB must never reinterpret an existing identity as a fresh server.
    let lost = mold_db::MetadataDb::open_in_memory().unwrap();
    assert!(apply(home.path(), &lost, &input).is_err());
}

#[test]
fn conflicting_and_unjournaled_identities_are_never_overwritten() {
    let (home, db, mut input) = fixture();
    let existing = Identity::from_bytes(&[3; 64]).unwrap();
    identity_file::write_private_file(
        &home.path().join("secure/host.identity"),
        &existing.to_bytes(),
        false,
    )
    .unwrap();
    assert!(apply(home.path(), &db, &input).is_err());
    assert_eq!(
        identity_file::read_host_identity(&home.path().join("secure/host.identity"))
            .unwrap()
            .to_bytes(),
        existing.to_bytes()
    );
    assert!(mold_db::secure_devices::SecureDevices::new(&db)
        .list(&input.instance_id)
        .unwrap()
        .is_empty());
    let (home, db, original) = fixture();
    apply(home.path(), &db, &original).unwrap();
    input.id = original.id.clone();
    assert!(apply(home.path(), &db, &input).is_err());
    assert_eq!(
        crate::instance::resolve_instance_id(Some(&db), 7680),
        original.instance_id
    );
}

#[test]
fn forged_grant_and_credential_fail_before_writing_identity() {
    let (home, db, mut input) = fixture();
    input.client.binding = Identity::from_bytes(&[3; 64]).unwrap().public_binding();
    assert!(apply(home.path(), &db, &input).is_err());
    assert!(!home.path().join("secure/host.identity").exists());
    let (home, db, mut input) = fixture();
    input.client.api_key = "operator-key".into();
    assert!(apply(home.path(), &db, &input).is_err());
    assert!(!home.path().join("secure/host.identity").exists());
}

#[test]
fn existing_instance_or_media_refuses_fresh_import() {
    let (home, db, input) = fixture();
    let old = crate::instance::resolve_instance_id(Some(&db), 7680);
    assert!(apply(home.path(), &db, &input).is_err());
    assert_eq!(crate::instance::resolve_instance_id(Some(&db), 7680), old);
    assert!(!home.path().join("secure/host.identity").exists());
    let (home, db, input) = fixture();
    std::fs::create_dir(home.path().join("output")).unwrap();
    std::fs::write(home.path().join("output/preserve.png"), b"existing media").unwrap();
    assert!(apply(home.path(), &db, &input).is_err());
    assert_eq!(
        std::fs::read(home.path().join("output/preserve.png")).unwrap(),
        b"existing media"
    );
    assert!(!home.path().join("secure/host.identity").exists());
}

#[test]
fn parse_errors_are_bounded_and_never_echo_private_input() {
    let secret = "secret-value-must-not-be-in-errors";
    let error = BootstrapInput::parse(secret).err().unwrap().to_string();
    assert!(!error.contains(secret));
    assert!(BootstrapInput::parse(&" ".repeat(16385)).is_err());
    let (_, _, input) = fixture();
    let mut value = serde_json::to_value(&input).unwrap();
    value["unknown"] = serde_json::json!(secret);
    assert!(BootstrapInput::parse(&value.to_string()).is_err());
}
