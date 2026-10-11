use mold_secure::{decode_hex, Identity};
use mold_secure_client::{prepare_bootstrap, prepare_bootstrap_for_binding};

#[test]
fn bootstrap_preserves_installation_binding_without_exporting_device_private_key() {
    let device = Identity::from_bytes(&[47; 64]).unwrap();
    let prepared =
        prepare_bootstrap_for_binding("Desktop", "desktop", &device.public_binding()).unwrap();
    let profile = prepared
        .public_profile()
        .bind_origin("https://pod-7680.proxy.runpod.net")
        .unwrap();
    assert_eq!(
        profile.websocket,
        "wss://pod-7680.proxy.runpod.net/_mold/secure"
    );
    assert_eq!(
        profile.device_binding.signing_key,
        device.public_binding().signing_key
    );
    let json = serde_json::to_value(&profile).unwrap();
    assert!(json.get("device_identity").is_none());
    assert!(json.get("host_identity").is_none());
    let environment = prepared.environment();
    let bootstrap: serde_json::Value =
        serde_json::from_str(&environment["MOLD_SECURE_BOOTSTRAP"]).unwrap();
    assert_eq!(bootstrap["instance_id"], profile.instance_id);
    assert_eq!(bootstrap["id"], prepared.id());
    let host = Identity::from_bytes(
        &decode_hex::<64>(bootstrap["host_identity"].as_str().unwrap()).unwrap(),
    )
    .unwrap();
    profile
        .grant
        .verify(
            &host.signing_public(),
            &device.public_binding(),
            mold_secure_client::now(),
            0,
        )
        .unwrap();
    assert_eq!(profile.host_key, hex::encode(host.signing_public()));
    assert!(prepared
        .profile("https://pod-7680.proxy.runpod.net")
        .is_err());
}

#[test]
fn cli_bootstrap_retains_only_client_authority_in_paired_profile() {
    let prepared = prepare_bootstrap("CLI", "cli").unwrap();
    let profile = prepared.profile("http://127.0.0.1:7680").unwrap();
    profile.validate().unwrap();
    assert_eq!(profile.websocket, "ws://127.0.0.1:7680/_mold/secure");
    let other = prepare_bootstrap("CLI", "cli").unwrap();
    assert_ne!(prepared.id(), other.id());
    assert_ne!(profile.host_key, other.public_profile().host_key);
}

#[test]
fn bootstrap_rejects_forged_binding_and_remote_cleartext_origin() {
    let mut binding = Identity::from_bytes(&[48; 64]).unwrap().public_binding();
    binding.signature = "00".repeat(64);
    assert!(prepare_bootstrap_for_binding("Desktop", "desktop", &binding).is_err());
    let prepared = prepare_bootstrap("CLI", "cli").unwrap();
    for origin in [
        "http://untrusted.example",
        "https://user:secret@example.com",
        "https://example.com/path",
        "https://example.com?token=x",
    ] {
        assert!(
            prepared.public_profile().bind_origin(origin).is_err(),
            "{origin}"
        );
    }
}
