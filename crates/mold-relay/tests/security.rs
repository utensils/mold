use mold_relay::{validate_endpoint, validate_target};
#[test]
fn remote_plaintext_and_nonloopback_targets_are_rejected() {
    assert!(validate_endpoint("ws://example.com", true).is_err());
    assert!(validate_endpoint("ws://127.0.0.1:1", false).is_err());
    assert!(validate_endpoint("ws://127.0.0.1:1", true).is_ok());
    assert!(validate_endpoint("wss://user:password@example.com", false).is_err());
    assert!(validate_target("192.168.1.2:7680".parse().unwrap()).is_err());
    assert!(validate_target("127.0.0.1:7680".parse().unwrap()).is_ok());
}
#[cfg(unix)]
#[test]
fn token_file_permissions_and_contents_are_checked() {
    use std::os::unix::fs::PermissionsExt;
    let path = std::env::temp_dir().join(format!("mold-relay-token-test-{}", uuid::Uuid::new_v4()));
    std::fs::write(&path, "0123456789012345678901234567890123456789\n").unwrap();
    std::fs::set_permissions(&path, std::fs::Permissions::from_mode(0o644)).unwrap();
    assert!(mold_relay::read_token(&path).is_err());
    std::fs::set_permissions(&path, std::fs::Permissions::from_mode(0o600)).unwrap();
    assert!(mold_relay::read_token(&path).is_ok());
    std::fs::write(&path, "short").unwrap();
    assert!(mold_relay::read_token(&path).is_err());
    std::fs::write(
        &path,
        "0123456789012345678901234567890123456789\nheader-injection",
    )
    .unwrap();
    assert!(mold_relay::read_token(&path).is_err());
    std::fs::remove_file(path).unwrap();
}
