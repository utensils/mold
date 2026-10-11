use mold_secure::{
    namespace, verify_signature, Headers, Identity, Open, Record, Session, SessionConfig,
};
fn pair() -> (Identity, Identity, SessionConfig) {
    let host = Identity::from_bytes(&[1; 64]).unwrap();
    let device = Identity::from_bytes(&[2; 64]).unwrap();
    let config = SessionConfig {
        initiator: true,
        namespace: namespace(&host.signing_public()),
        host_key: hex::encode(host.signing_public()),
        device_key: hex::encode(device.signing_public()),
        host_noise_key: Some(hex::encode(host.noise_public())),
        device_noise_key: Some(hex::encode(device.noise_public())),
        sid: "epoch".into(),
        rid: "device-connection".into(),
    };
    (host, device, config)
}
fn established() -> (Session, Session) {
    let (host, device, config) = pair();
    let mut client = Session::new(&device, config.clone()).unwrap();
    let mut server = Session::new(
        &host,
        SessionConfig {
            initiator: false,
            ..config
        },
    )
    .unwrap();
    server
        .handshake_read(&client.handshake_write().unwrap())
        .unwrap();
    client
        .handshake_read(&server.handshake_write().unwrap())
        .unwrap();
    server
        .handshake_read(&client.handshake_write().unwrap())
        .unwrap();
    assert!(client.established() && server.established());
    (client, server)
}
#[test]
fn pinned_peers_exchange_binary_http_records_without_plaintext() {
    let (mut client, mut server) = established();
    let record = Record::Open {
        stream_id: 1,
        request: Open {
            method: "POST".into(),
            path: "/api/generate?x=1&x=2".into(),
            headers: vec![("x-api-key".into(), "fixture-secret".into())],
            length: None,
        },
    }
    .encode()
    .unwrap();
    let sealed = client.seal(&record).unwrap();
    assert!(!sealed
        .windows(b"fixture-secret".len())
        .any(|w| w == b"fixture-secret"));
    assert_eq!(server.open(&sealed).unwrap(), record);
    let response = Record::Headers {
        stream_id: 1,
        response: Headers {
            status: 206,
            headers: vec![("content-range".into(), "bytes 0-2/4".into())],
        },
    }
    .encode()
    .unwrap();
    assert_eq!(
        client.open(&server.seal(&response).unwrap()).unwrap(),
        response
    );
}
#[test]
fn wrong_pin_wrong_namespace_and_session_substitution_fail_closed() {
    let (host, device, config) = pair();
    let mut wrong = config.clone();
    wrong.namespace = "0".repeat(32);
    assert!(Session::new(&device, wrong).is_err());
    let mut client = Session::new(&device, config.clone()).unwrap();
    let mut other = config.clone();
    other.initiator = false;
    other.sid = "different".into();
    let mut server = Session::new(&host, other).unwrap();
    server
        .handshake_read(&client.handshake_write().unwrap())
        .unwrap();
    assert!(client
        .handshake_read(&server.handshake_write().unwrap())
        .is_err());
    assert!(!client.established());
    assert!(client.handshake_write().is_err());
}
#[test]
fn replay_corruption_and_out_of_order_records_poison_the_session() {
    for attack in ["replay", "corrupt", "skip"] {
        let (mut client, mut server) = established();
        let record = Record::Data {
            stream_id: 1,
            bytes: vec![0, 255, 1],
        }
        .encode()
        .unwrap();
        let mut a = client.seal(&record).unwrap();
        let b = client.seal(&record).unwrap();
        if attack == "replay" {
            server.open(&a).unwrap();
        }
        if attack == "corrupt" {
            a[9] ^= 1;
        }
        assert!(server.open(if attack == "skip" { &b } else { &a }).is_err());
        assert!(server.open(&b).is_err());
    }
}
#[test]
fn record_bounds_preserve_raw_bytes_and_refuse_ambiguous_requests() {
    let bytes = Record::Data {
        stream_id: 1,
        bytes: vec![0, 255],
    }
    .encode()
    .unwrap();
    assert_eq!(bytes, &[3, 0, 0, 0, 1, 0, 0, 0, 2, 0, 255]);
    assert_eq!(Record::decode(&bytes).unwrap().encode().unwrap(), bytes);
    assert!(Record::Data {
        stream_id: 1,
        bytes: vec![0; 16385]
    }
    .encode()
    .is_err());
    assert!(Record::End { stream_id: 0 }.encode().is_err());
    assert!(Record::Credit {
        stream_id: 1,
        bytes: mold_secure::STREAM_WINDOW + 1
    }
    .encode()
    .is_err());
    let mut bad = bytes.clone();
    bad.push(0);
    assert!(Record::decode(&bad).is_err());
    let request = Record::Open {
        stream_id: 1,
        request: Open {
            method: "GET".into(),
            path: "//attacker.invalid/".into(),
            headers: vec![],
            length: Some(0),
        },
    };
    assert!(request.encode().is_err());
}
#[test]
fn signing_binding_and_identity_restoration_are_consistent() {
    let identity = Identity::generate().unwrap();
    let restored = Identity::from_bytes(&identity.to_bytes()).unwrap();
    assert_eq!(identity.binding(), restored.binding());
    let message = b"a fixed interoperability message";
    let signature = identity.sign(message);
    assert!(verify_signature(
        &identity.signing_public(),
        message,
        &signature
    ));
    assert!(!verify_signature(
        &identity.signing_public(),
        b"altered",
        &signature
    ));
    assert_eq!(namespace(&identity.signing_public()).len(), 32);
}

#[test]
fn handshake_hash_binds_invitation_claim_to_exact_authenticated_session() {
    let (client, server) = established();
    assert_eq!(
        client.handshake_hash().unwrap(),
        server.handshake_hash().unwrap()
    );
    let (other, _) = established();
    assert_ne!(
        client.handshake_hash().unwrap(),
        other.handshake_hash().unwrap()
    );
    let (host, device, _) = pair();
    let a = mold_secure::claim_bytes(
        &namespace(&host.signing_public()),
        &"11".repeat(16),
        &client.handshake_hash().unwrap(),
        &device.signing_public(),
        &device.noise_public(),
    )
    .unwrap();
    let b = mold_secure::claim_bytes(
        &namespace(&host.signing_public()),
        &"11".repeat(16),
        &other.handshake_hash().unwrap(),
        &device.signing_public(),
        &device.noise_public(),
    )
    .unwrap();
    assert!(!verify_signature(
        &host.signing_public(),
        &b,
        &host.sign(&a)
    ));
}
