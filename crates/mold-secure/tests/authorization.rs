use mold_secure::*;
#[test]
fn attestation_binds_peer_certificate_and_exact_request_target() {
    let host = Identity::from_bytes(&[1; 64]).unwrap();
    let device = Identity::from_bytes(&[2; 64]).unwrap();
    let mut grant = DeviceGrant {
        id: "11".repeat(16),
        expires_at: 2000,
        epoch: 3,
        signature: String::new(),
    };
    grant.signature = hex::encode(
        host.sign(
            &grant_bytes(
                &namespace(&host.signing_public()),
                &device.signing_public(),
                &device.noise_public(),
                &grant.id,
                grant.expires_at,
                grant.epoch,
            )
            .unwrap(),
        ),
    );
    let mut context = RequestContext {
        binding: device.public_binding(),
        grant: Some(grant),
        invite: None,
        handshake_hash: "22".repeat(32),
        stream_id: 1,
        issued_at: 1000,
        request_nonce: "33".repeat(16),
        signature: String::new(),
    };
    context.sign(&host, "GET", "/api/gallery?x=1").unwrap();
    context
        .verify(&host.signing_public(), "GET", "/api/gallery?x=1", 1001)
        .unwrap();
    assert!(context
        .verify(&host.signing_public(), "POST", "/api/gallery?x=1", 1001)
        .is_err());
    assert!(context
        .verify(&host.signing_public(), "GET", "/api/gallery?x=2", 1001)
        .is_err());
    assert!(context
        .verify(&host.signing_public(), "GET", "/api/gallery?x=1", 1031)
        .is_err());
    context.grant.as_mut().unwrap().epoch = 4;
    assert!(context
        .verify(&host.signing_public(), "GET", "/api/gallery?x=1", 1001)
        .is_err());
}
#[test]
fn cloud_canonical_vectors_match_exact_bytes_and_signatures() {
    let fixture: serde_json::Value = serde_json::from_str(include_str!(
        "../../../relay/aws/test/fixtures/admission-v3.json"
    ))
    .unwrap();
    let host = decode_hex::<32>(fixture["hostKey"].as_str().unwrap()).unwrap();
    assert_eq!(namespace(&host), fixture["namespace"]);
    let ns = fixture["namespace"].as_str().unwrap();
    for kind in ["challenge", "grant", "invite"] {
        let v = &fixture[kind]["value"];
        let key = decode_hex::<32>(v["signingKey"].as_str().unwrap()).unwrap();
        let bytes = match kind {
            "challenge" => join_bytes(
                ns,
                2,
                v["connectionId"].as_str().unwrap(),
                v["nonce"].as_str().unwrap(),
                &key,
                v["expiresAt"].as_u64().unwrap(),
            ),
            "grant" => grant_bytes(
                ns,
                &key,
                &decode_hex(v["noiseKey"].as_str().unwrap()).unwrap(),
                v["id"].as_str().unwrap(),
                v["expiresAt"].as_u64().unwrap(),
                v["epoch"].as_u64().unwrap(),
            ),
            _ => invite_bytes(
                ns,
                v["id"].as_str().unwrap(),
                &key,
                v["expiresAt"].as_u64().unwrap(),
                v["epoch"].as_u64().unwrap(),
            ),
        }
        .unwrap();
        assert_eq!(hex::encode(&bytes), fixture[kind]["bytes"]);
        assert!(verify_signature(
            if kind == "challenge" { &key } else { &host },
            &bytes,
            &decode_hex::<64>(fixture[kind]["signature"].as_str().unwrap()).unwrap()
        ));
    }
}
