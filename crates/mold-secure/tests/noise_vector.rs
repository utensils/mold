//! Exact upstream Cacophony vector distributed by snow 0.10.0 under its license.
//! Source: https://github.com/mcginty/snow/blob/v0.10.0/tests/vectors/cacophony.txt
use mold_secure::NOISE_SUITE;
#[test]
fn fixed_suite_matches_independent_cacophony_handshake_and_transport_vector() {
    let v: serde_json::Value =
        serde_json::from_str(include_str!("fixtures/noise-xx-cacophony.json")).unwrap();
    assert_eq!(v["protocol_name"], NOISE_SUITE);
    let bytes = |key: &str| hex::decode(v[key].as_str().unwrap()).unwrap();
    let is = bytes("init_static");
    let ie = bytes("init_ephemeral");
    let ip = bytes("init_prologue");
    let rs = bytes("resp_static");
    let re = bytes("resp_ephemeral");
    let rp = bytes("resp_prologue");
    let mut initiator = snow::Builder::new(NOISE_SUITE.parse().unwrap())
        .local_private_key(&is)
        .unwrap()
        .fixed_ephemeral_key_for_testing_only(&ie)
        .prologue(&ip)
        .unwrap()
        .build_initiator()
        .unwrap();
    let mut responder = snow::Builder::new(NOISE_SUITE.parse().unwrap())
        .local_private_key(&rs)
        .unwrap()
        .fixed_ephemeral_key_for_testing_only(&re)
        .prologue(&rp)
        .unwrap()
        .build_responder()
        .unwrap();
    let mut out = [0; 512];
    let mut plain = [0; 512];
    for (index, m) in v["messages"].as_array().unwrap()[..3].iter().enumerate() {
        let payload = hex::decode(m["payload"].as_str().unwrap()).unwrap();
        let (sender, receiver) = if index % 2 == 0 {
            (&mut initiator, &mut responder)
        } else {
            (&mut responder, &mut initiator)
        };
        let n = sender.write_message(&payload, &mut out).unwrap();
        assert_eq!(hex::encode(&out[..n]), m["ciphertext"]);
        let n = receiver.read_message(&out[..n], &mut plain).unwrap();
        assert_eq!(&plain[..n], payload);
    }
    assert_eq!(
        hex::encode(initiator.get_handshake_hash()),
        v["handshake_hash"]
    );
    let mut initiator = initiator.into_transport_mode().unwrap();
    let mut responder = responder.into_transport_mode().unwrap();
    for (index, m) in v["messages"].as_array().unwrap()[3..].iter().enumerate() {
        let payload = hex::decode(m["payload"].as_str().unwrap()).unwrap();
        let (sender, receiver) = if index % 2 == 0 {
            (&mut responder, &mut initiator)
        } else {
            (&mut initiator, &mut responder)
        };
        let n = sender.write_message(&payload, &mut out).unwrap();
        assert_eq!(hex::encode(&out[..n]), m["ciphertext"]);
        let n = receiver.read_message(&out[..n], &mut plain).unwrap();
        assert_eq!(&plain[..n], payload);
    }
}
