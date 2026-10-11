use mold_secure::*;

fn frames() -> Vec<WireFrame> {
    let mut channel = ReliableChannel::new("s".into(), "r".into());
    for _ in 0..12 {
        channel.queue_packet(&[42; 16320]).unwrap();
    }
    channel.outgoing(0).unwrap()
}
#[test]
fn opaque_object_restores_exact_original_frames_and_rejects_corruption() {
    let frames = frames();
    let object = pack_ciphertext_frames(&frames).unwrap();
    let reference = CiphertextObject {
        id: "11".repeat(16),
        count: frames.len() as u8,
        bytes: object.len() as u32,
        sha256: hex::encode(sha2::Sha256::digest(&object)),
        url: "https://bucket.s3.us-west-2.amazonaws.com/ciphertext/v3/ns/key".into(),
    };
    let envelope = ObjectFrame {
        a: "data".into(),
        v: 3,
        sid: "s".into(),
        rid: "r".into(),
        seq: 0,
        o: reference,
    };
    assert_eq!(
        unpack_ciphertext_frames(&envelope, &object).unwrap(),
        frames
    );
    let mut bad = object.clone();
    bad[8] ^= 1;
    assert!(unpack_ciphertext_frames(&envelope, &bad).is_err());
    let mut malformed = envelope.clone();
    malformed.o.count += 1;
    assert!(unpack_ciphertext_frames(&malformed, &object).is_err());
    let mut shifted = frames.clone();
    shifted[3].seq = Some(99);
    assert!(pack_ciphertext_frames(&shifted).is_err());
}
use sha2::Digest;
