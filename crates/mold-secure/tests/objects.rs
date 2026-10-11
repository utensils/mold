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

#[test]
fn large_object_includes_frame_prefixes_inside_unchanged_one_mib_budget() {
    use base64::{engine::general_purpose::STANDARD, Engine};
    let mut channel = ReliableChannel::new("s".into(), "r".into());
    for _ in 0..64 {
        channel.queue_packet(&[42; 16380]).unwrap();
    }
    let mut frames = channel.outgoing(0).unwrap();
    assert_eq!(frames.len(), 64);
    assert_eq!(
        pack_ciphertext_frames(&frames[..63]).unwrap().len(),
        63 * 16388
    );
    assert!(pack_ciphertext_frames(&frames).is_err());
    frames[63].d = Some(STANDARD.encode([42; 128]));
    assert!(pack_ciphertext_frames(&frames).unwrap().len() <= 1048576);
    let mut extra = frames[63].clone();
    extra.seq = Some(64);
    frames.push(extra);
    assert!(pack_ciphertext_frames(&frames).is_err());
}
