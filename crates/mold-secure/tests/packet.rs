use mold_secure::*;
#[test]
fn packet_layer_reassembles_reordered_duplicate_frames_and_bounds_pressure() {
    let mut a = ReliableChannel::new("s".into(), "r".into());
    let mut b = ReliableChannel::new("s".into(), "r".into());
    let payload = vec![42; 65535];
    a.queue_packet(&payload).unwrap();
    let mut frames = a.outgoing(0).unwrap();
    assert_eq!(frames.len(), 5);
    assert!(b.receive(frames.remove(1), 0).unwrap().is_empty());
    let first = frames.remove(0);
    assert!(b.receive(first.clone(), 1).unwrap().is_empty());
    assert!(b.receive(first, 2).unwrap().is_empty());
    let mut packets = vec![];
    for f in frames {
        packets.extend(b.receive(f, 3).unwrap());
    }
    assert_eq!(packets, vec![payload]);
    for ack in b.outgoing(4).unwrap() {
        a.receive(ack, 4).unwrap();
    }
    assert!(a.is_idle());
}
#[test]
fn retries_do_not_reencrypt_and_conflicting_duplicates_fail() {
    let mut a = ReliableChannel::new("s".into(), "r".into());
    let mut b = ReliableChannel::new("s".into(), "r".into());
    a.queue_packet(&[1, 2, 3]).unwrap();
    let sent = a.outgoing(0).unwrap();
    assert_eq!(a.outgoing(10_000).unwrap(), sent);
    assert_eq!(b.receive(sent[0].clone(), 0).unwrap(), vec![vec![1, 2, 3]]);
    let mut bad = sent[0].clone();
    bad.d = Some("AAAA".into());
    assert!(b.receive(bad, 1).is_err());
}
#[test]
fn delayed_duplicates_older_than_history_do_not_reenter_noise_or_close_session() {
    let mut a = ReliableChannel::new("s".into(), "r".into());
    let mut b = ReliableChannel::new("s".into(), "r".into());
    let mut first = None;
    for n in 0..64 {
        a.queue_packet(&[n]).unwrap();
        let data = a
            .outgoing(n as u64)
            .unwrap()
            .into_iter()
            .find(|f| f.a == "data")
            .unwrap();
        if first.is_none() {
            first = Some(data.clone())
        }
        assert_eq!(b.receive(data, n as u64).unwrap(), vec![vec![n]]);
        for ack in b.outgoing(n as u64).unwrap() {
            a.receive(ack, n as u64).unwrap();
        }
    }
    let mut old = first.unwrap();
    old.d = Some("AAAA".into());
    assert!(b.receive(old, 100).unwrap().is_empty());
    assert!(b
        .outgoing(100)
        .unwrap()
        .iter()
        .any(|f| f.a == "ack" && f.next == Some(64)));
}
#[test]
fn object_retry_deadline_starts_only_when_ciphertext_is_actually_published() {
    let mut wire = ReliableChannel::new("s".into(), "r".into());
    wire.defer_transmission();
    wire.queue_packet(&[1, 2, 3]).unwrap();
    let frames = wire.outgoing(0).unwrap();
    assert!(wire.outgoing(60_000).unwrap().is_empty());
    wire.mark_transmitted(0, 1, 60_000);
    assert!(wire.outgoing(69_999).unwrap().is_empty());
    assert_eq!(wire.outgoing(70_000).unwrap(), frames);
    assert!(wire.outgoing(149_999).is_ok());
    assert!(wire.outgoing(150_001).is_err());
}
#[test]
fn adjacent_ciphertext_packets_share_bounded_frames_without_changing_packet_boundaries() {
    let mut sender = ReliableChannel::new("s".into(), "r".into());
    let mut receiver = ReliableChannel::new("s".into(), "r".into());
    for n in 0..16 {
        sender.queue_packet(&vec![n; 512]).unwrap();
    }
    let frames = sender.outgoing(0).unwrap();
    assert_eq!(frames.len(), 1);
    let packets = receiver.receive(frames[0].clone(), 0).unwrap();
    assert_eq!(packets, (0..16).map(|n| vec![n; 512]).collect::<Vec<_>>());
}
#[test]
fn four_ciphertext_objects_can_be_in_flight_with_a_bounded_sixty_four_frame_window() {
    let mut sender = ReliableChannel::new("s".into(), "r".into());
    for _ in 0..65 {
        sender.queue_packet(&vec![42; 16380]).unwrap();
    }
    assert_eq!(sender.outgoing(0).unwrap().len(), 64);
    assert!(sender.outgoing(1).unwrap().is_empty());
    let mut receiver = ReliableChannel::new("s".into(), "r".into());
    let excessive = WireFrame {
        a: "data".into(),
        v: 3,
        sid: "s".into(),
        rid: "r".into(),
        seq: Some(64),
        d: Some("AQ==".into()),
        next: None,
        credit: None,
        ack_seq: None,
    };
    assert!(receiver.receive(excessive, 0).is_err());
}
