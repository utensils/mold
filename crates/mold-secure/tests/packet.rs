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
