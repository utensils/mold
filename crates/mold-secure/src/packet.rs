//! Bounded reliable ciphertext delivery. Retries reuse exact ciphertext bytes.
use crate::*;
use base64::{engine::general_purpose::STANDARD, Engine};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, VecDeque};
pub const WIRE_WINDOW: u64 = 16;
const WINDOW: u64 = WIRE_WINDOW;
const CHUNK: usize = 16384;
const RETRY: u64 = 10_000;
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct WireFrame {
    pub a: String,
    pub v: u8,
    pub sid: String,
    pub rid: String,
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub seq: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub d: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub next: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub credit: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub ack_seq: Option<u64>,
}
struct Pending {
    frame: WireFrame,
    last: u64,
    first: u64,
}
pub struct ReliableChannel {
    sid: String,
    rid: String,
    send_next: u64,
    send_acked: u64,
    peer_credit: u64,
    peer_ack: u64,
    seen_ack: bool,
    pending: BTreeMap<u64, Pending>,
    queued: VecDeque<Vec<u8>>,
    recv_next: u64,
    held: BTreeMap<u64, Vec<u8>>,
    history: VecDeque<(u64, [u8; 32])>,
    packet: Vec<u8>,
    ack_dirty: bool,
    ack_serial: u64,
    gap_since: Option<u64>,
}
impl ReliableChannel {
    pub fn new(sid: String, rid: String) -> Self {
        Self {
            sid,
            rid,
            send_next: 0,
            send_acked: 0,
            peer_credit: WINDOW,
            peer_ack: 0,
            seen_ack: false,
            pending: BTreeMap::new(),
            queued: VecDeque::new(),
            recv_next: 0,
            held: BTreeMap::new(),
            history: VecDeque::new(),
            packet: vec![],
            ack_dirty: false,
            ack_serial: 0,
            gap_since: None,
        }
    }
    fn frame(&self, a: &str) -> WireFrame {
        WireFrame {
            a: a.into(),
            v: 3,
            sid: self.sid.clone(),
            rid: self.rid.clone(),
            seq: None,
            d: None,
            next: None,
            credit: None,
            ack_seq: None,
        }
    }
    pub fn is_idle(&self) -> bool {
        self.pending.is_empty() && self.queued.is_empty()
    }
    pub fn can_queue(&self) -> bool {
        self.queued.len() + self.pending.len() < WINDOW as usize * 2
    }
    pub fn queue_packet(&mut self, packet: &[u8]) -> Result<()> {
        if packet.is_empty() || packet.len() > MAX_CIPHERTEXT {
            return Err(Error::Limit);
        }
        let mut framed = Vec::with_capacity(packet.len() + 4);
        framed.extend_from_slice(&(packet.len() as u32).to_be_bytes());
        framed.extend_from_slice(packet);
        if self.queued.len() + self.pending.len() + framed.len().div_ceil(CHUNK)
            > WINDOW as usize * 2 + 5
        {
            return Err(Error::Limit);
        }
        self.queued.extend(framed.chunks(CHUNK).map(<[u8]>::to_vec));
        Ok(())
    }
    pub fn outgoing(&mut self, now: u64) -> Result<Vec<WireFrame>> {
        if self
            .gap_since
            .is_some_and(|t| now.saturating_sub(t) > 30_000)
            || self
                .pending
                .values()
                .any(|p| now.saturating_sub(p.first) > 30_000)
        {
            return Err(Error::State);
        }
        let mut frames = vec![];
        if self.ack_dirty {
            let mut ack = self.frame("ack");
            ack.next = Some(self.recv_next);
            ack.credit = Some(WINDOW);
            ack.ack_seq = Some(self.ack_serial);
            self.ack_serial += 1;
            frames.push(ack);
            self.ack_dirty = false;
        }
        for p in self.pending.values_mut() {
            if now.saturating_sub(p.last) >= RETRY {
                p.last = now;
                frames.push(p.frame.clone());
            }
        }
        while self.send_next < self.send_acked + self.peer_credit
            && self.pending.len() < WINDOW as usize
        {
            let Some(bytes) = self.queued.pop_front() else {
                break;
            };
            if self.send_next > u32::MAX as u64 {
                return Err(Error::Limit);
            }
            let mut frame = self.frame("data");
            frame.seq = Some(self.send_next);
            frame.d = Some(STANDARD.encode(bytes));
            self.pending.insert(
                self.send_next,
                Pending {
                    frame: frame.clone(),
                    last: now,
                    first: now,
                },
            );
            self.send_next += 1;
            frames.push(frame);
        }
        Ok(frames)
    }
    pub fn receive(&mut self, frame: WireFrame, now: u64) -> Result<Vec<Vec<u8>>> {
        if frame.v != 3 || frame.sid != self.sid || frame.rid != self.rid {
            return Err(Error::Authentication);
        }
        if frame.a == "ack" {
            let (Some(next), Some(credit), Some(serial)) =
                (frame.next, frame.credit, frame.ack_seq)
            else {
                return Err(Error::Input);
            };
            if frame.seq.is_some() || frame.d.is_some() || next > self.send_next || credit > WINDOW
            {
                return Err(Error::Input);
            }
            if self.seen_ack && serial <= self.peer_ack {
                return Ok(vec![]);
            }
            if next < self.send_acked {
                return Err(Error::Input);
            }
            self.seen_ack = true;
            self.peer_ack = serial;
            self.send_acked = next;
            self.peer_credit = credit;
            self.pending.retain(|n, _| *n >= next);
            return Ok(vec![]);
        }
        if frame.a != "data"
            || frame.next.is_some()
            || frame.credit.is_some()
            || frame.ack_seq.is_some()
        {
            return Err(Error::Input);
        }
        let seq = frame.seq.ok_or(Error::Input)?;
        let encoded = frame.d.ok_or(Error::Input)?;
        if encoded.len() > CHUNK.div_ceil(3) * 4 {
            return Err(Error::Limit);
        }
        let bytes = STANDARD.decode(&encoded).map_err(|_| Error::Input)?;
        if bytes.is_empty() || bytes.len() > CHUNK || STANDARD.encode(&bytes) != encoded {
            return Err(Error::Input);
        }
        let hash: [u8; 32] = Sha256::digest(&bytes).into();
        self.ack_dirty = true;
        if seq < self.recv_next {
            // Old retries are already consumed and must never enter Noise again.
            // Retained hashes detect a recent conflict; aged-out duplicates are
            // harmless ACK-only traffic, including after a delayed cloud retry.
            if self.history.iter().any(|(n, h)| *n == seq && *h != hash) {
                return Err(Error::Authentication);
            }
            return Ok(vec![]);
        }
        if seq >= self.recv_next + WINDOW {
            return Err(Error::Limit);
        }
        if let Some(old) = self.held.get(&seq) {
            if old != &bytes {
                return Err(Error::Authentication);
            }
            return Ok(vec![]);
        }
        self.held.insert(seq, bytes);
        let mut packets = vec![];
        while let Some(bytes) = self.held.remove(&self.recv_next) {
            self.history
                .push_back((self.recv_next, Sha256::digest(&bytes).into()));
            if self.history.len() > 32 {
                self.history.pop_front();
            }
            self.recv_next += 1;
            self.packet.extend(bytes);
            loop {
                if self.packet.len() < 4 {
                    break;
                }
                let n = u32::from_be_bytes(self.packet[..4].try_into().map_err(|_| Error::Input)?)
                    as usize;
                if n == 0 || n > MAX_CIPHERTEXT {
                    return Err(Error::Limit);
                }
                if self.packet.len() < n + 4 {
                    break;
                }
                packets.push(self.packet[4..n + 4].to_vec());
                self.packet.drain(..n + 4);
            }
            if self.packet.len() > MAX_CIPHERTEXT + 4 {
                return Err(Error::Limit);
            }
        }
        if self.held.is_empty() {
            self.gap_since = None
        } else {
            self.gap_since.get_or_insert(now);
        }
        Ok(packets)
    }
}
