//! Storage framing for already-encrypted reliable frames. This layer has no
//! application records, plaintext, keys, or additional cryptographic schedule.
use crate::{decode_hex, Error, Result, WireFrame};
use base64::{engine::general_purpose::STANDARD, Engine};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

pub const MAX_OBJECT_BYTES: usize = 262144;
pub const MAX_OBJECT_FRAMES: usize = 16;
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct CiphertextObject {
    pub id: String,
    pub bytes: u32,
    pub sha256: String,
    pub url: String,
    pub count: u8,
}
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct ObjectFrame {
    pub a: String,
    pub v: u8,
    pub sid: String,
    pub rid: String,
    pub seq: u64,
    pub o: CiphertextObject,
}
impl ObjectFrame {
    pub fn validate(&self) -> Result<()> {
        if self.a != "data"
            || self.v != 3
            || self.sid.is_empty()
            || self.sid.len() > 256
            || self.rid.is_empty()
            || self.rid.len() > 256
            || !(1..=MAX_OBJECT_FRAMES as u8).contains(&self.o.count)
            || self.o.bytes < u32::from(self.o.count) * 5
            || self.o.bytes as usize > MAX_OBJECT_BYTES
            || self
                .seq
                .checked_add(u64::from(self.o.count))
                .is_none_or(|n| n > u64::from(u32::MAX) + 1)
            || self.o.url.len() > 4096
        {
            return Err(Error::Limit);
        }
        decode_hex::<16>(&self.o.id)?;
        decode_hex::<32>(&self.o.sha256)?;
        Ok(())
    }
}
pub fn pack_ciphertext_frames(frames: &[WireFrame]) -> Result<Vec<u8>> {
    let first = frames.first().ok_or(Error::Input)?;
    let start = first.seq.ok_or(Error::Input)?;
    if frames.len() > MAX_OBJECT_FRAMES || start > u64::from(u32::MAX) {
        return Err(Error::Limit);
    }
    let mut object = Vec::new();
    for (index, frame) in frames.iter().enumerate() {
        if frame.a != "data"
            || frame.v != 3
            || frame.sid != first.sid
            || frame.rid != first.rid
            || frame.seq != Some(start + index as u64)
            || frame.next.is_some()
            || frame.credit.is_some()
            || frame.ack_seq.is_some()
        {
            return Err(Error::Input);
        }
        let encoded = frame.d.as_deref().ok_or(Error::Input)?;
        if encoded.len() > 21848 {
            return Err(Error::Limit);
        }
        let raw = STANDARD.decode(encoded).map_err(|_| Error::Input)?;
        if raw.is_empty() || raw.len() > 16384 || STANDARD.encode(&raw) != encoded {
            return Err(Error::Input);
        }
        if object.len() + 4 + raw.len() > MAX_OBJECT_BYTES {
            return Err(Error::Limit);
        }
        object.extend_from_slice(&(raw.len() as u32).to_be_bytes());
        object.extend_from_slice(&raw);
    }
    Ok(object)
}
pub fn unpack_ciphertext_frames(envelope: &ObjectFrame, object: &[u8]) -> Result<Vec<WireFrame>> {
    envelope.validate()?;
    if object.len() != envelope.o.bytes as usize
        || <[u8; 32]>::from(Sha256::digest(object)) != decode_hex::<32>(&envelope.o.sha256)?
    {
        return Err(Error::Authentication);
    }
    let mut offset = 0usize;
    let mut frames = Vec::new();
    for index in 0..envelope.o.count {
        let prefix = object.get(offset..offset + 4).ok_or(Error::Input)?;
        let length = u32::from_be_bytes(prefix.try_into().map_err(|_| Error::Input)?) as usize;
        if length == 0 || length > 16384 {
            return Err(Error::Limit);
        }
        offset += 4;
        let raw = object.get(offset..offset + length).ok_or(Error::Input)?;
        offset += length;
        frames.push(WireFrame {
            a: "data".into(),
            v: 3,
            sid: envelope.sid.clone(),
            rid: envelope.rid.clone(),
            seq: Some(envelope.seq + u64::from(index)),
            d: Some(STANDARD.encode(raw)),
            next: None,
            credit: None,
            ack_seq: None,
        });
    }
    // Nothing is returned until the complete object is validated. A trailing
    // byte or missing chunk cannot partially advance a Noise session.
    if offset != object.len() {
        return Err(Error::Input);
    }
    Ok(frames)
}
