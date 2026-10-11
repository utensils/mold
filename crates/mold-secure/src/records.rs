use crate::{Error, Result};
use serde::{Deserialize, Serialize};
pub const MAX_CIPHERTEXT: usize = 65535;
pub const MAX_PLAINTEXT: usize = MAX_CIPHERTEXT - 16;
/// Leaves room for record header, AEAD tag, and packet prefix in one outer frame.
pub const DATA_CHUNK: usize = 16320;
pub const MAX_DATA_CHUNK: usize = 16384;
pub const STREAM_WINDOW: u32 = 256 * 1024;
pub const CREDIT_BATCH: u32 = STREAM_WINDOW / 2;
/// Tracks advertised receive capacity independently from consumed but not yet
/// advertised bytes. Credit is returned in half-window batches, so a progressing
/// stream replenishes before exhausting its window without one frame per read.
#[derive(Debug)]
pub struct ReceiveCredit {
    remaining: u32,
    consumed: u32,
}
impl Default for ReceiveCredit {
    fn default() -> Self {
        Self {
            remaining: STREAM_WINDOW,
            consumed: 0,
        }
    }
}
impl ReceiveCredit {
    pub fn receive(&mut self, bytes: u32) -> Result<()> {
        self.remaining = self.remaining.checked_sub(bytes).ok_or(Error::Input)?;
        Ok(())
    }
    pub fn consume(&mut self, bytes: u32) -> Result<Option<u32>> {
        self.consumed = self
            .consumed
            .checked_add(bytes)
            .filter(|n| *n <= STREAM_WINDOW - self.remaining)
            .ok_or(Error::Input)?;
        if self.consumed < CREDIT_BATCH {
            return Ok(None);
        }
        let credit = std::mem::take(&mut self.consumed);
        self.remaining += credit;
        Ok(Some(credit))
    }
}
pub const MAX_STREAMS: usize = 32;
pub const MAX_UPLOAD: u64 = 64 * 1024 * 1024;
#[cfg(test)]
mod flow_tests {
    use super::*;
    #[test]
    fn sustained_stream_window_accepts_bounded_256k_credit() {
        assert_eq!(STREAM_WINDOW, 256 * 1024);
        assert!(Record::Credit {
            stream_id: 1,
            bytes: 128 * 1024
        }
        .validate()
        .is_ok());
    }
    #[test]
    fn credit_batch_requires_consumption_and_replenishes_before_stall() {
        let mut credit = ReceiveCredit::default();
        for _ in 0..7 {
            credit.receive(16384).unwrap();
            assert_eq!(credit.consume(16384).unwrap(), None);
        }
        credit.receive(16384).unwrap();
        assert_eq!(credit.consume(16384).unwrap(), Some(CREDIT_BATCH));
        assert!(credit.consume(1).is_err());
        let mut paused = ReceiveCredit::default();
        paused.receive(STREAM_WINDOW).unwrap();
        assert!(paused.receive(1).is_err());
        assert_eq!(paused.consume(CREDIT_BATCH).unwrap(), Some(CREDIT_BATCH));
        paused.receive(CREDIT_BATCH).unwrap();
        assert!(paused.receive(1).is_err());
    }
}
pub type HeaderPairs = Vec<(String, String)>;
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct Open {
    pub method: String,
    pub path: String,
    pub headers: HeaderPairs,
    pub length: Option<u64>,
}
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct Headers {
    pub status: u16,
    pub headers: HeaderPairs,
}
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Record {
    Open { stream_id: u32, request: Open },
    Headers { stream_id: u32, response: Headers },
    Data { stream_id: u32, bytes: Vec<u8> },
    End { stream_id: u32 },
    Cancel { stream_id: u32, code: String },
    Credit { stream_id: u32, bytes: u32 },
}
fn headers_valid(headers: &HeaderPairs) -> bool {
    headers.len() <= 128
        && headers
            .iter()
            .map(|(k, v)| k.len() + v.len())
            .sum::<usize>()
            <= 49152
        && headers.iter().all(|(k, v)| {
            !k.is_empty()
                && k.bytes()
                    .all(|b| b.is_ascii_alphanumeric() || b"!#$%&'*+-.^_`|~".contains(&b))
                && !v.bytes().any(|b| b == b'\r' || b == b'\n' || b == 0)
        })
}
impl Record {
    pub fn stream_id(&self) -> u32 {
        match self {
            Self::Open { stream_id, .. }
            | Self::Headers { stream_id, .. }
            | Self::Data { stream_id, .. }
            | Self::End { stream_id }
            | Self::Cancel { stream_id, .. }
            | Self::Credit { stream_id, .. } => *stream_id,
        }
    }
    pub fn kind(&self) -> u8 {
        match self {
            Self::Open { .. } => 1,
            Self::Headers { .. } => 2,
            Self::Data { .. } => 3,
            Self::End { .. } => 4,
            Self::Cancel { .. } => 5,
            Self::Credit { .. } => 6,
        }
    }
    pub fn payload(&self) -> Result<Vec<u8>> {
        self.validate()?;
        match self {
            Self::Open { request, .. } => serde_json::to_vec(request).map_err(|_| Error::Input),
            Self::Headers { response, .. } => {
                serde_json::to_vec(response).map_err(|_| Error::Input)
            }
            Self::Data { bytes, .. } => Ok(bytes.clone()),
            Self::End { .. } => Ok(vec![]),
            Self::Cancel { code, .. } => Ok(code.as_bytes().to_vec()),
            Self::Credit { bytes, .. } => Ok(bytes.to_be_bytes().to_vec()),
        }
    }
    pub fn validate(&self) -> Result<()> {
        if self.stream_id() == 0 {
            return Err(Error::Input);
        }
        let valid = match self {
            Self::Open { stream_id, request } => {
                stream_id % 2 == 1
                    && !request.method.is_empty()
                    && request.method.len() <= 16
                    && request.method.bytes().all(|b| b.is_ascii_uppercase())
                    && request.method != "CONNECT"
                    && request.path.starts_with('/')
                    && !request.path.starts_with("//")
                    && request.path.len() <= 8192
                    && !request.path.bytes().any(|b| b <= 32 || b == 127)
                    && request.length.is_none_or(|n| n <= MAX_UPLOAD)
                    && headers_valid(&request.headers)
            }
            Self::Headers { response, .. } => {
                (100..=599).contains(&response.status) && headers_valid(&response.headers)
            }
            Self::Data { bytes, .. } => !bytes.is_empty() && bytes.len() <= MAX_DATA_CHUNK,
            Self::End { .. } => true,
            Self::Cancel { code, .. } => {
                !code.is_empty()
                    && code.len() <= 64
                    && code
                        .bytes()
                        .all(|b| b.is_ascii_alphanumeric() || b == b'-' || b == b'_')
            }
            Self::Credit { bytes, .. } => (1..=STREAM_WINDOW).contains(bytes),
        };
        if !valid {
            return Err(Error::Input);
        }
        Ok(())
    }
    pub fn encode(&self) -> Result<Vec<u8>> {
        let payload = self.payload()?;
        if payload.len() + 9 > MAX_PLAINTEXT {
            return Err(Error::Limit);
        }
        let mut bytes = Vec::with_capacity(payload.len() + 9);
        bytes.push(self.kind());
        bytes.extend_from_slice(&self.stream_id().to_be_bytes());
        bytes.extend_from_slice(&(payload.len() as u32).to_be_bytes());
        bytes.extend_from_slice(&payload);
        Ok(bytes)
    }
    pub fn decode(bytes: &[u8]) -> Result<Self> {
        if bytes.len() < 9 || bytes.len() > MAX_PLAINTEXT {
            return Err(Error::Input);
        }
        let stream_id = u32::from_be_bytes(bytes[1..5].try_into().map_err(|_| Error::Input)?);
        let n = u32::from_be_bytes(bytes[5..9].try_into().map_err(|_| Error::Input)?) as usize;
        if bytes.len() != n + 9 {
            return Err(Error::Input);
        }
        let payload = &bytes[9..];
        let value = match bytes[0] {
            1 => Self::Open {
                stream_id,
                request: serde_json::from_slice(payload).map_err(|_| Error::Input)?,
            },
            2 => Self::Headers {
                stream_id,
                response: serde_json::from_slice(payload).map_err(|_| Error::Input)?,
            },
            3 => Self::Data {
                stream_id,
                bytes: payload.to_vec(),
            },
            4 if n == 0 => Self::End { stream_id },
            5 => Self::Cancel {
                stream_id,
                code: std::str::from_utf8(payload)
                    .map_err(|_| Error::Input)?
                    .into(),
            },
            6 if n == 4 => Self::Credit {
                stream_id,
                bytes: u32::from_be_bytes(payload.try_into().map_err(|_| Error::Input)?),
            },
            _ => return Err(Error::Input),
        };
        value.validate()?;
        Ok(value)
    }
}
