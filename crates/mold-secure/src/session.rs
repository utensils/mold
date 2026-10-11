use crate::*;
use serde::{Deserialize, Serialize};
use zeroize::Zeroizing;
pub const NOISE_SUITE: &str = "Noise_XX_25519_ChaChaPoly_SHA256";
const REKEY_RECORDS: u64 = 1 << 20;
#[derive(Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SessionConfig {
    pub initiator: bool,
    pub namespace: String,
    pub host_key: String,
    pub device_key: String,
    #[serde(default)]
    pub host_noise_key: Option<String>,
    #[serde(default)]
    pub device_noise_key: Option<String>,
    pub sid: String,
    pub rid: String,
}
impl SessionConfig {
    pub fn prologue(&self) -> Result<Vec<u8>> {
        let host = decode_hex::<32>(&self.host_key)?;
        let device = decode_hex::<32>(&self.device_key)?;
        if namespace(&host) != self.namespace
            || self.sid.is_empty()
            || self.rid.is_empty()
            || self.sid.len() > 128
            || self.rid.len() > 128
            || !self
                .sid
                .bytes()
                .chain(self.rid.bytes())
                .all(|b| b.is_ascii_graphic())
        {
            return Err(Error::Input);
        }
        let mut bytes = b"MOLD3-NOISE\0".to_vec();
        bytes.extend_from_slice(&decode_hex::<16>(&self.namespace)?);
        bytes.extend_from_slice(&host);
        bytes.extend_from_slice(&device);
        for id in [&self.sid, &self.rid] {
            bytes.extend_from_slice(&(id.len() as u16).to_be_bytes());
            bytes.extend_from_slice(id.as_bytes());
        }
        Ok(bytes)
    }
}
enum State {
    Handshake(Box<snow::HandshakeState>),
    Transport(Box<snow::TransportState>),
    Closed,
}
pub struct Session {
    state: State,
    binding: [u8; 128],
    peer: [u8; 32],
    peer_noise: Option<[u8; 32]>,
    initiator: bool,
    first_sent: bool,
    peer_verified: bool,
    sent: u64,
    received: u64,
    handshake_hash: Option<[u8; 32]>,
}
impl Session {
    pub fn new(identity: &Identity, config: SessionConfig) -> Result<Self> {
        let local = if config.initiator {
            &config.device_key
        } else {
            &config.host_key
        };
        if decode_hex::<32>(local)? != identity.signing_public() {
            return Err(Error::Authentication);
        }
        let peer = decode_hex(if config.initiator {
            &config.host_key
        } else {
            &config.device_key
        })?;
        let peer_noise = (if config.initiator {
            config.host_noise_key.as_deref()
        } else {
            config.device_noise_key.as_deref()
        })
        .map(decode_hex)
        .transpose()?;
        let prologue = config.prologue()?;
        let private = Zeroizing::new(identity.noise_secret());
        let builder = snow::Builder::new(NOISE_SUITE.parse().map_err(|_| Error::Input)?)
            .local_private_key(private.as_ref())
            .map_err(|_| Error::Input)?
            .prologue(&prologue)
            .map_err(|_| Error::Input)?;
        let state = if config.initiator {
            builder.build_initiator()
        } else {
            builder.build_responder()
        }
        .map_err(|_| Error::State)?;
        Ok(Self {
            state: State::Handshake(Box::new(state)),
            binding: identity.binding(),
            peer,
            peer_noise,
            initiator: config.initiator,
            first_sent: false,
            peer_verified: false,
            sent: 0,
            received: 0,
            handshake_hash: None,
        })
    }
    pub fn established(&self) -> bool {
        matches!(self.state, State::Transport(_))
    }
    pub fn close(&mut self) {
        self.state = State::Closed;
        self.handshake_hash = None;
    }
    pub fn handshake_hash(&self) -> Result<[u8; 32]> {
        if !self.established() {
            return Err(Error::State);
        }
        self.handshake_hash.ok_or(Error::State)
    }
    fn promote(&mut self) -> Result<()> {
        let done = matches!(&self.state,State::Handshake(h) if h.is_handshake_finished());
        if done {
            if !self.peer_verified {
                self.close();
                return Err(Error::Authentication);
            }
            let State::Handshake(h) = std::mem::replace(&mut self.state, State::Closed) else {
                return Err(Error::State);
            };
            self.handshake_hash = Some(
                h.get_handshake_hash()
                    .try_into()
                    .map_err(|_| Error::State)?,
            );
            self.state =
                State::Transport(Box::new(h.into_transport_mode().map_err(|_| Error::State)?));
        }
        Ok(())
    }
    pub fn handshake_write(&mut self) -> Result<Vec<u8>> {
        let State::Handshake(h) = &mut self.state else {
            return Err(Error::State);
        };
        let payload = if self.initiator && !self.first_sent {
            &[][..]
        } else {
            &self.binding
        };
        let mut output = vec![0; 512];
        let written = h.write_message(payload, &mut output);
        let Ok(n) = written else {
            self.close();
            return Err(Error::Authentication);
        };
        self.first_sent = true;
        output.truncate(n);
        self.promote()?;
        Ok(output)
    }
    pub fn handshake_read(&mut self, message: &[u8]) -> Result<()> {
        if message.len() > 512 {
            self.close();
            return Err(Error::Limit);
        }
        let State::Handshake(h) = &mut self.state else {
            return Err(Error::State);
        };
        let mut payload = [0; 256];
        let read = h.read_message(message, &mut payload);
        let Ok(n) = read else {
            self.close();
            return Err(Error::Authentication);
        };
        if let Some(remote) = h.get_remote_static() {
            if n != 128
                || Binding::verify_bytes(&payload[..n]).is_err()
                || payload[..32] != self.peer
                || payload[32..64] != *remote
                || self.peer_noise.is_some_and(|key| key != payload[32..64])
            {
                self.close();
                return Err(Error::Authentication);
            }
            self.peer_verified = true;
        } else if n != 0 {
            self.close();
            return Err(Error::Authentication);
        }
        self.promote()
    }
    pub fn seal(&mut self, plaintext: &[u8]) -> Result<Vec<u8>> {
        Record::decode(plaintext)?;
        let State::Transport(t) = &mut self.state else {
            return Err(Error::State);
        };
        let mut output = vec![0; plaintext.len() + 16];
        let Ok(n) = t.write_message(plaintext, &mut output) else {
            self.close();
            return Err(Error::Authentication);
        };
        self.sent += 1;
        if self.sent.is_multiple_of(REKEY_RECORDS) {
            t.rekey_outgoing();
        }
        output.truncate(n);
        Ok(output)
    }
    pub fn open(&mut self, ciphertext: &[u8]) -> Result<Vec<u8>> {
        if ciphertext.len() < 16 || ciphertext.len() > MAX_CIPHERTEXT {
            self.close();
            return Err(Error::Limit);
        }
        let State::Transport(t) = &mut self.state else {
            return Err(Error::State);
        };
        let mut output = vec![0; ciphertext.len() - 16];
        let Ok(n) = t.read_message(ciphertext, &mut output) else {
            self.close();
            return Err(Error::Authentication);
        };
        self.received += 1;
        if self.received.is_multiple_of(REKEY_RECORDS) {
            t.rekey_incoming();
        }
        output.truncate(n);
        if Record::decode(&output).is_err() {
            self.close();
            return Err(Error::Input);
        }
        Ok(output)
    }
}
