//! Byte-bounded body buffering, independent of peer-chosen record boundaries.
use anyhow::{bail, Result};
use bytes::Bytes;
use std::{
    collections::VecDeque,
    sync::{Arc, Mutex},
};
use tokio::sync::Notify;
const CHUNK: usize = 16 * 1024;
struct State {
    chunks: VecDeque<Vec<u8>>,
    bytes: usize,
    closed: bool,
    error: Option<anyhow::Error>,
}
struct Shared {
    state: Mutex<State>,
    ready: Notify,
}
pub struct BodySender(Arc<Shared>);
pub struct BodyReceiver(Arc<Shared>);
pub fn body_channel() -> (BodySender, BodyReceiver) {
    let shared = Arc::new(Shared {
        state: Mutex::new(State {
            chunks: VecDeque::new(),
            bytes: 0,
            closed: false,
            error: None,
        }),
        ready: Notify::new(),
    });
    (BodySender(shared.clone()), BodyReceiver(shared))
}
impl BodySender {
    pub fn try_send(&self, item: Result<Bytes>) -> Result<()> {
        let mut state = self.0.state.lock().unwrap_or_else(|e| e.into_inner());
        if state.closed {
            bail!("body consumer closed")
        }
        match item {
            Ok(bytes) => {
                if state.bytes + bytes.len() > mold_secure::STREAM_WINDOW as usize {
                    bail!("body byte bound exceeded")
                }
                state.bytes += bytes.len();
                let mut bytes = bytes.as_ref();
                while !bytes.is_empty() {
                    if state.chunks.back().is_none_or(|chunk| chunk.len() == CHUNK) {
                        state.chunks.push_back(Vec::with_capacity(CHUNK));
                    }
                    let tail = state.chunks.back_mut().unwrap();
                    let copied = (CHUNK - tail.len()).min(bytes.len());
                    tail.extend_from_slice(&bytes[..copied]);
                    bytes = &bytes[copied..];
                }
            }
            Err(error) => {
                state.error = Some(error);
                state.closed = true;
            }
        }
        drop(state);
        self.0.ready.notify_one();
        Ok(())
    }
}
impl Drop for BodySender {
    fn drop(&mut self) {
        self.0
            .state
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .closed = true;
        self.0.ready.notify_one();
    }
}
impl BodyReceiver {
    pub async fn recv(&mut self) -> Option<Result<Bytes>> {
        loop {
            let ready = self.0.ready.notified();
            {
                let mut state = self.0.state.lock().unwrap_or_else(|e| e.into_inner());
                if let Some(bytes) = state.chunks.pop_front() {
                    state.bytes -= bytes.len();
                    return Some(Ok(bytes.into()));
                }
                if let Some(error) = state.error.take() {
                    return Some(Err(error));
                }
                if state.closed {
                    return None;
                }
            }
            ready.await;
        }
    }
}
impl Drop for BodyReceiver {
    fn drop(&mut self) {
        let mut state = self.0.state.lock().unwrap_or_else(|e| e.into_inner());
        state.closed = true;
        state.chunks.clear();
        state.bytes = 0;
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    #[tokio::test]
    async fn paused_one_byte_records_are_coalesced_without_waiting_to_deliver_sse() {
        let (tx, mut rx) = body_channel();
        for _ in 0..65536 {
            tx.try_send(Ok(Bytes::from_static(b"x"))).unwrap();
        }
        assert_eq!(
            tx.0.state
                .lock()
                .unwrap_or_else(|e| e.into_inner())
                .chunks
                .len(),
            65536 / CHUNK
        );
        assert_eq!(rx.recv().await.unwrap().unwrap().len(), CHUNK);
        drop(tx);
        let mut read = CHUNK;
        while let Some(bytes) = rx.recv().await {
            read += bytes.unwrap().len();
        }
        assert_eq!(read, 65536);
        let (tx, mut rx) = body_channel();
        tx.try_send(Ok(Bytes::from_static(b"s"))).unwrap();
        assert_eq!(
            tokio::time::timeout(std::time::Duration::from_millis(50), rx.recv())
                .await
                .unwrap()
                .unwrap()
                .unwrap(),
            Bytes::from_static(b"s")
        );
    }
    #[test]
    fn paused_buffer_rejects_only_excess_bytes_and_frees_on_drop() {
        let (tx, rx) = body_channel();
        tx.try_send(Ok(Bytes::from(vec![
            42;
            mold_secure::STREAM_WINDOW as usize
        ])))
        .unwrap();
        assert!(tx.try_send(Ok(Bytes::from_static(b"x"))).is_err());
        drop(rx);
        assert_eq!(
            tx.0.state.lock().unwrap_or_else(|e| e.into_inner()).bytes,
            0
        );
        assert!(tx.try_send(Ok(Bytes::from_static(b"x"))).is_err());
    }
}
