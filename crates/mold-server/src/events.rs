//! Server-wide lifecycle event broadcast.
//!
//! One `tokio::sync::broadcast` channel fans out [`mold_core::ServerEvent`]s
//! (job lifecycle + gallery mutations) to every subscriber of
//! `GET /api/events`, so a client can watch the whole server over a single
//! SSE connection instead of one per job. Unlike [`crate::resources`] there
//! is no "latest" cache — these are deltas, not snapshots; new subscribers
//! bootstrap from `GET /api/queue` + `GET /api/gallery`.

use mold_core::ServerEvent;
use std::sync::Arc;
use tokio::sync::broadcast;

/// Events are bursty (a batch submit queues several jobs at once), so the
/// buffer is larger than the resources channel's. Lagged receivers skip the
/// gap and recover via the REST endpoints.
const BROADCAST_BUFFER: usize = 64;

pub struct EventBroadcaster {
    tx: broadcast::Sender<ServerEvent>,
}

impl EventBroadcaster {
    pub fn new() -> Arc<Self> {
        let (tx, _rx) = broadcast::channel(BROADCAST_BUFFER);
        Arc::new(Self { tx })
    }

    /// Publish an event. Synchronous — safe to call from `spawn_blocking`
    /// contexts (the queue worker's save path). No-subscriber send errors
    /// are deliberately ignored.
    pub fn publish(&self, event: ServerEvent) {
        let _ = self.tx.send(event);
    }

    pub fn subscribe(&self) -> broadcast::Receiver<ServerEvent> {
        self.tx.subscribe()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn publish_reaches_all_subscribers() {
        let events = EventBroadcaster::new();
        let mut a = events.subscribe();
        let mut b = events.subscribe();

        events.publish(ServerEvent::JobEnded { id: "j1".into() });

        for rx in [&mut a, &mut b] {
            match rx.recv().await.unwrap() {
                ServerEvent::JobEnded { id } => assert_eq!(id, "j1"),
                other => panic!("unexpected event: {other:?}"),
            }
        }
    }

    #[tokio::test]
    async fn publish_without_subscribers_does_not_panic() {
        let events = EventBroadcaster::new();
        events.publish(ServerEvent::JobQueued {
            id: "j1".into(),
            model: "flux-dev:q4".into(),
        });
        // A subscriber attached afterwards sees nothing (no replay cache).
        let mut rx = events.subscribe();
        assert!(matches!(
            rx.try_recv(),
            Err(broadcast::error::TryRecvError::Empty)
        ));
    }
}
