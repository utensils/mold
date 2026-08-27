//! Server-wide lifecycle event broadcast.
//!
//! One `tokio::sync::broadcast` channel fans out [`mold_core::ServerEvent`]s
//! (job lifecycle + gallery mutations) to every subscriber of
//! `GET /api/events`, so a client can watch the whole server over a single
//! SSE connection instead of one per job. Unlike [`crate::resources`] there
//! is no "latest" cache — these are deltas, not snapshots; new subscribers
//! bootstrap from the REST snapshots, and lagged subscribers receive an
//! explicit `resync_required` frame telling them to repair from those same
//! authorities.

use mold_core::ServerEvent;
use std::sync::Arc;
use tokio::sync::broadcast;
use tokio_stream::wrappers::errors::BroadcastStreamRecvError;

/// Events are bursty (a batch submit queues several jobs at once), so the
/// buffer is larger than the resources channel's. The SSE adapter turns a
/// lag notification into an explicit recovery frame.
const BROADCAST_BUFFER: usize = 64;

pub struct EventBroadcaster {
    tx: broadcast::Sender<ServerEvent>,
    shutdown: tokio_util::sync::CancellationToken,
}

/// What the host-wide SSE stream must do with one broadcast delivery.
///
/// Keeping this classification separate from the async stream makes the
/// reliability contract deterministic to test: a lag notification is a
/// mandatory resync frame, never an event the handler may silently discard.
#[derive(Debug)]
pub(crate) enum BroadcastDelivery {
    Event(ServerEvent),
    ResyncRequired { missed_events: u64 },
}

pub(crate) fn classify_delivery(
    delivery: Result<ServerEvent, BroadcastStreamRecvError>,
) -> BroadcastDelivery {
    match delivery {
        Ok(event) => BroadcastDelivery::Event(event),
        Err(BroadcastStreamRecvError::Lagged(missed_events)) => {
            BroadcastDelivery::ResyncRequired { missed_events }
        }
    }
}

impl EventBroadcaster {
    pub fn new() -> Arc<Self> {
        let (tx, _rx) = broadcast::channel(BROADCAST_BUFFER);
        Arc::new(Self {
            tx,
            shutdown: tokio_util::sync::CancellationToken::new(),
        })
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

    /// End every open-ended SSE subscription before Axum starts its HTTP
    /// drain. Finite request streams keep their normal completion semantics.
    pub fn shutdown(&self) {
        self.shutdown.cancel();
    }

    pub fn shutdown_token(&self) -> tokio_util::sync::CancellationToken {
        self.shutdown.clone()
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

    #[test]
    fn lagged_delivery_requires_explicit_resync() {
        match classify_delivery(Err(BroadcastStreamRecvError::Lagged(17))) {
            BroadcastDelivery::ResyncRequired { missed_events } => {
                assert_eq!(missed_events, 17);
            }
            BroadcastDelivery::Event(event) => {
                panic!("lag was incorrectly forwarded as a normal event: {event:?}")
            }
        }
    }

    #[test]
    fn normal_delivery_preserves_the_server_event() {
        let event = ServerEvent::JobEnded { id: "j1".into() };
        match classify_delivery(Ok(event)) {
            BroadcastDelivery::Event(ServerEvent::JobEnded { id }) => {
                assert_eq!(id, "j1");
            }
            other => panic!("normal event was changed by delivery classification: {other:?}"),
        }
    }

    #[tokio::test]
    async fn shutdown_wakes_all_event_stream_waiters() {
        let events = EventBroadcaster::new();
        let first = events.shutdown_token();
        let second = events.shutdown_token();

        events.shutdown();

        first.cancelled().await;
        second.cancelled().await;
    }
}
