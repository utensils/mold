//! Bounded, payload-free observation ingress for directly admitted durable jobs.
//!
//! SQLite remains execution authority. This registry only lets one connected
//! direct caller observe the exact feeder-owned job it admitted. A dropped
//! registration detaches observation; it never cancels durable work.

use std::collections::{HashMap, VecDeque};
use std::sync::{Arc, Mutex, Weak};

use crate::job_supervisor::SupervisedOutcome;
use crate::state::{SseCompletionPayload, SseMessage};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum ObserverMode {
    Raw,
    Sse(SseCompletionPayload),
}

pub(crate) enum AttachedObserver {
    Raw {
        outcome: tokio::sync::oneshot::Receiver<SupervisedOutcome>,
    },
    Sse {
        messages: tokio::sync::mpsc::UnboundedReceiver<SseMessage>,
    },
}

struct ObserverEntry {
    mode: ObserverMode,
    committed: bool,
    dispatch: tokio::sync::oneshot::Sender<AttachedObserver>,
}

#[derive(Default)]
struct RegistryState {
    entries: HashMap<String, ObserverEntry>,
    committed_ids: VecDeque<String>,
}

pub(crate) struct QueueMediaIngress {
    capacity: usize,
    state: Mutex<RegistryState>,
}

impl QueueMediaIngress {
    pub(crate) fn new(capacity: usize) -> Arc<Self> {
        Arc::new(Self {
            capacity,
            state: Mutex::new(RegistryState::default()),
        })
    }

    /// Reserve at most one observer for one not-yet-published job. Exhaustion
    /// is deliberately `None`: durable admission still succeeds detached.
    pub(crate) fn reserve(
        self: &Arc<Self>,
        job_id: &str,
        mode: ObserverMode,
    ) -> Option<ObserverRegistration> {
        let mut state = self
            .state
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        if self.capacity == 0
            || state.entries.len() >= self.capacity
            || state.entries.contains_key(job_id)
        {
            return None;
        }
        let (dispatch, receiver) = tokio::sync::oneshot::channel();
        state.entries.insert(
            job_id.to_string(),
            ObserverEntry {
                mode,
                committed: false,
                dispatch,
            },
        );
        Some(ObserverRegistration {
            registry: Arc::downgrade(self),
            job_id: job_id.to_string(),
            receiver: Some(receiver),
        })
    }

    /// Publish an exact ID only after its DB transaction committed.
    pub(crate) fn publish_committed(&self, job_id: &str) {
        let mut state = self
            .state
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        let Some(entry) = state.entries.get_mut(job_id) else {
            return;
        };
        if entry.committed {
            return;
        }
        entry.committed = true;
        state.committed_ids.push_back(job_id.to_string());
        debug_assert!(state.committed_ids.len() <= self.capacity);
    }

    /// Peek the oldest still-attached committed ID. Stale queue entries are
    /// pruned without creating an unbounded waiter or task.
    pub(crate) fn next_committed_id(&self) -> Option<String> {
        let mut state = self
            .state
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        loop {
            let id = state.committed_ids.front()?.clone();
            if state.entries.get(&id).is_some_and(|entry| entry.committed) {
                return Some(id);
            }
            state.committed_ids.pop_front();
        }
    }

    /// Transfer one claimed observer to the sole feeder. FIFO claims use the
    /// same method, so an attached row claimed as ordinary oldest work still
    /// preserves its response channel exactly once.
    pub(crate) fn take_claimed(&self, job_id: &str) -> Option<ObserverClaim> {
        let mut state = self
            .state
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        let entry = state.entries.remove(job_id)?;
        if !entry.committed {
            state.entries.insert(job_id.to_string(), entry);
            return None;
        }
        if state.committed_ids.front().is_some_and(|id| id == job_id) {
            state.committed_ids.pop_front();
        }
        Some(ObserverClaim {
            mode: entry.mode,
            dispatch: Some(entry.dispatch),
        })
    }

    /// Stop preferentially claiming an attached row that is still behind the
    /// bounded runtime prefix, while preserving its observer for the eventual
    /// authoritative FIFO claim.
    ///
    /// The row remains committed and attached. It is removed only from the
    /// exact-ID hint deque, so `take_claimed` still transfers the observer when
    /// the ordinary durable feeder reaches it.
    pub(crate) fn defer_claimed_hint(&self, job_id: &str) {
        let mut state = self
            .state
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        state.committed_ids.retain(|id| id != job_id);
    }

    /// An exact claim that found no row means cancellation or another claim
    /// already won. Close only the observer; execution authority is untouched.
    pub(crate) fn discard_hint(&self, job_id: &str) {
        self.detach(job_id);
    }

    fn detach(&self, job_id: &str) {
        let mut state = self
            .state
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        state.entries.remove(job_id);
    }

    #[cfg(test)]
    pub(crate) fn attached_len(&self) -> usize {
        self.state
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .entries
            .len()
    }
}

pub(crate) struct ObserverRegistration {
    registry: Weak<QueueMediaIngress>,
    job_id: String,
    receiver: Option<tokio::sync::oneshot::Receiver<AttachedObserver>>,
}

impl ObserverRegistration {
    pub(crate) async fn attached(mut self) -> Result<AttachedObserver, &'static str> {
        self.receiver
            .take()
            .expect("observer receiver is consumed once")
            .await
            .map_err(|_| "durable observer detached before feeder claim")
    }
}

impl Drop for ObserverRegistration {
    fn drop(&mut self) {
        if let Some(registry) = self.registry.upgrade() {
            registry.detach(&self.job_id);
        }
    }
}

pub(crate) struct ObserverClaim {
    mode: ObserverMode,
    dispatch: Option<tokio::sync::oneshot::Sender<AttachedObserver>>,
}

impl ObserverClaim {
    pub(crate) fn mode(&self) -> ObserverMode {
        self.mode
    }

    pub(crate) fn deliver(mut self, observer: AttachedObserver) {
        if let Some(dispatch) = self.dispatch.take() {
            let _ = dispatch.send(observer);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn reserved_ids_are_invisible_until_commit_and_disconnect_only_detaches() {
        let ingress = QueueMediaIngress::new(1);
        let registration = ingress.reserve("job", ObserverMode::Raw).unwrap();
        assert_eq!(ingress.next_committed_id(), None);
        ingress.publish_committed("job");
        assert_eq!(ingress.next_committed_id().as_deref(), Some("job"));
        drop(registration);
        assert_eq!(ingress.next_committed_id(), None);
        assert_eq!(ingress.attached_len(), 0);
    }

    #[test]
    fn capacity_is_the_existing_runtime_bound_and_overflow_is_detached() {
        let ingress = QueueMediaIngress::new(1);
        let _first = ingress.reserve("one", ObserverMode::Raw).unwrap();
        assert!(ingress.reserve("two", ObserverMode::Raw).is_none());
        assert_eq!(ingress.attached_len(), 1);
    }

    #[tokio::test]
    async fn deferred_exact_hint_keeps_observer_for_later_fifo_claim() {
        let ingress = QueueMediaIngress::new(1);
        let registration = ingress.reserve("job", ObserverMode::Raw).unwrap();
        ingress.publish_committed("job");

        ingress.defer_claimed_hint("job");

        assert_eq!(ingress.next_committed_id(), None);
        assert_eq!(ingress.attached_len(), 1);
        let claimed = ingress
            .take_claimed("job")
            .expect("the eventual FIFO claim retains the observer");
        let (_outcome_tx, outcome_rx) = tokio::sync::oneshot::channel();
        claimed.deliver(AttachedObserver::Raw {
            outcome: outcome_rx,
        });
        assert!(matches!(
            registration.attached().await.unwrap(),
            AttachedObserver::Raw { .. }
        ));
    }
}
