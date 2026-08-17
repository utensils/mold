//! Detached ownership of a generation job's result channel.
//!
//! A submitted job runs to completion even if the client that asked for it
//! goes away. That guarantee is delivered structurally rather than by teaching
//! each of the fifteen `result_tx.is_closed()` sites about durability: the
//! job's `result_tx` receiver is owned by a detached supervisor task, never by
//! the HTTP handler, so the channel stays open for the job's whole life and
//! every one of those sites keeps reading `false`.
//!
//! The supervisor forwards the outcome to the handler when one is still
//! connected and logs it otherwise. It also owns the job's cancellation
//! signal: an explicit `DELETE /api/queue/:id` is the *only* thing that closes
//! the result channel early, which is what still makes the scheduler and
//! worker skip a cancelled job. Client disconnection no longer does.

use std::sync::Arc;
use tokio::sync::Notify;

use crate::state::GenerationJobResult;

/// What the submitting handler learns about its job.
pub enum SupervisedOutcome {
    /// The worker resolved the job (successfully or not). Boxed because a
    /// successful result carries the whole raster.
    Finished(Box<Result<GenerationJobResult, String>>),
    /// `DELETE /api/queue/:id` removed the job while it was still queued.
    Cancelled,
}

/// The two halves a submitting handler needs: the sender that rides along on
/// the [`crate::state::GenerationJob`], and the receiver the handler awaits.
pub struct SupervisedJob {
    pub result_tx: tokio::sync::oneshot::Sender<Result<GenerationJobResult, String>>,
    pub outcome_rx: tokio::sync::oneshot::Receiver<SupervisedOutcome>,
}

/// Spawn the detached supervisor for one submitted job.
///
/// `cancel` is the signal `JobRegistry::register_job` returned. The supervisor
/// consumes it exclusively — `Notify::notify_one` stores a single permit, so
/// the handler must never also wait on it or one of the two would miss the
/// wakeup.
pub fn supervise_job(job_id: String, cancel: Arc<Notify>) -> SupervisedJob {
    let (result_tx, result_rx) = tokio::sync::oneshot::channel();
    let (outcome_tx, outcome_rx) = tokio::sync::oneshot::channel();

    tokio::spawn(async move {
        // `result_rx` is moved into this block and dropped when it ends, which
        // is what closes the channel after a cancel. It is deliberately never
        // dropped on the delivery path before the outcome is forwarded.
        let delivered = {
            let mut result_rx = result_rx;
            tokio::select! {
                delivered = &mut result_rx => Some(delivered),
                _ = cancel.notified() => None,
            }
        };

        let outcome = match delivered {
            Some(Ok(result)) => SupervisedOutcome::Finished(Box::new(result)),
            Some(Err(_)) => {
                // The job was dropped without resolving. Submission failures
                // are reported by the handler's own error path; anything else
                // here is a genuine leak worth a line.
                tracing::debug!(job = %job_id, "generation job dropped without a result");
                return;
            }
            None => {
                tracing::debug!(job = %job_id, "generation job cancelled while queued");
                SupervisedOutcome::Cancelled
            }
        };

        if outcome_tx.send(outcome).is_err() {
            tracing::info!(
                job = %job_id,
                "generation finished after its client disconnected; the output was still saved"
            );
        }
    });

    SupervisedJob {
        result_tx,
        outcome_rx,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ok_result() -> Result<GenerationJobResult, String> {
        Err("stub".to_string())
    }

    #[tokio::test]
    async fn result_channel_stays_open_after_the_http_receiver_drops() {
        let cancel = Arc::new(Notify::new());
        let SupervisedJob {
            result_tx,
            outcome_rx,
        } = supervise_job("job-1".to_string(), cancel);

        // The HTTP client vanished: axum dropped the handler future, taking
        // the outcome receiver with it.
        drop(outcome_rx);
        tokio::task::yield_now().await;
        tokio::task::yield_now().await;

        assert!(
            !result_tx.is_closed(),
            "a disconnected client must not close the job's result channel"
        );

        // …and the worker's eventual send still succeeds.
        assert!(result_tx.send(ok_result()).is_ok());
    }

    #[tokio::test]
    async fn outcome_reaches_a_connected_handler() {
        let cancel = Arc::new(Notify::new());
        let SupervisedJob {
            result_tx,
            outcome_rx,
        } = supervise_job("job-2".to_string(), cancel);

        assert!(result_tx.send(Err("boom".to_string())).is_ok());

        match outcome_rx.await {
            Ok(SupervisedOutcome::Finished(outcome)) => match *outcome {
                Err(message) => assert_eq!(message, "boom"),
                Ok(_) => panic!("expected the worker's error to survive the hop"),
            },
            _ => panic!("expected the worker outcome to reach the handler"),
        }
    }

    #[tokio::test]
    async fn explicit_cancel_closes_the_result_channel_so_the_worker_skips_the_job() {
        let cancel = Arc::new(Notify::new());
        let SupervisedJob {
            mut result_tx,
            outcome_rx,
        } = supervise_job("job-3".to_string(), cancel.clone());

        cancel.notify_one();

        assert!(matches!(outcome_rx.await, Ok(SupervisedOutcome::Cancelled)));
        result_tx.closed().await;
        assert!(
            result_tx.is_closed(),
            "a cancelled job must still be skipped by the dispatch-time is_closed() gate"
        );
    }
}
