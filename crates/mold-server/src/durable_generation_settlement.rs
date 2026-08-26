//! One ordering boundary for durable generation terminal/blocked state.
//!
//! An accepted job's observer must never resolve before SQLite reflects the
//! disposition the observer is about to report. Both the Tokio single-worker
//! path and dedicated GPU owner threads pass through this module.

use crate::queue_journal::{QueueTicket, RetainOutcome};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum DurableDisposition {
    RetryableHold,
    NonRetryableHold,
    Retain,
}

fn settle_one(
    ticket: QueueTicket,
    disposition: DurableDisposition,
    message: &str,
) -> RetainOutcome {
    if ticket.retention_requested() || disposition == DurableDisposition::Retain {
        ticket.retain()
    } else {
        ticket.hold_exact(message, disposition == DurableDisposition::RetryableHold)
    }
}

/// Complete the requested durable transition before a worker publishes its
/// observer result. A retry outcome always keeps and reuses the token-owning
/// ticket; shutdown/fatal-CUDA retention fences take precedence over a hold.
pub(crate) fn settle_blocking(
    ticket: &mut Option<QueueTicket>,
    disposition: DurableDisposition,
    message: &str,
) {
    let Some(mut owned) = ticket.take() else {
        return;
    };
    loop {
        let job_id = owned.id().to_string();
        match settle_one(owned, disposition, message) {
            RetainOutcome::Released | RetainOutcome::Stale => return,
            RetainOutcome::Retry { ticket, error } => {
                tracing::warn!(
                    job = %job_id,
                    %error,
                    ?disposition,
                    "durable generation settlement will retry before observer delivery"
                );
                owned = ticket;
                #[cfg(not(test))]
                std::thread::sleep(mold_db::METADATA_DB_BUSY_TIMEOUT);
                #[cfg(test)]
                std::thread::yield_now();
            }
        }
    }
}

pub(crate) async fn settle_async(
    ticket: &mut Option<QueueTicket>,
    disposition: DurableDisposition,
    message: &str,
) {
    let Some(owned) = ticket.take() else {
        return;
    };
    let message = message.to_string();
    tokio::task::spawn_blocking(move || {
        let mut ticket = Some(owned);
        settle_blocking(&mut ticket, disposition, &message);
    })
    .await
    .expect("durable generation settlement worker panicked");
}
