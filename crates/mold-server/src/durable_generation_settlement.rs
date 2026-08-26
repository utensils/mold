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

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum SettlementOutcome {
    /// No durable ticket was attached; preserve legacy observer behavior.
    Untracked,
    /// SQLite committed the requested terminal/blocked transition.
    Settled,
    /// The claim is intentionally replayable, or its exact transition could
    /// not be committed within the bounded owner-thread retry budget.
    Retained,
}

impl SettlementOutcome {
    pub(crate) fn is_retained(self) -> bool {
        self == Self::Retained
    }
}

const MAX_EXACT_ATTEMPTS: usize = 3;

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
) -> SettlementOutcome {
    let Some(mut owned) = ticket.take() else {
        return SettlementOutcome::Untracked;
    };
    for attempt in 1..=MAX_EXACT_ATTEMPTS {
        let job_id = owned.id().to_string();
        let effective = if owned.retention_requested() || disposition == DurableDisposition::Retain
        {
            DurableDisposition::Retain
        } else {
            disposition
        };
        match settle_one(owned, disposition, message) {
            RetainOutcome::Released | RetainOutcome::Stale => {
                return if effective == DurableDisposition::Retain {
                    SettlementOutcome::Retained
                } else {
                    SettlementOutcome::Settled
                };
            }
            RetainOutcome::Retry { ticket, error } => {
                tracing::warn!(
                    job = %job_id,
                    %error,
                    ?disposition,
                    attempt,
                    "durable generation settlement will retry before observer delivery"
                );
                owned = ticket;
            }
        }
    }
    tracing::error!(
        job = %owned.id(),
        ?disposition,
        attempts = MAX_EXACT_ATTEMPTS,
        "durable generation settlement exhausted its bounded retry budget; retaining for restart"
    );
    // `RetainOutcome::Retry` made this ticket's drop inert. Leaving the exact
    // claim in SQLite lets startup recovery reconcile it without pinning a GPU
    // owner or process shutdown forever.
    drop(owned);
    SettlementOutcome::Retained
}

pub(crate) fn settle_completion_blocking(
    ticket: &mut Option<QueueTicket>,
    result_json: &str,
) -> SettlementOutcome {
    let Some(mut owned) = ticket.take() else {
        return SettlementOutcome::Untracked;
    };
    for attempt in 1..=MAX_EXACT_ATTEMPTS {
        let job_id = owned.id().to_string();
        match owned.complete_exact_with_result(Some(result_json)) {
            RetainOutcome::Released | RetainOutcome::Stale => return SettlementOutcome::Settled,
            RetainOutcome::Retry { ticket, error } => {
                tracing::warn!(
                    job = %job_id,
                    %error,
                    attempt,
                    "durable completion commit will retry before success observer delivery"
                );
                owned = ticket;
            }
        }
    }
    tracing::error!(
        job = %owned.id(),
        attempts = MAX_EXACT_ATTEMPTS,
        "durable completion commit exhausted its bounded retry budget; retaining for restart"
    );
    drop(owned);
    SettlementOutcome::Retained
}

pub(crate) async fn settle_async(
    ticket: &mut Option<QueueTicket>,
    disposition: DurableDisposition,
    message: &str,
) -> SettlementOutcome {
    let Some(owned) = ticket.take() else {
        return SettlementOutcome::Untracked;
    };
    let message = message.to_string();
    tokio::task::spawn_blocking(move || {
        let mut ticket = Some(owned);
        settle_blocking(&mut ticket, disposition, &message)
    })
    .await
    .expect("durable generation settlement worker panicked")
}

pub(crate) async fn settle_completion_async(
    ticket: &mut Option<QueueTicket>,
    result_json: &str,
) -> SettlementOutcome {
    let Some(owned) = ticket.take() else {
        return SettlementOutcome::Untracked;
    };
    let result_json = result_json.to_string();
    tokio::task::spawn_blocking(move || {
        let mut ticket = Some(owned);
        settle_completion_blocking(&mut ticket, &result_json)
    })
    .await
    .expect("durable completion settlement worker panicked")
}
