//! What a durable queue row becomes when the work it names cannot finish now.
//!
//! One lattice for admission-authority restore, deferred media hydration, and
//! worker-side settlement. Three separate enums used to say this — one per
//! producer — and every seam between them was a hand-written 1:1 map. The
//! single consumer of their meaning is `settle_one`, which decodes to
//! `(retain, retryable)`; the type below simply says that directly.

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum DurableDisposition {
    /// Not a failure of the request: the claim returns to the backlog and the
    /// next feeder pass or boot replays it unchanged.
    Retain,
    /// Parked for a human. `retryable` is whether `POST /api/queue/:id/retry`
    /// may return the row to the queue unchanged.
    Hold { retryable: bool },
}
