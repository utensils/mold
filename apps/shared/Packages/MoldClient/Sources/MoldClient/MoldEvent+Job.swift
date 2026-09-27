import Foundation

/// The queue frames `MoldEvent.init(name:data:)` decodes, split out of
/// `MoldEvent.swift` for size. `ServerEvent` (`types.rs:13137-13234`); every
/// one of these arrives under the SSE name `event`, never its own frame name.
public extension MoldEvent {
    enum Job: Hashable, Sendable {
        case queued(id: String, model: String)
        case started(id: String, model: String, gpu: Int?)
        /// Left the queue for ANY reason -- completed, errored, cancelled, or
        /// the client disconnected. NOT a success signal: `gallery(.added)`
        /// is (`types.rs:13153-13157`).
        case ended(id: String)
        /// One durable child committed a new authoritative state, AFTER the
        /// SQLite transaction landed -- so it is safe to reconcile through
        /// `/api/generation-batches/status` the moment it arrives.
        case stateCommitted(id: String)
        /// ONE transaction touched many children (a cancel-all, a batch
        /// cancel). Reconcile the machine ONCE; treating it as an event storm
        /// is exactly what the server emitted it to avoid
        /// (`types.rs:13163-13167`).
        case statesCommitted
    }

    enum Queue: Hashable, Sendable {
        /// Edge-triggered: an idempotent no-op pause is silent
        /// (`types.rs:13228-13231`).
        case paused, resumed
        /// The V2 scheduler published a newer plan. Carries nothing here:
        /// this app draws no lane plan, and decoding `QueuePlan` to discard
        /// it would be a wire surface with no reader.
        case planChanged
    }
}
