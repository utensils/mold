import Foundation

/// The two queue flags whose absence means something other than "older
/// server, ask anyway" -- kept beside `Capabilities+Reading`'s own answers so
/// a client never reads a raw `Bool?` and guesses.
public extension Capabilities {
    /// Whether this machine can stop work that has already STARTED.
    ///
    /// Absence is a definitive no, and unusually it is the server that says
    /// so: "Older servers omit this and clients keep running rows read-only"
    /// (`types.rs:11475-11479`). Web gates its running-row cancel on exactly
    /// this (`useQueueInspection.ts:153-155`) and re-checks it at action time
    /// (`:299-305`); desktop reads it at `jobs.ts:257`.
    ///
    /// A queued, held or paused row is a different question -- nothing is
    /// running, and `DELETE /api/queue/:id` removes it on every host.
    var canCancelRunningJob: Bool { queue?.cooperativeCancellation ?? false }
}
