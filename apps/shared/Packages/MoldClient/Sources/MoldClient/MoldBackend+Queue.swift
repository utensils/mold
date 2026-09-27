import Foundation

/// The live queue: reading it and acting on one entry, one batch, or the
/// whole waiting set.
public protocol MoldQueueBackend: Sendable {
    func queue() async throws -> QueueListing
    /// One job in full, settings included -- `GET /api/queue` cannot answer
    /// this (`routes.rs:6817-6828`).
    func queueJob(id: String) async throws -> QueueJobDetail
    func cancelJob(id: String) async throws
    /// Cancels the row ONLY if it is held right now (`?only_held=true`).
    /// Returns `false` when the machine says it is no longer held -- a Retry
    /// got there first -- and nothing was touched. An older machine ignores
    /// the query and cancels whatever the row is.
    func cancelHeldJob(id: String) async throws -> Bool
    func pauseJob(id: String) async throws
    func resumeJob(id: String) async throws
    /// Moves a QUEUED row to `position` in the machine's dispatch order --
    /// not the row's own `position` field and not its place on screen
    /// (`generation_queue.rs:1815-1824`, `job_registry.rs:57-65`).
    func reorderJob(id: String, position: Int) async throws
    /// The one route that moves a job BACKWARD, from held to queued.
    func retryJob(_ authority: QueueAuthority) async throws
    /// Cancels every queued or restart-paused row. Running work is untouched.
    @discardableResult
    func cancelAllQueued() async throws -> QueueCancelResult
    /// Authoritative state for many batches in one call -- a READ despite the
    /// verb (`routes.rs:3383-3418`).
    func batchStatuses(batchIds: [String]) async throws -> BatchStatusListing

    /// Exports a HELD row as a portable request with its media inlined. The
    /// bytes are opaque and must never be decoded (`queue_transfer.rs:34-97`).
    func exportHeldJob(_ authority: QueueAuthority) async throws -> Data
    /// Admits one exported request HERE, fenced on this machine's identity
    /// via `x-mold-destination-instance` (`routes.rs:2973-2994`).
    func admitTransfer(
        clientBatchId: String, portable: Data, destinationInstance: String
    ) async throws -> BatchStatus
    /// Cancels the held source row, and ONLY after the destination accepted
    /// (`routes.rs:7584-7599`).
    func completeTransfer(_ authority: QueueAuthority) async throws
}
