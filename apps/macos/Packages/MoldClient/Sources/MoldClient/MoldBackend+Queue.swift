import Foundation

/// The live queue: reading it and acting on one entry.
public protocol MoldQueueBackend: Sendable {
    func queue() async throws -> QueueListing
    func cancelJob(id: String) async throws
    func pauseJob(id: String) async throws
    func resumeJob(id: String) async throws
    /// The one route that moves a job BACKWARD, from held to accepted.
    func retryJob(_ entry: QueueEntry, instanceId: String) async throws
}
