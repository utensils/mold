import Foundation

/// The whole-queue gate: stop dispatching new work, and start again.
///
/// Distinct from pausing ONE row (`MoldQueueBackend.pauseJob`): this names
/// the QUEUE. Running work is untouched either way -- pausing stops the next
/// job starting, it does not stop the one already on the GPU.
///
/// Gated on `capabilities.queue.can_pause`, which defaults to `false` so an
/// older machine that omits it is treated as lacking the control
/// (`types.rs:11458-11465`).
public protocol MoldQueueGateBackend: Sendable {
    /// `POST /api/queue/pause`. Answers the gate's new state.
    @discardableResult
    func pauseQueue() async throws -> QueuePauseState
    /// `POST /api/queue/resume`.
    @discardableResult
    func resumeQueue() async throws -> QueuePauseState
}

/// `{"paused": true}` (`types.rs:4486-4491`).
public struct QueuePauseState: Codable, Hashable, Sendable {
    public let paused: Bool

    public init(paused: Bool) { self.paused = paused }
}
