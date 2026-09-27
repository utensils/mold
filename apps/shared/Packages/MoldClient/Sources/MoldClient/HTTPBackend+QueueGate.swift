import Foundation

public extension HTTPBackend {
    /// `POST /api/queue/pause` (`routes.rs`). No body either way; the answer
    /// is the gate's new state, which is what a caller writes down rather
    /// than the intent it sent.
    @discardableResult
    func pauseQueue() async throws -> QueuePauseState {
        try await gate("/api/queue/pause")
    }

    @discardableResult
    func resumeQueue() async throws -> QueuePauseState {
        try await gate("/api/queue/resume")
    }

    private func gate(_ path: String) async throws -> QueuePauseState {
        let data = try await bytes(for: request(path, method: "POST"))
        return try decoded(QueuePauseState.self, from: data, route: path)
    }
}
