import Foundation

// The durable chain job an over-long clip becomes. A NEW file rather than
// growing `+Generation.swift`: `HTTPBackend` is already well past the
// 600-line advisory `make lint` tracks, and a chain is not a batch.
public extension HTTPBackend {
    func createChainJob(
        _ request: AutoChainRequest, operationId: String
    ) async throws -> CreateChainJobResponse {
        var http = try body("/api/chain-jobs", method: "POST", request)
        // The server's own idempotency key. Same contract as a batch's
        // `client_batch_id`: replaying it returns the work already held
        // rather than starting a second render.
        http.setValue(operationId, forHTTPHeaderField: "x-mold-operation-id")
        return try decoded(CreateChainJobResponse.self,
                           from: try await bytes(for: http), route: "/api/chain-jobs")
    }

    func chainJob(id: String) async throws -> ChainJobDetail {
        try await get("/api/chain-jobs/\(escaped(id))")
    }

    func cancelChainJob(id: String) async throws {
        try await send("/api/chain-jobs/\(escaped(id))/cancel", method: "POST", body: EmptyBody())
    }

    /// Follows one chain job.
    ///
    /// Unlike a batch's stream, these frames are DELTAS: `stage_start`,
    /// `denoise_step` and `stage_done` each say one thing and the `snapshot`
    /// at the head says the rest. So the buffering policy is NOT `latestOnly`
    /// -- dropping an older frame here loses a stage boundary the newer one
    /// does not repeat, and the stage counter would stall.
    func chainJobEvents(id: String) -> AsyncThrowingStream<ChainJobEvent, Error> {
        AsyncThrowingStream(bufferingPolicy: .unbounded) { continuation in
            let task = Task {
                do {
                    // A chain sits idle between clips for as long as a clip
                    // takes; a stream has no business timing out there.
                    for try await frame in stream(
                        "/api/chain-jobs/\(escaped(id))/events", timeout: 3_600
                    ) {
                        guard let data = frame.data.data(using: .utf8),
                              let event = try? MoldJSON.decoder.decode(
                                  ChainJobEvent.self, from: data)
                        else { continue }
                        continuation.yield(event)
                        if case let .stateChanged(state, _) = event, state.isTerminal { break }
                    }
                    continuation.finish()
                } catch {
                    continuation.finish(throwing: error)
                }
            }
            continuation.onTermination = { _ in task.cancel() }
        }
    }
}
