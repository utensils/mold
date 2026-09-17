import Foundation

// Asking a host to make something, and following it while it does.
// Split from the core transport purely for size.
//
// The ids are UUIDs today, so `escaped(_:)` changes nothing on the wire --
// it is here because `URLComponents.percentEncodedPath` RAISES rather than
// answering nil, so an id shape that ever grows a `?` or a space would crash
// instead of failing the request (`HTTPBackend+Work.swift` carries the same
// note).
public extension HTTPBackend {
    func submit(_ admission: BatchAdmission) async throws -> BatchStatus {
        try await post("/api/generation-batches", body: admission)
    }

    func batchStatus(id: String) async throws -> BatchStatus {
        try await get("/api/generation-batches/\(escaped(id))")
    }

    func batchStatus(clientBatchId: String) async throws -> BatchStatus {
        try await get("/api/generation-batches/by-client/\(escaped(clientBatchId))")
    }

    func jobPreview(jobId: String) async throws -> JobProgress? {
        // The route answers `null` while there is nothing to show yet, which
        // is an ordinary state and not an error.
        try await get("/api/queue/\(escaped(jobId))/preview")
    }

    func cancelBatch(id: String) async throws {
        var request = self.request("/api/generation-batches/\(escaped(id))")
        request.httpMethod = "DELETE"
        _ = try await bytes(for: request)
    }

    func placementPreview(
        _ request: GenerateRequest, copies: Int
    ) async throws -> PlacementPreview {
        try await post("/api/generate/placement-preview",
                       body: PlacementRequest(request: request, copies: copies))
    }

    /// Follows a batch to settlement.
    ///
    /// Every frame is a COMPLETE status, never a delta, so a dropped
    /// connection costs nothing: reconnecting re-reads the whole truth. That
    /// is also why the buffering policy is `latestOnly` -- an older frame
    /// says nothing the newer one does not, and the settled frame is always
    /// the last (`StreamBuffering`).
    func batchEvents(id: String) -> AsyncThrowingStream<BatchStatus, Error> {
        AsyncThrowingStream(bufferingPolicy: .bufferingNewest(StreamBuffering.latestOnly)) { continuation in
            let task = Task {
                do {
                    // A stream has no business timing out while it sits idle
                    // between denoise steps.
                    for try await frame in stream(
                        "/api/generation-batches/\(escaped(id))/events", timeout: 3_600
                    ) {
                        guard frame.name == "generation_batch",
                              let data = frame.data.data(using: .utf8),
                              let status = try? MoldJSON.decoder.decode(
                                  BatchStatus.self, from: data)
                        else { continue }
                        continuation.yield(status)
                        if status.isSettled { break }
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
