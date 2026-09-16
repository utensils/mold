import Foundation

// Asking a host to make something, and following it while it does.
// Split from the core transport purely for size.
public extension HTTPBackend {
    func submit(_ admission: BatchAdmission) async throws -> BatchStatus {
        try await post("/api/generation-batches", body: admission)
    }

    func batchStatus(id: String) async throws -> BatchStatus {
        try await get("/api/generation-batches/\(id)")
    }

    func batchStatus(clientBatchId: String) async throws -> BatchStatus {
        try await get("/api/generation-batches/by-client/\(clientBatchId)")
    }

    func jobPreview(jobId: String) async throws -> JobProgress? {
        // The route answers `null` while there is nothing to show yet, which
        // is an ordinary state and not an error.
        try await get("/api/queue/\(jobId)/preview")
    }

    func cancelBatch(id: String) async throws {
        var request = self.request("/api/generation-batches/\(id)")
        request.httpMethod = "DELETE"
        _ = try await bytes(for: request)
    }

    func placementPreview(
        _ request: GenerateRequest, copies: Int = 1
    ) async throws -> PlacementPreview {
        try await post("/api/generate/placement-preview",
                       body: PlacementRequest(request: request, copies: copies))
    }

    /// Follows a batch to settlement.
    ///
    /// Every frame is a COMPLETE status, never a delta, so a dropped
    /// connection costs nothing: reconnecting re-reads the whole truth.
    func batchEvents(id: String) -> AsyncThrowingStream<BatchStatus, Error> {
        AsyncThrowingStream { continuation in
            let task = Task {
                do {
                    var request = self.request("/api/generation-batches/\(id)/events")
                    request.setValue("text/event-stream", forHTTPHeaderField: "Accept")
                    // A stream has no business timing out while it sits idle
                    // between denoise steps.
                    request.timeoutInterval = 3_600

                    let (bytes, response) = try await session.bytes(for: request)
                    guard let http = response as? HTTPURLResponse,
                          (200..<300).contains(http.statusCode)
                    else {
                        throw streamFailure(response)
                    }
                    var parser = SSEParser()
                    for try await line in bytes.lines {
                        guard let event = parser.consume(line: line),
                              event.name == "generation_batch",
                              let data = event.data.data(using: .utf8),
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

    private func streamFailure(_ response: URLResponse) -> MoldClientError {
        guard let http = response as? HTTPURLResponse else { return .malformedResponse }
        if http.statusCode == 401 { return .unauthorized }
        return .http(status: http.statusCode, code: nil, message: nil)
    }
}
