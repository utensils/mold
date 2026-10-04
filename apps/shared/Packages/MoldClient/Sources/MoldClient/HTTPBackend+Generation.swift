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
        if let refusal = admission.retainedMediaBatchRefusal { throw refusal }
        return try await submitWithReferenceUploads(admission)
    }

    internal func postAdmission(_ admission: BatchAdmission) async throws -> BatchStatus {
        var request = try body("/api/generation-batches", method: "POST", admission)
        if let handle = admission.retainedMediaSession {
            request.setValue(handle, forHTTPHeaderField: RetainedSourceMedia.sessionHeader)
        }
        guard (request.httpBody?.count ?? 0) <= RequestBodyLimit.bytes else {
            throw ReferenceUploadPolicy.refusal("REFERENCE_REQUEST_TOO_LARGE", "The reference batch exceeds the host's request limit. Make fewer copies at a time.")
        }
        // Scoped upload and retained credentials must never travel in a redirected POST.
        let containsHandles = admission.retainedMediaSession != nil || admission.requests.contains {
            $0.references?.contains { $0.media.authority == "upload" } == true
        }
        let data: Data
        if containsHandles { data = try await referenceUploadBytes(request) }
        else { data = try await bytes(for: request) }
        return try decoded(BatchStatus.self, from: data, route: "/api/generation-batches")
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
