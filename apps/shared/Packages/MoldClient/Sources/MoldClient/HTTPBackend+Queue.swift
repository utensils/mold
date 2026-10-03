import Foundation

/// The queue routes `HTTPBackend+Work.swift` didn't have: reading one job in
/// full, reordering, bulk cancel, and bulk batch status.
public extension HTTPBackend {
    /// One job in full, settings included. `GET /api/queue` cannot answer
    /// this: its projection is payload-free by design (`routes.rs:6817-6828`).
    func queueJob(id: String) async throws -> QueueJobDetail {
        try await get("/api/queue/\(escaped(id))")
    }

    /// Private, bounded preview of this job's sealed source image.
    func queueInputThumbnail(id: String) async throws -> Data {
        let (stream, http) = try await relayBytes(request(queueInputThumbnailPath(id)))
        try Self.validateQueueThumbnail(http)
        return try await stream.collected(upTo: Self.queueThumbnailCeiling)
    }

    /// Moves a QUEUED row. `position` is an index into the machine's
    /// `state = 'queued'` rows in dispatch order -- NOT the row's `position`
    /// field, which also counts running rows, and NOT its place on screen
    /// (`generation_queue.rs:1815-1824`, `job_registry.rs:57-65`). Large
    /// values clamp to the back. 409 when the job is already running.
    ///
    /// The body carries `position` alone: the PATCH's other two fields are a
    /// double `Option` (`routes.rs:7230-7247`), and sending a bare object
    /// with only `position` set is what leaves an existing lane pin
    /// untouched rather than resetting it to Auto.
    func reorderJob(id: String, position: Int) async throws {
        try await send("/api/queue/\(escaped(id))", method: "PATCH",
                       body: QueueReorderPatch(position: position))
    }

    /// Cancels every queued or restart-paused row. RUNNING WORK IS UNTOUCHED
    /// (`routes.rs:7854-7859`). Answers how many rows went.
    @discardableResult
    func cancelAllQueued() async throws -> QueueCancelResult {
        let data = try await bytes(for: request("/api/queue", method: "DELETE"))
        do {
            return try MoldJSON.decoder.decode(QueueCancelResult.self, from: data)
        } catch {
            throw MoldClientError.malformedResponse
        }
    }

    /// Authoritative state for many batches in one call. A READ despite the
    /// verb -- `spawn_queue_read` (`routes.rs:3396-3401`) -- and a POST only
    /// because up to `QueueBatchStatusLimit.identities` ids do not fit in a
    /// query string. This backend does not chunk for you.
    func batchStatuses(batchIds: [String]) async throws -> BatchStatusListing {
        try await post("/api/generation-batches/status",
                       body: GenerationBatchStatusQuery(batchIds: batchIds))
    }
}

/// `position`-only: the whole point is that the other two `QueuePatchRequest`
/// fields are simply absent from the JSON object, never `null`.
struct QueueReorderPatch: Encodable {
    let position: Int
}

/// `GenerationBatchStatusRequest`'s wire shape (`types.rs:11056-11062`) has a
/// `client_batch_ids` field too; this backend only ever asks by `batch_ids`,
/// and `#[serde(default)]` on the server means omitting the other key entirely
/// is a real empty list, not a different request.
struct GenerationBatchStatusQuery: Encodable {
    let batchIds: [String]
}

// Streaming validation is separate so header refusal happens before a body
// allocation, including when the host omits or lies about Content-Length.
extension HTTPBackend {
    static let queueThumbnailCeiling = 2 * 1_024 * 1_024

    func queueInputThumbnailPath(_ id: String) -> String {
        "/api/queue/\(escaped(id))/input-thumbnail"
    }

    static func validateQueueThumbnail(_ http: HTTPURLResponse) throws {
        guard (200 ..< 300).contains(http.statusCode) else {
            if http.statusCode == 401 { throw MoldClientError.unauthorized }
            throw MoldClientError.http(status: http.statusCode, code: nil, message: nil)
        }
        guard http.expectedContentLength <= Int64(queueThumbnailCeiling) else {
            throw ResponseCeiling.Exceeded(bytes: Int(clamping: http.expectedContentLength),
                ceiling: queueThumbnailCeiling, what: "queue source thumbnail")
        }
    }
}
