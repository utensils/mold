import Foundation

// Acting on queued work, and fetching models.
public extension HTTPBackend {
    // MARK: - Queue

    func cancelJob(id: String) async throws {
        var request = self.request("/api/queue/\(id)")
        request.httpMethod = "DELETE"
        _ = try await bytes(for: request)
    }

    func pauseJob(id: String) async throws {
        _ = try await postRaw("/api/queue/\(id)/pause", body: EmptyBody())
    }

    func resumeJob(id: String) async throws {
        _ = try await postRaw("/api/queue/\(id)/resume", body: EmptyBody())
    }

    /// The one route that moves a job BACKWARD, from held to accepted. It needs
    /// the full fenced identity so a retry cannot be aimed at the wrong job.
    func retryJob(_ entry: QueueEntry, instanceId: String) async throws {
        struct Retry: Encodable {
            let instanceId: String
            let batchId: String?
            let clientBatchId: String?
            let jobId: String
        }
        _ = try await postRaw("/api/queue/\(entry.id)/retry", body: Retry(
            instanceId: instanceId, batchId: entry.batchId,
            clientBatchId: entry.clientBatchId, jobId: entry.id))
    }

    func pauseQueue() async throws {
        _ = try await postRaw("/api/queue/pause", body: EmptyBody())
    }

    func resumeQueue() async throws {
        _ = try await postRaw("/api/queue/resume", body: EmptyBody())
    }

    // MARK: - Downloads

    /// Queues a model fetch.
    ///
    /// Returns the existing ticket on a 409, because "already downloading" is
    /// the outcome the caller wanted, not a failure.
    func startDownload(_ request: DownloadRequest) async throws -> DownloadTicket {
        var urlRequest = self.request("/api/downloads")
        urlRequest.httpMethod = "POST"
        urlRequest.setValue("application/json", forHTTPHeaderField: "Content-Type")
        urlRequest.httpBody = try MoldJSON.encoder.encode(request)

        let (data, http) = try await send(urlRequest)
        if http.statusCode == 409 || (200..<300).contains(http.statusCode) {
            if let ticket = try? MoldJSON.decoder.decode(DownloadTicket.self, from: data) {
                return ticket
            }
        }
        try check(http, data)
        throw MoldClientError.malformedResponse
    }

    func cancelDownload(id: String) async throws {
        var request = self.request("/api/downloads/\(id)")
        request.httpMethod = "DELETE"
        _ = try await bytes(for: request)
    }

    /// Progress for every download on this host.
    func downloadEvents() -> AsyncThrowingStream<DownloadEvent, Error> {
        AsyncThrowingStream { continuation in
            let task = Task {
                do {
                    var request = self.request("/api/downloads/stream")
                    request.setValue("text/event-stream", forHTTPHeaderField: "Accept")
                    request.timeoutInterval = 3_600
                    let (bytes, _) = try await session.bytes(for: request)
                    var parser = SSEParser()
                    for try await line in bytes.lines {
                        guard let event = parser.consume(line: line),
                              let data = event.data.data(using: .utf8),
                              let decoded = try? MoldJSON.decoder.decode(
                                  DownloadEvent.self, from: data)
                        else { continue }
                        continuation.yield(decoded)
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

struct EmptyBody: Encodable {}
