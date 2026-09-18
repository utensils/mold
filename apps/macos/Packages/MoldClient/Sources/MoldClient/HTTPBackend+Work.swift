import Foundation

// Acting on queued work, and fetching models.
//
// Every id below is server- or client-minted and is a UUID today, so nothing
// here is reachable -- but `URLComponents.percentEncodedPath` RAISES on an
// invalid character rather than answering nil, so the failure mode if an id
// shape ever changes is a crash, not a bad request. Every comparable route
// (`queueJob`, `modelPath`, `transferExportPath`, `revokePairedClient`)
// already escapes; these were the exceptions.
public extension HTTPBackend {
    // MARK: - Queue

    func cancelJob(id: String) async throws {
        var request = self.request("/api/queue/\(escaped(id))")
        request.httpMethod = "DELETE"
        _ = try await bytes(for: request)
    }

    func pauseJob(id: String) async throws {
        _ = try await postRaw("/api/queue/\(escaped(id))/pause", body: EmptyBody())
    }

    func resumeJob(id: String) async throws {
        _ = try await postRaw("/api/queue/\(escaped(id))/resume", body: EmptyBody())
    }

    /// The one route that moves a job BACKWARD, from held to queued. It needs
    /// the full fenced identity so a retry cannot be aimed at the wrong job
    /// (`routes.rs:7617-7650`).
    func retryJob(_ authority: QueueAuthority) async throws {
        _ = try await postRaw("/api/queue/\(escaped(authority.jobId))/retry", body: authority)
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
        try HTTPRefusal.check(http, data)
        throw MoldClientError.malformedResponse
    }

    func cancelDownload(id: String) async throws {
        var request = self.request("/api/downloads/\(escaped(id))")
        request.httpMethod = "DELETE"
        _ = try await bytes(for: request)
    }

    /// Progress for every download on this host.
    ///
    /// Several models fetch at once, so a frame is one model's progress
    /// rather than a whole picture and the ceiling is generous
    /// (`StreamBuffering.frames`) -- but it keeps the NEWEST, because a
    /// progress figure a later frame supersedes is the one worth losing and
    /// a terminal frame is always the last one a job sends.
    func downloadEvents() -> AsyncThrowingStream<DownloadEvent, Error> {
        AsyncThrowingStream(bufferingPolicy: .bufferingNewest(StreamBuffering.frames)) { continuation in
            let task = Task {
                do {
                    for try await frame in stream("/api/downloads/stream", timeout: 3_600) {
                        guard let data = frame.data.data(using: .utf8),
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
