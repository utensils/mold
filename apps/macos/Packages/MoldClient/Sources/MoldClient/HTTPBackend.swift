import Foundation

/// Talks to a `mold serve` over HTTP.
///
/// This is the only transport the app has, and it is the same one an embedded
/// engine will use -- see `MoldBackend`.
public struct HTTPBackend: MoldBackend {
    public let host: MoldHost
    let session: URLSession

    public init(host: MoldHost, session: URLSession = .shared) {
        self.host = host
        self.session = session
    }

    public func status() async throws -> ServerStatus {
        try await get("/api/status")
    }

    public func capabilities() async throws -> Capabilities {
        try await get("/api/capabilities")
    }

    public func models() async throws -> [Model] {
        try await get("/api/models")
    }

    public func queue() async throws -> QueueListing {
        try await get("/api/queue")
    }

    public func gallery(etag: String?) async throws -> Fetched<[GalleryPrint]> {
        try await galleryListing(view: nil, etag: etag)
    }

    /// `view` is `trash` for the deleted shelf, absent for the live library.
    func galleryListing(view: String?, etag: String?) async throws -> Fetched<[GalleryPrint]> {
        var request = self.request(view.map { "/api/gallery?view=\($0)" } ?? "/api/gallery")
        // The index is large and mostly unchanged between refreshes, so ask
        // the host whether it changed at all before it serializes 1.2 MB.
        if let etag { request.setValue(etag, forHTTPHeaderField: "If-None-Match") }
        // Listing a full gallery takes longer than a status probe.
        request.timeoutInterval = 60

        let (data, http) = try await send(request)
        if http.statusCode == 304 { return .notModified }
        try check(http, data)
        do {
            // `GalleryListing`, not `[GalleryPrint]`: a row whose filename is
            // not a safe path component is dropped and logged rather than
            // losing the whole index over one of them.
            let listing = try MoldJSON.decoder.decode(GalleryListing.self, from: data)
            return .fresh(listing.prints, etag: http.value(forHTTPHeaderField: "ETag"))
        } catch {
            throw MoldClientError.malformedResponse
        }
    }
}

/// mold's error envelope. `code` is the part to branch on.
///
/// Every field is optional because half the routes this app calls do not send
/// all of them: `create_download`'s 400 is `{"error": …}` with no code
/// (`routes.rs:11563-11569`), and the catalog routes answer plain text
/// (`catalog_api.rs:586-590`). Requiring both threw the machine's own
/// sentence away and left "the machine answered with an error (400)".
struct APIError: Decodable, Sendable {
    let error: String?
    let code: String?
    /// Present only on a licence refusal (`routes.rs:38-49`).
    let license: LicenseRefusal?
}

extension HTTPBackend {
    /// How much of a refused stream's body is worth reading.
    ///
    /// A refusal body is mold's small `APIError` envelope. Nothing about a
    /// non-2xx promises the other side closes the connection, though, so this
    /// is a ceiling and not an expectation -- 8 KiB is an order of magnitude
    /// more than the largest licence refusal and still nothing to hold.
    static let refusalBodyLimit = 8 * 1024

    /// How long a refused response gets to finish saying why.
    ///
    /// The size ceiling is not enough on its own: a non-2xx promises neither
    /// that the body is small nor that the connection CLOSES, and the error
    /// path already knows the status, so waiting on a held-open socket for a
    /// sentence it does not need was a hang -- up to the request's own
    /// timeout, which on `events` and `resourceStream` is 86,400 seconds.
    static let refusalBodyDeadline: Duration = .seconds(3)

    /// At most `refusalBodyLimit` bytes of a refused response, and at most
    /// `within` waiting for them.
    ///
    /// Whatever arrived before the deadline or a read failure is what there
    /// is to report: the status is already known, and a truncated body simply
    /// decodes to nothing, which is the same answer as no body at all.
    static func refusalBody(
        _ bytes: some AsyncSequence<UInt8, some Error> & Sendable,
        within deadline: Duration = refusalBodyDeadline
    ) async -> Data {
        // A box rather than a return value: the racing read may be CANCELLED
        // part way, and what it had by then is still the best answer there is.
        let read = PartialBody()
        await withTaskGroup(of: Void.self) { group in
            group.addTask {
                do {
                    for try await byte in bytes {
                        if read.append(byte) >= refusalBodyLimit { return }
                    }
                } catch {}
            }
            group.addTask { try? await Task.sleep(for: deadline) }
            await group.next()
            group.cancelAll()
        }
        return read.bytes
    }

    /// A body that is not mold's JSON envelope, when it is short enough to be
    /// a sentence rather than a proxy's HTML page.
    static func plainMessage(_ data: Data) -> String? {
        guard !data.isEmpty, data.count <= 400,
              let text = String(data: data, encoding: .utf8)?
                  .trimmingCharacters(in: .whitespacesAndNewlines),
              !text.isEmpty, !text.hasPrefix("<")
        else { return nil }
        return text
    }
}
