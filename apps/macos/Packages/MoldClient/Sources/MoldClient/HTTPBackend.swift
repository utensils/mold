import Foundation

/// Talks to a `mold serve` over HTTP.
///
/// This is the only transport the app has, and it is the same one an embedded
/// engine will use -- see `MoldBackend`.
public struct HTTPBackend: MoldBackend {
    public let host: MoldHost
    let session: URLSession

    public init(host: MoldHost, session: URLSession = APISession.api) {
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
        try HTTPRefusal.check(http, data)
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
