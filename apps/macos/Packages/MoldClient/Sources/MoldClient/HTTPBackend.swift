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
            let prints = try MoldJSON.decoder.decode([GalleryPrint].self, from: data)
            return .fresh(prints, etag: http.value(forHTTPHeaderField: "ETag"))
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
