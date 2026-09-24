import Foundation

// Changing what is in a gallery, and getting things out of it.
public extension HTTPBackend {
    func patch(_ filename: String, with patch: GalleryPatch) async throws {
        var request = self.request("/api/gallery/image/\(escaped(filename))")
        request.httpMethod = "PATCH"
        request.setValue("application/json", forHTTPHeaderField: "Content-Type")
        request.httpBody = try MoldJSON.encoder.encode(patch)
        _ = try await bytes(for: request)
    }

    /// Replay-safe by `operationId`, so a retry cannot double-apply.
    func mutate(_ mutation: GalleryBulkMutation) async throws {
        _ = try await postRaw("/api/gallery/mutations", body: mutation)
    }

    /// Moves prints to the trash, where they keep their own purge countdown.
    func trash(_ filenames: [String]) async throws {
        _ = try await postRaw("/api/gallery/trash", body: TrashRequest(filenames: filenames))
    }

    func restoreFromTrash(_ filenames: [String]) async throws {
        _ = try await postRaw("/api/gallery/trash/restore",
                              body: TrashRequest(filenames: filenames))
    }

    /// Permanent. There is no undo on the host side.
    func deleteForever(_ filenames: [String]) async throws {
        _ = try await postRaw("/api/gallery/trash/delete-forever",
                              body: TrashRequest(filenames: filenames))
    }

    func trashedPrints(etag: String?) async throws -> Fetched<[GalleryPrint]> {
        try await galleryListing(view: "trash", etag: etag)
    }

    func collections() async throws -> [Collection] {
        try await get("/api/gallery/collections")
    }

    func tags() async throws -> [TagCount] {
        try await get("/api/gallery/tags")
    }

    func exportOptions() async throws -> ExportOptions {
        try await get("/api/gallery/export-options")
    }

    /// Asks the host to convert a stored print and hands back the bytes.
    ///
    /// The conversion happens on the machine that holds the print, so the app
    /// never needs a decoder for every container mold can write.
    func export(_ filename: String, format: String) async throws -> Data {
        try await export(filename, request: .geometry(format: format, nil))
    }

    /// The same route with the optional controls filled in.
    ///
    /// `MeshExportRequest` carries at most ONE of the two groups, because the
    /// server refuses geometry keys on a turntable and turntable keys on a
    /// geometry container -- so a body built here can never be one it rejects.
    func export(_ filename: String, request body: MeshExportRequest) async throws -> Data {
        var request = self.request("/api/gallery/export/\(escaped(filename))")
        request.httpMethod = "POST"
        request.setValue("application/json", forHTTPHeaderField: "Content-Type")
        request.httpBody = try MoldJSON.encoder.encode(body)
        request.timeoutInterval = 300
        return try await bytes(for: request)
    }

    /// The host's own rendered POSTER for a print, with the API key.
    ///
    /// Thumbnails are deliberately not ticketable (`MediaURL`), so this is the
    /// one route that reaches one -- which is what lets Quick Look show a
    /// mesh's poster instead of a container macOS has no previewer for.
    func thumbnail(_ filename: String, size: Int, trashed: Bool) async throws -> Data {
        var path = "/api/gallery/thumbnail/\(escaped(filename))"
            + "?size=\(RouteEscaping.escapedQueryValue(String(size)))"
        if trashed { path += "&view=trash" }
        return try await bytes(for: request(path))
    }

    /// The original bytes as stored.
    func media(_ filename: String, trashed: Bool) async throws -> Data {
        try await bytes(for: mediaRequest(filename, trashed: trashed))
    }

    func mediaFile(_ filename: String, trashed: Bool) async throws -> URL {
        let request = mediaRequest(filename, trashed: trashed)
        let temporary: URL
        let response: URLResponse
        do {
            (temporary, response) = try await session.download(for: request, delegate: redirectGuard)
        } catch let error as URLError {
            throw TransportFailure.from(error)
        }
        guard let http = response as? HTTPURLResponse else {
            throw MoldClientError.malformedResponse
        }
        if !(200...299).contains(http.statusCode) {
            let handle = try FileHandle(forReadingFrom: temporary)
            defer { try? handle.close() }
            let details = try handle.read(upToCount: 1_024 * 1_024) ?? Data()
            try HTTPRefusal.check(http, details)
            throw MoldClientError.malformedResponse
        }
        let owned = FileManager.default.temporaryDirectory
            .appendingPathComponent("mold-library-sync-\(UUID().uuidString)")
        try FileManager.default.moveItem(at: temporary, to: owned)
        return owned
    }

    /// A trashed print is behind the trash view, exactly as the listing is --
    /// asking the live route for one answers 404 on a print that is right
    /// there.
    internal func mediaRequest(_ filename: String, trashed: Bool) -> URLRequest {
        let path = "/api/gallery/image/\(escaped(filename))"
        var request = request(trashed ? path + "?view=trash" : path)
        // A clip is tens of megabytes; the 10 s idle timeout every other route
        // gets is for a JSON answer. Same allowance as `export`.
        request.timeoutInterval = 300
        return request
    }

    internal func postRaw<Body: Encodable>(_ path: String, body: Body) async throws -> Data {
        var request = self.request(path)
        request.httpMethod = "POST"
        request.setValue("application/json", forHTTPHeaderField: "Content-Type")
        request.httpBody = try MoldJSON.encoder.encode(body)
        return try await bytes(for: request)
    }
}
