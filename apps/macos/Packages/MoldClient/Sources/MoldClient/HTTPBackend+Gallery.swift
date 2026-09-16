import Foundation

// Changing what is in a gallery, and getting things out of it.
public extension HTTPBackend {
    func patch(_ filename: String, with patch: GalleryPatch) async throws {
        var request = self.request("/api/gallery/image/\(filename)")
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
        var request = self.request("/api/gallery/export/\(filename)")
        request.httpMethod = "POST"
        request.setValue("application/json", forHTTPHeaderField: "Content-Type")
        request.httpBody = try JSONSerialization.data(withJSONObject: ["format": format])
        request.timeoutInterval = 300
        return try await bytes(for: request)
    }

    /// The original bytes as stored.
    func media(_ filename: String) async throws -> Data {
        try await bytes(for: request("/api/gallery/image/\(filename)"))
    }

    internal func postRaw<Body: Encodable>(_ path: String, body: Body) async throws -> Data {
        var request = self.request(path)
        request.httpMethod = "POST"
        request.setValue("application/json", forHTTPHeaderField: "Content-Type")
        request.httpBody = try MoldJSON.encoder.encode(body)
        return try await bytes(for: request)
    }
}
