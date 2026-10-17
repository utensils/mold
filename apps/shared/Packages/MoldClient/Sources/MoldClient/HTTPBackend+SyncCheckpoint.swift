import Foundation

public extension HTTPBackend {
    /// What the server accepts in one bounded request
    /// (`gallery_sync::MAX_CHECKPOINT_FILENAMES`).
    static let boundedCheckpointLimit = 2_000

    /// `POST /api/gallery/sync-checkpoint {filenames}`, in chunks the server
    /// accepts. An older server (404/405) answers through the whole GET,
    /// restricted to the same names; one that cannot answer at all is nil.
    func gallerySyncCheckpoint(filenames: [String]) async throws -> GallerySyncCheckpoint? {
        let names = Array(Set(filenames)).sorted()
        var combined: GallerySyncCheckpoint?
        var start = 0
        repeat {
            let chunk = Array(names[start..<min(start + Self.boundedCheckpointLimit, names.count)])
            start += max(chunk.count, 1)
            let answer: GallerySyncCheckpoint
            do {
                let data = try await postRaw("/api/gallery/sync-checkpoint",
                                             body: BoundedCheckpointRequest(filenames: chunk), timeout: 60)
                answer = try MoldJSON.decoder.decode(GallerySyncCheckpoint.self, from: data)
            } catch let error as MoldClientError {
                guard case let .http(status, _, _) = error, status == 404 || status == 405 else { throw error }
                return try await gallerySyncCheckpoint()?.restricted(to: names)
            }
            guard answer.protocolVersion == 1 else { return nil }
            if let earlier = combined {
                // A restart between chunks is not one checkpoint.
                guard let merged = earlier.merged(with: answer) else { return nil }
                combined = merged
            } else {
                combined = answer
            }
        } while start < names.count
        return combined
    }
}

private struct BoundedCheckpointRequest: Encodable {
    let filenames: [String]
}
