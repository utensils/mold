import Foundation
import MoldClient

/// One machine's library as last listed, kept on this device so the grid
/// shows at once on launch and stays browsable with no connection.
nonisolated struct LibrarySnapshot: Codable, Equatable {
    var prints: [GalleryPrint]
    var trashed: [GalleryPrint]?
    var collections: [Collection]?
    var etag: String?
    var trashEtag: String?
}

/// Where snapshots live: one small JSON file per machine, beside the offline
/// pictures (`ThumbnailLoader.defaultDirectory`), written with the app's own
/// strategy-free coders so it reads back exactly as written.
nonisolated struct LibrarySnapshots: Sendable {
    let directory: URL

    static let standard = LibrarySnapshots(directory: ThumbnailLoader.defaultDirectory.appending(path: "library"))

    nonisolated func load(_ id: UUID) -> LibrarySnapshot? {
        guard let data = try? Data(contentsOf: file(id)) else { return nil }
        return try? MoldJSON.localDecoder.decode(LibrarySnapshot.self, from: data)
    }

    nonisolated func save(_ snapshot: LibrarySnapshot, for id: UUID) {
        guard let data = try? MoldJSON.localEncoder.encode(snapshot) else { return }
        try? FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        try? data.write(to: file(id), options: .atomic)
    }

    nonisolated func remove(_ id: UUID) {
        try? FileManager.default.removeItem(at: file(id))
    }

    nonisolated func purge() {
        try? FileManager.default.removeItem(at: directory)
    }

    nonisolated private func file(_ id: UUID) -> URL { directory.appending(path: "\(id.uuidString).json") }
}
