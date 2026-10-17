import Foundation
import MoldClient
import Synchronization

/// One machine's collections as last listed, kept on this device so shelves
/// show at once on launch and stay browsable with no connection. Its PRINTS
/// live in the machine's `GalleryReplica` file (`LibrarySnapshots.replicas`),
/// the one saved copy of its rows.
nonisolated struct LibrarySnapshot: Codable, Equatable {
    var collections: [Collection]?

    /// What builds before replicas saved here: the whole listing. Read once
    /// by `LibrarySnapshots.migrate`, which hands the rows to the replica.
    struct Legacy: Codable, Equatable {
        var prints: [GalleryPrint]
        var trashed: [GalleryPrint]?
        var collections: [Collection]?
        var etag: String?
        var trashEtag: String?
    }
}

/// Where the offline Library lives: one small collections file per machine,
/// and the machines' replicas beside them, under the offline pictures
/// (`ThumbnailLoader.defaultDirectory`: Application Support, excluded from
/// backup), all written with the app's own strategy-free coders.
nonisolated struct LibrarySnapshots: Sendable {
    let directory: URL
    /// Clear Cache's generation, and the lock a migration and a purge share:
    /// a migration begun before a purge can never write after it.
    let gate = Gate()

    final class Gate: Sendable {
        let generation = Mutex(0)
    }

    /// What `migrate(_:from:)` must still find for its writes to happen.
    nonisolated var generation: Int { gate.generation.withLock { $0 } }

    static let standard = LibrarySnapshots(directory: ThumbnailLoader.defaultDirectory.appending(path: "library"))

    /// Every machine's `GalleryReplica` state: rows, cursor, validators.
    nonisolated var replicas: GalleryReplicaStore {
        GalleryReplicaStore(directory: directory.appending(path: "replicas"))
    }

    nonisolated func load(_ id: UUID) -> LibrarySnapshot? {
        guard let data = try? Data(contentsOf: file(id)) else { return nil }
        return try? MoldJSON.localDecoder.decode(LibrarySnapshot.self, from: data)
    }

    nonisolated func save(_ snapshot: LibrarySnapshot, for id: UUID) {
        guard let data = try? MoldJSON.localEncoder.encode(snapshot) else { return }
        try? FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        try? data.write(to: file(id), options: .atomic)
    }

    /// Once per machine, after an update: an older build's saved listing
    /// becomes the replica's starting state (validators kept, so an unchanged
    /// machine still answers 304) and the file keeps only its collections.
    nonisolated func migrate(_ id: UUID, from generation: Int) {
        gate.generation.withLock { current in
            guard current == generation,
                  let data = try? Data(contentsOf: file(id)),
                  let legacy = try? MoldJSON.localDecoder.decode(LibrarySnapshot.Legacy.self, from: data)
            else { return }
            // Only once the rows are safe in a replica file -- seeded now, or
            // already there -- may the listing leave this one.
            let seeded = replicas.seedIfAbsent(id, live: legacy.prints, trashed: legacy.trashed ?? [],
                                               liveETag: legacy.etag, trashETag: legacy.trashEtag)
            guard seeded || replicas.hasSaved(id) else { return }
            save(LibrarySnapshot(collections: legacy.collections), for: id)
        }
    }

    /// The machine was removed: its collections and its replica go.
    nonisolated func remove(_ id: UUID) {
        try? FileManager.default.removeItem(at: file(id))
        replicas.remove(id)
    }

    /// Files a machine removed in an earlier launch left behind.
    nonisolated func prune(keeping ids: Set<UUID>) {
        replicas.prune(keeping: ids)
        guard let files = try? FileManager.default.contentsOfDirectory(at: directory, includingPropertiesForKeys: nil)
        else { return }
        for file in files where file.pathExtension == "json" {
            guard let id = UUID(uuidString: file.deletingPathExtension().lastPathComponent),
                  !ids.contains(id) else { continue }
            try? FileManager.default.removeItem(at: file)
        }
    }

    nonisolated func purge() {
        gate.generation.withLock { generation in
            generation += 1
            try? FileManager.default.removeItem(at: directory)
        }
    }

    nonisolated var diskBytes: Int64 {
        guard let files = FileManager.default.enumerator(at: directory, includingPropertiesForKeys: [.fileSizeKey],
                                                         options: [.skipsHiddenFiles]) else { return 0 }
        var total: Int64 = 0
        for case let file as URL in files {
            total += Int64((try? file.resourceValues(forKeys: [.fileSizeKey]).fileSize) ?? 0)
        }
        return total
    }

    nonisolated private func file(_ id: UUID) -> URL { directory.appending(path: "\(id.uuidString).json") }
}
