import Foundation

/// Where `GalleryReplica`s keep their state between launches: one JSON file
/// per machine, so a relaunch resumes from the saved cursor instead of reading
/// every row again.
///
/// Written with the app's strategy-free local coders (`MoldJSON.localEncoder`)
/// and atomically, from the replica's own executor -- never the main actor.
/// The file names its server instance; a replica reading a cursor for another
/// instance starts over (see `GalleryReplica`).
public struct GalleryReplicaStore: Sendable {
    public let directory: URL

    public init(directory: URL) { self.directory = directory }

    /// In Caches: everything here is rebuilt by reading the machine again, so
    /// it has no business in a backup. The folder is the app's own (the
    /// throwaway one under native UAT, as `SecretStore` resolves it).
    public static func standard() -> GalleryReplicaStore {
        let fresh = SecretStore.applicationSupport().lastPathComponent == SecretStore.freshDirectoryName
        let caches = FileManager.default.urls(for: .cachesDirectory, in: .userDomainMask).first
            ?? URL.homeDirectory.appending(path: "Library/Caches")
        return GalleryReplicaStore(directory: caches
            .appending(path: fresh ? SecretStore.freshDirectoryName : SecretStore.directoryName)
            .appending(path: "GalleryReplicas"))
    }

    /// The saved state, versioned so a later format can refuse an older one.
    struct Snapshot: Codable {
        static let currentVersion = 1
        var version = currentVersion
        var hostID: UUID
        var instanceID: String?
        var cursor: String?
        var liveETag: String?
        var trashETag: String?
        var rows: [Row]
        /// Names the legacy trash listing put in Recently Deleted.
        var listedInTrash: [String]?
    }

    /// A row plus the original recipe JSON `GalleryPrint` does not encode,
    /// which a cross-machine import needs byte for byte.
    ///
    /// Where the raw recipe exists it is the ONLY copy stored -- the typed
    /// metadata is decoded from it on load, as from the wire -- so a 30K-row
    /// file does not hold every recipe twice.
    struct Row: Codable {
        var print: GalleryPrint
        var rawMetadata: Data?

        private enum CodingKeys: String, CodingKey { case print, rawMetadata }
        private static let blank = (try? MoldJSON.decoder.decode(OutputMetadata.self, from: Data("{}".utf8)))

        init(print: GalleryPrint, rawMetadata: Data?) {
            self.print = print
            self.rawMetadata = rawMetadata
        }

        init(from decoder: any Decoder) throws {
            let row = try decoder.container(keyedBy: CodingKeys.self)
            let stored = try row.decode(GalleryPrint.self, forKey: .print)
            rawMetadata = try row.decodeIfPresent(Data.self, forKey: .rawMetadata)
            if let rawMetadata, let metadata = try? MoldJSON.decoder.decode(OutputMetadata.self, from: rawMetadata) {
                print = stored.replacing(metadata: metadata, raw: rawMetadata)
            } else {
                print = stored
            }
        }

        func encode(to encoder: any Encoder) throws {
            var row = encoder.container(keyedBy: CodingKeys.self)
            if let rawMetadata, let blank = Self.blank {
                try row.encode(print.replacing(metadata: blank, raw: nil), forKey: .print)
                try row.encode(rawMetadata, forKey: .rawMetadata)
            } else {
                try row.encode(print, forKey: .print)
            }
        }
    }

    /// Deletes every saved replica whose machine is no longer in the list.
    public func prune(keeping hostIDs: Set<UUID>) {
        guard let files = try? FileManager.default.contentsOfDirectory(at: directory, includingPropertiesForKeys: nil)
        else { return }
        for file in files where file.pathExtension == "json" {
            guard let id = UUID(uuidString: file.deletingPathExtension().lastPathComponent),
                  !hostIDs.contains(id) else { continue }
            try? FileManager.default.removeItem(at: file)
        }
    }

    func load(_ hostID: UUID) -> Snapshot? {
        guard let data = try? Data(contentsOf: file(hostID)),
              let snapshot = try? MoldJSON.localDecoder.decode(Snapshot.self, from: data),
              snapshot.version == Snapshot.currentVersion, snapshot.hostID == hostID
        else { return nil }
        return snapshot
    }

    func save(_ snapshot: Snapshot) throws {
        let data = try MoldJSON.localEncoder.encode(snapshot)
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        try data.write(to: file(snapshot.hostID), options: .atomic)
    }

    public func remove(_ hostID: UUID) {
        try? FileManager.default.removeItem(at: file(hostID))
    }

    private func file(_ hostID: UUID) -> URL { directory.appending(path: "\(hostID.uuidString).json") }
}

// Loading and saving, on the replica's executor.
extension GalleryReplica {
    func ensureLoaded() {
        guard !loaded else { return }
        loaded = true
        guard let snapshot = store?.load(hostID) else { return }
        var restored: [String: GalleryPrint] = [:]
        restored.reserveCapacity(snapshot.rows.count)
        for row in snapshot.rows {
            var print = row.print
            if print.rawMetadataJSON == nil { print.rawMetadataJSON = row.rawMetadata }
            restored[print.filename] = print
        }
        rows = restored
        cursor = snapshot.cursor
        instanceID = snapshot.instanceID
        liveETag = snapshot.liveETag
        trashETag = snapshot.trashETag
        listedInTrash = Set(snapshot.listedInTrash ?? [])
    }

    /// Debounced: a burst of pulls writes the file once, with the latest state.
    func scheduleSave() {
        guard store != nil, saveTask == nil else { return }
        let delay = saveDelay
        saveTask = Task {
            try? await Task.sleep(for: delay)
            guard !Task.isCancelled else { return }
            self.persistNow()
        }
    }

    /// Writes now rather than after the debounce -- for quitting, and tests.
    public func flush() {
        saveTask?.cancel()
        persistNow()
    }

    /// Whether this replica was discarded (its machine removed).
    public var isDiscarded: Bool { discarded }

    /// Forgets everything, on disk too: the machine was removed.
    public func discard() {
        discarded = true
        saveTask?.cancel()
        saveTask = nil
        rows = [:]
        cursor = nil
        instanceID = nil
        liveETag = nil
        trashETag = nil
        listedInTrash = []
        deliveredScopes = []
        loaded = true
        store?.remove(hostID)
    }

    func persistNow() {
        saveTask = nil
        guard let store, loaded, !discarded else { return }
        let snapshot = GalleryReplicaStore.Snapshot(
            hostID: hostID, instanceID: instanceID, cursor: cursor, liveETag: liveETag, trashETag: trashETag,
            rows: rows.values.sorted { $0.filename < $1.filename }
                .map { GalleryReplicaStore.Row(print: $0, rawMetadata: $0.rawMetadataJSON) },
            listedInTrash: listedInTrash.sorted())
        try? store.save(snapshot)
    }
}

extension GalleryPrint {
    func replacing(metadata: OutputMetadata, raw: Data?) -> GalleryPrint {
        GalleryPrint(
            filename: filename, assets: assets, metadata: metadata, timestamp: timestamp, format: format,
            sizeBytes: sizeBytes, mediaVersion: mediaVersion, title: title, tags: tags, favorite: favorite,
            collections: collections, trashedAt: trashedAt, purgeAt: purgeAt,
            metadataSynthetic: metadataSynthetic, rawMetadataJSON: raw)
    }
}
