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

    /// Application Support's own folder for them (the throwaway one under
    /// native UAT, exactly as `SecretStore` resolves it).
    public static func standard() -> GalleryReplicaStore {
        GalleryReplicaStore(directory: SecretStore.applicationSupport().appending(path: "GalleryReplicas"))
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
    }

    /// A row plus the original recipe JSON `GalleryPrint` does not encode,
    /// which a cross-machine import needs byte for byte.
    struct Row: Codable {
        var print: GalleryPrint
        var rawMetadata: Data?
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
            print.rawMetadataJSON = row.rawMetadata
            restored[print.filename] = print
        }
        rows = restored
        cursor = snapshot.cursor
        instanceID = snapshot.instanceID
        liveETag = snapshot.liveETag
        trashETag = snapshot.trashETag
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

    /// Forgets everything, on disk too: the machine was removed.
    public func discard() {
        saveTask?.cancel()
        saveTask = nil
        rows = [:]
        cursor = nil
        instanceID = nil
        liveETag = nil
        trashETag = nil
        deliveredScopes = []
        loaded = true
        store?.remove(hostID)
    }

    func persistNow() {
        saveTask = nil
        guard let store, loaded else { return }
        let snapshot = GalleryReplicaStore.Snapshot(
            hostID: hostID, instanceID: instanceID, cursor: cursor, liveETag: liveETag, trashETag: trashETag,
            rows: rows.values.sorted { $0.filename < $1.filename }
                .map { GalleryReplicaStore.Row(print: $0, rawMetadata: $0.rawMetadataJSON) })
        try? store.save(snapshot)
    }
}
