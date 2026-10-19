import Foundation

/// Durable originals downloaded to files, with caller-owned copies for playback
/// and export. Clearing/removing a host fences even a noncooperative download.
public actor OriginalMediaCache {
    private let store: DiskThumbnailStore
    private var generation: UInt64 = 0
    private var hostGenerations: [String: UInt64] = [:]
    private struct Flight {
        let id: UUID
        let task: Task<URL, Error>
        var waiters: Set<UUID>
    }
    private var flights: [DiskThumbnailStore.Key: Flight] = [:]

    public init(directory: URL, maxBytes: Int) {
        store = DiskThumbnailStore(directory: directory, maxItems: .max,
                                   maxBytes: maxBytes, maxItemBytes: .max)
    }

    /// The returned file belongs to the caller, including when storage is full.
    public func file(for key: DiskThumbnailStore.Key,
                     download: @escaping @Sendable () async throws -> URL) async throws -> URL {
        try Task.checkCancellation()
        let requestEpoch = generation, requestHostEpoch = hostGenerations[key.host, default: 0]
        if let file = try await store.file(for: key) {
            guard !Task.isCancelled, requestEpoch == generation,
                  requestHostEpoch == hostGenerations[key.host, default: 0] else {
                try? FileManager.default.removeItem(at: file)
                throw CancellationError()
            }
            return file
        }
        guard requestEpoch == generation, requestHostEpoch == hostGenerations[key.host, default: 0] else { throw CancellationError() }
        let token = UUID()
        var flight: Flight
        if let running = flights[key] { flight = running }
        else {
            let epoch = generation, hostEpoch = hostGenerations[key.host, default: 0]
            let task = Task {
                let source = try await download()
                do {
                    try Task.checkCancellation()
                    try await self.persist(source, key: key, epoch: epoch, hostEpoch: hostEpoch)
                    return source
                } catch {
                    try? FileManager.default.removeItem(at: source)
                    throw error
                }
            }
            flight = Flight(id: UUID(), task: task, waiters: [])
        }
        flight.waiters.insert(token)
        flights[key] = flight
        let id = flight.id, task = flight.task
        defer { release(key, id: id, token: token) }
        return try await withTaskCancellationHandler {
            let source = try await task.value
            try Task.checkCancellation()
            guard requestEpoch == generation, requestHostEpoch == hostGenerations[key.host, default: 0] else { throw CancellationError() }
            let copy = FileManager.default.temporaryDirectory.appending(path: "mold-media-\(UUID())")
                .appendingPathExtension((key.filename as NSString).pathExtension)
            do {
                try FileManager.default.copyItem(at: source, to: copy)
                try Task.checkCancellation()
                return copy
            } catch {
                try? FileManager.default.removeItem(at: copy)
                throw error
            }
        } onCancel: {
            Task { await self.release(key, id: id, token: token) }
        }
    }

    private func persist(_ source: URL, key: DiskThumbnailStore.Key, epoch: UInt64, hostEpoch: UInt64) async throws {
        guard epoch == generation, hostEpoch == hostGenerations[key.host, default: 0] else { throw CancellationError() }
        // A full/unwritable cache must not discard a successfully downloaded
        // caller-owned original; the viewer reports that it was not retained.
        _ = try? await store.storeFile(source, for: key)
        guard epoch == generation, hostEpoch == hostGenerations[key.host, default: 0] else { throw CancellationError() }
    }

    private func release(_ key: DiskThumbnailStore.Key, id: UUID, token: UUID) {
        guard var flight = flights[key], flight.id == id, flight.waiters.remove(token) != nil else { return }
        if flight.waiters.isEmpty {
            flights[key] = nil
            flight.task.cancel()
            Task {
                if let source = try? await flight.task.value { try? FileManager.default.removeItem(at: source) }
            }
        } else { flights[key] = flight }
    }

    private func retire(_ key: DiskThumbnailStore.Key) {
        guard let flight = flights.removeValue(forKey: key) else { return }
        flight.task.cancel()
        Task {
            if let source = try? await flight.task.value { try? FileManager.default.removeItem(at: source) }
        }
    }

    public func contains(_ key: DiskThumbnailStore.Key) async -> Bool { await store.contains(key) }
    public var totalBytes: Int { get async { await store.totalBytes } }
    public func setLimit(_ bytes: Int) async { await store.setLimits(maxItems: .max, maxBytes: bytes) }
    public func purge() async {
        generation &+= 1
        for key in Array(flights.keys) { retire(key) }
        await store.purge()
    }
    public func evict(host: String) async {
        hostGenerations[host, default: 0] &+= 1
        for key in Array(flights.keys) where key.host == host { retire(key) }
        await store.evict(host: host)
    }
}
