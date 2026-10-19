import Foundation
import Testing
@testable import MoldClient

struct OriginalMediaCacheTests {
    private func key(version: String = "v1") -> DiskThumbnailStore.Key {
        .init(host: "machine", filename: "movie.mp4", mediaVersion: version, size: 0)
    }
    private func root() -> URL { FileManager.default.temporaryDirectory.appending(path: "original-test-\(UUID())") }
    private func download() throws -> URL {
        let file = root()
        try Data([1, 2, 3]).write(to: file)
        return file
    }
    @Test func freshCacheExportsOfflineAndVersionChangesMiss() async throws {
        let directory = root()
        defer { try? FileManager.default.removeItem(at: directory) }
        let first = OriginalMediaCache(directory: directory, maxBytes: .max)
        let viewed = try await first.file(for: key()) { try download() }
        defer { try? FileManager.default.removeItem(at: viewed) }
        let relaunched = OriginalMediaCache(directory: directory, maxBytes: .max)
        let exported = try await relaunched.file(for: key()) { throw CancellationError() }
        defer { try? FileManager.default.removeItem(at: exported) }
        #expect(try Data(contentsOf: exported) == Data([1, 2, 3]))
        await #expect(throws: CancellationError.self) {
            try await relaunched.file(for: key(version: "v2")) { throw CancellationError() }
        }
    }
    @Test func purgeFencesACompletedLateDownload() async throws {
        let directory = root()
        defer { try? FileManager.default.removeItem(at: directory) }
        let cache = OriginalMediaCache(directory: directory, maxBytes: .max)
        let operation = Task {
            try await cache.file(for: key()) {
                try? await Task.sleep(for: .milliseconds(100))
                return try download()
            }
        }
        try await Task.sleep(for: .milliseconds(20))
        await cache.purge()
        _ = await operation.result
        #expect(await cache.totalBytes == 0)
    }
    @Test func finiteLimitLeavesCallerFileUsableWithoutClaimingOfflineStorage() async throws {
        let cache = OriginalMediaCache(directory: root(), maxBytes: 2)
        let file = try await cache.file(for: key()) { try download() }
        defer { try? FileManager.default.removeItem(at: file) }
        #expect(try Data(contentsOf: file) == Data([1, 2, 3]))
        #expect(!(await cache.contains(key())))
    }
    @Test func removingMachineFencesLateDownload() async throws {
        let cache = OriginalMediaCache(directory: root(), maxBytes: .max)
        let operation = Task {
            try await cache.file(for: key()) {
                try? await Task.sleep(for: .milliseconds(100))
                return try download()
            }
        }
        try await Task.sleep(for: .milliseconds(20))
        await cache.evict(host: "machine")
        _ = await operation.result
        #expect(await cache.totalBytes == 0)
    }

    private actor Gate {
        var arrivals = 0
        var continuation: CheckedContinuation<Void, Never>?
        func wait() async { arrivals += 1; await withCheckedContinuation { continuation = $0 } }
        func open() { continuation?.resume(); continuation = nil }
    }
    @Test func concurrentViewAndShareDownloadOnceAndOwnIndependentFiles() async throws {
        let cache = OriginalMediaCache(directory: root(), maxBytes: .max), gate = Gate()
        let first = Task { try await cache.file(for: key()) { await gate.wait(); return try download() } }
        while await gate.arrivals == 0 { await Task.yield() }
        let second = Task { try await cache.file(for: key()) { await gate.wait(); return try download() } }
        try await Task.sleep(for: .milliseconds(20))
        await gate.open()
        let one = try await first.value, two = try await second.value
        defer { try? FileManager.default.removeItem(at: two) }
        try FileManager.default.removeItem(at: one)
        #expect(try Data(contentsOf: two) == Data([1, 2, 3]))
        #expect(await gate.arrivals == 1)
    }
    @Test func cancelledDownloadCannotPersist() async throws {
        let cache = OriginalMediaCache(directory: root(), maxBytes: .max), gate = Gate()
        let operation = Task {
            try await cache.file(for: key()) { await gate.wait(); return try download() }
        }
        while await gate.arrivals == 0 { await Task.yield() }
        operation.cancel()
        await gate.open()
        _ = await operation.result
        #expect(await cache.totalBytes == 0)
    }

}
