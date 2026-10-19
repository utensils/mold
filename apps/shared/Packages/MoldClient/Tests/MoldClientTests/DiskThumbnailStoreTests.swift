import Foundation
import Testing

@testable import MoldClient

/// The phone's thumbnail cache: bounded by count, bytes and item size, oldest
/// read first out, and emptied per machine or per print on demand.
struct DiskThumbnailStoreTests {
    private func store(items: Int = 100, bytes: Int = 1 << 20, item: Int = 1 << 20) -> DiskThumbnailStore {
        DiskThumbnailStore(
            directory: FileManager.default.temporaryDirectory.appending(path: "thumbs-\(UUID().uuidString)"),
            maxItems: items, maxBytes: bytes, maxItemBytes: item)
    }

    private func key(_ file: String, host: String = "inst-1", version: String = "v1", size: Int = 256)
        -> DiskThumbnailStore.Key {
        .init(host: host, filename: file, mediaVersion: version, size: size)
    }

    @Test func aStoredThumbnailComesBack() async {
        let cache = store()
        #expect(await cache.store(Data([1, 2, 3]), for: key("a.png")))
        #expect(await cache.data(for: key("a.png")) == Data([1, 2, 3]))
    }

    @Test func theTotalIsWhatIsOnDiskAndPurgeEmptiesIt() async {
        let cache = store()
        await cache.store(Data(count: 10), for: key("a.png"))
        await cache.store(Data(count: 5), for: key("b.png"))
        #expect(await cache.totalBytes == 15)
        await cache.purge()
        #expect(await cache.totalBytes == 0)
    }

    /// Settings' limit applies at once: lowering it drops the least recently
    /// read until the rest fit.
    @Test func aLowerLimitTrimsTheOldestAtOnce() async throws {
        let cache = store(bytes: 1 << 20)
        await cache.store(Data(count: 10), for: key("old.png"))
        try await Task.sleep(for: .milliseconds(20))
        await cache.store(Data(count: 10), for: key("new.png"))
        await cache.setLimits(maxItems: 100, maxBytes: 15)
        #expect(await cache.data(for: key("old.png")) == nil)
        #expect(await cache.data(for: key("new.png")) != nil)
        #expect(await cache.totalBytes == 10)
    }

    /// Prefetching asks whether a thumbnail is already saved without reading it.
    @Test func containsAnswersWithoutReading() async {
        let cache = store()
        #expect(!(await cache.contains(key("a.png"))))
        await cache.store(Data([1]), for: key("a.png"))
        #expect(await cache.contains(key("a.png")))
    }

    /// Replacing a thumbnail counts its bytes once.
    @Test func replacingAThumbnailCountsItOnce() async {
        let cache = store()
        await cache.store(Data(count: 10), for: key("a.png"))
        await cache.store(Data(count: 4), for: key("a.png"))
        #expect(await cache.totalBytes == 4)
    }

    @Test func aNewMediaVersionIsANewThumbnail() async {
        let cache = store()
        await cache.store(Data([1]), for: key("a.png", version: "v1"))
        #expect(await cache.data(for: key("a.png", version: "v2")) == nil)
    }

    @Test func anOversizedThumbnailIsRefusedNotStored() async {
        let cache = store(item: 4)
        #expect(await !cache.store(Data(count: 5), for: key("big.png")))
        #expect(await cache.totals.count == 0)
    }

    @Test func theCountCapEvictsTheLeastRecentlyRead() async throws {
        let cache = store(items: 2)
        await cache.store(Data([1]), for: key("old.png"))
        try await Task.sleep(for: .milliseconds(20))
        await cache.store(Data([2]), for: key("kept.png"))
        try await Task.sleep(for: .milliseconds(20))
        _ = await cache.data(for: key("old.png"))  // read: now the most recent
        try await Task.sleep(for: .milliseconds(20))
        await cache.store(Data([3]), for: key("new.png"))
        #expect(await cache.data(for: key("kept.png")) == nil)
        #expect(await cache.data(for: key("old.png")) != nil)
        #expect(await cache.totals.count == 2)
    }

    @Test func theByteCapHolds() async {
        let cache = store(bytes: 10)
        for index in 0..<5 { await cache.store(Data(count: 4), for: key("\(index).png")) }
        #expect(await cache.totals.bytes <= 10)
    }

    @Test func evictionIsPerMachineAndPerPrint() async {
        let cache = store()
        await cache.store(Data([1]), for: key("a.png", host: "one"))
        await cache.store(Data([1]), for: key("a.png", host: "one", size: 512))
        await cache.store(Data([1]), for: key("b.png", host: "one"))
        await cache.store(Data([1]), for: key("a.png", host: "two"))
        await cache.evict(host: "one", filename: "a.png")
        #expect(await cache.data(for: key("a.png", host: "one")) == nil)
        #expect(await cache.data(for: key("a.png", host: "one", size: 512)) == nil)
        #expect(await cache.data(for: key("b.png", host: "one")) != nil)
        await cache.evict(host: "one")
        #expect(await cache.data(for: key("b.png", host: "one")) == nil)
        #expect(await cache.data(for: key("a.png", host: "two")) != nil)
    }

    @Test func itSurvivesARelaunch() async {
        let directory = FileManager.default.temporaryDirectory.appending(path: "thumbs-\(UUID().uuidString)")
        await DiskThumbnailStore(directory: directory).store(Data([9]), for: key("a.png"))
        #expect(await DiskThumbnailStore(directory: directory).data(for: key("a.png")) == Data([9]))
    }
}

extension DiskThumbnailStoreTests {
    @Test func missingBackingOriginalBecomesACacheMissAndClearsItsAccounting() async throws {
        let cache = store(), item = key("movie.mp4")
        let directory = await cache.directory
        defer { try? FileManager.default.removeItem(at: directory) }
        await cache.store(Data([1, 2, 3]), for: item)
        try FileManager.default.removeItem(at: directory.appending(path: DiskThumbnailStore.name(for: item)))
        let file = try await cache.file(for: item)
        #expect(file == nil)
        #expect(!(await cache.contains(item)))
        #expect(await cache.totalBytes == 0)
    }

    @Test func unreadableBackingOriginalBecomesACacheMiss() async throws {
        let cache = store(), item = key("private.mp4")
        let directory = await cache.directory
        defer { try? FileManager.default.removeItem(at: directory) }
        await cache.store(Data([1, 2, 3]), for: item)
        let backing = directory.appending(path: DiskThumbnailStore.name(for: item))
        try FileManager.default.setAttributes([.posixPermissions: 0o000], ofItemAtPath: backing.path)
        defer { try? FileManager.default.setAttributes([.posixPermissions: 0o600], ofItemAtPath: backing.path) }
        #expect(!FileManager.default.isReadableFile(atPath: backing.path))
        let file = try await cache.file(for: item)
        #expect(file == nil)
        #expect(!(await cache.contains(item)))
        #expect(await cache.totalBytes == 0)
    }

    @Test func unavailableSourceDoesNotPreventOriginalRedownload() async throws {
        let directory = FileManager.default.temporaryDirectory.appending(path: "original-recovery-\(UUID())")
        defer { try? FileManager.default.removeItem(at: directory) }
        let cache = OriginalMediaCache(directory: directory, maxBytes: 1 << 20)
        let item = key("movie.mp4")
        @Sendable func download(_ byte: UInt8) throws -> URL {
            let file = FileManager.default.temporaryDirectory.appending(path: "download-recovery-\(UUID())")
            try Data([byte]).write(to: file)
            return file
        }
        let first = try await cache.file(for: item) { try download(1) }
        defer { try? FileManager.default.removeItem(at: first) }
        try FileManager.default.removeItem(at: directory.appending(path: DiskThumbnailStore.name(for: item)))
        let recovered = try await cache.file(for: item) { try download(2) }
        defer { try? FileManager.default.removeItem(at: recovered) }
        #expect(try Data(contentsOf: recovered) == Data([2]))
        #expect(await cache.contains(item))
    }
}
