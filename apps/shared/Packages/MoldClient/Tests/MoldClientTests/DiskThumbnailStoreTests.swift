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
