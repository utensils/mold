import Foundation
import MoldClient
import MoldClientTesting
import Testing
@testable import MoldCompanion

private actor QueuePreviewProbe {
    var active = 0
    var peak = 0
    func load() async throws -> Data {
        active += 1; peak = max(peak, active)
        defer { active -= 1 }
        try await Task.sleep(for: .milliseconds(30))
        return Data([1])
    }
}

@MainActor struct QueuePreviewBudgetTests {
    private func fixture() async throws -> (QueueStore, HostStore, FakeBackend, [QueueEntry]) {
        let (queue, hosts, fake) = try await QueueStoreTests.setUp()
        let rows = (0..<20).map { "{\"id\":\"j\($0)\",\"state\":\"held\"}" }.joined(separator: ",")
        let listing = try QueueStoreTests.decode(QueueListing.self, "{\"entries\":[\(rows)]}")
        fake.stub("queue()", returning: listing)
        fake.stub("queueInputs(id:)", returning: [QueueInput(index: 0, label: "Source", preview: true)])
        await queue.poll(hosts.hosts[0].id)
        return (queue, hosts, fake, listing.entries)
    }

    @Test func liveQueuePreviewsHaveAnAggregateByteBudget() async throws {
        let (queue, hosts, fake, rows) = try await fixture()
        let host = hosts.hosts[0].id
        fake.stub("queueInputThumbnail(id:index:)", returning: Data(repeating: 1, count: 2 * 1024 * 1024))
        for row in rows { await queue.loadSourceThumbnail(for: row, on: host) }
        let retained = rows.flatMap { queue.inputPreviews(for: $0, on: host) }.reduce(0) { $0 + ($1.bytes?.count ?? 0) }
        #expect(retained > 0)
        #expect(retained <= 32 * 1024 * 1024)
        #expect(queue.sourceThumbnail(for: rows.last!, on: host) != nil)
    }

    @Test func differentRowsShareFourPreviewReadLanes() async throws {
        let (queue, hosts, fake, rows) = try await fixture()
        let host = hosts.hosts[0].id
        let probe = QueuePreviewProbe()
        fake.stub("queueInputThumbnail(id:index:)") { _ in try await probe.load() }
        await withTaskGroup(of: Void.self) { group in
            for row in rows { group.addTask { await queue.loadSourceThumbnail(for: row, on: host) } }
        }
        #expect(await probe.peak <= 4)
        #expect(fake.count("queueInputThumbnail(id:index:)") == rows.count)
    }
}
