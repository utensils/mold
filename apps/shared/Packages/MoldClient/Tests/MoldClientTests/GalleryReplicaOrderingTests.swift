import Foundation
import MoldClientTesting
import Testing

@testable import MoldClient

/// What can never happen however pulls interleave, are cancelled, or are
/// overtaken by a consumer's own splices -- and what a removed machine leaves
/// on disk.
struct GalleryReplicaOrderingTests {
    @Test(arguments: [true, false])
    func bothSnapshotReadingsDeliverTheSame(snapshotsSayReset: Bool) async throws {
        let feed = FakeGalleryFeed(rows: GalleryReplicaTests.rows(["a", "b"]), snapshotsSayReset: snapshotsSayReset)
        let backend = GalleryReplicaTests.feedBackend(feed)
        let replica = GalleryReplica(hostID: UUID(), pageLimit: 1)
        let first = try #require(try await replica.pull(using: backend))
        #expect(first.reset)
        #expect(first.upserted.map(\.filename) == ["a", "b"])
        feed.upsert(PrintFixtures.print("c"))
        let next = try #require(try await replica.pull(using: backend))
        #expect(!next.reset)
        #expect(next.upserted.map(\.filename) == ["c"])
    }

    @Test func aCancelledMultiPageCatchUpAppliesNoPage() async throws {
        let feed = FakeGalleryFeed(rows: GalleryReplicaTests.rows(["a"]))
        let gate = FeedGate()
        let backend = FakeBackend()
        let calls = Counter()
        backend.stub("galleryChanges(since:limit:)") { args in
            let page = try feed.page(since: args[0] as? String, limit: 1)
            // Hold the SECOND page of the catch-up (the first pull is call 1).
            if await calls.next() == 3 { await gate.pass() }
            return page
        }
        let replica = GalleryReplica(hostID: UUID(), pageLimit: 1)
        try await replica.pull(using: backend)
        let cursor = await replica.cursor
        feed.upsert(PrintFixtures.print("b"))
        feed.upsert(PrintFixtures.print("c"))
        let pull = Task { try await replica.pull(using: backend) }
        while await gate.entered == 0 { await Task.yield() }
        pull.cancel()
        await gate.release()
        _ = try? await pull.value
        #expect(await replica.rows.keys.sorted() == ["a"])
        #expect(await replica.cursor == cursor)
    }

    @Test func aCancelledFeedPullDoesNotUndoAnotherLanesAnswer() async throws {
        let feed = FakeGalleryFeed(rows: GalleryReplicaTests.rows(["a"]))
        let gate = FeedGate()
        let backend = FakeBackend()
        backend.stub("galleryChanges(since:limit:)") { args in
            let page = try feed.page(since: args[0] as? String, limit: 2_000)
            await gate.pass()
            return page
        }
        backend.stub("gallery(etag:)") { _ in Fetched.fresh(feed.listing(), etag: nil) }
        let replica = GalleryReplica(hostID: UUID())
        let feedPull = Task { try await replica.pull(using: backend) }
        while await gate.entered == 0 { await Task.yield() }
        feed.upsert(PrintFixtures.print("b"))
        try await replica.pull(.legacy(.live), using: backend)
        feedPull.cancel()
        await gate.release()
        _ = try? await feedPull.value
        #expect(await replica.rows.keys.sorted() == ["a", "b"])
    }

    @Test func anOlderListingAnswerLandingLastIsDropped() async throws {
        let listings = Listings(live: [PrintFixtures.print("old")])
        let gate = FeedGate()
        let backend = FakeBackend()
        let calls = Counter()
        backend.stub("gallery(etag:)") { _ in
            let answer = listings.live
            if await calls.next() == 1 { await gate.pass() }
            return Fetched.fresh(answer, etag: nil)
        }
        backend.stub("trashedPrints(etag:)") { _ in Fetched<[GalleryPrint]>.fresh([], etag: nil) }
        let replica = GalleryReplica(hostID: UUID())
        let slow = Task { try await replica.pull(.legacy(.live), using: backend) }
        while await gate.entered == 0 { await Task.yield() }
        listings.live = [PrintFixtures.print("new")]
        try await replica.pull(.legacy(.all), using: backend)
        await gate.release()
        _ = try await slow.value
        #expect(await replica.rows.keys.sorted() == ["new"])
    }

    @Test func aForgottenNameIsEmittedAsItIsNow() async throws {
        let feed = FakeGalleryFeed(rows: GalleryReplicaTests.rows(["a", "b"]))
        let backend = GalleryReplicaTests.feedBackend(feed)
        let replica = GalleryReplica(hostID: UUID())
        try await replica.pull(.legacy(.all), using: backend)
        // The consumer spliced "a" away and invented "ghost"; the machine
        // still has "a" and never had "ghost".
        replica.forget(["a", "ghost"])
        let repair = try #require(try await replica.pull(.legacy(.live), using: backend))
        #expect(repair.upserted.map(\.filename) == ["a"])
        #expect(repair.removed == ["ghost"])
    }

    @Test func aDiscardedReplicaNeverWritesItsFileAgain() async throws {
        let directory = FileManager.default.temporaryDirectory.appending(path: "replica-\(UUID().uuidString)")
        defer { try? FileManager.default.removeItem(at: directory) }
        let store = GalleryReplicaStore(directory: directory)
        let host = UUID()
        let feed = FakeGalleryFeed(rows: GalleryReplicaTests.rows(["a"]))
        let replica = GalleryReplica(hostID: host, store: store, saveDelay: .milliseconds(1))
        try await replica.pull(using: GalleryReplicaTests.feedBackend(feed))
        await replica.discard()
        await replica.flush()
        try await Task.sleep(for: .milliseconds(20))
        #expect(store.load(host) == nil)
    }

    @Test func pruningRemovesEveryOtherMachinesFile() async throws {
        let directory = FileManager.default.temporaryDirectory.appending(path: "replica-\(UUID().uuidString)")
        defer { try? FileManager.default.removeItem(at: directory) }
        let store = GalleryReplicaStore(directory: directory)
        let kept = UUID(), gone = UUID()
        let feed = FakeGalleryFeed(rows: GalleryReplicaTests.rows(["a"]))
        for host in [kept, gone] {
            let replica = GalleryReplica(hostID: host, store: store)
            try await replica.pull(using: GalleryReplicaTests.feedBackend(feed))
            await replica.flush()
        }
        store.prune(keeping: [kept])
        #expect(store.load(kept) != nil)
        #expect(store.load(gone) == nil)
    }
}

actor Counter {
    private var value = 0
    func next() -> Int {
        value += 1
        return value
    }
}
