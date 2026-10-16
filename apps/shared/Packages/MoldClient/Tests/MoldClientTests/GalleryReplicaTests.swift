import Foundation
import MoldClientTesting
import Testing

@testable import MoldClient

/// One machine's replica following its change log -- and, without one, the
/// full listings -- into the one delta shape every caller reads.
struct GalleryReplicaTests {
    static func feedBackend(_ feed: FakeGalleryFeed) -> FakeBackend {
        let backend = FakeBackend()
        feed.serve(backend)
        return backend
    }

    static func rows(_ names: [String], timestamp: UInt64 = 1_000) -> [GalleryPrint] {
        names.enumerated().map { PrintFixtures.print($1, timestamp: timestamp + UInt64($0)) }
    }

    @Test func aFirstPullPagesTheSnapshotAndHandsItOverWhole() async throws {
        let feed = FakeGalleryFeed(rows: Self.rows(["a", "b", "c", "d", "e"]))
        let replica = GalleryReplica(hostID: UUID(), pageLimit: 2)
        let delta = try #require(try await replica.pull(using: Self.feedBackend(feed)))
        #expect(delta.reset)
        #expect(delta.scope == .all)
        #expect(delta.upserted.map(\.filename) == ["a", "b", "c", "d", "e"])
        #expect(feed.requestedCursors == [nil, "1:2", "1:4"])
        #expect(await replica.cursor == "1:5")
        #expect(await replica.instanceID == "instance-1")
    }

    @Test func laterPullsCarryOnlyWhatChanged() async throws {
        let feed = FakeGalleryFeed(rows: Self.rows(["a", "b"]))
        let backend = Self.feedBackend(feed)
        let replica = GalleryReplica(hostID: UUID())
        try await replica.pull(using: backend)
        feed.upsert(PrintFixtures.print("a", favorite: true))
        feed.upsert(PrintFixtures.print("c"))
        let delta = try #require(try await replica.pull(using: backend))
        #expect(!delta.reset)
        #expect(delta.upserted.map(\.filename) == ["a", "c"])
        #expect(delta.upserted.first?.isFavorite == true)
        #expect(delta.removed.isEmpty)
        #expect(delta.revision == 2)
        #expect(feed.requestedCursors.last == "1:2")
    }

    @Test func aRemovedNameIsARemoval() async throws {
        let feed = FakeGalleryFeed(rows: Self.rows(["a", "b"]))
        let backend = Self.feedBackend(feed)
        let replica = GalleryReplica(hostID: UUID())
        try await replica.pull(using: backend)
        feed.remove("a")
        let delta = try #require(try await replica.pull(using: backend))
        #expect(delta.removed == ["a"])
        #expect(delta.upserted.isEmpty)
        #expect(await replica.rows["a"] == nil)
    }

    @Test func nothingChangedIsNil() async throws {
        let feed = FakeGalleryFeed(rows: Self.rows(["a"]))
        let backend = Self.feedBackend(feed)
        let replica = GalleryReplica(hostID: UUID())
        try await replica.pull(using: backend)
        #expect(try await replica.pull(using: backend) == nil)
        // A re-marked row that reads the same is not a change either.
        feed.upsert(Self.rows(["a"])[0])
        #expect(try await replica.pull(using: backend) == nil)
    }

    @Test func aServerResetIsAResetDelta() async throws {
        let feed = FakeGalleryFeed(rows: Self.rows(["a", "b"]))
        let backend = Self.feedBackend(feed)
        let replica = GalleryReplica(hostID: UUID())
        try await replica.pull(using: backend)
        feed.rotateEpoch()
        let delta = try #require(try await replica.pull(using: backend))
        #expect(delta.reset)
        #expect(delta.upserted.map(\.filename) == ["a", "b"])
    }

    @Test func anotherInstanceStartsOverFromASnapshot() async throws {
        let feed = FakeGalleryFeed(rows: Self.rows(["a", "b"]))
        let backend = Self.feedBackend(feed)
        let replica = GalleryReplica(hostID: UUID())
        try await replica.pull(using: backend)
        feed.reinstance("instance-2")
        feed.upsert(PrintFixtures.print("c"))
        let delta = try #require(try await replica.pull(using: backend))
        #expect(delta.reset)
        #expect(Set(delta.upserted.map(\.filename)) == ["a", "b", "c"])
        #expect(feed.requestedCursors.suffix(2) == ["1:2", nil])
        #expect(await replica.instanceID == "instance-2")
    }

    @Test func trashedRowsAreSplitFromLiveOnes() async throws {
        let feed = FakeGalleryFeed(rows: Self.rows(["a", "b"]))
        let backend = Self.feedBackend(feed)
        let replica = GalleryReplica(hostID: UUID())
        try await replica.pull(using: backend)
        feed.upsert(PrintFixtures.print("a", trashedAt: 5, purgeAt: 10))
        let delta = try #require(try await replica.pull(using: backend))
        #expect(delta.trashed.map(\.filename) == ["a"])
        #expect(delta.live.isEmpty)
    }

    @Test func aFeedThatIsNotReadyFallsBackToTheListing() async throws {
        let feed = FakeGalleryFeed(rows: Self.rows(["a"]))
        let backend = Self.feedBackend(feed)
        feed.setNotReady(true)
        let replica = GalleryReplica(hostID: UUID())
        let delta = try #require(try await replica.pull(using: backend))
        #expect(delta.reset)
        #expect(delta.upserted.map(\.filename) == ["a"])
        #expect(backend.count("gallery(etag:)") == 1)
        #expect(backend.count("trashedPrints(etag:)") == 1)
    }

    @Test func deltasReachTheSinkInRevisionOrder() async throws {
        let feed = FakeGalleryFeed(rows: Self.rows(["a"]))
        let backend = Self.feedBackend(feed)
        let replica = GalleryReplica(hostID: UUID())
        let seen = Seen()
        await replica.setSink { await seen.append($0.revision) }
        try await replica.pull(using: backend)
        feed.upsert(PrintFixtures.print("b"))
        try await replica.pull(using: backend)
        #expect(await seen.values == [1, 2])
    }
}

actor Seen {
    var values: [Int] = []
    func append(_ value: Int) { values.append(value) }
}
