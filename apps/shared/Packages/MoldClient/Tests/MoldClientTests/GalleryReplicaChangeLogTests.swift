import Foundation
import MoldClientTesting
import Testing

@testable import MoldClient

/// The bounded change log a Library mirror plans from: "which names changed
/// since my cursor", or nil when the log cannot honestly say.
struct GalleryReplicaChangeLogTests {
    @Test func namesChangedSinceACursorAreExactlyTheLaterDeltas() async throws {
        let feed = FakeGalleryFeed(rows: GalleryReplicaTests.rows(["a", "b", "c"]))
        let backend = GalleryReplicaTests.feedBackend(feed)
        let replica = GalleryReplica(hostID: UUID())
        try await replica.pull(using: backend)
        let cursor = await replica.mirrorView().cursor
        #expect(await replica.changedNames(since: cursor) == [])
        feed.upsert(PrintFixtures.print("a", favorite: true))
        feed.remove("b")
        try await replica.pull(using: backend)
        #expect(await replica.changedNames(since: cursor) == ["a", "b"])
        let later = await replica.mirrorView().cursor
        feed.upsert(PrintFixtures.print("d"))
        try await replica.pull(using: backend)
        #expect(await replica.changedNames(since: later) == ["d"])
        #expect(await replica.changedNames(since: cursor) == ["a", "b", "d"])
    }

    @Test func anUnchangedPullAddsNothingToTheLog() async throws {
        let feed = FakeGalleryFeed(rows: GalleryReplicaTests.rows(["a"]))
        let backend = GalleryReplicaTests.feedBackend(feed)
        let replica = GalleryReplica(hostID: UUID())
        try await replica.pull(using: backend)
        let cursor = await replica.mirrorView().cursor
        try await replica.pull(using: backend)
        #expect(await replica.mirrorView().cursor == cursor)
        #expect(await replica.changedNames(since: cursor) == [])
    }

    @Test func anotherEpochOrAnUnreachedRevisionCannotBeAnswered() async throws {
        let feed = FakeGalleryFeed(rows: GalleryReplicaTests.rows(["a"]))
        let backend = GalleryReplicaTests.feedBackend(feed)
        let replica = GalleryReplica(hostID: UUID())
        try await replica.pull(using: backend)
        let cursor = await replica.mirrorView().cursor
        // A relaunch is a new replica: its revisions mean something else.
        let relaunched = GalleryReplica(hostID: replica.hostID)
        try await relaunched.pull(using: backend)
        #expect(await relaunched.changedNames(since: cursor) == nil)
        #expect(await replica.changedNames(since: ReplicaCursor(epoch: replica.epoch, revision: 99)) == nil)
    }

    @Test func aTrimmedLogRefusesCursorsOlderThanItsFloor() async throws {
        let feed = FakeGalleryFeed(rows: GalleryReplicaTests.rows(["a"]))
        let backend = GalleryReplicaTests.feedBackend(feed)
        let replica = GalleryReplica(hostID: UUID(), changeLogLimit: 2)
        try await replica.pull(using: backend)
        let first = await replica.mirrorView().cursor
        for name in ["b", "c", "d"] {
            feed.upsert(PrintFixtures.print(name))
            try await replica.pull(using: backend)
        }
        #expect(await replica.changedNames(since: first) == nil)
        let recent = await replica.mirrorView().cursor
        feed.upsert(PrintFixtures.print("e"))
        try await replica.pull(using: backend)
        #expect(await replica.changedNames(since: recent) == ["e"])
    }

    @Test func aLegacyListingDiffFeedsTheSameLog() async throws {
        let feed = FakeGalleryFeed(rows: GalleryReplicaTests.rows(["a", "b"]))
        let backend = GalleryReplicaTests.feedBackend(feed)
        feed.setNotReady(true)
        let replica = GalleryReplica(hostID: UUID())
        try await replica.pull(using: backend)
        let cursor = await replica.mirrorView().cursor
        feed.remove("b")
        try await replica.pull(using: backend)
        #expect(backend.count("gallery(etag:)") == 2)
        #expect(await replica.changedNames(since: cursor) == ["b"])
        let view = await replica.mirrorView()
        #expect(view.live("a") != nil)
        #expect(view.live("b") == nil)
    }
}
