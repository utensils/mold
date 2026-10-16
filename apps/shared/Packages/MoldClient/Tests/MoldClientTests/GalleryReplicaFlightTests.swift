import Foundation
import MoldClientTesting
import Testing

@testable import MoldClient

/// Holds the first feed request until a test lets it go.
actor FeedGate {
    private var open = false
    private var waiters: [CheckedContinuation<Void, Never>] = []
    private(set) var entered = 0

    func pass() async {
        entered += 1
        guard !open else { return }
        await withCheckedContinuation { waiters.append($0) }
    }

    func release() {
        open = true
        waiters.forEach { $0.resume() }
        waiters = []
    }
}

struct GalleryReplicaFlightTests {
    @Test func callersDuringAPullShareExactlyOneFollowUp() async throws {
        let feed = FakeGalleryFeed(rows: GalleryReplicaTests.rows(["a"]))
        let gate = FeedGate()
        let backend = FakeBackend()
        backend.stub("galleryChanges(since:limit:)") { args in
            // Answered as the request arrives, then held: what lands while it
            // is held is the follow-up's to read.
            let page = try feed.page(since: args[0] as? String, limit: 2_000)
            await gate.pass()
            return page
        }
        let replica = GalleryReplica(hostID: UUID())
        let first = Task { try await replica.pull(using: backend) }
        while await gate.entered == 0 { await Task.yield() }
        feed.upsert(PrintFixtures.print("b"))
        let tickles = (0..<3).map { _ in Task { try await replica.pull(using: backend) } }
        while await replica.followers < 3 { await Task.yield() }
        await gate.release()
        let initial = try await first.value
        let followUps = try await tickles.asyncMap { try await $0.value }
        #expect(initial?.reset == true)
        #expect(await replica.pullCount == 2)
        #expect(backend.count("galleryChanges(since:limit:)") == 2)
        // Every tickle saw the same follow-up, which started after them.
        #expect(Set(followUps.map { $0?.revision }) == [2])
        #expect(followUps.first??.upserted.map(\.filename) == ["b"])
    }

    @Test func aSlowTrashListingDoesNotHoldTheTimelineBehindIt() async throws {
        let feed = FakeGalleryFeed(rows: GalleryReplicaTests.rows(["a"]))
        let gate = FeedGate()
        let backend = FakeBackend()
        backend.stub("gallery(etag:)") { _ in Fetched.fresh(feed.listing(), etag: nil) }
        backend.stub("trashedPrints(etag:)") { _ in
            await gate.pass()
            return Fetched.fresh(feed.trashListing(), etag: nil)
        }
        let replica = GalleryReplica(hostID: UUID())
        let trash = Task { try await replica.pull(.legacy(.trash), using: backend) }
        while await gate.entered == 0 { await Task.yield() }
        let live = try await replica.pull(.legacy(.live), using: backend)
        #expect(live?.upserted.map(\.filename) == ["a"])
        await gate.release()
        _ = try await trash.value
    }

    @Test func aCancelledPullChangesNothing() async throws {
        let feed = FakeGalleryFeed(rows: GalleryReplicaTests.rows(["a"]))
        let gate = FeedGate()
        let backend = FakeBackend()
        backend.stub("galleryChanges(since:limit:)") { args in
            let page = try feed.page(since: args[0] as? String, limit: 2_000)
            await gate.pass()
            return page
        }
        let replica = GalleryReplica(hostID: UUID())
        let pull = Task { try await replica.pull(using: backend) }
        while await gate.entered == 0 { await Task.yield() }
        pull.cancel()
        await gate.release()
        await #expect(throws: CancellationError.self) { try await pull.value }
        #expect(await replica.rows.isEmpty)
        #expect(await replica.cursor == nil)
        // The next pull still starts from the beginning and hands it over.
        let next = try await replica.pull(using: backend)
        #expect(next?.reset == true)
        #expect(next?.upserted.map(\.filename) == ["a"])
    }

    @Test func theListingPathDiffsIntoTheSameDeltas() async throws {
        let feed = FakeGalleryFeed(rows: GalleryReplicaTests.rows(["a", "b", "c"]))
        let backend = GalleryReplicaTests.feedBackend(feed)
        let following = GalleryReplica(hostID: UUID())
        let listing = GalleryReplica(hostID: UUID())
        let first = try await following.pull(.feed, using: backend)
        let legacyFirst = try await listing.pull(.legacy(.all), using: backend)
        #expect(first?.upserted == legacyFirst?.upserted)

        feed.upsert(PrintFixtures.print("a", favorite: true))
        feed.upsert(PrintFixtures.print("b", trashedAt: 7, purgeAt: 9))
        feed.remove("c")
        feed.upsert(PrintFixtures.print("d"))
        let next = try #require(try await following.pull(.feed, using: backend))
        let legacyNext = try #require(try await listing.pull(.legacy(.all), using: backend))
        #expect(next.upserted == legacyNext.upserted)
        #expect(next.removed == legacyNext.removed)
        #expect(next.removed == ["c"])
        #expect(await following.rows == listing.rows)
        #expect(try await listing.pull(.legacy(.all), using: backend) == nil)
    }

    @Test func aLiveOnlyListingSpeaksOnlyForTheLiveTimeline() async throws {
        let feed = FakeGalleryFeed(rows: GalleryReplicaTests.rows(["a"]))
        let backend = GalleryReplicaTests.feedBackend(feed)
        let replica = GalleryReplica(hostID: UUID())
        let delta = try #require(try await replica.pull(.legacy(.live), using: backend))
        #expect(delta.scope == .live)
        #expect(backend.count("trashedPrints(etag:)") == 0)
        // The trash is handed over whole the first time it is read.
        let trash = try #require(try await replica.pull(.legacy(.trash), using: backend))
        #expect(trash.reset)
        #expect(trash.scope == .trash)
    }

    @Test func aFullPullRepairsEvenWhenNothingChanged() async throws {
        let feed = FakeGalleryFeed(rows: GalleryReplicaTests.rows(["a"]))
        let backend = GalleryReplicaTests.feedBackend(feed)
        let replica = GalleryReplica(hostID: UUID())
        try await replica.pull(using: backend)
        let repair = try #require(try await replica.pull(GalleryReplica.Pull(useFeed: true, full: true), using: backend))
        #expect(repair.reset)
        #expect(repair.upserted.map(\.filename) == ["a"])
    }

    @Test func aRelaunchResumesFromTheSavedCursor() async throws {
        let directory = FileManager.default.temporaryDirectory.appending(path: "replica-\(UUID().uuidString)")
        defer { try? FileManager.default.removeItem(at: directory) }
        let store = GalleryReplicaStore(directory: directory)
        let host = UUID()
        var row = PrintFixtures.print("a", prompt: "a cat")
        row.rawMetadataJSON = Data(#"{"prompt":"a cat","future_field":1}"#.utf8)
        let feed = FakeGalleryFeed(rows: [row])
        let backend = GalleryReplicaTests.feedBackend(feed)
        let before = GalleryReplica(hostID: host, store: store)
        try await before.pull(using: backend)
        await before.flush()

        let after = GalleryReplica(hostID: host, store: store)
        let delta = try #require(try await after.pull(using: backend))
        #expect(feed.requestedCursors.last == "1:1")
        // A new process is still handed its rows whole -- from the file.
        #expect(delta.reset)
        #expect(delta.upserted.first?.metadata.prompt == "a cat")
        #expect(delta.upserted.first?.rawMetadataJSON == row.rawMetadataJSON)
    }

    @Test func aSavedCursorForAnotherInstanceIsDiscarded() async throws {
        let directory = FileManager.default.temporaryDirectory.appending(path: "replica-\(UUID().uuidString)")
        defer { try? FileManager.default.removeItem(at: directory) }
        let store = GalleryReplicaStore(directory: directory)
        let host = UUID()
        let feed = FakeGalleryFeed(rows: GalleryReplicaTests.rows(["a", "b"]))
        let backend = GalleryReplicaTests.feedBackend(feed)
        let before = GalleryReplica(hostID: host, store: store)
        try await before.pull(using: backend)
        await before.flush()
        feed.reinstance("instance-2")
        feed.remove("b")

        let after = GalleryReplica(hostID: host, store: store)
        let delta = try #require(try await after.pull(using: backend))
        #expect(delta.reset)
        #expect(delta.upserted.map(\.filename) == ["a"])
        #expect(await after.instanceID == "instance-2")
    }
}

extension Array {
    func asyncMap<T>(_ transform: (Element) async throws -> T) async rethrows -> [T] {
        var result: [T] = []
        for element in self { result.append(try await transform(element)) }
        return result
    }
}
