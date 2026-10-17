import Foundation
import MoldClientTesting
import Testing

@testable import MoldClient

/// A replica's saved state shown before its machine is asked -- the phone's
/// offline Library -- and the one-time seed from a listing an app saved on
/// its own before it had replicas.
struct GalleryReplicaRestoreTests {
    private func directory() -> URL {
        FileManager.default.temporaryDirectory.appending(path: "replica-restore-\(UUID().uuidString)")
    }

    @Test func aSavedStateIsHandedOverOnceAsAResetBeforeAnyPull() async throws {
        let directory = directory()
        defer { try? FileManager.default.removeItem(at: directory) }
        let store = GalleryReplicaStore(directory: directory)
        let host = UUID()
        let feed = FakeGalleryFeed(rows: GalleryReplicaTests.rows(["a", "b"]))
        let first = GalleryReplica(hostID: host, store: store)
        try await first.pull(using: GalleryReplicaTests.feedBackend(feed))
        await first.flush()

        let heard = Recorder()
        let relaunched = GalleryReplica(hostID: host, store: store) { await heard.add($0) }
        let restored = try #require(await relaunched.restoreSaved())
        #expect(restored.reset)
        #expect(restored.scope == .all)
        #expect(restored.upserted.map(\.filename) == ["a", "b"])
        #expect(await heard.all == [restored], "the sink hears it like any other delta")
        #expect(await relaunched.restoreSaved() == nil, "handed over once")
        // Resumes from the saved cursor: nothing new is no delta at all.
        let backend = GalleryReplicaTests.feedBackend(feed)
        #expect(try await relaunched.pull(using: backend) == nil)
        #expect(feed.requestedCursors.last == "1:2")
    }

    @Test func nothingSavedRestoresNothingAndTheFirstPullIsStillAReset() async throws {
        let directory = directory()
        defer { try? FileManager.default.removeItem(at: directory) }
        let replica = GalleryReplica(hostID: UUID(), store: GalleryReplicaStore(directory: directory))
        #expect(await replica.restoreSaved() == nil)
        let feed = FakeGalleryFeed(rows: GalleryReplicaTests.rows(["a"]))
        let delta = try #require(try await replica.pull(using: GalleryReplicaTests.feedBackend(feed)))
        #expect(delta.reset)
    }

    @Test func aReplicaThatAlreadyDeliveredOrWasDiscardedRestoresNothing() async throws {
        let directory = directory()
        defer { try? FileManager.default.removeItem(at: directory) }
        let store = GalleryReplicaStore(directory: directory)
        let feed = FakeGalleryFeed(rows: GalleryReplicaTests.rows(["a"]))
        let replica = GalleryReplica(hostID: UUID(), store: store)
        try await replica.pull(using: GalleryReplicaTests.feedBackend(feed))
        await replica.flush()
        #expect(await replica.restoreSaved() == nil)
        await replica.discard()
        #expect(await replica.restoreSaved() == nil)
    }

    @Test func flushIfDirtyWritesOnlyAPendingSave() async throws {
        let directory = directory()
        defer { try? FileManager.default.removeItem(at: directory) }
        let store = GalleryReplicaStore(directory: directory)
        let host = UUID()
        let replica = GalleryReplica(hostID: host, store: store, saveDelay: .seconds(60))
        await replica.flushIfDirty()
        #expect(!store.hasSaved(host), "nothing pending, nothing written")
        let feed = FakeGalleryFeed(rows: GalleryReplicaTests.rows(["a"]))
        try await replica.pull(using: GalleryReplicaTests.feedBackend(feed))
        await replica.flushIfDirty()
        #expect(store.load(host)?.rows.map(\.print.filename) == ["a"])
        try FileManager.default.removeItem(at: directory)
        await replica.flushIfDirty()
        #expect(!store.hasSaved(host), "already written: not written again")
    }

    @Test func aRetiredReplicaNeverWritesOverItsSuccessor() async throws {
        let directory = directory()
        defer { try? FileManager.default.removeItem(at: directory) }
        let store = GalleryReplicaStore(directory: directory)
        let host = UUID()
        let old = GalleryReplica(hostID: host, store: store, saveDelay: .milliseconds(20))
        try await old.pull(using: GalleryReplicaTests.feedBackend(FakeGalleryFeed(rows: GalleryReplicaTests.rows(["old"]))))
        old.retire()
        let successor = GalleryReplica(hostID: host, store: store)
        try await successor.pull(using: GalleryReplicaTests.feedBackend(
            FakeGalleryFeed(instanceID: "instance-2", rows: GalleryReplicaTests.rows(["new"]))))
        await successor.flush()
        await old.flush()
        await old.flushIfDirty()
        #expect(store.load(host)?.rows.map(\.print.filename) == ["new"])
    }

    @Test func aSeedBecomesTheSavedStateAndKeepsTheTrashListingsWord() async throws {
        let directory = directory()
        defer { try? FileManager.default.removeItem(at: directory) }
        let store = GalleryReplicaStore(directory: directory)
        let host = UUID()
        // An older machine's trash row need not carry `trashed_at`.
        let live = GalleryReplicaTests.rows(["a"]), trashed = GalleryReplicaTests.rows(["old"])
        #expect(store.seedIfAbsent(host, live: live, trashed: trashed, liveETag: "e1", trashETag: "t1"))
        #expect(!store.seedIfAbsent(host, live: [], trashed: [], liveETag: nil, trashETag: nil),
                "never overwrites a saved replica")
        let replica = GalleryReplica(hostID: host, store: store)
        let restored = try #require(await replica.restoreSaved())
        #expect(restored.live.map(\.filename) == ["a"])
        #expect(restored.trashed.map(\.filename) == ["old"])
        // The seeded validators still describe what the machine holds.
        let backend = FakeBackend()
        backend.stub("gallery(etag:)", returning: Fetched<[GalleryPrint]>.notModified)
        backend.stub("trashedPrints(etag:)", returning: Fetched<[GalleryPrint]>.notModified)
        #expect(try await replica.pull(.legacy(.all), using: backend) == nil)
        let sent = backend.calls.filter { $0.route == "gallery(etag:)" }.compactMap { $0.arguments.first as? String }
        #expect(sent == ["e1"])
    }
}

private actor Recorder {
    private(set) var all: [ReplicaDelta] = []
    func add(_ delta: ReplicaDelta) { all.append(delta) }
}
