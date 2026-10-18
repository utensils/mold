import Foundation
import MoldClient
import MoldClientTesting
import Testing

@testable import MoldCompanion

/// The Library read through one `GalleryReplica` per machine: a change costs
/// the rows that changed, a quiet machine costs nothing on screen, and events
/// tickle the replica instead of re-reading the machine -- for a machine with
/// a change feed and for one with only its listings.
@MainActor
struct LibraryReplicaTests {
    private func row(_ name: String, at timestamp: Int = 1_790_000_000, seed: Int = 7, trashedAt: Int? = nil,
                     collections: [String]? = nil, favourite: Bool = false) throws -> GalleryPrint {
        var json = #"{"filename":"\#(name)","metadata":{"prompt":"an owl","seed":\#(seed),"model":"flux-dev:q4"},"#
            + #""timestamp":\#(timestamp),"size_bytes":\#(1000 + seed),"favorite":\#(favourite)"#
        if let trashedAt { json += #","trashed_at":\#(trashedAt)"# }
        if let collections {
            json += #","collections":[\#(collections.map { "\"\($0)\"" }.joined(separator: ","))]"#
        }
        return try MoldJSON.decoder.decode(GalleryPrint.self, from: Data((json + "}").utf8))
    }

    private func capabilities(feed: Bool, trash: Bool = true) throws -> Capabilities {
        let changes = feed ? #","changes_feed":{"protocol_version":1}"# : ""
        let json = #"{"gallery":{"organize":true,"bulk_mutations":true,"trash":{"enabled":\#(trash)}\#(changes)}}"#
        return try MoldJSON.decoder.decode(Capabilities.self, from: Data(json.utf8))
    }

    private func temp(_ name: String) -> URL {
        FileManager.default.temporaryDirectory.appending(path: "\(name)-\(UUID())")
    }

    /// Machines named `names`, each answered by its own fake, all up.
    private func machines(_ fakes: [(String, FakeBackend, Bool)], snapshots: LibrarySnapshots? = nil)
        async throws -> (LibraryStore, HostStore) {
        for (_, fake, feed) in fakes {
            fake.stub("status()", returning: try MoldJSON.decoder.decode(ServerStatus.self,
                from: Data(#"{"version":"0.32.0","busy":false,"uptime_secs":1}"#.utf8)))
            fake.stub("capabilities()", returning: try capabilities(feed: feed))
            fake.stub("models()", returning: [Model]())
            fake.stub("collections()", returning: [Collection]())
        }
        let byName = Dictionary(uniqueKeysWithValues: fakes.map { ($0.0, $0.1) })
        let hosts = HostStore(list: HostListFile(url: temp("hosts").appending(path: "h.json")),
                              credentials: HostStoreTests.MemoryCredentials(),
                              makeBackend: { byName[$0.name] ?? FakeBackend() })
        for (index, (name, _, _)) in fakes.enumerated() {
            try hosts.add(name: name, address: "10.0.0.\(index + 1)", apiKey: nil, makeDefault: index == 0)
        }
        await hosts.refreshAll()
        let library = LibraryStore(hosts: hosts, snapshots: snapshots ?? LibrarySnapshots(directory: temp("lib")))
        return (library, hosts)
    }

    private func feedMachine(_ rows: [GalleryPrint]) -> (FakeGalleryFeed, FakeBackend) {
        let feed = FakeGalleryFeed(rows: rows)
        let fake = FakeBackend()
        feed.serve(fake)
        return (feed, fake)
    }

    private func hear(_ event: MoldEvent, on id: MoldHost.ID, in hosts: HostStore) {
        for listener in hosts.listeners { listener(id, event) }
    }

    private func waitUntil(_ condition: () -> Bool) async throws {
        for _ in 0..<200 where !condition() { try await Task.sleep(for: .milliseconds(20)) }
        #expect(condition())
    }

    // MARK: - Deltas

    @Test func aFeedMachinesNewPrintArrivesAsADeltaFromTheCursor() async throws {
        let (feed, fake) = feedMachine([try row("a.png", seed: 1), try row("b.png", seed: 2)])
        let (library, hosts) = try await machines([("alpha", fake, true)])
        await library.reload()
        #expect(Set(library.pool.map(\.print.filename)) == ["a.png", "b.png"])
        let cursors = feed.requestedCursors.count
        feed.upsert(try row("c.png", at: 1_790_000_100, seed: 3))
        hear(.gallery(.added(filename: "c.png", row: nil)), on: hosts.hosts[0].id, in: hosts)
        try await waitUntil { library.pool.contains { $0.print.filename == "c.png" } }
        #expect(feed.requestedCursors.count == cursors + 1, "one page, from the saved cursor")
        #expect(feed.requestedCursors.last != nil)
        #expect(fake.count("gallery(etag:)") == 0, "a feed machine is never listed")
    }

    @Test(arguments: [true, false])
    func aPullThatChangesNothingChangesNothingOnScreen(feed: Bool) async throws {
        let (_, fake) = feedMachine([try row("a.png", seed: 1)])
        let (library, _) = try await machines([("alpha", fake, feed)])
        await library.reload()
        let revision = library.revision, pool = library.pool, access = library.sourceAccessRevision
        await library.reload()
        await library.reload(library.hosts.hosts[0].id)
        #expect(library.revision == revision)
        #expect(library.pool == pool)
        #expect(library.sourceAccessRevision == access, "the pool was not even reassigned")
    }

    @Test func burstsOfFramesShareOnePull() async throws {
        let (feed, fake) = feedMachine([try row("a.png", seed: 1)])
        let (library, hosts) = try await machines([("alpha", fake, true)])
        await library.reload()
        let before = fake.count("galleryChanges(since:limit:)")
        for index in 0..<5 {
            feed.upsert(try row("n\(index).png", at: 1_790_000_100 + index, seed: 10 + index))
            hear(.gallery(.added(filename: "n\(index).png", row: nil)), on: hosts.hosts[0].id, in: hosts)
        }
        try await waitUntil { library.pool.count == 6 }
        try await Task.sleep(for: .milliseconds(400))
        #expect(fake.count("galleryChanges(since:limit:)") == before + 1)
    }

    /// The foreground's reconcile answers the resync its own reconnect
    /// raised: one pull, not one for each.
    @Test func aReturnToTheForegroundPullsOnce() async throws {
        let (_, fake) = feedMachine([try row("a.png", seed: 1)])
        let (library, hosts) = try await machines([("alpha", fake, true)])
        hear(.resyncRequired, on: hosts.hosts[0].id, in: hosts)
        await library.reload()
        try await Task.sleep(for: .milliseconds(450))
        #expect(fake.count("galleryChanges(since:limit:)") == 1)
        #expect(library.pool.count == 1)
    }

    /// The trash LISTING says which prints are in Recently Deleted, not the
    /// row's own fields: an older machine's trash rows need not carry a date.
    @Test func trashIsPartitionedByWhereTheRowCameFrom() async throws {
        let legacy = FakeBackend()
        legacy.stub("gallery(etag:)", returning: Fetched<[GalleryPrint]>.fresh([try row("live.png", seed: 1)], etag: "e1"))
        legacy.stub("trashedPrints(etag:)", returning: Fetched<[GalleryPrint]>.fresh([try row("old.png", seed: 2)], etag: "t1"))
        let (_, feedFake) = feedMachine([try row("kept.png", seed: 3),
                                         try row("binned.png", seed: 4, trashedAt: 1_790_000_500)])
        let (library, _) = try await machines([("legacy", legacy, false), ("modern", feedFake, true)])
        await library.reload()
        #expect(Set(library.pool.map(\.print.filename)) == ["live.png", "kept.png"])
        #expect(Set(library.trashPool.map(\.print.filename)) == ["old.png", "binned.png"])
    }

    /// Incremental application never differs from merging every row again.
    @Test func randomDeltasMatchAFullMerge() async throws {
        let (alphaFeed, alpha) = feedMachine([])
        let (betaFeed, beta) = feedMachine([])
        let (library, hosts) = try await machines([("alpha", alpha, true), ("beta", beta, true)])
        await library.reload()
        var generator = SeededGenerator(seed: 42)
        for step in 0..<60 {
            for feed in [alphaFeed, betaFeed] where Bool.random(using: &generator) {
                let name = "p\(Int.random(in: 0..<8, using: &generator)).png"
                switch Int.random(in: 0..<4, using: &generator) {
                case 0: feed.remove(name)
                case 1: feed.upsert(try row(name, at: 1_790_000_000 + Int.random(in: 0..<3, using: &generator),
                                            seed: Int.random(in: 0..<3, using: &generator),
                                            trashedAt: 1_790_001_000 + step))
                default: feed.upsert(try row(name, at: 1_790_000_000 + Int.random(in: 0..<3, using: &generator),
                                             seed: Int.random(in: 0..<3, using: &generator),
                                             favourite: Bool.random(using: &generator)))
                }
            }
            for host in hosts.hosts { await library.reload(host.id) }
            for (host, feed) in zip(hosts.hosts, [alphaFeed, betaFeed]) {
                #expect(Set(library.live[host.id] ?? []) == Set(feed.listing()), "step \(step)")
                #expect(Set(library.trashed[host.id] ?? []) == Set(feed.trashListing()), "step \(step)")
            }
            var full = LibraryTimeline(order: .newestFirst)
            full.rebuild(hosts.hosts.flatMap { host in
                (library.live[host.id] ?? []).map { LibraryEntry(host: host, print: $0) }
            }, localHost: nil)
            #expect(library.pool == full.tiles, "step \(step)")
            var fullTrash = LibraryTimeline(order: .recentlyTrashedFirst)
            fullTrash.rebuild(hosts.hosts.flatMap { host in
                (library.trashed[host.id] ?? []).map { LibraryEntry(host: host, print: $0) }
            }, localHost: nil)
            #expect(library.trashPool == fullTrash.tiles, "step \(step)")
        }
    }

    // MARK: - Collections

    @Test func collectionsAreReadOnlyWhenSomethingSaysTheyChanged() async throws {
        let (feed, fake) = feedMachine([try row("a.png", seed: 1)])
        let (library, hosts) = try await machines([("alpha", fake, true)])
        let id = hosts.hosts[0].id
        await library.reload()
        let read = fake.count("collections()")
        #expect(read == 1)
        feed.upsert(try row("b.png", at: 1_790_000_100, seed: 2))
        await library.reload(id)
        #expect(fake.count("collections()") == read, "an unfiled print leaves the shelves alone")
        feed.upsert(try row("c.png", at: 1_790_000_200, seed: 3, collections: ["shelf"]))
        await library.reload(id)
        #expect(fake.count("collections()") == read + 1, "a filed print can change a shelf's count")
        hear(.gallery(.collectionsChanged), on: id, in: hosts)
        try await waitUntil { fake.count("collections()") == read + 2 }
    }

    // MARK: - Edits and removal

    @Test func aRefusedEditIsUndoneByAFullRead() async throws {
        let (_, fake) = feedMachine([try row("a.png", seed: 1)])
        let (library, hosts) = try await machines([("alpha", fake, true)])
        await library.reload()
        fake.stub("mutate(_:)", throwing: MoldClientError.http(status: 500, code: nil, message: "disk full"))
        library.outbox.maxAttempts = 1
        library.apply(.favorite(true), to: library.pool)
        #expect(library.pool.first?.isFavorite == true)
        try await waitUntil { !hosts.failures.isEmpty && library.pool.first?.isFavorite == false }
    }

    @Test func removingAMachineDropsItsRowsAndItsSavedFiles() async throws {
        let snapshots = LibrarySnapshots(directory: temp("lib"))
        let (_, alpha) = feedMachine([try row("a.png", seed: 1, collections: ["shelf"])])
        let (_, beta) = feedMachine([try row("b.png", seed: 2)])
        alpha.stub("collections()", returning: [Collection(id: "shelf", name: "Shelf", slug: "shelf")])
        let (library, hosts) = try await machines([("alpha", alpha, true), ("beta", beta, true)], snapshots: snapshots)
        await library.reload()
        await library.waitForSnapshotWrites()
        let gone = hosts.hosts[0].id
        #expect(snapshots.load(gone) != nil)
        #expect(await GalleryReplica(hostID: gone, store: snapshots.replicas).restoreSaved() != nil)
        var forgotten: [MoldHost.ID] = []
        library.forgot = { forgotten.append($0) }
        hosts.remove(gone)
        await library.reload()
        #expect(library.pool.map(\.print.filename) == ["b.png"])
        #expect(forgotten == [gone])
        #expect(snapshots.load(gone) == nil)
        // The discarded replica's own removal is asynchronous; the file is
        // already gone either way, and nothing may write it back.
        #expect(!snapshots.replicas.hasSaved(gone))
        await library.waitForSnapshotWrites()
        #expect(!snapshots.replicas.hasSaved(gone))
    }

    /// A launch that cannot read its machine list yet (prewarmed before
    /// first unlock) sees no machines. That must never cost the offline
    /// Library of the machines it could not read.
    @Test func anEmptyMachineListDeletesNoSavedLibrary() async throws {
        let snapshots = LibrarySnapshots(directory: temp("lib"))
        let saved = UUID()
        #expect(snapshots.replicas.seedIfAbsent(saved, live: [try row("a.png")], trashed: [],
                                                liveETag: "e1", trashETag: nil))
        snapshots.save(LibrarySnapshot(collections: [Collection(id: "c", name: "C", slug: "c")]), for: saved)
        let (library, _) = try await machines([], snapshots: snapshots)
        await library.restoreSaved()
        await library.reload()
        await library.waitForSnapshotWrites()
        #expect(snapshots.replicas.hasSaved(saved))
        #expect(snapshots.load(saved) != nil)
    }

    /// Clear Cache keeps the shelves of machines still answering on disk.
    @Test func clearingTheCacheKeepsAnsweringMachinesCollections() async throws {
        let snapshots = LibrarySnapshots(directory: temp("lib"))
        let (_, fake) = feedMachine([try row("a.png")])
        let (library, hosts) = try await machines([("alpha", fake, true)], snapshots: snapshots)
        // machines installs default routes; the scenario overrides that default.
        fake.stub("collections()", returning: [Collection(id: "c", name: "C", slug: "c")])
        await library.reload()
        await library.clearOfflineCache()
        await library.waitForSnapshotWrites()
        #expect(snapshots.load(hosts.hosts[0].id)?.collections?.map(\.id) == ["c"])
        #expect(library.pool.count == 1, "an answering machine's prints stay on screen")
    }
}

/// Deterministic randomness, so a failing step can be replayed.
struct SeededGenerator: RandomNumberGenerator {
    private var state: UInt64
    init(seed: UInt64) { state = seed }
    mutating func next() -> UInt64 {
        state &+= 0x9E37_79B9_7F4A_7C15
        var z = state
        z = (z ^ (z >> 30)) &* 0xBF58_476D_1CE4_E5B9
        z = (z ^ (z >> 27)) &* 0x94D0_49BB_1331_11EB
        return z ^ (z >> 31)
    }
}
