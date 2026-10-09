import Foundation
import MoldClient
import MoldClientTesting
import Testing

@testable import MoldCompanion

/// The Library's data: merged across machines, edited on every copy, trashed
/// according to what each machine can do -- against fakes, never the network.
@MainActor
struct LibraryStoreTests {
    private func print(_ file: String, favourite: Bool = false, size: Int = 1000) throws -> GalleryPrint {
        let json = #"{"filename":"\#(file)","metadata":{"prompt":"an owl","seed":7,"model":"flux-dev:q4"},"#
            + #""timestamp":1790000000,"size_bytes":\#(size),"favorite":\#(favourite)}"#
        return try MoldJSON.decoder.decode(GalleryPrint.self, from: Data(json.utf8))
    }

    private func capabilities(trash: Bool = true) throws -> Capabilities {
        let json = #"{"gallery":{"organize":true,"bulk_mutations":true,"trash":{"enabled":\#(trash)}}}"#
        return try MoldJSON.decoder.decode(Capabilities.self, from: Data(json.utf8))
    }

    private func status() throws -> ServerStatus {
        try MoldJSON.decoder.decode(ServerStatus.self, from: Data(#"{"version":"0.32.0","busy":false,"uptime_secs":1}"#.utf8))
    }

    /// Two machines, each with its own fake, both up.
    private func fleet(first: [GalleryPrint], second: [GalleryPrint], trash: Bool = true)
        async throws -> (LibraryStore, HostStore, FakeBackend, FakeBackend) {
        let one = FakeBackend(), two = FakeBackend()
        for (fake, prints) in [(one, first), (two, second)] {
            fake.stub("status()", returning: try status())
            fake.stub("capabilities()", returning: try capabilities(trash: trash))
            fake.stub("models()", returning: [Model]())
            fake.stub("gallery(etag:)", returning: Fetched<[GalleryPrint]>.fresh(prints, etag: "e1"))
            fake.stub("trashedPrints(etag:)", returning: Fetched<[GalleryPrint]>.fresh([], etag: "t1"))
            fake.stub("collections()", returning: [Collection]())
        }
        let file = HostListFile(url: FileManager.default.temporaryDirectory.appending(path: "h-\(UUID()).json"))
        let fakes: [String: FakeBackend] = ["alpha": one, "beta": two]
        let hosts = HostStore(list: file, credentials: HostStoreTests.MemoryCredentials(),
                              makeBackend: { fakes[$0.name] ?? FakeBackend() })
        try hosts.add(name: "alpha", address: "10.0.0.1", apiKey: nil, makeDefault: true)
        try hosts.add(name: "beta", address: "10.0.0.2", apiKey: nil, makeDefault: false)
        await hosts.refreshAll()
        let library = LibraryStore(hosts: hosts, snapshots: LibrarySnapshots(
            directory: FileManager.default.temporaryDirectory.appending(path: "lib-\(UUID())")))
        await library.reload()
        return (library, hosts, one, two)
    }

    @Test func theSamePrintOnTwoMachinesIsOneEntry() async throws {
        let (library, _, _, _) = try await fleet(first: [try print("a.png")], second: [try print("a.png")])
        #expect(library.pool.count == 1)
        #expect(library.pool.first?.everyCopy.count == 2)
    }

    @Test func aFavouriteReachesEveryCopyWithAnOperationID() async throws {
        let (library, _, one, two) = try await fleet(first: [try print("a.png")], second: [try print("a.png")])
        one.stub("mutate(_:)") { _ in () }
        two.stub("mutate(_:)") { _ in () }
        library.apply(.favorite(true), to: library.pool)
        #expect(library.pool.first?.print.isFavorite == true, "shown at once, before either machine answers")
        try await waitUntil { one.count("mutate(_:)") == 1 && two.count("mutate(_:)") == 1 }
        let sent = try #require(one.calls.last { $0.route == "mutate(_:)" }?.arguments.first as? GalleryBulkMutation)
        #expect(sent.favorite == true)
        #expect(sent.filenames == ["a.png"])
        #expect(!sent.operationId.isEmpty)
    }

    @Test func undoPutsTheChangeBack() async throws {
        let (library, _, one, two) = try await fleet(first: [try print("a.png")], second: [])
        one.stub("mutate(_:)") { _ in () }
        two.stub("mutate(_:)") { _ in () }
        library.apply(.favorite(true), to: library.pool)
        #expect(library.undoName == "Favourite")
        library.undo()
        #expect(library.pool.first?.print.isFavorite == false)
    }

    @Test func theSystemsUndoShakeOrCommandZPutsItBack() async throws {
        let (library, _, one, two) = try await fleet(first: [try print("a.png")], second: [])
        one.stub("mutate(_:)") { _ in () }
        two.stub("mutate(_:)") { _ in () }
        let undo = UndoManager()
        undo.groupsByEvent = false
        library.undoManager = undo
        undo.beginUndoGrouping()
        library.apply(.favorite(true), to: library.pool)
        undo.endUndoGrouping()
        #expect(undo.undoActionName == "Favourite")
        undo.undo()
        #expect(library.pool.first?.print.isFavorite == false)
        #expect(library.lastEdit == nil, "the Undo button has nothing left to put back")
    }

    @Test func machineFilteredDeletionPreservesTheOtherCopy() async throws {
        let (library, hosts, one, two) = try await fleet(first: [try print("a.png")], second: [try print("a.png")])
        let beta = try #require(hosts.hosts.first { $0.name == "beta" })
        var query = LibraryQuery()
        query.tokens = [.machine(id: beta.id, name: beta.name)]
        two.stub("trash(_:)") { _ in () }
        await library.trash(query.apply(to: library.pool))
        #expect(two.count("trash(_:)") == 1)
        #expect(one.count("trash(_:)") == 0)
    }

    @Test func permanentTrashDeletionUsesTheTrashOnlyAuthority() async throws {
        let (library, hosts, one, two) = try await fleet(first: [try print("a.png")], second: [try print("a.png")])
        two.stub("deleteTrashed(_:)") { _ in () }
        let beta = try #require(hosts.hosts.first { $0.name == "beta" })
        let entry = try #require(library.pool.first?.presented(onAnyOf: [beta.id]))
        await library.deleteImmediately([entry])
        #expect(two.count("deleteTrashed(_:)") == 1)
        #expect(one.count("deleteTrashed(_:)") == 0)
        #expect(two.count("deleteForever(_:)") == 0)
    }

    @Test func aMachineWithoutATrashDeletesForGood() async throws {
        let (library, _, one, _) = try await fleet(first: [try print("a.png")], second: [], trash: false)
        one.stub("deleteForever(_:)") { _ in () }
        await library.trash(library.pool)
        #expect(one.count("deleteForever(_:)") == 1)
        #expect(one.count("trash(_:)") == 0)
    }

    @Test func aMachineWithATrashMovesPrintsThere() async throws {
        let (library, _, one, _) = try await fleet(first: [try print("a.png")], second: [])
        one.stub("trash(_:)") { _ in () }
        await library.trash(library.pool)
        #expect(one.count("trash(_:)") == 1)
    }

    @Test func aRefusedChangeIsSaidAndTheGridIsReRead() async throws {
        let (library, hosts, one, _) = try await fleet(first: [try print("a.png")], second: [])
        one.stub("mutate(_:)", throwing: MoldClientError.http(status: 500, code: nil, message: "disk full"))
        library.outbox.maxAttempts = 1
        library.apply(.favorite(true), to: library.pool)
        try await waitUntil { !hosts.failures.isEmpty }
        #expect(hosts.failures.first?.text.contains("disk full") == true)
    }

    @Test(arguments: ["gallery(etag:)", "trashedPrints(etag:)", "collections()"])
    func aCancelledRefreshKeepsPrintsWithoutAFailureBanner(route: String) async throws {
        let (library, hosts, one, _) = try await fleet(first: [try print("a.png")], second: [])
        one.stub(route, throwing: CancellationError())
        await library.reload(hosts.hosts[0].id)
        #expect(library.pool.map(\.print.filename) == ["a.png"])
        #expect(hosts.failures.isEmpty)
        #expect(!library.isLoading)
    }

    @Test func aRealListingFailureStillShowsABanner() async throws {
        let (library, hosts, one, _) = try await fleet(first: [try print("a.png")], second: [])
        one.stub("gallery(etag:)", throwing: MoldClientError.http(status: 500, code: nil, message: "disk unavailable"))
        await library.reload(hosts.hosts[0].id)
        #expect(library.pool.map(\.print.filename) == ["a.png"])
        #expect(hosts.failures.first?.text.contains("disk unavailable") == true)
    }

    @Test func hidingACollectionUpdatesEveryMachineCopy() async throws {
        let (library, hosts, one, two) = try await fleet(first: [try print("a.png")], second: [])
        let first = Collection(id: "alpha-shelf", name: "Drafts", slug: "drafts", hidden: false)
        let second = Collection(id: "beta-shelf", name: "Drafts", slug: "drafts", hidden: false)
        one.stub("collections()", returning: [first])
        two.stub("collections()", returning: [second])
        await library.reload()
        let shelf = try #require(library.shelves.first)
        one.stub("updateCollection(id:change:)", returning: first)
        two.stub("updateCollection(id:change:)", returning: second)
        one.stub("collections()", returning: [Collection(id: first.id, name: first.name, slug: first.slug, hidden: true)])
        two.stub("collections()", returning: [Collection(id: second.id, name: second.name, slug: second.slug, hidden: true)])
        await library.setShelfHidden(shelf, hidden: true)
        for (fake, id) in [(one, first.id), (two, second.id)] {
            let sent = try #require(fake.calls.last { $0.route == "updateCollection(id:change:)" })
            #expect(sent.arguments.first as? String == id)
            #expect((sent.arguments.last as? CollectionChange)?.hidden == true)
        }
        #expect(library.hiddenCollectionIDs[hosts.hosts[0].id] == [first.id])
        #expect(library.hiddenCollectionIDs[hosts.hosts[1].id] == [second.id])
        #expect(library.shelves.first?.hidden == true)
    }

    private func waitUntil(_ condition: () -> Bool) async throws {
        for _ in 0..<200 where !condition() { try await Task.sleep(for: .milliseconds(20)) }
        #expect(condition())
    }
}
