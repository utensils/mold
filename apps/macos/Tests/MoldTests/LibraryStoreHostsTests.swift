import Foundation
import MoldClient
import Testing

@testable import Mold

/// `LibraryStore` used to infer "every machine" from whichever per-host cache
/// dictionary a method trusted, rather than asking `HostStore` -- the one
/// object that actually knows. These pin the bugs that produced.
@MainActor
struct LibraryStoreHostsTests {
    private func host(_ name: String) -> MoldHost {
        MoldHost(name: name, baseURL: URL(string: "http://\(name)")!)
    }

    @Test func cancelledStartupCannotApplyLateResults() async {
        let machine = host("startup")
        let fake = FakeBackend(host: machine)
        fake.prints = [FakeFixtures.print("late.png")]
        fake.trashedRows = [FakeFixtures.print("trash.png")]
        fake.collectionRows = [Collection(id: "late", name: "Late", slug: "late", count: 1)]
        for route in ["gallery", "trashedPrints", "collections", "tags"] {
            fake.delays[route] = .seconds(10)
        }
        let hosts = HostStore(hosts: [machine]) { _ in fake }
        let library = LibraryStore(hosts: hosts)
        let load = Task { await library.reload() }
        await settle { fake.calls.contains("gallery") && fake.calls.contains("collections") }
        load.cancel()
        await load.value

        #expect(library.items.isEmpty)
        #expect(library.trashed.isEmpty)
        #expect(library.shelves.isEmpty)
        #expect(!library.isLoading)
    }

    @Test func collectionsLoadWhileTheImageAndTrashListingsAreStillWaiting() async {
        let machine = host("slow-images")
        let fake = FakeBackend(host: machine)
        fake.collectionRows = [Collection(id: "album", name: "Album", slug: "album", count: 3)]
        fake.delays["gallery"] = .seconds(10)
        fake.delays["trashedPrints"] = .seconds(10)
        let hosts = HostStore(hosts: [machine]) { _ in fake }
        let library = LibraryStore(hosts: hosts)
        let load = Task { await library.reload() }
        await settle { !library.shelves.isEmpty }
        #expect(library.shelves.first?.name == "Album")
        #expect(library.isLoading)
        load.cancel()
        await load.value
    }

    @Test(arguments: [false, true])
    func standaloneSidebarRefreshesDiscardResultsAfterMachineChanges(edit: Bool) async {
        let machine = host("old")
        let fake = FakeBackend(host: machine)
        fake.trashedRows = [FakeFixtures.print("old-trash.png")]
        fake.collectionRows = [Collection(id: "old", name: "Old", slug: "old", count: 1)]
        for route in ["trashedPrints", "collections", "tags"] {
            fake.delays[route] = .milliseconds(200)
        }
        let hosts = HostStore(hosts: [machine]) { _ in fake }
        let library = LibraryStore(hosts: hosts)
        async let trash: Void = library.refreshTrash()
        async let organization: Void = library.refreshOrganization()
        await settle { fake.calls.contains("collections") && fake.calls.contains("trashedPrints") }
        var updated = machine
        updated.baseURL = URL(string: "http://replacement")!
        hosts.hosts = edit ? [updated] : []
        _ = await (trash, organization)
        #expect(library.trashed.isEmpty)
        #expect(library.shelves.isEmpty)
        #expect(library.tags.perHost.isEmpty)
    }

    @Test func removingAMachineAlsoRemovesItsShelvesWithoutAnyCachedPrints() {
        let machine = host("empty-gallery")
        let hosts = HostStore(hosts: []) { FakeBackend(host: $0) }
        let library = LibraryStore(hosts: hosts)
        library.collectionsPerHost[machine.id] = [
            Collection(id: "empty", name: "Empty collection", slug: "empty", count: 0),
        ]
        library.prune(to: [])
        #expect(library.shelves.isEmpty)
    }

    /// **Fails without the `insert(_:on:)` fix**: it used to read the new
    /// row's host name off an EXISTING entry for that machine
    /// (`perHost[host]?.first?.hostName`), so a machine with nothing in
    /// `perHost` yet had no name to borrow and the row was silently dropped.
    @Test func aMachineWithNoPrintsStillTakesAnInsertedRow() {
        let empty = host("empty")
        let other = host("other")
        var fakes: [MoldHost.ID: FakeBackend] = [:]
        let hosts = HostStore(hosts: [empty, other]) { host in
            fakes[host.id] ?? { let f = FakeBackend(host: host); fakes[host.id] = f; return f }()
        }
        // Listening from `init`, so there is nothing to start.
        let library = LibraryStore(hosts: hosts)

        // `empty` has never reported a print -- `perHost[empty.id]` is absent,
        // not merely empty.
        #expect(library.perHost[empty.id] == nil)

        hosts.listeners.forEach {
            $0(empty.id, .gallery(.added(filename: "new.png", row: FakeFixtures.print("new.png"))))
        }

        #expect(library.perHost[empty.id]?.map(\.print.filename) == ["new.png"])
    }

    /// **Fails without the `renameTag`/`reloadTags` fix**: they used to loop
    /// `tagsPerHost.keys`, which names only machines that have already
    /// answered `tags()` once -- a machine that hasn't yet (or never got
    /// asked) was silently skipped by a rename meant to reach every machine.
    @Test func renamingATagReachesAMachineThatHasReportedNoTags() async throws {
        let known = host("known")
        let silent = host("silent")
        let fakeKnown = FakeBackend(host: known)
        let fakeSilent = FakeBackend(host: silent)
        let fakes: [MoldHost.ID: FakeBackend] = [known.id: fakeKnown, silent.id: fakeSilent]
        let hosts = HostStore(hosts: [known, silent]) { fakes[$0.id]! }
        let library = LibraryStore(hosts: hosts)
        // Only `known` has ever reported tags.
        library.tags.perHost[known.id] = [TagCount(name: "old", count: 1)]

        library.tags.rename("old", to: "new", in: library)
        try await waitUntil { fakeSilent.calls.contains("renameTag") }

        #expect(fakeSilent.calls.contains("renameTag"))
    }

    /// Pins the `repair`/`reread` merge into one `relist(_:)`: a resync must
    /// re-list the machine AND replay whatever edit was still queued for it,
    /// onto the freshly-read rows.
    @Test func aResyncReplaysWhatIsStillQueued() async throws {
        let machine = host("workstation")
        let fake = FakeBackend(host: machine)
        fake.prints = [FakeFixtures.print("a.png")]
        let hosts = HostStore(hosts: [machine]) { _ in fake }
        // Listening from `init`, so there is nothing to start.
        let library = LibraryStore(hosts: hosts)

        // Still on its way to the machine when the resync arrives.
        library.mutations.outbox.enqueue(PrintEdit(change: .favorite(true), targets: [machine.id: ["a.png"]]))

        hosts.listeners.forEach { $0(machine.id, .resyncRequired) }
        try await waitUntil {
            library.perHost[machine.id]?.first(where: { $0.print.filename == "a.png" }) != nil
        }

        let row = try #require(library.perHost[machine.id]?.first { $0.print.filename == "a.png" })
        #expect(row.print.isFavorite)
    }
}

/// Polls a condition instead of a fixed sleep, so the test is as fast as the
/// work actually is and does not flake under load.
func waitUntil(timeout: Duration = .seconds(2), _ condition: () -> Bool) async throws {
    let deadline = ContinuousClock.now + timeout
    while !condition() {
        guard ContinuousClock.now < deadline else {
            Issue.record("condition never became true")
            return
        }
        try await Task.sleep(for: .milliseconds(5))
    }
}
