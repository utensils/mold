import Foundation
import MoldClient
import Testing

@testable import Mold

@MainActor
struct NewCollectionTests {
    @Test func namingActionKeepsTheWholeMenuSelection() {
        let host = MoldHost(name: "test", baseURL: URL(string: "http://test")!)
        let hosts = HostStore(hosts: [host]) { FakeBackend(host: $0) }
        let library = LibraryStore(hosts: hosts)
        let entries = ["one.png", "two.png"].map {
            LibraryEntry(host: host, print: FakeFixtures.print($0))
        }
        var captured: [LibraryEntry] = []
        let actions = LibraryActions(hosts: hosts, library: library, newCollection: { captured = $0 })
        actions.perform(.newCollection, on: entries, scope: .all)
        #expect(captured.map(\.id) == entries.map(\.id))
    }

    @Test func failedCreationDoesNotOfferAnExistingShelfForFiling() async {
        let host = MoldHost(name: "test", baseURL: URL(string: "http://test")!)
        let backend = FakeBackend(host: host)
        let hosts = HostStore(hosts: [host]) { _ in backend }
        let library = LibraryStore(hosts: hosts)
        backend.collectionRows = [Collection(id: "old", name: "Drafts", slug: "drafts")]
        let shelf = await library.createShelf(named: "Drafts", on: host.id)
        #expect(shelf == nil)
    }

    @Test func creationUsesTheServersReturnedSlug() async throws {
        let host = MoldHost(name: "test", baseURL: URL(string: "http://test")!)
        let backend = FakeBackend(host: host)
        let hosts = HostStore(hosts: [host]) { _ in backend }
        let library = LibraryStore(hosts: hosts)
        backend.collectionCreateResponses["Drafts"] = Collection(id: "new", name: "Drafts", slug: "server-slug")
        let shelf = try #require(await library.createShelf(named: "Drafts", on: host.id))
        #expect(shelf.slug == "server-slug")
        #expect(shelf.hosts[host.id] == "new")
    }
}
