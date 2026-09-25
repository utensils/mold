import Foundation
import MoldClient
import Testing

@testable import Mold

/// A print saved to This Mac is ONE tile, and what is done to that tile is
/// done to every copy of it.
@MainActor
struct LibraryMergeStoreTests {
    private func bench() async -> (LibraryStore, FakeBackend, FakeBackend) {
        let local = MoldEngine.localHost(port: 1, apiKey: "k")!
        let workstation = MoldHost(name: "workstation", baseURL: URL(string: "http://workstation")!)
        let here = FakeBackend(host: local)
        let there = FakeBackend(host: workstation)
        here.prints = [FakeFixtures.print("cat.png")]
        there.prints = [FakeFixtures.print("cat.png"), FakeFixtures.print("dog.png")]
        let hosts = HostStore(hosts: [workstation, local]) { $0.id == local.id ? here : there }
        let library = LibraryStore(hosts: hosts)
        await library.refresh()
        return (library, here, there)
    }

    @Test func aSavedCopyIsOneTileLedByThisMac() async {
        let (library, _, _) = await bench()

        #expect(library.items.count == 2)
        let cat = library.items.first { $0.print.filename == "cat.png" }
        #expect(cat?.hostID == MoldEngine.localHostID)
        #expect(cat?.hostNames == ["This Mac", "workstation"])
    }

    @Test func trashingTheTileTrashesEveryCopy() async throws {
        let (library, here, there) = await bench()
        let cat = try #require(library.items.first { $0.print.filename == "cat.png" })

        await library.moveToTrash([cat])

        #expect(here.callCount("trash") == 1)
        #expect(there.callCount("trash") == 1)
        #expect(library.items.map(\.print.filename) == ["dog.png"])
    }

    @Test func aCopyIsStillFoundByItsOwnID() async {
        let (library, _, _) = await bench()
        let remote = library.items.flatMap(\.copies).first
        let id = PrintID(host: remote?.hostID ?? UUID(), filename: "cat.png")

        #expect(library.entry(id)?.hostName == "workstation")
        #expect(library.tile(containing: id)?.hostID == MoldEngine.localHostID)
    }
}
