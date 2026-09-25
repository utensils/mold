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

    /// Renaming a merged print and undoing it puts back what EACH copy was
    /// called, not the lead's old name on both.
    @Test func undoingARenameRestoresEachCopysOwnTitle() async throws {
        let (library, here, there) = await bench()
        func titled(_ name: String) -> GalleryPrint {
            var print = GalleryPrint.Mutable(FakeFixtures.print("cat.png"))
            print.title = name
            return print.build()
        }
        here.prints = [titled("Kitty")]
        there.prints = [titled("Cat"), FakeFixtures.print("dog.png")]
        library.etags.removeAll()
        await library.refresh()
        let manager = UndoManager()
        manager.groupsByEvent = false
        library.undo.manager = manager
        let cat = try #require(library.items.first { $0.print.filename == "cat.png" })

        manager.beginUndoGrouping()
        library.setTitle("Tabby", on: cat)
        manager.endUndoGrouping()
        #expect(Set(library.items.flatMap(\.everyCopy).filter { $0.print.filename == "cat.png" }
            .map(\.print.title)) == ["Tabby"])

        manager.undo()

        let titles = library.items.flatMap(\.everyCopy).filter { $0.print.filename == "cat.png" }
            .reduce(into: [String: String]()) { $0[$1.hostName] = $1.print.title ?? "" }
        #expect(titles == ["This Mac": "Kitty", "workstation": "Cat"])
    }

    @Test func aCopyIsStillFoundByItsOwnID() async {
        let (library, _, _) = await bench()
        let remote = library.items.flatMap(\.copies).first
        let id = PrintID(host: remote?.hostID ?? UUID(), filename: "cat.png")

        #expect(library.entry(id)?.hostName == "workstation")
        #expect(library.tile(containing: id)?.hostID == MoldEngine.localHostID)
    }
}
