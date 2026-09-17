import Foundation
import MoldClient
import Testing

@testable import Mold

/// The Edit menu's Undo, which belongs to the WINDOW and is shared with every
/// field editor in it.
///
/// **Fails today**: `MoldUndo.forget()` calls `removeAllActions()`
/// (`MoldUndo.swift:61-63`), so "Delete Tag Everywhere…" also throws away
/// whatever a focused text field had recorded -- and a refused edit leaves its
/// inverse on the stack, offering to undo something that never happened.
@MainActor
struct LibraryUndoTests {
    /// Captures what an undo actually invoked. A class, because the handler
    /// escapes.
    private final class Fired {
        var names: [String] = []
    }

    private func manager() -> UndoManager {
        let manager = UndoManager()
        // Explicit grouping, or a registration sits in a group that only
        // closes on the next pass of the run loop and `canUndo` is false.
        manager.groupsByEvent = false
        return manager
    }

    private func host(_ name: String) -> MoldHost {
        MoldHost(name: name, baseURL: URL(string: "http://\(name)")!)
    }

    @Test func forgettingOurEntriesLeavesEverybodyElsesAlone() {
        let manager = manager()
        let undo = MoldUndo()
        undo.manager = manager
        let fired = Fired()
        let fieldEditor = NSObject()

        manager.beginUndoGrouping()
        manager.registerUndo(withTarget: fieldEditor) { _ in fired.names.append("field") }
        undo.register("Rename Tag") { fired.names.append("ours") }
        manager.endUndoGrouping()

        undo.forget()

        #expect(manager.canUndo)
        manager.undo()
        #expect(fired.names == ["field"])
    }

    @Test func aRefusedEditLeavesNoUndoEntryBehind() async {
        let machine = host("plato")
        let fake = FakeBackend(host: machine)
        fake.prints = [FakeFixtures.print("star.png")]
        fake.refuses = ["mutate"]
        let hosts = HostStore(hosts: [machine]) { _ in fake }
        let library = LibraryStore(hosts: hosts)
        let manager = manager()
        library.undo.manager = manager
        library.perHost[machine.id] = [LibraryEntry(host: machine,
                                                    print: FakeFixtures.print("star.png"))]

        manager.beginUndoGrouping()
        library.setFavorite(true, on: library.perHost[machine.id] ?? [])
        manager.endUndoGrouping()
        #expect(manager.canUndo)

        await settle(until: { !manager.canUndo })
        #expect(!manager.canUndo)
    }
}
