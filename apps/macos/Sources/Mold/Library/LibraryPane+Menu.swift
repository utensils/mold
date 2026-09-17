import MoldClient
import SwiftUI

// What the Library menu is offered. Split from the pane for size.
extension LibraryPane {

    /// The selection, as the menu bar needs to see it.
    ///
    /// Built from the same `actions` the contextual menu uses, so an item in
    /// the menu bar and the same item on a right-click are literally the same
    /// call -- there is no second path to keep in step.
    func menuSelection(_ showing: LibraryShowing) -> LibrarySelection {
        let entries = showing.selected
        return LibrarySelection(
            count: entries.count,
            allFavorite: !entries.isEmpty && entries.allSatisfy(\.print.isFavorite),
            scope: navigation.scope,
            shelves: library.shelves,
            enclosingShelf: enclosingShelf,
            share: navigation.scope.isTrash ? [] : entries.map(actions.draggable),
            quickLook: { actions.quickLook(entries) },
            favorite: { _ in actions.toggleFavorite(entries) },
            file: { shelf in library.file(entries, into: shelf) },
            unfile: { shelf in library.unfile(entries, from: shelf) },
            trash: { actions.moveToTrash(entries) },
            putBack: { actions.restore(entries) },
            deleteForever: { actions.deleteForever(entries) },
            emptyTrash: { actions.emptyTrash() }
        )
    }

    /// The machines a file could be imported into, and the door to do it.
    var menuImport: LibraryImport {
        let actions = self.actions
        return LibraryImport(machines: hosts.hosts.filter(hosts.isUp)) { machine in
            actions.importFiles(into: machine)
        }
    }

    /// The shelf the grid is currently showing, if it is showing one.
    private var enclosingShelf: CollectionShelf? {
        guard case let .collection(slug) = navigation.scope else { return nil }
        return library.shelf(slug: slug)
    }
}
