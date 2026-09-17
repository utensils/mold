import MoldClient
import SwiftUI

// What the Library menu is offered. Split from the pane for size.
extension LibraryPane {

    /// The selection, as the menu bar needs to see it.
    ///
    /// WHAT is offered is `LibraryMenuPlan`'s -- the same list a tile's
    /// right-click menu draws -- and this supplies what the plan cannot see
    /// from a menu: the selection, the shelves, and the one door every item is
    /// performed through.
    func menuSelection(_ showing: LibraryShowing) -> LibrarySelection {
        let entries = showing.selected
        let actions = self.actions
        return LibrarySelection(
            count: entries.count,
            allFavorite: !entries.isEmpty && entries.allSatisfy(\.print.isFavorite),
            scope: navigation.scope,
            shelves: library.shelves,
            enclosingShelf: enclosingShelf,
            isEditingText: isEditingText,
            exportFormats: entries.count == 1 ? actions.exportFormats(for: entries[0]) : [],
            meshExports: entries.count == 1 && entries[0].print.isMesh
                ? actions.meshExports(for: entries[0]) : nil,
            trashCount: library.trashed.count,
            name: entries.count == 1 ? entries[0].print.displayName : nil,
            canReuse: entries.count == 1,
            share: navigation.scope.isTrash ? [] : entries.map(actions.draggable),
            perform: { action in
                actions.perform(action, on: entries, scope: navigation.scope,
                                open: entries.count == 1 ? { viewing = entries[0].id } : nil)
            }
        )
    }

    /// The machines a file could be imported into, and the door to do it.
    var menuImport: LibraryImport {
        let actions = self.actions
        return LibraryImport(machines: hosts.hosts.filter(hosts.isUp)) { machine in
            actions.importFiles(into: machine)
        }
    }

    /// What File ▸ Export… and File ▸ Save a Copy… offer -- the same calls
    /// `LibraryMenu.swift`'s own "Save a Copy…" and "Export…" make, off the
    /// same selection (design S6).
    func menuFile(_ showing: LibraryShowing) -> LibraryFile {
        let entries = showing.selected
        let actions = self.actions
        return LibraryFile(
            count: entries.count,
            exportFormats: entries.count == 1 ? actions.exportFormats(for: entries[0]) : [],
            save: { actions.save(entries) },
            export: { format in
                guard entries.count == 1, let entry = entries.first else { return }
                actions.export(entry, as: format)
            }
        )
    }

    /// Whether a caret in this window has the better claim on a bare key.
    /// The Library menu's Quick Look item binds an unmodified space, which
    /// AppKit offers to the menu before the field editor ever sees it.
    ///
    /// `TextEditingFocus` is the authority and answers for EVERY text field,
    /// including the two sheets that publish no `editingText` of their own.
    /// The two SwiftUI signals stay as a second opinion: they cost nothing and
    /// they are the ones that answer for a focus AppKit has not posted about
    /// yet.
    var isEditingText: Bool {
        TextEditingFocus.shared.isEditing || isSearchFocused || editingText == true
    }

    /// The shelf the grid is currently showing, if it is showing one.
    var enclosingShelf: CollectionShelf? {
        guard case let .collection(slug) = navigation.scope else { return nil }
        return library.shelf(slug: slug)
    }
}
