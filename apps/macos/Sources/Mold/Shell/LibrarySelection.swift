import MoldClient
import SwiftUI

// What the Library menu is offered, and the focused values that carry it.
// Split from `LibraryCommands` for size: one file is the menu, this one is
// what the menu reads.

/// What the showing library can do to what is selected in it.
///
/// Equatable on the STATE only, never the closures: a closure is never equal
/// to itself, and the menu needs to redraw when the words on its items change,
/// not on every rebuild of the pane.
struct LibrarySelection: Equatable {
    let count: Int
    let allFavorite: Bool
    let scope: LibraryScope
    let shelves: [CollectionShelf]
    /// The shelf being looked at, when the scope is one. "Remove from…" is
    /// only ever honest about the shelf you are standing in.
    let enclosingShelf: CollectionShelf?
    /// Whether something in the window has a caret in it. A menu item with a
    /// bare key equivalent is offered the key first, so anything the Library
    /// binds unmodified has to yield to a field being typed into.
    let isEditingText: Bool
    /// What one selected print can be converted into, and how much is in the
    /// trash -- both of which the plan needs and the menu bar cannot see.
    let exportFormats: [String]
    /// A MESH's containers, from the holding host's advertised list. Nil for
    /// anything else, which keeps `exportFormats` the clip's answer.
    let meshExports: MeshExport.Split?
    let trashCount: Int
    let name: String?
    let canReuse: Bool
    /// Whether the machine holding the one selected print advertises
    /// upscaling it. The menu bar cannot see a capability block.
    let canUpscale: Bool

    /// Not compared: a `DraggablePrint` is a closure in a trench coat, and the
    /// count above already changes whenever this list does.
    let share: [DraggablePrint]

    /// The one door every item goes through -- the same one the tile's menu
    /// uses, so an item cannot mean two things.
    let perform: (LibraryAction) -> Void

    var isEmpty: Bool { count == 0 }

    /// Quick Look's item, which owns the bare space bar, is offered only when
    /// there is something to preview AND nothing is being typed into.
    var canQuickLook: Bool { !isEmpty && !isEditingText }

    /// What to offer, declared once and drawn by both menus.
    var plan: LibraryMenuPlan {
        LibraryMenuPlan(scope: scope.menuKind, count: count, allFavorite: allFavorite,
                        name: name, shelves: shelves, enclosingShelf: enclosingShelf,
                        exportFormats: exportFormats, meshExports: meshExports,
                        canReuse: canReuse, canUpscale: canUpscale, trashCount: trashCount)
    }

    static func == (lhs: Self, rhs: Self) -> Bool {
        lhs.count == rhs.count && lhs.allFavorite == rhs.allFavorite
            && lhs.scope == rhs.scope && lhs.shelves == rhs.shelves
            && lhs.enclosingShelf == rhs.enclosingShelf
            && lhs.isEditingText == rhs.isEditingText
            && lhs.exportFormats == rhs.exportFormats
            && lhs.meshExports == rhs.meshExports && lhs.trashCount == rhs.trashCount
            && lhs.name == rhs.name && lhs.canReuse == rhs.canReuse
            && lhs.canUpscale == rhs.canUpscale
    }
}

/// Which machines a file could be imported into, and how.
struct LibraryImport: Equatable {
    let machines: [MoldHost]
    let run: (MoldHost) -> Void

    static func == (lhs: Self, rhs: Self) -> Bool { lhs.machines == rhs.machines }
}

struct LibraryImportKey: FocusedValueKey {
    typealias Value = LibraryImport
}

struct LibrarySelectionKey: FocusedValueKey {
    typealias Value = LibrarySelection
}

extension FocusedValues {
    var librarySelection: LibrarySelection? {
        get { self[LibrarySelectionKey.self] }
        set { self[LibrarySelectionKey.self] = newValue }
    }

    var libraryImport: LibraryImport? {
        get { self[LibraryImportKey.self] }
        set { self[LibraryImportKey.self] = newValue }
    }
}
