import MoldClient
import SwiftUI

/// The Library menu.
///
/// Bulk actions belong in the menu bar, not in a floating bar that appears
/// over the grid when something is selected. The menu bar is where macOS says
/// to look for what can be done to a selection, it is searchable from Help, it
/// is reachable from the keyboard, and VoiceOver reads it -- none of which is
/// true of a bar that materialises over the content.
///
/// Every item here is the same call the contextual menu makes.
struct LibraryCommands: Commands {
    @FocusedValue(\.librarySelection) private var library
    @FocusedValue(\.libraryImport) private var importer

    var body: some Commands {
        // Importing is a File thing, not a Library thing: it is where the
        // Mac puts "bring something in from outside".
        CommandGroup(after: .newItem) {
            Menu("Import to") {
                ForEach(importer?.machines ?? []) { machine in
                    Button(machine.name) { importer?.run(machine) }
                }
            }
            .disabled(importer?.machines.isEmpty ?? true)
        }

        CommandMenu("Library") {
            if library?.scope.isTrash == true {
                trash
            } else {
                prints
            }
        }
    }

    @ViewBuilder private var prints: some View {
        Button("Quick Look") { library?.quickLook() }
            .keyboardShortcut(.space, modifiers: [])
            .disabled(library?.isEmpty ?? true)
        // Share moved to File ▸ Share, off the same `librarySelection.share`
        // (design decision 24, `MoldCommands.swift`) -- a Mac's Share belongs
        // in File, not in a feature menu.
        Divider()

        Button(library?.allFavorite == true ? "Unfavorite" : "Favorite") {
            library?.favorite(!(library?.allFavorite ?? false))
        }
        .keyboardShortcut("f", modifiers: [.command, .option])
        .disabled(library?.isEmpty ?? true)

        Menu("Move to Collection") {
            ForEach(library?.shelves ?? []) { shelf in
                Button(shelf.name) { library?.file(shelf) }
            }
        }
        .disabled(library?.shelves.isEmpty ?? true || library?.isEmpty ?? true)

        if let shelf = library?.enclosingShelf {
            Button("Remove from \(shelf.name)") { library?.unfile(shelf) }
                .disabled(library?.isEmpty ?? true)
        }

        Divider()

        Button("Move to Trash") { library?.trash() }
            .keyboardShortcut(.delete, modifiers: .command)
            .disabled(library?.isEmpty ?? true)
    }

    @ViewBuilder private var trash: some View {
        // The Finder's own words. "Restore" describes the mechanism; "Put
        // Back" describes what happens to your picture.
        Button("Put Back") { library?.putBack() }
            .disabled(library?.isEmpty ?? true)
        Button("Delete Immediately…") { library?.deleteForever() }
            .disabled(library?.isEmpty ?? true)
        Divider()
        Button("Empty Trash…") { library?.emptyTrash() }
    }
}

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

    /// Not compared: a `DraggablePrint` is a closure in a trench coat, and the
    /// count above already changes whenever this list does.
    let share: [DraggablePrint]

    let quickLook: () -> Void
    let favorite: (Bool) -> Void
    let file: (CollectionShelf) -> Void
    let unfile: (CollectionShelf) -> Void
    let trash: () -> Void
    let putBack: () -> Void
    let deleteForever: () -> Void
    let emptyTrash: () -> Void

    var isEmpty: Bool { count == 0 }

    static func == (lhs: Self, rhs: Self) -> Bool {
        lhs.count == rhs.count && lhs.allFavorite == rhs.allFavorite
            && lhs.scope == rhs.scope && lhs.shelves == rhs.shelves
            && lhs.enclosingShelf == rhs.enclosingShelf
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
