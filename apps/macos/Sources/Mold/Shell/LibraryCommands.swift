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
        // A bare space as a key equivalent is offered to the main menu BEFORE
        // the field editor sees it, so an enabled item here takes the space
        // bar out of the search field, the inspector's Title and "Add a tag"
        // -- all three of which are only reachable with a selection, which is
        // exactly when this item is not disabled. It stands down while text is
        // being edited, the way the viewer's own key equivalents already do.
        Button("Quick Look") { library?.quickLook() }
            .keyboardShortcut(.space, modifiers: [])
            .disabled(!(library?.canQuickLook ?? false))
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
