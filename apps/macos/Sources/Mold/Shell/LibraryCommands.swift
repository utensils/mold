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
/// Every item here is `LibraryMenuPlan`'s, which is also what a tile's
/// right-click menu draws: the same items, in the same order, with the same
/// words. The two used to be hand-written lists that disagreed about all
/// three. What is added here and not there is the KEYS -- a contextual menu
/// carries no shortcuts.
struct LibraryCommands: Commands {
    @FocusedValue(\.librarySelection) private var library

    var body: some Commands {
        CommandGroup(after: .newItem) { ImportCommands() }

        CommandMenu("Library") {
            if let library {
                RowActionMenu(actions: library.plan.items, perform: library.perform)
                    .modifier(LibraryShortcuts(selection: library))
            }
        }
    }
}

/// The keys the menu bar adds to the shared plan.
///
/// A modifier rather than items of its own: the plan decides WHAT is offered,
/// and this decides which of those rows a chord reaches. Applied to the whole
/// group because SwiftUI has no way to name one item from outside it -- so the
/// two that carry keys are bound here as their own copies, and the plan's
/// rows stay what the tile draws.
private struct LibraryShortcuts: ViewModifier {
    let selection: LibrarySelection

    func body(content: Content) -> some View {
        content
        Divider()
        // Quick Look's bare space stands down while a caret is in the window:
        // AppKit offers a key equivalent to the menu before the field editor
        // sees it. See `TextEditingFocus`.
        Button("Quick Look") { selection.perform(.quickLook) }
            .keyboardShortcut(.space, modifiers: [])
            .disabled(!selection.canQuickLook)
        Button(selection.allFavorite ? "Remove from Favourites" : "Add to Favourites") {
            selection.perform(.favorite(!selection.allFavorite))
        }
        .keyboardShortcut("f", modifiers: [.command, .option])
        .disabled(selection.isEmpty)
        Button("Move to Trash") { selection.perform(.trash) }
            .keyboardShortcut(.delete, modifiers: .command)
            .disabled(selection.isEmpty || selection.scope.isTrash)
    }
}

/// Importing is a File thing, not a Library thing: it is where the Mac puts
/// "bring something in from outside".
private struct ImportCommands: View {
    @FocusedValue(\.libraryImport) private var importer

    var body: some View {
        Menu("Import to") {
            ForEach(importer?.machines ?? []) { machine in
                Button(machine.name) { importer?.run(machine) }
            }
        }
        .disabled(importer?.machines.isEmpty ?? true)
    }
}
