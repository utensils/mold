import MoldClient
import SwiftUI

/// The right-click menu for a print.
///
/// Acts on the whole selection when the clicked print is part of it, and on
/// just that print when it isn't -- which is what every Mac app does and what
/// stops a stray right-click throwing away a careful selection.
struct LibraryMenu: View {
    let targets: [LibraryEntry]
    let scope: LibraryScope
    let actions: LibraryActions
    let open: (() -> Void)?

    var body: some View {
        if let open, targets.count == 1 {
            Button("Open", action: open)
            Divider()
        }

        if scope.isTrash {
            Button("Put Back") { actions.restore(targets) }
            Divider()
            Button("Delete Immediately…", role: .destructive) {
                actions.deleteForever(targets)
            }
        } else {
            Button(favoriteTitle) { actions.toggleFavorite(targets) }
            Divider()
            Button("Copy") { actions.copy(targets) }
            Button(targets.count == 1 ? "Save a Copy…" : "Save \(targets.count) Copies…") {
                actions.save(targets)
            }
            Divider()
            Button("Move to Trash", role: .destructive) { actions.moveToTrash(targets) }
        }
    }

    private var favoriteTitle: String {
        targets.contains { !$0.print.isFavorite } ? "Add to Favorites" : "Remove from Favorites"
    }
}
