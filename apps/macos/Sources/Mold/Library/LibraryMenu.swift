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

        if let reuse = actions.reuse, targets.count == 1, let entry = targets.first,
           !scope.isTrash {
            Button("Use These Settings") { reuse(entry) }
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
            exportMenu
            Divider()
            Button("Move to Trash", role: .destructive) { actions.moveToTrash(targets) }
        }
    }

    /// Only offered for prints that have another form. A PNG has nothing to
    /// convert to that "Save a Copy" does not already give you.
    @ViewBuilder private var exportMenu: some View {
        if targets.count == 1, let entry = targets.first {
            let formats = actions.exportFormats(for: entry)
            if !formats.isEmpty {
                Menu("Export As") {
                    ForEach(formats, id: \.self) { format in
                        Button(format.uppercased()) { actions.export(entry, as: format) }
                    }
                }
            }
        }
    }

    private var favoriteTitle: String {
        targets.contains { !$0.print.isFavorite } ? "Add to Favorites" : "Remove from Favorites"
    }
}
