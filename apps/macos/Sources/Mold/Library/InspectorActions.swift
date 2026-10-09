import MoldClient
import SwiftUI

/// The buttons at the foot of the inspector.
struct InspectorActions: View {
    let entries: [LibraryEntry]
    let scope: LibraryScope
    let actions: LibraryActions

    var body: some View {
        if scope.isTrash {
            HStack {
                // The Finder's own words. "Restore" and "Delete" describe the
                // mechanism; these describe what happens to your picture.
                Button("Put Back") { actions.restore(entries) }
                Button("Delete Immediately…", role: .destructive) {
                    actions.deleteForever(entries)
                }
            }
        } else {
            VStack(spacing: 10) {
                if let reuse = actions.reuse, entries.count == 1, let entry = entries.first {
                    Button(LibraryMenuPlan.reuseTitle) { reuse(entry) }
                        .frame(maxWidth: .infinity)
                }
                HStack {
                    Button { actions.toggleFavorite(entries) } label: {
                        Label("Favourite", systemImage: allFavorite ? "star.fill" : "star")
                    }
                    .help(allFavorite ? "Remove these prints from Favourites" : "Add these prints to Favourites")
                    Button { actions.quickLook(entries) } label: {
                        Label("Quick Look", systemImage: "eye")
                    }
                    .help("Preview the selected prints without leaving the Library")
                    ShareLink(items: entries.map(actions.draggable)) { print in
                        SharePreview(print.filename)
                    } label: {
                        Label("Share", systemImage: "square.and.arrow.up")
                    }
                    .help("Share the selected prints with another app or person")
                    Button { actions.save(entries) } label: {
                        Label("Save", systemImage: "square.and.arrow.down")
                    }
                    .help("Save a copy of the selected prints to a folder you choose")
                    Button { actions.copy(entries) } label: {
                        Label("Copy", systemImage: "doc.on.doc")
                    }
                    .help("Copy the selected prints to the clipboard")
                    Button(role: .destructive) { actions.moveToTrash(entries) } label: {
                        Label("Trash", systemImage: "trash")
                    }
                    .help("Move the selected prints to Trash")
                }
                .labelStyle(.iconOnly)
                .buttonStyle(.bordered)
            }
        }
    }

    private var allFavorite: Bool { entries.allSatisfy(\.print.isFavorite) }
}

/// How long the machine will keep these, when they are in the trash.
struct TrashCountdownBlock: View {
    let entries: [LibraryEntry]

    var body: some View {
        // Each trashed print carries its OWN countdown, and the trash is never
        // collapsed or grouped -- hiding one behind another would let retention
        // purge something nobody was ever shown.
        let remaining = entries.compactMap { TrashRetention.remaining(for: $0.print) }
        if let first = remaining.first {
            Label(Set(remaining).count == 1 ? first : "Deleting on their own schedules",
                  systemImage: "clock")
                .font(.caption)
                .foregroundStyle(.secondary)
        }
    }
}
