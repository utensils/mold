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
                    Button("Use These Settings") { reuse(entry) }
                        .frame(maxWidth: .infinity)
                }
                HStack {
                    Button { actions.toggleFavorite(entries) } label: {
                        Label("Favourite", systemImage: allFavorite ? "star.fill" : "star")
                    }
                    Button { actions.quickLook(entries) } label: {
                        Label("Quick Look", systemImage: "eye")
                    }
                    ShareLink(items: entries.map(actions.draggable)) { print in
                        SharePreview(print.filename)
                    } label: {
                        Label("Share", systemImage: "square.and.arrow.up")
                    }
                    Button { actions.save(entries) } label: {
                        Label("Save", systemImage: "square.and.arrow.down")
                    }
                    Button { actions.copy(entries) } label: {
                        Label("Copy", systemImage: "doc.on.doc")
                    }
                    Button(role: .destructive) { actions.moveToTrash(entries) } label: {
                        Label("Trash", systemImage: "trash")
                    }
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
