import MoldClient
import MoldStyle
import SwiftUI

/// What the selection is made of, and what you can do with it.
struct LibraryInspector: View {
    let entries: [LibraryEntry]
    let host: MoldHost?
    let scope: LibraryScope
    let actions: LibraryActions
    let filterByTag: (String) -> Void

    var body: some View {
        Group {
            if entries.isEmpty {
                ContentUnavailableView("Nothing selected", systemImage: "sidebar.right")
            } else if entries.count == 1, let entry = entries.first {
                single(entry)
            } else {
                many
            }
        }
        .frame(maxWidth: .infinity, maxHeight: .infinity)
    }

    private func single(_ entry: LibraryEntry) -> some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 14) {
                if let host {
                    LibraryThumbnail(entry: entry, host: host, edge: 320)
                        .frame(maxWidth: .infinity)
                }
                if let prompt = entry.print.metadata.prompt, !prompt.isEmpty {
                    Text(prompt)
                        .font(.callout)
                        .textSelection(.enabled)
                        .fixedSize(horizontal: false, vertical: true)
                }
                facts(entry)
                if !scope.isTrash {
                    TagEditor(entries: entries, actions: actions, filterBy: filterByTag)
                }
                buttons
            }
            .padding(16)
        }
    }

    private var many: some View {
        VStack(spacing: 14) {
            Image(systemName: "square.stack")
                .font(.largeTitle)
                .foregroundStyle(.tertiary)
            Text("\(entries.count) prints selected").font(.headline)
            if let span = machines {
                Text(span).font(.caption).foregroundStyle(.secondary)
            }
            if scope.isTrash {
                countdown
            } else {
                TagEditor(entries: entries, actions: actions, filterBy: filterByTag)
            }
            buttons
            Spacer()
        }
        .padding(16)
    }

    /// Says when a selection spans machines, because the actions below will
    /// then touch more than one.
    private var machines: String? {
        let names = Set(entries.map(\.hostName)).sorted()
        return names.count > 1 ? "On \(names.joined(separator: ", "))" : names.first
    }

    /// Each trashed print carries its OWN countdown, and the trash is never
    /// collapsed or grouped -- hiding one behind another would let retention
    /// purge something nobody was ever shown.
    @ViewBuilder private var countdown: some View {
        let remaining = entries.compactMap { TrashRetention.remaining(for: $0.print) }
        if let first = remaining.first {
            Label(Set(remaining).count == 1 ? first : "Deleting on their own schedules",
                  systemImage: "clock")
                .font(.caption)
                .foregroundStyle(.secondary)
        }
    }

    @ViewBuilder private var buttons: some View {
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
            HStack {
                Button { actions.toggleFavorite(entries) } label: {
                    Label("Favorite", systemImage: allFavorite ? "star.fill" : "star")
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

    private var allFavorite: Bool { entries.allSatisfy(\.print.isFavorite) }

    private func facts(_ entry: LibraryEntry) -> some View {
        let meta = entry.print.metadata
        return Grid(alignment: .leadingFirstTextBaseline, horizontalSpacing: 12, verticalSpacing: 6) {
            row("Machine", entry.hostName)
            row("Model", meta.model)
            row("Seed", meta.seed.map(String.init))
            row("Steps", meta.steps.map(String.init))
            row("Guidance", meta.guidance.map { $0.formatted(.number.precision(.fractionLength(1))) })
            row("Size", size(meta))
            row("Made", entry.createdAt.formatted(date: .abbreviated, time: .shortened))
            row("File", entry.print.filename)
        }
        .font(.caption)
    }

    private func size(_ meta: OutputMetadata) -> String? {
        guard let width = meta.width, let height = meta.height else { return nil }
        guard let frames = meta.frames else { return "\(width) × \(height)" }
        return "\(width) × \(height) · \(frames) frames"
    }

    @ViewBuilder private func row(_ label: String, _ value: String?) -> some View {
        if let value {
            GridRow {
                Text(label).foregroundStyle(.secondary).gridColumnAlignment(.trailing)
                Text(value).textSelection(.enabled).monospacedDigit().lineLimit(3)
            }
        }
    }
}
