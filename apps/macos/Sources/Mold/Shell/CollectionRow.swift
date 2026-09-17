import MoldClient
import SwiftUI

/// One collection in the sidebar.
///
/// It is a drop target: dragging prints onto it files them, on each machine
/// that holds them, by NAME -- so a print on a machine that has never heard of
/// this shelf gets its own copy of it rather than landing nowhere.
struct CollectionRow: View {
    let shelf: CollectionShelf
    @Binding var renaming: CollectionShelf?

    @Environment(LibraryStore.self) private var library
    @State private var isTargeted = false
    @State private var pending: LibraryActions.Destruction?

    /// What opening this row would show, which is not the host's own `count`
    /// -- see `CollectionShelf.count(in:)`.
    private var shown: Int { shelf.count(in: library.items) }

    var body: some View {
        Label {
            HStack {
                Text(shelf.name).lineLimit(1)
                Spacer(minLength: 6)
                Text(shown.formatted())
                    .font(.caption)
                    .monospacedDigit()
                    .foregroundStyle(.secondary)
            }
        } icon: {
            Image(systemName: shelf.hidden ? "rectangle.stack.badge.minus" : "rectangle.stack")
        }
        .contextMenu {
            Button("Rename…") { renaming = shelf }
            Button(shelf.hidden ? "Show in All Prints" : "Hide from All Prints") {
                Task { await library.setShelfHidden(shelf, hidden: !shelf.hidden) }
            }
            Divider()
            Button("Delete Collection…", role: .destructive) { confirmDelete() }
        }
        .destructionDialog($pending)
        .dropDestination(for: PrintID.self) { ids, _ in
            file(ids)
            return true
        } isTargeted: { isTargeted = $0 }
        .listRowBackground(
            isTargeted
                ? RoundedRectangle(cornerRadius: 6).fill(.selection.opacity(0.35))
                : nil
        )
    }

    private func file(_ ids: [PrintID]) {
        let entries = library.items.filter { ids.contains($0.id) }
        guard !entries.isEmpty else { return }
        library.file(entries, into: shelf)
    }

    private func confirmDelete() {
        pending = LibraryActions.Destruction(
            title: "Delete “\(shelf.name)”?",
            // Worth saying plainly: people hesitate over this exact question.
            message: "The \(shown.formatted()) prints in it are kept. Only the collection goes.",
            verb: "Delete Collection"
        ) { Task { await library.deleteShelf(shelf) } }
    }
}

/// Naming a collection, new or existing.
struct ShelfNameSheet: View {
    let shelf: CollectionShelf?

    @Environment(HostStore.self) private var hosts
    @Environment(LibraryStore.self) private var library
    @Environment(\.dismiss) private var dismiss
    @State private var name = ""

    var body: some View {
        VStack(alignment: .leading, spacing: 14) {
            Text(shelf == nil ? "New Collection" : "Rename Collection").font(.headline)
            TextField("Name", text: $name, prompt: Text("Smurf Village"))
                .textFieldStyle(.roundedBorder)
                .onSubmit(commit)
            HStack {
                Spacer()
                Button("Cancel") { dismiss() }.keyboardShortcut(.cancelAction)
                Button(shelf == nil ? "Create" : "Rename", action: commit)
                    .keyboardShortcut(.defaultAction)
                    .disabled(trimmed.isEmpty)
            }
        }
        .padding(20)
        .frame(width: 360)
        .onAppear { name = shelf?.name ?? "" }
    }

    private var trimmed: String { name.trimmingCharacters(in: .whitespacesAndNewlines) }

    private func commit() {
        guard !trimmed.isEmpty else { return }
        let named = trimmed
        Task {
            if let shelf {
                await library.renameShelf(shelf, to: named)
            } else if let first = hosts.hosts.first {
                // Made on one machine; the others get their copy the first
                // time something of theirs is filed into it.
                await library.createShelf(named: named, on: first.id)
            }
        }
        dismiss()
    }
}
